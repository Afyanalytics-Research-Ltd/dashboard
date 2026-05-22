"""
Dispensing trend queries — feeds Page 3 (Dead Stock) and Page 4 (Clinical Intelligence).
"""

import pandas as pd
from utils.snowflake_conn import run_query


def get_dispensing_trends(facility_filter: str) -> pd.DataFrame:
    """Monthly consumption per canonical drug per facility.

    Returns columns:
        source_schema, canonical_product_id, canonical_name, therapeutic_class,
        dispensing_month, total_units_dispensed, total_dispensing_value,
        unique_patients, dispensing_events, stockout_dispenses,
        mom_change_pct (month-over-month % change)
    """
    return run_query(f"""
        WITH monthly AS (
            SELECT
                f.source_schema,
                f.canonical_product_id,
                DATE_TRUNC('month', f.dispensed_at)                 AS dispensing_month,
                SUM(f.quantity_dispensed)                           AS total_units_dispensed,
                SUM(f.line_total)                                   AS total_dispensing_value,
                COUNT(DISTINCT f.patient_id)                        AS unique_patients,
                COUNT(DISTINCT f.raw_dispensing_id)                 AS dispensing_events,
                COUNT_IF(f.is_stockout_dispense = TRUE)             AS stockout_dispenses
            FROM HOSPITALS.REPORTING.fact_dispensing f
            WHERE f.source_schema IN {facility_filter}
              AND f.canonical_product_id IS NOT NULL
            GROUP BY f.source_schema, f.canonical_product_id, DATE_TRUNC('month', f.dispensed_at)
        ),
        with_mom AS (
            SELECT
                m.*,
                LAG(m.total_units_dispensed) OVER (
                    PARTITION BY m.source_schema, m.canonical_product_id
                    ORDER BY m.dispensing_month
                )                                                   AS prev_month_units,
                CASE
                    WHEN LAG(m.total_units_dispensed) OVER (
                        PARTITION BY m.source_schema, m.canonical_product_id
                        ORDER BY m.dispensing_month
                    ) > 0
                    THEN ROUND(
                        (m.total_units_dispensed - LAG(m.total_units_dispensed) OVER (
                            PARTITION BY m.source_schema, m.canonical_product_id
                            ORDER BY m.dispensing_month
                        )) * 100.0 /
                        LAG(m.total_units_dispensed) OVER (
                            PARTITION BY m.source_schema, m.canonical_product_id
                            ORDER BY m.dispensing_month
                        ), 1)
                    ELSE NULL
                END                                                 AS mom_change_pct
            FROM monthly m
        )
        SELECT
            wm.source_schema,
            wm.canonical_product_id,
            t.canonical_name,
            t.therapeutic_class,
            t.therapeutic_subclass,
            wm.dispensing_month,
            wm.total_units_dispensed,
            wm.total_dispensing_value,
            wm.unique_patients,
            wm.dispensing_events,
            wm.stockout_dispenses,
            wm.mom_change_pct
        FROM with_mom wm
        LEFT JOIN HOSPITALS.REPORTING.CANONICAL_KEYS t
            ON wm.canonical_product_id = t.canonical_product_id
        ORDER BY wm.source_schema, t.canonical_name, wm.dispensing_month
    """)


def get_dead_stock_candidates(facility_filter: str, idle_threshold_days: int = 30) -> pd.DataFrame:
    """Products with no dispense activity beyond the idle threshold.
    Scored by days idle and total historical dispensing value (as KES proxy).

    Returns columns:
        source_schema, canonical_name, therapeutic_class,
        last_dispensed_at, days_idle, total_historical_value,
        total_historical_units, idle_category (slow | dead)
    """
    return run_query(f"""
        WITH product_activity AS (
            SELECT
                f.source_schema,
                f.product_id,
                f.canonical_product_id,
                MAX(f.dispensed_at)                                 AS last_dispensed_at,
                DATEDIFF('day', MAX(f.dispensed_at), CURRENT_DATE)  AS days_idle,
                SUM(f.line_total)                                   AS total_historical_value,
                SUM(f.quantity_dispensed)                           AS total_historical_units,
                COUNT(DISTINCT f.raw_dispensing_id)                 AS total_dispense_events
            FROM HOSPITALS.REPORTING.fact_dispensing f
            WHERE f.source_schema IN {facility_filter}
            GROUP BY f.source_schema, f.product_id, f.canonical_product_id
        ),
        current_soh AS (
            SELECT
                source_schema,
                product_id,
                soh_after_display                                   AS current_soh,
                ROW_NUMBER() OVER (
                    PARTITION BY source_schema, product_id
                    ORDER BY dispensed_at DESC
                )                                                   AS rn
            FROM HOSPITALS.REPORTING.fact_dispensing
            WHERE source_schema IN {facility_filter}
        )
        SELECT
            pa.source_schema,
            t.canonical_name,
            t.therapeutic_class,
            t.therapeutic_subclass,
            pa.last_dispensed_at,
            pa.days_idle,
            pa.total_historical_value,
            pa.total_historical_units,
            pa.total_dispense_events,
            cs.current_soh,
            CASE
                WHEN pa.days_idle >= 90 THEN 'dead'
                WHEN pa.days_idle >= {idle_threshold_days} THEN 'slow'
                ELSE 'active'
            END                                                     AS idle_category
        FROM product_activity pa
        JOIN current_soh cs
            ON pa.source_schema = cs.source_schema
            AND pa.product_id   = cs.product_id
            AND cs.rn = 1
        LEFT JOIN HOSPITALS.REPORTING.CANONICAL_PRODUCT_TAXONOMY t
            ON UPPER(pa.source_schema) = UPPER(t.facility)
            AND pa.product_id          = t.product_id
        WHERE pa.days_idle >= {idle_threshold_days}
          AND cs.current_soh > 0
          AND t.product_category = 'pharma'
        ORDER BY pa.days_idle DESC, pa.total_historical_value DESC
    """)
