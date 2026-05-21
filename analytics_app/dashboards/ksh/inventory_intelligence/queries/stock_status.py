"""
Stock status queries — feeds Page 1 (Command Centre) and Page 2 (Stockout Engine).
All functions accept a facility_filter string from utils.sql_in_filter().
"""

import pandas as pd
from utils.snowflake_conn import run_query


def get_stock_status(facility_filter: str) -> pd.DataFrame:
    """Current SOH per product per facility with stock status classification.

    Returns one row per (facility, canonical_product_id).
    Status: adequate | low | critical | zero | negative
    """
    return run_query(f"""
        WITH latest_soh AS (
            SELECT
                f.source_schema,
                f.product_id,
                f.canonical_product_id,
                f.soh_after_raw                                         AS current_soh,
                f.soh_after_display                                     AS current_soh_display,
                f.dispensed_at                                          AS last_dispensed_at,
                ROW_NUMBER() OVER (
                    PARTITION BY f.source_schema, f.product_id
                    ORDER BY f.dispensed_at DESC
                )                                                       AS rn
            FROM HOSPITALS.REPORTING.fact_dispensing f
            WHERE f.source_schema IN {facility_filter}
        ),
        avg_consumption AS (
            SELECT
                source_schema,
                product_id,
                SUM(quantity_dispensed) / NULLIF(
                    DATEDIFF('day',
                        MIN(dispensed_at),
                        MAX(dispensed_at)
                    ), 0
                )                                                       AS avg_daily_units,
                COUNT(DISTINCT dispensing_month)                        AS active_months
            FROM HOSPITALS.REPORTING.fact_dispensing
            WHERE source_schema IN {facility_filter}
              AND dispensed_at >= DATEADD('day', -90, CURRENT_DATE)
            GROUP BY source_schema, product_id
        )
        SELECT
            s.source_schema,
            s.product_id,
            s.canonical_product_id,
            t.canonical_name,
            t.therapeutic_class,
            t.therapeutic_subclass,
            t.product_category,
            t.inn_map_status,
            s.current_soh,
            s.current_soh_display,
            s.last_dispensed_at,
            DATEDIFF('day', s.last_dispensed_at, CURRENT_DATE)         AS days_since_last_dispense,
            c.avg_daily_units,
            CASE
                WHEN c.avg_daily_units > 0
                THEN ROUND(s.current_soh_display / c.avg_daily_units, 1)
                ELSE NULL
            END                                                         AS days_of_stock,
            CASE
                WHEN s.current_soh < 0             THEN 'negative'
                WHEN s.current_soh = 0             THEN 'zero'
                WHEN s.current_soh_display
                     / NULLIF(c.avg_daily_units, 0) < 7  THEN 'critical'
                WHEN s.current_soh_display
                     / NULLIF(c.avg_daily_units, 0) < 30 THEN 'low'
                ELSE                                    'adequate'
            END                                                         AS stock_status
        FROM latest_soh s
        LEFT JOIN HOSPITALS.REPORTING.CANONICAL_PRODUCT_TAXONOMY t
            ON UPPER(s.source_schema) = UPPER(t.facility)
            AND s.product_id = t.product_id
        LEFT JOIN avg_consumption c
            ON s.source_schema = c.source_schema
            AND s.product_id   = c.product_id
        WHERE s.rn = 1
          AND t.product_category = 'pharma'
        ORDER BY
            CASE s.stock_status
                WHEN 'negative' THEN 1
                WHEN 'zero'     THEN 2
                WHEN 'critical' THEN 3
                WHEN 'low'      THEN 4
                ELSE                 5
            END,
            s.current_soh_display ASC
    """)


def get_kpi_summary(facility_filter: str) -> pd.DataFrame:
    """Aggregate KPIs for the Command Centre top row.

    Returns one row with columns:
        total_products, active_stockouts, critical_count,
        low_count, total_dispensing_value_30d,
        chronic_patients_at_risk, opioid_patients_at_risk
    """
    return run_query(f"""
        WITH stock AS (
            SELECT
                f.source_schema,
                f.product_id,
                f.soh_after_raw,
                f.soh_after_display,
                f.dispensed_at,
                ROW_NUMBER() OVER (
                    PARTITION BY f.source_schema, f.product_id
                    ORDER BY f.dispensed_at DESC
                ) AS rn
            FROM HOSPITALS.REPORTING.fact_dispensing f
            WHERE f.source_schema IN {facility_filter}
        ),
        consumption_30d AS (
            SELECT
                source_schema,
                SUM(line_total)       AS total_dispensing_value_30d,
                SUM(quantity_dispensed) AS total_units_30d
            FROM HOSPITALS.REPORTING.fact_dispensing
            WHERE source_schema IN {facility_filter}
              AND dispensed_at >= DATEADD('day', -30, CURRENT_DATE)
        ),
        avg_c AS (
            SELECT
                source_schema,
                product_id,
                SUM(quantity_dispensed) / NULLIF(
                    DATEDIFF('day', MIN(dispensed_at), MAX(dispensed_at)), 0
                ) AS avg_daily_units
            FROM HOSPITALS.REPORTING.fact_dispensing
            WHERE source_schema IN {facility_filter}
              AND dispensed_at >= DATEADD('day', -90, CURRENT_DATE)
            GROUP BY source_schema, product_id
        ),
        status_agg AS (
            SELECT
                COUNT(DISTINCT s.product_id)                            AS total_products,
                COUNT_IF(s.soh_after_raw <= 0)                         AS active_stockouts,
                COUNT_IF(
                    s.soh_after_display / NULLIF(a.avg_daily_units, 0) < 7
                    AND s.soh_after_raw > 0
                )                                                       AS critical_count,
                COUNT_IF(
                    s.soh_after_display / NULLIF(a.avg_daily_units, 0) BETWEEN 7 AND 30
                )                                                       AS low_count
            FROM stock s
            LEFT JOIN avg_c a
                ON s.source_schema = a.source_schema
                AND s.product_id   = a.product_id
            WHERE s.rn = 1
        ),
        patient_risk AS (
            SELECT
                COUNT_IF(has_chronic_drug = 1
                    AND last_dispensed_at >= DATEADD('day', -60, CURRENT_DATE)
                )                                                       AS chronic_patients_active,
                COUNT_IF(has_opioid = 1
                    AND last_dispensed_at >= DATEADD('day', -60, CURRENT_DATE)
                )                                                       AS opioid_patients_active
            FROM HOSPITALS.REPORTING.fact_patient_dispensing
            WHERE source_schema IN {facility_filter}
        )
        SELECT
            sa.total_products,
            sa.active_stockouts,
            sa.critical_count,
            sa.low_count,
            c.total_dispensing_value_30d,
            c.total_units_30d,
            pr.chronic_patients_active,
            pr.opioid_patients_active
        FROM status_agg sa
        CROSS JOIN consumption_30d c
        CROSS JOIN patient_risk pr
    """)


def get_inventory_capital_breakdown(facility_filter: str) -> pd.DataFrame:
    """Capital breakdown by stock movement category.
    Uses dispensing value as a proxy for stock value (no unit cost in pipeline).

    Returns one row with columns:
        healthy_value, slow_moving_value, dead_value,
        stockout_value, total_value
    """
    return run_query(f"""
        WITH last_dispense AS (
            SELECT
                source_schema,
                product_id,
                MAX(dispensed_at)                               AS last_dispensed_at,
                DATEDIFF('day', MAX(dispensed_at), CURRENT_DATE) AS days_idle
            FROM HOSPITALS.REPORTING.fact_dispensing
            WHERE source_schema IN {facility_filter}
            GROUP BY source_schema, product_id
        ),
        product_value AS (
            SELECT
                f.source_schema,
                f.product_id,
                SUM(f.line_total)                               AS total_dispensing_value,
                ld.days_idle
            FROM HOSPITALS.REPORTING.fact_dispensing f
            JOIN last_dispense ld
                ON f.source_schema = ld.source_schema
                AND f.product_id   = ld.product_id
            WHERE f.source_schema IN {facility_filter}
            GROUP BY f.source_schema, f.product_id, ld.days_idle
        ),
        soh_status AS (
            SELECT
                f.source_schema,
                f.product_id,
                f.soh_after_raw,
                ROW_NUMBER() OVER (
                    PARTITION BY f.source_schema, f.product_id
                    ORDER BY f.dispensed_at DESC
                ) AS rn
            FROM HOSPITALS.REPORTING.fact_dispensing f
            WHERE f.source_schema IN {facility_filter}
        )
        SELECT
            SUM(CASE WHEN ss.soh_after_raw <= 0              THEN pv.total_dispensing_value ELSE 0 END) AS stockout_value,
            SUM(CASE WHEN pv.days_idle > 90
                          AND ss.soh_after_raw > 0           THEN pv.total_dispensing_value ELSE 0 END) AS dead_value,
            SUM(CASE WHEN pv.days_idle BETWEEN 30 AND 90
                          AND ss.soh_after_raw > 0           THEN pv.total_dispensing_value ELSE 0 END) AS slow_moving_value,
            SUM(CASE WHEN pv.days_idle < 30
                          AND ss.soh_after_raw > 0           THEN pv.total_dispensing_value ELSE 0 END) AS healthy_value,
            SUM(pv.total_dispensing_value)                                                              AS total_value
        FROM product_value pv
        JOIN soh_status ss
            ON pv.source_schema = ss.source_schema
            AND pv.product_id   = ss.product_id
            AND ss.rn = 1
    """)
