"""
Stockout alert queries — feeds Page 2 (Stockout Engine) and Page 6 (Compliance).
"""

import pandas as pd
from utils.snowflake_conn import run_query


# Products that are clinically critical — elevate alert priority
_CRITICAL_CANONICAL_NAMES = (
    "MORPHINE",
    "MISOPROSTOL",
    "LINEZOLID",
    "MEROPENEM",
    "VANCOMYCIN",
    "OXYTOCIN",
)

_CRITICAL_SUBCLASSES = (
    "Opioid Analgesics",
    "Carbapenems",
    "Glycopeptides",
    "Oxazolidinones",
    "Uterotonics",
)


def get_stockout_alerts(facility_filter: str) -> pd.DataFrame:
    """Current and recent stockout alerts with clinical priority flag.

    A stockout episode = consecutive dispense events where soh_after_raw <= 0
    grouped by product.

    Returns columns:
        source_schema, canonical_name, therapeutic_class, therapeutic_subclass,
        current_soh, first_stockout_at, last_stockout_at, stockout_dispense_count,
        units_dispensed_at_zero, clinical_priority (CRITICAL | HIGH | STANDARD)
    """
    critical_names    = ", ".join(f"'{n}'" for n in _CRITICAL_CANONICAL_NAMES)
    critical_subclass = ", ".join(f"'{s}'" for s in _CRITICAL_SUBCLASSES)

    return run_query(f"""
        WITH stockout_events AS (
            SELECT
                f.source_schema,
                f.product_id,
                f.canonical_product_id,
                f.quantity_dispensed,
                f.line_total,
                f.soh_after_raw,
                f.is_stockout_dispense,
                f.dispensed_from_negative_stock,
                f.dispensed_at
            FROM HOSPITALS.REPORTING.fact_dispensing f
            WHERE f.source_schema IN {facility_filter}
              AND (f.is_stockout_dispense = TRUE
                   OR f.dispensed_from_negative_stock = TRUE)
        ),
        aggregated AS (
            SELECT
                se.source_schema,
                se.product_id,
                se.canonical_product_id,
                COUNT(*)                            AS stockout_dispense_count,
                SUM(se.quantity_dispensed)          AS units_dispensed_at_zero,
                SUM(se.line_total)                  AS value_dispensed_at_zero,
                MIN(se.dispensed_at)                AS first_stockout_at,
                MAX(se.dispensed_at)                AS last_stockout_at,
                DATEDIFF('day',
                    MIN(se.dispensed_at),
                    MAX(se.dispensed_at)
                )                                   AS stockout_span_days
            FROM stockout_events se
            GROUP BY se.source_schema, se.product_id, se.canonical_product_id
        ),
        current_soh AS (
            SELECT
                source_schema,
                product_id,
                soh_after_raw                       AS current_soh,
                ROW_NUMBER() OVER (
                    PARTITION BY source_schema, product_id
                    ORDER BY dispensed_at DESC
                )                                   AS rn
            FROM HOSPITALS.REPORTING.fact_dispensing
            WHERE source_schema IN {facility_filter}
        )
        SELECT
            a.source_schema,
            t.canonical_name,
            t.therapeutic_class,
            t.therapeutic_subclass,
            cs.current_soh,
            a.stockout_dispense_count,
            a.units_dispensed_at_zero,
            a.value_dispensed_at_zero,
            a.first_stockout_at,
            a.last_stockout_at,
            a.stockout_span_days,
            DATEDIFF('day', a.last_stockout_at, CURRENT_DATE) AS days_since_last_stockout,
            CASE
                WHEN UPPER(t.canonical_name) LIKE ANY ({critical_names})
                  OR t.therapeutic_subclass IN ({critical_subclass})
                THEN 'CRITICAL'
                WHEN t.therapeutic_class IN ('Cardiovascular', 'Endocrine & Metabolic',
                                              'Antimicrobials', 'Neurological')
                THEN 'HIGH'
                ELSE 'STANDARD'
            END                                               AS clinical_priority
        FROM aggregated a
        JOIN current_soh cs
            ON a.source_schema = cs.source_schema
            AND a.product_id   = cs.product_id
            AND cs.rn = 1
        LEFT JOIN HOSPITALS.REPORTING.CANONICAL_PRODUCT_TAXONOMY t
            ON UPPER(a.source_schema) = UPPER(t.facility)
            AND a.product_id          = t.product_id
        WHERE t.product_category = 'pharma'
        ORDER BY
            CASE clinical_priority
                WHEN 'CRITICAL' THEN 1
                WHEN 'HIGH'     THEN 2
                ELSE                 3
            END,
            a.stockout_dispense_count DESC
    """)


def get_deficit_dispenses(facility_filter: str) -> pd.DataFrame:
    """Products dispensed from negative stock — PPB compliance flag.
    Used on Page 6 (Compliance Tracker).

    Returns columns:
        source_schema, canonical_name, therapeutic_subclass,
        dispensed_by_user_id, dispensed_at,
        soh_before, quantity_dispensed, soh_after_raw
    """
    return run_query(f"""
        SELECT
            f.source_schema,
            t.canonical_name,
            t.therapeutic_class,
            t.therapeutic_subclass,
            h.dispensed_by_user_id,
            f.dispensed_at,
            f.soh_before,
            f.quantity_dispensed,
            f.soh_after_raw,
            f.line_total
        FROM HOSPITALS.REPORTING.fact_dispensing f
        JOIN HOSPITALS.STAGING.stg_dispensing_header h
            ON f.source_schema     = h.source_schema
            AND f.raw_dispensing_id = h.raw_dispensing_id
        LEFT JOIN HOSPITALS.REPORTING.CANONICAL_PRODUCT_TAXONOMY t
            ON UPPER(f.source_schema) = UPPER(t.facility)
            AND f.product_id          = t.product_id
        WHERE f.source_schema IN {facility_filter}
          AND f.dispensed_from_negative_stock = TRUE
          AND t.product_category = 'pharma'
        ORDER BY f.dispensed_at DESC
    """)
