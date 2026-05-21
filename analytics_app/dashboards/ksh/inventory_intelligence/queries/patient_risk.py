"""
Patient risk exposure queries.
Surfaces patients on chronic or opioid medication whose supply is at risk.
"""

from __future__ import annotations

import pandas as pd

from utils.snowflake_conn import run_query


def get_patient_risk_exposure(schema: str, ref_date: str = "CURRENT_DATE") -> pd.DataFrame:
    """
    Products at risk (stockout or <30d DOS) cross-joined with active patient exposure.
    Returns one row per at-risk drug with patient counts.

    Requires HOSPITALS.REPORTING.FACT_PATIENT_DISPENSING to have columns:
        source_schema, patient_id, canonical_product_id,
        has_chronic_drug, has_opioid, last_dispensed_at
    """
    return run_query(f"""
        WITH at_risk AS (
            -- Latest SOH snapshot per product
            SELECT
                f.product_id,
                f.soh_after_raw     AS current_soh,
                f.soh_after_display AS current_soh_display,
                ROW_NUMBER() OVER (PARTITION BY f.product_id ORDER BY f.dispensed_at DESC) AS rn
            FROM HOSPITALS.REPORTING.FACT_DISPENSING f
            WHERE f.source_schema = '{schema}'
        ),
        consumption AS (
            SELECT
                product_id,
                SUM(quantity_dispensed) /
                    NULLIF(COUNT(DISTINCT DATE_TRUNC('day', dispensed_at)), 0) AS avg_daily_units
            FROM HOSPITALS.REPORTING.FACT_DISPENSING
            WHERE source_schema = '{schema}'
              AND dispensed_at >= DATEADD('day', -90, {ref_date})
            GROUP BY product_id
        ),
        at_risk_products AS (
            SELECT
                ar.product_id,
                ar.current_soh,
                ar.current_soh_display,
                c.avg_daily_units,
                ROUND(ar.current_soh_display / NULLIF(c.avg_daily_units, 0), 1) AS days_of_stock,
                CASE
                    WHEN ar.current_soh <= 0 THEN 'stockout'
                    WHEN ar.current_soh_display / NULLIF(c.avg_daily_units, 0) < 7  THEN 'critical'
                    WHEN ar.current_soh_display / NULLIF(c.avg_daily_units, 0) < 30 THEN 'low'
                END AS risk_tier
            FROM at_risk ar
            LEFT JOIN consumption c ON ar.product_id = c.product_id
            WHERE ar.rn = 1
              AND (ar.current_soh <= 0
                   OR ar.current_soh_display / NULLIF(c.avg_daily_units, 0) < 30)
        ),
        -- Patients who received each at-risk product within the last 90 days
        recent_product_patients AS (
            SELECT DISTINCT
                f.product_id,
                f.patient_id
            FROM HOSPITALS.REPORTING.FACT_DISPENSING f
            JOIN at_risk_products arp ON f.product_id = arp.product_id
            WHERE f.source_schema = '{schema}'
              AND f.dispensed_at >= DATEADD('day', -90, {ref_date})
              AND f.patient_id IS NOT NULL
        )
        SELECT
            t.canonical_name,
            t.therapeutic_class,
            t.therapeutic_subclass,
            arp.current_soh,
            arp.days_of_stock,
            arp.risk_tier,
            COUNT(DISTINCT rpp.patient_id)             AS total_patients_at_risk,
            COUNT_IF(pd.has_chronic_drug = 1)          AS chronic_patients,
            COUNT_IF(pd.has_opioid = 1)                AS opioid_patients
        FROM at_risk_products arp
        LEFT JOIN HOSPITALS.REPORTING.CANONICAL_PRODUCT_TAXONOMY t
            ON UPPER('{schema}') = UPPER(t.facility)
            AND arp.product_id = t.product_id
        LEFT JOIN recent_product_patients rpp
            ON arp.product_id = rpp.product_id
        LEFT JOIN HOSPITALS.REPORTING.FACT_PATIENT_DISPENSING pd
            ON rpp.patient_id = pd.patient_id
            AND pd.source_schema = '{schema}'
        WHERE t.product_category = 'pharma'
        GROUP BY
            t.canonical_name, t.therapeutic_class, t.therapeutic_subclass,
            arp.current_soh, arp.days_of_stock, arp.risk_tier
        HAVING total_patients_at_risk > 0
        ORDER BY
            CASE arp.risk_tier WHEN 'stockout' THEN 1 WHEN 'critical' THEN 2 ELSE 3 END,
            chronic_patients DESC, opioid_patients DESC
    """)


def get_patient_risk_totals(schema: str, ref_date: str = "CURRENT_DATE") -> pd.DataFrame:
    """
    Deduplicated patient counts across ALL at-risk products.
    Avoids double-counting patients who are on multiple at-risk drugs.
    Used for the KPI strip on the Patient Risk page.
    """
    return run_query(f"""
        WITH at_risk AS (
            SELECT
                f.product_id,
                f.soh_after_raw      AS current_soh,
                f.soh_after_display  AS current_soh_display,
                ROW_NUMBER() OVER (PARTITION BY f.product_id ORDER BY f.dispensed_at DESC) AS rn
            FROM HOSPITALS.REPORTING.FACT_DISPENSING f
            WHERE f.source_schema = '{schema}'
        ),
        consumption AS (
            SELECT
                product_id,
                SUM(quantity_dispensed) /
                    NULLIF(COUNT(DISTINCT DATE_TRUNC('day', dispensed_at)), 0) AS avg_daily_units
            FROM HOSPITALS.REPORTING.FACT_DISPENSING
            WHERE source_schema = '{schema}'
              AND dispensed_at >= DATEADD('day', -90, {ref_date})
            GROUP BY product_id
        ),
        at_risk_products AS (
            SELECT ar.product_id
            FROM at_risk ar
            LEFT JOIN consumption c ON ar.product_id = c.product_id
            WHERE ar.rn = 1
              AND (ar.current_soh <= 0
                   OR ar.current_soh_display / NULLIF(c.avg_daily_units, 0) < 30)
        ),
        exposed_patients AS (
            SELECT DISTINCT f.patient_id
            FROM HOSPITALS.REPORTING.FACT_DISPENSING f
            JOIN at_risk_products arp ON f.product_id = arp.product_id
            WHERE f.source_schema = '{schema}'
              AND f.dispensed_at >= DATEADD('day', -90, {ref_date})
              AND f.patient_id IS NOT NULL
        )
        SELECT
            COUNT(DISTINCT ep.patient_id) AS total_patients_at_risk,
            COUNT(DISTINCT CASE WHEN pd.has_chronic_drug = 1 THEN ep.patient_id END) AS chronic_patients_at_risk,
            COUNT(DISTINCT CASE WHEN pd.has_opioid = 1      THEN ep.patient_id END) AS opioid_patients_at_risk,
            (SELECT COUNT(DISTINCT patient_id)
             FROM HOSPITALS.REPORTING.FACT_PATIENT_DISPENSING
             WHERE source_schema = '{schema}') AS total_active_patients
        FROM exposed_patients ep
        LEFT JOIN HOSPITALS.REPORTING.FACT_PATIENT_DISPENSING pd
            ON ep.patient_id = pd.patient_id
            AND pd.source_schema = '{schema}'
    """)


def get_high_risk_patient_summary(schema: str, ref_date: str = "CURRENT_DATE") -> pd.DataFrame:
    """Legacy — kept for compatibility. Use get_patient_risk_totals() for new code."""
    return get_patient_risk_totals(schema, ref_date=ref_date)
