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
            COALESCE(t.canonical_name, arp.product_id::VARCHAR) AS canonical_name,
            t.therapeutic_class,
            t.therapeutic_subclass,
            arp.current_soh,
            arp.days_of_stock,
            arp.risk_tier,
            COUNT(DISTINCT rpp.patient_id)                                          AS total_patients_at_risk,
            COUNT(DISTINCT CASE WHEN pd.has_chronic_drug = 1 THEN rpp.patient_id END) AS chronic_patients,
            COUNT(DISTINCT CASE WHEN pd.has_opioid = 1       THEN rpp.patient_id END) AS opioid_patients
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
            COALESCE(t.canonical_name, arp.product_id::VARCHAR), t.therapeutic_class, t.therapeutic_subclass,
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
             WHERE source_schema = '{schema}'
               AND last_dispensed_at >= DATEADD('day', -90, {ref_date})) AS total_active_patients
        FROM exposed_patients ep
        LEFT JOIN HOSPITALS.REPORTING.FACT_PATIENT_DISPENSING pd
            ON ep.patient_id = pd.patient_id
            AND pd.source_schema = '{schema}'
    """)


def get_high_risk_patient_summary(schema: str, ref_date: str = "CURRENT_DATE") -> pd.DataFrame:
    """Legacy — kept for compatibility. Use get_patient_risk_totals() for new code."""
    return get_patient_risk_totals(schema, ref_date=ref_date)


def get_patient_refill_overdue(schema: str, ref_date: str = "CURRENT_DATE") -> pd.DataFrame:
    """
    Phase 2 Rule 5: Patients whose refill is overdue AND the drug has < 7 days cover.

    Quantity-based supply estimation (visit-interval method):
      avg_qty_per_visit  = total_qty / n_distinct_visit_days
      mean_interval_days = span_days / (n_visits - 1)
      estimated_supply   = (last_qty / avg_qty_per_visit) × mean_interval_days  (capped 180d)
      overdue when       = days_since_last_dispense > estimated_supply × 1.2

    Why visit-interval instead of total_qty/span_days:
      total_qty/span_days explodes for inpatient stays (3 daily doses in 2 days →
      avg_daily = 45 → supply = 0.7d for a 30-tablet script). The visit-interval
      formula is robust: it asks "given the patient's typical prescription size and
      typical return frequency, how long should this last fill carry them?"

    mean_interval_days >= 7 guard screens out inpatient dispensing patterns (daily
    hospital administration), which would otherwise produce nonsensically short
    estimated supplies. mean_interval_days capped at 90 to avoid flagging patients
    who only ever had one meaningful gap.

    Minimum last_qty ≥ 5 units filters out single-dose hospital administrations and
    test/reconciliation entries that are not meaningful take-home prescriptions.
    Injectable formulations excluded by product name (' INJECTION' suffix) pending
    a route_of_administration field in the taxonomy.

    Restricted to therapeutic classes where repeat outpatient dispensing is clinically
    meaningful. Hospital-administered subclasses (Carbapenems, Glycopeptides,
    Oxazolidinones, Uterotonics) excluded. Unmapped drugs (NULL class/subclass) excluded.

    Minimum 2 visits required; mean_interval >= 7d required (screens inpatient patterns).

    Requires FACT_DISPENSING to have: patient_id, product_id, dispensed_at,
                                      soh_after_raw, soh_after_display, quantity_dispensed
    Returns columns: canonical_name, days_of_cover, overdue_patient_count,
                     avg_days_overdue, avg_estimated_supply_days, therapeutic_class,
                     therapeutic_subclass
    """
    # Therapeutic classes where repeat dispensing IS meaningful
    _refill_classes = (
        "Cardiovascular",
        "Endocrine & Metabolic",
        "Neurological",
        "Antiepileptics",
        "Oncology",
        "Antiretrovirals",
        "Psychiatric",
        "Respiratory",
        "Musculoskeletal",
        "Gastrointestinal",
    )
    # Subclasses that are always outpatient-refillable even if the class is broad
    _refill_subclasses = (
        "Opioid Analgesics",
    )
    # Subclasses to exclude regardless of class (hospital-administered)
    _exclude_subclasses = (
        "Carbapenems",
        "Glycopeptides",
        "Oxazolidinones",
        "Uterotonics",
    )

    # Known data-entry errors: quantity recorded in drug name field.
    # See DATA_QUALITY.md for details. Exclude until fixed at source.
    _bad_canonical_names = (
        "AMLODIPINE 80MG TABLET",   # "80MG" is a dispensed qty, not a dose strength
    )
    _bad_names_excl = ", ".join(f"'{n}'" for n in _bad_canonical_names)

    _cls_in   = ", ".join(f"'{c}'" for c in _refill_classes)
    _sub_in   = ", ".join(f"'{s}'" for s in _refill_subclasses)
    _sub_excl = ", ".join(f"'{s}'" for s in _exclude_subclasses)

    return run_query(f"""
        WITH eligible_products AS (
            -- Only drugs where repeat outpatient dispensing makes clinical sense.
            -- NULLs in either taxonomy column → excluded (can't confirm drug is a
            -- refill drug; safer to omit than to incorrectly flag hospital-administered
            -- drugs like carbapenems whose subclass may simply be unmapped).
            SELECT DISTINCT t.product_id
            FROM HOSPITALS.REPORTING.CANONICAL_PRODUCT_TAXONOMY t
            WHERE UPPER(t.facility) = UPPER('{schema}')
              AND t.product_category = 'pharma'
              AND t.therapeutic_subclass IS NOT NULL
              AND t.therapeutic_class    IS NOT NULL
              AND t.therapeutic_subclass NOT IN ({_sub_excl})
              AND (
                  t.therapeutic_class    IN ({_cls_in})
                  OR t.therapeutic_subclass IN ({_sub_in})
              )
              -- Exclude injectable formulations (hospital-administered, not take-home).
              -- TODO: replace with route_of_administration filter when taxonomy is updated.
              AND UPPER(t.canonical_name) NOT LIKE '% INJECTION'
              -- Exclude known data-entry errors (see DATA_QUALITY.md).
              AND t.canonical_name NOT IN ({_bad_names_excl})
        ),
        soh_latest AS (
            SELECT
                product_id,
                soh_after_raw      AS current_soh,
                soh_after_display  AS current_soh_display,
                ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY dispensed_at DESC) AS rn
            FROM HOSPITALS.REPORTING.FACT_DISPENSING
            WHERE source_schema = '{schema}'
        ),
        consumption AS (
            SELECT
                product_id,
                SUM(quantity_dispensed) /
                    NULLIF(COUNT(DISTINCT DATE_TRUNC('day', dispensed_at)), 0) AS avg_daily
            FROM HOSPITALS.REPORTING.FACT_DISPENSING
            WHERE source_schema = '{schema}'
              AND dispensed_at >= DATEADD('day', -90, {ref_date})
            GROUP BY product_id
        ),
        at_risk_products AS (
            -- Eligible products with < 7 days of cover
            SELECT
                s.product_id,
                ROUND(s.current_soh_display / NULLIF(c.avg_daily, 0), 1) AS days_of_cover
            FROM soh_latest s
            INNER JOIN eligible_products ep ON s.product_id = ep.product_id
            LEFT JOIN consumption c ON s.product_id = c.product_id
            WHERE s.rn = 1
              AND s.current_soh_display / NULLIF(c.avg_daily, 0) < 7
        ),
        patient_last_qty AS (
            -- Most recent quantity dispensed per patient-product.
            -- 180-day lookback: limits list to patients still active at this facility.
            -- Patients last seen > 6 months ago are unlikely to be actionable
            -- (self-discharged, transferred, or getting care elsewhere).
            -- patient_stats uses the deeper 365-day window to compute reliable visit patterns.
            SELECT
                patient_id,
                product_id,
                quantity_dispensed AS last_qty_dispensed,
                dispensed_at       AS last_dispensed
            FROM HOSPITALS.REPORTING.FACT_DISPENSING
            WHERE source_schema = '{schema}'
              AND dispensed_at >= DATEADD('day', -180, {ref_date})
              AND patient_id IS NOT NULL
              AND quantity_dispensed > 0
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY patient_id, product_id ORDER BY dispensed_at DESC
            ) = 1
        ),
        patient_stats AS (
            -- Per-patient visit statistics used for supply estimation.
            -- Uses 365-day history (deeper than patient_last_qty) so the visit pattern
            -- is computed from enough data even if the patient's most recent fill was 5 months ago.
            -- avg_qty_per_visit  = total_qty  / n_distinct_visit_days
            -- mean_interval_days = span_days  / (n_visits - 1)
            -- mean_interval >= 7d rejects inpatient daily-dose patterns.
            -- mean_interval <= 90d caps chronic outliers with very long gaps.
            SELECT
                f.patient_id,
                f.product_id,
                COUNT(DISTINCT DATE_TRUNC('day', f.dispensed_at))            AS n_visits,
                SUM(f.quantity_dispensed)                                     AS total_qty,
                DATEDIFF('day', MIN(f.dispensed_at), MAX(f.dispensed_at)) /
                    NULLIF(COUNT(DISTINCT DATE_TRUNC('day', f.dispensed_at)) - 1, 0)
                    AS mean_interval_days,
                SUM(f.quantity_dispensed) /
                    NULLIF(COUNT(DISTINCT DATE_TRUNC('day', f.dispensed_at)), 0)
                    AS avg_qty_per_visit
            FROM HOSPITALS.REPORTING.FACT_DISPENSING f
            INNER JOIN at_risk_products arp ON f.product_id = arp.product_id
            WHERE f.source_schema = '{schema}'
              AND f.dispensed_at >= DATEADD('day', -365, {ref_date})
              AND f.patient_id IS NOT NULL
              AND f.quantity_dispensed > 0
            GROUP BY f.patient_id, f.product_id
            HAVING COUNT(DISTINCT DATE_TRUNC('day', f.dispensed_at)) >= 2
               AND DATEDIFF('day', MIN(f.dispensed_at), MAX(f.dispensed_at)) /
                       NULLIF(COUNT(DISTINCT DATE_TRUNC('day', f.dispensed_at)) - 1, 0) >= 7
               AND DATEDIFF('day', MIN(f.dispensed_at), MAX(f.dispensed_at)) /
                       NULLIF(COUNT(DISTINCT DATE_TRUNC('day', f.dispensed_at)) - 1, 0) <= 90
        ),
        overdue AS (
            -- estimated_supply = LEAST(last_qty/avg_qty, 2.0) × mean_interval_days, capped 180d.
            -- The 2.0× ratio cap prevents over-inflation when a patient's last fill was
            -- unusually large (e.g. got 30 tablets when they typically get 10 — without the
            -- cap the formula would triple the estimated supply to 90 days for a daily drug).
            -- last_qty >= 5 excludes single-dose inpatient administrations and test entries.
            -- Overdue when days_since_last > estimated_supply × 1.2.
            SELECT
                ps.patient_id,
                ps.product_id,
                plq.last_dispensed,
                plq.last_qty_dispensed,
                LEAST(
                    ROUND(
                        LEAST(plq.last_qty_dispensed / NULLIF(ps.avg_qty_per_visit, 0), 2.0)
                        * ps.mean_interval_days,
                    0),
                    180
                )                                                            AS estimated_supply_days,
                ROUND(
                    DATEDIFF('day', plq.last_dispensed, {ref_date})
                    - LEAST(
                        LEAST(plq.last_qty_dispensed / NULLIF(ps.avg_qty_per_visit, 0), 2.0)
                        * ps.mean_interval_days,
                        180
                      ) * 1.2,
                0)                                                           AS days_overdue
            FROM patient_stats ps
            JOIN patient_last_qty plq
                ON ps.patient_id = plq.patient_id AND ps.product_id = plq.product_id
            WHERE plq.last_qty_dispensed >= 5
              AND DATEDIFF('day', plq.last_dispensed, {ref_date})
                  > LEAST(
                      LEAST(plq.last_qty_dispensed / NULLIF(ps.avg_qty_per_visit, 0), 2.0)
                      * ps.mean_interval_days,
                      180
                    ) * 1.2
        )
        SELECT
            COALESCE(t.canonical_name, o.product_id::VARCHAR)  AS canonical_name,
            arp.days_of_cover,
            COUNT(DISTINCT o.patient_id)                       AS overdue_patient_count,
            ROUND(AVG(o.days_overdue), 0)                      AS avg_days_overdue,
            ROUND(AVG(o.estimated_supply_days), 0)             AS avg_estimated_supply_days,
            t.therapeutic_class,
            t.therapeutic_subclass
        FROM overdue o
        JOIN at_risk_products arp ON o.product_id = arp.product_id
        LEFT JOIN HOSPITALS.REPORTING.CANONICAL_PRODUCT_TAXONOMY t
            ON UPPER('{schema}') = UPPER(t.facility)
            AND o.product_id = t.product_id
        GROUP BY
            COALESCE(t.canonical_name, o.product_id::VARCHAR),
            arp.days_of_cover,
            t.therapeutic_class,
            t.therapeutic_subclass
        HAVING overdue_patient_count > 0
        ORDER BY overdue_patient_count DESC, arp.days_of_cover ASC
    """)


def get_overdue_patient_list(schema: str, ref_date: str = "CURRENT_DATE") -> pd.DataFrame:
    """
    Patient-level detail for the overdue refill contact list.
    Same logic as get_patient_refill_overdue but returns one row per
    (patient_id, drug) instead of aggregated counts.

    Uses the visit-interval method (see get_patient_refill_overdue):
      estimated_supply = (last_qty / avg_qty_per_visit) × mean_interval_days
    mean_interval >= 7d guard screens out inpatient dispensing patterns.
    Injectable formulations excluded. Minimum last_qty ≥ 5 units.

    Used on the Patient Risk page to give staff actionable patient IDs
    for clinical follow-up.

    Returns columns:
        patient_id, canonical_name, therapeutic_class, therapeutic_subclass,
        last_dispensed, last_qty_dispensed, estimated_supply_days,
        days_overdue, days_of_cover
    """
    _refill_classes = (
        "Cardiovascular",
        "Endocrine & Metabolic",
        "Neurological",
        "Antiepileptics",
        "Oncology",
        "Antiretrovirals",
        "Psychiatric",
        "Respiratory",
        "Musculoskeletal",
        "Gastrointestinal",
    )
    _refill_subclasses = ("Opioid Analgesics",)
    _exclude_subclasses = (
        "Carbapenems",
        "Glycopeptides",
        "Oxazolidinones",
        "Uterotonics",
    )
    _bad_canonical_names = (
        "AMLODIPINE 80MG TABLET",
    )
    _bad_names_excl = ", ".join(f"'{n}'" for n in _bad_canonical_names)

    _cls_in   = ", ".join(f"'{c}'" for c in _refill_classes)
    _sub_in   = ", ".join(f"'{s}'" for s in _refill_subclasses)
    _sub_excl = ", ".join(f"'{s}'" for s in _exclude_subclasses)

    return run_query(f"""
        WITH eligible_products AS (
            SELECT DISTINCT t.product_id
            FROM HOSPITALS.REPORTING.CANONICAL_PRODUCT_TAXONOMY t
            WHERE UPPER(t.facility) = UPPER('{schema}')
              AND t.product_category = 'pharma'
              AND t.therapeutic_subclass IS NOT NULL
              AND t.therapeutic_class    IS NOT NULL
              AND t.therapeutic_subclass NOT IN ({_sub_excl})
              AND (
                  t.therapeutic_class    IN ({_cls_in})
                  OR t.therapeutic_subclass IN ({_sub_in})
              )
              AND UPPER(t.canonical_name) NOT LIKE '% INJECTION'
              AND t.canonical_name NOT IN ({_bad_names_excl})
        ),
        soh_latest AS (
            SELECT
                product_id,
                soh_after_display AS current_soh_display,
                ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY dispensed_at DESC) AS rn
            FROM HOSPITALS.REPORTING.FACT_DISPENSING
            WHERE source_schema = '{schema}'
        ),
        consumption AS (
            SELECT
                product_id,
                SUM(quantity_dispensed) /
                    NULLIF(COUNT(DISTINCT DATE_TRUNC('day', dispensed_at)), 0) AS avg_daily
            FROM HOSPITALS.REPORTING.FACT_DISPENSING
            WHERE source_schema = '{schema}'
              AND dispensed_at >= DATEADD('day', -90, {ref_date})
            GROUP BY product_id
        ),
        at_risk_products AS (
            SELECT
                s.product_id,
                ROUND(s.current_soh_display / NULLIF(c.avg_daily, 0), 1) AS days_of_cover
            FROM soh_latest s
            INNER JOIN eligible_products ep ON s.product_id = ep.product_id
            LEFT JOIN consumption c ON s.product_id = c.product_id
            WHERE s.rn = 1
              AND s.current_soh_display / NULLIF(c.avg_daily, 0) < 7
        ),
        patient_last_qty AS (
            -- 180-day lookback: limits list to patients still active at this facility.
            -- patient_stats uses the deeper 365-day window to compute reliable visit patterns.
            SELECT
                patient_id,
                product_id,
                quantity_dispensed AS last_qty_dispensed,
                dispensed_at       AS last_dispensed
            FROM HOSPITALS.REPORTING.FACT_DISPENSING
            WHERE source_schema = '{schema}'
              AND dispensed_at >= DATEADD('day', -180, {ref_date})
              AND patient_id IS NOT NULL
              AND quantity_dispensed > 0
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY patient_id, product_id ORDER BY dispensed_at DESC
            ) = 1
        ),
        patient_stats AS (
            SELECT
                f.patient_id,
                f.product_id,
                COUNT(DISTINCT DATE_TRUNC('day', f.dispensed_at))            AS n_visits,
                SUM(f.quantity_dispensed)                                     AS total_qty,
                DATEDIFF('day', MIN(f.dispensed_at), MAX(f.dispensed_at)) /
                    NULLIF(COUNT(DISTINCT DATE_TRUNC('day', f.dispensed_at)) - 1, 0)
                    AS mean_interval_days,
                SUM(f.quantity_dispensed) /
                    NULLIF(COUNT(DISTINCT DATE_TRUNC('day', f.dispensed_at)), 0)
                    AS avg_qty_per_visit
            FROM HOSPITALS.REPORTING.FACT_DISPENSING f
            INNER JOIN at_risk_products arp ON f.product_id = arp.product_id
            WHERE f.source_schema = '{schema}'
              AND f.dispensed_at >= DATEADD('day', -365, {ref_date})
              AND f.patient_id IS NOT NULL
              AND f.quantity_dispensed > 0
            GROUP BY f.patient_id, f.product_id
            HAVING COUNT(DISTINCT DATE_TRUNC('day', f.dispensed_at)) >= 2
               AND DATEDIFF('day', MIN(f.dispensed_at), MAX(f.dispensed_at)) /
                       NULLIF(COUNT(DISTINCT DATE_TRUNC('day', f.dispensed_at)) - 1, 0) >= 7
               AND DATEDIFF('day', MIN(f.dispensed_at), MAX(f.dispensed_at)) /
                       NULLIF(COUNT(DISTINCT DATE_TRUNC('day', f.dispensed_at)) - 1, 0) <= 90
        ),
        overdue AS (
            -- 2.0× ratio cap: prevents supply over-inflation when last fill was unusually large.
            SELECT
                ps.patient_id,
                ps.product_id,
                plq.last_dispensed,
                plq.last_qty_dispensed,
                LEAST(
                    ROUND(
                        LEAST(plq.last_qty_dispensed / NULLIF(ps.avg_qty_per_visit, 0), 2.0)
                        * ps.mean_interval_days,
                    0),
                    180
                )                                                            AS estimated_supply_days,
                ROUND(
                    DATEDIFF('day', plq.last_dispensed, {ref_date})
                    - LEAST(
                        LEAST(plq.last_qty_dispensed / NULLIF(ps.avg_qty_per_visit, 0), 2.0)
                        * ps.mean_interval_days,
                        180
                      ) * 1.2,
                0)                                                           AS days_overdue
            FROM patient_stats ps
            JOIN patient_last_qty plq
                ON ps.patient_id = plq.patient_id AND ps.product_id = plq.product_id
            WHERE plq.last_qty_dispensed >= 5
              AND DATEDIFF('day', plq.last_dispensed, {ref_date})
                  > LEAST(
                      LEAST(plq.last_qty_dispensed / NULLIF(ps.avg_qty_per_visit, 0), 2.0)
                      * ps.mean_interval_days,
                      180
                    ) * 1.2
        )
        SELECT
            o.patient_id,
            COALESCE(t.canonical_name, o.product_id::VARCHAR)  AS canonical_name,
            t.therapeutic_class,
            t.therapeutic_subclass,
            o.last_dispensed::DATE                             AS last_dispensed,
            o.last_qty_dispensed,
            o.estimated_supply_days,
            o.days_overdue,
            arp.days_of_cover
        FROM overdue o
        JOIN at_risk_products arp ON o.product_id = arp.product_id
        LEFT JOIN HOSPITALS.REPORTING.CANONICAL_PRODUCT_TAXONOMY t
            ON UPPER('{schema}') = UPPER(t.facility)
            AND o.product_id = t.product_id
        ORDER BY arp.days_of_cover ASC, o.days_overdue DESC
    """)
