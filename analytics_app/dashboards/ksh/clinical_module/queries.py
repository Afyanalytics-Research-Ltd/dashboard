"""
queries.py — Afya Clinical Analytics
======================================
All SQL query functions. One function per query.
Each function accepts filters: dict and run_query callable,
returns pd.DataFrame.

Naming convention: load_<query_name>

Tab 1 — Operations:
  load_tab1_kpis               KPI row (visits, discharges, active admissions, costs)
  load_service_growth          Q1A: monthly inpatient/outpatient line chart
  load_ward_breakdown          Q1B: ward summary table
  load_ward_admission_trend    Q1B2: ward admission growth over time
  load_ward_discharge_latency  Q1B3: discharge latency vs patient count per ward
  load_ward_active_vs_hours    Q1B4: active admissions vs avg hours monthly
  load_ward_cost_volume        Q1B5: cost per ward vs patient volumes
  load_ward_segment_why        Q1B6: top diagnoses + new/returning per ward category
  load_avg_admission_cost_full Q1B7: avg admission cost from invoice line items
  load_volume_spike_context    Q1B8: diagnosis breakdown for anomalous spike/dip months
  load_peak_demand_monthly     Q1B9: monthly volume by type with z-score peak flags
  load_volume_new_vs_returning B1:  new vs returning patients monthly
  load_volume_top_diagnoses    B2:  top diagnoses by month
  load_journey_times           Q3:  patient journey times (in hours)
  load_investigation_columns   discover available columns in STG_EVALUATION_INVESTIGATIONS
  load_lab_turnaround_by_discipline Q3B: lab turnaround by clinical discipline (Haematology, Chemistry…)
  load_lab_turnaround_by_test  Q3B-fallback: lab turnaround by investigation_type only
  load_inpatient_funnel        Q3C: inpatient conversion funnel
  load_encounter_forecast      Q4:  forecast with confidence intervals, split by type
  load_clinician_load          Q5:  clinician load + documentation quality
  load_peak_demand_heatmap     Q6A: hour × day heatmap with visit type split

Tab 2 — Segmentation:
  load_seg_kpis                KPI row
  load_demographics_age_sex    Q2:  age × sex × chronic grid
  load_new_vs_returning        Q4:  new vs returning trend
  load_payer_mix               Q5:  payer mix by age group
  load_revenue_by_segment      Q6:  revenue by clinical segment
  load_pareto                  Q7:  revenue Pareto by spend tier
  load_cohort_forecast         Q9:  age cohort monthly counts

Tab 3 — Retention:
  load_retention_kpis          KPI row
  load_lifecycle               Q1:  lifecycle active/lapsing/LTFU
  load_retention_by_payer      Q3:  90-day retention by payer
  load_dropout_causes          Q6:  dropout cause attribution
  load_revenue_at_risk         Q7:  revenue at risk from LTFU
  load_outreach_list           Q11: re-engagement outreach list

Tab 4 — Disease Burden:
  load_burden_kpis             A1:  burden KPI snapshot
  load_burden_trend            A2:  burden group monthly trend
  load_top_diagnoses           A3:  top 10 diagnoses
  load_undetected_ncd          A6:  elevated vitals without NCD code
  load_ncd_kpis                B1:  NCD KPI snapshot
  load_ncd_by_age              B2:  NCD by age group
  load_htn_controlled          B5:  HTN controlled vs uncontrolled
  load_anc_funnel              C2:  ANC funnel
  load_deliveries_by_age       C3:  deliveries by maternal age
  load_communicable_trend      D2:  communicable disease trend
  load_hiv_profile             D4:  HIV patient profile
  load_mh_kpis                 E1:  mental health KPI snapshot
  load_mh_by_age_sex           E2:  MH by age and sex
  load_revenue_by_burden_group F1:  revenue by burden group

Tab 5 — Workload & Quality:
  load_shortcut_rate           shortcut rate per clinician
  load_bp_omission_rate        BP omission on HTN visits
  load_return_72h              unplanned 72h return rate

Clinician View:
  load_todays_patients         CL1: today's patient list + priority score
  load_patient_vitals_trend    CL2: last 6 vitals + trend direction
  load_medication_continuity   CL3: expected drugs vs active prescriptions
"""


import sys
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # → dashboards/

try:
    from snowflake_service.snowflake_client import SnowflakeClient as _SnowflakeClient

    @st.cache_resource
    def _get_client():
        return _SnowflakeClient()

    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.lower() for c in df.columns]
        return df

    @st.cache_data(ttl=3600, show_spinner=False)
    def run_query(sql: str) -> pd.DataFrame:
        try:
            return _normalise(_get_client().query(sql))
        except Exception as e:
            if any(k in str(e).lower() for k in ("expired", "authentication", "390114")):
                _get_client.clear()
                run_query.clear()
                return _normalise(_get_client().query(sql))
            raise

except ImportError:
    def run_query(sql: str) -> pd.DataFrame:
        raise RuntimeError("SnowflakeClient not available — check snowflake_service path")


# ─── FILTER HELPERS ───────────────────────────────────────────────────────────

def _w(filters: dict, alias: str = "v") -> str:
    """Build optional WHERE additions for visit-based tables (have source_schema + clinic)."""
    parts = []
    schemas = filters.get("source_schemas") or (
        [filters["schema"]] if filters.get("schema") else []
    )
    if schemas:
        quoted = ", ".join(f"'{s}'" for s in schemas)
        parts.append(f"AND {alias}.source_schema IN ({quoted})")
    facilities = filters.get("facilities") or (
        [filters["facility"]] if filters.get("facility") else []
    )
    if facilities:
        quoted = ", ".join(f"'{f}'" for f in facilities)
        parts.append(f"AND {alias}.clinic IN ({quoted})")
    if filters.get("date_from"):
        parts.append(f"AND {alias}.created_at >= '{filters['date_from']}'")
    if filters.get("date_to"):
        parts.append(f"AND {alias}.created_at <= '{filters['date_to']}'")
    return "\n    ".join(parts)


def _w_adm(filters: dict, alias: str = "a") -> str:
    """Build optional WHERE additions for STG_INPATIENT_ADMISSIONS.
    Admissions source_schema is stored with '_clean' suffix; no clinic column."""
    parts = []
    schemas = filters.get("source_schemas") or (
        [filters["schema"]] if filters.get("schema") else []
    )
    if schemas:
        quoted = ", ".join(f"'{s}'" for s in schemas)
        parts.append(
            f"AND REPLACE(LOWER({alias}.source_schema), '_clean', '') IN ({quoted})"
        )
    if filters.get("date_from"):
        parts.append(f"AND {alias}.admitted_at >= '{filters['date_from']}'")
    if filters.get("date_to"):
        parts.append(f"AND {alias}.admitted_at <= '{filters['date_to']}'")
    return "\n    ".join(parts)


def _wsa(filters: dict) -> str:
    """Schema-only filter for schema_anchor CTEs.
    Restricts MAX(created_at) to selected schemas so the date anchor is correct
    and the INNER JOIN naturally excludes every other schema from all downstream CTEs."""
    parts = []
    schemas = filters.get("source_schemas") or (
        [filters["schema"]] if filters.get("schema") else []
    )
    if schemas:
        quoted = ", ".join(f"'{s}'" for s in schemas)
        parts.append(f"source_schema IN ({quoted})")
    if filters.get("date_from"):
        parts.append(f"created_at >= '{filters['date_from']}'")
    if filters.get("date_to"):
        parts.append(f"created_at <= '{filters['date_to']}'")
    if parts:
        return "WHERE " + "\n    AND ".join(parts)
    return ""


def _mo(filters: dict) -> int:
    if filters.get("months_back"):
        return int(filters["months_back"])
    # Custom date range: use a large window — _w() applies the exact date_from/date_to
    if filters.get("date_range") == "Custom" or filters.get("date_from"):
        return 999
    mapping = {
        "Last 12 months": 12,
        "Last 6 months":  6,
        "Last 90 days":   3,
    }
    return mapping.get(filters.get("date_range", "Last 12 months"), 12)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_tab1_kpis(filters: dict, run_query) -> pd.DataFrame:
    """KPI row: total visits, inpatient/outpatient split, discharges, active admissions, costs."""
    wh   = _w(filters)
    wsa = _wsa(filters)
    wh_a = _w_adm(filters)
    mo   = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    {wsa}
    GROUP BY source_schema
),
base AS (
    SELECT v.source_schema, v.id AS visit_id
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
),
admissions AS (
    SELECT REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema,
           a.visit_id,
           a.admission_cost,
           a.is_open_admission
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    INNER JOIN schema_anchor sa
        ON REPLACE(LOWER(a.source_schema), '_clean', '') = sa.source_schema
    WHERE a.admitted_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh_a}
),
op_costs AS (
    SELECT v.source_schema, v.id AS visit_id,
           SUM(il.item_amount) AS visit_cost
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a2
        ON v.id = a2.visit_id
       AND v.source_schema = REPLACE(LOWER(a2.source_schema), '_clean', '')
    INNER JOIN HOSPITALS.STAGING.STG_INVOICE_LINE_ITEMS il
        ON v.id = il.visit_id AND v.source_schema = il.source_schema
       AND il.invoice_deleted_at IS NULL
       AND (il.auto_cancelled IS NULL OR il.auto_cancelled = 0)
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      AND a2.visit_id IS NULL
      {wh}
    GROUP BY 1, 2
)
SELECT
    COUNT(DISTINCT b.visit_id)                              AS total_visits,
    COUNT(DISTINCT CASE WHEN adm.visit_id IS NOT NULL
                        THEN b.visit_id END)                AS inpatient_visits,
    COUNT(DISTINCT CASE WHEN adm.visit_id IS NULL
                        THEN b.visit_id END)                AS outpatient_visits,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN adm.visit_id IS NOT NULL THEN b.visit_id END),
        COUNT(DISTINCT b.visit_id)
    ) * 100, 1)                                             AS inpatient_pct,
    COUNT(DISTINCT CASE WHEN adm.is_open_admission = 0
                        THEN adm.visit_id END)              AS total_discharges,
    COUNT(DISTINCT CASE WHEN adm.is_open_admission = 1
                        THEN adm.visit_id END)              AS active_admissions,
    ROUND(AVG(CASE WHEN adm.visit_id IS NOT NULL
                   THEN adm.admission_cost END), 0)         AS avg_admission_cost,
    ROUND(AVG(c.visit_cost), 0)                             AS avg_op_cost
FROM base b
LEFT JOIN admissions adm
    ON b.visit_id = adm.visit_id AND b.source_schema = adm.source_schema
LEFT JOIN op_costs c
    ON b.visit_id = c.visit_id AND b.source_schema = c.source_schema
"""
    return run_query(sql)


def load_service_growth(filters: dict, run_query) -> pd.DataFrame:
    """Q1A: Monthly visit volume by service type."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    {wsa}
    GROUP BY source_schema
),
visit_base AS (
    SELECT
        v.source_schema,
        v.clinic                                              AS facility,
        v.id                                                  AS visit_id,
        DATE_TRUNC('month', v.created_at)                    AS visit_month,
        CASE WHEN a.visit_id  IS NOT NULL THEN 1 ELSE 0 END  AS is_inpatient,
        CASE WHEN inv.visit_id IS NOT NULL THEN 1 ELSE 0 END AS has_investigation,
        CASE WHEN pp.visit_id  IS NOT NULL THEN 1 ELSE 0 END AS has_prescription
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
        ON v.id = a.visit_id
       AND v.source_schema = REPLACE(LOWER(a.source_schema), '_clean', '')
    LEFT JOIN (
        SELECT DISTINCT visit_id, source_schema
        FROM HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS
        WHERE (cancelled IS NULL OR cancelled = 0)
          AND (remove_from_report IS NULL OR remove_from_report = 0)
          AND LOWER(TRIM(investigation_type))
              IN ('laboratory', 'lab', 'radiology', 'ultrasound')
    ) inv ON v.id = inv.visit_id AND v.source_schema = inv.source_schema
    LEFT JOIN (
        SELECT DISTINCT visit_id, source_schema
        FROM HOSPITALS.STAGING.STG_PRESCRIPTION_PAYMENTS
        WHERE remove_from_report IS NULL OR remove_from_report = 0
    ) pp ON v.id = pp.visit_id AND v.source_schema = pp.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
)
SELECT
    visit_month,
    SUM(is_inpatient)                                        AS inpatient,
    SUM(CASE WHEN is_inpatient=0 AND has_investigation=1
             THEN 1 ELSE 0 END)                             AS outpatient_with_lab,
    SUM(CASE WHEN is_inpatient=0 AND has_investigation=0
              AND has_prescription=1 THEN 1 ELSE 0 END)     AS outpatient_rx_only,
    SUM(CASE WHEN is_inpatient=0 AND has_investigation=0
              AND has_prescription=0 THEN 1 ELSE 0 END)     AS consult_only,
    COUNT(*)                                                 AS total_visits
FROM visit_base
GROUP BY 1
ORDER BY 1
"""
    return run_query(sql)


def load_ward_breakdown(filters: dict, run_query) -> pd.DataFrame:
    """Q1B: Ward summary — admissions, LOS, discharge latency, invoice-based cost, readmissions."""
    wh   = _w_adm(filters)
    wsa = _wsa(filters)
    wh_v = _w(filters, alias="v")
    mo   = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    {wsa}
    GROUP BY source_schema
),
ward_invoices AS (
    -- True ward revenue from invoice line items, scoped to admitted visits
    SELECT
        COALESCE(a.ward_name, 'Unknown')    AS ward,
        COALESCE(a.ward_category, 'Unknown') AS ward_category,
        a.visit_id,
        REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema,
        SUM(il.item_amount)                 AS visit_invoice_total
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    INNER JOIN schema_anchor sa
        ON REPLACE(LOWER(a.source_schema), '_clean', '') = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON a.visit_id = v.id
       AND REPLACE(LOWER(a.source_schema), '_clean', '') = v.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_INVOICE_LINE_ITEMS il
        ON v.id = il.visit_id AND v.source_schema = il.source_schema
       AND il.invoice_deleted_at IS NULL
       AND (il.auto_cancelled IS NULL OR il.auto_cancelled = 0)
    WHERE a.admitted_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
    GROUP BY 1, 2, 3, 4
)
SELECT
    COALESCE(a.ward_name, 'Unknown')                        AS ward,
    COALESCE(a.ward_category, 'Unknown')                    AS ward_category,
    COUNT(DISTINCT a.visit_id)                              AS admissions,
    COUNT(DISTINCT CASE WHEN a.is_open_admission = 1
                        THEN a.visit_id END)                AS active_admissions,
    ROUND(AVG(NULLIF(a.los_days, 0)), 1)                    AS avg_los_days,
    ROUND(AVG(NULLIF(a.discharge_latency_hours, 0)), 1)     AS avg_discharge_latency_hrs,
    COUNT(DISTINCT CASE WHEN a.is_30day_readmission = 1
                        THEN a.visit_id END)                AS readmissions_30day,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN a.is_30day_readmission = 1 THEN a.visit_id END),
        COUNT(DISTINCT a.visit_id)
    ) * 100, 1)                                             AS readmission_rate_pct,
    -- Invoice-based cost (accurate) with fallback to admission_cost field (partial)
    ROUND(COALESCE(
        AVG(wi.visit_invoice_total),
        AVG(a.admission_cost)
    ), 0)                                                   AS avg_admission_cost,
    ROUND(COALESCE(
        SUM(wi.visit_invoice_total),
        SUM(a.admission_cost)
    ), 0)                                                   AS total_admission_revenue
FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
INNER JOIN schema_anchor sa
    ON REPLACE(LOWER(a.source_schema), '_clean', '') = sa.source_schema
LEFT JOIN ward_invoices wi
    ON a.visit_id = wi.visit_id
   AND REPLACE(LOWER(a.source_schema), '_clean', '') = wi.source_schema
   AND COALESCE(a.ward_name, 'Unknown') = wi.ward
WHERE a.admitted_at >= DATEADD('month', -{mo}, sa.max_date)
{wh}
GROUP BY 1, 2
ORDER BY admissions DESC
"""
    return run_query(sql)


def load_ward_admission_trend(filters: dict, run_query) -> pd.DataFrame:
    """Ward admission numbers over time (monthly time series per ward, top 6 wards)."""
    wh = _w_adm(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    {wsa}
    GROUP BY source_schema
),
monthly AS (
    SELECT
        DATE_TRUNC('month', a.admitted_at)                  AS visit_month,
        COALESCE(a.ward_name, 'Unknown')                    AS ward,
        COUNT(DISTINCT a.visit_id)                          AS admissions
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    INNER JOIN schema_anchor sa
        ON REPLACE(LOWER(a.source_schema), '_clean', '') = sa.source_schema
    WHERE a.admitted_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
    GROUP BY 1, 2
),
top_wards AS (
    SELECT ward FROM monthly GROUP BY ward ORDER BY SUM(admissions) DESC LIMIT 6
)
SELECT m.visit_month, m.ward, m.admissions
FROM monthly m
INNER JOIN top_wards tw ON m.ward = tw.ward
ORDER BY 1, 2
"""
    return run_query(sql)


def load_ward_discharge_latency(filters: dict, run_query) -> pd.DataFrame:
    """Discharge latency (hrs) vs patient count per ward — which ward takes longest."""
    wh = _w_adm(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    {wsa}
    GROUP BY source_schema
)
SELECT
    COALESCE(a.ward_name, 'Unknown')                        AS ward,
    COUNT(DISTINCT a.visit_id)                              AS patient_count,
    ROUND(AVG(NULLIF(a.discharge_latency_hours, 0)), 1)     AS avg_discharge_latency_hrs,
    ROUND(MEDIAN(NULLIF(a.discharge_latency_hours, 0)), 1)  AS median_discharge_latency_hrs
FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
INNER JOIN schema_anchor sa
    ON REPLACE(LOWER(a.source_schema), '_clean', '') = sa.source_schema
WHERE a.admitted_at >= DATEADD('month', -{mo}, sa.max_date)
  AND a.is_open_admission = 0
  AND a.discharge_latency_hours IS NOT NULL
{wh}
GROUP BY 1
HAVING patient_count >= 5
ORDER BY avg_discharge_latency_hrs DESC
"""
    return run_query(sql)


def load_ward_active_vs_hours(filters: dict, run_query) -> pd.DataFrame:
    """Monthly active admissions vs avg admission hours per ward."""
    wh = _w_adm(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    {wsa}
    GROUP BY source_schema
)
SELECT
    DATE_TRUNC('month', a.admitted_at)                      AS visit_month,
    COUNT(DISTINCT CASE WHEN a.is_open_admission = 1
                        THEN a.visit_id END)                AS active_admissions,
    ROUND(AVG(NULLIF(a.los_days, 0)) * 24, 1)              AS avg_admission_hours,
    COUNT(DISTINCT a.visit_id)                              AS total_admissions
FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
INNER JOIN schema_anchor sa
    ON REPLACE(LOWER(a.source_schema), '_clean', '') = sa.source_schema
WHERE a.admitted_at >= DATEADD('month', -{mo}, sa.max_date)
{wh}
GROUP BY 1
ORDER BY 1
"""
    return run_query(sql)


def load_ward_cost_volume(filters: dict, run_query) -> pd.DataFrame:
    """Cost per ward vs patient volumes — invoice-line-item based cost."""
    wh = _w_adm(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    {wsa}
    GROUP BY source_schema
),
visit_invoice AS (
    SELECT
        v.id          AS visit_id,
        v.source_schema,
        SUM(il.item_amount) AS invoice_total
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_INVOICE_LINE_ITEMS il
        ON v.id = il.visit_id AND v.source_schema = il.source_schema
       AND il.invoice_deleted_at IS NULL
       AND (il.auto_cancelled IS NULL OR il.auto_cancelled = 0)
    GROUP BY 1, 2
)
SELECT
    COALESCE(a.ward_name, 'Unknown')                        AS ward,
    COUNT(DISTINCT a.visit_id)                              AS admissions,
    ROUND(AVG(NULLIF(vi.invoice_total, 0)), 0)              AS avg_admission_cost,
    ROUND(SUM(COALESCE(vi.invoice_total, 0)), 0)            AS total_revenue,
    ROUND(DIV0(SUM(COALESCE(vi.invoice_total, 0)),
               NULLIF(COUNT(DISTINCT a.visit_id), 0)), 0)   AS revenue_per_patient
FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
INNER JOIN schema_anchor sa
    ON REPLACE(LOWER(a.source_schema), '_clean', '') = sa.source_schema
LEFT JOIN visit_invoice vi
    ON a.visit_id = vi.visit_id
   AND REPLACE(LOWER(a.source_schema), '_clean', '') = vi.source_schema
WHERE a.admitted_at >= DATEADD('month', -{mo}, sa.max_date)
{wh}
GROUP BY 1
HAVING admissions >= 5
ORDER BY admissions DESC
LIMIT 15
"""
    return run_query(sql)


def load_ward_segment_why(filters: dict, run_query) -> pd.DataFrame:
    """Why are female/paediatric wards busiest: top diagnoses + new vs returning per ward category."""
    wh   = _w(filters)
    wsa = _wsa(filters)
    wh_a = _w_adm(filters)
    mo   = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    {wsa}
    GROUP BY source_schema
),
atf AS (
    SELECT source_schema, patient, MIN(created_at) AS first_ever
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS GROUP BY 1, 2
),
admitted AS (
    SELECT REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema,
           a.visit_id,
           COALESCE(a.ward_category, 'Unknown')          AS ward_category,
           NULLIF(a.los_days, 0)                         AS los_days
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    INNER JOIN schema_anchor sa
        ON REPLACE(LOWER(a.source_schema), '_clean', '') = sa.source_schema
    WHERE a.admitted_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh_a}
),
visits AS (
    SELECT v.source_schema, v.id AS visit_id, v.patient,
           DATE_TRUNC('month', v.created_at) AS visit_month,
           COALESCE(NULLIF(TRIM(dx.disease_group_1), ''),
               CASE
                   WHEN n.diagnosis ILIKE '%malaria%'      THEN 'Malaria'
                   WHEN n.diagnosis ILIKE '%urti%'         THEN 'URTI'
                   WHEN n.diagnosis ILIKE '%hypertension%' THEN 'Hypertension'
                   WHEN n.diagnosis ILIKE '%diabetes%'     THEN 'Diabetes'
                   WHEN n.diagnosis ILIKE '%anc%'          THEN 'ANC / Maternal'
                   WHEN n.diagnosis ILIKE '%delivery%'     THEN 'Delivery'
                   WHEN n.diagnosis ILIKE '%pneumonia%'    THEN 'Pneumonia'
                   ELSE 'Other / Unspecified'
               END)                                        AS diagnosis_group,
           CASE WHEN DATE_TRUNC('month', atf.first_ever) = DATE_TRUNC('month', v.created_at)
                THEN 'New' ELSE 'Returning' END            AS patient_type
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON v.id = n.visit_id AND v.source_schema = n.source_schema
    LEFT JOIN atf ON v.patient = atf.patient AND v.source_schema = atf.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
)
SELECT
    COALESCE(adm.ward_category, 'Unknown')               AS ward_category,
    vis.diagnosis_group,
    vis.patient_type,
    COUNT(DISTINCT adm.visit_id)                         AS admissions,
    COUNT(DISTINCT vis.patient)                          AS unique_patients,
    ROUND(AVG(adm.los_days), 1)                          AS avg_los_days
FROM admitted adm
LEFT JOIN visits vis
    ON adm.visit_id = vis.visit_id AND adm.source_schema = vis.source_schema
GROUP BY 1, 2, 3
ORDER BY 1, admissions DESC
"""
    return run_query(sql)


def load_top_ward_summary(filters: dict, run_query) -> pd.DataFrame:
    """Top-3 wards by admissions with a full operational profile:
    capacity pressure, top condition, new/returning split, payer mix,
    avg LOS, investigation rate, clinician-to-patient ratio."""
    wh_a = _w_adm(filters)
    wsa  = _wsa(filters)
    mo   = _mo(filters)
    sql  = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
atf AS (
    SELECT patient, source_schema, MIN(DATE_TRUNC('month', created_at)) AS first_month
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    GROUP BY 1, 2
),
admissions_base AS (
    SELECT
        COALESCE(a.ward_name, 'Unknown')                             AS ward,
        a.visit_id,
        REPLACE(LOWER(a.source_schema), '_clean', '')                AS source_schema,
        DATE_TRUNC('month', a.admitted_at)                           AS admit_month,
        a.los_days,
        a.is_open_admission
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    INNER JOIN schema_anchor sa
        ON REPLACE(LOWER(a.source_schema), '_clean', '') = sa.source_schema
    WHERE a.admitted_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh_a}
),
top_wards AS (
    SELECT ward
    FROM admissions_base
    GROUP BY 1
    ORDER BY COUNT(DISTINCT visit_id) DESC
    LIMIT 3
),
ab AS (
    SELECT ab.* FROM admissions_base ab INNER JOIN top_wards tw ON ab.ward = tw.ward
),
-- ── Capacity pressure: recent 3 months vs prior 3 months ─────────────
period_bounds AS (
    SELECT MAX(admit_month) AS last_month FROM ab
),
monthly_vol AS (
    SELECT ab.ward, ab.admit_month,
           COUNT(DISTINCT ab.visit_id) AS month_adm
    FROM ab CROSS JOIN period_bounds pb
    GROUP BY 1, 2
),
pressure AS (
    SELECT
        mv.ward,
        ROUND(AVG(CASE WHEN mv.admit_month > DATEADD('month', -3, pb.last_month)
                       THEN mv.month_adm END), 1) AS recent_avg,
        ROUND(AVG(CASE WHEN mv.admit_month <= DATEADD('month', -3, pb.last_month)
                       AND mv.admit_month > DATEADD('month', -6, pb.last_month)
                       THEN mv.month_adm END), 1) AS prior_avg,
        ROUND(STDDEV(mv.month_adm), 1)             AS monthly_stddev,
        ROUND(AVG(mv.month_adm), 1)                AS monthly_mean
    FROM monthly_vol mv CROSS JOIN period_bounds pb
    GROUP BY 1
),
-- ── Top condition (surgery fallback for Other) ────────────────────────
dx_raw AS (
    SELECT
        ab.ward,
        COALESCE(
            NULLIF(TRIM(dx.disease_group_1), ''),
            CASE
                WHEN n.diagnosis ILIKE '%surgery%'
                  OR n.diagnosis ILIKE '%operation%'
                  OR n.diagnosis ILIKE '%operative%'
                  OR n.diagnosis ILIKE '%theatre%'
                THEN 'Surgical Procedure'
                ELSE 'Other / Unspecified'
            END
        )                              AS condition_group,
        COUNT(DISTINCT ab.visit_id)    AS cnt
    FROM ab
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON ab.visit_id = dx.visit_id AND ab.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON ab.visit_id = n.visit_id AND ab.source_schema = n.source_schema
    GROUP BY 1, 2
),
dx_totals AS (
    SELECT ward, SUM(cnt) AS total_dx FROM dx_raw GROUP BY 1
),
top_condition AS (
    SELECT dr.ward, dr.condition_group, dr.cnt,
           dt.total_dx,
           ROW_NUMBER() OVER (PARTITION BY dr.ward ORDER BY dr.cnt DESC) AS rk
    FROM dx_raw dr JOIN dx_totals dt ON dr.ward = dt.ward
),
-- ── New vs returning ──────────────────────────────────────────────────
pt_raw AS (
    SELECT
        ab.ward,
        COUNT(DISTINCT ab.visit_id)                                      AS total_adm,
        COUNT(DISTINCT CASE
            WHEN atf.first_month = ab.admit_month THEN ab.visit_id END)  AS new_adm
    FROM ab
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON ab.visit_id = v.id AND ab.source_schema = v.source_schema
    LEFT JOIN atf
        ON v.patient = atf.patient AND v.source_schema = atf.source_schema
    GROUP BY 1
),
-- ── Payer mix ─────────────────────────────────────────────────────────
payer_raw AS (
    SELECT
        ab.ward,
        CASE
            WHEN LOWER(v.payment_mode) IN ('nhif','shif','sha','national scheme')
                THEN 'NHIF / SHA'
            WHEN LOWER(v.payment_mode) IN ('cash','self-pay','out-of-pocket','copay')
                THEN 'Cash'
            WHEN v.payment_mode IS NULL OR TRIM(v.payment_mode) = ''
                THEN 'Unknown'
            ELSE 'Insurance'
        END                               AS payer,
        COUNT(DISTINCT ab.visit_id)       AS payer_cnt
    FROM ab
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON ab.visit_id = v.id AND ab.source_schema = v.source_schema
    GROUP BY 1, 2
),
payer_totals AS (SELECT ward, SUM(payer_cnt) AS total FROM payer_raw GROUP BY 1),
top_payer AS (
    SELECT pr.ward, pr.payer, pr.payer_cnt, pt.total,
           ROW_NUMBER() OVER (PARTITION BY pr.ward ORDER BY pr.payer_cnt DESC) AS rk
    FROM payer_raw pr JOIN payer_totals pt ON pr.ward = pt.ward
    WHERE pr.payer != 'Unknown'
),
-- ── LOS ───────────────────────────────────────────────────────────────
los AS (
    SELECT ward, ROUND(AVG(NULLIF(los_days, 0)), 1) AS avg_los_days
    FROM ab
    GROUP BY 1
),
-- ── Investigation rate ────────────────────────────────────────────────
inv AS (
    SELECT
        ab.ward,
        COUNT(DISTINCT ab.visit_id)                                      AS total_adm,
        COUNT(DISTINCT CASE WHEN i.visit_id IS NOT NULL
                            THEN ab.visit_id END)                        AS with_inv
    FROM ab
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS i
        ON ab.visit_id = i.visit_id
       AND (i.cancelled IS NULL OR i.cancelled = 0)
    GROUP BY 1
),
-- ── Clinician ratio ───────────────────────────────────────────────────
clinicians AS (
    SELECT
        ab.ward,
        COUNT(DISTINCT v.user)    AS n_clinicians,
        COUNT(DISTINCT v.patient) AS n_patients
    FROM ab
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON ab.visit_id = v.id AND ab.source_schema = v.source_schema
    WHERE v.user IS NOT NULL AND TRIM(v.user) != ''
    GROUP BY 1
)
SELECT
    tw.ward,
    -- Volume
    COUNT(DISTINCT ab.visit_id)                                          AS admissions,
    -- Capacity pressure
    pr.recent_avg,
    pr.prior_avg,
    ROUND(DIV0(pr.recent_avg - pr.prior_avg,
               NULLIF(pr.prior_avg, 0)) * 100, 0)                       AS pressure_pct,
    CASE
        WHEN DIV0(pr.recent_avg - pr.prior_avg,
                  NULLIF(pr.prior_avg, 0)) >  0.10 THEN 'Rising'
        WHEN DIV0(pr.recent_avg - pr.prior_avg,
                  NULLIF(pr.prior_avg, 0)) < -0.10 THEN 'Easing'
        ELSE 'Stable'
    END                                                                  AS pressure_signal,
    ROUND(DIV0(pr.monthly_stddev, NULLIF(pr.monthly_mean, 0)), 2)       AS cv_seasonality,
    -- Top condition
    tc.condition_group                                                   AS top_condition,
    ROUND(DIV0(tc.cnt, tc.total_dx) * 100, 0)                           AS top_condition_pct,
    CASE WHEN DIV0(tc.cnt, tc.total_dx) < 0.35
         THEN 'Varies' ELSE 'Concentrated' END                          AS condition_pattern,
    -- New vs returning
    ROUND(DIV0(pt.new_adm,              pt.total_adm) * 100, 0)         AS new_pct,
    ROUND(DIV0(pt.total_adm - pt.new_adm, pt.total_adm) * 100, 0)      AS returning_pct,
    -- Payer
    tp.payer                                                             AS top_payer,
    ROUND(DIV0(tp.payer_cnt, tp.total) * 100, 0)                        AS top_payer_pct,
    -- LOS
    l.avg_los_days,
    -- Investigation rate
    ROUND(DIV0(i.with_inv, i.total_adm) * 100, 0)                       AS investigation_rate_pct,
    -- Clinician ratio
    cl.n_clinicians                                                      AS clinicians,
    ROUND(DIV0(cl.n_patients, NULLIF(cl.n_clinicians, 0)), 1)           AS patients_per_clinician
FROM top_wards tw
INNER JOIN ab          ON tw.ward = ab.ward
LEFT  JOIN pressure pr ON tw.ward = pr.ward
LEFT  JOIN top_condition tc ON tw.ward = tc.ward AND tc.rk = 1
LEFT  JOIN pt_raw pt   ON tw.ward = pt.ward
LEFT  JOIN top_payer tp ON tw.ward = tp.ward AND tp.rk = 1
LEFT  JOIN los l       ON tw.ward = l.ward
LEFT  JOIN inv i       ON tw.ward = i.ward
LEFT  JOIN clinicians cl ON tw.ward = cl.ward
GROUP BY tw.ward, pr.recent_avg, pr.prior_avg, pr.monthly_stddev, pr.monthly_mean,
         tc.condition_group, tc.cnt, tc.total_dx,
         pt.new_adm, pt.total_adm,
         tp.payer, tp.payer_cnt, tp.total,
         l.avg_los_days, i.with_inv, i.total_adm,
         cl.n_clinicians, cl.n_patients
ORDER BY admissions DESC
"""
    return run_query(sql)


def load_avg_admission_cost_full(filters: dict, run_query) -> pd.DataFrame:
    """Avg admission cost from invoice line items (more accurate than admission_cost field)."""
    wh   = _w(filters)
    wsa = _wsa(filters)
    wh_a = _w_adm(filters)
    mo   = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    {wsa}
    GROUP BY source_schema
),
admitted_visits AS (
    SELECT REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema,
           a.visit_id
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    INNER JOIN schema_anchor sa
        ON REPLACE(LOWER(a.source_schema), '_clean', '') = sa.source_schema
    WHERE a.admitted_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh_a}
),
visit_costs AS (
    SELECT v.source_schema, v.id AS visit_id,
           SUM(il.item_amount)                           AS total_cost,
           MAX(CASE WHEN av.visit_id IS NOT NULL THEN 1 ELSE 0 END) AS is_inpatient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN admitted_visits av ON v.id = av.visit_id AND v.source_schema = av.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_INVOICE_LINE_ITEMS il
        ON v.id = il.visit_id AND v.source_schema = il.source_schema
       AND il.invoice_deleted_at IS NULL
       AND (il.auto_cancelled IS NULL OR il.auto_cancelled = 0)
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
    GROUP BY 1, 2
)
SELECT
    ROUND(AVG(CASE WHEN is_inpatient = 1 THEN total_cost END), 0)  AS avg_ip_cost_full,
    ROUND(AVG(CASE WHEN is_inpatient = 0 THEN total_cost END), 0)  AS avg_op_cost_full,
    COUNT(CASE WHEN is_inpatient = 1 THEN 1 END)                   AS ip_visits_with_cost,
    COUNT(CASE WHEN is_inpatient = 0 THEN 1 END)                   AS op_visits_with_cost
FROM visit_costs
"""
    return run_query(sql)


def load_volume_spike_context(filters: dict, run_query) -> pd.DataFrame:
    """For spike/dip months: top diagnoses breakdown to explain what drove the anomaly."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
monthly_vol AS (
    SELECT DATE_TRUNC('month', v.created_at) AS visit_month,
           COUNT(DISTINCT v.id)              AS total_visits
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
    GROUP BY 1
),
stats AS (
    SELECT visit_month, total_visits,
           AVG(total_visits) OVER ()      AS avg_vol,
           STDDEV(total_visits) OVER ()   AS sd_vol,
           LAG(total_visits) OVER (ORDER BY visit_month) AS prev_visits
    FROM monthly_vol
),
anomalous AS (
    SELECT visit_month, total_visits, prev_visits,
           ROUND(DIV0(total_visits - avg_vol, NULLIF(sd_vol, 0)), 2) AS z_score,
           CASE WHEN total_visits > avg_vol + 1.0 * sd_vol THEN 'Spike'
                WHEN total_visits < avg_vol - 1.0 * sd_vol THEN 'Dip'
                ELSE 'Normal' END                                    AS month_type,
           ROUND(DIV0(total_visits - prev_visits, NULLIF(prev_visits, 0)) * 100, 1) AS mom_pct
    FROM stats
    WHERE ABS(DIV0(total_visits - avg_vol, NULLIF(sd_vol, 0))) >= 1.0
)
SELECT
    a.visit_month,
    a.month_type,
    a.total_visits,
    a.z_score,
    a.mom_pct,
    COALESCE(NULLIF(TRIM(dx.disease_group_1), ''),
        CASE
            WHEN n.diagnosis ILIKE '%malaria%'      THEN 'Malaria'
            WHEN n.diagnosis ILIKE '%urti%'         THEN 'URTI'
            WHEN n.diagnosis ILIKE '%hypertension%' THEN 'Hypertension'
            WHEN n.diagnosis ILIKE '%anc%'          THEN 'ANC / Maternal'
            WHEN n.diagnosis ILIKE '%pneumonia%'    THEN 'Pneumonia'
            ELSE 'Other / Unspecified'
        END)                                                         AS diagnosis_group,
    CASE WHEN adm.visit_id IS NOT NULL THEN 'Inpatient' ELSE 'Outpatient' END AS visit_type,
    COUNT(DISTINCT v.id)                                             AS visit_count,
    COUNT(DISTINCT CASE WHEN atf.first_ever = v.created_at THEN v.patient END) AS new_patients,
    COUNT(DISTINCT v.patient)                                        AS unique_patients
FROM anomalous a
INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    ON DATE_TRUNC('month', v.created_at) = a.visit_month
INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
    ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
    ON v.id = n.visit_id AND v.source_schema = n.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS adm
    ON v.id = adm.visit_id
   AND v.source_schema = REPLACE(LOWER(adm.source_schema), '_clean', '')
LEFT JOIN (
    SELECT source_schema, patient, MIN(DATE_TRUNC('month', created_at)) AS first_ever
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS GROUP BY 1, 2
) atf ON v.patient = atf.patient AND v.source_schema = atf.source_schema
WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
{wh}
GROUP BY 1, 2, 3, 4, 5, 6, 7
ORDER BY 1, 8 DESC
"""
    return run_query(sql)


def load_peak_demand_monthly(filters: dict, run_query) -> pd.DataFrame:
    """Monthly visit volume by type with z-score to flag peak months."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
monthly AS (
    SELECT
        DATE_TRUNC('month', v.created_at)                    AS visit_month,
        COUNT(DISTINCT v.id)                                  AS total_visits,
        COUNT(DISTINCT CASE WHEN adm.visit_id IS NOT NULL THEN v.id END) AS inpatient_visits,
        COUNT(DISTINCT CASE WHEN adm.visit_id IS NULL     THEN v.id END) AS outpatient_visits,
        COUNT(DISTINCT v.patient)                             AS unique_patients,
        COUNT(DISTINCT CASE
            WHEN LOWER(COALESCE(dx.disease_burden_group_1,'')) LIKE '%communicable%'
            THEN v.id END)                                    AS communicable_visits,
        COUNT(DISTINCT CASE WHEN dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1
                            THEN v.id END)                    AS ncd_visits
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS adm
        ON v.id = adm.visit_id
       AND v.source_schema = REPLACE(LOWER(adm.source_schema), '_clean', '')
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
    GROUP BY 1
)
SELECT *,
    AVG(total_visits) OVER ()    AS avg_vol,
    STDDEV(total_visits) OVER () AS sd_vol,
    ROUND(DIV0(total_visits - AVG(total_visits) OVER (),
               NULLIF(STDDEV(total_visits) OVER (), 0)), 2) AS z_score,
    CASE WHEN total_visits > AVG(total_visits) OVER () + 1.0 * STDDEV(total_visits) OVER ()
         THEN 1 ELSE 0 END AS is_peak_month,
    ROUND(DIV0(communicable_visits, NULLIF(total_visits, 0)) * 100, 1) AS communicable_pct,
    ROUND(DIV0(ncd_visits, NULLIF(total_visits, 0)) * 100, 1) AS ncd_pct
FROM monthly
ORDER BY 1
"""
    return run_query(sql)


def load_monthly_volume_anomalies(filters: dict, run_query) -> pd.DataFrame:
    """All months with z-scores and spike/dip flags at 1.0 SD threshold.
    Returns every month so the view can build a full time-series chart and apply
    fallback logic (top-2 / bottom-2) when no months breach the threshold.
    """
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    {wsa}
    GROUP BY source_schema
),
monthly_vol AS (
    SELECT
        DATE_TRUNC('month', v.created_at) AS visit_month,
        COUNT(DISTINCT v.id)              AS total_visits
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
    GROUP BY 1
),
ranked AS (
    SELECT
        visit_month,
        total_visits,
        LAG(total_visits) OVER (ORDER BY visit_month) AS prev_visits,
        ROW_NUMBER() OVER (ORDER BY visit_month)       AS rn
    FROM monthly_vol
),
-- compute avg/sd only on full months (exclude rn=1 which may be a partial start month)
baseline AS (
    SELECT
        ROUND(AVG(total_visits), 0)    AS avg_vol,
        ROUND(STDDEV(total_visits), 0) AS sd_vol
    FROM ranked
    WHERE rn > 1
)
SELECT
    r.visit_month,
    r.total_visits,
    b.avg_vol,
    b.sd_vol,
    ROUND(DIV0(r.total_visits - b.avg_vol, NULLIF(b.sd_vol, 0)), 2) AS z_score,
    CASE
        WHEN r.rn = 1                                        THEN 'Partial'
        WHEN r.total_visits > b.avg_vol + 1.0 * b.sd_vol   THEN 'Spike'
        WHEN r.total_visits < b.avg_vol - 1.0 * b.sd_vol   THEN 'Dip'
        ELSE 'Normal'
    END AS month_type,
    CASE
        WHEN r.prev_visits IS NOT NULL
        THEN ROUND(DIV0(r.total_visits - r.prev_visits, NULLIF(r.prev_visits, 0)) * 100, 1)
        ELSE NULL
    END AS mom_pct,
    CASE WHEN r.rn = 1 THEN 1 ELSE 0 END AS first_in_range
FROM ranked r
CROSS JOIN baseline b
ORDER BY 1
"""
    return run_query(sql)


def load_diagnosis_by_month(filters: dict, run_query) -> pd.DataFrame:
    """All months × diagnosis group × visit type — used to compute baselines
    for the anomalous-month disease mix diagnostic panel."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    {wsa}
    GROUP BY source_schema
),
atf AS (
    SELECT source_schema, patient,
           MIN(DATE_TRUNC('month', created_at)) AS first_month
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    GROUP BY 1, 2
)
SELECT
    DATE_TRUNC('month', v.created_at)                        AS visit_month,
    COALESCE(NULLIF(TRIM(dx.disease_group_1), ''),
        CASE
            WHEN n.diagnosis ILIKE '%malaria%'      THEN 'Malaria'
            WHEN n.diagnosis ILIKE '%urti%'         THEN 'URTI'
            WHEN n.diagnosis ILIKE '%hypertension%' THEN 'Hypertension'
            WHEN n.diagnosis ILIKE '%anc%'          THEN 'ANC / Maternal'
            WHEN n.diagnosis ILIKE '%pneumonia%'    THEN 'Pneumonia'
            ELSE 'Other / Unspecified'
        END)                                                 AS diagnosis_group,
    CASE WHEN adm.visit_id IS NOT NULL
         THEN 'Inpatient' ELSE 'Outpatient' END              AS visit_type,
    COUNT(DISTINCT v.id)                                     AS visit_count,
    COUNT(DISTINCT CASE
        WHEN atf.first_month = DATE_TRUNC('month', v.created_at)
        THEN v.patient END)                                  AS new_patients,
    COUNT(DISTINCT CASE
        WHEN atf.first_month != DATE_TRUNC('month', v.created_at)
        THEN v.patient END)                                  AS returning_patients
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
    ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
    ON v.id = n.visit_id AND v.source_schema = n.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS adm
    ON v.id = adm.visit_id
   AND v.source_schema = REPLACE(LOWER(adm.source_schema), '_clean', '')
LEFT JOIN atf ON v.patient = atf.patient AND v.source_schema = atf.source_schema
WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
{wh}
GROUP BY 1, 2, 3
ORDER BY 1, visit_count DESC
"""
    return run_query(sql)


def load_diagnosis_cost_outliers(filters: dict, run_query) -> pd.DataFrame:
    """Diagnoses ranked by cost-to-volume ratio.
    ratio > 1 means the diagnosis consumes more cost share than its visit share justifies.
    Used as a surface-level flag for the operations view."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    {wsa}
    GROUP BY source_schema
),
visit_costs AS (
    SELECT
        v.id            AS visit_id,
        v.source_schema,
        COALESCE(
            SUM(il.item_amount),
            MAX(a.admission_cost)
        )               AS total_cost
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
        ON v.id = a.visit_id
       AND v.source_schema = REPLACE(LOWER(a.source_schema), '_clean', '')
    LEFT JOIN HOSPITALS.STAGING.STG_INVOICE_LINE_ITEMS il
        ON v.id = il.visit_id AND v.source_schema = il.source_schema
       AND il.invoice_deleted_at IS NULL
       AND (il.auto_cancelled IS NULL OR il.auto_cancelled = 0)
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
    GROUP BY v.id, v.source_schema
),
dx_costs AS (
    SELECT
        COALESCE(NULLIF(TRIM(dx.disease_group_1), ''), 'Other / Unspecified') AS diagnosis_group,
        COUNT(DISTINCT vc.visit_id)   AS visit_count,
        SUM(vc.total_cost)            AS total_cost,
        ROUND(AVG(vc.total_cost), 0)  AS avg_cost_per_case
    FROM visit_costs vc
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON vc.visit_id = dx.visit_id AND vc.source_schema = dx.source_schema
    WHERE vc.total_cost > 0
    GROUP BY 1
),
totals AS (
    SELECT
        SUM(visit_count) AS total_visits,
        SUM(total_cost)  AS grand_total_cost
    FROM dx_costs
)
SELECT
    dc.diagnosis_group,
    dc.visit_count,
    dc.avg_cost_per_case,
    dc.total_cost,
    ROUND(DIV0(dc.visit_count,   t.total_visits)    * 100, 2) AS volume_share_pct,
    ROUND(DIV0(dc.total_cost,    t.grand_total_cost) * 100, 2) AS cost_share_pct,
    ROUND(DIV0(
        DIV0(dc.total_cost,   t.grand_total_cost),
        DIV0(dc.visit_count,  t.total_visits)
    ), 2)                                                       AS cost_volume_ratio
FROM dx_costs dc
CROSS JOIN totals t
WHERE dc.visit_count >= 5
ORDER BY cost_volume_ratio DESC
LIMIT 20
"""
    return run_query(sql)


def load_volume_new_vs_returning(filters: dict, run_query) -> pd.DataFrame:
    """B1: Monthly new vs returning patients — explains who drove volume."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
atf AS (
    SELECT source_schema, patient, MIN(created_at) AS first_ever
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS GROUP BY 1, 2
)
SELECT
    DATE_TRUNC('month', v.created_at)                       AS visit_month,
    COUNT(DISTINCT v.patient)                               AS total_patients,
    COUNT(DISTINCT v.id)                                    AS total_visits,
    COUNT(DISTINCT CASE
        WHEN DATE_TRUNC('month', atf.first_ever)
             = DATE_TRUNC('month', v.created_at)
        THEN v.patient END)                                 AS new_patients,
    COUNT(DISTINCT CASE
        WHEN DATE_TRUNC('month', atf.first_ever)
             != DATE_TRUNC('month', v.created_at)
        THEN v.patient END)                                 AS returning_patients,
    COUNT(DISTINCT CASE WHEN a.visit_id IS NOT NULL
                        THEN v.id END)                      AS inpatient_visits,
    COUNT(DISTINCT CASE WHEN a.visit_id IS NULL
                        THEN v.id END)                      AS outpatient_visits
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
LEFT JOIN atf ON v.patient = atf.patient AND v.source_schema = atf.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    ON v.id = a.visit_id
   AND v.source_schema = REPLACE(LOWER(a.source_schema), '_clean', '')
WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
{wh}
GROUP BY 1
ORDER BY 1
"""
    return run_query(sql)


def load_volume_top_diagnoses(filters: dict, run_query) -> pd.DataFrame:
    """B2: Top diagnoses by month — what brought patients."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
)
SELECT
    DATE_TRUNC('month', v.created_at)                       AS visit_month,
    COALESCE(NULLIF(TRIM(dx.disease_group_1), ''),
             CASE
                 WHEN n.diagnosis ILIKE '%malaria%'      THEN 'Malaria'
                 WHEN n.diagnosis ILIKE '%urti%'         THEN 'URTI'
                 WHEN n.diagnosis ILIKE '%hypertension%' THEN 'Hypertension'
                 WHEN n.diagnosis ILIKE '%diabetes%'     THEN 'Diabetes'
                 WHEN n.diagnosis ILIKE '%anc%'          THEN 'ANC / Maternal'
                 ELSE 'Other / Unspecified'
             END)                                           AS diagnosis_group,
    COUNT(DISTINCT v.id)                                    AS visit_count,
    ROUND(AVG(COALESCE(il_sum.visit_cost, 0)), 0)           AS avg_visit_cost
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
    ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
    ON v.id = n.visit_id AND v.source_schema = n.source_schema
LEFT JOIN (
    SELECT visit_id, source_schema, SUM(item_amount) AS visit_cost
    FROM HOSPITALS.STAGING.STG_INVOICE_LINE_ITEMS
    WHERE invoice_deleted_at IS NULL
      AND (auto_cancelled IS NULL OR auto_cancelled = 0)
    GROUP BY 1, 2
) il_sum ON v.id = il_sum.visit_id AND v.source_schema = il_sum.source_schema
WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
{wh}
GROUP BY 1, 2
ORDER BY 1, 3 DESC
"""
    return run_query(sql)


def load_journey_times(filters: dict, run_query) -> pd.DataFrame:
    """Q3: Patient journey times — averages + P50/P75/P90 percentiles per stage."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    {wsa}
    GROUP BY source_schema
),
visit_base AS (
    SELECT
        v.source_schema, v.clinic AS facility, v.id AS visit_id,
        DATE_TRUNC('month', v.created_at) AS visit_month,
        CASE WHEN a.visit_id IS NOT NULL THEN 'Inpatient' ELSE 'Outpatient' END AS visit_type,
        COALESCE(a.admitted_at, v.created_at) AS journey_start
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
        ON v.id = a.visit_id
       AND v.source_schema = REPLACE(LOWER(a.source_schema), '_clean', '')
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
),
vitals_first AS (
    SELECT visit_id, source_schema, MIN(created_at) AS first_vitals
    FROM HOSPITALS.STAGING.STG_EVALUATION_VITALS GROUP BY 1, 2
),
notes_first AS (
    SELECT visit_id, source_schema, MIN(created_at) AS first_note
    FROM HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES GROUP BY 1, 2
),
lab_inv AS (
    SELECT visit_id, source_schema,
           MIN(investigation_created_at) AS ordered_at,
           MIN(result_created_at)        AS first_result_at
    FROM HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS
    WHERE (cancelled IS NULL OR cancelled = 0)
      AND LOWER(TRIM(investigation_type)) IN ('laboratory', 'lab')
    GROUP BY 1, 2
),
journey_durations AS (
    SELECT
        vb.visit_type,
        vb.visit_id,
        CASE
            WHEN vt.first_vitals > vb.journey_start
             AND DATEDIFF('hour', vb.journey_start, vt.first_vitals)
                 < IFF(vb.visit_type='Inpatient', 48, 12)
            THEN DATEDIFF('minute', vb.journey_start, vt.first_vitals) / 60.0
        END AS hrs_to_triage,
        CASE
            WHEN n.first_note > vt.first_vitals
             AND DATEDIFF('hour', vt.first_vitals, n.first_note)
                 < IFF(vb.visit_type='Inpatient', 48, 12)
            THEN DATEDIFF('minute', vt.first_vitals, n.first_note) / 60.0
        END AS hrs_triage_to_consult,
        CASE
            WHEN li.first_result_at > n.first_note
             AND DATEDIFF('hour', n.first_note, li.first_result_at)
                 < IFF(vb.visit_type='Inpatient', 48, 12)
            THEN DATEDIFF('minute', n.first_note, li.first_result_at) / 60.0
        END AS hrs_consult_to_lab,
        CASE
            WHEN li.first_result_at > li.ordered_at
             AND DATEDIFF('hour', li.ordered_at, li.first_result_at)
                 < IFF(vb.visit_type='Inpatient', 48, 12)
            THEN DATEDIFF('minute', li.ordered_at, li.first_result_at) / 60.0
        END AS hrs_lab_turnaround
    FROM visit_base vb
    LEFT JOIN vitals_first vt ON vb.visit_id = vt.visit_id AND vb.source_schema = vt.source_schema
    LEFT JOIN notes_first   n ON vb.visit_id = n.visit_id  AND vb.source_schema = n.source_schema
    LEFT JOIN lab_inv       li ON vb.visit_id = li.visit_id AND vb.source_schema = li.source_schema
)
SELECT
    visit_type,
    -- Averages (kept for backward compatibility)
    ROUND(AVG(hrs_to_triage), 2)          AS avg_hrs_to_triage,
    ROUND(AVG(hrs_triage_to_consult), 2)  AS avg_hrs_triage_to_consult,
    ROUND(AVG(hrs_consult_to_lab), 2)     AS avg_hrs_consult_to_lab,
    ROUND(AVG(hrs_lab_turnaround), 2)     AS avg_hrs_lab_turnaround,
    -- Arrival → Triage percentiles (target: 15 min = 0.25 h)
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY hrs_to_triage), 2)  AS p50_hrs_to_triage,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY hrs_to_triage), 2)  AS p75_hrs_to_triage,
    ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY hrs_to_triage), 2)  AS p90_hrs_to_triage,
    ROUND(100.0 * COUNT(CASE WHEN hrs_to_triage > 0.25 THEN 1 END)
          / NULLIF(COUNT(hrs_to_triage), 0), 1)                            AS pct_exceed_triage,
    -- Triage → Consult percentiles (target: 1 h)
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY hrs_triage_to_consult), 2) AS p50_hrs_triage_to_consult,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY hrs_triage_to_consult), 2) AS p75_hrs_triage_to_consult,
    ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY hrs_triage_to_consult), 2) AS p90_hrs_triage_to_consult,
    ROUND(100.0 * COUNT(CASE WHEN hrs_triage_to_consult > 1.0 THEN 1 END)
          / NULLIF(COUNT(hrs_triage_to_consult), 0), 1)                    AS pct_exceed_consult,
    -- Consult → Lab Result percentiles (often not captured — may be null)
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY hrs_consult_to_lab), 2)    AS p50_hrs_consult_to_lab,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY hrs_consult_to_lab), 2)    AS p75_hrs_consult_to_lab,
    ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY hrs_consult_to_lab), 2)    AS p90_hrs_consult_to_lab,
    -- Lab Result → Discharge percentiles (target: 4 h)
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY hrs_lab_turnaround), 2)    AS p50_hrs_lab_turnaround,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY hrs_lab_turnaround), 2)    AS p75_hrs_lab_turnaround,
    ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY hrs_lab_turnaround), 2)    AS p90_hrs_lab_turnaround,
    ROUND(100.0 * COUNT(CASE WHEN hrs_lab_turnaround > 4.0 THEN 1 END)
          / NULLIF(COUNT(hrs_lab_turnaround), 0), 1)                       AS pct_exceed_lab,
    -- Coverage rates
    ROUND(100.0 * COUNT(hrs_to_triage)         / NULLIF(COUNT(*), 0), 1)  AS pct_triage_recorded,
    ROUND(100.0 * COUNT(hrs_triage_to_consult) / NULLIF(COUNT(*), 0), 1)  AS pct_consult_recorded,
    COUNT(*) AS total_visits
FROM journey_durations
GROUP BY 1
ORDER BY 1
"""
    return run_query(sql)


def load_investigation_columns(run_query) -> list:
    """Return lowercase column names available in STG_EVALUATION_INVESTIGATIONS."""
    sql = """
SELECT LOWER(COLUMN_NAME) AS col
FROM HOSPITALS.INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = 'STAGING'
  AND TABLE_NAME   = 'STG_EVALUATION_INVESTIGATIONS'
ORDER BY ORDINAL_POSITION
"""
    try:
        df = run_query(sql)
        return df["col"].tolist() if not df.empty else []
    except Exception:
        return []


def load_lab_turnaround_by_discipline(filters: dict, run_query, name_col: str = "") -> pd.DataFrame:
    """Lab/investigation turnaround by clinical discipline.

    If `name_col` is a column that holds procedure-level detail (e.g. 'name',
    'description', 'investigation_name'), it is used to map tests to clinical
    disciplines via keyword matching.  Falls back to investigation_type grouping
    when no detailed column is available.
    """
    wh = _w(filters, alias="v")
    wsa = _wsa(filters)
    mo = _mo(filters)

    if name_col:
        raw_expr = f"UPPER(TRIM(i.{name_col}))"
    else:
        raw_expr = "UPPER(TRIM(i.investigation_type))"

    discipline_case = f"""
        CASE
            -- Haematology
            WHEN {raw_expr} LIKE '%HAEMATOL%' OR {raw_expr} LIKE '%HEMATOL%'
              OR {raw_expr} LIKE '%FULL BLOOD%' OR {raw_expr} LIKE '%CBC%'
              OR {raw_expr} LIKE '%HAEMOGLOBIN%' OR {raw_expr} LIKE '%HEMOGLOBIN%'
              OR {raw_expr} LIKE '%BLOOD GROUP%' OR {raw_expr} LIKE '%BLOOD FILM%'
              OR {raw_expr} LIKE '%ESR%' OR {raw_expr} LIKE '%PLATELET%'
              OR {raw_expr} LIKE '%WHITE CELL%' OR {raw_expr} LIKE '%WBC%'
              OR {raw_expr} LIKE '%RBC%' OR {raw_expr} LIKE '%COAGUL%'
              OR {raw_expr} LIKE '%PT%INR%' OR {raw_expr} LIKE '%APTT%'
                THEN 'Haematology'
            -- Clinical Chemistry / Biochemistry
            WHEN {raw_expr} LIKE '%CHEM%' OR {raw_expr} LIKE '%BIOCHEM%'
              OR {raw_expr} LIKE '%GLUCOSE%' OR {raw_expr} LIKE '%RBS%'
              OR {raw_expr} LIKE '%FBS%' OR {raw_expr} LIKE '%HBA1C%'
              OR {raw_expr} LIKE '%LIPID%' OR {raw_expr} LIKE '%CHOLESTEROL%'
              OR {raw_expr} LIKE '%CREATININE%' OR {raw_expr} LIKE '%UREA%'
              OR {raw_expr} LIKE '%BUN%' OR {raw_expr} LIKE '%ELECTROLYTE%'
              OR {raw_expr} LIKE '%SODIUM%' OR {raw_expr} LIKE '%POTASSIUM%'
              OR {raw_expr} LIKE '%LIVER%' OR {raw_expr} LIKE '%LFT%'
              OR {raw_expr} LIKE '%ALT%' OR {raw_expr} LIKE '%AST%'
              OR {raw_expr} LIKE '%TSH%' OR {raw_expr} LIKE '%THYROID%'
              OR {raw_expr} LIKE '%PSA%' OR {raw_expr} LIKE '%URIC ACID%'
              OR {raw_expr} LIKE '%AMYLASE%' OR {raw_expr} LIKE '%TROPONIN%'
                THEN 'Clinical Chemistry'
            -- Microbiology
            WHEN {raw_expr} LIKE '%MICROBIO%' OR {raw_expr} LIKE '%CULTURE%'
              OR {raw_expr} LIKE '%SENSITIV%' OR {raw_expr} LIKE '%C&S%'
              OR {raw_expr} LIKE '%AFB%' OR {raw_expr} LIKE '%TB%'
              OR {raw_expr} LIKE '%MALARIA%' OR {raw_expr} LIKE '%GRAM STAIN%'
              OR {raw_expr} LIKE '%MRSA%' OR {raw_expr} LIKE '%GeneXpert%'
              OR {raw_expr} LIKE '%XPERT%' OR {raw_expr} LIKE '%STOOL%'
              OR {raw_expr} LIKE '%SWAB%'
                THEN 'Microbiology'
            -- Immunology / Serology
            WHEN {raw_expr} LIKE '%IMMUNOL%' OR {raw_expr} LIKE '%SEROL%'
              OR {raw_expr} LIKE '%HIV%' OR {raw_expr} LIKE '%HEPATITIS%'
              OR {raw_expr} LIKE '%HBsAg%' OR {raw_expr} LIKE '%HCV%'
              OR {raw_expr} LIKE '%SYPHILIS%' OR {raw_expr} LIKE '%VDRL%'
              OR {raw_expr} LIKE '%RPR%' OR {raw_expr} LIKE '%BRUCELLA%'
              OR {raw_expr} LIKE '%WIDAL%' OR {raw_expr} LIKE '%ELISA%'
              OR {raw_expr} LIKE '%RAPID TEST%' OR {raw_expr} LIKE '%ANTIGEN%'
              OR {raw_expr} LIKE '%ANTIBODY%' OR {raw_expr} LIKE '%COVID%'
                THEN 'Immunology / Serology'
            -- Urinalysis
            WHEN {raw_expr} LIKE '%URINAL%' OR {raw_expr} LIKE '%URINE%'
              OR {raw_expr} LIKE '%DIPSTICK%' OR {raw_expr} LIKE '%URINE M%'
                THEN 'Urinalysis'
            -- Pathology / Cytology / Histology
            WHEN {raw_expr} LIKE '%PATHOL%' OR {raw_expr} LIKE '%HISTOL%'
              OR {raw_expr} LIKE '%BIOPSY%' OR {raw_expr} LIKE '%CYTOL%'
              OR {raw_expr} LIKE '%PAP%' OR {raw_expr} LIKE '%FNAC%'
              OR {raw_expr} LIKE '%SMEAR%'
                THEN 'Pathology & Cytology'
            -- Radiology / Imaging
            WHEN {raw_expr} LIKE '%RADIOL%' OR {raw_expr} LIKE '%IMAGING%'
              OR {raw_expr} LIKE '%X-RAY%' OR {raw_expr} LIKE '%XRAY%'
              OR {raw_expr} LIKE '%ULTRASOUND%' OR {raw_expr} LIKE '%ECHO%'
              OR {raw_expr} LIKE '%CT SCAN%' OR {raw_expr} LIKE '%MRI%'
              OR {raw_expr} LIKE '%MAMMOGRAM%' OR {raw_expr} LIKE '%SCAN%'
                THEN 'Radiology & Imaging'
            ELSE 'Other / Unclassified'
        END"""

    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    {wsa}
    GROUP BY source_schema
)
SELECT
    {discipline_case}                                                AS discipline,
    INITCAP(TRIM(i.investigation_type))                             AS investigation_type,
    COUNT(*)                                                        AS test_count,
    ROUND(AVG(CASE
        WHEN i.result_created_at > i.investigation_created_at
         AND DATEDIFF('hour', i.investigation_created_at, i.result_created_at) <= 72
        THEN DATEDIFF('minute', i.investigation_created_at, i.result_created_at) / 60.0
    END), 2)                                                        AS avg_turnaround_hrs,
    ROUND(MEDIAN(CASE
        WHEN i.result_created_at > i.investigation_created_at
         AND DATEDIFF('hour', i.investigation_created_at, i.result_created_at) <= 72
        THEN DATEDIFF('minute', i.investigation_created_at, i.result_created_at) / 60.0
    END), 2)                                                        AS median_turnaround_hrs,
    ROUND(DIV0(
        COUNT(CASE WHEN i.result_created_at IS NOT NULL THEN 1 END),
        COUNT(*)
    ) * 100, 1)                                                     AS result_rate_pct
FROM HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS i
INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    ON i.visit_id = v.id
INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
WHERE (i.cancelled IS NULL OR i.cancelled = 0)
  AND (i.remove_from_report IS NULL OR i.remove_from_report = 0)
  AND i.investigation_type IS NOT NULL
  AND TRIM(i.investigation_type) != ''
  AND v.created_at >= DATEADD('month', -{mo}, sa.max_date)
  {wh}
GROUP BY 1, 2
HAVING test_count >= 3
ORDER BY discipline, avg_turnaround_hrs DESC NULLS LAST
"""
    return run_query(sql)


def load_lab_turnaround_by_test(filters: dict, run_query) -> pd.DataFrame:
    """Lab/investigation turnaround broken down by investigation_type (legacy fallback)."""
    wh = _w(filters, alias="v")
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    {wsa}
    GROUP BY source_schema
)
SELECT
    INITCAP(TRIM(i.investigation_type))                             AS test_type,
    COUNT(*)                                                        AS test_count,
    ROUND(AVG(CASE
        WHEN i.result_created_at > i.investigation_created_at
         AND DATEDIFF('hour', i.investigation_created_at, i.result_created_at) <= 72
        THEN DATEDIFF('minute', i.investigation_created_at, i.result_created_at) / 60.0
    END), 2)                                                        AS avg_turnaround_hrs,
    ROUND(MEDIAN(CASE
        WHEN i.result_created_at > i.investigation_created_at
         AND DATEDIFF('hour', i.investigation_created_at, i.result_created_at) <= 72
        THEN DATEDIFF('minute', i.investigation_created_at, i.result_created_at) / 60.0
    END), 2)                                                        AS median_turnaround_hrs,
    ROUND(DIV0(
        COUNT(CASE WHEN i.result_created_at IS NOT NULL THEN 1 END),
        COUNT(*)
    ) * 100, 1)                                                     AS result_rate_pct
FROM HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS i
INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    ON i.visit_id = v.id
INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
WHERE (i.cancelled IS NULL OR i.cancelled = 0)
  AND (i.remove_from_report IS NULL OR i.remove_from_report = 0)
  AND TRIM(i.investigation_type) != ''
  AND i.investigation_type IS NOT NULL
  AND v.created_at >= DATEADD('month', -{mo}, sa.max_date)
  {wh}
GROUP BY 1
HAVING test_count >= 5
ORDER BY avg_turnaround_hrs DESC NULLS LAST
"""
    return run_query(sql)


def load_inpatient_funnel(filters: dict, run_query) -> pd.DataFrame:
    """Q3C: Inpatient conversion funnel — five gates with drop-off counts."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    {wsa}
    GROUP BY source_schema
),
visit_base AS (
    SELECT v.source_schema, v.clinic AS facility, v.id AS visit_id, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
),
note_recorded AS (
    SELECT DISTINCT visit_id, source_schema
    FROM HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES
    WHERE diagnosis IS NOT NULL
      AND TRIM(diagnosis) NOT IN ('', '[]', 'null', '{{}}')
),
admitted AS (
    SELECT DISTINCT a.visit_id,
           REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema,
           a.los_days, a.admission_cost
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
),
inv_ordered AS (
    SELECT DISTINCT visit_id, source_schema
    FROM HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS
    WHERE (cancelled IS NULL OR cancelled = 0)
      AND (remove_from_report IS NULL OR remove_from_report = 0)
      AND LOWER(TRIM(investigation_type))
          IN ('laboratory', 'lab', 'radiology', 'ultrasound')
),
inv_resulted_24h AS (
    SELECT DISTINCT visit_id, source_schema
    FROM HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS
    WHERE (cancelled IS NULL OR cancelled = 0)
      AND (remove_from_report IS NULL OR remove_from_report = 0)
      AND LOWER(TRIM(investigation_type))
          IN ('laboratory', 'lab', 'radiology', 'ultrasound')
      AND result_created_at IS NOT NULL
      AND DATEDIFF('hour', investigation_created_at, result_created_at) <= 24
      AND result_created_at > investigation_created_at
)
SELECT
    COUNT(DISTINCT vb.visit_id)                             AS g1_total_visits,
    COUNT(DISTINCT CASE WHEN nr.visit_id IS NOT NULL
                        THEN vb.visit_id END)               AS g2_note_recorded,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN nr.visit_id IS NOT NULL THEN vb.visit_id END),
        COUNT(DISTINCT vb.visit_id)
    ) * 100, 1)                                             AS g2_note_coverage_pct,
    COUNT(DISTINCT CASE WHEN a.visit_id IS NOT NULL
                        THEN vb.visit_id END)               AS g3_admitted,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN a.visit_id IS NOT NULL
                             AND nr.visit_id IS NOT NULL THEN vb.visit_id END),
        COUNT(DISTINCT CASE WHEN nr.visit_id IS NOT NULL THEN vb.visit_id END)
    ) * 100, 1)                                             AS g3_consult_to_admit_pct,
    COUNT(DISTINCT CASE WHEN a.visit_id  IS NOT NULL
                         AND io.visit_id IS NOT NULL
                        THEN vb.visit_id END)               AS g4_with_investigation,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN a.visit_id  IS NOT NULL
                             AND io.visit_id IS NOT NULL THEN vb.visit_id END),
        COUNT(DISTINCT CASE WHEN a.visit_id  IS NOT NULL THEN vb.visit_id END)
    ) * 100, 1)                                             AS g4_investigation_rate_pct,
    COUNT(DISTINCT CASE WHEN a.visit_id  IS NOT NULL
                         AND io.visit_id IS NOT NULL
                         AND ir.visit_id IS NOT NULL
                        THEN vb.visit_id END)               AS g5_resulted_24h,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN a.visit_id  IS NOT NULL
                             AND io.visit_id IS NOT NULL
                             AND ir.visit_id IS NOT NULL THEN vb.visit_id END),
        COUNT(DISTINCT CASE WHEN a.visit_id  IS NOT NULL
                             AND io.visit_id IS NOT NULL THEN vb.visit_id END)
    ) * 100, 1)                                             AS g5_result_24h_pct,
    ROUND(AVG(CASE WHEN a.visit_id IS NOT NULL THEN a.admission_cost END), 0)
                                                            AS avg_admission_cost,
    COUNT(DISTINCT CASE WHEN nr.visit_id IS NOT NULL
                         AND a.visit_id IS NULL
                        THEN vb.visit_id END)               AS dropoff_consult_not_admitted,
    COUNT(DISTINCT CASE WHEN a.visit_id  IS NOT NULL
                         AND io.visit_id IS NULL
                        THEN vb.visit_id END)               AS dropoff_no_investigation,
    COUNT(DISTINCT CASE WHEN a.visit_id  IS NOT NULL
                         AND io.visit_id IS NOT NULL
                         AND ir.visit_id IS NULL
                        THEN vb.visit_id END)               AS dropoff_investigation_delayed
FROM visit_base vb
LEFT JOIN note_recorded    nr ON vb.visit_id = nr.visit_id AND vb.source_schema = nr.source_schema
LEFT JOIN admitted          a ON vb.visit_id = a.visit_id  AND vb.source_schema = a.source_schema
LEFT JOIN inv_ordered      io ON vb.visit_id = io.visit_id AND vb.source_schema = io.source_schema
LEFT JOIN inv_resulted_24h ir ON vb.visit_id = ir.visit_id AND vb.source_schema = ir.source_schema
"""
    return run_query(sql)


def load_encounter_forecast(filters: dict, run_query) -> pd.DataFrame:
    """Q4: Encounter forecast — actuals + future projection with confidence band, split by type."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    {wsa}
    GROUP BY source_schema
),
monthly AS (
    SELECT
        DATE_TRUNC('month', v.created_at)                   AS visit_month,
        COUNT(DISTINCT v.id)                                 AS total_visits,
        COUNT(DISTINCT CASE WHEN a.visit_id IS NOT NULL
                            THEN v.id END)                  AS inpatient_visits,
        COUNT(DISTINCT CASE WHEN a.visit_id IS NULL
                            THEN v.id END)                  AS outpatient_visits
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
        ON v.id = a.visit_id
       AND v.source_schema = REPLACE(LOWER(a.source_schema), '_clean', '')
    WHERE v.created_at >= DATEADD('month', -{mo + 3}, sa.max_date)
    {wh}
    GROUP BY 1
),
with_lags AS (
    SELECT *,
        LAG(total_visits, 1)  OVER (ORDER BY visit_month) AS m1,
        LAG(total_visits, 2)  OVER (ORDER BY visit_month) AS m2,
        LAG(total_visits, 3)  OVER (ORDER BY visit_month) AS m3,
        LAG(total_visits, 12) OVER (ORDER BY visit_month) AS yoy,
        STDDEV(total_visits) OVER ()                       AS vol_stddev,
        MAX(visit_month) OVER ()                           AS last_actual_month
    FROM monthly
),
forecast AS (
    SELECT *,
        ROUND(DIV0(
            3 * COALESCE(m1, 0) + 2 * COALESCE(m2, 0) + COALESCE(m3, 0),
            3 * IFF(m1 IS NOT NULL, 1, 0)
          + 2 * IFF(m2 IS NOT NULL, 1, 0)
          +     IFF(m3 IS NOT NULL, 1, 0)
        ), 0)                                              AS trend_component,
        CASE WHEN yoy IS NOT NULL
             THEN ROUND(DIV0(yoy, NULLIF(ROUND(DIV0(
                 3 * COALESCE(m1, 0) + 2 * COALESCE(m2, 0) + COALESCE(m3, 0),
                 3 * IFF(m1 IS NOT NULL, 1, 0)
               + 2 * IFF(m2 IS NOT NULL, 1, 0)
               +     IFF(m3 IS NOT NULL, 1, 0)
             ), 0), 0)), 2)
             ELSE 1.0
        END                                               AS seasonal_index
    FROM with_lags
    WHERE m1 IS NOT NULL
)
SELECT
    visit_month,
    total_visits                                            AS actual_visits,
    inpatient_visits                                        AS actual_inpatient,
    outpatient_visits                                       AS actual_outpatient,
    ROUND(trend_component * seasonal_index, 0)             AS forecast_total,
    ROUND(trend_component * seasonal_index
          - 1.64 * COALESCE(vol_stddev, 0), 0)            AS forecast_low,
    ROUND(trend_component * seasonal_index
          + 1.64 * COALESCE(vol_stddev, 0), 0)            AS forecast_high,
    last_actual_month,
    CASE WHEN visit_month = last_actual_month THEN 1 ELSE 0 END AS is_last_actual
FROM forecast
ORDER BY visit_month
"""
    return run_query(sql)


def load_clinician_load(filters: dict, run_query) -> pd.DataFrame:
    """Q5: Clinician load — visits, patients, vitals/notes rate split by new vs returning."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    {wsa}
    GROUP BY source_schema
),
atf AS (
    SELECT source_schema, patient, MIN(created_at) AS first_ever
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    GROUP BY 1, 2
),
base AS (
    SELECT
        v.user                                              AS clinician,
        DATE_TRUNC('day', v.created_at)                    AS visit_date,
        v.id                                                AS visit_id,
        v.patient,
        v.source_schema,
        CASE WHEN atf.first_ever >= DATEADD('month', -{mo}, sa.max_date)
             THEN 1 ELSE 0 END                              AS is_new_patient,
        CASE WHEN ev.visit_id IS NOT NULL THEN 1 ELSE 0 END AS has_vitals,
        CASE WHEN dn.visit_id IS NOT NULL THEN 1 ELSE 0 END AS has_notes
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN atf ON v.patient = atf.patient AND v.source_schema = atf.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_VITALS ev
        ON v.id = ev.visit_id AND v.source_schema = ev.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES dn
        ON v.id = dn.visit_id AND v.source_schema = dn.source_schema
    WHERE v.user IS NOT NULL
      AND v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
)
SELECT
    clinician,
    COUNT(DISTINCT visit_date)                                          AS days_worked,
    COUNT(DISTINCT visit_id)                                            AS total_visits,
    ROUND(COUNT(DISTINCT visit_id)::FLOAT / NULLIF(COUNT(DISTINCT visit_date), 0), 1)
                                                                        AS avg_daily_patients,
    SUM(is_new_patient)                                                 AS new_visits,
    COUNT(DISTINCT visit_id) - SUM(is_new_patient)                     AS returning_visits,
    ROUND(DIV0(SUM(is_new_patient), COUNT(DISTINCT visit_id)) * 100, 1) AS new_visit_pct,
    ROUND(DIV0(SUM(has_vitals), COUNT(DISTINCT visit_id)) * 100, 1)    AS vitals_rate_pct,
    ROUND(DIV0(SUM(has_notes),  COUNT(DISTINCT visit_id)) * 100, 1)    AS notes_rate_pct,
    ROUND(DIV0(
        SUM(CASE WHEN is_new_patient = 1 THEN has_vitals ELSE 0 END),
        NULLIF(SUM(is_new_patient), 0)
    ) * 100, 1)                                                         AS vitals_rate_new_pct,
    ROUND(DIV0(
        SUM(CASE WHEN is_new_patient = 0 THEN has_vitals ELSE 0 END),
        NULLIF(COUNT(DISTINCT visit_id) - SUM(is_new_patient), 0)
    ) * 100, 1)                                                         AS vitals_rate_returning_pct
FROM base
GROUP BY 1
HAVING days_worked >= 5
ORDER BY avg_daily_patients DESC
LIMIT 20
"""
    return run_query(sql)


def load_night_ae_conversion(filters: dict, run_query) -> pd.DataFrame:
    """G1: Night-shift A&E visits → morning admission conversion.
    Night shift = 20:00–06:59 EAT. Morning admit = 06:00–11:59 EAT.
    Returns one row per shift label with conversion rate, payer, surgery flag."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
visits AS (
    SELECT
        v.id              AS visit_id,
        v.source_schema,
        v.patient,
        v.created_at,
        v.payment_mode,
        HOUR(CONVERT_TIMEZONE('UTC', 'Africa/Nairobi', v.created_at)) AS visit_hour_eat,
        CASE
            WHEN HOUR(CONVERT_TIMEZONE('UTC', 'Africa/Nairobi', v.created_at)) >= 20
              OR HOUR(CONVERT_TIMEZONE('UTC', 'Africa/Nairobi', v.created_at)) <= 6
            THEN 'Night (8 pm – 7 am)'
            ELSE 'Day (7 am – 8 pm)'
        END AS shift
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
),
admissions AS (
    SELECT
        a.visit_id,
        REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema,
        a.admitted_at,
        HOUR(CONVERT_TIMEZONE('UTC', 'Africa/Nairobi', a.admitted_at)) AS admit_hour_eat,
        CASE
            WHEN HOUR(CONVERT_TIMEZONE('UTC', 'Africa/Nairobi', a.admitted_at)) BETWEEN 6 AND 11
            THEN 1 ELSE 0
        END AS is_morning_admit
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    INNER JOIN schema_anchor sa
        ON REPLACE(LOWER(a.source_schema), '_clean', '') = sa.source_schema
    WHERE a.admitted_at >= DATEADD('month', -{mo}, sa.max_date)
),
dx AS (
    SELECT DISTINCT visit_id, source_schema,
        CASE WHEN LOWER(disease_group_1) LIKE '%surg%'
              OR LOWER(disease_group_1) LIKE '%operat%'
              OR LOWER(disease_group_1) LIKE '%trauma%'
             THEN 1 ELSE 0 END AS is_surgery
    FROM HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED
)
SELECT
    v.shift,
    COUNT(DISTINCT v.visit_id)                                          AS total_visits,
    COUNT(DISTINCT a.visit_id)                                          AS admitted,
    ROUND(DIV0(COUNT(DISTINCT a.visit_id),
               COUNT(DISTINCT v.visit_id)) * 100, 1)                    AS conversion_rate_pct,
    ROUND(DIV0(SUM(a.is_morning_admit),
               COUNT(DISTINCT a.visit_id)) * 100, 1)                    AS morning_admit_pct,
    ROUND(AVG(CASE
        WHEN a.admitted_at > v.created_at
         AND DATEDIFF('hour', v.created_at, a.admitted_at) <= 72
        THEN DATEDIFF('minute', v.created_at, a.admitted_at) / 60.0
    END), 1)                                                             AS avg_wait_to_admit_hrs,
    ROUND(DIV0(
        COUNT(DISTINCT CASE
            WHEN a.visit_id IS NOT NULL
             AND LOWER(COALESCE(v.payment_mode, '')) NOT IN ('cash','self-pay','out-of-pocket','copay')
             AND TRIM(COALESCE(v.payment_mode, '')) != ''
            THEN v.visit_id END),
        COUNT(DISTINCT a.visit_id)
    ) * 100, 1)                                                          AS insurance_pct,
    ROUND(DIV0(
        COUNT(DISTINCT CASE
            WHEN a.visit_id IS NOT NULL AND dx.is_surgery = 1
            THEN v.visit_id END),
        COUNT(DISTINCT a.visit_id)
    ) * 100, 1)                                                          AS surgery_pct
FROM visits v
LEFT JOIN admissions a
    ON v.visit_id = a.visit_id AND v.source_schema = a.source_schema
LEFT JOIN dx
    ON v.visit_id = dx.visit_id AND v.source_schema = dx.source_schema
GROUP BY 1
ORDER BY 1
"""
    return run_query(sql)


def load_peak_day_service_times(filters: dict, run_query) -> pd.DataFrame:
    """G2: Triage / consult / investigation wait times bucketed by daily load tier.
    Peak = top 25% of daily visit volume. Quiet = bottom 25%. Normal = middle 50%."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
daily_vol AS (
    SELECT
        DATE_TRUNC('day', v.created_at)   AS visit_day,
        v.source_schema,
        COUNT(DISTINCT v.id)              AS day_visits,
        COUNT(DISTINCT v.user)            AS day_clinicians
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
    GROUP BY 1, 2
),
day_stats AS (
    SELECT
        source_schema,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY day_visits) AS p25,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY day_visits) AS p75
    FROM daily_vol GROUP BY 1
),
day_tier AS (
    SELECT
        dv.visit_day, dv.source_schema, dv.day_visits, dv.day_clinicians,
        CASE
            WHEN dv.day_visits >= ds.p75 THEN 'Peak (top 25%)'
            WHEN dv.day_visits <= ds.p25 THEN 'Quiet (bottom 25%)'
            ELSE 'Normal'
        END AS load_tier
    FROM daily_vol dv
    INNER JOIN day_stats ds ON dv.source_schema = ds.source_schema
),
vitals_first AS (
    SELECT visit_id, source_schema, MIN(created_at) AS first_vitals
    FROM HOSPITALS.STAGING.STG_EVALUATION_VITALS GROUP BY 1, 2
),
notes_first AS (
    SELECT visit_id, source_schema, MIN(created_at) AS first_note
    FROM HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES GROUP BY 1, 2
),
inv_first AS (
    SELECT visit_id, source_schema, MIN(investigation_created_at) AS first_inv
    FROM HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS
    WHERE (cancelled IS NULL OR cancelled = 0)
    GROUP BY 1, 2
),
rx_first AS (
    SELECT visit_id, source_schema, MIN(prescription_created_at) AS first_rx
    FROM HOSPITALS.STAGING.STG_PRESCRIPTION_PAYMENTS
    WHERE (remove_from_report IS NULL OR remove_from_report = 0)
    GROUP BY 1, 2
),
visit_waits AS (
    SELECT
        dt.load_tier,
        dt.day_visits,
        dt.day_clinicians,
        CASE
            WHEN vf.first_vitals IS NOT NULL
             AND DATEDIFF('minute', v.created_at, vf.first_vitals) BETWEEN 0 AND 720
            THEN DATEDIFF('minute', v.created_at, vf.first_vitals)
        END AS mins_to_triage,
        CASE
            WHEN nf.first_note IS NOT NULL
             AND DATEDIFF('minute', v.created_at, nf.first_note) BETWEEN 0 AND 720
            THEN DATEDIFF('minute', v.created_at, nf.first_note)
        END AS mins_to_consult,
        CASE
            WHEN inf.first_inv IS NOT NULL
             AND DATEDIFF('minute', v.created_at, inf.first_inv) BETWEEN 0 AND 1440
            THEN DATEDIFF('minute', v.created_at, inf.first_inv)
        END AS mins_to_inv,
        CASE
            WHEN rf.first_rx IS NOT NULL
             AND DATEDIFF('minute', v.created_at, rf.first_rx) BETWEEN 0 AND 1440
            THEN DATEDIFF('minute', v.created_at, rf.first_rx)
        END AS mins_to_rx,
        CASE WHEN vf.first_vitals IS NOT NULL THEN 1 ELSE 0 END AS had_vitals,
        CASE WHEN nf.first_note   IS NOT NULL THEN 1 ELSE 0 END AS had_note,
        CASE WHEN inf.first_inv   IS NOT NULL THEN 1 ELSE 0 END AS had_inv,
        CASE WHEN rf.first_rx     IS NOT NULL THEN 1 ELSE 0 END AS had_rx
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN day_tier dt
        ON DATE_TRUNC('day', v.created_at) = dt.visit_day
       AND v.source_schema = dt.source_schema
    LEFT JOIN vitals_first vf ON v.id = vf.visit_id AND v.source_schema = vf.source_schema
    LEFT JOIN notes_first  nf ON v.id = nf.visit_id AND v.source_schema = nf.source_schema
    LEFT JOIN inv_first   inf ON v.id = inf.visit_id AND v.source_schema = inf.source_schema
    LEFT JOIN rx_first     rf ON v.id = rf.visit_id  AND v.source_schema = rf.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
)
SELECT
    load_tier,
    COUNT(*)                                                         AS total_visits,
    COUNT(DISTINCT CASE WHEN mins_to_triage  IS NOT NULL THEN 1 END) AS visits_with_triage,
    ROUND(MEDIAN(mins_to_triage), 0)                                 AS median_mins_to_triage,
    ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP
          (ORDER BY mins_to_triage), 0)                              AS p90_mins_to_triage,
    ROUND(MEDIAN(mins_to_consult), 0)                                AS median_mins_to_consult,
    ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP
          (ORDER BY mins_to_consult), 0)                             AS p90_mins_to_consult,
    ROUND(MEDIAN(mins_to_inv), 0)                                    AS median_mins_to_inv,
    ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP
          (ORDER BY mins_to_inv), 0)                                 AS p90_mins_to_inv,
    ROUND(MEDIAN(mins_to_rx), 0)                                     AS median_mins_to_rx,
    ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP
          (ORDER BY mins_to_rx), 0)                                  AS p90_mins_to_rx,
    ROUND(DIV0(SUM(had_vitals), COUNT(*)) * 100, 1)                  AS vitals_rate_pct,
    ROUND(DIV0(SUM(had_note),   COUNT(*)) * 100, 1)                  AS notes_rate_pct,
    ROUND(DIV0(SUM(had_inv),    COUNT(*)) * 100, 1)                  AS inv_rate_pct,
    ROUND(DIV0(SUM(had_rx),     COUNT(*)) * 100, 1)                  AS rx_rate_pct,
    ROUND(AVG(day_clinicians), 1)                                    AS avg_clinicians_on_day,
    ROUND(AVG(day_visits::FLOAT / NULLIF(day_clinicians, 0)), 1)     AS avg_patients_per_clinician
FROM visit_waits
GROUP BY 1
ORDER BY
    CASE load_tier
        WHEN 'Peak (top 25%)' THEN 1
        WHEN 'Normal'         THEN 2
        WHEN 'Quiet (bottom 25%)' THEN 3
    END
"""
    return run_query(sql)


def load_offpeak_investigation_pattern(filters: dict, run_query) -> pd.DataFrame:
    """G3: Investigation ordering rate by hour of day (EAT) — detects off-peak over-ordering."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
visit_inv AS (
    SELECT
        HOUR(CONVERT_TIMEZONE('UTC', 'Africa/Nairobi', v.created_at)) AS hour_eat,
        v.id AS visit_id,
        v.source_schema,
        COUNT(i.investigation_created_at)                              AS inv_count,
        MAX(CASE WHEN i.visit_id IS NOT NULL THEN 1 ELSE 0 END)        AS had_inv
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS i
        ON v.id = i.visit_id
       AND (i.cancelled IS NULL OR i.cancelled = 0)
       AND (i.remove_from_report IS NULL OR i.remove_from_report = 0)
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
    GROUP BY 1, 2, 3
)
SELECT
    hour_eat,
    CASE
        WHEN hour_eat >= 20 OR hour_eat <= 6 THEN 'Night shift'
        WHEN hour_eat BETWEEN 7 AND 12        THEN 'Morning'
        WHEN hour_eat BETWEEN 13 AND 17       THEN 'Afternoon'
        ELSE 'Evening'
    END AS shift_label,
    COUNT(DISTINCT visit_id)                                           AS total_visits,
    ROUND(DIV0(SUM(had_inv), COUNT(DISTINCT visit_id)) * 100, 1)       AS inv_rate_pct,
    ROUND(DIV0(SUM(inv_count), COUNT(DISTINCT visit_id)), 2)           AS avg_inv_per_visit
FROM visit_inv
GROUP BY 1, 2
ORDER BY 1
"""
    return run_query(sql)


def load_offpeak_ipop_split(filters: dict, run_query) -> pd.DataFrame:
    """E3 drill-down: Investigation count split by Inpatient vs Outpatient for off-peak vs peak."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
classified AS (
    SELECT
        v.id AS visit_id,
        v.source_schema,
        CASE
            WHEN HOUR(CONVERT_TIMEZONE('UTC', 'Africa/Nairobi', v.created_at)) >= 20
              OR HOUR(CONVERT_TIMEZONE('UTC', 'Africa/Nairobi', v.created_at)) <= 6
            THEN 'Off-peak (Night)'
            ELSE 'Peak Hours'
        END AS shift_type,
        CASE WHEN a.visit_id IS NOT NULL THEN 'Inpatient' ELSE 'Outpatient' END AS visit_type
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
        ON v.id = a.visit_id
       AND v.source_schema = REPLACE(LOWER(a.source_schema), '_clean', '')
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
)
SELECT
    c.shift_type,
    c.visit_type,
    COUNT(*)                   AS inv_count,
    COUNT(DISTINCT c.visit_id) AS visit_count
FROM classified c
INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS i
    ON c.visit_id = i.visit_id
   AND (i.cancelled IS NULL OR i.cancelled = 0)
   AND (i.remove_from_report IS NULL OR i.remove_from_report = 0)
GROUP BY 1, 2
ORDER BY 1, 2
"""
    return run_query(sql)


def load_offpeak_top_investigations(filters: dict, run_query) -> pd.DataFrame:
    """E3 drill-down: Investigations ordered during off-peak night hours, grouped by procedure type (investigation_type)."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
offpeak_visits AS (
    SELECT
        v.id AS visit_id,
        v.source_schema,
        CASE WHEN a.visit_id IS NOT NULL THEN 'Inpatient' ELSE 'Outpatient' END AS visit_type
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
        ON v.id = a.visit_id
       AND v.source_schema = REPLACE(LOWER(a.source_schema), '_clean', '')
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      AND (
          HOUR(CONVERT_TIMEZONE('UTC', 'Africa/Nairobi', v.created_at)) >= 20
          OR HOUR(CONVERT_TIMEZONE('UTC', 'Africa/Nairobi', v.created_at)) <= 6
      )
    {wh}
)
SELECT
    INITCAP(TRIM(i.investigation_type)) AS discipline,
    ov.visit_type,
    COUNT(*)                            AS inv_count
FROM offpeak_visits ov
INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS i
    ON ov.visit_id = i.visit_id
   AND (i.cancelled IS NULL OR i.cancelled = 0)
   AND (i.remove_from_report IS NULL OR i.remove_from_report = 0)
   AND i.investigation_type IS NOT NULL
GROUP BY 1, 2
ORDER BY inv_count DESC
"""
    return run_query(sql)


def load_discharge_timing(filters: dict, run_query) -> pd.DataFrame:
    """E4: Actual discharge hour distribution (EAT) using discharged_at.
    Returns hour × day_of_week discharge count for overlap comparison with admissions."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
discharge_hours AS (
    SELECT
        CASE DAYNAME(CONVERT_TIMEZONE('UTC', 'Africa/Nairobi', a.discharged_at))
            WHEN 'Mon' THEN 'Monday'    WHEN 'Tue' THEN 'Tuesday'
            WHEN 'Wed' THEN 'Wednesday' WHEN 'Thu' THEN 'Thursday'
            WHEN 'Fri' THEN 'Friday'    WHEN 'Sat' THEN 'Saturday'
            WHEN 'Sun' THEN 'Sunday'
        END                                                                   AS day_name,
        DAYOFWEEK(CONVERT_TIMEZONE('UTC', 'Africa/Nairobi', a.discharged_at)) AS day_num,
        HOUR(CONVERT_TIMEZONE('UTC', 'Africa/Nairobi', a.discharged_at))      AS hour_eat,
        COUNT(DISTINCT a.visit_id)                                            AS discharge_count
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    INNER JOIN schema_anchor sa
        ON REPLACE(LOWER(a.source_schema), '_clean', '') = sa.source_schema
    WHERE a.discharged_at >= DATEADD('month', -{mo}, sa.max_date)
      AND a.discharged_at IS NOT NULL
      AND a.is_open_admission = 0
    GROUP BY 1, 2, 3
)
SELECT day_name, day_num, hour_eat, discharge_count
FROM discharge_hours
ORDER BY day_num, hour_eat
"""
    return run_query(sql)


def load_peak_demand_heatmap(filters: dict, run_query) -> pd.DataFrame:
    """Q6A: Hour × day visit volume heatmap in EAT timezone, split by outpatient/inpatient."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    {wsa}
    GROUP BY source_schema
)
SELECT
    CASE DAYNAME(CONVERT_TIMEZONE('UTC', 'Africa/Nairobi', v.created_at))
        WHEN 'Mon' THEN 'Monday'    WHEN 'Tue' THEN 'Tuesday'
        WHEN 'Wed' THEN 'Wednesday' WHEN 'Thu' THEN 'Thursday'
        WHEN 'Fri' THEN 'Friday'    WHEN 'Sat' THEN 'Saturday'
        WHEN 'Sun' THEN 'Sunday'
    END                                                                 AS day_name,
    DAYOFWEEK(CONVERT_TIMEZONE('UTC', 'Africa/Nairobi', v.created_at)) AS day_num,
    HOUR(CONVERT_TIMEZONE('UTC', 'Africa/Nairobi', v.created_at))       AS hour_of_day,
    COUNT(DISTINCT v.id)                                                AS visit_count,
    COUNT(DISTINCT CASE WHEN a.visit_id IS NOT NULL
                        THEN v.id END)                                  AS inpatient_count,
    COUNT(DISTINCT CASE WHEN a.visit_id IS NULL
                        THEN v.id END)                                  AS outpatient_count
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    ON v.id = a.visit_id
   AND v.source_schema = REPLACE(LOWER(a.source_schema), '_clean', '')
WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
{wh}
GROUP BY 1, 2, 3
ORDER BY 2, 3
"""
    return run_query(sql)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PATIENT SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def load_seg_kpis(filters: dict, run_query) -> pd.DataFrame:
    """Tab 2 KPI row: total patients, chronic rate, repeat rate, single-visit count."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
visit_counts AS (
    SELECT v.source_schema, v.patient, COUNT(DISTINCT v.id) AS vc
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
    GROUP BY 1, 2
),
chronic AS (
    SELECT DISTINCT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON v.id = n.visit_id AND v.source_schema = n.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND (dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1
           OR n.diagnosis ILIKE '%hypertension%'
           OR n.diagnosis ILIKE '%diabetes%'
           OR n.diagnosis ILIKE '%hiv%'
           OR n.diagnosis ILIKE '%renal%')
)
SELECT
    COUNT(DISTINCT vc.patient)                              AS total_patients,
    ROUND(AVG(vc.vc), 1)                                    AS avg_visits,
    COUNT(DISTINCT cp.patient)                              AS chronic_patients,
    COUNT(DISTINCT CASE WHEN vc.vc = 1 THEN vc.patient END) AS single_visit,
    COUNT(DISTINCT CASE WHEN vc.vc >= 2 THEN vc.patient END) AS repeat_patients,
    ROUND(DIV0(COUNT(DISTINCT cp.patient),
               COUNT(DISTINCT vc.patient)) * 100, 1)        AS chronic_rate_pct,
    ROUND(DIV0(COUNT(DISTINCT CASE WHEN vc.vc >= 2 THEN vc.patient END),
               COUNT(DISTINCT vc.patient)) * 100, 1)        AS repeat_rate_pct
FROM visit_counts vc
LEFT JOIN chronic cp
    ON vc.patient = cp.patient AND vc.source_schema = cp.source_schema
"""
    return run_query(sql)


def load_demographics_age_sex(filters: dict, run_query) -> pd.DataFrame:
    """Q2: Age × sex × chronic status grid."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
pp AS (
    SELECT v.source_schema, v.patient,
        UPPER(COALESCE(p.sex, 'Unknown')) AS sex,
           CASE
            WHEN p.dob IS NULL THEN 'Unknown'
            WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 5
                THEN 'Toddler (0–4)'
            WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 13
                THEN 'Child (5–12)'
            WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 18
                THEN 'Adolescent (13–17)'
            WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 25
                THEN 'Youth (18–24)'
            WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 35
                THEN 'Young Adult (25–34)'
            WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 45
                THEN 'Adult (35–44)'
            WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 55
                THEN 'Middle Age (45–54)'
            WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 65
                THEN 'Older Adult (55–64)'
            ELSE 'Senior (65+)'
        END                                             AS age_group,
        MAX(CASE
            WHEN dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1
              OR n.diagnosis ILIKE '%hypertension%'
              OR n.diagnosis ILIKE '%diabetes%'
              OR n.diagnosis ILIKE '%hiv%'
              OR n.diagnosis ILIKE '%renal%'
            THEN 1 ELSE 0
        END) AS is_chronic
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS p
        ON v.patient = p.patient_id AND v.source_schema = p.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON v.id = n.visit_id AND v.source_schema = n.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
    GROUP BY 1, 2, 3, 4
)
SELECT age_group, sex,
    COUNT(DISTINCT patient)                     AS total,
    SUM(is_chronic)                             AS chronic,
    COUNT(DISTINCT patient) - SUM(is_chronic)   AS non_chronic
FROM pp
WHERE age_group != 'Unknown' AND sex NOT IN ('UNKNOWN', 'Unknown')
GROUP BY 1, 2
ORDER BY 1, 2
"""
    return run_query(sql)


def load_new_vs_returning(filters: dict, run_query) -> pd.DataFrame:
    """Q4: New vs returning patient trend monthly."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
atf AS (
    SELECT source_schema, patient, MIN(created_at) AS first_ever
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS GROUP BY 1, 2
)
SELECT
    DATE_TRUNC('month', v.created_at)           AS visit_month,
    COUNT(DISTINCT v.patient)                   AS total_patients,
    COUNT(DISTINCT CASE
        WHEN DATE_TRUNC('month', atf.first_ever)
             = DATE_TRUNC('month', v.created_at)
        THEN v.patient END)                     AS new_patients,
    COUNT(DISTINCT CASE
        WHEN DATE_TRUNC('month', atf.first_ever)
             != DATE_TRUNC('month', v.created_at)
        THEN v.patient END)                     AS returning_patients
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
LEFT JOIN atf ON v.patient = atf.patient AND v.source_schema = atf.source_schema
WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
{wh}
GROUP BY 1
ORDER BY 1
"""
    return run_query(sql)


def load_visit_distribution(filters: dict, run_query) -> pd.DataFrame:
    """Distribution of patients by age group, visit type, and new/returning status."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
atf AS (
    SELECT source_schema, patient, MIN(created_at) AS first_ever
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS GROUP BY 1, 2
)
SELECT
   CASE
            WHEN p.dob IS NULL THEN 'Unknown'
            WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 5
                THEN 'Toddler (0–4)'
            WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 13
                THEN 'Child (5–12)'
            WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 18
                THEN 'Adolescent (13–17)'
            WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 25
                THEN 'Youth (18–24)'
            WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 35
                THEN 'Young Adult (25–34)'
            WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 45
                THEN 'Adult (35–44)'
            WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 55
                THEN 'Middle Age (45–54)'
            WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 65
                THEN 'Older Adult (55–64)'
            ELSE 'Senior (65+)'
        END                                             AS age_group,
    CASE WHEN a.visit_id IS NOT NULL THEN 'Inpatient' ELSE 'Outpatient'
    END                                                             AS visit_type,
    CASE
        WHEN DATE_TRUNC('month', atf.first_ever) = DATE_TRUNC('month', v.created_at)
        THEN 'New' ELSE 'Returning'
    END                                                             AS patient_type,
    COUNT(DISTINCT v.patient)                                       AS patient_count,
    COUNT(DISTINCT v.id)                                            AS visit_count
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
LEFT JOIN atf ON v.patient = atf.patient AND v.source_schema = atf.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS p
    ON v.patient = p.patient_id AND v.source_schema = p.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    ON v.id = a.visit_id
   AND REPLACE(LOWER(a.source_schema), '_clean', '') = v.source_schema
WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
  AND age_group != 'Unknown'
{wh}
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3
"""
    return run_query(sql)


def load_payer_switch_sankey(filters: dict, run_query) -> pd.DataFrame:
    """Payer type on first visit (source) → payer type on subsequent visits (target).
    Only patients with ≥2 visits are included. Returns flow pairs with patient counts."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
payer_label AS (
    SELECT
        v.patient, v.source_schema, v.created_at,
        CASE
            WHEN LOWER(v.payment_mode) IN ('nhif','shif','sha','national scheme')
                THEN 'NHIF / SHA'
            WHEN LOWER(v.payment_mode) IN ('cash','self-pay','out-of-pocket','copay')
                THEN 'Cash'
            WHEN v.payment_mode IS NULL OR TRIM(v.payment_mode) = ''
                THEN 'Unknown'
            ELSE 'Insurance'
        END AS payer
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
),
first_visit AS (
    SELECT patient, source_schema,
           MIN(created_at) AS first_visit_at
    FROM payer_label GROUP BY 1, 2
),
first_payer AS (
    SELECT pl.patient, pl.source_schema, pl.payer AS source_payer
    FROM payer_label pl
    INNER JOIN first_visit fv
        ON pl.patient = fv.patient
       AND pl.source_schema = fv.source_schema
       AND pl.created_at = fv.first_visit_at
),
return_payers AS (
    SELECT pl.patient, pl.source_schema, pl.payer AS target_payer
    FROM payer_label pl
    INNER JOIN first_visit fv
        ON pl.patient = fv.patient
       AND pl.source_schema = fv.source_schema
    WHERE pl.created_at > fv.first_visit_at
      AND pl.payer != 'Unknown'
)
SELECT
    fp.source_payer,
    rp.target_payer,
    COUNT(DISTINCT fp.patient)  AS patient_count
FROM first_payer fp
INNER JOIN return_payers rp
    ON fp.patient = rp.patient AND fp.source_schema = rp.source_schema
WHERE fp.source_payer != 'Unknown'
GROUP BY 1, 2
HAVING patient_count >= 3
ORDER BY patient_count DESC
"""
    return run_query(sql)


def load_payer_mix(filters: dict, run_query) -> pd.DataFrame:
    """Q5: Payer mix by age group."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
)
SELECT
    CASE
        WHEN p.dob IS NULL THEN 'Unknown'
        WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 5  THEN 'Toddler (0–4)'
        WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 13 THEN 'Child (5–12)'
        WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 18 THEN 'Adolescent (13–17)'
        WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 25 THEN 'Youth (18–24)'
        WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 35 THEN 'Young Adult (25–34)'
        WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 45 THEN 'Adult (35–44)'
        WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 55 THEN 'Middle Age (45–54)'
        WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 65 THEN 'Older Adult (55–64)'
        ELSE 'Senior (65+)'
    END                                                     AS age_group,
    CASE
        WHEN LOWER(v.payment_mode) IN ('nhif','shif','sha','national scheme')
            THEN 'NHIF / SHA'
        WHEN LOWER(v.payment_mode) IN ('cash','self-pay','out-of-pocket','copay')
            THEN 'Cash'
        WHEN v.payment_mode IS NULL OR TRIM(v.payment_mode) = ''
            THEN 'Unknown'
        ELSE 'Insurance'
    END                                                     AS payer_type,
    COUNT(DISTINCT v.patient)                               AS unique_patients,
    COUNT(DISTINCT v.id)                                    AS total_visits
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS p
    ON v.patient = p.patient_id AND v.source_schema = p.source_schema
WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
  AND payer_type != 'Unknown'
  AND age_group  != 'Unknown'
{wh}
GROUP BY 1, 2
ORDER BY 1, total_visits DESC
"""
    return run_query(sql)


def load_revenue_by_segment(filters: dict, run_query) -> pd.DataFrame:
    """Q6: Revenue by clinical segment (condition × payer)."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
vr AS (
    SELECT v.source_schema, v.patient, v.id AS visit_id,
        CASE WHEN LOWER(v.payment_mode) IN ('nhif','shif','sha','national scheme')
             THEN 'NHIF / SHA'
             WHEN LOWER(v.payment_mode) IN ('cash','self-pay','out-of-pocket')
             THEN 'Cash' ELSE 'Insurance'
        END AS payer_type,
        COALESCE(SUM(il.item_amount), 0) AS visit_revenue
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_INVOICE_LINE_ITEMS il
        ON v.id = il.visit_id AND v.source_schema = il.source_schema
       AND il.invoice_deleted_at IS NULL
       AND (il.auto_cancelled IS NULL OR il.auto_cancelled = 0)
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
    GROUP BY 1, 2, 3, 4
),
ps AS (
    SELECT v.source_schema, v.patient,
        COALESCE(
            MAX(NULLIF(TRIM(dx.disease_group_1), '')),
            MAX(CASE
                WHEN n.diagnosis ILIKE '%hypertension%' THEN 'Hypertension'
                WHEN n.diagnosis ILIKE '%diabetes%'     THEN 'Diabetes'
                WHEN n.diagnosis ILIKE '%hiv%'          THEN 'HIV'
                WHEN n.diagnosis ILIKE '%malaria%'      THEN 'Malaria'
                WHEN n.diagnosis ILIKE '%urti%'         THEN 'URTI'
                WHEN n.diagnosis ILIKE '%anc%'          THEN 'ANC / Maternal'
                ELSE NULL
            END),
            'Non-chronic / Acute'
        ) AS primary_condition
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON v.id = n.visit_id AND v.source_schema = n.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
    GROUP BY 1, 2
)
SELECT
    ps.primary_condition,
    vr.payer_type,
    COUNT(DISTINCT vr.patient)                  AS patient_count,
    ROUND(SUM(vr.visit_revenue), 0)             AS total_revenue,
    ROUND(AVG(vr.visit_revenue), 0)             AS avg_revenue_per_visit
FROM vr
LEFT JOIN ps ON vr.patient = ps.patient AND vr.source_schema = ps.source_schema
GROUP BY 1, 2
HAVING COUNT(DISTINCT vr.patient) >= 5
ORDER BY total_revenue DESC
LIMIT 30
"""
    return run_query(sql)


def load_pareto(filters: dict, run_query) -> pd.DataFrame:
    """Q7: Revenue Pareto — concentration by spend tier."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
pr AS (
    SELECT v.source_schema, v.patient,
        SUM(il.item_amount) AS total_spend,
        COUNT(DISTINCT v.id) AS visit_count
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_INVOICE_LINE_ITEMS il
        ON v.id = il.visit_id AND v.source_schema = il.source_schema
       AND il.invoice_deleted_at IS NULL
       AND (il.auto_cancelled IS NULL OR il.auto_cancelled = 0)
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
    GROUP BY 1, 2
),
ranked AS (
    SELECT *,
        CASE
            WHEN PERCENT_RANK() OVER (
                PARTITION BY source_schema ORDER BY total_spend DESC) <= 0.10
                THEN 'Top 10%'
            WHEN PERCENT_RANK() OVER (
                PARTITION BY source_schema ORDER BY total_spend DESC) <= 0.20
                THEN 'Top 11–20%'
            WHEN PERCENT_RANK() OVER (
                PARTITION BY source_schema ORDER BY total_spend DESC) <= 0.50
                THEN 'Middle 21–50%'
            ELSE 'Bottom 50%'
        END AS revenue_tier
    FROM pr WHERE total_spend > 0
)
SELECT
    revenue_tier,
    COUNT(DISTINCT patient)                             AS patient_count,
    ROUND(SUM(total_spend), 0)                          AS tier_revenue,
    ROUND(DIV0(SUM(total_spend),
               SUM(SUM(total_spend)) OVER ()) * 100, 1) AS revenue_share_pct,
    ROUND(AVG(total_spend), 0)                          AS avg_spend,
    ROUND(AVG(visit_count), 1)                          AS avg_visits
FROM ranked
GROUP BY 1
ORDER BY revenue_share_pct DESC
"""
    return run_query(sql)


def load_revenue_profile_matrix(filters: dict, run_query) -> pd.DataFrame:
    """Unified Revenue Profile Matrix: per condition — revenue share, pareto tier,
    IP vs OP split, and Cash / Corporate / NHIF-SHA payer mix."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
visit_base AS (
    SELECT
        v.id AS visit_id, v.source_schema, v.payment_mode,
        COALESCE(
            MAX(NULLIF(TRIM(dx.disease_group_1), '')),
            MAX(CASE
                WHEN n.diagnosis ILIKE '%hypertension%' THEN 'Hypertension'
                WHEN n.diagnosis ILIKE '%diabetes%'     THEN 'Diabetes'
                WHEN n.diagnosis ILIKE '%hiv%'          THEN 'HIV'
                WHEN n.diagnosis ILIKE '%malaria%'      THEN 'Malaria'
                WHEN n.diagnosis ILIKE '%typhoid%'      THEN 'Communicable: Typhoid'
                WHEN n.diagnosis ILIKE '%trauma%'
                  OR n.diagnosis ILIKE '%injury%'       THEN 'Trauma & Injury'
                WHEN n.diagnosis ILIKE '%urti%'         THEN 'Respiratory'
                ELSE NULL
            END),
            'Unclassified'
        ) AS condition,
        CASE
            WHEN LOWER(v.payment_mode) IN ('nhif','shif','sha','national scheme')
                THEN 'NHIF / SHA'
            WHEN LOWER(v.payment_mode) IN ('cash','self-pay','out-of-pocket','copay')
                THEN 'Cash'
            WHEN v.payment_mode IS NULL OR TRIM(v.payment_mode) = ''
                THEN 'Unknown'
            ELSE 'Corporate'
        END AS payer_bucket
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON v.id = n.visit_id AND v.source_schema = n.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
    GROUP BY v.id, v.source_schema, v.payment_mode
),
with_revenue AS (
    SELECT
        vb.visit_id, vb.source_schema, vb.condition, vb.payer_bucket,
        COALESCE(SUM(il.item_amount), 0)                              AS revenue,
        MAX(CASE WHEN a.visit_id IS NOT NULL THEN 1 ELSE 0 END)       AS is_inpatient
    FROM visit_base vb
    LEFT JOIN HOSPITALS.STAGING.STG_INVOICE_LINE_ITEMS il
        ON vb.visit_id = il.visit_id AND vb.source_schema = il.source_schema
       AND il.invoice_deleted_at IS NULL
       AND (il.auto_cancelled IS NULL OR il.auto_cancelled = 0)
    LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
        ON vb.visit_id = a.visit_id
       AND REPLACE(LOWER(a.source_schema), '_clean', '') = vb.source_schema
    GROUP BY 1, 2, 3, 4
),
cond_agg AS (
    SELECT
        condition,
        ROUND(SUM(revenue), 0)                                        AS total_revenue,
        COUNT(*)                                                      AS visit_count,
        SUM(is_inpatient)                                             AS ip_visits,
        COUNT(*) - SUM(is_inpatient)                                  AS op_visits,
        SUM(CASE WHEN payer_bucket = 'Cash'      THEN 1 ELSE 0 END)   AS cash_visits,
        SUM(CASE WHEN payer_bucket = 'NHIF / SHA' THEN 1 ELSE 0 END)  AS nhif_visits,
        SUM(CASE WHEN payer_bucket = 'Corporate' THEN 1 ELSE 0 END)   AS corp_visits
    FROM with_revenue
    GROUP BY 1
    HAVING SUM(revenue) > 0 AND COUNT(*) >= 5
),
grand AS (SELECT SUM(total_revenue) AS grand_total FROM cond_agg),
ranked AS (
    SELECT ca.*,
           ROUND(ca.total_revenue / grand.grand_total * 100, 1)       AS revenue_share_pct,
           ROUND(SUM(ca.total_revenue) OVER (
               ORDER BY ca.total_revenue DESC
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
           ) / grand.grand_total * 100, 1)                            AS cumul_pct
    FROM cond_agg ca, grand
)
SELECT
    condition,
    total_revenue,
    revenue_share_pct,
    CASE
        WHEN cumul_pct <= 10  THEN 'Top 10%'
        WHEN cumul_pct <= 20  THEN 'Top 11–20%'
        WHEN cumul_pct <= 50  THEN 'Middle 21–50%'
        ELSE 'Bottom 50%'
    END                                                                AS pareto_tier,
    visit_count,
    ROUND(ip_visits * 100.0 / NULLIF(visit_count, 0), 0)              AS ip_pct,
    ROUND(op_visits * 100.0 / NULLIF(visit_count, 0), 0)              AS op_pct,
    ROUND(cash_visits * 100.0 / NULLIF(visit_count, 0), 0)            AS cash_pct,
    ROUND(nhif_visits * 100.0 / NULLIF(visit_count, 0), 0)            AS nhif_pct,
    ROUND(corp_visits * 100.0 / NULLIF(visit_count, 0), 0)            AS corp_pct
FROM ranked
ORDER BY total_revenue DESC
LIMIT 25
"""
    return run_query(sql)


def load_cohort_forecast(filters: dict, run_query) -> pd.DataFrame:
    """Q9: Age cohort monthly patient counts for stacked area forecast."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
)
SELECT
    DATE_TRUNC('month', v.created_at)   AS visit_month,
    CASE
        WHEN p.dob IS NULL THEN 'Unknown'
        WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 5  THEN 'Toddler (0–4)'
        WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 13 THEN 'Child (5–12)'
        WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 18 THEN 'Adolescent (13–17)'
        WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 25 THEN 'Youth (18–24)'
        WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 35 THEN 'Young Adult (25–34)'
        WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 45 THEN 'Adult (35–44)'
        WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 55 THEN 'Middle Age (45–54)'
        WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 65 THEN 'Older Adult (55–64)'
        ELSE 'Senior (65+)'
    END                                 AS age_cohort,
    COUNT(DISTINCT v.patient)           AS patient_count
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS p
    ON v.patient = p.patient_id AND v.source_schema = p.source_schema
WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
  AND age_cohort != 'Unknown'
{wh}
GROUP BY 1, 2
ORDER BY 1, 2
"""
    return run_query(sql)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PATIENT FLOW & RETENTION
# ══════════════════════════════════════════════════════════════════════════════

def load_retention_kpis(filters: dict, run_query) -> pd.DataFrame:
    """Tab 3 KPI row: chronic patients, 90-day retention rate, LTFU count."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
chronic_patients AS (
    SELECT DISTINCT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON v.id = n.visit_id AND v.source_schema = n.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND (dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1
           OR n.diagnosis ILIKE '%hypertension%'
           OR n.diagnosis ILIKE '%diabetes%'
           OR n.diagnosis ILIKE '%hiv%')
),
visit_gaps AS (
    SELECT v.source_schema, v.patient, v.created_at,
        LEAD(v.created_at) OVER (
            PARTITION BY v.source_schema, v.patient
            ORDER BY v.created_at
        ) AS next_visit
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN chronic_patients cp
        ON v.patient = cp.patient AND v.source_schema = cp.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
)
SELECT
    COUNT(DISTINCT patient)                                 AS chronic_patients,
    COUNT(DISTINCT CASE
        WHEN next_visit IS NOT NULL
         AND next_visit <= created_at + INTERVAL '90 days'
        THEN patient END)                                   AS retained_patients,
    ROUND(DIV0(
        COUNT(DISTINCT CASE
            WHEN next_visit IS NOT NULL
             AND next_visit <= created_at + INTERVAL '90 days'
            THEN patient END),
        COUNT(DISTINCT patient)
    ) * 100, 1)                                             AS retention_rate_pct,
    COUNT(DISTINCT CASE
        WHEN (next_visit IS NULL
              OR next_visit > created_at + INTERVAL '90 days')
        THEN patient END)                                   AS ltfu_patients
FROM visit_gaps
"""
    return run_query(sql)


def load_lifecycle(filters: dict, run_query) -> pd.DataFrame:
    """Q1: Patient lifecycle — active / lapsing / LTFU for chronic patients."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
patient_status AS (
    SELECT v.source_schema, v.patient,
        MAX(v.created_at)   AS last_visit,
        sa.max_date,
        MAX(CASE
            WHEN dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1
              OR n.diagnosis ILIKE '%hypertension%'
              OR n.diagnosis ILIKE '%diabetes%'
              OR n.diagnosis ILIKE '%hiv%'
            THEN 1 ELSE 0
        END) AS is_chronic
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON v.id = n.visit_id AND v.source_schema = n.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
    GROUP BY 1, 2, sa.max_date
)
SELECT
    CASE
        WHEN DATEDIFF('day', last_visit, max_date) <= 90  THEN '1. Active (≤90 days)'
        WHEN DATEDIFF('day', last_visit, max_date) <= 180 THEN '2. Lapsing (91–180 days)'
        ELSE '3. LTFU (>180 days)'
    END                                                 AS lifecycle_status,
    is_chronic,
    COUNT(DISTINCT patient)                             AS patient_count
FROM patient_status
WHERE is_chronic = 1
GROUP BY 1, 2
ORDER BY 1
"""
    return run_query(sql)


def load_retention_by_payer(filters: dict, run_query) -> pd.DataFrame:
    """Q3: 90-day retention rate by payer type for chronic patients."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
chronic_pts AS (
    SELECT DISTINCT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND (dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1)
),
pv AS (
    SELECT v.source_schema, v.patient,
        CASE WHEN LOWER(v.payment_mode) IN ('nhif','shif','sha') THEN 'NHIF / SHA'
             WHEN LOWER(v.payment_mode) IN ('cash','self-pay','out-of-pocket') THEN 'Cash'
             ELSE 'Insurance'
        END AS payer_type,
        v.created_at,
        LEAD(v.created_at) OVER (
            PARTITION BY v.source_schema, v.patient ORDER BY v.created_at
        ) AS next_visit
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN chronic_pts cp
        ON v.patient = cp.patient AND v.source_schema = cp.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
)
SELECT
    payer_type,
    COUNT(DISTINCT patient)                             AS total_patients,
    COUNT(DISTINCT CASE
        WHEN next_visit IS NOT NULL
         AND next_visit <= created_at + INTERVAL '90 days'
        THEN patient END)                               AS retained,
    ROUND(DIV0(
        COUNT(DISTINCT CASE
            WHEN next_visit IS NOT NULL
             AND next_visit <= created_at + INTERVAL '90 days'
            THEN patient END),
        COUNT(DISTINCT patient)
    ) * 100, 1)                                         AS retention_pct
FROM pv
GROUP BY 1
ORDER BY retention_pct DESC
"""
    return run_query(sql)


def load_dropout_causes(filters: dict, run_query) -> pd.DataFrame:
    """Q6: Dropout cause attribution for LTFU chronic patients."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
ltfu AS (
    SELECT v.source_schema, v.patient, MAX(v.created_at) AS last_visit
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON v.id = n.visit_id AND v.source_schema = n.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND (dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1
           OR n.diagnosis ILIKE '%hypertension%'
           OR n.diagnosis ILIKE '%diabetes%')
    GROUP BY 1, 2
    HAVING DATEDIFF('day', MAX(v.created_at), MAX(sa.max_date)) > 90
),
rx_gap AS (
    SELECT v.source_schema, v.patient,
        ROUND(DIV0(
            COUNT(CASE WHEN (pp.stopped IS NULL OR pp.stopped = 0)
                        AND (pp.canceled IS NULL OR pp.canceled = 0) THEN 1 END),
            COUNT(pp.prescription_id)
        ) * 100, 1) AS rx_completion_pct
    FROM HOSPITALS.STAGING.STG_PRESCRIPTION_PAYMENTS pp
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON pp.visit_id = v.id AND pp.source_schema = v.source_schema
    WHERE pp.remove_from_report IS NULL OR pp.remove_from_report = 0
    GROUP BY 1, 2
),
care_frag AS (
    SELECT source_schema, patient, COUNT(DISTINCT user) AS unique_clinicians
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    WHERE user IS NOT NULL
    GROUP BY 1, 2
),
bp AS (
    SELECT v.source_schema, v.patient,
        AVG(vt.bp_systolic)  AS avg_sys,
        AVG(vt.bp_diastolic) AS avg_dia
    FROM HOSPITALS.STAGING.STG_EVALUATION_VITALS vt
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON vt.visit_id = v.id AND vt.source_schema = v.source_schema
    WHERE vt.bp_systolic IS NOT NULL
    GROUP BY 1, 2
)
SELECT
    SUM(CASE WHEN COALESCE(rx.rx_completion_pct, 100) < 70 THEN 1 ELSE 0 END)
        AS ltfu_rx_gap,
    SUM(CASE WHEN COALESCE(cf.unique_clinicians, 0) > 3 THEN 1 ELSE 0 END)
        AS ltfu_fragmented_care,
    SUM(CASE WHEN COALESCE(bp.avg_sys, 0) >= 140
              OR COALESCE(bp.avg_dia, 0) >= 90 THEN 1 ELSE 0 END)
        AS ltfu_uncontrolled_bp,
    COUNT(DISTINCT lf.patient)                              AS total_ltfu
FROM ltfu lf
LEFT JOIN rx_gap    rx ON lf.patient = rx.patient AND lf.source_schema = rx.source_schema
LEFT JOIN care_frag cf ON lf.patient = cf.patient AND lf.source_schema = cf.source_schema
LEFT JOIN bp           ON lf.patient = bp.patient AND lf.source_schema = bp.source_schema
"""
    return run_query(sql)


def load_ltfu_correlation(filters: dict, run_query) -> pd.DataFrame:
    """B: Compare LTFU vs retained chronic patients by age group, sex, and payer."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
chronic_patients AS (
    SELECT v.source_schema, v.patient,
        MAX(v.created_at) AS last_visit,
        COUNT(DISTINCT v.id) AS total_visits,
        MAX(CASE
            WHEN UPPER(v.payment_mode) IN ('CASH','PRIVATE') THEN 'Cash'
            WHEN UPPER(v.payment_mode) LIKE '%NHIF%'
              OR UPPER(v.payment_mode) LIKE '%SHA%' THEN 'NHIF / SHA'
            ELSE 'Insurance / Corporate'
        END) AS payer
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE (dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1)
      AND v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
    GROUP BY 1, 2
),
patient_status AS (
    SELECT cp.source_schema, cp.patient, cp.total_visits, cp.payer,
        CASE WHEN DATEDIFF('day', cp.last_visit, sa.max_date) > 90
             THEN 'LTFU' ELSE 'Retained' END AS retention_status
    FROM chronic_patients cp
    INNER JOIN schema_anchor sa ON cp.source_schema = sa.source_schema
),
patient_profile AS (
    SELECT
        ps.patient, ps.retention_status, ps.payer,
        CASE
            WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 18  THEN 'Under 18'
            WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 35  THEN '18–34'
            WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 50  THEN '35–49'
            WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 65  THEN '50–64'
            ELSE '65+'
        END AS age_group,
        COALESCE(rp.sex, 'Unknown') AS sex
    FROM patient_status ps
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON ps.patient = v.patient AND ps.source_schema = v.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS rp
        ON ps.patient = rp.patient_id AND ps.source_schema = rp.source_schema
    GROUP BY 1, 2, 3, 4, 5
),
by_age AS (
    SELECT 'Age Group' AS factor, age_group AS dimension,
        COUNT(DISTINCT CASE WHEN retention_status = 'Retained' THEN patient END) AS retained,
        COUNT(DISTINCT CASE WHEN retention_status = 'LTFU'     THEN patient END) AS ltfu,
        COUNT(DISTINCT patient) AS total,
        ROUND(DIV0(
            COUNT(DISTINCT CASE WHEN retention_status = 'LTFU' THEN patient END),
            COUNT(DISTINCT patient)
        ) * 100, 1) AS ltfu_rate_pct
    FROM patient_profile
    GROUP BY age_group
),
by_sex AS (
    SELECT 'Sex' AS factor, sex AS dimension,
        COUNT(DISTINCT CASE WHEN retention_status = 'Retained' THEN patient END),
        COUNT(DISTINCT CASE WHEN retention_status = 'LTFU'     THEN patient END),
        COUNT(DISTINCT patient),
        ROUND(DIV0(
            COUNT(DISTINCT CASE WHEN retention_status = 'LTFU' THEN patient END),
            COUNT(DISTINCT patient)
        ) * 100, 1)
    FROM patient_profile
    GROUP BY sex
),
by_payer AS (
    SELECT 'Payer' AS factor, payer AS dimension,
        COUNT(DISTINCT CASE WHEN retention_status = 'Retained' THEN patient END),
        COUNT(DISTINCT CASE WHEN retention_status = 'LTFU'     THEN patient END),
        COUNT(DISTINCT patient),
        ROUND(DIV0(
            COUNT(DISTINCT CASE WHEN retention_status = 'LTFU' THEN patient END),
            COUNT(DISTINCT patient)
        ) * 100, 1)
    FROM patient_profile
    GROUP BY payer
)
SELECT * FROM by_age
UNION ALL SELECT * FROM by_sex
UNION ALL SELECT * FROM by_payer
ORDER BY factor, ltfu_rate_pct DESC
"""
    return run_query(sql)


def load_revenue_at_risk(filters: dict, run_query) -> pd.DataFrame:
    """Q7: Revenue at risk from chronic LTFU patients."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
avg_rev AS (
    SELECT v.source_schema,
        ROUND(AVG(vt.total), 0) AS avg_rev_per_visit
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN (
        SELECT visit_id, source_schema, SUM(item_amount) AS total
        FROM HOSPITALS.STAGING.STG_INVOICE_LINE_ITEMS
        WHERE invoice_deleted_at IS NULL
          AND (auto_cancelled IS NULL OR auto_cancelled = 0)
        GROUP BY 1, 2
    ) vt ON v.id = vt.visit_id AND v.source_schema = vt.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
    GROUP BY 1
),
ltfu_counts AS (
    SELECT v.source_schema,
        COUNT(DISTINCT CASE
            WHEN (dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1)
             AND DATEDIFF('day', lv.last_v, sa.max_date) > 90
            THEN v.patient END)                             AS chronic_ltfu,
        COUNT(DISTINCT CASE
            WHEN (dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1)
             AND DATEDIFF('day', lv.last_v, sa.max_date) BETWEEN 31 AND 90
            THEN v.patient END)                             AS chronic_lapsing
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN (
        SELECT source_schema, patient, MAX(created_at) AS last_v
        FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS GROUP BY 1, 2
    ) lv ON v.patient = lv.patient AND v.source_schema = lv.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
    GROUP BY 1
)
SELECT
    lc.source_schema,
    lc.chronic_ltfu,
    lc.chronic_lapsing,
    ar.avg_rev_per_visit,
    ROUND(lc.chronic_ltfu    * 4 * ar.avg_rev_per_visit, 0) AS chronic_ltfu_revenue_at_risk,
    ROUND(lc.chronic_lapsing * 4 * ar.avg_rev_per_visit, 0) AS lapsing_revenue_recoverable
FROM ltfu_counts lc
LEFT JOIN avg_rev ar ON lc.source_schema = ar.source_schema
"""
    return run_query(sql)


def load_outreach_list(filters: dict, run_query) -> pd.DataFrame:
    """Q11: Re-engagement outreach list — Campaign A (30–60d) and B (61–90d)."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    # Build raw WHERE for the last_visit CTE which lacks a join alias
    schema_raw = f"AND source_schema = '{filters['schema']}'" if filters.get("schema") else ""
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
lv AS (
    SELECT source_schema, patient, MAX(created_at) AS last_visit_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    WHERE 1=1 {schema_raw}
    GROUP BY 1, 2
),
chronic AS (
    SELECT v.source_schema, v.patient,
        COALESCE(
            MAX(NULLIF(TRIM(dx.disease_group_1), '')),
            MAX(CASE
                WHEN n.diagnosis ILIKE '%hypertension%' THEN 'Hypertension'
                WHEN n.diagnosis ILIKE '%diabetes%'     THEN 'Diabetes'
                WHEN n.diagnosis ILIKE '%hiv%'          THEN 'HIV'
                ELSE NULL
            END),
            'Chronic - unspecified'
        ) AS primary_condition
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON v.id = n.visit_id AND v.source_schema = n.source_schema
    WHERE dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1
       OR n.diagnosis ILIKE '%hypertension%'
       OR n.diagnosis ILIKE '%diabetes%'
       OR n.diagnosis ILIKE '%hiv%'
    GROUP BY v.source_schema, v.patient
),
rx AS (
    SELECT v.source_schema, v.patient,
        ROUND(DIV0(
            COUNT(CASE WHEN (pp.stopped IS NULL OR pp.stopped = 0)
                        AND (pp.canceled IS NULL OR pp.canceled = 0) THEN 1 END),
            COUNT(pp.prescription_id)
        ) * 100, 1) AS rx_completion_pct
    FROM HOSPITALS.STAGING.STG_PRESCRIPTION_PAYMENTS pp
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON pp.visit_id = v.id AND pp.source_schema = v.source_schema
    WHERE pp.remove_from_report IS NULL OR pp.remove_from_report = 0
    GROUP BY 1, 2
)
SELECT
    lv.patient,
    lv.last_visit_date,
    DATEDIFF('day', lv.last_visit_date, sa.max_date)    AS days_since,
    c.primary_condition,
    COALESCE(rx.rx_completion_pct, 100)                 AS rx_completion_pct,
    CASE WHEN DATEDIFF('day', lv.last_visit_date, sa.max_date) BETWEEN 30 AND 60
         THEN 'Campaign A — early lapsing (30–60d)'
         ELSE 'Campaign B — deep lapsing (61–90d)'
    END                                                 AS campaign,
    ROUND(
        40
      + CASE WHEN COALESCE(rx.rx_completion_pct, 100) < 50 THEN 30
             WHEN COALESCE(rx.rx_completion_pct, 100) < 70 THEN 15
             ELSE 0 END
      + LEAST(DATEDIFF('day', lv.last_visit_date, sa.max_date) - 30, 60) / 60.0 * 30
    , 0)                                                AS priority_score
FROM lv
INNER JOIN schema_anchor sa ON lv.source_schema = sa.source_schema
INNER JOIN chronic c ON lv.patient = c.patient AND lv.source_schema = c.source_schema
LEFT JOIN rx ON lv.patient = rx.patient AND lv.source_schema = rx.source_schema
WHERE DATEDIFF('day', lv.last_visit_date, sa.max_date) BETWEEN 30 AND 90
ORDER BY priority_score DESC
LIMIT 50
"""
    return run_query(sql)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 EXTENDED — RETENTION DEEP DIVES
# ─────────────────────────────────────────────────────────────────────────────

def load_demographic_diagnosis_revenue_risk(filters: dict, run_query) -> pd.DataFrame:
    """R1: Demographic × diagnosis intersection for LTFU revenue at risk."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
chronic_flag AS (
    SELECT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE dx.disease_burden_group_1 ILIKE ANY ('%Cardiovascular%','%Diabetes%',
          '%Chronic%','%HIV%','%Mental%','%Neurolog%')
      AND v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
    GROUP BY 1, 2
),
patient_last AS (
    SELECT v.source_schema, v.patient,
           MAX(v.created_at) AS last_visit,
           COUNT(DISTINCT v.id) AS total_visits
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
    GROUP BY 1, 2
),
ltfu_patients AS (
    SELECT pl.source_schema, pl.patient
    FROM patient_last pl
    INNER JOIN schema_anchor sa ON pl.source_schema = sa.source_schema
    INNER JOIN chronic_flag cf ON pl.patient = cf.patient AND pl.source_schema = cf.source_schema
    WHERE DATEDIFF('day', pl.last_visit, sa.max_date) > 180
),
patient_profile AS (
    SELECT
        v.source_schema, v.patient,
        CASE
            WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 18  THEN 'Under 18'
            WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 35  THEN '18–34'
            WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 50  THEN '35–49'
            WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 65  THEN '50–64'
            ELSE '65+'
        END AS age_group,
        COALESCE(rp.sex, 'Unknown') AS gender,
        CASE
            WHEN UPPER(v.payment_mode) IN ('CASH','PRIVATE') THEN 'Cash'
            WHEN UPPER(v.payment_mode) LIKE '%NHIF%' OR UPPER(v.payment_mode) LIKE '%SHA%' THEN 'NHIF / SHA'
            ELSE 'Insurance / Corporate'
        END AS payer,
        COALESCE(dx.disease_burden_group_1, 'Unclassified') AS condition,
        SUM(COALESCE(ili.item_amount, 0)) AS patient_revenue,
        COUNT(DISTINCT v.id) AS visit_count
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN ltfu_patients lp ON v.patient = lp.patient AND v.source_schema = lp.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS rp
        ON v.patient = rp.patient_id AND v.source_schema = rp.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_INVOICE_LINE_ITEMS ili
        ON v.id = ili.visit_id AND v.source_schema = ili.source_schema
        AND ili.invoice_deleted_at IS NULL
        AND (ili.auto_cancelled IS NULL OR ili.auto_cancelled = 0)
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
    GROUP BY 1, 2, 3, 4, 5, 6
)
SELECT
    age_group,
    gender,
    condition,
    payer,
    COUNT(DISTINCT patient)                                AS ltfu_patients,
    ROUND(AVG(DIV0(patient_revenue, visit_count)), 0)      AS avg_rev_per_visit,
    ROUND(COUNT(DISTINCT patient) * 4
          * AVG(DIV0(patient_revenue, visit_count)), 0)    AS revenue_at_risk
FROM patient_profile
GROUP BY 1, 2, 3, 4
HAVING ltfu_patients >= 2
ORDER BY revenue_at_risk DESC
LIMIT 40
"""
    return run_query(sql)


def load_retained_patient_footprint(filters: dict, run_query) -> pd.DataFrame:
    """R2: For retained chronic patients — are they pharmacy-only or full-service?"""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
chronic_flag AS (
    SELECT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE dx.disease_burden_group_1 ILIKE ANY ('%Cardiovascular%','%Diabetes%',
          '%Chronic%','%HIV%','%Mental%','%Neurolog%')
      AND v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
    GROUP BY 1, 2
),
retained AS (
    SELECT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN chronic_flag cf ON v.patient = cf.patient AND v.source_schema = cf.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
    GROUP BY 1, 2, sa.max_date
    HAVING DATEDIFF('day', MAX(v.created_at), MAX(sa.max_date)) <= 90
),
visit_services AS (
    SELECT
        v.source_schema, v.patient, v.id AS visit_id,
        CASE
            WHEN UPPER(v.payment_mode) IN ('CASH','PRIVATE') THEN 'Cash'
            WHEN UPPER(v.payment_mode) LIKE '%NHIF%' OR UPPER(v.payment_mode) LIKE '%SHA%' THEN 'NHIF / SHA'
            ELSE 'Insurance / Corporate'
        END AS payer,
        CASE WHEN dn.visit_id IS NOT NULL THEN 1 ELSE 0 END AS has_consult,
        CASE WHEN ei.visit_id IS NOT NULL THEN 1 ELSE 0 END AS has_investigation,
        CASE WHEN rx.visit_id IS NOT NULL THEN 1 ELSE 0 END AS has_rx,
        CASE WHEN ia.visit_id IS NOT NULL THEN 1 ELSE 0 END AS has_admission
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN retained r ON v.patient = r.patient AND v.source_schema = r.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES dn
        ON v.id = dn.visit_id AND v.source_schema = dn.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS ei
        ON v.id = ei.visit_id AND v.source_schema = ei.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_PRESCRIPTION_PAYMENTS rx
        ON v.id = rx.visit_id AND v.source_schema = rx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS ia
        ON v.id = ia.visit_id AND v.source_schema = ia.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
)
SELECT
    payer,
    COUNT(DISTINCT patient)                                              AS retained_patients,
    COUNT(DISTINCT visit_id)                                             AS total_visits,
    ROUND(DIV0(SUM(has_consult),   COUNT(DISTINCT visit_id)) * 100, 1)  AS consult_rate_pct,
    ROUND(DIV0(SUM(has_investigation), COUNT(DISTINCT visit_id))*100,1) AS investigation_rate_pct,
    ROUND(DIV0(SUM(has_rx),        COUNT(DISTINCT visit_id)) * 100, 1)  AS rx_rate_pct,
    ROUND(DIV0(SUM(has_admission), COUNT(DISTINCT visit_id)) * 100, 1)  AS admission_rate_pct,
    ROUND(DIV0(
        SUM(CASE WHEN has_rx = 1 AND has_consult = 0 AND has_investigation = 0
                 AND has_admission = 0 THEN 1 ELSE 0 END),
        COUNT(DISTINCT visit_id)
    ) * 100, 1)                                                          AS pharmacy_only_pct
FROM visit_services
GROUP BY 1
ORDER BY retained_patients DESC
"""
    return run_query(sql)


def load_cost_dropout_correlation(filters: dict, run_query) -> pd.DataFrame:
    """R3/R4: Medication cost vs LTFU for cash patients; investigation waits for insured.
    Returns one row per payer × lifecycle with cost and wait metrics."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
chronic_flag AS (
    SELECT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE dx.disease_burden_group_1 ILIKE ANY ('%Cardiovascular%','%Diabetes%',
          '%Chronic%','%HIV%','%Mental%','%Neurolog%')
      AND v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
    GROUP BY 1, 2
),
patient_last AS (
    SELECT v.source_schema, v.patient,
           MAX(v.created_at) AS last_visit,
           MAX(sa.max_date)  AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN chronic_flag cf ON v.patient = cf.patient AND v.source_schema = cf.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
    GROUP BY 1, 2
),
visit_level AS (
    SELECT
        v.source_schema, v.patient, v.id AS visit_id,
        CASE
            WHEN UPPER(v.payment_mode) IN ('CASH','PRIVATE') THEN 'Cash'
            WHEN UPPER(v.payment_mode) LIKE '%NHIF%' OR UPPER(v.payment_mode) LIKE '%SHA%' THEN 'NHIF / SHA'
            ELSE 'Insurance / Corporate'
        END AS payer,
        CASE WHEN ia.visit_id IS NOT NULL THEN 'Inpatient' ELSE 'Outpatient' END AS visit_type,
        CASE
            WHEN DATEDIFF('day', pl.last_visit, pl.max_date) > 180 THEN 'LTFU (>180d)'
            WHEN DATEDIFF('day', pl.last_visit, pl.max_date) > 90  THEN 'Lapsing (91-180d)'
            ELSE 'Active (≤90d)'
        END AS lifecycle,
        NULL::FLOAT              AS rx_amount,
        COALESCE(ili.visit_rev, 0) AS visit_rev,
        DATEDIFF('minute',
            ei.investigation_created_at, ei.result_created_at) AS inv_wait_mins
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN patient_last pl ON v.patient = pl.patient AND v.source_schema = pl.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS ia
        ON v.id = ia.visit_id AND v.source_schema = ia.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_PRESCRIPTION_PAYMENTS rx
        ON v.id = rx.visit_id AND v.source_schema = rx.source_schema
    LEFT JOIN (
        SELECT visit_id, source_schema, SUM(item_amount) AS visit_rev
        FROM HOSPITALS.STAGING.STG_INVOICE_LINE_ITEMS
        WHERE invoice_deleted_at IS NULL AND (auto_cancelled IS NULL OR auto_cancelled = 0)
        GROUP BY 1, 2
    ) ili ON v.id = ili.visit_id AND v.source_schema = ili.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS ei
        ON v.id = ei.visit_id AND v.source_schema = ei.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND ia.visit_id IS NULL
)
SELECT
    payer,
    lifecycle,
    COUNT(DISTINCT patient)                             AS patient_count,
    ROUND(AVG(NULLIF(rx_amount, 0)), 0)                AS avg_rx_cost,
    ROUND(AVG(NULLIF(visit_rev, 0)), 0)                AS avg_invoice_size,
    ROUND(AVG(NULLIF(inv_wait_mins, 0)) / 60.0, 1)    AS avg_inv_wait_hrs,
    COUNT(DISTINCT CASE WHEN inv_wait_mins IS NOT NULL THEN visit_id END) AS visits_with_inv
FROM visit_level
GROUP BY 1, 2
ORDER BY payer, lifecycle
"""
    return run_query(sql)


def load_investigation_parity(filters: dict, run_query) -> pd.DataFrame:
    """R5: Cash vs insured diagnostic investigations per visit for the same diagnosis."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
visit_inv AS (
    SELECT
        v.source_schema, v.id AS visit_id,
        COALESCE(dx.disease_burden_group_1, 'Unclassified') AS condition,
        CASE
            WHEN UPPER(v.payment_mode) IN ('CASH','PRIVATE') THEN 'Cash'
            WHEN UPPER(v.payment_mode) LIKE '%NHIF%' OR UPPER(v.payment_mode) LIKE '%SHA%' THEN 'NHIF / SHA'
            ELSE 'Insurance / Corporate'
        END AS payer,
        CASE WHEN ia.visit_id IS NOT NULL THEN 'Inpatient' ELSE 'Outpatient' END AS visit_type,
        COUNT(ei.investigation_type) AS inv_count
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS ei
        ON v.id = ei.visit_id AND v.source_schema = ei.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS ia
        ON v.id = ia.visit_id AND v.source_schema = ia.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
    GROUP BY 1, 2, 3, 4, 5
)
SELECT
    condition,
    visit_type,
    payer,
    COUNT(DISTINCT visit_id)          AS visit_count,
    ROUND(AVG(inv_count), 2)          AS avg_inv_per_visit
FROM visit_inv
WHERE condition != 'Unclassified'
GROUP BY 1, 2, 3
HAVING visit_count >= 10
ORDER BY condition, visit_type, payer
LIMIT 60
"""
    return run_query(sql)


def load_ltfu_peak_hour_analysis(filters: dict, run_query) -> pd.DataFrame:
    """R6: For 180-day LTFU insured patients — were their final visits on peak days/hours?"""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
chronic_flag AS (
    SELECT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE dx.disease_burden_group_1 ILIKE ANY ('%Cardiovascular%','%Diabetes%',
          '%Chronic%','%HIV%','%Mental%','%Neurolog%')
      AND v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
    GROUP BY 1, 2
),
daily_volume AS (
    SELECT
        v.source_schema,
        DATE_TRUNC('day', v.created_at) AS visit_day,
        COUNT(DISTINCT v.id)            AS day_vol
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
    GROUP BY 1, 2
),
peak_days AS (
    SELECT source_schema, visit_day
    FROM daily_volume
    QUALIFY day_vol >= PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY day_vol)
                       OVER (PARTITION BY source_schema)
),
ltfu_insured AS (
    SELECT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN chronic_flag cf ON v.patient = cf.patient AND v.source_schema = cf.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      AND UPPER(v.payment_mode) NOT IN ('CASH','PRIVATE')
      {wh}
    GROUP BY 1, 2, sa.max_date
    HAVING DATEDIFF('day', MAX(v.created_at), MAX(sa.max_date)) > 180
),
last_visits AS (
    SELECT
        v.source_schema, v.patient,
        MAX(v.created_at) AS last_visit_ts,
        DATE_TRUNC('day', MAX(v.created_at)) AS last_visit_day,
        HOUR(CONVERT_TIMEZONE('UTC','Africa/Nairobi', MAX(v.created_at))) AS last_visit_hour,
        DAYOFWEEK(MAX(v.created_at)) AS day_of_week
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN ltfu_insured li ON v.patient = li.patient AND v.source_schema = li.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
    GROUP BY 1, 2
)
SELECT
    COUNT(DISTINCT lv.patient)                                  AS total_ltfu_insured,
    COUNT(DISTINCT CASE WHEN pd.visit_day IS NOT NULL
                   THEN lv.patient END)                         AS last_visit_on_peak_day,
    ROUND(DIV0(COUNT(DISTINCT CASE WHEN pd.visit_day IS NOT NULL
               THEN lv.patient END),
               COUNT(DISTINCT lv.patient)) * 100, 1)            AS pct_on_peak_day,
    COUNT(DISTINCT CASE WHEN lv.last_visit_hour BETWEEN 7 AND 10
                   THEN lv.patient END)                         AS morning_rush_patients,
    COUNT(DISTINCT CASE WHEN lv.day_of_week IN (2, 6)
                   THEN lv.patient END)                         AS mon_fri_patients,
    ROUND(AVG(lv.last_visit_hour), 1)                           AS avg_final_visit_hour
FROM last_visits lv
LEFT JOIN peak_days pd
    ON lv.source_schema = pd.source_schema AND lv.last_visit_day = pd.visit_day
"""
    return run_query(sql)


def load_insured_surge_followup(filters: dict, run_query) -> pd.DataFrame:
    """R7: Do insured patients return less frequently during high-volume months?"""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
monthly_vol AS (
    SELECT
        v.source_schema,
        DATE_TRUNC('month', v.created_at) AS visit_month,
        COUNT(DISTINCT v.id)               AS month_vol
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
    GROUP BY 1, 2
),
monthly_z AS (
    SELECT
        source_schema, visit_month, month_vol,
        AVG(month_vol) OVER (PARTITION BY source_schema) AS avg_vol,
        STDDEV(month_vol) OVER (PARTITION BY source_schema) AS std_vol
    FROM monthly_vol
),
surge_months AS (
    SELECT source_schema, visit_month,
           CASE WHEN (month_vol - avg_vol) / NULLIF(std_vol, 0) > 1 THEN 1 ELSE 0 END AS is_surge
    FROM monthly_z
),
insured_gaps AS (
    SELECT
        v.source_schema,
        v.patient,
        DATE_TRUNC('month', v.created_at) AS visit_month,
        DATEDIFF('day', v.created_at,
            LEAD(v.created_at) OVER (PARTITION BY v.source_schema, v.patient
                                     ORDER BY v.created_at)) AS days_to_next
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE UPPER(v.payment_mode) NOT IN ('CASH','PRIVATE')
      AND v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
)
SELECT
    sm.visit_month,
    MAX(sm.is_surge)                           AS is_surge_month,
    MAX(mv.month_vol)                          AS total_visits,
    COUNT(DISTINCT ig.patient)                 AS insured_patients_with_gap,
    ROUND(AVG(ig.days_to_next), 1)             AS avg_days_to_next_visit
FROM surge_months sm
LEFT JOIN monthly_vol mv
    ON sm.source_schema = mv.source_schema AND sm.visit_month = mv.visit_month
LEFT JOIN insured_gaps ig
    ON sm.source_schema = ig.source_schema AND sm.visit_month = ig.visit_month
    AND ig.days_to_next IS NOT NULL
WHERE sm.visit_month >= DATEADD('month', -{mo}, CURRENT_DATE)
GROUP BY sm.visit_month
ORDER BY sm.visit_month
"""
    return run_query(sql)


def load_low_engagement_revenue_risk(filters: dict, run_query) -> pd.DataFrame:
    """R8: What % of 180-day LTFU revenue risk comes from patients with only 1-2 visits?"""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
chronic_flag AS (
    SELECT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE dx.disease_burden_group_1 ILIKE ANY ('%Cardiovascular%','%Diabetes%',
          '%Chronic%','%HIV%','%Mental%','%Neurolog%')
      AND v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
    GROUP BY 1, 2
),
patient_summary AS (
    SELECT
        v.source_schema, v.patient,
        COUNT(DISTINCT v.id)          AS total_visits,
        MAX(v.created_at)             AS last_visit,
        MAX(sa.max_date)              AS max_date,
        SUM(COALESCE(ili.item_amount, 0))  AS total_revenue
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN chronic_flag cf ON v.patient = cf.patient AND v.source_schema = cf.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_INVOICE_LINE_ITEMS ili
        ON v.id = ili.visit_id AND v.source_schema = ili.source_schema
        AND ili.invoice_deleted_at IS NULL AND (ili.auto_cancelled IS NULL OR ili.auto_cancelled = 0)
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
    GROUP BY 1, 2
)
SELECT
    CASE WHEN total_visits <= 2 THEN '1–2 Visits' ELSE '3+ Visits' END AS engagement_tier,
    COUNT(DISTINCT patient)                                               AS ltfu_patients,
    ROUND(AVG(DIV0(total_revenue, total_visits)), 0)                     AS avg_rev_per_visit,
    ROUND(COUNT(DISTINCT patient) * 4
          * AVG(DIV0(total_revenue, total_visits)), 0)                    AS revenue_at_risk
FROM patient_summary
WHERE DATEDIFF('day', last_visit, max_date) > 180
GROUP BY 1
ORDER BY ltfu_patients DESC
"""
    return run_query(sql)


def load_clinician_ltfu_rate(filters: dict, run_query) -> pd.DataFrame:
    """R9: Which clinicians have the highest rate of chronic patients crossing 180-day LTFU?"""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
chronic_flag AS (
    SELECT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE dx.disease_burden_group_1 ILIKE ANY ('%Cardiovascular%','%Diabetes%',
          '%Chronic%','%HIV%','%Mental%','%Neurolog%')
      AND v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
    GROUP BY 1, 2
),
patient_last AS (
    SELECT v.source_schema, v.patient,
           MAX(v.created_at) AS last_visit,
           MAX(sa.max_date)  AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN chronic_flag cf ON v.patient = cf.patient AND v.source_schema = cf.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
    GROUP BY 1, 2
),
clinician_chronic AS (
    SELECT
        v.user AS clinician,
        v.patient,
        MAX(CASE WHEN DATEDIFF('day', pl.last_visit, pl.max_date) > 180
                 THEN 1 ELSE 0 END) AS is_ltfu
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN chronic_flag cf ON v.patient = cf.patient AND v.source_schema = cf.source_schema
    INNER JOIN patient_last pl ON v.patient = pl.patient AND v.source_schema = pl.source_schema
    WHERE v.user IS NOT NULL
      AND v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
    GROUP BY 1, 2
)
SELECT
    clinician,
    COUNT(DISTINCT patient)                              AS chronic_patients_seen,
    SUM(is_ltfu)                                        AS ltfu_count,
    ROUND(DIV0(SUM(is_ltfu), COUNT(DISTINCT patient)) * 100, 1) AS ltfu_rate_pct
FROM clinician_chronic
GROUP BY 1
HAVING chronic_patients_seen >= 5
ORDER BY ltfu_rate_pct DESC
LIMIT 20
"""
    return run_query(sql)


import pandas as pd


def _w(filters: dict, alias: str = "v") -> str:
    parts = []
    if filters.get("schema"):
        parts.append(f"AND {alias}.source_schema = '{filters['schema']}'")
    if filters.get("facility"):
        parts.append(f"AND {alias}.clinic = '{filters['facility']}'")
    return "\n    ".join(parts)


def _mo(filters: dict) -> int:
    return filters.get("months_back", 12)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DISEASE BURDEN
# ══════════════════════════════════════════════════════════════════════════════

def load_burden_kpis(filters: dict, run_query) -> pd.DataFrame:
    """A1: Burden KPI snapshot — diagnosed visits, comorbidity rate, group shares, undetected NCD."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
period_visits AS (
    SELECT v.source_schema, v.id AS visit_id, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
),
dx_all AS (
    SELECT source_schema, visit_id, disease_burden_group_1 AS bg, is_chronic_1 AS ic
    FROM HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED
    WHERE icd10_code_1 IS NOT NULL
    UNION ALL
    SELECT source_schema, visit_id, disease_burden_group_2, is_chronic_2
    FROM HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED
    WHERE icd10_code_2 IS NOT NULL
),
patient_dx_count AS (
    SELECT dx.source_schema, pv.patient,
           COUNT(DISTINCT dx.bg) AS dg_count
    FROM period_visits pv
    INNER JOIN dx_all dx
        ON dx.visit_id = pv.visit_id AND dx.source_schema = pv.source_schema
    GROUP BY 1, 2
),
at_risk AS (
    SELECT DISTINCT pv.source_schema, pv.visit_id
    FROM period_visits pv
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VITALS vt
        ON vt.visit_id = pv.visit_id AND vt.source_schema = pv.source_schema
    WHERE (vt.bp_systolic >= 140 OR vt.bp_diastolic >= 90 OR vt.blood_sugar >= 200)
      AND NOT EXISTS (
          SELECT 1 FROM dx_all dx2
          WHERE dx2.visit_id = pv.visit_id AND dx2.source_schema = pv.source_schema
            AND (LOWER(dx2.bg) LIKE '%ncd%' OR LOWER(dx2.bg) LIKE '%chronic%')
      )
)
SELECT
    COUNT(DISTINCT pv.visit_id)                             AS total_diagnosed,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN pc.dg_count >= 2 THEN pv.patient END),
        COUNT(DISTINCT pv.patient)
    ) * 100, 1)                                             AS comorbidity_rate_pct,
    ROUND(DIV0(
        COUNT(DISTINCT CASE
            WHEN LOWER(dx.bg) LIKE '%ncd%' OR LOWER(dx.bg) LIKE '%chronic%'
            THEN pv.visit_id END),
        COUNT(DISTINCT pv.visit_id)
    ) * 100, 1)                                             AS ncd_share_pct,
    ROUND(DIV0(
        COUNT(DISTINCT CASE
            WHEN LOWER(dx.bg) LIKE '%communicable%' THEN pv.visit_id END),
        COUNT(DISTINCT pv.visit_id)
    ) * 100, 1)                                             AS communicable_share_pct,
    ROUND(DIV0(
        COUNT(DISTINCT CASE
            WHEN LOWER(dx.bg) LIKE '%rmnch%' OR LOWER(dx.bg) LIKE '%maternal%'
            THEN pv.visit_id END),
        COUNT(DISTINCT pv.visit_id)
    ) * 100, 1)                                             AS rmnch_share_pct,
    COUNT(DISTINCT ar.visit_id)                             AS undetected_ncd
FROM period_visits pv
INNER JOIN dx_all dx ON dx.visit_id = pv.visit_id AND dx.source_schema = pv.source_schema
LEFT JOIN patient_dx_count pc
    ON pv.patient = pc.patient AND pv.source_schema = pc.source_schema
LEFT JOIN at_risk ar ON ar.visit_id = pv.visit_id AND ar.source_schema = pv.source_schema
"""
    return run_query(sql)


def load_burden_trend(filters: dict, run_query) -> pd.DataFrame:
    """A2: Burden group monthly trend — for stacked area chart."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
)
SELECT
    DATE_TRUNC('month', v.created_at)                   AS visit_month,
    COALESCE(dx.disease_burden_group_1, 'Unclassified') AS burden_group,
    COUNT(DISTINCT v.id)                                AS visit_count
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
    ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
  AND dx.icd10_code_1 IS NOT NULL
{wh}
GROUP BY 1, 2
ORDER BY 1
"""
    return run_query(sql)


def load_top_diagnoses(filters: dict, run_query) -> pd.DataFrame:
    """A3: Top 10 diagnoses by visit count."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
)
SELECT
    dx.disease_group_1                                  AS disease_group,
    COUNT(DISTINCT v.id)                                AS visit_count
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
    ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
  AND dx.icd10_code_1 IS NOT NULL
  AND dx.disease_group_1 IS NOT NULL
{wh}
GROUP BY 1
ORDER BY 2 DESC
LIMIT 10
"""
    return run_query(sql)


def load_undetected_ncd(filters: dict, run_query) -> pd.DataFrame:
    """A6: Elevated vitals without a coded NCD diagnosis, by age group."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
elevated AS (
    SELECT DISTINCT v.source_schema, v.id AS visit_id,
        CASE WHEN p.dob IS NULL THEN 'Unknown'
             WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 5  THEN 'Toddler (0–4)'
             WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 13 THEN 'Child (5–12)'
             WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 18 THEN 'Adolescent (13–17)'
             WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 25 THEN 'Youth (18–24)'
             WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 35 THEN 'Young Adult (25–34)'
             WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 45 THEN 'Adult (35–44)'
             WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 55 THEN 'Middle Age (45–54)'
             WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 65 THEN 'Older Adult (55–64)'
             ELSE 'Senior (65+)'
        END AS age_group,
        UPPER(COALESCE(p.sex, 'Unknown')) AS sex
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VITALS vt
        ON vt.visit_id = v.id AND vt.source_schema = v.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS p
        ON v.patient = p.patient_id AND v.source_schema = p.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND (vt.bp_systolic >= 140 OR vt.bp_diastolic >= 90 OR vt.blood_sugar >= 200)
),
ncd_coded AS (
    SELECT DISTINCT source_schema, visit_id
    FROM HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED
    WHERE is_chronic_1 = 1 OR is_chronic_2 = 1 OR is_chronic_3 = 1
)
SELECT age_group,
    COUNT(DISTINCT ev.visit_id)                         AS elevated_visits,
    COUNT(DISTINCT CASE WHEN nc.visit_id IS NULL
                        THEN ev.visit_id END)            AS undetected,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN nc.visit_id IS NULL THEN ev.visit_id END),
        COUNT(DISTINCT ev.visit_id)
    ) * 100, 1)                                         AS undetected_pct
FROM elevated ev
LEFT JOIN ncd_coded nc
    ON ev.visit_id = nc.visit_id AND ev.source_schema = nc.source_schema
WHERE age_group != 'Unknown'
GROUP BY 1
ORDER BY undetected_pct DESC
"""
    return run_query(sql)


def load_ncd_kpis(filters: dict, run_query) -> pd.DataFrame:
    """B1: NCD KPI snapshot — patients, controlled HTN %, comorbidity rate, undetected NCD."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
ncd_patients AS (
    SELECT DISTINCT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND dx.disease_burden_group_1 ILIKE ANY (
          '%Cardiovascular%','%Diabetes%','%Endocrin%',
          '%Neurolog%','%Mental%','%Musculo%','%Chronic%')
),
multi_ncd AS (
    SELECT v.source_schema, v.patient,
           COUNT(DISTINCT
               CASE
                   WHEN dx.disease_burden_group_1 ILIKE '%Endocrin%'
                     OR dx.disease_burden_group_1 ILIKE '%Diabetes%' THEN 'Diabetes_Metabolic'
                   ELSE dx.disease_burden_group_1
               END
           ) AS distinct_ncds
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND dx.disease_burden_group_1 ILIKE ANY (
          '%Cardiovascular%','%Diabetes%','%Endocrin%',
          '%Neurolog%','%Mental%','%Musculo%','%Chronic%')
    GROUP BY 1, 2
),
htn_bp AS (
    SELECT v.source_schema, v.patient,
        AVG(vt.bp_systolic)  AS avg_sys,
        AVG(vt.bp_diastolic) AS avg_dia
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VITALS vt
        ON vt.visit_id = v.id AND vt.source_schema = v.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND (dx.disease_group_1 ILIKE '%hypertension%' OR dx.disease_group_1 ILIKE '%HTN%')
      AND vt.bp_systolic IS NOT NULL
    GROUP BY 1, 2
),
undetected AS (
    SELECT DISTINCT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VITALS vt
        ON vt.visit_id = v.id AND vt.source_schema = v.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND (vt.bp_systolic >= 140 OR vt.blood_sugar >= 10)
    QUALIFY COUNT(DISTINCT v.id) OVER (PARTITION BY v.source_schema, v.patient) >= 2
)
SELECT
    COUNT(DISTINCT np.patient)                                   AS ncd_patients,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN mn.distinct_ncds >= 2 THEN np.patient END),
        COUNT(DISTINCT np.patient)
    ) * 100, 1)                                                  AS comorbidity_rate_pct,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN hb.avg_sys < 140 AND hb.avg_dia < 90
                            THEN hb.patient END),
        NULLIF(COUNT(DISTINCT hb.patient), 0)
    ) * 100, 1)                                                  AS controlled_htn_pct,
    COUNT(DISTINCT u.patient)                                    AS undetected_ncd_patients
FROM ncd_patients np
LEFT JOIN multi_ncd mn  ON np.patient = mn.patient AND np.source_schema = mn.source_schema
LEFT JOIN htn_bp hb     ON np.patient = hb.patient AND np.source_schema = hb.source_schema
LEFT JOIN undetected u  ON np.patient = u.patient  AND np.source_schema = u.source_schema
"""
    return run_query(sql)


def load_ncd_by_age(filters: dict, run_query) -> pd.DataFrame:
    """B2: NCD patient count by age group and condition."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
)
SELECT
    CASE WHEN p.dob IS NULL THEN 'Unknown'
         WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 5  THEN 'Toddler (0–4)'
         WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 13 THEN 'Child (5–12)'
         WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 18 THEN 'Adolescent (13–17)'
         WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 25 THEN 'Youth (18–24)'
         WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 35 THEN 'Young Adult (25–34)'
         WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 45 THEN 'Adult (35–44)'
         WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 55 THEN 'Middle Age (45–54)'
         WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 65 THEN 'Older Adult (55–64)'
         ELSE 'Senior (65+)'
    END                                                     AS age_group,
    COALESCE(dx.disease_group_1, 'Unspecified')             AS chronic_condition,
    COUNT(DISTINCT v.patient)                               AS patient_count
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
    ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS p
    ON v.patient = p.patient_id AND v.source_schema = p.source_schema
WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
  AND dx.is_chronic_1 = 1
  AND age_group != 'Unknown'
{wh}
GROUP BY 1, 2
ORDER BY 1, patient_count DESC
"""
    return run_query(sql)


def load_htn_controlled(filters: dict, run_query) -> pd.DataFrame:
    """B5: Hypertension controlled vs uncontrolled."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    {wsa}
    GROUP BY source_schema
),
patient_bp AS (
    SELECT
        v.source_schema,
        v.patient,
        AVG(vt.bp_systolic)  AS avg_systolic,
        AVG(vt.bp_diastolic) AS avg_diastolic
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_VITALS vt
        ON vt.visit_id = v.id AND vt.source_schema = v.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND LOWER(COALESCE(dx.disease_group_1, '')) LIKE '%hypertension%'
    GROUP BY v.source_schema, v.patient
)
SELECT
    CASE WHEN avg_systolic IS NULL                            THEN 'No BP Recorded'
         WHEN avg_systolic < 140 AND avg_diastolic < 90      THEN 'Controlled'
         ELSE 'Uncontrolled'
    END                                                       AS htn_status,
    COUNT(DISTINCT patient)                                   AS patient_count,
    ROUND(AVG(avg_systolic), 0)                               AS avg_systolic
FROM patient_bp
GROUP BY 1
ORDER BY patient_count DESC
"""
    return run_query(sql)


def load_anc_funnel(filters: dict, run_query) -> pd.DataFrame:
    """C2: ANC funnel — ANC1 through ANC4 completion."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
anc AS (
    SELECT v.source_schema, v.patient, v.created_at,
        ROW_NUMBER() OVER (
            PARTITION BY v.source_schema, v.patient ORDER BY v.created_at
        ) AS anc_seq
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND (dx.icd10_code_1 LIKE 'Z34%' OR dx.icd10_code_2 LIKE 'Z34%')
)
SELECT
    COUNT(DISTINCT CASE WHEN anc_seq >= 1 THEN patient END) AS anc1,
    COUNT(DISTINCT CASE WHEN anc_seq >= 2 THEN patient END) AS anc2,
    COUNT(DISTINCT CASE WHEN anc_seq >= 3 THEN patient END) AS anc3,
    COUNT(DISTINCT CASE WHEN anc_seq >= 4 THEN patient END) AS anc4,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN anc_seq >= 4 THEN patient END),
        COUNT(DISTINCT CASE WHEN anc_seq >= 1 THEN patient END)
    ) * 100, 1)                                             AS anc4_completion_pct
FROM anc
"""
    return run_query(sql)


def load_deliveries_by_age(filters: dict, run_query) -> pd.DataFrame:
    """C3: Deliveries by maternal age group."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
)
SELECT
    CASE WHEN p.dob IS NULL THEN 'Unknown'
         WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 18 THEN 'Adolescent (<18)'
         WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 25 THEN 'Young Adult (18–24)'
         WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 35 THEN 'Adult (25–34)'
         WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 45 THEN 'Older Adult (35–44)'
         ELSE 'Advanced Maternal Age (45+)'
    END                                                     AS maternal_age_group,
    COUNT(DISTINCT v.id)                                    AS delivery_count
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
    ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS p
    ON v.patient = p.patient_id AND v.source_schema = p.source_schema
WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
  {wh}
  AND (dx.icd10_code_1 LIKE 'O8%' OR dx.icd10_code_2 LIKE 'O8%')
GROUP BY 1
ORDER BY delivery_count DESC
"""
    return run_query(sql)


def load_anc_dropout_by_payer(filters: dict, run_query) -> pd.DataFrame:
    """RMNCH: ANC completion funnel by payer — why patients don't complete ANC4."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
anc_seq AS (
    SELECT v.source_schema, v.patient,
           ROW_NUMBER() OVER (
               PARTITION BY v.source_schema, v.patient ORDER BY v.created_at
           ) AS anc_num,
           CASE WHEN UPPER(v.payment_mode) IN ('CASH','PRIVATE','SELF PAY','OUT-OF-POCKET')
                THEN 'Cash'
                WHEN UPPER(v.payment_mode) LIKE '%NHIF%'
                  OR UPPER(v.payment_mode) LIKE '%SHA%'
                  OR UPPER(v.payment_mode) LIKE '%SHIF%' THEN 'NHIF / SHA'
                ELSE 'Insurance / Corporate'
           END AS payer_type
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND (dx.icd10_code_1 LIKE 'Z34%' OR dx.icd10_code_2 LIKE 'Z34%'
        OR dx.icd10_code_1 LIKE 'Z35%' OR dx.icd10_code_2 LIKE 'Z35%')
),
first_payer AS (
    SELECT source_schema, patient, payer_type
    FROM anc_seq WHERE anc_num = 1
),
max_stage AS (
    SELECT source_schema, patient, MAX(anc_num) AS max_reached
    FROM anc_seq GROUP BY 1, 2
)
SELECT
    fp.payer_type,
    COUNT(DISTINCT fp.patient)                                                AS total_anc1_patients,
    SUM(CASE WHEN ms.max_reached >= 2 THEN 1 ELSE 0 END)                     AS reached_anc2,
    SUM(CASE WHEN ms.max_reached >= 3 THEN 1 ELSE 0 END)                     AS reached_anc3,
    SUM(CASE WHEN ms.max_reached >= 4 THEN 1 ELSE 0 END)                     AS reached_anc4,
    ROUND(DIV0(SUM(CASE WHEN ms.max_reached >= 2 THEN 1 ELSE 0 END),
               COUNT(DISTINCT fp.patient)) * 100, 1)                          AS anc2_ret_pct,
    ROUND(DIV0(SUM(CASE WHEN ms.max_reached >= 4 THEN 1 ELSE 0 END),
               COUNT(DISTINCT fp.patient)) * 100, 1)                          AS anc4_completion_pct
FROM first_payer fp
INNER JOIN max_stage ms
    ON fp.patient = ms.patient AND fp.source_schema = ms.source_schema
GROUP BY 1
ORDER BY total_anc1_patients DESC
"""
    return run_query(sql)


def load_anc_patient_cohort_profile(filters: dict, run_query) -> pd.DataFrame:
    """RMNCH: Per-patient ANC journey — same patients or different at each stage?"""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
anc_seq AS (
    SELECT v.source_schema, v.patient,
           ROW_NUMBER() OVER (
               PARTITION BY v.source_schema, v.patient ORDER BY v.created_at
           ) AS anc_num,
           CASE WHEN UPPER(v.payment_mode) IN ('CASH','PRIVATE','SELF PAY','OUT-OF-POCKET')
                THEN 'Cash'
                WHEN UPPER(v.payment_mode) LIKE '%NHIF%'
                  OR UPPER(v.payment_mode) LIKE '%SHA%'
                  OR UPPER(v.payment_mode) LIKE '%SHIF%' THEN 'NHIF / SHA'
                ELSE 'Insurance / Corporate'
           END AS payer_type,
           CASE WHEN p.dob IS NULL THEN 'Unknown'
                WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 18 THEN 'Adolescent (<18)'
                WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 25 THEN '18-24'
                WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 35 THEN '25-34'
                ELSE '35+'
           END AS age_group
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS p
        ON v.patient = p.patient_id AND v.source_schema = p.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND (dx.icd10_code_1 LIKE 'Z34%' OR dx.icd10_code_2 LIKE 'Z34%'
        OR dx.icd10_code_1 LIKE 'Z35%' OR dx.icd10_code_2 LIKE 'Z35%')
),
first_visit AS (
    SELECT source_schema, patient, payer_type, age_group
    FROM anc_seq WHERE anc_num = 1
),
patient_summary AS (
    SELECT s.source_schema, s.patient,
           fv.payer_type, fv.age_group,
           MAX(s.anc_num) AS stages_completed
    FROM anc_seq s
    INNER JOIN first_visit fv
        ON s.patient = fv.patient AND s.source_schema = fv.source_schema
    GROUP BY 1, 2, 3, 4
)
SELECT
    CASE stages_completed
        WHEN 1 THEN '1 - Dropped after ANC1'
        WHEN 2 THEN '2 - Stopped at ANC2'
        WHEN 3 THEN '3 - Stopped at ANC3'
        ELSE '4 - Completed ANC4'
    END                     AS anc_journey,
    stages_completed,
    payer_type,
    age_group,
    COUNT(DISTINCT patient) AS patient_count
FROM patient_summary
GROUP BY 1, 2, 3, 4
ORDER BY 2, 3
"""
    return run_query(sql)


def load_anc_vs_delivery_pnc(filters: dict, run_query) -> pd.DataFrame:
    """RMNCH: Facility profile — ANC vs delivery vs PNC vs complications share."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
categorised AS (
    SELECT v.id AS visit_id, v.patient,
           CASE
               WHEN dx.icd10_code_1 LIKE 'Z34%' OR dx.icd10_code_2 LIKE 'Z34%'
                 OR dx.icd10_code_1 LIKE 'Z35%' OR dx.icd10_code_2 LIKE 'Z35%'
                 OR dx.icd10_code_1 LIKE 'Z36%' OR dx.icd10_code_2 LIKE 'Z36%' THEN 'Antenatal Care (ANC)'
               WHEN dx.icd10_code_1 LIKE 'O8%'  OR dx.icd10_code_2 LIKE 'O8%'  THEN 'Delivery'
               WHEN dx.icd10_code_1 LIKE 'Z39%' OR dx.icd10_code_2 LIKE 'Z39%' THEN 'Postnatal Care (PNC)'
               WHEN dx.icd10_code_1 LIKE 'O04%' OR dx.icd10_code_2 LIKE 'O04%'
                 OR dx.icd10_code_1 LIKE 'O05%' OR dx.icd10_code_2 LIKE 'O05%'
                 OR dx.icd10_code_1 LIKE 'O06%' OR dx.icd10_code_2 LIKE 'O06%' THEN 'Abortion / Miscarriage'
               WHEN dx.icd10_code_1 LIKE 'Z30%' OR dx.icd10_code_2 LIKE 'Z30%'
                 OR dx.icd10_code_1 LIKE 'Z31%' OR dx.icd10_code_2 LIKE 'Z31%' THEN 'Family Planning'
               WHEN (dx.icd10_code_1 LIKE 'O1%' OR dx.icd10_code_2 LIKE 'O1%'
                  OR dx.icd10_code_1 LIKE 'O2%' OR dx.icd10_code_2 LIKE 'O2%'
                  OR dx.icd10_code_1 LIKE 'O3%' OR dx.icd10_code_2 LIKE 'O3%'
                  OR dx.icd10_code_1 LIKE 'O4%' OR dx.icd10_code_2 LIKE 'O4%'
                  OR dx.icd10_code_1 LIKE 'O5%' OR dx.icd10_code_2 LIKE 'O5%'
                  OR dx.icd10_code_1 LIKE 'O6%' OR dx.icd10_code_2 LIKE 'O6%'
                  OR dx.icd10_code_1 LIKE 'O7%' OR dx.icd10_code_2 LIKE 'O7%') THEN 'Obstetric Complication'
               ELSE NULL
           END AS care_category
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
)
SELECT
    care_category,
    COUNT(DISTINCT visit_id) AS visit_count,
    COUNT(DISTINCT patient)  AS patient_count
FROM categorised
WHERE care_category IS NOT NULL
GROUP BY 1
ORDER BY visit_count DESC
"""
    return run_query(sql)


def load_pnc_profile(filters: dict, run_query) -> pd.DataFrame:
    """RMNCH: Postnatal care (PNC) visits — payer and age profile."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
)
SELECT
    CASE WHEN p.dob IS NULL THEN 'Unknown'
         WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 18 THEN 'Adolescent (<18)'
         WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 25 THEN '18-24'
         WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 35 THEN '25-34'
         ELSE '35+'
    END AS age_group,
    CASE WHEN UPPER(v.payment_mode) IN ('CASH','PRIVATE','SELF PAY','OUT-OF-POCKET') THEN 'Cash'
         WHEN UPPER(v.payment_mode) LIKE '%NHIF%'
           OR UPPER(v.payment_mode) LIKE '%SHA%'
           OR UPPER(v.payment_mode) LIKE '%SHIF%'                             THEN 'NHIF / SHA'
         ELSE 'Insurance / Corporate'
    END AS payer_type,
    COUNT(DISTINCT v.id)      AS visit_count,
    COUNT(DISTINCT v.patient) AS patient_count
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
    ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS p
    ON v.patient = p.patient_id AND v.source_schema = p.source_schema
WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
  {wh}
  AND (dx.icd10_code_1 LIKE 'Z39%' OR dx.icd10_code_2 LIKE 'Z39%')
GROUP BY 1, 2
ORDER BY visit_count DESC
"""
    return run_query(sql)


def load_high_risk_pregnancy_profile(filters: dict, run_query) -> pd.DataFrame:
    """RMNCH: High-risk pregnancy — clinical flags, age-based, vitals-detected."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
anc_patients AS (
    SELECT DISTINCT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND (dx.icd10_code_1 LIKE 'Z34%' OR dx.icd10_code_2 LIKE 'Z34%'
        OR dx.icd10_code_1 LIKE 'Z35%' OR dx.icd10_code_2 LIKE 'Z35%')
),
dx_risk AS (
    SELECT DISTINCT source_schema, patient, risk_type
    FROM (
        SELECT v.source_schema, v.patient,
               CASE
                   WHEN dx.icd10_code_1 LIKE 'O15%' OR dx.icd10_code_2 LIKE 'O15%' THEN 'Eclampsia'
                   WHEN dx.icd10_code_1 LIKE 'O14%' OR dx.icd10_code_2 LIKE 'O14%' THEN 'Pre-eclampsia'
                   WHEN dx.icd10_code_1 LIKE 'O13%' OR dx.icd10_code_2 LIKE 'O13%' THEN 'Gestational Hypertension'
                   WHEN dx.icd10_code_1 LIKE 'O10%' OR dx.icd10_code_2 LIKE 'O10%'
                     OR dx.icd10_code_1 LIKE 'O11%' OR dx.icd10_code_2 LIKE 'O11%' THEN 'Chronic HTN in Pregnancy'
                   WHEN dx.icd10_code_1 LIKE 'O24%' OR dx.icd10_code_2 LIKE 'O24%' THEN 'Gestational Diabetes'
                   WHEN dx.icd10_code_1 LIKE 'O35%' OR dx.icd10_code_2 LIKE 'O35%'
                     OR dx.icd10_code_1 LIKE 'O36%' OR dx.icd10_code_2 LIKE 'O36%' THEN 'Foetal Risk / Growth Concern'
                   WHEN dx.icd10_code_1 LIKE 'O20%' OR dx.icd10_code_2 LIKE 'O20%' THEN 'Haemorrhage in Pregnancy'
                   WHEN dx.icd10_code_1 LIKE 'Z35%' OR dx.icd10_code_2 LIKE 'Z35%' THEN 'Supervised High-Risk (Z35)'
                   ELSE NULL
               END AS risk_type
        FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
        INNER JOIN anc_patients ap
            ON v.patient = ap.patient AND v.source_schema = ap.source_schema
        INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
            ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
        WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
          {wh}
    ) sub
    WHERE risk_type IS NOT NULL
),
age_risk AS (
    SELECT DISTINCT ap.source_schema, ap.patient,
           CASE
               WHEN TIMESTAMPDIFF('year', p.dob, CURRENT_DATE) < 18 THEN 'Adolescent Pregnancy (<18)'
               WHEN TIMESTAMPDIFF('year', p.dob, CURRENT_DATE) >= 35 THEN 'Advanced Maternal Age (35+)'
           END AS risk_type
    FROM anc_patients ap
    INNER JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS p
        ON ap.patient = p.patient_id AND ap.source_schema = p.source_schema
    WHERE p.dob IS NOT NULL
      AND (TIMESTAMPDIFF('year', p.dob, CURRENT_DATE) < 18
        OR TIMESTAMPDIFF('year', p.dob, CURRENT_DATE) >= 35)
),
vitals_risk AS (
    SELECT DISTINCT ap.source_schema, ap.patient,
           'Elevated BP During ANC (Vitals)' AS risk_type
    FROM anc_patients ap
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON ap.patient = v.patient AND ap.source_schema = v.source_schema
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VITALS vt
        ON v.id = vt.visit_id AND v.source_schema = vt.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND vt.bp_systolic >= 140
    GROUP BY 1, 2, 3
    HAVING COUNT(DISTINCT v.id) >= 2
),
all_risk AS (
    SELECT source_schema, patient, risk_type FROM dx_risk
    UNION ALL
    SELECT source_schema, patient, risk_type FROM age_risk WHERE risk_type IS NOT NULL
    UNION ALL
    SELECT source_schema, patient, risk_type FROM vitals_risk
)
SELECT
    risk_type,
    COUNT(DISTINCT patient) AS patient_count
FROM all_risk
GROUP BY 1
ORDER BY patient_count DESC
"""
    return run_query(sql)


def load_high_risk_pregnancy_patients(filters: dict, run_query) -> pd.DataFrame:
    """RMNCH: High-risk patient list — age, payer, risk flags, visit frequency."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
anc_visits AS (
    SELECT v.source_schema, v.patient, v.id AS visit_id, v.created_at,
           CASE WHEN UPPER(v.payment_mode) IN ('CASH','PRIVATE','SELF PAY','OUT-OF-POCKET') THEN 'Cash'
                WHEN UPPER(v.payment_mode) LIKE '%NHIF%'
                  OR UPPER(v.payment_mode) LIKE '%SHA%'
                  OR UPPER(v.payment_mode) LIKE '%SHIF%' THEN 'NHIF / SHA'
                ELSE 'Insurance / Corporate'
           END AS payer_type
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND (dx.icd10_code_1 LIKE 'Z34%' OR dx.icd10_code_2 LIKE 'Z34%'
        OR dx.icd10_code_1 LIKE 'Z35%' OR dx.icd10_code_2 LIKE 'Z35%')
),
anc_summary AS (
    SELECT source_schema, patient,
           MAX(payer_type)          AS payer_type,
           COUNT(DISTINCT visit_id) AS anc_visits,
           MAX(created_at)          AS last_anc_date
    FROM anc_visits
    GROUP BY 1, 2
),
risk_flags AS (
    SELECT av.source_schema, av.patient,
           LISTAGG(DISTINCT
               CASE
                   WHEN dx.icd10_code_1 LIKE 'O15%' OR dx.icd10_code_2 LIKE 'O15%' THEN 'Eclampsia'
                   WHEN dx.icd10_code_1 LIKE 'O14%' OR dx.icd10_code_2 LIKE 'O14%' THEN 'Pre-eclampsia'
                   WHEN dx.icd10_code_1 LIKE 'O13%' OR dx.icd10_code_2 LIKE 'O13%' THEN 'Gest.HTN'
                   WHEN dx.icd10_code_1 LIKE 'O10%' OR dx.icd10_code_2 LIKE 'O10%'
                     OR dx.icd10_code_1 LIKE 'O11%' OR dx.icd10_code_2 LIKE 'O11%' THEN 'Chronic HTN'
                   WHEN dx.icd10_code_1 LIKE 'O24%' OR dx.icd10_code_2 LIKE 'O24%' THEN 'Gest.Diabetes'
                   WHEN dx.icd10_code_1 LIKE 'O35%' OR dx.icd10_code_2 LIKE 'O35%'
                     OR dx.icd10_code_1 LIKE 'O36%' OR dx.icd10_code_2 LIKE 'O36%' THEN 'Foetal Risk'
                   WHEN dx.icd10_code_1 LIKE 'O20%' OR dx.icd10_code_2 LIKE 'O20%' THEN 'Haemorrhage'
                   WHEN dx.icd10_code_1 LIKE 'Z35%' OR dx.icd10_code_2 LIKE 'Z35%' THEN 'Supervised HR'
                   ELSE NULL
               END, ' | '
           ) WITHIN GROUP (ORDER BY 1)                          AS risk_flags,
           1                                                    AS has_clinical_risk
    FROM anc_visits av
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON av.visit_id = dx.visit_id AND av.source_schema = dx.source_schema
    WHERE (dx.icd10_code_1 LIKE 'O1%' OR dx.icd10_code_2 LIKE 'O1%'
        OR dx.icd10_code_1 LIKE 'O2%' OR dx.icd10_code_2 LIKE 'O2%'
        OR dx.icd10_code_1 LIKE 'O24%' OR dx.icd10_code_2 LIKE 'O24%'
        OR dx.icd10_code_1 LIKE 'O35%' OR dx.icd10_code_2 LIKE 'O35%'
        OR dx.icd10_code_1 LIKE 'O36%' OR dx.icd10_code_2 LIKE 'O36%'
        OR dx.icd10_code_1 LIKE 'Z35%' OR dx.icd10_code_2 LIKE 'Z35%')
    GROUP BY 1, 2
),
demographics AS (
    SELECT p.patient_id, p.source_schema,
           CASE WHEN p.dob IS NULL THEN 'Unknown'
                WHEN TIMESTAMPDIFF('year', p.dob, CURRENT_DATE) < 18 THEN 'Adolescent (<18)'
                WHEN TIMESTAMPDIFF('year', p.dob, CURRENT_DATE) < 25 THEN '18-24'
                WHEN TIMESTAMPDIFF('year', p.dob, CURRENT_DATE) < 35 THEN '25-34'
                ELSE '35+'
           END AS age_group,
           CASE WHEN p.dob IS NULL THEN NULL
                ELSE TIMESTAMPDIFF('year', p.dob, CURRENT_DATE)
           END AS age_years,
           CASE WHEN p.dob IS NOT NULL
                  AND (TIMESTAMPDIFF('year', p.dob, CURRENT_DATE) < 18
                    OR TIMESTAMPDIFF('year', p.dob, CURRENT_DATE) >= 35)
                THEN 1 ELSE 0
           END AS age_risk_flag
    FROM HOSPITALS.STAGING.STG_RECEPTION_PATIENTS p
    INNER JOIN anc_summary asq ON p.patient_id = asq.patient AND p.source_schema = asq.source_schema
),
bp_risk AS (
    SELECT DISTINCT av.source_schema, av.patient, 1 AS bp_elevated
    FROM anc_visits av
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VITALS vt
        ON av.visit_id = vt.visit_id AND av.source_schema = vt.source_schema
    WHERE vt.bp_systolic >= 140
    GROUP BY 1, 2
    HAVING COUNT(*) >= 2
)
SELECT
    asq.patient,
    asq.source_schema,
    COALESCE(d.age_group, 'Unknown')    AS age_group,
    d.age_years,
    asq.payer_type,
    asq.anc_visits,
    DATEDIFF('day', asq.last_anc_date, sa.max_date) AS days_since_last_anc,
    COALESCE(rf.risk_flags, '')         AS risk_flags,
    COALESCE(rf.has_clinical_risk, 0)   AS clinical_risk,
    COALESCE(d.age_risk_flag, 0)        AS age_risk,
    COALESCE(bp.bp_elevated, 0)         AS vitals_risk
FROM anc_summary asq
INNER JOIN schema_anchor sa ON asq.source_schema = sa.source_schema
LEFT JOIN demographics d ON asq.patient = d.patient_id AND asq.source_schema = d.source_schema
LEFT JOIN risk_flags rf ON asq.patient = rf.patient AND asq.source_schema = rf.source_schema
LEFT JOIN bp_risk bp ON asq.patient = bp.patient AND asq.source_schema = bp.source_schema
WHERE COALESCE(rf.has_clinical_risk, 0) = 1
   OR COALESCE(d.age_risk_flag, 0) = 1
   OR COALESCE(bp.bp_elevated, 0) = 1
ORDER BY clinical_risk DESC, age_risk DESC, days_since_last_anc DESC
LIMIT 100
"""
    return run_query(sql)


def load_pregnancy_comorbidities(filters: dict, run_query) -> pd.DataFrame:
    """RMNCH: Other illnesses found in pregnant women during the same visit period."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
pregnant_patients AS (
    SELECT DISTINCT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND (dx.icd10_code_1 LIKE 'Z34%' OR dx.icd10_code_2 LIKE 'Z34%'
        OR dx.icd10_code_1 LIKE 'Z35%' OR dx.icd10_code_2 LIKE 'Z35%')
),
other_conditions AS (
    SELECT v.source_schema, v.patient,
           COALESCE(NULLIF(TRIM(dx.disease_group_1), ''), 'Unclassified') AS condition_group,
           dx.icd10_code_1
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN pregnant_patients pp
        ON v.patient = pp.patient AND v.source_schema = pp.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND dx.icd10_code_1 NOT LIKE 'Z34%'
      AND dx.icd10_code_1 NOT LIKE 'Z35%'
      AND dx.icd10_code_1 NOT LIKE 'Z36%'
      AND dx.icd10_code_1 NOT LIKE 'Z39%'
      AND dx.disease_group_1 IS NOT NULL
      AND TRIM(dx.disease_group_1) != ''
)
SELECT
    condition_group,
    COUNT(DISTINCT patient) AS patient_count,
    COUNT(*)                AS occurrence_count
FROM other_conditions
GROUP BY 1
HAVING patient_count >= 2
ORDER BY patient_count DESC
LIMIT 20
"""
    return run_query(sql)


def load_under5_profile(filters: dict, run_query) -> pd.DataFrame:
    """RMNCH: Under-5 visits — cause category, age bucket, admission rate."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
under5 AS (
    SELECT v.source_schema, v.id AS visit_id, v.patient,
           TIMESTAMPDIFF('month', p.dob, v.created_at) AS age_months,
           CASE
               WHEN TIMESTAMPDIFF('month', p.dob, v.created_at) < 12  THEN 'Infant (0-12m)'
               WHEN TIMESTAMPDIFF('month', p.dob, v.created_at) < 24  THEN 'Toddler (1-2y)'
               WHEN TIMESTAMPDIFF('month', p.dob, v.created_at) < 36  THEN 'Toddler (2-3y)'
               ELSE '3-5 years'
           END AS age_bucket
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS p
        ON v.patient = p.patient_id AND v.source_schema = p.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND p.dob IS NOT NULL
      AND TIMESTAMPDIFF('year', p.dob, v.created_at) < 5
      AND TIMESTAMPDIFF('year', p.dob, v.created_at) >= 0
),
with_category AS (
    SELECT u.source_schema, u.visit_id, u.patient, u.age_bucket,
           CASE
               WHEN dx.icd10_code_1 LIKE 'Z23%' OR dx.icd10_code_2 LIKE 'Z23%'
                 OR dx.icd10_code_1 LIKE 'Z24%' OR dx.icd10_code_2 LIKE 'Z24%'
                 OR dx.icd10_code_1 LIKE 'Z25%' OR dx.icd10_code_2 LIKE 'Z25%'
                 OR dx.icd10_code_1 LIKE 'Z26%' OR dx.icd10_code_2 LIKE 'Z26%'
                 OR dx.disease_group_1 ILIKE '%vaccin%'
                 OR dx.disease_group_1 ILIKE '%immuniz%'
                 OR dx.disease_group_1 ILIKE '%immunis%'                        THEN 'Vaccination / Immunisation'
               WHEN dx.icd10_code_1 LIKE 'J%'   OR dx.icd10_code_2 LIKE 'J%'
                 OR dx.icd10_code_1 LIKE 'A15%' OR dx.icd10_code_2 LIKE 'A15%' THEN 'Respiratory Illness'
               WHEN dx.icd10_code_1 LIKE 'B50%' OR dx.icd10_code_2 LIKE 'B50%'
                 OR dx.icd10_code_1 LIKE 'B51%' OR dx.icd10_code_2 LIKE 'B51%'
                 OR dx.icd10_code_1 LIKE 'B54%' OR dx.icd10_code_2 LIKE 'B54%'
                 OR dx.disease_group_1 ILIKE '%malaria%'                        THEN 'Malaria'
               WHEN dx.icd10_code_1 LIKE 'A0%'  OR dx.icd10_code_2 LIKE 'A0%'
                 OR dx.disease_group_1 ILIKE '%diarrhoe%'
                 OR dx.disease_group_1 ILIKE '%diarrhea%'
                 OR dx.disease_group_1 ILIKE '%gastro%'                         THEN 'Diarrhoea / GI'
               WHEN dx.icd10_code_1 LIKE 'E40%' OR dx.icd10_code_2 LIKE 'E40%'
                 OR dx.icd10_code_1 LIKE 'E41%' OR dx.icd10_code_2 LIKE 'E41%'
                 OR dx.icd10_code_1 LIKE 'E43%' OR dx.icd10_code_2 LIKE 'E43%'
                 OR dx.icd10_code_1 LIKE 'E44%' OR dx.icd10_code_2 LIKE 'E44%'
                 OR dx.disease_group_1 ILIKE '%malnutri%'                       THEN 'Malnutrition'
               WHEN dx.icd10_code_1 LIKE 'Z00%' OR dx.icd10_code_2 LIKE 'Z00%' THEN 'Well-child Check'
               ELSE 'Other Acute'
           END AS visit_category,
           ia.visit_id AS admitted_visit_id
    FROM under5 u
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON u.visit_id = dx.visit_id AND u.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS ia
        ON CAST(u.visit_id AS VARCHAR) = CAST(ia.visit_id AS VARCHAR)
    QUALIFY ROW_NUMBER() OVER (PARTITION BY u.visit_id ORDER BY dx.icd10_code_1 NULLS LAST) = 1
)
SELECT
    age_bucket,
    visit_category,
    COUNT(DISTINCT visit_id)                        AS visit_count,
    COUNT(DISTINCT patient)                         AS patient_count,
    COUNT(DISTINCT admitted_visit_id)               AS admitted_count,
    ROUND(DIV0(COUNT(DISTINCT admitted_visit_id),
               COUNT(DISTINCT visit_id)) * 100, 1) AS admission_rate_pct
FROM with_category
GROUP BY 1, 2
ORDER BY visit_count DESC
"""
    return run_query(sql)


def load_adolescent_rh_profile(filters: dict, run_query) -> pd.DataFrame:
    """RMNCH: Adolescent (10-19) reproductive health — family planning, pregnancy, complications."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
adolescents AS (
    SELECT v.source_schema, v.id AS visit_id, v.patient,
           CASE WHEN p.sex IS NULL THEN 'Unknown'
                ELSE UPPER(CAST(p.sex AS VARCHAR))
           END AS sex,
           CASE WHEN UPPER(v.payment_mode) IN ('CASH','PRIVATE','SELF PAY','OUT-OF-POCKET') THEN 'Cash'
                WHEN UPPER(v.payment_mode) LIKE '%NHIF%'
                  OR UPPER(v.payment_mode) LIKE '%SHA%'
                  OR UPPER(v.payment_mode) LIKE '%SHIF%' THEN 'NHIF / SHA'
                ELSE 'Insurance / Corporate'
           END AS payer_type
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS p
        ON v.patient = p.patient_id AND v.source_schema = p.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND p.dob IS NOT NULL
      AND TIMESTAMPDIFF('year', p.dob, v.created_at) BETWEEN 10 AND 19
),
with_category AS (
    SELECT a.source_schema, a.visit_id, a.patient, a.sex, a.payer_type,
           CASE
               WHEN dx.icd10_code_1 LIKE 'Z30%' OR dx.icd10_code_2 LIKE 'Z30%'
                 OR dx.icd10_code_1 LIKE 'Z31%' OR dx.icd10_code_2 LIKE 'Z31%'
                 OR dx.disease_group_1 ILIKE '%family plan%'
                 OR dx.disease_group_1 ILIKE '%contracepti%'                    THEN 'Family Planning / Contraception'
               WHEN dx.icd10_code_1 LIKE 'Z34%' OR dx.icd10_code_2 LIKE 'Z34%'
                 OR dx.icd10_code_1 LIKE 'Z35%' OR dx.icd10_code_2 LIKE 'Z35%' THEN 'ANC / Adolescent Pregnancy'
               WHEN dx.icd10_code_1 LIKE 'O04%' OR dx.icd10_code_2 LIKE 'O04%'
                 OR dx.icd10_code_1 LIKE 'O05%' OR dx.icd10_code_2 LIKE 'O05%'
                 OR dx.icd10_code_1 LIKE 'O06%' OR dx.icd10_code_2 LIKE 'O06%'
                 OR dx.disease_group_1 ILIKE '%abort%'                          THEN 'Abortion / Pregnancy Loss'
               WHEN dx.icd10_code_1 LIKE 'N70%' OR dx.icd10_code_2 LIKE 'N70%'
                 OR dx.icd10_code_1 LIKE 'N71%' OR dx.icd10_code_2 LIKE 'N71%'
                 OR dx.icd10_code_1 LIKE 'N72%' OR dx.icd10_code_2 LIKE 'N72%'
                 OR dx.icd10_code_1 LIKE 'N73%' OR dx.icd10_code_2 LIKE 'N73%'
                 OR dx.disease_group_1 ILIKE '%pelvic inflam%'
                 OR dx.disease_group_1 ILIKE '%PID%'                            THEN 'PID / Pelvic Inflammatory Disease'
               WHEN (dx.icd10_code_1 LIKE 'A5%' OR dx.icd10_code_2 LIKE 'A5%'
                  OR dx.icd10_code_1 LIKE 'A6%' OR dx.icd10_code_2 LIKE 'A6%')
                 AND (dx.icd10_code_1 NOT LIKE 'A5' OR dx.icd10_code_2 NOT LIKE 'A5') THEN 'STI'
               WHEN dx.icd10_code_1 LIKE 'Z39%' OR dx.icd10_code_2 LIKE 'Z39%' THEN 'Postnatal (PNC)'
               WHEN dx.icd10_code_1 LIKE 'O8%'  OR dx.icd10_code_2 LIKE 'O8%'  THEN 'Delivery'
               ELSE 'Other Gynaecology / General'
           END AS rh_category
    FROM adolescents a
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON a.visit_id = dx.visit_id AND a.source_schema = dx.source_schema
    QUALIFY ROW_NUMBER() OVER (PARTITION BY a.visit_id ORDER BY dx.icd10_code_1 NULLS LAST) = 1
)
SELECT
    rh_category,
    sex,
    payer_type,
    COUNT(DISTINCT visit_id) AS visit_count,
    COUNT(DISTINCT patient)  AS patient_count
FROM with_category
GROUP BY 1, 2, 3
ORDER BY patient_count DESC
"""
    return run_query(sql)


def load_rmnch_revenue_trend(filters: dict, run_query) -> pd.DataFrame:
    """RMNCH: Monthly revenue by segment — Maternal, Paediatric, Adolescent RH."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
categorised AS (
    SELECT v.source_schema, v.id AS visit_id, v.patient,
           DATE_TRUNC('month', v.created_at) AS visit_month,
           CASE
               WHEN dx.icd10_code_1 LIKE 'Z34%' OR dx.icd10_code_2 LIKE 'Z34%'
                 OR dx.icd10_code_1 LIKE 'Z35%' OR dx.icd10_code_2 LIKE 'Z35%'
                 OR dx.icd10_code_1 LIKE 'Z36%' OR dx.icd10_code_2 LIKE 'Z36%'
                 OR dx.icd10_code_1 LIKE 'Z39%' OR dx.icd10_code_2 LIKE 'Z39%'
                 OR dx.icd10_code_1 LIKE 'O%'   OR dx.icd10_code_2 LIKE 'O%' THEN 'Maternal'
               WHEN p.dob IS NOT NULL
                    AND TIMESTAMPDIFF('year', p.dob, v.created_at) BETWEEN 10 AND 19
                    AND (dx.icd10_code_1 LIKE 'Z30%' OR dx.icd10_code_1 LIKE 'Z34%'
                      OR dx.icd10_code_1 LIKE 'Z35%' OR dx.icd10_code_1 LIKE 'O04%'
                      OR dx.icd10_code_1 LIKE 'N70%' OR dx.icd10_code_1 LIKE 'N71%') THEN 'Adolescent RH'
               WHEN p.dob IS NOT NULL
                    AND TIMESTAMPDIFF('year', p.dob, v.created_at) < 12               THEN 'Paediatric (<12y)'
               ELSE NULL
           END AS rmnch_segment
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS p
        ON v.patient = p.patient_id AND v.source_schema = p.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
    QUALIFY ROW_NUMBER() OVER (PARTITION BY v.id ORDER BY dx.icd10_code_1 NULLS LAST) = 1
),
revenue AS (
    SELECT v.source_schema, v.id AS visit_id,
           COALESCE(SUM(il.item_amount), 0) AS visit_revenue
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_INVOICE_LINE_ITEMS il
        ON v.id = il.visit_id AND v.source_schema = il.source_schema
       AND il.invoice_deleted_at IS NULL
       AND (il.auto_cancelled IS NULL OR il.auto_cancelled = 0)
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
    GROUP BY 1, 2
)
SELECT
    c.visit_month,
    c.rmnch_segment,
    COUNT(DISTINCT c.visit_id)          AS visit_count,
    COUNT(DISTINCT c.patient)           AS patient_count,
    ROUND(SUM(r.visit_revenue), 0)      AS revenue
FROM categorised c
LEFT JOIN revenue r ON c.visit_id = r.visit_id AND c.source_schema = r.source_schema
WHERE c.rmnch_segment IS NOT NULL
GROUP BY 1, 2
ORDER BY 1, 2
"""
    return run_query(sql)


def load_communicable_trend(filters: dict, run_query) -> pd.DataFrame:
    """D2: Communicable disease monthly trend — top 6 groups."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
top6 AS (
    SELECT dx.disease_group_1 AS disease_group
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND dx.disease_burden_group_1 ILIKE '%communicable%'
      AND dx.disease_group_1 IS NOT NULL
    GROUP BY 1
    QUALIFY RANK() OVER (ORDER BY COUNT(DISTINCT v.id) DESC) <= 6
)
SELECT
    DATE_TRUNC('month', v.created_at)                   AS visit_month,
    dx.disease_group_1                                  AS disease_group,
    COUNT(DISTINCT v.id)                                AS visit_count
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
    ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
INNER JOIN top6 t6 ON dx.disease_group_1 = t6.disease_group
WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
{wh}
GROUP BY 1, 2
ORDER BY 1
"""
    return run_query(sql)


def load_hiv_profile(filters: dict, run_query) -> pd.DataFrame:
    """D4: HIV patient profile — count, age, sex."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
hiv_pts AS (
    SELECT DISTINCT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND (dx.icd10_code_1 LIKE 'B20%' OR dx.icd10_code_2 LIKE 'B20%')
)
SELECT
    COUNT(DISTINCT hp.patient)                              AS hiv_patients,
    COUNT(DISTINCT CASE WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 18
                        THEN hp.patient END)                AS paediatric,
    COUNT(DISTINCT CASE WHEN UPPER(p.sex) = 'F'
                        THEN hp.patient END)                AS female,
    COUNT(DISTINCT CASE WHEN UPPER(p.sex) = 'M'
                        THEN hp.patient END)                AS male
FROM hiv_pts hp
INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    ON v.patient = hp.patient AND v.source_schema = hp.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS p
    ON hp.patient = p.patient_id AND hp.source_schema = p.source_schema
"""
    return run_query(sql)


def load_mh_kpis(filters: dict, run_query) -> pd.DataFrame:
    """E1: Mental health KPI snapshot."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
)
SELECT
    COUNT(DISTINCT v.id)                                AS total_mh_visits,
    COUNT(DISTINCT v.patient)                           AS total_mh_patients,
    COUNT(DISTINCT CASE WHEN a.visit_id IS NOT NULL
                        THEN v.id END)                  AS inpatient_mh,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN a.visit_id IS NOT NULL THEN v.id END),
        COUNT(DISTINCT v.id)
    ) * 100, 1)                                         AS inpatient_share_pct
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
    ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    ON v.id = a.visit_id
   AND v.source_schema = REPLACE(LOWER(a.source_schema), '_clean', '')
WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
  {wh}
  AND (dx.icd10_code_1 LIKE 'F%' OR dx.icd10_code_2 LIKE 'F%')
"""
    return run_query(sql)


def load_mh_by_age_sex(filters: dict, run_query) -> pd.DataFrame:
    """E2: Mental health visits by age group and sex."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
)
SELECT
    CASE WHEN p.dob IS NULL THEN 'Unknown'
         WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 18 THEN 'Youth (<18)'
         WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 35 THEN 'Young Adult (18–34)'
         WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 55 THEN 'Adult (35–54)'
         ELSE 'Senior (55+)'
    END                                                     AS age_group,
    UPPER(COALESCE(p.sex, 'Unknown'))                       AS sex,
    COUNT(DISTINCT v.patient)                               AS patient_count
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
    ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS p
    ON v.patient = p.patient_id AND v.source_schema = p.source_schema
WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
  {wh}
  AND (dx.icd10_code_1 LIKE 'F%' OR dx.icd10_code_2 LIKE 'F%')
  AND age_group != 'Unknown' AND sex NOT IN ('UNKNOWN', 'Unknown')
GROUP BY 1, 2
ORDER BY patient_count DESC
"""
    return run_query(sql)


def load_revenue_by_burden_group(filters: dict, run_query) -> pd.DataFrame:
    """F1: Revenue and avg revenue per visit by disease burden group."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
vr AS (
    SELECT
        v.source_schema, v.id AS visit_id,
        COALESCE(NULLIF(TRIM(dx.disease_burden_group_1), ''),
            CASE
                WHEN n.diagnosis ILIKE '%hypertension%'
                  OR n.diagnosis ILIKE '%diabetes%'
                  OR n.diagnosis ILIKE '%renal%'
                  OR n.diagnosis ILIKE '%epilepsy%'       THEN 'NCD / Chronic'
                WHEN n.diagnosis ILIKE '%malaria%'
                  OR n.diagnosis ILIKE '%urti%'
                  OR n.diagnosis ILIKE '%typhoid%'
                  OR n.diagnosis ILIKE '%sepsis%'         THEN 'Communicable'
                WHEN n.diagnosis ILIKE '%anc%'
                  OR n.diagnosis ILIKE '%delivery%'
                  OR n.diagnosis ILIKE '%svd%'
                  OR n.diagnosis ILIKE '%caesarean%'      THEN 'RMNCH - Maternal'
                WHEN n.diagnosis ILIKE '%fracture%'
                  OR n.diagnosis ILIKE '%wound%'
                  OR n.diagnosis ILIKE '%head injury%'    THEN 'Injury'
                ELSE NULL
            END
        )                                                   AS burden_group,
        COALESCE(SUM(il.item_amount), 0)                    AS revenue
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON v.id = n.visit_id AND v.source_schema = n.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_INVOICE_LINE_ITEMS il
        ON v.id = il.visit_id AND v.source_schema = il.source_schema
       AND il.invoice_deleted_at IS NULL
       AND (il.auto_cancelled IS NULL OR il.auto_cancelled = 0)
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
    GROUP BY 1, 2, 3
)
SELECT
    burden_group,
    ROUND(SUM(revenue), 0)                              AS total_revenue,
    COUNT(DISTINCT visit_id)                            AS total_visits,
    ROUND(DIV0(SUM(revenue), COUNT(DISTINCT visit_id)), 0) AS avg_rev_per_visit
FROM vr
WHERE burden_group IS NOT NULL
GROUP BY 1
ORDER BY total_revenue DESC
"""
    return run_query(sql)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — CLINICAL WORKLOAD & QUALITY
# ══════════════════════════════════════════════════════════════════════════════

def load_shortcut_rate(filters: dict, run_query) -> pd.DataFrame:
    """Single-diagnosis rate on chronic patients per clinician."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
chronic_pts AS (
    SELECT DISTINCT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND (dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1)
),
visit_dx_count AS (
    SELECT v.source_schema, v.user AS clinician, v.id AS visit_id,
        SUM(
            CASE WHEN dx.icd10_code_1 IS NOT NULL THEN 1 ELSE 0 END
          + CASE WHEN dx.icd10_code_2 IS NOT NULL THEN 1 ELSE 0 END
          + CASE WHEN dx.icd10_code_3 IS NOT NULL THEN 1 ELSE 0 END
        ) AS dx_count
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN chronic_pts cp
        ON v.patient = cp.patient AND v.source_schema = cp.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      AND v.user IS NOT NULL
    {wh}
    GROUP BY 1, 2, 3
)
SELECT
    clinician,
    COUNT(DISTINCT visit_id)                            AS chronic_visits,
    COUNT(DISTINCT CASE WHEN dx_count = 1
                        THEN visit_id END)              AS single_dx_visits,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN dx_count = 1 THEN visit_id END),
        COUNT(DISTINCT visit_id)
    ) * 100, 1)                                         AS shortcut_rate_pct
FROM visit_dx_count
GROUP BY 1
HAVING chronic_visits >= 10
ORDER BY shortcut_rate_pct DESC
LIMIT 15
"""
    return run_query(sql)


def load_bp_omission_rate(filters: dict, run_query) -> pd.DataFrame:
    """BP not recorded on hypertension visits, per clinician."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
htn_visits AS (
    SELECT v.source_schema, v.user AS clinician, v.id AS visit_id
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      AND v.user IS NOT NULL
      {wh}
      AND LOWER(COALESCE(dx.disease_group_1, '')) LIKE '%hypertension%'
),
with_bp AS (
    SELECT DISTINCT visit_id, source_schema
    FROM HOSPITALS.STAGING.STG_EVALUATION_VITALS
    WHERE bp_systolic IS NOT NULL
)
SELECT
    clinician,
    COUNT(DISTINCT hv.visit_id)                         AS htn_visits,
    COUNT(DISTINCT CASE WHEN wb.visit_id IS NULL
                        THEN hv.visit_id END)            AS missing_bp,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN wb.visit_id IS NULL THEN hv.visit_id END),
        COUNT(DISTINCT hv.visit_id)
    ) * 100, 1)                                         AS omission_pct
FROM htn_visits hv
LEFT JOIN with_bp wb
    ON hv.visit_id = wb.visit_id AND hv.source_schema = wb.source_schema
GROUP BY 1
HAVING htn_visits >= 5
ORDER BY omission_pct DESC
LIMIT 15
"""
    return run_query(sql)


def load_return_72h(filters: dict, run_query) -> pd.DataFrame:
    """Unplanned 72h return rate per clinician."""
    wh = _w(filters, alias="v1")
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
)
SELECT
    v1.user                                             AS clinician,
    COUNT(DISTINCT v1.id)                               AS total_visits,
    COUNT(DISTINCT CASE WHEN v2.id IS NOT NULL
                        THEN v1.id END)                 AS return_visits,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN v2.id IS NOT NULL THEN v1.id END),
        COUNT(DISTINCT v1.id)
    ) * 100, 1)                                         AS return_72h_pct
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v1
INNER JOIN schema_anchor sa ON v1.source_schema = sa.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v2
    ON v1.patient = v2.patient
   AND v1.source_schema = v2.source_schema
   AND v2.created_at > v1.created_at
   AND DATEDIFF('hour', v1.created_at, v2.created_at) <= 72
WHERE v1.created_at >= DATEADD('month', -{mo}, sa.max_date)
  AND v1.user IS NOT NULL
  {wh}
GROUP BY 1
HAVING total_visits >= 20
ORDER BY return_72h_pct DESC
LIMIT 15
"""
    return run_query(sql)


# ══════════════════════════════════════════════════════════════════════════════
# CLINICIAN VIEW
# ══════════════════════════════════════════════════════════════════════════════

def load_todays_patients(filters: dict, run_query) -> pd.DataFrame:
    """CL1: Today's patient list for a clinician with priority scoring."""
    schema      = filters.get("schema", "")
    facility    = filters.get("facility", "")
    clinician   = filters.get("clinician_id", "")
    wsa         = _wsa(filters)

    schema_clause   = f"AND v.source_schema = '{schema}'"    if schema    else ""
    facility_clause = f"AND v.clinic = '{facility}'"         if facility  else ""
    clinician_clause = f"AND v.user = '{clinician}'"         if clinician else ""

    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
todays_visits AS (
    SELECT v.source_schema, v.clinic AS facility, v.id AS visit_id,
           v.patient, v.user AS clinician, v.created_at
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE DATE_TRUNC('day', v.created_at) = DATE_TRUNC('day', sa.max_date)
      AND v.user IS NOT NULL
      {schema_clause}
      {facility_clause}
      {clinician_clause}
),
condition_per_visit AS (
    SELECT v.source_schema, v.patient, v.created_at,
        MAX(NULLIF(TRIM(dx.disease_group_1), ''))       AS icd10_condition,
        MAX(CASE
            WHEN dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1
              OR n.diagnosis ILIKE '%hypertension%'
              OR n.diagnosis ILIKE '%diabetes%'
              OR n.diagnosis ILIKE '%hiv%'
              OR n.diagnosis ILIKE '%renal%'
              OR n.diagnosis ILIKE '%epilepsy%'
              OR n.diagnosis ILIKE '%asthma%'
            THEN 1 ELSE 0
        END)                                            AS is_chronic,
        MAX(CASE
            WHEN n.diagnosis ILIKE '%hypertension%'
              OR n.diagnosis ILIKE '%htn%'         THEN 'Hypertension'
            WHEN n.diagnosis ILIKE '%diabetes%'
              OR n.diagnosis ILIKE '%dm type%'     THEN 'Diabetes'
            WHEN n.diagnosis ILIKE '%hiv%'         THEN 'HIV'
            WHEN n.diagnosis ILIKE '%renal%'
              OR n.diagnosis ILIKE '%kidney%'      THEN 'Renal Disease'
            WHEN n.diagnosis ILIKE '%asthma%'      THEN 'Asthma'
            WHEN n.diagnosis ILIKE '%epilepsy%'    THEN 'Epilepsy'
            WHEN n.diagnosis ILIKE '%malaria%'
              OR n.diagnosis ILIKE '%afi%'         THEN 'Malaria / Febrile Illness'
            WHEN n.diagnosis ILIKE '%pneumonia%'
              OR n.diagnosis ILIKE '%urti%'        THEN 'Respiratory'
            WHEN n.diagnosis ILIKE '%fracture%'
              OR n.diagnosis ILIKE '%head injury%' THEN 'Injury'
            WHEN n.diagnosis ILIKE '%anc%'
              OR n.diagnosis ILIKE '%pregnancy%'   THEN 'ANC / Maternal'
            ELSE NULL
        END)                                            AS notes_condition
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN todays_visits tv
        ON v.patient = tv.patient AND v.source_schema = tv.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON v.id = n.visit_id AND v.source_schema = n.source_schema
    GROUP BY v.source_schema, v.patient, v.created_at
),
primary_condition AS (
    SELECT source_schema, patient,
        COALESCE(icd10_condition, notes_condition, 'Not recorded') AS primary_condition,
        is_chronic
    FROM condition_per_visit
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY source_schema, patient ORDER BY created_at DESC
    ) = 1
),
patient_info AS (
    SELECT p.patient_id, p.source_schema,
        UPPER(COALESCE(p.sex, 'Unknown')) AS sex,
        CASE WHEN p.dob IS NULL THEN 'Unknown'
             WHEN TIMESTAMPDIFF('year', p.dob, CURRENT_DATE) < 18 THEN 'Child'
             WHEN TIMESTAMPDIFF('year', p.dob, CURRENT_DATE) < 35 THEN 'Young Adult (18–34)'
             WHEN TIMESTAMPDIFF('year', p.dob, CURRENT_DATE) < 55 THEN 'Adult (35–54)'
             WHEN TIMESTAMPDIFF('year', p.dob, CURRENT_DATE) < 65 THEN 'Older Adult (55–64)'
             ELSE 'Senior (65+)'
        END AS age_group
    FROM HOSPITALS.STAGING.STG_RECEPTION_PATIENTS p
    INNER JOIN todays_visits tv
        ON p.patient_id = tv.patient AND p.source_schema = tv.source_schema
),
visit_type AS (
    SELECT tv.source_schema, tv.patient,
        CASE WHEN a.visit_id IS NOT NULL THEN 'Inpatient' ELSE 'Outpatient' END AS visit_type
    FROM todays_visits tv
    LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
        ON tv.visit_id = a.visit_id
       AND tv.source_schema = REPLACE(LOWER(a.source_schema), '_clean', '')
),
prev_visit AS (
    SELECT v.source_schema, v.patient, MAX(v.created_at) AS prev_visit_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN todays_visits tv
        ON v.patient = tv.patient AND v.source_schema = tv.source_schema
    WHERE DATE_TRUNC('day', v.created_at) < DATE_TRUNC('day', sa.max_date)
    GROUP BY v.source_schema, v.patient
),
abnormal_vitals AS (
    SELECT source_schema, patient, abnormal_count
    FROM (
        SELECT v.source_schema, v.patient, v.created_at,
            SUM(
                CASE WHEN COALESCE(vt.bp_systolic_status, 'Normal')
                          NOT IN ('Normal','Unknown') THEN 1 ELSE 0 END
              + CASE WHEN COALESCE(vt.bp_diastolic_status, 'Normal')
                          NOT IN ('Normal','Unknown') THEN 1 ELSE 0 END
              + CASE WHEN COALESCE(vt.blood_sugar_status, 'Normal')
                          NOT IN ('Normal','Unknown') THEN 1 ELSE 0 END
              + CASE WHEN COALESCE(vt.pulse_status, 'Normal')
                          NOT IN ('Normal','Unknown') THEN 1 ELSE 0 END
            ) AS abnormal_count
        FROM HOSPITALS.STAGING.STG_EVALUATION_VITALS vt
        INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
            ON vt.visit_id = v.id AND vt.source_schema = v.source_schema
        INNER JOIN todays_visits tv
            ON v.patient = tv.patient AND v.source_schema = tv.source_schema
        GROUP BY v.source_schema, v.patient, v.created_at
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY v.source_schema, v.patient ORDER BY v.created_at DESC
        ) = 1
    )
),
patient_status AS (
    SELECT pv.source_schema, pv.patient,
        DATEDIFF('day', pv.prev_visit_date, sa.max_date) AS days_since_prev,
        CASE WHEN pv.prev_visit_date IS NULL THEN 'New patient'
             WHEN DATEDIFF('day', pv.prev_visit_date, sa.max_date) <= 30 THEN 'Active'
             WHEN DATEDIFF('day', pv.prev_visit_date, sa.max_date) <= 90 THEN 'Lapsing'
             ELSE 'LTFU — returning'
        END AS patient_status
    FROM prev_visit pv
    INNER JOIN schema_anchor sa ON pv.source_schema = sa.source_schema
),
scored AS (
    SELECT
        tv.source_schema, tv.facility, tv.clinician, tv.patient, tv.visit_id,
        COALESCE(pi.sex, 'Unknown')                     AS sex,
        COALESCE(pi.age_group, 'Unknown')               AS age_group,
        COALESCE(pc.primary_condition, 'Not recorded')  AS primary_condition,
        COALESCE(pc.is_chronic, 0)                      AS is_chronic,
        COALESCE(vt.visit_type, 'Outpatient')           AS visit_type,
        ps.days_since_prev,
        COALESCE(ps.patient_status, 'New patient')      AS patient_status,
        COALESCE(av.abnormal_count, 0)                  AS abnormal_vitals_count,
        ROUND(
            LEAST(COALESCE(ps.days_since_prev, 0) / 90.0, 1.0) * 40
          + LEAST(COALESCE(av.abnormal_count, 0) / 6.0, 1.0) * 20
          + CASE WHEN COALESCE(pc.is_chronic, 0) = 1 THEN 15 ELSE 0 END
          + CASE WHEN COALESCE(ps.patient_status, '') = 'LTFU — returning'
                 THEN 15 ELSE 0 END
        , 0) AS priority_score
    FROM todays_visits tv
    LEFT JOIN patient_info pi
        ON tv.patient = pi.patient_id AND tv.source_schema = pi.source_schema
    LEFT JOIN primary_condition pc
        ON tv.patient = pc.patient AND tv.source_schema = pc.source_schema
    LEFT JOIN visit_type vt
        ON tv.patient = vt.patient AND tv.source_schema = vt.source_schema
    LEFT JOIN patient_status ps
        ON tv.patient = ps.patient AND tv.source_schema = ps.source_schema
    LEFT JOIN abnormal_vitals av
        ON tv.patient = av.patient AND tv.source_schema = av.source_schema
)
SELECT *,
    CASE WHEN priority_score >= 60 THEN 'HIGH'
         WHEN priority_score >= 30 THEN 'WATCH'
         ELSE 'OK'
    END AS risk_badge,
    CASE WHEN patient_status = 'LTFU — returning'
             THEN 'Returning after ' || days_since_prev || ' days — review full history'
         WHEN abnormal_vitals_count > 2
             THEN 'Abnormal vitals — ' || abnormal_vitals_count || ' flags'
         WHEN is_chronic = 1
             THEN 'Chronic patient — review meds and vitals'
         WHEN patient_status = 'Lapsing'
             THEN 'Lapsing — ' || days_since_prev || ' days since last visit'
         ELSE 'Routine visit'
    END AS flag_reason
FROM scored
ORDER BY priority_score DESC
"""
    return run_query(sql)


def load_patient_vitals_trend(patient_id: str, source_schema: str,
                               run_query) -> pd.DataFrame:
    """CL2: Last 6 vitals readings with trend direction per metric."""
    sql = f"""
WITH vitals_ordered AS (
    SELECT
        vt.created_at AS reading_date,
        vt.bp_systolic, vt.bp_diastolic, vt.blood_sugar,
        vt.blood_sugar_units,
        vt.bp_systolic_status, vt.bp_diastolic_status, vt.blood_sugar_status,
        ROW_NUMBER() OVER (ORDER BY vt.created_at DESC) AS reading_rank
    FROM HOSPITALS.STAGING.STG_EVALUATION_VITALS vt
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON vt.visit_id = v.id AND vt.source_schema = v.source_schema
    WHERE v.patient       = '{patient_id}'
      AND v.source_schema = '{source_schema}'
      AND (vt.bp_systolic IS NOT NULL OR vt.blood_sugar IS NOT NULL)
    QUALIFY ROW_NUMBER() OVER (ORDER BY vt.created_at DESC) <= 6
),
recent AS (
    SELECT AVG(bp_systolic) AS avg_sys, AVG(bp_diastolic) AS avg_dia,
        AVG(CASE WHEN LOWER(COALESCE(blood_sugar_units, '')) LIKE '%mmol%'
                 THEN blood_sugar * 18.0 ELSE blood_sugar END) AS avg_sugar
    FROM vitals_ordered WHERE reading_rank <= 2
),
prior AS (
    SELECT AVG(bp_systolic) AS avg_sys, AVG(bp_diastolic) AS avg_dia,
        AVG(CASE WHEN LOWER(COALESCE(blood_sugar_units, '')) LIKE '%mmol%'
                 THEN blood_sugar * 18.0 ELSE blood_sugar END) AS avg_sugar
    FROM vitals_ordered WHERE reading_rank BETWEEN 3 AND 6
),
trend AS (
    SELECT
        r.avg_sys AS recent_sys, r.avg_dia AS recent_dia, r.avg_sugar AS recent_sugar,
        p.avg_sys AS prior_sys,  p.avg_dia AS prior_dia,  p.avg_sugar AS prior_sugar,
        CASE WHEN p.avg_sys IS NULL         THEN 'Insufficient data'
             WHEN r.avg_sys - p.avg_sys > 14 THEN 'Worsening'
             WHEN p.avg_sys - r.avg_sys > 14 THEN 'Improving'
             ELSE 'Stable' END AS systolic_trend,
        CASE WHEN p.avg_dia IS NULL         THEN 'Insufficient data'
             WHEN r.avg_dia - p.avg_dia >  9 THEN 'Worsening'
             WHEN p.avg_dia - r.avg_dia >  9 THEN 'Improving'
             ELSE 'Stable' END AS diastolic_trend,
        CASE WHEN p.avg_sugar IS NULL                  THEN 'Insufficient data'
             WHEN r.avg_sugar - p.avg_sugar > 20       THEN 'Worsening'
             WHEN p.avg_sugar - r.avg_sugar > 20       THEN 'Improving'
             ELSE 'Stable' END AS sugar_trend,
        CASE WHEN (r.avg_sys > 140 OR r.avg_dia > 90)
              AND (r.avg_sys - COALESCE(p.avg_sys, r.avg_sys)) > 14
                 THEN 'BP elevated and rising — review medication'
             WHEN r.avg_sys > 140 OR r.avg_dia > 90
                 THEN 'BP elevated — monitor'
             WHEN COALESCE(r.avg_sugar, 0) > 200
                 THEN 'Blood sugar elevated — review'
             ELSE 'Within expected range'
        END AS clinical_signal
    FROM recent r CROSS JOIN prior p
)
SELECT
    t.recent_sys, t.recent_dia, t.recent_sugar,
    t.systolic_trend, t.diastolic_trend, t.sugar_trend,
    t.clinical_signal,
    vo.reading_date, vo.reading_rank,
    vo.bp_systolic, vo.bp_diastolic, vo.blood_sugar,
    vo.bp_systolic_status, vo.bp_diastolic_status
FROM vitals_ordered vo CROSS JOIN trend t
ORDER BY vo.reading_rank
"""
    return run_query(sql)


def load_medication_continuity(patient_id: str, source_schema: str,
                                run_query) -> pd.DataFrame:
    """CL3: Expected drug classes vs active prescriptions for chronic patients."""
    sql = f"""
WITH chronic_conditions AS (
    SELECT DISTINCT disease_group_1 AS condition
    FROM HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.patient       = '{patient_id}'
      AND v.source_schema = '{source_schema}'
      AND dx.is_chronic_1 = 1 AND dx.disease_group_1 IS NOT NULL
    UNION
    SELECT DISTINCT disease_group_2
    FROM HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.patient       = '{patient_id}'
      AND v.source_schema = '{source_schema}'
      AND dx.is_chronic_2 = 1 AND dx.disease_group_2 IS NOT NULL
),
expected_classes AS (
    SELECT cc.condition, m.expected_drug_class
    FROM chronic_conditions cc
    CROSS JOIN (
        SELECT 'Antihypertensive' AS expected_drug_class
        UNION ALL SELECT 'Antidiabetic'
        UNION ALL SELECT 'ARV'
        UNION ALL SELECT 'Bronchodilator'
        UNION ALL SELECT 'Anticonvulsant'
        UNION ALL SELECT 'Diuretic'
    ) m
    WHERE
        (LOWER(cc.condition) LIKE '%hypertension%' AND m.expected_drug_class = 'Antihypertensive')
     OR (LOWER(cc.condition) LIKE '%diabet%'       AND m.expected_drug_class = 'Antidiabetic')
     OR (LOWER(cc.condition) LIKE '%hiv%'          AND m.expected_drug_class = 'ARV')
     OR (LOWER(cc.condition) LIKE '%asthma%'       AND m.expected_drug_class = 'Bronchodilator')
     OR (LOWER(cc.condition) LIKE '%epilep%'       AND m.expected_drug_class = 'Anticonvulsant')
     OR (LOWER(cc.condition) LIKE '%renal%'        AND m.expected_drug_class = 'Diuretic')
),
active_prescriptions AS (
    SELECT pp.drug_name, pp.prescription_created_at AS last_prescribed,
        CASE
            WHEN LOWER(pp.drug_name) LIKE '%amlodipine%'
              OR LOWER(pp.drug_name) LIKE '%lisinopril%'
              OR LOWER(pp.drug_name) LIKE '%atenolol%'
              OR LOWER(pp.drug_name) LIKE '%nifedipine%'
              OR LOWER(pp.drug_name) LIKE '%losartan%'
              OR LOWER(pp.drug_name) LIKE '%captopril%'
              OR LOWER(pp.drug_name) LIKE '%methyldopa%'  THEN 'Antihypertensive'
            WHEN LOWER(pp.drug_name) LIKE '%metformin%'
              OR LOWER(pp.drug_name) LIKE '%glibenclamide%'
              OR LOWER(pp.drug_name) LIKE '%insulin%'     THEN 'Antidiabetic'
            WHEN LOWER(pp.drug_name) LIKE '%tenofovir%'
              OR LOWER(pp.drug_name) LIKE '%lamivudine%'
              OR LOWER(pp.drug_name) LIKE '%efavirenz%'
              OR LOWER(pp.drug_name) LIKE '%dolutegravir%' THEN 'ARV'
            WHEN LOWER(pp.drug_name) LIKE '%salbutamol%'
              OR LOWER(pp.drug_name) LIKE '%budesonide%'  THEN 'Bronchodilator'
            WHEN LOWER(pp.drug_name) LIKE '%phenobarbit%'
              OR LOWER(pp.drug_name) LIKE '%carbamazepine%'
              OR LOWER(pp.drug_name) LIKE '%valproate%'   THEN 'Anticonvulsant'
            WHEN LOWER(pp.drug_name) LIKE '%furosemide%'
              OR LOWER(pp.drug_name) LIKE '%frusemide%'
              OR LOWER(pp.drug_name) LIKE '%spironolactone%' THEN 'Diuretic'
            ELSE 'Other'
        END AS drug_class
    FROM HOSPITALS.STAGING.STG_PRESCRIPTION_PAYMENTS pp
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON pp.visit_id = v.id AND pp.source_schema = v.source_schema
    WHERE v.patient       = '{patient_id}'
      AND v.source_schema = '{source_schema}'
      AND (pp.stopped IS NULL OR pp.stopped = 0)
      AND (pp.canceled IS NULL OR pp.canceled = 0)
      AND (pp.remove_from_report IS NULL OR pp.remove_from_report = 0)
),
latest_per_class AS (
    SELECT drug_class,
           MAX(drug_name)       AS drug_name,
           MAX(last_prescribed) AS last_prescribed
    FROM active_prescriptions
    WHERE drug_class != 'Other'
    GROUP BY drug_class
)
SELECT
    ec.condition, ec.expected_drug_class,
    lpc.drug_name AS active_drug, lpc.last_prescribed,
    DATEDIFF('day', lpc.last_prescribed, CURRENT_DATE) AS days_since_prescribed,
    CASE WHEN lpc.drug_class IS NULL
             THEN 'Gap detected — no active prescription found'
         WHEN DATEDIFF('day', lpc.last_prescribed, CURRENT_DATE) > 90
             THEN 'Gap detected — last prescribed '
              || DATEDIFF('day', lpc.last_prescribed, CURRENT_DATE) || ' days ago'
         ELSE 'On track'
    END AS continuity_status,
    CASE WHEN lpc.drug_class IS NULL THEN 1
         WHEN DATEDIFF('day', lpc.last_prescribed, CURRENT_DATE) > 90 THEN 1
         ELSE 0
    END AS is_gap
FROM expected_classes ec
LEFT JOIN latest_per_class lpc ON ec.expected_drug_class = lpc.drug_class
ORDER BY is_gap DESC, ec.condition, ec.expected_drug_class
"""
    return run_query(sql)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 EXTENDED — DISEASE BURDEN DEEP DIVES
# ══════════════════════════════════════════════════════════════════════════════

def load_burden_trend_master_categories(filters: dict, run_query) -> pd.DataFrame:
    """100% stacked area chart — monthly visit trend split into NCD / Communicable / RMNCH / Other."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
)
SELECT
    DATE_TRUNC('month', v.created_at)   AS visit_month,
    CASE
        WHEN dx.disease_burden_group_1 ILIKE ANY (
             '%Cardiovascular%','%Diabetes%','%Chronic%','%Neurolog%',
             '%Mental%','%Endocrin%','%Musculo%')
             THEN 'NCD'
        WHEN dx.disease_burden_group_1 ILIKE ANY (
             '%Communicable%','%HIV%','%TB%','%Malaria%',
             '%URTI%','%Typhoid%','%Respiratory: Infect%')
             THEN 'Communicable'
        WHEN dx.disease_burden_group_1 ILIKE ANY (
             '%RMNCH%','%Maternal%','%Paediatric%','%Reproductive%','%Obstetric%')
             THEN 'RMNCH'
        ELSE 'Other'
    END                                 AS master_category,
    COUNT(DISTINCT v.id)                AS visit_count
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
    ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
  AND dx.disease_burden_group_1 IS NOT NULL
{wh}
GROUP BY 1, 2
ORDER BY 1
"""
    return run_query(sql)


def load_ncd_leakage_kpi(filters: dict, run_query) -> pd.DataFrame:
    """Patients with elevated vitals but no NCD diagnosis — undetected NCD leakage KPI."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
elevated_visits AS (
    SELECT DISTINCT v.source_schema, v.id AS visit_id, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VITALS vt
        ON vt.visit_id = v.id AND vt.source_schema = v.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      AND (vt.bp_systolic >= 140 OR vt.blood_sugar >= 10)
    {wh}
),
ncd_coded AS (
    SELECT DISTINCT source_schema, visit_id
    FROM HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED
    WHERE disease_burden_group_1 ILIKE ANY (
          '%Cardiovascular%','%Diabetes%','%Endocrin%')
),
undetected AS (
    SELECT ev.source_schema, ev.patient
    FROM elevated_visits ev
    LEFT JOIN ncd_coded nc
        ON ev.visit_id = nc.visit_id AND ev.source_schema = nc.source_schema
    WHERE nc.visit_id IS NULL
    GROUP BY 1, 2
),
avg_fee AS (
    SELECT
        v.source_schema,
        AVG(ili.item_amount) AS avg_consult_fee
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_INVOICE_LINE_ITEMS ili
        ON v.id = ili.visit_id AND v.source_schema = ili.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      AND ili.invoice_deleted_at IS NULL
      AND (ili.auto_cancelled IS NULL OR ili.auto_cancelled = 0)
    {wh}
    GROUP BY 1
)
SELECT
    COUNT(DISTINCT u.patient)                              AS undetected_ncd_patients,
    ROUND(AVG(af.avg_consult_fee), 0)                     AS avg_consult_fee,
    ROUND(COUNT(DISTINCT u.patient) * AVG(af.avg_consult_fee), 0) AS estimated_leakage_kes
FROM undetected u
CROSS JOIN avg_fee af
"""
    return run_query(sql)


def load_top_diagnoses_ip_op(filters: dict, run_query) -> pd.DataFrame:
    """Top 15 diagnosis groups split by inpatient vs outpatient visit share."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
)
SELECT
    dx.disease_burden_group_1                             AS burden_group,
    COUNT(DISTINCT v.id)                                  AS total_visits,
    COUNT(DISTINCT CASE WHEN ia.visit_id IS NOT NULL
                        THEN v.id END)                    AS inpatient_visits,
    COUNT(DISTINCT CASE WHEN ia.visit_id IS NULL
                        THEN v.id END)                    AS outpatient_visits,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN ia.visit_id IS NOT NULL THEN v.id END),
        COUNT(DISTINCT v.id)
    ) * 100, 1)                                           AS ip_pct,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN ia.visit_id IS NULL THEN v.id END),
        COUNT(DISTINCT v.id)
    ) * 100, 1)                                           AS op_pct
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
    ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS ia
    ON v.id = ia.visit_id
   AND v.source_schema = REPLACE(LOWER(ia.source_schema), '_clean', '')
WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
  AND dx.disease_burden_group_1 IS NOT NULL
{wh}
GROUP BY 1
ORDER BY total_visits DESC
LIMIT 15
"""
    return run_query(sql)


def load_ncd_leakage_by_clinician(filters: dict, run_query) -> pd.DataFrame:
    """Per clinician: visits with elevated vitals but no NCD code — miss rate ranking."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
elevated_visits AS (
    SELECT DISTINCT v.source_schema, v.id AS visit_id, v.user AS clinician
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VITALS vt
        ON vt.visit_id = v.id AND vt.source_schema = v.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      AND v.user IS NOT NULL
      AND (vt.bp_systolic >= 140 OR vt.blood_sugar >= 10)
    {wh}
),
ncd_coded AS (
    SELECT DISTINCT source_schema, visit_id
    FROM HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED
    WHERE disease_burden_group_1 ILIKE ANY (
          '%Cardiovascular%','%Diabetes%','%Endocrin%')
),
all_visits AS (
    SELECT v.source_schema, v.user AS clinician, COUNT(DISTINCT v.id) AS total_visits
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      AND v.user IS NOT NULL
    {wh}
    GROUP BY 1, 2
)
SELECT
    ev.clinician,
    COUNT(DISTINCT CASE WHEN nc.visit_id IS NULL THEN ev.visit_id END) AS missed_ncd_visits,
    av.total_visits,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN nc.visit_id IS NULL THEN ev.visit_id END),
        av.total_visits
    ) * 100, 1)                                                        AS miss_rate_pct
FROM elevated_visits ev
LEFT JOIN ncd_coded nc
    ON ev.visit_id = nc.visit_id AND ev.source_schema = nc.source_schema
INNER JOIN all_visits av
    ON ev.clinician = av.clinician AND ev.source_schema = av.source_schema
GROUP BY ev.clinician, av.total_visits
HAVING av.total_visits >= 10
ORDER BY miss_rate_pct DESC
LIMIT 15
"""
    return run_query(sql)


def load_emerging_diagnoses_90d(filters: dict, run_query) -> pd.DataFrame:
    """Conditions ranked 6-30 by volume showing fastest growth (last 90d vs prior 90d)."""
    wh = _w(filters)
    wsa = _wsa(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
recent AS (
    SELECT dx.disease_burden_group_1 AS condition,
           COUNT(DISTINCT v.id)       AS recent_90d_visits
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('day', -90, sa.max_date)
      AND dx.disease_burden_group_1 IS NOT NULL
    {wh}
    GROUP BY 1
),
prior AS (
    SELECT dx.disease_burden_group_1 AS condition,
           COUNT(DISTINCT v.id)       AS prior_90d_visits
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('day', -180, sa.max_date)
      AND v.created_at <  DATEADD('day', -90,  sa.max_date)
      AND dx.disease_burden_group_1 IS NOT NULL
    {wh}
    GROUP BY 1
),
ranked AS (
    SELECT r.condition, r.recent_90d_visits,
           COALESCE(p.prior_90d_visits, 0) AS prior_90d_visits
    FROM recent r
    LEFT JOIN prior p ON r.condition = p.condition
    QUALIFY RANK() OVER (ORDER BY r.recent_90d_visits DESC) BETWEEN 6 AND 30
),
ip_pct_cte AS (
    SELECT dx.disease_burden_group_1 AS condition,
           ROUND(DIV0(
               COUNT(DISTINCT CASE WHEN ia.visit_id IS NOT NULL THEN v.id END),
               COUNT(DISTINCT v.id)
           ) * 100, 1) AS inpatient_pct
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS ia
        ON v.id = ia.visit_id
       AND v.source_schema = REPLACE(LOWER(ia.source_schema), '_clean', '')
    WHERE v.created_at >= DATEADD('day', -90, sa.max_date)
      AND dx.disease_burden_group_1 IS NOT NULL
    {wh}
    GROUP BY 1
),
age_mode AS (
    SELECT condition, primary_age_group
    FROM (
        SELECT dx.disease_burden_group_1 AS condition,
               CASE
                   WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 18 THEN 'Under 18'
                   WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 35 THEN '18-34'
                   WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 50 THEN '35-49'
                   WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 65 THEN '50-64'
                   ELSE '65+'
               END AS primary_age_group,
               COUNT(*) AS cnt
        FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
        INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
            ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
        LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS rp
            ON v.patient = rp.patient_id AND v.source_schema = rp.source_schema
        WHERE v.created_at >= DATEADD('day', -90, sa.max_date)
          AND dx.disease_burden_group_1 IS NOT NULL
        {wh}
        GROUP BY 1, 2
    )
    QUALIFY ROW_NUMBER() OVER (PARTITION BY condition ORDER BY cnt DESC) = 1
),
payer_mode AS (
    SELECT condition, primary_payer
    FROM (
        SELECT dx.disease_burden_group_1 AS condition,
               CASE
                   WHEN UPPER(v.payment_mode) IN ('CASH','PRIVATE') THEN 'Cash'
                   WHEN UPPER(v.payment_mode) LIKE '%NHIF%'
                     OR UPPER(v.payment_mode) LIKE '%SHA%'          THEN 'NHIF / SHA'
                   ELSE 'Insurance / Corporate'
               END AS primary_payer,
               COUNT(*) AS cnt
        FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
        INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
            ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
        WHERE v.created_at >= DATEADD('day', -90, sa.max_date)
          AND dx.disease_burden_group_1 IS NOT NULL
        {wh}
        GROUP BY 1, 2
    )
    QUALIFY ROW_NUMBER() OVER (PARTITION BY condition ORDER BY cnt DESC) = 1
)
SELECT
    rk.condition,
    rk.recent_90d_visits,
    rk.prior_90d_visits,
    ROUND(DIV0(
        (rk.recent_90d_visits - rk.prior_90d_visits),
        NULLIF(rk.prior_90d_visits, 0)
    ) * 100, 1)                                           AS mom_growth_pct,
    ipc.inpatient_pct,
    am.primary_age_group,
    pm.primary_payer
FROM ranked rk
LEFT JOIN ip_pct_cte ipc ON rk.condition = ipc.condition
LEFT JOIN age_mode am    ON rk.condition = am.condition
LEFT JOIN payer_mode pm  ON rk.condition = pm.condition
ORDER BY mom_growth_pct DESC NULLS LAST
"""
    return run_query(sql)


def load_disease_intelligence_matrix(filters: dict, run_query) -> pd.DataFrame:
    """Per burden group (top 15 recent 90d): trend, demographics, payer, IP/OP split."""
    wh = _w(filters)
    wsa = _wsa(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
recent_visits AS (
    SELECT v.source_schema, v.id AS visit_id, v.patient, v.payment_mode,
           dx.disease_burden_group_1 AS condition
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('day', -90, sa.max_date)
      AND dx.disease_burden_group_1 IS NOT NULL
    {wh}
),
prior_visits AS (
    SELECT dx.disease_burden_group_1 AS condition,
           COUNT(DISTINCT v.id)       AS prior_count
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('day', -180, sa.max_date)
      AND v.created_at <  DATEADD('day', -90,  sa.max_date)
      AND dx.disease_burden_group_1 IS NOT NULL
    {wh}
    GROUP BY 1
),
top15 AS (
    SELECT condition
    FROM recent_visits
    GROUP BY 1
    QUALIFY RANK() OVER (ORDER BY COUNT(DISTINCT visit_id) DESC) <= 15
),
base AS (
    SELECT rv.condition,
           COUNT(DISTINCT rv.visit_id)        AS total_visits,
           COUNT(DISTINCT rv.patient)         AS total_patients
    FROM recent_visits rv
    INNER JOIN top15 t ON rv.condition = t.condition
    GROUP BY 1
),
ip_stats AS (
    SELECT rv.condition,
           ROUND(DIV0(
               COUNT(DISTINCT CASE WHEN ia.visit_id IS NOT NULL THEN rv.visit_id END),
               COUNT(DISTINCT rv.visit_id)
           ) * 100, 1) AS ip_pct,
           ROUND(DIV0(
               COUNT(DISTINCT CASE WHEN ia.visit_id IS NULL THEN rv.visit_id END),
               COUNT(DISTINCT rv.visit_id)
           ) * 100, 1) AS op_pct
    FROM recent_visits rv
    INNER JOIN top15 t ON rv.condition = t.condition
    LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS ia
        ON rv.visit_id = ia.visit_id
       AND rv.source_schema = REPLACE(LOWER(ia.source_schema), '_clean', '')
    GROUP BY 1
),
age_mode AS (
    SELECT condition, primary_age_group
    FROM (
        SELECT rv.condition,
               CASE
                   WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 18 THEN 'Under 18'
                   WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 35 THEN '18-34'
                   WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 50 THEN '35-49'
                   WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 65 THEN '50-64'
                   ELSE '65+'
               END AS primary_age_group,
               COUNT(*) AS cnt
        FROM recent_visits rv
        INNER JOIN top15 t ON rv.condition = t.condition
        LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS rp
            ON rv.patient = rp.patient_id AND rv.source_schema = rp.source_schema
        GROUP BY 1, 2
    )
    QUALIFY ROW_NUMBER() OVER (PARTITION BY condition ORDER BY cnt DESC) = 1
),
gender_mode AS (
    SELECT condition, primary_gender
    FROM (
        SELECT rv.condition,
               UPPER(COALESCE(rp.sex, 'Unknown')) AS primary_gender,
               COUNT(*) AS cnt
        FROM recent_visits rv
        INNER JOIN top15 t ON rv.condition = t.condition
        LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS rp
            ON rv.patient = rp.patient_id AND rv.source_schema = rp.source_schema
        GROUP BY 1, 2
    )
    QUALIFY ROW_NUMBER() OVER (PARTITION BY condition ORDER BY cnt DESC) = 1
),
payer_mode AS (
    SELECT condition, primary_payer
    FROM (
        SELECT rv.condition,
               CASE
                   WHEN UPPER(rv.payment_mode) IN ('CASH','PRIVATE') THEN 'Cash'
                   WHEN UPPER(rv.payment_mode) LIKE '%NHIF%'
                     OR UPPER(rv.payment_mode) LIKE '%SHA%'          THEN 'NHIF / SHA'
                   ELSE 'Insurance / Corporate'
               END AS primary_payer,
               COUNT(*) AS cnt
        FROM recent_visits rv
        INNER JOIN top15 t ON rv.condition = t.condition
        GROUP BY 1, 2
    )
    QUALIFY ROW_NUMBER() OVER (PARTITION BY condition ORDER BY cnt DESC) = 1
)
SELECT
    b.condition,
    b.total_visits,
    ROUND(DIV0(
        (b.total_visits - COALESCE(pv.prior_count, 0)),
        NULLIF(pv.prior_count, 0)
    ) * 100, 1)                                           AS trend_90d_pct,
    am.primary_age_group,
    gm.primary_gender,
    ip.ip_pct,
    ip.op_pct,
    pm.primary_payer
FROM base b
LEFT JOIN prior_visits pv  ON b.condition = pv.condition
LEFT JOIN ip_stats ip      ON b.condition = ip.condition
LEFT JOIN age_mode am      ON b.condition = am.condition
LEFT JOIN gender_mode gm   ON b.condition = gm.condition
LEFT JOIN payer_mode pm    ON b.condition = pm.condition
ORDER BY b.total_visits DESC
LIMIT 15
"""
    return run_query(sql)


def load_ncd_age_heatmap(filters: dict, run_query) -> pd.DataFrame:
    """Patient count per (age_group, NCD condition) combination — heatmap data."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
)
SELECT
    CASE
        WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 18 THEN 'Under 18'
        WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 35 THEN '18-34'
        WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 50 THEN '35-49'
        WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 65 THEN '50-64'
        ELSE '65+'
    END                                                   AS age_group,
    dx.disease_burden_group_1                             AS chronic_condition,
    COUNT(DISTINCT v.patient)                             AS patient_count
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
    ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS rp
    ON v.patient = rp.patient_id AND v.source_schema = rp.source_schema
WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
  AND dx.disease_burden_group_1 ILIKE ANY (
      '%Cardiovascular%','%Diabetes%','%Chronic%','%Neurolog%',
      '%Mental%','%Endocrin%','%Musculo%')
  AND rp.dob IS NOT NULL
{wh}
GROUP BY 1, 2
HAVING patient_count >= 3
ORDER BY 1, 2
"""
    return run_query(sql)


def load_htn_scatter_data(filters: dict, run_query) -> pd.DataFrame:
    """HTN patients: avg annual visits vs unique doctors, coloured by control status."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
htn_patients AS (
    SELECT DISTINCT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      AND dx.disease_burden_group_1 ILIKE '%Cardiovascular%'
      AND (dx.disease_group_1 ILIKE '%Hypertension%' OR dx.disease_group_1 ILIKE '%HTN%')
    {wh}
),
visit_stats AS (
    SELECT
        v.source_schema, v.patient,
        COUNT(DISTINCT v.id)                              AS total_visits,
        COUNT(DISTINCT v.user)                            AS unique_doctors,
        DATEDIFF('year',
            MIN(v.created_at),
            MAX(v.created_at)
        )                                                 AS years_in_dataset
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN htn_patients hp
        ON v.patient = hp.patient AND v.source_schema = hp.source_schema
    GROUP BY 1, 2
),
bp_stats AS (
    SELECT
        v.source_schema, v.patient,
        AVG(vt.bp_systolic)  AS avg_systolic,
        AVG(vt.bp_diastolic) AS avg_diastolic
    FROM HOSPITALS.STAGING.STG_EVALUATION_VITALS vt
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON vt.visit_id = v.id AND vt.source_schema = v.source_schema
    INNER JOIN htn_patients hp
        ON v.patient = hp.patient AND v.source_schema = hp.source_schema
    WHERE vt.bp_systolic IS NOT NULL
    GROUP BY 1, 2
)
SELECT
    ROW_NUMBER() OVER (ORDER BY vs.patient)               AS patient,
    ROUND(DIV0(vs.total_visits, NULLIF(vs.years_in_dataset, 0) + 1), 1) AS avg_annual_visits,
    vs.unique_doctors,
    CASE
        WHEN COALESCE(bs.avg_systolic, 999) < 140
         AND COALESCE(bs.avg_diastolic, 999) < 90 THEN 'Controlled'
        ELSE 'Uncontrolled'
    END                                                   AS htn_status
FROM visit_stats vs
LEFT JOIN bp_stats bs
    ON vs.patient = bs.patient AND vs.source_schema = bs.source_schema
"""
    return run_query(sql)


def load_chronic_comorbidity_pairs(filters: dict, run_query) -> pd.DataFrame:
    """Top 10 chronic condition pairs by patient count with avg days between diagnoses."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
patient_conditions AS (
    SELECT
        v.source_schema, v.patient,
        CASE
            WHEN dx.disease_burden_group_1 ILIKE '%Endocrin%'
              OR dx.disease_burden_group_1 ILIKE '%Diabetes%' THEN 'NCD — Diabetes & Metabolic'
            ELSE dx.disease_burden_group_1
        END                                               AS condition,
        MIN(v.created_at)                                 AS first_seen
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      AND dx.disease_burden_group_1 ILIKE ANY (
          '%Cardiovascular%','%Diabetes%','%Chronic%','%Neurolog%',
          '%Mental%','%Endocrin%','%Musculo%')
    {wh}
    GROUP BY 1, 2, 3
),
ranked_conditions AS (
    SELECT source_schema, patient, condition, first_seen,
           ROW_NUMBER() OVER (
               PARTITION BY source_schema, patient
               ORDER BY first_seen
           ) AS cond_rank
    FROM patient_conditions
),
pairs AS (
    SELECT
        a.source_schema, a.patient,
        a.condition                                       AS condition_1,
        b.condition                                       AS condition_2,
        DATEDIFF('day', a.first_seen, b.first_seen)       AS days_between
    FROM ranked_conditions a
    INNER JOIN ranked_conditions b
        ON a.source_schema = b.source_schema
       AND a.patient = b.patient
       AND b.cond_rank = a.cond_rank + 1
    WHERE a.condition <> b.condition
)
SELECT
    condition_1 || ' → ' || condition_2                   AS condition_pair,
    COUNT(DISTINCT patient)                               AS patient_count,
    ROUND(AVG(days_between), 0)                          AS avg_days_between_diagnoses
FROM pairs
GROUP BY 1
HAVING patient_count >= 3
ORDER BY patient_count DESC
LIMIT 10
"""
    return run_query(sql)


def load_ncd_ranked_with_gender(filters: dict, run_query) -> pd.DataFrame:
    """Top NCDs by patient count with gender split. Diabetes+Endocrine consolidated."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
raw AS (
    SELECT
        CASE
            WHEN dx.disease_burden_group_1 ILIKE '%Endocrin%'
              OR dx.disease_burden_group_1 ILIKE '%Diabetes%' THEN 'NCD — Diabetes & Metabolic'
            ELSE dx.disease_burden_group_1
        END                                               AS ncd_group,
        CASE WHEN rp.sex IS NULL THEN 'Unknown'
             ELSE UPPER(CAST(rp.sex AS VARCHAR)) END      AS gender,
        COUNT(DISTINCT v.patient)                         AS patient_count
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS rp
        ON v.patient = rp.patient_id AND v.source_schema = rp.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      AND dx.disease_burden_group_1 ILIKE ANY (
          '%Cardiovascular%','%Diabetes%','%Endocrin%','%Neurolog%',
          '%Mental%','%Musculo%','%Chronic%')
    {wh}
    GROUP BY 1, 2
    HAVING COUNT(DISTINCT v.patient) >= 3
),
group_totals AS (
    SELECT ncd_group, SUM(patient_count) AS group_total
    FROM raw
    GROUP BY ncd_group
)
SELECT r.ncd_group, r.gender, r.patient_count
FROM raw r
INNER JOIN group_totals gt ON r.ncd_group = gt.ncd_group
ORDER BY gt.group_total DESC, r.ncd_group, r.gender
"""
    return run_query(sql)


def load_ncd_complexity_distribution(filters: dict, run_query) -> pd.DataFrame:
    """NCD complexity: share of patients with 1, 2, 3, 4+ distinct NCDs."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
patient_ncd_count AS (
    SELECT
        v.source_schema, v.patient,
        COUNT(DISTINCT
            CASE
                WHEN dx.disease_burden_group_1 ILIKE '%Endocrin%'
                  OR dx.disease_burden_group_1 ILIKE '%Diabetes%' THEN 'Diabetes_Metabolic'
                ELSE dx.disease_burden_group_1
            END
        )                                                 AS distinct_ncds
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      AND dx.disease_burden_group_1 ILIKE ANY (
          '%Cardiovascular%','%Diabetes%','%Endocrin%','%Neurolog%',
          '%Mental%','%Musculo%','%Chronic%')
    {wh}
    GROUP BY 1, 2
)
SELECT
    CASE
        WHEN distinct_ncds = 1 THEN '1 NCD'
        WHEN distinct_ncds = 2 THEN '2 NCDs'
        WHEN distinct_ncds = 3 THEN '3 NCDs'
        ELSE '4+ NCDs (Complex)'
    END                                                   AS ncd_complexity,
    distinct_ncds,
    COUNT(DISTINCT patient)                               AS patient_count,
    ROUND(DIV0(
        COUNT(DISTINCT patient) * 100,
        SUM(COUNT(DISTINCT patient)) OVER ()
    ), 1)                                                 AS pct_of_ncd_patients
FROM patient_ncd_count
GROUP BY 1, 2
ORDER BY distinct_ncds
"""
    return run_query(sql)


def load_htn_uncontrolled_profile(filters: dict, run_query) -> pd.DataFrame:
    """HTN patients: controlled vs uncontrolled by age, payer, comorbidity count,
    medication use, and investigation rate. Informs why patients are uncontrolled."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
htn_patients AS (
    SELECT DISTINCT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      AND (dx.disease_group_1 ILIKE '%hypertension%' OR dx.disease_group_1 ILIKE '%HTN%')
    {wh}
),
bp_status AS (
    SELECT v.source_schema, v.patient,
           CASE WHEN AVG(vt.bp_systolic) < 140 AND AVG(vt.bp_diastolic) < 90 THEN 'Controlled'
                WHEN AVG(vt.bp_systolic) IS NULL THEN 'No BP Recorded'
                ELSE 'Uncontrolled'
           END AS htn_status,
           AVG(vt.bp_systolic)  AS avg_systolic,
           AVG(vt.bp_diastolic) AS avg_diastolic
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN htn_patients hp ON v.patient = hp.patient AND v.source_schema = hp.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_VITALS vt
        ON vt.visit_id = v.id AND vt.source_schema = v.source_schema
    GROUP BY 1, 2
),
demographics AS (
    SELECT rp.patient_id, rp.source_schema,
           CASE WHEN rp.sex IS NULL THEN 'Unknown' ELSE UPPER(CAST(rp.sex AS VARCHAR)) END AS gender,
           CASE
               WHEN rp.dob IS NULL THEN 'Unknown'
               WHEN TIMESTAMPDIFF('year', rp.dob, CURRENT_DATE) < 35 THEN 'Under 35'
               WHEN TIMESTAMPDIFF('year', rp.dob, CURRENT_DATE) < 50 THEN '35-49'
               WHEN TIMESTAMPDIFF('year', rp.dob, CURRENT_DATE) < 65 THEN '50-64'
               ELSE '65+'
           END                               AS age_group
    FROM HOSPITALS.STAGING.STG_RECEPTION_PATIENTS rp
    INNER JOIN htn_patients hp ON rp.patient_id = hp.patient AND rp.source_schema = hp.source_schema
),
payer_mode AS (
    SELECT v.source_schema, v.patient,
           CASE
               WHEN UPPER(v.payment_mode) IN ('CASH','PRIVATE') THEN 'Cash'
               WHEN UPPER(v.payment_mode) LIKE '%NHIF%'
                 OR UPPER(v.payment_mode) LIKE '%SHA%' THEN 'NHIF / SHA'
               ELSE 'Insurance / Corporate'
           END AS payer
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN htn_patients hp ON v.patient = hp.patient AND v.source_schema = hp.source_schema
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
    QUALIFY ROW_NUMBER() OVER (PARTITION BY v.source_schema, v.patient
                               ORDER BY v.created_at DESC) = 1
),
comorbidity_count AS (
    SELECT v.source_schema, v.patient,
           COUNT(DISTINCT
               CASE
                   WHEN dx.disease_burden_group_1 ILIKE '%Endocrin%'
                     OR dx.disease_burden_group_1 ILIKE '%Diabetes%' THEN 'Diabetes_Metabolic'
                   ELSE dx.disease_burden_group_1
               END
           ) - 1                                          AS other_ncd_count
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN htn_patients hp ON v.patient = hp.patient AND v.source_schema = hp.source_schema
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      AND dx.disease_burden_group_1 ILIKE ANY (
          '%Cardiovascular%','%Diabetes%','%Endocrin%','%Neurolog%','%Mental%','%Musculo%')
    {wh}
    GROUP BY 1, 2
),
antihypertensive AS (
    SELECT DISTINCT v.source_schema, v.patient, 1 AS has_antihypertensive
    FROM HOSPITALS.STAGING.STG_PRESCRIPTION_PAYMENTS pp
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON pp.visit_id = v.id AND pp.source_schema = v.source_schema
    INNER JOIN htn_patients hp ON v.patient = hp.patient AND v.source_schema = hp.source_schema
    WHERE (pp.stopped IS NULL OR pp.stopped = 0)
      AND (pp.canceled IS NULL OR pp.canceled = 0)
      AND pp.drug_name ILIKE ANY (
          '%amlodipine%','%nifedipine%','%atenolol%','%metoprolol%',
          '%lisinopril%','%enalapril%','%losartan%','%valsartan%',
          '%hydrochlorothiazide%','%furosemide%','%spironolactone%',
          '%bisoprolol%','%carvedilol%','%ramipril%','%captopril%')
),
investigations AS (
    SELECT v.source_schema, v.patient,
           COUNT(*) AS investigation_count
    FROM HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS i
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON i.visit_id = v.id AND i.source_schema = v.source_schema
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN htn_patients hp ON v.patient = hp.patient AND v.source_schema = hp.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      AND (i.cancelled IS NULL OR i.cancelled = 0)
    {wh}
    GROUP BY 1, 2
),
visit_counts AS (
    SELECT v.source_schema, v.patient, COUNT(DISTINCT v.id) AS visit_count
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN htn_patients hp ON v.patient = hp.patient AND v.source_schema = hp.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
    GROUP BY 1, 2
)
SELECT
    bs.htn_status,
    COALESCE(d.age_group, 'Unknown')                      AS age_group,
    COALESCE(d.gender, 'Unknown')                         AS gender,
    COALESCE(pm.payer, 'Unknown')                         AS payer,
    CASE WHEN COALESCE(cc.other_ncd_count, 0) = 0 THEN 'HTN Only'
         WHEN COALESCE(cc.other_ncd_count, 0) = 1 THEN '1 Other NCD'
         ELSE '2+ Other NCDs'
    END                                                   AS comorbidity_group,
    COALESCE(ah.has_antihypertensive, 0)                  AS on_antihypertensive,
    COUNT(DISTINCT hp.patient)                            AS patient_count,
    ROUND(AVG(COALESCE(inv.investigation_count, 0)), 1)   AS avg_investigations,
    ROUND(AVG(COALESCE(vc.visit_count, 0)), 1)            AS avg_visits,
    ROUND(AVG(bs.avg_systolic), 0)                        AS avg_systolic
FROM htn_patients hp
INNER JOIN bp_status bs   ON hp.patient = bs.patient AND hp.source_schema = bs.source_schema
LEFT JOIN demographics d  ON hp.patient = d.patient_id AND hp.source_schema = d.source_schema
LEFT JOIN payer_mode pm   ON hp.patient = pm.patient AND hp.source_schema = pm.source_schema
LEFT JOIN comorbidity_count cc ON hp.patient = cc.patient AND hp.source_schema = cc.source_schema
LEFT JOIN antihypertensive ah ON hp.patient = ah.patient AND hp.source_schema = ah.source_schema
LEFT JOIN investigations inv   ON hp.patient = inv.patient AND hp.source_schema = inv.source_schema
LEFT JOIN visit_counts vc      ON hp.patient = vc.patient AND hp.source_schema = vc.source_schema
GROUP BY 1, 2, 3, 4, 5, 6
ORDER BY patient_count DESC
"""
    return run_query(sql)


def load_chronic_care_matrix(filters: dict, run_query) -> pd.DataFrame:
    """Data-driven NCD quality matrix: top conditions with trend, control, payer,
    visit rate, investigation rate, and average revenue. No hardcoded rows."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    half = max(1, mo // 2)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
ncd_visits AS (
    SELECT
        v.source_schema, v.patient, v.id AS visit_id, v.created_at,
        CASE
            WHEN dx.disease_burden_group_1 ILIKE '%Endocrin%'
              OR dx.disease_burden_group_1 ILIKE '%Diabetes%' THEN 'NCD — Diabetes & Metabolic'
            ELSE dx.disease_burden_group_1
        END                                               AS condition,
        CASE
            WHEN UPPER(v.payment_mode) IN ('CASH','PRIVATE') THEN 'Cash'
            WHEN UPPER(v.payment_mode) LIKE '%NHIF%'
              OR UPPER(v.payment_mode) LIKE '%SHA%' THEN 'NHIF / SHA'
            ELSE 'Insurance / Corporate'
        END                                               AS payer,
        CASE WHEN ia.visit_id IS NOT NULL THEN 1 ELSE 0 END AS is_ip
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS ia
        ON v.id = ia.visit_id
       AND v.source_schema = REPLACE(LOWER(ia.source_schema), '_clean', '')
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      AND dx.disease_burden_group_1 ILIKE ANY (
          '%Cardiovascular%','%Diabetes%','%Endocrin%','%Neurolog%',
          '%Mental%','%Musculo%','%Chronic%')
    {wh}
),
top_conditions AS (
    SELECT condition
    FROM ncd_visits
    GROUP BY condition
    ORDER BY COUNT(DISTINCT patient) DESC
    LIMIT 10
),
trend AS (
    SELECT nv.condition,
           COUNT(DISTINCT CASE WHEN nv.created_at >= DATEADD('month', -{half}, sa.max_date)
                               THEN nv.patient END) AS recent_patients,
           COUNT(DISTINCT CASE WHEN nv.created_at < DATEADD('month', -{half}, sa.max_date)
                               THEN nv.patient END) AS prior_patients
    FROM ncd_visits nv
    INNER JOIN schema_anchor sa ON nv.source_schema = sa.source_schema
    INNER JOIN top_conditions tc ON nv.condition = tc.condition
    GROUP BY 1
),
htn_bp AS (
    SELECT v.source_schema, v.patient,
           AVG(vt.bp_systolic) AS avg_sys, AVG(vt.bp_diastolic) AS avg_dia
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VITALS vt
        ON vt.visit_id = v.id AND vt.source_schema = v.source_schema
    WHERE (dx.disease_group_1 ILIKE '%hypertension%' OR dx.disease_group_1 ILIKE '%HTN%')
      AND vt.bp_systolic IS NOT NULL
    GROUP BY 1, 2
),
revenue AS (
    SELECT nv.condition,
           SUM(ili.item_amount)    AS total_revenue,
           COUNT(DISTINCT nv.patient) AS rev_patients
    FROM ncd_visits nv
    INNER JOIN HOSPITALS.STAGING.STG_INVOICE_LINE_ITEMS ili
        ON nv.visit_id = ili.visit_id AND nv.source_schema = ili.source_schema
    INNER JOIN top_conditions tc ON nv.condition = tc.condition
    WHERE ili.invoice_deleted_at IS NULL
      AND (ili.auto_cancelled IS NULL OR ili.auto_cancelled = 0)
    GROUP BY 1
),
investigations AS (
    SELECT nv.condition,
           COUNT(*) AS inv_count,
           COUNT(DISTINCT nv.visit_id) AS visit_count_inv
    FROM ncd_visits nv
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS i
        ON nv.visit_id = i.visit_id AND nv.source_schema = i.source_schema
    INNER JOIN top_conditions tc ON nv.condition = tc.condition
    WHERE (i.cancelled IS NULL OR i.cancelled = 0)
    GROUP BY 1
)
SELECT
    nv.condition,
    COUNT(DISTINCT nv.patient)                            AS patient_count,
    ROUND(DIV0(
        t.recent_patients - t.prior_patients,
        NULLIF(t.prior_patients, 0)
    ) * 100, 1)                                           AS trend_pct,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN nv.is_ip = 1 THEN nv.visit_id END),
        COUNT(DISTINCT nv.visit_id)
    ) * 100, 1)                                           AS ip_rate_pct,
    MODE(nv.payer)                                        AS top_payer,
    ROUND(DIV0(
        COUNT(DISTINCT nv.visit_id),
        COUNT(DISTINCT nv.patient)
    ), 1)                                                 AS avg_visits_per_patient,
    ROUND(DIV0(
        inv.inv_count,
        NULLIF(inv.visit_count_inv, 0)
    ), 2)                                                 AS investigations_per_visit,
    ROUND(DIV0(
        r.total_revenue,
        NULLIF(r.rev_patients, 0)
    ), 0)                                                 AS avg_revenue_per_patient,
    CASE
        WHEN nv.condition ILIKE '%Cardiovascular%' THEN
            ROUND(DIV0(
                COUNT(DISTINCT CASE WHEN hb.avg_sys < 140 AND hb.avg_dia < 90
                                    THEN hb.patient END),
                NULLIF(COUNT(DISTINCT hb.patient), 0)
            ) * 100, 1)
        ELSE NULL
    END                                                   AS controlled_pct
FROM ncd_visits nv
INNER JOIN top_conditions tc ON nv.condition = tc.condition
INNER JOIN trend t ON nv.condition = t.condition
LEFT JOIN revenue r ON nv.condition = r.condition
LEFT JOIN investigations inv ON nv.condition = inv.condition
LEFT JOIN htn_bp hb ON nv.patient = hb.patient AND nv.source_schema = hb.source_schema
GROUP BY nv.condition, t.recent_patients, t.prior_patients,
         inv.inv_count, inv.visit_count_inv, r.total_revenue, r.rev_patients
ORDER BY patient_count DESC
"""
    return run_query(sql)


def load_chronic_pharmacy_only(filters: dict, run_query) -> pd.DataFrame:
    """Documentation-gap metric: chronic visits with a prescription but no vitals
    AND no clinical note — signals possible prescription without clinical review."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
chronic_visits AS (
    SELECT
        v.source_schema, v.id AS visit_id, v.patient,
        dx.disease_burden_group_1                         AS condition,
        CASE
            WHEN UPPER(v.payment_mode) IN ('CASH','PRIVATE') THEN 'Cash'
            WHEN UPPER(v.payment_mode) LIKE '%NHIF%'
              OR UPPER(v.payment_mode) LIKE '%SHA%'        THEN 'NHIF / SHA'
            ELSE 'Insurance / Corporate'
        END                                               AS payer
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      AND dx.disease_burden_group_1 ILIKE ANY (
          '%Cardiovascular%','%Diabetes%','%Chronic%','%Neurolog%',
          '%Mental%','%Endocrin%','%Musculo%','%HIV%')
    {wh}
),
has_rx AS (
    SELECT DISTINCT visit_id, source_schema
    FROM HOSPITALS.STAGING.STG_PRESCRIPTION_PAYMENTS
),
has_note AS (
    SELECT DISTINCT visit_id, source_schema
    FROM HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES
),
has_vitals AS (
    SELECT DISTINCT visit_id, source_schema
    FROM HOSPITALS.STAGING.STG_EVALUATION_VITALS
    WHERE bp_systolic IS NOT NULL OR blood_sugar IS NOT NULL
),
revenue AS (
    SELECT visit_id, source_schema, SUM(item_amount) AS visit_revenue
    FROM HOSPITALS.STAGING.STG_INVOICE_LINE_ITEMS
    WHERE invoice_deleted_at IS NULL
      AND (auto_cancelled IS NULL OR auto_cancelled = 0)
    GROUP BY 1, 2
)
SELECT
    cv.payer,
    cv.condition,
    COUNT(DISTINCT cv.patient)                            AS patient_count,
    COUNT(DISTINCT cv.visit_id)                           AS total_visits,
    COUNT(DISTINCT CASE WHEN rx.visit_id IS NOT NULL
                         AND hn.visit_id IS NULL
                         AND hv.visit_id IS NULL
                         THEN cv.visit_id END)            AS pharmacy_only_visits,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN rx.visit_id IS NOT NULL
                             AND hn.visit_id IS NULL
                             AND hv.visit_id IS NULL
                             THEN cv.visit_id END),
        COUNT(DISTINCT cv.visit_id)
    ) * 100, 1)                                           AS pharmacy_only_pct,
    ROUND(DIV0(
        SUM(COALESCE(rv.visit_revenue, 0)),
        NULLIF(DATEDIFF('year',
            MIN(v2.created_at), MAX(v2.created_at)
        ), 0) + 1
    ), 0)                                                 AS avg_annual_revenue
FROM chronic_visits cv
LEFT JOIN has_rx rx
    ON cv.visit_id = rx.visit_id AND cv.source_schema = rx.source_schema
LEFT JOIN has_note hn
    ON cv.visit_id = hn.visit_id AND cv.source_schema = hn.source_schema
LEFT JOIN has_vitals hv
    ON cv.visit_id = hv.visit_id AND cv.source_schema = hv.source_schema
LEFT JOIN revenue rv
    ON cv.visit_id = rv.visit_id AND cv.source_schema = rv.source_schema
INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v2
    ON cv.patient = v2.patient AND cv.source_schema = v2.source_schema
GROUP BY 1, 2
ORDER BY pharmacy_only_pct DESC
"""
    return run_query(sql)


def load_communicable_demographic_split(filters: dict, run_query) -> pd.DataFrame:
    """Top communicable diseases by age/sex category with IP rate and payer."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
)
SELECT
    CASE
        WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 18 THEN 'Paediatric (<18)'
        WHEN UPPER(COALESCE(rp.sex, '')) = 'F'           THEN 'Adult Female'
        ELSE 'Adult Male'
    END                                                   AS age_sex_group,
    dx.disease_group_1                                    AS disease_group,
    COUNT(DISTINCT v.id)                                  AS visit_count,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN ia.visit_id IS NOT NULL THEN v.id END),
        COUNT(DISTINCT v.id)
    ) * 100, 1)                                           AS inpatient_pct,
    CASE
        WHEN UPPER(v.payment_mode) IN ('CASH','PRIVATE') THEN 'Cash'
        WHEN UPPER(v.payment_mode) LIKE '%NHIF%'
          OR UPPER(v.payment_mode) LIKE '%SHA%'           THEN 'NHIF / SHA'
        ELSE 'Insurance / Corporate'
    END                                                   AS primary_payer
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
    ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS rp
    ON v.patient = rp.patient_id AND v.source_schema = rp.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS ia
    ON v.id = ia.visit_id
   AND v.source_schema = REPLACE(LOWER(ia.source_schema), '_clean', '')
WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
  AND dx.disease_group_1 ILIKE ANY (
      '%URTI%','%Malaria%','%Typhoid%','%TB%',
      '%HIV%','%Gastroenteritis%')
{wh}
GROUP BY 1, 2, 5
ORDER BY visit_count DESC
LIMIT 60
"""
    return run_query(sql)


def load_communicable_pipeline_matrix(filters: dict, run_query) -> pd.DataFrame:
    """Top 10 communicable diseases: lab confirmation, IP rate, comorbidity, payer."""
    wh = _w(filters)
    wsa = _wsa(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
comm_visits AS (
    SELECT v.source_schema, v.id AS visit_id, v.patient,
           dx.disease_group_1 AS disease_group,
           CASE
               WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 18 THEN 'Paediatric (<18)'
               WHEN UPPER(COALESCE(rp.sex, '')) = 'F'           THEN 'Adult Female'
               ELSE 'Adult Male'
           END AS age_sex_cat,
           CASE
               WHEN UPPER(v.payment_mode) IN ('CASH','PRIVATE') THEN 'Cash'
               WHEN UPPER(v.payment_mode) LIKE '%NHIF%'
                 OR UPPER(v.payment_mode) LIKE '%SHA%'          THEN 'NHIF / SHA'
               ELSE 'Insurance / Corporate'
           END AS payer
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS rp
        ON v.patient = rp.patient_id AND v.source_schema = rp.source_schema
    WHERE v.created_at >= DATEADD('day', -90, sa.max_date)
      AND dx.disease_burden_group_1 ILIKE ANY (
          '%Communicable%','%HIV%','%TB%','%Malaria%',
          '%URTI%','%Typhoid%','%Respiratory: Infect%')
    {wh}
),
top10 AS (
    SELECT disease_group
    FROM comm_visits
    GROUP BY 1
    QUALIFY RANK() OVER (ORDER BY COUNT(DISTINCT visit_id) DESC) <= 10
),
has_lab AS (
    SELECT DISTINCT visit_id, source_schema
    FROM HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS
    WHERE investigation_type IS NOT NULL
),
ip_flag AS (
    SELECT DISTINCT visit_id, source_schema
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS
),
age_mode AS (
    SELECT disease_group, primary_age_sex
    FROM (
        SELECT disease_group, age_sex_cat AS primary_age_sex, COUNT(*) AS cnt
        FROM comm_visits INNER JOIN top10 USING (disease_group)
        GROUP BY 1, 2
    )
    QUALIFY ROW_NUMBER() OVER (PARTITION BY disease_group ORDER BY cnt DESC) = 1
),
payer_mode AS (
    SELECT disease_group, primary_payer
    FROM (
        SELECT disease_group, payer AS primary_payer, COUNT(*) AS cnt
        FROM comm_visits INNER JOIN top10 USING (disease_group)
        GROUP BY 1, 2
    )
    QUALIFY ROW_NUMBER() OVER (PARTITION BY disease_group ORDER BY cnt DESC) = 1
),
comorbidity AS (
    SELECT disease_group, primary_comorbidity
    FROM (
        SELECT cv.disease_group,
               dx2.disease_burden_group_1 AS primary_comorbidity,
               COUNT(*) AS cnt
        FROM comm_visits cv
        INNER JOIN top10 t ON cv.disease_group = t.disease_group
        INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx2
            ON cv.visit_id = dx2.visit_id
        WHERE dx2.disease_burden_group_1 IS NOT NULL
          AND NOT (dx2.disease_burden_group_1 ILIKE ANY (
              '%Communicable%','%HIV%','%TB%','%Malaria%','%URTI%','%Typhoid%','%Respiratory: Infect%'))
        GROUP BY 1, 2
    ) t
    QUALIFY ROW_NUMBER() OVER (PARTITION BY disease_group ORDER BY cnt DESC) = 1
)
SELECT
    cv.disease_group,
    COUNT(DISTINCT cv.visit_id)                           AS quarterly_visits,
    am.primary_age_sex,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN hl.visit_id IS NOT NULL THEN cv.visit_id END),
        COUNT(DISTINCT cv.visit_id)
    ) * 100, 1)                                           AS lab_confirmation_pct,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN ipf.visit_id IS NOT NULL THEN cv.visit_id END),
        COUNT(DISTINCT cv.visit_id)
    ) * 100, 1)                                           AS inpatient_admission_pct,
    COALESCE(cm.primary_comorbidity, '—')                 AS primary_comorbidity,
    pm.primary_payer
FROM comm_visits cv
INNER JOIN top10 t  ON cv.disease_group = t.disease_group
LEFT JOIN has_lab hl
    ON cv.visit_id = hl.visit_id AND cv.source_schema = hl.source_schema
LEFT JOIN ip_flag ipf
    ON cv.visit_id = ipf.visit_id
   AND cv.source_schema = REPLACE(LOWER(ipf.source_schema), '_clean', '')
LEFT JOIN age_mode am  ON cv.disease_group = am.disease_group
LEFT JOIN payer_mode pm ON cv.disease_group = pm.disease_group
LEFT JOIN comorbidity cm ON cv.disease_group = cm.disease_group
GROUP BY cv.disease_group, am.primary_age_sex, pm.primary_payer, cm.primary_comorbidity
ORDER BY quarterly_visits DESC
LIMIT 10
"""
    return run_query(sql)


def load_disease_kpi_snapshot(filters: dict, run_query) -> pd.DataFrame:
    """D: TB, Malaria, URTI, Typhoid, HIV, Enteric — patient/visit/admission counts."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
base AS (
    SELECT v.source_schema, v.id AS visit_id, v.patient,
           CASE
               WHEN dx.icd10_code_1 LIKE 'A15%' OR dx.icd10_code_2 LIKE 'A15%'
                 OR dx.icd10_code_1 LIKE 'A16%' OR dx.icd10_code_2 LIKE 'A16%'
                 OR dx.icd10_code_1 LIKE 'A17%' OR dx.icd10_code_2 LIKE 'A17%'
                 OR dx.icd10_code_1 LIKE 'A18%' OR dx.icd10_code_2 LIKE 'A18%'
                 OR dx.icd10_code_1 LIKE 'A19%' OR dx.icd10_code_2 LIKE 'A19%'
                 OR dx.disease_group_1 ILIKE '%Tubercul%'                       THEN 'TB'
               WHEN dx.icd10_code_1 LIKE 'B50%' OR dx.icd10_code_2 LIKE 'B50%'
                 OR dx.icd10_code_1 LIKE 'B51%' OR dx.icd10_code_2 LIKE 'B51%'
                 OR dx.icd10_code_1 LIKE 'B54%' OR dx.icd10_code_2 LIKE 'B54%'
                 OR dx.disease_group_1 ILIKE '%Malaria%'                        THEN 'Malaria'
               WHEN dx.icd10_code_1 LIKE 'J0%'  OR dx.icd10_code_2 LIKE 'J0%'
                 OR dx.disease_group_1 ILIKE '%URTI%'
                 OR dx.disease_group_1 ILIKE '%Upper Resp%'                     THEN 'URTI'
               WHEN dx.icd10_code_1 LIKE 'A01%' OR dx.icd10_code_2 LIKE 'A01%'
                 OR dx.disease_group_1 ILIKE '%Typhoid%'                        THEN 'Typhoid'
               WHEN dx.icd10_code_1 LIKE 'A00%' OR dx.icd10_code_2 LIKE 'A00%'
                 OR dx.icd10_code_1 LIKE 'A02%' OR dx.icd10_code_2 LIKE 'A02%'
                 OR dx.icd10_code_1 LIKE 'A03%' OR dx.icd10_code_2 LIKE 'A03%'
                 OR dx.icd10_code_1 LIKE 'A04%' OR dx.icd10_code_2 LIKE 'A04%'
                 OR dx.icd10_code_1 LIKE 'A05%' OR dx.icd10_code_2 LIKE 'A05%'
                 OR dx.disease_group_1 ILIKE '%Enteric%'
                 OR dx.disease_group_1 ILIKE '%Gastroenterit%'                  THEN 'Enteric / GI'
               WHEN dx.icd10_code_1 LIKE 'B20%' OR dx.icd10_code_2 LIKE 'B20%'
                 OR dx.icd10_code_1 LIKE 'B21%' OR dx.icd10_code_2 LIKE 'B21%'
                 OR dx.disease_group_1 ILIKE '%HIV%'                            THEN 'HIV'
               ELSE NULL
           END AS disease_label
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
    QUALIFY ROW_NUMBER() OVER (PARTITION BY v.id ORDER BY dx.icd10_code_1 NULLS LAST) = 1
),
admissions AS (
    SELECT DISTINCT CAST(visit_id AS VARCHAR) AS adm_visit_id
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS
)
SELECT
    disease_label,
    COUNT(DISTINCT b.patient)  AS patient_count,
    COUNT(DISTINCT b.visit_id) AS visit_count,
    COUNT(DISTINCT CASE WHEN a.adm_visit_id IS NOT NULL THEN b.visit_id END) AS admitted_count,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN a.adm_visit_id IS NOT NULL THEN b.visit_id END),
        COUNT(DISTINCT b.visit_id)
    ) * 100, 1) AS admission_rate_pct
FROM base b
LEFT JOIN admissions a ON CAST(b.visit_id AS VARCHAR) = a.adm_visit_id
WHERE disease_label IS NOT NULL
GROUP BY 1
ORDER BY visit_count DESC
"""
    return run_query(sql)


def load_disease_demographics(filters: dict, run_query) -> pd.DataFrame:
    """D: TB, Malaria, URTI, Typhoid, HIV, Enteric — who they affect (age/sex)."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
base AS (
    SELECT v.source_schema, v.id AS visit_id, v.patient,
           CASE
               WHEN dx.icd10_code_1 LIKE 'A15%' OR dx.icd10_code_1 LIKE 'A16%'
                 OR dx.icd10_code_1 LIKE 'A17%' OR dx.icd10_code_1 LIKE 'A18%'
                 OR dx.icd10_code_1 LIKE 'A19%'
                 OR dx.disease_group_1 ILIKE '%Tubercul%'                       THEN 'TB'
               WHEN dx.icd10_code_1 LIKE 'B5%'
                 OR dx.disease_group_1 ILIKE '%Malaria%'                        THEN 'Malaria'
               WHEN dx.icd10_code_1 LIKE 'J0%'
                 OR dx.disease_group_1 ILIKE '%URTI%'                           THEN 'URTI'
               WHEN dx.icd10_code_1 LIKE 'A01%'
                 OR dx.disease_group_1 ILIKE '%Typhoid%'                        THEN 'Typhoid'
               WHEN dx.icd10_code_1 LIKE 'A0%'
                 OR dx.disease_group_1 ILIKE '%Enteric%'
                 OR dx.disease_group_1 ILIKE '%Gastroenterit%'                  THEN 'Enteric / GI'
               WHEN dx.icd10_code_1 LIKE 'B2%'
                 OR dx.disease_group_1 ILIKE '%HIV%'                            THEN 'HIV'
               ELSE NULL
           END AS disease_label,
           CASE
               WHEN p.dob IS NULL THEN 'Unknown'
               WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 5  THEN 'Under 5'
               WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 18 THEN '5-17'
               WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 35 THEN '18-34'
               WHEN TIMESTAMPDIFF('year', p.dob, v.created_at) < 55 THEN '35-54'
               ELSE '55+'
           END AS age_group,
           CASE WHEN p.sex IS NULL THEN 'Unknown'
                ELSE UPPER(CAST(p.sex AS VARCHAR))
           END AS sex
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS p
        ON v.patient = p.patient_id AND v.source_schema = p.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
    QUALIFY ROW_NUMBER() OVER (PARTITION BY v.id ORDER BY dx.icd10_code_1 NULLS LAST) = 1
)
SELECT
    disease_label,
    age_group,
    sex,
    COUNT(DISTINCT patient) AS patient_count,
    COUNT(DISTINCT visit_id) AS visit_count
FROM base
WHERE disease_label IS NOT NULL
GROUP BY 1, 2, 3
ORDER BY disease_label, patient_count DESC
"""
    return run_query(sql)


def load_disease_monthly_trend(filters: dict, run_query) -> pd.DataFrame:
    """D: Monthly visit trend for TB, Malaria, URTI, Typhoid, HIV, Enteric — spike detection."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
deduped AS (
    SELECT v.source_schema, v.id AS visit_id, v.patient,
           DATE_TRUNC('month', v.created_at) AS visit_month,
           CASE
               WHEN dx.icd10_code_1 LIKE 'A15%' OR dx.icd10_code_1 LIKE 'A16%'
                 OR dx.icd10_code_1 LIKE 'A17%' OR dx.icd10_code_1 LIKE 'A18%'
                 OR dx.icd10_code_1 LIKE 'A19%'
                 OR dx.disease_group_1 ILIKE '%Tubercul%'                       THEN 'TB'
               WHEN dx.icd10_code_1 LIKE 'B5%'
                 OR dx.disease_group_1 ILIKE '%Malaria%'                        THEN 'Malaria'
               WHEN dx.icd10_code_1 LIKE 'J0%'
                 OR dx.disease_group_1 ILIKE '%URTI%'                           THEN 'URTI'
               WHEN dx.icd10_code_1 LIKE 'A01%'
                 OR dx.disease_group_1 ILIKE '%Typhoid%'                        THEN 'Typhoid'
               WHEN dx.icd10_code_1 LIKE 'A0%'
                 OR dx.disease_group_1 ILIKE '%Enteric%'
                 OR dx.disease_group_1 ILIKE '%Gastroenterit%'                  THEN 'Enteric / GI'
               WHEN dx.icd10_code_1 LIKE 'B2%'
                 OR dx.disease_group_1 ILIKE '%HIV%'                            THEN 'HIV'
               ELSE NULL
           END AS disease_label
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
    QUALIFY ROW_NUMBER() OVER (PARTITION BY v.id ORDER BY dx.icd10_code_1 NULLS LAST) = 1
)
SELECT
    visit_month,
    disease_label,
    COUNT(DISTINCT visit_id) AS visit_count,
    COUNT(DISTINCT patient)  AS patient_count
FROM deduped
WHERE disease_label IS NOT NULL
GROUP BY 1, 2
ORDER BY 1, 2
"""
    return run_query(sql)


def load_tb_hiv_coinfection(filters: dict, run_query) -> pd.DataFrame:
    """D: TB-HIV co-infection — patients with both diagnoses, lab HIV test status, vitals markers."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
tb_patients AS (
    SELECT DISTINCT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND (dx.icd10_code_1 LIKE 'A15%' OR dx.icd10_code_2 LIKE 'A15%'
        OR dx.icd10_code_1 LIKE 'A16%' OR dx.icd10_code_2 LIKE 'A16%'
        OR dx.icd10_code_1 LIKE 'A17%' OR dx.icd10_code_2 LIKE 'A17%'
        OR dx.icd10_code_1 LIKE 'A18%' OR dx.icd10_code_2 LIKE 'A18%'
        OR dx.icd10_code_1 LIKE 'A19%' OR dx.icd10_code_2 LIKE 'A19%'
        OR dx.disease_group_1 ILIKE '%Tubercul%')
),
hiv_patients AS (
    SELECT DISTINCT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND (dx.icd10_code_1 LIKE 'B20%' OR dx.icd10_code_2 LIKE 'B20%'
        OR dx.icd10_code_1 LIKE 'B21%' OR dx.icd10_code_2 LIKE 'B21%'
        OR dx.disease_group_1 ILIKE '%HIV%')
),
hiv_test_done AS (
    SELECT DISTINCT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN tb_patients tb ON v.patient = tb.patient AND v.source_schema = tb.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS inv
        ON v.id = inv.visit_id AND v.source_schema = inv.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND (inv.investigation_type ILIKE '%HIV%'
        OR inv.investigation_type ILIKE '%retrovir%'
        OR inv.investigation_type ILIKE '%CD4%'
        OR inv.investigation_type ILIKE '%viral load%')
),
fever_signal AS (
    SELECT DISTINCT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN tb_patients tb ON v.patient = tb.patient AND v.source_schema = tb.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VITALS vt
        ON v.id = vt.visit_id AND v.source_schema = vt.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND vt.temperature IS NOT NULL
      AND vt.temperature > 37.5
)
SELECT
    COUNT(DISTINCT tb.patient)                              AS tb_patients,
    COUNT(DISTINCT hp.patient)                             AS hiv_patients,
    COUNT(DISTINCT CASE WHEN hp.patient IS NOT NULL THEN tb.patient END) AS tb_hiv_coinfected,
    COUNT(DISTINCT htd.patient)                            AS tb_with_hiv_test,
    COUNT(DISTINCT fs.patient)                             AS tb_with_fever,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN hp.patient IS NOT NULL THEN tb.patient END),
        COUNT(DISTINCT tb.patient)
    ) * 100, 1)                                            AS coinfection_rate_pct,
    ROUND(DIV0(
        COUNT(DISTINCT htd.patient),
        COUNT(DISTINCT tb.patient)
    ) * 100, 1)                                            AS hiv_test_coverage_pct
FROM tb_patients tb
LEFT JOIN hiv_patients hp ON tb.patient = hp.patient AND tb.source_schema = hp.source_schema
LEFT JOIN hiv_test_done htd ON tb.patient = htd.patient AND tb.source_schema = htd.source_schema
LEFT JOIN fever_signal fs ON tb.patient = fs.patient AND tb.source_schema = fs.source_schema
"""
    return run_query(sql)


def load_malaria_lab_accuracy(filters: dict, run_query) -> pd.DataFrame:
    """D: Malaria — test done rate, positive vs negative, clinical-only diagnosis rate."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
malaria_visits AS (
    SELECT v.source_schema, v.id AS visit_id, v.patient,
           DATE_TRUNC('month', v.created_at) AS visit_month
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND (dx.icd10_code_1 LIKE 'B5%' OR dx.icd10_code_2 LIKE 'B5%'
        OR dx.disease_group_1 ILIKE '%Malaria%')
    QUALIFY ROW_NUMBER() OVER (PARTITION BY v.id ORDER BY dx.icd10_code_1 NULLS LAST) = 1
),
malaria_tests AS (
    -- any investigation ordered on a malaria visit (investigation_type contains test category)
    SELECT DISTINCT mv.source_schema, mv.visit_id,
           CASE WHEN inv.result_created_at IS NOT NULL THEN 'Resulted'
                ELSE 'Ordered Only'
           END AS test_status
    FROM malaria_visits mv
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS inv
        ON mv.visit_id = inv.visit_id AND mv.source_schema = inv.source_schema
    WHERE (inv.investigation_type ILIKE '%malaria%'
        OR inv.investigation_type ILIKE '%RDT%'
        OR inv.investigation_type ILIKE '%parasit%'
        OR inv.investigation_type ILIKE '%blood film%'
        OR inv.investigation_type ILIKE '%microscopy%'
        OR inv.investigation_type ILIKE '%lab%')
      AND (inv.cancelled IS NULL OR inv.cancelled = 0)
)
SELECT
    COUNT(DISTINCT mv.visit_id)                                                  AS total_malaria_visits,
    COUNT(DISTINCT mt.visit_id)                                                  AS visits_with_test,
    COUNT(DISTINCT CASE WHEN mt.test_status = 'Resulted' THEN mv.visit_id END)  AS test_resulted,
    COUNT(DISTINCT CASE WHEN mt.test_status = 'Ordered Only' THEN mv.visit_id END) AS test_ordered_only,
    COUNT(DISTINCT CASE WHEN mt.visit_id IS NULL THEN mv.visit_id END)           AS no_test_done,
    ROUND(DIV0(
        COUNT(DISTINCT mt.visit_id),
        COUNT(DISTINCT mv.visit_id)
    ) * 100, 1)                                                                  AS test_rate_pct,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN mt.test_status = 'Resulted' THEN mv.visit_id END),
        COUNT(DISTINCT mt.visit_id)
    ) * 100, 1)                                                                  AS result_rate_pct
FROM malaria_visits mv
LEFT JOIN malaria_tests mt ON mv.visit_id = mt.visit_id AND mv.source_schema = mt.source_schema
"""
    return run_query(sql)


def load_communicable_comorbidities(filters: dict, run_query) -> pd.DataFrame:
    """D: Top comorbidities found in patients with each communicable disease."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
comm_patients AS (
    SELECT DISTINCT v.source_schema, v.patient,
           CASE
               WHEN dx.icd10_code_1 LIKE 'A15%' OR dx.icd10_code_1 LIKE 'A16%'
                 OR dx.icd10_code_1 LIKE 'A17%' OR dx.icd10_code_1 LIKE 'A18%'
                 OR dx.icd10_code_1 LIKE 'A19%'
                 OR dx.disease_group_1 ILIKE '%Tubercul%'                       THEN 'TB'
               WHEN dx.icd10_code_1 LIKE 'B5%'
                 OR dx.disease_group_1 ILIKE '%Malaria%'                        THEN 'Malaria'
               WHEN dx.icd10_code_1 LIKE 'J0%'
                 OR dx.disease_group_1 ILIKE '%URTI%'                           THEN 'URTI'
               WHEN dx.icd10_code_1 LIKE 'A01%'
                 OR dx.disease_group_1 ILIKE '%Typhoid%'                        THEN 'Typhoid'
               WHEN dx.icd10_code_1 LIKE 'A0%'
                 OR dx.disease_group_1 ILIKE '%Enteric%'
                 OR dx.disease_group_1 ILIKE '%Gastroenterit%'                  THEN 'Enteric / GI'
               ELSE NULL
           END AS disease_label
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
    QUALIFY ROW_NUMBER() OVER (PARTITION BY v.id ORDER BY dx.icd10_code_1 NULLS LAST) = 1
),
other_conditions AS (
    SELECT cp.disease_label,
           COALESCE(NULLIF(TRIM(dx2.disease_group_1), ''), 'Unclassified') AS comorbidity,
           cp.patient
    FROM comm_patients cp
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v2
        ON cp.patient = v2.patient AND cp.source_schema = v2.source_schema
    INNER JOIN schema_anchor sa ON v2.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx2
        ON v2.id = dx2.visit_id AND v2.source_schema = dx2.source_schema
    WHERE v2.created_at >= DATEADD('month', -{mo}, sa.max_date)
      {wh}
      AND cp.disease_label IS NOT NULL
      AND dx2.disease_group_1 IS NOT NULL
      AND dx2.disease_group_1 NOT ILIKE '%Tubercul%'
      AND dx2.disease_group_1 NOT ILIKE '%Malaria%'
      AND dx2.disease_group_1 NOT ILIKE '%URTI%'
      AND dx2.disease_group_1 NOT ILIKE '%Typhoid%'
      AND dx2.disease_group_1 NOT ILIKE '%Enteric%'
      AND dx2.disease_group_1 NOT ILIKE '%Gastroenterit%'
)
SELECT
    disease_label,
    comorbidity,
    COUNT(DISTINCT patient) AS patient_count
FROM other_conditions
GROUP BY 1, 2
QUALIFY ROW_NUMBER() OVER (PARTITION BY disease_label ORDER BY COUNT(DISTINCT patient) DESC) <= 5
ORDER BY disease_label, patient_count DESC
"""
    return run_query(sql)


def load_mh_diagnostic_breakdown(filters: dict, run_query) -> pd.DataFrame:
    """Mental health visits grouped by category, age group, sex, with payer."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
)
SELECT
    CASE
        WHEN dx.disease_group_1 ILIKE '%Depress%'
          OR dx.disease_group_1 ILIKE '%Anxiety%'     THEN 'Depression & Anxiety'
        WHEN dx.disease_group_1 ILIKE '%Substance%'
          OR dx.disease_group_1 ILIKE '%Alcohol%'     THEN 'Substance & Alcohol'
        WHEN dx.disease_group_1 ILIKE '%Psychos%'
          OR dx.disease_group_1 ILIKE '%Schizo%'
          OR dx.disease_group_1 ILIKE '%Bipolar%'     THEN 'Psychotic Disorders'
        WHEN dx.disease_group_1 ILIKE '%Dementia%'
          OR dx.disease_group_1 ILIKE '%Organic%'     THEN 'Dementia / Organic Brain'
        ELSE 'Other Mental Health'
    END                                               AS mh_category,
    CASE
        WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 18 THEN 'Under 18'
        WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 35 THEN '18-34'
        WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 50 THEN '35-49'
        WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 65 THEN '50-64'
        ELSE '65+'
    END                                               AS age_group,
    UPPER(COALESCE(rp.sex, 'Unknown'))                AS sex,
    COUNT(DISTINCT v.patient)                         AS patient_count,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN ia.visit_id IS NOT NULL THEN v.id END),
        COUNT(DISTINCT v.id)
    ) * 100, 1)                                       AS inpatient_pct,
    CASE
        WHEN UPPER(v.payment_mode) IN ('CASH','PRIVATE') THEN 'Cash'
        WHEN UPPER(v.payment_mode) LIKE '%NHIF%'
          OR UPPER(v.payment_mode) LIKE '%SHA%'        THEN 'NHIF / SHA'
        ELSE 'Insurance / Corporate'
    END                                               AS primary_payer
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
    ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS rp
    ON v.patient = rp.patient_id AND v.source_schema = rp.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS ia
    ON v.id = ia.visit_id
   AND v.source_schema = REPLACE(LOWER(ia.source_schema), '_clean', '')
WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
  AND dx.disease_burden_group_1 ILIKE '%Mental%'
{wh}
GROUP BY 1, 2, 3, 6
ORDER BY patient_count DESC
"""
    return run_query(sql)


def load_mh_comorbidity_profile(filters: dict, run_query) -> pd.DataFrame:
    """Per MH category: standalone vs comorbid patients and top comorbidity."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
mh_patients AS (
    SELECT DISTINCT
        v.source_schema, v.patient,
        CASE
            WHEN dx.disease_group_1 ILIKE '%Depress%'
              OR dx.disease_group_1 ILIKE '%Anxiety%'     THEN 'Depression & Anxiety'
            WHEN dx.disease_group_1 ILIKE '%Substance%'
              OR dx.disease_group_1 ILIKE '%Alcohol%'     THEN 'Substance & Alcohol'
            WHEN dx.disease_group_1 ILIKE '%Psychos%'
              OR dx.disease_group_1 ILIKE '%Schizo%'
              OR dx.disease_group_1 ILIKE '%Bipolar%'     THEN 'Psychotic Disorders'
            WHEN dx.disease_group_1 ILIKE '%Dementia%'
              OR dx.disease_group_1 ILIKE '%Organic%'     THEN 'Dementia / Organic Brain'
            ELSE 'Other Mental Health'
        END AS mh_category
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      AND dx.disease_burden_group_1 ILIKE '%Mental%'
    {wh}
),
non_mh_dx AS (
    SELECT DISTINCT v.source_schema, v.patient,
           dx.disease_burden_group_1 AS other_condition
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    INNER JOIN mh_patients mp
        ON v.patient = mp.patient AND v.source_schema = mp.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      AND dx.disease_burden_group_1 IS NOT NULL
      AND dx.disease_burden_group_1 NOT ILIKE '%Mental%'
    {wh}
),
top_comorbidity AS (
    SELECT mh_category, top_comorbidity
    FROM (
        SELECT mp.mh_category,
               nd.other_condition AS top_comorbidity,
               COUNT(DISTINCT mp.patient) AS cnt
        FROM mh_patients mp
        INNER JOIN non_mh_dx nd
            ON mp.patient = nd.patient AND mp.source_schema = nd.source_schema
        GROUP BY 1, 2
    )
    QUALIFY ROW_NUMBER() OVER (PARTITION BY mh_category ORDER BY cnt DESC) = 1
)
SELECT
    mp.mh_category,
    COUNT(DISTINCT CASE WHEN nd.patient IS NULL THEN mp.patient END) AS standalone_patients,
    COUNT(DISTINCT CASE WHEN nd.patient IS NOT NULL THEN mp.patient END) AS comorbid_patients,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN nd.patient IS NULL THEN mp.patient END),
        COUNT(DISTINCT mp.patient)
    ) * 100, 1)                                                      AS standalone_pct,
    tc.top_comorbidity
FROM mh_patients mp
LEFT JOIN (SELECT DISTINCT source_schema, patient FROM non_mh_dx) nd
    ON mp.patient = nd.patient AND mp.source_schema = nd.source_schema
LEFT JOIN top_comorbidity tc ON mp.mh_category = tc.mh_category
GROUP BY mp.mh_category, tc.top_comorbidity
ORDER BY comorbid_patients DESC
"""
    return run_query(sql)


def load_mh_monthly_trend(filters: dict, run_query) -> pd.DataFrame:
    """Mental health visit and patient count per month and MH category — MoM trend."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
)
SELECT
    DATE_TRUNC('month', v.created_at)                   AS visit_month,
    CASE
        WHEN dx.disease_group_1 ILIKE '%Depress%'
          OR dx.disease_group_1 ILIKE '%Anxiety%'       THEN 'Depression & Anxiety'
        WHEN dx.disease_group_1 ILIKE '%Substance%'
          OR dx.disease_group_1 ILIKE '%Alcohol%'       THEN 'Substance & Alcohol'
        WHEN dx.disease_group_1 ILIKE '%Psychos%'
          OR dx.disease_group_1 ILIKE '%Schizo%'
          OR dx.disease_group_1 ILIKE '%Bipolar%'       THEN 'Psychotic Disorders'
        WHEN dx.disease_group_1 ILIKE '%Dementia%'
          OR dx.disease_group_1 ILIKE '%Organic%'       THEN 'Dementia / Organic Brain'
        ELSE 'Other Mental Health'
    END                                                 AS mh_category,
    COUNT(DISTINCT v.id)                                AS visit_count,
    COUNT(DISTINCT v.patient)                           AS patient_count
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
    ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
  AND dx.disease_burden_group_1 ILIKE '%Mental%'
{wh}
GROUP BY 1, 2
ORDER BY 1
"""
    return run_query(sql)


def load_bounce_back_patients(filters: dict, run_query) -> pd.DataFrame:
    """Chronic NCD inpatients seen in outpatient within 72h of admission — bounce-back list."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
admissions AS (
    SELECT
        v.source_schema, v.patient,
        ia.admitted_at, ia.los_days,
        dx.disease_burden_group_1   AS condition,
        CASE
            WHEN UPPER(v.payment_mode) IN ('CASH','PRIVATE') THEN 'Cash'
            WHEN UPPER(v.payment_mode) LIKE '%NHIF%'
              OR UPPER(v.payment_mode) LIKE '%SHA%'           THEN 'NHIF / SHA'
            ELSE 'Insurance / Corporate'
        END                         AS payer
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS ia
        ON v.id = ia.visit_id
       AND v.source_schema = REPLACE(LOWER(ia.source_schema), '_clean', '')
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      AND dx.disease_burden_group_1 ILIKE ANY (
          '%Cardiovascular%','%Diabetes%','%Chronic%','%Neurolog%',
          '%Mental%','%Endocrin%','%Musculo%')
    {wh}
),
followup AS (
    SELECT
        v2.source_schema, v2.patient,
        v2.created_at                           AS readmission_date,
        dx2.disease_burden_group_1              AS followup_condition
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v2
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx2
        ON v2.id = dx2.visit_id AND v2.source_schema = dx2.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS ia2
        ON v2.id = ia2.visit_id
    WHERE ia2.visit_id IS NULL  -- outpatient only
)
SELECT
    ROW_NUMBER() OVER (ORDER BY a.admitted_at DESC)      AS patient,
    a.condition,
    a.admitted_at                                        AS admission_date,
    DATEDIFF('hour', a.admitted_at, f.readmission_date)  AS readmission_hours,
    a.payer,
    a.los_days
FROM admissions a
INNER JOIN followup f
    ON a.patient = f.patient
   AND a.source_schema = f.source_schema
   AND f.readmission_date > a.admitted_at
   AND DATEDIFF('hour', a.admitted_at, f.readmission_date) <= 72
   AND a.condition = f.followup_condition
ORDER BY readmission_hours ASC
LIMIT 50
"""
    return run_query(sql)


def load_elevated_vitals_no_ncd_patients(filters: dict, run_query) -> pd.DataFrame:
    """Patients with persistently elevated vitals (>=2 visits) but no NCD diagnosis — patient list."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
elevated AS (
    SELECT v.source_schema, v.patient,
           COUNT(DISTINCT v.id)           AS visit_count,
           MAX(vt.bp_systolic)            AS latest_systolic,
           MAX(vt.blood_sugar)            AS latest_blood_sugar,
           MAX(v.created_at)              AS last_visit_date,
           CASE
               WHEN UPPER(v.payment_mode) IN ('CASH','PRIVATE') THEN 'Cash'
               WHEN UPPER(v.payment_mode) LIKE '%NHIF%'
                 OR UPPER(v.payment_mode) LIKE '%SHA%'           THEN 'NHIF / SHA'
               ELSE 'Insurance / Corporate'
           END                            AS payer
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VITALS vt
        ON vt.visit_id = v.id AND vt.source_schema = v.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      AND (vt.bp_systolic >= 140 OR vt.blood_sugar >= 10)
    {wh}
    GROUP BY 1, 2, 7
    HAVING COUNT(DISTINCT v.id) >= 2
),
ncd_patients AS (
    SELECT DISTINCT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE dx.disease_burden_group_1 ILIKE ANY (
          '%Cardiovascular%','%Diabetes%','%Endocrin%')
)
SELECT
    ROW_NUMBER() OVER (ORDER BY e.visit_count DESC)      AS patient,
    e.visit_count,
    e.latest_systolic,
    e.latest_blood_sugar,
    DATEDIFF('day', e.last_visit_date, sa.max_date)       AS days_since_last_visit,
    e.payer
FROM elevated e
INNER JOIN schema_anchor sa ON e.source_schema = sa.source_schema
LEFT JOIN ncd_patients np
    ON e.patient = np.patient AND e.source_schema = np.source_schema
WHERE np.patient IS NULL
ORDER BY e.visit_count DESC
LIMIT 50
"""
    return run_query(sql)


def load_patient_visit_gap_profile(filters: dict, run_query) -> pd.DataFrame:
    """Overdue chronic patients: actual vs expected visit gap, ordered by variance."""
    wh = _w(filters)
    wsa = _wsa(filters)
    mo = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
chronic_patients AS (
    SELECT DISTINCT
        v.source_schema, v.patient,
        dx.disease_burden_group_1                         AS condition,
        CASE
            WHEN dx.disease_group_1 ILIKE '%Hypertension%'
              OR dx.disease_group_1 ILIKE '%HTN%'         THEN 30
            WHEN dx.disease_group_1 ILIKE '%Diabetes%'   THEN 30
            WHEN dx.disease_group_1 ILIKE '%HIV%'         THEN 90
            ELSE 60
        END                                               AS expected_gap_days,
        CASE
            WHEN UPPER(v.payment_mode) IN ('CASH','PRIVATE') THEN 'Cash'
            WHEN UPPER(v.payment_mode) LIKE '%NHIF%'
              OR UPPER(v.payment_mode) LIKE '%SHA%'        THEN 'NHIF / SHA'
            ELSE 'Insurance / Corporate'
        END                                               AS payer
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      AND (dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1)
    {wh}
),
last_visit AS (
    SELECT v.source_schema, v.patient, MAX(v.created_at) AS last_visit_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
    GROUP BY 1, 2
),
lifecycle AS (
    SELECT lv.source_schema, lv.patient,
           CASE
               WHEN DATEDIFF('day', lv.last_visit_date, sa.max_date) <= 30  THEN 'Active'
               WHEN DATEDIFF('day', lv.last_visit_date, sa.max_date) <= 90  THEN 'Lapsing'
               ELSE 'LTFU'
           END AS lifecycle
    FROM last_visit lv
    INNER JOIN schema_anchor sa ON lv.source_schema = sa.source_schema
)
SELECT
    ROW_NUMBER() OVER (ORDER BY
        DATEDIFF('day', lv.last_visit_date, sa.max_date) - cp.expected_gap_days DESC
    )                                                     AS patient,
    cp.condition,
    DATEDIFF('day', lv.last_visit_date, sa.max_date)      AS days_since_last_visit,
    cp.expected_gap_days,
    DATEDIFF('day', lv.last_visit_date, sa.max_date)
        - cp.expected_gap_days                            AS gap_variance_days,
    cp.payer,
    lc.lifecycle
FROM chronic_patients cp
INNER JOIN last_visit lv
    ON cp.patient = lv.patient AND cp.source_schema = lv.source_schema
INNER JOIN schema_anchor sa ON cp.source_schema = sa.source_schema
LEFT JOIN lifecycle lc
    ON cp.patient = lc.patient AND cp.source_schema = lc.source_schema
WHERE DATEDIFF('day', lv.last_visit_date, sa.max_date) - cp.expected_gap_days > 0
ORDER BY gap_variance_days DESC
LIMIT 50
"""
    return run_query(sql)


def load_priority_patients(filters: dict, run_query) -> pd.DataFrame:
    """Priority patient list for Clinician View: chronic, undetected NCD, OP→IP escalation."""
    wh     = _w(filters)
    wh_op  = _w(filters, alias="v_op")
    wsa    = _wsa(filters)
    mo     = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
all_patients AS (
    SELECT DISTINCT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
),
last_visit AS (
    SELECT v.source_schema, v.patient,
           MAX(v.created_at) AS last_visit_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh}
    GROUP BY 1, 2
),
latest_clinician AS (
    SELECT v.source_schema, v.patient,
           CAST(v.user AS VARCHAR) AS current_clinician
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      AND v.user IS NOT NULL
    {wh}
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY v.source_schema, v.patient ORDER BY v.created_at DESC
    ) = 1
),
clinician_count AS (
    SELECT v.source_schema, v.patient,
           COUNT(DISTINCT v.user) AS unique_clinicians
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      AND v.user IS NOT NULL
    {wh}
    GROUP BY 1, 2
),
chronic AS (
    SELECT v.source_schema, v.patient,
           MAX(NULLIF(TRIM(dx.disease_burden_group_1), '')) AS primary_condition,
           1 AS is_chronic
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      AND (dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1)
    {wh}
    GROUP BY 1, 2
),
elevated_vitals AS (
    SELECT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VITALS vt
        ON vt.visit_id = v.id AND vt.source_schema = v.source_schema
    WHERE v.created_at >= DATEADD('month', -{mo}, sa.max_date)
      AND (vt.bp_systolic >= 140 OR vt.blood_sugar >= 10)
    {wh}
    GROUP BY 1, 2
    HAVING COUNT(DISTINCT v.id) >= 2
),
ncd_diagnosed AS (
    SELECT DISTINCT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE dx.disease_burden_group_1 ILIKE ANY (
          '%Cardiovascular%','%Diabetes%','%Endocrin%')
),
undetected_ncd AS (
    SELECT ev.source_schema, ev.patient, 1 AS has_undetected_ncd
    FROM elevated_vitals ev
    LEFT JOIN ncd_diagnosed nd
        ON ev.patient = nd.patient AND ev.source_schema = nd.source_schema
    WHERE nd.patient IS NULL
),
op_to_ip AS (
    SELECT DISTINCT v_op.source_schema, v_op.patient, 1 AS had_op_to_ip
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v_op
    INNER JOIN schema_anchor sa ON v_op.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx_op
        ON v_op.id = dx_op.visit_id AND v_op.source_schema = dx_op.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v_ip
        ON v_op.patient = v_ip.patient AND v_op.source_schema = v_ip.source_schema
       AND v_ip.created_at > v_op.created_at
       AND DATEDIFF('day', v_op.created_at, v_ip.created_at) <= 14
    INNER JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS ia
        ON CAST(v_ip.id AS VARCHAR) = CAST(ia.visit_id AS VARCHAR)
       AND v_op.source_schema = REPLACE(LOWER(ia.source_schema), '_clean', '')
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx_ip
        ON v_ip.id = dx_ip.visit_id AND v_ip.source_schema = dx_ip.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS ia_op
        ON CAST(v_op.id AS VARCHAR) = CAST(ia_op.visit_id AS VARCHAR)
    WHERE ia_op.visit_id IS NULL
      AND dx_op.disease_burden_group_1 = dx_ip.disease_burden_group_1
      AND v_op.created_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh_op}
),
demographics AS (
    SELECT p.patient_id, p.source_schema,
           CASE WHEN p.sex IS NULL THEN 'Unknown' ELSE UPPER(CAST(p.sex AS VARCHAR)) END AS gender,
           CASE
               WHEN p.dob IS NULL THEN 'Unknown'
               WHEN TIMESTAMPDIFF('year', p.dob, CURRENT_DATE) < 18  THEN 'Child'
               WHEN TIMESTAMPDIFF('year', p.dob, CURRENT_DATE) < 35  THEN 'Young Adult (18-34)'
               WHEN TIMESTAMPDIFF('year', p.dob, CURRENT_DATE) < 55  THEN 'Adult (35-54)'
               WHEN TIMESTAMPDIFF('year', p.dob, CURRENT_DATE) < 65  THEN 'Older Adult (55-64)'
               ELSE 'Senior (65+)'
           END                                             AS age_group
    FROM HOSPITALS.STAGING.STG_RECEPTION_PATIENTS p
    INNER JOIN all_patients ap
        ON p.patient_id = ap.patient AND p.source_schema = ap.source_schema
)
SELECT
    ap.patient,
    ap.source_schema,
    COALESCE(d.gender, 'Unknown')                         AS gender,
    COALESCE(d.age_group, 'Unknown')                      AS age_group,
    COALESCE(ch.primary_condition, 'Not recorded')        AS primary_condition,
    COALESCE(ch.is_chronic, 0)                            AS is_chronic,
    DATEDIFF('day', lv.last_visit_date, sa.max_date)      AS days_since_last_visit,
    COALESCE(un.has_undetected_ncd, 0)                    AS has_undetected_ncd,
    COALESCE(oi.had_op_to_ip, 0)                          AS had_op_to_ip,
    COALESCE(cc.unique_clinicians, 1)                     AS unique_clinicians,
    COALESCE(lc.current_clinician, 'Unknown')             AS current_clinician,
    CASE
        WHEN (COALESCE(ch.is_chronic, 0) = 1
              AND DATEDIFF('day', lv.last_visit_date, sa.max_date) >= 90)
          OR COALESCE(un.has_undetected_ncd, 0) = 1
          OR COALESCE(oi.had_op_to_ip, 0) = 1             THEN 'HIGH'
        WHEN (COALESCE(ch.is_chronic, 0) = 1
              AND DATEDIFF('day', lv.last_visit_date, sa.max_date) >= 30)
          OR COALESCE(cc.unique_clinicians, 1) >= 3        THEN 'MEDIUM'
        ELSE 'MONITOR'
    END                                                   AS priority_flag
FROM all_patients ap
INNER JOIN last_visit lv
    ON ap.patient = lv.patient AND ap.source_schema = lv.source_schema
INNER JOIN schema_anchor sa ON ap.source_schema = sa.source_schema
LEFT JOIN chronic ch
    ON ap.patient = ch.patient AND ap.source_schema = ch.source_schema
LEFT JOIN undetected_ncd un
    ON ap.patient = un.patient AND ap.source_schema = un.source_schema
LEFT JOIN op_to_ip oi
    ON ap.patient = oi.patient AND ap.source_schema = oi.source_schema
LEFT JOIN clinician_count cc
    ON ap.patient = cc.patient AND ap.source_schema = cc.source_schema
LEFT JOIN latest_clinician lc
    ON ap.patient = lc.patient AND ap.source_schema = lc.source_schema
LEFT JOIN demographics d
    ON ap.patient = d.patient_id AND ap.source_schema = d.source_schema
ORDER BY
    CASE priority_flag WHEN 'HIGH' THEN 1 WHEN 'MEDIUM' THEN 2 ELSE 3 END,
    days_since_last_visit DESC
LIMIT 300
"""
    return run_query(sql)


def load_patient_medication_change_timeline(patient_id: str, source_schema: str,
                                             run_query) -> pd.DataFrame:
    """For a specific patient: prescription history with drug-change flags per condition."""
    sql = f"""
WITH rx_ordered AS (
    SELECT
        pp.prescription_created_at                        AS prescription_date,
        pp.drug_name,
        LAG(pp.drug_name) OVER (
            ORDER BY pp.prescription_created_at
        )                                                 AS prev_drug,
        LAG(pp.prescription_created_at) OVER (
            ORDER BY pp.prescription_created_at
        )                                                 AS prev_prescription_date
    FROM HOSPITALS.STAGING.STG_PRESCRIPTION_PAYMENTS pp
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON pp.visit_id = v.id AND pp.source_schema = v.source_schema
    WHERE v.patient       = '{patient_id}'
      AND v.source_schema = '{source_schema}'
      AND (pp.stopped IS NULL OR pp.stopped = 0)
      AND (pp.canceled IS NULL OR pp.canceled = 0)
      AND (pp.remove_from_report IS NULL OR pp.remove_from_report = 0)
)
SELECT
    prescription_date,
    drug_name,
    CASE WHEN prev_drug IS NULL OR drug_name != prev_drug THEN 1 ELSE 0 END AS is_new_drug,
    prev_drug,
    DATEDIFF('day', prev_prescription_date, prescription_date) AS days_since_last_prescription
FROM rx_ordered
ORDER BY prescription_date DESC
"""
    return run_query(sql)


def load_patient_illness_history(patient_id: str, source_schema: str,
                                  run_query) -> pd.DataFrame:
    """Full illness timeline for a specific patient: visits, diagnoses, admissions."""
    sql = f"""
SELECT
    v.created_at                                          AS visit_date,
    dx.disease_group_1                                    AS disease_group,
    dx.disease_burden_group_1,
    CASE WHEN ia.visit_id IS NOT NULL THEN 'Inpatient' ELSE 'Outpatient' END AS visit_type,
    ia.los_days,
    CASE
        WHEN UPPER(v.payment_mode) IN ('CASH','PRIVATE') THEN 'Cash'
        WHEN UPPER(v.payment_mode) LIKE '%NHIF%'
          OR UPPER(v.payment_mode) LIKE '%SHA%'           THEN 'NHIF / SHA'
        ELSE 'Insurance / Corporate'
    END                                                   AS payer
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
    ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS ia
    ON v.id = ia.visit_id
   AND v.source_schema = REPLACE(LOWER(ia.source_schema), '_clean', '')
WHERE v.patient       = '{patient_id}'
  AND v.source_schema = '{source_schema}'
ORDER BY v.created_at DESC
LIMIT 50
"""
    return run_query(sql)


def load_patient_visit_cadence(patient_id: str, source_schema: str,
                                run_query) -> pd.DataFrame:
    """Per-patient visit cadence: each visit with gap from previous and visit type."""
    sql = f"""
SELECT
    CONVERT_TIMEZONE('UTC', 'Africa/Nairobi', v.created_at)  AS visit_date,
    CASE WHEN ia.visit_id IS NOT NULL THEN 'Inpatient' ELSE 'Outpatient' END AS visit_type,
    DATEDIFF('day',
        LAG(v.created_at) OVER (ORDER BY v.created_at),
        v.created_at
    )                                                         AS gap_days,
    DATEDIFF('day',
        v.created_at,
        LEAD(v.created_at) OVER (ORDER BY v.created_at)
    )                                                         AS days_to_next
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS ia
    ON v.id = ia.visit_id
   AND v.source_schema = REPLACE(LOWER(ia.source_schema), '_clean', '')
WHERE v.patient       = '{patient_id}'
  AND v.source_schema = '{source_schema}'
ORDER BY visit_date
"""
    return run_query(sql)


def load_patient_lab_tests(patient_id: str, source_schema: str,
                            run_query) -> pd.DataFrame:
    """Per-patient: all investigations ordered, sorted most recent first. Flags abnormals."""
    sql = f"""
SELECT
    CONVERT_TIMEZONE('UTC', 'Africa/Nairobi', i.investigation_created_at) AS test_date,
    i.procedure_name,
    INITCAP(TRIM(i.investigation_type))                       AS investigation_type,
    i.flag,
    i.alert_level,
    CASE
        WHEN DATEDIFF('minute', i.investigation_created_at, i.result_created_at)
             BETWEEN 0 AND 2880
        THEN DATEDIFF('minute', i.investigation_created_at, i.result_created_at)
        ELSE NULL
    END                                                       AS turnaround_mins
FROM HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS i
INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    ON i.visit_id = v.id
WHERE v.patient       = '{patient_id}'
  AND v.source_schema = '{source_schema}'
  AND (i.cancelled IS NULL OR i.cancelled = 0)
  AND (i.remove_from_report IS NULL OR i.remove_from_report = 0)
  AND i.procedure_name IS NOT NULL
ORDER BY test_date DESC
LIMIT 60
"""
    return run_query(sql)