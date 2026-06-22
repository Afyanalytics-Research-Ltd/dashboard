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
  load_lab_bottleneck_kpis     C2a: active admissions + avg LOS with vs without lab
  load_lab_bottleneck_by_discipline C2b: per-discipline turnaround + LOS on IP visits
  load_same_day_escalation_kpis    C3a: total same-day OP→IP escalations + rate + top condition
  load_same_day_escalation_by_condition C3b: per-condition escalation count + rate
  load_encounter_forecast      Q4:  forecast with confidence intervals, split by type
  load_peak_demand_heatmap     Q6A: hour × day heatmap with visit type split

Tab 3 — Retention:
  load_retention_kpis          KPI row
  load_lifecycle               Q1:  lifecycle active/lapsing/LTFU
  load_retention_by_payer      Q3:  90-day retention by payer
  load_dropout_causes          Q6:  dropout cause attribution
  load_revenue_at_risk         Q7:  revenue at risk from LTFU
  load_lapsing_by_payer             lapsing chronic patients by payer (same population as load_lifecycle)
  load_outreach_list           Q11: re-engagement outreach list
  load_ltfu_priority2_patients R10: one row per chronic LTFU patient — profile dimensions for P2 section

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

OPD to IPD Conversion Tab:
  load_opd_ipd_segments        header KPIs + 4 segment rows (Chronic/Maternal/Oncology/MH)
  load_opd_ipd_monthly         monthly overall + retention-universe dual trend
  load_opd_ipd_by_diagnosis    Section B: diagnosis-level benchmark
  load_comorbidity_conversion  Section C: single / comorbid / chronic-comorbid monthly
  load_opd_ipd_chronic_by_age  Section C right: chronic conversion by age group
  load_escalation_by_age       Section D: 72h OP→IP escalations by age group
  load_operational_triangle    Section E: monthly clinician strain + wait gap + signal

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

def _fmt_date(val) -> str:
    """Normalise any date value to YYYY-MM-DD so Snowflake can parse it reliably."""
    return str(val).replace("/", "-")


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
        parts.append(f"AND {alias}.created_at >= '{_fmt_date(filters['date_from'])}'")
    if filters.get("date_to"):
        parts.append(f"AND {alias}.created_at <= '{_fmt_date(filters['date_to'])}'")
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
        parts.append(f"AND {alias}.admitted_at >= '{_fmt_date(filters['date_from'])}'")
    if filters.get("date_to"):
        parts.append(f"AND {alias}.admitted_at <= '{_fmt_date(filters['date_to'])}'")
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
        "Last 24 months": 24,
        "Last 12 months": 12,
        "Last 6 months":  6,
        "Last 90 days":   3,
    }
    return mapping.get(filters.get("date_range", "Last 24 months"), 24)


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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE a.admitted_at >= '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
          AND (procedure_clinical_division IN ('Pathology / Laboratory Medicine', 'Radiology & Imaging')
               OR LOWER(TRIM(investigation_type)) IN ('laboratory', 'lab', 'radiology', 'ultrasound'))
    ) inv ON v.id = inv.visit_id AND v.source_schema = inv.source_schema
    LEFT JOIN (
        SELECT DISTINCT visit_id, source_schema
        FROM HOSPITALS.STAGING.STG_PRESCRIPTION_PAYMENTS
        WHERE remove_from_report IS NULL OR remove_from_report = 0
    ) pp ON v.id = pp.visit_id AND v.source_schema = pp.source_schema
    WHERE v.created_at >= '2024-09-01'
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
    WHERE a.admitted_at >= '2024-09-01'
    {wh}
    GROUP BY 1, 2, 3, 4
),
ward_beds AS (
    -- Count of physical beds per ward across selected schemas
    SELECT
        COALESCE(a.ward_name, 'Unknown')   AS ward,
        COUNT(DISTINCT b.composite_bed_id) AS num_beds
    FROM HOSPITALS.STAGING.STG_BEDS b
    INNER JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
        ON b.source_schema = a.source_schema
       AND b.ward_id       = a.ward_id
    INNER JOIN schema_anchor sa
        ON REPLACE(LOWER(b.source_schema), '_clean', '') = sa.source_schema
    GROUP BY 1
)
SELECT
    COALESCE(a.ward_name, 'Unknown')                        AS ward,
    COALESCE(a.ward_category, 'Unknown')                    AS ward_category,
    COALESCE(wb.num_beds, 0)                                AS num_beds,
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
LEFT JOIN ward_beds wb
    ON COALESCE(a.ward_name, 'Unknown') = wb.ward
WHERE a.admitted_at >= '2024-09-01'
{wh}
GROUP BY 1, 2, 3
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
    WHERE a.admitted_at >= '2024-09-01'
    {wh}
    GROUP BY 1, 2
),
top_wards AS (
    SELECT ward FROM monthly GROUP BY ward ORDER BY SUM(admissions) DESC LIMIT 6
),
filtered AS (
    SELECT m.visit_month, m.ward, m.admissions
    FROM monthly m
    INNER JOIN top_wards tw ON m.ward = tw.ward
)
SELECT
    visit_month,
    ward,
    admissions,
    CASE
        WHEN ROW_NUMBER() OVER (PARTITION BY ward ORDER BY visit_month) < 3 THEN NULL
        ELSE ROUND(
            AVG(admissions) OVER (
                PARTITION BY ward ORDER BY visit_month
                ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
            ), 1)
    END AS admissions_3mo_avg
FROM filtered
ORDER BY ward, visit_month
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
WHERE a.admitted_at >= '2024-09-01'
  AND a.is_open_admission = 0
  AND a.discharge_latency_hours IS NOT NULL
{wh}
GROUP BY 1
HAVING patient_count >= 5
ORDER BY avg_discharge_latency_hrs DESC
"""
    return run_query(sql)


def load_ward_active_vs_hours(filters: dict, run_query) -> pd.DataFrame:
    """Monthly admissions vs avg hours from evaluation visit to admission.

    avg_admission_hours = DATEDIFF(hour, MAX(v.created_at), a.admitted_at)
    — time between when the evaluation visit was recorded and when the
    patient was formally admitted as inpatient.
    """
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
eval_latest AS (
    -- Latest evaluation visit timestamp per visit_id
    SELECT
        v.id            AS visit_id,
        v.source_schema,
        MAX(v.created_at) AS eval_created_at
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    GROUP BY 1, 2
)
SELECT
    DATE_TRUNC('month', a.admitted_at)                          AS visit_month,
    COUNT(DISTINCT CASE WHEN a.is_open_admission = 1
                        THEN a.visit_id END)                    AS active_admissions,
    ROUND(AVG(
        CASE WHEN a.admitted_at > el.eval_created_at
             THEN DATEDIFF('hour', el.eval_created_at, a.admitted_at)
        END
    ), 1)                                                       AS avg_admission_hours,
    COUNT(DISTINCT a.visit_id)                                  AS total_admissions
FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
INNER JOIN schema_anchor sa
    ON REPLACE(LOWER(a.source_schema), '_clean', '') = sa.source_schema
LEFT JOIN eval_latest el
    ON  a.visit_id      = el.visit_id
    AND REPLACE(LOWER(a.source_schema), '_clean', '') = el.source_schema
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
WHERE a.admitted_at >= '2024-09-01'
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
    WHERE a.admitted_at >= '2024-09-01'
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
    WHERE v.created_at >= '2024-09-01' -- Use a fixed recent anchor to ensure atf logic is consistent and meaningful
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
    WHERE a.admitted_at >= '2024-09-01'
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
top3_conditions AS (
    SELECT
        ward,
        LISTAGG(
            condition_group || ' (' || ROUND(DIV0(cnt, total_dx) * 100, 0)::INT || '%)',
            ' · '
        ) WITHIN GROUP (ORDER BY rk) AS top_conditions
    FROM top_condition
    WHERE rk <= 3
    GROUP BY 1
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
    -- Top 3 conditions
    t3.top_conditions,
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
LEFT  JOIN top3_conditions t3 ON tw.ward = t3.ward
LEFT  JOIN pt_raw pt   ON tw.ward = pt.ward
LEFT  JOIN top_payer tp ON tw.ward = tp.ward AND tp.rk = 1
LEFT  JOIN los l       ON tw.ward = l.ward
LEFT  JOIN inv i       ON tw.ward = i.ward
LEFT  JOIN clinicians cl ON tw.ward = cl.ward
GROUP BY tw.ward, pr.recent_avg, pr.prior_avg, pr.monthly_stddev, pr.monthly_mean,
         t3.top_conditions,
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
    WHERE a.admitted_at >= '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
WHERE v.created_at >=  '2024-09-01'
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
WHERE v.created_at >=  '2024-09-01'
{wh}
GROUP BY 1, 2
ORDER BY 1, 3 DESC
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
    WHERE v.created_at >=  '2024-09-01'
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
      AND (procedure_clinical_division IN ('Pathology / Laboratory Medicine', 'Radiology & Imaging')
           OR LOWER(TRIM(investigation_type)) IN ('laboratory', 'lab', 'radiology', 'ultrasound'))
),
inv_resulted_24h AS (
    SELECT DISTINCT visit_id, source_schema
    FROM HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS
    WHERE (cancelled IS NULL OR cancelled = 0)
      AND (remove_from_report IS NULL OR remove_from_report = 0)
      AND (procedure_clinical_division IN ('Pathology / Laboratory Medicine', 'Radiology & Imaging')
           OR LOWER(TRIM(investigation_type)) IN ('laboratory', 'lab', 'radiology', 'ultrasound'))
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


def load_same_day_escalation_kpis(filters: dict, run_query) -> pd.DataFrame:
    """Same-day OP→IP escalation KPIs: total escalations, rate vs all visits, top condition."""
    wh   = _w(filters)
    wh_a = _w_adm(filters)
    wsa  = _wsa(filters)
    mo   = _mo(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    {wsa}
    GROUP BY source_schema
),
admissions AS (
    SELECT
        a.visit_id,
        REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    WHERE 1=1
    {wh_a}
),
escalated_visits AS (
    SELECT
        v.id AS visit_id,
        v.source_schema
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN admissions a ON v.id = a.visit_id AND v.source_schema = a.source_schema
    WHERE v.created_at >=  '2024-09-01'
    {wh}
),
all_visit_count AS (
    SELECT COUNT(DISTINCT v.id) AS total_op_visits
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >=  '2024-09-01'
    {wh}
),
top_cond AS (
    SELECT
        COALESCE(dx.disease_burden_group_1, 'Not recorded') AS condition,
        COUNT(DISTINCT ev.visit_id) AS esc_count
    FROM escalated_visits ev
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON ev.visit_id = dx.visit_id AND ev.source_schema = dx.source_schema
    GROUP BY 1
    ORDER BY 2 DESC
    LIMIT 1
)
SELECT
    COUNT(DISTINCT ev.visit_id)                                                    AS total_escalations,
    av.total_op_visits,
    ROUND(100.0 * COUNT(DISTINCT ev.visit_id) / NULLIF(av.total_op_visits, 0), 2) AS escalation_rate_pct,
    MAX(tc.condition)                                                              AS top_condition
FROM escalated_visits ev
CROSS JOIN all_visit_count av
CROSS JOIN top_cond tc
GROUP BY av.total_op_visits, tc.condition
"""
    return run_query(sql)


def load_same_day_escalation_by_condition(filters: dict, run_query) -> pd.DataFrame:
    """Same-day OP→IP escalation per condition: count + escalation rate vs all visits for that condition.
    Escalation = OPD visit followed by admission within 72 hours."""
    wh   = _w(filters)
    wh_a = _w_adm(filters)
    wsa  = _wsa(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    {wsa}
    GROUP BY source_schema
),
admissions AS (
    SELECT
        a.visit_id,
        a.admitted_at,
        REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    WHERE 1=1
    {wh_a}
),
all_visits_by_cond AS (
    SELECT
        v.id          AS visit_id,
        v.created_at  AS visit_time,
        v.source_schema,
        COALESCE(dx.disease_burden_group_1, 'Not recorded') AS condition
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE v.created_at >= '2024-09-01'
    {wh}
),
escalations AS (
    SELECT
        av.condition,
        COUNT(DISTINCT av.visit_id) AS esc_count
    FROM all_visits_by_cond av
    INNER JOIN admissions a
        ON av.visit_id = a.visit_id
        AND av.source_schema = a.source_schema
        AND DATEDIFF('hour', av.visit_time, a.admitted_at) BETWEEN 0 AND 72
    GROUP BY 1
),
totals AS (
    SELECT condition, COUNT(DISTINCT visit_id) AS total_count
    FROM all_visits_by_cond
    GROUP BY 1
)
SELECT
    e.condition,
    e.esc_count,
    t.total_count,
    ROUND(100.0 * e.esc_count / NULLIF(t.total_count, 0), 1) AS escalation_rate_pct
FROM escalations e
INNER JOIN totals t ON e.condition = t.condition
WHERE e.condition != 'Not recorded'
ORDER BY e.esc_count DESC
LIMIT 12
"""
    return run_query(sql)
# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PATIENT FLOW & RETENTION
# ══════════════════════════════════════════════════════════════════════════════

def load_retention_kpis(filters: dict, run_query) -> pd.DataFrame:
    """Tab 3 KPI row: chronic patients bucketed by last-visit gap from max_date.
    Active/Retained ≤90d, Lapsing 91-180d, LTFU >180d — same definition as load_lifecycle.
    Date floor: 1 year back from max_date per source_schema (dynamic).
    """
    wh = _w(filters)
    wsa = _wsa(filters)
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
    WHERE v.created_at >= DATEADD('year', -1, sa.max_date)
      {wh}
      AND (dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1
           OR n.diagnosis ILIKE '%hypertension%'
           OR n.diagnosis ILIKE '%diabetes%'
           OR n.diagnosis ILIKE '%hiv%')
),
last_visits AS (
    SELECT v.source_schema, v.patient, MAX(v.created_at) AS last_visit_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN chronic_patients cp
        ON v.patient = cp.patient AND v.source_schema = cp.source_schema
    WHERE v.created_at >= DATEADD('year', -1, sa.max_date)
    {wh}
    GROUP BY v.source_schema, v.patient
),
patient_status AS (
    SELECT lv.source_schema, lv.patient,
        DATEDIFF('day', lv.last_visit_date, sa.max_date) AS days_gap
    FROM last_visits lv
    INNER JOIN schema_anchor sa ON lv.source_schema = sa.source_schema
)
SELECT
    COUNT(DISTINCT patient)                                                        AS chronic_patients,
    COUNT(DISTINCT CASE WHEN days_gap <= 90               THEN patient END)        AS retained_patients,
    COUNT(DISTINCT CASE WHEN days_gap BETWEEN 91 AND 180  THEN patient END)        AS lapsing_patients,
    COUNT(DISTINCT CASE WHEN days_gap > 180               THEN patient END)        AS ltfu_patients,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN days_gap <= 90 THEN patient END),
        COUNT(DISTINCT patient)
    ) * 100, 1)                                                                    AS retention_rate_pct
FROM patient_status
"""
    return run_query(sql)


def load_acquisition_segments(filters: dict, run_query) -> pd.DataFrame:
    """Acquisition segment retention: new vs returning split by segment and age group.
    Segments: Chronic, Oncology, Maternal, Mental Health, Post-Op.
    Uses direct note-keyword detection (more permissive than UDF-based classification)
    following the same pattern as load_retention_kpis.
    """
    wh  = _w(filters)
    wsa = _wsa(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
first_visits AS (
    SELECT v.patient, v.source_schema, MIN(v.created_at) AS first_visit_at
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    GROUP BY v.patient, v.source_schema
),
note_flags AS (
    SELECT
        n.visit_id,
        n.source_schema,
        MAX(CASE WHEN n.diagnosis ILIKE '%hypertension%'
                   OR n.diagnosis ILIKE '%diabetes%'
                   OR n.diagnosis ILIKE '%hiv%'
                   OR n.diagnosis ILIKE '% tb %'
                   OR n.diagnosis ILIKE 'tb %'
                   OR n.diagnosis ILIKE '%tuberculosis%'
                   OR n.diagnosis ILIKE '%asthma%'
                   OR n.diagnosis ILIKE '%copd%'
                   OR n.diagnosis ILIKE '%epilep%'
                   OR n.diagnosis ILIKE '%renal%'
                   OR n.diagnosis ILIKE '%hypothyroid%'
                   OR n.diagnosis ILIKE '%sickle%'
                 THEN 1 ELSE 0 END)                       AS chronic_flag,
        MAX(CASE WHEN n.diagnosis ILIKE '%cancer%'
                   OR n.diagnosis ILIKE '%oncol%'
                   OR n.diagnosis ILIKE '%neoplasm%'
                   OR n.diagnosis ILIKE '%lymphoma%'
                   OR n.diagnosis ILIKE '%leukaemi%'
                   OR n.diagnosis ILIKE '%leukemi%'
                 THEN 1 ELSE 0 END)                       AS oncology_flag,
        MAX(CASE WHEN n.diagnosis ILIKE '%pregnant%'
                   OR n.diagnosis ILIKE '%antenatal%'
                   OR n.diagnosis ILIKE '%postnatal%'
                   OR n.diagnosis ILIKE '%delivery%'
                   OR n.diagnosis ILIKE '%obstet%'
                   OR n.diagnosis ILIKE '%maternal%'
                   OR n.diagnosis ILIKE '%neonatal%'
                 THEN 1 ELSE 0 END)                       AS maternal_flag,
        MAX(CASE WHEN n.diagnosis ILIKE '%mental%'
                   OR n.diagnosis ILIKE '%psychiatr%'
                   OR n.diagnosis ILIKE '%depression%'
                   OR n.diagnosis ILIKE '%anxiety%'
                   OR n.diagnosis ILIKE '%psychosis%'
                   OR n.diagnosis ILIKE '%substance%'
                 THEN 1 ELSE 0 END)                       AS mental_flag,
        MAX(CASE WHEN n.diagnosis ILIKE '%post-op%'
                   OR n.diagnosis ILIKE '%post op%'
                   OR n.diagnosis ILIKE '%post-operative%'
                   OR n.diagnosis ILIKE '%post operative%'
                   OR n.diagnosis ILIKE '%postop%'
                   OR n.diagnosis ILIKE '%post surgery%'
                   OR n.diagnosis ILIKE '%post surgical%'
                   OR n.diagnosis ILIKE '%wound review%'
                   OR n.diagnosis ILIKE '%suture removal%'
                   OR n.diagnosis ILIKE '%stitch removal%'
                   OR n.diagnosis ILIKE '%follow up after%'
                   OR n.diagnosis ILIKE '%review after surgery%'
                 THEN 1 ELSE 0 END)                       AS postop_flag
    FROM HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
    GROUP BY n.visit_id, n.source_schema
),
visit_segments AS (
    SELECT
        v.source_schema,
        v.id                                              AS visit_id,
        v.patient                                         AS patient_id,
        CASE WHEN v.created_at = fv.first_visit_at
             THEN 'New' ELSE 'Returning'
        END                                               AS patient_type,
        CASE
            WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 5  THEN 'Toddler (0-4)'
            WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 13 THEN 'Child (5-12)'
            WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 18 THEN 'Adolescent (13-17)'
            WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 25 THEN 'Youth (18-24)'
            WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 35 THEN 'Young Adult (25-34)'
            WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 45 THEN 'Adult (35-44)'
            WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 55 THEN 'Middle Age (45-54)'
            WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 65 THEN 'Older Adult (55-64)'
            ELSE                                               'Senior (65+)'
        END                                               AS age_group,
        CASE
            WHEN COALESCE(nf.oncology_flag, 0) = 1
              OR COALESCE(dx.disease_burden_group_1, '') ILIKE '%oncol%'
              OR COALESCE(dx.disease_burden_group_1, '') ILIKE '%cancer%'
              OR COALESCE(dx.disease_burden_group_2, '') ILIKE '%oncol%'
              OR COALESCE(dx.disease_burden_group_2, '') ILIKE '%cancer%'
              OR COALESCE(dx.disease_burden_group_3, '') ILIKE '%oncol%'
              OR COALESCE(dx.disease_burden_group_3, '') ILIKE '%cancer%'
                THEN 'Oncology'
            WHEN COALESCE(nf.maternal_flag, 0) = 1
              OR COALESCE(dx.disease_burden_group_1, '') ILIKE '%maternal%'
              OR COALESCE(dx.disease_burden_group_1, '') ILIKE '%obstet%'
              OR COALESCE(dx.disease_burden_group_2, '') ILIKE '%maternal%'
              OR COALESCE(dx.disease_burden_group_2, '') ILIKE '%obstet%'
              OR COALESCE(dx.disease_burden_group_3, '') ILIKE '%maternal%'
              OR COALESCE(dx.disease_burden_group_3, '') ILIKE '%obstet%'
                THEN 'Maternal'
            WHEN COALESCE(nf.mental_flag, 0) = 1
              OR COALESCE(dx.disease_burden_group_1, '') ILIKE '%mental%'
              OR COALESCE(dx.disease_burden_group_1, '') ILIKE '%psychiatr%'
              OR COALESCE(dx.disease_burden_group_2, '') ILIKE '%mental%'
              OR COALESCE(dx.disease_burden_group_2, '') ILIKE '%psychiatr%'
              OR COALESCE(dx.disease_burden_group_3, '') ILIKE '%mental%'
              OR COALESCE(dx.disease_burden_group_3, '') ILIKE '%psychiatr%'
                THEN 'Mental Health'
            WHEN COALESCE(nf.postop_flag, 0) = 1
              OR COALESCE(dx.disease_burden_group_1, '') ILIKE '%post-op%'
              OR COALESCE(dx.disease_burden_group_1, '') ILIKE '%post op%'
              OR COALESCE(dx.disease_burden_group_1, '') ILIKE '%surgical%'
              OR COALESCE(dx.disease_burden_group_2, '') ILIKE '%post-op%'
              OR COALESCE(dx.disease_burden_group_2, '') ILIKE '%post op%'
              OR COALESCE(dx.disease_burden_group_2, '') ILIKE '%surgical%'
              OR COALESCE(dx.disease_burden_group_3, '') ILIKE '%post-op%'
              OR COALESCE(dx.disease_burden_group_3, '') ILIKE '%post op%'
              OR COALESCE(dx.disease_burden_group_3, '') ILIKE '%surgical%'
                THEN 'Post-Op'
            WHEN COALESCE(dx.is_chronic_1, 0) = 1
              OR COALESCE(dx.is_chronic_2, 0) = 1
              OR COALESCE(nf.chronic_flag, 0) = 1
                THEN 'Chronic'
            ELSE NULL
        END                                               AS acquisition_segment
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa  ON v.source_schema = sa.source_schema
    LEFT JOIN first_visits fv    ON v.patient = fv.patient AND v.source_schema = fv.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS rp
        ON v.patient = rp.patient_id AND v.source_schema = rp.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN note_flags nf      ON v.id = nf.visit_id AND v.source_schema = nf.source_schema
    WHERE v.created_at >= '2024-09-01'
    {wh}
),
agg AS (
    SELECT
        source_schema,
        acquisition_segment,
        age_group,
        COUNT(visit_id)                                                          AS total_visits,
        COUNT(DISTINCT CASE WHEN patient_type = 'New'       THEN patient_id END) AS new_patients,
        COUNT(DISTINCT CASE WHEN patient_type = 'Returning' THEN patient_id END) AS returning_patients
    FROM visit_segments
    WHERE acquisition_segment IS NOT NULL
    GROUP BY source_schema, acquisition_segment, age_group
)
SELECT
    source_schema,
    acquisition_segment,
    age_group,
    1                                                                             AS is_retention_universe,
    total_visits,
    new_patients,
    returning_patients,
    ROUND(new_patients       * 100.0 / NULLIF(new_patients + returning_patients, 0), 1) AS new_pct,
    ROUND(returning_patients * 100.0 / NULLIF(new_patients + returning_patients, 0), 1) AS returning_pct,
    ROUND(DIV0(returning_patients, NULLIF(new_patients, 0)), 2)                  AS returning_per_new_ratio,
    CASE
        WHEN acquisition_segment IN ('Chronic', 'Oncology', 'Mental Health') THEN '>1.0'
        WHEN acquisition_segment = 'Maternal'                               THEN '>=1.0'
        WHEN acquisition_segment = 'Post-Op'                                THEN '<1.0'
    END                                                                          AS expected_ratio_direction,
    CASE
        WHEN acquisition_segment = 'Post-Op'                                          THEN 'REVIEW_DATA'
        WHEN acquisition_segment IN ('Chronic', 'Oncology', 'Mental Health')
             AND DIV0(returning_patients, NULLIF(new_patients, 0)) >= 1.0             THEN 'AS_EXPECTED'
        WHEN acquisition_segment = 'Maternal'
             AND DIV0(returning_patients, NULLIF(new_patients, 0)) >= 1.0             THEN 'AS_EXPECTED'
        ELSE                                                                               'CONCERN'
    END                                                                          AS ratio_signal,
    ROUND(DIV0(returning_patients, NULLIF(new_patients, 0)) - 1.0, 2)           AS divergence_from_threshold
FROM agg
ORDER BY acquisition_segment, age_group
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
    WHERE v.created_at >= DATEADD('year', -1, sa.max_date)
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    """B: Compare LTFU vs retained chronic patients by age group, sex, and payer.
    LTFU = last visit >180d from max_date. Chronic: ICD10 + doctor notes keywords.
    Floor: 1 year from max_date (dynamic).
    """
    wh = _w(filters)
    wsa = _wsa(filters)
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
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON v.id = n.visit_id AND v.source_schema = n.source_schema
    WHERE v.created_at >= DATEADD('year', -1, sa.max_date)
      {wh}
      AND (dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1
           OR n.diagnosis ILIKE '%hypertension%'
           OR n.diagnosis ILIKE '%diabetes%'
           OR n.diagnosis ILIKE '%hiv%')
    GROUP BY 1, 2
),
patient_status AS (
    SELECT cp.source_schema, cp.patient, cp.total_visits, cp.payer,
        CASE WHEN DATEDIFF('day', cp.last_visit, sa.max_date) > 180
             THEN 'LTFU' ELSE 'Retained' END AS retention_status
    FROM chronic_patients cp
    INNER JOIN schema_anchor sa ON cp.source_schema = sa.source_schema
),
patient_profile AS (
    SELECT
        ps.patient, ps.retention_status, ps.payer,
        CASE
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 18 THEN 'Under 18'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 25 THEN 'Youth (18–24)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 35 THEN 'Young Adult (25–34)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 45 THEN 'Adult (35–44)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 55 THEN 'Middle Age (45–54)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 65 THEN 'Older Adult (55–64)'
            ELSE '65+'
        END AS age_group,
        CASE
            WHEN UPPER(COALESCE(rp.sex, '')) IN ('F', 'FEMALE') THEN 'Female'
            WHEN UPPER(COALESCE(rp.sex, '')) IN ('M', 'MALE')   THEN 'Male'
            ELSE NULL
        END AS sex
    FROM patient_status ps
    INNER JOIN schema_anchor sa ON ps.source_schema = sa.source_schema
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
    SELECT 'Gender' AS factor, sex AS dimension,
        COUNT(DISTINCT CASE WHEN retention_status = 'Retained' THEN patient END),
        COUNT(DISTINCT CASE WHEN retention_status = 'LTFU'     THEN patient END),
        COUNT(DISTINCT patient),
        ROUND(DIV0(
            COUNT(DISTINCT CASE WHEN retention_status = 'LTFU' THEN patient END),
            COUNT(DISTINCT patient)
        ) * 100, 1)
    FROM patient_profile
    WHERE sex IS NOT NULL
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
    """Q7: Revenue at risk from chronic LTFU/lapsing patients.
    LTFU >180d, Lapsing 91-180d — aligned with load_lifecycle thresholds.
    Chronic detection: ICD10 + doctor notes keywords. Floor: 1 year from max_date.
    """
    wh = _w(filters)
    wsa = _wsa(filters)
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
    WHERE v.created_at >= DATEADD('year', -1, sa.max_date)
    {wh}
    GROUP BY 1
),
chronic_last_visit AS (
    SELECT v.source_schema, v.patient, MAX(v.created_at) AS last_v
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON v.id = n.visit_id AND v.source_schema = n.source_schema
    WHERE v.created_at >= DATEADD('year', -1, sa.max_date)
      {wh}
      AND (dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1
           OR n.diagnosis ILIKE '%hypertension%'
           OR n.diagnosis ILIKE '%diabetes%'
           OR n.diagnosis ILIKE '%hiv%')
    GROUP BY v.source_schema, v.patient
),
ltfu_counts AS (
    SELECT clv.source_schema,
        COUNT(DISTINCT CASE
            WHEN DATEDIFF('day', clv.last_v, sa.max_date) > 180
            THEN clv.patient END)                           AS chronic_ltfu,
        COUNT(DISTINCT CASE
            WHEN DATEDIFF('day', clv.last_v, sa.max_date) BETWEEN 91 AND 180
            THEN clv.patient END)                           AS chronic_lapsing
    FROM chronic_last_visit clv
    INNER JOIN schema_anchor sa ON clv.source_schema = sa.source_schema
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


def load_lapsing_by_payer(filters: dict, run_query) -> pd.DataFrame:
    """Lapsing chronic patients (91–180 days) broken out by payer.
    Uses identical population logic as load_lifecycle so the payer bars
    always sum to the same total shown in the lifecycle KPI strip."""
    wh  = _w(filters)
    wsa = _wsa(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
patient_status AS (
    SELECT
        v.source_schema,
        v.patient,
        MAX(v.created_at)  AS last_visit,
        sa.max_date,
        CASE
            WHEN UPPER(v.payment_mode) IN ('CASH','PRIVATE')              THEN 'Cash'
            WHEN UPPER(v.payment_mode) LIKE '%NHIF%'
              OR UPPER(v.payment_mode) LIKE '%SHA%'                       THEN 'NHIF / SHA'
            ELSE 'Insurance / Corporate'
        END                AS payer,
        MAX(CASE
            WHEN dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1
              OR n.diagnosis ILIKE '%hypertension%'
              OR n.diagnosis ILIKE '%diabetes%'
              OR n.diagnosis ILIKE '%hiv%'
            THEN 1 ELSE 0
        END)               AS is_chronic
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON v.id = n.visit_id AND v.source_schema = n.source_schema
    WHERE v.created_at >= DATEADD('year', -1, sa.max_date)
    {wh}
    GROUP BY v.source_schema, v.patient, v.payment_mode, sa.max_date
)
SELECT
    payer,
    COUNT(DISTINCT patient) AS patient_count
FROM patient_status
WHERE is_chronic = 1
  AND DATEDIFF('day', last_visit, max_date) BETWEEN 91 AND 180
GROUP BY payer
ORDER BY patient_count DESC
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
      AND v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
      AND v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
      AND v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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


def load_ltfu_priority2_patients(filters: dict, run_query) -> pd.DataFrame:
    """R10: One row per chronic LTFU patient — all profile dimensions for the Priority 2 section.

    Returns: PATIENT_ID, VISIT_COUNT, IS_CHRONIC, LAST_VISIT_DATE, DAYS_SINCE_LAST_VISIT,
             ACQUISITION_SEGMENT, CLEAN_DIAGNOSIS, AGE_GROUP, GENDER, REVENUE_AT_RISK
    Filters:  DAYS_SINCE_LAST_VISIT >= 180  AND at least one chronic ICD10/note keyword.
    """
    wh  = _w(filters)
    wsa = _wsa(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
patient_visits AS (
    SELECT
        v.source_schema,
        v.patient                                          AS patient_id,
        COUNT(DISTINCT v.id)                               AS visit_count,
        MAX(v.created_at)                                  AS last_visit_date,
        DATEDIFF('day', MAX(v.created_at), sa.max_date)   AS days_since_last_visit
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= DATEADD('year', -1, sa.max_date)
    {wh}
    GROUP BY v.source_schema, v.patient, sa.max_date
),
chronic_patients AS (
    SELECT DISTINCT v.source_schema, v.patient AS patient_id
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON v.id = n.visit_id AND v.source_schema = n.source_schema
    WHERE v.created_at >= DATEADD('year', -1, sa.max_date)
      AND (COALESCE(dx.is_chronic_1, 0) = 1
        OR COALESCE(dx.is_chronic_2, 0) = 1
        OR n.diagnosis ILIKE '%hypertension%'
        OR n.diagnosis ILIKE '%diabetes%'
        OR n.diagnosis ILIKE '%hiv%')
    {wh}
),
patient_revenue AS (
    SELECT
        v.source_schema,
        v.patient                                          AS patient_id,
        ROUND(SUM(ili.item_amount), 0)                    AS revenue_at_risk
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_INVOICE_LINE_ITEMS ili
        ON v.id = ili.visit_id AND v.source_schema = ili.source_schema
    WHERE v.created_at >= DATEADD('year', -1, sa.max_date)
      AND ili.invoice_deleted_at IS NULL
      AND (ili.auto_cancelled IS NULL OR ili.auto_cancelled = 0)
    {wh}
    GROUP BY v.source_schema, v.patient
),
latest_diagnosis AS (
    SELECT
        v.source_schema,
        v.patient                                          AS patient_id,
        COALESCE(
            NULLIF(TRIM(dx.disease_burden_group_1), ''),
            CASE
                WHEN n.diagnosis ILIKE '%hypertension%'         THEN 'Hypertension'
                WHEN n.diagnosis ILIKE '%diabetes%'             THEN 'Diabetes'
                WHEN n.diagnosis ILIKE '%hiv%'                  THEN 'HIV/AIDS'
                WHEN n.diagnosis ILIKE '%cancer%'
                  OR n.diagnosis ILIKE '%oncol%'                THEN 'Oncology'
                WHEN n.diagnosis ILIKE '%chronic kidney%'
                  OR n.diagnosis ILIKE '%ckd%'                  THEN 'CKD'
                ELSE 'Other'
            END
        )                                                  AS clean_diagnosis
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON v.id = n.visit_id AND v.source_schema = n.source_schema
    WHERE v.created_at >= DATEADD('year', -1, sa.max_date)
    {wh}
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY v.source_schema, v.patient
        ORDER BY v.created_at DESC
    ) = 1
),
latest_demographics AS (
    SELECT
        v.source_schema,
        v.patient                                          AS patient_id,
        CASE
            WHEN rp.dob IS NULL                                          THEN 'Unknown'
            WHEN TIMESTAMPDIFF('year', rp.dob, v.created_at) < 18       THEN 'Under 18'
            WHEN TIMESTAMPDIFF('year', rp.dob, v.created_at) < 25       THEN 'Youth (18–24)'
            WHEN TIMESTAMPDIFF('year', rp.dob, v.created_at) < 35       THEN 'Young Adult (25–34)'
            WHEN TIMESTAMPDIFF('year', rp.dob, v.created_at) < 45       THEN 'Adult (35–44)'
            WHEN TIMESTAMPDIFF('year', rp.dob, v.created_at) < 55       THEN 'Middle Age (45–54)'
            WHEN TIMESTAMPDIFF('year', rp.dob, v.created_at) < 65       THEN 'Older Adult (55–64)'
            ELSE '65+'
        END                                                AS age_group,
        COALESCE(NULLIF(TRIM(rp.sex), ''), 'Unknown')     AS gender
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS rp
        ON v.patient = rp.patient_id AND v.source_schema = rp.source_schema
    WHERE v.created_at >= DATEADD('year', -1, sa.max_date)
    {wh}
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY v.source_schema, v.patient
        ORDER BY v.created_at DESC
    ) = 1
)
SELECT
    pv.patient_id,
    pv.visit_count,
    TRUE                                                   AS is_chronic,
    pv.last_visit_date,
    pv.days_since_last_visit,
    'CHRONIC'                                              AS acquisition_segment,
    COALESCE(ld.clean_diagnosis, 'Other')                  AS clean_diagnosis,
    COALESCE(dem.age_group,      'Unknown')                AS age_group,
    COALESCE(dem.gender,         'Unknown')                AS gender,
    COALESCE(pr.revenue_at_risk, 0)                        AS revenue_at_risk
FROM patient_visits pv
INNER JOIN chronic_patients cp
    ON pv.patient_id = cp.patient_id AND pv.source_schema = cp.source_schema
LEFT JOIN latest_diagnosis ld
    ON pv.patient_id = ld.patient_id AND pv.source_schema = ld.source_schema
LEFT JOIN latest_demographics dem
    ON pv.patient_id = dem.patient_id AND pv.source_schema = dem.source_schema
LEFT JOIN patient_revenue pr
    ON pv.patient_id = pr.patient_id AND pv.source_schema = pr.source_schema
WHERE pv.days_since_last_visit > 180
ORDER BY pv.days_since_last_visit DESC
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
    WHERE v.created_at >=  '2024-09-01'
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
    CASE
        WHEN dx.disease_burden_group_1 ILIKE '%Endocrin%'
          OR dx.disease_burden_group_1 ILIKE '%Diabetes%' THEN 'NCD — Diabetes & Metabolic'
        ELSE COALESCE(dx.disease_burden_group_1, 'Unclassified')
    END                                                 AS burden_group,
    COUNT(DISTINCT v.id)                                AS visit_count
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
    ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
WHERE v.created_at >=  '2024-09-01'
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
WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
WHERE age_group != 'Unknown' AND  sex NOT IN ('UNKNOWN', 'Unknown') AND sex != ''
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
        WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
WHERE v.created_at >=  '2024-09-01'
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
WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
# ══════════════════════════════════════════════════════════════════════════════
# PATIENT CONVERSION & VALUE TAB
# ══════════════════════════════════════════════════════════════════════════════

def _visit_base_cte(filters: dict, start_date: str = None) -> str:
    """
    Returns the visit_base CTE block (no leading WITH) for composing queries.

    Includes doctor_note_groups CTE that parses the diagnosis JSON array and
    calls MAP_DIAGNOSIS_TO_BURDEN_GROUP() UDF for fallback mapping when ICD-10
    structured labels are absent or generic ('Other%').

    start_date: optional ISO date string (e.g. '2024-09-01') used as a hard
    floor on v.created_at, overriding the rolling mo window.
    """
    wh  = _w(filters)
    wsa = _wsa(filters)
    mo  = _mo(filters)
    date_floor = ("'" + start_date + "'") if start_date else ("DATEADD('month', -" + str(mo) + ", sa.max_date)")
    return f"""schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
doctor_note_groups AS (
    SELECT
        visit_id,
        source_schema,
        TRIM(REPLACE(
            SPLIT_PART(REGEXP_REPLACE(diagnosis, '^\\\\["?|"?\\\\]$', ''), '", "', 1),
        '"', ''))                                               AS note_condition_1,
        NULLIF(TRIM(REPLACE(
            SPLIT_PART(REGEXP_REPLACE(diagnosis, '^\\\\["?|"?\\\\]$', ''), '", "', 2),
        '"', '')), '')                                          AS note_condition_2,
        NULLIF(TRIM(REPLACE(
            SPLIT_PART(REGEXP_REPLACE(diagnosis, '^\\\\["?|"?\\\\]$', ''), '", "', 3),
        '"', '')), '')                                          AS note_condition_3
    FROM (
        SELECT visit_id, source_schema, diagnosis,
               ROW_NUMBER() OVER (
                   PARTITION BY visit_id, source_schema
                   ORDER BY created_at ASC NULLS LAST)          AS rn
        FROM HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES
    ) dn
    WHERE rn = 1
),
chronic_note_visits AS (
    SELECT DISTINCT visit_id, source_schema
    FROM HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES
    WHERE diagnosis ILIKE '%hypertension%'
       OR diagnosis ILIKE '%diabetes%'
       OR diagnosis ILIKE '%hiv%'
),
visit_base AS (
    SELECT
        v.source_schema,
        v.id                                                    AS visit_id,
        v.patient                                               AS patient_id,
        DATE(v.created_at)                                      AS visit_date,
        DATE_TRUNC('month', v.created_at)                       AS visit_month,
        CASE WHEN a.visit_id IS NOT NULL THEN 'Inpatient'
             ELSE 'Outpatient' END                              AS visit_type,
        CASE
            WHEN LOWER(v.payment_mode) IN ('nhif','shif','sha','national scheme')
                THEN 'NHIF / SHA'
            WHEN LOWER(v.payment_mode) IN ('cash','self-pay','out-of-pocket','copay')
                THEN 'Cash'
            WHEN v.payment_mode IS NULL OR TRIM(v.payment_mode) = ''
                THEN 'Unknown'
            ELSE 'Insurance'
        END                                                     AS payer_type,
        CASE
            WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 5  THEN 'Toddler (0-4)'
            WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 13 THEN 'Child (5-12)'
            WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 18 THEN 'Adolescent (13-17)'
            WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 25 THEN 'Youth (18-24)'
            WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 35 THEN 'Young Adult (25-34)'
            WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 45 THEN 'Adult (35-44)'
            WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 55 THEN 'Middle Age (45-54)'
            WHEN DATEDIFF('year', rp.dob, CURRENT_DATE) < 65 THEN 'Older Adult (55-64)'
            ELSE 'Senior (65+)'
        END                                                     AS age_group,
        COALESCE(rp.sex, 'Unknown')                             AS gender,
        CASE
            WHEN d.disease_burden_group_1 IS NOT NULL
             AND d.disease_burden_group_1 NOT ILIKE 'Other%'
             AND d.disease_burden_group_1 NOT ILIKE 'Communicable - Other%'
            THEN d.disease_burden_group_1
            WHEN d.disease_burden_group_2 IS NOT NULL
             AND d.disease_burden_group_2 NOT ILIKE 'Other%'
             AND d.disease_burden_group_2 NOT ILIKE 'Communicable - Other%'
            THEN d.disease_burden_group_2
            WHEN d.disease_burden_group_3 IS NOT NULL
             AND d.disease_burden_group_3 NOT ILIKE 'Other%'
             AND d.disease_burden_group_3 NOT ILIKE 'Communicable - Other%'
            THEN d.disease_burden_group_3
            WHEN HOSPITALS.STAGING.MAP_DIAGNOSIS_TO_BURDEN_GROUP(dn.note_condition_1) IS NOT NULL
            THEN HOSPITALS.STAGING.MAP_DIAGNOSIS_TO_BURDEN_GROUP(dn.note_condition_1)
            WHEN HOSPITALS.STAGING.MAP_DIAGNOSIS_TO_BURDEN_GROUP(dn.note_condition_2) IS NOT NULL
            THEN HOSPITALS.STAGING.MAP_DIAGNOSIS_TO_BURDEN_GROUP(dn.note_condition_2)
            WHEN HOSPITALS.STAGING.MAP_DIAGNOSIS_TO_BURDEN_GROUP(dn.note_condition_3) IS NOT NULL
            THEN HOSPITALS.STAGING.MAP_DIAGNOSIS_TO_BURDEN_GROUP(dn.note_condition_3)
            ELSE 'Unclassified'
        END                                                     AS disease_burden_group,
        dn.note_condition_1                                     AS raw_note_condition_1,
        dn.note_condition_2                                     AS raw_note_condition_2,
        dn.note_condition_3                                     AS raw_note_condition_3,
        CASE
            WHEN d.disease_burden_group_1 IS NOT NULL
             AND d.disease_burden_group_1 NOT ILIKE 'Other%'
             AND d.disease_burden_group_1 NOT ILIKE 'Communicable - Other%'
            THEN 'ICD-10'
            WHEN d.disease_burden_group_2 IS NOT NULL
             AND d.disease_burden_group_2 NOT ILIKE 'Other%'
             AND d.disease_burden_group_2 NOT ILIKE 'Communicable - Other%'
            THEN 'ICD-10'
            WHEN d.disease_burden_group_3 IS NOT NULL
             AND d.disease_burden_group_3 NOT ILIKE 'Other%'
             AND d.disease_burden_group_3 NOT ILIKE 'Communicable - Other%'
            THEN 'ICD-10'
            WHEN HOSPITALS.STAGING.MAP_DIAGNOSIS_TO_BURDEN_GROUP(dn.note_condition_1) IS NOT NULL
            THEN 'Doctor Note - Mapped'
            WHEN HOSPITALS.STAGING.MAP_DIAGNOSIS_TO_BURDEN_GROUP(dn.note_condition_2) IS NOT NULL
            THEN 'Doctor Note - Mapped'
            WHEN HOSPITALS.STAGING.MAP_DIAGNOSIS_TO_BURDEN_GROUP(dn.note_condition_3) IS NOT NULL
            THEN 'Doctor Note - Mapped'
            WHEN dn.note_condition_1 IS NOT NULL
            THEN 'Unclassified - Note Exists, No ICD-10'
            WHEN d.visit_id IS NOT NULL
             AND COALESCE(d.disease_burden_group_1,
                          d.disease_burden_group_2,
                          d.disease_burden_group_3) IS NULL
            THEN 'Unclassified - ICD-10 Record Exists But No Group'
            WHEN d.visit_id IS NULL AND dn.visit_id IS NULL
            THEN 'Unclassified - No ICD-10 And No Doctor Note'
            WHEN d.visit_id IS NULL AND dn.note_condition_1 IS NULL
            THEN 'Unclassified - No ICD-10, Empty Doctor Note'
            ELSE 'Unclassified - Other'
        END                                                     AS classification_source,
        CASE
            WHEN COALESCE(d.is_chronic_1, 0) = 1
              OR COALESCE(d.is_chronic_2, 0) = 1
              OR cnv.visit_id IS NOT NULL
            THEN 1 ELSE 0
        END                                                     AS is_chronic,
        COALESCE(inv.total_revenue, 0)                          AS visit_revenue,
        CASE
            WHEN v.created_at = MIN(v.created_at) OVER (
                PARTITION BY v.patient, v.source_schema)
            THEN 'New' ELSE 'Returning'
        END                                                     AS patient_type,
        COUNT(v.id) OVER (
            PARTITION BY v.patient, v.source_schema)            AS total_visit_count
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa
        ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
        ON v.id = a.visit_id
       AND v.source_schema = REPLACE(LOWER(a.source_schema), '_clean', '')
    LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS rp
        ON v.patient = rp.patient_id AND v.source_schema = rp.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED d
        ON v.id = d.visit_id AND v.source_schema = d.source_schema
    LEFT JOIN doctor_note_groups dn
        ON v.id = dn.visit_id AND v.source_schema = dn.source_schema
    LEFT JOIN chronic_note_visits cnv
        ON v.id = cnv.visit_id AND v.source_schema = cnv.source_schema
    LEFT JOIN (
        SELECT visit_id, source_schema, SUM(item_amount) AS total_revenue
        FROM HOSPITALS.STAGING.STG_INVOICE_LINE_ITEMS
        WHERE invoice_deleted_at IS NULL
          AND (auto_cancelled IS NULL OR auto_cancelled = 0)
        GROUP BY visit_id, source_schema
    ) inv ON v.id = inv.visit_id AND v.source_schema = inv.source_schema
    WHERE v.created_at >= {date_floor}
    {wh}
)"""


def load_cv_overview(filters: dict, run_query) -> pd.DataFrame:
    """Section 1 — 6 KPI cards: total patients, chronic, repeat, single-visit,
    avg visits/patient, new vs returning split."""
    wsa = _wsa(filters)
    wh  = _w(filters)
    vb  = _visit_base_cte(filters)
    sql = f"""
WITH {vb},
chronic_base AS (
    SELECT DISTINCT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN (
        SELECT source_schema, MAX(created_at) AS max_date
        FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
        GROUP BY source_schema
    ) sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON v.id = n.visit_id AND v.source_schema = n.source_schema
    WHERE v.created_at >= '2024-09-01'
    {wh}
    AND (
        COALESCE(dx.is_chronic_1, 0) = 1
        OR COALESCE(dx.is_chronic_2, 0) = 1
        OR n.diagnosis ILIKE '%hypertension%'
        OR n.diagnosis ILIKE '%diabetes%'
        OR n.diagnosis ILIKE '%hiv%'
    )
)
SELECT
    COUNT(DISTINCT vb.patient_id)                                        AS total_patients,
    (SELECT COUNT(DISTINCT patient) FROM chronic_base)                   AS chronic_patients,
    COUNT(DISTINCT CASE WHEN vb.total_visit_count >= 2 THEN vb.patient_id END) AS repeat_patients,
    COUNT(DISTINCT CASE WHEN vb.total_visit_count = 1  THEN vb.patient_id END) AS single_visit_patients,
    ROUND(COUNT(vb.visit_id) * 1.0 /
          NULLIF(COUNT(DISTINCT vb.patient_id), 0), 1)                   AS avg_visits_per_patient,
    COUNT(DISTINCT CASE WHEN vb.patient_type = 'New'       THEN vb.patient_id END) AS new_patients,
    COUNT(DISTINCT CASE WHEN vb.patient_type = 'Returning' THEN vb.patient_id END) AS returning_patients
FROM visit_base vb
"""
    return run_query(sql)


def load_cv_demographics(filters: dict, run_query) -> pd.DataFrame:
    """Section 2 Row 1 — gender, age group, chronic flag, payer type counts."""
    vb = _visit_base_cte(filters)
    sql = f"""
WITH {vb}
SELECT
    age_group,
    gender,
    payer_type,
    is_chronic,
    COUNT(DISTINCT patient_id) AS patients,
    SUM(visit_revenue)         AS revenue
FROM visit_base
WHERE age_group != 'Unknown'
  AND UPPER(COALESCE(gender,'')) IN ('F','FEMALE','M','MALE')
GROUP BY age_group, gender, payer_type, is_chronic
ORDER BY age_group, gender, payer_type
"""
    return run_query(sql)


def load_cv_cohort_growth(filters: dict, run_query) -> pd.DataFrame:
    """Section 2 Row 2 — monthly patient count by age group for growth index and mix."""
    vb = _visit_base_cte(filters, start_date='2024-09-01')
    sql = f"""
WITH {vb}
SELECT
    visit_month,
    age_group,
    COUNT(DISTINCT patient_id) AS patients
FROM visit_base
WHERE age_group != 'Unknown'
GROUP BY visit_month, age_group
ORDER BY visit_month, age_group
"""
    return run_query(sql)


def load_cv_chronic_growth(filters: dict, run_query) -> pd.DataFrame:
    """Monthly chronic patient count by age group and gender."""
    vb = _visit_base_cte(filters, start_date='2024-09-01')
    sql = f"""
WITH {vb}
SELECT
    visit_month,
    age_group,
    CASE
        WHEN UPPER(gender) IN ('F', 'FEMALE') THEN 'Female'
        WHEN UPPER(gender) IN ('M', 'MALE')   THEN 'Male'
        ELSE NULL
    END AS gender,
    COUNT(DISTINCT patient_id) AS chronic_patients
FROM visit_base
WHERE is_chronic = 1
  AND age_group != 'Unknown'
  AND UPPER(COALESCE(gender,'')) IN ('F','FEMALE','M','MALE')
GROUP BY visit_month, age_group, 3
ORDER BY visit_month, age_group
"""
    return run_query(sql)


def load_patient_profile(filters: dict, run_query) -> pd.DataFrame:
    """Patient profile dashboard — one row per deduplicated visit with segment,
    gender, age, patient_type (new/returning) and visit_type (in/outpatient)."""
    wh  = _w(filters)
    schemas = filters.get("source_schemas") or ["kisumu"]
    schema  = schemas[0] if schemas else "kisumu"
    sql = f"""
WITH diag_clean AS (
    SELECT
        i.source_schema,
        i.visit_id,
        HOSPITALS.STAGING.MAP_DIAGNOSIS_TO_BURDEN_GROUP(dn.diagnosis) AS notes_disease_burden_group,
        CASE
            WHEN (i.disease_burden_group_1 LIKE '%Other%' OR i.disease_burden_group_1 = 'Other')
                AND notes_disease_burden_group IS NOT NULL
                THEN IFF(notes_disease_burden_group = 'Communicable - Other Infectious','Communicable - Sepsis',notes_disease_burden_group)
            ELSE IFF(i.disease_burden_group_1 = 'Communicable - Other Infectious','Communicable - Sepsis',i.disease_burden_group_1)
        END AS clean_dbg_1,
        CASE
            WHEN (i.disease_burden_group_2 LIKE '%Other%' OR i.disease_burden_group_2 = 'Other')
                AND notes_disease_burden_group IS NOT NULL
                THEN IFF(notes_disease_burden_group = 'Communicable - Other Infectious','Communicable - Sepsis',notes_disease_burden_group)
            ELSE IFF(i.disease_burden_group_2 = 'Communicable - Other Infectious','Communicable - Sepsis',i.disease_burden_group_2)
        END AS clean_dbg_2,
        CASE
            WHEN (i.disease_burden_group_3 LIKE '%Other%' OR i.disease_burden_group_3 = 'Other')
                AND notes_disease_burden_group IS NOT NULL
                THEN IFF(notes_disease_burden_group = 'Communicable - Other Infectious','Communicable - Sepsis',notes_disease_burden_group)
            ELSE IFF(i.disease_burden_group_3 = 'Communicable - Other Infectious','Communicable - Sepsis',i.disease_burden_group_3)
        END AS clean_dbg_3,
        ARRAY_TO_STRING(ARRAY_SORT(ARRAY_DISTINCT(ARRAY_CONSTRUCT_COMPACT(
            clean_dbg_1, clean_dbg_2, clean_dbg_3))),' & ') AS clean_disease_burden_group,
        IFF(i.icd10_code_1 IS NOT NULL AND (i.icd10_code_2 IS NOT NULL OR i.icd10_code_3 IS NOT NULL),TRUE,FALSE) AS is_comorbidity,
        IFF(i.has_chronic_diagnosis >= 1,TRUE,FALSE) AS has_chronic_diagnosis,
        ARRAY_TO_STRING(ARRAY_SORT(ARRAY_DISTINCT(ARRAY_CONSTRUCT_COMPACT(
            CASE
                WHEN clean_dbg_1 = 'Digestive: Other'           THEN 'Digestive: Other'
                WHEN clean_dbg_1 = 'MNCH - Maternal: Other'     THEN 'Maternal: Other'
                WHEN clean_dbg_1 = 'NCD - Haematology: Other'   THEN 'Haematology: Other'
                WHEN clean_dbg_1 = 'NCD - Mental Health: Other' THEN 'Mental Health: Other'
                WHEN clean_dbg_1 = 'NCD - Haematology: Anaemia' THEN 'Haematology: Anaemia'
                WHEN clean_dbg_1 = 'Symptom - Anaemia'          THEN 'Symptom: Anaemia'
                WHEN clean_dbg_1 = 'NCD - Musculoskeletal'      THEN 'NCD - Musculoskeletal'
                WHEN clean_dbg_1 = 'Musculoskeletal'            THEN 'Musculoskeletal'
                WHEN clean_dbg_1 LIKE '%:%'   THEN TRIM(SPLIT_PART(clean_dbg_1,':',2))
                WHEN clean_dbg_1 LIKE '% - %' THEN TRIM(SPLIT_PART(clean_dbg_1,' - ',2))
                ELSE TRIM(clean_dbg_1)
            END,
            CASE
                WHEN clean_dbg_2 = 'Digestive: Other'           THEN 'Digestive: Other'
                WHEN clean_dbg_2 = 'MNCH - Maternal: Other'     THEN 'Maternal: Other'
                WHEN clean_dbg_2 = 'NCD - Haematology: Other'   THEN 'Haematology: Other'
                WHEN clean_dbg_2 = 'NCD - Mental Health: Other' THEN 'Mental Health: Other'
                WHEN clean_dbg_2 = 'NCD - Haematology: Anaemia' THEN 'Haematology: Anaemia'
                WHEN clean_dbg_2 = 'Symptom - Anaemia'          THEN 'Symptom: Anaemia'
                WHEN clean_dbg_2 = 'NCD - Musculoskeletal'      THEN 'NCD - Musculoskeletal'
                WHEN clean_dbg_2 = 'Musculoskeletal'            THEN 'Musculoskeletal'
                WHEN clean_dbg_2 LIKE '%:%'   THEN TRIM(SPLIT_PART(clean_dbg_2,':',2))
                WHEN clean_dbg_2 LIKE '% - %' THEN TRIM(SPLIT_PART(clean_dbg_2,' - ',2))
                ELSE TRIM(clean_dbg_2)
            END,
            CASE
                WHEN clean_dbg_3 = 'Digestive: Other'           THEN 'Digestive: Other'
                WHEN clean_dbg_3 = 'MNCH - Maternal: Other'     THEN 'Maternal: Other'
                WHEN clean_dbg_3 = 'NCD - Haematology: Other'   THEN 'Haematology: Other'
                WHEN clean_dbg_3 = 'NCD - Mental Health: Other' THEN 'Mental Health: Other'
                WHEN clean_dbg_3 = 'NCD - Haematology: Anaemia' THEN 'Haematology: Anaemia'
                WHEN clean_dbg_3 = 'Symptom - Anaemia'          THEN 'Symptom: Anaemia'
                WHEN clean_dbg_3 = 'NCD - Musculoskeletal'      THEN 'NCD - Musculoskeletal'
                WHEN clean_dbg_3 = 'Musculoskeletal'            THEN 'Musculoskeletal'
                WHEN clean_dbg_3 LIKE '%:%'   THEN TRIM(SPLIT_PART(clean_dbg_3,':',2))
                WHEN clean_dbg_3 LIKE '% - %' THEN TRIM(SPLIT_PART(clean_dbg_3,' - ',2))
                ELSE TRIM(clean_dbg_3)
            END
        ))),' & ') AS clean_diagnosis,
        CASE
            WHEN clean_disease_burden_group ILIKE '%NCD - Oncology%' THEN 'ONCOLOGY'
            WHEN clean_disease_burden_group ILIKE '%MNCH - Maternal%'
              OR clean_disease_burden_group ILIKE '%MNCH - Perinatal%'
              OR clean_disease_burden_group ILIKE '%MNCH - Congenital%'
              OR clean_disease_burden_group ILIKE '%MNCH - Nutrition%' THEN 'MATERNAL'
            WHEN clean_disease_burden_group ILIKE '%NCD - Mental Health%' THEN 'MENTAL_HEALTH'
            WHEN clean_disease_burden_group ILIKE '%NCD - Cardiovascular%'
              OR clean_disease_burden_group ILIKE '%NCD - Diabetes%'
              OR clean_disease_burden_group ILIKE '%NCD - Renal%'
              OR clean_disease_burden_group ILIKE '%NCD - Respiratory%'
              OR clean_disease_burden_group ILIKE '%NCD - Neurologic%'
              OR clean_disease_burden_group ILIKE '%NCD - Endocrine%'
              OR clean_disease_burden_group ILIKE '%NCD - Haematology%'
              OR clean_disease_burden_group ILIKE '%NCD - Musculoskeletal%'
              OR clean_disease_burden_group ILIKE '%NCD - Genitourinary%'
              OR clean_disease_burden_group ILIKE '%NCD - Digestive%'
              OR clean_disease_burden_group ILIKE '%NCD - Ophthalmology%'
              OR clean_disease_burden_group ILIKE '%NCD - Dermatology%'
              OR clean_disease_burden_group ILIKE '%Communicable - HIV/AIDS%'
              OR clean_disease_burden_group ILIKE '%Communicable - TB%'
              OR clean_disease_burden_group ILIKE '%Communicable - Hepatitis%' THEN 'CHRONIC'
            ELSE 'EXCLUDED'
        END AS acquisition_segment,
        IFF(acquisition_segment != 'EXCLUDED',TRUE,FALSE) AS is_retention_universe
    FROM HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED i
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES dn
        ON i.visit_id = dn.visit_id AND i.source_schema = dn.source_schema
    WHERE i.source_schema = '{schema}'
      AND dn.created_at  >= '2024-09-01'
    GROUP BY ALL
),
visits AS (
    SELECT
        v.source_schema,
        v.id         AS visit_id,
        v.patient    AS patient_id,
        v.created_at AS visit_date,
        p.sex        AS gender,
        CASE
            WHEN p.dob IS NULL THEN 'Unknown'
            WHEN TIMESTAMPDIFF('year',p.dob,v.created_at) < 5  THEN 'Toddler (0-4)'
            WHEN TIMESTAMPDIFF('year',p.dob,v.created_at) < 13 THEN 'Child (5-12)'
            WHEN TIMESTAMPDIFF('year',p.dob,v.created_at) < 18 THEN 'Adolescent (13-17)'
            WHEN TIMESTAMPDIFF('year',p.dob,v.created_at) < 25 THEN 'Youth (18-24)'
            WHEN TIMESTAMPDIFF('year',p.dob,v.created_at) < 35 THEN 'Young Adult (25-34)'
            WHEN TIMESTAMPDIFF('year',p.dob,v.created_at) < 45 THEN 'Adult (35-44)'
            WHEN TIMESTAMPDIFF('year',p.dob,v.created_at) < 55 THEN 'Middle Age (45-54)'
            WHEN TIMESTAMPDIFF('year',p.dob,v.created_at) < 65 THEN 'Older Adult (55-64)'
            ELSE 'Senior (65+)'
        END AS age_group
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS p
        ON v.patient = p.patient_id AND v.source_schema = p.source_schema
    WHERE v.source_schema = '{schema}'
      AND p.dob IS NOT NULL
      AND UPPER(COALESCE(p.sex,'')) IN ('F','FEMALE','M','MALE')
),
inpatient_visits AS (
    SELECT DISTINCT visit_id,
        LOWER(REPLACE(source_schema,'_CLEAN','')) AS source_schema
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS
),
patient_first_visit AS (
    SELECT source_schema, patient_id,
        MIN(visit_date) AS first_visit_date,
        MIN(visit_id)   AS first_visit_id
    FROM visits GROUP BY source_schema, patient_id
)
SELECT
    v.source_schema, v.visit_id, v.patient_id, v.visit_date,
    v.gender, v.age_group,
    s.acquisition_segment, s.clean_disease_burden_group, s.clean_diagnosis,
    s.is_comorbidity, s.has_chronic_diagnosis,
    IFF(v.visit_id = pf.first_visit_id,'new','returning') AS patient_type,
    IFF(ip.visit_id IS NOT NULL,'inpatient','outpatient')  AS visit_type
FROM visits v
INNER JOIN diag_clean s ON v.visit_id = s.visit_id AND v.source_schema = s.source_schema
LEFT JOIN patient_first_visit pf ON v.patient_id = pf.patient_id AND v.source_schema = pf.source_schema
LEFT JOIN inpatient_visits ip ON v.visit_id = ip.visit_id AND v.source_schema = ip.source_schema
WHERE s.is_retention_universe = TRUE
{wh}
"""
    return run_query(sql)


def load_cv_new_returning_trend(filters: dict, run_query) -> pd.DataFrame:
    """Section 2 Row 3 — monthly new vs returning patient counts and revenue."""
    vb = _visit_base_cte(filters, start_date='2024-09-01')
    sql = f"""
WITH {vb}
SELECT
    visit_month,
    patient_type,
    COUNT(DISTINCT patient_id) AS patients,
    SUM(visit_revenue)         AS revenue
FROM visit_base
GROUP BY visit_month, patient_type
ORDER BY visit_month, patient_type
"""
    return run_query(sql)


def load_tenri_benchmark(run_query) -> dict:
    """Return Tenri hospital new vs returning patient split as benchmark percentages.

    Queries the tenri source_schema directly (no filters applied) so the
    benchmark reflects Tenri's all-time patient mix, independent of the
    current dashboard filter window.
    Returns {"New": float, "Returning": float} in percentage points.
    """
    sql = """
WITH first_visits AS (
    SELECT
        patient,
        source_schema,
        MIN(created_at) AS first_visit_at
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    WHERE source_schema = 'tenri'
    GROUP BY patient, source_schema
),
classified AS (
    SELECT
        v.id AS visit_id,
        CASE WHEN v.created_at = fv.first_visit_at THEN 'New' ELSE 'Returning' END AS patient_type
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    JOIN first_visits fv
        ON v.patient = fv.patient AND v.source_schema = fv.source_schema
    WHERE v.source_schema = 'tenri'
),
totals AS (
    SELECT
        COUNT(DISTINCT CASE WHEN patient_type = 'New'       THEN visit_id END) AS new_count,
        COUNT(DISTINCT CASE WHEN patient_type = 'Returning' THEN visit_id END) AS ret_count
    FROM classified
)
SELECT
    new_count,
    ret_count,
    ROUND(new_count * 100.0 / NULLIF(new_count + ret_count, 0), 1) AS new_pct,
    ROUND(ret_count * 100.0 / NULLIF(new_count + ret_count, 0), 1) AS ret_pct
FROM totals
"""
    df = run_query(sql)
    if df.empty:
        return {"New": 40.0, "Returning": 60.0}
    row = df.iloc[0]
    return {
        "New":       float(row.get("new_pct") or 40.0),
        "Returning": float(row.get("ret_pct") or 60.0),
    }


def load_cv_diagnosis_by_age(filters: dict, run_query, age_group: str) -> pd.DataFrame:
    """Section 2 Row 4 — top 3 diagnoses by patient type for a selected age group."""
    vb  = _visit_base_cte(filters)
    ag  = age_group.replace("'", "''")
    sql = f"""
WITH {vb},
ranked AS (
    SELECT
        patient_type,
        disease_burden_group,
        COUNT(DISTINCT patient_id) AS patients,
        SUM(visit_revenue)         AS revenue,
        ROW_NUMBER() OVER (
            PARTITION BY patient_type
            ORDER BY COUNT(DISTINCT patient_id) DESC
        ) AS rk
    FROM visit_base
    WHERE age_group = '{ag}'
      AND disease_burden_group != 'Unclassified'
    GROUP BY patient_type, disease_burden_group
)
SELECT patient_type, disease_burden_group, patients, revenue
FROM ranked
WHERE rk <= 3
ORDER BY patient_type, rk
"""
    return run_query(sql)



def load_cv_age_groups(filters: dict, run_query) -> pd.DataFrame:
    """Age groups available for the Section 2 Row 4 dropdown, sorted by patient count."""
    vb = _visit_base_cte(filters)
    sql = f"""
WITH {vb}
SELECT age_group, COUNT(DISTINCT patient_id) AS patients
FROM visit_base
GROUP BY age_group
ORDER BY patients DESC
"""
    return run_query(sql)


# ══════════════════════════════════════════════════════════════════════════════
# WARD DEEP-DIVE
# ══════════════════════════════════════════════════════════════════════════════

def load_deepdive_ward_list(filters: dict, run_query) -> pd.DataFrame:
    """Wards sorted by total admissions descending — drives the deep-dive dropdown."""
    wh_a = _w_adm(filters)
    wsa  = _wsa(filters)
    mo   = _mo(filters)
    sql  = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
)
SELECT
    COALESCE(a.ward_name, 'Unknown') AS ward,
    COUNT(DISTINCT a.visit_id)       AS total_admissions
FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
INNER JOIN schema_anchor sa
    ON REPLACE(LOWER(a.source_schema), '_clean', '') = sa.source_schema
WHERE a.admitted_at >= DATEADD('month', -{mo}, sa.max_date)
{wh_a}
GROUP BY 1
ORDER BY 2 DESC
"""
    return run_query(sql)


def load_deepdive_monthly(filters: dict, run_query, ward_name: str) -> pd.DataFrame:
    """Monthly admission totals for one ward — used for dip detection."""
    wh_a = _w_adm(filters)
    wsa  = _wsa(filters)
    mo   = _mo(filters)
    w    = ward_name.replace("'", "''")
    sql  = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
)
SELECT
    DATE_TRUNC('month', a.admitted_at) AS admit_month,
    COUNT(DISTINCT a.visit_id)         AS total_admissions
FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
INNER JOIN schema_anchor sa
    ON REPLACE(LOWER(a.source_schema), '_clean', '') = sa.source_schema
WHERE COALESCE(a.ward_name, 'Unknown') = '{w}'
  AND a.admitted_at >= DATEADD('month', -{mo}, sa.max_date)
{wh_a}
GROUP BY 1
ORDER BY 1
"""
    return run_query(sql)


def load_deepdive_ward_id(filters: dict, run_query, ward_name: str) -> "int | None":
    """Return the ward_id for the selected ward — needed by the diagnosis query."""
    wsa = _wsa(filters)
    w   = ward_name.replace("'", "''")
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
)
SELECT a.ward_id
FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
INNER JOIN schema_anchor sa
    ON REPLACE(LOWER(a.source_schema), '_clean', '') = sa.source_schema
WHERE COALESCE(a.ward_name, 'Unknown') = '{w}'
  AND a.ward_id IS NOT NULL
LIMIT 1
"""
    df = run_query(sql)
    if df.empty or df.iloc[0, 0] is None:
        return None
    return int(df.iloc[0, 0])


def load_h3_diagnosis_trends(filters: dict, run_query, ward_id: int) -> pd.DataFrame:
    """
    Diagnosis trend lines for the H3 panel.

    Handles three data quality issues:
      1. disease_burden_group values of "Other" / "Communicable - Other"
         are replaced with the doctor note diagnosis.
      2. group_2 and group_3 nulls when a prior group exists are also
         substituted with the doctor note.
      3. All three burden groups are UNPIVOTED so each is counted separately.

    Returns one row per (month, diagnosis_name) for the top-5 diagnoses
    by total visit count across the full period for the ward.
    """
    wh_a = _w_adm(filters)
    wsa  = _wsa(filters)
    mo   = _mo(filters)
    sql  = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
raw_unpivoted AS (
    SELECT
        DATE_TRUNC('day', a.admitted_at)  AS daily,
        a.visit_id,
        pvt.group_source,
        pvt.raw_group_value,
        pvt.orig_group_1,
        pvt.orig_group_2,
        dn.diagnosis                      AS doctor_note_diagnosis
    FROM (
        SELECT
            visit_id,
            source_schema,
            disease_burden_group_1,
            disease_burden_group_2,
            disease_burden_group_3,
            disease_burden_group_1 AS orig_group_1,
            disease_burden_group_2 AS orig_group_2
        FROM HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED
    ) d
    UNPIVOT (
        raw_group_value FOR group_source IN (
            disease_burden_group_1,
            disease_burden_group_2,
            disease_burden_group_3
        )
    ) pvt
    JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
        ON  pvt.source_schema = REPLACE(LOWER(a.source_schema), '_clean', '')
        AND pvt.visit_id      = a.visit_id
    INNER JOIN schema_anchor sa
        ON REPLACE(LOWER(a.source_schema), '_clean', '') = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES dn
        ON pvt.visit_id     = dn.visit_id
       AND pvt.source_schema = dn.source_schema
    WHERE a.ward_id = {ward_id}
      AND a.admitted_at >= DATEADD('month', -{mo}, sa.max_date)
    {wh_a}
),
unified_diagnoses AS (
    SELECT
        daily,
        visit_id,
        CASE
            WHEN group_source = 'DISEASE_BURDEN_GROUP_1' THEN
                CASE WHEN raw_group_value ILIKE 'Other%'
                          OR raw_group_value ILIKE 'Communicable - Other%'
                     THEN doctor_note_diagnosis ELSE raw_group_value END
            WHEN group_source = 'DISEASE_BURDEN_GROUP_2' THEN
                CASE WHEN raw_group_value ILIKE 'Other%'
                          OR raw_group_value ILIKE 'Communicable - Other%'
                          OR (raw_group_value IS NULL AND orig_group_1 IS NOT NULL)
                     THEN doctor_note_diagnosis ELSE raw_group_value END
            WHEN group_source = 'DISEASE_BURDEN_GROUP_3' THEN
                CASE WHEN raw_group_value ILIKE 'Other%'
                          OR raw_group_value ILIKE 'Communicable - Other%'
                          OR (raw_group_value IS NULL
                              AND orig_group_1 IS NOT NULL
                              AND orig_group_2 IS NOT NULL)
                     THEN doctor_note_diagnosis ELSE raw_group_value END
        END AS diagnosis_name
    FROM raw_unpivoted
),
top_5_list AS (
    SELECT diagnosis_name
    FROM unified_diagnoses
    WHERE diagnosis_name IS NOT NULL
    GROUP BY diagnosis_name
    ORDER BY COUNT(DISTINCT visit_id) DESC
    LIMIT 5
)
SELECT
    DATE_TRUNC('month', u.daily) AS visit_month,
    u.diagnosis_name,
    COUNT(DISTINCT u.visit_id)   AS monthly_visit_count
FROM unified_diagnoses u
JOIN top_5_list t ON u.diagnosis_name = t.diagnosis_name
WHERE u.diagnosis_name IS NOT NULL
GROUP BY ALL
ORDER BY visit_month ASC, monthly_visit_count DESC
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
WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
            WHEN rp.dob IS NULL THEN 'Unknown'
            WHEN TIMESTAMPDIFF('year', rp.dob, v.created_at) < 5
                THEN 'Toddler (0–4)'
            WHEN TIMESTAMPDIFF('year', rp.dob, v.created_at) < 13
                THEN 'Child (5–12)'
            WHEN TIMESTAMPDIFF('year', rp.dob, v.created_at) < 18
                THEN 'Adolescent (13–17)'
            WHEN TIMESTAMPDIFF('year', rp.dob, v.created_at) < 25
                THEN 'Youth (18–24)'
            WHEN TIMESTAMPDIFF('year', rp.dob, v.created_at) < 35
                THEN 'Young Adult (25–34)'
            WHEN TIMESTAMPDIFF('year', rp.dob, v.created_at) < 45
                THEN 'Adult (35–44)'
            WHEN TIMESTAMPDIFF('year', rp.dob, v.created_at) < 55
                THEN 'Middle Age (45–54)'
            WHEN TIMESTAMPDIFF('year', rp.dob, v.created_at) < 65
                THEN 'Older Adult (55–64)'
            ELSE 'Senior (65+)'
        END                                             AS age_group,
    CASE
        WHEN dx.disease_burden_group_1 ILIKE '%Endocrin%'
          OR dx.disease_burden_group_1 ILIKE '%Diabetes%' THEN 'NCD — Diabetes & Metabolic'
        ELSE dx.disease_burden_group_1
    END                                                   AS chronic_condition,
    COUNT(DISTINCT v.patient)                             AS patient_count
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
    ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS rp
    ON v.patient = rp.patient_id AND v.source_schema = rp.source_schema
WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
      AND (i.cancelled IS NULL OR i.cancelled = 0)
    {wh}
    GROUP BY 1, 2
),
visit_counts AS (
    SELECT v.source_schema, v.patient, COUNT(DISTINCT v.id) AS visit_count
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN htn_patients hp ON v.patient = hp.patient AND v.source_schema = hp.source_schema
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
WHERE v.created_at >=  '2024-09-01'
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
           END AS disease_group,
           CASE
               WHEN TIMESTAMPDIFF('year', rp.dob, v.created_at) < 18 THEN 'Paediatric (<18)'
               WHEN UPPER(COALESCE(rp.sex, '')) IN ('F', 'FEMALE')   THEN 'Adult Female'
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
      AND (
          dx.icd10_code_1 LIKE 'A0%' OR dx.icd10_code_1 LIKE 'A01%'
          OR dx.icd10_code_1 LIKE 'A15%' OR dx.icd10_code_1 LIKE 'A16%'
          OR dx.icd10_code_1 LIKE 'A17%' OR dx.icd10_code_1 LIKE 'A18%'
          OR dx.icd10_code_1 LIKE 'A19%'
          OR dx.icd10_code_1 LIKE 'B2%' OR dx.icd10_code_1 LIKE 'B5%'
          OR dx.icd10_code_1 LIKE 'J0%'
          OR dx.disease_group_1 ILIKE ANY (
              '%Tubercul%','%Malaria%','%URTI%','%Typhoid%',
              '%Enteric%','%Gastroenterit%','%HIV%','%Communicable%')
      )
    {wh}
    QUALIFY ROW_NUMBER() OVER (PARTITION BY v.id ORDER BY dx.icd10_code_1 NULLS LAST) = 1
),
top10 AS (
    SELECT disease_group
    FROM comm_visits
    WHERE disease_group IS NOT NULL
    GROUP BY 1
    QUALIFY RANK() OVER (ORDER BY COUNT(DISTINCT visit_id) DESC) <= 10
),
any_inv AS (
    SELECT DISTINCT visit_id
    FROM HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS
    WHERE (cancelled IS NULL OR cancelled = 0)
),
confirmed_lab AS (
    SELECT DISTINCT visit_id
    FROM HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS
    WHERE (cancelled IS NULL OR cancelled = 0)
      AND investigation_type IS NOT NULL
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
        COUNT(DISTINCT CASE WHEN cl.visit_id IS NOT NULL THEN cv.visit_id END),
        COUNT(DISTINCT cv.visit_id)
    ) * 100, 1)                                           AS lab_confirmation_pct,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN ai.visit_id IS NOT NULL THEN cv.visit_id END),
        COUNT(DISTINCT cv.visit_id)
    ) * 100, 1)                                           AS data_completeness_pct,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN ipf.visit_id IS NOT NULL THEN cv.visit_id END),
        COUNT(DISTINCT cv.visit_id)
    ) * 100, 1)                                           AS inpatient_admission_pct,
    COALESCE(cm.primary_comorbidity, '—')                 AS primary_comorbidity,
    pm.primary_payer
FROM comm_visits cv
INNER JOIN top10 t  ON cv.disease_group = t.disease_group
LEFT JOIN confirmed_lab cl ON cv.visit_id = cl.visit_id
LEFT JOIN any_inv ai       ON cv.visit_id = ai.visit_id
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
    WHERE v.created_at >=  '2024-09-01'
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
           CASE WHEN p.sex IS NULL THEN 'Unknown'
                ELSE UPPER(CAST(p.sex AS VARCHAR))
           END AS sex
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS p
        ON v.patient = p.patient_id AND v.source_schema = p.source_schema
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v2.created_at >=  '2024-09-01'
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
WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
      AND (dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1)
    {wh}
),
last_visit AS (
    SELECT v.source_schema, v.patient, MAX(v.created_at) AS last_visit_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
    {wh}
),
last_visit AS (
    SELECT v.source_schema, v.patient,
           MAX(v.created_at) AS last_visit_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >=  '2024-09-01'
    {wh}
    GROUP BY 1, 2
),
latest_clinician AS (
    SELECT v.source_schema, v.patient,
           CAST(v.user AS VARCHAR) AS current_clinician
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
    WHERE v.created_at >=  '2024-09-01'
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
vitals_ranked AS (
    SELECT
        v.source_schema,
        v.patient,
        vt.bp_systolic,
        ROW_NUMBER() OVER (
            PARTITION BY v.source_schema, v.patient
            ORDER BY v.created_at DESC
        ) AS rn
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VITALS vt
        ON vt.visit_id = v.id AND vt.source_schema = v.source_schema
    WHERE v.created_at >=  '2024-09-01'
      AND vt.bp_systolic IS NOT NULL
    {wh}
),
worsening_vitals AS (
    SELECT source_schema, patient, 1 AS has_worsening_vitals
    FROM vitals_ranked
    WHERE rn <= 3
    GROUP BY source_schema, patient
    HAVING COUNT(*) >= 3
       AND MAX(CASE WHEN rn = 1 THEN bp_systolic END) >
           AVG(CASE WHEN rn IN (2, 3) THEN bp_systolic END)
),
rx_ordered AS (
    SELECT
        v.source_schema,
        v.patient,
        pp.drug_name,
        LAG(pp.drug_name) OVER (
            PARTITION BY v.source_schema, v.patient
            ORDER BY v.created_at, pp.prescription_created_at
        ) AS prev_drug,
        ROW_NUMBER() OVER (
            PARTITION BY v.source_schema, v.patient
            ORDER BY v.created_at DESC, pp.prescription_created_at DESC
        ) AS rn
    FROM HOSPITALS.STAGING.STG_PRESCRIPTION_PAYMENTS pp
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON pp.visit_id = v.id AND pp.source_schema = v.source_schema
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >=  '2024-09-01'
      AND (pp.stopped IS NULL OR pp.stopped = 0)
      AND (pp.canceled IS NULL OR pp.canceled = 0)
      AND (pp.remove_from_report IS NULL OR pp.remove_from_report = 0)
    {wh}
),
medication_change AS (
    SELECT DISTINCT source_schema, patient, 1 AS has_medication_change
    FROM rx_ordered
    WHERE rn = 1
      AND prev_drug IS NOT NULL
      AND drug_name != prev_drug
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
      AND v_op.created_at >=  '2024-09-01'
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
    COALESCE(wv.has_worsening_vitals, 0)                  AS has_worsening_vitals,
    COALESCE(mc.has_medication_change, 0)                 AS has_medication_change,
    COALESCE(oi.had_op_to_ip, 0)                          AS had_op_to_ip,
    COALESCE(cc.unique_clinicians, 1)                     AS unique_clinicians,
    COALESCE(lc.current_clinician, 'Unknown')             AS current_clinician,
    CASE
        WHEN COALESCE(un.has_undetected_ncd, 0) = 1
          OR COALESCE(wv.has_worsening_vitals, 0) = 1     THEN 'HIGH'
        WHEN COALESCE(mc.has_medication_change, 0) = 1
          OR DATEDIFF('day', lv.last_visit_date, sa.max_date) >= 90
                                                           THEN 'MEDIUM'
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
LEFT JOIN worsening_vitals wv
    ON ap.patient = wv.patient AND ap.source_schema = wv.source_schema
LEFT JOIN medication_change mc
    ON ap.patient = mc.patient AND ap.source_schema = mc.source_schema
LEFT JOIN op_to_ip oi
    ON ap.patient = oi.patient AND ap.source_schema = oi.source_schema
LEFT JOIN clinician_count cc
    ON ap.patient = cc.patient AND ap.source_schema = cc.source_schema
LEFT JOIN latest_clinician lc
    ON ap.patient = lc.patient AND ap.source_schema = lc.source_schema
LEFT JOIN demographics d
    ON ap.patient = d.patient_id AND ap.source_schema = d.source_schema
WHERE DATEDIFF('day', lv.last_visit_date, sa.max_date) <= 365
ORDER BY
    CASE
        WHEN COALESCE(un.has_undetected_ncd, 0) = 1      THEN 1
        WHEN COALESCE(wv.has_worsening_vitals, 0) = 1    THEN 2
        WHEN COALESCE(mc.has_medication_change, 0) = 1   THEN 3
        WHEN DATEDIFF('day', lv.last_visit_date, sa.max_date) >= 90 THEN 4
        ELSE 5
    END,
    DATEDIFF('day', lv.last_visit_date, sa.max_date) DESC
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

# ══════════════════════════════════════════════════════════════════════════════
# OPD TO IPD CONVERSION TAB
# ══════════════════════════════════════════════════════════════════════════════

_OPD_IPD_FLOOR = "'2024-09-01'"

_SEGMENT_CASE = """
        CASE
            WHEN COALESCE(nf.oncology_flag,0)=1
              OR COALESCE(dx.disease_burden_group_1,'') ILIKE '%oncol%'
              OR COALESCE(dx.disease_burden_group_1,'') ILIKE '%cancer%'
              OR COALESCE(dx.disease_burden_group_2,'') ILIKE '%oncol%'
              OR COALESCE(dx.disease_burden_group_2,'') ILIKE '%cancer%'
                THEN 'Oncology'
            WHEN COALESCE(nf.maternal_flag,0)=1
              OR COALESCE(dx.disease_burden_group_1,'') ILIKE '%maternal%'
              OR COALESCE(dx.disease_burden_group_1,'') ILIKE '%obstet%'
              OR COALESCE(dx.disease_burden_group_2,'') ILIKE '%maternal%'
              OR COALESCE(dx.disease_burden_group_2,'') ILIKE '%obstet%'
                THEN 'Maternal'
            WHEN COALESCE(nf.mental_flag,0)=1
              OR COALESCE(dx.disease_burden_group_1,'') ILIKE '%mental%'
              OR COALESCE(dx.disease_burden_group_1,'') ILIKE '%psychiatr%'
              OR COALESCE(dx.disease_burden_group_2,'') ILIKE '%mental%'
                THEN 'Mental Health'
            WHEN COALESCE(dx.is_chronic_1,0)=1
              OR COALESCE(dx.is_chronic_2,0)=1
              OR COALESCE(nf.chronic_flag,0)=1
                THEN 'Chronic'
            ELSE NULL
        END"""

_NOTE_FLAGS_CTE = """
note_flags AS (
    SELECT source_schema, visit_id,
        MAX(CASE WHEN diagnosis ILIKE '%oncol%' OR diagnosis ILIKE '%cancer%'
                   OR diagnosis ILIKE '%chemo%' THEN 1 ELSE 0 END)            AS oncology_flag,
        MAX(CASE WHEN diagnosis ILIKE '%maternal%' OR diagnosis ILIKE '%obstet%'
                   OR diagnosis ILIKE '%antenatal%' OR diagnosis ILIKE '%anc%'
                   OR diagnosis ILIKE '%pregnant%' OR diagnosis ILIKE '%labour%'
                   OR diagnosis ILIKE '%delivery%' THEN 1 ELSE 0 END)          AS maternal_flag,
        MAX(CASE WHEN diagnosis ILIKE '%mental%' OR diagnosis ILIKE '%psychiatr%'
                   OR diagnosis ILIKE '%depress%' OR diagnosis ILIKE '%anxiety%'
                   THEN 1 ELSE 0 END)                                          AS mental_flag,
        MAX(CASE WHEN diagnosis ILIKE '%hypertension%' OR diagnosis ILIKE '%diabetes%'
                   OR diagnosis ILIKE '%hiv%' OR diagnosis ILIKE '%chronic%'
                   OR diagnosis ILIKE '%ckd%' THEN 1 ELSE 0 END)               AS chronic_flag
    FROM HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES
    GROUP BY 1, 2
)"""


def load_opd_ipd_segments(filters: dict, run_query) -> pd.DataFrame:
    """OPD->IPD: overall totals + per acquisition segment. Columns: acquisition_segment,
    total_opd_visits, ipd_admissions, conversion_rate_pct."""
    wh   = _w(filters)
    wh_a = _w_adm(filters)
    wsa  = _wsa(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
{_NOTE_FLAGS_CTE},
admissions AS (
    SELECT a.visit_id, REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    WHERE 1=1 {wh_a}
),
visit_base AS (
    SELECT
        v.source_schema, v.id AS visit_id,
        {_SEGMENT_CASE} AS acquisition_segment,
        CASE WHEN a.visit_id IS NOT NULL THEN 1 ELSE 0 END AS is_ipd
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT  JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT  JOIN note_flags nf ON v.id = nf.visit_id AND v.source_schema = nf.source_schema
    LEFT  JOIN admissions a  ON v.id = a.visit_id  AND v.source_schema = a.source_schema
    WHERE v.created_at >= {_OPD_IPD_FLOOR}
    {wh}
)
SELECT 'All' AS acquisition_segment,
    COUNT(DISTINCT visit_id) AS total_opd_visits,
    SUM(is_ipd) AS ipd_admissions,
    ROUND(DIV0(SUM(is_ipd), COUNT(DISTINCT visit_id)) * 100, 2) AS conversion_rate_pct
FROM visit_base
UNION ALL
SELECT acquisition_segment,
    COUNT(DISTINCT visit_id) AS total_opd_visits,
    SUM(is_ipd) AS ipd_admissions,
    ROUND(DIV0(SUM(is_ipd), COUNT(DISTINCT visit_id)) * 100, 2) AS conversion_rate_pct
FROM visit_base
WHERE acquisition_segment IS NOT NULL
GROUP BY acquisition_segment
ORDER BY CASE WHEN acquisition_segment = 'All' THEN 0 ELSE 1 END, conversion_rate_pct DESC
"""
    return run_query(sql)


def load_opd_ipd_monthly(filters: dict, run_query) -> pd.DataFrame:
    """OPD->IPD: monthly rate for overall and retention-universe series.
    Columns: series, visit_month, total_opd_visits, ipd_admissions, monthly_rate_pct."""
    wh   = _w(filters)
    wh_a = _w_adm(filters)
    wsa  = _wsa(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
{_NOTE_FLAGS_CTE},
admissions AS (
    SELECT a.visit_id, REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    WHERE 1=1 {wh_a}
),
visit_base AS (
    SELECT
        v.source_schema, v.id AS visit_id,
        DATE_TRUNC('month', v.created_at) AS visit_month,
        {_SEGMENT_CASE} AS acquisition_segment,
        CASE WHEN a.visit_id IS NOT NULL THEN 1 ELSE 0 END AS is_ipd
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT  JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT  JOIN note_flags nf ON v.id = nf.visit_id AND v.source_schema = nf.source_schema
    LEFT  JOIN admissions a  ON v.id = a.visit_id  AND v.source_schema = a.source_schema
    WHERE v.created_at >= {_OPD_IPD_FLOOR}
    {wh}
)
SELECT 'overall' AS series, visit_month,
    COUNT(DISTINCT visit_id) AS total_opd_visits,
    SUM(is_ipd) AS ipd_admissions,
    ROUND(DIV0(SUM(is_ipd), COUNT(DISTINCT visit_id)) * 100, 2) AS monthly_rate_pct
FROM visit_base
GROUP BY visit_month
UNION ALL
SELECT 'retention' AS series, visit_month,
    COUNT(DISTINCT visit_id) AS total_opd_visits,
    SUM(is_ipd) AS ipd_admissions,
    ROUND(DIV0(SUM(is_ipd), COUNT(DISTINCT visit_id)) * 100, 2) AS monthly_rate_pct
FROM visit_base
WHERE acquisition_segment IS NOT NULL
GROUP BY visit_month
ORDER BY series, visit_month
"""
    return run_query(sql)


def load_opd_ipd_by_diagnosis(filters: dict, run_query) -> pd.DataFrame:
    """OPD->IPD Section B: conversion rate per cleaned diagnosis name.
    Columns: cleaned_diagnosis_name, total_opd_cases_with_this_diagnosis,
             successful_ipd_admissions, true_opd_to_ipd_conversion_rate."""
    wh   = _w(filters)
    wh_a = _w_adm(filters)
    wsa  = _wsa(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
admissions AS (
    SELECT a.visit_id, REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    WHERE 1=1 {wh_a}
),
visit_dx AS (
    SELECT
        v.source_schema, v.id AS visit_id,
        COALESCE(
            NULLIF(TRIM(dx.disease_burden_group_1), ''),
            CASE
                WHEN n.diagnosis ILIKE '%typhoid%'                       THEN 'Typhoid'
                WHEN n.diagnosis ILIKE '%malaria%'                       THEN 'Malaria'
                WHEN n.diagnosis ILIKE '%sepsis%'                        THEN 'Sepsis'
                WHEN n.diagnosis ILIKE '%upper airway%'
                  OR n.diagnosis ILIKE '%respiratory%'                   THEN 'Chr Upper Airway'
                WHEN n.diagnosis ILIKE '%oncol%' OR n.diagnosis ILIKE '%cancer%' THEN 'Oncology'
                WHEN n.diagnosis ILIKE '%hypertension%'                  THEN 'Hypertension'
                WHEN n.diagnosis ILIKE '%antenatal%'
                  OR n.diagnosis ILIKE '%anc%'                           THEN 'Antenatal Care'
                ELSE NULL
            END
        ) AS cleaned_diagnosis_name,
        CASE WHEN a.visit_id IS NOT NULL THEN 1 ELSE 0 END AS is_ipd
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT  JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT  JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON v.id = n.visit_id AND v.source_schema = n.source_schema
    LEFT  JOIN admissions a ON v.id = a.visit_id AND v.source_schema = a.source_schema
    WHERE v.created_at >= {_OPD_IPD_FLOOR}
    {wh}
)
SELECT
    cleaned_diagnosis_name,
    COUNT(DISTINCT visit_id)                                     AS total_opd_cases_with_this_diagnosis,
    SUM(is_ipd)                                                  AS successful_ipd_admissions,
    ROUND(DIV0(SUM(is_ipd), COUNT(DISTINCT visit_id)) * 100, 2) AS true_opd_to_ipd_conversion_rate
FROM visit_dx
WHERE cleaned_diagnosis_name IS NOT NULL
GROUP BY 1
HAVING COUNT(DISTINCT visit_id) >= 5
ORDER BY true_opd_to_ipd_conversion_rate DESC
"""
    return run_query(sql)


def load_comorbidity_conversion(filters: dict, run_query) -> pd.DataFrame:
    """OPD->IPD Section C: conversion by comorbidity group + monthly trend.
    Aggregate rows have visit_month=NULL. Columns: patient_group, visit_month,
    total_opd_visits, ipd_admissions, conversion_rate_pct, delta_from_group_avg."""
    wh   = _w(filters)
    wh_a = _w_adm(filters)
    wsa  = _wsa(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
admissions AS (
    SELECT a.visit_id, REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    WHERE 1=1 {wh_a}
),
visit_base AS (
    SELECT
        v.source_schema, v.id AS visit_id,
        DATE_TRUNC('month', v.created_at) AS visit_month,
        CASE
            WHEN COALESCE(dx.is_chronic_1,0)=1 AND COALESCE(dx.is_chronic_2,0)=1
              AND COALESCE(NULLIF(TRIM(dx.disease_burden_group_2),''), NULL) IS NOT NULL
                THEN 'Chronic Comorbid'
            WHEN COALESCE(NULLIF(TRIM(dx.disease_burden_group_2),''), NULL) IS NOT NULL
                THEN 'Comorbid'
            ELSE 'Single Diagnosis'
        END AS patient_group,
        CASE WHEN a.visit_id IS NOT NULL THEN 1 ELSE 0 END AS is_ipd
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT  JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT  JOIN admissions a ON v.id = a.visit_id AND v.source_schema = a.source_schema
    WHERE v.created_at >= {_OPD_IPD_FLOOR}
    {wh}
),
overall_avg AS (
    SELECT ROUND(DIV0(SUM(is_ipd), COUNT(DISTINCT visit_id)) * 100, 2) AS avg_rate
    FROM visit_base
),
group_agg AS (
    SELECT patient_group,
        COUNT(DISTINCT visit_id) AS total_opd_visits,
        SUM(is_ipd) AS ipd_admissions,
        ROUND(DIV0(SUM(is_ipd), COUNT(DISTINCT visit_id)) * 100, 2) AS conversion_rate_pct
    FROM visit_base
    GROUP BY patient_group
)
SELECT
    g.patient_group, NULL::DATE AS visit_month,
    g.total_opd_visits, g.ipd_admissions, g.conversion_rate_pct,
    ROUND(g.conversion_rate_pct - o.avg_rate, 2) AS delta_from_group_avg
FROM group_agg g CROSS JOIN overall_avg o
UNION ALL
SELECT
    patient_group, visit_month,
    COUNT(DISTINCT visit_id) AS total_opd_visits,
    SUM(is_ipd) AS ipd_admissions,
    ROUND(DIV0(SUM(is_ipd), COUNT(DISTINCT visit_id)) * 100, 2) AS conversion_rate_pct,
    NULL::FLOAT AS delta_from_group_avg
FROM visit_base
GROUP BY patient_group, visit_month
ORDER BY visit_month NULLS FIRST, patient_group
"""
    return run_query(sql)


def load_opd_ipd_chronic_by_age(filters: dict, run_query) -> pd.DataFrame:
    """OPD->IPD Section C right: chronic segment conversion by age group.
    Columns: age_group, total_opd_visits, ipd_admissions, conversion_rate_pct."""
    wh   = _w(filters)
    wh_a = _w_adm(filters)
    wsa  = _wsa(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
{_NOTE_FLAGS_CTE},
admissions AS (
    SELECT a.visit_id, REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    WHERE 1=1 {wh_a}
),
visit_base AS (
    SELECT
        v.source_schema, v.id AS visit_id,
        CASE
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 5   THEN 'Toddler (0-4)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 13  THEN 'Child (5-12)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 18  THEN 'Adolescent (13-17)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 25  THEN 'Youth (18-24)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 35  THEN 'Young Adult (25-34)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 45  THEN 'Adult (35-44)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 55  THEN 'Middle Age (45-54)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 65  THEN 'Older Adult (55-64)'
            ELSE 'Senior (65+)'
        END AS age_group,
        CASE WHEN a.visit_id IS NOT NULL THEN 1 ELSE 0 END AS is_ipd
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT  JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT  JOIN note_flags nf ON v.id = nf.visit_id AND v.source_schema = nf.source_schema
    LEFT  JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS rp
        ON v.patient = rp.patient_id AND v.source_schema = rp.source_schema
    LEFT  JOIN admissions a ON v.id = a.visit_id AND v.source_schema = a.source_schema
    WHERE v.created_at >= {_OPD_IPD_FLOOR}
      AND (COALESCE(dx.is_chronic_1,0)=1 OR COALESCE(dx.is_chronic_2,0)=1
           OR COALESCE(nf.chronic_flag,0)=1)
    {wh}
)
SELECT age_group,
    COUNT(DISTINCT visit_id) AS total_opd_visits,
    SUM(is_ipd) AS ipd_admissions,
    ROUND(DIV0(SUM(is_ipd), COUNT(DISTINCT visit_id)) * 100, 2) AS conversion_rate_pct
FROM visit_base
GROUP BY age_group
HAVING COUNT(DISTINCT visit_id) >= 5
ORDER BY conversion_rate_pct DESC
"""
    return run_query(sql)


def load_escalation_by_age(filters: dict, run_query) -> pd.DataFrame:
    """Section D: 72h OP->IP escalations by age group.
    Columns: age_group, total_escalations."""
    wh   = _w(filters)
    wh_a = _w_adm(filters)
    wsa  = _wsa(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
admissions AS (
    SELECT a.visit_id, a.admitted_at,
           REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    WHERE 1=1 {wh_a}
),
escalations AS (
    SELECT v.id AS visit_id, v.source_schema,
        CASE
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 5   THEN 'Toddler (0-4)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 13  THEN 'Child (5-12)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 18  THEN 'Adolescent (13-17)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 25  THEN 'Youth (18-24)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 35  THEN 'Young Adult (25-34)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 45  THEN 'Adult (35-44)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 55  THEN 'Middle Age (45-54)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 65  THEN 'Older Adult (55-64)'
            ELSE 'Senior (65+)'
        END AS age_group
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN admissions a
        ON v.id = a.visit_id AND v.source_schema = a.source_schema
        AND DATEDIFF('hour', v.created_at, a.admitted_at) BETWEEN 0 AND 72
    LEFT  JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS rp
        ON v.patient = rp.patient_id AND v.source_schema = rp.source_schema
    WHERE v.created_at >= {_OPD_IPD_FLOOR}
    {wh}
)
SELECT age_group, COUNT(DISTINCT visit_id) AS total_escalations
FROM escalations
GROUP BY age_group
ORDER BY total_escalations DESC
"""
    return run_query(sql)


def load_operational_triangle(filters: dict, run_query) -> pd.DataFrame:
    """Section E: monthly clinician load + conversion rate + wait gap + strain signal.
    Covers last 13 months. Columns: visit_month, conversion_rate_pct,
    avg_visits_per_clinician, wait_time_gap_mins, strain_signal."""
    wh   = _w(filters)
    wh_a = _w_adm(filters)
    wsa  = _wsa(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
admissions AS (
    SELECT a.visit_id, REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    WHERE 1=1 {wh_a}
),
first_note AS (
    SELECT visit_id, source_schema, MIN(created_at) AS note_time
    FROM HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES
    GROUP BY 1, 2
),
monthly AS (
    SELECT
        DATE_TRUNC('month', v.created_at)                            AS visit_month,
        COUNT(DISTINCT v.id)                                         AS total_opd_visits,
        COUNT(DISTINCT v.user)                                       AS total_clinicians,
        SUM(CASE WHEN a.visit_id IS NOT NULL THEN 1 ELSE 0 END)     AS ipd_admissions,
        AVG(DATEDIFF('minute', v.created_at, fn.note_time))         AS avg_wait_to_note_mins
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT  JOIN admissions a  ON v.id = a.visit_id  AND v.source_schema = a.source_schema
    LEFT  JOIN first_note fn ON v.id = fn.visit_id AND v.source_schema = fn.source_schema
    WHERE v.created_at >= DATEADD('month', -13, sa.max_date)
      AND v.user IS NOT NULL
    {wh}
    GROUP BY 1
)
SELECT
    visit_month,
    ROUND(DIV0(ipd_admissions, total_opd_visits) * 100, 2)           AS conversion_rate_pct,
    ROUND(DIV0(total_opd_visits, NULLIF(total_clinicians,0)), 1)      AS avg_visits_per_clinician,
    ROUND(COALESCE(avg_wait_to_note_mins, 30) - 30, 1)               AS wait_time_gap_mins,
    CASE
        WHEN DIV0(total_opd_visits, NULLIF(total_clinicians,0)) > 90
             AND DIV0(ipd_admissions, total_opd_visits) * 100 < 5.0  THEN 'HIGH_STRAIN'
        WHEN DIV0(total_opd_visits, NULLIF(total_clinicians,0)) > 75
             AND (DIV0(ipd_admissions, total_opd_visits) * 100 < 5.0
                  OR ABS(COALESCE(avg_wait_to_note_mins,30) - 30) > 8) THEN 'CAPACITY_GAP'
        ELSE 'AS_EXPECTED'
    END                                                               AS strain_signal
FROM monthly
ORDER BY visit_month
"""
    return run_query(sql)


# ── OPD → IPD v2 functions (column contract matches render_tab_opd_ipd) ──────

def load_opd_ipd_overall(filters: dict, run_query) -> pd.DataFrame:
    """Single-row header KPIs.
    Columns (lowercase after _normalise): true_average_opd_to_ipd_rate,
    total_opd_to_ipd, total_opd_visits."""
    wh   = _w(filters)
    wh_a = _w_adm(filters)
    wsa  = _wsa(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
admissions AS (
    SELECT a.visit_id, REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    WHERE 1=1 {wh_a}
),
visit_base AS (
    SELECT v.source_schema, v.id AS visit_id,
        CASE WHEN a.visit_id IS NOT NULL THEN 1 ELSE 0 END AS is_ipd
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT  JOIN admissions a ON v.id = a.visit_id AND v.source_schema = a.source_schema
    WHERE v.created_at >= {_OPD_IPD_FLOOR}
    {wh}
)
SELECT
    ROUND(DIV0(SUM(is_ipd), COUNT(DISTINCT visit_id)) * 100, 2) AS TRUE_AVERAGE_OPD_TO_IPD_RATE,
    SUM(is_ipd)                                                  AS TOTAL_OPD_TO_IPD,
    COUNT(DISTINCT visit_id)                                     AS TOTAL_OPD_VISITS
FROM visit_base
"""
    return run_query(sql)


def load_opd_ipd_monthly_rate(filters: dict, run_query) -> pd.DataFrame:
    """Monthly overall conversion rate.
    Columns: conversion_month, monthly_opd_to_ipd_rate."""
    wh   = _w(filters)
    wh_a = _w_adm(filters)
    wsa  = _wsa(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
admissions AS (
    SELECT a.visit_id, REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    WHERE 1=1 {wh_a}
),
visit_base AS (
    SELECT v.source_schema, v.id AS visit_id,
        DATE_TRUNC('month', v.created_at) AS visit_month,
        CASE WHEN a.visit_id IS NOT NULL THEN 1 ELSE 0 END AS is_ipd
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT  JOIN admissions a ON v.id = a.visit_id AND v.source_schema = a.source_schema
    WHERE v.created_at >= {_OPD_IPD_FLOOR}
    {wh}
)
SELECT
    visit_month                                                      AS CONVERSION_MONTH,
    ROUND(DIV0(SUM(is_ipd), COUNT(DISTINCT visit_id)) * 100, 2)    AS MONTHLY_OPD_TO_IPD_RATE
FROM visit_base
GROUP BY visit_month
ORDER BY visit_month
"""
    return run_query(sql)


def load_opd_ipd_disease_mix(filters: dict, run_query) -> pd.DataFrame:
    """Section B: per-diagnosis breakdown.
    Columns: cleaned_diagnosis_name, total_opd_cases_with_this_diagnosis,
             successful_ipd_admissions."""
    wh   = _w(filters)
    wh_a = _w_adm(filters)
    wsa  = _wsa(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
admissions AS (
    SELECT a.visit_id, REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    WHERE 1=1 {wh_a}
),
visit_dx AS (
    SELECT v.source_schema, v.id AS visit_id,
        COALESCE(
            NULLIF(TRIM(dx.disease_burden_group_1), ''),
            CASE
                WHEN n.diagnosis ILIKE '%typhoid%'                              THEN 'Typhoid'
                WHEN n.diagnosis ILIKE '%malaria%'                              THEN 'Malaria'
                WHEN n.diagnosis ILIKE '%sepsis%'                               THEN 'Sepsis'
                WHEN n.diagnosis ILIKE '%upper airway%'
                  OR n.diagnosis ILIKE '%respiratory%'                          THEN 'Chr Upper Airway'
                WHEN n.diagnosis ILIKE '%oncol%' OR n.diagnosis ILIKE '%cancer%' THEN 'Oncology'
                WHEN n.diagnosis ILIKE '%hypertension%'                         THEN 'Hypertension'
                WHEN n.diagnosis ILIKE '%antenatal%'
                  OR n.diagnosis ILIKE '%anc%'                                  THEN 'Antenatal Care'
                ELSE NULL
            END
        ) AS cleaned_diagnosis_name,
        CASE WHEN a.visit_id IS NOT NULL THEN 1 ELSE 0 END AS is_ipd
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT  JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT  JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON v.id = n.visit_id AND v.source_schema = n.source_schema
    LEFT  JOIN admissions a ON v.id = a.visit_id AND v.source_schema = a.source_schema
    WHERE v.created_at >= {_OPD_IPD_FLOOR}
    {wh}
)
SELECT
    cleaned_diagnosis_name,
    COUNT(DISTINCT visit_id) AS TOTAL_OPD_CASES_WITH_THIS_DIAGNOSIS,
    SUM(is_ipd)              AS SUCCESSFUL_IPD_ADMISSIONS
FROM visit_dx
WHERE cleaned_diagnosis_name IS NOT NULL
GROUP BY 1
HAVING COUNT(DISTINCT visit_id) >= 5
ORDER BY SUM(is_ipd) DESC
"""
    return run_query(sql)


def load_opd_ipd_comorbidity(filters: dict, run_query) -> pd.DataFrame:
    """Section C: comorbidity conversion — aggregate + monthly encoded rows.
    Aggregate rows: patient_group in ('Single Diagnosis','Comorbid','Chronic Comorbid').
    Monthly rows: patient_group = 'Group — YYYY-MM-DD'.
    Columns: patient_group, total_opd_visits, conversion_rate_pct, delta_from_group_avg."""
    wh   = _w(filters)
    wh_a = _w_adm(filters)
    wsa  = _wsa(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
admissions AS (
    SELECT a.visit_id, REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    WHERE 1=1 {wh_a}
),
visit_base AS (
    SELECT v.source_schema, v.id AS visit_id,
        DATE_TRUNC('month', v.created_at) AS visit_month,
        CASE
            WHEN COALESCE(dx.is_chronic_1,0)=1 AND COALESCE(dx.is_chronic_2,0)=1
              AND COALESCE(NULLIF(TRIM(dx.disease_burden_group_2),''), NULL) IS NOT NULL
                THEN 'Chronic Comorbid'
            WHEN COALESCE(NULLIF(TRIM(dx.disease_burden_group_2),''), NULL) IS NOT NULL
                THEN 'Comorbid'
            ELSE 'Single Diagnosis'
        END AS patient_group,
        CASE WHEN a.visit_id IS NOT NULL THEN 1 ELSE 0 END AS is_ipd
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT  JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT  JOIN admissions a ON v.id = a.visit_id AND v.source_schema = a.source_schema
    WHERE v.created_at >= {_OPD_IPD_FLOOR}
    {wh}
),
overall_avg AS (
    SELECT ROUND(DIV0(SUM(is_ipd), COUNT(DISTINCT visit_id)) * 100, 2) AS avg_rate
    FROM visit_base
),
group_agg AS (
    SELECT patient_group,
        COUNT(DISTINCT visit_id) AS total_opd_visits,
        SUM(is_ipd) AS ipd_admissions,
        ROUND(DIV0(SUM(is_ipd), COUNT(DISTINCT visit_id)) * 100, 2) AS conversion_rate_pct
    FROM visit_base
    GROUP BY patient_group
)
SELECT
    g.patient_group                               AS PATIENT_GROUP,
    g.total_opd_visits                            AS TOTAL_OPD_VISITS,
    g.conversion_rate_pct                         AS CONVERSION_RATE_PCT,
    ROUND(g.conversion_rate_pct - o.avg_rate, 2) AS DELTA_FROM_GROUP_AVG
FROM group_agg g CROSS JOIN overall_avg o
UNION ALL
SELECT
    patient_group || ' — ' || TO_CHAR(visit_month, 'YYYY-MM-DD') AS PATIENT_GROUP,
    COUNT(DISTINCT visit_id)                                       AS TOTAL_OPD_VISITS,
    ROUND(DIV0(SUM(is_ipd), COUNT(DISTINCT visit_id)) * 100, 2)  AS CONVERSION_RATE_PCT,
    NULL::FLOAT                                                    AS DELTA_FROM_GROUP_AVG
FROM visit_base
GROUP BY patient_group, visit_month
ORDER BY 1
"""
    return run_query(sql)


def load_opd_ipd_retention(filters: dict, run_query) -> pd.DataFrame:
    """Section A & C: multi-dim retention universe table.
    Rows cover: RETENTION_UNIVERSE_TOTAL agg, per-segment agg, monthly per-segment,
    and CHRONIC by age group (visit_month=NULL, age_group set).
    Columns: acquisition_segment, visit_month, age_group,
             total_opd_visits, ipd_admissions, conversion_rate_pct."""
    wh   = _w(filters)
    wh_a = _w_adm(filters)
    wsa  = _wsa(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
{_NOTE_FLAGS_CTE},
admissions AS (
    SELECT a.visit_id, REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    WHERE 1=1 {wh_a}
),
visit_base AS (
    SELECT
        v.source_schema, v.id AS visit_id,
        DATE_TRUNC('month', v.created_at)  AS visit_month,
        {_SEGMENT_CASE}                    AS acquisition_segment,
        CASE
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 5   THEN 'Toddler (0-4)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 13  THEN 'Child (5-12)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 18  THEN 'Adolescent (13-17)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 25  THEN 'Youth (18-24)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 35  THEN 'Young Adult (25-34)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 45  THEN 'Adult (35-44)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 55  THEN 'Middle Age (45-54)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 65  THEN 'Older Adult (55-64)'
            ELSE 'Senior (65+)'
        END                                AS age_group,
        CASE WHEN a.visit_id IS NOT NULL THEN 1 ELSE 0 END AS is_ipd
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT  JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT  JOIN note_flags nf ON v.id = nf.visit_id AND v.source_schema = nf.source_schema
    LEFT  JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS rp
        ON v.patient = rp.patient_id AND v.source_schema = rp.source_schema
    LEFT  JOIN admissions a ON v.id = a.visit_id AND v.source_schema = a.source_schema
    WHERE v.created_at >= {_OPD_IPD_FLOOR}
    {wh}
),
retention_base AS (
    SELECT * FROM visit_base WHERE acquisition_segment IS NOT NULL
)
-- 1. RETENTION_UNIVERSE_TOTAL aggregate
SELECT
    'RETENTION_UNIVERSE_TOTAL' AS ACQUISITION_SEGMENT,
    NULL::DATE                 AS VISIT_MONTH,
    NULL::VARCHAR              AS AGE_GROUP,
    COUNT(DISTINCT visit_id)   AS TOTAL_OPD_VISITS,
    SUM(is_ipd)                AS IPD_ADMISSIONS,
    ROUND(DIV0(SUM(is_ipd), COUNT(DISTINCT visit_id)) * 100, 2) AS CONVERSION_RATE_PCT
FROM retention_base
UNION ALL
-- 2. Per-segment aggregates
SELECT
    acquisition_segment        AS ACQUISITION_SEGMENT,
    NULL::DATE                 AS VISIT_MONTH,
    NULL::VARCHAR              AS AGE_GROUP,
    COUNT(DISTINCT visit_id)   AS TOTAL_OPD_VISITS,
    SUM(is_ipd)                AS IPD_ADMISSIONS,
    ROUND(DIV0(SUM(is_ipd), COUNT(DISTINCT visit_id)) * 100, 2) AS CONVERSION_RATE_PCT
FROM retention_base
GROUP BY acquisition_segment
UNION ALL
-- 3. Monthly per-segment (age_group stays NULL)
SELECT
    acquisition_segment        AS ACQUISITION_SEGMENT,
    visit_month                AS VISIT_MONTH,
    NULL::VARCHAR              AS AGE_GROUP,
    COUNT(DISTINCT visit_id)   AS TOTAL_OPD_VISITS,
    SUM(is_ipd)                AS IPD_ADMISSIONS,
    ROUND(DIV0(SUM(is_ipd), COUNT(DISTINCT visit_id)) * 100, 2) AS CONVERSION_RATE_PCT
FROM retention_base
GROUP BY acquisition_segment, visit_month
UNION ALL
-- 4. CHRONIC by age group (visit_month stays NULL)
SELECT
    'Chronic'                  AS ACQUISITION_SEGMENT,
    NULL::DATE                 AS VISIT_MONTH,
    age_group                  AS AGE_GROUP,
    COUNT(DISTINCT visit_id)   AS TOTAL_OPD_VISITS,
    SUM(is_ipd)                AS IPD_ADMISSIONS,
    ROUND(DIV0(SUM(is_ipd), COUNT(DISTINCT visit_id)) * 100, 2) AS CONVERSION_RATE_PCT
FROM retention_base
WHERE acquisition_segment = 'Chronic' AND age_group IS NOT NULL
GROUP BY age_group
ORDER BY ACQUISITION_SEGMENT, VISIT_MONTH NULLS FIRST, AGE_GROUP NULLS FIRST
"""
    return run_query(sql)


def load_opd_ipd_escalation_72h(filters: dict, run_query) -> pd.DataFrame:
    """Section D: 72h OPD→IPD escalations by age group and diagnosis.
    Columns: age_group, clean_diagnosis, total_escalations."""
    wh   = _w(filters)
    wh_a = _w_adm(filters)
    wsa  = _wsa(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
admissions AS (
    SELECT a.visit_id, a.admitted_at,
           REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    WHERE 1=1 {wh_a}
),
escalations AS (
    SELECT v.id AS visit_id, v.source_schema,
        CASE
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 5   THEN 'Toddler (0-4)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 13  THEN 'Child (5-12)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 18  THEN 'Adolescent (13-17)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 25  THEN 'Youth (18-24)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 35  THEN 'Young Adult (25-34)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 45  THEN 'Adult (35-44)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 55  THEN 'Middle Age (45-54)'
            WHEN TIMESTAMPDIFF('year', rp.dob, sa.max_date) < 65  THEN 'Older Adult (55-64)'
            ELSE 'Senior (65+)'
        END AS age_group,
        IFF(
            COALESCE(
                NULLIF(TRIM(dx.disease_burden_group_1), ''),
                CASE
                    WHEN n.diagnosis ILIKE '%respiratory%'
                      OR n.diagnosis ILIKE '%pneumonia%'    THEN 'Respiratory'
                    WHEN n.diagnosis ILIKE '%malaria%'      THEN 'Malaria'
                    WHEN n.diagnosis ILIKE '%sepsis%'       THEN 'Sepsis'
                    WHEN n.diagnosis ILIKE '%trauma%'
                      OR n.diagnosis ILIKE '%injury%'       THEN 'Trauma'
                    ELSE 'Unknown'
                END
            ) = 'Communicable - Other Infectious',
            'Communicable Sepsis',
            COALESCE(
                NULLIF(TRIM(dx.disease_burden_group_1), ''),
                CASE
                    WHEN n.diagnosis ILIKE '%respiratory%'
                      OR n.diagnosis ILIKE '%pneumonia%'    THEN 'Respiratory'
                    WHEN n.diagnosis ILIKE '%malaria%'      THEN 'Malaria'
                    WHEN n.diagnosis ILIKE '%sepsis%'       THEN 'Sepsis'
                    WHEN n.diagnosis ILIKE '%trauma%'
                      OR n.diagnosis ILIKE '%injury%'       THEN 'Trauma'
                    ELSE 'Unknown'
                END
            )
        ) AS clean_diagnosis
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN admissions a
        ON v.id = a.visit_id AND v.source_schema = a.source_schema
        AND DATEDIFF('hour', v.created_at, a.admitted_at) BETWEEN 0 AND 72
    LEFT  JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS rp
        ON v.patient = rp.patient_id AND v.source_schema = rp.source_schema
    LEFT  JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT  JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON v.id = n.visit_id AND v.source_schema = n.source_schema
    WHERE v.created_at >= {_OPD_IPD_FLOOR}
    {wh}
)
SELECT
    age_group                AS AGE_GROUP,
    clean_diagnosis          AS CLEAN_DIAGNOSIS,
    COUNT(DISTINCT visit_id) AS TOTAL_ESCALATIONS
FROM escalations
GROUP BY age_group, clean_diagnosis
ORDER BY TOTAL_ESCALATIONS DESC
"""
    return run_query(sql)


# ══════════════════════════════════════════════════════════════════════════════
# CLINICAL ACTIVITY TAB
# ══════════════════════════════════════════════════════════════════════════════

def load_ca_ward_summary(filters: dict, run_query) -> pd.DataFrame:
    """A1: Ward summary — admissions, LOS, readmission rate, patient-request discharge %."""
    wh_a = _w_adm(filters)
    sql = f"""
SELECT
    LOWER(REPLACE(a.source_schema, '_CLEAN', ''))                              AS source_schema,
    ward_name,
    COUNT(original_id)                                                         AS total_admissions,
    COUNT(CASE WHEN discharged_at IS NOT NULL THEN 1 END)                      AS total_discharges,
    COUNT(CASE WHEN discharged_at IS NULL     THEN 1 END)                      AS still_admitted,
    ROUND(MEDIAN(los_days), 1)                                                 AS median_los,
    ROUND(AVG(los_days), 1)                                                    AS avg_los,
    COUNT(CASE WHEN is_30day_readmission = TRUE THEN 1 END)                    AS day_readmission_30,
    ROUND(DIV0(
        COUNT(CASE WHEN is_30day_readmission = TRUE THEN 1 END),
        COUNT(CASE WHEN discharged_at IS NOT NULL   THEN 1 END)
    ) * 100.0, 2)                                                              AS readmission_rate,
    COUNT(CASE WHEN discharge_type ILIKE '%Request' THEN 1 END)                AS patient_request_discharge,
    ROUND(DIV0(
        COUNT(CASE WHEN discharge_type ILIKE '%Request' THEN 1 END),
        COUNT(CASE WHEN discharged_at IS NOT NULL       THEN 1 END)
    ) * 100.0, 2)                                                              AS patient_request_discharge_pct
FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
WHERE 1=1
{wh_a}
GROUP BY ALL
ORDER BY total_admissions DESC
"""
    return run_query(sql)


def load_ca_admission_growth(filters: dict, run_query) -> pd.DataFrame:
    """A2: One row per admission — ward, month, disease burden group, LOS, readmission flag."""
    wh_a = _w_adm(filters)
    schemas = filters.get("source_schemas") or []
    dn_schema = (
        "AND source_schema IN (" + ", ".join(repr(s) for s in schemas) + ")"
        if schemas else ""
    )
    sql = f"""
WITH deduped_notes AS (
    SELECT visit_id, source_schema, TRIM(diagnosis) AS diagnosis
    FROM HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES
    WHERE 1=1 {dn_schema}
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY visit_id ORDER BY created_at ASC NULLS LAST
    ) = 1
)
SELECT
    LOWER(REPLACE(a.source_schema, '_CLEAN', ''))                              AS source_schema,
    a.visit_id,
    a.patient_id,
    a.admitted_at,
    DATE_TRUNC('month', a.admitted_at)                                         AS month,
    a.ward_name,
    a.ward_category,
    a.los_days,
    a.is_30day_readmission,
    a.discharge_type,
    ARRAY_TO_STRING(ARRAY_SORT(ARRAY_DISTINCT(
        ARRAY_CONSTRUCT_COMPACT(dp.icd10_code_1, dp.icd10_code_2, dp.icd10_code_3)
    )), '+')                                                                   AS icd10_code,
    ARRAY_TO_STRING(ARRAY_SORT(ARRAY_DISTINCT(
        ARRAY_CONSTRUCT_COMPACT(dp.icd10_name_1, dp.icd10_name_2, dp.icd10_name_3)
    )), '+')                                                                   AS icd10_name,
    ARRAY_TO_STRING(ARRAY_SORT(ARRAY_DISTINCT(
        ARRAY_CONSTRUCT_COMPACT(
            dp.disease_burden_group_1, dp.disease_burden_group_2, dp.disease_burden_group_3
        )
    )), '+')                                                                   AS disease_burden_group,
    dp.has_chronic_diagnosis,
    dp.has_comorbidity,
    dn.diagnosis                                                               AS notes_diagnosis,
    HOSPITALS.STAGING.MAP_DIAGNOSIS_TO_BURDEN_GROUP(dn.diagnosis)              AS notes_disease_burden_group
FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dp
    ON  a.visit_id = dp.visit_id
    AND LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = dp.source_schema
LEFT JOIN deduped_notes dn
    ON  a.visit_id = dn.visit_id
    AND LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = dn.source_schema
WHERE 1=1
{wh_a}
ORDER BY a.admitted_at DESC
"""
    return run_query(sql)


def load_ca_los_boxplot(filters: dict, run_query) -> pd.DataFrame:
    """B1: LOS distribution per ward per month — boxplot five-number summary."""
    wh_a = _w_adm(filters)
    sql = f"""
WITH stats_raw AS (
    SELECT
        LOWER(REPLACE(a.source_schema, '_CLEAN', ''))                          AS source_schema,
        DATE_TRUNC('month', admitted_at)                                       AS month,
        ward_name,
        a.los_days,
        MIN(a.los_days) OVER (PARTITION BY DATE_TRUNC('month', admitted_at), ward_name) AS min_los,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY a.los_days)
            OVER (PARTITION BY DATE_TRUNC('month', admitted_at), ward_name)    AS q1_los,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY a.los_days)
            OVER (PARTITION BY DATE_TRUNC('month', admitted_at), ward_name)    AS median_los,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY a.los_days)
            OVER (PARTITION BY DATE_TRUNC('month', admitted_at), ward_name)    AS q3_los,
        MAX(a.los_days) OVER (PARTITION BY DATE_TRUNC('month', admitted_at), ward_name) AS max_los
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    WHERE los_days IS NOT NULL
    {wh_a}
),
fences AS (
    SELECT
        source_schema, month, ward_name, los_days,
        min_los, q1_los, median_los, q3_los, max_los,
        (q3_los - q1_los)                   AS iqr,
        (q1_los - 1.5 * (q3_los - q1_los)) AS lower_fence,
        (q3_los + 1.5 * (q3_los - q1_los)) AS upper_fence
    FROM stats_raw
)
SELECT
    source_schema, month, ward_name,
    min_los, q1_los, median_los, q3_los, max_los, iqr, lower_fence, upper_fence,
    MIN(CASE WHEN los_days >= lower_fence THEN los_days END) AS lower_whisker,
    MAX(CASE WHEN los_days <= upper_fence THEN los_days END) AS upper_whisker
FROM fences
GROUP BY ALL
"""
    return run_query(sql)


def load_ca_los_outliers(filters: dict, run_query) -> pd.DataFrame:
    """B3/Q4: Individual admissions beyond the IQR upper fence, with diagnosis detail."""
    wh_a = _w_adm(filters)
    schemas = filters.get("source_schemas") or []
    dn_schema = (
        "AND source_schema IN (" + ", ".join(repr(s) for s in schemas) + ")"
        if schemas else ""
    )
    sql = f"""
WITH ward_fences AS (
    SELECT
        LOWER(REPLACE(a.source_schema, '_CLEAN', ''))                          AS source_schema,
        a.ward_name,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY a.los_days)              AS q1,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY a.los_days)              AS q3,
        (q3 - q1)             AS iqr,
        q3 + 1.5 * iqr        AS upper_fence
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    WHERE a.los_days IS NOT NULL
    {wh_a}
    GROUP BY ALL
),
deduped_notes AS (
    SELECT visit_id, source_schema, TRIM(diagnosis) AS diagnosis
    FROM HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES
    WHERE 1=1 {dn_schema}
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY visit_id ORDER BY created_at ASC NULLS LAST
    ) = 1
)
SELECT
    LOWER(REPLACE(a.source_schema, '_CLEAN', ''))                              AS source_schema,
    DATE_TRUNC('month', a.admitted_at)                                         AS month,
    a.ward_name,
    a.patient_id,
    a.admitted_at,
    a.discharged_at,
    a.los_days,
    ROUND(wf.upper_fence, 1)                                                   AS ward_upper_fence,
    ROUND(a.los_days - wf.upper_fence, 1)                                      AS days_above_fence,
    a.discharge_type,
    a.is_30day_readmission,
    ARRAY_TO_STRING(ARRAY_SORT(ARRAY_DISTINCT(
        ARRAY_CONSTRUCT_COMPACT(dp.icd10_name_1, dp.icd10_name_2, dp.icd10_name_3)
    )), '+')                                                                   AS icd10_name,
    COALESCE(dp.disease_burden_group_1, 'Unclassified')                        AS primary_burden_group,
    dn.diagnosis                                                               AS notes_diagnosis
FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
INNER JOIN ward_fences wf
    ON  LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = wf.source_schema
    AND a.ward_name  = wf.ward_name
    AND a.los_days   > wf.upper_fence
LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dp
    ON  a.visit_id = dp.visit_id
    AND LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = dp.source_schema
LEFT JOIN deduped_notes dn
    ON  a.visit_id = dn.visit_id
    AND LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = dn.source_schema
WHERE a.los_days IS NOT NULL
{wh_a}
ORDER BY a.los_days DESC
"""
    return run_query(sql)


def load_ca_maternity_outlier_notes(filters: dict, run_query) -> pd.DataFrame:
    """Lookup: all doctor notes for the longest General Maternity LOS admission."""
    wh_a = _w_adm(filters)
    sql = f"""
WITH outlier_case AS (
    SELECT
        a.visit_id,
        a.patient_id,
        LOWER(REPLACE(a.source_schema, '_CLEAN', ''))  AS source_schema,
        a.ward_name,
        a.admitted_at,
        a.discharged_at,
        a.los_days,
        a.discharge_type
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    WHERE LOWER(a.ward_name) LIKE '%general maternity%'
      AND a.los_days IS NOT NULL
    {wh_a}
    ORDER BY a.los_days DESC
    LIMIT 1
)
SELECT
    oc.visit_id,
    oc.patient_id,
    oc.ward_name,
    oc.admitted_at,
    oc.discharged_at,
    oc.los_days,
    oc.discharge_type,
    dn.created_at                                      AS note_time,
    TRIM(dn.diagnosis)                                 AS note_diagnosis
FROM outlier_case oc
LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES dn
    ON  oc.visit_id      = dn.visit_id
    AND oc.source_schema = dn.source_schema
ORDER BY dn.created_at ASC NULLS LAST
"""
    return run_query(sql)


def load_ca_readmission_layer1(filters: dict, run_query) -> pd.DataFrame:
    """C1: Readmission rate by ward and discharge type per month."""
    wh_a = _w_adm(filters)
    sql = f"""
SELECT
    LOWER(REPLACE(a.source_schema, '_CLEAN', ''))                              AS source_schema,
    DATE_TRUNC('month', admitted_at)                                           AS month,
    ward_name,
    discharge_type,
    COUNT(DISTINCT patient_id)                                                 AS total_unique_patients,
    COUNT(DISTINCT CASE WHEN is_30day_readmission = TRUE THEN patient_id END)  AS total_readmitted_unique_patients,
    COUNT(CASE WHEN is_30day_readmission = TRUE THEN 1 END)                    AS total_day_readmission_30,
    ROUND(DIV0(
        COUNT(CASE WHEN is_30day_readmission = TRUE THEN 1 END),
        COUNT(CASE WHEN discharged_at IS NOT NULL   THEN 1 END)
    ) * 100.0, 2)                                                              AS readmission_rate_pct
FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
WHERE 1=1
{wh_a}
GROUP BY ALL
"""
    return run_query(sql)


def load_ca_readmission_layer2(filters: dict, run_query) -> pd.DataFrame:
    """C2: Discharge type split for readmitted patients."""
    wh_a = _w_adm(filters)
    sql = f"""
SELECT
    LOWER(REPLACE(a.source_schema, '_CLEAN', ''))                              AS source_schema,
    discharge_type,
    COUNT(DISTINCT CASE WHEN is_30day_readmission = TRUE THEN patient_id END)  AS total_readmitted_unique_patients
FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
WHERE 1=1
{wh_a}
GROUP BY ALL
"""
    return run_query(sql)


def load_ca_readmission_layer3(filters: dict, run_query) -> pd.DataFrame:
    """C3: LOS at index admission vs at readmission visit, by ward."""
    wh_a = _w_adm(filters)
    sql = f"""
SELECT
    LOWER(REPLACE(a.source_schema, '_CLEAN', ''))                              AS source_schema,
    a.ward_name,
    a.is_30day_readmission,
    COUNT(*)                                                                   AS total_admissions,
    ROUND(MEDIAN(a.los_days), 1)                                               AS median_los,
    ROUND(AVG(a.los_days), 1)                                                  AS avg_los,
    COUNT(CASE WHEN a.los_days BETWEEN 0 AND 1 THEN 1 END)                     AS los_0_1_days,
    COUNT(CASE WHEN a.los_days BETWEEN 2 AND 3 THEN 1 END)                     AS los_2_3_days,
    COUNT(CASE WHEN a.los_days BETWEEN 4 AND 7 THEN 1 END)                     AS los_4_7_days,
    COUNT(CASE WHEN a.los_days > 7             THEN 1 END)                     AS los_over_7_days
FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
WHERE a.los_days IS NOT NULL
{wh_a}
GROUP BY ALL
ORDER BY a.ward_name, a.is_30day_readmission
"""
    return run_query(sql)


def load_ca_readmission_layer4(filters: dict, run_query) -> pd.DataFrame:
    """C4: Timing distribution — how many days after discharge patients return."""
    wh_a = _w_adm(filters)
    sql = f"""
SELECT
    LOWER(REPLACE(a.source_schema, '_CLEAN', ''))                              AS source_schema,
    a.ward_name,
    COUNT(*)                                                                   AS total_readmissions,
    COUNT(CASE WHEN a.days_since_last_admission BETWEEN 0 AND 7   THEN 1 END)  AS days_0_7,
    COUNT(CASE WHEN a.days_since_last_admission BETWEEN 8 AND 14  THEN 1 END)  AS days_8_14,
    COUNT(CASE WHEN a.days_since_last_admission BETWEEN 15 AND 30 THEN 1 END)  AS days_15_30,
    ROUND(MEDIAN(a.days_since_last_admission), 1)                              AS median_days_to_readmission,
    ROUND(AVG(a.days_since_last_admission), 1)                                 AS avg_days_to_readmission,
    MIN(a.days_since_last_admission)                                           AS min_days_to_readmission
FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
WHERE a.is_30day_readmission = TRUE
  AND a.days_since_last_admission >= 0
{wh_a}
GROUP BY ALL
ORDER BY a.ward_name
"""
    return run_query(sql)


def load_ca_readmission_layer4_detail(filters: dict, run_query) -> pd.DataFrame:
    """C4b: Discharge type x return band x ward for early/late returner analysis."""
    wh_a = _w_adm(filters)
    sql = f"""
SELECT
    LOWER(REPLACE(a.source_schema, '_CLEAN', ''))                              AS source_schema,
    a.ward_name,
    a.discharge_type,
    CASE
        WHEN a.days_since_last_admission BETWEEN 0  AND 7  THEN '0-7 days (early)'
        WHEN a.days_since_last_admission BETWEEN 8  AND 14 THEN '8-14 days'
        WHEN a.days_since_last_admission BETWEEN 15 AND 30 THEN '15-30 days (late)'
    END                                                                        AS return_band,
    COUNT(*)                                                                   AS readmissions,
    ROUND(MEDIAN(a.days_since_last_admission), 1)                              AS median_days_to_return,
    ROUND(MEDIAN(a.los_days), 1)                                               AS median_los_at_index
FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
WHERE a.is_30day_readmission = TRUE
  AND a.days_since_last_admission >= 0
  AND a.days_since_last_admission <= 30
{wh_a}
GROUP BY ALL
ORDER BY a.ward_name, return_band, readmissions DESC
"""
    return run_query(sql)


def load_ca_readmission_layer5(filters: dict, run_query) -> pd.DataFrame:
    """C5: TCA documentation rate and OPD follow-up before readmission."""
    wh_a = _w_adm(filters, alias="a")
    sql = f"""
WITH latest_discharge_request AS (
    SELECT
        dr.visit_id,
        dr.tca,
        dr.conditions,
        dr.created_at AS discharge_request_date
    FROM HOSPITALS.KISUMU_CLEAN.INPATIENT_DISCHARGE_REQUESTS dr
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY dr.visit_id ORDER BY dr.created_at DESC
    ) = 1
),
discharge_base AS (
    SELECT
        LOWER(REPLACE(a.source_schema, '_CLEAN', ''))                          AS source_schema,
        a.visit_id,
        a.patient_id,
        a.ward_name,
        a.ward_category,
        a.discharged_at,
        a.is_30day_readmission,
        a.discharge_type,
        IFF(
            dr.tca IS NOT NULL
            AND UPPER(TRIM(dr.tca)) NOT IN ('NOT NEEDED', 'NOT REQUIRED', 'N/A'),
            TRUE, FALSE
        )                                                                      AS tca_documented
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    LEFT JOIN latest_discharge_request dr ON a.visit_id = dr.visit_id
    WHERE a.discharged_at IS NOT NULL
    {wh_a}
),
followup_opd AS (
    SELECT DISTINCT
        a.patient_id,
        LOWER(REPLACE(a.source_schema, '_CLEAN', ''))                          AS source_schema,
        TRUE                                                                   AS had_followup_opd_visit
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON  a.patient_id = v.patient
        AND LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = v.source_schema
        AND v.created_at > DATEADD('day', -30, a.admitted_at)
        AND v.created_at < a.admitted_at
    WHERE a.is_30day_readmission = TRUE
    {wh_a}
)
SELECT
    db.source_schema,
    db.ward_name,
    db.ward_category,
    db.is_30day_readmission,
    db.discharge_type,
    COUNT(*)                                                                   AS total_admissions,
    SUM(IFF(db.tca_documented, 1, 0))                                          AS tca_documented_count,
    ROUND(DIV0(
        SUM(IFF(db.tca_documented, 1, 0)), COUNT(*)
    ) * 100.0, 2)                                                              AS tca_documented_pct,
    SUM(IFF(fo.had_followup_opd_visit = TRUE, 1, 0))                           AS had_followup_opd_count,
    ROUND(DIV0(
        SUM(IFF(fo.had_followup_opd_visit = TRUE, 1, 0)),
        NULLIF(COUNT(CASE WHEN db.is_30day_readmission = TRUE THEN 1 END), 0)
    ) * 100.0, 2)                                                              AS followup_opd_before_readmission_pct
FROM discharge_base db
LEFT JOIN followup_opd fo
    ON  db.patient_id    = fo.patient_id
    AND db.source_schema = fo.source_schema
GROUP BY ALL
ORDER BY db.ward_name, db.is_30day_readmission
"""
    return run_query(sql)


def load_ca_section_d(filters: dict, run_query) -> pd.DataFrame:
    """D: Diagnosis-driven readmission — readmission rate per diagnosis group per ward."""
    wh_a = _w_adm(filters)
    schemas = filters.get("source_schemas") or []
    dn_schema = (
        "AND source_schema IN (" + ", ".join(repr(s) for s in schemas) + ")"
        if schemas else ""
    )
    sql = f"""
WITH deduped_notes AS (
    SELECT visit_id, source_schema, TRIM(diagnosis) AS diagnosis
    FROM HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES
    WHERE 1=1 {dn_schema}
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY visit_id ORDER BY created_at ASC NULLS LAST
    ) = 1
),
base AS (
    SELECT
        LOWER(REPLACE(a.source_schema, '_CLEAN', ''))                          AS source_schema,
        DATE_TRUNC('month', a.admitted_at)                                     AS month,
        a.ward_name,
        a.patient_id,
        a.is_30day_readmission,
        a.discharged_at,
        a.los_days,
        ARRAY_TO_STRING(ARRAY_SORT(ARRAY_DISTINCT(
            ARRAY_CONSTRUCT_COMPACT(dp.icd10_code_1, dp.icd10_code_2, dp.icd10_code_3)
        )), '+')                                                               AS icd10_code,
        ARRAY_TO_STRING(ARRAY_SORT(ARRAY_DISTINCT(
            ARRAY_CONSTRUCT_COMPACT(dp.icd10_name_1, dp.icd10_name_2, dp.icd10_name_3)
        )), '+')                                                               AS icd10_name,
        COALESCE(
            NULLIF(TRIM(dp.disease_burden_group_1), ''),
            HOSPITALS.STAGING.MAP_DIAGNOSIS_TO_BURDEN_GROUP(dn.diagnosis),
            'Unclassified'
        )                                                                      AS final_disease_burden_group,
        dp.has_chronic_diagnosis,
        dp.has_comorbidity
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dp
        ON  a.visit_id = dp.visit_id
        AND LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = dp.source_schema
    LEFT JOIN deduped_notes dn
        ON  a.visit_id = dn.visit_id
        AND LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = dn.source_schema
    WHERE a.los_days IS NOT NULL
    {wh_a}
)
SELECT
    source_schema,
    month,
    ward_name,
    icd10_code,
    icd10_name,
    has_chronic_diagnosis,
    has_comorbidity,
    final_disease_burden_group,
    COUNT(DISTINCT patient_id)                                                 AS total_unique_patients,
    COUNT(*)                                                                   AS total_admissions,
    COUNT(CASE WHEN is_30day_readmission = TRUE THEN 1 END)                    AS readmissions_30d,
    COUNT(CASE WHEN discharged_at IS NOT NULL   THEN 1 END)                    AS total_discharges,
    ROUND(DIV0(
        COUNT(CASE WHEN is_30day_readmission = TRUE THEN 1 END),
        COUNT(CASE WHEN discharged_at IS NOT NULL   THEN 1 END)
    ) * 100.0, 2)                                                              AS readmission_rate_pct,
    ROUND(MEDIAN(los_days), 1)                                                 AS median_los,
    ROUND(AVG(los_days), 1)                                                    AS avg_los,
    CASE
        WHEN final_disease_burden_group ILIKE '%Cardiovascular%' THEN 'Priority - Cardiac'
        WHEN final_disease_burden_group ILIKE '%Respiratory%'    THEN 'Priority - Respiratory'
        WHEN final_disease_burden_group ILIKE '%Diabetes%'       THEN 'Priority - Diabetes'
        WHEN final_disease_burden_group ILIKE '%Renal%'          THEN 'Priority - Renal'
        WHEN final_disease_burden_group ILIKE '%Sepsis%'         THEN 'Priority - Sepsis'
        WHEN final_disease_burden_group ILIKE '%LRTI%'
          OR final_disease_burden_group ILIKE '%Pneumonia%'      THEN 'Priority - Pneumonia'
        ELSE 'Standard'
    END                                                                        AS international_priority_flag
FROM base
GROUP BY ALL
HAVING COUNT(*) >= 1
ORDER BY readmission_rate_pct DESC
"""
    return run_query(sql)


def load_ca_section_e(filters: dict, run_query) -> pd.DataFrame:
    """E1: Infection burden by ward and month — communicable disease share."""
    wh_a = _w_adm(filters)
    schemas = filters.get("source_schemas") or []
    dn_schema = (
        "AND source_schema IN (" + ", ".join(repr(s) for s in schemas) + ")"
        if schemas else ""
    )
    sql = f"""
WITH deduped_notes AS (
    SELECT visit_id, source_schema, TRIM(diagnosis) AS diagnosis
    FROM HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES
    WHERE 1=1 {dn_schema}
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY visit_id ORDER BY created_at ASC NULLS LAST
    ) = 1
),
base AS (
    SELECT
        LOWER(REPLACE(a.source_schema, '_CLEAN', ''))                          AS source_schema,
        DATE_TRUNC('month', a.admitted_at)                                     AS month,
        a.ward_name,
        a.ward_category,
        COALESCE(
            NULLIF(TRIM(dp.disease_burden_group_1), ''),
            HOSPITALS.STAGING.MAP_DIAGNOSIS_TO_BURDEN_GROUP(dn.diagnosis),
            'Unclassified'
        )                                                                      AS primary_burden_group
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dp
        ON  a.visit_id = dp.visit_id
        AND LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = dp.source_schema
    LEFT JOIN deduped_notes dn
        ON  a.visit_id = dn.visit_id
        AND LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = dn.source_schema
    WHERE 1=1
    {wh_a}
)
SELECT
    source_schema,
    month,
    ward_name,
    ward_category,
    COUNT(*)                                                                   AS total_admissions,
    COUNT(CASE WHEN primary_burden_group ILIKE '%Communicable%' THEN 1 END)    AS communicable_admissions,
    ROUND(DIV0(
        COUNT(CASE WHEN primary_burden_group ILIKE '%Communicable%' THEN 1 END),
        COUNT(*)
    ) * 100.0, 2)                                                              AS communicable_pct,
    COUNT(CASE WHEN primary_burden_group ILIKE '%Typhoid%'      THEN 1 END)    AS typhoid_admissions,
    COUNT(CASE WHEN primary_burden_group ILIKE '%Malaria%'      THEN 1 END)    AS malaria_admissions,
    COUNT(CASE WHEN primary_burden_group ILIKE '%Sepsis%'
              OR primary_burden_group ILIKE '%Other Infectious%' THEN 1 END)   AS sepsis_other_admissions,
    COUNT(CASE WHEN primary_burden_group ILIKE '%URTI%'
              OR primary_burden_group ILIKE '%LRTI%'
              OR primary_burden_group ILIKE '%Pneumonia%'        THEN 1 END)   AS respiratory_infection_admissions
FROM base
GROUP BY ALL
ORDER BY month DESC, communicable_admissions DESC
"""
    return run_query(sql)


def load_ca_typhoid(filters: dict, run_query) -> pd.DataFrame:
    """E2: Typhoid monthly trend with rolling average and spike flag, per ward."""
    wh_a = _w_adm(filters)
    sql = f"""
WITH monthly_typhoid AS (
    SELECT
        LOWER(REPLACE(a.source_schema, '_CLEAN', ''))                          AS source_schema,
        DATE_TRUNC('month', a.admitted_at)                                     AS month,
        a.ward_name,
        COUNT(*)                                                               AS typhoid_admissions
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dp
        ON  a.visit_id = dp.visit_id
        AND LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = dp.source_schema
    WHERE dp.disease_burden_group_1 ILIKE '%Typhoid%'
    {wh_a}
    GROUP BY ALL
)
SELECT
    source_schema, month, ward_name, typhoid_admissions,
    ROUND(AVG(typhoid_admissions) OVER (
        PARTITION BY source_schema, ward_name
        ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 2)                                                                      AS rolling_3m_avg,
    IFF(
        typhoid_admissions > (
            AVG(typhoid_admissions) OVER (
                PARTITION BY source_schema, ward_name
                ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
            ) + 1.5 * STDDEV(typhoid_admissions) OVER (
                PARTITION BY source_schema, ward_name
                ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
            )
        ), TRUE, FALSE
    )                                                                          AS is_spike_month
FROM monthly_typhoid
ORDER BY ward_name, month
"""
    return run_query(sql)


def load_ca_los_diagnosis(filters: dict, run_query) -> pd.DataFrame:
    """B2: LOS by diagnosis per ward — ward dropdown chart."""
    wh_a    = _w_adm(filters)
    schemas = filters.get("source_schemas") or []
    dn_schema = (
        "AND source_schema IN (" + ", ".join(repr(s) for s in schemas) + ")"
        if schemas else ""
    )
    sql = f"""
WITH deduped_notes AS (
    SELECT visit_id, source_schema, TRIM(diagnosis) AS diagnosis
    FROM HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES
    WHERE 1=1 {dn_schema}
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY visit_id ORDER BY created_at ASC NULLS LAST
    ) = 1
)
SELECT
    LOWER(REPLACE(a.source_schema, '_CLEAN', ''))                              AS source_schema,
    a.ward_name,
    COALESCE(
        NULLIF(TRIM(dp.disease_burden_group_1), ''),
        'Unclassified'
    )                                                                          AS final_disease_burden_group,
    ROUND(AVG(a.los_days), 1)                                                  AS average_los_days,
    ROUND(MEDIAN(a.los_days), 1)                                               AS median_los_days
FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dp
    ON  a.visit_id = dp.visit_id
    AND LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = dp.source_schema
WHERE a.los_days IS NOT NULL
{wh_a}
GROUP BY ALL
HAVING COUNT(*) >= 3
ORDER BY a.ward_name, median_los_days DESC
"""
    return run_query(sql)


def load_ca_general_male(filters: dict, run_query) -> pd.DataFrame:
    """Q1: General Male ward readmission patient profile — age band and diagnosis."""
    wh_a    = _w_adm(filters)
    schemas = filters.get("source_schemas") or []
    dn_schema = (
        "AND source_schema IN (" + ", ".join(repr(s) for s in schemas) + ")"
        if schemas else ""
    )
    sql = f"""
SELECT
    LOWER(REPLACE(a.source_schema, '_CLEAN', ''))                              AS source_schema,
    a.patient_id,
    a.sex,
    a.dob,
    a.ward_name,
    a.admitted_at,
    a.discharged_at,
    a.discharge_type,
    a.los_days,
    a.days_since_last_admission,
    a.admission_number,
    a.is_30day_readmission,
    CASE
        WHEN DATEDIFF('year', a.dob, a.admitted_at) < 5   THEN 'Toddler (0-4)'
        WHEN DATEDIFF('year', a.dob, a.admitted_at) < 13  THEN 'Child (5-12)'
        WHEN DATEDIFF('year', a.dob, a.admitted_at) < 18  THEN 'Adolescent (13-17)'
        WHEN DATEDIFF('year', a.dob, a.admitted_at) < 25  THEN 'Youth (18-24)'
        WHEN DATEDIFF('year', a.dob, a.admitted_at) < 35  THEN 'Young Adult (25-34)'
        WHEN DATEDIFF('year', a.dob, a.admitted_at) < 45  THEN 'Adult (35-44)'
        WHEN DATEDIFF('year', a.dob, a.admitted_at) < 55  THEN 'Middle Age (45-54)'
        WHEN DATEDIFF('year', a.dob, a.admitted_at) < 65  THEN 'Older Adult (55-64)'
        ELSE 'Senior (65+)'
    END                                                                        AS age_band,
    COALESCE(
        NULLIF(TRIM(dp.disease_burden_group_1), ''),
        'Unclassified'
    )                                                                          AS final_disease_burden_group,
    dp.has_chronic_diagnosis,
    dp.has_comorbidity
FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dp
    ON  a.visit_id = dp.visit_id
    AND LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = dp.source_schema
WHERE LOWER(a.ward_name) = 'general male'
  AND a.is_30day_readmission = TRUE
{wh_a}
ORDER BY a.admitted_at DESC
"""
    return run_query(sql)


def load_ca_layer4_conditions(filters: dict, run_query) -> pd.DataFrame:
    """C4-conditions: Patient-level readmission profile for 3 high-volume wards."""
    wh_a    = _w_adm(filters)
    schemas = filters.get("source_schemas") or []
    dn_schema = (
        "AND source_schema IN (" + ", ".join(repr(s) for s in schemas) + ")"
        if schemas else ""
    )
    sql = f"""
WITH deduped_notes AS (
    SELECT visit_id, source_schema, TRIM(diagnosis) AS diagnosis
    FROM HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES
    WHERE 1=1 {dn_schema}
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY visit_id ORDER BY created_at ASC NULLS LAST
    ) = 1
)
SELECT
    LOWER(REPLACE(a.source_schema, '_CLEAN', ''))           AS source_schema,
    a.patient_id,
    a.sex,
    a.dob,
    a.ward_name,
    a.admitted_at,
    a.discharged_at,
    a.discharge_type,
    a.los_days,
    a.days_since_last_admission,
    a.admission_number,
    a.is_30day_readmission,
    CASE
        WHEN a.days_since_last_admission BETWEEN 0  AND 7  THEN '0–7d (early)'
        WHEN a.days_since_last_admission BETWEEN 8  AND 14 THEN '8–14d'
        WHEN a.days_since_last_admission BETWEEN 15 AND 30 THEN '15–30d (late)'
    END                                                     AS return_band,
    CASE
        WHEN DATEDIFF('year', a.dob, a.admitted_at) < 5   THEN 'Toddler (0-4)'
        WHEN DATEDIFF('year', a.dob, a.admitted_at) < 13  THEN 'Child (5-12)'
        WHEN DATEDIFF('year', a.dob, a.admitted_at) < 18  THEN 'Adolescent (13-17)'
        WHEN DATEDIFF('year', a.dob, a.admitted_at) < 25  THEN 'Youth (18-24)'
        WHEN DATEDIFF('year', a.dob, a.admitted_at) < 35  THEN 'Young Adult (25-34)'
        WHEN DATEDIFF('year', a.dob, a.admitted_at) < 45  THEN 'Adult (35-44)'
        WHEN DATEDIFF('year', a.dob, a.admitted_at) < 55  THEN 'Middle Age (45-54)'
        WHEN DATEDIFF('year', a.dob, a.admitted_at) < 65  THEN 'Older Adult (55-64)'
        ELSE 'Senior (65+)'
    END                                                     AS age_band,
    ARRAY_TO_STRING(ARRAY_SORT(ARRAY_DISTINCT(
        ARRAY_CONSTRUCT_COMPACT(dp.icd10_code_1, dp.icd10_code_2, dp.icd10_code_3)
    )), '+')                                                AS icd10_code,
    ARRAY_TO_STRING(ARRAY_SORT(ARRAY_DISTINCT(
        ARRAY_CONSTRUCT_COMPACT(dp.icd10_name_1, dp.icd10_name_2, dp.icd10_name_3)
    )), '+')                                                AS icd10_name,
    ARRAY_TO_STRING(ARRAY_SORT(ARRAY_DISTINCT(
        ARRAY_CONSTRUCT_COMPACT(
            dp.disease_burden_group_1,
            dp.disease_burden_group_2,
            dp.disease_burden_group_3
        )
    )), '+')                                                AS clean_disease_burden_group,
    dn.diagnosis                                            AS notes_diagnosis,
    HOSPITALS.STAGING.MAP_DIAGNOSIS_TO_BURDEN_GROUP(dn.diagnosis)
                                                            AS notes_disease_burden_group,
    ARRAY_TO_STRING(ARRAY_SORT(ARRAY_DISTINCT(
        ARRAY_CONSTRUCT_COMPACT(clean_disease_burden_group, notes_disease_burden_group)
    )), '+')                                                AS final_disease_burden_group,
    dp.has_chronic_diagnosis,
    dp.has_comorbidity
FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dp
    ON  a.visit_id = dp.visit_id
    AND LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = dp.source_schema
LEFT JOIN deduped_notes dn
    ON  a.visit_id = dn.visit_id
    AND LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = dn.source_schema
WHERE a.ward_name IN ('General Male', 'General Female', 'Pediatric General')
  AND a.is_30day_readmission = TRUE
  AND a.days_since_last_admission >= 0
{wh_a}
ORDER BY a.ward_name, a.days_since_last_admission ASC
"""
    return run_query(sql)


def _ltfu_cohort_ctes(wh_a: str = "") -> str:
    """Shared CTEs for the four care-pathway signal queries."""
    return f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS
    WHERE source_schema = 'kisumu'
    GROUP BY source_schema
),
patient_visits AS (
    SELECT
        v.source_schema,
        v.patient                                         AS patient_id,
        DATEDIFF('day', MAX(v.created_at), sa.max_date)   AS days_since_last_visit
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= DATEADD('year', -1, sa.max_date)
    GROUP BY v.source_schema, v.patient, sa.max_date
),
last_visit_ids AS (
    SELECT
        v.source_schema,
        v.patient    AS patient_id,
        v.id         AS last_visit_id,
        v.created_at AS last_visit_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= DATEADD('year', -1, sa.max_date)
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY v.source_schema, v.patient
        ORDER BY v.created_at DESC
    ) = 1
),
chronic_patients AS (
    SELECT DISTINCT v.source_schema, v.patient AS patient_id
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON v.id = n.visit_id AND v.source_schema = n.source_schema
    WHERE v.created_at >= DATEADD('year', -1, sa.max_date)
      AND (
            COALESCE(dx.is_chronic_1, 0) = 1
         OR COALESCE(dx.is_chronic_2, 0) = 1
         OR n.diagnosis ILIKE '%cardiovascular%'
         OR n.diagnosis ILIKE '%diabetes%'
         OR n.diagnosis ILIKE '%hiv%'
         OR n.diagnosis ILIKE '%neurolog%'
         OR n.diagnosis ILIKE '%chronic%'
         OR n.diagnosis ILIKE '%renal%'
         OR n.diagnosis ILIKE '%respiratory%'
         OR n.diagnosis ILIKE '%oncolog%'
         OR n.diagnosis ILIKE '%mental%'
      )
),
ltfu_cohort AS (
    SELECT pv.source_schema, pv.patient_id, lvi.last_visit_id, lvi.last_visit_date
    FROM patient_visits pv
    INNER JOIN chronic_patients cp
        ON pv.patient_id = cp.patient_id AND pv.source_schema = cp.source_schema
    INNER JOIN last_visit_ids lvi
        ON pv.patient_id = lvi.patient_id AND pv.source_schema = lvi.source_schema
    WHERE pv.days_since_last_visit > 180
)"""


def load_care_pathway_lab(filters: dict, run_query) -> pd.DataFrame:
    """Care Pathway Signal 1: % of chronic LTFU patients who had a lab test at their last visit."""
    sql = _ltfu_cohort_ctes() + """
,
lab_flag AS (
    SELECT DISTINCT i.visit_id
    FROM HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS i
    INNER JOIN ltfu_cohort c ON i.visit_id = c.last_visit_id
    WHERE i.source_schema = 'kisumu'
      AND i.procedure_clinical_division = 'Pathology / Laboratory Medicine'
      AND i.investigation_deleted_at IS NULL
      AND i.cancelled = 0
)
SELECT
    COUNT(DISTINCT c.patient_id)                                               AS total_ltfu_patients,
    COUNT(DISTINCT CASE WHEN l.visit_id IS NOT NULL THEN c.patient_id END)    AS patients_with_lab,
    ROUND(
        100.0 * COUNT(DISTINCT CASE WHEN l.visit_id IS NOT NULL THEN c.patient_id END)
              / NULLIF(COUNT(DISTINCT c.patient_id), 0), 1
    )                                                                          AS pct_had_lab
FROM ltfu_cohort c
LEFT JOIN lab_flag l ON l.visit_id = c.last_visit_id
"""
    return run_query(sql)


def load_care_pathway_rx(filters: dict, run_query) -> pd.DataFrame:
    """Care Pathway Signal 2: % of chronic LTFU patients who received a prescription at their last visit."""
    sql = _ltfu_cohort_ctes() + """
,
rx_flag AS (
    SELECT DISTINCT p.visit_id
    FROM HOSPITALS.STAGING.STG_PRESCRIPTION_PAYMENTS p
    INNER JOIN ltfu_cohort c ON p.visit_id = c.last_visit_id
    WHERE p.source_schema = 'kisumu'
      AND (p.stopped   = 0 OR p.stopped   IS NULL)
      AND (p.canceled  = 0 OR p.canceled  IS NULL)
)
SELECT
    COUNT(DISTINCT c.patient_id)                                               AS total_ltfu_patients,
    COUNT(DISTINCT CASE WHEN r.visit_id IS NOT NULL THEN c.patient_id END)    AS patients_with_rx,
    ROUND(
        100.0 * COUNT(DISTINCT CASE WHEN r.visit_id IS NOT NULL THEN c.patient_id END)
              / NULLIF(COUNT(DISTINCT c.patient_id), 0), 1
    )                                                                          AS pct_had_rx
FROM ltfu_cohort c
LEFT JOIN rx_flag r ON r.visit_id = c.last_visit_id
"""
    return run_query(sql)


def load_care_pathway_followup(filters: dict, run_query) -> pd.DataFrame:
    """Care Pathway Signal 3: Follow-up documentation gap and return rate for chronic LTFU patients."""
    sql = _ltfu_cohort_ctes() + """
,
followup_notes AS (
    SELECT
        n.visit_id,
        CASE
            WHEN n.next_visit_date IS NOT NULL
             AND TRIM(CAST(n.next_visit_date AS VARCHAR)) != ''
            THEN TRUE ELSE FALSE
        END AS has_structured_date,
        CASE
            WHEN n.treatment_plan ILIKE '%follow%'
              OR n.next_steps     ILIKE '%follow%'
              OR n.diagnosis      ILIKE '%follow%'
            THEN TRUE ELSE FALSE
        END AS has_followup_freetext
    FROM HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
    INNER JOIN ltfu_cohort c ON n.visit_id = c.last_visit_id
    WHERE n.source_schema = 'kisumu'
),
return_visits AS (
    SELECT DISTINCT c.patient_id, TRUE AS did_return
    FROM ltfu_cohort c
    INNER JOIN followup_notes fn
        ON fn.visit_id = c.last_visit_id AND fn.has_followup_freetext = TRUE
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON v.patient      = c.patient_id
       AND v.source_schema = c.source_schema
       AND v.created_at   > c.last_visit_date
)
SELECT
    COUNT(DISTINCT c.patient_id)                                                        AS total_ltfu_patients,
    COUNT(DISTINCT CASE WHEN COALESCE(fn.has_structured_date,    FALSE) = FALSE
                        THEN c.patient_id END)                                          AS patients_no_structured_date,
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN COALESCE(fn.has_structured_date, FALSE) = FALSE
                                      THEN c.patient_id END)
                / NULLIF(COUNT(DISTINCT c.patient_id), 0), 1)                          AS pct_no_structured_date,
    COUNT(DISTINCT CASE WHEN COALESCE(fn.has_followup_freetext, FALSE) = TRUE
                        THEN c.patient_id END)                                          AS patients_followup_mentioned,
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN COALESCE(fn.has_followup_freetext, FALSE) = TRUE
                                      THEN c.patient_id END)
                / NULLIF(COUNT(DISTINCT c.patient_id), 0), 1)                          AS pct_followup_mentioned,
    COUNT(DISTINCT CASE WHEN COALESCE(fn.has_followup_freetext, FALSE) = TRUE
                         AND rv.did_return = TRUE
                        THEN c.patient_id END)                                          AS patients_followup_mentioned_and_returned,
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN COALESCE(fn.has_followup_freetext, FALSE) = TRUE
                                       AND rv.did_return = TRUE
                                      THEN c.patient_id END)
                / NULLIF(COUNT(DISTINCT CASE WHEN COALESCE(fn.has_followup_freetext, FALSE) = TRUE
                                            THEN c.patient_id END), 0), 1)             AS pct_returned_after_followup_note,
    COUNT(DISTINCT CASE WHEN COALESCE(fn.has_followup_freetext, FALSE) = TRUE
                         AND rv.did_return IS NULL
                        THEN c.patient_id END)                                          AS patients_followup_mentioned_not_returned,
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN COALESCE(fn.has_followup_freetext, FALSE) = TRUE
                                       AND rv.did_return IS NULL
                                      THEN c.patient_id END)
                / NULLIF(COUNT(DISTINCT CASE WHEN COALESCE(fn.has_followup_freetext, FALSE) = TRUE
                                            THEN c.patient_id END), 0), 1)             AS pct_not_returned_despite_followup_note
FROM ltfu_cohort c
LEFT JOIN followup_notes fn ON fn.visit_id = c.last_visit_id
LEFT JOIN return_visits rv  ON rv.patient_id = c.patient_id
"""
    return run_query(sql)


def load_care_pathway_radiology(filters: dict, run_query) -> pd.DataFrame:
    """Care Pathway Signal 4: % of chronic LTFU patients who had radiology at their last visit."""
    sql = _ltfu_cohort_ctes() + """
,
radiology_flag AS (
    SELECT DISTINCT i.visit_id
    FROM HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS i
    INNER JOIN ltfu_cohort c ON i.visit_id = c.last_visit_id
    WHERE i.source_schema = 'kisumu'
      AND (
            i.procedure_clinical_division ILIKE 'Radiology%'
         OR i.procedure_discipline        ILIKE 'Radiology%'
         OR i.procedure_discipline        ILIKE 'Imaging%'
      )
      AND i.investigation_deleted_at IS NULL
      AND i.cancelled = 0
)
SELECT
    COUNT(DISTINCT c.patient_id)                                               AS total_ltfu_patients,
    COUNT(DISTINCT CASE WHEN r.visit_id IS NOT NULL THEN c.patient_id END)    AS patients_with_radiology,
    ROUND(
        100.0 * COUNT(DISTINCT CASE WHEN r.visit_id IS NOT NULL THEN c.patient_id END)
              / NULLIF(COUNT(DISTINCT c.patient_id), 0), 1
    )                                                                          AS pct_had_radiology
FROM ltfu_cohort c
LEFT JOIN radiology_flag r ON r.visit_id = c.last_visit_id
"""
    return run_query(sql)


def load_retention_overview(filters: dict, run_query) -> pd.DataFrame:
    """Flow & Retention: header KPIs — chronic patients, active/lapsing/LTFU counts and %."""
    wh = _w(filters)
    wsa = _wsa(filters)
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
    WHERE v.created_at >= DATEADD('year', -1, sa.max_date)
      {wh}
      AND (dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1
           OR n.diagnosis ILIKE '%hypertension%'
           OR n.diagnosis ILIKE '%diabetes%'
           OR n.diagnosis ILIKE '%hiv%')
),
patient_status AS (
    SELECT cp.source_schema, cp.patient,
        DATEDIFF('day', MAX(v.created_at), sa.max_date) AS days_gap
    FROM chronic_patients cp
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON cp.patient = v.patient AND cp.source_schema = v.source_schema
    INNER JOIN schema_anchor sa ON cp.source_schema = sa.source_schema
    GROUP BY cp.source_schema, cp.patient, sa.max_date
)
SELECT
    COUNT(DISTINCT patient)                                                         AS chronic_patients,
    COUNT(DISTINCT CASE WHEN days_gap <= 90              THEN patient END)          AS active_count,
    COUNT(DISTINCT CASE WHEN days_gap BETWEEN 91 AND 180 THEN patient END)          AS lapsing_count,
    COUNT(DISTINCT CASE WHEN days_gap > 180              THEN patient END)          AS ltfu_count,
    ROUND(DIV0(COUNT(DISTINCT CASE WHEN days_gap <= 90              THEN patient END), COUNT(DISTINCT patient)) * 100, 1) AS active_pct,
    ROUND(DIV0(COUNT(DISTINCT CASE WHEN days_gap BETWEEN 91 AND 180 THEN patient END), COUNT(DISTINCT patient)) * 100, 1) AS lapsing_pct,
    ROUND(DIV0(COUNT(DISTINCT CASE WHEN days_gap > 180              THEN patient END), COUNT(DISTINCT patient)) * 100, 1) AS ltfu_pct
FROM patient_status
"""
    return run_query(sql)


def load_retention_trend(filters: dict, run_query) -> pd.DataFrame:
    """Flow & Retention: monthly chronic patient visits split by current lifecycle stage."""
    wh = _w(filters)
    wsa = _wsa(filters)
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
    WHERE v.created_at >= DATEADD('year', -1, sa.max_date)
      {wh}
      AND (dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1
           OR n.diagnosis ILIKE '%hypertension%'
           OR n.diagnosis ILIKE '%diabetes%'
           OR n.diagnosis ILIKE '%hiv%')
),
patient_lifecycle AS (
    SELECT cp.patient, cp.source_schema,
        CASE
            WHEN DATEDIFF('day', MAX(v.created_at), sa.max_date) <= 90  THEN 'Active'
            WHEN DATEDIFF('day', MAX(v.created_at), sa.max_date) <= 180 THEN 'Lapsing'
            ELSE 'LTFU'
        END AS lifecycle
    FROM chronic_patients cp
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON cp.patient = v.patient AND cp.source_schema = v.source_schema
    INNER JOIN schema_anchor sa ON cp.source_schema = sa.source_schema
    GROUP BY cp.patient, cp.source_schema, sa.max_date
),
monthly AS (
    SELECT DATE_TRUNC('month', v.created_at) AS visit_month, v.patient, v.source_schema
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= DATEADD('year', -1, sa.max_date)
    {wh}
)
SELECT
    m.visit_month,
    COUNT(DISTINCT CASE WHEN pl.lifecycle = 'Active'  THEN m.patient END) AS active_count,
    COUNT(DISTINCT CASE WHEN pl.lifecycle = 'Lapsing' THEN m.patient END) AS lapsing_count,
    COUNT(DISTINCT CASE WHEN pl.lifecycle = 'LTFU'    THEN m.patient END) AS ltfu_count
FROM monthly m
INNER JOIN patient_lifecycle pl ON m.patient = pl.patient AND m.source_schema = pl.source_schema
GROUP BY m.visit_month
ORDER BY m.visit_month
"""
    return run_query(sql)


def load_lapsing_cohort(filters: dict, run_query) -> pd.DataFrame:
    """Flow & Retention Section C: lapsing count, recoverable revenue, cash %."""
    wh = _w(filters)
    wsa = _wsa(filters)
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
    WHERE v.created_at >= DATEADD('year', -1, sa.max_date)
      {wh}
      AND (dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1
           OR n.diagnosis ILIKE '%hypertension%'
           OR n.diagnosis ILIKE '%diabetes%'
           OR n.diagnosis ILIKE '%hiv%')
),
lapsing AS (
    SELECT cp.patient, cp.source_schema,
        MAX(CASE
            WHEN UPPER(v.payment_mode) IN ('CASH','PRIVATE') THEN 'Cash'
            WHEN UPPER(v.payment_mode) LIKE '%NHIF%' OR UPPER(v.payment_mode) LIKE '%SHA%' THEN 'NHIF / SHA'
            ELSE 'Insurance / Corporate'
        END) AS payer
    FROM chronic_patients cp
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON cp.patient = v.patient AND cp.source_schema = v.source_schema
    INNER JOIN schema_anchor sa ON cp.source_schema = sa.source_schema
    GROUP BY cp.patient, cp.source_schema, sa.max_date
    HAVING DATEDIFF('day', MAX(v.created_at), sa.max_date) BETWEEN 91 AND 180
),
avg_rev AS (
    SELECT v.source_schema, ROUND(AVG(ili.item_amount), 0) AS avg_rev_per_visit
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_INVOICE_LINE_ITEMS ili
        ON v.id = ili.visit_id AND v.source_schema = ili.source_schema
    WHERE ili.invoice_deleted_at IS NULL AND (ili.auto_cancelled IS NULL OR ili.auto_cancelled = 0)
      AND v.created_at >= DATEADD('year', -1, sa.max_date)
    GROUP BY v.source_schema
)
SELECT
    COUNT(DISTINCT l.patient)                                                        AS total_lapsing,
    ROUND(COUNT(DISTINCT l.patient) * COALESCE(ar.avg_rev_per_visit, 0) * 4, 0)    AS recoverable_revenue_kes,
    ROUND(DIV0(COUNT(DISTINCT CASE WHEN l.payer = 'Cash' THEN l.patient END),
               COUNT(DISTINCT l.patient)) * 100, 1)                                  AS cash_pct,
    ROUND(DIV0(COUNT(DISTINCT CASE WHEN l.payer != 'Cash' THEN l.patient END),
               COUNT(DISTINCT l.patient)) * 100, 1)                                  AS insurance_pct
FROM lapsing l
LEFT JOIN avg_rev ar ON l.source_schema = ar.source_schema
GROUP BY ar.avg_rev_per_visit
"""
    return run_query(sql)


def load_visit_tier(filters: dict, run_query) -> pd.DataFrame:
    """Flow & Retention Section D: LTFU chronic patients bucketed by total visit count."""
    wsa = _wsa(filters)
    wh = _w(filters)
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
    WHERE v.created_at >= DATEADD('year', -1, sa.max_date)
      {wh}
      AND (dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1
           OR n.diagnosis ILIKE '%hypertension%'
           OR n.diagnosis ILIKE '%diabetes%'
           OR n.diagnosis ILIKE '%hiv%')
),
patient_stats AS (
    SELECT cp.patient,
        COUNT(DISTINCT v.id) AS visit_count,
        DATEDIFF('day', MAX(v.created_at), sa.max_date) AS days_gap
    FROM chronic_patients cp
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON cp.patient = v.patient AND cp.source_schema = v.source_schema
    INNER JOIN schema_anchor sa ON cp.source_schema = sa.source_schema
    GROUP BY cp.patient, sa.max_date
    HAVING days_gap > 180
),
tiered AS (
    SELECT
        CASE
            WHEN visit_count <= 2 THEN '1-2 visits'
            WHEN visit_count <= 5 THEN '3-5 visits'
            ELSE '5+ visits'
        END AS visit_tier,
        patient
    FROM patient_stats
)
SELECT
    visit_tier,
    COUNT(DISTINCT patient)                                         AS patient_count,
    ROUND(DIV0(COUNT(DISTINCT patient),
               SUM(COUNT(DISTINCT patient)) OVER ()) * 100, 1)    AS share_pct
FROM tiered
GROUP BY visit_tier
ORDER BY CASE visit_tier WHEN '1-2 visits' THEN 1 WHEN '3-5 visits' THEN 2 ELSE 3 END
"""
    return run_query(sql)


def load_dropout_profile(filters: dict, run_query) -> pd.DataFrame:
    """Flow & Retention Section D: 1-2 visit chronic LTFU patients by age group and diagnosis."""
    wsa = _wsa(filters)
    wh = _w(filters)
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
    WHERE v.created_at >= DATEADD('year', -1, sa.max_date)
      {wh}
      AND (dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1
           OR n.diagnosis ILIKE '%hypertension%'
           OR n.diagnosis ILIKE '%diabetes%'
           OR n.diagnosis ILIKE '%hiv%')
),
ltfu_early AS (
    SELECT cp.patient, cp.source_schema,
        COUNT(DISTINCT v.id) AS visit_count,
        MAX(v.created_at)    AS last_visit
    FROM chronic_patients cp
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON cp.patient = v.patient AND cp.source_schema = v.source_schema
    INNER JOIN schema_anchor sa ON cp.source_schema = sa.source_schema
    GROUP BY cp.patient, cp.source_schema, sa.max_date
    HAVING visit_count <= 2 AND DATEDIFF('day', MAX(v.created_at), sa.max_date) > 180
),
by_age AS (
    SELECT 'Age group' AS dimension,
        CASE
            WHEN TIMESTAMPDIFF('year', rp.dob, le.last_visit) < 18 THEN 'Under 18'
            WHEN TIMESTAMPDIFF('year', rp.dob, le.last_visit) < 25 THEN 'Youth (18-24)'
            WHEN TIMESTAMPDIFF('year', rp.dob, le.last_visit) < 35 THEN 'Young Adult (25-34)'
            WHEN TIMESTAMPDIFF('year', rp.dob, le.last_visit) < 45 THEN 'Adult (35-44)'
            WHEN TIMESTAMPDIFF('year', rp.dob, le.last_visit) < 55 THEN 'Middle Age (45-54)'
            WHEN TIMESTAMPDIFF('year', rp.dob, le.last_visit) < 65 THEN 'Older Adult (55-64)'
            ELSE 'Senior (65+)'
        END AS category,
        COUNT(DISTINCT le.patient) AS patient_count
    FROM ltfu_early le
    LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS rp
        ON le.patient = rp.patient_id AND le.source_schema = rp.source_schema
    GROUP BY category
),
by_dx AS (
    SELECT 'Diagnosis' AS dimension,
        COALESCE(dx.disease_burden_group_1, 'Unclassified') AS category,
        COUNT(DISTINCT le.patient) AS patient_count
    FROM ltfu_early le
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON le.patient = v.patient AND le.source_schema = v.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    GROUP BY category
)
SELECT * FROM by_age WHERE category IS NOT NULL
UNION ALL
SELECT * FROM by_dx WHERE category != 'Unclassified'
ORDER BY dimension, patient_count DESC
"""
    return run_query(sql)


def load_care_pathway(filters: dict, run_query) -> pd.DataFrame:
    """Flow & Retention Section E: unified care pathway signals for chronic LTFU patients.
    Returns one row per signal: SIGNAL, PCT, PATIENT_COUNT, TOTAL_PATIENTS."""
    sql = _ltfu_cohort_ctes() + """
,
total AS (SELECT COUNT(DISTINCT patient_id) AS n FROM ltfu_cohort),
lab_flag AS (
    SELECT DISTINCT i.visit_id FROM HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS i
    INNER JOIN ltfu_cohort c ON i.visit_id = c.last_visit_id AND i.source_schema = c.source_schema
    WHERE i.procedure_clinical_division = 'Pathology / Laboratory Medicine'
      AND (i.cancelled IS NULL OR i.cancelled = 0)
      AND (i.remove_from_report IS NULL OR i.remove_from_report = 0)
),
rx_flag AS (
    SELECT DISTINCT p.visit_id FROM HOSPITALS.STAGING.STG_PRESCRIPTION_PAYMENTS p
    INNER JOIN ltfu_cohort c ON p.visit_id = c.last_visit_id AND p.source_schema = c.source_schema
    WHERE (p.stopped IS NULL OR p.stopped = 0)
      AND (p.canceled IS NULL OR p.canceled = 0)
),
rad_flag AS (
    SELECT DISTINCT i.visit_id FROM HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS i
    INNER JOIN ltfu_cohort c ON i.visit_id = c.last_visit_id AND i.source_schema = c.source_schema
    WHERE (i.procedure_clinical_division ILIKE 'Radiology%' OR i.procedure_discipline ILIKE 'Radiology%')
      AND (i.cancelled IS NULL OR i.cancelled = 0)
      AND (i.remove_from_report IS NULL OR i.remove_from_report = 0)
),
no_fup_flag AS (
    SELECT c.patient_id FROM ltfu_cohort c
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON n.visit_id = c.last_visit_id AND n.source_schema = 'kisumu'
    GROUP BY c.patient_id
    HAVING MAX(CASE WHEN n.next_visit_date IS NOT NULL
                    AND TRIM(CAST(n.next_visit_date AS VARCHAR)) != '' THEN 1 ELSE 0 END) = 0
)
SELECT signal, patient_count, total.n AS total_patients,
    ROUND(100.0 * patient_count / NULLIF(total.n, 0), 1) AS pct
FROM (
    SELECT 'No follow-up date'    AS signal,
           COUNT(DISTINCT nf.patient_id) AS patient_count FROM no_fup_flag nf
    UNION ALL
    SELECT 'Prescription received',
           COUNT(DISTINCT c.patient_id) FROM ltfu_cohort c
           INNER JOIN rx_flag r ON r.visit_id = c.last_visit_id
    UNION ALL
    SELECT 'Lab tests ordered',
           COUNT(DISTINCT c.patient_id) FROM ltfu_cohort c
           INNER JOIN lab_flag l ON l.visit_id = c.last_visit_id
    UNION ALL
    SELECT 'Radiology ordered',
           COUNT(DISTINCT c.patient_id) FROM ltfu_cohort c
           INNER JOIN rad_flag rd ON rd.visit_id = c.last_visit_id
) signals
CROSS JOIN total
ORDER BY CASE signal
    WHEN 'No follow-up date'     THEN 1
    WHEN 'Prescription received' THEN 2
    WHEN 'Lab tests ordered'     THEN 3
    ELSE 4 END
"""
    return run_query(sql)


def load_wait_times(filters: dict, run_query) -> pd.DataFrame:
    """Flow & Retention Section E: avg investigation wait hours by lifecycle stage."""
    wh = _w(filters)
    wsa = _wsa(filters)
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
    WHERE v.created_at >= DATEADD('year', -1, sa.max_date)
      {wh}
      AND (dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1
           OR n.diagnosis ILIKE '%hypertension%'
           OR n.diagnosis ILIKE '%diabetes%'
           OR n.diagnosis ILIKE '%hiv%')
),
patient_lifecycle AS (
    SELECT cp.patient, cp.source_schema,
        CASE
            WHEN DATEDIFF('day', MAX(v.created_at), sa.max_date) <= 90  THEN 'Active'
            WHEN DATEDIFF('day', MAX(v.created_at), sa.max_date) <= 180 THEN 'Lapsing'
            ELSE 'LTFU'
        END AS lifecycle_stage
    FROM chronic_patients cp
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON cp.patient = v.patient AND cp.source_schema = v.source_schema
    INNER JOIN schema_anchor sa ON cp.source_schema = sa.source_schema
    GROUP BY cp.patient, cp.source_schema, sa.max_date
)
SELECT
    pl.lifecycle_stage,
    ROUND(AVG(NULLIF(
        DATEDIFF('minute', v.created_at, ei.investigation_created_at), 0
    )) / 60.0, 1) AS avg_wait_hours
FROM patient_lifecycle pl
INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    ON pl.patient = v.patient AND pl.source_schema = v.source_schema
INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_INVESTIGATIONS ei
    ON v.id = ei.visit_id AND v.source_schema = ei.source_schema
WHERE (ei.investigation_deleted_at IS NULL OR ei.investigation_deleted_at > CURRENT_TIMESTAMP)
  AND (ei.cancelled IS NULL OR ei.cancelled = 0)
  AND ei.investigation_created_at > v.created_at
  AND DATEDIFF('hour', v.created_at, ei.investigation_created_at) < 24
GROUP BY pl.lifecycle_stage
ORDER BY CASE pl.lifecycle_stage WHEN 'Active' THEN 1 WHEN 'Lapsing' THEN 2 ELSE 3 END
"""
    return run_query(sql)


def load_clinician_ltfu(filters: dict, run_query) -> pd.DataFrame:
    """Flow & Retention Section F: per-clinician LTFU rate for chronic patients. Min 5 chronic patients."""
    wh = _w(filters)
    wsa = _wsa(filters)
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
    WHERE v.created_at >= DATEADD('year', -1, sa.max_date)
      {wh}
      AND (dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1
           OR n.diagnosis ILIKE '%hypertension%'
           OR n.diagnosis ILIKE '%diabetes%'
           OR n.diagnosis ILIKE '%hiv%')
),
patient_lifecycle AS (
    SELECT cp.patient, cp.source_schema,
        DATEDIFF('day', MAX(v.created_at), sa.max_date) > 180 AS is_ltfu
    FROM chronic_patients cp
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON cp.patient = v.patient AND cp.source_schema = v.source_schema
    INNER JOIN schema_anchor sa ON cp.source_schema = sa.source_schema
    GROUP BY cp.patient, cp.source_schema, sa.max_date
),
clinician_patients AS (
    SELECT v.user AS clinician_id, v.source_schema,
        v.patient,
        MAX(pl.is_ltfu::INT) AS is_ltfu
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    INNER JOIN patient_lifecycle pl ON v.patient = pl.patient AND v.source_schema = pl.source_schema
    WHERE v.user IS NOT NULL AND TRIM(v.user) != ''
      AND v.created_at >= DATEADD('year', -1, sa.max_date)
      {wh}
    GROUP BY v.user, v.source_schema, v.patient
)
SELECT
    clinician_id,
    COUNT(DISTINCT patient)                                             AS chronic_seen,
    COUNT(DISTINCT CASE WHEN is_ltfu = 1 THEN patient END)            AS ltfu_count,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN is_ltfu = 1 THEN patient END),
        COUNT(DISTINCT patient)
    ) * 100, 1)                                                        AS ltfu_rate_pct
FROM clinician_patients
GROUP BY clinician_id
HAVING COUNT(DISTINCT patient) >= 5
ORDER BY ltfu_rate_pct DESC
"""
    return run_query(sql)


def load_ca_sepsis_enriched(filters: dict, run_query) -> pd.DataFrame:
    """Section B: Sepsis LOS outliers with corrected prior-contact classification.

    Changes from v1:
      - prior_inpatient exposes discharged_at + hours_since_prior_discharge
        (v1 only had days since prior admission — insufficient for 72h threshold)
      - prior_opd GROUP BY bug fixed: icd10_name_1 removed from GROUP BY;
        diagnosis joined after aggregation on visit_id only to prevent
        multiple rows per patient before QUALIFY fires
      - pathway_classification column added directly in SQL for inspection
    """
    wh_a = _w_adm(filters)
    wh_v  = _w(filters, alias="v")
    sql = f"""
WITH ward_fences AS (
    SELECT
        LOWER(REPLACE(a.source_schema, '_CLEAN', ''))  AS source_schema,
        a.ward_name,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY a.los_days)
            + 1.5 * (PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY a.los_days)
                   - PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY a.los_days)) AS upper_fence
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    WHERE a.los_days IS NOT NULL
    {wh_a}
    GROUP BY ALL
),
deduped_notes AS (
    SELECT visit_id, source_schema, TRIM(diagnosis) AS diagnosis
    FROM HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY visit_id ORDER BY created_at ASC NULLS LAST
    ) = 1
),
sepsis_outliers AS (
    SELECT
        a.patient_id,
        LOWER(REPLACE(a.source_schema, '_CLEAN', ''))                           AS source_schema,
        a.visit_id,
        a.ward_name,
        a.los_days,
        a.admitted_at,
        a.discharge_type,
        a.is_30day_readmission,
        COALESCE(dp.icd10_name_1, 'Other sepsis')                               AS sepsis_condition,
        dn.diagnosis                                                             AS notes_diagnosis
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    INNER JOIN ward_fences wf
        ON  LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = wf.source_schema
        AND a.ward_name = wf.ward_name
        AND a.los_days  > wf.upper_fence
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dp
        ON  a.visit_id  = dp.visit_id
        AND LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = dp.source_schema
    LEFT JOIN deduped_notes dn
        ON  a.visit_id  = dn.visit_id
        AND LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = dn.source_schema
    WHERE a.los_days IS NOT NULL
    {wh_a}
    AND (COALESCE(dp.disease_burden_group_1, '') ILIKE '%Infectious%'
      OR COALESCE(dp.disease_burden_group_1, '') ILIKE '%Sepsis%')
),
prior_opd_base AS (
    SELECT
        so.patient_id,
        so.source_schema,
        so.admitted_at                                                           AS sepsis_admitted_at,
        v.id                                                                     AS opd_visit_id,
        v.created_at                                                             AS opd_visit_date
    FROM sepsis_outliers so
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON  v.patient       = so.patient_id
        AND v.source_schema = so.source_schema
        AND v.created_at    < so.admitted_at
    WHERE 1=1 {wh_v}
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY so.patient_id, so.source_schema
        ORDER BY v.created_at DESC
    ) = 1
),
prior_opd AS (
    SELECT
        ob.patient_id,
        ob.source_schema,
        DATEDIFF('day', ob.opd_visit_date, ob.sepsis_admitted_at)               AS last_opd_days_before,
        dp.icd10_name_1                                                          AS last_opd_diagnosis
    FROM prior_opd_base ob
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dp
        ON  dp.visit_id      = ob.opd_visit_id
        AND dp.source_schema = ob.source_schema
),
prior_inpatient AS (
    SELECT
        so.patient_id,
        so.source_schema,
        dp.icd10_name_1                                                          AS prior_condition_display,
        a.admitted_at                                                            AS prior_admitted_at,
        a.discharged_at                                                          AS prior_discharged_at,
        DATEDIFF('day',  a.admitted_at,   so.admitted_at)                       AS prior_condition_days,
        DATEDIFF('hour', a.discharged_at, so.admitted_at)                       AS hours_since_prior_discharge
    FROM sepsis_outliers so
    INNER JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
        ON  a.patient_id  = so.patient_id
        AND LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = so.source_schema
        AND a.admitted_at < so.admitted_at
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dp
        ON  a.visit_id    = dp.visit_id
        AND LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = dp.source_schema
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY so.patient_id, so.source_schema
        ORDER BY a.admitted_at DESC
    ) = 1
)
SELECT
    so.ward_name,
    so.los_days,
    so.admitted_at                                                               AS sepsis_admitted_at,
    so.sepsis_condition,
    so.discharge_type,
    so.is_30day_readmission,
    so.notes_diagnosis,
    pi.prior_condition_display,
    pi.prior_admitted_at,
    pi.prior_discharged_at,
    pi.prior_condition_days,
    pi.hours_since_prior_discharge,
    po.last_opd_days_before,
    po.last_opd_diagnosis,
    CASE
        WHEN pi.prior_condition_display IS NOT NULL
             AND pi.prior_condition_days = 0
            THEN 'comorbid'
        WHEN pi.prior_condition_display IS NOT NULL
             AND pi.hours_since_prior_discharge IS NOT NULL
             AND pi.hours_since_prior_discharge > 0
             AND pi.hours_since_prior_discharge <= 72
            THEN 'hospital_acquired'
        WHEN po.last_opd_days_before IS NOT NULL
             AND po.last_opd_days_before = 0
            THEN 'same_day_escalation'
        WHEN po.last_opd_days_before IS NOT NULL
             AND po.last_opd_days_before BETWEEN 1 AND 7
            THEN 'opd_progression'
        ELSE 'community_acquired'
    END                                                                          AS pathway_classification
FROM sepsis_outliers so
LEFT JOIN prior_inpatient pi
    ON  pi.patient_id    = so.patient_id
    AND pi.source_schema = so.source_schema
LEFT JOIN prior_opd po
    ON  po.patient_id    = so.patient_id
    AND po.source_schema = so.source_schema
ORDER BY so.los_days DESC
"""
    return run_query(sql)


def load_ca_sepsis_prior_conditions(filters: dict, run_query) -> pd.DataFrame:
    """Section E: Prior inpatient ICD10 conditions for Sepsis LOS outlier patients."""
    wh_a = _w_adm(filters)
    sql = f"""
WITH ward_fences AS (
    SELECT
        LOWER(REPLACE(a.source_schema, '_CLEAN', ''))  AS source_schema,
        a.ward_name,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY a.los_days)
            + 1.5 * (PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY a.los_days)
                   - PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY a.los_days)) AS upper_fence
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    WHERE a.los_days IS NOT NULL
    {wh_a}
    GROUP BY ALL
),
outlier_patients AS (
    SELECT DISTINCT
        a.patient_id,
        LOWER(REPLACE(a.source_schema, '_CLEAN', ''))  AS source_schema,
        a.admitted_at                                   AS sepsis_admitted_at
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    INNER JOIN ward_fences wf
        ON  LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = wf.source_schema
        AND a.ward_name  = wf.ward_name
        AND a.los_days   > wf.upper_fence
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dp
        ON  a.visit_id   = dp.visit_id
        AND LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = dp.source_schema
    WHERE a.los_days IS NOT NULL
    {wh_a}
    AND (COALESCE(dp.disease_burden_group_1, '') ILIKE '%Infectious%'
      OR COALESCE(dp.disease_burden_group_1, '') ILIKE '%Sepsis%')
),
prior_admissions AS (
    SELECT
        op.patient_id,
        op.source_schema,
        dp.icd10_name_1                                                          AS prior_icd10_name,
        COALESCE(dp.disease_burden_group_1, 'Unclassified')                      AS prior_disease_burden_group,
        DATEDIFF('day', a.admitted_at, op.sepsis_admitted_at)                   AS days_before_outlier,
        CASE
            WHEN DATEDIFF('day', a.admitted_at, op.sepsis_admitted_at) <= 30  THEN '≤30 days'
            WHEN DATEDIFF('day', a.admitted_at, op.sepsis_admitted_at) <= 90  THEN '31–90 days'
            WHEN DATEDIFF('day', a.admitted_at, op.sepsis_admitted_at) <= 180 THEN '91–180 days'
            ELSE '>180 days'
        END                                                                      AS relationship_window
    FROM outlier_patients op
    INNER JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
        ON  a.patient_id  = op.patient_id
        AND LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = op.source_schema
        AND a.admitted_at < op.sepsis_admitted_at
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dp
        ON  a.visit_id    = dp.visit_id
        AND LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = dp.source_schema
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY op.patient_id, op.source_schema
        ORDER BY a.admitted_at DESC
    ) = 1
)
SELECT
    op.patient_id,
    op.source_schema,
    pa.prior_icd10_name,
    pa.prior_disease_burden_group,
    pa.days_before_outlier,
    pa.relationship_window
FROM outlier_patients op
LEFT JOIN prior_admissions pa
    ON  pa.patient_id   = op.patient_id
    AND pa.source_schema = op.source_schema
"""
    return run_query(sql)


def load_ca_sepsis_opd_history(filters: dict, run_query) -> pd.DataFrame:
    """Section E: OPD visit history for Sepsis outlier patients with no prior inpatient admission."""
    wh_a = _w_adm(filters)
    wh_v = _w(filters, alias="v")
    sql = f"""
WITH sepsis_outlier_patients AS (
    SELECT DISTINCT
        a.patient_id,
        LOWER(REPLACE(a.source_schema, '_CLEAN', ''))  AS source_schema,
        a.ward_name                                     AS sepsis_ward,
        a.los_days                                      AS sepsis_los,
        a.admitted_at                                   AS sepsis_admitted_at
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dp
        ON  a.visit_id = dp.visit_id
        AND LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = dp.source_schema
    WHERE a.los_days IS NOT NULL
    {wh_a}
    AND (COALESCE(dp.disease_burden_group_1, '') ILIKE '%Infectious%'
      OR COALESCE(dp.disease_burden_group_1, '') ILIKE '%Sepsis%')
),
no_prior_inpatient AS (
    SELECT sop.patient_id, sop.source_schema, sop.sepsis_ward,
           sop.sepsis_los, sop.sepsis_admitted_at
    FROM sepsis_outlier_patients sop
    WHERE NOT EXISTS (
        SELECT 1 FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS prev
        WHERE prev.patient_id   = sop.patient_id
          AND LOWER(REPLACE(prev.source_schema, '_CLEAN', '')) = sop.source_schema
          AND prev.admitted_at  < sop.sepsis_admitted_at
    )
),
prior_opd AS (
    SELECT
        npi.patient_id,
        npi.source_schema,
        MAX(v.created_at)                                                       AS opd_visit_date,
        DATEDIFF('day', MAX(v.created_at), npi.sepsis_admitted_at)             AS days_before_sepsis_opd,
        CASE
            WHEN DATEDIFF('day', MAX(v.created_at), npi.sepsis_admitted_at) <= 30  THEN '≤30 days'
            WHEN DATEDIFF('day', MAX(v.created_at), npi.sepsis_admitted_at) <= 90  THEN '31–90 days'
            ELSE '>90 days'
        END                                                                     AS opd_relationship_window
    FROM no_prior_inpatient npi
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON  v.patient      = npi.patient_id
        AND v.source_schema = npi.source_schema
        AND v.created_at   < npi.sepsis_admitted_at
    WHERE 1=1 {wh_v}
    GROUP BY npi.patient_id, npi.source_schema, npi.sepsis_admitted_at
)
SELECT
    npi.patient_id,
    npi.sepsis_ward,
    npi.sepsis_los,
    CASE WHEN po.patient_id IS NOT NULL THEN TRUE ELSE FALSE END                AS had_prior_opd_visit,
    po.opd_visit_date,
    po.days_before_sepsis_opd,
    po.opd_relationship_window
FROM no_prior_inpatient npi
LEFT JOIN prior_opd po
    ON  po.patient_id   = npi.patient_id
    AND po.source_schema = npi.source_schema
ORDER BY npi.sepsis_los DESC
"""
    return run_query(sql)


def load_ca_section_f(filters: dict, run_query) -> pd.DataFrame:
    """F: OPD to admission time by ward — median hours and within-4h rate."""
    wh   = _w(filters, alias="v")
    wh_a = _w_adm(filters, alias="a")
    sql = f"""
SELECT
    LOWER(REPLACE(a.source_schema, '_CLEAN', ''))                              AS source_schema,
    DATE_TRUNC('month', a.admitted_at)                                         AS month,
    a.ward_name,
    a.ward_category,
    COUNT(DISTINCT a.visit_id)                                                 AS total_admissions_with_prior_opd,
    ROUND(MEDIAN(DATEDIFF('hour', v.created_at, a.admitted_at)), 1)            AS median_hours_opd_to_admission,
    ROUND(AVG(DATEDIFF('hour', v.created_at, a.admitted_at)), 1)               AS avg_hours_opd_to_admission,
    COUNT(CASE WHEN DATEDIFF('hour', v.created_at, a.admitted_at) <= 4
               THEN 1 END)                                                     AS admitted_within_4h,
    COUNT(CASE WHEN DATEDIFF('hour', v.created_at, a.admitted_at) BETWEEN 5  AND 24 THEN 1 END) AS admitted_5_24h,
    COUNT(CASE WHEN DATEDIFF('hour', v.created_at, a.admitted_at) BETWEEN 25 AND 72 THEN 1 END) AS admitted_25_72h,
    ROUND(DIV0(
        COUNT(CASE WHEN DATEDIFF('hour', v.created_at, a.admitted_at) <= 4 THEN 1 END),
        COUNT(DISTINCT a.visit_id)
    ) * 100.0, 2)                                                              AS pct_admitted_within_4h
FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    ON  a.visit_id   = v.id
    AND LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = v.source_schema
    AND v.created_at < a.admitted_at
    AND DATEDIFF('hour', v.created_at, a.admitted_at) <= 72
WHERE 1=1
{wh_a}
GROUP BY ALL
ORDER BY month DESC, median_hours_opd_to_admission DESC
"""
    return run_query(sql)


def load_ca_opd_revisits(filters: dict, run_query) -> pd.DataFrame:
    """OPD return visits within 1-7 days with same or related diagnosis group."""
    wh = _w(filters, alias="v")
    schemas = filters.get("source_schemas") or []
    schema_filter = (
        "AND v.source_schema IN (" + ", ".join(repr(s) for s in schemas) + ")"
        if schemas else ""
    )
    sql = f"""
WITH index_visits AS (
    SELECT
        v.id                                                        AS index_visit_id,
        v.patient,
        v.source_schema,
        v.created_at                                                AS index_date,
        dx.disease_burden_group_1                                   AS index_group,
        COALESCE(dx.icd10_name_1, 'Unclassified')                   AS index_diagnosis
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON  v.id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE 1=1 {schema_filter}
    {wh}
),
return_visits AS (
    SELECT
        iv.index_visit_id,
        iv.source_schema,
        iv.index_group,
        iv.index_diagnosis,
        v2.id                                                       AS return_visit_id,
        v2.created_at                                               AS return_date,
        DATEDIFF('day', iv.index_date, v2.created_at)              AS days_to_return,
        dx2.disease_burden_group_1                                  AS return_group,
        IFF(
            iv.index_group IS NOT NULL
            AND dx2.disease_burden_group_1 IS NOT NULL
            AND iv.index_group = dx2.disease_burden_group_1,
            'SAME_GROUP', 'DIFFERENT_GROUP'
        )                                                           AS match_type,
        IFF(adm.visit_id IS NOT NULL, TRUE, FALSE)                  AS resulted_in_admission
    FROM index_visits iv
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v2
        ON  iv.patient       = v2.patient
        AND iv.source_schema = v2.source_schema
        AND DATEDIFF('day', iv.index_date, v2.created_at) BETWEEN 5 AND 7
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx2
        ON  v2.id = dx2.visit_id AND v2.source_schema = dx2.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS adm
        ON  v2.id = adm.visit_id
        AND LOWER(REPLACE(adm.source_schema, '_CLEAN', '')) = v2.source_schema
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY iv.index_visit_id ORDER BY v2.created_at ASC
    ) = 1
)
SELECT
    rv.index_visit_id,
    rv.source_schema,
    rv.index_diagnosis,
    rv.match_type,
    rv.days_to_return,
    rv.resulted_in_admission,
    CASE
        WHEN DATEDIFF('year', rp.dob, iv.index_date) < 5   THEN 'Toddler (0-4)'
        WHEN DATEDIFF('year', rp.dob, iv.index_date) < 13  THEN 'Child (5-12)'
        WHEN DATEDIFF('year', rp.dob, iv.index_date) < 18  THEN 'Adolescent (13-17)'
        WHEN DATEDIFF('year', rp.dob, iv.index_date) < 25  THEN 'Youth (18-24)'
        WHEN DATEDIFF('year', rp.dob, iv.index_date) < 35  THEN 'Young Adult (25-34)'
        WHEN DATEDIFF('year', rp.dob, iv.index_date) < 45  THEN 'Adult (35-44)'
        WHEN DATEDIFF('year', rp.dob, iv.index_date) < 55  THEN 'Middle Age (45-54)'
        WHEN DATEDIFF('year', rp.dob, iv.index_date) < 65  THEN 'Older Adult (55-64)'
        ELSE                                                      'Senior (65+)'
    END                                                             AS age_group
FROM return_visits rv
INNER JOIN index_visits iv ON rv.index_visit_id = iv.index_visit_id
                           AND rv.source_schema  = iv.source_schema
LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS rp
    ON  iv.patient       = rp.patient_id
    AND iv.source_schema = rp.source_schema
ORDER BY rv.days_to_return ASC
"""
    return run_query(sql)


def load_ca_total_opd_visits(filters: dict, run_query) -> pd.DataFrame:
    """Total OPD visit count — denominator for OPD re-visit rate."""
    wh = _w(filters, alias="v")
    schemas = filters.get("source_schemas") or []
    schema_filter = (
        "AND v.source_schema IN (" + ", ".join(repr(s) for s in schemas) + ")"
        if schemas else ""
    )
    sql = f"""
SELECT COUNT(DISTINCT id) AS total_opd_visits
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
WHERE 1=1 {schema_filter}
{wh}
"""
    return run_query(sql)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — OPD TO IPD CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def load_opd_ipd_overall(filters: dict, run_query) -> pd.DataFrame:
    """OPD→IPD: header KPIs — overall rate, retention universe rate, mix gap,
    total IPD admissions, strain months, total months."""
    wh    = _w(filters)
    wsa   = _wsa(filters)
    wh_a  = _w_adm(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
visits AS (
    SELECT v.source_schema, v.id AS visit_id, v.patient, v.created_at, v.user
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= '2024-09-01' {wh}
),
admissions AS (
    SELECT REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema,
           a.visit_id
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    WHERE a.admitted_at >= '2024-09-01' {wh_a}
),
complex AS (
    SELECT DISTINCT v.source_schema, v.visit_id
    FROM visits v
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.visit_id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON v.visit_id = n.visit_id AND v.source_schema = n.source_schema
    WHERE dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1
       OR dx.disease_burden_group_1 ILIKE '%maternal%'
       OR dx.disease_burden_group_1 ILIKE '%obstet%'
       OR dx.disease_burden_group_1 ILIKE '%oncol%'
       OR dx.disease_burden_group_1 ILIKE '%cancer%'
       OR dx.disease_burden_group_1 ILIKE '%mental%'
       OR dx.disease_burden_group_1 ILIKE '%psychiat%'
       OR n.diagnosis ILIKE '%hypertension%'
       OR n.diagnosis ILIKE '%diabetes%'
       OR n.diagnosis ILIKE '%hiv%'
),
monthly AS (
    SELECT
        DATE_TRUNC('month', v.created_at) AS visit_month,
        COUNT(DISTINCT v.visit_id) AS total_visits,
        COUNT(DISTINCT CASE WHEN a.visit_id IS NOT NULL THEN v.visit_id END) AS ipd_visits,
        COUNT(DISTINCT v.user) AS clinician_count,
        ROUND(DIV0(
            COUNT(DISTINCT CASE WHEN a.visit_id IS NOT NULL THEN v.visit_id END),
            COUNT(DISTINCT v.visit_id)
        ) * 100, 2) AS conv_rate
    FROM visits v
    LEFT JOIN admissions a ON v.visit_id = a.visit_id AND v.source_schema = a.source_schema
    GROUP BY 1
),
overall_stats AS (
    SELECT
        ROUND(DIV0(SUM(ipd_visits), SUM(total_visits)) * 100, 2) AS overall_rate,
        SUM(ipd_visits)                                            AS total_ipd,
        COUNT(*)                                                   AS total_months,
        AVG(conv_rate)                                             AS avg_conv,
        AVG(total_visits / NULLIF(clinician_count, 0))            AS avg_load
    FROM monthly
),
strain AS (
    SELECT COUNT(*) AS strain_months
    FROM monthly m
    CROSS JOIN overall_stats o
    WHERE m.total_visits / NULLIF(m.clinician_count, 0) > o.avg_load * 1.3
      AND m.conv_rate < o.avg_conv
),
universe_stats AS (
    SELECT
        ROUND(DIV0(
            COUNT(DISTINCT CASE WHEN a.visit_id IS NOT NULL THEN v.visit_id END),
            COUNT(DISTINCT v.visit_id)
        ) * 100, 1) AS universe_rate
    FROM visits v
    INNER JOIN complex cx ON v.visit_id = cx.visit_id AND v.source_schema = cx.source_schema
    LEFT JOIN admissions a ON v.visit_id = a.visit_id AND v.source_schema = a.source_schema
)
SELECT
    o.overall_rate        AS overall_rate_pct,
    u.universe_rate       AS retention_universe_rate_pct,
    ROUND(u.universe_rate - o.overall_rate, 1) AS mix_gap_pp,
    o.total_ipd           AS total_ipd_admissions,
    s.strain_months,
    o.total_months
FROM overall_stats o
CROSS JOIN universe_stats u
CROSS JOIN strain s
"""
    return run_query(sql)


def load_opd_ipd_segments(filters: dict, run_query) -> pd.DataFrame:
    """OPD→IPD: conversion rate by clinical segment (Chronic / Maternal / Oncology / Mental Health)."""
    wh    = _w(filters)
    wsa   = _wsa(filters)
    wh_a  = _w_adm(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
visits AS (
    SELECT v.source_schema, v.id AS visit_id
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= '2024-09-01' {wh}
),
admissions AS (
    SELECT REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema, a.visit_id
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a WHERE a.admitted_at >= '2024-09-01' {wh_a}
),
classified AS (
    SELECT v.source_schema, v.visit_id,
        CASE
            WHEN dx.disease_burden_group_1 ILIKE '%maternal%'
              OR dx.disease_burden_group_1 ILIKE '%obstet%'
              OR dx.disease_burden_group_1 ILIKE '%perinatal%'
              OR dx.disease_burden_group_1 ILIKE '%mnch%'    THEN 'Maternal'
            WHEN dx.disease_burden_group_1 ILIKE '%oncol%'
              OR dx.disease_burden_group_1 ILIKE '%cancer%'
              OR dx.disease_burden_group_1 ILIKE '%chemo%'   THEN 'Oncology'
            WHEN dx.disease_burden_group_1 ILIKE '%mental%'
              OR dx.disease_burden_group_1 ILIKE '%psychiat%'
              OR dx.disease_burden_group_1 ILIKE '%behavio%' THEN 'Mental Health'
            WHEN dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1
              OR n.diagnosis ILIKE '%hypertension%'
              OR n.diagnosis ILIKE '%diabetes%'
              OR n.diagnosis ILIKE '%hiv%'                   THEN 'Chronic'
            ELSE NULL
        END AS segment
    FROM visits v
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.visit_id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON v.visit_id = n.visit_id AND v.source_schema = n.source_schema
)
SELECT
    c.segment,
    COUNT(DISTINCT c.visit_id)                                                              AS total_opd_visits,
    COUNT(DISTINCT CASE WHEN a.visit_id IS NOT NULL THEN c.visit_id END)                   AS ipd_admissions,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN a.visit_id IS NOT NULL THEN c.visit_id END),
        COUNT(DISTINCT c.visit_id)
    ) * 100, 2)                                                                             AS conversion_rate_pct,
    CASE c.segment
        WHEN 'Chronic'      THEN 8
        WHEN 'Maternal'     THEN 15
        WHEN 'Oncology'     THEN 15
        WHEN 'Mental Health' THEN 8
    END AS ref_lower,
    CASE c.segment
        WHEN 'Chronic'      THEN 15
        WHEN 'Maternal'     THEN 25
        WHEN 'Oncology'     THEN 25
        WHEN 'Mental Health' THEN 15
    END AS ref_upper
FROM classified c
LEFT JOIN admissions a ON c.visit_id = a.visit_id AND c.source_schema = a.source_schema
WHERE c.segment IS NOT NULL
GROUP BY c.segment
ORDER BY CASE c.segment
    WHEN 'Chronic' THEN 1 WHEN 'Maternal' THEN 2
    WHEN 'Oncology' THEN 3 ELSE 4 END
"""
    return run_query(sql)


def load_opd_ipd_monthly(filters: dict, run_query) -> pd.DataFrame:
    """OPD→IPD: monthly overall conversion rate and retention universe rate."""
    wh    = _w(filters)
    wsa   = _wsa(filters)
    wh_a  = _w_adm(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
visits AS (
    SELECT v.source_schema, v.id AS visit_id, v.created_at
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= '2024-09-01' {wh}
),
admissions AS (
    SELECT REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema, a.visit_id
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a WHERE a.admitted_at >= '2024-09-01' {wh_a}
),
complex AS (
    SELECT DISTINCT v.source_schema, v.visit_id
    FROM visits v
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.visit_id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON v.visit_id = n.visit_id AND v.source_schema = n.source_schema
    WHERE dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1
       OR dx.disease_burden_group_1 ILIKE ANY ('%maternal%','%obstet%','%oncol%','%cancer%','%mental%','%psychiat%')
       OR n.diagnosis ILIKE ANY ('%hypertension%','%diabetes%','%hiv%')
)
SELECT
    DATE_TRUNC('month', v.created_at) AS visit_month,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN a.visit_id IS NOT NULL THEN v.visit_id END),
        COUNT(DISTINCT v.visit_id)
    ) * 100, 2) AS overall_rate_pct,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN a.visit_id IS NOT NULL AND cx.visit_id IS NOT NULL THEN v.visit_id END),
        COUNT(DISTINCT CASE WHEN cx.visit_id IS NOT NULL THEN v.visit_id END)
    ) * 100, 2) AS retention_universe_rate_pct
FROM visits v
LEFT JOIN admissions a  ON v.visit_id = a.visit_id  AND v.source_schema = a.source_schema
LEFT JOIN complex cx    ON v.visit_id = cx.visit_id AND v.source_schema = cx.source_schema
GROUP BY 1
ORDER BY 1
"""
    return run_query(sql)


def load_opd_ipd_benchmark(filters: dict, run_query) -> pd.DataFrame:
    """OPD→IPD: conversion rate by diagnosis — min 50 OPD visits."""
    wh    = _w(filters)
    wsa   = _wsa(filters)
    wh_a  = _w_adm(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
visits AS (
    SELECT v.source_schema, v.id AS visit_id
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= '2024-09-01' {wh}
),
admissions AS (
    SELECT REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema, a.visit_id
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a WHERE a.admitted_at >= '2024-09-01' {wh_a}
),
dx_visits AS (
    SELECT v.source_schema, v.visit_id,
        COALESCE(dx.disease_burden_group_1, 'Unclassified') AS cleaned_diagnosis_name
    FROM visits v
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.visit_id = dx.visit_id AND v.source_schema = dx.source_schema
    WHERE dx.disease_burden_group_1 IS NOT NULL
      AND dx.disease_burden_group_1 != 'Unclassified'
)
SELECT
    dv.cleaned_diagnosis_name,
    COUNT(DISTINCT dv.visit_id)                                                        AS total_opd_visits,
    COUNT(DISTINCT CASE WHEN a.visit_id IS NOT NULL THEN dv.visit_id END)              AS ipd_admissions,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN a.visit_id IS NOT NULL THEN dv.visit_id END),
        COUNT(DISTINCT dv.visit_id)
    ) * 100, 1)                                                                        AS actual_rate_pct,
    8.0  AS ref_lower,
    15.0 AS ref_upper
FROM dx_visits dv
LEFT JOIN admissions a ON dv.visit_id = a.visit_id AND dv.source_schema = a.source_schema
GROUP BY dv.cleaned_diagnosis_name
HAVING COUNT(DISTINCT dv.visit_id) >= 50
ORDER BY actual_rate_pct DESC
"""
    return run_query(sql)


def load_opd_ipd_comorbidity(filters: dict, run_query) -> pd.DataFrame:
    """OPD→IPD: conversion rate by comorbidity group — monthly + overall (VISIT_MONTH = NULL)."""
    wh    = _w(filters)
    wsa   = _wsa(filters)
    wh_a  = _w_adm(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
visits AS (
    SELECT v.source_schema, v.id AS visit_id, v.created_at
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= '2024-09-01' {wh}
),
admissions AS (
    SELECT REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema, a.visit_id
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a WHERE a.admitted_at >= '2024-09-01' {wh_a}
),
dx_counts AS (
    SELECT v.source_schema, v.visit_id, v.created_at,
        MAX(CASE
            WHEN dx.icd10_code_1 IS NOT NULL
             AND (dx.icd10_code_2 IS NOT NULL OR dx.icd10_code_3 IS NOT NULL)
            THEN 2 ELSE 1
        END) AS dx_count,
        MAX(CASE WHEN dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1 THEN 1 ELSE 0 END) AS is_chronic
    FROM visits v
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.visit_id = dx.visit_id AND v.source_schema = dx.source_schema
    GROUP BY v.source_schema, v.visit_id, v.created_at
),
grouped AS (
    SELECT source_schema, visit_id, created_at,
        CASE
            WHEN is_chronic = 1 AND dx_count >= 2 THEN 'Chronic comorbid'
            WHEN dx_count >= 2                     THEN 'Comorbid'
            ELSE 'Single diagnosis'
        END AS patient_group
    FROM dx_counts
),
monthly AS (
    SELECT
        DATE_TRUNC('month', g.created_at) AS visit_month,
        g.patient_group,
        COUNT(DISTINCT g.visit_id)                                                   AS total_opd_visits,
        COUNT(DISTINCT CASE WHEN a.visit_id IS NOT NULL THEN g.visit_id END)         AS ipd_admissions,
        ROUND(DIV0(
            COUNT(DISTINCT CASE WHEN a.visit_id IS NOT NULL THEN g.visit_id END),
            COUNT(DISTINCT g.visit_id)
        ) * 100, 2) AS conversion_rate_pct
    FROM grouped g
    LEFT JOIN admissions a ON g.visit_id = a.visit_id AND g.source_schema = a.source_schema
    GROUP BY 1, 2
),
overall AS (
    SELECT
        NULL::DATE AS visit_month,
        g.patient_group,
        COUNT(DISTINCT g.visit_id)                                                   AS total_opd_visits,
        COUNT(DISTINCT CASE WHEN a.visit_id IS NOT NULL THEN g.visit_id END)         AS ipd_admissions,
        ROUND(DIV0(
            COUNT(DISTINCT CASE WHEN a.visit_id IS NOT NULL THEN g.visit_id END),
            COUNT(DISTINCT g.visit_id)
        ) * 100, 2) AS conversion_rate_pct
    FROM grouped g
    LEFT JOIN admissions a ON g.visit_id = a.visit_id AND g.source_schema = a.source_schema
    GROUP BY g.patient_group
)
SELECT * FROM monthly
UNION ALL
SELECT * FROM overall
ORDER BY visit_month NULLS FIRST, patient_group
"""
    return run_query(sql)


def load_opd_ipd_age_conversion(filters: dict, run_query) -> pd.DataFrame:
    """OPD→IPD: conversion rate by age group for chronic patients."""
    wh    = _w(filters)
    wsa   = _wsa(filters)
    wh_a  = _w_adm(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
visits AS (
    SELECT v.source_schema, v.id AS visit_id, v.patient, v.created_at
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= '2024-09-01' {wh}
),
admissions AS (
    SELECT REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema, a.visit_id
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a WHERE a.admitted_at >= '2024-09-01' {wh_a}
),
chronic_visits AS (
    SELECT DISTINCT v.source_schema, v.visit_id, v.patient, v.created_at
    FROM visits v
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.visit_id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON v.visit_id = n.visit_id AND v.source_schema = n.source_schema
    WHERE dx.is_chronic_1 = 1 OR dx.is_chronic_2 = 1
       OR n.diagnosis ILIKE '%hypertension%' OR n.diagnosis ILIKE '%diabetes%'
       OR n.diagnosis ILIKE '%hiv%'
),
with_age AS (
    SELECT cv.source_schema, cv.visit_id,
        CASE
            WHEN TIMESTAMPDIFF('year', rp.dob, cv.created_at) < 5   THEN 'Child Under 5'
            WHEN TIMESTAMPDIFF('year', rp.dob, cv.created_at) < 13  THEN 'Child 5–12'
            WHEN TIMESTAMPDIFF('year', rp.dob, cv.created_at) < 18  THEN 'Adolescent 13–17'
            WHEN TIMESTAMPDIFF('year', rp.dob, cv.created_at) < 25  THEN 'Young Adult 18–24'
            WHEN TIMESTAMPDIFF('year', rp.dob, cv.created_at) < 35  THEN 'Adult 25–34'
            WHEN TIMESTAMPDIFF('year', rp.dob, cv.created_at) < 45  THEN 'Adult 35–44'
            WHEN TIMESTAMPDIFF('year', rp.dob, cv.created_at) < 55  THEN 'Adult 45–54'
            WHEN TIMESTAMPDIFF('year', rp.dob, cv.created_at) < 65  THEN 'Adult 55–64'
            ELSE 'Senior 65+'
        END AS age_group
    FROM chronic_visits cv
    LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS rp
        ON cv.patient = rp.patient_id AND cv.source_schema = rp.source_schema
)
SELECT
    age_group,
    COUNT(DISTINCT wa.visit_id)                                                     AS total_chronic_visits,
    COUNT(DISTINCT CASE WHEN a.visit_id IS NOT NULL THEN wa.visit_id END)           AS ipd_admissions,
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN a.visit_id IS NOT NULL THEN wa.visit_id END),
        COUNT(DISTINCT wa.visit_id)
    ) * 100, 1)                                                                     AS conversion_rate_pct
FROM with_age wa
LEFT JOIN admissions a ON wa.visit_id = a.visit_id AND wa.source_schema = a.source_schema
GROUP BY age_group
HAVING COUNT(DISTINCT wa.visit_id) >= 10
ORDER BY conversion_rate_pct ASC
"""
    return run_query(sql)


def load_opd_ipd_escalation(filters: dict, run_query) -> pd.DataFrame:
    """OPD→IPD: 72h escalation — OPD visit not admitted, then admitted within 72h.
    Returns per-age-group rows PLUS scalar columns on every row."""
    wh    = _w(filters)
    wsa   = _wsa(filters)
    wh_a  = _w_adm(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
all_visits AS (
    SELECT v.source_schema, v.id AS visit_id, v.patient, v.created_at
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= '2024-09-01' {wh}
),
admissions AS (
    SELECT REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema,
           a.visit_id, a.patient_id, a.admitted_at,
           COALESCE(dx.icd10_name_1, dx.disease_burden_group_1) AS classified_dx,
           dx.disease_burden_group_1 AS disease_burden_group
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON a.visit_id = dx.visit_id
       AND REPLACE(LOWER(a.source_schema), '_clean', '') = dx.source_schema
    WHERE a.admitted_at >= '2024-09-01' {wh_a}
),
opd_only AS (
    SELECT av.source_schema, av.visit_id, av.patient, av.created_at
    FROM all_visits av
    LEFT JOIN admissions adm ON av.visit_id = adm.visit_id AND av.source_schema = adm.source_schema
    WHERE adm.visit_id IS NULL
),
escalated AS (
    SELECT DISTINCT op.source_schema, op.visit_id AS opd_visit_id, op.patient,
        adm.visit_id AS adm_visit_id,
        adm.admitted_at
    FROM opd_only op
    INNER JOIN admissions adm
        ON op.patient = adm.patient_id
       AND op.source_schema = adm.source_schema
       AND adm.admitted_at > op.created_at
       AND DATEDIFF('hour', op.created_at, adm.admitted_at) <= 72
),
with_age AS (
    SELECT e.source_schema, e.opd_visit_id, e.patient,
        adm.classified_dx,
        adm.disease_burden_group,
        CASE
            WHEN TIMESTAMPDIFF('year', rp.dob, e.admitted_at) < 5   THEN 'Child Under 5'
            WHEN TIMESTAMPDIFF('year', rp.dob, e.admitted_at) < 13  THEN 'Child 5–12'
            WHEN TIMESTAMPDIFF('year', rp.dob, e.admitted_at) < 18  THEN 'Adolescent 13–17'
            WHEN TIMESTAMPDIFF('year', rp.dob, e.admitted_at) < 25  THEN 'Young Adult 18–24'
            WHEN TIMESTAMPDIFF('year', rp.dob, e.admitted_at) < 35  THEN 'Adult 25–34'
            WHEN TIMESTAMPDIFF('year', rp.dob, e.admitted_at) < 45  THEN 'Adult 35–44'
            WHEN TIMESTAMPDIFF('year', rp.dob, e.admitted_at) < 55  THEN 'Adult 45–54'
            WHEN TIMESTAMPDIFF('year', rp.dob, e.admitted_at) < 65  THEN 'Adult 55–64'
            ELSE 'Senior 65+'
        END AS age_group
    FROM escalated e
    LEFT JOIN admissions adm ON e.adm_visit_id = adm.visit_id AND e.source_schema = adm.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_RECEPTION_PATIENTS rp
        ON e.patient = rp.patient_id AND e.source_schema = rp.source_schema
),
top_dx AS (
    SELECT classified_dx,
           COUNT(*) AS cnt
    FROM with_age WHERE classified_dx IS NOT NULL
    GROUP BY 1 ORDER BY cnt DESC LIMIT 1
),
top_burden_group AS (
    SELECT disease_burden_group, COUNT(*) AS cnt
    FROM with_age WHERE disease_burden_group IS NOT NULL
    GROUP BY 1 ORDER BY cnt DESC LIMIT 1
),
totals AS (
    SELECT COUNT(DISTINCT opd_visit_id) AS total_72h
    FROM with_age
),
all_opd AS (
    SELECT COUNT(DISTINCT visit_id) AS n FROM opd_only
)
SELECT
    wa.age_group,
    COUNT(DISTINCT wa.opd_visit_id)  AS total_escalations,
    t.total_72h                      AS total_72h_escalations,
    ROUND(DIV0(t.total_72h, ao.n) * 100, 2) AS escalation_rate_pct,
    (SELECT classified_dx FROM top_dx LIMIT 1)          AS top_classified_diagnosis,
    (SELECT disease_burden_group FROM top_burden_group LIMIT 1) AS top_disease_burden_group
FROM with_age wa
CROSS JOIN totals t
CROSS JOIN all_opd ao
GROUP BY wa.age_group, t.total_72h, ao.n
ORDER BY total_escalations DESC
"""
    return run_query(sql)


def load_opd_ipd_workload_triangle(filters: dict, run_query) -> pd.DataFrame:
    """OPD→IPD: monthly conversion rate vs clinician load — strain signal classification."""
    wh    = _w(filters)
    wsa   = _wsa(filters)
    wh_a  = _w_adm(filters)
    sql = f"""
WITH schema_anchor AS (
    SELECT source_schema, MAX(created_at) AS max_date
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
    GROUP BY source_schema
),
visits AS (
    SELECT v.source_schema, v.id AS visit_id, v.created_at, v.user
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN schema_anchor sa ON v.source_schema = sa.source_schema
    WHERE v.created_at >= '2024-09-01'
      AND v.user IS NOT NULL AND TRIM(v.user) != ''
    {wh}
),
admissions AS (
    SELECT REPLACE(LOWER(a.source_schema), '_clean', '') AS source_schema, a.visit_id
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a WHERE a.admitted_at >= '2024-09-01' {wh_a}
),
monthly AS (
    SELECT
        DATE_TRUNC('month', v.created_at)                   AS visit_month,
        COUNT(DISTINCT v.visit_id)                          AS total_visits,
        COUNT(DISTINCT CASE WHEN a.visit_id IS NOT NULL THEN v.visit_id END) AS ipd_visits,
        COUNT(DISTINCT v.user)                              AS clinician_count,
        ROUND(DIV0(
            COUNT(DISTINCT CASE WHEN a.visit_id IS NOT NULL THEN v.visit_id END),
            COUNT(DISTINCT v.visit_id)
        ) * 100, 2)                                         AS conversion_rate_pct
    FROM visits v
    LEFT JOIN admissions a ON v.visit_id = a.visit_id AND v.source_schema = a.source_schema
    GROUP BY 1
),
stats AS (
    SELECT AVG(conversion_rate_pct) AS avg_conv,
           AVG(total_visits / NULLIF(clinician_count, 0)) AS avg_load
    FROM monthly
)
SELECT
    m.visit_month,
    m.conversion_rate_pct,
    ROUND(m.total_visits / NULLIF(m.clinician_count, 0), 1) AS avg_visits_per_clinician,
    CASE
        WHEN m.total_visits / NULLIF(m.clinician_count, 0) > s.avg_load * 1.3
             AND m.conversion_rate_pct < s.avg_conv        THEN 'HIGH_STRAIN'
        WHEN m.total_visits / NULLIF(m.clinician_count, 0) > s.avg_load * 1.3 THEN 'CAPACITY_GAP'
        ELSE 'AS_EXPECTED'
    END AS strain_signal
FROM monthly m
CROSS JOIN stats s
ORDER BY m.visit_month
"""
    return run_query(sql)


# ══════════════════════════════════════════════════════════════════════════════
# PATIENT ACQUISITION TAB
# ══════════════════════════════════════════════════════════════════════════════

def load_acquisition_overview(filters: dict, run_query) -> pd.DataFrame:
    """Acquisition: header KPIs — total, new, returning, chronic, avg visits, return rate."""
    vb  = _visit_base_cte(filters, start_date='2024-09-01')
    wh  = _w(filters)
    wsa = _wsa(filters)
    sql = f"""
WITH {vb},
chronic_base AS (
    SELECT DISTINCT v.source_schema, v.patient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    INNER JOIN (
        SELECT source_schema, MAX(created_at) AS max_date
        FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS {wsa}
        GROUP BY source_schema
    ) sa ON v.source_schema = sa.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED dx
        ON v.id = dx.visit_id AND v.source_schema = dx.source_schema
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES n
        ON v.id = n.visit_id AND v.source_schema = n.source_schema
    WHERE v.created_at >= '2024-09-01' {wh}
      AND (
          COALESCE(dx.is_chronic_1, 0) = 1
          OR COALESCE(dx.is_chronic_2, 0) = 1
          OR n.diagnosis ILIKE '%hypertension%'
          OR n.diagnosis ILIKE '%diabetes%'
          OR n.diagnosis ILIKE '%hiv%'
      )
)
SELECT
    COUNT(DISTINCT vb.patient_id)                                                    AS total_patients,
    COUNT(DISTINCT CASE WHEN vb.patient_type = 'New'       THEN vb.patient_id END)  AS new_patients,
    COUNT(DISTINCT CASE WHEN vb.patient_type = 'Returning' THEN vb.patient_id END)  AS returning_patients,
    (SELECT COUNT(DISTINCT patient) FROM chronic_base)                               AS chronic_patients,
    COUNT(DISTINCT CASE WHEN vb.total_visit_count >= 2     THEN vb.patient_id END)  AS repeat_patients,
    ROUND(COUNT(vb.visit_id) * 1.0 / NULLIF(COUNT(DISTINCT vb.patient_id), 0), 1)  AS avg_visits_per_patient,
    ROUND(
        COUNT(DISTINCT CASE WHEN vb.patient_type = 'Returning' THEN vb.patient_id END) * 100.0
        / NULLIF(COUNT(DISTINCT vb.patient_id), 0), 1
    ) AS return_rate_pct
FROM visit_base vb
"""
    return run_query(sql)


def load_age_gender(filters: dict, run_query) -> pd.DataFrame:
    """Acquisition: patient count by age group and gender."""
    vb = _visit_base_cte(filters, start_date='2024-09-01')
    sql = f"""
WITH {vb}
SELECT
    age_group,
    gender,
    COUNT(DISTINCT patient_id) AS patient_count
FROM visit_base
WHERE age_group != 'Unknown'
  AND UPPER(COALESCE(gender, '')) IN ('F', 'FEMALE', 'M', 'MALE')
GROUP BY age_group, gender
ORDER BY age_group, gender
"""
    return run_query(sql)


def load_age_growth_index(filters: dict, run_query) -> pd.DataFrame:
    """Acquisition: monthly growth index per age cohort.
    GROWTH_INDEX = current_month_patients / prior_month_patients * 100."""
    vb = _visit_base_cte(filters, start_date='2024-09-01')
    sql = f"""
WITH {vb},
monthly AS (
    SELECT age_group, visit_month,
           COUNT(DISTINCT patient_id) AS patients
    FROM visit_base
    WHERE age_group != 'Unknown'
    GROUP BY age_group, visit_month
),
indexed AS (
    SELECT age_group, visit_month, patients,
           LAG(patients) OVER (PARTITION BY age_group ORDER BY visit_month) AS prior_patients
    FROM monthly
)
SELECT
    age_group,
    visit_month,
    ROUND(DIV0(patients * 100.0, NULLIF(prior_patients, 0)), 1) AS growth_index
FROM indexed
WHERE prior_patients IS NOT NULL
ORDER BY age_group, visit_month
"""
    return run_query(sql)


def load_condition_profile(filters: dict, run_query) -> pd.DataFrame:
    """Acquisition: patient count by condition, visit type (New/Returning),
    gender, IP/OP flag, age group."""
    vb = _visit_base_cte(filters, start_date='2024-09-01')
    sql = f"""
WITH {vb}
SELECT
    disease_burden_group        AS condition,
    patient_type                AS visit_type,
    gender,
    visit_type                  AS ip_op_flag,
    age_group,
    COUNT(DISTINCT patient_id)  AS patient_count
FROM visit_base
WHERE disease_burden_group != 'Unclassified'
  AND age_group != 'Unknown'
GROUP BY disease_burden_group, patient_type, gender, visit_type, age_group
ORDER BY condition, visit_type
"""
    return run_query(sql)


def load_rn_ratios(filters: dict, run_query) -> pd.DataFrame:
    """Acquisition: Return-to-New ratio per clinical segment.
    VISIT_MONTH = NULL for overall, populated for monthly rows."""
    vb = _visit_base_cte(filters, start_date='2024-09-01')
    sql = f"""
WITH {vb},
segmented AS (
    SELECT
        visit_month,
        patient_type,
        patient_id,
        CASE
            WHEN (disease_burden_group ILIKE '%oncol%' OR disease_burden_group ILIKE '%cancer%')
            THEN 'Oncology'
            WHEN (disease_burden_group ILIKE '%maternal%' OR disease_burden_group ILIKE '%obstet%'
               OR disease_burden_group ILIKE '%mnch%'    OR disease_burden_group ILIKE '%perinatal%')
            THEN 'Maternal'
            WHEN (disease_burden_group ILIKE '%mental%' OR disease_burden_group ILIKE '%psychiatr%')
            THEN 'Mental Health'
            WHEN is_chronic = 1
            THEN 'Chronic'
            ELSE NULL
        END AS segment
    FROM visit_base
),
filt AS (SELECT * FROM segmented WHERE segment IS NOT NULL),
monthly_agg AS (
    SELECT
        visit_month,
        segment,
        COUNT(DISTINCT CASE WHEN patient_type = 'New'       THEN patient_id END) AS new_patients,
        COUNT(DISTINCT CASE WHEN patient_type = 'Returning' THEN patient_id END) AS returning_patients,
        ROUND(DIV0(
            COUNT(DISTINCT CASE WHEN patient_type = 'Returning' THEN patient_id END),
            NULLIF(COUNT(DISTINCT CASE WHEN patient_type = 'New' THEN patient_id END), 0)
        ), 2) AS rn_ratio
    FROM filt
    GROUP BY visit_month, segment
),
overall_agg AS (
    SELECT
        NULL::DATE AS visit_month,
        segment,
        COUNT(DISTINCT CASE WHEN patient_type = 'New'       THEN patient_id END) AS new_patients,
        COUNT(DISTINCT CASE WHEN patient_type = 'Returning' THEN patient_id END) AS returning_patients,
        ROUND(DIV0(
            COUNT(DISTINCT CASE WHEN patient_type = 'Returning' THEN patient_id END),
            NULLIF(COUNT(DISTINCT CASE WHEN patient_type = 'New' THEN patient_id END), 0)
        ), 2) AS rn_ratio
    FROM filt
    GROUP BY segment
)
SELECT * FROM overall_agg
UNION ALL
SELECT * FROM monthly_agg
ORDER BY visit_month NULLS FIRST, segment
"""
    return run_query(sql)


def load_new_returning_trend(filters: dict, run_query) -> pd.DataFrame:
    """Acquisition: monthly new vs returning patient counts."""
    vb = _visit_base_cte(filters, start_date='2024-09-01')
    sql = f"""
WITH {vb}
SELECT
    visit_month,
    COUNT(DISTINCT CASE WHEN patient_type = 'New'       THEN patient_id END) AS new_patients,
    COUNT(DISTINCT CASE WHEN patient_type = 'Returning' THEN patient_id END) AS returning_patients
FROM visit_base
GROUP BY visit_month
ORDER BY visit_month
"""
    return run_query(sql)


def load_level4_benchmark(filters: dict, run_query) -> pd.DataFrame:
    """Acquisition: facility patient mix vs Level 4 benchmark (Tenri as Level 4 proxy).
    Returns PATIENT_TYPE, FACILITY_PCT, BENCHMARK_PCT, GAP_PP."""
    vb = _visit_base_cte(filters, start_date='2024-09-01')
    sql = f"""
WITH {vb},
facility AS (
    SELECT
        patient_type,
        COUNT(DISTINCT patient_id) AS patients
    FROM visit_base
    GROUP BY patient_type
),
facility_pct AS (
    SELECT
        patient_type,
        ROUND(patients * 100.0 / NULLIF(SUM(patients) OVER (), 0), 1) AS facility_pct
    FROM facility
),
tenri_classified AS (
    SELECT
        v.patient,
        CASE
            WHEN v.created_at = MIN(v.created_at) OVER (PARTITION BY v.patient, v.source_schema)
            THEN 'New' ELSE 'Returning'
        END AS patient_type
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    WHERE v.source_schema = 'tenri'
),
tenri_pct AS (
    SELECT
        patient_type,
        ROUND(COUNT(DISTINCT patient) * 100.0
              / NULLIF(SUM(COUNT(DISTINCT patient)) OVER (), 0), 1) AS benchmark_pct
    FROM tenri_classified
    GROUP BY patient_type
)
SELECT
    f.patient_type,
    f.facility_pct,
    COALESCE(t.benchmark_pct, 0) AS benchmark_pct,
    ROUND(f.facility_pct - COALESCE(t.benchmark_pct, 0), 1) AS gap_pp
FROM facility_pct f
LEFT JOIN tenri_pct t ON f.patient_type = t.patient_type
ORDER BY f.patient_type
"""
    return run_query(sql)


def load_ltfu_demographics(filters: dict, run_query) -> pd.DataFrame:
    """Retention Section B: LTFU rate by Age group, Payer, Gender.
    Returns DIMENSION, CATEGORY, LTFU_RATE_PCT."""
    df = load_ltfu_correlation(filters, run_query)
    if df.empty:
        return df
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"factor": "dimension", "dimension": "category"})
    df["dimension"] = df["dimension"].replace({
        "Age Group": "Age group",
        "Gender":    "Gender",
        "Payer":     "Payer",
    })
    return df[["dimension", "category", "ltfu_rate_pct"]].copy()
