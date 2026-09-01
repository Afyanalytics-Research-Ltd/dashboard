"""
sph/clinical_activity_module/ca_queries.py
============================================
All SQL for the SPH Clinical Activity tab, rebuilt to match
sph_clinical_activity_build_spec.md's exact Section 5 query map
(function names, return columns, taxonomies).

Every function is @st.cache_data(ttl=3600), returns a pd.DataFrame, and
never raises — an empty DataFrame signals "not queryable" and every
render function in ca_views.py must show an empty state for it.

Two taxonomy items in the spec don't exist in this schema and are
substituted with the closest real signal (documented at each function):
  - SSI comorbidity: spec asks for Obesity + Malnutrition; this schema
    has no such flags. Substituted with Cardiac condition + Anaemia.
  - "Current system" = source_system = 'EMR_V2' throughout. EMR_V1
    readmission detection is confirmed structurally broken (returns 0%
    regardless of ward) — every readmission/blind-spot/SSI-timing query
    below scopes to EMR_V2 only. LOS queries also exclude EMR_V1 by the
    same discharge-date artifact rule used elsewhere in this codebase.
"""

import decimal

import pandas as pd
import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clinicals.opd_ipd_module.queries import _run as _run_raw

_CURRENT_SYSTEM_FILTER = "source_system = 'EMR_V2'"
_ARTIFACT_DATE = "2025-09-01"


def _run(sql: str) -> pd.DataFrame:
    """Wraps the shared _run() and coerces decimal.Decimal columns (as
    returned by the Snowflake connector for ROUND()/AVG()/etc.) to float64.
    Without this, Decimal values mix badly with the plain Python floats
    used elsewhere in this module (e.g. static benchmark ceilings) and
    raise TypeError on any +/-/*// between the two."""
    df = _run_raw(sql)
    if df is None or df.empty:
        return df
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, decimal.Decimal)).any():
            df[col] = df[col].astype(float)
    return df

_ADMISSIONS_BASE_CTE = f"""
admissions_base AS (
    SELECT
        visit_id, patient_id, source_system, ward, ward_type,
        age_years, LOWER(gender) AS gender, admission_date, discharge_date,
        DATEDIFF('day', admission_date, discharge_date) AS los_days
    FROM HOSPITALS.STAGING.STG_ADMISSIONS
    WHERE discharge_date IS NOT NULL
      AND DATEDIFF('day', admission_date, discharge_date) BETWEEN 0 AND 365
      AND discharge_date != '{_ARTIFACT_DATE}'
)
"""

# 31-90 day return-type classification — used by both get_ca_blind_spot_type()
# (the donut) and get_ca_delayed_complications() (the named-diagnosis
# drill-down), interpolated identically into both so the two never drift
# apart. 7 categories, replacing the earlier 4-category version:
# Infection / sepsis, Post-surgical complication, Planned hardware removal,
# Staged / revision surgery, New trauma / re-fracture, Unrelated medical,
# Other / unclear. Classifies on the RETURN visit's diagnosis text only —
# the index-visit diagnosis is available in window_31_90 as
# index_diagnosis_label but is not part of the bucketing rule.
_RETURN_TYPE_CASE = """
    CASE
        -- Infection / sepsis
        WHEN return_dx_text LIKE '%septic%' OR return_dx_text LIKE '%sepsis%'
             OR return_dx_text LIKE '%infection%' OR return_dx_text LIKE '%cellulitis%'
             OR return_dx_text LIKE '%dehiscen%' OR return_dx_text LIKE '%surgical site%'
            THEN 'Infection / sepsis'
        -- Non-infectious post-surgical complications
        WHEN return_dx_text LIKE '%thrombos%' OR return_dx_text LIKE '%dvt%'
             OR return_dx_text LIKE '%embolism%' OR return_dx_text LIKE '%pneumonia%'
             OR return_dx_text LIKE '%anaemia%' OR return_dx_text LIKE '%anemia%'
             OR return_dx_text LIKE '%stiffness%' OR return_dx_text LIKE '%failed%'
             OR return_dx_text LIKE '%dislocat%' OR return_dx_text LIKE '%nonunion%'
             OR return_dx_text LIKE '%malunion%' OR return_dx_text LIKE '%symptomatic implant%'
             OR return_dx_text LIKE '%symtomatic implant%'
            THEN 'Post-surgical complication'
        -- Hardware removal / management
        WHEN return_dx_text LIKE '%implant removal%' OR return_dx_text LIKE '%pin removal%'
             OR return_dx_text LIKE '%exofix removal%' OR return_dx_text LIKE '%illizarov removal%'
             OR return_dx_text LIKE '%screw removal%' OR return_dx_text LIKE '%hardware removal%'
            THEN 'Planned hardware removal'
        -- Staged / revision surgery
        WHEN return_dx_text LIKE '%staged%' OR return_dx_text LIKE '%planned%'
             OR return_dx_text LIKE '%elective%' OR return_dx_text LIKE '%review%'
             OR return_dx_text LIKE '%revision%' OR return_dx_text LIKE '%skin graft%'
             OR return_dx_text LIKE '%wash out%' OR return_dx_text LIKE '%closure%'
             OR return_dx_text LIKE '%plating%' OR return_dx_text LIKE '%nailing%'
             OR return_dx_text LIKE '%fixation%' OR return_dx_text LIKE '%buttress%'
             OR return_dx_text LIKE '%arthroplasty%' OR return_dx_text LIKE '%replacement%'
             OR return_dx_text LIKE '%excision%' OR return_dx_text LIKE '%sequestec%'
            THEN 'Staged / revision surgery'
        -- New trauma or re-fracture
        WHEN return_dx_text LIKE '%fracture%' OR return_dx_text LIKE '%injury%'
             OR return_dx_text LIKE '%compound%'
            THEN 'New trauma / re-fracture'
        -- Non-surgical / medical
        WHEN return_dx_text LIKE '%medical management%' OR return_dx_text LIKE '%circumcision%'
             OR return_dx_text LIKE '%hydrocele%' OR return_dx_text LIKE '%ganglion%'
            THEN 'Unrelated medical'
        ELSE 'Other / unclear'
    END
"""

_READMIT_PAIRS_CTE = """
readmit_pairs AS (
    SELECT
        a1.visit_id AS index_visit_id,
        a2.visit_id AS readmit_visit_id,
        a1.patient_id, a1.ward AS index_ward, a1.age_years, a1.gender,
        a1.discharge_date AS index_discharge_date, a1.los_days AS index_los_days,
        a2.admission_date AS readmit_admission_date, a2.los_days AS readmit_los_days,
        DATEDIFF('day', a1.discharge_date, a2.admission_date) AS days_to_readmission
    FROM admissions_base a1
    JOIN admissions_base a2
        ON a1.patient_id = a2.patient_id AND a1.source_system = a2.source_system
        AND a2.admission_date > a1.discharge_date
        AND DATEDIFF('day', a1.discharge_date, a2.admission_date) <= 30
        AND a1.source_system = 'EMR_V2'
    QUALIFY ROW_NUMBER() OVER (PARTITION BY a1.visit_id ORDER BY a2.admission_date ASC) = 1
)
"""

_READMIT_DX_CTE = """
readmit_with_dx AS (
    SELECT
        rp.*,
        LOWER(COALESCE(d1.diagnosis_name_expanded, '') || ' ' || COALESCE(d1.icd10_names, '')) AS index_dx_text,
        LOWER(COALESCE(d2.diagnosis_name_expanded, '') || ' ' || COALESCE(d2.icd10_names, '')) AS readmit_dx_text,
        COALESCE(d2.diagnosis_name_expanded, 'Unspecified') AS readmit_dx_label
    FROM readmit_pairs rp
    LEFT JOIN HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED d1 ON d1.visit_id = rp.index_visit_id
    LEFT JOIN HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED d2 ON d2.visit_id = rp.readmit_visit_id
)
"""

_SSI_FLAG_SQL = """
CASE
    WHEN clean_dx_text LIKE '%septic surgical site%' OR clean_dx_text LIKE '%surgical site infection%'
         OR clean_dx_text LIKE '%septic wound%' OR clean_dx_text LIKE '%septic implant%'
        THEN 1
    WHEN clean_dx_text LIKE '%septic%'
         AND (clean_dx_text LIKE '%replacement%' OR clean_dx_text LIKE '%arthroplasty%'
              OR clean_dx_text LIKE '%nail%' OR clean_dx_text LIKE '%implant%'
              OR clean_dx_text LIKE '%prosthesis%' OR clean_dx_text LIKE '%antibiotic cement%'
              OR clean_dx_text LIKE '%amputation%' OR clean_dx_text LIKE '%stump%')
         AND clean_dx_text NOT LIKE '%tonsil%'
         AND clean_dx_text NOT LIKE '%pressure sore%'
         AND clean_dx_text NOT LIKE '%decubit%'
        THEN 1
    ELSE 0
END
"""

_SURGICAL_CATEGORY_CASE = """
CASE
    WHEN clean_dx_text LIKE '%total hip replacement%' OR clean_dx_text LIKE '%total knee replacement%'
         OR clean_dx_text LIKE '%arthroplasty%'
        THEN 'Elective Joint Replacement'
    WHEN clean_dx_text LIKE '%spine%' OR clean_dx_text LIKE '%vertebral%'
         OR clean_dx_text LIKE '%disc%' OR clean_dx_text LIKE '%laminectomy%'
        THEN 'Complex Spine and Back Pain Care'
    WHEN clean_dx_text LIKE '%hernia%'
        THEN 'Clean General Surgery'
    WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%compound%'
         OR clean_dx_text LIKE '%open%' OR clean_dx_text LIKE '%degloving%'
         OR clean_dx_text LIKE '%exfix%' OR clean_dx_text LIKE '%ex-fix%'
         OR clean_dx_text LIKE '%locking plate%' OR clean_dx_text LIKE '%tension band wire%'
         OR clean_dx_text LIKE '%ankle pinning%' OR clean_dx_text LIKE '%tibia nail%'
         OR clean_dx_text LIKE '%femur nail%' OR clean_dx_text LIKE '%amputation%'
        THEN 'MSK Trauma / Hip Fracture'
    ELSE NULL
END
"""

# Published benchmark ceilings — static reference values, per spec.
_BENCHMARKS = {
    "Elective Joint Replacement": 1.5,
    "Complex Spine and Back Pain Care":      4.0,
    "MSK Trauma / Hip Fracture":  9.0,
    "Clean General Surgery":      2.5,
}

# Readmission-type taxonomy per spec Q_CA_3a — refined after manually
# reviewing what "Other / unclear" actually contained (see judgment calls
# below). Priority order matters: more specific/severe categories are
# checked first so e.g. a dislocation isn't miscounted as routine staged
# follow-up just because "revision" also appears in the text.
#
# Judgment calls baked into this ordering:
#   - "old"/"healed" fracture mentions are treated as a staged follow-up
#     visit for a known injury, not a new/unrelated fracture.
#   - Cataract and cleft lip are flagged as likely ward misattribution —
#     neither is an orthopaedic or general-surgical-ward condition, so a
#     record landing on SIMBA/NDOVU most likely reflects a data-entry
#     error, not a genuine readmission to that ward.
#   - "medical management", "sphincterotomy", truncated text, and similar
#     have no stated clinical link back to the index admission and are
#     left in Other / unclear rather than guessed at.
#   - Misspelled terms (e.g. "osteomylitis" for osteomyelitis,
#     "sequescretomy" for sequestrectomy) will not match — free-text
#     typos are a known, unfixed gap in keyword-based classification.
_READMISSION_TYPE_CASE = """
CASE
    WHEN readmit_dx_text LIKE '%dislocat%' OR readmit_dx_text LIKE '%failed implant%'
         OR readmit_dx_text LIKE '%nonunion%' OR readmit_dx_text LIKE '%non-union%'
         OR readmit_dx_text LIKE '%loosen%' OR readmit_dx_text LIKE '%hardware failure%'
         OR readmit_dx_text LIKE '%revision%'
        THEN 'Hardware / implant complication'
    WHEN readmit_dx_text LIKE '%dehiscence%' OR readmit_dx_text LIKE '%wound breakdown%'
         OR readmit_dx_text LIKE '%degloving%' OR readmit_dx_text LIKE '%necrotic%'
         OR readmit_dx_text LIKE '%septic%' OR readmit_dx_text LIKE '%infection%'
         OR readmit_dx_text LIKE '%cellulitis%' OR readmit_dx_text LIKE '%abscess%'
         OR readmit_dx_text LIKE '%osteomyeli%' OR readmit_dx_text LIKE '%sequestr%'
         OR readmit_dx_text LIKE '%gangren%' OR readmit_dx_text LIKE '%incision and drainage%'
         OR readmit_dx_text LIKE '%wound closure%'
        THEN 'Wound / infection complication'
    WHEN readmit_dx_text LIKE '%disarticulation%' OR readmit_dx_text LIKE '%amputat%'
        THEN 'Amputation-related'
    WHEN readmit_dx_text LIKE '%cataract%' OR readmit_dx_text LIKE '%cleft lip%'
         OR readmit_dx_text LIKE '%cleft palate%'
        THEN 'Likely ward misattribution'
    WHEN readmit_dx_text LIKE '%pain%'
        THEN 'Pain management'
    WHEN readmit_dx_text LIKE '%bowel%' OR readmit_dx_text LIKE '%gastro%'
         OR readmit_dx_text LIKE '%nausea%' OR readmit_dx_text LIKE '%vomit%'
         OR readmit_dx_text LIKE '%diarrh%' OR readmit_dx_text LIKE '%metabolic%'
         OR readmit_dx_text LIKE '%electrolyte%'
        THEN 'GI / metabolic'
    WHEN readmit_dx_text LIKE '%cardiac%' OR readmit_dx_text LIKE '%arrhythmia%'
         OR readmit_dx_text LIKE '%myocardial%' OR readmit_dx_text LIKE '%chest pain%'
        THEN 'Cardiac'
    WHEN readmit_dx_text LIKE '%respiratory%' OR readmit_dx_text LIKE '%pneumonia%'
         OR readmit_dx_text LIKE '%copd%' OR readmit_dx_text LIKE '%asthma%'
        THEN 'Respiratory'
    WHEN (readmit_dx_text LIKE '%fracture%' AND readmit_dx_text NOT LIKE '%old%'
          AND readmit_dx_text NOT LIKE '%healed%')
         OR readmit_dx_text LIKE '%tear%' OR readmit_dx_text LIKE '%osteoarthritis%'
         OR readmit_dx_text LIKE '%oa knee%' OR readmit_dx_text LIKE '%oa hip%'
        THEN 'New / unrelated fracture or injury'
    WHEN readmit_dx_text LIKE '%removal%' OR readmit_dx_text LIKE '%plating%'
         OR readmit_dx_text LIKE '%nailing%' OR readmit_dx_text LIKE '% nail%'
         OR readmit_dx_text LIKE '%ex-fix%' OR readmit_dx_text LIKE '%exfix%'
         OR readmit_dx_text LIKE '%k-wire%' OR readmit_dx_text LIKE '%debridement%'
         OR readmit_dx_text LIKE '%injection%' OR readmit_dx_text LIKE '%endoscopy%'
         OR readmit_dx_text LIKE '%cement%' OR readmit_dx_text LIKE '%bone nibling%'
         OR readmit_dx_text LIKE '%mua%' OR readmit_dx_text LIKE '%skin graft%'
         OR readmit_dx_text LIKE '%healed fracture%' OR readmit_dx_text LIKE '%old %fracture%'
        THEN 'Planned follow-up / staged procedure'
    ELSE 'Other / unclear'
END
"""

_COMORBIDITY_COLS = {
    "Diabetes":     "has_diabetes",
    "Hypertension": "has_hypertension",
    "CKD":          "has_renal_condition",
    # Substituted for the spec's Obesity / Malnutrition — this schema has
    # no such flags. Cardiac condition and Anaemia are the closest real
    # comorbidity signals available on STG_SPH_DIAGNOSIS_ENRICHED.
    "Cardiac condition": "has_cardiac_condition",
    "Anaemia":           "has_anaemia",
}


# ---------------------------------------------------------------------------
# Q_CA_KPI — Overview KPIs
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_ca_overview_kpis() -> pd.DataFrame:
    sql = f"""
    WITH {_ADMISSIONS_BASE_CTE},
    {_READMIT_PAIRS_CTE},
    monthly AS (
        SELECT
            DATE_TRUNC('month', ab.discharge_date) AS month,
            COUNT(DISTINCT ab.visit_id) AS discharges,
            COUNT(DISTINCT rp.index_visit_id) AS readmissions
        FROM admissions_base ab
        LEFT JOIN readmit_pairs rp ON rp.index_visit_id = ab.visit_id
        WHERE ab.{_CURRENT_SYSTEM_FILTER} AND ab.discharge_date < DATE_TRUNC('month', CURRENT_DATE())
        GROUP BY month
    ),
    monthly_rate AS (
        SELECT month, ROUND(100.0 * readmissions / NULLIF(discharges, 0), 2) AS rate_pct
        FROM monthly
    ),
    ssi_by_cat AS (
        SELECT
            {_SURGICAL_CATEGORY_CASE} AS surgical_category,
            {_SSI_FLAG_SQL} AS is_ssi
        FROM (
            SELECT LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
            FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
        )
    ),
    ssi_rates AS (
        SELECT surgical_category, ROUND(100.0 * SUM(is_ssi) / COUNT(*), 2) AS ssi_rate
        FROM ssi_by_cat
        WHERE surgical_category IS NOT NULL
        GROUP BY surgical_category
    ),
    blind_spot AS (
        SELECT COUNT(*) AS n
        FROM (
            SELECT a1.visit_id
            FROM admissions_base a1
            JOIN admissions_base a2
                ON a1.patient_id = a2.patient_id AND a1.source_system = a2.source_system
                AND a2.admission_date > a1.discharge_date
                AND a1.{_CURRENT_SYSTEM_FILTER}
            QUALIFY ROW_NUMBER() OVER (PARTITION BY a1.visit_id ORDER BY a2.admission_date ASC) = 1
                AND DATEDIFF('day', a1.discharge_date, a2.admission_date) BETWEEN 31 AND 90
        )
    )
    SELECT
        (SELECT ROUND(100.0 * SUM(readmissions) / NULLIF(SUM(discharges), 0), 2) FROM monthly) AS READMISSION_RATE,
        (SELECT MIN(rate_pct) FROM monthly_rate) AS READMISSION_RATE_MIN,
        (SELECT MAX(rate_pct) FROM monthly_rate) AS READMISSION_RATE_MAX,
        (SELECT ROUND(AVG(los_days), 1) FROM admissions_base WHERE {_CURRENT_SYSTEM_FILTER}) AS AVG_LOS,
        (SELECT ROUND(MEDIAN(los_days), 1) FROM admissions_base WHERE {_CURRENT_SYSTEM_FILTER}) AS MEDIAN_LOS,
        (SELECT MAX(ssi_rate) FROM ssi_rates) AS WORST_SSI_RATE,
        (SELECT surgical_category FROM ssi_rates ORDER BY ssi_rate DESC LIMIT 1) AS WORST_SSI_CATEGORY,
        (SELECT n FROM blind_spot) AS BLIND_SPOT_COUNT
    """
    df = _run(sql)
    if df.empty:
        return df
    df["WORST_SSI_BENCHMARK"] = df["WORST_SSI_CATEGORY"].map(lambda c: _BENCHMARKS.get(c, 0.0))
    return df


# ---------------------------------------------------------------------------
# Q_CA_1 — Monthly readmission trend
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_ca_monthly_readmission() -> pd.DataFrame:
    sql = f"""
    WITH {_ADMISSIONS_BASE_CTE},
    {_READMIT_PAIRS_CTE}
    SELECT
        TO_VARCHAR(DATE_TRUNC('month', ab.discharge_date), 'Mon YYYY') AS VISIT_MONTH,
        DATE_TRUNC('month', ab.discharge_date)                        AS SORT_MONTH,
        COUNT(DISTINCT ab.visit_id)                                   AS DISCHARGE_COUNT,
        ROUND(100.0 * COUNT(DISTINCT rp.index_visit_id)
              / NULLIF(COUNT(DISTINCT ab.visit_id), 0), 2)            AS READMISSION_RATE
    FROM admissions_base ab
    LEFT JOIN readmit_pairs rp ON rp.index_visit_id = ab.visit_id
    WHERE ab.{_CURRENT_SYSTEM_FILTER} AND ab.discharge_date < DATE_TRUNC('month', CURRENT_DATE())
    GROUP BY SORT_MONTH, VISIT_MONTH
    ORDER BY SORT_MONTH ASC
    """
    return _run(sql)


# Confirmed by visual inspection of the monthly trend chart (Section 1) —
# these are the single-month rate spikes worth drilling into. Update this
# list if a new spike month becomes visible as more data lands.
_SPIKE_MONTHS = ["2025-05-01", "2025-07-01", "2025-11-01", "2026-03-01", "2026-05-01"]


@st.cache_data(ttl=3600)
def get_ca_spike_month_drilldown() -> pd.DataFrame:
    months_sql = ",".join(f"'{m}'" for m in _SPIKE_MONTHS)
    sql = f"""
    WITH {_ADMISSIONS_BASE_CTE},
    {_READMIT_PAIRS_CTE},
    {_READMIT_DX_CTE},
    spike_readmissions AS (
        SELECT
            TO_VARCHAR(DATE_TRUNC('month', index_discharge_date), 'Mon YYYY') AS spike_month,
            DATE_TRUNC('month', index_discharge_date)                        AS sort_month,
            index_ward AS ward,
            {_READMISSION_TYPE_CASE} AS readmission_type,
            readmit_dx_label
        FROM readmit_with_dx
        WHERE DATE_TRUNC('month', index_discharge_date) IN ({months_sql})
    )
    SELECT
        spike_month       AS SPIKE_MONTH,
        sort_month        AS SORT_MONTH,
        ward              AS WARD,
        readmission_type  AS READMISSION_TYPE,
        readmit_dx_label  AS READMIT_DX_LABEL,
        COUNT(*)          AS READMISSION_COUNT
    FROM spike_readmissions
    GROUP BY spike_month, sort_month, ward, readmission_type, readmit_dx_label
    ORDER BY sort_month, readmission_count DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# Q_CA_2 — Ward analysis
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_ca_ward_readmission_rates() -> pd.DataFrame:
    sql = f"""
    WITH {_ADMISSIONS_BASE_CTE},
    {_READMIT_PAIRS_CTE}
    SELECT
        ab.ward                                                       AS WARD,
        ROUND(100.0 * COUNT(DISTINCT rp.index_visit_id)
              / NULLIF(COUNT(DISTINCT ab.visit_id), 0), 2)            AS READMISSION_RATE,
        COUNT(DISTINCT ab.visit_id)                                   AS DISCHARGE_COUNT
    FROM admissions_base ab
    LEFT JOIN readmit_pairs rp ON rp.index_visit_id = ab.visit_id
    WHERE ab.{_CURRENT_SYSTEM_FILTER} AND ab.ward IS NOT NULL
    GROUP BY ab.ward
    HAVING COUNT(DISTINCT ab.visit_id) >= 20
    ORDER BY READMISSION_RATE DESC
    """
    return _run(sql)


@st.cache_data(ttl=3600)
def get_ca_ward_readmission_cause() -> pd.DataFrame:
    sql = f"""
    WITH {_ADMISSIONS_BASE_CTE},
    {_READMIT_PAIRS_CTE},
    {_READMIT_DX_CTE},
    classified AS (
        SELECT
            index_ward,
            CASE
                WHEN readmit_dx_text LIKE '%implant removal%' OR readmit_dx_text LIKE '%ex-fix removal%'
                     OR readmit_dx_text LIKE '%staged%' OR readmit_dx_text LIKE '%planned%'
                     OR readmit_dx_text LIKE '%elective%' OR readmit_dx_text LIKE '%follow up%'
                     OR readmit_dx_text LIKE '%review%'
                    THEN 'Expected'
                WHEN readmit_dx_text LIKE '%dehiscence%' OR readmit_dx_text LIKE '%wound%'
                     OR readmit_dx_text LIKE '%pain%' OR readmit_dx_text LIKE '%septic%'
                     OR readmit_dx_text LIKE '%infection%'
                    THEN 'Potentially preventable'
                ELSE 'Unclear / other'
            END AS cause_type
        FROM readmit_with_dx
        WHERE index_ward IS NOT NULL
    )
    SELECT
        index_ward       AS WARD,
        cause_type       AS CAUSE_TYPE,
        COUNT(*)         AS READMISSION_COUNT
    FROM classified
    GROUP BY index_ward, cause_type
    ORDER BY index_ward, cause_type
    """
    return _run(sql)


@st.cache_data(ttl=3600)
def get_ca_ward_top_diagnoses() -> pd.DataFrame:
    """
    Readmission diagnoses by ward, grouped into the same clinical
    categories as _READMISSION_TYPE_CASE (Hardware / implant complication,
    Wound / infection complication, Amputation-related, Likely ward
    misattribution, Pain management, GI / metabolic, Cardiac, Respiratory,
    New / unrelated fracture or injury, Planned follow-up / staged
    procedure, Other / unclear), plus a separate "No diagnosis recorded"
    category broken out ahead of the classifier — readmits with zero linked
    diagnosis rows are a documentation gap, not a clinically ambiguous
    case, and folding them into "Other / unclear" hid how much of that
    bucket was really just missing data. Grouping by category instead of
    raw free-text label also means every readmission is represented; the
    old free-text grouping only surfaced whichever exact phrasing
    happened to repeat into a top-5 cutoff, silently dropping most
    readmissions at high-volume wards where diagnosis text is mostly
    unique per patient.
    """
    sql = f"""
    WITH {_ADMISSIONS_BASE_CTE},
    {_READMIT_PAIRS_CTE},
    {_READMIT_DX_CTE},
    categorized AS (
        SELECT
            index_ward AS WARD,
            CASE
                WHEN TRIM(readmit_dx_text) = '' THEN 'No diagnosis recorded'
                ELSE {_READMISSION_TYPE_CASE}
            END AS DIAGNOSIS_LABEL
        FROM readmit_with_dx
        WHERE index_ward IS NOT NULL
    )
    SELECT WARD, DIAGNOSIS_LABEL, COUNT(*) AS READMISSION_COUNT
    FROM categorized
    GROUP BY WARD, DIAGNOSIS_LABEL
    ORDER BY WARD, READMISSION_COUNT DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# Q_CA_3 — Readmission type / clinical area
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_ca_readmission_type_breakdown() -> pd.DataFrame:
    sql = f"""
    WITH {_ADMISSIONS_BASE_CTE},
    {_READMIT_PAIRS_CTE},
    {_READMIT_DX_CTE},
    classified AS (
        SELECT {_READMISSION_TYPE_CASE} AS readmission_type
        FROM readmit_with_dx
    )
    SELECT
        readmission_type                                            AS READMISSION_TYPE,
        COUNT(*)                                                    AS COUNT,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)          AS PCT
    FROM classified
    GROUP BY readmission_type
    ORDER BY COUNT DESC
    """
    return _run(sql)


@st.cache_data(ttl=3600)
def get_ca_readmission_area() -> pd.DataFrame:
    sql = f"""
    WITH {_ADMISSIONS_BASE_CTE},
    {_READMIT_PAIRS_CTE},
    visit_burden_groups AS (
        SELECT DISTINCT visit_id, TRIM(bg.value::STRING) AS burden_group
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
        LATERAL FLATTEN(input => SPLIT(burden_group, '|')) bg
    ),
    groups AS (
        SELECT visit_id, LISTAGG(DISTINCT burden_group, '|') WITHIN GROUP (ORDER BY burden_group) AS groups
        FROM visit_burden_groups GROUP BY visit_id
    ),
    matched AS (
        SELECT
            CASE
                WHEN ig.groups IS NULL OR rg.groups IS NULL THEN 'Unknown'
                WHEN ARRAY_SIZE(ARRAY_INTERSECTION(SPLIT(ig.groups, '|'), SPLIT(rg.groups, '|'))) > 0
                    THEN 'Same clinical area'
                ELSE 'Different clinical area'
            END AS area_group
        FROM readmit_pairs rp
        LEFT JOIN groups ig ON ig.visit_id = rp.index_visit_id
        LEFT JOIN groups rg ON rg.visit_id = rp.readmit_visit_id
    )
    SELECT
        area_group                                                  AS AREA_GROUP,
        COUNT(*)                                                    AS COUNT,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)          AS PCT
    FROM matched
    GROUP BY area_group
    ORDER BY COUNT DESC
    """
    return _run(sql)


@st.cache_data(ttl=3600)
def get_ca_readmission_top_by_area() -> pd.DataFrame:
    sql = f"""
    WITH {_ADMISSIONS_BASE_CTE},
    {_READMIT_PAIRS_CTE},
    visit_burden_groups AS (
        SELECT DISTINCT visit_id, TRIM(bg.value::STRING) AS burden_group
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
        LATERAL FLATTEN(input => SPLIT(burden_group, '|')) bg
    ),
    groups AS (
        SELECT visit_id, LISTAGG(DISTINCT burden_group, '|') WITHIN GROUP (ORDER BY burden_group) AS groups
        FROM visit_burden_groups GROUP BY visit_id
    ),
    matched AS (
        SELECT
            rp.readmit_visit_id,
            CASE
                WHEN ig.groups IS NULL OR rg.groups IS NULL THEN 'Unknown'
                WHEN ARRAY_SIZE(ARRAY_INTERSECTION(SPLIT(ig.groups, '|'), SPLIT(rg.groups, '|'))) > 0
                    THEN 'Same clinical area'
                ELSE 'Different clinical area'
            END AS area_group
        FROM readmit_pairs rp
        LEFT JOIN groups ig ON ig.visit_id = rp.index_visit_id
        LEFT JOIN groups rg ON rg.visit_id = rp.readmit_visit_id
    ),
    with_dx AS (
        SELECT m.area_group, COALESCE(d.diagnosis_name_expanded, 'Unspecified') AS diagnosis
        FROM matched m
        LEFT JOIN HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED d ON d.visit_id = m.readmit_visit_id
        WHERE m.area_group IN ('Same clinical area', 'Different clinical area')
    ),
    ranked AS (
        SELECT
            area_group                                                        AS AREA_GROUP,
            diagnosis                                                         AS DIAGNOSIS_LABEL,
            COUNT(*)                                                          AS COUNT,
            ROW_NUMBER() OVER (PARTITION BY area_group ORDER BY COUNT(*) DESC) AS rn
        FROM with_dx
        GROUP BY area_group, diagnosis
    )
    SELECT AREA_GROUP, DIAGNOSIS_LABEL, COUNT
    FROM ranked WHERE rn <= 5
    ORDER BY AREA_GROUP, COUNT DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# Q_CA_4 — Age × complication profile
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_ca_age_complication() -> pd.DataFrame:
    sql = f"""
    WITH {_ADMISSIONS_BASE_CTE},
    {_READMIT_PAIRS_CTE},
    {_READMIT_DX_CTE},
    classified AS (
        SELECT
            CASE
                WHEN age_years IS NULL OR age_years < 0 THEN NULL
                WHEN age_years < 18 THEN '<18'
                WHEN age_years < 25 THEN '18-24'
                WHEN age_years < 35 THEN '25-34'
                WHEN age_years < 45 THEN '35-44'
                WHEN age_years < 55 THEN '45-54'
                WHEN age_years < 65 THEN '55-64'
                ELSE '65+'
            END                                                    AS age_group,
            INITCAP(gender)                                        AS gender,
            {_READMISSION_TYPE_CASE}                                AS complication_type
        FROM readmit_with_dx
        WHERE LEFT(gender, 1) IN ('m', 'f')
    )
    SELECT
        age_group           AS AGE_GROUP,
        complication_type    AS COMPLICATION_TYPE,
        gender               AS GENDER,
        COUNT(*)             AS READMISSION_COUNT
    FROM classified
    WHERE age_group IS NOT NULL
    GROUP BY age_group, complication_type, gender
    ORDER BY age_group, complication_type
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# Q_CA_5 — 31-90 day blind spot
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_ca_blind_spot() -> pd.DataFrame:
    sql = f"""
    WITH {_ADMISSIONS_BASE_CTE},
    all_pairs AS (
        SELECT DATEDIFF('day', a1.discharge_date, a2.admission_date) AS days_to_return
        FROM admissions_base a1
        JOIN admissions_base a2
            ON a1.patient_id = a2.patient_id AND a1.source_system = a2.source_system
            AND a2.admission_date > a1.discharge_date AND a1.{_CURRENT_SYSTEM_FILTER}
        QUALIFY ROW_NUMBER() OVER (PARTITION BY a1.visit_id ORDER BY a2.admission_date ASC) = 1
    )
    SELECT
        CASE
            WHEN days_to_return <= 7 THEN '0-7 days'
            WHEN days_to_return <= 14 THEN '8-14 days'
            WHEN days_to_return <= 30 THEN '15-30 days'
            WHEN days_to_return <= 60 THEN '31-60 days'
            WHEN days_to_return <= 90 THEN '61-90 days'
            WHEN days_to_return <= 180 THEN '91-180 days'
            ELSE '180+ days'
        END                                                          AS GAP_BUCKET,
        CASE
            WHEN days_to_return <= 7 THEN 1 WHEN days_to_return <= 14 THEN 2
            WHEN days_to_return <= 30 THEN 3 WHEN days_to_return <= 60 THEN 4
            WHEN days_to_return <= 90 THEN 5 WHEN days_to_return <= 180 THEN 6
            ELSE 7
        END                                                          AS BUCKET_ORDER,
        COUNT(*)                                                     AS PATIENT_COUNT,
        days_to_return BETWEEN 31 AND 90                             AS IS_BLIND_SPOT
    FROM all_pairs
    GROUP BY GAP_BUCKET, BUCKET_ORDER, IS_BLIND_SPOT
    ORDER BY BUCKET_ORDER
    """
    return _run(sql)


@st.cache_data(ttl=3600)
def get_ca_blind_spot_type() -> pd.DataFrame:
    sql = f"""
    WITH {_ADMISSIONS_BASE_CTE},
    return_pairs AS (
        SELECT a1.visit_id AS index_visit_id, a2.visit_id AS return_visit_id,
               a1.patient_id, a1.source_system, a2.ward AS return_ward,
               DATEDIFF('day', a1.discharge_date, a2.admission_date) AS days_to_return
        FROM admissions_base a1
        JOIN admissions_base a2
            ON a1.patient_id = a2.patient_id AND a1.source_system = a2.source_system
            AND a2.admission_date > a1.discharge_date AND a1.{_CURRENT_SYSTEM_FILTER}
        QUALIFY ROW_NUMBER() OVER (PARTITION BY a1.visit_id ORDER BY a2.admission_date ASC) = 1
    ),
    window_31_90 AS (
        SELECT rp.*,
               COALESCE(d_idx.diagnosis_name_expanded, 'Unspecified') AS index_diagnosis_label,
               COALESCE(d_ret.diagnosis_name_expanded, 'Unspecified') AS return_diagnosis_label,
               LOWER(COALESCE(d_ret.diagnosis_name_expanded, '') || ' ' || COALESCE(d_ret.icd10_names, '')) AS return_dx_text
        FROM return_pairs rp
        LEFT JOIN HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED d_idx ON d_idx.visit_id = rp.index_visit_id
        LEFT JOIN HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED d_ret ON d_ret.visit_id = rp.return_visit_id
        WHERE rp.days_to_return BETWEEN 31 AND 90
    ),
    classified AS (
        SELECT {_RETURN_TYPE_CASE} AS return_type
        FROM window_31_90
    )
    SELECT
        return_type                                                AS RETURN_TYPE,
        COUNT(*)                                                   AS COUNT,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)         AS PCT
    FROM classified
    GROUP BY return_type
    ORDER BY COUNT DESC
    """
    return _run(sql)


@st.cache_data(ttl=3600)
def get_ca_delayed_complications(return_type: str = "Infection / sepsis") -> pd.DataFrame:
    """Named diagnoses behind one 31-90 day return-type bucket. Defaults to
    'Infection / sepsis' — the closest single-category match to the old
    'Delayed complication' bucket this replaced (which combined septic/
    infection/wound/failed/dislocation keywords now split across
    'Infection / sepsis' and 'Post-surgical complication'). Pass a
    different return_type to drill into any of the other 6 categories from
    _RETURN_TYPE_CASE instead."""
    sql = f"""
    WITH {_ADMISSIONS_BASE_CTE},
    return_pairs AS (
        SELECT a1.visit_id AS index_visit_id, a2.visit_id AS return_visit_id,
               a1.patient_id, a1.source_system, a2.ward AS return_ward,
               DATEDIFF('day', a1.discharge_date, a2.admission_date) AS days_to_return
        FROM admissions_base a1
        JOIN admissions_base a2
            ON a1.patient_id = a2.patient_id AND a1.source_system = a2.source_system
            AND a2.admission_date > a1.discharge_date AND a1.{_CURRENT_SYSTEM_FILTER}
        QUALIFY ROW_NUMBER() OVER (PARTITION BY a1.visit_id ORDER BY a2.admission_date ASC) = 1
    ),
    window_31_90 AS (
        SELECT rp.*,
               COALESCE(d_idx.diagnosis_name_expanded, 'Unspecified') AS index_diagnosis_label,
               COALESCE(d_ret.diagnosis_name_expanded, 'Unspecified') AS return_diagnosis_label,
               LOWER(COALESCE(d_ret.diagnosis_name_expanded, '') || ' ' || COALESCE(d_ret.icd10_names, '')) AS return_dx_text
        FROM return_pairs rp
        LEFT JOIN HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED d_idx ON d_idx.visit_id = rp.index_visit_id
        LEFT JOIN HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED d_ret ON d_ret.visit_id = rp.return_visit_id
        WHERE rp.days_to_return BETWEEN 31 AND 90
    ),
    classified AS (
        SELECT *, {_RETURN_TYPE_CASE} AS return_type
        FROM window_31_90
    )
    SELECT
        return_diagnosis_label   AS COMPLICATION_LABEL,
        COUNT(*)                 AS PATIENT_COUNT
    FROM classified
    WHERE return_type = '{return_type}'
    GROUP BY return_diagnosis_label
    ORDER BY PATIENT_COUNT DESC
    LIMIT 8
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# Q_CA_6 — Length of stay
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_ca_los_by_ward() -> pd.DataFrame:
    sql = f"""
    WITH {_ADMISSIONS_BASE_CTE}
    SELECT
        ward                            AS WARD,
        ROUND(AVG(los_days), 1)         AS AVG_LOS,
        ROUND(MEDIAN(los_days), 1)      AS MEDIAN_LOS
    FROM admissions_base
    WHERE {_CURRENT_SYSTEM_FILTER} AND ward IS NOT NULL
    GROUP BY ward
    HAVING COUNT(*) >= 20
    ORDER BY AVG_LOS ASC
    """
    return _run(sql)


@st.cache_data(ttl=3600)
def get_ca_los_distribution() -> pd.DataFrame:
    sql = f"""
    WITH {_ADMISSIONS_BASE_CTE},
    with_dx AS (
        SELECT
            ab.visit_id, ab.ward, ab.los_days,
            COALESCE(d.diagnosis_name_expanded, 'Unspecified') AS condition_label
        FROM admissions_base ab
        LEFT JOIN HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED d ON d.visit_id = ab.visit_id
        WHERE ab.{_CURRENT_SYSTEM_FILTER} AND ab.ward IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (PARTITION BY ab.visit_id ORDER BY d.diagnosis_name_expanded NULLS LAST) = 1
    )
    SELECT ward AS WARD, los_days AS LOS_DAYS, condition_label AS CONDITION_LABEL
    FROM with_dx
    QUALIFY COUNT(*) OVER (PARTITION BY ward) >= 20
    ORDER BY ward, los_days
    """
    return _run(sql)


@st.cache_data(ttl=3600)
def get_ca_top_los_conditions() -> pd.DataFrame:
    """
    Ranks conditions by how often they actually produce a statistical LOS
    outlier — not by raw average LOS across all their cases. Previously this
    ranked purely on AVG(los_days), which surfaces conditions where every
    case happens to run long (often just a handful of genuinely complex
    cases) mixed in with conditions that are mostly normal-length but have
    one or two extreme cases dragging nothing (since it's an average of a
    small n). This instead uses the same Tukey 1.5×IQR-per-ward rule
    Plotly's own box plot applies for "LOS distribution by ward (IQR)"
    (boxpoints="outliers"), so a case counts here exactly when it would
    render as one of those outlier dots — then reports PCT_OF_CASES_OUTLIER
    per condition, which is the real signal for "is this an expected LOS
    for this condition, or a rare/unexpected extreme": a condition where
    most of its cases are outliers runs long as a rule (expected), while a
    condition where only a small fraction are outliers means something
    unusual happened in those specific cases (worth reviewing).
    """
    sql = f"""
    WITH {_ADMISSIONS_BASE_CTE},
    with_dx AS (
        SELECT
            ab.visit_id, ab.ward, ab.los_days, ab.age_years,
            COALESCE(d.diagnosis_name_expanded, 'Unspecified') AS condition_label
        FROM admissions_base ab
        LEFT JOIN HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED d ON d.visit_id = ab.visit_id
        WHERE ab.{_CURRENT_SYSTEM_FILTER} AND ab.ward IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (PARTITION BY ab.visit_id ORDER BY d.diagnosis_name_expanded NULLS LAST) = 1
    ),
    ward_iqr AS (
        SELECT DISTINCT
            ward,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY los_days) OVER (PARTITION BY ward) AS q1,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY los_days) OVER (PARTITION BY ward) AS q3
        FROM with_dx
    ),
    flagged AS (
        SELECT
            wd.*,
            wi.q3 + 1.5 * (wi.q3 - wi.q1) AS upper_fence
        FROM with_dx wd
        JOIN ward_iqr wi ON wi.ward = wd.ward
    ),
    condition_totals AS (
        -- 'Unspecified' (no diagnosis recorded) is kept in ward_iqr above so
        -- the outlier fence itself is computed against the ward's real
        -- admission population, but excluded here — there's no actual
        -- condition to judge "is a long stay expected" against when the
        -- diagnosis is simply missing, so it can't sit in the same ranked
        -- list as real diagnoses under an expected/unexpected framing.
        SELECT condition_label, COUNT(*) AS total_cases
        FROM flagged
        WHERE condition_label != 'Unspecified'
        GROUP BY condition_label
    ),
    ward_counts AS (
        SELECT condition_label, ward, COUNT(*) AS n
        FROM flagged
        WHERE condition_label != 'Unspecified'
        GROUP BY condition_label, ward
    ),
    ward_mode AS (
        SELECT condition_label, ward
        FROM ward_counts
        QUALIFY ROW_NUMBER() OVER (PARTITION BY condition_label ORDER BY n DESC) = 1
    ),
    outlier_agg AS (
        SELECT
            condition_label,
            COUNT(*)                       AS outlier_case_count,
            ROUND(AVG(age_years), 0)       AS avg_age,
            ROUND(AVG(los_days), 1)        AS avg_los
        FROM flagged
        WHERE los_days > upper_fence AND condition_label != 'Unspecified'
        GROUP BY condition_label
    )
    SELECT
        oa.condition_label                                                    AS CONDITION_LABEL,
        wm.ward                                                               AS WARD,
        oa.outlier_case_count                                                 AS CASE_COUNT,
        ct.total_cases                                                        AS TOTAL_CASE_COUNT,
        ROUND(100.0 * oa.outlier_case_count / ct.total_cases, 1)              AS PCT_OF_CASES_OUTLIER,
        oa.avg_age                                                            AS AVG_AGE,
        oa.avg_los                                                            AS AVG_LOS
    FROM outlier_agg oa
    JOIN condition_totals ct ON ct.condition_label = oa.condition_label
    LEFT JOIN ward_mode wm ON wm.condition_label = oa.condition_label
    -- Severity first, not frequency first — a condition with a single
    -- 300+ day outlier is more worth seeing than one with five mild
    -- outliers just past the ward's fence, but ORDER BY count DESC alone
    -- buried the former under the latter. avg_los here is already the
    -- average among just that condition's outlier cases (not all its
    -- cases), so this ranks by "how severe are this condition's outliers,"
    -- with outlier frequency only as the tiebreaker.
    ORDER BY oa.avg_los DESC, oa.outlier_case_count DESC
    LIMIT 8
    """
    return _run(sql)


@st.cache_data(ttl=3600)
def get_ca_los_vs_readmission_scatter() -> pd.DataFrame:
    sql = f"""
    WITH {_ADMISSIONS_BASE_CTE},
    {_READMIT_PAIRS_CTE}
    SELECT
        ab.ward                                                      AS WARD,
        ROUND(AVG(ab.los_days), 1)                                   AS AVG_LOS,
        ROUND(100.0 * COUNT(DISTINCT rp.index_visit_id)
              / NULLIF(COUNT(DISTINCT ab.visit_id), 0), 2)           AS READMISSION_RATE
    FROM admissions_base ab
    LEFT JOIN readmit_pairs rp ON rp.index_visit_id = ab.visit_id
    WHERE ab.{_CURRENT_SYSTEM_FILTER} AND ab.ward IS NOT NULL
    GROUP BY ab.ward
    HAVING COUNT(DISTINCT ab.visit_id) >= 20
    """
    return _run(sql)


@st.cache_data(ttl=3600)
def get_ca_index_vs_readmit_los() -> pd.DataFrame:
    sql = f"""
    WITH {_ADMISSIONS_BASE_CTE},
    {_READMIT_PAIRS_CTE}
    SELECT
        ROUND(AVG(index_los_days), 1)      AS avg_index,
        ROUND(MEDIAN(index_los_days), 1)   AS median_index,
        ROUND(AVG(readmit_los_days), 1)    AS avg_readmit,
        ROUND(MEDIAN(readmit_los_days), 1) AS median_readmit,
        ROUND(100.0 * SUM(CASE WHEN readmit_los_days < index_los_days THEN 1 ELSE 0 END)
              / NULLIF(COUNT(*), 0), 1)    AS pct_shorter
    FROM readmit_pairs
    """
    df = _run(sql)
    if df.empty:
        return df
    row = df.iloc[0]
    out = pd.DataFrame([
        {"LOS_TYPE": "Average", "INDEX_STAY": row["AVG_INDEX"], "READMIT_STAY": row["AVG_READMIT"]},
        {"LOS_TYPE": "Median", "INDEX_STAY": row["MEDIAN_INDEX"], "READMIT_STAY": row["MEDIAN_READMIT"]},
    ])
    out.attrs["pct_shorter"] = row["PCT_SHORTER"]
    return out


# ---------------------------------------------------------------------------
# Q_CA_7 — SSI benchmark
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_ca_ssi_benchmark() -> pd.DataFrame:
    sql = f"""
    WITH all_visits AS (
        SELECT LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
    ),
    categorized AS (
        SELECT {_SURGICAL_CATEGORY_CASE} AS surgical_category, {_SSI_FLAG_SQL} AS is_ssi
        FROM all_visits
    )
    SELECT
        surgical_category                            AS SURGICAL_CATEGORY,
        ROUND(100.0 * SUM(is_ssi) / COUNT(*), 2)      AS ACTUAL_SSI_RATE
    FROM categorized
    WHERE surgical_category IS NOT NULL
    GROUP BY surgical_category
    ORDER BY ACTUAL_SSI_RATE DESC
    """
    df = _run(sql)
    if df.empty:
        return df
    df["BENCHMARK_CEILING"] = df["SURGICAL_CATEGORY"].map(lambda c: _BENCHMARKS.get(c, 0.0))
    return df


# ---------------------------------------------------------------------------
# Q_CA_8 — SSI risk factors
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_ca_ssi_comorbidity() -> pd.DataFrame:
    parts = []
    for label, col in _COMORBIDITY_COLS.items():
        parts.append(f"""
        SELECT '{label}' AS condition,
            ROUND(100.0 * SUM(CASE WHEN is_ssi=1 AND {col} THEN 1 ELSE 0 END) / NULLIF(SUM(is_ssi),0), 1) AS ssi_prevalence,
            ROUND(100.0 * SUM(CASE WHEN {col} THEN 1 ELSE 0 END) / COUNT(*), 1) AS overall_prevalence
        FROM flagged
        """)
    union_sql = " UNION ALL ".join(parts)
    sql = f"""
    WITH all_visits AS (
        SELECT
            has_diabetes, has_anaemia, has_hypertension, has_cardiac_condition, has_renal_condition,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
    ),
    flagged AS (SELECT *, {_SSI_FLAG_SQL} AS is_ssi FROM all_visits)
    SELECT condition AS CONDITION, ssi_prevalence AS SSI_PREVALENCE, overall_prevalence AS OVERALL_PREVALENCE
    FROM ({union_sql})
    ORDER BY SSI_PREVALENCE DESC
    """
    return _run(sql)


@st.cache_data(ttl=3600)
def get_ca_ssi_multimorbidity() -> pd.DataFrame:
    sql = f"""
    WITH all_visits AS (
        SELECT
            (CASE WHEN has_diabetes THEN 1 ELSE 0 END + CASE WHEN has_hypertension THEN 1 ELSE 0 END
             + CASE WHEN has_cardiac_condition THEN 1 ELSE 0 END + CASE WHEN has_renal_condition THEN 1 ELSE 0 END
             + CASE WHEN has_anaemia THEN 1 ELSE 0 END + CASE WHEN has_hiv THEN 1 ELSE 0 END) AS comorbidity_count,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
    ),
    flagged AS (SELECT *, {_SSI_FLAG_SQL} AS is_ssi FROM all_visits)
    SELECT
        CASE WHEN comorbidity_count = 0 THEN '0 conditions'
             WHEN comorbidity_count = 1 THEN '1 condition'
             ELSE '2+ conditions' END                                AS CONDITION_COUNT,
        CASE WHEN comorbidity_count = 0 THEN 1 WHEN comorbidity_count = 1 THEN 2 ELSE 3 END AS TIER_ORDER,
        ROUND(100.0 * SUM(is_ssi) / COUNT(*), 2)                     AS SSI_RATE
    FROM flagged
    GROUP BY CONDITION_COUNT, TIER_ORDER
    ORDER BY TIER_ORDER
    """
    return _run(sql)


@st.cache_data(ttl=3600)
def get_ca_ssi_by_gender_category() -> pd.DataFrame:
    sql = f"""
    WITH all_visits AS (
        SELECT visit_id, source_system,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
    ),
    categorized AS (
        SELECT *, {_SURGICAL_CATEGORY_CASE} AS surgical_category, {_SSI_FLAG_SQL} AS is_ssi
        FROM all_visits
    ),
    with_gender AS (
        SELECT c.*,
            CASE
                WHEN LEFT(COALESCE(LOWER(a.gender), LOWER(v.gender)), 1) = 'f' THEN 'Female'
                WHEN LEFT(COALESCE(LOWER(a.gender), LOWER(v.gender)), 1) = 'm' THEN 'Male'
                ELSE NULL
            END AS gender
        FROM categorized c
        LEFT JOIN HOSPITALS.STAGING.STG_ADMISSIONS a ON a.visit_id = c.visit_id AND a.source_system = c.source_system
        LEFT JOIN HOSPITALS.STAGING.STG_VISITS v ON v.visit_id = c.visit_id AND v.source_system = c.source_system
    )
    SELECT
        surgical_category                          AS SURGICAL_CATEGORY,
        gender                                      AS GENDER,
        ROUND(100.0 * SUM(is_ssi) / COUNT(*), 1)    AS SSI_RATE
    FROM with_gender
    WHERE surgical_category IS NOT NULL AND gender IS NOT NULL
    GROUP BY surgical_category, gender
    ORDER BY surgical_category, gender
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# Q_CA_9 — SSI timing
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_ca_ssi_timing() -> pd.DataFrame:
    sql = f"""
    WITH {_ADMISSIONS_BASE_CTE},
    all_diagnoses AS (
        SELECT visit_id, patient_id, source_system, diagnosis_created_at,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
    ),
    surgical_index AS (
        SELECT a.visit_id AS index_visit_id, a.patient_id, a.source_system, a.discharge_date,
               {_SURGICAL_CATEGORY_CASE} AS surgical_category
        FROM admissions_base a
        JOIN all_diagnoses d ON d.visit_id = a.visit_id
        WHERE a.{_CURRENT_SYSTEM_FILTER}
    ),
    ssi_diagnoses AS (
        SELECT visit_id, patient_id, source_system, diagnosis_created_at
        FROM all_diagnoses WHERE {_SSI_FLAG_SQL} = 1
    ),
    matched AS (
        SELECT DATEDIFF('day', si.discharge_date, sd.diagnosis_created_at) AS days_after_discharge
        FROM surgical_index si
        JOIN ssi_diagnoses sd
            ON sd.patient_id = si.patient_id AND sd.source_system = si.source_system
            AND sd.diagnosis_created_at >= si.discharge_date
            AND sd.diagnosis_created_at <= DATEADD('day', 180, si.discharge_date)
        WHERE si.surgical_category IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (PARTITION BY si.index_visit_id ORDER BY sd.diagnosis_created_at ASC) = 1
    )
    SELECT
        CASE
            WHEN days_after_discharge <= 7 THEN '0-7 days'
            WHEN days_after_discharge <= 14 THEN '8-14 days'
            WHEN days_after_discharge <= 30 THEN '15-30 days'
            WHEN days_after_discharge <= 60 THEN '31-60 days'
            WHEN days_after_discharge <= 90 THEN '61-90 days'
            ELSE '90+ days'
        END                                                          AS TIMING_BUCKET,
        CASE WHEN days_after_discharge <= 7 THEN 1 WHEN days_after_discharge <= 14 THEN 2
             WHEN days_after_discharge <= 30 THEN 3 WHEN days_after_discharge <= 60 THEN 4
             WHEN days_after_discharge <= 90 THEN 5 ELSE 6 END       AS BUCKET_ORDER,
        COUNT(*)                                                     AS EPISODE_COUNT,
        days_after_discharge > 30                                    AS IS_POST_WINDOW
    FROM matched
    WHERE days_after_discharge >= 0
    GROUP BY TIMING_BUCKET, BUCKET_ORDER, IS_POST_WINDOW
    ORDER BY BUCKET_ORDER
    """
    return _run(sql)


@st.cache_data(ttl=3600)
def get_ca_ssi_during_vs_after() -> pd.DataFrame:
    sql = f"""
    WITH {_ADMISSIONS_BASE_CTE},
    all_diagnoses AS (
        SELECT visit_id, patient_id, source_system, diagnosis_created_at,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
    ),
    surgical_index AS (
        SELECT a.visit_id AS index_visit_id, a.patient_id, a.source_system, a.admission_date, a.discharge_date,
               {_SURGICAL_CATEGORY_CASE} AS surgical_category
        FROM admissions_base a
        JOIN all_diagnoses d ON d.visit_id = a.visit_id
        WHERE a.{_CURRENT_SYSTEM_FILTER}
    ),
    ssi_diagnoses AS (
        SELECT visit_id, patient_id, source_system, diagnosis_created_at
        FROM all_diagnoses WHERE {_SSI_FLAG_SQL} = 1
    ),
    matched AS (
        SELECT si.surgical_category,
               CASE WHEN sd.diagnosis_created_at <= si.discharge_date THEN 'Found during index stay'
                    ELSE 'Found after discharge' END AS detection_timing
        FROM surgical_index si
        JOIN ssi_diagnoses sd
            ON sd.patient_id = si.patient_id AND sd.source_system = si.source_system
            AND sd.diagnosis_created_at >= si.admission_date
            AND sd.diagnosis_created_at <= DATEADD('day', 180, si.discharge_date)
        WHERE si.surgical_category IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (PARTITION BY si.index_visit_id ORDER BY sd.diagnosis_created_at ASC) = 1
    )
    SELECT
        surgical_category    AS SURGICAL_CATEGORY,
        detection_timing      AS DETECTION_TIMING,
        COUNT(*)               AS EPISODE_COUNT
    FROM matched
    GROUP BY surgical_category, detection_timing
    ORDER BY surgical_category, detection_timing
    """
    return _run(sql)


@st.cache_data(ttl=3600)
def get_ca_ssi_monthly_trend() -> pd.DataFrame:
    sql = f"""
    WITH all_visits AS (
        SELECT visit_id, source_system, diagnosis_created_at,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
    ),
    flagged AS (
        SELECT visit_id, source_system, DATE_TRUNC('month', diagnosis_created_at) AS month,
            {_SSI_FLAG_SQL} AS is_ssi,
            CASE WHEN clean_dx_text LIKE '%septic pressure%' OR clean_dx_text LIKE '%hospital acquired%'
                      OR clean_dx_text LIKE '%nosocomial%' OR clean_dx_text LIKE '%catheter associated%'
                      OR clean_dx_text LIKE '%c. diff%' OR clean_dx_text LIKE '%clostridium difficile%'
                 THEN 1 ELSE 0 END AS is_other_hai
        FROM all_visits
    ),
    monthly AS (
        SELECT month, COUNT(DISTINCT visit_id) AS total_visits,
               SUM(is_ssi) AS ssi_visits, SUM(is_other_hai) AS other_hai_visits
        FROM flagged GROUP BY month
    )
    SELECT
        TO_VARCHAR(month, 'Mon YYYY')                                          AS VISIT_MONTH,
        month                                                                   AS SORT_MONTH,
        ROUND(100.0 * ssi_visits / NULLIF(total_visits, 0), 2)                  AS SSI_RATE,
        ROUND(100.0 * other_hai_visits / NULLIF(total_visits, 0), 2)            AS OTHER_HAI_RATE,
        ssi_visits                                                              AS SSI_CASES,
        other_hai_visits                                                        AS OTHER_HAI_CASES
    FROM monthly
    ORDER BY SORT_MONTH ASC
    """
    return _run(sql)
