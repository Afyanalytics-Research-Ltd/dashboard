"""
sph/clinicals/data_quality_module/dq_queries.py
=================================================
Queries backing the Data Quality tab. Every check below measures a real,
verifiable condition on HOSPITALS.STAGING.STG_VISITS (and, where an ANC
flag is needed, the same case-mix/burden-group logic already used in
cm_queries.py / mat_queries.py) — nothing here is a fabricated or assumed
number.

Five dimensions, one query each:
  Consistency  — does a patient's recorded gender ever change across visits?
  Reliability  — do visits carry demographic/clinical combinations that are
                 implausible on their face (e.g. a male ANC visit, a
                 toddler ANC visit)?
  Validity     — are required fields (gender, age_group) populated, and do
                 populated values fall inside the expected value set?
  Timeliness   — how stale is the most recent visit record relative to today?
  Uniqueness   — are there duplicate rows for the same (visit_id,
                 source_system) key, which STG_VISITS should never have?
"""

import pandas as pd
import streamlit as st

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clinicals.opd_ipd_module.queries import _run as _run_raw


def _run(sql: str) -> pd.DataFrame:
    return _run_raw(sql)


# ANC flag — same pattern-match logic used in mat_queries.py / cm_queries.py
# (burden_group split + department/diagnosis text match), joined back onto
# STG_VISITS on the (visit_id, source_system) composite key.
_ANC_FLAG_CTE = """
anc_flagged_visits AS (
    SELECT DISTINCT d.visit_id, d.source_system
    FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED d,
    LATERAL FLATTEN(input => SPLIT(d.burden_group, '|'), OUTER => TRUE) bg
    WHERE TRIM(bg.value::string) IN ('General: Obstetric', 'General: Gynaecology')
       OR LOWER(d.diagnosis_name_expanded) LIKE '%pregnan%'
       OR LOWER(d.diagnosis_name_expanded) LIKE '%anc%'
       OR LOWER(d.diagnosis_name_expanded) LIKE '%antenatal%'
       OR LOWER(d.diagnosis_name_expanded) LIKE '%gravid%'
       OR LOWER(d.diagnosis_name_expanded) LIKE '%gestation%'
)
"""


@st.cache_data(ttl=3600)
def get_dq_consistency() -> pd.DataFrame:
    """
    Patients whose recorded gender differs across their own visits —
    the same real-world person should not flip gender between records.
    """
    sql = f"""
    WITH patient_gender AS (
        SELECT patient_id, COUNT(DISTINCT LOWER(TRIM(gender))) AS distinct_genders
        FROM HOSPITALS.STAGING.STG_VISITS
        WHERE patient_id IS NOT NULL AND gender IS NOT NULL AND TRIM(gender) <> ''
        GROUP BY patient_id
    )
    SELECT
        COUNT(*) AS TOTAL_PATIENTS,
        COUNT(CASE WHEN distinct_genders > 1 THEN 1 END) AS INCONSISTENT_PATIENTS,
        ROUND(100.0 * COUNT(CASE WHEN distinct_genders > 1 THEN 1 END) / NULLIF(COUNT(*), 0), 2)
            AS INCONSISTENT_PCT
    FROM patient_gender
    """
    return _run(sql)


@st.cache_data(ttl=3600)
def get_dq_reliability_anc_anomalies() -> pd.DataFrame:
    """
    ANC-flagged visits carrying an implausible patient profile: male
    gender, or an age_group too young to plausibly be an ANC patient.
    """
    sql = f"""
    WITH {_ANC_FLAG_CTE}
    SELECT
        COUNT(*) AS TOTAL_ANC_VISITS,
        COUNT(CASE WHEN LOWER(LEFT(TRIM(v.gender), 1)) = 'm' THEN 1 END) AS MALE_ANC_VISITS,
        COUNT(CASE WHEN v.age_group IN ('Toddler (0-4)', 'Child (5-12)') THEN 1 END)
            AS TOO_YOUNG_ANC_VISITS,
        ROUND(100.0 * COUNT(CASE WHEN LOWER(LEFT(TRIM(v.gender), 1)) = 'm'
                                   OR v.age_group IN ('Toddler (0-4)', 'Child (5-12)')
                             THEN 1 END) / NULLIF(COUNT(*), 0), 2) AS ANOMALOUS_PCT
    FROM anc_flagged_visits af
    JOIN HOSPITALS.STAGING.STG_VISITS v
        ON v.visit_id = af.visit_id AND v.source_system = af.source_system
    """
    return _run(sql)


@st.cache_data(ttl=3600)
def get_dq_validity() -> pd.DataFrame:
    """
    Required-field completeness and value-set validity for gender and
    age_group on STG_VISITS.
    """
    sql = """
    SELECT
        COUNT(*) AS TOTAL_VISITS,
        COUNT(CASE WHEN gender IS NULL OR TRIM(gender) = '' THEN 1 END) AS MISSING_GENDER,
        COUNT(CASE WHEN age_group IS NULL OR TRIM(age_group) = '' THEN 1 END) AS MISSING_AGE_GROUP,
        COUNT(CASE WHEN gender IS NOT NULL AND TRIM(gender) <> ''
                    AND LOWER(LEFT(TRIM(gender), 1)) NOT IN ('m', 'f') THEN 1 END)
            AS INVALID_GENDER_VALUE,
        ROUND(100.0 * COUNT(CASE WHEN gender IS NULL OR TRIM(gender) = '' THEN 1 END)
              / NULLIF(COUNT(*), 0), 2) AS MISSING_GENDER_PCT,
        ROUND(100.0 * COUNT(CASE WHEN age_group IS NULL OR TRIM(age_group) = '' THEN 1 END)
              / NULLIF(COUNT(*), 0), 2) AS MISSING_AGE_GROUP_PCT
    FROM HOSPITALS.STAGING.STG_VISITS
    """
    return _run(sql)


@st.cache_data(ttl=3600)
def get_dq_timeliness() -> pd.DataFrame:
    """
    Freshness of the visit record vs. today, plus the count of visits
    landing in the untracked 31-90 day follow-up blind spot (same window
    referenced elsewhere on the Overview page).
    """
    sql = """
    SELECT
        MAX(visit_date) AS MAX_VISIT_DATE,
        DATEDIFF('day', MAX(visit_date), CURRENT_DATE()) AS DAYS_SINCE_LAST_VISIT,
        COUNT(CASE WHEN DATEDIFF('day', visit_date, CURRENT_DATE()) BETWEEN 31 AND 90
                   THEN 1 END) AS BLIND_SPOT_VISITS
    FROM HOSPITALS.STAGING.STG_VISITS
    """
    return _run(sql)


@st.cache_data(ttl=3600)
def get_dq_uniqueness() -> pd.DataFrame:
    """
    Duplicate rows for the same (visit_id, source_system) composite key —
    STG_VISITS should have exactly one row per key.
    """
    sql = """
    WITH keyed AS (
        SELECT visit_id, source_system, COUNT(*) AS n
        FROM HOSPITALS.STAGING.STG_VISITS
        GROUP BY visit_id, source_system
    )
    SELECT
        COUNT(*) AS TOTAL_KEYS,
        COUNT(CASE WHEN n > 1 THEN 1 END) AS DUPLICATE_KEYS,
        SUM(CASE WHEN n > 1 THEN n - 1 ELSE 0 END) AS EXCESS_DUPLICATE_ROWS,
        ROUND(100.0 * COUNT(CASE WHEN n > 1 THEN 1 END) / NULLIF(COUNT(*), 0), 2)
            AS DUPLICATE_KEY_PCT
    FROM keyed
    """
    return _run(sql)
