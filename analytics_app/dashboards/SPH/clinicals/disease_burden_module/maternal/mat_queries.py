"""
sph/disease_burden_module/maternal/mat_queries.py
========================================================
All SQL for the Disease Burden → Maternal health sub-tab.

Rules enforced here:
  - Every function is decorated with @st.cache_data(ttl=3600).
  - Every function returns a pd.DataFrame.
  - No rendering logic — zero st.* calls except the cache decorator.
  - Named get_mat_* to namespace from the other tabs' queries.
  - Uses the Round 3 (final) classifier from "Maternal,obgyn queries.txt" —
    never the Round 1/2 superseded versions.
"""

import pandas as pd
import streamlit as st

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clinicals.opd_ipd_module.queries import _run


# ---------------------------------------------------------------------------
# Shared CTE fragment — Round 3 case mix classifier
# ---------------------------------------------------------------------------

_CASE_MIX_CTE = """
WITH obgyn_visits AS (
    SELECT
        visit_id, source_system, visit_type, has_diabetes, has_hypertension,
        TRIM(bg.value::STRING) AS split_burden,
        LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
    FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
    LATERAL FLATTEN(input => SPLIT(burden_group, '|')) bg
),
obgyn_filtered AS (
    SELECT * FROM obgyn_visits
    WHERE split_burden IN ('General: Obstetric', 'General: Gynaecology')
       OR clean_dx_text LIKE '%pregnan%' OR clean_dx_text LIKE '%pregnac%' OR clean_dx_text LIKE '%anc%'
       OR clean_dx_text LIKE '%antenatal%' OR clean_dx_text LIKE '%gravid%' OR clean_dx_text LIKE '%gestation%'
),
case_mix_classified AS (
    SELECT
        *,
        CASE
            WHEN clean_dx_text LIKE '%pre-eclamp%' OR clean_dx_text LIKE '%preeclamp%'
                 OR clean_dx_text LIKE '%eclamp%' OR clean_dx_text LIKE '%hypertens%'
                 OR clean_dx_text LIKE '%haemorrhage%' OR clean_dx_text LIKE '%hemorrhage%'
                 OR clean_dx_text LIKE '%obstructed labour%' OR clean_dx_text LIKE '%gestational diabetes%'
                 OR clean_dx_text LIKE '%hyperemesis%' OR clean_dx_text LIKE '%pprom%'
                 OR clean_dx_text LIKE '%premature rupture%' OR clean_dx_text LIKE '%subchorionic bleeding%'
                THEN 'High-Risk Pregnancy / Complications'
            WHEN clean_dx_text LIKE '%miscarriage%' OR clean_dx_text LIKE '%blighted ovum%'
                 OR clean_dx_text LIKE '%abortion%'
                THEN 'Pregnancy Loss'
            WHEN clean_dx_text LIKE '%labour%' OR clean_dx_text LIKE '%labor%'
                 OR clean_dx_text LIKE '%delivery%' OR clean_dx_text LIKE '%caesar%'
                 OR clean_dx_text LIKE '%c-section%' OR clean_dx_text LIKE '% svd %' OR clean_dx_text LIKE 'svd %'
                 OR clean_dx_text LIKE '%nuchal cord%' OR clean_dx_text LIKE '%breech presentation%'
                 OR clean_dx_text LIKE '%vulva laceration%'
                THEN 'Labour & Delivery'
            WHEN clean_dx_text LIKE '%postnatal%' OR clean_dx_text LIKE '%post natal%'
                 OR clean_dx_text LIKE '%postpartum%' OR clean_dx_text LIKE '%mastitis%'
                THEN 'Postnatal Care'
            WHEN clean_dx_text LIKE '%post tah%' OR clean_dx_text LIKE '%post bso%'
                THEN 'Post-Hysterectomy Follow-up'
            WHEN clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%' OR clean_dx_text LIKE '%fiboroids%'
                THEN 'Fibroids'
            WHEN clean_dx_text LIKE '%adenomyos%' OR clean_dx_text LIKE '%adenomos%'
                THEN 'Adenomyosis'
            WHEN clean_dx_text LIKE '%dysmenorr%' OR clean_dx_text LIKE '%dismenorr%'
                THEN 'Dysmenorrhea'
            WHEN clean_dx_text LIKE '%amenorrh%'
                THEN 'Amenorrhea'
            WHEN clean_dx_text LIKE '%endometri%'
                THEN 'Endometriosis / Endometrial conditions'
            WHEN clean_dx_text LIKE '%aub%' OR clean_dx_text LIKE '%abnormal uterine%'
                 OR clean_dx_text LIKE '%menorrhagia%' OR clean_dx_text LIKE '%uterine bleeding%'
                THEN 'Abnormal Uterine Bleeding'
            WHEN clean_dx_text LIKE '%ovarian cyst%' OR clean_dx_text LIKE '%polycystic ovar%' OR clean_dx_text LIKE '%polycyctic ovary%'
                THEN 'Ovarian conditions (cysts/PCOS)'
            WHEN clean_dx_text LIKE '%pelvic pain%' OR clean_dx_text LIKE '%pelvic mass%'
                 OR clean_dx_text LIKE '%pelvic congestion%'
                THEN 'Pelvic pain / mass'
            WHEN clean_dx_text LIKE '%tubal blockage%' OR clean_dx_text LIKE '%infertility%'
                 OR clean_dx_text LIKE '%pelvic inflammatory%'
                THEN 'Infertility / PID'
            WHEN clean_dx_text LIKE '% ca %' AND (clean_dx_text LIKE '%vulva%' OR clean_dx_text LIKE '%cervi%'
                 OR clean_dx_text LIKE '%ovary%' OR clean_dx_text LIKE '%uter%')
                THEN 'Gynaecological malignancy'
            WHEN clean_dx_text LIKE '%dyspareunia%'
                THEN 'Dyspareunia'
            WHEN clean_dx_text LIKE '%pregnan%' OR clean_dx_text LIKE '%pregnac%' OR clean_dx_text LIKE '%anc%'
                 OR clean_dx_text LIKE '%antenatal%' OR clean_dx_text LIKE '%gravid%' OR clean_dx_text LIKE '%gestation%'
                THEN 'ANC / Routine Pregnancy Care'
            ELSE 'Other OBGYN'
        END AS case_mix_category
    FROM obgyn_filtered
)
"""

_HIGH_RISK_ONLY_CTE = """
WITH obgyn_visits AS (
    SELECT
        visit_id,
        TRIM(bg.value::STRING) AS split_burden,
        LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
    FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
    LATERAL FLATTEN(input => SPLIT(burden_group, '|')) bg
),
obgyn_filtered AS (
    SELECT * FROM obgyn_visits
    WHERE split_burden IN ('General: Obstetric', 'General: Gynaecology')
       OR clean_dx_text LIKE '%pregnan%' OR clean_dx_text LIKE '%pregnac%' OR clean_dx_text LIKE '%anc%'
       OR clean_dx_text LIKE '%antenatal%' OR clean_dx_text LIKE '%gravid%' OR clean_dx_text LIKE '%gestation%'
),
high_risk_only AS (
    SELECT * FROM obgyn_filtered
    WHERE clean_dx_text LIKE '%pre-eclamp%' OR clean_dx_text LIKE '%preeclamp%'
       OR clean_dx_text LIKE '%eclamp%' OR clean_dx_text LIKE '%hypertens%'
       OR clean_dx_text LIKE '%haemorrhage%' OR clean_dx_text LIKE '%hemorrhage%'
       OR clean_dx_text LIKE '%obstructed labour%' OR clean_dx_text LIKE '%gestational diabetes%'
       OR clean_dx_text LIKE '%hyperem%' OR clean_dx_text LIKE '%pprom%'
       OR clean_dx_text LIKE '%premature ruptur%' OR clean_dx_text LIKE '%premature raptur%'
       OR clean_dx_text LIKE '%subchorionic bleeding%'
)
"""


# ---------------------------------------------------------------------------
# KPI 1 — Case mix volume and conversion
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_mat_case_mix() -> pd.DataFrame:
    sql = _CASE_MIX_CTE + """
    SELECT
        case_mix_category                                                              AS CASE_MIX_CATEGORY,
        COUNT(DISTINCT visit_id)                                                        AS TOTAL_VISITS,
        COUNT(DISTINCT CASE WHEN visit_type = 'Inpatient' THEN visit_id END)             AS INPATIENT_ADMISSIONS,
        ROUND(100.0 * COUNT(DISTINCT CASE WHEN visit_type = 'Inpatient' THEN visit_id END)
                   / NULLIF(COUNT(DISTINCT visit_id), 0), 1)                             AS CONVERSION_RATE_PCT,
        ROUND(100.0 * COUNT(DISTINCT visit_id) / SUM(COUNT(DISTINCT visit_id)) OVER (), 1)
                                                                                          AS PCT_OF_OBGYN_VOLUME
    FROM case_mix_classified
    GROUP BY case_mix_category
    ORDER BY TOTAL_VISITS DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# KPI 2 — Demographics (age group per case mix category)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_mat_demographics() -> pd.DataFrame:
    sql = _CASE_MIX_CTE + """
    SELECT
        cmc.case_mix_category AS CASE_MIX_CATEGORY,
        COALESCE(
            v.age_group,
            CASE
                WHEN age_years IS NULL        THEN 'Unknown'
                WHEN age_years < 0            THEN 'Invalid Age (Negative)'
                WHEN age_years < 5            THEN 'Toddler (0-4)'
                WHEN age_years < 13           THEN 'Child (5-12)'
                WHEN age_years < 18           THEN 'Adolescent (13-17)'
                WHEN age_years < 25           THEN 'Youth (18-24)'
                WHEN age_years < 35           THEN 'Young Adult (25-34)'
                WHEN age_years < 45           THEN 'Adult (35-44)'
                WHEN age_years < 55           THEN 'Middle Age (45-54)'
                WHEN age_years < 65           THEN 'Older Adult (55-64)'
                ELSE                                  'Senior (65+)'
            END
        ) AS AGE_GROUP,
        COUNT(DISTINCT cmc.visit_id) AS TOTAL_VISITS
    FROM case_mix_classified cmc
    LEFT JOIN HOSPITALS.STAGING.STG_VISITS v ON v.visit_id = cmc.visit_id AND v.source_system = cmc.source_system
    LEFT JOIN HOSPITALS.STAGING.STG_ADMISSIONS d ON d.visit_id = cmc.visit_id AND d.source_system = cmc.source_system
    GROUP BY ALL
    ORDER BY cmc.case_mix_category, TOTAL_VISITS DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# KPI 3 — Comorbidities per case mix category
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_mat_comorbidities() -> pd.DataFrame:
    sql = _CASE_MIX_CTE + """
    SELECT
        case_mix_category AS CASE_MIX_CATEGORY,
        COUNT(DISTINCT visit_id) AS TOTAL_VISITS,
        ROUND(100.0 * COUNT(DISTINCT CASE WHEN has_diabetes THEN visit_id END)
                   / NULLIF(COUNT(DISTINCT visit_id), 0), 1) AS PCT_DIABETES,
        ROUND(100.0 * COUNT(DISTINCT CASE WHEN has_hypertension THEN visit_id END)
                   / NULLIF(COUNT(DISTINCT visit_id), 0), 1) AS PCT_HYPERTENSION
    FROM case_mix_classified
    GROUP BY case_mix_category
    ORDER BY TOTAL_VISITS DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# PART 3 REBUILT — ANC visit-count distribution
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_mat_anc_visit_distribution() -> pd.DataFrame:
    sql = """
    WITH anc_visits AS (
        SELECT
            visit_id, patient_id, source_system, diagnosis_created_at,
            TRIM(bg.value::STRING) AS split_burden,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
        LATERAL FLATTEN(input => SPLIT(burden_group, '|')) bg
    ),
    anc_filtered AS (
        SELECT DISTINCT visit_id, patient_id, diagnosis_created_at
        FROM anc_visits
        WHERE split_burden IN ('General: Obstetric')
           OR clean_dx_text LIKE '%pregnan%' OR clean_dx_text LIKE '%pregnac%' OR clean_dx_text LIKE '%anc%'
           OR clean_dx_text LIKE '%antenatal%' OR clean_dx_text LIKE '%gravid%' OR clean_dx_text LIKE '%gestation%'
    ),
    visits_per_patient AS (
        SELECT
            patient_id,
            COUNT(DISTINCT visit_id) AS total_anc_visits,
            MIN(diagnosis_created_at) AS first_visit,
            MAX(diagnosis_created_at) AS last_visit
        FROM anc_filtered
        GROUP BY patient_id
    )
    SELECT
        CASE
            WHEN total_anc_visits = 1 THEN '1 visit'
            WHEN total_anc_visits = 2 THEN '2 visits'
            WHEN total_anc_visits = 3 THEN '3 visits'
            ELSE '4+ visits (meets paper''s quality-predictive threshold)'
        END AS VISIT_COUNT_BUCKET,
        COUNT(*) AS TOTAL_PATIENTS,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS PCT_OF_ANC_PATIENTS
    FROM visits_per_patient
    GROUP BY VISIT_COUNT_BUCKET
    ORDER BY VISIT_COUNT_BUCKET
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# ANC quality — PART A (coverage rates) and PART B (composite score)
# ---------------------------------------------------------------------------

_ANC_QUALITY_CTE = """
WITH anc_visits AS (
    SELECT DISTINCT visit_id, patient_id, source_system
    FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
    LATERAL FLATTEN(input => SPLIT(burden_group, '|')) bg
    WHERE TRIM(bg.value::STRING) = 'General: Obstetric'
       OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%pregnan%'
       OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%antenatal%'
       OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%gravid%'
),
bp_check AS (
    SELECT DISTINCT visit_id, source_system
    FROM HOSPITALS.STAGING.STG_SPH_VITALS
    WHERE systolic_blood_pressure IS NOT NULL OR diastolic_blood_pressure IS NOT NULL
),
urine_check AS (
    SELECT DISTINCT visit_id, source_system
    FROM HOSPITALS.STAGING.STG_SPH_INVESTIGATIONS
    WHERE UPPER(COALESCE(canonical_name,'')) LIKE '%URIN%'
       OR UPPER(COALESCE(test_group,'')) LIKE '%URINALYSIS%'
       OR UPPER(COALESCE(panel,'')) LIKE '%URIN%'
),
blood_check AS (
    SELECT DISTINCT visit_id, source_system
    FROM HOSPITALS.STAGING.STG_SPH_INVESTIGATIONS
    WHERE discipline IS NOT NULL
      AND NOT (UPPER(COALESCE(canonical_name,'')) LIKE '%URIN%' OR UPPER(COALESCE(test_group,'')) LIKE '%URINALYSIS%')
),
iron_check AS (
    SELECT DISTINCT visit_id, source_system
    FROM HOSPITALS.STAGING.stg_pharmacy_orders
    WHERE LOWER(item_name) LIKE ANY ('%iron%', '%ferrous%', '%folic%', '%ranferon%')
),
ultrasound_check AS (
    SELECT DISTINCT visit_id, source_system
    FROM HOSPITALS.STAGING.STG_IMAGING_ORDERS
    WHERE UPPER(study_name) LIKE '%OBSTETRIC%'
),
combined AS (
    SELECT
        av.visit_id,
        CASE WHEN bp.visit_id IS NOT NULL THEN 1 ELSE 0 END AS has_bp,
        CASE WHEN ur.visit_id IS NOT NULL THEN 1 ELSE 0 END AS has_urine,
        CASE WHEN bl.visit_id IS NOT NULL THEN 1 ELSE 0 END AS has_blood,
        CASE WHEN ir.visit_id IS NOT NULL THEN 1 ELSE 0 END AS has_iron,
        CASE WHEN us.visit_id IS NOT NULL THEN 1 ELSE 0 END AS has_ultrasound
    FROM anc_visits av
    LEFT JOIN bp_check bp ON bp.visit_id = av.visit_id AND bp.source_system = av.source_system
    LEFT JOIN urine_check ur ON ur.visit_id = av.visit_id AND ur.source_system = av.source_system
    LEFT JOIN blood_check bl ON bl.visit_id = av.visit_id AND bl.source_system = av.source_system
    LEFT JOIN iron_check ir ON ir.visit_id = av.visit_id AND ir.source_system = av.source_system
    LEFT JOIN ultrasound_check us ON us.visit_id = av.visit_id AND us.source_system = av.source_system
)
"""


@st.cache_data(ttl=3600)
def get_mat_anc_quality_part_a() -> pd.DataFrame:
    sql = _ANC_QUALITY_CTE + """
    SELECT
        COUNT(*) AS TOTAL_ANC_VISITS,
        ROUND(100.0 * SUM(has_bp) / COUNT(*), 1) AS PCT_BP_TAKEN,
        ROUND(100.0 * SUM(has_urine) / COUNT(*), 1) AS PCT_URINE_SAMPLE,
        ROUND(100.0 * SUM(has_blood) / COUNT(*), 1) AS PCT_BLOOD_SAMPLE,
        ROUND(100.0 * SUM(has_iron) / COUNT(*), 1) AS PCT_IRON_GIVEN,
        ROUND(100.0 * SUM(has_ultrasound) / COUNT(*), 1) AS PCT_ULTRASOUND_FETAL_PROXY
    FROM combined
    """
    return _run(sql)


@st.cache_data(ttl=3600)
def get_mat_anc_quality_part_b() -> pd.DataFrame:
    sql = _ANC_QUALITY_CTE + """
    SELECT
        (has_bp + has_urine + has_blood + has_iron + has_ultrasound) AS ANC_QUALITY_SCORE_OUT_OF_5,
        COUNT(*) AS TOTAL_VISITS,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS PCT_OF_ANC_VISITS
    FROM combined
    GROUP BY ANC_QUALITY_SCORE_OUT_OF_5
    ORDER BY ANC_QUALITY_SCORE_OUT_OF_5 DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# KPI 5 CORRECTED — Maternal complications (high-risk pregnancy only)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_mat_complications() -> pd.DataFrame:
    sql = _HIGH_RISK_ONLY_CTE + """
    SELECT
        CASE
            WHEN clean_dx_text LIKE '%pre-eclamp%' OR clean_dx_text LIKE '%preeclamp%' THEN 'Pre-eclampsia'
            WHEN clean_dx_text LIKE '%eclamp%' THEN 'Eclampsia'
            WHEN clean_dx_text LIKE '%haemorrhage%' OR clean_dx_text LIKE '%hemorrhage%' THEN 'Haemorrhage'
            WHEN clean_dx_text LIKE '%obstructed labour%' THEN 'Obstructed labour'
            WHEN clean_dx_text LIKE '%gestational diabetes%' THEN 'Gestational diabetes'
            WHEN clean_dx_text LIKE '%hyperem%' THEN 'Hyperemesis Gravidarum'
            WHEN clean_dx_text LIKE '%pprom%' OR clean_dx_text LIKE '%premature ruptur%' OR clean_dx_text LIKE '%premature raptur%'
                THEN 'PPROM / Premature rupture of membranes'
            WHEN clean_dx_text LIKE '%subchorionic bleeding%' THEN 'Subchorionic bleeding'
            WHEN clean_dx_text LIKE '%hypertens%' THEN 'Hypertensive disorder (general)'
            ELSE NULL
        END AS COMPLICATION_TYPE,
        COUNT(DISTINCT visit_id) AS DISTINCT_VISITS
    FROM high_risk_only
    GROUP BY COMPLICATION_TYPE
    HAVING COMPLICATION_TYPE IS NOT NULL
    ORDER BY DISTINCT_VISITS DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# PART 2A REBUILT — BP readings for hypertensive-pregnancy patients
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_mat_bp_hypertensive() -> pd.DataFrame:
    sql = """
    WITH hypertensive_visits AS (
        SELECT
            visit_id, patient_id, source_system,
            diagnosis_created_at AS diagnosis_date,
            diagnosis_name_expanded
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
        WHERE LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%pre-eclamp%'
           OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%preeclamp%'
           OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%eclamp%'
           OR (LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%hypertens%'
               AND LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%pregnan%')
    ),
    all_bp_readings AS (
        SELECT
            visit_id, patient_id, source_system,
            triage_date, systolic_blood_pressure, diastolic_blood_pressure
        FROM HOSPITALS.STAGING.STG_SPH_VITALS
        WHERE systolic_blood_pressure IS NOT NULL OR diastolic_blood_pressure IS NOT NULL
    )
    SELECT
        hv.patient_id AS PATIENT_ID,
        bp.systolic_blood_pressure AS SYSTOLIC_BLOOD_PRESSURE,
        bp.diastolic_blood_pressure AS DIASTOLIC_BLOOD_PRESSURE
    FROM hypertensive_visits hv
    LEFT JOIN all_bp_readings bp ON bp.patient_id = hv.patient_id AND bp.source_system = hv.source_system
    ORDER BY hv.patient_id
    """
    df_raw = _run(sql)
    if df_raw.empty:
        return pd.DataFrame([{"N_WITH_BP": 0, "N_TOTAL": 0}])
    n_total = df_raw["PATIENT_ID"].nunique()
    n_with_bp = (
        df_raw[df_raw["SYSTOLIC_BLOOD_PRESSURE"].notna() | df_raw["DIASTOLIC_BLOOD_PRESSURE"].notna()]
        ["PATIENT_ID"].nunique()
    )
    return pd.DataFrame([{"N_WITH_BP": n_with_bp, "N_TOTAL": n_total}])


# ---------------------------------------------------------------------------
# PART 2B — Haemorrhage workup
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_mat_haemorrhage_workup() -> pd.DataFrame:
    sql = """
    WITH haemorrhage_visits AS (
        SELECT visit_id, patient_id, source_system
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
        WHERE LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%haemorrhage%'
           OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%hemorrhage%'
    ),
    lab_check AS (
        SELECT
            hv.visit_id,
            MAX(CASE WHEN inv.canonical_name IN ('HB','Hemoglobin','HGB','Haemoglobin') THEN 1 ELSE 0 END) AS has_hemoglobin,
            MAX(CASE WHEN inv.canonical_name LIKE '%Blood Group%' THEN 1 ELSE 0 END) AS has_blood_group,
            MAX(CASE WHEN inv.canonical_name IN ('PT (Prothrombin Time)','APTT','INR') THEN 1 ELSE 0 END) AS has_coagulation,
            COUNT(inv.visit_id) AS any_investigation_rows
        FROM haemorrhage_visits hv
        LEFT JOIN HOSPITALS.STAGING.STG_SPH_INVESTIGATIONS inv
            ON inv.visit_id = hv.visit_id AND inv.source_system = hv.source_system
        GROUP BY hv.visit_id
    ),
    vitals_check AS (
        SELECT
            hv.visit_id,
            MAX(CASE WHEN v.pulse_rate IS NOT NULL THEN 1 ELSE 0 END) AS has_pulse,
            MAX(CASE WHEN v.systolic_blood_pressure IS NOT NULL THEN 1 ELSE 0 END) AS has_bp,
            MAX(CASE WHEN v.pulse_rate > 100 THEN 1 ELSE 0 END) AS has_tachycardia,
            MAX(CASE WHEN v.systolic_blood_pressure < 90 THEN 1 ELSE 0 END) AS has_hypotension
        FROM haemorrhage_visits hv
        LEFT JOIN HOSPITALS.STAGING.STG_SPH_VITALS v
            ON v.visit_id = hv.visit_id AND v.source_system = hv.source_system
        GROUP BY hv.visit_id
    )
    SELECT
        COUNT(DISTINCT hv.visit_id) AS TOTAL_HAEMORRHAGE_VISITS,
        SUM(CASE WHEN lc.any_investigation_rows > 0 THEN 1 ELSE 0 END) AS VISITS_WITH_ANY_LAB_RECORD,
        ROUND(100.0 * SUM(CASE WHEN lc.any_investigation_rows > 0 THEN 1 ELSE 0 END) / COUNT(DISTINCT hv.visit_id), 1) AS PCT_WITH_ANY_LAB,
        SUM(lc.has_hemoglobin) AS WITH_HEMOGLOBIN_CHECK,
        SUM(lc.has_blood_group) AS WITH_BLOOD_GROUP_CHECK,
        SUM(lc.has_coagulation) AS WITH_COAGULATION_CHECK,
        SUM(vc.has_pulse) AS WITH_PULSE_RECORDED,
        SUM(vc.has_bp) AS WITH_BP_RECORDED,
        SUM(vc.has_tachycardia) AS VISITS_SHOWING_TACHYCARDIA,
        SUM(vc.has_hypotension) AS VISITS_SHOWING_HYPOTENSION
    FROM haemorrhage_visits hv
    LEFT JOIN lab_check lc ON lc.visit_id = hv.visit_id
    LEFT JOIN vitals_check vc ON vc.visit_id = hv.visit_id
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# Headline KPIs — assembled from the queries above, not one monolithic SQL
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_mat_headline_kpis() -> pd.DataFrame:
    df_cm = get_mat_case_mix()
    df_anc = get_mat_anc_visit_distribution()
    df_ql = get_mat_anc_quality_part_b()

    fibroids_pct = 0.0
    fibroids_conv = 0.0
    if not df_cm.empty:
        m = df_cm[df_cm["CASE_MIX_CATEGORY"] == "Fibroids"]
        if not m.empty:
            fibroids_pct = float(m.iloc[0]["PCT_OF_OBGYN_VOLUME"])
            fibroids_conv = float(m.iloc[0]["CONVERSION_RATE_PCT"])

    single_pct = four_plus_pct = 0.0
    if not df_anc.empty:
        single = df_anc[df_anc["VISIT_COUNT_BUCKET"].str.startswith("1 visit")]
        four_p = df_anc[df_anc["VISIT_COUNT_BUCKET"].str.startswith("4+")]
        single_pct = float(single.iloc[0]["PCT_OF_ANC_PATIENTS"]) if not single.empty else 0.0
        four_plus_pct = float(four_p.iloc[0]["PCT_OF_ANC_PATIENTS"]) if not four_p.empty else 0.0

    zero_pct = 0.0
    if not df_ql.empty:
        zero_q = df_ql[df_ql["ANC_QUALITY_SCORE_OUT_OF_5"] == 0]
        zero_pct = float(zero_q.iloc[0]["PCT_OF_ANC_VISITS"]) if not zero_q.empty else 0.0

    return pd.DataFrame([{
        "SINGLE_VISIT_PCT": single_pct,
        "FOUR_PLUS_VISIT_PCT": four_plus_pct,
        "ZERO_INDICATOR_PCT": zero_pct,
        "FIBROIDS_PCT_OF_VOLUME": fibroids_pct,
        "FIBROIDS_CONV_RATE": fibroids_conv,
    }])
