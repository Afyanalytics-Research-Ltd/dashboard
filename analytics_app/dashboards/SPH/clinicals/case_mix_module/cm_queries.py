"""
sph/case_mix_module/cm_queries.py
====================================
All SQL for the SPH Case Mix tab.

Rules enforced here:
  - Every function is decorated with @st.cache_data(ttl=3600).
  - Every function returns a pd.DataFrame.
  - No rendering logic — zero st.* calls except the cache decorator.
  - Named get_cm_* to namespace from the OPD and Clinical Activity tabs.
  - The 16-segment classifier (_SEGMENT_CTE) is the single source of
    truth for "primary_visit_segment" — copied from Case_mix_queries.txt
    verbatim. If the classifier changes, update it here only.
"""

import pandas as pd
import streamlit as st

from sph.clinicals.opd_ipd_module.queries import _run

# ---------------------------------------------------------------------------
# Shared segment classifier CTE
# ---------------------------------------------------------------------------

_SEGMENT_CTE = """
WITH visit_diagnoses AS (
    SELECT
        visit_id, source_system, patient_id, visit_type, department, diagnosis_name_expanded,
        diagnosis_created_at,
        TRIM(bg.value::STRING) AS split_burden,
        LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
    FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
    -- OUTER => TRUE is required here — without it, LATERAL FLATTEN silently
    -- drops any visit whose burden_group is NULL/blank from this entire
    -- pipeline (not just from burden classification), which was deflating
    -- every downstream visit count for this module.
    LATERAL FLATTEN(input => SPLIT(burden_group, '|'), OUTER => TRUE) bg
    GROUP BY ALL
),
department_rules (pattern, segment) AS (
    SELECT * FROM (VALUES
        ('%orthopaedic%','ortho'),('%orthop%','ortho'),('%physiotherapy%','ortho'),
        ('%spine%','spine'),('%arthroscop%','ortho'),('%ent%','ent'),
        ('eye consultation%','eye'),('%eye specialist%','eye'),('%urology%','urology'),
        ('%gynaec%','obgyn'),('%obs/gyn%','obgyn'),('%maternity%','obgyn'),
        ('%anc%','obgyn'),('%cwc%','obgyn'),('%surgical%','surgery'),
        ('%general surgery%','surgery'),('%neurosurg%','neurosurgery'),
        ('%neurology%','neurology'),('%mopc%','chronic_medical'),('%plastic%','plastic'),
        ('%maxillofacial%','maxillofacial'),('%maxilofacial%','maxillofacial'),
        ('%dental%','dental'),('%dermatolog%','dermatology')
    ) AS t(pattern, segment)
),
-- visit_id alone is NOT globally unique — it's only unique within a
-- source_system (EMR_V1 vs EMR_V2). Every other module in this codebase
-- joins on (visit_id, source_system) together for exactly this reason;
-- this CTE previously joined/deduped on visit_id alone, which could
-- silently collapse two distinct visits from different source systems
-- that happen to share a visit_id value.
department_matched AS (
    SELECT visit_id, source_system, segment FROM (
        SELECT DISTINCT v.visit_id, v.source_system, r.pattern, r.segment
        FROM visit_diagnoses v
        JOIN department_rules r ON LOWER(COALESCE(v.department,'')) ILIKE r.pattern
    )
    QUALIFY ROW_NUMBER() OVER (PARTITION BY visit_id, source_system ORDER BY LENGTH(pattern) DESC) = 1
),
visit_classification AS (
    SELECT
        vd.visit_id, vd.source_system, vd.patient_id, vd.visit_type,
        MIN(vd.diagnosis_created_at) AS diagnosis_created_at,
        ANY_VALUE(dm.segment) AS dept_segment,
        MAX(CASE WHEN dm.segment='plastic' OR clean_dx_text LIKE '%cleft%' OR clean_dx_text LIKE '%palate%'
                 OR clean_dx_text LIKE '%circum%' OR clean_dx_text LIKE '%lipoma%' THEN 1 ELSE 0 END) AS is_plastic,
        MAX(CASE WHEN dm.segment='maxillofacial' OR clean_dx_text LIKE '%mandibular%' OR clean_dx_text LIKE '%mandible%'
                 OR clean_dx_text LIKE '%maxillary%' OR clean_dx_text LIKE '%maxilla%'
                 OR clean_dx_text LIKE '%trigeminal%' OR clean_dx_text LIKE '%salivary gland%' THEN 1 ELSE 0 END) AS is_maxillofacial,
        MAX(CASE WHEN dm.segment='dental' OR split_burden LIKE '%Dental%' OR clean_dx_text LIKE '%tooth%'
                 OR clean_dx_text LIKE '%dental%' THEN 1 ELSE 0 END) AS is_dental,
        MAX(CASE WHEN dm.segment='eye' OR split_burden = 'General: Eye' OR clean_dx_text LIKE '%ophthalmo%'
                 OR clean_dx_text LIKE '%cataract%' THEN 1 ELSE 0 END) AS is_ophthalmology,
        MAX(CASE WHEN dm.segment='ent' OR split_burden = 'General: ENT' OR clean_dx_text LIKE '%tonsil%'
                 OR clean_dx_text LIKE '%sinusitis%' THEN 1 ELSE 0 END) AS is_ent,
        MAX(CASE WHEN dm.segment='obgyn' OR split_burden IN ('General: Obstetric','General: Gynaecology')
                 OR clean_dx_text LIKE '%pregnan%' OR clean_dx_text LIKE '%delivery%' THEN 1 ELSE 0 END) AS is_obgyn,
        MAX(CASE WHEN dm.segment='spine' OR split_burden = 'Ortho: Spine' OR clean_dx_text LIKE '%spine%'
                 OR clean_dx_text LIKE '%spinal%' OR clean_dx_text LIKE '%lumbar%' OR clean_dx_text LIKE '%sciatica%'
                 OR clean_dx_text LIKE '%disc%' OR clean_dx_text LIKE '%spondyl%'
                 OR clean_dx_text LIKE '%back pain%' THEN 1 ELSE 0 END) AS is_spine,
        MAX(CASE WHEN dm.segment='ortho' OR split_burden LIKE 'Ortho:%' OR clean_dx_text LIKE '%fracture%'
                 OR clean_dx_text LIKE '%dislocat%' OR clean_dx_text LIKE '%tendin%'
                 OR clean_dx_text LIKE '%osteoarthritis%' OR clean_dx_text LIKE '%implant%' THEN 1 ELSE 0 END) AS is_ortho,
        MAX(CASE WHEN dm.segment='surgery' OR split_burden = 'General: Surgery' OR clean_dx_text LIKE '%hernia%'
                 OR clean_dx_text LIKE '%appendic%' OR clean_dx_text LIKE '%cholelithiasis%' THEN 1 ELSE 0 END) AS is_surgery,
        MAX(CASE WHEN dm.segment='neurosurgery' OR clean_dx_text LIKE '%craniotomy%' OR clean_dx_text LIKE '%subdural%'
                 THEN 1 ELSE 0 END) AS is_neurosurgery,
        MAX(CASE WHEN dm.segment='neurology' OR split_burden = 'General: Neurology' OR clean_dx_text LIKE '%stroke%'
                 OR clean_dx_text LIKE '%epilep%' THEN 1 ELSE 0 END) AS is_neurology,
        MAX(CASE WHEN dm.segment='urology' OR split_burden = 'General: Genitourinary' OR clean_dx_text LIKE '%prostat%'
                 THEN 1 ELSE 0 END) AS is_urology,
        MAX(CASE
            WHEN clean_dx_text LIKE ANY (
                '%urti%','%lrti%','%upper respiratory%','%lower respiratory%','%tonsillitis%',
                '%pharyngitis%','%bronchitis%','%gastroenteritis%','%acute ge%',
                ' ge ','%food poisoning%','%amoebiasis%','%dysentery%',
                '%tinea%','%ringworm%','%conjunctivitis%'
            ) THEN 0
            WHEN split_burden = 'General: Infection & Sepsis' OR clean_dx_text LIKE '%sepsis%'
                 OR clean_dx_text LIKE '%septic%' THEN 1 ELSE 0 END) AS has_sepsis,
        MAX(CASE WHEN split_burden = 'General: Cardiovascular' OR clean_dx_text LIKE '%hypertens%'
                 THEN 1 ELSE 0 END) AS has_cardio,
        MAX(CASE WHEN split_burden = 'General: Endocrine & Metabolic' OR clean_dx_text LIKE '%diabet%'
                 THEN 1 ELSE 0 END) AS has_metabolic,
        MAX(CASE WHEN clean_dx_text LIKE '%post %' OR clean_dx_text LIKE '%s/p %'
                 OR clean_dx_text LIKE '%follow up%' OR clean_dx_text LIKE '%followup%'
                 OR clean_dx_text LIKE '%f/u %' OR clean_dx_text LIKE '%review%'
                 OR clean_dx_text LIKE '%/52%' OR clean_dx_text LIKE '%/12%'
                 OR clean_dx_text LIKE '%weeks post%' OR clean_dx_text LIKE '%months post%'
                 OR clean_dx_text LIKE '%on physio%'
                 THEN 1 ELSE 0 END) AS is_followup_or_chronic_mgmt
    FROM visit_diagnoses vd
    LEFT JOIN department_matched dm ON dm.visit_id = vd.visit_id AND dm.source_system = vd.source_system
    GROUP BY ALL
),
final_hierarchy AS (
    SELECT *,
        -- Safe to COUNT(DISTINCT ...) across source systems — visit_id
        -- alone is not a unique key, this pair is.
        visit_id || '::' || source_system AS visit_key,
        CASE
            WHEN is_spine = 1 THEN 'Core Orthopedics: Spine and Back Pain Care'
            WHEN is_ortho = 1 THEN 'Core Orthopedics: General'
            WHEN is_surgery = 1 AND is_plastic = 0 THEN 'Core General Surgery'
            WHEN is_plastic = 1 THEN 'Standalone Specialty: Plastic Surgery'
            WHEN is_maxillofacial = 1 THEN 'Standalone Specialty: Maxillofacial'
            WHEN is_dental = 1 THEN 'Standalone Specialty: Dental'
            WHEN is_ophthalmology = 1 THEN 'Standalone Specialty: Eye/Ophthalmology'
            WHEN is_ent = 1 THEN 'Standalone Specialty: ENT'
            WHEN is_obgyn = 1 THEN 'Standalone Specialty: Obstetrics & Gynaecology'
            WHEN is_neurosurgery = 1 THEN 'Standalone Specialty: Neurosurgery (structural/acute)'
            WHEN is_neurology = 1 THEN 'Standalone Medical: Neurology (chronic/medical)'
            WHEN is_urology = 1 THEN 'Standalone Specialty: Urology'
            WHEN has_sepsis = 1 THEN 'Standalone Medical: Sepsis/Infection'
            WHEN has_cardio = 1 THEN 'Standalone Medical: Cardiovascular'
            WHEN has_metabolic = 1 THEN 'Standalone Medical: Endocrine/Metabolic'
            ELSE 'Other General Outpatient'
        END AS primary_visit_segment
    FROM visit_classification
)
"""


# ---------------------------------------------------------------------------
# S1 — Headline KPIs
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_cm_headline_kpis() -> pd.DataFrame:
    """TOTAL_VISITS (and every share % against it) is counted from
    STG_VISITS — the canonical, all-visits table — not from the
    diagnosis-classification CTE. STG_SPH_DIAGNOSIS_ENRICHED only contains
    visits that had a diagnosis entered, so using it as the "total" universe
    silently excludes every visit without one. Segment counts (spine/ortho/
    etc.) still come from the classifier, since a segment can only be
    assigned where a diagnosis exists — they're now expressed as a share of
    the true total, not of the diagnosed subset."""
    sql = f"""
    {_SEGMENT_CTE},
    true_total AS (
        SELECT COUNT(DISTINCT visit_id) AS total_visits_all
        FROM HOSPITALS.STAGING.STG_VISITS
    )
    SELECT
        tt.total_visits_all                                                     AS TOTAL_VISITS,
        COUNT(CASE WHEN fh.is_spine=1 OR fh.is_ortho=1 THEN 1 END)              AS CORE_ORTHO_VISITS,
        ROUND(100.0 * COUNT(CASE WHEN fh.is_spine=1 OR fh.is_ortho=1 THEN 1 END)
              / tt.total_visits_all, 1)                                         AS CORE_ORTHO_SHARE_PCT,
        COUNT(CASE WHEN fh.is_spine=1 THEN 1 END)                               AS SPINE_VISITS,
        ROUND(100.0 * COUNT(CASE WHEN fh.is_spine=1 AND YEAR(fh.diagnosis_created_at)=2022 THEN 1 END)
              / NULLIF(COUNT(CASE WHEN YEAR(fh.diagnosis_created_at)=2022 THEN 1 END), 0), 1)
                                                                                  AS SPINE_SHARE_2022_PCT,
        ROUND(100.0 * COUNT(CASE WHEN fh.is_spine=1 AND YEAR(fh.diagnosis_created_at)=2026 THEN 1 END)
              / NULLIF(COUNT(CASE WHEN YEAR(fh.diagnosis_created_at)=2026 THEN 1 END), 0), 1)
                                                                                  AS SPINE_SHARE_LATEST_PCT,
        COUNT(CASE WHEN fh.is_spine=0 AND fh.is_ortho=0 THEN 1 END)             AS DIVERSIFICATION_VISITS,
        ROUND(100.0 * COUNT(CASE WHEN fh.is_spine=0 AND fh.is_ortho=0 THEN 1 END)
              / tt.total_visits_all, 1)                                         AS DIVERSIFICATION_SHARE_PCT
    FROM final_hierarchy fh
    CROSS JOIN true_total tt
    GROUP BY tt.total_visits_all
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# S2 — Overall composition (treemap)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_cm_overall_composition() -> pd.DataFrame:
    sql = f"""
    {_SEGMENT_CTE}
    SELECT
        primary_visit_segment                                                         AS PRIMARY_VISIT_SEGMENT,
        COUNT(DISTINCT visit_id)                                                      AS TOTAL_VISITS,
        ROUND(100.0 * COUNT(DISTINCT visit_id) / SUM(COUNT(DISTINCT visit_id)) OVER(), 1)
                                                                                      AS PCT_OF_ALL_VISITS
    FROM final_hierarchy
    GROUP BY primary_visit_segment
    ORDER BY TOTAL_VISITS DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# S3 — Encounter type split (dumbbell)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_cm_encounter_type_split() -> pd.DataFrame:
    sql = f"""
    {_SEGMENT_CTE}
    SELECT
        primary_visit_segment                                                         AS PRIMARY_VISIT_SEGMENT,
        COUNT(DISTINCT visit_id)                                                      AS TOTAL_VISITS,
        SUM(CASE WHEN is_followup_or_chronic_mgmt = 0 THEN 1 ELSE 0 END)            AS NEW_ACUTE_VISITS,
        SUM(CASE WHEN is_followup_or_chronic_mgmt = 1 THEN 1 ELSE 0 END)            AS FOLLOW_UP_VISITS,
        ROUND(100.0 * SUM(CASE WHEN is_followup_or_chronic_mgmt = 0 THEN 1 ELSE 0 END)
              / COUNT(DISTINCT visit_id), 1)                                          AS PCT_NEW_ACUTE,
        ROUND(100.0 * SUM(CASE WHEN is_followup_or_chronic_mgmt = 1 THEN 1 ELSE 0 END)
              / COUNT(DISTINCT visit_id), 1)                                          AS PCT_FOLLOW_UP
    FROM final_hierarchy
    GROUP BY primary_visit_segment
    ORDER BY TOTAL_VISITS DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# S4 — Yearly trend + growth
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_cm_yearly_trend() -> pd.DataFrame:
    sql = f"""
    {_SEGMENT_CTE},
    yearly AS (
        SELECT
            primary_visit_segment,
            YEAR(diagnosis_created_at) AS year,
            COUNT(DISTINCT visit_id) AS total_visits
        FROM final_hierarchy
        WHERE YEAR(diagnosis_created_at) BETWEEN 2022 AND 2026
        GROUP BY primary_visit_segment, year
    )
    SELECT
        year                                                                     AS VISIT_YEAR,
        primary_visit_segment                                                    AS PRIMARY_VISIT_SEGMENT,
        total_visits                                                             AS TOTAL_VISITS,
        ROUND(100.0 * total_visits / SUM(total_visits) OVER (PARTITION BY year), 1) AS PCT_OF_YEAR_TOTAL,
        LAG(total_visits) OVER (PARTITION BY primary_visit_segment ORDER BY year) AS PRIOR_YEAR_VISITS,
        ROUND(100.0 * (total_visits - LAG(total_visits) OVER (PARTITION BY primary_visit_segment ORDER BY year))
                   / NULLIF(LAG(total_visits) OVER (PARTITION BY primary_visit_segment ORDER BY year), 0), 1)
                                                                                  AS PCT_GROWTH_VS_PRIOR_YEAR
    FROM yearly
    ORDER BY year, total_visits DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# S5 — New vs. returning, and seasonality
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_cm_new_returning_patients() -> pd.DataFrame:
    sql = f"""
    {_SEGMENT_CTE},
    segmented AS (
        SELECT visit_id, patient_id, diagnosis_created_at, primary_visit_segment
        FROM final_hierarchy
    ),
    patient_first_visit AS (
        SELECT patient_id, primary_visit_segment, MIN(diagnosis_created_at) AS first_ever_visit_to_segment
        FROM segmented
        WHERE primary_visit_segment <> 'Other General Outpatient'
        GROUP BY patient_id, primary_visit_segment
    ),
    visit_flagged AS (
        SELECT
            s.visit_id, s.primary_visit_segment, s.patient_id,
            YEAR(s.diagnosis_created_at) AS year,
            MONTH(s.diagnosis_created_at) AS month_num,
            CASE WHEN s.diagnosis_created_at = pfv.first_ever_visit_to_segment
                 THEN 'New patient (first-ever visit to this segment)'
                 ELSE 'Returning patient' END AS patient_status
        FROM segmented s
        JOIN patient_first_visit pfv
            ON pfv.patient_id = s.patient_id AND pfv.primary_visit_segment = s.primary_visit_segment
        WHERE s.primary_visit_segment <> 'Other General Outpatient'
    )
    SELECT
        primary_visit_segment                                                    AS PRIMARY_VISIT_SEGMENT,
        year                                                                     AS VISIT_YEAR,
        month_num                                                                AS MONTH_NUM,
        patient_status                                                           AS PATIENT_STATUS,
        COUNT(DISTINCT visit_id)                                                 AS TOTAL_VISITS,
        COUNT(DISTINCT patient_id)                                               AS DISTINCT_PATIENTS
    FROM visit_flagged
    WHERE year BETWEEN 2022 AND 2026
    GROUP BY primary_visit_segment, year, month_num, patient_status
    ORDER BY primary_visit_segment, year, month_num, patient_status
    """
    df = _run(sql)
    if not df.empty:
        _MONTHS = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
        df["MONTH_LABEL"] = df["MONTH_NUM"].map(_MONTHS)
    return df


@st.cache_data(ttl=3600)
def get_cm_seasonality() -> pd.DataFrame:
    sql = f"""
    {_SEGMENT_CTE},
    by_year_month AS (
        SELECT
            primary_visit_segment,
            YEAR(diagnosis_created_at) AS year,
            MONTH(diagnosis_created_at) AS month_num,
            COUNT(DISTINCT visit_id) AS total_visits
        FROM final_hierarchy
        WHERE primary_visit_segment <> 'Other General Outpatient'
          AND YEAR(diagnosis_created_at) BETWEEN 2022 AND 2026
        GROUP BY primary_visit_segment, year, month_num
    )
    SELECT
        primary_visit_segment                                                    AS PRIMARY_VISIT_SEGMENT,
        year                                                                      AS VISIT_YEAR,
        month_num                                                                AS MONTH_NUM,
        total_visits                                                             AS TOTAL_VISITS
    FROM by_year_month
    ORDER BY primary_visit_segment, year, month_num
    """
    df = _run(sql)
    if not df.empty:
        _MONTHS = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
        df["MONTH_LABEL"] = df["MONTH_NUM"].map(_MONTHS)
    return df


# ---------------------------------------------------------------------------
# S6 — Comorbidity profile
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_cm_comorbidity_profile() -> pd.DataFrame:
    sql = """
    WITH visit_diagnoses AS (
        SELECT
            visit_id, patient_id, department, has_chronic_condition,
            has_hypertension, has_diabetes, has_hiv, has_anaemia, has_asthma,
            has_epilepsy, has_sickle_cell, has_thyroid_condition, has_renal_condition,
            has_cardiac_condition, has_psychiatric_condition,
            TRIM(bg.value::STRING) AS split_burden,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
        -- OUTER => TRUE — see _SEGMENT_CTE; without it, visits with no
        -- burden_group are silently dropped from this query entirely.
        LATERAL FLATTEN(input => SPLIT(burden_group, '|'), OUTER => TRUE) bg
        GROUP BY ALL
    ),
    department_rules (pattern, segment) AS (
        SELECT * FROM (VALUES
            ('%orthopaedic%','ortho'), ('%orthop%','ortho'), ('%physiotherapy%','ortho'),
            ('%spine%','spine'), ('%arthroscop%','ortho'), ('%ent%','ent'),
            ('eye consultation%','eye'), ('%eye specialist%','eye'), ('%urology%','urology'),
            ('%gynaec%','obgyn'), ('%obs/gyn%','obgyn'), ('%maternity%','obgyn'),
            ('%anc%','obgyn'), ('%cwc%','obgyn'), ('%surgical%','surgery'),
            ('%general surgery%','surgery'), ('%neurosurg%','neurosurgery'),
            ('%neurology%','neurology'), ('%mopc%','chronic_medical'), ('%plastic%','plastic'),
            ('%maxillofacial%','maxillofacial'), ('%maxilofacial%','maxillofacial'),
            ('%dental%','dental'), ('%dermatolog%','dermatology')
        ) AS t(pattern, segment)
    ),
    department_matched AS (
        SELECT visit_id, segment
        FROM (
            SELECT DISTINCT v.visit_id, r.pattern, r.segment
            FROM visit_diagnoses v
            JOIN department_rules r ON LOWER(COALESCE(v.department,'')) ILIKE r.pattern
        )
        QUALIFY ROW_NUMBER() OVER (PARTITION BY visit_id ORDER BY LENGTH(pattern) DESC) = 1
    ),
    flagged AS (
        SELECT
            vd.visit_id, vd.patient_id,
            ANY_VALUE(vd.has_chronic_condition) AS has_chronic_condition,
            ANY_VALUE(vd.has_hypertension) AS has_hypertension,
            ANY_VALUE(vd.has_diabetes) AS has_diabetes,
            ANY_VALUE(vd.has_hiv) AS has_hiv,
            ANY_VALUE(vd.has_anaemia) AS has_anaemia,
            ANY_VALUE(vd.has_asthma) AS has_asthma,
            ANY_VALUE(vd.has_renal_condition) AS has_renal_condition,
            ANY_VALUE(vd.has_cardiac_condition) AS has_cardiac_condition,
            ANY_VALUE(vd.has_thyroid_condition) AS has_thyroid_condition,
            MAX(CASE WHEN dm.segment='plastic' OR clean_dx_text LIKE '%cleft%' OR clean_dx_text LIKE '%palate%'
                     OR clean_dx_text LIKE '%circum%' OR clean_dx_text LIKE '%lipoma%' THEN 1 ELSE 0 END) AS is_plastic,
            MAX(CASE WHEN dm.segment='maxillofacial' OR clean_dx_text LIKE '%mandibular%' OR clean_dx_text LIKE '%mandible%'
                     OR clean_dx_text LIKE '%maxillary%' OR clean_dx_text LIKE '%maxilla%' OR clean_dx_text LIKE '%trigeminal%'
                     OR clean_dx_text LIKE '%salivary gland%' THEN 1 ELSE 0 END) AS is_maxillofacial,
            MAX(CASE WHEN dm.segment='dental' OR split_burden LIKE '%Dental%' OR clean_dx_text LIKE '%tooth%'
                     OR clean_dx_text LIKE '%dental%' THEN 1 ELSE 0 END) AS is_dental,
            MAX(CASE WHEN dm.segment='eye' OR split_burden = 'General: Eye' OR clean_dx_text LIKE '%ophthalmo%'
                     OR clean_dx_text LIKE '%cataract%' THEN 1 ELSE 0 END) AS is_ophthalmology,
            MAX(CASE WHEN dm.segment='ent' OR split_burden = 'General: ENT' OR clean_dx_text LIKE '%tonsil%'
                     OR clean_dx_text LIKE '%sinusitis%' THEN 1 ELSE 0 END) AS is_ent,
            MAX(CASE WHEN dm.segment='obgyn' OR split_burden IN ('General: Obstetric','General: Gynaecology')
                     OR clean_dx_text LIKE '%pregnan%' OR clean_dx_text LIKE '%delivery%' THEN 1 ELSE 0 END) AS is_obgyn,
            MAX(CASE WHEN dm.segment='spine' OR split_burden = 'Ortho: Spine' OR clean_dx_text LIKE '%spine%'
                     OR clean_dx_text LIKE '%spinal%' OR clean_dx_text LIKE '%lumbar%' OR clean_dx_text LIKE '%sciatica%'
                     OR clean_dx_text LIKE '%disc%' OR clean_dx_text LIKE '%spondyl%' THEN 1 ELSE 0 END) AS is_spine,
            MAX(CASE WHEN dm.segment='ortho' OR split_burden LIKE 'Ortho:%' OR clean_dx_text LIKE '%fracture%'
                     OR clean_dx_text LIKE '%dislocat%' OR clean_dx_text LIKE '%tendin%' OR clean_dx_text LIKE '%osteoarthritis%'
                     OR clean_dx_text LIKE '%implant%' THEN 1 ELSE 0 END) AS is_ortho,
            MAX(CASE WHEN dm.segment='surgery' OR split_burden = 'General: Surgery' OR clean_dx_text LIKE '%hernia%'
                     OR clean_dx_text LIKE '%appendic%' OR clean_dx_text LIKE '%cholelithiasis%' THEN 1 ELSE 0 END) AS is_surgery,
            MAX(CASE WHEN dm.segment='neurosurgery' OR clean_dx_text LIKE '%craniotomy%' OR clean_dx_text LIKE '%subdural%'
                     THEN 1 ELSE 0 END) AS is_neurosurgery,
            MAX(CASE WHEN dm.segment='neurology' OR split_burden = 'General: Neurology' OR clean_dx_text LIKE '%stroke%'
                     OR clean_dx_text LIKE '%epilep%' THEN 1 ELSE 0 END) AS is_neurology,
            MAX(CASE WHEN dm.segment='urology' OR split_burden = 'General: Genitourinary' OR clean_dx_text LIKE '%prostat%'
                     THEN 1 ELSE 0 END) AS is_urology,
            MAX(CASE
                WHEN clean_dx_text LIKE ANY (
                    '%urti%','%lrti%','%upper respiratory%','%lower respiratory%','%tonsillitis%',
                    '%pharyngitis%','%bronchitis%','%gastroenteritis%','%acute ge%',
                    ' ge ','%food poisoning%','%amoebiasis%','%dysentery%',
                    '%tinea%','%ringworm%','%conjunctivitis%'
                ) THEN 0
                WHEN split_burden = 'General: Infection & Sepsis' OR clean_dx_text LIKE '%sepsis%'
                     OR clean_dx_text LIKE '%septic%' THEN 1 ELSE 0 END) AS has_sepsis,
            MAX(CASE WHEN split_burden = 'General: Cardiovascular' OR clean_dx_text LIKE '%hypertens%'
                     THEN 1 ELSE 0 END) AS has_cardio,
            MAX(CASE WHEN split_burden = 'General: Endocrine & Metabolic' OR clean_dx_text LIKE '%diabet%'
                     THEN 1 ELSE 0 END) AS has_metabolic
        FROM visit_diagnoses vd
        LEFT JOIN department_matched dm ON dm.visit_id = vd.visit_id
        GROUP BY ALL
    ),
    segmented AS (
        SELECT *,
            CASE
                WHEN is_spine = 1 THEN 'Core Orthopedics: Spine and Back Pain Care'
                WHEN is_ortho = 1 THEN 'Core Orthopedics: General'
                WHEN is_surgery = 1 AND is_plastic = 0 THEN 'Core General Surgery'
                WHEN is_plastic = 1 THEN 'Standalone Specialty: Plastic Surgery'
                WHEN is_maxillofacial = 1 THEN 'Standalone Specialty: Maxillofacial'
                WHEN is_dental = 1 THEN 'Standalone Specialty: Dental'
                WHEN is_ophthalmology = 1 THEN 'Standalone Specialty: Eye/Ophthalmology'
                WHEN is_ent = 1 THEN 'Standalone Specialty: ENT'
                WHEN is_obgyn = 1 THEN 'Standalone Specialty: Obstetrics & Gynaecology'
                WHEN is_neurosurgery = 1 THEN 'Standalone Specialty: Neurosurgery'
                WHEN is_neurology = 1 THEN 'Standalone Medical: Neurology'
                WHEN is_urology = 1 THEN 'Standalone Specialty: Urology'
                WHEN has_sepsis = 1 THEN 'Standalone Medical: Sepsis/Infection'
                WHEN has_cardio = 1 THEN 'Standalone Medical: Cardiovascular'
                WHEN has_metabolic = 1 THEN 'Standalone Medical: Endocrine/Metabolic'
                ELSE 'Other General Outpatient'
            END AS primary_visit_segment
        FROM flagged
    )
    SELECT
        primary_visit_segment                                                    AS PRIMARY_VISIT_SEGMENT,
        COUNT(DISTINCT visit_id)                                                 AS TOTAL_VISITS,
        ROUND(100.0 * SUM(CASE WHEN has_chronic_condition THEN 1 ELSE 0 END) / COUNT(DISTINCT visit_id), 1) AS PCT_ANY_COMORBIDITY,
        ROUND(100.0 * SUM(CASE WHEN has_hypertension THEN 1 ELSE 0 END) / COUNT(DISTINCT visit_id), 1)      AS PCT_HYPERTENSION,
        ROUND(100.0 * SUM(CASE WHEN has_diabetes THEN 1 ELSE 0 END) / COUNT(DISTINCT visit_id), 1)          AS PCT_DIABETES,
        ROUND(100.0 * SUM(CASE WHEN has_anaemia THEN 1 ELSE 0 END) / COUNT(DISTINCT visit_id), 1)           AS PCT_ANAEMIA,
        ROUND(100.0 * SUM(CASE WHEN has_cardiac_condition THEN 1 ELSE 0 END) / COUNT(DISTINCT visit_id), 1) AS PCT_CARDIAC,
        ROUND(100.0 * SUM(CASE WHEN has_renal_condition THEN 1 ELSE 0 END) / COUNT(DISTINCT visit_id), 1)   AS PCT_RENAL,
        ROUND(100.0 * SUM(CASE WHEN has_hiv THEN 1 ELSE 0 END) / COUNT(DISTINCT visit_id), 1)               AS PCT_HIV,
        ROUND(100.0 * SUM(CASE WHEN has_thyroid_condition THEN 1 ELSE 0 END) / COUNT(DISTINCT visit_id), 1) AS PCT_THYROID,
        ROUND(100.0 * SUM(CASE WHEN has_asthma THEN 1 ELSE 0 END) / COUNT(DISTINCT visit_id), 1)            AS PCT_ASTHMA
    FROM segmented
    GROUP BY primary_visit_segment
    ORDER BY TOTAL_VISITS DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# S7 — Other General Outpatient breakdown + trend
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_cm_other_opd_breakdown() -> pd.DataFrame:
    sql = f"""
    {_SEGMENT_CTE},
    other_general AS (
        SELECT fh.visit_id AS visit_id, fh.patient_id AS patient_id, d.diagnosis_name_expanded
        FROM final_hierarchy fh
        JOIN HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED d ON d.visit_id = fh.visit_id
        WHERE fh.primary_visit_segment = 'Other General Outpatient'
    ),
    prepped AS (
        SELECT *,
            LOWER(COALESCE(diagnosis_name_expanded, '')) AS t,
            (diagnosis_name_expanded LIKE '%?%') AS is_suspected
        FROM other_general
    ),
    s01 AS (SELECT *, REGEXP_REPLACE(t, '^impre[ss]*[;:\\\\-]\\\\s*', '', 1, 0, 'i') AS t2 FROM prepped),
    s02 AS (SELECT * EXCLUDE t, REGEXP_REPLACE(t2, '^imp[r]*[;:\\\\-]\\\\s*', '', 1, 0, 'i') AS t FROM s01),
    s03 AS (SELECT * EXCLUDE t2, REGEXP_REPLACE(t, '^impression[;:\\\\-]\\\\s*', '', 1, 0, 'i') AS t2 FROM s02),
    s04 AS (SELECT * EXCLUDE t, REGEXP_REPLACE(t2,
                '^(urti|upper airway infection|upper resp(iratory)?\\\\s*(tract)?\\\\s*infection|acute upper (airway|resp\\\\w*) infection[,]?\\\\s*(unspecified)?|upper respiratory infection)\\\\s*\\\\??\\\\s*(cause)?$',
                'upper respiratory tract infection', 1, 0, 'i') AS t FROM s03),
    s05 AS (SELECT * EXCLUDE t2, REGEXP_REPLACE(t,
                '^(lrti|lower resp(iratory)?\\\\s*(tract)?\\\\s*infection|lower resp tract infxn|lower respiratory infection)\\\\s*\\\\??\\\\s*(cause)?$',
                'lower respiratory tract infection', 1, 0, 'i') AS t2 FROM s04),
    s06 AS (SELECT * EXCLUDE t, REGEXP_REPLACE(t2,
                '^(uti|urinary tract infection)\\\\s*,?\\\\s*(site not specified)?\\\\s*\\\\??\\\\s*(cause)?$',
                'urinary tract infection', 1, 0, 'i') AS t FROM s05),
    s07 AS (SELECT * EXCLUDE t2, REGEXP_REPLACE(t,
                '^(ge|gastroenteritis|acute ge)\\\\s*\\\\??\\\\s*(cause)?\\\\s*(with|no)?\\\\s*(some)?\\\\s*dehydration?$',
                'gastroenteritis', 1, 0, 'i') AS t2 FROM s06),
    cleaned AS (
        SELECT
            *,
            LOWER(TRIM(REGEXP_REPLACE(
                REGEXP_REPLACE(REGEXP_REPLACE(t2, '\\\\?+', ' '), '\\\\bcause\\\\b\\\\s*$', ' ', 1, 0, 'i'),
                '\\\\s+', ' '))) AS clean_dx_lower,
            INITCAP(TRIM(REGEXP_REPLACE(
                REGEXP_REPLACE(REGEXP_REPLACE(t2, '\\\\?+', ' '), '\\\\bcause\\\\b\\\\s*$', ' ', 1, 0, 'i'),
                '\\\\s+', ' '))) AS clean_dx_display
        FROM s07
    ),
    canonicalized AS (
        SELECT
            *,
            CASE
                WHEN clean_dx_lower = 'urti' THEN 'Upper Respiratory Tract Infection'
                WHEN clean_dx_lower = 'lrti' THEN 'Lower Respiratory Tract Infection'
                WHEN clean_dx_lower LIKE '%upper%' AND (clean_dx_lower LIKE '%resp%' OR clean_dx_lower LIKE '%airway%') THEN 'Upper Respiratory Tract Infection'
                WHEN clean_dx_lower LIKE '%lower%' AND (clean_dx_lower LIKE '%resp%' OR clean_dx_lower LIKE '%airway%') THEN 'Lower Respiratory Tract Infection'
                WHEN clean_dx_lower LIKE '%urinary%' OR clean_dx_lower = 'uti' THEN 'Urinary Tract Infection'
                WHEN clean_dx_lower LIKE '%gastroenterit%' OR clean_dx_lower IN ('ge','acute ge') THEN 'Gastroenteritis'
                WHEN clean_dx_lower LIKE '%pneumonia%' THEN 'Pneumonia'
                WHEN clean_dx_lower NOT LIKE '%chronic%'
                     AND (clean_dx_lower = 'pud' OR clean_dx_lower LIKE '%peptic ulcer%')
                THEN 'Peptic Ulcer Disease'
                ELSE NULL
            END AS canonical_dx
        FROM cleaned
    ),
    merged AS (
        SELECT
            visit_id, patient_id,
            CASE
                WHEN is_suspected THEN 'Suspected: ' || COALESCE(canonical_dx, NULLIF(clean_dx_display, ''))
                ELSE COALESCE(canonical_dx, NULLIF(clean_dx_display, ''))
            END AS unified_diagnosis
        FROM canonicalized
    )
    SELECT
        unified_diagnosis                                                        AS UNIFIED_DIAGNOSIS,
        COUNT(*)                                                                 AS OCCURRENCES,
        COUNT(DISTINCT visit_id)                                                 AS DISTINCT_VISITS,
        COUNT(DISTINCT patient_id)                                               AS DISTINCT_PATIENTS
    FROM merged
    GROUP BY unified_diagnosis
    ORDER BY OCCURRENCES DESC
    LIMIT 20
    """
    return _run(sql)


@st.cache_data(ttl=3600)
def get_cm_other_opd_trend() -> pd.DataFrame:
    sql = f"""
    {_SEGMENT_CTE}
    SELECT
        YEAR(diagnosis_created_at)                                                    AS VISIT_YEAR,
        COUNT(DISTINCT visit_id)                                                      AS TOTAL_VISITS,
        ROUND(100.0 * COUNT(DISTINCT visit_id) / SUM(COUNT(DISTINCT visit_id))
              OVER (PARTITION BY YEAR(diagnosis_created_at)), 1)                     AS PCT_OF_YEAR_TOTAL
    FROM final_hierarchy
    WHERE primary_visit_segment = 'Other General Outpatient'
      AND YEAR(diagnosis_created_at) BETWEEN 2022 AND 2026
      AND visit_id IN (
          SELECT DISTINCT visit_id FROM visit_diagnoses
          WHERE diagnosis_name_expanded IS NOT NULL AND TRIM(diagnosis_name_expanded) <> ''
      )
    GROUP BY YEAR(diagnosis_created_at)
    ORDER BY VISIT_YEAR
    """
    return _run(sql)


@st.cache_data(ttl=3600)
def get_cm_other_opd_monthly() -> pd.DataFrame:
    """Monthly version of get_cm_other_opd_trend() — used for the seasonal-trend projection."""
    sql = f"""
    {_SEGMENT_CTE}
    SELECT
        DATE_TRUNC('month', diagnosis_created_at)::DATE AS VISIT_MONTH,
        COUNT(DISTINCT visit_id)                         AS TOTAL_VISITS
    FROM final_hierarchy
    WHERE primary_visit_segment = 'Other General Outpatient'
      AND diagnosis_created_at >= '2022-06-01'
      AND visit_id IN (
          SELECT DISTINCT visit_id FROM visit_diagnoses
          WHERE diagnosis_name_expanded IS NOT NULL AND TRIM(diagnosis_name_expanded) <> ''
      )
    GROUP BY 1
    ORDER BY 1
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# S5b — Spine and Back Pain Care: monthly diagnosis breakdown
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_cm_spine_diagnosis_monthly() -> pd.DataFrame:
    sql = """
    WITH visit_diagnoses AS (
        SELECT
            visit_id, patient_id, visit_type, department, diagnosis_name_expanded,
            diagnosis_created_at,
            TRIM(bg.value::STRING) AS split_burden,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
        -- OUTER => TRUE — see _SEGMENT_CTE; without it, visits with no
        -- burden_group are silently dropped from this query entirely.
        LATERAL FLATTEN(input => SPLIT(burden_group, '|'), OUTER => TRUE) bg
        GROUP BY ALL
    ),
    department_rules (pattern, segment) AS (
        SELECT * FROM (VALUES
            ('%orthopaedic%','ortho'),('%orthop%','ortho'),('%physiotherapy%','ortho'),
            ('%spine%','spine'),('%arthroscop%','ortho'),('%ent%','ent'),
            ('eye consultation%','eye'),('%eye specialist%','eye'),('%urology%','urology'),
            ('%gynaec%','obgyn'),('%obs/gyn%','obgyn'),('%maternity%','obgyn'),
            ('%anc%','obgyn'),('%cwc%','obgyn'),('%surgical%','surgery'),
            ('%general surgery%','surgery'),('%neurosurg%','neurosurgery'),
            ('%neurology%','neurology'),('%mopc%','chronic_medical'),('%plastic%','plastic'),
            ('%maxillofacial%','maxillofacial'),('%maxilofacial%','maxillofacial'),
            ('%dental%','dental'),('%dermatolog%','dermatology')
        ) AS t(pattern, segment)
    ),
    department_matched AS (
        SELECT visit_id, segment FROM (
            SELECT DISTINCT v.visit_id, r.pattern, r.segment
            FROM visit_diagnoses v
            JOIN department_rules r ON LOWER(COALESCE(v.department,'')) ILIKE r.pattern
        )
        QUALIFY ROW_NUMBER() OVER (PARTITION BY visit_id ORDER BY LENGTH(pattern) DESC) = 1
    ),
    visit_classification AS (
        SELECT
            vd.visit_id,
            MAX(CASE WHEN dm.segment='spine' OR split_burden = 'Ortho: Spine' OR clean_dx_text LIKE '%spine%'
                     OR clean_dx_text LIKE '%spinal%' OR clean_dx_text LIKE '%lumbar%' OR clean_dx_text LIKE '%sciatica%'
                     OR clean_dx_text LIKE '%disc%' OR clean_dx_text LIKE '%spondyl%'
                     OR clean_dx_text LIKE '%back pain%' THEN 1 ELSE 0 END) AS is_spine
        FROM visit_diagnoses vd
        LEFT JOIN department_matched dm ON dm.visit_id = vd.visit_id
        GROUP BY vd.visit_id
    ),
    spine_visit_ids AS (
        SELECT visit_id FROM visit_classification WHERE is_spine = 1
    )
    SELECT
        DATE_TRUNC('month', vd.diagnosis_created_at)::DATE AS VISIT_MONTH,
        vd.clean_dx_text                                    AS CLEAN_DX_TEXT,
        COUNT(DISTINCT vd.visit_id)                         AS TOTAL_VISITS
    FROM visit_diagnoses vd
    JOIN spine_visit_ids sv ON sv.visit_id = vd.visit_id
    WHERE vd.diagnosis_created_at >= '2024-01-01'
    GROUP BY 1, 2
    QUALIFY ROW_NUMBER() OVER (PARTITION BY VISIT_MONTH ORDER BY TOTAL_VISITS DESC) <= 12
    ORDER BY VISIT_MONTH, TOTAL_VISITS DESC
    """
    return _run(sql)
