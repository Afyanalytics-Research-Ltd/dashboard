"""
sph/opd_ipd_module/queries.py
==============================
All SQL for the SPH OPD → IPD Conversion tab.

Rules enforced here:
  - Every function is decorated with @st.cache_data(ttl=3600).
  - Every function returns a pd.DataFrame.
  - No rendering logic — zero st.* calls except the cache decorator.
  - SnowflakeClient is the only database interface used.
  - Sepsis exclusion (respiratory/GI/skin low-acuity infections) is
    baked into every has_sepsis flag — do not remove without updating
    the insight text in views.py.
  - The full segment CASE hierarchy is intentionally duplicated across
    queries because primary_visit_segment is not yet materialised as a
    view. Once that view exists, replace each CTE chain with a direct
    SELECT from the view.

Query index
-----------
  Q1  get_headline_kpis()              — total visits, admissions, overall rate
  Q2  get_monthly_trend()              — monthly volume + conversion rate (S2)
  Q3  get_segment_conversion()         — segment rollup, no encounter split (S3 treemap)
  Q5  get_ortho_burden_breakdown()     — ortho burden group × encounter type (S5 grouped bar)
  Q6  get_spine_volume_trend()         — monthly spine volume + admissions (S5 callout)
  Q7  get_non_ortho_case_mix()         — patient origin × conversion, standalone segs (S6 bubble)
  Q8  get_workload_vs_conversion()     — clinician caseload bucket → conversion (S7a)
  Q9  get_staffing_trend()             — yearly clinician count + conversion (S7b)
  Q10 get_comorbidity_conversion()     — surgical/non-surgical × comorbid/not (S7c)
  Q11 get_escalation_trend()           — yearly escalation rate + avg hours (S8)
  Q12 get_escalation_investigation_coverage() — investigation coverage by pattern, blended join (retained
                                                 for documentation only — no longer feeds S8 directly)
  Q13 get_ortho_general_conversion_by_year()  — annual conversion rate, Ortho General scope (S8 rebuild)
  Q14 get_escalation_investigation_timing()   — investigation timing split, before/after/none (S8 rebuild)
"""

import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# Database client — matches the pattern used in ksh_clinicals_dashboard.py
# ---------------------------------------------------------------------------
try:
    from snowflake_service.snowflake_client import SnowflakeClient
    _CLIENT = SnowflakeClient()
except ImportError:
    _CLIENT = None


def _run(sql: str) -> pd.DataFrame:
    """Execute SQL and return a DataFrame. Returns empty DataFrame on error."""
    if _CLIENT is None:
        return pd.DataFrame()
    try:
        return _CLIENT.query(sql)
    except Exception as exc:
        st.error(f"Query error: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Shared CTE fragments — assembled into each query that needs them.
# Keeping these as string constants avoids copy-paste drift.
# ---------------------------------------------------------------------------

_VISIT_DIAGNOSES_CTE = """
visit_diagnoses AS (
    SELECT
        visit_id,
        patient_id,
        visit_type,
        department,
        diagnosis_created_at,
        has_chronic_condition,
        TRIM(bg.value::STRING)  AS split_burden,
        LOWER(
            COALESCE(diagnosis_name_expanded, '')
            || ' '
            || COALESCE(icd10_names, '')
        ) AS clean_dx_text
    FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
    LATERAL FLATTEN(input => SPLIT(burden_group, '|')) bg
    GROUP BY ALL
)
"""

_DEPARTMENT_RULES_CTE = """
department_rules (pattern, segment) AS (
    SELECT * FROM (VALUES
        ('%orthopaedic%',   'ortho'),
        ('%orthop%',        'ortho'),
        ('%physiotherapy%', 'ortho'),
        ('%spine%',         'spine'),
        ('%arthroscop%',    'ortho'),
        ('%ent%',           'ent'),
        ('eye consultation%','eye'),
        ('%eye specialist%','eye'),
        ('%urology%',       'urology'),
        ('%gynaec%',        'obgyn'),
        ('%obs/gyn%',       'obgyn'),
        ('%maternity%',     'obgyn'),
        ('%anc%',           'obgyn'),
        ('%cwc%',           'obgyn'),
        ('%surgical%',      'surgery'),
        ('%general surgery%','surgery'),
        ('%neurosurg%',     'neurosurgery'),
        ('%neurology%',     'neurology'),
        ('%mopc%',          'chronic_medical'),
        ('%plastic%',       'plastic'),
        ('%maxillofacial%', 'maxillofacial'),
        ('%maxilofacial%',  'maxillofacial'),
        ('%dental%',        'dental'),
        ('%dermatolog%',    'dermatology')
    ) AS t(pattern, segment)
)
"""

_DEPARTMENT_MATCHED_CTE = """
department_matched AS (
    SELECT visit_id, segment
    FROM (
        SELECT DISTINCT v.visit_id, r.pattern, r.segment
        FROM visit_diagnoses v
        JOIN department_rules r
          ON LOWER(COALESCE(v.department, '')) ILIKE r.pattern
    )
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY visit_id
        ORDER BY LENGTH(pattern) DESC
    ) = 1
)
"""

# Segment flags — one MAX(CASE…) per segment per visit.
# Sepsis exclusion is applied within has_sepsis to remove low-acuity
# respiratory, GI, and skin infections that inflate that bucket.
_FLAGGED_CTE = """
flagged AS (
    SELECT
        vd.visit_id,
        vd.patient_id,
        vd.visit_type,
        ANY_VALUE(vd.diagnosis_created_at)   AS diagnosis_created_at,
        ANY_VALUE(vd.has_chronic_condition)  AS has_chronic_condition,

        MAX(CASE WHEN dm.segment = 'plastic'
                 OR clean_dx_text LIKE '%cleft%'
                 OR clean_dx_text LIKE '%palate%'
                 OR clean_dx_text LIKE '%circum%'
                 OR clean_dx_text LIKE '%lipoma%'
            THEN 1 ELSE 0 END) AS is_plastic,

        MAX(CASE WHEN dm.segment = 'maxillofacial'
                 OR clean_dx_text LIKE '%mandibular%'
                 OR clean_dx_text LIKE '%mandible%'
                 OR clean_dx_text LIKE '%maxillary%'
                 OR clean_dx_text LIKE '%maxilla%'
                 OR clean_dx_text LIKE '%trigeminal%'
                 OR clean_dx_text LIKE '%salivary gland%'
                 OR clean_dx_text LIKE '%sialadenitis%'
                 OR clean_dx_text LIKE '%sialolithiasis%'
            THEN 1 ELSE 0 END) AS is_maxillofacial,

        MAX(CASE WHEN dm.segment = 'dental'
                 OR split_burden LIKE '%Dental%'
                 OR clean_dx_text LIKE '%tooth%'
                 OR clean_dx_text LIKE '%teeth%'
                 OR clean_dx_text LIKE '%gingivit%'
                 OR clean_dx_text LIKE '%caries%'
                 OR clean_dx_text LIKE '%pulpitis%'
                 OR clean_dx_text LIKE '%retained root%'
                 OR clean_dx_text LIKE '%dental%'
            THEN 1 ELSE 0 END) AS is_dental,

        MAX(CASE WHEN dm.segment = 'eye'
                 OR split_burden = 'General: Eye'
                 OR clean_dx_text LIKE '%ophthalmo%'
                 OR clean_dx_text LIKE '%corneal%'
                 OR clean_dx_text LIKE '%conjunctiv%'
                 OR clean_dx_text LIKE '%cataract%'
            THEN 1 ELSE 0 END) AS is_ophthalmology,

        MAX(CASE WHEN dm.segment = 'ent'
                 OR split_burden = 'General: ENT'
                 OR clean_dx_text LIKE '%nasal%'
                 OR clean_dx_text LIKE '%tinnitus%'
                 OR clean_dx_text LIKE '%tonsil%'
                 OR clean_dx_text LIKE '%sinusitis%'
                 OR clean_dx_text LIKE '%rhinitis%'
            THEN 1 ELSE 0 END) AS is_ent,

        MAX(CASE WHEN dm.segment = 'obgyn'
                 OR split_burden IN ('General: Obstetric', 'General: Gynaecology')
                 OR clean_dx_text LIKE '%miscarriage%'
                 OR clean_dx_text LIKE '%ovarian%'
                 OR clean_dx_text LIKE '%myomectomy%'
                 OR clean_dx_text LIKE '%fibroid%'
                 OR clean_dx_text LIKE '%pregnan%'
                 OR clean_dx_text LIKE '%gravid%'
                 OR clean_dx_text LIKE '%delivery%'
                 OR clean_dx_text LIKE '%abortion%'
            THEN 1 ELSE 0 END) AS is_obgyn,

        MAX(CASE WHEN dm.segment = 'spine'
                 OR split_burden = 'Ortho: Spine'
                 OR clean_dx_text LIKE '%spine%'
                 OR clean_dx_text LIKE '%spinal%'
                 OR clean_dx_text LIKE '%lumbar%'
                 OR clean_dx_text LIKE '%lumbago%'
                 OR clean_dx_text LIKE '%sciatica%'
                 OR clean_dx_text LIKE '%disc%'
                 OR clean_dx_text LIKE '%spondyl%'
                 OR clean_dx_text LIKE '%radiculop%'
                 OR clean_dx_text LIKE '%decompression%'
                 OR clean_dx_text LIKE '%laminectomy%'
                 OR clean_dx_text LIKE '%fusion%'
                 OR clean_dx_text LIKE '%sacroilit%'
                 OR clean_dx_text LIKE '%back pain%'
                 OR clean_dx_text LIKE '%piriformis%'
            THEN 1 ELSE 0 END) AS is_spine,

        MAX(CASE WHEN dm.segment = 'ortho'
                 OR split_burden LIKE 'Ortho:%'
                 OR clean_dx_text LIKE '%ingrown%'
                 OR clean_dx_text LIKE '%toe nail%'
                 OR clean_dx_text LIKE '%fracture%'
                 OR clean_dx_text LIKE '% # %'
                 OR clean_dx_text LIKE '%dislocat%'
                 OR clean_dx_text LIKE '%tendin%'
                 OR clean_dx_text LIKE '%sprain%'
                 OR clean_dx_text LIKE '%strain%'
                 OR clean_dx_text LIKE '%arthralgia%'
                 OR clean_dx_text LIKE '%gunstock%'
                 OR clean_dx_text LIKE '%deformity%'
                 OR clean_dx_text LIKE '%osteoarthritis%'
                 OR clean_dx_text LIKE '%myalgia%'
                 OR clean_dx_text LIKE '%bursit%'
                 OR clean_dx_text LIKE '%meniscus%'
                 OR clean_dx_text LIKE '%rotator cuff%'
                 OR clean_dx_text LIKE '%implant%'
            THEN 1 ELSE 0 END) AS is_ortho,

        MAX(CASE WHEN dm.segment = 'surgery'
                 OR split_burden = 'General: Surgery'
                 OR clean_dx_text LIKE '%hernia%'
                 OR clean_dx_text LIKE '%appendic%'
                 OR clean_dx_text LIKE '%fistula%'
                 OR clean_dx_text LIKE '%hydrocele%'
                 OR clean_dx_text LIKE '%stab wound%'
                 OR clean_dx_text LIKE '%deep cut%'
                 OR clean_dx_text LIKE '%cholelithiasis%'
                 OR clean_dx_text LIKE '%cholecystitis%'
            THEN 1 ELSE 0 END) AS is_surgery,

        MAX(CASE WHEN dm.segment = 'neurosurgery'
                 OR clean_dx_text LIKE '%craniotomy%'
                 OR clean_dx_text LIKE '%subdural%'
                 OR clean_dx_text LIKE '%extradural%'
                 OR clean_dx_text LIKE '%epidural hemat%'
                 OR clean_dx_text LIKE '%burr hole%'
                 OR clean_dx_text LIKE '%brain tumour%'
                 OR clean_dx_text LIKE '%brain tumor%'
                 OR clean_dx_text LIKE '%hydrocephalus%'
                 OR clean_dx_text LIKE '%intracerebral%'
                 OR clean_dx_text LIKE '%intra-cerebral%'
                 OR clean_dx_text LIKE '%cerebral hemorrhage%'
                 OR clean_dx_text LIKE '%cerebral haemorrhage%'
                 OR clean_dx_text LIKE '%quadriplegia%'
                 OR clean_dx_text LIKE '%paraplegia%'
            THEN 1 ELSE 0 END) AS is_neurosurgery,

        MAX(CASE WHEN dm.segment = 'neurology'
                 OR split_burden = 'General: Neurology'
                 OR clean_dx_text LIKE '%stroke%'
                 OR clean_dx_text LIKE '%cva%'
                 OR clean_dx_text LIKE '%parkinson%'
                 OR clean_dx_text LIKE '%epilep%'
                 OR clean_dx_text LIKE '%seizure%'
                 OR clean_dx_text LIKE '%guillain%'
                 OR clean_dx_text LIKE '% gbs%'
                 OR clean_dx_text LIKE '%hemiplegia%'
                 OR clean_dx_text LIKE '%hemiparesis%'
                 OR clean_dx_text LIKE '%paraparesis%'
                 OR clean_dx_text LIKE '%quadriparesis%'
                 OR clean_dx_text LIKE '%cerebral palsy%'
                 OR clean_dx_text LIKE '%migraine%'
                 OR clean_dx_text LIKE '%head injury%'
                 OR clean_dx_text LIKE '%dementia%'
            THEN 1 ELSE 0 END) AS is_neurology,

        MAX(CASE WHEN dm.segment = 'urology'
                 OR split_burden = 'General: Genitourinary'
                 OR clean_dx_text LIKE '%bph%'
                 OR clean_dx_text LIKE '%prostat%'
                 OR clean_dx_text LIKE '%urinary%'
                 OR clean_dx_text LIKE '% uti%'
                 OR clean_dx_text LIKE '%uti %'
                 OR clean_dx_text LIKE '%testicular%'
                 OR clean_dx_text LIKE '%undescended%'
                 OR clean_dx_text LIKE '%hypospadia%'
                 OR clean_dx_text LIKE '%cryptorchid%'
                 OR clean_dx_text LIKE '%cryptochirdism%'
            THEN 1 ELSE 0 END) AS is_urology,

        -- Sepsis exclusion: respiratory/GI/skin infections set flag to 0
        -- before the WHEN condition fires, removing low-acuity contamination.
        MAX(CASE
            WHEN clean_dx_text LIKE ANY (
                '%urti%', '%lrti%', '%upper respiratory%', '%lower respiratory%',
                '%tonsillitis%', '%tonsilitis%', '%pharyngitis%', '%rhinitis%',
                '%sinusitis%', '%bronchitis%',
                '%gastroenteritis%', '%acute ge%', ' ge ', '%food poisoning%',
                '%food posioning%', '%amoebiasis%', '%dysentery%',
                '%giardiasis%', '%lambliasis%', '%tinea%', '%ringworm%',
                '%conjunctivitis%'
            ) THEN 0
            WHEN split_burden = 'General: Infection & Sepsis'
                 OR clean_dx_text LIKE '%sepsis%'
                 OR clean_dx_text LIKE '%septic%'
                 OR clean_dx_text LIKE '%septicemia%'
            THEN 1
            ELSE 0
        END) AS has_sepsis,

        MAX(CASE WHEN split_burden = 'General: Cardiovascular'
                 OR clean_dx_text LIKE '%hypertens%'
            THEN 1 ELSE 0 END) AS has_cardio,

        MAX(CASE WHEN split_burden = 'General: Endocrine & Metabolic'
                 OR dm.segment = 'chronic_medical'
                 OR clean_dx_text LIKE '%diabet%'
            THEN 1 ELSE 0 END) AS has_metabolic,

        -- Follow-up / chronic management detector
        MAX(CASE
            WHEN clean_dx_text LIKE '%post %'
                 OR clean_dx_text LIKE '%s/p %'
                 OR clean_dx_text LIKE '%status post%'
                 OR clean_dx_text LIKE '%follow up%'
                 OR clean_dx_text LIKE '%followup%'
                 OR clean_dx_text LIKE '%f/u %'
                 OR clean_dx_text LIKE '%review%'
                 OR clean_dx_text LIKE '%/52%'
                 OR clean_dx_text LIKE '%/12%'
                 OR clean_dx_text LIKE '%weeks post%'
                 OR clean_dx_text LIKE '%months post%'
                 OR clean_dx_text LIKE '%yr post%'
                 OR clean_dx_text LIKE '%yrs post%'
                 OR clean_dx_text LIKE '%healed %'
                 OR clean_dx_text LIKE '%known case of%'
                 OR clean_dx_text LIKE '%known %pt%'
                 OR clean_dx_text LIKE '%known %patient%'
                 OR clean_dx_text LIKE '%on meds%'
                 OR clean_dx_text LIKE '%currently on%'
                 OR clean_dx_text LIKE '%on physio%'
            THEN 1 ELSE 0
        END) AS is_followup_or_chronic_mgmt

    FROM visit_diagnoses vd
    LEFT JOIN department_matched dm ON dm.visit_id = vd.visit_id
    GROUP BY vd.visit_id, vd.patient_id, vd.visit_type
)
"""

_SEGMENT_CASE = """
CASE
    WHEN is_spine        = 1 THEN 'Core Orthopedics: Spine and Back Pain Care'
    WHEN is_ortho        = 1 THEN 'Core Orthopedics: General'
    WHEN is_surgery      = 1 AND is_plastic = 0 THEN 'Core General Surgery'
    WHEN is_plastic      = 1 THEN 'Standalone Specialty: Plastic Surgery'
    WHEN is_maxillofacial= 1 THEN 'Standalone Specialty: Maxillofacial'
    WHEN is_dental       = 1 THEN 'Standalone Specialty: Dental'
    WHEN is_ophthalmology= 1 THEN 'Standalone Specialty: Eye/Ophthalmology'
    WHEN is_ent          = 1 THEN 'Standalone Specialty: ENT'
    WHEN is_obgyn        = 1 THEN 'Standalone Specialty: Obstetrics & Gynaecology'
    WHEN is_neurosurgery = 1 THEN 'Standalone Specialty: Neurosurgery'
    WHEN is_neurology    = 1 THEN 'Standalone Medical: Neurology'
    WHEN is_urology      = 1 THEN 'Standalone Specialty: Urology'
    WHEN has_sepsis      = 1 THEN 'Standalone Medical: Sepsis/Infection'
    WHEN has_cardio      = 1 THEN 'Standalone Medical: Cardiovascular'
    WHEN has_metabolic   = 1 THEN 'Standalone Medical: Endocrine/Metabolic'
    ELSE 'Other General Outpatient'
END AS primary_visit_segment
"""


# ---------------------------------------------------------------------------
# Q1 — Headline KPIs
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_headline_kpis() -> pd.DataFrame:
    """
    Returns one row:
      TOTAL_VISITS, TOTAL_ADMISSIONS, OVERALL_CONVERSION_PCT,
      ACUTE_CONVERSION_PCT (new/acute visits only, latest full year)
    """
    sql = f"""
    WITH {_VISIT_DIAGNOSES_CTE},
         {_DEPARTMENT_RULES_CTE},
         {_DEPARTMENT_MATCHED_CTE},
         {_FLAGGED_CTE},
    segmented AS (
        SELECT
            visit_id, patient_id, visit_type,
            is_followup_or_chronic_mgmt,
            {_SEGMENT_CASE}
        FROM flagged
    )
    SELECT
        COUNT(DISTINCT visit_id)                                              AS TOTAL_VISITS,
        COUNT(DISTINCT CASE WHEN visit_type = 'Inpatient' THEN visit_id END) AS TOTAL_ADMISSIONS,
        ROUND(
            100.0 * COUNT(DISTINCT CASE WHEN visit_type = 'Inpatient' THEN visit_id END)
            / NULLIF(COUNT(DISTINCT visit_id), 0), 2
        )                                                                     AS OVERALL_CONVERSION_PCT,
        ROUND(
            100.0 * COUNT(DISTINCT CASE
                WHEN visit_type = 'Inpatient'
                 AND is_followup_or_chronic_mgmt = 0
                THEN visit_id END)
            / NULLIF(COUNT(DISTINCT CASE
                WHEN is_followup_or_chronic_mgmt = 0
                THEN visit_id END), 0), 2
        )                                                                     AS ACUTE_CONVERSION_PCT
    FROM segmented
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# Q2 — Monthly trend (volume + rate)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_monthly_trend() -> pd.DataFrame:
    """
    Returns one row per month:
      VISIT_MONTH, TOTAL_VISITS, ADMITTED, OUTPATIENT, CONVERSION_PCT

    Pre-Jun 2022 rows have zero admissions (data gap) and should be
    greyed out or excluded in the chart — handled in views.py.
    """
    sql = """
    SELECT
        DATE_TRUNC('month', visit_date)                                       AS VISIT_MONTH,
        COUNT(v.visit_id)                                                     AS TOTAL_VISITS,
        COUNT(a.visit_id)                                                     AS ADMITTED,
        COUNT(v.visit_id) - COUNT(a.visit_id)                                AS OUTPATIENT,
        ROUND(
            DIV0(COUNT(a.visit_id), COUNT(v.visit_id)) * 100.0, 2
        )                                                                     AS CONVERSION_PCT
    FROM HOSPITALS.STAGING.STG_VISITS v
    LEFT JOIN HOSPITALS.STAGING.STG_ADMISSIONS a
      ON v.visit_id = a.visit_id AND v.source_system = a.source_system
    GROUP BY DATE_TRUNC('month', visit_date)
    ORDER BY 1
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# Q3 — Segment conversion rollup (treemap source)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_segment_conversion() -> pd.DataFrame:
    """
    Returns one row per segment:
      PRIMARY_VISIT_SEGMENT, TOTAL_VISITS, INPATIENT_ADMISSIONS,
      CONVERSION_RATE_PCT, PCT_OF_ALL_VISITS
    """
    sql = f"""
    WITH {_VISIT_DIAGNOSES_CTE},
         {_DEPARTMENT_RULES_CTE},
         {_DEPARTMENT_MATCHED_CTE},
         {_FLAGGED_CTE},
    segmented AS (
        SELECT visit_id, visit_type, {_SEGMENT_CASE}
        FROM flagged
    )
    SELECT
        primary_visit_segment                                                  AS PRIMARY_VISIT_SEGMENT,
        COUNT(DISTINCT visit_id)                                               AS TOTAL_VISITS,
        COUNT(DISTINCT CASE WHEN visit_type = 'Inpatient' THEN visit_id END)  AS INPATIENT_ADMISSIONS,
        ROUND(
            100.0 * COUNT(DISTINCT CASE WHEN visit_type = 'Inpatient' THEN visit_id END)
            / NULLIF(COUNT(DISTINCT visit_id), 0), 1
        )                                                                      AS CONVERSION_RATE_PCT,
        ROUND(
            100.0 * COUNT(DISTINCT visit_id)
            / SUM(COUNT(DISTINCT visit_id)) OVER (), 1
        )                                                                      AS PCT_OF_ALL_VISITS
    FROM segmented
    GROUP BY primary_visit_segment
    ORDER BY TOTAL_VISITS DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# Q4 — Segment conversion rollup, by year (trend version of Q3)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_segment_conversion_by_year() -> pd.DataFrame:
    """
    Returns one row per (segment, year):
      PRIMARY_VISIT_SEGMENT, VISIT_YEAR, TOTAL_VISITS, INPATIENT_ADMISSIONS,
      CONVERSION_RATE_PCT
    """
    sql = f"""
    WITH {_VISIT_DIAGNOSES_CTE},
         {_DEPARTMENT_RULES_CTE},
         {_DEPARTMENT_MATCHED_CTE},
         {_FLAGGED_CTE},
    segmented AS (
        SELECT visit_id, visit_type, diagnosis_created_at, {_SEGMENT_CASE}
        FROM flagged
    )
    SELECT
        primary_visit_segment                                                  AS PRIMARY_VISIT_SEGMENT,
        YEAR(diagnosis_created_at)                                             AS VISIT_YEAR,
        COUNT(DISTINCT visit_id)                                               AS TOTAL_VISITS,
        COUNT(DISTINCT CASE WHEN visit_type = 'Inpatient' THEN visit_id END)  AS INPATIENT_ADMISSIONS,
        ROUND(
            100.0 * COUNT(DISTINCT CASE WHEN visit_type = 'Inpatient' THEN visit_id END)
            / NULLIF(COUNT(DISTINCT visit_id), 0), 1
        )                                                                      AS CONVERSION_RATE_PCT
    FROM segmented
    GROUP BY primary_visit_segment, YEAR(diagnosis_created_at)
    ORDER BY primary_visit_segment, VISIT_YEAR
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# Q5 — Ortho burden group × encounter type (S5 grouped bar)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_ortho_burden_breakdown() -> pd.DataFrame:
    """
    Returns one row per (burden_group, encounter_type), scoped to
    Ortho:% burden groups only.

    Columns:
      BURDEN_GROUP, ENCOUNTER_TYPE, TOTAL_VISITS, TOTAL_PATIENTS,
      INPATIENT_ADMISSIONS, CONVERSION_RATE_PCT, PCT_OF_ALL_ORTHO_VISITS
    """
    sql = """
    WITH exploded AS (
        SELECT
            visit_id,
            patient_id,
            visit_type,
            diagnosis_created_at,
            TRIM(bg.value::STRING) AS burden_group,
            LOWER(
                COALESCE(diagnosis_name_expanded, '')
                || ' '
                || COALESCE(icd10_names, '')
            ) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
        LATERAL FLATTEN(input => SPLIT(burden_group, '|')) bg
        WHERE TRIM(bg.value::STRING) LIKE 'Ortho:%'
    ),
    tagged AS (
        SELECT
            *,
            CASE
                WHEN clean_dx_text LIKE '%post %'
                     OR clean_dx_text LIKE '%s/p %'
                     OR clean_dx_text LIKE '%status post%'
                     OR clean_dx_text LIKE '%follow up%'
                     OR clean_dx_text LIKE '%followup%'
                     OR clean_dx_text LIKE '%f/u %'
                     OR clean_dx_text LIKE '%review%'
                     OR clean_dx_text LIKE '%/52%'
                     OR clean_dx_text LIKE '%/12%'
                     OR clean_dx_text LIKE '%weeks post%'
                     OR clean_dx_text LIKE '%months post%'
                     OR clean_dx_text LIKE '%yr post%'
                     OR clean_dx_text LIKE '%yrs post%'
                     OR clean_dx_text LIKE '%healed %'
                     OR clean_dx_text LIKE '%known case of%'
                     OR clean_dx_text LIKE '%known %pt%'
                     OR clean_dx_text LIKE '%known %patient%'
                     OR clean_dx_text LIKE '%on meds%'
                     OR clean_dx_text LIKE '%currently on%'
                     OR clean_dx_text LIKE '%on physio%'
                THEN 'Follow-up / Chronic Mgmt'
                ELSE 'New / Acute'
            END AS encounter_type
        FROM exploded
    )
    SELECT
        burden_group                                                            AS BURDEN_GROUP,
        encounter_type                                                          AS ENCOUNTER_TYPE,
        COUNT(DISTINCT visit_id)                                                AS TOTAL_VISITS,
        COUNT(DISTINCT patient_id)                                              AS TOTAL_PATIENTS,
        COUNT(DISTINCT CASE WHEN visit_type = 'Inpatient' THEN visit_id END)   AS INPATIENT_ADMISSIONS,
        ROUND(
            100.0 * COUNT(DISTINCT CASE WHEN visit_type = 'Inpatient' THEN visit_id END)
            / NULLIF(COUNT(DISTINCT visit_id), 0), 1
        )                                                                       AS CONVERSION_RATE_PCT,
        ROUND(
            100.0 * COUNT(DISTINCT visit_id)
            / SUM(COUNT(DISTINCT visit_id)) OVER (), 1
        )                                                                       AS PCT_OF_ALL_ORTHO_VISITS
    FROM tagged
    GROUP BY burden_group, encounter_type
    ORDER BY burden_group, encounter_type
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# Q6 — Spine volume trend (S5 callout bar)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_spine_volume_trend() -> pd.DataFrame:
    """
    Returns one row per year:
      VISIT_YEAR, TOTAL_VISITS, INPATIENT_ADMISSIONS, CONVERSION_RATE_PCT,
      AVG_MONTHLY_VISITS

    Scoped to the Spine and Back Pain Care segment only.
    """
    sql = f"""
    WITH {_VISIT_DIAGNOSES_CTE},
         {_DEPARTMENT_RULES_CTE},
         {_DEPARTMENT_MATCHED_CTE},
         {_FLAGGED_CTE},
    spine_visits AS (
        SELECT visit_id, visit_type, diagnosis_created_at
        FROM flagged
        WHERE is_spine = 1
    )
    SELECT
        YEAR(diagnosis_created_at)                                             AS VISIT_YEAR,
        COUNT(DISTINCT visit_id)                                               AS TOTAL_VISITS,
        COUNT(DISTINCT CASE WHEN visit_type = 'Inpatient' THEN visit_id END)  AS INPATIENT_ADMISSIONS,
        ROUND(
            100.0 * COUNT(DISTINCT CASE WHEN visit_type = 'Inpatient' THEN visit_id END)
            / NULLIF(COUNT(DISTINCT visit_id), 0), 1
        )                                                                      AS CONVERSION_RATE_PCT,
        ROUND(COUNT(DISTINCT visit_id) / 12.0, 0)                             AS AVG_MONTHLY_VISITS
    FROM spine_visits
    GROUP BY YEAR(diagnosis_created_at)
    ORDER BY 1
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# Q7 — Non-ortho case mix: patient origin vs conversion (S6 bubble)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_non_ortho_case_mix() -> pd.DataFrame:
    """
    For each standalone medical/specialty segment, returns:
      PRIMARY_VISIT_SEGMENT, PATIENT_ORIGIN, TOTAL_VISITS, TOTAL_PATIENTS,
      INPATIENT_ADMISSIONS, CONVERSION_RATE_PCT, PCT_NEVER_ORTHO

    PATIENT_ORIGIN: 'Also seen for orthopaedics' | 'Never seen for orthopaedics'
    Scoped to: Neurology, Urology, Obstetrics & Gynaecology.
    """
    sql = f"""
    WITH {_VISIT_DIAGNOSES_CTE},
         {_DEPARTMENT_RULES_CTE},
         {_DEPARTMENT_MATCHED_CTE},
         {_FLAGGED_CTE},
    segmented AS (
        SELECT
            visit_id, patient_id, visit_type,
            is_ortho, is_spine,
            {_SEGMENT_CASE}
        FROM flagged
    ),
    patient_has_ortho AS (
        SELECT DISTINCT patient_id
        FROM segmented
        WHERE is_ortho = 1 OR is_spine = 1
    ),
    target_segments AS (
        SELECT visit_id, patient_id, visit_type, primary_visit_segment
        FROM segmented
        WHERE primary_visit_segment IN (
            'Standalone Medical: Neurology',
            'Standalone Specialty: Urology',
            'Standalone Specialty: Obstetrics & Gynaecology'
        )
    )
    SELECT
        ts.primary_visit_segment                                               AS PRIMARY_VISIT_SEGMENT,
        CASE WHEN pho.patient_id IS NOT NULL
             THEN 'Also seen for orthopaedics'
             ELSE 'Never seen for orthopaedics'
        END                                                                    AS PATIENT_ORIGIN,
        COUNT(DISTINCT ts.visit_id)                                            AS TOTAL_VISITS,
        COUNT(DISTINCT ts.patient_id)                                          AS TOTAL_PATIENTS,
        COUNT(DISTINCT CASE WHEN ts.visit_type = 'Inpatient' THEN ts.visit_id END)
                                                                               AS INPATIENT_ADMISSIONS,
        ROUND(
            100.0 * COUNT(DISTINCT CASE WHEN ts.visit_type = 'Inpatient' THEN ts.visit_id END)
            / NULLIF(COUNT(DISTINCT ts.visit_id), 0), 1
        )                                                                      AS CONVERSION_RATE_PCT,
        ROUND(
            100.0 * COUNT(DISTINCT CASE WHEN pho.patient_id IS NULL THEN ts.patient_id END)
            / NULLIF(COUNT(DISTINCT ts.patient_id), 0), 1
        )                                                                      AS PCT_NEVER_ORTHO
    FROM target_segments ts
    LEFT JOIN patient_has_ortho pho ON pho.patient_id = ts.patient_id
    GROUP BY ts.primary_visit_segment, patient_origin
    ORDER BY ts.primary_visit_segment, patient_origin
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# Q8 — Clinician caseload vs conversion (S7a bar)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_workload_vs_conversion() -> pd.DataFrame:
    """
    Returns one row per workload bucket:
      WORKLOAD_BUCKET, CLINICIAN_MONTHS, AVG_CONVERSION_RATE_PCT, AVG_WORKLOAD

    Scoped to EMR_V1 only (2022–Jan 2025) where clinician IDs are
    consistent. Cross-period tracking is blocked by the ID scheme
    change at the Feb 2025 EMR cutover.
    """
    sql = """
    WITH visit_diagnoses AS (
        SELECT
            d.visit_id,
            d.visit_type,
            d.diagnosis_created_at,
            d.source_system,
            v.filled_by_user_id
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED d
        LEFT JOIN HOSPITALS.STAGING.STG_SPH_DIAGNOSIS v
          ON d.visit_id = v.visit_id AND d.source_system = v.source_system
        WHERE d.source_system = 'EMR_V1'
        GROUP BY ALL
    ),
    clinician_month AS (
        SELECT
            filled_by_user_id,
            DATE_TRUNC('month', diagnosis_created_at) AS month,
            COUNT(DISTINCT visit_id)                   AS monthly_workload,
            ROUND(
                100.0 * COUNT(DISTINCT CASE WHEN visit_type = 'Inpatient' THEN visit_id END)
                / NULLIF(COUNT(DISTINCT visit_id), 0), 1
            )                                          AS conversion_rate_pct
        FROM visit_diagnoses
        WHERE filled_by_user_id IS NOT NULL
        GROUP BY filled_by_user_id, month
    )
    SELECT
        CASE
            WHEN monthly_workload < 25  THEN '1: <25/month'
            WHEN monthly_workload < 75  THEN '2: 25–74/month'
            WHEN monthly_workload < 150 THEN '3: 75–149/month'
            WHEN monthly_workload < 300 THEN '4: 150–299/month'
            ELSE                             '5: 300+/month'
        END                                            AS WORKLOAD_BUCKET,
        COUNT(*)                                       AS CLINICIAN_MONTHS,
        ROUND(AVG(conversion_rate_pct), 1)            AS AVG_CONVERSION_RATE_PCT,
        ROUND(AVG(monthly_workload), 0)               AS AVG_WORKLOAD
    FROM clinician_month
    GROUP BY WORKLOAD_BUCKET
    ORDER BY WORKLOAD_BUCKET
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# Q9 — Staffing trend: yearly clinician count + conversion (S7b dual line)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_staffing_trend() -> pd.DataFrame:
    """
    Returns one row per year (EMR_V1 only, 2022–2024):
      VISIT_YEAR, ACTIVE_CLINICIANS, AVG_VISITS_PER_CLINICIAN_PER_MONTH,
      AVG_CONVERSION_RATE_PCT
    """
    sql = """
    WITH visit_diagnoses AS (
        SELECT
            d.visit_id,
            d.visit_type,
            d.diagnosis_created_at,
            d.source_system,
            v.filled_by_user_id
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED d
        LEFT JOIN HOSPITALS.STAGING.STG_SPH_DIAGNOSIS v
          ON d.visit_id = v.visit_id AND d.source_system = v.source_system
        WHERE d.source_system = 'EMR_V1'
        GROUP BY ALL
    ),
    clinician_month AS (
        SELECT
            filled_by_user_id,
            DATE_TRUNC('month', diagnosis_created_at) AS month,
            COUNT(DISTINCT visit_id)                   AS monthly_workload,
            ROUND(
                100.0 * COUNT(DISTINCT CASE WHEN visit_type = 'Inpatient' THEN visit_id END)
                / NULLIF(COUNT(DISTINCT visit_id), 0), 1
            )                                          AS conversion_rate_pct
        FROM visit_diagnoses
        WHERE filled_by_user_id IS NOT NULL
        GROUP BY filled_by_user_id, month
    )
    SELECT
        YEAR(month)                             AS VISIT_YEAR,
        COUNT(DISTINCT filled_by_user_id)       AS ACTIVE_CLINICIANS,
        ROUND(AVG(monthly_workload), 0)         AS AVG_VISITS_PER_CLINICIAN_PER_MONTH,
        ROUND(AVG(conversion_rate_pct), 1)     AS AVG_CONVERSION_RATE_PCT
    FROM clinician_month
    GROUP BY YEAR(month)
    ORDER BY VISIT_YEAR
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# Q10 — Comorbidity × surgical/non-surgical (S7c grouped bar)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_comorbidity_conversion() -> pd.DataFrame:
    """
    Returns one row per (segment_type, has_chronic_condition):
      SEGMENT_TYPE, HAS_CHRONIC_CONDITION, TOTAL_VISITS,
      INPATIENT_ADMISSIONS, CONVERSION_RATE_PCT

    SEGMENT_TYPE: 'Surgical' | 'Non-Surgical'
    HAS_CHRONIC_CONDITION: True | False
    """
    sql = f"""
    WITH {_VISIT_DIAGNOSES_CTE},
         {_DEPARTMENT_RULES_CTE},
         {_DEPARTMENT_MATCHED_CTE},
         {_FLAGGED_CTE},
    classified AS (
        SELECT
            visit_id, visit_type, has_chronic_condition,
            CASE
                WHEN is_ortho = 1
                  OR is_spine = 1
                  OR is_surgery = 1
                  OR is_plastic = 1
                  OR is_maxillofacial = 1
                  OR is_neurosurgery = 1
                THEN 'Surgical'
                ELSE 'Non-Surgical'
            END AS segment_type
        FROM flagged
    )
    SELECT
        segment_type                                                             AS SEGMENT_TYPE,
        has_chronic_condition                                                    AS HAS_CHRONIC_CONDITION,
        COUNT(DISTINCT visit_id)                                                 AS TOTAL_VISITS,
        COUNT(DISTINCT CASE WHEN visit_type = 'Inpatient' THEN visit_id END)    AS INPATIENT_ADMISSIONS,
        ROUND(
            100.0 * COUNT(DISTINCT CASE WHEN visit_type = 'Inpatient' THEN visit_id END)
            / NULLIF(COUNT(DISTINCT visit_id), 0), 1
        )                                                                        AS CONVERSION_RATE_PCT
    FROM classified
    GROUP BY segment_type, has_chronic_condition
    ORDER BY segment_type, has_chronic_condition
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# Q11 — 72-hour escalation trend (S8 dual line)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_escalation_trend() -> pd.DataFrame:
    """
    Returns one row per year:
      VISIT_YEAR, TOTAL_TRAUMA_ESCALATIONS, TOTAL_TRAUMA_OPD_VISITS,
      ESCALATION_RATE_PCT, AVG_HOURS_TO_ADMISSION

    Scoped to trauma-pattern Ortho General OPD visits followed by
    inpatient admission within 72 hours.
    """
    sql = """
    WITH opd_visits AS (
        SELECT DISTINCT
            visit_id,
            patient_id,
            diagnosis_created_at AS opd_date,
            LOWER(COALESCE(diagnosis_name_expanded, '')) AS clean_dx
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
        WHERE visit_type = 'Outpatient'
    ),
    ipd_visits AS (
        SELECT DISTINCT
            visit_id,
            patient_id,
            diagnosis_created_at AS ipd_date
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
        WHERE visit_type = 'Inpatient'
    ),
    escalations AS (
        SELECT
            o.visit_id        AS opd_visit_id,
            o.patient_id,
            o.opd_date,
            o.clean_dx,
            DATEDIFF('hour', o.opd_date, i.ipd_date) AS hours_to_admission
        FROM opd_visits o
        JOIN ipd_visits i
          ON i.patient_id = o.patient_id
         AND i.ipd_date > o.opd_date
         AND DATEDIFF('hour', o.opd_date, i.ipd_date) <= 72
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY i.visit_id ORDER BY o.opd_date DESC
        ) = 1
    ),
    trauma_escalations AS (
        SELECT
            YEAR(opd_date)           AS visit_year,
            COUNT(*)                 AS escalation_count,
            ROUND(AVG(hours_to_admission), 1) AS avg_hours
        FROM escalations
        WHERE clean_dx LIKE '%fracture%'
           OR clean_dx LIKE '%dislocat%'
           OR clean_dx LIKE '%avulsion%'
           OR clean_dx LIKE '%laceration%'
           OR clean_dx LIKE '%rupture%'
           OR clean_dx LIKE '%rta%'
           OR clean_dx LIKE '%stab%'
           OR clean_dx LIKE '%gunshot%'
           OR clean_dx LIKE '%crush injury%'
        GROUP BY YEAR(opd_date)
    ),
    trauma_opd AS (
        SELECT
            YEAR(diagnosis_created_at) AS visit_year,
            COUNT(DISTINCT visit_id)   AS trauma_opd_visits
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
        WHERE visit_type = 'Outpatient'
          AND (LOWER(diagnosis_name_expanded) LIKE '%fracture%'
               OR LOWER(diagnosis_name_expanded) LIKE '%dislocat%')
        GROUP BY YEAR(diagnosis_created_at)
    )
    SELECT
        te.visit_year                                                      AS VISIT_YEAR,
        te.escalation_count                                                AS TOTAL_TRAUMA_ESCALATIONS,
        to_.trauma_opd_visits                                             AS TOTAL_TRAUMA_OPD_VISITS,
        ROUND(
            100.0 * te.escalation_count
            / NULLIF(to_.trauma_opd_visits, 0), 1
        )                                                                  AS ESCALATION_RATE_PCT,
        te.avg_hours                                                       AS AVG_HOURS_TO_ADMISSION
    FROM trauma_escalations te
    JOIN trauma_opd to_ ON to_.visit_year = te.visit_year
    ORDER BY VISIT_YEAR
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# Q12 — Escalation investigation coverage (S8 supplement)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_escalation_investigation_coverage() -> pd.DataFrame:
    """
    Returns one row per escalation pattern:
      ESCALATION_PATTERN, TOTAL_ESCALATIONS, HAD_INVESTIGATIONS,
      PCT_HAD_INVESTIGATIONS

    All 72-hour OPD→IPD escalations, joined to STG_IMAGING_ORDERS on
    EITHER the OPD visit_id or the resulting IPD visit_id — the original
    join (OPD visit_id only, plus an AND source_system match) returned an
    implausible ~0.2% match rate. Matching against either visit_id
    (imaging is frequently logged against the admission it precedes, not
    the outpatient visit) and dropping the redundant source_system filter
    raises the match rate to a defensible ~24%, without changing the
    escalation universe itself (still 1,951 total escalations).
    """
    sql = """
    WITH opd_visits AS (
        SELECT DISTINCT
            visit_id, patient_id, diagnosis_created_at AS opd_date,
            LOWER(COALESCE(diagnosis_name_expanded, '')) AS clean_dx
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
        WHERE visit_type = 'Outpatient'
    ),
    ipd_visits AS (
        SELECT DISTINCT visit_id, patient_id, diagnosis_created_at AS ipd_date
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
        WHERE visit_type = 'Inpatient'
    ),
    escalations AS (
        SELECT
            o.visit_id AS opd_visit_id,
            i.visit_id AS ipd_visit_id,
            o.clean_dx,
            DATEDIFF('hour', o.opd_date, i.ipd_date) AS hours_to_admission
        FROM opd_visits o
        JOIN ipd_visits i
            ON i.patient_id = o.patient_id
           AND i.ipd_date > o.opd_date
           AND DATEDIFF('hour', o.opd_date, i.ipd_date) <= 72
        QUALIFY ROW_NUMBER() OVER (PARTITION BY i.visit_id ORDER BY o.opd_date DESC) = 1
    ),
    classified AS (
        SELECT
            *,
            CASE
                WHEN clean_dx LIKE '%fracture%' OR clean_dx LIKE '%dislocat%'
                     OR clean_dx LIKE '%avulsion%' OR clean_dx LIKE '%laceration%'
                     OR clean_dx LIKE '%rupture%' OR clean_dx LIKE '%rta%'
                     OR clean_dx LIKE '%stab%' OR clean_dx LIKE '%gunshot%'
                     OR clean_dx LIKE '%crush injury%'
                THEN 'Acute/Trauma pattern'
                WHEN clean_dx LIKE '%osteoarthritis%' OR clean_dx LIKE '% oa %'
                     OR clean_dx LIKE '%elective%' OR clean_dx LIKE '%scheduled%'
                     OR clean_dx LIKE '%for tkra%' OR clean_dx LIKE '%for thra%'
                THEN 'Elective/Scheduled pattern'
                WHEN clean_dx = '' OR clean_dx IS NULL
                THEN 'Blank diagnosis (data gap)'
                ELSE 'Other/Unclear'
            END AS escalation_pattern
        FROM escalations
    ),
    inv_agg AS (
        SELECT
            c.opd_visit_id,
            c.escalation_pattern,
            COUNT(DISTINCT inv.radno) AS investigation_count
        FROM classified c
        LEFT JOIN HOSPITALS.STAGING.STG_IMAGING_ORDERS inv
            ON inv.visit_id IN (c.opd_visit_id, c.ipd_visit_id)
        GROUP BY c.opd_visit_id, c.escalation_pattern
    )
    SELECT
        escalation_pattern                                                    AS ESCALATION_PATTERN,
        COUNT(*)                                                              AS TOTAL_ESCALATIONS,
        SUM(CASE WHEN investigation_count > 0 THEN 1 ELSE 0 END)             AS HAD_INVESTIGATIONS,
        ROUND(100.0 * SUM(CASE WHEN investigation_count > 0 THEN 1 ELSE 0 END)
              / COUNT(*), 1)                                                  AS PCT_HAD_INVESTIGATIONS
    FROM inv_agg
    GROUP BY escalation_pattern
    ORDER BY TOTAL_ESCALATIONS DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# Q13 — Annual conversion rate, Ortho General scope (S8 rebuild — Chart 1)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_ortho_general_conversion_by_year() -> pd.DataFrame:
    """
    Returns one row per year: VISIT_YEAR, TOTAL_VISITS, INPATIENT_ADMISSIONS,
    CONVERSION_RATE_PCT.

    Scoped to the Ortho General segment (is_ortho = 1, is_spine = 0) —
    matches the escalation-rate query's population (88.6% of 72-hour
    escalations are Ortho General trauma), so the two annual series in the
    rebuilt Chart 1 are actually comparable. Do not swap this for the
    hospital-wide blended conversion rate.
    """
    sql = f"""
    WITH {_VISIT_DIAGNOSES_CTE},
         {_DEPARTMENT_RULES_CTE},
         {_DEPARTMENT_MATCHED_CTE},
         {_FLAGGED_CTE},
    ortho_general_visits AS (
        SELECT visit_id, visit_type, diagnosis_created_at
        FROM flagged
        WHERE is_ortho = 1 AND is_spine = 0
    )
    SELECT
        YEAR(diagnosis_created_at)                                             AS VISIT_YEAR,
        COUNT(DISTINCT visit_id)                                               AS TOTAL_VISITS,
        COUNT(DISTINCT CASE WHEN visit_type = 'Inpatient' THEN visit_id END)  AS INPATIENT_ADMISSIONS,
        ROUND(
            100.0 * COUNT(DISTINCT CASE WHEN visit_type = 'Inpatient' THEN visit_id END)
            / NULLIF(COUNT(DISTINCT visit_id), 0), 1
        )                                                                      AS CONVERSION_RATE_PCT
    FROM ortho_general_visits
    GROUP BY YEAR(diagnosis_created_at)
    ORDER BY 1
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# Q14 — Investigation timing split for escalating patients (S8 rebuild — Chart 2)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_escalation_investigation_timing() -> pd.DataFrame:
    """
    Returns one row per escalation pattern: ESCALATION_PATTERN,
    TOTAL_ESCALATIONS, INVESTIGATED_BEFORE_ADMISSION,
    INVESTIGATED_AFTER_ADMISSION_ONLY, NO_INVESTIGATION, plus the three as
    _PCT columns.

    STG_IMAGING_ORDERS has no date/timestamp column referenced anywhere in
    this codebase — but it doesn't need one here. An investigation row is
    tied to a specific visit_id, either the OPD visit or the IPD admission
    that followed it, not to the patient generically. So the join itself
    carries the timing signal: a row matched on opd_visit_id happened
    at/around the OPD visit (before admission); a row matched only on
    ipd_visit_id happened after. Two separate LEFT JOINs below, checked in
    that order, instead of one combined join plus a date comparison.
    """
    sql = """
    WITH opd_visits AS (
        SELECT DISTINCT
            visit_id, patient_id, diagnosis_created_at AS opd_date,
            LOWER(COALESCE(diagnosis_name_expanded, '')) AS clean_dx
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
        WHERE visit_type = 'Outpatient'
    ),
    ipd_visits AS (
        SELECT DISTINCT visit_id, patient_id, diagnosis_created_at AS ipd_date
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
        WHERE visit_type = 'Inpatient'
    ),
    escalations AS (
        SELECT
            o.visit_id AS opd_visit_id,
            i.visit_id AS ipd_visit_id,
            o.opd_date,
            o.clean_dx,
            DATEDIFF('hour', o.opd_date, i.ipd_date) AS hours_to_admission
        FROM opd_visits o
        JOIN ipd_visits i
            ON i.patient_id = o.patient_id
           AND i.ipd_date > o.opd_date
           AND DATEDIFF('hour', o.opd_date, i.ipd_date) <= 72
        QUALIFY ROW_NUMBER() OVER (PARTITION BY i.visit_id ORDER BY o.opd_date DESC) = 1
    ),
    classified AS (
        SELECT
            *,
            CASE
                WHEN clean_dx LIKE '%fracture%' OR clean_dx LIKE '%dislocat%'
                     OR clean_dx LIKE '%avulsion%' OR clean_dx LIKE '%laceration%'
                     OR clean_dx LIKE '%rupture%' OR clean_dx LIKE '%rta%'
                     OR clean_dx LIKE '%stab%' OR clean_dx LIKE '%gunshot%'
                     OR clean_dx LIKE '%crush injury%'
                THEN 'Acute/Trauma pattern'
                WHEN clean_dx LIKE '%osteoarthritis%' OR clean_dx LIKE '% oa %'
                     OR clean_dx LIKE '%elective%' OR clean_dx LIKE '%scheduled%'
                     OR clean_dx LIKE '%for tkra%' OR clean_dx LIKE '%for thra%'
                THEN 'Elective/Scheduled pattern'
                WHEN clean_dx = '' OR clean_dx IS NULL
                THEN 'Blank diagnosis (data gap)'
                ELSE 'Other/Unclear'
            END AS escalation_pattern
        FROM escalations
    ),
    -- No source_system filter on either join — matches the already-confirmed
    -- fix in get_escalation_investigation_coverage(), where adding a
    -- source_system match dropped the join to an implausible ~0.2% (see
    -- OPD to IPD correction.txt's "collective insights" note on the same
    -- bug); dropping it raised the match rate to a defensible ~24%.
    inv_before AS (
        SELECT DISTINCT c.opd_visit_id
        FROM classified c
        JOIN HOSPITALS.STAGING.STG_IMAGING_ORDERS io ON io.visit_id = c.opd_visit_id
    ),
    inv_after AS (
        SELECT DISTINCT c.opd_visit_id
        FROM classified c
        JOIN HOSPITALS.STAGING.STG_IMAGING_ORDERS io ON io.visit_id = c.ipd_visit_id
    ),
    timed AS (
        SELECT
            c.opd_visit_id,
            c.escalation_pattern,
            CASE
                WHEN ib.opd_visit_id IS NOT NULL THEN 'before'
                WHEN ia.opd_visit_id IS NOT NULL THEN 'after_only'
                ELSE 'none'
            END AS timing
        FROM classified c
        LEFT JOIN inv_before ib ON ib.opd_visit_id = c.opd_visit_id
        LEFT JOIN inv_after ia ON ia.opd_visit_id = c.opd_visit_id
    )
    SELECT
        escalation_pattern                                                       AS ESCALATION_PATTERN,
        COUNT(*)                                                                  AS TOTAL_ESCALATIONS,
        SUM(CASE WHEN timing = 'before' THEN 1 ELSE 0 END)                       AS INVESTIGATED_BEFORE_ADMISSION,
        SUM(CASE WHEN timing = 'after_only' THEN 1 ELSE 0 END)                   AS INVESTIGATED_AFTER_ADMISSION_ONLY,
        SUM(CASE WHEN timing = 'none' THEN 1 ELSE 0 END)                         AS NO_INVESTIGATION,
        ROUND(100.0 * SUM(CASE WHEN timing = 'before' THEN 1 ELSE 0 END) / COUNT(*), 1)     AS PCT_BEFORE,
        ROUND(100.0 * SUM(CASE WHEN timing = 'after_only' THEN 1 ELSE 0 END) / COUNT(*), 1) AS PCT_AFTER_ONLY,
        ROUND(100.0 * SUM(CASE WHEN timing = 'none' THEN 1 ELSE 0 END) / COUNT(*), 1)       AS PCT_NONE
    FROM timed
    GROUP BY escalation_pattern
    ORDER BY TOTAL_ESCALATIONS DESC
    """
    return _run(sql)