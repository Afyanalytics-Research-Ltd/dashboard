"""
sph/disease_burden_module/orthopedics/orth_queries.py
========================================================
All SQL for the Disease Burden → Orthopedics sub-tab.

Rules enforced here:
  - Every function is decorated with @st.cache_data(ttl=3600).
  - Every function returns a pd.DataFrame.
  - No rendering logic — zero st.* calls except the cache decorator.
  - Named get_orth_* to namespace from the other tabs' queries.
  - Uses the validated/deduplicated query versions from
    "Orthopedics Only queries.txt" — never the superseded first-draft
    versions (see that file's "QUERIES TO IGNORE" section).
"""

import pandas as pd
import streamlit as st

from sph.clinicals.opd_ipd_module.queries import _run


# ---------------------------------------------------------------------------
# Q5 — Population demographics (trauma-type vs. degenerative-type)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_orth_population_demographics() -> pd.DataFrame:
    sql = """
    WITH ortho_classified AS (
        SELECT
            visit_id, source_system,
            CASE
                WHEN LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%fracture%'
                     OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%dislocat%'
                    THEN 'Trauma-type (fracture/dislocation)'
                WHEN LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%osteoarthritis%'
                    THEN 'Degenerative-type (osteoarthritis)'
                ELSE NULL
            END AS injury_type
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
    ),
    with_demo AS (
        SELECT
            oc.injury_type,
            v.age_group,
            CASE
                WHEN LOWER(v.gender) = 'm' THEN 'male'
                WHEN LOWER(v.gender) = 'f' THEN 'female'
                ELSE LOWER(v.gender)
            END AS gender
        FROM ortho_classified oc
        LEFT JOIN HOSPITALS.STAGING.STG_VISITS v ON v.visit_id = oc.visit_id AND v.source_system = oc.source_system
        WHERE oc.injury_type IS NOT NULL
    )
    SELECT
        injury_type                                                                AS INJURY_TYPE,
        COALESCE(age_group, 'Unknown')                                             AS AGE_GROUP,
        gender                                                                      AS GENDER,
        COUNT(*)                                                                    AS TOTAL_VISITS,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY injury_type), 1)  AS PCT_WITHIN_INJURY_TYPE
    FROM with_demo
    WHERE gender IN ('male', 'female')
    GROUP BY injury_type, AGE_GROUP, gender
    ORDER BY injury_type, TOTAL_VISITS DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# Q3/Q4 — Spine case-type by year, and top spine diagnoses
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_orth_spine_casetype_by_year() -> pd.DataFrame:
    sql = """
    WITH spine_visits AS (
        SELECT
            visit_id, visit_type, diagnosis_created_at,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
        WHERE LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%spine%'
           OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%spinal%'
           OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%lumbar%'
           OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%sciatica%'
           OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%disc%'
           OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%spondyl%'
    ),
    classified AS (
        SELECT
            *,
            CASE
                WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%dislocat%'
                     OR clean_dx_text LIKE '%instability%' OR clean_dx_text LIKE '%stenosis%'
                     OR clean_dx_text LIKE '%herniat%' OR clean_dx_text LIKE '%prolapse%'
                     OR clean_dx_text LIKE '%compression%' OR clean_dx_text LIKE '%tumour%'
                     OR clean_dx_text LIKE '%tumor%' OR clean_dx_text LIKE '%infection%'
                    THEN 'Structural / potentially surgical'
                WHEN clean_dx_text LIKE '%back pain%' OR clean_dx_text LIKE '%backache%'
                     OR clean_dx_text LIKE '%lumbago%' OR clean_dx_text LIKE '%sciatica%'
                     OR clean_dx_text LIKE '%muscle spasm%' OR clean_dx_text LIKE '%strain%'
                    THEN 'General pain / likely conservative management'
                ELSE 'Other / unclear'
            END AS spine_case_type
        FROM spine_visits
    )
    SELECT
        YEAR(diagnosis_created_at)                                                       AS YEAR,
        spine_case_type                                                                  AS SPINE_CASE_TYPE,
        COUNT(DISTINCT visit_id)                                                         AS TOTAL_VISITS,
        COUNT(DISTINCT CASE WHEN visit_type = 'Inpatient' THEN visit_id END)              AS INPATIENT_ADMISSIONS,
        ROUND(100.0 * COUNT(DISTINCT CASE WHEN visit_type = 'Inpatient' THEN visit_id END)
                   / NULLIF(COUNT(DISTINCT visit_id), 0), 1)                              AS CONVERSION_RATE_PCT,
        ROUND(100.0 * COUNT(DISTINCT visit_id)
              / SUM(COUNT(DISTINCT visit_id)) OVER (PARTITION BY YEAR(diagnosis_created_at)), 1)
                                                                                            AS PCT_OF_YEAR_SPINE_VOLUME
    FROM classified
    WHERE YEAR(diagnosis_created_at) BETWEEN 2022 AND 2025
    GROUP BY YEAR, spine_case_type
    ORDER BY YEAR, TOTAL_VISITS DESC
    """
    return _run(sql)


@st.cache_data(ttl=3600)
def get_orth_top_spine_diagnoses() -> pd.DataFrame:
    """Deduplicated version — see Query 3 note in Orthopedics Only queries.txt."""
    sql = """
    WITH spine_visits AS (
        SELECT
            visit_id, diagnosis_created_at,
            diagnosis_name_expanded, icd10_names,
            CASE
                WHEN icd10_names IS NULL OR TRIM(icd10_names) = '' THEN LOWER(COALESCE(diagnosis_name_expanded, ''))
                WHEN diagnosis_name_expanded IS NULL OR TRIM(diagnosis_name_expanded) = '' THEN LOWER(COALESCE(icd10_names, ''))
                WHEN LOWER(icd10_names) LIKE '%' || LOWER(diagnosis_name_expanded) || '%' THEN LOWER(icd10_names)
                WHEN LOWER(diagnosis_name_expanded) LIKE '%' || LOWER(icd10_names) || '%' THEN LOWER(diagnosis_name_expanded)
                ELSE LOWER(diagnosis_name_expanded) || ' ' || LOWER(icd10_names)
            END AS clean_dx_text_deduped
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
        WHERE LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%spine%'
           OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%spinal%'
           OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%lumbar%'
           OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%sciatica%'
           OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%disc%'
           OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%spondyl%'
    )
    SELECT
        YEAR(diagnosis_created_at)         AS YEAR,
        clean_dx_text_deduped              AS CLEAN_DX_TEXT_DEDUPED,
        COUNT(DISTINCT visit_id)           AS OCCURRENCES
    FROM spine_visits
    WHERE YEAR(diagnosis_created_at) BETWEEN 2022 AND 2025
    GROUP BY YEAR, clean_dx_text_deduped
    QUALIFY ROW_NUMBER() OVER (PARTITION BY YEAR ORDER BY OCCURRENCES DESC) <= 15
    ORDER BY YEAR, OCCURRENCES DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# Q1 — Top procedures
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_orth_top_procedures() -> pd.DataFrame:
    sql = """
    SELECT
        INITCAP(p.procedure_name) AS PROCEDURE_NAME,
        COUNT(*)                  AS OCCURRENCES,
        COUNT(DISTINCT p.visit_id) AS DISTINCT_VISITS
    FROM HOSPITALS.STAGING.STG_PROCEDURES p
    WHERE p.procedure_name IS NOT NULL
    GROUP BY p.procedure_name
    ORDER BY OCCURRENCES DESC
    LIMIT 40
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# Q2 — Imaging coverage
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_orth_imaging_coverage() -> pd.DataFrame:
    sql = """
    WITH ortho_visits AS (
        SELECT visit_id, source_system, visit_type
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
        WHERE LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%fracture%'
           OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%dislocat%'
           OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%osteoarthritis%'
    ),
    imaging_check AS (
        SELECT
            ov.visit_id, ov.visit_type,
            COUNT(io.visit_id) AS imaging_row_count,
            MAX(CASE WHEN io.modality_group = 'X-Ray' THEN 1 ELSE 0 END) AS has_xray
        FROM ortho_visits ov
        LEFT JOIN HOSPITALS.STAGING.STG_IMAGING_ORDERS io
            ON io.visit_id = ov.visit_id AND io.source_system = ov.source_system
        GROUP BY ov.visit_id, ov.visit_type
    )
    SELECT
        visit_type                                                                     AS VISIT_TYPE,
        COUNT(*)                                                                        AS TOTAL_ORTHO_VISITS,
        SUM(CASE WHEN imaging_row_count > 0 THEN 1 ELSE 0 END)                          AS VISITS_WITH_ANY_IMAGING,
        ROUND(100.0 * SUM(CASE WHEN imaging_row_count > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS PCT_WITH_ANY_IMAGING,
        SUM(has_xray)                                                                    AS VISITS_WITH_XRAY_SPECIFICALLY,
        ROUND(100.0 * SUM(has_xray) / COUNT(*), 1)                                       AS PCT_WITH_XRAY
    FROM imaging_check
    GROUP BY visit_type
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# Q7 — Ortho-scoped scheduled follow-up continuity
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_orth_followup_continuity() -> pd.DataFrame:
    sql = """
    WITH oie_flat AS (
        SELECT
            o."id" AS oie_id, o."ref" AS soi_id, o."item" AS sale_item_id,
            f.VALUE:"name"::STRING AS field_name, v.VALUE::STRING AS field_value
        FROM HOSPITALS.ORTHOPEDIC_CLEAN."ORDERITEMENTRIES" o,
        LATERAL FLATTEN(INPUT => TRY_PARSE_JSON(o."fields"), OUTER => TRUE) f,
        LATERAL FLATTEN(INPUT => f.VALUE:"value", OUTER => TRUE) v
        WHERE f.VALUE:"name"::STRING IS NOT NULL
    ),
    doctor_raw AS (
        SELECT oie.oie_id, oie.soi_id, oie.field_name, oie.field_value
        FROM oie_flat oie
        JOIN HOSPITALS.ORTHOPEDIC_CLEAN."SALEITEMS" si ON si."id" = oie.sale_item_id
        WHERE si."category" = 'Doctor'
    ),
    doctor_pivot AS (
        SELECT
            oie_id, soi_id,
            MAX(CASE WHEN field_name = 'Schedule Follow Up' THEN TRY_TO_TIMESTAMP(field_value) END) AS scheduled_follow_up_date
        FROM doctor_raw
        GROUP BY oie_id, soi_id
    ),
    soi AS (SELECT "id" AS soi_id, "patient" AS patient_id FROM HOSPITALS.ORTHOPEDIC_CLEAN."SINGLEORDERITEMS"),
    orders_flat AS (
        SELECT oo.id AS order_id, f.VALUE::STRING AS item_id, oo.registered_at AS consult_date
        FROM HOSPITALS.ORTHOPEDIC_CLEAN."ORTHO_ORDERS" oo,
        LATERAL FLATTEN(INPUT => TRY_PARSE_JSON(oo.items)) f
        QUALIFY ROW_NUMBER() OVER (PARTITION BY f.VALUE::STRING ORDER BY oo.registered_at DESC) = 1
    ),
    consultations AS (
        SELECT of_.order_id AS visit_id, s.patient_id, of_.consult_date, dp.scheduled_follow_up_date
        FROM doctor_pivot dp
        JOIN soi s ON s.soi_id = dp.soi_id
        JOIN orders_flat of_ ON of_.item_id = dp.soi_id
    ),
    ortho_only AS (
        SELECT DISTINCT visit_id
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
        WHERE LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%fracture%'
           OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%osteoarthritis%'
           OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%dislocat%'
           OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%arthroplasty%'
    ),
    with_return AS (
        SELECT
            c.visit_id, c.patient_id, c.scheduled_follow_up_date,
            MIN(c2.consult_date) AS next_consult_date
        FROM consultations c
        JOIN ortho_only oo ON oo.visit_id = c.visit_id
        LEFT JOIN consultations c2 ON c2.patient_id = c.patient_id AND c2.consult_date > c.consult_date
        WHERE c.scheduled_follow_up_date IS NOT NULL
        GROUP BY c.visit_id, c.patient_id, c.scheduled_follow_up_date
    )
    SELECT
        COUNT(*)                                                                        AS TOTAL_SCHEDULED,
        SUM(CASE WHEN next_consult_date IS NOT NULL THEN 1 ELSE 0 END)                  AS ATTENDED,
        ROUND(100.0 * SUM(CASE WHEN next_consult_date IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 1) AS PCT_ATTENDED,
        ROUND(AVG(CASE WHEN next_consult_date IS NOT NULL
                  THEN DATEDIFF('day', scheduled_follow_up_date, next_consult_date) END), 1) AS AVG_DAYS_EARLY_OR_LATE
    FROM with_return
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# Q8 — Complications
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_orth_complications() -> pd.DataFrame:
    sql = """
    SELECT
        CASE
            WHEN LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%non union%'
                 OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%nonunion%'
                 OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%malunion%'
                THEN 'Non-union / malunion'
            WHEN LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%dislocat%'
                 AND (LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%prosthe%'
                      OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%replacement%'
                      OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%arthroplasty%')
                THEN 'Post-arthroplasty dislocation'
            WHEN LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%hardware fail%'
                 OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%implant fail%'
                 OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%broken implant%'
                 OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%failed implant%'
                THEN 'Hardware failure'
            WHEN LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%dvt%'
                 OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%deep vein thrombosis%'
                THEN 'DVT'
            WHEN LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%pulmonary embol%'
                 OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '% pe %'
                THEN 'Pulmonary embolism'
            ELSE NULL
        END AS COMPLICATION_TYPE,
        COUNT(*)                  AS OCCURRENCES,
        COUNT(DISTINCT visit_id)  AS DISTINCT_VISITS
    FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
    WHERE
        LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%non union%'
        OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%nonunion%'
        OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%malunion%'
        OR (LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%dislocat%'
            AND (LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%prosthe%'
                 OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%replacement%'
                 OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%arthroplasty%'))
        OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%hardware fail%'
        OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%implant fail%'
        OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%broken implant%'
        OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%failed implant%'
        OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%dvt%'
        OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%deep vein thrombosis%'
        OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%pulmonary embol%'
        OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '% pe %'
    GROUP BY COMPLICATION_TYPE
    ORDER BY OCCURRENCES DESC
    """
    df = _run(sql)
    if not df.empty:
        df = df[df["COMPLICATION_TYPE"].notna()]
    return df


# ---------------------------------------------------------------------------
# Standard 1 — VTE prophylaxis compliance (compliance column only)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_orth_vte_compliance() -> pd.DataFrame:
    sql = """
    WITH major_ortho_procedures AS (
        SELECT visit_id, source_system, procedure_name,
            CASE
                WHEN UPPER(procedure_name) LIKE '%TKRA%' OR UPPER(procedure_name) LIKE '%TOTAL KNEE%'
                    THEN 'Total Knee Replacement'
                WHEN UPPER(procedure_name) LIKE '%THRA%' OR UPPER(procedure_name) LIKE '%TOTAL HIP%'
                    THEN 'Total Hip Replacement'
                WHEN UPPER(procedure_name) LIKE '%NECK OF FEMUR%' OR UPPER(procedure_name) LIKE '%DYNAMIC HIP SCREW%'
                    THEN 'Hip Fracture Surgery'
                WHEN UPPER(procedure_name) LIKE '%FEMUR NAILING%' OR UPPER(procedure_name) LIKE '%TIBIA NAILING%'
                    THEN 'Long Bone Nailing (major fixation)'
                ELSE NULL
            END AS major_procedure_category
        FROM HOSPITALS.STAGING.STG_PROCEDURES
    ),
    flagged_procedures AS (
        SELECT * FROM major_ortho_procedures WHERE major_procedure_category IS NOT NULL
    ),
    anticoagulant_rx AS (
        SELECT DISTINCT visit_id, source_system
        FROM HOSPITALS.STAGING.STG_PHARMACY_ORDERS
        WHERE LOWER(item_name) LIKE ANY (
            '%rivaroxaban%', '%enoxaparin%', '%clexane%', '%heparin%', '%warfarin%', '%aspirin%'
        )
    ),
    dvt_diagnosis AS (
        SELECT DISTINCT visit_id, source_system
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
        WHERE LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%dvt%'
           OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%deep vein thrombosis%'
    )
    SELECT
        fp.major_procedure_category                                                     AS MAJOR_PROCEDURE_CATEGORY,
        COUNT(DISTINCT fp.visit_id)                                                      AS TOTAL_PROCEDURES,
        COUNT(DISTINCT ar.visit_id)                                                      AS WITH_ANTICOAGULANT,
        ROUND(100.0 * COUNT(DISTINCT ar.visit_id) / COUNT(DISTINCT fp.visit_id), 1)      AS PCT_PROPHYLAXIS_COMPLIANCE,
        COUNT(DISTINCT dv.visit_id)                                                      AS CONFIRMED_DVT_CASES,
        ROUND(100.0 * COUNT(DISTINCT dv.visit_id) / COUNT(DISTINCT fp.visit_id), 2)      AS DVT_RATE_PCT
    FROM flagged_procedures fp
    LEFT JOIN anticoagulant_rx ar ON ar.visit_id = fp.visit_id AND ar.source_system = fp.source_system
    LEFT JOIN dvt_diagnosis dv ON dv.visit_id = fp.visit_id AND dv.source_system = fp.source_system
    GROUP BY fp.major_procedure_category
    ORDER BY TOTAL_PROCEDURES DESC
    """
    df = _run(sql)
    # DVT_RATE_PCT is confirmed unreliable (linkage gap — see build spec
    # Section 8.6). Never let it reach the view.
    return df.drop(columns=["DVT_RATE_PCT"], errors="ignore")


# ---------------------------------------------------------------------------
# Standard 2, Step 2 — Open fracture antibiotic coverage
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_orth_open_fracture_antibiotics() -> pd.DataFrame:
    sql = """
    WITH open_fractures AS (
        SELECT visit_id, source_system,
            LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
        WHERE LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%fracture%'
          AND (
            LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%open%'
            OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%compound%'
            OR LOWER(COALESCE(diagnosis_name_expanded,'') || ' ' || COALESCE(icd10_names,'')) LIKE '%degloving%'
          )
    ),
    antibiotic_rx AS (
        SELECT visit_id, source_system, MIN(request_date) AS first_antibiotic_date
        FROM HOSPITALS.STAGING.STG_PHARMACY_ORDERS
        WHERE LOWER(item_name) LIKE ANY (
            '%amoxicillin%', '%augmentin%', '%ceftriaxone%', '%ciprofloxacin%', '%metronidazole%',
            '%flucloxacillin%', '%gentamicin%', '%vancomycin%', '%clindamycin%', '%meropenem%',
            '%piperacillin%', '%cefuroxime%', '%levofloxacin%'
        )
        GROUP BY visit_id, source_system
    ),
    with_admission AS (
        SELECT
            of_.visit_id, of_.source_system,
            a.admission_date,
            ar.first_antibiotic_date,
            DATEDIFF('day', a.admission_date, ar.first_antibiotic_date) AS days_to_antibiotic
        FROM open_fractures of_
        LEFT JOIN antibiotic_rx ar ON ar.visit_id = of_.visit_id AND ar.source_system = of_.source_system
        LEFT JOIN HOSPITALS.STAGING.STG_ADMISSIONS a ON a.visit_id = of_.visit_id AND a.source_system = of_.source_system
    )
    SELECT
        COUNT(*)                                                                        AS TOTAL_OPEN_FRACTURE_VISITS,
        SUM(CASE WHEN first_antibiotic_date IS NOT NULL THEN 1 ELSE 0 END)              AS WITH_ANY_ANTIBIOTIC,
        ROUND(100.0 * SUM(CASE WHEN first_antibiotic_date IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 1)
                                                                                          AS PCT_WITH_ANTIBIOTIC,
        SUM(CASE WHEN days_to_antibiotic = 0 THEN 1 ELSE 0 END)                          AS SAME_DAY_AS_ADMISSION,
        ROUND(100.0 * SUM(CASE WHEN days_to_antibiotic = 0 THEN 1 ELSE 0 END)
                   / NULLIF(SUM(CASE WHEN first_antibiotic_date IS NOT NULL THEN 1 ELSE 0 END), 0), 1)
                                                                                          AS PCT_SAME_DAY_OF_THOSE_TREATED
    FROM with_admission
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# Headline KPIs — assembled from the queries above, not one monolithic SQL
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_orth_headline_kpis() -> pd.DataFrame:
    # Spine's share of TOTAL HOSPITAL volume (not its internal case-type
    # split, which get_orth_spine_casetype_by_year() covers separately) —
    # reuse the Case Mix module's existing, validated computation of this
    # exact metric rather than duplicating it here.
    import sph.clinicals.case_mix_module.cm_queries as CMQ
    df_case_mix = CMQ.get_cm_headline_kpis()

    df_followup = get_orth_followup_continuity()
    df_complications = get_orth_complications()
    df_vte = get_orth_vte_compliance()

    spine_2022 = spine_latest = 0.0
    if not df_case_mix.empty:
        row = df_case_mix.iloc[0]
        spine_2022 = float(row.get("SPINE_SHARE_2022_PCT", 0) or 0)
        spine_latest = float(row.get("SPINE_SHARE_LATEST_PCT", 0) or 0)

    nonunion_count = 0
    if not df_complications.empty:
        m = df_complications[df_complications["COMPLICATION_TYPE"] == "Non-union / malunion"]
        nonunion_count = int(m.iloc[0]["DISTINCT_VISITS"]) if not m.empty else 0

    followup_pct = avg_days_late = 0.0
    if not df_followup.empty:
        followup_pct = float(df_followup.iloc[0].get("PCT_ATTENDED", 0) or 0)
        avg_days_late = float(df_followup.iloc[0].get("AVG_DAYS_EARLY_OR_LATE", 0) or 0)

    tkr_compliance = 0.0
    if not df_vte.empty:
        tkr = df_vte[df_vte["MAJOR_PROCEDURE_CATEGORY"] == "Total Knee Replacement"]
        tkr_compliance = float(tkr.iloc[0]["PCT_PROPHYLAXIS_COMPLIANCE"]) if not tkr.empty else 0.0

    return pd.DataFrame([{
        "SPINE_SHARE_2022_PCT": spine_2022,
        "SPINE_SHARE_LATEST_PCT": spine_latest,
        "NONUNION_COUNT": nonunion_count,
        "FOLLOWUP_ATTENDANCE_PCT": followup_pct,
        "AVG_DAYS_LATE": avg_days_late,
        "TKR_VTE_COMPLIANCE_PCT": tkr_compliance,
    }])
