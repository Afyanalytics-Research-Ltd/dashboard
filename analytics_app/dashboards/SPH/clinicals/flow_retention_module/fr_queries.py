"""
sph/flow_retention_module/fr_queries.py
==========================================
All SQL for the SPH Flow and Retention tab, replaced verbatim from
"Flow and retention queries.txt" (the verified query set behind
sph_clinical_activity_build_spec.md's Flow and Retention companion,
Flow_Retention_Template_B_Build_Spec.md).

Segment taxonomy (used by every query below via the shared CASE logic,
re-interpolated per function since Snowflake CTEs do not persist across
statement boundaries):
    Core Orthopedics: General, Spine-conservative, Spine-structural,
    ANC / Routine Pregnancy, High-Risk Pregnancy, Fibroids-conservative
  EXCLUDE: General Surgery and EXCLUDE: Fibroids-surgical are filtered
  out entirely, matching the source query file's own exclusion rule.

Rules enforced here:
  - Every function is decorated with @st.cache_data(ttl=3600).
  - Every function returns a pd.DataFrame.
  - No rendering logic — zero st.* calls except the cache decorator.
  - Query bodies are transcribed as-given from the verified source file,
    not rewritten — only wrapped in Python and given a return statement.
    Column names come back upper-cased by Snowflake's default identifier
    casing (unquoted aliases), matching this codebase's convention.
"""

import decimal

import pandas as pd
import streamlit as st

from sph.clinicals.opd_ipd_module.queries import _run as _run_raw


def _run(sql: str) -> pd.DataFrame:
    """Wraps the shared _run() and coerces decimal.Decimal columns (as
    returned by the Snowflake connector for ROUND()/AVG()/MEDIAN()/etc.)
    to float64. Without this, Decimal values mix badly with plain Python
    floats used elsewhere and raise TypeError on any +/-/*// between the
    two — the same class of bug hit and fixed in ca_queries.py."""
    df = _run_raw(sql)
    if df is None or df.empty:
        return df
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, decimal.Decimal)).any():
            df[col] = df[col].astype(float)
    return df


# ---------------------------------------------------------------------------
# A1 Part A — overall status (window-scoped, 365-day population)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_fr_status_overall() -> pd.DataFrame:
    """Columns: STATUS, TOTAL_PATIENTS, PCT_OF_CLASSIFIABLE_PATIENTS."""
    sql = """
    WITH raw_diag AS (
        SELECT
            visit_id, patient_id, diagnosis_created_at AS visit_date,
            TRIM(bg.value::STRING) AS split_burden,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
        LATERAL FLATTEN(input => SPLIT(burden_group, '|')) bg
    ),
    segmented AS (
        SELECT DISTINCT
            visit_id, patient_id, visit_date,
            CASE
                WHEN clean_dx_text LIKE '%hernia%' THEN 'EXCLUDE: General Surgery'
                WHEN (clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%')
                     AND (clean_dx_text LIKE '%post %' OR clean_dx_text LIKE '%revision%')
                    THEN 'EXCLUDE: Fibroids-surgical'
                WHEN clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%'
                    THEN 'Fibroids-conservative'
                WHEN clean_dx_text LIKE '%spine%' OR clean_dx_text LIKE '%sciatica%' THEN
                    CASE
                        WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%dislocat%' OR clean_dx_text LIKE '%stenosis%'
                             OR clean_dx_text LIKE '%herniat%' OR clean_dx_text LIKE '%compression%' OR clean_dx_text LIKE '%tumour%'
                            THEN 'Spine-structural'
                        ELSE 'Spine-conservative'
                    END
                WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%osteoarthritis%' THEN 'Core Orthopedics: General'
                WHEN clean_dx_text LIKE '%pre-eclamp%' OR clean_dx_text LIKE '%preeclamp%' OR clean_dx_text LIKE '%eclamp%'
                     OR clean_dx_text LIKE '%hyperem%' OR clean_dx_text LIKE '%pprom%'
                    THEN 'High-Risk Pregnancy'
                WHEN clean_dx_text LIKE '%pregnan%' OR clean_dx_text LIKE '%anc%' OR clean_dx_text LIKE '%antenatal%'
                     OR clean_dx_text LIKE '%gravid%' OR split_burden = 'General: Obstetric'
                    THEN 'ANC / Routine Pregnancy'
                ELSE 'Other'
            END AS segment
        FROM raw_diag
    ),
    classifiable_all_time AS (
        SELECT * FROM segmented
        WHERE segment NOT IN ('EXCLUDE: General Surgery', 'EXCLUDE: Fibroids-surgical', 'Other')
    ),
    dataset_max_date AS (SELECT MAX(visit_date) AS max_date FROM classifiable_all_time),
    window_bounds AS (
        SELECT DATEADD('day', -365, max_date) AS window_start, max_date AS window_end
        FROM dataset_max_date
    ),
    patients_active_in_window AS (
        SELECT DISTINCT ca.patient_id, ca.segment
        FROM classifiable_all_time ca
        CROSS JOIN window_bounds wb
        WHERE ca.visit_date BETWEEN wb.window_start AND wb.window_end
    ),
    relevant_visits AS (
        SELECT ca.visit_id, ca.patient_id, ca.segment, ca.visit_date
        FROM classifiable_all_time ca
        JOIN patients_active_in_window paiw
            ON paiw.patient_id = ca.patient_id AND paiw.segment = ca.segment
    ),
    pregnancy_capped AS (
        SELECT
            rv.*,
            MIN(rv.visit_date) OVER (PARTITION BY rv.patient_id, rv.segment) AS first_preg_visit
        FROM relevant_visits rv
        WHERE rv.segment IN ('ANC / Routine Pregnancy', 'High-Risk Pregnancy')
    ),
    pregnancy_within_window AS (
        SELECT visit_id, patient_id, visit_date, segment
        FROM pregnancy_capped
        WHERE DATEDIFF('day', first_preg_visit, visit_date) <= 300
    ),
    non_pregnancy AS (
        SELECT visit_id, patient_id, visit_date, segment
        FROM relevant_visits
        WHERE segment NOT IN ('ANC / Routine Pregnancy', 'High-Risk Pregnancy')
    ),
    final_visits AS (
        SELECT * FROM non_pregnancy
        UNION ALL
        SELECT * FROM pregnancy_within_window
    ),
    patient_segment_last_visit AS (
        SELECT patient_id, segment, MAX(visit_date) AS last_visit_date
        FROM final_visits
        GROUP BY patient_id, segment
    ),
    consecutive_gaps AS (
        SELECT
            patient_id, segment, visit_date,
            LAG(visit_date) OVER (PARTITION BY patient_id, segment ORDER BY visit_date) AS prior_visit_date
        FROM final_visits
    ),
    segment_tc AS (
        SELECT
            segment,
            MEDIAN(DATEDIFF('day', prior_visit_date, visit_date)) AS t_c_days,
            COUNT(*) AS gap_observations
        FROM consecutive_gaps
        WHERE prior_visit_date IS NOT NULL
          AND segment IN ('ANC / Routine Pregnancy', 'High-Risk Pregnancy', 'Core Orthopedics: General',
                           'Spine-structural', 'Fibroids-conservative')
        GROUP BY segment
    ),
    trailing_90_visit_count AS (
        SELECT
            fv.patient_id, fv.segment,
            COUNT(*) AS visits_in_trailing_90
        FROM final_visits fv
        JOIN patient_segment_last_visit psl
            ON psl.patient_id = fv.patient_id AND psl.segment = fv.segment
        CROSS JOIN dataset_max_date dmd
        WHERE DATEDIFF('day', fv.visit_date, dmd.max_date) <= 90
        GROUP BY fv.patient_id, fv.segment
    ),
    classified AS (
        SELECT
            psl.patient_id, psl.segment, psl.last_visit_date,
            DATEDIFF('day', psl.last_visit_date, dmd.max_date) AS days_since_last_visit,
            st.t_c_days,
            COALESCE(t90.visits_in_trailing_90, 0) AS visits_in_trailing_90,
            CASE
                WHEN DATEDIFF('day', psl.last_visit_date, dmd.max_date) > 180 THEN 'LTFU'
                WHEN DATEDIFF('day', psl.last_visit_date, dmd.max_date) > 90 THEN 'Lapsing'
                ELSE 'Active'
            END AS status
        FROM patient_segment_last_visit psl
        CROSS JOIN dataset_max_date dmd
        LEFT JOIN segment_tc st ON st.segment = psl.segment
        LEFT JOIN trailing_90_visit_count t90 ON t90.patient_id = psl.patient_id AND t90.segment = psl.segment
    ),
    classified_with_pace AS (
        SELECT
            *,
            CASE
                WHEN status = 'Active' AND t_c_days IS NOT NULL THEN
                    CASE WHEN days_since_last_visit <= 2 * t_c_days THEN 'Active - On Pace'
                         ELSE 'Active - Below Pace' END
                WHEN status = 'Active' THEN 'Active - No Pace Standard for This Segment'
                ELSE status
            END AS status_detailed
        FROM classified
    )
    SELECT
        status,
        COUNT(*) AS total_patients,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_classifiable_patients
    FROM classified_with_pace
    GROUP BY status
    ORDER BY CASE status WHEN 'Active' THEN 1 WHEN 'Lapsing' THEN 2 ELSE 3 END
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# A1 Part B — status by segment, with T_c and pace detail
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_fr_status_by_segment() -> pd.DataFrame:
    """Columns: SEGMENT, SEGMENT_T_C_DAYS, T_C_BASED_ON_N_GAPS, STATUS_DETAILED,
    TOTAL_PATIENTS, PCT_WITHIN_SEGMENT."""
    sql = """
    WITH raw_diag AS (
        SELECT
            visit_id, patient_id, diagnosis_created_at AS visit_date,
            TRIM(bg.value::STRING) AS split_burden,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
        LATERAL FLATTEN(input => SPLIT(burden_group, '|')) bg
    ),
    segmented AS (
        SELECT DISTINCT
            visit_id, patient_id, visit_date,
            CASE
                WHEN clean_dx_text LIKE '%hernia%' THEN 'EXCLUDE: General Surgery'
                WHEN (clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%')
                     AND (clean_dx_text LIKE '%post %' OR clean_dx_text LIKE '%revision%')
                    THEN 'EXCLUDE: Fibroids-surgical'
                WHEN clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%'
                    THEN 'Fibroids-conservative'
                WHEN clean_dx_text LIKE '%spine%' OR clean_dx_text LIKE '%sciatica%' THEN
                    CASE
                        WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%dislocat%' OR clean_dx_text LIKE '%stenosis%'
                             OR clean_dx_text LIKE '%herniat%' OR clean_dx_text LIKE '%compression%' OR clean_dx_text LIKE '%tumour%'
                            THEN 'Spine-structural'
                        ELSE 'Spine-conservative'
                    END
                WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%osteoarthritis%' THEN 'Core Orthopedics: General'
                WHEN clean_dx_text LIKE '%pre-eclamp%' OR clean_dx_text LIKE '%preeclamp%' OR clean_dx_text LIKE '%eclamp%'
                     OR clean_dx_text LIKE '%hyperem%' OR clean_dx_text LIKE '%pprom%'
                    THEN 'High-Risk Pregnancy'
                WHEN clean_dx_text LIKE '%pregnan%' OR clean_dx_text LIKE '%anc%' OR clean_dx_text LIKE '%antenatal%'
                     OR clean_dx_text LIKE '%gravid%' OR split_burden = 'General: Obstetric'
                    THEN 'ANC / Routine Pregnancy'
                ELSE 'Other'
            END AS segment
        FROM raw_diag
    ),
    classifiable_all_time AS (
        SELECT * FROM segmented
        WHERE segment NOT IN ('EXCLUDE: General Surgery', 'EXCLUDE: Fibroids-surgical', 'Other')
    ),
    dataset_max_date AS (SELECT MAX(visit_date) AS max_date FROM classifiable_all_time),
    window_bounds AS (
        SELECT DATEADD('day', -365, max_date) AS window_start, max_date AS window_end
        FROM dataset_max_date
    ),
    patients_active_in_window AS (
        SELECT DISTINCT ca.patient_id, ca.segment
        FROM classifiable_all_time ca
        CROSS JOIN window_bounds wb
        WHERE ca.visit_date BETWEEN wb.window_start AND wb.window_end
    ),
    relevant_visits AS (
        SELECT ca.visit_id, ca.patient_id, ca.segment, ca.visit_date
        FROM classifiable_all_time ca
        JOIN patients_active_in_window paiw
            ON paiw.patient_id = ca.patient_id AND paiw.segment = ca.segment
    ),
    pregnancy_capped AS (
        SELECT
            rv.*,
            MIN(rv.visit_date) OVER (PARTITION BY rv.patient_id, rv.segment) AS first_preg_visit
        FROM relevant_visits rv
        WHERE rv.segment IN ('ANC / Routine Pregnancy', 'High-Risk Pregnancy')
    ),
    pregnancy_within_window AS (
        SELECT visit_id, patient_id, visit_date, segment
        FROM pregnancy_capped
        WHERE DATEDIFF('day', first_preg_visit, visit_date) <= 300
    ),
    non_pregnancy AS (
        SELECT visit_id, patient_id, visit_date, segment
        FROM relevant_visits
        WHERE segment NOT IN ('ANC / Routine Pregnancy', 'High-Risk Pregnancy')
    ),
    final_visits AS (
        SELECT * FROM non_pregnancy
        UNION ALL
        SELECT * FROM pregnancy_within_window
    ),
    patient_segment_last_visit AS (
        SELECT patient_id, segment, MAX(visit_date) AS last_visit_date
        FROM final_visits
        GROUP BY patient_id, segment
    ),
    consecutive_gaps AS (
        SELECT
            patient_id, segment, visit_date,
            LAG(visit_date) OVER (PARTITION BY patient_id, segment ORDER BY visit_date) AS prior_visit_date
        FROM final_visits
    ),
    segment_tc AS (
        SELECT
            segment,
            MEDIAN(DATEDIFF('day', prior_visit_date, visit_date)) AS t_c_days,
            COUNT(*) AS gap_observations
        FROM consecutive_gaps
        WHERE prior_visit_date IS NOT NULL
          AND segment IN ('ANC / Routine Pregnancy', 'High-Risk Pregnancy', 'Core Orthopedics: General',
                           'Spine-structural', 'Fibroids-conservative')
        GROUP BY segment
    ),
    trailing_90_visit_count AS (
        SELECT
            fv.patient_id, fv.segment,
            COUNT(*) AS visits_in_trailing_90
        FROM final_visits fv
        JOIN patient_segment_last_visit psl
            ON psl.patient_id = fv.patient_id AND psl.segment = fv.segment
        CROSS JOIN dataset_max_date dmd
        WHERE DATEDIFF('day', fv.visit_date, dmd.max_date) <= 90
        GROUP BY fv.patient_id, fv.segment
    ),
    classified AS (
        SELECT
            psl.patient_id, psl.segment, psl.last_visit_date,
            DATEDIFF('day', psl.last_visit_date, dmd.max_date) AS days_since_last_visit,
            st.t_c_days, st.gap_observations,
            COALESCE(t90.visits_in_trailing_90, 0) AS visits_in_trailing_90,
            CASE
                WHEN DATEDIFF('day', psl.last_visit_date, dmd.max_date) > 180 THEN 'LTFU'
                WHEN DATEDIFF('day', psl.last_visit_date, dmd.max_date) > 90 THEN 'Lapsing'
                ELSE 'Active'
            END AS status
        FROM patient_segment_last_visit psl
        CROSS JOIN dataset_max_date dmd
        LEFT JOIN segment_tc st ON st.segment = psl.segment
        LEFT JOIN trailing_90_visit_count t90 ON t90.patient_id = psl.patient_id AND t90.segment = psl.segment
    ),
    classified_with_pace AS (
        SELECT
            *,
            CASE
                WHEN status = 'Active' AND t_c_days IS NOT NULL THEN
                    CASE WHEN days_since_last_visit <= 2 * t_c_days THEN 'Active - On Pace'
                         ELSE 'Active - Below Pace' END
                WHEN status = 'Active' THEN 'Active - No Pace Standard for This Segment'
                ELSE status
            END AS status_detailed
        FROM classified
    )
    SELECT
        segment,
        ANY_VALUE(t_c_days) AS segment_t_c_days,
        ANY_VALUE(gap_observations) AS t_c_based_on_n_gaps,
        status_detailed,
        COUNT(*) AS total_patients,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY segment), 1) AS pct_within_segment
    FROM classified_with_pace
    GROUP BY segment, status_detailed
    ORDER BY segment,
        CASE
            WHEN status_detailed LIKE 'Active%' THEN 1
            WHEN status_detailed = 'Lapsing' THEN 2
            ELSE 3
        END
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# A2/A3 — retention rate + full status share, trailing 12 months
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_fr_retention_trend() -> pd.DataFrame:
    """Columns: AS_OF_MONTH, STATUS, TOTAL_PATIENTS, PCT_OF_CLASSIFIABLE_PATIENTS.
    Both A2 (read Active % as the headline retention line) and A3 (full
    3-way share over time) in one view — pivot or filter as needed."""
    sql = """
    WITH raw_diag AS (
        SELECT
            visit_id, patient_id, diagnosis_created_at AS visit_date,
            TRIM(bg.value::STRING) AS split_burden,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
        LATERAL FLATTEN(input => SPLIT(burden_group, '|')) bg
    ),
    segmented AS (
        SELECT DISTINCT
            visit_id, patient_id, visit_date,
            CASE
                WHEN clean_dx_text LIKE '%hernia%' THEN 'EXCLUDE: General Surgery'
                WHEN (clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%')
                     AND (clean_dx_text LIKE '%post %' OR clean_dx_text LIKE '%revision%')
                    THEN 'EXCLUDE: Fibroids-surgical'
                WHEN clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%'
                    THEN 'Fibroids-conservative'
                WHEN clean_dx_text LIKE '%spine%' OR clean_dx_text LIKE '%sciatica%' THEN
                    CASE
                        WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%dislocat%' OR clean_dx_text LIKE '%stenosis%'
                             OR clean_dx_text LIKE '%herniat%' OR clean_dx_text LIKE '%compression%' OR clean_dx_text LIKE '%tumour%'
                            THEN 'Spine-structural'
                        ELSE 'Spine-conservative'
                    END
                WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%osteoarthritis%' THEN 'Core Orthopedics: General'
                WHEN clean_dx_text LIKE '%pre-eclamp%' OR clean_dx_text LIKE '%preeclamp%' OR clean_dx_text LIKE '%eclamp%'
                     OR clean_dx_text LIKE '%hyperem%' OR clean_dx_text LIKE '%pprom%'
                    THEN 'High-Risk Pregnancy'
                WHEN clean_dx_text LIKE '%pregnan%' OR clean_dx_text LIKE '%anc%' OR clean_dx_text LIKE '%antenatal%'
                     OR clean_dx_text LIKE '%gravid%' OR split_burden = 'General: Obstetric'
                    THEN 'ANC / Routine Pregnancy'
                ELSE 'Other'
            END AS segment
        FROM raw_diag
    ),
    classifiable_all_time AS (
        SELECT * FROM segmented
        WHERE segment NOT IN ('EXCLUDE: General Surgery', 'EXCLUDE: Fibroids-surgical', 'Other')
    ),
    dataset_max_date AS (SELECT MAX(visit_date) AS max_date FROM classifiable_all_time),
    consecutive_gaps AS (
        SELECT
            patient_id, segment, visit_date,
            LAG(visit_date) OVER (PARTITION BY patient_id, segment ORDER BY visit_date) AS prior_visit_date
        FROM classifiable_all_time
    ),
    segment_tc AS (
        SELECT segment, MEDIAN(DATEDIFF('day', prior_visit_date, visit_date)) AS t_c_days
        FROM consecutive_gaps
        WHERE prior_visit_date IS NOT NULL
          AND segment IN ('ANC / Routine Pregnancy', 'High-Risk Pregnancy', 'Core Orthopedics: General',
                           'Spine-structural', 'Fibroids-conservative')
        GROUP BY segment
    ),
    trend_points AS (
        SELECT DATEADD('month', -seq4(), DATE_TRUNC('month', dmd.max_date)) AS as_of_month_start
        FROM dataset_max_date dmd, TABLE(GENERATOR(ROWCOUNT => 12))
    ),
    trend_bounds AS (
        SELECT
            as_of_month_start,
            LAST_DAY(as_of_month_start) AS as_of_date,
            DATEADD('day', -365, LAST_DAY(as_of_month_start)) AS window_start_for_this_point
        FROM trend_points
    ),
    patient_status_per_trend_point AS (
        SELECT
            tb.as_of_date,
            ca.patient_id,
            ca.segment,
            MAX(ca.visit_date) AS last_visit_as_of_point
        FROM trend_bounds tb
        JOIN classifiable_all_time ca
            ON ca.visit_date BETWEEN tb.window_start_for_this_point AND tb.as_of_date
        GROUP BY tb.as_of_date, ca.patient_id, ca.segment
    ),
    classified_trend AS (
        SELECT
            pstp.*,
            CASE
                WHEN DATEDIFF('day', pstp.last_visit_as_of_point, pstp.as_of_date) > 180 THEN 'LTFU'
                WHEN DATEDIFF('day', pstp.last_visit_as_of_point, pstp.as_of_date) > 90 THEN 'Lapsing'
                ELSE 'Active'
            END AS status
        FROM patient_status_per_trend_point pstp
    )
    SELECT
        TO_CHAR(as_of_date, 'YYYY-MM') AS as_of_month,
        status,
        COUNT(*) AS total_patients,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY as_of_date), 1) AS pct_of_classifiable_patients
    FROM classified_trend
    GROUP BY as_of_date, status
    ORDER BY as_of_date, CASE status WHEN 'Active' THEN 1 WHEN 'Lapsing' THEN 2 ELSE 3 END
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# A4 — channel explaining Active status
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_fr_active_channel() -> pd.DataFrame:
    """Columns: SEGMENT, TOTAL_ACTIVE_PATIENTS, EXPLAINED_BY_SCHEDULED,
    EXPLAINED_BY_TEXT_DETECTOR, EXPLAINED_BY_REPEAT_VISIT, NO_CLEAR_CHANNEL.
    Columns are not mutually exclusive — will not sum to total_active_patients."""
    sql = """
    WITH raw_diag AS (
        SELECT
            visit_id, patient_id, diagnosis_created_at AS visit_date,
            TRIM(bg.value::STRING) AS split_burden,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text,
            CASE WHEN clean_dx_text LIKE '%post %' OR clean_dx_text LIKE '%s/p %' OR clean_dx_text LIKE '%status post%'
                 OR clean_dx_text LIKE '%follow up%' OR clean_dx_text LIKE '%followup%' OR clean_dx_text LIKE '%f/u %'
                 OR clean_dx_text LIKE '%review%' OR clean_dx_text LIKE '%/52%' OR clean_dx_text LIKE '%/12%'
                THEN 1 ELSE 0 END AS is_text_detected_followup
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
        LATERAL FLATTEN(input => SPLIT(burden_group, '|')) bg
    ),
    segmented AS (
        SELECT DISTINCT
            visit_id, patient_id, visit_date, is_text_detected_followup,
            CASE
                WHEN clean_dx_text LIKE '%hernia%' THEN 'EXCLUDE: General Surgery'
                WHEN (clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%')
                     AND (clean_dx_text LIKE '%post %' OR clean_dx_text LIKE '%revision%')
                    THEN 'EXCLUDE: Fibroids-surgical'
                WHEN clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%'
                    THEN 'Fibroids-conservative'
                WHEN clean_dx_text LIKE '%spine%' OR clean_dx_text LIKE '%sciatica%' THEN
                    CASE
                        WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%dislocat%' OR clean_dx_text LIKE '%stenosis%'
                             OR clean_dx_text LIKE '%herniat%' OR clean_dx_text LIKE '%compression%' OR clean_dx_text LIKE '%tumour%'
                            THEN 'Spine-structural'
                        ELSE 'Spine-conservative'
                    END
                WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%osteoarthritis%' THEN 'Core Orthopedics: General'
                WHEN clean_dx_text LIKE '%pre-eclamp%' OR clean_dx_text LIKE '%preeclamp%' OR clean_dx_text LIKE '%eclamp%'
                     OR clean_dx_text LIKE '%hyperem%' OR clean_dx_text LIKE '%pprom%'
                    THEN 'High-Risk Pregnancy'
                WHEN clean_dx_text LIKE '%pregnan%' OR clean_dx_text LIKE '%anc%' OR clean_dx_text LIKE '%antenatal%'
                     OR clean_dx_text LIKE '%gravid%' OR split_burden = 'General: Obstetric'
                    THEN 'ANC / Routine Pregnancy'
                ELSE 'Other'
            END AS segment
        FROM raw_diag
    ),
    classifiable_all_time AS (
        SELECT * FROM segmented
        WHERE segment NOT IN ('EXCLUDE: General Surgery', 'EXCLUDE: Fibroids-surgical', 'Other')
    ),
    dataset_max_date AS (SELECT MAX(visit_date) AS max_date FROM classifiable_all_time),
    window_bounds AS (SELECT DATEADD('day', -365, max_date) AS window_start, max_date AS window_end FROM dataset_max_date),
    patients_active_in_window AS (
        SELECT DISTINCT ca.patient_id, ca.segment
        FROM classifiable_all_time ca CROSS JOIN window_bounds wb
        WHERE ca.visit_date BETWEEN wb.window_start AND wb.window_end
    ),
    relevant_visits AS (
        SELECT ca.* FROM classifiable_all_time ca
        JOIN patients_active_in_window paiw ON paiw.patient_id = ca.patient_id AND paiw.segment = ca.segment
    ),
    patient_segment_last_visit AS (
        SELECT patient_id, segment, MAX(visit_date) AS last_visit_date
        FROM relevant_visits GROUP BY patient_id, segment
    ),
    active_patients AS (
        SELECT psl.patient_id, psl.segment, psl.last_visit_date
        FROM patient_segment_last_visit psl CROSS JOIN dataset_max_date dmd
        WHERE DATEDIFF('day', psl.last_visit_date, dmd.max_date) <= 90
    ),
    text_signal AS (
        SELECT DISTINCT ap.patient_id, ap.segment
        FROM active_patients ap
        JOIN relevant_visits rv ON rv.patient_id = ap.patient_id AND rv.segment = ap.segment
        WHERE rv.is_text_detected_followup = 1
    ),
    repeat_signal AS (
        SELECT patient_id, segment FROM relevant_visits
        GROUP BY patient_id, segment HAVING COUNT(*) >= 2
    ),
    oie_flat AS (
        SELECT o."id" AS oie_id, o."ref" AS soi_id, o."item" AS sale_item_id,
            f.VALUE:"name"::STRING AS field_name, v.VALUE::STRING AS field_value
        FROM HOSPITALS.ORTHOPEDIC_CLEAN."ORDERITEMENTRIES" o,
        LATERAL FLATTEN(INPUT => TRY_PARSE_JSON(o."fields"), OUTER => TRUE) f,
        LATERAL FLATTEN(INPUT => f.VALUE:"value", OUTER => TRUE) v
        WHERE f.VALUE:"name"::STRING IS NOT NULL
    ),
    doctor_raw AS (
        SELECT oie.oie_id, oie.soi_id, oie.field_name, oie.field_value
        FROM oie_flat oie JOIN HOSPITALS.ORTHOPEDIC_CLEAN."SALEITEMS" si ON si."id" = oie.sale_item_id
        WHERE si."category" = 'Doctor'
    ),
    doctor_pivot AS (
        SELECT oie_id, soi_id,
            MAX(CASE WHEN field_name = 'Schedule Follow Up' THEN TRY_TO_TIMESTAMP(field_value) END) AS scheduled_follow_up_date
        FROM doctor_raw GROUP BY oie_id, soi_id
    ),
    soi AS (SELECT "id" AS soi_id, "patient" AS patient_id FROM HOSPITALS.ORTHOPEDIC_CLEAN."SINGLEORDERITEMS"),
    orders_flat AS (
        SELECT oo.id AS order_id, f.VALUE::STRING AS item_id
        FROM HOSPITALS.ORTHOPEDIC_CLEAN."ORTHO_ORDERS" oo,
        LATERAL FLATTEN(INPUT => TRY_PARSE_JSON(oo.items)) f
        QUALIFY ROW_NUMBER() OVER (PARTITION BY f.VALUE::STRING ORDER BY oo.registered_at DESC) = 1
    ),
    raw_schedule_visits AS (
        SELECT of_.order_id AS visit_id FROM doctor_pivot dp
        JOIN soi s ON s.soi_id = dp.soi_id JOIN orders_flat of_ ON of_.item_id = dp.soi_id
        WHERE dp.scheduled_follow_up_date IS NOT NULL
    ),
    scheduled_signal_raw AS (
        SELECT DISTINCT v.patient_id AS clean_patient_id
        FROM raw_schedule_visits rsv JOIN HOSPITALS.STAGING.STG_VISITS v ON v.visit_id = rsv.visit_id
    ),
    scheduled_signal AS (
        SELECT DISTINCT ap.patient_id, ap.segment
        FROM active_patients ap
        JOIN scheduled_signal_raw ssr ON ssr.clean_patient_id = ap.patient_id
    )
    SELECT
        ap.segment,
        COUNT(*) AS total_active_patients,
        SUM(CASE WHEN ss.patient_id IS NOT NULL THEN 1 ELSE 0 END) AS explained_by_scheduled,
        SUM(CASE WHEN ts.patient_id IS NOT NULL THEN 1 ELSE 0 END) AS explained_by_text_detector,
        SUM(CASE WHEN rs.patient_id IS NOT NULL THEN 1 ELSE 0 END) AS explained_by_repeat_visit,
        SUM(CASE WHEN ss.patient_id IS NULL AND ts.patient_id IS NULL AND rs.patient_id IS NULL THEN 1 ELSE 0 END) AS no_clear_channel
    FROM active_patients ap
    LEFT JOIN scheduled_signal ss ON ss.patient_id = ap.patient_id AND ss.segment = ap.segment
    LEFT JOIN text_signal ts ON ts.patient_id = ap.patient_id AND ts.segment = ap.segment
    LEFT JOIN repeat_signal rs ON rs.patient_id = ap.patient_id AND rs.segment = ap.segment
    GROUP BY ap.segment
    ORDER BY total_active_patients DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# A4 diagnostic — is the scheduled-pipeline itself broken?
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_fr_scheduled_pipeline_diagnostic() -> pd.DataFrame:
    """Columns: TOTAL_SCHEDULED_SIGNAL_PATIENTS, EARLIEST_SCHEDULED_RECORD,
    LATEST_SCHEDULED_RECORD, SCHEDULED_RECORDS_WITHIN_LAST_365_DAYS."""
    sql = """
    WITH oie_flat AS (
        SELECT o."id" AS oie_id, o."ref" AS soi_id, o."item" AS sale_item_id,
            f.VALUE:"name"::STRING AS field_name, v.VALUE::STRING AS field_value
        FROM HOSPITALS.ORTHOPEDIC_CLEAN."ORDERITEMENTRIES" o,
        LATERAL FLATTEN(INPUT => TRY_PARSE_JSON(o."fields"), OUTER => TRUE) f,
        LATERAL FLATTEN(INPUT => f.VALUE:"value", OUTER => TRUE) v
        WHERE f.VALUE:"name"::STRING IS NOT NULL
    ),
    doctor_raw AS (
        SELECT oie.oie_id, oie.soi_id, oie.field_name, oie.field_value
        FROM oie_flat oie JOIN HOSPITALS.ORTHOPEDIC_CLEAN."SALEITEMS" si ON si."id" = oie.sale_item_id
        WHERE si."category" = 'Doctor'
    ),
    doctor_pivot AS (
        SELECT oie_id, soi_id,
            MAX(CASE WHEN field_name = 'Schedule Follow Up' THEN TRY_TO_TIMESTAMP(field_value) END) AS scheduled_follow_up_date
        FROM doctor_raw GROUP BY oie_id, soi_id
    ),
    soi AS (SELECT "id" AS soi_id, "patient" AS patient_id FROM HOSPITALS.ORTHOPEDIC_CLEAN."SINGLEORDERITEMS"),
    orders_flat AS (
        SELECT oo.id AS order_id, f.VALUE::STRING AS item_id, oo.registered_at
        FROM HOSPITALS.ORTHOPEDIC_CLEAN."ORTHO_ORDERS" oo,
        LATERAL FLATTEN(INPUT => TRY_PARSE_JSON(oo.items)) f
        QUALIFY ROW_NUMBER() OVER (PARTITION BY f.VALUE::STRING ORDER BY oo.registered_at DESC) = 1
    ),
    raw_schedule_visits AS (
        SELECT of_.order_id AS visit_id, of_.registered_at
        FROM doctor_pivot dp
        JOIN soi s ON s.soi_id = dp.soi_id JOIN orders_flat of_ ON of_.item_id = dp.soi_id
        WHERE dp.scheduled_follow_up_date IS NOT NULL
    ),
    scheduled_signal_raw AS (
        SELECT DISTINCT v.patient_id AS clean_patient_id, rsv.registered_at
        FROM raw_schedule_visits rsv JOIN HOSPITALS.STAGING.STG_VISITS v ON v.visit_id = rsv.visit_id
    )
    SELECT
        COUNT(*) AS total_scheduled_signal_patients,
        MIN(registered_at) AS earliest_scheduled_record,
        MAX(registered_at) AS latest_scheduled_record,
        SUM(CASE WHEN registered_at >= DATEADD('day', -365, (SELECT MAX(diagnosis_created_at) FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED))
                 THEN 1 ELSE 0 END) AS scheduled_records_within_last_365_days
    FROM scheduled_signal_raw
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# A5 — LTFU demographics
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_fr_ltfu_demographics() -> pd.DataFrame:
    """Columns: AGE_GROUP, GENDER, TOTAL_LTFU_PATIENTS. HAVING >= 5."""
    sql = """
    WITH raw_diag AS (
        SELECT
            visit_id, patient_id, diagnosis_created_at AS visit_date,
            TRIM(bg.value::STRING) AS split_burden,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
        LATERAL FLATTEN(input => SPLIT(burden_group, '|')) bg
    ),
    segmented AS (
        SELECT DISTINCT
            visit_id, patient_id, visit_date,
            CASE
                WHEN clean_dx_text LIKE '%hernia%' THEN 'EXCLUDE: General Surgery'
                WHEN (clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%')
                     AND (clean_dx_text LIKE '%post %' OR clean_dx_text LIKE '%revision%')
                    THEN 'EXCLUDE: Fibroids-surgical'
                WHEN clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%'
                    THEN 'Fibroids-conservative'
                WHEN clean_dx_text LIKE '%spine%' OR clean_dx_text LIKE '%sciatica%' THEN
                    CASE
                        WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%dislocat%' OR clean_dx_text LIKE '%stenosis%'
                             OR clean_dx_text LIKE '%herniat%' OR clean_dx_text LIKE '%compression%' OR clean_dx_text LIKE '%tumour%'
                            THEN 'Spine-structural'
                        ELSE 'Spine-conservative'
                    END
                WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%osteoarthritis%' THEN 'Core Orthopedics: General'
                WHEN clean_dx_text LIKE '%pre-eclamp%' OR clean_dx_text LIKE '%preeclamp%' OR clean_dx_text LIKE '%eclamp%'
                     OR clean_dx_text LIKE '%hyperem%' OR clean_dx_text LIKE '%pprom%'
                    THEN 'High-Risk Pregnancy'
                WHEN clean_dx_text LIKE '%pregnan%' OR clean_dx_text LIKE '%anc%' OR clean_dx_text LIKE '%antenatal%'
                     OR clean_dx_text LIKE '%gravid%' OR split_burden = 'General: Obstetric'
                    THEN 'ANC / Routine Pregnancy'
                ELSE 'Other'
            END AS segment
        FROM raw_diag
    ),
    classifiable_all_time AS (
        SELECT * FROM segmented
        WHERE segment NOT IN ('EXCLUDE: General Surgery', 'EXCLUDE: Fibroids-surgical', 'Other')
    ),
    dataset_max_date AS (SELECT MAX(visit_date) AS max_date FROM classifiable_all_time),
    window_bounds AS (SELECT DATEADD('day', -365, max_date) AS window_start, max_date AS window_end FROM dataset_max_date),
    patients_active_in_window AS (
        SELECT DISTINCT ca.patient_id, ca.segment
        FROM classifiable_all_time ca CROSS JOIN window_bounds wb
        WHERE ca.visit_date BETWEEN wb.window_start AND wb.window_end
    ),
    relevant_visits AS (
        SELECT ca.* FROM classifiable_all_time ca
        JOIN patients_active_in_window paiw ON paiw.patient_id = ca.patient_id AND paiw.segment = ca.segment
    ),
    patient_segment_last_visit AS (
        SELECT patient_id, segment, MAX(visit_date) AS last_visit_date
        FROM relevant_visits GROUP BY patient_id, segment
    ),
    ltfu_patients AS (
        SELECT psl.patient_id, psl.segment, psl.last_visit_date
        FROM patient_segment_last_visit psl CROSS JOIN dataset_max_date dmd
        WHERE DATEDIFF('day', psl.last_visit_date, dmd.max_date) > 180
    ),
    ltfu_visit_id AS (
        SELECT lp.patient_id, lp.segment, rv.visit_id
        FROM ltfu_patients lp
        JOIN relevant_visits rv ON rv.patient_id = lp.patient_id AND rv.segment = lp.segment AND rv.visit_date = lp.last_visit_date
        QUALIFY ROW_NUMBER() OVER (PARTITION BY lp.patient_id, lp.segment ORDER BY rv.visit_id) = 1
    )
    SELECT
        age_group,
        CASE WHEN LOWER(v.gender) = 'f' THEN 'female' WHEN LOWER(v.gender) = 'm' THEN 'male' ELSE LOWER(v.gender) END AS gender,
        COUNT(*) AS total_ltfu_patients
    FROM ltfu_visit_id lvi
    LEFT JOIN HOSPITALS.STAGING.STG_VISITS v ON v.visit_id = lvi.visit_id
    GROUP BY age_group, gender
    HAVING COUNT(*) >= 5
    ORDER BY total_ltfu_patients DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# A6 — LTFU share by condition (segment) x age_group x gender
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_fr_ltfu_share_by_segment_age_gender() -> pd.DataFrame:
    """LTFU broken down by age and gender, expressed as a share of each
    segment's TOTAL population (not the age/gender cell's own population) —
    each cell answers 'what fraction of this whole segment is accounted for
    by LTFU patients in this specific age/gender slice,' not 'what % of
    just this subgroup went LTFU.' Missing age/gender is kept as its own
    'Unknown' bucket rather than dropped, so the heatmap can show it
    explicitly.

    Columns: SEGMENT, AGE_GROUP, GENDER ('male'/'female'/'unknown'),
    TOTAL_PATIENTS (population of that specific age/gender cell, for
    reference/hover only), SEGMENT_TOTAL_PATIENTS (the actual % denominator),
    TOTAL_LTFU_PATIENTS, LTFU_SHARE_PCT.
    """
    sql = """
    WITH raw_diag AS (
        SELECT
            visit_id, patient_id, diagnosis_created_at AS visit_date,
            TRIM(bg.value::STRING) AS split_burden,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
        LATERAL FLATTEN(input => SPLIT(burden_group, '|')) bg
    ),
    segmented AS (
        SELECT DISTINCT
            visit_id, patient_id, visit_date,
            CASE
                WHEN clean_dx_text LIKE '%hernia%' THEN 'EXCLUDE: General Surgery'
                WHEN (clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%')
                     AND (clean_dx_text LIKE '%post %' OR clean_dx_text LIKE '%revision%')
                    THEN 'EXCLUDE: Fibroids-surgical'
                WHEN clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%'
                    THEN 'Fibroids-conservative'
                WHEN clean_dx_text LIKE '%spine%' OR clean_dx_text LIKE '%sciatica%' THEN
                    CASE
                        WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%dislocat%' OR clean_dx_text LIKE '%stenosis%'
                             OR clean_dx_text LIKE '%herniat%' OR clean_dx_text LIKE '%compression%' OR clean_dx_text LIKE '%tumour%'
                            THEN 'Spine-structural'
                        ELSE 'Spine-conservative'
                    END
                WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%osteoarthritis%' THEN 'Core Orthopedics: General'
                WHEN clean_dx_text LIKE '%pre-eclamp%' OR clean_dx_text LIKE '%preeclamp%' OR clean_dx_text LIKE '%eclamp%'
                     OR clean_dx_text LIKE '%hyperem%' OR clean_dx_text LIKE '%pprom%'
                    THEN 'High-Risk Pregnancy'
                WHEN clean_dx_text LIKE '%pregnan%' OR clean_dx_text LIKE '%anc%' OR clean_dx_text LIKE '%antenatal%'
                     OR clean_dx_text LIKE '%gravid%' OR split_burden = 'General: Obstetric'
                    THEN 'ANC / Routine Pregnancy'
                ELSE 'Other'
            END AS segment
        FROM raw_diag
    ),
    classifiable_all_time AS (
        SELECT * FROM segmented
        WHERE segment NOT IN ('EXCLUDE: General Surgery', 'EXCLUDE: Fibroids-surgical', 'Other')
    ),
    dataset_max_date AS (SELECT MAX(visit_date) AS max_date FROM classifiable_all_time),
    window_bounds AS (SELECT DATEADD('day', -365, max_date) AS window_start, max_date AS window_end FROM dataset_max_date),
    patients_active_in_window AS (
        SELECT DISTINCT ca.patient_id, ca.segment
        FROM classifiable_all_time ca CROSS JOIN window_bounds wb
        WHERE ca.visit_date BETWEEN wb.window_start AND wb.window_end
    ),
    relevant_visits AS (
        SELECT ca.* FROM classifiable_all_time ca
        JOIN patients_active_in_window paiw ON paiw.patient_id = ca.patient_id AND paiw.segment = ca.segment
    ),
    -- Pregnancy visits are capped to the 300 days following a patient's
    -- first visit in that pregnancy episode — same rule get_fr_status_overall()
    -- uses to classify current LTFU status. Without this, a patient's "last
    -- visit" (and therefore their LTFU status) can disagree between this
    -- heatmap and the hospital-wide LTFU total shown elsewhere in the tab.
    pregnancy_capped AS (
        SELECT rv.*, MIN(rv.visit_date) OVER (PARTITION BY rv.patient_id, rv.segment) AS first_preg_visit
        FROM relevant_visits rv
        WHERE rv.segment IN ('ANC / Routine Pregnancy', 'High-Risk Pregnancy')
    ),
    pregnancy_within_window AS (
        SELECT visit_id, patient_id, visit_date, segment
        FROM pregnancy_capped
        WHERE DATEDIFF('day', first_preg_visit, visit_date) <= 300
    ),
    non_pregnancy AS (
        SELECT visit_id, patient_id, visit_date, segment
        FROM relevant_visits
        WHERE segment NOT IN ('ANC / Routine Pregnancy', 'High-Risk Pregnancy')
    ),
    final_visits AS (
        SELECT * FROM non_pregnancy
        UNION ALL
        SELECT * FROM pregnancy_within_window
    ),
    -- Every patient in the segment's population, not just LTFU ones — this
    -- is the denominator. Demographics taken from each patient's most
    -- recent visit within the segment, same reference point used for the
    -- LTFU determination itself.
    population_last_visit AS (
        SELECT patient_id, segment, MAX(visit_date) AS last_visit_date
        FROM final_visits GROUP BY patient_id, segment
    ),
    population_visit_id AS (
        SELECT pl.patient_id, pl.segment, pl.last_visit_date, fv.visit_id
        FROM population_last_visit pl
        JOIN final_visits fv ON fv.patient_id = pl.patient_id AND fv.segment = pl.segment AND fv.visit_date = pl.last_visit_date
        QUALIFY ROW_NUMBER() OVER (PARTITION BY pl.patient_id, pl.segment ORDER BY fv.visit_id) = 1
    ),
    population_demo_raw AS (
        SELECT
            pvi.segment,
            COALESCE(v.age_group, 'Unknown') AS age_group,
            CASE WHEN LOWER(v.gender) = 'f' THEN 'female'
                 WHEN LOWER(v.gender) = 'm' THEN 'male'
                 ELSE 'unknown' END AS gender,
            CASE WHEN DATEDIFF('day', pvi.last_visit_date, dmd.max_date) > 180 THEN 1 ELSE 0 END AS is_ltfu
        FROM population_visit_id pvi
        CROSS JOIN dataset_max_date dmd
        LEFT JOIN HOSPITALS.STAGING.STG_VISITS v ON v.visit_id = pvi.visit_id
    ),
    -- ANC / Routine Pregnancy is female-only by definition — a recorded
    -- male gender is a data-entry error, and an unknown gender defaults to
    -- female since the segment guarantees it. Age is only trustworthy
    -- within the plausible reproductive-age brackets; anything outside
    -- that (or already unknown) is shown as Unknown rather than a specific,
    -- almost certainly wrong, bucket.
    population_demo AS (
        SELECT
            segment,
            CASE
                WHEN segment = 'ANC / Routine Pregnancy'
                     AND age_group NOT IN ('Adolescent (13-17)', 'Youth (18-24)', 'Young Adult (25-34)',
                                            'Adult (35-44)', 'Middle Age (45-54)')
                    THEN 'Unknown'
                ELSE age_group
            END AS age_group,
            CASE
                WHEN segment = 'ANC / Routine Pregnancy' AND gender != 'female'
                    THEN 'female'
                ELSE gender
            END AS gender,
            is_ltfu
        FROM population_demo_raw
    ),
    -- Denominator is the segment's total population, not the age/gender
    -- cell's own population — each cell shows how much of the WHOLE
    -- segment that specific demographic-and-LTFU slice represents.
    segment_totals AS (
        SELECT segment, COUNT(*) AS segment_total_patients
        FROM population_demo GROUP BY segment
    )
    SELECT
        pd.segment,
        pd.age_group,
        pd.gender,
        COUNT(*) AS total_patients,
        st.segment_total_patients,
        SUM(pd.is_ltfu) AS total_ltfu_patients,
        ROUND(100.0 * SUM(pd.is_ltfu) / st.segment_total_patients, 1) AS ltfu_share_pct
    FROM population_demo pd
    JOIN segment_totals st ON st.segment = pd.segment
    GROUP BY pd.segment, pd.age_group, pd.gender, st.segment_total_patients
    ORDER BY pd.segment, ltfu_share_pct DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# A7 — LTFU by visit number
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_fr_ltfu_by_visit_number() -> pd.DataFrame:
    """Columns: LTFU_AT_VISIT_NUMBER ('1'..'6','7+'), TOTAL_PATIENTS."""
    sql = """
    WITH raw_diag AS (
        SELECT
            visit_id, patient_id, diagnosis_created_at AS visit_date,
            TRIM(bg.value::STRING) AS split_burden,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
        LATERAL FLATTEN(input => SPLIT(burden_group, '|')) bg
    ),
    segmented AS (
        SELECT DISTINCT
            visit_id, patient_id, visit_date,
            CASE
                WHEN clean_dx_text LIKE '%hernia%' THEN 'EXCLUDE: General Surgery'
                WHEN (clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%')
                     AND (clean_dx_text LIKE '%post %' OR clean_dx_text LIKE '%revision%')
                    THEN 'EXCLUDE: Fibroids-surgical'
                WHEN clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%'
                    THEN 'Fibroids-conservative'
                WHEN clean_dx_text LIKE '%spine%' OR clean_dx_text LIKE '%sciatica%' THEN
                    CASE
                        WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%dislocat%' OR clean_dx_text LIKE '%stenosis%'
                             OR clean_dx_text LIKE '%herniat%' OR clean_dx_text LIKE '%compression%' OR clean_dx_text LIKE '%tumour%'
                            THEN 'Spine-structural'
                        ELSE 'Spine-conservative'
                    END
                WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%osteoarthritis%' THEN 'Core Orthopedics: General'
                WHEN clean_dx_text LIKE '%pre-eclamp%' OR clean_dx_text LIKE '%preeclamp%' OR clean_dx_text LIKE '%eclamp%'
                     OR clean_dx_text LIKE '%hyperem%' OR clean_dx_text LIKE '%pprom%'
                    THEN 'High-Risk Pregnancy'
                WHEN clean_dx_text LIKE '%pregnan%' OR clean_dx_text LIKE '%anc%' OR clean_dx_text LIKE '%antenatal%'
                     OR clean_dx_text LIKE '%gravid%' OR split_burden = 'General: Obstetric'
                    THEN 'ANC / Routine Pregnancy'
                ELSE 'Other'
            END AS segment
        FROM raw_diag
    ),
    classifiable_all_time AS (
        SELECT * FROM segmented
        WHERE segment NOT IN ('EXCLUDE: General Surgery', 'EXCLUDE: Fibroids-surgical', 'Other')
    ),
    dataset_max_date AS (SELECT MAX(visit_date) AS max_date FROM classifiable_all_time),
    window_bounds AS (SELECT DATEADD('day', -365, max_date) AS window_start, max_date AS window_end FROM dataset_max_date),
    patients_active_in_window AS (
        SELECT DISTINCT ca.patient_id, ca.segment
        FROM classifiable_all_time ca CROSS JOIN window_bounds wb
        WHERE ca.visit_date BETWEEN wb.window_start AND wb.window_end
    ),
    relevant_visits AS (
        SELECT ca.* FROM classifiable_all_time ca
        JOIN patients_active_in_window paiw ON paiw.patient_id = ca.patient_id AND paiw.segment = ca.segment
    ),
    visit_sequence AS (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY patient_id, segment ORDER BY visit_date) AS visit_number
        FROM relevant_visits
    ),
    patient_segment_last_visit AS (
        SELECT patient_id, segment, MAX(visit_date) AS last_visit_date
        FROM relevant_visits GROUP BY patient_id, segment
    ),
    ltfu_patients AS (
        SELECT psl.patient_id, psl.segment, psl.last_visit_date
        FROM patient_segment_last_visit psl CROSS JOIN dataset_max_date dmd
        WHERE DATEDIFF('day', psl.last_visit_date, dmd.max_date) > 180
    ),
    ltfu_with_visit_number AS (
        SELECT lp.patient_id, lp.segment, vs.visit_number
        FROM ltfu_patients lp
        JOIN visit_sequence vs ON vs.patient_id = lp.patient_id AND vs.segment = lp.segment AND vs.visit_date = lp.last_visit_date
    )
    SELECT
        CASE WHEN visit_number >= 7 THEN '7+' ELSE visit_number::STRING END AS ltfu_at_visit_number,
        COUNT(*) AS total_patients
    FROM ltfu_with_visit_number
    GROUP BY ltfu_at_visit_number
    ORDER BY
        CASE ltfu_at_visit_number WHEN '7+' THEN 8 ELSE TRY_TO_NUMBER(ltfu_at_visit_number) END
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# A7b — LTFU by segment AND visit number (cross-tab)
# ---------------------------------------------------------------------------
# Same population and same visit_number logic as get_fr_ltfu_by_visit_number()
# above — that function already computes (patient_id, segment, visit_number)
# per LTFU patient and then collapses segment away in the final aggregation.
# This version keeps both dimensions, answering "of Spine-conservative's LTFU
# patients, how many left after visit 1?" instead of only the hospital-wide
# answer to that question.

@st.cache_data(ttl=3600)
def get_fr_ltfu_by_segment_and_visit_number() -> pd.DataFrame:
    """Columns: SEGMENT, LTFU_AT_VISIT_NUMBER ('1'..'6','7+'), TOTAL_PATIENTS,
    PCT_WITHIN_SEGMENT (each segment's visit-number bars sum to 100%)."""
    sql = """
    WITH raw_diag AS (
        SELECT
            visit_id, patient_id, diagnosis_created_at AS visit_date,
            TRIM(bg.value::STRING) AS split_burden,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
        LATERAL FLATTEN(input => SPLIT(burden_group, '|')) bg
    ),
    segmented AS (
        SELECT DISTINCT
            visit_id, patient_id, visit_date,
            CASE
                WHEN clean_dx_text LIKE '%hernia%' THEN 'EXCLUDE: General Surgery'
                WHEN (clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%')
                     AND (clean_dx_text LIKE '%post %' OR clean_dx_text LIKE '%revision%')
                    THEN 'EXCLUDE: Fibroids-surgical'
                WHEN clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%'
                    THEN 'Fibroids-conservative'
                WHEN clean_dx_text LIKE '%spine%' OR clean_dx_text LIKE '%sciatica%' THEN
                    CASE
                        WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%dislocat%' OR clean_dx_text LIKE '%stenosis%'
                             OR clean_dx_text LIKE '%herniat%' OR clean_dx_text LIKE '%compression%' OR clean_dx_text LIKE '%tumour%'
                            THEN 'Spine-structural'
                        ELSE 'Spine-conservative'
                    END
                WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%osteoarthritis%' THEN 'Core Orthopedics: General'
                WHEN clean_dx_text LIKE '%pre-eclamp%' OR clean_dx_text LIKE '%preeclamp%' OR clean_dx_text LIKE '%eclamp%'
                     OR clean_dx_text LIKE '%hyperem%' OR clean_dx_text LIKE '%pprom%'
                    THEN 'High-Risk Pregnancy'
                WHEN clean_dx_text LIKE '%pregnan%' OR clean_dx_text LIKE '%anc%' OR clean_dx_text LIKE '%antenatal%'
                     OR clean_dx_text LIKE '%gravid%' OR split_burden = 'General: Obstetric'
                    THEN 'ANC / Routine Pregnancy'
                ELSE 'Other'
            END AS segment
        FROM raw_diag
    ),
    classifiable_all_time AS (
        SELECT * FROM segmented
        WHERE segment NOT IN ('EXCLUDE: General Surgery', 'EXCLUDE: Fibroids-surgical', 'Other')
    ),
    dataset_max_date AS (SELECT MAX(visit_date) AS max_date FROM classifiable_all_time),
    window_bounds AS (SELECT DATEADD('day', -365, max_date) AS window_start, max_date AS window_end FROM dataset_max_date),
    patients_active_in_window AS (
        SELECT DISTINCT ca.patient_id, ca.segment
        FROM classifiable_all_time ca CROSS JOIN window_bounds wb
        WHERE ca.visit_date BETWEEN wb.window_start AND wb.window_end
    ),
    relevant_visits AS (
        SELECT ca.* FROM classifiable_all_time ca
        JOIN patients_active_in_window paiw ON paiw.patient_id = ca.patient_id AND paiw.segment = ca.segment
    ),
    visit_sequence AS (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY patient_id, segment ORDER BY visit_date) AS visit_number
        FROM relevant_visits
    ),
    patient_segment_last_visit AS (
        SELECT patient_id, segment, MAX(visit_date) AS last_visit_date
        FROM relevant_visits GROUP BY patient_id, segment
    ),
    ltfu_patients AS (
        SELECT psl.patient_id, psl.segment, psl.last_visit_date
        FROM patient_segment_last_visit psl CROSS JOIN dataset_max_date dmd
        WHERE DATEDIFF('day', psl.last_visit_date, dmd.max_date) > 180
    ),
    ltfu_with_visit_number AS (
        SELECT lp.patient_id, lp.segment, vs.visit_number
        FROM ltfu_patients lp
        JOIN visit_sequence vs ON vs.patient_id = lp.patient_id AND vs.segment = lp.segment AND vs.visit_date = lp.last_visit_date
    ),
    bucketed AS (
        SELECT
            segment,
            CASE WHEN visit_number >= 7 THEN '7+' ELSE visit_number::STRING END AS ltfu_at_visit_number
        FROM ltfu_with_visit_number
    )
    SELECT
        segment                                                                       AS SEGMENT,
        ltfu_at_visit_number                                                          AS LTFU_AT_VISIT_NUMBER,
        COUNT(*)                                                                      AS TOTAL_PATIENTS,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY segment), 1)        AS PCT_WITHIN_SEGMENT
    FROM bucketed
    GROUP BY segment, ltfu_at_visit_number
    ORDER BY segment,
        CASE ltfu_at_visit_number WHEN '7+' THEN 8 ELSE TRY_TO_NUMBER(ltfu_at_visit_number) END
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# A8 — last care pathway for LTFU patients
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_fr_ltfu_last_pathway() -> pd.DataFrame:
    """Columns: SEGMENT, TOTAL_LTFU_PATIENTS, HAD_THEATRE_PROCEDURE,
    HAD_MEDICATION_PICKUP, HAD_INVESTIGATION_OR_LAB, HAD_IMAGING,
    CONSULTATION_ONLY_NO_OTHER_RECORD. Columns are not mutually exclusive."""
    sql = """
    WITH raw_diag AS (
        SELECT
            visit_id, patient_id, diagnosis_created_at AS visit_date, source_system,
            TRIM(bg.value::STRING) AS split_burden,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
        LATERAL FLATTEN(input => SPLIT(burden_group, '|')) bg
    ),
    segmented AS (
        SELECT DISTINCT
            visit_id, patient_id, visit_date, source_system,
            CASE
                WHEN clean_dx_text LIKE '%hernia%' THEN 'EXCLUDE: General Surgery'
                WHEN (clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%')
                     AND (clean_dx_text LIKE '%post %' OR clean_dx_text LIKE '%revision%')
                    THEN 'EXCLUDE: Fibroids-surgical'
                WHEN clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%'
                    THEN 'Fibroids-conservative'
                WHEN clean_dx_text LIKE '%spine%' OR clean_dx_text LIKE '%sciatica%' THEN
                    CASE
                        WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%dislocat%' OR clean_dx_text LIKE '%stenosis%'
                             OR clean_dx_text LIKE '%herniat%' OR clean_dx_text LIKE '%compression%' OR clean_dx_text LIKE '%tumour%'
                            THEN 'Spine-structural'
                        ELSE 'Spine-conservative'
                    END
                WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%osteoarthritis%' THEN 'Core Orthopedics: General'
                WHEN clean_dx_text LIKE '%pre-eclamp%' OR clean_dx_text LIKE '%preeclamp%' OR clean_dx_text LIKE '%eclamp%'
                     OR clean_dx_text LIKE '%hyperem%' OR clean_dx_text LIKE '%pprom%'
                    THEN 'High-Risk Pregnancy'
                WHEN clean_dx_text LIKE '%pregnan%' OR clean_dx_text LIKE '%anc%' OR clean_dx_text LIKE '%antenatal%'
                     OR clean_dx_text LIKE '%gravid%' OR split_burden = 'General: Obstetric'
                    THEN 'ANC / Routine Pregnancy'
                ELSE 'Other'
            END AS segment
        FROM raw_diag
    ),
    classifiable_all_time AS (
        SELECT * FROM segmented
        WHERE segment NOT IN ('EXCLUDE: General Surgery', 'EXCLUDE: Fibroids-surgical', 'Other')
    ),
    dataset_max_date AS (SELECT MAX(visit_date) AS max_date FROM classifiable_all_time),
    window_bounds AS (SELECT DATEADD('day', -365, max_date) AS window_start, max_date AS window_end FROM dataset_max_date),
    patients_active_in_window AS (
        SELECT DISTINCT ca.patient_id, ca.segment
        FROM classifiable_all_time ca CROSS JOIN window_bounds wb
        WHERE ca.visit_date BETWEEN wb.window_start AND wb.window_end
    ),
    relevant_visits AS (
        SELECT ca.* FROM classifiable_all_time ca
        JOIN patients_active_in_window paiw ON paiw.patient_id = ca.patient_id AND paiw.segment = ca.segment
    ),
    patient_segment_last_visit AS (
        SELECT patient_id, segment, MAX(visit_date) AS last_visit_date
        FROM relevant_visits GROUP BY patient_id, segment
    ),
    ltfu_patients AS (
        SELECT psl.patient_id, psl.segment, psl.last_visit_date
        FROM patient_segment_last_visit psl CROSS JOIN dataset_max_date dmd
        WHERE DATEDIFF('day', psl.last_visit_date, dmd.max_date) > 180
    ),
    ltfu_final_visit AS (
        SELECT lp.patient_id, lp.segment, rv.visit_id, rv.source_system
        FROM ltfu_patients lp
        JOIN relevant_visits rv ON rv.patient_id = lp.patient_id AND rv.segment = lp.segment AND rv.visit_date = lp.last_visit_date
        QUALIFY ROW_NUMBER() OVER (PARTITION BY lp.patient_id, lp.segment ORDER BY rv.visit_id) = 1
    ),
    had_procedure AS (
        SELECT DISTINCT visit_id, source_system FROM HOSPITALS.STAGING.STG_PROCEDURES
    ),
    had_medication AS (
        SELECT DISTINCT visit_id, source_system FROM HOSPITALS.STAGING.stg_pharmacy_orders
    ),
    had_investigation AS (
        SELECT DISTINCT visit_id, source_system FROM HOSPITALS.STAGING.STG_SPH_INVESTIGATIONS
    ),
    had_imaging AS (
        SELECT DISTINCT visit_id, source_system FROM HOSPITALS.STAGING.STG_IMAGING_ORDERS
    )
    SELECT
        lfv.segment,
        COUNT(*) AS total_ltfu_patients,
        SUM(CASE WHEN hp.visit_id IS NOT NULL THEN 1 ELSE 0 END) AS had_theatre_procedure,
        SUM(CASE WHEN hm.visit_id IS NOT NULL THEN 1 ELSE 0 END) AS had_medication_pickup,
        SUM(CASE WHEN hi.visit_id IS NOT NULL THEN 1 ELSE 0 END) AS had_investigation_or_lab,
        SUM(CASE WHEN himg.visit_id IS NOT NULL THEN 1 ELSE 0 END) AS had_imaging,
        SUM(CASE WHEN hp.visit_id IS NULL AND hm.visit_id IS NULL AND hi.visit_id IS NULL AND himg.visit_id IS NULL
                 THEN 1 ELSE 0 END) AS consultation_only_no_other_record
    FROM ltfu_final_visit lfv
    LEFT JOIN had_procedure hp ON hp.visit_id = lfv.visit_id AND hp.source_system = lfv.source_system
    LEFT JOIN had_medication hm ON hm.visit_id = lfv.visit_id AND hm.source_system = lfv.source_system
    LEFT JOIN had_investigation hi ON hi.visit_id = lfv.visit_id AND hi.source_system = lfv.source_system
    LEFT JOIN had_imaging himg ON himg.visit_id = lfv.visit_id AND himg.source_system = lfv.source_system
    GROUP BY lfv.segment
    ORDER BY total_ltfu_patients DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# A8b — patient-level LTFU investigation signals, all segments
# ---------------------------------------------------------------------------
# Answers "of this segment's LTFU patients, which ones look like true,
# unexpected loss vs. expected completion/transfer?" at the patient level,
# using only signals that actually exist in this schema:
#   - diagnosis text + a text-detected-followup pattern already used
#     elsewhere in this file (get_fr_active_channel) — "review", "f/u",
#     "/52", "/12", "post", "status post" in the diagnosis text itself
#   - a "Schedule Follow Up" signal from the same order-level EAV field
#     already used for scheduled_follow_up_date elsewhere in this file
#   - whether a procedure/medication/investigation/imaging record exists
#     for the LTFU-triggering visit (reuses get_fr_ltfu_last_pathway's joins)
#   - whether the patient has ANY later visit anywhere else in the hospital
#     (any segment, including ones excluded from this taxonomy) — the only
#     way to tell "disappeared" from "kept coming, just not to this segment"
# NOTE: no column for referral destination, discharge status, or clinician
# recommendation was found anywhere in this codebase's queries — if that
# data exists elsewhere in the source system, this query does not cover it.

@st.cache_data(ttl=3600)
def get_fr_ltfu_patient_level_signals() -> pd.DataFrame:
    """One row per LTFU patient per segment, for ad-hoc investigation of why
    each patient stopped returning. Columns: SEGMENT, PATIENT_ID,
    LAST_VISIT_DATE, DAYS_SINCE_LAST_VISIT, LTFU_THRESHOLD_DAYS (fixed at
    180 — the definition applied everywhere in this module),
    VISIT_NUMBER_AT_LTFU, DIAGNOSIS_TEXT, IS_TEXT_DETECTED_FOLLOWUP,
    SCHEDULED_FOLLOWUP_DATE (actual date from the "Schedule Follow Up" order
    field, NULL if none), HAD_SCHEDULED_FOLLOWUP, HAD_PROCEDURE,
    PROCEDURE_NAMES (array of procedure names on the final visit, NULL if
    none), HAD_MEDICATION, HAD_INVESTIGATION, HAD_IMAGING,
    HAS_LATER_VISIT_ELSEWHERE, NEXT_VISIT_ELSEWHERE_DATE,
    NEXT_VISIT_ELSEWHERE_SEGMENT (date/segment of the next visit anywhere in
    the hospital after the LTFU-triggering one — a proxy for internal
    transfer, since no referral-destination field exists in this schema),
    LIKELY_EXPLAINED (1 if any of HAD_SCHEDULED_FOLLOWUP /
    IS_TEXT_DETECTED_FOLLOWUP / HAS_LATER_VISIT_ELSEWHERE is true, else 0 —
    the unexplained-and-worth-auditing subset).

    Discharge status, clinician recommendation, and referral destination
    are not present anywhere in this schema and are not represented here.
    """
    sql = """
    WITH raw_diag AS (
        SELECT
            visit_id, patient_id, diagnosis_created_at AS visit_date, source_system,
            TRIM(bg.value::STRING) AS split_burden,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text,
            COALESCE(diagnosis_name_expanded, icd10_names, '') AS dx_display,
            CASE WHEN LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%post %'
                 OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%s/p %'
                 OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%status post%'
                 OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%follow up%'
                 OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%followup%'
                 OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%f/u %'
                 OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%review%'
                 OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%/52%'
                 OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%/12%'
                THEN 1 ELSE 0 END AS is_text_detected_followup
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
        LATERAL FLATTEN(input => SPLIT(burden_group, '|')) bg
    ),
    segmented AS (
        SELECT DISTINCT
            visit_id, patient_id, visit_date, source_system, dx_display, is_text_detected_followup,
                       CASE
                -- Exclusions
                WHEN clean_dx_text LIKE '%hernia%' THEN 'EXCLUDE: General Surgery'
                WHEN (clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%')
                     AND (clean_dx_text LIKE '%post %' OR clean_dx_text LIKE '%revision%')
                    THEN 'EXCLUDE: Fibroids-surgical'

                -- Fibroids
                WHEN clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%'
                    THEN 'Fibroids-conservative'

                -- Spine-structural: explicit structural pathology
                WHEN (clean_dx_text LIKE '%spine%' OR clean_dx_text LIKE '%spinal%' OR clean_dx_text LIKE '%sciatica%'
                      OR clean_dx_text LIKE '%lumbar%' OR clean_dx_text LIKE '%lumbago%' OR clean_dx_text LIKE '%lumbarg%'
                      OR clean_dx_text LIKE '%low back%' OR clean_dx_text LIKE '%lbp%'
                      OR clean_dx_text LIKE '%lordosis%' OR clean_dx_text LIKE '%spondylo%'
                      OR clean_dx_text LIKE '%disc bulge%' OR clean_dx_text LIKE '%disc herniat%'
                      OR clean_dx_text LIKE '%radiculopath%' OR clean_dx_text LIKE '%degenerative disc%'
                      OR split_burden = 'Ortho: Spine')
                     AND (clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%dislocat%' OR clean_dx_text LIKE '%stenosis%'
                          OR clean_dx_text LIKE '%herniat%' OR clean_dx_text LIKE '%compression%' OR clean_dx_text LIKE '%tumour%'
                          OR clean_dx_text LIKE '%decompression%' OR clean_dx_text LIKE '%fusion%'
                          OR clean_dx_text LIKE '%disc bulge%' OR clean_dx_text LIKE '%multilevel%'
                          OR clean_dx_text LIKE '%degenerative disc%' OR clean_dx_text LIKE '%degenarative%')
                    THEN 'Spine-structural'

                -- Spine-conservative: pain/functional without structural pathology
                WHEN clean_dx_text LIKE '%spine%' OR clean_dx_text LIKE '%spinal%' OR clean_dx_text LIKE '%sciatica%'
                     OR clean_dx_text LIKE '%lumbar%' OR clean_dx_text LIKE '%lumbago%' OR clean_dx_text LIKE '%lumbarg%'
                     OR clean_dx_text LIKE '%low back%' OR clean_dx_text LIKE '%lbp%'
                     OR clean_dx_text LIKE '%lordosis%' OR clean_dx_text LIKE '%spondylo%'
                     OR clean_dx_text LIKE '%radiculopath%' OR clean_dx_text LIKE '%upper back pain%'
                     OR (clean_dx_text LIKE '%back%' AND clean_dx_text LIKE '%pain%')
                     OR (split_burden = 'Ortho: Spine' AND clean_dx_text NOT LIKE '%post %')
                    THEN 'Spine-conservative'

                -- Core Orthopedics: fracture, OA, trauma
                WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%osteoarthritis%'
                     OR clean_dx_text LIKE '%osteoathritis%'
                     OR (clean_dx_text LIKE '%oa %' AND (clean_dx_text LIKE '%knee%' OR clean_dx_text LIKE '%hip%'))
                     OR clean_dx_text LIKE '%osgood%'
                     OR split_burden = 'Ortho: Fracture & Trauma'
                    THEN 'Core Orthopedics: Fracture & Trauma'

                -- Joint: Knee
                WHEN clean_dx_text LIKE '%knee%'
                     OR split_burden = 'Ortho: Knee'
                    THEN 'Joint: Knee'

                -- Joint: Hip
                WHEN clean_dx_text LIKE '%hip%'
                     OR split_burden = 'Ortho: Hip'
                    THEN 'Joint: Hip'

                -- Soft Tissue & MSK
                WHEN clean_dx_text LIKE '%soft tissue%' OR clean_dx_text LIKE '%myalgia%'
                     OR clean_dx_text LIKE '%muscle spasm%' OR clean_dx_text LIKE '%sprain%'
                     OR clean_dx_text LIKE '%plantar fasci%' OR clean_dx_text LIKE '%piriformis%'
                     OR clean_dx_text LIKE '%calcaneal spur%' OR clean_dx_text LIKE '%tendin%'
                     OR clean_dx_text LIKE '%ligament%' OR clean_dx_text LIKE '%bursitis%'
                     OR clean_dx_text LIKE '%shoulder%' OR clean_dx_text LIKE '%ankle%'
                     OR split_burden = 'Ortho: Soft Tissue & MSK'
                    THEN 'Soft Tissue & MSK'

                -- Post-Surgical Follow-up
                WHEN clean_dx_text LIKE '%post %replacement%' OR clean_dx_text LIKE '%post %nailing%'
                     OR clean_dx_text LIKE '%post %fusion%' OR clean_dx_text LIKE '%post %fixat%'
                     OR clean_dx_text LIKE '%post decompression%' OR clean_dx_text LIKE '%post implant%'
                     OR clean_dx_text LIKE '%post ankle%' OR clean_dx_text LIKE '%post %arthroplasty%'
                     OR REGEXP_LIKE(clean_dx_text, '\\d+/52 post|\\d+/12 post')
                    THEN 'Post-Surgical Follow-up'

                -- High-Risk Pregnancy
                WHEN clean_dx_text LIKE '%pre-eclamp%' OR clean_dx_text LIKE '%preeclamp%' OR clean_dx_text LIKE '%eclamp%'
                     OR clean_dx_text LIKE '%hyperem%' OR clean_dx_text LIKE '%pprom%'
                    THEN 'High-Risk Pregnancy'

                -- ANC / Routine Pregnancy (fixed: removed risky %anc%)
                WHEN clean_dx_text LIKE '%pregnan%' OR clean_dx_text LIKE '%antenatal%'
                     OR clean_dx_text LIKE '%gravid%' OR split_burden = 'General: Obstetric'
                    THEN 'ANC / Routine Pregnancy'

                -- General Medical: Respiratory & ENT
                WHEN clean_dx_text LIKE '%respiratory%' OR clean_dx_text LIKE '%urti%'
                     OR clean_dx_text LIKE '%tonsil%' OR clean_dx_text LIKE '%pharyngitis%'
                     OR clean_dx_text LIKE '%bronchit%' OR clean_dx_text LIKE '%pneumonia%'
                     OR split_burden = 'General: Respiratory' OR split_burden = 'General: ENT'
                    THEN 'General Medical: Respiratory & ENT'

                -- General Medical: Gastrointestinal
                WHEN clean_dx_text LIKE '%gastri%' OR clean_dx_text LIKE '%gastroenter%'
                     OR clean_dx_text LIKE '%acute ge%' OR clean_dx_text LIKE '%bowel%'
                     OR split_burden = 'General: Gastrointestinal'
                    THEN 'General Medical: Gastrointestinal'

                -- General Medical: Genitourinary
                WHEN clean_dx_text LIKE '%urinary tract%' OR clean_dx_text LIKE '% uti%'
                     OR clean_dx_text LIKE '%bph%' OR clean_dx_text LIKE '%prostat%'
                     OR split_burden = 'General: Genitourinary'
                    THEN 'General Medical: Genitourinary'

                -- General Medical: Neurology
                WHEN clean_dx_text LIKE '%stroke%' OR clean_dx_text LIKE '%paraplegia%'
                     OR clean_dx_text LIKE '%epilep%' OR clean_dx_text LIKE '%cerebral%'
                     OR split_burden = 'General: Neurology'
                    THEN 'General Medical: Neurology'

                -- General Medical: Cardiovascular
                WHEN split_burden = 'General: Cardiovascular' OR split_burden = 'Chronic Disease: Cardiovascular'
                    THEN 'General Medical: Cardiovascular'

                -- General Medical: Endocrine & Metabolic
                WHEN clean_dx_text LIKE '%goitre%' OR clean_dx_text LIKE '%thyroid%' OR clean_dx_text LIKE '%diabet%'
                     OR split_burden = 'General: Endocrine & Metabolic' OR split_burden = 'Chronic Disease: Metabolic'
                    THEN 'General Medical: Endocrine & Metabolic'

                -- General Medical: Infection & Sepsis
                WHEN clean_dx_text LIKE '%septi%' OR clean_dx_text LIKE '%sepsis%'
                     OR clean_dx_text LIKE '%abscess%' OR clean_dx_text LIKE '%cellulitis%'
                     OR split_burden = 'General: Infection & Sepsis'
                    THEN 'General Medical: Infection & Sepsis'

                -- General Medical: Gynaecology
                WHEN split_burden = 'General: Gynaecology'
                    THEN 'General Medical: Gynaecology'

                -- General Medical: Surgery (non-hernia)
                WHEN split_burden = 'General: Surgery'
                    THEN 'General Medical: Surgery'

                -- Catch remaining burden groups
                WHEN split_burden LIKE 'General:%' OR split_burden LIKE 'Chronic Disease:%'
                    THEN 'General Medical: ' || REPLACE(REPLACE(split_burden, 'General: ', ''), 'Chronic Disease: ', '')

                ELSE 'Unclassified'
            END AS segment
        FROM raw_diag
    ),
    -- All visits, unfiltered by segment — used only to detect "did this
    -- patient show up anywhere else in the hospital after their last visit
    -- in the segment being examined?" Includes 'Other'/excluded categories
    -- on purpose: a transfer to General Surgery still means they didn't vanish.
    all_patient_visits AS (
        SELECT DISTINCT patient_id, visit_date FROM segmented
    ),
    classifiable_all_time AS (
        SELECT * FROM segmented
        WHERE segment NOT IN ('EXCLUDE: General Surgery', 'EXCLUDE: Fibroids-surgical', 'Other')
    ),
    dataset_max_date AS (SELECT MAX(visit_date) AS max_date FROM classifiable_all_time),
    window_bounds AS (SELECT DATEADD('day', -365, max_date) AS window_start, max_date AS window_end FROM dataset_max_date),
    patients_active_in_window AS (
        SELECT DISTINCT ca.patient_id, ca.segment
        FROM classifiable_all_time ca CROSS JOIN window_bounds wb
        WHERE ca.visit_date BETWEEN wb.window_start AND wb.window_end
    ),
    relevant_visits AS (
        SELECT ca.* FROM classifiable_all_time ca
        JOIN patients_active_in_window paiw ON paiw.patient_id = ca.patient_id AND paiw.segment = ca.segment
    ),
    visit_sequence AS (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY patient_id, segment ORDER BY visit_date) AS visit_number
        FROM relevant_visits
    ),
    patient_segment_last_visit AS (
        SELECT patient_id, segment, MAX(visit_date) AS last_visit_date
        FROM relevant_visits GROUP BY patient_id, segment
    ),
    ltfu_patients AS (
        SELECT psl.patient_id, psl.segment, psl.last_visit_date
        FROM patient_segment_last_visit psl CROSS JOIN dataset_max_date dmd
        WHERE DATEDIFF('day', psl.last_visit_date, dmd.max_date) > 180
    ),
    ltfu_final_visit AS (
        SELECT
            lp.patient_id, lp.segment, lp.last_visit_date,
            vs.visit_id, vs.source_system, vs.dx_display, vs.is_text_detected_followup, vs.visit_number
        FROM ltfu_patients lp
        JOIN visit_sequence vs
            ON vs.patient_id = lp.patient_id AND vs.segment = lp.segment AND vs.visit_date = lp.last_visit_date
        QUALIFY ROW_NUMBER() OVER (PARTITION BY lp.patient_id, lp.segment ORDER BY vs.visit_id) = 1
    ),
    had_procedure AS (SELECT DISTINCT visit_id, source_system FROM HOSPITALS.STAGING.STG_PROCEDURES),
    procedure_names AS (
        SELECT visit_id, source_system, ARRAY_AGG(DISTINCT procedure_name) AS procedure_names
        FROM HOSPITALS.STAGING.STG_PROCEDURES
        WHERE procedure_name IS NOT NULL
        GROUP BY visit_id, source_system
    ),
    had_medication AS (SELECT DISTINCT visit_id, source_system FROM HOSPITALS.STAGING.stg_pharmacy_orders),
    had_investigation AS (SELECT DISTINCT visit_id, source_system FROM HOSPITALS.STAGING.STG_SPH_INVESTIGATIONS),
    had_imaging AS (SELECT DISTINCT visit_id, source_system FROM HOSPITALS.STAGING.STG_IMAGING_ORDERS),
    -- Same "Schedule Follow Up" EAV signal used for scheduled_follow_up_date
    -- elsewhere in this file — kept here as the actual date, not just a flag,
    -- per the "planned follow-up dates" investigation dimension.
    oie_flat AS (
        SELECT o."id" AS oie_id, o."ref" AS soi_id, o."item" AS sale_item_id,
            f.VALUE:"name"::STRING AS field_name, v.VALUE::STRING AS field_value
        FROM HOSPITALS.ORTHOPEDIC_CLEAN."ORDERITEMENTRIES" o,
        LATERAL FLATTEN(INPUT => TRY_PARSE_JSON(o."fields"), OUTER => TRUE) f,
        LATERAL FLATTEN(INPUT => f.VALUE:"value", OUTER => TRUE) v
        WHERE f.VALUE:"name"::STRING IS NOT NULL
    ),
    doctor_pivot AS (
        SELECT oie.soi_id,
            MAX(CASE WHEN oie.field_name = 'Schedule Follow Up' THEN TRY_TO_TIMESTAMP(oie.field_value) END) AS scheduled_follow_up_date
        FROM oie_flat oie
        JOIN HOSPITALS.ORTHOPEDIC_CLEAN."SALEITEMS" si ON si."id" = oie.sale_item_id
        WHERE si."category" = 'Doctor'
        GROUP BY oie.soi_id
    ),
    orders_flat AS (
        SELECT oo.id AS order_id, f.VALUE::STRING AS item_id
        FROM HOSPITALS.ORTHOPEDIC_CLEAN."ORTHO_ORDERS" oo,
        LATERAL FLATTEN(INPUT => TRY_PARSE_JSON(oo.items)) f
        QUALIFY ROW_NUMBER() OVER (PARTITION BY f.VALUE::STRING ORDER BY oo.registered_at DESC) = 1
    ),
    scheduled_signal_raw AS (
        SELECT v.patient_id AS clean_patient_id, MAX(dp.scheduled_follow_up_date) AS scheduled_follow_up_date
        FROM doctor_pivot dp
        JOIN orders_flat of_ ON of_.item_id = dp.soi_id
        JOIN HOSPITALS.STAGING.STG_VISITS v ON v.visit_id = of_.order_id
        WHERE dp.scheduled_follow_up_date IS NOT NULL
        GROUP BY v.patient_id
    ),
    -- Earliest visit after the LTFU-triggering one, anywhere in the hospital
    -- (any segment/department) — a proxy for "did they transfer to another
    -- clinical area" since no explicit referral-destination field exists.
    later_visit_elsewhere AS (
        SELECT
            lfv.patient_id, lfv.segment,
            MIN(apv.visit_date) AS next_visit_date
        FROM ltfu_final_visit lfv
        JOIN all_patient_visits apv
            ON apv.patient_id = lfv.patient_id AND apv.visit_date > lfv.last_visit_date
        GROUP BY lfv.patient_id, lfv.segment
    ),
    later_visit_segment AS (
        SELECT lve.patient_id, lve.segment, MIN(s.segment) AS next_visit_segment
        FROM later_visit_elsewhere lve
        JOIN segmented s ON s.patient_id = lve.patient_id AND s.visit_date = lve.next_visit_date
        GROUP BY lve.patient_id, lve.segment
    )
    SELECT
        lfv.segment                                                                  AS SEGMENT,
        lfv.patient_id                                                               AS PATIENT_ID,
        lfv.last_visit_date                                                          AS LAST_VISIT_DATE,
        DATEDIFF('day', lfv.last_visit_date, dmd.max_date)                           AS DAYS_SINCE_LAST_VISIT,
        180                                                                          AS LTFU_THRESHOLD_DAYS,
        CASE WHEN lfv.visit_number >= 7 THEN '7+' ELSE lfv.visit_number::STRING END  AS VISIT_NUMBER_AT_LTFU,
        lfv.dx_display                                                              AS DIAGNOSIS_TEXT,
        lfv.is_text_detected_followup                                               AS IS_TEXT_DETECTED_FOLLOWUP,
        ssr.scheduled_follow_up_date                                                AS SCHEDULED_FOLLOWUP_DATE,
        CASE WHEN ssr.clean_patient_id IS NOT NULL THEN 1 ELSE 0 END                 AS HAD_SCHEDULED_FOLLOWUP,
        CASE WHEN hp.visit_id IS NOT NULL THEN 1 ELSE 0 END                          AS HAD_PROCEDURE,
        pn.procedure_names                                                          AS PROCEDURE_NAMES,
        CASE WHEN hm.visit_id IS NOT NULL THEN 1 ELSE 0 END                          AS HAD_MEDICATION,
        CASE WHEN hi.visit_id IS NOT NULL THEN 1 ELSE 0 END                          AS HAD_INVESTIGATION,
        CASE WHEN himg.visit_id IS NOT NULL THEN 1 ELSE 0 END                        AS HAD_IMAGING,
        CASE WHEN lve.patient_id IS NOT NULL THEN 1 ELSE 0 END                       AS HAS_LATER_VISIT_ELSEWHERE,
        lve.next_visit_date                                                         AS NEXT_VISIT_ELSEWHERE_DATE,
        lvs.next_visit_segment                                                      AS NEXT_VISIT_ELSEWHERE_SEGMENT,
        CASE WHEN ssr.clean_patient_id IS NOT NULL
               OR lfv.is_text_detected_followup = 1
               OR lve.patient_id IS NOT NULL
             THEN 1 ELSE 0 END                                                       AS LIKELY_EXPLAINED
    FROM ltfu_final_visit lfv
    CROSS JOIN dataset_max_date dmd
    LEFT JOIN had_procedure hp ON hp.visit_id = lfv.visit_id AND hp.source_system = lfv.source_system
    LEFT JOIN procedure_names pn ON pn.visit_id = lfv.visit_id AND pn.source_system = lfv.source_system
    LEFT JOIN had_medication hm ON hm.visit_id = lfv.visit_id AND hm.source_system = lfv.source_system
    LEFT JOIN had_investigation hi ON hi.visit_id = lfv.visit_id AND hi.source_system = lfv.source_system
    LEFT JOIN had_imaging himg ON himg.visit_id = lfv.visit_id AND himg.source_system = lfv.source_system
    LEFT JOIN scheduled_signal_raw ssr ON ssr.clean_patient_id = lfv.patient_id
    LEFT JOIN later_visit_elsewhere lve ON lve.patient_id = lfv.patient_id AND lve.segment = lfv.segment
    LEFT JOIN later_visit_segment lvs ON lvs.patient_id = lfv.patient_id AND lvs.segment = lfv.segment
    ORDER BY lfv.segment, lfv.last_visit_date DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# A8c — patient-level LTFU pathway signals, restricted to the 6 core
# segments (same population as get_fr_status_overall's "2.8K of 5.8K
# classifiable" KPI). get_fr_ltfu_patient_level_signals() above uses a much
# wider ~18-segment taxonomy AND never applies the 300-day pregnancy-episode
# cap that get_fr_status_overall does for ANC/High-Risk Pregnancy, so
# filtering its output by segment name afterward still won't reconcile to
# the KPI strip. This function reuses get_fr_status_overall's exact
# population-defining CTE chain (segmentation, 365-day window, pregnancy
# cap, LTFU classification) verbatim, then joins the same investigative
# signals (procedure/medication/investigation/imaging/scheduled follow-up/
# next-visit-elsewhere) used in get_fr_ltfu_patient_level_signals on top of
# that identical population — so this is the version that actually matches.
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_fr_ltfu_pathway_signals_core_segments() -> pd.DataFrame:
    """Same population as get_fr_status_overall's LTFU count (6 core
    segments, 365-day active window, 300-day pregnancy-episode cap).
    Columns: SEGMENT, PATIENT_ID, LAST_VISIT_DATE, DAYS_SINCE_LAST_VISIT,
    VISIT_NUMBER_AT_LTFU, HAD_SCHEDULED_FOLLOWUP, HAD_PROCEDURE,
    HAD_MEDICATION, HAD_INVESTIGATION, HAD_IMAGING,
    HAS_LATER_VISIT_ELSEWHERE, NEXT_VISIT_ELSEWHERE_DATE,
    NEXT_VISIT_ELSEWHERE_SEGMENT."""
    sql = """
    WITH raw_diag AS (
        SELECT
            visit_id, patient_id, diagnosis_created_at AS visit_date, source_system,
            TRIM(bg.value::STRING) AS split_burden,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
        LATERAL FLATTEN(input => SPLIT(burden_group, '|')) bg
    ),
    segmented AS (
        SELECT DISTINCT
            visit_id, patient_id, visit_date, source_system,
            CASE
                WHEN clean_dx_text LIKE '%hernia%' THEN 'EXCLUDE: General Surgery'
                WHEN (clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%')
                     AND (clean_dx_text LIKE '%post %' OR clean_dx_text LIKE '%revision%')
                    THEN 'EXCLUDE: Fibroids-surgical'
                WHEN clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%'
                    THEN 'Fibroids-conservative'
                WHEN clean_dx_text LIKE '%spine%' OR clean_dx_text LIKE '%sciatica%' THEN
                    CASE
                        WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%dislocat%' OR clean_dx_text LIKE '%stenosis%'
                             OR clean_dx_text LIKE '%herniat%' OR clean_dx_text LIKE '%compression%' OR clean_dx_text LIKE '%tumour%'
                            THEN 'Spine-structural'
                        ELSE 'Spine-conservative'
                    END
                WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%osteoarthritis%' THEN 'Core Orthopedics: General'
                WHEN clean_dx_text LIKE '%pre-eclamp%' OR clean_dx_text LIKE '%preeclamp%' OR clean_dx_text LIKE '%eclamp%'
                     OR clean_dx_text LIKE '%hyperem%' OR clean_dx_text LIKE '%pprom%'
                    THEN 'High-Risk Pregnancy'
                WHEN clean_dx_text LIKE '%pregnan%' OR clean_dx_text LIKE '%anc%' OR clean_dx_text LIKE '%antenatal%'
                     OR clean_dx_text LIKE '%gravid%' OR split_burden = 'General: Obstetric'
                    THEN 'ANC / Routine Pregnancy'
                ELSE 'Other'
            END AS segment
        FROM raw_diag
    ),
    all_patient_visits AS (
        SELECT DISTINCT patient_id, visit_date FROM segmented
    ),
    classifiable_all_time AS (
        SELECT * FROM segmented
        WHERE segment NOT IN ('EXCLUDE: General Surgery', 'EXCLUDE: Fibroids-surgical', 'Other')
    ),
    dataset_max_date AS (SELECT MAX(visit_date) AS max_date FROM classifiable_all_time),
    window_bounds AS (
        SELECT DATEADD('day', -365, max_date) AS window_start, max_date AS window_end
        FROM dataset_max_date
    ),
    patients_active_in_window AS (
        SELECT DISTINCT ca.patient_id, ca.segment
        FROM classifiable_all_time ca CROSS JOIN window_bounds wb
        WHERE ca.visit_date BETWEEN wb.window_start AND wb.window_end
    ),
    relevant_visits AS (
        SELECT ca.visit_id, ca.patient_id, ca.segment, ca.visit_date, ca.source_system
        FROM classifiable_all_time ca
        JOIN patients_active_in_window paiw
            ON paiw.patient_id = ca.patient_id AND paiw.segment = ca.segment
    ),
    pregnancy_capped AS (
        SELECT
            rv.*,
            MIN(rv.visit_date) OVER (PARTITION BY rv.patient_id, rv.segment) AS first_preg_visit
        FROM relevant_visits rv
        WHERE rv.segment IN ('ANC / Routine Pregnancy', 'High-Risk Pregnancy')
    ),
    pregnancy_within_window AS (
        SELECT visit_id, patient_id, visit_date, source_system, segment
        FROM pregnancy_capped
        WHERE DATEDIFF('day', first_preg_visit, visit_date) <= 300
    ),
    non_pregnancy AS (
        SELECT visit_id, patient_id, visit_date, source_system, segment
        FROM relevant_visits
        WHERE segment NOT IN ('ANC / Routine Pregnancy', 'High-Risk Pregnancy')
    ),
    final_visits AS (
        SELECT * FROM non_pregnancy
        UNION ALL
        SELECT * FROM pregnancy_within_window
    ),
    visit_sequence AS (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY patient_id, segment ORDER BY visit_date) AS visit_number
        FROM final_visits
    ),
    patient_segment_last_visit AS (
        SELECT patient_id, segment, MAX(visit_date) AS last_visit_date
        FROM final_visits
        GROUP BY patient_id, segment
    ),
    ltfu_patients AS (
        SELECT psl.patient_id, psl.segment, psl.last_visit_date
        FROM patient_segment_last_visit psl CROSS JOIN dataset_max_date dmd
        WHERE DATEDIFF('day', psl.last_visit_date, dmd.max_date) > 180
    ),
    ltfu_final_visit AS (
        SELECT
            lp.patient_id, lp.segment, lp.last_visit_date,
            vs.visit_id, vs.source_system, vs.visit_number
        FROM ltfu_patients lp
        JOIN visit_sequence vs
            ON vs.patient_id = lp.patient_id AND vs.segment = lp.segment AND vs.visit_date = lp.last_visit_date
        QUALIFY ROW_NUMBER() OVER (PARTITION BY lp.patient_id, lp.segment ORDER BY vs.visit_id) = 1
    ),
    had_procedure AS (SELECT DISTINCT visit_id, source_system FROM HOSPITALS.STAGING.STG_PROCEDURES),
    had_medication AS (SELECT DISTINCT visit_id, source_system FROM HOSPITALS.STAGING.stg_pharmacy_orders),
    had_investigation AS (SELECT DISTINCT visit_id, source_system FROM HOSPITALS.STAGING.STG_SPH_INVESTIGATIONS),
    had_imaging AS (SELECT DISTINCT visit_id, source_system FROM HOSPITALS.STAGING.STG_IMAGING_ORDERS),
    oie_flat AS (
        SELECT o."id" AS oie_id, o."ref" AS soi_id, o."item" AS sale_item_id,
            f.VALUE:"name"::STRING AS field_name, v.VALUE::STRING AS field_value
        FROM HOSPITALS.ORTHOPEDIC_CLEAN."ORDERITEMENTRIES" o,
        LATERAL FLATTEN(INPUT => TRY_PARSE_JSON(o."fields"), OUTER => TRUE) f,
        LATERAL FLATTEN(INPUT => f.VALUE:"value", OUTER => TRUE) v
        WHERE f.VALUE:"name"::STRING IS NOT NULL
    ),
    doctor_pivot AS (
        SELECT oie.soi_id,
            MAX(CASE WHEN oie.field_name = 'Schedule Follow Up' THEN TRY_TO_TIMESTAMP(oie.field_value) END) AS scheduled_follow_up_date
        FROM oie_flat oie
        JOIN HOSPITALS.ORTHOPEDIC_CLEAN."SALEITEMS" si ON si."id" = oie.sale_item_id
        WHERE si."category" = 'Doctor'
        GROUP BY oie.soi_id
    ),
    orders_flat AS (
        SELECT oo.id AS order_id, f.VALUE::STRING AS item_id
        FROM HOSPITALS.ORTHOPEDIC_CLEAN."ORTHO_ORDERS" oo,
        LATERAL FLATTEN(INPUT => TRY_PARSE_JSON(oo.items)) f
        QUALIFY ROW_NUMBER() OVER (PARTITION BY f.VALUE::STRING ORDER BY oo.registered_at DESC) = 1
    ),
    scheduled_signal_raw AS (
        SELECT v.patient_id AS clean_patient_id, MAX(dp.scheduled_follow_up_date) AS scheduled_follow_up_date
        FROM doctor_pivot dp
        JOIN orders_flat of_ ON of_.item_id = dp.soi_id
        JOIN HOSPITALS.STAGING.STG_VISITS v ON v.visit_id = of_.order_id
        WHERE dp.scheduled_follow_up_date IS NOT NULL
        GROUP BY v.patient_id
    ),
    later_visit_elsewhere AS (
        SELECT
            lfv.patient_id, lfv.segment,
            MIN(apv.visit_date) AS next_visit_date
        FROM ltfu_final_visit lfv
        JOIN all_patient_visits apv
            ON apv.patient_id = lfv.patient_id AND apv.visit_date > lfv.last_visit_date
        GROUP BY lfv.patient_id, lfv.segment
    ),
    later_visit_segment AS (
        SELECT lve.patient_id, lve.segment, MIN(s.segment) AS next_visit_segment
        FROM later_visit_elsewhere lve
        JOIN segmented s ON s.patient_id = lve.patient_id AND s.visit_date = lve.next_visit_date
        GROUP BY lve.patient_id, lve.segment
    )
    SELECT
        lfv.segment                                                                  AS SEGMENT,
        lfv.patient_id                                                               AS PATIENT_ID,
        lfv.last_visit_date                                                          AS LAST_VISIT_DATE,
        DATEDIFF('day', lfv.last_visit_date, dmd.max_date)                           AS DAYS_SINCE_LAST_VISIT,
        CASE WHEN lfv.visit_number >= 7 THEN '7+' ELSE lfv.visit_number::STRING END  AS VISIT_NUMBER_AT_LTFU,
        ssr.scheduled_follow_up_date                                                AS SCHEDULED_FOLLOWUP_DATE,
        CASE WHEN ssr.clean_patient_id IS NOT NULL THEN 1 ELSE 0 END                 AS HAD_SCHEDULED_FOLLOWUP,
        CASE WHEN hp.visit_id IS NOT NULL THEN 1 ELSE 0 END                          AS HAD_PROCEDURE,
        CASE WHEN hm.visit_id IS NOT NULL THEN 1 ELSE 0 END                          AS HAD_MEDICATION,
        CASE WHEN hi.visit_id IS NOT NULL THEN 1 ELSE 0 END                          AS HAD_INVESTIGATION,
        CASE WHEN himg.visit_id IS NOT NULL THEN 1 ELSE 0 END                        AS HAD_IMAGING,
        CASE WHEN lve.patient_id IS NOT NULL THEN 1 ELSE 0 END                       AS HAS_LATER_VISIT_ELSEWHERE,
        lve.next_visit_date                                                         AS NEXT_VISIT_ELSEWHERE_DATE,
        lvs.next_visit_segment                                                      AS NEXT_VISIT_ELSEWHERE_SEGMENT
    FROM ltfu_final_visit lfv
    CROSS JOIN dataset_max_date dmd
    LEFT JOIN had_procedure hp ON hp.visit_id = lfv.visit_id AND hp.source_system = lfv.source_system
    LEFT JOIN had_medication hm ON hm.visit_id = lfv.visit_id AND hm.source_system = lfv.source_system
    LEFT JOIN had_investigation hi ON hi.visit_id = lfv.visit_id AND hi.source_system = lfv.source_system
    LEFT JOIN had_imaging himg ON himg.visit_id = lfv.visit_id AND himg.source_system = lfv.source_system
    LEFT JOIN scheduled_signal_raw ssr ON ssr.clean_patient_id = lfv.patient_id
    LEFT JOIN later_visit_elsewhere lve ON lve.patient_id = lfv.patient_id AND lve.segment = lfv.segment
    LEFT JOIN later_visit_segment lvs ON lvs.patient_id = lfv.patient_id AND lvs.segment = lfv.segment
    ORDER BY lfv.segment, lfv.last_visit_date DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# A9 — LTFU condition breakdown (Ortho General + both Spine sub-segments)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_fr_ltfu_condition_breakdown() -> pd.DataFrame:
    """Columns: SEGMENT, CONDITION_CATEGORY, DISTINCT_LTFU_PATIENTS, PCT_WITHIN_SEGMENT."""
    sql = """
    WITH raw_diag AS (
        SELECT
            visit_id, patient_id, diagnosis_created_at AS visit_date,
            TRIM(bg.value::STRING) AS split_burden,
            CASE
                WHEN icd10_names IS NULL OR TRIM(icd10_names) = '' THEN LOWER(COALESCE(diagnosis_name_expanded, ''))
                WHEN diagnosis_name_expanded IS NULL OR TRIM(diagnosis_name_expanded) = '' THEN LOWER(COALESCE(icd10_names, ''))
                WHEN LOWER(icd10_names) LIKE '%' || LOWER(diagnosis_name_expanded) || '%' THEN LOWER(icd10_names)
                WHEN LOWER(diagnosis_name_expanded) LIKE '%' || LOWER(icd10_names) || '%' THEN LOWER(diagnosis_name_expanded)
                ELSE LOWER(diagnosis_name_expanded) || ' ' || LOWER(icd10_names)
            END AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
        LATERAL FLATTEN(input => SPLIT(burden_group, '|')) bg
    ),
    segmented AS (
        SELECT DISTINCT
            visit_id, patient_id, visit_date, clean_dx_text,
            CASE
                WHEN clean_dx_text LIKE '%hernia%' THEN 'EXCLUDE: General Surgery'
                WHEN (clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%')
                     AND (clean_dx_text LIKE '%post %' OR clean_dx_text LIKE '%revision%')
                    THEN 'EXCLUDE: Fibroids-surgical'
                WHEN clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%'
                    THEN 'Fibroids-conservative'
                WHEN clean_dx_text LIKE '%spine%' OR clean_dx_text LIKE '%sciatica%' THEN
                    CASE
                        WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%dislocat%' OR clean_dx_text LIKE '%stenosis%'
                             OR clean_dx_text LIKE '%herniat%' OR clean_dx_text LIKE '%compression%' OR clean_dx_text LIKE '%tumour%'
                            THEN 'Spine-structural'
                        ELSE 'Spine-conservative'
                    END
                WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%osteoarthritis%' THEN 'Core Orthopedics: General'
                WHEN clean_dx_text LIKE '%pre-eclamp%' OR clean_dx_text LIKE '%preeclamp%' OR clean_dx_text LIKE '%eclamp%'
                     OR clean_dx_text LIKE '%hyperem%' OR clean_dx_text LIKE '%pprom%'
                    THEN 'High-Risk Pregnancy'
                WHEN clean_dx_text LIKE '%pregnan%' OR clean_dx_text LIKE '%anc%' OR clean_dx_text LIKE '%antenatal%'
                     OR clean_dx_text LIKE '%gravid%' OR split_burden = 'General: Obstetric'
                    THEN 'ANC / Routine Pregnancy'
                ELSE 'Other'
            END AS segment
        FROM raw_diag
    ),
    classifiable_all_time AS (
        SELECT * FROM segmented
        WHERE segment IN ('Core Orthopedics: General', 'Spine-structural', 'Spine-conservative')
    ),
    dataset_max_date AS (SELECT MAX(visit_date) AS max_date FROM classifiable_all_time),
    window_bounds AS (SELECT DATEADD('day', -365, max_date) AS window_start, max_date AS window_end FROM dataset_max_date),
    patients_active_in_window AS (
        SELECT DISTINCT ca.patient_id, ca.segment
        FROM classifiable_all_time ca CROSS JOIN window_bounds wb
        WHERE ca.visit_date BETWEEN wb.window_start AND wb.window_end
    ),
    relevant_visits AS (
        SELECT ca.* FROM classifiable_all_time ca
        JOIN patients_active_in_window paiw ON paiw.patient_id = ca.patient_id AND paiw.segment = ca.segment
    ),
    patient_segment_last_visit AS (
        SELECT patient_id, segment, MAX(visit_date) AS last_visit_date
        FROM relevant_visits GROUP BY patient_id, segment
    ),
    ltfu_patients AS (
        SELECT psl.patient_id, psl.segment, psl.last_visit_date
        FROM patient_segment_last_visit psl CROSS JOIN dataset_max_date dmd
        WHERE DATEDIFF('day', psl.last_visit_date, dmd.max_date) > 180
    ),
    ltfu_final_dx AS (
        SELECT lp.segment, rv.clean_dx_text, rv.patient_id
        FROM ltfu_patients lp
        JOIN relevant_visits rv ON rv.patient_id = lp.patient_id AND rv.segment = lp.segment AND rv.visit_date = lp.last_visit_date
        QUALIFY ROW_NUMBER() OVER (PARTITION BY lp.patient_id, lp.segment ORDER BY rv.visit_id) = 1
    ),
    categorized AS (
        SELECT
            segment, patient_id,
            CASE
                -- Spine segments are matched on spine-specific vocabulary first, so a
                -- combined diagnosis like "knee osteoarthritis + lumbar degenerative
                -- disease" doesn't get labeled "Knee Osteoarthritis" for a patient the
                -- segment step already classified as Spine-conservative/-structural
                -- (which itself requires 'spine' or 'sciatica' in the text). Only once
                -- no spine-specific label matches does it fall through to the shared
                -- fracture/osteoarthritis vocabulary below.
                WHEN segment IN ('Spine-conservative', 'Spine-structural') THEN
                    CASE
                        WHEN clean_dx_text LIKE '%scoliosis%' OR clean_dx_text LIKE '%kyphosis%'
                            THEN 'Spinal Deformity'
                        WHEN clean_dx_text LIKE '%decompression%' OR clean_dx_text LIKE '%fusion%'
                             OR clean_dx_text LIKE '%post spine surgery%' OR clean_dx_text LIKE '%discectomy%'
                             OR clean_dx_text LIKE '%laminectomy%'
                            THEN 'Post-Spine-Surgery Follow-up'
                        WHEN clean_dx_text LIKE '%disc bulge%' OR clean_dx_text LIKE '%disc degeneration%'
                             OR clean_dx_text LIKE '%prolapsed disc%' OR clean_dx_text LIKE '%nerve compression%'
                             OR clean_dx_text LIKE '%nerve root compression%' OR clean_dx_text LIKE '%stenosis%'
                            THEN 'Disc Bulge / Herniation with Nerve Compression'
                        WHEN clean_dx_text LIKE '%compression fracture%' OR clean_dx_text LIKE '%rami fracture%'
                             OR clean_dx_text LIKE '%edge fracture%'
                            THEN 'Spine/Pelvis Fracture'
                        WHEN clean_dx_text LIKE '%cervical%'
                            THEN 'Cervical Spine Pain'
                        WHEN clean_dx_text LIKE '%lumbago%' OR clean_dx_text LIKE '%sciatica%' OR clean_dx_text LIKE '%low back pain%'
                             OR clean_dx_text LIKE '%lbp%' OR clean_dx_text LIKE '%lumbar radiculopathy%'
                            THEN 'Lumbago / Sciatica / Low Back Pain'
                        ELSE 'Other / Unclassified'
                    END
                WHEN clean_dx_text LIKE '%osteoarthritis%knee%' OR clean_dx_text LIKE '%knee%osteoarthritis%'
                     OR clean_dx_text LIKE '%o.a knee%' OR clean_dx_text LIKE '%o.a. knee%'
                    THEN 'Knee Osteoarthritis'
                WHEN clean_dx_text LIKE '%osteoarthritis%hip%' OR clean_dx_text LIKE '%hip%osteoarthritis%'
                    THEN 'Hip Osteoarthritis'
                WHEN clean_dx_text LIKE '%soft tissue injury%r/o fracture%' OR clean_dx_text LIKE '%soft tissue disorder%'
                    THEN 'Soft Tissue Injury / Rule-Out Fracture'
                WHEN clean_dx_text LIKE '%post total hip replacement%' OR clean_dx_text LIKE '%post%hip%arthroplasty%'
                    THEN 'Post-Hip Replacement Follow-up'
                WHEN clean_dx_text LIKE '%post total knee replacement%' OR clean_dx_text LIKE '%post%knee%arthroplasty%'
                    THEN 'Post-Knee Replacement Follow-up'
                WHEN clean_dx_text LIKE '%post%ankle%orif%' OR clean_dx_text LIKE '%bimalleolar%'
                    THEN 'Post-Ankle ORIF Follow-up'
                WHEN clean_dx_text LIKE '%clavicle%'
                    THEN 'Clavicle Fracture'
                WHEN clean_dx_text LIKE '%humerus%'
                    THEN 'Humerus Fracture'
                WHEN clean_dx_text LIKE '%tibia%'
                    THEN 'Tibia Fracture'
                WHEN clean_dx_text LIKE '%neck of femur%' OR clean_dx_text LIKE '%nof fracture%' OR clean_dx_text LIKE '%nof #%'
                    THEN 'Neck of Femur (Hip) Fracture'
                WHEN clean_dx_text LIKE '%femur%'
                    THEN 'Femur Fracture'
                WHEN clean_dx_text LIKE '%shafts of both ulna and radius%' OR clean_dx_text LIKE '%radial ulna%' OR clean_dx_text LIKE '%r/u fracture%'
                    THEN 'Radius-Ulna Fracture'
                WHEN clean_dx_text LIKE '%radius%' OR clean_dx_text LIKE '%colles%'
                    THEN 'Radius Fracture'
                WHEN clean_dx_text LIKE '%malleol%'
                    THEN 'Malleolus Fracture (ankle, non-ORIF)'
                WHEN clean_dx_text LIKE '%metacarpal%' OR clean_dx_text LIKE '%metatarsal%'
                     OR clean_dx_text LIKE '%phalan%' OR clean_dx_text LIKE '%finger%' OR clean_dx_text LIKE '%toe%'
                    THEN 'Hand / Foot Fracture'
                WHEN clean_dx_text LIKE '%patella%'
                    THEN 'Patella Fracture'
                WHEN clean_dx_text LIKE '%malunion%' OR clean_dx_text LIKE '%non union%' OR clean_dx_text LIKE '%nonunion%'
                    THEN 'Malunion / Nonunion'
                WHEN clean_dx_text LIKE '%pelvic%' OR clean_dx_text LIKE '%pelvis%' OR clean_dx_text LIKE '%acetabul%'
                    THEN 'Pelvic / Acetabulum Fracture'
                WHEN clean_dx_text LIKE '% rib%' OR clean_dx_text LIKE '%rib fracture%' OR clean_dx_text LIKE '%rib,%'
                    THEN 'Rib Fracture'
                WHEN clean_dx_text LIKE '%mandib%' OR clean_dx_text LIKE '%nasal%' OR clean_dx_text LIKE '%skull%'
                     OR clean_dx_text LIKE '%facial bone%' OR clean_dx_text LIKE '%maxilla%' OR clean_dx_text LIKE '%zygoma%'
                    THEN 'Facial / Skull Fracture'
                WHEN clean_dx_text LIKE '%contusion%'
                    THEN 'Contusion'
                ELSE 'Other / Unclassified'
            END AS condition_category
        FROM ltfu_final_dx
    )
    SELECT
        segment,
        condition_category,
        COUNT(DISTINCT patient_id) AS distinct_ltfu_patients,
        ROUND(100.0 * COUNT(DISTINCT patient_id) / SUM(COUNT(DISTINCT patient_id)) OVER (PARTITION BY segment), 1) AS pct_within_segment
    FROM categorized
    GROUP BY segment, condition_category
    ORDER BY segment, distinct_ltfu_patients DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# A10 — who exactly are the "lost after visit 1" patients
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_fr_lost_after_visit1() -> pd.DataFrame:
    """Columns: SEGMENT, CONDITION_CATEGORY, AGE_GROUP, GENDER, TOTAL_PATIENTS."""
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
        SELECT oie_id, soi_id,
            MAX(CASE WHEN field_name = 'Schedule Follow Up' THEN TRY_TO_TIMESTAMP(field_value) END) AS scheduled_follow_up_date
        FROM doctor_raw GROUP BY oie_id, soi_id
    ),
    soi AS (SELECT "id" AS soi_id, "patient" AS patient_id FROM HOSPITALS.ORTHOPEDIC_CLEAN."SINGLEORDERITEMS"),
    orders_flat AS (
        SELECT oo.id AS order_id, f.VALUE::STRING AS item_id
        FROM HOSPITALS.ORTHOPEDIC_CLEAN."ORTHO_ORDERS" oo,
        LATERAL FLATTEN(INPUT => TRY_PARSE_JSON(oo.items)) f
        QUALIFY ROW_NUMBER() OVER (PARTITION BY f.VALUE::STRING ORDER BY oo.registered_at DESC) = 1
    ),
    raw_schedule_visits AS (
        SELECT of_.order_id AS visit_id
        FROM doctor_pivot dp
        JOIN soi s ON s.soi_id = dp.soi_id JOIN orders_flat of_ ON of_.item_id = dp.soi_id
        WHERE dp.scheduled_follow_up_date IS NOT NULL
    ),
    scheduled_signal AS (
        SELECT DISTINCT v.patient_id AS clean_patient_id
        FROM raw_schedule_visits rsv JOIN HOSPITALS.STAGING.STG_VISITS v ON v.visit_id = rsv.visit_id
    ),
    diag_flagged AS (
        SELECT
            patient_id, visit_id,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text,
            has_diabetes, has_hypertension, has_chronic_condition
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
    ),
    text_detector_signal AS (
        SELECT DISTINCT patient_id FROM diag_flagged
        WHERE clean_dx_text LIKE '%post %' OR clean_dx_text LIKE '%s/p %' OR clean_dx_text LIKE '%status post%'
           OR clean_dx_text LIKE '%follow up%' OR clean_dx_text LIKE '%followup%' OR clean_dx_text LIKE '%f/u %'
           OR clean_dx_text LIKE '%review%' OR clean_dx_text LIKE '%/52%' OR clean_dx_text LIKE '%/12%'
    ),
    chronic_monitoring_signal AS (
        SELECT DISTINCT patient_id FROM diag_flagged
        WHERE has_diabetes OR has_hypertension OR has_chronic_condition
           OR clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%'
           OR clean_dx_text LIKE '%pre-eclamp%' OR clean_dx_text LIKE '%preeclamp%' OR clean_dx_text LIKE '%eclamp%'
           OR clean_dx_text LIKE '%hyperem%' OR clean_dx_text LIKE '%pprom%'
    ),
    expected_to_return AS (
        SELECT patient_id FROM text_detector_signal
        UNION
        SELECT patient_id FROM chronic_monitoring_signal
        UNION
        SELECT clean_patient_id AS patient_id FROM scheduled_signal
    ),
    all_visits AS (
        SELECT
            visit_id, patient_id, diagnosis_created_at AS visit_date,
            TRIM(bg.value::STRING) AS split_burden,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
        LATERAL FLATTEN(input => SPLIT(burden_group, '|')) bg
    ),
    segmented AS (
        SELECT DISTINCT
            visit_id, patient_id, visit_date, clean_dx_text,
            CASE
                WHEN clean_dx_text LIKE '%hernia%' THEN 'EXCLUDE: General Surgery'
                WHEN (clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%')
                     AND (clean_dx_text LIKE '%post %' OR clean_dx_text LIKE '%revision%')
                    THEN 'EXCLUDE: Fibroids-surgical'
                WHEN clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%'
                    THEN 'Fibroids-conservative'
                WHEN clean_dx_text LIKE '%spine%' OR clean_dx_text LIKE '%sciatica%' THEN
                    CASE
                        WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%dislocat%' OR clean_dx_text LIKE '%stenosis%'
                             OR clean_dx_text LIKE '%herniat%' OR clean_dx_text LIKE '%compression%' OR clean_dx_text LIKE '%tumour%'
                            THEN 'Spine-structural'
                        ELSE 'Spine-conservative'
                    END
                WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%osteoarthritis%' THEN 'Core Orthopedics: General'
                WHEN clean_dx_text LIKE '%pre-eclamp%' OR clean_dx_text LIKE '%preeclamp%' OR clean_dx_text LIKE '%eclamp%'
                     OR clean_dx_text LIKE '%hyperem%' OR clean_dx_text LIKE '%pprom%'
                    THEN 'High-Risk Pregnancy'
                WHEN clean_dx_text LIKE '%pregnan%' OR clean_dx_text LIKE '%anc%' OR clean_dx_text LIKE '%antenatal%'
                     OR clean_dx_text LIKE '%gravid%' OR split_burden = 'General: Obstetric'
                    THEN 'ANC / Routine Pregnancy'
                ELSE 'Other'
            END AS segment
        FROM all_visits
    ),
    classifiable_all_time AS (
        SELECT * FROM segmented
        WHERE segment NOT IN ('EXCLUDE: General Surgery', 'EXCLUDE: Fibroids-surgical', 'Other')
    ),
    dataset_max_date AS (SELECT MAX(visit_date) AS max_date FROM classifiable_all_time),
    window_bounds AS (SELECT DATEADD('day', -365, max_date) AS window_start, max_date AS window_end FROM dataset_max_date),
    patients_active_in_window AS (
        SELECT DISTINCT ca.patient_id, ca.segment
        FROM classifiable_all_time ca CROSS JOIN window_bounds wb
        WHERE ca.visit_date BETWEEN wb.window_start AND wb.window_end
    ),
    relevant_visits AS (
        SELECT ca.* FROM classifiable_all_time ca
        JOIN patients_active_in_window paiw ON paiw.patient_id = ca.patient_id AND paiw.segment = ca.segment
    ),
    scoped_visits AS (
        SELECT rv.* FROM relevant_visits rv
        JOIN expected_to_return etr ON etr.patient_id = rv.patient_id
    ),
    patient_summary AS (
        SELECT
            patient_id, segment,
            COUNT(*) AS total_visits,
            MAX(visit_date) AS last_visit_date,
            MAX(visit_id) AS the_visit_id,
            MAX(clean_dx_text) AS the_dx_text
        FROM scoped_visits
        GROUP BY patient_id, segment
    ),
    first_visit_ltfu AS (
        SELECT ps.*
        FROM patient_summary ps
        CROSS JOIN dataset_max_date dmd
        WHERE ps.total_visits = 1
          AND DATEDIFF('day', ps.last_visit_date, dmd.max_date) > 180
    ),
    categorized AS (
        SELECT
            segment, patient_id, the_visit_id,
            CASE
                -- Spine segments are matched on spine-specific vocabulary first — see
                -- the identical fix and rationale in get_fr_ltfu_condition_breakdown().
                WHEN segment IN ('Spine-conservative', 'Spine-structural') THEN
                    CASE
                        WHEN the_dx_text LIKE '%decompression%' OR the_dx_text LIKE '%fusion%' OR the_dx_text LIKE '%post spine surgery%'
                            THEN 'Post-Spine-Surgery Follow-up'
                        WHEN the_dx_text LIKE '%disc bulge%' OR the_dx_text LIKE '%stenosis%'
                            THEN 'Disc Bulge / Herniation with Nerve Compression'
                        WHEN the_dx_text LIKE '%cervical%'
                            THEN 'Cervical Spine Pain'
                        WHEN the_dx_text LIKE '%lumbago%' OR the_dx_text LIKE '%sciatica%' OR the_dx_text LIKE '%low back pain%'
                            THEN 'Lumbago / Sciatica / Low Back Pain'
                        ELSE 'Other / Unclassified'
                    END
                WHEN the_dx_text LIKE '%osteoarthritis%knee%' OR the_dx_text LIKE '%knee%osteoarthritis%' THEN 'Knee Osteoarthritis'
                WHEN the_dx_text LIKE '%osteoarthritis%hip%' OR the_dx_text LIKE '%hip%osteoarthritis%' THEN 'Hip Osteoarthritis'
                WHEN the_dx_text LIKE '%soft tissue injury%r/o fracture%' OR the_dx_text LIKE '%soft tissue disorder%' THEN 'Soft Tissue Injury / Rule-Out Fracture'
                WHEN the_dx_text LIKE '%clavicle%' THEN 'Clavicle Fracture'
                WHEN the_dx_text LIKE '%humerus%' THEN 'Humerus Fracture'
                WHEN the_dx_text LIKE '%tibia%' THEN 'Tibia Fracture'
                WHEN the_dx_text LIKE '%neck of femur%' OR the_dx_text LIKE '%nof fracture%' THEN 'Neck of Femur (Hip) Fracture'
                WHEN the_dx_text LIKE '%femur%' THEN 'Femur Fracture'
                WHEN the_dx_text LIKE '%shafts of both ulna and radius%' OR the_dx_text LIKE '%radial ulna%' THEN 'Radius-Ulna Fracture'
                WHEN the_dx_text LIKE '%radius%' OR the_dx_text LIKE '%colles%' THEN 'Radius Fracture'
                WHEN the_dx_text LIKE '%malleol%' THEN 'Malleolus Fracture (ankle, non-ORIF)'
                WHEN the_dx_text LIKE '%metacarpal%' OR the_dx_text LIKE '%metatarsal%' OR the_dx_text LIKE '%phalan%' THEN 'Hand / Foot Fracture'
                WHEN the_dx_text LIKE '%patella%' THEN 'Patella Fracture'
                WHEN the_dx_text LIKE '%malunion%' OR the_dx_text LIKE '%nonunion%' THEN 'Malunion / Nonunion'
                WHEN the_dx_text LIKE '%pelvic%' OR the_dx_text LIKE '%acetabul%' THEN 'Pelvic / Acetabulum Fracture'
                WHEN the_dx_text LIKE '%contusion%' THEN 'Contusion'
                ELSE 'Other / Unclassified'
            END AS condition_category
        FROM first_visit_ltfu
    )
    SELECT
        c.segment,
        c.condition_category,
        age_group,
        CASE WHEN LOWER(v.gender) = 'f' THEN 'female' WHEN LOWER(v.gender) = 'm' THEN 'male' ELSE LOWER(v.gender) END AS gender,
        COUNT(*) AS total_patients
    FROM categorized c
    LEFT JOIN HOSPITALS.STAGING.STG_VISITS v ON v.visit_id = c.the_visit_id
    GROUP BY c.segment, c.condition_category, age_group, gender
    ORDER BY c.segment, total_patients DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# A10 Verification — named Spine-structural post-surgery first-visit LTFU list
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_fr_spine_structural_outreach_list() -> pd.DataFrame:
    """Columns: PATIENT_ID, LAST_VISIT_DATE, DAYS_SINCE_VISIT, DIAGNOSIS_TEXT,
    AGE_GROUP, GENDER. One row per patient — the manually-verifiable
    outreach list (build spec Section 5 / Pattern D)."""
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
        SELECT oie_id, soi_id,
            MAX(CASE WHEN field_name = 'Schedule Follow Up' THEN TRY_TO_TIMESTAMP(field_value) END) AS scheduled_follow_up_date
        FROM doctor_raw GROUP BY oie_id, soi_id
    ),
    soi AS (SELECT "id" AS soi_id, "patient" AS patient_id FROM HOSPITALS.ORTHOPEDIC_CLEAN."SINGLEORDERITEMS"),
    orders_flat AS (
        SELECT oo.id AS order_id, f.VALUE::STRING AS item_id
        FROM HOSPITALS.ORTHOPEDIC_CLEAN."ORTHO_ORDERS" oo,
        LATERAL FLATTEN(INPUT => TRY_PARSE_JSON(oo.items)) f
        QUALIFY ROW_NUMBER() OVER (PARTITION BY f.VALUE::STRING ORDER BY oo.registered_at DESC) = 1
    ),
    raw_schedule_visits AS (
        SELECT of_.order_id AS visit_id
        FROM doctor_pivot dp
        JOIN soi s ON s.soi_id = dp.soi_id JOIN orders_flat of_ ON of_.item_id = dp.soi_id
        WHERE dp.scheduled_follow_up_date IS NOT NULL
    ),
    scheduled_signal AS (
        SELECT DISTINCT v.patient_id AS clean_patient_id
        FROM raw_schedule_visits rsv JOIN HOSPITALS.STAGING.STG_VISITS v ON v.visit_id = rsv.visit_id
    ),
    diag_flagged AS (
        SELECT
            patient_id,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text,
            has_diabetes, has_hypertension, has_chronic_condition
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
    ),
    text_detector_signal AS (
        SELECT DISTINCT patient_id FROM diag_flagged
        WHERE clean_dx_text LIKE '%post %' OR clean_dx_text LIKE '%s/p %' OR clean_dx_text LIKE '%status post%'
           OR clean_dx_text LIKE '%follow up%' OR clean_dx_text LIKE '%followup%' OR clean_dx_text LIKE '%f/u %'
           OR clean_dx_text LIKE '%review%' OR clean_dx_text LIKE '%/52%' OR clean_dx_text LIKE '%/12%'
    ),
    chronic_monitoring_signal AS (
        SELECT DISTINCT patient_id FROM diag_flagged
        WHERE has_diabetes OR has_hypertension OR has_chronic_condition
           OR clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%'
           OR clean_dx_text LIKE '%pre-eclamp%' OR clean_dx_text LIKE '%preeclamp%' OR clean_dx_text LIKE '%eclamp%'
           OR clean_dx_text LIKE '%hyperem%' OR clean_dx_text LIKE '%pprom%'
    ),
    expected_to_return AS (
        SELECT patient_id FROM text_detector_signal
        UNION
        SELECT patient_id FROM chronic_monitoring_signal
        UNION
        SELECT clean_patient_id AS patient_id FROM scheduled_signal
    ),
    all_visits AS (
        SELECT
            visit_id, patient_id, diagnosis_created_at AS visit_date,
            TRIM(bg.value::STRING) AS split_burden,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
        LATERAL FLATTEN(input => SPLIT(burden_group, '|')) bg
    ),
    segmented AS (
        SELECT DISTINCT
            visit_id, patient_id, visit_date, clean_dx_text,
            CASE
                WHEN clean_dx_text LIKE '%hernia%' THEN 'EXCLUDE: General Surgery'
                WHEN (clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%')
                     AND (clean_dx_text LIKE '%post %' OR clean_dx_text LIKE '%revision%')
                    THEN 'EXCLUDE: Fibroids-surgical'
                WHEN clean_dx_text LIKE '%fibroid%' OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%'
                    THEN 'Fibroids-conservative'
                WHEN clean_dx_text LIKE '%spine%' OR clean_dx_text LIKE '%sciatica%' THEN
                    CASE
                        WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%dislocat%' OR clean_dx_text LIKE '%stenosis%'
                             OR clean_dx_text LIKE '%herniat%' OR clean_dx_text LIKE '%compression%' OR clean_dx_text LIKE '%tumour%'
                            THEN 'Spine-structural'
                        ELSE 'Spine-conservative'
                    END
                WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%osteoarthritis%' THEN 'Core Orthopedics: General'
                WHEN clean_dx_text LIKE '%pre-eclamp%' OR clean_dx_text LIKE '%preeclamp%' OR clean_dx_text LIKE '%eclamp%'
                     OR clean_dx_text LIKE '%hyperem%' OR clean_dx_text LIKE '%pprom%'
                    THEN 'High-Risk Pregnancy'
                WHEN clean_dx_text LIKE '%pregnan%' OR clean_dx_text LIKE '%anc%' OR clean_dx_text LIKE '%antenatal%'
                     OR clean_dx_text LIKE '%gravid%' OR split_burden = 'General: Obstetric'
                    THEN 'ANC / Routine Pregnancy'
                ELSE 'Other'
            END AS segment
        FROM all_visits
    ),
    classifiable_all_time AS (
        SELECT * FROM segmented
        WHERE segment = 'Spine-structural'
    ),
    dataset_max_date AS (SELECT MAX(visit_date) AS max_date FROM (SELECT visit_date FROM segmented WHERE segment NOT IN ('EXCLUDE: General Surgery', 'EXCLUDE: Fibroids-surgical', 'Other'))),
    window_bounds AS (SELECT DATEADD('day', -365, max_date) AS window_start, max_date AS window_end FROM dataset_max_date),
    patients_active_in_window AS (
        SELECT DISTINCT ca.patient_id
        FROM classifiable_all_time ca CROSS JOIN window_bounds wb
        WHERE ca.visit_date BETWEEN wb.window_start AND wb.window_end
    ),
    relevant_visits AS (
        SELECT ca.* FROM classifiable_all_time ca
        JOIN patients_active_in_window paiw ON paiw.patient_id = ca.patient_id
    ),
    scoped_visits AS (
        SELECT rv.* FROM relevant_visits rv
        JOIN expected_to_return etr ON etr.patient_id = rv.patient_id
    ),
    patient_summary AS (
        SELECT
            patient_id,
            COUNT(*) AS total_visits,
            MAX(visit_date) AS last_visit_date,
            MAX(visit_id) AS the_visit_id,
            MAX(clean_dx_text) AS the_dx_text
        FROM scoped_visits
        GROUP BY patient_id
    ),
    first_visit_ltfu AS (
        SELECT ps.*
        FROM patient_summary ps
        CROSS JOIN dataset_max_date dmd
        WHERE ps.total_visits = 1
          AND DATEDIFF('day', ps.last_visit_date, dmd.max_date) > 180
          AND (ps.the_dx_text LIKE '%decompression%' OR ps.the_dx_text LIKE '%fusion%' OR ps.the_dx_text LIKE '%post spine surgery%')
    )
    SELECT
        fvl.patient_id,
        fvl.last_visit_date,
        DATEDIFF('day', fvl.last_visit_date, dmd.max_date) AS days_since_visit,
        fvl.the_dx_text AS diagnosis_text,
        CASE
            WHEN v.age_at_visit IS NULL THEN 'Unknown'
            WHEN v.age_at_visit < 18 THEN 'Under 18'
            WHEN v.age_at_visit < 35 THEN '18-34'
            WHEN v.age_at_visit < 55 THEN '35-54'
            WHEN v.age_at_visit < 65 THEN '55-64'
            ELSE '65+'
        END AS age_group,
        CASE WHEN LOWER(v.gender) = 'f' THEN 'female' WHEN LOWER(v.gender) = 'm' THEN 'male' ELSE LOWER(v.gender) END AS gender
    FROM first_visit_ltfu fvl
    CROSS JOIN dataset_max_date dmd
    LEFT JOIN HOSPITALS.STAGING.STG_VISITS v ON v.visit_id = fvl.the_visit_id
    ORDER BY days_since_visit DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# SECTION B — Q1: scheduled vs. non-scheduled repeat visits, per segment
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_fr_scheduled_vs_organic() -> pd.DataFrame:
    """Columns: SEGMENT, TOTAL_RETURN_VISITS, MATCHED_A_SCHEDULE, PCT_SCHEDULED,
    ORGANIC_UNSCHEDULED_RETURNS, PCT_ORGANIC. 4-segment Section B taxonomy
    (Core Orthopedics: Spine and Back Pain Care, not split conservative/structural)."""
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
        SELECT oo.id AS order_id, f.VALUE::STRING AS item_id
        FROM HOSPITALS.ORTHOPEDIC_CLEAN."ORTHO_ORDERS" oo,
        LATERAL FLATTEN(INPUT => TRY_PARSE_JSON(oo.items)) f
        QUALIFY ROW_NUMBER() OVER (PARTITION BY f.VALUE::STRING ORDER BY oo.registered_at DESC) = 1
    ),
    raw_schedules AS (
        SELECT of_.order_id AS visit_id, s.patient_id AS raw_patient_id, dp.scheduled_follow_up_date
        FROM doctor_pivot dp
        JOIN soi s ON s.soi_id = dp.soi_id
        JOIN orders_flat of_ ON of_.item_id = dp.soi_id
        WHERE dp.scheduled_follow_up_date IS NOT NULL
    ),
    all_schedules AS (
        SELECT v.patient_id AS clean_patient_id, rs.scheduled_follow_up_date
        FROM raw_schedules rs
        JOIN HOSPITALS.STAGING.STG_VISITS v ON v.visit_id = rs.visit_id
    ),
    all_visits AS (
        SELECT
            visit_id, patient_id, diagnosis_created_at AS visit_date,
            TRIM(bg.value::STRING) AS split_burden,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
        LATERAL FLATTEN(input => SPLIT(burden_group, '|')) bg
    ),
    patient_visits AS (
        SELECT DISTINCT
            visit_id, patient_id, visit_date,
            CASE
                WHEN clean_dx_text LIKE '%spine%' OR clean_dx_text LIKE '%sciatica%' THEN 'Core Orthopedics: Spine and Back Pain Care'
                WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%osteoarthritis%' THEN 'Core Orthopedics: General'
                WHEN clean_dx_text LIKE '%hernia%' THEN 'Core General Surgery'
                WHEN clean_dx_text LIKE '%pregnan%' OR clean_dx_text LIKE '%anc%' OR clean_dx_text LIKE '%antenatal%'
                     OR clean_dx_text LIKE '%gravid%' OR split_burden = 'General: Obstetric'
                    THEN 'ANC'
                WHEN split_burden = 'General: Gynaecology' OR clean_dx_text LIKE '%fibroid%'
                     OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%'
                    THEN 'Maternal Health'
                ELSE 'Other'
            END AS segment
        FROM all_visits
    ),
    visit_sequence AS (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY patient_id ORDER BY visit_date) AS visit_number
        FROM patient_visits
    ),
    returning_patients AS (
        SELECT patient_id, segment, visit_date AS return_visit_date, visit_id
        FROM visit_sequence
        WHERE visit_number > 1
    ),
    returns_matched AS (
        SELECT
            rp.*,
            MAX(CASE WHEN ABS(DATEDIFF('day', asch.scheduled_follow_up_date, rp.return_visit_date)) <= 7
                     THEN 1 ELSE 0 END) AS matches_a_schedule
        FROM returning_patients rp
        LEFT JOIN all_schedules asch ON asch.clean_patient_id = rp.patient_id
        GROUP BY rp.patient_id, rp.segment, rp.return_visit_date, rp.visit_id
    )
    SELECT
        segment,
        COUNT(*) AS total_return_visits,
        SUM(matches_a_schedule) AS matched_a_schedule,
        ROUND(100.0 * SUM(matches_a_schedule) / COUNT(*), 1) AS pct_scheduled,
        COUNT(*) - SUM(matches_a_schedule) AS organic_unscheduled_returns,
        ROUND(100.0 * (COUNT(*) - SUM(matches_a_schedule)) / COUNT(*), 1) AS pct_organic
    FROM returns_matched
    WHERE segment IS NOT NULL AND segment <> 'Other'
    GROUP BY segment
    ORDER BY total_return_visits DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# SECTION B — Q2: non-scheduled returns — condition, actual window, age_group
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_fr_nonscheduled_returns() -> pd.DataFrame:
    """Columns: SEGMENT, AGE_GROUP, TOTAL_NON_SCHEDULED_RETURNS, AVG_ACTUAL_GAP_DAYS,
    SEGMENT_TYPICAL_GAP_DAYS, WITHIN_EXPECTED_WINDOW, PCT_WITHIN_EXPECTED_WINDOW.
    HAVING >= 5."""
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
        SELECT oo.id AS order_id, f.VALUE::STRING AS item_id
        FROM HOSPITALS.ORTHOPEDIC_CLEAN."ORTHO_ORDERS" oo,
        LATERAL FLATTEN(INPUT => TRY_PARSE_JSON(oo.items)) f
        QUALIFY ROW_NUMBER() OVER (PARTITION BY f.VALUE::STRING ORDER BY oo.registered_at DESC) = 1
    ),
    raw_schedules AS (
        SELECT of_.order_id AS visit_id, s.patient_id AS raw_patient_id, dp.scheduled_follow_up_date
        FROM doctor_pivot dp
        JOIN soi s ON s.soi_id = dp.soi_id
        JOIN orders_flat of_ ON of_.item_id = dp.soi_id
        WHERE dp.scheduled_follow_up_date IS NOT NULL
    ),
    all_schedules AS (
        SELECT v.patient_id AS clean_patient_id, rs.scheduled_follow_up_date
        FROM raw_schedules rs
        JOIN HOSPITALS.STAGING.STG_VISITS v ON v.visit_id = rs.visit_id
    ),
    all_visits AS (
        SELECT
            visit_id, patient_id, diagnosis_created_at AS visit_date,
            TRIM(bg.value::STRING) AS split_burden,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
        LATERAL FLATTEN(input => SPLIT(burden_group, '|')) bg
    ),
    patient_visits AS (
        SELECT DISTINCT
            visit_id, patient_id, visit_date,
            CASE
                WHEN clean_dx_text LIKE '%spine%' OR clean_dx_text LIKE '%sciatica%' THEN 'Core Orthopedics: Spine and Back Pain Care'
                WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%osteoarthritis%' THEN 'Core Orthopedics: General'
                WHEN clean_dx_text LIKE '%hernia%' THEN 'Core General Surgery'
                WHEN clean_dx_text LIKE '%pregnan%' OR clean_dx_text LIKE '%anc%' OR clean_dx_text LIKE '%antenatal%'
                     OR clean_dx_text LIKE '%gravid%' OR split_burden = 'General: Obstetric'
                    THEN 'ANC'
                WHEN split_burden = 'General: Gynaecology' OR clean_dx_text LIKE '%fibroid%'
                     OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%'
                    THEN 'Maternal Health'
                ELSE 'Other'
            END AS segment
        FROM all_visits
    ),
    visit_sequence AS (
        SELECT
            *,
            LAG(visit_date) OVER (PARTITION BY patient_id ORDER BY visit_date) AS prior_visit_date,
            ROW_NUMBER() OVER (PARTITION BY patient_id ORDER BY visit_date) AS visit_number
        FROM patient_visits
    ),
    returning_patients AS (
        SELECT visit_id, patient_id, segment, visit_date AS return_visit_date, prior_visit_date
        FROM visit_sequence
        WHERE visit_number > 1
    ),
    returns_matched AS (
        SELECT
            rp.*,
            MAX(CASE WHEN ABS(DATEDIFF('day', asch.scheduled_follow_up_date, rp.return_visit_date)) <= 7
                     THEN 1 ELSE 0 END) AS matches_a_schedule
        FROM returning_patients rp
        LEFT JOIN all_schedules asch ON asch.clean_patient_id = rp.patient_id
        GROUP BY rp.visit_id, rp.patient_id, rp.segment, rp.return_visit_date, rp.prior_visit_date
    ),
    segment_typical_gap AS (
        SELECT segment, MEDIAN(DATEDIFF('day', prior_visit_date, return_visit_date)) AS typical_gap_days
        FROM returns_matched
        WHERE segment <> 'Other'
        GROUP BY segment
    ),
    non_scheduled AS (
        SELECT
            rm.*,
            DATEDIFF('day', rm.prior_visit_date, rm.return_visit_date) AS actual_gap_days,
            stg.typical_gap_days
        FROM returns_matched rm
        JOIN segment_typical_gap stg ON stg.segment = rm.segment
        WHERE rm.matches_a_schedule = 0 AND rm.segment <> 'Other'
    )
    SELECT
        ns.segment,
        age_group,
        COUNT(*) AS total_non_scheduled_returns,
        ROUND(AVG(ns.actual_gap_days), 1) AS avg_actual_gap_days,
        ROUND(AVG(ns.typical_gap_days), 1) AS segment_typical_gap_days,
        SUM(CASE WHEN ABS(ns.actual_gap_days - ns.typical_gap_days) <= 14 THEN 1 ELSE 0 END) AS within_expected_window,
        ROUND(100.0 * SUM(CASE WHEN ABS(ns.actual_gap_days - ns.typical_gap_days) <= 14 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_within_expected_window
    FROM non_scheduled ns
    LEFT JOIN HOSPITALS.STAGING.STG_VISITS v ON v.visit_id = ns.visit_id
    GROUP BY ns.segment, age_group
    HAVING COUNT(*) >= 5
    ORDER BY ns.segment, total_non_scheduled_returns DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# SECTION B — Q2 follow-up: Spine's Unknown-age-group conditions driving the gap
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_fr_spine_unknown_age_gap_conditions() -> pd.DataFrame:
    """Columns: DIAGNOSIS_TEXT, TOTAL_PATIENTS, AVG_GAP_DAYS, MIN_GAP_DAYS, MAX_GAP_DAYS."""
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
        SELECT oo.id AS order_id, f.VALUE::STRING AS item_id
        FROM HOSPITALS.ORTHOPEDIC_CLEAN."ORTHO_ORDERS" oo,
        LATERAL FLATTEN(INPUT => TRY_PARSE_JSON(oo.items)) f
        QUALIFY ROW_NUMBER() OVER (PARTITION BY f.VALUE::STRING ORDER BY oo.registered_at DESC) = 1
    ),
    raw_schedules AS (
        SELECT of_.order_id AS visit_id, s.patient_id AS raw_patient_id, dp.scheduled_follow_up_date
        FROM doctor_pivot dp
        JOIN soi s ON s.soi_id = dp.soi_id
        JOIN orders_flat of_ ON of_.item_id = dp.soi_id
        WHERE dp.scheduled_follow_up_date IS NOT NULL
    ),
    all_schedules AS (
        SELECT v.patient_id AS clean_patient_id, rs.scheduled_follow_up_date
        FROM raw_schedules rs
        JOIN HOSPITALS.STAGING.STG_VISITS v ON v.visit_id = rs.visit_id
    ),
    all_visits AS (
        SELECT
            visit_id, patient_id, diagnosis_created_at AS visit_date,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
    ),
    spine_visits AS (
        SELECT DISTINCT visit_id, patient_id, visit_date, clean_dx_text
        FROM all_visits
        WHERE clean_dx_text LIKE '%spine%' OR clean_dx_text LIKE '%sciatica%'
    ),
    visit_sequence AS (
        SELECT
            *,
            LAG(visit_date) OVER (PARTITION BY patient_id ORDER BY visit_date) AS prior_visit_date,
            ROW_NUMBER() OVER (PARTITION BY patient_id ORDER BY visit_date) AS visit_number
        FROM spine_visits
    ),
    returning_patients AS (
        SELECT visit_id, patient_id, visit_date AS return_visit_date, prior_visit_date, clean_dx_text
        FROM visit_sequence
        WHERE visit_number > 1
    ),
    returns_matched AS (
        SELECT
            rp.*,
            MAX(CASE WHEN ABS(DATEDIFF('day', asch.scheduled_follow_up_date, rp.return_visit_date)) <= 7
                     THEN 1 ELSE 0 END) AS matches_a_schedule
        FROM returning_patients rp
        LEFT JOIN all_schedules asch ON asch.clean_patient_id = rp.patient_id
        GROUP BY rp.visit_id, rp.patient_id, rp.return_visit_date, rp.prior_visit_date, rp.clean_dx_text
    ),
    unknown_age_non_scheduled AS (
        SELECT rm.*
        FROM returns_matched rm
        LEFT JOIN HOSPITALS.STAGING.STG_VISITS v ON v.visit_id = rm.visit_id
        WHERE rm.matches_a_schedule = 0 AND v.age_at_visit IS NULL
    )
    SELECT
        clean_dx_text AS diagnosis_text,
        COUNT(*) AS total_patients,
        ROUND(AVG(DATEDIFF('day', prior_visit_date, return_visit_date)), 1) AS avg_gap_days,
        MIN(DATEDIFF('day', prior_visit_date, return_visit_date)) AS min_gap_days,
        MAX(DATEDIFF('day', prior_visit_date, return_visit_date)) AS max_gap_days
    FROM unknown_age_non_scheduled
    GROUP BY diagnosis_text
    ORDER BY total_patients DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# SECTION B — Q3: scheduled returns — condition, age_group
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_fr_scheduled_returns_by_age() -> pd.DataFrame:
    """Columns: SEGMENT, AGE_GROUP, GENDER, TOTAL_SCHEDULED_RETURNS. HAVING >= 5."""
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
        SELECT oo.id AS order_id, f.VALUE::STRING AS item_id
        FROM HOSPITALS.ORTHOPEDIC_CLEAN."ORTHO_ORDERS" oo,
        LATERAL FLATTEN(INPUT => TRY_PARSE_JSON(oo.items)) f
        QUALIFY ROW_NUMBER() OVER (PARTITION BY f.VALUE::STRING ORDER BY oo.registered_at DESC) = 1
    ),
    raw_schedules AS (
        SELECT of_.order_id AS visit_id, s.patient_id AS raw_patient_id, dp.scheduled_follow_up_date
        FROM doctor_pivot dp
        JOIN soi s ON s.soi_id = dp.soi_id
        JOIN orders_flat of_ ON of_.item_id = dp.soi_id
        WHERE dp.scheduled_follow_up_date IS NOT NULL
    ),
    all_schedules AS (
        SELECT v.patient_id AS clean_patient_id, rs.scheduled_follow_up_date
        FROM raw_schedules rs
        JOIN HOSPITALS.STAGING.STG_VISITS v ON v.visit_id = rs.visit_id
    ),
    all_visits AS (
        SELECT
            visit_id, patient_id, diagnosis_created_at AS visit_date,
            TRIM(bg.value::STRING) AS split_burden,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
        LATERAL FLATTEN(input => SPLIT(burden_group, '|')) bg
    ),
    patient_visits AS (
        SELECT DISTINCT
            visit_id, patient_id, visit_date,
            CASE
                WHEN clean_dx_text LIKE '%spine%' OR clean_dx_text LIKE '%sciatica%' THEN 'Core Orthopedics: Spine and Back Pain Care'
                WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%osteoarthritis%' THEN 'Core Orthopedics: General'
                WHEN clean_dx_text LIKE '%hernia%' THEN 'Core General Surgery'
                WHEN clean_dx_text LIKE '%pregnan%' OR clean_dx_text LIKE '%anc%' OR clean_dx_text LIKE '%antenatal%'
                     OR clean_dx_text LIKE '%gravid%' OR split_burden = 'General: Obstetric'
                    THEN 'ANC'
                WHEN split_burden = 'General: Gynaecology' OR clean_dx_text LIKE '%fibroid%'
                     OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%'
                    THEN 'Maternal Health'
                ELSE 'Other'
            END AS segment
        FROM all_visits
    ),
    visit_sequence AS (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY patient_id ORDER BY visit_date) AS visit_number
        FROM patient_visits
    ),
    returning_patients AS (
        SELECT visit_id, patient_id, segment, visit_date AS return_visit_date
        FROM visit_sequence
        WHERE visit_number > 1
    ),
    returns_matched AS (
        SELECT
            rp.*,
            MAX(CASE WHEN ABS(DATEDIFF('day', asch.scheduled_follow_up_date, rp.return_visit_date)) <= 7
                     THEN 1 ELSE 0 END) AS matches_a_schedule
        FROM returning_patients rp
        LEFT JOIN all_schedules asch ON asch.clean_patient_id = rp.patient_id
        GROUP BY rp.visit_id, rp.patient_id, rp.segment, rp.return_visit_date
    )
    SELECT
        rm.segment,
        age_group,
        CASE WHEN LOWER(v.gender) = 'f' THEN 'female' WHEN LOWER(v.gender) = 'm' THEN 'male' ELSE LOWER(v.gender) END AS gender,
        COUNT(*) AS total_scheduled_returns
    FROM returns_matched rm
    LEFT JOIN HOSPITALS.STAGING.STG_VISITS v ON v.visit_id = rm.visit_id
    WHERE rm.matches_a_schedule = 1 AND rm.segment <> 'Other'
    GROUP BY rm.segment, age_group, gender
    HAVING COUNT(*) >= 5
    ORDER BY rm.segment, total_scheduled_returns DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# SECTION B — Q4: clinician patterns — scheduled vs. non-scheduled
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_fr_clinician_scheduling_rate() -> pd.DataFrame:
    """Columns: FILLED_BY_USER_ID, TOTAL_CONSULTATIONS, WITH_SCHEDULED_FOLLOW_UP,
    PCT_SCHEDULED. HAVING total_consultations >= 20."""
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
        SELECT oo.id AS order_id, f.VALUE::STRING AS item_id
        FROM HOSPITALS.ORTHOPEDIC_CLEAN."ORTHO_ORDERS" oo,
        LATERAL FLATTEN(INPUT => TRY_PARSE_JSON(oo.items)) f
        QUALIFY ROW_NUMBER() OVER (PARTITION BY f.VALUE::STRING ORDER BY oo.registered_at DESC) = 1
    ),
    consultations AS (
        SELECT of_.order_id AS visit_id, dp.scheduled_follow_up_date
        FROM doctor_pivot dp
        JOIN soi s ON s.soi_id = dp.soi_id
        JOIN orders_flat of_ ON of_.item_id = dp.soi_id
    ),
    clinician_link AS (
        SELECT DISTINCT d.visit_id, v.filled_by_user_id
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED d
        LEFT JOIN HOSPITALS.STAGING.STG_SPH_DIAGNOSIS v
            ON d.visit_id = v.visit_id AND d.source_system = v.source_system
        QUALIFY ROW_NUMBER() OVER (PARTITION BY d.visit_id ORDER BY v.filled_by_user_id) = 1
    )
    SELECT
        cl.filled_by_user_id,
        COUNT(DISTINCT c.visit_id) AS total_consultations,
        COUNT(DISTINCT CASE WHEN c.scheduled_follow_up_date IS NOT NULL THEN c.visit_id END) AS with_scheduled_follow_up,
        ROUND(100.0 * COUNT(DISTINCT CASE WHEN c.scheduled_follow_up_date IS NOT NULL THEN c.visit_id END)
                   / COUNT(DISTINCT c.visit_id), 1) AS pct_scheduled
    FROM consultations c
    LEFT JOIN clinician_link cl ON cl.visit_id = c.visit_id
    WHERE cl.filled_by_user_id IS NOT NULL
    GROUP BY cl.filled_by_user_id
    HAVING COUNT(DISTINCT c.visit_id) >= 20
    ORDER BY total_consultations DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# SECTION B — Q5 rewrite: attendance outcome, by segment/condition/age
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_fr_attendance_outcome() -> pd.DataFrame:
    """Columns: PRIMARY_VISIT_SEGMENT, CONDITION_GROUP, ATTENDANCE_OUTCOME,
    TOTAL_PATIENTS. HAVING >= 3. Only NEVER_RETURN-style never-return-rate
    reduction is confirmed at build-time — this raw query also carries the
    EARLY / ON-TIME-OR-MILD-LATE / 30+-DAYS-LATE breakdown, per the Q5
    rewrite note that fixed the original dead-code bug."""
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
    diag_full AS (
        SELECT
            visit_id, patient_id,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text,
            COALESCE(NULLIF(TRIM(diagnosis_name_expanded), ''), NULLIF(TRIM(icd10_names), ''), 'Unspecified')
                AS raw_dx_label,
            TRIM(bg.value::STRING) AS split_burden
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
        LATERAL FLATTEN(input => SPLIT(burden_group, '|')) bg
    ),
    segmented_visits AS (
        SELECT DISTINCT
            visit_id, clean_dx_text, raw_dx_label,
            CASE
                WHEN clean_dx_text LIKE '%spine%' OR clean_dx_text LIKE '%sciatica%' THEN 'Core Orthopedics: Spine and Back Pain Care'
                WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%osteoarthritis%' THEN 'Core Orthopedics: General'
                WHEN clean_dx_text LIKE '%hernia%' THEN 'Core General Surgery'
                WHEN clean_dx_text LIKE '%pregnan%' OR clean_dx_text LIKE '%anc%' OR clean_dx_text LIKE '%antenatal%'
                     OR clean_dx_text LIKE '%gravid%' OR split_burden = 'General: Obstetric'
                    THEN 'ANC'
                WHEN split_burden = 'General: Gynaecology' OR clean_dx_text LIKE '%fibroid%'
                     OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%'
                    THEN 'Maternal Health'
                ELSE 'Other'
            END AS primary_visit_segment
        FROM diag_full
    ),
    with_return_status AS (
        SELECT
            c.visit_id, c.patient_id, c.consult_date AS index_visit_date,
            c.scheduled_follow_up_date, sv.primary_visit_segment, sv.clean_dx_text, sv.raw_dx_label,
            MIN(c2.consult_date) AS next_consult_date
        FROM consultations c
        LEFT JOIN segmented_visits sv ON sv.visit_id = c.visit_id
        LEFT JOIN consultations c2
            ON c2.patient_id = c.patient_id AND c2.consult_date > c.consult_date
        WHERE c.scheduled_follow_up_date IS NOT NULL
        GROUP BY c.visit_id, c.patient_id, c.consult_date, c.scheduled_follow_up_date, sv.primary_visit_segment,
                 sv.clean_dx_text, sv.raw_dx_label
    ),
    classified AS (
        SELECT
            wrs.*,
            DATEDIFF('day', wrs.scheduled_follow_up_date, wrs.next_consult_date) AS days_vs_scheduled,
            CASE
                WHEN wrs.next_consult_date IS NULL THEN 'Never returned'
                WHEN wrs.next_consult_date < wrs.scheduled_follow_up_date THEN 'Showed EARLY'
                WHEN DATEDIFF('day', wrs.scheduled_follow_up_date, wrs.next_consult_date) <= 30 THEN 'Showed ON TIME / mildly LATE (within 30 days)'
                ELSE 'Returned, but well beyond scheduled date (30+ days late)'
            END AS attendance_outcome,
            -- Ortho-specific procedure names are only assigned when the visit
            -- did NOT already segment as Spine and Back Pain Care — a visit's
            -- diagnosis text can incidentally contain both an ortho-procedure
            -- keyword and "spine" (e.g. concatenated ICD descriptions), and
            -- without this guard the same condition name showed up under two
            -- different segments, which isn't a real dual-segment population.
            -- Segment-gated conditions fall through to the spine-specific
            -- branches or the raw diagnosis text, never to an Ortho label.
            CASE
                WHEN wrs.primary_visit_segment <> 'Core Orthopedics: Spine and Back Pain Care'
                     AND (wrs.clean_dx_text LIKE '%total knee replacement%' OR wrs.clean_dx_text LIKE '%tkr%')
                    THEN 'Knee Replacement'
                WHEN wrs.primary_visit_segment <> 'Core Orthopedics: Spine and Back Pain Care'
                     AND (wrs.clean_dx_text LIKE '%total hip replacement%' OR wrs.clean_dx_text LIKE '%thr%')
                    THEN 'Hip Replacement'
                WHEN wrs.primary_visit_segment <> 'Core Orthopedics: Spine and Back Pain Care'
                     AND (wrs.clean_dx_text LIKE '%ankle open reduction internal fixation%' OR wrs.clean_dx_text LIKE '%bimalleolar%')
                    THEN 'Ankle ORIF'
                WHEN wrs.primary_visit_segment <> 'Core Orthopedics: Spine and Back Pain Care'
                     AND (wrs.clean_dx_text LIKE '%nailing%' OR wrs.clean_dx_text LIKE '%plating%'
                          OR wrs.clean_dx_text LIKE '%open reduction internal fixation%')
                    THEN 'Long Bone Fracture Fixation (nailing/plating/ORIF)'
                WHEN wrs.clean_dx_text LIKE '%fusion%' OR wrs.clean_dx_text LIKE '%decompression%' THEN 'Post-Spine-Surgery Follow-up'
                WHEN wrs.clean_dx_text LIKE '%cervical%' THEN 'Cervical Spine Pain'
                WHEN wrs.clean_dx_text LIKE '%disc bulge%' OR wrs.clean_dx_text LIKE '%disc degeneration%'
                     OR wrs.clean_dx_text LIKE '%stenosis%' OR wrs.clean_dx_text LIKE '%nerve compression%'
                    THEN 'Disc Bulge / Herniation with Nerve Compression'
                WHEN wrs.clean_dx_text LIKE '%sciatica%' OR wrs.clean_dx_text LIKE '%lumbago%'
                     OR wrs.clean_dx_text LIKE '%low back pain%'
                    THEN 'Lumbago / Sciatica / Low Back Pain'
                WHEN wrs.primary_visit_segment <> 'Core Orthopedics: Spine and Back Pain Care'
                     AND wrs.clean_dx_text LIKE '%osteoarthritis%'
                    THEN 'Osteoarthritis (non-surgical)'
                WHEN wrs.primary_visit_segment <> 'Core Orthopedics: Spine and Back Pain Care'
                     AND wrs.clean_dx_text LIKE '%fracture%'
                    THEN 'Fracture -- Other / Various'
                WHEN wrs.clean_dx_text LIKE '%fibroid%' OR wrs.clean_dx_text LIKE '%myoma%' THEN 'Fibroids'
                -- No named-pattern match — show the real diagnosis text
                -- (truncated) instead of an opaque "Other / Unclassified"
                -- bucket, so the chart reflects actual conditions.
                ELSE LEFT(INITCAP(wrs.raw_dx_label), 60)
            END AS condition_group
        FROM with_return_status wrs
    )
    -- Two grains, unioned: condition-level detail (thresholded per
    -- (segment, condition, outcome) — appropriate here, since a single
    -- named condition with only 1-2 patients is too noisy to chart) and a
    -- separate SEGMENT-level rollup thresholded only per (segment,
    -- outcome), computed straight from `classified` rather than by
    -- re-summing the condition-level rows in Python. Re-summing the
    -- thresholded detail would silently drop any condition/outcome
    -- combination with 1-2 patients before it ever reached the segment
    -- total — for a segment split across many condition labels (e.g. ANC),
    -- that can delete most of the real "showed up" volume and leave only
    -- whichever bucket happened to clear 3, producing a segment-level rate
    -- that looks like 100% never-returned but is actually a sampling
    -- artifact of the condition-level cutoff, not the segment's true rate.
    SELECT primary_visit_segment, condition_group, attendance_outcome, total_patients
    FROM (
        SELECT
            primary_visit_segment,
            condition_group,
            attendance_outcome,
            COUNT(*) AS total_patients
        FROM classified c
        WHERE primary_visit_segment IS NOT NULL AND primary_visit_segment <> 'Other'
        GROUP BY primary_visit_segment, condition_group, attendance_outcome
        HAVING COUNT(*) >= 3

        UNION ALL

        SELECT
            primary_visit_segment,
            '__SEGMENT_TOTAL__' AS condition_group,
            attendance_outcome,
            COUNT(*) AS total_patients
        FROM classified c
        WHERE primary_visit_segment IS NOT NULL AND primary_visit_segment <> 'Other'
        GROUP BY primary_visit_segment, attendance_outcome
        HAVING COUNT(*) >= 3
    )
    ORDER BY primary_visit_segment, condition_group,
        CASE attendance_outcome
            WHEN 'Showed EARLY' THEN 1
            WHEN 'Showed ON TIME / mildly LATE (within 30 days)' THEN 2
            WHEN 'Returned, but well beyond scheduled date (30+ days late)' THEN 3
            ELSE 4
        END
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# DIAGNOSTIC — ANC patients who never returned after a scheduled follow-up:
# were they truly one-and-done, and did any carry a high-risk complication
# flag? Answers the "what does 100% never-return mean" question directly,
# patient by patient, rather than leaving it as an unexplained rate.
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_fr_anc_never_return_profile() -> pd.DataFrame:
    """One row per ANC patient who had a scheduled follow-up and never
    returned. Columns: PATIENT_ID, INDEX_VISIT_DATE, INDEX_DIAGNOSIS,
    LIFETIME_VISIT_COUNT (all diagnosis-enriched visits ever recorded for
    this patient, any segment), IS_SINGLE_VISIT_EVER (bool — did they only
    ever show up this once, anywhere), HAS_COMPLICATION_FLAG (bool — does
    ANY of their diagnosis text anywhere mention a high-risk pregnancy
    complication), COMPLICATION_TEXT (the matching diagnosis text, if any)."""
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
    diag_full AS (
        SELECT
            visit_id, patient_id,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text,
            COALESCE(NULLIF(TRIM(diagnosis_name_expanded), ''), NULLIF(TRIM(icd10_names), ''), 'Unspecified')
                AS raw_dx_label,
            TRIM(bg.value::STRING) AS split_burden
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED,
        LATERAL FLATTEN(input => SPLIT(burden_group, '|')) bg
    ),
    segmented_visits AS (
        SELECT DISTINCT
            visit_id, patient_id, clean_dx_text, raw_dx_label,
            CASE
                WHEN clean_dx_text LIKE '%spine%' OR clean_dx_text LIKE '%sciatica%' THEN 'Core Orthopedics: Spine and Back Pain Care'
                WHEN clean_dx_text LIKE '%fracture%' OR clean_dx_text LIKE '%osteoarthritis%' THEN 'Core Orthopedics: General'
                WHEN clean_dx_text LIKE '%hernia%' THEN 'Core General Surgery'
                WHEN clean_dx_text LIKE '%pregnan%' OR clean_dx_text LIKE '%anc%' OR clean_dx_text LIKE '%antenatal%'
                     OR clean_dx_text LIKE '%gravid%' OR split_burden = 'General: Obstetric'
                    THEN 'ANC'
                WHEN split_burden = 'General: Gynaecology' OR clean_dx_text LIKE '%fibroid%'
                     OR clean_dx_text LIKE '%myoma%' OR clean_dx_text LIKE '%leiomyoma%'
                    THEN 'Maternal Health'
                ELSE 'Other'
            END AS primary_visit_segment
        FROM diag_full
    ),
    with_return_status AS (
        SELECT
            c.visit_id, c.patient_id, c.consult_date AS index_visit_date,
            c.scheduled_follow_up_date, sv.primary_visit_segment, sv.raw_dx_label,
            MIN(c2.consult_date) AS next_consult_date
        FROM consultations c
        LEFT JOIN segmented_visits sv ON sv.visit_id = c.visit_id
        LEFT JOIN consultations c2
            ON c2.patient_id = c.patient_id AND c2.consult_date > c.consult_date
        WHERE c.scheduled_follow_up_date IS NOT NULL
        GROUP BY c.visit_id, c.patient_id, c.consult_date, c.scheduled_follow_up_date, sv.primary_visit_segment,
                 sv.raw_dx_label
    ),
    anc_never_return AS (
        SELECT visit_id, patient_id, index_visit_date, raw_dx_label
        FROM with_return_status
        WHERE primary_visit_segment = 'ANC' AND next_consult_date IS NULL
    ),
    -- Every diagnosis-bearing visit this patient has EVER had, anywhere,
    -- any segment — not just ANC — to answer "did they come for this
    -- ANC visit only, or have they been seen elsewhere too."
    patient_lifetime_visits AS (
        SELECT patient_id, COUNT(DISTINCT visit_id) AS lifetime_visit_count
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
        GROUP BY patient_id
    ),
    patient_complication_flag AS (
        SELECT
            patient_id,
            MAX(CASE
                WHEN LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%pre-eclamp%'
                  OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%preeclamp%'
                  OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%eclamp%'
                  OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%hyperem%'
                  OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%pprom%'
                  OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%antepartum haemorrhage%'
                  OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%antepartum hemorrhage%'
                  OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%gestational diabetes%'
                  OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%miscarriage%'
                  OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%abortion%'
                  OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%stillbirth%'
                  OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%iugr%'
                  OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%intrauterine growth%'
                THEN 1 ELSE 0
            END) AS has_complication_flag,
            MAX(CASE
                WHEN LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%pre-eclamp%'
                  OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%preeclamp%'
                  OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%eclamp%'
                  OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%hyperem%'
                  OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%pprom%'
                  OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%antepartum haemorrhage%'
                  OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%antepartum hemorrhage%'
                  OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%gestational diabetes%'
                  OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%miscarriage%'
                  OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%abortion%'
                  OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%stillbirth%'
                  OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%iugr%'
                  OR LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) LIKE '%intrauterine growth%'
                THEN COALESCE(diagnosis_name_expanded, icd10_names)
            END) AS complication_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
        GROUP BY patient_id
    )
    SELECT
        anr.patient_id,
        anr.index_visit_date,
        INITCAP(anr.raw_dx_label) AS index_diagnosis,
        plv.lifetime_visit_count,
        (plv.lifetime_visit_count = 1) AS is_single_visit_ever,
        COALESCE(pcf.has_complication_flag, 0)::BOOLEAN AS has_complication_flag,
        pcf.complication_text
    FROM anc_never_return anr
    LEFT JOIN patient_lifetime_visits plv ON plv.patient_id = anr.patient_id
    LEFT JOIN patient_complication_flag pcf ON pcf.patient_id = anr.patient_id
    ORDER BY anr.index_visit_date DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# SECTION B — Q6: General Surgery counter-example
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_fr_general_surgery_counterexample() -> pd.DataFrame:
    """Single row. Columns: TOTAL_GS_VISITS, WITH_SCHEDULED_FOLLOW_UP,
    PCT_SCHEDULED, NOT_SCHEDULED_PATIENTS, NOT_SCHEDULED_BUT_RETURNED,
    PCT_UNSCHEDULED_RETURNED_ANYWAY."""
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
        SELECT oo.id AS order_id, f.VALUE::STRING AS item_id
        FROM HOSPITALS.ORTHOPEDIC_CLEAN."ORTHO_ORDERS" oo,
        LATERAL FLATTEN(INPUT => TRY_PARSE_JSON(oo.items)) f
        QUALIFY ROW_NUMBER() OVER (PARTITION BY f.VALUE::STRING ORDER BY oo.registered_at DESC) = 1
    ),
    consultations AS (
        SELECT of_.order_id AS visit_id, s.patient_id, dp.scheduled_follow_up_date
        FROM doctor_pivot dp
        JOIN soi s ON s.soi_id = dp.soi_id
        JOIN orders_flat of_ ON of_.item_id = dp.soi_id
    ),
    all_visits AS (
        SELECT
            visit_id, patient_id, diagnosis_created_at AS visit_date,
            LOWER(COALESCE(diagnosis_name_expanded, '') || ' ' || COALESCE(icd10_names, '')) AS clean_dx_text
        FROM HOSPITALS.STAGING.STG_SPH_DIAGNOSIS_ENRICHED
    ),
    gs_visits AS (
        SELECT DISTINCT visit_id, patient_id, visit_date
        FROM all_visits
        WHERE clean_dx_text LIKE '%hernia%'
    ),
    gs_consultations AS (
        SELECT c.visit_id, c.scheduled_follow_up_date, gv.patient_id, gv.visit_date
        FROM consultations c
        JOIN gs_visits gv ON gv.visit_id = c.visit_id
    ),
    visit_sequence AS (
        SELECT
            av.visit_id, av.patient_id, av.visit_date,
            ROW_NUMBER() OVER (PARTITION BY av.patient_id ORDER BY av.visit_date) AS visit_number
        FROM all_visits av
        WHERE av.clean_dx_text LIKE '%hernia%'
    ),
    returning_gs_patients AS (
        SELECT patient_id FROM visit_sequence WHERE visit_number > 1
    )
    SELECT
        COUNT(DISTINCT gc.visit_id) AS total_gs_visits,
        SUM(CASE WHEN gc.scheduled_follow_up_date IS NOT NULL THEN 1 ELSE 0 END) AS with_scheduled_follow_up,
        ROUND(100.0 * SUM(CASE WHEN gc.scheduled_follow_up_date IS NOT NULL THEN 1 ELSE 0 END)
                   / COUNT(DISTINCT gc.visit_id), 1) AS pct_scheduled,
        COUNT(DISTINCT CASE WHEN gc.scheduled_follow_up_date IS NULL THEN gc.patient_id END) AS not_scheduled_patients,
        COUNT(DISTINCT CASE WHEN gc.scheduled_follow_up_date IS NULL AND rgp.patient_id IS NOT NULL THEN gc.patient_id END) AS not_scheduled_but_returned,
        ROUND(100.0 * COUNT(DISTINCT CASE WHEN gc.scheduled_follow_up_date IS NULL AND rgp.patient_id IS NOT NULL THEN gc.patient_id END)
                   / NULLIF(COUNT(DISTINCT CASE WHEN gc.scheduled_follow_up_date IS NULL THEN gc.patient_id END), 0), 1) AS pct_unscheduled_returned_anyway
    FROM gs_consultations gc
    LEFT JOIN returning_gs_patients rgp ON rgp.patient_id = gc.patient_id
    """
    return _run(sql)
