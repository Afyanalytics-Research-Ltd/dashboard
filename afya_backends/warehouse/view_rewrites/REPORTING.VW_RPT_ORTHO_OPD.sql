-- for_hash: 923f1fc8c23ede2cf3c9a1f3b1c499c2
--
-- Databend stand-in for HOSPITALS.REPORTING.VW_RPT_ORTHO_OPD (Snowflake definition with that md5), used by
-- warehouse_bisync instead of translating the Snowflake text. Same rows, written so Databend can plan it.
--
-- Why: the Snowflake view filters V2 visits with a correlated
--   NOT EXISTS (SELECT 1 FROM stg_admissions ipd WHERE ipd.patient_id = v.patient_id
--               AND v.visit_date > ipd.admission_date AND (v.visit_date <= ipd.discharge_date OR ...))
-- and Databend can't decorrelate it when the outer side (STAGING.STG_VISITS) contains a FLATTEN:
-- "join left plan can't contain ProjectSet to dcorrelated join right plan".
--
-- Rewrite: the NOT EXISTS only reads v.patient_id and v.visit_date, so the (patient_id, visit_date) pairs that
-- DO have a covering admission are computed once (v2_after_admission, DISTINCT -> at most one match per visit,
-- no fan-out) and anti-joined: NOT EXISTS(...) <=> x.patient_id IS NULL. NULL patient_id / visit_date match
-- nothing in either form. Everything else is the Snowflake definition unchanged.
--
-- If the Snowflake view changes, its md5 no longer matches and bisync translates the new Snowflake text
-- (reporting this file as stale if that fails) - update this file then.
CREATE OR REPLACE VIEW HOSPITALS.REPORTING.vw_rpt_ortho_opd AS
WITH v2_after_admission AS (
    SELECT DISTINCT vv.patient_id, vv.visit_date
    FROM (SELECT DISTINCT patient_id, visit_date
          FROM HOSPITALS.STAGING.stg_visits
          WHERE source_system = 'EMR_V2') vv
    JOIN HOSPITALS.STAGING.stg_admissions ipd
      ON ipd.patient_id = vv.patient_id
     AND vv.visit_date > ipd.admission_date
     AND (vv.visit_date <= ipd.discharge_date OR ipd.discharge_date IS NULL)
)
SELECT
    v.visit_id,
    v.patient_id,
    v.composite_patient_id,
    v.facility,
    v.source_system,
    v.visit_date,
    v.visit_timestamp,
    HOUR(v.visit_timestamp)             AS hour_of_day,
    COALESCE(
        HOUR(v.visit_timestamp),
        HOUR(TIMESTAMP_FROM_PARTS(v.visit_date, TRY_TO_TIME(v.start_time)))
    )                                   AS arrival_hour,
    v.visit_close_timestamp,
    DATE_TRUNC('month', v.visit_date)   AS visit_month,
    DATE_TRUNC('week',  v.visit_date)   AS visit_week,
    YEAR(v.visit_date)                  AS visit_year,
    DAYOFWEEK(v.visit_date)             AS visit_dow,
    DAYNAME(v.visit_date)               AS visit_day_name,
    v.visit_type,
    CASE
        WHEN v.visit_type = 'New Patient'           THEN TRUE
        WHEN v.visit_type IN ('Revisit', 'Walk-In') THEN FALSE
        ELSE NULL
    END AS is_new_patient,
    CASE WHEN v.visit_type = 'Walk-In' THEN TRUE ELSE FALSE END AS is_walk_in,
    v.gender,
    v.age_at_visit,
    v.age_group
FROM HOSPITALS.STAGING.stg_visits v
LEFT JOIN HOSPITALS.STAGING.stg_admissions a
    ON v.visit_id = a.visit_id
LEFT JOIN v2_after_admission x
    ON x.patient_id = v.patient_id AND x.visit_date = v.visit_date
WHERE
    (v.source_system = 'EMR_V2'
     AND v.visit_type != 'Inpatient'
     AND x.patient_id IS NULL)
    OR
    (v.source_system = 'EMR_V1' AND a.visit_id IS NULL)
