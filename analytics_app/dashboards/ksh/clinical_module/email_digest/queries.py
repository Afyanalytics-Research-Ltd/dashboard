QUERY = '''
/*
================================================================================
MONTHLY DIGEST SUMMARY QUERY
Afya Clinical Analytics — Kisumu Specialists
================================================================================
PURPOSE:
    Single query that powers the Head of Clinician monthly email digest.
    Returns two rows — last month and the month before — with all metrics
    needed to compute deltas and generate AI recommendations.

    n8n runs this query on the 1st of every month at 08:00 EAT.
    Output is passed as JSON context to Groq for recommendation generation.

METRICS RETURNED:
    report_month                Month being reported
    total_opd_visits            Total outpatient visits
    total_ipd_admissions        Total inpatient conversions
    conversion_rate_pct         Overall OPD to IPD rate
    retention_universe_visits   Visits from complex patients (retention universe)
    retention_ipd_admissions    IPD admissions from retention universe
    retention_rate_pct          Retention universe conversion rate
    comorbid_rate_pct           Comorbid patient conversion rate
    single_dx_rate_pct          Single diagnosis conversion rate
    avg_visits_per_clinician    Clinician workload signal
    wait_time_gap_mins          Triage gap (admitted vs not admitted)
    strain_signal               HIGH_STRAIN / CAPACITY_GAP / AS_EXPECTED
    total_escalations           72-hour OPD to IPD escalations
    top_escalation_age_group    Age group with most escalations this month

PERIOD:
    last_month  = DATE_TRUNC('month', DATEADD('month', -1, CURRENT_DATE))
    prior_month = DATE_TRUNC('month', DATEADD('month', -2, CURRENT_DATE))
================================================================================
*/

WITH

-- ── Period anchors ────────────────────────────────────────────────────────────
periods AS (
    SELECT
        DATE_TRUNC('month', DATEADD('month', -1, CURRENT_DATE)) AS last_month,
        DATE_TRUNC('month', DATEADD('month', -2, CURRENT_DATE)) AS prior_month
),

-- ── Base visits + conversion flag ────────────────────────────────────────────
visit_base AS (
    SELECT
        v.source_schema,
        v.id                                                    AS visit_id,
        v.patient                                               AS patient_id,
        v.user                                                  AS clinician,
        DATE_TRUNC('month', v.created_at)                       AS visit_month,
        v.created_at,
        IFF(a.visit_id IS NOT NULL, TRUE, FALSE)                AS converted_to_inpatient
    FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
    LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
        ON  v.id            = a.visit_id
        AND v.source_schema = LOWER(REPLACE(a.source_schema, '_CLEAN', ''))
    CROSS JOIN periods p
    WHERE v.source_schema = 'kisumu'
      AND DATE_TRUNC('month', v.created_at) IN (p.last_month, p.prior_month)
),

-- ── Conversion summary per month ─────────────────────────────────────────────
conversion AS (
    SELECT
        visit_month                                             AS report_month,
        COUNT(DISTINCT visit_id)                                AS total_opd_visits,
        COUNT(DISTINCT CASE
            WHEN converted_to_inpatient THEN visit_id END)      AS total_ipd_admissions,
        ROUND(DIV0(
            COUNT(DISTINCT CASE WHEN converted_to_inpatient THEN visit_id END),
            COUNT(DISTINCT visit_id)
        ) * 100.0, 2)                                           AS conversion_rate_pct
    FROM visit_base
    GROUP BY visit_month
),

-- ── Clinician workload per month ──────────────────────────────────────────────
clinician_workload AS (
    SELECT
        visit_month                                             AS report_month,
        COUNT(DISTINCT visit_id)                                AS total_visits,
        COUNT(DISTINCT clinician)                               AS active_clinicians,
        ROUND(DIV0(
            COUNT(DISTINCT visit_id),
            COUNT(DISTINCT clinician)
        ), 1)                                                   AS avg_visits_per_clinician
    FROM visit_base
    WHERE clinician IS NOT NULL
    GROUP BY visit_month
),

-- ── ED wait time gap per month ────────────────────────────────────────────────
vitals_first AS (
    SELECT visit_id, source_schema, MIN(created_at) AS first_vitals
    FROM HOSPITALS.STAGING.STG_EVALUATION_VITALS
    GROUP BY visit_id, source_schema
),
notes_first AS (
    SELECT visit_id, source_schema, MIN(created_at) AS first_note
    FROM HOSPITALS.STAGING.STG_EVALUATION_DOCTOR_NOTES
    GROUP BY visit_id, source_schema
),
wait_times AS (
    SELECT
        vb.visit_month                                          AS report_month,
        vb.converted_to_inpatient,
        CASE
            WHEN vt.first_vitals IS NOT NULL
             AND n.first_note    IS NOT NULL
             AND n.first_note     > vt.first_vitals
             AND DATEDIFF('hour', vt.first_vitals, n.first_note) < 12
                THEN DATEDIFF('minute', vt.first_vitals, n.first_note)
        END                                                     AS mins_triage_to_consult
    FROM visit_base vb
    LEFT JOIN vitals_first vt
        ON  vb.visit_id      = vt.visit_id
        AND vb.source_schema = vt.source_schema
    LEFT JOIN notes_first n
        ON  vb.visit_id      = n.visit_id
        AND vb.source_schema = n.source_schema
),
ed_gap AS (
    SELECT
        report_month,
        ROUND(
            MEDIAN(CASE WHEN converted_to_inpatient
                THEN mins_triage_to_consult END)
            - MEDIAN(CASE WHEN NOT converted_to_inpatient
                THEN mins_triage_to_consult END),
        1)                                                      AS wait_time_gap_mins
    FROM wait_times
    GROUP BY report_month
),

-- ── Comorbidity conversion rates per month ────────────────────────────────────
diag_flags AS (
    SELECT
        i.source_schema,
        i.visit_id,
        IFF(
            i.icd10_code_1 IS NOT NULL
            AND (i.icd10_code_2 IS NOT NULL OR i.icd10_code_3 IS NOT NULL),
            TRUE, FALSE
        )                                                       AS is_comorbidity
    FROM HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED i
    WHERE i.source_schema = 'kisumu'
),
comorbidity_rates AS (
    SELECT
        vb.visit_month                                          AS report_month,
        ROUND(DIV0(
            COUNT(DISTINCT CASE
                WHEN vb.converted_to_inpatient
                 AND df.is_comorbidity = TRUE
                THEN vb.visit_id END),
            NULLIF(COUNT(DISTINCT CASE
                WHEN df.is_comorbidity = TRUE
                THEN vb.visit_id END), 0)
        ) * 100.0, 2)                                           AS comorbid_rate_pct,
        ROUND(DIV0(
            COUNT(DISTINCT CASE
                WHEN vb.converted_to_inpatient
                 AND df.is_comorbidity = FALSE
                THEN vb.visit_id END),
            NULLIF(COUNT(DISTINCT CASE
                WHEN df.is_comorbidity = FALSE
                THEN vb.visit_id END), 0)
        ) * 100.0, 2)                                           AS single_dx_rate_pct
    FROM visit_base vb
    LEFT JOIN diag_flags df
        ON  vb.visit_id      = df.visit_id
        AND vb.source_schema = df.source_schema
    GROUP BY vb.visit_month
),

-- ── Retention universe rate per month ─────────────────────────────────────────
-- Uses ILIKE pattern matching on disease_burden_group_1 as a lightweight proxy
-- Full pipeline runs in retention_universe_conversion_rate.sql

retention_proxy AS (
    SELECT
        vb.visit_month                                          AS report_month,
        COUNT(DISTINCT vb.visit_id)                             AS retention_universe_visits,
        COUNT(DISTINCT CASE
            WHEN vb.converted_to_inpatient THEN vb.visit_id END) AS retention_ipd_admissions,
        ROUND(DIV0(
            COUNT(DISTINCT CASE WHEN vb.converted_to_inpatient THEN vb.visit_id END),
            COUNT(DISTINCT vb.visit_id)
        ) * 100.0, 2)                                           AS retention_rate_pct
    FROM visit_base vb
    LEFT JOIN HOSPITALS.STAGING.STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED i
        ON  vb.visit_id      = i.visit_id
        AND vb.source_schema = i.source_schema
    WHERE (
        i.disease_burden_group_1 ILIKE '%NCD%'
        OR i.disease_burden_group_1 ILIKE '%Oncology%'
        OR i.disease_burden_group_1 ILIKE '%MNCH - Maternal%'
        OR i.disease_burden_group_1 ILIKE '%Mental Health%'
        OR i.disease_burden_group_1 ILIKE '%HIV%'
        OR i.disease_burden_group_1 ILIKE '%Tuberculosis%'
        OR i.disease_burden_group_1 ILIKE '%Hepatitis%'
    )
    GROUP BY vb.visit_month
),

-- ── 72-hour escalations per month ────────────────────────────────────────────
escalation_summary AS (
    SELECT
        DATE_TRUNC('month', a.admitted_at)                      AS report_month,
        COUNT(DISTINCT a.visit_id)                              AS total_escalations
    FROM HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    INNER JOIN HOSPITALS.STAGING.STG_EVALUATION_VISITS v
        ON  a.visit_id      = v.id
        AND LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = v.source_schema
    CROSS JOIN periods p
    WHERE LOWER(REPLACE(a.source_schema, '_CLEAN', '')) = 'kisumu'
      AND DATE_TRUNC('month', a.admitted_at) IN (p.last_month, p.prior_month)
      AND DATEDIFF('hour', v.created_at, a.admitted_at) BETWEEN 0 AND 72
    GROUP BY DATE_TRUNC('month', a.admitted_at)
),

-- ── Strain signal per month ───────────────────────────────────────────────────
strain_signal AS (
    SELECT
        c.report_month,
        CASE
            WHEN cw.avg_visits_per_clinician > AVG(cw.avg_visits_per_clinician)
                    OVER (PARTITION BY c.report_month)
             AND (ed.wait_time_gap_mins IS NULL OR ed.wait_time_gap_mins > -5)
                THEN 'HIGH_STRAIN'
            WHEN c.conversion_rate_pct < AVG(c.conversion_rate_pct)
                    OVER (PARTITION BY c.report_month)
             AND cw.avg_visits_per_clinician > AVG(cw.avg_visits_per_clinician)
                    OVER (PARTITION BY c.report_month)
                THEN 'CAPACITY_GAP'
            ELSE 'AS_EXPECTED'
        END                                                     AS strain_signal
    FROM conversion c
    LEFT JOIN clinician_workload cw ON c.report_month = cw.report_month
    LEFT JOIN ed_gap ed             ON c.report_month = ed.report_month
)

-- ── Final output — two rows: last_month and prior_month ───────────────────────
SELECT
    c.report_month,
    IFF(c.report_month = p.last_month, 'last_month', 'prior_month')
                                                                AS period_label,
    c.total_opd_visits,
    c.total_ipd_admissions,
    c.conversion_rate_pct,
    r.retention_universe_visits,
    r.retention_ipd_admissions,
    r.retention_rate_pct,
    cm.comorbid_rate_pct,
    cm.single_dx_rate_pct,
    cw.avg_visits_per_clinician,
    cw.active_clinicians,
    ed.wait_time_gap_mins,
    ss.strain_signal,
    e.total_escalations
FROM conversion c
CROSS JOIN periods p
LEFT JOIN retention_proxy   r  ON c.report_month = r.report_month
LEFT JOIN comorbidity_rates cm ON c.report_month = cm.report_month
LEFT JOIN clinician_workload cw ON c.report_month = cw.report_month
LEFT JOIN ed_gap            ed ON c.report_month = ed.report_month
LEFT JOIN strain_signal     ss ON c.report_month = ss.report_month
LEFT JOIN escalation_summary e ON c.report_month = e.report_month
WHERE c.report_month IN (p.last_month, p.prior_month)
ORDER BY c.report_month DESC
'''
;