from .db import run_query_df

_VALID_FACILITIES = frozenset({"TENRI", "KISUMU_CLEAN"})


def _flt(facility, col="facility"):
    if facility and facility in _VALID_FACILITIES:
        return f"AND {col} = '{facility}'"
    return ""


# ── Page 1: The Business Today ────────────────────────────────────────────


def q_overview_gap():
    """G6: billed, collected, outstanding by facility — KPI cards + bar."""
    return run_query_df("""
        SELECT
            facility,
            SUM(total_billed)       AS total_billed,
            SUM(total_collected)    AS total_collected,
            SUM(total_outstanding)  AS total_outstanding,
            ROUND(100.0 * SUM(total_collected)
                  / NULLIF(SUM(total_billed), 0), 2) AS collection_rate_pct
        FROM HOSPITALS.REPORTING.rpt_insurance_ar
        GROUP BY facility
        ORDER BY facility
    """)


def q_overview_alerts():
    """G7: full insurer × facility × bucket summary — dashboard extracts 4 act-now signals."""
    return run_query_df("""
        SELECT
            facility,
            insurer,
            aging_bucket,
            SUM(invoices)             AS invoices,
            SUM(total_outstanding)    AS total_outstanding,
            SUM(dispatched_invoices)  AS dispatched_invoices,
            ROUND(100.0 * SUM(dispatched_invoices)
                  / NULLIF(SUM(invoices), 0), 2) AS dispatch_rate_pct,
            ROUND(AVG(avg_days_outstanding), 0)  AS avg_days_outstanding,
            MAX(facility_data_end)               AS facility_data_end
        FROM HOSPITALS.REPORTING.rpt_ar_aging
        GROUP BY facility, insurer, aging_bucket
        ORDER BY facility, total_outstanding DESC
    """)


# ── Page 2: Where the Money Isn't Arriving ───────────────────────────────


def q_leakage_gap(facility=None):
    """G6: top insurers by billed vs collected — stacked bar."""
    f = _flt(facility)
    return run_query_df(f"""
        SELECT
            facility,
            insurer,
            SUM(total_billed)       AS total_billed,
            SUM(total_collected)    AS total_collected,
            SUM(total_outstanding)  AS total_outstanding,
            ROUND(100.0 * SUM(total_collected)
                  / NULLIF(SUM(total_billed), 0), 2) AS collection_rate_pct,
            SUM(invoices)           AS invoices
        FROM HOSPITALS.REPORTING.rpt_insurance_ar
        WHERE 1=1 {f}
        GROUP BY facility, insurer
        ORDER BY total_billed DESC
        LIMIT 20
    """)


def q_leakage_submission_rate(facility=None):
    """G6: % claims submitted (dispatch rate) by insurer."""
    f = _flt(facility)
    return run_query_df(f"""
        SELECT
            facility,
            insurer,
            SUM(invoices)             AS invoices,
            SUM(dispatched_invoices)  AS dispatched_invoices,
            ROUND(100.0 * SUM(dispatched_invoices)
                  / NULLIF(SUM(invoices), 0), 2) AS dispatch_rate_pct,
            SUM(total_outstanding)    AS total_outstanding
        FROM HOSPITALS.REPORTING.rpt_insurance_ar
        WHERE 1=1 {f}
        GROUP BY facility, insurer
        ORDER BY dispatch_rate_pct ASC
    """)


def q_leakage_ksh_dispatch_trend():
    """G6: monthly dispatched invoices for KSH from Jan 2025 — shows Sep 2025 cliff (system-wide, all insurers)."""
    return run_query_df("""
        SELECT
            invoice_month,
            SUM(invoices)             AS invoices,
            SUM(dispatched_invoices)  AS dispatched_invoices,
            ROUND(100.0 * SUM(dispatched_invoices)
                  / NULLIF(SUM(invoices), 0), 2) AS dispatch_rate_pct,
            SUM(total_outstanding)    AS total_outstanding
        FROM HOSPITALS.REPORTING.rpt_insurance_ar
        WHERE facility    = 'KISUMU_CLEAN'
          AND invoice_month >= '2025-01-01'
          AND invoice_month >= '2000-01-01'
        GROUP BY invoice_month
        ORDER BY invoice_month
    """)


def q_leakage_aging_dist(facility=None):
    """G7: aging bucket distribution by insurer — stacked bar, outstanding only."""
    f = _flt(facility)
    return run_query_df(f"""
        SELECT
            facility,
            insurer,
            aging_bucket,
            SUM(invoices)           AS invoices,
            SUM(total_outstanding)  AS total_outstanding
        FROM HOSPITALS.REPORTING.rpt_ar_aging
        WHERE aging_bucket != 'Collected'
          {f}
        GROUP BY facility, insurer, aging_bucket
        ORDER BY facility, total_outstanding DESC
    """)


def q_leakage_recovery_priority(facility=None):
    """G7: recovery priority table — outstanding, dispatch %, days waiting, expected recoverable."""
    f = _flt(facility)
    return run_query_df(f"""
        SELECT
            facility,
            insurer,
            SUM(invoices)                                        AS invoices,
            SUM(total_outstanding)                               AS outstanding_kes,
            SUM(dispatched_invoices)                             AS dispatched_invoices,
            ROUND(100.0 * SUM(dispatched_invoices)
                  / NULLIF(SUM(invoices), 0), 2)                AS dispatch_rate_pct,
            ROUND(AVG(avg_days_outstanding), 0)                 AS avg_days_outstanding,
            SUM(CASE WHEN aging_bucket IN ('0-30','31-60','61-90')
                     THEN total_outstanding ELSE 0 END)          AS expected_recoverable_kes,
            SUM(CASE WHEN aging_bucket = '90+'
                     THEN total_outstanding ELSE 0 END)          AS outstanding_90plus
        FROM HOSPITALS.REPORTING.rpt_ar_aging
        WHERE aging_bucket != 'Collected'
          {f}
        GROUP BY facility, insurer
        ORDER BY outstanding_kes DESC
    """)


# ── Page 3: How We're Using What We Have ─────────────────────────────────


def q_theatre_trend():
    """G3: monthly completion rate + revenue — KSH only."""
    return run_query_df("""
        SELECT
            session_month,
            SUM(total_sessions)     AS total_sessions,
            SUM(completed_sessions) AS completed_sessions,
            ROUND(100.0 * SUM(completed_sessions)
                  / NULLIF(SUM(total_sessions), 0), 2) AS completion_rate_pct,
            SUM(total_revenue)      AS total_revenue,
            SUM(emergency_sessions) AS emergency_sessions
        FROM HOSPITALS.REPORTING.rpt_theatre_utilization
        GROUP BY session_month
        ORDER BY session_month
    """)


def q_theatre_by_type():
    """G3: revenue + sessions by theatre_type — KSH only."""
    return run_query_df("""
        SELECT
            theatre_type,
            theatre_name,
            SUM(total_sessions)                   AS total_sessions,
            SUM(completed_sessions)               AS completed_sessions,
            ROUND(100.0 * SUM(completed_sessions)
                  / NULLIF(SUM(total_sessions), 0), 2) AS completion_rate_pct,
            SUM(total_revenue)                    AS total_revenue,
            ROUND(AVG(avg_revenue_per_session), 2) AS avg_revenue_per_session,
            ROUND(SUM(total_duration_hrs), 1)     AS total_duration_hrs
        FROM HOSPITALS.REPORTING.rpt_theatre_utilization
        GROUP BY theatre_type, theatre_name
        ORDER BY total_revenue DESC
    """)


def q_beds_revpab(facility=None):
    """G1: revenue per available bed proxy (revenue / bed_days) by ward — ranked bar."""
    f = _flt(facility)
    return run_query_df(f"""
        SELECT
            facility,
            ward_category,
            ward_name,
            SUM(total_admissions)        AS total_admissions,
            SUM(total_bed_days)          AS total_bed_days,
            SUM(total_admission_revenue) AS total_revenue,
            ROUND(SUM(total_admission_revenue)
                  / NULLIF(SUM(total_bed_days), 0), 2) AS revpab,
            ROUND(SUM(total_bed_days)
                  / NULLIF(SUM(discharged_admissions), 0), 2) AS avg_los_days
        FROM HOSPITALS.REPORTING.rpt_bed_occupancy
        WHERE 1=1 {f}
        GROUP BY facility, ward_category, ward_name
        ORDER BY revpab DESC NULLS LAST
    """)


def q_beds_los(facility=None):
    """G1: avg LOS by ward_category — horizontal bar comparison."""
    f = _flt(facility)
    return run_query_df(f"""
        SELECT
            facility,
            ward_category,
            ROUND(SUM(total_bed_days)
                  / NULLIF(SUM(discharged_admissions), 0), 2) AS avg_los_days,
            SUM(total_admissions)       AS total_admissions,
            SUM(discharged_admissions)  AS discharged_admissions,
            SUM(total_admission_revenue) AS total_revenue
        FROM HOSPITALS.REPORTING.rpt_bed_occupancy
        WHERE 1=1 {f}
        GROUP BY facility, ward_category
        ORDER BY avg_los_days DESC
    """)


def q_beds_monthly():
    """G1: monthly bed days + admissions + LOS per ward_name — KSH only.
    Groups by ward_name (7 distinct wards) not ward_category (5 groups) — Inv 36."""
    return run_query_df("""
        SELECT
            ward_name,
            admission_month,
            SUM(total_admissions)                                                AS total_admissions,
            SUM(discharged_admissions)                                           AS discharged_admissions,
            SUM(total_bed_days)                                                  AS total_bed_days,
            ROUND(SUM(total_bed_days)
                  / NULLIF(SUM(discharged_admissions), 0), 2)                   AS avg_los_days
        FROM HOSPITALS.REPORTING.rpt_bed_occupancy
        WHERE facility = 'KISUMU_CLEAN'
          AND ward_name IS NOT NULL
        GROUP BY ward_name, admission_month
        ORDER BY ward_name, admission_month
    """)


def q_dialysis_trend(facility=None):
    """G4: monthly sessions + revenue per session — dual-axis trend."""
    f = _flt(facility)
    return run_query_df(f"""
        SELECT
            facility,
            session_month,
            SUM(total_sessions)                                AS total_sessions,
            SUM(distinct_patients)                             AS distinct_patients,
            ROUND(AVG(avg_duration_hrs), 2)                    AS avg_duration_hrs,
            SUM(total_dialysis_revenue)                        AS total_revenue,
            ROUND(SUM(total_dialysis_revenue)
                  / NULLIF(SUM(total_sessions), 0), 2)         AS revenue_per_session
        FROM HOSPITALS.REPORTING.rpt_dialysis
        WHERE 1=1 {f}
        GROUP BY facility, session_month
        ORDER BY facility, session_month
    """)


def q_specialty_admissions():
    """G4b: TENRI only — day case % + LOS by ward, monthly trend."""
    return run_query_df("""
        SELECT
            ward_name,
            admission_month,
            SUM(total_admissions)   AS total_admissions,
            SUM(day_cases)          AS day_cases,
            SUM(inpatient_stays)    AS inpatient_stays,
            ROUND(100.0 * SUM(day_cases)
                  / NULLIF(SUM(total_admissions), 0), 1) AS day_case_pct,
            ROUND(AVG(avg_los_days), 2)                  AS avg_los_days,
            SUM(total_admission_revenue)                 AS total_revenue
        FROM HOSPITALS.REPORTING.rpt_specialty_admissions
        WHERE facility = 'TENRI'
        GROUP BY ward_name, admission_month
        ORDER BY admission_month, ward_name
    """)


# ── Page 4: Patients Coming Back ─────────────────────────────────────────


def q_readmission_pattern():
    """G2: 30d readmission rate by discharge_type × facility."""
    return run_query_df("""
        SELECT
            facility,
            discharge_type,
            SUM(total_admissions)              AS total_admissions,
            SUM(readmissions_30day)            AS readmissions_30day,
            ROUND(100.0 * SUM(readmissions_30day)
                  / NULLIF(SUM(total_admissions), 0), 2) AS readmission_30day_rate_pct,
            SUM(insured_30day_revenue_at_risk) AS revenue_at_risk
        FROM HOSPITALS.REPORTING.rpt_readmissions
        WHERE discharge_type IS NOT NULL
        GROUP BY facility, discharge_type
        ORDER BY facility, readmission_30day_rate_pct DESC
    """)


def q_readmission_trend():
    """G2: monthly 30d readmission rate by facility — line chart."""
    return run_query_df("""
        SELECT
            facility,
            admission_month,
            SUM(total_admissions)   AS total_admissions,
            SUM(readmissions_30day) AS readmissions_30day,
            ROUND(100.0 * SUM(readmissions_30day)
                  / NULLIF(SUM(total_admissions), 0), 2) AS readmission_30day_rate_pct
        FROM HOSPITALS.REPORTING.rpt_readmissions
        GROUP BY facility, admission_month
        ORDER BY facility, admission_month
    """)


def q_readmission_exposure():
    """G2: KES at risk by ward — insured 30d readmissions only."""
    return run_query_df("""
        SELECT
            facility,
            ward_category,
            SUM(total_admissions)              AS total_admissions,
            SUM(readmissions_30day)            AS readmissions_30day,
            ROUND(100.0 * SUM(readmissions_30day)
                  / NULLIF(SUM(total_admissions), 0), 2) AS rate_pct,
            SUM(insured_30day_revenue_at_risk) AS revenue_at_risk
        FROM HOSPITALS.REPORTING.rpt_readmissions
        WHERE payment_mode = 'insured'
        GROUP BY facility, ward_category
        ORDER BY revenue_at_risk DESC
    """)


def q_readmission_benchmark():
    """G2: full side-by-side benchmark — all discharge types × facilities × wards."""
    return run_query_df("""
        SELECT
            facility,
            discharge_type,
            ward_category,
            SUM(total_admissions)              AS total_admissions,
            SUM(readmissions_30day)            AS readmissions_30day,
            ROUND(100.0 * SUM(readmissions_30day)
                  / NULLIF(SUM(total_admissions), 0), 2) AS rate_pct,
            SUM(insured_30day_revenue_at_risk) AS revenue_at_risk,
            SUM(approx_gap_30day_count)        AS approx_gap_count
        FROM HOSPITALS.REPORTING.rpt_readmissions
        GROUP BY facility, discharge_type, ward_category
        ORDER BY facility, rate_pct DESC NULLS LAST
    """)


# ── Page 5: What We Sell and Who Pays ────────────────────────────────────


def q_service_mix(facility=None):
    """G5: revenue by category — Rebate row included (negative), rest are positive."""
    f = _flt(facility)
    return run_query_df(f"""
        SELECT
            facility,
            revenue_category,
            SUM(total_revenue)   AS total_revenue,
            SUM(line_items)      AS line_items,
            SUM(distinct_visits) AS distinct_visits
        FROM HOSPITALS.REPORTING.rpt_procedure_revenue
        WHERE 1=1 {f}
        GROUP BY facility, revenue_category
        ORDER BY facility, total_revenue DESC
    """)


def q_rebate_by_insurer(facility=None):
    """G5: rebate KES by insurer ranked — contra-revenue exposure."""
    f = _flt(facility)
    return run_query_df(f"""
        SELECT
            facility,
            COALESCE(insurer, 'Unattributed') AS insurer,
            SUM(total_revenue) AS rebate_kes,
            SUM(line_items)    AS line_items
        FROM HOSPITALS.REPORTING.rpt_procedure_revenue
        WHERE revenue_category = 'Rebate'
          {f}
        GROUP BY facility, COALESCE(insurer, 'Unattributed')
        ORDER BY rebate_kes ASC
    """)


def q_payer_trend(facility=None):
    """G5: insured vs cash revenue % by month — payer concentration trend."""
    f = _flt(facility)
    return run_query_df(f"""
        SELECT
            facility,
            revenue_month,
            SUM(CASE WHEN payment_mode = 'insured'
                          AND revenue_category != 'Rebate'
                     THEN total_revenue ELSE 0 END)  AS insured_revenue,
            SUM(CASE WHEN payment_mode = 'cash'
                          AND revenue_category != 'Rebate'
                     THEN total_revenue ELSE 0 END)  AS cash_revenue,
            SUM(CASE WHEN revenue_category != 'Rebate'
                     THEN total_revenue ELSE 0 END)  AS gross_revenue,
            ROUND(100.0 * SUM(CASE WHEN payment_mode = 'insured'
                                        AND revenue_category != 'Rebate'
                                   THEN total_revenue ELSE 0 END)
                  / NULLIF(SUM(CASE WHEN revenue_category != 'Rebate'
                                    THEN total_revenue ELSE 0 END), 0), 2) AS insured_pct
        FROM HOSPITALS.REPORTING.rpt_procedure_revenue
        WHERE 1=1 {f}
        GROUP BY facility, revenue_month
        ORDER BY facility, revenue_month
    """)


# ── Page 4: Ward-level readmission trend (monthly, for ward breakdowns) ──────


def q_readmission_ward_trend(facility=None):
    """G2: monthly 30d readmission rate by ward_category — for ward-level trend charts."""
    f = _flt(facility)
    return run_query_df(f"""
        SELECT
            facility,
            ward_category,
            admission_month,
            SUM(total_admissions)   AS total_admissions,
            SUM(readmissions_30day) AS readmissions_30day,
            ROUND(100.0 * SUM(readmissions_30day)
                  / NULLIF(SUM(total_admissions), 0), 2) AS readmission_30day_rate_pct
        FROM HOSPITALS.REPORTING.rpt_readmissions
        WHERE 1=1 {f}
        GROUP BY facility, ward_category, admission_month
        ORDER BY facility, ward_category, admission_month
    """)


# ── Page 3: Imaging diagnostics (sourced from staging — pending G8 gold table) ──


def q_imaging_trend(facility=None):
    """Imaging revenue by modality + month — CT, MRI, ECHO, Ultrasound, X-Ray.
    NOTE: queries stg_procedure_revenue directly (no gold table yet). Flag: G8 pending."""
    fac_filter = f"AND facility = '{facility}'" if facility in _VALID_FACILITIES else ""
    return run_query_df(f"""
        SELECT
            DATE_TRUNC('month', invoice_date)::DATE  AS revenue_month,
            CASE
                WHEN (   item_name ILIKE 'CT %'
                      OR item_name ILIKE 'CT-%'
                      OR item_name ILIKE '% CT %'
                      OR item_name ILIKE '% CT-%'
                      OR item_name ILIKE '% CT'
                      OR item_name ILIKE 'HRCT%'
                      OR item_name ILIKE '%HRCT %'
                      OR item_name ILIKE '%computed%'
                      OR (item_name ILIKE '%angio%'
                          AND item_name NOT ILIKE '%angiotensin%'))
                                                      THEN 'CT / Angio'
                WHEN item_name ILIKE '%echo%'         THEN 'ECHO / Cardiac'
                WHEN item_name ILIKE '%ultrasound%'
                  OR item_name ILIKE '%USS%'
                  OR item_name ILIKE '%sonograph%'    THEN 'Ultrasound'
                WHEN item_name ILIKE '%x-ray%'
                  OR item_name ILIKE '%xray%'
                  OR item_name ILIKE '%radiograph%'   THEN 'X-Ray'
                WHEN item_name ILIKE '%MRI%'          THEN 'MRI'
                ELSE 'Other Imaging'
            END                                       AS modality,
            COUNT(*)                                  AS sessions,
            SUM(item_amount)                          AS revenue,
            ROUND(AVG(item_amount), 0)                AS avg_per_session
        FROM HOSPITALS.STAGING.stg_procedure_revenue
        WHERE item_type != 'copay'
          AND (
              item_name ILIKE 'CT %'       OR item_name ILIKE 'CT-%'
           OR item_name ILIKE '% CT %'     OR item_name ILIKE '% CT-%'
           OR item_name ILIKE '% CT'       OR item_name ILIKE 'HRCT%'
           OR item_name ILIKE '%HRCT %'    OR item_name ILIKE '%computed%'
           OR (item_name ILIKE '%angio%' AND item_name NOT ILIKE '%angiotensin%')
           OR item_name ILIKE '%echo%'
           OR item_name ILIKE '%ultrasound%' OR item_name ILIKE '%USS%'
           OR item_name ILIKE '%sonograph%'
           OR item_name ILIKE '%x-ray%' OR item_name ILIKE '%xray%'
           OR item_name ILIKE '%radiograph%'
           OR item_name ILIKE '%MRI%'
          )
          {fac_filter}
        GROUP BY
            DATE_TRUNC('month', invoice_date)::DATE,
            CASE
                WHEN (   item_name ILIKE 'CT %'
                      OR item_name ILIKE 'CT-%'
                      OR item_name ILIKE '% CT %'
                      OR item_name ILIKE '% CT-%'
                      OR item_name ILIKE '% CT'
                      OR item_name ILIKE 'HRCT%'
                      OR item_name ILIKE '%HRCT %'
                      OR item_name ILIKE '%computed%'
                      OR (item_name ILIKE '%angio%'
                          AND item_name NOT ILIKE '%angiotensin%'))
                                                      THEN 'CT / Angio'
                WHEN item_name ILIKE '%echo%'         THEN 'ECHO / Cardiac'
                WHEN item_name ILIKE '%ultrasound%'
                  OR item_name ILIKE '%USS%'
                  OR item_name ILIKE '%sonograph%'    THEN 'Ultrasound'
                WHEN item_name ILIKE '%x-ray%'
                  OR item_name ILIKE '%xray%'
                  OR item_name ILIKE '%radiograph%'   THEN 'X-Ray'
                WHEN item_name ILIKE '%MRI%'          THEN 'MRI'
                ELSE 'Other Imaging'
            END
        ORDER BY revenue_month, revenue DESC
    """)


# ── Phase 13: Intelligence Layer queries (KSH-only data sources) ──────────


def q_ward_admissions_monthly(facility=None):
    """G1: monthly admissions per ward_category — for traffic volume rules (Inv 21)."""
    f = _flt(facility)
    return run_query_df(f"""
        SELECT
            facility,
            ward_category,
            admission_month,
            SUM(total_admissions) AS admissions
        FROM HOSPITALS.REPORTING.rpt_bed_occupancy
        WHERE 1=1 {f}
        GROUP BY facility, ward_category, admission_month
        ORDER BY ward_category, admission_month
    """)


def q_ward_los_monthly(facility=None):
    """Monthly median LOS per ward — from stg_inpatient_admissions (silver exception, G9 pending).
    Median required — avg is unreliable due to extreme outliers (max 139d Maternity, Inv 22)."""
    f_val = facility if facility in _VALID_FACILITIES else None
    flt = f"AND source_schema = '{f_val}'" if f_val else ""
    return run_query_df(f"""
        SELECT
            source_schema                              AS facility,
            ward_category,
            DATE_TRUNC('month', ADMITTED_AT)::DATE     AS admission_month,
            MEDIAN(LOS_DAYS)                           AS median_los_days,
            COUNT(*)                                   AS admissions
        FROM HOSPITALS.STAGING.stg_inpatient_admissions
        WHERE LOS_DAYS IS NOT NULL
          AND ADMITTED_AT >= '2000-01-01'
          {flt}
        GROUP BY source_schema, ward_category,
                 DATE_TRUNC('month', ADMITTED_AT)::DATE
        ORDER BY ward_category, admission_month
    """)


def q_ward_discharge_monthly(facility=None):
    """G2: monthly Patient Request discharge rate per ward — for discharge pattern rules (Inv 23)."""
    f = _flt(facility)
    return run_query_df(f"""
        SELECT
            facility,
            ward_category,
            admission_month,
            SUM(total_admissions) AS total_admissions,
            SUM(CASE WHEN discharge_type ILIKE '%patient request%'
                     THEN total_admissions ELSE 0 END) AS patient_request_admissions,
            ROUND(
                100.0 * SUM(CASE WHEN discharge_type ILIKE '%patient request%'
                            THEN total_admissions ELSE 0 END)
                / NULLIF(SUM(total_admissions), 0), 2
            ) AS patient_request_pct
        FROM HOSPITALS.REPORTING.rpt_readmissions
        WHERE ward_category IS NOT NULL
          {f}
        GROUP BY facility, ward_category, admission_month
        ORDER BY facility, ward_category, admission_month
    """)


def q_doctor_workload_monthly():
    """KSH only: monthly evaluation visits per doctor — burnout + concentration rules (Inv 24).
    created_at is VARCHAR — uses TRY_TO_TIMESTAMP. IS_EMPLOYEE not populated for KSH."""
    return run_query_df("""
        SELECT
            DATE_TRUNC('month', TRY_TO_TIMESTAMP(ev.created_at))::DATE AS visit_month,
            u.username,
            COUNT(*) AS monthly_visits
        FROM HOSPITALS.KISUMU_CLEAN.EVALUATION_VISITS ev
        JOIN HOSPITALS.KISUMU_CLEAN.USERS u ON ev.user = u.id
        WHERE ev.deleted_at IS NULL
          AND u.active = 1
          AND u.username NOT REGEXP '.*[0-9].*'
          AND u.username NOT IN ('sudo', 'Billclinton')
          AND TRY_TO_TIMESTAMP(ev.created_at) >= '2024-01-01'
        GROUP BY
            DATE_TRUNC('month', TRY_TO_TIMESTAMP(ev.created_at))::DATE,
            u.username
        ORDER BY visit_month, monthly_visits DESC
    """)


def q_doctor_conversion_monthly():
    """KSH only: monthly evaluation-to-admission conversion rate per doctor.
    Join path: EVALUATION_VISITS.id → INPATIENT_ADMISSIONS.visit_id (confirmed Inv 31 / CD6).
    admitted CTE deduplicates INPATIENT_ADMISSIONS to avoid fan-out (same pattern as q_cd12_monthly_rate).
    Columns: visit_month, username, evaluations, admissions, conversion_rate_pct."""
    return run_query_df("""
        WITH admitted AS (
            SELECT DISTINCT visit_id
            FROM HOSPITALS.KISUMU_CLEAN.INPATIENT_ADMISSIONS
            WHERE ward_name IS NOT NULL
        )
        SELECT
            DATE_TRUNC('month', TRY_TO_TIMESTAMP(ev.created_at))::DATE  AS visit_month,
            u.username,
            COUNT(*)                                                     AS evaluations,
            COUNT(a.visit_id)                                            AS admissions,
            ROUND(COUNT(a.visit_id) * 100.0
                  / NULLIF(COUNT(*), 0), 1)                              AS conversion_rate_pct
        FROM HOSPITALS.KISUMU_CLEAN.EVALUATION_VISITS ev
        JOIN HOSPITALS.KISUMU_CLEAN.USERS u ON ev.user = u.id
        LEFT JOIN admitted a ON a.visit_id = ev.id
        WHERE ev.deleted_at IS NULL
          AND u.active = 1
          AND u.username NOT REGEXP '.*[0-9].*'
          AND u.username NOT IN ('sudo', 'Billclinton')
          AND TRY_TO_TIMESTAMP(ev.created_at) >= '2024-09-01'
        GROUP BY
            DATE_TRUNC('month', TRY_TO_TIMESTAMP(ev.created_at))::DATE,
            u.username
        ORDER BY visit_month, evaluations DESC
    """)


def q_visit_summary():
    """KSH only: monthly total visit count from EVALUATION_VISITS (Inv 27 confirmed table + column).
    created_at is VARCHAR — uses TRY_TO_TIMESTAMP. Filtered to active clean doctors (matches CD11
    methodology — excludes test accounts and inactive staff). Inpatient derived in dashboard from ward_adm."""
    return run_query_df("""
        SELECT
            DATE_TRUNC('month', TRY_TO_TIMESTAMP(ev.created_at))::DATE AS visit_month,
            COUNT(*) AS total_visits
        FROM HOSPITALS.KISUMU_CLEAN.EVALUATION_VISITS ev
        JOIN HOSPITALS.KISUMU_CLEAN.USERS u ON ev.user = u.id
        WHERE ev.deleted_at IS NULL
          AND u.active = 1
          AND u.username NOT REGEXP '.*[0-9].*'
          AND u.username NOT IN ('sudo', 'Billclinton')
          AND TRY_TO_TIMESTAMP(ev.created_at) >= '2024-09-01'
        GROUP BY DATE_TRUNC('month', TRY_TO_TIMESTAMP(ev.created_at))::DATE
        ORDER BY visit_month
    """)


def q_peak_ward_dist():
    """KSH only: ward admission distribution during Monday 14-18h peak vs off-peak (CD5).
    Join: EVALUATION_VISITS (visit time → peak classification) → STG_INPATIENT_ADMISSIONS (WARD_CATEGORY).
    STG used for clean WARD_CATEGORY + proper ADMITTED_AT timestamp. Sep 2024 cutoff.
    Columns: time_bucket, ward_category, admissions."""
    return run_query_df("""
        SELECT
            CASE WHEN DAYOFWEEKISO(TRY_TO_TIMESTAMP(ev.created_at)) = 1
                      AND HOUR(TRY_TO_TIMESTAMP(ev.created_at)) BETWEEN 14 AND 17
                 THEN 'Peak'
                 ELSE 'Off-Peak'
            END                                    AS time_bucket,
            ia.ward_category,
            COUNT(DISTINCT ia.original_id)         AS admissions
        FROM HOSPITALS.KISUMU_CLEAN.EVALUATION_VISITS ev
        JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS ia ON ia.visit_id = ev.id
        WHERE ev.deleted_at IS NULL
          AND ia.ward_category IS NOT NULL
          AND ia.source_schema = 'KISUMU_CLEAN'
          AND TRY_TO_TIMESTAMP(ev.created_at) IS NOT NULL
          AND TRY_TO_TIMESTAMP(ev.created_at) >= '2024-09-01'
        GROUP BY 1, 2
        ORDER BY 1, 3 DESC
    """)


def q_doctor_ward_share():
    """KSH only: doctor share of admissions per ward per month (CD6 — physician concentration).
    Joins INPATIENT_ADMISSIONS → EVALUATION_VISITS → USERS to get username (eawando format).
    INPATIENT_ADMISSIONS.doctor_username is a different field — not the canonical username.
    Sep 2024 cutoff via ADMITTED_AT.
    Columns: admission_month, username, ward_name, admissions."""
    return run_query_df("""
        SELECT
            DATE_TRUNC('month', TRY_TO_TIMESTAMP(ia.admitted_at))::DATE AS admission_month,
            u.username                         AS username,
            ia.ward_name,
            COUNT(DISTINCT ia.id)              AS admissions
        FROM HOSPITALS.KISUMU_CLEAN.INPATIENT_ADMISSIONS ia
        JOIN HOSPITALS.KISUMU_CLEAN.EVALUATION_VISITS ev ON ev.id = ia.visit_id
        JOIN HOSPITALS.KISUMU_CLEAN.USERS u ON u.id = ev.user
        WHERE ia.ward_name IS NOT NULL
          AND ev.deleted_at IS NULL
          AND u.active = 1
          AND u.username NOT REGEXP '.*[0-9].*'
          AND u.username NOT IN ('sudo', 'Billclinton')
          AND TRY_TO_TIMESTAMP(ia.admitted_at) >= '2024-09-01'
        GROUP BY 1, 2, 3
        ORDER BY 4 DESC
    """)


def q_peak_breakdown():
    """KSH only: monthly peak (09-12h) vs off-peak visit counts from EVALUATION_VISITS.
    Peak = hours 9,10,11,12 — confirmed highest volume window (Inv 29 Q2). Sep 2024 cutoff.
    Columns: visit_month, peak_visits, offpeak_visits, total_visits, peak_vs_offpeak_pct."""
    return run_query_df("""
        SELECT
            DATE_TRUNC('month', TRY_TO_TIMESTAMP(created_at))::DATE  AS visit_month,
            SUM(CASE WHEN HOUR(TRY_TO_TIMESTAMP(created_at)) BETWEEN 9 AND 12
                     THEN 1 ELSE 0 END)                              AS peak_visits,
            SUM(CASE WHEN HOUR(TRY_TO_TIMESTAMP(created_at)) NOT BETWEEN 9 AND 12
                     THEN 1 ELSE 0 END)                              AS offpeak_visits,
            COUNT(*)                                                 AS total_visits,
            ROUND(
                SUM(CASE WHEN HOUR(TRY_TO_TIMESTAMP(created_at)) BETWEEN 9 AND 12
                         THEN 1 ELSE 0 END)
                / NULLIF(SUM(CASE WHEN HOUR(TRY_TO_TIMESTAMP(created_at)) NOT BETWEEN 9 AND 12
                                  THEN 1 ELSE 0 END), 0) * 100
            , 1)                                                     AS peak_vs_offpeak_pct
        FROM HOSPITALS.KISUMU_CLEAN.EVALUATION_VISITS
        WHERE deleted_at IS NULL
          AND TRY_TO_TIMESTAMP(created_at) >= '2024-09-01'
          AND TRY_TO_TIMESTAMP(created_at) IS NOT NULL
        GROUP BY DATE_TRUNC('month', TRY_TO_TIMESTAMP(created_at))::DATE
        ORDER BY visit_month
    """)


def q_cd12_monthly_rate():
    """KSH only: monthly critical creatinine non-admission rate (Rule 29 / CD12).
    Source: KISUMU_RAW.EVENTS_RAW. Critical flags stored as HTML strings: CL/CH.
    Data available from Jul 2025 onward (CL/CH flag format introduced mid-2025).
    QUALIFY deduplicates same visit flagged multiple times in a month.
    admitted CTE uses DISTINCT to flatten INPATIENT_ADMISSIONS fan-out.
    Columns: critical_month, total_critical, admitted, not_admitted, non_admission_rate_pct."""
    return run_query_df("""
        WITH critical_cr AS (
            SELECT
                DATE_TRUNC('month', TRY_TO_TIMESTAMP(payload:created_at::STRING))::DATE
                                                             AS critical_month,
                TRY_TO_NUMBER(payload:visit_id::STRING)      AS visit_id
            FROM HOSPITALS.KISUMU_RAW.EVENTS_RAW
            WHERE payload:test::STRING = 'Creatinine'
              AND (   CONTAINS(payload:flag::STRING, '(CL)')
                   OR CONTAINS(payload:flag::STRING, '(CH)'))
              AND TRY_TO_TIMESTAMP(payload:created_at::STRING) >= '2024-09-01'
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY TRY_TO_NUMBER(payload:visit_id::STRING),
                             DATE_TRUNC('month', TRY_TO_TIMESTAMP(payload:created_at::STRING))
                ORDER BY TRY_TO_TIMESTAMP(payload:created_at::STRING)
            ) = 1
        ),
        admitted AS (
            SELECT DISTINCT visit_id
            FROM HOSPITALS.KISUMU_CLEAN.INPATIENT_ADMISSIONS
        )
        SELECT
            cc.critical_month,
            COUNT(DISTINCT cc.visit_id)                                          AS total_critical,
            COUNT(DISTINCT adm.visit_id)                                         AS admitted,
            COUNT(DISTINCT cc.visit_id) - COUNT(DISTINCT adm.visit_id)          AS not_admitted,
            ROUND(
                100.0 * (COUNT(DISTINCT cc.visit_id) - COUNT(DISTINCT adm.visit_id))
                / NULLIF(COUNT(DISTINCT cc.visit_id), 0), 1
            )                                                                    AS non_admission_rate_pct
        FROM critical_cr cc
        LEFT JOIN admitted adm ON adm.visit_id = cc.visit_id
        GROUP BY cc.critical_month
        ORDER BY cc.critical_month
    """)


def q_lab_monthly():
    """KSH only: monthly lab/imaging volume + abnormal rate — for lab rules (Inv 25b).
    Source: KISUMU_RAW.EVENTS_RAW. All fields inside JSON payload: column.
    Reliable from Sep 2024. flag IN ('H','L') = abnormal."""
    return run_query_df("""
        SELECT
            DATE_TRUNC('month', TRY_TO_TIMESTAMP(payload:created_at::STRING))::DATE
                                      AS lab_month,
            COUNT(DISTINCT payload:visit_id::STRING) AS distinct_visits,
            COUNT(*)                  AS total_components,
            SUM(CASE WHEN payload:flag::STRING IN ('H', 'L') THEN 1 ELSE 0 END)
                                      AS abnormal_count,
            ROUND(
                100.0 * SUM(CASE WHEN payload:flag::STRING IN ('H', 'L') THEN 1 ELSE 0 END)
                / NULLIF(COUNT(*), 0), 2
            )                         AS abnormal_pct
        FROM HOSPITALS.KISUMU_RAW.EVENTS_RAW
        WHERE payload:test IS NOT NULL
          AND TRY_TO_TIMESTAMP(payload:created_at::STRING) >= '2024-09-01'
        GROUP BY DATE_TRUNC('month', TRY_TO_TIMESTAMP(payload:created_at::STRING))::DATE
        ORDER BY lab_month
    """)


def q_btr_bti_monthly():
    """KSH only: monthly BTR + BTI + BOR per ward (Rule 29/31 — Inv 46/48).
    BTR  = total_admissions / bed_count.
    BTI  = (available_bed_days - occupied_bed_days) / discharged_admissions.
    BOR  = total_bed_days / (bed_count * days_in_month) * 100.
    Beds: hardcoded per-ward counts (32 total, Inv 54 confirmed 2026-06-18). Do NOT query INPATIENT_BEDS —
    Snowflake inflates to 161 via flatten duplicates + orphaned ward IDs.
    Columns: ward_name, month, total_admissions, discharged_admissions,
             total_bed_days, bed_count, btr, bti_days, bor_pct."""
    return run_query_df("""
        WITH beds AS (
            SELECT ward_name, bed_count
            FROM (VALUES
                ('General Female',    7),
                ('General Maternity', 7),
                ('Pediatric General', 6),
                ('General Male',      4),
                ('Private Male',      3),
                ('Private Female',    3),
                ('Private Maternity', 2)
            ) AS t(ward_name, bed_count)
        ),
        occ AS (
            SELECT
                ward_name,
                admission_month,
                total_admissions,
                discharged_admissions,
                total_bed_days,
                DAY(LAST_DAY(admission_month)) AS days_in_month
            FROM HOSPITALS.REPORTING.rpt_bed_occupancy
            WHERE facility = 'KISUMU_CLEAN'
              AND admission_month >= DATEADD('month', -12, DATE_TRUNC('month', CURRENT_DATE))
              AND admission_month <  DATE_TRUNC('month', CURRENT_DATE)
              AND discharged_admissions > 0
        )
        SELECT
            o.ward_name,
            o.admission_month                                                        AS month,
            o.total_admissions,
            o.discharged_admissions,
            o.total_bed_days,
            b.bed_count,
            ROUND(o.total_admissions / b.bed_count, 2)                              AS btr,
            ROUND(
                (b.bed_count * o.days_in_month - o.total_bed_days)
                / o.discharged_admissions, 1)                                        AS bti_days,
            ROUND(o.total_bed_days / (b.bed_count * o.days_in_month) * 100, 1)     AS bor_pct
        FROM occ o
        JOIN beds b ON o.ward_name = b.ward_name
        ORDER BY o.ward_name, o.admission_month
    """)


def q_admission_tat_bimodal():
    """KSH only: admission TAT by day of week — fast-track % + evaluation volume (B2 / P16-2).
    TAT = minutes from evaluation visit creation to first admission record.
    Dedup CTE required: 97% of INPATIENT_ADMISSIONS visit_ids have multiple rows (C2 confirmed).
    Fast-track: TAT < 60 min. Slow pathway: TAT 60-480 min. Cap at 480 min (>8h = data quality zone).
    visits CTE counts ALL evaluation visits per day (no admission join) — volume context.
    Sep 2024+ data window (reliable per Inv 25b/29). No doctor filter — full population.
    Columns: day_num, day_name, total_admissions, fast_track, slow_pathway, fast_pct,
             p50_tat_min, p75_tat_min, total_evaluations."""
    return run_query_df("""
        WITH ia_dedup AS (
            SELECT
                visit_id,
                MIN(created_at) AS admission_at
            FROM HOSPITALS.KISUMU_CLEAN.INPATIENT_ADMISSIONS
            WHERE DELETED_AT IS NULL
            GROUP BY visit_id
        ),
        tat AS (
            SELECT
                DAYOFWEEK(TRY_TO_TIMESTAMP(ev.CREATED_AT))         AS day_num,
                DAYNAME(TRY_TO_TIMESTAMP(ev.CREATED_AT))           AS day_name,
                DATEDIFF('minute',
                    TRY_TO_TIMESTAMP(ev.CREATED_AT),
                    ia.admission_at)                                AS tat_min
            FROM HOSPITALS.KISUMU_CLEAN.EVALUATION_VISITS ev
            JOIN ia_dedup ia ON ia.visit_id = ev.ID
            WHERE ev.DELETED_AT IS NULL
              AND TRY_TO_TIMESTAMP(ev.CREATED_AT) >= '2024-09-01'
              AND DATEDIFF('minute',
                    TRY_TO_TIMESTAMP(ev.CREATED_AT),
                    ia.admission_at) BETWEEN 1 AND 480
        ),
        visits AS (
            SELECT
                DAYOFWEEK(TRY_TO_TIMESTAMP(created_at))            AS day_num,
                COUNT(*)                                            AS total_evaluations
            FROM HOSPITALS.KISUMU_CLEAN.EVALUATION_VISITS
            WHERE DELETED_AT IS NULL
              AND TRY_TO_TIMESTAMP(created_at) >= '2024-09-01'
            GROUP BY 1
        )
        SELECT
            t.day_num,
            t.day_name,
            COUNT(*)                                                                AS total_admissions,
            SUM(CASE WHEN t.tat_min < 60 THEN 1 ELSE 0 END)                        AS fast_track,
            SUM(CASE WHEN t.tat_min >= 60 THEN 1 ELSE 0 END)                       AS slow_pathway,
            ROUND(SUM(CASE WHEN t.tat_min < 60 THEN 1 ELSE 0 END)
                  * 100.0 / COUNT(*), 1)                                            AS fast_pct,
            ROUND(MEDIAN(t.tat_min), 0)                                             AS p50_tat_min,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY t.tat_min), 0)      AS p75_tat_min,
            v.total_evaluations
        FROM tat t
        JOIN visits v ON v.day_num = t.day_num
        GROUP BY t.day_num, t.day_name, v.total_evaluations
        ORDER BY t.day_num
    """)


def q_admission_tat_monthly():
    """KSH only: monthly fast-track % and p50 TAT for admission TAT alert (Rule 30 — Inv 47).
    TAT = minutes from evaluation visit creation to first inpatient admission record.
    Fast-track: TAT < 60 min. Cap at 480 min. Oct 2024+ window.
    Oct 2025 excluded (pipeline gap). Current partial month excluded.
    Columns: tat_month, total_admissions, fast_track, fast_pct, p50_tat_min."""
    return run_query_df("""
        WITH ia_dedup AS (
            SELECT
                visit_id,
                MIN(created_at) AS admission_at
            FROM HOSPITALS.KISUMU_CLEAN.INPATIENT_ADMISSIONS
            WHERE DELETED_AT IS NULL
            GROUP BY visit_id
        ),
        tat AS (
            SELECT
                DATE_TRUNC('month', TRY_TO_TIMESTAMP(ev.CREATED_AT))::DATE AS tat_month,
                DATEDIFF('minute',
                    TRY_TO_TIMESTAMP(ev.CREATED_AT),
                    ia.admission_at)                                        AS tat_min
            FROM HOSPITALS.KISUMU_CLEAN.EVALUATION_VISITS ev
            JOIN ia_dedup ia ON ia.visit_id = ev.ID
            WHERE ev.DELETED_AT IS NULL
              AND TRY_TO_TIMESTAMP(ev.CREATED_AT) >= '2024-10-01'
              AND TRY_TO_TIMESTAMP(ev.CREATED_AT) <  DATE_TRUNC('month', CURRENT_DATE)
              AND DATE_TRUNC('month', TRY_TO_TIMESTAMP(ev.CREATED_AT))::DATE != '2025-10-01'
              AND DATEDIFF('minute',
                    TRY_TO_TIMESTAMP(ev.CREATED_AT),
                    ia.admission_at) BETWEEN 1 AND 480
        )
        SELECT
            tat_month,
            COUNT(*)                                                                        AS total_admissions,
            SUM(CASE WHEN tat_min < 60 THEN 1 ELSE 0 END)                                  AS fast_track,
            ROUND(SUM(CASE WHEN tat_min < 60 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1)    AS fast_pct,
            ROUND(MEDIAN(tat_min), 0)                                                       AS p50_tat_min
        FROM tat
        GROUP BY tat_month
        ORDER BY tat_month
    """)


def q_revpab_private_monthly():
    """KSH only: monthly combined revenue for Private Female + Male (Rule 32 — Inv 49).
    Private Maternity excluded — sparse volume. Window: last 7 months.
    Columns: admission_month, total_revenue, total_admissions."""
    return run_query_df("""
        SELECT
            admission_month,
            SUM(total_admission_revenue) AS total_revenue,
            SUM(total_admissions)        AS total_admissions
        FROM HOSPITALS.REPORTING.rpt_bed_occupancy
        WHERE facility = 'KISUMU_CLEAN'
          AND ward_name IN ('Private Female', 'Private Male')
          AND admission_month >= DATEADD('month', -7, DATE_TRUNC('month', CURRENT_DATE))
          AND admission_month <  DATE_TRUNC('month', CURRENT_DATE)
          AND admission_month != '2025-10-01'
        GROUP BY admission_month
        ORDER BY admission_month
    """)


def q_peak_tat_conversion():
    """KSH only: conversion rate + TAT during Monday 14-17h peak vs off-peak (Inv 58).
    Conversion = admitted / evaluated. TAT = ev.created_at → first admission. Cap 1–480 min.
    Columns: time_bucket, total_evaluations, admissions, conversion_pct, valid_tat_n,
             p50_tat_min, p75_tat_min."""
    return run_query_df("""
        WITH ia_dedup AS (
            SELECT visit_id, MIN(created_at) AS admission_at
            FROM HOSPITALS.KISUMU_CLEAN.INPATIENT_ADMISSIONS
            WHERE DELETED_AT IS NULL
            GROUP BY visit_id
        ),
        classified AS (
            SELECT
                CASE
                    WHEN DAYOFWEEKISO(TRY_TO_TIMESTAMP(ev.CREATED_AT)) = 1
                     AND HOUR(TRY_TO_TIMESTAMP(ev.CREATED_AT)) BETWEEN 14 AND 17
                    THEN 'Peak (Mon 14-17h)'
                    ELSE 'Off-Peak'
                END                                                         AS time_bucket,
                ia.admission_at IS NOT NULL                                 AS admitted,
                DATEDIFF('minute',
                    TRY_TO_TIMESTAMP(ev.CREATED_AT),
                    ia.admission_at)                                        AS tat_raw
            FROM HOSPITALS.KISUMU_CLEAN.EVALUATION_VISITS ev
            LEFT JOIN ia_dedup ia ON ia.visit_id = ev.ID
            WHERE ev.DELETED_AT IS NULL
              AND TRY_TO_TIMESTAMP(ev.CREATED_AT) >= '2024-09-01'
        ),
        tat_filtered AS (
            SELECT
                time_bucket,
                admitted,
                CASE WHEN admitted AND tat_raw BETWEEN 1 AND 480 THEN tat_raw END AS tat_min
            FROM classified
        )
        SELECT
            time_bucket,
            COUNT(*)                                                        AS total_evaluations,
            SUM(CASE WHEN admitted THEN 1 ELSE 0 END)                       AS admissions,
            ROUND(SUM(CASE WHEN admitted THEN 1 ELSE 0 END) * 100.0
                  / COUNT(*), 1)                                            AS conversion_pct,
            COUNT(tat_min)                                                  AS valid_tat_n,
            ROUND(MEDIAN(tat_min), 0)                                       AS p50_tat_min,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY tat_min), 0) AS p75_tat_min
        FROM tat_filtered
        GROUP BY time_bucket
        ORDER BY time_bucket
    """)


def q_peak_doctor_load():
    """KSH only: doctor evaluation share during Monday 14-17h peak vs off-peak (Inv 58).
    Columns: time_bucket, username, evaluations, pct_of_bucket."""
    return run_query_df("""
        WITH classified AS (
            SELECT
                u.username,
                CASE
                    WHEN DAYOFWEEKISO(TRY_TO_TIMESTAMP(ev.CREATED_AT)) = 1
                     AND HOUR(TRY_TO_TIMESTAMP(ev.CREATED_AT)) BETWEEN 14 AND 17
                    THEN 'Peak (Mon 14-17h)'
                    ELSE 'Off-Peak'
                END AS time_bucket
            FROM HOSPITALS.KISUMU_CLEAN.EVALUATION_VISITS ev
            JOIN HOSPITALS.KISUMU_CLEAN.USERS u ON ev.user = u.id
            WHERE ev.DELETED_AT IS NULL
              AND u.active = 1
              AND u.username NOT REGEXP '.*[0-9].*'
              AND u.username NOT IN ('sudo', 'Billclinton')
              AND TRY_TO_TIMESTAMP(ev.CREATED_AT) >= '2024-09-01'
        )
        SELECT
            time_bucket,
            username,
            COUNT(*)                                                        AS evaluations,
            ROUND(COUNT(*) * 100.0
                  / SUM(COUNT(*)) OVER (PARTITION BY time_bucket), 1)      AS pct_of_bucket
        FROM classified
        GROUP BY time_bucket, username
        ORDER BY time_bucket, evaluations DESC
    """)


def q_peak_patient_funnel():
    """KSH only: non-admitted patient return pathway after Monday 14-17h peak (Inv 58).
    Cohort: Sep 2024–present. Columns: total_non_admitted_peak, returned, never_returned,
    return_pct, later_admitted, admitted_of_returned_pct, median_days_to_return."""
    return run_query_df("""
        WITH peak_evals AS (
            SELECT
                ev.ID                                   AS visit_id,
                ev.PATIENT                              AS patient_id,
                TRY_TO_TIMESTAMP(ev.CREATED_AT)         AS visit_time
            FROM HOSPITALS.KISUMU_CLEAN.EVALUATION_VISITS ev
            WHERE ev.DELETED_AT IS NULL
              AND DAYOFWEEKISO(TRY_TO_TIMESTAMP(ev.CREATED_AT)) = 1
              AND HOUR(TRY_TO_TIMESTAMP(ev.CREATED_AT)) BETWEEN 14 AND 17
              AND TRY_TO_TIMESTAMP(ev.CREATED_AT) >= '2024-09-01'
        ),
        peak_non_admitted AS (
            SELECT pe.*
            FROM peak_evals pe
            WHERE NOT EXISTS (
                SELECT 1 FROM HOSPITALS.KISUMU_CLEAN.INPATIENT_ADMISSIONS ia
                WHERE ia.visit_id = pe.visit_id AND ia.DELETED_AT IS NULL
            )
        ),
        return_visits AS (
            SELECT
                pna.patient_id,
                pna.visit_time                              AS index_time,
                MIN(TRY_TO_TIMESTAMP(ev2.CREATED_AT))       AS next_visit_time
            FROM peak_non_admitted pna
            JOIN HOSPITALS.KISUMU_CLEAN.EVALUATION_VISITS ev2
                ON ev2.PATIENT = pna.patient_id
               AND TRY_TO_TIMESTAMP(ev2.CREATED_AT) > pna.visit_time
               AND ev2.DELETED_AT IS NULL
            GROUP BY pna.patient_id, pna.visit_time
        ),
        later_admissions AS (
            SELECT DISTINCT pna.patient_id
            FROM peak_non_admitted pna
            JOIN HOSPITALS.KISUMU_CLEAN.EVALUATION_VISITS ev2
                ON ev2.PATIENT = pna.patient_id
               AND TRY_TO_TIMESTAMP(ev2.CREATED_AT) > pna.visit_time
               AND ev2.DELETED_AT IS NULL
            JOIN HOSPITALS.KISUMU_CLEAN.INPATIENT_ADMISSIONS ia2
                ON ia2.visit_id = ev2.ID
               AND ia2.DELETED_AT IS NULL
        )
        SELECT
            COUNT(DISTINCT pna.patient_id)                                          AS total_non_admitted_peak,
            COUNT(DISTINCT rv.patient_id)                                           AS returned,
            COUNT(DISTINCT pna.patient_id) - COUNT(DISTINCT rv.patient_id)         AS never_returned,
            ROUND(COUNT(DISTINCT rv.patient_id) * 100.0
                  / NULLIF(COUNT(DISTINCT pna.patient_id), 0), 1)                  AS return_pct,
            COUNT(DISTINCT la.patient_id)                                           AS later_admitted,
            ROUND(COUNT(DISTINCT la.patient_id) * 100.0
                  / NULLIF(COUNT(DISTINCT rv.patient_id), 0), 1)                   AS admitted_of_returned_pct,
            ROUND(MEDIAN(DATEDIFF('day', rv.index_time, rv.next_visit_time)), 0)   AS median_days_to_return
        FROM peak_non_admitted pna
        LEFT JOIN return_visits rv    ON rv.patient_id = pna.patient_id
        LEFT JOIN later_admissions la ON la.patient_id = pna.patient_id
    """)
