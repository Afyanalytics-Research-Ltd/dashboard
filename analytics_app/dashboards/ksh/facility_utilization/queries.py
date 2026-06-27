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
    """G3: monthly completion rate + revenue + payer mix — KSH only."""
    return run_query_df("""
        SELECT
            session_month,
            SUM(total_sessions)     AS total_sessions,
            SUM(completed_sessions) AS completed_sessions,
            ROUND(100.0 * SUM(completed_sessions)
                  / NULLIF(SUM(total_sessions), 0), 2) AS completion_rate_pct,
            SUM(total_revenue)      AS total_revenue,
            SUM(emergency_sessions) AS emergency_sessions,
            SUM(insured_revenue)    AS insured_revenue,
            SUM(cash_revenue)       AS cash_revenue
        FROM HOSPITALS.REPORTING.rpt_theatre_utilization
        GROUP BY session_month
        ORDER BY session_month
    """)


def q_theatre_by_type():
    """G3: per-theatre summary — KSH only. Inv 76 confirmed revenue cols are FLOAT."""
    return run_query_df("""
        SELECT
            theatre_name,
            SUM(total_sessions)                                              AS total_sessions,
            SUM(completed_sessions)                                          AS completed_sessions,
            ROUND(100.0 * SUM(completed_sessions)
                  / NULLIF(SUM(total_sessions), 0), 1)                      AS completion_rate_pct,
            SUM(emergency_sessions)                                          AS emergency_sessions,
            SUM(elective_sessions)                                           AS elective_sessions,
            ROUND(SUM(emergency_sessions) * 100.0
                  / NULLIF(SUM(emergency_sessions) + SUM(elective_sessions), 0), 1)
                                                                             AS emergency_pct,
            SUM(total_revenue)                                               AS total_revenue,
            ROUND(SUM(total_revenue) / NULLIF(SUM(completed_sessions), 0), 0)
                                                                             AS avg_rev_per_completed,
            SUM(insured_revenue)                                             AS insured_revenue,
            SUM(cash_revenue)                                                AS cash_revenue,
            ROUND(SUM(insured_revenue) * 100.0
                  / NULLIF(SUM(total_revenue), 0), 1)                       AS insured_pct,
            MIN(session_month)                                               AS first_month,
            MAX(session_month)                                               AS last_month
        FROM HOSPITALS.REPORTING.rpt_theatre_utilization
        WHERE booking_status = 'booked'
        GROUP BY theatre_name
        ORDER BY total_sessions DESC
    """)


def q_theatre_procedures():
    """All-time procedure booking counts — KSH only. Source: rpt_theatre_case_mix (gold).
    Note: 11 bookings with orphaned procedure IDs excluded (no match in EVALUATION_PROCEDURES).
    Unscheduled bookings also excluded (no session_month) — only cases that reached theatre.
    """
    return run_query_df("""
        SELECT
            procedure_name,
            SUM(bookings) AS bookings
        FROM HOSPITALS.REPORTING.rpt_theatre_case_mix
        GROUP BY procedure_name
        ORDER BY bookings DESC
        LIMIT 20
    """)


def q_theatre_trend_by_theatre():
    """Monthly revenue + sessions per theatre from gold table — KSH only.
    Used for MoM revenue comparison. Revenue cols are FLOAT (Inv 76) — no TRY_TO_NUMBER().
    Jan 2025 onwards — matches q_theatre_procedures_monthly() window.
    """
    return run_query_df("""
        SELECT
            session_month,
            theatre_name,
            SUM(total_sessions)                                              AS total_sessions,
            SUM(completed_sessions)                                          AS completed_sessions,
            ROUND(SUM(total_revenue) / 1000, 0)                             AS total_revenue_kes_k,
            ROUND(SUM(total_revenue) / NULLIF(SUM(completed_sessions), 0), 0)
                                                                             AS avg_rev_per_completed
        FROM HOSPITALS.REPORTING.rpt_theatre_utilization
        WHERE booking_status = 'booked'
          AND session_month  >= '2025-01-01'
        GROUP BY session_month, theatre_name
        ORDER BY session_month DESC, theatre_name
    """)


def q_theatre_procedures_monthly():
    """Monthly procedure booking counts per theatre — KSH only. Source: rpt_theatre_case_mix (gold).
    Jan 2025+ window applied at query time — gold table covers all scheduled sessions.
    """
    return run_query_df("""
        SELECT
            session_month,
            theatre_name,
            procedure_name,
            bookings
        FROM HOSPITALS.REPORTING.rpt_theatre_case_mix
        WHERE session_month >= '2025-01-01'
        ORDER BY session_month DESC, bookings DESC
    """)


def q_theatre_emergency_tat():
    """Raw emergency booking-to-theatre lags (minutes) — KSH only. Source: rpt_theatre_emergency_tat (gold).
    One row per positive-TAT case (n=55). Binning and aggregation done in Python.
    Column renamed booking_to_start_min → tat_min in gold; dashboard must use tat_min.
    """
    return run_query_df("""
        SELECT tat_min AS booking_to_start_min, declaration_day
        FROM HOSPITALS.REPORTING.rpt_theatre_emergency_tat
        ORDER BY tat_min
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


def q_dialysis_ops_monthly():
    """KSH only: monthly dialysis throughput + utilisation. Source: rpt_dialysis_ops (gold).
    Silver stg_dialysis_billing dedups both FINANCE_INVOICE_ITEMS and FINANCE_INVOICES
    (both had exact-duplicate rows from ingestion — Inv 80).
    Columns unchanged: invoice_month, sessions_billed, sessions_insured, sessions_cash,
    session_fee_revenue, total_dialysis_revenue, ancillary_revenue,
    avg_session_fee, utilisation_pct_theoretical, is_partial_month.
    """
    return run_query_df("""
        SELECT
            invoice_month, sessions_billed, sessions_insured, sessions_cash,
            session_fee_revenue, total_dialysis_revenue, ancillary_revenue,
            avg_session_fee, utilisation_pct_theoretical, is_partial_month
        FROM HOSPITALS.REPORTING.rpt_dialysis_ops
        ORDER BY invoice_month
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
    Source: rpt_imaging_ops (gold). Grain: revenue_month x facility x modality.
    """
    fac_filter = f"AND facility = '{facility}'" if facility in _VALID_FACILITIES else ""
    return run_query_df(f"""
        SELECT revenue_month, modality, sessions, revenue, avg_per_session
        FROM HOSPITALS.REPORTING.rpt_imaging_ops
        WHERE 1=1 {fac_filter}
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
    """Monthly median LOS per ward. Source: rpt_ward_los (gold). Silver exception removed.
    Median used — avg unreliable due to extreme outliers (max 139d Maternity, Inv 22)."""
    f_val = facility if facility in _VALID_FACILITIES else None
    flt = f"AND facility = '{f_val}'" if f_val else ""
    return run_query_df(f"""
        SELECT facility, ward_category, admission_month, median_los_days, admissions
        FROM HOSPITALS.REPORTING.rpt_ward_los
        WHERE 1=1 {flt}
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
    Source: rpt_doctor_performance (gold). Silver: STG_EVAL_VISITS."""
    return run_query_df("""
        SELECT
            visit_month,
            username,
            evaluations AS monthly_visits
        FROM HOSPITALS.REPORTING.rpt_doctor_performance
        WHERE visit_month >= '2024-01-01'
        ORDER BY visit_month, monthly_visits DESC
    """)


def q_doctor_conversion_monthly():
    """KSH only: monthly evaluation-to-admission conversion rate per doctor.
    Source: rpt_doctor_performance (gold). Silver: STG_EVAL_VISITS."""
    return run_query_df("""
        SELECT
            visit_month,
            username,
            evaluations,
            admissions,
            conversion_rate_pct
        FROM HOSPITALS.REPORTING.rpt_doctor_performance
        WHERE visit_month >= '2024-09-01'
        ORDER BY visit_month, evaluations DESC
    """)


def q_visit_summary():
    """KSH only: monthly total visit count across all tracked doctors.
    Source: rpt_doctor_performance (gold). Silver: STG_EVAL_VISITS."""
    return run_query_df("""
        SELECT
            visit_month,
            SUM(evaluations) AS total_visits
        FROM HOSPITALS.REPORTING.rpt_doctor_performance
        WHERE visit_month >= '2024-09-01'
        GROUP BY visit_month
        ORDER BY visit_month
    """)


def q_peak_ward_dist():
    """KSH only: ward admission distribution during Monday 14-17h peak vs off-peak (CD5).
    Source: rpt_peak_performance (gold). time_bucket = Mon 14-17h vs Off-Peak.
    Columns: time_bucket, ward_category, admissions."""
    return run_query_df("""
        SELECT
            time_bucket,
            ward_category,
            COUNT(visit_id) AS admissions
        FROM HOSPITALS.REPORTING.rpt_peak_performance
        WHERE ward_category IS NOT NULL
        GROUP BY time_bucket, ward_category
        ORDER BY time_bucket, admissions DESC
    """)


def q_doctor_ward_share():
    """KSH only: doctor share of admissions per ward per month (CD6 — physician concentration).
    Source: rpt_peak_performance (gold). admission_month from ADMITTED_AT in silver.
    Columns: admission_month, username, ward_name, admissions."""
    return run_query_df("""
        SELECT
            admission_month,
            username,
            ward_name,
            COUNT(visit_id) AS admissions
        FROM HOSPITALS.REPORTING.rpt_peak_performance
        WHERE ward_name IS NOT NULL
          AND admission_month IS NOT NULL
        GROUP BY admission_month, username, ward_name
        ORDER BY admissions DESC
    """)


def q_peak_breakdown():
    """KSH only: monthly peak (09-12h) vs off-peak visit counts. Source: rpt_peak_performance (gold).
    Peak = hours 9,10,11,12 — confirmed highest volume window (Inv 29 Q2).
    Columns: visit_month, peak_visits, offpeak_visits, total_visits, peak_vs_offpeak_pct."""
    return run_query_df("""
        SELECT
            visit_month,
            SUM(CASE WHEN visit_hour BETWEEN 9 AND 12 THEN 1 ELSE 0 END)     AS peak_visits,
            SUM(CASE WHEN visit_hour NOT BETWEEN 9 AND 12 THEN 1 ELSE 0 END)  AS offpeak_visits,
            COUNT(*)                                                           AS total_visits,
            ROUND(
                SUM(CASE WHEN visit_hour BETWEEN 9 AND 12 THEN 1 ELSE 0 END)
                / NULLIF(SUM(CASE WHEN visit_hour NOT BETWEEN 9 AND 12 THEN 1 ELSE 0 END), 0) * 100
            , 1)                                                               AS peak_vs_offpeak_pct
        FROM HOSPITALS.REPORTING.rpt_peak_performance
        GROUP BY visit_month
        ORDER BY visit_month
    """)


def q_cd12_monthly_rate():
    """KSH only: monthly critical creatinine non-admission rate (Rule 29 / CD12).
    Source: rpt_cd12_monthly_rate (gold). Raw exception resolved 2026-06-27.
    Chain: KISUMU_RAW.EVENTS_RAW → KISUMU_CLEAN.EVENTS_LAB_FLAGS → stg_lab_events → gold.
    Data available from Jul 2025 onward (CL/CH flag format introduced mid-2025).
    Columns: critical_month, total_critical, admitted, not_admitted, non_admission_rate_pct."""
    return run_query_df("""
        SELECT
            critical_month, total_critical, admitted, not_admitted, non_admission_rate_pct
        FROM HOSPITALS.REPORTING.rpt_cd12_monthly_rate
        ORDER BY critical_month
    """)


def q_lab_monthly():
    """KSH only: monthly lab/imaging volume + abnormal rate — for lab rules (Inv 25b).
    Source: rpt_lab_monthly (gold). Raw exception resolved 2026-06-27.
    Chain: KISUMU_RAW.EVENTS_RAW → KISUMU_CLEAN.EVENTS_LAB_FLAGS → stg_lab_events → gold.
    Columns: lab_month, distinct_visits, total_components, abnormal_count, abnormal_pct."""
    return run_query_df("""
        SELECT lab_month, distinct_visits, total_components, abnormal_count, abnormal_pct
        FROM HOSPITALS.REPORTING.rpt_lab_monthly
        ORDER BY lab_month
    """)


def q_btr_bti_monthly():
    """KSH only: monthly BTR + BTI + BOR per ward (A2 / P16-6 Ward Turnover Efficiency).
    BTR  = total_admissions / bed_count  (admissions per available bed per month).
    BTI  = (available_bed_days - occupied_bed_days) / discharged_admissions  (avg empty days between admissions).
    BOR  = total_bed_days / (bed_count * days_in_month) * 100.
    Beds: hardcoded per-ward counts (32 total, Inv 54 confirmed 2026-06-18). Do NOT query INPATIENT_BEDS —
    Snowflake inflates to 161 via flatten duplicates + orphaned ward IDs. Partial months excluded (< current
    month). Oct 2025 pipeline gap noted (Inv 32).
    Columns: ward_name, month, total_admissions, discharged_admissions, total_bed_days, bed_count, btr, bti_days, bor_pct."""
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
    Source: rpt_admission_tat_bimodal (gold). Silver exception resolved 2026-06-27.
    Columns: day_num, day_name, total_admissions, fast_track, slow_pathway, fast_pct,
             p50_tat_min, p75_tat_min, total_evaluations."""
    return run_query_df("""
        SELECT
            day_num, day_name, total_admissions, fast_track, slow_pathway,
            fast_pct, p50_tat_min, p75_tat_min, total_evaluations
        FROM HOSPITALS.REPORTING.rpt_admission_tat_bimodal
        ORDER BY day_num
    """)


def q_admission_tat_monthly():
    """KSH only: monthly fast-track % and p50/p75 TAT for admission TAT alert (Rule 30 / Inv 47).
    Source: rpt_admission_tat (gold). Silver: stg_admission_tat.
    Columns: tat_month, total_admissions, fast_track, fast_pct, p50_tat_min, p75_tat_min."""
    return run_query_df("""
        SELECT
            tat_month,
            total_admissions,
            fast_track,
            fast_pct,
            p50_tat_min,
            p75_tat_min
        FROM HOSPITALS.REPORTING.rpt_admission_tat
        ORDER BY tat_month
    """)


def q_revpab_private_monthly():
    """KSH only: monthly combined revenue for Private Female + Male (Rule 32 / Inv 49).
    Metric: total_admission_revenue combined across both wards per month.
    Private Maternity excluded — admission volume too sparse for a stable rolling baseline.
    Source: rpt_bed_occupancy gold table (same as BTR/BTI/BOR).
    Window: last 7 months to guarantee 4+ months after Oct 2025 exclusion.
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
    Source: rpt_peak_performance (gold). tat_min from stg_admission_tat (1-480 min range).
    Columns: time_bucket, total_evaluations, admissions, conversion_pct, valid_tat_n,
             p50_tat_min, p75_tat_min."""
    return run_query_df("""
        SELECT
            time_bucket,
            COUNT(*)                                                         AS total_evaluations,
            SUM(CASE WHEN is_admitted THEN 1 ELSE 0 END)                     AS admissions,
            ROUND(SUM(CASE WHEN is_admitted THEN 1 ELSE 0 END) * 100.0
                  / COUNT(*), 1)                                             AS conversion_pct,
            COUNT(tat_min)                                                   AS valid_tat_n,
            ROUND(MEDIAN(tat_min), 0)                                        AS p50_tat_min,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY tat_min), 0)  AS p75_tat_min
        FROM HOSPITALS.REPORTING.rpt_peak_performance
        GROUP BY time_bucket
        ORDER BY time_bucket
    """)


def q_peak_doctor_load():
    """KSH only: doctor evaluation share during Monday 14-17h peak vs off-peak (Inv 58).
    Source: rpt_peak_performance (gold).
    Columns: time_bucket, username, evaluations, pct_of_bucket."""
    return run_query_df("""
        SELECT
            time_bucket,
            username,
            COUNT(*)                                                         AS evaluations,
            ROUND(COUNT(*) * 100.0
                  / SUM(COUNT(*)) OVER (PARTITION BY time_bucket), 1)       AS pct_of_bucket
        FROM HOSPITALS.REPORTING.rpt_peak_performance
        GROUP BY time_bucket, username
        ORDER BY time_bucket, evaluations DESC
    """)


def q_peak_patient_funnel():
    """KSH only: non-admitted patient return pathway after Monday 14-17h peak (Inv 58).
    Source: rpt_patient_return_funnel (gold). Single aggregate row — funnel pre-computed.
    Columns: total_non_admitted_peak, returned, never_returned, return_pct,
             later_admitted, admitted_of_returned_pct, median_days_to_return."""
    return run_query_df("""
        SELECT * FROM HOSPITALS.REPORTING.rpt_patient_return_funnel
    """)


def q_data_freshness():
    """Last recorded date per facility — used in sidebar data range label.
    KSH: MAX(visit_ts) from STG_EVAL_VISITS (visit-level, actual date).
    TENRI: MAX(ADMISSION_MONTH) from rpt_bed_occupancy (month-grain, first of month).
    Returns one row per facility with max_date as a calendar date."""
    return run_query_df("""
        SELECT 'KISUMU_CLEAN' AS facility, MAX(visit_ts)::DATE AS max_date
        FROM HOSPITALS.STAGING.STG_EVAL_VISITS
        UNION ALL
        SELECT 'TENRI', MAX(ADMISSION_MONTH)::DATE
        FROM HOSPITALS.REPORTING.rpt_bed_occupancy
        WHERE facility = 'TENRI'
    """)
