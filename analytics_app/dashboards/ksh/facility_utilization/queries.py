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
    """G3: monthly completion rate + revenue + payer mix — KSH only.
    Theatre-side partial months stripped via stg_theatre_utilization freshness check
    (Inv 137: KSH theatre schedule data stops mid-month at the same pipeline cutoff
    as inpatient and billing — April 2026 stops day 21). Any session_month where
    max schedule_start_time day < 25 is excluded from the result."""
    return run_query_df("""
        WITH theatre_freshness AS (
            SELECT
                DATE_TRUNC('month', schedule_start_time::DATE)::DATE AS session_month,
                MAX(DAY(schedule_start_time::DATE))                  AS max_day_in_month
            FROM HOSPITALS.STAGING.stg_theatre_utilization
            WHERE facility = 'KISUMU_CLEAN'
            GROUP BY DATE_TRUNC('month', schedule_start_time::DATE)::DATE
        ),
        complete_months AS (
            SELECT session_month
            FROM theatre_freshness
            WHERE max_day_in_month >= 25
        )
        SELECT
            r.session_month,
            SUM(r.total_sessions)     AS total_sessions,
            SUM(r.completed_sessions) AS completed_sessions,
            ROUND(100.0 * SUM(r.completed_sessions)
                  / NULLIF(SUM(r.total_sessions), 0), 2) AS completion_rate_pct,
            SUM(r.total_revenue)      AS total_revenue,
            SUM(r.emergency_sessions) AS emergency_sessions,
            SUM(r.insured_revenue)    AS insured_revenue,
            SUM(r.cash_revenue)       AS cash_revenue
        FROM HOSPITALS.REPORTING.rpt_theatre_utilization r
        JOIN complete_months cm ON cm.session_month = r.session_month
        GROUP BY r.session_month
        ORDER BY r.session_month
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
            SUM(total_sessions) AS bookings
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
            total_sessions AS bookings
        FROM HOSPITALS.REPORTING.rpt_theatre_case_mix
        WHERE session_month >= '2025-01-01'
        ORDER BY session_month DESC, total_sessions DESC
    """)


def q_theatre_cur_month_by_theatre():
    """Per-theatre totals for penultimate month — from rpt_theatre_case_mix (validated source).
    Uses second-to-last month because the latest month may be incomplete in EMR."""
    return run_query_df("""
        WITH cur AS (
            SELECT MAX(session_month) AS latest
            FROM HOSPITALS.REPORTING.rpt_theatre_case_mix
            WHERE session_month < (SELECT MAX(session_month) FROM HOSPITALS.REPORTING.rpt_theatre_case_mix)
        )
        SELECT
            cm.theatre_name,
            SUM(cm.total_sessions)         AS total_sessions,
            SUM(cm.completed_sessions)     AS completed_sessions,
            SUM(cm.non_completed_sessions) AS non_completed_sessions,
            ROUND(100.0 * SUM(cm.completed_sessions)
                  / NULLIF(SUM(cm.total_sessions), 0), 1) AS completion_pct
        FROM HOSPITALS.REPORTING.rpt_theatre_case_mix cm
        CROSS JOIN cur
        WHERE cm.session_month = cur.latest
        GROUP BY cm.theatre_name
        ORDER BY non_completed_sessions DESC
    """)


def q_theatre_procedure_rates():
    """Penultimate month all-procedure completion rates — completed and failed.
    Uses second-to-last month because the latest month may be incomplete in EMR."""
    return run_query_df("""
        WITH cur AS (
            SELECT MAX(session_month) AS latest
            FROM HOSPITALS.REPORTING.rpt_theatre_case_mix
            WHERE session_month < (SELECT MAX(session_month) FROM HOSPITALS.REPORTING.rpt_theatre_case_mix)
        )
        SELECT
            cm.procedure_name,
            SUM(cm.total_sessions)         AS total_sessions,
            SUM(cm.completed_sessions)     AS completed_sessions,
            SUM(cm.non_completed_sessions) AS non_completed_sessions,
            ROUND(100.0 * SUM(cm.completed_sessions)
                  / NULLIF(SUM(cm.total_sessions), 0), 1) AS completion_pct
        FROM HOSPITALS.REPORTING.rpt_theatre_case_mix cm
        CROSS JOIN cur
        WHERE cm.session_month = cur.latest
          AND cm.procedure_name != '(no procedure recorded)'
        GROUP BY cm.procedure_name
        ORDER BY non_completed_sessions DESC, total_sessions DESC
    """)


def q_theatre_non_completion():
    """Non-completed procedures for current month + all-time avg revenue — KSH only.
    Source: rpt_theatre_case_mix (extended gold). All-time weighted avg used for
    revenue exposure to avoid instability from low monthly session counts.
    Returns one row per procedure with non_completed > 0 in the latest session_month.
    """
    return run_query_df("""
        WITH all_time_avg AS (
            SELECT
                procedure_name,
                SUM(completed_sessions * COALESCE(avg_revenue_completed_kes, 0))
                    / NULLIF(SUM(completed_sessions), 0)    AS avg_revenue_kes
            FROM HOSPITALS.REPORTING.rpt_theatre_case_mix
            WHERE avg_revenue_completed_kes IS NOT NULL
              AND avg_revenue_completed_kes > 0
              AND completed_sessions        > 0
            GROUP BY procedure_name
        ),
        cur_month AS (
            SELECT MAX(session_month) AS latest FROM HOSPITALS.REPORTING.rpt_theatre_case_mix
            WHERE session_month < (SELECT MAX(session_month) FROM HOSPITALS.REPORTING.rpt_theatre_case_mix)
        ),
        non_completed AS (
            SELECT
                cm.procedure_name,
                cm.theatre_name,
                cm.non_completed_sessions,
                cm.cash_non_completed,
                cm.insured_non_completed
            FROM HOSPITALS.REPORTING.rpt_theatre_case_mix cm
            CROSS JOIN cur_month c
            WHERE cm.session_month        = c.latest
              AND cm.non_completed_sessions > 0
        )
        SELECT
            nc.procedure_name,
            nc.theatre_name,
            nc.non_completed_sessions,
            nc.cash_non_completed,
            nc.insured_non_completed,
            ROUND(COALESCE(a.avg_revenue_kes, 0), 0)                              AS avg_revenue_kes,
            ROUND(nc.non_completed_sessions * COALESCE(a.avg_revenue_kes, 0), 0)  AS revenue_exposure_kes
        FROM non_completed nc
        LEFT JOIN all_time_avg a ON a.procedure_name = nc.procedure_name
        ORDER BY revenue_exposure_kes DESC NULLS LAST
    """)


def q_theatre_status_breakdown():
    """Booking status breakdown for current month — KSH only.
    Source: rpt_theatre_utilization. Shows how non-completions distribute
    across booking lifecycle stages (approved / booked / pending etc).
    """
    return run_query_df("""
        WITH cur_month AS (
            SELECT MAX(session_month) AS latest
            FROM HOSPITALS.REPORTING.rpt_theatre_utilization
            WHERE facility = 'KISUMU_CLEAN'
              AND session_month < (
                  SELECT MAX(session_month) FROM HOSPITALS.REPORTING.rpt_theatre_utilization
                  WHERE facility = 'KISUMU_CLEAN'
              )
        )
        SELECT
            tu.booking_status,
            SUM(tu.total_sessions)                                                          AS total_sessions,
            SUM(tu.completed_sessions)                                                      AS completed_sessions,
            SUM(tu.total_sessions) - SUM(tu.completed_sessions)                             AS non_completed,
            ROUND(100.0 * SUM(tu.completed_sessions)
                  / NULLIF(SUM(tu.total_sessions), 0), 1)                                   AS completion_pct
        FROM HOSPITALS.REPORTING.rpt_theatre_utilization tu
        CROSS JOIN cur_month c
        WHERE tu.facility      = 'KISUMU_CLEAN'
          AND tu.session_month = c.latest
        GROUP BY tu.booking_status
        ORDER BY total_sessions DESC
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
    """G1: monthly bed days + admissions + LOS + revenue per ward_name — KSH only.
    Grain: ward_name × admission_month. ward_category added for operational chain grouping.
    Revenue = ward accommodation fee only (KES 6,000/12,000 flat) — excludes procedures/ancillary.
    Inv 81: revenue columns added 2026-06-29 after fan-out + sufficiency validation."""
    return run_query_df("""
        SELECT
            ward_category,
            ward_name,
            admission_month,
            SUM(total_admissions)                                                AS total_admissions,
            SUM(discharged_admissions)                                           AS discharged_admissions,
            SUM(total_bed_days)                                                  AS total_bed_days,
            ROUND(SUM(total_bed_days)
                  / NULLIF(SUM(discharged_admissions), 0), 2)                   AS avg_los_days,
            SUM(total_admission_revenue)                                         AS total_admission_revenue,
            ROUND(SUM(total_admission_revenue)
                  / NULLIF(SUM(total_bed_days), 0), 2)                          AS revpab
        FROM HOSPITALS.REPORTING.rpt_bed_occupancy
        WHERE facility = 'KISUMU_CLEAN'
          AND ward_name IS NOT NULL
        GROUP BY ward_category, ward_name, admission_month
        ORDER BY ward_category, ward_name, admission_month
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
    Billing-side partial months stripped via stg_procedure_revenue freshness check
    (Inv 136: KSH billing data stops mid-month independently of OPD cutoff).
    Any revenue_month where max invoice_date day < 25 is excluded.
    """
    fac_filter = f"AND i.facility = '{facility}'" if facility in _VALID_FACILITIES else ""
    fac_fresh  = f"AND facility = '{facility}'" if facility in _VALID_FACILITIES else ""
    return run_query_df(f"""
        WITH billing_freshness AS (
            SELECT
                DATE_TRUNC('month', invoice_date::DATE)::DATE AS revenue_month,
                MAX(DAY(invoice_date::DATE))                  AS max_day_in_month
            FROM HOSPITALS.STAGING.stg_procedure_revenue
            WHERE 1=1 {fac_fresh}
            GROUP BY DATE_TRUNC('month', invoice_date::DATE)::DATE
        ),
        complete_months AS (
            SELECT revenue_month
            FROM billing_freshness
            WHERE max_day_in_month >= 25
        )
        SELECT i.revenue_month, i.modality, i.sessions, i.revenue, i.avg_per_session
        FROM HOSPITALS.REPORTING.rpt_imaging_ops i
        JOIN complete_months cm ON cm.revenue_month = i.revenue_month
        WHERE 1=1 {fac_filter}
        ORDER BY i.revenue_month, i.revenue DESC
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


def q_patient_flow_transitions():
    """KSH only: stage-to-stage patient transition times from RECEPTION_TIME_TRACKERS.
    Uses TIME_IN only (TIME_OUT near-universally null). Gap = next stage TIME_IN - current TIME_IN.
    Corridor stages: Reception, doctor, laboratory, radiology, pharmacy. Inv 98.
    Columns: stage, next_stage, transitions, median_gap_min, p75_gap_min."""
    return run_query_df("""
        WITH ordered AS (
            SELECT
                VISIT_ID,
                DESTINATION_NAME                                        AS stage,
                TRY_TO_TIMESTAMP(TIME_IN)                              AS t_in,
                LEAD(TRY_TO_TIMESTAMP(TIME_IN)) OVER (
                    PARTITION BY VISIT_ID
                    ORDER BY TRY_TO_TIMESTAMP(TIME_IN)
                )                                                       AS next_t_in,
                LEAD(DESTINATION_NAME) OVER (
                    PARTITION BY VISIT_ID
                    ORDER BY TRY_TO_TIMESTAMP(TIME_IN)
                )                                                       AS next_stage
            FROM HOSPITALS.KISUMU_CLEAN.RECEPTION_TIME_TRACKERS
            WHERE TRY_TO_TIMESTAMP(TIME_IN) >= '2025-01-01'
              AND TRY_TO_TIMESTAMP(TIME_IN)  < DATE_TRUNC('month', CURRENT_DATE)
              AND TIME_IN IS NOT NULL AND TIME_IN != ''
              AND DESTINATION_NAME IN (
                  'Reception','doctor','laboratory','radiology','pharmacy'
              )
        )
        SELECT
            stage,
            next_stage,
            COUNT(*)                                                    AS transitions,
            ROUND(MEDIAN(DATEDIFF('minute', t_in, next_t_in)))         AS median_gap_min,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (
                ORDER BY DATEDIFF('minute', t_in, next_t_in)
            ))                                                          AS p75_gap_min
        FROM ordered
        WHERE next_t_in IS NOT NULL
          AND DATEDIFF('minute', t_in, next_t_in) > 0
          AND DATEDIFF('minute', t_in, next_t_in) <= 480
        GROUP BY stage, next_stage
        ORDER BY transitions DESC
    """)


def q_patient_flow_dow():
    """KSH only: patient journey span + lab/pharmacy mix by day of week.
    Journey span = first stage TIME_IN to last stage TIME_IN (proxy — TIME_OUT unavailable).
    Corridor stages only. 2025 onward, complete months. Inv 98.
    Columns: day_num, day_name, visits, median_journey_min, p75_journey_min,
             avg_stages, lab_visits, pct_lab, pharmacy_visits, pct_pharmacy."""
    return run_query_df("""
        WITH visit_span AS (
            SELECT
                VISIT_ID,
                DAYOFWEEK(MIN(TRY_TO_TIMESTAMP(TIME_IN)))              AS day_num,
                DAYNAME(MIN(TRY_TO_TIMESTAMP(TIME_IN)))                AS day_name,
                COUNT(DISTINCT DESTINATION_NAME)                       AS stage_count,
                DATEDIFF('minute',
                    MIN(TRY_TO_TIMESTAMP(TIME_IN)),
                    MAX(TRY_TO_TIMESTAMP(TIME_IN)))                    AS journey_span_min,
                MAX(CASE WHEN DESTINATION_NAME = 'laboratory'
                     THEN 1 ELSE 0 END)                                AS had_lab,
                MAX(CASE WHEN DESTINATION_NAME = 'pharmacy'
                     THEN 1 ELSE 0 END)                                AS had_pharmacy
            FROM HOSPITALS.KISUMU_CLEAN.RECEPTION_TIME_TRACKERS
            WHERE TRY_TO_TIMESTAMP(TIME_IN) >= '2025-01-01'
              AND TRY_TO_TIMESTAMP(TIME_IN)  < DATE_TRUNC('month', CURRENT_DATE)
              AND TIME_IN IS NOT NULL AND TIME_IN != ''
            GROUP BY VISIT_ID
            HAVING COUNT(DISTINCT DESTINATION_NAME) >= 2
        )
        SELECT
            day_num, day_name,
            COUNT(*)                                                    AS visits,
            ROUND(MEDIAN(journey_span_min))                            AS median_journey_min,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP
                  (ORDER BY journey_span_min))                         AS p75_journey_min,
            ROUND(AVG(stage_count), 1)                                 AS avg_stages,
            SUM(had_lab)                                               AS lab_visits,
            ROUND(SUM(had_lab)*100.0/NULLIF(COUNT(*),0),1)            AS pct_lab,
            SUM(had_pharmacy)                                          AS pharmacy_visits,
            ROUND(SUM(had_pharmacy)*100.0/NULLIF(COUNT(*),0),1)       AS pct_pharmacy
        FROM visit_span
        WHERE journey_span_min > 0
          AND journey_span_min <= 480
        GROUP BY day_num, day_name
        ORDER BY day_num
    """)


def q_lab_tat_dow():
    """KSH only: lab TAT (order→result) by day of week. Jan–Aug 2025 baseline only —
    order linkage broke at Oct 2025 system migration. Outliers >72h excluded. Inv 98.
    Columns: day_num, day_name, test_count, median_tat_min, p75_tat_min, pct_within_8h."""
    return run_query_df("""
        SELECT
            DAYOFWEEK(i.CREATED_AT)                                    AS day_num,
            DAYNAME(i.CREATED_AT)                                      AS day_name,
            COUNT(*)                                                    AS test_count,
            ROUND(MEDIAN(
                DATEDIFF('minute', i.CREATED_AT, TRY_TO_TIMESTAMP(r.PUBLISHED_ON))
            ))                                                          AS median_tat_min,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY
                DATEDIFF('minute', i.CREATED_AT, TRY_TO_TIMESTAMP(r.PUBLISHED_ON))
            ))                                                          AS p75_tat_min,
            ROUND(SUM(CASE WHEN DATEDIFF('minute', i.CREATED_AT,
                TRY_TO_TIMESTAMP(r.PUBLISHED_ON)) <= 480 THEN 1 ELSE 0 END)
                * 100.0 / NULLIF(COUNT(*), 0), 1)                     AS pct_within_8h
        FROM HOSPITALS.KISUMU_CLEAN.EVALUATION_INVESTIGATION_RESULTS r
        JOIN HOSPITALS.KISUMU_CLEAN.EVALUATION_INVESTIGATIONS i
          ON r.INVESTIGATION = i.ID
        WHERE i.CREATED_AT IS NOT NULL
          AND TRY_TO_TIMESTAMP(r.PUBLISHED_ON) IS NOT NULL
          AND TRY_TO_TIMESTAMP(r.PUBLISHED_ON) > i.CREATED_AT
          AND DATEDIFF('hour', i.CREATED_AT, TRY_TO_TIMESTAMP(r.PUBLISHED_ON)) <= 72
          AND DATE_TRUNC('month', i.CREATED_AT) >= '2025-01-01'
          AND DATE_TRUNC('month', i.CREATED_AT) <  DATE_TRUNC('month', CURRENT_DATE)
        GROUP BY day_num, day_name
        ORDER BY day_num
    """)


def q_lab_morning_completion():
    """KSH only: monthly % of lab results delivered before 09:00 (morning rounds proxy).
    Source: stg_lab_events (silver). EVENT_TS is result timestamp, not draw time — TAT from
    draw is not measurable. This is result delivery timing only. Inv 95.
    Columns: lab_month, total_results, distinct_visits, results_by_09h, pct_by_09h."""
    return run_query_df("""
        SELECT
            EVENT_MONTH                                                    AS lab_month,
            COUNT(*)                                                       AS total_results,
            COUNT(DISTINCT VISIT_ID)                                       AS distinct_visits,
            SUM(CASE WHEN HOUR(EVENT_TS) < 9 THEN 1 ELSE 0 END)          AS results_by_09h,
            ROUND(SUM(CASE WHEN HOUR(EVENT_TS) < 9 THEN 1 ELSE 0 END)
                  * 100.0 / NULLIF(COUNT(*), 0), 1)                       AS pct_by_09h
        FROM HOSPITALS.STAGING.stg_lab_events
        WHERE EVENT_MONTH >= '2025-01-01'
          AND EVENT_MONTH <  DATE_TRUNC('month', CURRENT_DATE)
        GROUP BY EVENT_MONTH
        ORDER BY EVENT_MONTH
    """)


def q_lab_tat_monthly():
    """KSH only: monthly order-to-result TAT — Jan–Aug 2025 only (system migration Sep 2025).
    Join: EVALUATION_INVESTIGATIONS.CREATED_AT (order) → EVALUATION_INVESTIGATION_RESULTS.PUBLISHED_ON.
    Outliers >72h excluded (delayed data entry). Inv 96.
    Columns: order_month, test_count, median_tat_min, p25_tat_min, p75_tat_min,
             within_2h, within_8h, pct_within_8h."""
    return run_query_df("""
        SELECT
            DATE_TRUNC('month', i.CREATED_AT)                             AS order_month,
            COUNT(*)                                                       AS test_count,
            ROUND(MEDIAN(
                DATEDIFF('minute', i.CREATED_AT, TRY_TO_TIMESTAMP(r.PUBLISHED_ON))
            ))                                                             AS median_tat_min,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY
                DATEDIFF('minute', i.CREATED_AT, TRY_TO_TIMESTAMP(r.PUBLISHED_ON))
            ))                                                             AS p25_tat_min,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY
                DATEDIFF('minute', i.CREATED_AT, TRY_TO_TIMESTAMP(r.PUBLISHED_ON))
            ))                                                             AS p75_tat_min,
            SUM(CASE WHEN DATEDIFF('minute', i.CREATED_AT,
                TRY_TO_TIMESTAMP(r.PUBLISHED_ON)) <= 120 THEN 1 ELSE 0 END) AS within_2h,
            SUM(CASE WHEN DATEDIFF('minute', i.CREATED_AT,
                TRY_TO_TIMESTAMP(r.PUBLISHED_ON)) <= 480 THEN 1 ELSE 0 END) AS within_8h,
            ROUND(SUM(CASE WHEN DATEDIFF('minute', i.CREATED_AT,
                TRY_TO_TIMESTAMP(r.PUBLISHED_ON)) <= 480 THEN 1 ELSE 0 END)
                * 100.0 / NULLIF(COUNT(*), 0), 1)                        AS pct_within_8h
        FROM HOSPITALS.KISUMU_CLEAN.EVALUATION_INVESTIGATION_RESULTS r
        JOIN HOSPITALS.KISUMU_CLEAN.EVALUATION_INVESTIGATIONS i
          ON r.INVESTIGATION = i.ID
        WHERE i.CREATED_AT IS NOT NULL
          AND TRY_TO_TIMESTAMP(r.PUBLISHED_ON) IS NOT NULL
          AND TRY_TO_TIMESTAMP(r.PUBLISHED_ON) > i.CREATED_AT
          AND DATEDIFF('hour', i.CREATED_AT, TRY_TO_TIMESTAMP(r.PUBLISHED_ON)) <= 72
          AND DATE_TRUNC('month', i.CREATED_AT) >= '2025-01-01'
          AND DATE_TRUNC('month', i.CREATED_AT) <  DATE_TRUNC('month', CURRENT_DATE)
        GROUP BY order_month
        ORDER BY order_month
    """)


def q_lab_tat_monthly_clean():
    """KSH only: outpatient same-visit lab TAT by month. Jan–Aug 2025 only.
    Excludes inpatient 03:00 scheduled orders and next-day results.
    Columns: order_month, test_count, median_tat_min, p25_tat_min, p75_tat_min,
             within_1h, within_2h, pct_within_2h. Inv 99."""
    return run_query_df("""
        SELECT
            DATE_TRUNC('month', i.CREATED_AT)                             AS order_month,
            COUNT(*)                                                       AS test_count,
            ROUND(MEDIAN(
                DATEDIFF('minute', i.CREATED_AT, TRY_TO_TIMESTAMP(r.PUBLISHED_ON))
            ))                                                             AS median_tat_min,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY
                DATEDIFF('minute', i.CREATED_AT, TRY_TO_TIMESTAMP(r.PUBLISHED_ON))
            ))                                                             AS p25_tat_min,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY
                DATEDIFF('minute', i.CREATED_AT, TRY_TO_TIMESTAMP(r.PUBLISHED_ON))
            ))                                                             AS p75_tat_min,
            SUM(CASE WHEN DATEDIFF('minute', i.CREATED_AT,
                TRY_TO_TIMESTAMP(r.PUBLISHED_ON)) <= 60 THEN 1 ELSE 0 END)  AS within_1h,
            SUM(CASE WHEN DATEDIFF('minute', i.CREATED_AT,
                TRY_TO_TIMESTAMP(r.PUBLISHED_ON)) <= 120 THEN 1 ELSE 0 END) AS within_2h,
            ROUND(SUM(CASE WHEN DATEDIFF('minute', i.CREATED_AT,
                TRY_TO_TIMESTAMP(r.PUBLISHED_ON)) <= 120 THEN 1 ELSE 0 END)
                * 100.0 / NULLIF(COUNT(*), 0), 1)                        AS pct_within_2h
        FROM HOSPITALS.KISUMU_CLEAN.EVALUATION_INVESTIGATION_RESULTS r
        JOIN HOSPITALS.KISUMU_CLEAN.EVALUATION_INVESTIGATIONS i
          ON r.INVESTIGATION = i.ID
        WHERE i.CREATED_AT IS NOT NULL
          AND TRY_TO_TIMESTAMP(r.PUBLISHED_ON) IS NOT NULL
          AND TRY_TO_TIMESTAMP(r.PUBLISHED_ON) > i.CREATED_AT
          AND DATEDIFF('day', i.CREATED_AT, TRY_TO_TIMESTAMP(r.PUBLISHED_ON)) = 0
          AND NOT (HOUR(i.CREATED_AT) = 3 AND MINUTE(i.CREATED_AT) = 0
                   AND SECOND(i.CREATED_AT) = 0)
          AND DATE_TRUNC('month', i.CREATED_AT) >= '2025-01-01'
          AND DATE_TRUNC('month', i.CREATED_AT) <  DATE_TRUNC('month', CURRENT_DATE)
        GROUP BY order_month
        ORDER BY order_month
    """)


def q_lab_tat_dow_clean():
    """KSH only: outpatient same-visit lab TAT by day of week. Jan–Aug 2025 only.
    Excludes inpatient 03:00 scheduled orders and next-day results. Inv 99.
    Columns: day_num, day_name, test_count, median_tat_min, p75_tat_min, pct_within_2h."""
    return run_query_df("""
        SELECT
            DAYOFWEEK(i.CREATED_AT)                                    AS day_num,
            DAYNAME(i.CREATED_AT)                                      AS day_name,
            COUNT(*)                                                    AS test_count,
            ROUND(MEDIAN(
                DATEDIFF('minute', i.CREATED_AT, TRY_TO_TIMESTAMP(r.PUBLISHED_ON))
            ))                                                          AS median_tat_min,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY
                DATEDIFF('minute', i.CREATED_AT, TRY_TO_TIMESTAMP(r.PUBLISHED_ON))
            ))                                                          AS p75_tat_min,
            ROUND(SUM(CASE WHEN DATEDIFF('minute', i.CREATED_AT,
                TRY_TO_TIMESTAMP(r.PUBLISHED_ON)) <= 120 THEN 1 ELSE 0 END)
                * 100.0 / NULLIF(COUNT(*), 0), 1)                     AS pct_within_2h
        FROM HOSPITALS.KISUMU_CLEAN.EVALUATION_INVESTIGATION_RESULTS r
        JOIN HOSPITALS.KISUMU_CLEAN.EVALUATION_INVESTIGATIONS i
          ON r.INVESTIGATION = i.ID
        WHERE i.CREATED_AT IS NOT NULL
          AND TRY_TO_TIMESTAMP(r.PUBLISHED_ON) IS NOT NULL
          AND TRY_TO_TIMESTAMP(r.PUBLISHED_ON) > i.CREATED_AT
          AND DATEDIFF('day', i.CREATED_AT, TRY_TO_TIMESTAMP(r.PUBLISHED_ON)) = 0
          AND NOT (HOUR(i.CREATED_AT) = 3 AND MINUTE(i.CREATED_AT) = 0
                   AND SECOND(i.CREATED_AT) = 0)
          AND DATE_TRUNC('month', i.CREATED_AT) >= '2025-01-01'
          AND DATE_TRUNC('month', i.CREATED_AT) <  DATE_TRUNC('month', CURRENT_DATE)
        GROUP BY day_num, day_name
        ORDER BY day_num
    """)


def q_lab_tat_by_test():
    """KSH only: outpatient same-visit lab TAT by test name. Jan–Aug 2025 only.
    Same clean population as q_lab_tat_monthly_clean (no 03:00 orders, same-day only).
    Ordered by median TAT descending. Inv 100.
    Columns: test_name, test_count, median_tat_min, p25_tat_min, p75_tat_min,
             pct_within_1h, pct_within_2h."""
    return run_query_df("""
        SELECT
            r.PROCEDURE_NAME                                               AS test_name,
            COUNT(*)                                                       AS test_count,
            ROUND(MEDIAN(
                DATEDIFF('minute', i.CREATED_AT, TRY_TO_TIMESTAMP(r.PUBLISHED_ON))
            ))                                                             AS median_tat_min,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY
                DATEDIFF('minute', i.CREATED_AT, TRY_TO_TIMESTAMP(r.PUBLISHED_ON))
            ))                                                             AS p25_tat_min,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY
                DATEDIFF('minute', i.CREATED_AT, TRY_TO_TIMESTAMP(r.PUBLISHED_ON))
            ))                                                             AS p75_tat_min,
            ROUND(SUM(CASE WHEN DATEDIFF('minute', i.CREATED_AT,
                TRY_TO_TIMESTAMP(r.PUBLISHED_ON)) <= 60 THEN 1 ELSE 0 END)
                * 100.0 / NULLIF(COUNT(*), 0), 0)                        AS pct_within_1h,
            ROUND(SUM(CASE WHEN DATEDIFF('minute', i.CREATED_AT,
                TRY_TO_TIMESTAMP(r.PUBLISHED_ON)) <= 120 THEN 1 ELSE 0 END)
                * 100.0 / NULLIF(COUNT(*), 0), 0)                        AS pct_within_2h
        FROM HOSPITALS.KISUMU_CLEAN.EVALUATION_INVESTIGATION_RESULTS r
        JOIN HOSPITALS.KISUMU_CLEAN.EVALUATION_INVESTIGATIONS i
            ON r.INVESTIGATION = i.ID
        WHERE i.CREATED_AT >= '2025-01-01'
          AND i.CREATED_AT  < '2025-09-01'
          AND r.PUBLISHED_ON IS NOT NULL
          AND TRY_TO_TIMESTAMP(r.PUBLISHED_ON) IS NOT NULL
          AND TRY_TO_TIMESTAMP(r.PUBLISHED_ON) > i.CREATED_AT
          AND DATEDIFF('day', i.CREATED_AT, TRY_TO_TIMESTAMP(r.PUBLISHED_ON)) = 0
          AND NOT (HOUR(i.CREATED_AT) = 3 AND MINUTE(i.CREATED_AT) = 0
                   AND SECOND(i.CREATED_AT) = 0)
          AND r.PROCEDURE_NAME IS NOT NULL
          AND r.PROCEDURE_NAME NOT LIKE '%Malaria%Rapid%'
        GROUP BY r.PROCEDURE_NAME
        HAVING COUNT(*) >= 10
        ORDER BY median_tat_min DESC
    """)


def q_pharmacy_wait_dow():
    """KSH only: pharmacy wait time by day of week from RECEPTION_TIME_TRACKERS.WAIT_TIME.
    2025 onward, complete months. WAIT_TIME = active stage wait (queue before service).
    Inv 100. Columns: day_num, day_name, visits, median_wait_min, p75_wait_min, pct_over_30min."""
    return run_query_df("""
        SELECT
            DAYOFWEEK(TRY_TO_TIMESTAMP(TIME_IN))                          AS day_num,
            DAYNAME(TRY_TO_TIMESTAMP(TIME_IN))                            AS day_name,
            COUNT(*)                                                       AS visits,
            ROUND(MEDIAN(WAIT_TIME))                                       AS median_wait_min,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY WAIT_TIME)) AS p75_wait_min,
            ROUND(SUM(CASE WHEN WAIT_TIME > 30 THEN 1 ELSE 0 END)
                  * 100.0 / NULLIF(COUNT(*), 0), 1)                       AS pct_over_30min
        FROM HOSPITALS.KISUMU_CLEAN.RECEPTION_TIME_TRACKERS
        WHERE DESTINATION_NAME = 'pharmacy'
          AND WAIT_TIME IS NOT NULL
          AND WAIT_TIME > 0
          AND TRY_TO_TIMESTAMP(TIME_IN) >= '2025-01-01'
          AND TRY_TO_TIMESTAMP(TIME_IN) <  DATE_TRUNC('month', CURRENT_DATE)
        GROUP BY day_num, day_name
        ORDER BY day_num
    """)


def q_lab_flow_delta():
    """KSH only: lab + pharmacy WAIT_TIME — last 28d vs prior 28d, anchored on data cutoff.
    Returns visits, median_wait_min, pct_over_30min per stage per period. Inv 101.
    Columns: stage, period, visits, median_wait_min, pct_over_30min."""
    return run_query_df("""
        WITH cutoff AS (
            SELECT MAX(TRY_TO_TIMESTAMP(TIME_IN)) AS max_ts
            FROM HOSPITALS.KISUMU_CLEAN.RECEPTION_TIME_TRACKERS
            WHERE DESTINATION_NAME IN ('laboratory','pharmacy')
              AND TIME_IN IS NOT NULL AND TIME_IN != ''
        )
        SELECT
            stage,
            period,
            COUNT(DISTINCT VISIT_ID)                                       AS visits,
            ROUND(MEDIAN(WAIT_TIME))                                       AS median_wait_min,
            ROUND(SUM(CASE WHEN WAIT_TIME > 30 THEN 1 ELSE 0 END)
                  * 100.0 / NULLIF(COUNT(*), 0), 1)                       AS pct_over_30min
        FROM (
            SELECT
                DESTINATION_NAME                                           AS stage,
                WAIT_TIME,
                VISIT_ID,
                CASE
                    WHEN TRY_TO_TIMESTAMP(TIME_IN) >= DATEADD('day',-28,(SELECT max_ts FROM cutoff))
                     AND TRY_TO_TIMESTAMP(TIME_IN) <  (SELECT max_ts FROM cutoff)
                    THEN 'last_28'
                    WHEN TRY_TO_TIMESTAMP(TIME_IN) >= DATEADD('day',-56,(SELECT max_ts FROM cutoff))
                     AND TRY_TO_TIMESTAMP(TIME_IN) <  DATEADD('day',-28,(SELECT max_ts FROM cutoff))
                    THEN 'prior_28'
                    ELSE NULL
                END                                                        AS period
            FROM HOSPITALS.KISUMU_CLEAN.RECEPTION_TIME_TRACKERS
            WHERE DESTINATION_NAME IN ('laboratory','pharmacy')
              AND TIME_IN IS NOT NULL AND TIME_IN != ''
              AND WAIT_TIME IS NOT NULL AND WAIT_TIME > 0
              AND TRY_TO_TIMESTAMP(TIME_IN) >= DATEADD('day',-56,
                  (SELECT max_ts FROM cutoff))
        ) sub
        WHERE period IS NOT NULL
        GROUP BY stage, period
        ORDER BY stage, period
    """)


def q_lab_handoff_delta():
    """KSH only: lab handoff gaps to pharmacy, radiology, doctor — last 28d vs prior 28d.
    Anchored on data cutoff. Inv 101.
    Columns: next_stage, period, transitions, median_gap_min."""
    return run_query_df("""
        WITH cutoff AS (
            SELECT MAX(TRY_TO_TIMESTAMP(TIME_IN)) AS max_ts
            FROM HOSPITALS.KISUMU_CLEAN.RECEPTION_TIME_TRACKERS
            WHERE TIME_IN IS NOT NULL AND TIME_IN != ''
        ),
        tr AS (
            SELECT
                VISIT_ID,
                DESTINATION_NAME                                           AS stage,
                TRY_TO_TIMESTAMP(TIME_IN)                                  AS t_in,
                LEAD(TRY_TO_TIMESTAMP(TIME_IN)) OVER (
                    PARTITION BY VISIT_ID ORDER BY TRY_TO_TIMESTAMP(TIME_IN)
                )                                                          AS next_t_in,
                LEAD(DESTINATION_NAME) OVER (
                    PARTITION BY VISIT_ID ORDER BY TRY_TO_TIMESTAMP(TIME_IN)
                )                                                          AS next_stage,
                CASE
                    WHEN TRY_TO_TIMESTAMP(TIME_IN) >= DATEADD('day',-28,(SELECT max_ts FROM cutoff))
                     AND TRY_TO_TIMESTAMP(TIME_IN) <  (SELECT max_ts FROM cutoff)
                    THEN 'last_28'
                    WHEN TRY_TO_TIMESTAMP(TIME_IN) >= DATEADD('day',-56,(SELECT max_ts FROM cutoff))
                     AND TRY_TO_TIMESTAMP(TIME_IN) <  DATEADD('day',-28,(SELECT max_ts FROM cutoff))
                    THEN 'prior_28'
                    ELSE NULL
                END                                                        AS period
            FROM HOSPITALS.KISUMU_CLEAN.RECEPTION_TIME_TRACKERS
            WHERE TIME_IN IS NOT NULL AND TIME_IN != ''
              AND TRY_TO_TIMESTAMP(TIME_IN) >= DATEADD('day',-56,
                  (SELECT max_ts FROM cutoff))
        )
        SELECT
            next_stage,
            period,
            COUNT(*)                                                       AS transitions,
            ROUND(MEDIAN(DATEDIFF('minute', t_in, next_t_in)))            AS median_gap_min
        FROM tr
        WHERE stage = 'laboratory'
          AND next_stage IN ('pharmacy','radiology','doctor')
          AND next_t_in IS NOT NULL
          AND period IS NOT NULL
          AND DATEDIFF('minute', t_in, next_t_in) BETWEEN 1 AND 480
        GROUP BY next_stage, period
        ORDER BY next_stage, period
    """)


def q_lab_weekly_trend():
    """KSH only: weekly lab visits + median queue — last 8 weeks anchored on data cutoff.
    Inv 101. Columns: week_start, lab_visits, median_queue_min."""
    return run_query_df("""
        WITH cutoff AS (
            SELECT MAX(TRY_TO_TIMESTAMP(TIME_IN)) AS max_ts
            FROM HOSPITALS.KISUMU_CLEAN.RECEPTION_TIME_TRACKERS
            WHERE DESTINATION_NAME = 'laboratory'
              AND TIME_IN IS NOT NULL AND TIME_IN != ''
        )
        SELECT
            DATE_TRUNC('week', TRY_TO_TIMESTAMP(TIME_IN))                 AS week_start,
            COUNT(DISTINCT VISIT_ID)                                       AS lab_visits,
            ROUND(MEDIAN(WAIT_TIME))                                       AS median_queue_min
        FROM HOSPITALS.KISUMU_CLEAN.RECEPTION_TIME_TRACKERS
        WHERE DESTINATION_NAME = 'laboratory'
          AND TIME_IN IS NOT NULL AND TIME_IN != ''
          AND WAIT_TIME IS NOT NULL AND WAIT_TIME > 0
          AND TRY_TO_TIMESTAMP(TIME_IN) >= DATEADD('week',-8,
              (SELECT max_ts FROM cutoff))
          AND TRY_TO_TIMESTAMP(TIME_IN) <  (SELECT max_ts FROM cutoff)
        GROUP BY week_start
        ORDER BY week_start
    """)


def q_pharmacy_tat():
    """KSH only: pharmacy TAT — prescription written (EVALUATION_PRESCRIPTIONS.CREATED_AT)
    to dispensed (INVENTORY_EVALUATION_DISPENSING.CREATED_AT). Same-day only. Inv 104.
    Columns: prescriptions, median_tat_min, p25_min, p75_min, pct_within_30min, pct_within_60min."""
    return run_query_df("""
        SELECT
            COUNT(*)                                                        AS prescriptions,
            ROUND(MEDIAN(
                DATEDIFF('minute', p.CREATED_AT, d.CREATED_AT)
            ))                                                              AS median_tat_min,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (
                ORDER BY DATEDIFF('minute', p.CREATED_AT, d.CREATED_AT)
            ))                                                              AS p25_min,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (
                ORDER BY DATEDIFF('minute', p.CREATED_AT, d.CREATED_AT)
            ))                                                              AS p75_min,
            ROUND(SUM(CASE WHEN
                DATEDIFF('minute', p.CREATED_AT, d.CREATED_AT) <= 30
                THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 1)      AS pct_within_30min,
            ROUND(SUM(CASE WHEN
                DATEDIFF('minute', p.CREATED_AT, d.CREATED_AT) <= 60
                THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 1)      AS pct_within_60min
        FROM HOSPITALS.KISUMU_CLEAN.EVALUATION_PRESCRIPTIONS p
        JOIN HOSPITALS.KISUMU_CLEAN.INVENTORY_EVALUATION_DISPENSING d
            ON d.PRESCRIPTION = p.ID
        WHERE p.CREATED_AT >= '2025-01-01'
          AND p.CREATED_AT  < '2025-09-01'
          AND DATEDIFF('day', p.CREATED_AT, d.CREATED_AT) = 0
          AND DATEDIFF('minute', p.CREATED_AT, d.CREATED_AT) BETWEEN 1 AND 480
    """)


def q_stage_wait_delta():
    """KSH only: WAIT_TIME delta for doctor, laboratory, radiology, pharmacy — last 28d vs prior 28d.
    Anchored on MAX(TIME_IN). Used for overview patient journey cards. Inv 102.
    Columns: stage, period, visits, median_wait_min, pct_over_30min."""
    return run_query_df("""
        WITH cutoff AS (
            SELECT MAX(TRY_TO_TIMESTAMP(TIME_IN)) AS max_ts
            FROM HOSPITALS.KISUMU_CLEAN.RECEPTION_TIME_TRACKERS
            WHERE DESTINATION_NAME IN ('doctor','laboratory','radiology','pharmacy')
              AND TIME_IN IS NOT NULL AND TIME_IN != ''
        )
        SELECT
            stage,
            period,
            COUNT(DISTINCT VISIT_ID)                                       AS visits,
            ROUND(MEDIAN(WAIT_TIME))                                       AS median_wait_min,
            ROUND(SUM(CASE WHEN WAIT_TIME > 30 THEN 1 ELSE 0 END)
                  * 100.0 / NULLIF(COUNT(*), 0), 1)                       AS pct_over_30min
        FROM (
            SELECT
                DESTINATION_NAME                                           AS stage,
                WAIT_TIME,
                VISIT_ID,
                CASE
                    WHEN TRY_TO_TIMESTAMP(TIME_IN) >= DATEADD('day',-28,(SELECT max_ts FROM cutoff))
                     AND TRY_TO_TIMESTAMP(TIME_IN) <  (SELECT max_ts FROM cutoff)
                    THEN 'last_28'
                    WHEN TRY_TO_TIMESTAMP(TIME_IN) >= DATEADD('day',-56,(SELECT max_ts FROM cutoff))
                     AND TRY_TO_TIMESTAMP(TIME_IN) <  DATEADD('day',-28,(SELECT max_ts FROM cutoff))
                    THEN 'prior_28'
                    ELSE NULL
                END                                                        AS period
            FROM HOSPITALS.KISUMU_CLEAN.RECEPTION_TIME_TRACKERS
            WHERE DESTINATION_NAME IN ('doctor','laboratory','radiology','pharmacy')
              AND TIME_IN IS NOT NULL AND TIME_IN != ''
              AND WAIT_TIME IS NOT NULL AND WAIT_TIME > 0
              AND TRY_TO_TIMESTAMP(TIME_IN) >= DATEADD('day',-56,
                  (SELECT max_ts FROM cutoff))
        ) sub
        WHERE period IS NOT NULL
        GROUP BY stage, period
        ORDER BY stage, period
    """)


def q_rpt_stage_wait():
    """KSH: inter-station TAT P50/P75/P90 from rpt_stage_wait gold table.
    Returns latest complete month + prior month.
    Latest complete = MAX(visit_month) < DATE_TRUNC('month', CURRENT_DATE()).
    Columns: visit_month, from_station, to_station, p50_min, p75_min, p90_min, transitions."""
    return run_query_df("""
        WITH latest_complete AS (
            SELECT MAX(visit_month) AS cutoff_month
            FROM HOSPITALS.REPORTING.rpt_stage_wait
            WHERE visit_month < DATE_TRUNC('month', CURRENT_DATE())
        )
        SELECT
            visit_month, from_station, to_station,
            p50_min, p75_min, p90_min, transitions
        FROM HOSPITALS.REPORTING.rpt_stage_wait
        WHERE visit_month >= DATEADD('month', -1, (SELECT cutoff_month FROM latest_complete))
          AND visit_month <= (SELECT cutoff_month FROM latest_complete)
        ORDER BY visit_month, transitions DESC
    """)


def q_lab_result_volume_monthly():
    """KSH: monthly lab result volume from EVALUATION_INVESTIGATION_RESULTS.
    Covers Jun 2024–Apr 2026 (through system migration). Shows total results entered per month.
    CREATED_AT is TEXT — cast via TRY_TO_TIMESTAMP. Inv 109.
    Columns: result_month, result_count."""
    return run_query_df("""
        SELECT
            DATE_TRUNC('month', TRY_TO_TIMESTAMP(CREATED_AT))::DATE    AS result_month,
            COUNT(*)                                                    AS result_count
        FROM HOSPITALS.KISUMU_CLEAN.EVALUATION_INVESTIGATION_RESULTS
        WHERE TRY_TO_TIMESTAMP(CREATED_AT) IS NOT NULL
          AND PROCEDURE_NAME IS NOT NULL
        GROUP BY result_month
        HAVING COUNT(*) >= 50
        ORDER BY result_month
    """)


def q_lab_utilization_delta():
    """KSH: lab tests per encounter — rolling 28d vs prior 28d.
    Anchor: MIN(max stg_lab_events, max STG_EVAL_VISITS) so both sources are complete within window.
    Numerator: stg_lab_events. Denominator: STG_EVAL_VISITS distinct visits. Inv 107.
    Columns: period, lab_orders, encounters, tests_per_encounter."""
    return run_query_df("""
        WITH anchor AS (
            SELECT LEAST(
                (SELECT MAX(EVENT_TS) FROM HOSPITALS.STAGING.stg_lab_events),
                (SELECT MAX(VISIT_TS) FROM HOSPITALS.STAGING.STG_EVAL_VISITS)
            ) AS max_ts
        ),
        lab AS (
            SELECT
                CASE
                    WHEN EVENT_TS >= DATEADD('day', -28, (SELECT max_ts FROM anchor))
                     AND EVENT_TS <  (SELECT max_ts FROM anchor)
                    THEN 'last_28'
                    WHEN EVENT_TS >= DATEADD('day', -56, (SELECT max_ts FROM anchor))
                     AND EVENT_TS <  DATEADD('day', -28, (SELECT max_ts FROM anchor))
                    THEN 'prior_28'
                END                                                    AS period,
                COUNT(*)                                               AS lab_orders
            FROM HOSPITALS.STAGING.stg_lab_events
            WHERE EVENT_TS >= DATEADD('day', -56, (SELECT max_ts FROM anchor))
              AND EVENT_TS <  (SELECT max_ts FROM anchor)
            GROUP BY period
        ),
        visits AS (
            SELECT
                CASE
                    WHEN VISIT_TS >= DATEADD('day', -28, (SELECT max_ts FROM anchor))
                     AND VISIT_TS <  (SELECT max_ts FROM anchor)
                    THEN 'last_28'
                    WHEN VISIT_TS >= DATEADD('day', -56, (SELECT max_ts FROM anchor))
                     AND VISIT_TS <  DATEADD('day', -28, (SELECT max_ts FROM anchor))
                    THEN 'prior_28'
                END                                                    AS period,
                COUNT(DISTINCT VISIT_ID)                               AS encounters
            FROM HOSPITALS.STAGING.STG_EVAL_VISITS
            WHERE VISIT_TS >= DATEADD('day', -56, (SELECT max_ts FROM anchor))
              AND VISIT_TS <  (SELECT max_ts FROM anchor)
            GROUP BY period
        )
        SELECT
            l.period,
            l.lab_orders,
            v.encounters,
            ROUND(l.lab_orders / NULLIF(v.encounters, 0), 2)          AS tests_per_encounter
        FROM lab l
        JOIN visits v ON v.period = l.period
        WHERE l.period IS NOT NULL
        ORDER BY l.period DESC
    """)


def q_lab_downstream_monthly():
    """KSH only: monthly lab visit branching — to beds, to imaging, no observed downstream.
    Imaging detected by TEST_NAME pattern (ultrasound/x-ray/CT/echo/doppler in stg_lab_events).
    INPATIENT_ADMISSIONS deduplicated via DISTINCT VISIT_ID to avoid row inflation. Inv 97.
    Columns: visit_month, total_visits, to_beds, to_imaging, no_downstream,
             pct_to_beds, pct_to_imaging, pct_no_downstream."""
    return run_query_df("""
        WITH admitted AS (
            SELECT DISTINCT VISIT_ID
            FROM HOSPITALS.KISUMU_CLEAN.INPATIENT_ADMISSIONS
        ),
        visit_flags AS (
            SELECT
                VISIT_ID,
                DATE_TRUNC('month', MIN(EVENT_TS))                        AS visit_month,
                MAX(CASE WHEN UPPER(TEST_NAME) LIKE '%ULTRASOUND%'
                           OR UPPER(TEST_NAME) LIKE '%X-RAY%'
                           OR UPPER(TEST_NAME) LIKE '%X RAY%'
                           OR UPPER(TEST_NAME) LIKE '%CT SCAN%'
                           OR UPPER(TEST_NAME) LIKE '%CT-SCAN%'
                           OR UPPER(TEST_NAME) LIKE '%ECHO%'
                           OR UPPER(TEST_NAME) LIKE '%DOPPLER%'
                     THEN 1 ELSE 0 END)                                   AS has_imaging
            FROM HOSPITALS.STAGING.stg_lab_events
            WHERE EVENT_MONTH >= '2025-01-01'
              AND EVENT_MONTH <  DATE_TRUNC('month', CURRENT_DATE)
            GROUP BY VISIT_ID
        )
        SELECT
            f.visit_month,
            COUNT(*)                                                       AS total_visits,
            SUM(CASE WHEN a.VISIT_ID IS NOT NULL THEN 1 ELSE 0 END)      AS to_beds,
            SUM(CASE WHEN f.has_imaging = 1 THEN 1 ELSE 0 END)           AS to_imaging,
            SUM(CASE WHEN a.VISIT_ID IS NULL
                      AND f.has_imaging = 0 THEN 1 ELSE 0 END)           AS no_downstream,
            ROUND(SUM(CASE WHEN a.VISIT_ID IS NOT NULL THEN 1 ELSE 0 END)
                  * 100.0 / NULLIF(COUNT(*), 0), 1)                      AS pct_to_beds,
            ROUND(SUM(CASE WHEN f.has_imaging = 1 THEN 1 ELSE 0 END)
                  * 100.0 / NULLIF(COUNT(*), 0), 1)                      AS pct_to_imaging,
            ROUND(SUM(CASE WHEN a.VISIT_ID IS NULL
                            AND f.has_imaging = 0 THEN 1 ELSE 0 END)
                  * 100.0 / NULLIF(COUNT(*), 0), 1)                      AS pct_no_downstream
        FROM visit_flags f
        LEFT JOIN admitted a ON f.VISIT_ID = a.VISIT_ID
        GROUP BY f.visit_month
        ORDER BY f.visit_month
    """)


def q_lab_to_bed_monthly():
    """KSH only: monthly lab-to-bed time — last result → ADMITTED_AT for outpatient-first admissions.
    Only includes visits where admission follows the last lab result (outpatient pathway).
    Outliers >48h excluded. ADMITTED_AT is TEXT in source — cast via TRY_TO_TIMESTAMP. Inv 97.
    Columns: result_month, admitted_visits, median_lab_to_bed_min, p75_lab_to_bed_min,
             within_4h, pct_within_4h."""
    return run_query_df("""
        WITH last_lab AS (
            SELECT
                VISIT_ID,
                DATE_TRUNC('month', MAX(EVENT_TS))  AS result_month,
                MAX(EVENT_TS)                        AS last_result_ts
            FROM HOSPITALS.STAGING.stg_lab_events
            WHERE EVENT_MONTH >= '2025-01-01'
            GROUP BY VISIT_ID
        )
        SELECT
            l.result_month,
            COUNT(*)                                                       AS admitted_visits,
            ROUND(MEDIAN(
                DATEDIFF('minute', l.last_result_ts, TRY_TO_TIMESTAMP(a.ADMITTED_AT))
            ))                                                             AS median_lab_to_bed_min,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY
                DATEDIFF('minute', l.last_result_ts, TRY_TO_TIMESTAMP(a.ADMITTED_AT))
            ))                                                             AS p75_lab_to_bed_min,
            SUM(CASE WHEN DATEDIFF('minute', l.last_result_ts,
                TRY_TO_TIMESTAMP(a.ADMITTED_AT)) <= 240 THEN 1 ELSE 0 END) AS within_4h,
            ROUND(SUM(CASE WHEN DATEDIFF('minute', l.last_result_ts,
                TRY_TO_TIMESTAMP(a.ADMITTED_AT)) <= 240 THEN 1 ELSE 0 END)
                * 100.0 / NULLIF(COUNT(*), 0), 1)                        AS pct_within_4h
        FROM last_lab l
        JOIN HOSPITALS.KISUMU_CLEAN.INPATIENT_ADMISSIONS a
          ON l.VISIT_ID = a.VISIT_ID
        WHERE TRY_TO_TIMESTAMP(a.ADMITTED_AT) IS NOT NULL
          AND TRY_TO_TIMESTAMP(a.ADMITTED_AT) >= l.last_result_ts
          AND DATEDIFF('hour', l.last_result_ts, TRY_TO_TIMESTAMP(a.ADMITTED_AT)) <= 48
        GROUP BY l.result_month
        ORDER BY l.result_month
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
    month). Inpatient-side partial months also excluded via silver freshness check: any month where the
    latest ADMITTED_AT is before day 25 is stripped (Inv 135 — inpatient data can lag OPD freshness).
    Oct 2025 pipeline gap noted (Inv 32).
    Columns: ward_name, month, total_admissions, discharged_admissions, insured_admissions, total_bed_days, bed_count, btr, bti_days, bor_pct."""
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
        -- Detect inpatient data freshness per month independently of KSH_DATA_END.
        -- rpt_bed_occupancy is a gold aggregate; stg_inpatient_admissions is the source.
        -- Any month where MAX(ADMITTED_AT) < day 25 is incomplete — strip it (Inv 135).
        silver_freshness AS (
            SELECT
                DATE_TRUNC('month', ADMITTED_AT::DATE)::DATE AS admission_month,
                MAX(DAY(ADMITTED_AT::DATE))                  AS max_day_in_month
            FROM HOSPITALS.STAGING.stg_inpatient_admissions
            WHERE source_schema = 'KISUMU_CLEAN'
              AND ADMITTED_AT >= DATEADD('month', -12, DATE_TRUNC('month', CURRENT_DATE))
              AND ADMITTED_AT <  DATE_TRUNC('month', CURRENT_DATE)
            GROUP BY DATE_TRUNC('month', ADMITTED_AT::DATE)::DATE
        ),
        complete_months AS (
            SELECT admission_month
            FROM silver_freshness
            WHERE max_day_in_month >= 25
        ),
        occ AS (
            SELECT
                r.ward_name,
                r.admission_month,
                r.total_admissions,
                r.discharged_admissions,
                r.insured_admissions,
                r.total_bed_days,
                DAY(LAST_DAY(r.admission_month)) AS days_in_month
            FROM HOSPITALS.REPORTING.rpt_bed_occupancy r
            JOIN complete_months cm ON cm.admission_month = r.admission_month
            WHERE r.facility = 'KISUMU_CLEAN'
              AND r.admission_month >= DATEADD('month', -12, DATE_TRUNC('month', CURRENT_DATE))
              AND r.admission_month <  DATE_TRUNC('month', CURRENT_DATE)
              AND r.discharged_admissions > 0
        )
        SELECT
            o.ward_name,
            o.admission_month                                                        AS month,
            o.total_admissions,
            o.discharged_admissions,
            o.insured_admissions,
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


def q_admission_tat_dow():
    """KSH only: day-of-week evaluation traffic + TAT — dynamic last-complete-month anchor.
    Source: rpt_admission_tat_dow (gold, Inv 89/90).
    Anchors to MAX(tat_month) where total_admissions > 0 and month is closed.
    Excludes partial months (e.g. May 2026 had 0 admissions in stg_admission_tat).
    Self-updating: as each new month closes with real data, the anchor advances.
    Columns: tat_month, day_num, day_name, total_evaluations, total_admissions,
             conversion_pct, fast_pct, p50_tat_min, p75_tat_min."""
    return run_query_df("""
        SELECT
            tat_month, day_num, day_name,
            total_evaluations, total_admissions,
            conversion_pct, fast_pct, p50_tat_min, p75_tat_min
        FROM HOSPITALS.REPORTING.rpt_admission_tat_dow
        WHERE tat_month = (
            SELECT MAX(tat_month)
            FROM HOSPITALS.REPORTING.rpt_admission_tat_dow
            WHERE total_admissions > 0
              AND tat_month < DATE_TRUNC('month', CURRENT_DATE)
        )
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


def q_discharge_tat():
    """KSH only: monthly discharge TAT from rpt_discharge_tat (gold, Inv 84).
    Grain: one row per discharge_month.
    Clean window: Oct 2024–Aug 2025 (discharge request sync paused 2025-09-07).
    48h outlier cap applied to mean. Cohort: pre-departure requests only (~50%).
    Columns: discharge_month, total_discharges, pre_departure_cohort, cohort_pct,
             mean_tat_capped_hrs, median_tat_hours, delayed_gt4h, delayed_pct."""
    return run_query_df("""
        SELECT
            discharge_month,
            total_discharges,
            pre_departure_cohort,
            cohort_pct,
            mean_tat_capped_hrs,
            median_tat_hours,
            delayed_gt4h,
            delayed_pct
        FROM HOSPITALS.REPORTING.rpt_discharge_tat
        WHERE source_schema = 'kisumu'
          AND discharge_month <= '2025-08-01'
        ORDER BY discharge_month
    """)


def q_discharge_dow():
    """KSH only: discharge count by DOW — dynamic last-complete-month anchor.
    Source: rpt_discharge_dow (gold, Inv 91).
    Anchors to MAX(discharge_month) before current month.
    INPATIENT_DISCHARGES has no deleted_at — no soft-delete filter applied.
    Supports: admissions vs discharges by DOW (bed accumulation pattern).
    Columns: discharge_month, day_num, day_name, total_discharges."""
    return run_query_df("""
        SELECT discharge_month, day_num, day_name, total_discharges
        FROM HOSPITALS.REPORTING.rpt_discharge_dow
        WHERE discharge_month = (
            SELECT MAX(discharge_month)
            FROM HOSPITALS.REPORTING.rpt_discharge_dow
            WHERE discharge_month < DATE_TRUNC('month', CURRENT_DATE)
        )
        ORDER BY day_num
    """)


def q_revpab_private_monthly():
    """KSH only: monthly combined revenue for Private Female + Male (Rule 32 / Inv 49).
    Metric: total_admission_revenue combined across both wards per month.
    Private Maternity excluded — admission volume too sparse for a stable rolling baseline.
    Source: rpt_bed_occupancy gold table (same as BTR/BTI/BOR).
    Window: last 7 months to guarantee 4+ months after Oct 2025 exclusion.
    Inpatient-side partial months stripped via silver freshness check (same pattern as
    q_btr_bti_monthly — Inv 135/136: KSH pipeline stops mid-month independently of OPD).
    Columns: admission_month, total_revenue, total_admissions."""
    return run_query_df("""
        WITH silver_freshness AS (
            SELECT
                DATE_TRUNC('month', ADMITTED_AT::DATE)::DATE AS admission_month,
                MAX(DAY(ADMITTED_AT::DATE))                  AS max_day_in_month
            FROM HOSPITALS.STAGING.stg_inpatient_admissions
            WHERE source_schema = 'KISUMU_CLEAN'
              AND ADMITTED_AT >= DATEADD('month', -7, DATE_TRUNC('month', CURRENT_DATE))
              AND ADMITTED_AT <  DATE_TRUNC('month', CURRENT_DATE)
            GROUP BY DATE_TRUNC('month', ADMITTED_AT::DATE)::DATE
        ),
        complete_months AS (
            SELECT admission_month
            FROM silver_freshness
            WHERE max_day_in_month >= 25
        )
        SELECT
            r.admission_month,
            SUM(r.total_admission_revenue) AS total_revenue,
            SUM(r.total_admissions)        AS total_admissions
        FROM HOSPITALS.REPORTING.rpt_bed_occupancy r
        JOIN complete_months cm ON cm.admission_month = r.admission_month
        WHERE r.facility = 'KISUMU_CLEAN'
          AND r.ward_name IN ('Private Female', 'Private Male')
          AND r.admission_month >= DATEADD('month', -7, DATE_TRUNC('month', CURRENT_DATE))
          AND r.admission_month <  DATE_TRUNC('month', CURRENT_DATE)
          AND r.admission_month != '2025-10-01'
        GROUP BY r.admission_month
        ORDER BY r.admission_month
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


def q_patient_journey_sankey():
    """KSH OPD patient flow Sankey — structural pathway, all available tracked visits.
    Source: HOSPITALS.REPORTING.rpt_opd_flow (gold). All months summed.
    Scope: OPD only — Other excluded, post-admission excluded (Inv 117–126).
    Self-transitions excluded (Doctor→Doctor, Pharmacy→Pharmacy — re-consultations
    valid in gold but have no meaning on a Sankey flow diagram).
    Columns: from_stage, to_stage, visits."""
    return run_query_df("""
        SELECT
            from_station        AS from_stage,
            to_station          AS to_stage,
            SUM(transitions)    AS visits
        FROM HOSPITALS.REPORTING.rpt_opd_flow
        WHERE from_station != to_station
        GROUP BY 1, 2
        ORDER BY visits DESC
    """)


def q_opd_kpi_28d():
    """KSH: OPD arrivals + doctor reach count, last 28d vs prior 28d.
    OPD arrivals from EVALUATION_VISITS (authoritative, DELETED_AT IS NULL).
    Doctor reach from RECEPTION_TIME_TRACKERS (DESTINATION_NAME = 'doctor').
    Two separate anchors: EVALUATION_VISITS for arrivals, tracker for doctor reach.
    Inv 115/114. Columns: arrivals_last28, arrivals_prior28, doctor_reach_last28, doctor_reach_prior28."""
    return run_query_df("""
        WITH ev_cutoff AS (
            SELECT MAX(TRY_TO_TIMESTAMP(CREATED_AT)) AS max_ts
            FROM HOSPITALS.KISUMU_CLEAN.EVALUATION_VISITS
            WHERE DELETED_AT IS NULL
              AND TRY_TO_TIMESTAMP(CREATED_AT) IS NOT NULL
        ),
        opd AS (
            SELECT
                SUM(CASE WHEN TRY_TO_TIMESTAMP(CREATED_AT) >=
                              DATEADD('day', -28, (SELECT max_ts FROM ev_cutoff))
                         THEN 1 ELSE 0 END)                        AS arrivals_last28,
                SUM(CASE WHEN TRY_TO_TIMESTAMP(CREATED_AT) >=
                              DATEADD('day', -56, (SELECT max_ts FROM ev_cutoff))
                          AND TRY_TO_TIMESTAMP(CREATED_AT) <
                              DATEADD('day', -28, (SELECT max_ts FROM ev_cutoff))
                         THEN 1 ELSE 0 END)                        AS arrivals_prior28
            FROM HOSPITALS.KISUMU_CLEAN.EVALUATION_VISITS
            WHERE DELETED_AT IS NULL
              AND TRY_TO_TIMESTAMP(CREATED_AT) >=
                  DATEADD('day', -56, (SELECT max_ts FROM ev_cutoff))
        ),
        tr_cutoff AS (
            SELECT MAX(TRY_TO_TIMESTAMP(TIME_IN)) AS max_ts
            FROM HOSPITALS.KISUMU_CLEAN.RECEPTION_TIME_TRACKERS
            WHERE LOWER(TRIM(DESTINATION_NAME)) = 'doctor'
              AND TRY_TO_TIMESTAMP(TIME_IN) IS NOT NULL
        ),
        doctor AS (
            SELECT
                COUNT(DISTINCT CASE WHEN TRY_TO_TIMESTAMP(TIME_IN) >=
                                         DATEADD('day', -28, (SELECT max_ts FROM tr_cutoff))
                                    THEN VISIT_ID END)             AS doctor_reach_last28,
                COUNT(DISTINCT CASE WHEN TRY_TO_TIMESTAMP(TIME_IN) >=
                                         DATEADD('day', -56, (SELECT max_ts FROM tr_cutoff))
                                     AND TRY_TO_TIMESTAMP(TIME_IN) <
                                         DATEADD('day', -28, (SELECT max_ts FROM tr_cutoff))
                                    THEN VISIT_ID END)             AS doctor_reach_prior28
            FROM HOSPITALS.KISUMU_CLEAN.RECEPTION_TIME_TRACKERS
            WHERE LOWER(TRIM(DESTINATION_NAME)) = 'doctor'
              AND TRY_TO_TIMESTAMP(TIME_IN) >=
                  DATEADD('day', -56, (SELECT max_ts FROM tr_cutoff))
        )
        SELECT
            o.arrivals_last28,
            o.arrivals_prior28,
            d.doctor_reach_last28,
            d.doctor_reach_prior28
        FROM opd o, doctor d
    """)


def q_pharmacy_source_split():
    """KSH: pharmacy inter-station TAT split by visit source — post-doctor vs direct.
    Post-doctor: pharmacy visit where same visit_id has a prior doctor station.
    Direct: no doctor station recorded for that visit.
    Inv 116 Q1. Columns: source_type, visits, median_tat_min, p75_tat_min."""
    return run_query_df("""
        WITH doctor_ts AS (
            SELECT VISIT_ID, MIN(TRY_TO_TIMESTAMP(TIME_IN)) AS doctor_time
            FROM HOSPITALS.KISUMU_CLEAN.RECEPTION_TIME_TRACKERS
            WHERE LOWER(TRIM(DESTINATION_NAME)) = 'doctor'
              AND TRY_TO_TIMESTAMP(TIME_IN) IS NOT NULL
            GROUP BY VISIT_ID
        ),
        pharmacy_rows AS (
            SELECT
                r.VISIT_ID,
                r.WAIT_TIME,
                TRY_TO_TIMESTAMP(r.TIME_IN) AS pharm_time
            FROM HOSPITALS.KISUMU_CLEAN.RECEPTION_TIME_TRACKERS r
            WHERE LOWER(TRIM(r.DESTINATION_NAME)) = 'pharmacy'
              AND r.WAIT_TIME IS NOT NULL AND r.WAIT_TIME >= 0
              AND TRY_TO_TIMESTAMP(r.TIME_IN) IS NOT NULL
        )
        SELECT
            CASE
                WHEN d.VISIT_ID IS NOT NULL AND d.doctor_time < p.pharm_time
                THEN 'Post-Doctor'
                ELSE 'Direct'
            END                                                         AS source_type,
            COUNT(*)                                                    AS visits,
            ROUND(MEDIAN(p.WAIT_TIME))                                  AS median_tat_min,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY p.WAIT_TIME)) AS p75_tat_min
        FROM pharmacy_rows p
        LEFT JOIN doctor_ts d ON d.VISIT_ID = p.VISIT_ID
        GROUP BY 1
        ORDER BY 1
    """)


def q_pharmacy_hour_of_day():
    """KSH: pharmacy inter-station TAT by hour of day (7am–9pm).
    Uses WAIT_TIME at pharmacy station. Inv 116 Q3.
    Columns: hour_of_day, visits, median_tat_min."""
    return run_query_df("""
        SELECT
            HOUR(TRY_TO_TIMESTAMP(TIME_IN))                             AS hour_of_day,
            COUNT(*)                                                    AS visits,
            ROUND(MEDIAN(WAIT_TIME))                                    AS median_tat_min
        FROM HOSPITALS.KISUMU_CLEAN.RECEPTION_TIME_TRACKERS
        WHERE LOWER(TRIM(DESTINATION_NAME)) = 'pharmacy'
          AND WAIT_TIME IS NOT NULL AND WAIT_TIME >= 0
          AND TRY_TO_TIMESTAMP(TIME_IN) IS NOT NULL
          AND HOUR(TRY_TO_TIMESTAMP(TIME_IN)) BETWEEN 7 AND 21
        GROUP BY 1
        ORDER BY 1
    """)


def q_pharmacy_monthly_tat():
    """KSH: pharmacy inter-station TAT monthly trend. Complete months only (partial excluded).
    Uses WAIT_TIME at pharmacy station. Inv 116 Q4.
    Columns: pharm_month, visits, median_tat_min, p75_tat_min."""
    return run_query_df("""
        WITH cutoff AS (
            SELECT DATE_TRUNC('month', MAX(TRY_TO_TIMESTAMP(TIME_IN)))::DATE AS partial_month
            FROM HOSPITALS.KISUMU_CLEAN.RECEPTION_TIME_TRACKERS
            WHERE LOWER(TRIM(DESTINATION_NAME)) = 'pharmacy'
              AND TRY_TO_TIMESTAMP(TIME_IN) IS NOT NULL
        )
        SELECT
            DATE_TRUNC('month', TRY_TO_TIMESTAMP(TIME_IN))::DATE        AS pharm_month,
            COUNT(*)                                                    AS visits,
            ROUND(MEDIAN(WAIT_TIME))                                    AS median_tat_min,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY WAIT_TIME)) AS p75_tat_min
        FROM HOSPITALS.KISUMU_CLEAN.RECEPTION_TIME_TRACKERS
        WHERE LOWER(TRIM(DESTINATION_NAME)) = 'pharmacy'
          AND WAIT_TIME IS NOT NULL AND WAIT_TIME >= 0
          AND TRY_TO_TIMESTAMP(TIME_IN) IS NOT NULL
          AND DATE_TRUNC('month', TRY_TO_TIMESTAMP(TIME_IN))::DATE < (SELECT partial_month FROM cutoff)
        GROUP BY 1
        HAVING COUNT(*) >= 50
        ORDER BY 1
    """)


def q_pharmacy_wait_dist():
    """KSH: pharmacy inter-station TAT distribution in 6 wait-time buckets.
    Uses WAIT_TIME at pharmacy station. Inv 116 Q5.
    Columns: wait_bucket, sort_order, visits, pct."""
    return run_query_df("""
        WITH buckets AS (
            SELECT
                CASE
                    WHEN WAIT_TIME < 30   THEN '<30 min'
                    WHEN WAIT_TIME < 60   THEN '30–60 min'
                    WHEN WAIT_TIME < 90   THEN '60–90 min'
                    WHEN WAIT_TIME < 120  THEN '90–120 min'
                    WHEN WAIT_TIME < 240  THEN '2–4 hrs'
                    ELSE '>4 hrs'
                END AS wait_bucket,
                CASE
                    WHEN WAIT_TIME < 30   THEN 1
                    WHEN WAIT_TIME < 60   THEN 2
                    WHEN WAIT_TIME < 90   THEN 3
                    WHEN WAIT_TIME < 120  THEN 4
                    WHEN WAIT_TIME < 240  THEN 5
                    ELSE 6
                END AS sort_order
            FROM HOSPITALS.KISUMU_CLEAN.RECEPTION_TIME_TRACKERS
            WHERE LOWER(TRIM(DESTINATION_NAME)) = 'pharmacy'
              AND WAIT_TIME IS NOT NULL AND WAIT_TIME >= 0
        )
        SELECT
            wait_bucket,
            sort_order,
            COUNT(*)                                                    AS visits,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1)        AS pct
        FROM buckets
        GROUP BY 1, 2
        ORDER BY 2
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


# ── OPD Patient Flow — rpt_opd_visit_spine (Inv 128–131) ─────────────────


def q_opd_spine_summary():
    """KSH: overall OPD activity KPIs from rpt_opd_visit_spine (Section A).
    Single summary row — totals, TAT percentiles, station reach rates.
    Columns: total_visits, data_from, data_to, visits_with_tat, pct_with_tat,
             p50_tat, p75_tat, p90_tat,
             pct_had_doctor, pct_had_lab, pct_had_pharmacy, pct_had_radiology."""
    return run_query_df("""
        SELECT
            COUNT(*)                                                              AS total_visits,
            MIN(visit_date)::DATE                                                 AS data_from,
            MAX(visit_date)::DATE                                                 AS data_to,
            COUNT(tat_rd_min)                                                     AS visits_with_tat,
            ROUND(COUNT(tat_rd_min) * 100.0 / NULLIF(COUNT(*), 0), 1)            AS pct_with_tat,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY tat_rd_min), 0)   AS p50_tat,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY tat_rd_min), 0)   AS p75_tat,
            ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY tat_rd_min), 0)   AS p90_tat,
            ROUND(AVG(had_doctor::INT)    * 100, 1)                               AS pct_had_doctor,
            ROUND(AVG(had_lab::INT)       * 100, 1)                               AS pct_had_lab,
            ROUND(AVG(had_pharmacy::INT)  * 100, 1)                               AS pct_had_pharmacy,
            ROUND(AVG(had_radiology::INT) * 100, 1)                               AS pct_had_radiology
        FROM HOSPITALS.REPORTING.rpt_opd_visit_spine
    """)


def q_opd_monthly_volume():
    """KSH: monthly OPD visit volume — complete months only (Section A chart).
    Partial month (MAX month in spine) excluded. Min 10 visits per month.
    Columns: visit_month, visits."""
    return run_query_df("""
        WITH cutoff AS (
            SELECT DATE_TRUNC('month', MAX(visit_date))::DATE AS partial_month
            FROM HOSPITALS.REPORTING.rpt_opd_visit_spine
        )
        SELECT
            visit_month,
            COUNT(*) AS visits
        FROM HOSPITALS.REPORTING.rpt_opd_visit_spine
        WHERE visit_month < (SELECT partial_month FROM cutoff)
        GROUP BY visit_month
        HAVING COUNT(*) >= 10
        ORDER BY visit_month
    """)


def q_opd_dow_visits():
    """KSH: OPD visit count by day of week — all visits in spine (Section B DOW chart).
    DAYOFWEEK: 0=Sunday … 6=Saturday. Sorted numerically so caller gets Sunday-first order.
    Columns: dow, day_name, visits."""
    return run_query_df("""
        SELECT
            dow,
            day_name,
            COUNT(*) AS visits
        FROM HOSPITALS.REPORTING.rpt_opd_visit_spine
        GROUP BY dow, day_name
        ORDER BY dow
    """)


def q_opd_peak_band_tat():
    """KSH: visits + TAT percentiles by arrival band — Section B/C peak-band table.
    Bands: Early 07–09 · Peak 10–12 · After-peak 13–18.
    Filter: arrival_hour 7–18 (all map to a defined band; no NULL band rows).
    TAT = Reception → Doctor. Min 5 timed visits per band.
    Columns: band_label, band_sort, visits, visits_with_tat, p50_tat, p75_tat, p90_tat."""
    return run_query_df("""
        SELECT
            CASE
                WHEN arrival_hour BETWEEN 7  AND  9 THEN 'Early (07–09)'
                WHEN arrival_hour BETWEEN 10 AND 12 THEN 'Peak (10–12)'
                WHEN arrival_hour BETWEEN 13 AND 18 THEN 'After-peak (13–18)'
            END                                                                  AS band_label,
            CASE
                WHEN arrival_hour BETWEEN 7  AND  9 THEN 1
                WHEN arrival_hour BETWEEN 10 AND 12 THEN 2
                WHEN arrival_hour BETWEEN 13 AND 18 THEN 3
            END                                                                  AS band_sort,
            COUNT(*)                                                             AS visits,
            COUNT(tat_rd_min)                                                    AS visits_with_tat,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY tat_rd_min), 0)  AS p50_tat,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY tat_rd_min), 0)  AS p75_tat,
            ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY tat_rd_min), 0)  AS p90_tat
        FROM HOSPITALS.REPORTING.rpt_opd_visit_spine
        WHERE arrival_hour BETWEEN 7 AND 18
        GROUP BY 1, 2
        HAVING COUNT(tat_rd_min) >= 5
        ORDER BY 2
    """)


def q_opd_hourly_tat():
    """KSH: visits + median TAT by arrival hour 07–21 (Section C dual chart).
    p50_tat NULL when fewer than 5 timed visits that hour.
    Columns: arrival_hour, visits, p50_tat."""
    return run_query_df("""
        SELECT
            arrival_hour,
            COUNT(*)                                                             AS visits,
            CASE WHEN COUNT(tat_rd_min) >= 5
                 THEN ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY tat_rd_min), 0)
            END                                                                  AS p50_tat
        FROM HOSPITALS.REPORTING.rpt_opd_visit_spine
        WHERE arrival_hour BETWEEN 7 AND 21
        GROUP BY arrival_hour
        ORDER BY arrival_hour
    """)


def q_opd_weekly_pressure():
    """KSH: weekly volume + pressure-day classification (Section D persistence chart).
    Pressure = operating day with daily median TAT > 19 min (Inv 131 threshold, KSH-specific).
    Scope: arrival_hour 7–22. Days with < 5 timed visits excluded from classification.
    Min 10 visits per week.
    Columns: week_start, weekly_visits, days_in_week, pressure_days, pct_pressure, week_median_tat."""
    return run_query_df("""
        WITH daily_stats AS (
            -- One row per operating day with enough timed data for pressure classification
            SELECT
                visit_date,
                DATE_TRUNC('week', visit_date)::DATE                             AS week_start,
                COUNT(*)                                                         AS daily_visits,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY tat_rd_min)        AS daily_p50_tat
            FROM HOSPITALS.REPORTING.rpt_opd_visit_spine
            WHERE arrival_hour BETWEEN 7 AND 22
            GROUP BY visit_date, DATE_TRUNC('week', visit_date)::DATE
            HAVING COUNT(tat_rd_min) >= 5
        )
        SELECT
            week_start,
            SUM(daily_visits)                                                    AS weekly_visits,
            COUNT(*)                                                             AS days_in_week,
            COUNT(CASE WHEN daily_p50_tat > 19 THEN 1 END)                      AS pressure_days,
            ROUND(
                COUNT(CASE WHEN daily_p50_tat > 19 THEN 1 END) * 100.0
                / NULLIF(COUNT(*), 0), 1
            )                                                                    AS pct_pressure,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY daily_p50_tat), 0)
                                                                                 AS week_median_tat
        FROM daily_stats
        GROUP BY week_start
        HAVING SUM(daily_visits) >= 10
        ORDER BY week_start
    """)


def q_opd_spillover_summary():
    """KSH: station reach + TAT on Pressure vs Normal days (Section E cascade check).
    Pressure = operating day with daily median TAT > 19 min · arrival_hour 7–22 · min 5 timed.
    INNER JOIN: visits from days without enough timed data are unclassifiable — excluded.
    Columns: day_type, total_days, visits, p50_tat, p75_tat,
             pct_had_lab, pct_had_pharmacy, pct_had_radiology."""
    return run_query_df("""
        WITH daily_pressure AS (
            SELECT
                visit_date,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY tat_rd_min) AS daily_p50_tat
            FROM HOSPITALS.REPORTING.rpt_opd_visit_spine
            WHERE arrival_hour BETWEEN 7 AND 22
            GROUP BY visit_date
            HAVING COUNT(tat_rd_min) >= 5
        ),
        classified AS (
            SELECT
                s.visit_date,
                s.tat_rd_min,
                s.had_lab,
                s.had_pharmacy,
                s.had_radiology,
                CASE WHEN dp.daily_p50_tat > 19 THEN 'Pressure' ELSE 'Normal' END AS day_type
            FROM HOSPITALS.REPORTING.rpt_opd_visit_spine s
            -- INNER JOIN: only visits on days with reliable TAT classification
            JOIN daily_pressure dp ON dp.visit_date = s.visit_date
            WHERE s.arrival_hour BETWEEN 7 AND 22
        )
        SELECT
            day_type,
            COUNT(DISTINCT visit_date)                                           AS total_days,
            COUNT(*)                                                             AS visits,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY tat_rd_min), 0)  AS p50_tat,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY tat_rd_min), 0)  AS p75_tat,
            ROUND(AVG(had_lab::INT)       * 100, 1)                             AS pct_had_lab,
            ROUND(AVG(had_pharmacy::INT)  * 100, 1)                             AS pct_had_pharmacy,
            ROUND(AVG(had_radiology::INT) * 100, 1)                             AS pct_had_radiology
        FROM classified
        GROUP BY day_type
        ORDER BY day_type DESC
    """)


def q_opd_flagged_heatmap():
    """KSH: DOW × arrival-hour visit concentration on pressure days only (Section F heatmap).
    Pressure = operating day with daily median TAT > 19 min · classification uses 7–22 scope.
    Heatmap scope: 07:00–21:00 arrivals (agreed guard).
    INNER JOIN: only flagged days included.
    Columns: dow, day_name, arrival_hour, visits, p50_tat."""
    return run_query_df("""
        WITH pressure_days AS (
            SELECT visit_date
            FROM HOSPITALS.REPORTING.rpt_opd_visit_spine
            WHERE arrival_hour BETWEEN 7 AND 22
            GROUP BY visit_date
            HAVING COUNT(tat_rd_min) >= 5
               AND PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY tat_rd_min) > 19
        )
        SELECT
            s.dow,
            s.day_name,
            s.arrival_hour,
            COUNT(*)                                                             AS visits,
            CASE WHEN COUNT(s.tat_rd_min) >= 5
                 THEN ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY s.tat_rd_min), 0)
            END                                                                  AS p50_tat
        FROM HOSPITALS.REPORTING.rpt_opd_visit_spine s
        -- INNER JOIN: pressure days only
        JOIN pressure_days pd ON pd.visit_date = s.visit_date
        WHERE s.arrival_hour BETWEEN 7 AND 21
        GROUP BY s.dow, s.day_name, s.arrival_hour
        ORDER BY s.dow, s.arrival_hour
    """)


def q_opd_daily_28d():
    """KSH: daily OPD activity for last 56 days (28 operational + prior 28 comparison).
    Anchor: MAX(visit_date) in spine. Scope: arrival_hour 7–22, min 5 timed visits/day.
    Pressure flag: daily median TAT > 19 min (Inv 131, KSH-specific threshold).
    Powers Section A (28d KPI cards) and Section D (daily chart + deltas).
    Columns: visit_date, period ('last28'/'prior28'), daily_visits,
             daily_p50_tat, daily_p75_tat, is_pressure,
             pct_had_doctor, pct_had_lab, pct_had_pharmacy, pct_had_radiology."""
    return run_query_df("""
        WITH anchor AS (
            SELECT MAX(visit_date) AS data_end
            FROM HOSPITALS.REPORTING.rpt_opd_visit_spine
        ),
        window AS (
            SELECT
                data_end,
                DATEADD('day', -28, data_end) AS period_start,
                DATEADD('day', -56, data_end) AS prior_start
            FROM anchor
        )
        SELECT
            s.visit_date,
            CASE
                WHEN s.visit_date >  w.period_start THEN 'last28'
                ELSE 'prior28'
            END                                                              AS period,
            COUNT(*)                                                         AS daily_visits,
            CASE WHEN COUNT(s.tat_rd_min) >= 5
                 THEN ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY s.tat_rd_min), 0)
            END                                                              AS daily_p50_tat,
            CASE WHEN COUNT(s.tat_rd_min) >= 5
                 THEN ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY s.tat_rd_min), 0)
            END                                                              AS daily_p75_tat,
            CASE
                WHEN COUNT(s.tat_rd_min) >= 5
                 AND PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY s.tat_rd_min) > 19
                THEN TRUE ELSE FALSE
            END                                                              AS is_pressure,
            ROUND(AVG(s.had_doctor::INT)    * 100, 1)                       AS pct_had_doctor,
            ROUND(AVG(s.had_lab::INT)       * 100, 1)                       AS pct_had_lab,
            ROUND(AVG(s.had_pharmacy::INT)  * 100, 1)                       AS pct_had_pharmacy,
            ROUND(AVG(s.had_radiology::INT) * 100, 1)                       AS pct_had_radiology
        FROM HOSPITALS.REPORTING.rpt_opd_visit_spine s
        CROSS JOIN window w
        WHERE s.visit_date > w.prior_start   -- strict: both windows = exactly 28 days
          AND s.arrival_hour BETWEEN 7 AND 22
        GROUP BY s.visit_date, period
        ORDER BY s.visit_date
    """)


def q_opd_station_band_wait():
    """KSH: median inter-station wait by arrival band (morning / peak / afternoon) — OPD only.
    Source: stg_opd_tracker (silver). WAIT_MIN = time from previous station to this station.
    Arrival band anchored to Reception EVENT_TS hour. IS_POST_ADMISSION = FALSE.
    Stations: Doctor, Laboratory, Pharmacy, Radiology.
    Columns: station, arrival_band, band_sort, visits, median_wait_min."""
    return run_query_df("""
        WITH reception AS (
            SELECT
                VISIT_ID,
                DATE_PART('hour', EVENT_TS) AS arrival_hour
            FROM HOSPITALS.STAGING.stg_opd_tracker
            WHERE STATION = 'Reception'
              AND IS_POST_ADMISSION = FALSE
        ),
        waits AS (
            SELECT
                t.STATION,
                t.WAIT_MIN,
                CASE
                    WHEN r.arrival_hour BETWEEN 7  AND 9  THEN 'Morning (07–09)'
                    WHEN r.arrival_hour BETWEEN 10 AND 12 THEN 'Peak (10–12)'
                    WHEN r.arrival_hour BETWEEN 13 AND 18 THEN 'Afternoon (13–18)'
                END AS arrival_band,
                CASE
                    WHEN r.arrival_hour BETWEEN 7  AND 9  THEN 1
                    WHEN r.arrival_hour BETWEEN 10 AND 12 THEN 2
                    WHEN r.arrival_hour BETWEEN 13 AND 18 THEN 3
                END AS band_sort
            FROM HOSPITALS.STAGING.stg_opd_tracker t
            JOIN reception r ON r.VISIT_ID = t.VISIT_ID
            WHERE t.IS_POST_ADMISSION = FALSE
              AND t.STATION IN ('Doctor', 'Laboratory', 'Pharmacy', 'Radiology')
              AND t.WAIT_MIN IS NOT NULL
              AND t.WAIT_MIN BETWEEN 1 AND 480
              AND r.arrival_hour BETWEEN 7 AND 18
        )
        SELECT
            STATION                                                       AS station,
            arrival_band,
            band_sort,
            COUNT(*)                                                      AS visits,
            ROUND(MEDIAN(WAIT_MIN))                                       AS median_wait_min
        FROM waits
        WHERE arrival_band IS NOT NULL
        GROUP BY STATION, arrival_band, band_sort
        ORDER BY STATION, band_sort
    """)
