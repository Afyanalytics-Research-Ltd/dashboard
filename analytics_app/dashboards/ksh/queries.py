from db import run_query_df

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
                WHEN item_name ILIKE '%CT%'
                  OR item_name ILIKE '%angio%'
                  OR item_name ILIKE '%computed%'     THEN 'CT / Angio'
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
              item_name ILIKE '%CT%'    OR item_name ILIKE '%angio%'
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
                WHEN item_name ILIKE '%CT%'
                  OR item_name ILIKE '%angio%'
                  OR item_name ILIKE '%computed%'     THEN 'CT / Angio'
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
