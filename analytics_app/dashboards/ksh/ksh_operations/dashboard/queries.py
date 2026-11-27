"""
Homepage query functions for KSH V3 Operations Dashboard.

Pattern: one function per metric surface, returns a DataFrame.
Homepage callers use _load_row() to get a single-row dict.
All column names are UPPERCASE (run_query normalises them).
"""

from .snowflake_conn import run_query


def _load_row(query_fn) -> dict:
    """Call query_fn(), return first row as dict. Empty dict on no data."""
    df = query_fn()
    return df.iloc[0].to_dict() if len(df) else {}


# ── OPD ───────────────────────────────────────────────────────────────────────

def q_opd_homepage():
    """Last 7 settled days — 1 summary row.
    door_to_doctor_pct_change is NULL when baseline is thin (gold suppression).
    """
    return run_query("""
        WITH ranked AS (
            SELECT
                visit_date,
                visit_count,
                door_to_doctor_p50,
                door_to_doctor_28d_p50,
                door_to_doctor_pct_change,
                partial_journey_pct,
                conversion_rate,
                coverage_n,
                wait_over_a_n,
                wait_over_b_n,
                station_gap_count,
                ROW_NUMBER() OVER (ORDER BY visit_date DESC) AS rn
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_FLOW
            WHERE is_settled = TRUE
        )
        SELECT
            ROUND(AVG(CASE WHEN rn <= 7 THEN visit_count END), 0)                       AS avg_daily_visits,
            ROUND(MEDIAN(CASE WHEN rn <= 7 THEN door_to_doctor_p50 END), 1)             AS wait_p50,
            MAX(CASE WHEN rn <= 7 THEN door_to_doctor_28d_p50 END)                      AS wait_baseline_p50,
            ROUND(AVG(CASE WHEN rn <= 7 THEN door_to_doctor_pct_change END), 1)         AS wait_pct_change,
            ROUND(AVG(CASE WHEN rn <= 7 THEN partial_journey_pct END), 1)               AS partial_journey_pct,
            ROUND(AVG(CASE WHEN rn <= 7 THEN conversion_rate END), 2)                   AS conversion_rate,
            SUM(CASE WHEN rn <= 7 THEN coverage_n END)                                  AS timed_n,
            SUM(CASE WHEN rn <= 7 THEN wait_over_a_n END)                               AS wait_over_a_n,
            SUM(CASE WHEN rn <= 7 THEN wait_over_b_n END)                               AS wait_over_b_n,
            SUM(CASE WHEN rn <= 7 THEN visit_count END)                                 AS total_visit_count,
            SUM(CASE WHEN rn <= 7 THEN station_gap_count END)                           AS station_gap_n,
            SUM(CASE WHEN rn BETWEEN 8 AND 14 THEN station_gap_count END)               AS prior_station_gap_n,
            SUM(CASE WHEN rn BETWEEN 8 AND 14 THEN visit_count END)                     AS prior_visit_count,
            ROUND(AVG(CASE WHEN rn BETWEEN 8 AND 14 THEN conversion_rate END), 2)       AS prior_conversion_rate,
            MIN(CASE WHEN rn <= 7 THEN visit_date END)                                  AS period_from,
            MAX(CASE WHEN rn <= 7 THEN visit_date END)                                  AS period_to,
            ROUND(AVG(CASE WHEN rn BETWEEN 8 AND 14 THEN partial_journey_pct END), 1)   AS prior_partial_journey_pct
        FROM ranked
        WHERE rn <= 14
    """)


# ── Lab ───────────────────────────────────────────────────────────────────────

def q_lab_homepage():
    """Last complete LAB month + prior month for MoM delta. RADIOLOGY excluded."""
    return run_query("""
        WITH complete_months AS (
            SELECT
                month,
                order_count,
                coverage_n,
                coverage_pct,
                order_to_published_p50   AS tat_p50,
                pct_within_60min,
                over_1d_count,
                ROW_NUMBER() OVER (ORDER BY month DESC) AS rn
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_LAB_TAT
            WHERE test_type = 'LAB'
              AND is_partial_month = FALSE
        )
        SELECT
            cur.month,
            cur.order_count,
            cur.coverage_n,
            cur.coverage_pct,
            cur.tat_p50,
            cur.pct_within_60min,
            cur.over_1d_count,
            pri.month                                                              AS prior_month,
            pri.tat_p50                                                            AS prior_tat_p50,
            pri.over_1d_count                                                      AS prior_over_1d_count,
            CASE WHEN DATEDIFF('month', pri.month, cur.month) = 1
                 THEN ROUND((cur.tat_p50 - pri.tat_p50) / NULLIF(pri.tat_p50, 0) * 100, 1)
            END                                                                    AS tat_mom_pct
        FROM complete_months cur
        LEFT JOIN complete_months pri ON pri.rn = 2
        WHERE cur.rn = 1
    """)


# ── Pharmacy ──────────────────────────────────────────────────────────────────

def q_pharmacy_homepage():
    """Last complete month (PHARMACY store) + prior month deltas + trailing-12m avg orders.
    Fulfillment = join-based (dispensing record exists). Cancelled/stopped lines in denominator.
    Label card 'Drug lines dispensed' until bucket probe runs.
    pct_within_60min denominator = same-day dispensings only, not all dispensed lines.
    is_partial_month = latest gold month — card lags if gold not rebuilt.
    """
    return run_query("""
        WITH complete_months AS (
            SELECT
                month,
                prescription_count,
                dispensed_count,
                ROUND(fulfillment_rate * 100, 1)        AS fulfillment_pct,
                tat_p50,
                tat_p90,
                tat_3mo_baseline,
                tat_pct_change,
                ROUND(pct_within_60min * 100, 1)        AS pct_within_60min,
                coverage_n,
                ROW_NUMBER() OVER (ORDER BY month DESC) AS rn
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_PHARMACY_TAT
            WHERE store_code = 'PHARMACY'
              AND is_partial_month = FALSE
        ),
        trailing_12_avg AS (
            SELECT ROUND(AVG(prescription_count), 0) AS avg_orders_12m
            FROM complete_months
            WHERE month > DATEADD('month', -12, (SELECT MAX(month) FROM complete_months))
        ),
        opd_visits AS (
            SELECT COUNT(*) AS opd_visit_count
            FROM HOSPITALS.STAGING.STG_KSH_V3_OPD_VISITS v
            WHERE DATE_TRUNC('month', v.created_at) = (SELECT MAX(month) FROM complete_months)
              AND NOT EXISTS (
                  SELECT 1 FROM HOSPITALS.STAGING.STG_KSH_V3_INPATIENT_ADMISSIONS ia
                  WHERE ia.visit_id = v.visit_id
              )
        )
        SELECT
            cur.month,
            cur.prescription_count,
            cur.dispensed_count,
            cur.fulfillment_pct,
            cur.tat_p50,
            cur.tat_p90,
            cur.tat_3mo_baseline,
            cur.tat_pct_change,
            cur.pct_within_60min,
            cur.coverage_n,
            t12.avg_orders_12m,
            ov.opd_visit_count,
            pri.month                                                              AS prior_month,
            pri.fulfillment_pct                                                    AS prior_fulfillment_pct,
            CASE WHEN DATEDIFF('month', pri.month, cur.month) = 1
                 THEN ROUND(cur.fulfillment_pct - pri.fulfillment_pct, 1)
            END                                                                    AS fulfillment_mom_pp,
            pri.tat_p50                                                            AS prior_tat_p50,
            CASE WHEN DATEDIFF('month', pri.month, cur.month) = 1
                 THEN ROUND((cur.tat_p50 - pri.tat_p50) / NULLIF(pri.tat_p50, 0) * 100, 1)
            END                                                                    AS tat_mom_pct
        FROM complete_months cur
        CROSS JOIN trailing_12_avg t12
        CROSS JOIN opd_visits ov
        LEFT JOIN complete_months pri ON pri.rn = 2
        WHERE cur.rn = 1
    """)


def q_pharmacy_trend():
    """12-month OPD-segment dispensing trend — PHARMACY store, complete months, ordered ASC.
    OPD segment: prescriptions whose visit_id is a pure OPD visit (anti-join vs inpatient).
    dispensed_per_opd_visit = OPD dispensed lines / total OPD visits (all visits, not just those with Rx).
    opd_fulfillment_pct = OPD dispensed / OPD lines (OPD-only rate, lower than gold overall rate).
    Window: last_complete DATEADD -11 months (12 rows inclusive).
    """
    return run_query("""
        WITH latest AS (
            SELECT MAX(month) AS last_complete
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_PHARMACY_TAT
            WHERE store_code = 'PHARMACY'
              AND is_partial_month = FALSE
        ),
        opd_ids AS (
            SELECT DISTINCT v.visit_id
            FROM HOSPITALS.STAGING.STG_KSH_V3_OPD_VISITS v
            WHERE NOT EXISTS (
                SELECT 1 FROM HOSPITALS.STAGING.STG_KSH_V3_INPATIENT_ADMISSIONS ia
                WHERE ia.visit_id = v.visit_id
            )
        ),
        dispensed_ids AS (
            SELECT DISTINCT prescription_id
            FROM HOSPITALS.STAGING.STG_KSH_V3_DISPENSING
        ),
        opd_visits_monthly AS (
            SELECT
                DATE_TRUNC('month', v.created_at)::DATE AS month,
                COUNT(*)                                AS opd_visit_count
            FROM HOSPITALS.STAGING.STG_KSH_V3_OPD_VISITS v
            INNER JOIN opd_ids o ON o.visit_id = v.visit_id
            GROUP BY 1
        ),
        opd_rx_monthly AS (
            SELECT
                DATE_TRUNC('month', p.created_at)::DATE    AS month,
                COUNT(*)                                   AS opd_lines,
                COUNT_IF(d.prescription_id IS NOT NULL)    AS opd_dispensed
            FROM HOSPITALS.STAGING.STG_KSH_V3_PRESCRIPTIONS p
            INNER JOIN opd_ids o ON o.visit_id = p.visit_id
            LEFT JOIN dispensed_ids d ON d.prescription_id = p.id
            WHERE p.store_id > 0
            GROUP BY 1
        )
        SELECT
            t.month,
            t.prescription_count                                            AS all_lines,
            t.dispensed_count                                               AS all_dispensed,
            ROUND(t.fulfillment_rate * 100, 1)                             AS fulfillment_pct,
            t.tat_p50,
            t.tat_p90,
            t.tat_3mo_baseline,
            ov.opd_visit_count,
            rx.opd_lines,
            rx.opd_dispensed,
            ROUND(rx.opd_dispensed / NULLIF(rx.opd_lines, 0) * 100, 1)    AS opd_fulfillment_pct,
            ROUND(rx.opd_lines    / NULLIF(ov.opd_visit_count, 0), 2)      AS lines_per_opd_visit,
            ROUND(rx.opd_dispensed / NULLIF(ov.opd_visit_count, 0), 2)     AS dispensed_per_opd_visit
        FROM HOSPITALS.REPORTING.RPT_KSH_V3_PHARMACY_TAT t
        CROSS JOIN latest
        LEFT JOIN opd_visits_monthly ov ON ov.month = t.month
        LEFT JOIN opd_rx_monthly     rx ON rx.month = t.month
        WHERE t.store_code = 'PHARMACY'
          AND t.is_partial_month = FALSE
          AND t.month >= DATEADD('month', -11, latest.last_complete)
        ORDER BY t.month ASC
    """)


def q_pharmacy_heatmap():
    """TAT heatmap by day-of-week × hour — trailing 12 months, PHARMACY store.
    Scope: 07:00–23:00, TAT 0–1,439 min (batch excluded). thin_cell = lines < 30.
    """
    return run_query("""
        WITH fd AS (
            SELECT prescription_id, MIN(created_at) AS disp_ts
            FROM HOSPITALS.STAGING.STG_KSH_V3_DISPENSING
            GROUP BY prescription_id
        ),
        rx AS (
            SELECT
                DAYOFWEEKISO(p.created_at)                     AS dow,
                HOUR(p.created_at)                             AS hr,
                p.created_at::DATE                             AS d,
                DATEDIFF('minute', p.created_at, fd.disp_ts)   AS tat
            FROM HOSPITALS.STAGING.STG_KSH_V3_PRESCRIPTIONS p
            JOIN fd ON fd.prescription_id = p.id
            WHERE p.store_id > 0
              AND p.created_at >= DATEADD('month', -12, CURRENT_DATE())
              AND p.created_at < CURRENT_DATE()
              AND HOUR(p.created_at) BETWEEN 7 AND 23
              AND DATEDIFF('minute', p.created_at, fd.disp_ts) BETWEEN 0 AND 1439
        )
        SELECT
            dow,
            DECODE(dow,1,'Mon',2,'Tue',3,'Wed',4,'Thu',5,'Fri',6,'Sat',7,'Sun') AS day_name,
            hr,
            COUNT(*)                                                        AS lines,
            ROUND(MEDIAN(tat), 1)                                           AS p50,
            ROUND(PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY tat), 1)      AS p90,
            ROUND(COUNT_IF(tat BETWEEN 0 AND 59) * 100.0 / COUNT(*), 1)     AS pct_within_60,
            COUNT(*) < 30                                                   AS thin_cell
        FROM rx
        GROUP BY dow, hr
        ORDER BY dow, hr
    """)

# ── Inpatient ─────────────────────────────────────────────────────────────────

def q_inpatient_homepage():
    """Latest 2 complete weeks aggregated — 1 row (current) with prior-week deltas for monitor cards.
    consult_to_bed_p50 and discharge_processing_p50 are already in HOURS in the gold table.
    """
    return run_query("""
        WITH week_ranks AS (
            SELECT week_start,
                   ROW_NUMBER() OVER (ORDER BY week_start DESC) AS rn
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_INPATIENT_SUMMARY
            WHERE is_trailing_week = FALSE
            GROUP BY week_start
        ),
        agg AS (
            SELECT
                i.week_start,
                w.rn,
                SUM(i.admission_count)                            AS admission_count,
                SUM(i.open_admission_count)                       AS open_admission_count,
                SUM(i.stale_open_count)                           AS stale_open_count,
                ROUND(AVG(i.alos_days), 2)                        AS alos_days,
                ROUND(AVG(i.alos_vs_baseline_days), 2)            AS alos_vs_baseline_days,
                ROUND(AVG(i.insurance_alos_days), 2)              AS insurance_alos_days,
                ROUND(AVG(i.cash_alos_days), 2)                   AS cash_alos_days,
                ROUND(MEDIAN(i.discharge_processing_p50), 1)      AS discharge_processing_p50,
                ROUND(MEDIAN(i.consult_to_bed_p50), 1)            AS consult_to_bed_p50,
                ROUND(MAX(i.discharge_processing_p50), 1)         AS worst_disch_p50,
                MAX_BY(i.ward_name, i.discharge_processing_p50)   AS worst_disch_ward,
                ROUND(MAX(i.consult_to_bed_p50), 1)               AS worst_c2b_p50,
                MAX_BY(i.ward_name, i.consult_to_bed_p50)         AS worst_c2b_ward,
                SUM(i.total_bed_days)                             AS total_bed_days,
                SUM(i.latent_capacity_bed_days)                   AS latent_capacity_bed_days,
                SUM(i.estimated_bed_revenue)                      AS estimated_bed_revenue,
                MAX(i.as_of_date)                                 AS as_of_date
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_INPATIENT_SUMMARY i
            JOIN week_ranks w ON w.week_start = i.week_start
            WHERE w.rn IN (1, 2)
            GROUP BY i.week_start, w.rn
        )
        SELECT
            cur.week_start,
            cur.admission_count,
            cur.open_admission_count,
            cur.stale_open_count,
            cur.alos_days,
            cur.alos_vs_baseline_days,
            cur.insurance_alos_days,
            cur.cash_alos_days,
            cur.discharge_processing_p50,
            cur.consult_to_bed_p50,
            cur.worst_disch_p50,
            cur.worst_disch_ward,
            cur.worst_c2b_p50,
            cur.worst_c2b_ward,
            cur.total_bed_days,
            cur.latent_capacity_bed_days,
            cur.estimated_bed_revenue,
            cur.as_of_date,
            pri.week_start                      AS prior_week_start,
            pri.discharge_processing_p50        AS prior_discharge_p50,
            pri.consult_to_bed_p50              AS prior_consult_to_bed_p50,
            pri.estimated_bed_revenue           AS prior_estimated_bed_revenue,
            pri.worst_disch_p50                 AS prior_worst_disch_p50,
            pri.worst_c2b_p50                   AS prior_worst_c2b_p50
        FROM agg cur
        LEFT JOIN agg pri ON pri.rn = 2
        WHERE cur.rn = 1
    """)


def q_inpatient_bor():
    """Latest complete week — worst ward by BOR with prior-week delta. 1 row."""
    return run_query("""
        WITH week_ranks AS (
            SELECT week_start,
                   ROW_NUMBER() OVER (ORDER BY week_start DESC) AS rn
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_INPATIENT_SUMMARY
            WHERE is_trailing_week = FALSE
            GROUP BY week_start
        ),
        worst_ward AS (
            SELECT i.ward_name, i.week_start, ROUND(i.bor_pct, 1) AS bor_pct
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_INPATIENT_SUMMARY i
            JOIN week_ranks w ON w.week_start = i.week_start AND w.rn = 1
            WHERE i.bor_pct IS NOT NULL
            ORDER BY i.bor_pct DESC
            LIMIT 1
        ),
        prior_week AS (
            SELECT i.ward_name, ROUND(i.bor_pct, 1) AS prior_bor_pct
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_INPATIENT_SUMMARY i
            JOIN week_ranks w ON w.week_start = i.week_start AND w.rn = 2
        )
        SELECT
            ww.ward_name,
            ww.week_start,
            ww.bor_pct,
            pw.prior_bor_pct
        FROM worst_ward ww
        LEFT JOIN prior_week pw ON pw.ward_name = ww.ward_name
    """)


def q_inpatient_breach():
    """Trailing 4 complete weeks — facility-wide C2B and discharge breach counts. 1 row."""
    return run_query("""
        WITH week_ranks AS (
            SELECT week_start,
                   ROW_NUMBER() OVER (ORDER BY week_start DESC) AS rn
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_INPATIENT_SUMMARY
            WHERE is_trailing_week = FALSE
            GROUP BY week_start
        )
        SELECT
            SUM(i.c2b_timed_n_all)       AS c2b_timed_n,
            SUM(i.c2b_over_4h_n)         AS c2b_over_4h_n,
            SUM(i.discharge_timed_n_all)  AS discharge_timed_n,
            SUM(i.discharge_over_4h_n)   AS discharge_over_4h_n,
            MIN(w.week_start)                           AS period_from,
            DATEADD('day', 6, MAX(w.week_start))        AS period_to
        FROM HOSPITALS.REPORTING.RPT_KSH_V3_INPATIENT_SUMMARY i
        JOIN week_ranks w ON w.week_start = i.week_start
        WHERE w.rn <= 4
    """)


def q_inpatient_bor_monthly():
    """Last 2 complete facility months from RPT_KSH_V3_BOR_MONTHLY. 1 row (current + prior)."""
    return run_query("""
        WITH months AS (
            SELECT month_start, bor_pct, mom_delta_pp,
                   bed_days AS occupied_bed_days, capacity_bed_days AS available_bed_days,
                   data_as_of,
                   ROW_NUMBER() OVER (ORDER BY month_start DESC) AS rn
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_BOR_MONTHLY
            WHERE ward_id = 0
              AND is_partial_month = FALSE
        )
        SELECT
            cur.month_start                                    AS month,
            cur.bor_pct,
            cur.mom_delta_pp,
            cur.occupied_bed_days,
            cur.available_bed_days,
            cur.data_as_of,
            DATEDIFF('day', cur.data_as_of, CURRENT_DATE)     AS days_since_data,
            pri.month_start                                    AS prior_month,
            pri.bor_pct                                        AS prior_bor_pct
        FROM months cur
        LEFT JOIN months pri ON pri.rn = 2
        WHERE cur.rn = 1
    """)


def q_inpatient_flow_trend():
    """Last 6 complete months from RPT_KSH_V3_INPATIENT_FLOW for trend chart."""
    return run_query("""
        SELECT visit_month, opd_visit_count, converted_any, any_conversion_pct
        FROM HOSPITALS.REPORTING.RPT_KSH_V3_INPATIENT_FLOW
        WHERE is_partial_month = FALSE
        ORDER BY visit_month DESC
        LIMIT 12
    """)


def q_inpatient_bor_payer():
    """Latest complete month — bed-days by ward_type × payment_mode + spare capacity. For underutilisation chart."""
    return run_query("""
        WITH latest AS (
            SELECT MAX(month_start) AS m
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_BOR_MONTHLY
            WHERE ward_id = 0 AND is_partial_month = FALSE
        ),
        mw AS (SELECT m AS ms, DATEADD('month', 1, m) AS me FROM latest),
        ao AS (
            SELECT GREATEST(MAX(admitted_at), MAX(discharged_at))::DATE AS d
            FROM HOSPITALS.STAGING.STG_KSH_V3_INPATIENT_ADMISSIONS WHERE ward_id > 0
        ),
        stays AS (
            SELECT s.ward_id, s.payment_mode,
                   s.admitted_at::DATE AS a_d,
                   CASE WHEN s.discharged_at IS NULL THEN DATEADD('day', 1, ao.d)
                        ELSE GREATEST(s.discharged_at::DATE,
                                      DATEADD('day', 1, s.admitted_at::DATE)) END AS end_excl,
                   (s.discharged_at IS NULL
                    AND DATEDIFF('day', s.admitted_at::DATE, ao.d) > 90) AS is_stale
            FROM HOSPITALS.STAGING.STG_KSH_V3_INPATIENT_ADMISSIONS s
            CROSS JOIN ao
            WHERE NOT COALESCE(s.discharged_at < s.admitted_at, FALSE)
        ),
        bed_days AS (
            SELECT c.ward_type, s.payment_mode,
                   SUM(IFF(s.is_stale, 0, GREATEST(0, DATEDIFF('day',
                       GREATEST(s.a_d, mw.ms), LEAST(s.end_excl, mw.me))))) AS bed_days
            FROM stays s
            JOIN HOSPITALS.STAGING.STG_KSH_V3_WARD_CONFIG c ON c.ward_id = s.ward_id
            CROSS JOIN mw
            WHERE s.a_d < mw.me AND s.end_excl > mw.ms
            GROUP BY 1, 2
        ),
        capacity AS (
            SELECT c.ward_type,
                   SUM(c.registered_in_his) * DATEDIFF('day', mw.ms, mw.me) AS capacity_bed_days
            FROM HOSPITALS.STAGING.STG_KSH_V3_WARD_CONFIG c
            CROSS JOIN mw
            WHERE c.registered_in_his IS NOT NULL
            GROUP BY c.ward_type, mw.ms, mw.me
        )
        SELECT bd.ward_type, bd.payment_mode, bd.bed_days,
               cap.capacity_bed_days,
               mw.ms AS month_start
        FROM bed_days bd
        JOIN capacity cap ON cap.ward_type = bd.ward_type
        CROSS JOIN mw
        ORDER BY bd.ward_type, bd.payment_mode
    """)


def q_inpatient_bti_ward():
    """Latest complete month — BTI per ward. BTI = idle bed-days / discharges (overnight only, exact month filter)."""
    return run_query("""
        WITH latest AS (
            SELECT MAX(month_start) AS m,
                   DATEADD('month', 1, MAX(month_start)) AS m_end
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_BOR_MONTHLY
            WHERE ward_id != 0 AND is_partial_month = FALSE
        ),
        bor AS (
            SELECT b.ward_id, b.ward_name, b.month_start,
                   b.bed_days AS occupied_bed_days,
                   b.capacity_bed_days
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_BOR_MONTHLY b
            JOIN latest l ON l.m = b.month_start
            WHERE b.ward_id != 0
        ),
        discharges AS (
            SELECT ward_id, COUNT(*) AS discharge_count
            FROM HOSPITALS.STAGING.STG_KSH_V3_INPATIENT_ADMISSIONS
            WHERE discharged_at::DATE >= (SELECT m     FROM latest)
              AND discharged_at::DATE <  (SELECT m_end FROM latest)
              AND is_open_admission = FALSE
              AND admission_type_id NOT IN (10, 16)
              AND NOT COALESCE(discharged_at < admitted_at, FALSE)
            GROUP BY ward_id
        )
        SELECT b.ward_id, b.ward_name, b.month_start,
               b.occupied_bed_days, b.capacity_bed_days,
               COALESCE(d.discharge_count, 0)                                    AS discharges,
               ROUND((b.capacity_bed_days - b.occupied_bed_days)
                     / NULLIF(d.discharge_count, 0), 1)                          AS bti_days
        FROM bor b
        LEFT JOIN discharges d ON d.ward_id = b.ward_id
        ORDER BY bti_days DESC NULLS LAST
    """)


def q_inpatient_alos_monthly():
    """Last 2 complete months — facility-wide ALOS. 1 row (current + prior delta). Overnight only.
    Eligible population matches q_inpatient_alos_ward: los_days > 0, ward_id > 0."""
    return run_query("""
        WITH latest AS (
            SELECT MAX(month_start) AS m,
                   DATEADD('month', -1, MAX(month_start)) AS m_prior
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_BOR_MONTHLY
            WHERE ward_id = 0 AND is_partial_month = FALSE
        ),
        monthly_alos AS (
            SELECT
                DATE_TRUNC('month', discharged_at::DATE)  AS discharge_month,
                ROUND(AVG(los_days), 1)                   AS alos_days,
                COUNT(*)                                  AS discharge_n
            FROM HOSPITALS.STAGING.STG_KSH_V3_INPATIENT_ADMISSIONS
            WHERE is_open_admission = FALSE
              AND admission_type_id NOT IN (10, 16)
              AND NOT COALESCE(discharged_at < admitted_at, FALSE)
              AND los_days IS NOT NULL AND los_days > 0
              AND ward_id > 0
              AND DATE_TRUNC('month', discharged_at::DATE)
                      IN ((SELECT m FROM latest), (SELECT m_prior FROM latest))
            GROUP BY 1
        )
        SELECT cur.alos_days, cur.discharge_n, cur.discharge_month,
               pri.alos_days AS prior_alos_days, pri.discharge_month AS prior_month
        FROM monthly_alos cur
        JOIN latest l ON cur.discharge_month = l.m
        LEFT JOIN monthly_alos pri ON pri.discharge_month = l.m_prior
    """)


def q_inpatient_bor_ward():
    """Latest complete month — BOR, beds required, opportunity cost per ward."""
    return run_query("""
        WITH latest AS (
            SELECT MAX(month_start) AS m
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_BOR_MONTHLY
            WHERE ward_id != 0 AND is_partial_month = FALSE
        )
        SELECT
            b.ward_id,
            b.ward_name,
            b.month_start,
            b.bed_days,
            b.capacity_bed_days,
            b.bor_pct,
            c.registered_in_his                                          AS beds_registered,
            ROUND(b.bed_days / NULLIF(0.80 * DATEDIFF('day', b.month_start, DATEADD('month', 1, b.month_start)), 0), 1) AS beds_required,
            b.capacity_bed_days - b.bed_days                             AS idle_bed_days,
            (b.capacity_bed_days - b.bed_days) * c.bed_rate_cash_ksh     AS opportunity_cost_ksh,
            ROUND(b.bed_days * c.bed_rate_cash_ksh
                  / NULLIF(b.capacity_bed_days, 0), 0)                   AS revenue_per_bed_day
        FROM HOSPITALS.REPORTING.RPT_KSH_V3_BOR_MONTHLY b
        JOIN latest l ON l.m = b.month_start
        JOIN HOSPITALS.STAGING.STG_KSH_V3_WARD_CONFIG c ON c.ward_id = b.ward_id
        WHERE b.ward_id != 0
          AND b.is_partial_month = FALSE
        ORDER BY b.bor_pct DESC
    """)


def q_inpatient_flow():
    """Last 2 complete months from RPT_KSH_V3_INPATIENT_FLOW. 1 row (current) with prior-month deltas."""
    return run_query("""
        WITH ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (ORDER BY visit_month DESC) AS rn
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_INPATIENT_FLOW
            WHERE is_partial_month = FALSE
        )
        SELECT
            cur.visit_month,
            cur.opd_visit_count,
            cur.converted_overnight,
            cur.converted_any,
            cur.converted_day_case_only,
            cur.overnight_conversion_pct,
            cur.any_conversion_pct,
            cur.overnight_insurance,
            cur.overnight_cash,
            cur.tat_n,
            cur.tat_p50_hrs,
            cur.tat_p90_hrs,
            cur.tat_excluded_pct,
            cur.discharge_n,
            cur.discharge_p50_hrs,
            cur.discharge_p90_hrs,
            cur.over_12h_n,
            cur.ins_n,
            cur.ins_discharge_p50_hrs,
            cur.cash_n,
            cur.cash_discharge_p50_hrs,
            cur.total_admissions,
            cur.medical_count,
            cur.general_count,
            cur.major_surgery_count,
            cur.minor_surgery_count,
            cur.day_case_count,
            cur.post_medical_shift_flag,
            pri.visit_month                  AS prior_month,
            pri.any_conversion_pct           AS prior_any_conversion_pct,
            pri.converted_any                AS prior_converted_any,
            pri.opd_visit_count              AS prior_opd_visit_count,
            pri.overnight_conversion_pct     AS prior_overnight_conversion_pct,
            pri.converted_overnight          AS prior_converted_overnight,
            pri.tat_p90_hrs                  AS prior_tat_p90_hrs,
            pri.discharge_p50_hrs            AS prior_discharge_p50_hrs
        FROM ranked cur
        LEFT JOIN ranked pri ON pri.rn = 2
        WHERE cur.rn = 1
    """)


# ── Theatre ───────────────────────────────────────────────────────────────────

def q_theatre_homepage():
    """Last complete month (Operating Theatre) + prior month for MoM delta.
    Completion rate = elective-only (is_emergency excluded from numerator and denominator).
    Emergency rate = emergency bookings / all bookings (scheduling signal, not completion).
    due_count and completed_count are elective only — matches the Theatre detail page.
    """
    return run_query("""
        WITH params AS (
            SELECT DATE_TRUNC('month', MAX(schedule_start_time))::DATE AS current_month
            FROM HOSPITALS.STAGING.STG_KSH_V3_THEATRE_CASES
            WHERE theatre_id IS NOT NULL
              AND schedule_start_time <= CURRENT_TIMESTAMP()
        ),
        bookings AS (
            SELECT
                booking_id,
                DATE_TRUNC('month', MAX(schedule_start_time))::DATE AS month,
                LOWER(MAX(theatre_name))                            AS theatre_name,
                BOOLOR_AGG(has_operation)                           AS done,
                BOOLOR_AGG(is_emergency)                            AS is_emergency
            FROM HOSPITALS.STAGING.STG_KSH_V3_THEATRE_CASES
            WHERE theatre_id IS NOT NULL
            GROUP BY booking_id
        ),
        monthly AS (
            SELECT
                month,
                COUNT(*)                                            AS scheduled_count,
                COUNT_IF(NOT is_emergency)                          AS due_count,
                COUNT_IF(NOT is_emergency AND done)                 AS completed_count,
                COUNT_IF(NOT is_emergency AND NOT done)             AS non_completion_count,
                COUNT_IF(is_emergency)                              AS emergency_count,
                ROUND(COUNT_IF(NOT is_emergency AND done) * 100.0
                      / NULLIF(COUNT_IF(NOT is_emergency), 0), 1)   AS completion_rate_pct,
                ROUND(COUNT_IF(is_emergency) * 100.0
                      / NULLIF(COUNT(*), 0), 1)                     AS emergency_rate_pct
            FROM bookings CROSS JOIN params
            WHERE theatre_name = 'operating theatre'
              AND month < current_month
            GROUP BY month
        ),
        ranked AS (
            SELECT *, ROW_NUMBER() OVER (ORDER BY month DESC) AS rn FROM monthly
        )
        SELECT
            cur.month,
            cur.scheduled_count,
            cur.due_count,
            cur.completed_count,
            cur.non_completion_count,
            cur.emergency_count,
            cur.completion_rate_pct,
            cur.emergency_rate_pct,
            pri.month                                               AS prior_month,
            pri.completion_rate_pct                                 AS prior_completion_rate_pct,
            CASE WHEN DATEDIFF('month', pri.month, cur.month) = 1
                 THEN ROUND(cur.completion_rate_pct - pri.completion_rate_pct, 1)
            END                                                     AS completion_rate_mom_pp
        FROM ranked cur
        LEFT JOIN ranked pri ON pri.rn = 2
        WHERE cur.rn = 1
    """)


# ── Clinician ─────────────────────────────────────────────────────────────────

def q_clinician_homepage():
    """Latest week — concentration metric for signal card.
    top3_concentration_pct: % of attributed OPD visits handled by top 3 clinicians.
    OPD is an undercount (user-attributed events only) — declared on card.
    """
    return run_query("""
        WITH latest AS (
            SELECT MAX(week_start) AS latest_week
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_CLINICIAN_WORKLOAD
            WHERE NOT is_partial_week
        ),
        week_data AS (
            SELECT c.user_id, c.opd_visits_seen, c.theatre_procedures, c.days_active
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_CLINICIAN_WORKLOAD c
            CROSS JOIN latest l
            WHERE c.week_start = l.latest_week
        ),
        totals AS (
            SELECT
                SUM(IFF(opd_visits_seen > 0, opd_visits_seen, 0))  AS total_opd_visits,
                COUNT_IF(opd_visits_seen > 0)                       AS active_clinicians,
                COUNT_IF(theatre_procedures > 0)                    AS theatre_active_clinicians
            FROM week_data
        ),
        ranked AS (
            SELECT user_id, opd_visits_seen,
                   ROW_NUMBER() OVER (ORDER BY opd_visits_seen DESC) AS rn
            FROM week_data
            WHERE opd_visits_seen > 0
        )
        SELECT
            t.total_opd_visits,
            t.active_clinicians,
            t.theatre_active_clinicians,
            SUM(CASE WHEN r.rn <= 3 THEN r.opd_visits_seen ELSE 0 END)       AS top3_opd_visits,
            ROUND(SUM(CASE WHEN r.rn <= 3 THEN r.opd_visits_seen ELSE 0 END)
                  / NULLIF(t.total_opd_visits, 0) * 100, 1)                  AS top3_concentration_pct
        FROM ranked r
        CROSS JOIN totals t
        GROUP BY t.total_opd_visits, t.active_clinicians, t.theatre_active_clinicians
    """)


def q_clinician_weekly():
    """gold: per user_id metrics for the latest complete week.
    Sorted by opd_visits_seen DESC. user_id only — no name join."""
    return run_query("""
        WITH params AS (
            SELECT
                MAX(CASE WHEN NOT is_partial_week THEN week_start END) AS latest_week,
                MAX(CASE WHEN is_partial_week     THEN week_start END) AS partial_week_start
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_CLINICIAN_WORKLOAD
        )
        SELECT
            c.user_id,
            c.opd_visits_seen,
            c.theatre_procedures,
            c.avg_station_wait_min,
            c.days_active,
            c.opd_visits_per_opd_day,
            p.latest_week,
            p.partial_week_start                                        AS data_as_of
        FROM HOSPITALS.REPORTING.RPT_KSH_V3_CLINICIAN_WORKLOAD c
        CROSS JOIN params p
        WHERE c.week_start = p.latest_week
        ORDER BY c.opd_visits_seen DESC
        LIMIT 30
    """)


def q_clinician_concentration_wait():
    """gold: 12-week weekly top-3 concentration % + weighted avg patient wait at Doctor station.
    Used for consequence chart: does concentration correlate with longer patient waits?"""
    return run_query("""
        WITH params AS (
            SELECT MAX(week_start) AS latest_week
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_CLINICIAN_WORKLOAD
            WHERE NOT is_partial_week
        ),
        ranked AS (
            SELECT
                c.week_start,
                c.opd_visits_seen,
                c.avg_station_wait_min,
                ROW_NUMBER() OVER (
                    PARTITION BY c.week_start
                    ORDER BY c.opd_visits_seen DESC
                ) AS rn
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_CLINICIAN_WORKLOAD c
            CROSS JOIN params p
            WHERE c.week_start >= DATEADD('week', -12, p.latest_week)
              AND c.week_start <= p.latest_week
              AND NOT c.is_partial_week
              AND c.opd_visits_seen > 0
        )
        SELECT
            week_start,
            SUM(opd_visits_seen)                                                        AS total_opd_visits,
            COUNT(*)                                                                     AS active_clinicians,
            ROUND(
                SUM(CASE WHEN rn <= 3 THEN opd_visits_seen ELSE 0 END) * 100.0
                / NULLIF(SUM(opd_visits_seen), 0), 1
            )                                                                           AS top3_concentration_pct,
            ROUND(
                SUM(avg_station_wait_min * opd_visits_seen)
                / NULLIF(SUM(opd_visits_seen), 0), 0
            )                                                                           AS avg_wait_min
        FROM ranked
        GROUP BY week_start
        ORDER BY week_start ASC
    """)


def q_clinician_opd_to_inpatient():
    """silver: per user_id monthly OPD→inpatient conversion — last 6 complete months.
    shortcut: no gold table for this grain yet; rebuild when gold covers admission FK."""
    return run_query("""
        WITH params AS (
            SELECT DATE_TRUNC('month', MAX(time_in))::DATE AS current_month
            FROM HOSPITALS.STAGING.STG_KSH_V3_STATION_EVENTS
            WHERE station = 'Doctor' AND time_in IS NOT NULL
        ),
        doctor_visits AS (
            SELECT
                user_id,
                visit_id,
                DATE_TRUNC('month', MIN(time_in))::DATE AS month
            FROM HOSPITALS.STAGING.STG_KSH_V3_STATION_EVENTS
            WHERE station = 'Doctor'
              AND user_id IS NOT NULL
              AND visit_id IS NOT NULL
            GROUP BY user_id, visit_id
        ),
        admissions AS (
            SELECT DISTINCT visit_id
            FROM HOSPITALS.STAGING.STG_KSH_V3_INPATIENT_ADMISSIONS
        )
        SELECT
            dv.month,
            dv.user_id,
            COUNT(DISTINCT dv.visit_id)                                                            AS opd_visits,
            COUNT(DISTINCT a.visit_id)                                                             AS admitted,
            ROUND(COUNT(DISTINCT a.visit_id) * 100.0 / NULLIF(COUNT(DISTINCT dv.visit_id), 0), 1) AS conversion_pct
        FROM doctor_visits dv
        LEFT JOIN admissions a ON a.visit_id = dv.visit_id
        CROSS JOIN params p
        WHERE dv.month >= DATEADD('month', -6, p.current_month)
          AND dv.month < p.current_month
        GROUP BY dv.month, dv.user_id
        ORDER BY dv.month ASC, admitted DESC
    """)


def q_clinician_lab_per_doctor():
    """gold: per user_id lab + radiology orders per 100 OPD — latest complete month."""
    return run_query("""
        WITH latest AS (
            SELECT MAX(month) AS latest_month
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_LAB_UTILISATION
            WHERE is_partial_month = FALSE
        )
        SELECT
            u.user_id,
            SUM(u.opd_visits)       AS opd_visits,
            SUM(u.lab_orders)       AS lab_orders,
            SUM(u.radiology_orders) AS radiology_orders,
            ROUND(SUM(u.lab_orders) * 100.0 / NULLIF(SUM(u.opd_visits), 0), 1)       AS lab_per_100_opd,
            ROUND(SUM(u.radiology_orders) * 100.0 / NULLIF(SUM(u.opd_visits), 0), 1) AS rad_per_100_opd,
            l.latest_month
        FROM HOSPITALS.REPORTING.RPT_KSH_V3_LAB_UTILISATION u
        CROSS JOIN latest l
        WHERE u.month = l.latest_month
          AND u.is_partial_month = FALSE
          AND u.user_id IS NOT NULL
        GROUP BY u.user_id, l.latest_month
        ORDER BY lab_per_100_opd DESC
        LIMIT 30
    """)


# ── OPD Detail ────────────────────────────────────────────────────────────────

def q_opd_summary():
    """silver: total visits, unique patients, full date range."""
    return run_query("""
        SELECT
            COUNT(DISTINCT patient_id)   AS unique_patients,
            COUNT(*)                     AS total_visits,
            MIN(DATE(created_at))        AS first_visit,
            MAX(DATE(created_at))        AS last_visit
        FROM HOSPITALS.STAGING.STG_KSH_V3_OPD_VISITS
    """)


def q_opd_monthly():
    """gold: monthly aggregates — ramp months (<200 visits) excluded via HAVING."""
    return run_query("""
        SELECT
            DATE_TRUNC('month', visit_date)                                     AS month,
            SUM(visit_count)                                                    AS total_visits,
            SUM(insurance_count)                                                AS insurance_visits,
            SUM(cash_count)                                                     AS cash_visits,
            ROUND(SUM(insurance_count) / NULLIF(SUM(visit_count), 0) * 100, 1) AS insurance_pct,
            ROUND(MEDIAN(door_to_doctor_p50), 1)                                AS wait_p50,
            ROUND(SUM(wait_over_a_n) / NULLIF(SUM(coverage_n), 0) * 100, 1)    AS pct_over_40min,
            SUM(coverage_n)                                                     AS timed_n,
            ROUND(MEDIAN(partial_journey_pct), 1)                               AS partial_journey_pct,
            ROUND(MEDIAN(conversion_rate), 2)                                   AS conversion_rate,
            BOOLOR_AGG(is_partial_month)                                        AS is_partial
        FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_FLOW
        WHERE is_settled = TRUE
        GROUP BY 1
        HAVING SUM(visit_count) >= 200
        ORDER BY 1
    """)


def q_opd_demographics():
    """silver: gender × age_band visit matrix — aggregate in Python."""
    return run_query("""
        SELECT
            COALESCE(NULLIF(LOWER(TRIM(sex)), ''), 'unknown') AS gender,
            CASE
                WHEN dob IS NULL
                  OR DATEDIFF('year', dob, DATE(created_at)) < 0
                  OR DATEDIFF('year', dob, DATE(created_at)) > 110 THEN 'Unknown'
                WHEN DATEDIFF('year', dob, DATE(created_at)) < 5   THEN 'Under 5'
                WHEN DATEDIFF('year', dob, DATE(created_at)) < 18  THEN '5-17'
                WHEN DATEDIFF('year', dob, DATE(created_at)) < 36  THEN '18-35'
                WHEN DATEDIFF('year', dob, DATE(created_at)) < 60  THEN '36-59'
                ELSE '60+'
            END                                                     AS age_band,
            COUNT(*)                                                AS visit_count
        FROM HOSPITALS.STAGING.STG_KSH_V3_OPD_VISITS
        GROUP BY 1, 2
    """)


def q_opd_patient_segments():
    """silver: patient visit-frequency buckets — chronic concentration metric."""
    return run_query("""
        SELECT
            CASE
                WHEN visit_count = 1   THEN '1 visit'
                WHEN visit_count = 2   THEN '2 visits'
                WHEN visit_count <= 5  THEN '3-5 visits'
                WHEN visit_count <= 12 THEN '6-12 visits'
                ELSE '13+ visits'
            END                        AS segment,
            COUNT(*)                   AS patient_count,
            SUM(visit_count)           AS total_visits
        FROM (
            SELECT patient_id, COUNT(*) AS visit_count
            FROM HOSPITALS.STAGING.STG_KSH_V3_OPD_VISITS
            GROUP BY patient_id
        )
        GROUP BY 1
        ORDER BY MIN(visit_count)
    """)


def q_opd_visit_timing():
    """silver: visit counts by day-of-week and hour — from OPD visit created_at."""
    return run_query("""
        SELECT
            DAYOFWEEK(created_at)   AS day_num,
            DAYNAME(created_at)     AS day_name,
            HOUR(created_at)        AS hour,
            COUNT(*)                AS visit_count
        FROM HOSPITALS.STAGING.STG_KSH_V3_OPD_VISITS
        GROUP BY 1, 2, 3
        ORDER BY 1, 3
    """)


def q_opd_lwbs_gold():
    """gold: monthly LWBS rate — last 6 complete months, ascending for trend chart."""
    return run_query("""
        WITH ranked AS (
            SELECT
                DATE_TRUNC('month', visit_date)                                AS month,
                SUM(total)                                                     AS total_visits,
                SUM(lwbs_count)                                                AS lwbs_count,
                ROUND(SUM(lwbs_count) * 100.0 / NULLIF(SUM(total), 0), 1)     AS lwbs_pct,
                ROW_NUMBER() OVER (ORDER BY DATE_TRUNC('month', visit_date) DESC) AS rn
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_LWBS
            WHERE is_settled = TRUE AND is_partial_month = FALSE
            GROUP BY 1
        )
        SELECT month, total_visits, lwbs_count, lwbs_pct
        FROM ranked
        WHERE rn <= 6
        ORDER BY month ASC
    """)


def q_opd_lwbs_homepage():
    """gold: LWBS rate last 7 settled days + prior 7 for delta. 1 row. RPT_KSH_V3_OPD_LWBS."""
    return run_query("""
        WITH daily AS (
            SELECT
                visit_date,
                SUM(lwbs_count)  AS lwbs_count,
                SUM(total)       AS total_visits
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_LWBS
            WHERE is_settled = TRUE
            GROUP BY visit_date
        ),
        ranked AS (
            SELECT visit_date, lwbs_count, total_visits,
                   ROW_NUMBER() OVER (ORDER BY visit_date DESC) AS rn
            FROM daily
        )
        SELECT
            SUM(CASE WHEN rn <= 7 THEN lwbs_count   END)                              AS lwbs_count,
            SUM(CASE WHEN rn <= 7 THEN total_visits  END)                             AS total_visits,
            ROUND(SUM(CASE WHEN rn <= 7 THEN lwbs_count END) * 100.0
                  / NULLIF(SUM(CASE WHEN rn <= 7 THEN total_visits END), 0), 1)       AS lwbs_pct,
            ROUND(SUM(CASE WHEN rn BETWEEN 8 AND 14 THEN lwbs_count END) * 100.0
                  / NULLIF(SUM(CASE WHEN rn BETWEEN 8 AND 14 THEN total_visits END), 0), 1) AS prior_lwbs_pct,
            MIN(CASE WHEN rn <= 7 THEN visit_date END)                                AS period_from,
            MAX(CASE WHEN rn <= 7 THEN visit_date END)                                AS period_to
        FROM ranked
    """)


def q_opd_lwbs_hourly_gold():
    """gold: LWBS rate by arrival hour — last 3 complete months. From RPT_KSH_V3_OPD_LWBS."""
    return run_query("""
        WITH last3 AS (
            SELECT DISTINCT DATE_TRUNC('month', visit_date) AS month
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_LWBS
            WHERE is_settled = TRUE AND is_partial_month = FALSE
            ORDER BY 1 DESC
            LIMIT 3
        )
        SELECT
            arrival_hour,
            SUM(total)                                                         AS total,
            SUM(lwbs_count)                                                    AS lwbs_count,
            ROUND(SUM(lwbs_count) * 100.0 / NULLIF(SUM(total), 0), 1)         AS lwbs_pct
        FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_LWBS
        WHERE DATE_TRUNC('month', visit_date) IN (SELECT month FROM last3)
          AND arrival_hour BETWEEN 7 AND 20
        GROUP BY 1
        ORDER BY 1
    """)


def q_opd_lwbs_volume_gold():
    """gold: LWBS rate by daily volume band — last 3 complete months. From RPT_KSH_V3_OPD_LWBS."""
    return run_query("""
        WITH last3 AS (
            SELECT DISTINCT DATE_TRUNC('month', visit_date) AS month
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_LWBS
            WHERE is_settled = TRUE AND is_partial_month = FALSE
            ORDER BY 1 DESC
            LIMIT 3
        )
        SELECT
            CASE
                WHEN daily_visits <= 40  THEN '≤40/day'
                WHEN daily_visits <= 70  THEN '41–70/day'
                WHEN daily_visits <= 100 THEN '71–100/day'
                ELSE                         '>100/day'
            END                                                                AS volume_band,
            MIN(daily_visits)                                                  AS band_min,
            SUM(total)                                                         AS total,
            SUM(lwbs_count)                                                    AS lwbs_count,
            ROUND(SUM(lwbs_count) * 100.0 / NULLIF(SUM(total), 0), 1)         AS lwbs_pct
        FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_LWBS
        WHERE DATE_TRUNC('month', visit_date) IN (SELECT month FROM last3)
        GROUP BY 1
        ORDER BY band_min
    """)


def q_opd_lwbs_dow_gold():
    """gold: LWBS rate by day of week — last 3 complete months. From RPT_KSH_V3_OPD_LWBS.
    Sunday 38.4% vs Friday 13.3% — 25pp spread (probe Y6). Returns 7 rows ordered Mon–Sun.
    """
    return run_query("""
        WITH last3 AS (
            SELECT DISTINCT DATE_TRUNC('month', visit_date) AS month
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_LWBS
            WHERE is_settled = TRUE AND is_partial_month = FALSE
            ORDER BY 1 DESC
            LIMIT 3
        )
        SELECT
            dow_num,
            day_of_week,
            SUM(total)                                                         AS total,
            SUM(lwbs_count)                                                    AS lwbs_count,
            ROUND(SUM(lwbs_count) * 100.0 / NULLIF(SUM(total), 0), 1)         AS lwbs_pct
        FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_LWBS
        WHERE DATE_TRUNC('month', visit_date) IN (SELECT month FROM last3)
        GROUP BY 1, 2
        ORDER BY 1
    """)


def q_opd_lwbs_profile():
    """silver: LWBS rate by payment mode and age band — last 3 complete months.
    Returns rows tagged with dim='payment' or dim='age'. Filter in Python."""
    return run_query("""
        WITH last3_months AS (
            SELECT DISTINCT DATE_TRUNC('month', visit_date) AS month
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_FLOW
            WHERE is_settled = TRUE AND is_partial_month = FALSE
            ORDER BY 1 DESC
            LIMIT 3
        ),
        visit_summary AS (
            SELECT
                visit_id,
                MAX(CASE WHEN station = 'Doctor' THEN 1 ELSE 0 END) AS reached_doctor
            FROM HOSPITALS.STAGING.STG_KSH_V3_STATION_EVENTS
            WHERE DATE_TRUNC('month', DATE(created_at)) IN (SELECT month FROM last3_months)
            GROUP BY 1
        ),
        visit_profile AS (
            SELECT
                s.visit_id,
                s.reached_doctor,
                COALESCE(v.payment_mode, 'unknown') AS payment_mode,
                CASE
                    WHEN v.dob IS NULL
                      OR DATEDIFF('year', v.dob, DATE(v.created_at)) < 0
                      OR DATEDIFF('year', v.dob, DATE(v.created_at)) > 110 THEN 'Unknown'
                    WHEN DATEDIFF('year', v.dob, DATE(v.created_at)) < 5   THEN 'Under 5'
                    WHEN DATEDIFF('year', v.dob, DATE(v.created_at)) < 18  THEN '5-17'
                    WHEN DATEDIFF('year', v.dob, DATE(v.created_at)) < 36  THEN '18-35'
                    WHEN DATEDIFF('year', v.dob, DATE(v.created_at)) < 60  THEN '36-59'
                    ELSE '60+'
                END AS age_band
            FROM visit_summary s
            JOIN HOSPITALS.STAGING.STG_KSH_V3_OPD_VISITS v ON v.visit_id = s.visit_id
        )
        SELECT 'payment' AS dim, payment_mode AS segment,
            COUNT(*) AS total,
            SUM(CASE WHEN reached_doctor = 0 THEN 1 ELSE 0 END) AS lwbs_count,
            ROUND(SUM(CASE WHEN reached_doctor = 0 THEN 1 ELSE 0 END) * 100.0
                  / NULLIF(COUNT(*), 0), 1) AS lwbs_pct
        FROM visit_profile
        WHERE payment_mode IN ('cash', 'insurance')
        GROUP BY 2
        UNION ALL
        SELECT 'age' AS dim, age_band AS segment,
            COUNT(*) AS total,
            SUM(CASE WHEN reached_doctor = 0 THEN 1 ELSE 0 END) AS lwbs_count,
            ROUND(SUM(CASE WHEN reached_doctor = 0 THEN 1 ELSE 0 END) * 100.0
                  / NULLIF(COUNT(*), 0), 1) AS lwbs_pct
        FROM visit_profile
        WHERE age_band != 'Unknown'
        GROUP BY 2
        ORDER BY dim, lwbs_pct DESC
    """)


def q_opd_lwbs_hourly():
    """silver: LWBS rate by arrival hour — last 3 complete months. Hours 7-20 only."""
    return run_query("""
        WITH last3_months AS (
            SELECT DISTINCT DATE_TRUNC('month', visit_date) AS month
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_FLOW
            WHERE is_settled = TRUE AND is_partial_month = FALSE
            ORDER BY 1 DESC
            LIMIT 3
        )
        SELECT
            HOUR(MIN(created_at))                                              AS arrival_hour,
            COUNT(*)                                                           AS total,
            SUM(CASE WHEN reached_doctor = 0 THEN 1 ELSE 0 END)               AS lwbs_count,
            ROUND(SUM(CASE WHEN reached_doctor = 0 THEN 1 ELSE 0 END) * 100.0
                  / NULLIF(COUNT(*), 0), 1)                                    AS lwbs_pct
        FROM (
            SELECT
                visit_id,
                MIN(created_at)                                                AS created_at,
                MAX(CASE WHEN station = 'Doctor' THEN 1 ELSE 0 END)           AS reached_doctor
            FROM HOSPITALS.STAGING.STG_KSH_V3_STATION_EVENTS
            WHERE DATE_TRUNC('month', DATE(created_at)) IN (SELECT month FROM last3_months)
            GROUP BY 1
        ) vs
        WHERE HOUR(created_at) BETWEEN 7 AND 20
        GROUP BY 1
        ORDER BY 1
    """)


def q_opd_lwbs_volume():
    """silver: LWBS rate by daily visit volume band — last 3 complete months."""
    return run_query("""
        WITH last3_months AS (
            SELECT DISTINCT DATE_TRUNC('month', visit_date) AS month
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_FLOW
            WHERE is_settled = TRUE AND is_partial_month = FALSE
            ORDER BY 1 DESC
            LIMIT 3
        ),
        visit_stations AS (
            SELECT
                visit_id,
                DATE(created_at)                                               AS day,
                MAX(CASE WHEN station = 'Doctor' THEN 1 ELSE 0 END)           AS reached_doctor
            FROM HOSPITALS.STAGING.STG_KSH_V3_STATION_EVENTS
            WHERE DATE_TRUNC('month', DATE(created_at)) IN (SELECT month FROM last3_months)
            GROUP BY 1, 2
        ),
        daily_volume AS (
            SELECT day, COUNT(DISTINCT visit_id) AS daily_visits
            FROM visit_stations
            GROUP BY 1
        )
        SELECT
            CASE
                WHEN dv.daily_visits <= 40  THEN '≤40/day'
                WHEN dv.daily_visits <= 70  THEN '41–70/day'
                WHEN dv.daily_visits <= 100 THEN '71–100/day'
                ELSE                             '>100/day'
            END                                                                AS volume_band,
            MIN(dv.daily_visits)                                               AS band_min,
            COUNT(*)                                                           AS total,
            SUM(CASE WHEN vs.reached_doctor = 0 THEN 1 ELSE 0 END)            AS lwbs_count,
            ROUND(SUM(CASE WHEN vs.reached_doctor = 0 THEN 1 ELSE 0 END) * 100.0
                  / NULLIF(COUNT(*), 0), 1)                                    AS lwbs_pct
        FROM visit_stations vs
        JOIN daily_volume dv ON dv.day = vs.day
        GROUP BY 1
        ORDER BY band_min
    """)


def q_opd_lwbs():
    """silver: LWBS rate per month — last 3 complete months.
    LWBS = visit touched at least one station but never reached Doctor."""
    return run_query("""
        WITH last3_months AS (
            SELECT DISTINCT DATE_TRUNC('month', visit_date) AS month
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_FLOW
            WHERE is_settled = TRUE AND is_partial_month = FALSE
            ORDER BY 1 DESC
            LIMIT 3
        ),
        visit_stations AS (
            SELECT
                visit_id,
                DATE_TRUNC('month', DATE(created_at)) AS month,
                MAX(CASE WHEN station = 'Doctor' THEN 1 ELSE 0 END) AS reached_doctor
            FROM HOSPITALS.STAGING.STG_KSH_V3_STATION_EVENTS
            WHERE DATE_TRUNC('month', DATE(created_at)) IN (SELECT month FROM last3_months)
            GROUP BY 1, 2
        )
        SELECT
            month,
            ROUND(SUM(CASE WHEN reached_doctor = 0 THEN 1 ELSE 0 END) * 100.0
                  / NULLIF(COUNT(*), 0), 1) AS lwbs_pct
        FROM visit_stations
        GROUP BY 1
        ORDER BY 1
    """)


def q_opd_lab_conversion():
    """silver: lab referral + fulfillment rate per month — last 3 complete months."""
    return run_query("""
        WITH last3_months AS (
            SELECT DISTINCT DATE_TRUNC('month', visit_date) AS month
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_FLOW
            WHERE is_settled = TRUE AND is_partial_month = FALSE
            ORDER BY 1 DESC
            LIMIT 3
        ),
        opd_visits AS (
            SELECT visit_id,
                   DATE_TRUNC('month', DATE(created_at)) AS month
            FROM HOSPITALS.STAGING.STG_KSH_V3_OPD_VISITS
            WHERE DATE_TRUNC('month', DATE(created_at)) IN (SELECT month FROM last3_months)
        ),
        lab_orders AS (
            SELECT visit_id,
                   COUNT(*)                                                  AS active_orders,
                   SUM(CASE WHEN published_on IS NOT NULL THEN 1 ELSE 0 END) AS fulfilled_orders
            FROM HOSPITALS.STAGING.STG_KSH_V3_LAB_INVESTIGATIONS
            WHERE type = 'LAB'
            GROUP BY 1
        )
        SELECT
            v.month,
            COUNT(DISTINCT v.visit_id)                                                        AS total_visits,
            COUNT(DISTINCT CASE WHEN lo.active_orders   > 0 THEN v.visit_id END)              AS visits_with_lab,
            COUNT(DISTINCT CASE WHEN lo.fulfilled_orders > 0 THEN v.visit_id END)             AS visits_fulfilled,
            ROUND(COUNT(DISTINCT CASE WHEN lo.active_orders > 0
                        THEN v.visit_id END) * 100.0 / NULLIF(COUNT(DISTINCT v.visit_id), 0), 1) AS referral_pct,
            ROUND(COUNT(DISTINCT CASE WHEN lo.fulfilled_orders > 0
                        THEN v.visit_id END) * 100.0 / NULLIF(COUNT(DISTINCT CASE WHEN lo.active_orders > 0
                        THEN v.visit_id END), 0), 1)                                          AS fulfillment_pct
        FROM opd_visits v
        LEFT JOIN lab_orders lo ON lo.visit_id = v.visit_id
        GROUP BY 1
        ORDER BY 1
    """)


def q_opd_dow_volume():
    """gold: avg daily visits by day of week — latest complete month only."""
    return run_query("""
        WITH latest_month AS (
            SELECT MAX(DATE_TRUNC('month', visit_date)) AS month
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_FLOW
            WHERE is_settled = TRUE AND is_partial_month = FALSE
        )
        SELECT
            DAYOFWEEK(visit_date)                                       AS dow_num,
            DAYNAME(visit_date)                                         AS day_name,
            COUNT(DISTINCT visit_date)                                  AS week_count,
            ROUND(SUM(visit_count) / NULLIF(COUNT(DISTINCT visit_date), 0), 0) AS avg_visits
        FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_FLOW
        WHERE DATE_TRUNC('month', visit_date) = (SELECT month FROM latest_month)
          AND is_settled = TRUE
        GROUP BY 1, 2
        ORDER BY CASE WHEN DAYOFWEEK(visit_date) = 0 THEN 7 ELSE DAYOFWEEK(visit_date) END
    """)


def q_opd_station_visits():
    """silver STATION_EVENTS: distinct visits per station — last 3 complete months."""
    return run_query("""
        WITH last3_months AS (
            SELECT DISTINCT DATE_TRUNC('month', visit_date) AS month
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_FLOW
            WHERE is_settled = TRUE AND is_partial_month = FALSE
            ORDER BY 1 DESC
            LIMIT 3
        )
        SELECT
            station,
            COUNT(DISTINCT visit_id)                                            AS visit_count,
            ROUND(COUNT(DISTINCT visit_id) / SUM(COUNT(DISTINCT visit_id)) OVER () * 100, 1) AS pct
        FROM HOSPITALS.STAGING.STG_KSH_V3_STATION_EVENTS
        WHERE DATE_TRUNC('month', DATE(created_at)) IN (SELECT month FROM last3_months)
        GROUP BY station
        ORDER BY visit_count DESC
    """)


def q_opd_door_to_doctor_hourly():
    """silver: true pooled median Reception.time_in -> Doctor.time_in by time_band.
    All days combined (Sun excluded). March 2026+."""
    return run_query("""
        WITH reception AS (
            SELECT visit_id, time_in AS reception_time
            FROM HOSPITALS.STAGING.STG_KSH_V3_STATION_EVENTS
            WHERE station = 'Reception' AND time_in IS NOT NULL
            QUALIFY ROW_NUMBER() OVER (PARTITION BY visit_id ORDER BY time_in ASC) = 1
        ),
        doctor_timed AS (
            SELECT visit_id, time_in AS doctor_time
            FROM HOSPITALS.STAGING.STG_KSH_V3_STATION_EVENTS
            WHERE station = 'Doctor' AND time_in IS NOT NULL
            QUALIFY ROW_NUMBER() OVER (PARTITION BY visit_id ORDER BY time_in ASC) = 1
        ),
        visit_waits AS (
            SELECT
                CASE
                    WHEN HOUR(r.reception_time) BETWEEN 7  AND 9  THEN 'Early'
                    WHEN HOUR(r.reception_time) BETWEEN 10 AND 12 THEN 'Peak'
                    WHEN HOUR(r.reception_time) BETWEEN 13 AND 16 THEN 'After-peak'
                END                                                      AS time_band,
                DATEDIFF('minute', r.reception_time, d.doctor_time)     AS raw_wait
            FROM reception r
            JOIN doctor_timed d ON d.visit_id = r.visit_id
            WHERE DATE(r.reception_time) >= '2026-03-01'
              AND HOUR(r.reception_time) BETWEEN 7 AND 16
              AND DAYNAME(r.reception_time) != 'Sun'
        )
        SELECT
            time_band,
            COUNT(*)                                                     AS visit_n,
            ROUND(MEDIAN(
                CASE WHEN raw_wait BETWEEN 1 AND 240 THEN raw_wait END
            ), 0)                                                        AS p50,
            COUNT_IF(raw_wait BETWEEN 1 AND 240)                        AS timed_n
        FROM visit_waits
        WHERE time_band IS NOT NULL
        GROUP BY 1
        ORDER BY CASE time_band WHEN 'Early' THEN 1 WHEN 'Peak' THEN 2 ELSE 3 END
    """)


def q_opd_doc_to_pharmacy_hourly():
    """silver: true pooled median Doctor.created_at -> Pharmacy.created_at by time_band.
    All days combined (Sun excluded). March 2026+."""
    return run_query("""
        WITH visit_stations AS (
            SELECT
                v.visit_id,
                CASE
                    WHEN HOUR(v.created_at) BETWEEN 7  AND 9  THEN 'Early'
                    WHEN HOUR(v.created_at) BETWEEN 10 AND 12 THEN 'Peak'
                    WHEN HOUR(v.created_at) BETWEEN 13 AND 16 THEN 'After-peak'
                END                                                      AS time_band,
                MIN(CASE WHEN se.station = 'Doctor'   THEN se.created_at END) AS doctor_ts,
                MIN(CASE WHEN se.station = 'Pharmacy' THEN se.created_at END) AS pharmacy_ts
            FROM HOSPITALS.STAGING.STG_KSH_V3_OPD_VISITS v
            LEFT JOIN HOSPITALS.STAGING.STG_KSH_V3_STATION_EVENTS se
                ON se.visit_id = v.visit_id
            WHERE DATE(v.created_at) >= '2026-03-01'
              AND HOUR(v.created_at) BETWEEN 7 AND 16
              AND DAYNAME(v.created_at) != 'Sun'
            GROUP BY 1, 2
        )
        SELECT
            time_band,
            COUNT(*)                                                     AS visit_n,
            ROUND(MEDIAN(
                CASE WHEN pharmacy_ts > doctor_ts
                      AND DATEDIFF('minute', doctor_ts, pharmacy_ts) BETWEEN 1 AND 240
                     THEN DATEDIFF('minute', doctor_ts, pharmacy_ts) END
            ), 0)                                                        AS p50,
            COUNT_IF(
                pharmacy_ts > doctor_ts
                AND DATEDIFF('minute', doctor_ts, pharmacy_ts) BETWEEN 1 AND 240
            )                                                            AS timed_n
        FROM visit_stations
        WHERE time_band IS NOT NULL
        GROUP BY 1
        ORDER BY CASE time_band WHEN 'Early' THEN 1 WHEN 'Peak' THEN 2 ELSE 3 END
    """)


def q_opd_station_tat_dist():
    """silver: visit counts by station × wait bucket — for 100% stacked bar.
    Shows wait distribution shape per station, not just the median."""
    return run_query("""
        SELECT
            station,
            CASE
                WHEN wait_minutes <  15 THEN '1 · <15 min'
                WHEN wait_minutes <  30 THEN '2 · 15–30 min'
                WHEN wait_minutes <  45 THEN '3 · 30–45 min'
                WHEN wait_minutes <  60 THEN '4 · 45–60 min'
                WHEN wait_minutes <  90 THEN '5 · 60–90 min'
                ELSE                         '6 · 90+ min'
            END                             AS wait_bucket,
            COUNT(DISTINCT visit_id)        AS visits
        FROM HOSPITALS.STAGING.STG_KSH_V3_STATION_EVENTS
        WHERE station IN ('Pharmacy', 'Laboratory', 'Radiology', 'Doctor')
          AND wait_minutes BETWEEN 1 AND 240
        GROUP BY 1, 2
        ORDER BY 1, 2
    """)


def q_opd_pharmacy_hourly():
    """silver: pharmacy P50 + P75 by hour of day — shows queue buildup pattern.
    Sunday excluded (low volume, different profile)."""
    return run_query("""
        SELECT
            HOUR(created_at)                    AS hour,
            COUNT(DISTINCT visit_id)            AS visits,
            ROUND(MEDIAN(
                CASE WHEN wait_minutes BETWEEN 1 AND 240 THEN wait_minutes END
            ), 0)                               AS p50_min,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY
                CASE WHEN wait_minutes BETWEEN 1 AND 240 THEN wait_minutes END
            ), 0)                               AS p75_min
        FROM HOSPITALS.STAGING.STG_KSH_V3_STATION_EVENTS
        WHERE station = 'Pharmacy'
          AND DAYOFWEEK(created_at) != 1
        GROUP BY 1
        HAVING p50_min IS NOT NULL
        ORDER BY 1
    """)


def q_opd_pharmacy_daily():
    """silver: one row per day — daily visit count and pharmacy P50.
    For scatter plot showing volume vs wait relationship. Sunday excluded."""
    return run_query("""
        SELECT
            DATE(created_at)                                                       AS day,
            COUNT(DISTINCT visit_id)                                               AS daily_visits,
            ROUND(MEDIAN(
                CASE WHEN wait_minutes BETWEEN 1 AND 240 THEN wait_minutes END
            ), 0)                                                                  AS p50_min
        FROM HOSPITALS.STAGING.STG_KSH_V3_STATION_EVENTS
        WHERE station = 'Pharmacy'
          AND DAYOFWEEK(created_at) != 1
        GROUP BY 1
        HAVING p50_min IS NOT NULL
        ORDER BY 1
    """)


def q_opd_pharmacy_lockup():
    """silver: pharmacy P50 by hour × day type (quiet ≤20/day vs busy >40/day).
    Shows when the queue locks up on each day type. Sunday excluded."""
    return run_query("""
        WITH daily_volume AS (
            SELECT
                DATE(created_at)         AS day,
                COUNT(DISTINCT visit_id) AS daily_visits
            FROM HOSPITALS.STAGING.STG_KSH_V3_STATION_EVENTS
            WHERE station = 'Pharmacy'
              AND DAYOFWEEK(created_at) != 1
            GROUP BY 1
        )
        SELECT
            CASE
                WHEN dv.daily_visits <= 20 THEN 'Quiet'
                ELSE                            'Busy'
            END                                        AS day_type,
            HOUR(se.created_at)                        AS hour,
            COUNT(DISTINCT se.visit_id)                AS visits,
            ROUND(MEDIAN(
                CASE WHEN se.wait_minutes BETWEEN 1 AND 240
                     THEN se.wait_minutes END
            ), 0)                                      AS p50_min
        FROM HOSPITALS.STAGING.STG_KSH_V3_STATION_EVENTS se
        JOIN daily_volume dv ON dv.day = DATE(se.created_at)
        WHERE se.station = 'Pharmacy'
          AND DAYOFWEEK(se.created_at) != 1
          AND (dv.daily_visits <= 20 OR dv.daily_visits > 40)
        GROUP BY 1, 2
        HAVING p50_min IS NOT NULL
           AND COUNT(DISTINCT se.visit_id) >= 3
        ORDER BY 1, 2
    """)


def q_opd_pharmacy_heatmap():
    """silver: pharmacy P50 by hour × day of week. Sunday excluded, March 2026+."""
    return run_query("""
        SELECT
            DAYOFWEEK(created_at)  AS dow,
            DAYNAME(created_at)    AS day_name,
            HOUR(created_at)       AS hour,
            ROUND(MEDIAN(
                CASE WHEN wait_minutes BETWEEN 1 AND 240 THEN wait_minutes END
            ), 0)                  AS p50_min,
            COUNT(DISTINCT visit_id) AS visits
        FROM HOSPITALS.STAGING.STG_KSH_V3_STATION_EVENTS
        WHERE station = 'Pharmacy'
          AND DAYOFWEEK(created_at) != 1
        GROUP BY 1, 2, 3
        HAVING p50_min IS NOT NULL
           AND COUNT(DISTINCT visit_id) >= 3
        ORDER BY 1, 3
    """)


def q_opd_pharmacy_volume_band():
    """silver: pharmacy median P50/P25/P75 by daily visit volume band.
    Structural proof: P50 = 64 min even at ≤20 patients/day."""
    return run_query("""
        WITH daily AS (
            SELECT
                DATE(created_at)                                                       AS day,
                COUNT(DISTINCT visit_id)                                               AS daily_visits,
                MEDIAN(CASE WHEN wait_minutes BETWEEN 1 AND 240 THEN wait_minutes END) AS p50,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY
                    CASE WHEN wait_minutes BETWEEN 1 AND 240 THEN wait_minutes END)    AS p25,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY
                    CASE WHEN wait_minutes BETWEEN 1 AND 240 THEN wait_minutes END)    AS p75
            FROM HOSPITALS.STAGING.STG_KSH_V3_STATION_EVENTS
            WHERE station = 'Pharmacy'
            GROUP BY 1
            HAVING p50 IS NOT NULL
        )
        SELECT
            CASE
                WHEN daily_visits <= 20 THEN '≤20/day'
                WHEN daily_visits <= 30 THEN '21–30/day'
                WHEN daily_visits <= 40 THEN '31–40/day'
                ELSE                         '>40/day'
            END                              AS volume_band,
            COUNT(*)                         AS days,
            ROUND(MEDIAN(daily_visits), 0)   AS median_daily_visits,
            ROUND(MEDIAN(p50), 0)            AS median_p50_min,
            ROUND(MEDIAN(p25), 0)            AS p25_min,
            ROUND(MEDIAN(p75), 0)            AS p75_min
        FROM daily
        GROUP BY 1
        ORDER BY MIN(daily_visits)
    """)


def q_opd_station_tat():
    """gold: last 8 complete weeks × 4 stations — for bottleneck ranking + trend chart.
    Returns weekly grain; Python aggregates current/prior 4-week windows."""
    return run_query("""
        WITH settled_weeks AS (
            -- keep only weeks where all 4 stations have >= 30 visits (filters partial/thin weeks)
            SELECT week_start
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_STATION_TAT
            WHERE is_current_week = FALSE
            GROUP BY week_start
            HAVING MIN(visit_n) >= 30
        ),
        ranked_weeks AS (
            SELECT week_start,
                   ROW_NUMBER() OVER (ORDER BY week_start DESC) AS rn
            FROM settled_weeks
        )
        SELECT
            t.week_start,
            t.station,
            t.visit_n,
            t.wait_p50,
            t.coverage_n,
            t.coverage_pct,
            w.rn AS week_rank
        FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_STATION_TAT t
        JOIN ranked_weeks w ON w.week_start = t.week_start
        WHERE w.rn <= 8
        ORDER BY t.week_start DESC, t.station
    """)


def q_opd_station_tat_monthly():
    """gold: monthly median P50 per station from weekly TAT table.
    Excludes partial months and thin weeks (visit_n < 30)."""
    return run_query("""
        SELECT
            DATE_TRUNC('month', week_start)              AS month,
            station,
            ROUND(MEDIAN(wait_p50), 0)                   AS wait_p50,
            SUM(visit_n)                                 AS visit_n
        FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_STATION_TAT
        WHERE is_current_week = FALSE
          AND visit_n >= 30
        GROUP BY 1, 2
        HAVING COUNT(*) >= 2
        ORDER BY 1, 2
    """)


def q_opd_terminal_state_monthly():
    """gold: monthly visit count by terminal state — for outcome mix stacked bar.
    Excludes partial months and ramp months (<200 visits)."""
    return run_query("""
        WITH base AS (
            SELECT
                DATE_TRUNC('month', visit_date)  AS month,
                terminal_state,
                COUNT(*)                         AS visit_n
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_JOURNEY
            WHERE is_settled = TRUE
              AND is_partial_month = FALSE
            GROUP BY 1, 2
        ),
        month_totals AS (
            SELECT month, SUM(visit_n) AS total_n
            FROM base
            GROUP BY 1
        )
        SELECT b.month, b.terminal_state, b.visit_n
        FROM base b
        JOIN month_totals m ON m.month = b.month
        WHERE m.total_n >= 200
        ORDER BY 1, 2
    """)


def q_opd_peak_baseline_summary():
    """gold: journey duration + conversion rate by peak/baseline band.
    Peak = dow×hour averaging >= 8 visits, Baseline = <= 4. Trailing 12 complete months."""
    return run_query("""
        WITH last12 AS (
            SELECT DISTINCT DATE_TRUNC('month', visit_date) AS month
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_JOURNEY
            WHERE is_settled = TRUE AND is_partial_month = FALSE
            ORDER BY 1 DESC LIMIT 12
        ),
        hourly_avg AS (
            SELECT dow_num, arrival_hour, AVG(daily_n) AS avg_visits
            FROM (
                SELECT visit_date, DAYOFWEEKISO(visit_date) AS dow_num,
                       arrival_hour, COUNT(*) AS daily_n
                FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_JOURNEY
                WHERE DATE_TRUNC('month', visit_date) IN (SELECT month FROM last12)
                GROUP BY 1, 2, 3
            ) GROUP BY 1, 2
        ),
        banded AS (
            SELECT
                j.journey_duration_min,
                j.is_admitted,
                CASE WHEN h.avg_visits >= 8 THEN 'peak'
                     WHEN h.avg_visits <= 4 THEN 'baseline'
                     ELSE 'mid' END AS band
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_JOURNEY j
            JOIN hourly_avg h
              ON h.dow_num = DAYOFWEEKISO(j.visit_date)
             AND h.arrival_hour = j.arrival_hour
            WHERE DATE_TRUNC('month', j.visit_date) IN (SELECT month FROM last12)
              AND j.is_settled = TRUE
        )
        SELECT
            band,
            COUNT(*)                                                         AS total_visits,
            ROUND(MEDIAN(CASE WHEN journey_duration_min BETWEEN 1 AND 480
                         THEN journey_duration_min END), 0)                  AS journey_p50,
            ROUND(AVG(CASE WHEN is_admitted = 1 THEN 1.0 ELSE 0 END) * 100, 1) AS conversion_pct
        FROM banded
        WHERE band IN ('peak', 'baseline')
        GROUP BY 1
        ORDER BY 1
    """)


def q_opd_peak_baseline_tat():
    """silver: station wait P50 by peak/baseline band. Trailing 12 complete months."""
    return run_query("""
        WITH last12 AS (
            SELECT DISTINCT DATE_TRUNC('month', visit_date) AS month
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_JOURNEY
            WHERE is_settled = TRUE AND is_partial_month = FALSE
            ORDER BY 1 DESC LIMIT 12
        ),
        hourly_avg AS (
            SELECT dow_num, arrival_hour, AVG(daily_n) AS avg_visits
            FROM (
                SELECT visit_date, DAYOFWEEKISO(visit_date) AS dow_num,
                       arrival_hour, COUNT(*) AS daily_n
                FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_JOURNEY
                WHERE DATE_TRUNC('month', visit_date) IN (SELECT month FROM last12)
                GROUP BY 1, 2, 3
            ) GROUP BY 1, 2
        ),
        visit_bands AS (
            SELECT
                j.visit_id,
                CASE WHEN h.avg_visits >= 8 THEN 'peak'
                     WHEN h.avg_visits <= 4 THEN 'baseline'
                     ELSE 'mid' END AS band
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_JOURNEY j
            JOIN hourly_avg h
              ON h.dow_num = DAYOFWEEKISO(j.visit_date)
             AND h.arrival_hour = j.arrival_hour
            WHERE DATE_TRUNC('month', j.visit_date) IN (SELECT month FROM last12)
              AND j.is_settled = TRUE
        ),
        station_waits AS (
            SELECT e.visit_id, LOWER(TRIM(e.station)) AS station, e.wait_minutes
            FROM HOSPITALS.STAGING.STG_KSH_V3_STATION_EVENTS e
            WHERE DATE_TRUNC('month', DATE(e.time_in)) IN (SELECT month FROM last12)
              AND LOWER(TRIM(e.station)) IN ('doctor','pharmacy','laboratory','radiology')
              AND e.wait_minutes BETWEEN 1 AND 480
            QUALIFY ROW_NUMBER() OVER (PARTITION BY e.visit_id, LOWER(TRIM(e.station))
                                       ORDER BY e.time_in) = 1
        )
        SELECT
            b.band,
            s.station,
            COUNT(*)                                            AS n,
            ROUND(MEDIAN(s.wait_minutes), 0)                   AS wait_p50
        FROM station_waits s
        JOIN visit_bands b ON b.visit_id = s.visit_id
        WHERE b.band IN ('peak', 'baseline')
        GROUP BY 1, 2
        ORDER BY 2, 1
    """)


def q_opd_peak_hours():
    """gold: average daily visits by day-of-week × arrival hour — trailing 12 complete months.
    Averaged per day so month length doesn't distort the colour scale."""
    return run_query("""
        WITH last12 AS (
            SELECT DISTINCT DATE_TRUNC('month', visit_date) AS month
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_JOURNEY
            WHERE is_settled = TRUE AND is_partial_month = FALSE
            ORDER BY 1 DESC
            LIMIT 12
        ),
        daily AS (
            SELECT
                visit_date,
                DAYOFWEEKISO(visit_date)   AS dow_num,
                DAYNAME(visit_date)        AS dow_name,
                arrival_hour,
                COUNT(*)                   AS visit_n
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_JOURNEY
            WHERE DATE_TRUNC('month', visit_date) IN (SELECT month FROM last12)
            GROUP BY 1, 2, 3, 4
        )
        SELECT
            dow_num,
            dow_name,
            arrival_hour,
            ROUND(AVG(visit_n), 1)         AS avg_visits
        FROM daily
        GROUP BY 1, 2, 3
        ORDER BY 1, 3
    """)


def q_opd_arrival_hour():
    """gold: visit count by arrival hour — last 3 complete settled months.
    Used for the OPD arrival shape chart (descriptive, not comparative)."""
    return run_query("""
        WITH last3 AS (
            SELECT DISTINCT DATE_TRUNC('month', visit_date) AS month
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_JOURNEY
            WHERE is_settled = TRUE AND is_partial_month = FALSE
            ORDER BY 1 DESC
            LIMIT 3
        )
        SELECT
            arrival_hour,
            COUNT(*) AS visit_n
        FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_JOURNEY
        WHERE DATE_TRUNC('month', visit_date) IN (SELECT month FROM last3)
        GROUP BY 1
        ORDER BY 1
    """)


def q_opd_journey_duration_monthly():
    """gold: monthly median total journey duration from OPD_JOURNEY.
    journey_duration_min = time from first to last station event + last station wait.
    Capped at 480 min to exclude stale/overnight records."""
    return run_query("""
        SELECT
            DATE_TRUNC('month', visit_date)              AS month,
            ROUND(MEDIAN(journey_duration_min), 0)       AS journey_p50_min,
            COUNT(*)                                     AS visit_n,
            BOOLOR_AGG(is_partial_month)                 AS is_partial
        FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_JOURNEY
        WHERE is_settled = TRUE
          AND journey_duration_min > 0
          AND journey_duration_min <= 480
        GROUP BY 1
        HAVING COUNT(*) >= 200
        ORDER BY 1
    """)


def q_opd_triage_routing():
    """silver STATION_EVENTS: canonical next station after Triage per visit."""
    return run_query("""
        WITH triage AS (
            SELECT visit_id, created_at AS done_at,
                   ROW_NUMBER() OVER (PARTITION BY visit_id ORDER BY created_at) AS rn
            FROM HOSPITALS.STAGING.STG_KSH_V3_STATION_EVENTS
            WHERE station = 'Triage'
        ),
        next_dest AS (
            SELECT t.visit_id, s.station,
                   ROW_NUMBER() OVER (PARTITION BY t.visit_id ORDER BY s.created_at) AS rn
            FROM triage t
            JOIN HOSPITALS.STAGING.STG_KSH_V3_STATION_EVENTS s
              ON s.visit_id = t.visit_id AND s.created_at > t.done_at
            WHERE t.rn = 1
        )
        SELECT
            station,
            COUNT(*)                                                AS visit_count,
            ROUND(COUNT(*) / SUM(COUNT(*)) OVER () * 100, 1)       AS pct
        FROM next_dest
        WHERE rn = 1
        GROUP BY station
        ORDER BY visit_count DESC
    """)


# ── Lab TAT ───────────────────────────────────────────────────────────────────

def q_lab_monthly():
    """gold: LAB and RADIOLOGY monthly TAT — last 12 complete months each, ascending."""
    return run_query("""
        SELECT
            test_type, month, order_count, coverage_n, coverage_pct,
            order_to_published_p50, order_to_published_p90,
            order_to_published_3mo_p50, order_to_published_pct_change,
            pct_within_60min, pct_within_120min, over_1d_count,
            opd_p50_minutes, inpatient_p50_minutes,
            opd_coverage_n, inpatient_coverage_n,
            is_partial_month
        FROM HOSPITALS.REPORTING.RPT_KSH_V3_LAB_TAT
        WHERE is_partial_month = FALSE
        QUALIFY ROW_NUMBER() OVER (PARTITION BY test_type ORDER BY month DESC) <= 12
        ORDER BY test_type, month ASC
    """)


def q_lab_slow_test_share():
    """silver: share of structurally slow tests (UECs/ESR/Liver/Lipid) — current vs prior 28 days."""
    return run_query("""
        SELECT
            CASE
                WHEN DATE(ordered_at) >= DATEADD('day', -28, CURRENT_DATE) THEN 'current'
                WHEN DATE(ordered_at) >= DATEADD('day', -56, CURRENT_DATE) THEN 'prior'
            END                                                         AS window,
            COUNT(*)                                                    AS total_orders,
            COUNT_IF(procedure_name IN (
                'Uecs', 'ESR', 'Liver Panel 1 (Lft)', 'Lipid Profile',
                'Psa', 'Serum Tsh', 'UECS (Dialysis)', 'Urine Albumine Creat'
            ))                                                          AS slow_orders,
            ROUND(COUNT_IF(procedure_name IN (
                'Uecs', 'ESR', 'Liver Panel 1 (Lft)', 'Lipid Profile',
                'Psa', 'Serum Tsh', 'UECS (Dialysis)', 'Urine Albumine Creat'
            )) * 100.0 / NULLIF(COUNT(*), 0), 1)                       AS slow_pct
        FROM HOSPITALS.STAGING.STG_KSH_V3_LAB_INVESTIGATIONS
        WHERE type = 'LAB'
          AND COALESCE(cancelled, 0) = 0
          AND ordered_at IS NOT NULL
          AND DATE(ordered_at) >= DATEADD('day', -56, CURRENT_DATE)
        GROUP BY 1
        HAVING window IS NOT NULL
        ORDER BY 1 DESC
    """)


def q_lab_pharmacy_conversion():
    """silver: % of OPD LAB visits with same-visit pharmacy station — current vs prior 28 days."""
    return run_query("""
        WITH lab_visits AS (
            SELECT
                visit_id,
                CASE
                    WHEN DATE(ordered_at) >= DATEADD('day', -28, CURRENT_DATE) THEN 'current'
                    WHEN DATE(ordered_at) >= DATEADD('day', -56, CURRENT_DATE) THEN 'prior'
                END AS window
            FROM HOSPITALS.STAGING.STG_KSH_V3_LAB_INVESTIGATIONS
            WHERE type = 'LAB'
              AND COALESCE(cancelled, 0) = 0
              AND admission_id IS NULL
              AND ordered_at IS NOT NULL
              AND DATE(ordered_at) >= DATEADD('day', -56, CURRENT_DATE)
            GROUP BY 1, 2
            HAVING window IS NOT NULL
        ),
        pharmacy_visits AS (
            SELECT DISTINCT visit_id
            FROM HOSPITALS.STAGING.STG_KSH_V3_STATION_EVENTS
            WHERE station = 'Pharmacy'
              AND DATE(created_at) >= DATEADD('day', -56, CURRENT_DATE)
        )
        SELECT
            lv.window,
            COUNT(DISTINCT lv.visit_id)                                             AS lab_visits,
            COUNT(DISTINCT CASE WHEN pv.visit_id IS NOT NULL THEN lv.visit_id END)  AS converted,
            ROUND(COUNT(DISTINCT CASE WHEN pv.visit_id IS NOT NULL THEN lv.visit_id END)
                  * 100.0 / NULLIF(COUNT(DISTINCT lv.visit_id), 0), 1)             AS conversion_pct
        FROM lab_visits lv
        LEFT JOIN pharmacy_visits pv ON pv.visit_id = lv.visit_id
        GROUP BY 1
        ORDER BY 1 DESC
    """)


def q_lab_utilisation():
    """gold: facility-level lab utilization — last 2 complete months for delta."""
    return run_query("""
        SELECT
            month,
            MAX(facility_total_encounters)                              AS total_encounters,
            MAX(facility_total_lab_orders)                              AS total_lab_orders,
            MAX(facility_lab_utilization_rate)                          AS lab_utilization_rate,
            ROUND(SUM(lab_orders) * 100.0 / NULLIF(SUM(opd_visits), 0), 1)       AS lab_per_100_opd,
            ROUND(SUM(radiology_orders) * 100.0 / NULLIF(SUM(opd_visits), 0), 1) AS rad_per_100_opd
        FROM HOSPITALS.REPORTING.RPT_KSH_V3_LAB_UTILISATION
        WHERE is_partial_month = FALSE
        GROUP BY month
        QUALIFY ROW_NUMBER() OVER (ORDER BY month DESC) <= 2
        ORDER BY month DESC
    """)


def q_lab_utilisation_trend():
    """gold: facility-level OPD visits + lab/radiology order volume — all complete months, ascending."""
    return run_query("""
        SELECT
            month,
            SUM(opd_visits)       AS opd_visits,
            SUM(lab_orders)       AS lab_orders,
            SUM(radiology_orders) AS radiology_orders
        FROM HOSPITALS.REPORTING.RPT_KSH_V3_LAB_UTILISATION
        WHERE is_partial_month = FALSE
        GROUP BY month
        ORDER BY month ASC
    """)


def q_lab_modality():
    """gold: radiology modality summary — last 2 complete months for MoM delta."""
    return run_query("""
        SELECT
            modality_group                                              AS modality,
            month,
            SUM(orders)                                                 AS orders,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
                (ORDER BY p50_minutes), 0)                             AS median_tat_min,
            SUM(within_60_min)                                         AS within_60_min,
            ROUND(SUM(within_60_min) * 100.0
                  / NULLIF(SUM(resulted_orders), 0), 1)                AS pct_within_60_min
        FROM HOSPITALS.REPORTING.RPT_KSH_V3_LAB_MODALITY
        WHERE is_partial_month = FALSE
        GROUP BY modality_group, month
        QUALIFY DENSE_RANK() OVER (ORDER BY month DESC) <= 2
        ORDER BY month DESC, orders DESC
    """)


def q_xray_coverage_trend():
    """gold: X-ray orders placed vs results recorded — all complete months."""
    return run_query("""
        SELECT month, orders, resulted_orders
        FROM HOSPITALS.REPORTING.RPT_KSH_V3_LAB_MODALITY
        WHERE modality_group = 'X-Ray'
          AND is_partial_month = FALSE
        ORDER BY month ASC
    """)


def q_ultrasound_by_doctor():
    """gold + users: per-doctor ultrasound orders — last 2 complete months for MoM comparison."""
    return run_query("""
        SELECT
            u.full_name,
            d.user_id,
            d.month,
            d.orders
        FROM HOSPITALS.REPORTING.RPT_KSH_V3_LAB_MODALITY_DOCTOR d
        LEFT JOIN HOSPITALS.STAGING.STG_KSH_V3_USERS u ON u.user_id = d.user_id
        WHERE d.modality_group = 'Ultrasound'
          AND d.is_partial_month = FALSE
        QUALIFY DENSE_RANK() OVER (ORDER BY d.month DESC) <= 2
        ORDER BY d.month DESC, d.orders DESC
    """)


def q_lab_completion():
    """gold: all-time lab result completion rate — orders with vs without a result on file."""
    return run_query("""
        SELECT
            SUM(order_count)                                            AS total_orders,
            SUM(coverage_n)                                             AS with_result,
            SUM(order_count) - SUM(coverage_n)                         AS without_result,
            ROUND(SUM(coverage_n) * 100.0
                  / NULLIF(SUM(order_count), 0), 1)                    AS completion_rate,
            ROUND((SUM(order_count) - SUM(coverage_n)) * 100.0
                  / NULLIF(SUM(order_count), 0), 1)                    AS without_result_pct
        FROM HOSPITALS.REPORTING.RPT_KSH_V3_LAB_TAT
        WHERE test_type = 'LAB'
          AND is_partial_month = FALSE
    """)


def q_lab_doctor_conversion():
    """gold: per-doctor lab orders — latest complete month only. No names."""
    return run_query("""
        SELECT
            l.user_id,
            l.opd_visits,
            l.lab_orders,
            l.lab_per_100_opd,
            l.month
        FROM HOSPITALS.REPORTING.RPT_KSH_V3_LAB_UTILISATION l
        WHERE l.is_partial_month = FALSE
        QUALIFY DENSE_RANK() OVER (ORDER BY l.month DESC) = 1
        ORDER BY l.lab_orders DESC
    """)


# ── THEATRE ──────────────────────────────────────────────────────────────────
# All four card queries share the same params pattern:
#   current_month = latest month where schedule_start_time <= now (no future bookings)
#   scope = Operating Theatre, complete months only (month < current_month)
#   completion signal = has_operation (operation record exists)

def q_theatre_elective_completion():
    """Card 1 — Elective completion rate + MoM delta. Operating Theatre, complete months.
    Returns 1 row. Signal: Indicative (has_operation proxy).
    """
    return run_query("""
        WITH params AS (
            SELECT DATE_TRUNC('month', MAX(schedule_start_time))::DATE AS current_month
            FROM HOSPITALS.STAGING.STG_KSH_V3_THEATRE_CASES
            WHERE theatre_id IS NOT NULL
              AND schedule_start_time <= CURRENT_TIMESTAMP()
        ),
        bookings AS (
            SELECT
                booking_id,
                DATE_TRUNC('month', MAX(schedule_start_time))::DATE AS month,
                LOWER(MAX(theatre_name))                            AS theatre_name,
                BOOLOR_AGG(has_operation)                           AS done,
                BOOLOR_AGG(is_emergency)                            AS is_emergency
            FROM HOSPITALS.STAGING.STG_KSH_V3_THEATRE_CASES
            WHERE theatre_id IS NOT NULL
            GROUP BY booking_id
        ),
        monthly AS (
            SELECT
                month,
                COUNT(*)                                            AS elective_due_count,
                COUNT_IF(NOT done)                                  AS non_completion_count,
                ROUND(COUNT_IF(done) * 100.0 / COUNT(*), 1)         AS completion_rate_pct
            FROM bookings CROSS JOIN params
            WHERE theatre_name = 'operating theatre'
              AND NOT is_emergency
              AND month < current_month
            GROUP BY month
        ),
        ranked AS (
            SELECT *, ROW_NUMBER() OVER (ORDER BY month DESC) AS rn FROM monthly
        )
        SELECT
            cur.month, cur.elective_due_count, cur.non_completion_count, cur.completion_rate_pct,
            pri.completion_rate_pct AS prior_rate_pct,
            ROUND(cur.completion_rate_pct - pri.completion_rate_pct, 1) AS mom_delta_pp
        FROM ranked cur
        LEFT JOIN ranked pri ON pri.rn = 2
        WHERE cur.rn = 1
    """)


def q_theatre_procedure_exposure():
    """Chart 2 — Non-completed KES value by procedure, top 5 + Other, last 6 complete months.
    Joins bronze bookings for proc_cash_price and operation_procedure → evaluation_procedures.
    Returns ≤6 rows ordered by nc_value DESC. Signal: Indicative.
    """
    return run_query("""
        WITH params AS (
            SELECT DATE_TRUNC('month', MAX(schedule_start_time))::DATE AS current_month
            FROM HOSPITALS.STAGING.STG_KSH_V3_THEATRE_CASES
            WHERE theatre_id IS NOT NULL
              AND schedule_start_time <= CURRENT_TIMESTAMP()
        ),
        bk AS (
            SELECT
                booking_id,
                DATE_TRUNC('month', MAX(schedule_start_time))::DATE AS month,
                LOWER(MAX(theatre_name))                            AS theatre_name,
                BOOLOR_AGG(has_operation)                           AS done,
                MAX(booking_status) = 'approved'                    AS is_approved,
                BOOLOR_AGG(is_emergency)                            AS is_emergency
            FROM HOSPITALS.STAGING.STG_KSH_V3_THEATRE_CASES
            WHERE theatre_id IS NOT NULL
            GROUP BY booking_id
        ),
        agg AS (
            SELECT
                COALESCE(ep.name, 'Unrecorded')                          AS procedure_name,
                COUNT(*)                                                 AS scheduled,
                COUNT_IF(NOT bk.done)                                    AS non_completions,
                SUM(IFF(NOT bk.done, COALESCE(b.proc_cash_price, 0), 0)) AS nc_value,
                SUM(IFF(NOT bk.done AND bk.is_approved,
                        COALESCE(b.proc_cash_price, 0), 0))              AS value_approved
            FROM bk
            CROSS JOIN params p
            JOIN HOSPITALS.BRONZE.BRZ_KSH_V3_THEATRE_BOOKINGS b ON b.id = bk.booking_id
            LEFT JOIN HOSPITALS.KISUMU_V3_CLEAN.EVALUATION_PROCEDURES ep
                   ON ep.id = b.operation_procedure
            WHERE bk.theatre_name = 'operating theatre'
              AND NOT bk.is_emergency
              AND bk.month >= DATEADD('month', -6, p.current_month)
              AND bk.month <  p.current_month
            GROUP BY 1
        ),
        ranked AS (
            SELECT *, ROW_NUMBER() OVER (ORDER BY nc_value DESC) AS rk
            FROM agg WHERE non_completions > 0
        )
        SELECT
            IFF(rk <= 5, procedure_name, 'Other') AS label,
            SUM(non_completions)                  AS non_completions,
            SUM(scheduled)                        AS scheduled,
            SUM(nc_value)                         AS nc_value,
            SUM(value_approved)                   AS value_approved
        FROM ranked
        GROUP BY 1
        ORDER BY nc_value DESC
    """)


def q_theatre_completion_trend():
    """12-month elective non-completion trend — Operating Theatre only, complete months.
    Returns 12 rows ordered by month ASC. Signal: Indicative.
    """
    return run_query("""
        WITH params AS (
            SELECT DATE_TRUNC('month', MAX(schedule_start_time))::DATE AS current_month
            FROM HOSPITALS.STAGING.STG_KSH_V3_THEATRE_CASES
            WHERE theatre_id IS NOT NULL
              AND schedule_start_time <= CURRENT_TIMESTAMP()
        ),
        bookings AS (
            SELECT
                booking_id,
                DATE_TRUNC('month', MAX(schedule_start_time))::DATE AS month,
                LOWER(MAX(theatre_name))                            AS theatre_name,
                BOOLOR_AGG(has_operation)                           AS done,
                MAX(booking_status) = 'approved'                    AS is_approved,
                BOOLOR_AGG(is_emergency)                            AS is_emergency
            FROM HOSPITALS.STAGING.STG_KSH_V3_THEATRE_CASES
            WHERE theatre_id IS NOT NULL
            GROUP BY booking_id
        )
        SELECT
            month,
            COUNT(*)                                          AS elective_due,
            COUNT_IF(NOT done)                                AS non_completions,
            COUNT_IF(NOT done AND is_approved)                AS of_which_approved,
            ROUND(COUNT_IF(NOT done) * 100.0 / COUNT(*), 1)   AS non_completion_rate_pct
        FROM bookings
        CROSS JOIN params
        WHERE theatre_name = 'operating theatre'
          AND NOT is_emergency
          AND month >= DATEADD('month', -12, current_month)
          AND month <  current_month
        GROUP BY month
        ORDER BY month
    """)


def q_theatre_emergency_rate_trend():
    """6-month monthly emergency rate trend — Operating Theatre only, complete months.
    Returns up to 6 rows ordered by month ASC. Signal: Descriptive.
    """
    return run_query("""
        WITH params AS (
            SELECT DATE_TRUNC('month', MAX(schedule_start_time))::DATE AS current_month
            FROM HOSPITALS.STAGING.STG_KSH_V3_THEATRE_CASES
            WHERE theatre_id IS NOT NULL
              AND schedule_start_time <= CURRENT_TIMESTAMP()
        ),
        bookings AS (
            SELECT
                booking_id,
                DATE_TRUNC('month', MAX(schedule_start_time))::DATE AS month,
                LOWER(MAX(theatre_name))                            AS theatre_name,
                BOOLOR_AGG(is_emergency)                            AS is_emergency
            FROM HOSPITALS.STAGING.STG_KSH_V3_THEATRE_CASES
            WHERE theatre_id IS NOT NULL
            GROUP BY booking_id
        )
        SELECT
            month,
            COUNT(*)                                                        AS total_cases,
            COUNT_IF(is_emergency)                                          AS emergency_cases,
            ROUND(COUNT_IF(is_emergency) * 100.0 / NULLIF(COUNT(*), 0), 1) AS emergency_rate_pct
        FROM bookings
        CROSS JOIN params
        WHERE theatre_name = 'operating theatre'
          AND month >= DATEADD('month', -6, current_month)
          AND month <  current_month
        GROUP BY month
        ORDER BY month ASC
    """)


def q_theatre_emergency_rate():
    """Card 2 — Booked emergency rate, latest complete month + 6-month. Operating Theatre.
    Returns 1 row. Signal: Descriptive.
    """
    return run_query("""
        WITH params AS (
            SELECT DATE_TRUNC('month', MAX(schedule_start_time))::DATE AS current_month
            FROM HOSPITALS.STAGING.STG_KSH_V3_THEATRE_CASES
            WHERE theatre_id IS NOT NULL
              AND schedule_start_time <= CURRENT_TIMESTAMP()
        ),
        bookings AS (
            SELECT
                booking_id,
                DATE_TRUNC('month', MAX(schedule_start_time))::DATE AS month,
                LOWER(MAX(theatre_name))                            AS theatre_name,
                BOOLOR_AGG(is_emergency)                            AS is_emergency
            FROM HOSPITALS.STAGING.STG_KSH_V3_THEATRE_CASES
            WHERE theatre_id IS NOT NULL
            GROUP BY booking_id
        ),
        scoped AS (
            SELECT b.*, MAX(b.month) OVER () AS latest
            FROM bookings b CROSS JOIN params p
            WHERE b.theatre_name = 'operating theatre'
              AND b.month >= DATEADD('month', -6, p.current_month)
              AND b.month <  p.current_month
        )
        SELECT
            MAX(latest)                                                        AS month,
            COUNT_IF(month = latest)                                           AS cases_latest,
            COUNT_IF(month = latest AND is_emergency)                          AS emergency_latest,
            ROUND(COUNT_IF(month = latest AND is_emergency) * 100.0
                  / NULLIF(COUNT_IF(month = latest), 0), 1)                    AS rate_latest_pct,
            COUNT(*)                                                           AS cases_6m,
            COUNT_IF(is_emergency)                                             AS emergency_6m,
            ROUND(COUNT_IF(is_emergency) * 100.0 / COUNT(*), 1)                AS rate_6m_pct
        FROM scoped
    """)


def q_theatre_out_of_hours():
    """Card 3 — Booked out-of-hours performed cases, latest complete month + 6-month. Operating Theatre.
    In-hours = 07:00–17:59 (cfg). Returns 1 row. Signal: Descriptive.
    """
    return run_query("""
        WITH params AS (
            SELECT DATE_TRUNC('month', MAX(schedule_start_time))::DATE AS current_month
            FROM HOSPITALS.STAGING.STG_KSH_V3_THEATRE_CASES
            WHERE theatre_id IS NOT NULL
              AND schedule_start_time <= CURRENT_TIMESTAMP()
        ),
        cfg AS (SELECT 7 AS in_hours_start, 17 AS in_hours_last_hour),
        bookings AS (
            SELECT
                booking_id,
                MAX(schedule_start_time)  AS t,
                LOWER(MAX(theatre_name))  AS theatre_name,
                BOOLOR_AGG(has_operation) AS done
            FROM HOSPITALS.STAGING.STG_KSH_V3_THEATRE_CASES
            WHERE theatre_id IS NOT NULL
            GROUP BY booking_id
        ),
        monthly AS (
            SELECT
                DATE_TRUNC('month', t)::DATE AS month,
                COUNT_IF(done)               AS performed,
                COUNT_IF(done AND HOUR(t) NOT BETWEEN in_hours_start AND in_hours_last_hour) AS ooh
            FROM bookings CROSS JOIN params CROSS JOIN cfg
            WHERE theatre_name = 'operating theatre'
              AND DATE_TRUNC('month', t)::DATE >= DATEADD('month', -6, current_month)
              AND DATE_TRUNC('month', t)::DATE <  current_month
            GROUP BY 1
        )
        SELECT
            MAX(month)                                                   AS month,
            MAX_BY(performed, month)                                     AS performed_latest,
            MAX_BY(ooh, month)                                           AS ooh_latest,
            ROUND(MAX_BY(ooh, month) * 100.0
                  / NULLIF(MAX_BY(performed, month), 0), 1)              AS rate_latest_pct,
            SUM(performed)                                               AS performed_6m,
            SUM(ooh)                                                     AS ooh_6m,
            ROUND(SUM(ooh) * 100.0 / NULLIF(SUM(performed), 0), 1)       AS rate_6m_pct
        FROM monthly
    """)


def q_inpatient_c2b_trend():
    """gold: weekly C2B P50 — aggregated across all wards, last 12 complete weeks.
    consult_to_bed_p50 IS NOT NULL filter removed: wards with breach counts but no median
    were silently excluded, undercounting c2b_n and c2b_over_4h_n."""
    return run_query("""
        WITH wk AS (
            SELECT week_start,
                   ROUND(MEDIAN(consult_to_bed_p50), 1)      AS c2b_p50_hrs,
                   SUM(c2b_timed_n_all)                       AS c2b_n,
                   SUM(c2b_over_4h_n)                         AS c2b_over_4h_n,
                   ROUND(SUM(c2b_over_4h_n) * 100.0
                         / NULLIF(SUM(c2b_timed_n_all), 0), 1) AS pct_over_4h
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_INPATIENT_SUMMARY
            WHERE is_trailing_week = FALSE
            GROUP BY week_start
            QUALIFY ROW_NUMBER() OVER (ORDER BY week_start DESC) <= 12
        )
        SELECT week_start, c2b_p50_hrs, c2b_n, c2b_over_4h_n, pct_over_4h
        FROM wk
        ORDER BY week_start ASC
    """)


def q_inpatient_discharge_tat_trend():
    """gold: weekly discharge TAT P50 (hrs) — facility-wide, last 12 complete weeks.
    discharge_processing_p50 is already in hours. discharge_n uses admission_count proxy."""
    return run_query("""
        WITH wk AS (
            SELECT week_start,
                   ROUND(MEDIAN(discharge_processing_p50), 1) AS disch_p50_hrs,
                   SUM(admission_count)                        AS discharge_n
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_INPATIENT_SUMMARY
            WHERE is_trailing_week = FALSE
              AND discharge_processing_p50 IS NOT NULL
            GROUP BY week_start
            QUALIFY ROW_NUMBER() OVER (ORDER BY week_start DESC) <= 12
        )
        SELECT week_start, disch_p50_hrs, discharge_n
        FROM wk
        ORDER BY week_start ASC
    """)


def q_inpatient_alos_ward():
    """silver: ALOS and discharge count by ward — latest complete month. Overnight only."""
    return run_query("""
        WITH latest AS (
            SELECT MAX(month_start) AS m
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_BOR_MONTHLY
            WHERE ward_id = 0 AND is_partial_month = FALSE
        )
        SELECT w.ward_name,
               COUNT(*)                   AS discharges,
               ROUND(AVG(a.los_days), 1)  AS alos_days
        FROM HOSPITALS.STAGING.STG_KSH_V3_INPATIENT_ADMISSIONS a
        JOIN HOSPITALS.STAGING.STG_KSH_V3_WARD_CONFIG w ON w.ward_id = a.ward_id
        CROSS JOIN latest l
        WHERE a.is_open_admission = FALSE
          AND a.admission_type_id NOT IN (10, 16)
          AND NOT COALESCE(a.discharged_at < a.admitted_at, FALSE)
          AND a.los_days IS NOT NULL
          AND DATE_TRUNC('month', a.discharged_at::DATE) = l.m
        GROUP BY w.ward_name
        HAVING COUNT(*) >= 3
        ORDER BY alos_days DESC
    """)


def q_opd_inter_station_transit():
    """silver: transit time between consecutive station pairs — trailing 3 complete months.
    Transit = time_in of next station minus time_in of current station.
    Capped at 480 min (8 hrs) to exclude inter-day gaps.
    """
    return run_query("""
        WITH last3 AS (
            SELECT DISTINCT DATE_TRUNC('month', visit_date) AS month
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_FLOW
            WHERE is_settled = TRUE AND is_partial_month = FALSE
            ORDER BY 1 DESC
            LIMIT 3
        ),
        pairs AS (
            SELECT
                visit_id,
                LOWER(TRIM(station))                                                           AS from_station,
                LOWER(TRIM(LEAD(station) OVER (PARTITION BY visit_id ORDER BY time_in)))       AS to_station,
                DATEDIFF('minute', time_in,
                    LEAD(time_in) OVER (PARTITION BY visit_id ORDER BY time_in))               AS transit_min
            FROM HOSPITALS.STAGING.STG_KSH_V3_STATION_EVENTS
            WHERE DATE_TRUNC('month', DATE(time_in)) IN (SELECT month FROM last3)
              AND time_in IS NOT NULL
        )
        SELECT
            from_station,
            to_station,
            COUNT(*)                                                                       AS visit_n,
            ROUND(MEDIAN(transit_min), 0)                                                  AS p50_min,
            ROUND(PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY transit_min), 0)             AS p90_min
        FROM pairs
        WHERE to_station IS NOT NULL
          AND transit_min BETWEEN 1 AND 480
          AND from_station IN ('doctor', 'laboratory', 'radiology', 'triage', 'reception')
          AND to_station   IN ('doctor', 'laboratory', 'radiology', 'pharmacy', 'triage')
        GROUP BY 1, 2
        HAVING COUNT(*) >= 30
        ORDER BY visit_n DESC
    """)


def q_opd_journey_sankey():
    """gold: pre-aggregated patient journey counts for Sankey chart.
    Driving table: RPT_KSH_V3_OPD_JOURNEY (all OPD visits, is_settled = TRUE).
    Returns 1 row. terminal_state drives buckets; intermediate counts derived.
    doctor_visit_count = service_count - direct_ancillary_count - incomplete_opd_count.
    Verification: admitted_count + with_rx_count + no_rx_count = doctor_visit_count.
    admitted splits: admitted_inpatient (open admission) + admitted_discharged (closed).
    """
    return run_query("""
        SELECT
            COUNT(*)                                                                AS total_visits,
            SUM(CASE WHEN terminal_state = 'LWBS — No Contact'          THEN 1 ELSE 0 END) AS lwbs_count,
            SUM(has_station_events)                                                 AS service_count,
            SUM(CASE WHEN terminal_state = 'Direct Ancillary'            THEN 1 ELSE 0 END) AS direct_ancillary_count,
            SUM(CASE WHEN terminal_state = 'Incomplete OPD'              THEN 1 ELSE 0 END) AS incomplete_opd_count,
            SUM(is_admitted)                                                        AS admitted_count,
            SUM(CASE WHEN terminal_state = 'Admitted — Inpatient'        THEN 1 ELSE 0 END) AS admitted_inpatient_count,
            SUM(CASE WHEN terminal_state = 'Admitted — Discharged'       THEN 1 ELSE 0 END) AS admitted_discharged_count,
            SUM(CASE WHEN terminal_state = 'OPD Discharge — With Rx'     THEN 1 ELSE 0 END) AS with_rx_count,
            SUM(CASE WHEN terminal_state = 'OPD Discharge — No Rx'       THEN 1 ELSE 0 END) AS no_rx_count
        FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_JOURNEY
        WHERE visit_month = (
            SELECT MAX(visit_month)
            FROM HOSPITALS.REPORTING.RPT_KSH_V3_OPD_JOURNEY
            WHERE is_settled = TRUE AND is_partial_month = FALSE
        )
    """)


def q_theatre_value_exposure():
    """Card 4 — KES value of elective cases not completed, latest month + 6-month. Operating Theatre.
    Price = proc_cash_price at booking (not billed). ~46% of 6-mo exposure is 'approved' status (unconfirmed).
    Returns 1 row. Signal: Indicative.
    """
    return run_query("""
        WITH params AS (
            SELECT DATE_TRUNC('month', MAX(schedule_start_time))::DATE AS current_month
            FROM HOSPITALS.STAGING.STG_KSH_V3_THEATRE_CASES
            WHERE theatre_id IS NOT NULL
              AND schedule_start_time <= CURRENT_TIMESTAMP()
        ),
        bookings AS (
            SELECT
                booking_id,
                DATE_TRUNC('month', MAX(schedule_start_time))::DATE AS month,
                LOWER(MAX(theatre_name))                            AS theatre_name,
                COALESCE(MAX(proc_cash_price), 0)                   AS cash,
                BOOLOR_AGG(has_operation)                           AS done,
                MAX(booking_status) = 'approved'                    AS is_approved,
                BOOLOR_AGG(is_emergency)                            AS is_emergency
            FROM HOSPITALS.STAGING.STG_KSH_V3_THEATRE_CASES
            WHERE theatre_id IS NOT NULL
            GROUP BY booking_id
        ),
        scoped AS (
            SELECT b.*, MAX(b.month) OVER () AS latest
            FROM bookings b CROSS JOIN params p
            WHERE b.theatre_name = 'operating theatre'
              AND NOT b.is_emergency
              AND b.month >= DATEADD('month', -6, p.current_month)
              AND b.month <  p.current_month
        )
        SELECT
            MAX(latest)                                                          AS month,
            COUNT_IF(month = latest AND NOT done)                                AS nc_cases_latest,
            SUM(IFF(month = latest AND NOT done, cash, 0))                       AS nc_value_latest,
            ROUND(SUM(IFF(month = latest AND NOT done, cash, 0)) * 100.0
                  / NULLIF(SUM(IFF(month = latest, cash, 0)), 0), 1)             AS exposure_latest_pct,
            COUNT_IF(NOT done)                                                   AS nc_cases_6m,
            SUM(IFF(NOT done, cash, 0))                                          AS nc_value_6m,
            ROUND(SUM(IFF(NOT done, cash, 0)) * 100.0 / NULLIF(SUM(cash), 0), 1) AS exposure_6m_pct,
            SUM(IFF(NOT done AND is_approved, cash, 0))                          AS value_approved_6m
        FROM scoped
    """)


def q_theatre_revenue_trend():
    """silver: 12-month monthly elective non-completion revenue exposure. Operating Theatre.
    Columns: month, elective_booked, non_completions, nc_value, approved_nc.
    Signal: Indicative. Price at booking, not billed.
    """
    return run_query("""
        WITH params AS (
            SELECT DATE_TRUNC('month', MAX(schedule_start_time))::DATE AS current_month
            FROM HOSPITALS.STAGING.STG_KSH_V3_THEATRE_CASES
            WHERE theatre_id IS NOT NULL
              AND schedule_start_time <= CURRENT_TIMESTAMP()
        ),
        bookings AS (
            SELECT
                booking_id,
                DATE_TRUNC('month', MAX(schedule_start_time))::DATE AS month,
                LOWER(MAX(theatre_name))                            AS theatre_name,
                COALESCE(MAX(proc_cash_price), 0)                   AS cash,
                BOOLOR_AGG(has_operation)                           AS done,
                MAX(booking_status) = 'approved'                    AS is_approved,
                BOOLOR_AGG(is_emergency)                            AS is_emergency
            FROM HOSPITALS.STAGING.STG_KSH_V3_THEATRE_CASES
            WHERE theatre_id IS NOT NULL
            GROUP BY booking_id
        )
        SELECT
            month,
            COUNT(*)                                AS elective_booked,
            COUNT_IF(NOT done)                      AS non_completions,
            SUM(IFF(NOT done, cash, 0))             AS nc_value,
            COUNT_IF(NOT done AND is_approved)      AS approved_nc
        FROM bookings
        CROSS JOIN params
        WHERE theatre_name = 'operating theatre'
          AND NOT is_emergency
          AND month >= DATEADD('month', -12, current_month)
          AND month <  current_month
        GROUP BY month
        ORDER BY month ASC
    """)


def q_theatre_procedure_nc():
    """silver+bronze: top 15 procedures by non-completion count, last 12 months. Operating Theatre.
    Joins THEATRE_BOOKINGS → EVALUATION_PROCEDURES to get name on non-completed bookings.
    Columns: procedure_name, times_booked, times_not_completed, nc_rate_pct, nc_value.
    Signal: Indicative — has_operation proxy; proc_cash_price at booking not billed.
    """
    return run_query("""
        WITH params AS (
            SELECT DATE_TRUNC('month', MAX(schedule_start_time))::DATE AS current_month
            FROM HOSPITALS.STAGING.STG_KSH_V3_THEATRE_CASES
            WHERE theatre_id IS NOT NULL
              AND schedule_start_time <= CURRENT_TIMESTAMP()
        ),
        bookings AS (
            SELECT
                t.booking_id,
                DATE_TRUNC('month', MAX(t.schedule_start_time))::DATE AS month,
                LOWER(MAX(t.theatre_name))                            AS theatre_name,
                COALESCE(TRY_TO_NUMBER(MAX(t.proc_cash_price)), 0)   AS cash,
                BOOLOR_AGG(t.has_operation)                           AS done,
                BOOLOR_AGG(t.is_emergency)                            AS is_emergency,
                ep.name                                               AS procedure_name
            FROM HOSPITALS.STAGING.STG_KSH_V3_THEATRE_CASES t
            LEFT JOIN HOSPITALS.KISUMU_V3_CLEAN.THEATRE_BOOKINGS tb
                   ON t.booking_id = tb.id
            LEFT JOIN HOSPITALS.KISUMU_V3_CLEAN.EVALUATION_PROCEDURES ep
                   ON tb.operation_procedure = ep.id
            WHERE t.theatre_id IS NOT NULL
            GROUP BY t.booking_id, ep.name
        )
        SELECT
            COALESCE(procedure_name, '(unnamed)')                           AS procedure_name,
            COUNT(*)                                                         AS times_booked,
            COUNT_IF(NOT done)                                               AS times_not_completed,
            ROUND(COUNT_IF(NOT done) * 100.0 / NULLIF(COUNT(*), 0), 0)      AS nc_rate_pct,
            SUM(IFF(NOT done, cash, 0))                                      AS nc_value
        FROM bookings
        CROSS JOIN params
        WHERE theatre_name = 'operating theatre'
          AND NOT is_emergency
          AND month >= DATEADD('month', -12, current_month)
          AND month <  current_month
        GROUP BY procedure_name
        HAVING COUNT_IF(NOT done) > 0
        ORDER BY times_not_completed DESC, nc_value DESC
        LIMIT 15
    """)
