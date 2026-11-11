"""
Revenue, payer, service-line, timing and item queries.
All read from HOSPITALS.REPORTING.rpt_rev_* (B's reporting layer), verified
against the live schema. Fully-qualified table names, cached 1h via run_query.
"""

from __future__ import annotations

import pandas as pd

from utils.snowflake_conn import run_query

# Billing go-live. Apr–Aug 2024 are pre-go-live ramp (e.g. Apr 2024 = 21
# invoices, 0 patients) and are excluded from every trend so they don't drag
# rolling averages and growth rates.
GO_LIVE = "2024-09-01"


# ── Monthly billing summary (master fact) ─────────────────────────────────────

def get_monthly_summary() -> pd.DataFrame:
    """
    One row per calendar month (Oct 2025 already excluded in the table).
    Returns full history; the pre-go-live ramp is filtered downstream in
    analytics.prepare_monthly() so the raw table stays inspectable.
    """
    return run_query("""
        SELECT
            rev_month,
            total_invoiced, cash_invoiced, insurer_invoiced,
            invoice_count, cash_invoice_count, insurer_invoice_count,
            unique_patients,
            avg_invoice_amount, avg_daily_revenue,
            arpu_invoiced, arpu_collected,
            insurer_pct_of_total,
            total_collected, collection_rate_pct,
            undispatched_ar, dispatched_unpaid_ar,
            dispatched_count, undispatched_count, dispatch_rate_pct,
            credit_note_value
        FROM HOSPITALS.REPORTING.RPT_REV_MONTHLY_SUMMARY
        ORDER BY rev_month
    """)


# ── Service lines ─────────────────────────────────────────────────────────────

def get_service_line_monthly() -> pd.DataFrame:
    """Service line × month (aggregated across billing_source)."""
    return run_query("""
        SELECT
            rev_month,
            service_line,
            SUM(revenue)     AS revenue,
            SUM(line_items)  AS line_items,
            SUM(invoices)    AS invoices
        FROM HOSPITALS.REPORTING.RPT_REV_SERVICE_LINE_MONTHLY
        GROUP BY 1, 2
        ORDER BY 1, 3 DESC
    """)


# ── Timing ────────────────────────────────────────────────────────────────────

def get_timing_dow() -> pd.DataFrame:
    return run_query("""
        SELECT
            day_of_week_num, day_of_week_name,
            AVG(avg_daily_invoiced)  AS avg_daily_invoiced,
            SUM(total_invoiced)      AS total_invoiced,
            SUM(data_days)           AS data_days
        FROM HOSPITALS.REPORTING.RPT_REV_TIMING
        GROUP BY 1, 2
        ORDER BY 1
    """)


def get_timing_wom() -> pd.DataFrame:
    return run_query("""
        SELECT
            week_of_month,
            ROUND(SUM(total_invoiced) / NULLIF(SUM(data_days), 0), 0) AS effective_daily_avg,
            SUM(total_invoiced) AS total_invoiced
        FROM HOSPITALS.REPORTING.RPT_REV_TIMING
        GROUP BY 1
        ORDER BY
            CASE week_of_month
                WHEN 'Week 1 (1–7)'   THEN 1
                WHEN 'Week 2 (8–14)'  THEN 2
                WHEN 'Week 3 (15–21)' THEN 3
                ELSE 4 END
    """)


# ── Top items ─────────────────────────────────────────────────────────────────

def get_top_items(limit: int = 20) -> pd.DataFrame:
    return run_query(f"""
        SELECT
            service_line, item_name, store_code, item_classify,
            times_billed, unique_invoices, total_revenue,
            avg_unit_price, median_unit_price, first_billed, last_billed
        FROM HOSPITALS.REPORTING.RPT_REV_TOP_ITEMS
        ORDER BY total_revenue DESC
        LIMIT {int(limit)}
    """)


# ── Payer mix & concentration ────────────────────────────────────────────────

def get_payer_type_trend() -> pd.DataFrame:
    """Cash vs insurer invoiced by month."""
    return run_query("""
        SELECT
            rev_month,
            SUM(CASE WHEN for_cash = 1 THEN invoiced ELSE 0 END) AS cash_invoiced,
            SUM(CASE WHEN for_cash = 0 THEN invoiced ELSE 0 END) AS insurer_invoiced,
            SUM(invoiced) AS total_invoiced
        FROM HOSPITALS.REPORTING.RPT_REV_PAYER_MONTHLY
        GROUP BY 1
        ORDER BY 1
    """)


def get_insurer_concentration() -> pd.DataFrame:
    """
    All-time insurer invoiced with cumulative % (Pareto). Insurer rows only.
    """
    return run_query("""
        WITH t AS (
            SELECT
                payer_label, payer_class,
                SUM(invoiced)         AS total_invoiced,
                SUM(undispatched)     AS undispatched_kes,
                SUM(dispatched)       AS dispatched_kes,
                ROUND(SUM(dispatched_count) * 100.0
                      / NULLIF(SUM(invoice_count), 0), 1) AS dispatch_rate_pct
            FROM HOSPITALS.REPORTING.RPT_REV_PAYER_MONTHLY
            WHERE for_cash = 0
            GROUP BY 1, 2
        )
        SELECT
            payer_label, payer_class, total_invoiced,
            undispatched_kes, dispatched_kes, dispatch_rate_pct,
            ROUND(total_invoiced / NULLIF(SUM(total_invoiced) OVER (), 0) * 100, 2) AS pct_of_insurer,
            ROUND(SUM(total_invoiced) OVER (
                ORDER BY total_invoiced DESC
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) / NULLIF(SUM(total_invoiced) OVER (), 0) * 100, 2) AS cumulative_pct
        FROM t
        ORDER BY total_invoiced DESC
    """)


def get_sha_trend() -> pd.DataFrame:
    """SHA share of insurer invoiced by month."""
    return run_query("""
        SELECT
            rev_month,
            SUM(CASE WHEN payer_label ILIKE '%SHA%' OR payer_label ILIKE '%Social Health%'
                     THEN invoiced ELSE 0 END) AS sha_invoiced,
            SUM(invoiced) AS insurer_invoiced,
            ROUND(SUM(CASE WHEN payer_label ILIKE '%SHA%' OR payer_label ILIKE '%Social Health%'
                           THEN invoiced ELSE 0 END)
                  / NULLIF(SUM(invoiced), 0) * 100, 1) AS sha_pct
        FROM HOSPITALS.REPORTING.RPT_REV_PAYER_MONTHLY
        WHERE for_cash = 0
        GROUP BY 1
        ORDER BY 1
    """)


# ── Patient value (Pareto) ────────────────────────────────────────────────────

def get_patient_pareto_breakpoints() -> pd.DataFrame:
    return run_query("""
        WITH ranked AS (
            SELECT
                patient_id, cumulative_pct_collected,
                ROW_NUMBER() OVER (ORDER BY total_collected DESC) AS rk,
                COUNT(*) OVER () AS total_patients
            FROM HOSPITALS.REPORTING.RPT_REV_PATIENT_VALUE
            WHERE total_collected > 0
        )
        SELECT
            threshold,
            MIN(rk)            AS patients_needed,
            MIN(total_patients) AS total_patients,
            ROUND(MIN(rk) * 100.0 / MIN(total_patients), 1) AS pct_of_patients
        FROM ranked
        CROSS JOIN (SELECT 50 AS threshold UNION ALL SELECT 70
                    UNION ALL SELECT 80 UNION ALL SELECT 90) t
        WHERE cumulative_pct_collected >= threshold
        GROUP BY threshold
        ORDER BY threshold
    """)
