"""
Accounts-receivable and collections queries.
AR snapshot uses the two-clock model: Undispatched (our delay) vs
Dispatched-awaiting-payment (insurer delay).
"""

from __future__ import annotations

import pandas as pd

from utils.snowflake_conn import run_query

UNKNOWN_LABEL = "⚠ Unknown (NULL company_id)"
STATE_UNDISPATCHED = "Undispatched — Claim Not Sent"
STATE_DISPATCHED = "Dispatched — Awaiting Payment"


# ── AR snapshot per insurer × state ───────────────────────────────────────────

def get_ar_snapshot() -> pd.DataFrame:
    return run_query("""
        SELECT
            insurer_label, payer_class, ar_state,
            open_invoices, outstanding_kes,
            avg_days_outstanding, median_days_outstanding,
            avg_days_to_dispatch, pct_of_insurer_ar,
            oldest_invoice, newest_invoice, snapshot_date
        FROM HOSPITALS.REPORTING.RPT_REV_AR_SNAPSHOT
    """)


def get_ar_aging() -> pd.DataFrame:
    return run_query("""
        SELECT
            insurer_label, payer_class, ar_state,
            aging_bucket, bucket_sort, invoice_count, outstanding_kes, avg_days
        FROM HOSPITALS.REPORTING.RPT_REV_AR_AGING
    """)


# ── Dispatch-rate trend (the Sept-2025 collapse) ──────────────────────────────

def get_dispatch_trend() -> pd.DataFrame:
    return run_query("""
        SELECT
            rev_month,
            SUM(invoice_count)        AS insurer_invoices,
            SUM(dispatched_count)     AS dispatched,
            SUM(undispatched_count)   AS undispatched,
            SUM(undispatched)         AS undispatched_kes,
            ROUND(SUM(dispatched_count) * 100.0
                  / NULLIF(SUM(invoice_count), 0), 1) AS dispatch_rate_pct
        FROM HOSPITALS.REPORTING.RPT_REV_PAYER_MONTHLY
        WHERE for_cash = 0
        GROUP BY 1
        ORDER BY 1
    """)


# ── Unknown-insurer (NULL company_id) blind spot over time ────────────────────

def get_unknown_insurer_trend() -> pd.DataFrame:
    return run_query("""
        SELECT
            rev_month,
            invoice_count,
            invoiced      AS blind_spot_kes,
            undispatched  AS undispatched_kes
        FROM HOSPITALS.REPORTING.RPT_REV_PAYER_MONTHLY
        WHERE payer_label ILIKE '%NULL company_id%'
        ORDER BY rev_month
    """)


# ── Collections & channels ────────────────────────────────────────────────────

def get_data_cutoff() -> pd.Timestamp:
    """
    Most recent invoice date in the source data. Used to age receivables against
    the data cutoff (not today's date), and to strip the data-lag from the
    reporting snapshot's day-counts.
    """
    df = run_query("SELECT MAX(invoice_date)::DATE AS cutoff FROM HOSPITALS.STAGING.STG_INVOICES")
    return pd.to_datetime(df.iloc[0, 0])


def get_undispatched_aging() -> pd.DataFrame:
    """
    True age distribution of the undispatched insurer book, aged from invoice
    date to the DATA CUTOFF (max invoice date) rather than CURRENT_DATE.

    The reporting snapshot ages to today, which — because the data ends ~4 months
    before today — collapses every balance into the 90+ bucket. Re-aging at
    invoice grain against the cutoff recovers the real distribution and the
    fresh-vs-stale split that drives submission-deadline urgency.
    """
    return run_query("""
        WITH cutoff AS (SELECT MAX(invoice_date) AS mx FROM HOSPITALS.STAGING.STG_INVOICES)
        SELECT
            CASE
                WHEN DATEDIFF('day', invoice_date, (SELECT mx FROM cutoff)) <= 30  THEN '0–30 days'
                WHEN DATEDIFF('day', invoice_date, (SELECT mx FROM cutoff)) <= 60  THEN '31–60 days'
                WHEN DATEDIFF('day', invoice_date, (SELECT mx FROM cutoff)) <= 90  THEN '61–90 days'
                ELSE '90+ days'
            END                                                    AS aging_bucket,
            COUNT(*)                                               AS invoices,
            SUM(COALESCE(balance, invoice_amount))                 AS outstanding_kes
        FROM HOSPITALS.STAGING.STG_INVOICES
        WHERE for_cash = 0
          AND dispatch_id IS NULL
          AND COALESCE(balance, invoice_amount) > 0
          AND COALESCE(include_in_reporting, TRUE)
        GROUP BY 1
    """)


def get_collections_monthly() -> pd.DataFrame:
    return run_query("""
        SELECT
            rev_month, payment_count,
            total_collected, total_invoiced, collection_rate_pct,
            cash, mpesa, card, pesapal, cheque, patient_account,
            cash_pct, mpesa_pct, card_pct, pesapal_pct,
            total_waived, total_discounted, mom_collected_delta
        FROM HOSPITALS.REPORTING.RPT_REV_COLLECTIONS_MONTHLY
        ORDER BY rev_month
    """)
