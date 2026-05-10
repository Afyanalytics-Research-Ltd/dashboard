"""
Cash Integrity & Forensics — all 9 analyses.
"""

import pandas as pd
from connect_to_snowflake import run_query_


# ── Config ─────────────────────────────────────────────────────────────────────

SHIFTS   = "hospitals.xanalife_clean.reception_reception_shifts"
GO_LIVE  = "'2025-09-01'"
EXCL_SYS = "(5, 1719, 1741, 1744, 1746, 1739)"
EXCL_SUP = "('sudo', 'Mrs. Xana  Admin', 'Mrs. Carole  Herzog')"


# ── Closed-shift analyses ───────────────────────────────────────────────────────

SQL_PARETO = f"""
WITH user_variance AS (
    SELECT
        user_id,
        SUM(TRY_TO_NUMBER(closing_variance))                                   AS net_variance,
        COUNT(*)                                                                AS shifts,
        SUM(TRY_TO_NUMBER(system_closing_balance))                              AS cash_at_risk,
        ROUND(SUM(TRY_TO_NUMBER(closing_variance)) /
              NULLIF(SUM(TRY_TO_NUMBER(system_closing_balance)), 0) * 100, 2)  AS variance_pct
    FROM {SHIFTS}
    WHERE TRY_TO_TIMESTAMP(created_at) >= {GO_LIVE}
      AND closed_at IS NOT NULL AND closed_at != ''
      AND user_id NOT IN {EXCL_SYS}
      AND confirmed_by NOT IN {EXCL_SUP}
      AND TRY_TO_NUMBER(cashier_closing_balance) > 0
    GROUP BY user_id
),
loss_pool AS (
    SELECT ABS(SUM(CASE WHEN net_variance < 0 THEN net_variance ELSE 0 END)) AS total_losses
    FROM user_variance
)
SELECT
    u.user_id,
    u.net_variance,
    u.shifts,
    u.cash_at_risk,
    u.variance_pct,
    ROUND(u.net_variance / NULLIF(l.total_losses, 0) * 100, 1) AS pct_of_loss_pool
FROM user_variance u
CROSS JOIN loss_pool l
ORDER BY u.net_variance ASC
"""

SQL_MONTHLY_TREND = f"""
SELECT
    DATE_TRUNC('month', TRY_TO_TIMESTAMP(shift_date))              AS month,
    COUNT(*)                                                         AS shifts,
    SUM(TRY_TO_NUMBER(closing_variance))                             AS net_variance,
    SUM(TRY_TO_NUMBER(system_closing_balance))                       AS cash_at_risk,
    ROUND(SUM(TRY_TO_NUMBER(closing_variance)) /
          NULLIF(SUM(TRY_TO_NUMBER(system_closing_balance)),0)*100,2) AS variance_pct
FROM {SHIFTS}
WHERE TRY_TO_TIMESTAMP(created_at) >= {GO_LIVE}
  AND closed_at IS NOT NULL AND closed_at != ''
  AND user_id NOT IN {EXCL_SYS}
  AND confirmed_by NOT IN {EXCL_SUP}
  AND TRY_TO_NUMBER(cashier_closing_balance) > 0
GROUP BY 1
ORDER BY 1
"""

SQL_ANOMALOUS = f"""
SELECT
    id                                               AS shift_id,
    user_id,
    shift_date,
    confirmed_by                                     AS supervisor,
    TRY_TO_NUMBER(total_cash)                        AS total_cash,
    TRY_TO_NUMBER(system_closing_balance)            AS system_closing,
    TRY_TO_NUMBER(cashier_closing_balance)           AS cashier_closing,
    TRY_TO_NUMBER(closing_variance)                  AS variance,
    TRY_TO_NUMBER(opening_balance)                   AS opening_balance,
    opened_at,
    closed_at
FROM {SHIFTS}
WHERE TRY_TO_TIMESTAMP(created_at) >= {GO_LIVE}
  AND closed_at IS NOT NULL AND closed_at != ''
  AND TRY_TO_NUMBER(cashier_closing_balance) > 0
  AND ABS(TRY_TO_NUMBER(closing_variance)) > ABS(TRY_TO_NUMBER(system_closing_balance))
  AND confirmed_by NOT IN {EXCL_SUP}
ORDER BY TRY_TO_NUMBER(closing_variance) ASC
"""

SQL_SUPERVISOR = f"""
SELECT
    confirmed_by                                                    AS supervisor,
    COUNT(*)                                                         AS shifts_confirmed,
    SUM(TRY_TO_NUMBER(closing_variance))                             AS total_variance_under_supervision,
    ROUND(AVG(TRY_TO_NUMBER(closing_variance)), 0)                   AS avg_variance_per_shift,
    SUM(TRY_TO_NUMBER(system_closing_balance))                       AS cash_at_risk_supervised
FROM {SHIFTS}
WHERE TRY_TO_TIMESTAMP(created_at) >= {GO_LIVE}
  AND closed_at IS NOT NULL AND closed_at != ''
  AND user_id NOT IN {EXCL_SYS}
  AND confirmed_by NOT IN {EXCL_SUP}
  AND TRY_TO_NUMBER(cashier_closing_balance) > 0
GROUP BY confirmed_by
ORDER BY total_variance_under_supervision ASC
"""

SQL_UNCLOSED_EXPOSURE = f"""
SELECT
    COUNT(*)                                              AS unclosed_shifts,
    SUM(TRY_TO_NUMBER(total_sales))                       AS revenue_in_unclosed,
    ROUND(AVG(TRY_TO_NUMBER(total_sales)), 0)             AS avg_revenue_unclosed,
    COUNT(DISTINCT user_id)                               AS stations_with_unclosed
FROM {SHIFTS}
WHERE TRY_TO_TIMESTAMP(created_at) >= {GO_LIVE}
  AND (closed_at IS NULL OR closed_at = '')
  AND user_id NOT IN {EXCL_SYS}
"""


# ── Unclosed-shift drill-down ───────────────────────────────────────────────────

SQL_UNCLOSED_BY_STATION = f"""
SELECT
    user_id,
    COUNT(*)                                                            AS total_shifts,
    COUNT(CASE WHEN closed_at IS NULL OR closed_at = '' THEN 1 END)    AS unclosed,
    COUNT(CASE WHEN closed_at IS NOT NULL AND closed_at != '' THEN 1 END) AS closed,
    ROUND(
        COUNT(CASE WHEN closed_at IS NULL OR closed_at = '' THEN 1 END) * 100.0 /
        NULLIF(COUNT(*), 0), 1)                                         AS unclosed_rate_pct,
    ROUND(SUM(CASE WHEN closed_at IS NULL OR closed_at = ''
              THEN TRY_TO_NUMBER(total_sales) ELSE 0 END), 0)           AS revenue_unclosed_kes
FROM {SHIFTS}
WHERE TRY_TO_TIMESTAMP(created_at) >= {GO_LIVE}
  AND user_id NOT IN {EXCL_SYS}
GROUP BY user_id
ORDER BY unclosed_rate_pct DESC
"""

SQL_UNCLOSED_TREND = f"""
SELECT
    DATE_TRUNC('month', TRY_TO_TIMESTAMP(created_at))                  AS month,
    COUNT(*)                                                            AS total_shifts,
    COUNT(CASE WHEN closed_at IS NULL OR closed_at = '' THEN 1 END)    AS unclosed_shifts,
    ROUND(
        COUNT(CASE WHEN closed_at IS NULL OR closed_at = '' THEN 1 END) * 100.0 /
        NULLIF(COUNT(*), 0), 1)                                         AS unclosed_rate_pct,
    ROUND(SUM(CASE WHEN closed_at IS NULL OR closed_at = ''
              THEN TRY_TO_NUMBER(total_sales) ELSE 0 END), 0)           AS revenue_at_risk_kes
FROM {SHIFTS}
WHERE TRY_TO_TIMESTAMP(created_at) >= {GO_LIVE}
  AND user_id NOT IN {EXCL_SYS}
GROUP BY 1
ORDER BY 1
"""

SQL_LONG_OPEN = f"""
SELECT
    id                                         AS shift_id,
    user_id,
    shift_date,
    opened_at,
    closed_at,
    DATEDIFF('hour',
        TRY_TO_TIMESTAMP(opened_at),
        COALESCE(TRY_TO_TIMESTAMP(closed_at), CURRENT_TIMESTAMP())) AS hours_open,
    TRY_TO_NUMBER(total_sales)                 AS revenue_kes,
    confirmed_by                               AS supervisor
FROM {SHIFTS}
WHERE TRY_TO_TIMESTAMP(created_at) >= {GO_LIVE}
  AND user_id NOT IN {EXCL_SYS}
  AND DATEDIFF('hour',
        TRY_TO_TIMESTAMP(opened_at),
        COALESCE(TRY_TO_TIMESTAMP(closed_at), CURRENT_TIMESTAMP())) > 12
ORDER BY hours_open DESC
"""

SQL_DAILY_COMPLIANCE = f"""
SELECT
    id                                                                  AS shift_id,
    user_id,
    shift_date,
    opened_at,
    TRY_TO_NUMBER(total_sales)                                          AS revenue_recorded_kes,
    DATEDIFF('hour', TRY_TO_TIMESTAMP(opened_at), CURRENT_TIMESTAMP()) AS hours_since_open,
    confirmed_by                                                        AS last_supervisor_action
FROM {SHIFTS}
WHERE (closed_at IS NULL OR closed_at = '')
  AND TRY_TO_TIMESTAMP(opened_at) < DATEADD('hour', -12, CURRENT_TIMESTAMP())
  AND user_id NOT IN {EXCL_SYS}
ORDER BY hours_since_open DESC
"""


# ── Invoice analyses ────────────────────────────────────────────────────────────

SQL_INVOICE_SUMMARY = """
WITH deduped AS (
    SELECT
        id,
        MIN(TRY_TO_NUMBER(balance))  AS balance,
        MAX(TRY_TO_NUMBER(amount))   AS amount,
        MIN(FOR_CASH)                AS for_cash,
        MIN(CORPORATE_ID)            AS corporate_id
    FROM hospitals.xanalife_clean.finance_invoices
    WHERE TRY_TO_NUMBER(balance) > 0
      AND TRY_TO_TIMESTAMP(created_at) >= '2025-09-01'
      AND deleted_at IS NULL
    GROUP BY id
)
SELECT
    COUNT(*)                                                            AS unique_invoices,
    ROUND(SUM(balance), 0)                                              AS total_outstanding_kes,
    ROUND(SUM(CASE WHEN for_cash = 0 THEN balance ELSE 0 END), 0)      AS corporate_kes,
    COUNT(CASE WHEN for_cash = 0 THEN 1 END)                           AS corporate_count,
    ROUND(SUM(CASE WHEN for_cash = 1 THEN balance ELSE 0 END), 0)      AS cash_patient_kes,
    COUNT(CASE WHEN for_cash = 1 THEN 1 END)                           AS cash_patient_count
FROM deduped
"""

SQL_INVOICE_BY_CREDITOR = """
WITH deduped AS (
    SELECT
        id,
        MIN(TRY_TO_NUMBER(balance))              AS balance,
        MIN(FOR_CASH)                            AS for_cash,
        MIN(CORPORATE_ID)                        AS corporate_id,
        MIN(TRY_TO_TIMESTAMP(created_at))::DATE  AS invoice_date
    FROM hospitals.xanalife_clean.finance_invoices
    WHERE TRY_TO_NUMBER(balance) > 0
      AND TRY_TO_TIMESTAMP(created_at) >= '2025-09-01'
      AND deleted_at IS NULL
    GROUP BY id
)
SELECT
    CASE WHEN for_cash = 0 THEN COALESCE(CAST(corporate_id AS VARCHAR), 'Unknown')
         ELSE 'Cash Patients' END               AS creditor,
    for_cash,
    corporate_id,
    COUNT(*)                                    AS invoice_count,
    ROUND(SUM(balance), 0)                      AS outstanding_kes,
    MIN(invoice_date)                           AS oldest_invoice,
    MAX(invoice_date)                           AS latest_invoice
FROM deduped
GROUP BY 1, 2, 3
ORDER BY outstanding_kes DESC
"""

SQL_INVOICE_LIST = """
WITH deduped AS (
    SELECT
        id,
        MIN(invoice_no)                                 AS invoice_number,
        MIN(TRY_TO_TIMESTAMP(created_at))::DATE         AS created_date,
        MIN(store_code)                                 AS store,
        MIN(for_cash)                                   AS for_cash,
        MIN(corporate_id)                               AS corporate_id,
        MAX(TRY_TO_NUMBER(amount))                      AS amount_kes,
        MIN(TRY_TO_NUMBER(balance))                     AS outstanding_balance_kes
    FROM hospitals.xanalife_clean.finance_invoices
    WHERE TRY_TO_NUMBER(balance) > 0
      AND TRY_TO_TIMESTAMP(created_at) >= '2025-09-01'
      AND deleted_at IS NULL
    GROUP BY id
)
SELECT
    id                                                          AS invoice_id,
    invoice_number,
    created_date,
    CASE WHEN for_cash = 0 THEN 'Corporate / Insurance'
         ELSE 'Cash Patient' END                                AS payer_type,
    COALESCE(CAST(corporate_id AS VARCHAR), '—')               AS corporate_id,
    amount_kes,
    outstanding_balance_kes
FROM deduped
ORDER BY outstanding_balance_kes DESC
"""


# ── Registry ────────────────────────────────────────────────────────────────────

ANALYSES = [
    ("Analysis 1 — Pareto by Station",         SQL_PARETO),
    ("Analysis 2 — Monthly Variance Trend",    SQL_MONTHLY_TREND),
    ("Analysis 3 — Anomalous Shifts",          SQL_ANOMALOUS),
    ("Analysis 4 — Supervisor Correlation",    SQL_SUPERVISOR),
    ("Analysis 5 — Unclosed Shift Exposure",   SQL_UNCLOSED_EXPOSURE),
    ("Analysis 6 — Unclosed by Station",       SQL_UNCLOSED_BY_STATION),
    ("Analysis 7 — Unclosed Shift Trend",      SQL_UNCLOSED_TREND),
    ("Analysis 8 — Long-Duration Open Shifts", SQL_LONG_OPEN),
    ("Analysis 9 — Daily Compliance Report",   SQL_DAILY_COMPLIANCE),
    ("Invoice Summary",                        SQL_INVOICE_SUMMARY),
    ("Invoice By Creditor",                    SQL_INVOICE_BY_CREDITOR),
    ("Invoice List",                           SQL_INVOICE_LIST),
]


def run_all() -> dict:
    results = {}
    for label, sql in ANALYSES:
        print(f"\n{'='*60}\n  {label}\n{'='*60}")
        df = run_query_(sql)
        df.columns = df.columns.str.upper()
        print(df.to_string(index=False))
        results[label] = df
    print("\n\nAll 9 analyses complete.")
    return results


if __name__ == "__main__":
    import sys
    run_all()
