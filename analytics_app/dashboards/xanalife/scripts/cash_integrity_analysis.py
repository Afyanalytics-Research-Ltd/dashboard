"""
Cash Integrity & Forensics — all 9 analyses.

Local run:  python scripts/cash_integrity_analysis.py
Streamlit:  streamlit run scripts/streamlit_cash_integrity.py

To move to Streamlit in Snowflake, replace run_query() with:
    from snowflake.snowpark.context import get_active_session
    return get_active_session().sql(sql).to_pandas()
"""

import pandas as pd
from connection import connect


def run_query(sql: str, conn) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute(sql)
    df = cur.fetch_pandas_all()
    cur.close()
    return df


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
]


def run_all(passcode: str) -> dict:
    conn = connect(passcode)
    results = {}
    for label, sql in ANALYSES:
        print(f"\n{'='*60}\n  {label}\n{'='*60}")
        df = run_query(sql, conn)
        print(df.to_string(index=False))
        results[label] = df
    conn.close()
    print("\n\nAll 9 analyses complete.")
    return results


if __name__ == "__main__":
    import sys
    passcode = sys.argv[1] if len(sys.argv) > 1 else input("Enter TOTP passcode: ")
    run_all(passcode)
