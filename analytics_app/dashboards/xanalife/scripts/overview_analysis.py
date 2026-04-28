"""
Overview — Executive Summary Module
Run standalone: python scripts/overview_analysis.py <passcode>
"""

import sys
import pandas as pd
from snowflake.snowflake_client import SnowflakeClient

connection = SnowflakeClient().conn




def run_query(sql: str, conn) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute(sql)
    df = cur.fetch_pandas_all()
    cur.close()
    return df


def _store_clause(store_names, pos_col="pos.store_product_id"):
    """IN-subquery filter — no fan-out. inventory_stores is deduplicated inside."""
    if not store_names:
        return ""
    names = ", ".join(f"'{n}'" for n in store_names)
    return f"""AND {pos_col} IN (
        SELECT DISTINCT sp2.id
        FROM hospitals.xanalife_clean.inventory_store_products sp2
        JOIN (
            SELECT id, MIN(name) AS name
            FROM hospitals.xanalife_clean.inventory_stores
            GROUP BY id
        ) st2 ON sp2.store_id = st2.id
        WHERE st2.name IN ({names})
    )"""


# ── SQL templates ──────────────────────────────────────────────────────────────
# Revenue + Stockout: filterable via IN subquery — no fan-out.
# Cash, Invoices, Loyalty: not filterable (no store linkage).

# Original structure preserved — no inventory_stores join in the main body.
# {store_filter} uses IN subquery so no extra rows are introduced.
SQL_REVENUE_SUMMARY = """
WITH max_month AS (
    SELECT DATE_TRUNC('month', MAX(TRY_TO_TIMESTAMP(pos.created_at))) AS latest_month
    FROM hospitals.xanalife_clean.evaluation_pos_sale_details pos
    WHERE TRY_TO_TIMESTAMP(pos.created_at) >= '2025-09-01'
      {store_filter}
)
SELECT
    ROUND(SUM(pos.amount), 0)                                                          AS total_revenue,
    COUNT(DISTINCT pos.sale_id)                                                        AS total_transactions,
    ROUND(SUM(CASE WHEN DATE_TRUNC('month', TRY_TO_TIMESTAMP(pos.created_at))
                        = (SELECT latest_month FROM max_month)
                   THEN pos.amount ELSE 0 END), 0)                                     AS revenue_this_month,
    ROUND(SUM(CASE WHEN DATE_TRUNC('month', TRY_TO_TIMESTAMP(pos.created_at))
                        = DATEADD('month', -1, (SELECT latest_month FROM max_month))
                   THEN pos.amount ELSE 0 END), 0)                                     AS revenue_last_month,
    (SELECT latest_month FROM max_month)                                               AS latest_data_month
FROM hospitals.xanalife_clean.evaluation_pos_sale_details pos
WHERE TRY_TO_TIMESTAMP(pos.created_at) >= '2025-09-01'
  {store_filter}
"""

SQL_REVENUE_TREND = """
SELECT
    DATE_TRUNC('day', TRY_TO_TIMESTAMP(pos.created_at)) AS sale_date,
    ROUND(SUM(pos.amount), 0)                            AS daily_revenue
FROM hospitals.xanalife_clean.evaluation_pos_sale_details pos
WHERE TRY_TO_TIMESTAMP(pos.created_at) >= DATEADD('day', -60, CURRENT_DATE())
  {store_filter}
GROUP BY 1
ORDER BY 1
"""

SQL_CASH_SUMMARY = """
SELECT
    ROUND(SUM(CASE WHEN closed_at IS NOT NULL AND closed_at != ''
                        AND TRY_TO_NUMBER(cashier_closing_balance) > 0
                   THEN TRY_TO_NUMBER(closing_variance) ELSE 0 END), 0)          AS net_variance,
    ROUND(SUM(CASE WHEN closed_at IS NOT NULL AND closed_at != ''
                        AND TRY_TO_NUMBER(cashier_closing_balance) > 0
                   THEN TRY_TO_NUMBER(system_closing_balance) ELSE 0 END), 0)     AS cash_at_risk,
    ROUND(SUM(CASE WHEN closed_at IS NOT NULL AND closed_at != ''
                        AND TRY_TO_NUMBER(cashier_closing_balance) > 0
                   THEN TRY_TO_NUMBER(closing_variance) ELSE 0 END) /
          NULLIF(SUM(CASE WHEN closed_at IS NOT NULL AND closed_at != ''
                               AND TRY_TO_NUMBER(cashier_closing_balance) > 0
                          THEN TRY_TO_NUMBER(system_closing_balance) ELSE 0 END), 0)
          * 100, 2)                                                                AS variance_pct,
    COUNT(CASE WHEN closed_at IS NULL OR closed_at = '' THEN 1 END)               AS unclosed_shifts,
    (SELECT COUNT(*)
     FROM hospitals.xanalife_clean.reception_reception_shifts
     WHERE TRY_TO_TIMESTAMP(created_at) >= '2025-09-01'
       AND closed_at IS NOT NULL AND closed_at != ''
       AND TRY_TO_NUMBER(cashier_closing_balance) > 0
       AND ABS(TRY_TO_NUMBER(closing_variance)) > ABS(TRY_TO_NUMBER(system_closing_balance))
       AND confirmed_by NOT IN ('sudo', 'Mrs. Xana  Admin', 'Mrs. Carole  Herzog')
    )                                                                              AS anomalous_shifts
FROM hospitals.xanalife_clean.reception_reception_shifts
WHERE TRY_TO_TIMESTAMP(created_at) >= '2025-09-01'
  AND user_id NOT IN (5, 1719, 1741, 1744, 1746, 1739)
  AND (confirmed_by IS NULL OR confirmed_by NOT IN ('sudo', 'Mrs. Xana  Admin', 'Mrs. Carole  Herzog'))
"""

# COUNT(DISTINCT s.product) handles any fan-out from the product_id join.
# Store filter uses IN subquery on product_id space.
SQL_STOCKOUT_COUNT = """
SELECT COUNT(DISTINCT s.product) AS products_at_zero_stock
FROM hospitals.xanalife_clean.inventory_inventory_stocks s
WHERE s.quantity <= 0
  AND s.product IN (
      SELECT DISTINCT sp2.product_id
      FROM hospitals.xanalife_clean.inventory_store_products sp2
      JOIN (
          SELECT id, MIN(name) AS name
          FROM hospitals.xanalife_clean.inventory_stores
          GROUP BY id
      ) st2 ON sp2.store_id = st2.id
      WHERE sp2.product_name IS NOT NULL
        {store_name_filter}
  )
"""

SQL_INVOICES_SUMMARY = """
SELECT
    COUNT(*)                                    AS total_invoices,
    ROUND(SUM(TRY_TO_NUMBER(amount)), 0)        AS total_amount,
    ROUND(SUM(TRY_TO_NUMBER(balance)), 0)       AS total_balance
FROM hospitals.xanalife_clean.finance_invoices
"""

SQL_LOYALTY_SUMMARY = """
SELECT
    SUM(CASE WHEN LOWER(type) = 'earned'   THEN points ELSE 0 END) AS total_earned,
    COUNT(CASE WHEN LOWER(type) = 'redeemed' THEN 1 END)           AS redemption_count,
    SUM(CASE WHEN LOWER(type) = 'redeemed' THEN points ELSE 0 END) AS total_redeemed,
    COUNT(DISTINCT customer_id)                                     AS customers_with_points
FROM hospitals.xanalife_clean.points
WHERE TRY_TO_TIMESTAMP(created_at) >= '2025-09-01'
"""


# ── Analyses registry ──────────────────────────────────────────────────────────

def get_analyses(store_names=None):
    sf = _store_clause(store_names, pos_col="pos.store_product_id")

    # Stockout uses product_id space — different placeholder name
    if store_names:
        names = ", ".join(f"'{n}'" for n in store_names)
        stockout_filter = f"AND st2.name IN ({names})"
    else:
        stockout_filter = ""

    return [
        ("Revenue Summary",  SQL_REVENUE_SUMMARY.format(store_filter=sf)),
        ("Revenue Trend",    SQL_REVENUE_TREND.format(store_filter=sf)),
        ("Cash Summary",     SQL_CASH_SUMMARY),
        ("Stockout Count",   SQL_STOCKOUT_COUNT.format(store_name_filter=stockout_filter)),
        ("Invoices Summary", SQL_INVOICES_SUMMARY),
        ("Loyalty Summary",  SQL_LOYALTY_SUMMARY),
    ]

ANALYSES = get_analyses()


def run_all(passcode: str) -> dict:
    conn = connection
    results = {}
    for label, sql in ANALYSES:
        print(f"Running: {label}…")
        results[label] = run_query(sql, conn)
    conn.close()
    return results


if __name__ == "__main__":
    passcode = sys.argv[1] if len(sys.argv) > 1 else input("TOTP passcode: ")
    data = run_all(passcode)
    for label, df in data.items():
        print(f"\n=== {label} ===")
        print(df.to_string(index=False))
