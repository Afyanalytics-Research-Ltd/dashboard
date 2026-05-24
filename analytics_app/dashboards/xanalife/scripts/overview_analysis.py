"""
Overview — Executive Summary Module
"""

import pandas as pd
from connect_to_snowflake import run_query_


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
# Revenue + Operating Margin: filterable via IN subquery — no fan-out.
# Invoices, Loyalty: not filterable (no store linkage).

SQL_REVENUE_SUMMARY = """
WITH max_month AS (
    SELECT DATE_TRUNC('month', MAX(TRY_TO_TIMESTAMP(pos.created_at))) AS latest_month
    FROM hospitals.xanalife_clean.evaluation_pos_sale_details pos
    WHERE TRY_TO_TIMESTAMP(pos.created_at) >= {start_date}
      AND TRY_TO_TIMESTAMP(pos.created_at) <= {end_date}
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
WHERE TRY_TO_TIMESTAMP(pos.created_at) >= {start_date}
  AND TRY_TO_TIMESTAMP(pos.created_at) <= {end_date}
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

SQL_OPERATING_MARGIN = """
WITH line_margins AS (
    SELECT
        pos.sale_id,
        SUM(pos.amount)                                                 AS revenue,
        SUM(pos.amount - COALESCE(sp.unit_cost, 0) * pos.quantity)     AS gross_profit
    FROM hospitals.xanalife_clean.evaluation_pos_sale_details pos
    JOIN hospitals.xanalife_clean.inventory_store_products sp
        ON pos.store_product_id = sp.id
    WHERE TRY_TO_TIMESTAMP(pos.created_at) >= {start_date}
      AND TRY_TO_TIMESTAMP(pos.created_at) <= {end_date}
      AND pos.status != 'canceled'
      AND sp.unit_cost > 0
      {store_filter}
    GROUP BY pos.sale_id
)
SELECT
    ROUND(AVG(gross_profit / NULLIF(revenue, 0) * 100), 1)             AS avg_sale_margin_pct,
    ROUND(SUM(gross_profit) / NULLIF(SUM(revenue), 0) * 100, 1)        AS portfolio_margin_pct
FROM line_margins
WHERE revenue > 0
"""

# COUNT(DISTINCT s.product) handles any fan-out from the product_id join.
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
WITH deduped AS (
    SELECT
        id,
        MIN(TRY_TO_NUMBER(amount))  AS amount,
        MIN(TRY_TO_NUMBER(balance)) AS balance
    FROM hospitals.xanalife_clean.finance_invoices
    WHERE TRY_TO_TIMESTAMP(created_at) >= {start_date}
      AND TRY_TO_TIMESTAMP(created_at) <= {end_date}
      AND deleted_at IS NULL
      AND TRY_TO_NUMBER(balance) > 0
    GROUP BY id
)
SELECT
    COUNT(*)                AS total_invoices,
    ROUND(SUM(amount), 0)   AS total_amount,
    ROUND(SUM(balance), 0)  AS total_balance
FROM deduped
"""

# Sep/Oct 2025 contain bulk-imported historical points (not organic activity).
# loyalty_start is floored to 2025-11-01 inside get_analyses() — passed as {loyalty_start}.
SQL_LOYALTY_SUMMARY = """
SELECT
    SUM(CASE WHEN LOWER(type) = 'earned'   THEN points ELSE 0 END) AS total_earned,
    COUNT(CASE WHEN LOWER(type) = 'redeemed' THEN 1 END)           AS redemption_count,
    SUM(CASE WHEN LOWER(type) = 'redeemed' THEN points ELSE 0 END) AS total_redeemed,
    COUNT(DISTINCT customer_id)                                     AS customers_with_points
FROM hospitals.xanalife_clean.points
WHERE TRY_TO_TIMESTAMP(created_at) >= {loyalty_start}
  AND TRY_TO_TIMESTAMP(created_at) <= {end_date}
  AND type IS NOT NULL AND TRIM(type) != ''
"""


# ── Analyses registry ──────────────────────────────────────────────────────────

def get_analyses(store_names=None, start_date='2025-09-01', end_date='2026-03-31'):
    sf = _store_clause(store_names, pos_col="pos.store_product_id")
    sd = f"'{start_date}'"
    ed = f"'{end_date}'"
    # Sep/Oct 2025 are bulk historical imports — loyalty always starts Nov 2025 minimum
    loyalty_raw   = '2025-11-01' if start_date < '2025-11-01' else start_date
    loyalty_start = f"'{loyalty_raw}'"

    if store_names:
        names = ", ".join(f"'{n}'" for n in store_names)
        stockout_filter = f"AND st2.name IN ({names})"
    else:
        stockout_filter = ""

    return [
        ("Revenue Summary",   SQL_REVENUE_SUMMARY.format(store_filter=sf, start_date=sd, end_date=ed)),
        ("Revenue Trend",     SQL_REVENUE_TREND.format(store_filter=sf)),
        ("Stockout Count",    SQL_STOCKOUT_COUNT.format(store_name_filter=stockout_filter)),
        ("Invoices Summary",  SQL_INVOICES_SUMMARY.format(start_date=sd, end_date=ed)),
        ("Loyalty Summary",   SQL_LOYALTY_SUMMARY.format(loyalty_start=loyalty_start, end_date=ed)),
        ("Operating Margin",  SQL_OPERATING_MARGIN.format(store_filter=sf, start_date=sd, end_date=ed)),
    ]

ANALYSES = get_analyses()


def run_all() -> dict:
    results = {}
    for label, sql in ANALYSES:
        print(f"Running: {label}…")
        results[label] = run_query_(sql)
    return results


if __name__ == "__main__":
    data = run_all()
    for label, df in data.items():
        print(f"\n=== {label} ===")
        print(df.to_string(index=False))
