"""
Revenue Intelligence — Sales Analysis Module
Run standalone: python scripts/sales_analysis.py <passcode>
"""

import sys
from datetime import timedelta
import pandas as pd
import numpy as np
from snowflake.snowflake_client import SnowflakeClient
connection =  SnowflakeClient().conn


def run_query(sql: str, conn) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute(sql)
    df = cur.fetch_pandas_all()
    cur.close()
    return df


# ── Store filter helpers ───────────────────────────────────────────────────────

def _store_clause(store_names, pos_col="pos.store_product_id"):
    """
    IN-subquery filter for total-aggregate queries.
    Avoids fan-out — inventory_stores is deduplicated inside the subquery.
    Returns empty string when no filter is needed (all stores).
    """
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


def _by_store_clause(store_names):
    """
    WHERE clause for by-store queries.
    Must be used alongside the deduplicated inventory_stores JOIN already in the SQL.
    """
    if not store_names:
        return ""
    names = ", ".join(f"'{n}'" for n in store_names)
    return f"AND st.name IN ({names})"


# ── SQL templates ──────────────────────────────────────────────────────────────
# Filter-only queries use {store_filter} via _store_clause() — no inventory_stores join in main body.
# By-store queries use {store_filter} via _by_store_clause() — inventory_stores is deduped inline.

SQL_DAILY_REVENUE = """
SELECT
    DATE_TRUNC('day', TRY_TO_TIMESTAMP(pos.created_at)) AS sale_date,
    COUNT(DISTINCT pos.sale_id)                          AS transactions,
    SUM(pos.amount)                                      AS daily_revenue
FROM hospitals.xanalife_clean.evaluation_pos_sale_details pos
WHERE TRY_TO_TIMESTAMP(pos.created_at) >= '2025-09-01'
  {store_filter}
GROUP BY 1
ORDER BY 1
"""

SQL_BASKET_SUMMARY = """
WITH basket AS (
    SELECT
        pos.sale_id,
        COUNT(*)        AS items,
        SUM(pos.amount) AS basket_value
    FROM hospitals.xanalife_clean.evaluation_pos_sale_details pos
    WHERE TRY_TO_TIMESTAMP(pos.created_at) >= '2025-09-01'
      {store_filter}
    GROUP BY pos.sale_id
)
SELECT
    COUNT(*)                                                               AS total_transactions,
    ROUND(AVG(items), 1)                                                   AS avg_items_per_sale,
    ROUND(AVG(basket_value), 0)                                            AS avg_basket_value_kes,
    COUNT(CASE WHEN items = 1 THEN 1 END)                                  AS single_item_count,
    ROUND(COUNT(CASE WHEN items = 1 THEN 1 END) * 100.0 / COUNT(*), 1)    AS single_item_pct,
    COUNT(CASE WHEN items >= 3 THEN 1 END)                                 AS multi_item_count,
    ROUND(COUNT(CASE WHEN items >= 3 THEN 1 END) * 100.0 / COUNT(*), 1)   AS multi_item_pct
FROM basket
"""

# sp join is safe here (inventory_store_products.id is unique — no fan-out risk).
# inventory_stores is NOT joined; store filtering uses the IN subquery.
SQL_BASKET_TOP_SINGLE_PRODUCTS = """
WITH basket AS (
    SELECT pos.sale_id, COUNT(*) AS items
    FROM hospitals.xanalife_clean.evaluation_pos_sale_details pos
    WHERE TRY_TO_TIMESTAMP(pos.created_at) >= '2025-09-01'
      {store_filter}
    GROUP BY pos.sale_id
)
SELECT
    sp.product_name,
    COUNT(*)                                                               AS times_bought_alone,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1)                    AS pct_of_single_item_sales
FROM hospitals.xanalife_clean.evaluation_pos_sale_details pos
JOIN hospitals.xanalife_clean.inventory_store_products sp
    ON pos.store_product_id = sp.id
JOIN basket b
    ON b.sale_id = pos.sale_id AND b.items = 1
WHERE sp.product_name IS NOT NULL
  {store_filter}
GROUP BY sp.product_name
ORDER BY times_bought_alone DESC
LIMIT 15
"""

# By-store: inventory_stores deduplicated with GROUP BY id to prevent fan-out.
SQL_BASKET_BY_STORE = """
WITH basket AS (
    SELECT
        pos.sale_id,
        st.name         AS store_name,
        COUNT(*)        AS items,
        SUM(pos.amount) AS basket_value
    FROM hospitals.xanalife_clean.evaluation_pos_sale_details pos
    JOIN hospitals.xanalife_clean.inventory_store_products sp
        ON pos.store_product_id = sp.id
    JOIN (
        SELECT id, MIN(name) AS name
        FROM hospitals.xanalife_clean.inventory_stores
        GROUP BY id
    ) st ON sp.store_id = st.id
    WHERE TRY_TO_TIMESTAMP(pos.created_at) >= '2025-09-01'
      {store_filter}
    GROUP BY pos.sale_id, st.name
)
SELECT
    store_name,
    COUNT(*)                                                                AS transactions,
    ROUND(AVG(items), 1)                                                    AS avg_items,
    ROUND(AVG(basket_value), 0)                                             AS avg_basket_kes,
    ROUND(COUNT(CASE WHEN items = 1 THEN 1 END) * 100.0 / COUNT(*), 1)     AS single_item_pct
FROM basket
GROUP BY store_name
ORDER BY avg_basket_kes DESC
"""

SQL_PEAK_HEATMAP = """
SELECT
    CASE DAYOFWEEK(TRY_TO_TIMESTAMP(pos.created_at))
        WHEN 0 THEN 'Sunday'
        WHEN 1 THEN 'Monday'
        WHEN 2 THEN 'Tuesday'
        WHEN 3 THEN 'Wednesday'
        WHEN 4 THEN 'Thursday'
        WHEN 5 THEN 'Friday'
        WHEN 6 THEN 'Saturday'
    END                                              AS day_name,
    DAYOFWEEK(TRY_TO_TIMESTAMP(pos.created_at))      AS day_num,
    HOUR(TRY_TO_TIMESTAMP(pos.created_at))           AS hour_of_day,
    COUNT(DISTINCT pos.sale_id)                      AS transactions,
    ROUND(SUM(pos.amount), 0)                        AS revenue
FROM hospitals.xanalife_clean.evaluation_pos_sale_details pos
WHERE TRY_TO_TIMESTAMP(pos.created_at) >= '2025-09-01'
  {store_filter}
GROUP BY 1, 2, 3
ORDER BY day_num, hour_of_day
"""


# sp_store CTE deduplicates inventory_store_products at sp.id level — guarantees 1 row per
# store_product_id regardless of any duplicates, preventing fan-out on the pos join.
SQL_REVENUE_BY_STORE = """
WITH sp_store AS (
    SELECT
        sp.id          AS store_product_id,
        MIN(st.name)   AS store_name,
        MIN(st.code)   AS store_code
    FROM hospitals.xanalife_clean.inventory_store_products sp
    JOIN (
        SELECT id, MIN(name) AS name, MIN(code) AS code
        FROM hospitals.xanalife_clean.inventory_stores
        GROUP BY id
    ) st ON sp.store_id = st.id
    GROUP BY sp.id
)
SELECT
    ss.store_name,
    ss.store_code,
    ROUND(SUM(pos.amount), 0)   AS revenue,
    COUNT(DISTINCT pos.sale_id) AS transactions
FROM hospitals.xanalife_clean.evaluation_pos_sale_details pos
JOIN sp_store ss ON pos.store_product_id = ss.store_product_id
WHERE TRY_TO_TIMESTAMP(pos.created_at) >= '2025-09-01'
  {store_filter}
GROUP BY ss.store_name, ss.store_code
ORDER BY revenue DESC
"""

# Location derived from store code — Katani codes: KP (pharmacy), KW (wholesale). All others = Syokimau.
# sp_store CTE deduplicates at sp.id level before joining to POS.
SQL_TOP_PRODUCTS_BY_LOCATION = """
WITH sp_store AS (
    SELECT
        sp.id                    AS store_product_id,
        MIN(sp.product_name)     AS product_name,
        MIN(sp.product_category) AS product_category,
        CASE WHEN MIN(st.code) IN ('KP', 'KW') THEN 'Katani' ELSE 'Syokimau' END AS location
    FROM hospitals.xanalife_clean.inventory_store_products sp
    JOIN (
        SELECT id, MIN(code) AS code
        FROM hospitals.xanalife_clean.inventory_stores
        GROUP BY id
    ) st ON sp.store_id = st.id
    GROUP BY sp.id
),
ranked AS (
    SELECT
        ss.location,
        ss.product_name,
        ss.product_category,
        ROUND(SUM(pos.amount), 0)   AS revenue,
        COUNT(DISTINCT pos.sale_id) AS transactions,
        ROW_NUMBER() OVER (
            PARTITION BY ss.location
            ORDER BY SUM(pos.amount) DESC
        ) AS rn
    FROM hospitals.xanalife_clean.evaluation_pos_sale_details pos
    JOIN sp_store ss ON pos.store_product_id = ss.store_product_id
    WHERE TRY_TO_TIMESTAMP(pos.created_at) >= '2025-09-01'
      AND ss.product_name IS NOT NULL
      {store_filter}
    GROUP BY ss.location, ss.product_name, ss.product_category
)
SELECT location, product_name, product_category, revenue, transactions, rn AS rank
FROM ranked
WHERE rn <= 15
ORDER BY location, rn
"""

# Cross-location opportunity: products with significant revenue gap between locations.
# Positive gap = Syokimau stronger (potential to grow at Katani).
# Negative gap = Katani stronger (potential to grow at Syokimau).
# sp_store CTE deduplicates at sp.id level before joining to POS.
SQL_LOCATION_OPPORTUNITY = """
WITH sp_store AS (
    SELECT
        sp.id                    AS store_product_id,
        MIN(sp.product_name)     AS product_name,
        MIN(sp.product_category) AS product_category,
        CASE WHEN MIN(st.code) IN ('KP', 'KW') THEN 'Katani' ELSE 'Syokimau' END AS location
    FROM hospitals.xanalife_clean.inventory_store_products sp
    JOIN (
        SELECT id, MIN(code) AS code
        FROM hospitals.xanalife_clean.inventory_stores
        GROUP BY id
    ) st ON sp.store_id = st.id
    GROUP BY sp.id
),
loc_revenue AS (
    SELECT
        ss.location,
        ss.product_name,
        ss.product_category,
        ROUND(SUM(pos.amount), 0) AS revenue
    FROM hospitals.xanalife_clean.evaluation_pos_sale_details pos
    JOIN sp_store ss ON pos.store_product_id = ss.store_product_id
    WHERE TRY_TO_TIMESTAMP(pos.created_at) >= '2025-09-01'
      AND ss.product_name IS NOT NULL
      {store_filter}
    GROUP BY ss.location, ss.product_name, ss.product_category
),
pivot AS (
    SELECT
        product_name,
        MIN(product_category)                                              AS product_category,
        SUM(CASE WHEN location = 'Syokimau' THEN revenue ELSE 0 END)     AS syokimau_revenue,
        SUM(CASE WHEN location = 'Katani'   THEN revenue ELSE 0 END)     AS katani_revenue
    FROM loc_revenue
    GROUP BY product_name
)
SELECT
    product_name,
    product_category,
    syokimau_revenue,
    katani_revenue,
    syokimau_revenue - katani_revenue AS gap
FROM pivot
WHERE (syokimau_revenue > 10000 OR katani_revenue > 10000)
  AND ABS(syokimau_revenue - katani_revenue) > 5000
ORDER BY ABS(gap) DESC
LIMIT 25
"""


# ── Analyses registry ──────────────────────────────────────────────────────────

def get_analyses(store_names=None):
    sf  = _store_clause(store_names)       # IN subquery — for total-aggregate queries
    bsf = _by_store_clause(store_names)    # AND st.name IN (...) — for by-store queries
    return [
        ("Daily Revenue",              SQL_DAILY_REVENUE.format(store_filter=sf)),
        ("Basket Summary",             SQL_BASKET_SUMMARY.format(store_filter=sf)),
        ("Top Single-Item Products",   SQL_BASKET_TOP_SINGLE_PRODUCTS.format(store_filter=sf)),
        ("Basket by Store",            SQL_BASKET_BY_STORE.format(store_filter=bsf)),
        ("Peak Revenue Heatmap",       SQL_PEAK_HEATMAP.format(store_filter=sf)),
        ("Revenue by Store",           SQL_REVENUE_BY_STORE.format(store_filter=sf)),
        ("Top Products by Location",   SQL_TOP_PRODUCTS_BY_LOCATION.format(store_filter=sf)),
        ("Location Opportunity",       SQL_LOCATION_OPPORTUNITY.format(store_filter=sf)),
    ]

ANALYSES = get_analyses()


# ── Forecast ───────────────────────────────────────────────────────────────────

KE_PUBLIC_HOLIDAYS = pd.to_datetime([
    "2025-10-20",  # Mashujaa Day
    "2025-12-12",  # Jamhuri Day
    "2025-12-25",  # Christmas
    "2025-12-26",  # Boxing Day
    "2026-01-01",  # New Year
    "2026-04-03",  # Good Friday
    "2026-04-06",  # Easter Monday
    "2026-05-01",  # Labour Day
])


def build_forecast(daily_df: pd.DataFrame, days_forward: int = 30):
    df = daily_df.copy()
    df["SALE_DATE"]     = pd.to_datetime(df["SALE_DATE"])
    df["DAILY_REVENUE"] = pd.to_numeric(df["DAILY_REVENUE"], errors="coerce").fillna(0)

    df = df[df["SALE_DATE"] >= "2025-09-14"].sort_values("SALE_DATE").reset_index(drop=True)

    full_range = pd.date_range(df["SALE_DATE"].min(), df["SALE_DATE"].max(), freq="D")
    df = (
        df.set_index("SALE_DATE")
          .reindex(full_range, fill_value=0)
          .reset_index()
          .rename(columns={"index": "SALE_DATE"})
    )

    df["ROLLING_7"] = df["DAILY_REVENUE"].rolling(7, min_periods=1).mean()

    x      = np.arange(len(df))
    coeffs = np.polyfit(x, df["ROLLING_7"].values, 1)
    df["TREND"] = np.polyval(coeffs, x)

    future_x     = np.arange(len(df), len(df) + days_forward)
    future_dates = pd.date_range(df["SALE_DATE"].max() + timedelta(days=1), periods=days_forward)
    forecast_df  = pd.DataFrame({
        "SALE_DATE": future_dates,
        "FORECAST":  np.polyval(coeffs, future_x).clip(min=0),
    })

    return df, forecast_df


# ── CLI ────────────────────────────────────────────────────────────────────────

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
