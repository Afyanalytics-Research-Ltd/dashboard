"""
Substitution Capture Rate (SCR) — Analysis Module
Run standalone: python scripts/scr_analysis.py <passcode>
"""

import sys
import pandas as pd
from connection import connect


def run_query(sql: str, conn) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute(sql)
    df = cur.fetch_pandas_all()
    cur.close()
    return df


def _store_clause(store_names, dis_col="d.store_product_id"):
    """IN-subquery filter — no fan-out. inventory_stores is deduplicated inside.
    Default (no store_names): supermarket only — pharmacy codes excluded."""
    if not store_names:
        return f"""AND {dis_col} IN (
        SELECT DISTINCT sp2.id
        FROM hospitals.xanalife_clean.inventory_store_products sp2
        JOIN (
            SELECT id, MIN(code) AS code
            FROM hospitals.xanalife_clean.inventory_stores
            GROUP BY id
        ) st2 ON sp2.store_id = st2.id
        WHERE st2.code NOT IN ('PHARMACY', 'KP')
    )"""
    names = ", ".join(f"'{n}'" for n in store_names)
    return f"""AND {dis_col} IN (
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
# All SCR queries use dispensing (d.store_product_id) as the fact join point.
# Store filter uses IN subquery — no direct inventory_stores join in the weekly CTE.
# Previous hardcoded pharmacy exclusion removed; Division filter drives store selection.

SQL_SCR_SUMMARY = """
WITH weekly AS (
    SELECT
        d.product,
        sp.product_name,
        sp.product_category,
        DATE_TRUNC('week', TRY_TO_TIMESTAMP(d.created_at)) AS week,
        SUM(d.quantity) AS qty
    FROM hospitals.xanalife_clean.inventory_inventory_dispensing d
    JOIN hospitals.xanalife_clean.inventory_store_products sp
        ON d.store_product_id = sp.id
    WHERE TRY_TO_TIMESTAMP(d.created_at) >= '2025-09-01'
      AND sp.product_name IS NOT NULL
      {store_filter}
    GROUP BY 1, 2, 3, 4
),
gaps AS (
    SELECT
        product, product_name, product_category, week, qty,
        LAG(week) OVER (PARTITION BY product ORDER BY week) AS prev_week,
        DATEDIFF('day',
            LAG(week) OVER (PARTITION BY product ORDER BY week),
            week) AS gap_days
    FROM weekly
),
stockouts AS (
    SELECT
        product,
        product_name,
        product_category,
        MAX(gap_days)                                        AS longest_gap_days,
        COUNT(*)                                             AS active_weeks,
        ROUND(AVG(qty), 1)                                   AS avg_weekly_qty,
        COUNT(CASE WHEN gap_days >= 14 THEN 1 END)           AS gap_occurrences,
        MIN(CASE WHEN gap_days >= 14 THEN prev_week END)     AS gap_start,
        MIN(CASE WHEN gap_days >= 14 THEN week END)          AS gap_end
    FROM gaps
    WHERE gap_days IS NOT NULL
    GROUP BY product, product_name, product_category
    HAVING longest_gap_days >= 14
       AND active_weeks >= 8
       AND avg_weekly_qty >= 3
),
same_cat AS (
    SELECT DISTINCT
        s.product       AS focal_product,
        sp.product_id   AS sub_product_id,
        sp.product_name AS sub_name
    FROM stockouts s
    JOIN hospitals.xanalife_clean.inventory_store_products sp
        ON  sp.product_category = s.product_category
        AND sp.product_id != s.product
        AND sp.product_name IS NOT NULL
),
sub_dispensing AS (
    SELECT
        sc.focal_product,
        sc.sub_product_id,
        sc.sub_name,
        SUM(CASE WHEN TRY_TO_TIMESTAMP(d.created_at)
                      BETWEEN DATEADD('week', -4, s.gap_start) AND s.gap_start
                 THEN d.quantity ELSE 0 END) AS baseline_qty,
        SUM(CASE WHEN TRY_TO_TIMESTAMP(d.created_at)
                      BETWEEN s.gap_start AND s.gap_end
                 THEN d.quantity ELSE 0 END) AS gap_qty
    FROM same_cat sc
    JOIN stockouts s   ON s.product = sc.focal_product
    JOIN hospitals.xanalife_clean.inventory_inventory_dispensing d
        ON d.product = sc.sub_product_id
    GROUP BY sc.focal_product, sc.sub_product_id, sc.sub_name
),
top_substitute AS (
    SELECT *,
        gap_qty - baseline_qty AS uplift,
        ROW_NUMBER() OVER (PARTITION BY focal_product ORDER BY gap_qty - baseline_qty DESC) AS rn
    FROM sub_dispensing
    WHERE gap_qty > 0
),
focal_price AS (
    SELECT product_id, MAX(selling_price) AS selling_price
    FROM hospitals.xanalife_clean.inventory_store_products
    WHERE selling_price > 0
    GROUP BY product_id
)
SELECT
    s.product_name                                                              AS product_name,
    s.product_category,
    ROUND(s.avg_weekly_qty, 0)                                                  AS avg_weekly_units,
    s.longest_gap_days                                                          AS gap_days,
    s.gap_occurrences,
    s.gap_start,
    s.gap_end,
    COALESCE(ts.sub_name, '—')                                                  AS top_substitute,
    COALESCE(ts.uplift, 0)                                                      AS substitute_uplift,
    ROUND(COALESCE(fp.selling_price, 0) * s.avg_weekly_qty
          * s.longest_gap_days / 7.0, 0)                                        AS revenue_at_risk_kes,
    ROUND(COALESCE(ts.uplift, 0) * 1.2, 0)                                      AS recommended_prestock_qty,
    CASE
        WHEN COALESCE(ts.uplift, 0) > 0 THEN 'Pre-stock ' || ts.sub_name
        ELSE 'Monitor — no substitute absorbed demand'
    END                                                                         AS action
FROM stockouts s
LEFT JOIN top_substitute ts  ON ts.focal_product = s.product AND ts.rn = 1
LEFT JOIN focal_price fp     ON fp.product_id    = s.product
ORDER BY revenue_at_risk_kes DESC
"""

SQL_CATEGORY_EXPOSURE = """
WITH weekly AS (
    SELECT
        d.product,
        sp.product_category,
        DATE_TRUNC('week', TRY_TO_TIMESTAMP(d.created_at)) AS week,
        SUM(d.quantity) AS qty
    FROM hospitals.xanalife_clean.inventory_inventory_dispensing d
    JOIN hospitals.xanalife_clean.inventory_store_products sp
        ON d.store_product_id = sp.id
    WHERE TRY_TO_TIMESTAMP(d.created_at) >= '2025-09-01'
      AND sp.product_name IS NOT NULL
      {store_filter}
    GROUP BY 1, 2, 3
),
gaps AS (
    SELECT
        product, product_category, week, qty,
        DATEDIFF('day',
            LAG(week) OVER (PARTITION BY product ORDER BY week),
            week) AS gap_days
    FROM weekly
),
stockouts AS (
    SELECT product, product_category
    FROM gaps
    WHERE gap_days IS NOT NULL
    GROUP BY product, product_category
    HAVING MAX(gap_days) >= 14
       AND COUNT(*) >= 8
       AND AVG(qty) >= 3
)
SELECT
    COALESCE(cat.name, CAST(s.product_category AS VARCHAR)) AS name,
    COUNT(DISTINCT s.product)                               AS stockout_products
FROM stockouts s
LEFT JOIN hospitals.xanalife_clean.inventory_inventory_categories cat
    ON cat.id = s.product_category
GROUP BY 1
ORDER BY 2 DESC
LIMIT 10
"""

SQL_DEPLETION_RISK = """
WITH velocity AS (
    SELECT
        d.product,
        sp.product_name,
        sp.product_category,
        MAX(sp.selling_price)               AS selling_price,
        ROUND(SUM(d.quantity) / 28.0, 2)    AS avg_daily_rate
    FROM hospitals.xanalife_clean.inventory_inventory_dispensing d
    JOIN hospitals.xanalife_clean.inventory_store_products sp
        ON d.store_product_id = sp.id
    WHERE TRY_TO_TIMESTAMP(d.created_at) >= DATEADD('day', -28, CURRENT_DATE())
      AND sp.product_name IS NOT NULL
      {store_filter}
    GROUP BY 1, 2, 3
),
stock AS (
    SELECT product, SUM(quantity) AS current_qty
    FROM hospitals.xanalife_clean.inventory_inventory_stocks
    GROUP BY product
)
SELECT
    v.product_name,
    v.product_category,
    ROUND(s.current_qty, 0)                                                     AS current_stock,
    v.avg_daily_rate,
    ROUND(s.current_qty / v.avg_daily_rate, 0)                                  AS days_until_stockout,
    CASE
        WHEN s.current_qty / v.avg_daily_rate <= 7  THEN 'Critical'
        WHEN s.current_qty / v.avg_daily_rate <= 14 THEN 'Warning'
        ELSE                                              'Watch'
    END                                                                         AS urgency
FROM velocity v
JOIN stock s ON s.product = v.product
WHERE v.avg_daily_rate > 0
  AND s.current_qty IS NOT NULL
  AND s.current_qty / v.avg_daily_rate <= 30
ORDER BY days_until_stockout ASC
LIMIT 25
"""

SQL_NULL_PRODUCTS_INVESTIGATION = """
SELECT
    sp.id              AS store_product_id,
    sp.product_id,
    sp.product_name,
    sp.product_category,
    sp.product_department,
    COUNT(pos.id)      AS pos_records,
    SUM(pos.quantity)  AS total_qty_sold
FROM hospitals.xanalife_clean.evaluation_pos_sale_details pos
JOIN hospitals.xanalife_clean.inventory_store_products sp
    ON pos.store_product_id = sp.id
WHERE sp.product_name IS NULL
  AND TRY_TO_TIMESTAMP(pos.created_at) >= '2025-09-01'
GROUP BY 1, 2, 3, 4, 5
ORDER BY pos_records DESC
LIMIT 50
"""


# ── Analyses registry ──────────────────────────────────────────────────────────

def get_analyses(store_names=None):
    sf = _store_clause(store_names, dis_col="d.store_product_id")
    return [
        ("SCR Summary — Stockouts + Top Substitute",  SQL_SCR_SUMMARY.format(store_filter=sf)),
        ("Category Exposure — Stockouts by Category", SQL_CATEGORY_EXPOSURE.format(store_filter=sf)),
        ("Depletion Risk — Days Until Stockout",      SQL_DEPLETION_RISK.format(store_filter=sf)),
    ]

ANALYSES = get_analyses()


def run_all(passcode: str) -> dict:
    conn = connect(passcode)
    results = {}
    for label, sql in ANALYSES:
        print(f"Running: {label}…")
        results[label] = run_query(sql, conn)
    conn.close()
    return results


if __name__ == "__main__":
    passcode = sys.argv[1] if len(sys.argv) > 1 else input("TOTP passcode: ")

    if len(sys.argv) > 2 and sys.argv[2] == "nulls":
        conn = connect(passcode)
        print("\n=== NULL product_name investigation ===")
        print(run_query(SQL_NULL_PRODUCTS_INVESTIGATION, conn).to_string(index=False))
        conn.close()
    else:
        data = run_all(passcode)
        for label, df in data.items():
            print(f"\n=== {label} ===")
            print(df.to_string(index=False))
