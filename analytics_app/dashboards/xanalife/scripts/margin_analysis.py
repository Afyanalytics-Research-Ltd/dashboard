"""
Margin Intelligence — MVaR (Margin Value-at-Risk) Analysis Module
"""

import pandas as pd
from connect_to_snowflake import run_query


def _store_clause(store_names, pos_col="s.store_product_id"):
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


def _by_store_clause(store_names):
    """WHERE clause for by-store queries using the deduped inventory_stores JOIN."""
    if not store_names:
        return ""
    names = ", ".join(f"'{n}'" for n in store_names)
    return f"AND st.name IN ({names})"


# ── SQL templates ──────────────────────────────────────────────────────────────
# Overall and Distribution: filter-only — sp join only (safe, no inventory_stores join).
# By Store: deduped inventory_stores join inline.

SQL_MVAR_OVERALL = """
WITH line_margins AS (
    SELECT
        s.sale_id,
        SUM(s.amount)                                              AS revenue,
        SUM(s.amount - COALESCE(p.unit_cost, 0) * s.quantity)     AS gross_profit
    FROM hospitals.xanalife_clean.evaluation_pos_sale_details s
    JOIN hospitals.xanalife_clean.inventory_store_products p
        ON s.store_product_id = p.id
    WHERE TRY_TO_TIMESTAMP(s.created_at) >= '2025-09-01'
      AND s.status != 'canceled'
      AND p.unit_cost > 0
      {store_filter}
    GROUP BY s.sale_id
),
with_margin AS (
    SELECT
        sale_id,
        revenue,
        gross_profit,
        ROUND(gross_profit / NULLIF(revenue, 0) * 100, 2) AS margin_pct
    FROM line_margins
    WHERE revenue > 0
)
SELECT
    COUNT(*)                                                                    AS total_transactions,
    ROUND(AVG(margin_pct), 1)                                                  AS avg_margin_pct,
    ROUND(PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY margin_pct), 1)        AS mvar_5pct,
    ROUND(PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY margin_pct), 1)        AS mvar_10pct,
    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY margin_pct), 1)        AS p25_margin_pct,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY margin_pct), 1)        AS median_margin_pct,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY margin_pct), 1)        AS p75_margin_pct,
    ROUND(MIN(margin_pct), 1)                                                  AS min_margin_pct,
    ROUND(MAX(margin_pct), 1)                                                  AS max_margin_pct,
    COUNT(CASE WHEN margin_pct < 0 THEN 1 END)                                AS loss_making_transactions,
    ROUND(COUNT(CASE WHEN margin_pct < 0 THEN 1 END) * 100.0 / COUNT(*), 1)  AS loss_making_pct
FROM with_margin
"""

# By-store: inventory_stores deduplicated with GROUP BY id to prevent fan-out.
SQL_MVAR_BY_STORE = """
WITH line_margins AS (
    SELECT
        s.sale_id,
        st.name                                                    AS store_name,
        st.code                                                    AS store_code,
        SUM(s.amount)                                              AS revenue,
        SUM(s.amount - COALESCE(p.unit_cost, 0) * s.quantity)     AS gross_profit
    FROM hospitals.xanalife_clean.evaluation_pos_sale_details s
    JOIN hospitals.xanalife_clean.inventory_store_products p
        ON s.store_product_id = p.id
    JOIN (
        SELECT id, MIN(name) AS name, MIN(code) AS code
        FROM hospitals.xanalife_clean.inventory_stores
        GROUP BY id
    ) st ON p.store_id = st.id
    WHERE TRY_TO_TIMESTAMP(s.created_at) >= '2025-09-01'
      AND s.status != 'canceled'
      AND p.unit_cost > 0
      {store_filter}
    GROUP BY s.sale_id, st.name, st.code
),
with_margin AS (
    SELECT
        sale_id, store_name, store_code, revenue, gross_profit,
        ROUND(gross_profit / NULLIF(revenue, 0) * 100, 2) AS margin_pct
    FROM line_margins
    WHERE revenue > 0
)
SELECT
    store_name,
    store_code,
    COUNT(*)                                                                    AS transactions,
    ROUND(AVG(margin_pct), 1)                                                  AS avg_margin_pct,
    ROUND(PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY margin_pct), 1)        AS mvar_5pct,
    ROUND(PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY margin_pct), 1)        AS mvar_10pct,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY margin_pct), 1)        AS median_margin_pct,
    COUNT(CASE WHEN margin_pct < 0 THEN 1 END)                                AS loss_making_count
FROM with_margin
GROUP BY store_name, store_code
ORDER BY avg_margin_pct DESC
"""

SQL_MVAR_DISTRIBUTION = """
WITH line_margins AS (
    SELECT
        s.sale_id,
        SUM(s.amount)                                              AS revenue,
        SUM(s.amount - COALESCE(p.unit_cost, 0) * s.quantity)     AS gross_profit
    FROM hospitals.xanalife_clean.evaluation_pos_sale_details s
    JOIN hospitals.xanalife_clean.inventory_store_products p
        ON s.store_product_id = p.id
    WHERE TRY_TO_TIMESTAMP(s.created_at) >= '2025-09-01'
      AND s.status != 'canceled'
      AND p.unit_cost > 0
      {store_filter}
    GROUP BY s.sale_id
),
with_margin AS (
    SELECT
        ROUND(gross_profit / NULLIF(revenue, 0) * 100, 0) AS margin_pct
    FROM line_margins
    WHERE revenue > 0
)
SELECT
    FLOOR(margin_pct / 5) * 5   AS bucket_start,
    COUNT(*)                     AS transaction_count
FROM with_margin
WHERE margin_pct BETWEEN -50 AND 150
GROUP BY 1
ORDER BY 1
"""


# ── Analyses registry ──────────────────────────────────────────────────────────

def get_analyses(store_names=None):
    sf  = _store_clause(store_names)       # IN subquery for filter-only queries
    bsf = _by_store_clause(store_names)    # AND st.name IN (...) for by-store
    return [
        ("MVaR — Overall",      SQL_MVAR_OVERALL.format(store_filter=sf)),
        ("MVaR — By Store",     SQL_MVAR_BY_STORE.format(store_filter=bsf)),
        ("MVaR — Distribution", SQL_MVAR_DISTRIBUTION.format(store_filter=sf)),
    ]

ANALYSES = get_analyses()


def run_all() -> dict:
    results = {}
    for label, sql in ANALYSES:
        print(f"Running: {label}…")
        results[label] = run_query(sql)
    return results


if __name__ == "__main__":
    data = run_all()
    for label, df in data.items():
        print(f"\n=== {label} ===")
        print(df.to_string(index=False))
