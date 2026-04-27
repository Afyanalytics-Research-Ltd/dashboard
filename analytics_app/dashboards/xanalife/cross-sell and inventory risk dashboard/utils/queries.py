CSUC_QUERY = """
WITH total_baskets AS (
    SELECT COUNT(DISTINCT SALE_ID) AS n
    FROM evaluation_pos_sale_details
    WHERE STATUS = 'Paid'
      AND NAME IS NOT NULL AND NAME != '' AND NAME != 'ctl-product'
      AND AMOUNT > 0 AND QUANTITY > 0
      AND CREATED_AT::DATE >= '2025-09-01'
),

product_avg_price AS (
    SELECT
        UPPER(TRIM(NAME))                                        AS product_name,
        ROUND(SUM(AMOUNT) / NULLIF(SUM(QUANTITY), 0), 2)        AS avg_unit_price
    FROM evaluation_pos_sale_details
    WHERE STATUS = 'Paid' AND AMOUNT > 0 AND QUANTITY > 0
    GROUP BY UPPER(TRIM(NAME))
),

clean_items AS (
    -- Join path: sale_detail -> inventory_store_products (STORE_PRODUCT_ID = isp.ID)
    --                        -> inventory_stores          (isp.STORE_ID     = s.ID)
    SELECT DISTINCT
        epsd.SALE_ID,
        UPPER(TRIM(epsd.NAME))                                                  AS product_name,
        SUM(epsd.AMOUNT) OVER (PARTITION BY epsd.SALE_ID, UPPER(TRIM(epsd.NAME))) AS line_total,
        COALESCE(s.NAME, 'Unknown')                                             AS store_name
    FROM evaluation_pos_sale_details epsd
    LEFT JOIN inventory_store_products isp ON epsd.STORE_PRODUCT_ID = isp.ID
    LEFT JOIN inventory_stores         s   ON isp.STORE_ID = s.ID
    WHERE epsd.STATUS = 'Paid'
      AND epsd.NAME IS NOT NULL AND epsd.NAME != '' AND epsd.NAME != 'ctl-product'
      AND epsd.AMOUNT > 0 AND epsd.QUANTITY > 0
      AND epsd.CREATED_AT::DATE >= '2025-09-01'
),

frequent_products AS (
    -- Frequency is global (cross-store) so baseline probabilities are consistent
    SELECT product_name, COUNT(DISTINCT SALE_ID) AS product_basket_count
    FROM clean_items
    GROUP BY product_name
    HAVING COUNT(DISTINCT SALE_ID) >= 10
),

baseline AS (
    SELECT fp.product_name, fp.product_basket_count,
           ROUND(fp.product_basket_count * 1.0 / tb.n, 6) AS prob
    FROM frequent_products fp CROSS JOIN total_baskets tb
),

cooccurrence AS (
    -- Co-occurrence is per store so the filter works correctly
    SELECT
        a.product_name  AS product_a_name,
        b.product_name  AS product_b_name,
        a.store_name    AS store_name,
        COUNT(DISTINCT a.SALE_ID) AS baskets_with_both
    FROM clean_items a
    JOIN clean_items b
        ON  a.SALE_ID       = b.SALE_ID
        AND a.store_name    = b.store_name
        AND a.product_name != b.product_name
        AND a.product_name  < b.product_name
    WHERE a.product_name IN (SELECT product_name FROM frequent_products)
      AND b.product_name IN (SELECT product_name FROM frequent_products)
    GROUP BY a.product_name, b.product_name, a.store_name
    HAVING COUNT(DISTINCT a.SALE_ID) >= 5
),

scored AS (
    SELECT
        co.store_name,
        co.product_a_name, co.product_b_name, co.baskets_with_both,
        base_a.product_basket_count                                         AS baskets_with_a,
        base_b.product_basket_count                                         AS baskets_with_b,
        ROUND(co.baskets_with_both * 1.0 / base_a.product_basket_count, 4) AS prob_b_given_a,
        ROUND(base_b.prob, 4)                                               AS prob_b_baseline,
        ROUND((co.baskets_with_both * 1.0 / base_a.product_basket_count) - base_b.prob, 4) AS csuc_score,
        ROUND((co.baskets_with_both * 1.0 / base_a.product_basket_count) / NULLIF(base_b.prob, 0), 2) AS lift
    FROM cooccurrence co
    JOIN baseline base_a ON co.product_a_name = base_a.product_name
    JOIN baseline base_b ON co.product_b_name = base_b.product_name
    WHERE (co.baskets_with_both * 1.0 / base_a.product_basket_count) > base_b.prob
      AND (co.baskets_with_both * 1.0 / base_a.product_basket_count) / NULLIF(base_b.prob, 0) > 1.2
)

SELECT
    store_name      AS "Store",
    product_a_name  AS "Product A",
    product_b_name  AS "Product B",
    baskets_with_both AS "Baskets together",
    baskets_with_a  AS "Baskets with A",
    baskets_with_b  AS "Baskets with B",
    prob_b_given_a  AS "P(B|A)",
    prob_b_baseline AS "P(B) baseline",
    csuc_score      AS "CSUC score",
    lift            AS "Lift",
    CASE
        WHEN baskets_with_both >= 50 THEN 'High'
        WHEN baskets_with_both >= 20 THEN 'Medium'
        ELSE 'Low'
    END AS "Signal strength",
    ROUND(COALESCE(pb.avg_unit_price, 0), 2)  AS "Avg Price B (KES)"
FROM scored
LEFT JOIN product_avg_price pb ON scored.product_b_name = pb.product_name
WHERE product_b_name NOT LIKE '%CARRIER BAG%'
  AND product_a_name NOT LIKE '%CARRIER BAG%'
  AND NOT (
        product_b_name LIKE '%WHITE BREAD%'
        AND product_a_name NOT LIKE '%BREAD%'
        AND product_a_name NOT LIKE '%BUNS%'
        AND product_a_name NOT LIKE '%CAKE%'
  )
  AND (baskets_with_both >= 20 OR (baskets_with_both >= 5 AND csuc_score > 0.15))
ORDER BY csuc_score DESC
"""


STORES_QUERY = """
SELECT DISTINCT
    s.NAME      AS store_name
FROM inventory_stores s
ORDER BY s.NAME
"""

TOP_PRODUCTS_QUERY = """
SELECT
    UPPER(TRIM(epsd.NAME))        AS product_name,
    ROUND(SUM(epsd.AMOUNT), 0)    AS total_revenue,
    SUM(epsd.QUANTITY)            AS total_units,
    COUNT(DISTINCT epsd.SALE_ID)  AS transaction_count
FROM evaluation_pos_sale_details epsd
WHERE epsd.STATUS   = 'Paid'
  AND epsd.AMOUNT   > 0
  AND epsd.QUANTITY > 0
  AND epsd.NAME IS NOT NULL
  AND epsd.NAME != ''
  AND epsd.NAME != 'ctl-product'
  AND epsd.CREATED_AT::DATE >= '2025-09-01'
GROUP BY UPPER(TRIM(epsd.NAME))
ORDER BY total_revenue DESC
LIMIT 50
"""

HOME_STATS_QUERY = """
SELECT
    (SELECT COUNT(DISTINCT s.ID)
     FROM inventory_stores s)                                          AS total_stores,

    (SELECT COUNT(DISTINCT sp.PRODUCT_ID)
     FROM inventory_store_products sp
     WHERE sp.PRODUCT_ACTIVE = TRUE)                                   AS active_skus,

    (SELECT COUNT(DISTINCT epsd.SALE_ID)
     FROM evaluation_pos_sale_details epsd
     WHERE epsd.STATUS = 'Paid'
       AND epsd.CREATED_AT::DATE >= '2025-09-01')                      AS total_transactions,

    (SELECT ROUND(SUM(epsd.AMOUNT), 0)
     FROM evaluation_pos_sale_details epsd
     WHERE epsd.STATUS = 'Paid'
       AND epsd.AMOUNT > 0
       AND epsd.CREATED_AT::DATE >= '2025-09-01')                      AS total_revenue
"""

HOME_ALERTS_QUERY = """
WITH soh AS (
    SELECT
        sp.PRODUCT_ID,
        sp.STORE_ID,
        SUM(ABS(sp.QUANTITY::FLOAT)) AS stock_on_hand
    FROM inventory_store_products sp
    WHERE sp.PRODUCT_ACTIVE = TRUE
    GROUP BY sp.PRODUCT_ID, sp.STORE_ID
),
demand AS (
    SELECT
        sp.PRODUCT_ID,
        sp.STORE_ID,
        ROUND(SUM(CASE
            WHEN d.CREATED_AT::DATE >= DATEADD('day', -30, '2026-03-18')
            THEN ABS(d.QUANTITY) ELSE 0 END) / 30.0, 4)                          AS daily_demand,
        ROUND(SUM(ABS(d.QUANTITY) * COALESCE(d.PRICE, 0)) /
              NULLIF(DATEDIFF('day', MIN(d.CREATED_AT::DATE), '2026-03-18'::DATE) + 1, 0), 2)
                                                                                  AS daily_revenue
    FROM inventory_inventory_dispensing d
    JOIN inventory_store_products sp ON d.STORE_PRODUCT_ID = sp.ID
    WHERE d.CREATED_AT::DATE BETWEEN '2025-09-01' AND '2026-03-18'
      AND d.QUANTITY != 0
    GROUP BY sp.PRODUCT_ID, sp.STORE_ID
),
classified AS (
    SELECT
        COALESCE(d.daily_revenue, 0) AS daily_rev,
        CASE
            WHEN s.stock_on_hand <= 0                                                THEN 'Stockout'
            WHEN COALESCE(d.daily_demand, 0) > 0
             AND s.stock_on_hand / d.daily_demand <= 7                               THEN 'Critical'
            ELSE 'Other'
        END AS status
    FROM soh s
    LEFT JOIN demand d ON s.PRODUCT_ID = d.PRODUCT_ID AND s.STORE_ID = d.STORE_ID
)
SELECT
    COUNT(CASE WHEN status = 'Stockout' THEN 1 END)                               AS stockouts,
    COUNT(CASE WHEN status = 'Critical' THEN 1 END)                               AS critical,
    ROUND(SUM(CASE WHEN status IN ('Stockout','Critical')
              THEN daily_rev * 7 ELSE 0 END), 0)                                  AS rev_at_risk_7d
FROM classified
"""

STORE_PULSE_QUERY = """
SELECT
    s.NAME                                                                         AS store_name,
    ROUND(SUM(CASE WHEN e.STATUS = 'Paid' AND e.AMOUNT > 0
                   THEN e.AMOUNT ELSE 0 END), 0)                                  AS total_revenue,
    COUNT(DISTINCT CASE WHEN e.STATUS = 'Paid' THEN e.SALE_ID END)                AS transactions,
    ROUND(SUM(CASE WHEN e.STATUS = 'Paid' AND e.AMOUNT > 0 THEN e.AMOUNT ELSE 0 END) /
          NULLIF(COUNT(DISTINCT CASE WHEN e.STATUS = 'Paid'
                                     THEN e.SALE_ID END), 0), 0)                  AS avg_basket_kes
FROM inventory_stores s
JOIN inventory_store_products isp ON s.ID = isp.STORE_ID
JOIN evaluation_pos_sale_details e
    ON e.STORE_PRODUCT_ID = isp.ID
    AND e.CREATED_AT::DATE >= '2025-09-01'
GROUP BY s.NAME
ORDER BY total_revenue DESC
"""

STOCKOUT_PREDICTION_QUERY = """
WITH stock_on_hand AS (
    SELECT
        sp.PRODUCT_ID,
        sp.STORE_ID                                         AS store_id,
        s.NAME                                              AS store_name,
        UPPER(TRIM(sp.PRODUCT_NAME))                        AS product_name,
        sp.PRODUCT_CATEGORY                                 AS category_id,
        ic.NAME                                             AS category_name,
        SUM(ABS(sp.QUANTITY::FLOAT))                        AS stock_on_hand,
        SUM(ABS(sp.QUANTITY::FLOAT) * COALESCE(sp.UNIT_COST, 0)) AS stock_value_ksh,
        MAX(sp.SELLING_PRICE::FLOAT)                        AS selling_price,
        AVG(NULLIF(sp.UNIT_COST, 0))                        AS avg_unit_cost,
        MAX(sp.RE_ORDER_LEVEL)                              AS reorder_level,
        MAX(sp.UPDATED_AT)                                  AS stock_last_updated,
        MAX(sp.PRODUCT_UNITS_SYMBOL)                        AS unit_symbol
    FROM inventory_store_products sp
    LEFT JOIN inventory_inventory_categories ic
        ON sp.PRODUCT_CATEGORY = ic.ID
    LEFT JOIN inventory_stores s
        ON sp.STORE_ID = s.ID
    WHERE sp.PRODUCT_ACTIVE = TRUE

    GROUP BY
        sp.PRODUCT_ID,
        sp.STORE_ID,
        s.NAME,
        UPPER(TRIM(sp.PRODUCT_NAME)),
        sp.PRODUCT_CATEGORY,
        ic.NAME
),

dispensing_demand AS (
    SELECT
        sp.PRODUCT_ID,
        sp.STORE_ID                                         AS store_id,
        TRY_TO_DATE(disp.CREATED_AT)                        AS dispensed_date,
        SUM(ABS(disp.QUANTITY))                             AS units_consumed,
        SUM(ABS(disp.QUANTITY) * disp.PRICE)                AS revenue_generated
    FROM inventory_inventory_dispensing disp
    JOIN inventory_store_products sp
        ON disp.STORE_PRODUCT_ID = sp.ID
    WHERE TRY_TO_DATE(disp.CREATED_AT) BETWEEN '2025-09-01' AND '2026-03-18'
      AND disp.QUANTITY != 0
    GROUP BY sp.PRODUCT_ID, sp.STORE_ID, TRY_TO_DATE(disp.CREATED_AT)
),

demand_summary AS (
    SELECT
        PRODUCT_ID,
        STORE_ID                                            AS store_id,
        COUNT(DISTINCT dispensed_date)                      AS days_with_sales,
        MIN(dispensed_date)                                 AS first_sale_date,
        MAX(dispensed_date)                                 AS last_sale_date,
        SUM(units_consumed)                                 AS total_units_consumed,
        SUM(revenue_generated)                              AS total_revenue,

        ROUND(
            SUM(units_consumed) /
            NULLIF(DATEDIFF('day', MIN(dispensed_date), MAX(dispensed_date)) + 1, 0),
        4)                                                  AS avg_daily_demand,

        ROUND(
            SUM(CASE
                WHEN dispensed_date >= DATEADD('day', -30, '2026-03-18'::DATE)
                THEN units_consumed ELSE 0 END)
            / 30.0, 4)                                      AS recent_30d_daily_demand,

        ROUND(
            SUM(CASE
                WHEN dispensed_date >= DATEADD('day', -60, '2026-03-18'::DATE)
                 AND dispensed_date <  DATEADD('day', -30, '2026-03-18'::DATE)
                THEN units_consumed ELSE 0 END)
            / 30.0, 4)                                      AS prior_30d_daily_demand,

        ROUND(
            SUM(revenue_generated) /
            NULLIF(DATEDIFF('day', MIN(dispensed_date), MAX(dispensed_date)) + 1, 0),
        2)                                                  AS avg_daily_revenue

    FROM dispensing_demand
    GROUP BY PRODUCT_ID, STORE_ID
),

combined AS (
    SELECT
        soh.*,

        COALESCE(ds.avg_daily_demand, 0)            AS avg_daily_demand,
        COALESCE(ds.recent_30d_daily_demand, 0)     AS recent_daily_demand,
        COALESCE(ds.prior_30d_daily_demand, 0)      AS prior_daily_demand,
        COALESCE(ds.avg_daily_revenue, 0)           AS avg_daily_revenue,
        COALESCE(ds.total_revenue, 0)               AS total_revenue,
        COALESCE(ds.total_units_consumed, 0)        AS total_units_consumed,
        COALESCE(ds.days_with_sales, 0)             AS days_with_sales,
        ds.first_sale_date,
        ds.last_sale_date,

        CASE
            WHEN COALESCE(ds.recent_30d_daily_demand, 0) > 0
            THEN ds.recent_30d_daily_demand
            WHEN COALESCE(ds.avg_daily_demand, 0) > 0
            THEN ds.avg_daily_demand
            ELSE 0
        END                                         AS effective_daily_demand

    FROM stock_on_hand soh
    LEFT JOIN demand_summary ds
        ON  soh.PRODUCT_ID = ds.PRODUCT_ID
        AND soh.store_id   = ds.store_id
),

predictions AS (
    SELECT
        *,

        CASE
            WHEN effective_daily_demand > 0 AND stock_on_hand > 0
            THEN ROUND(stock_on_hand / effective_daily_demand, 1)
            WHEN stock_on_hand <= 0 THEN 0
            ELSE NULL
        END                                         AS days_stock_remaining,

        CASE
            WHEN stock_on_hand <= 0
            THEN '2026-03-18'::DATE
            WHEN effective_daily_demand > 0
            THEN DATEADD('day',
                    LEAST(
                        ROUND(stock_on_hand / effective_daily_demand, 0)::INT,
                        3650
                    ),
                    '2026-03-18'::DATE)
            ELSE NULL
        END                                         AS predicted_stockout_date,

        ROUND(avg_daily_revenue * 7, 0)             AS revenue_at_risk_7d,

        CASE
            WHEN prior_daily_demand = 0 AND recent_daily_demand > 0 THEN 'New / Growing'
            WHEN prior_daily_demand = 0 AND recent_daily_demand = 0 THEN 'No recent sales'
            WHEN recent_daily_demand >= prior_daily_demand * 1.2 THEN 'Growing'
            WHEN recent_daily_demand <= prior_daily_demand * 0.8 THEN 'Declining'
            ELSE 'Stable'
        END                                         AS demand_trend,

        CASE
            WHEN effective_daily_demand > 0
            THEN CEIL(effective_daily_demand * 37)
            ELSE 0
        END                                         AS recommended_reorder_qty,

        ROUND(COALESCE(selling_price, 0) - COALESCE(avg_unit_cost, 0), 2) AS unit_margin,

        CASE
            WHEN COALESCE(selling_price, 0) > 0
            THEN ROUND(
                (selling_price - COALESCE(avg_unit_cost, 0)) / selling_price * 100,
            1)
            ELSE NULL
        END                                         AS margin_pct

    FROM combined
),

final AS (
    SELECT
        *,

        CASE
            WHEN stock_on_hand <= 0 THEN 'Stockout'
            WHEN effective_daily_demand = 0 AND stock_on_hand > 0 THEN 'No demand data'
            WHEN days_stock_remaining <= 7 THEN 'Critical'
            WHEN days_stock_remaining <= 14 THEN 'Warning'
            WHEN days_stock_remaining <= 30 THEN 'Monitor'
            WHEN days_stock_remaining > 90 THEN 'Overstocked'
            ELSE 'Healthy'
        END                                         AS stock_status,

        CASE
            WHEN stock_on_hand <= 0 THEN 1
            WHEN days_stock_remaining <= 7 THEN 2
            WHEN days_stock_remaining <= 14 THEN 3
            WHEN days_stock_remaining <= 30 THEN 4
            WHEN days_stock_remaining > 90 THEN 6
            ELSE 5
        END                                         AS urgency_rank

    FROM predictions
    WHERE (total_units_consumed > 0 OR stock_on_hand > 0)
      AND product_name NOT LIKE '%TRANSPORT%'
      AND product_name NOT LIKE '%SERVICE CHARGE%'
      AND product_name NOT LIKE '%DELIVERY CHARGE%'
      AND product_name NOT LIKE '%TRANSPORT COST%'
      AND product_name NOT LIKE '%AWAKEN TRAINING FOUNDATION%'
)

SELECT
    store_name                                  AS "Store",
    product_name                                AS "Product",
    category_name                               AS "Category",
    unit_symbol                                 AS "Unit",
    ROUND(stock_on_hand, 2)                     AS "Stock on Hand",
    ROUND(effective_daily_demand, 4)            AS "Daily Demand",
    days_stock_remaining                        AS "Days Stock Remaining",
    predicted_stockout_date                     AS "Predicted Stockout Date",
    stock_status                                AS "Stock Status",
    demand_trend                                AS "Demand Trend",
    ROUND(revenue_at_risk_7d, 0)                AS "7-Day Revenue at Risk (KES)",
    ROUND(total_revenue, 0)                     AS "Total Revenue (KES)",
    recommended_reorder_qty                     AS "Recommended Reorder Qty",
    ROUND(selling_price, 2)                     AS "Selling Price (KES)",
    ROUND(avg_unit_cost, 2)                     AS "Unit Cost (KES)",
    margin_pct                                  AS "Margin %",
    ROUND(stock_value_ksh, 0)                   AS "Stock Value (KES)",
    reorder_level                               AS "System Reorder Level",
    days_with_sales                             AS "Days with Sales",
    last_sale_date                              AS "Last Sale Date",
    stock_last_updated                          AS "Stock Last Updated",
    urgency_rank                                AS "Urgency Rank"

FROM final
ORDER BY urgency_rank ASC, revenue_at_risk_7d DESC;
"""
