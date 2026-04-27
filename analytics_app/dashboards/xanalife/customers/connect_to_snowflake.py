"""
data.py — XanaLife Snowflake Connection + Query Layer  v3
=========================================================
Changes from v2:
  - Removed SQL_HOUR_DOW (never used in app)
  - store_type added to every query that needs sidebar filtering
  - daily_velocity renamed to daily_spend_intensity throughout
  - avg_txns_per_day replaced with avg_active_days in segment query
  - STORE_CLUSTER_SQL / STORE_TYPE_SQL now use a single st alias
    to avoid copy-paste substitution errors
  - DLC OR-precedence bug fixed
  - load_mvar_coverage / overall / by_store now read from views
"""

import os
from pathlib import Path
import streamlit as st
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

SF_ACCOUNT       = os.getenv("SF_ACCOUNT")
SF_USER          = os.getenv("SF_USER")
SF_PASSWORD      = os.getenv("SF_PASSWORD")
SF_AUTHENTICATOR = os.getenv("SF_AUTHENTICATOR", "username_password_mfa")
SF_ROLE          = os.getenv("SF_ROLE")
SF_WAREHOUSE     = os.getenv("SF_WAREHOUSE")
SF_DATABASE      = os.getenv("SF_DATABASE", "HOSPITALS")
SF_SCHEMA        = os.getenv("SF_SCHEMA",   "XANALIFE_CLEAN")
MIN_DATE         = "2025-09-01"


# ─── CONNECTION ───────────────────────────────────────────────────────────────

def _snowflake_connect(**kwargs):
    return snowflake.connector.connect(
        account=SF_ACCOUNT, user=SF_USER, password=SF_PASSWORD,
        role=SF_ROLE, warehouse=SF_WAREHOUSE,
        database=SF_DATABASE, schema=SF_SCHEMA,
        authenticator=SF_AUTHENTICATOR,client_request_mfa_token=True, **kwargs,
    )

def connect_with_mfa_push():
    try:
        return _snowflake_connect()
    except Exception as e:
        if "MFA" in str(e) or "passcode" in str(e).lower():
            print("MFA push timed out. Falling back to passcode.")
            passcode = input("Enter MFA passcode: ").strip()
            return _snowflake_connect(passcode=passcode)
        raise


def get_connection():
    if "sf_conn" not in st.session_state:
        with st.spinner("Connecting to Snowflake — please approve the MFA push…"):
            st.session_state["sf_conn"] = connect_with_mfa_push()
    return st.session_state["sf_conn"]

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns, cast Decimal→float, date strings→datetime."""
    for col in df.columns:
        if df[col].dtype != object:
            continue
        if any(kw in col for kw in ("month", "date", "created_at")):
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            try:
                converted = pd.to_numeric(df[col])
                df[col] = converted
            except (ValueError, TypeError):
                pass
    return df

@st.cache_data(ttl=3600, show_spinner="Loading data…")
def run_query(sql: str) -> pd.DataFrame:
    def _fetch(conn):
        cur = conn.cursor()
        cur.execute(sql)
        cols = [d[0].lower() for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)
    try:
        df = _fetch(get_connection())
    except Exception as e:
        if "connection" in str(e).lower() or "session" in str(e).lower():
            st.session_state.pop("sf_conn", None)
            df = _fetch(get_connection())
        else:
            raise
    return _normalize_df(df)


# ─── STORE CLASSIFICATION ─────────────────────────────────────────────────────
# Both helpers use alias `st` — call them inside a query that already has
# inventory_stores joined as `st`.
# Wholesale stores: anything with 'Wholesale' OR 'Bulk' in the name.

STORE_CLUSTER_SQL = """
    CASE WHEN st.name ILIKE 'Katani%' THEN 'Katani' ELSE 'Syokimau' END
"""

STORE_TYPE_SQL = """
    CASE WHEN st.code = 'PHARMACY' OR st.name ILIKE '%Pharmacy%'
         THEN 'Pharmacy' ELSE 'Supermarket' END
"""

# Inline filter for excluding wholesale/bulk stores inside SQL
EXCL_WHOLESALE_SQL = """
    AND st.name NOT ILIKE '%Wholesale%'
    AND st.name NOT ILIKE '%Bulk%'
"""


# ═════════════════════════════════════════════════════════════════════════════
# CUSTOMER BASE + GROWTH
# ═════════════════════════════════════════════════════════════════════════════

SQL_KPIS = f"""
WITH active AS (
    SELECT patient, COUNT(DISTINCT id) AS t_transactions
    FROM inventory_inventory_batch_product_sales
    WHERE created_at::DATE >= '{MIN_DATE}'
      AND patient NOT IN (273017, 276430)
      AND status != 'canceled'
    GROUP BY patient
),
anchor AS (SELECT MAX(created_at) AS max_date FROM inventory_inventory_batch_product_sales),
hist AS (
    SELECT patient,
           MIN(created_at) AS first_ever_visit,
           COUNT(id)       AS lifetime_visits
    FROM inventory_inventory_batch_product_sales
    WHERE created_at::DATE >= '{MIN_DATE}'
    GROUP BY patient
)
SELECT
    COUNT(DISTINCT a.patient)                                          AS active_customers,
    COUNT(CASE WHEN a.t_transactions > 1 THEN 1 END)                   AS repeat_customers,
    COUNT(CASE WHEN h.lifetime_visits = 1 THEN 1 END)                  AS one_time_customers,
    COUNT(CASE WHEN h.first_ever_visit >= DATEADD(day,-30,an.max_date)
               THEN 1 END)                                             AS new_last_30d,
    (SELECT COUNT(DISTINCT customer_id) FROM points)                   AS loyalty_members
FROM active a
JOIN hist h ON a.patient = h.patient
CROSS JOIN anchor an
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_kpis() -> pd.DataFrame:
    return run_query(SQL_KPIS)


SQL_AVG_BASKET = f"""
SELECT
    ROUND(SUM(quantity) / NULLIF(COUNT(DISTINCT sale_id),0), 1) AS avg_items_per_basket,
    ROUND(SUM(amount)   / NULLIF(COUNT(DISTINCT sale_id),0), 0) AS avg_basket_value
FROM evaluation_pos_sale_details
WHERE created_at::DATE >= '{MIN_DATE}'
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_avg_basket() -> pd.DataFrame:
    return run_query(SQL_AVG_BASKET)


SQL_REGULAR_LOYAL = f"""
WITH anchor AS (SELECT MAX(created_at) AS last_point FROM evaluation_pos_sale_details),
regular AS (
    SELECT ps.patient, COUNT(s.sale_id) AS visit_count
    FROM inventory_inventory_batch_product_sales ps
    LEFT JOIN evaluation_pos_sale_details s ON ps.id = s.sale_id
    CROSS JOIN anchor
    WHERE s.created_at >= DATEADD(day, -90, anchor.last_point)
      AND ps.patient NOT IN (273017, 276430)
      AND ps.status != 'canceled'
    GROUP BY 1
    HAVING visit_count >= 6
),
loyal AS (
    SELECT ps.patient, COUNT(s.sale_id) AS visit_count,
           MAX(s.created_at) AS last_visit
    FROM inventory_inventory_batch_product_sales ps
    LEFT JOIN evaluation_pos_sale_details s ON ps.id = s.sale_id
    WHERE s.created_at >= DATEADD(day, -90, (SELECT MAX(created_at) FROM evaluation_pos_sale_details))
      AND ps.patient NOT IN (273017, 276430)
    GROUP BY 1
    HAVING visit_count >= 12
       AND last_visit >= DATEADD(day, -21, (SELECT MAX(created_at) FROM evaluation_pos_sale_details))
)
SELECT
    (SELECT COUNT(DISTINCT patient) FROM regular) AS regular_customers,
    (SELECT COUNT(DISTINCT patient) FROM loyal)   AS loyal_customers
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_regular_loyal() -> pd.DataFrame:
    return run_query(SQL_REGULAR_LOYAL)


SQL_DOW_BY_CLUSTER = f"""
SELECT
    DAYOFWEEK(TRY_TO_TIMESTAMP(b.created_at))                        AS day_num,
    COALESCE({STORE_CLUSTER_SQL}, 'Unknown')                          AS cluster,
    COALESCE({STORE_TYPE_SQL}, 'Supermarket')                         AS store_type,
    COUNT(DISTINCT b.id)                                             AS transaction_count,
    COUNT(DISTINCT b.patient)                                        AS unique_patients
FROM inventory_inventory_batch_product_sales b
LEFT JOIN evaluation_pos_sale_details s ON b.id = s.sale_id
LEFT JOIN inventory_store_products sp   ON s.store_product_id = sp.id
LEFT JOIN inventory_stores st           ON sp.store_id = st.id
WHERE TRY_TO_TIMESTAMP(b.created_at)::DATE >= '{MIN_DATE}'
  AND b.status != 'canceled'
  AND b.patient NOT IN (273017, 276430)
  AND (sp.product_active = TRUE OR sp.id IS NULL)
GROUP BY 1,2,3
ORDER BY 1
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_dow_by_cluster() -> pd.DataFrame:
    return run_query(SQL_DOW_BY_CLUSTER)


SQL_ACTIVE_CUSTOMERS_OVER_TIME = f"""
SELECT
    DATE_TRUNC('month', ps.created_at::DATE) AS month,
    COUNT(DISTINCT ps.patient)               AS active_customers
FROM inventory_inventory_batch_product_sales ps
LEFT JOIN evaluation_pos_sale_details s  ON ps.id = s.sale_id
LEFT JOIN inventory_store_products sp    ON s.store_product_id = sp.id
LEFT JOIN inventory_stores ins           ON sp.store_id = ins.id
WHERE ps.created_at::DATE >= '{MIN_DATE}'
  AND s.created_at::DATE  >= '{MIN_DATE}'
  AND ps.patient NOT IN (273017, 276430)
  AND sp.product_active = TRUE
GROUP BY 1
HAVING COUNT(s.sale_id) >= 1
ORDER BY 1
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_active_customers_over_time() -> pd.DataFrame:
    return run_query(SQL_ACTIVE_CUSTOMERS_OVER_TIME)


SQL_CONSISTENCY_SEGMENTS = """
SELECT
    CASE
        WHEN active_days = 1 THEN 'One-Time'
        WHEN active_days > 1
             AND DATEDIFF('day', first_purchase_date, last_purchase_date)
                 / NULLIF(active_days - 1, 0) <= 7  THEN 'Weekly'
        WHEN active_days > 1
             AND DATEDIFF('day', first_purchase_date, last_purchase_date)
                 / NULLIF(active_days - 1, 0) <= 14 THEN 'Bi-Weekly'
        WHEN active_days > 1
             AND DATEDIFF('day', first_purchase_date, last_purchase_date)
                 / NULLIF(active_days - 1, 0) <= 30 THEN 'Monthly'
        ELSE 'Sporadic'
    END                                                              AS shopping_rhythm,
    COUNT(customer_id)                                               AS customer_count,
    ROUND(AVG(avg_basket_value), 0)                                  AS avg_basket_value,
    ROUND(AVG(total_revenue), 0)                                     AS avg_lifetime_revenue,
    ROUND(AVG(
        DATEDIFF('day', first_purchase_date, last_purchase_date)
        / NULLIF(active_days - 1, 0)
    ), 1)                                                            AS avg_days_between_visits
FROM VW_CUSTOMER_SEGMENTATION
GROUP BY 1
ORDER BY avg_days_between_visits ASC NULLS LAST
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_consistency_segments() -> pd.DataFrame:
    return run_query(SQL_CONSISTENCY_SEGMENTS)


# ═════════════════════════════════════════════════════════════════════════════
# ONE-TIME CUSTOMER ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

SQL_ONETIMER_BASKET_TYPE = f"""
WITH ot_baskets AS (
    SELECT b.patient, SUM(s.amount) AS basket_value
    FROM inventory_inventory_batch_product_sales b
    JOIN evaluation_pos_sale_details s ON b.id = s.sale_id
    WHERE b.status != 'canceled'
      AND b.created_at::DATE >= '{MIN_DATE}'
      AND b.patient IN (
          SELECT customer_id FROM VW_CUSTOMER_SEGMENTATION
          WHERE refined_tier = '0 - One Time'
      )
    GROUP BY b.patient
)
SELECT
    CASE
        WHEN basket_value > 5000  THEN '1. Full Shop (>KSh 5K)'
        WHEN basket_value >= 1500 THEN '2. Stock-Up (1.5–5K)'
        ELSE                           '3. Top-Up (<1.5K)'
    END                                                          AS shop_type,
    COUNT(*)                                                     AS patient_count,
    ROUND(RATIO_TO_REPORT(COUNT(*)) OVER () * 100, 1)            AS pct_of_one_timers,
    ROUND(AVG(basket_value), 0)                                  AS avg_basket_value
FROM ot_baskets GROUP BY 1 ORDER BY 1
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_onetimer_basket_type() -> pd.DataFrame:
    return run_query(SQL_ONETIMER_BASKET_TYPE)


SQL_ONETIMER_PHARMACY_SPLIT = f"""
WITH ot_store_visits AS (
    SELECT DISTINCT b.patient,
        {STORE_TYPE_SQL} AS store_type
    FROM inventory_inventory_batch_product_sales b
    JOIN evaluation_pos_sale_details s ON b.id = s.sale_id
    JOIN inventory_store_products sp   ON s.store_product_id = sp.id
    JOIN inventory_stores st           ON sp.store_id = st.id
    WHERE b.status != 'canceled'
      AND b.created_at::DATE >= '{MIN_DATE}'
      AND sp.product_active = TRUE
      AND b.patient IN (
          SELECT customer_id FROM VW_CUSTOMER_SEGMENTATION
          WHERE refined_tier = '0 - One Time'
      )
),
classified AS (
    SELECT patient,
           MAX(CASE WHEN store_type = 'Pharmacy'    THEN 1 ELSE 0 END) AS saw_pharmacy,
           MAX(CASE WHEN store_type = 'Supermarket' THEN 1 ELSE 0 END) AS saw_super
    FROM ot_store_visits GROUP BY patient
)
SELECT
    CASE
        WHEN saw_pharmacy = 1 AND saw_super = 0 THEN 'Pharmacy Only'
        WHEN saw_pharmacy = 0 AND saw_super = 1 THEN 'Supermarket Only'
        ELSE 'Both'
    END                                                          AS entry_type,
    COUNT(*)                                                     AS patient_count,
    ROUND(RATIO_TO_REPORT(COUNT(*)) OVER () * 100, 1)            AS pct_of_one_timers
FROM classified GROUP BY 1 ORDER BY patient_count DESC
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_onetimer_pharmacy_split() -> pd.DataFrame:
    return run_query(SQL_ONETIMER_PHARMACY_SPLIT)


SQL_ONETIMER_URGENCY = f"""
WITH anchor AS (
    SELECT MAX(created_at::DATE) AS max_date
    FROM inventory_inventory_batch_product_sales
),
ot_timing AS (
    SELECT v.customer_id, v.last_purchase_date,
           DATEDIFF('day', v.last_purchase_date, a.max_date) AS days_since
    FROM VW_CUSTOMER_SEGMENTATION v
    CROSS JOIN anchor a
    WHERE v.refined_tier = '0 - One Time'
)
SELECT
    CASE
        WHEN days_since <= 14 THEN '1. Window open (≤14d) — act now'
        WHEN days_since <= 30 THEN '2. Cooling (15–30d) — still reachable'
        WHEN days_since <= 60 THEN '3. Cold (31–60d) — harder to convert'
        ELSE                       '4. Lost (60d+) — low ROI'
    END                                                          AS urgency_bucket,
    COUNT(*)                                                     AS patient_count,
    ROUND(RATIO_TO_REPORT(COUNT(*)) OVER () * 100, 1)            AS pct_of_one_timers
FROM ot_timing GROUP BY 1 ORDER BY 1
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_onetimer_urgency() -> pd.DataFrame:
    return run_query(SQL_ONETIMER_URGENCY)


SQL_ONETIMER_PRICE_SENSITIVE = f"""
WITH ot_cats AS (
    SELECT b.patient, c.name AS category_name,
           COUNT(DISTINCT c.name) OVER (PARTITION BY b.patient) AS cat_count
    FROM inventory_inventory_batch_product_sales b
    JOIN evaluation_pos_sale_details s     ON b.id = s.sale_id
    JOIN inventory_store_products sp       ON s.store_product_id = sp.id
    JOIN inventory_inventory_products ip   ON sp.product_id = ip.id
    JOIN inventory_inventory_categories c  ON ip.category = c.id
    WHERE b.status != 'canceled'
      AND b.created_at::DATE >= '{MIN_DATE}'
      AND sp.product_active = TRUE
      AND b.patient IN (
          SELECT customer_id FROM VW_CUSTOMER_SEGMENTATION
          WHERE refined_tier = '0 - One Time'
      )
),
ot_profile AS (
    SELECT patient, cat_count,
           MAX(CASE WHEN category_name ILIKE '%sugar%'
                     OR category_name ILIKE '%flour%'
                     OR category_name ILIKE '%oil%'
                     OR category_name ILIKE '%rice%'
                     OR category_name ILIKE '%bread%'
                    THEN 1 ELSE 0 END) AS only_staples
    FROM ot_cats GROUP BY patient, cat_count
)
SELECT
    CASE
        WHEN only_staples = 1 AND cat_count = 1 THEN 'Staples-only (price sensitive)'
        WHEN cat_count = 1                       THEN 'Single category (other)'
        ELSE                                          'Multi-category (genuine intent)'
    END                                                          AS customer_profile,
    COUNT(*)                                                     AS patient_count,
    ROUND(RATIO_TO_REPORT(COUNT(*)) OVER () * 100, 1)            AS pct
FROM ot_profile GROUP BY 1 ORDER BY patient_count DESC
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_onetimer_price_sensitive() -> pd.DataFrame:
    return run_query(SQL_ONETIMER_PRICE_SENSITIVE)


SQL_GROWTH_OVERALL = f"""
SELECT
    DATE_TRUNC('MONTH', first_visit_date) AS month,
    COUNT(DISTINCT patient)               AS new_customers
FROM (
    SELECT patient, MIN(created_at::DATE) AS first_visit_date
    FROM inventory_inventory_batch_product_sales
    WHERE created_at::DATE >= '{MIN_DATE}'
      AND patient NOT IN (273017, 276430)
      AND status != 'canceled'
    GROUP BY patient
) sub
GROUP BY 1 ORDER BY 1
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_growth_overall() -> pd.DataFrame:
    return run_query(SQL_GROWTH_OVERALL)


SQL_GROWTH_PER_STORE = f"""
WITH pb AS (
    SELECT
        DATE_TRUNC('month', ps.created_at::DATE) AS month,
        ps.patient,
        st.name                                  AS store_name,
        {STORE_CLUSTER_SQL}                      AS cluster,
        {STORE_TYPE_SQL}                         AS store_type,
        COUNT(s.sale_id)                         AS t_transactions
    FROM inventory_inventory_batch_product_sales ps
    LEFT JOIN evaluation_pos_sale_details s  ON ps.id = s.sale_id
    LEFT JOIN inventory_store_products sp    ON s.store_product_id = sp.id
    LEFT JOIN inventory_stores st            ON sp.store_id = st.id
    WHERE ps.created_at::DATE >= '{MIN_DATE}'
      AND s.created_at::DATE  >= '{MIN_DATE}'
      AND ps.patient NOT IN (273017, 276430)
      AND sp.product_active = TRUE
    GROUP BY 1, 2, 3, st.code
    HAVING t_transactions >= 1
),
first_visits AS (
    SELECT patient, MIN(month) AS first_month
    FROM pb
    GROUP BY patient
)
SELECT
    pb.month,
    pb.store_name,
    pb.cluster,
    pb.store_type,
    COUNT(DISTINCT CASE WHEN pb.month = fv.first_month THEN pb.patient END) AS new_customers
FROM pb
JOIN first_visits fv ON pb.patient = fv.patient
GROUP BY 1, 2, 3, 4
ORDER BY 1
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_growth_per_store() -> pd.DataFrame:
    return run_query(SQL_GROWTH_PER_STORE)


SQL_STORE_SPLIT = f"""
-- Classify each patient ONCE by majority spend store type.
-- Prevents double-counting patients who shopped at both pharmacy and supermarket.
WITH patient_spend_by_type AS (
    SELECT
        b.patient,
        {STORE_TYPE_SQL}               AS store_type,
        SUM(s.amount)                  AS spend
    FROM inventory_inventory_batch_product_sales b
    LEFT JOIN evaluation_pos_sale_details s ON b.id = s.sale_id
    LEFT JOIN inventory_store_products sp   ON s.store_product_id = sp.id
    LEFT JOIN inventory_stores st           ON sp.store_id = st.id
    WHERE b.created_at::DATE >= '{MIN_DATE}'
      AND b.patient NOT IN (273017, 276430)
      AND b.status != 'canceled'
      AND (sp.product_active = TRUE OR sp.id IS NULL)
    GROUP BY 1, 2
),
patient_primary_type AS (
    -- Each patient gets the store type where they spent the most
    SELECT patient, store_type
    FROM patient_spend_by_type
    QUALIFY ROW_NUMBER() OVER (PARTITION BY patient ORDER BY spend DESC) = 1
),
patient_store AS (
    -- Also get their primary store name and cluster for filtering
    SELECT
        b.patient,
        st.name                        AS store_name,
        {STORE_CLUSTER_SQL}            AS cluster
    FROM inventory_inventory_batch_product_sales b
    LEFT JOIN evaluation_pos_sale_details s ON b.id = s.sale_id
    LEFT JOIN inventory_store_products sp   ON s.store_product_id = sp.id
    LEFT JOIN inventory_stores st           ON sp.store_id = st.id
    WHERE b.created_at::DATE >= '{MIN_DATE}'
      AND b.patient NOT IN (273017, 276430)
      AND b.status != 'canceled'
      AND (sp.product_active = TRUE OR sp.id IS NULL)
    GROUP BY 1, 2, 3
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY b.patient
        ORDER BY SUM(s.amount) DESC
    ) = 1
)
SELECT
    pt.store_type,
    ps.store_name,
    ps.cluster,
    COUNT(DISTINCT pt.patient) AS customer_count
FROM patient_primary_type pt
JOIN patient_store ps ON pt.patient = ps.patient
GROUP BY 1, 2, 3
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_store_split() -> pd.DataFrame:
    return run_query(SQL_STORE_SPLIT)


SQL_SHOP_TYPE = f"""
WITH bt AS (
    SELECT sale_id,
        CASE
            WHEN SUM(amount) > 5000  THEN '1. Full Shop (>KSh 5K)'
            WHEN SUM(amount) >= 1500 THEN '2. Stock-Up (1.5–5K)'
            ELSE                          '3. Top-Up (<1.5K)'
        END AS shop_type,
        SUM(amount) AS basket_value
    FROM evaluation_pos_sale_details
    WHERE created_at::DATE >= '{MIN_DATE}'
    GROUP BY 1
)
SELECT
    shop_type,
    COUNT(*)                                                      AS basket_count,
    ROUND(RATIO_TO_REPORT(COUNT(*)) OVER () * 100, 1)             AS pct_of_trips,
    ROUND(AVG(basket_value), 0)                                   AS avg_basket_value
FROM bt GROUP BY 1 ORDER BY 1
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_shop_type() -> pd.DataFrame:
    return run_query(SQL_SHOP_TYPE)


SQL_CROSS_SELL = f"""
WITH psv AS (
    SELECT DATE_TRUNC('MONTH', s.created_at::DATE) AS month,
           b.patient,
           {STORE_CLUSTER_SQL}                      AS cluster,
           {STORE_TYPE_SQL}                         AS store_category
    FROM inventory_inventory_batch_product_sales b
    LEFT JOIN evaluation_pos_sale_details s ON b.id = s.sale_id
    LEFT JOIN inventory_store_products sp   ON s.store_product_id = sp.id
    LEFT JOIN inventory_stores st           ON sp.store_id = st.id
    WHERE b.created_at::DATE >= '{MIN_DATE}'
      AND b.patient NOT IN (273017, 276430)
      AND (sp.product_active = TRUE OR sp.id IS NULL)
),
mc AS (
    SELECT month, cluster, patient,
           COUNT(DISTINCT store_category) AS cats_visited
    FROM psv GROUP BY 1,2,3
)
SELECT
    month, cluster,
    COUNT(DISTINCT patient)                                        AS total_patients,
    COUNT(CASE WHEN cats_visited > 1 THEN patient END)             AS cross_shoppers,
    ROUND(
        COUNT(CASE WHEN cats_visited > 1 THEN patient END)
        / NULLIF(COUNT(DISTINCT patient),0) * 100, 1)              AS cross_sell_pct
FROM mc GROUP BY 1,2 ORDER BY 1
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_cross_sell() -> pd.DataFrame:
    return run_query(SQL_CROSS_SELL)


SQL_COOCCURRENCE = f"""
WITH pairs AS (
    SELECT a.sale_id,
           a.name       AS product_a,
           b.name       AS product_b,
           st.name      AS store_name,
           {STORE_TYPE_SQL}  AS store_type,
           {STORE_CLUSTER_SQL} AS cluster
    FROM evaluation_pos_sale_details a
    JOIN evaluation_pos_sale_details b
        ON a.sale_id = b.sale_id AND a.name < b.name
    LEFT JOIN inventory_store_products sp ON a.store_product_id = sp.id
    LEFT JOIN inventory_stores st         ON sp.store_id = st.id
    WHERE a.created_at::DATE >= '{MIN_DATE}'
      AND a.status != 'canceled'
),
ptc AS (
    SELECT name, COUNT(DISTINCT sale_id) AS product_txns
    FROM evaluation_pos_sale_details
    WHERE created_at::DATE >= '{MIN_DATE}'
    GROUP BY 1
),
agg AS (
    SELECT store_name, store_type, cluster,
           product_a, product_b,
           COUNT(*)                                              AS times_together,
           ROUND(COUNT(*) / NULLIF(MAX(ptc.product_txns),0) * 100, 1) AS pct_of_a_txns
    FROM pairs p
    JOIN ptc ON p.product_a = ptc.name
    GROUP BY 1,2,3,4,5
    HAVING COUNT(*) >= 5
)
SELECT store_name, store_type, cluster,
       product_a, product_b, times_together, pct_of_a_txns
FROM agg
ORDER BY store_name, times_together DESC
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_cooccurrence() -> pd.DataFrame:
    return run_query(SQL_COOCCURRENCE)


SQL_BASKET_BY_SIZE = f"""
WITH ss AS (
    SELECT sale_id,
           SUM(quantity) AS items,
           SUM(amount)   AS revenue
    FROM evaluation_pos_sale_details
    WHERE created_at::DATE >= '{MIN_DATE}'
    GROUP BY 1
),
bs AS (
    SELECT *,
        CASE
            WHEN items = 1              THEN '1 Item'
            WHEN items BETWEEN 2 AND 5  THEN '2–5 Items'
            WHEN items BETWEEN 6 AND 10 THEN '6–10 Items'
            ELSE                             '10+ Items'
        END AS basket_size_category
    FROM ss
)
SELECT
    basket_size_category,
    COUNT(sale_id)                                                AS transactions,
    ROUND(SUM(revenue), 0)                                        AS total_revenue,
    ROUND(RATIO_TO_REPORT(SUM(revenue)) OVER () * 100, 1)         AS pct_of_revenue,
    ROUND(AVG(revenue), 0)                                        AS avg_basket_value
FROM bs GROUP BY 1 ORDER BY MIN(items)
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_basket_by_size() -> pd.DataFrame:
    return run_query(SQL_BASKET_BY_SIZE)


SQL_BASKET_PER_STORE = f"""
SELECT
    st.name                                                        AS store_name,
    {STORE_CLUSTER_SQL}                                            AS cluster,
    {STORE_TYPE_SQL}                                               AS store_type,
    ROUND(SUM(s.quantity) / NULLIF(COUNT(DISTINCT s.sale_id),0),1) AS avg_items,
    ROUND(SUM(s.amount)   / NULLIF(COUNT(DISTINCT s.sale_id),0),0) AS avg_basket_value,
    COUNT(DISTINCT s.sale_id)                                      AS transactions
FROM evaluation_pos_sale_details s
LEFT JOIN inventory_store_products sp ON s.store_product_id = sp.id
LEFT JOIN inventory_stores st         ON sp.store_id = st.id
WHERE s.created_at::DATE >= '{MIN_DATE}'
  AND s.status != 'canceled'
  AND sp.product_active = TRUE
GROUP BY 1,2,3
ORDER BY avg_basket_value DESC
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_basket_per_store() -> pd.DataFrame:
    return run_query(SQL_BASKET_PER_STORE)


SQL_NEW_VS_RETURNING = f"""
WITH fp AS (
    SELECT patient, MIN(created_at::DATE) AS first_date
    FROM inventory_inventory_batch_product_sales
    WHERE status != 'canceled' AND patient NOT IN (273017,276430)
    GROUP BY patient
)
SELECT
    DATE_TRUNC('MONTH', b.created_at::DATE)                       AS month,
    CASE WHEN b.created_at::DATE = fp.first_date
         THEN 'New Customer Revenue'
         ELSE 'Returning Customer Revenue' END                     AS revenue_type,
    ROUND(SUM(b.amount), 0)                                        AS revenue
FROM inventory_inventory_batch_product_sales b
JOIN fp ON b.patient = fp.patient
WHERE b.created_at::DATE >= '{MIN_DATE}'
  AND b.status != 'canceled'
GROUP BY 1,2 ORDER BY 1
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_new_vs_returning() -> pd.DataFrame:
    return run_query(SQL_NEW_VS_RETURNING)


# ═════════════════════════════════════════════════════════════════════════════
# SEGMENTATION
# ═════════════════════════════════════════════════════════════════════════════

# Note: daily_velocity renamed to daily_spend_intensity in the view alias
SQL_SEGMENTS = """
SELECT
    refined_tier,
    COUNT(customer_id)                                             AS customer_count,
    ROUND(SUM(total_revenue), 0)                                   AS total_revenue,
    ROUND(AVG(avg_basket_value), 0)                                AS avg_basket_value,
    ROUND(AVG(active_days), 1)                                     AS avg_visits,
    ROUND(AVG(daily_velocity), 0)                                  AS avg_daily_spend_intensity,
    ROUND(RATIO_TO_REPORT(COUNT(customer_id)) OVER () * 100, 1)    AS pct_customers,
    ROUND(RATIO_TO_REPORT(SUM(total_revenue)) OVER () * 100, 1)    AS pct_revenue
FROM VW_CUSTOMER_SEGMENTATION
GROUP BY 1 ORDER BY 1
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_segments() -> pd.DataFrame:
    return run_query(SQL_SEGMENTS)


SQL_SEG_TREND = """
SELECT
    DATE_TRUNC('MONTH', first_purchase_date) AS month,
    refined_tier,
    COUNT(customer_id)                       AS customer_count
FROM VW_CUSTOMER_SEGMENTATION
GROUP BY 1,2 ORDER BY 1
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_seg_trend() -> pd.DataFrame:
    return run_query(SQL_SEG_TREND)


SQL_TOP_PRODUCTS_SEG = f"""
WITH tp AS (
    SELECT v.refined_tier,
           s.name                          AS product_name,
           COUNT(*)                        AS purchase_frequency,
           ROUND(SUM(s.amount), 0)         AS total_spend
    FROM inventory_inventory_batch_product_sales bp
    LEFT JOIN evaluation_pos_sale_details s ON bp.id = s.sale_id
    JOIN VW_CUSTOMER_SEGMENTATION v         ON bp.patient = v.customer_id
    WHERE bp.created_at::DATE >= '{MIN_DATE}'
      AND bp.status != 'canceled'
    GROUP BY 1,2
)
SELECT * FROM tp
QUALIFY ROW_NUMBER() OVER (PARTITION BY refined_tier ORDER BY total_spend DESC) <= 5
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_top_products_seg() -> pd.DataFrame:
    return run_query(SQL_TOP_PRODUCTS_SEG)


SQL_LAPSING_BY_SEGMENT = """
WITH cb AS (
    SELECT refined_tier, customer_id, total_revenue,
           DATEDIFF('day', first_purchase_date, last_purchase_date)
               / NULLIF(active_days - 1, 0)                        AS avg_gap,
           DATEDIFF('day', last_purchase_date,
               (SELECT MAX(created_at::DATE)
                FROM inventory_inventory_batch_product_sales))      AS absence
    FROM VW_CUSTOMER_SEGMENTATION
    WHERE active_days > 1
)
SELECT
    refined_tier,
    COUNT(customer_id)              AS lapsing_customers,
    ROUND(SUM(total_revenue), 0)    AS revenue_at_risk
FROM cb
WHERE absence > (avg_gap * 2) AND absence > 7
GROUP BY 1 ORDER BY revenue_at_risk DESC
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_lapsing_by_segment() -> pd.DataFrame:
    return run_query(SQL_LAPSING_BY_SEGMENT)


SQL_HEARTBEAT = """
SELECT
    refined_tier,
    ROUND(AVG(
        DATEDIFF('day', first_purchase_date, last_purchase_date)
        / NULLIF(active_days - 1, 0)
    ), 1)                           AS avg_heartbeat_days,
    ROUND(AVG(active_days), 1)      AS avg_active_days,
    COUNT(customer_id)              AS customer_count
FROM VW_CUSTOMER_SEGMENTATION
WHERE active_days > 1
GROUP BY 1 ORDER BY 1
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_heartbeat() -> pd.DataFrame:
    return run_query(SQL_HEARTBEAT)


SQL_CONVERSION_VELOCITY = f"""
WITH raw AS (
    SELECT patient, amount,
           created_at::DATE                                        AS visit_date,
           COUNT(*) OVER (PARTITION BY patient)                    AS total_visits,
           ROW_NUMBER() OVER (PARTITION BY patient ORDER BY created_at) AS visit_num
    FROM inventory_inventory_batch_product_sales
    WHERE status != 'canceled'
      AND created_at::DATE >= '{MIN_DATE}'
      AND patient NOT IN (273017,276430)
),
half AS (
    SELECT patient,
           CASE WHEN visit_num <= (total_visits/2)
                THEN 'Early' ELSE 'Recent' END                    AS phase,
           SUM(amount)                                            AS phase_rev,
           COUNT(DISTINCT visit_date)                             AS phase_days
    FROM raw WHERE total_visits >= 4
    GROUP BY 1,2
),
vel AS (
    SELECT patient,
           MAX(CASE WHEN phase='Early'  THEN phase_rev/NULLIF(phase_days,0) END) AS early_v,
           MAX(CASE WHEN phase='Recent' THEN phase_rev/NULLIF(phase_days,0) END) AS recent_v
    FROM half GROUP BY 1
)
SELECT
    CASE
        WHEN recent_v > early_v * 1.2 THEN 'Up-Converting'
        WHEN recent_v < early_v * 0.8 THEN 'Down-Converting'
        ELSE 'Stable'
    END                             AS conversion_status,
    COUNT(*)                        AS customer_count,
    ROUND(AVG(recent_v - early_v),0) AS avg_spend_shift_kes
FROM vel GROUP BY 1
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_conversion_velocity() -> pd.DataFrame:
    return run_query(SQL_CONVERSION_VELOCITY)


SQL_PAYMENT_BY_SEGMENT = f"""
SELECT
    COALESCE(v.refined_tier, 'Unregistered')                      AS refined_tier,
    COUNT(CASE WHEN f.cash_amount > 0 THEN 1 END)                 AS cash_count,
    COUNT(CASE WHEN f.mpesa_amount > 0 THEN 1 END)                AS mpesa_count,
    COUNT(CASE WHEN f.card_amount > 0 THEN 1 END)                 AS card_count,
    COUNT(CASE WHEN f.jambopay_amount > 0 THEN 1 END)             AS jambopay_count,
    COUNT(CASE WHEN f.pesa_pal_card_amount > 0 THEN 1 END)        AS pesapal_card_count,
    COUNT(CASE WHEN f.pesa_pal_mpesa_amount > 0 THEN 1 END)       AS pesapal_mpesa_count,
    COUNT(CASE WHEN f.voucher_amount > 0 THEN 1 END)              AS voucher_count,
    COUNT(CASE WHEN f.giftcard_amount > 0 THEN 1 END)             AS giftcard_count,
    COUNT(CASE WHEN f.loyalty_amount > 0 THEN 1 END)             AS loyalty_count,
    COUNT(CASE WHEN f.waiver_amount > 0 THEN 1 END)              AS waiver_count,
    COUNT(*)                                                       AS total_transactions
FROM finance_evaluation_payments f
LEFT JOIN VW_CUSTOMER_SEGMENTATION v ON f.patient = v.customer_id
WHERE f.created_at::DATE >= '{MIN_DATE}'
  AND (f.status IS NULL OR f.status != 'canceled')
GROUP BY 1 ORDER BY 1
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_payment_by_segment() -> pd.DataFrame:
    return run_query(SQL_PAYMENT_BY_SEGMENT)


SQL_FIRST_CATEGORY = f"""
WITH fp AS (
    SELECT b.patient,
           MIN(b.created_at::DATE)                                 AS first_purchase_date,
           FIRST_VALUE(s.sale_id) OVER (
               PARTITION BY b.patient ORDER BY b.created_at ASC
           )                                                       AS first_sale_id
    FROM inventory_inventory_batch_product_sales b
    JOIN evaluation_pos_sale_details s ON b.id = s.sale_id
    WHERE b.status != 'canceled'
      AND b.patient NOT IN (273017,276430)
    GROUP BY b.patient, s.sale_id, b.created_at
),
fc AS (
    SELECT fp.patient, fp.first_purchase_date, c.name AS first_category
    FROM fp
    JOIN evaluation_pos_sale_details s  ON fp.first_sale_id = s.sale_id
    JOIN inventory_store_products sp    ON s.store_product_id = sp.id
    JOIN inventory_inventory_products ip ON sp.product_id = ip.id
    JOIN inventory_inventory_categories c ON ip.category = c.id
    WHERE sp.product_active = TRUE
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY fp.patient ORDER BY s.created_at ASC
    ) = 1
)
SELECT
    fc.first_category,
    COUNT(DISTINCT fc.patient)                                     AS customer_count,
    ROUND(RATIO_TO_REPORT(COUNT(DISTINCT fc.patient)) OVER () * 100, 1) AS pct_new_customers,
    ROUND(AVG(v.active_days), 1)                                   AS avg_visits,
    ROUND(AVG(v.daily_velocity), 0)                                AS avg_daily_spend_intensity,
    ROUND(AVG(v.total_revenue), 0)                                 AS avg_lifetime_revenue,
    ROUND(COUNT(CASE WHEN v.active_days >= 6 THEN 1 END)
          / NULLIF(COUNT(*),0) * 100, 1)                           AS pct_became_regular,
    COUNT(CASE WHEN v.refined_tier LIKE '%Elite%' THEN 1 END)      AS became_elite,
    COUNT(CASE WHEN v.refined_tier = '0 - One Time' THEN 1 END)    AS stayed_one_time
FROM fc
JOIN VW_CUSTOMER_SEGMENTATION v ON fc.patient = v.customer_id
GROUP BY 1
ORDER BY avg_daily_spend_intensity DESC
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_first_category() -> pd.DataFrame:
    return run_query(SQL_FIRST_CATEGORY)


SQL_BASKET_EVOLUTION = f"""
WITH vp AS (
    SELECT b.patient, s.sale_id, c.name AS category_name, s.amount,
           ROW_NUMBER() OVER (PARTITION BY b.patient ORDER BY b.created_at ASC) AS visit_num,
           COUNT(*) OVER (PARTITION BY b.patient)                               AS total_visits
    FROM inventory_inventory_batch_product_sales b
    JOIN evaluation_pos_sale_details s     ON b.id = s.sale_id
    JOIN inventory_store_products sp       ON s.store_product_id = sp.id
    JOIN inventory_inventory_products ip   ON sp.product_id = ip.id
    JOIN inventory_inventory_categories c  ON ip.category = c.id
    WHERE b.status != 'canceled'
      AND b.patient NOT IN (273017,276430)
      AND b.created_at::DATE >= '{MIN_DATE}'
      AND sp.product_active = TRUE
),
ph AS (
    SELECT patient,
           CASE WHEN visit_num <= 3 THEN 'Early' ELSE 'Recent' END AS phase,
           COUNT(DISTINCT category_name)    AS cat_div,
           SUM(amount)                      AS phase_rev,
           COUNT(DISTINCT sale_id)          AS phase_visits
    FROM vp
    WHERE total_visits >= 4
      AND (visit_num <= 3 OR visit_num >= total_visits - 2)
    GROUP BY 1,2
),
pv AS (
    SELECT patient,
           MAX(CASE WHEN phase='Early'  THEN cat_div END)           AS early_cats,
           MAX(CASE WHEN phase='Recent' THEN cat_div END)           AS recent_cats,
           MAX(CASE WHEN phase='Early'  THEN phase_rev/NULLIF(phase_visits,0) END) AS early_basket,
           MAX(CASE WHEN phase='Recent' THEN phase_rev/NULLIF(phase_visits,0) END) AS recent_basket
    FROM ph GROUP BY 1
)
SELECT
    v.refined_tier,
    COUNT(pv.patient)                                              AS customer_count,
    ROUND(AVG(pv.early_cats), 1)                                   AS avg_early_categories,
    ROUND(AVG(pv.recent_cats), 1)                                  AS avg_recent_categories,
    ROUND(AVG(pv.recent_cats - pv.early_cats), 1)                  AS avg_diversity_change,
    ROUND(AVG(pv.early_basket), 0)                                 AS avg_early_basket,
    ROUND(AVG(pv.recent_basket), 0)                                AS avg_recent_basket,
    ROUND(AVG(pv.recent_basket - pv.early_basket), 0)              AS avg_basket_growth,
    COUNT(CASE WHEN pv.recent_cats > pv.early_cats
               AND pv.recent_basket > pv.early_basket THEN 1 END)  AS expanding,
    COUNT(CASE WHEN pv.recent_cats < pv.early_cats
               AND pv.recent_basket < pv.early_basket THEN 1 END)  AS shrinking
FROM pv
JOIN VW_CUSTOMER_SEGMENTATION v ON pv.patient = v.customer_id
GROUP BY 1 ORDER BY 1
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_basket_evolution() -> pd.DataFrame:
    return run_query(SQL_BASKET_EVOLUTION)


# ═════════════════════════════════════════════════════════════════════════════
# RETENTION
# ═════════════════════════════════════════════════════════════════════════════

SQL_RETURN_WINDOW = """
WITH cb AS (
    SELECT v.refined_tier,
           DATEDIFF('day', v.last_purchase_date,
               (SELECT MAX(created_at::DATE)
                FROM inventory_inventory_batch_product_sales))     AS days_since
    FROM VW_CUSTOMER_SEGMENTATION v
)
SELECT
    refined_tier,
    COUNT(CASE WHEN days_since <  30  THEN 1 END)                  AS within_30d,
    COUNT(CASE WHEN days_since BETWEEN 30 AND 60 THEN 1 END)       AS within_60d,
    COUNT(CASE WHEN days_since BETWEEN 61 AND 90 THEN 1 END)       AS within_90d,
    COUNT(CASE WHEN days_since > 90   THEN 1 END)                  AS over_90d,
    COUNT(*)                                                       AS total
FROM cb GROUP BY 1 ORDER BY 1
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_return_window() -> pd.DataFrame:
    return run_query(SQL_RETURN_WINDOW)


SQL_CHURN_BY_SEGMENT = """
WITH rec AS (
    SELECT customer_id, refined_tier, total_revenue,
           DATEDIFF('day', last_purchase_date,
               (SELECT MAX(created_at::DATE)
                FROM inventory_inventory_batch_product_sales))     AS days_since
    FROM VW_CUSTOMER_SEGMENTATION
)
SELECT
    refined_tier,
    COUNT(CASE WHEN days_since <  30  THEN 1 END)                  AS active,
    COUNT(CASE WHEN days_since BETWEEN 30 AND 60 THEN 1 END)       AS at_risk,
    COUNT(CASE WHEN days_since BETWEEN 61 AND 90 THEN 1 END)       AS lapsed,
    COUNT(CASE WHEN days_since > 90   THEN 1 END)                  AS lost,
    COUNT(*)                                                       AS total
FROM rec GROUP BY 1 ORDER BY 1
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_churn_by_segment() -> pd.DataFrame:
    return run_query(SQL_CHURN_BY_SEGMENT)


SQL_SECOND_PURCHASE = f"""
WITH ordered AS (
    SELECT patient, created_at::DATE AS visit_date,
           ROW_NUMBER() OVER (PARTITION BY patient ORDER BY created_at ASC) AS rn
    FROM inventory_inventory_batch_product_sales
    WHERE status != 'canceled'
      AND patient NOT IN (273017,276430)
      AND created_at::DATE >= '{MIN_DATE}'
),
gaps AS (
    SELECT o1.patient,
           DATEDIFF('day', o1.visit_date, o2.visit_date) AS days_to_second
    FROM ordered o1
    JOIN ordered o2 ON o1.patient = o2.patient AND o1.rn=1 AND o2.rn=2
)
SELECT
    v.refined_tier,
    ROUND(AVG(g.days_to_second), 1)                                AS avg_days_to_second,
    COUNT(g.patient)                                               AS customers_with_second,
    COUNT(v.customer_id)                                           AS total_customers,
    ROUND(COUNT(g.patient) * 100.0 / NULLIF(COUNT(v.customer_id),0), 1) AS pct_had_second_visit
FROM VW_CUSTOMER_SEGMENTATION v
LEFT JOIN gaps g ON v.customer_id = g.patient
GROUP BY 1 ORDER BY 1
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_second_purchase() -> pd.DataFrame:
    return run_query(SQL_SECOND_PURCHASE)


# ═════════════════════════════════════════════════════════════════════════════
# LOYALTY + PRODUCTS
# ═════════════════════════════════════════════════════════════════════════════

SQL_POINTS_BUCKETS = """
SELECT
    CASE
        WHEN points = 0    THEN '0 — No Points'
        WHEN points < 100  THEN '1 — Low (1–99)'
        WHEN points < 500  THEN '2 — Medium (100–499)'
        WHEN points < 1000 THEN '3 — High (500–999)'
        ELSE                    '4 — Elite (1000+)'
    END                             AS points_bucket,
    COUNT(DISTINCT customer_id)     AS customer_count
FROM points GROUP BY 1 ORDER BY 1
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_points_buckets() -> pd.DataFrame:
    return run_query(SQL_POINTS_BUCKETS)


SQL_LOYALTY_TREND = """
SELECT
    DATE_TRUNC('MONTH', created_at::DATE) AS month,
    COUNT(DISTINCT customer_id)           AS new_loyalty_members
FROM points GROUP BY 1 ORDER BY 1
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_loyalty_trend() -> pd.DataFrame:
    df = run_query(SQL_LOYALTY_TREND)
    df["cumulative"] = df["new_loyalty_members"].cumsum()
    return df


SQL_LOYALTY_RETURN = """
SELECT
    CASE WHEN p.customer_id IS NOT NULL THEN 'Loyalty Member'
         ELSE 'Non-Member' END                                     AS member_status,
    ROUND(AVG(
        DATEDIFF('day', v.first_purchase_date, v.last_purchase_date)
        / NULLIF(v.active_days - 1, 0)
    ), 1)                                                          AS avg_days_between_visits,
    COUNT(v.customer_id)                                           AS customer_count
FROM VW_CUSTOMER_SEGMENTATION v
LEFT JOIN (SELECT DISTINCT customer_id FROM points) p
       ON v.customer_id = p.customer_id
GROUP BY 1
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_loyalty_return() -> pd.DataFrame:
    return run_query(SQL_LOYALTY_RETURN)


SQL_LOYALTY_BY_SEGMENT = """
SELECT
    v.refined_tier,
    COUNT(DISTINCT v.customer_id)                                  AS total_customers,
    COUNT(DISTINCT p.customer_id)                                  AS loyalty_members,
    ROUND(COUNT(DISTINCT p.customer_id)
          / NULLIF(COUNT(DISTINCT v.customer_id),0) * 100, 1)      AS loyalty_pct
FROM VW_CUSTOMER_SEGMENTATION v
LEFT JOIN (SELECT DISTINCT customer_id FROM points) p
       ON v.customer_id = p.customer_id
GROUP BY 1 ORDER BY 1
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_loyalty_by_segment() -> pd.DataFrame:
    return run_query(SQL_LOYALTY_BY_SEGMENT)


SQL_LOYALTY_REDEMPTION = f"""
SELECT
    DATE_TRUNC('MONTH', f.created_at::DATE)                        AS month,
    COUNT(CASE WHEN f.loyalty_amount > 0 THEN 1 END)               AS redemption_transactions,
    ROUND(SUM(f.loyalty_amount), 0)                                AS loyalty_kes_redeemed,
    COUNT(DISTINCT CASE WHEN f.loyalty_amount > 0
                        THEN f.patient END)                        AS redeeming_customers
FROM finance_evaluation_payments f
WHERE f.created_at::DATE >= '{MIN_DATE}'
  AND f.status != 'canceled'
GROUP BY 1 ORDER BY 1
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_loyalty_redemption() -> pd.DataFrame:
    return run_query(SQL_LOYALTY_REDEMPTION)


SQL_LOYALTY_CONVERSION_LAG = f"""
WITH fp AS (
    SELECT patient, MIN(created_at::DATE) AS first_purchase_date
    FROM inventory_inventory_batch_product_sales
    WHERE status != 'canceled' AND patient NOT IN (273017,276430)
    GROUP BY patient
),
lj AS (
    SELECT customer_id, MIN(created_at::DATE) AS loyalty_join_date
    FROM points GROUP BY 1
)
SELECT
    CASE
        WHEN DATEDIFF('day', fp.first_purchase_date, lj.loyalty_join_date) = 0
            THEN 'Day 1'
        WHEN DATEDIFF('day', fp.first_purchase_date, lj.loyalty_join_date) <= 7
            THEN 'Within 1 week'
        WHEN DATEDIFF('day', fp.first_purchase_date, lj.loyalty_join_date) <= 30
            THEN 'Within 1 month'
        WHEN DATEDIFF('day', fp.first_purchase_date, lj.loyalty_join_date) <= 90
            THEN '1–3 months'
        ELSE '3+ months'
    END                                                            AS enrolment_lag,
    COUNT(*)                                                       AS customer_count,
    ROUND(AVG(v.daily_velocity), 0)                                AS avg_daily_spend_intensity,
    ROUND(AVG(v.active_days), 1)                                   AS avg_visits
FROM fp
JOIN lj ON fp.patient = lj.customer_id
JOIN VW_CUSTOMER_SEGMENTATION v ON fp.patient = v.customer_id
GROUP BY 1
ORDER BY MIN(DATEDIFF('day', fp.first_purchase_date, lj.loyalty_join_date))
"""

@st.cache_data(ttl=3600, show_spinner=False)
def load_loyalty_conversion_lag() -> pd.DataFrame:
    return run_query(SQL_LOYALTY_CONVERSION_LAG)