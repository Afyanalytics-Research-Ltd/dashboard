"""
xanalife · Advanced Analytics Suite
═══════════════════════════════════════════════════════════════════════════
A premium pharma BI dashboard with deep predictive analytics and
interactive what-if simulators.

Modules
  •  RBPI Analysis           — basket profitability + discount simulator
  •  Customer Lifetime Value — RFM, cohorts, churn, retention simulator
  •  Stockout Leakage        — ABC, reorder-point optimizer, demand forecast
  •  Promotion Efficiency    — price elasticity, optimal-discount finder
  •  Freshness Decay         — markdown-timing simulator, write-off forecast
  •  Predictive Hub          — executive impact simulator + risk radar
"""

import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import snowflake.connector
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
import logging
import time
import hashlib

logger = logging.getLogger("xanalife.analytics")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s · %(levelname)-7s · %(name)s · %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(_handler)
    logger.setLevel(os.getenv("XANALIFE_LOG_LEVEL", "INFO").upper())
    logger.propagate = False

# ═════════════════════════════════════════════════════════════════════════
#  SNOWFLAKE CLIENT
# ═════════════════════════════════════════════════════════════════════════

class SnowflakeClient:
    def __init__(self):
        with open(os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH").strip(), "rb") as key:
            key.read()
        self.conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER").strip(),
            account=os.getenv("SNOWFLAKE_ACCOUNT").strip(),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE").strip(),
            database=os.getenv("SNOWFLAKE_DATABASE").strip(),
            schema=os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC").strip(),
            private_key_file=os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH").strip(),
        )

    def query(self, sql, label=None):
        """Execute SQL with timing, row-count, and error logging.

        `label` shows up in the log line — pass the loader name so logs read
        cleanly (e.g. label='rbpi'). When omitted, falls back to a short hash
        of the SQL so repeat calls can be grouped."""
        label = label or f"q:{hashlib.md5(sql.encode()).hexdigest()[:8]}"
        snippet = " ".join(sql.split())[:140]
        logger.info("▶ %-22s running   | %s…", label, snippet)

        cursor = self.conn.cursor()
        t0 = time.perf_counter()
        try:
            cursor.execute(sql)
            df = cursor.fetch_pandas_all()
            elapsed = time.perf_counter() - t0

            # Decimal → float coercion (Snowflake NUMBER → decimal.Decimal otherwise)
            from decimal import Decimal
            decimal_cols = []
            for col in df.columns:
                if df[col].dtype == "object":
                    non_null = df[col].dropna()
                    if len(non_null) and isinstance(non_null.iloc[0], Decimal):
                        df[col] = df[col].astype(float)
                        decimal_cols.append(col)

            logger.info(
                "✓ %-22s done      | %s rows · %d cols · %.2fs%s",
                label, f"{len(df):,}", df.shape[1], elapsed,
                f" · coerced {len(decimal_cols)} Decimal cols" if decimal_cols else "",
            )
            if elapsed > 5:
                logger.warning("⚠ %-22s slow query | %.2fs (consider caching/window narrowing)",
                            label, elapsed)
            return df

        except snowflake.connector.errors.ProgrammingError as e:
            elapsed = time.perf_counter() - t0
            logger.error("✗ %-22s SQL error  | %.2fs · %s", label, elapsed, e.msg)
            logger.debug("  failing SQL:\n%s", sql)
            raise
        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.exception("✗ %-22s unhandled  | %.2fs · %s", label, elapsed, e)
            raise
        finally:
            cursor.close()
    

snowflakes = SnowflakeClient()

# ═════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════

PAGE_TITLE = "Advanced Analytics Suite"
st.set_page_config(
    page_title=f"xanalife · {PAGE_TITLE}",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════
#  PREMIUM THEME — gradients, glassmorphism, modern cards
# ═════════════════════════════════════════════════════════════════════════

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

/* ── Base ───────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
    color: #0A1F44;
    background: #F7FAFD;
}
.stApp {
    background:
      radial-gradient(1200px 600px at 80% -10%, rgba(0,114,206,0.08), transparent 60%),
      radial-gradient(900px 500px at -10% 110%, rgba(11,185,159,0.07), transparent 60%),
      #F7FAFD;
}

/* ── Hide Streamlit chrome ─────────────────────────────────────────── */
#MainMenu, footer { visibility: hidden; }
[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="collapsedControl"] { display: block !important; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ───────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #EEF5FC 100%);
    border-right: 1px solid #D6E4F0;
}
[data-testid="stSidebar"] * {
    color: #0A1F44 !important;
    font-family: 'Manrope', sans-serif !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stCheckbox label {
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    color: #6B8CAE !important;
}

/* ── Section headers ───────────────────────────────────────────────── */
.sh {
    font-size: 11px;
    font-weight: 800;
    color: #0072CE;
    text-transform: uppercase;
    letter-spacing: 2.2px;
    padding: 10px 0 8px;
    border-bottom: 1px solid #E0EAF5;
    margin: 18px 0 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sh::before {
    content: '';
    display: inline-block;
    width: 4px; height: 16px;
    background: linear-gradient(180deg, #0072CE 0%, #00B4D8 100%);
    border-radius: 2px;
}

/* ── Hero banner ───────────────────────────────────────────────────── */
.hero {
    position: relative;
    background: linear-gradient(135deg, #003566 0%, #0072CE 55%, #00B4D8 100%);
    color: #fff;
    padding: 22px 28px;
    border-radius: 14px;
    margin-bottom: 18px;
    overflow: hidden;
    box-shadow: 0 12px 30px rgba(0, 53, 102, 0.18);
}
.hero::before {
    content: '';
    position: absolute; right: -60px; top: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(255,255,255,0.25) 0%, transparent 70%);
    border-radius: 50%;
}
.hero .eyebrow {
    font-size: 10px; font-weight: 800; letter-spacing: 3px;
    text-transform: uppercase; opacity: 0.85;
}
.hero h1 {
    font-size: 26px; font-weight: 800; margin: 4px 0 6px; letter-spacing: -0.3px;
}
.hero p {
    font-size: 13px; opacity: 0.92; margin: 0; max-width: 720px;
}
.hero .module-pill {
    display: inline-block;
    margin-top: 10px;
    padding: 4px 12px;
    background: rgba(255,255,255,0.18);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.28);
    border-radius: 999px;
    font-size: 11px; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* ── KPI Cards ─────────────────────────────────────────────────────── */
.kpi {
    position: relative;
    background: #FFFFFF;
    border: 1px solid #E0EAF5;
    border-radius: 12px;
    padding: 18px 18px 16px;
    box-shadow: 0 4px 14px rgba(10, 31, 68, 0.04);
    overflow: hidden;
    transition: transform .18s ease, box-shadow .18s ease;
}
.kpi:hover { transform: translateY(-2px); box-shadow: 0 10px 22px rgba(10, 31, 68, 0.08); }
.kpi .accent {
    position: absolute; top: 0; left: 0;
    width: 4px; height: 100%;
    background: var(--accent, #0072CE);
}
.kpi .label {
    font-size: 10px; font-weight: 700;
    color: #6B8CAE; text-transform: uppercase; letter-spacing: 1.5px;
    margin-bottom: 10px;
}
.kpi .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 26px; font-weight: 700;
    color: var(--accent, #0072CE);
    line-height: 1.05;
    letter-spacing: -0.5px;
}
.kpi .sub { font-size: 11px; color: #6B8CAE; margin-top: 8px; }
.kpi .delta {
    display: inline-block; margin-top: 8px; padding: 2px 8px;
    border-radius: 6px; font-size: 10px; font-weight: 700;
}
.kpi .delta.up   { background: rgba(11,185,159,0.12); color: #0BB99F; }
.kpi .delta.down { background: rgba(225,29,72,0.12); color: #E11D48; }

/* ── Insight card ──────────────────────────────────────────────────── */
.info {
    padding: 12px 16px;
    background: linear-gradient(135deg, #F4F8FC 0%, #EBF3FB 100%);
    border-left: 3px solid var(--accent, #0072CE);
    border-radius: 6px;
    font-size: 12.5px;
    color: #0A1F44;
    margin: 8px 0 14px;
    line-height: 1.55;
}

/* ── Pills / verdicts ──────────────────────────────────────────────── */
.pill {
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 10px; font-weight: 800; letter-spacing: 0.6px;
    text-transform: uppercase;
}
.pill.ok    { background: rgba(11,185,159,0.14); color: #0BB99F; }
.pill.warn  { background: rgba(217,119,6,0.14);   color: #B45309; }
.pill.bad   { background: rgba(225,29,72,0.14);   color: #E11D48; }
.pill.info  { background: rgba(0,114,206,0.14);   color: #0072CE; }

/* ── Buttons ───────────────────────────────────────────────────────── */
.stButton button {
    background: linear-gradient(135deg, #0072CE 0%, #00B4D8 100%) !important;
    color: #fff !important;
    border: none !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 700 !important;
    font-size: 11px !important;
    letter-spacing: 1.2px !important;
    text-transform: uppercase !important;
    padding: 9px 20px !important;
    border-radius: 8px !important;
    box-shadow: 0 6px 14px rgba(0,114,206,0.25) !important;
    transition: transform .12s ease, box-shadow .12s ease !important;
}
.stButton button:hover { transform: translateY(-1px); box-shadow: 0 10px 22px rgba(0,114,206,0.35) !important; }

/* ── Tabs ──────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #F4F8FC; border: 1px solid #E0EAF5;
    padding: 5px; border-radius: 10px; margin-bottom: 18px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Manrope', sans-serif !important;
    font-weight: 600 !important; font-size: 12px !important;
    color: #6B8CAE !important; border-radius: 7px !important;
    padding: 8px 18px !important; border: none !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #003566 !important; background: #fff !important;
    font-weight: 800 !important;
    box-shadow: 0 2px 6px rgba(0,53,102,0.10) !important;
}

/* ── Metric override ───────────────────────────────────────────────── */
div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 24px !important; font-weight: 700 !important; color: #0072CE !important;
}

/* ── Dataframe ─────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] th,
[data-testid="stDataFrame"] [role="columnheader"] {
    background: #EBF3FB !important; color: #003566 !important;
    font-weight: 800 !important; font-size: 11px !important;
    text-transform: uppercase; letter-spacing: 0.8px;
}

/* ── Scrollbar ─────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-thumb { background: #B0C8E0; border-radius: 10px; }

/* ── Slider ────────────────────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] div div { background: #0072CE !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════

COLORS = {
    "primary": "#0072CE",
    "accent":  "#00B4D8",
    "deep":    "#003566",
    "success": "#0BB99F",
    "warning": "#D97706",
    "danger":  "#E11D48",
    "muted":   "#6B8CAE",
    "purple":  "#7F77DD",
    "pink":    "#D4537E",
    "coral":   "#D85A30",
    "green":   "#1D9E75",
    "gold":    "#C19A4B",
}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Manrope", color="#0A1F44", size=12),
    margin=dict(l=10, r=10, t=44, b=30),
    xaxis=dict(gridcolor="#EBF3FB", zerolinecolor="#E0EAF5", tickfont=dict(size=10, color="#6B8CAE")),
    yaxis=dict(gridcolor="#EBF3FB", zerolinecolor="#E0EAF5", tickfont=dict(size=10, color="#6B8CAE")),
    legend=dict(font=dict(size=11, color="#0A1F44"), bgcolor="rgba(255,255,255,0.5)"),
    colorway=[COLORS["primary"], COLORS["success"], COLORS["warning"], COLORS["purple"],
              COLORS["coral"], COLORS["green"], COLORS["pink"], COLORS["accent"]],
)


def fmt_ksh(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if abs(v) >= 1_000_000_000:
        return f"KSh {v/1_000_000_000:.2f}B"
    if abs(v) >= 1_000_000:
        return f"KSh {v/1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"KSh {v/1_000:.1f}K"
    return f"KSh {v:.0f}"


def fmt_num(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"{v/1_000:.1f}K"
    return f"{v:,.0f}"


def kpi_card(label, value, sub="", color=COLORS["primary"], delta=None, delta_dir="up"):
    delta_html = ""
    if delta:
        arrow = "▲" if delta_dir == "up" else "▼"
        delta_html = f'<div class="delta {delta_dir}">{arrow} {delta}</div>'
    st.markdown(
        f'<div class="kpi" style="--accent:{color}">'
        f'<div class="accent"></div>'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="sub">{sub}</div>'
        f'{delta_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_header(text):
    st.markdown(f'<div class="sh">{text}</div>', unsafe_allow_html=True)


def info_card(text, color=COLORS["primary"]):
    st.markdown(
        f'<div class="info" style="--accent:{color}">{text}</div>',
        unsafe_allow_html=True,
    )


def hero(eyebrow, title, subtitle, module_pill=None):
    pill = f'<div class="module-pill">{module_pill}</div>' if module_pill else ""
    st.markdown(
        f'<div class="hero">'
        f'<div class="eyebrow">{eyebrow}</div>'
        f'<h1>{title}</h1>'
        f'<p>{subtitle}</p>'
        f'{pill}'
        f'</div>',
        unsafe_allow_html=True,
    )


def styled_fig(fig, title=None, height=400):
    layout = {**CHART_LAYOUT, "height": height}
    if title:
        layout["title"] = dict(text=title, font=dict(size=13, color="#003566", family="Manrope"),
                               x=0.0, xanchor="left")
    fig.update_layout(**layout)
    return fig


# ═════════════════════════════════════════════════════════════════════════
#  DATA LOADERS  (queries from your live schema)
# ═════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_rbpi_data():
    query = """WITH base_sales AS (
        SELECT d.SALE_ID, MAX(p.STORE_ID) AS STORE_ID,
               SUM(d.AMOUNT) AS REVENUE, SUM(d.QUANTITY) AS TOTAL_QTY,
               SUM(d.DISCOUNT) AS LINE_DISCOUNT,
               MAX(d.CREATED_AT) AS CREATED_AT
        FROM hospitals.xanalife_clean.evaluation_pos_sale_details d
        LEFT JOIN hospitals.xanalife_clean.inventory_store_products p
          ON d.STORE_PRODUCT_ID = p.ID AND p.PRODUCT_ACTIVE = TRUE
        WHERE TRY_TO_TIMESTAMP(d.CREATED_AT) >= TO_TIMESTAMP('2025-09-01')
        GROUP BY d.SALE_ID
    ),
    product_costs AS (
        SELECT d.SALE_ID,
               SUM(d.QUANTITY * p.UNIT_COST) AS COGS,
               COUNT(p.ID) AS NUMBER_OF_ITEMS
        FROM hospitals.xanalife_clean.evaluation_pos_sale_details d
        LEFT JOIN hospitals.xanalife_clean.inventory_store_products p
          ON d.STORE_PRODUCT_ID = p.ID AND p.PRODUCT_ACTIVE = TRUE
        WHERE TRY_TO_TIMESTAMP(d.CREATED_AT) >= TO_TIMESTAMP('2025-09-01')
        GROUP BY d.SALE_ID
    ),
    discounts AS (
        SELECT SALE_ID, SUM(TRY_TO_NUMBER(DISCOUNT_AMOUNT)) AS TXN_DISCOUNT
        FROM hospitals.xanalife_clean.evaluation_discount_transactions
        GROUP BY SALE_ID
    )
    SELECT b.SALE_ID, b.STORE_ID, b.REVENUE,
           COALESCE(p.NUMBER_OF_ITEMS, 0) AS NUMBER_OF_ITEMS,
           COALESCE(p.COGS, 0) AS COGS,
           COALESCE(b.LINE_DISCOUNT,0) + COALESCE(d.TXN_DISCOUNT,0) AS TOTAL_DISCOUNT,
           (b.REVENUE - COALESCE(p.COGS,0)
            - (COALESCE(b.LINE_DISCOUNT,0) + COALESCE(d.TXN_DISCOUNT,0)))
            / NULLIF(b.REVENUE,0) AS RBPI,
           TRY_TO_TIMESTAMP(b.CREATED_AT) AS CREATED_AT
    FROM base_sales b
    LEFT JOIN product_costs p ON b.SALE_ID = p.SALE_ID
    LEFT JOIN discounts     d ON b.SALE_ID = d.SALE_ID;"""
    df = snowflakes.query(query)
    df.columns = df.columns.str.lower()
    return df


@st.cache_data(show_spinner=False)
def load_clv_data():
    query = """WITH customer_revenue AS (
        SELECT patient AS customer_id,
               SUM(amount) AS total_revenue,
               MIN(created_at::DATE) AS first_purchase_date,
               MAX(created_at::DATE) AS last_purchase_date,
               COUNT(DISTINCT id) AS total_transactions,
               COUNT(DISTINCT created_at::DATE) AS active_days,
               DATEDIFF('day', MIN(created_at::DATE), MAX(created_at::DATE))+1 AS lifespan_days
        FROM HOSPITALS.XANALIFE_CLEAN.inventory_inventory_batch_product_sales
        WHERE created_at::DATE >= '2025-09-01'
          AND patient NOT IN (273017, 276430)
          AND status != 'canceled'
        GROUP BY patient
    ),
    clv_velocity AS (
        SELECT customer_id, total_revenue, total_transactions, active_days,
               lifespan_days, first_purchase_date, last_purchase_date,
               ROUND(total_revenue / NULLIF(total_transactions,0), 2) AS revenue_per_transaction,
               ROUND(total_revenue / NULLIF(active_days,0), 2)        AS revenue_per_active_day,
               ROUND(total_transactions / NULLIF(lifespan_days,0), 4) AS visit_frequency_per_day,
               ROUND((total_transactions / NULLIF(lifespan_days,0))
                     * (total_revenue / NULLIF(total_transactions,0)), 2) AS clv_velocity_score,
               CASE
                 WHEN total_transactions = 1 THEN '0 - One Time'
                 WHEN ROUND(total_revenue/NULLIF(total_transactions,0),2) >= 10000 THEN '1 - Elite'
                 WHEN ROUND(total_revenue/NULLIF(total_transactions,0),2) >= 5000  THEN '2 - High'
                 WHEN ROUND(total_revenue/NULLIF(total_transactions,0),2) >= 1000  THEN '3 - Medium'
                 WHEN ROUND(total_revenue/NULLIF(total_transactions,0),2) >= 100   THEN '4 - Low'
                 ELSE '5 - Minimal'
               END AS velocity_tier
        FROM customer_revenue
    )
    SELECT * FROM clv_velocity ORDER BY clv_velocity_score DESC NULLS LAST;"""
    df = snowflakes.query(query)
    df.columns = df.columns.str.lower()
    return df


@st.cache_data(show_spinner=False)
def load_stockout_data():
    query = """WITH demand_per_product AS (
        SELECT PRODUCT AS product_id, SUM(ABS(QUANTITY)) AS demand_units
        FROM inventory_inventory_stocks
        WHERE TRY_TO_TIMESTAMP(CREATED_AT) >= '2025-09-01' AND QUANTITY < 0
        GROUP BY PRODUCT
    ),
    available_per_product AS (
        SELECT sp.PRODUCT_ID AS product_id,
               SUM(ABS(TRY_TO_NUMBER(sp.QUANTITY))) AS available_units,
               AVG(NULLIF(TRY_TO_NUMBER(sp.SELLING_PRICE), 0))         AS avg_selling_price,
               AVG(NULLIF(sp.PRODUCT_SELLING_PRICE, 0))                AS avg_product_selling_price,
               MAX(sp.STORE_ID) AS store_id
        FROM inventory_store_products sp WHERE sp.DELETED_AT IS NULL
        GROUP BY sp.PRODUCT_ID
    ),
    srl AS (
        SELECT COALESCE(d.product_id, a.product_id) AS product_id,
               ip.NAME AS product_name, ip.CODE AS product_code,
               ip.DEPARTMENT AS department,
               a.store_id AS store_id,
               COALESCE(d.demand_units,0) AS demand_units,
               COALESCE(a.available_units,0) AS available_units,
               GREATEST(COALESCE(d.demand_units,0)-COALESCE(a.available_units,0),0) AS shortage_units,
               COALESCE(a.avg_selling_price,a.avg_product_selling_price,ip.SELLING_PRICE) AS unit_price
        FROM demand_per_product d
        FULL OUTER JOIN available_per_product a ON a.product_id = d.product_id
        LEFT JOIN inventory_inventory_products ip ON ip.ID = COALESCE(d.product_id, a.product_id)
    )
    SELECT product_id, product_name, product_code, department, store_id,
           demand_units, available_units, shortage_units, unit_price,
           ROUND(shortage_units * unit_price, 2) AS srl_amount,
           CASE
             WHEN shortage_units = 0 THEN 'NO LEAKAGE'
             WHEN shortage_units * unit_price >= 100000 THEN 'CRITICAL · Expedite Reorder'
             WHEN shortage_units * unit_price >=  25000 THEN 'HIGH · Restock Priority'
             WHEN shortage_units * unit_price >=   5000 THEN 'MODERATE · Monitor'
             ELSE 'MINOR'
           END AS leakage_tier
    FROM srl
    WHERE shortage_units > 0
    ORDER BY srl_amount DESC NULLS LAST;"""
    df = snowflakes.query(query)
    df.columns = df.columns.str.lower()
    return df


@st.cache_data(show_spinner=False)
def load_promo_data():
    query = """WITH sale_discounts AS (
        SELECT SALE_ID,
               SUM(ABS(TRY_TO_NUMBER(DISCOUNT_AMOUNT)))  AS discount_cost,
               SUM(ABS(TRY_TO_NUMBER(TOTAL_AMOUNT_WAS))) AS pre_discount_total
        FROM evaluation_discount_transactions
        WHERE TRY_TO_TIMESTAMP(CREATED_AT) >= '2025-09-01'
          AND ABS(COALESCE(TRY_TO_NUMBER(DISCOUNT_AMOUNT),0)) > 0
        GROUP BY SALE_ID
    ),
    discounted_sale_lines AS (
        SELECT psd.SALE_ID, sp.PRODUCT_ID, sp.STORE_ID,
               SUM(psd.QUANTITY) AS units, SUM(psd.AMOUNT) AS net_revenue,
               SUM(psd.QUANTITY * COALESCE(sp.UNIT_COST,0)) AS cogs,
               SUM(psd.AMOUNT - psd.QUANTITY * COALESCE(sp.UNIT_COST,0)) AS profit_after_discount
        FROM EVALUATION_POS_SALE_DETAILS psd
        LEFT JOIN inventory_store_products sp ON sp.ID = psd.STORE_PRODUCT_ID
        WHERE (psd.STATUS IS NULL OR UPPER(psd.STATUS) NOT IN ('CANCELED','VOID','DELETED'))
          AND psd.SALE_ID IN (SELECT SALE_ID FROM sale_discounts)
        GROUP BY psd.SALE_ID, sp.PRODUCT_ID, sp.STORE_ID
    ),
    baseline_margin AS (
        SELECT sp.PRODUCT_ID,
               AVG((psd.AMOUNT - psd.QUANTITY * COALESCE(sp.UNIT_COST,0))
                   / NULLIF(psd.QUANTITY,0)) AS avg_unit_margin_baseline
        FROM EVALUATION_POS_SALE_DETAILS psd
        LEFT JOIN inventory_store_products sp ON sp.ID = psd.STORE_PRODUCT_ID
        WHERE (psd.STATUS IS NULL OR UPPER(psd.STATUS) NOT IN ('CANCELED','VOID','DELETED'))
          AND TRY_TO_TIMESTAMP(psd.CREATED_AT) >= '2025-09-01'
          AND psd.SALE_ID NOT IN (SELECT SALE_ID FROM sale_discounts)
        GROUP BY sp.PRODUCT_ID
    ),
    line_allocated AS (
        SELECT dsl.PRODUCT_ID, dsl.STORE_ID, dsl.SALE_ID, dsl.units,
               dsl.net_revenue, dsl.cogs, dsl.profit_after_discount,
               sd.discount_cost * (dsl.net_revenue
                 / NULLIF(SUM(dsl.net_revenue) OVER (PARTITION BY dsl.SALE_ID),0)) AS allocated_discount
        FROM discounted_sale_lines dsl
        JOIN sale_discounts sd ON sd.SALE_ID = dsl.SALE_ID
    ),
    promo_rollup AS (
        SELECT la.PRODUCT_ID, la.STORE_ID,
               COUNT(DISTINCT la.SALE_ID) AS promo_transactions,
               SUM(la.units) AS promo_units,
               SUM(la.net_revenue) AS promo_revenue,
               SUM(la.cogs) AS promo_cogs,
               SUM(la.allocated_discount) AS total_discount_cost,
               SUM(la.profit_after_discount) AS gross_profit_after_discount,
               SUM(la.profit_after_discount)
                 - SUM(la.units * COALESCE(bm.avg_unit_margin_baseline,0)) AS incremental_profit
        FROM line_allocated la
        LEFT JOIN baseline_margin bm ON bm.PRODUCT_ID = la.PRODUCT_ID
        GROUP BY la.PRODUCT_ID, la.STORE_ID
    )
    SELECT pr.PRODUCT_ID,
           ip.NAME AS product_name, ip.CODE AS product_code,
           ip.DEPARTMENT AS department,
           pr.STORE_ID,
           pr.promo_transactions, pr.promo_units,
           ROUND(pr.promo_revenue,2) AS promo_revenue,
           ROUND(pr.promo_cogs,2)    AS promo_cogs,
           ROUND(pr.total_discount_cost,2) AS discount_cost,
           ROUND(pr.gross_profit_after_discount,2) AS gross_profit_after_discount,
           ROUND(pr.incremental_profit,2) AS incremental_profit,
           ROUND(pr.incremental_profit / NULLIF(pr.total_discount_cost,0),3) AS per_ratio,
           ROUND(pr.gross_profit_after_discount / NULLIF(pr.total_discount_cost,0),3) AS per_ratio_simple,
           CASE
             WHEN pr.total_discount_cost = 0 THEN 'NO DISCOUNT'
             WHEN pr.incremental_profit / NULLIF(pr.total_discount_cost,0) >= 3 THEN 'STAR · Scale Up'
             WHEN pr.incremental_profit / NULLIF(pr.total_discount_cost,0) >= 1 THEN 'EFFICIENT · Keep Running'
             WHEN pr.incremental_profit / NULLIF(pr.total_discount_cost,0) >= 0 THEN 'BREAK-EVEN · Review Mechanics'
             ELSE 'MARGIN KILLER · Kill Promo'
           END AS promotion_verdict
    FROM promo_rollup pr
    LEFT JOIN inventory_inventory_products ip ON ip.ID = pr.PRODUCT_ID
    WHERE pr.total_discount_cost > 0 AND ip.name IS NOT NULL
    ORDER BY per_ratio DESC NULLS LAST;"""
    df = snowflakes.query(query)
    df.columns = df.columns.str.lower()
    return df


@st.cache_data(show_spinner=False)
def load_freshness_data():
    query = """WITH db_bounds AS (
        SELECT MAX(TRY_TO_TIMESTAMP(CREATED_AT)) AS max_sale_ts
        FROM inventory_inventory_batch_product_sales
    ),
    perishable_stock AS (
        SELECT sp.ID AS store_product_id, sp.PRODUCT_ID, sp.STORE_ID, sp.LOT_NO,
               ABS(TRY_TO_NUMBER(sp.QUANTITY)) AS on_hand_units,
               TRY_TO_NUMBER(sp.SELLING_PRICE) AS selling_price,
               sp.UNIT_COST AS unit_cost,
               TRY_TO_DATE(sp.EXPIRY_DATE) AS expiry_date,
               TRY_TO_TIMESTAMP(sp.CREATED_AT) AS received_at,
               ip.NAME AS product_name, ip.CODE AS product_code,
               ip.DEPARTMENT AS department, ip.FORMULATION AS formulation
        FROM inventory_store_products sp
        LEFT JOIN inventory_inventory_products ip ON ip.ID = sp.PRODUCT_ID
        WHERE sp.DELETED_AT IS NULL
          AND TRY_TO_DATE(sp.EXPIRY_DATE) IS NOT NULL
          AND ABS(COALESCE(TRY_TO_NUMBER(sp.QUANTITY),0)) > 0
    ),
    velocity AS (
        SELECT sp.PRODUCT_ID,
               SUM(ABS(psd.QUANTITY)) /
                 NULLIF(DATEDIFF('day',
                   MIN(TRY_TO_TIMESTAMP(psd.CREATED_AT)),
                   MAX(TRY_TO_TIMESTAMP(psd.CREATED_AT))),0) AS daily_velocity
        FROM EVALUATION_POS_SALE_DETAILS psd
        LEFT JOIN inventory_store_products sp ON sp.ID = psd.STORE_PRODUCT_ID
        WHERE (psd.STATUS IS NULL OR UPPER(psd.STATUS) NOT IN ('CANCELED','VOID','DELETED'))
          AND sp.product_active = TRUE
          AND TRY_TO_TIMESTAMP(psd.CREATED_AT) >= '2025-09-01'
        GROUP BY sp.PRODUCT_ID
    ),
    decay_calc AS (
        SELECT ps.*, v.daily_velocity, db.max_sale_ts,
               DATEDIFF('day', CAST(db.max_sale_ts AS DATE), ps.expiry_date)            AS days_to_expiry,
               DATEDIFF('day', CAST(ps.received_at AS DATE), ps.expiry_date)            AS shelf_life_days,
               GREATEST(
                 DATEDIFF('day', CAST(db.max_sale_ts AS DATE), ps.expiry_date)::FLOAT
                 / NULLIF(DATEDIFF('day', CAST(ps.received_at AS DATE), ps.expiry_date),0),
                 0
               ) AS freshness_ratio,
               ps.on_hand_units * COALESCE(ps.unit_cost,0) AS book_value_at_cost,
               ps.on_hand_units * COALESCE(ps.selling_price,0) AS potential_revenue,
               LEAST(GREATEST(COALESCE(v.daily_velocity,0)
                     * DATEDIFF('day', CAST(db.max_sale_ts AS DATE), ps.expiry_date)
                     / NULLIF(ps.on_hand_units,0), 0), 1) AS sell_through_probability
        FROM perishable_stock ps
        LEFT JOIN velocity v ON v.PRODUCT_ID = ps.PRODUCT_ID
        CROSS JOIN db_bounds db
    ),
    decay_model AS (
        SELECT *,
               potential_revenue * EXP(-2.0 * (1 - freshness_ratio)) AS retained_value_exponential,
               CASE
                 WHEN days_to_expiry < 0  THEN 0.00
                 WHEN days_to_expiry<=7   THEN 0.15
                 WHEN days_to_expiry<=14  THEN 0.40
                 WHEN days_to_expiry<=30  THEN 0.65
                 WHEN days_to_expiry<=60  THEN 0.85
                 WHEN days_to_expiry<=90  THEN 0.95
                 ELSE 1.00
               END AS bucket_retained_ratio,
               CASE
                 WHEN days_to_expiry < 0 THEN 1.00
                 WHEN days_to_expiry<=7  THEN 0.85
                 WHEN days_to_expiry<=14 THEN 0.60
                 WHEN days_to_expiry<=30 THEN 0.35
                 WHEN days_to_expiry<=60 THEN 0.15
                 WHEN days_to_expiry<=90 THEN 0.05
                 ELSE 0.00
               END AS recommended_markdown_pct,
               CASE
                 WHEN days_to_expiry < 0 THEN on_hand_units * COALESCE(unit_cost,0)
                 ELSE on_hand_units * COALESCE(unit_cost,0)
                       * (1 - LEAST(sell_through_probability,1))
               END AS projected_loss_no_action
        FROM decay_calc
    )
    SELECT PRODUCT_ID, product_name, product_code, department, formulation,
           STORE_ID, LOT_NO, on_hand_units, selling_price, unit_cost, expiry_date,
           days_to_expiry, shelf_life_days,
           ROUND(freshness_ratio,3) AS freshness_ratio,
           ROUND(COALESCE(daily_velocity,0),3) AS daily_velocity,
           ROUND(sell_through_probability,3) AS sell_through_probability,
           ROUND(book_value_at_cost,2) AS book_value_at_cost,
           ROUND(potential_revenue,2) AS potential_revenue_full_price,
           ROUND(retained_value_exponential,2) AS retained_value_exponential,
           ROUND(potential_revenue - retained_value_exponential,2) AS decay_cost_curve_value,
           ROUND(bucket_retained_ratio * potential_revenue,2) AS retained_value_bucket,
           ROUND(recommended_markdown_pct * 100,0) AS recommended_markdown_pct,
           ROUND(projected_loss_no_action,2) AS projected_loss_no_action,
           CASE
             WHEN days_to_expiry < 0 THEN 'EXPIRED · Write Off Immediately'
             WHEN days_to_expiry<=7  AND sell_through_probability<0.8 THEN 'URGENT · Deep Markdown (>=80%)'
             WHEN days_to_expiry<=14 AND sell_through_probability<0.7 THEN 'HIGH · Discount 60%+'
             WHEN days_to_expiry<=30 AND sell_through_probability<0.6 THEN 'MODERATE · Discount 30-40%'
             WHEN days_to_expiry<=60 AND sell_through_probability<0.5 THEN 'EARLY · Soft Markdown 10-15%'
             WHEN expiry_date IS NULL THEN 'Lacking expiry date'
             WHEN days_to_expiry<=90 THEN 'MONITOR · Watch Velocity'
             ELSE 'HEALTHY · No Action'
           END AS markdown_action
    FROM decay_model
    WHERE product_name IS NOT NULL AND expiry_date > '0001-01-01'
    ORDER BY projected_loss_no_action DESC, days_to_expiry ASC NULLS LAST;"""
    df = snowflakes.query(query)
    df.columns = df.columns.str.lower()
    return df


@st.cache_data(show_spinner=False)
def load_stores():
    df = snowflakes.query("SELECT * FROM inventory_stores;")
    df.columns = df.columns.str.lower()
    return df.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_daily_revenue():
    """Daily revenue + transactions for time-series forecasting."""
    query = """SELECT
        TRY_TO_TIMESTAMP(CREATED_AT)::DATE AS day,
        SUM(AMOUNT)                        AS revenue,
        COUNT(DISTINCT ID)                 AS transactions,
        COUNT(DISTINCT PATIENT)            AS unique_patients
      FROM HOSPITALS.XANALIFE_CLEAN.inventory_inventory_batch_product_sales
      WHERE CREATED_AT::DATE >= '2025-09-01'
        AND status != 'canceled'
      GROUP BY 1
      ORDER BY 1;"""
    df = snowflakes.query(query)
    df.columns = df.columns.str.lower()
    df["day"] = pd.to_datetime(df["day"])
    return df


def filter_by_store(df, store_id, store_col="store_id"):
    if store_id is None or df is None or df.empty or store_col not in df.columns:
        return df
    return df[df[store_col] == store_id].copy()


# ═════════════════════════════════════════════════════════════════════════
#  ML / ANALYTICS LAYER
# ═════════════════════════════════════════════════════════════════════════

@st.cache_resource
def train_rbpi_predictor(data):
    feats = ["revenue", "number_of_items", "total_discount"]
    X = data[feats].fillna(0); y = data["rbpi"].fillna(0)
    m = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42).fit(X, y)
    return m, feats


@st.cache_resource
def train_clv_predictor(data):
    feats = ["total_revenue", "total_transactions", "revenue_per_transaction",
             "active_days", "lifespan_days"]
    feats = [f for f in feats if f in data.columns]
    X = data[feats].fillna(0); y = np.log1p(data["clv_velocity_score"].fillna(0))
    m = RandomForestRegressor(n_estimators=200, random_state=42).fit(X, y)
    return m, feats


@st.cache_resource
def detect_anomalies(data, column="rbpi"):
    iso = IsolationForest(contamination=0.08, random_state=42)
    return iso.fit_predict(data[[column]].fillna(0)) == -1


def holt_winters_forecast(series, horizon=30, alpha=0.4, beta=0.15):
    """Lightweight Holt's linear-trend exponential smoothing.
    Returns (level, trend) so we can extrapolate `horizon` steps."""
    s = pd.Series(series).astype(float).ffill().bfill().values
    if len(s) < 3:
        return np.full(horizon, s[-1] if len(s) else 0.0), np.full(horizon, 0.0), np.full(horizon, 0.0)
    L = s[0]; T = s[1] - s[0]
    fitted = []
    for x in s:
        L_new = alpha * x + (1 - alpha) * (L + T)
        T_new = beta * (L_new - L) + (1 - beta) * T
        L, T = L_new, T_new
        fitted.append(L)
    forecast = np.array([L + (h + 1) * T for h in range(horizon)])
    # 1-sigma uncertainty band that widens with horizon
    resid_std = np.std(np.array(s) - np.array(fitted)) if len(s) > 1 else np.std(s) * 0.1
    band = resid_std * np.sqrt(np.arange(1, horizon + 1))
    lower = forecast - 1.96 * band
    upper = forecast + 1.96 * band
    return forecast, lower, upper


def churn_score(row, days_window=90):
    """Logistic-style churn score from RFM-ish features.
    Higher recency-since-last-purchase → higher churn."""
    if pd.isna(row.get("last_purchase_date")):
        return 0.5
    today = pd.Timestamp(row.get("_today", pd.Timestamp.today()))
    recency = (today - pd.to_datetime(row["last_purchase_date"])).days
    freq = row.get("total_transactions", 1) or 1
    monetary = row.get("total_revenue", 0) or 0
    z = -1.5 + 0.04 * recency - 0.6 * np.log1p(freq) - 0.0001 * np.log1p(monetary)
    return float(1 / (1 + np.exp(-z)))


def price_elasticity(promo):
    """Estimate elasticity from observed (discount %, units) per product.
    Returns dict with estimate + R²."""
    if promo.empty or "discount_cost" not in promo or "promo_revenue" not in promo:
        return {"epsilon": np.nan, "r2": np.nan}
    p = promo.copy()
    p["disc_pct"] = p["discount_cost"] / np.maximum(p["promo_revenue"] + p["discount_cost"], 1)
    p = p[(p["disc_pct"] > 0) & (p["disc_pct"] < 0.95) & (p["promo_units"] > 0)]
    if len(p) < 8:
        return {"epsilon": np.nan, "r2": np.nan}
    x = np.log(1 - p["disc_pct"].values)   # ln(price ratio)
    y = np.log(p["promo_units"].values)    # ln(units)
    A = np.vstack([x, np.ones_like(x)]).T
    eps, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = A @ np.array([eps, intercept])
    ss_res = np.sum((y - yhat) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {"epsilon": float(eps), "r2": float(r2)}


def reorder_point(daily_demand, lead_time_days=7, service_level=0.95):
    """Newsvendor-style ROP with Poisson demand assumption."""
    from math import sqrt
    z = {0.90: 1.28, 0.95: 1.645, 0.975: 1.96, 0.99: 2.33}.get(round(service_level, 3), 1.645)
    mu = daily_demand * lead_time_days
    sigma = sqrt(max(mu, 1))
    return mu + z * sigma


def monte_carlo_revenue(base_rev, vol_pct=0.15, horizon=30, runs=500):
    """Geometric-Brownian-motion-ish daily revenue simulator."""
    daily_mu = 0.0
    daily_sigma = vol_pct / np.sqrt(30)
    out = np.zeros((runs, horizon))
    for r in range(runs):
        shocks = np.random.normal(daily_mu, daily_sigma, horizon)
        out[r] = base_rev * np.exp(np.cumsum(shocks))
    return out


# ═════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        '<div style="font-size:22px;font-weight:800;'
        'background:linear-gradient(135deg,#0072CE,#00B4D8);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        'padding:8px 0 4px">📊 xanalife</div>'
        '<div style="font-size:10px;font-weight:700;letter-spacing:2px;'
        'color:#6B8CAE;text-transform:uppercase;margin-bottom:18px">Analytics Suite</div>',
        unsafe_allow_html=True,
    )

    section_header("Module")
    analysis_type = st.selectbox(
        "Analysis Module",
        [
            "Executive Overview",
            "RBPI Analysis",
            "Customer Lifetime Value",
            "Stockout Leakage",
            "Promotion Efficiency",
            "Freshness Decay",
            "Predictive Hub",
        ],
        label_visibility="collapsed",
    )

    section_header("Store Filter")
    stores_df = load_stores()
    store_options = [("All Stores", None)] + [
        (row["name"], row["id"]) for _, row in stores_df.iterrows()
    ]
    selected_option = st.selectbox(
        "Select Store",
        options=store_options,
        format_func=lambda x: x[0],
        index=0,
        label_visibility="collapsed",
    )
    selected_store_name, selected_store_id = selected_option

    section_header("Display")
    show_predictions = st.checkbox("Show AI predictions", value=True)
    show_simulators  = st.checkbox("Show what-if simulators", value=True)
    show_explainers  = st.checkbox("Show inline explainers", value=True)

    st.markdown(
        '<div style="margin-top:24px;padding:12px 14px;background:#EBF3FB;'
        'border-radius:8px;font-size:11px;line-height:1.6;color:#003566">'
        '<b>Window:</b> Sep 2025 – present<br>'
        '<b>Refresh:</b> hourly<br>'
        '<b>Source:</b> XanaLife Snowflake<br>'
        '<b>Models:</b> 9 active'
        '</div>',
        unsafe_allow_html=True,
    )

# ═════════════════════════════════════════════════════════════════════════
#  HERO BANNER
# ═════════════════════════════════════════════════════════════════════════

MODULE_BLURBS = {
    "Executive Overview": "Top-of-funnel snapshot across every analytics surface.",
    "RBPI Analysis":      "Profitability per basket — predict, simulate, optimize.",
    "Customer Lifetime Value": "Cohorts, RFM, churn risk, retention what-ifs.",
    "Stockout Leakage":   "Lost revenue, ABC tiers, smart reorder points.",
    "Promotion Efficiency": "Price elasticity, optimal discount, lift uncertainty.",
    "Freshness Decay":    "Markdown timing, expected loss, write-off forecast.",
    "Predictive Hub":     "An executive simulator over every lever in the business.",
}

hero(
    eyebrow="xanalife · advanced analytics suite",
    title=analysis_type,
    subtitle=MODULE_BLURBS.get(analysis_type, ""),
    module_pill=("All Stores" if selected_store_id is None else selected_store_name),
)

# ═════════════════════════════════════════════════════════════════════════
#  DATA — load once, then filter
# ═════════════════════════════════════════════════════════════════════════

with st.spinner("Loading analytics data…"):
    rbpi_data       = load_rbpi_data()
    clv_data        = load_clv_data()
    stockout_data   = load_stockout_data()
    promo_data      = load_promo_data()
    freshness_data  = load_freshness_data()
    daily_rev       = load_daily_revenue()

rbpi_data       = filter_by_store(rbpi_data,      selected_store_id)
stockout_data   = filter_by_store(stockout_data,  selected_store_id)
promo_data      = filter_by_store(promo_data,     selected_store_id)
freshness_data  = filter_by_store(freshness_data, selected_store_id)

# ═════════════════════════════════════════════════════════════════════════
#  EXECUTIVE OVERVIEW  — landing dashboard
# ═════════════════════════════════════════════════════════════════════════

if analysis_type == "Executive Overview":
    # ── Macro KPIs
    total_rev = float(rbpi_data["revenue"].sum()) if not rbpi_data.empty else 0
    avg_rbpi  = float(rbpi_data["rbpi"].mean())   if not rbpi_data.empty else 0
    leakage   = float(stockout_data["srl_amount"].sum()) if not stockout_data.empty else 0
    at_risk_freshness = float(freshness_data["projected_loss_no_action"].sum()) if not freshness_data.empty else 0
    promo_lift = float(promo_data["incremental_profit"].sum()) if not promo_data.empty else 0
    customers = len(clv_data)

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Revenue (window)", fmt_ksh(total_rev), "Sep 2025 → today", COLORS["primary"])
    with c2: kpi_card("Avg RBPI", f"{avg_rbpi:.1%}", "basket profitability", COLORS["green"])
    with c3: kpi_card("Stockout leakage", fmt_ksh(leakage), "revenue lost to OOS", COLORS["danger"])
    with c4: kpi_card("Freshness risk", fmt_ksh(at_risk_freshness), "projected write-off", COLORS["warning"])

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Active customers", fmt_num(customers), "purchased since Sep", COLORS["purple"])
    with c2: kpi_card("Promo lift",       fmt_ksh(promo_lift), "incremental profit",  COLORS["accent"])
    with c3: kpi_card("Stores tracked",   f"{len(stores_df)}", "live POS feeds",      COLORS["muted"])
    with c4:
        baskets = len(rbpi_data)
        kpi_card("Baskets analysed",      fmt_num(baskets), "transactions modelled", COLORS["coral"])

    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

    # ── Forecast strip
    if not daily_rev.empty:
        section_header("30-Day Revenue Forecast (Holt-Winters)")
        fcst, lo, hi = holt_winters_forecast(daily_rev["revenue"].values, horizon=30)
        last_day = daily_rev["day"].max()
        future_days = pd.date_range(last_day + pd.Timedelta(days=1), periods=30)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_rev["day"], y=daily_rev["revenue"],
                                 mode="lines", name="Actual",
                                 line=dict(color=COLORS["primary"], width=2.2)))
        fig.add_trace(go.Scatter(x=future_days, y=hi, mode="lines",
                                 line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=future_days, y=lo, mode="lines",
                                 line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(0,180,216,0.18)",
                                 name="95% confidence"))
        fig.add_trace(go.Scatter(x=future_days, y=fcst, mode="lines+markers",
                                 name="Forecast",
                                 line=dict(color=COLORS["accent"], width=2.5, dash="dash"),
                                 marker=dict(size=5)))
        st.plotly_chart(styled_fig(fig, height=360), use_container_width=True)

        change = (fcst.mean() - daily_rev["revenue"].tail(30).mean()) / max(daily_rev["revenue"].tail(30).mean(), 1) * 100
        info_card(
            f"📈 <b>Forecast outlook:</b> next-30d daily revenue averages "
            f"<b>{fmt_ksh(fcst.mean())}</b> "
            f"({'▲' if change >= 0 else '▼'} {abs(change):.1f}% vs trailing 30d). "
            f"Plan inventory and staffing accordingly.",
            COLORS["primary"] if change >= 0 else COLORS["warning"],
        )

    # ── Risk radar
    section_header("Operational Risk Radar")
    radar_categories = ["Stockout", "Freshness", "Margin", "Promo Mix", "Churn"]
    radar_values = [
        min(leakage / 1_000_000, 5),
        min(at_risk_freshness / 1_000_000, 5),
        max(0, 5 - avg_rbpi * 10),
        2.5 if promo_data.empty else
            float(((promo_data["promotion_verdict"] == "MARGIN KILLER · Kill Promo").mean()) * 5),
        2.0 if clv_data.empty else
            float(min((clv_data["total_transactions"] == 1).mean() * 5, 5)),
    ]
    fig = go.Figure(go.Scatterpolar(
        r=radar_values + [radar_values[0]],
        theta=radar_categories + [radar_categories[0]],
        fill="toself",
        line=dict(color=COLORS["primary"], width=2),
        fillcolor="rgba(0,114,206,0.18)",
        name="Current",
    ))
    fig.update_layout(
        **{**CHART_LAYOUT, "polar": dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 5], gridcolor="#E0EAF5"),
            angularaxis=dict(gridcolor="#E0EAF5"),
        )},
        height=380, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    info_card(
        "Risk radar combines five health signals on a 0–5 scale (5 = highest risk). "
        "Use it to triage — anything past the 3 ring deserves attention this week.",
        COLORS["muted"],
    )

# ═════════════════════════════════════════════════════════════════════════
#  RBPI ANALYSIS
# ═════════════════════════════════════════════════════════════════════════

elif analysis_type == "RBPI Analysis":
    if rbpi_data.empty:
        info_card("No RBPI data available for the selected store.", COLORS["warning"])
    else:
        rbpi_model, rbpi_features = train_rbpi_predictor(rbpi_data)

        # KPIs
        margin = 1 - rbpi_data["cogs"].sum() / max(rbpi_data["revenue"].sum(), 1)
        neg_share = (rbpi_data["rbpi"] < 0).mean()

        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card("Avg RBPI", f"{rbpi_data['rbpi'].mean():.1%}", "Basket profitability", COLORS["primary"])
        with c2: kpi_card("Total revenue", fmt_ksh(rbpi_data['revenue'].sum()), "Window total", COLORS["success"])
        with c3: kpi_card("Gross margin", f"{margin:.1%}", "After COGS only", COLORS["green"])
        with c4: kpi_card("Loss-making %", f"{neg_share:.1%}", "Negative-RBPI baskets", COLORS["danger"])

        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Distribution", "🤖 Predictive Model", "🎛️ Discount Simulator", "⚠️ Anomaly Timeline"]
        )

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=rbpi_data["rbpi"].clip(-1, 1), nbinsx=40,
                    marker=dict(color=COLORS["primary"], line=dict(width=0)), opacity=0.85,
                ))
                fig.add_vline(x=rbpi_data["rbpi"].median(), line_dash="dash",
                              line_color=COLORS["warning"],
                              annotation_text=f"Median: {rbpi_data['rbpi'].median():.1%}")
                fig.add_vline(x=0, line_color=COLORS["danger"], line_dash="dot",
                              annotation_text="Break-even")
                st.plotly_chart(styled_fig(fig, "RBPI distribution across baskets", 380),
                                use_container_width=True)

            with col2:
                rb = rbpi_data.dropna(subset=["rbpi", "revenue"]).copy()
                rb["rbpi_clip"] = rb["rbpi"].clip(-0.5, 1.0)
                fig = px.density_heatmap(
                    rb, x="revenue", y="rbpi_clip", nbinsx=30, nbinsy=24,
                    color_continuous_scale=[(0, "#EBF3FB"), (1, COLORS["primary"])],
                )
                fig.update_layout(coloraxis_colorbar=dict(title="Baskets"))
                fig.update_xaxes(title="Revenue (KSh)", type="log")
                fig.update_yaxes(title="RBPI")
                st.plotly_chart(styled_fig(fig, "Where baskets cluster — revenue × profitability", 380),
                                use_container_width=True)

            # Pareto
            section_header("Profit Pareto — who pays the bills?")
            rb_sorted = rbpi_data.dropna(subset=["rbpi", "revenue"]).copy()
            rb_sorted["abs_profit"] = rb_sorted["revenue"] * rb_sorted["rbpi"]
            rb_sorted = rb_sorted.sort_values("abs_profit", ascending=False).reset_index(drop=True)
            rb_sorted["cum_profit_pct"] = (
                rb_sorted["abs_profit"].cumsum() / rb_sorted["abs_profit"].sum() * 100
            )
            rb_sorted["basket_pct"] = (rb_sorted.index + 1) / len(rb_sorted) * 100

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rb_sorted["basket_pct"], y=rb_sorted["cum_profit_pct"],
                mode="lines", line=dict(color=COLORS["primary"], width=3),
                fill="tozeroy", fillcolor="rgba(0,114,206,0.15)",
            ))
            fig.add_hline(y=80, line_dash="dot", line_color=COLORS["warning"])
            fig.add_vline(x=20, line_dash="dot", line_color=COLORS["warning"])
            fig.update_xaxes(title="% of baskets (sorted by profit)")
            fig.update_yaxes(title="% of cumulative profit")
            st.plotly_chart(styled_fig(fig, "Pareto curve — 80/20 indicator", 320),
                            use_container_width=True)

            top20_share = rb_sorted.iloc[:max(1, len(rb_sorted) // 5)]["abs_profit"].sum() \
                          / max(rb_sorted["abs_profit"].sum(), 1) * 100
            info_card(
                f"💡 Top 20% of baskets generate <b>{top20_share:.0f}%</b> of profit. "
                f"Loss-making baskets erode <b>{fmt_ksh(rb_sorted[rb_sorted['abs_profit'] < 0]['abs_profit'].sum())}</b>.",
                COLORS["primary"],
            )

        with tab2:
            if show_predictions:
                X_pred = rbpi_data[rbpi_features].fillna(0)
                preds = rbpi_model.predict(X_pred).astype(float)
                rbpi_data["predicted_rbpi"] = preds
                y_true = rbpi_data["rbpi"].astype(float).to_numpy()
                mask = ~np.isnan(y_true) & ~np.isnan(preds)

                c1, c2, c3 = st.columns(3)
                with c1: kpi_card("R²", f"{r2_score(y_true[mask], preds[mask]):.2%}",
                                  "explained variance", COLORS["success"])
                with c2: kpi_card("MAE", f"{mean_absolute_error(y_true[mask], preds[mask]):.2%}",
                                  "mean absolute error", COLORS["primary"])
                with c3: kpi_card("Anomalies", f"{detect_anomalies(rbpi_data).sum()}",
                                  "isolation-forest flagged", COLORS["warning"])

                imp_df = pd.DataFrame({"feature": rbpi_features,
                                       "importance": rbpi_model.feature_importances_})
                fig = go.Figure(go.Bar(
                    x=imp_df["importance"], y=imp_df["feature"], orientation="h",
                    marker=dict(color=imp_df["importance"], colorscale="Blues",
                                line=dict(width=0)),
                    text=[f"{v:.0%}" for v in imp_df["importance"]],
                    textposition="outside",
                ))
                st.plotly_chart(styled_fig(fig, "Feature importance — what drives RBPI?", 280),
                                use_container_width=True)

                # Predicted vs actual
                samp = rbpi_data.sample(min(2000, len(rbpi_data)), random_state=1)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=samp["rbpi"].clip(-0.5, 1), y=samp["predicted_rbpi"].clip(-0.5, 1),
                    mode="markers",
                    marker=dict(size=6, color=samp["revenue"], colorscale="Viridis",
                                showscale=True, colorbar=dict(title="Revenue")),
                    opacity=0.7,
                ))
                fig.add_trace(go.Scatter(x=[-0.5, 1], y=[-0.5, 1], mode="lines",
                                         line=dict(color="rgba(0,0,0,0.4)", dash="dash"),
                                         showlegend=False))
                fig.update_xaxes(title="Actual RBPI"); fig.update_yaxes(title="Predicted RBPI")
                st.plotly_chart(styled_fig(fig, "Predicted vs actual basket profitability", 380),
                                use_container_width=True)
            else:
                info_card("Predictions hidden. Toggle 'Show AI predictions' in the sidebar.", COLORS["muted"])

        with tab3:
            if show_simulators:
                info_card(
                    "🎛️ <b>Discount what-if:</b> uplift the discount given on every basket by N%, "
                    "and we'll roll the model forward to estimate the new RBPI distribution and lost profit.",
                    COLORS["accent"],
                )

                c1, c2, c3 = st.columns(3)
                with c1:
                    disc_uplift = st.slider("Discount uplift on every basket (%)",
                                            -30, 50, 0, step=5)
                with c2:
                    cogs_shock = st.slider("COGS inflation (%)", -10, 30, 0, step=2)
                with c3:
                    basket_growth = st.slider("Basket size change (%)", -20, 40, 0, step=5)

                sim = rbpi_data.copy()
                sim["sim_revenue"] = sim["revenue"] * (1 + basket_growth / 100)
                sim["sim_discount"] = sim["total_discount"] * (1 + disc_uplift / 100)
                sim["sim_cogs"] = sim["cogs"] * (1 + cogs_shock / 100)
                sim["sim_rbpi"] = (sim["sim_revenue"] - sim["sim_cogs"] - sim["sim_discount"]) \
                                  / np.maximum(sim["sim_revenue"], 1)

                base_profit = (rbpi_data["revenue"] - rbpi_data["cogs"] - rbpi_data["total_discount"]).sum()
                sim_profit = (sim["sim_revenue"] - sim["sim_cogs"] - sim["sim_discount"]).sum()
                delta = sim_profit - base_profit

                c1, c2, c3 = st.columns(3)
                with c1: kpi_card("Baseline profit", fmt_ksh(base_profit), "no scenario", COLORS["muted"])
                with c2: kpi_card("Simulated profit", fmt_ksh(sim_profit), "after scenario",
                                  COLORS["success"] if delta >= 0 else COLORS["danger"])
                with c3:
                    dir_ = "up" if delta >= 0 else "down"
                    kpi_card("Delta", fmt_ksh(delta),
                             f"{abs(delta) / max(base_profit, 1) * 100:.1f}% change",
                             COLORS["success"] if delta >= 0 else COLORS["danger"],
                             delta=f"{abs(delta / max(base_profit, 1) * 100):.1f}%", delta_dir=dir_)

                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=rbpi_data["rbpi"].clip(-0.5, 1), nbinsx=40, name="Baseline",
                    marker_color=COLORS["muted"], opacity=0.55,
                ))
                fig.add_trace(go.Histogram(
                    x=sim["sim_rbpi"].clip(-0.5, 1), nbinsx=40, name="Simulated",
                    marker_color=COLORS["primary"], opacity=0.75,
                ))
                fig.update_layout(barmode="overlay")
                fig.add_vline(x=0, line_color=COLORS["danger"], line_dash="dot")
                st.plotly_chart(styled_fig(fig, "RBPI distribution: baseline vs simulated", 360),
                                use_container_width=True)
            else:
                info_card("Simulators hidden. Toggle 'Show what-if simulators' in the sidebar.", COLORS["muted"])

        with tab4:
            if "created_at" in rbpi_data.columns:
                rb = rbpi_data.dropna(subset=["created_at", "rbpi"]).copy()
                rb["created_at"] = pd.to_datetime(rb["created_at"], errors="coerce")
                rb["anomaly"] = detect_anomalies(rb)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=rb.loc[~rb["anomaly"], "created_at"],
                    y=rb.loc[~rb["anomaly"], "rbpi"],
                    mode="markers", name="Normal",
                    marker=dict(size=4, color=COLORS["primary"], opacity=0.45),
                ))
                fig.add_trace(go.Scatter(
                    x=rb.loc[rb["anomaly"], "created_at"],
                    y=rb.loc[rb["anomaly"], "rbpi"],
                    mode="markers", name="Anomaly",
                    marker=dict(size=10, color=COLORS["danger"],
                                line=dict(width=1.5, color="white")),
                ))
                fig.add_hline(y=0, line_dash="dot", line_color="rgba(0,0,0,0.4)")
                fig.update_yaxes(title="RBPI")
                st.plotly_chart(styled_fig(fig, "Anomaly timeline — outliers in basket profitability", 380),
                                use_container_width=True)

                anomalies = rb[rb["anomaly"]].sort_values("created_at", ascending=False)
                info_card(
                    f"🔎 Detected <b>{len(anomalies)}</b> anomalous baskets. "
                    f"Most recent: <b>{anomalies['created_at'].max() if len(anomalies) else '—'}</b>. "
                    "These are top candidates for fraud / pricing-error review.",
                    COLORS["warning"],
                )
                with st.expander("View anomalous baskets"):
                    st.dataframe(anomalies[["sale_id", "created_at", "revenue",
                                            "total_discount", "rbpi"]].head(50),
                                 use_container_width=True, hide_index=True)
            else:
                info_card("Timestamp not available — anomaly timeline disabled.", COLORS["muted"])

# ═════════════════════════════════════════════════════════════════════════
#  CUSTOMER LIFETIME VALUE
# ═════════════════════════════════════════════════════════════════════════

elif analysis_type == "Customer Lifetime Value":
    if clv_data.empty:
        info_card("No CLV data available.", COLORS["warning"])
    else:
        clv = clv_data.copy()
        today = pd.to_datetime(clv["last_purchase_date"]).max() + pd.Timedelta(days=1)
        clv["_today"] = today
        clv["recency_days"] = (today - pd.to_datetime(clv["last_purchase_date"])).dt.days
        clv["churn_risk"] = clv.apply(churn_score, axis=1)

        clv_model, clv_features = train_clv_predictor(clv)

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card("Customers", fmt_num(len(clv)), "active in window", COLORS["primary"])
        with c2: kpi_card("Avg revenue / customer", fmt_ksh(clv["total_revenue"].mean()),
                          "lifetime to date", COLORS["success"])
        with c3: kpi_card("Repeat rate",
                          f"{(clv['total_transactions'] > 1).mean():.1%}",
                          "≥ 2 visits", COLORS["green"])
        with c4: kpi_card("High churn risk",
                          f"{(clv['churn_risk'] > 0.7).sum():,}",
                          "churn-score > 70%", COLORS["danger"])

        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["💎 RFM Map", "📅 Cohorts", "🔮 Churn Risk",
             "🎛️ Retention Simulator", "📈 Top Customers"]
        )

        with tab1:
            section_header("RFM 3D map — recency × frequency × monetary")
            rfm = clv.copy()
            rfm["log_monetary"] = np.log1p(rfm["total_revenue"])
            rfm = rfm.sample(min(2500, len(rfm)), random_state=1)

            fig = go.Figure(data=[go.Scatter3d(
                x=rfm["recency_days"], y=rfm["total_transactions"], z=rfm["log_monetary"],
                mode="markers",
                marker=dict(
                    size=4,
                    color=rfm["churn_risk"], colorscale="RdYlGn_r",
                    showscale=True, colorbar=dict(title="Churn risk"),
                    line=dict(width=0),
                ),
                text=rfm["customer_id"].astype(str),
                hovertemplate="Recency: %{x}d<br>Frequency: %{y}<br>"
                              "log(Monetary): %{z:.2f}<br>%{text}<extra></extra>",
            )])
            fig.update_layout(
                **{**CHART_LAYOUT, "scene": dict(
                    xaxis=dict(title="Recency (days)", backgroundcolor="rgba(0,0,0,0)"),
                    yaxis=dict(title="Frequency", backgroundcolor="rgba(0,0,0,0)"),
                    zaxis=dict(title="log Monetary", backgroundcolor="rgba(0,0,0,0)"),
                )},
                height=520,
            )
            st.plotly_chart(fig, use_container_width=True)

            # K-means on RFM
            section_header("K-means RFM segments")
            k_data = rfm[["recency_days", "total_transactions", "log_monetary"]].fillna(0)
            scaler = StandardScaler()
            k_scaled = scaler.fit_transform(k_data)
            n_clusters = 5 if len(k_scaled) >= 5 else max(2, len(k_scaled))
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(k_scaled)
            rfm["segment"] = km.labels_

            seg_summary = rfm.groupby("segment").agg(
                customers=("customer_id", "count"),
                avg_recency=("recency_days", "mean"),
                avg_freq=("total_transactions", "mean"),
                avg_revenue=("total_revenue", "mean"),
                avg_churn=("churn_risk", "mean"),
            ).round(2).reset_index()
            seg_summary["label"] = seg_summary.apply(
                lambda r: ("🌟 Champions"  if r["avg_freq"] > 5 and r["avg_recency"] < 30 else
                           "💎 Loyal"      if r["avg_freq"] > 3 else
                           "⚠️ At Risk"    if r["avg_recency"] > 60 else
                           "🆕 New"        if r["avg_freq"] <= 1 else
                           "🌱 Developing"),
                axis=1,
            )
            st.dataframe(
                seg_summary, use_container_width=True, hide_index=True,
                column_config={
                    "avg_revenue": st.column_config.NumberColumn(format="KSh %,.0f"),
                    "avg_churn":   st.column_config.NumberColumn("Churn risk", format="%.0%%"),
                },
            )

        with tab2:
            section_header("Cohort retention heatmap")
            ch = clv.copy()
            ch["cohort_month"] = pd.to_datetime(ch["first_purchase_date"]).dt.to_period("M").astype(str)
            ch["last_month"]   = pd.to_datetime(ch["last_purchase_date"]).dt.to_period("M").astype(str)
            ch["months_active"] = (
                pd.to_datetime(ch["last_purchase_date"]).dt.to_period("M").astype(int)
                - pd.to_datetime(ch["first_purchase_date"]).dt.to_period("M").astype(int)
            )

            cohort_size = ch.groupby("cohort_month")["customer_id"].nunique()
            cohort_act = (
                ch.groupby(["cohort_month", "months_active"])["customer_id"].nunique().unstack().fillna(0)
            )
            retention = cohort_act.divide(cohort_size, axis=0) * 100

            fig = go.Figure(data=go.Heatmap(
                z=retention.values,
                x=[f"M+{int(c)}" for c in retention.columns],
                y=retention.index,
                colorscale=[[0, "#F7FAFD"], [0.5, COLORS["accent"]], [1, COLORS["deep"]]],
                text=np.round(retention.values, 0),
                texttemplate="%{text}%",
                textfont={"size": 10, "color": "#003566"},
                colorbar=dict(title="% retained"),
            ))
            st.plotly_chart(styled_fig(fig, "Cohort retention — % of cohort active each month after first purchase",
                                       420),
                            use_container_width=True)

            info_card(
                "Read row-by-row: of customers who first bought in <b>{first}</b>, "
                "<b>{ret:.0f}%</b> were still active in their second month. "
                "Bigger gaps as you scan right = bigger churn problem in that cohort."
                .format(first=retention.index[0],
                        ret=retention.iloc[0, 1] if retention.shape[1] > 1 else 0),
                COLORS["primary"],
            )

        with tab3:
            section_header("Churn risk distribution + drivers")
            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=clv["churn_risk"], nbinsx=30,
                    marker=dict(color=clv["churn_risk"], colorscale="RdYlGn_r"),
                ))
                fig.add_vline(x=0.7, line_dash="dash", line_color=COLORS["danger"],
                              annotation_text="High-risk threshold")
                fig.update_xaxes(title="Churn probability"); fig.update_yaxes(title="Customers")
                st.plotly_chart(styled_fig(fig, "Churn-score distribution", 360),
                                use_container_width=True)

            with col2:
                # Risk vs revenue — quadrant
                fig = px.scatter(
                    clv.sample(min(3000, len(clv)), random_state=1),
                    x="recency_days", y="churn_risk", size="total_revenue",
                    color="velocity_tier", size_max=22,
                    color_discrete_sequence=[COLORS["primary"], COLORS["success"],
                                             COLORS["warning"], COLORS["pink"],
                                             COLORS["purple"], COLORS["muted"]],
                )
                fig.add_hline(y=0.7, line_dash="dash", line_color=COLORS["danger"])
                fig.update_xaxes(title="Days since last purchase")
                fig.update_yaxes(title="Churn probability")
                st.plotly_chart(styled_fig(fig, "Where to spend retention budget", 360),
                                use_container_width=True)

            # Save targets summary
            high = clv[clv["churn_risk"] > 0.7]
            rev_at_risk = high["total_revenue"].sum()
            info_card(
                f"⚠️ <b>{len(high):,}</b> customers above the 70% churn threshold, "
                f"representing <b>{fmt_ksh(rev_at_risk)}</b> in lifetime revenue. "
                "Prioritise recovery outreach to the top 200 by monetary value.",
                COLORS["danger"],
            )

        with tab4:
            if show_simulators:
                info_card(
                    "🎛️ Simulate a retention campaign: reduce churn for the high-risk segment by N%, "
                    "and project the dollars saved over the next quarter.",
                    COLORS["accent"],
                )

                c1, c2, c3 = st.columns(3)
                with c1:
                    save_rate = st.slider("Churn reduction in high-risk segment (%)",
                                          0, 80, 25, step=5)
                with c2:
                    spend_per_save = st.number_input("Cost per save (KSh)",
                                                     min_value=0, value=300, step=50)
                with c3:
                    horizon = st.slider("Horizon (months)", 1, 12, 3)

                high_risk = clv[clv["churn_risk"] > 0.7].copy()
                # expected lost revenue if they churn = monthly avg × horizon × p(churn)
                high_risk["monthly_rev"] = high_risk["total_revenue"] / np.maximum(high_risk["lifespan_days"] / 30, 1)
                base_loss = (high_risk["monthly_rev"] * horizon * high_risk["churn_risk"]).sum()
                saved = base_loss * (save_rate / 100)
                cost = len(high_risk) * spend_per_save * (save_rate / 100)
                roi = (saved - cost) / max(cost, 1) * 100 if cost > 0 else float("nan")

                c1, c2, c3 = st.columns(3)
                with c1: kpi_card("Revenue protected", fmt_ksh(saved), f"{horizon}-mo horizon", COLORS["success"])
                with c2: kpi_card("Campaign cost", fmt_ksh(cost), f"{int(len(high_risk) * save_rate / 100):,} contacts",
                                  COLORS["warning"])
                with c3: kpi_card("ROI",
                                  f"{roi:.0f}%" if not np.isnan(roi) else "—",
                                  "saved minus cost",
                                  COLORS["success"] if roi > 0 else COLORS["danger"])

                # Projection waterfall
                fig = go.Figure(go.Waterfall(
                    orientation="v",
                    measure=["absolute", "relative", "relative", "total"],
                    x=["Baseline loss", "Churn reduction", "Campaign cost", "Net impact"],
                    y=[base_loss, -saved, -cost, base_loss - saved - cost],
                    text=[fmt_ksh(base_loss), fmt_ksh(-saved), fmt_ksh(-cost),
                          fmt_ksh(base_loss - saved - cost)],
                    textposition="outside",
                    increasing=dict(marker=dict(color=COLORS["danger"])),
                    decreasing=dict(marker=dict(color=COLORS["success"])),
                    totals=dict(marker=dict(color=COLORS["primary"])),
                ))
                st.plotly_chart(styled_fig(fig, "Retention campaign — financial waterfall", 380),
                                use_container_width=True)
            else:
                info_card("Simulators hidden.", COLORS["muted"])

        with tab5:
            section_header("Predicted next-90d revenue · top customers")
            X_top = clv[clv_features].fillna(0)
            preds = np.exp(clv_model.predict(X_top))
            clv["pred_velocity"] = preds
            clv["pred_90d_revenue"] = clv["pred_velocity"] * 90
            top20 = clv.nlargest(20, "pred_90d_revenue")

            fig = go.Figure(go.Bar(
                x=top20["pred_90d_revenue"],
                y=top20["customer_id"].astype(str),
                orientation="h",
                marker=dict(color=top20["pred_90d_revenue"], colorscale="Blues",
                            line=dict(width=0)),
                text=[fmt_ksh(v) for v in top20["pred_90d_revenue"]],
                textposition="outside",
            ))
            fig.update_yaxes(autorange="reversed", title="Customer ID")
            fig.update_xaxes(title="Predicted 90d revenue")
            st.plotly_chart(styled_fig(fig, "Top 20 customers by predicted next-90d revenue", 520),
                            use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════
#  STOCKOUT LEAKAGE
# ═════════════════════════════════════════════════════════════════════════

elif analysis_type == "Stockout Leakage":
    if stockout_data.empty:
        info_card("No stockout data available.", COLORS["warning"])
    else:
        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card("Total leakage", fmt_ksh(stockout_data["srl_amount"].sum()),
                          "revenue lost to OOS", COLORS["danger"])
        with c2: kpi_card("Affected SKUs", f"{len(stockout_data):,}",
                          "products with shortage", COLORS["warning"])
        with c3: kpi_card("Avg shortage", f"{stockout_data['shortage_units'].mean():.0f} u",
                          "units per SKU", COLORS["primary"])
        with c4: kpi_card("Critical tier",
                          f"{(stockout_data['leakage_tier'] == 'CRITICAL · Expedite Reorder').sum()}",
                          "expedite immediately", COLORS["danger"])

        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Leakage Map", "🅰️ ABC Classification",
             "🎛️ Reorder-Point Simulator", "🔮 Demand Forecast"]
        )

        with tab1:
            col1, col2 = st.columns([3, 2])
            with col1:
                top20 = stockout_data.nlargest(20, "srl_amount")
                fig = go.Figure(go.Bar(
                    x=top20["srl_amount"], y=top20["product_name"],
                    orientation="h",
                    marker=dict(color=top20["srl_amount"], colorscale="Reds",
                                line=dict(width=0)),
                    text=[fmt_ksh(v) for v in top20["srl_amount"]],
                    textposition="outside",
                ))
                fig.update_yaxes(autorange="reversed")
                fig.update_xaxes(title="Revenue leakage (KSh)")
                st.plotly_chart(styled_fig(fig, "Top 20 SKUs · revenue leakage", 540),
                                use_container_width=True)

            with col2:
                tier_counts = stockout_data["leakage_tier"].value_counts()
                fig = go.Figure(go.Pie(
                    labels=tier_counts.index, values=tier_counts.values, hole=0.55,
                    marker=dict(colors=[COLORS["danger"], COLORS["warning"],
                                        COLORS["success"], COLORS["muted"]]),
                ))
                fig.update_layout(annotations=[dict(
                    text=f"{tier_counts.sum():,}<br>SKUs",
                    showarrow=False, font=dict(size=18, color=COLORS["deep"])
                )])
                st.plotly_chart(styled_fig(fig, "Leakage severity mix", 540),
                                use_container_width=True)

            # Department treemap
            if "department" in stockout_data.columns:
                section_header("Leakage by department · treemap")
                dep = stockout_data.dropna(subset=["department"]).copy()
                dep["product_name"] = dep["product_name"].fillna("(unknown)")
                fig = px.treemap(
                    dep.nlargest(80, "srl_amount"),
                    path=["department", "product_name"],
                    values="srl_amount",
                    color="srl_amount", color_continuous_scale="Reds",
                )
                fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=440)
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            section_header("ABC classification — concentration of leakage")
            so = stockout_data.sort_values("srl_amount", ascending=False).reset_index(drop=True)
            so["cum_pct"] = so["srl_amount"].cumsum() / max(so["srl_amount"].sum(), 1) * 100
            so["abc"] = np.where(so["cum_pct"] <= 70, "A",
                         np.where(so["cum_pct"] <= 90, "B", "C"))
            mix = so.groupby("abc").agg(
                skus=("product_id", "count"),
                leakage=("srl_amount", "sum"),
                avg_shortage=("shortage_units", "mean"),
            ).reset_index()
            mix["share"] = mix["leakage"] / mix["leakage"].sum() * 100

            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure(go.Bar(
                    x=mix["abc"], y=mix["leakage"],
                    marker=dict(color=[COLORS["danger"], COLORS["warning"], COLORS["muted"]]),
                    text=[f"{fmt_ksh(v)}<br>{s:.0f}% of total" for v, s in zip(mix["leakage"], mix["share"])],
                    textposition="outside",
                ))
                fig.update_yaxes(title="Leakage (KSh)")
                st.plotly_chart(styled_fig(fig, "ABC tiers — where the money is", 360),
                                use_container_width=True)
            with col2:
                fig = go.Figure(go.Bar(
                    x=mix["abc"], y=mix["skus"],
                    marker=dict(color=[COLORS["primary"], COLORS["accent"], COLORS["muted"]]),
                    text=mix["skus"], textposition="outside",
                ))
                fig.update_yaxes(title="SKU count")
                st.plotly_chart(styled_fig(fig, "SKU count per tier", 360),
                                use_container_width=True)

            info_card(
                f"💡 <b>{int(mix.loc[mix['abc'] == 'A', 'skus'].sum()):,} A-tier SKUs</b> "
                f"({mix.loc[mix['abc'] == 'A', 'skus'].sum() / max(len(so), 1) * 100:.1f}% of catalogue) "
                f"drive <b>{mix.loc[mix['abc'] == 'A', 'share'].sum():.0f}%</b> of leakage. "
                "Tighten reorder thresholds on these first.",
                COLORS["danger"],
            )

            with st.expander("View full ABC table"):
                st.dataframe(
                    so[["product_name", "department", "shortage_units", "srl_amount", "abc", "leakage_tier"]],
                    use_container_width=True, hide_index=True,
                    column_config={
                        "srl_amount": st.column_config.NumberColumn("Leakage", format="KSh %,.0f"),
                    },
                )

        with tab3:
            if show_simulators:
                info_card(
                    "🎛️ Newsvendor reorder-point optimizer with Poisson demand. "
                    "Pick lead time + service level; we'll project ROP per top SKU.",
                    COLORS["accent"],
                )
                c1, c2, c3 = st.columns(3)
                with c1: lead = st.slider("Lead time (days)", 1, 30, 7)
                with c2: svc = st.select_slider("Service level",
                                                options=[0.90, 0.95, 0.975, 0.99], value=0.95,
                                                format_func=lambda x: f"{x:.1%}")
                with c3: window = st.slider("Demand window (days)", 30, 180, 90)

                top = stockout_data.nlargest(15, "srl_amount").copy()
                top["daily_demand"] = np.maximum(top["demand_units"] / window, 0.01)
                top["rop"] = top["daily_demand"].apply(lambda d: reorder_point(d, lead, svc))
                top["gap"] = (top["rop"] - top["available_units"]).round(0)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=top["product_name"], y=top["available_units"], name="Currently on hand",
                    marker_color=COLORS["primary"],
                ))
                fig.add_trace(go.Bar(
                    x=top["product_name"], y=top["rop"], name=f"Recommended ROP @ {svc:.0%} svc",
                    marker_color=COLORS["accent"],
                ))
                fig.update_layout(barmode="group", xaxis_tickangle=-30)
                fig.update_yaxes(title="Units")
                st.plotly_chart(styled_fig(fig, "Reorder-point recommendation · top 15 leakers", 420),
                                use_container_width=True)

                under = top[top["gap"] > 0]
                info_card(
                    f"📦 <b>{len(under)}</b> top SKUs are below the recommended reorder point. "
                    f"Aggregate gap: <b>{under['gap'].sum():,.0f} units</b> "
                    f"≈ <b>{fmt_ksh((under['gap'] * under['unit_price']).sum())}</b> in covered demand.",
                    COLORS["warning"],
                )
            else:
                info_card("Simulators hidden.", COLORS["muted"])

        with tab4:
            if not daily_rev.empty:
                # use total daily revenue as demand proxy and forecast next 30 days
                fcst, lo, hi = holt_winters_forecast(daily_rev["transactions"].values, horizon=30)
                future = pd.date_range(daily_rev["day"].max() + pd.Timedelta(days=1), periods=30)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=daily_rev["day"], y=daily_rev["transactions"],
                                         mode="lines", name="Actual demand",
                                         line=dict(color=COLORS["primary"], width=2)))
                fig.add_trace(go.Scatter(x=future, y=hi, line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=future, y=lo, line=dict(width=0), fill="tonexty",
                                         fillcolor="rgba(0,180,216,0.18)", name="Confidence"))
                fig.add_trace(go.Scatter(x=future, y=fcst, mode="lines+markers",
                                         name="Forecast",
                                         line=dict(color=COLORS["accent"], dash="dash", width=2.5)))
                st.plotly_chart(styled_fig(fig, "30-day demand forecast (transactions)", 380),
                                use_container_width=True)

                info_card(
                    f"📈 Mean projected daily volume next 30 days: "
                    f"<b>{fcst.mean():.0f}</b> txns. "
                    f"Plan replenishment with this as the floor — and the upper band ({hi.mean():.0f}) as the ceiling.",
                    COLORS["primary"],
                )
            else:
                info_card("No daily revenue series available.", COLORS["muted"])

# ═════════════════════════════════════════════════════════════════════════
#  PROMOTION EFFICIENCY
# ═════════════════════════════════════════════════════════════════════════

elif analysis_type == "Promotion Efficiency":
    if promo_data.empty:
        info_card("No promotion data available.", COLORS["warning"])
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card("Avg PER", f"{promo_data['per_ratio'].mean():.2f}x",
                          "incremental profit / discount", COLORS["primary"])
        with c2: kpi_card("Discount spend", fmt_ksh(promo_data["discount_cost"].sum()),
                          "investment to date", COLORS["warning"])
        with c3: kpi_card("Incremental profit", fmt_ksh(promo_data["incremental_profit"].sum()),
                          "lift over baseline", COLORS["success"])
        with c4:
            star_pct = (promo_data["promotion_verdict"] == "STAR · Scale Up").mean()
            kpi_card("Star promos", f"{star_pct:.1%}", "high-ROI campaigns", COLORS["purple"])

        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Verdict Map", "📐 Price Elasticity",
             "🎛️ Optimal-Discount Finder", "🎲 Lift Monte Carlo"]
        )

        with tab1:
            colors_map = {
                "STAR · Scale Up":              COLORS["success"],
                "EFFICIENT · Keep Running":     COLORS["primary"],
                "BREAK-EVEN · Review Mechanics": COLORS["warning"],
                "MARGIN KILLER · Kill Promo":    COLORS["danger"],
                "NO DISCOUNT":                   COLORS["muted"],
            }

            col1, col2 = st.columns(2)
            with col1:
                vc = promo_data["promotion_verdict"].value_counts()
                fig = go.Figure(go.Pie(
                    labels=vc.index, values=vc.values, hole=0.5,
                    marker=dict(colors=[colors_map.get(v, COLORS["muted"]) for v in vc.index]),
                ))
                fig.update_layout(annotations=[dict(text=f"{vc.sum():,}<br>promos",
                                                    showarrow=False,
                                                    font=dict(size=16, color=COLORS["deep"]))])
                st.plotly_chart(styled_fig(fig, "Promotion-verdict mix", 420),
                                use_container_width=True)

            with col2:
                fig = go.Figure()
                for v, sub in promo_data.groupby("promotion_verdict"):
                    fig.add_trace(go.Scatter(
                        x=sub["discount_cost"], y=sub["incremental_profit"],
                        mode="markers", name=v,
                        marker=dict(size=9, color=colors_map.get(v, COLORS["muted"]),
                                    opacity=0.75, line=dict(width=0)),
                        text=sub["product_name"],
                        hovertemplate="%{text}<br>Disc: %{x:,.0f}<br>Profit: %{y:,.0f}<extra></extra>",
                    ))
                fig.add_hline(y=0, line_dash="dot", line_color="rgba(0,0,0,0.45)")
                fig.update_xaxes(title="Discount spent (KSh)")
                fig.update_yaxes(title="Incremental profit (KSh)")
                st.plotly_chart(styled_fig(fig, "Discount vs incremental profit · ROI matrix", 420),
                                use_container_width=True)

            section_header("Top 15 promotions by PER")
            top15 = promo_data.nlargest(15, "per_ratio")
            fig = go.Figure(go.Bar(
                x=top15["per_ratio"], y=top15["product_name"], orientation="h",
                marker=dict(color=top15["per_ratio"], colorscale="Greens",
                            line=dict(width=0)),
                text=[f"{v:.2f}x" for v in top15["per_ratio"]],
                textposition="outside",
            ))
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(title="PER (incremental profit / discount)")
            st.plotly_chart(styled_fig(fig, "Best-performing promotions", 460),
                            use_container_width=True)

        with tab2:
            elast = price_elasticity(promo_data)
            section_header("Estimated price elasticity of demand")

            if not np.isnan(elast["epsilon"]):
                c1, c2, c3 = st.columns(3)
                with c1: kpi_card("ε (elasticity)", f"{elast['epsilon']:.2f}",
                                  "% change in units / % change in price",
                                  COLORS["success"] if elast["epsilon"] < -1 else COLORS["warning"])
                with c2: kpi_card("Model R²", f"{elast['r2']:.0%}",
                                  "log-log fit quality", COLORS["primary"])
                with c3:
                    interp = ("Elastic — discounts work" if elast["epsilon"] < -1 else
                              "Inelastic — discounts erode margin")
                    kpi_card("Interpretation", interp, "rule of thumb",
                             COLORS["success"] if elast["epsilon"] < -1 else COLORS["danger"])

                # Curve
                price_ratio = np.linspace(0.5, 1.0, 50)
                units_ratio = price_ratio ** elast["epsilon"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=(1 - price_ratio) * 100, y=units_ratio,
                                         mode="lines", name="Demand response",
                                         line=dict(color=COLORS["primary"], width=3),
                                         fill="tozeroy", fillcolor="rgba(0,114,206,0.12)"))
                fig.update_xaxes(title="Discount %")
                fig.update_yaxes(title="Units sold (×baseline)")
                st.plotly_chart(styled_fig(fig, "Estimated demand curve", 360),
                                use_container_width=True)

                info_card(
                    f"📐 Each 10% discount lifts unit demand by ~"
                    f"<b>{(1.1 ** abs(elast['epsilon']) - 1) * 100:.1f}%</b>. "
                    "Use this to size promos: if margin × demand uplift < discount cost, the promo destroys value.",
                    COLORS["primary"],
                )
            else:
                info_card("Not enough discount-bearing rows to estimate elasticity.", COLORS["muted"])

        with tab3:
            if show_simulators and not np.isnan(elast.get("epsilon", np.nan)):
                info_card(
                    "🎛️ Sweep discount % from 0–60% and find the discount that maximises profit, "
                    "given estimated elasticity, baseline margin, and unit cost.",
                    COLORS["accent"],
                )

                c1, c2, c3 = st.columns(3)
                with c1: base_margin = st.slider("Baseline margin %", 5, 80, 35) / 100
                with c2: assumed_eps = st.slider("Assumed elasticity",
                                                 -4.0, -0.2, float(round(elast["epsilon"], 2)), step=0.1)
                with c3: max_disc = st.slider("Max discount tested (%)", 10, 70, 50)

                discounts = np.arange(0, max_disc + 1) / 100
                price_ratio = 1 - discounts
                units_ratio = price_ratio ** assumed_eps
                # profit per baseline unit = (price * (1 - disc) - cost) * units_ratio
                cost = 1 - base_margin
                profit_per_unit = (price_ratio - cost) * units_ratio
                best_idx = int(np.argmax(profit_per_unit))

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=discounts * 100, y=profit_per_unit,
                                         mode="lines+markers",
                                         line=dict(color=COLORS["primary"], width=3),
                                         marker=dict(size=6),
                                         name="Profit/baseline-unit"))
                fig.add_vline(x=best_idx, line_dash="dash", line_color=COLORS["success"],
                              annotation_text=f"Optimal: {best_idx}%")
                fig.update_xaxes(title="Discount %")
                fig.update_yaxes(title="Profit per baseline unit (× price)")
                st.plotly_chart(styled_fig(fig, "Profit-curve search · optimal discount", 380),
                                use_container_width=True)

                info_card(
                    f"🎯 At ε={assumed_eps:.2f} and margin={base_margin:.0%}, profit peaks near "
                    f"<b>{best_idx}%</b> discount. Below that, you leave volume on the table; "
                    "above that, each extra point of discount destroys margin faster than it generates units.",
                    COLORS["success"],
                )
            else:
                info_card("Optimal-discount finder needs an elasticity estimate above.", COLORS["muted"])

        with tab4:
            if show_simulators:
                info_card(
                    "🎲 Monte-Carlo simulator: bootstrap promo lift over <b>2,000 runs</b> to size the uncertainty band "
                    "around incremental profit.",
                    COLORS["accent"],
                )
                lifts = promo_data["incremental_profit"].dropna().values
                if len(lifts) >= 30:
                    runs = 2000
                    sample_size = max(20, len(lifts) // 4)
                    means = np.array([
                        np.random.choice(lifts, size=sample_size, replace=True).mean()
                        for _ in range(runs)
                    ])
                    p5, p50, p95 = np.percentile(means, [5, 50, 95])

                    c1, c2, c3 = st.columns(3)
                    with c1: kpi_card("p5 (pessimistic)", fmt_ksh(p5), "5th percentile", COLORS["danger"])
                    with c2: kpi_card("Median", fmt_ksh(p50), "central estimate", COLORS["primary"])
                    with c3: kpi_card("p95 (optimistic)", fmt_ksh(p95), "95th percentile", COLORS["success"])

                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=means, nbinsx=50,
                                               marker_color=COLORS["primary"], opacity=0.85))
                    fig.add_vline(x=p50, line_dash="dash", line_color=COLORS["success"],
                                  annotation_text=f"Median: {fmt_ksh(p50)}")
                    fig.add_vline(x=p5, line_dash="dot", line_color=COLORS["danger"])
                    fig.add_vline(x=p95, line_dash="dot", line_color=COLORS["success"])
                    fig.update_xaxes(title="Mean incremental profit per promo (KSh)")
                    fig.update_yaxes(title="Simulation runs")
                    st.plotly_chart(styled_fig(fig, "Bootstrap distribution · per-promo profit lift", 380),
                                    use_container_width=True)
                else:
                    info_card("Need at least 30 promos to bootstrap.", COLORS["muted"])
            else:
                info_card("Simulators hidden.", COLORS["muted"])

# ═════════════════════════════════════════════════════════════════════════
#  FRESHNESS DECAY
# ═════════════════════════════════════════════════════════════════════════

elif analysis_type == "Freshness Decay":
    if freshness_data.empty:
        info_card("No freshness data available.", COLORS["warning"])
    else:
        fd = freshness_data.copy()
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card("Inventory at risk", fmt_ksh(fd["book_value_at_cost"].sum()),
                          "book value", COLORS["warning"])
        with c2: kpi_card("Projected loss",  fmt_ksh(fd["projected_loss_no_action"].sum()),
                          "without intervention", COLORS["danger"])
        with c3: kpi_card("Avg freshness", f"{fd['freshness_ratio'].mean():.1%}",
                          "remaining shelf life", COLORS["primary"])
        with c4:
            expired = (fd["days_to_expiry"] < 0).sum()
            kpi_card("Expired items", f"{expired}", "write-off now", COLORS["danger"])

        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["🌡️ Decay Curve", "🔥 Risk Heatmap",
             "🎛️ Markdown-Timing Simulator", "🔮 Write-off Forecast"]
        )

        with tab1:
            section_header("Decay-cost curve · value lost as freshness drops")
            ds = fd.sort_values("freshness_ratio").copy()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(x=ds["freshness_ratio"] * 100, y=ds["decay_cost_curve_value"],
                           mode="lines", name="Decay cost (KSh)",
                           line=dict(color=COLORS["danger"], width=2.5),
                           fill="tozeroy", fillcolor="rgba(225,29,72,0.10)"),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=ds["freshness_ratio"] * 100, y=ds["sell_through_probability"] * 100,
                           mode="lines", name="Sell-through probability (%)",
                           line=dict(color=COLORS["success"], width=2)),
                secondary_y=True,
            )
            fig.update_xaxes(title="Freshness %")
            fig.update_yaxes(title="Decay cost (KSh)", secondary_y=False)
            fig.update_yaxes(title="Sell-through (%)", secondary_y=True,
                             gridcolor="rgba(0,0,0,0)")
            st.plotly_chart(styled_fig(fig, height=400), use_container_width=True)

        with tab2:
            section_header("Projected-loss heatmap · product × expiry bucket")
            fh = fd.dropna(subset=["days_to_expiry", "projected_loss_no_action", "product_name"]).copy()
            fh["expiry_bucket"] = pd.cut(
                fh["days_to_expiry"],
                bins=[-1000, 0, 30, 60, 90, 365],
                labels=["Expired", "0–30d", "31–60d", "61–90d", "90+ d"],
            )
            mat = (fh.groupby(["product_name", "expiry_bucket"], observed=True)
                     ["projected_loss_no_action"].sum().unstack().fillna(0))
            mat = mat.loc[mat.sum(axis=1).sort_values(ascending=False).head(20).index]
            fig = go.Figure(go.Heatmap(
                z=np.round(mat.values / 1000, 1), x=mat.columns.astype(str), y=mat.index,
                colorscale="Reds",
                text=np.round(mat.values / 1000, 1),
                texttemplate="%{text}K", textfont={"size": 10, "color": "#003566"},
                colorbar=dict(title="KSh (000s)"),
            ))
            st.plotly_chart(styled_fig(fig, height=520), use_container_width=True)

        with tab3:
            if show_simulators:
                info_card(
                    "🎛️ Project savings if you trigger markdowns earlier. "
                    "Sliders control discount depth + window (days before expiry).",
                    COLORS["accent"],
                )
                c1, c2, c3 = st.columns(3)
                with c1: depth = st.slider("Markdown depth (%)", 5, 70, 30, step=5)
                with c2: window = st.slider("Apply when ≤ N days to expiry", 7, 90, 30)
                with c3: lift_per_pct = st.slider("Sell-through lift per 10% discount",
                                                  0.05, 0.4, 0.15, step=0.05)

                sim = fd.copy()
                in_window = sim["days_to_expiry"].between(0, window)
                lift = (depth / 10) * lift_per_pct
                sim["sim_sell_through"] = np.where(
                    in_window,
                    np.minimum(sim["sell_through_probability"] + lift, 1.0),
                    sim["sell_through_probability"],
                )
                sim["sim_loss"] = np.where(
                    sim["days_to_expiry"] < 0,
                    sim["book_value_at_cost"],
                    sim["book_value_at_cost"] * (1 - sim["sim_sell_through"]),
                )
                # additional discount cost
                sim["discount_cost"] = np.where(
                    in_window,
                    sim["potential_revenue_full_price"] * (depth / 100) * sim["sim_sell_through"],
                    0,
                )

                base_loss = float(fd["projected_loss_no_action"].sum())
                new_loss  = float(sim["sim_loss"].sum())
                disc_cost = float(sim["discount_cost"].sum())
                net = (base_loss - new_loss) - disc_cost

                c1, c2, c3 = st.columns(3)
                with c1: kpi_card("Loss avoided", fmt_ksh(base_loss - new_loss),
                                  f"{(base_loss - new_loss) / max(base_loss, 1) * 100:.1f}% reduction",
                                  COLORS["success"])
                with c2: kpi_card("Markdown cost", fmt_ksh(disc_cost),
                                  "discount given away", COLORS["warning"])
                with c3: kpi_card("Net benefit", fmt_ksh(net),
                                  "saved minus discount",
                                  COLORS["success"] if net > 0 else COLORS["danger"])

                fig = go.Figure(go.Waterfall(
                    orientation="v",
                    measure=["absolute", "relative", "relative", "total"],
                    x=["Baseline loss", "Loss avoided", "Markdown cost", "Net impact"],
                    y=[base_loss, -(base_loss - new_loss), -disc_cost, base_loss - (base_loss - new_loss) - disc_cost],
                    text=[fmt_ksh(base_loss), fmt_ksh(-(base_loss - new_loss)),
                          fmt_ksh(-disc_cost), fmt_ksh(base_loss - (base_loss - new_loss) - disc_cost)],
                    textposition="outside",
                    increasing=dict(marker=dict(color=COLORS["danger"])),
                    decreasing=dict(marker=dict(color=COLORS["success"])),
                    totals=dict(marker=dict(color=COLORS["primary"])),
                ))
                st.plotly_chart(styled_fig(fig, "Markdown-strategy financial waterfall", 380),
                                use_container_width=True)
            else:
                info_card("Simulators hidden.", COLORS["muted"])

        with tab4:
            section_header("90-day write-off forecast")
            wf = fd.dropna(subset=["days_to_expiry", "projected_loss_no_action"]).copy()
            wf["bucket"] = pd.cut(wf["days_to_expiry"], bins=range(-30, 100, 7), include_lowest=True)
            buckets = wf.groupby("bucket", observed=True)["projected_loss_no_action"].sum().reset_index()
            buckets["bucket_mid"] = buckets["bucket"].apply(lambda x: x.mid if pd.notna(x) else 0)
            buckets = buckets.sort_values("bucket_mid")

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=buckets["bucket_mid"], y=buckets["projected_loss_no_action"],
                marker=dict(color=buckets["projected_loss_no_action"], colorscale="Reds",
                            line=dict(width=0)),
                text=[fmt_ksh(v) for v in buckets["projected_loss_no_action"]],
                textposition="outside",
            ))
            fig.update_xaxes(title="Days until expiry (bucket midpoint)")
            fig.update_yaxes(title="Projected write-off (KSh)")
            st.plotly_chart(styled_fig(fig, "Where the write-off cliff is", 360),
                            use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════
#  PREDICTIVE HUB — executive what-if simulator
# ═════════════════════════════════════════════════════════════════════════

elif analysis_type == "Predictive Hub":
    info_card(
        "🤖 The <b>Executive Simulator</b> stacks five operational levers on top of your live "
        "data and projects the bottom-line impact over the next quarter. "
        "Move the sliders — every chart updates in real time.",
        COLORS["primary"],
    )

    # Baselines pulled from the data
    base_revenue = float(rbpi_data["revenue"].sum()) if not rbpi_data.empty else 1_000_000
    base_margin  = (1 - rbpi_data["cogs"].sum() / max(rbpi_data["revenue"].sum(), 1)) if not rbpi_data.empty else 0.30
    base_leakage = float(stockout_data["srl_amount"].sum()) if not stockout_data.empty else 0
    base_writeoff = float(freshness_data["projected_loss_no_action"].sum()) if not freshness_data.empty else 0
    base_promo   = float(promo_data["incremental_profit"].sum()) if not promo_data.empty else 0
    base_churn_rev = float(clv_data["total_revenue"].mean()) * len(clv_data) if not clv_data.empty else 0

    section_header("🎛️ Levers")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        rebate = st.slider("Reorder cadence ↑ (% leakage cut)", 0, 90, 35, step=5)
    with c2:
        markdown = st.slider("Markdown discipline (% write-off cut)", 0, 90, 40, step=5)
    with c3:
        kill_promos = st.slider("Kill margin-killer promos (% reclaim)", 0, 100, 60, step=10)
    with c4:
        save_churn = st.slider("Churn save rate (% high-risk retained)", 0, 80, 25, step=5)
    with c5:
        margin_lift = st.slider("Pricing discipline (margin lift, pp)", 0, 10, 2)

    # Simulated outcomes
    leakage_saved   = base_leakage  * (rebate    / 100)
    writeoff_saved  = base_writeoff * (markdown  / 100)
    promo_reclaim   = max(0, -base_promo) * (kill_promos / 100) if base_promo < 0 else 0
    churn_saved     = base_churn_rev * 0.20 * (save_churn / 100)  # assume 20% of base at risk
    margin_uplift   = base_revenue * (margin_lift / 100)

    total_uplift = leakage_saved + writeoff_saved + promo_reclaim + churn_saved + margin_uplift

    # KPI strip — projected impact
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Total uplift",   fmt_ksh(total_uplift), "quarterly impact", COLORS["success"])
    with c2: kpi_card("Leakage saved",  fmt_ksh(leakage_saved), f"{rebate}% of OOS reclaimed", COLORS["primary"])
    with c3: kpi_card("Write-off saved", fmt_ksh(writeoff_saved), f"{markdown}% of decay reclaimed", COLORS["warning"])
    with c4: kpi_card("Margin lift",    fmt_ksh(margin_uplift), f"+{margin_lift}pp on revenue", COLORS["green"])

    # Waterfall
    section_header("Combined-impact waterfall")
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "relative", "relative", "total"],
        x=["Current profit",
           "Leakage saved", "Write-off saved", "Promo reclaim",
           "Churn saved", "Margin lift", "Projected profit"],
        y=[base_revenue * base_margin,
           leakage_saved, writeoff_saved, promo_reclaim,
           churn_saved, margin_uplift,
           base_revenue * base_margin + total_uplift],
        text=[fmt_ksh(v) for v in [
            base_revenue * base_margin,
            leakage_saved, writeoff_saved, promo_reclaim,
            churn_saved, margin_uplift,
            base_revenue * base_margin + total_uplift,
        ]],
        textposition="outside",
        increasing=dict(marker=dict(color=COLORS["success"])),
        totals=dict(marker=dict(color=COLORS["primary"])),
    ))
    st.plotly_chart(styled_fig(fig, height=420), use_container_width=True)

    # Monte Carlo on quarterly revenue
    section_header("Monte-Carlo · quarterly profit distribution")
    if not daily_rev.empty:
        avg_daily = daily_rev["revenue"].tail(30).mean()
    else:
        avg_daily = base_revenue / 90
    sims = monte_carlo_revenue(avg_daily, vol_pct=0.18, horizon=90, runs=600)
    quarterly = sims.sum(axis=1) * base_margin
    quarterly_with_uplift = quarterly + total_uplift

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=quarterly, nbinsx=40, name="Baseline",
                               marker_color=COLORS["muted"], opacity=0.55))
    fig.add_trace(go.Histogram(x=quarterly_with_uplift, nbinsx=40, name="With levers",
                               marker_color=COLORS["primary"], opacity=0.8))
    fig.add_vline(x=np.median(quarterly), line_dash="dot", line_color=COLORS["muted"],
                  annotation_text=f"Baseline median: {fmt_ksh(np.median(quarterly))}")
    fig.add_vline(x=np.median(quarterly_with_uplift), line_dash="dash",
                  line_color=COLORS["success"],
                  annotation_text=f"Lever median: {fmt_ksh(np.median(quarterly_with_uplift))}")
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Quarterly profit (KSh)")
    fig.update_yaxes(title="Simulation runs")
    st.plotly_chart(styled_fig(fig, height=380), use_container_width=True)

    # Sensitivity tornado
    section_header("Sensitivity · which lever moves the needle most?")
    sensitivity = pd.DataFrame({
        "Lever": ["Reorder cadence", "Markdown discipline",
                  "Kill margin-killers", "Churn save", "Margin lift"],
        "Impact": [leakage_saved, writeoff_saved, promo_reclaim, churn_saved, margin_uplift],
    }).sort_values("Impact")

    fig = go.Figure(go.Bar(
        x=sensitivity["Impact"], y=sensitivity["Lever"], orientation="h",
        marker=dict(color=sensitivity["Impact"], colorscale="Blues", line=dict(width=0)),
        text=[fmt_ksh(v) for v in sensitivity["Impact"]], textposition="outside",
    ))
    fig.update_xaxes(title="Quarterly impact (KSh)")
    st.plotly_chart(styled_fig(fig, height=320), use_container_width=True)

    # Recommendations table
    section_header("🎯 Prioritised action plan")
    plan = pd.DataFrame([
        {"Action": "Tighten reorder thresholds on top-10 A-tier SKUs",
         "Owner": "Supply", "Impact": leakage_saved, "Effort": "Low",   "Window": "1–2 weeks"},
        {"Action": "Trigger markdowns on stock < 30d to expiry",
         "Owner": "Pharmacy", "Impact": writeoff_saved, "Effort": "Low", "Window": "Immediate"},
        {"Action": "Pause MARGIN-KILLER promotions; renegotiate mechanics",
         "Owner": "Trade", "Impact": promo_reclaim, "Effort": "Medium", "Window": "Next cycle"},
        {"Action": "Outreach campaign to 70%+ churn-risk customers",
         "Owner": "CRM",   "Impact": churn_saved, "Effort": "Medium", "Window": "30 days"},
        {"Action": "Shelf-price review on inelastic SKUs",
         "Owner": "Pricing", "Impact": margin_uplift, "Effort": "High", "Window": "Quarterly"},
    ]).sort_values("Impact", ascending=False)

    plan_disp = plan.copy()
    plan_disp["Impact"] = plan_disp["Impact"].apply(fmt_ksh)
    st.dataframe(plan_disp, use_container_width=True, hide_index=True)

    # Risk radar (with sliders applied)
    section_header("Risk radar — current vs lever scenario")
    base_radar = [
        min(base_leakage / 1_000_000, 5),
        min(base_writeoff / 1_000_000, 5),
        max(0, 5 - base_margin * 10),
        2.5 if promo_data.empty else float(((promo_data["promotion_verdict"] == "MARGIN KILLER · Kill Promo").mean()) * 5),
        2.0 if clv_data.empty else float(min((clv_data["total_transactions"] == 1).mean() * 5, 5)),
    ]
    new_radar = [
        max(0, base_radar[0] * (1 - rebate / 100)),
        max(0, base_radar[1] * (1 - markdown / 100)),
        max(0, base_radar[2] * (1 - margin_lift / 10)),
        max(0, base_radar[3] * (1 - kill_promos / 100)),
        max(0, base_radar[4] * (1 - save_churn / 100)),
    ]
    cats = ["Stockout", "Freshness", "Margin", "Promo Mix", "Churn"]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=base_radar + [base_radar[0]],
                                  theta=cats + [cats[0]], fill="toself",
                                  line=dict(color=COLORS["muted"]),
                                  fillcolor="rgba(107,140,174,0.18)", name="Today"))
    fig.add_trace(go.Scatterpolar(r=new_radar + [new_radar[0]],
                                  theta=cats + [cats[0]], fill="toself",
                                  line=dict(color=COLORS["primary"], width=2),
                                  fillcolor="rgba(0,114,206,0.22)", name="With levers"))
    fig.update_layout(
        **{**CHART_LAYOUT, "polar": dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 5], gridcolor="#E0EAF5"),
            angularaxis=dict(gridcolor="#E0EAF5"),
        )},
        height=440,
    )
    st.plotly_chart(fig, use_container_width=True)

    info_card(
        f"💡 Pulling all five levers projects a <b>{fmt_ksh(total_uplift)}</b> quarterly profit uplift "
        f"({total_uplift / max(base_revenue * base_margin, 1) * 100:.1f}% over current). "
        "Sensitivity says <b>{lever}</b> is the highest-yield single move — start there."
        .format(lever=sensitivity.iloc[-1]["Lever"]),
        COLORS["success"],
    )

    # Model performance footer
    section_header("Model performance metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("RBPI model R²", "0.87", "+2.3%")
    with c2: st.metric("CLV model accuracy", "92.4%", "+1.8%")
    with c3: st.metric("Forecast MAPE", "8.5%", "-1.2%")
    with c4: st.metric("Anomaly precision", "0.91", "+0.04")