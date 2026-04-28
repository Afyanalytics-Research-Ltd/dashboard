import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath('__file__')),
        "analytics_app",
        "dashboards",
        "xanalife",
        "cross_sell"
    )
)

import streamlit as st
import plotly.express as px
import pandas as pd
from xanalife.cross_sell.utils.snowflake_conn import run_query
from xanalife.cross_sell.utils.queries import HOME_STATS_QUERY, TOP_PRODUCTS_QUERY, STOCKOUT_PREDICTION_QUERY, STORE_PULSE_QUERY
from xanalife.cross_sell.utils.theme import inject_css, COLORS, CHART_LAYOUT, fmt_kes, sidebar_nav, section_header

st.set_page_config(
    page_title="XanaLife Analytics",
    page_icon="Logo.png" if os.path.exists("Logo.png") else "📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    sidebar_nav()
    section_header("About", margin_top=0)
    st.markdown(
        '<div style="font-size:11px;color:#8AABCC;line-height:2">'
        'Data: Sep 2025 – Mar 2026<br>'
        'Source: XanaLife POS &amp; Inventory<br>'
        'Refreshed every hour'
        '</div>',
        unsafe_allow_html=True,
    )

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner(""):
    try:
        stats       = run_query(HOME_STATS_QUERY)
        products    = run_query(TOP_PRODUCTS_QUERY)
        stockout_df = run_query(STOCKOUT_PREDICTION_QUERY)   # same query + cache as Inventory Risk page
        pulse_df    = run_query(STORE_PULSE_QUERY)

        products.columns    = [c.strip() for c in products.columns]
        stockout_df.columns = [c.strip() for c in stockout_df.columns]
        pulse_df.columns    = [c.strip() for c in pulse_df.columns]

        products["TOTAL_REVENUE"]                      = pd.to_numeric(products["TOTAL_REVENUE"],                      errors="coerce")
        products["TOTAL_UNITS"]                        = pd.to_numeric(products["TOTAL_UNITS"],                        errors="coerce")
        stockout_df["7-Day Revenue at Risk (KES)"]     = pd.to_numeric(stockout_df["7-Day Revenue at Risk (KES)"],     errors="coerce")
        pulse_df["TOTAL_REVENUE"]                      = pd.to_numeric(pulse_df["TOTAL_REVENUE"],                      errors="coerce")
        pulse_df["TRANSACTIONS"]                       = pd.to_numeric(pulse_df["TRANSACTIONS"],                       errors="coerce")
        pulse_df["AVG_BASKET_KES"]                     = pd.to_numeric(pulse_df["AVG_BASKET_KES"],                     errors="coerce")

        # Alerts — derived from the exact same dataset the Inventory Risk page uses
        n_stockouts = int((stockout_df["Stock Status"] == "Stockout").sum())
        n_critical  = int((stockout_df["Stock Status"] == "Critical").sum())
        rev_at_risk = float(
            stockout_df[stockout_df["Stock Status"].isin(["Stockout", "Critical"])][
                "7-Day Revenue at Risk (KES)"
            ].sum()
        )

        data_loaded = True
    except Exception as e:
        data_loaded = False
        products = pulse_df = stockout_df = pd.DataFrame()
        n_stockouts = n_critical = 0
        rev_at_risk = 0.0
        st.error(f"⚠ Data load failed: {e}")

# ══ HERO ══════════════════════════════════════════════════════════════════════
col_logo, col_text = st.columns([1, 3], gap="large")
with col_logo:
    st.markdown("<div style='margin-top:8px'>", unsafe_allow_html=True)
    if os.path.exists("Logo.png"):
        st.image("Logo.png", width=200)
    st.markdown("</div>", unsafe_allow_html=True)

with col_text:
    st.markdown(
        '<div style="padding-top:12px">'
        '<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
        'letter-spacing:2px;color:#0072CE;margin-bottom:8px">Analytics Dashboard</div>'
        '<h1 style="font-size:32px;font-weight:800;color:#003467;margin:0 0 10px;line-height:1.15">'
        'Decision Intelligence<br>for XanaLife</h1>'
        '<p style="font-size:14px;color:#6B8CAE;margin:0;line-height:1.7;max-width:560px">'
        'Live analytics built on your POS and inventory data — giving management '
        'the insight to grow revenue, eliminate stockouts, and act with confidence.'
        '</p></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div style="display:flex;gap:10px;margin:20px 0 28px;flex-wrap:wrap">'
    '<span style="background:#EBF3FB;color:#0072CE;font-size:11px;font-weight:600;'
    'padding:4px 12px;border-radius:20px">📅 Sep 2025 – Mar 2026</span>'
    '<span style="background:#F0FDF9;color:#0BB99F;font-size:11px;font-weight:600;'
    'padding:4px 12px;border-radius:20px">🔄 Refreshes every hour</span>'
    '<span style="background:#FFF7ED;color:#D97706;font-size:11px;font-weight:600;'
    'padding:4px 12px;border-radius:20px">📍 XanaLife POS + Inventory</span>'
    '</div>',
    unsafe_allow_html=True,
)

# ══ ACTION REQUIRED ═══════════════════════════════════════════════════════════
if data_loaded:
    has_alerts = n_stockouts > 0 or n_critical > 0

    if has_alerts:
        st.markdown(
            '<div style="background:#FFF5F7;border:1px solid #FEC5CF;border-left:4px solid #E11D48;'
            'border-radius:8px;padding:16px 20px;margin-bottom:28px">'
            '<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:2px;'
            'color:#E11D48;margin-bottom:14px">⚡ Action Required</div>',
            unsafe_allow_html=True,
        )
        a1, a2, a3, a4 = st.columns([1, 1, 1, 1])
        with a1:
            st.markdown(
                f'<div style="padding:4px 0">'
                f'<div style="font-size:10px;font-weight:700;color:#E11D48;text-transform:uppercase;'
                f'letter-spacing:1px;margin-bottom:6px">🔴 Out of Stock</div>'
                f'<div style="font-size:28px;font-weight:800;color:#E11D48;line-height:1">{n_stockouts}</div>'
                f'<div style="font-size:11px;color:#8AABCC;margin-top:4px">products — no stock left</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with a2:
            st.markdown(
                f'<div style="padding:4px 0">'
                f'<div style="font-size:10px;font-weight:700;color:#f97316;text-transform:uppercase;'
                f'letter-spacing:1px;margin-bottom:6px">🟠 Order Today</div>'
                f'<div style="font-size:28px;font-weight:800;color:#f97316;line-height:1">{n_critical}</div>'
                f'<div style="font-size:11px;color:#8AABCC;margin-top:4px">products — less than 7 days left</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with a3:
            st.markdown(
                f'<div style="padding:4px 0">'
                f'<div style="font-size:10px;font-weight:700;color:#003467;text-transform:uppercase;'
                f'letter-spacing:1px;margin-bottom:6px">💰 Revenue at Risk</div>'
                f'<div style="font-size:28px;font-weight:800;color:#003467;line-height:1">{fmt_kes(rev_at_risk)}</div>'
                f'<div style="font-size:11px;color:#8AABCC;margin-top:4px">if no action in 7 days</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with a4:
            st.markdown(
                '<div style="display:flex;align-items:center;height:100%;padding:4px 0">'
                '<div>'
                '<div style="font-size:12px;color:#003467;font-weight:600;margin-bottom:8px">'
                'Reorder list is ready.</div>'
                '<div style="font-size:11px;color:#8AABCC;line-height:1.6">'
                'Go to <b>Inventory Risk → Order Now</b><br>to download the prioritised list.</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="background:#F0FDF9;border:1px solid #6EE7D4;border-radius:8px;'
            'padding:14px 20px;margin-bottom:28px;font-size:13px;color:#0BB99F;font-weight:600">'
            '✅ &nbsp; Stock health looks good — no products are currently out of stock or critical.'
            '</div>',
            unsafe_allow_html=True,
        )

# ══ MODULE CARDS ══════════════════════════════════════════════════════════════
st.markdown(
    '<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:2px;'
    'color:#0072CE;margin-bottom:16px">Analytics Modules</div>',
    unsafe_allow_html=True,
)

card_l, card_r = st.columns(2, gap="large")

with card_l:
    st.markdown(
        '<div style="border:1px solid #D6E4F0;border-radius:12px;overflow:hidden">'
        '<div style="background:linear-gradient(135deg,#0072CE,#005fad);padding:22px 24px 18px">'
        '<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:2px;'
        'color:rgba(255,255,255,0.6);margin-bottom:6px">Module 01</div>'
        '<div style="font-size:20px;font-weight:800;color:#ffffff;margin-bottom:4px">'
        'Cross-Sell Intelligence</div>'
        '<div style="font-size:12px;color:rgba(255,255,255,0.75)">What do your customers buy together?</div>'
        '</div>'
        '<div style="padding:20px 24px">'
        '<div style="font-size:12px;color:#6B8CAE;line-height:2.2">'
        '<span style="color:#0BB99F;font-weight:700">✓</span>&nbsp; Statistically validated buying patterns<br>'
        '<span style="color:#0BB99F;font-weight:700">✓</span>&nbsp; Point-of-sale recommendation scripts<br>'
        '<span style="color:#0BB99F;font-weight:700">✓</span>&nbsp; Revenue opportunity sizing per product pair<br>'
        '<span style="color:#0BB99F;font-weight:700">✓</span>&nbsp; Filter by store, confidence, or product'
        '</div>'
        '<div style="margin-top:16px;padding-top:14px;border-top:1px solid #EBF3FB">'
        '<span style="font-size:11px;color:#0072CE;font-weight:700">→ Select in sidebar menu</span>'
        '</div></div></div>',
        unsafe_allow_html=True,
    )

with card_r:
    st.markdown(
        '<div style="border:1px solid #D6E4F0;border-radius:12px;overflow:hidden">'
        '<div style="background:linear-gradient(135deg,#003467,#1a4f8a);padding:22px 24px 18px">'
        '<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:2px;'
        'color:rgba(255,255,255,0.6);margin-bottom:6px">Module 02</div>'
        '<div style="font-size:20px;font-weight:800;color:#ffffff;margin-bottom:4px">'
        'Inventory Risk Monitor</div>'
        '<div style="font-size:12px;color:rgba(255,255,255,0.75)">Which products are about to run out?</div>'
        '</div>'
        '<div style="padding:20px 24px">'
        '<div style="font-size:12px;color:#6B8CAE;line-height:2.2">'
        '<span style="color:#0BB99F;font-weight:700">✓</span>&nbsp; 30-day stockout calendar<br>'
        '<span style="color:#0BB99F;font-weight:700">✓</span>&nbsp; Revenue at risk in KES (7-day)<br>'
        '<span style="color:#0BB99F;font-weight:700">✓</span>&nbsp; Smart reorder quantities from demand data<br>'
        '<span style="color:#0BB99F;font-weight:700">✓</span>&nbsp; ABC product classification'
        '</div>'
        '<div style="margin-top:16px;padding-top:14px;border-top:1px solid #EBF3FB">'
        '<span style="font-size:11px;color:#0072CE;font-weight:700">→ Select in sidebar menu</span>'
        '</div></div></div>',
        unsafe_allow_html=True,
    )

# ══ PERFORMANCE SNAPSHOT ══════════════════════════════════════════════════════
st.markdown("<div style='margin-top:44px'></div>", unsafe_allow_html=True)
st.markdown('<div style="border-bottom:1px solid #EBF3FB;margin-bottom:28px"></div>', unsafe_allow_html=True)
st.markdown(
    '<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:2px;'
    'color:#0072CE;margin-bottom:4px">Performance Snapshot</div>'
    '<div style="font-size:22px;font-weight:800;color:#003467;margin-bottom:6px">Top 10 Products</div>'
    '<div style="font-size:13px;color:#8AABCC;margin-bottom:24px">'
    'Sep 2025 – Mar 2026 &nbsp;·&nbsp; All stores combined</div>',
    unsafe_allow_html=True,
)

if data_loaded and not products.empty:
    ch_l, ch_r = st.columns(2, gap="large")

    with ch_l:
        top_rev = (
            products.sort_values("TOTAL_REVENUE", ascending=False).head(10)
            .sort_values("TOTAL_REVENUE", ascending=True).copy()
        )
        top_rev["Label"]    = top_rev["PRODUCT_NAME"].str.title()
        top_rev["RevLabel"] = top_rev["TOTAL_REVENUE"].apply(
            lambda v: f"KES {v/1_000:.0f}K" if v >= 1_000 else f"KES {v:.0f}"
        )
        fig_rev = px.bar(
            top_rev, x="TOTAL_REVENUE", y="Label", orientation="h",
            text="RevLabel",
            color="TOTAL_REVENUE",
            color_continuous_scale=[[0, "#B0D4F1"], [1, "#0072CE"]],
            height=380,
        )
        fig_rev.update_traces(textposition="outside", textfont=dict(size=10, color="#003467"), marker_line_width=0)
        fig_rev.update_layout(**{**CHART_LAYOUT, "margin": dict(l=0, r=60, t=10, b=10)}, height=380, coloraxis_showscale=False)
        fig_rev.update_xaxes(title=None, showticklabels=False, showgrid=False, zeroline=False)
        fig_rev.update_yaxes(title=None, tickfont=dict(size=11, color="#003467"))
        st.markdown('<div style="font-size:12px;font-weight:700;color:#003467;margin-bottom:10px">By Revenue (KES)</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_rev, use_container_width=True)

    with ch_r:
        top_vol = (
            products.sort_values("TOTAL_UNITS", ascending=False).head(10)
            .sort_values("TOTAL_UNITS", ascending=True).copy()
        )
        top_vol["Label"]    = top_vol["PRODUCT_NAME"].str.title()
        top_vol["VolLabel"] = top_vol["TOTAL_UNITS"].apply(
            lambda v: f"{v/1_000:.1f}K units" if v >= 1_000 else f"{v:.0f} units"
        )
        fig_vol = px.bar(
            top_vol, x="TOTAL_UNITS", y="Label", orientation="h",
            text="VolLabel",
            color="TOTAL_UNITS",
            color_continuous_scale=[[0, "#A8E6DC"], [1, "#0BB99F"]],
            height=380,
        )
        fig_vol.update_traces(textposition="outside", textfont=dict(size=10, color="#003467"), marker_line_width=0)
        fig_vol.update_layout(**{**CHART_LAYOUT, "margin": dict(l=0, r=80, t=10, b=10)}, height=380, coloraxis_showscale=False)
        fig_vol.update_xaxes(title=None, showticklabels=False, showgrid=False, zeroline=False)
        fig_vol.update_yaxes(title=None, tickfont=dict(size=11, color="#003467"))
        st.markdown('<div style="font-size:12px;font-weight:700;color:#003467;margin-bottom:10px">By Volume (units sold)</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_vol, use_container_width=True)

    top_by_rev = products.sort_values("TOTAL_REVENUE", ascending=False).iloc[0]
    top_by_vol = products.sort_values("TOTAL_UNITS",   ascending=False).iloc[0]
    if top_by_rev["PRODUCT_NAME"] != top_by_vol["PRODUCT_NAME"]:
        st.markdown(
            f'<div style="background:#F4F8FC;border-left:3px solid #0072CE;border-radius:4px;'
            f'padding:12px 16px;margin-top:4px;font-size:12px;color:#003467">'
            f'<b>Insight:</b> Your highest-revenue product (<b>{top_by_rev["PRODUCT_NAME"].title()}</b>) '
            f'is not your highest-volume product (<b>{top_by_vol["PRODUCT_NAME"].title()}</b>). '
            f'The volume leader sells more units but generates less revenue per unit — worth reviewing its margin.'
            f'</div>',
            unsafe_allow_html=True,
        )

# ══ STORE PULSE ═══════════════════════════════════════════════════════════════
if data_loaded and not pulse_df.empty:
    st.markdown("<div style='margin-top:44px'></div>", unsafe_allow_html=True)
    st.markdown('<div style="border-bottom:1px solid #EBF3FB;margin-bottom:28px"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:2px;'
        'color:#0072CE;margin-bottom:4px">Store Performance</div>'
        '<div style="font-size:22px;font-weight:800;color:#003467;margin-bottom:6px">'
        'How is each store performing?</div>'
        '<div style="font-size:13px;color:#8AABCC;margin-bottom:24px">'
        'Ranked by total revenue · Sep 2025 – Mar 2026</div>',
        unsafe_allow_html=True,
    )

    grand_total = pulse_df["TOTAL_REVENUE"].sum()
    n_stores    = min(len(pulse_df), 6)
    store_cols  = st.columns(n_stores, gap="medium")

    for i, (_, row) in enumerate(pulse_df.head(n_stores).iterrows()):
        pct      = (row["TOTAL_REVENUE"] / grand_total * 100) if grand_total > 0 else 0
        bar_w    = int(pct)
        is_top   = i == 0
        border   = "border-top:3px solid #0072CE;" if is_top else "border-top:3px solid #D6E4F0;"

        with store_cols[i]:
            st.markdown(
                f'<div style="background:#F4F8FC;border:1px solid #D6E4F0;{border}'
                f'border-radius:8px;padding:16px 14px">'
                f'<div style="font-size:10px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
                f'letter-spacing:0.8px;margin-bottom:10px;white-space:nowrap;overflow:hidden;'
                f'text-overflow:ellipsis">{row["STORE_NAME"]}</div>'
                f'<div style="font-size:20px;font-weight:800;color:#003467;line-height:1;margin-bottom:4px">'
                f'{fmt_kes(row["TOTAL_REVENUE"])}</div>'
                f'<div style="font-size:10px;color:#8AABCC;margin-bottom:12px">'
                f'{int(row["TRANSACTIONS"]):,} transactions &nbsp;·&nbsp; '
                f'{fmt_kes(row["AVG_BASKET_KES"])} avg basket</div>'
                f'<div style="background:#D6E4F0;border-radius:4px;height:4px;margin-bottom:6px">'
                f'<div style="background:#0072CE;height:4px;border-radius:4px;width:{bar_w}%"></div></div>'
                f'<div style="font-size:10px;color:#6B8CAE">{pct:.1f}% of total revenue</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

st.markdown("<div style='margin-top:40px'></div>", unsafe_allow_html=True)
