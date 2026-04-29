"""
xanalife_app.py  v4
===================
XanaLife Analytics — Customer Intelligence

Changes from v3:
  - Overall New Customer Growth removed — per-store chart IS the primary chart
  - IndentationError in one-timer section fixed — all inside with tab3
  - SyntaxError stray 'a' on fig_lag.update_yaxes removed
  - legend duplicate key fixed on fig_d and fig_pay (split into two update_layout calls)
  - **AXIS, showgrid=False duplicate key fixed throughout via _ax() helper
  - width='stretch' throughout (use_container_width removed)
  - One-Time excluded from return window funnel selector
"""

import io
import os
import sys
from pathlib import Path

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath('__file__')),
        "analytics_app",
        "dashboards",
        "xanalife",
        "customers"
    )
)


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date

from ui_template import (
    AFYA_BLUE, TEAL, COOL_BLUE, ORANGE, CORAL, PURPLE, GRAY,
    CHART_LAYOUT, AXIS, BG_LIGHT, BORDER, SEG_COLORS, SEQ,
)
import connect_to_snowflake as D

st.set_page_config(page_title="XanaLife Analytics", page_icon="🛒",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Montserrat',sans-serif;background:#fff;color:#003467}
.stApp{background:#fff}
[data-testid="stSidebar"]{background:#F4F8FC;border-right:1px solid #D6E4F0}
[data-testid="stSidebar"] *{color:#003467!important;font-family:'Montserrat',sans-serif!important}
.sh{font-size:12px;font-weight:800;color:#0072CE;text-transform:uppercase;
    letter-spacing:2.5px;padding:8px 0;border-bottom:2px solid #EBF3FB;margin-bottom:12px}
[data-baseweb="tab"]{font-family:'Montserrat',sans-serif!important;font-weight:600!important;
  color:#6B8CAE!important;font-size:14px!important}
[aria-selected="true"]{color:#0072CE!important;border-bottom-color:#0072CE!important}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-thumb{background:#B0C8E0;border-radius:10px}
.warn{background:#FFF8E7;border-left:4px solid #D97706;border-radius:4px;
  padding:10px 14px;font-size:13px;color:#92400E;margin-bottom:10px;font-weight:600}
.is  {background:#F4F8FC;border:1px solid #D6E4F0;border-radius:8px;padding:11px 13px;margin-top:6px}
.is-r{background:#FFF5F5;border:1px solid #FEB2B2;border-radius:8px;padding:11px 13px;margin-top:6px}
.is-a{background:#FFFBEB;border:1px solid #FCD34D;border-radius:8px;padding:11px 13px;margin-top:6px}
.is-g{background:#F0FFF4;border:1px solid #9AE6B4;border-radius:8px;padding:11px 13px;margin-top:6px}
.il{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:3px}
.iv{font-size:16px;font-weight:700;margin-bottom:2px}
.ia{font-size:13px;color:#4A5568}
.nb{background:#F4F8FC;border-left:3px solid #0072CE;padding:7px 11px;
  font-size:14px;color:#003467;margin-top:5px;border-radius:0 4px 4px 0;font-style:italic}
.nw{background:#FFFBEB;border-left:3px solid #D97706;padding:7px 11px;
  font-size:14px;color:#92400E;margin-top:5px;border-radius:0 4px 4px 0}
</style>
""", unsafe_allow_html=True)

_CL = {**CHART_LAYOUT, "margin": {"t": 10, "b": 10, "l": 0, "r": 80}}


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = int(hex_color[1:3],16), int(hex_color[3:5],16), int(hex_color[5:7],16)
    return f"rgba({r},{g},{b},{alpha})"

def _ax(**overrides):
    return {**AXIS, **overrides}

def fmt_ksh(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    if abs(v) >= 1_000_000: return f"KSh {v/1_000_000:.2f}M"
    if abs(v) >= 1_000:     return f"KSh {v/1_000:.1f}K"
    return f"KSh {v:.0f}"

def fmt_num(v): return "—" if v is None else f"{v:,.0f}"

def safe_str(v, suffix=""):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    return f"{v}{suffix}"

def kpi_card(label, value, sub="", color=AFYA_BLUE):
    st.markdown(
        f'<div style="background:{BG_LIGHT};border:1px solid {BORDER};border-radius:8px;padding:14px 12px">'
        f'<div style="font-size:9px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:5px">{label}</div>'
        f'<div style="font-size:22px;font-weight:800;color:{color};line-height:1">{value}</div>'
        f'<div style="font-size:10px;color:#6B8CAE;margin-top:4px">{sub}</div>'
        f'</div>', unsafe_allow_html=True)

def sh(text, mt=0):
    st.markdown(f'<div class="sh" style="margin-top:{mt}px">{text}</div>', unsafe_allow_html=True)

def insight(label, signal, action, t="blue"):
    css = {"blue":"is","red":"is-r","amber":"is-a","green":"is-g"}.get(t,"is")
    lc  = {"blue":"#0072CE","red":"#C53030","amber":"#92400E","green":"#276749"}.get(t,AFYA_BLUE)
    sc  = {"blue":"#003467","red":"#C53030","amber":"#744210","green":"#22543D"}.get(t,COOL_BLUE)
    st.markdown(
        f'<div class="{css}"><div class="il" style="color:{lc}">{label}</div>'
        f'<div class="iv" style="color:{sc}">{signal}</div>'
        f'<div class="ia">{action}</div></div>', unsafe_allow_html=True)

def note(text, warn=False):
    st.markdown(f'<div class="{"nw" if warn else "nb"}">{text}</div>', unsafe_allow_html=True)

def gap(px=10):
    st.markdown(f'<div style="margin:{px}px 0"></div>', unsafe_allow_html=True)

def seg_color(s): return SEG_COLORS.get(str(s) if s else "", GRAY)

def to_csv(df: pd.DataFrame) -> bytes:
    buf = io.StringIO(); df.to_csv(buf, index=False); return buf.getvalue().encode()

def pc(fig): st.plotly_chart(fig, width='stretch')

def apply_filters(df: pd.DataFrame,
                  cluster_col="cluster", type_col="store_type") -> pd.DataFrame:
    if cluster_col in df.columns and cluster != "Both":
        df = df[df[cluster_col].fillna("") == cluster.replace(" only", "")]
    if type_col in df.columns and biz != "All":
        wanted = "Supermarket" if "Supermarket" in biz else "Pharmacy"
        df = df[df[type_col].fillna("") == wanted]
    return df

def excl_wholesale(df: pd.DataFrame, col="store_name") -> pd.DataFrame:
    if col not in df.columns: return df
    mask = (df[col].fillna("").str.contains("Wholesale", case=False) |
            df[col].fillna("").str.contains("Bulk",      case=False))
    return df[~mask]

def plotly_table(header_vals, cell_vals, col_widths=None,
                 row_fill=None, header_color=COOL_BLUE,
                 cell_font_color=COOL_BLUE, height=None):
    fig = go.Figure(go.Table(
        columnwidth=col_widths,
        header=dict(values=header_vals, fill_color=header_color,
                    font=dict(color="white", size=10, family="Montserrat"),
                    align="left", height=28),
        cells=dict(values=cell_vals,
                   fill_color=row_fill or BG_LIGHT,
                   font=dict(color=cell_font_color, size=10, family="Montserrat"),
                   align="left", height=24),
    ))
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0),
                      **({"height": height} if height else {}))
    return fig


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="font-size:17px;font-weight:800;color:{AFYA_BLUE};padding:6px 0 18px">'
        '🛒 XanaLife Analytics</div>', unsafe_allow_html=True)

    st.markdown('<div class="sh">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("Page", ["👥  Customer Analytics"], label_visibility="collapsed")
    gap(12)

    st.markdown('<div class="sh">Date Range</div>', unsafe_allow_html=True)
    d_from = st.date_input("From", value=date(2025, 9, 1))
    d_to   = st.date_input("To",   value=date(2026, 3, 31))
    gap(12)

    st.markdown('<div class="sh">Store Cluster</div>', unsafe_allow_html=True)
    cluster = st.radio("Cluster", ["Both", "Katani only", "Syokimau only"],
        help="Syokimau = Main Store, Pharmacy Store + all Syokimau branches\n"
             "Katani = Katani Pharmacy + Katani Wholesale")
    gap(12)

    st.markdown('<div class="sh">Business Unit</div>', unsafe_allow_html=True)
    biz = st.radio("Unit", ["All", "Supermarket only", "Pharmacy only"],
        help="Pharmacy = Pharmacy Store + Katani Pharmacy\n"
             "All other stores = Supermarket")
    gap(12)

    excl_ws = st.checkbox("Exclude Wholesale + Bulk",
        help="Removes Katani Wholesale and Syokimau Bulk Store from all charts")

    st.markdown("---")
    st.markdown(
        f'<div style="font-size:10px;color:#6B8CAE;line-height:1.7">'
        f'Data: Sep 2025 – Mar 2026<br>Source: HOSPITALS.XANALIFE_CLEAN</div>',
        unsafe_allow_html=True)
    gap(8)
    fs = f"{cluster} · {biz}" + (" · No Wholesale/Bulk" if excl_ws else "")
    st.markdown(
        f'<div style="background:#E0EDFA;border-radius:6px;padding:8px 10px;'
        f'font-size:10px;font-weight:700;color:{AFYA_BLUE}">'
        f'Active: {fs}<br>{d_from} → {d_to}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — CUSTOMER ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    f'<p style="font-size:11px;font-weight:800;letter-spacing:3px;'
    f'text-transform:uppercase;color:{AFYA_BLUE};margin-bottom:2px">'
    f'XanaLife · Customer Intelligence</p>'
    f'<p style="font-size:12px;color:#6B8CAE;margin-bottom:8px">'
    f'Who are our customers, are they returning, and where is value concentrated?</p>',
    unsafe_allow_html=True)


gap(12)

kpi_df = D.load_kpis()
bsk_df = D.load_avg_basket()
rl_df  = D.load_regular_loyal()

def _kpi_metric(label, value, definition, color):
    return (
        f'<div style="flex:1;min-width:0;padding:0 12px">'
        f'<div style="font-size:13px;font-weight:600;color:{color};margin-bottom:3px">{label}</div>'
        f'<div style="font-size:32px;font-weight:800;color:{color};line-height:1.1">{value}</div>'
        f'<div style="font-size:12px;color:#8A9BB0;margin-top:4px;line-height:1.4">{definition}</div>'
        f'</div>'
    )

def _kpi_tile(title, metrics_html):
    return (
        f'<div style="background:{BG_LIGHT};border:1px solid {BORDER};border-radius:10px;'
        f'padding:18px 20px;height:100%">'
        f'<div style="font-size:10px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid {BORDER}">'
        f'{title}</div>'
        f'<div style="display:flex;gap:0">{metrics_html}</div>'
        f'</div>'
    )

_t1, _t2, _t3 = st.columns(3, gap="large")
with _t1:
    st.markdown(_kpi_tile("Customer Base", (
        _kpi_metric("Active",   fmt_num(kpi_df["active_customers"].iloc[0]),   "total customers (lifetime)", AFYA_BLUE) +
        _kpi_metric("Repeat",   fmt_num(kpi_df["repeat_customers"].iloc[0]),   "visited more than once",     TEAL) +
        _kpi_metric("One-Time", fmt_num(kpi_df["one_time_customers"].iloc[0]), "never returned",             CORAL)
    )), unsafe_allow_html=True)
with _t2:
    st.markdown(_kpi_tile("Loyalty & Retention — Customers", (
        _kpi_metric("Regular",  fmt_num(rl_df["regular_customers"].iloc[0]),  "6+ visits in 90 days",    PURPLE) +
        _kpi_metric("Loyal",    fmt_num(rl_df["loyal_customers"].iloc[0]),    "12+ visits, seen in 21d", COOL_BLUE) +
        _kpi_metric("New (30d)",fmt_num(kpi_df["new_last_30d"].iloc[0]),      "first purchase",          ORANGE)
    )), unsafe_allow_html=True)
with _t3:
    st.markdown(_kpi_tile("Basket", (
        _kpi_metric("Avg Value", fmt_ksh(bsk_df["avg_basket_value"].iloc[0]),    "KES per transaction", TEAL) +
        _kpi_metric("Avg Items", fmt_num(bsk_df["avg_items_per_basket"].iloc[0]),"per transaction",     GRAY)
    )), unsafe_allow_html=True)

gap(16)

tab1, tab2, tab3, tab4 = st.tabs([
    "◉  Customer Base & Growth",
    "△  Segmentation (CLV)",
    "↩  Retention",
    "★  Loyalty",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    sh("Customer Growth — Total Active vs New Acquisitions")
    act_df = D.load_active_customers_over_time()
    growth_df = D.load_growth_per_store()
    growth_df["store_name"] = growth_df["store_name"].fillna("Unknown")
    if excl_ws: growth_df = excl_wholesale(growth_df)
    growth_df = apply_filters(growth_df)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        if "store_name" not in act_df.columns:
            D.load_active_customers_over_time.clear()
            act_df = D.load_active_customers_over_time()
        act_df["store_name"] = act_df["store_name"].fillna("Unknown")
        if excl_ws: act_df = excl_wholesale(act_df)
        act_df = apply_filters(act_df)
        stores_in_view = sorted([s for s in act_df["store_name"].unique() if s])
        palette = [AFYA_BLUE, TEAL, PURPLE, ORANGE, CORAL, GRAY, "#185FA5", "#0BB99F"]
        cmap = {s: palette[i % len(palette)] for i, s in enumerate(stores_in_view)}
        fig_act = go.Figure()
        for store in stores_in_view:
            d = act_df[act_df["store_name"] == store]
            fig_act.add_trace(go.Scatter(
                x=d["month"], y=d["active_customers"], name=store,
                mode="lines+markers",
                line=dict(color=cmap[store], width=2), marker=dict(size=5),
                fill="tozeroy", fillcolor=_rgba(cmap[store], 0.07),
                hovertemplate=f"<b>{store}</b><br>%{{x|%b %Y}}: %{{y}}<extra></extra>",
            ))
        fig_act.update_layout(**_CL, height=300)
        fig_act.update_layout(
            legend=dict(orientation="h", yanchor="top", y=-0.18,
                        xanchor="left", x=0, font=dict(size=9)),
            margin=dict(t=10, b=80, l=0, r=10))
        fig_act.update_xaxes(**AXIS, tickformat="%b %Y")
        fig_act.update_yaxes(**AXIS, title_text="Customers")
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Overall Customers per Store (Active & New Combined)</div>', unsafe_allow_html=True)
        pc(fig_act)
        note("All customers who transacted that month — new and returning combined.")

    with c2:
        stores_in_view = sorted([s for s in growth_df["store_name"].unique() if s])
        palette = [AFYA_BLUE, TEAL, PURPLE, ORANGE, CORAL, GRAY, "#185FA5", "#0BB99F"]
        cmap = {s: palette[i % len(palette)] for i, s in enumerate(stores_in_view)}
        fig_gs = go.Figure()
        for store in stores_in_view:
            d = growth_df[growth_df["store_name"] == store]
            fig_gs.add_trace(go.Scatter(
                x=d["month"], y=d["new_customers"], name=store,
                mode="lines+markers",
                line=dict(color=cmap[store], width=2), marker=dict(size=5),
                fill="tozeroy", fillcolor=_rgba(cmap[store], 0.07),
                hovertemplate=f"<b>{store}</b><br>%{{x|%b %Y}}: %{{y}}<extra></extra>",
            ))
        fig_gs.update_layout(**_CL, height=300)
        fig_gs.update_layout(
            legend=dict(orientation="h", yanchor="top", y=-0.18,
                        xanchor="left", x=0, font=dict(size=9)),
            margin=dict(t=0, b=80, l=0, r=10))
        fig_gs.update_xaxes(**AXIS, tickformat="%b %Y")
        fig_gs.update_yaxes(**AXIS, title_text="New customers")
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">New Customer Acquisition — Per Store</div>', unsafe_allow_html=True)
        pc(fig_gs)
        note("First-time buyers only. Filter by cluster or business unit using the sidebar.")

    gap(8)
    c1, c2 = st.columns(2, gap="large")

    with c1:
        sh("Supermarket vs Pharmacy Split")
        split_df = D.load_store_split()
        split_df["store_name"] = split_df["store_name"].fillna("Unknown")
        if excl_ws: split_df = excl_wholesale(split_df)
        split_df = apply_filters(split_df)
        store_options = ["All stores"] + sorted(split_df["store_name"].unique().tolist())
        sel_store = st.selectbox("Filter by store", store_options, key="split_store")
        if sel_store != "All stores":
            split_df = split_df[split_df["store_name"] == sel_store]
        donut_df = split_df.groupby("store_type", as_index=False)["customer_count"].sum()
        total = donut_df["customer_count"].sum()
        fig_d = go.Figure(go.Pie(
            labels=donut_df["store_type"], values=donut_df["customer_count"],
            hole=0.65, marker_colors=[AFYA_BLUE, TEAL],
            textfont=dict(size=11, family="Montserrat"),
            hovertemplate="%{label}: %{value:,} customers (%{percent})<extra></extra>",
        ))
        fig_d.update_layout(**_CL, height=260, showlegend=True,
            annotations=[dict(
                text=f"{total:,}<br><span style='font-size:10px;color:#6B8CAE'>customers</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=15, color=COOL_BLUE, family="Montserrat"))])
        fig_d.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0.1))
        pc(fig_d)

    with c2:
        sh("New vs Returning Revenue — Monthly")
        nvr = D.load_new_vs_returning()
        color_nvr = {"New Customer Revenue": TEAL, "Returning Customer Revenue": AFYA_BLUE}
        fig_nvr = go.Figure()
        for rtype in ["New Customer Revenue", "Returning Customer Revenue"]:
            d = nvr[nvr["revenue_type"] == rtype]
            if not d.empty:
                fig_nvr.add_trace(go.Bar(
                    x=d["month"], y=d["revenue"], name=rtype,
                    marker_color=color_nvr.get(rtype, GRAY),
                    hovertemplate=f"<b>{rtype}</b><br>%{{x|%b %Y}}: KSh %{{y:,.0f}}<extra></extra>",
                ))
        fig_nvr.update_layout(**_CL, height=260, barmode="stack")
        fig_nvr.update_xaxes(**AXIS, tickformat="%b %Y")
        fig_nvr.update_yaxes(**AXIS, tickformat=",.0f", tickprefix="KSh ")
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">New vs Returning Revenue</div>', unsafe_allow_html=True)
        pc(fig_nvr)
        note("Returning (blue) flat while new (teal) grows = leaky bucket.")

    gap(8)
    sh("Basket Behaviour — Shop Type · Cross-Sell", mt=4)
    c1, c2 = st.columns(2, gap="large")

    with c1:
        shop_df = D.load_shop_type()
        fig_shop = go.Figure(go.Bar(
            x=shop_df["pct_of_trips"], y=shop_df["shop_type"],
            orientation="h",
            marker_color=[AFYA_BLUE, TEAL, ORANGE][:len(shop_df)],
            text=[f'{v}%' for v in shop_df["pct_of_trips"]],
            textposition="outside", cliponaxis=False,
            customdata=shop_df["basket_count"],
            hovertemplate="<b>%{y}</b><br>%{x}% of trips · %{customdata:,} transactions<extra></extra>",
        ))
        fig_shop.update_layout(**_CL, height=240)
        fig_shop.update_xaxes(**_ax(showgrid=False, showticklabels=False), range=[0, 60])
        fig_shop.update_yaxes(**_ax(showgrid=False))
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Is XanaLife a Primary Store?</div>', unsafe_allow_html=True)
        pc(fig_shop)

    with c2:
        xs_df = D.load_cross_sell()
        xs_df = apply_filters(xs_df)
        fig_xs = go.Figure()
        for cl, color in [("Katani", AFYA_BLUE), ("Syokimau", TEAL)]:
            d = xs_df[xs_df["cluster"].fillna("") == cl]
            if not d.empty:
                fig_xs.add_trace(go.Scatter(
                    x=d["month"], y=d["cross_sell_pct"].fillna(0), name=cl,
                    mode="lines+markers",
                    line=dict(color=color, width=2), marker=dict(size=5),
                    hovertemplate=f"<b>{cl}</b><br>%{{x|%b %Y}}: %{{y}}%<extra></extra>",
                ))
        fig_xs.update_layout(**_CL, height=240)
        fig_xs.update_xaxes(**AXIS, tickformat="%b")
        fig_xs.update_yaxes(**AXIS, ticksuffix="%")
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Pharmacy &amp; Supermarket Cross-Shopping Rate (%)</div>', unsafe_allow_html=True)
        pc(fig_xs)
        note("% of registered customers who visited both pharmacy and grocery in the same month.")

    gap(8)
    sh("Basket Size Analysis", mt=4)
    c1, c2 = st.columns(2, gap="large")

    with c1:
        bsz_df = D.load_basket_by_size()
        fig_bsz = go.Figure(go.Bar(
            x=bsz_df["basket_size_category"],
            y=bsz_df["pct_of_revenue"].fillna(0),
            marker_color=[GRAY, TEAL, AFYA_BLUE, COOL_BLUE][:len(bsz_df)],
            text=[f'{v}%' for v in bsz_df["pct_of_revenue"].fillna(0)],
            textposition="outside", cliponaxis=False, constraintext="none",
            textfont=dict(color=AFYA_BLUE, size=11, family="Montserrat"),
            customdata=bsz_df["avg_basket_value"].fillna(0),
            hovertemplate="<b>%{x}</b><br>%{y}% of revenue<br>Avg basket: KSh %{customdata:,.0f}<extra></extra>",
        ))
        fig_bsz.update_layout(**_CL, height=280)
        fig_bsz.update_xaxes(**_ax(showgrid=False))
        fig_bsz.update_yaxes(**AXIS, ticksuffix="%")
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Which Basket Size Drives Revenue?</div>', unsafe_allow_html=True)
        pc(fig_bsz)

    with c2:
        bst_df = D.load_basket_per_store()
        bst_df["store_name"] = bst_df["store_name"].fillna("Unknown")
        if excl_ws: bst_df = excl_wholesale(bst_df)
        bst_df = apply_filters(bst_df)
        _txn_max = bst_df["transactions"].fillna(0).max()
        palette = [AFYA_BLUE, TEAL, PURPLE, ORANGE, CORAL, GRAY, "#185FA5", COOL_BLUE]
        fig_bst = go.Figure()
        for i, row in bst_df.iterrows():
            fig_bst.add_trace(go.Scatter(
                x=[row["avg_items"]], y=[row["avg_basket_value"]],
                mode="markers+text",
                name=row["store_name"],
                marker=dict(
                    size=max(12, row["transactions"] / _txn_max * 60),
                    color=palette[i % len(palette)],
                    opacity=0.85,
                    line=dict(width=1, color="white"),
                ),
                text=[row["store_name"]],
                textposition="top center",
                textfont=dict(size=9, family="Montserrat", color=AFYA_BLUE),
                hovertemplate=(
                    f"<b>{row['store_name']}</b><br>"
                    "Avg items: %{x:.1f}<br>"
                    "Avg basket: KSh %{y:,.0f}<br>"
                    f"Transactions: {row['transactions']:,.0f}"
                    "<extra></extra>"
                ),
            ))
        fig_bst.update_layout(**_CL, height=370, showlegend=False)
        fig_bst.update_layout(margin=dict(t=10, b=40, l=0, r=10))
        fig_bst.update_xaxes(**_ax(title_text="Avg items per basket"))
        fig_bst.update_yaxes(**_ax(title_text="Avg basket value (KSh)", tickformat=",.0f"))
        note("Bubble size = number of transactions. Top-right = high-value, high-volume baskets.")
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Basket Profile by Store</div>', unsafe_allow_html=True)
        pc(fig_bst)

    gap(8)
    sh("When Do Customers Shop? — Day of Week by Cluster", mt=4)
    dow_df = D.load_dow_by_cluster()
    dow_df = apply_filters(dow_df)
    if excl_ws: dow_df = excl_wholesale(dow_df)

    # DAYOFWEEK in Snowflake: 0=Sunday, 1=Monday ... 6=Saturday
    day_map = {0:"Sunday",1:"Monday",2:"Tuesday",3:"Wednesday",
               4:"Thursday",5:"Friday",6:"Saturday"}
    day_order = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
    dow_df["day_name"] = dow_df["day_num"].map(day_map).fillna("Unknown")
    dow_df["day_name"] = pd.Categorical(dow_df["day_name"], categories=day_order, ordered=True)

    dow_agg = (dow_df.groupby(["day_name","cluster"], observed=True)
               .agg(transactions=("transaction_count","sum"),
                    patients=("unique_patients","sum"))
               .reset_index().sort_values("day_name"))

    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig_dow = go.Figure()
        for cl, color in [("Katani", AFYA_BLUE), ("Syokimau", TEAL)]:
            d = dow_agg[dow_agg["cluster"] == cl]
            if not d.empty:
                fig_dow.add_trace(go.Bar(
                    x=d["day_name"].astype(str), y=d["transactions"],
                    name=cl, marker_color=color, cliponaxis=False,
                    hovertemplate=f"<b>{cl}</b> — %{{x}}<br>Transactions: %{{y:,}}<extra></extra>",
                ))
        fig_dow.update_layout(**_CL, height=280, barmode="group")
        fig_dow.update_xaxes(**_ax(showgrid=False))
        fig_dow.update_yaxes(**AXIS, title_text="Transactions")
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Transactions by Day of Week</div>', unsafe_allow_html=True)
        pc(fig_dow)

    with c2:
        fig_dow2 = go.Figure()
        for cl, color in [("Katani", AFYA_BLUE), ("Syokimau", TEAL)]:
            d = dow_agg[dow_agg["cluster"] == cl]
            if not d.empty:
                fig_dow2.add_trace(go.Bar(
                    x=d["day_name"].astype(str), y=d["patients"],
                    name=cl, marker_color=color, cliponaxis=False,
                    hovertemplate=f"<b>{cl}</b> — %{{x}}<br>Unique patients: %{{y:,}}<extra></extra>",
                ))
        fig_dow2.update_layout(**_CL, height=280, barmode="group")
        fig_dow2.update_xaxes(**_ax(showgrid=False))
        fig_dow2.update_yaxes(**AXIS, title_text="Unique patients")
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Unique Customers by Day of Week</div>', unsafe_allow_html=True)
        pc(fig_dow2)
    note("Busiest day per cluster = highest-ROI window for marketing. "
         "Low-traffic days = opportunities for targeted incentives.")

    gap(8)
    sh("Where Is Retail Growth Coming From?", mt=4)
    st.markdown(
        f'<ul style="font-size:12px;color:#4A5568;line-height:2;margin-bottom:10px;padding-left:18px">'
        f'<li><b style="color:{TEAL}">Healthcare</b> — Pharmacy stores. Growth is need-driven: customers coming for medicine, potentially cross-shopping for groceries.</li>'
        f'<li><b style="color:{PURPLE}">Specialty / High Margin</b> — Wine and Deli stores. Growth signals premiumisation — customers buying higher-margin, discretionary items.</li>'
        f'<li><b style="color:{AFYA_BLUE}">Standard Retail</b> — All other stores (supermarkets, general grocery). The core volume driver; growth comes from everyday basket expansion or more frequent visits.</li>'
        f'</ul>',
        unsafe_allow_html=True)
    # import pdb;pdb.set_trace()
    rgs_df = D.load_retail_growth_segments()
    seg_palette = {"Healthcare": TEAL, "Specialty / High Margin": PURPLE, "Standard Retail": AFYA_BLUE}
    c1, c2 = st.columns(2, gap="large")

    with c1:
        fig_rg = go.Figure()
        for seg, grp in rgs_df.groupby("retail_segment"):
            grp = grp.sort_values("month")
            fig_rg.add_trace(go.Scatter(
                x=grp["month"], y=grp["revenue"],
                name=seg, mode="lines+markers",
                line=dict(color=seg_palette.get(seg, GRAY), width=2),
                marker=dict(size=5),
                fill="tozeroy", fillcolor=_rgba(seg_palette.get(seg, GRAY), 0.07),
                hovertemplate=f"<b>{seg}</b><br>%{{x|%b %Y}}<br>Revenue: KSh %{{y:,.0f}}<extra></extra>",
            ))
        fig_rg.update_layout(**_CL, height=300)
        fig_rg.update_layout(
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0, font=dict(size=9)),
            margin=dict(t=10, b=70, l=0, r=10))
        fig_rg.update_xaxes(**_ax(tickformat="%b %Y"))
        fig_rg.update_yaxes(**_ax(title_text="Revenue (KSh)", tickformat=",.0f"))
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Monthly Revenue by Retail Segment</div>', unsafe_allow_html=True)
        pc(fig_rg)

    with c2:
        rgs_recent = rgs_df.copy()
        rgs_recent["month"] = pd.to_datetime(rgs_recent["month"], errors="coerce")
        last_6 = sorted(rgs_recent["month"].dropna().unique())[-6:]
        rgs_recent = rgs_recent[rgs_recent["month"].isin(last_6)]
        fig_mom = go.Figure()
        for seg, clr in seg_palette.items():
            d = rgs_recent[rgs_recent["retail_segment"] == seg].sort_values("month")
            if d.empty:
                continue
            fig_mom.add_trace(go.Bar(
                name=seg, x=d["month"].dt.strftime("%b %Y"), y=d["revenue"].fillna(0),
                marker_color=clr,
                customdata=d[["rev_change_kes", "avg_basket_value", "transaction_count"]].fillna(0).values,
                hovertemplate=(
                    f"<b>{seg}</b><br>%{{x}}<br>Revenue: KSh %{{y:,.0f}}<br>"
                    "MoM: KSh %{customdata[0]:,.0f}<br>"
                    "Avg basket: KSh %{customdata[1]:,.0f}<br>"
                    "Transactions: %{customdata[2]:,.0f}<extra></extra>"
                ),
            ))
        fig_mom.update_layout(**_CL, height=300, barmode="group")
        fig_mom.update_layout(
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0, font=dict(size=9)),
            margin=dict(t=10, b=70, l=0, r=10))
        fig_mom.update_xaxes(**_ax(showgrid=False, tickfont=dict(size=9)))
        fig_mom.update_yaxes(**_ax(title_text="Revenue (KSh)", tickformat=",.0f"))
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">MoM Revenue Change by Segment — Last 6 Months</div>', unsafe_allow_html=True)
        pc(fig_mom)
        note("Grouped by retail segment. Hover for MoM change, avg basket, and transaction count.")

    gap(8); sh("Insights")
    ic1, ic2, ic3 = st.columns(3, gap="large")
    with ic1:
        insight("Growth", "Which stores are driving acquisition?",
                "Filter by cluster or business unit to isolate where growth is concentrated "
                "vs where acquisition has plateaued.", "green")
    with ic2:
        insight("Shop Type", "What % of trips are Full Shops?",
                "If <30% of trips are Full Shops, XanaLife is a secondary store. "
                "Full-basket promotions on staples can shift this.", "amber")
    with ic3:
        insight("New vs Returning", "Is growth healthy or a leaky bucket?",
                "Healthy = both bars growing. New grows but returning flat = "
                "acquiring and losing at the same rate.", "blue")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    sh("Segment Distribution & Revenue Concentration")
    st.markdown("""
<div class="nb">
<b>Tiers ranked by Daily Spend Intensity</b> (revenue ÷ days visited):
<ul style="margin:6px 0 0 0;padding-left:18px;line-height:1.9">
<li><b>Elite (Weekly Loyal)</b> — ≥ KSh 10,000/day AND ≥ 4 visits/month</li>
<li><b>Elite (Bulk/Wholesale)</b> — ≥ KSh 10,000/day but low visit frequency</li>
<li><b>High</b> — ≥ KSh 5,000/day</li>
<li><b>Medium</b> — ≥ KSh 1,000/day</li>
<li><b>Low</b> — below KSh 1,000/day · largest segment by count</li>
<li><b>One-Time</b> — single purchase only · Daily Spend Intensity cannot be calculated</li>
</ul>
</div>
""", unsafe_allow_html=True)
    seg_df = D.load_segments()
    seg_df = seg_df.fillna(0)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig_sc = go.Figure(go.Bar(
            x=seg_df["customer_count"], y=seg_df["refined_tier"],
            orientation="h",
            marker_color=[seg_color(t) for t in seg_df["refined_tier"]],
            text=seg_df["customer_count"], textposition="outside", cliponaxis=False,
            customdata=seg_df["pct_revenue"],
            hovertemplate="<b>%{y}</b><br>%{x} customers · %{customdata}% of revenue<extra></extra>",
        ))
        fig_sc.update_layout(**_CL, height=320)
        fig_sc.update_xaxes(**_ax(showgrid=False, showticklabels=False))
        fig_sc.update_yaxes(**_ax(showgrid=False))
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Customers per Segment</div>', unsafe_allow_html=True)
        pc(fig_sc)

    with c2:
        fig_rc = go.Figure(go.Bar(
            x=seg_df["refined_tier"], y=seg_df["pct_revenue"].fillna(0),
            marker_color=[seg_color(t) for t in seg_df["refined_tier"]],
            text=[f'{v}%' for v in seg_df["pct_revenue"].fillna(0)],
            textposition="outside", cliponaxis=False,
            hovertemplate="<b>%{x}</b><br>%{y}% of revenue<extra></extra>",
        ))
        fig_rc.update_layout(**_CL, height=320)
        fig_rc.update_xaxes(**_ax(showgrid=False, tickfont=dict(size=8)))
        fig_rc.update_yaxes(**AXIS, ticksuffix="%", range=[0, 45])
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Revenue Concentration by Segment</div>', unsafe_allow_html=True)
        pc(fig_rc)

    gap(8); sh("Purchasing Characteristics", mt=4)

    action_map = {
        "1 - Elite (Weekly Loyal)":    "VIP tier — retain aggressively",
        "1 - Elite (Bulk/Wholesale)":  "Relationship mgmt — key accounts",
        "2 - High":                    "Re-engage — bring back sooner",
        "3 - Medium":                  "Upsell — bigger baskets",
        "4 - Low":                     "Build habit — loyalty enrolment",
        "0 - One Time":                "Win-back — first-purchase category as hook",
    }
    _seg_rows = seg_df["refined_tier"].tolist()
    _row_fills = [seg_color(t) for t in _seg_rows]
    _row_fills_light = [_rgba(c, 0.18) for c in _row_fills]
    fig_chars = plotly_table(
        header_vals=["Segment","Customers","Total Revenue","Avg Basket",
                     "Visits","Daily Spend Intensity","% Revenue","Action"],
        cell_vals=[
            seg_df["refined_tier"].fillna("—"),
            seg_df["customer_count"].astype(int),
            [fmt_ksh(v) for v in seg_df["total_revenue"]],
            [fmt_ksh(v) for v in seg_df["avg_basket_value"]],
            [f'{v:.1f}' for v in seg_df["avg_visits"]],
            [fmt_ksh(v) + "/day" for v in seg_df["avg_daily_spend_intensity"]],
            [f'{v}%' for v in seg_df["pct_revenue"]],
            [action_map.get(t, "—") for t in _seg_rows],
        ],
        col_widths=[160, 80, 110, 100, 60, 155, 80, 220],
        row_fill=[_row_fills_light],
        header_color=AFYA_BLUE,
        cell_font_color=COOL_BLUE,
        height=240,
    )
    pc(fig_chars)
    note("'Visits' = avg distinct days transacted. 'Daily Spend Intensity' = CLV Velocity in plain terms.")

    gap(8); sh("Shopping Frequency — Heartbeat per Segment", mt=4)
    st.markdown("""
<div class="nb">
<ul style="margin:0;padding-left:18px;line-height:1.9">
<li><b>Heartbeat</b> — avg days between a customer's consecutive visits. A 7-day heartbeat = expected back weekly.</li>
<li><b>Lapsing</b> — absent for more than 2× their heartbeat. Not yet churned but overdue for a visit.</li>
</ul>
</div>
""", unsafe_allow_html=True)
    hb_df = D.load_heartbeat()
    hb_df["avg_heartbeat_days"] = pd.to_numeric(
        hb_df["avg_heartbeat_days"], errors="coerce").fillna(0)

    fig_hb = go.Figure(go.Bar(
        x=hb_df["refined_tier"].fillna("Unknown"),
        y=hb_df["avg_heartbeat_days"],
        marker_color=[seg_color(t) for t in hb_df["refined_tier"].fillna("")],
        text=[safe_str(v, "d") for v in hb_df["avg_heartbeat_days"]],
        textposition="outside", cliponaxis=False,
        customdata=(hb_df["avg_heartbeat_days"] * 2).round(0),
        hovertemplate="<b>%{x}</b><br>Avg gap: %{y} days<br>"
                      "Lapsing if absent > %{customdata} days<extra></extra>",
    ))
    fig_hb.update_layout(**_CL, height=280)
    fig_hb.update_xaxes(**_ax(showgrid=False, tickfont=dict(size=9)))
    fig_hb.update_yaxes(**AXIS, title_text="Avg days between visits")
    st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Average Days Between Visits (Heartbeat) per Segment</div>', unsafe_allow_html=True)
    pc(fig_hb)
    # note("Lapsing threshold = 2× heartbeat — not a fixed 60-day cutoff.")

    gap(8)
    sh("Spend Trajectory — Are Customers Growing or Slipping?", mt=4)
    cv_df = D.load_conversion_velocity()
    cv_df = cv_df.fillna(0)
    cv_colors = {"Up-Converting": TEAL, "Stable": AFYA_BLUE, "Down-Converting": CORAL}

    _cv_texts = [
        f"{int(n)}<br>{'▲' if v >= 0 else '▼'} KSh {abs(int(v)):,}/day"
        for n, v in zip(cv_df["customer_count"].fillna(0), cv_df["avg_spend_shift_kes"].fillna(0))
    ]

    _cv1, _cv2 = st.columns(2, gap="large")
    with _cv1:
        fig_cv = go.Figure(go.Bar(
            x=cv_df["conversion_status"].fillna("Unknown"),
            y=cv_df["customer_count"].fillna(0),
            marker_color=[cv_colors.get(s, GRAY) for s in cv_df["conversion_status"].fillna("")],
            text=_cv_texts,
            textposition="outside", cliponaxis=False,
            textfont=dict(size=10, family="Montserrat", color=AFYA_BLUE),
            hovertemplate="<b>%{x}</b><br>%{y} customers<extra></extra>",
        ))
        fig_cv.update_layout(**_CL, height=300)
        fig_cv.update_xaxes(**_ax(showgrid=False))
        fig_cv.update_yaxes(**_ax(showgrid=False, showticklabels=False))
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">How Many Customers Are Up vs Down?</div>', unsafe_allow_html=True)
        pc(fig_cv)

    with _cv2:
        cvt_df = D.load_conversion_velocity_by_tier()
        cvt_df = cvt_df.fillna(0)
        cvt_df = cvt_df[cvt_df["refined_tier"].astype(str).str.strip() != ""]
        cvt_df = cvt_df[~cvt_df["refined_tier"].astype(str).str.startswith("0 -")]
        _tier_order = [t for t in SEG_COLORS.keys() if t in cvt_df["refined_tier"].values]
        _tier_order += [t for t in cvt_df["refined_tier"].unique() if t not in _tier_order]
        fig_cvt = go.Figure()
        for status, color in cv_colors.items():
            _d = cvt_df[cvt_df["conversion_status"] == status]
            _d = _d.set_index("refined_tier").reindex(_tier_order).reset_index()
            fig_cvt.add_trace(go.Bar(
                name=status,
                x=_d["refined_tier"],
                y=_d["customer_count"].fillna(0),
                marker_color=color,
                customdata=_d["customer_count"].fillna(0),
                hovertemplate=f"<b>%{{x}}</b><br>{status}: %{{customdata:.0f}} customers (%{{y:.1f}}%)<extra></extra>",
            ))
        fig_cvt.update_layout(**_CL, height=300, barmode="stack", barnorm="percent")
        fig_cvt.update_xaxes(**_ax(showgrid=False, tickfont=dict(size=9)))
        fig_cvt.update_yaxes(**_ax(showgrid=False), ticksuffix="%", range=[0, 100])
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Which Segments Are Slipping?</div>', unsafe_allow_html=True)
        pc(fig_cvt)

    note("Up-Converting = spending 20%+ more per day now vs when they started. "
         "Down-Converting = spending 20%+ less — still shopping but intensity is dropping. "
         "Only customers with ≥4 visits are included — enough history to show movement.")

    gap(8); sh("Segment Growth Over Time + Top Products per Segment", mt=4)
    c1, c2 = st.columns(2, gap="large")

    with c1:
        seg_trend = D.load_seg_trend()
        seg_trend["month"] = pd.to_datetime(seg_trend["month"], errors="coerce")
        fig_st = go.Figure()
        _st_tiers_present = seg_trend["refined_tier"].fillna("Unknown").unique().tolist()
        _st_order = [t for t in SEG_COLORS.keys() if t in _st_tiers_present]
        _st_order += [t for t in _st_tiers_present if t not in _st_order]
        for tier in _st_order:
            d = seg_trend[seg_trend["refined_tier"] == tier]
            fig_st.add_trace(go.Scatter(
                x=d["month"], y=d["customer_count"].fillna(0),
                name=tier, mode="lines", stackgroup="one",
                line=dict(color=seg_color(tier), width=0.5),
                fillcolor=_rgba(seg_color(tier), 0.5),
                hovertemplate=f"<b>{tier}</b>: %{{y}} customers<extra></extra>",
            ))
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Segment Growth — Stacked Area</div>', unsafe_allow_html=True)
        fig_st.update_layout(**_CL, height=300)
        fig_st.update_xaxes(**AXIS, tickformat="%b %Y")
        fig_st.update_yaxes(**AXIS, title_text="Customers")
        pc(fig_st)

    with c2:
        tp_df = D.load_top_products_seg()
        tiers_avail = tp_df["refined_tier"].dropna().unique().tolist()
        sel_tier = st.selectbox("Select segment", tiers_avail, key="tier_sel")
        tp_filt = tp_df[tp_df["refined_tier"] == sel_tier]
        fig_tp = go.Figure(go.Bar(
            x=tp_filt["total_spend"].fillna(0),
            y=tp_filt["product_name"].fillna("Unknown"),
            orientation="h", marker_color=seg_color(sel_tier),
            text=[fmt_ksh(v) for v in tp_filt["total_spend"].fillna(0)],
            textposition="outside", cliponaxis=False,
        ))
        fig_tp.update_layout(**_CL, height=300)
        fig_tp.update_xaxes(**_ax(showgrid=False, showticklabels=False))
        fig_tp.update_yaxes(**_ax(showgrid=False, tickfont=dict(size=9)))
        st.markdown(f'<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Top 5 Products — {sel_tier}</div>', unsafe_allow_html=True)
        pc(fig_tp)

    gap(8); sh("First Purchase Category — What Brought Them to XanaLife?", mt=4)
    fcat_df = D.load_first_category()
    fcat_df = fcat_df.fillna(0)
    fevo_df = D.load_first_cat_evolution()
    fevo_df = fevo_df.fillna(0)

    fcat_top10 = fcat_df.nlargest(10, "customer_count").copy()
    fcat_top10["first_category"] = fcat_top10["first_category"].replace(0, "Unknown")
    _pal10 = (SEQ * 3)[:len(fcat_top10)]

    fig_fc = go.Figure(go.Bar(
        y=fcat_top10["first_category"],
        x=fcat_top10["customer_count"],
        orientation="h",
        marker_color=_pal10,
        text=fcat_top10["customer_count"].astype(int),
        textposition="outside", cliponaxis=False, constraintext="none",
        textfont=dict(size=11, family="Montserrat", color=AFYA_BLUE),
        hovertemplate="<b>%{y}</b><br>%{x:,} first-time customers<extra></extra>",
    ))
    fig_fc.update_layout(**_CL, height=370)
    fig_fc.update_xaxes(**_ax(showgrid=False, showticklabels=False),
        range=[0, fcat_top10["customer_count"].max() * 1.3])
    fig_fc.update_yaxes(**_ax(showgrid=False, tickfont=dict(size=10)),
        categoryorder="total ascending")

    col_fc, col_ret = st.columns(2, gap="large")
    with col_fc:
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">1. What Product Brought Them to XanaLife for the First Time?</div>', unsafe_allow_html=True)
        pc(fig_fc)

    with col_ret:
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">2. Did They Come Back? — Retention by First Product</div>', unsafe_allow_html=True)

    _ret = fcat_df.copy()
    _ret["first_category"] = _ret["first_category"].replace(0, "Unknown")
    for _c in ["regular_customers", "returning_customers", "one_time_customers"]:
        if _c not in _ret.columns:
            _ret[_c] = 0
    _ret["retention_rate"] = (
        (_ret["regular_customers"] + _ret["returning_customers"])
        / _ret["customer_count"].replace(0, pd.NA) * 100
    ).fillna(0).round(1)
    _ret["pct_regular"]  = (_ret["regular_customers"]  / _ret["customer_count"].replace(0, pd.NA) * 100).fillna(0).round(1)
    _ret["pct_one_time"] = (_ret["one_time_customers"]  / _ret["customer_count"].replace(0, pd.NA) * 100).fillna(0).round(1)

    for _c in ["regular_customers", "returning_customers", "one_time_customers", "customer_count"]:
        _ret[_c] = pd.to_numeric(_ret.get(_c, 0), errors="coerce").fillna(0).astype(int)
    ret_display = pd.DataFrame({
        "First Category":    _ret["first_category"].reset_index(drop=True),
        "Customers":         _ret["customer_count"].reset_index(drop=True),
        "Regular ≥6 visits": _ret["regular_customers"].reset_index(drop=True),
        "Regular %":         (_ret["pct_regular"].astype(str) + "%").reset_index(drop=True),
        "Returning (2–5)":   _ret["returning_customers"].reset_index(drop=True),
        "One-Time":          _ret["one_time_customers"].reset_index(drop=True),
        "One-Time %":        (_ret["pct_one_time"].astype(str) + "%").reset_index(drop=True),
        "Retention Rate":    (_ret["retention_rate"].astype(str) + "%").reset_index(drop=True),
    })

    def _style_ret_df(df):
        style = pd.DataFrame("", index=df.index, columns=df.columns)
        for i, v_str in enumerate(df["Retention Rate"]):
            try:
                v = float(str(v_str).replace("%", ""))
                style.loc[i, "Retention Rate"] = (
                    "background-color: #e8f5e9" if v >= 70 else
                    "background-color: #fff8e1" if v >= 40 else
                    "background-color: #fdecea"
                )
            except: pass
        for i, v_str in enumerate(df["One-Time %"]):
            try:
                v = float(str(v_str).replace("%", ""))
                style.loc[i, "One-Time %"] = (
                    "background-color: #fdecea" if v >= 60 else
                    "background-color: #fff8e1" if v >= 30 else ""
                )
            except: pass
        return style

    with col_ret:
        st.dataframe(ret_display.style.apply(_style_ret_df, axis=None), width='stretch', hide_index=True)

    if not fevo_df.empty:
        gap(4)
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">3. Did Their Basket Grow? — Basket Evolution for Returning Customers</div>', unsafe_allow_html=True)
        _ev = fevo_df.copy()
        _ev["first_category"] = _ev["first_category"].replace(0, "Unknown")

        _sig_bg = {
            "Exploring — primary store": "#d4f5f1",
            "Growing basket":            "#d0e8f9",
            "Slight growth":             "#fff3dc",
        }

        def _style_evo_df(df):
            style = pd.DataFrame("", index=df.index, columns=df.columns)
            for i, v_str in enumerate(df["Value Change %"]):
                try:
                    v = float(str(v_str).replace("%", "").replace("+", ""))
                    style.loc[i, "Value Change %"] = (
                        "background-color: #e8f5e9" if v > 10 else
                        "background-color: #fff8e1" if v >= 0 else
                        "background-color: #fdecea"
                    )
                except: pass
            for i, sig in enumerate(df["Signal"]):
                style.loc[i, "Signal"] = f"background-color: {_sig_bg.get(sig, '#fde8e8')}"
            return style

        evo_display = pd.DataFrame({
            "First Category":       _ev["first_category"].reset_index(drop=True),
            "Customers":            _ev["customer_count"].astype(int).reset_index(drop=True),
            "First Visit Value":    _ev["first_visit_value"].apply(lambda v: f"KSh {v:,.0f}").reset_index(drop=True),
            "Recent Visit Value":   _ev["recent_visit_value"].apply(lambda v: f"KSh {v:,.0f}").reset_index(drop=True),
            "Value Change %":       _ev["value_change_pct"].apply(lambda v: f"{int(v):+}%").reset_index(drop=True),
            "New Categories Added": _ev["new_categories_added"].round(1).reset_index(drop=True),
            "Signal":               _ev["signal"].reset_index(drop=True),
        })
        st.dataframe(evo_display.style.apply(_style_evo_df, axis=None), width='stretch', hide_index=True)

    gap(8); sh("How Have Purchases Changed? — Early vs Recent Basket", mt=4)
    bev_df = D.load_basket_evolution()
    for c in ["avg_diversity_change","avg_basket_growth","avg_early_basket",
              "avg_recent_basket","customer_count","expanding","shrinking"]:
        if c in bev_df.columns:
            bev_df[c] = pd.to_numeric(bev_df[c], errors="coerce").fillna(0)

    fig_bev = go.Figure()
    fig_bev.add_trace(go.Bar(
        name="Early basket (visits 1–3)",
        x=bev_df["refined_tier"].fillna("Unknown"),
        y=bev_df["avg_early_basket"], marker_color=GRAY,
        hovertemplate="<b>%{x}</b><br>Early avg basket: KSh %{y:,.0f}<extra></extra>",
    ))
    fig_bev.add_trace(go.Bar(
        name="Recent basket (last 3 visits)",
        x=bev_df["refined_tier"].fillna("Unknown"),
        y=bev_df["avg_recent_basket"], marker_color=AFYA_BLUE,
        hovertemplate="<b>%{x}</b><br>Recent avg basket: KSh %{y:,.0f}<extra></extra>",
    ))
    fig_bev.update_layout(**_CL, height=300, barmode="group")
    fig_bev.update_xaxes(**_ax(showgrid=False, tickfont=dict(size=9)))
    fig_bev.update_yaxes(**AXIS, tickformat=",.0f", tickprefix="KSh ")
    st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Early vs Recent Basket Value — Has Spending Changed?</div>', unsafe_allow_html=True)
    pc(fig_bev)

    with st.expander("📊 Detailed evolution table — Expanding vs Shrinking customers"):
        bev_display = pd.DataFrame({
            "Segment":                   bev_df["refined_tier"].fillna("Unknown"),
            "Customers":                 bev_df["customer_count"].astype(int),
            "Category Diversity Change": [f'+{v}' if v >= 0 else str(v)
                                          for v in bev_df["avg_diversity_change"]],
            "Basket Growth":             [fmt_ksh(v) for v in bev_df["avg_basket_growth"]],
            "Expanding":                 bev_df["expanding"].astype(int),
            "Shrinking":                 bev_df["shrinking"].astype(int),
        })
        st.dataframe(bev_display, width='stretch', hide_index=True)

    gap(8); sh("Payment Method by Segment", mt=4)
    pay_df = D.load_payment_by_segment()
    pay_df = pay_df.fillna(0)

    pay_methods = ["cash_count","mpesa_count","card_count","jambopay_count",
                   "pesapal_card_count","pesapal_mpesa_count",
                   "voucher_count","giftcard_count"]
    pay_labels  = ["Cash","M-Pesa","Card","JamboPay","PesaPal Card",
                   "PesaPal M-Pesa","Voucher","Gift Card"]
    pay_colors  = [GRAY,TEAL,AFYA_BLUE,PURPLE,ORANGE,CORAL,
                   "#1D9E75","#534AB7","#3B6D11","#854F0B"]

    fig_pay = go.Figure()
    for col, label, color in zip(pay_methods, pay_labels, pay_colors):
        if col in pay_df.columns:
            totals = pay_df["total_transactions"].replace(0, np.nan)
            pct = (pay_df[col] / totals * 100).fillna(0).round(1)
            fig_pay.add_trace(go.Bar(
                name=label, x=pay_df["refined_tier"].fillna("Unregistered"), y=pct,
                marker_color=color,
                hovertemplate=f"<b>{label}</b><br>%{{x}}: %{{y}}%<extra></extra>",
            ))
    fig_pay.update_layout(**_CL, height=360, barmode="stack")
    fig_pay.update_layout(legend=dict(orientation="v", yanchor="top", y=1,
                    xanchor="left", x=1.02, font=dict(size=9)))
    fig_pay.update_xaxes(**_ax(showgrid=False, tickfont=dict(size=9)))
    fig_pay.update_yaxes(**AXIS, ticksuffix="%", range=[0, 105])
    st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Payment Method Mix — % of Transactions per Segment</div>', unsafe_allow_html=True)
    pc(fig_pay)
    note("M-Pesa dominance in lower segments is expected. "
         "High loyalty_count in Elite confirms the programme reaches your most valuable customers.")

    gap(8); sh("Insights")
    ic1, ic2 = st.columns(2, gap="large")
    with ic1:
        insight("Priority Action",
                "Elite + High: shrinking AND lapsing = highest priority",
                "Cross-reference basket evolution with lapsing segment table in Retention. "
                "<strong>Contact this week.</strong>", "red")
    with ic2:
        insight("Growth Opportunity",
                "Low is the largest segment — and the most improvable",
                "Enrol them in loyalty from day 1 — "
                "loyalty members return 2.2× faster and spend more per visit.", "amber")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3
# ══════════════════════════════════════════════════════════════════════════════
with tab3:

    sh("Return Window — How Quickly Do Customers Come Back?")
    ret_win = D.load_return_window()

    # One-Time excluded — no return window by definition
    tiers_for_funnel = [t for t in ret_win["refined_tier"].dropna().unique().tolist()
                        if t != "0 - One Time"]
    funnel_seg = st.selectbox(
        "Select segment for funnel",
        ["All"] + tiers_for_funnel,
        key="funnel_sel",
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        if funnel_seg == "All":
            fv_df = ret_win[ret_win["refined_tier"] != "0 - One Time"]
            fv = fv_df[["within_30d","within_60d","within_90d","over_90d"]].sum().tolist()
        else:
            row = ret_win[ret_win["refined_tier"] == funnel_seg]
            fv = [0,0,0,0] if row.empty else [
                row.iloc[0]["within_30d"], row.iloc[0]["within_60d"],
                row.iloc[0]["within_90d"], row.iloc[0]["over_90d"]]

        fig_funnel = go.Figure(go.Funnel(
            y=["Returned < 30 days","Returned 31–60 days",
               "Returned 61–90 days","Last seen 90d+ ago"],
            x=fv,
            textinfo="value+percent initial",
            textfont=dict(family="Montserrat", size=11, color="white"),
            marker=dict(color=[TEAL, AFYA_BLUE, ORANGE, CORAL],
                        line=dict(width=1, color="white")),
            connector=dict(line=dict(color=BORDER, width=2, dash="dot")),
            hovertemplate="<b>%{y}</b><br>%{x} customers · %{percentInitial}<extra></extra>",
        ))
        fig_funnel.update_layout(**_CL, height=340)
        st.markdown(f'<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Return Window — {funnel_seg}</div>', unsafe_allow_html=True)
        pc(fig_funnel)
        note("One-Time customers excluded — they have no return window. "
             "Heavy bottom band = chronic lapsers needing a win-back campaign.")

    with c2:
        sh("Churn Status per Segment")
        st.markdown("""
<div class="nb">
<b>Churn Proxy Buckets</b> — rule-based thresholds (a statistical model needs 12+ months of data):
<ul style="margin:6px 0 0 0;padding-left:18px;line-height:1.9">
<li><b>Active</b> — last visit within 30 days</li>
<li><b>At Risk</b> — 31–60 days since last visit</li>
<li><b>Lapsed</b> — 61–90 days since last visit</li>
<li><b>Lost</b> — 90+ days since last visit</li>
</ul>
</div>
""", unsafe_allow_html=True)
        churn_df = D.load_churn_by_segment()
        churn_df = churn_df.fillna(0)
        _n = len(churn_df)
        _active_fill  = [_rgba("#276749", 0.15)] * _n
        _atrisk_fill  = [_rgba("#D97706", 0.15)] * _n
        _lapsed_fill  = [_rgba("#C05621", 0.15)] * _n
        _lost_fill    = [_rgba("#C53030", 0.15)] * _n
        _seg_fill     = [_rgba(seg_color(t), 0.15) for t in churn_df["refined_tier"].fillna("")]
        fig_churn = go.Figure(go.Table(
            columnwidth=[160, 85, 100, 95, 80, 65],
            header=dict(
                values=["Segment","Active<br>(<30d)",
                        "At Risk<br>(31–60d)","Lapsed<br>(61–90d)","Lost<br>(90d+)","Total"],
                fill_color=AFYA_BLUE,
                font=dict(color="white", size=10, family="Montserrat"),
                align="left", height=28),
            cells=dict(
                values=[
                    churn_df["refined_tier"].fillna("Unknown"),
                    churn_df["active"].astype(int),
                    churn_df["at_risk"].astype(int),
                    churn_df["lapsed"].astype(int),
                    churn_df["lost"].astype(int),
                    churn_df["total"].astype(int),
                ],
                fill_color=[_seg_fill, _active_fill, _atrisk_fill,
                            _lapsed_fill, _lost_fill, [BG_LIGHT]*_n],
                font=dict(color=COOL_BLUE, size=10, family="Montserrat"),
                align="left", height=26),
        ))
        fig_churn.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=240)
        pc(fig_churn)

    gap(8)
    sp_df = D.load_second_purchase()
    sp_df = sp_df.fillna(0)
    sp_ret = sp_df[~sp_df["refined_tier"].astype(str).str.startswith("0 -")].copy()

    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig_sp = go.Figure(go.Bar(
            x=sp_ret["refined_tier"].fillna("Unknown"),
            y=sp_ret["avg_days_to_second"].fillna(0),
            marker_color=[seg_color(t) for t in sp_ret["refined_tier"].fillna("")],
            text=[safe_str(v, "d") for v in sp_ret["avg_days_to_second"].fillna(0)],
            textposition="outside", cliponaxis=False,
            hovertemplate="<b>%{x}</b><br>Avg %{y} days to second visit<extra></extra>",
        ))
        fig_sp.update_layout(**_CL, height=300)
        fig_sp.update_xaxes(**_ax(showgrid=False, tickfont=dict(size=9)))
        fig_sp.update_yaxes(**_ax(showgrid=False), title_text="Days")
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Avg Days from First to Second Purchase</div>', unsafe_allow_html=True)
        pc(fig_sp)
        note("Prompt a second visit within this window = significantly more likely to become regular.")

    with c2:
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Win-Back Window + Recommended Action per Segment</div>', unsafe_allow_html=True)
        lap_merge = D.load_lapsing_by_segment().fillna(0)
        prompt_map = {
            "1 - Elite (Weekly Loyal)":    "Personalised restock reminder for their top category",
            "1 - Elite (Bulk/Wholesale)":  "Account manager outreach — check on order needs",
            "2 - High":                    "Loyalty points reminder + top category offer",
            "3 - Medium":                  "Discount on first-purchase category",
            "4 - Low":                     "Enrol in loyalty + new product in their category",
            "0 - One Time":                "Win-back offer using first-purchase product as hook",
        }
        action_rows = []
        for _, row in sp_df.iterrows():
            t = row["refined_tier"]
            lap_row = lap_merge[lap_merge["refined_tier"] == t]
            action_rows.append({
                "Segment":            t,
                "Win-back window":    f"{row['avg_days_to_second']:.0f} days",
                "Lapsing customers":  int(lap_row["lapsing_customers"].iloc[0]) if not lap_row.empty else 0,
                "Revenue at risk":    fmt_ksh(lap_row["revenue_at_risk"].iloc[0]) if not lap_row.empty else "—",
                "Recommended prompt": prompt_map.get(t, "—"),
            })
        st.dataframe(pd.DataFrame(action_rows), width='stretch', hide_index=True)

    gap(8)
    sh("Shopping Rhythm — How Reliably Do Customers Visit?", mt=4)
    cons_df = D.load_consistency_segments()
    cons_df = cons_df.fillna(0)
    rhythm_colors = {"Weekly": TEAL, "Bi-Weekly": AFYA_BLUE, "Monthly": PURPLE,
                     "Sporadic": ORANGE, "One-Time": CORAL}

    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig_cons = go.Figure(go.Bar(
            x=cons_df["shopping_rhythm"].fillna("Unknown"),
            y=cons_df["customer_count"].fillna(0),
            marker_color=[rhythm_colors.get(r, GRAY) for r in cons_df["shopping_rhythm"].fillna("")],
            text=cons_df["customer_count"].fillna(0).astype(int),
            textposition="outside", cliponaxis=False,
            customdata=cons_df["avg_days_between_visits"],
            hovertemplate="<b>%{x}</b><br>%{y} customers<br>Avg gap: %{customdata} days<extra></extra>",
        ))
        fig_cons.update_layout(**_CL, height=280)
        fig_cons.update_xaxes(**_ax(showgrid=False, tickfont=dict(size=9)))
        fig_cons.update_yaxes(**_ax(showgrid=False, showticklabels=False))
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Customers by Shopping Rhythm</div>', unsafe_allow_html=True)
        pc(fig_cons)

    with c2:
        cons_display = pd.DataFrame({
            "Rhythm":           cons_df["shopping_rhythm"].fillna("Unknown"),
            "Customers":        cons_df["customer_count"].astype(int),
            "Avg Basket":       [fmt_ksh(v) for v in cons_df["avg_basket_value"]],
            "Avg Lifetime Rev": [fmt_ksh(v) for v in cons_df["avg_lifetime_revenue"]],
            "Avg Days Between": [f'{v}d' for v in cons_df["avg_days_between_visits"]],
        })
        st.dataframe(cons_display, width='stretch', hide_index=True, height=280)
    st.markdown("""
<div class="nb">
<b>Shopping rhythm is classified by avg days between visits:</b>
<ul style="margin:6px 0 0 0;padding-left:18px;line-height:1.9">
<li><b>Weekly</b> — avg gap ≤ 7 days · highest lifetime revenue · most loyal segment</li>
<li><b>Bi-Weekly</b> — avg gap ≤ 14 days · regular but less frequent</li>
<li><b>Monthly</b> — avg gap ≤ 30 days · occasional shoppers</li>
<li><b>Sporadic</b> — avg gap > 30 days · unpredictable visit pattern</li>
<li><b>One-Time</b> — only one visit on record · no rhythm can be calculated</li>
</ul>
</div>
""", unsafe_allow_html=True)

    gap(8)
    sh("One-Time Customer Analysis — Why Didn't They Return?", mt=4)
    note("One-time rates of 40–60% are normal for supermarkets. Separate internal failures "
         "(fixable) from customers who were never the right fit (external).")

    c1, c2, c3, c4 = st.columns(4, gap="large")

    with c1:
        st.markdown(f'<div style="font-size:10px;font-weight:600;color:{AFYA_BLUE};'
                    f'margin-bottom:6px">What did they buy?</div>', unsafe_allow_html=True)
        ot_basket = D.load_onetimer_basket_type().fillna(0)
        ot_basket["label"] = ot_basket["shop_type"].str.replace(r'^\d\. ', '', regex=True)
        fig_otb = go.Figure(go.Bar(
            x=ot_basket["label"], y=ot_basket["pct_of_one_timers"],
            marker_color=[AFYA_BLUE, TEAL, ORANGE][:len(ot_basket)],
            text=[f'{v}%' for v in ot_basket["pct_of_one_timers"]],
            textposition="outside", cliponaxis=False,
            customdata=ot_basket["avg_basket_value"],
            hovertemplate="<b>%{x}</b><br>%{y}% · avg KSh %{customdata:,.0f}<extra></extra>",
        ))
        fig_otb.update_layout(**_CL, height=240)
        fig_otb.update_xaxes(**_ax(showgrid=False, tickfont=dict(size=8)))
        fig_otb.update_yaxes(**_ax(ticksuffix="%"), range=[0, 100])
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Shop Type — One-Timers</div>', unsafe_allow_html=True)
        pc(fig_otb)
        note("If >60% Top-Up: first visit was low-engagement — offer a welcome bundle at checkout to drive a second trip.", warn=True)

    with c2:
        st.markdown(f'<div style="font-size:10px;font-weight:600;color:{AFYA_BLUE};'
                    f'margin-bottom:6px">Where did they enter?</div>', unsafe_allow_html=True)
        ot_pharm = D.load_onetimer_pharmacy_split().fillna(0)
        fig_otp = go.Figure(go.Bar(
            x=ot_pharm["entry_type"], y=ot_pharm["pct_of_one_timers"],
            marker_color=[TEAL, AFYA_BLUE, PURPLE][:len(ot_pharm)],
            text=[f'{v}%' for v in ot_pharm["pct_of_one_timers"]],
            textposition="outside", cliponaxis=False,
            hovertemplate="<b>%{x}</b><br>%{y}%<extra></extra>",
        ))
        fig_otp.update_layout(**_CL, height=240)
        fig_otp.update_xaxes(**_ax(showgrid=False, tickfont=dict(size=8)))
        fig_otp.update_yaxes(**_ax(ticksuffix="%"), range=[0, 100])
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Entry Point</div>', unsafe_allow_html=True)
        pc(fig_otp)
        note("Pharmacy-only entry = high-conversion target. A grocery discount on the receipt gives them a reason to explore the rest of the store.", warn=True)

    with c3:
        st.markdown(f'<div style="font-size:10px;font-weight:600;color:{AFYA_BLUE};'
                    f'margin-bottom:6px">14-Day Golden Window</div>', unsafe_allow_html=True)
        ot_urg = D.load_onetimer_urgency().fillna(0)
        ot_urg["label"] = ot_urg["urgency_bucket"].str.replace(r'^\d\. ', '', regex=True)
        fig_otu = go.Figure(go.Bar(
            x=ot_urg["label"], y=ot_urg["pct_of_one_timers"],
            marker_color=[TEAL, AFYA_BLUE, ORANGE, CORAL][:len(ot_urg)],
            text=[f'{v}%' for v in ot_urg["pct_of_one_timers"]],
            textposition="outside", cliponaxis=False,
            hovertemplate="<b>%{x}</b><br>%{y}%<extra></extra>",
        ))
        fig_otu.update_layout(**_CL, height=240)
        fig_otu.update_xaxes(**_ax(showgrid=False, tickfont=dict(size=7)))
        fig_otu.update_yaxes(**_ax(ticksuffix="%"), range=[0, 100])
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">14-Day Golden Window — Urgency Buckets</div>', unsafe_allow_html=True)
        pc(fig_otu)
        note("'Window open' = contact within 7 days with an offer tied to their first purchase category.", warn=True)

    with c4:
        st.markdown(f'<div style="font-size:10px;font-weight:600;color:{AFYA_BLUE};'
                    f'margin-bottom:6px">Shopper profile</div>', unsafe_allow_html=True)
        ot_ps = D.load_onetimer_price_sensitive().fillna(0)
        fig_otps = go.Figure(go.Bar(
            x=ot_ps["customer_profile"], y=ot_ps["pct"],
            marker_color=[CORAL, ORANGE, TEAL][:len(ot_ps)],
            text=[f'{v}%' for v in ot_ps["pct"]],
            textposition="outside", cliponaxis=False,
            hovertemplate="<b>%{x}</b><br>%{y}%<extra></extra>",
        ))
        fig_otps.update_layout(**_CL, height=240)
        fig_otps.update_xaxes(**_ax(showgrid=False, tickfont=dict(size=7)))
        fig_otps.update_yaxes(**_ax(ticksuffix="%"), range=[0, 100])
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Shopper Profile</div>', unsafe_allow_html=True)
        pc(fig_otps)
        note("Staples-only = price cherry-picker. Loyalty won't work.", warn=True)

    gap(8); sh("Insights")
    ic1, ic2, ic3 = st.columns(3, gap="large")
    with ic1:
        insight("Win-Back Window", "Contact within the second-purchase window",
                "Each segment has a different window. Acting within it is the "
                "highest-leverage retention action.", "green")
    with ic2:
        insight("Elite + High Risk", "Check lapsing customers in the table above",
                "Revenue at risk from Elite + High lapsers is your most urgent number. "
                "<strong>These customers are profitable and can still be saved.</strong>", "red")
    with ic3:
        insight("Low Segment", "High one-time rate in Low segment",
                "Converting 10% of Low one-timers adds significant recurring revenue. "
                "First repeat visit = critical conversion moment.", "amber")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4
# ══════════════════════════════════════════════════════════════════════════════
with tab4:

    sh("Loyalty Programme Health")
    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        pts_df = D.load_points_buckets()
        fig_pts = go.Figure(go.Bar(
            x=pts_df["points_bucket"].fillna("Unknown"),
            y=pts_df["customer_count"].fillna(0),
            marker_color=[GRAY, ORANGE, TEAL, AFYA_BLUE, COOL_BLUE][:len(pts_df)],
            text=pts_df["customer_count"].fillna(0),
            textposition="outside", cliponaxis=False,
            hovertemplate="<b>%{x}</b>: %{y} customers<extra></extra>",
        ))
        fig_pts.update_layout(**_CL, height=260)
        fig_pts.update_xaxes(**_ax(showgrid=False, tickfont=dict(size=8)))
        fig_pts.update_yaxes(**_ax(showgrid=False, showticklabels=False))
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Points Bucket Distribution</div>', unsafe_allow_html=True)
        pc(fig_pts)
        note("High concentration in lowest bucket = redemption threshold too high. "
             "A low-entry tier (50 pts = KSh 50 off) would activate dormant members.", warn=True)

    with c2:
        lseg_df = D.load_loyalty_by_segment()
        fig_lseg = go.Figure(go.Bar(
            x=lseg_df["loyalty_pct"].fillna(0),
            y=lseg_df["refined_tier"].fillna("Unknown"),
            orientation="h",
            marker_color=[seg_color(t) for t in lseg_df["refined_tier"].fillna("")],
            text=[f'{v}%' for v in lseg_df["loyalty_pct"].fillna(0)],
            textposition="outside", cliponaxis=False,
            hovertemplate="<b>%{y}</b>: %{x}% loyalty rate<extra></extra>",
        ))
        fig_lseg.update_layout(**_CL, height=260)
        fig_lseg.update_xaxes(**_ax(showgrid=False), ticksuffix="%",
                              title_text="% of segment enrolled in loyalty")
        fig_lseg.update_yaxes(**_ax(showgrid=False))
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Loyalty Rate by Segment</div>', unsafe_allow_html=True)
        pc(fig_lseg)

    with c3:
        lr_df = D.load_loyalty_return()
        fig_lr = go.Figure(go.Bar(
            x=lr_df["member_status"].fillna("Unknown"),
            y=lr_df["avg_days_between_visits"].fillna(0),
            marker_color=[TEAL, GRAY],
            text=[safe_str(v, "d") for v in lr_df["avg_days_between_visits"].fillna(0)],
            textposition="outside", cliponaxis=False,
            textfont=dict(size=12, family="Montserrat"),
            hovertemplate="<b>%{x}</b>: avg %{y} days between visits<extra></extra>",
        ))
        fig_lr.update_layout(**_CL, height=260)
        fig_lr.update_xaxes(**_ax(showgrid=False))
        fig_lr.update_yaxes(**_ax(showgrid=False, showticklabels=False),
                            title_text="Avg days between visits")
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Return Speed — Loyalty vs Non-Loyalty</div>', unsafe_allow_html=True)
        pc(fig_lr)

    gap(8)
    sh("Loyalty Redemption Activity + Enrolment Timing", mt=4)
    c1, c2 = st.columns(2, gap="large")

    with c1:
        red_df = D.load_loyalty_redemption()
        fig_red = go.Figure()
        fig_red.add_trace(go.Bar(
            x=red_df["month"], y=red_df["loyalty_kes_redeemed"].fillna(0),
            name="KES redeemed", marker_color=_rgba(AFYA_BLUE, 0.6),
            hovertemplate="%{x|%b %Y}: KSh %{y:,.0f} redeemed<extra></extra>",
        ))
        fig_red.add_trace(go.Scatter(
            x=red_df["month"], y=red_df["redeeming_customers"].fillna(0),
            name="Redeeming customers", mode="lines+markers",
            line=dict(color=TEAL, width=2), marker=dict(size=5), yaxis="y2",
            hovertemplate="%{x|%b %Y}: %{y} customers<extra></extra>",
        ))
        fig_red.update_layout(**_CL, height=280,
            yaxis2=dict(overlaying="y", side="right", showgrid=False,
                        tickfont=dict(size=9, color=TEAL, family="Montserrat")))
        fig_red.update_xaxes(**AXIS, tickformat="%b %Y")
        fig_red.update_yaxes(**AXIS, tickformat=",.0f", tickprefix="KSh ",
                             title_text="KES redeemed")
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Is Loyalty Redemption Actually Happening?</div>', unsafe_allow_html=True)
        pc(fig_red)
        note("Bars = KES redeemed. Line = customers who redeemed. "
             "Flat bars = programme exists but nobody uses it.")

    with c2:
        lag_df = D.load_loyalty_conversion_lag()
        fig_lag = go.Figure(go.Bar(
            x=lag_df["customer_count"].fillna(0),
            y=lag_df["enrolment_lag"].fillna("Unknown"),
            orientation="h",
            marker_color=[TEAL, AFYA_BLUE, PURPLE, ORANGE, CORAL][:len(lag_df)],
            text=lag_df["customer_count"].fillna(0),
            textposition="outside", cliponaxis=False,
            customdata=lag_df["avg_daily_spend_intensity"].fillna(0),
            hovertemplate="<b>%{y}</b>: %{x} customers<br>"
                          "Daily Spend Intensity: KSh %{customdata:,.0f}/day<extra></extra>",
        ))
        fig_lag.update_layout(**_CL, height=280)
        fig_lag.update_xaxes(**_ax(showgrid=False, showticklabels=False))
        fig_lag.update_yaxes(**_ax(showgrid=False))
        st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">When Do Customers Join Loyalty?</div>', unsafe_allow_html=True)
        pc(fig_lag)
        note("Day 1 enrolees show higher Daily Spend Intensity. "
             "Enrolment at first purchase should be standard process, not optional.")

    gap(8)
    sh("Loyalty Sign-Up Growth", mt=4)
    lt_df = D.load_loyalty_trend()
    fig_lt = go.Figure()
    fig_lt.add_trace(go.Bar(
        x=lt_df["month"], y=lt_df["new_loyalty_members"].fillna(0),
        name="New members / month", marker_color=_rgba(AFYA_BLUE, 0.5),
        hovertemplate="%{x|%b %Y}: %{y} new members<extra></extra>",
    ))
    fig_lt.add_trace(go.Scatter(
        x=lt_df["month"], y=lt_df["cumulative"].fillna(0),
        name="Cumulative members", mode="lines+markers",
        line=dict(color=TEAL, width=2), marker=dict(size=5), yaxis="y2",
        hovertemplate="%{x|%b %Y}: %{y} total<extra></extra>",
    ))
    fig_lt.update_layout(**_CL, height=260,
        yaxis2=dict(overlaying="y", side="right", showgrid=False,
                    tickfont=dict(size=9, color=TEAL, family="Montserrat")))
    fig_lt.update_xaxes(**AXIS, tickformat="%b %Y")
    fig_lt.update_yaxes(**AXIS, title_text="New members / month")
    st.markdown('<div style="font-size:15px;font-weight:600;color:#003467;margin-bottom:4px">Loyalty Sign-Up Growth</div>', unsafe_allow_html=True)
    pc(fig_lt)

    gap(8); sh("Insights")
    ic1, ic2 = st.columns(2, gap="large")
    with ic1:
        insight("Loyalty Programme Signal",
                "Is redemption actually happening?",
                "Flat redemption chart = programme exists on paper only. "
                "Add a low-entry tier (50 pts = KSh 50 off).", "amber")
    with ic2:
        insight("Enrolment Timing",
                "Enrol customers on Day 1 — not later",
                "Day 1 enrolees show higher Daily Spend Intensity. "
                "Every day of enrolment lag is a missed behaviour-change opportunity.", "green")