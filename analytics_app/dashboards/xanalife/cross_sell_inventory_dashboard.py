
try:
    import pyarrow  # noqa: F401
except Exception:
    pass

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
from xanalife.cross_sell.utils.queries import (
    HOME_STATS_QUERY, HOME_PERIOD_QUERY, HOME_TREND_QUERY,
    TOP_PRODUCTS_QUERY, STOCKOUT_PREDICTION_QUERY, STORE_TREND_QUERY,
)
from xanalife.cross_sell.utils.theme import (
     inject_css, COLORS, CHART_LAYOUT, fmt_kes,
    sidebar_nav, section_header, page_banner, kpi_card, info_card,
)

try:
    st.set_page_config(
        page_title="XanaLife Analytics",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

# inject_css()

if "xc_page" not in st.session_state:
    st.session_state.xc_page = "home"

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    _logo = "analytics_app/dashboards/xanalife/cross_sell/utils/Logo.png"
    if os.path.exists(_logo):
        st.markdown('<div style="padding:20px 16px 12px">', unsafe_allow_html=True)
        st.image(_logo, width=148)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="padding:20px 16px 12px">'
            '<span style="font-size:18px;font-weight:800;color:#0072CE">XanaLife</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div style="border-bottom:1px solid #D6E4F0;margin:0 0 12px"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:9px;font-weight:700;color:#8AABCC;text-transform:uppercase;'
        'letter-spacing:1.5px;padding:0 4px;margin-bottom:6px">Menu</div>',
        unsafe_allow_html=True,
    )
    if st.button("🏠  Home", use_container_width=True, key="nav_home",
                 type="primary" if st.session_state.xc_page == "home" else "secondary"):
        st.session_state.xc_page = "home"
        st.rerun()
    if st.button("📊  Cross Selling Intelligence", use_container_width=True, key="nav_csuc",
                 type="primary" if st.session_state.xc_page == "csuc" else "secondary"):
        st.session_state.xc_page = "csuc"
        st.rerun()
    if st.button("📦  Inventory Risk", use_container_width=True, key="nav_stockout",
                 type="primary" if st.session_state.xc_page == "stockout" else "secondary"):
        st.session_state.xc_page = "stockout"
        st.rerun()
    st.markdown(
        '<div style="border-bottom:1px solid #D6E4F0;margin:12px 0 16px"></div>',
        unsafe_allow_html=True,
    )

    if st.session_state.xc_page == "home":
        section_header("About", margin_top=0)
        st.markdown(
            '<div style="font-size:11px;color:#8AABCC;line-height:2">'
            'Data: Sep 2025 – Mar 2026<br>'
            'Source: XanaLife POS &amp; Inventory<br>'
            'Refreshed every hour'
            '</div>',
            unsafe_allow_html=True,
        )

# ── Page routing ──────────────────────────────────────────────────────────────
_PAGES_DIR = os.path.join(
    "analytics_app", "dashboards", "xanalife", "cross_sell", "utils", "pages"
)

if st.session_state.xc_page == "csuc":
    _path = os.path.join(_PAGES_DIR, "1_CSUC.py")
    with open(_path, encoding="utf-8") as _f:
        exec(compile(_f.read(), _path, "exec"), {**globals(), "__name__": "__main__"})

elif st.session_state.xc_page == "stockout":
    _path = os.path.join(_PAGES_DIR, "3_Stockout_Prediction.py")
    with open(_path, encoding="utf-8") as _f:
        exec(compile(_f.read(), _path, "exec"), {**globals(), "__name__": "__main__"})

else:
    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner(""):
        try:
            stats          = run_query(HOME_STATS_QUERY)
            period         = run_query(HOME_PERIOD_QUERY)
            trend_df       = run_query(HOME_TREND_QUERY)
            products       = run_query(TOP_PRODUCTS_QUERY)
            stockout_df    = run_query(STOCKOUT_PREDICTION_QUERY)
            store_trend_df = run_query(STORE_TREND_QUERY)

            stats.columns          = [c.strip() for c in stats.columns]
            period.columns         = [c.strip() for c in period.columns]
            trend_df.columns       = [c.strip() for c in trend_df.columns]
            products.columns       = [c.strip() for c in products.columns]
            stockout_df.columns    = [c.strip() for c in stockout_df.columns]
            store_trend_df.columns = [c.strip() for c in store_trend_df.columns]

            def _col(df, *candidates):
                mapping = {c.upper(): c for c in df.columns}
                for name in candidates:
                    if name.upper() in mapping:
                        return mapping[name.upper()]
                return candidates[0]

            total_stores  = int(pd.to_numeric(stats[_col(stats, "TOTAL_STORES")].iloc[0],  errors="coerce") or 0)
            active_skus   = int(pd.to_numeric(stats[_col(stats, "ACTIVE_SKUS")].iloc[0],   errors="coerce") or 0)

            rev_30d       = float(pd.to_numeric(period[_col(period, "REV_LAST_30D")].iloc[0],  errors="coerce") or 0)
            rev_prior_30d = float(pd.to_numeric(period[_col(period, "REV_PRIOR_30D")].iloc[0], errors="coerce") or 0)
            txns_30d      = int(pd.to_numeric(period[_col(period, "TXNS_LAST_30D")].iloc[0],   errors="coerce") or 0)

            trend_df["REVENUE"]      = pd.to_numeric(trend_df[_col(trend_df, "REVENUE")],      errors="coerce")
            trend_df["TRANSACTIONS"] = pd.to_numeric(trend_df[_col(trend_df, "TRANSACTIONS")], errors="coerce")
            trend_df["MONTH"]        = pd.to_datetime(trend_df[_col(trend_df, "MONTH")])

            products["TOTAL_REVENUE"] = pd.to_numeric(products[_col(products, "TOTAL_REVENUE")], errors="coerce")

            stockout_df["7-Day Revenue at Risk (KES)"] = pd.to_numeric(stockout_df["7-Day Revenue at Risk (KES)"], errors="coerce")
            stockout_df["Margin %"]                    = pd.to_numeric(stockout_df["Margin %"],                    errors="coerce")
            stockout_df["Total Revenue (KES)"]         = pd.to_numeric(stockout_df["Total Revenue (KES)"],         errors="coerce")
            stockout_df["Stock Value (KES)"]           = pd.to_numeric(stockout_df["Stock Value (KES)"],           errors="coerce")

            store_trend_df["REVENUE"]        = pd.to_numeric(store_trend_df[_col(store_trend_df, "REVENUE")],        errors="coerce")
            store_trend_df["TRANSACTIONS"]   = pd.to_numeric(store_trend_df[_col(store_trend_df, "TRANSACTIONS")],   errors="coerce")
            store_trend_df["AVG_BASKET_KES"] = pd.to_numeric(store_trend_df[_col(store_trend_df, "AVG_BASKET_KES")], errors="coerce")
            store_trend_df["MONTH"]          = pd.to_datetime(store_trend_df[_col(store_trend_df, "MONTH")])

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
            products = stockout_df = trend_df = store_trend_df = pd.DataFrame()
            rev_30d = rev_prior_30d = txns_30d = total_stores = active_skus = 0
            n_stockouts = n_critical = 0
            rev_at_risk = 0.0
            st.error(f"Data load failed: {e}")

    # ══ PAGE HEADER ═══════════════════════════════════════════════════════════
    page_banner(
        title    = "Executive Overview",
        subtitle = "How the business is performing — and where to look first.",
    )

    if data_loaded:
        pct_vs_prior = ((rev_30d - rev_prior_30d) / rev_prior_30d * 100) if rev_prior_30d > 0 else 0
        arrow        = "▲" if pct_vs_prior >= 0 else "▼"
        headline_txt = (
            f'Revenue last 30d: <b>{fmt_kes(rev_30d)}</b> '
            f'({arrow} {abs(pct_vs_prior):.1f}% vs prior 30d) &nbsp;·&nbsp; '
            f'<b>{n_stockouts}</b> stockouts, <b>{fmt_kes(rev_at_risk)}</b> at risk this week.'
        )
        info_card(headline_txt, border_color=COLORS["primary"])

    if data_loaded:
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card("Revenue — Last 30d",  fmt_kes(rev_30d),    "rolling 30 days",    COLORS["primary"])
        with c2: kpi_card("Active Stores",        f"{total_stores:,}", "all locations",      COLORS["success"])
        with c3: kpi_card("Active SKUs",          f"{active_skus:,}",  "products tracked",   COLORS["muted"])
        with c4: kpi_card("Transactions — 30d",   f"{txns_30d:,}",     "paid transactions",  COLORS["warning"])

    st.markdown("<div style='margin-top:36px'></div>", unsafe_allow_html=True)

    # ── Revenue trend ──────────────────────────────────────────────────────────
    if data_loaded and not trend_df.empty:
        section_header("Revenue Trend — Monthly")

        trend_sorted = trend_df.sort_values("MONTH")
        fig_trend = px.line(trend_sorted, x="MONTH", y="REVENUE", markers=True, height=260)
        fig_trend.update_traces(
            line_color=COLORS["primary"], marker_color=COLORS["primary"],
            line_width=2.5, marker_size=6,
        )
        fig_trend.update_layout(
            **{**CHART_LAYOUT, "margin": dict(l=0, r=0, t=16, b=8)},
            height=260, xaxis_title=None, yaxis_title="Revenue (KES)",
        )
        fig_trend.update_xaxes(dtick="M1", tickformat="%b '%y")
        fig_trend.update_yaxes(tickformat=",.0f")
        st.plotly_chart(fig_trend, use_container_width=True)

        if len(trend_sorted) >= 2:
            last_r  = trend_sorted.iloc[-1]
            prior_r = trend_sorted.iloc[-2]
            mom_pct = ((last_r["REVENUE"] - prior_r["REVENUE"]) / prior_r["REVENUE"] * 100) if prior_r["REVENUE"] > 0 else 0
            mom_arrow = "▲" if mom_pct >= 0 else "▼"
            st.markdown(
                f'<p style="font-size:11px;color:#8AABCC;margin:-8px 0 0">'
                f'{last_r["MONTH"].strftime("%B")} vs {prior_r["MONTH"].strftime("%B")}: '
                f'{mom_arrow} {abs(mom_pct):.1f}%</p>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin-top:36px'></div>", unsafe_allow_html=True)

    # ── Store performance ──────────────────────────────────────────────────────
    if data_loaded and not store_trend_df.empty:
        section_header("Store Performance — Last vs Prior Month")

        store_trend_s = store_trend_df.sort_values("MONTH")
        months = sorted(store_trend_s["MONTH"].unique())

        if len(months) >= 2:
            last_m  = months[-1]
            prior_m = months[-2]

            last_rev  = store_trend_s[store_trend_s["MONTH"] == last_m].set_index(_col(store_trend_s, "STORE_NAME"))
            prior_rev = store_trend_s[store_trend_s["MONTH"] == prior_m].set_index(_col(store_trend_s, "STORE_NAME"))["REVENUE"]

            store_growth = last_rev[["REVENUE", "AVG_BASKET_KES"]].copy()
            store_growth = store_growth.join(prior_rev.rename("REVENUE_PRIOR"), how="left")
            store_growth["delta_pct"] = (
                (store_growth["REVENUE"] - store_growth["REVENUE_PRIOR"])
                / store_growth["REVENUE_PRIOR"] * 100
            ).round(1)
            store_growth = store_growth.reset_index().rename(
                columns={store_growth.index.name or "index": "STORE_NAME"}
            ).sort_values("REVENUE", ascending=False)

            store_display = pd.DataFrame({
                "Store":            store_growth["STORE_NAME"],
                "Revenue (KES)":    store_growth["REVENUE"].round(0),
                "Δ vs Prior (%)":   store_growth["delta_pct"],
                "Avg Basket (KES)": store_growth["AVG_BASKET_KES"].round(0),
            }).reset_index(drop=True)

            def _style_delta(series):
                return [
                    "color: #0BB99F; font-weight: 700" if pd.notna(v) and v >= 5
                    else "color: #E11D48; font-weight: 700" if pd.notna(v) and v <= -5
                    else "color: #D97706; font-weight: 700" if pd.notna(v)
                    else ""
                    for v in series
                ]

            st.dataframe(
                store_display.style.apply(_style_delta, subset=["Δ vs Prior (%)"]),
                use_container_width=True, hide_index=True,
                height=min(len(store_display) * 38 + 48, 340),
                column_config={
                    "Revenue (KES)":    st.column_config.NumberColumn(format="KES %,.0f"),
                    "Δ vs Prior (%)":   st.column_config.NumberColumn("Δ vs Prior", format="%.1f%%"),
                    "Avg Basket (KES)": st.column_config.NumberColumn(format="KES %,.0f"),
                },
            )
        else:
            st.info("Not enough monthly data to compute store Δ.")

    st.markdown("<div style='margin-top:36px'></div>", unsafe_allow_html=True)

    # ── Top profit drivers ─────────────────────────────────────────────────────
    if data_loaded and not products.empty and not stockout_df.empty:
        section_header("Top Profit Drivers — Last 6 Months")

        margin_by_product = (
            stockout_df[stockout_df["Margin %"].notna()]
            .groupby("Product")["Margin %"].mean().reset_index()
            .rename(columns={"Product": "PRODUCT_NAME"})
        )
        top50 = products.sort_values("TOTAL_REVENUE", ascending=False).head(50).copy()
        top50["PRODUCT_NAME_UPPER"] = top50[_col(products, "PRODUCT_NAME")].str.upper().str.strip()
        margin_by_product["PRODUCT_NAME_UPPER"] = margin_by_product["PRODUCT_NAME"].str.upper().str.strip()
        top50 = top50.merge(margin_by_product[["PRODUCT_NAME_UPPER", "Margin %"]], on="PRODUCT_NAME_UPPER", how="left")
        top50["Margin %"]    = top50["Margin %"].fillna(0)
        top50["Gross Profit"] = (top50["TOTAL_REVENUE"] * top50["Margin %"] / 100).round(0)
        top10 = top50[top50["Gross Profit"] > 0].sort_values("Gross Profit", ascending=False).head(10)

        if not top10.empty:
            top10 = top10.sort_values("Gross Profit", ascending=True).copy()
            top10["Label"]       = top10[_col(top10, "PRODUCT_NAME")].str.title()
            top10["ProfitLabel"] = top10["Gross Profit"].apply(
                lambda v: f"KES {v/1_000:.0f}K" if v >= 1_000 else f"KES {v:.0f}"
            )
            fig_gp = px.bar(
                top10, x="Gross Profit", y="Label", orientation="h",
                text="ProfitLabel", color="Gross Profit",
                color_continuous_scale=[[0, "#A8E6DC"], [1, COLORS["success"]]],
                height=360,
            )
            fig_gp.update_traces(textposition="outside", textfont=dict(size=10, color="#003467"), marker_line_width=0)
            fig_gp.update_layout(
                **{**CHART_LAYOUT, "margin": dict(l=0, r=70, t=10, b=8)},
                height=360, coloraxis_showscale=False,
            )
            fig_gp.update_xaxes(title=None, showticklabels=False, showgrid=False, zeroline=False)
            fig_gp.update_yaxes(title=None, tickfont=dict(size=11))
            st.plotly_chart(fig_gp, use_container_width=True)

    st.markdown("<div style='margin-top:36px'></div>", unsafe_allow_html=True)

    # ── Live alert CTA ─────────────────────────────────────────────────────────
    if data_loaded:
        has_alerts = n_stockouts > 0 or n_critical > 0
        if has_alerts:
            st.markdown(
                f'<div style="background:#FFF5F7;border:1px solid #FEC5CF;'
                f'border-left:4px solid #E11D48;border-radius:8px;'
                f'padding:14px 20px;margin-bottom:8px">'
                f'<span style="font-size:10px;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:1.5px;color:#E11D48">⚡ Action Required &nbsp;—&nbsp; </span>'
                f'<span style="font-size:13px;color:#003467;font-weight:600">'
                f'{n_stockouts} stockouts + {n_critical} critical = '
                f'{fmt_kes(rev_at_risk)} at risk over 7 days.</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("📦  Open Inventory Risk"):
                st.session_state.xc_page = "stockout"
                st.rerun()
        else:
            info_card("✅ Stock health looks good — no products are currently out of stock or critical.", border_color=COLORS["success"])

    st.markdown("<div style='margin-top:40px'></div>", unsafe_allow_html=True)