try:
    import pyarrow  # noqa: F401 — pre-load before Streamlit lazy-imports it
except Exception:
    pass

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from xanalife.cross_sell.utils.snowflake_conn import run_query
from xanalife.cross_sell.utils.queries import STOCKOUT_PREDICTION_QUERY, STORES_QUERY
from xanalife.cross_sell.utils.theme import (
    inject_css, COLORS, CHART_LAYOUT, STATUS_COLORS,
    kpi_card, section_header, page_banner, sidebar_nav, fmt_kes, info_card,
)

st.set_page_config(page_title="XanaLife · Inventory Risk", layout="wide", initial_sidebar_state="expanded")
inject_css()

def note(text):
    st.markdown(
        f'<p style="font-size:11px;color:#8AABCC;margin:2px 0 14px;line-height:1.5">{text}</p>',
        unsafe_allow_html=True,
    )

DATA_DATE = pd.Timestamp("2026-03-18")

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading..."):
    df     = run_query(STOCKOUT_PREDICTION_QUERY)
    stores = run_query(STORES_QUERY)

if df.empty:
    st.warning("No data returned.")
    st.stop()

df.columns = [c.strip() for c in df.columns]
stores.columns = [c.strip() for c in stores.columns]

def _col(frame, name):
    mapping = {c.upper(): c for c in frame.columns}
    return mapping.get(name.upper(), name)

# ── Numeric casts ─────────────────────────────────────────────────────────────
for col in [
    "Days Stock Remaining", "7-Day Revenue at Risk (KES)", "Total Revenue (KES)",
    "Stock on Hand", "Stock Value (KES)", "Margin %", "Urgency Rank",
    "Recommended Reorder Qty", "Unit Cost (KES)", "Selling Price (KES)",
    "Daily Demand",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["Predicted Stockout Date"] = pd.to_datetime(df["Predicted Stockout Date"], errors="coerce")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    sidebar_nav()
    section_header("Filters", margin_top=0)

    store_name_col = _col(stores, "STORE_NAME")
    store_options  = ["All stores"] + stores[store_name_col].dropna().tolist()
    store_filter   = st.selectbox("Store", store_options)
    status_filter  = st.multiselect(
        "Stock status",
        options=["Stockout", "Critical", "Warning", "Monitor", "Healthy", "Overstocked", "No demand data"],
        default=["Stockout", "Critical", "Warning"],
    )
    cat_options   = ["All"] + sorted(df["Category"].dropna().unique().tolist())
    cat_filter    = st.selectbox("Category", cat_options)
    trend_options = ["All"] + sorted(df["Demand Trend"].dropna().unique().tolist())
    trend_filter  = st.selectbox("Demand trend", trend_options)
    min_rev       = st.number_input("Min revenue (KES)", min_value=0, value=0, step=5000)

    st.markdown('<div style="border-bottom:1px solid #D6E4F0;margin:20px 0"></div>', unsafe_allow_html=True)
    section_header("Status guide", margin_top=0)
    STATUS_DESCRIPTIONS = {
        "Stockout":       "No stock left",
        "Critical":       "< 7 days remaining",
        "Warning":        "7–14 days remaining",
        "Monitor":        "14–30 days remaining",
        "Healthy":        "30–90 days remaining",
        "Overstocked":    "> 90 days on hand",
        "No demand data": "Stock exists, no recent sales",
    }
    for status, color in STATUS_COLORS.items():
        desc = STATUS_DESCRIPTIONS.get(status, "")
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:8px">'
            f'<div style="width:8px;height:8px;border-radius:50%;background:{color};'
            f'flex-shrink:0;margin-top:3px"></div>'
            f'<div><div style="font-size:11px;font-weight:600;color:#003467">{status}</div>'
            f'<div style="font-size:10px;color:#8AABCC">{desc}</div></div></div>',
            unsafe_allow_html=True,
        )

# ── KPI base (store-filtered, before drill-down) ──────────────────────────────
kpi_base    = df if store_filter == "All stores" else df[df["Store"] == store_filter]
stockouts   = int((kpi_base["Stock Status"] == "Stockout").sum())
critical    = int((kpi_base["Stock Status"] == "Critical").sum())
warning     = int((kpi_base["Stock Status"] == "Warning").sum())
overstocked = int((kpi_base["Stock Status"] == "Overstocked").sum())
rev_at_risk = float(kpi_base[kpi_base["Stock Status"].isin(["Stockout", "Critical"])]["7-Day Revenue at Risk (KES)"].sum())
stock_val   = float(kpi_base["Stock Value (KES)"].sum())

# Reorder economics
urgent_base = kpi_base[kpi_base["Stock Status"].isin(["Stockout", "Critical"])].copy()
reorder_cost = float((urgent_base["Recommended Reorder Qty"].fillna(0) * urgent_base["Unit Cost (KES)"].fillna(0)).sum())
roi_headline = round(rev_at_risk / reorder_cost, 1) if reorder_cost > 0 else 0

# ── Page header ───────────────────────────────────────────────────────────────
page_banner(
    title    = "Inventory Risk",
    subtitle = "Predicts when each product will run out based on current stock and recent demand velocity.",
    tag      = "Inventory Risk Monitor",
)

# ── Exec headline ─────────────────────────────────────────────────────────────
info_card(
    f'Action this week: <b>{stockouts}</b> stockouts + <b>{critical}</b> critical = '
    f'<b>{fmt_kes(rev_at_risk)}</b> at risk over 7d. '
    f'Reorder cost: <b>{fmt_kes(reorder_cost)}</b>. '
    f'ROI: <b>{roi_headline:.1f}×</b>.',
    border_color=COLORS["danger"] if (stockouts + critical) > 0 else COLORS["success"],
)

# ── KPI cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: kpi_card("Out of Stock",        f"{stockouts:,}",        "immediate reorder needed",    COLORS["danger"])
with c2: kpi_card("Critical (< 7 days)", f"{critical:,}",         "order today",                 "#f97316")
with c3: kpi_card("Warning (< 14 days)", f"{warning:,}",          "reorder this week",           COLORS["warning"])
with c4: kpi_card("Overstocked",         f"{overstocked:,}",      "90+ days on hand",            COLORS["primary"])
with c5: kpi_card("7-Day Rev at Risk",   fmt_kes(rev_at_risk),    "stockouts + critical",        COLORS["danger"])
with c6: kpi_card("Total Stock Value",   fmt_kes(stock_val),      "all active products",         COLORS["muted"])

st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)

# ── Apply drill-down filters ───────────────────────────────────────────────────
filtered = df.copy()
if store_filter != "All stores":
    filtered = filtered[filtered["Store"] == store_filter]
if status_filter:
    filtered = filtered[filtered["Stock Status"].isin(status_filter)]
if cat_filter != "All":
    filtered = filtered[filtered["Category"] == cat_filter]
if trend_filter != "All":
    filtered = filtered[filtered["Demand Trend"] == trend_filter]
if min_rev > 0:
    filtered = filtered[filtered["Total Revenue (KES)"] >= min_rev]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Order Now", "Overstocked", "Full Table", "ABC Analysis"])


# ══ TAB 1 — Overview ══════════════════════════════════════════════════════════
with tab1:

    # ── Store Inventory Scorecard ──────────────────────────────────────────────
    section_header("Store Inventory Scorecard", margin_top=0)

    def _build_scorecard(base_df):
        rows = []
        for store, g in base_df.groupby("Store"):
            at_risk = g[g["Stock Status"].isin(["Stockout", "Critical"])]
            healthy_n = g["Stock Status"].isin(["Healthy", "Monitor"]).sum()
            rows.append({
                "Store":                store,
                "Stockouts":            int((g["Stock Status"] == "Stockout").sum()),
                "Critical":             int((g["Stock Status"] == "Critical").sum()),
                "7d Risk (KES)":        float(at_risk["7-Day Revenue at Risk (KES)"].sum()),
                "Healthy SKUs %":       round(healthy_n / len(g) * 100, 1) if len(g) else 0.0,
                "Overstock Value (KES)": float(g[g["Stock Status"] == "Overstocked"]["Stock Value (KES)"].sum()),
            })
        return pd.DataFrame(rows).sort_values("7d Risk (KES)", ascending=False).reset_index(drop=True)

    scorecard = _build_scorecard(kpi_base)

    if not scorecard.empty:
        def _style_risk(series):
            q75 = series.quantile(0.75)
            q25 = series.quantile(0.25)
            return [
                "background-color: #FFF5F7; color: #E11D48; font-weight:700" if v >= q75 and v > 0
                else "background-color: #FFFBEB; color: #D97706; font-weight:600" if v >= q25 and v > 0
                else "background-color: #F0FDF9; color: #0BB99F" if v == 0
                else ""
                for v in series
            ]

        sc_styled = scorecard.style.apply(_style_risk, subset=["7d Risk (KES)"])
        st.dataframe(
            sc_styled,
            use_container_width=True,
            hide_index=True,
            height=min(len(scorecard) * 38 + 48, 340),
            column_config={
                "7d Risk (KES)":         st.column_config.NumberColumn(format="KES %,.0f"),
                "Overstock Value (KES)": st.column_config.NumberColumn(format="KES %,.0f"),
                "Healthy SKUs %":        st.column_config.NumberColumn(format="%.1f%%"),
                "Stockouts":             st.column_config.NumberColumn(format="%.0f"),
                "Critical":              st.column_config.NumberColumn(format="%.0f"),
            },
        )

    st.markdown('<hr style="border:none;border-top:1px solid #EBF3FB;margin:20px 0">', unsafe_allow_html=True)

    # ── ABC × Status heatmap ───────────────────────────────────────────────────
    section_header("ABC × Stock Status Matrix", margin_top=0)

    abc_src = df[df["Total Revenue (KES)"] > 0].copy()
    if not abc_src.empty:
        abc_rev = (
            abc_src.groupby("Product")["Total Revenue (KES)"].sum()
            .reset_index()
            .sort_values("Total Revenue (KES)", ascending=False)
            .reset_index(drop=True)
        )
        abc_rev["CumRev"]  = abc_rev["Total Revenue (KES)"].cumsum()
        abc_rev["Cum%"]    = abc_rev["CumRev"] / abc_rev["Total Revenue (KES)"].sum() * 100
        abc_rev["ABC"]     = abc_rev["Cum%"].apply(lambda x: "A" if x <= 80 else ("B" if x <= 95 else "C"))
        abc_map            = abc_rev.set_index("Product")["ABC"].to_dict()

        df_abc = df.copy()
        df_abc["ABC"] = df_abc["Product"].map(abc_map).fillna("C")

        matrix_statuses = ["Stockout", "Critical", "Warning"]
        hm_data = (
            df_abc[df_abc["Stock Status"].isin(matrix_statuses)]
            .groupby(["ABC", "Stock Status"])["7-Day Revenue at Risk (KES)"]
            .sum()
            .unstack(fill_value=0)
        )
        for cls in ["A", "B", "C"]:
            if cls not in hm_data.index:
                hm_data.loc[cls] = 0
        for st_col in matrix_statuses:
            if st_col not in hm_data.columns:
                hm_data[st_col] = 0
        hm_data = hm_data.loc[["A", "B", "C"], matrix_statuses]

        z_vals    = hm_data.values.tolist()
        z_text    = [[fmt_kes(v) if v > 0 else "—" for v in row] for row in z_vals]

        fig_hm = go.Figure(data=go.Heatmap(
            z=z_vals, x=matrix_statuses, y=["A", "B", "C"],
            colorscale=[[0, "#F4F8FC"], [0.4, "#fde68a"], [1, "#E11D48"]],
            text=z_text, texttemplate="%{text}",
            textfont={"size": 11, "color": "#003467"},
            hovertemplate="Class %{y} × %{x}: %{text}<extra></extra>",
            showscale=False,
        ))
        fig_hm.update_layout(
            **{**CHART_LAYOUT, "margin": dict(l=0, r=0, t=16, b=8)},
            height=230,
            yaxis_title="ABC Class", xaxis_title=None,
        )
        st.plotly_chart(fig_hm, use_container_width=True)
        st.markdown(
            '<p style="font-size:11px;color:#8AABCC;margin:-8px 0 0">'
            'Class-A stockouts are the only fires you fight today.</p>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr style="border:none;border-top:1px solid #EBF3FB;margin:20px 0">', unsafe_allow_html=True)

    # ── Stockout Calendar ──────────────────────────────────────────────────────
    section_header("Stockout Calendar — Next 30 Days")
    note("Each dot is a product about to stock out. Left = sooner. Higher = more revenue at risk.")

    timeline_df = df[
        df["Predicted Stockout Date"].notna() &
        (df["Predicted Stockout Date"] <= DATA_DATE + pd.Timedelta(days=30)) &
        (df["7-Day Revenue at Risk (KES)"] > 0)
    ].copy()

    if not timeline_df.empty:
        timeline_df["Bubble"] = timeline_df["7-Day Revenue at Risk (KES)"].clip(upper=50000)
        fig_tl = px.scatter(
            timeline_df,
            x="Predicted Stockout Date", y="7-Day Revenue at Risk (KES)",
            size="Bubble", color="Stock Status", color_discrete_map=STATUS_COLORS,
            hover_name="Product",
            hover_data=["Category", "Days Stock Remaining", "Recommended Reorder Qty", "Demand Trend"],
            height=320,
        )
        fig_tl.update_layout(
            **CHART_LAYOUT, height=320,
            xaxis_title="Predicted stockout date",
            yaxis_title="Revenue at risk (KES)",
        )
        st.plotly_chart(fig_tl, use_container_width=True)
    else:
        st.success("No stockouts predicted in the next 30 days.")


# ══ TAB 2 — Order Now ════════════════════════════════════════════════════════
with tab2:
    urgent = df[df["Stock Status"].isin(["Stockout", "Critical"])].copy()

    if urgent.empty:
        st.success("No products currently at critical stock levels.")
    else:
        # Compute financial columns
        urgent["Reorder Cost (KES)"]       = (urgent["Recommended Reorder Qty"].fillna(0) * urgent["Unit Cost (KES)"].fillna(0)).round(0)
        urgent["Recovered Revenue (KES)"]  = (urgent["Recommended Reorder Qty"].fillna(0) * urgent["Selling Price (KES)"].fillna(0)).round(0)
        urgent["ROI"] = (
            urgent["Recovered Revenue (KES)"] /
            urgent["Reorder Cost (KES)"].replace(0, float("nan"))
        ).round(1)
        urgent["_profit_sort"] = urgent["Recovered Revenue (KES)"] * urgent["Margin %"].fillna(0) / 100
        urgent = urgent.sort_values("_profit_sort", ascending=False)

        # Top-of-tab KPIs + CTA
        po_cols = [
            "Store", "Product", "Category", "Unit",
            "Recommended Reorder Qty", "Reorder Cost (KES)",
            "Recovered Revenue (KES)", "ROI",
        ]
        total_reorder_cost     = urgent["Reorder Cost (KES)"].sum()
        total_recovered        = urgent["Recovered Revenue (KES)"].sum()
        total_roi              = round(total_recovered / total_reorder_cost, 1) if total_reorder_cost > 0 else 0

        k1, k2, k3, k4 = st.columns(4)
        with k1: kpi_card("Total Reorder Cost",    fmt_kes(total_reorder_cost), f"{len(urgent)} lines",        COLORS["warning"])
        with k2: kpi_card("Recovered Revenue",      fmt_kes(total_recovered),   "if fully restocked",          COLORS["success"])
        with k3: kpi_card("Portfolio ROI",          f"{total_roi:.1f}×",        "recovered / reorder cost",    COLORS["primary"])
        with k4: kpi_card("Products to Order",      f"{len(urgent):,}",         "Stockout + Critical",         COLORS["danger"])

        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
        st.download_button(
            "⬇  Generate PO — top 20",
            data=urgent.head(20)[po_cols].to_csv(index=False).encode("utf-8"),
            file_name="xanalife_purchase_order_top20.csv",
            mime="text/csv",
        )

        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
        section_header("Immediate Reorder List", margin_top=0)
        note("Sorted by profit-weighted recovered revenue. Share with procurement today.")

        urgent_cols = [
            "Store", "Product", "Category", "Unit",
            "Stock on Hand", "Daily Demand", "Days Stock Remaining",
            "Stock Status", "Recommended Reorder Qty",
            "Reorder Cost (KES)", "Recovered Revenue (KES)", "ROI",
            "Margin %", "7-Day Revenue at Risk (KES)", "Demand Trend",
        ]
        st.dataframe(
            urgent[urgent_cols],
            use_container_width=True, height=420,
            hide_index=True,
            column_config={
                "Stock on Hand":               st.column_config.NumberColumn(format="%.2f"),
                "Daily Demand":                st.column_config.NumberColumn(format="%.2f"),
                "Days Stock Remaining":        st.column_config.NumberColumn("Days Left",         format="%.1f"),
                "7-Day Revenue at Risk (KES)": st.column_config.NumberColumn("Rev at Risk",       format="KES %,.0f"),
                "Reorder Cost (KES)":          st.column_config.NumberColumn("Reorder Cost",      format="KES %,.0f"),
                "Recovered Revenue (KES)":     st.column_config.NumberColumn("Recovered Rev.",    format="KES %,.0f"),
                "ROI":                         st.column_config.NumberColumn("ROI",               format="%.1f×"),
                "Margin %":                    st.column_config.NumberColumn(format="%.1f%%"),
                "Recommended Reorder Qty":     st.column_config.NumberColumn("Reorder Qty",       format="%.0f"),
                "Stock Status":                st.column_config.TextColumn("Status"),
            },
        )
        st.download_button(
            "⬇  Download full reorder list",
            data=urgent[urgent_cols].to_csv(index=False).encode("utf-8"),
            file_name="xanalife_urgent_reorders.csv",
            mime="text/csv",
        )


# ══ TAB 3 — Overstocked ══════════════════════════════════════════════════════
with tab3:
    overstock_df = df[df["Stock Status"] == "Overstocked"].sort_values("Stock Value (KES)", ascending=False)

    if overstock_df.empty:
        st.success("No overstocked products detected.")
    else:
        total_overstock_value = float(overstock_df["Stock Value (KES)"].sum())
        info_card(
            f'<b>{fmt_kes(total_overstock_value)}</b> working capital trapped in slow-movers. '
            f'Consider pausing reorders or running promotions to free up cash.',
            border_color=COLORS["warning"],
        )

        k1, k2, k3 = st.columns(3)
        with k1: kpi_card("Overstocked SKUs",  f"{len(overstock_df):,}",                                "products",    COLORS["primary"])
        with k2: kpi_card("Capital Tied Up",   fmt_kes(total_overstock_value),                          "stock value", COLORS["warning"])
        with k3: kpi_card("Avg Days on Hand",  f"{overstock_df['Days Stock Remaining'].median():.0f}d", "median",      COLORS["muted"])

        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        section_header("Overstocked Products by Stock Value", margin_top=0)

        overstock_display = overstock_df.head(20).copy()
        overstock_display = overstock_display.sort_values("Stock Value (KES)", ascending=True)

        fig_over = px.bar(
            overstock_display,
            x="Stock Value (KES)", y="Product", orientation="h",
            color="Days Stock Remaining",
            color_continuous_scale=[[0, "#D6E4F0"], [1, COLORS["primary"]]],
            hover_data=["Daily Demand", "Days Stock Remaining", "Category"],
            height=460,
        )
        fig_over.update_layout(
            **CHART_LAYOUT, height=460,
            yaxis_title=None, xaxis_title="Stock value / cash tied up (KES)",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_over, use_container_width=True)

        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
        section_header("Slow-Mover Detail", margin_top=0)
        over_table_cols = ["Store", "Product", "Category", "Days Stock Remaining", "Stock Value (KES)", "Daily Demand", "Demand Trend"]
        over_table = overstock_df[over_table_cols].copy()
        over_table = over_table.rename(columns={"Stock Value (KES)": "Cash Tied Up (KES)"})
        st.dataframe(
            over_table,
            use_container_width=True, height=340, hide_index=True,
            column_config={
                "Cash Tied Up (KES)":   st.column_config.NumberColumn(format="KES %,.0f"),
                "Days Stock Remaining": st.column_config.NumberColumn("Days on Hand", format="%.0f"),
                "Daily Demand":         st.column_config.NumberColumn(format="%.3f"),
            },
        )


# ══ TAB 4 — Full Table ════════════════════════════════════════════════════════
with tab4:
    section_header(f"Full Inventory Risk Table  ·  {len(filtered):,} products", margin_top=0)

    display_cols = [
        "Store", "Product", "Category", "Unit",
        "Stock on Hand", "Daily Demand", "Days Stock Remaining",
        "Predicted Stockout Date", "Stock Status", "Demand Trend",
        "7-Day Revenue at Risk (KES)", "Total Revenue (KES)",
        "Recommended Reorder Qty", "Margin %", "Stock Value (KES)",
    ]
    st.dataframe(
        filtered[display_cols],
        use_container_width=True, height=460,
        hide_index=True,
        column_config={
            "Stock on Hand":               st.column_config.NumberColumn(format="%.2f"),
            "Daily Demand":                st.column_config.NumberColumn(format="%.4f"),
            "Days Stock Remaining":        st.column_config.NumberColumn("Days Left",     format="%.1f"),
            "7-Day Revenue at Risk (KES)": st.column_config.NumberColumn("Rev at Risk",   format="KES %,.0f"),
            "Total Revenue (KES)":         st.column_config.NumberColumn("Total Revenue", format="KES %,.0f"),
            "Stock Value (KES)":           st.column_config.NumberColumn("Stock Value",   format="KES %,.0f"),
            "Margin %":                    st.column_config.NumberColumn(format="%.1f%%"),
            "Recommended Reorder Qty":     st.column_config.NumberColumn("Reorder Qty",   format="%.0f"),
        },
    )
    st.download_button(
        "⬇  Download filtered table",
        data=filtered[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="xanalife_stockout_predictions.csv",
        mime="text/csv",
    )


# ══ TAB 5 — ABC Analysis ══════════════════════════════════════════════════════
with tab5:
    section_header("ABC Product Classification", margin_top=0)
    note("A products drive ~80% of revenue. They should never stock out.")

    abc_src = df[df["Total Revenue (KES)"] > 0].copy()
    if abc_src.empty:
        st.info("No revenue data available for ABC classification.")
    else:
        abc = (
            abc_src.groupby("Product")["Total Revenue (KES)"].sum()
            .reset_index()
            .sort_values("Total Revenue (KES)", ascending=False)
            .reset_index(drop=True)
        )
        abc["Cumulative Revenue"] = abc["Total Revenue (KES)"].cumsum()
        total_rev_abc = abc["Total Revenue (KES)"].sum()
        abc["Cumulative %"] = (abc["Cumulative Revenue"] / total_rev_abc * 100).round(2)
        abc["Product %"]    = ((abc.index + 1) / len(abc) * 100).round(2)
        abc["ABC"]          = abc["Cumulative %"].apply(
            lambda x: "A" if x <= 80 else ("B" if x <= 95 else "C")
        )
        status_map = df.groupby("Product")["Stock Status"].first().to_dict()
        abc["Stock Status"] = abc["Product"].map(status_map).fillna("Unknown")

        a_prods = abc[abc["ABC"] == "A"]
        b_prods = abc[abc["ABC"] == "B"]
        c_prods = abc[abc["ABC"] == "C"]

        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(
                f'<div style="background:#EBF3FB;border:1px solid #B0C8E0;border-top:3px solid #0072CE;'
                f'border-radius:8px;padding:18px 16px">'
                f'<div style="font-size:10px;font-weight:700;color:#0072CE;text-transform:uppercase;'
                f'letter-spacing:1px;margin-bottom:8px">A — Priority Products</div>'
                f'<div style="font-size:28px;font-weight:800;color:#003467">{len(a_prods)}</div>'
                f'<div style="font-size:11px;color:#6B8CAE;margin-top:4px">'
                f'{len(a_prods)/len(abc)*100:.0f}% of SKUs · ~80% of revenue · never let these stock out</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f'<div style="background:#FFF7ED;border:1px solid #FDE68A;border-top:3px solid #D97706;'
                f'border-radius:8px;padding:18px 16px">'
                f'<div style="font-size:10px;font-weight:700;color:#D97706;text-transform:uppercase;'
                f'letter-spacing:1px;margin-bottom:8px">B — Monitor Products</div>'
                f'<div style="font-size:28px;font-weight:800;color:#003467">{len(b_prods)}</div>'
                f'<div style="font-size:11px;color:#6B8CAE;margin-top:4px">'
                f'{len(b_prods)/len(abc)*100:.0f}% of SKUs · ~15% of revenue · watch for movement</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f'<div style="background:#F4F8FC;border:1px solid #D6E4F0;border-top:3px solid #6B8CAE;'
                f'border-radius:8px;padding:18px 16px">'
                f'<div style="font-size:10px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
                f'letter-spacing:1px;margin-bottom:8px">C — Long Tail</div>'
                f'<div style="font-size:28px;font-weight:800;color:#003467">{len(c_prods)}</div>'
                f'<div style="font-size:11px;color:#6B8CAE;margin-top:4px">'
                f'{len(c_prods)/len(abc)*100:.0f}% of SKUs · ~5% of revenue · review for delisting</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)

        ch_pareto, ch_at_risk = st.columns([3, 2], gap="large")

        with ch_pareto:
            section_header("Revenue Concentration — The 80/20 Rule", margin_top=0)
            ABC_COLORS = {"A": "#0072CE", "B": "#D97706", "C": "#6B8CAE"}
            fig_pareto = px.line(
                abc, x="Product %", y="Cumulative %",
                color="ABC", color_discrete_map=ABC_COLORS,
                height=320,
            )
            fig_pareto.add_hline(y=80, line_dash="dot", line_color="#D97706", line_width=1,
                                 annotation_text="80% revenue", annotation_position="right")
            fig_pareto.add_hline(y=95, line_dash="dot", line_color="#6B8CAE", line_width=1,
                                 annotation_text="95%", annotation_position="right")
            fig_pareto.update_layout(**CHART_LAYOUT, height=320, showlegend=False,
                                     xaxis_title="% of products (ranked by revenue)",
                                     yaxis_title="Cumulative % of revenue")
            st.plotly_chart(fig_pareto, use_container_width=True)

        with ch_at_risk:
            section_header("A Products Currently at Risk", margin_top=0)
            a_at_risk = a_prods[a_prods["Stock Status"].isin(["Stockout", "Critical"])].copy()

            if a_at_risk.empty:
                st.markdown(
                    '<div style="background:#F0FDF9;border:1px solid #6EE7D4;border-radius:8px;'
                    'padding:20px;text-align:center;color:#0BB99F;font-size:13px;font-weight:600">'
                    '✅ All A products are healthy.</div>',
                    unsafe_allow_html=True,
                )
            else:
                for _, row in a_at_risk.head(8).iterrows():
                    sc = STATUS_COLORS.get(row["Stock Status"], "#6B8CAE")
                    st.markdown(
                        f'<div style="display:flex;align-items:center;justify-content:space-between;'
                        f'padding:8px 0;border-bottom:1px solid #F4F8FC">'
                        f'<div>'
                        f'<div style="font-size:12px;font-weight:600;color:#003467">{row["Product"].title()}</div>'
                        f'<div style="font-size:10px;color:#8AABCC">{fmt_kes(row["Total Revenue (KES)"])} revenue</div>'
                        f'</div>'
                        f'<span style="background:{sc};color:#fff;font-size:9px;'
                        f'font-weight:700;padding:3px 8px;border-radius:4px">{row["Stock Status"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f'<div style="font-size:11px;color:#E11D48;font-weight:600;margin-top:12px">'
                    f'{len(a_at_risk)} A product(s) need immediate reorder.</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
        section_header("Full ABC Classification Table", margin_top=0)

        abc_display = abc[["ABC", "Product", "Total Revenue (KES)", "Cumulative %", "Stock Status"]].copy()
        abc_display["Total Revenue (KES)"] = abc_display["Total Revenue (KES)"].round(0)

        st.dataframe(
            abc_display,
            use_container_width=True, height=360, hide_index=True,
            column_config={
                "ABC":                  st.column_config.TextColumn("Class", width="small"),
                "Total Revenue (KES)":  st.column_config.NumberColumn("Revenue (KES)", format="KES %,.0f"),
                "Cumulative %":         st.column_config.NumberColumn("Cumulative %",  format="%.1f%%"),
                "Stock Status":         st.column_config.TextColumn("Stock Status"),
            },
        )
        st.download_button(
            "⬇  Download ABC classification",
            data=abc_display.to_csv(index=False).encode("utf-8"),
            file_name="xanalife_abc_classification.csv",
            mime="text/csv",
        )
