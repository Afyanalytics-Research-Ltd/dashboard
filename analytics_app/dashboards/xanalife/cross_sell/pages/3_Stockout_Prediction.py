import streamlit as st
import pandas as pd
import plotly.express as px
from utils.snowflake_conn import run_query
from utils.queries import STOCKOUT_PREDICTION_QUERY, STORES_QUERY
from utils.theme import (
    inject_css, COLORS, CHART_LAYOUT, STATUS_COLORS,
    kpi_card, section_header, page_banner, sidebar_nav, fmt_kes,
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

df.columns                            = [c.strip() for c in df.columns]
df["Days Stock Remaining"]            = pd.to_numeric(df["Days Stock Remaining"],            errors="coerce")
df["7-Day Revenue at Risk (KES)"]     = pd.to_numeric(df["7-Day Revenue at Risk (KES)"],     errors="coerce")
df["Total Revenue (KES)"]             = pd.to_numeric(df["Total Revenue (KES)"],             errors="coerce")
df["Stock on Hand"]                   = pd.to_numeric(df["Stock on Hand"],                   errors="coerce")
df["Stock Value (KES)"]               = pd.to_numeric(df["Stock Value (KES)"],               errors="coerce")
df["Margin %"]                        = pd.to_numeric(df["Margin %"],                        errors="coerce")
df["Predicted Stockout Date"]         = pd.to_datetime(df["Predicted Stockout Date"],        errors="coerce")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    sidebar_nav()
    section_header("Filters", margin_top=0)

    store_options = ["All stores"] + stores["STORE_NAME"].dropna().tolist()
    store_filter  = st.selectbox("Store", store_options)
    status_filter = st.multiselect(
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

# ── KPIs (reflect store selection, not drill-down filters) ────────────────────
kpi_base    = df if store_filter == "All stores" else df[df["Store"] == store_filter]
stockouts   = len(kpi_base[kpi_base["Stock Status"] == "Stockout"])
critical    = len(kpi_base[kpi_base["Stock Status"] == "Critical"])
warning     = len(kpi_base[kpi_base["Stock Status"] == "Warning"])
overstocked = len(kpi_base[kpi_base["Stock Status"] == "Overstocked"])
rev_at_risk = kpi_base[kpi_base["Stock Status"].isin(["Stockout", "Critical"])]["7-Day Revenue at Risk (KES)"].sum()
stock_val   = kpi_base["Stock Value (KES)"].sum()

# ── Page header ───────────────────────────────────────────────────────────────
page_banner(
    title    = "Inventory Risk",
    subtitle = "Predicts when each product will run out based on current stock and recent sales velocity. "
               "Products are ranked by urgency so your team knows exactly what to reorder first.",
    tag      = "Inventory Risk Monitor",
)

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: kpi_card("Out of Stock",        f"{stockouts:,}",        "immediate reorder needed",       COLORS["danger"])
with c2: kpi_card("Critical (< 7 days)", f"{critical:,}",         "order today",                    "#f97316")
with c3: kpi_card("Warning (< 14 days)", f"{warning:,}",          "reorder this week",              COLORS["warning"])
with c4: kpi_card("Overstocked",         f"{overstocked:,}",      "90+ days on hand",               COLORS["primary"])
with c5: kpi_card("7-Day Rev at Risk",   fmt_kes(rev_at_risk),    "stockouts + critical",           COLORS["danger"])
with c6: kpi_card("Total Stock Value",   fmt_kes(stock_val),      "all active products",            COLORS["muted"])

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


# ══ TAB 1 — Overview ──────────────────────────────────────────────────────────
with tab1:
    section_header("7-Day Revenue at Risk by Status", margin_top=0)
    note("How much revenue is at risk from each stock status category over the next 7 days.")

    rev_by_status = (
        df.groupby("Stock Status")["7-Day Revenue at Risk (KES)"]
        .sum().reset_index()
        .sort_values("7-Day Revenue at Risk (KES)", ascending=True)
    )
    rev_by_status = rev_by_status[rev_by_status["7-Day Revenue at Risk (KES)"] > 0]

    rev_col, count_col = st.columns([3, 1], gap="large")
    with rev_col:
        fig_rev = px.bar(
            rev_by_status,
            x="7-Day Revenue at Risk (KES)", y="Stock Status",
            orientation="h", color="Stock Status",
            color_discrete_map=STATUS_COLORS,
            text="7-Day Revenue at Risk (KES)",
            height=300,
        )
        fig_rev.update_traces(
            texttemplate="KES %{x:,.0f}",
            textposition="outside",
            textfont=dict(size=10, color="#003467"),
        )
        fig_rev.update_layout(
            **CHART_LAYOUT, height=300,
            showlegend=False, yaxis_title=None,
            xaxis_title="Revenue at risk (KES)",
        )
        fig_rev.update_xaxes(showticklabels=False)
        st.plotly_chart(fig_rev, use_container_width=True)

    with count_col:
        section_header("Products by status", margin_top=0)
        status_counts = df["Stock Status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        for _, row in status_counts.iterrows():
            color = STATUS_COLORS.get(row["Status"], "#6B8CAE")
            st.markdown(
                f'<div style="display:flex;align-items:center;justify-content:space-between;'
                f'padding:6px 0;border-bottom:1px solid #F4F8FC">'
                f'<div style="display:flex;align-items:center;gap:8px">'
                f'<div style="width:8px;height:8px;border-radius:50%;background:{color};flex-shrink:0"></div>'
                f'<span style="font-size:11px;color:#003467">{row["Status"]}</span></div>'
                f'<span style="font-size:12px;font-weight:700;color:#003467">{row["Count"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<hr style="border:none;border-top:1px solid #EBF3FB;margin:8px 0 20px">', unsafe_allow_html=True)
    section_header("Stockout Calendar — Next 30 Days")
    note("Each dot is a product about to stock out. Left = sooner. Higher = more revenue at risk. Hover for details.")

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
            height=360,
        )
        fig_tl.update_layout(
            **CHART_LAYOUT, height=360,
            xaxis_title="Predicted stockout date",
            yaxis_title="Revenue at risk (KES)",
        )
        st.plotly_chart(fig_tl, use_container_width=True)
    else:
        st.success("No stockouts predicted in the next 30 days.")


# ══ TAB 2 — Order Now ─────────────────────────────────────────────────────────
with tab2:
    section_header("Immediate Reorder List", margin_top=0)
    note("Products already out of stock or running out within 7 days — sorted by revenue at risk. Share with procurement today.")

    urgent = df[df["Stock Status"].isin(["Stockout", "Critical"])].sort_values(
        "7-Day Revenue at Risk (KES)", ascending=False
    )

    if urgent.empty:
        st.success("No products currently at critical stock levels.")
    else:
        urgent_cols = [
            "Store", "Product", "Category", "Unit",
            "Stock on Hand", "Daily Demand", "Days Stock Remaining",
            "Predicted Stockout Date", "Stock Status",
            "Recommended Reorder Qty", "7-Day Revenue at Risk (KES)",
            "Margin %", "Demand Trend",
        ]
        st.dataframe(
            urgent[urgent_cols],
            use_container_width=True, height=420,
            hide_index=True,
            column_config={
                "Stock on Hand":               st.column_config.NumberColumn(format="%.2f"),
                "Daily Demand":                st.column_config.NumberColumn(format="%.2f"),
                "Days Stock Remaining":        st.column_config.NumberColumn("Days Left",    format="%.1f"),
                "7-Day Revenue at Risk (KES)": st.column_config.NumberColumn("Rev at Risk",  format="KES %,.0f"),
                "Margin %":                    st.column_config.NumberColumn(format="%.1f%%"),
                "Recommended Reorder Qty":     st.column_config.NumberColumn("Reorder Qty",  format="%d"),
                "Stock Status":                st.column_config.TextColumn("Status"),
            },
        )
        st.download_button(
            "⬇  Download reorder list",
            data=urgent[urgent_cols].to_csv(index=False).encode("utf-8"),
            file_name="xanalife_urgent_reorders.csv",
            mime="text/csv",
        )


# ══ TAB 3 — Overstocked ───────────────────────────────────────────────────────
with tab3:
    overstock_df = df[df["Stock Status"] == "Overstocked"].sort_values("Stock Value (KES)", ascending=False)

    if overstock_df.empty:
        st.success("No overstocked products detected.")
    else:
        c_kpis, _ = st.columns([3, 1])
        with c_kpis:
            k1, k2, k3 = st.columns(3)
            with k1: kpi_card("Overstocked SKUs",  f"{len(overstock_df):,}",                                 "products",       COLORS["primary"])
            with k2: kpi_card("Capital tied up",   fmt_kes(overstock_df["Stock Value (KES)"].sum()),         "stock value",    COLORS["warning"])
            with k3: kpi_card("Avg days on hand",  f"{overstock_df['Days Stock Remaining'].median():.0f}d",  "median",         COLORS["muted"])

        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        section_header("Overstocked Products by Stock Value", margin_top=0)
        note("Consider pausing reorders or running promotions on these items to free up working capital.")

        fig_over = px.bar(
            overstock_df.head(20).sort_values("Stock Value (KES)"),
            x="Stock Value (KES)", y="Product", orientation="h",
            color="Days Stock Remaining",
            color_continuous_scale=[[0, "#D6E4F0"], [1, COLORS["primary"]]],
            hover_data=["Daily Demand", "Days Stock Remaining", "Category"],
            height=460,
        )
        fig_over.update_layout(
            **CHART_LAYOUT, height=460,
            yaxis_title=None, xaxis_title="Stock value (KES)",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_over, use_container_width=True)


# ══ TAB 4 — Full Table ────────────────────────────────────────────────────────
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
            "Recommended Reorder Qty":     st.column_config.NumberColumn("Reorder Qty",   format="%d"),
        },
    )
    st.download_button(
        "⬇  Download filtered table",
        data=filtered[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="xanalife_stockout_predictions.csv",
        mime="text/csv",
    )

# ══ TAB 5 — ABC Analysis ──────────────────────────────────────────────────────
with tab5:
    section_header("ABC Product Classification", margin_top=0)
    note("A products drive ~80% of revenue. They should never stock out. B and C products are managed differently.")

    # Build ABC from revenue data
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

        # Merge stock status back in
        status_map = df.groupby("Product")["Stock Status"].first().to_dict()
        abc["Stock Status"] = abc["Product"].map(status_map).fillna("Unknown")

        a_prods = abc[abc["ABC"] == "A"]
        b_prods = abc[abc["ABC"] == "B"]
        c_prods = abc[abc["ABC"] == "C"]

        # ── KPI summary ──
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

        # ── Pareto curve ──
        ch_pareto, ch_at_risk = st.columns([3, 2], gap="large")

        with ch_pareto:
            section_header("Revenue Concentration — The 80/20 Rule", margin_top=0)
            note("The steep early curve shows how few products drive most of the revenue.")

            ABC_COLORS = {"A": "#0072CE", "B": "#D97706", "C": "#6B8CAE"}
            fig_pareto = px.line(
                abc, x="Product %", y="Cumulative %",
                color="ABC", color_discrete_map=ABC_COLORS,
                labels={"Product %": "% of products (ranked by revenue)", "Cumulative %": "Cumulative % of revenue"},
                height=340,
            )
            fig_pareto.add_hline(y=80,  line_dash="dot", line_color="#D97706", line_width=1,
                                 annotation_text="80% revenue threshold", annotation_position="right")
            fig_pareto.add_hline(y=95,  line_dash="dot", line_color="#6B8CAE", line_width=1,
                                 annotation_text="95%", annotation_position="right")
            fig_pareto.update_layout(**CHART_LAYOUT, height=340, showlegend=False,
                                     xaxis_title="% of products (ranked by revenue)",
                                     yaxis_title="Cumulative % of revenue")
            st.plotly_chart(fig_pareto, use_container_width=True)

        with ch_at_risk:
            section_header("A Products Currently at Risk", margin_top=0)
            note("Your highest-value products that are Stockout or Critical right now.")

            a_at_risk = a_prods[a_prods["Stock Status"].isin(["Stockout", "Critical"])].copy()

            if a_at_risk.empty:
                st.markdown(
                    '<div style="background:#F0FDF9;border:1px solid #6EE7D4;border-radius:8px;'
                    'padding:20px;text-align:center;color:#0BB99F;font-size:13px;font-weight:600">'
                    '✅ All A products are healthy or within safe stock levels.'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                for _, row in a_at_risk.head(8).iterrows():
                    status_color = STATUS_COLORS.get(row["Stock Status"], "#6B8CAE")
                    st.markdown(
                        f'<div style="display:flex;align-items:center;justify-content:space-between;'
                        f'padding:8px 0;border-bottom:1px solid #F4F8FC">'
                        f'<div>'
                        f'<div style="font-size:12px;font-weight:600;color:#003467">'
                        f'{row["Product"].title()}</div>'
                        f'<div style="font-size:10px;color:#8AABCC">'
                        f'{fmt_kes(row["Total Revenue (KES)"])} revenue</div>'
                        f'</div>'
                        f'<span style="background:{status_color};color:#fff;font-size:9px;'
                        f'font-weight:700;padding:3px 8px;border-radius:4px;white-space:nowrap">'
                        f'{row["Stock Status"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f'<div style="font-size:11px;color:#E11D48;font-weight:600;margin-top:12px">'
                    f'⚠ {len(a_at_risk)} A product(s) need immediate reorder — go to Order Now tab.'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Full ABC table ──
        st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
        section_header("Full ABC Classification Table", margin_top=0)

        abc_display = abc[["ABC", "Product", "Total Revenue (KES)", "Cumulative %", "Stock Status"]].copy()
        abc_display["Total Revenue (KES)"] = abc_display["Total Revenue (KES)"].round(0)

        st.dataframe(
            abc_display,
            use_container_width=True,
            height=360,
            hide_index=True,
            column_config={
                "ABC":                   st.column_config.TextColumn("Class", width="small"),
                "Total Revenue (KES)":   st.column_config.NumberColumn("Revenue (KES)", format="KES %,.0f"),
                "Cumulative %":          st.column_config.NumberColumn("Cumulative %",  format="%.1f%%"),
                "Stock Status":          st.column_config.TextColumn("Stock Status"),
            },
        )
        st.download_button(
            "⬇  Download ABC classification",
            data=abc_display.to_csv(index=False).encode("utf-8"),
            file_name="xanalife_abc_classification.csv",
            mime="text/csv",
        )
