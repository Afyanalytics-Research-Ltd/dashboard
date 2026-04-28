try:
    import pyarrow 
except Exception:
    pass

import streamlit as st
import pandas as pd
import plotly.express as px
from xanalife.cross_sell.utils.snowflake_conn import run_query
from xanalife.cross_sell.utils.queries import CSUC_QUERY, STORES_QUERY, HOME_STATS_QUERY, STOCKOUT_PREDICTION_QUERY
from xanalife.cross_sell.utils.theme import (
    inject_css, COLORS, CHART_LAYOUT,
    kpi_card, section_header, page_banner, sidebar_nav, fmt_kes, info_card,
)

st.set_page_config(page_title="XanaLife · Cross-Sell", layout="wide", initial_sidebar_state="expanded")
inject_css()

DATA_MONTHS = 6.5   # Sep 2025 – Mar 2026

def note(text):
    st.markdown(
        f'<p style="font-size:11px;color:#8AABCC;margin:2px 0 14px;line-height:1.5">{text}</p>',
        unsafe_allow_html=True,
    )

def action_type(signal, capture_pct):
    if signal == "High" and capture_pct < 40:
        return "Bundle Deal",      "🏷️", "#7F77DD"
    return "Promote Together", "📢", COLORS["success"]

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading..."):
    try:
        df          = run_query(CSUC_QUERY)
        stores      = run_query(STORES_QUERY)
        stats       = run_query(HOME_STATS_QUERY)
        stockout_df = run_query(STOCKOUT_PREDICTION_QUERY)
    except Exception as _load_err:
        st.error(f"Data load failed: {_load_err}")
        st.stop()

if df.empty:
    st.warning("No data returned.")
    st.stop()

df.columns          = [c.strip() for c in df.columns]
stores.columns      = [c.strip() for c in stores.columns]
stats.columns       = [c.strip() for c in stats.columns]
stockout_df.columns = [c.strip() for c in stockout_df.columns]

# ── Numeric casts ─────────────────────────────────────────────────────────────
df["CSUC score"]         = pd.to_numeric(df["CSUC score"],         errors="coerce")
df["Lift"]               = pd.to_numeric(df["Lift"],               errors="coerce")
df["Baskets with A"]     = pd.to_numeric(df["Baskets with A"],     errors="coerce")
df["Baskets together"]   = pd.to_numeric(df["Baskets together"],   errors="coerce")
df["P(B|A)"]             = pd.to_numeric(df["P(B|A)"],             errors="coerce")
df["P(B) baseline"]      = pd.to_numeric(df["P(B) baseline"],      errors="coerce")
df["Avg Price B (KES)"]  = pd.to_numeric(df["Avg Price B (KES)"],  errors="coerce").fillna(0)
df["Margin B %"]         = pd.to_numeric(df["Margin B %"],         errors="coerce").fillna(0)

stockout_df["7-Day Revenue at Risk (KES)"] = pd.to_numeric(stockout_df["7-Day Revenue at Risk (KES)"], errors="coerce")
stockout_df["Urgency Rank"]                = pd.to_numeric(stockout_df["Urgency Rank"],                errors="coerce")

# Total revenue (for % of revenue headline)
def _col(frame, name):
    mapping = {c.upper(): c for c in frame.columns}
    return mapping.get(name.upper(), name)

total_revenue = float(
    pd.to_numeric(stats[_col(stats, "TOTAL_REVENUE")].iloc[0], errors="coerce") or 0
)

# ── Derived columns ───────────────────────────────────────────────────────────
df["Opportunity (baskets)"]   = (df["Baskets with A"] * df["CSUC score"]).round(0).astype("Int64")
df["Capture Rate %"]          = (df["P(B|A)"] * 100).round(1)
df["Monthly KES Opportunity"] = (
    df["Opportunity (baskets)"].astype(float) / DATA_MONTHS
    * df["Avg Price B (KES)"]
    * 0.50
).round(0)
df["profit_score"] = df["Monthly KES Opportunity"] * df["Margin B %"] / 100

SIGNAL_COLORS = {"High": COLORS["success"], "Medium": COLORS["warning"], "Low": COLORS["muted"]}

# Stock health lookup(Product B per Store) 
stock_worst = (
    stockout_df[stockout_df["Urgency Rank"].notna()]
    .sort_values("Urgency Rank")
    .groupby(["Store", "Product"])["Stock Status"]
    .first()
    .reset_index()
    .rename(columns={"Product": "Product B", "Stock Status": "_stock_b"})
)

# Sidebar 
with st.sidebar:
    sidebar_nav()
    section_header("Filters", margin_top=0)

    store_options = ["All stores"] + stores[_col(stores, "STORE_NAME")].dropna().tolist()
    store_filter  = st.selectbox("Store", store_options)
    signal_filter = st.multiselect(
        "Confidence",
        ["High", "Medium", "Low"],
        default=["High", "Medium"],
        help="High = 50+ shared transactions · Medium = 20–49 · Low = 5–19",
    )
    min_csuc = st.slider(
        "Min uplift score",
        0.0, float(df["CSUC score"].max()), 0.05, 0.01,
        help="Extra probability of buying B when A is in the basket",
    )
    search_product = st.text_input("Search product", placeholder="e.g. CHAPATI")

    st.markdown('<div style="border-bottom:1px solid #D6E4F0;margin:20px 0"></div>', unsafe_allow_html=True)
    section_header("How to read this", margin_top=0)
    for label, desc in [
        ("Uplift score",        "Extra probability of buying B when A is in the basket"),
        ("Lift ×",              ">1.2 = genuine pattern, not random chance"),
        ("Capture rate",        "% of A transactions that already include B"),
        ("Monthly opp.",        "Extra KES/month at 50% additional conversion"),
        ("🏷️ Bundle Deal",      "High signal + low capture → combo price closes the gap"),
        ("📢 Promote Together", "Validated pair → feature in campaigns, flyers, signage"),
    ]:
        st.markdown(
            f'<div style="margin-bottom:7px">'
            f'<span style="font-size:10px;font-weight:700;color:#003467">{label}</span><br>'
            f'<span style="font-size:10px;color:#8AABCC">{desc}</span></div>',
            unsafe_allow_html=True,
        )

# Apply filters
filtered = df.copy()
if store_filter != "All stores":
    filtered = filtered[filtered["Store"] == store_filter]
filtered = filtered[
    filtered["Signal strength"].isin(signal_filter) &
    (filtered["CSUC score"] >= min_csuc)
]
if search_product:
    q = search_product.upper()
    filtered = filtered[
        filtered["Product A"].str.contains(q, na=False) |
        filtered["Product B"].str.contains(q, na=False)
    ]

# Attach stock status for Product B
filtered = filtered.merge(stock_worst[["Store", "Product B", "_stock_b"]], on=["Store", "Product B"], how="left")
filtered["_stock_b"] = filtered["_stock_b"].fillna("Unknown")

# Chart base (all stores, respects confidence + uplift but not store selector)
chart_base = df[
    df["Signal strength"].isin(signal_filter) &
    (df["CSUC score"] >= min_csuc)
].copy()

# ── Page header ───────────────────────────────────────────────────────────────
page_banner(
    title    = "Cross-Sell Intelligence",
    subtitle = "Statistically validated product pairs — ranked by profit impact, ready to act on.",
    tag      = "Cross-Sell Intelligence",
)

# ── Headline ──────────────────────────────────────────────────────────────────
annual_upside = float(filtered["Monthly KES Opportunity"].sum() * 12)
top10_annual  = float(
    filtered.nlargest(10, "Monthly KES Opportunity")["Monthly KES Opportunity"].sum() * 12
)
pct_of_rev = round(annual_upside / total_revenue * 100, 1) if total_revenue > 0 else 0
top10_pct  = round(top10_annual / annual_upside * 100, 0)  if annual_upside > 0 else 0

info_card(
    f'Capturing 50% of identified pairs = <b>{fmt_kes(annual_upside)}</b> annual upside '
    f'(~{pct_of_rev}% of revenue). Top 10 actions deliver <b>{top10_pct:.0f}%</b> of that.',
    border_color=COLORS["success"],
)

# ── KPI cards ─────────────────────────────────────────────────────────────────
avg_margin_b  = filtered[filtered["Margin B %"] > 0]["Margin B %"].mean() if len(filtered) else 0
ready_to_push = len(filtered[filtered["Signal strength"] == "High"])

c1, c2, c3, c4 = st.columns(4)
with c1: kpi_card("Annual Opportunity",   fmt_kes(annual_upside),                    "at 50% conversion",        COLORS["success"])
with c2: kpi_card("% of Annual Revenue",  f"{pct_of_rev:.1f}%",                      "of total business",        COLORS["primary"])
with c3: kpi_card("High-Signal Pairs",    f"{ready_to_push:,}",                      "ready to push",            COLORS["warning"])
with c4: kpi_card("Avg Margin B",         f"{avg_margin_b:.1f}%" if ready_to_push else "—", "on Product B",      COLORS["muted"])

st.markdown("<div style='margin-top:36px'></div>", unsafe_allow_html=True)

# ══ ACTION PLAYBOOK ═══════════════════════════════════════════════════════════
section_header("Action Playbook — Top Opportunities", margin_top=0)

SUPPRESS   = {"Critical", "Stockout", "No demand data"}
playbook_all = (
    filtered[
        filtered["Signal strength"].isin(["High", "Medium"]) &
        (filtered["Monthly KES Opportunity"] > 0)
    ]
    .sort_values("profit_score", ascending=False)
    .copy()
)

n_suppressed  = int(playbook_all["_stock_b"].isin(SUPPRESS).sum())
playbook_show = playbook_all[~playbook_all["_stock_b"].isin(SUPPRESS)].head(8)

if n_suppressed > 0:
    st.markdown(
        f'<p style="font-size:11px;color:#E11D48;font-weight:600;margin:0 0 12px">'
        f'{n_suppressed} pair(s) hidden — fix stock first before pushing these products.</p>',
        unsafe_allow_html=True,
    )

ACTION_DESCRIPTIONS = {
    "Bundle Deal": (
        "Only {capture:.0f}% of {a} buyers also pick up {b} — despite a {lift:.1f}× affinity. "
        "A bundled offer (combo price or 'add {b} for KES X') would close that gap."
    ),
    "Promote Together": (
        "Customers who buy {a} are {lift:.1f}× more likely to also buy {b}. "
        "Feature both in the same campaign, flyer, or in-store promo."
    ),
}

STOCK_BADGE = {
    "Healthy":         ("🟢", "Ready to push"),
    "Monitor":         ("🟢", "Ready to push"),
    "Warning":         ("🟡", "Push lightly"),
    "Unknown":         ("🟢", "Ready to push"),
}

if playbook_show.empty:
    st.info("No actionable pairs for the current filters.")
else:
    for _, row in playbook_show.iterrows():
        sig         = row["Signal strength"]
        sig_color   = SIGNAL_COLORS.get(sig, COLORS["muted"])
        capture     = float(row["Capture Rate %"])
        opp_kes     = float(row["Monthly KES Opportunity"])
        annual_opp  = opp_kes * 12
        lift        = float(row["Lift"])
        margin_b    = float(row["Margin B %"])
        prod_a      = str(row["Product A"]).title()
        prod_b      = str(row["Product B"]).title()
        stock_status = row["_stock_b"]

        act_label, act_icon, act_color = action_type(sig, capture)
        description = ACTION_DESCRIPTIONS[act_label].format(
            capture=capture, a=prod_a, b=prod_b, lift=lift
        )
        stock_icon, stock_label = STOCK_BADGE.get(stock_status, ("🟢", "Ready to push"))

        # Session state for actioned badge
        key      = f"acted_{row['Product A']}_{row['Product B']}_{row['Store']}"
        actioned = st.session_state.get(key, False)
        actioned_badge = (
            '<span style="background:#0BB99F;color:#fff;font-size:9px;font-weight:700;'
            'padding:2px 8px;border-radius:10px;margin-left:8px">✓ Actioned</span>'
            if actioned else ""
        )

        st.markdown(
            f'<div style="background:#fff;border:1px solid #D6E4F0;'
            f'border-left:4px solid {act_color};border-radius:8px;'
            f'padding:14px 18px;margin-bottom:6px">'

            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap">'
            f'<span style="font-size:13px;font-weight:800;color:#003467">{prod_a}</span>'
            f'<span style="font-size:10px;color:#B0C8E0">+</span>'
            f'<span style="font-size:13px;font-weight:800;color:#0072CE">{prod_b}</span>'
            f'<span style="flex:1"></span>'
            f'<span style="background:{act_color};color:#fff;font-size:9px;font-weight:700;'
            f'padding:3px 10px;border-radius:20px;white-space:nowrap">{act_icon} {act_label}</span>'
            f'<span style="background:{sig_color}22;color:{sig_color};font-size:9px;font-weight:600;'
            f'padding:3px 10px;border-radius:20px;white-space:nowrap">{sig}</span>'
            f'<span style="font-size:10px;color:#6B8CAE">{stock_icon} {stock_label}</span>'
            f'{actioned_badge}'
            f'</div>'

            f'<div style="font-size:11px;color:#6B8CAE;line-height:1.55;margin-bottom:12px">'
            f'{description}</div>'

            f'<div style="display:flex;gap:32px">'
            f'<div><div style="font-size:9px;color:#8AABCC;text-transform:uppercase;'
            f'letter-spacing:0.8px;margin-bottom:2px">Annual Opp.</div>'
            f'<div style="font-size:20px;font-weight:800;color:#0BB99F">{fmt_kes(annual_opp)}</div></div>'

            f'<div><div style="font-size:9px;color:#8AABCC;text-transform:uppercase;'
            f'letter-spacing:0.8px;margin-bottom:2px">Capture Rate</div>'
            f'<div style="font-size:20px;font-weight:800;color:#003467">{capture:.0f}%</div></div>'

            f'<div title="Lift: how much more likely B is when A is in the basket">'
            f'<div style="font-size:9px;color:#8AABCC;text-transform:uppercase;'
            f'letter-spacing:0.8px;margin-bottom:2px">Margin B</div>'
            f'<div style="font-size:20px;font-weight:800;color:#003467">'
            f'{margin_b:.1f}%</div></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        st.checkbox("Mark as actioned", key=key)

st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
st.markdown('<hr style="border:none;border-top:1px solid #EBF3FB;margin:0">', unsafe_allow_html=True)
st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)

# ══ PRODUCT PAIR FINDER ════════════════════════════════════════════════════════
section_header("Product Pair Finder", margin_top=0)
note("Select any product to see every validated pair it belongs to, ranked by opportunity.")

all_products = sorted(
    set(df["Product A"].str.title().unique()) |
    set(df["Product B"].str.title().unique())
)
selected = st.selectbox("", ["— pick a product —"] + all_products, label_visibility="collapsed")

if selected and selected != "— pick a product —":
    sel_upper = selected.upper()
    as_a = filtered[filtered["Product A"] == sel_upper][
        ["Product B", "CSUC score", "Lift", "Baskets together", "Signal strength",
         "Capture Rate %", "Monthly KES Opportunity"]
    ].copy().rename(columns={"Product B": "Recommend"})
    as_b = filtered[filtered["Product B"] == sel_upper][
        ["Product A", "CSUC score", "Lift", "Baskets together", "Signal strength",
         "Capture Rate %", "Monthly KES Opportunity"]
    ].copy().rename(columns={"Product A": "Recommend"})

    recs = pd.concat([as_a, as_b]).sort_values("Monthly KES Opportunity", ascending=False).head(6)

    if recs.empty:
        st.info("No validated pairs found for this product with the current filters.")
    else:
        cols = st.columns(min(len(recs), 3))
        for i, (_, row) in enumerate(recs.iterrows()):
            sig        = row["Signal strength"]
            sig_color  = SIGNAL_COLORS.get(sig, COLORS["muted"])
            rec_name   = str(row["Recommend"]).title()
            cap        = float(row["Capture Rate %"])
            lift_val   = float(row["Lift"])
            act_label, act_icon, act_color = action_type(sig, cap)
            with cols[i % 3]:
                st.markdown(
                    f'<div style="background:#fff;border:1px solid #D6E4F0;'
                    f'border-top:3px solid {act_color};border-radius:8px;'
                    f'padding:16px;margin-bottom:12px">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:flex-start;margin-bottom:8px">'
                    f'<span style="font-size:9px;font-weight:700;color:{sig_color};'
                    f'text-transform:uppercase;letter-spacing:1px">{sig}</span>'
                    f'<span style="background:{act_color}22;color:{act_color};'
                    f'font-size:9px;font-weight:700;padding:2px 7px;border-radius:10px">'
                    f'{act_icon} {act_label}</span>'
                    f'</div>'
                    f'<div style="font-size:14px;font-weight:700;color:#003467;'
                    f'margin-bottom:12px;line-height:1.3">{rec_name}</div>'
                    f'<div style="display:flex;gap:14px;flex-wrap:wrap">'
                    f'<div><div style="font-size:9px;color:#8AABCC;margin-bottom:2px">UPLIFT</div>'
                    f'<div style="font-size:16px;font-weight:800;color:{sig_color}">'
                    f'+{row["CSUC score"]*100:.1f}%</div></div>'
                    f'<div><div style="font-size:9px;color:#8AABCC;margin-bottom:2px">LIFT</div>'
                    f'<div style="font-size:16px;font-weight:800;color:#003467">'
                    f'{lift_val:.1f}×</div></div>'
                    f'<div><div style="font-size:9px;color:#8AABCC;margin-bottom:2px">CAPTURE</div>'
                    f'<div style="font-size:16px;font-weight:800;color:#003467">'
                    f'{cap:.0f}%</div></div>'
                    f'<div><div style="font-size:9px;color:#8AABCC;margin-bottom:2px">ANN. OPP</div>'
                    f'<div style="font-size:16px;font-weight:800;color:#0BB99F">'
                    f'{fmt_kes(row["Monthly KES Opportunity"] * 12)}</div></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
else:
    st.markdown(
        '<div style="background:#F4F8FC;border:1px dashed #D6E4F0;border-radius:8px;'
        'padding:20px;text-align:center;color:#8AABCC;font-size:12px">'
        'Select a product above to see its validated pairs and recommended store actions</div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-top:32px'></div>", unsafe_allow_html=True)
st.markdown('<hr style="border:none;border-top:1px solid #EBF3FB;margin:0">', unsafe_allow_html=True)
st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)

# ══ CHARTS ROW: KES Opportunity | Store Strength ──────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    section_header("Top 15 Pairs by Annual KES Opportunity", margin_top=0)

    opp_df  = filtered[filtered["Monthly KES Opportunity"] > 0].copy()
    opp_df["Pair"]               = opp_df["Product A"].str.title() + " + " + opp_df["Product B"].str.title()
    opp_df["Annual KES Opp"]     = (opp_df["Monthly KES Opportunity"] * 12).round(0)
    opp_top = opp_df.sort_values("Annual KES Opp", ascending=False).head(15)

    if opp_top.empty:
        st.info("No pairs with price data under current filters.")
    else:
        fig_opp = px.bar(
            opp_top.sort_values("Annual KES Opp"),
            x="Annual KES Opp", y="Pair", orientation="h",
            color="Signal strength", color_discrete_map=SIGNAL_COLORS,
            text="Annual KES Opp",
            hover_data={"Lift": True, "Capture Rate %": True, "Signal strength": False},
            height=460,
        )
        fig_opp.update_traces(
            texttemplate="KES %{x:,.0f}",
            textposition="outside",
            textfont=dict(size=9, color="#003467"),
        )
        fig_opp.update_layout(
            **CHART_LAYOUT, height=460,
            yaxis_title=None, xaxis_title="Est. annual opportunity (KES)",
            legend_title_text="Confidence",
            legend_orientation="h", legend_yanchor="bottom",
            legend_y=1.02, legend_xanchor="left", legend_x=0,
        )
        fig_opp.update_xaxes(showticklabels=False)
        st.plotly_chart(fig_opp, use_container_width=True)

with col_right:
    section_header("Buying Patterns by Store", margin_top=0)

    store_pairs = (
        chart_base[chart_base["Signal strength"].isin(["High", "Medium"])]
        .groupby(["Store", "Signal strength"])
        .size()
        .reset_index(name="Pairs")
    )

    if store_pairs.empty:
        st.info("No validated pairs under current filters.")
    else:
        fig_store = px.bar(
            store_pairs,
            x="Store", y="Pairs",
            color="Signal strength",
            color_discrete_map=SIGNAL_COLORS,
            barmode="stack",
            text="Pairs",
            height=460,
        )
        fig_store.update_traces(textposition="inside", textfont=dict(size=10, color="#ffffff"))
        fig_store.update_layout(
            **CHART_LAYOUT, height=460,
            xaxis_title=None, yaxis_title="Validated pairs",
            legend_title_text="Confidence",
            legend_orientation="h", legend_yanchor="bottom",
            legend_y=1.02, legend_xanchor="left", legend_x=0,
        )
        st.plotly_chart(fig_store, use_container_width=True)

st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
st.markdown('<hr style="border:none;border-top:1px solid #EBF3FB;margin:0">', unsafe_allow_html=True)
st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)

# ══ FULL TABLE ════════════════════════════════════════════════════════════════
section_header(f"All Validated Pairs  ·  {len(filtered):,} results", margin_top=0)
note("Sorted by annual KES opportunity. P(B|A) and baseline probabilities included in CSV download.")

display_cols = [
    "Store", "Product A", "Product B", "Signal strength",
    "Baskets together", "Baskets with A",
    "CSUC score", "Lift", "Capture Rate %",
    "Monthly KES Opportunity", "Avg Price B (KES)", "Margin B %",
]
csv_cols = display_cols + ["P(B|A)", "P(B) baseline"]

filtered_display = filtered[display_cols].sort_values("Monthly KES Opportunity", ascending=False)

st.dataframe(
    filtered_display,
    use_container_width=True,
    height=400,
    column_config={
        "CSUC score":              st.column_config.NumberColumn("Uplift Score",      format="%.4f"),
        "Lift":                    st.column_config.NumberColumn("Lift ×",            format="%.2f"),
        "Capture Rate %":          st.column_config.NumberColumn("Capture Rate",      format="%.1f%%"),
        "Monthly KES Opportunity": st.column_config.NumberColumn("Monthly Opp (KES)", format="KES %,.0f"),
        "Avg Price B (KES)":       st.column_config.NumberColumn("Avg Price B",       format="KES %,.2f"),
        "Margin B %":              st.column_config.NumberColumn("Margin B",          format="%.1f%%"),
        "Baskets together":        st.column_config.NumberColumn("Bought together",   format="%.0f"),
        "Baskets with A":          st.column_config.NumberColumn("Txns with A",       format="%.0f"),
        "Signal strength":         st.column_config.TextColumn("Confidence"),
    },
    hide_index=True,
)

csv_data = filtered[csv_cols].sort_values("Monthly KES Opportunity", ascending=False)
st.download_button(
    "⬇  Download as CSV",
    data=csv_data.to_csv(index=False).encode("utf-8"),
    file_name="xanalife_cross_sell_pairs.csv",
    mime="text/csv",
)
