import streamlit as st
import pandas as pd
import plotly.express as px
from utils.snowflake_conn import run_query
from utils.queries import CSUC_QUERY, STORES_QUERY
from utils.theme import (
    inject_css, COLORS, CHART_LAYOUT,
    kpi_card, section_header, page_banner, sidebar_nav, fmt_kes,
)

st.set_page_config(page_title="XanaLife · Cross-Sell", layout="wide", initial_sidebar_state="expanded")
inject_css()

DATA_MONTHS = 6.5   # Sep 2025 – Mar 2026

def note(text):
    st.markdown(
        f'<p style="font-size:11px;color:#8AABCC;margin:2px 0 14px;line-height:1.5">{text}</p>',
        unsafe_allow_html=True,
    )

def action_type(signal, capture_pct, lift):
    """Return (label, icon, color, description template) for a product pair."""
    if signal == "High" and capture_pct < 40:
        return "Bundle Deal",      "🏷️", "#7F77DD"
    elif lift >= 2.5:
        return "Place Together",   "🏪", COLORS["primary"]
    else:
        return "Promote Together", "📢", COLORS["success"]

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading..."):
    df     = run_query(CSUC_QUERY)
    stores = run_query(STORES_QUERY)

if df.empty:
    st.warning("No data returned.")
    st.stop()

df.columns = [c.strip() for c in df.columns]
df["CSUC score"]         = pd.to_numeric(df["CSUC score"],         errors="coerce")
df["Lift"]               = pd.to_numeric(df["Lift"],               errors="coerce")
df["Baskets with A"]     = pd.to_numeric(df["Baskets with A"],     errors="coerce")
df["Baskets together"]   = pd.to_numeric(df["Baskets together"],   errors="coerce")
df["P(B|A)"]             = pd.to_numeric(df["P(B|A)"],             errors="coerce")
df["P(B) baseline"]      = pd.to_numeric(df["P(B) baseline"],      errors="coerce")
df["Avg Price B (KES)"]  = pd.to_numeric(df["Avg Price B (KES)"],  errors="coerce").fillna(0)

# ── Derived columns ───────────────────────────────────────────────────────────
df["Opportunity (baskets)"]   = (df["Baskets with A"] * df["CSUC score"]).round(0).astype("Int64")
df["Capture Rate %"]          = (df["P(B|A)"] * 100).round(1)
# Monthly KES opportunity = extra baskets over period / months * price B * 50% conversion assumption
df["Monthly KES Opportunity"] = (
    df["Opportunity (baskets)"].astype(float) / DATA_MONTHS
    * df["Avg Price B (KES)"]
    * 0.50
).round(0)

SIGNAL_COLORS = {"High": COLORS["success"], "Medium": COLORS["warning"], "Low": COLORS["muted"]}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    sidebar_nav()
    section_header("Filters", margin_top=0)

    store_options = ["All stores"] + stores["STORE_NAME"].dropna().tolist()
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
        help="Uplift = how much more likely B is bought when A is in the basket",
    )
    search_product = st.text_input("Search product", placeholder="e.g. CHAPATI")

    st.markdown('<div style="border-bottom:1px solid #D6E4F0;margin:20px 0"></div>', unsafe_allow_html=True)
    section_header("How to read this", margin_top=0)
    for label, desc in [
        ("Uplift score",   "Extra probability of buying B when A is in the basket"),
        ("Lift ×",         ">1.2 = genuine pattern, not random chance"),
        ("Capture rate",   "% of A transactions that already include B"),
        ("Monthly opp.",   "Extra KES/month at 50% additional conversion"),
        ("Bundle Deal",    "Strong signal, low capture — a combo price would help"),
        ("Place Together", "Very strong affinity — adjacency on shelf drives it passively"),
        ("Promote",        "Validated pair — good for WhatsApp / flyer campaigns"),
    ]:
        st.markdown(
            f'<div style="margin-bottom:7px">'
            f'<span style="font-size:10px;font-weight:700;color:#003467">{label}</span><br>'
            f'<span style="font-size:10px;color:#8AABCC">{desc}</span></div>',
            unsafe_allow_html=True,
        )

# ── Filter (applied to all sections below) ────────────────────────────────────
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

# Store cross-sell chart needs all stores visible but still respects
# confidence and uplift filters (just not the store selector)
chart_base = df[
    df["Signal strength"].isin(signal_filter) &
    (df["CSUC score"] >= min_csuc)
].copy()

# ── Page header ───────────────────────────────────────────────────────────────
page_banner(
    title    = "Cross-Sell Intelligence",
    subtitle = "Identifies product pairs your customers already buy together — and tells you what to do "
               "with each pattern: adjust placement, build a bundle deal, or amplify through promotions.",
    tag      = "Cross-Sell Intelligence",
)

# ── KPIs ──────────────────────────────────────────────────────────────────────
total_monthly_kes = filtered["Monthly KES Opportunity"].sum()
high_conf         = len(filtered[filtered["Signal strength"] == "High"])
avg_capture       = filtered["Capture Rate %"].mean() if len(filtered) else 0

c1, c2, c3, c4 = st.columns(4)
with c1: kpi_card(
    "Monthly Revenue Opportunity",
    fmt_kes(total_monthly_kes),
    "at 50% additional conversion",
    COLORS["success"],
)
with c2: kpi_card(
    "High-Confidence Pairs",
    f"{high_conf:,}",
    "50+ validated transactions",
    COLORS["primary"],
)
with c3: kpi_card(
    "Validated Pairs",
    f"{len(filtered):,}",
    "matching current filters",
    COLORS["muted"],
)
with c4: kpi_card(
    "Avg Capture Rate",
    f"{avg_capture:.0f}%" if len(filtered) else "—",
    "of co-purchases already happening",
    COLORS["warning"],
)

st.markdown("<div style='margin-top:36px'></div>", unsafe_allow_html=True)

# ══ ACTION PLAYBOOK ══════════════════════════════════════════════════════════
section_header("Action Playbook — Top Opportunities", margin_top=0)
note(
    "Top 5 pairs ranked by monthly KES opportunity. "
    "Each shows the recommended store action based on the signal strength and capture rate."
)

playbook = (
    filtered[
        filtered["Signal strength"].isin(["High", "Medium"]) &
        (filtered["Monthly KES Opportunity"] > 0)
    ]
    .sort_values("Monthly KES Opportunity", ascending=False)
    .head(5)
)

ACTION_DESCRIPTIONS = {
    "Bundle Deal": (
        "Only {capture:.0f}% of customers buying {a} currently take {b}. "
        "A combo price (e.g. 'Buy both, save X%') converts more without any extra effort."
    ),
    "Place Together": (
        "Customers are {lift:.1f}× more likely to buy these together than by chance. "
        "Placing {a} and {b} adjacently — same shelf or end-cap — captures this passively."
    ),
    "Promote Together": (
        "A validated buying pattern across {baskets:,} transactions. "
        "Featuring {a} + {b} in your next WhatsApp blast or weekly flyer will amplify it."
    ),
}

if playbook.empty:
    st.info("No high-confidence pairs with price data available for the current filters.")
else:
    for _, row in playbook.iterrows():
        sig       = row["Signal strength"]
        sig_color = SIGNAL_COLORS.get(sig, COLORS["muted"])
        capture   = float(row["Capture Rate %"])
        gap       = max(0.0, 100.0 - capture)
        opp_kes   = row["Monthly KES Opportunity"]
        lift      = float(row["Lift"])
        prod_a    = str(row["Product A"]).title()
        prod_b    = str(row["Product B"]).title()
        baskets   = int(row["Baskets together"])

        act_label, act_icon, act_color = action_type(sig, capture, lift)
        desc_tmpl = ACTION_DESCRIPTIONS[act_label]
        description = desc_tmpl.format(
            capture=capture, a=prod_a, b=prod_b, lift=lift, baskets=baskets
        )

        st.markdown(
            f'<div style="background:#fff;border:1px solid #D6E4F0;'
            f'border-left:4px solid {act_color};border-radius:8px;'
            f'padding:16px 20px;margin-bottom:10px">'

            # ── top row: pair + badges
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;flex-wrap:wrap">'
            f'<span style="font-size:14px;font-weight:800;color:#003467">{prod_a}</span>'
            f'<span style="font-size:11px;color:#B0C8E0;font-weight:600">+</span>'
            f'<span style="font-size:14px;font-weight:800;color:#0072CE">{prod_b}</span>'
            f'<span style="background:{act_color};color:#fff;font-size:9px;font-weight:700;'
            f'padding:3px 10px;border-radius:20px;white-space:nowrap">'
            f'{act_icon} {act_label}</span>'
            f'<span style="background:{sig_color}22;color:{sig_color};font-size:9px;font-weight:700;'
            f'padding:3px 10px;border-radius:20px;white-space:nowrap">{sig} confidence</span>'
            f'</div>'

            # ── description
            f'<div style="font-size:12px;color:#6B8CAE;line-height:1.6;margin-bottom:14px">'
            f'{description}</div>'

            # ── metrics row
            f'<div style="display:flex;gap:28px;flex-wrap:wrap">'

            f'<div><div style="font-size:9px;color:#8AABCC;text-transform:uppercase;'
            f'letter-spacing:0.8px;margin-bottom:3px">Monthly opp.</div>'
            f'<div style="font-size:18px;font-weight:800;color:#0BB99F">{fmt_kes(opp_kes)}</div></div>'

            f'<div><div style="font-size:9px;color:#8AABCC;text-transform:uppercase;'
            f'letter-spacing:0.8px;margin-bottom:3px">Capture rate</div>'
            f'<div style="font-size:18px;font-weight:800;color:#003467">{capture:.0f}%</div></div>'

            f'<div><div style="font-size:9px;color:#8AABCC;text-transform:uppercase;'
            f'letter-spacing:0.8px;margin-bottom:3px">Gap to close</div>'
            f'<div style="font-size:18px;font-weight:800;color:#f97316">{gap:.0f}%</div></div>'

            f'<div><div style="font-size:9px;color:#8AABCC;text-transform:uppercase;'
            f'letter-spacing:0.8px;margin-bottom:3px">Lift</div>'
            f'<div style="font-size:18px;font-weight:800;color:#003467">{lift:.1f}×</div></div>'

            f'<div><div style="font-size:9px;color:#8AABCC;text-transform:uppercase;'
            f'letter-spacing:0.8px;margin-bottom:3px">Transactions</div>'
            f'<div style="font-size:18px;font-weight:800;color:#003467">{baskets:,}</div></div>'

            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
st.markdown('<hr style="border:none;border-top:1px solid #EBF3FB;margin:0">', unsafe_allow_html=True)
st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)

# ══ PRODUCT FINDER ════════════════════════════════════════════════════════════
section_header("Product Pair Finder", margin_top=0)
note("Select any product to see every validated pair it belongs to, ranked by monthly opportunity.")

# Dropdown shows all products from the full dataset regardless of filters
all_products = sorted(
    set(df["Product A"].str.title().unique()) |
    set(df["Product B"].str.title().unique())
)
selected = st.selectbox("", ["— pick a product —"] + all_products, label_visibility="collapsed")

if selected and selected != "— pick a product —":
    sel_upper = selected.upper()
    # Recs come from `filtered` so store / confidence / uplift filters apply
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
        st.info("No validated pairs found for this product with the current filters. Try relaxing the confidence or uplift filters.")
    else:
        cols = st.columns(min(len(recs), 3))
        for i, (_, row) in enumerate(recs.iterrows()):
            sig        = row["Signal strength"]
            sig_color  = SIGNAL_COLORS.get(sig, COLORS["muted"])
            rec_name   = str(row["Recommend"]).title()
            cap        = float(row["Capture Rate %"])
            lift_val   = float(row["Lift"])
            act_label, act_icon, act_color = action_type(sig, cap, lift_val)
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
                    f'<div><div style="font-size:9px;color:#8AABCC;margin-bottom:2px">OPP/MO</div>'
                    f'<div style="font-size:16px;font-weight:800;color:#0BB99F">'
                    f'{fmt_kes(row["Monthly KES Opportunity"])}</div></div>'
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
    section_header("Top 15 Pairs by Monthly KES Opportunity", margin_top=0)
    note("Ranked by estimated monthly revenue at 50% additional conversion. Colour shows confidence level.")

    opp_df  = filtered[filtered["Monthly KES Opportunity"] > 0].copy()
    opp_df["Pair"] = opp_df["Product A"].str.title() + " + " + opp_df["Product B"].str.title()
    opp_top = opp_df.sort_values("Monthly KES Opportunity", ascending=False).head(15)

    if opp_top.empty:
        st.info("No pairs with price data under current filters.")
    else:
        fig_opp = px.bar(
            opp_top.sort_values("Monthly KES Opportunity"),
            x="Monthly KES Opportunity", y="Pair", orientation="h",
            color="Signal strength", color_discrete_map=SIGNAL_COLORS,
            text="Monthly KES Opportunity",
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
            yaxis_title=None, xaxis_title="Est. monthly opportunity (KES)",
            legend_title_text="Confidence",
            legend_orientation="h", legend_yanchor="bottom",
            legend_y=1.02, legend_xanchor="left", legend_x=0,
        )
        fig_opp.update_xaxes(showticklabels=False)
        st.plotly_chart(fig_opp, use_container_width=True)

with col_right:
    section_header("Buying Patterns by Store", margin_top=0)
    note("Number of validated pairs per store. Tells you where customers are naturally co-purchasing most.")

    # Uses chart_base (all stores, respects confidence + uplift but not store selector)
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

# ══ MOST CONNECTED PRODUCTS ════════════════════════════════════════════════════
section_header("Most Connected Products", margin_top=0)
note(
    "Products that appear in the most validated pairs. These are your hub products — "
    "ideal candidates for end-cap displays, bundle pricing, and promotional features."
)

strong_f  = filtered[filtered["Signal strength"].isin(["High", "Medium"])]
counts_a  = strong_f.groupby("Product A").size().reset_index(name="count").rename(columns={"Product A": "Product"})
counts_b  = strong_f.groupby("Product B").size().reset_index(name="count").rename(columns={"Product B": "Product"})
hub_counts = (
    pd.concat([counts_a, counts_b])
    .groupby("Product")["count"].sum()
    .reset_index()
    .sort_values("count", ascending=False)
    .head(15)
)

if hub_counts.empty:
    st.info("No data under current filters.")
else:
    fig_hub = px.bar(
        hub_counts.sort_values("count"),
        x="count", y="Product", orientation="h",
        color="count",
        color_continuous_scale=[[0, "#D6E4F0"], [1, COLORS["primary"]]],
        labels={"count": "Validated pairs"},
        height=420,
    )
    fig_hub.update_layout(
        **CHART_LAYOUT, height=420,
        yaxis_title=None, xaxis_title="Number of validated co-purchase pairs",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_hub, use_container_width=True)

st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
st.markdown('<hr style="border:none;border-top:1px solid #EBF3FB;margin:0">', unsafe_allow_html=True)
st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)

# ══ FULL TABLE ════════════════════════════════════════════════════════════════
section_header(f"All Validated Pairs  ·  {len(filtered):,} results", margin_top=0)
note("Sorted by monthly KES opportunity. Download to share with store managers.")

display_cols = [
    "Store", "Product A", "Product B", "Signal strength",
    "Baskets together", "Baskets with A",
    "CSUC score", "Lift", "Capture Rate %",
    "Monthly KES Opportunity", "Avg Price B (KES)",
]
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
        "Baskets together":        st.column_config.NumberColumn("Bought together",   format="%d"),
        "Baskets with A":          st.column_config.NumberColumn("Txns with A",       format="%d"),
        "Signal strength":         st.column_config.TextColumn("Confidence"),
    },
    hide_index=True,
)

st.download_button(
    "⬇  Download as CSV",
    data=filtered_display.to_csv(index=False).encode("utf-8"),
    file_name="xanalife_cross_sell_pairs.csv",
    mime="text/csv",
)
