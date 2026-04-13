"""
Pharma Stock Intelligence Dashboard
New Branch Stock Prediction & Inventory Optimization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from pathlib import Path
import os

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pharma Stock Intelligence",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #f8fafc;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 8px;
  }
  .metric-card.danger  { border-color: #ef4444; }
  .metric-card.warning { border-color: #f59e0b; }
  .metric-card.success { border-color: #10b981; }
  .metric-card h3 { margin:0; font-size:1.7rem; font-weight:700; }
  .metric-card p  { margin:4px 0 0; color:#64748b; font-size:.82rem; }
  .section-header { font-size:1.1rem; font-weight:600; margin:12px 0 6px; color:#1e293b; }
  div[data-testid="stMetric"] { background:#f8fafc; border-radius:8px; padding:10px; }
</style>
""", unsafe_allow_html=True)

# ── Data Loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_data():
    # Try several common locations
    candidates = [
        Path(__file__).parent / "data_export.pkl",
        Path("data_export.pkl"),
        Path("/sessions/ecstatic-dreamy-archimedes/mnt/uploads/data_export.pkl"),
    ]
    for p in candidates:
        if p.exists():
            with open(p, "rb") as f:
                raw = pickle.load(f)
            break
    else:
        st.error("data_export.pkl not found. Place it in the same folder as this script.")
        st.stop()

    disp        = raw["disp"].copy()
    inv         = raw["inv"].copy()
    disp_df     = raw["disp_df"].copy()
    pred        = raw["pred"].copy()
    pred_prod   = raw["pred_products"].copy()

    # ── Parse dates ──────────────────────────────────────────────────────────
    disp["date"]           = pd.to_datetime(disp["date"])
    inv["snapshot_date"]   = pd.to_datetime(inv["snapshot_date"])
    disp_df["months"]      = pd.to_datetime(disp_df["months"])

    # ── Velocity classification ───────────────────────────────────────────────
    vel = disp.groupby("product_id")["qty_dispensed"].sum()
    q33, q66 = vel.quantile(0.33), vel.quantile(0.66)
    vel_map = vel.apply(lambda x: "Fast" if x >= q66 else ("Medium" if x >= q33 else "Slow"))
    disp["velocity"] = disp["product_id"].map(vel_map)

    # ── Monthly trend ─────────────────────────────────────────────────────────
    disp["month"] = disp["date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        disp.groupby(["month", "facility_id"])
        .agg(qty=("qty_dispensed","sum"), sales=("total_sales_value","sum"),
             products=("product_id","nunique"))
        .reset_index()
    )

    # ── Pred: parse CI and compute stock coverage ─────────────────────────────
    def parse_ci(s):
        try:
            parts = s.replace("(80% CI)","").strip().split("–")
            return float(parts[0].strip()), float(parts[1].strip())
        except:
            return np.nan, np.nan
    pred[["ci_lo","ci_hi"]] = pred["Stock Range (80% CI)"].apply(
        lambda x: pd.Series(parse_ci(x)))
    pred["months_coverage"] = (pred["Opening Stock Qty"] / pred["Predicted Monthly (Steady State)"].replace(0, np.nan)).round(1)
    pred["overstock"]  = pred["months_coverage"] > 3
    pred["understock"] = pred["months_coverage"] < 1

    # ── Criticality scoring ───────────────────────────────────────────────────
    CRITICAL_KW = [
        "insulin","antidiabetic","antihypertensive","anticoagulant","anticonvulsant",
        "antiepileptic","cardiac","nitrate","diuretic","loop","beta-blocker","ace",
        "arb","ccb","injectable","antiretroviral","hiv","chemotherapy","alkylating",
        "oncology","immunosuppressant","transplant","corticosteroid","steroid",
        "antiarrhythmic","antianginal","antipsychotic","antidepressant","biguanide",
        "sulfonylurea","glp","sglt","thyroxine","thyroid","inhaler","bronchodilator",
        "respiratory","opioid","morphine","analgesic"
    ]
    CHRONIC_KW = [
        "antihypertensive","antidiabetic","biguanide","sulfonylurea","anticoagulant",
        "anticonvulsant","antidepressant","antipsychotic","thyroxine","thyroid",
        "antiretroviral","immunosuppressant","statin","lipid"
    ]
    def crit_score(cat):
        c = cat.lower()
        if any(k in c for k in ["insulin","antiretroviral","alkylating","cardiac arrest",
                                  "anticoagulant","antiarrhythmic","emergency"]):
            return "Life-Saving"
        if any(k in c for k in CRITICAL_KW):
            return "Critical"
        if any(k in c for k in CHRONIC_KW):
            return "Chronic"
        return "Standard"

    pred["criticality"]      = pred["Category"].apply(crit_score)
    pred_prod["criticality"] = pred_prod["Category"].apply(crit_score)

    # ── Demand variability (CV) from disp_df ─────────────────────────────────
    cv_df = (
        disp_df.groupby(["facility_id","new_category_name"])["total_qty_dispensed"]
        .agg(["mean","std","count"])
        .reset_index()
    )
    cv_df["cv"] = (cv_df["std"] / cv_df["mean"].replace(0, np.nan)).round(3)
    cv_df.rename(columns={"new_category_name":"category"}, inplace=True)

    # ── Safety stock stub (Z=1.65 for 95% SL, assume lead_time=2 weeks) ──────
    LEAD_TIME_WEEKS = 2
    cv_df["safety_stock_rec"] = (
        1.65 * cv_df["std"] * np.sqrt(LEAD_TIME_WEEKS / 4)
    ).round(0)
    cv_df["reorder_point"] = (cv_df["mean"] * (LEAD_TIME_WEEKS / 4) + cv_df["safety_stock_rec"]).round(0)

    # ── Ramp-up (facility 4 = comparable branch) ──────────────────────────────
    ramp = (
        disp_df[disp_df["facility_id"]==4]
        .groupby("months")["total_qty_dispensed"].sum()
        .reset_index().sort_values("months")
    )
    ramp["month_no"] = range(1, len(ramp)+1)
    ramp["cum_30"]   = ramp["total_qty_dispensed"].cumsum()

    return {
        "disp": disp, "inv": inv, "disp_df": disp_df,
        "pred": pred, "pred_prod": pred_prod,
        "monthly": monthly, "cv_df": cv_df, "ramp": ramp,
        "vel_q33": q33, "vel_q66": q66,
    }


d = load_data()
pred       = d["pred"]
pred_prod  = d["pred_prod"]
disp       = d["disp"]
inv        = d["inv"]
disp_df    = d["disp_df"]
monthly    = d["monthly"]
cv_df      = d["cv_df"]
ramp       = d["ramp"]

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("💊 Pharma Stock Intelligence")
st.caption("New Branch · Demand Forecasting · Risk Mitigation · Inventory Optimization")

# ── Top KPI Strip ──────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)

total_opening = pred["Opening Stock Qty"].sum()
total_pred    = pred["Predicted Monthly (Steady State)"].sum()
dead_risk_n   = int(pred["Dead Stock Risk"].sum())
high_conf     = int((pred["Confidence"]=="High").sum())
critical_n    = int((pred["criticality"].isin(["Life-Saving","Critical"])).sum())
overstock_n   = int(pred["overstock"].sum())

k1.metric("📦 Opening Stock (units)", f"{total_opening:,.0f}")
k2.metric("📈 Predicted Monthly (units)", f"{total_pred:,.0f}")
k3.metric("⚠️ Dead Stock Risk SKUs", f"{dead_risk_n}", delta=f"{dead_risk_n/len(pred)*100:.0f}% of categories", delta_color="inverse")
k4.metric("✅ High-Confidence Forecasts", f"{high_conf}/{len(pred)}", f"{high_conf/len(pred)*100:.0f}%")
k5.metric("🚑 Critical Categories", f"{critical_n}", delta="require priority")
k6.metric("📉 Overstocked (>3 mo)", f"{overstock_n}", delta_color="inverse")

st.divider()

# ── 8 Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🎯 Forecast Performance",
    "🏥 Service Level",
    "📦 Safety Stock",
    "🚑 Product Criticality",
    "🧠 Demand Patterns",
    "🚀 Branch Ramp-Up",
    "🧩 Assortment",
    "🔧 Operations",
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 · FORECAST PERFORMANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[0]:
    st.subheader("🎯 Forecast Performance Metrics")
    st.caption("How reliable are our predictions before the first order?")

    c1, c2, c3 = st.columns(3)
    conf_counts = pred["Confidence"].value_counts()
    c1.metric("High Confidence", conf_counts.get("High",0))
    c2.metric("Medium Confidence", conf_counts.get("Medium",0))
    c3.metric("Low Confidence", conf_counts.get("Low",0))

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Confidence Distribution by Category**")
        fig = px.pie(
            pred, names="Confidence",
            color="Confidence",
            color_discrete_map={"High":"#10b981","Medium":"#f59e0b","Low":"#ef4444"},
            hole=0.45,
        )
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(margin=dict(t=10,b=10), height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Stock Coverage (Months of Supply)**")
        coverage_df = pred[["Category","months_coverage","Confidence","overstock","understock"]].copy()
        fig2 = px.histogram(
            coverage_df.dropna(subset=["months_coverage"]),
            x="months_coverage", color="Confidence",
            color_discrete_map={"High":"#10b981","Medium":"#f59e0b","Low":"#ef4444"},
            nbins=30, barmode="stack",
            labels={"months_coverage":"Months of Opening Stock"},
        )
        fig2.add_vline(x=1, line_dash="dash", line_color="#ef4444", annotation_text="Min (1 mo)")
        fig2.add_vline(x=3, line_dash="dash", line_color="#f59e0b", annotation_text="Max (3 mo)")
        fig2.update_layout(margin=dict(t=10,b=10), height=300)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Bias Analysis — Over vs Under Stocked Categories**")
    bias_df = pred[["Category","Opening Stock Qty","Predicted Monthly (Steady State)",
                    "months_coverage","Confidence","Dead Stock Risk"]].copy()
    bias_df["bias_flag"] = bias_df["months_coverage"].apply(
        lambda x: "🔴 Understock" if x < 1 else ("🟡 Overstock" if x > 3 else "🟢 Optimal"))
    bias_df.columns = ["Category","Opening Stock","Predicted Monthly","Coverage (mo)","Confidence","Dead Stock","Status"]

    col_filter, _ = st.columns([2,4])
    status_filter = col_filter.selectbox("Filter by Status", ["All","🟢 Optimal","🟡 Overstock","🔴 Understock"], key="f1")
    df_show = bias_df if status_filter == "All" else bias_df[bias_df["Status"]==status_filter]
    st.dataframe(
        df_show.sort_values("Coverage (mo)").reset_index(drop=True),
        use_container_width=True, height=280,
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 · SERVICE LEVEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[1]:
    st.subheader("🏥 Service Level Metrics")
    st.caption("Availability is a patient safety metric, not just an operational one.")

    # Current inventory snapshot
    stockout_pct = inv["is_stockout"].mean() * 100
    lowstock_pct = inv["is_low_stock"].mean() * 100
    fill_rate    = 100 - stockout_pct
    total_inv_val = inv["total_inventory_value"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fill Rate %",      f"{fill_rate:.1f}%",   delta="target ≥ 95%",  delta_color="off")
    c2.metric("Stockout Rate %",  f"{stockout_pct:.1f}%", delta="⚠️ high" if stockout_pct>10 else "✅ ok", delta_color="inverse" if stockout_pct>10 else "normal")
    c3.metric("Low-Stock Rate %", f"{lowstock_pct:.1f}%")
    c4.metric("Inventory Value",  f"${total_inv_val:,.0f}")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Monthly Dispensing Volume — Both Facilities**")
        fig = px.line(
            monthly, x="month", y="qty", color="facility_id",
            color_discrete_sequence=["#3b82f6","#10b981"],
            labels={"qty":"Units Dispensed","facility_id":"Facility","month":"Month"},
        )
        fig.update_layout(margin=dict(t=10,b=10), height=280, legend_title="Facility")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Predicted Stock Coverage by Criticality**")
        cov_crit = pred.groupby("criticality")["months_coverage"].mean().reset_index()
        cov_crit.columns = ["Criticality","Avg Coverage (mo)"]
        color_map = {"Life-Saving":"#7c3aed","Critical":"#ef4444","Chronic":"#f59e0b","Standard":"#10b981"}
        fig2 = px.bar(
            cov_crit.sort_values("Avg Coverage (mo)"), x="Avg Coverage (mo)", y="Criticality",
            orientation="h", color="Criticality", color_discrete_map=color_map,
            text_auto=".1f",
        )
        fig2.add_vline(x=1, line_dash="dash", line_color="#ef4444")
        fig2.update_layout(margin=dict(t=10,b=10), height=280, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Lost Sales Risk — Categories with < 1 Month Coverage**")
    at_risk = pred[pred["months_coverage"] < 1][
        ["Category","Opening Stock Qty","Predicted Monthly (Steady State)","months_coverage","criticality","Confidence"]
    ].copy()
    at_risk.columns = ["Category","Opening Stock","Predicted Monthly","Coverage (mo)","Criticality","Confidence"]
    at_risk = at_risk.sort_values("Coverage (mo)")
    at_risk["Est. Lost Units (mo 1)"] = (at_risk["Predicted Monthly"] - at_risk["Opening Stock"]).round(0)
    if len(at_risk):
        st.dataframe(at_risk.reset_index(drop=True), use_container_width=True, height=220)
    else:
        st.success("All categories have ≥ 1 month of opening stock coverage.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 · SAFETY STOCK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[2]:
    st.subheader("📦 Safety Stock Metrics")
    st.caption("Forecasting answers *how much to expect*. Safety stock answers *how much buffer to hold*.")

    st.info("📌 Assumptions: Lead time = 2 weeks · Service level = 95% (Z = 1.65) · Based on Facility 4 historical demand")

    fac_sel = st.radio("Reference Facility", [4, 5], horizontal=True, key="ss_fac")
    ss_df = cv_df[cv_df["facility_id"]==fac_sel].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Demand Variability (CV)", f"{ss_df['cv'].mean():.2f}")
    c2.metric("High-Variability Categories (CV>1)", f"{(ss_df['cv']>1).sum()}")
    c3.metric("Avg Safety Stock (units/mo)", f"{ss_df['safety_stock_rec'].mean():,.0f}")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Demand Variability (CV) by Category**")
        top_cv = ss_df.nlargest(15, "cv")[["category","cv","mean","std"]].copy()
        fig = px.bar(
            top_cv, x="cv", y="category", orientation="h",
            color="cv", color_continuous_scale="RdYlGn_r",
            text_auto=".2f",
            labels={"cv":"Coefficient of Variation","category":"Category"},
        )
        fig.add_vline(x=1, line_dash="dash", line_color="#ef4444", annotation_text="High volatility")
        fig.update_layout(margin=dict(t=10,b=10), height=350, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Safety Stock & Reorder Point Recommendations**")
        ss_show = ss_df[["category","mean","safety_stock_rec","reorder_point","cv"]].copy()
        ss_show.columns = ["Category","Avg Monthly Demand","Safety Stock (units)","Reorder Point","CV"]
        ss_show = ss_show.sort_values("Safety Stock (units)", ascending=False).reset_index(drop=True)
        st.dataframe(ss_show.head(20), use_container_width=True, height=350)

    st.markdown("**Safety Stock vs Predicted Demand — Scatter**")
    ss_pred = ss_df.merge(
        pred[["Category","Predicted Monthly (Steady State)"]].rename(columns={"Category":"category"}),
        on="category", how="inner"
    )
    if len(ss_pred):
        fig3 = px.scatter(
            ss_pred, x="Predicted Monthly (Steady State)", y="safety_stock_rec",
            size="cv", color="cv",
            color_continuous_scale="RdYlGn_r",
            hover_name="category",
            labels={"Predicted Monthly (Steady State)":"Predicted Monthly Demand",
                    "safety_stock_rec":"Recommended Safety Stock"},
            text="category",
        )
        fig3.update_traces(textposition="top center", textfont_size=9)
        fig3.update_layout(margin=dict(t=10,b=10), height=380, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 · PRODUCT CRITICALITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[3]:
    st.subheader("🚑 Product Criticality Scoring")
    st.caption("Running out of shampoo = annoying. Running out of insulin = unacceptable.")

    crit_counts = pred["criticality"].value_counts().reset_index()
    crit_counts.columns = ["Criticality","Count"]
    color_map = {"Life-Saving":"#7c3aed","Critical":"#ef4444","Chronic":"#f59e0b","Standard":"#10b981"}

    c1, c2, c3, c4 = st.columns(4)
    for col, label in zip([c1,c2,c3,c4], ["Life-Saving","Critical","Chronic","Standard"]):
        n = int(crit_counts[crit_counts["Criticality"]==label]["Count"].values[0]) if label in crit_counts["Criticality"].values else 0
        col.metric(label, n)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Criticality Distribution (Categories)**")
        fig = px.pie(
            crit_counts, names="Criticality", values="Count",
            color="Criticality", color_discrete_map=color_map, hole=0.4,
        )
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(margin=dict(t=10,b=10), height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Coverage vs Criticality — Risk Matrix**")
        fig2 = px.box(
            pred.dropna(subset=["months_coverage"]),
            x="criticality", y="months_coverage",
            color="criticality", color_discrete_map=color_map,
            category_orders={"criticality":["Life-Saving","Critical","Chronic","Standard"]},
            labels={"months_coverage":"Coverage (months)","criticality":"Criticality"},
            points="all",
        )
        fig2.add_hline(y=1, line_dash="dash", line_color="#ef4444", annotation_text="Min 1 month")
        fig2.update_layout(margin=dict(t=10,b=10), height=300, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Critical & Life-Saving Categories — Priority Watch**")
    priority = pred[pred["criticality"].isin(["Life-Saving","Critical"])][
        ["Category","criticality","Opening Stock Qty","Predicted Monthly (Steady State)",
         "months_coverage","Confidence","Dead Stock Risk"]
    ].sort_values(["criticality","months_coverage"]).reset_index(drop=True)
    priority.columns = ["Category","Criticality","Opening Stock","Predicted Monthly","Coverage (mo)","Confidence","Dead Stock"]
    st.dataframe(priority, use_container_width=True, height=280)

    st.markdown("**Product-Level Critical Items**")
    pp_crit = pred_prod[pred_prod["criticality"].isin(["Life-Saving","Critical"])][
        ["Product","Category","criticality","Opening Stock Qty","Confidence","Dead Stock Risk"]
    ].reset_index(drop=True)
    pp_crit.columns = ["Product","Category","Criticality","Opening Stock","Confidence","Dead Stock Risk"]
    st.dataframe(pp_crit, use_container_width=True, height=240)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 · DEMAND PATTERNS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[4]:
    st.subheader("🧠 Demand Pattern Classification")
    st.caption("Different SKUs need different forecasting methods.")

    # Classify at product level from historical disp
    prod_agg = disp.groupby(["product_id","product_name","velocity"]).agg(
        total_qty=("qty_dispensed","sum"),
        active_months=("month","nunique"),
    ).reset_index()

    # Intermittent = appeared in fewer than 30% of months
    total_months = disp["month"].nunique()
    prod_agg["activity_rate"] = prod_agg["active_months"] / total_months
    prod_agg["pattern"] = prod_agg.apply(
        lambda r: "Intermittent" if r["activity_rate"] < 0.3
        else ("Fast Mover" if r["velocity"]=="Fast"
              else ("Slow Mover" if r["velocity"]=="Slow"
                    else "Regular")),
        axis=1
    )

    pattern_counts = prod_agg["pattern"].value_counts().reset_index()
    pattern_counts.columns = ["Pattern","SKUs"]
    color_map2 = {"Fast Mover":"#10b981","Regular":"#3b82f6","Slow Mover":"#f59e0b","Intermittent":"#ef4444"}

    c1, c2, c3, c4 = st.columns(4)
    for col, label in zip([c1,c2,c3,c4], ["Fast Mover","Regular","Slow Mover","Intermittent"]):
        n = int(pattern_counts[pattern_counts["Pattern"]==label]["SKUs"].values[0]) if label in pattern_counts["Pattern"].values else 0
        col.metric(label, f"{n:,} SKUs")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**SKU Distribution by Demand Pattern**")
        fig = px.pie(
            pattern_counts, names="Pattern", values="SKUs",
            color="Pattern", color_discrete_map=color_map2, hole=0.4,
        )
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(margin=dict(t=10,b=10), height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Volume Share by Pattern**")
        vol = prod_agg.groupby("pattern")["total_qty"].sum().reset_index()
        vol.columns = ["Pattern","Total Units"]
        fig2 = px.bar(
            vol.sort_values("Total Units", ascending=False),
            x="Pattern", y="Total Units",
            color="Pattern", color_discrete_map=color_map2,
            text_auto=".2s",
        )
        fig2.update_layout(margin=dict(t=10,b=10), height=300, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Seasonal Demand Trend (Facility 4)**")
    monthly_cat = disp_df[disp_df["facility_id"]==4].groupby(
        [disp_df[disp_df["facility_id"]==4]["months"].dt.month_name(), "new_category_name"]
    )["total_qty_dispensed"].sum().reset_index()
    monthly_cat.columns = ["Month","Category","Total Qty"]
    top_cats = disp_df[disp_df["facility_id"]==4].groupby("new_category_name")["total_qty_dispensed"].sum().nlargest(5).index
    seasonal = monthly_cat[monthly_cat["Category"].isin(top_cats)]
    month_order = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]
    seasonal = seasonal[seasonal["Month"].isin(month_order)]
    seasonal["Month"] = pd.Categorical(seasonal["Month"], categories=month_order, ordered=True)
    seasonal = seasonal.sort_values("Month")
    fig3 = px.line(
        seasonal, x="Month", y="Total Qty", color="Category",
        labels={"Total Qty":"Units Dispensed"},
    )
    fig3.update_layout(margin=dict(t=10,b=10), height=300)
    st.plotly_chart(fig3, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 6 · BRANCH RAMP-UP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[5]:
    st.subheader("🚀 New Branch Ramp-Up Metrics")
    st.caption("Based on Facility 4 (comparable branch). First 90 days determine your steady state.")

    ramp = d["ramp"]
    m1  = float(ramp[ramp["month_no"]==1]["total_qty_dispensed"].values[0]) if 1 in ramp["month_no"].values else 0
    m3  = float(ramp[ramp["month_no"]==3]["total_qty_dispensed"].values[0]) if 3 in ramp["month_no"].values else 0
    m6  = float(ramp[ramp["month_no"]==6]["total_qty_dispensed"].values[0]) if 6 in ramp["month_no"].values else 0
    steady = float(ramp["total_qty_dispensed"].tail(6).mean())
    ramp_pct = ((m3 - m1) / m1 * 100) if m1 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Month 1 Demand", f"{m1:,.0f} units")
    c2.metric("Month 3 Demand", f"{m3:,.0f} units", f"{ramp_pct:+.0f}% vs M1")
    c3.metric("Month 6 Demand", f"{m6:,.0f} units")
    c4.metric("Steady State (avg last 6 mo)", f"{steady:,.0f} units")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Demand Ramp-Up Curve (Facility 4 — Reference)**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ramp["month_no"], y=ramp["total_qty_dispensed"],
            mode="lines+markers", name="Monthly Demand",
            line=dict(color="#3b82f6", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=ramp["month_no"], y=ramp["cum_30"],
            mode="lines", name="Cumulative Demand",
            line=dict(color="#10b981", width=2, dash="dot"),
            yaxis="y2",
        ))
        fig.add_vrect(x0=0.5, x1=1.5, fillcolor="#ef4444", opacity=0.07, annotation_text="30d")
        fig.add_vrect(x0=0.5, x1=3.5, fillcolor="#f59e0b", opacity=0.05, annotation_text="90d")
        fig.update_layout(
            yaxis=dict(title="Monthly Units"),
            yaxis2=dict(title="Cumulative Units", overlaying="y", side="right"),
            legend=dict(orientation="h"), margin=dict(t=10,b=10), height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Top Categories — First Month vs Steady State**")
        cat_m1 = disp_df[(disp_df["facility_id"]==4) &
                          (disp_df["months"] == disp_df[disp_df["facility_id"]==4]["months"].min())
                         ].groupby("new_category_name")["total_qty_dispensed"].sum()
        cat_ss = disp_df[(disp_df["facility_id"]==4)].groupby("new_category_name")["total_qty_dispensed"].mean()
        ramp_cat = pd.DataFrame({"Month 1": cat_m1, "Avg Monthly (SS)": cat_ss}).dropna()
        ramp_cat = ramp_cat.nlargest(10, "Avg Monthly (SS)").reset_index()
        fig2 = px.bar(
            ramp_cat.melt(id_vars="new_category_name", var_name="Period", value_name="Units"),
            x="new_category_name", y="Units", color="Period",
            barmode="group",
            color_discrete_map={"Month 1":"#f59e0b","Avg Monthly (SS)":"#3b82f6"},
            labels={"new_category_name":"Category"},
        )
        fig2.update_layout(margin=dict(t=10,b=10), height=320,
                           xaxis_tickangle=-35, legend_title="Period")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**SKU Adoption Rate — Unique Products Dispensed Over Time**")
    sku_growth = disp[disp["facility_id"]==4].groupby("month")["product_id"].nunique().reset_index()
    sku_growth.columns = ["Month","Unique SKUs Active"]
    sku_growth = sku_growth.sort_values("Month")
    sku_growth["month_no"] = range(1, len(sku_growth)+1)
    fig3 = px.area(
        sku_growth, x="Month", y="Unique SKUs Active",
        color_discrete_sequence=["#3b82f6"],
    )
    fig3.update_layout(margin=dict(t=10,b=10), height=240)
    st.plotly_chart(fig3, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 7 · ASSORTMENT OPTIMIZATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[6]:
    st.subheader("🧩 Assortment Optimization Metrics")
    st.caption("Are we stocking the right products?")

    total_pred_skus    = len(pred_prod)
    dead_stock_skus    = int((pred_prod["Dead Stock Risk"]=="⚠️ Yes").sum())
    high_conf_skus     = int((pred_prod["Confidence"]=="High").sum())
    cat_coverage       = pred_prod["Category"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Predicted SKUs", f"{total_pred_skus:,}")
    c2.metric("SKUs with Dead Stock Risk", f"{dead_stock_skus}", delta_color="inverse")
    c3.metric("High-Confidence SKUs", f"{high_conf_skus}")
    c4.metric("Category Coverage", f"{cat_coverage} categories")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**SKU Count by Category (Top 20)**")
        sku_cat = pred_prod.groupby("Category").size().reset_index(name="SKU Count")
        sku_cat = sku_cat.nlargest(20, "SKU Count")
        fig = px.bar(
            sku_cat.sort_values("SKU Count"),
            x="SKU Count", y="Category", orientation="h",
            color="SKU Count", color_continuous_scale="Blues",
            text_auto=True,
        )
        fig.update_layout(margin=dict(t=10,b=10), height=420, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Dead Stock Risk by Confidence Level**")
        ds_conf = pred_prod.groupby(["Confidence","Dead Stock Risk"]).size().reset_index(name="SKUs")
        fig2 = px.bar(
            ds_conf, x="Confidence", y="SKUs", color="Dead Stock Risk",
            color_discrete_map={"⚠️ Yes":"#ef4444","✓ No":"#10b981"},
            barmode="stack", text_auto=True,
        )
        fig2.update_layout(margin=dict(t=10,b=10), height=280, legend_title="Dead Stock Risk")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Historical Share Distribution**")
        hs = pred_prod["Historical Share"].str.replace("%","").astype(float)
        fig3 = px.histogram(hs, nbins=20, labels={"value":"Historical Share %"},
                            color_discrete_sequence=["#3b82f6"])
        fig3.update_layout(margin=dict(t=10,b=10), height=160)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("**Dead Stock Risk Products — Full List**")
    dead_prods = pred_prod[pred_prod["Dead Stock Risk"]=="⚠️ Yes"][
        ["Product","Category","criticality","Opening Stock Qty","Confidence","Historical Share"]
    ].reset_index(drop=True)
    dead_prods.columns = ["Product","Category","Criticality","Opening Stock","Confidence","Historical Share"]
    st.dataframe(dead_prods, use_container_width=True, height=260)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 8 · OPERATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[7]:
    st.subheader("🔧 Operational Metrics")
    st.caption("Real-world constraints that every forecast must respect.")

    # Reorder levels from inventory
    inv_reorder = inv[inv["re_order_level"] > 0][
        ["product_name","product_id","qty_on_hand","re_order_level","smart_reorder_level","is_stockout","is_low_stock"]
    ].copy()
    needs_reorder = inv_reorder[inv_reorder["qty_on_hand"] <= inv_reorder["re_order_level"]]

    # Replenishment frequency from disp
    daily_orders = disp.groupby(["product_id","date"])["qty_dispensed"].sum().reset_index()
    active_days  = daily_orders.groupby("product_id")["date"].nunique()
    avg_active_days = active_days.mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SKUs at/below Reorder Point", f"{len(needs_reorder):,}")
    c2.metric("SKUs with Smart Reorder Set", f"{(inv['smart_reorder_level']>0).sum():,}")
    c3.metric("Avg SKU Active Days / Month", f"{avg_active_days:.0f}")
    c4.metric("Inventory Snapshot Date", str(inv["snapshot_date"].max().date()))

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Reorder Level vs. Stock on Hand (Top 30 Products)**")
        top_ro = inv_reorder.nlargest(30, "re_order_level").copy()
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Qty on Hand",  x=top_ro["product_name"], y=top_ro["qty_on_hand"],  marker_color="#3b82f6"))
        fig.add_trace(go.Bar(name="Reorder Level",x=top_ro["product_name"], y=top_ro["re_order_level"],marker_color="#ef4444", opacity=0.6))
        fig.update_layout(
            barmode="overlay", xaxis_tickangle=-45,
            margin=dict(t=10,b=10), height=350,
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Dispensing Frequency Distribution**")
        freq_df = disp.groupby("product_id").agg(
            days_active=("date","nunique"),
            total_qty=("qty_dispensed","sum"),
        ).reset_index()
        fig2 = px.histogram(
            freq_df, x="days_active", nbins=30,
            labels={"days_active":"Days Active (out of total)"},
            color_discrete_sequence=["#10b981"],
        )
        fig2.update_layout(margin=dict(t=10,b=10), height=200)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Payment Mode Split (Historical)**")
        pm = disp["payment_mode"].value_counts().reset_index()
        pm.columns = ["Mode","Count"]
        fig3 = px.pie(pm, names="Mode", values="Count",
                      color_discrete_sequence=["#3b82f6","#10b981"], hole=0.5)
        fig3.update_traces(textposition="outside", textinfo="percent+label")
        fig3.update_layout(margin=dict(t=0,b=0), height=180, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("**Products Currently Below Reorder Point**")
    if len(needs_reorder):
        needs_reorder["gap"] = needs_reorder["re_order_level"] - needs_reorder["qty_on_hand"]
        st.dataframe(
            needs_reorder[["product_name","qty_on_hand","re_order_level","gap","is_stockout"]]
            .sort_values("gap", ascending=False).reset_index(drop=True),
            use_container_width=True, height=240,
        )
    else:
        st.success("All products are above their reorder points.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("💊 Pharma Stock Intelligence · Data: Facility 4 & 5 Historical · New Branch Forecast · Built with Streamlit")