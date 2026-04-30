"""
app_health_insurance.py

Health Insurance Strategy Console — a consultant-grade analytics dashboard
built directly on the Kenyan Civil Servants medical-claims book
(healthinsurance.csv, 224,893 paid-claim rows, 53 columns).

Run with:
    streamlit run app_health_insurance.py

The script reads `healthinsurance.csv` from the same directory as the script
(or the WSL path noted below).  No synthetic fallback, no schema mapping —
the dashboard is wired directly against the real columns.
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# ─── CONFIG ───────────────────────────────────────────────────────────────────

# Hardcoded data location — no fallback to synthetic data.
CSV_CANDIDATES = [
    "healthinsurance.csv",                                        # next to the script
]

PAGE_TITLE = "Health Insurance Strategy Console"

st.set_page_config(
    page_title=f"Wingspan · {PAGE_TITLE}",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── THEME ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Montserrat',sans-serif;background:#fff;color:#003467}
.stApp{background:#fff}
[data-testid="stSidebar"]{background:#F4F8FC;border-right:1px solid #D6E4F0}
[data-testid="stSidebar"] *{color:#003467!important;font-family:'Montserrat',sans-serif!important}
.sh{font-size:10px;font-weight:800;color:#0072CE;text-transform:uppercase;
    letter-spacing:2.5px;padding:8px 0;border-bottom:2px solid #EBF3FB;margin-bottom:16px}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700}
.stButton button{background:#0072CE!important;color:#fff!important;border:none!important;
  font-family:'Montserrat',sans-serif!important;font-size:11px!important;font-weight:700!important;
  letter-spacing:1px!important;padding:8px 18px!important;border-radius:6px!important}
.stButton button:hover{background:#003467!important}
[data-baseweb="tab"]{font-family:'Montserrat',sans-serif!important;font-weight:600!important;
  color:#6B8CAE!important;font-size:12px!important}
[aria-selected="true"]{color:#0072CE!important;border-bottom-color:#0072CE!important}
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-thumb{background:#B0C8E0;border-radius:10px}

.narrative {
    background: linear-gradient(135deg,#003467 0%,#0072CE 100%);
    color:#fff!important;
    padding: 22px 26px; border-radius:10px; margin-bottom:18px;
}
.narrative h3 {
    color:#fff!important;
    font-size:14px; font-weight:800; letter-spacing:2px;
    text-transform:uppercase; margin:0 0 8px 0;
}
.narrative p {
    color:#E5F1FB!important;
    font-size:13px; line-height:1.55; margin:0;
}

.chip {display:inline-block;padding:3px 10px;border-radius:14px;font-size:10px;
       font-weight:700;letter-spacing:.6px;text-transform:uppercase;margin-right:6px}
.chip-low   {background:#E6F7F2;color:#0BB99F}
.chip-mid   {background:#FFF2DC;color:#D97706}
.chip-high  {background:#FCE5EC;color:#E11D48}
.chip-elite {background:#EBF3FB;color:#0072CE}
</style>
""", unsafe_allow_html=True)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

CCY = "KSh"


def fmt_ksh(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    if abs(v) >= 1_000_000_000:
        return f"{CCY} {v/1_000_000_000:.2f}B"
    if abs(v) >= 1_000_000:
        return f"{CCY} {v/1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"{CCY} {v/1_000:.1f}K"
    return f"{CCY} {v:,.0f}"


def fmt_pct(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v*100:.1f}%"


def fmt_int(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{int(v):,}"


def kpi_card(label, value, sub="", color="#003467"):
    st.markdown(
        f'<div style="background:#F4F8FC;border:1px solid #D6E4F0;'
        f'border-radius:8px;padding:18px 16px;height:118px">'
        f'<div style="font-size:10px;font-weight:700;color:#6B8CAE;'
        f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px">{label}</div>'
        f'<div style="font-size:24px;font-weight:800;color:{color};line-height:1.05">{value}</div>'
        f'<div style="font-size:11px;color:#6B8CAE;margin-top:8px">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_header(text, margin_top=0):
    style = f"margin-top:{margin_top}px" if margin_top else ""
    st.markdown(f'<div class="sh" style="{style}">{text}</div>', unsafe_allow_html=True)


def info_card(text, border_color="#0072CE"):
    st.markdown(
        f'<div style="padding:10px 14px;background:#F4F8FC;'
        f'border-left:3px solid {border_color};border-radius:4px;'
        f'font-size:12px;color:#003467;margin-bottom:10px;line-height:1.55">{text}</div>',
        unsafe_allow_html=True,
    )


def narrative(title, body):
    st.markdown(
        f'<div class="narrative"><h3>{title}</h3><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


BASE_LAYOUT = dict(
    paper_bgcolor="#fff",
    plot_bgcolor="#fff",
    font=dict(family="Montserrat", color="#003467"),
    margin=dict(l=0, r=0, t=20, b=30),
)
CHART_LAYOUT = dict(
    **BASE_LAYOUT,
    xaxis=dict(gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")),
    yaxis=dict(gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")),
)

COLORS = {
    "primary": "#0072CE", "navy": "#003467", "success": "#0BB99F",
    "warning": "#D97706", "danger": "#E11D48", "muted": "#6B8CAE",
    "purple": "#7F77DD", "pink": "#D4537E", "coral": "#D85A30",
    "green": "#1D9E75",  "soft": "#EBF3FB",
}
CAT_PALETTE = [
    COLORS["primary"], COLORS["success"], COLORS["warning"],
    COLORS["danger"],  COLORS["purple"],  COLORS["pink"],
    COLORS["coral"],   COLORS["green"],   COLORS["navy"],
]


# ─── DATA LOAD ────────────────────────────────────────────────────────────────


def _resolve_csv_path() -> str:
    for p in CSV_CANDIDATES:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(
        "healthinsurance.csv not found.  Place it next to this script "
        "or update CSV_CANDIDATES at the top of the file."
    )


@st.cache_data(show_spinner="Loading 224k claims from healthinsurance.csv …")
def load_data() -> pd.DataFrame:
    path = _resolve_csv_path()
    df = pd.read_csv(path, low_memory=False)

    # standardise column names — strip + lower + spaces→underscore
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # numeric coercions
    num_cols = [
        "age_or_dob", "los", "total_bill", "claim_total",
        "case_code", "icd10_chap", "family_size",
        "coverage_limit_kes", "deductible_pct", "claim_frequency",
        "avg_claim_amount", "total_claims_cost", "annual_premium",
        "loss_ratio", "profit_margin", "risk_score",
        "fraud_probability", "churn_probability", "county_cost_index",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Dates
    for c in ("adm_date", "dis_date", "claim_date", "batch_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", format="mixed")

    # Derive age (claim_date_year − dob_year). dob is stored in age_or_dob.
    if "claim_date" in df.columns and "age_or_dob" in df.columns:
        claim_year = df["claim_date"].dt.year
        df["age"] = (claim_year - df["age_or_dob"]).clip(lower=0, upper=110)
    else:
        df["age"] = np.nan

    # Hospital category labelling
    cat_map = {"P": "Private", "G": "Government", "M": "Mission", "F": "Faith-based"}
    if "hosp_cat" in df.columns:
        df["hosp_cat_label"] = df["hosp_cat"].map(cat_map).fillna(df["hosp_cat"])

    # Yes/No → bool helpers
    for c in ("smoker", "wellness_program", "telemedicine_usage",
              "mpesa_usage", "nhif_usage"):
        if c in df.columns:
            df[c + "_b"] = df[c].astype(str).str.strip().str.lower().eq("yes")

    # Age bands
    df["age_band"] = pd.cut(
        df["age"],
        bins=[-1, 17, 24, 34, 44, 54, 64, 200],
        labels=["<18", "18–24", "25–34", "35–44", "45–54", "55–64", "65+"],
    ).astype(str)

    # Claim payout ratio (claim_total / total_bill) — copay reality
    df["payout_ratio"] = np.where(
        df["total_bill"].fillna(0) > 0,
        df["claim_total"].fillna(0) / df["total_bill"].replace(0, np.nan),
        np.nan,
    ).clip(0, 1.5)

    # Net underwriting result per row
    df["net_uw"] = df["annual_premium"].fillna(0) - df["total_claims_cost"].fillna(0)

    # Behavioural risk class — derive from existing risk_score column.
    df["risk_tier"] = pd.cut(
        df["risk_score"],
        bins=[-0.01, 0.25, 0.50, 0.75, 1.01],
        labels=["Low", "Moderate", "High", "Critical"],
    ).astype(str)

    # Fraud-flag tier
    df["fraud_tier"] = pd.cut(
        df["fraud_probability"],
        bins=[-0.01, 0.10, 0.25, 0.50, 1.01],
        labels=["Clean", "Watch", "Investigate", "High alert"],
    ).astype(str)

    return df


try:
    df_raw = load_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div style="font-size:18px;font-weight:800;color:#0072CE;'
        'padding:8px 0 4px">⚕ HealthLens KE</div>'
        '<div style="font-size:10px;color:#6B8CAE;letter-spacing:2px;'
        'text-transform:uppercase;margin-bottom:18px">Strategy Console</div>',
        unsafe_allow_html=True,
    )

    st.caption(f"Data · {len(df_raw):,} rows · "
               f"{df_raw['claim_date'].min():%b %Y} – {df_raw['claim_date'].max():%b %Y}")

    section_header("Filters")

    fy_opts = sorted(df_raw["financial_year"].dropna().unique().tolist())
    sel_fy = st.multiselect("Financial year", fy_opts, default=fy_opts)

    quarter_opts = ["Q1", "Q2", "Q3", "Q4"]
    sel_quarter = st.multiselect("Quarter", quarter_opts, default=quarter_opts)

    cnty_opts = sorted(df_raw["county_name"].dropna().unique().tolist())
    sel_cnty = st.multiselect("County", cnty_opts, default=cnty_opts)

    cat_opts = sorted(df_raw["hosp_cat_label"].dropna().unique().tolist())
    sel_cat = st.multiselect("Hospital category", cat_opts, default=cat_opts)

    plan_opts = sorted(df_raw["plan_tier"].dropna().unique().tolist())
    sel_plan = st.multiselect("Plan tier", plan_opts, default=plan_opts)

    income_opts = ["Low", "Middle", "High"]
    sel_income = st.multiselect("Income band", income_opts, default=income_opts)

    emp_opts = sorted(df_raw["employment_type"].dropna().unique().tolist())
    sel_emp = st.multiselect("Employment", emp_opts, default=emp_opts)

    sex_opts = sorted(df_raw["p_sex"].dropna().unique().tolist())
    sel_sex = st.multiselect("Sex", sex_opts, default=sex_opts)

    disease_cat_opts = sorted(df_raw["disease_category"].dropna().unique().tolist())
    sel_dcat = st.multiselect("Disease category", disease_cat_opts, default=disease_cat_opts)

    section_header("Strategy levers", 16)
    smoker_quit_rate = st.slider("Smoker quit rate", 0.0, 1.0, 0.20, 0.05)
    wellness_uplift = st.slider("Wellness program cost-reduction (per enrollee)", 0.0, 0.30, 0.12, 0.01)
    fraud_recovery = st.slider("Fraud recovery rate", 0.0, 1.0, 0.50, 0.05)
    target_loss_ratio = st.slider("Target loss ratio", 0.50, 1.20, 0.85, 0.05)


# Apply filters
mask = (
    df_raw["financial_year"].isin(sel_fy)
    & df_raw["quarters"].isin(sel_quarter)
    & df_raw["county_name"].isin(sel_cnty)
    & df_raw["hosp_cat_label"].isin(sel_cat)
    & df_raw["plan_tier"].isin(sel_plan)
    & df_raw["income_band"].isin(sel_income)
    & df_raw["employment_type"].isin(sel_emp)
    & df_raw["p_sex"].isin(sel_sex)
    & df_raw["disease_category"].isin(sel_dcat)
)
dff = df_raw.loc[mask].copy()

if len(dff) == 0:
    st.warning("No rows match your filters. Loosen the controls in the sidebar.")
    st.stop()


# ─── PAGE HEADER ──────────────────────────────────────────────────────────────

st.markdown(
    f'<p style="font-size:11px;font-weight:800;letter-spacing:3px;'
    f'text-transform:uppercase;color:#0072CE;margin-bottom:4px">'
    f'Wingspan · {PAGE_TITLE}</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<h2 style="font-size:24px;font-weight:800;color:#003467;margin:0 0 4px 0">'
    'Where the next shilling of medical cost will come from — and how to bend it.</h2>'
    '<p style="font-size:13px;color:#6B8CAE;margin-bottom:18px">'
    'Descriptive · Qualitative · Predictive · Transformative analytics on the '
    'Civil Servants medical book.</p>',
    unsafe_allow_html=True,
)


# ─── HEADLINE KPIs ────────────────────────────────────────────────────────────

n_rows       = len(dff)
n_members    = dff["mem_id"].nunique()
n_hospitals  = dff["hosp_name"].nunique()
n_counties   = dff["county_name"].nunique()
total_bill   = dff["total_bill"].sum()
total_claim  = dff["claim_total"].sum()
total_premium = dff.drop_duplicates("mem_id")["annual_premium"].sum()
overall_lr   = (
    dff.drop_duplicates("mem_id")["total_claims_cost"].sum()
    / max(total_premium, 1)
)
mean_los     = dff["los"].mean()
fraud_alerts = (dff["fraud_probability"] >= 0.5).sum()
avg_fraud    = dff["fraud_probability"].mean()
churn_at_risk = (dff["churn_probability"] >= 0.5).sum()
smoker_share = dff.drop_duplicates("mem_id")["smoker_b"].mean()

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Claims in scope", fmt_int(n_rows),
             f"{fmt_int(n_members)} unique members", COLORS["primary"])
with c2:
    kpi_card("Total claim payout", fmt_ksh(total_claim),
             f"on {fmt_ksh(total_bill)} of billings", COLORS["navy"])
with c3:
    kpi_card("Portfolio loss ratio", fmt_pct(overall_lr),
             f"target {fmt_pct(target_loss_ratio)}",
             COLORS["danger"] if overall_lr > target_loss_ratio else COLORS["success"])
with c4:
    kpi_card("Fraud alerts (≥50%)", fmt_int(fraud_alerts),
             f"avg fraud-prob {fmt_pct(avg_fraud)}",
             COLORS["warning"])

st.markdown("<div style='margin-bottom:18px'></div>", unsafe_allow_html=True)


# ─── TABS ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "◉  Executive pulse",
    "△  Member & risk DNA",
    "✚  Provider & disease",
    "Σ  Predictive engine",
    "↗  Transformative strategy",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE PULSE
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    narrative(
        "Executive pulse",
        "Five financial years of paid claims on the Civil Servants book. "
        "We surface the shape of the cost curve, where claims concentrate, "
        "and how the loss ratio is trending. This is the first ten minutes "
        "of an executive committee meeting — not the operating dashboard."
    )

    # --- Row A: claims volume + loss ratio over time -----------------------
    section_header("Claims volume & loss ratio over time")
    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        ts = dff.dropna(subset=["claim_date"]).copy()
        ts["yq"] = ts["claim_date"].dt.to_period("Q").astype(str)
        agg = ts.groupby("yq", as_index=False).agg(
            claims=("doc_no", "count"),
            payout=("claim_total", "sum"),
        ).sort_values("yq")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=agg["yq"], y=agg["claims"],
            marker_color=COLORS["primary"], opacity=0.85,
            name="Claims",
            yaxis="y",
            hovertemplate="<b>%{x}</b><br>Claims %{y:,}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=agg["yq"], y=agg["payout"], name="Payout (KSh)",
            mode="lines+markers", yaxis="y2",
            line=dict(color=COLORS["danger"], width=3),
            marker=dict(size=7, color=COLORS["danger"]),
            hovertemplate="<b>%{x}</b><br>Payout %{y:,.0f}<extra></extra>",
        ))
        fig.update_layout(
            **BASE_LAYOUT, height=360,
            xaxis=dict(title="Quarter", gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
            yaxis=dict(title="Claims", gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
            yaxis2=dict(title="Payout (KSh)", overlaying="y", side="right",
                        gridcolor="rgba(0,0,0,0)",
                        tickfont=dict(size=10, color="#6B8CAE")),
            legend=dict(orientation="h", y=1.10, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Loss-ratio trend (per FY, member-level dedup)
        lr = (
            dff.drop_duplicates("mem_id")
               .groupby("financial_year", as_index=False)
               .agg(prem=("annual_premium", "sum"),
                    cost=("total_claims_cost", "sum"))
        ).sort_values("financial_year")
        lr["lr"] = lr["cost"] / lr["prem"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lr["financial_year"], y=lr["lr"],
            mode="lines+markers+text",
            line=dict(color=COLORS["danger"], width=3),
            marker=dict(size=10, color=COLORS["danger"],
                        line=dict(color="#fff", width=2)),
            text=[fmt_pct(v) for v in lr["lr"]],
            textposition="top center",
            textfont=dict(size=10, color=COLORS["danger"]),
            hovertemplate="%{x}<br>Loss ratio %{y:.1%}<extra></extra>",
        ))
        fig.add_hline(y=target_loss_ratio,
                      line=dict(color=COLORS["success"], dash="dash"),
                      annotation_text=f"Target {fmt_pct(target_loss_ratio)}",
                      annotation_font_color=COLORS["success"])
        fig.update_layout(
            **BASE_LAYOUT, height=360,
            xaxis=dict(title="Financial year", gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
            yaxis=dict(title="Loss ratio", tickformat=".0%",
                       gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
        )
        st.plotly_chart(fig, use_container_width=True)

        if (lr["lr"] > target_loss_ratio).any():
            info_card(
                f"<b>{int((lr['lr'] > target_loss_ratio).sum())} of "
                f"{len(lr)} financial years</b> are above the target loss "
                f"ratio. Pricing or medical management is leaking margin.",
                COLORS["danger"],
            )

    # --- Row B: distribution of claim_total + Lorenz -----------------------
    section_header("Claim-size distribution & cost concentration", 14)
    cdist_l, cdist_r = st.columns([3, 2], gap="large")

    with cdist_l:
        nonzero = dff.loc[dff["claim_total"] > 0, "claim_total"]
        median = nonzero.median()
        p90 = nonzero.quantile(0.90)
        p99 = nonzero.quantile(0.99)

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=np.log10(nonzero.clip(lower=1)),
            nbinsx=60,
            marker_color=COLORS["primary"], opacity=0.9,
        ))
        for label, val, color in [
            ("Median", median, COLORS["success"]),
            ("P90", p90, COLORS["warning"]),
            ("P99", p99, COLORS["danger"]),
        ]:
            fig.add_vline(
                x=np.log10(val),
                line=dict(color=color, width=2, dash="dot"),
                annotation_text=f"{label} {fmt_ksh(val)}",
                annotation_font_color=color,
            )
        fig.update_layout(
            **BASE_LAYOUT, height=320, bargap=0.05,
            xaxis=dict(title="log₁₀(claim payout, KSh)",
                       gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
            yaxis=dict(title="Claims", gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
        )
        st.plotly_chart(fig, use_container_width=True)
        info_card(
            f"Distribution is heavily right-skewed. Median claim is "
            f"<b>{fmt_ksh(median)}</b>, but the 99th percentile reaches "
            f"<b>{fmt_ksh(p99)}</b>. The mean is a weak planning anchor — "
            "use the percentile profile for reinsurance design.",
            COLORS["primary"],
        )

    with cdist_r:
        # Lorenz / Pareto on claim payout
        sc = np.sort(dff["claim_total"].fillna(0).values)
        cum = np.cumsum(sc) / max(sc.sum(), 1)
        share = np.linspace(0, 1, len(cum))
        gini = 1 - 2 * np.trapezoid(cum, share)
        top1 = sc[int(0.99 * len(sc)):].sum() / max(sc.sum(), 1)
        top10 = sc[int(0.90 * len(sc)):].sum() / max(sc.sum(), 1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=share, y=cum, mode="lines",
            line=dict(color=COLORS["danger"], width=3),
            fill="tozeroy", fillcolor="rgba(225,29,72,0.10)",
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color=COLORS["muted"], dash="dash"),
        ))
        fig.update_layout(
            **BASE_LAYOUT, height=320, showlegend=False,
            xaxis=dict(tickformat=".0%", title="Share of claims",
                       gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
            yaxis=dict(tickformat=".0%", title="Share of payout",
                       gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
        )
        st.plotly_chart(fig, use_container_width=True)
        info_card(
            f"<b>Gini = {gini:.2f}.</b> The top 1% of claims drive "
            f"<b>{fmt_pct(top1)}</b>, the top 10% drive "
            f"<b>{fmt_pct(top10)}</b> of payout. Stop-loss / reinsurance "
            "design should target the upper tail, not the average.",
            COLORS["danger"],
        )

    # --- Row C: county map + plan-tier mix ---------------------------------
    section_header("Where the spend lands geographically", 14)
    g_l, g_r = st.columns([3, 2], gap="large")

    with g_l:
        cnty = (
            dff.groupby("county_name", as_index=False)
               .agg(claims=("doc_no", "count"),
                    payout=("claim_total", "sum"),
                    bill=("total_bill", "sum"),
                    avg=("claim_total", "mean"))
               .sort_values("payout", ascending=True)
               .tail(20)
        )
        fig = go.Figure(go.Bar(
            x=cnty["payout"], y=cnty["county_name"], orientation="h",
            marker=dict(
                color=cnty["avg"],
                colorscale=[[0, COLORS["success"]],
                            [0.5, COLORS["warning"]],
                            [1, COLORS["danger"]]],
                colorbar=dict(title="Avg claim",
                              tickfont=dict(size=9))),
            text=[fmt_ksh(v) for v in cnty["payout"]],
            textposition="outside", textfont=dict(size=10),
            hovertemplate="<b>%{y}</b><br>Payout %{x:,.0f}"
                          "<br>Claims %{customdata[0]:,}"
                          "<br>Avg %{customdata[1]:,.0f}<extra></extra>",
            customdata=cnty[["claims", "avg"]].values,
        ))
        fig.update_layout(
            **BASE_LAYOUT, height=420,
            xaxis=dict(title="Total claim payout (KSh)",
                       gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
            yaxis=dict(gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
        )
        st.plotly_chart(fig, use_container_width=True)

    with g_r:
        plan = (
            dff.drop_duplicates("mem_id")
               .groupby("plan_tier", as_index=False)
               .agg(members=("mem_id", "count"),
                    prem=("annual_premium", "sum"),
                    cost=("total_claims_cost", "sum"))
        )
        plan["lr"] = plan["cost"] / plan["prem"]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=plan["plan_tier"], y=plan["members"],
            marker_color=COLORS["primary"], opacity=0.85,
            name="Members",
            text=[fmt_int(v) for v in plan["members"]],
            textposition="outside", textfont=dict(size=10),
            yaxis="y",
        ))
        fig.add_trace(go.Scatter(
            x=plan["plan_tier"], y=plan["lr"],
            mode="lines+markers+text",
            text=[fmt_pct(v) for v in plan["lr"]],
            textposition="top center",
            line=dict(color=COLORS["danger"], width=3),
            marker=dict(size=10, color=COLORS["danger"]),
            yaxis="y2", name="Loss ratio",
        ))
        fig.update_layout(
            **BASE_LAYOUT, height=420,
            xaxis=dict(title="Plan tier", gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
            yaxis=dict(title="Members", gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
            yaxis2=dict(title="Loss ratio", overlaying="y", side="right",
                        tickformat=".0%",
                        gridcolor="rgba(0,0,0,0)",
                        tickfont=dict(size=10, color="#6B8CAE")),
            legend=dict(orientation="h", y=1.10, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        info_card(
            "Loss-ratio differences across plan tiers signal mispricing in "
            "tier construction. Premium plans should not subsidise basic "
            "tiers.",
            COLORS["primary"],
        )

    # --- Row D: top diseases by spend --------------------------------------
    section_header("Top disease drivers", 14)
    d_l, d_r = st.columns(2, gap="large")

    with d_l:
        dis = (
            dff.groupby("disease_name", as_index=False)
               .agg(claims=("doc_no", "count"),
                    payout=("claim_total", "sum"),
                    avg=("claim_total", "mean"),
                    los=("los", "mean"))
               .sort_values("payout", ascending=False)
               .head(12)
        )
        fig = go.Figure(go.Bar(
            x=dis["disease_name"], y=dis["payout"],
            marker_color=COLORS["danger"], opacity=0.88,
            text=[fmt_ksh(v) for v in dis["payout"]],
            textposition="outside", textfont=dict(size=10),
        ))
        fig.update_layout(
            **BASE_LAYOUT, height=320,
            xaxis=dict(title="Disease", gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
            yaxis=dict(title="Total payout (KSh)", gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
        )
        fig.update_xaxes(tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with d_r:
        cat_split = dff.groupby(
            ["disease_category", "disease_name"], as_index=False
        ).agg(payout=("claim_total", "sum"))
        fig = px.sunburst(
            cat_split, path=["disease_category", "disease_name"],
            values="payout",
            color="payout",
            color_continuous_scale=[
                [0, COLORS["success"]], [0.5, COLORS["warning"]],
                [1, COLORS["danger"]]],
        )
        fig.update_traces(textfont=dict(family="Montserrat", size=11))
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0), height=320,
            font=dict(family="Montserrat", color="#003467"),
            coloraxis_colorbar=dict(title="Payout",
                                    tickfont=dict(size=9)),
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MEMBER & RISK DNA
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    narrative(
        "Member & risk DNA",
        "Behind every claim is a person, an employer, a household. "
        "We split the book by behaviour (smoker, exercise, wellness "
        "enrolment), affordability (income, plan tier, family size) and "
        "embedded risk scores. Each lens points at a different intervention "
        "team: medical management, underwriting, retention."
    )

    # --- Row A: risk pyramid + smoker/exercise mix ----------------------------
    rp_l, rp_r = st.columns([2, 3], gap="large")

    with rp_l:
        section_header("Risk pyramid")
        tier_summary = (
            dff.drop_duplicates("mem_id")
               .groupby("risk_tier", as_index=False)
               .agg(members=("mem_id", "count"),
                    cost=("total_claims_cost", "sum"))
               .reindex(columns=["risk_tier", "members", "cost"])
        )
        order = ["Low", "Moderate", "High", "Critical"]
        tier_summary = tier_summary.set_index("risk_tier").reindex(order).fillna(0).reset_index()
        total_mem = tier_summary["members"].sum()
        total_cost = tier_summary["cost"].sum()
        tier_summary["mem_share"] = tier_summary["members"] / max(total_mem, 1)
        tier_summary["cost_share"] = tier_summary["cost"] / max(total_cost, 1)

        tier_colors = {
            "Low": COLORS["success"], "Moderate": COLORS["primary"],
            "High": COLORS["warning"], "Critical": COLORS["danger"],
        }
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=tier_summary["risk_tier"], x=tier_summary["mem_share"],
            orientation="h", name="Members",
            marker_color=[tier_colors[t] for t in tier_summary["risk_tier"]],
            opacity=0.5,
            text=[fmt_int(m) for m in tier_summary["members"]],
            textposition="inside", textfont=dict(color="#fff", size=11),
        ))
        fig.add_trace(go.Bar(
            y=tier_summary["risk_tier"], x=tier_summary["cost_share"],
            orientation="h", name="Cost",
            marker_color=[tier_colors[t] for t in tier_summary["risk_tier"]],
            text=[fmt_pct(s) for s in tier_summary["cost_share"]],
            textposition="outside", textfont=dict(size=10),
        ))
        fig.update_layout(
            **BASE_LAYOUT, height=320, barmode="group",
            xaxis=dict(tickformat=".0%", title="Share of book",
                       gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
            yaxis=dict(categoryorder="array", categoryarray=order,
                       gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
            legend=dict(orientation="h", y=1.12, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with rp_r:
        section_header("Behaviour × cost matrix")
        beh = (
            dff.drop_duplicates("mem_id")
               .groupby(["smoker", "exercise_frequency"], as_index=False)
               .agg(members=("mem_id", "count"),
                    cost=("total_claims_cost", "mean"))
        )
        beh = beh.dropna()
        if not beh.empty:
            piv = beh.pivot(index="smoker",
                            columns="exercise_frequency",
                            values="cost")
            ord_ex = [c for c in ["Low", "Medium", "High"] if c in piv.columns]
            piv = piv[ord_ex] if ord_ex else piv
            members = beh.pivot(index="smoker",
                                columns="exercise_frequency",
                                values="members")[ord_ex] if ord_ex else None
            fig = go.Figure(data=go.Heatmap(
                z=piv.values,
                x=list(piv.columns), y=list(piv.index),
                colorscale=[[0, COLORS["success"]], [0.5, COLORS["warning"]],
                            [1, COLORS["danger"]]],
                hovertemplate="<b>Smoker %{y} · Exercise %{x}</b><br>"
                              "Avg cost KSh %{z:,.0f}<extra></extra>",
                colorbar=dict(title="Avg cost", tickfont=dict(size=9)),
            ))
            # annotate cells
            for i, smk in enumerate(piv.index):
                for j, ex in enumerate(piv.columns):
                    val = piv.values[i, j]
                    if not np.isnan(val):
                        fig.add_annotation(
                            x=ex, y=smk,
                            text=f"<b>{fmt_ksh(val)}</b><br>"
                                 f"<span style='font-size:9px'>"
                                 f"{fmt_int(members.values[i, j]) if members is not None else ''} mbrs"
                                 f"</span>",
                            showarrow=False,
                            font=dict(size=10, color="#fff"),
                        )
            fig.update_layout(
                **BASE_LAYOUT, height=320,
                xaxis=dict(title="Exercise frequency",
                           tickfont=dict(size=10, color="#6B8CAE")),
                yaxis=dict(title="Smoker",
                           tickfont=dict(size=10, color="#6B8CAE")),
            )
            st.plotly_chart(fig, use_container_width=True)
            info_card(
                "Smoking and low exercise compound — high-exercise non-smokers "
                "form the cheapest cohort, low-exercise smokers the most "
                "expensive. The 2×2 of behavioural choice already explains "
                "a meaningful slice of the loss curve.",
                COLORS["primary"],
            )

    # --- Row B: age × claim cost violin + age × disease ----------------------
    ad_l, ad_r = st.columns(2, gap="large")

    with ad_l:
        section_header("Cost by age band")
        sample = dff.dropna(subset=["age", "claim_total"])
        # downsample for performance on violin
        if len(sample) > 25000:
            sample = sample.sample(25000, random_state=7)
        fig = px.violin(
            sample, x="age_band", y="claim_total",
            color="age_band", box=True, points=False,
            category_orders={"age_band": ["<18", "18–24", "25–34", "35–44",
                                          "45–54", "55–64", "65+"]},
            color_discrete_sequence=CAT_PALETTE,
        )
        fig.update_layout(
            **CHART_LAYOUT, height=360, showlegend=False,
            xaxis_title="Age band", yaxis_title="Claim payout (KSh)",
        )
        fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Y-axis is log-scaled — the upper tail balloons at 45+.")

    with ad_r:
        section_header("Disease incidence by age band")
        ad = (
            dff.groupby(["age_band", "disease_name"], as_index=False)
               .agg(claims=("doc_no", "count"))
        )
        order_ab = ["<18", "18–24", "25–34", "35–44", "45–54", "55–64", "65+"]
        # keep only the top diseases overall
        top_dis = (
            dff["disease_name"].value_counts().head(8).index.tolist()
        )
        ad = ad[ad["disease_name"].isin(top_dis)]
        ad["age_band"] = pd.Categorical(ad["age_band"],
                                        categories=order_ab, ordered=True)
        ad = ad.sort_values("age_band")
        fig = px.bar(
            ad, x="age_band", y="claims", color="disease_name",
            barmode="stack",
            color_discrete_sequence=CAT_PALETTE,
            category_orders={"age_band": order_ab},
        )
        fig.update_layout(
            **CHART_LAYOUT, height=360,
            xaxis_title="Age band", yaxis_title="Claims",
            legend=dict(orientation="h", y=-0.15, x=0,
                        font=dict(size=10)),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Row C: income × employment heatmap + plan switch ---------------------
    section_header("Affordability: income × employment × plan", 14)
    af_l, af_r = st.columns(2, gap="large")

    with af_l:
        m = (
            dff.drop_duplicates("mem_id")
               .groupby(["income_band", "employment_type"], as_index=False)
               .agg(lr=("loss_ratio", "mean"),
                    members=("mem_id", "count"))
        )
        ord_inc = ["Low", "Middle", "High"]
        piv = m.pivot(index="income_band", columns="employment_type",
                      values="lr").reindex(ord_inc)
        cnt = m.pivot(index="income_band", columns="employment_type",
                      values="members").reindex(ord_inc)
        fig = go.Figure(data=go.Heatmap(
            z=piv.values, x=list(piv.columns), y=list(piv.index),
            colorscale=[[0, COLORS["success"]], [0.5, COLORS["warning"]],
                        [1, COLORS["danger"]]],
            colorbar=dict(title="Loss ratio",
                          tickformat=".0%",
                          tickfont=dict(size=9)),
            hovertemplate="<b>Income %{y} · %{x}</b><br>"
                          "Loss ratio %{z:.1%}<extra></extra>",
        ))
        for i, inc in enumerate(piv.index):
            for j, emp in enumerate(piv.columns):
                v = piv.values[i, j]
                if not np.isnan(v):
                    fig.add_annotation(
                        x=emp, y=inc,
                        text=f"<b>{fmt_pct(v)}</b><br>"
                             f"<span style='font-size:9px'>"
                             f"{fmt_int(cnt.values[i, j])} mbrs</span>",
                        showarrow=False,
                        font=dict(size=10, color="#fff"),
                    )
        fig.update_layout(**CHART_LAYOUT, height=320,
                          xaxis_title="Employment type",
                          yaxis_title="Income band")
        st.plotly_chart(fig, use_container_width=True)

    with af_r:
        # Plan tier × family size: average premium and average claims
        fs = (
            dff.drop_duplicates("mem_id")
               .groupby(["plan_tier", "family_size"], as_index=False)
               .agg(prem=("annual_premium", "mean"),
                    cost=("total_claims_cost", "mean"))
        )
        fs["lr"] = fs["cost"] / fs["prem"]
        fig = px.scatter(
            fs, x="family_size", y="prem", size="cost", color="plan_tier",
            color_discrete_sequence=CAT_PALETTE,
            hover_data={"family_size": True, "prem": ":,.0f",
                        "cost": ":,.0f", "lr": ":.1%"},
        )
        fig.update_layout(
            **CHART_LAYOUT, height=320,
            xaxis_title="Family size",
            yaxis_title="Avg annual premium (KSh)",
            legend=dict(orientation="h", y=1.10, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Row D: wellness program impact + telemed adoption -------------------
    we_l, we_r = st.columns(2, gap="large")

    with we_l:
        section_header("Wellness program impact")
        m = (
            dff.drop_duplicates("mem_id")
               .groupby("wellness_program", as_index=False)
               .agg(members=("mem_id", "count"),
                    avg_cost=("total_claims_cost", "mean"),
                    avg_prem=("annual_premium", "mean"))
        )
        m["lr"] = m["avg_cost"] / m["avg_prem"]
        if "Yes" in m["wellness_program"].values and "No" in m["wellness_program"].values:
            yes_lr = m.loc[m["wellness_program"] == "Yes", "lr"].iloc[0]
            no_lr = m.loc[m["wellness_program"] == "No", "lr"].iloc[0]
            yes_cost = m.loc[m["wellness_program"] == "Yes", "avg_cost"].iloc[0]
            no_cost = m.loc[m["wellness_program"] == "No", "avg_cost"].iloc[0]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["No wellness", "On wellness"],
                y=[no_cost, yes_cost],
                marker_color=[COLORS["danger"], COLORS["success"]],
                text=[fmt_ksh(no_cost), fmt_ksh(yes_cost)],
                textposition="outside", textfont=dict(size=11),
            ))
            fig.update_layout(
                **CHART_LAYOUT, height=300,
                yaxis_title="Avg claims cost (KSh)",
            )
            st.plotly_chart(fig, use_container_width=True)
            delta_cost = no_cost - yes_cost
            info_card(
                f"Members on the wellness program post an average claims "
                f"cost of <b>{fmt_ksh(yes_cost)}</b> versus "
                f"<b>{fmt_ksh(no_cost)}</b> for non-enrolees — a delta of "
                f"<b>{fmt_ksh(delta_cost)}</b> per member ({fmt_pct(delta_cost/max(no_cost,1))} "
                "lower). Loss-ratio differential: "
                f"<b>{fmt_pct(no_lr - yes_lr)}</b>.",
                COLORS["success"] if delta_cost > 0 else COLORS["warning"],
            )

    with we_r:
        section_header("Channel adoption (telemedicine, M-Pesa, NHIF)")
        dedup = dff.drop_duplicates("mem_id")
        chan = pd.DataFrame({
            "channel": ["Telemedicine", "M-Pesa", "NHIF"],
            "adoption": [
                dedup["telemedicine_usage_b"].mean(),
                dedup["mpesa_usage_b"].mean(),
                dedup["nhif_usage_b"].mean(),
            ],
        })
        fig = go.Figure(go.Bar(
            x=chan["channel"], y=chan["adoption"],
            marker_color=[COLORS["primary"], COLORS["success"], COLORS["purple"]],
            text=[fmt_pct(v) for v in chan["adoption"]],
            textposition="outside", textfont=dict(size=11),
        ))
        fig.update_layout(
            **BASE_LAYOUT, height=300,
            xaxis=dict(title="Channel", gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
            yaxis=dict(title="Adoption", tickformat=".0%",
                       gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
        )
        st.plotly_chart(fig, use_container_width=True)
        info_card(
            "Channel adoption is a leading indicator for cost trajectory: "
            "telemedicine reduces unnecessary admissions, M-Pesa shortens "
            "claim cycle time, NHIF crowding-in cuts the insurer's net "
            "exposure.",
            COLORS["primary"],
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PROVIDER & DISEASE INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    narrative(
        "Provider & disease intelligence",
        "Where the money goes inside the medical economy. We rank "
        "hospitals by spend, expose Government / Private / Mission cost "
        "differentials, surface the ICD-10 chapters that move the needle, "
        "and highlight providers with anomalous payout ratios — the "
        "shortlist for negotiation and audit."
    )

    # --- Row A: top hospitals + category mix ---------------------------------
    h_l, h_r = st.columns([3, 2], gap="large")

    with h_l:
        section_header("Top hospitals by total payout")
        hosp = (
            dff.groupby("hosp_name", as_index=False)
               .agg(claims=("doc_no", "count"),
                    payout=("claim_total", "sum"),
                    bill=("total_bill", "sum"),
                    avg=("claim_total", "mean"),
                    los=("los", "mean"))
               .sort_values("payout", ascending=False)
               .head(15)
        )
        hosp["pay_ratio"] = hosp["payout"] / hosp["bill"].replace(0, np.nan)
        fig = go.Figure(go.Bar(
            x=hosp["payout"], y=hosp["hosp_name"], orientation="h",
            marker=dict(
                color=hosp["pay_ratio"],
                colorscale=[[0, COLORS["success"]], [1, COLORS["danger"]]],
                colorbar=dict(title="Pay/Bill",
                              tickformat=".0%",
                              tickfont=dict(size=9))),
            text=[fmt_ksh(v) for v in hosp["payout"]],
            textposition="outside", textfont=dict(size=10),
            hovertemplate="<b>%{y}</b><br>Payout %{x:,.0f}"
                          "<br>Claims %{customdata[0]:,}"
                          "<br>Avg LOS %{customdata[1]:.1f} d"
                          "<extra></extra>",
            customdata=hosp[["claims", "los"]].values,
        ))
        fig.update_layout(
            **BASE_LAYOUT, height=520,
            xaxis=dict(title="Total payout (KSh)", gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
            yaxis=dict(autorange="reversed", gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
        )
        st.plotly_chart(fig, use_container_width=True)

    with h_r:
        section_header("Hospital category economics")
        cat = (
            dff.groupby("hosp_cat_label", as_index=False)
               .agg(claims=("doc_no", "count"),
                    payout=("claim_total", "sum"),
                    avg_bill=("total_bill", "mean"),
                    avg_pay=("claim_total", "mean"),
                    avg_los=("los", "mean"))
        )
        cat["pay_ratio"] = cat["avg_pay"] / cat["avg_bill"].replace(0, np.nan)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cat["hosp_cat_label"], y=cat["avg_pay"],
            marker_color=COLORS["primary"], opacity=0.85,
            text=[fmt_ksh(v) for v in cat["avg_pay"]],
            textposition="outside",
            name="Avg payout",
        ))
        fig.update_layout(
            **CHART_LAYOUT, height=260,
            yaxis_title="Avg payout (KSh)",
            xaxis_title="Hospital category",
        )
        st.plotly_chart(fig, use_container_width=True)
        cat_show = cat.copy()
        cat_show["claims"] = cat_show["claims"].apply(fmt_int)
        cat_show["payout"] = cat_show["payout"].apply(fmt_ksh)
        cat_show["avg_bill"] = cat_show["avg_bill"].apply(fmt_ksh)
        cat_show["avg_pay"] = cat_show["avg_pay"].apply(fmt_ksh)
        cat_show["avg_los"] = cat_show["avg_los"].round(2)
        cat_show["pay_ratio"] = cat_show["pay_ratio"].apply(fmt_pct)
        cat_show.columns = ["Category", "Claims", "Payout", "Avg bill",
                            "Avg payout", "Avg LOS (d)", "Pay/Bill"]
        st.dataframe(cat_show, hide_index=True, use_container_width=True)

    # --- Row B: ICD10 chapter analysis + payout-ratio outliers ---------------
    section_header("ICD-10 chapter cost analysis", 14)
    ic_l, ic_r = st.columns([3, 2], gap="large")

    with ic_l:
        chap = (
            dff.dropna(subset=["icd10_chap"])
               .groupby("icd10_chap", as_index=False)
               .agg(claims=("doc_no", "count"),
                    payout=("claim_total", "sum"),
                    avg_los=("los", "mean"))
               .sort_values("payout", ascending=False)
        )
        fig = go.Figure(go.Bar(
            x=chap["icd10_chap"].astype(int).astype(str),
            y=chap["payout"],
            marker=dict(
                color=chap["avg_los"],
                colorscale=[[0, COLORS["success"]], [1, COLORS["danger"]]],
                colorbar=dict(title="Avg LOS",
                              tickfont=dict(size=9))),
            text=[fmt_ksh(v) for v in chap["payout"]],
            textposition="outside", textfont=dict(size=9),
        ))
        fig.update_layout(
            **CHART_LAYOUT, height=360,
            xaxis_title="ICD-10 chapter",
            yaxis_title="Total payout (KSh)",
        )
        st.plotly_chart(fig, use_container_width=True)
        info_card(
            "Chapters with high LOS and low claim volume signal complex care "
            "(oncology, mental health, neurology) — these are the cohorts "
            "where case management and second-opinion programs pay back "
            "fastest.",
            COLORS["primary"],
        )

    with ic_r:
        section_header("Payout-ratio outliers")
        # Hospitals with anomalous payout/bill ratios — too low (claims rejected,
        # member dissatisfaction) or too high (provider over-reimbursement).
        outl = (
            dff.groupby("hosp_name", as_index=False)
               .agg(claims=("doc_no", "count"),
                    bill=("total_bill", "sum"),
                    pay=("claim_total", "sum"))
        )
        outl = outl[outl["claims"] >= 50].copy()
        outl["pay_ratio"] = outl["pay"] / outl["bill"].replace(0, np.nan)
        outl = outl.dropna(subset=["pay_ratio"])
        outl["abs_dev"] = (outl["pay_ratio"] - outl["pay_ratio"].median()).abs()
        outl = outl.sort_values("abs_dev", ascending=False).head(12)
        outl_show = outl[["hosp_name", "claims", "pay_ratio"]].copy()
        outl_show["claims"] = outl_show["claims"].apply(fmt_int)
        outl_show["pay_ratio"] = outl_show["pay_ratio"].apply(fmt_pct)
        outl_show.columns = ["Hospital", "Claims", "Pay / Bill"]
        st.dataframe(outl_show, hide_index=True, use_container_width=True)
        info_card(
            "Hospitals furthest from the network-median payout ratio. "
            "Low ratios → audit member experience; high ratios → "
            "renegotiate the case-rate schedule.",
            COLORS["warning"],
        )

    # --- Row C: length of stay distribution ----------------------------------
    section_header("Length-of-stay & disease severity", 14)
    los_l, los_r = st.columns([3, 2], gap="large")

    with los_l:
        los_data = dff[dff["los"].between(0, 30)]
        if len(los_data) > 30000:
            los_data = los_data.sample(30000, random_state=7)
        fig = px.box(
            los_data, x="disease_category", y="los",
            color="hosp_cat_label",
            color_discrete_sequence=CAT_PALETTE,
            points=False,
        )
        fig.update_layout(
            **CHART_LAYOUT, height=320,
            xaxis_title="Disease category", yaxis_title="LOS (days)",
            legend=dict(orientation="h", y=1.10, x=0,
                        title=dict(text="")),
        )
        st.plotly_chart(fig, use_container_width=True)

    with los_r:
        # Top diseases by avg cost AND prevalence — bubble chart
        bd = (
            dff.groupby("disease_name", as_index=False)
               .agg(claims=("doc_no", "count"),
                    avg=("claim_total", "mean"),
                    payout=("claim_total", "sum"))
               .sort_values("payout", ascending=False)
               .head(15)
        )
        fig = px.scatter(
            bd, x="claims", y="avg", size="payout", color="disease_name",
            color_discrete_sequence=CAT_PALETTE,
            hover_data={"payout": ":,.0f", "claims": ":,",
                        "avg": ":,.0f"},
        )
        fig.update_layout(
            **CHART_LAYOUT, height=320,
            xaxis_title="Claim volume",
            yaxis_title="Avg payout (KSh)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Bubble size = total payout. The upper-right quadrant "
                   "is where case management dollars compound fastest.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICTIVE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    narrative(
        "Predictive engine",
        "A gradient-boosted model trained on the in-scope book to forecast "
        "claim payout from the structural rating factors. The same engine "
        "drives the underwriting what-if tool below — the loss-cost "
        "expectation pricing should respect or override consciously."
    )

    section_header("Build & evaluate model")

    # Sample for training speed — 30k rows is plenty for a tree ensemble.
    train_df = dff.dropna(subset=["claim_total", "age", "los"])
    if len(train_df) > 40000:
        train_df = train_df.sample(40000, random_state=7)

    feat_cols = ["age", "los", "family_size", "deductible_pct",
                 "coverage_limit_kes", "county_cost_index"]
    feat_cols = [c for c in feat_cols if c in train_df.columns]

    X = train_df[feat_cols].fillna(train_df[feat_cols].median()).copy()
    X["smoker_yes"] = train_df["smoker_b"].astype(int)
    X["wellness"] = train_df["wellness_program_b"].astype(int)
    X["telemed"] = train_df["telemedicine_usage_b"].astype(int)
    X["sex_f"] = (train_df["p_sex"] == "F").astype(int)
    for c in ["plan_tier", "income_band", "employment_type",
              "hosp_cat_label", "disease_category"]:
        if c in train_df.columns:
            X = pd.concat(
                [X, pd.get_dummies(train_df[c], prefix=c[:6])], axis=1
            )

    y = train_df["claim_total"].values

    Xtr, Xte, ytr, yte = train_test_split(
        X.values.astype(float), y, test_size=0.25, random_state=7
    )
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=4,
        learning_rate=0.05, random_state=7,
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    r2 = r2_score(yte, pred)
    mae = mean_absolute_error(yte, pred)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Model R²", f"{r2*100:.1f}%",
                 "variance of payout explained", COLORS["primary"])
    with k2:
        kpi_card("Mean abs. error", fmt_ksh(mae),
                 "average miss per claim", COLORS["warning"])
    with k3:
        kpi_card("Training rows", fmt_int(len(train_df)),
                 "sampled from in-scope book", COLORS["navy"])
    with k4:
        kpi_card("Features", str(X.shape[1]),
                 "structural + behavioural", COLORS["success"])

    st.markdown("<div style='margin-bottom:14px'></div>", unsafe_allow_html=True)

    # --- Feature importance + predicted-vs-actual ----------------------------
    p1, p2 = st.columns([2, 3], gap="large")

    with p1:
        section_header("Feature importance")
        fi = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_,
        }).sort_values("importance").tail(15)
        fig = go.Figure(go.Bar(
            x=fi["importance"], y=fi["feature"], orientation="h",
            marker_color=COLORS["primary"], opacity=0.9,
            text=[f"{v*100:.1f}%" for v in fi["importance"]],
            textposition="outside", textfont=dict(size=10),
        ))
        fig.update_layout(
            **BASE_LAYOUT, height=460,
            xaxis=dict(tickformat=".0%", title="Relative importance",
                       gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
            yaxis=dict(gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
        )
        st.plotly_chart(fig, use_container_width=True)

    with p2:
        section_header("Predicted vs actual (held-out)")
        fig = go.Figure()
        # log-log scatter to handle the heavy tail
        m_pos = (yte > 0) & (pred > 0)
        fig.add_trace(go.Scattergl(
            x=yte[m_pos], y=pred[m_pos], mode="markers",
            marker=dict(size=4, color=COLORS["primary"], opacity=0.45,
                        line=dict(width=0)),
            hovertemplate="Actual %{x:,.0f}<br>"
                          "Predicted %{y:,.0f}<extra></extra>",
        ))
        lo = float(min(yte[m_pos].min(), pred[m_pos].min(), 1))
        hi = float(max(yte[m_pos].max(), pred[m_pos].max()))
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(color=COLORS["danger"], dash="dash"),
        ))
        fig.update_layout(
            **BASE_LAYOUT, height=460, showlegend=False,
            xaxis=dict(title="Actual (KSh)", type="log",
                       gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
            yaxis=dict(title="Predicted (KSh)", type="log",
                       gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Fraud probability deep-dive -----------------------------------------
    section_header("Fraud-probability landscape", 14)
    fr_l, fr_r = st.columns([2, 3], gap="large")

    with fr_l:
        ft = (
            dff.groupby("fraud_tier", as_index=False)
               .agg(claims=("doc_no", "count"),
                    payout=("claim_total", "sum"))
               .reindex(columns=["fraud_tier", "claims", "payout"])
        )
        order_ft = ["Clean", "Watch", "Investigate", "High alert"]
        ft = ft.set_index("fraud_tier").reindex(order_ft).fillna(0).reset_index()
        cmap = {"Clean": COLORS["success"], "Watch": COLORS["primary"],
                "Investigate": COLORS["warning"], "High alert": COLORS["danger"]}
        fig = go.Figure(go.Bar(
            x=ft["fraud_tier"], y=ft["payout"],
            marker_color=[cmap[t] for t in ft["fraud_tier"]],
            text=[fmt_ksh(v) for v in ft["payout"]],
            textposition="outside",
            customdata=ft[["claims"]].values,
            hovertemplate="<b>%{x}</b><br>"
                          "Payout %{y:,.0f}<br>"
                          "Claims %{customdata[0]:,}<extra></extra>",
        ))
        fig.update_layout(
            **CHART_LAYOUT, height=320,
            xaxis_title="Fraud tier",
            yaxis_title="Total payout (KSh)",
        )
        st.plotly_chart(fig, use_container_width=True)
        h_a = ft.loc[ft["fraud_tier"] == "High alert", "payout"].sum()
        info_card(
            f"<b>High-alert payouts in scope: {fmt_ksh(h_a)}</b>. "
            f"Even at a {fmt_pct(fraud_recovery)} recovery rate the SIU "
            f"could claw back <b>{fmt_ksh(h_a * fraud_recovery)}</b>.",
            COLORS["danger"],
        )

    with fr_r:
        # Fraud probability distribution + threshold
        threshold = st.slider("Fraud-flag threshold",
                              0.05, 0.95, 0.50, 0.05)
        flagged = (dff["fraud_probability"] >= threshold).sum()
        flagged_payout = dff.loc[
            dff["fraud_probability"] >= threshold, "claim_total"].sum()
        sample = dff.sample(min(20000, len(dff)), random_state=7)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=sample["fraud_probability"], nbinsx=40,
            marker_color=COLORS["primary"], opacity=0.85,
        ))
        fig.add_vline(x=threshold,
                      line=dict(color=COLORS["danger"], width=3, dash="dash"),
                      annotation_text=f"Threshold {threshold:.2f}",
                      annotation_font_color=COLORS["danger"])
        fig.update_layout(
            **CHART_LAYOUT, height=300,
            xaxis_title="Fraud probability",
            yaxis_title="Claims",
        )
        st.plotly_chart(fig, use_container_width=True)
        cf1, cf2, cf3 = st.columns(3)
        with cf1:
            kpi_card("Flagged claims", fmt_int(flagged),
                     f"of {fmt_int(len(dff))}", COLORS["warning"])
        with cf2:
            kpi_card("Flagged payout", fmt_ksh(flagged_payout),
                     "subject to SIU review", COLORS["danger"])
        with cf3:
            kpi_card("Recoverable @ rate",
                     fmt_ksh(flagged_payout * fraud_recovery),
                     f"@ {fmt_pct(fraud_recovery)} recovery",
                     COLORS["success"])

    # --- Underwriting what-if simulator --------------------------------------
    section_header("Underwriting what-if simulator", 14)
    info_card(
        "Set a hypothetical new member profile and read the loss-cost "
        "expectation. Use this to interrogate underwriting decisions, "
        "stress-test a quoted premium, or build counterfactuals.",
        COLORS["primary"],
    )

    sim_a, sim_b, sim_c, sim_d = st.columns(4)
    with sim_a:
        _age_min = dff["age"].dropna().min()
        _age_max = dff["age"].dropna().max()
        sim_age = st.slider(
            "Member age",
            int(_age_min) if pd.notna(_age_min) else 18,
            int(_age_max) if pd.notna(_age_max) else 80,
            40,
        )
        sim_los = st.slider("Length of stay (days)", 0, 30, 2)
    with sim_b:
        sim_fam = st.slider("Family size", 1, 6, 3)
        sim_smoker = st.radio("Smoker", ["No", "Yes"], horizontal=True)
    with sim_c:
        sim_well = st.radio("Wellness program",
                            ["No", "Yes"], horizontal=True)
        sim_tel = st.radio("Telemedicine user",
                           ["No", "Yes"], horizontal=True)
    with sim_d:
        sim_plan = st.selectbox("Plan tier",
                                sorted(dff["plan_tier"].dropna().unique()))
        sim_quote = st.number_input("Quoted annual premium (KSh)",
                                    0, 1_000_000, 35000, 500)

    sim_row = pd.DataFrame([{
        "age": sim_age, "los": sim_los,
        "family_size": sim_fam,
        "deductible_pct": train_df["deductible_pct"].median(),
        "coverage_limit_kes": train_df["coverage_limit_kes"].median(),
        "county_cost_index": train_df["county_cost_index"].median(),
        "smoker_yes": 1 if sim_smoker == "Yes" else 0,
        "wellness": 1 if sim_well == "Yes" else 0,
        "telemed": 1 if sim_tel == "Yes" else 0,
        "sex_f": 0,
    }])
    sim_row = sim_row.reindex(columns=X.columns, fill_value=0)
    plan_col = f"plan_t_{sim_plan}"
    if plan_col in sim_row.columns:
        sim_row[plan_col] = 1
    sim_pred = float(model.predict(sim_row.values.astype(float))[0])

    train_resid = ytr - model.predict(Xtr)
    lo_band = sim_pred + np.quantile(train_resid, 0.10)
    hi_band = sim_pred + np.quantile(train_resid, 0.90)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        kpi_card("Expected claim", fmt_ksh(sim_pred),
                 "model best estimate", COLORS["primary"])
    with s2:
        kpi_card("80% range",
                 f"{fmt_ksh(lo_band)} – {fmt_ksh(hi_band)}",
                 "model uncertainty", COLORS["warning"])
    with s3:
        implied_lr = sim_pred / max(sim_quote, 1)
        colour = (COLORS["danger"] if implied_lr > target_loss_ratio + 0.05
                  else COLORS["success"] if implied_lr < target_loss_ratio - 0.05
                  else COLORS["primary"])
        kpi_card("Implied loss ratio", fmt_pct(implied_lr),
                 f"target {fmt_pct(target_loss_ratio)}", colour)
    with s4:
        verdict = ("Decline / re-rate" if implied_lr > target_loss_ratio + 0.05
                   else "On target" if implied_lr < target_loss_ratio - 0.05
                   else "At threshold")
        verdict_color = (COLORS["danger"] if "Decline" in verdict
                         else COLORS["success"] if "target" in verdict
                         else COLORS["warning"])
        kpi_card("Underwriting verdict", verdict,
                 "vs target loss ratio", verdict_color)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TRANSFORMATIVE STRATEGY
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    narrative(
        "Transformative strategy",
        "Numbers without a verb are noise. Five investable moves, each "
        "tied to a slider in the sidebar — wellness coverage, smoker "
        "cessation, fraud recovery, telemedicine penetration and county "
        "cost arbitrage. Move the levers to negotiate the strategy live."
    )

    dedup = dff.drop_duplicates("mem_id")
    n_smokers = dedup["smoker_b"].sum()
    avg_smoker_cost = dedup.loc[dedup["smoker_b"], "total_claims_cost"].mean()
    avg_nonsmoker_cost = dedup.loc[~dedup["smoker_b"], "total_claims_cost"].mean()

    avg_well_cost = dedup.loc[dedup["wellness_program_b"], "total_claims_cost"].mean()
    avg_no_well_cost = dedup.loc[~dedup["wellness_program_b"], "total_claims_cost"].mean()
    n_no_well = (~dedup["wellness_program_b"]).sum()

    fraud_pay = dff.loc[dff["fraud_probability"] >= 0.50, "claim_total"].sum()

    # --- Lever 1: smoker cessation -------------------------------------------
    section_header("Lever 1 · Smoker cessation")
    if n_smokers and not np.isnan(avg_smoker_cost) and not np.isnan(avg_nonsmoker_cost):
        per_member = avg_smoker_cost - avg_nonsmoker_cost
        quitters = int(round(n_smokers * smoker_quit_rate))
        annual_save = quitters * per_member

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Smokers in book", fmt_int(n_smokers),
                     "behavioural target population", COLORS["primary"])
        with c2:
            kpi_card("Cost gap per smoker",
                     fmt_ksh(per_member),
                     "vs non-smoker avg", COLORS["warning"])
        with c3:
            kpi_card("Quitters modelled", fmt_int(quitters),
                     f"@ {fmt_pct(smoker_quit_rate)} quit rate",
                     COLORS["primary"])
        with c4:
            kpi_card("Avoided medical cost",
                     fmt_ksh(annual_save),
                     "annual", COLORS["success"])

        # Sensitivity curve
        rates = np.arange(0.0, 1.01, 0.05)
        saves = [int(round(n_smokers * r)) * per_member for r in rates]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rates, y=saves, mode="lines",
            line=dict(color=COLORS["success"], width=3),
            fill="tozeroy", fillcolor="rgba(11,185,159,0.10)",
        ))
        fig.add_trace(go.Scatter(
            x=[smoker_quit_rate], y=[annual_save], mode="markers+text",
            marker=dict(size=14, color=COLORS["danger"],
                        line=dict(color="#fff", width=2)),
            text=["Current"], textposition="top center",
            textfont=dict(size=10, color=COLORS["danger"]),
        ))
        fig.update_layout(
            **BASE_LAYOUT, height=260, showlegend=False,
            xaxis=dict(tickformat=".0%", title="Smoker quit rate",
                       gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
            yaxis=dict(title="Annual cost avoided (KSh)",
                       gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need both smoker and non-smoker members in scope.")

    # --- Lever 2: wellness program enrolment ---------------------------------
    section_header("Lever 2 · Wellness program enrolment expansion", 14)
    if not np.isnan(avg_well_cost) and not np.isnan(avg_no_well_cost):
        delta = avg_no_well_cost - avg_well_cost
        new_enrollees = int(n_no_well * (wellness_uplift / max(0.01, wellness_uplift)))  # placeholder
        # Use a direct interpretation: wellness_uplift is % cost reduction we can drive
        # by enrolling currently-non-enrolled members.
        target_enrol = int(n_no_well * 0.5)  # half of non-enrolees
        save_per = avg_no_well_cost * wellness_uplift
        cohort_save = target_enrol * save_per

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Currently not enrolled",
                     fmt_int(n_no_well),
                     "addressable cohort", COLORS["warning"])
        with c2:
            kpi_card("Observed cost gap",
                     fmt_ksh(delta),
                     "non-enrolled vs enrolled (avg)",
                     COLORS["primary"])
        with c3:
            kpi_card("Per-member save",
                     fmt_ksh(save_per),
                     f"@ {fmt_pct(wellness_uplift)} reduction",
                     COLORS["success"])
        with c4:
            kpi_card("If 50% enrol →",
                     fmt_ksh(cohort_save),
                     "annual cost avoided",
                     COLORS["success"])

    # --- Lever 3: fraud recovery ---------------------------------------------
    section_header("Lever 3 · Fraud recovery (SIU programme)", 14)
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        kpi_card("Flagged payout (≥0.50)",
                 fmt_ksh(fraud_pay),
                 "high-alert claims",
                 COLORS["danger"])
    with fc2:
        kpi_card("Recovery rate target",
                 fmt_pct(fraud_recovery),
                 "SIU operational lever",
                 COLORS["primary"])
    with fc3:
        kpi_card("Recoverable",
                 fmt_ksh(fraud_pay * fraud_recovery),
                 "annual claw-back",
                 COLORS["success"])

    # Visualise fraud opportunity by hospital
    fr_h = (
        dff[dff["fraud_probability"] >= 0.50]
           .groupby("hosp_name", as_index=False)
           .agg(claims=("doc_no", "count"),
                payout=("claim_total", "sum"))
           .sort_values("payout", ascending=False)
           .head(10)
    )
    if not fr_h.empty:
        fig = go.Figure(go.Bar(
            x=fr_h["payout"], y=fr_h["hosp_name"], orientation="h",
            marker_color=COLORS["danger"], opacity=0.85,
            text=[fmt_ksh(v) for v in fr_h["payout"]],
            textposition="outside", textfont=dict(size=10),
            customdata=fr_h[["claims"]].values,
            hovertemplate="<b>%{y}</b><br>Payout %{x:,.0f}"
                          "<br>Claims %{customdata[0]:,}<extra></extra>",
        ))
        fig.update_layout(
            **BASE_LAYOUT, height=360,
            xaxis=dict(title="High-alert payout (KSh)",
                       gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
            yaxis=dict(autorange="reversed", gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
        )
        st.plotly_chart(fig, use_container_width=True)
        info_card(
            "Hospitals carrying the highest absolute volume of high-alert "
            "(fraud-probability ≥ 0.50) payouts. Direct your SIU and "
            "audit teams here first.",
            COLORS["danger"],
        )

    # --- Lever 4: county cost arbitrage --------------------------------------
    section_header("Lever 4 · County cost arbitrage", 14)
    cnty = (
        dff.groupby("county_name", as_index=False)
           .agg(claims=("doc_no", "count"),
                payout=("claim_total", "sum"),
                avg=("claim_total", "mean"),
                idx=("county_cost_index", "mean"))
    )
    cnty = cnty[cnty["claims"] >= 30]
    fig = px.scatter(
        cnty, x="idx", y="avg", size="claims",
        hover_name="county_name",
        color="avg",
        color_continuous_scale=[
            [0, COLORS["success"]], [0.5, COLORS["warning"]],
            [1, COLORS["danger"]]],
        hover_data={"claims": ":,", "payout": ":,.0f"},
    )
    fig.add_vline(x=cnty["idx"].median(),
                  line=dict(color=COLORS["muted"], dash="dash"))
    fig.add_hline(y=cnty["avg"].median(),
                  line=dict(color=COLORS["muted"], dash="dash"))
    fig.update_layout(
        **CHART_LAYOUT, height=360,
        xaxis_title="County cost index",
        yaxis_title="Avg claim payout (KSh)",
    )
    st.plotly_chart(fig, use_container_width=True)
    high_idx_high_avg = cnty[(cnty["idx"] > cnty["idx"].median())
                             & (cnty["avg"] > cnty["avg"].median())]
    info_card(
        f"<b>{len(high_idx_high_avg)} counties</b> show both above-median "
        "cost index and above-median avg claim — the upper-right quadrant "
        "is the network re-design priority list (negotiate, expand "
        "telemed, NHIF crowd-in).",
        COLORS["danger"],
    )

    # --- Lever 5: priority intervention list ---------------------------------
    section_header("Lever 5 · Top intervention candidates (members)", 14)
    info_card(
        "Members ranked by composite priority = 0.5·risk_score + "
        "0.3·churn_probability + 0.2·fraud_probability_avg, restricted to "
        "members with above-median cost. These are the first calls case "
        "management should make tomorrow morning.",
        COLORS["danger"],
    )
    cand = (
        dff.groupby("mem_id", as_index=False)
           .agg(age=("age", "first"),
                sex=("p_sex", "first"),
                county=("county_name", "first"),
                plan=("plan_tier", "first"),
                claims=("doc_no", "count"),
                payout=("claim_total", "sum"),
                risk=("risk_score", "mean"),
                churn=("churn_probability", "mean"),
                fraud=("fraud_probability", "mean"),
                smoker=("smoker", "first"),
                wellness=("wellness_program", "first"))
    )
    cand = cand[cand["payout"] > cand["payout"].median()]
    cand["priority"] = (
        0.5 * cand["risk"] + 0.3 * cand["churn"] + 0.2 * cand["fraud"]
    )
    cand = cand.sort_values("priority", ascending=False).head(25)
    show = cand.copy()
    show["age"] = show["age"].round(0)
    show["payout"] = show["payout"].apply(fmt_ksh)
    show["risk"] = show["risk"].apply(fmt_pct)
    show["churn"] = show["churn"].apply(fmt_pct)
    show["fraud"] = show["fraud"].apply(fmt_pct)
    show["priority"] = show["priority"].round(3)
    show.columns = ["Member ID", "Age", "Sex", "County", "Plan",
                    "Claims", "Payout", "Risk", "Churn", "Fraud",
                    "Smoker", "Wellness", "Priority"]
    st.dataframe(show, hide_index=True, use_container_width=True)

    # --- Final boardroom narrative -------------------------------------------
    section_header("Boardroom one-page", 14)
    def _safe(v, default=0.0):
        try:
            return default if (v is None or np.isnan(v)) else v
        except Exception:
            return default

    smoker_save = (
        _safe(avg_smoker_cost) - _safe(avg_nonsmoker_cost)
    ) * int(round(_safe(n_smokers) * smoker_quit_rate))
    wellness_save = int(_safe(n_no_well) * 0.5) * _safe(avg_no_well_cost) * wellness_uplift
    fraud_save = _safe(fraud_pay) * fraud_recovery
    total_lever = smoker_save + wellness_save + fraud_save

    st.markdown(
        f"<div style='font-size:13px;color:#003467;line-height:1.7'>"
        f"The book in scope holds <b>{fmt_int(n_members)}</b> members and "
        f"<b>{fmt_int(n_rows)}</b> paid claims totalling "
        f"<b>{fmt_ksh(total_claim)}</b> in payouts on "
        f"<b>{fmt_ksh(total_bill)}</b> of billed care. "
        f"Portfolio loss ratio currently sits at "
        f"<b>{fmt_pct(overall_lr)}</b> against the <b>{fmt_pct(target_loss_ratio)}</b> "
        f"target. Three levers, set against the assumptions in the "
        f"sidebar, jointly avert "
        f"<b>{fmt_ksh(total_lever)}</b> of cost annually: smoker cessation "
        f"({fmt_ksh(smoker_save)}), wellness expansion "
        f"({fmt_ksh(wellness_save)}), and fraud recovery "
        f"({fmt_ksh(fraud_save)}). Each is independently financeable; "
        f"taken together they re-frame the loss-ratio conversation by "
        f"<b>{fmt_pct(total_lever / max(total_premium, 1))}</b>."
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
    st.markdown(
        f'<span class="chip chip-elite">Descriptive · high</span>'
        f'<span class="chip chip-mid">Qualitative · medium</span>'
        f'<span class="chip chip-low">Predictive · medium</span>'
        f'<span class="chip chip-high">Transformative · directional</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Badges reflect epistemic strength: descriptive numbers read "
        "cleanly off the data; transformative numbers depend on the "
        "behavioural assumptions you can move in the sidebar."
    )