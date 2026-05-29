"""
Private Hospitals Executive Dashboard — TENRI + KSH
Run: streamlit run private_analysis/dashboard.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath('__file__')))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

import warnings
warnings.filterwarnings("ignore")

from ksh.facility_utilization.m1_ward_forecast import get_forecast
from ksh.facility_utilization.queries import (
    q_overview_gap, q_overview_alerts,
    q_leakage_gap, q_leakage_submission_rate, q_leakage_ksh_dispatch_trend,
    q_leakage_aging_dist, q_leakage_recovery_priority,
    q_theatre_trend, q_theatre_by_type,
    q_beds_revpab, q_beds_los, q_dialysis_trend, q_specialty_admissions,
    q_imaging_trend,
    q_readmission_pattern, q_readmission_trend,
    q_readmission_exposure, q_readmission_benchmark, q_readmission_ward_trend,
    q_service_mix, q_rebate_by_insurer, q_payer_trend,
)

# ── Feature flags ─────────────────────────────────────────────────────────────
# AR_PAGE_ENABLED: set True once SMART/SLADE payment data is available for
# reconciliation. Disabled 2026-05-26 — FINANCE_INVOICES.paid is zero for all
# insured invoices; payments are recorded externally in SMART/SLADE, not in system.
AR_PAGE_ENABLED = False

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Private Hospitals · Executive Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Montserrat',sans-serif;background:#fff;color:#003467}
.stApp{background:#fff}
[data-testid="stSidebar"]{background:#F4F8FC!important;border-right:1px solid #D6E4F0!important}
[data-testid="stSidebar"] *{font-family:'Montserrat',sans-serif!important}
[data-testid="stSidebar"] .stButton button{
  background:#EBF3FB!important;color:#003467!important;
  border:1px solid #D6E4F0!important;font-size:12px!important}
[data-testid="stSidebar"] .stButton button:hover{background:#D6E4F0!important;color:#003467!important}
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
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

COLORS = {
    "primary": "#0072CE", "success": "#0BB99F", "warning": "#D97706",
    "danger":  "#E11D48", "muted":   "#6B8CAE", "purple":  "#7F77DD",
    "coral":   "#D85A30", "green":   "#1D9E75",
}

FAC_DISPLAY = {"KISUMU_CLEAN": "KSH", "TENRI": "TENRI"}
FAC_OTHER   = {"KISUMU_CLEAN": "TENRI", "TENRI": "KISUMU_CLEAN"}

CHART_LAYOUT = dict(
    paper_bgcolor="#fff", plot_bgcolor="#fff",
    font=dict(family="Montserrat", color="#003467"),
    margin=dict(l=0, r=0, t=10, b=30),
    xaxis=dict(gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")),
    yaxis=dict(gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")),
)

TENRI_DATA_END = "2022-07-27"
KSH_DISPATCH_CLIFF = "2025-09-01"


def cl(**kw):
    return {**CHART_LAYOUT, **kw}


def fmt_kes(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if abs(v) >= 1_000_000:
        return f"KES {v/1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"KES {v/1_000:.0f}K"
    return f"KES {v:.0f}"


def kpi_card(label, value, sub="", color="#003467", icon=""):
    _accent = {COLORS["danger"], COLORS["warning"], COLORS["success"]}
    bl = f"border-left:4px solid {color};" if color in _accent else ""
    icon_html = f'<span style="font-size:13px;margin-right:5px">{icon}</span>' if icon else ""
    st.markdown(
        f'<div style="background:#F4F8FC;border:1px solid #D6E4F0;border-radius:8px;'
        f'padding:24px 20px;{bl}">'
        f'<div style="font-size:10px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:10px">{icon_html}{label}</div>'
        f'<div style="font-size:42px;font-weight:800;color:{color};line-height:1">{value}</div>'
        f'<div style="font-size:12px;color:#6B8CAE;margin-top:8px">{sub}</div>'
        f'</div>', unsafe_allow_html=True)


def section_header(text, margin_top=0):
    style = f"margin-top:{margin_top}px" if margin_top else ""
    st.markdown(f'<div class="sh" style="{style}">{text}</div>', unsafe_allow_html=True)


def info_card(text, border_color="#0072CE"):
    st.markdown(
        f'<div style="padding:10px 14px;background:#F4F8FC;border-left:3px solid {border_color};'
        f'border-radius:4px;font-size:12px;color:#003467;margin-bottom:10px">{text}</div>',
        unsafe_allow_html=True)



def dq_note(text):
    st.markdown(
        f'<div style="font-size:10px;color:#6B8CAE;margin-top:4px;font-style:italic">{text}</div>',
        unsafe_allow_html=True)


def _dot(series, higher_is_good=True, n=3, label="vs prior period"):
    """Return HTML ▲/▼ indicator comparing mean of last n points to prior n points."""
    if series is None:
        return ""
    vals = pd.Series(series).dropna().values
    if len(vals) < n + 1:
        return ""
    recent = vals[-n:].mean()
    prior  = vals[-n * 2:-n].mean() if len(vals) >= n * 2 else vals[: len(vals) - n].mean()
    if abs(prior) < 1e-10:
        return ""
    pct    = (recent - prior) / abs(prior) * 100
    is_up  = pct >= 0
    is_good = is_up == higher_is_good
    clr    = COLORS["success"] if is_good else COLORS["danger"]
    arrow  = "▲" if is_up else "▼"
    return f'<span style="color:{clr};font-size:10px">{arrow} {abs(pct):.1f}% {label}</span>'


def _add_rolling_mean(fig, x_series, y_series, n=3, name="3-mo avg", color=None, dash="dot"):
    """Overlay a rolling mean line on an existing Plotly figure."""
    if color is None:
        color = COLORS["muted"]
    roll = pd.Series(y_series.values if hasattr(y_series, "values") else y_series).rolling(n, min_periods=2).mean()
    fig.add_scatter(
        x=x_series, y=roll,
        mode="lines", name=name,
        line=dict(color=color, width=2, dash=dash),
        hovertemplate=f"<b>{name}</b>: %{{y:.1f}}<extra></extra>",
    )


def _add_regression(fig, x_series, y_series, name="Trend", color=None):
    """Overlay a linear regression trendline on an existing Plotly figure."""
    if color is None:
        color = COLORS["warning"]
    y_arr = np.array(y_series.values if hasattr(y_series, "values") else y_series, dtype=float)
    x_num = np.arange(len(y_arr))
    mask  = ~np.isnan(y_arr)
    if mask.sum() < 3:
        return
    m, b  = np.polyfit(x_num[mask], y_arr[mask], 1)
    y_fit = m * x_num + b
    fig.add_scatter(
        x=x_series, y=y_fit,
        mode="lines", name=name,
        line=dict(color=color, width=1.5, dash="longdash"),
        hoverinfo="skip",
    )


def _add_data_end_line(fig, date_str, label, row=None, col=None):
    kwargs = dict(row=row, col=col) if row else {}
    # add_vline requires numeric x for datetime axes — convert to ms-since-epoch
    x_ms = pd.Timestamp(date_str).timestamp() * 1000
    fig.add_vline(
        x=x_ms, line_width=1, line_dash="dot", line_color="#D97706",
        annotation_text=label,
        annotation_font_size=9,
        annotation_font_color="#D97706",
        **kwargs,
    )


def _filter_epoch(df, date_col):
    """Drop rows where date_col is before 2000."""
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df[df[date_col] >= "2000-01-01"].copy()


# ── Session state init ────────────────────────────────────────────────────────

for k in ("p1", "p2", "p3", "p4", "p5", "p6"):
    if k not in st.session_state:
        st.session_state[k] = {}

if "selected_facility" not in st.session_state:
    st.session_state.selected_facility = "KISUMU_CLEAN"

# ── Landing screen ────────────────────────────────────────────────────────────

if st.session_state.selected_facility is None:
    st.markdown("""
    <style>
    .landing-card{background:#F4F8FC;border:1px solid #D6E4F0;border-radius:12px;
      padding:40px 32px;text-align:center;cursor:pointer}
    .landing-card h2{font-size:28px;font-weight:800;color:#003467;margin-bottom:8px}
    .landing-card p{font-size:13px;color:#6B8CAE;margin-bottom:0}
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<p style="font-size:11px;font-weight:800;letter-spacing:3px;text-transform:uppercase;'
        'color:#0072CE;margin-bottom:4px">Private Hospitals · Analytics</p>',
        unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:22px;font-weight:800;color:#003467;margin-bottom:4px">'
        'Select a facility to begin</p>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:13px;color:#6B8CAE;margin-bottom:32px">'
        'Analytics are scoped to one facility at a time.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="landing-card">
          <h2>TENRI</h2>
          <p>Insurance AR · Readmissions · Service Mix · Beds</p>
          <p style="margin-top:8px;font-size:11px;color:#B0C8E0">Data through July 2022</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
        if st.button("Open TENRI Dashboard", use_container_width=True, key="sel_tenri"):
            st.session_state.selected_facility = "TENRI"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="landing-card">
          <h2>KSH</h2>
          <p>Insurance AR · Theatre · Dialysis · Readmissions</p>
          <p style="margin-top:8px;font-size:11px;color:#B0C8E0">Data through 2026</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
        if st.button("Open KSH Dashboard", use_container_width=True, key="sel_ksh"):
            st.session_state.selected_facility = "KISUMU_CLEAN"
            st.rerun()

    st.stop()

# Facility is set — derive display vars used throughout
facility  = st.session_state.selected_facility
fac_name  = FAC_DISPLAY[facility]
bench_fac = FAC_OTHER[facility]
fac_key   = facility

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    _logo = os.path.join(os.path.dirname(os.path.abspath('__file__')), "ksh_logo.png")
    if os.path.exists(_logo):
        st.image(_logo, width=80)
    st.markdown(
        f'<div style="font-size:11px;font-weight:800;color:#003467;text-transform:uppercase;'
        f'letter-spacing:1.5px;padding:2px 0 2px">Private Hospitals</div>'
        f'<div style="font-size:10px;color:#6B8CAE;padding-bottom:14px;'
        f'border-bottom:1px solid #D6E4F0;margin-bottom:10px">{fac_name}</div>',
        unsafe_allow_html=True)

    page = option_menu(
        menu_title=None,
        options=[
            "Business Overview",
            # "Revenue Leakage",  # AR_PAGE_DISABLED — re-enable when AR_PAGE_ENABLED = True
            "Capacity & Operations",
            "Readmissions",
            "Service Mix",
            "Predictive Analytics",
        ],
        icons=[
            "graph-up-arrow",
            # "cash-coin",  # AR_PAGE_DISABLED
            "hospital",
            "arrow-repeat",
            "pie-chart-fill",
            "cpu",
        ],
        default_index=0,
        styles={
            "container": {
                "padding": "0",
                "background-color": "#F4F8FC",
            },
            "icon": {"color": "#0072CE", "font-size": "13px"},
            "nav-link": {
                "font-size": "12px",
                "font-weight": "600",
                "color": "#6B8CAE",
                "font-family": "Montserrat, sans-serif",
                "padding": "9px 12px",
                "border-radius": "7px",
                "margin-bottom": "2px",
            },
            "nav-link-selected": {
                "background-color": "#0072CE",
                "color": "#fff",
                "font-weight": "700",
                "icon-color": "#fff",
            },
        },
    )

    _data_end = "Apr 2026" if facility == "KISUMU_CLEAN" else "Jul 2022"
    st.markdown(
        f'<div style="font-size:9px;color:#6B8CAE;margin-top:16px;padding-top:8px;'
        f'border-top:1px solid #D6E4F0">Data through {_data_end}</div>',
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — The Business Today
# ══════════════════════════════════════════════════════════════════════════════

if page == "Business Overview":

    if not st.session_state.p1 or st.session_state.p1.get("_fac") != fac_key:
        with st.spinner("Loading…"):
            st.session_state.p1 = {
                "_fac":        fac_key,
                # AR queries retained for future SMART/SLADE reconciliation — AR_PAGE_DISABLED
                # "gap":       q_overview_gap(),
                # "alerts":    q_overview_alerts(),
                # "ksh_trend": q_leakage_ksh_dispatch_trend() if fac_key == "KISUMU_CLEAN" else pd.DataFrame(),
                "beds":        q_beds_los(facility),
                "revpab":      q_beds_revpab(facility),
                "theatre":     q_theatre_trend() if fac_key == "KISUMU_CLEAN" else pd.DataFrame(),
                "readm_trend": q_readmission_trend(),
                "readm_ward":  q_readmission_ward_trend(facility),
                "payer":       q_payer_trend(facility),
                "dialysis":    q_dialysis_trend(facility),
            }

    P = st.session_state.p1

    # ── Computed values ───────────────────────────────────────────────────────

    # Admissions — last 3 months
    readm_fac = P["readm_trend"].copy()
    readm_fac = readm_fac[readm_fac["FACILITY"] == facility]
    readm_fac = _filter_epoch(readm_fac, "ADMISSION_MONTH").sort_values("ADMISSION_MONTH")
    admissions_3mo    = int(readm_fac.tail(3)["TOTAL_ADMISSIONS"].sum()) if len(readm_fac) else 0
    readm_latest_rate = float(readm_fac.tail(1)["READMISSION_30DAY_RATE_PCT"].iloc[0]) if len(readm_fac) else 0

    # Theatre — trailing 3-month + historical peak
    th = _filter_epoch(P["theatre"].copy(), "SESSION_MONTH") if len(P["theatre"]) else pd.DataFrame()
    if len(th):
        th = th.sort_values("SESSION_MONTH")
        th_3mo       = th.tail(3)
        th_comp_rate = round(th_3mo["COMPLETED_SESSIONS"].sum() / max(th_3mo["TOTAL_SESSIONS"].sum(), 1) * 100, 1)
        _pk_idx      = th["COMPLETION_RATE_PCT"].idxmax()
        th_peak_rate = round(float(th.loc[_pk_idx, "COMPLETION_RATE_PCT"]), 0)
        th_peak_lbl  = pd.to_datetime(th.loc[_pk_idx, "SESSION_MONTH"]).strftime("%b %Y")
    else:
        th_comp_rate = th_peak_rate = th_peak_lbl = None

    # Avg LOS weighted by discharged admissions (TENRI fallback for c2)
    beds = P["beds"].copy()
    if len(beds) and beds["DISCHARGED_ADMISSIONS"].sum() > 0:
        avg_los = round(
            (beds["AVG_LOS_DAYS"] * beds["DISCHARGED_ADMISSIONS"]).sum()
            / beds["DISCHARGED_ADMISSIONS"].sum(), 1
        )
    else:
        avg_los = None

    # Direct pay — last 3 months vs prior 3 months
    payer_fac = P["payer"].copy()
    if len(payer_fac):
        payer_fac = payer_fac[payer_fac["FACILITY"] == facility].sort_values("REVENUE_MONTH")
    direct_pay_3mo    = float(payer_fac.tail(3)["CASH_REVENUE"].sum()) if len(payer_fac) else 0
    direct_pay_prior  = float(payer_fac.iloc[-6:-3]["CASH_REVENUE"].sum()) if len(payer_fac) >= 6 else 0

    # Medical Male — latest month rate
    rw = P["readm_ward"].copy()
    if len(rw):
        rw = rw[rw["FACILITY"] == facility]
        rw = _filter_epoch(rw, "ADMISSION_MONTH")
        mm = rw[rw["WARD_CATEGORY"].str.upper() == "MEDICAL — MALE"].sort_values("ADMISSION_MONTH")
    else:
        mm = pd.DataFrame()
    mm_rate  = float(mm.tail(1)["READMISSION_30DAY_RATE_PCT"].iloc[0]) if len(mm) else 0
    mm_month = mm.tail(1)["ADMISSION_MONTH"].dt.strftime("%b %Y").iloc[0] if len(mm) else ""

    # Dialysis — months idle
    dial = P["dialysis"].copy()
    if len(dial):
        dial = dial[dial["FACILITY"] == facility]
    if len(dial):
        last_session = pd.to_datetime(dial["SESSION_MONTH"]).max()
        _data_end_dt = pd.Timestamp("2026-04-01" if facility == "KISUMU_CLEAN" else TENRI_DATA_END)
        months_idle  = (_data_end_dt.year - last_session.year) * 12 + (_data_end_dt.month - last_session.month)
    else:
        months_idle = None

    # Ward RevPAB — compute category-level revenue per bed-day
    revpab_raw = P["revpab"].copy()
    if len(revpab_raw):
        revpab_raw = revpab_raw[revpab_raw["FACILITY"] == facility]
        _pvt_kws   = ["private", "amenity", "vip", "maternity"]
        revpab_raw["ward_type"] = revpab_raw["WARD_CATEGORY"].str.lower().apply(
            lambda x: "Private" if any(k in x for k in _pvt_kws) else "General"
        )
        revpab_cat = (
            revpab_raw.groupby("WARD_CATEGORY", as_index=False)
            .apply(lambda g: pd.Series({
                "REVPAB":         g["TOTAL_REVENUE"].sum() / max(g["TOTAL_BED_DAYS"].sum(), 1),
                "TOTAL_BED_DAYS": g["TOTAL_BED_DAYS"].sum(),
                "ward_type":      g["ward_type"].iloc[0],
            }))
            .sort_values("REVPAB")
        )
    else:
        revpab_cat = pd.DataFrame()

    # ── Threshold rules — edit here to adjust alert sensitivity ─────────────
    _READM_CRITICAL = 15    # % Medical Male 30-day readmission rate
    _READM_WATCH    =  5    # %
    _DIALYSIS_IDLE  =  6    # months idle before surfacing
    _THEATRE_WATCH  = 85    # % completion — below = watch
    _THEATRE_CRIT   = 75    # % completion — below = critical

    # ── Derived signals ───────────────────────────────────────────────────────

    # Facility-wide readmission trend rising 3 consecutive months
    _readm_rising = False
    if len(readm_fac) >= 4:
        _rates = readm_fac["READMISSION_30DAY_RATE_PCT"].tolist()
        _readm_rising = all(_rates[-i - 1] > _rates[-i - 2] for i in range(3))

    # Medical Male delta vs 3-month prior baseline (computed from data)
    _mm_baseline = None
    if len(mm) >= 4:
        _mm_baseline = float(mm.iloc[-4:-1]["READMISSION_30DAY_RATE_PCT"].mean())

    # Theatre revenue gap — avg monthly revenue × unused capacity fraction
    _th_rev_gap = None
    if len(th) and th_comp_rate is not None and th_comp_rate < _THEATRE_WATCH:
        if "TOTAL_REVENUE" in th.columns:
            _th_avg_rev = float(th.tail(6)["TOTAL_REVENUE"].mean())
            if _th_avg_rev > 0:
                _th_rev_gap = _th_avg_rev * (1 - th_comp_rate / 100)

    # Dialysis: KES foregone at historical session rate × months idle
    _dial_kes_low = _dial_kes_high = None
    if months_idle is not None and months_idle >= _DIALYSIS_IDLE and len(dial):
        _avg_sess = float(dial["TOTAL_SESSIONS"].mean())
        if _avg_sess > 0:
            _dial_kes_low  = _avg_sess * 52_000  * months_idle
            _dial_kes_high = _avg_sess * 119_000 * months_idle

    # ── Page label ────────────────────────────────────────────────────────────

    st.markdown(
        '<p style="font-size:11px;font-weight:800;letter-spacing:3px;text-transform:uppercase;'
        'color:#0072CE;margin-bottom:20px">Private Hospitals · The Business Today</p>',
        unsafe_allow_html=True)

    # ── Two-column layout ─────────────────────────────────────────────────────

    col_l, col_r = st.columns([1, 1.6], gap="large")

    with col_l:
        section_header("Active Notices")

        def _notice_card(severity, title, value, delta_line, implication, color):
            _badge_bg = COLORS["danger"] if severity == "CRITICAL" else COLORS["warning"]
            st.markdown(
                f'<div style="background:#fff;border:1px solid #D6E4F0;border-left:4px solid {color};'
                f'border-radius:8px;padding:14px 16px;margin-bottom:12px">'
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
                f'<span style="background:{_badge_bg};color:#fff;font-size:9px;font-weight:800;'
                f'letter-spacing:1.5px;padding:2px 7px;border-radius:3px">{severity}</span>'
                f'<span style="font-size:11px;font-weight:700;color:#003467;text-transform:uppercase;'
                f'letter-spacing:0.8px">{title}</span></div>'
                f'<div style="font-size:22px;font-weight:800;color:{color};line-height:1.2">{value}</div>'
                f'<div style="font-size:11px;color:#6B8CAE;margin-top:4px">{delta_line}</div>'
                f'<div style="font-size:11px;color:#003467;margin-top:6px;line-height:1.5;'
                f'border-top:1px solid #EBF3FB;padding-top:6px">{implication}</div>'
                f'</div>', unsafe_allow_html=True)

        _active = 0

        # Rule 1 — Medical Male readmissions
        if mm_rate > _READM_WATCH:
            _sev = "CRITICAL" if mm_rate > _READM_CRITICAL else "WATCH"
            _col = COLORS["danger"] if mm_rate > _READM_CRITICAL else COLORS["warning"]
            _delta = (
                f"+{mm_rate - _mm_baseline:.0f}pp vs prior 3-month avg ({_mm_baseline:.0f}%)"
                if _mm_baseline is not None else f"Latest reading: {mm_month}"
            )
            _notice_card(
                _sev,
                "Medical Male — Readmissions",
                f"{mm_rate:.1f}%",
                _delta,
                "60+ insured males driving returns · insurers authorising shorter stays · "
                "patients discharged before recovery. Review discharge protocol with clinical lead.",
                _col,
            )
            _active += 1

        # Rule 2 — Facility-wide trend rising (only if Medical Male not already flagged)
        if _readm_rising and mm_rate <= _READM_WATCH:
            _notice_card(
                "WATCH",
                "Readmissions — Rising Trend",
                f"{readm_latest_rate:.1f}%",
                "3 consecutive months increasing · facility-wide",
                "Early signal. Review ward-level breakdown on the Readmissions page.",
                COLORS["warning"],
            )
            _active += 1

        # Rule 3 — Theatre completion below target (KSH only)
        if th_comp_rate is not None and th_comp_rate < _THEATRE_WATCH:
            _sev = "CRITICAL" if th_comp_rate < _THEATRE_CRIT else "WATCH"
            _col = COLORS["danger"] if th_comp_rate < _THEATRE_CRIT else COLORS["warning"]
            _gap_line = (
                f"Est. KES {fmt_kes(_th_rev_gap)}/month in unbilled capacity"
                if _th_rev_gap else
                f"Down {th_peak_rate - th_comp_rate:.0f}pp from peak {th_peak_rate:.0f}% in {th_peak_lbl}"
            )
            _notice_card(
                _sev,
                "Theatre — Completion Below Target",
                f"{th_comp_rate:.0f}%",
                _gap_line,
                f"Trailing 3-month average · peak was {th_peak_rate:.0f}% in {th_peak_lbl}. "
                "Check cancellation and no-show rates on Capacity & Ops page.",
                _col,
            )
            _active += 1

        # Rule 4 — Dialysis idle (KSH only)
        if months_idle is not None and months_idle >= _DIALYSIS_IDLE and facility == "KISUMU_CLEAN":
            _kes_line = (
                f"Est. KES {fmt_kes(_dial_kes_low)}–{fmt_kes(_dial_kes_high)} foregone at historical session rate"
                if _dial_kes_low else "Insufficient session history to estimate foregone revenue"
            )
            _notice_card(
                "WATCH",
                "Dialysis — Equipment Idle",
                f"{months_idle} months",
                _kes_line,
                "Last session Apr 2025 · KES 52K–119K per session. "
                "Referral pipeline needed before equipment utilisation recovers.",
                COLORS["warning"],
            )
            _active += 1

        # All-clear state — shown when nothing crosses a threshold
        if _active == 0:
            _latest_mo = readm_fac["ADMISSION_MONTH"].max().strftime("%b %Y") if len(readm_fac) else "—"
            st.markdown(
                f'<div style="background:#F4F8FC;border-radius:8px;padding:16px 18px;'
                f'color:#6B8CAE;font-size:12px;line-height:1.8">'
                f'<span style="font-weight:700;color:#0BB99F">✓ No active notices</span><br>'
                f'All monitored indicators within range.<br>'
                f'<span style="font-size:10px">Data as at {_latest_mo}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with col_r:
        # ── Ward Revenue per Bed-Day — stat boxes ────────────────────────────
        section_header("Ward Revenue per Bed-Day")
        if len(revpab_cat):
            revpab_grp = (
                revpab_raw.groupby("ward_type", as_index=False)
                .apply(lambda g: pd.Series({
                    "avg_revpab":     g["TOTAL_REVENUE"].sum() / max(g["TOTAL_BED_DAYS"].sum(), 1),
                    "total_bed_days": g["TOTAL_BED_DAYS"].sum(),
                    "total_revenue":  g["TOTAL_REVENUE"].sum(),
                }))
                .reset_index(drop=True)
            )
            _gen = revpab_grp[revpab_grp["ward_type"] == "General"].iloc[0] if "General" in revpab_grp["ward_type"].values else None
            _pvt = revpab_grp[revpab_grp["ward_type"] == "Private"].iloc[0] if "Private" in revpab_grp["ward_type"].values else None
            _total_bd = revpab_grp["total_bed_days"].sum()

            _sb1, _sb2 = st.columns(2)
            for _col_obj, _row, _label in [(_sb1, _gen, "General Wards"), (_sb2, _pvt, "Private Wards")]:
                with _col_obj:
                    if _row is not None:
                        _pct = _row["total_bed_days"] / max(_total_bd, 1) * 100
                        kpi_card(
                            _label,
                            f"KES {_row['avg_revpab']:,.0f}",
                            f"/bed-day · {_pct:.0f}% of admissions",
                            COLORS["muted"] if _label.startswith("General") else COLORS["primary"],
                        )

            if _gen is not None and _pvt is not None:
                _mult = _pvt["avg_revpab"] / max(_gen["avg_revpab"], 1)
                _pvt_pct = _pvt["total_bed_days"] / max(_total_bd, 1) * 100
                dq_note(
                    f"Private wards earn <strong>{_mult:.1f}×</strong> more per bed-day "
                    f"but hold only <strong>{_pvt_pct:.0f}%</strong> of admissions. "
                    "Filling private capacity is the highest-yield lever available."
                )
            if facility == "KISUMU_CLEAN":
                dq_note(
                    "Insured exposure · If 20–30% of insured admissions carry private-tier "
                    "authorisation but are placed in general wards: "
                    "<strong>KES 970K–1.4M/year billed at the wrong rate.</strong>"
                )

        st.markdown("<div style='margin-bottom:4px'></div>", unsafe_allow_html=True)

        # ── Monthly Admissions — 12-month pulse ──────────────────────────────
        section_header("Admissions — 12-Month Pulse")
        if len(readm_fac):
            _adm12 = readm_fac.tail(12)
            _ytd_start = pd.Timestamp(f"{pd.Timestamp.now().year}-01-01")
            _admissions_ytd = int(
                readm_fac[readm_fac["ADMISSION_MONTH"] >= _ytd_start]["TOTAL_ADMISSIONS"].sum()
            )
            fig_adm = go.Figure()
            fig_adm.add_bar(
                x=_adm12["ADMISSION_MONTH"],
                y=_adm12["TOTAL_ADMISSIONS"],
                marker_color=COLORS["primary"],
                opacity=0.65,
                hovertemplate="%{x|%b %Y}: %{y:,} admissions<extra></extra>",
                showlegend=False,
            )
            _add_rolling_mean(fig_adm, _adm12["ADMISSION_MONTH"], _adm12["TOTAL_ADMISSIONS"])
            fig_adm.update_layout(**cl(height=200, yaxis_title="Admissions", showlegend=True,
                                       margin=dict(l=0, r=0, t=10, b=30)))
            st.plotly_chart(fig_adm, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f'<div style="font-size:11px;color:#6B8CAE;margin-top:-8px">'
                f'<strong style="color:#003467">{_admissions_ytd:,}</strong> admissions YTD · '
                f'<strong style="color:#003467">{admissions_3mo:,}</strong> in last 3 months'
                f'</div>',
                unsafe_allow_html=True,
            )



# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Where the Money Isn't Arriving
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Revenue Leakage" and AR_PAGE_ENABLED:  # AR_PAGE_DISABLED — see AR_PAGE_ENABLED flag

    if not st.session_state.p2 or st.session_state.p2.get("_fac") != fac_key:
        with st.spinner("Loading…"):
            st.session_state.p2 = {
                "_fac":        fac_key,
                "gap":         q_leakage_gap(facility),
                "submit_rate": q_leakage_submission_rate(facility),
                "ksh_trend":   q_leakage_ksh_dispatch_trend(),
                "aging":       q_leakage_aging_dist(facility),
                "recovery":    q_leakage_recovery_priority(facility),
            }

    P = st.session_state.p2
    gap_df    = P["gap"]
    submit_df = P["submit_rate"]
    ksh_trend = _filter_epoch(P["ksh_trend"], "INVOICE_MONTH")
    aging_df  = P["aging"]
    rec_df    = P["recovery"]

    st.markdown(
        '<p style="font-size:11px;font-weight:800;letter-spacing:3px;text-transform:uppercase;'
        'color:#0072CE;margin-bottom:4px">Private Hospitals · Where the Money Isn\'t Arriving</p>',
        unsafe_allow_html=True)
    st.caption(f"{fac_name} — insurance AR leakage analysis")
    st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)

    # ── KPI cards ─────────────────────────────────────────────────────────────

    # G7 total: all 33 payers, all-time outstanding
    total_outstanding = rec_df["OUTSTANDING_KES"].sum() if len(rec_df) else gap_df["TOTAL_OUTSTANDING"].sum()
    total_billed      = gap_df["TOTAL_BILLED"].sum()
    total_collected   = gap_df["TOTAL_COLLECTED"].sum()

    # Cliff-specific: outstanding from invoices dated Sep 2025+ (Investigation 6: KES 91.4M)
    ksh_trend_raw = _filter_epoch(P["ksh_trend"], "INVOICE_MONTH")
    cliff_outstanding = float(
        ksh_trend_raw[ksh_trend_raw["INVOICE_MONTH"] >= pd.Timestamp(KSH_DISPATCH_CLIFF)]["TOTAL_OUTSTANDING"].sum()
    ) if facility == "KISUMU_CLEAN" and len(ksh_trend_raw) else 0

    never_submitted = submit_df[submit_df["DISPATCH_RATE_PCT"] == 0]["TOTAL_OUTSTANDING"].sum()
    max_days_row    = rec_df.loc[rec_df["AVG_DAYS_OUTSTANDING"].idxmax()] if len(rec_df) else None
    biggest_exp     = rec_df.iloc[0] if len(rec_df) else None

    c1, c2, c3 = st.columns(3)
    with c1:
        if facility == "KISUMU_CLEAN":
            kpi_card("Total Insurance Outstanding", fmt_kes(total_outstanding),
                     "All 33 payers · all-time uncollected", COLORS["danger"], icon="⚠")
        else:
            kpi_card("Total Uncollected", fmt_kes(total_outstanding), "", COLORS["danger"], icon="⚠")
    with c2:
        if facility == "KISUMU_CLEAN":
            kpi_card("Accumulated Since Sep 2025", fmt_kes(cliff_outstanding),
                     "+KES 11–15M every month since dispatch cliff", COLORS["danger"], icon="⚠")
        else:
            kpi_card("KES Never Submitted", fmt_kes(never_submitted), "", COLORS["danger"], icon="⚠")
    with c3:
        biggest_val = fmt_kes(biggest_exp["OUTSTANDING_KES"]) if biggest_exp is not None else "—"
        biggest_ins = f"{biggest_exp['INSURER']}" if biggest_exp is not None else ""
        kpi_card("Biggest Exposure", biggest_val, biggest_ins, COLORS["danger"], icon="⚠")

    st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────

    tab1, tab2, tab3 = st.tabs(["◉  The Gap", "△  Why It's Sitting There", "∑  Recovery Priority"])

    # ── Tab 1: The Gap ────────────────────────────────────────────────────────

    with tab1:
        if len(gap_df) == 0:
            st.info("No data for selected facility.")
        elif facility == "KISUMU_CLEAN":
            # KSH: collected = 0 for all payers — show outstanding by insurer coloured by dispatch status
            top_gap = gap_df.nlargest(10, "TOTAL_OUTSTANDING").copy()
            top_gap = top_gap.merge(
                submit_df[["INSURER", "DISPATCH_RATE_PCT"]],
                on="INSURER", how="left", suffixes=("", "_sub"))
            top_gap = top_gap.sort_values("TOTAL_OUTSTANDING", ascending=True)

            sha_total = top_gap["TOTAL_OUTSTANDING"].sum()
            sha_row   = top_gap[top_gap["INSURER"].str.upper().str.startswith("SHA")]
            sha_share = sha_row["TOTAL_OUTSTANDING"].sum() / sha_total * 100 if sha_total > 0 and len(sha_row) else 0

            section_header(f"{fmt_kes(sha_total)} outstanding — top 10 payers · where is it and why isn't it moving")

            def _ksh_gap_color(insurer, dispatch_rate):
                if insurer.upper().startswith("AAR"):
                    return COLORS["purple"]
                if pd.isna(dispatch_rate) or dispatch_rate < 5:
                    return COLORS["danger"]
                return COLORS["warning"]

            bar_colors_t1 = [
                _ksh_gap_color(ins, dsp)
                for ins, dsp in zip(top_gap["INSURER"], top_gap["DISPATCH_RATE_PCT"])
            ]
            fig = go.Figure()
            fig.add_bar(
                x=top_gap["TOTAL_OUTSTANDING"],
                y=top_gap["INSURER"],
                orientation="h",
                marker_color=bar_colors_t1,
                text=top_gap["TOTAL_OUTSTANDING"].apply(fmt_kes),
                textposition="inside",
                textfont=dict(size=9, family="Montserrat", color="#fff"),
                hovertemplate=(
                    "<b>%{y}</b><br>Outstanding: %{customdata[0]}<br>"
                    "% Submitted: %{customdata[1]}<extra></extra>"
                ),
                customdata=list(zip(
                    top_gap["TOTAL_OUTSTANDING"].apply(fmt_kes),
                    top_gap["DISPATCH_RATE_PCT"].apply(
                        lambda x: f"{x:.1f}%" if pd.notna(x) else "—"),
                )),
            )
            fig.update_layout(**cl(
                height=480, xaxis_title="KES Outstanding",
                yaxis=dict(gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")),
                xaxis=dict(gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")),
            ))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            dq_note(
                "Red = not submitted (cliff-affected).  "
                "Purple = AAR — submitted but non-paying, root cause required (E7).  "
                f"SHA = {sha_share:.0f}% of outstanding. All collected = 0 — settlement workflow not in system.")
        else:
            # TENRI: genuine billed vs collected comparison
            top_gap = gap_df.nlargest(10, "TOTAL_BILLED").copy()
            top_gap["GAP_KES"] = top_gap["TOTAL_BILLED"] - top_gap["TOTAL_COLLECTED"]
            top_gap["GAP_PCT"] = (
                top_gap["GAP_KES"] / top_gap["TOTAL_BILLED"].replace(0, np.nan) * 100
            ).fillna(0)
            top_gap = top_gap.sort_values("GAP_PCT", ascending=True)

            total_gap    = top_gap["GAP_KES"].sum()
            total_billed = top_gap["TOTAL_BILLED"].sum()
            gap_pct      = total_gap / total_billed * 100 if total_billed > 0 else 0
            section_header(f"{fmt_kes(total_gap)} uncollected across top 10 insurers ({gap_pct:.0f}% of billed)")

            def _gap_color(insurer, pct):
                return (COLORS["danger"] if pct > 70 else
                        COLORS["warning"] if pct > 30 else
                        COLORS["success"])

            gap_colors = [_gap_color(ins, pct) for ins, pct in zip(top_gap["INSURER"], top_gap["GAP_PCT"])]
            fig = go.Figure()
            fig.add_bar(
                name="Collected",
                x=top_gap["TOTAL_COLLECTED"], y=top_gap["INSURER"],
                orientation="h", marker_color=COLORS["success"], opacity=0.85,
                hovertemplate="<b>%{y}</b><br>Collected: %{customdata}<extra></extra>",
                customdata=top_gap["TOTAL_COLLECTED"].apply(fmt_kes),
            )
            fig.add_bar(
                name="Uncollected Gap",
                x=top_gap["GAP_KES"], y=top_gap["INSURER"],
                orientation="h", marker_color=gap_colors, opacity=0.85,
                text=[f"{p:.0f}% gap" for p in top_gap["GAP_PCT"]],
                textposition="inside",
                textfont=dict(size=9, family="Montserrat", color="#fff"),
                hovertemplate=(
                    "<b>%{y}</b><br>Gap: %{customdata[0]}<br>Gap%: %{customdata[1]:.0f}%<extra></extra>"
                ),
                customdata=list(zip(top_gap["GAP_KES"].apply(fmt_kes), top_gap["GAP_PCT"])),
            )
            fig.update_layout(**cl(
                barmode="stack", height=480, xaxis_title="KES",
                legend=dict(orientation="h", y=1.05),
                yaxis=dict(gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")),
                xaxis=dict(gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")),
            ))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            dq_note("Sorted by gap % — worst at top.")

    # ── Tab 2: Why It's Sitting There ─────────────────────────────────────────

    with tab2:
        col_l, col_r = st.columns(2, gap="large")

        with col_l:
            if facility == "KISUMU_CLEAN":
                section_header("All Payers Near-Zero Since Sep 2025 — System Failure, Not Individual Gaps")
            else:
                section_header("Submission Rate by Insurer — Where Claims Are Stalling")
            top_submit = submit_df.nlargest(15, "TOTAL_OUTSTANDING")
            if len(top_submit):
                fig = go.Figure()
                colors = []
                for _, _row in top_submit.iterrows():
                    if _row["INSURER"].upper().startswith("AAR"):
                        colors.append(COLORS["purple"])
                    elif _row["DISPATCH_RATE_PCT"] < 10:
                        colors.append(COLORS["danger"])
                    elif _row["DISPATCH_RATE_PCT"] < 40:
                        colors.append(COLORS["warning"])
                    else:
                        colors.append(COLORS["success"])
                fig.add_bar(
                    x=top_submit["DISPATCH_RATE_PCT"],
                    y=top_submit["INSURER"],
                    orientation="h",
                    marker_color=colors,
                    showlegend=False)
                fig.add_scatter(x=[None], y=[None], mode="markers",
                                marker=dict(symbol="square", size=10, color=COLORS["danger"]),
                                name="< 10% submitted — cliff-affected")
                fig.add_scatter(x=[None], y=[None], mode="markers",
                                marker=dict(symbol="square", size=10, color=COLORS["warning"]),
                                name="10–40% submitted — partial")
                fig.add_scatter(x=[None], y=[None], mode="markers",
                                marker=dict(symbol="square", size=10, color=COLORS["success"]),
                                name="≥ 40% submitted — active")
                fig.add_scatter(x=[None], y=[None], mode="markers",
                                marker=dict(symbol="square", size=10, color=COLORS["purple"]),
                                name="AAR — submits but collects KES 0")
                fig.update_layout(**cl(height=520, xaxis_title="% Claims Submitted",
                                       showlegend=True,
                                       legend=dict(orientation="h", y=-0.18,
                                                   xanchor="left", x=0,
                                                   font=dict(size=9, family="Montserrat"))))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col_r:
            if facility == "KISUMU_CLEAN":
                section_header("Submission Rate Hit Zero Sep 2025 — Eight Months, No Recovery")
                if len(ksh_trend):
                    fig = go.Figure()
                    fig.add_scatter(
                        x=ksh_trend["INVOICE_MONTH"], y=ksh_trend["DISPATCH_RATE_PCT"],
                        mode="lines+markers", name="% Submitted",
                        line=dict(color=COLORS["primary"], width=2),
                        marker=dict(size=6))
                    _add_data_end_line(fig, KSH_DISPATCH_CLIFF, "Dispatch stopped")
                    fig.update_layout(**cl(height=380, yaxis_title="% Claims Submitted",
                                           yaxis_range=[0, 110],
                                           legend=dict(orientation="h", y=1.08)))
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.info("KSH dispatch trend: no data from Jan 2025 onward.")

        # Aging heatmap
        section_header("Where Is Money Sitting? — Insurer × Age", margin_top=16)
        if len(aging_df):
            bucket_order = ["0-30", "31-60", "61-90", "90+"]
            top_aging = aging_df.groupby("INSURER")["TOTAL_OUTSTANDING"].sum().nlargest(10).index
            aging_top = aging_df[aging_df["INSURER"].isin(top_aging)].copy()

            pivot = (
                aging_top
                .pivot_table(index="INSURER", columns="AGING_BUCKET",
                             values="TOTAL_OUTSTANDING", aggfunc="sum", fill_value=0)
                .reindex(columns=[b for b in bucket_order if b in aging_top["AGING_BUCKET"].unique()])
            )
            pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]

            text_vals = [
                [fmt_kes(v) if v > 0 else "—" for v in row]
                for row in pivot.values
            ]

            heat_fig = go.Figure(go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale=[
                    [0.0, "#F0FDF4"],
                    [0.3, "#FEF3C7"],
                    [0.6, "#FED7AA"],
                    [1.0, "#FEE2E2"],
                ],
                text=text_vals,
                texttemplate="%{text}",
                textfont={"size": 9, "family": "Montserrat"},
                hovertemplate="<b>%{y}</b><br>%{x} days: %{text}<extra></extra>",
                showscale=True,
                colorbar=dict(title="KES", thickness=12,
                              tickfont=dict(size=9)),
            ))
            heat_fig.update_layout(**cl(
                height=max(280, min(440, len(pivot) * 38)),
                xaxis_title="Days Outstanding",
                margin=dict(l=0, r=70, t=10, b=20),
            ))
            st.plotly_chart(heat_fig, use_container_width=True,
                            config={"displayModeBar": False})
            dq_note("Darker red = more KES sitting in that aging bucket. "
                    "Top-right corner (90+ days) is the write-off risk zone.")

    # ── Tab 3: Recovery Priority ───────────────────────────────────────────────

    with tab3:

        if len(rec_df):
            section_header("Recovery Priority")

        # ── Section 2: Incoming Admissions Opportunity (KSH only) ────────────
        if facility == "KISUMU_CLEAN":
            section_header("Incoming Admissions — What the Backlog Will Cost If Dispatch Stays Down",
                           margin_top=24)
            info_card(
                "<b>364 admissions</b> forecast over the next 3 months at 94.1% model confidence. "
                "Clinical demand did not drop. If dispatch stays down, these admissions "
                "join the existing backlog — each one starts a new SHA 90-day clock.",
                COLORS["primary"])

            adm_c1, adm_c2 = st.columns(2)
            with adm_c1:
                insured_share = st.slider(
                    "% insured patients (estimate)",
                    min_value=20, max_value=80, value=60, step=5, format="%d%%",
                    key="t3_insured_share",
                )
            with adm_c2:
                sha_rows_t3 = rec_df[
                    (rec_df["FACILITY"] == "KISUMU_CLEAN") &
                    (rec_df["INSURER"].str.upper().str.startswith("SHA"))
                ] if len(rec_df) else pd.DataFrame()
                default_avg = int(
                    sha_rows_t3["OUTSTANDING_KES"].sum() / max(sha_rows_t3["INVOICES"].sum(), 1)
                ) if len(sha_rows_t3) and sha_rows_t3["INVOICES"].sum() > 0 else 20000
                default_avg = min(max(default_avg, 5000), 50000)
                avg_claim_val = st.slider(
                    "Avg SHA claim value per admission (KES)",
                    min_value=5000, max_value=50000,
                    value=default_avg, step=1000, format="KES %d",
                    key="t3_avg_claim",
                )

            forecast_adm = 364
            insured_adm  = int(forecast_adm * insured_share / 100)
            new_backlog  = insured_adm * avg_claim_val

            adm_kc1, adm_kc2, adm_kc3 = st.columns(3)
            with adm_kc1:
                kpi_card("Forecast Admissions", f"{forecast_adm:,}",
                         "Next 3 months · 94.1% model confidence", COLORS["primary"])
            with adm_kc2:
                kpi_card("Insured Admissions", f"{insured_adm:,}",
                         f"At {insured_share}% insured share", COLORS["warning"])
            with adm_kc3:
                kpi_card("New Backlog If Dispatch Stays Down", fmt_kes(new_backlog),
                         "Joins existing backlog · starts new SHA 90-day clock per admission",
                         COLORS["danger"], icon="⚠")

            dq_note("Forecast from Holt's linear trend model (Page 6). "
                    "Avg claim default: SHA outstanding ÷ SHA invoices from G7. "
                    "Adjust sliders for different clinical profiles.")

        # ── Section 3: Appeals Urgency Calendar ──────────────────────────────
        if facility == "KISUMU_CLEAN" and len(ksh_trend):
            section_header("Appeals Urgency Calendar — Where Is Each Invoice Batch Right Now?",
                           margin_top=24)
            info_card(
                "SHA rule: claims must be submitted within <b>90 days</b> of service. "
                "Past 90 days: formal SHA appeals process required — not routine dispatch. "
                "Past 180 days: SHA may reject without dispute resolution. "
                "Each row is one invoice month showing current age and the process required to recover it.",
                COLORS["warning"])

            today_dt = pd.Timestamp.today().normalize()
            kt_cal = ksh_trend[
                ksh_trend["INVOICE_MONTH"] >= pd.Timestamp(KSH_DISPATCH_CLIFF)
            ].sort_values("INVOICE_MONTH").copy()
            kt_cal["days_old"] = (today_dt - kt_cal["INVOICE_MONTH"]).dt.days

            def _tier(days):
                if days > 180:
                    return ("CRITICAL", COLORS["danger"],
                            "rgba(225,29,72,0.09)", "SHA formal dispute only")
                elif days >= 90:
                    return ("Appeals Required", COLORS["warning"],
                            "rgba(217,119,6,0.09)", "File SHA appeals — act before 180 days")
                return ("Routine Dispatch", COLORS["success"],
                        "rgba(11,185,159,0.09)", "Within SHA 90-day window")

            cal_rows = ""
            for _, row in kt_cal.iterrows():
                label, color, bg, action = _tier(int(row["days_old"]))
                cal_rows += (
                    f'<tr style="border-bottom:1px solid #EBF3FB">'
                    f'<td style="padding:10px 14px;font-weight:700;color:#003467">'
                    f'{row["INVOICE_MONTH"].strftime("%b %Y")}</td>'
                    f'<td style="padding:10px 14px;color:#6B8CAE">{int(row["days_old"])} days old</td>'
                    f'<td style="padding:10px 14px">'
                    f'<span style="background:{bg};color:{color};padding:3px 8px;border-radius:4px;'
                    f'font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:0.5px">'
                    f'{label}</span></td>'
                    f'<td style="padding:10px 14px;font-weight:700;color:{color}">'
                    f'{fmt_kes(float(row["TOTAL_OUTSTANDING"]))}</td>'
                    f'<td style="padding:10px 14px;font-size:11px;color:#6B8CAE">{action}</td>'
                    f'</tr>'
                )

            total_critical = float(kt_cal[kt_cal["days_old"] > 180]["TOTAL_OUTSTANDING"].sum())
            total_appeals  = float(
                kt_cal[(kt_cal["days_old"] >= 90) & (kt_cal["days_old"] <= 180)]["TOTAL_OUTSTANDING"].sum()
            )
            total_routine  = float(kt_cal[kt_cal["days_old"] < 90]["TOTAL_OUTSTANDING"].sum())
            grand_total    = total_critical + total_appeals + total_routine

            cal_rows += (
                f'<tr style="background:#F4F8FC;font-weight:800;border-top:2px solid #D6E4F0">'
                f'<td style="padding:10px 14px;color:#003467">Total</td>'
                f'<td style="padding:10px 14px"></td>'
                f'<td style="padding:10px 14px;font-size:11px">'
                f'<span style="color:{COLORS["danger"]};margin-right:10px">Critical {fmt_kes(total_critical)}</span>'
                f'<span style="color:{COLORS["warning"]};margin-right:10px">Appeals {fmt_kes(total_appeals)}</span>'
                f'<span style="color:{COLORS["success"]}">Routine {fmt_kes(total_routine)}</span>'
                f'</td>'
                f'<td style="padding:10px 14px;color:#003467">{fmt_kes(grand_total)}</td>'
                f'<td style="padding:10px 14px;font-size:10px;color:#6B8CAE">'
                f'{fmt_kes(total_critical + total_appeals)} requires active process now</td>'
                f'</tr>'
            )

            st.markdown(
                f'<table style="width:100%;border-collapse:collapse;font-family:Montserrat,sans-serif;'
                f'font-size:12px;color:#003467;margin-bottom:12px">'
                f'<thead><tr style="background:#F4F8FC;font-size:10px;font-weight:700;'
                f'color:#6B8CAE;text-transform:uppercase;letter-spacing:1px">'
                f'<th style="padding:8px 14px;text-align:left">Invoice Month</th>'
                f'<th style="padding:8px 14px;text-align:left">Age</th>'
                f'<th style="padding:8px 14px;text-align:left">SHA Status</th>'
                f'<th style="padding:8px 14px;text-align:left">KES Outstanding</th>'
                f'<th style="padding:8px 14px;text-align:left">Action Required</th>'
                f'</tr></thead>'
                f'<tbody>{cal_rows}</tbody></table>',
                unsafe_allow_html=True)
            dq_note("Age computed from invoice month start to today — actual service date may vary by days. "
                    "Critical tier: requires SHA formal dispute process, not routine claims submission.")

        # ── 90-Day Sprint Table (KSH only) ────────────────────────────────────
        if facility == "KISUMU_CLEAN" and len(rec_df) >= 3:
            sprint_df = (
                rec_df[rec_df["FACILITY"] == "KISUMU_CLEAN"]
                .pipe(lambda d: d[~d["INSURER"].str.contains("AAR", case=False, na=False)])
                .nlargest(3, "OUTSTANDING_KES")
                .copy()
            )
            sprint_df["_recovery_60"] = sprint_df["OUTSTANDING_90PLUS"] * 0.60
            total_sprint_out  = sprint_df["OUTSTANDING_KES"].sum()
            total_sprint_90   = sprint_df["OUTSTANDING_90PLUS"].sum()
            total_sprint_rec  = sprint_df["_recovery_60"].sum()

            section_header(f"90-Day Recovery Sprint — Three Payers · {fmt_kes(total_sprint_rec)} Target", margin_top=24)
            rows_html = ""
            for i, (_, row) in enumerate(sprint_df.iterrows(), 1):
                rows_html += (
                    f'<tr style="border-bottom:1px solid #EBF3FB">'
                    f'<td style="padding:10px 12px;font-weight:700;color:#0072CE">{i}</td>'
                    f'<td style="padding:10px 12px;font-weight:600">{row["INSURER"]}</td>'
                    f'<td style="padding:10px 12px;color:#E11D48;font-weight:700">{fmt_kes(row["OUTSTANDING_KES"])}</td>'
                    f'<td style="padding:10px 12px;color:#6B8CAE">{row["DISPATCH_RATE_PCT"]:.1f}%</td>'
                    f'<td style="padding:10px 12px;color:#D97706">{fmt_kes(row["OUTSTANDING_90PLUS"])}</td>'
                    f'<td style="padding:10px 12px;color:#0BB99F;font-weight:700">{fmt_kes(row["_recovery_60"])}</td>'
                    f'</tr>'
                )
            rows_html += (
                f'<tr style="background:#F4F8FC;font-weight:800;border-top:2px solid #D6E4F0">'
                f'<td style="padding:10px 12px"></td>'
                f'<td style="padding:10px 12px;color:#003467">Total</td>'
                f'<td style="padding:10px 12px;color:#E11D48">{fmt_kes(total_sprint_out)}</td>'
                f'<td style="padding:10px 12px"></td>'
                f'<td style="padding:10px 12px;color:#D97706">{fmt_kes(total_sprint_90)}</td>'
                f'<td style="padding:10px 12px;color:#0BB99F">{fmt_kes(total_sprint_rec)}</td>'
                f'</tr>'
            )
            st.markdown(
                f'<table style="width:100%;border-collapse:collapse;font-family:Montserrat,sans-serif;'
                f'font-size:12px;color:#003467;margin-bottom:12px">'
                f'<thead><tr style="background:#F4F8FC;font-size:10px;font-weight:700;'
                f'color:#6B8CAE;text-transform:uppercase;letter-spacing:1px">'
                f'<th style="padding:8px 12px;text-align:left">#</th>'
                f'<th style="padding:8px 12px;text-align:left">Insurer</th>'
                f'<th style="padding:8px 12px;text-align:left">Outstanding</th>'
                f'<th style="padding:8px 12px;text-align:left">% Submitted</th>'
                f'<th style="padding:8px 12px;text-align:left">90+ Days</th>'
                f'<th style="padding:8px 12px;text-align:left">Recovery at 60%</th>'
                f'</tr></thead>'
                f'<tbody>{rows_html}</tbody></table>',
                unsafe_allow_html=True)
            dq_note("AAR excluded — zero collections recorded even when dispatched. Root cause required before inclusion (E7). Recovery at 60% = 60% of 90+ days bucket.")

        # ── Collection Scenario Model (P2-A) ──────────────────────────────────
        section_header("Recovery Scenario — What If?", margin_top=24)
        info_card(
            "Adjust the sliders to project KES cash inflow under different recovery assumptions. "
            "Based on the top non-AAR payers' 90+ day outstanding.",
            COLORS["primary"])

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            dispatch_pct = st.slider(
                "% of backlog dispatched",
                min_value=0, max_value=100, value=60, step=5,
                format="%d%%",
            )
        with s_col2:
            collection_eff = st.slider(
                "Collection efficiency on dispatched",
                min_value=0, max_value=100, value=60, step=5,
                format="%d%%",
            )

        if len(rec_df):
            scenario_df = (
                rec_df[rec_df["FACILITY"] == "KISUMU_CLEAN"]
                .pipe(lambda d: d[~d["INSURER"].str.contains("AAR", case=False, na=False)])
                .copy()
            ) if facility == "KISUMU_CLEAN" else rec_df.copy()

            scenario_df = scenario_df[scenario_df["OUTSTANDING_90PLUS"] > 0].copy()
            scenario_df["_dispatched"]  = scenario_df["OUTSTANDING_90PLUS"] * (dispatch_pct / 100)
            scenario_df["_collected"]   = scenario_df["_dispatched"] * (collection_eff / 100)
            total_backlog    = scenario_df["OUTSTANDING_90PLUS"].sum()
            total_dispatched = scenario_df["_dispatched"].sum()
            total_projected  = scenario_df["_collected"].sum()

            r1, r2, r3 = st.columns(3)
            with r1:
                kpi_card("90+ Day Backlog", fmt_kes(total_backlog), "Subject to dispatch", COLORS["warning"], icon="⚠")
            with r2:
                kpi_card("Invoices Dispatched", fmt_kes(total_dispatched),
                         f"At {dispatch_pct}% dispatch rate", COLORS["primary"])
            with r3:
                kpi_card("Projected Cash Inflow", fmt_kes(total_projected),
                         f"At {collection_eff}% collection efficiency", COLORS["success"], icon="✓")

            dq_note(f"Scenario: {dispatch_pct}% dispatched × {collection_eff}% collected = "
                    f"{fmt_kes(total_projected)} projected inflow. AAR excluded (non-paying payer, E7).")

        # Detail table in expander
        with st.expander("Full recovery table", expanded=False):
            display_rec = rec_df.copy()
            display_rec["FACILITY"] = display_rec["FACILITY"].replace(FAC_DISPLAY)
            display_rec["DISPATCH_RATE_PCT"] = display_rec["DISPATCH_RATE_PCT"].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
            display_rec["AVG_DAYS_OUTSTANDING"] = display_rec["AVG_DAYS_OUTSTANDING"].apply(
                lambda x: f"{int(x)}" if pd.notna(x) else "—")
            display_rec["OUTSTANDING_KES"]          = display_rec["OUTSTANDING_KES"].apply(fmt_kes)
            display_rec["EXPECTED_RECOVERABLE_KES"] = display_rec["EXPECTED_RECOVERABLE_KES"].apply(fmt_kes)
            display_rec["OUTSTANDING_90PLUS"]       = display_rec["OUTSTANDING_90PLUS"].apply(fmt_kes)
            col_map = {
                "FACILITY": "Facility", "INSURER": "Insurer",
                "INVOICES": "Invoices", "OUTSTANDING_KES": "Outstanding",
                "DISPATCH_RATE_PCT": "% Submitted", "AVG_DAYS_OUTSTANDING": "Avg Days",
                "EXPECTED_RECOVERABLE_KES": "Expected Recoverable",
                "OUTSTANDING_90PLUS": "In 90+ Days",
            }
            st.dataframe(
                display_rec[list(col_map.keys())].rename(columns=col_map),
                hide_index=True, use_container_width=True)
            st.download_button(
                "Download",
                data=rec_df.to_csv(index=False).encode(),
                file_name="recovery_priority.csv",
                mime="text/csv")

        dq_note("AAR KSH: Zero collections recorded across all months — including Oct 2024 when 69 invoices were dispatched. Non-paying payer: root cause required before any recovery sprint (E7).")

    # ── Executive Recommendation ──────────────────────────────────────────────



# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — How We're Using What We Have
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Capacity & Operations":

    if not st.session_state.p3 or st.session_state.p3.get("_fac") != fac_key:
        with st.spinner("Loading…"):
            st.session_state.p3 = {
                "_fac":        fac_key,
                "th_trend":    q_theatre_trend(),
                "th_type":     q_theatre_by_type(),
                "beds_revpab": q_beds_revpab(facility),
                "beds_los":    q_beds_los(facility),
                "dialysis":    q_dialysis_trend(facility),
                "specialty":   q_specialty_admissions(),
                "imaging":     q_imaging_trend(facility),
            }

    P = st.session_state.p3
    th_trend  = _filter_epoch(P["th_trend"], "SESSION_MONTH")
    th_type   = P["th_type"]
    beds_r    = P["beds_revpab"]
    beds_l    = P["beds_los"]
    dialysis  = _filter_epoch(P["dialysis"], "SESSION_MONTH")
    specialty = _filter_epoch(P["specialty"], "ADMISSION_MONTH")
    imaging   = _filter_epoch(P["imaging"], "REVENUE_MONTH")

    st.markdown(
        '<p style="font-size:11px;font-weight:800;letter-spacing:3px;text-transform:uppercase;'
        'color:#0072CE;margin-bottom:4px">Private Hospitals · How We\'re Using What We Have</p>',
        unsafe_allow_html=True)
    st.caption(f"{fac_name} — theatre, beds, imaging, dialysis and specialty capacity")
    st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)

    # ── KPI cards ─────────────────────────────────────────────────────────────

    th_overall_rate = (
        100 * th_trend["COMPLETED_SESSIONS"].sum() / max(th_trend["TOTAL_SESSIONS"].sum(), 1)
        if len(th_trend) else 0)
    _recent_th_rev = th_trend.nlargest(3, "SESSION_MONTH") if len(th_trend) >= 3 else th_trend
    th_monthly_rev = float(_recent_th_rev["TOTAL_REVENUE"].mean()) if len(_recent_th_rev) else 0


    # Trailing 3-month completion rate — more current than all-time average
    _recent_th = th_trend.nlargest(3, "SESSION_MONTH") if len(th_trend) >= 3 else th_trend
    th_recent_rate = (
        100 * _recent_th["COMPLETED_SESSIONS"].sum() / max(_recent_th["TOTAL_SESSIONS"].sum(), 1)
        if len(_recent_th) else th_overall_rate)
    th_rate_color = (COLORS["danger"] if th_recent_rate < 90
                     else COLORS["warning"] if th_recent_rate < 95
                     else COLORS["success"])

    top_revpab_row = beds_r.iloc[0] if len(beds_r) else None
    top_revpab_val = fmt_kes(float(top_revpab_row["REVPAB"])) if top_revpab_row is not None else "—"
    top_revpab_label = (f"{top_revpab_row['WARD_NAME']}" if top_revpab_row is not None else "")

    fac_dialysis = dialysis[dialysis["FACILITY"] == facility]
    dial_sessions = int(fac_dialysis.nlargest(1, "SESSION_MONTH")["TOTAL_SESSIONS"].sum()) if len(fac_dialysis) else 0

    th_dot = _dot(th_trend["COMPLETION_RATE_PCT"] if len(th_trend) else None, higher_is_good=True)

    c1, c2, c3, c4 = st.columns(4)
    if facility == "KISUMU_CLEAN":
        with c1:
            kpi_card("Theatre Completion", f"{th_recent_rate:.1f}%",
                     f"Trailing 3 months · all-time avg: {th_overall_rate:.1f}% {th_dot}",
                     th_rate_color)
        with c2:
            kpi_card("Monthly Theatre Revenue", fmt_kes(th_monthly_rev),
                     "Trailing 3-month avg", COLORS["success"])
        with c3:
            kpi_card("Top Ward RevPAB", top_revpab_val, top_revpab_label, COLORS["warning"])
        with c4:
            if dial_sessions == 0:
                kpi_card("Dialysis Revenue Potential",
                         "KES 52K–140K/mo",
                         "3–5 specialist referrals · Zero capital investment",
                         COLORS["success"], icon="✓")
            else:
                kpi_card("Dialysis Sessions / Month", str(dial_sessions), "Most recent month", COLORS["purple"])
    else:
        avg_los = float(beds_l["AVG_LOS_DAYS"].mean()) if len(beds_l) else 0
        spec_dc_pct = float(specialty["DAY_CASE_PCT"].mean()) if len(specialty) else 0
        with c1:
            kpi_card("Top Ward RevPAB", top_revpab_val, top_revpab_label, COLORS["primary"])
        with c2:
            kpi_card("Avg Length of Stay", f"{avg_los:.1f} days", "", COLORS["warning"])
        with c3:
            kpi_card("Specialty Day Case Rate", f"{spec_dc_pct:.1f}%", "", COLORS["success"])
        with c4:
            kpi_card("Dialysis Sessions / Month", str(dial_sessions), "Most recent month", COLORS["purple"])

    st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────

    tab1, tab2, tab3 = st.tabs(["◉  Theatre", "△  Beds", "∑  Imaging, Diagnostics & Dialysis"])

    # ── Tab 1: Theatre ────────────────────────────────────────────────────────

    with tab1:
        if facility == "TENRI":
            st.info("Theatre analytics are KSH-specific — not applicable for TENRI.")
        else:
            col_l, col_r = st.columns(2, gap="large")

            with col_l:
                section_header(f"Theatre Completion Declining — {th_recent_rate:.0f}% Recent vs {th_overall_rate:.0f}% All-Time Avg")
                if len(th_trend):
                    fig = go.Figure()
                    fig.add_scatter(
                        x=th_trend["SESSION_MONTH"], y=th_trend["COMPLETION_RATE_PCT"],
                        mode="lines+markers", name="Completion %",
                        line=dict(color=COLORS["primary"], width=2), marker=dict(size=5))
                    _add_regression(fig, th_trend["SESSION_MONTH"],
                                    th_trend["COMPLETION_RATE_PCT"], name="Trend",
                                    color=COLORS["warning"])
                    _add_data_end_line(fig, "2025-10-01", "Completion drop")
                    fig.update_layout(**cl(height=360, yaxis_title="Completion %", yaxis_range=[0, 110],
                                           legend=dict(orientation="h", y=1.08)))
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with col_r:
                if len(th_trend):
                    _pk = th_trend.loc[th_trend["TOTAL_REVENUE"].idxmax()]
                    _pk_lbl = (f"{fmt_kes(float(_pk['TOTAL_REVENUE']))} Peak "
                               f"({pd.Timestamp(_pk['SESSION_MONTH']).strftime('%b %Y')})")
                    section_header(f"Monthly Theatre Revenue — {_pk_lbl}, Trending Down")
                else:
                    section_header("Monthly Theatre Revenue — KSH")
                if len(th_trend):
                    fig = go.Figure()
                    fig.add_bar(
                        x=th_trend["SESSION_MONTH"], y=th_trend["TOTAL_REVENUE"],
                        name="Revenue",
                        marker_color=COLORS["success"], opacity=0.75,
                        hovertemplate="%{x|%b %Y}: %{customdata}<extra></extra>",
                        customdata=th_trend["TOTAL_REVENUE"].apply(fmt_kes),
                    )
                    _add_regression(fig, th_trend["SESSION_MONTH"],
                                    th_trend["TOTAL_REVENUE"], name="Trend",
                                    color=COLORS["warning"])
                    fig.update_layout(**cl(height=360, yaxis_title="KES Revenue",
                                           legend=dict(orientation="h", y=1.08)))
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Tab 2: Beds ───────────────────────────────────────────────────────────

    with tab2:
        col_l, col_r = st.columns(2, gap="large")

        with col_l:
            section_header("Revenue per Bed-Day by Ward")
            if len(beds_r):
                top15 = beds_r.dropna(subset=["REVPAB"]).head(15)
                fig = go.Figure()
                fig.add_bar(
                    x=top15["REVPAB"],
                    y=top15["WARD_NAME"],
                    orientation="h",
                    marker_color=COLORS["primary"])
                fig.update_layout(**cl(height=480, xaxis_title="KES per Bed-Day"))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                dq_note("RevPAB = ward revenue / bed-days. Specialty wards excluded (dialysis LOS=0 distorts metric).")
                if facility == "KISUMU_CLEAN":
                    info_card(
                        "KSH RevPAB is currently understated. Insured procedure revenue has not been "
                        "recognised since Sep 2025 — these figures reflect cash-patient revenue only. "
                        "Ward rankings will shift significantly once dispatch is restored.",
                        COLORS["warning"])

        with col_r:
            section_header("Avg Length of Stay by Ward Category")
            if len(beds_l):
                fig = go.Figure()
                fac_colors = {"TENRI": COLORS["primary"], "KISUMU_CLEAN": COLORS["success"]}
                for fac in beds_l["FACILITY"].unique():
                    sub = beds_l[beds_l["FACILITY"] == fac]
                    fig.add_bar(
                        name=FAC_DISPLAY.get(fac, fac),
                        x=sub["AVG_LOS_DAYS"],
                        y=sub["WARD_CATEGORY"],
                        orientation="h",
                        marker_color=fac_colors.get(fac, COLORS["muted"]))
                fig.update_layout(
                    **cl(barmode="group", height=380, xaxis_title="Avg Days",
                         legend=dict(orientation="h", y=1.08)))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        if facility == "KISUMU_CLEAN":
            st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
            section_header("Private Ward Under-Billing — Rate Differential Confirmed")
            ub_c1, ub_c2 = st.columns(2, gap="large")
            with ub_c1:
                kpi_card(
                    "Estimated Annual Under-Billing",
                    "KES 970K–1.4M",
                    "If 20–30% of general ward insured patients hold Private-tier auth",
                    COLORS["warning"], icon="⚠")
            with ub_c2:
                info_card(
                    "Rate differential confirmed: Private Male KES 3,643/bed-day vs General KES 1,668 (2.2×). "
                    "Private Female KES 2,575 vs General KES 1,671 (1.5×). "
                    "1,410 general male + 2,259 general female insured bed-days/year in scope. "
                    "One-week audit of 100 SHA invoices against authorisation tier confirms exact proportion.",
                    COLORS["warning"])

    # ── Tab 3: Imaging, Diagnostics & Dialysis ────────────────────────────────

    with tab3:

        if facility == "KISUMU_CLEAN":
            fac_img = imaging[imaging["FACILITY"] == facility].copy() if "FACILITY" in imaging.columns else imaging.copy()

            # ── Imaging modality summary cards ────────────────────────────────
            MODALITY_ORDER = ["CT / Angio", "MRI", "ECHO / Cardiac", "Ultrasound", "X-Ray"]
            MODALITY_COLORS = {
                "CT / Angio":    COLORS["primary"],
                "MRI":           COLORS["purple"],
                "ECHO / Cardiac":COLORS["success"],
                "Ultrasound":    COLORS["warning"],
                "X-Ray":         COLORS["muted"],
            }

            if len(fac_img):
                # Last 3 months avg per modality
                recent_months = sorted(fac_img["REVENUE_MONTH"].unique())[-3:]
                recent = fac_img[fac_img["REVENUE_MONTH"].isin(recent_months)]
                mod_summary = (
                    recent.groupby("MODALITY")
                    .agg(sessions=("SESSIONS", "sum"),
                         revenue=("REVENUE", "sum"),
                         avg_per=("AVG_PER_SESSION", "mean"))
                    .reindex([m for m in MODALITY_ORDER if m in recent.groupby("MODALITY").groups])
                    .reset_index()
                )
                total_img_rev = mod_summary["revenue"].sum()
                section_header(
                    f"Imaging — {fmt_kes(total_img_rev)} across "
                    f"{len(mod_summary)} modalities (last 3 months)"
                )

                img_cols = st.columns(len(mod_summary), gap="small")
                for col_i, (_, row) in zip(img_cols, mod_summary.iterrows()):
                    with col_i:
                        kpi_card(
                            row["MODALITY"],
                            fmt_kes(row["revenue"]),
                            f"{int(row['sessions'])} sessions · "
                            f"KES {int(row['avg_per']):,}/session",
                            MODALITY_COLORS.get(row["MODALITY"], COLORS["primary"]),
                        )

                st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)

                # ── Imaging revenue trend — stacked bar by modality ───────────
                section_header("Revenue by Modality — Monthly Trend")
                pivot = (
                    fac_img.groupby(["REVENUE_MONTH", "MODALITY"])["REVENUE"]
                    .sum().reset_index()
                )
                fig = go.Figure()
                for mod in MODALITY_ORDER:
                    sub = pivot[pivot["MODALITY"] == mod]
                    if len(sub):
                        fig.add_bar(
                            name=mod,
                            x=sub["REVENUE_MONTH"],
                            y=sub["REVENUE"],
                            marker_color=MODALITY_COLORS.get(mod, COLORS["muted"]),
                            hovertemplate=(
                                f"<b>{mod}</b><br>%{{x|%b %Y}}: "
                                "%{customdata}<extra></extra>"
                            ),
                            customdata=sub["REVENUE"].apply(fmt_kes),
                        )
                fig.update_layout(**cl(
                    barmode="stack", height=400,
                    yaxis_title="KES",
                    legend=dict(orientation="h", y=1.05),
                ))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                dq_note(
                    "Imaging revenue sourced from billing items (stg_procedure_revenue). "
                    "CT dominates — KES 3–7M/month. MRI highest per-session rate (avg KES 35K+). "
                    "Invoices generated but not submitted post-Sep 2025. "
                    "Note: these same items appear within the 'Investigations (incl. fees)' block on the Service Mix page — "
                    "do not add figures from both pages. Pending promotion to gold table G8."
                )

            st.markdown("---")

        # ── Dialysis section ──────────────────────────────────────────────────
        col_l, col_r = st.columns(2, gap="large")

        with col_l:
            section_header("Dialysis Sessions by Month")
            fac_dial = dialysis[dialysis["FACILITY"] == facility]
            if len(fac_dial) < 3:
                st.caption(
                    f"{fac_name} dialysis session data covers fewer than 3 months — "
                    f"trend chart not shown. {len(fac_dial)} month(s) of records present.")
                if facility == "KISUMU_CLEAN":
                    info_card(
                        "KSH dialysis: 3 sessions total (Mar–Apr 2025), zero activity "
                        "for 12 consecutive months since. Machines are genuinely idle — "
                        "no referral pipeline to Kisumu specialists. "
                        "15–20 sessions/month achievable with targeted outreach to 3–5 specialists.",
                        COLORS["purple"])
            else:
                fig = go.Figure()
                color = COLORS["primary"] if facility == "TENRI" else COLORS["success"]
                fig.add_scatter(
                    x=fac_dial["SESSION_MONTH"], y=fac_dial["TOTAL_SESSIONS"],
                    mode="lines+markers", name=fac_name,
                    line=dict(color=color, width=2), marker=dict(size=4))
                if facility == "TENRI":
                    _add_data_end_line(fig, TENRI_DATA_END, "TENRI data end")
                fig.update_layout(**cl(height=360, yaxis_title="Sessions"))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col_r:
            if facility == "TENRI":
                section_header("Specialty Admissions — Day Cases vs Inpatient")
                spec_sum = specialty.groupby("WARD_NAME")[["DAY_CASES","INPATIENT_STAYS"]].sum().reset_index()
                if len(spec_sum):
                    fig = go.Figure()
                    fig.add_bar(name="Day Cases",      x=spec_sum["WARD_NAME"], y=spec_sum["DAY_CASES"],
                                marker_color=COLORS["primary"])
                    fig.add_bar(name="Inpatient Stays", x=spec_sum["WARD_NAME"], y=spec_sum["INPATIENT_STAYS"],
                                marker_color=COLORS["success"])
                    fig.update_layout(
                        **cl(barmode="group", height=360,
                             xaxis_tickangle=-20,
                             legend=dict(orientation="h", y=1.08)))
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    dq_note("Day cases = same-day admit+discharge (primarily dialysis).")



# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Patients Coming Back
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Readmissions":

    if not st.session_state.p4 or st.session_state.p4.get("_fac") != fac_key:
        with st.spinner("Loading…"):
            st.session_state.p4 = {
                "_fac":       fac_key,
                "pattern":    q_readmission_pattern(),
                "trend":      q_readmission_trend(),
                "exposure":   q_readmission_exposure(),
                "benchmark":  q_readmission_benchmark(),
                "ward_trend": q_readmission_ward_trend(facility),
            }

    P = st.session_state.p4
    pattern    = P["pattern"]
    trend      = _filter_epoch(P["trend"], "ADMISSION_MONTH")
    exposure   = P["exposure"]
    benchmark  = P["benchmark"]
    ward_trend = _filter_epoch(P["ward_trend"], "ADMISSION_MONTH")

    st.markdown(
        '<p style="font-size:11px;font-weight:800;letter-spacing:3px;text-transform:uppercase;'
        'color:#0072CE;margin-bottom:4px">Private Hospitals · Patients Coming Back</p>',
        unsafe_allow_html=True)
    st.caption(f"{fac_name} — 30-day readmission rates and revenue at risk")
    st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)

    # ── KPI cards ─────────────────────────────────────────────────────────────

    tenri_rate = (
        100 * pattern[pattern["FACILITY"]=="TENRI"]["READMISSIONS_30DAY"].sum()
        / max(pattern[pattern["FACILITY"]=="TENRI"]["TOTAL_ADMISSIONS"].sum(), 1))
    ksh_rate = (
        100 * pattern[pattern["FACILITY"]=="KISUMU_CLEAN"]["READMISSIONS_30DAY"].sum()
        / max(pattern[pattern["FACILITY"]=="KISUMU_CLEAN"]["TOTAL_ADMISSIONS"].sum(), 1))

    # Primary = selected facility; the other facility is an internal comparison only (not a peer benchmark)
    primary_rate = tenri_rate if facility == "TENRI" else ksh_rate
    bench_rate   = ksh_rate   if facility == "TENRI" else tenri_rate
    gap_pp       = primary_rate - bench_rate
    gap_label    = f"{gap_pp:+.2f}pp"
    gap_color    = COLORS["danger"] if gap_pp > 0 else COLORS["success"]
    kes_exposed  = exposure[exposure["FACILITY"] == facility]["REVENUE_AT_RISK"].sum()

    fac_trend_series = trend[trend["FACILITY"] == facility]["READMISSION_30DAY_RATE_PCT"] if len(trend) else None
    readm_dot = _dot(fac_trend_series, higher_is_good=False)

    # Medical Male latest rate — the actionable alert for KSH
    if facility == "KISUMU_CLEAN" and len(ward_trend):
        _mm_raw = ward_trend[
            (ward_trend["FACILITY"] == "KISUMU_CLEAN") &
            (ward_trend["WARD_CATEGORY"].str.upper() == "MEDICAL — MALE")
        ]
        if len(_mm_raw):
            _mm = _mm_raw.groupby("ADMISSION_MONTH", as_index=False).agg(
                TOTAL_ADMISSIONS=("TOTAL_ADMISSIONS", "sum"),
                READMISSIONS_30DAY=("READMISSIONS_30DAY", "sum"),
            ).sort_values("ADMISSION_MONTH")
            _mm["READMISSION_30DAY_RATE_PCT"] = (
                100.0 * _mm["READMISSIONS_30DAY"] / _mm["TOTAL_ADMISSIONS"].replace(0, pd.NA)
            ).fillna(0)
            med_male_latest = float(_mm.iloc[-1]["READMISSION_30DAY_RATE_PCT"])
        else:
            med_male_latest = None
    else:
        med_male_latest = None

    # KSH: recent monthly values may already exceed the TENRI benchmark even when all-time avg looks good.
    # Use ⚠ when ward crisis is active (med_male_latest above threshold) regardless of all-time gap.
    _ksh_ward_alert = (
        facility == "KISUMU_CLEAN" and
        med_male_latest is not None and
        med_male_latest > 15
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if gap_pp > 0 or _ksh_ward_alert:
            primary_color = COLORS["danger"]
            readm_icon = "⚠"
        else:
            primary_color = COLORS["success"]
            readm_icon = "✓"
        kpi_card(f"{fac_name} 30-Day Rate", f"{primary_rate:.2f}%", readm_dot,
                 primary_color, icon=readm_icon)
    with c2:
        kpi_card(f"{FAC_DISPLAY[bench_fac]} Rate", f"{bench_rate:.2f}%",
                 "Internal comparison · different city and patient population",
                 COLORS["muted"])
    with c3:
        kpi_card("Internal Gap", gap_label, f"vs {FAC_DISPLAY[bench_fac]}", gap_color)
    with c4:
        if facility == "KISUMU_CLEAN" and med_male_latest is not None:
            kpi_card("Medical Male — Latest", f"{med_male_latest:.1f}%",
                     "Most recent month · accelerating since Jan 2026",
                     COLORS["danger"], icon="⚠")
        else:
            kpi_card("KES at Risk", fmt_kes(kes_exposed), "", COLORS["danger"], icon="⚠")

    st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────

    tab1, tab2, tab3 = st.tabs(["◉  The Pattern", "△  What It Costs", "∑  Facility Comparison"])

    # ── Tab 1: The Pattern ────────────────────────────────────────────────────

    with tab1:
        if facility == "KISUMU_CLEAN":
            st.markdown(
                '<div style="background:#FEF2F2;border:1px solid #FECACA;border-left:4px solid #E11D48;'
                'border-radius:6px;padding:14px 18px;margin-bottom:16px">'
                '<div style="font-size:11px;font-weight:800;color:#E11D48;text-transform:uppercase;'
                'letter-spacing:1.5px;margin-bottom:6px">⚠ Ward-Level Alert — Medical Male</div>'
                '<div style="font-size:13px;font-weight:600;color:#003467">'
                'Readmission rate: 16.7% Jan 2026 → <span style="color:#E11D48">26.3% Apr 2026</span></div>'
                '<div style="font-size:11px;color:#6B8CAE;margin-top:4px">'
                'By Mar 2026 the pattern spread to Medical Female, Paediatric, and Private/Amenity simultaneously. '
                'Structural, not statistical noise — requires ward-level action now.'
                '</div></div>',
                unsafe_allow_html=True)

        # Medical Male monthly bar — KSH only
        if facility == "KISUMU_CLEAN" and len(ward_trend):
            _mm_chart_raw = ward_trend[
                (ward_trend["FACILITY"] == "KISUMU_CLEAN") &
                (ward_trend["WARD_CATEGORY"].str.upper() == "MEDICAL — MALE")
            ]
            if len(_mm_chart_raw) >= 2:
                med_male = _mm_chart_raw.groupby("ADMISSION_MONTH", as_index=False).agg(
                    TOTAL_ADMISSIONS=("TOTAL_ADMISSIONS", "sum"),
                    READMISSIONS_30DAY=("READMISSIONS_30DAY", "sum"),
                ).sort_values("ADMISSION_MONTH")
                med_male["READMISSION_30DAY_RATE_PCT"] = (
                    100.0 * med_male["READMISSIONS_30DAY"] / med_male["TOTAL_ADMISSIONS"].replace(0, pd.NA)
                ).fillna(0)

                st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
                _mm_first = float(med_male.iloc[0]["READMISSION_30DAY_RATE_PCT"])
                _mm_last  = float(med_male.iloc[-1]["READMISSION_30DAY_RATE_PCT"])
                _mm_f_mo  = pd.Timestamp(med_male.iloc[0]["ADMISSION_MONTH"]).strftime("%b %Y")
                _mm_l_mo  = pd.Timestamp(med_male.iloc[-1]["ADMISSION_MONTH"]).strftime("%b %Y")
                section_header(
                    f"Medical Male Ward — {_mm_first:.0f}% ({_mm_f_mo}) → {_mm_last:.0f}%"
                    f" ({_mm_l_mo}): Structural Acceleration"
                )
                bar_colors = [
                    COLORS["danger"] if r >= 20 else
                    COLORS["warning"] if r >= 15 else
                    COLORS["primary"]
                    for r in med_male["READMISSION_30DAY_RATE_PCT"].fillna(0)
                ]
                mm_fig = go.Figure()
                mm_fig.add_bar(
                    x=med_male["ADMISSION_MONTH"],
                    y=med_male["READMISSION_30DAY_RATE_PCT"],
                    marker_color=bar_colors,
                    showlegend=False,
                    text=med_male["READMISSION_30DAY_RATE_PCT"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else ""),
                    textposition="outside",
                    textfont=dict(size=9, color="#003467", family="Montserrat"),
                    hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>",
                )
                mm_fig.add_scatter(x=[None], y=[None], mode="markers",
                                   marker=dict(symbol="square", size=10, color=COLORS["primary"]),
                                   name="< 15% — within normal range")
                mm_fig.add_scatter(x=[None], y=[None], mode="markers",
                                   marker=dict(symbol="square", size=10, color=COLORS["warning"]),
                                   name="15–20% — elevated, monitor")
                mm_fig.add_scatter(x=[None], y=[None], mode="markers",
                                   marker=dict(symbol="square", size=10, color=COLORS["danger"]),
                                   name="≥ 20% — critical, action required")
                mm_fig.update_layout(**cl(
                    height=320,
                    yaxis_title="30-Day Rate (%)",
                    yaxis_range=[0, max(med_male["READMISSION_30DAY_RATE_PCT"].max() * 1.3, 35)],
                    showlegend=True,
                    legend=dict(orientation="h", y=1.08, font=dict(size=9, family="Montserrat")),
                ))
                st.plotly_chart(mm_fig, use_container_width=True, config={"displayModeBar": False})

    # ── Tab 2: What It Costs ──────────────────────────────────────────────────

    with tab2:
        col_l, col_r = st.columns(2, gap="large")

        with col_l:
            section_header("KES at Risk by Ward — Insured Readmissions")
            exp_fac = exposure[exposure["FACILITY"] == facility]
            if len(exp_fac):
                top_exp = exp_fac.nlargest(10, "REVENUE_AT_RISK")
                fig = go.Figure()
                fig.add_bar(
                    x=top_exp["REVENUE_AT_RISK"],
                    y=top_exp["WARD_CATEGORY"],
                    orientation="h",
                    marker_color=COLORS["danger"])
                fig.update_layout(**cl(height=400, xaxis_title="KES at Risk"))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col_r:
            if facility == "KISUMU_CLEAN" and med_male_latest is not None:
                section_header(f"Facility Average Masks Ward Crisis — Medical Male at {med_male_latest:.0f}% in Latest Month")
            else:
                section_header(f"{fac_name} 30-Day Readmission Rate — Monthly Trend")
            sub = trend[trend["FACILITY"] == facility] if len(trend) else pd.DataFrame()
            if len(sub):
                fig = go.Figure()
                color = COLORS["danger"] if gap_pp > 0 else COLORS["primary"]
                fig.add_scatter(
                    x=sub["ADMISSION_MONTH"], y=sub["READMISSION_30DAY_RATE_PCT"],
                    mode="lines+markers", name=fac_name,
                    line=dict(color=color, width=2), marker=dict(size=4))
                _add_regression(fig, sub["ADMISSION_MONTH"],
                                sub["READMISSION_30DAY_RATE_PCT"],
                                name="Trend", color=COLORS["warning"])
                if facility == "TENRI":
                    _add_data_end_line(fig, TENRI_DATA_END, "TENRI data end")
                fig.update_layout(**cl(height=380, yaxis_title="30-Day Rate (%)",
                                       legend=dict(orientation="h", y=1.08)))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                if facility == "KISUMU_CLEAN":
                    _mm_note = (f"Medical Male at {med_male_latest:.0f}% in latest month. "
                                if med_male_latest is not None else "")
                    dq_note(f"{_mm_note}AMA discharge log + 72hr callback protocol recommended.")

        # AMA KPI — shown for both facilities when benchmark data is available
        ama_df = benchmark[
            benchmark["DISCHARGE_TYPE"].str.upper().str.contains("AMA|AGAINST MEDICAL", na=False)
        ] if len(benchmark) else pd.DataFrame()
        ama_fac = ama_df[ama_df["FACILITY"] == facility] if len(ama_df) else pd.DataFrame()
        ama_rate = (
            100 * ama_fac["READMISSIONS_30DAY"].sum()
            / max(ama_fac["TOTAL_ADMISSIONS"].sum(), 1)
        ) if len(ama_fac) else None

        if ama_rate is not None and ama_rate > 0:
            st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
            ama_c1, ama_c2 = st.columns(2)
            with ama_c1:
                kpi_card("AMA Discharge 30-Day Return Rate",
                         f"{ama_rate:.1f}%",
                         "Against Medical Advice — highest-risk discharge type",
                         COLORS["danger"], icon="⚠")
            with ama_c2:
                info_card(
                    "AMA patients historically return within 30 days at 6× the facility benchmark. "
                    "Priority for 72-hour post-discharge callback protocol.",
                    COLORS["danger"])
        elif len(ama_df) == 0:
            st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
            info_card(
                "<b>AMA cohort:</b> Against Medical Advice discharge type not found in data. "
                "Historical benchmark: 50% 30-day return rate, 6× facility average. "
                "Ensure AMA is captured as a distinct discharge type in source system.",
                COLORS["warning"])

    # ── Tab 3: Facility Comparison ─────────────────────────────────────────────

    with tab3:
        section_header("Facility Head-to-Head — Overall Readmission Rate")

        fac_summary = {}
        for fac in ["TENRI", "KISUMU_CLEAN"]:
            fac_pat = pattern[pattern["FACILITY"] == fac]
            fac_trend_sub = trend[trend["FACILITY"] == fac] if len(trend) else pd.DataFrame()
            overall_rate = (
                100 * fac_pat["READMISSIONS_30DAY"].sum()
                / max(fac_pat["TOTAL_ADMISSIONS"].sum(), 1)
            ) if len(fac_pat) else 0
            dot_str = _dot(fac_trend_sub["READMISSION_30DAY_RATE_PCT"] if len(fac_trend_sub) else None,
                           higher_is_good=False)
            kes_risk = exposure[exposure["FACILITY"] == fac]["REVENUE_AT_RISK"].sum() if len(exposure) else 0
            fac_summary[fac] = {"rate": overall_rate, "dot": dot_str, "kes": kes_risk}

        cmp1, cmp2 = st.columns(2)
        for col_obj, fac in zip([cmp1, cmp2], ["TENRI", "KISUMU_CLEAN"]):
            with col_obj:
                s      = fac_summary[fac]
                c_name = FAC_DISPLAY[fac]
                r_color = COLORS["danger"] if s["rate"] > 7 else COLORS["warning"] if s["rate"] > 4 else COLORS["success"]
                st.markdown(
                    f'<div style="background:#F4F8FC;border:1px solid #D6E4F0;border-radius:10px;'
                    f'border-left:4px solid {r_color};padding:20px 18px">'
                    f'<div style="font-size:13px;font-weight:800;color:#003467;margin-bottom:12px">{c_name}</div>'
                    f'<div style="font-size:11px;color:#6B8CAE;text-transform:uppercase;letter-spacing:1px">'
                    f'30-Day Rate</div>'
                    f'<div style="font-size:32px;font-weight:800;color:{r_color};line-height:1.1">'
                    f'{s["rate"]:.2f}%</div>'
                    f'<div style="font-size:10px;margin-top:4px">{s["dot"]}</div>'
                    f'<div style="margin-top:12px;font-size:11px;color:#6B8CAE;text-transform:uppercase;'
                    f'letter-spacing:1px">KES at Risk</div>'
                    f'<div style="font-size:18px;font-weight:700;color:{COLORS["danger"]}">'
                    f'{fmt_kes(s["kes"])}</div>'
                    f'</div>', unsafe_allow_html=True)

        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
        with st.expander("Full benchmark data table", expanded=False):
            if len(benchmark):
                disp_bm = benchmark.copy()
                disp_bm["RATE_PCT"] = disp_bm["RATE_PCT"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "—")
                disp_bm["REVENUE_AT_RISK"] = disp_bm["REVENUE_AT_RISK"].apply(fmt_kes)
                disp_bm["FACILITY"] = disp_bm["FACILITY"].replace(FAC_DISPLAY)
                col_map = {
                    "FACILITY": "Facility", "DISCHARGE_TYPE": "Discharge Type",
                    "WARD_CATEGORY": "Ward", "TOTAL_ADMISSIONS": "Admissions",
                    "READMISSIONS_30DAY": "30-Day Readmits",
                    "RATE_PCT": "Rate", "REVENUE_AT_RISK": "KES at Risk",
                    "APPROX_GAP_COUNT": "Approx Gap Count",
                }
                st.dataframe(
                    disp_bm[list(col_map.keys())].rename(columns=col_map),
                    hide_index=True, use_container_width=True)
                st.download_button(
                    "Download Benchmark Table",
                    data=benchmark.to_csv(index=False).encode(),
                    file_name="readmission_benchmark.csv",
                    mime="text/csv")
                dq_note("Approx Gap Count: readmissions where gap was estimated admission-to-admission "
                        "(prior discharge was still open). Directionally conservative — may undercount true 30-day readmits.")



# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — What We Sell and Who Pays
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Service Mix":

    if not st.session_state.p5 or st.session_state.p5.get("_fac") != fac_key:
        with st.spinner("Loading…"):
            st.session_state.p5 = {
                "_fac":     fac_key,
                "mix":      q_service_mix(facility),
                "rebate":   q_rebate_by_insurer(facility),
                "payer":    q_payer_trend(facility),
            }

    P = st.session_state.p5
    mix_df    = P["mix"]
    rebate_df = P["rebate"]
    payer_df  = _filter_epoch(P["payer"], "REVENUE_MONTH")

    st.markdown(
        '<p style="font-size:11px;font-weight:800;letter-spacing:3px;text-transform:uppercase;'
        'color:#0072CE;margin-bottom:4px">Private Hospitals · What We Sell and Who Pays</p>',
        unsafe_allow_html=True)
    st.caption(f"{fac_name} — service line revenue, rebate exposure, payer mix")
    st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)

    # ── KPI cards ─────────────────────────────────────────────────────────────

    excl_rebate = mix_df[mix_df["REVENUE_CATEGORY"] != "Rebate"]
    top_cat_row = excl_rebate.loc[excl_rebate["TOTAL_REVENUE"].idxmax()] if len(excl_rebate) else None
    top_cat_val   = fmt_kes(float(top_cat_row["TOTAL_REVENUE"])) if top_cat_row is not None else "—"
    top_cat_label = f"{top_cat_row['REVENUE_CATEGORY']}" if top_cat_row is not None else ""

    total_rebate = rebate_df["REBATE_KES"].sum()

    fac_payer       = payer_df[payer_df["FACILITY"] == facility]
    fac_insured_pct = float(fac_payer["INSURED_PCT"].mean()) if len(fac_payer) else 0
    fac_cash_pct    = 100 - float(fac_payer.nlargest(3, "REVENUE_MONTH")["INSURED_PCT"].mean()) if len(fac_payer) else 0

    insured_dot = _dot(
        fac_payer["INSURED_PCT"] if len(fac_payer) else None,
        higher_is_good=True,
    )
    cash_dot = _dot(
        (100 - fac_payer["INSURED_PCT"]) if len(fac_payer) else None,
        higher_is_good=(facility != "KISUMU_CLEAN"),  # KSH: rising cash share = insured collapse, not growth
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Top Revenue Category", top_cat_val, top_cat_label, COLORS["primary"])
    with c2:
        kpi_card("Revenue Returned (Rebate)", fmt_kes(abs(total_rebate)), "", COLORS["danger"], icon="⚠")
    with c3:
        kpi_card(f"{fac_name} Insured %", f"{fac_insured_pct:.1f}%", insured_dot, COLORS["warning"])
    with c4:
        if facility == "KISUMU_CLEAN":
            kpi_card(f"{fac_name} Direct-Pay Share", f"{fac_cash_pct:.1f}%",
                     f"Dispatch cliff effect — not demand growth {cash_dot}",
                     COLORS["warning"])
        else:
            kpi_card(f"{fac_name} Direct-Pay Share", f"{fac_cash_pct:.1f}%", cash_dot, COLORS["success"])

    st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────

    tab1, tab2, tab3 = st.tabs(["◉  What's Generating Revenue", "△  Revenue Being Returned", "∑  How Dependent Are We"])

    # ── Tab 1: What's Generating Revenue ─────────────────────────────────────

    with tab1:
        if len(mix_df):
            main_mix    = mix_df[mix_df["REVENUE_CATEGORY"] != "Rebate"].copy()
            rebate_only = mix_df[mix_df["REVENUE_CATEGORY"] == "Rebate"].copy()
            fac_mix     = main_mix[main_mix["FACILITY"] == facility].copy()

            if len(fac_mix):
                fac_mix  = fac_mix.sort_values("TOTAL_REVENUE", ascending=False)
                # Relabel Investigations to flag the professional fees inclusion
                fac_mix["REVENUE_CATEGORY"] = fac_mix["REVENUE_CATEGORY"].replace(
                    {"Investigations": "Investigations (incl. fees)"}
                )
                top_cat  = fac_mix.iloc[0]["REVENUE_CATEGORY"]
                top_rev  = float(fac_mix.iloc[0]["TOTAL_REVENUE"])
                total_r  = float(fac_mix["TOTAL_REVENUE"].sum())
                top_pct  = top_rev / total_r * 100 if total_r > 0 else 0
                section_header(
                    f"{top_cat} drives {top_pct:.0f}% of {fac_name} revenue"
                )

                if facility == "KISUMU_CLEAN":
                    info_card(
                        "<b>Investigations (incl. fees) label is misleading.</b> "
                        "KSH's billing system assigns Surgeon Fees, Consultant Reviews, and Anaesthetist Fees "
                        "to the 'investigation' item type — ~53% of this block is professional fees, "
                        "not clinical investigations. This figure is not a usable growth baseline "
                        "until a billing code audit separates the two categories.",
                        COLORS["warning"])

                fac_mix["_pct_str"] = (
                    fac_mix["TOTAL_REVENUE"] / total_r * 100
                ).apply(lambda x: f"{x:.0f}%")
                tree_fig = go.Figure(go.Treemap(
                    labels=fac_mix["REVENUE_CATEGORY"].tolist(),
                    parents=[""] * len(fac_mix),
                    values=fac_mix["TOTAL_REVENUE"].tolist(),
                    customdata=list(zip(
                        fac_mix["TOTAL_REVENUE"].apply(fmt_kes),
                        fac_mix["_pct_str"],
                    )),
                    texttemplate=(
                        "<b>%{label}</b><br>"
                        "%{customdata[0]}<br>"
                        "%{customdata[1]}"
                    ),
                    textfont=dict(size=12, family="Montserrat"),
                    hovertemplate=(
                        "<b>%{label}</b><br>"
                        "Revenue: %{customdata[0]}<br>"
                        "Share: %{customdata[1]}"
                        "<extra></extra>"
                    ),
                    marker=dict(
                        colorscale=[
                            [0.0, "#EBF3FB"],
                            [0.5, "#5BA4E0"],
                            [1.0, "#003467"],
                        ],
                        showscale=False,
                    ),
                ))
                tree_fig.update_layout(
                    paper_bgcolor="#fff",
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=460,
                    font=dict(family="Montserrat"),
                )
                st.plotly_chart(tree_fig, use_container_width=True,
                                config={"displayModeBar": False})


            if len(rebate_only):
                fac_rebate_only = rebate_only[rebate_only["FACILITY"] == facility]
                if len(fac_rebate_only):
                    info_card(
                        f"Rebate excluded above. Total rebate: "
                        f"{fmt_kes(abs(fac_rebate_only['TOTAL_REVENUE'].sum()))} "
                        f"— see 'Revenue Being Returned' tab.",
                        COLORS["warning"])

    # ── Tab 2: Revenue Being Returned ─────────────────────────────────────────

    with tab2:
        if len(rebate_df):
            rb_p = rebate_df.copy()
            rb_p["ABS_REBATE"] = rb_p["REBATE_KES"].abs()
            rb_p = rb_p.sort_values("ABS_REBATE", ascending=False)
            total_rebate_p = float(rb_p["ABS_REBATE"].sum())
            rb_p["CUM_PCT"] = rb_p["ABS_REBATE"].cumsum() / total_rebate_p * 100

            top3_pct = (
                rb_p.head(3)["ABS_REBATE"].sum() / total_rebate_p * 100
                if total_rebate_p > 0 else 0
            )
            section_header(
                f"Top 3 insurers account for {top3_pct:.0f}% of {fmt_kes(total_rebate_p)} "
                f"returned — focus recovery effort here"
            )

            bar_colors = [
                COLORS["danger"] if v > 10_000_000 else COLORS["warning"]
                for v in rb_p["ABS_REBATE"]
            ]

            pfig = go.Figure()
            pfig.add_bar(
                name="KES Returned",
                x=rb_p["INSURER"],
                y=rb_p["ABS_REBATE"],
                marker_color=bar_colors,
                text=rb_p["ABS_REBATE"].apply(fmt_kes),
                textposition="outside",
                textfont=dict(size=9, family="Montserrat", color="#003467"),
                hovertemplate="<b>%{x}</b><br>Returned: %{customdata}<extra></extra>",
                customdata=rb_p["ABS_REBATE"].apply(fmt_kes),
            )
            pfig.add_scatter(
                name="Cumulative %",
                x=rb_p["INSURER"],
                y=rb_p["CUM_PCT"],
                mode="lines+markers",
                yaxis="y2",
                line=dict(color=COLORS["muted"], width=2, dash="dot"),
                marker=dict(size=6, color=COLORS["muted"]),
                hovertemplate="<b>%{x}</b><br>Cumulative: %{y:.0f}%<extra></extra>",
            )
            pfig.update_layout(**cl(
                height=500,
                yaxis_title="KES Returned to Insurer",
                yaxis2=dict(
                    title="Cumulative %",
                    overlaying="y",
                    side="right",
                    range=[0, 115],
                    ticksuffix="%",
                    tickfont=dict(size=9, color=COLORS["muted"]),
                    showgrid=False,
                    gridcolor="rgba(0,0,0,0)",
                ),
                xaxis=dict(
                    tickangle=-25,
                    tickfont=dict(size=9, color="#6B8CAE"),
                    gridcolor="rgba(0,0,0,0)",
                ),
                legend=dict(orientation="h", y=1.06),
                margin=dict(l=0, r=60, t=20, b=60),
            ))
            st.plotly_chart(pfig, use_container_width=True,
                            config={"displayModeBar": False})

            if facility == "TENRI":
                dq_note("~KES 529K rebate unattributed (26 items — NHIF-discount applied to cash invoices, "
                        "no claim filed). Tracked as E12.")

    # ── Tab 3: How Dependent Are We ───────────────────────────────────────────

    with tab3:
        if facility == "KISUMU_CLEAN":
            section_header("KSH Insured Revenue Collapsed Post Sep 2025 — Cash Now the Majority")
        else:
            section_header("TENRI 99.3% Insured — High Single-Payer Concentration Risk")
        if len(payer_df):
            fig = go.Figure()
            for fac, color in [("TENRI", COLORS["primary"]), ("KISUMU_CLEAN", COLORS["success"])]:
                sub = payer_df[payer_df["FACILITY"] == fac]
                if len(sub):
                    fig.add_scatter(
                        x=sub["REVENUE_MONTH"], y=sub["INSURED_PCT"],
                        mode="lines+markers", name=FAC_DISPLAY.get(fac, fac) + " — % Insured",
                        line=dict(color=color, width=2), marker=dict(size=4))
                    _add_rolling_mean(fig, sub["REVENUE_MONTH"], sub["INSURED_PCT"],
                                      name=f"{FAC_DISPLAY.get(fac, fac)} 3-mo avg",
                                      color=color, dash="dot")
            if facility == "TENRI":
                fig.add_hline(y=99.3, line_dash="dot", line_color=COLORS["danger"],
                              annotation_text="99.3% insured", annotation_font_size=9)
                _add_data_end_line(fig, TENRI_DATA_END, "TENRI data end")
            if facility == "KISUMU_CLEAN":
                _add_data_end_line(fig, KSH_DISPATCH_CLIFF, "Dispatch stopped")
            fig.update_layout(**cl(height=400, yaxis_title="% Insured Revenue",
                                   yaxis_range=[0, 110],
                                   legend=dict(orientation="h", y=1.08)))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            dq_note("KSH insured revenue recognition collapsed from ~80% → ~20% in Jan 2026. "
                    "Patient volume unchanged — dispatch failure prevents recognition.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Predictive Analytics
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Predictive Analytics":

    # Load once — covers both facilities, no facility-scoping needed
    if not st.session_state.p6:
        with st.spinner("Computing demand forecasts…"):
            df_hist, df_fcast = get_forecast()
            st.session_state.p6 = {"hist": df_hist, "fcast": df_fcast}

    df_hist  = st.session_state.p6["hist"]
    df_fcast = st.session_state.p6["fcast"]

    fac_label = "KSH" if facility == "KISUMU_CLEAN" else "TENRI"

    st.markdown(
        '<p style="font-size:11px;font-weight:800;letter-spacing:3px;text-transform:uppercase;'
        'color:#0072CE;margin-bottom:4px">Private Hospitals · Predictive Analytics</p>',
        unsafe_allow_html=True)
    st.caption(f"{fac_name} — admission demand projections, ward planning, model confidence")
    st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)

    # ── Controls row (horizon + capacity — must be above KPIs) ──────────────────

    ctrl1, ctrl2, _ = st.columns([1, 1, 2])
    with ctrl1:
        horizon = st.select_slider(
            "Forecast horizon",
            options=[1, 2, 3],
            value=3,
            format_func=lambda x: f"{x} month{'s' if x > 1 else ''}",
        )
    with ctrl2:
        capacity = st.number_input(
            "Monthly admission capacity",
            min_value=0, value=0, step=10,
            help="Set to see capacity utilisation on the KPI and chart.",
        )

    st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)

    # ── KPI derived values ────────────────────────────────────────────────────

    # Historical mean (all available months)
    h_fac_all = df_hist[
        (df_hist["facility"] == fac_label) & (df_hist["ward"] == "Facility")
    ]
    hist_mean = float(h_fac_all["admissions"].mean()) if len(h_fac_all) else None

    # Forecast for selected horizon month
    fcast_row = df_fcast[
        (df_fcast["facility"] == fac_label) &
        (df_fcast["ward"] == "Facility") &
        (df_fcast["month_offset"] == horizon)
    ]
    fcast_all_fac = df_fcast[
        (df_fcast["facility"] == fac_label) &
        (df_fcast["ward"] == "Facility") &
        (df_fcast["month_offset"] <= horizon)
    ]

    if not fcast_row.empty:
        r          = fcast_row.iloc[0]
        point_val  = int(r["point"])
        lo_val     = int(r["low_90"])
        hi_val     = int(r["high_90"])
        month_lbl  = r["forecast_month"].strftime("%b %Y")
        model_t    = r["model_type"]
        fac_mape   = r["mape"]
    else:
        point_val = lo_val = hi_val = 0
        month_lbl = "—"
        model_t   = "holts"
        fac_mape  = None

    # Delta vs historical mean
    if hist_mean and hist_mean > 0:
        delta_pct = ((point_val - hist_mean) / hist_mean) * 100
        arrow     = "▲" if delta_pct >= 0 else "▼"
        d_color   = COLORS["warning"] if delta_pct > 10 else COLORS["success"] if delta_pct >= 0 else COLORS["muted"]
        delta_html = (
            f"<span style='color:{d_color};font-weight:700'>"
            f"{arrow} {abs(delta_pct):.1f}% vs avg</span>"
        )
    else:
        delta_html = ""

    horizon_labels = {1: "Next Month", 2: "Month 2", 3: "Month 3"}
    card1_title = f"{fac_label} — {horizon_labels[horizon]}"
    card1_sub   = f"{month_lbl}  ·  range {lo_val}–{hi_val}  &nbsp; {delta_html}"

    # Card 2: capacity fill OR cumulative total
    cum_total = int(fcast_all_fac["point"].sum()) if len(fcast_all_fac) else 0
    if capacity > 0:
        fill_pct   = (point_val / capacity * 100) if capacity > 0 else 0
        fill_color = (COLORS["danger"] if fill_pct > 95
                      else COLORS["warning"] if fill_pct > 80
                      else COLORS["success"])
        card2_title = f"Capacity Fill — {horizon_labels[horizon]}"
        card2_val   = f"{fill_pct:.0f}%"
        card2_sub   = f"{point_val} projected / {capacity} capacity"
    else:
        fill_color  = COLORS["primary"]
        card2_title = f"{horizon}-Month Cumulative"
        card2_val   = f"{cum_total:,}"
        card2_sub   = f"admissions over {horizon} month{'s' if horizon > 1 else ''}"

    # Gauge: model confidence = 100 - MAPE (for Holt's), else 0
    if model_t == "holts" and fac_mape is not None:
        confidence  = max(0.0, 100.0 - fac_mape)
        gauge_color = (COLORS["success"] if confidence >= 90
                       else COLORS["warning"] if confidence >= 85
                       else COLORS["danger"])
    else:
        confidence  = None
        gauge_color = COLORS["muted"]

    # ── KPI cards ─────────────────────────────────────────────────────────────

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f'<div style="background:#F4F8FC;border:1px solid #D6E4F0;border-radius:8px;padding:18px 16px">'
            f'<div style="font-size:10px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
            f'letter-spacing:1.5px;margin-bottom:8px">{card1_title}</div>'
            f'<div style="font-size:28px;font-weight:800;color:{COLORS["primary"]};line-height:1">'
            f'{point_val}</div>'
            f'<div style="font-size:11px;color:#6B8CAE;margin-top:6px">{card1_sub}</div>'
            f'</div>', unsafe_allow_html=True)

    with c2:
        st.markdown(
            f'<div style="background:#F4F8FC;border:1px solid #D6E4F0;border-radius:8px;padding:18px 16px">'
            f'<div style="font-size:10px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
            f'letter-spacing:1.5px;margin-bottom:8px">{card2_title}</div>'
            f'<div style="font-size:28px;font-weight:800;color:{fill_color};line-height:1">'
            f'{card2_val}</div>'
            f'<div style="font-size:11px;color:#6B8CAE;margin-top:6px">{card2_sub}</div>'
            f'</div>', unsafe_allow_html=True)

    with c3:
        if confidence is not None:
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                number={"suffix": "%", "font": {"size": 22, "color": "#003467",
                                                "family": "Montserrat"}},
                gauge={
                    "axis": {"range": [0, 100],
                             "tickfont": {"size": 9, "color": "#6B8CAE"}},
                    "bar":  {"color": gauge_color, "thickness": 0.6},
                    "steps": [
                        {"range": [0,  70], "color": "#FEE2E2"},
                        {"range": [70, 85], "color": "#FEF3C7"},
                        {"range": [85, 100], "color": "#D1FAE5"},
                    ],
                    "threshold": {
                        "line": {"color": COLORS["danger"], "width": 2},
                        "thickness": 0.75, "value": 85,
                    },
                },
                title={"text": "Holdout Accuracy (100 − MAPE)",
                       "font": {"size": 10, "color": "#6B8CAE",
                                "family": "Montserrat"}},
            ))
            gauge_fig.update_layout(
                paper_bgcolor="#F4F8FC",
                height=160,
                margin=dict(l=10, r=10, t=30, b=5),
                font=dict(family="Montserrat", color="#003467"),
            )
            st.markdown(
                '<div style="background:#F4F8FC;border:1px solid #D6E4F0;'
                'border-radius:8px;padding:4px 4px 0px 4px">',
                unsafe_allow_html=True)
            st.plotly_chart(gauge_fig, use_container_width=True,
                            config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
            if fac_mape:
                dq_note(f"Holdout MAPE: {fac_mape:.1f}%  ·  Accuracy = 100 − MAPE  ·  Validated threshold: MAPE < 15%")
        else:
            kpi_card("Holdout Accuracy", "Trendline fallback",
                     "Model did not clear 15% MAPE — linear extrapolation only", COLORS["muted"])

    st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────

    tab1, tab2 = st.tabs(["◉  Facility Forecast", "△  Ward Demand"])

    # ── Tab 1: Facility forecast ──────────────────────────────────────────────

    with tab1:
        section_header(f"{fac_label} — Facility-Level Forecast")
        info_card(
            "<b>Holt's Linear Trend</b> — extrapolates the historical direction. "
            "Dashed line = trendline fallback (model did not clear 15% MAPE threshold). "
            "Shaded band = 90% prediction interval.",
            COLORS["muted"])

        h_fac = (
            df_hist[(df_hist["facility"] == fac_label) & (df_hist["ward"] == "Facility")]
            .sort_values("admission_month").tail(18)
        )
        f_fac = (
            df_fcast[
                (df_fcast["facility"] == fac_label) &
                (df_fcast["ward"] == "Facility") &
                (df_fcast["month_offset"] <= horizon)
            ].sort_values("forecast_month")
        )

        chart_model_type = f_fac["model_type"].iloc[0] if len(f_fac) else "holts"
        line_dash  = "solid" if chart_model_type == "holts" else "dash"
        band_color = "rgba(0,114,206,0.12)"

        fig = go.Figure()
        fig.add_scatter(
            x=h_fac["admission_month"], y=h_fac["admissions"],
            mode="lines+markers", name="Historical",
            line=dict(color=COLORS["primary"], width=2),
            marker=dict(size=5))

        if len(f_fac):
            fig.add_scatter(
                x=f_fac["forecast_month"], y=f_fac["high_90"],
                mode="lines", line=dict(width=0), showlegend=False, name="Upper")
            fig.add_scatter(
                x=f_fac["forecast_month"], y=f_fac["low_90"],
                mode="lines", fill="tonexty", fillcolor=band_color,
                line=dict(width=0), showlegend=False, name="90% interval")
            fig.add_scatter(
                x=f_fac["forecast_month"], y=f_fac["point"],
                mode="lines+markers", name="Forecast",
                line=dict(color=COLORS["primary"], width=2, dash=line_dash),
                marker=dict(size=7, symbol="circle-open"))

        cutoff = h_fac["admission_month"].max() if len(h_fac) else None
        if cutoff:
            _add_data_end_line(fig, str(cutoff.date()), "Data cutoff")

        if capacity > 0:
            fig.add_hline(
                y=capacity, line_dash="dot", line_color=COLORS["danger"],
                annotation_text=f"Capacity: {capacity}",
                annotation_font_size=9, annotation_font_color=COLORS["danger"])

        fig.update_layout(**cl(
            height=420,
            yaxis_title="Admissions / Month",
            legend=dict(orientation="h", y=1.06)))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        chart_mape = f_fac["mape"].iloc[0] if len(f_fac) else None
        label_text = (
            f"Holt's Linear Trend · Holdout MAPE: {chart_mape:.1f}%"
            if chart_model_type == "holts" and chart_mape is not None
            else "Linear trendline fallback · Model did not clear 15% MAPE threshold"
        )
        dq_note(label_text)

    # ── Tab 2: Ward Demand ────────────────────────────────────────────────────

    with tab2:
        if facility == "KISUMU_CLEAN":
            info_card(
                "Ward-level forecasting is not available for KSH. All ward volumes fall below "
                "the 25 admissions/ward/month threshold required for reliable predictions. "
                "KSH demand is tracked at facility level.",
                COLORS["muted"])
        else:
            section_header("TENRI — Projected Admissions by Ward (Next Month)")

            ward_fcast = df_fcast[
                (df_fcast["facility"] == "TENRI") &
                (df_fcast["ward"] != "Facility") &
                (df_fcast["month_offset"] == 1)
            ].sort_values("point", ascending=True).reset_index(drop=True)

            if len(ward_fcast):
                bar_colors = [
                    COLORS["primary"] if m == "holts" else COLORS["warning"]
                    for m in ward_fcast["model_type"]
                ]
                err_plus  = (ward_fcast["high_90"] - ward_fcast["point"]).tolist()
                err_minus = (ward_fcast["point"]   - ward_fcast["low_90"]).tolist()

                fig2 = go.Figure()
                fig2.add_bar(
                    x=ward_fcast["point"],
                    y=ward_fcast["ward"],
                    orientation="h",
                    marker_color=bar_colors,
                    showlegend=False,
                    error_x=dict(
                        type="data", symmetric=False,
                        array=err_plus, arrayminus=err_minus,
                        color=COLORS["muted"], thickness=1.5, width=4),
                    text=[f"{int(p)}" for p in ward_fcast["point"]],
                    textposition="outside")
                fig2.add_scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(symbol="square", size=10, color=COLORS["primary"]),
                                 name="Holt's Linear Trend (validated, MAPE < 15%)")
                fig2.add_scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(symbol="square", size=10, color=COLORS["warning"]),
                                 name="Trendline fallback (MAPE > 15% — directional only)")

                fig2.update_layout(**cl(
                    height=420,
                    xaxis_title="Projected Admissions",
                    showlegend=True,
                    legend=dict(orientation="h", y=1.08, font=dict(size=9, family="Montserrat"))))
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
                dq_note("Orange bars: General, Medical Female, Paediatric — high month-to-month variability.")

                # Horizon-adjusted view if > 1 month
                if horizon > 1:
                    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
                    section_header(f"TENRI — {horizon}-Month Outlook by Ward")
                    ward_h = df_fcast[
                        (df_fcast["facility"] == "TENRI") &
                        (df_fcast["ward"] != "Facility") &
                        (df_fcast["month_offset"] <= horizon)
                    ].copy()
                    ward_h["label"] = ward_h["forecast_month"].dt.strftime("%b %Y")

                    fig3 = go.Figure()
                    palette = [COLORS["primary"], COLORS["success"], COLORS["purple"]]
                    for i, offset in enumerate(range(1, horizon + 1)):
                        sub = ward_h[ward_h["month_offset"] == offset].sort_values("ward")
                        fig3.add_bar(
                            name=sub["label"].iloc[0] if len(sub) else f"Month {offset}",
                            x=sub["ward"], y=sub["point"],
                            marker_color=palette[i % len(palette)])
                    fig3.update_layout(**cl(
                        barmode="group", height=380,
                        yaxis_title="Projected Admissions",
                        xaxis_tickangle=-20,
                        legend=dict(orientation="h", y=1.08)))
                    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # ── Model health ──────────────────────────────────────────────────────────

    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
    with st.expander("Model Health", expanded=False):
        meta = (
            df_fcast[
                (df_fcast["facility"] == fac_label) &
                (df_fcast["month_offset"] == 1)
            ][["series", "model_type", "mape"]]
            .drop_duplicates()
            .copy()
        )
        meta["status"] = meta.apply(
            lambda r: "✓ Validated" if r["model_type"] == "holts" else "~ Trendline fallback", axis=1)
        meta["mape"] = meta["mape"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
        meta.columns = ["Series", "Model", "Holdout MAPE", "Status"]
        st.dataframe(meta, hide_index=True, use_container_width=True)

        from ksh.facility_utilization.m1_ward_forecast import VALIDATED_DATE, RETRAIN_DATE
        dq_note(
            f"Last validated: {VALIDATED_DATE.strftime('%Y-%m-%d')}  ·  "
            f"Retrain recommended by: {RETRAIN_DATE.strftime('%Y-%m-%d')}  ·  "
            f"Holdout = last 3 months per facility  ·  Acceptance threshold: MAPE < 15%")

