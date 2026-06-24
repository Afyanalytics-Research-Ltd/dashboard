"""
Private Hospitals Executive Dashboard — TENRI + KSH
Run: streamlit run private_analysis/dashboard.py
"""

import sys
import os
import json
import urllib.parse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests as _requests

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu

import warnings
warnings.filterwarnings("ignore")

from facility_utilization.m1_ward_forecast import get_forecast
from facility_utilization.forecasting.adapter import build_contract as _build_forecast_contract
from notifier import send_digest, get_recipients, write_current_notices
from facility_utilization.queries import (
    q_overview_gap, q_overview_alerts,
    q_leakage_gap, q_leakage_submission_rate, q_leakage_ksh_dispatch_trend,
    q_leakage_aging_dist, q_leakage_recovery_priority,
    q_theatre_trend, q_theatre_by_type, q_theatre_emergency_tat,
    q_beds_revpab, q_beds_los, q_beds_monthly, q_dialysis_trend, q_specialty_admissions,
    q_imaging_trend,
    # q_readmission_pattern, q_readmission_trend,                     # READM_HIDDEN — clinical finding, not ops metric
    # q_readmission_exposure, q_readmission_benchmark, q_readmission_ward_trend,
    q_service_mix, q_rebate_by_insurer, q_payer_trend,
    q_ward_admissions_monthly, q_ward_los_monthly, q_ward_discharge_monthly,
    q_doctor_workload_monthly, q_lab_monthly, q_visit_summary, q_peak_breakdown,
    q_peak_ward_dist, q_doctor_ward_share, q_cd12_monthly_rate,
    q_doctor_conversion_monthly,
    q_btr_bti_monthly, q_admission_tat_bimodal, q_admission_tat_monthly,
    q_revpab_private_monthly,
    q_peak_tat_conversion, q_peak_doctor_load, q_peak_patient_funnel,
    q_dialysis_ops_monthly,
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

st.markdown(
    '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css"/>',
    unsafe_allow_html=True,
)

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
@keyframes fadeUp{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
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

CHAT_URL = os.getenv("CHAT_URL", "http://localhost:8001")

CHART_LAYOUT = dict(
    paper_bgcolor="#fff", plot_bgcolor="#fff",
    font=dict(family="Montserrat", color="#003467"),
    margin=dict(l=0, r=0, t=10, b=30),
    xaxis=dict(gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")),
    yaxis=dict(gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")),
)

TENRI_DATA_END     = "2022-07-27"
KSH_DATA_END       = pd.Timestamp("2026-05-08")
KSH_DISPATCH_CLIFF = "2025-09-01"

# ── Phase 13 — Ward intelligence thresholds (Inv 20–25b) ─────────────────────
_KSH_WARDS = ["MEDICAL — MALE", "MEDICAL — FEMALE", "MATERNITY", "PRIVATE / AMENITY", "PAEDIATRIC"]

_TRAFFIC_WATCH    = {"MEDICAL — FEMALE": 40, "PAEDIATRIC": 32, "MEDICAL — MALE": 25,
                     "MATERNITY": 20, "PRIVATE / AMENITY": 18}
_TRAFFIC_CRIT     = {"MEDICAL — FEMALE": 45, "PAEDIATRIC": 37, "MEDICAL — MALE": None,
                     "MATERNITY": 25, "PRIVATE / AMENITY": 22}

_LOS_WATCH        = {"MEDICAL — MALE": 5.0, "MEDICAL — FEMALE": 5.0,
                     "PRIVATE / AMENITY": 5.5, "MATERNITY": 4.0, "PAEDIATRIC": 3.5}
_LOS_CRIT         = {"MEDICAL — MALE": 7.0, "MEDICAL — FEMALE": 7.0,
                     "PRIVATE / AMENITY": 8.0, "MATERNITY": 6.0, "PAEDIATRIC": 5.0}

_PR_WATCH         = {"MATERNITY": 82, "PAEDIATRIC": 68, "MEDICAL — FEMALE": 68,
                     "PRIVATE / AMENITY": 75, "MEDICAL — MALE": 62}
_PR_CRIT          = {"MATERNITY": None, "PAEDIATRIC": 78, "MEDICAL — FEMALE": 78,
                     "PRIVATE / AMENITY": None, "MEDICAL — MALE": 72}

# READM_HIDDEN — ward readmission thresholds: clinical metric, not surfaced on dashboard
# _READM_WARD_WATCH = {"MEDICAL — MALE": 10, "MEDICAL — FEMALE": 8, "MATERNITY": 10,
#                      "PRIVATE / AMENITY": 12, "PAEDIATRIC": 8}
# _READM_WARD_CRIT  = {"MEDICAL — MALE": 15, "MEDICAL — FEMALE": 12, "MATERNITY": 15,
#                      "PRIVATE / AMENITY": 18, "PAEDIATRIC": 12}

_LAB_VOL_WATCH    = 430
_LAB_VOL_CRIT     = 350
_LAB_ABNORM_WATCH = 9.0
_LAB_ABNORM_CRIT  = 11.0

_DOC_CONC_WATCH   = 40
_DOC_CONC_CRIT    = 50

# Rule 29 — Ward Idle BTR/BTI — per-ward P25 BTR floor and P75 BTI ceiling
# Recalibrated 2026-06-18 using corrected 32-bed denominator (Inv 54). Full Sep 2024–Apr 2026 window.
_BTR_P25 = {
    "General Female":    3.86, "General Male":      4.50,
    "General Maternity": 1.43, "Pediatric General": 3.17,
    "Private Female":    1.50, "Private Male":      1.33,
    "Private Maternity": 0.50,
}
_BTI_P75 = {
    "General Female":     4.6, "General Male":       3.7,
    "General Maternity": 19.0, "Pediatric General":  7.4,
    "Private Female":    18.0, "Private Male":      20.2,
    "Private Maternity": 56.0,
}
_OCT_2025_GAP  = "2025-10-01"  # pipeline gap month — exclude from all alert computations

# Rule 30 — Admission TAT monthly deterioration (Inv 47)
_TAT_WATCH     = 45.0    # fast_pct below this for 2 consecutive months = WATCH
_TAT_CRIT      = 35.0    # fast_pct below this for 1 month = CRITICAL
_TAT_P75_WATCH = 240.0   # p75 TAT above this (4h) for 2 consecutive months = WATCH
_TAT_P75_CRIT  = 360.0   # p75 TAT above this (6h) for 1 month = CRITICAL

# Rule 32 — Private ward revenue drop (Inv 49) — combined Private Female + Male vs 3-month rolling avg
_REVPAB_WATCH_DROP = 25.0   # % drop below 3-month rolling avg triggers WATCH

# Rule 33 — Physician workload (Inv 50) — WATCH when visits > P90 for 2 consecutive months
# P75 values (clearing threshold for future stateful digest): eawando 734, lowino 481, jogutu 343
# makinyi excluded — departed Dec 2025
_DOC_WL_TRACKED = frozenset({"eawando", "lowino", "jogutu"})
_DOC_WL_P90 = {"eawando": 795, "lowino": 595, "jogutu": 378}

# Rule 34 — CD12 Critical Creatinine Non-Admission (Inv 51)
# CRITICAL flag format CL/CH confirmed Jul 2025+. Mar–May 2026 data absent (pipeline gap).
# Staleness guard: skip if latest qualifying month > 3 months ago.
_CD12_WATCH    = 50.0   # non-admission rate % above this = WATCH (min 8 critical events)
_CD12_CRIT     = 65.0   # non-admission rate % above this = CRITICAL
_CD12_MIN_EVTS = 8      # minimum critical events required — below this the rate is noise

# Rule 35 — CT Imaging Volume Drop (Inv 52) — CT/Angio sessions vs 3-month rolling avg
# Source: stg_procedure_revenue (silver exception — G8 gold table pending)
# Apr 2026 will fire CRITICAL on first deployment (87 sessions vs avg ~149)
_IMAGING_WATCH_PCT = 80.0   # sessions < 80% of 3-month rolling avg = WATCH
_IMAGING_CRIT_PCT  = 65.0   # sessions < 65% of 3-month rolling avg = CRITICAL

# Rule 31 — BOR ward low occupancy (Inv 48) — per-ward P25 BOR floor, 2 consecutive months
_BOR_P25 = {  # P25 floors from Inv 48 re-run — corrected 32-bed denominator (Inv 54, 2026-06-18)
    "General Female":    40.0, "General Male":      46.7,
    "General Maternity": 11.1, "Pediatric General": 23.7,
    "Private Female":    11.8, "Private Male":       9.7,
    "Private Maternity":  8.1,
}
# Rule 31b — BOR ward high occupancy — MoH optimal anchor is 85% (CLAUDE.md L1)
# WATCH: 2 consecutive months > 85% (sustained pressure, no buffer for emergencies)
# CRITICAL: single month > 95% (ward effectively full)
_BOR_HIGH_WATCH = 85.0
_BOR_HIGH_CRIT  = 95.0


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
        f'<div style="background:#F4F8FC;border-left:3px solid #B0C8E0;border-radius:4px;'
        f'padding:8px 12px;margin:10px 0;font-size:12px;color:#003467;line-height:1.5">'
        f'<span style="font-weight:700;color:#6B8CAE">Note · </span>{text}</div>',
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


def _ema_next(monthly_series: pd.Series, span: int = 3):
    """EMA(span) next-month point estimate from a monthly value series.
    Drops the last month when KSH_DATA_END falls before day 25 (partial month
    depresses the average). Returns None if fewer than 3 usable points remain."""
    s = pd.to_numeric(monthly_series, errors="coerce").dropna().reset_index(drop=True)
    if len(s) > 1 and KSH_DATA_END.day < 25:
        s = s.iloc[:-1]
    if len(s) < 3:
        return None
    return float(s.ewm(span=span, adjust=False).mean().iloc[-1])


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


# ── ML platform paths ─────────────────────────────────────────────────────────

_ML_PLATFORM         = Path(os.path.abspath(__file__)).parent / "ml_platform"
_FORECAST_CACHE      = _ML_PLATFORM / "forecast_cache.json"
_RETRAIN_STATUS_FILE = _ML_PLATFORM / "retrain_status.json"
_DJANGO_RETRAIN_URL  = "http://127.0.0.1:8001/forecast/retrain/"

# ── Session state init ────────────────────────────────────────────────────────

for k in ("p1", "p2", "p3", "p4", "p5", "p6", "p6_ksh", "p_causal"):
    if k not in st.session_state:
        st.session_state[k] = {}

if "active_notices" not in st.session_state:
    st.session_state["active_notices"] = []

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
          <p>Insurance AR · Service Mix · Beds</p>
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
          <p>Insurance AR · Theatre · Dialysis</p>
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
    _logo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ksh_logo.png")
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
            "Causal Intelligence",
            "Service Mix",
            "Predictive Analytics",
        ],
        icons=[
            "graph-up-arrow",
            # "cash-coin",  # AR_PAGE_DISABLED
            "hospital",
            "diagram-3",
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

    # ── Notify ────────────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:9px;font-weight:800;color:#0072CE;text-transform:uppercase;'
        'letter-spacing:2px;padding-bottom:8px;border-bottom:1px solid #D6E4F0;'
        'margin-bottom:10px">Notify</div>',
        unsafe_allow_html=True)

    _n_count     = len(st.session_state.get("active_notices", []))
    _recipients  = get_recipients()
    _notice_dot  = (
        f'<span style="color:#E11D48;font-weight:700">&#9679; {_n_count} notice'
        f'{"s" if _n_count != 1 else ""} firing</span>'
        if _n_count else
        '<span style="color:#6B8CAE">No active notices</span>'
    )
    _to_display  = ", ".join(_recipients) if _recipients else "No recipients configured"

    _sidebar_notices = st.session_state.get("active_notices", [])
    _notice_rows_html = ""
    for _sn in _sidebar_notices:
        _dot_col = "#E11D48" if _sn["level"] == "CRITICAL" else "#D97706"
        _notice_rows_html += (
            f'<div style="display:flex;align-items:flex-start;gap:6px;'
            f'padding:5px 0;border-bottom:1px solid #EBF3FB">'
            f'<span style="color:{_dot_col};font-size:8px;margin-top:2px">&#9679;</span>'
            f'<div><div style="font-size:10px;font-weight:700;color:#003467;line-height:1.3">'
            f'{_sn["title"]}</div>'
            f'<div style="font-size:9px;color:#6B8CAE">{_sn["metric"]}</div></div>'
            f'</div>'
        )

    st.markdown(
        f'<div style="font-size:10px;margin-bottom:4px">{_notice_dot}</div>'
        f'{_notice_rows_html}'
        f'<div style="font-size:9px;color:#6B8CAE;margin-top:8px;margin-bottom:10px">'
        f'To: {_to_display}</div>',
        unsafe_allow_html=True)

    if st.button("Send Executive Digest", use_container_width=True, key="send_digest_btn"):
        if not _recipients:
            st.error("Set DIGEST_RECIPIENTS in .env")
        else:
            _notices  = st.session_state.get("active_notices", [])
            _fac_disp = FAC_DISPLAY.get(facility, facility)
            _ok, _msg = send_digest(_fac_disp, _notices)
            if _ok:
                st.success("Sent ✓")
            else:
                st.error(f"Failed: {_msg}")

    # ── Chat ──────────────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:9px;font-weight:800;color:#0072CE;text-transform:uppercase;'
        'letter-spacing:2px;padding-bottom:8px;border-bottom:1px solid #D6E4F0;'
        'margin-bottom:10px">Intelligence Chat</div>',
        unsafe_allow_html=True)
    st.link_button(
        "Open Chat →",
        "http://localhost:8001/",
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — The Business Today
# ══════════════════════════════════════════════════════════════════════════════

if page == "Business Overview":

    if not st.session_state.p1 or st.session_state.p1.get("_fac") != fac_key:
        with st.spinner("Loading…"):
            _is_ksh = (fac_key == "KISUMU_CLEAN")
            st.session_state.p1 = {
                "_fac":        fac_key,
                # AR queries retained for future SMART/SLADE reconciliation — AR_PAGE_DISABLED
                # "gap":       q_overview_gap(),
                # "alerts":    q_overview_alerts(),
                # "ksh_trend": q_leakage_ksh_dispatch_trend() if fac_key == "KISUMU_CLEAN" else pd.DataFrame(),
                "beds":        q_beds_los(facility),
                "revpab":      q_beds_revpab(facility),
                "theatre":     q_theatre_trend() if _is_ksh else pd.DataFrame(),
                # "readm_trend": q_readmission_trend(),       # READM_HIDDEN
                # "readm_ward":  q_readmission_ward_trend(facility),
                "payer":       q_payer_trend(facility),
                "dialysis":    q_dialysis_trend(facility),
                "dialysis_ops": q_dialysis_ops_monthly() if _is_ksh else pd.DataFrame(),
                # Phase 13 — KSH-only intelligence layer data
                "ward_adm":    q_ward_admissions_monthly(facility) if _is_ksh else pd.DataFrame(),
                "ward_los":    q_ward_los_monthly(facility)        if _is_ksh else pd.DataFrame(),
                "ward_dc":     q_ward_discharge_monthly(facility)  if _is_ksh else pd.DataFrame(),
                "doctor_wl":   q_doctor_workload_monthly()         if _is_ksh else pd.DataFrame(),
                "lab":         q_lab_monthly()                     if _is_ksh else pd.DataFrame(),
                "visit_sum":   q_visit_summary().rename(columns=str.lower) if _is_ksh else pd.DataFrame(),
                "btr_bti":         q_btr_bti_monthly()           if _is_ksh else pd.DataFrame(),
                "adm_tat_monthly": q_admission_tat_monthly()    if _is_ksh else pd.DataFrame(),
                "revpab_priv":     q_revpab_private_monthly()   if _is_ksh else pd.DataFrame(),
                "cd12_rate":       q_cd12_monthly_rate()        if _is_ksh else pd.DataFrame(),
                "imaging_alert":   q_imaging_trend("KISUMU_CLEAN") if _is_ksh else pd.DataFrame(),
            }

    P = st.session_state.p1

    # ── Computed values ───────────────────────────────────────────────────────

    # READM_HIDDEN — readmission data not loaded; defaults used where downstream code references these vars
    readm_fac         = pd.DataFrame()
    admissions_3mo    = 0
    readm_latest_rate = 0

    # Visit summary (KSH only) — total visits from EVALUATION_VISITS, inpatient from ward_adm
    _vs = _filter_epoch(P["visit_sum"].copy(), "visit_month").sort_values("visit_month") if len(P["visit_sum"]) else pd.DataFrame()
    _wa = _filter_epoch(P["ward_adm"].copy(), "ADMISSION_MONTH") if len(P["ward_adm"]) else pd.DataFrame()
    if len(_vs):
        _vs_cur = int(_vs["total_visits"].sum())  # cumulative from Sep 2024
        if len(_wa):
            _wa_mo  = (_wa[(_wa["FACILITY"] == facility) &
                           (_wa["ADMISSION_MONTH"] >= "2024-09-01")]
                       .groupby("ADMISSION_MONTH", as_index=False)["ADMISSIONS"].sum())
            _ip_cur = int(_wa_mo["ADMISSIONS"].sum()) if len(_wa_mo) else 0
        else:
            _ip_cur = 0
        _op_cur = max(_vs_cur - _ip_cur, 0)
    else:
        _vs_cur = _ip_cur = _op_cur = None

    # Theatre — trailing 3-month + historical peak
    th = _filter_epoch(P["theatre"].copy(), "SESSION_MONTH") if len(P["theatre"]) else pd.DataFrame()
    if len(th):
        th = th.sort_values("SESSION_MONTH")
        th_3mo       = th.tail(3)
        th_comp_rate = round(th_3mo["COMPLETED_SESSIONS"].sum() / max(th_3mo["TOTAL_SESSIONS"].sum(), 1) * 100, 1)
        _th_last_row = th.iloc[-1]
        th_last_rate = round(float(_th_last_row["COMPLETION_RATE_PCT"]), 1)
        th_last_lbl  = pd.to_datetime(_th_last_row["SESSION_MONTH"]).strftime("%b %Y")
        _pk_idx      = th["COMPLETION_RATE_PCT"].idxmax()
        th_peak_rate = round(float(th.loc[_pk_idx, "COMPLETION_RATE_PCT"]), 0)
        th_peak_lbl  = pd.to_datetime(th.loc[_pk_idx, "SESSION_MONTH"]).strftime("%b %Y")
    else:
        th_comp_rate = th_peak_rate = th_peak_lbl = None
        th_last_rate = th_last_lbl = None

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

    # READM_HIDDEN — ward readmission data not loaded
    rw       = pd.DataFrame()
    mm       = pd.DataFrame()
    mm_rate  = 0
    mm_month = ""

    # Dialysis — months idle
    dial = P["dialysis"].copy()
    if len(dial):
        dial = dial[dial["FACILITY"] == facility]
    if facility == "KISUMU_CLEAN":
        # KSH: use FINANCE_INVOICES ops data — rpt_dialysis.total_sessions is broken (Inv 63)
        _dial_ops = P.get("dialysis_ops", pd.DataFrame()).copy()
        _dial_ops_complete = (
            _dial_ops[~_dial_ops["IS_PARTIAL_MONTH"]]
            if len(_dial_ops) and "IS_PARTIAL_MONTH" in _dial_ops.columns
            else _dial_ops
        )
        if len(_dial_ops_complete):
            _last_ops = pd.to_datetime(_dial_ops_complete["INVOICE_MONTH"]).max()
            _ops_end  = pd.Timestamp("2026-04-01")
            months_idle = (_ops_end.year - _last_ops.year) * 12 + (_ops_end.month - _last_ops.month)
        else:
            months_idle = None
    elif len(dial):
        last_session = pd.to_datetime(dial["SESSION_MONTH"]).max()
        _data_end_dt = pd.Timestamp(TENRI_DATA_END)
        months_idle  = (_data_end_dt.year - last_session.year) * 12 + (_data_end_dt.month - last_session.month)
    else:
        months_idle = None

    # ── Phase 13 computed values (KSH only) ──────────────────────────────────

    _is_ksh = (facility == "KISUMU_CLEAN")

    # Helper: last N monthly values for a ward in a dataframe
    def _ward_tail(df, ward, n, val_col, month_col="admission_month"):
        if not len(df):
            return []
        w = df[df["WARD_CATEGORY"].str.upper() == ward.upper()]
        w = _filter_epoch(w, month_col).sort_values(month_col)
        return w.tail(n)[val_col].tolist() if len(w) >= n else w[val_col].tolist()

    # Helper: 2-consecutive-month breach check (last 2 values both exceed threshold)
    def _two_consec(vals, threshold, direction="above"):
        if len(vals) < 2:
            return False
        if direction == "above":
            return vals[-1] > threshold and vals[-2] > threshold
        return vals[-1] < threshold and vals[-2] < threshold

    # Helper: Pulse trend arrow from last 3 monthly values (slope relative to threshold size)
    def _trend_arrow(vals, threshold):
        if len(vals) < 2:
            return "→"
        slope = (vals[-1] - vals[0]) / max(len(vals) - 1, 1) if len(vals) >= 2 else 0
        rel = abs(slope) / max(abs(threshold), 1)
        if slope > 0 and rel > 0.15:
            return "↑↑"
        if slope > 0 and rel > 0.05:
            return "↑"
        if slope < 0 and rel > 0.05:
            return "↓"
        return "→"

    # Helper: Pulse metric status (GREEN/AMBER/RED) for higher=worse metrics
    def _status_hi(current, watch, critical=None):
        # RED: at/above watch threshold, or within 10% of critical
        if critical and current >= critical * 0.9:
            return "RED"
        if current >= watch:
            return "RED"
        # AMBER: within 15% of watch (approaching)
        if current >= watch * 0.85:
            return "AMBER"
        return "GREEN"

    # Helper: Pulse metric status for lower=worse metrics (lab volume)
    def _status_lo(current, watch_low, critical_low=None):
        if critical_low and current <= critical_low:
            return "RED"
        if current <= watch_low:
            return "RED"
        # AMBER: within 10% above watch (approaching floor)
        if current <= watch_low * 1.10:
            return "AMBER"
        return "GREEN"

    _STATUS_COLOR = {"GREEN": "#0BB99F", "AMBER": "#D97706", "RED": "#E11D48"}
    _STATUS_EMOJI = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴"}

    # ── Pre-compute rule signals for Operational Pulse + notice rules ─────────

    # Ward traffic: latest month admissions per ward
    _ward_adm_df = _filter_epoch(P["ward_adm"].copy(), "ADMISSION_MONTH") if _is_ksh and len(P["ward_adm"]) else pd.DataFrame()
    if len(_ward_adm_df):
        _ward_adm_df = _ward_adm_df[_ward_adm_df["FACILITY"] == facility].sort_values("ADMISSION_MONTH")

    def _ward_adm_latest(ward):
        if not len(_ward_adm_df):
            return None
        w = _ward_adm_df[_ward_adm_df["WARD_CATEGORY"].str.upper() == ward]
        return float(w.tail(1)["ADMISSIONS"].iloc[0]) if len(w) else None

    def _ward_adm_prev2(ward):
        if not len(_ward_adm_df):
            return []
        w = _ward_adm_df[_ward_adm_df["WARD_CATEGORY"].str.upper() == ward]
        return w.tail(2)["ADMISSIONS"].tolist() if len(w) >= 2 else w["ADMISSIONS"].tolist()

    # Ward LOS: last 2 months median per ward
    _ward_los_df = _filter_epoch(P["ward_los"].copy(), "ADMISSION_MONTH") if _is_ksh and len(P["ward_los"]) else pd.DataFrame()
    if len(_ward_los_df):
        _ward_los_df = _ward_los_df[_ward_los_df["FACILITY"] == facility].sort_values("ADMISSION_MONTH")

    def _ward_los_vals(ward, n=3):
        if not len(_ward_los_df):
            return []
        w = _ward_los_df[_ward_los_df["WARD_CATEGORY"].str.upper() == ward]
        return w.tail(n)["MEDIAN_LOS_DAYS"].tolist() if len(w) else []

    # Ward discharge — Patient Request rate
    _ward_dc_df = _filter_epoch(P["ward_dc"].copy(), "ADMISSION_MONTH") if _is_ksh and len(P["ward_dc"]) else pd.DataFrame()
    if len(_ward_dc_df):
        _ward_dc_df = _ward_dc_df[_ward_dc_df["FACILITY"] == facility].sort_values("ADMISSION_MONTH")

    def _ward_pr_vals(ward, n=3):
        if not len(_ward_dc_df):
            return []
        w = _ward_dc_df[_ward_dc_df["WARD_CATEGORY"].str.upper() == ward]
        return w.tail(n)["PATIENT_REQUEST_PCT"].tolist() if len(w) else []

    def _ward_pr_admissions(ward):
        if not len(_ward_dc_df):
            return 0
        w = _ward_dc_df[_ward_dc_df["WARD_CATEGORY"].str.upper() == ward]
        return float(w.tail(1)["TOTAL_ADMISSIONS"].iloc[0]) if len(w) else 0

    # Doctor workload
    _doc_df = _filter_epoch(P["doctor_wl"].copy(), "VISIT_MONTH") if _is_ksh and len(P["doctor_wl"]) else pd.DataFrame()
    if len(_doc_df):
        _doc_df = _doc_df.sort_values("VISIT_MONTH")

    _doc_latest_month = None
    _top_doc_name     = None
    _top_doc_pct      = 0.0
    _top_doc_visits   = 0
    _burnout_alerts   = []  # list of (name, pct_over_avg, visits, avg)

    if len(_doc_df):
        _latest_vm = _doc_df["VISIT_MONTH"].max()
        _doc_latest_month = _latest_vm
        _doc_latest = _doc_df[_doc_df["VISIT_MONTH"] == _latest_vm]
        _total_latest_visits = int(_doc_latest["MONTHLY_VISITS"].sum())
        if _total_latest_visits > 0:
            _top_row = _doc_latest.nlargest(1, "MONTHLY_VISITS").iloc[0]
            _top_doc_name   = _top_row["USERNAME"]
            _top_doc_visits = int(_top_row["MONTHLY_VISITS"])
            _top_doc_pct    = round(_top_doc_visits / _total_latest_visits * 100, 1)

        # Burnout: check each doctor's latest 2 months vs their 3-month rolling avg
        for _uname in _doc_df["USERNAME"].unique():
            _d = _doc_df[_doc_df["USERNAME"] == _uname].sort_values("VISIT_MONTH")
            if len(_d) < 4:
                continue
            _rolling_avg = float(_d.iloc[-4:-1]["MONTHLY_VISITS"].mean())
            if _rolling_avg < 1:
                continue
            _last2 = _d.tail(2)["MONTHLY_VISITS"].tolist()
            _pct_latest  = _last2[-1] / _rolling_avg * 100 if len(_last2) >= 1 else 0
            _pct_prev    = _last2[-2] / _rolling_avg * 100 if len(_last2) >= 2 else 0
            if _pct_latest > 150 and _pct_prev > 150:
                _display_name = f"{_uname[0].upper()}.{_uname[1:].capitalize()}"
                _burnout_alerts.append((_display_name, _pct_latest, int(_last2[-1]), int(_rolling_avg)))

    # Lab
    _lab_df = _filter_epoch(P["lab"].copy(), "LAB_MONTH") if _is_ksh and len(P["lab"]) else pd.DataFrame()
    if len(_lab_df):
        _lab_df = _lab_df.sort_values("LAB_MONTH")

    _lab_latest_visits  = None
    _lab_latest_abnorm  = None
    _lab_latest_month   = None

    if len(_lab_df):
        # Exclude partial current month (use only months with > 50 visits to avoid partial-month noise)
        _lab_complete = _lab_df[_lab_df["DISTINCT_VISITS"] > 50]
        if len(_lab_complete):
            _lab_row = _lab_complete.tail(1).iloc[0]
            _lab_latest_visits = int(_lab_row["DISTINCT_VISITS"])
            _lab_latest_abnorm = float(_lab_row["ABNORMAL_PCT"])
            _lab_latest_month  = _lab_row["LAB_MONTH"]

    def _lab_visits_vals(n=3):
        if not len(_lab_df):
            return []
        _lc = _lab_df[_lab_df["DISTINCT_VISITS"] > 50]
        return _lc.tail(n)["DISTINCT_VISITS"].tolist() if len(_lc) else []

    def _lab_abnorm_vals(n=3):
        if not len(_lab_df):
            return []
        _lc = _lab_df[_lab_df["DISTINCT_VISITS"] > 50]
        return _lc.tail(n)["ABNORMAL_PCT"].tolist() if len(_lc) else []

    # Ward readmission signals (from existing q_readmission_ward_trend)
    _rw_df = rw.copy()

    def _ward_readm_vals(ward, n=3):
        if not len(_rw_df):
            return []
        w = _rw_df[_rw_df["WARD_CATEGORY"].str.upper() == ward.upper()]
        w = _filter_epoch(w.copy(), "ADMISSION_MONTH").sort_values("ADMISSION_MONTH")
        return w.tail(n)["READMISSION_30DAY_RATE_PCT"].tolist() if len(w) else []

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

    # Rule 29 pre-compute — Ward idle BTR/BTI (latest complete month per ward)
    _btr_bti_alert_df = P.get("btr_bti", pd.DataFrame()).copy()
    _ward_idle_alerts = []   # list of (ward_name, btr, btr_p25, bti, bti_p75, month_lbl)
    if _is_ksh and len(_btr_bti_alert_df):
        _btr_bti_alert_df.columns = _btr_bti_alert_df.columns.str.lower()
        _btr_bti_alert_df = _btr_bti_alert_df[
            _btr_bti_alert_df["month"].astype(str) != _OCT_2025_GAP
        ].sort_values("month")
        for _wi in _btr_bti_alert_df["ward_name"].unique():
            _wi_p25 = _BTR_P25.get(_wi)
            _wi_p75 = _BTI_P75.get(_wi)
            if _wi_p25 is None or _wi_p75 is None:
                continue
            _wd = _btr_bti_alert_df[_btr_bti_alert_df["ward_name"] == _wi].tail(1)
            if not len(_wd):
                continue
            _wi_btr     = float(_wd["btr"].iloc[0])
            _wi_bti     = float(_wd["bti_days"].iloc[0])
            _wi_mo_lbl  = pd.to_datetime(_wd["month"].iloc[0]).strftime("%b %Y")
            if _wi_btr < _wi_p25 and _wi_bti > _wi_p75:
                _ward_idle_alerts.append((_wi, _wi_btr, _wi_p25, _wi_bti, _wi_p75, _wi_mo_lbl))

    # Rule 30 pre-compute — Admission TAT monthly deterioration
    _tat_mo_df           = P.get("adm_tat_monthly", pd.DataFrame()).copy()
    _tat_latest_fast_pct = None
    _tat_latest_p50      = None
    _tat_latest_p75      = None
    _tat_latest_month    = None
    _tat_fast_pcts       = []
    _tat_p75_vals        = []
    if _is_ksh and len(_tat_mo_df):
        _tat_mo_df.columns = _tat_mo_df.columns.str.lower()
        _tat_mo_df = _tat_mo_df[
            _tat_mo_df["tat_month"].astype(str) != _OCT_2025_GAP
        ].sort_values("tat_month")
        if len(_tat_mo_df):
            _tat_row             = _tat_mo_df.tail(1).iloc[0]
            _tat_latest_fast_pct = float(_tat_row["fast_pct"])
            _tat_latest_p50      = int(_tat_row["p50_tat_min"])
            _tat_latest_p75      = int(_tat_row["p75_tat_min"]) if "p75_tat_min" in _tat_mo_df.columns else None
            _tat_latest_month    = _tat_row["tat_month"]
            _tat_fast_pcts       = _tat_mo_df.tail(2)["fast_pct"].tolist()
            _tat_p75_vals        = _tat_mo_df.tail(2)["p75_tat_min"].tolist() if "p75_tat_min" in _tat_mo_df.columns else []

    # Rule 31 pre-compute — BOR ward low occupancy (last 2 complete months per ward)
    # Re-uses P["btr_bti"] — bor_pct column already present from q_btr_bti_monthly()
    # Suppress if Rule 29 already fired for the same ward (BTR/BTI already covers that signal)
    _bor_alert_df    = P.get("btr_bti", pd.DataFrame()).copy()
    _bor_ward_alerts = []   # list of (ward_name, bor_latest, bor_p25, bor_gap, month_lbl)
    _rule29_wards    = {w for w, *_ in _ward_idle_alerts}
    if _is_ksh and len(_bor_alert_df):
        _bor_alert_df.columns = _bor_alert_df.columns.str.lower()
    if _is_ksh and len(_bor_alert_df) and "bor_pct" in _bor_alert_df.columns:
        _bor_alert_df = _bor_alert_df[
            _bor_alert_df["month"].astype(str) != _OCT_2025_GAP
        ].sort_values("month")
        for _bw in _bor_alert_df["ward_name"].unique():
            if _bw in _rule29_wards:
                continue   # Rule 29 already fired — suppress to avoid double-alert
            _bw_p25 = _BOR_P25.get(_bw)
            if _bw_p25 is None:
                continue
            _bw_tail = _bor_alert_df[_bor_alert_df["ward_name"] == _bw].tail(2)
            if len(_bw_tail) < 2:
                continue
            _bw_months = pd.to_datetime(_bw_tail["month"]).tolist()
            # guard: only fire if the 2 rows are truly adjacent calendar months
            if ((_bw_months[1].year * 12 + _bw_months[1].month)
                    - (_bw_months[0].year * 12 + _bw_months[0].month)) != 1:
                continue
            _bw_vals = _bw_tail["bor_pct"].tolist()
            if _two_consec(_bw_vals, _bw_p25, direction="below"):
                _bor_latest  = float(_bw_vals[-1])
                _bor_mo_lbl  = _bw_months[1].strftime("%b %Y")
                _bor_ward_alerts.append((_bw, _bor_latest, _bw_p25, round(_bw_p25 - _bor_latest, 1), _bor_mo_lbl))

    # Rule 31b pre-compute — BOR ward high occupancy (Inv 48)
    # Same source data. WATCH: 2 consecutive months > 85%. CRITICAL: single month > 95%.
    # Not suppressed by Rule 29 — high BOR + low BTR would be contradictory and worth surfacing.
    _bor_high_alerts = []   # list of (ward_name, bor_latest, sev, month_lbl)
    if _is_ksh and len(_bor_alert_df) and "bor_pct" in _bor_alert_df.columns:
        for _bh in _bor_alert_df["ward_name"].unique():
            _bh_tail = _bor_alert_df[_bor_alert_df["ward_name"] == _bh].tail(2)
            if not len(_bh_tail):
                continue
            _bh_months = pd.to_datetime(_bh_tail["month"]).tolist()
            _bh_vals   = _bh_tail["bor_pct"].tolist()
            _bh_latest = float(_bh_vals[-1])
            _bh_mo_lbl = _bh_months[-1].strftime("%b %Y")
            if _bh_latest > _BOR_HIGH_CRIT:
                _bor_high_alerts.append((_bh, _bh_latest, "CRITICAL", _bh_mo_lbl))
            elif (len(_bh_tail) == 2
                  and ((_bh_months[1].year * 12 + _bh_months[1].month)
                       - (_bh_months[0].year * 12 + _bh_months[0].month)) == 1
                  and _two_consec(_bh_vals, _BOR_HIGH_WATCH, direction="above")):
                _bor_high_alerts.append((_bh, _bh_latest, "WATCH", _bh_mo_lbl))

    # Rule 32 pre-compute — Private ward revenue drop (Inv 49)
    # Combined Private Female + Male monthly revenue vs 3-month rolling avg
    _revpab_priv_df = P.get("revpab_priv", pd.DataFrame()).copy()
    _revpab_alert   = None  # (latest_rev, rolling_avg, drop_pct, latest_adm, month_lbl)
    if _is_ksh and len(_revpab_priv_df):
        _revpab_priv_df.columns = _revpab_priv_df.columns.str.lower()
        _revpab_priv_df = _revpab_priv_df[
            _revpab_priv_df["admission_month"].astype(str) != _OCT_2025_GAP
        ].sort_values("admission_month")
        if len(_revpab_priv_df) >= 4:
            _rv_tail4        = _revpab_priv_df.tail(4)
            _rv_rolling_avg  = float(_rv_tail4.head(3)["total_revenue"].mean())
            _rv_latest       = float(_rv_tail4.iloc[-1]["total_revenue"])
            _rv_latest_adm   = int(_rv_tail4.iloc[-1]["total_admissions"])
            _rv_mo_lbl       = pd.to_datetime(_rv_tail4.iloc[-1]["admission_month"]).strftime("%b %Y")
            if _rv_rolling_avg > 0:
                _rv_drop_pct = round(100.0 * (1 - _rv_latest / _rv_rolling_avg), 1)
                if _rv_drop_pct > _REVPAB_WATCH_DROP:
                    _revpab_alert = (_rv_latest, _rv_rolling_avg, _rv_drop_pct, _rv_latest_adm, _rv_mo_lbl)

    # Rule 33 pre-compute — Physician workload (Inv 50)
    # Re-uses P["doctor_wl"]. Tracks eawando, lowino, jogutu only (makinyi departed Dec 2025).
    # WATCH: last 2 consecutive months both > P90. Consecutive-month guard applied.
    _doc_wl_df     = P.get("doctor_wl", pd.DataFrame()).copy()
    _doc_wl_alerts = []   # list of (username, latest_visits, p90, month_lbl)
    if _is_ksh and len(_doc_wl_df):
        _doc_wl_df.columns = _doc_wl_df.columns.str.lower()
        _doc_wl_df = _doc_wl_df[
            _doc_wl_df["visit_month"].astype(str) != _OCT_2025_GAP
        ].sort_values("visit_month")
        for _dr in _DOC_WL_TRACKED:
            _dr_p90  = _DOC_WL_P90.get(_dr)
            if _dr_p90 is None:
                continue
            _dr_tail = _doc_wl_df[_doc_wl_df["username"] == _dr].tail(2)
            if len(_dr_tail) < 2:
                continue
            _dr_months = pd.to_datetime(_dr_tail["visit_month"]).tolist()
            if ((_dr_months[1].year * 12 + _dr_months[1].month)
                    - (_dr_months[0].year * 12 + _dr_months[0].month)) != 1:
                continue   # non-consecutive months — skip
            _dr_vals = _dr_tail["monthly_visits"].tolist()
            if _two_consec(_dr_vals, _dr_p90, direction="above"):
                _doc_wl_alerts.append((
                    _dr, int(_dr_vals[-1]), _dr_p90,
                    _dr_months[1].strftime("%b %Y"),
                ))

    # Rule 34 pre-compute — CD12 Critical Creatinine Non-Admission (Inv 51)
    # Staleness guard: skip if latest qualifying month is > 3 months before current month.
    # Mar–May 2026 data absent — this guard ensures stale data doesn't fire as a live alert.
    _cd12_df    = P.get("cd12_rate", pd.DataFrame()).copy()
    _cd12_alert = None  # (sev, rate, total_critical, not_admitted, month_lbl)
    if _is_ksh and len(_cd12_df):
        _cd12_df.columns = _cd12_df.columns.str.lower()
        _cd12_df = _cd12_df[
            _cd12_df["critical_month"].astype(str) != _OCT_2025_GAP
        ].sort_values("critical_month")
        _cd12_qual   = _cd12_df[_cd12_df["total_critical"] >= _CD12_MIN_EVTS]
        if len(_cd12_qual):
            _cd12_row    = _cd12_qual.tail(1).iloc[0]
            _cd12_mo     = pd.to_datetime(_cd12_row["critical_month"])
            _cd12_cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(months=3)
            if _cd12_mo >= _cd12_cutoff:
                _cd12_rate_v  = float(_cd12_row["non_admission_rate_pct"])
                _cd12_total   = int(_cd12_row["total_critical"])
                _cd12_not_adm = int(_cd12_row["not_admitted"])
                _cd12_mo_lbl  = _cd12_mo.strftime("%b %Y")
                if _cd12_rate_v > _CD12_CRIT:
                    _cd12_alert = ("CRITICAL", _cd12_rate_v, _cd12_total, _cd12_not_adm, _cd12_mo_lbl)
                elif _cd12_rate_v > _CD12_WATCH:
                    _cd12_alert = ("WATCH", _cd12_rate_v, _cd12_total, _cd12_not_adm, _cd12_mo_lbl)

    # Rule 35 pre-compute — CT Imaging Volume Drop (Inv 52)
    # Filters to CT/Angio only. Excludes current partial month in Python (query has no such filter).
    _img_alert_df = P.get("imaging_alert", pd.DataFrame()).copy()
    _img_alert    = None  # (sev, sessions_latest, rolling_avg, pct_of_avg, drop_pct, month_lbl)
    if _is_ksh and len(_img_alert_df):
        _img_alert_df.columns = _img_alert_df.columns.str.lower()
        _img_ct = _img_alert_df[_img_alert_df["modality"] == "CT / Angio"].copy()
        _img_current_mo = pd.Timestamp.today().replace(day=1)
        _img_ct = _img_ct[
            (pd.to_datetime(_img_ct["revenue_month"]) < _img_current_mo) &
            (_img_ct["revenue_month"].astype(str) != _OCT_2025_GAP)
        ].sort_values("revenue_month")
        if len(_img_ct) >= 4:
            _img_tail4       = _img_ct.tail(4)
            _img_rolling_avg = float(_img_tail4.head(3)["sessions"].mean())
            _img_latest_sess = int(_img_tail4.iloc[-1]["sessions"])
            _img_mo_lbl      = pd.to_datetime(_img_tail4.iloc[-1]["revenue_month"]).strftime("%b %Y")
            if _img_rolling_avg > 0:
                _img_pct_of_avg = round(100.0 * _img_latest_sess / _img_rolling_avg, 1)
                _img_drop_pct   = round(100.0 - _img_pct_of_avg, 1)
                if _img_pct_of_avg < _IMAGING_CRIT_PCT:
                    _img_alert = ("CRITICAL", _img_latest_sess, _img_rolling_avg, _img_pct_of_avg, _img_drop_pct, _img_mo_lbl)
                elif _img_pct_of_avg < _IMAGING_WATCH_PCT:
                    _img_alert = ("WATCH", _img_latest_sess, _img_rolling_avg, _img_pct_of_avg, _img_drop_pct, _img_mo_lbl)

    # ── Threshold rules — edit here to adjust alert sensitivity ─────────────
    # _READM_CRITICAL = 15  # READM_HIDDEN
    # _READM_WATCH    =  5  # READM_HIDDEN
    _DIALYSIS_IDLE  =  6    # months idle before surfacing
    _THEATRE_WATCH  = 85    # % completion — below = watch
    _THEATRE_CRIT   = 75    # % completion — below = critical

    # ── Derived signals ───────────────────────────────────────────────────────

    # READM_HIDDEN — readmission derived signals suppressed
    _readm_rising = False
    _mm_baseline  = None

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

    # ── Operational Pulse (KSH only — always-on domain health) ──────────────

    if _is_ksh:
        section_header("Operational Pulse", margin_top=8)

        # Compute domain-level status for 5 domains
        _KSH_WARDS = ["MEDICAL — MALE", "MEDICAL — FEMALE", "MATERNITY",
                      "PRIVATE / AMENITY", "PAEDIATRIC"]

        def _domain_status(*statuses):
            if "RED"   in statuses: return "RED"
            if "AMBER" in statuses: return "AMBER"
            return "GREEN"

        # CAPACITY: ward traffic + LOS
        _cap_statuses = []
        _cap_worst_msg = "All traffic and LOS metrics within range"
        for _w in _KSH_WARDS:
            _wkey = _w
            _adm_v = _ward_adm_latest(_wkey)
            if _adm_v is not None and _TRAFFIC_WATCH.get(_wkey):
                _s = _status_hi(_adm_v, _TRAFFIC_WATCH[_wkey],
                                  _TRAFFIC_CRIT.get(_wkey))
                _cap_statuses.append(_s)
                if _s in ("RED", "AMBER"):
                    _cap_worst_msg = f"{_w.title()} admissions: {_adm_v:.0f}/mo vs WATCH {_TRAFFIC_WATCH[_wkey]}"
            _los_v = _ward_los_vals(_wkey, 1)
            if _los_v and _LOS_WATCH.get(_wkey):
                _s = _status_hi(_los_v[-1], _LOS_WATCH[_wkey], _LOS_CRIT.get(_wkey))
                _cap_statuses.append(_s)
                if _s == "RED":
                    _cap_worst_msg = f"{_w.title()} median LOS: {_los_v[-1]:.1f}d vs WATCH {_LOS_WATCH[_wkey]}d"
        _cap_status = _domain_status(*_cap_statuses) if _cap_statuses else "GREEN"

        # PATIENT FLOW: Patient Request rate
        _pf_statuses = []
        _pf_worst_msg = "Patient Request rates within ward baselines"
        for _w in _KSH_WARDS:
            _pr_v = _ward_pr_vals(_w, 1)
            if _pr_v and _PR_WATCH.get(_w):
                _s = _status_hi(_pr_v[-1], _PR_WATCH[_w], _PR_CRIT.get(_w))
                _pf_statuses.append(_s)
                if _s in ("RED", "AMBER"):
                    _pf_worst_msg = f"{_w.title()} Patient Request: {_pr_v[-1]:.0f}% vs WATCH {_PR_WATCH[_w]}%"
        _pf_status = _domain_status(*_pf_statuses) if _pf_statuses else "GREEN"

        # READM_HIDDEN — readmission pulse card suppressed
        _rm_status    = "GREEN"
        _rm_worst_msg = "Readmission monitoring not active"

        # STAFFING: doctor workload
        _sf_statuses = []
        _sf_worst_msg = "Doctor workload within normal range"
        if _top_doc_pct > 0:
            _s = _status_hi(_top_doc_pct, _DOC_CONC_WATCH, _DOC_CONC_CRIT)
            _sf_statuses.append(_s)
            if _s in ("RED", "AMBER"):
                _dname = f"{_top_doc_name[0].upper()}.{_top_doc_name[1:].capitalize()}" if _top_doc_name else "—"
                _sf_worst_msg = f"{_dname}: {_top_doc_pct:.0f}% of all visits vs WATCH {_DOC_CONC_WATCH}%"
        if _burnout_alerts:
            _sf_statuses.append("AMBER")
            if _sf_worst_msg == "Doctor workload within normal range":
                _bn = _burnout_alerts[0]
                _sf_worst_msg = f"{_bn[0]}: {_bn[1]:.0f}% above personal avg for 2 months"
        _sf_status = _domain_status(*_sf_statuses) if _sf_statuses else "GREEN"

        # LAB: volume + abnormal rate — message tracks worst metric, not last non-green
        _lb_statuses = []
        _lb_vol_s   = "GREEN"
        _lb_abn_s   = "GREEN"
        if _lab_latest_visits is not None:
            _lb_vol_s = _status_lo(_lab_latest_visits, _LAB_VOL_WATCH, _LAB_VOL_CRIT)
            _lb_statuses.append(_lb_vol_s)
        if _lab_latest_abnorm is not None:
            _lb_abn_s = _status_hi(_lab_latest_abnorm, _LAB_ABNORM_WATCH, _LAB_ABNORM_CRIT)
            _lb_statuses.append(_lb_abn_s)
        _lb_status = _domain_status(*_lb_statuses) if _lb_statuses else "GREEN"
        _SRANK = {"RED": 2, "AMBER": 1, "GREEN": 0}
        if _SRANK[_lb_vol_s] >= _SRANK[_lb_abn_s] and _lb_vol_s != "GREEN":
            _lb_worst_msg = f"Lab visits: {_lab_latest_visits}/mo vs WATCH <{_LAB_VOL_WATCH}"
        elif _lb_abn_s != "GREEN":
            _lb_worst_msg = f"Abnormal rate: {_lab_latest_abnorm:.1f}% vs WATCH >{_LAB_ABNORM_WATCH}%"
        else:
            _lb_worst_msg = "Lab volume and abnormal rate within range"

        # THEATRE: last-month rate as primary signal; 3-month weighted average as context
        _th_status    = "GREEN"
        _th_worst_msg = "Theatre completion rate within range"
        if th_last_rate is not None:
            _th_3mo_ctx = f" (3-mo avg {th_comp_rate:.0f}%)" if th_comp_rate is not None else ""
            if th_last_rate < 75:
                _th_status    = "RED"
                _th_worst_msg = f"{th_last_lbl}: {th_last_rate:.0f}%{_th_3mo_ctx} — below 75% threshold"
            elif th_last_rate < 90:
                _th_status    = "AMBER"
                _th_worst_msg = f"{th_last_lbl}: {th_last_rate:.0f}%{_th_3mo_ctx} — monitor surgical throughput"
            else:
                _th_worst_msg = f"{th_last_lbl}: {th_last_rate:.0f}%{_th_3mo_ctx} — on target"

        # ── Operational Pulse — compact 6-domain health strip ────────────────
        _PULSE_DOMAINS = [
            ("fa-solid fa-bed-pulse",                   "Capacity",     _cap_status, _cap_worst_msg),
            ("fa-solid fa-person-walking-arrow-right",  "Patient Flow", _pf_status,  _pf_worst_msg),
            # ("fa-solid fa-clock-rotate-left",  "Readmissions", _rm_status,  _rm_worst_msg),  # READM_HIDDEN
            ("fa-solid fa-user-doctor",                 "Staffing",     _sf_status,  _sf_worst_msg),
            ("fa-solid fa-microscope",                  "Lab",          _lb_status,  _lb_worst_msg),
            ("fa-solid fa-syringe",                     "Theatre",      _th_status,  _th_worst_msg),
        ]
        _STATUS_BG = {"GREEN": "#F0FBF8", "AMBER": "#FFFBEB", "RED": "#FFF1F3"}
        _STATUS_BORDER = {"GREEN": "#0BB99F", "AMBER": "#D97706", "RED": "#E11D48"}
        _STATUS_LABEL = {"GREEN": "ALL CLEAR", "AMBER": "WATCH", "RED": "ALERT"}

        _pulse_cols = st.columns(5, gap="small")
        for _i, (_pc, (_icon, _domain, _dstatus, _dmsg)) in enumerate(
            zip(_pulse_cols, _PULSE_DOMAINS)
        ):
            _dc   = _STATUS_BORDER[_dstatus]
            _dbg  = _STATUS_BG[_dstatus]
            _de   = _STATUS_EMOJI[_dstatus]
            _dlbl = _STATUS_LABEL[_dstatus]
            _delay = f"{_i * 0.07:.2f}s"
            with _pc:
                st.markdown(
                    f'<div style="'
                    f'background:{_dbg};'
                    f'border:1px solid {_dc}40;'
                    f'border-top:4px solid {_dc};'
                    f'border-radius:10px;'
                    f'padding:20px 18px 16px;'
                    f'animation:fadeUp 0.4s ease {_delay} both;'
                    f'min-height:130px;'
                    f'box-shadow:0 2px 8px {_dc}18">'
                    # Domain row
                    f'<div style="display:flex;align-items:center;'
                    f'justify-content:space-between;margin-bottom:12px">'
                    f'<i class="{_icon}" style="font-size:26px;color:{_dc};opacity:0.9"></i>'
                    f'<span style="font-size:22px;line-height:1">{_de}</span>'
                    f'</div>'
                    f'<div style="font-size:13px;font-weight:800;color:#003467;'
                    f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px">'
                    f'{_domain}</div>'
                    # Status badge
                    f'<div style="margin-bottom:10px">'
                    f'<span style="'
                    f'background:{_dc};color:#fff;'
                    f'font-size:10px;font-weight:800;'
                    f'letter-spacing:1.2px;'
                    f'padding:3px 9px;border-radius:4px">'
                    f'{_dlbl}</span>'
                    f'</div>'
                    # Message
                    f'<div style="font-size:12px;color:#003467;'
                    f'line-height:1.55;opacity:0.85">'
                    f'{_dmsg}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        st.markdown(
            '<div style="font-size:11px;color:#6B8CAE;text-align:right;'
            'margin-top:6px;margin-bottom:20px">'
            'Full detail → Capacity &amp; Operations</div>',
            unsafe_allow_html=True,
        )

    # ── Visit summary cards (KSH only) ───────────────────────────────────────
    if _vs_cur is not None:
        _vs_outlook = _ema_next(_vs["total_visits"])
        _vc1, _vc2, _vc3, _vc4 = st.columns(4, gap="large")
        with _vc1:
            kpi_card("Total Visits", f"{_vs_cur:,}", "", COLORS["primary"])
        with _vc2:
            kpi_card("Inpatient Admissions", f"{_ip_cur:,}", "", COLORS["warning"])
        with _vc3:
            kpi_card("Outpatient Visits", f"{_op_cur:,}", "", COLORS["success"])
        with _vc4:
            if _vs_outlook is not None:
                kpi_card("Expected Next Month", f"~{int(round(_vs_outlook)):,}",
                         "visits · 3-month EMA", COLORS["muted"])
        st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    # ── Active Alerts — full width ────────────────────────────────────────────

    col_l = st.container()

    with col_l:
        section_header("Active Alerts")

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
        _notices = []

        # READM_HIDDEN — Rules 1 & 2 (Medical Male readmissions + facility-wide trend) suppressed
        # Rule 1 — Medical Male readmissions
        # if mm_rate > _READM_WATCH:
        #     _sev = "CRITICAL" if mm_rate > _READM_CRITICAL else "WATCH"
        #     ...

        # Rule 2 — Facility-wide trend rising
        # if _readm_rising and mm_rate <= _READM_WATCH:
        #     ...

        # ── Phase 13 rules (KSH only) — domain order: Capacity → Patient Flow → Staffing → Lab & Diagnostics → Theatre

        if _is_ksh:

            # READM_HIDDEN — Rules 5-9 (ward readmissions, all 5 wards) suppressed
            # Re-enable when clinical page is built. Data intact in rpt_readmissions gold table.
            # _WARD_READM_OTHERS = [...]
            # for _wk, _wlabel in _WARD_READM_OTHERS: ...

            # ── CAPACITY ──────────────────────────────────────────────────────

            # Rules 10-14 — Ward traffic volume (Inv 21)
            _WARD_TRAFFIC_ALL = [
                ("MEDICAL — FEMALE", "Medical Female"),
                ("PAEDIATRIC",       "Paediatric"),
                ("MEDICAL — MALE",   "Medical Male"),
                ("MATERNITY",        "Maternity"),
                ("PRIVATE / AMENITY","Private / Amenity"),
            ]
            for _wk, _wlabel in _WARD_TRAFFIC_ALL:
                _tw = _TRAFFIC_WATCH.get(_wk)
                _tc = _TRAFFIC_CRIT.get(_wk)
                if _tw is None:
                    continue
                _av2 = _ward_adm_prev2(_wk)
                if not _av2:
                    continue
                _latest_av = _av2[-1]
                if _two_consec(_av2, _tw):
                    _sev = "CRITICAL" if (_tc and _latest_av > _tc) else "WATCH"
                    _col = COLORS["danger"] if _sev == "CRITICAL" else COLORS["warning"]
                    _notice_card(
                        _sev,
                        f"{_wlabel} — High Volume",
                        f"{_latest_av:.0f} admissions/mo",
                        f"2 consecutive months above WATCH {_tw}/mo",
                        f"Sustained volume pressure in {_wlabel}. "
                        "Review staffing and bed allocation before capacity ceiling.",
                        _col,
                    )
                    _active += 1
                    _notices.append({"level": _sev,
                                     "title": f"{_wlabel} — High Volume",
                                     "metric": f"{_latest_av:.0f} adm/mo",
                                     "action": f"Review staffing and bed allocation — {_wlabel}"})

            # Rules 15-19 — LOS deviation (median-based, Inv 22)
            for _wk, _wlabel in _WARD_TRAFFIC_ALL:
                _lw = _LOS_WATCH.get(_wk)
                _lc = _LOS_CRIT.get(_wk)
                if _lw is None:
                    continue
                _lv = _ward_los_vals(_wk, 2)
                if not _lv:
                    continue
                _latest_lv = _lv[-1]
                if _two_consec(_lv, _lw):
                    _sev = "CRITICAL" if (_lc and _latest_lv > _lc) else "WATCH"
                    _col = COLORS["danger"] if _sev == "CRITICAL" else COLORS["warning"]
                    _notice_card(
                        _sev,
                        f"{_wlabel} — Extended LOS",
                        f"{_latest_lv:.1f}d median",
                        f"2 consecutive months above WATCH {_lw}d",
                        f"Median LOS rising in {_wlabel}. Investigate discharge delays "
                        "or acuity increase — not a single outlier (median, not avg).",
                        _col,
                    )
                    _active += 1
                    _notices.append({"level": _sev,
                                     "title": f"{_wlabel} — Extended LOS",
                                     "metric": f"{_latest_lv:.1f}d median",
                                     "action": f"Investigate discharge delays — {_wlabel}"})

            # Rule 29 — Ward Idle BTR/BTI (Inv 46)
            for _wi_ward, _wi_btr, _wi_btr_p25, _wi_bti, _wi_bti_p75, _wi_mo_lbl in _ward_idle_alerts:
                _notice_card(
                    "WATCH",
                    f"{_wi_ward} — Ward Idle",
                    f"BTR {_wi_btr:.2f} · BTI {_wi_bti:.0f}d",
                    f"BTR {_wi_btr_p25 - _wi_btr:.2f} below ward floor ({_wi_btr_p25:.2f}) · "
                    f"BTI {_wi_bti - _wi_bti_p75:.0f}d above ward ceiling ({_wi_bti_p75:.0f}d)",
                    f"{_wi_mo_lbl} — {_wi_ward}: {_wi_btr:.2f} admissions/bed (ward floor {_wi_btr_p25:.2f}) "
                    f"and beds idle avg {_wi_bti:.0f} days (ward ceiling {_wi_bti_p75:.0f}d). "
                    "Investigate before acting — if visit volume to this ward is also low, this is a demand gap; "
                    "if visits are normal but admissions are low, investigate the admissions process. "
                    "Flag to ward manager with visit volume context.",
                    COLORS["warning"],
                )
                _active += 1
                _notices.append({"level": "WATCH",
                                 "title": f"{_wi_ward} — Ward Idle",
                                 "metric": f"BTR {_wi_btr:.2f} · BTI {_wi_bti:.0f}d",
                                 "action": f"Flag to ward manager — {_wi_ward} below ward baseline"})

            # Rule 31 — BOR ward low occupancy (Inv 48)
            for _bw_ward, _bw_bor, _bw_p25, _bw_gap, _bw_mo_lbl in _bor_ward_alerts:
                _notice_card(
                    "WATCH",
                    f"{_bw_ward} — Low Occupancy",
                    f"BOR {_bw_bor:.1f}%",
                    f"{_bw_gap}pp below ward P25 floor ({_bw_p25:.1f}%) · 2 consecutive months",
                    f"{_bw_mo_lbl} — {_bw_ward} occupancy at {_bw_bor:.1f}% for 2 consecutive months "
                    f"(ward P25 floor {_bw_p25:.1f}%, gap {_bw_gap}pp). "
                    "Sustained low occupancy — flag to ward manager to review admission referral patterns.",
                    COLORS["warning"],
                )
                _active += 1
                _notices.append({"level": "WATCH",
                                 "title": f"{_bw_ward} — Low Occupancy",
                                 "metric": f"BOR {_bw_bor:.1f}%",
                                 "action": f"Flag to ward manager — BOR {_bw_gap}pp below ward P25 floor ({_bw_p25:.1f}%)"})

            # Rule 31b — BOR ward high occupancy (Inv 48)
            for _bh_ward, _bh_bor, _bh_sev, _bh_mo_lbl in _bor_high_alerts:
                _bh_thresh = _BOR_HIGH_CRIT if _bh_sev == "CRITICAL" else _BOR_HIGH_WATCH
                _bh_gap    = round(_bh_bor - _bh_thresh, 1)
                _bh_clr    = COLORS["danger"] if _bh_sev == "CRITICAL" else COLORS["warning"]
                _bh_dur    = "single month" if _bh_sev == "CRITICAL" else "2 consecutive months"
                _notice_card(
                    _bh_sev,
                    f"{_bh_ward} — High Occupancy",
                    f"BOR {_bh_bor:.1f}%",
                    f"{_bh_gap}pp above {_BOR_HIGH_WATCH:.0f}% MoH anchor · {_bh_dur}",
                    f"{_bh_mo_lbl} — {_bh_ward} at {_bh_bor:.1f}% occupancy ({_bh_gap}pp above "
                    f"{_bh_thresh:.0f}% threshold, {_bh_dur}). "
                    "Ward near capacity — investigate discharge clearance and assess whether "
                    "any elective admissions should be deferred. Flag to ward manager and facility management.",
                    _bh_clr,
                )
                _active += 1
                _notices.append({"level": _bh_sev,
                                 "title": f"{_bh_ward} — High Occupancy",
                                 "metric": f"BOR {_bh_bor:.1f}%",
                                 "action": f"Flag to ward manager — {_bh_ward} {_bh_gap}pp above "
                                           f"{_bh_thresh:.0f}% for {_bh_dur}"})

            # Rule 32 — Private ward revenue drop (Inv 49)
            if _revpab_alert is not None:
                _rv_latest, _rv_avg, _rv_drop, _rv_adm, _rv_mo_lbl = _revpab_alert
                _notice_card(
                    "WATCH",
                    "Private Wards — Revenue Drop",
                    f"KES {_rv_latest/1000:.0f}K · {_rv_adm} admissions",
                    f"{_rv_drop:.1f}% below 3-month rolling average (KES {_rv_avg/1000:.0f}K)",
                    f"{_rv_mo_lbl} — Private Female + Male combined revenue KES {_rv_latest/1000:.0f}K "
                    f"({_rv_drop:.1f}% below 3-month rolling avg of KES {_rv_avg/1000:.0f}K). "
                    f"{_rv_adm} admissions last month. Flag to finance lead — review private ward admission volume.",
                    COLORS["warning"],
                )
                _active += 1
                _notices.append({"level": "WATCH",
                                 "title": "Private Wards — Revenue Drop",
                                 "metric": f"KES {_rv_latest/1000:.0f}K ({_rv_drop:.1f}% below avg)",
                                 "action": f"Flag to finance lead — {_rv_drop:.1f}% below 3-month rolling avg"})

            # ── PATIENT FLOW ──────────────────────────────────────────────────

            # Rules 20-24 — Patient Request rate per ward (Inv 23)
            for _wk, _wlabel in _WARD_TRAFFIC_ALL:
                _pw = _PR_WATCH.get(_wk)
                _pc = _PR_CRIT.get(_wk)
                if _pw is None:
                    continue
                # Gate: min 10 admissions
                if _ward_pr_admissions(_wk) < 10:
                    continue
                _pv = _ward_pr_vals(_wk, 2)
                if not _pv:
                    continue
                _latest_pv = _pv[-1]
                if _two_consec(_pv, _pw):
                    _sev = "CRITICAL" if (_pc and _latest_pv > _pc) else "WATCH"
                    _col = COLORS["danger"] if _sev == "CRITICAL" else COLORS["warning"]
                    _notice_card(
                        _sev,
                        f"{_wlabel} — Patient Request Rate",
                        f"{_latest_pv:.0f}%",
                        f"2 consecutive months above WATCH {_pw}% (ward baseline)",
                        f"Patient Request rate elevated above {_wlabel} baseline. "
                        "This discharge pathway readmits at 50% — address before readmissions spike.",
                        _col,
                    )
                    _active += 1
                    _notices.append({"level": _sev,
                                     "title": f"{_wlabel} — Patient Request Rate",
                                     "metric": f"{_latest_pv:.0f}%",
                                     "action": f"Address Patient Request discharge drivers — {_wlabel}"})

            # Rule 30 — Admission TAT monthly deterioration (Inv 47)
            if _tat_latest_fast_pct is not None:
                _tat_is_crit      = _tat_latest_fast_pct < _TAT_CRIT
                _tat_is_watch     = _two_consec(_tat_fast_pcts, _TAT_WATCH, direction="below")
                _tat_p75_is_crit  = _tat_latest_p75 is not None and _tat_latest_p75 > _TAT_P75_CRIT
                _tat_p75_is_watch = (len(_tat_p75_vals) == 2 and
                                     _two_consec(_tat_p75_vals, _TAT_P75_WATCH, direction="above"))
                if _tat_is_crit or _tat_is_watch or _tat_p75_is_crit or _tat_p75_is_watch:
                    _tat_sev    = "CRITICAL" if (_tat_is_crit or _tat_p75_is_crit) else "WATCH"
                    _tat_clr    = COLORS["danger"] if _tat_sev == "CRITICAL" else COLORS["warning"]
                    _tat_mo_lbl = pd.to_datetime(_tat_latest_month).strftime("%b %Y")
                    _p75_str    = f" · p75 {_tat_latest_p75} min" if _tat_latest_p75 else ""
                    _breaches   = []
                    if _tat_is_crit:        _breaches.append(f"fast-track {_tat_latest_fast_pct:.1f}% < {_TAT_CRIT:.0f}% (single month)")
                    elif _tat_is_watch:     _breaches.append(f"fast-track {_tat_latest_fast_pct:.1f}% < {_TAT_WATCH:.0f}% × 2 months")
                    if _tat_p75_is_crit:    _breaches.append(f"p75 TAT {_tat_latest_p75} min > {int(_TAT_P75_CRIT)} min (single month)")
                    elif _tat_p75_is_watch: _breaches.append(f"p75 TAT {_tat_latest_p75} min > {int(_TAT_P75_WATCH)} min × 2 months")
                    _notice_card(
                        _tat_sev,
                        "Admission TAT Deterioration",
                        f"{_tat_latest_fast_pct:.1f}% fast-track · p50 {_tat_latest_p50} min{_p75_str}",
                        " · ".join(_breaches),
                        f"{_tat_mo_lbl} — {_tat_latest_fast_pct:.1f}% of admissions completed within 60 min. "
                        f"Median TAT {_tat_latest_p50} min{_p75_str}. "
                        "Flag to ops lead — review ED-to-ward handoff process.",
                        _tat_clr,
                    )
                    _active += 1
                    _notices.append({"level": _tat_sev,
                                     "title": "Admission TAT Deterioration",
                                     "metric": f"{_tat_latest_fast_pct:.1f}% fast-track · p50 {_tat_latest_p50} min{_p75_str}",
                                     "action": f"Flag to ops lead — {' · '.join(_breaches)}"})

            # ── STAFFING ──────────────────────────────────────────────────────

            # Rule 25 — Doctor workload: concentration risk (Inv 24)
            if _top_doc_pct > _DOC_CONC_WATCH:
                _sev = "CRITICAL" if _top_doc_pct > _DOC_CONC_CRIT else "WATCH"
                _col = COLORS["danger"] if _sev == "CRITICAL" else COLORS["warning"]
                _ddisp = f"{_top_doc_name[0].upper()}.{_top_doc_name[1:].capitalize()}" if _top_doc_name else "—"
                _notice_card(
                    _sev,
                    "Staffing — Doctor Concentration",
                    f"{_ddisp}: {_top_doc_pct:.0f}%",
                    f"Single doctor handling >{_DOC_CONC_WATCH}% of all evaluation visits",
                    f"{_ddisp} carries {_top_doc_pct:.0f}% of all outpatient evaluations this month "
                    f"({_top_doc_visits:,} visits). makinyi departure Dec 2025 created this concentration. "
                    "Risk: departure or absence disrupts outpatient capacity.",
                    _col,
                )
                _active += 1
                _notices.append({"level": _sev,
                                 "title": "Staffing — Doctor Concentration",
                                 "metric": f"{_ddisp}: {_top_doc_pct:.0f}% of visits",
                                 "action": "Flag to clinical lead — single-doctor dependency risk."})

            # Rule 26 — Doctor workload: individual burnout (Inv 24)
            for _bn in _burnout_alerts:
                _notice_card(
                    "WATCH",
                    f"Staffing — {_bn[0]} Burnout Signal",
                    f"{_bn[2]:,} visits",
                    f"{_bn[1]:.0f}% of personal 3-month avg ({_bn[3]:,}) for 2 consecutive months",
                    f"{_bn[0]} is sustaining >150% of their baseline for 2 months. "
                    "Burnout risk and quality risk — flag to clinical lead.",
                    COLORS["warning"],
                )
                _active += 1
                _notices.append({"level": "WATCH",
                                 "title": f"Staffing — {_bn[0]} Burnout Signal",
                                 "metric": f"{_bn[2]:,} visits ({_bn[1]:.0f}% of avg)",
                                 "action": f"Flag to clinical lead — {_bn[0]} volume unsustainable."})

            # Rule 33 — Physician workload (Inv 50)
            for _dr_name, _dr_vis, _dr_p90_v, _dr_mo_lbl in _doc_wl_alerts:
                _dr_excess = _dr_vis - _dr_p90_v
                _notice_card(
                    "WATCH",
                    f"Dr {_dr_name} — High Visit Load",
                    f"{_dr_vis} visits",
                    f"{_dr_excess} above P90 ({_dr_p90_v}) · 2 consecutive months",
                    f"{_dr_mo_lbl} — {_dr_name} recorded {_dr_vis} evaluation visits "
                    f"({_dr_excess} above personal P90 of {_dr_p90_v}). "
                    "Sustained for 2 consecutive months. "
                    "Flag to ops lead — sustained high load may affect evaluation quality.",
                    COLORS["warning"],
                )
                _active += 1
                _notices.append({"level": "WATCH",
                                 "title": f"Dr {_dr_name} — High Visit Load",
                                 "metric": f"{_dr_vis} visits",
                                 "action": f"Flag to ops lead — {_dr_name} {_dr_excess} above P90 for 2 months"})

            # ── LAB & DIAGNOSTICS ─────────────────────────────────────────────

            # Rule 27 — Lab volume drop (Inv 25b)
            if _lab_latest_visits is not None:
                _lv2 = _lab_visits_vals(2)
                if _two_consec(_lv2, _LAB_VOL_WATCH, direction="below") or _lab_latest_visits < _LAB_VOL_CRIT:
                    _sev = "CRITICAL" if _lab_latest_visits < _LAB_VOL_CRIT else "WATCH"
                    _col = COLORS["danger"] if _sev == "CRITICAL" else COLORS["warning"]
                    _lb_mo = pd.to_datetime(_lab_latest_month).strftime("%b %Y") if _lab_latest_month else "—"
                    _notice_card(
                        _sev,
                        "Lab — Volume Drop",
                        f"{_lab_latest_visits} visits",
                        f"{_lb_mo} · WATCH <{_LAB_VOL_WATCH} · CRITICAL <{_LAB_VOL_CRIT}",
                        "Sustained lab volume drop signals equipment downtime or staffing failure. "
                        "Oct 2025 dip (371 visits) was facility-wide — undetected for 1 month. "
                        "Confirm lab capacity with department head.",
                        _col,
                    )
                    _active += 1
                    _notices.append({"level": _sev,
                                     "title": "Lab — Volume Drop",
                                     "metric": f"{_lab_latest_visits} visits/mo",
                                     "action": "Confirm lab capacity with department head"})

            # Rule 28 — Lab abnormal rate spike (Inv 25b)
            if _lab_latest_abnorm is not None:
                _la2 = _lab_abnorm_vals(2)
                if _two_consec(_la2, _LAB_ABNORM_WATCH):
                    _sev = "CRITICAL" if _lab_latest_abnorm > _LAB_ABNORM_CRIT else "WATCH"
                    _col = COLORS["danger"] if _sev == "CRITICAL" else COLORS["warning"]
                    _lb_mo = pd.to_datetime(_lab_latest_month).strftime("%b %Y") if _lab_latest_month else "—"
                    _notice_card(
                        _sev,
                        "Lab — Abnormal Rate Elevated",
                        f"{_lab_latest_abnorm:.1f}%",
                        f"{_lb_mo} · WATCH >{_LAB_ABNORM_WATCH}% · CRITICAL >{_LAB_ABNORM_CRIT}%",
                        "Rising abnormal flag rate signals population acuity increase. "
                        "Cross-reference with ward admissions and readmission trends.",
                        _col,
                    )
                    _active += 1
                    _notices.append({"level": _sev,
                                     "title": "Lab — Abnormal Rate Elevated",
                                     "metric": f"{_lab_latest_abnorm:.1f}%",
                                     "action": "Cross-reference with ward admissions trends"})

            # Rule 34 — CD12 Critical Creatinine Non-Admission (Inv 51)
            if _cd12_alert is not None:
                _c12_sev, _c12_rate, _c12_total, _c12_not_adm, _c12_mo_lbl = _cd12_alert
                _c12_thresh = _CD12_CRIT if _c12_sev == "CRITICAL" else _CD12_WATCH
                _c12_gap    = round(_c12_rate - _c12_thresh, 1)
                _c12_clr    = COLORS["danger"] if _c12_sev == "CRITICAL" else COLORS["warning"]
                _notice_card(
                    _c12_sev,
                    "Renal — Critical Creatinine Non-Admission",
                    f"{_c12_rate:.1f}% not admitted · {_c12_total} critical events",
                    f"{_c12_gap}pp above {'critical' if _c12_sev == 'CRITICAL' else 'watch'} threshold "
                    f"({_c12_thresh:.0f}%) · {_c12_not_adm} of {_c12_total} not admitted",
                    f"{_c12_mo_lbl} — {_c12_rate:.1f}% of critical creatinine results not followed by admission "
                    f"({_c12_not_adm} of {_c12_total} patients, {_c12_gap}pp above {_c12_thresh:.0f}% threshold). "
                    "Not-admitted could reflect AMA, affordability, or outward referral — investigate the routing "
                    "pathway before concluding operational failure. Flag to clinical lead for case-level audit.",
                    _c12_clr,
                )
                _active += 1
                _notices.append({"level": _c12_sev,
                                 "title": "Renal — Critical Creatinine Non-Admission",
                                 "metric": f"{_c12_rate:.1f}% not admitted ({_c12_total} events)",
                                 "action": f"Flag to clinical lead — {_c12_gap}pp above {_c12_thresh:.0f}% threshold"})

            # Rule 4 — Dialysis idle (KSH only)
            if months_idle is not None and months_idle >= _DIALYSIS_IDLE:
                _kes_line = (
                    f"Est. {fmt_kes(_dial_kes_low)}–{fmt_kes(_dial_kes_high)} foregone at historical session rate"
                    if _dial_kes_low else "Insufficient session history to estimate foregone revenue"
                )
                _notice_card(
                    "WATCH",
                    "Dialysis — Equipment Idle",
                    f"{months_idle} months",
                    _kes_line,
                    "Last session Apr 2025 · 22 enrolled, 5 ever scheduled, 3 sessions delivered (9% slot fulfilment). "
                    "77 of 78 admitted critical creatinine patients never enrolled. Process gap — not an equipment gap. Clinical lead review needed.",
                    COLORS["warning"],
                )
                _active += 1
                _notices.append({"level": "WATCH", "title": "Dialysis — Equipment Idle",
                                 "metric": f"{months_idle} months",
                                 "action": "Programme gap — 77 of 78 admitted critical creatinine patients never enrolled in dialysis. Clinical lead review needed."})

            # Rule 35 — CT Imaging Volume Drop (Inv 52)
            if _img_alert is not None:
                _img_sev, _img_sess, _img_avg, _img_pct, _img_drop, _img_mo_lbl = _img_alert
                _img_thresh = _IMAGING_CRIT_PCT if _img_sev == "CRITICAL" else _IMAGING_WATCH_PCT
                _img_clr    = COLORS["danger"] if _img_sev == "CRITICAL" else COLORS["warning"]
                _img_avg_r  = round(_img_avg)
                _notice_card(
                    _img_sev,
                    "Imaging — CT Volume Drop",
                    f"{_img_sess} CT sessions · {_img_pct:.1f}% of 3-month avg",
                    f"{_img_drop:.1f}% below 3-month rolling average ({_img_avg_r} sessions)",
                    f"{_img_mo_lbl} — {_img_sess} CT/Angio sessions recorded "
                    f"({_img_pct:.1f}% of 3-month avg of {_img_avg_r}, drop of {_img_drop:.1f}%). "
                    f"Below {'critical' if _img_sev == 'CRITICAL' else 'watch'} threshold of {_img_thresh:.0f}% of avg. "
                    "Investigate cause before acting — could be equipment downtime, scheduling backlog, "
                    "or referring doctor absence. Flag to imaging lead.",
                    _img_clr,
                )
                _active += 1
                _notices.append({"level": _img_sev,
                                 "title": "Imaging — CT Volume Drop",
                                 "metric": f"{_img_sess} sessions ({_img_pct:.1f}% of avg)",
                                 "action": f"Flag to imaging lead — {_img_drop:.1f}% below "
                                           f"{'critical' if _img_sev == 'CRITICAL' else 'watch'} threshold"})

            # ── THEATRE ───────────────────────────────────────────────────────

            # Rule 3 — Theatre completion below target (KSH only)
            if th_comp_rate is not None and th_comp_rate < _THEATRE_WATCH:
                _sev = "CRITICAL" if th_comp_rate < _THEATRE_CRIT else "WATCH"
                _col = COLORS["danger"] if th_comp_rate < _THEATRE_CRIT else COLORS["warning"]
                _gap_line = (
                    f"Est. {fmt_kes(_th_rev_gap)}/month in unbilled capacity"
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
                _notices.append({"level": _sev, "title": "Theatre — Completion Below Target",
                                 "metric": f"{th_comp_rate:.0f}%",
                                 "action": "Check cancellation and no-show rates on Capacity & Ops"})

        st.session_state["active_notices"] = _notices
        write_current_notices(FAC_DISPLAY.get(facility, facility), _notices)

        # All-clear state — shown when nothing crosses a threshold
        if _active == 0:
            _latest_mo = readm_fac["ADMISSION_MONTH"].max().strftime("%b %Y") if len(readm_fac) else "—"
            st.markdown(
                f'<div style="background:#F4F8FC;border-radius:8px;padding:16px 18px;'
                f'color:#6B8CAE;font-size:12px;line-height:1.8">'
                f'<span style="font-weight:700;color:#0BB99F">✓ No active alerts</span><br>'
                f'All monitored indicators within range.<br>'
                f'<span style="font-size:10px">Data as at {_latest_mo}</span>'
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

    c1, c2, c3, c4 = st.columns(4)
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
        max_days_val = f"{int(max_days_row['AVG_DAYS_OUTSTANDING'])} days" if max_days_row is not None else "—"
        max_days_ins = f"{max_days_row['INSURER']}" if max_days_row is not None else ""
        kpi_card("Longest Outstanding", max_days_val, max_days_ins, COLORS["warning"])
    with c4:
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
                    marker_color=colors)
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
                    _add_rolling_mean(fig, ksh_trend["INVOICE_MONTH"],
                                      ksh_trend["DISPATCH_RATE_PCT"],
                                      name="3-mo avg", color=COLORS["muted"])
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

        # ── Section 1: Per-insurer recovery tool (KSH only) ──────────────────
        if facility == "KISUMU_CLEAN" and len(rec_df):
            section_header("Per-Insurer Recovery — What Does Restoring Dispatch Actually Unlock?")

            ksh_rec = (
                rec_df[rec_df["FACILITY"] == "KISUMU_CLEAN"]
                .pipe(lambda d: d[~d["INSURER"].str.contains("AAR", case=False, na=False)])
                .sort_values("OUTSTANDING_KES", ascending=False)
                .copy()
            )

            if len(ksh_rec):
                # Age-tier fractions: split ksh_trend cliff months into 0-90 / 90-180 / 180+ day bands
                today_dt = pd.Timestamp.today().normalize()
                kt_cliff = ksh_trend[
                    ksh_trend["INVOICE_MONTH"] >= pd.Timestamp(KSH_DISPATCH_CLIFF)
                ].copy()

                if len(kt_cliff) and kt_cliff["TOTAL_OUTSTANDING"].sum() > 0:
                    kt_cliff["days_old"] = (today_dt - kt_cliff["INVOICE_MONTH"]).dt.days
                    kt_total      = kt_cliff["TOTAL_OUTSTANDING"].sum()
                    forfeit_frac  = (
                        kt_cliff[kt_cliff["days_old"] > 180]["TOTAL_OUTSTANDING"].sum() / kt_total
                    )
                    appeals_frac  = (
                        kt_cliff[
                            (kt_cliff["days_old"] >= 90) & (kt_cliff["days_old"] <= 180)
                        ]["TOTAL_OUTSTANDING"].sum() / kt_total
                    )
                else:
                    forfeit_frac = 0.45
                    appeals_frac = 0.35

                insurer_options  = ksh_rec["INSURER"].tolist()
                selected_insurer = st.selectbox(
                    "Select insurer to model recovery",
                    insurer_options, index=0,
                    key="t3_insurer_select",
                )

                ins_row        = ksh_rec[ksh_rec["INSURER"] == selected_insurer].iloc[0]
                total_outs     = float(ins_row["OUTSTANDING_KES"])
                within_window  = float(ins_row["EXPECTED_RECOVERABLE_KES"])
                past_window    = float(ins_row["OUTSTANDING_90PLUS"])
                ins_appeals    = past_window * appeals_frac
                ins_forfeiture = past_window * forfeit_frac

                t3_c1, t3_c2, t3_c3, t3_c4 = st.columns(4)
                with t3_c1:
                    kpi_card("Total Outstanding", fmt_kes(total_outs),
                             f"{int(ins_row['INVOICES'])} invoices · {ins_row['DISPATCH_RATE_PCT']:.1f}% dispatched",
                             COLORS["danger"], icon="⚠")
                with t3_c2:
                    kpi_card("Within SHA Window", fmt_kes(within_window),
                             "0–90 days · routine dispatch", COLORS["success"])
                with t3_c3:
                    kpi_card("SHA Appeals Zone", fmt_kes(ins_appeals),
                             "~90–180 days · formal appeals required", COLORS["warning"])
                with t3_c4:
                    kpi_card("Forfeiture Risk", fmt_kes(ins_forfeiture),
                             "~180+ days · SHA hard deadline may apply", COLORS["danger"], icon="⚠")

                st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)

                rec_pct = st.slider(
                    f"Collection efficiency on {selected_insurer}",
                    min_value=10, max_value=80, value=60, step=5, format="%d%%",
                    key="t3_rec_pct",
                )
                # Appeals at half efficiency: formal process with uncertain outcome
                projected_rec = (within_window + ins_appeals * 0.5) * (rec_pct / 100)

                pr_c1, pr_c2 = st.columns(2)
                with pr_c1:
                    kpi_card("Projected Recovery", fmt_kes(projected_rec),
                             f"{rec_pct}% on in-window · {rec_pct // 2}% on appeals · forfeiture excluded",
                             COLORS["success"], icon="✓")
                with pr_c2:
                    kpi_card("Forfeiture Exposure", fmt_kes(ins_forfeiture),
                             "Recoverable only via SHA formal dispute — not routine operations",
                             COLORS["danger"])

                dq_note("Age-tier split estimated from monthly ksh_trend population-level totals. "
                        "Individual insurer aging may vary. "
                        "SHA 90-day rule: formal appeals required past deadline; forfeiture risk after 180 days.")

        elif len(rec_df):
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

            try:
                _p6_ksh = st.session_state.get("p6_ksh") or {}
                _fc3 = _p6_ksh.get("forecast", []) if isinstance(_p6_ksh, dict) else []
                if _fc3:
                    forecast_adm = int(round(sum(r["point"] for r in _fc3)))
                    _fcast_src   = f"Prophet · {len(_fc3)}-month sum"
                else:
                    _ct3 = _build_forecast_contract(_FORECAST_CACHE)
                    forecast_adm = int(round(sum(r["point"] for r in _ct3["forecast"])))
                    _fcast_src   = f"Prophet · {len(_ct3['forecast'])}-month sum"
            except Exception:
                forecast_adm = 364
                _fcast_src   = "Holt trend estimate (Prophet cache not ready)"
            insured_adm  = int(forecast_adm * insured_share / 100)
            new_backlog  = insured_adm * avg_claim_val

            adm_kc1, adm_kc2, adm_kc3 = st.columns(3)
            with adm_kc1:
                kpi_card("Forecast Admissions", f"{forecast_adm:,}",
                         _fcast_src, COLORS["primary"])
            with adm_kc2:
                kpi_card("Insured Admissions", f"{insured_adm:,}",
                         f"At {insured_share}% insured share", COLORS["warning"])
            with adm_kc3:
                kpi_card("New Backlog If Dispatch Stays Down", fmt_kes(new_backlog),
                         "Joins existing backlog · starts new SHA 90-day clock per admission",
                         COLORS["danger"], icon="⚠")

            dq_note(
                f"Forecast: {_fcast_src}. "
                "Avg claim default: SHA outstanding ÷ SHA invoices from G7. "
                "Adjust sliders for different clinical profiles."
            )

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

    # ── Executive Recommendation ──────────────────────────────────────────────



# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — How We're Using What We Have
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Capacity & Operations":

    if not st.session_state.p3 or st.session_state.p3.get("_fac") != fac_key:
        with st.spinner("Loading…"):
            _is_ksh_p3 = (fac_key == "KISUMU_CLEAN")
            st.session_state.p3 = {
                "_fac":        fac_key,
                "th_trend":    q_theatre_trend(),
                "beds_revpab":   q_beds_revpab(facility),
                "beds_los":      q_beds_los(facility),
                "beds_monthly":  q_beds_monthly() if _is_ksh_p3 else pd.DataFrame(),
                "dialysis":    q_dialysis_trend(facility),
                "dialysis_ops": q_dialysis_ops_monthly() if _is_ksh_p3 else pd.DataFrame(),
                "specialty":   q_specialty_admissions(),
                "imaging":     q_imaging_trend(facility),
                # Phase 13 ward intelligence (KSH only)
                "ward_adm":    q_ward_admissions_monthly(facility) if _is_ksh_p3 else pd.DataFrame(),
                "ward_los":    q_ward_los_monthly(facility)        if _is_ksh_p3 else pd.DataFrame(),
                "ward_dc":     q_ward_discharge_monthly(facility)  if _is_ksh_p3 else pd.DataFrame(),
                "ward_readm":  pd.DataFrame(),  # READM_HIDDEN — q_readmission_ward_trend suppressed
                "doctor_wl":   q_doctor_workload_monthly()         if _is_ksh_p3 else pd.DataFrame(),
                "lab":         q_lab_monthly()                     if _is_ksh_p3 else pd.DataFrame(),
                "visit_sum":   q_visit_summary()                   if _is_ksh_p3 else pd.DataFrame(),
                "cd12_rate":   q_cd12_monthly_rate()               if _is_ksh_p3 else pd.DataFrame(),
                "doctor_conv": q_doctor_conversion_monthly()       if _is_ksh_p3 else pd.DataFrame(),
                "peak_bk":     q_peak_breakdown()                  if _is_ksh_p3 else pd.DataFrame(),
                "btr_bti":     q_btr_bti_monthly()           if _is_ksh_p3 else pd.DataFrame(),
                "adm_tat":     q_admission_tat_bimodal()     if _is_ksh_p3 else pd.DataFrame(),
                "th_emer_tat": q_theatre_emergency_tat()    if _is_ksh_p3 else pd.DataFrame(),
            }

    P = st.session_state.p3
    th_trend  = _filter_epoch(P["th_trend"], "SESSION_MONTH")
    beds_r    = P["beds_revpab"]
    beds_l    = P["beds_los"]
    dialysis  = _filter_epoch(P["dialysis"], "SESSION_MONTH")
    _dial_ops_raw = P.get("dialysis_ops", pd.DataFrame()).copy()
    fac_dialysis_ops = (
        _dial_ops_raw[~_dial_ops_raw["IS_PARTIAL_MONTH"]]
        if len(_dial_ops_raw) and "IS_PARTIAL_MONTH" in _dial_ops_raw.columns
        else _dial_ops_raw
    )
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

    if facility == "KISUMU_CLEAN":
        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("Theatre Completion", f"{th_recent_rate:.1f}%",
                     f"Trailing 3 months · all-time avg: {th_overall_rate:.1f}% {th_dot}",
                     th_rate_color)
        with c2:
            kpi_card("Monthly Theatre Revenue", fmt_kes(th_monthly_rev),
                     "Trailing 3-month avg", COLORS["success"])
        with c3:
            kpi_card("Top Ward RevPAB", top_revpab_val, top_revpab_label, COLORS["warning"])
    else:
        c1, c2, c3, c4 = st.columns(4)
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

    _is_ksh_p3 = (facility == "KISUMU_CLEAN")
    st.markdown("""
<style>
[data-testid="stTabs"] [role="tablist"] button:nth-child(1)::before{
    font-family:"Font Awesome 6 Free";font-weight:900;
    content:"\f48e\00a0\00a0";color:#0072CE}
[data-testid="stTabs"] [role="tablist"] button:nth-child(2)::before{
    font-family:"Font Awesome 6 Free";font-weight:900;
    content:"\f236\00a0\00a0";color:#0072CE}
[data-testid="stTabs"] [role="tablist"] button:nth-child(3)::before{
    font-family:"Font Awesome 6 Free";font-weight:900;
    content:"\f610\00a0\00a0";color:#0072CE}
[data-testid="stTabs"] [role="tablist"] button:nth-child(4)::before{
    font-family:"Font Awesome 6 Free";font-weight:900;
    content:"\f0f0\00a0\00a0";color:#0072CE}
</style>
""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Theatre",
        "Beds & Wards",
        "Lab & Diagnostics",
        "Staffing",
    ])

    # ── Tab 1: Theatre ────────────────────────────────────────────────────────

    with tab1:
        if facility == "TENRI":
            st.info("Theatre analytics are KSH-specific — not applicable for TENRI.")
        else:
            col_l, col_r = st.columns(2, gap="large")

            with col_l:
                _th_direction = ("Declining" if th_recent_rate < th_overall_rate - 3
                                 else "Improving" if th_recent_rate > th_overall_rate + 3
                                 else "Stable")
                section_header(f"Theatre Completion {_th_direction} — {th_recent_rate:.0f}% Recent vs {th_overall_rate:.0f}% All-Time Avg")
                if len(th_trend):
                    _th_plot = th_trend[th_trend["SESSION_MONTH"] >= "2024-09-01"].copy()
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_bar(
                        x=_th_plot["SESSION_MONTH"], y=_th_plot["TOTAL_SESSIONS"],
                        name="Sessions booked",
                        marker_color=COLORS["muted"], opacity=0.30,
                        hovertemplate="%{x|%b %Y}: %{y} sessions booked<extra></extra>",
                        secondary_y=True)
                    fig.add_scatter(
                        x=_th_plot["SESSION_MONTH"], y=_th_plot["COMPLETION_RATE_PCT"],
                        mode="lines+markers", name="Completion %",
                        line=dict(color=COLORS["primary"], width=2), marker=dict(size=5),
                        hovertemplate="%{x|%b %Y}: %{y:.1f}% completed<extra></extra>",
                        secondary_y=False)
                    _add_data_end_line(fig, "2025-07-01", "Jul drop")
                    _add_data_end_line(fig, "2025-10-01", "Oct drop")
                    fig.update_layout(**cl(
                        height=360,
                        legend=dict(orientation="h", y=1.08),
                        margin=dict(l=0, r=50, t=10, b=30),
                    ))
                    fig.update_yaxes(title_text="Completion %", range=[0, 110],
                                     ticksuffix="%", secondary_y=False)
                    fig.update_yaxes(title_text="Sessions", secondary_y=True,
                                     showgrid=False, rangemode="tozero")
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with col_r:
                if len(th_trend):
                    _pk = th_trend.loc[th_trend["TOTAL_REVENUE"].idxmax()]
                    _pk_lbl = (f"{fmt_kes(float(_pk['TOTAL_REVENUE']))} Peak "
                               f"({pd.Timestamp(_pk['SESSION_MONTH']).strftime('%b %Y')})")
                    _rev_recent3 = float(th_trend.nlargest(3, "SESSION_MONTH")["TOTAL_REVENUE"].mean()) if len(th_trend) >= 3 else 0
                    _rev_prior3  = float(th_trend.nlargest(6, "SESSION_MONTH").iloc[3:]["TOTAL_REVENUE"].mean()) if len(th_trend) >= 6 else _rev_recent3
                    _rev_dir     = ("Trending Down" if _rev_recent3 < _rev_prior3 * 0.95
                                    else "Trending Up" if _rev_recent3 > _rev_prior3 * 1.05
                                    else "Stable")
                    section_header(f"Monthly Theatre Revenue — {_pk_lbl}, {_rev_dir}")
                else:
                    section_header("Monthly Theatre Revenue — KSH")
                if len(th_trend):
                    _th_rev_plot = th_trend[th_trend["SESSION_MONTH"] >= "2024-09-01"].copy()
                    fig = go.Figure()
                    fig.add_bar(
                        x=_th_rev_plot["SESSION_MONTH"], y=_th_rev_plot["TOTAL_REVENUE"],
                        name="Revenue",
                        marker_color=COLORS["success"], opacity=0.75,
                        hovertemplate="%{x|%b %Y}: %{customdata}<extra></extra>",
                        customdata=_th_rev_plot["TOTAL_REVENUE"].apply(fmt_kes),
                    )
                    _add_regression(fig, _th_rev_plot["SESSION_MONTH"],
                                    _th_rev_plot["TOTAL_REVENUE"], name="Trend",
                                    color=COLORS["warning"])
                    fig.update_layout(**cl(height=360, yaxis_title="KES Revenue",
                                           legend=dict(orientation="h", y=1.08)))
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # ── Completion loss KPI (full-width) ─────────────────────────────────
            if len(th_trend):
                _th_cl = th_trend.copy()
                _th_cl.columns = [c.upper() for c in _th_cl.columns]
                _th_cl = _th_cl[_th_cl["TOTAL_SESSIONS"] > 0].sort_values("SESSION_MONTH")
                if len(_th_cl) and "COMPLETED_SESSIONS" in _th_cl.columns:
                    _th_lm       = _th_cl.iloc[-1]
                    _th_missed   = int(_th_lm["TOTAL_SESSIONS"]) - int(_th_lm["COMPLETED_SESSIONS"])
                    _th_avg_r    = float(_th_lm["TOTAL_REVENUE"]) / max(int(_th_lm["COMPLETED_SESSIONS"]), 1)
                    _th_miss_kes = _th_missed * _th_avg_r
                    _th_lm_lbl   = pd.Timestamp(_th_lm["SESSION_MONTH"]).strftime("%b %Y")
                    _th_below85  = int((_th_cl["COMPLETION_RATE_PCT"] < 85).sum())
                    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
                    _cl1, _cl2 = st.columns(2, gap="large")
                    with _cl1:
                        kpi_card(
                            f"Revenue Missed — {_th_lm_lbl}",
                            fmt_kes(_th_miss_kes),
                            f"{_th_missed} incomplete sessions × {fmt_kes(_th_avg_r)} avg/session",
                            COLORS["danger"], icon="⚠",
                        )
                    with _cl2:
                        info_card(
                            f"At 100% completion, {_th_missed} additional sessions this month would have "
                            f"generated <b>{fmt_kes(_th_miss_kes)}</b>. "
                            f"Completion fell below 85% in <b>{_th_below85}</b> of the last "
                            f"{len(_th_cl)} months — the gap is structural, not a one-off."
                        )

            # ── Payer mix (full width) — colored line, cliff auto-detected ────────
            _th_cols = {c.upper() for c in th_trend.columns}
            if (len(th_trend)
                    and "INSURED_REVENUE" in _th_cols
                    and "CASH_REVENUE" in _th_cols):
                _pm = th_trend[th_trend["SESSION_MONTH"] >= "2024-09-01"].copy()
                _pm.columns = [c.upper() for c in _pm.columns]
                _pm["_total"] = _pm["INSURED_REVENUE"] + _pm["CASH_REVENUE"]
                _pm["_ins_pct"] = (
                    100 * _pm["INSURED_REVENUE"]
                    / _pm["_total"].replace(0, float("nan"))
                ).round(1)
                _pm = _pm.dropna(subset=["_ins_pct"]).reset_index(drop=True)

                # Cliff = largest single-month drop ≥ 30pp (fully dynamic — no hardcoded date)
                _drops = _pm["_ins_pct"].shift(1) - _pm["_ins_pct"]
                _cliff_idx = (
                    int(_drops.idxmax())
                    if len(_pm) > 1 and float(_drops.max()) >= 30.0
                    else None
                )
                if _cliff_idx == 0:
                    _cliff_idx = None  # no baseline to compare against

                _curr_pct = float(_pm["_ins_pct"].iloc[-1])
                _norm_pct = (
                    float(_pm["_ins_pct"].iloc[:_cliff_idx].median())
                    if _cliff_idx else float(_pm["_ins_pct"].median())
                )
                _cliff_lbl = (
                    pd.Timestamp(_pm["SESSION_MONTH"].iloc[_cliff_idx]).strftime("%b %Y")
                    if _cliff_idx is not None else None
                )

                _hdr = (
                    f"Theatre Payer Mix — Insured {_curr_pct:.0f}% (from {_norm_pct:.0f}% baseline)"
                    if _cliff_idx is not None else
                    f"Theatre Payer Mix — Insured Revenue {_curr_pct:.0f}%"
                )
                st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
                section_header(_hdr)

                fig_pm = go.Figure()
                if _cliff_idx is not None:
                    _seg_n = _pm.iloc[:_cliff_idx]       # normal period (excludes cliff month)
                    _seg_a = _pm.iloc[_cliff_idx - 1:]   # anomaly (overlap 1 pt for line continuity)
                    fig_pm.add_scatter(
                        x=_seg_n["SESSION_MONTH"], y=_seg_n["_ins_pct"],
                        mode="lines+markers",
                        name=f"Normal billing (~{_norm_pct:.0f}%)",
                        line=dict(color=COLORS["primary"], width=2.5),
                        marker=dict(size=6),
                        hovertemplate="%{x|%b %Y}: %{y:.1f}% insured<extra></extra>",
                    )
                    fig_pm.add_scatter(
                        x=_seg_a["SESSION_MONTH"], y=_seg_a["_ins_pct"],
                        mode="lines+markers",
                        name=f"Billing pattern change ({_cliff_lbl})",
                        line=dict(color=COLORS["danger"], width=2.5),
                        marker=dict(size=6),
                        hovertemplate="%{x|%b %Y}: %{y:.1f}% insured<extra></extra>",
                    )
                    _cliff_date_str = str(_pm["SESSION_MONTH"].iloc[_cliff_idx])[:10]
                    _add_data_end_line(fig_pm, _cliff_date_str, _cliff_lbl)
                else:
                    fig_pm.add_scatter(
                        x=_pm["SESSION_MONTH"], y=_pm["_ins_pct"],
                        mode="lines+markers",
                        name="Insured revenue %",
                        line=dict(color=COLORS["primary"], width=2.5),
                        marker=dict(size=6),
                        hovertemplate="%{x|%b %Y}: %{y:.1f}% insured<extra></extra>",
                    )
                fig_pm.update_layout(**cl(
                    height=280,
                    legend=dict(orientation="h", y=1.12),
                    yaxis=dict(title="Insured revenue %", range=[-5, 110],
                               ticksuffix="%"),
                    margin=dict(l=0, r=50, t=10, b=30),
                ))
                st.plotly_chart(fig_pm, use_container_width=True,
                                config={"displayModeBar": False})
                if _cliff_idx is not None:
                    dq_note(
                        f"Insured revenue share dropped from ~{_norm_pct:.0f}% to "
                        f"{_curr_pct:.1f}% from {_cliff_lbl}. "
                        "Insurers still listed on invoices — likely a billing workflow change. "
                        "Escalate to KSH finance team to confirm cause before acting on this signal."
                    )


            # ── Emergency TAT distribution + day-of-week analysis (KSH only) ─
            _tat_df = P.get("th_emer_tat", pd.DataFrame())
            if facility == "KISUMU_CLEAN" and len(_tat_df):
                _tat_df.columns = [c.upper() for c in _tat_df.columns]
                _lags = _tat_df["BOOKING_TO_START_MIN"].values.astype(float)
                _total_n  = len(_lags)
                _median_h = round(float(np.median(_lags)) / 60, 1)
                _over_24  = int((_lags > 1440).sum())
                _over_24_pct = round(100 * _over_24 / _total_n, 1)

                _BIN_DEF = [
                    ("0–2h",   0,    120,  COLORS["success"]),
                    ("2–6h",   120,  360,  COLORS["primary"]),
                    ("6–12h",  360,  720,  COLORS["warning"]),
                    ("12–24h", 720,  1440, "#F97316"),
                    (">24h",   1440, 1e9,  COLORS["danger"]),
                ]
                _bin_counts = [
                    (lbl, int(((_lags > lo) & (_lags <= hi)).sum()), col)
                    for lbl, lo, hi, col in _BIN_DEF
                ]

                _hdr = (
                    f"Emergency Booking-to-Theatre — Median {_median_h}h  "
                    f"({_over_24} of {_total_n} cases, {_over_24_pct}% waited >24h)"
                )
                st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
                section_header(_hdr)

                # Option A: single stacked horizontal bar — proportion-of-whole
                fig_wf = go.Figure()
                for lbl, cnt, col in _bin_counts:
                    _seg_pct = round(100 * cnt / _total_n, 1)
                    fig_wf.add_trace(go.Bar(
                        name=f"{lbl} — {cnt} cases",
                        x=[cnt],
                        y=["Emergency TAT"],
                        orientation="h",
                        marker_color=col,
                        text=[f"<b>{lbl}</b><br>{cnt}"],
                        textposition="inside",
                        insidetextanchor="middle",
                        constraintext="inside",
                        hovertemplate=(
                            f"<b>{lbl}</b><br>{cnt} cases ({_seg_pct}%)<extra></extra>"
                        ),
                    ))
                fig_wf.update_layout(**cl(
                    barmode="stack",
                    height=120,
                    legend=dict(orientation="h", y=-0.6, x=0),
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    margin=dict(l=0, r=0, t=4, b=55),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                ))
                st.plotly_chart(fig_wf, use_container_width=True,
                                config={"displayModeBar": False})
                dq_note(
                    f"n={_total_n} emergency surgical cases. Each segment = TAT bin from booking "
                    "entry to theatre start. Booking entry time used as emergency-declared proxy. "
                    "These are inpatient deteriorations — patients already admitted, not walk-in emergencies. "
                    f"Escalate the {_over_24} cases that exceeded 24h to KSH theatre management."
                )

                # ── Day-of-week delay rate ─────────────────────────────────────
                if "DECLARATION_DAY" in _tat_df.columns:
                    _DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                    _tat_df["_over_24"] = _tat_df["BOOKING_TO_START_MIN"] > 1440
                    _dow = (
                        _tat_df.groupby("DECLARATION_DAY")
                        .agg(n_total=("BOOKING_TO_START_MIN", "count"),
                             n_delayed=("_over_24", "sum"))
                        .reset_index()
                    )
                    _dow["_sort"] = _dow["DECLARATION_DAY"].map(
                        {d: i for i, d in enumerate(_DAY_ORDER)}
                    )
                    _dow = _dow.sort_values("_sort").reset_index(drop=True)
                    _dow["delay_pct"] = (
                        100 * _dow["n_delayed"] / _dow["n_total"]
                    ).round(1)
                    _dow["bar_label"] = _dow.apply(
                        lambda r: f"{int(r.n_delayed)}/{int(r.n_total)}", axis=1
                    )

                    def _dow_color(pct):
                        if pct == 0:
                            return COLORS["success"]
                        if pct < 25:
                            return COLORS["warning"]
                        return COLORS["danger"]

                    _bar_colors = [_dow_color(p) for p in _dow["delay_pct"]]

                    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
                    section_header("Emergency Delay by Day of Week — Over-24h Rate")

                    fig_dow = go.Figure(go.Bar(
                        x=_dow["DECLARATION_DAY"].tolist(),
                        y=_dow["delay_pct"].tolist(),
                        text=_dow["bar_label"].tolist(),
                        textposition="outside",
                        marker_color=_bar_colors,
                        hovertemplate=(
                            "<b>%{x}</b><br>Delay rate: %{y:.1f}%"
                            "<br>%{text} cases<extra></extra>"
                        ),
                    ))

                    _sat_row = _dow[_dow["DECLARATION_DAY"] == "Sat"]
                    if len(_sat_row):
                        fig_dow.add_annotation(
                            x="Sat", y=4,
                            text="No elective<br>competition<br>164 min median",
                            showarrow=True, arrowhead=2,
                            arrowcolor=COLORS["success"],
                            font=dict(size=9, color=COLORS["success"]),
                            bgcolor="rgba(255,255,255,0.85)",
                            bordercolor=COLORS["success"],
                            borderwidth=1,
                            ax=50, ay=-50,
                        )

                    fig_dow.update_layout(**cl(
                        height=300,
                        yaxis=dict(title="% over-24h delay", range=[0, 48]),
                        xaxis=dict(
                            title=None,
                            categoryorder="array",
                            categoryarray=_DAY_ORDER,
                        ),
                        margin=dict(l=0, r=0, t=10, b=20),
                        showlegend=False,
                    ))
                    st.plotly_chart(fig_dow, use_container_width=True,
                                    config={"displayModeBar": False})
                    dq_note(
                        "INDICATIVE — n=8 total delayed cases (n=55 with positive TAT). "
                        "Bar labels = delayed/total cases per day. "
                        "Delays concentrate on weekdays when elective theatre is busiest. "
                        "Saturday: 10 cases, 0 delayed, 164 min median — internal optimal benchmark. "
                        "Tuesday cause unresolved (3 delays, does not fit elective-competition pattern). "
                        "Mechanism inferred, not directly measured."
                    )

    # ── Tab 2: Beds ───────────────────────────────────────────────────────────

    with tab2:
        col_l, col_r = st.columns(2, gap="large")

        with col_l:
            section_header("Ward Revenue Hierarchy — Revenue per Bed-Day")
            if len(beds_r):
                _fn_df = (beds_r.dropna(subset=["REVPAB"])
                          .sort_values("REVPAB", ascending=False).head(10))
                fig = go.Figure(go.Funnel(
                    y=_fn_df["WARD_NAME"].tolist(),
                    x=_fn_df["REVPAB"].tolist(),
                    text=[f"KES {v:,.0f}" for v in _fn_df["REVPAB"]],
                    textposition="inside",
                    textinfo="text",
                    marker=dict(color=COLORS["primary"], opacity=0.82),
                    connector=dict(visible=False),
                ))
                fig.update_layout(**cl(height=400, margin=dict(l=0, r=0, t=10, b=20),
                                       showlegend=False))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                dq_note("RevPAB = ward revenue / bed-days. Wards ranked highest to lowest. Specialty wards excluded (dialysis LOS=0 distorts metric).")
                if facility == "KISUMU_CLEAN":
                    info_card(
                        "KSH RevPAB is currently understated. Insured procedure revenue has not been "
                        "recognised since Sep 2025 — these figures reflect cash-patient revenue only. "
                        "Ward rankings will shift significantly once dispatch is restored.",
                        COLORS["warning"])

        with col_r:
            section_header("Ward Revenue per Bed-Day")
            _rvpb_raw = beds_r.copy() if len(beds_r) else pd.DataFrame()
            if len(_rvpb_raw):
                _pvt_kws = ["private", "amenity", "vip", "maternity"]
                _rvpb_raw["ward_type"] = _rvpb_raw["WARD_CATEGORY"].str.lower().apply(
                    lambda x: "Private" if any(k in x for k in _pvt_kws) else "General"
                )
                _rvpb_grp = (
                    _rvpb_raw.groupby("ward_type", as_index=False)
                    .apply(lambda g: pd.Series({
                        "avg_revpab":     g["TOTAL_REVENUE"].sum() / max(g["TOTAL_BED_DAYS"].sum(), 1),
                        "total_bed_days": g["TOTAL_BED_DAYS"].sum(),
                    }))
                    .reset_index(drop=True)
                )
                _rvpb_total_bd = _rvpb_grp["total_bed_days"].sum()
                _rg = _rvpb_grp[_rvpb_grp["ward_type"] == "General"].iloc[0] if "General" in _rvpb_grp["ward_type"].values else None
                _rp = _rvpb_grp[_rvpb_grp["ward_type"] == "Private"].iloc[0] if "Private" in _rvpb_grp["ward_type"].values else None
                _rs1, _rs2 = st.columns(2)
                for _rc, _rrow, _rlbl in [(_rs1, _rg, "General Wards"), (_rs2, _rp, "Private Wards")]:
                    with _rc:
                        if _rrow is not None:
                            _rpct = _rrow["total_bed_days"] / max(_rvpb_total_bd, 1) * 100
                            kpi_card(
                                _rlbl,
                                f"KES {_rrow['avg_revpab']:,.0f}",
                                f"/bed-day · {_rpct:.0f}% of admissions",
                                COLORS["muted"] if _rlbl.startswith("General") else COLORS["primary"],
                            )
                if _rg is not None and _rp is not None:
                    _rmult = _rp["avg_revpab"] / max(_rg["avg_revpab"], 1)
                    _rpvt_pct = _rp["total_bed_days"] / max(_rvpb_total_bd, 1) * 100
                    dq_note(
                        f"Private wards earn <strong>{_rmult:.1f}×</strong> more per bed-day "
                        f"but hold only <strong>{_rpvt_pct:.0f}%</strong> of admissions. "
                        "Filling private capacity is the highest-yield lever available."
                    )
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

        # ── Admissions Pulse (KSH only) ───────────────────────────────────────
        if _is_ksh_p3:
            _adm_raw = _filter_epoch(P["ward_adm"].copy(), "ADMISSION_MONTH") if len(P["ward_adm"]) else pd.DataFrame()
            if len(_adm_raw):
                _adm_total = (
                    _adm_raw[_adm_raw["FACILITY"] == facility]
                    .groupby("ADMISSION_MONTH", as_index=False)["ADMISSIONS"].sum()
                    .sort_values("ADMISSION_MONTH")
                )
                if len(_adm_total):
                    st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
                    section_header("Admissions — 12-Month Pulse")
                    _adm12 = _adm_total.tail(12)
                    _ytd_start = pd.Timestamp(f"{pd.Timestamp.now().year}-01-01")
                    _admissions_ytd = int(_adm_total[_adm_total["ADMISSION_MONTH"] >= _ytd_start]["ADMISSIONS"].sum())
                    _admissions_3mo = int(_adm_total.tail(3)["ADMISSIONS"].sum())
                    fig_adm = go.Figure()
                    # Connected dots — actual monthly path IS the trendline
                    fig_adm.add_scatter(
                        x=_adm12["ADMISSION_MONTH"],
                        y=_adm12["ADMISSIONS"],
                        mode="lines+markers",
                        marker=dict(size=7, color=COLORS["primary"]),
                        line=dict(color=COLORS["primary"], width=2),
                        hovertemplate="%{x|%b %Y}: %{y:,} admissions<extra></extra>",
                        showlegend=False,
                        name="",
                    )
                    # Dashed projection: EMA(3) next month extending from last actual point
                    _adm_proj    = _ema_next(_adm_total["ADMISSIONS"])
                    _adm_last_dt = pd.to_datetime(_adm12["ADMISSION_MONTH"].iloc[-1])
                    if _adm_proj is not None:
                        _adm_next_dt = _adm_last_dt + pd.DateOffset(months=1)
                        fig_adm.add_scatter(
                            x=[_adm_last_dt, _adm_next_dt],
                            y=[float(_adm12["ADMISSIONS"].iloc[-1]), _adm_proj],
                            mode="lines+markers",
                            name="Projection",
                            line=dict(color=COLORS["warning"], width=2, dash="dot"),
                            marker=dict(size=7, symbol="circle-open", color=COLORS["warning"]),
                            hovertemplate="Projection %{x|%b %Y}: ~%{y:.0f} admissions<extra></extra>",
                        )
                    fig_adm.update_layout(**cl(height=240, yaxis_title="Admissions", showlegend=True,
                                               margin=dict(l=0, r=0, t=10, b=30)))
                    st.plotly_chart(fig_adm, use_container_width=True, config={"displayModeBar": False})
                    st.markdown(
                        f'<div style="font-size:11px;color:#6B8CAE;margin-top:-8px">'
                        f'<strong style="color:#003467">{_admissions_ytd:,}</strong> admissions YTD · '
                        f'<strong style="color:#003467">{_admissions_3mo:,}</strong> in last 3 months'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    info_card(
                        "This trend is the upstream signal for monthly operational planning. "
                        "A rising 3-month average points to higher bed occupancy, increased consumables draw, "
                        "and greater shift cover demand in the following month. "
                        "A falling trend creates headroom to schedule maintenance, training, or ward downtime. "
                        "The dashed projection is a 3-month rolling estimate — adjust for known events "
                        "(public holidays, outreach campaigns, seasonal disease peaks).",
                        border_color="#B0C8E0",
                    )

        # ── Operational Demand Outlook — ward-level, next month (KSH only) ─────
        if _is_ksh_p3 and len(P["ward_adm"]):
            _ol_raw = _filter_epoch(P["ward_adm"].copy(), "ADMISSION_MONTH")
            _ol_raw = _ol_raw[_ol_raw["FACILITY"] == facility]
            _ol_ward_map = {
                "MEDICAL — MALE":    "Medical — Male",
                "MEDICAL — FEMALE":  "Medical — Female",
                "MATERNITY":         "Maternity",
                "PRIVATE / AMENITY": "Private / Amenity",
                "PAEDIATRIC":        "Paediatric",
            }
            _ol_labels, _ol_vals = [], []
            for _wk, _wlabel in _ol_ward_map.items():
                _ws = (
                    _ol_raw[_ol_raw["WARD_CATEGORY"].str.upper() == _wk]
                    .sort_values("ADMISSION_MONTH")["ADMISSIONS"]
                )
                _est = _ema_next(_ws)
                if _est is not None:
                    _ol_labels.append(_wlabel)
                    _ol_vals.append(round(_est, 1))
            if _ol_labels:
                st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
                section_header("Operational Demand Outlook — Next Month")
                fig_ol = go.Figure(go.Bar(
                    x=_ol_vals,
                    y=_ol_labels,
                    orientation="h",
                    marker_color=COLORS["primary"],
                    opacity=0.75,
                    text=[f"~{int(v)}" for v in _ol_vals],
                    textposition="outside",
                    hovertemplate="%{y}: ~%{x:.0f} admissions<extra></extra>",
                ))
                fig_ol.update_layout(**cl(
                    height=220,
                    xaxis_title="Projected admissions",
                    margin=dict(l=140, r=60, t=10, b=30),
                    showlegend=False,
                ))
                st.plotly_chart(fig_ol, use_container_width=True, config={"displayModeBar": False})
                dq_note(
                    "Ward projections reflect recent admission patterns (3-month EMA). "
                    "Staffing changes affect these figures within 1–2 months."
                )
                info_card(
                    "Ward-level projections give procurement leads, ward managers, and scheduling teams "
                    "a 4–6 week planning window. Higher projected volume in a ward signals the need to "
                    "align bed readiness, ward-specific stock (e.g. delivery supplies for Maternity, "
                    "paediatric medications for Paediatric), and shift cover ahead of the month. "
                    "Note: Monday 09:00–12:00 and 16:00 remain the peak admission windows regardless of "
                    "monthly volume — same-day demand on those windows runs above the monthly average.",
                    border_color="#B0C8E0",
                )

        # ── Ward Capacity Pressure (KSH only) ────────────────────────────────
        if _is_ksh_p3 and len(P["beds_monthly"]):
            _bm = _filter_epoch(P["beds_monthly"].copy(), "ADMISSION_MONTH")
            if len(_bm):
                _WARD_COLORS = {
                    "General Female":    COLORS["primary"],
                    "General Male":      COLORS["success"],
                    "Pediatric General": COLORS["warning"],
                    "General Maternity": COLORS["danger"],
                    "Private Maternity": "#E8A0A0",
                    "Private Female":    "#9B59B6",
                    "Private Male":      "#5DADE2",
                }
                st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
                section_header("Ward Capacity Pressure — Bed Days Used & Length of Stay")
                fig_cp = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    subplot_titles=("Bed Days Used per Ward", "Avg Length of Stay per Ward"),
                    vertical_spacing=0.10,
                )
                for _wrd in sorted(_bm["WARD_NAME"].unique()):
                    _ws  = _bm[_bm["WARD_NAME"] == _wrd].sort_values("ADMISSION_MONTH")
                    _col = _WARD_COLORS.get(_wrd, COLORS["muted"])
                    fig_cp.add_scatter(
                        row=1, col=1,
                        x=_ws["ADMISSION_MONTH"], y=_ws["TOTAL_BED_DAYS"],
                        mode="lines+markers", name=_wrd, legendgroup=_wrd,
                        line=dict(color=_col, width=2), marker=dict(size=5),
                        hovertemplate=f"{_wrd} %{{x|%b %Y}}: %{{y:,}} bed-days<extra></extra>",
                        showlegend=True,
                    )
                    fig_cp.add_scatter(
                        row=2, col=1,
                        x=_ws["ADMISSION_MONTH"], y=_ws["AVG_LOS_DAYS"],
                        mode="lines+markers", name=_wrd, legendgroup=_wrd,
                        line=dict(color=_col, width=2), marker=dict(size=5),
                        hovertemplate=f"{_wrd} %{{x|%b %Y}}: %{{y:.1f}}d<extra></extra>",
                        showlegend=False,
                    )
                fig_cp.update_yaxes(title_text="Bed Days", row=1, col=1)
                fig_cp.update_yaxes(title_text="Avg Days", row=2, col=1)
                fig_cp.update_layout(**cl(
                    height=580,
                    legend=dict(orientation="h", y=-0.08, x=0.5,
                                xanchor="center", font_size=10),
                    margin=dict(b=80),
                ))
                st.plotly_chart(fig_cp, use_container_width=True,
                                config={"displayModeBar": True, "displaylogo": False,
                                        "modeBarButtonsToRemove": ["select2d", "lasso2d"]})
                # ── Capacity pressure insight card ───────────────────────────
                _bm_agg = (
                    _bm.groupby("ADMISSION_MONTH")
                    .agg(total_bed_days=("TOTAL_BED_DAYS", "sum"),
                         avg_los=("AVG_LOS_DAYS", "mean"))
                    .reset_index().sort_values("ADMISSION_MONTH")
                )
                if len(_bm_agg) >= 4:
                    _rec3    = _bm_agg.iloc[-3:]
                    _pri3    = _bm_agg.iloc[-6:-3] if len(_bm_agg) >= 6 else _bm_agg.iloc[:-3]
                    _bd_up   = _rec3["total_bed_days"].mean() > _pri3["total_bed_days"].mean() * 1.05
                    _los_up  = _rec3["avg_los"].mean() > _pri3["avg_los"].mean() * 1.05
                    _los_dn  = _rec3["avg_los"].mean() < _pri3["avg_los"].mean() * 0.95
                    _is_max  = _bd_up and _los_up
                    _is_min  = (not _bd_up) and _los_dn
                else:
                    _is_max = _is_min = False

                _c_max = "#FEF2F2" if _is_max else "#F9FAFB"
                _b_max = "#DC2626" if _is_max else "#E5E7EB"
                _t_max = "#991B1B" if _is_max else "#9CA3AF"
                _c_min = "#F0FDF4" if _is_min else "#F9FAFB"
                _b_min = "#16A34A" if _is_min else "#E5E7EB"
                _t_min = "#166534" if _is_min else "#9CA3AF"
                _curr_lbl = (
                    "▲ Currently: Maximum Pressure — beds are blocked, new admissions constrained"
                    if _is_max else
                    "▼ Currently: Minimum Pressure — beds cycling freely"
                    if _is_min else
                    "Current pattern: Mixed — read chart for ward-level detail"
                )
                _curr_clr = "#DC2626" if _is_max else "#16A34A" if _is_min else "#6B8CAE"
                st.markdown(
                    f'<div style="background:#F4F8FC;border:1px solid #D6E4F0;border-radius:8px;'
                    f'padding:14px 18px;margin:10px 0">'
                    f'<div style="font-size:10px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
                    f'letter-spacing:1.5px;margin-bottom:10px">WARD CAPACITY PRESSURE — WHAT THE TREND MEANS</div>'
                    f'<div style="display:flex;gap:10px;margin-bottom:10px">'
                    f'<div style="flex:1;background:{_c_max};border-left:3px solid {_b_max};border-radius:4px;padding:10px 12px">'
                    f'<div style="font-size:11px;font-weight:700;color:{_t_max}">HIGH BED DAYS + LONG ALOS</div>'
                    f'<div style="font-size:12px;font-weight:600;color:{_t_max};margin-top:4px">Maximum Pressure</div>'
                    f'<div style="font-size:11px;color:{_t_max};margin-top:3px">Beds rarely vacate. New patients cannot be admitted.</div>'
                    f'</div>'
                    f'<div style="flex:1;background:{_c_min};border-left:3px solid {_b_min};border-radius:4px;padding:10px 12px">'
                    f'<div style="font-size:11px;font-weight:700;color:{_t_min}">LOW BED DAYS + SHORT ALOS</div>'
                    f'<div style="font-size:12px;font-weight:600;color:{_t_min};margin-top:4px">Minimum Pressure</div>'
                    f'<div style="font-size:11px;color:{_t_min};margin-top:3px">Beds open quickly. Plenty of room for new admissions.</div>'
                    f'</div>'
                    f'</div>'
                    f'<div style="font-size:11px;font-weight:700;color:{_curr_clr}">{_curr_lbl}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Ward Signals (KSH only) ──────────────────────────────────────────
        if _is_ksh_p3 and len(P["beds_monthly"]):
            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)

            _bm_all = P["beds_monthly"].copy()

            # Baselines from full history — not epoch-filtered
            _bl = (
                _bm_all.groupby("WARD_NAME", as_index=False)
                .agg(base_bd=("TOTAL_BED_DAYS", "mean"),
                     base_los=("AVG_LOS_DAYS",   "mean"))
            )

            # Epoch-filtered for display, merged with baselines
            _bm_sig = _filter_epoch(_bm_all, "ADMISSION_MONTH").merge(_bl, on="WARD_NAME", how="left")
            _bm_sig["bd_ratio"]  = _bm_sig["TOTAL_BED_DAYS"] / _bm_sig["base_bd"].clip(lower=0.1)
            _bm_sig["los_ratio"] = _bm_sig["AVG_LOS_DAYS"]   / _bm_sig["base_los"].clip(lower=0.1)

            def _ward_signal(row):
                bd, los = row["bd_ratio"], row["los_ratio"]
                if bd > 1.2 and los > 1.2:
                    return "🔴 Capacity Compression", "Similar volume, beds blocked by long-stayers. Complex or severe case load."
                if bd > 1.2 and 0.8 <= los <= 1.2:
                    return "🟡 Demand Growth", "More patients admitted at normal acuity. Volume-driven pressure."
                if bd < 0.8 and los < 0.8:
                    return "🟢 Efficient Throughput", "Patients cycling faster than baseline. Ward flowing well."
                if bd < 0.8 and los > 1.2:
                    return "🟠 Low Volume, Complex Cases", "Fewer patients but staying longer. Possible case mix shift."
                return "⚪ Normal", "Bed days and LOS within ±20% of baseline."

            _bm_sig[["Signal", "What Happened"]] = _bm_sig.apply(
                _ward_signal, axis=1, result_type="expand"
            )

            with st.expander("Ward Signals — What Happened Each Month", expanded=False):
                _show_all = st.checkbox("Show full history", value=False, key="ward_sig_full")
                if _show_all:
                    _bm_disp = _bm_sig.copy()
                else:
                    _max_mo  = pd.to_datetime(_bm_sig["ADMISSION_MONTH"]).max()
                    _bm_disp = _bm_sig[
                        pd.to_datetime(_bm_sig["ADMISSION_MONTH"]) >= _max_mo - pd.DateOffset(months=5)
                    ]

                _bm_disp = _bm_disp.sort_values(
                    ["ADMISSION_MONTH", "WARD_NAME"], ascending=[False, True]
                ).copy()
                _bm_disp["ADMISSION_MONTH"] = pd.to_datetime(_bm_disp["ADMISSION_MONTH"]).dt.strftime("%b %Y")
                _bm_disp["AVG_LOS_DAYS"]    = _bm_disp["AVG_LOS_DAYS"].round(1)

                st.dataframe(
                    _bm_disp[[
                        "WARD_NAME", "ADMISSION_MONTH", "DISCHARGED_ADMISSIONS",
                        "TOTAL_BED_DAYS", "AVG_LOS_DAYS", "Signal", "What Happened"
                    ]].rename(columns={
                        "WARD_NAME":              "Ward",
                        "ADMISSION_MONTH":        "Month",
                        "DISCHARGED_ADMISSIONS":  "Discharges",
                        "TOTAL_BED_DAYS":         "Bed Days",
                        "AVG_LOS_DAYS":           "Avg LOS (d)",
                    }),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Signal":       st.column_config.TextColumn(width="medium"),
                        "What Happened": st.column_config.TextColumn(width="large"),
                    },
                )
                dq_note(
                    "Signals computed vs all-time ward baseline (±20% threshold). "
                    "Bed days = discharged admissions only — open admissions excluded."
                )

        # ── Ward Turnover Efficiency (KSH only / B3 / P16-6) ─────────────────
        _btr_df = P.get("btr_bti", pd.DataFrame())
        if _is_ksh_p3 and len(_btr_df):
            _btr_df = _btr_df.copy()
            _btr_df.columns = _btr_df.columns.str.lower()
            _btr_df = _btr_df[_btr_df["month"].notna()].sort_values(["ward_name", "month"])
            if len(_btr_df):
                st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
                section_header("Ward Turnover Efficiency")
                _btr_wards = sorted(_btr_df["ward_name"].unique())
                _sel_btr   = st.selectbox("Select ward", _btr_wards, key="btr_ward_sel")
                _btr_w     = _btr_df[_btr_df["ward_name"] == _sel_btr].copy()
                _btr_w["month_lbl"] = pd.to_datetime(_btr_w["month"]).dt.strftime("%b %Y")
                _fig_btr = go.Figure()
                _fig_btr.add_trace(go.Bar(
                    x=_btr_w["month_lbl"], y=_btr_w["btr"],
                    name="BTR", marker_color=COLORS["primary"],
                    hovertemplate="%{x}: BTR %{y:.2f}<extra></extra>",
                ))
                _fig_btr.add_trace(go.Scatter(
                    x=_btr_w["month_lbl"], y=_btr_w["bti_days"],
                    name="BTI (days)", mode="lines+markers",
                    line=dict(color=COLORS["warning"], width=2),
                    marker=dict(size=7),
                    yaxis="y2",
                    hovertemplate="%{x}: BTI %{y:.1f} days<extra></extra>",
                ))
                _btr_alos = (
                    _btr_w["total_bed_days"].sum()
                    / max(_btr_w["total_admissions"].sum(), 1)
                )
                _fig_btr.update_layout(**cl(
                    height=320,
                    yaxis_title="BTR (admissions / bed)",
                    yaxis2=dict(
                        title="BTI (empty days between admissions)",
                        overlaying="y", side="right",
                        tickfont=dict(size=10, color="#6B8CAE"),
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1),
                    margin=dict(l=10, r=10, t=30, b=10),
                    transition_duration=400,
                ))
                st.plotly_chart(_fig_btr, use_container_width=True,
                                config={"displayModeBar": False})
                st.caption(
                    f"**BTR** (Bed Turnover Rate) = admissions ÷ available beds. "
                    f"**BTI** (Bed Turnover Interval) = avg days a bed sits empty between admissions "
                    f"— lower BTI = faster cycling. "
                    f"**ALOS** = {_btr_alos:.1f} days (12-month avg, {_sel_btr})."
                )
                # ── BTI/BTR quadrant insight card ─────────────────────────────
                if len(_btr_w) >= 2:
                    _btr_med_r = float(_btr_w["btr"].median())
                    _btr_med_i = float(_btr_w["bti_days"].median())
                    _btr_cur_r = float(_btr_w.iloc[-1]["btr"])
                    _btr_cur_i = float(_btr_w.iloc[-1]["bti_days"])
                    _hi_btr    = _btr_cur_r >= _btr_med_r
                    _hi_bti    = _btr_cur_i >= _btr_med_i

                    if not _hi_bti and _hi_btr:
                        _q_lbl  = "LOW BTI + HIGH BTR — High Efficiency, High Occupancy"
                        _q_body = "Beds fill almost instantly and patient volume is high. Peak operational state — vulnerable to sudden surges."
                        _q_bg, _q_br, _q_tc = "#FFFBEB", "#D97706", "#92400E"
                    elif _hi_bti and not _hi_btr:
                        _q_lbl  = "HIGH BTI + LOW BTR — Low Efficiency, Low Occupancy"
                        _q_body = "Beds sit empty for long periods between patients. Low demand or delayed admissions — investigate intake protocol."
                        _q_bg, _q_br, _q_tc = "#F9FAFB", "#9CA3AF", "#6B7280"
                    elif not _hi_bti and not _hi_btr:
                        _q_lbl  = "LOW BTI + LOW BTR — Long-Stay Bottleneck"
                        _q_body = "Beds are always full but few new patients cycle through. Existing patients cannot leave — ALOS is the pressure driver, not demand volume."
                        _q_bg, _q_br, _q_tc = "#FEF2F2", "#DC2626", "#991B1B"
                    else:
                        _q_lbl  = "HIGH BTI + HIGH BTR — Surge Pattern"
                        _q_body = "Long gaps between patients but high overall volume — short ALOS ward cycling rapidly. Pressure swings between extremes."
                        _q_bg, _q_br, _q_tc = "#EFF6FF", "#0072CE", "#1E40AF"

                    st.markdown(
                        f'<div style="background:{_q_bg};border-left:3px solid {_q_br};'
                        f'border-radius:6px;padding:12px 16px;margin:10px 0">'
                        f'<div style="font-size:10px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
                        f'letter-spacing:1.5px;margin-bottom:6px">WARD TURNOVER STATE — {_sel_btr.upper()}</div>'
                        f'<div style="font-size:13px;font-weight:700;color:{_q_tc}">{_q_lbl}</div>'
                        f'<div style="font-size:12px;color:{_q_tc};margin-top:5px">{_q_body}</div>'
                        f'<div style="font-size:11px;color:#6B8CAE;margin-top:6px">'
                        f'Latest: BTR {_btr_cur_r:.2f} (median {_btr_med_r:.2f}) · '
                        f'BTI {_btr_cur_i:.1f} d (median {_btr_med_i:.1f} d) · '
                        f'ALOS {_btr_alos:.1f} d · Oct 2025 excluded (data gap Inv 32)</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # ── Admission TAT by Day of Week ──────────────────────────────────────
        if _is_ksh_p3:
            _tat_df = P.get("adm_tat", pd.DataFrame())
            if len(_tat_df):
                _tat_df = _tat_df.copy()
                _tat_df.columns = _tat_df.columns.str.lower()
                _tat_df = _tat_df.sort_values("day_num")
                st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
                section_header("Admission TAT — How Fast Are Beds Being Assigned")
                _fig_tat = make_subplots(specs=[[{"secondary_y": True}]])
                _bar_colors = [
                    COLORS["danger"] if p < 45
                    else COLORS["warning"] if p < 55
                    else COLORS["success"]
                    for p in _tat_df["fast_pct"]
                ]
                _fig_tat.add_bar(
                    x=_tat_df["day_name"], y=_tat_df["fast_pct"],
                    name="Admission speed",
                    marker_color=_bar_colors,
                    text=_tat_df["fast_pct"].apply(lambda p: f"{p:.0f}%"),
                    textposition="outside",
                    secondary_y=False,
                    showlegend=False,
                    hovertemplate="%{x}: %{y:.1f}% admitted fast · 1-in-4 wait: "
                                  + _tat_df["p75_tat_min"].apply(lambda v: f"{v:.0f} min").astype(str)
                                  + "<extra></extra>",
                )
                _fig_tat.add_scatter(
                    x=_tat_df["day_name"], y=_tat_df["total_evaluations"],
                    name="Visit load (total evaluations)",
                    mode="lines+markers",
                    line=dict(color=COLORS["primary"], width=2, dash="dot"),
                    marker=dict(size=7),
                    secondary_y=True,
                    hovertemplate="%{x}: %{y:,} evaluation visits<extra></extra>",
                )
                for _lbl, _clr in [
                    ("Fast — majority admitted within 1 hour", COLORS["success"]),
                    ("Mixed — roughly half wait over 1 hour", COLORS["warning"]),
                    ("Slow — majority wait over 1 hour", COLORS["danger"]),
                ]:
                    _fig_tat.add_scatter(
                        x=[None], y=[None], mode="markers",
                        marker=dict(color=_clr, size=10, symbol="square"),
                        name=_lbl, secondary_y=False, showlegend=True,
                    )
                _fig_tat.update_layout(**cl(
                    height=320,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=10, r=60, t=30, b=10),
                ))
                _fig_tat.update_yaxes(
                    title_text="Admission speed (%)", ticksuffix="%",
                    range=[0, 100], secondary_y=False,
                )
                _fig_tat.update_yaxes(
                    title_text="Visit load", secondary_y=True,
                    rangemode="tozero",
                )
                st.plotly_chart(_fig_tat, use_container_width=True,
                                config={"displayModeBar": False})
                _worst = _tat_df.sort_values("fast_pct").iloc[0]
                _second = _tat_df.sort_values("fast_pct").iloc[1]
                st.caption(
                    f"**{_worst['day_name']}** slowest — {_worst['fast_pct']:.1f}% admitted fast, "
                    f"1 in 4 waited {_worst['p75_tat_min']:.0f}+ min. "
                    f"**{_second['day_name']}** second slowest ({_second['fast_pct']:.1f}%, "
                    f"1 in 4 waited {_second['p75_tat_min']:.0f}+ min). "
                    "When visit load line is high and bars are red — volume is driving the delay."
                )
                # ── TAT–volume correlation insight card ───────────────────────
                _tat_corr     = _tat_df[["fast_pct", "total_evaluations"]].corr().iloc[0, 1]
                _vol_driven   = _tat_corr < -0.3   # higher visits → lower fast_pct
                _tat_wst      = _tat_df.sort_values("fast_pct").iloc[0]
                _tat_bst      = _tat_df.sort_values("fast_pct").iloc[-1]
                if _vol_driven:
                    _ti_title = "VOLUME IS DRIVING DELAYS"
                    _ti_body  = (
                        f"Days with more evaluation visits show slower bed assignment "
                        f"(r = {abs(_tat_corr):.2f}). "
                        f"{_tat_wst['day_name']} is the slowest day "
                        f"({_tat_wst['fast_pct']:.0f}% fast) and carries the highest visit load. "
                        f"Staggering slots or adding triage capacity at peak would recover the delay."
                    )
                    _ti_bg, _ti_br, _ti_tc = "#FFFBEB", "#D97706", "#92400E"
                else:
                    _ti_title = "DELAYS NOT DRIVEN BY VOLUME ALONE"
                    _ti_body  = (
                        f"Visit volume does not fully explain variation in bed assignment speed. "
                        f"{_tat_wst['day_name']} is slowest ({_tat_wst['fast_pct']:.0f}% fast), "
                        f"{_tat_bst['day_name']} fastest ({_tat_bst['fast_pct']:.0f}% fast). "
                        f"Staffing pattern, bed availability, or process gaps may be the primary driver."
                    )
                    _ti_bg, _ti_br, _ti_tc = "#F4F8FC", "#6B8CAE", "#003467"
                st.markdown(
                    f'<div style="background:{_ti_bg};border-left:3px solid {_ti_br};'
                    f'border-radius:6px;padding:12px 16px;margin:10px 0">'
                    f'<div style="font-size:10px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
                    f'letter-spacing:1.5px;margin-bottom:6px">ADMISSION SPEED — WHAT IS DRIVING IT</div>'
                    f'<div style="font-size:13px;font-weight:700;color:{_ti_tc}">{_ti_title}</div>'
                    f'<div style="font-size:12px;color:{_ti_tc};margin-top:5px">{_ti_body}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "▸ When visit load spikes (Monday peak), TAT degrades — conversion drops and "
                    "patients are lost. See **Causal Intelligence → CD5** for the full cross-domain impact."
                )

    # ── Tab 3: Lab & Diagnostics ──────────────────────────────────────────────

    with tab3:

        # ── Lab Volume + Abnormal Rate (KSH only) ─────────────────────────────
        # LAB_VOL_HIDDEN — reactivate by changing `if False` to `if _is_ksh_p3 and len(P["lab"])`
        if False:  # noqa
            _lab3 = _filter_epoch(P["lab"].copy(), "LAB_MONTH")
            _lab3 = _lab3[_lab3["DISTINCT_VISITS"] > 50].sort_values("LAB_MONTH")
            if len(_lab3):
                section_header("Lab — Volume & Abnormal Rate")
                _lc1, _lc2 = st.columns(2, gap="large")

                with _lc1:
                    _fig_lv = go.Figure()
                    _lv_colors = [
                        COLORS["danger"]  if v < 350
                        else COLORS["warning"] if v < 430
                        else COLORS["primary"]
                        for v in _lab3["DISTINCT_VISITS"]
                    ]
                    _fig_lv.add_bar(
                        x=_lab3["LAB_MONTH"], y=_lab3["DISTINCT_VISITS"],
                        marker_color=_lv_colors,
                        hovertemplate="%{x|%b %Y}: %{y:,} visits<extra></extra>",
                        showlegend=False,
                    )
                    _fig_lv.add_hline(y=430, line_dash="dot", line_color=COLORS["warning"],
                                      line_width=1.5,
                                      annotation_text="WATCH <430",
                                      annotation_font_size=9,
                                      annotation_font_color=COLORS["warning"],
                                      annotation_position="top right")
                    _fig_lv.add_hline(y=350, line_dash="dot", line_color=COLORS["danger"],
                                      line_width=1.5,
                                      annotation_text="CRITICAL <350",
                                      annotation_font_size=9,
                                      annotation_font_color=COLORS["danger"],
                                      annotation_position="bottom right")
                    _fig_lv.update_layout(**cl(height=280, yaxis_title="Distinct visits/month"))
                    st.plotly_chart(_fig_lv, use_container_width=True, config={"displayModeBar": False})

                with _lc2:
                    _fig_la = go.Figure()
                    _la_colors = [
                        COLORS["danger"]  if v > 11
                        else COLORS["warning"] if v > 9
                        else COLORS["success"]
                        for v in _lab3["ABNORMAL_PCT"]
                    ]
                    _fig_la.add_bar(
                        x=_lab3["LAB_MONTH"], y=_lab3["ABNORMAL_PCT"],
                        marker_color=_la_colors,
                        hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>",
                        showlegend=False,
                    )
                    _fig_la.add_hline(y=9, line_dash="dot", line_color=COLORS["warning"],
                                      line_width=1.5,
                                      annotation_text="WATCH >9%",
                                      annotation_font_size=9,
                                      annotation_font_color=COLORS["warning"],
                                      annotation_position="top right")
                    _fig_la.add_hline(y=11, line_dash="dot", line_color=COLORS["danger"],
                                      line_width=1.5,
                                      annotation_text="CRITICAL >11%",
                                      annotation_font_size=9,
                                      annotation_font_color=COLORS["danger"],
                                      annotation_position="top right")
                    _fig_la.update_layout(**cl(height=280, yaxis_title="Abnormal flag rate (%)"))
                    st.plotly_chart(_fig_la, use_container_width=True, config={"displayModeBar": False})

                dq_note(
                    "Lab volume: distinct patient visits with at least one lab result. "
                    "WATCH <430/month for 2 consecutive months; CRITICAL <350 single month. "
                    "Abnormal rate: % of all lab test results flagged H or L (high or low) across every test type — "
                    "not specific to any one test. A rising rate signals a sicker patient mix or a specific category spiking. "
                    "WATCH >9% for 2 months; CRITICAL >11%."
                )
                st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

        # ── Critical Creatinine — Admission Outcome (KSH only) ────────────────
        if _is_ksh_p3 and len(P["cd12_rate"]):
            _cd12 = P["cd12_rate"].copy().rename(columns=str.lower)
            _cd12 = _cd12.sort_values("critical_month")
            if len(_cd12) >= 2:
                section_header("Renal Patients — Critical Creatinine Non-Admission Rate")
                fig_cd12 = go.Figure()
                _cd12_labels = [
                    f"{int(r.not_admitted)}/{int(r.total_critical)}"
                    for r in _cd12.itertuples()
                ]
                fig_cd12.add_scatter(
                    x=_cd12["critical_month"],
                    y=_cd12["non_admission_rate_pct"],
                    mode="lines+markers+text",
                    marker=dict(size=7, color=COLORS["coral"]),
                    line=dict(color=COLORS["coral"], width=2),
                    text=_cd12_labels,
                    textposition="top center",
                    textfont=dict(size=10, color=COLORS["coral"]),
                    hovertemplate=(
                        "%{x|%b %Y}: %{y:.1f}% not admitted"
                        " (%{text} patients)<extra></extra>"
                    ),
                    showlegend=False,
                )
                fig_cd12.add_hline(
                    y=41, line_dash="dot", line_color=COLORS["muted"], line_width=1.5,
                    annotation_text="CD12 baseline 41%",
                    annotation_font_size=9,
                    annotation_font_color=COLORS["muted"],
                    annotation_position="top left",
                )
                fig_cd12.update_layout(**cl(
                    height=300, yaxis_title="Non-admission rate (%)",
                    yaxis=dict(range=[0, 85]),
                    margin=dict(l=0, r=0, t=24, b=30),
                ))
                st.plotly_chart(fig_cd12, use_container_width=True,
                                config={"displayModeBar": False})
                st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

        # ── Dialysis — Programme Status (KSH only) ────────────────────────────
        if _is_ksh_p3:
            st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
            section_header("Dialysis — Why It Connects to the Above")
            _dial_peak_row   = fac_dialysis_ops.nlargest(1, "SESSIONS_BILLED") if len(fac_dialysis_ops) else pd.DataFrame()
            _dial_peak_sess  = int(_dial_peak_row["SESSIONS_BILLED"].iloc[0])  if len(_dial_peak_row) else 0
            _dial_peak_month = (pd.to_datetime(_dial_peak_row["INVOICE_MONTH"].iloc[0]).strftime("%b %Y")
                                if len(_dial_peak_row) else "")
            _dial_rev_sorted = fac_dialysis_ops.sort_values("INVOICE_MONTH")
            if len(_dial_rev_sorted) >= 2:
                _rev_latest = float(_dial_rev_sorted.iloc[-1]["SESSION_FEE_REVENUE"])
                _rev_prev   = float(_dial_rev_sorted.iloc[-2]["SESSION_FEE_REVENUE"])
                _rev_chg    = (_rev_latest - _rev_prev) / max(_rev_prev, 1) * 100
                _rev_arrow  = "▲" if _rev_chg >= 0 else "▼"
                _rev_clr    = COLORS["success"] if _rev_chg >= 0 else COLORS["danger"]
                _rev_sub    = (f'<span style="color:{_rev_clr};font-weight:700">'
                               f'{_rev_arrow} {abs(_rev_chg):.1f}%</span> vs prior month')
            elif len(_dial_rev_sorted) == 1:
                _rev_latest = float(_dial_rev_sorted.iloc[-1]["SESSION_FEE_REVENUE"])
                _rev_clr    = COLORS["success"]
                _rev_sub    = "First complete month on record"
            else:
                _rev_latest = 0
                _rev_clr    = COLORS["muted"]
                _rev_sub    = "—"
            # Row 1 — programme performance
            _dr1, _dr2 = st.columns(2, gap="large")
            with _dr1:
                kpi_card("Programme Sessions",
                         str(_dial_peak_sess) if _dial_peak_sess else "—",
                         f"Peak {_dial_peak_month} · NHIF-funded", COLORS["primary"])
            with _dr2:
                kpi_card("Monthly Session Revenue",
                         fmt_kes(_rev_latest) if _rev_latest else "—",
                         _rev_sub, _rev_clr)
            st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
            # Row 2 — the gap
            _demand_kes = 122 * 8 * 10_650
            _dr3, _dr4 = st.columns(2, gap="large")
            with _dr3:
                kpi_card("CD12 Routing Gap", "96.8%",
                         "122 of 126 critical creatinine patients not in programme", COLORS["danger"])
            with _dr4:
                kpi_card(
                    "Patient Demand Not Reached",
                    fmt_kes(_demand_kes),
                    "122 patients × 8 sessions/month × KES 10,650 · indicative upper bound",
                    COLORS["warning"], icon="⚠",
                )
            _dial3 = fac_dialysis_ops.sort_values("INVOICE_MONTH") if len(fac_dialysis_ops) else pd.DataFrame()
            if len(_dial3):
                # Chart 1 — sessions by payer: stacked NHIF / cash (growth story)
                _fig_dial1 = go.Figure()
                _fig_dial1.add_bar(
                    x=_dial3["INVOICE_MONTH"], y=_dial3["SESSIONS_INSURED"],
                    name="NHIF", marker_color=COLORS["primary"],
                    hovertemplate="%{x|%b %Y}: %{y} NHIF<extra></extra>",
                )
                _fig_dial1.add_bar(
                    x=_dial3["INVOICE_MONTH"], y=_dial3["SESSIONS_CASH"],
                    name="Cash", marker_color=COLORS["warning"],
                    hovertemplate="%{x|%b %Y}: %{y} cash<extra></extra>",
                )
                _fig_dial1.update_layout(**cl(
                    barmode="stack", height=230, yaxis_title="Sessions / month",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1, font=dict(size=10)),
                    margin=dict(l=0, r=0, t=30, b=30),
                ))
                st.plotly_chart(_fig_dial1, use_container_width=True,
                                config={"displayModeBar": False})

                # Charts 2 + 3 — utilisation % and session revenue side by side
                _dc1, _dc2 = st.columns(2, gap="medium")
                with _dc1:
                    st.caption("Utilisation — % of one-shift theoretical max (264 sessions/month)")
                    _fig_util = go.Figure()
                    _fig_util.add_scatter(
                        x=_dial3["INVOICE_MONTH"],
                        y=_dial3["UTILISATION_PCT_THEORETICAL"],
                        mode="lines+markers",
                        line=dict(color=COLORS["primary"], width=2),
                        marker=dict(size=6),
                        hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>",
                        showlegend=False,
                    )
                    _fig_util.add_hline(y=100, line_dash="dot",
                                        line_color=COLORS["muted"],
                                        annotation_text="100% cap")
                    _fig_util.update_layout(**cl(
                        height=200, yaxis_title="Utilisation %",
                        margin=dict(l=0, r=0, t=10, b=30),
                    ))
                    _fig_util.update_yaxes(range=[0, 115])
                    st.plotly_chart(_fig_util, use_container_width=True,
                                    config={"displayModeBar": False})
                with _dc2:
                    st.caption("Session fee revenue per month (NHIF + cash, KES)")
                    _fig_rev = go.Figure()
                    _fig_rev.add_bar(
                        x=_dial3["INVOICE_MONTH"],
                        y=(_dial3["SESSION_FEE_REVENUE"] / 1_000).round(0),
                        marker_color=COLORS["success"],
                        hovertemplate="%{x|%b %Y}: KES %{y:.0f}K<extra></extra>",
                        showlegend=False,
                    )
                    _fig_rev.update_layout(**cl(
                        height=200, yaxis_title="KES (thousands)",
                        margin=dict(l=0, r=0, t=10, b=30),
                    ))
                    st.plotly_chart(_fig_rev, use_container_width=True,
                                    config={"displayModeBar": False})
            info_card(
                "KSH dialysis programme is operational — 135 sessions in December 2025, predominantly "
                "NHIF-funded at KES 10,650/session. Running at 35–51% of one-shift theoretical capacity "
                "(6 machines, 264 sessions/month maximum). Data to April 21 2026. "
                "The clinical gap is referral routing: 122 of 126 patients with critical creatinine results "
                "(96.8%) have no dialysis billing record. Capacity exists — the pathway from critical "
                "creatinine detection to dialysis enrolment is not functioning. See Causal Intelligence → CD12.",
                border_color=COLORS["warning"],
            )

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
                         revenue=("REVENUE", "sum"))
                    .reindex([m for m in MODALITY_ORDER if m in recent.groupby("MODALITY").groups])
                    .reset_index()
                )
                mod_summary["avg_per"] = mod_summary["revenue"] / mod_summary["sessions"]
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
                            f"avg KES {int(row['avg_per']):,}/session",
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
                    "Invoices generated but not submitted post-Sep 2025. "
                    "Note: these same items appear within the 'Investigations (incl. fees)' block on the Service Mix page — "
                    "do not add figures from both pages. Pending promotion to gold table G8."
                )

                # ── Imaging Utilisation — Sessions per 100 Outpatient Visits ──
                _vis_s = P["visit_sum"].copy()
                if len(_vis_s):
                    _vis_s.columns = _vis_s.columns.str.upper()
                    _vis_s["MONTH"] = pd.to_datetime(_vis_s["VISIT_MONTH"])
                    _img_u = fac_img[fac_img["MODALITY"] != "Other Imaging"].copy()
                    _img_u["MONTH"] = pd.to_datetime(_img_u["REVENUE_MONTH"])
                    _img_u = _img_u.merge(_vis_s[["MONTH", "TOTAL_VISITS"]], on="MONTH", how="inner")
                    _img_u["rate"] = (
                        _img_u["SESSIONS"] / _img_u["TOTAL_VISITS"].clip(lower=1) * 100
                    ).round(2)

                    _BENCH = {
                        "X-Ray":          {"low": 5.0,  "high": 10.0, "src": "WHO Essential Imaging 2020"},
                        "Ultrasound":     {"low": 2.0,  "high": 5.0,  "src": "WHO Essential Imaging 2020"},
                        "CT / Angio":     {"low": 0.5,  "high": 2.0,  "src": "RSNA low-resource guidance"},
                        "ECHO / Cardiac": {"low": 0.5,  "high": 1.0,  "src": "Indicative — no Kenya standard"},
                        "MRI":            {"low": 0.1,  "high": 0.5,  "src": "RSNA low-resource guidance"},
                    }
                    _ut_mods = [m for m in _BENCH if m in _img_u["MODALITY"].unique()]
                    if _ut_mods:
                        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
                        section_header("Imaging Utilisation — Sessions per 100 Outpatient Visits")
                        _n_cols = 2
                        _n_rows = -(-len(_ut_mods) // _n_cols)
                        fig_ut = make_subplots(
                            rows=_n_rows, cols=_n_cols,
                            subplot_titles=_ut_mods,
                            vertical_spacing=0.14,
                            horizontal_spacing=0.10,
                        )
                        for _mi, _mod in enumerate(_ut_mods):
                            _row = _mi // _n_cols + 1
                            _col = _mi % _n_cols + 1
                            _b   = _BENCH[_mod]
                            _sub = _img_u[_img_u["MODALITY"] == _mod].sort_values("MONTH")
                            fig_ut.add_hrect(
                                y0=_b["low"], y1=_b["high"],
                                fillcolor="rgba(144,238,144,0.2)",
                                line_width=0,
                                row=_row, col=_col,
                            )
                            fig_ut.add_scatter(
                                row=_row, col=_col,
                                x=_sub["MONTH"], y=_sub["rate"],
                                mode="lines+markers",
                                name=_mod,
                                line=dict(color=MODALITY_COLORS.get(_mod, COLORS["muted"]), width=2),
                                marker=dict(size=5),
                                hovertemplate=f"{_mod} %{{x|%b %Y}}: %{{y:.2f}} per 100 visits<extra></extra>",
                                showlegend=False,
                            )
                        fig_ut.update_layout(**cl(
                            height=160 * _n_rows + 80,
                            showlegend=False,
                            margin=dict(b=40),
                        ))
                        st.plotly_chart(fig_ut, use_container_width=True,
                                        config={"displayModeBar": False})
                        # ── Imaging benchmark position insight card ──────────
                        _img_latest = (
                            _img_u[_img_u["MODALITY"].isin(_ut_mods)]
                            .sort_values("MONTH")
                            .groupby("MODALITY", as_index=False)
                            .last()
                        )
                        _img_above, _img_below = [], []
                        for _, _ir in _img_latest.iterrows():
                            _b = _BENCH.get(_ir["MODALITY"])
                            if _b:
                                if _ir["rate"] > _b["high"]:
                                    _img_above.append(_ir["MODALITY"])
                                elif _ir["rate"] < _b["low"]:
                                    _img_below.append(_ir["MODALITY"])

                        _img_parts = []
                        if _img_above:
                            _img_parts.append(
                                f'<div style="flex:1;background:#FFFBEB;border-left:3px solid #D97706;'
                                f'border-radius:4px;padding:10px 12px">'
                                f'<div style="font-size:11px;font-weight:700;color:#92400E">ABOVE BENCHMARK</div>'
                                f'<div style="font-size:12px;font-weight:600;color:#D97706;margin-top:4px">'
                                f'{", ".join(_img_above)}</div>'
                                f'<div style="font-size:11px;color:#92400E;margin-top:3px">'
                                f'Higher sessions per visit than expected. '
                                f'Investigate: high-acuity referral mix or over-ordering.</div>'
                                f'</div>'
                            )
                        if _img_below:
                            _img_parts.append(
                                f'<div style="flex:1;background:#EFF6FF;border-left:3px solid #0072CE;'
                                f'border-radius:4px;padding:10px 12px">'
                                f'<div style="font-size:11px;font-weight:700;color:#1E40AF">BELOW BENCHMARK</div>'
                                f'<div style="font-size:12px;font-weight:600;color:#0072CE;margin-top:4px">'
                                f'{", ".join(_img_below)}</div>'
                                f'<div style="font-size:11px;color:#1E40AF;margin-top:3px">'
                                f'Fewer sessions per visit than expected. '
                                f'Investigate: under-utilised equipment or referral pathway gap.</div>'
                                f'</div>'
                            )
                        if not _img_parts:
                            _img_parts.append(
                                f'<div style="flex:1;background:#F0FDF4;border-left:3px solid #16A34A;'
                                f'border-radius:4px;padding:10px 12px">'
                                f'<div style="font-size:11px;font-weight:700;color:#166534">ALL MODALITIES IN RANGE</div>'
                                f'<div style="font-size:12px;color:#166534;margin-top:4px">'
                                f'Current utilisation rates are within indicative benchmarks across all modalities.</div>'
                                f'</div>'
                            )
                        st.markdown(
                            f'<div style="background:#F4F8FC;border:1px solid #D6E4F0;border-radius:8px;'
                            f'padding:14px 18px;margin:10px 0">'
                            f'<div style="font-size:10px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
                            f'letter-spacing:1.5px;margin-bottom:10px">'
                            f'IMAGING UTILISATION — WHAT THE POSITION MEANS</div>'
                            f'<div style="display:flex;gap:10px">{"".join(_img_parts)}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            st.markdown("---")


    # ── Tab 4: Staffing ───────────────────────────────────────────────────────

    with tab4:
        if not _is_ksh_p3:
            st.info("Staffing analytics require evaluation visit data — KSH only.")
        elif not len(P["doctor_wl"]):
            st.caption("Doctor workload data not available.")
        else:
            _doc3 = _filter_epoch(P["doctor_wl"].copy(), "VISIT_MONTH").sort_values("VISIT_MONTH")

            # ── Workload trend: top 4 doctors, 12 months ─────────────────────
            st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
            section_header("Monthly Visits per Doctor — 12-Month Trend")
            _top4_names = (
                _doc3.groupby("USERNAME")["MONTHLY_VISITS"].sum()
                .nlargest(4).index.tolist()
            )
            _doc3_12mo = _doc3[_doc3["VISIT_MONTH"] >= _doc3["VISIT_MONTH"].max() - pd.DateOffset(months=11)]
            _trend_colors = [COLORS["primary"], COLORS["success"], COLORS["warning"], COLORS["coral"]]
            _fig_wl = go.Figure()
            for _i, _uname in enumerate(_top4_names):
                _ud = _doc3_12mo[_doc3_12mo["USERNAME"] == _uname]
                _ddisp = f"{_uname[0].upper()}.{_uname[1:].capitalize()}"
                _fig_wl.add_scatter(
                    x=_ud["VISIT_MONTH"], y=_ud["MONTHLY_VISITS"],
                    mode="lines+markers", name=_ddisp,
                    line=dict(color=_trend_colors[_i % 4], width=2),
                    marker=dict(size=5),
                    hovertemplate=f"<b>{_ddisp}</b><br>%{{x|%b %Y}}: %{{y:,}} visits<extra></extra>",
                )
            _fig_wl.update_layout(**cl(
                height=320, yaxis_title="Monthly visits",
                legend=dict(orientation="h", y=1.08),
                transition_duration=400,
            ))
            st.plotly_chart(_fig_wl, use_container_width=True, config={"displayModeBar": False})
            dq_note("Concentration rule fires when top doctor exceeds 40% of monthly visits.")

            # ── Conversion Rate per Doctor ────────────────────────────────────
            _conv_df = P.get("doctor_conv", pd.DataFrame())
            if len(_conv_df):
                _conv_df = _filter_epoch(_conv_df.copy(), "VISIT_MONTH")
                _conv_df.columns = _conv_df.columns.str.upper()
                if KSH_DATA_END.day < 25:
                    _partial = pd.Timestamp(KSH_DATA_END.year, KSH_DATA_END.month, 1)
                    _conv_df = _conv_df[_conv_df["VISIT_MONTH"] < _partial]
                _conv_latest_mo = _conv_df["VISIT_MONTH"].max()
                _conv_lat = (
                    _conv_df[_conv_df["VISIT_MONTH"] == _conv_latest_mo]
                    .sort_values("CONVERSION_RATE_PCT", ascending=True)
                )
                _conv_lat = _conv_lat[_conv_lat["EVALUATIONS"] >= 10].copy()
                if len(_conv_lat):
                    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
                    section_header("Facility Conversion Rate")
                    _conv_lat["display_name"] = _conv_lat["USERNAME"].apply(
                        lambda u: f"{u[0].upper()}.{u[1:].capitalize()}"
                    )
                    # Trend: evaluations (bars) vs conversion rate (line) — co-movement signal
                    # Built first so KPI uses the same unfiltered population as the chart
                    _trend = (
                        _conv_df.groupby("VISIT_MONTH")[["EVALUATIONS", "ADMISSIONS"]]
                        .sum().reset_index()
                    )
                    _trend["fac_conv_pct"] = (
                        _trend["ADMISSIONS"] / _trend["EVALUATIONS"].clip(lower=1) * 100
                    ).round(1)
                    _baseline_conv = round(
                        _trend["ADMISSIONS"].sum() / max(_trend["EVALUATIONS"].sum(), 1) * 100, 1
                    )
                    _latest_row = _trend[_trend["VISIT_MONTH"] == _conv_latest_mo]
                    _fac_conv_rate = float(_latest_row["fac_conv_pct"].iloc[0]) if len(_latest_row) else _baseline_conv

                    kpi_card(
                        "Facility Conversion Rate",
                        f"{_fac_conv_rate}%",
                        f"{pd.to_datetime(_conv_latest_mo).strftime('%b %Y')} · admissions ÷ evaluation visits",
                        COLORS["primary"],
                    )
                    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
                    _fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
                    _fig_trend.add_bar(
                        x=_trend["VISIT_MONTH"], y=_trend["EVALUATIONS"],
                        name="Evaluations", marker_color=COLORS["muted"],
                        opacity=0.6, secondary_y=False,
                        hovertemplate="%{x|%b %Y}: %{y:,} evaluations<extra></extra>",
                    )
                    _fig_trend.add_scatter(
                        x=_trend["VISIT_MONTH"], y=_trend["fac_conv_pct"],
                        name="Conversion %", mode="lines+markers",
                        line=dict(color=COLORS["primary"], width=2),
                        marker=dict(size=6),
                        secondary_y=True,
                        hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>",
                    )
                    _fig_trend.add_hline(
                        y=_baseline_conv, line_dash="dot",
                        line_color=COLORS["primary"], line_width=1,
                        secondary_y=True,
                        annotation_text=f"Baseline {_baseline_conv}%",
                        annotation_font_size=9,
                        annotation_font_color=COLORS["primary"],
                    )
                    _fig_trend.update_layout(**cl(
                        height=240,
                        legend=dict(orientation="h", y=1.12, x=0),
                        margin=dict(l=0, r=60, t=10, b=30),
                        barmode="overlay",
                    ))
                    _fig_trend.update_yaxes(title_text="Evaluations", secondary_y=False)
                    _fig_trend.update_yaxes(
                        title_text="Conversion %", secondary_y=True,
                        ticksuffix="%", rangemode="tozero",
                    )
                    st.plotly_chart(_fig_trend, use_container_width=True,
                                    config={"displayModeBar": False})
                    dq_note(
                        "Bars = monthly evaluation volume. Line = facility conversion rate. "
                        "Both moving together = demand signal. "
                        "Evaluations high but conversion drops = process signal (TAT, staffing)."
                    )
                    st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
                    section_header("Conversion Rate per Doctor — Latest Month")

                    _fig_conv = go.Figure()
                    _fig_conv.add_bar(
                        x=_conv_lat["CONVERSION_RATE_PCT"],
                        y=_conv_lat["display_name"],
                        orientation="h",
                        marker_color=COLORS["primary"],
                        text=_conv_lat.apply(
                            lambda r: f"{int(r['ADMISSIONS'])}/{int(r['EVALUATIONS'])}",
                            axis=1,
                        ),
                        textposition="outside",
                        hovertemplate="<b>%{y}</b>: %{x:.1f}% (%{text})<extra></extra>",
                        showlegend=False,
                    )
                    _fig_conv.add_vline(
                        x=_fac_conv_rate,
                        line_dash="dot", line_color=COLORS["muted"], line_width=1.5,
                        annotation_text=f"Facility avg {_fac_conv_rate:.1f}%",
                        annotation_font_size=9,
                        annotation_font_color=COLORS["muted"],
                    )
                    _fig_conv.update_layout(**cl(
                        height=max(200, len(_conv_lat) * 46),
                        xaxis_title=(
                            f"Conversion rate (%) — "
                            f"{pd.to_datetime(_conv_latest_mo).strftime('%b %Y')}"
                        ),
                        margin=dict(l=0, r=90, t=10, b=30),
                    ))
                    st.plotly_chart(_fig_conv, use_container_width=True,
                                    config={"displayModeBar": False})
                    dq_note(
                        "Conversion rate = admissions ÷ evaluation visits per doctor. "
                        "Label shows admissions / evaluations. "
                        "Doctors with fewer than 10 evaluations in the month are excluded. "
                        "Differences reflect case-mix as much as clinical decision-making — "
                        "a doctor seeing more severe cases will convert at a higher rate. "
                        "Dotted line = facility average for the month."
                    )
                    st.caption(
                        "▸ When conversion concentrates in one doctor, the facility is exposed to absence risk. "
                        "See **Causal Intelligence → CD6** for concentration analysis and absence simulation."
                    )

            # ── Peak vs Off-Peak visit load (Step 7) ─────────────────────────
            _pkbk = P["peak_bk"]
            if len(_pkbk):
                _pkbk = _pkbk.rename(columns=str.lower).sort_values("visit_month")
                # Exclude pipeline-lagged months: drop trailing rows whose total_visits
                # is < 50% of the 3-month rolling mean before them (catches partial loads
                # that aren't the calendar current month).
                if len(_pkbk) > 3:
                    _rolling_mean = _pkbk["total_visits"].iloc[:-1].tail(3).mean()
                    while len(_pkbk) > 1 and _pkbk.iloc[-1]["total_visits"] < _rolling_mean * 0.5:
                        _pkbk = _pkbk.iloc[:-1]
                if len(_pkbk):
                    _pk_row    = _pkbk.iloc[-1]
                    _pk_visits = int(_pk_row["peak_visits"])
                    _op_visits = int(_pk_row["offpeak_visits"])
                    _pk_pct    = float(_pk_row["peak_vs_offpeak_pct"])
                    _pk_mo_lbl = pd.to_datetime(_pk_row["visit_month"]).strftime("%b %Y")

                    # 3-month average peak ratio for context
                    _prior3        = _pkbk.iloc[:-1].tail(3)
                    _avg_pct_prior = float(_prior3["peak_vs_offpeak_pct"].mean()) if len(_prior3) else _pk_pct
                    _pct_delta     = _pk_pct - _avg_pct_prior
                    _delta_lbl     = (f"▲ {_pct_delta:+.0f} pp vs 3-month avg" if _pct_delta > 2
                                      else f"▼ {_pct_delta:.0f} pp vs 3-month avg" if _pct_delta < -2
                                      else f"Stable vs 3-month avg ({_avg_pct_prior:.0f}%)")

                    # What the ratio means in plain language
                    if _pk_pct >= 80:
                        _interp = (f"Morning hours carry {_pk_pct:.0f}% as many visits as all other hours combined "
                                   f"— demand is heavily front-loaded. Staffing must peak before noon.")
                    elif _pk_pct >= 50:
                        _interp = (f"Morning hours carry {_pk_pct:.0f}% as many visits as all other hours combined "
                                   f"— a clear morning concentration. Afternoon volume is materially lower.")
                    else:
                        _interp = (f"Morning hours carry {_pk_pct:.0f}% as many visits as all other hours combined "
                                   f"— load is spread more evenly than the facility average.")

                    st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
                    section_header(f"Visit Load — Peak vs Off-Peak · {_pk_mo_lbl}")
                    _pc1, _pc2, _pc3 = st.columns(3, gap="large")
                    with _pc1:
                        kpi_card("Morning Peak Visits", f"{_pk_visits:,}",
                                 "09:00–12:59", COLORS["primary"])
                    with _pc2:
                        kpi_card("Off-Peak Visits", f"{_op_visits:,}",
                                 "All other hours", COLORS["warning"])
                    with _pc3:
                        kpi_card("Peak Load Ratio", f"{_pk_pct:.0f}%",
                                 _delta_lbl, COLORS["success"])
                    st.caption(_interp)
                    dq_note(
                        "Peak = 09:00–12:59 — confirmed highest visit volume window (Inv 29). "
                        "Informant only. Months with <50% of prior 3-month average excluded (pipeline lag)."
                    )
                    st.caption(
                        "▸ This peak costs more than congestion — conversion drops 33%, TAT spikes 81%, "
                        "and 44% of deflected patients never return. See **Causal Intelligence → CD5**."
                    )



# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Patients Coming Back
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — Causal Intelligence
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Causal Intelligence":

    if facility == "TENRI":
        st.info("Causal Intelligence findings are KSH-specific — not applicable for TENRI.")
    else:
        if not st.session_state.p_causal or st.session_state.p_causal.get("_fac") != fac_key:
            with st.spinner("Loading…"):
                st.session_state.p_causal = {
                    "_fac":          fac_key,
                    "peak_ward":     q_peak_ward_dist(),
                    "doc_ward":      q_doctor_ward_share(),
                    "peak_conv":     q_peak_tat_conversion(),
                    "peak_doc_load": q_peak_doctor_load(),
                    "peak_funnel":   q_peak_patient_funnel(),
                    "doctor_conv":   q_doctor_conversion_monthly(),
                    "dialysis_ops":  q_dialysis_ops_monthly(),
                }

        P7 = st.session_state.p_causal

        st.markdown(
            '<p style="font-size:11px;font-weight:800;letter-spacing:3px;text-transform:uppercase;'
            'color:#0072CE;margin-bottom:4px">Causal Intelligence · KSH</p>',
            unsafe_allow_html=True)
        st.caption("Confirmed cross-domain findings — what the data connected across departments")
        st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)

        # ── CD5: Monday Peak — System Stress Confirmed ───────────────────────

        section_header("Monday Afternoon Peak · System Stress Across Four Dimensions (CD5)")

        _pc = P7["peak_conv"].copy()
        _pc.columns = [c.lower() for c in _pc.columns]
        _pk_row = _pc[_pc["time_bucket"].str.startswith("Peak")]
        _op_row = _pc[_pc["time_bucket"] == "Off-Peak"]

        if len(_pk_row) and len(_op_row):
            _pk = _pk_row.iloc[0]
            _op = _op_row.iloc[0]

            # ── KPI row: conversion + TAT ─────────────────────────────────
            _kc1, _kc2, _kc3, _kc4 = st.columns(4, gap="large")
            with _kc1:
                kpi_card(
                    "Conversion · Peak",
                    f"{_pk['conversion_pct']}%",
                    f"Mon 14–17h · n={int(_pk['total_evaluations']):,}",
                    COLORS["danger"],
                )
            with _kc2:
                kpi_card(
                    "Conversion · Off-Peak",
                    f"{_op['conversion_pct']}%",
                    f"All other hours · n={int(_op['total_evaluations']):,}",
                    COLORS["primary"],
                )
            with _kc3:
                kpi_card(
                    "TAT · Peak",
                    f"P50 {int(_pk['p50_tat_min'])} min",
                    f"P75 {int(_pk['p75_tat_min'])} min · n={int(_pk['valid_tat_n']):,}",
                    COLORS["danger"],
                )
            with _kc4:
                kpi_card(
                    "TAT · Off-Peak",
                    f"P50 {int(_op['p50_tat_min'])} min",
                    f"P75 {int(_op['p75_tat_min'])} min · n={int(_op['valid_tat_n']):,}",
                    COLORS["primary"],
                )

            # Private capture callout — derived from existing peak_ward data
            _pw2 = P7["peak_ward"].copy()
            _pw2.columns = [c.lower() for c in _pw2.columns]
            _pw2["ward_type"] = _pw2["ward_category"].apply(
                lambda w: "Private" if any(x in w.lower() for x in ("private", "amenity")) else "General"
            )
            _pw2_agg = _pw2.groupby(["time_bucket", "ward_type"])["admissions"].sum().reset_index()
            _pw2_tot = _pw2_agg.groupby("time_bucket")["admissions"].sum().reset_index()
            _pw2_tot.columns = ["time_bucket", "total"]
            _pw2_agg = _pw2_agg.merge(_pw2_tot, on="time_bucket")
            _pw2_agg["share"] = (_pw2_agg["admissions"] / _pw2_agg["total"] * 100).round(1)
            _priv_pk = _pw2_agg[
                (_pw2_agg["time_bucket"] == "Peak") & (_pw2_agg["ward_type"] == "Private")
            ]["share"].values
            _priv_op = _pw2_agg[
                (_pw2_agg["time_bucket"] == "Off-Peak") & (_pw2_agg["ward_type"] == "Private")
            ]["share"].values
            if len(_priv_pk) and len(_priv_op):
                st.markdown(
                    f"<div style='margin-top:8px;padding:8px 14px;background:#F8FAFC;"
                    f"border-left:3px solid #CBD5E1;border-radius:4px;font-size:12px;color:#475569'>"
                    f"Private ward capture: <b>{_priv_op[0]}%</b> off-peak → "
                    f"<b style='color:#DC2626'>{_priv_pk[0]}%</b> during peak "
                    f"· observed shift only · Sep 2024–May 2026</div>",
                    unsafe_allow_html=True,
                )

            # ── CD5 four-cost connector card ─────────────────────────────────
            st.markdown(
                '<div style="background:#FEF2F2;border:1px solid #FCA5A5;border-radius:8px;'
                'padding:14px 18px;margin:10px 0">'
                '<div style="font-size:10px;font-weight:700;color:#DC2626;text-transform:uppercase;'
                'letter-spacing:1.5px;margin-bottom:10px">MONDAY 14–17H — FOUR CONNECTED COSTS</div>'
                '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">'
                '<div style="font-size:12px;color:#991B1B">▼ <b>Conversion −33%</b> '
                '— fewer patients admitted per evaluation during peak</div>'
                '<div style="font-size:12px;color:#991B1B">▲ <b>TAT +81% median</b> '
                '— bed wait nearly doubles, 43 → 78 min</div>'
                '<div style="font-size:12px;color:#991B1B">▼ <b>Private capture −30%</b> '
                '— higher-yield admissions deflected at peak</div>'
                '<div style="font-size:12px;color:#991B1B">✕ <b>44% of peak non-admissions are permanent</b> '
                '— patients who leave during peak do not return to KSH</div>'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)

            # ── Doctor load chart ─────────────────────────────────────────
            _dl = P7["peak_doc_load"].copy()
            _dl.columns = [c.lower() for c in _dl.columns]

            # Top 4 by peak evaluations, sorted by peak pct ascending for chart display
            _dl_peak = _dl[_dl["time_bucket"].str.startswith("Peak")].copy()
            _top4 = _dl_peak.nlargest(4, "evaluations")["username"].tolist()
            _top4_sorted = (
                _dl_peak[_dl_peak["username"].isin(_top4)]
                .sort_values("pct_of_bucket")["username"]
                .tolist()
            )
            _dl_filt = _dl[_dl["username"].isin(_top4)].copy()
            _pk_dl = _dl_filt[_dl_filt["time_bucket"].str.startswith("Peak")].set_index("username")
            _op_dl = _dl_filt[_dl_filt["time_bucket"] == "Off-Peak"].set_index("username")

            _fig_dl = go.Figure()
            _fig_dl.add_bar(
                name="Off-Peak",
                x=[_op_dl.loc[d, "pct_of_bucket"] if d in _op_dl.index else 0 for d in _top4_sorted],
                y=_top4_sorted,
                orientation="h",
                marker_color=COLORS["primary"],
                opacity=0.75,
                text=[
                    f"{_op_dl.loc[d, 'pct_of_bucket']:.1f}% · {int(_op_dl.loc[d, 'evaluations']):,}"
                    if d in _op_dl.index else ""
                    for d in _top4_sorted
                ],
                textposition="outside",
                hovertemplate="<b>%{y}</b> off-peak: %{x:.1f}%<extra></extra>",
            )
            _fig_dl.add_bar(
                name="Peak · Mon 14–17h",
                x=[_pk_dl.loc[d, "pct_of_bucket"] if d in _pk_dl.index else 0 for d in _top4_sorted],
                y=_top4_sorted,
                orientation="h",
                marker_color=COLORS["danger"],
                text=[
                    f"{_pk_dl.loc[d, 'pct_of_bucket']:.1f}% · {int(_pk_dl.loc[d, 'evaluations']):,}"
                    if d in _pk_dl.index else ""
                    for d in _top4_sorted
                ],
                textposition="outside",
                hovertemplate="<b>%{y}</b> peak: %{x:.1f}%<extra></extra>",
            )
            _fig_dl.update_layout(**cl(
                barmode="group",
                height=300,
                xaxis_title="Share of evaluations in time bucket · Sep 2024–May 2026 (%)",
                xaxis_range=[0, 65],
                margin=dict(l=0, r=160, t=10, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            ))
            st.plotly_chart(_fig_dl, use_container_width=True, config={"displayModeBar": False})

            st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)

            # ── Patient funnel ────────────────────────────────────────────
            _pf = P7["peak_funnel"].copy()
            _pf.columns = [c.lower() for c in _pf.columns]
            if len(_pf):
                _pfr = _pf.iloc[0]
                _pf_total   = int(_pfr["total_non_admitted_peak"])
                _pf_never   = int(_pfr["never_returned"])
                _pf_nvr_pct = round(100 - float(_pfr["return_pct"]), 1)
                _pf_later   = int(_pfr["later_admitted"])
                _pf_lat_pct = float(_pfr["admitted_of_returned_pct"])

                _ff1, _ff2, _ff3 = st.columns(3, gap="large")
                with _ff1:
                    kpi_card(
                        "Not Admitted · Peak Window",
                        f"{_pf_total:,}",
                        "Evaluated Mon 14–17h, no admission · Sep 2024–May 2026",
                        COLORS["warning"],
                    )
                with _ff2:
                    kpi_card(
                        "Never Returned to KSH",
                        f"{_pf_nvr_pct}%",
                        f"{_pf_never:,} patients · destination unknown",
                        COLORS["danger"],
                    )
                with _ff3:
                    kpi_card(
                        "Eventually Admitted",
                        f"{_pf_lat_pct}%",
                        f"of returnees · {_pf_later:,} patients",
                        COLORS["coral"],
                    )


            with st.expander("Analysis"):
                st.markdown(
                    "- Conversion drops 33% during peak (5.9% → 3.9%) against a 2,466-evaluation window — "
                    "not a low-volume artefact.\n"
                    "- TAT nearly doubles at median (43 → 78 min). P75 rises from 173 → 196 min — "
                    "peak pushes the median into the slow zone; the upper tail was already high off-peak.\n"
                    "- Doctor load redistribution is the structural driver: lowino absorbs 49.2% of peak "
                    "evaluations (vs 17.1% off-peak) while eawando's share falls from 37.1% to 28.1%. "
                    "Peak window and facility-wide concentration risk (CD6) have different key actors.\n"
                    "- Private ward capture falls 30% (11.9% → 8.4%). Observed shift — case-mix "
                    "contribution not isolated.\n"
                    "- 44% of non-admitted peak patients never returned to KSH. Of those who returned, "
                    "only 17.3% were eventually admitted — peak non-admission is largely permanent patient "
                    "loss, not deferral. Observation window: Sep 2024–May 2026."
                )

        else:
            st.caption("Peak operational data not available.")

        st.markdown("<div style='margin-top:32px'></div>", unsafe_allow_html=True)

        # ── CD6: Physician Dependence ─────────────────────────────────────────

        section_header("Physician Dependence · Concentration Risk (CD6)")

        _dw = P7["doc_ward"].copy()
        _dw.columns = [c.lower() for c in _dw.columns]

        if len(_dw):
            # Name formatter: first char = initial, rest = surname (eawando → E. Awando)
            def _fmt_doc(u):
                u = u.strip()
                return f"{u[0].upper()}. {u[1:].capitalize()}" if len(u) > 1 else u.upper()

            # KPI: dominant doctor for latest complete month (current month may be partial)
            import datetime as _dt
            _cur_month    = _dt.date.today().replace(day=1)
            _dw_complete  = _dw[_dw["admission_month"] < _cur_month]
            _latest_month = _dw_complete["admission_month"].max() if len(_dw_complete) else _dw["admission_month"].max()
            _dw_latest    = _dw[_dw["admission_month"] == _latest_month]
            _doc_latest   = _dw_latest.groupby("username")["admissions"].sum()
            _fac_latest   = _doc_latest.sum()
            _dom_doc      = _doc_latest.idxmax()
            _dom_pct      = round(float(_doc_latest.max()) / float(_fac_latest) * 100, 1)
            _month_lbl    = pd.Timestamp(_latest_month).strftime("%b %Y")

            kpi_card(
                "Physician Dependence",
                _fmt_doc(_dom_doc),
                f"{_dom_pct}% of facility admissions · {_month_lbl}",
                COLORS["danger"] if _dom_pct >= 30 else COLORS["warning"],
            )

            st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

            # Chart: aggregate all months for full historical ward concentration
            _dw_agg   = _dw.groupby(["username", "ward_name"])["admissions"].sum().reset_index()
            _ward_tot = _dw_agg.groupby("ward_name")["admissions"].sum().reset_index()
            _ward_tot.columns = ["ward_name", "ward_total"]
            _dw_agg   = _dw_agg.merge(_ward_tot, on="ward_name")
            _dw_agg["pct"] = (_dw_agg["admissions"] / _dw_agg["ward_total"] * 100).round(1)

            _aw = _dw_agg[_dw_agg["username"] == _dom_doc].sort_values("pct", ascending=True)

            if len(_aw):
                _aw["display_ward"] = _aw["ward_name"]
                _fig_cd6 = go.Figure()
                _fig_cd6.add_bar(
                    x=_aw["pct"],
                    y=_aw["display_ward"],
                    orientation="h",
                    marker_color=[
                        COLORS["danger"] if p >= 40 else COLORS["warning"]
                        for p in _aw["pct"]
                    ],
                    text=_aw["pct"].apply(lambda p: f"{p:.0f}%"),
                    textposition="outside",
                    hovertemplate=f"<b>%{{y}}</b>: {_fmt_doc(_dom_doc)} %{{x:.1f}}% of ward admissions<extra></extra>",
                    showlegend=False,
                )
                _fig_cd6.add_vline(
                    x=40, line_dash="dot", line_color=COLORS["warning"], line_width=1.5,
                    annotation_text="40% concentration threshold",
                    annotation_font_size=9, annotation_font_color=COLORS["warning"],
                )
                _fig_cd6.update_layout(**cl(
                    height=max(240, len(_aw) * 44),
                    xaxis_title=f"% of ward admissions attributed to {_fmt_doc(_dom_doc)}",
                    xaxis_range=[0, 70],
                    margin=dict(l=0, r=60, t=10, b=30),
                ))
                st.plotly_chart(_fig_cd6, use_container_width=True, config={"displayModeBar": False})

                _aw_max = float(_aw["pct"].max())
                _aw_min = float(_aw["pct"].min())
                st.caption(
                    f"One doctor absence = facility-wide intake drop across all 7 wards simultaneously — not one ward. "
                    f"J.Ogutu (14–17% per ward) is the only named backup. "
                    f"Review: Medical Director."
                )
                with st.expander("Analysis"):
                    st.markdown(
                        "- E.Awando evaluates **34–46% of admissions in every ward** — the risk is facility-wide, not concentrated in one area.\n"
                        "- A single absence triggers intake reduction across all wards with no pre-defined cover.\n"
                        "- M.Akinyi's departure (Dec 2025) added ~57% volume onto E.Awando silently — no flag fired until months later.\n"
                        "- J.Ogutu is the only confirmed backup but carries a distant 14–17% per ward.\n"
                        "- Private wards are most exposed: fewest distinct evaluators and no fallback when E.Awando is absent."
                    )

                # ── Simulated absence impact ──────────────────────────────────
                st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
                section_header("Simulated Impact — If E.Awando Is Absent")
                _sim_raw = P7.get("doctor_conv", pd.DataFrame())
                if len(_sim_raw):
                    _sim = _sim_raw.copy()
                    _sim.columns = _sim.columns.str.upper()
                    _sim_total_eval = int(_sim["EVALUATIONS"].sum())
                    # Admissions from _dw (COUNT DISTINCT ia.id per doctor) — no fan-out
                    _sim_total_adm  = int(_dw["admissions"].sum())
                    _sim_dom_adm    = int(_dw[_dw["username"] == _dom_doc]["admissions"].sum())
                    _actual_rate    = round(_sim_total_adm / max(_sim_total_eval, 1) * 100, 1)
                    _sim_rate       = round((_sim_total_adm - _sim_dom_adm) / max(_sim_total_eval, 1) * 100, 1)
                    _drop           = round(_actual_rate - _sim_rate, 1)
                    _s1, _s2 = st.columns(2, gap="large")
                    with _s1:
                        kpi_card("Actual Conversion Rate",
                                 f"{_actual_rate}%",
                                 "Via evaluation pathway · all doctors",
                                 COLORS["primary"])
                    with _s2:
                        kpi_card(f"Without {_fmt_doc(_dom_doc)}",
                                 f"{_sim_rate}%",
                                 f"−{_drop} percentage points · simulation",
                                 COLORS["danger"])
                    st.caption(
                        f"Simulation: {_fmt_doc(_dom_doc)}'s {_sim_dom_adm:,} admissions removed. "
                        "Evaluation volume held constant. This is a modelled estimate, not a measured outcome."
                    )
            else:
                st.caption("E.Awando ward share data not found — username may differ.")
        else:
            st.caption("Doctor ward data not available.")

        st.markdown("<div style='margin-top:32px'></div>", unsafe_allow_html=True)

        # ── CD12: Renal Patient Safety ────────────────────────────────────────

        section_header("Renal Pathway — Critical Patients Leaving Without Admission (CD12)")

        _r1, _r2, _r3 = st.columns(3, gap="large")
        with _r1:
            kpi_card("Critical Creatinine Patients", "126",
                     "Since Jan 2024", COLORS["danger"])
        with _r2:
            kpi_card("Not Admitted at Index Visit", "41%",
                     "Seen and discharged without admission", COLORS["warning"])
        with _r3:
            kpi_card("Never Returned to KSH", "28%",
                     "Destination unknown", COLORS["coral"])

        st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

        _r4, _r5 = st.columns(2, gap="large")
        with _r4:
            kpi_card("Returned to KSH", "72%",
                     "Most came back outpatient only", COLORS["primary"])
        with _r5:
            kpi_card("Dialysis Routing Gap", "96.8%",
                     "122 of 126 patients never billed for dialysis", COLORS["danger"])

        _ci_dial_raw = P7.get("dialysis_ops", pd.DataFrame())
        _ci_dial = (
            _ci_dial_raw[~_ci_dial_raw["IS_PARTIAL_MONTH"]]
            if len(_ci_dial_raw) and "IS_PARTIAL_MONTH" in _ci_dial_raw.columns
            else _ci_dial_raw
        )
        if len(_ci_dial):
            _ci_peak_row = _ci_dial.nlargest(1, "SESSIONS_BILLED")
            _ci_peak     = int(_ci_peak_row["SESSIONS_BILLED"].iloc[0])
            _ci_headroom = (264 - _ci_peak) * 10_650
            _ci_val_str  = f"KES {_ci_headroom / 1e6:.2f}M/month"
            _ci_sub_str  = f"(264 − {_ci_peak}) × KES 10,650 NHIF tariff"
            st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
            _r6, _ = st.columns(2, gap="large")
            with _r6:
                kpi_card("Unused Dialysis Capacity — Revenue Headroom",
                         _ci_val_str, _ci_sub_str, COLORS["warning"])

        st.caption(
            "Non-admission is a clinical decision, not patient-driven — DAMA rate matches the facility baseline. "
            "The dialysis programme is operational (80–135 sessions/month, NHIF-funded), "
            "but 96.8% of critical creatinine patients are not being routed into it. "
            "Escalate to Clinical Director and Renal Lead."
        )
        dq_note("Patient safety finding — not an operational alert. Analysis covers January 2024 onward.")
        with st.expander("Analysis"):
            st.markdown(
                "- 41% of critical creatinine patients were discharged without admission at their index visit. "
                "This is a clinical pathway decision — DAMA rate matches the facility-wide baseline, so patients are not leaving against advice.\n"
                "- Of those who left without admission and returned: a quarter required admission on their return visit — delayed escalation.\n"
                "- Referral patients waited an average of **18 days** before transfer. No rapid referral pathway confirmed in data.\n"
                "- 1 patient death within 24 hours of admission. 28% never returned — death, transfer, or lost to follow-up.\n"
                "- Every patient in this cohort had **no dialysis billing record** at KSH. "
                "The programme served 80–135 sessions/month (NHIF-funded) — the clinical referral pathway "
                "from critical creatinine detection to dialysis enrolment is not functioning."
            )

        # ── Go deeper — suggested chat questions ─────────────────────────────

        st.markdown("<div style='margin-top:36px'></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:11px;font-weight:800;letter-spacing:2px;text-transform:uppercase;'
            'color:#6B8CAE;margin-bottom:6px">Go Deeper · Operations Intelligence Chat</p>',
            unsafe_allow_html=True
        )
        st.caption(
            "The investigations behind these findings covered 12 cross-domain questions — bed occupancy, "
            "insured patient routing, demand vs conversion, and five disproved hypotheses. "
            "Ask the Operations Intelligence assistant for the full narrative."
        )
        st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)
        st.markdown("""
<style>
a.cq { display:block; text-decoration:none;
  border:1.5px solid #D6E4F0; border-radius:8px;
  padding:10px 14px; margin-bottom:8px;
  font-size:12px; color:#003467; line-height:1.5;
  transition:border-color .15s, box-shadow .15s; }
a.cq:hover { border-color:#0072CE;
  box-shadow:0 2px 8px rgba(0,114,206,0.12); color:#003467; }
a.cq .cq-label { font-size:10px; font-weight:800; color:#0072CE;
  letter-spacing:1px; text-transform:uppercase; display:block;
  margin-bottom:3px; }
</style>""", unsafe_allow_html=True)

        _sq_cols = st.columns(2, gap="large")
        _suggested = [
            "Why do insured patients mostly go to general wards instead of private?",
            "What did the investigation find about bed occupancy across all wards?",
            "Is low ward occupancy a capacity problem or a demand problem?",
            "What hypotheses were tested and disproved in the causal investigations?",
            "What is unusual about Private Female ward compared to the others?",
            "What happens to critical renal patients who were never admitted at KSH?",
        ]
        for _i, _q in enumerate(_suggested):
            _href = f"{CHAT_URL}/?q={urllib.parse.quote_plus(_q)}"
            with _sq_cols[_i % 2]:
                st.markdown(
                    f'<a class="cq" href="{_href}" target="_blank">'
                    f'<span class="cq-label">Ask the chat →</span>{_q}'
                    f'</a>',
                    unsafe_allow_html=True
                )


# ══════════════════════════════════════════════════════════════════════════════
# READM_HIDDEN — page removed from navigation. This block is unreachable until re-enabled.
# To restore: un-comment "Readmissions" in option_menu options + icons lists above,
# and un-comment q_readmission_* imports at the top of the file.

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
                _add_rolling_mean(fig, sub["ADMISSION_MONTH"],
                                  sub["READMISSION_30DAY_RATE_PCT"],
                                  name="3-mo avg", color=COLORS["muted"])
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
                    if _mm_note:
                        dq_note(_mm_note.strip())

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


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Predictive Analytics
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Predictive Analytics":

    st.markdown(
        '<p style="font-size:11px;font-weight:800;letter-spacing:3px;text-transform:uppercase;'
        'color:#0072CE;margin-bottom:4px">Private Hospitals · Predictive Analytics</p>',
        unsafe_allow_html=True)
    st.caption(f"{fac_name} — admission demand projections, ward planning, model quality")
    st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # KSH — Prophet (probabilistic, daily grain, monthly display)
    # ══════════════════════════════════════════════════════════════════════════

    if facility == "KISUMU_CLEAN":

        # ── Model Controls (always visible — retrain + reload) ─────────────────
        with st.expander("Model Controls", expanded=False):
            # Read retrain status from file — no HTTP call on every rerun
            _rs = {}
            if _RETRAIN_STATUS_FILE.exists():
                try:
                    with open(_RETRAIN_STATUS_FILE, encoding="utf-8") as _f:
                        _rs = json.load(_f)
                except Exception:
                    pass

            _rs_status = _rs.get("status", "never_run")
            _rs_color  = {"success": "#22C55E", "running": "#F59E0B",
                          "error": "#EF4444"}.get(_rs_status, "#6B8CAE")
            _rs_time   = str(_rs.get("completed_at", _rs.get("started_at", "—")))[:19]
            st.markdown(
                f'<div style="font-size:12px;margin-bottom:12px">'
                f'<span style="color:{_rs_color};font-weight:700">● {_rs_status.upper()}</span>'
                f'<span style="color:#6B8CAE;margin-left:12px">{_rs_time}</span>'
                + (f'<span style="color:#6B8CAE;margin-left:12px">run: {_rs["run_id"][:8]}…</span>'
                   if "run_id" in _rs else "")
                + '</div>',
                unsafe_allow_html=True,
            )

            _mc1, _mc2 = st.columns(2)
            with _mc1:
                if st.button("Trigger Retrain", key="p6_retrain_btn", use_container_width=True):
                    try:
                        _resp = _requests.post(_DJANGO_RETRAIN_URL, timeout=5)
                        _rd   = _resp.json()
                        if _rd.get("status") == "accepted":
                            st.success("Retrain started. Takes ~60–90s. Click Reload when done.")
                        elif _rd.get("status") == "already_running":
                            st.warning("Retrain already in progress — check status above.")
                        else:
                            st.error(str(_rd))
                    except Exception as _e:
                        st.error(f"Django not reachable at {_DJANGO_RETRAIN_URL}: {_e}")
            with _mc2:
                if st.button("Reload Forecast", key="p6_reload_btn", use_container_width=True):
                    st.session_state.p6_ksh = {}
                    st.rerun()

            st.caption(
                "Retrain pulls fresh data, refits Prophet, and writes a new cache. "
                "Run after each data load. Reload picks up the new cache without restarting Streamlit."
            )

        # Load contract once per session (facility-keyed cache)
        _contract_error = None
        if not st.session_state.p6_ksh:
            with st.spinner("Loading forecast…"):
                try:
                    st.session_state.p6_ksh = _build_forecast_contract(_FORECAST_CACHE)
                except Exception as _e:
                    _contract_error = str(_e)

        if _contract_error:
            st.warning(_contract_error)
            st.info("Use **Model Controls → Trigger Retrain**, wait ~60–90s, then click **Reload Forecast**.")
            st.stop()

        _ct      = st.session_state.p6_ksh
        _fc_rows = _ct["forecast"]
        _metrics = _ct["metrics"]
        _wmape    = _metrics.get("wmape")
        _coverage = _metrics.get("coverage")
        _wmape_str    = f"{_wmape:.1f}%" if _wmape is not None else "—"
        _coverage_str = f"{_coverage:.0f}%" if _coverage is not None else "—"
        _gen_date = str(_ct.get("generated_at", ""))[:10]

        # ── Model quality badge (inline, not a card) ───────────────────────────
        st.markdown(
            f'<div style="font-size:10px;color:#6B8CAE;margin-bottom:20px">'
            f'Probabilistic · {_ct.get("model_version", "—")} &nbsp;·&nbsp; '
            f'Coverage {_coverage_str} &nbsp;·&nbsp; WMAPE {_wmape_str} &nbsp;·&nbsp; '
            f'Updated {_gen_date}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Three month trajectory cards ───────────────────────────────────────
        _TREND_THRESHOLD = 0.05   # <5% = Stable; ≥5% = Rising / Falling

        def _trend(current, prior):
            if prior is None or prior == 0:
                return None, None, COLORS["muted"]
            pct = (current - prior) / prior
            if abs(pct) < _TREND_THRESHOLD:
                return "Stable", None, COLORS["muted"]
            elif pct > 0:
                return f"Rising +{abs(pct)*100:.0f}%", pct, COLORS["warning"]
            else:
                return f"Falling {pct*100:.0f}%", pct, "#5DADE2"

        _mc = st.columns(len(_fc_rows)) if _fc_rows else []
        _prior_pt = None
        for _col, _row in zip(_mc, _fc_rows):
            _pt     = _row["point"]
            _mo_lbl = pd.Timestamp(_row["forecast_month"]).strftime("%b %Y").upper()
            _lo     = _row.get("low_approx")
            _hi     = _row.get("high_approx")
            _range_html = (
                f'<div style="font-size:10px;color:#6B8CAE;margin-top:4px">'
                f'~{int(_lo):,} – {int(_hi):,}</div>'
                if _lo is not None and _hi is not None else ""
            )
            _trend_lbl, _trend_pct, _trend_color = _trend(_pt, _prior_pt)
            _trend_html = (
                f'<div style="font-size:11px;font-weight:700;color:{_trend_color};margin-top:6px">'
                f'{_trend_lbl}</div>'
                if _trend_lbl else ""
            )
            with _col:
                st.markdown(
                    f'<div style="background:#F4F8FC;border:1px solid #D6E4F0;border-radius:8px;'
                    f'padding:20px 16px;text-align:center">'
                    f'<div style="font-size:10px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
                    f'letter-spacing:1.5px;margin-bottom:10px">{_mo_lbl}</div>'
                    f'<div style="font-size:36px;font-weight:800;color:{COLORS["primary"]};line-height:1">'
                    f'{int(round(_pt)):,}</div>'
                    f'{_trend_html}'
                    f'{_range_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            _prior_pt = _pt

        st.markdown("<div style='margin-bottom:4px'></div>", unsafe_allow_html=True)
        dq_note(
            "Range = conservative approximation from daily interval sums — wider than true monthly uncertainty. "
            "Month-to-month changes <5% are labelled Stable regardless of direction. "
            "Yearly seasonality weakly identified at 1.5 training cycles — treat trajectory directionally."
        )
        st.markdown("<div style='margin-bottom:24px'></div>", unsafe_allow_html=True)

        # ── Bar chart — 3-month trajectory ────────────────────────────────────

        _bar_labels = [
            pd.Timestamp(r["forecast_month"]).strftime("%b %Y") for r in _fc_rows
        ]
        _bar_vals   = [int(round(r["point"])) for r in _fc_rows]

        _bar_colors = []
        _prev = None
        for _v in _bar_vals:
            if _prev is None:
                _bar_colors.append(COLORS["primary"])
            elif (_v - _prev) / max(_prev, 1) >= _TREND_THRESHOLD:
                _bar_colors.append(COLORS["warning"])
            elif (_v - _prev) / max(_prev, 1) <= -_TREND_THRESHOLD:
                _bar_colors.append("#5DADE2")
            else:
                _bar_colors.append(COLORS["primary"])
            _prev = _v

        _fig_bar = go.Figure()
        _fig_bar.add_bar(
            x=_bar_labels,
            y=_bar_vals,
            marker_color=_bar_colors,
            text=[f"{v:,}" for v in _bar_vals],
            textposition="outside",
            textfont=dict(size=13, color=COLORS["primary"], family="Montserrat"),
            hovertemplate="%{x}: %{y:,} admissions<extra></extra>",
            width=0.45,
        )
        _fig_bar.update_layout(**cl(
            height=260,
            yaxis=dict(visible=False),
            xaxis=dict(tickfont=dict(size=12, family="Montserrat", color=COLORS["primary"])),
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=20),
            plot_bgcolor="#FFFFFF",
        ))
        st.plotly_chart(_fig_bar, use_container_width=True, config={"displayModeBar": False})

        # ── Model health expander ──────────────────────────────────────────────

        st.markdown("<div style='margin-top:4px'></div>", unsafe_allow_html=True)
        with st.expander("Model Health", expanded=False):
            _mh_rows = [
                {"Series": "KSH · Facility", "Model": "Prophet (probabilistic)",
                 "WMAPE": _wmape_str, "Coverage (90%)": _coverage_str,
                 "Grain": "Daily", "Status": "✓ Champion"},
            ]
            st.dataframe(pd.DataFrame(_mh_rows), hide_index=True, use_container_width=True)
            dq_note(
                f"Model version: {_ct.get('model_version', '—')}  ·  "
                f"Contract version: {_ct.get('contract_version', '—')}  ·  "
                f"Generated: {str(_ct.get('generated_at', '—'))[:19]}"
            )

    # ══════════════════════════════════════════════════════════════════════════
    # TENRI — Holt's Linear Trend (deterministic, monthly grain) — unchanged
    # ══════════════════════════════════════════════════════════════════════════

    else:

        if not st.session_state.p6:
            with st.spinner("Computing demand forecasts…"):
                df_hist, df_fcast = get_forecast()
                st.session_state.p6 = {"hist": df_hist, "fcast": df_fcast}

        df_hist  = st.session_state.p6["hist"]
        df_fcast = st.session_state.p6["fcast"]

        fac_label = "TENRI"

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

        h_fac_all = df_hist[
            (df_hist["facility"] == fac_label) & (df_hist["ward"] == "Facility")
        ]
        hist_mean = float(h_fac_all["admissions"].mean()) if len(h_fac_all) else None

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

        if model_t == "holts" and fac_mape is not None:
            confidence  = max(0.0, 100.0 - fac_mape)
            gauge_color = (COLORS["success"] if confidence >= 90
                           else COLORS["warning"] if confidence >= 85
                           else COLORS["danger"])
        else:
            confidence  = None
            gauge_color = COLORS["muted"]

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
                    title={"text": "Holdout Accuracy (100 − MAPE) · Deterministic",
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

        tab1, tab2 = st.tabs(["◉  Facility Forecast", "△  Ward Demand"])

        with tab1:
            section_header(f"{fac_label} — Facility-Level Forecast (Holt's Linear Trend · Deterministic)")
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

        with tab2:
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

            from facility_utilization.m1_ward_forecast import VALIDATED_DATE, RETRAIN_DATE
            dq_note(
                f"Last validated: {VALIDATED_DATE.strftime('%Y-%m-%d')}  ·  "
                f"Retrain recommended by: {RETRAIN_DATE.strftime('%Y-%m-%d')}  ·  "
                f"Holdout = last 3 months per facility  ·  Acceptance threshold: MAPE < 15%"
            )

