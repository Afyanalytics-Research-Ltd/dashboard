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
from facility_utilization.notifier import send_digest, get_recipients, write_current_notices
from facility_utilization.queries import (
    q_overview_gap, q_overview_alerts,
    q_leakage_gap, q_leakage_submission_rate, q_leakage_ksh_dispatch_trend,
    q_leakage_aging_dist, q_leakage_recovery_priority,
    q_theatre_trend, q_theatre_by_type, q_theatre_emergency_tat,
    q_theatre_non_completion, q_theatre_status_breakdown,
    q_theatre_procedure_rates, q_theatre_cur_month_by_theatre,
    q_theatre_procedures, q_theatre_procedures_monthly, q_theatre_trend_by_theatre,
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
    q_admission_tat_dow, q_discharge_tat, q_discharge_dow,
    q_revpab_private_monthly,
    q_peak_tat_conversion, q_peak_doctor_load, q_peak_patient_funnel,
    q_data_freshness,
    q_dialysis_ops_monthly,
    q_lab_morning_completion, q_lab_tat_monthly,
    q_lab_tat_monthly_clean, q_lab_tat_dow_clean,
    q_lab_tat_by_test, q_pharmacy_wait_dow,
    q_lab_flow_delta, q_lab_handoff_delta, q_lab_weekly_trend,
    q_stage_wait_delta, q_pharmacy_tat, q_lab_utilization_delta,
    q_lab_result_volume_monthly,
    q_lab_downstream_monthly, q_lab_to_bed_monthly,
    q_patient_flow_transitions, q_patient_flow_dow, q_lab_tat_dow,
    q_opd_kpi_28d, q_patient_journey_sankey, q_rpt_stage_wait,
    q_pharmacy_source_split, q_pharmacy_hour_of_day,
    q_pharmacy_monthly_tat, q_pharmacy_wait_dist,
    q_opd_spine_summary, q_opd_monthly_volume, q_opd_dow_visits,
    q_opd_peak_band_tat, q_opd_hourly_tat, q_opd_weekly_pressure,
    q_opd_spillover_summary, q_opd_flagged_heatmap,
    q_opd_daily_28d,
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

@st.cache_data(ttl=3600)
def _load_data_freshness():
    df = q_data_freshness()
    out = {}
    for _, row in df.iterrows():
        d = pd.to_datetime(row["MAX_DATE"])
        out[row["FACILITY"]] = f"{d.day} {d.strftime('%b %Y')}"
    return out

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
_CD12_WATCH    = 50.0   # non-admission rate % above this = WATCH (min 3 critical events)
_CD12_CRIT     = 65.0   # non-admission rate % above this = CRITICAL
_CD12_MIN_EVTS = 3      # minimum critical events required — below this the rate is noise

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


def _flow_card_html(title, metric_value, metric_label,
                    badge_text=None, badge_color=None, is_pending=False):
    """Patient journey stage card — center-aligned, top-border accent, optional status badge."""
    if badge_text and badge_color:
        badge_html = (
            f'<div style="margin-top:10px">'
            f'<span style="background:{badge_color}18;border:1px solid {badge_color}50;'
            f'color:{badge_color};font-size:10px;font-weight:700;padding:3px 10px;'
            f'border-radius:12px">{badge_text}</span></div>'
        )
        top_border = f"border-top:4px solid {badge_color}"
    elif is_pending:
        badge_html = (
            '<div style="margin-top:10px">'
            '<span style="background:#F4F8FC;border:1px solid #C5D8EC;color:#9BAEC8;'
            'font-size:10px;font-weight:600;padding:3px 10px;border-radius:12px">'
            'Data pending</span></div>'
        )
        top_border = "border-top:4px solid #D6E4F0"
    else:
        badge_html = ""
        top_border = "border-top:4px solid #D6E4F0"
    return (
        f'<div style="background:#F4F8FC;border:1px solid #D6E4F0;{top_border};'
        f'border-radius:10px;padding:24px 14px 20px;text-align:center;min-height:180px">'
        f'<div style="font-size:9px;font-weight:700;color:#9BAEC8;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:10px">{title}</div>'
        f'<div style="font-size:30px;font-weight:800;color:#003467;line-height:1.1">{metric_value}</div>'
        f'<div style="font-size:10px;color:#9BAEC8;margin-top:6px">{metric_label}</div>'
        f'{badge_html}'
        f'</div>'
    )


_h_arrow_html = (
    '<div style="padding-top:28px;text-align:center;color:#C5D8EC;'
    'font-size:28px;font-weight:300;line-height:1">›</div>'
)


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
    _freshness   = _load_data_freshness()
    _data_start  = "Sep 2024" if facility == "KISUMU_CLEAN" else "Jan 2022"
    _data_end    = _freshness.get(facility, "—")
    st.markdown(
        f'<div style="font-size:11px;font-weight:800;color:#003467;text-transform:uppercase;'
        f'letter-spacing:1.5px;padding:2px 0 2px">Private Hospitals</div>'
        f'<div style="font-size:10px;color:#6B8CAE;padding-bottom:10px;'
        f'border-bottom:1px solid #D6E4F0;margin-bottom:8px">{fac_name}</div>'
        f'<div style="font-size:10px;background:#EEF4FB;border-radius:5px;'
        f'padding:6px 9px;margin-bottom:10px;color:#4A6B8A">'
        f'<span style="font-weight:700">Data:</span> {_data_start} &mdash; {_data_end}</div>',
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

    # ── Abbreviations ─────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
    _ABBREVS = [
        ("BOR",  "Bed Occupancy Rate",      "% of beds occupied at a given time. High BOR = constrained capacity."),
        ("BTR",  "Bed Turnover Rate",        "Admissions per bed per month. Low BTR = beds are underused."),
        ("BTI",  "Bed Turnover Interval",    "Days a bed sits idle between patients. High BTI = slow throughput."),
        ("LOS",  "Length of Stay",           "Days a patient remains admitted. Prolonged LOS blocks bed availability."),
        ("TAT",  "Turnaround Time",          "Elapsed time between two clinical events (e.g. reception → doctor)."),
        ("OPD",  "Outpatient Department",    "Walk-in visits — patient seen and discharged same day, not admitted."),
        ("P50",  "50th Percentile (Median)", "Half of cases are faster, half slower. Robust to outliers."),
        ("P75",  "75th Percentile",          "75% of cases fall below this value. Used as pressure threshold."),
        ("P90",  "90th Percentile",          "90% of cases fall below this value. Captures the slow tail."),
        ("MoM",  "Month over Month",         "Change compared to the previous calendar month."),
        ("DOW",  "Day of Week",              "Monday–Sunday breakdown of activity patterns."),
        ("CDC",  "Change Data Capture",      "Pipeline that syncs live hospital records into the analytics layer."),
    ]
    _ab_rows = "".join(
        f'<div style="padding:6px 0;border-bottom:1px solid #EBF3FB">'
        f'<div style="display:flex;align-items:baseline;gap:6px">'
        f'<span style="font-size:10px;font-weight:800;color:#0072CE;min-width:36px;flex-shrink:0">{ab}</span>'
        f'<span style="font-size:10px;font-weight:700;color:#003467">{full}</span>'
        f'</div>'
        f'<div style="font-size:9px;color:#6B8CAE;margin-top:2px;padding-left:42px">{defn}</div>'
        f'</div>'
        for ab, full, defn in _ABBREVS
    )
    st.markdown(
        f'<details style="margin-bottom:4px">'
        f'<summary style="font-size:9px;font-weight:800;color:#0072CE;text-transform:uppercase;'
        f'letter-spacing:2px;padding-bottom:8px;border-bottom:1px solid #D6E4F0;'
        f'cursor:pointer;list-style:none">Abbreviation Guide</summary>'
        f'<div style="font-size:10px;margin-top:8px">{_ab_rows}</div>'
        f'</details>',
        unsafe_allow_html=True,
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
                "opd_28d_ov":     q_opd_daily_28d()                       if _is_ksh else pd.DataFrame(),
                "btr_bti":         q_btr_bti_monthly()           if _is_ksh else pd.DataFrame(),
                "adm_tat_monthly": q_admission_tat_monthly()    if _is_ksh else pd.DataFrame(),
                "revpab_priv":     q_revpab_private_monthly()   if _is_ksh else pd.DataFrame(),
                "cd12_rate":       q_cd12_monthly_rate()        if _is_ksh else pd.DataFrame(),
                "imaging_alert":   q_imaging_trend("KISUMU_CLEAN") if _is_ksh else pd.DataFrame(),
                "stage_wait":      q_stage_wait_delta()            if _is_ksh else pd.DataFrame(),
                "pharm_tat":       q_pharmacy_tat()                if _is_ksh else pd.DataFrame(),
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
        if KSH_DATA_END.day < 25:
            _th_partial = pd.Timestamp(KSH_DATA_END.year, KSH_DATA_END.month, 1)
            th = th[pd.to_datetime(th["SESSION_MONTH"]) < _th_partial]
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
        if _is_ksh and KSH_DATA_END.day < 25:
            _lab_partial = pd.Timestamp(KSH_DATA_END.year, KSH_DATA_END.month, 1)
            _lab_df = _lab_df[pd.to_datetime(_lab_df["LAB_MONTH"]) < _lab_partial]

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
        if KSH_DATA_END.day < 25:
            _btr_partial = pd.Timestamp(KSH_DATA_END.year, KSH_DATA_END.month, 1)
            _btr_bti_alert_df = _btr_bti_alert_df[
                pd.to_datetime(_btr_bti_alert_df["month"]) < _btr_partial
            ]
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
    _rv_latest = _rv_rolling_avg = _rv_mo_lbl = _rv_latest_adm = None
    if _is_ksh and len(_revpab_priv_df):
        _revpab_priv_df.columns = _revpab_priv_df.columns.str.lower()
        _revpab_priv_df = _revpab_priv_df[
            _revpab_priv_df["admission_month"].astype(str) != _OCT_2025_GAP
        ].sort_values("admission_month")
        if KSH_DATA_END.day < 25:
            _rv_partial = pd.Timestamp(KSH_DATA_END.year, KSH_DATA_END.month, 1)
            _revpab_priv_df = _revpab_priv_df[
                pd.to_datetime(_revpab_priv_df["admission_month"]) < _rv_partial
            ]
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
    # Re-uses P["doctor_wl"]. Tracks eawando, lowino, jogutu only (makinyi departed Jan 2026).
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
    _cd12_rate_v = _cd12_total = _cd12_mo_lbl = None
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
    _img_latest_sess = _img_pct_of_avg = _img_mo_lbl = _img_rolling_avg = None
    if _is_ksh and len(_img_alert_df):
        _img_alert_df.columns = _img_alert_df.columns.str.lower()
        _img_ct = _img_alert_df[_img_alert_df["modality"] == "CT / Angio"].copy()
        _img_current_mo = pd.Timestamp.today().replace(day=1)
        _img_cutoff = (
            pd.Timestamp(KSH_DATA_END.year, KSH_DATA_END.month, 1)
            if KSH_DATA_END.day < 25 else _img_current_mo
        )
        _img_ct = _img_ct[
            (pd.to_datetime(_img_ct["revenue_month"]) < min(_img_cutoff, _img_current_mo)) &
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
        'color:#0072CE;margin-bottom:8px">Private Hospitals · The Business Today</p>',
        unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:12px;color:#6B8CAE;margin-bottom:20px;line-height:1.7">'
        'KSH operational intelligence — beds, theatre, lab, imaging, and outpatient flow tracked in one view. '
        'Metrics update automatically as data refreshes; alerts fire only when clinical or financial thresholds are crossed. '
        'Each alert names the signal, the gap, and who to flag.'
        '</p>',
        unsafe_allow_html=True)

    # ── Patient Journey (KSH only) ───────────────────────────────────────────

    if _is_ksh:
        section_header("Patient Journey", margin_top=8)

        _sw = P["stage_wait"].copy() if len(P["stage_wait"]) else pd.DataFrame()

        def _sw_val(stage, period, col):
            if not len(_sw):
                return None
            r = _sw[(_sw["STAGE"] == stage) & (_sw["PERIOD"] == period)]
            return r.iloc[0][col] if len(r) else None

        def _journey_badge(cur, prv, high_thresh, crit_thresh=None):
            """Return (badge_text, badge_color) encoding delta + status."""
            if cur is None:
                return None, None
            if crit_thresh and cur >= crit_thresh:
                status_txt, badge_col = "Bottleneck", COLORS["danger"]
            elif cur >= high_thresh:
                status_txt, badge_col = "Elevated", COLORS["warning"]
            else:
                status_txt, badge_col = "Normal", COLORS["success"]
            if prv is not None:
                d = int(cur) - int(prv)
                arrow = "↑" if d > 0 else "↓"
                delta_str = f" · {arrow} {abs(d)} min"
            else:
                delta_str = ""
            return f"{status_txt}{delta_str}", badge_col

        # Reception — last 28 days OPD visits with delta vs prior 28
        _rec_visits = None
        _rec_delta  = None
        _opd_28_ov = P.get("opd_28d_ov", pd.DataFrame()).copy()
        if len(_opd_28_ov):
            _opd_28_ov.columns = [c.upper() for c in _opd_28_ov.columns]
            _ov_last  = _opd_28_ov[_opd_28_ov["PERIOD"] == "last28"]
            _ov_prior = _opd_28_ov[_opd_28_ov["PERIOD"] == "prior28"]
            if len(_ov_last):
                _rec_visits = int(_ov_last["DAILY_VISITS"].sum())
            if len(_ov_prior):
                _rec_prior = int(_ov_prior["DAILY_VISITS"].sum())
                if _rec_visits is not None:
                    _rec_delta = _rec_visits - _rec_prior
        _rec_arrow = ("↑" if _rec_delta > 0 else "↓") if _rec_delta is not None else ""
        _rec_val   = f"{_rec_visits:,}" if _rec_visits else "—"
        _rec_label = f"OPD visits · last 28 days{f' · {_rec_arrow} {abs(_rec_delta)} vs prior 28' if _rec_delta is not None else ''}"

        # Doctor queue
        _doc_cur = _sw_val("doctor",    "last_28",  "MEDIAN_WAIT_MIN")
        _doc_prv = _sw_val("doctor",    "prior_28", "MEDIAN_WAIT_MIN")
        _doc_val = f"{int(_doc_cur)} min" if _doc_cur else "—"
        _doc_badge, _doc_bcol = _journey_badge(_doc_cur, _doc_prv,
                                                high_thresh=20, crit_thresh=45)

        # Lab queue
        _lab_cur = _sw_val("laboratory", "last_28",  "MEDIAN_WAIT_MIN")
        _lab_prv = _sw_val("laboratory", "prior_28", "MEDIAN_WAIT_MIN")
        _lab_val = f"{int(_lab_cur)} min" if _lab_cur else "—"
        _lab_badge, _lab_bcol = _journey_badge(_lab_cur, _lab_prv,
                                                high_thresh=20, crit_thresh=40)

        # Radiology queue
        _rad_cur = _sw_val("radiology",  "last_28",  "MEDIAN_WAIT_MIN")
        _rad_prv = _sw_val("radiology",  "prior_28", "MEDIAN_WAIT_MIN")
        _rad_val = f"{int(_rad_cur)} min" if _rad_cur else "—"
        _rad_badge, _rad_bcol = _journey_badge(_rad_cur, _rad_prv,
                                                high_thresh=20, crit_thresh=45)

        # Pharmacy TAT — prescription written → dispensed (Inv 104)
        _pharm_tat_df = P["pharm_tat"].copy() if len(P["pharm_tat"]) else pd.DataFrame()
        _ph_tat    = int(_pharm_tat_df.iloc[0]["MEDIAN_TAT_MIN"]) if len(_pharm_tat_df) else None
        _ph_w30    = float(_pharm_tat_df.iloc[0]["PCT_WITHIN_30MIN"]) if len(_pharm_tat_df) else None
        _ph_val    = f"{_ph_tat} min" if _ph_tat else "—"
        if _ph_tat and _ph_w30:
            _ph_badge = f"{'Normal' if _ph_tat <= 30 else 'Elevated'} · {_ph_w30:.0f}% within 30 min"
            _ph_bcol  = COLORS["success"] if _ph_tat <= 30 else COLORS["warning"]
        else:
            _ph_badge, _ph_bcol = None, None

        _js1, _ja1, _js2, _ja2, _js3, _ja3, _js4, _ja4, _js5 = st.columns(
            [2.5, 0.4, 2.5, 0.4, 2.5, 0.4, 2.5, 0.4, 2.5]
        )
        with _js1:
            st.markdown(
                _flow_card_html("OPD Evaluations", _rec_val, _rec_label),
                unsafe_allow_html=True,
            )
        with _ja1:
            st.markdown(_h_arrow_html, unsafe_allow_html=True)
        with _js2:
            st.markdown(
                _flow_card_html("Doctor", _doc_val, "queue wait · last 28d",
                                badge_text=_doc_badge, badge_color=_doc_bcol),
                unsafe_allow_html=True,
            )
        with _ja2:
            st.markdown(_h_arrow_html, unsafe_allow_html=True)
        with _js3:
            _lab_badge_html = (
                f'<div style="margin-top:10px">'
                f'<span style="background:{_lab_bcol}18;border:1px solid {_lab_bcol}50;'
                f'color:{_lab_bcol};font-size:10px;font-weight:700;padding:3px 10px;'
                f'border-radius:12px">{_lab_badge}</span></div>'
            ) if _lab_badge and _lab_bcol else ""
            _lab_top_border = f"border-top:4px solid {_lab_bcol}" if _lab_bcol else "border-top:4px solid #D6E4F0"
            st.markdown(
                f'<div style="background:#F4F8FC;border:1px solid #D6E4F0;{_lab_top_border};'
                f'border-radius:10px;padding:24px 14px 20px;text-align:center;min-height:180px">'
                f'<div style="font-size:9px;font-weight:700;color:#9BAEC8;text-transform:uppercase;'
                f'letter-spacing:1.5px;margin-bottom:10px">Lab</div>'
                f'<div style="font-size:30px;font-weight:800;color:#003467;line-height:1.1">{_lab_val}</div>'
                f'<div style="font-size:10px;color:#9BAEC8;margin-top:6px">queue wait · last 28d</div>'
                f'<div style="margin-top:10px;padding-top:10px;border-top:1px solid #E2EDF7">'
                f'<div style="font-size:20px;font-weight:700;color:#6B8CAE">84 min</div>'
                f'<div style="font-size:10px;color:#9BAEC8;margin-top:3px">order → result · Jan–Aug 2025</div>'
                f'</div>'
                f'{_lab_badge_html}'
                f'</div>',
                unsafe_allow_html=True,
            )
        with _ja3:
            st.markdown(_h_arrow_html, unsafe_allow_html=True)
        with _js4:
            st.markdown(
                _flow_card_html("Radiology", _rad_val, "queue wait · last 28d",
                                badge_text=_rad_badge, badge_color=_rad_bcol),
                unsafe_allow_html=True,
            )
        with _ja4:
            st.markdown(_h_arrow_html, unsafe_allow_html=True)
        with _js5:
            st.markdown(
                _flow_card_html("Pharmacy", _ph_val, "prescription → dispensed · Jan–Aug 2025",
                                badge_text=_ph_badge, badge_color=_ph_bcol),
                unsafe_allow_html=True,
            )
    # ── Active Alerts — full width ────────────────────────────────────────────

    col_l = st.container()

    with col_l:
        section_header("Active Alerts")

        # _notice_card() is defined above — collects into _pending_alerts for sorted render

        _active = 0
        _notices = []
        _pending_alerts = []  # collected then sorted CRITICAL → WATCH before render

        def _notice_card(severity, title, value, delta_line, implication, color):
            _pending_alerts.append({
                "severity": severity,
                "title": title,
                "value": value,
                "delta_line": delta_line,
                "implication": implication,
                "color": color,
            })

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
                    f"{_wi_mo_lbl} — {_wi_ward}: BTR {_wi_btr:.2f} vs floor {_wi_btr_p25:.2f}, "
                    f"beds idle {_wi_bti:.0f}d avg vs ceiling {_wi_bti_p75:.0f}d. "
                    "If visit volume is also low this is a demand gap — if normal, investigate the admissions process. "
                    "Flag to ward manager.",
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
                    f"{_rv_mo_lbl} — KES {_rv_latest/1000:.0f}K from {_rv_adm} admissions "
                    f"({_rv_drop:.1f}% below 3-month avg of KES {_rv_avg/1000:.0f}K). "
                    "Revenue follows volume — investigate why private ward intake is low. Flag to finance lead.",
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
                    f"Single doctor handling >{_DOC_CONC_CRIT if _sev == 'CRITICAL' else _DOC_CONC_WATCH}% of all evaluation visits",
                    f"{_ddisp} carries {_top_doc_pct:.0f}% of all OPD evaluations ({_top_doc_visits:,} visits). "
                    "Any absence halts outpatient flow — single-doctor dependency since M.Akinyi departed Jan 2026. "
                    "Flag to clinical lead.",
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
                    f"{_img_mo_lbl} — {_img_sess} sessions vs {_img_avg_r}-session 3-month avg ({_img_drop:.1f}% drop). "
                    "Could be equipment downtime, scheduling gap, or referral drop. "
                    "Investigate cause before acting. Flag to imaging lead.",
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
                    f"Down {th_peak_rate - th_comp_rate:.0f}pp from peak {th_peak_rate:.0f}% ({th_peak_lbl}). "
                    "Check cancellation and no-show rates on Capacity & Ops.",
                    _col,
                )
                _active += 1
                _notices.append({"level": _sev, "title": "Theatre — Completion Below Target",
                                 "metric": f"{th_comp_rate:.0f}%",
                                 "action": "Check cancellation and no-show rates on Capacity & Ops"})

        # Render alerts — CRITICAL first, then WATCH
        _SEV_ORDER = {"CRITICAL": 0, "WATCH": 1}
        for _al in sorted(_pending_alerts, key=lambda x: _SEV_ORDER.get(x["severity"], 2)):
            _badge_bg = COLORS["danger"] if _al["severity"] == "CRITICAL" else COLORS["warning"]
            st.markdown(
                f'<div style="background:#fff;border:1px solid #D6E4F0;border-left:4px solid {_al["color"]};'
                f'border-radius:8px;padding:14px 16px;margin-bottom:12px">'
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
                f'<span style="background:{_badge_bg};color:#fff;font-size:9px;font-weight:800;'
                f'letter-spacing:1.5px;padding:2px 7px;border-radius:3px">{_al["severity"]}</span>'
                f'<span style="font-size:11px;font-weight:700;color:#003467;text-transform:uppercase;'
                f'letter-spacing:0.8px">{_al["title"]}</span></div>'
                f'<div style="font-size:22px;font-weight:800;color:{_al["color"]};line-height:1.2">{_al["value"]}</div>'
                f'<div style="font-size:11px;color:#6B8CAE;margin-top:4px">{_al["delta_line"]}</div>'
                f'<div style="font-size:11px;color:#003467;margin-top:6px;line-height:1.5;'
                f'border-top:1px solid #EBF3FB;padding-top:6px">{_al["implication"]}</div>'
                f'</div>', unsafe_allow_html=True)

        st.session_state["active_notices"] = _notices
        write_current_notices(FAC_DISPLAY.get(facility, facility), _notices)

        # ── Continuous Monitoring — all evaluated domains ──────────────────────
        if _is_ksh:
            # Full monitor registry — every rule evaluated this session
            _KSH_MONITORS = [
                ("Capacity",     "Ward Volume"),
                ("Capacity",     "Ward LOS"),
                ("Capacity",     "Ward Turnover · BTR"),
                ("Capacity",     "Ward Idle · BTI"),
                ("Capacity",     "Occupancy · BOR"),
                ("Revenue",      "Private Ward Revenue"),
                ("Imaging",      "CT / Angio Volume"),
                ("Theatre",      "Completion Rate"),
                ("Theatre",      "Theatre Utilization"),
                ("Lab",          "Visit Volume"),
                ("Lab",          "Abnormal Rate"),
                ("Staffing",     "Doctor Concentration"),
                ("Staffing",     "Doctor Burnout"),
                ("Staffing",     "Visit Load"),
                ("Patient Flow", "Admission TAT"),
                ("Patient Flow", "Discharge Pattern"),
                ("Renal",        "Critical Non-Admission"),
                ("Dialysis",     "Equipment Idle"),
            ]
            # Map fired alert titles to monitor keys (partial match)
            _FIRED_MAP = {
                "concentration": "Doctor Concentration",
                "burnout":        "Doctor Burnout",
                "visit load":     "Visit Load",
                "theatre":        "Completion Rate",
                "theatre util":   "Theatre Utilization",
                "ct volume":      "CT / Angio Volume",
                "imaging":        "CT / Angio Volume",
                "revenue drop":   "Private Ward Revenue",
                "ward idle":      "Ward Idle · BTI",
                "low occupancy":  "Occupancy · BOR",
                "high occupancy": "Occupancy · BOR",
                "high volume":    "Ward Volume",
                "extended los":   "Ward LOS",
                "admission tat":  "Admission TAT",
                "patient request":"Discharge Pattern",
                "creatinine":     "Critical Non-Admission",
                "dialysis":       "Equipment Idle",
            }
            _fired_keys = set()
            for _al in _pending_alerts:
                _t = _al["title"].lower()
                for _kw, _mk in _FIRED_MAP.items():
                    if _kw in _t:
                        _fired_keys.add(_mk)

            _stable_monitors = [
                (cat, lbl) for cat, lbl in _KSH_MONITORS
                if lbl not in _fired_keys
            ]

            if _stable_monitors:
                st.markdown(
                    '<p style="font-size:9px;font-weight:800;letter-spacing:2px;'
                    'text-transform:uppercase;color:#6B8CAE;margin:20px 0 12px">Continuous Monitoring</p>',
                    unsafe_allow_html=True,
                )

                # ── Compute ward summary stats from _btr_bti_alert_df ─────────────
                _mon_btr_src = _btr_bti_alert_df.copy() if len(_btr_bti_alert_df) else pd.DataFrame()
                if len(_mon_btr_src):
                    _mon_all_mos    = sorted(_mon_btr_src["month"].unique())
                    _mon_latest_mo  = _mon_all_mos[-1]
                    _mon_lat        = _mon_btr_src[_mon_btr_src["month"] == _mon_latest_mo]
                    _mon_adm_total  = int(_mon_lat["total_admissions"].sum())
                    _mon_los_avg    = round(
                        float(_mon_lat["total_bed_days"].sum())
                        / max(float(_mon_lat["discharged_admissions"].sum()), 1), 1
                    )
                    _mon_btr_min    = round(float(_mon_lat["btr"].min()), 1)
                    _mon_btr_max    = round(float(_mon_lat["btr"].max()), 1)
                    _mon_bti_idx    = _mon_lat["bti_days"].idxmax()
                    _mon_bti_max    = round(float(_mon_lat.loc[_mon_bti_idx, "bti_days"]))
                    _mon_bti_ward   = str(_mon_lat.loc[_mon_bti_idx, "ward_name"])
                    _mon_bor_min    = round(float(_mon_lat["bor_pct"].min()))
                    _mon_bor_max    = round(float(_mon_lat["bor_pct"].max()))
                    _mon_ward_lbl   = pd.to_datetime(_mon_latest_mo).strftime("%b %Y")
                    # Admission MoM delta from previous month
                    if len(_mon_all_mos) >= 2:
                        _mon_prev_lat   = _mon_btr_src[_mon_btr_src["month"] == _mon_all_mos[-2]]
                        _mon_adm_prev   = int(_mon_prev_lat["total_admissions"].sum())
                        _mon_adm_diff   = _mon_adm_total - _mon_adm_prev
                        _mon_adm_delta  = f" · {'↑' if _mon_adm_diff >= 0 else '↓'}{abs(_mon_adm_diff)} MoM"
                    else:
                        _mon_adm_delta  = ""
                    # BTI ceiling proximity check
                    _mon_bti_p75    = _BTI_P75.get(_mon_bti_ward, float("inf"))
                    _mon_bti_warn   = _mon_bti_max >= _mon_bti_p75 * 0.90
                else:
                    _mon_adm_total  = None
                    _mon_adm_delta  = ""
                    _mon_los_avg    = None
                    _mon_btr_min    = _mon_btr_max = None
                    _mon_bti_max    = None
                    _mon_bti_ward   = "—"
                    _mon_bti_warn   = False
                    _mon_bor_min    = _mon_bor_max = None
                    _mon_ward_lbl   = "—"

                # Revenue — fall back to raw tail if _rv_latest not set; add MoM delta
                _mon_rv_val = _rv_latest
                _mon_rv_lbl = _rv_mo_lbl or "—"
                if _mon_rv_val is None and len(_revpab_priv_df):
                    _rv_fb = _revpab_priv_df.tail(1)
                    _mon_rv_val = float(_rv_fb["total_revenue"].iloc[0])
                    _mon_rv_lbl = pd.to_datetime(_rv_fb["admission_month"].iloc[0]).strftime("%b %Y")
                _mon_rv_delta = ""
                if _mon_rv_val is not None and len(_revpab_priv_df) >= 2:
                    _rv_prev_val = float(_revpab_priv_df.iloc[-2]["total_revenue"])
                    if _rv_prev_val > 0:
                        _rv_mom = round(100 * (_mon_rv_val - _rv_prev_val) / _rv_prev_val, 1)
                        _mon_rv_delta = f" · {'↑' if _rv_mom >= 0 else '↓'}{abs(_rv_mom):.0f}% MoM"

                # Lab month label
                _mon_lab_lbl = (
                    pd.to_datetime(_lab_latest_month).strftime("%b %Y")
                    if _lab_latest_month is not None else "—"
                )

                # TAT fast_pct MoM delta — badge and fires are on fast_pct, not P50
                # ↓ = fast_pct declining toward threshold (bad); ↑ = improving (good)
                _mon_tat_delta = ""
                if _tat_latest_fast_pct is not None and len(_tat_mo_df) >= 2:
                    _tat_prev_fast_pct = float(_tat_mo_df.iloc[-2]["fast_pct"])
                    _fast_pct_diff = round(_tat_latest_fast_pct - _tat_prev_fast_pct, 1)
                    _mon_tat_delta = f" · {'↑' if _fast_pct_diff >= 0 else '↓'}{abs(_fast_pct_diff):.1f}pp MoM"

                # ── Semantic badge logic ──────────────────────────────────────────
                # Palette: green=#22C55E  amber=#F59E0B  red=#EF4444  grey=#9BAEC8

                # BOR — WHO/industry optimal range 60–85%
                if _mon_bor_min is not None:
                    _bor_mid = (_mon_bor_min + _mon_bor_max) / 2
                    if _mon_bor_max <= 60:
                        _badge_bor = ("UNDERUTIL", "#F59E0B")
                    elif _mon_bor_min >= 85:
                        _badge_bor = ("OVERUTIL", "#EF4444")
                    elif _bor_mid >= 60 and _mon_bor_max <= 85:
                        _badge_bor = ("OPTIMAL", "#22C55E")
                    else:
                        _badge_bor = ("MIXED", "#F59E0B")
                else:
                    _badge_bor = ("NO DATA", "#9BAEC8")

                # BTI — at or near P75 ceiling
                _badge_bti = ("AT CEILING", "#F59E0B") if _mon_bti_warn else ("STABLE", "#22C55E")

                # Ward Volume — MoM direction
                if "↑" in _mon_adm_delta:
                    _badge_vol = ("GROWING", "#22C55E")
                elif "↓" in _mon_adm_delta:
                    _badge_vol = ("CONTRACTING", "#F59E0B")
                else:
                    _badge_vol = ("STABLE", "#9BAEC8")

                # Revenue — MoM direction
                _badge_rv = ("DECLINING", "#F59E0B") if "↓" in _mon_rv_delta else ("STABLE", "#22C55E")

                # CT volume — % of rolling avg vs thresholds
                if _img_pct_of_avg is None:
                    _badge_ct = ("NO DATA", "#9BAEC8")
                elif _img_pct_of_avg >= _IMAGING_WATCH_PCT:
                    _badge_ct = ("WITHIN RANGE", "#22C55E")
                else:
                    _badge_ct = ("NEAR WATCH", "#F59E0B")

                # Theatre completion
                if th_comp_rate is None:
                    _badge_th_comp = ("NO DATA", "#9BAEC8")
                elif th_comp_rate >= _THEATRE_WATCH:
                    _badge_th_comp = ("WITHIN RANGE", "#22C55E")
                else:
                    _badge_th_comp = ("BELOW TARGET", "#F59E0B")

                # Lab visit volume
                _badge_lab_vol = (
                    ("NEAR WATCH", "#F59E0B")
                    if (_lab_latest_visits and _lab_latest_visits < _LAB_VOL_WATCH)
                    else ("NORMAL", "#22C55E")
                )

                # Lab abnormal rate
                _badge_abnorm = (
                    ("ELEVATED", "#F59E0B")
                    if (_lab_latest_abnorm is not None and _lab_latest_abnorm >= _LAB_ABNORM_WATCH)
                    else ("NORMAL", "#22C55E")
                )

                # Doctor burnout / visit load
                _badge_burnout = ("CLEAR", "#22C55E") if not _burnout_alerts else ("ACTIVE", "#F59E0B")
                _badge_wl      = ("CLEAR", "#22C55E") if not _doc_wl_alerts else ("ACTIVE", "#F59E0B")

                # Admission TAT — fast-track proximity to threshold
                if _tat_latest_fast_pct is None:
                    _badge_tat = ("NO DATA", "#9BAEC8")
                elif _tat_latest_fast_pct < _TAT_WATCH:
                    _badge_tat = ("NEAR WATCH", "#F59E0B")
                elif _tat_latest_fast_pct < _TAT_WATCH + 5:
                    _badge_tat = ("BORDERLINE", "#F59E0B")
                else:
                    _badge_tat = ("STABLE", "#22C55E")

                # Equipment idle
                if months_idle is None:
                    _badge_dial = ("NO DATA", "#9BAEC8")
                elif months_idle < _DIALYSIS_IDLE:
                    _badge_dial = ("ACTIVE", "#22C55E")
                else:
                    _badge_dial = ("IDLE", "#EF4444")

                # Static badges — no logic benchmark available
                _BADGE_STABLE  = ("STABLE",  "#22C55E")
                _BADGE_NO_DATA = ("NO DATA", "#9BAEC8")

                # ── Card definitions — (value, unit, desc, threshold, badge_tuple) ──
                _CARD_DEFS = {
                    "Ward Volume": (
                        f"{_mon_adm_total:,}" if _mon_adm_total else "—",
                        f"admissions · {_mon_ward_lbl}{_mon_adm_delta}",
                        "Total inpatient admissions across all wards — direction shows whether volume is building or contracting",
                        "No standalone rule yet — volume tracked for context",
                        _badge_vol,
                    ),
                    "Ward LOS": (
                        f"{_mon_los_avg}d" if _mon_los_avg else "—",
                        f"avg LOS all wards · {_mon_ward_lbl}",
                        "Average length of stay (total bed-days / discharges) across all wards combined",
                        "No standalone rule yet — LOS tracked for context; alert fires when extended LOS detected per ward",
                        _BADGE_STABLE,
                    ),
                    "Ward Turnover · BTR": (
                        f"{_mon_btr_min}–{_mon_btr_max}" if _mon_btr_min is not None else "—",
                        f"admissions/bed · range across wards · {_mon_ward_lbl}",
                        "Admissions per available bed per month — low end is Private Maternity, high end is busiest general ward",
                        "BTR drops below ward P25 floor AND BTI rises above P75 ceiling simultaneously",
                        _BADGE_STABLE,
                    ),
                    "Ward Idle · BTI": (
                        (f"{int(_mon_bti_max)}d" if _mon_bti_max is not None else "—"),
                        (f"highest idle avg · {_mon_bti_ward} · {_mon_ward_lbl} · at P75 ceiling"
                         if _mon_bti_warn else f"highest idle avg · {_mon_bti_ward} · {_mon_ward_lbl}"),
                        "Total idle bed-days in the month divided by discharges — high BTI means beds sit empty longer between patients",
                        "BTI above ward P75 ceiling AND BTR below P25 floor for the same ward",
                        _badge_bti,
                    ),
                    "Occupancy · BOR": (
                        f"{int(_mon_bor_min)}–{int(_mon_bor_max)}%" if _mon_bor_min is not None else "—",
                        f"BOR range across wards · {_mon_ward_lbl} · optimal 60–85%",
                        "Occupied bed-days as % of available bed-days per ward. Low BOR with stable OPD volume signals an admissions conversion gap.",
                        f"Underutil: below 60% · Optimal: 60–85% · Overutil: above {_BOR_HIGH_WATCH:.0f}% for 2mo / {_BOR_HIGH_CRIT:.0f}% single month",
                        _badge_bor,
                    ),
                    "Private Ward Revenue": (
                        fmt_kes(_mon_rv_val) if _mon_rv_val else "—",
                        f"Private F + M · {_mon_rv_lbl}{_mon_rv_delta}",
                        "Combined Private Female + Male ward monthly revenue — MoM delta shows whether the trend is reversing",
                        f"Revenue drops more than {_REVPAB_WATCH_DROP:.0f}% below 3-month rolling average",
                        _badge_rv,
                    ),
                    "CT / Angio Volume": (
                        f"{_img_latest_sess}" if _img_latest_sess is not None else "—",
                        (f"sessions · {_img_mo_lbl} · {_img_pct_of_avg:.0f}% of 3mo avg · complete month"
                         if _img_pct_of_avg is not None else f"sessions · {_img_mo_lbl or '—'}"),
                        "Monthly CT and Angiography session count validated against 3-month rolling average — partial months excluded at source",
                        f"WATCH below {_IMAGING_WATCH_PCT:.0f}% · CRITICAL below {_IMAGING_CRIT_PCT:.0f}% of rolling avg",
                        _badge_ct,
                    ),
                    "Completion Rate": (
                        f"{th_comp_rate:.0f}%" if th_comp_rate is not None else "—",
                        (f"3-month completion · peak {th_peak_rate:.0f}% in {th_peak_lbl}"
                         if th_peak_rate is not None else "3-month theatre completion"),
                        "Theatre sessions completed as % of scheduled sessions (3-month rolling)",
                        f"WATCH below {_THEATRE_WATCH}% · CRITICAL below {_THEATRE_CRIT}%",
                        _badge_th_comp,
                    ),
                    "Theatre Utilization": (
                        "—",
                        "theatre room utilisation",
                        "% of scheduled time each theatre room was actively used — identifies idle rooms and underbooked slots. With room-level data this would show which theatres are carrying load and which are near-idle.",
                        "Pending — room-level schedule vs actual data required to compute",
                        _BADGE_NO_DATA,
                    ),
                    "Visit Volume": (
                        f"{_lab_latest_visits:,}" if _lab_latest_visits else "—",
                        (f"lab visits · {_mon_lab_lbl} · below {_LAB_VOL_WATCH} floor"
                         if (_lab_latest_visits and _lab_latest_visits < _LAB_VOL_WATCH)
                         else f"lab visits · {_mon_lab_lbl}"),
                        "Monthly unique patient visits generating at least one lab result — sustained drops signal equipment downtime or staffing gaps",
                        f"WATCH below {_LAB_VOL_WATCH} for 2mo · CRITICAL below {_LAB_VOL_CRIT} single month",
                        _badge_lab_vol,
                    ),
                    "Abnormal Rate": (
                        f"{_lab_latest_abnorm:.1f}%" if _lab_latest_abnorm is not None else "—",
                        f"of results outside reference range · {_mon_lab_lbl}",
                        "% of lab tests returning an H (high) or L (low) flag — 6 in every 100 results outside normal range at 6.1%. Sustained rise signals case-acuity shift.",
                        f"WATCH above {_LAB_ABNORM_WATCH:.0f}% · CRITICAL above {_LAB_ABNORM_CRIT:.0f}% for 2 consecutive months",
                        _badge_abnorm,
                    ),
                    "Doctor Concentration": (
                        f"{_top_doc_pct:.0f}%" if _top_doc_pct else "—",
                        f"of OPD by {_top_doc_name or '—'}",
                        "Share of total OPD evaluations carried by the single highest-volume doctor",
                        f"WATCH above {_DOC_CONC_WATCH}% · CRITICAL above {_DOC_CONC_CRIT}% of all evaluations",
                        _BADGE_STABLE,
                    ),
                    "Doctor Burnout": (
                        "None" if not _burnout_alerts else f"{len(_burnout_alerts)}",
                        "doctors above 150% of personal baseline",
                        "Doctors sustaining more than 150% of their 3-month visit average — signals unsustainable load before performance degrades",
                        "Any doctor at more than 150% of personal baseline for 2 consecutive months",
                        _badge_burnout,
                    ),
                    "Visit Load": (
                        "None" if not _doc_wl_alerts else f"{len(_doc_wl_alerts)}",
                        "doctors exceeding personal P90 load",
                        "Per-doctor monthly visits vs personal P90 — early signal before burnout appears",
                        "Any tracked doctor exceeds personal P90 for 2 consecutive months",
                        _badge_wl,
                    ),
                    "Admission TAT": (
                        f"{_tat_latest_p50}min" if _tat_latest_p50 is not None else "—",
                        (f"fast-track {_tat_latest_fast_pct:.1f}%{_mon_tat_delta}"
                         f" · {pd.to_datetime(_tat_latest_month).strftime('%b %Y')}"
                         if _tat_latest_fast_pct is not None else "median admission time"),
                        "Median time from decision-to-admit to ward placement. Fast-track = completed within 60 minutes.",
                        f"WATCH fast-track below {_TAT_WATCH:.0f}% for 2mo · CRITICAL below {_TAT_CRIT:.0f}% single month",
                        _badge_tat,
                    ),
                    "Discharge Pattern": (
                        "No data",
                        "patient-request discharges — baseline not yet established",
                        "Patient-request discharge rate per ward — patients who leave against medical advice. Baseline computation pending.",
                        "Rate exceeds ward-specific baseline for 2 consecutive months",
                        _BADGE_NO_DATA,
                    ),
                    "Critical Non-Admission": (
                        f"{_cd12_rate_v:.0f}%" if _cd12_rate_v is not None else "No recent data",
                        (f"CD12 patients not admitted · {_cd12_mo_lbl}"
                         if _cd12_rate_v is not None else "data outside 3-month window"),
                        "Critical creatinine (CD12) patients not admitted after flagging — each one is a patient who needed a bed and didn't get one",
                        f"WATCH above {_CD12_WATCH:.0f}% · CRITICAL above {_CD12_CRIT:.0f}% (min {_CD12_MIN_EVTS} events)",
                        _BADGE_NO_DATA,
                    ),
                    "Equipment Idle": (
                        f"{months_idle}mo" if months_idle is not None else "—",
                        "months since last dialysis session",
                        "Months elapsed since the last recorded dialysis session at KSH — idle equipment = unrealised dialysis revenue",
                        f"Equipment idle more than {_DIALYSIS_IDLE} months",
                        _badge_dial,
                    ),
                }

                # ── Paginated card grid — 4 per page, 2×2 ───────────────────────
                _card_list  = [(cat, lbl) for cat, lbl in _stable_monitors if lbl in _CARD_DEFS]
                _mon_total  = len(_card_list)
                _mon_per_pg = 4
                _mon_pages  = max(1, (_mon_total + _mon_per_pg - 1) // _mon_per_pg)

                if "_ksh_mon_pg" not in st.session_state:
                    st.session_state["_ksh_mon_pg"] = 0
                _mon_pg = min(int(st.session_state["_ksh_mon_pg"]), _mon_pages - 1)

                _pg_cards = _card_list[_mon_pg * _mon_per_pg : (_mon_pg + 1) * _mon_per_pg]

                # Colour palette per badge tone
                _CARD_PALETTE = {
                    "#22C55E": ("#F0FBF5", "#BBE5CC", "#1A6B3A", "#4A8C6A", "#BBE5CC"),
                    "#F59E0B": ("#FFFBF0", "#FCD34D", "#92400E", "#B45309", "#FCD34D"),
                    "#EF4444": ("#FFF5F5", "#FECACA", "#991B1B", "#B91C1C", "#FECACA"),
                    "#9BAEC8": ("#F8F9FB", "#DDE4EE", "#6B8CAE", "#8A9BB5", "#DDE4EE"),
                }  # (bg, border, value_color, unit_color, divider)

                for _ri in range(0, len(_pg_cards), 2):
                    _row_cols = st.columns(2, gap="small")
                    for _ci, (_, _lbl) in enumerate(_pg_cards[_ri:_ri + 2]):
                        _cval, _cunit, _cdesc, _cthr, (_cbadge, _cbcol) = _CARD_DEFS[_lbl]
                        _cbg, _cborder, _cvcolor, _cucolor, _cdivider = _CARD_PALETTE[_cbcol]
                        with _row_cols[_ci]:
                            st.markdown(
                                f'<div style="background:{_cbg};border:1px solid {_cborder};'
                                f'border-radius:8px;border-left:4px solid {_cbcol};'
                                f'padding:12px 14px;margin-bottom:8px">'
                                f'<span style="display:inline-block;font-size:8px;font-weight:800;'
                                f'letter-spacing:2px;text-transform:uppercase;color:#fff;'
                                f'background:{_cbcol};border-radius:3px;padding:2px 6px;'
                                f'margin-bottom:8px">{_cbadge}</span>'
                                f'<p style="font-size:8px;font-weight:700;letter-spacing:1.5px;'
                                f'text-transform:uppercase;color:#6B8CAE;margin:0 0 4px">{_lbl}</p>'
                                f'<p style="font-size:22px;font-weight:800;color:{_cvcolor};'
                                f'margin:0;line-height:1.1">{_cval}</p>'
                                f'<p style="font-size:9px;color:{_cucolor};margin:2px 0 10px;'
                                f'line-height:1.4">{_cunit}</p>'
                                f'<div style="border-top:1px solid {_cdivider};padding-top:8px">'
                                f'<p style="font-size:9px;color:#4A5568;margin:0 0 4px;'
                                f'line-height:1.4">{_cdesc}</p>'
                                f'<p style="font-size:8px;color:#6B8CAE;margin:0;line-height:1.3">'
                                f'<span style="font-weight:700">Fires:</span> {_cthr}</p>'
                                f'</div></div>',
                                unsafe_allow_html=True,
                            )

                # Navigation row
                if _mon_pages > 1:
                    _pnav_l, _pnav_m, _pnav_r = st.columns([1, 6, 1])
                    with _pnav_l:
                        if _mon_pg > 0:
                            if st.button("‹", key="_mon_prev"):
                                st.session_state["_ksh_mon_pg"] = _mon_pg - 1
                                st.rerun()
                    with _pnav_m:
                        st.markdown(
                            f'<p style="text-align:center;font-size:9px;color:#9BAEC8;'
                            f'margin:4px 0">{_mon_pg + 1} of {_mon_pages} '
                            f'· {_mon_total} metrics monitored</p>',
                            unsafe_allow_html=True,
                        )
                    with _pnav_r:
                        if _mon_pg < _mon_pages - 1:
                            if st.button("›", key="_mon_next"):
                                st.session_state["_ksh_mon_pg"] = _mon_pg + 1
                                st.rerun()

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

    _P3_VER = "v25"  # bump when adding new keys to force session-state rebuild
    if (not st.session_state.p3
            or st.session_state.p3.get("_fac") != fac_key
            or st.session_state.p3.get("_ver") != _P3_VER):
        with st.spinner("Loading…"):
            _is_ksh_p3 = (fac_key == "KISUMU_CLEAN")
            st.session_state.p3 = {
                "_fac":        fac_key,
                "_ver":        _P3_VER,
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
                "lab":             q_lab_monthly()                if _is_ksh_p3 else pd.DataFrame(),
                "lab_morning":     q_lab_morning_completion()     if _is_ksh_p3 else pd.DataFrame(),
                "lab_tat":         q_lab_tat_monthly_clean()      if _is_ksh_p3 else pd.DataFrame(),
                "lab_tat_dow":     q_lab_tat_dow_clean()          if _is_ksh_p3 else pd.DataFrame(),
                "lab_tat_test":    q_lab_tat_by_test()            if _is_ksh_p3 else pd.DataFrame(),
                "pharm_wait_dow":  q_pharmacy_wait_dow()          if _is_ksh_p3 else pd.DataFrame(),
                "lab_flow_delta":  q_lab_flow_delta()             if _is_ksh_p3 else pd.DataFrame(),
                "lab_handoff":     q_lab_handoff_delta()          if _is_ksh_p3 else pd.DataFrame(),
                "lab_weekly":      q_lab_weekly_trend()           if _is_ksh_p3 else pd.DataFrame(),
                "lab_downstream":  q_lab_downstream_monthly()     if _is_ksh_p3 else pd.DataFrame(),
                "lab_to_bed":      q_lab_to_bed_monthly()         if _is_ksh_p3 else pd.DataFrame(),
                "flow_transitions": q_patient_flow_transitions()  if _is_ksh_p3 else pd.DataFrame(),
                "flow_dow":         q_patient_flow_dow()          if _is_ksh_p3 else pd.DataFrame(),
                "visit_sum":   q_visit_summary()                   if _is_ksh_p3 else pd.DataFrame(),
                "cd12_rate":   q_cd12_monthly_rate()               if _is_ksh_p3 else pd.DataFrame(),
                "doctor_conv": q_doctor_conversion_monthly()       if _is_ksh_p3 else pd.DataFrame(),
                "peak_bk":     q_peak_breakdown()                  if _is_ksh_p3 else pd.DataFrame(),
                "btr_bti":     q_btr_bti_monthly()           if _is_ksh_p3 else pd.DataFrame(),
                "adm_tat":         q_admission_tat_bimodal()    if _is_ksh_p3 else pd.DataFrame(),
                "adm_tat_monthly": q_admission_tat_monthly()    if _is_ksh_p3 else pd.DataFrame(),
                "adm_tat_dow":     q_admission_tat_dow()        if _is_ksh_p3 else pd.DataFrame(),
                "discharge_tat":   q_discharge_tat()            if _is_ksh_p3 else pd.DataFrame(),
                "discharge_dow":   q_discharge_dow()            if _is_ksh_p3 else pd.DataFrame(),
                "th_emer_tat": q_theatre_emergency_tat()    if _is_ksh_p3 else pd.DataFrame(),
                "th_by_theatre":  q_theatre_by_type()      if _is_ksh_p3 else pd.DataFrame(),
                "th_procedures":     q_theatre_procedures()          if _is_ksh_p3 else pd.DataFrame(),
                "th_proc_monthly":   q_theatre_procedures_monthly()   if _is_ksh_p3 else pd.DataFrame(),
                "th_trend_theatre":  q_theatre_trend_by_theatre()     if _is_ksh_p3 else pd.DataFrame(),
                "th_non_comp":       q_theatre_non_completion()       if _is_ksh_p3 else pd.DataFrame(),
                "th_status":         q_theatre_status_breakdown()     if _is_ksh_p3 else pd.DataFrame(),
                "th_proc_rates":     q_theatre_procedure_rates()      if _is_ksh_p3 else pd.DataFrame(),
                "th_cur_by_th":      q_theatre_cur_month_by_theatre() if _is_ksh_p3 else pd.DataFrame(),
                "lab_util":          q_lab_utilization_delta()        if _is_ksh_p3 else pd.DataFrame(),
                "lab_vol_monthly":   q_lab_result_volume_monthly()    if _is_ksh_p3 else pd.DataFrame(),
                # Patient Flow tab (A–F, spine-based)
                "journey_sankey":    q_patient_journey_sankey()       if _is_ksh_p3 else pd.DataFrame(),
                "stage_wait_p3":     q_rpt_stage_wait()               if _is_ksh_p3 else pd.DataFrame(),
                "opd_summary":       q_opd_spine_summary()            if _is_ksh_p3 else pd.DataFrame(),
                "opd_28d":           q_opd_daily_28d()                if _is_ksh_p3 else pd.DataFrame(),
                "opd_monthly":       q_opd_monthly_volume()           if _is_ksh_p3 else pd.DataFrame(),
                "opd_dow":           q_opd_dow_visits()               if _is_ksh_p3 else pd.DataFrame(),
                "opd_peak_band":     q_opd_peak_band_tat()            if _is_ksh_p3 else pd.DataFrame(),
                "opd_hourly":        q_opd_hourly_tat()               if _is_ksh_p3 else pd.DataFrame(),
                "opd_weekly":        q_opd_weekly_pressure()          if _is_ksh_p3 else pd.DataFrame(),
                "opd_spillover":     q_opd_spillover_summary()        if _is_ksh_p3 else pd.DataFrame(),
                "opd_heatmap":       q_opd_flagged_heatmap()          if _is_ksh_p3 else pd.DataFrame(),
                # Legacy pharmacy (retained in state, not rendered post-rebuild)
                "opd_kpi_28d":       q_opd_kpi_28d()                  if _is_ksh_p3 else pd.DataFrame(),
                "pharm_source":      q_pharmacy_source_split()        if _is_ksh_p3 else pd.DataFrame(),
                "pharm_hour":        q_pharmacy_hour_of_day()         if _is_ksh_p3 else pd.DataFrame(),
                "pharm_monthly":     q_pharmacy_monthly_tat()         if _is_ksh_p3 else pd.DataFrame(),
                "pharm_dist":        q_pharmacy_wait_dist()           if _is_ksh_p3 else pd.DataFrame(),
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
    # q_theatre_trend() freshness filter already strips partial months (max_day < 25).
    # Use the last row as current — no further offset needed.
    _th_sorted  = th_trend.sort_values("SESSION_MONTH") if len(th_trend) else th_trend
    _th_cur_row = _th_sorted.iloc[-1] if len(_th_sorted) >= 1 else None
    _th_pri_row = _th_sorted.iloc[-2] if len(_th_sorted) >= 2 else None

    th_cur_rate = (100 * float(_th_cur_row["COMPLETED_SESSIONS"]) / max(float(_th_cur_row["TOTAL_SESSIONS"]), 1)
                   if _th_cur_row is not None else th_overall_rate)
    th_cur_rev  = float(_th_cur_row["TOTAL_REVENUE"]) if _th_cur_row is not None else 0
    th_cur_lbl  = pd.to_datetime(_th_cur_row["SESSION_MONTH"]).strftime("%b %Y") if _th_cur_row is not None else "—"
    th_pri_rate = (100 * float(_th_pri_row["COMPLETED_SESSIONS"]) / max(float(_th_pri_row["TOTAL_SESSIONS"]), 1)
                   if _th_pri_row is not None else None)
    th_pri_rev  = float(_th_pri_row["TOTAL_REVENUE"]) if _th_pri_row is not None else None

    _comp_delta    = th_cur_rate - th_pri_rate if th_pri_rate is not None else None
    _rev_delta_pct = ((th_cur_rev - th_pri_rev) / max(th_pri_rev, 1) * 100
                      if th_pri_rev is not None else None)

    th_rate_subtitle = (
        f"{th_cur_lbl} · {th_pri_rate:.1f}% prior month · {'+' if _comp_delta >= 0 else ''}{_comp_delta:.1f}pp"
        if _comp_delta is not None else th_cur_lbl)
    th_rev_subtitle = (
        f"{th_cur_lbl} · {fmt_kes(th_pri_rev)} prior month · {'+' if _rev_delta_pct >= 0 else ''}{_rev_delta_pct:.1f}%"
        if _rev_delta_pct is not None else th_cur_lbl)

    th_monthly_rev = th_cur_rev
    th_recent_rate = th_cur_rate  # alias used by section header below
    th_rate_color  = (COLORS["danger"]  if th_cur_rate < 90
                      else COLORS["warning"] if th_cur_rate < 95
                      else COLORS["success"])

    top_revpab_row = beds_r.iloc[0] if len(beds_r) else None
    top_revpab_val = fmt_kes(float(top_revpab_row["REVPAB"])) if top_revpab_row is not None else "—"
    top_revpab_label = (f"{top_revpab_row['WARD_NAME']}" if top_revpab_row is not None else "")

    fac_dialysis = dialysis[dialysis["FACILITY"] == facility]
    dial_sessions = int(fac_dialysis.nlargest(1, "SESSION_MONTH")["TOTAL_SESSIONS"].sum()) if len(fac_dialysis) else 0

    th_dot = _dot(th_trend["COMPLETION_RATE_PCT"] if len(th_trend) else None, higher_is_good=True)

    if facility == "KISUMU_CLEAN":
        pass  # Page-level KPIs removed — detail lives in the Theatre tab KPI row
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
    content:"\f610\00a0\00a0";color:#0072CE}
[data-testid="stTabs"] [role="tablist"] button:nth-child(2)::before{
    font-family:"Font Awesome 6 Free";font-weight:900;
    content:"\f48e\00a0\00a0";color:#0072CE}
[data-testid="stTabs"] [role="tablist"] button:nth-child(3)::before{
    font-family:"Font Awesome 6 Free";font-weight:900;
    content:"\f236\00a0\00a0";color:#0072CE}
[data-testid="stTabs"] [role="tablist"] button:nth-child(4)::before{
    font-family:"Font Awesome 6 Free";font-weight:900;
    content:"\f0f0\00a0\00a0";color:#0072CE}
[data-testid="stTabs"] [role="tablist"] button:nth-child(5)::before{
    font-family:"Font Awesome 6 Free";font-weight:900;
    content:"\f554\00a0\00a0";color:#0072CE}
</style>
""", unsafe_allow_html=True)

    tab5, tab3, tab1, tab2, tab4 = st.tabs([
        "Patient Flow",
        "Lab Operations",
        "Theatre",
        "Beds & Wards",
        "Staffing",
    ])

    # ── Theatre (tab slot 2) ──────────────────────────────────────────────────

    with tab1:
        if facility == "TENRI":
            st.info("Theatre analytics are KSH-specific — not applicable for TENRI.")
        else:
            # ── Pre-compute shared variables ──────────────────────────────────
            _th_nc_total     = int(_th_cur_row["TOTAL_SESSIONS"])     if _th_cur_row is not None else 0
            _th_nc_comp      = int(_th_cur_row["COMPLETED_SESSIONS"]) if _th_cur_row is not None else 0
            _th_nc_count     = _th_nc_total - _th_nc_comp
            _th_nc_rate      = round(100 * _th_nc_count / max(_th_nc_total, 1), 1)
            _th_nc_prv_total = int(_th_pri_row["TOTAL_SESSIONS"])     if _th_pri_row is not None else None
            _th_nc_prv_comp  = int(_th_pri_row["COMPLETED_SESSIONS"]) if _th_pri_row is not None else None
            _th_nc_prv_rate  = (round(100 * (_th_nc_prv_total - _th_nc_prv_comp)
                                      / max(_th_nc_prv_total, 1), 1)
                                if _th_nc_prv_total is not None else None)
            _th_nc_delta     = (round(_th_nc_rate - _th_nc_prv_rate, 1)
                                if _th_nc_prv_rate is not None else None)

            # Emergency TAT
            _tat_kpi = P.get("th_emer_tat", pd.DataFrame()).copy()
            if len(_tat_kpi):
                _tat_kpi.columns = [c.upper() for c in _tat_kpi.columns]
                _tat_lags      = _tat_kpi["BOOKING_TO_START_MIN"].values.astype(float)
                _th_tat_h      = round(float(np.median(_tat_lags)) / 60, 1)
                _th_tat_n      = len(_tat_lags)
                _th_tat_over24 = int((_tat_lags > 1440).sum())
            else:
                _th_tat_h = _th_tat_n = _th_tat_over24 = None

            # Non-completion data from extended gold table
            _th_nc_df = P.get("th_non_comp", pd.DataFrame()).copy()
            if len(_th_nc_df):
                _th_nc_df.columns = [c.upper() for c in _th_nc_df.columns]

            # Status breakdown
            _th_st_df = P.get("th_status", pd.DataFrame()).copy()
            if len(_th_st_df):
                _th_st_df.columns = [c.upper() for c in _th_st_df.columns]

            # Per-theatre data (for which-theatre context line)
            _th_by_df = P.get("th_by_theatre", pd.DataFrame()).copy()
            if len(_th_by_df):
                _th_by_df.columns = [c.upper() for c in _th_by_df.columns]

            # Revenue exposure total
            _th_rev_exp = (
                int(_th_nc_df["REVENUE_EXPOSURE_KES"].sum())
                if len(_th_nc_df) and "REVENUE_EXPOSURE_KES" in _th_nc_df.columns
                else None
            )

            # Which theatre has non-completions in current month
            _th_nc_theatre = "—"
            _th_ok_theatre = []
            if len(_th_nc_df) and "THEATRE_NAME" in _th_nc_df.columns:
                _nc_by_th = (
                    _th_nc_df.groupby("THEATRE_NAME")["NON_COMPLETED_SESSIONS"]
                    .sum().sort_values(ascending=False)
                )
                _th_nc_theatre = ", ".join(
                    f"{t} ({int(n)})" for t, n in _nc_by_th.items()
                )
                # Theatres with 0 non-completions in current month
                if len(_th_by_df):
                    _all_theatres = set(_th_by_df["THEATRE_NAME"].tolist())
                    _nc_theatres  = set(_nc_by_th.index.tolist())
                    _th_ok_theatre = sorted(_all_theatres - _nc_theatres)

            # Elective vs emergency: derived from trend (emergency completed = all emergency)
            _th_emerg_cur = (
                int(_th_cur_row["EMERGENCY_SESSIONS"])
                if _th_cur_row is not None and "EMERGENCY_SESSIONS" in _th_cur_row.index
                else None
            )

            # Dormancy: anchor to data cutoff, not today
            _DATA_CUTOFF = pd.Timestamp("2026-04-01")
            _dorm_cutoff = (_DATA_CUTOFF - pd.DateOffset(months=2)).to_period("M").to_timestamp()

            # Payer totals — computed here so both Section 4 annotation and Section 5 cards use same values
            _cash_total    = int(_th_nc_df["CASH_NON_COMPLETED"].sum())    if len(_th_nc_df) else 0
            _insured_total = int(_th_nc_df["INSURED_NON_COMPLETED"].sum()) if len(_th_nc_df) else 0
            _cash_pct      = round(100 * _cash_total    / max(_th_nc_count, 1))
            _insured_pct   = round(100 * _insured_total / max(_th_nc_count, 1))
            _payer_label   = (
                "all cash" if _insured_total == 0
                else f"{_cash_total} cash · {_insured_total} insured"
            )

            # ── KPI ROW ───────────────────────────────────────────────────────
            # 4 independent measures: outcome | volume+location | case type | consequence
            _tkc1, _tkc2, _tkc3, _tkc4 = st.columns(4, gap="large")

            with _tkc1:
                _th_comp_col = (COLORS["danger"]  if th_cur_rate < 75
                                else COLORS["warning"] if th_cur_rate < 90
                                else COLORS["success"])
                kpi_card(
                    "Theatre Completion",
                    f"{th_cur_rate:.1f}%",
                    (f"{th_cur_lbl} · {th_pri_rate:.1f}% prior · "
                     f"{'+' if _comp_delta >= 0 else ''}{_comp_delta:.1f}pp"
                     if _comp_delta is not None else th_cur_lbl),
                    _th_comp_col,
                )

            with _tkc2:
                _th_nc_col = (COLORS["danger"]  if _th_nc_rate > 25
                              else COLORS["warning"] if _th_nc_rate > 10
                              else COLORS["muted"])
                _nc_delta_str = (
                    f" · {'+' if _th_nc_delta >= 0 else ''}{_th_nc_delta:.1f}pp vs prior"
                    if _th_nc_delta is not None else ""
                )
                # Context line: which theatre, which are clear
                _ok_str = (
                    " · ".join(f"{t}: 0 non-completions" for t in _th_ok_theatre)
                    if _th_ok_theatre else ""
                )
                kpi_card(
                    "Non-Completion",
                    f"{_th_nc_count} cases ({_th_nc_rate:.1f}%)",
                    f"{th_cur_lbl}{_nc_delta_str}",
                    _th_nc_col,
                )

            with _tkc3:
                kpi_card(
                    "Elective vs Emergency",
                    "All elective" if _th_nc_count > 0 else "—",
                    f"{th_cur_lbl} · {_th_nc_count} incomplete · 0 emergency",
                    COLORS["muted"],
                )

            with _tkc4:
                if _th_rev_exp is not None:
                    _rev_col = COLORS["danger"] if _th_rev_exp > 1_000_000 else COLORS["warning"]
                    kpi_card(
                        "Revenue Exposure",
                        f"KES {_th_rev_exp / 1_000_000:.2f}M",
                        th_cur_lbl,
                        _rev_col,
                    )
                else:
                    kpi_card("Revenue Exposure", "—", "No non-completion data", COLORS["muted"])

            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)

            # ── SECTION 1: Completion trend ───────────────────────────────────
            _th_comp_dir = (
                "Declining" if (_comp_delta is not None and _comp_delta < -2) else
                "Improving" if (_comp_delta is not None and _comp_delta >  2) else
                "Stable"
            )
            section_header(
                f"Theatre Completion {_th_comp_dir} — "
                f"{th_cur_rate:.1f}% {th_cur_lbl} "
                f"({_th_nc_count} of {_th_nc_total} sessions did not complete)"
            )
            if len(th_trend):
                _th_plot = th_trend[th_trend["SESSION_MONTH"] >= "2024-09-01"].copy()
                _th_plot = _th_plot.sort_values("SESSION_MONTH")
                _th_plot["NOT_COMPLETED"] = (
                    _th_plot["TOTAL_SESSIONS"].astype(int)
                    - _th_plot["COMPLETED_SESSIONS"].astype(int)
                )
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_bar(
                    x=_th_plot["SESSION_MONTH"], y=_th_plot["COMPLETED_SESSIONS"],
                    name="Completed", marker_color=COLORS["success"], opacity=0.65,
                    hovertemplate="%{x|%b %Y}: %{y} completed<extra></extra>",
                    secondary_y=True)
                fig.add_bar(
                    x=_th_plot["SESSION_MONTH"], y=_th_plot["NOT_COMPLETED"],
                    name="Not completed", marker_color=COLORS["danger"], opacity=0.65,
                    hovertemplate="%{x|%b %Y}: %{y} not completed<extra></extra>",
                    secondary_y=True)
                fig.add_scatter(
                    x=_th_plot["SESSION_MONTH"], y=_th_plot["COMPLETION_RATE_PCT"],
                    mode="lines+markers", name="Completion %",
                    line=dict(color=COLORS["primary"], width=2.5), marker=dict(size=5),
                    hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>",
                    secondary_y=False)
                _add_data_end_line(fig, "2025-07-01", "Jul drop")
                _add_data_end_line(fig, "2025-10-01", "Oct drop")
                fig.update_layout(**cl(
                    height=300, barmode="stack",
                    legend=dict(orientation="h", y=1.08),
                    margin=dict(l=0, r=50, t=10, b=30),
                ))
                fig.update_yaxes(title_text="Completion %", range=[0, 110],
                                 ticksuffix="%", secondary_y=False)
                fig.update_yaxes(title_text="Sessions", secondary_y=True,
                                 showgrid=False, rangemode="tozero")
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})

            # ── SECTION 2: Which theatre room? ────────────────────────────────
            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
            section_header(f"Which Theatre Room — {th_cur_lbl}")

            # Source: rpt_theatre_case_mix (validated). rpt_theatre_utilization
            # disagrees on per-theatre NC counts — use case_mix as truth.
            _th_cbt_df = P.get("th_cur_by_th", pd.DataFrame()).copy()
            _th_trt_df = P.get("th_trend_theatre", pd.DataFrame()).copy()
            if len(_th_trt_df):
                _th_trt_df.columns = [c.upper() for c in _th_trt_df.columns]
                _th_trt_df["SESSION_MONTH"] = pd.to_datetime(_th_trt_df["SESSION_MONTH"])

            if len(_th_cbt_df):
                _th_cbt_df.columns = [c.upper() for c in _th_cbt_df.columns]
                _trt_cols = st.columns(max(len(_th_cbt_df), 1), gap="large")
                for _ti, (_, _tr) in enumerate(
                    _th_cbt_df.sort_values("NON_COMPLETED_SESSIONS", ascending=False).iterrows()
                ):
                    if _ti >= len(_trt_cols):
                        break
                    with _trt_cols[_ti]:
                        _t_name = str(_tr["THEATRE_NAME"])
                        _t_tot  = int(_tr["TOTAL_SESSIONS"])
                        _t_nc   = int(_tr["NON_COMPLETED_SESSIONS"])
                        _t_comp = int(_tr["COMPLETED_SESSIONS"])
                        _t_pct  = float(_tr["COMPLETION_PCT"])
                        _t_col  = (COLORS["danger"]  if _t_nc > 0 and _t_pct < 80
                                   else COLORS["warning"] if _t_nc > 0
                                   else COLORS["success"])
                        kpi_card(
                            _t_name,
                            f"{_t_pct:.0f}%",
                            (f"{_t_comp}/{_t_tot} completed · {_t_nc} incomplete"
                             if _t_nc > 0 else f"{_t_comp}/{_t_tot} completed"),
                            _t_col,
                        )
                if len(_th_st_df) and "NON_COMPLETED" in _th_st_df.columns:
                    _nc_parts = [
                        f"{str(r['BOOKING_STATUS']).capitalize()} {int(r['NON_COMPLETED'])}/{int(r['TOTAL_SESSIONS'])}"
                        for _, r in (
                            _th_st_df[_th_st_df["NON_COMPLETED"] > 0]
                            .sort_values("NON_COMPLETED", ascending=False).iterrows()
                        )
                    ]
                    if _nc_parts:
                        dq_note(
                            "By booking status: "
                            + " · ".join(_nc_parts)
                            + ". Non-completion concentrated in approved sessions."
                        )

            # ── SECTION 3: Procedures — stacked bar, completed vs incomplete ─────
            # One chart: green (completed) + red (not completed) per procedure.
            # Sorted by failures descending. Revenue exposure in hover only.
            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
            section_header(f"Procedures — Completed vs Not Completed · {th_cur_lbl}")

            _proc_rates = P.get("th_proc_rates", pd.DataFrame()).copy()
            if len(_proc_rates):
                _proc_rates.columns = [c.upper() for c in _proc_rates.columns]
                _rev_map = (
                    _th_nc_df.groupby("PROCEDURE_NAME")["REVENUE_EXPOSURE_KES"].sum().to_dict()
                    if len(_th_nc_df) and "REVENUE_EXPOSURE_KES" in _th_nc_df.columns
                    else {}
                )
                _proc_rates["REVENUE_EXPOSURE"] = (
                    _proc_rates["PROCEDURE_NAME"].map(_rev_map).fillna(0)
                )
                _proc_all = _proc_rates.sort_values(
                    ["NON_COMPLETED_SESSIONS", "TOTAL_SESSIONS"], ascending=[False, False]
                ).reset_index(drop=True)
                # Chart: top 10 by non-completions descending
                _proc_top = _proc_all.head(10)
                _nc_hover_top = [
                    (f"<b>{p}</b><br>Not completed: {int(nc)}"
                     + (f"<br>Est. exposure: KES {int(rev):,}" if rev > 0 else ""))
                    for p, nc, rev in zip(
                        _proc_top["PROCEDURE_NAME"],
                        _proc_top["NON_COMPLETED_SESSIONS"],
                        _proc_top["REVENUE_EXPOSURE"],
                    )
                ]
                fig_proc = go.Figure()
                fig_proc.add_bar(
                    y=_proc_top["PROCEDURE_NAME"].tolist(),
                    x=_proc_top["COMPLETED_SESSIONS"].astype(int).tolist(),
                    name="Completed",
                    orientation="h",
                    marker_color=COLORS["success"],
                    opacity=0.75,
                    hovertemplate="<b>%{y}</b><br>Completed: %{x}<extra></extra>",
                )
                fig_proc.add_bar(
                    y=_proc_top["PROCEDURE_NAME"].tolist(),
                    x=_proc_top["NON_COMPLETED_SESSIONS"].astype(int).tolist(),
                    name="Not completed",
                    orientation="h",
                    marker_color=COLORS["danger"],
                    opacity=0.75,
                    customdata=_nc_hover_top,
                    hovertemplate="%{customdata}<extra></extra>",
                )
                fig_proc.update_layout(**cl(
                    barmode="stack",
                    height=max(220, len(_proc_top) * 38),
                    xaxis=dict(title="Sessions", dtick=1),
                    legend=dict(orientation="h", y=1.10),
                    margin=dict(l=0, r=0, t=4, b=30),
                ))
                st.plotly_chart(fig_proc, use_container_width=True,
                                config={"displayModeBar": False})
                if _rev_map:
                    _total_exp = int(sum(_rev_map.values()))
                    dq_note(
                        f"Top 10 by non-completions · Hover red bars for revenue exposure · "
                        f"Total estimated: KES {_total_exp:,}"
                    )
                # Full list table below the chart
                if len(_proc_all) > 0:
                    _tbl_rows = ""
                    for _, _pr in _proc_all.iterrows():
                        _pn   = str(_pr["PROCEDURE_NAME"])
                        _ptot = int(_pr["TOTAL_SESSIONS"])
                        _pnc  = int(_pr["NON_COMPLETED_SESSIONS"])
                        _ppct = float(_pr["COMPLETION_PCT"])
                        _prev = int(_pr["REVENUE_EXPOSURE"])
                        _pclr = (COLORS["danger"]  if _ppct < 80
                                 else COLORS["warning"] if _ppct < 100
                                 else COLORS["success"])
                        _pbar = max(round(_ppct), 1)
                        _tbl_rows += (
                            f'<tr style="border-bottom:1px solid #EDF2F7">'
                            f'<td style="padding:4px 8px;font-size:10.5px;color:#1A3A5C;'
                            f'max-width:200px;overflow:hidden;text-overflow:ellipsis;'
                            f'white-space:nowrap" title="{_pn}">{_pn}</td>'
                            f'<td style="padding:4px 8px;text-align:center;font-size:10px;'
                            f'color:#64748B">{_ptot}</td>'
                            f'<td style="padding:4px 8px;text-align:center;font-size:10px;'
                            f'font-weight:{"700" if _pnc > 0 else "400"};'
                            f'color:{COLORS["danger"] if _pnc > 0 else "#64748B"}">'
                            f'{"—" if _pnc == 0 else _pnc}</td>'
                            f'<td style="padding:4px 8px;text-align:right;font-size:10px;'
                            f'color:#64748B">'
                            f'{"KES " + f"{_prev:,}" if _prev > 0 else "—"}</td>'
                            f'<td style="padding:4px 8px;min-width:90px">'
                            f'<div style="display:flex;align-items:center;gap:4px">'
                            f'<div style="width:{_pbar}%;height:5px;background:{_pclr};'
                            f'border-radius:3px;min-width:3px"></div>'
                            f'<span style="font-size:10px;font-weight:600;color:{_pclr}">'
                            f'{_ppct:.0f}%</span></div></td></tr>'
                        )
                    st.markdown(
                        f'<div style="margin-top:16px;font-size:10px;font-weight:600;'
                        f'color:#64748B;margin-bottom:4px">ALL PROCEDURES · {th_cur_lbl}</div>'
                        '<div style="overflow-y:auto;max-height:300px">'
                        '<table style="width:100%;border-collapse:collapse">'
                        '<thead><tr style="background:#F4F8FC;position:sticky;top:0">'
                        '<th style="padding:4px 8px;text-align:left;font-size:10px;color:#64748B;font-weight:600">Procedure</th>'
                        '<th style="padding:4px 8px;text-align:center;font-size:10px;color:#64748B;font-weight:600">Total</th>'
                        '<th style="padding:4px 8px;text-align:center;font-size:10px;color:#64748B;font-weight:600">Incomplete</th>'
                        '<th style="padding:4px 8px;text-align:right;font-size:10px;color:#64748B;font-weight:600">Exposure</th>'
                        '<th style="padding:4px 8px;text-align:left;font-size:10px;color:#64748B;font-weight:600">Completion</th>'
                        f'</tr></thead><tbody>{_tbl_rows}</tbody></table></div>',
                        unsafe_allow_html=True,
                    )
            elif len(_th_nc_df):
                _nc_fb = (
                    _th_nc_df[_th_nc_df["PROCEDURE_NAME"] != "(no procedure recorded)"]
                    .sort_values("NON_COMPLETED_SESSIONS", ascending=False).head(10).copy()
                )
                fig_fb = go.Figure(go.Bar(
                    y=_nc_fb["PROCEDURE_NAME"].tolist(),
                    x=_nc_fb["NON_COMPLETED_SESSIONS"].astype(int).tolist(),
                    orientation="h",
                    marker_color=COLORS["danger"],
                    opacity=0.75,
                    customdata=[
                        f"<b>{p}</b><br>Not completed: {int(nc)}"
                        f"<br>Est. exposure: KES {int(rev):,}"
                        for p, nc, rev in zip(
                            _nc_fb["PROCEDURE_NAME"],
                            _nc_fb["NON_COMPLETED_SESSIONS"],
                            _nc_fb["REVENUE_EXPOSURE_KES"],
                        )
                    ],
                    hovertemplate="%{customdata}<extra></extra>",
                ))
                fig_fb.update_layout(**cl(
                    height=max(220, len(_nc_fb) * 38),
                    xaxis=dict(title="Sessions", dtick=1),
                    margin=dict(l=0, r=0, t=4, b=30),
                    showlegend=False,
                ))
                st.plotly_chart(fig_fb, use_container_width=True,
                                config={"displayModeBar": False})
                dq_note("Incomplete procedures only — completed view available after gold table rebuild.")

            # ── SECTION 4: Per-theatre completion rate over time ─────────────
            # Insight: when did Operating Theatre start declining, and did other
            # rooms track it? Session volume alone is a fact; completion rate per
            # room is the operational question.
            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
            section_header("Operating Theatre Completion Rate — Monthly Trend per Room")

            if len(_th_trt_df):
                _comp_trend = _th_trt_df.sort_values("SESSION_MONTH").copy()
                _comp_trend["COMP_PCT"] = (
                    100 * _comp_trend["COMPLETED_SESSIONS"].astype(int)
                    / _comp_trend["TOTAL_SESSIONS"].astype(int).clip(lower=1)
                ).round(1)
                _cr_colours = [COLORS["primary"], COLORS["success"], COLORS["warning"]]
                fig_cr = go.Figure()
                for _vi, _vt in enumerate(_th_trt_df["THEATRE_NAME"].unique().tolist()):
                    _vd = _comp_trend[_comp_trend["THEATRE_NAME"] == _vt]
                    fig_cr.add_scatter(
                        x=_vd["SESSION_MONTH"],
                        y=_vd["COMP_PCT"],
                        mode="lines+markers", name=_vt,
                        line=dict(color=_cr_colours[_vi % len(_cr_colours)], width=2),
                        marker=dict(size=4),
                        hovertemplate=f"{_vt}: %{{y:.1f}}% %{{x|%b %Y}}<extra></extra>",
                    )
                # Annotate current month for the room with non-completions
                if len(_th_cbt_df):
                    _worst = _th_cbt_df[_th_cbt_df["NON_COMPLETED_SESSIONS"] > 0]
                    if len(_worst):
                        _w = _worst.iloc[0]
                        _ann_x = _comp_trend["SESSION_MONTH"].max()
                        fig_cr.add_annotation(
                            x=_ann_x,
                            y=float(_w["COMPLETION_PCT"]),
                            text=(
                                f"{int(_w['NON_COMPLETED_SESSIONS'])} incomplete<br>"
                                f"{_payer_label} · all elective"
                            ),
                            showarrow=True, arrowhead=2, arrowcolor=COLORS["danger"],
                            font=dict(size=10, color=COLORS["danger"]),
                            bgcolor="white", bordercolor=COLORS["danger"], borderwidth=1,
                            ax=60, ay=-40,
                        )
                fig_cr.add_hline(y=90, line_dash="dot",
                                 line_color=COLORS["muted"], opacity=0.5,
                                 annotation_text="90% target",
                                 annotation_position="left")
                fig_cr.update_layout(**cl(
                    height=250,
                    legend=dict(orientation="h", y=1.12),
                    yaxis=dict(title="Completion %", range=[50, 105], ticksuffix="%"),
                    margin=dict(l=0, r=0, t=10, b=30),
                ))
                st.plotly_chart(fig_cr, use_container_width=True,
                                config={"displayModeBar": False})
                dq_note(
                    f"{_payer_label.capitalize()} · all elective · all in Operating Theatre. "
                    "Compare with prior months to determine if this is a new pattern or recurring."
                )

            # ── SECTION 5: Payer split ─────────────────────────────────────────
            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
            section_header(f"Payer — Cash vs Insured · {th_cur_lbl}")

            if len(_th_nc_df):
                _seg_c1, _seg_c2 = st.columns(2, gap="large")
                with _seg_c1:
                    kpi_card(
                        "Cash · Incomplete",
                        str(_cash_total),
                        f"{_cash_pct}% of non-completions · {th_cur_lbl}",
                        COLORS["danger"] if _cash_total > 0 else COLORS["muted"],
                    )
                with _seg_c2:
                    kpi_card(
                        "Insured · Incomplete",
                        str(_insured_total),
                        f"{_insured_pct}% of non-completions · {th_cur_lbl}",
                        COLORS["warning"] if _insured_total > 0 else COLORS["success"],
                    )

            # ── Bottom: consolidated data note ────────────────────────────────
            st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
            dq_note(
                "Data limitations: (1) Cancellation reasons not in EMR — booking log is the only source. "
                "(2) Approved status = last recorded EMR state; sessions may have lapsed without an update. "
                "(3) Theatre utilization rate: 8 of 966 completed sessions have operation timestamps — "
                "not calculable."
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


    # ── Tab 3: Beds & Wards ───────────────────────────────────────────────────

    with tab2:
        if facility == "TENRI":
            st.info("Bed analytics are KSH-specific — not applicable for TENRI.")
        else:
            # ── Pre-compute ───────────────────────────────────────────────────

            _bor_df = P.get("btr_bti", pd.DataFrame()).copy()
            if len(_bor_df):
                _bor_df.columns = [c.upper() for c in _bor_df.columns]

            _bm_df = P.get("beds_monthly", pd.DataFrame()).copy()
            if len(_bm_df):
                _bm_df.columns = [c.upper() for c in _bm_df.columns]

            _tat_mo = P.get("adm_tat_monthly", pd.DataFrame()).copy()
            if len(_tat_mo):
                _tat_mo.columns = [c.upper() for c in _tat_mo.columns]

            _tat_bimodal = P.get("adm_tat", pd.DataFrame()).copy()
            if len(_tat_bimodal):
                _tat_bimodal.columns = [c.upper() for c in _tat_bimodal.columns]

            _dtat_df = P.get("discharge_tat", pd.DataFrame()).copy()
            if len(_dtat_df):
                _dtat_df.columns = [c.upper() for c in _dtat_df.columns]

            # ED conversion: aggregate from per-doctor grain in P["doctor_conv"]
            _dc_raw = P.get("doctor_conv", pd.DataFrame()).copy()
            if len(_dc_raw):
                _dc_raw.columns = [c.upper() for c in _dc_raw.columns]
                _ed_conv = (
                    _dc_raw.groupby("VISIT_MONTH")
                    .agg(total_evaluations=("EVALUATIONS", "sum"),
                         total_admissions=("ADMISSIONS",  "sum"))
                    .reset_index()
                    .sort_values("VISIT_MONTH")
                )
                _ed_conv["conversion_pct"] = (
                    100 * _ed_conv["total_admissions"]
                    / _ed_conv["total_evaluations"].clip(lower=1)
                ).round(1)
            else:
                _ed_conv = pd.DataFrame()

            # Sorted months from BTR/BTI data (already excludes current partial month)
            _bor_months = (sorted(_bor_df["MONTH"].unique())
                           if len(_bor_df) and "MONTH" in _bor_df.columns else [])
            _b2_cur_mo  = _bor_months[-1] if len(_bor_months) >= 1 else None
            _b2_prv_mo  = _bor_months[-2] if len(_bor_months) >= 2 else None
            _cur_rows   = (_bor_df[_bor_df["MONTH"] == _b2_cur_mo].copy()
                           if _b2_cur_mo is not None else pd.DataFrame())
            _prv_rows   = (_bor_df[_bor_df["MONTH"] == _b2_prv_mo].copy()
                           if _b2_prv_mo is not None else pd.DataFrame())

            _PRIVATE_WARDS = {"Private Female", "Private Male", "Private Maternity"}

            # Facility-level BOR (32 total beds, hardcoded — Inv 54)
            _days_cur = pd.Timestamp(_b2_cur_mo).days_in_month if _b2_cur_mo else 30
            _days_prv = pd.Timestamp(_b2_prv_mo).days_in_month if _b2_prv_mo else 30

            _fac_bor_cur = (round(_cur_rows["TOTAL_BED_DAYS"].sum() * 100 / (32 * _days_cur), 1)
                            if len(_cur_rows) else None)
            _fac_bor_prv = (round(_prv_rows["TOTAL_BED_DAYS"].sum() * 100 / (32 * _days_prv), 1)
                            if len(_prv_rows) else None)
            _bor_delta   = (round(_fac_bor_cur - _fac_bor_prv, 1)
                            if _fac_bor_cur is not None and _fac_bor_prv is not None else None)

            # Admissions
            _adm_cur   = int(_cur_rows["TOTAL_ADMISSIONS"].sum()) if len(_cur_rows) else None
            _adm_prv   = int(_prv_rows["TOTAL_ADMISSIONS"].sum()) if len(_prv_rows) else None
            _adm_delta = (_adm_cur - _adm_prv
                          if _adm_cur is not None and _adm_prv is not None else None)

            # Avg LOS (facility level, discharge-weighted)
            _los_cur = (round(_cur_rows["TOTAL_BED_DAYS"].sum()
                              / max(_cur_rows["DISCHARGED_ADMISSIONS"].sum(), 1), 1)
                        if len(_cur_rows) else None)
            _los_prv = (round(_prv_rows["TOTAL_BED_DAYS"].sum()
                              / max(_prv_rows["DISCHARGED_ADMISSIONS"].sum(), 1), 1)
                        if len(_prv_rows) else None)
            _los_delta = (round(_los_cur - _los_prv, 1)
                          if _los_cur is not None and _los_prv is not None else None)

            # Private utilisation (% of admissions in private wards)
            _pvt_adm_cur = (int(_cur_rows[_cur_rows["WARD_NAME"].isin(_PRIVATE_WARDS)]
                                ["TOTAL_ADMISSIONS"].sum()) if len(_cur_rows) else 0)
            _pvt_adm_prv = (int(_prv_rows[_prv_rows["WARD_NAME"].isin(_PRIVATE_WARDS)]
                                ["TOTAL_ADMISSIONS"].sum()) if len(_prv_rows) else 0)
            _pvt_pct     = round(100 * _pvt_adm_cur / max(_adm_cur or 1, 1), 1) if _adm_cur else None
            _pvt_pct_prv = round(100 * _pvt_adm_prv / max(_adm_prv or 1, 1), 1) if _adm_prv else None
            _pvt_delta   = (round(_pvt_pct - _pvt_pct_prv, 1)
                            if _pvt_pct is not None and _pvt_pct_prv is not None else None)

            _b2_cur_lbl  = pd.Timestamp(_b2_cur_mo).strftime("%b %Y") if _b2_cur_mo else "—"
            _b2_prv_lbl  = pd.Timestamp(_b2_prv_mo).strftime("%b %Y") if _b2_prv_mo else "—"

            # ── KPI STRIP ─────────────────────────────────────────────────────
            _bk1, _bk2, _bk3, _bk4 = st.columns(4, gap="large")

            with _bk1:
                _bor_col = (COLORS["danger"]  if (_fac_bor_cur or 0) > 85
                            else COLORS["warning"] if (_fac_bor_cur or 0) > 70
                            else COLORS["success"])
                kpi_card(
                    "Bed Occupancy",
                    f"{_fac_bor_cur:.1f}%" if _fac_bor_cur is not None else "—",
                    (f"{_b2_cur_lbl} · {_fac_bor_prv:.1f}% prior · "
                     f"{'+' if (_bor_delta or 0) >= 0 else ''}{_bor_delta:.1f}pp"
                     if _bor_delta is not None else _b2_cur_lbl),
                    _bor_col,
                )

            with _bk2:
                _adm_col = (COLORS["success"] if (_adm_delta or 0) > 0
                            else COLORS["danger"]  if (_adm_delta or 0) < -10
                            else COLORS["muted"])
                kpi_card(
                    "Admissions",
                    f"{_adm_cur:,}" if _adm_cur is not None else "—",
                    (f"{_b2_cur_lbl} · {_adm_prv:,} prior · "
                     f"{'+' if (_adm_delta or 0) >= 0 else ''}{_adm_delta}"
                     if _adm_delta is not None else _b2_cur_lbl),
                    _adm_col,
                )

            with _bk3:
                _los_col = (COLORS["warning"] if (_los_cur or 0) > 5
                            else COLORS["muted"])
                kpi_card(
                    "Avg Length of Stay",
                    f"{_los_cur:.1f} days" if _los_cur is not None else "—",
                    (f"{_b2_cur_lbl} · {_los_prv:.1f}d prior · "
                     f"{'+' if (_los_delta or 0) >= 0 else ''}{_los_delta:.1f}d"
                     if _los_delta is not None else _b2_cur_lbl),
                    _los_col,
                )

            with _bk4:
                _pvt_col = (COLORS["success"] if (_pvt_pct or 0) > 25
                            else COLORS["warning"] if (_pvt_pct or 0) > 15
                            else COLORS["danger"])
                kpi_card(
                    "Private Utilisation",
                    f"{_pvt_pct:.1f}%" if _pvt_pct is not None else "—",
                    (f"of admissions · {_pvt_pct_prv:.1f}% prior · "
                     f"{'+' if (_pvt_delta or 0) >= 0 else ''}{_pvt_delta:.1f}pp"
                     if _pvt_delta is not None else "share of admissions"),
                    _pvt_col,
                )

            st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)

            # Flow model: structural LOS baselines per ward (Inv 88, Q1 — Oct 2024–Aug 2025)
            # Private wards excluded from state classification — volumes too low for reliable signal
            _WARD_LOS_BASELINE = {
                "General Male":      3.8,
                "General Female":    3.4,
                "General Maternity": 2.8,
                "Pediatric General": 2.7,
                "Private Male":      3.6,
                "Private Female":    3.6,
                "Private Maternity": 3.4,
            }
            _PRIVATE_WARDS_LOS_UNRELIABLE = {"Private Male", "Private Female", "Private Maternity"}

            # ── S1: BOR by Ward ───────────────────────────────────────────────
            section_header("S1 — Bed Occupancy by Ward")
            if len(_cur_rows):
                _s1_df = _cur_rows.sort_values("BOR_PCT", ascending=True)
                # BOR_PCT can come back as object dtype (e.g. Decimal values
                # from Snowflake's NUMBER type) - comparisons/formatting below
                # tolerate that, but pandas' nlargest() requires a real
                # numeric dtype regardless of the actual values, so coerce
                # once here for every use of this column in the section.
                _s1_df["BOR_PCT"] = pd.to_numeric(_s1_df["BOR_PCT"], errors="coerce")
                _s1_colors = [
                    COLORS["danger"] if v > 85
                    else COLORS["warning"] if v > 70
                    else COLORS["primary"]
                    for v in _s1_df["BOR_PCT"]
                ]
                _s1_trend_cols = st.columns([3, 2], gap="large")

                with _s1_trend_cols[0]:
                    _s1_fig = go.Figure()
                    _s1_fig.add_trace(go.Bar(
                        x=_s1_df["BOR_PCT"],
                        y=_s1_df["WARD_NAME"],
                        orientation="h",
                        marker_color=_s1_colors,
                        text=[f"{v:.1f}%" for v in _s1_df["BOR_PCT"]],
                        textposition="outside",
                        textfont=dict(size=11),
                    ))
                    _s1_fig.add_vline(x=70, line_dash="dot", line_color=COLORS["warning"], line_width=1.5)
                    _s1_fig.add_annotation(
                        x=71, y=1.0, yref="paper", xanchor="left", showarrow=False,
                        text="Watch 70%", font=dict(color=COLORS["warning"], size=10),
                    )
                    _s1_fig.add_vline(x=85, line_dash="dot", line_color=COLORS["danger"], line_width=1.5)
                    _s1_fig.add_annotation(
                        x=86, y=0.85, yref="paper", xanchor="left", showarrow=False,
                        text="Critical 85%", font=dict(color=COLORS["danger"], size=10),
                    )
                    _s1_fig.update_layout(**cl(
                        height=340,
                        margin=dict(l=160, r=80, t=10, b=10),
                        xaxis=dict(range=[0, 115], ticksuffix="%"),
                        showlegend=False,
                    ))
                    st.plotly_chart(_s1_fig, use_container_width=True, config={"displayModeBar": False})
                    st.caption(f"Ward BOR · {_b2_cur_lbl}. Facility total: 32 beds (hardcoded Inv 54).")

                with _s1_trend_cols[1]:
                    # 12-month BOR trend for top 3 wards by current occupancy
                    _s1_top3 = _s1_df.nlargest(3, "BOR_PCT")["WARD_NAME"].tolist()
                    _s1_trend = _bor_df[_bor_df["WARD_NAME"].isin(_s1_top3)].sort_values("MONTH")
                    if len(_s1_trend):
                        _s1_tf = go.Figure()
                        _s1_line_colors = [COLORS["danger"], COLORS["warning"], COLORS["primary"]]
                        for _si, _ward in enumerate(_s1_top3):
                            _wd = _s1_trend[_s1_trend["WARD_NAME"] == _ward]
                            _s1_tf.add_trace(go.Scatter(
                                x=_wd["MONTH"], y=_wd["BOR_PCT"],
                                mode="lines+markers", name=_ward,
                                line=dict(width=2, color=_s1_line_colors[_si]),
                                marker=dict(size=5),
                            ))
                        _s1_tf.add_hline(y=85, line_dash="dot", line_color=COLORS["danger"],
                                         line_width=1)
                        _s1_tf.add_hline(y=70, line_dash="dot", line_color=COLORS["warning"],
                                         line_width=1)
                        _s1_tf.update_layout(**cl(
                            height=280,
                            margin=dict(l=0, r=20, t=10, b=10),
                            yaxis=dict(ticksuffix="%"),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                        ))
                        st.plotly_chart(_s1_tf, use_container_width=True,
                                        config={"displayModeBar": False})

                _top_ward  = _s1_df.iloc[-1]
                _tw_bor    = _top_ward["BOR_PCT"]
                _tw_state  = ("over capacity — no buffer for emergency intake"
                              if _tw_bor > 85 else "approaching threshold"
                              if _tw_bor > 70 else "within range")
                st.caption(
                    f"▸ **{_top_ward['WARD_NAME']}** leads at **{_tw_bor:.1f}%** BOR "
                    f"in {_b2_cur_lbl} — {_tw_state}. "
                    "BOR = admissions × LOS ÷ beds. "
                    "S2 shows the inflow volume and conversion rate. "
                    "S3 shows admission speed. S4 shows the LOS multiplier per ward. "
                    "S6 synthesises all into an operational state per ward."
                )

            # ── S2: Admissions Demand + ED Conversion ─────────────────────────
            section_header("S2 — Admissions Demand and ED Conversion")
            if len(_bor_df):
                _s2_adm = (
                    _bor_df.groupby("MONTH")["TOTAL_ADMISSIONS"].sum()
                    .reset_index()
                    .sort_values("MONTH")
                    .rename(columns={"MONTH": "mo", "TOTAL_ADMISSIONS": "admissions"})
                )

                _s2_fig = go.Figure()
                _s2_fig.add_trace(go.Bar(
                    x=_s2_adm["mo"], y=_s2_adm["admissions"],
                    name="Monthly Admissions",
                    marker_color=COLORS["primary"],
                    opacity=0.75,
                ))

                if len(_ed_conv):
                    _s2_ec_plot = (
                        _ed_conv[_ed_conv["VISIT_MONTH"].isin(_s2_adm["mo"])]
                        .sort_values("VISIT_MONTH")
                    )
                    _s2_fig.add_trace(go.Scatter(
                        x=_s2_ec_plot["VISIT_MONTH"],
                        y=_s2_ec_plot["conversion_pct"],
                        name="ED Conversion %",
                        mode="lines+markers",
                        line=dict(color=COLORS["coral"], width=2),
                        marker=dict(size=6),
                        yaxis="y2",
                    ))

                _s2_fig.update_layout(**cl(
                    height=300,
                    margin=dict(l=0, r=60, t=10, b=10),
                    yaxis=dict(title="Admissions"),
                    yaxis2=dict(title="ED Conversion %", overlaying="y", side="right",
                                ticksuffix="%", showgrid=False, range=[0, 30]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                ))
                st.plotly_chart(_s2_fig, use_container_width=True, config={"displayModeBar": False})

                if len(_ed_conv) >= 3:
                    _ec_last  = _ed_conv.sort_values("VISIT_MONTH").iloc[-1]
                    _ec_3mo   = _ed_conv.sort_values("VISIT_MONTH").iloc[-3:]["conversion_pct"].mean()
                    st.caption(
                        f"▸ ED Conversion: **{_ec_last['conversion_pct']:.1f}%** latest month · "
                        f"3-month avg **{_ec_3mo:.1f}%**. "
                        "Admissions are one of two BOR multipliers (S1). "
                        "Higher conversion at high BOR = throughput risk: "
                        "more entries compound pressure when LOS (S4) is also elevated. "
                        "See S3 for how fast those admissions are being processed."
                    )

            # ── S3: Admission TAT ─────────────────────────────────────────────
            section_header("S3 — Admission TAT — How Fast Are We Admitting")

            # Pre-compute: load DOW data, aggregate across tracked months
            _dow_raw = P.get("adm_tat_dow", pd.DataFrame()).copy()
            if len(_dow_raw):
                _dow_raw.columns = [c.upper() for c in _dow_raw.columns]
                _dow_df = (
                    _dow_raw.groupby(["DAY_NUM", "DAY_NAME"])
                    .agg(
                        total_evaluations=("TOTAL_EVALUATIONS", "sum"),
                        total_admissions=("TOTAL_ADMISSIONS",  "sum"),
                    )
                    .reset_index()
                    .sort_values("DAY_NUM")
                )
                _dow_df["conversion_pct"] = (
                    100 * _dow_df["total_admissions"]
                    / _dow_df["total_evaluations"].clip(lower=1)
                ).round(1)
                # p50_tat_min: admission-weighted median TAT across months
                _dow_p50 = (
                    _dow_raw.groupby("DAY_NUM")
                    .apply(lambda g: round(
                        (g["P50_TAT_MIN"] * g["TOTAL_ADMISSIONS"]).sum()
                        / g["TOTAL_ADMISSIONS"].clip(lower=1).sum(), 0
                    ))
                    .reset_index(name="p50_tat_min")
                )
                _dow_df = _dow_df.merge(_dow_p50, on="DAY_NUM", how="left")
                _dow_months = sorted(_dow_raw["TAT_MONTH"].unique())
                _dow_lbl = (f"{pd.Timestamp(_dow_months[0]).strftime('%b %Y')} – "
                            f"{pd.Timestamp(_dow_months[-1]).strftime('%b %Y')}"
                            if len(_dow_months) > 1
                            else pd.Timestamp(_dow_months[0]).strftime("%b %Y")
                            if len(_dow_months) == 1 else "recent months")
            else:
                _dow_df  = pd.DataFrame()
                _dow_lbl = "recent months"

            if len(_dow_df):
                _p50_colors = [
                    COLORS["success"] if v <= 60
                    else COLORS["warning"] if v <= 120
                    else COLORS["danger"]
                    for v in _dow_df["p50_tat_min"]
                ]
                _combined_fig = go.Figure()
                _combined_fig.add_trace(go.Bar(
                    name="Evaluations",
                    x=_dow_df["DAY_NAME"],
                    y=_dow_df["total_evaluations"],
                    marker_color=COLORS["primary"],
                    opacity=0.75,
                    text=[f"{int(v):,}" for v in _dow_df["total_evaluations"]],
                    textposition="outside",
                    textfont=dict(size=9),
                    yaxis="y",
                    offsetgroup=1,
                ))
                _combined_fig.add_trace(go.Bar(
                    name="Median TAT",
                    x=_dow_df["DAY_NAME"],
                    y=_dow_df["p50_tat_min"],
                    marker_color=_p50_colors,
                    text=[f"{int(v)} min" for v in _dow_df["p50_tat_min"]],
                    textposition="outside",
                    textfont=dict(size=9),
                    yaxis="y2",
                    offsetgroup=2,
                ))
                _combined_fig.update_layout(**cl(
                    barmode="group",
                    height=340,
                    margin=dict(l=0, r=55, t=30, b=50),
                    legend=dict(orientation="h", x=0.5, xanchor="center",
                                y=-0.18, font=dict(size=10)),
                    yaxis=dict(title="Visits", showgrid=False),
                    yaxis2=dict(
                        title="Median TAT (min)",
                        overlaying="y",
                        side="right",
                        ticksuffix=" min",
                        showgrid=False,
                        range=[0, _dow_df["p50_tat_min"].max() * 1.4],
                    ),
                    title=dict(text=f"Traffic & Admission TAT by day · {_dow_lbl}",
                               font=dict(size=11), x=0),
                ))
                st.plotly_chart(_combined_fig, use_container_width=True,
                                config={"displayModeBar": False})

            if len(_tat_mo):
                _tat_latest  = _tat_mo.sort_values("TAT_MONTH").iloc[-1]
                _tat_fast_v  = _tat_latest["FAST_PCT"]
                _tat_p50_v   = _tat_latest["P50_TAT_MIN"]
                _tat_state_v = ("strong" if _tat_fast_v >= 55
                                else "moderate" if _tat_fast_v >= 45 else "weak")
            if len(_dow_df):
                _fastest_day = _dow_df.sort_values("p50_tat_min").iloc[0]
                _slowest_day = _dow_df.sort_values("p50_tat_min", ascending=False).iloc[0]
                _busy_day    = _dow_df.sort_values("total_evaluations", ascending=False).iloc[0]
                _same        = _fastest_day["DAY_NAME"] == _busy_day["DAY_NAME"]
            if len(_dow_df):
                st.caption(
                    f"▸ Fastest day: **{_fastest_day['DAY_NAME']}** "
                    f"(median {int(_fastest_day['p50_tat_min'])} min · "
                    f"{_fastest_day['total_evaluations']:,.0f} visits · n={int(_fastest_day['total_admissions'])}). "
                    f"Slowest day: **{_slowest_day['DAY_NAME']}** "
                    f"(median {int(_slowest_day['p50_tat_min'])} min · n={int(_slowest_day['total_admissions'])}). "
                    f"Busiest day: **{_busy_day['DAY_NAME']}** ({_busy_day['total_evaluations']:,.0f} visits). "
                    + ("Busiest and fastest are the same day — volume is not the bottleneck here. "
                       if _same else
                       "Fastest and busiest are different days — TAT is not driven by volume (Inv 89). ")
                )

            # ── S6: Discharge TAT ─────────────────────────────────────────────
            section_header("S4 — Length of Stay — What Is Holding Beds")
            if len(_cur_rows) and len(_prv_rows):
                _s3_cur = _cur_rows.copy()
                _s3_cur["avg_los"] = (
                    _s3_cur["TOTAL_BED_DAYS"]
                    / _s3_cur["DISCHARGED_ADMISSIONS"].clip(lower=1)
                ).round(1)
                _s3_prv = _prv_rows.copy()
                _s3_prv["avg_los"] = (
                    _s3_prv["TOTAL_BED_DAYS"]
                    / _s3_prv["DISCHARGED_ADMISSIONS"].clip(lower=1)
                ).round(1)

                _s3_merged = (
                    _s3_cur[["WARD_NAME", "avg_los"]]
                    .merge(_s3_prv[["WARD_NAME", "avg_los"]].rename(columns={"avg_los": "prv_los"}),
                           on="WARD_NAME", how="left")
                    .sort_values("avg_los", ascending=True)
                )
                # Attach ward-specific structural baseline from Inv 88 Q1
                _s3_merged["baseline_los"] = _s3_merged["WARD_NAME"].map(_WARD_LOS_BASELINE)
                _s3_merged["above_baseline"] = _s3_merged["avg_los"] > _s3_merged["baseline_los"]

                _s3_bar_colors = [
                    COLORS["warning"] if ab else COLORS["primary"]
                    for ab in _s3_merged["above_baseline"]
                ]

                _s3_fig = go.Figure()
                _s3_fig.add_trace(go.Bar(
                    x=_s3_merged["avg_los"],
                    y=_s3_merged["WARD_NAME"],
                    orientation="h",
                    name=_b2_cur_lbl,
                    marker_color=_s3_bar_colors,
                    text=[f"{v:.1f}d" for v in _s3_merged["avg_los"]],
                    textposition="outside",
                ))
                _s3_fig.add_trace(go.Scatter(
                    x=_s3_merged["prv_los"],
                    y=_s3_merged["WARD_NAME"],
                    mode="markers",
                    name=_b2_prv_lbl,
                    marker=dict(symbol="diamond", size=10, color=COLORS["muted"], opacity=0.9),
                ))
                _s3_fig.add_trace(go.Scatter(
                    x=_s3_merged["baseline_los"],
                    y=_s3_merged["WARD_NAME"],
                    mode="markers",
                    name="Ward baseline (Inv 88)",
                    marker=dict(symbol="line-ns", size=14,
                                color=COLORS["coral"], opacity=0.85,
                                line=dict(width=2, color=COLORS["coral"])),
                ))
                _s3_fig.update_layout(**cl(
                    height=320,
                    margin=dict(l=160, r=80, t=10, b=10),
                    xaxis=dict(title="Avg LOS (days)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                ))
                st.plotly_chart(_s3_fig, use_container_width=True, config={"displayModeBar": False})

                _s3_above = _s3_merged[_s3_merged["above_baseline"] & ~_s3_merged["WARD_NAME"].isin(_PRIVATE_WARDS_LOS_UNRELIABLE)]
                _s3_high  = _s3_merged.sort_values("avg_los").iloc[-1]
                _s3_above_names = ", ".join(_s3_above["WARD_NAME"].tolist()) if len(_s3_above) else "none"
                st.caption(
                    f"▸ LOS is the second BOR multiplier (S1) — after inflow volume (S2) and admission speed (S3). "
                    f"Wards above their structural baseline (coral tick): **{_s3_above_names}**. "
                    "High BOR months correlate with longer LOS within the same ward (Inv 88). "
                    "S5 shows the discharge side. "
                    "Private ward LOS is unreliable at current volumes (1–12 patients/month)."
                )

            # ── S4: Ward Operational State ─────────────────────────────────────
            section_header("S5 — Discharge TAT — How Fast Are We Releasing Beds")
            # ── S5 DOW: Admissions vs Discharges by day (operational) ──────────
            _dis_dow_raw = P.get("discharge_dow", pd.DataFrame()).copy()
            _adm_dow_raw = P.get("adm_tat_dow",   pd.DataFrame()).copy()
            if len(_dis_dow_raw) and len(_adm_dow_raw):
                _dis_dow_raw.columns = [c.upper() for c in _dis_dow_raw.columns]
                _adm_dow_raw.columns = [c.upper() for c in _adm_dow_raw.columns]
                _adm_by_day = (
                    _adm_dow_raw.groupby("DAY_NUM")["TOTAL_ADMISSIONS"].sum()
                    .reset_index().rename(columns={"TOTAL_ADMISSIONS": "admissions"})
                )
                _dis_by_day = _dis_dow_raw[["DAY_NUM", "DAY_NAME", "TOTAL_DISCHARGES"]].copy()
                _dis_by_day = _dis_by_day.rename(columns={"TOTAL_DISCHARGES": "discharges"})
                _flow_df    = _dis_by_day.merge(_adm_by_day, on="DAY_NUM").sort_values("DAY_NUM")
                _flow_df["net"] = _flow_df["admissions"] - _flow_df["discharges"]

                _dis_mo_lbl = pd.Timestamp(
                    _dis_dow_raw["DISCHARGE_MONTH"].iloc[0]
                ).strftime("%b %Y")

                _flow_fig = go.Figure()
                _flow_fig.add_trace(go.Bar(
                    name="Admissions",
                    x=_flow_df["DAY_NAME"],
                    y=_flow_df["admissions"],
                    marker_color=COLORS["primary"],
                    opacity=0.85,
                    text=[str(int(v)) for v in _flow_df["admissions"]],
                    textposition="outside",
                    textfont=dict(size=9),
                    offsetgroup=1,
                ))
                _flow_fig.add_trace(go.Bar(
                    name="Discharges",
                    x=_flow_df["DAY_NAME"],
                    y=_flow_df["discharges"],
                    marker_color=COLORS["success"],
                    opacity=0.85,
                    text=[str(int(v)) for v in _flow_df["discharges"]],
                    textposition="outside",
                    textfont=dict(size=9),
                    offsetgroup=2,
                ))
                _flow_fig.update_layout(**cl(
                    barmode="group",
                    height=300,
                    margin=dict(l=0, r=10, t=30, b=50),
                    legend=dict(orientation="h", x=0.5, xanchor="center",
                                y=-0.18, font=dict(size=10)),
                    yaxis=dict(title="Patients", showgrid=False),
                    title=dict(
                        text=f"Admissions vs Discharges by day · {_dis_mo_lbl}",
                        font=dict(size=11), x=0,
                    ),
                ))
                st.plotly_chart(_flow_fig, use_container_width=True,
                                config={"displayModeBar": False})

                _net_fill  = _flow_df[_flow_df["net"] > 0]
                _net_drain = _flow_df[_flow_df["net"] < 0]
                _fill_days  = ", ".join(
                    f"{r['DAY_NAME']} (+{int(r['net'])})"
                    for _, r in _net_fill.sort_values("net", ascending=False).iterrows()
                )
                _drain_days = ", ".join(
                    f"{r['DAY_NAME']} ({int(r['net'])})"
                    for _, r in _net_drain.sort_values("net").iterrows()
                )
                st.caption(
                    f"▸ **Bed-filling days** ({_dis_mo_lbl}): {_fill_days}. "
                    f"**Bed-clearing days**: {_drain_days}. "
                    "Mid-week accumulation (Wednesday and Friday admit more than they discharge) "
                    "raises inpatient census to its weekly peak by Friday — confirmed by daily census "
                    "count (Inv 91 Q3: Friday 47 unique patients, Tuesday 30). "
                    "This census pressure is the root cause of the Friday admission TAT floor "
                    "(p25 = 84 min — every patient waits regardless of clinical complexity). "
                    "Intervention: coordinate Wednesday discharge planning to prevent mid-week accumulation."
                )

            # ── S5 historical: discharge TAT trend ────────────────────────────
            if len(_dtat_df):
                dq_note(
                    "Discharge TAT: pre-departure cohort only (~50%). "
                    "Data window Oct 2024–Aug 2025 — discharge request sync paused Sep 2025."
                )
                _s6_df     = _dtat_df.sort_values("DISCHARGE_MONTH")
                _s6_latest = _s6_df.iloc[-1]
                _s6_mo_lbl = pd.Timestamp(_s6_latest["DISCHARGE_MONTH"]).strftime("%b %Y")

                _s6_combo = go.Figure()
                _s6_combo.add_trace(go.Bar(
                    name="Median TAT (hrs)",
                    x=_s6_df["DISCHARGE_MONTH"],
                    y=_s6_df["MEDIAN_TAT_HOURS"],
                    marker_color=[
                        COLORS["danger"] if v > 4
                        else COLORS["warning"] if v > 2
                        else COLORS["success"]
                        for v in _s6_df["MEDIAN_TAT_HOURS"]
                    ],
                    text=[f"{v:.1f}h" for v in _s6_df["MEDIAN_TAT_HOURS"]],
                    textposition="outside",
                    textfont=dict(size=9),
                    yaxis="y",
                    offsetgroup=1,
                ))
                _s6_combo.add_trace(go.Scatter(
                    name="Delayed >4h (%)",
                    x=_s6_df["DISCHARGE_MONTH"],
                    y=_s6_df["DELAYED_PCT"],
                    mode="lines+markers",
                    line=dict(color=COLORS["warning"], width=2),
                    marker=dict(size=6),
                    yaxis="y2",
                ))
                _s6_combo.update_layout(**cl(
                    height=260,
                    margin=dict(l=0, r=55, t=10, b=50),
                    legend=dict(orientation="h", x=0.5, xanchor="center",
                                y=-0.22, font=dict(size=10)),
                    yaxis=dict(title="Median TAT (hrs)", showgrid=False),
                    yaxis2=dict(
                        title="Delayed >4h (%)",
                        overlaying="y",
                        side="right",
                        ticksuffix="%",
                        showgrid=False,
                        range=[0, max(_s6_df["DELAYED_PCT"].max() * 1.4, 30)],
                    ),
                ))
                st.plotly_chart(_s6_combo, use_container_width=True,
                                config={"displayModeBar": False})
                st.caption(
                    f"▸ Historical context (last data: {_s6_mo_lbl}): "
                    f"**{_s6_latest['MEDIAN_TAT_HOURS']:.1f}h** median discharge TAT · "
                    f"**{_s6_latest['DELAYED_PCT']:.0f}%** delayed >4h. "
                    "Discharge TAT is an operational congestion signal — it does not drive LOS or BOR "
                    "(Inv 87, Inv 88 — tested and not confirmed)."
                )

            # ── S6: Ward Operational State ────────────────────────────────────
            section_header("S6 — Ward Operational State — Synthesis")
            if len(_cur_rows) and len(_prv_rows):
                def _ward_state_label(bor, los, adm_cur_v, adm_prv_v, ward_name):
                    _baseline = _WARD_LOS_BASELINE.get(ward_name, 3.6)
                    _adm_chg  = 100 * (adm_cur_v - adm_prv_v) / max(adm_prv_v, 1)
                    if bor > 85:
                        return "Capacity Compression", COLORS["danger"]
                    elif _adm_chg > 10 and bor > 50:
                        return "Demand Growth", COLORS["warning"]
                    elif bor > 60 and los > _baseline * 1.25:
                        return "Long-Stay Bottleneck", COLORS["warning"]
                    elif 40 <= bor <= 80 and los <= _baseline * 1.15:
                        return "Efficient Throughput", COLORS["success"]
                    else:
                        return "Low Utilisation", COLORS["muted"]

                _s4_cur = _cur_rows.copy()
                _s4_cur["avg_los"] = (
                    _s4_cur["TOTAL_BED_DAYS"]
                    / _s4_cur["DISCHARGED_ADMISSIONS"].clip(lower=1)
                ).round(1)
                _s4_prv = _prv_rows[["WARD_NAME", "TOTAL_ADMISSIONS", "BOR_PCT"]].rename(
                    columns={"TOTAL_ADMISSIONS": "prv_admissions", "BOR_PCT": "prv_bor_pct"}
                )
                _s4_df  = _s4_cur.merge(_s4_prv, on="WARD_NAME", how="left")
                _s4_df["prv_admissions"] = _s4_df["prv_admissions"].fillna(_s4_df["TOTAL_ADMISSIONS"])
                _s4_df["bor_delta"]      = (_s4_df["BOR_PCT"] - _s4_df["prv_bor_pct"]).round(1)

                _s4_wards = _s4_df.reset_index(drop=True)
                _s4_n     = len(_s4_wards)
                _s4_ncols = 4
                _s4_nrows = (_s4_n + _s4_ncols - 1) // _s4_ncols

                for _s4_r in range(_s4_nrows):
                    _s4_chunk  = _s4_wards.iloc[_s4_r * _s4_ncols: (_s4_r + 1) * _s4_ncols]
                    _s4_gr     = st.columns(len(_s4_chunk), gap="small")
                    for _s4_ci, (_, _row) in enumerate(_s4_chunk.iterrows()):
                        _s4_state, _s4_color = _ward_state_label(
                            bor=_row.get("BOR_PCT", 0),
                            los=_row.get("avg_los", 0),
                            adm_cur_v=_row.get("TOTAL_ADMISSIONS", 0),
                            adm_prv_v=_row.get("prv_admissions", _row.get("TOTAL_ADMISSIONS", 0)),
                            ward_name=_row.get("WARD_NAME", ""),
                        )
                        with _s4_gr[_s4_ci]:
                            _bor_d = _row.get("bor_delta")
                            if pd.notna(_bor_d) and _bor_d != 0:
                                _bor_dir   = "↑" if _bor_d > 0 else "↓"
                                _delta_html = (
                                    f"<div style='font-size:10px;color:#6B8CAE;margin-top:3px'>"
                                    f"{_bor_dir} {abs(_bor_d):.0f}pp MoM</div>"
                                )
                            else:
                                _delta_html = ""
                            st.markdown(
                                f"<div style='border-left:4px solid {_s4_color};padding:10px 14px;"
                                f"background:#F4F8FC;border-radius:4px;margin-bottom:8px'>"
                                f"<div style='font-size:10px;font-weight:700;color:{_s4_color};"
                                f"text-transform:uppercase;letter-spacing:1px'>{_s4_state}</div>"
                                f"<div style='font-size:13px;font-weight:600;color:#003467;"
                                f"margin:3px 0'>{_row['WARD_NAME']}</div>"
                                f"<div style='font-size:11px;color:#6B8CAE'>"
                                f"BOR {_row.get('BOR_PCT', 0):.0f}% · "
                                f"LOS {_row.get('avg_los', 0):.1f}d · "
                                f"{int(_row.get('TOTAL_ADMISSIONS', 0))} adm</div>"
                                f"{_delta_html}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

            # ── S7: Bed Mix — Where Patients Go and Why ───────────────────────
            section_header("S7 — Bed Mix — Where Patients Go and Why")
            if len(_cur_rows):
                # _cur_rows from q_btr_bti_monthly: has BED_COUNT, BTI_DAYS, INSURED_ADMISSIONS
                _s7_wr = _cur_rows.copy()
                _s7_wr["insured_pct"] = (
                    _s7_wr["INSURED_ADMISSIONS"] * 100.0
                    / _s7_wr["TOTAL_ADMISSIONS"].clip(lower=1)
                ).round(1)
                _s7_wr = _s7_wr.dropna(subset=["BED_COUNT"])

                _s7_pvt_wr = (_s7_wr[_s7_wr["WARD_NAME"].isin(_PRIVATE_WARDS)]
                              .sort_values("BTI_DAYS", ascending=False))
                _s7_gen_wr = (_s7_wr[~_s7_wr["WARD_NAME"].isin(_PRIVATE_WARDS)]
                              .sort_values("insured_pct", ascending=False))

                _worst_bti     = _s7_pvt_wr.iloc[0] if len(_s7_pvt_wr) else None
                _gen_insured   = (_s7_wr[~_s7_wr["WARD_NAME"].isin(_PRIVATE_WARDS)]
                                  ["INSURED_ADMISSIONS"].sum())
                _total_insured = _s7_wr["INSURED_ADMISSIONS"].sum()
                _insured_to_gen_pct = round(
                    _gen_insured * 100.0 / max(_total_insured, 1), 1
                )
                _cur_mo_lbl = (pd.Timestamp(_b2_cur_mo).strftime("%b %Y")
                               if _b2_cur_mo else "latest month")
                _prv_mo_lbl = (pd.Timestamp(_b2_prv_mo).strftime("%b %Y")
                               if _b2_prv_mo else "prior month")

                # Prior-month BTI for private wards (for delta + grouped chart)
                _prv_pvt_idx = (
                    _prv_rows[_prv_rows["WARD_NAME"].isin(_PRIVATE_WARDS)]
                    .set_index("WARD_NAME")["BTI_DAYS"]
                    if len(_prv_rows) else pd.Series(dtype=float)
                )
                _worst_bti_delta = None
                if _worst_bti is not None and _worst_bti["WARD_NAME"] in _prv_pvt_idx.index:
                    _worst_bti_delta = round(
                        _worst_bti["BTI_DAYS"] - _prv_pvt_idx[_worst_bti["WARD_NAME"]], 0
                    )

                # ── KPI strip ─────────────────────────────────────────────────
                _s7k1, _s7k2, _s7k3 = st.columns(3, gap="large")
                with _s7k1:
                    if _worst_bti is not None:
                        _kpi1_sub = f"{_worst_bti['WARD_NAME']} · {_cur_mo_lbl}"
                        if _worst_bti_delta is not None and _worst_bti_delta != 0:
                            _d_dir = "↓" if _worst_bti_delta < 0 else "↑"
                            _kpi1_sub += f" · {_d_dir} {abs(_worst_bti_delta):.0f}d MoM"
                    else:
                        _kpi1_sub = ""
                    kpi_card(
                        "Worst Private Idle Time",
                        f"{_worst_bti['BTI_DAYS']:.0f} days" if _worst_bti is not None else "—",
                        _kpi1_sub,
                        COLORS["coral"],
                    )
                with _s7k2:
                    kpi_card(
                        "Insured Routed to General",
                        f"{_insured_to_gen_pct:.1f}%",
                        f"of all insured admissions · {_cur_mo_lbl}",
                        COLORS["warning"],
                    )
                with _s7k3:
                    kpi_card(
                        "Bed-Day Revenue Differential",
                        "~2×",
                        "private vs general · directional · accommodation only (Inv 16)",
                        COLORS["muted"],
                    )

                st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

                # ── Row 2: two panels answering different questions ────────────
                _s7_left, _s7_right = st.columns([55, 45], gap="large")

                with _s7_left:
                    if len(_s7_pvt_wr):
                        _prv_pvt_df = (
                            _prv_rows[_prv_rows["WARD_NAME"].isin(_PRIVATE_WARDS)]
                            [["WARD_NAME", "BTI_DAYS"]]
                            .rename(columns={"BTI_DAYS": "prv_bti_days"})
                            if len(_prv_rows) else pd.DataFrame()
                        )
                        _s7_pvt_chart = _s7_pvt_wr.merge(
                            _prv_pvt_df, on="WARD_NAME", how="left"
                        )
                        _bti_fig = go.Figure()
                        _bti_fig.add_trace(go.Bar(
                            x=_s7_pvt_chart["BTI_DAYS"],
                            y=_s7_pvt_chart["WARD_NAME"],
                            orientation="h",
                            name=_cur_mo_lbl,
                            marker_color=COLORS["coral"],
                            opacity=0.85,
                            text=[f"{v:.0f}d" for v in _s7_pvt_chart["BTI_DAYS"]],
                            textposition="outside",
                            textfont=dict(size=10),
                        ))
                        if "prv_bti_days" in _s7_pvt_chart.columns:
                            _bti_fig.add_trace(go.Bar(
                                x=_s7_pvt_chart["prv_bti_days"],
                                y=_s7_pvt_chart["WARD_NAME"],
                                orientation="h",
                                name=_prv_mo_lbl,
                                marker_color=COLORS["muted"],
                                opacity=0.55,
                                text=[
                                    f"{v:.0f}d" if pd.notna(v) else ""
                                    for v in _s7_pvt_chart["prv_bti_days"]
                                ],
                                textposition="outside",
                                textfont=dict(size=10),
                            ))
                        _bti_fig.update_layout(**cl(
                            height=250,
                            margin=dict(l=10, r=70, t=32, b=30),
                            barmode="group",
                            xaxis=dict(title="Days idle between patients",
                                       showgrid=False),
                            yaxis=dict(showgrid=False),
                            title=dict(
                                text=f"Bed idle time by ward — {_prv_mo_lbl} vs {_cur_mo_lbl}",
                                font=dict(size=11), x=0,
                            ),
                            legend=dict(orientation="h", y=1.12, x=1, xanchor="right",
                                        yanchor="top", font=dict(size=10)),
                        ))
                        st.plotly_chart(_bti_fig, use_container_width=True,
                                        config={"displayModeBar": False})

                with _s7_right:
                    st.markdown(
                        f"<div style='font-size:11px;font-weight:700;color:#003467;"
                        f"text-transform:uppercase;letter-spacing:0.8px;"
                        f"margin-bottom:10px'>Routing gap — insured in general · {_cur_mo_lbl}</div>",
                        unsafe_allow_html=True,
                    )
                    for _, _gr in _s7_gen_wr.iterrows():
                        _ins_pct = _gr["insured_pct"]
                        _ins_color = (COLORS["warning"] if _ins_pct >= 75
                                      else COLORS["muted"])
                        st.markdown(
                            f"<div style='display:flex;justify-content:space-between;"
                            f"align-items:center;padding:7px 10px;margin-bottom:4px;"
                            f"background:#F4F8FC;border-radius:4px;"
                            f"border-left:3px solid {_ins_color}'>"
                            f"<span style='font-size:12px;color:#003467;font-weight:500'>"
                            f"{_gr['WARD_NAME']}</span>"
                            f"<span style='font-size:12px;font-weight:700;color:{_ins_color}'>"
                            f"{_ins_pct:.0f}% insured</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    dq_note(
                        "Whether insured patients in general wards were eligible for "
                        "private accommodation is not established from available data."
                    )

                # ── Caption ───────────────────────────────────────────────────
                st.caption(
                    f"▸ {_cur_mo_lbl}: private beds idle "
                    f"{_s7_pvt_wr.iloc[0]['BTI_DAYS']:.0f}–"
                    f"{_s7_pvt_wr.iloc[-1]['BTI_DAYS']:.0f} days between patients "
                    f"(compare {_prv_mo_lbl} bars to see direction). "
                    f"{_insured_to_gen_pct:.1f}% of insured admissions went to general wards. "
                    "Whether those patients were eligible for private accommodation is not "
                    "established from available data. "
                    "Observed gap: private capacity is idle while insured admissions remain "
                    "concentrated in general wards. "
                    "Cause — tier restrictions, admissions workflow, or clinician choice — "
                    "is untested (Inv CD10)."
                )

    # ── Tab 1: Lab Operations ─────────────────────────────────────────────────

    with tab3:

        # ── Lab Operations (KSH only) ─────────────────────────────────────────
        if _is_ksh_p3:
            section_header("Lab Operations")

            _lab_fd       = P["lab_flow_delta"].copy()
            _lab_hd       = P["lab_handoff"].copy()
            _lab_wk       = P["lab_weekly"].copy()
            _lab_tat_test = P["lab_tat_test"].copy()

            # ── Helper: extract delta value (last_28 vs prior_28) ─────────────
            def _fd_val(df, stage, col, period):
                r = df[(df["STAGE"] == stage) & (df["PERIOD"] == period)]
                return r.iloc[0][col] if len(r) else None

            def _hd_val(df, dest, col, period):
                r = df[(df["NEXT_STAGE"] == dest) & (df["PERIOD"] == period)]
                return r.iloc[0][col] if len(r) else None

            def _delta_str(cur, prv, unit="min", pct=False):
                if cur is None or prv is None:
                    return ""
                d = cur - prv
                sign = "↑" if d > 0 else "↓"
                col  = COLORS["danger"] if d > 0 else COLORS["success"]
                val  = f"{abs(d):.0f}{unit}"
                return sign, col, val

            # ── Pre-compute all delta values ───────────────────────────────────
            _lab_vis_cur  = _fd_val(_lab_fd, "laboratory", "VISITS",          "last_28")
            _lab_vis_prv  = _fd_val(_lab_fd, "laboratory", "VISITS",          "prior_28")
            _lab_q_cur    = _fd_val(_lab_fd, "laboratory", "MEDIAN_WAIT_MIN", "last_28")
            _lab_q_prv    = _fd_val(_lab_fd, "laboratory", "MEDIAN_WAIT_MIN", "prior_28")
            _pharm_w_cur  = _fd_val(_lab_fd, "pharmacy",   "MEDIAN_WAIT_MIN", "last_28")
            _pharm_w_prv  = _fd_val(_lab_fd, "pharmacy",   "MEDIAN_WAIT_MIN", "prior_28")
            _pharm_o30_cur = _fd_val(_lab_fd, "pharmacy",  "PCT_OVER_30MIN",  "last_28")

            _lp_gap_cur   = _hd_val(_lab_hd, "pharmacy",  "MEDIAN_GAP_MIN",  "last_28")
            _lp_gap_prv   = _hd_val(_lab_hd, "pharmacy",  "MEDIAN_GAP_MIN",  "prior_28")
            _lp_tr_cur    = _hd_val(_lab_hd, "pharmacy",  "TRANSITIONS",     "last_28")
            _lp_tr_prv    = _hd_val(_lab_hd, "pharmacy",  "TRANSITIONS",     "prior_28")

            _lr_gap_cur   = _hd_val(_lab_hd, "radiology", "MEDIAN_GAP_MIN",  "last_28")
            _lr_gap_prv   = _hd_val(_lab_hd, "radiology", "MEDIAN_GAP_MIN",  "prior_28")
            _lr_tr_cur    = _hd_val(_lab_hd, "radiology", "TRANSITIONS",     "last_28")
            _lr_tr_prv    = _hd_val(_lab_hd, "radiology", "TRANSITIONS",     "prior_28")

            def _kpi_delta_card(col, label, cur, prv, unit="min",
                                base_color=None, danger_fn=None):
                """Render a KPI card with delta vs prior period."""
                val_str = f"{int(cur)} {unit}" if cur is not None else "—"
                if cur is not None and prv is not None:
                    d = int(cur) - int(prv)
                    arrow = "↑" if d > 0 else "↓"
                    d_color = (COLORS["danger"] if d > 0 else COLORS["success"]) \
                              if danger_fn is None else danger_fn(d)
                    sub = (
                        f"<span style='color:{d_color};font-weight:700'>"
                        f"{arrow} {abs(d)} {unit}</span> vs prior 28d"
                    )
                else:
                    sub = "vs prior 28d"
                color = base_color or COLORS["primary"]
                col.markdown(
                    f"<div style='border:1px solid #E5E7EB;border-radius:8px;"
                    f"padding:14px 16px;border-top:3px solid {color}'>"
                    f"<div style='font-size:10px;font-weight:700;color:{COLORS['muted']};"
                    f"text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px'>"
                    f"{label}</div>"
                    f"<div style='font-size:26px;font-weight:800;color:{color};line-height:1.1'>"
                    f"{val_str}</div>"
                    f"<div style='font-size:11px;color:#6B7280;margin-top:5px'>{sub}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # ── KPI strip — 3 cards ───────────────────────────────────────────
            _k1, _k2, _k3 = st.columns(3, gap="large")

            # Card 1: Lab Queue — live 28d delta
            _kpi_delta_card(
                _k1, "Lab Queue", _lab_q_cur, _lab_q_prv, unit="min",
                base_color=COLORS["warning"] if (_lab_q_cur and _lab_q_cur > 20)
                           else COLORS["success"],
            )

            # Card 2: Median TAT — Jan–Aug 2025 baseline, no live delta
            _k2.markdown(
                f"<div style='border:1px solid #E5E7EB;border-radius:8px;"
                f"padding:14px 16px;border-top:3px solid {COLORS['primary']}'>"
                f"<div style='font-size:10px;font-weight:700;color:{COLORS['muted']};"
                f"text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px'>"
                f"Median TAT</div>"
                f"<div style='font-size:26px;font-weight:800;color:{COLORS['primary']};line-height:1.1'>"
                f"84 min</div>"
                f"<div style='font-size:11px;color:#6B7280;margin-top:5px'>"
                f"order → result · Jan–Aug 2025</div>"
                f"<div style='margin-top:8px'>"
                f"<span style='background:#FFF7ED;border:1px solid #FED7AA;color:#C2410C;"
                f"font-size:9px;font-weight:700;padding:2px 8px;border-radius:10px'>"
                f"Data gap from Sep 2025</span></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Card 3: Tests / Encounter — rolling 28d vs prior 28d
            _util_df  = P["lab_util"].copy() if len(P.get("lab_util", pd.DataFrame())) else pd.DataFrame()
            _util_l28 = _util_df[_util_df["PERIOD"] == "last_28"]  if len(_util_df) else pd.DataFrame()
            _util_p28 = _util_df[_util_df["PERIOD"] == "prior_28"] if len(_util_df) else pd.DataFrame()
            _util_cur = float(_util_l28.iloc[0]["TESTS_PER_ENCOUNTER"]) if len(_util_l28) >= 1 else None
            _util_prv = float(_util_p28.iloc[0]["TESTS_PER_ENCOUNTER"]) if len(_util_p28) >= 1 else None
            if _util_cur is not None and _util_prv is not None:
                _ud      = round(_util_cur - _util_prv, 2)
                _u_arrow = "↑" if _ud > 0 else "↓"
                _u_col   = COLORS["warning"] if _ud > 0 else COLORS["success"]
                _u_sub   = (f"<span style='color:{_u_col};font-weight:700'>"
                            f"{_u_arrow} {abs(_ud):.2f}</span> vs prior 28d")
            else:
                _u_sub = "—"
            _k3.markdown(
                f"<div style='border:1px solid #E5E7EB;border-radius:8px;"
                f"padding:14px 16px;border-top:3px solid {COLORS['primary']}'>"
                f"<div style='font-size:10px;font-weight:700;color:{COLORS['muted']};"
                f"text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px'>"
                f"Avg Tests / Encounter</div>"
                f"<div style='font-size:26px;font-weight:800;color:{COLORS['primary']};line-height:1.1'>"
                f"{f'{_util_cur:.1f}' if _util_cur else '—'}</div>"
                f"<div style='font-size:11px;color:#6B7280;margin-top:5px'>{_u_sub}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)

            # ── Section 1: Peak Day TAT Table ──────────────────────────────────
            # Static from Inv 108. Baseline = Tue–Thu median. Peak days = Mon + Fri.
            # Confirms: Mon recovers after peak; Fri does not recover all day.
            section_header("Lab TAT · Peak Days vs Baseline", margin_top=4)
            st.markdown(
                "<div style='font-size:11px;color:#6B7280;margin-bottom:10px'>"
                "Median TAT (order → result) · Jan–Aug 2025 · outpatient same-visit · "
                "peak hours = 09–12h (confirmed facility peak, Inv 29) · "
                "baseline = Tue–Thu · indicative signal (thin sample)</div>",
                unsafe_allow_html=True,
            )

            # (day_label, peak_tat, offpeak_tat, delta_peak, delta_offpeak, recovery, rec_color, row_style)
            # Baseline row: Tue–Thu avg — peak ~70 min, off-peak ~74 min
            # Monday: peak 83 (+13), off-peak 75 (+1) → recovers
            # Friday: peak 87 (+17), off-peak 123 (+49) → stuck
            _peak_rows = [
                ("Tue – Thu", 70,  74,  "—",   "—",   "Baseline", "#6B7280", True),
                ("Monday",    83,  75,  "+13", "+1",  "Recovers", "#059669", False),
                ("Friday",    87,  123, "+17", "+49", "Stuck",    "#DC2626", False),
            ]

            def _tat_color(v):
                if v is None:
                    return "#6B7280"
                if v <= 60:
                    return "#059669"
                if v <= 100:
                    return "#D97706"
                return "#DC2626"

            def _delta_color(d):
                if d == "—":
                    return "#6B7280"
                return "#DC2626" if d.startswith("+") and int(d[1:]) > 5 else "#059669"

            _peak_html = (
                "<table style='width:100%;border-collapse:collapse;font-size:12px'>"
                "<thead><tr style='border-bottom:2px solid #E5E7EB'>"
                "<th style='text-align:left;padding:8px 10px;color:#6B7280;font-weight:600;"
                "font-size:10px;text-transform:uppercase;letter-spacing:.05em'>Day</th>"
                "<th style='text-align:center;padding:8px 10px;color:#6B7280;font-weight:600;"
                "font-size:10px;text-transform:uppercase;letter-spacing:.05em'>Peak<br>"
                "<span style='font-weight:400;font-size:9px'>09–12h</span></th>"
                "<th style='text-align:center;padding:8px 10px;color:#6B7280;font-weight:600;"
                "font-size:10px;text-transform:uppercase;letter-spacing:.05em'>Off-Peak</th>"
                "<th style='text-align:center;padding:8px 10px;color:#6B7280;font-weight:600;"
                "font-size:10px;text-transform:uppercase;letter-spacing:.05em'>Δ Peak</th>"
                "<th style='text-align:center;padding:8px 10px;color:#6B7280;font-weight:600;"
                "font-size:10px;text-transform:uppercase;letter-spacing:.05em'>Δ Off-Peak</th>"
                "<th style='text-align:center;padding:8px 10px;color:#6B7280;font-weight:600;"
                "font-size:10px;text-transform:uppercase;letter-spacing:.05em'>Recovery</th>"
                "</tr></thead><tbody>"
            )
            for _i, (_day, _pk, _op, _dpk, _dop, _rec, _rc, _is_base) in enumerate(_peak_rows):
                _bg = "#F0F9FF" if _is_base else ("#FAFAFA" if _i % 2 == 0 else "#FFFFFF")
                _day_style = "font-style:italic;color:#6B7280" if _is_base else "font-weight:700;color:#111827"
                _peak_html += (
                    f"<tr style='background:{_bg};border-bottom:1px solid #F3F4F6'>"
                    f"<td style='padding:9px 10px;{_day_style}'>{_day}</td>"
                    f"<td style='padding:9px 10px;text-align:center;font-weight:600;"
                    f"color:{_tat_color(_pk)}'>{_pk} min</td>"
                    f"<td style='padding:9px 10px;text-align:center;font-weight:600;"
                    f"color:{_tat_color(_op)}'>{_op} min</td>"
                    f"<td style='padding:9px 10px;text-align:center;font-weight:600;"
                    f"color:{_delta_color(_dpk)}'>{_dpk}</td>"
                    f"<td style='padding:9px 10px;text-align:center;font-weight:600;"
                    f"color:{_delta_color(_dop)}'>{_dop}</td>"
                    f"<td style='padding:9px 10px;text-align:center'>"
                    f"<span style='background:{_rc}18;border:1px solid {_rc}40;color:{_rc};"
                    f"font-size:10px;font-weight:700;padding:2px 10px;border-radius:10px'>"
                    f"{_rec}</span></td>"
                    f"</tr>"
                )
            _peak_html += "</tbody></table>"
            st.markdown(_peak_html, unsafe_allow_html=True)

            st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)

            # ── Section 2: Weekly trend ────────────────────────────────────────
            section_header("Weekly Lab Operations", margin_top=4)
            st.markdown(
                "<div style='font-size:11px;font-weight:600;color:#6B7280;"
                "margin-bottom:4px'>Lab intake — weekly visits & queue wait</div>",
                unsafe_allow_html=True,
            )
            if len(_lab_wk) > 1:
                _wk = _lab_wk.iloc[:-1].copy()  # drop partial final week
                _fig_wk = go.Figure()
                _fig_wk.add_bar(
                    x=_wk["WEEK_START"], y=_wk["LAB_VISITS"],
                    name="Visits", marker_color=f"rgba(0,114,206,0.6)",
                    yaxis="y1",
                    hovertemplate="%{x|%b %d}: %{y} visits<extra></extra>",
                )
                _fig_wk.add_scatter(
                    x=_wk["WEEK_START"], y=_wk["MEDIAN_QUEUE_MIN"],
                    name="Queue (min)", mode="lines+markers",
                    line=dict(color=COLORS["warning"], width=2),
                    marker_size=5, yaxis="y2",
                    hovertemplate="%{x|%b %d}: %{y} min queue<extra></extra>",
                )
                _fig_wk.update_layout(**cl(
                    height=240,
                    yaxis=dict(title="Visits", showgrid=False),
                    yaxis2=dict(
                        title="Queue (min)", overlaying="y", side="right",
                        showgrid=False, rangemode="tozero",
                    ),
                    legend=dict(orientation="h", y=1.12, x=0),
                ))
                st.plotly_chart(_fig_wk, use_container_width=True,
                                config={"displayModeBar": False})

            st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)

            # ── Section 3: TAT by test (reference, Jan–Aug 2025) ──────────────
            section_header("Test Mix · TAT Reference", margin_top=4)
            st.markdown(
                "<div style='font-size:11px;color:#6B7280;margin-bottom:6px'>"
                "Jan–Aug 2025 baseline · outpatient same-visit only · "
                "no delta available (order system unlinked Oct 2025)</div>",
                unsafe_allow_html=True,
            )
            if len(_lab_tat_test):
                _tt = _lab_tat_test[["TEST_NAME","TEST_COUNT",
                                     "MEDIAN_TAT_MIN","PCT_WITHIN_2H"]].copy()
                _tt.columns = ["Test", "N", "Median TAT (min)", "Within 2h %"]

                def _style_tat_cell(val):
                    if isinstance(val, (int, float)):
                        if val > 180:
                            return f"color:{COLORS['danger']};font-weight:600"
                        if val > 120:
                            return f"color:{COLORS['warning']};font-weight:600"
                    return ""

                st.dataframe(
                    _tt.style.map(_style_tat_cell, subset=["Median TAT (min)"]),
                    use_container_width=True,
                    hide_index=True,
                    height=240,
                )

            # ── Section 4: Monthly result volume trend ────────────────────────
            _lab_vol_m = P.get("lab_vol_monthly", pd.DataFrame()).copy()
            if len(_lab_vol_m) > 1:
                section_header("Monthly Lab Volume · Result Trend", margin_top=4)
                st.markdown(
                    "<div style='font-size:11px;color:#6B7280;margin-bottom:6px'>"
                    "Total results entered per month · Jun 2024–Apr 2026 · "
                    "EVALUATION_INVESTIGATION_RESULTS · Oct 2025 dip = system migration</div>",
                    unsafe_allow_html=True,
                )
                _lab_vol_m["result_month"] = pd.to_datetime(_lab_vol_m["RESULT_MONTH"])
                _fig_vol = go.Figure()
                _fig_vol.add_scatter(
                    x=_lab_vol_m["result_month"],
                    y=_lab_vol_m["RESULT_COUNT"],
                    mode="lines+markers",
                    line=dict(color=COLORS["primary"], width=2),
                    marker=dict(size=5, color=COLORS["primary"]),
                    hovertemplate="%{x|%b %Y}: %{y:,} results<extra></extra>",
                )
                _oct25 = pd.Timestamp("2025-10-01")
                if _oct25 in _lab_vol_m["result_month"].values:
                    _oct25_y = int(_lab_vol_m.loc[
                        _lab_vol_m["result_month"] == _oct25, "RESULT_COUNT"
                    ].iloc[0])
                    _fig_vol.add_scatter(
                        x=[_oct25], y=[_oct25_y],
                        mode="markers",
                        marker=dict(size=9, color=COLORS["warning"], symbol="diamond"),
                        hovertemplate="Oct 2025 (migration dip): %{y:,}<extra></extra>",
                    )
                _fig_vol.update_layout(**cl(
                    height=220,
                    yaxis=dict(title="Results", showgrid=False),
                    xaxis=dict(showgrid=False),
                    showlegend=False,
                ))
                st.plotly_chart(_fig_vol, use_container_width=True,
                                config={"displayModeBar": False})

            # ── DQ note ───────────────────────────────────────────────────────
            st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
            dq_note(
                "Lab Queue & Weekly Trend: RECEPTION_TIME_TRACKERS · data through Apr 2026 · "
                "WAIT_TIME = stage queue wait (time before patient received at lab window). "
                "Median TAT (84 min): order → result · Jan–Aug 2025 outpatient same-visit only · "
                "inpatient 03:00 batch orders excluded · new order system unlinked Oct 2025 — "
                "queue wait used as operational proxy. "
                "Avg Tests / Encounter: stg_lab_events ÷ STG_EVAL_VISITS · rolling 28d. "
                "Peak day TAT: Jan–Aug 2025 baseline · indicative signal. "
                "Monthly volume: EVALUATION_INVESTIGATION_RESULTS · result entry date, not order date."
            )

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

        # ── Renal / Dialysis Operations (KSH only) ───────────────────────────
        if _is_ksh_p3:
            section_header("Renal / Dialysis Operations")

            _dial3   = fac_dialysis_ops.sort_values("INVOICE_MONTH").copy() if len(fac_dialysis_ops) else pd.DataFrame()
            _cd12_df = (P["cd12_rate"].copy().rename(columns=str.lower).sort_values("critical_month")
                        if len(P.get("cd12_rate", pd.DataFrame())) else pd.DataFrame())

            # ── KPI strip ─────────────────────────────────────────────────────
            # Utilisation MoM (complete months only)
            _dial_full  = _dial3[_dial3["IS_PARTIAL_MONTH"] == False] if len(_dial3) else pd.DataFrame()
            _util_cur   = float(_dial_full.iloc[-1]["UTILISATION_PCT_THEORETICAL"]) if len(_dial_full) >= 1 else None
            _util_prv   = float(_dial_full.iloc[-2]["UTILISATION_PCT_THEORETICAL"]) if len(_dial_full) >= 2 else None
            _util_mo    = (pd.to_datetime(_dial_full.iloc[-1]["INVOICE_MONTH"]).strftime("%b %Y")
                           if len(_dial_full) >= 1 else "")
            _util_delta = round(_util_cur - _util_prv, 1) if (_util_cur is not None and _util_prv is not None) else None
            _util_dir   = ("↑" if _util_delta > 0 else "↓") if _util_delta is not None else ""
            _util_sub   = (f"{_util_dir} {abs(_util_delta):.1f}pp MoM · {_util_mo}"
                           if _util_delta is not None else _util_mo)

            # Creatinine bottleneck — cumulative (monthly volumes too small for trend)
            _cd12_total_crit = int(_cd12_df["total_critical"].sum()) if len(_cd12_df) else 0
            _cd12_not_adm    = int(_cd12_df["not_admitted"].sum())    if len(_cd12_df) else 0
            _cd12_window     = (
                f"{pd.to_datetime(_cd12_df['critical_month'].iloc[0]).strftime('%b %Y')}–"
                f"{pd.to_datetime(_cd12_df['critical_month'].iloc[-1]).strftime('%b %Y')}"
                if len(_cd12_df) else ""
            )

            # Cash share (latest full month)
            _cash_latest = int(_dial_full.iloc[-1]["SESSIONS_CASH"]) if len(_dial_full) >= 1 else 0
            _sess_latest = int(_dial_full.iloc[-1]["SESSIONS_BILLED"]) if len(_dial_full) >= 1 else 0
            _cash_pct    = round(_cash_latest * 100 / max(_sess_latest, 1), 1)

            _rk1, _rk2, _rk3, _rk4 = st.columns(4, gap="large")
            with _rk1:
                kpi_card(
                    "Dialysis Utilisation",
                    f"{_util_cur:.1f}%" if _util_cur is not None else "—",
                    _util_sub,
                    COLORS["success"] if (_util_delta or 0) >= 0 else COLORS["warning"],
                )
            with _rk2:
                kpi_card(
                    "Critical → Not Admitted",
                    f"{_cd12_not_adm} of {_cd12_total_crit}",
                    f"cumulative · {_cd12_window} · monthly volumes too small to trend",
                    COLORS["coral"],
                )
            with _rk3:
                kpi_card(
                    "Enrolment Gap",
                    "97.6%",
                    "123 of 126 critical patients never enrolled · billing audit Apr 2026",
                    COLORS["danger"],
                )
            with _rk4:
                kpi_card(
                    "Cash Sessions",
                    f"{_cash_pct:.1f}%",
                    f"of {_sess_latest} sessions · {_util_mo} · cash access pathway unclear",
                    COLORS["muted"],
                )

            st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)

            # ── 1. Process performance — utilisation trend ─────────────────────
            if len(_dial3):
                _fig_util = go.Figure()
                _fig_util.add_scatter(
                    x=_dial3["INVOICE_MONTH"],
                    y=_dial3["UTILISATION_PCT_THEORETICAL"],
                    mode="lines+markers",
                    line=dict(color=COLORS["primary"], width=2),
                    marker=dict(
                        size=7,
                        color=[
                            COLORS["muted"] if r.IS_PARTIAL_MONTH else COLORS["primary"]
                            for r in _dial3.itertuples()
                        ],
                    ),
                    hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>",
                    showlegend=False,
                )
                _fig_util.add_hline(
                    y=100, line_dash="dot", line_color=COLORS["muted"], line_width=1,
                    annotation_text="one-shift cap (264 sessions)",
                    annotation_font_size=9, annotation_font_color=COLORS["muted"],
                    annotation_position="top right",
                )
                _fig_util.update_layout(**cl(
                    height=260,
                    yaxis=dict(title="Utilisation %", range=[0, 115]),
                    margin=dict(l=0, r=0, t=30, b=30),
                    title=dict(
                        text="Dialysis utilisation — % of one-shift theoretical max (264 sessions/month)",
                        font=dict(size=11), x=0,
                    ),
                ))
                st.plotly_chart(_fig_util, use_container_width=True,
                                config={"displayModeBar": False})

            st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)

            # ── 2–4. Bottleneck + Access Gap + Observability — three columns ───
            _rb_left, _rb_mid, _rb_right = st.columns(3, gap="large")

            with _rb_left:
                st.markdown(
                    "<div style='font-size:11px;font-weight:700;color:#003467;"
                    "text-transform:uppercase;letter-spacing:0.8px;margin-bottom:10px'>"
                    "2 · Bottleneck</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='background:#FFF1F2;border-left:3px solid {COLORS['coral']};"
                    f"border-radius:4px;padding:12px 14px'>"
                    f"<div style='font-size:22px;font-weight:700;color:{COLORS['coral']}'>"
                    f"{_cd12_not_adm} / {_cd12_total_crit}</div>"
                    f"<div style='font-size:12px;color:#003467;margin-top:4px'>"
                    f"critical creatinine patients not admitted</div>"
                    f"<div style='font-size:11px;color:#6B8CAE;margin-top:6px'>"
                    f"Cumulative · {_cd12_window}<br>"
                    f"Monthly volumes (1–8/month) too small to show a reliable trend.</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with _rb_mid:
                st.markdown(
                    "<div style='font-size:11px;font-weight:700;color:#003467;"
                    "text-transform:uppercase;letter-spacing:0.8px;margin-bottom:10px'>"
                    "3 · Access Gap</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='background:#F4F8FC;border-left:3px solid {COLORS['muted']};"
                    f"border-radius:4px;padding:12px 14px'>"
                    f"<div style='font-size:13px;font-weight:600;color:#003467'>NHIF dominates activity</div>"
                    f"<div style='font-size:11px;color:#6B8CAE;margin-top:6px'>"
                    f"Cash sessions: 0–2/month across 14 months.<br>"
                    f"Dialysis activity is overwhelmingly NHIF-funded.<br><br>"
                    f"No meaningful cash access pathway is visible in available data.</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with _rb_right:
                st.markdown(
                    "<div style='font-size:11px;font-weight:700;color:#003467;"
                    "text-transform:uppercase;letter-spacing:0.8px;margin-bottom:10px'>"
                    "4 · Process Observability</div>",
                    unsafe_allow_html=True,
                )
                _obs_rows = [
                    ("✓", "Critical creatinine result captured",  COLORS["success"]),
                    ("✓", "Admission outcome captured",           COLORS["success"]),
                    ("✓", "Dialysis session captured",            COLORS["success"]),
                    ("✗", "Referral event not captured",          COLORS["danger"]),
                    ("✗", "Scheduling event not captured",        COLORS["danger"]),
                    ("✗", "Referral → session TAT not captured",  COLORS["danger"]),
                ]
                _obs_html = "".join(
                    f"<div style='display:flex;gap:8px;align-items:flex-start;"
                    f"margin-bottom:5px'>"
                    f"<span style='font-size:11px;font-weight:700;color:{c};min-width:12px'>{mark}</span>"
                    f"<span style='font-size:11px;color:#003467'>{txt}</span></div>"
                    for mark, txt, c in _obs_rows
                )
                st.markdown(
                    f"<div style='background:#F4F8FC;border-left:3px solid {COLORS['muted']};"
                    f"border-radius:4px;padding:12px 14px'>"
                    f"{_obs_html}"
                    f"<div style='font-size:10px;color:#6B8CAE;margin-top:8px;border-top:1px solid #D6E4F0;padding-top:6px'>"
                    f"Cannot identify where the dialysis pathway loses patients between referral and enrolment.</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


        if facility == "KISUMU_CLEAN":
            fac_img = imaging[imaging["FACILITY"] == facility].copy() if "FACILITY" in imaging.columns else imaging.copy()

            MODALITY_ORDER = ["CT / Angio", "MRI", "ECHO / Cardiac", "Ultrasound", "X-Ray"]
            MODALITY_COLORS = {
                "CT / Angio":    COLORS["primary"],
                "MRI":           COLORS["purple"],
                "ECHO / Cardiac":COLORS["success"],
                "Ultrasound":    COLORS["warning"],
                "X-Ray":         COLORS["muted"],
            }

            if len(fac_img):
                _BENCH = {
                    "X-Ray":          {"low": 5.0,  "high": 10.0, "src": "WHO Essential Imaging 2020"},
                    "Ultrasound":     {"low": 2.0,  "high": 5.0,  "src": "WHO Essential Imaging 2020"},
                    "CT / Angio":     {"low": 0.5,  "high": 2.0,  "src": "RSNA low-resource guidance"},
                    "ECHO / Cardiac": {"low": 0.5,  "high": 1.0,  "src": "Indicative — no Kenya standard"},
                    "MRI":            {"low": 0.1,  "high": 0.5,  "src": "RSNA low-resource guidance"},
                }

                # Complete months only (< current month start)
                _img_mo_start = pd.Timestamp.today().to_period("M").to_timestamp()
                _img_complete  = sorted([
                    m for m in fac_img["REVENUE_MONTH"].unique()
                    if pd.Timestamp(m) < _img_mo_start
                ])
                _img_cur_mo  = _img_complete[-1] if len(_img_complete) >= 1 else None
                _img_prv_mo  = _img_complete[-2] if len(_img_complete) >= 2 else None
                _img_cur_lbl = pd.Timestamp(_img_cur_mo).strftime("%b %Y") if _img_cur_mo else "—"
                _img_prv_lbl = pd.Timestamp(_img_prv_mo).strftime("%b %Y") if _img_prv_mo else "—"

                # OPD visits for rate computation
                _vis_s = P["visit_sum"].copy()
                _vis_s.columns = _vis_s.columns.str.upper()
                _vis_s["MONTH"] = pd.to_datetime(_vis_s["VISIT_MONTH"])
                _vis_idx = _vis_s.set_index("MONTH")["TOTAL_VISITS"]

                # Per-modality: sessions + rate for cur + prv
                _img_u = fac_img[fac_img["MODALITY"].isin(_BENCH)].copy()
                _img_u["MONTH"] = pd.to_datetime(_img_u["REVENUE_MONTH"])
                _img_u["visits"] = _img_u["MONTH"].map(_vis_idx)
                _img_u["rate"]   = (
                    _img_u["SESSIONS"] / _img_u["visits"].clip(lower=1) * 100
                ).round(2)

                def _img_month_agg(mo):
                    if mo is None:
                        return pd.DataFrame()
                    sub = _img_u[_img_u["REVENUE_MONTH"] == mo]
                    return sub.groupby("MODALITY").agg(
                        sessions=("SESSIONS", "sum"),
                        rate=("rate", "mean"),
                    )

                _mc = _img_month_agg(_img_cur_mo)
                _mp = _img_month_agg(_img_prv_mo)
                _img_active = [m for m in MODALITY_ORDER if m in _mc.index]

                # ── Cards ─────────────────────────────────────────────────────
                section_header(f"Imaging — Modality Performance · {_img_cur_lbl}")
                _ic = st.columns(len(_img_active) if _img_active else 1, gap="small")
                _interp_signals = []

                for _ci, mod in zip(_ic, _img_active):
                    _cur_s    = int(_mc.loc[mod, "sessions"]) if mod in _mc.index else 0
                    _prv_s    = int(_mp.loc[mod, "sessions"]) if mod in _mp.index else None
                    _cur_r    = round(float(_mc.loc[mod, "rate"]), 2) if mod in _mc.index else None
                    _prv_r    = round(float(_mp.loc[mod, "rate"]), 2) if mod in _mp.index else None
                    _b        = _BENCH.get(mod, {})
                    _bench_lo = _b.get("low")
                    _bench_hi = _b.get("high")

                    # Volume direction
                    if _prv_s is not None:
                        _vol_d   = _cur_s - _prv_s
                        _vol_dir = "↑" if _vol_d > 0 else ("↓" if _vol_d < 0 else "→")
                        _vol_sub = f"{_vol_dir} {abs(_vol_d)} sessions vs {_img_prv_lbl}"
                    else:
                        _vol_d, _vol_sub = 0, "no prior month"

                    # Rate vs benchmark
                    if _cur_r is not None and _bench_lo is not None:
                        if _cur_r > _bench_hi:
                            _rate_txt   = f"{_cur_r:.2f}/100 OPD · above benchmark"
                            _rate_color = COLORS["warning"]
                        elif _cur_r < _bench_lo:
                            _rate_txt   = f"{_cur_r:.2f}/100 OPD · below benchmark"
                            _rate_color = COLORS["coral"]
                        else:
                            _rate_txt   = f"{_cur_r:.2f}/100 OPD · within benchmark"
                            _rate_color = COLORS["success"]
                    else:
                        _rate_txt, _rate_color = "rate unavailable", COLORS["muted"]

                    # Rate direction (for interpretation)
                    if _prv_r is not None and _cur_r is not None:
                        _rate_d = round(_cur_r - _prv_r, 2)
                        _interp_signals.append({
                            "mod": mod, "vol_d": _vol_d, "rate_d": _rate_d,
                            "cur_s": _cur_s, "cur_r": _cur_r,
                        })

                    _card_color = (COLORS["danger"] if _vol_d < 0
                                   else MODALITY_COLORS.get(mod, COLORS["primary"]))
                    with _ci:
                        st.markdown(
                            f"<div style='border-left:4px solid {_card_color};"
                            f"background:#F4F8FC;border-radius:4px;padding:10px 12px;"
                            f"margin-bottom:4px'>"
                            f"<div style='font-size:10px;font-weight:700;color:{_card_color};"
                            f"text-transform:uppercase;letter-spacing:0.8px'>{mod}</div>"
                            f"<div style='font-size:18px;font-weight:700;color:#003467;"
                            f"margin:4px 0'>{_cur_s:,}</div>"
                            f"<div style='font-size:10px;color:#6B8CAE'>sessions</div>"
                            f"<div style='font-size:11px;color:#003467;margin-top:5px'>{_vol_sub}</div>"
                            f"<div style='font-size:10px;color:{_rate_color};margin-top:3px'>"
                            f"{_rate_txt}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                # ── Trend chart — sessions / 100 OPD, all modalities ──────────
                st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
                _img_trend = _img_u[_img_u["REVENUE_MONTH"].isin(_img_complete)].copy()
                if len(_img_trend):
                    _fig_img = go.Figure()
                    for mod in _img_active:
                        _ms = (_img_trend[_img_trend["MODALITY"] == mod]
                               .sort_values("MONTH"))
                        if len(_ms):
                            _b = _BENCH.get(mod, {})
                            if _b:
                                _fig_img.add_hrect(
                                    y0=_b["low"], y1=_b["high"],
                                    fillcolor=f"rgba(107,140,174,0.07)",
                                    line_width=0,
                                )
                            _fig_img.add_scatter(
                                x=_ms["MONTH"], y=_ms["rate"],
                                mode="lines+markers",
                                name=mod,
                                line=dict(color=MODALITY_COLORS.get(mod, COLORS["muted"]),
                                          width=2),
                                marker=dict(size=5),
                                hovertemplate=(
                                    f"<b>{mod}</b> %{{x|%b %Y}}: "
                                    "%{y:.2f} per 100 OPD visits<extra></extra>"
                                ),
                            )
                    _fig_img.update_layout(**cl(
                        height=280,
                        yaxis=dict(title="Sessions per 100 OPD visits"),
                        margin=dict(l=0, r=0, t=32, b=30),
                        title=dict(
                            text="Imaging demand intensity — sessions per 100 outpatient visits",
                            font=dict(size=11), x=0,
                        ),
                        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
                                    font=dict(size=10)),
                    ))
                    st.plotly_chart(_fig_img, use_container_width=True,
                                    config={"displayModeBar": False})

                # ── Interpretation card ───────────────────────────────────────
                if _interp_signals:
                    _both_down  = [s for s in _interp_signals if s["vol_d"] < 0 and s["rate_d"] < -0.1]
                    _vol_only   = [s for s in _interp_signals if s["vol_d"] < 0 and abs(s["rate_d"]) <= 0.1]
                    _rate_up    = [s for s in _interp_signals if s["vol_d"] < 0 and s["rate_d"] > 0.1]

                    _interp_lines = []
                    if _both_down:
                        mods = ", ".join(s["mod"] for s in _both_down)
                        _interp_lines.append(
                            f"<b>{mods}</b>: sessions fell and demand intensity also fell — "
                            "investigate referral/order pathway."
                        )
                    if _vol_only:
                        mods = ", ".join(s["mod"] for s in _vol_only)
                        _interp_lines.append(
                            f"<b>{mods}</b>: volume change broadly consistent with OPD volume — "
                            "no imaging-specific ordering signal established."
                        )
                    if _rate_up:
                        mods = ", ".join(s["mod"] for s in _rate_up)
                        _interp_lines.append(
                            f"<b>{mods}</b>: imaging demand intensity rose despite lower volume — "
                            "investigate ordering pattern."
                        )
                    if not _interp_lines:
                        _interp_lines.append(
                            "All modalities show volume changes consistent with OPD volume. "
                            "No imaging-specific ordering signal established this month."
                        )
                    st.markdown(
                        f"<div style='background:#F4F8FC;border:1px solid #D6E4F0;"
                        f"border-radius:6px;padding:12px 16px;margin-top:10px'>"
                        f"<div style='font-size:10px;font-weight:700;color:#6B8CAE;"
                        f"text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>"
                        f"Interpretation · {_img_cur_lbl} vs {_img_prv_lbl}</div>"
                        f"{''.join(f'<div style=font-size:11px;color:#003467;margin-bottom:4px>{l}</div>' for l in _interp_lines)}"
                        f"<div style='font-size:10px;color:#6B8CAE;margin-top:8px'>"
                        f"Benchmarks: shaded band on chart. Interpretation identifies investigation "
                        f"targets — does not establish cause.</div>"
                        f"</div>",
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

    # ── Tab 5: Patient Flow ───────────────────────────────────────────────────

    with tab5:
        if not _is_ksh_p3:
            st.info("Patient Flow analytics are KSH-specific — not applicable for TENRI.")
        else:
            st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

            # ── A: Activity Overview ──────────────────────────────────────────
            section_header("A · Activity Overview")

            _opd_mo  = P.get("opd_monthly", pd.DataFrame()).copy()
            _opd_28a = P.get("opd_28d", pd.DataFrame()).copy()

            if len(_opd_28a):
                _opd_28a.columns = [c.upper() for c in _opd_28a.columns]
                _a28_last  = _opd_28a[_opd_28a["PERIOD"] == "last28"].copy()
                _a28_prior = _opd_28a[_opd_28a["PERIOD"] == "prior28"].copy()

                if len(_a28_last) and len(_a28_prior):
                    # visits
                    _av_now   = int(_a28_last["DAILY_VISITS"].sum())
                    _av_prior = int(_a28_prior["DAILY_VISITS"].sum())
                    _av_delta = _av_now - _av_prior

                    # doctor reach — weighted by daily visits
                    _av_dv  = _a28_last["DAILY_VISITS"]
                    _av_dr  = (_a28_last["PCT_HAD_DOCTOR"] / 100 * _av_dv).sum() / _av_dv.sum() * 100
                    _pv_dv  = _a28_prior["DAILY_VISITS"]
                    _pv_dr  = (_a28_prior["PCT_HAD_DOCTOR"] / 100 * _pv_dv).sum() / _pv_dv.sum() * 100
                    _dr_delta = _av_dr - _pv_dr

                    # median TAT — median of daily medians
                    _at_vals  = _a28_last["DAILY_P50_TAT"].dropna()
                    _pt_vals  = _a28_prior["DAILY_P50_TAT"].dropna()
                    _av_tat   = int(_at_vals.median()) if len(_at_vals) else None
                    _pv_tat   = int(_pt_vals.median()) if len(_pt_vals) else None
                    _tat_delta = (_av_tat - _pv_tat) if (_av_tat is not None and _pv_tat is not None) else None

                    _ac1, _ac2, _ac3 = st.columns(3, gap="large")
                    with _ac1:
                        _v_arrow = "↑" if _av_delta > 0 else ("↓" if _av_delta < 0 else "→")
                        kpi_card(
                            "OPD Visits · Last 28 Days",
                            f"{_av_now:,}",
                            f"{_v_arrow} {abs(_av_delta):,} vs prior 28 days",
                            COLORS["primary"],
                        )
                    with _ac2:
                        _dr_color = COLORS["success"] if _av_dr >= 50 else COLORS["warning"]
                        _dr_arrow = "↑" if _dr_delta > 0 else ("↓" if _dr_delta < 0 else "→")
                        kpi_card(
                            "Doctor Reach · Last 28 Days",
                            f"{_av_dr:.1f}%",
                            f"{_dr_arrow} {abs(_dr_delta):.1f}pp vs prior 28 days",
                            _dr_color,
                        )
                    with _ac3:
                        if _av_tat is not None:
                            _tat_color = COLORS["success"] if _av_tat <= 14 else COLORS["warning"] if _av_tat <= 19 else COLORS["danger"]
                            _t_arrow = ("↑" if _tat_delta > 0 else ("↓" if _tat_delta < 0 else "→")) if _tat_delta is not None else ""
                            _t_sub = f"{_t_arrow} {abs(_tat_delta)} min vs prior 28 · threshold 19 min" if _tat_delta is not None else "threshold 19 min"
                            kpi_card("Median TAT · Last 28 Days", f"{_av_tat} min", _t_sub, _tat_color)
                        else:
                            kpi_card("Median TAT · Last 28 Days", "—", "Data unavailable", COLORS["muted"])

            st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

            if len(_opd_mo):
                _opd_mo.columns = [c.upper() for c in _opd_mo.columns]
                _opd_mo["VISIT_MONTH"] = pd.to_datetime(_opd_mo["VISIT_MONTH"])
                _fig_amo = go.Figure(go.Bar(
                    x=_opd_mo["VISIT_MONTH"],
                    y=_opd_mo["VISITS"],
                    marker_color=COLORS["primary"],
                    hovertemplate="%{x|%b %Y}: %{y:,} visits<extra></extra>",
                ))
                _fig_amo.update_layout(**cl(
                    height=220,
                    xaxis=dict(tickformat="%b %Y"),
                    yaxis_title="Visits",
                    margin=dict(l=0, r=10, t=10, b=10),
                    showlegend=False,
                ))
                st.plotly_chart(_fig_amo, use_container_width=True, config={"displayModeBar": False})
                dq_note(
                    "Source: rpt_opd_visit_spine · one row per OPD visit anchored at Reception. "
                    "Complete months only. CDC tracker pipeline stopped Apr 21, 2026."
                )

            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)

            # ── B: Patient Pathway ────────────────────────────────────────────
            section_header("B · Patient Pathway")

            _snk_raw = P.get("journey_sankey", pd.DataFrame()).copy()
            if len(_snk_raw):
                _snk_raw.columns = [c.upper() for c in _snk_raw.columns]

                _stage_order = [
                    "Reception", "Doctor", "Laboratory", "Radiology",
                    "Pharmacy", "Admitted", "Exit",
                ]
                _all_stages = list(pd.unique(_snk_raw[["FROM_STAGE", "TO_STAGE"]].values.ravel()))
                _all_stages = sorted(
                    _all_stages,
                    key=lambda n: _stage_order.index(n) if n in _stage_order else 99,
                )
                _node_idx = {n: i for i, n in enumerate(_all_stages)}

                _snk_source = [_node_idx[r["FROM_STAGE"]] for _, r in _snk_raw.iterrows()]
                _snk_target = [_node_idx[r["TO_STAGE"]]   for _, r in _snk_raw.iterrows()]
                _snk_value  = [int(r["VISITS"])            for _, r in _snk_raw.iterrows()]

                _node_color_map = {
                    "Reception":  "#0072CE",
                    "Doctor":     "#0BB99F",
                    "Laboratory": "#7F77DD",
                    "Radiology":  "#D97706",
                    "Pharmacy":   "#D85A30",
                    "Admitted":   "#1D9E75",
                    "Exit":       "#9BAEC8",
                }
                _link_color_map = {
                    "Reception":  "rgba(0,114,206,0.12)",
                    "Doctor":     "rgba(11,185,159,0.12)",
                    "Laboratory": "rgba(127,119,221,0.12)",
                    "Radiology":  "rgba(217,119,6,0.12)",
                    "Pharmacy":   "rgba(216,90,48,0.12)",
                    "Admitted":   "rgba(29,158,117,0.12)",
                }
                _snk_node_colors = [_node_color_map.get(n, "#9BAEC8") for n in _all_stages]
                _snk_link_colors = [
                    _link_color_map.get(r["FROM_STAGE"], "rgba(155,174,200,0.12)")
                    for _, r in _snk_raw.iterrows()
                ]

                _node_x_map = {
                    "Reception":  0.01,
                    "Doctor":     0.22,
                    "Laboratory": 0.45,
                    "Radiology":  0.45,
                    "Pharmacy":   0.68,
                    "Admitted":   0.68,
                    "Exit":       0.99,
                }
                _node_y_map = {
                    "Reception":  0.45,
                    "Doctor":     0.30,
                    "Laboratory": 0.10,
                    "Radiology":  0.28,
                    "Pharmacy":   0.48,
                    "Admitted":   0.72,
                    "Exit":       0.30,
                }
                _snk_x = [_node_x_map.get(n, 0.5) for n in _all_stages]
                _snk_y = [_node_y_map.get(n, 0.5) for n in _all_stages]

                _fig_snk = go.Figure(go.Sankey(
                    arrangement="freeform",
                    node=dict(
                        pad=20, thickness=18,
                        label=_all_stages,
                        color=_snk_node_colors,
                        x=_snk_x,
                        y=_snk_y,
                        hovertemplate="%{label}: %{value:,} patient-visits<extra></extra>",
                    ),
                    link=dict(
                        source=_snk_source,
                        target=_snk_target,
                        value=_snk_value,
                        color=_snk_link_colors,
                        hovertemplate="%{source.label} → %{target.label}: %{value:,}<extra></extra>",
                    ),
                ))
                _fig_snk.update_layout(**cl(
                    height=780,
                    margin=dict(l=10, r=10, t=10, b=10),
                    font=dict(size=11, color="#003467"),
                ))
                st.plotly_chart(_fig_snk, use_container_width=True, config={
                    "displayModeBar": "hover",
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                    "toImageButtonOptions": {"filename": "opd_patient_flow"},
                })
                dq_note(
                    "All available tracked visits · OPD journey only. Each link = patients who moved "
                    "from one stage to the next in timestamp order. Post-admission tracker activity "
                    "excluded — Admitted is the terminal OPD transition. "
                    "Source: rpt_opd_flow gold (Inv 117–126)."
                )
            else:
                st.caption("Sankey data unavailable.")

            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)

            # ── C: When Does TAT Rise? ────────────────────────────────────────
            section_header("C · When Does TAT Rise?")

            # Peak-band summary table
            _opd_pb = P.get("opd_peak_band", pd.DataFrame()).copy()
            if len(_opd_pb):
                _opd_pb.columns = [c.upper() for c in _opd_pb.columns]
                _opd_pb = _opd_pb.sort_values("BAND_SORT")
                _pb_rows = ""
                for _, _r in _opd_pb.iterrows():
                    _p50_v = int(_r["P50_TAT"]) if pd.notna(_r["P50_TAT"]) else "—"
                    _pb_rows += (
                        f'<tr style="border-bottom:1px solid #EEF2F8">'
                        f'<td style="padding:6px 12px;font-weight:600">{_r["BAND_LABEL"]}</td>'
                        f'<td style="padding:6px 12px;text-align:right">{int(_r["VISITS"]):,}</td>'
                        f'<td style="padding:6px 12px;text-align:right">{_p50_v}</td>'
                        f'</tr>'
                    )
                _pb_html = (
                    '<table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:20px">'
                    '<thead><tr style="background:#003467;color:#fff">'
                    '<th style="padding:7px 12px;text-align:left">Arrival band</th>'
                    '<th style="padding:7px 12px;text-align:right">Visits</th>'
                    '<th style="padding:7px 12px;text-align:right">Median TAT (min)</th>'
                    '</tr></thead>'
                    f'<tbody>{_pb_rows}</tbody>'
                    '</table>'
                    '<p style="font-size:10px;color:#6B8CAE;margin-top:-12px;margin-bottom:16px">'
                    'TAT = Reception → Doctor · 07:00–18:00 arrivals · all time</p>'
                )
                st.markdown(_pb_html, unsafe_allow_html=True)

            _opd_hr = P.get("opd_hourly", pd.DataFrame()).copy()

            if len(_opd_hr):
                _opd_hr.columns = [c.upper() for c in _opd_hr.columns]
                _cc1, _cc2 = st.columns(2, gap="large")

                with _cc1:
                    _fig_hrv = go.Figure(go.Bar(
                        x=_opd_hr["ARRIVAL_HOUR"],
                        y=_opd_hr["VISITS"],
                        marker_color=COLORS["primary"],
                        hovertemplate="Hour %{x}:00 — %{y:,} visits<extra></extra>",
                    ))
                    _fig_hrv.update_layout(**cl(
                        height=300,
                        yaxis_title="Visits",
                        title=dict(text="Arrivals by Hour · 07:00–21:00", font=dict(size=12)),
                        xaxis=dict(
                            tickvals=list(_opd_hr["ARRIVAL_HOUR"]),
                            ticktext=[f"{h}:00" for h in _opd_hr["ARRIVAL_HOUR"]],
                            tickangle=-45,
                        ),
                        margin=dict(l=0, r=10, t=40, b=50),
                        showlegend=False,
                    ))
                    st.plotly_chart(_fig_hrv, use_container_width=True, config={"displayModeBar": False})

                with _cc2:
                    _hr_tat = _opd_hr[_opd_hr["P50_TAT"].notna()].copy()
                    if len(_hr_tat):
                        _hr_colors = [
                            COLORS["success"] if v <= 14
                            else COLORS["warning"] if v <= 19
                            else COLORS["danger"]
                            for v in _hr_tat["P50_TAT"]
                        ]
                        _fig_hrtat = go.Figure(go.Bar(
                            x=_hr_tat["ARRIVAL_HOUR"],
                            y=_hr_tat["P50_TAT"],
                            marker_color=_hr_colors,
                            hovertemplate="Hour %{x}:00 — Median wait %{y:.0f} min<extra></extra>",
                        ))
                        _fig_hrtat.add_hline(
                            y=19, line_dash="dash", line_color="#D97706",
                            annotation_text="Pressure (19 min)",
                            annotation_position="top right",
                        )
                        _fig_hrtat.update_layout(**cl(
                            height=300,
                            yaxis_title="Median wait (min)",
                            title=dict(text="Median Reception → Doctor Wait by Hour", font=dict(size=12)),
                            xaxis=dict(
                                tickvals=list(_hr_tat["ARRIVAL_HOUR"]),
                                ticktext=[f"{h}:00" for h in _hr_tat["ARRIVAL_HOUR"]],
                                tickangle=-45,
                            ),
                            margin=dict(l=0, r=10, t=40, b=50),
                            showlegend=False,
                        ))
                        st.plotly_chart(_fig_hrtat, use_container_width=True, config={"displayModeBar": False})

                dq_note(
                    "TAT = Reception → first Doctor event · 07:00–21:00 arrivals. "
                    "Hours with < 5 timed visits omitted from TAT chart. "
                    "Pressure threshold 19 min = P75 of 598 daily medians (Inv 131)."
                )

            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)

            # ── D: Is the Bottleneck Real? ────────────────────────────────────
            section_header("D · Is the Bottleneck Real? — Last 28 Days")

            _opd_28d_d = P.get("opd_28d", pd.DataFrame()).copy()

            if len(_opd_28d_d):
                _opd_28d_d.columns = [c.upper() for c in _opd_28d_d.columns]
                _28d_last = _opd_28d_d[_opd_28d_d["PERIOD"] == "last28"].copy()

                if len(_28d_last):
                    _28d_last["VISIT_DATE"] = pd.to_datetime(_28d_last["VISIT_DATE"])
                    _28d_last = _28d_last.sort_values("VISIT_DATE")
                    _28d_tat = _28d_last[_28d_last["DAILY_P50_TAT"].notna()].copy()

                    _dot_colors = [
                        COLORS["danger"] if row else COLORS["primary"]
                        for row in _28d_tat["IS_PRESSURE"]
                    ]
                    _pressure_n = int(_28d_last["IS_PRESSURE"].sum())

                    _fig_d = go.Figure()
                    _fig_d.add_hline(
                        y=19, line_dash="dash", line_color="#D97706",
                        annotation_text="Pressure threshold · 19 min",
                        annotation_position="top left",
                    )
                    _fig_d.add_trace(go.Scatter(
                        x=_28d_tat["VISIT_DATE"],
                        y=_28d_tat["DAILY_P50_TAT"],
                        mode="markers",
                        marker=dict(color=_dot_colors, size=8),
                        hovertemplate="%{x|%d %b %Y}<br>Median TAT: %{y:.0f} min<extra></extra>",
                        showlegend=False,
                    ))
                    _fig_d.update_layout(**cl(
                        height=280,
                        yaxis=dict(title="Daily median TAT (min)", rangemode="tozero"),
                        xaxis=dict(title=""),
                        margin=dict(l=0, r=10, t=30, b=10),
                    ))
                    st.plotly_chart(_fig_d, use_container_width=True, config={"displayModeBar": False})
                    _c_danger  = COLORS["danger"]
                    _c_primary = COLORS["primary"]
                    _normal_n  = 28 - _pressure_n
                    st.markdown(
                        f"<p style='font-size:12px;color:#6B7280;margin-top:-8px'>"
                        f"<span style='color:{_c_danger};font-weight:600'>●</span> {_pressure_n} of 28 days above 19 min &nbsp;·&nbsp; "
                        f"<span style='color:{_c_primary};font-weight:600'>●</span> {_normal_n} normal days"
                        f"</p>",
                        unsafe_allow_html=True,
                    )

                dq_note(
                    "Last 28 days anchored at MAX(visit_date) · 07:00–22:00 arrivals · min 5 timed visits per day. "
                    "Threshold = P75 of all daily medians (Inv 131)."
                )

            # ── E: Does It Cascade? ───────────────────────────────────────────
            section_header("E · Does It Cascade? — Pressure vs Normal Days")

            _opd_spl = P.get("opd_spillover", pd.DataFrame()).copy()

            if len(_opd_spl):
                _opd_spl.columns = [c.upper() for c in _opd_spl.columns]
                _spl_p = _opd_spl[_opd_spl["DAY_TYPE"] == "Pressure"]
                _spl_n = _opd_spl[_opd_spl["DAY_TYPE"] == "Normal"]

                if len(_spl_p) and len(_spl_n):
                    _sp = _spl_p.iloc[0]
                    _sn = _spl_n.iloc[0]

                    def _delta_cell(p_val, n_val, unit="pp", higher_is_bad=False):
                        d = p_val - n_val
                        if abs(d) < 0.1:
                            return '<span style="color:#6B7280">—</span>'
                        arrow = "↑" if d > 0 else "↓"
                        bad = (d > 0) == higher_is_bad
                        color = "#DC2626" if bad and abs(d) > 3 else "#D97706" if bad else "#16A34A"
                        return f'<span style="color:{color};font-weight:600">{arrow} {abs(d):.1f}{unit}</span>'

                    _e_tat_p = int(_sp["P50_TAT"]) if pd.notna(_sp["P50_TAT"]) else None
                    _e_tat_n = int(_sn["P50_TAT"]) if pd.notna(_sn["P50_TAT"]) else None

                    _e_rows = [
                        ("Lab reach",      float(_sp["PCT_HAD_LAB"]),       float(_sn["PCT_HAD_LAB"]),       "%",   False),
                        ("Pharmacy reach",  float(_sp["PCT_HAD_PHARMACY"]),  float(_sn["PCT_HAD_PHARMACY"]),  "%",   False),
                        ("Radiology reach", float(_sp["PCT_HAD_RADIOLOGY"]), float(_sn["PCT_HAD_RADIOLOGY"]), "%",   False),
                    ]

                    # Median row first, then station reach rows
                    _e_html_rows = ""
                    if _e_tat_p is not None and _e_tat_n is not None:
                        _e_html_rows += (
                            f'<tr style="border-bottom:1px solid #EEF2F8">'
                            f'<td style="padding:6px 12px">Median (Reception → Doctor)</td>'
                            f'<td style="padding:6px 12px;text-align:right">{_e_tat_p} min</td>'
                            f'<td style="padding:6px 12px;text-align:right">{_e_tat_n} min</td>'
                            f'<td style="padding:6px 12px;text-align:right">{_delta_cell(_e_tat_p, _e_tat_n, " min", True)}</td>'
                            f'</tr>'
                        )
                    for _lbl, _pv, _nv, _unit, _hib in _e_rows:
                        _e_html_rows += (
                            f'<tr style="border-bottom:1px solid #EEF2F8">'
                            f'<td style="padding:6px 12px">{_lbl}</td>'
                            f'<td style="padding:6px 12px;text-align:right">{_pv:.1f}%</td>'
                            f'<td style="padding:6px 12px;text-align:right">{_nv:.1f}%</td>'
                            f'<td style="padding:6px 12px;text-align:right">{_delta_cell(_pv, _nv, "pp", _hib)}</td>'
                            f'</tr>'
                        )

                    _e_html = (
                        '<table style="width:100%;border-collapse:collapse;font-size:13px">'
                        '<thead><tr style="background:#003467;color:#fff">'
                        '<th style="padding:7px 12px;text-align:left">Metric</th>'
                        f'<th style="padding:7px 12px;text-align:right">Pressure ({int(_sp["TOTAL_DAYS"])} days)</th>'
                        f'<th style="padding:7px 12px;text-align:right">Normal ({int(_sn["TOTAL_DAYS"])} days)</th>'
                        '<th style="padding:7px 12px;text-align:right">Δ on pressure days</th>'
                        '</tr></thead>'
                        f'<tbody>{_e_html_rows}</tbody>'
                        '</table>'
                        '<p style="font-size:10px;color:#6B8CAE;margin-top:5px">'
                        'Station reach = % of visits touching each station · 07:00–22:00 arrivals · Pressure threshold: Inv 131.</p>'
                    )
                    st.markdown(_e_html, unsafe_allow_html=True)

            # Pressure vs normal day comparison — cascade chart
            _opd_sp = P.get("opd_spillover", pd.DataFrame()).copy()
            if len(_opd_sp):
                _opd_sp.columns = [c.lower() for c in _opd_sp.columns]
                _sp_norm = _opd_sp[_opd_sp["day_type"] == "Normal"]
                _sp_pres = _opd_sp[_opd_sp["day_type"] == "Pressure"]
                if len(_sp_norm) and len(_sp_pres):
                    _n = _sp_norm.iloc[0]
                    _p = _sp_pres.iloc[0]

                    _metrics   = ["Reception → Doctor (min)", "Lab reach (%)", "Pharmacy reach (%)", "Radiology reach (%)"]
                    _norm_vals = [float(_n["p50_tat"]), float(_n["pct_had_lab"]),
                                  float(_n["pct_had_pharmacy"]), float(_n["pct_had_radiology"])]
                    _pres_vals = [float(_p["p50_tat"]), float(_p["pct_had_lab"]),
                                  float(_p["pct_had_pharmacy"]), float(_p["pct_had_radiology"])]

                    _fig_sp = go.Figure()
                    _fig_sp.add_trace(go.Bar(
                        name=f'Normal ({int(_n["total_days"])} days)',
                        y=_metrics, x=_norm_vals,
                        orientation="h",
                        marker_color=COLORS["primary"],
                        text=[f"{v:.0f}" for v in _norm_vals],
                        textposition="outside",
                        hovertemplate="%{y}: %{x:.1f}<extra>Normal days</extra>",
                    ))
                    _fig_sp.add_trace(go.Bar(
                        name=f'Pressure ({int(_p["total_days"])} days)',
                        y=_metrics, x=_pres_vals,
                        orientation="h",
                        marker_color=COLORS["danger"],
                        text=[f"{v:.0f}" for v in _pres_vals],
                        textposition="outside",
                        hovertemplate="%{y}: %{x:.1f}<extra>Pressure days</extra>",
                    ))
                    _fig_sp.update_layout(**cl(
                        barmode="group",
                        height=280,
                        margin=dict(l=0, r=60, t=40, b=10),
                        title=dict(
                            text="Normal vs Pressure Days — How the System Changes",
                            font=dict(size=12),
                        ),
                        xaxis=dict(title="", showgrid=True, gridcolor="#EEF2F8"),
                        yaxis=dict(title="", autorange="reversed"),
                        legend=dict(orientation="h", y=1.12, x=0),
                    ))
                    st.plotly_chart(_fig_sp, use_container_width=True, config={"displayModeBar": False})
                    dq_note(
                        f"Reception → Doctor wait used to classify pressure days (threshold 19 min) · "
                        f"{int(_p['total_days'])} pressure days / {int(_n['total_days'])} normal days · "
                        "Reach = % of visits touching each station · 07:00–22:00 arrivals"
                    )

            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)

            # ── F: Why? — Flagged-Day Heatmap ────────────────────────────────
            section_header("F · Why? — When Do Pressure Days Concentrate?")

            _opd_hm = P.get("opd_heatmap", pd.DataFrame()).copy()

            if len(_opd_hm):
                _opd_hm.columns = [c.upper() for c in _opd_hm.columns]
                _hm_hours = sorted(_opd_hm["ARRIVAL_HOUR"].unique())
                # derive day order from DOW integer — avoids full-name vs abbreviation mismatch
                _hm_days = (
                    _opd_hm[["DOW", "DAY_NAME"]].drop_duplicates()
                    .sort_values("DOW")["DAY_NAME"]
                    .tolist()
                )

                # pivot_table (not pivot) — handles any duplicate index silently via sum/mean
                _hm_pivot = (
                    _opd_hm.pivot_table(
                        index="DAY_NAME", columns="ARRIVAL_HOUR",
                        values="VISITS", aggfunc="sum",
                    ).reindex(_hm_days)
                )
                _hm_tat_pivot = (
                    _opd_hm.pivot_table(
                        index="DAY_NAME", columns="ARRIVAL_HOUR",
                        values="P50_TAT", aggfunc="mean",
                    ).reindex(_hm_days)
                )

                # customdata carries TAT for hover; NaN → shown as blank cell
                _fig_hm = go.Figure(go.Heatmap(
                    z=_hm_pivot.values.tolist(),
                    x=[f"{h}:00" for h in _hm_hours],
                    y=_hm_days,
                    customdata=_hm_tat_pivot.values.tolist(),
                    hovertemplate=(
                        "<b>%{y} %{x}</b><br>"
                        "Visits: %{z:,.0f}<br>"
                        "P50 TAT: %{customdata:.0f} min"
                        "<extra></extra>"
                    ),
                    colorscale="Blues",
                    showscale=True,
                    colorbar=dict(title="Visits", thickness=14),
                ))
                _fig_hm.update_layout(**cl(
                    height=310,
                    xaxis=dict(title="Arrival Hour", tickangle=-45),
                    yaxis=dict(title="", autorange="reversed"),
                    margin=dict(l=0, r=10, t=10, b=50),
                ))
                st.plotly_chart(_fig_hm, use_container_width=True, config={"displayModeBar": False})
                st.markdown(
                    "<p style='font-size:12px;color:#6B7280;margin-top:-4px'>"
                    "<b>Monday 10:00–13:00</b> is the most consistently pressured block — appears on 27 of 139 pressure days, "
                    "median TAT 25–33 min. "
                    "<b>Wednesday 11:00</b> is the sharpest single-cell spike (37 min). "
                    "<b>Saturday 12:00</b> is the most severe when it occurs (41 min). "
                    "Across all days, TAT stays below the 19-min threshold at 09:00 and crosses it at 10:00."
                    "</p>",
                    unsafe_allow_html=True,
                )
                dq_note(
                    "Pressure days only (daily median TAT > 19 min · Inv 131) · 07:00–21:00 arrivals · "
                    "22 of 25 highest-volume cells have elevated TAT (Inv 132). "
                    "Source: rpt_opd_visit_spine."
                )

            st.markdown("<div style='margin-top:32px'></div>", unsafe_allow_html=True)


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
                '<div style="font-size:12px;color:#991B1B">▼ <b>Conversion −28%</b> '
                '— fewer patients admitted per evaluation during peak</div>'
                '<div style="font-size:12px;color:#991B1B">▲ <b>TAT +41% median</b> '
                '— 54 → 76 min</div>'
                '<div style="font-size:12px;color:#991B1B">▼ <b>Private capture −36%</b> '
                '— higher-yield admissions deflected at peak</div>'
                '<div style="font-size:12px;color:#991B1B">✕ <b>46% of peak non-admissions are permanent</b> '
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
                    "- Conversion drops 28% during peak (5.0% → 3.6%) — n=34,294 evaluations "
                    "(2,211 peak, 32,083 off-peak). Not a low-volume artefact.\n"
                    "- TAT rises 41% at median (54 → 76 min). P75 rises from 182 → 196 min — "
                    "peak pushes the median into the slow zone; the upper tail was already high off-peak.\n"
                    "- Doctor load redistribution is the structural driver: lowino absorbs 49.4% of peak "
                    "evaluations (vs 17.4% off-peak) while eawando's share falls from 37.3% to 27.8%. "
                    "Peak window and facility-wide concentration risk (CD6) have different key actors.\n"
                    "- Private ward capture falls 36% (11.9% → 7.6%). Observed shift — case-mix "
                    "contribution not isolated.\n"
                    "- 46.2% of non-admitted peak patients never returned to KSH. Of those who returned, "
                    "15.2% were eventually admitted — peak non-admission is largely permanent patient "
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
                    f"J.Ogutu (24.6% of facility admissions, Apr 2026) is the closest named backup. "
                    f"Review: Medical Director."
                )
                with st.expander("Analysis"):
                    st.markdown(
                        "- E.Awando evaluates **20–46% of admissions across all wards** (20% Private Maternity, up to 45.5% General Male) — the risk is facility-wide, not concentrated in one area.\n"
                        "- A single absence triggers intake reduction across all wards with no pre-defined cover.\n"
                        "- M.Akinyi's departure (Jan 2026) added ~57% volume onto E.Awando silently — no flag fired until months later.\n"
                        "- J.Ogutu is the closest confirmed backup at 24.6% of facility admissions (Apr 2026).\n"
                        "- Private wards are most exposed: fewest distinct evaluators and no fallback when E.Awando is absent."
                    )

                # ── Simulated absence impact ──────────────────────────────────
                st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
                section_header(f"{_fmt_doc(_dom_doc)}'s Contribution to Conversion Rate")
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
                                 "All-time aggregate · all doctors · all months",
                                 COLORS["primary"])
                    with _s2:
                        kpi_card("Facility Rate Without His Admissions",
                                 f"{_sim_rate}%",
                                 f"−{_drop}pp · {_fmt_doc(_dom_doc)}'s admissions removed · all-time basis",
                                 COLORS["danger"])
                    st.caption(
                        f"Structural dependency test: {_fmt_doc(_dom_doc)}'s {_sim_dom_adm:,} admissions "
                        "removed from all-time total (facility-wide, all months). Evaluation volume held "
                        "constant — gap shows what the facility rate would have been without his admitted "
                        "patients. Modelled estimate, not a measured outcome."
                    )
            else:
                st.caption("E.Awando ward share data not found — username may differ.")
        else:
            st.caption("Doctor ward data not available.")

        st.markdown("<div style='margin-top:32px'></div>", unsafe_allow_html=True)

        # ── CD12: Renal Patient Safety ────────────────────────────────────────

        section_header("Renal Pathway — Critical Patients Leaving Without Admission")

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
            kpi_card("Renal Routing Gap", "97.6%",
                     "123 of 126 critical creatinine patients never enrolled in dialysis", COLORS["danger"])

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
            "The dialysis programme is operational (54 patients enrolled, 80–135 sessions/month, NHIF-funded), "
            "but 97.6% of critical creatinine patients have never enrolled. Only 3 of 126 ever crossed over. "
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
                "- 123 of 126 critical creatinine patients (97.6%) have never enrolled in the dialysis programme. "
                "3 patients crossed over. The programme served 54 enrolled patients at 80–135 sessions/month "
                "(NHIF-funded) — the referral pathway from critical creatinine detection to dialysis enrolment "
                "is not functioning for the outpatient renal cohort."
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

