"""
Patient Flow — pages/1_opd.py
OPD operations: demand profile, peak behaviour, bottleneck detection, pathway impact, root cause.
Source: rpt_ortho_patient_journey (V2) · rpt_ortho_opd (V1)

Investigative chain (6 sections):
  A. Activity Overview      — What operation are we observing?
  B. Demand Profile         — When does demand build?
  C. What Changes at Peak?  — Which stage first deteriorates?
  D. Bottleneck Detection   — Is that deterioration persistent and volume-related?
  E. Pathway Impact         — Does it affect the rest of the pathway?
  F. Root Cause & Action    — Why is it happening and what should be done?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Patient Flow · SPH Ortho", layout="wide", initial_sidebar_state="expanded")

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from facility_operations.dashboard.theme import (
    apply_theme, render_sidebar, kpi_card, section_header, info_card,
    page_header, COLORS, cl, _add_rolling_mean,
)
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from facility_operations.dashboard.queries import (
    q_opd_summary, q_opd_monthly, q_opd_hour, q_opd_dow_v2,
    q_peak_stage_tat,
    q_waiting_rbi_summary, q_waiting_dept_tat,
    q_waiting_weekly_tat, q_waiting_heatmap_flagged, q_waiting_dept_pressure,
    q_waiting_spillover_summary, q_waiting_service_breakdown,
)

apply_theme()
render_sidebar("opd")

_DAY_ORDER = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

# ── Page header ───────────────────────────────────────────────────────────────

page_header("Patient Flow", subtitle="OPD Operations · Peak Behaviour · Bottleneck Analysis")

# ── Load data ─────────────────────────────────────────────────────────────────

try:
    opd            = q_opd_summary().iloc[0]
    monthly        = q_opd_monthly()
    hour           = q_opd_hour()
    dow_v2         = q_opd_dow_v2()
    peak_stage_tat = q_peak_stage_tat()
    rbi_df         = q_waiting_rbi_summary()
    dept_df        = q_waiting_dept_tat()
    weekly_df      = q_waiting_weekly_tat()
    heat_fl_df     = q_waiting_heatmap_flagged()
    pressure_df    = q_waiting_dept_pressure()
    spillover_df   = q_waiting_spillover_summary()
    svc_df         = q_waiting_service_breakdown()
except Exception as e:
    st.error(f"Failed to load data. Check Snowflake connection.\n\n{e}")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _caption(text):
    st.markdown(
        f'<div style="font-size:9px;color:#9BAEC8;margin-top:-6px;margin-bottom:4px">{text}</div>',
        unsafe_allow_html=True,
    )


def _rbi_color(label):
    if label == "Bottleneck":
        return COLORS["danger"]
    if label == "Watch":
        return COLORS["warning"]
    return COLORS["success"]


def _trend_arrow(pct):
    if pct is None or pd.isna(pct):
        return "—", COLORS["muted"]
    if pct > 5:
        return f"↑ {abs(pct):.0f}%", COLORS["danger"]
    if pct < -5:
        return f"↓ {abs(pct):.0f}%", COLORS["success"]
    return "→ stable", COLORS["muted"]


def _safe_float(val, default=0.0):
    try:
        return float(val) if pd.notna(val) else default
    except (TypeError, ValueError):
        return default


def _safe_int(val, default=0):
    try:
        return int(round(float(val))) if pd.notna(val) else default
    except (TypeError, ValueError):
        return default


def _live_kpi_card(title, value, sub=None, sub_color=None):
    _sub_html = (
        f'<div style="font-size:10px;color:{sub_color or COLORS["muted"]};margin-top:4px">{sub}</div>'
        if sub else ""
    )
    st.markdown(
        f'<div style="background:#F8FBFE;border:1px solid #D6E4F0;border-radius:8px;'
        f'padding:16px 20px;text-align:center">'
        f'<div style="font-size:9px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
        f'letter-spacing:2px;margin-bottom:6px">{title}</div>'
        f'<div style="font-size:22px;font-weight:800;color:#003467;line-height:1.1">{value}</div>'
        f'{_sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _spill_card(title, normal_val, bn_val, delta_str, delta_color, note=""):
    _note_html = (
        f'<div style="font-size:9px;color:#9BAEC8;margin-top:6px">{note}</div>' if note else ""
    )
    st.markdown(
        f'<div style="background:#F8FBFE;border:1px solid #D6E4F0;border-radius:8px;padding:16px 18px">'
        f'<div style="font-size:9px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
        f'letter-spacing:1.8px;margin-bottom:10px">{title}</div>'
        f'<div style="display:flex;align-items:center;gap:12px">'
        f'<div style="text-align:center">'
        f'<div style="font-size:9px;color:#9BAEC8;margin-bottom:2px">Daily Avg</div>'
        f'<div style="font-size:20px;font-weight:800;color:#003467">{normal_val}</div>'
        f'</div>'
        f'<div style="color:#9BAEC8;font-size:16px">→</div>'
        f'<div style="text-align:center">'
        f'<div style="font-size:9px;color:#9BAEC8;margin-bottom:2px">Bottleneck</div>'
        f'<div style="font-size:20px;font-weight:800;color:#003467">{bn_val}</div>'
        f'</div>'
        f'<div style="margin-left:auto">'
        f'<span style="background:{delta_color}22;color:{delta_color};border:1px solid {delta_color}55;'
        f'font-size:11px;font-weight:700;padding:4px 10px;border-radius:4px">{delta_str}</span>'
        f'</div>'
        f'</div>'
        f'{_note_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# A. ACTIVITY OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

section_header("Activity Overview", margin_top=24)

data_from = pd.to_datetime(opd["DATA_FROM"]).strftime("%b %Y")
data_to   = pd.to_datetime(opd["DATA_TO"]).strftime("%b %Y")

# Row 1 — Historical scale
st.markdown(
    '<div style="font-size:9px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
    'letter-spacing:2px;margin-bottom:10px">Historical Scale</div>',
    unsafe_allow_html=True,
)
c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Total OPD Visits", f"{int(opd['TOTAL_VISITS']):,}",
             f"All visits · {data_from} – {data_to}")
with c2:
    kpi_card("Female Patients", f"{opd['FEMALE_PCT']:.1f}%",
             f"{int(opd['FEMALE_VISITS']):,} visits", color=COLORS["primary"])
with c3:
    kpi_card("Male Patients", f"{opd['MALE_PCT']:.1f}%",
             f"{int(opd['MALE_VISITS']):,} visits", color=COLORS["success"])
with c4:
    kpi_card("Date Range", f"{data_from}", f"→ {data_to}", color=COLORS["muted"])

st.markdown('<div style="margin-top:20px"></div>', unsafe_allow_html=True)

# Row 2 — Current operations (28d live)
st.markdown(
    '<div style="font-size:9px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
    'letter-spacing:2px;margin-bottom:10px">Current Operations · Last 28 Days</div>',
    unsafe_allow_html=True,
)

if not rbi_df.empty:
    _cons_row  = rbi_df[rbi_df["STAGE"] == "Consult"]
    _top_row   = rbi_df.iloc[0]

    _avg_visits     = _safe_int(_cons_row.iloc[0]["CURRENT_AVG_VOL"]) if not _cons_row.empty else 0
    _cons_p50       = _safe_int(_cons_row.iloc[0]["CURRENT_P50_MINS"]) if not _cons_row.empty else 0
    _cons_delta     = _safe_float(_cons_row.iloc[0]["PCT_CHANGE_28D"], default=None) if not _cons_row.empty else None
    _top_label      = str(_top_row["RBI_LABEL"]) if pd.notna(_top_row["RBI_LABEL"]) else "Normal"
    _top_ic         = _rbi_color(_top_label)
    _cons_delta_txt = f"{_cons_delta:+.0f}%" if _cons_delta is not None else "—"
    _cons_delta_col = (
        COLORS["danger"] if (_cons_delta or 0) > 5
        else (COLORS["success"] if (_cons_delta or 0) < -5 else COLORS["muted"])
    )

    dept_df.columns = dept_df.columns.str.upper() if not dept_df.empty else dept_df.columns
    _ortho_row = dept_df[dept_df["DEPT"] == "ORTHOPEDIC CONSULTATION"] if not dept_df.empty else pd.DataFrame()
    _ortho_p50 = int(_ortho_row.iloc[0]["P50_MINS"]) if not _ortho_row.empty else None
    _top_dept  = dept_df.iloc[0]["DEPT"].title() if not dept_df.empty else str(_top_row["STAGE"])
    _cov_cons  = _safe_float(_cons_row.iloc[0]["COVERAGE_PCT"]) if not _cons_row.empty else 0.0
    _cov_col   = (
        COLORS["danger"] if _cov_cons < 40
        else (COLORS["warning"] if _cov_cons < 70 else COLORS["success"])
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        _live_kpi_card("Avg Daily Visits", f"{_avg_visits:,}", "Last 28 days")
    with k2:
        _live_kpi_card("Arrival → Doctor", f"{_cons_p50} min",
                       f"vs prior 28d: {_cons_delta_txt}", _cons_delta_col)
    with k3:
        _live_kpi_card(
            "Ortho Consult Wait",
            f"{_ortho_p50} min" if _ortho_p50 else "—",
            "Orthopedic Consultation dept · Median",
        )
    with k4:
        _live_kpi_card(
            "Primary Wait Driver",
            _top_dept,
            f'<span style="background:{_top_ic}22;color:{_top_ic};border:1px solid {_top_ic}55;'
            f'font-size:9px;font-weight:700;padding:2px 8px;border-radius:4px">{_top_label}</span>',
        )
    with k5:
        _live_kpi_card(
            "Timestamp Coverage",
            f"{_cov_cons:.0f}%",
            "Data quality for consult TAT",
            _cov_col,
        )

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# B. DEMAND PROFILE
# ══════════════════════════════════════════════════════════════════════════════

section_header("Demand Profile")

# Monthly trend — full width
st.markdown(
    '<div style="font-size:11px;font-weight:600;color:#003467;margin-bottom:8px">'
    'Monthly OPD Visits</div>',
    unsafe_allow_html=True,
)
if not monthly.empty:
    monthly["VISIT_MONTH"] = pd.to_datetime(monthly["VISIT_MONTH"])
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=monthly["VISIT_MONTH"],
        y=monthly["VISITS"],
        mode="lines+markers",
        name="Monthly visits",
        line=dict(color=COLORS["primary"], width=2.5),
        marker=dict(size=4, color=COLORS["primary"]),
        hovertemplate="<b>%{x|%b %Y}</b>: %{y:,} visits<extra></extra>",
    ))
    _add_rolling_mean(fig_trend, monthly["VISIT_MONTH"], monthly["VISITS"],
                      n=3, name="3-mo avg", color=COLORS["muted"])
    _cutoff = pd.Timestamp("2025-02-01")
    fig_trend.add_shape(
        type="line", x0=_cutoff, x1=_cutoff, y0=0, y1=1, yref="paper",
        line=dict(dash="dot", color=COLORS.get("warning", "#F59E0B"), width=1.5),
    )
    fig_trend.add_annotation(
        x=_cutoff, y=1, yref="paper", text="V2 start", showarrow=False,
        xanchor="left", yanchor="top", font=dict(size=9, color="#003467"),
    )
    fig_trend.update_layout(**cl(
        height=280,
        showlegend=True,
        legend=dict(orientation="h", x=0, y=1.12, font=dict(size=10)),
        yaxis=dict(title="Visits", gridcolor="#EBF3FB"),
        xaxis=dict(gridcolor="#EBF3FB"),
        margin=dict(l=0, r=0, t=30, b=20),
    ))
    st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

st.markdown('<div style="margin-top:16px"></div>', unsafe_allow_html=True)

# Day of week + Hour of day side by side
col_a, col_b = st.columns(2, gap="large")

with col_a:
    st.markdown(
        '<div style="font-size:11px;font-weight:600;color:#003467;margin-bottom:8px">'
        'Visits by Day of Week (V2)</div>',
        unsafe_allow_html=True,
    )
    if not dow_v2.empty:
        _dv2 = dow_v2.copy()
        _dv2["VISIT_DAY_NAME"] = _dv2["VISIT_DAY_NAME"].str[:3]
        _dv2 = _dv2.sort_values("VISIT_DOW")
        _peak_day_v2 = _dv2.loc[_dv2["VISITS"].idxmax(), "VISIT_DAY_NAME"]
        _dv2_colors  = [
            COLORS["primary"] if d == _peak_day_v2 else COLORS["muted"]
            for d in _dv2["VISIT_DAY_NAME"]
        ]
        fig_dow = go.Figure(go.Bar(
            x=_dv2["VISIT_DAY_NAME"].tolist(),
            y=_dv2["VISITS"].tolist(),
            marker_color=_dv2_colors,
            opacity=0.85,
            text=[f"{v:,}" for v in _dv2["VISITS"]],
            textposition="outside",
            textfont=dict(size=9, color="#003467"),
            hovertemplate="<b>%{x}</b>: %{y:,} visits<extra></extra>",
        ))
        fig_dow.update_layout(**cl(
            height=260,
            showlegend=False,
            xaxis=dict(categoryorder="array", categoryarray=_DAY_ORDER,
                       gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor="#EBF3FB", title="Visits"),
            margin=dict(l=0, r=0, t=30, b=20),
        ))
        st.plotly_chart(fig_dow, use_container_width=True, config={"displayModeBar": False})

with col_b:
    st.markdown(
        '<div style="font-size:11px;font-weight:600;color:#003467;margin-bottom:8px">'
        'Arrivals by Hour of Day (V2)</div>',
        unsafe_allow_html=True,
    )
    if not hour.empty:
        _hr_colors = [
            COLORS["primary"] if int(h) in (10, 11) else COLORS["muted"]
            for h in hour["ARRIVAL_HOUR"]
        ]
        _hr_labels = [f"{int(h):02d}:00" for h in hour["ARRIVAL_HOUR"]]
        fig_hr = go.Figure(go.Bar(
            x=_hr_labels,
            y=hour["VISITS"].tolist(),
            marker_color=_hr_colors,
            opacity=0.85,
            hovertemplate="<b>%{x}</b>: %{y:,} visits<extra></extra>",
        ))
        fig_hr.update_layout(**cl(
            height=260,
            showlegend=False,
            xaxis=dict(tickangle=-45, tickfont=dict(size=9), gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor="#EBF3FB", title="Visits"),
            margin=dict(l=0, r=10, t=30, b=40),
        ))
        st.plotly_chart(fig_hr, use_container_width=True, config={"displayModeBar": False})

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# C. WHAT CHANGES AT PEAK?
# ══════════════════════════════════════════════════════════════════════════════

section_header("What Changes at Peak?")

# TAT helpers — scoped to this section
def _mv(band, col, band_dict):
    r = band_dict.get(band)
    if r is None:
        return None
    v = r.get(col)
    return int(v) if pd.notna(v) else None

def _dv(a, b):
    return (b - a) if (a is not None and b is not None) else None

def _fmt(v):
    return f"{v} min" if v is not None else "—"

def _fmt_d(v):
    if v is None:
        return "—"
    if v > 0:
        return f"+{v}"
    return "0" if v == 0 else str(v)

def _dcol(v, highlight=False):
    if v is None or v == 0:
        return "#6B8CAE"
    return COLORS["danger"] if (highlight and v > 0) else "#6B8CAE"

_tc_dp = None
_tc_da = None

if not peak_stage_tat.empty:
    _pst = peak_stage_tat.copy()
    _band_map = {
        "1 Early (07-10)":      "Early",
        "2 Peak (10-13)":       "Peak",
        "3 After-peak (13-17)": "After-peak",
    }
    _pst["BAND"] = _pst["HOUR_BAND"].map(_band_map)
    _bd = {row["BAND"]: row for _, row in _pst.iterrows()}

    _pt_e  = _mv("Early",      "PRETRIAGE_MEDIAN_MINS",    _bd)
    _pt_p  = _mv("Peak",       "PRETRIAGE_MEDIAN_MINS",    _bd)
    _pt_a  = _mv("After-peak", "PRETRIAGE_MEDIAN_MINS",    _bd)
    _pt_dp = _dv(_pt_e, _pt_p)
    _pt_da = _dv(_pt_e, _pt_a)

    _tc_e  = _mv("Early",      "TRIAGE_CONS_MEDIAN_MINS",  _bd)
    _tc_p  = _mv("Peak",       "TRIAGE_CONS_MEDIAN_MINS",  _bd)
    _tc_a  = _mv("After-peak", "TRIAGE_CONS_MEDIAN_MINS",  _bd)
    _tc_n  = _mv("Peak",       "TRIAGE_CONS_N",            _bd)
    _tc_dp = _dv(_tc_e, _tc_p)
    _tc_da = _dv(_tc_e, _tc_a)

    _cp_e  = _mv("Early",      "CONS_PHARM_MEDIAN_MINS",   _bd)
    _cp_p  = _mv("Peak",       "CONS_PHARM_MEDIAN_MINS",   _bd)
    _cp_a  = _mv("After-peak", "CONS_PHARM_MEDIAN_MINS",   _bd)
    _cp_dp = _dv(_cp_e, _cp_p)
    _cp_da = _dv(_cp_e, _cp_a)

    _tc_n_str = f"{_tc_n:,}" if isinstance(_tc_n, int) else "n/a"

    _TH  = ("font-size:10px;font-weight:700;color:#003467;padding:8px 12px;"
            "border-bottom:2px solid #D6E4F0;text-align:right;background:#F7FAFF")
    _THL = ("font-size:10px;font-weight:700;color:#003467;padding:8px 12px;"
            "border-bottom:2px solid #D6E4F0;text-align:left;background:#F7FAFF")
    _TD  = ("font-size:11px;color:#1E3A55;padding:7px 12px;"
            "text-align:right;border-bottom:1px solid #EBF3FB")
    _TDL = ("font-size:11px;color:#1E3A55;padding:7px 12px;"
            "text-align:left;border-bottom:1px solid #EBF3FB")
    _TDH  = ("font-size:11px;font-weight:700;color:#1E3A55;padding:7px 12px;"
             "text-align:right;border-bottom:1px solid #EBF3FB;background:#FEF9F0")
    _TDHL = ("font-size:11px;font-weight:700;color:#1E3A55;padding:7px 12px;"
             "text-align:left;border-bottom:1px solid #EBF3FB;background:#FEF9F0")

    _table_html = (
        '<div style="overflow-x:auto;margin-top:8px">'
        '<table style="width:100%;border-collapse:collapse">'
        "<thead><tr>"
        f'<th style="{_THL}">Stage</th>'
        f'<th style="{_TH}">Early<br><span style="font-weight:400;font-size:9px">(07–10)</span></th>'
        f'<th style="{_TH}">Peak<br><span style="font-weight:400;font-size:9px">(10–13)</span></th>'
        f'<th style="{_TH}">Δ Peak</th>'
        f'<th style="{_TH}">After-peak<br><span style="font-weight:400;font-size:9px">(13–17)</span></th>'
        f'<th style="{_TH}">Δ After</th>'
        "</tr></thead><tbody>"
        f'<tr><td style="{_TDL}">Pre-triage'
        f'<br><span style="font-size:8px;color:#9BAEC8">arrival→triage · 25% triage_ts coverage</span></td>'
        f'<td style="{_TD}">{_fmt(_pt_e)}</td>'
        f'<td style="{_TD}">{_fmt(_pt_p)}</td>'
        f'<td style="{_TD};color:{_dcol(_pt_dp)}">{_fmt_d(_pt_dp)}</td>'
        f'<td style="{_TD}">{_fmt(_pt_a)}</td>'
        f'<td style="{_TD};color:{_dcol(_pt_da)}">{_fmt_d(_pt_da)}</td>'
        "</tr>"
        f'<tr><td style="{_TDHL}">Triage → Consult'
        f'<br><span style="font-size:8px;color:#9BAEC8">triage→consult · ~30% coverage · n = {_tc_n_str} at peak</span></td>'
        f'<td style="{_TDH}">{_fmt(_tc_e)}</td>'
        f'<td style="{_TDH}">{_fmt(_tc_p)}</td>'
        f'<td style="{_TDH};color:{_dcol(_tc_dp, highlight=True)}">{_fmt_d(_tc_dp)}</td>'
        f'<td style="{_TDH}">{_fmt(_tc_a)}</td>'
        f'<td style="{_TDH};color:{_dcol(_tc_da, highlight=True)}">{_fmt_d(_tc_da)}</td>'
        "</tr>"
        f'<tr><td style="{_TDL}">Consult → Pharmacy'
        f'<br><span style="font-size:8px;color:#9BAEC8">consult→pharmacy · 92.8% pharm_ts coverage</span></td>'
        f'<td style="{_TD}">{_fmt(_cp_e)}</td>'
        f'<td style="{_TD}">{_fmt(_cp_p)}</td>'
        f'<td style="{_TD};color:{_dcol(_cp_dp)}">{_fmt_d(_cp_dp)}</td>'
        f'<td style="{_TD}">{_fmt(_cp_a)}</td>'
        f'<td style="{_TD};color:{_dcol(_cp_da)}">{_fmt_d(_cp_da)}</td>'
        "</tr>"
        "</tbody></table></div>"
        '<div style="font-size:8px;color:#9BAEC8;margin-top:6px">'
        "Baseline: Mar–Dec 2025 · V2 OPD only · Δ = difference vs Early band · medians in minutes"
        "</div>"
    )
    st.markdown(_table_html, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# D. BOTTLENECK DETECTION
# ══════════════════════════════════════════════════════════════════════════════

section_header("Bottleneck Detection")

info_card(
    "V2 OPD · Median TAT per stage. "
    "<b>RBI</b> = deviation from 14-day baseline + 7-day trend "
    "(labelled <b>Relative</b> — no SLA thresholds set). "
    "Coverage = valid timestamps ÷ eligible visits."
)

# ── D1: Stage ranking — evidence first ───────────────────────────────────────

_caption(
    "Sorted by RBI score — highest = current operational priority. "
    "Median TAT = 28-day avg of daily medians. "
    "Pharmacy TAT here = consult → pharmacy queue handoff (≠ dispensing TAT on Home page)."
)

if not rbi_df.empty:
    _rows_html = ""
    for _, _row in rbi_df.iterrows():
        _sl         = str(_row["RBI_LABEL"]) if pd.notna(_row["RBI_LABEL"]) else "—"
        _sc         = _rbi_color(_sl) if _sl not in ("—", "None") else COLORS["muted"]
        _p50        = f'{_safe_int(_row["CURRENT_P50_MINS"])} min' if pd.notna(_row["CURRENT_P50_MINS"]) else "—"
        _pct_val    = _safe_float(_row["PCT_CHANGE_28D"], default=None)
        _arrow, _ac = _trend_arrow(_pct_val)
        _rbi_v      = f'{_safe_float(_row["RBI_SCORE"]):.3f}' if pd.notna(_row["RBI_SCORE"]) else "—"
        _cov        = f'{_safe_float(_row["COVERAGE_PCT"]):.1f}%' if pd.notna(_row["COVERAGE_PCT"]) else "—"
        _owner      = str(_row["OPERATIONAL_OWNER"])

        _rows_html += (
            f'<tr style="border-bottom:1px solid #F0F5FA">'
            f'<td style="font-weight:700;color:#003467;padding:10px 12px">{_row["STAGE"]}</td>'
            f'<td style="color:#003467;padding:10px 12px;font-variant-numeric:tabular-nums">{_p50}</td>'
            f'<td style="color:{_ac};font-weight:700;padding:10px 12px">{_arrow}</td>'
            f'<td style="font-weight:800;color:{_sc};padding:10px 12px;font-variant-numeric:tabular-nums">{_rbi_v}</td>'
            f'<td style="padding:10px 12px">'
            f'<span style="background:{_sc}22;color:{_sc};border:1px solid {_sc}55;'
            f'font-size:10px;font-weight:700;padding:2px 8px;border-radius:4px">{_sl}</span>'
            f'</td>'
            f'<td style="color:#6B8CAE;padding:10px 12px">{_cov}</td>'
            f'<td style="color:#6B8CAE;font-size:11px;padding:10px 12px">{_owner}</td>'
            f'</tr>'
        )

    _hdr = (
        '<th style="text-align:left;padding:8px 12px;font-size:9px;font-weight:700;'
        'color:#6B8CAE;text-transform:uppercase;letter-spacing:1.5px">'
    )
    st.markdown(
        f'<table style="width:100%;border-collapse:collapse;font-size:13px">'
        f'<thead><tr style="border-bottom:2px solid #D6E4F0">'
        f'{_hdr}Stage</th>{_hdr}Median TAT (28d)</th>{_hdr}vs. Prior 28d</th>'
        f'{_hdr}RBI Score</th>{_hdr}Status</th>{_hdr}Coverage</th>{_hdr}Owner</th>'
        f'</tr></thead><tbody>{_rows_html}</tbody></table>',
        unsafe_allow_html=True,
    )

# ── D2: RBI headline — synthesis after evidence ───────────────────────────────

# ── D3: Persistence — weekly TAT & volume (12 weeks) ─────────────────────────

st.markdown('<div style="margin-top:28px"></div>', unsafe_allow_html=True)
st.markdown(
    '<div style="font-size:11px;font-weight:600;color:#003467;margin-bottom:4px">'
    'Is the Bottleneck Persistent? — Weekly TAT &amp; Volume · Last 12 Weeks</div>',
    unsafe_allow_html=True,
)
_caption(
    "Bars = weekly OPD visits (color = classification). Line = Median consult TAT. "
    "Orange: TAT > 90 min + volume > 15% above baseline — capacity pressure. "
    "Amber: TAT > 90 min, volume at baseline — process or staffing constraint. "
    "Blue: within normal range."
)

_has_flags = False

if not weekly_df.empty:
    _wk = weekly_df.copy()
    _wk["WEEK_START"] = pd.to_datetime(_wk["WEEK_START"])
    _wk = _wk.sort_values("WEEK_START").reset_index(drop=True)

    _n_capacity = int((_wk["FLAG_TYPE"] == "capacity").sum())
    _n_process  = int((_wk["FLAG_TYPE"] == "process").sum())
    _has_flags  = (_n_capacity + _n_process) > 0

    _BAR_COLOR = {
        "normal":   "rgba(0,114,206,0.35)",
        "capacity": "rgba(230,126,34,0.85)",
        "process":  "rgba(243,156,18,0.85)",
    }
    _bar_colors_wk = [_BAR_COLOR.get(f, "rgba(0,114,206,0.35)") for f in _wk["FLAG_TYPE"]]
    _hover_detail  = [
        (
            "Capacity pressure" if f == "capacity"
            else "Process/staffing constraint" if f == "process"
            else "Normal"
        ) + f" · Median TAT: {int(t) if pd.notna(t) else '—'} min"
        for f, t in zip(_wk["FLAG_TYPE"], _wk["P50_TAT_MINS"])
    ]

    fig_wk = go.Figure()
    fig_wk.add_trace(go.Bar(
        x=_wk["WEEK_START"].tolist(),
        y=_wk["WEEKLY_VISITS"].tolist(),
        name="Weekly visits",
        marker_color=_bar_colors_wk,
        yaxis="y1",
        customdata=_hover_detail,
        hovertemplate="<b>Week of %{x|%b %d}</b><br>Visits: %{y:,}<br>%{customdata}<extra></extra>",
        width=5.5 * 24 * 3600 * 1000,
    ))
    fig_wk.add_trace(go.Scatter(
        x=_wk["WEEK_START"].tolist(),
        y=_wk["P50_TAT_MINS"].tolist(),
        name="Median Consult TAT",
        mode="lines+markers",
        line=dict(color=COLORS["danger"], width=2.5),
        marker=dict(size=7, color=COLORS["danger"]),
        yaxis="y2",
        hovertemplate="<b>Week of %{x|%b %d}</b><br>Median TAT: %{y:.0f} min<extra></extra>",
    ))
    _x0 = _wk["WEEK_START"].min()
    _x1 = _wk["WEEK_START"].max() + pd.Timedelta(days=6)
    fig_wk.add_shape(
        type="line", xref="x", yref="y2", x0=_x0, x1=_x1, y0=90, y1=90,
        line=dict(color=COLORS["danger"], width=1, dash="dot"),
    )
    fig_wk.add_annotation(
        x=_x1, y=90, yref="y2", text="90 min", showarrow=False,
        xanchor="left", font=dict(size=9, color=COLORS["danger"]), bgcolor="white",
    )
    fig_wk.update_layout(**cl(
        height=320,
        showlegend=True,
        legend=dict(orientation="h", x=0, y=1.12, font=dict(size=10)),
        xaxis=dict(tickformat="%b %d", gridcolor="#EBF3FB"),
        yaxis=dict(title="Weekly Visits", gridcolor="#EBF3FB"),
        yaxis2=dict(title="Median Consult TAT (min)", overlaying="y", side="right", showgrid=False),
        margin=dict(l=0, r=80, t=36, b=20),
        bargap=0.15,
    ))
    st.plotly_chart(fig_wk, use_container_width=True)

    if _has_flags:
        _cap_txt = (
            f"<b style='color:{COLORS['danger']}'>{_n_capacity} "
            f"week{'s' if _n_capacity != 1 else ''} capacity pressure</b>"
            if _n_capacity else ""
        )
        _pro_txt = (
            f"<b style='color:{COLORS['warning']}'>{_n_process} "
            f"week{'s' if _n_process != 1 else ''} possible process/staffing constraint</b>"
            if _n_process else ""
        )
        _flag_summary = " &nbsp;·&nbsp; ".join(x for x in [_cap_txt, _pro_txt] if x)
        st.markdown(
            f'<div style="background:#FEF9F0;border:1px solid #F0C580;border-left:4px solid #E67E22;'
            f'border-radius:6px;padding:10px 16px;margin-top:-8px;font-size:12px;color:#1E3A55">'
            f'{_flag_summary}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:#F0FBF8;border:1px solid #A8DCC8;border-left:4px solid #27AE60;'
            'border-radius:6px;padding:10px 16px;margin-top:-8px;font-size:12px;color:#1E3A55">'
            'No sustained operational pressure detected in the last 12 weeks.</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# E. PATHWAY IMPACT
# ══════════════════════════════════════════════════════════════════════════════

section_header("Pathway Impact")

_spill_valid = (
    not spillover_df.empty
    and len(spillover_df) == 2
    and set(spillover_df["DAY_TYPE"].str.lower()) == {"bottleneck", "normal"}
)

_svc_valid = (
    not svc_df.empty
    and len(svc_df) == 2
    and set(svc_df["DAY_TYPE"].str.lower()) == {"bottleneck", "normal"}
)

# Initialise variables used in Section F
_pharm_wait_n = 0
_pharm_wait_b = 0
_pharm_wait_d = 0
_pharm_n      = 0.0
_pharm_b      = 0.0
_pharm_d      = 0.0

if _spill_valid:
    spillover_df.columns = spillover_df.columns.str.upper()
    _sp_n = spillover_df[spillover_df["DAY_TYPE"] == "normal"].iloc[0]
    _sp_b = spillover_df[spillover_df["DAY_TYPE"] == "bottleneck"].iloc[0]

    _days_b      = _safe_int(_sp_b["DAYS_N"])
    _days_n      = _safe_int(_sp_n["DAYS_N"])
    _total_days  = _days_b + _days_n
    _pct_bn_days = round(100 * _days_b / _total_days) if _total_days > 0 else 0

    _tat_n = _safe_int(_sp_n["AVG_P50_CONSULT_MINS"])
    _tat_b = _safe_int(_sp_b["AVG_P50_CONSULT_MINS"])
    _tat_d = _tat_b - _tat_n

    _anc_n = _safe_float(_sp_n["AVG_ANCILLARY_COMPLETION_PCT"])
    _anc_b = _safe_float(_sp_b["AVG_ANCILLARY_COMPLETION_PCT"])
    _anc_d = round(_anc_b - _anc_n, 1)

    _pharm_n = _safe_float(_sp_n["AVG_PHARMACY_PCT"])
    _pharm_b = _safe_float(_sp_b["AVG_PHARMACY_PCT"])
    _pharm_d = round(_pharm_b - _pharm_n, 1)

    _pharm_wait_n = _safe_int(_sp_n["AVG_P50_PHARM_WAIT_MINS"])
    _pharm_wait_b = _safe_int(_sp_b["AVG_P50_PHARM_WAIT_MINS"])
    _pharm_wait_d = _pharm_wait_b - _pharm_wait_n

    st.markdown(
        f'<div style="background:#FEF9F0;border:1px solid #F0C580;border-left:4px solid #E67E22;'
        f'border-radius:6px;padding:10px 16px;margin-bottom:16px;font-size:12px;color:#1E3A55">'
        f'<b>{_days_b} of {_total_days} operating days ({_pct_bn_days}%) were bottleneck days</b> '
        f'in the last 12 weeks. The operational constraint is the persistent baseline, not an exception.'
        f'</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        _spill_card("Median Consult TAT", f"{_tat_n} min", f"{_tat_b} min",
                    f"+{_tat_d} min", COLORS["danger"])
    with c2:
        _spill_card("Ancillary Completion", f"{_anc_n:.1f}%", f"{_anc_b:.1f}%",
                    f"{_anc_d:+.1f}pp", COLORS["danger"])
    with c3:
        _spill_card("Pharmacy Progression", f"{_pharm_n:.1f}%", f"{_pharm_b:.1f}%",
                    f"{_pharm_d:+.1f}pp", COLORS["warning"])

if _svc_valid:
    st.markdown('<div style="margin-top:20px"></div>', unsafe_allow_html=True)
    svc_df.columns = svc_df.columns.str.upper()
    _sn = svc_df[svc_df["DAY_TYPE"] == "normal"].iloc[0]
    _sb = svc_df[svc_df["DAY_TYPE"] == "bottleneck"].iloc[0]

    _services   = [("Pharmacy", "PHARMACY_PCT"), ("Procedures", "PROC_PCT"),
                   ("Imaging", "IMAGING_PCT"), ("Lab", "LAB_PCT")]
    _svc_labels = [s[0] for s in _services]
    _svc_normal = [_safe_float(_sn[s[1]]) for s in _services]
    _svc_bn     = [_safe_float(_sb[s[1]]) for s in _services]
    _svc_deltas = [round(_svc_bn[i] - _svc_normal[i], 1) for i in range(len(_services))]
    _bn_colors  = [
        COLORS["danger"]  if d < -1.5 else
        COLORS["warning"] if d < 0    else
        COLORS["success"]
        for d in _svc_deltas
    ]

    fig_svc = go.Figure()
    fig_svc.add_trace(go.Bar(
        x=_svc_normal, y=_svc_labels, name="Normal days", orientation="h",
        marker_color="rgba(0,114,206,0.30)",
        hovertemplate="<b>%{y}</b><br>Normal days: %{x:.1f}%<extra></extra>",
    ))
    fig_svc.add_trace(go.Bar(
        x=_svc_bn, y=_svc_labels, name="Bottleneck days", orientation="h",
        marker_color=_bn_colors,
        text=[f"{d:+.1f}pp" for d in _svc_deltas],
        textposition="outside",
        textfont=dict(size=11, color="#1E3A55"),
        hovertemplate="<b>%{y}</b><br>Bottleneck days: %{x:.1f}%<br>%{text}<extra></extra>",
    ))
    _x_max = max(max(_svc_normal), max(_svc_bn))
    fig_svc.update_layout(**cl(
        barmode="group", height=280, showlegend=True,
        legend=dict(orientation="h", x=0, y=1.12, font=dict(size=10)),
        xaxis=dict(title="% of OPD visits with service record",
                   range=[0, _x_max * 1.35], gridcolor="#EBF3FB", ticksuffix="%"),
        yaxis=dict(tickfont=dict(size=12)),
        margin=dict(l=0, r=100, t=36, b=20),
    ))
    st.plotly_chart(fig_svc, use_container_width=True)

# ── Section E finding ─────────────────────────────────────────────────────────

if _spill_valid:
    st.markdown(
        '<div style="background:#EBF3FB;border-left:4px solid #003467;'
        'border-radius:4px;padding:10px 14px;margin:16px 0 0 0;'
        'font-size:11px;font-weight:600;color:#003467">'
        'The bottleneck reduces downstream pathway progression — fewer patients reach pharmacy, '
        'procedures, and ancillary services on constrained days. '
        'Downstream capacity is not the limiting factor: pharmacy queue wait is unchanged. '
        'The constraint is upstream.'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# F. ROOT CAUSE & ACTION
# ══════════════════════════════════════════════════════════════════════════════

section_header("Root Cause & Action")

# ── F drill-downs — gated on flagged weeks ────────────────────────────────────

if _has_flags:
    st.markdown('<div style="margin-top:28px"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:11px;font-weight:600;color:#003467;margin-bottom:4px">'
        'When Do Delays Occur? — Flagged Weeks Only · Day × Hour</div>',
        unsafe_allow_html=True,
    )
    _caption(
        "Median consult TAT per day-of-week × arrival hour, aggregated across flagged weeks only. "
        "Dark red = longest waits. Use this to direct roster review to the right shift and day."
    )

    if not heat_fl_df.empty:
        hm = heat_fl_df[heat_fl_df["MEDIAN_CONS_TAT_MINS"].notna()].copy()
        if not hm.empty:
            import numpy as np  # noqa: F401  used implicitly by plotly heatmap
            _DOW_HM   = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
            _dow_map  = {d: i for i, d in enumerate(_DOW_HM)}
            hm["_di"] = hm["DAY_NAME"].map(_dow_map)
            hm = hm.dropna(subset=["_di"])
            hm["_di"] = hm["_di"].astype(int)

            _hours = sorted(hm["HOUR_OF_DAY"].unique())
            _days  = [d for d in _DOW_HM if d in hm["DAY_NAME"].values]

            _z, _text = [], []
            for h in _hours:
                _z_row, _t_row = [], []
                for d in _days:
                    cell = hm[(hm["HOUR_OF_DAY"] == h) & (hm["DAY_NAME"] == d)]
                    if cell.empty:
                        _z_row.append(None); _t_row.append("")
                    else:
                        tat = float(cell.iloc[0]["MEDIAN_CONS_TAT_MINS"])
                        cnt = int(cell.iloc[0]["VISIT_COUNT"])
                        _z_row.append(tat); _t_row.append(f"{tat:.0f} min<br>{cnt} visits")
                _z.append(_z_row); _text.append(_t_row)

            fig_hm = go.Figure(go.Heatmap(
                z=_z, x=_days, y=[f"{h:02d}:00" for h in _hours],
                text=_text,
                hovertemplate="<b>%{x} %{y}</b><br>%{text}<extra></extra>",
                colorscale=[
                    [0.0, "#EBF7F0"], [0.35, "#7ECFA4"],
                    [0.65, "#F0A830"], [1.0,  "#C0392B"],
                ],
                colorbar=dict(
                    title=dict(text="Median TAT (min)", font=dict(size=10)),
                    thickness=12, len=0.8,
                ),
                zsmooth=False, xgap=2, ygap=2,
            ))
            fig_hm.update_layout(**cl(
                height=420, showlegend=False,
                xaxis=dict(side="top", tickfont=dict(size=11, color="#003467")),
                yaxis=dict(autorange="reversed", tickfont=dict(size=10, color="#6B8CAE")),
                margin=dict(l=0, r=60, t=40, b=20),
            ))
            st.plotly_chart(fig_hm, use_container_width=True)

if _has_flags:
    st.markdown('<div style="margin-top:20px"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:11px;font-weight:600;color:#003467;margin-bottom:4px">'
        'Which Departments Contribute Most During Periods of Operational Pressure?</div>',
        unsafe_allow_html=True,
    )
    _caption(
        "Blue = 28-day baseline median. Colored bar = median during flagged weeks. "
        "Large gap = department contributes disproportionately during flagged periods. "
        "Flat bars = elevated regardless of period — structural pattern, not pressure response."
    )

    if not pressure_df.empty:
        _dp = pressure_df[
            pressure_df["P50_BASELINE"].notna() & pressure_df["P50_FLAGGED"].notna()
        ].copy().sort_values("DELTA_MINS", ascending=True)

        _flagged_colors = [
            COLORS["danger"]  if d > 30 else
            COLORS["warning"] if d > 0  else
            COLORS["success"]
            for d in _dp["DELTA_MINS"]
        ]

        fig_pr = go.Figure()
        fig_pr.add_trace(go.Bar(
            x=_dp["P50_BASELINE"].tolist(),
            y=_dp["DEPT"].str.title().tolist(),
            name="28-day baseline",
            orientation="h",
            marker_color="rgba(0,114,206,0.35)",
            hovertemplate="<b>%{y}</b><br>Baseline Median: %{x} min<extra></extra>",
        ))
        _delta_labels = [
            f"+{int(d)} min ({pct:+.0f}%)" if d > 0 else f"{int(d)} min ({pct:+.0f}%)"
            for d, pct in zip(_dp["DELTA_MINS"], _dp["DELTA_PCT"])
        ]
        fig_pr.add_trace(go.Bar(
            x=_dp["P50_FLAGGED"].tolist(),
            y=_dp["DEPT"].str.title().tolist(),
            name="During flagged weeks",
            orientation="h",
            marker_color=_flagged_colors,
            text=_delta_labels,
            textposition="outside",
            textfont=dict(size=10),
            customdata=_delta_labels,
            hovertemplate="<b>%{y}</b><br>Flagged-week Median: %{x} min<br>%{customdata}<extra></extra>",
        ))
        _x_max = max(_dp["P50_BASELINE"].max(), _dp["P50_FLAGGED"].max())
        fig_pr.update_layout(**cl(
            barmode="group",
            height=max(260, len(_dp) * 52),
            showlegend=True,
            legend=dict(orientation="h", x=0, y=1.08, font=dict(size=10)),
            xaxis=dict(title="Median Consult TAT (min)",
                       range=[0, _x_max * 1.4], gridcolor="#EBF3FB"),
            yaxis=dict(tickfont=dict(size=11)),
            margin=dict(l=0, r=160, t=36, b=20),
        ))
        st.plotly_chart(fig_pr, use_container_width=True)
    else:
        st.markdown(
            '<div style="font-size:12px;color:#6B8CAE;padding:12px 0">'
            'Insufficient data — fewer than 5 timed consults per department in flagged weeks.</div>',
            unsafe_allow_html=True,
        )

# ── Mechanism & recommendation — always last ──────────────────────────────────

if _spill_valid and _svc_valid:
    st.markdown('<div style="margin-top:28px"></div>', unsafe_allow_html=True)
    col_mech, col_rec = st.columns(2)

    with col_mech:
        st.markdown(
            f'<div style="background:#F8FBFE;border:1px solid #D6E4F0;border-left:4px solid #0072CE;'
            f'border-radius:8px;padding:16px 20px;height:100%">'
            f'<div style="font-size:9px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
            f'letter-spacing:1.8px;margin-bottom:10px">Why Does This Happen?</div>'
            f'<div style="font-size:12px;color:#1E3A55;line-height:1.7">'
            f'During bottleneck days, pharmacy queue wait increases by '
            f'<b>{_pharm_wait_d} minutes</b> ({_pharm_wait_n} → {_pharm_wait_b} min) — '
            f'a negligible change for those who arrive.<br><br>'
            f'The pharmacy progression rate falls by <b>{abs(_pharm_d):.1f} percentage points</b> '
            f'({_pharm_n:.1f}% → {_pharm_b:.1f}%). '
            f'Fewer patients complete the consultation and reach the pharmacy queue. '
            f'The bottleneck is the consult handoff, not pharmacy capacity.'
            f'</div>'
            f'<div style="margin-top:12px;font-size:11px;color:#6B8CAE">'
            f'Mechanism: <b>reduced consult throughput</b>, not downstream queue growth.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_rec:
        st.markdown(
            f'<div style="background:#FEF9F0;border:1px solid #F0C580;border-left:4px solid #E67E22;'
            f'border-radius:8px;padding:16px 20px;height:100%">'
            f'<div style="font-size:9px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
            f'letter-spacing:1.8px;margin-bottom:10px">What Should Management Do?</div>'
            f'<div style="font-size:12px;color:#1E3A55;line-height:1.7">'
            f'Consult throughput is the limiting constraint. Downstream services '
            f'(pharmacy, procedures) appear underperforming because patients do not reach '
            f'them — not because those services are overloaded when patients arrive.<br><br>'
            f'Expanding pharmacy or diagnostic capacity will not resolve the bottleneck. '
            f'The intervention target is clinician allocation and the triage-to-clinician '
            f'handoff during peak and high-volume periods.'
            f'</div>'
            f'<div style="margin-top:12px;font-size:11px;color:#E67E22;font-weight:600">'
            f'Action: increase consult capacity on high-volume days.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
