"""
Capacity Pressure — pages/9_capacity.py
Source: mart_capacity · V2 (ward/theatre/lab) · last 90 days

Decision question: Which operational area is under the greatest capacity
pressure today, and where should managers intervene first?

Metric: Δ% = (current − 14d baseline) / 14d baseline × 100
Terms: Pressure / Load / Deviation from baseline only.
No utilization %, capacity %, or occupancy target — denominators not yet defined.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="Capacity Pressure · SPH Ortho",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from facility_operations.dashboard.theme import apply_theme, render_sidebar, page_header, COLORS, cl

from facility_operations.dashboard.queries import q_capacity_snapshot, q_capacity_trend

apply_theme()
render_sidebar("capacity")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_THRESHOLD   = 10.0   # Δ% above this triggers an alert
_MIN_BASE    = {"ward": 5.0, "theatre": 2.0, "lab": 10.0}

_SERVICE_META = {
    "ward":    {
        "label":  "Ward Census",
        "action": "Review bed availability and defer elective admissions where possible.",
    },
    "theatre": {
        "label":  "Theatre Load",
        "action": "Review theatre scheduling and pre-op pathway to manage queue.",
    },
    "lab":     {
        "label":  "Lab Throughput",
        "action": "Review lab turnaround capacity and prioritize urgent requests.",
    },
}

_SVC_COLORS = {
    "ward":    COLORS["primary"],
    "theatre": COLORS["warning"],
    "lab":     COLORS["green"],
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _act(label):
    st.markdown(
        f'<div style="border-top:1.5px solid #D6E4F0;margin:40px 0 20px 0;padding-top:14px">'
        f'<span style="font-size:9px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
        f'letter-spacing:2.5px">{label}</span></div>',
        unsafe_allow_html=True,
    )


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


def _status(delta_pct):
    if delta_pct > 20:  return "HIGH",           COLORS["danger"]
    if delta_pct > 10:  return "ELEVATED",        COLORS["warning"]
    if delta_pct >= -10: return "NORMAL",         COLORS["success"]
    return "Below Baseline", COLORS["muted"]


def _scorecard(title, current, baseline, delta_pct, unit="", min_base=None):
    suppress = min_base is not None and baseline < min_base
    if suppress:
        status_lbl, status_col = "Low Volume", COLORS["muted"]
        delta_str = "—"
    else:
        status_lbl, status_col = _status(delta_pct)
        sign = "+" if delta_pct > 0 else ""
        delta_str = f"{sign}{delta_pct:.0f}%"
    st.markdown(
        f'<div style="background:#F8FBFE;border:1px solid #D6E4F0;border-radius:8px;padding:28px 24px">'
        f'<div style="font-size:11px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
        f'letter-spacing:2px;margin-bottom:10px">{title}</div>'
        f'<div style="font-size:36px;font-weight:800;color:#003467;line-height:1">{current}{unit}</div>'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-top:14px">'
        f'<div style="font-size:11px;color:#6B8CAE">14d baseline: {baseline:.1f}{unit}</div>'
        f'<div style="display:flex;align-items:center;gap:8px">'
        f'<span style="font-size:14px;font-weight:700;color:{status_col}">{delta_str}</span>'
        f'<span style="background:{status_col};color:#fff;font-size:9px;font-weight:700;'
        f'padding:3px 8px;border-radius:4px;letter-spacing:1px">{status_lbl}</span>'
        f'</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

snap_df  = q_capacity_snapshot()
trend_df = q_capacity_trend()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════

page_header(
    "Where Is Capacity Under Pressure?",
    period="Last 90 days",
    mode="live",
)


# ══════════════════════════════════════════════════════════════════════════════
# KEY INSIGHT CARD
# ══════════════════════════════════════════════════════════════════════════════

if not snap_df.empty:
    _s = snap_df.iloc[0]
    _snap_date = pd.to_datetime(_s["PERIOD_DATE"]).strftime("%d %b %Y")

    _svc_vals = {
        "ward":    (_safe_float(_s["WARD_DELTA_PCT"]),    _safe_float(_s["WARD_BASELINE"])),
        "theatre": (_safe_float(_s["THEATRE_DELTA_PCT"]), _safe_float(_s["THEATRE_BASELINE"])),
        "lab":     (_safe_float(_s["LAB_DELTA_PCT"]),     _safe_float(_s["LAB_BASELINE"])),
    }

    # Rank: above threshold, above min baseline, highest Δ%
    candidates = {
        svc: delta
        for svc, (delta, base) in _svc_vals.items()
        if base >= _MIN_BASE[svc] and delta > _THRESHOLD
    }

    if candidates:
        _top = max(candidates, key=candidates.get)
        _top_delta = candidates[_top]
        _meta = _SERVICE_META[_top]
        _ic = COLORS["danger"] if _top_delta > 20 else COLORS["warning"]
        _headline = (
            f"{_meta['label']} is <b>+{_top_delta:.0f}% above baseline</b> "
            f"as of {_snap_date}."
        )
        _body = _meta["action"]
    else:
        _ic = COLORS["success"]
        _headline = f"No service is under unusual operational pressure as of {_snap_date}."
        _body = "All three areas — Ward, Theatre, and Lab — are within 10% of their 14-day baseline."

    st.markdown(
        f'<div style="background:#F8FBFE;border:1px solid #D6E4F0;border-left:5px solid {_ic};'
        f'border-radius:8px;padding:20px 24px;margin-bottom:28px">'
        f'<div style="font-size:9px;font-weight:700;color:{_ic};text-transform:uppercase;'
        f'letter-spacing:2px;margin-bottom:8px">Operational Pressure · {_snap_date}</div>'
        f'<div style="font-size:19px;font-weight:800;color:#003467;line-height:1.3;margin-bottom:8px">'
        f'{_headline}</div>'
        f'<div style="font-size:13px;color:#1E3A55;line-height:1.65">{_body}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PRESSURE SCORECARDS
# ══════════════════════════════════════════════════════════════════════════════

_act("Pressure Scorecards")

if not snap_df.empty:
    c1, c2, c3 = st.columns(3)
    with c1:
        _scorecard(
            "Ward Census",
            _safe_int(_s["WARD_CURRENT"]),
            _safe_float(_s["WARD_BASELINE"]),
            _safe_float(_s["WARD_DELTA_PCT"]),
            min_base=_MIN_BASE["ward"],
        )
    with c2:
        _scorecard(
            "Theatre Load",
            _safe_int(_s["THEATRE_CURRENT"]),
            _safe_float(_s["THEATRE_BASELINE"]),
            _safe_float(_s["THEATRE_DELTA_PCT"]),
            min_base=_MIN_BASE["theatre"],
        )
    with c3:
        _scorecard(
            "Lab Throughput",
            _safe_int(_s["LAB_CURRENT"]),
            _safe_float(_s["LAB_BASELINE"]),
            _safe_float(_s["LAB_DELTA_PCT"]),
            min_base=_MIN_BASE["lab"],
        )

    st.markdown(
        '<div style="margin-top:10px;font-size:11px;color:#6B8CAE;font-style:italic">'
        'Theatre data not yet available for 2026 — V2 source ends Dec 2025.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="margin-bottom:8px"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZED PRESSURE TREND
# ══════════════════════════════════════════════════════════════════════════════

_act("Pressure Trend — Δ% from 14-day Baseline")

if not trend_df.empty:
    td = trend_df.copy()
    td["PERIOD_DATE"] = pd.to_datetime(td["PERIOD_DATE"])

    # Winsorize to ±100% — prevents small theatre baseline causing extreme swings
    for col in ["WARD_DELTA_PCT", "THEATRE_DELTA_PCT", "LAB_DELTA_PCT"]:
        td[col] = td[col].clip(-100, 100)

    fig_trend = go.Figure()

    # Alert threshold band
    fig_trend.add_hrect(
        y0=_THRESHOLD, y1=100,
        fillcolor=COLORS["danger"], opacity=0.04,
        line_width=0,
    )

    # Threshold line
    fig_trend.add_hline(
        y=_THRESHOLD, line_dash="dot",
        line_color=COLORS["warning"], line_width=1,
        annotation_text="Alert threshold (+10%)",
        annotation_position="top left",
        annotation_font=dict(size=9, color=COLORS["warning"]),
    )

    # Baseline zero line
    fig_trend.add_hline(y=0, line_color="#D6E4F0", line_width=1)

    # Theatre suppressed — no V2 data after Dec 2025; near-zero baseline produces misleading spikes
    _traces = [
        ("WARD_DELTA_PCT", "WARD_CURRENT", "WARD_BASELINE", "Ward Census",    _SVC_COLORS["ward"]),
        ("LAB_DELTA_PCT",  "LAB_CURRENT",  "LAB_BASELINE",  "Lab Throughput", _SVC_COLORS["lab"]),
    ]

    for delta_col, curr_col, base_col, name, color in _traces:
        fig_trend.add_trace(go.Scatter(
            x=td["PERIOD_DATE"].tolist(),
            y=td[delta_col].tolist(),
            name=name,
            mode="lines",
            line=dict(color=color, width=1.8),
            hovertemplate=(
                f"<b>{name}</b><br>"
                "%{x|%d %b %Y}<br>"
                "Δ%%: %{y:+.1f}%%<br>"
                "Current: %{customdata[0]}<br>"
                "Baseline: %{customdata[1]:.1f}"
                "<extra></extra>"
            ),
            customdata=list(zip(td[curr_col].tolist(), td[base_col].tolist())),
        ))

    fig_trend.update_layout(**cl(
        height=320,
        showlegend=True,
        legend=dict(orientation="h", x=0, y=1.12, font=dict(size=10)),
        yaxis=dict(
            title="Δ% from 14-day baseline",
            gridcolor="#EBF3FB",
            zeroline=False,
            range=[-110, 110],
        ),
        xaxis=dict(gridcolor="#EBF3FB"),
        margin=dict(l=0, r=20, t=36, b=20),
    ))
    st.plotly_chart(fig_trend, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PRESSURE HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

_act("Pressure Heatmap — Sustained vs Spike")

if not trend_df.empty:
    td_hm = trend_df.copy()
    td_hm["PERIOD_DATE"] = pd.to_datetime(td_hm["PERIOD_DATE"])

    x_dates = [d.strftime("%b %d") for d in td_hm["PERIOD_DATE"]]
    # Theatre suppressed — baseline near zero produces misleading +900% spikes from single cases
    services = ["Ward Census", "Lab Throughput"]
    cols     = ["WARD_DELTA_PCT", "LAB_DELTA_PCT"]

    Z = [
        [max(-100.0, min(100.0, v)) for v in td_hm[c].tolist()]
        for c in cols
    ]

    fig_hm = go.Figure(go.Heatmap(
        z=Z,
        x=x_dates,
        y=services,
        colorscale=[
            [0.0,  "#1a6fa8"],   # −100% deep blue
            [0.5,  "#ffffff"],   # 0% white
            [1.0,  "#E11D48"],   # +100% red
        ],
        zmid=0,
        zmin=-100,
        zmax=100,
        colorbar=dict(
            title=dict(text="Δ%", side="right"),
            tickvals=[-100, -50, 0, 50, 100],
            ticktext=["−100%", "−50%", "Baseline", "+50%", "+100%"],
            thickness=12,
            len=0.8,
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "%{x}<br>"
            "Δ%%: %{z:+.1f}%%"
            "<extra></extra>"
        ),
    ))

    fig_hm.update_layout(**cl(
        height=220,
        margin=dict(l=0, r=60, t=10, b=40),
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=8),
            gridcolor="#EBF3FB",
            nticks=18,
        ),
        yaxis=dict(gridcolor="#EBF3FB"),
    ))
    st.plotly_chart(fig_hm, use_container_width=True)
