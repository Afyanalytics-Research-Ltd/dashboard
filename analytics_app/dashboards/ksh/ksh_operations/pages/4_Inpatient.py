"""
KSH V3 â€” Inpatient & Bed Management
Gold: RPT_KSH_V3_INPATIENT_FLOW Â· RPT_KSH_V3_BOR_MONTHLY Â· RPT_KSH_V3_INPATIENT_SUMMARY
Structure: Chapter 1 â€” Patient Flow Â· Chapter 2 â€” Capacity Â· Chapter 3 â€” Financial Impact
Signal: Indicative throughout â€” BOR from registered bed count (unconfirmed on site).
Thresholds: facility-calibrated operational thresholds, not clinical standards.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from ksh_operations.dashboard.theme import apply_theme, render_sidebar, section_header, page_header, COLORS
from ksh_operations.dashboard.queries import (
    q_inpatient_flow, q_inpatient_flow_trend, q_inpatient_bor_monthly,
    q_inpatient_alos_monthly, q_inpatient_bor_ward, q_inpatient_bor_payer,
    q_inpatient_c2b_trend, q_inpatient_alos_ward, q_inpatient_discharge_tat_trend,
)

apply_theme()
render_sidebar(active="inpatient")
page_header("Inpatient", subtitle="Patient flow Â· Capacity Â· Financial impact")

C_BLUE  = COLORS["primary"]
C_NAVY  = COLORS["dark"]
C_RED   = "#E11D48"
C_GREEN = "#16A34A"
C_AMBER = "#F59E0B"

# â”€â”€ Load â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_if  = q_inpatient_flow();         _if.columns  = [c.upper() for c in _if.columns]
_it  = q_inpatient_flow_trend();   _it.columns  = [c.upper() for c in _it.columns]
_bm  = q_inpatient_bor_monthly();  _bm.columns  = [c.upper() for c in _bm.columns]
_bw  = q_inpatient_bor_ward();     _bw.columns  = [c.upper() for c in _bw.columns]
_bp  = q_inpatient_bor_payer();    _bp.columns  = [c.upper() for c in _bp.columns]
_c2b = q_inpatient_c2b_trend();            _c2b.columns = [c.upper() for c in _c2b.columns]
_al  = q_inpatient_alos_monthly();         _al.columns  = [c.upper() for c in _al.columns]
_aw  = q_inpatient_alos_ward();            _aw.columns  = [c.upper() for c in _aw.columns]
_dt  = q_inpatient_discharge_tat_trend();  _dt.columns  = [c.upper() for c in _dt.columns]

_rbm = _bm.iloc[0] if not _bm.empty else None
_rf  = _if.iloc[0] if not _if.empty else None
_ral = _al.iloc[0] if not _al.empty else None
_it  = _it.sort_values("VISIT_MONTH")
_cap_month = str(_bw["MONTH_START"].iloc[0])[:7] if not _bw.empty else ""


def _get(row, col):
    if row is None: return None
    v = row.get(col)
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v


def _delta_html(val, unit="", good="down"):
    if val is None: return ""
    if good == "neutral": color = "#6B8CAE"
    else: color = C_GREEN if ((val < 0) if good == "down" else (val > 0)) else C_RED
    arrow = "â–¼" if val < 0 else "â–²"
    sign  = "+" if val > 0 else ""
    return (f'<div style="font-size:11px;font-weight:600;color:{color};margin-top:4px">'
            f'{arrow} {sign}{val}{unit}</div>')


def _card(col, label, value, sub=None, delta=None):
    sub_html   = "" if not sub   else f'<div style="font-size:10px;color:#6B8CAE;margin-top:5px;line-height:1.4">{sub}</div>'
    delta_html = "" if not delta else delta
    col.markdown(
        f'<div style="background:#FFFFFF;border:1px solid #0072CE;border-left:4px solid #0072CE;'
        f'border-radius:8px;padding:14px 16px;text-align:center;min-height:100px">'
        f'<div style="font-size:9px;font-weight:700;color:#0072CE;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:6px">{label}</div>'
        f'<div style="font-size:28px;font-weight:800;color:#003467">{value}</div>'
        f'{delta_html}{sub_html}</div>',
        unsafe_allow_html=True,
    )


# â”€â”€ Freshness â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_data_as_of = _get(_rbm, "DATA_AS_OF")
_days_stale = _get(_rbm, "DAYS_SINCE_DATA")
if _data_as_of:
    st.caption(f"Data as of {str(_data_as_of)[:10]}")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CHAPTER 1 â€” PATIENT FLOW
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
section_header("Patient Flow", margin_top=8)

_conv       = _get(_rf, "ANY_CONVERSION_PCT")
_conv_prior = _get(_rf, "PRIOR_ANY_CONVERSION_PCT")
_conv_delta = round(float(_conv) - float(_conv_prior), 2) if _conv and _conv_prior else None
_opd_n      = _get(_rf, "OPD_VISIT_COUNT")
_any_n      = _get(_rf, "CONVERTED_ANY")
_prior_opd  = _get(_rf, "PRIOR_OPD_VISIT_COUNT")
_prior_any  = _get(_rf, "PRIOR_CONVERTED_ANY")
_month_lbl  = str(_get(_rf, "VISIT_MONTH"))[:7] if _get(_rf, "VISIT_MONTH") else ""
_prior_lbl  = str(_get(_rf, "PRIOR_MONTH"))[:7] if _get(_rf, "PRIOR_MONTH") else "prior month"
_prior_line = (f" Â· {_prior_lbl}: {float(_conv_prior):.2f}%"
               if _conv_prior else "")

_alos_v     = _get(_ral, "ALOS_DAYS")
_alos_prior = _get(_ral, "PRIOR_ALOS_DAYS")
_alos_delta = round(float(_alos_v) - float(_alos_prior), 1) if _alos_v and _alos_prior else None
_alos_n     = _get(_ral, "DISCHARGE_N")
_alos_m     = str(_get(_ral, "DISCHARGE_MONTH"))[:7] if _get(_ral, "DISCHARGE_MONTH") else ""
_alos_pm    = str(_get(_ral, "PRIOR_MONTH"))[:7]     if _get(_ral, "PRIOR_MONTH")     else "prior month"

# Admission Decision-to-Bed â€” latest week + 12-week aggregate
_c2b_breach_latest = None
_c2b_week_n        = None
_c2b_week_lbl      = None
_c2b_agg_pct       = None
if not _c2b.empty:
    _lc2b              = _c2b.iloc[-1]
    _c2b_breach_latest = _lc2b.get("PCT_OVER_4H")
    _c2b_week_n        = int(_lc2b.get("C2B_N")) if _lc2b.get("C2B_N") else None
    _c2b_week_lbl      = str(_lc2b.get("WEEK_START"))[:10] if _lc2b.get("WEEK_START") else None
    _tot_n    = _c2b["C2B_N"].sum()
    _tot_over = _c2b["C2B_OVER_4H_N"].sum()
    _c2b_agg_pct = round(_tot_over / _tot_n * 100, 1) if _tot_n else None

_cf1, _cf2, _cf3 = st.columns(3)

_card(_cf1, "OPD â†’ IPD Conversion Rate",
    f"{float(_conv):.2f}%" if _conv else "â€”",
    sub=(f"{int(_any_n):,} of {int(_opd_n):,} OPD visits Â· {_month_lbl} Â· Indicative"
         if _any_n and _opd_n else f"{_month_lbl} Â· Indicative"),
    delta=_delta_html(_conv_delta, "pp vs prior month", good="neutral"))

_card(_cf2, "Avg Length of Stay",
    f"{_alos_v} days" if _alos_v else "â€”",
    sub=(f"{int(_alos_n):,} overnight discharges Â· {_alos_m} Â· Indicative"
         if _alos_n else "overnight discharges Â· Indicative"),
    delta=_delta_html(_alos_delta, "d vs prior month", good="down"))

_card(_cf3, "Admissions Exceeding 4 Hours",
    f"{_c2b_agg_pct}%" if _c2b_agg_pct is not None else "â€”",
    sub=(f"12-wk aggregate Â· latest week {_c2b_breach_latest}% (n={_c2b_week_n}) Â· Indicative"
         if _c2b_week_lbl else "decision-to-bed > 4 hrs Â· Indicative"))

st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

# â”€â”€ Chart 1 â€” OPD Volume vs Admission Rate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if not _it.empty:
    st.markdown('<div style="font-size:13px;font-weight:600;color:#003467;margin-bottom:4px">OPD Volume vs Admission Rate â€” 12-Month Trend</div>', unsafe_allow_html=True)
    _months = _it["VISIT_MONTH"].astype(str).str[:7].tolist()
    _fig = go.Figure()
    _fig.add_trace(go.Bar(
        x=_months, y=_it["OPD_VISIT_COUNT"].tolist(),
        name="OPD Visits", yaxis="y2",
        marker_color="#0D9488", opacity=0.75,
        hovertemplate="%{x}<br>OPD visits: %{y:,}<extra></extra>",
    ))
    _fig.add_trace(go.Scatter(
        x=_months, y=_it["ANY_CONVERSION_PCT"].tolist(),
        name="Admission rate %", yaxis="y1",
        mode="lines+markers",
        line=dict(color=C_BLUE, width=2),
        marker=dict(size=7, color=C_BLUE),
        hovertemplate="%{x}<br>Admission rate: %{y:.2f}%<extra></extra>",
    ))
    _fig.update_layout(
        height=300, margin=dict(l=0, r=0, t=36, b=8),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="top", y=1.12, x=0),
        yaxis=dict(title="Admission rate %", ticksuffix="%", side="left",
                   showgrid=True, gridcolor="#F0F4FA"),
        yaxis2=dict(title="OPD Visits", overlaying="y", side="right", showgrid=False),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(_fig, use_container_width=True)

# â”€â”€ Chart 2 â€” Admission Decision-to-Bed Time â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if not _c2b.empty:
    _c2b_weeks = _c2b["WEEK_START"].astype(str).str[:10].tolist()
    _c2b_p50   = _c2b["C2B_P50_HRS"].tolist()
    _c2b_ns    = _c2b["C2B_N"].tolist()
    _c2b_over  = _c2b["C2B_OVER_4H_N"].tolist()
    _fig_c2b = go.Figure()
    _fig_c2b.add_hline(y=4, line_dash="dot", line_color=C_RED, line_width=1,
                       annotation_text="4 hr target", annotation_position="top left",
                       annotation_font=dict(size=9, color=C_RED))
    _fig_c2b.add_trace(go.Scatter(
        x=_c2b_weeks, y=_c2b_p50,
        mode="lines+markers",
        line=dict(color=C_BLUE, width=2),
        marker=dict(size=7, color=C_BLUE),
        customdata=list(zip(_c2b_ns, _c2b_over)),
        hovertemplate="%{x}<br>P50: %{y} hrs<br>Timed: %{customdata[0]} Â· Over 4 hrs: %{customdata[1]}<extra></extra>",
    ))
    st.markdown('<div style="font-size:13px;font-weight:600;color:#003467;margin-bottom:4px">Admission Decision-to-Bed Time â€” 12-Week Trend</div>', unsafe_allow_html=True)
    _fig_c2b.update_layout(
        height=260, margin=dict(l=0, r=0, t=8, b=0),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        yaxis=dict(title="Hours", showgrid=True, gridcolor="#F0F4FA", rangemode="tozero"),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(_fig_c2b, use_container_width=True)
    st.caption("Time from last doctor note to bed assignment Â· Indicative.")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CHAPTER 2 â€” CAPACITY
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
if not _bw.empty:
    section_header(f"Capacity â€” {_cap_month}", margin_top=16)

    # Facility BOR card
    _bor_pct       = _get(_rbm, "BOR_PCT")
    _bor_prior_pct = _get(_rbm, "PRIOR_BOR_PCT")
    _bor_delta     = round(float(_bor_pct) - float(_bor_prior_pct), 1) if _bor_pct and _bor_prior_pct else None
    _bor_month_lbl = str(_get(_rbm, "MONTH"))[:7]       if _get(_rbm, "MONTH")       else ""
    _bor_prior_lbl = str(_get(_rbm, "PRIOR_MONTH"))[:7] if _get(_rbm, "PRIOR_MONTH") else "prior month"
    _, _bc, _, _ = st.columns(4)
    _card(_bc, "Bed Occupancy Rate",
        f"{_bor_pct}%" if _bor_pct else "â€”",
        sub=f"facility-wide Â· {_bor_month_lbl} Â· Indicative",
        delta=_delta_html(_bor_delta, "pp vs prior month", good="neutral"))

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    # â”€â”€ BOR by ward | ALOS by ward â€” side by side â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cap_l, cap_r = st.columns(2)

    _bw_sorted = _bw.sort_values("BOR_PCT", ascending=True)
    _bor_colors = [
        C_GREEN if v >= 60 else C_AMBER if v >= 50 else C_RED
        for v in _bw_sorted["BOR_PCT"]
    ]
    _fig_bor = go.Figure()
    _fig_bor.add_trace(go.Bar(
        x=_bw_sorted["BOR_PCT"].tolist(),
        y=_bw_sorted["WARD_NAME"].tolist(),
        orientation="h",
        marker_color=_bor_colors,
        text=[f"{v}%" for v in _bw_sorted["BOR_PCT"]],
        textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>BOR: %{x}%<extra></extra>",
    ))
    for _xr, _xl in [(50, "50%"), (60, "60%")]:
        _fig_bor.add_vline(x=_xr, line_dash="dot", line_color="#94A3B8", line_width=1,
                           annotation_text=_xl, annotation_position="top",
                           annotation_font_size=9, annotation_font_color="#94A3B8")
    cap_l.markdown('<div style="font-size:13px;font-weight:600;color:#003467;margin-bottom:4px">Bed Occupancy by Ward</div>', unsafe_allow_html=True)
    _fig_bor.update_layout(
        height=300, margin=dict(l=0, r=50, t=8, b=0),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        xaxis=dict(ticksuffix="%", range=[0, 110],
                   showgrid=True, gridcolor="#F0F4FA"),
        yaxis=dict(showgrid=False), showlegend=False,
    )
    cap_l.plotly_chart(_fig_bor, use_container_width=True)
    cap_l.caption("Green â‰¥ 60% Â· Amber 50â€“59% Â· Red < 50% Â· facility-calibrated thresholds.")

    if not _aw.empty:
        _aw_sorted = _aw.sort_values("ALOS_DAYS", ascending=True)
        _aw_colors = [
            C_RED if v > 7 else C_AMBER if v >= 4 else C_GREEN
            for v in _aw_sorted["ALOS_DAYS"]
        ]
        _fig_aw = go.Figure()
        _fig_aw.add_trace(go.Bar(
            x=_aw_sorted["ALOS_DAYS"].tolist(),
            y=_aw_sorted["WARD_NAME"].tolist(),
            orientation="h",
            marker_color=_aw_colors,
            text=[f"{v}d" for v in _aw_sorted["ALOS_DAYS"]],
            textposition="outside", cliponaxis=False,
            customdata=_aw_sorted["DISCHARGES"].tolist(),
            hovertemplate="<b>%{y}</b><br>ALOS: %{x}d<br>Discharges: %{customdata}<extra></extra>",
        ))
        cap_r.markdown('<div style="font-size:13px;font-weight:600;color:#003467;margin-bottom:4px">Avg Length of Stay by Ward</div>', unsafe_allow_html=True)
        _fig_aw.update_layout(
            height=300, margin=dict(l=0, r=50, t=8, b=0),
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            xaxis=dict(title="Days", showgrid=True, gridcolor="#F0F4FA"),
            yaxis=dict(showgrid=False), showlegend=False,
        )
        cap_r.plotly_chart(_fig_aw, use_container_width=True)
        cap_r.caption("Green < 4d Â· Amber 4â€“7d Â· Red > 7d Â· overnight discharges only Â· facility-calibrated thresholds.")

    # â”€â”€ Chart: Discharge Decision-to-Departure Time â€” weekly trend â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not _dt.empty:
        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
        _disch_weeks    = _dt["WEEK_START"].astype(str).str[:10].tolist()
        _disch_p50      = _dt["DISCH_P50_HRS"].tolist()
        _disch_ns       = _dt["DISCHARGE_N"].tolist()
        _disch_baseline = round(pd.Series(_disch_p50).median(), 1)
        _fig_dt = go.Figure()
        _fig_dt.add_hline(
            y=_disch_baseline, line_dash="dot", line_color="#94A3B8", line_width=1,
            annotation_text=f"12-wk baseline {_disch_baseline} hrs",
            annotation_position="top right",
            annotation_font=dict(size=9, color="#94A3B8"),
        )
        _fig_dt.add_trace(go.Scatter(
            x=_disch_weeks, y=_disch_p50,
            mode="lines+markers",
            line=dict(color=C_BLUE, width=2),
            marker=dict(size=7, color=C_BLUE),
            customdata=_disch_ns,
            hovertemplate="%{x}<br>P50: %{y} hrs<br>Admissions: %{customdata}<extra></extra>",
        ))
        st.markdown('<div style="font-size:13px;font-weight:600;color:#003467;margin-bottom:4px">Discharge Decision-to-Departure Time â€” 12-Week Trend</div>', unsafe_allow_html=True)
        _fig_dt.update_layout(
            height=260, margin=dict(l=0, r=0, t=8, b=0),
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            yaxis=dict(title="Hours", showgrid=True, gridcolor="#F0F4FA", rangemode="tozero"),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(_fig_dt, use_container_width=True)
        st.caption(
            "Weekly grain Â· monthly ward charts above. "
            "Time from discharge decision to patient departure Â· Indicative."
        )

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CHAPTER 3 â€” FINANCIAL IMPACT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
section_header("Financial Impact", margin_top=16)

st.markdown(
    '<div style="background:#FEF3C7;border-left:4px solid #F59E0B;border-radius:4px;'
    'padding:8px 12px;font-size:11px;color:#92400E;margin-bottom:12px">'
    'âš  Billed cash rates only â€” KSh 6,000 general wards Â· KSh 12,000 private wards. '
    'Not collected revenue. Ceiling figures. Registered bed count unconfirmed on site.</div>',
    unsafe_allow_html=True,
)

# â”€â”€ Chart 5 â€” Revenue per registered bed-day â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if not _bw.empty and "REVENUE_PER_BED_DAY" in _bw.columns:
    _bw_rev = _bw.sort_values("REVENUE_PER_BED_DAY", ascending=True)
    _fig_rev = go.Figure()
    _fig_rev.add_trace(go.Bar(
        x=_bw_rev["REVENUE_PER_BED_DAY"].tolist(),
        y=_bw_rev["WARD_NAME"].tolist(),
        orientation="h",
        marker_color=C_NAVY,
        text=[f"KSh {int(v):,}" for v in _bw_rev["REVENUE_PER_BED_DAY"]],
        textposition="outside", cliponaxis=False,
        customdata=list(zip(
            _bw_rev["BOR_PCT"].tolist(),
            [f"KSh {int(v):,}" for v in _bw_rev["OPPORTUNITY_COST_KSH"].tolist()],
        )),
        hovertemplate=(
            "<b>%{y}</b><br>Revenue/bed-day: %{text}<br>"
            "BOR: %{customdata[0]}%<br>Opportunity cost: %{customdata[1]}<extra></extra>"
        ),
    ))
    st.markdown(f'<div style="font-size:13px;font-weight:600;color:#003467;margin-bottom:4px">Revenue per Registered Bed-Day by Ward â€” {_cap_month}</div>', unsafe_allow_html=True)
    _fig_rev.update_layout(
        height=300, margin=dict(l=0, r=120, t=8, b=0),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        xaxis=dict(title="KSh", showgrid=True, gridcolor="#F0F4FA"),
        yaxis=dict(showgrid=False), showlegend=False,
    )
    st.plotly_chart(_fig_rev, use_container_width=True)
    st.caption("Billed ceiling rate per registered bed-day Â· hover for opportunity cost by ward.")

# â”€â”€ Chart 6 â€” Payer mix â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if not _bp.empty:
    _bp_month = str(_bp["MONTH_START"].iloc[0])[:7]
    _wtypes = ["general", "private"]
    _ins, _cash, _spare = [], [], []
    for _wt in _wtypes:
        _rows = _bp[_bp["WARD_TYPE"] == _wt]
        _cap  = int(_rows["CAPACITY_BED_DAYS"].iloc[0]) if not _rows.empty else 0
        _i    = int(_rows.loc[_rows["PAYMENT_MODE"] == "insurance", "BED_DAYS"].sum())
        _c    = int(_rows.loc[_rows["PAYMENT_MODE"] == "cash",      "BED_DAYS"].sum())
        _ins.append(_i); _cash.append(_c)
        _spare.append(max(0, _cap - _i - _c))

    _fig_p = go.Figure()
    _fig_p.add_trace(go.Bar(
        name="Insurance", x=_ins, y=["General", "Private"], orientation="h",
        marker_color="#0D9488",
        text=[f"{v:,}" for v in _ins], textposition="inside", insidetextanchor="middle",
    ))
    _fig_p.add_trace(go.Bar(
        name="Cash", x=_cash, y=["General", "Private"], orientation="h",
        marker_color=C_AMBER,
        text=[f"{v:,}" for v in _cash], textposition="inside", insidetextanchor="middle",
    ))
    _fig_p.add_trace(go.Bar(
        name="Unused capacity", x=_spare, y=["General", "Private"], orientation="h",
        marker_color="#E2E8F0", marker_line_color="#CBD5E1", marker_line_width=1,
        text=[f"{v:,}" for v in _spare], textposition="inside", insidetextanchor="middle",
        textfont=dict(color="#94A3B8"),
    ))
    st.markdown(f'<div style="font-size:13px;font-weight:600;color:#003467;margin-bottom:4px">Bed-Days by Payer Type â€” {_bp_month}</div>', unsafe_allow_html=True)
    _fig_p.update_layout(
        barmode="stack", height=220, margin=dict(l=0, r=0, t=48, b=0),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0),
        xaxis=dict(title="Occupied bed-days", showgrid=True, gridcolor="#F0F4FA"),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(_fig_p, use_container_width=True)
    st.caption(f"Bed-days by payer type Â· {_bp_month} Â· grey = unused registered capacity.")

