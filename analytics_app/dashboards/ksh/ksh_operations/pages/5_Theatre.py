"""
KSH V3 â€” Theatre
Source: STG_KSH_V3_THEATRE_CASES (silver, live) + THEATRE_BOOKINGS / EVALUATION_PROCEDURES (bronze).
Structure: Bottleneck Â· Driver Â· Consequence
Signal: Indicative â€” has_operation proxy; planned duration only.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from ksh_operations.dashboard.theme import apply_theme, render_sidebar, section_header, page_header
from ksh_operations.dashboard.queries import (
    q_theatre_elective_completion,
    q_theatre_emergency_rate,
    q_theatre_out_of_hours,
    q_theatre_value_exposure,
    q_theatre_completion_trend,
    q_theatre_procedure_nc,
    q_theatre_revenue_trend,
)

apply_theme()
render_sidebar(active="theatre")
page_header("Theatre", subtitle="Bottleneck Â· Driver Â· Consequence")

# â”€â”€ Load â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_c1  = q_theatre_elective_completion();  _c1.columns  = [c.upper() for c in _c1.columns]
_c2  = q_theatre_emergency_rate();       _c2.columns  = [c.upper() for c in _c2.columns]
_c3  = q_theatre_out_of_hours();         _c3.columns  = [c.upper() for c in _c3.columns]
_c4  = q_theatre_value_exposure();       _c4.columns  = [c.upper() for c in _c4.columns]
_tt  = q_theatre_completion_trend();     _tt.columns  = [c.upper() for c in _tt.columns]
_pnc = q_theatre_procedure_nc();         _pnc.columns = [c.upper() for c in _pnc.columns]
_rt  = q_theatre_revenue_trend();        _rt.columns  = [c.upper() for c in _rt.columns]

_r1 = _c1.iloc[0] if not _c1.empty else None
_r2 = _c2.iloc[0] if not _c2.empty else None
_r3 = _c3.iloc[0] if not _c3.empty else None
_r4 = _c4.iloc[0] if not _c4.empty else None


def _get(row, col):
    if row is None: return None
    v = row.get(col)
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v


def _delta_html(val, unit="", good="up"):
    if val is None: return ""
    if good == "neutral": color = "#6B8CAE"
    else: color = "#16A34A" if ((val > 0) if good == "up" else (val < 0)) else "#E11D48"
    arrow = "â–²" if val > 0 else "â–¼"
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


def _fmt_kes(v):
    if v is None: return "â€”"
    v = int(v)
    return f"KES {v/1_000_000:.2f}M" if v >= 1_000_000 else f"KES {v:,}"


# â”€â”€ Freshness â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_month_lbl = str(_get(_r1, "MONTH"))[:7] if _get(_r1, "MONTH") else ""
if _month_lbl:
    st.caption(f"Latest complete month: {_month_lbl} Â· Operating Theatre")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# BOTTLENECK â€” Elective Completion
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
section_header("Bottleneck â€” Elective Completion")

_rate     = _get(_r1, "COMPLETION_RATE_PCT")
_prior    = _get(_r1, "PRIOR_RATE_PCT")
_delta    = _get(_r1, "MOM_DELTA_PP")
_due      = _get(_r1, "ELECTIVE_DUE_COUNT")
_noncomp  = _get(_r1, "NON_COMPLETION_COUNT")
_delta_c1 = _delta_html(_delta, "pp vs prior month", good="up") if (_due or 0) >= 30 else (
    _delta_html(_delta, "pp vs prior month", good="neutral") if _delta is not None else ""
)

_emr_rate  = _get(_r2, "RATE_LATEST_PCT")
_emr_n     = _get(_r2, "EMERGENCY_LATEST")
_emr_total = _get(_r2, "CASES_LATEST")
_emr_6m    = _get(_r2, "RATE_6M_PCT")

_ooh      = _get(_r3, "OOH_LATEST")
_ooh_rate = _get(_r3, "RATE_LATEST_PCT")
_ooh_perf = _get(_r3, "PERFORMED_LATEST")
_ooh_6m   = _get(_r3, "RATE_6M_PCT")

_val        = _get(_r4, "NC_VALUE_LATEST")
_val_cases  = _get(_r4, "NC_CASES_LATEST")
_val_pct    = _get(_r4, "EXPOSURE_LATEST_PCT")
_val_6m     = _get(_r4, "NC_VALUE_6M")
_val_6m_pct = _get(_r4, "EXPOSURE_6M_PCT")

_col1, _col2, _col3, _col4 = st.columns(4)

_card(_col1, "Elective Completion Rate",
    f"{_rate}%" if _rate is not None else "â€”",
    sub=(f"{int(_noncomp):,} of {int(_due):,} not completed (no operation record) Â· "
         f"prior: {_prior}% Â· Indicative"),
    delta=_delta_c1)

_card(_col2, "Booked Emergency Rate",
    f"{_emr_rate}%" if _emr_rate is not None else "â€”",
    sub=(f"{int(_emr_n):,} of {int(_emr_total):,} cases Â· "
         f"6-mo avg: {_emr_6m}% Â· Descriptive"))

_card(_col3, "Booked Out-of-Hours",
    f"{int(_ooh):,}" if _ooh is not None else "â€”",
    sub=(f"{_ooh_rate}% of {int(_ooh_perf):,} performed Â· "
         f"6-mo avg: {_ooh_6m}% Â· Descriptive"))

_card(_col4, "Revenue Exposure",
    _fmt_kes(_val),
    sub=(f"{int(_val_cases):,} cases Â· {_val_pct}% of booked value Â· "
         f"6-mo: {_fmt_kes(_val_6m)} ({_val_6m_pct}%) Â· "
         f"Quoted at booking, not billed Â· Indicative")
    if _val_cases is not None else None)

st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)

# â”€â”€ 12-month completion rate trend â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if not _tt.empty:
    _months     = _tt["MONTH"].astype(str).str[:7].tolist()
    _comp_rates = [round(100 - r, 1) for r in _tt["NON_COMPLETION_RATE_PCT"].tolist()]

    st.markdown(
        '<div style="font-size:13px;font-weight:600;color:#003467;margin-bottom:4px">'
        'Elective Completion Rate â€” 12-Month Trend</div>',
        unsafe_allow_html=True,
    )
    _fig_tt = go.Figure()
    _fig_tt.add_trace(go.Scatter(
        x=_months, y=_comp_rates,
        mode="lines+markers",
        line=dict(color="#16A34A", width=2),
        marker=dict(size=7, color="#16A34A"),
        customdata=list(zip(_tt["ELECTIVE_DUE"].tolist(), _tt["NON_COMPLETIONS"].tolist())),
        hovertemplate="%{x}<br>Completion: %{y}%<br>%{customdata[1]} not completed of %{customdata[0]} cases<extra></extra>",
    ))
    _fig_tt.update_layout(
        height=260, margin=dict(l=0, r=0, t=8, b=0),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        yaxis=dict(title="Completion %", ticksuffix="%",
                   showgrid=True, gridcolor="#F0F4FA", rangemode="tozero"),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(_fig_tt, use_container_width=True)
    st.caption(
        "Elective cases only Â· "
        "Emergency demand (0â€“10% of bookings) does not explain non-completion â€” "
        "pressure is within the elective process itself Â· February 2026 (100%) may be an outlier"
    )

st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# DRIVER â€” Which Procedures Are Not Completing?
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
if not _pnc.empty:
    section_header("Driver â€” Which Procedures Are Not Completing?", margin_top=24)

    _pnc_names  = _pnc["PROCEDURE_NAME"].tolist()
    _pnc_nc     = _pnc["TIMES_NOT_COMPLETED"].tolist()
    _pnc_booked = _pnc["TIMES_BOOKED"].tolist()
    _pnc_rate   = _pnc["NC_RATE_PCT"].tolist()
    _pnc_val    = _pnc["NC_VALUE"].tolist()

    # colour by NC rate: red >50%, amber 25-49%, navy <25%
    _bar_colors = [
        "#E11D48" if r > 50 else "#F59E0B" if r >= 25 else "#003467"
        for r in _pnc_rate
    ]

    # dynamic caption signals â€” computed from data, not hardcoded
    _top_freq = _pnc.iloc[0]
    _top_rate = _pnc.loc[_pnc["NC_RATE_PCT"].idxmax()]
    _top_val  = _pnc.loc[_pnc["NC_VALUE"].idxmax()]

    st.markdown(
        '<div style="font-size:13px;font-weight:600;color:#003467;margin-bottom:4px">'
        'Elective Non-Completions by Procedure â€” Last 12 Months</div>',
        unsafe_allow_html=True,
    )
    _fig_pnc = go.Figure(go.Bar(
        x=_pnc_nc,
        y=_pnc_names,
        orientation="h",
        marker_color=_bar_colors,
        customdata=list(zip(_pnc_booked, _pnc_rate, _pnc_val)),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "%{x} not completed of %{customdata[0]} booked<br>"
            "NC rate: %{customdata[1]}%<br>"
            "Value: KES %{customdata[2]:,.0f}"
            "<extra></extra>"
        ),
    ))
    _fig_pnc.update_layout(
        height=max(260, len(_pnc_names) * 22),
        margin=dict(l=0, r=0, t=8, b=0),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        xaxis=dict(title="Cases not completed", showgrid=True, gridcolor="#F0F4FA", dtick=1),
        yaxis=dict(showgrid=False, autorange="reversed"),
    )
    st.plotly_chart(_fig_pnc, use_container_width=True)
    st.caption(
        f"Colour: red = >50% NC rate Â· amber = 25â€“49% Â· navy = <25% Â· "
        f"Most frequent: {_top_freq['PROCEDURE_NAME']} ({int(_top_freq['TIMES_NOT_COMPLETED'])} cases) Â· "
        f"Highest rate: {_top_rate['PROCEDURE_NAME']} ({int(_top_rate['NC_RATE_PCT'])}%) Â· "
        f"Highest value: {_top_val['PROCEDURE_NAME']} ({_fmt_kes(_top_val['NC_VALUE'])}) Â· Indicative"
    )

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CONSEQUENCE â€” Revenue Exposure
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
section_header("Consequence â€” Revenue Exposure", margin_top=24)

# â”€â”€ 12-month revenue exposure + NC rate trend â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if not _rt.empty:
    _rt_months  = _rt["MONTH"].astype(str).str[:7].tolist()
    _rt_values  = _rt["NC_VALUE"].tolist()
    _rt_nc      = _rt["NON_COMPLETIONS"].tolist()
    _rt_booked  = _rt["ELECTIVE_BOOKED"].tolist()
    _rt_nc_rate = [
        round(nc / b * 100, 1) if b else 0
        for nc, b in zip(_rt_nc, _rt_booked)
    ]

    st.markdown(
        '<div style="font-size:13px;font-weight:600;color:#003467;margin-bottom:4px">'
        'Revenue Exposure &amp; Non-Completion Rate â€” 12-Month Trend</div>',
        unsafe_allow_html=True,
    )
    _fig_rt = go.Figure()
    _fig_rt.add_trace(go.Bar(
        x=_rt_months, y=_rt_values,
        name="KES exposure",
        marker_color="#003467",
        hovertemplate="%{x}<br>KES %{y:,.0f}<extra></extra>",
    ))
    _fig_rt.add_trace(go.Scatter(
        x=_rt_months, y=_rt_nc_rate,
        name="NC rate %",
        mode="lines+markers",
        line=dict(color="#E11D48", width=2),
        marker=dict(size=7, color="#E11D48"),
        yaxis="y2",
        customdata=list(zip(_rt_nc, _rt_booked)),
        hovertemplate="%{x}<br>NC rate: %{y}%<br>%{customdata[0]} of %{customdata[1]} booked<extra></extra>",
    ))
    _fig_rt.update_layout(
        height=280, margin=dict(l=0, r=0, t=8, b=0),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0),
        yaxis=dict(title="KES exposure", showgrid=True, gridcolor="#F0F4FA",
                   tickformat=",.0f", rangemode="tozero"),
        yaxis2=dict(title="NC rate %", ticksuffix="%", overlaying="y",
                    side="right", showgrid=False, rangemode="tozero"),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(_fig_rt, use_container_width=True)
    st.caption("KES = booking price, not billed revenue Â· Indicative")

