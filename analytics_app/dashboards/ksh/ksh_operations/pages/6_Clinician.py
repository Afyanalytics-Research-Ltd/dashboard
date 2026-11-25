"""
KSH V3 â€” Clinician Workload
Scope: workforce availability, load distribution, and the consequence of concentration
       on patient waiting time.
Signal: Indicative â€” OPD attributed to Doctor station events only.
        user_id only â€” no name join.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from ksh_operations.dashboard.theme import apply_theme, render_sidebar, section_header, page_header
from ksh_operations.dashboard.queries import (
    q_clinician_homepage,
    q_clinician_weekly,
    q_clinician_concentration_wait,
)

apply_theme()
render_sidebar(active="clinician")
page_header("Clinician Workload")

# â”€â”€ Load â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_ch  = q_clinician_homepage();            _ch.columns  = [c.upper() for c in _ch.columns]
_cw  = q_clinician_weekly();              _cw.columns  = [c.upper() for c in _cw.columns]
_cct = q_clinician_concentration_wait();  _cct.columns = [c.upper() for c in _cct.columns]

_rh             = _ch.iloc[0] if not _ch.empty else None
_cw_latest_week = str(_cw.iloc[0]["LATEST_WEEK"])[:10] if not _cw.empty else ""
_cw_as_of       = str(_cw.iloc[0]["DATA_AS_OF"])[:10]  if not _cw.empty else ""


def _get(row, col):
    if row is None: return None
    v = row.get(col)
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v


def _card(col, label, value, sub=None):
    sub_html = (f'<div style="font-size:10px;color:#6B8CAE;margin-top:5px;line-height:1.4">{sub}</div>'
                if sub else "")
    col.markdown(
        f'<div style="background:#FFFFFF;border:1px solid #0072CE;border-left:4px solid #0072CE;'
        f'border-radius:8px;padding:14px 16px;text-align:center;min-height:100px">'
        f'<div style="font-size:9px;font-weight:700;color:#0072CE;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:6px">{label}</div>'
        f'<div style="font-size:28px;font-weight:800;color:#003467">{value}</div>'
        f'{sub_html}</div>',
        unsafe_allow_html=True,
    )


# â”€â”€ Freshness â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if _cw_latest_week:
    _as_of_note = f" Â· Data to {_cw_as_of} (partial week)" if _cw_as_of and _cw_as_of != _cw_latest_week else ""
    st.caption(f"Week of {_cw_latest_week}{_as_of_note}")

# â”€â”€ Summary stat line â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_active  = _get(_rh, "ACTIVE_CLINICIANS")
_opd_tot = _get(_rh, "TOTAL_OPD_VISITS")
_th_clin = _get(_rh, "THEATRE_ACTIVE_CLINICIANS")
_conc    = _get(_rh, "TOP3_CONCENTRATION_PCT")

if _active and _opd_tot:
    _avg_load = round(_opd_tot / _active, 1)
    _parts = [f"{int(_active)} clinicians active", f"{int(_opd_tot):,} OPD visits",
              f"{_avg_load} avg visits/clinician"]
    if _th_clin:
        _parts.append(f"{int(_th_clin)} in theatre this week")
    st.markdown(
        f'<div style="font-size:13px;color:#003467;margin-bottom:12px">'
        f'{"  Â·  ".join(_parts)}</div>',
        unsafe_allow_html=True,
    )

# â”€â”€ Cards â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_c1, _c2, _c3, _c4 = st.columns(4)
_card(_c1, "Active Clinicians",    f"{int(_active):,}"  if _active  else "â€”")
_card(_c2, "OPD Visits",           f"{int(_opd_tot):,}" if _opd_tot else "â€”")

_avg_load_card = round(_opd_tot / _active, 1) if (_active and _opd_tot) else None
_card(_c3, "Avg Visits / Clinician", f"{_avg_load_card}" if _avg_load_card else "â€”")

if _conc is not None:
    _conc_color = "#E11D48" if float(_conc) > 75 else "#F59E0B" if float(_conc) > 60 else "#16A34A"
    _conc_val   = f'<span style="color:{_conc_color}">{_conc}%</span>'
else:
    _conc_val = "â€”"
_card(_c4, "Top-3 Concentration", _conc_val, sub="of OPD visits handled by 3 doctors")

st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# LOAD DISTRIBUTION â€” who is carrying what this week
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
if not _cw.empty:
    _has_opd = _cw[_cw["OPD_VISITS_SEEN"] > 0].sort_values("OPD_VISITS_SEEN", ascending=True)

    if not _has_opd.empty:
        section_header(f"Load Distribution â€” Week of {_cw_latest_week}", margin_top=0)

        _fig_opd = go.Figure(go.Bar(
            x=_has_opd["OPD_VISITS_SEEN"].tolist(),
            y=_has_opd["USER_ID"].astype(str).tolist(),
            orientation="h",
            marker_color="#003467",
            customdata=list(zip(
                _has_opd["OPD_VISITS_PER_OPD_DAY"].tolist(),
                _has_opd["AVG_STATION_WAIT_MIN"].tolist(),
                _has_opd["DAYS_ACTIVE"].tolist(),
            )),
            hovertemplate=(
                "<b>User %{y}</b><br>"
                "Visits this week: %{x}<br>"
                "Visits per day: %{customdata[0]:.1f}<br>"
                "Avg patient wait: %{customdata[1]:.0f} min<br>"
                "Days active: %{customdata[2]}"
                "<extra></extra>"
            ),
        ))
        _fig_opd.update_layout(
            height=max(240, len(_has_opd) * 26),
            margin=dict(l=0, r=0, t=8, b=0),
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            xaxis=dict(title="OPD visits", showgrid=True, gridcolor="#F0F4FA", dtick=5),
            yaxis=dict(showgrid=False, type="category"),
        )
        st.plotly_chart(_fig_opd, use_container_width=True)
        st.caption("Hover for visits/day and avg patient wait Â· user_id only Â· Indicative")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CONSEQUENCE â€” does concentration drive patient waiting time?
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
if not _cct.empty:
    section_header("Does Concentration Drive Waiting Time? â€” 12 Weeks", margin_top=24)

    _cct["TOP3_CONCENTRATION_PCT"] = _cct["TOP3_CONCENTRATION_PCT"].astype(float)
    _cct["AVG_WAIT_MIN"]           = _cct["AVG_WAIT_MIN"].astype(float)

    _weeks_c   = _cct["WEEK_START"].astype(str).str[:10].tolist()
    _conc_w    = _cct["TOP3_CONCENTRATION_PCT"].tolist()
    _wait_w    = _cct["AVG_WAIT_MIN"].tolist()
    _visits_w  = _cct["TOTAL_OPD_VISITS"].tolist()
    _active_cw = _cct["ACTIVE_CLINICIANS"].tolist()

    _fig_cq = go.Figure()
    _fig_cq.add_trace(go.Bar(
        x=_weeks_c, y=_conc_w,
        name="Top-3 concentration %",
        marker_color="#003467",
        hovertemplate="%{x}<br>Concentration: %{y}%<extra></extra>",
    ))
    _fig_cq.add_trace(go.Scatter(
        x=_weeks_c, y=_wait_w,
        name="Avg patient wait (min)",
        mode="lines+markers",
        line=dict(color="#E11D48", width=2),
        marker=dict(size=7, color="#E11D48"),
        yaxis="y2",
        customdata=list(zip(_visits_w, _active_cw)),
        hovertemplate=(
            "%{x}<br>"
            "Avg wait: %{y} min<br>"
            "OPD visits: %{customdata[0]}<br>"
            "Active clinicians: %{customdata[1]}"
            "<extra></extra>"
        ),
    ))
    _fig_cq.update_layout(
        height=300, margin=dict(l=0, r=0, t=8, b=0),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0),
        yaxis=dict(title="Concentration %", ticksuffix="%",
                   showgrid=True, gridcolor="#F0F4FA", rangemode="tozero"),
        yaxis2=dict(title="Avg wait (min)", overlaying="y", side="right",
                    showgrid=False, rangemode="tozero"),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(_fig_cq, use_container_width=True)
    st.caption(
        "If the red line rises when bars are tall, fewer doctors are absorbing more demand "
        "and patients wait longer Â· Weighted by visit volume Â· Indicative"
    )

