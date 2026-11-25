"""
KSH V3 â€” Pharmacy
Source: RPT_KSH_V3_PHARMACY_TAT (gold, PHARMACY store).
TAT = prescription created â†’ dispensed. Batch entries (â‰¥1,440 min) excluded.
Signal: Fulfillment Strong Â· TAT Indicative.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from ksh_operations.dashboard.theme import apply_theme, render_sidebar, section_header, page_header, COLORS
from ksh_operations.dashboard.queries import q_pharmacy_homepage, q_pharmacy_trend, q_pharmacy_heatmap

apply_theme()
render_sidebar(active="pharmacy")
page_header("Pharmacy", subtitle="Access Â· Speed Â· Volume")

C_BLUE  = COLORS["primary"]
C_NAVY  = COLORS["dark"]
C_RED   = "#E11D48"
C_GREEN = "#16A34A"

# â”€â”€ Load â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_hd = q_pharmacy_homepage(); _hd.columns = [c.upper() for c in _hd.columns]
_tr = q_pharmacy_trend();    _tr.columns = [c.upper() for c in _tr.columns]
_hm = q_pharmacy_heatmap();  _hm.columns = [c.upper() for c in _hm.columns]
_r  = _hd.iloc[0] if not _hd.empty else None


def _get(col):
    if _r is None: return None
    v = _r.get(col)
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v


def _delta_html(val, unit="", good="up"):
    if val is None: return ""
    if good == "neutral": color = "#6B8CAE"
    else: color = C_GREEN if ((val > 0) if good == "up" else (val < 0)) else C_RED
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


# â”€â”€ Freshness â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_month_lbl = str(_get("MONTH"))[:7] if _get("MONTH") else ""
_prior_lbl = str(_get("PRIOR_MONTH"))[:7] if _get("PRIOR_MONTH") else "prior month"
if _month_lbl:
    st.caption(f"Latest complete month: {_month_lbl} Â· PHARMACY store Â· Prescription â†’ Dispensed TAT")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CARDS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
section_header("Access Â· Speed Â· Volume", margin_top=8)

_fulfil      = _get("FULFILLMENT_PCT")
_fulfil_mom  = _get("FULFILLMENT_MOM_PP")
_fulfil_pri  = _get("PRIOR_FULFILLMENT_PCT")
_dispensed   = _get("DISPENSED_COUNT")
_rx_count    = _get("PRESCRIPTION_COUNT")

_tat         = _get("TAT_P50")
_tat_p90     = _get("TAT_P90")
_tat_base    = _get("TAT_3MO_BASELINE")
_tat_chg     = _get("TAT_PCT_CHANGE")
_coverage    = _get("COVERAGE_N")

_w60         = _get("PCT_WITHIN_60MIN")
_avg_orders  = _get("AVG_ORDERS_12M")

_c1, _c2, _c3, _c4 = st.columns([1, 1, 1, 1])

_card(_c1, "Drug Lines Dispensed",
    f"{_fulfil}%" if _fulfil is not None else "â€”",
    sub=(f"{int(_dispensed):,} of {int(_rx_count):,} drug lines Â· "
         f"{_month_lbl} Â· prior: {_fulfil_pri}% Â· Strong"),
    delta=_delta_html(_fulfil_mom, "pp vs prior month", good="up"))

_card(_c2, "Median Dispensing Time",
    f"{int(_tat)} min" if _tat is not None else "â€”",
    sub=(f"P90: {int(_tat_p90)} min Â· 3-mo baseline: {_tat_base} min Â· "
         f"change: {_tat_chg:+.1f}% Â· n={int(_coverage):,} Â· {_month_lbl} Â· Indicative")
    if _tat_p90 is not None and _tat_base is not None else
         f"n={int(_coverage):,} Â· {_month_lbl} Â· Indicative")

_opd_visits  = _get("OPD_VISIT_COUNT")
_orders_pv   = round(_rx_count / _opd_visits, 1) if (_rx_count and _opd_visits) else None
_card(_c3, "Orders Per Visit",
    f"{_orders_pv}" if _orders_pv is not None else "â€”",
    sub=f"drug lines / OPD visit Â· {_month_lbl} Â· Indicative")

_card(_c4, "Avg Orders / Month",
    f"{int(_avg_orders):,}" if _avg_orders is not None else "â€”",
    sub=f"trailing 12 months Â· {_month_lbl} Â· Strong")

st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)

# â”€â”€ Fulfillment rate trend + OPD visit volume â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if not _tr.empty and "OPD_VISIT_COUNT" in _tr.columns:
    _months   = _tr["MONTH"].astype(str).str[:7].tolist()
    _visits   = _tr["OPD_VISIT_COUNT"].tolist()
    _fulfil_t = _tr["FULFILLMENT_PCT"].tolist()

    _fig = go.Figure()
    _fig.add_trace(go.Bar(
        x=_months, y=_visits,
        name="OPD visits",
        marker_color="#0F6E56",
        opacity=0.5,
        yaxis="y2",
        hovertemplate="%{x}<br>OPD visits: %{y:,}<extra></extra>",
    ))
    _fig.add_trace(go.Scatter(
        x=_months, y=_fulfil_t,
        name="Fulfillment rate",
        mode="lines+markers",
        line=dict(color=C_BLUE, width=2),
        marker=dict(size=8, color=C_BLUE, line=dict(color="white", width=1.5)),
        yaxis="y1",
        hovertemplate="%{x}<br>Fulfillment: %{y}%<extra></extra>",
    ))
    _fig.update_layout(
        title=dict(text="Fulfillment Rate vs OPD Visit Volume â€” 12-Month Trend",
                   font=dict(size=13, color=C_NAVY), x=0, xanchor="left"),
        height=260, margin=dict(l=0, r=0, t=36, b=0),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        yaxis=dict(title="Fulfillment %", ticksuffix="%",
                   showgrid=True, gridcolor="#F0F4FA", rangemode="tozero"),
        yaxis2=dict(title="OPD visits", overlaying="y", side="right",
                    showgrid=False, rangemode="tozero"),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        barmode="overlay",
    )
    st.plotly_chart(_fig, use_container_width=True)

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    # Chart 2 â€” Order volume vs median TAT
    _all_lines = _tr["ALL_LINES"].tolist()
    _tat_trend = _tr["TAT_P50"].tolist()

    _fig2 = go.Figure()
    _fig2.add_trace(go.Bar(
        x=_months, y=_all_lines,
        name="Order volume",
        marker_color="#0F6E56",
        opacity=0.5,
        yaxis="y2",
        hovertemplate="%{x}<br>Orders: %{y:,}<extra></extra>",
    ))
    _fig2.add_trace(go.Scatter(
        x=_months, y=_tat_trend,
        name="Median TAT",
        mode="lines+markers",
        line=dict(color=C_RED, width=2),
        marker=dict(size=8, color=C_RED, line=dict(color="white", width=1.5)),
        yaxis="y1",
        hovertemplate="%{x}<br>Median TAT: %{y} min<extra></extra>",
    ))
    _fig2.update_layout(
        title=dict(text="Monthly Order Volume vs Median Dispensing Time",
                   font=dict(size=13, color=C_NAVY), x=0, xanchor="left"),
        height=260, margin=dict(l=0, r=0, t=36, b=0),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        yaxis=dict(title="TAT P50 (min)", showgrid=True,
                   gridcolor="#F0F4FA", rangemode="tozero"),
        yaxis2=dict(title="Drug lines", overlaying="y", side="right",
                    showgrid=False, rangemode="tozero"),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        barmode="overlay",
    )
    st.plotly_chart(_fig2, use_container_width=True)

# â”€â”€ TAT heatmap â€” day Ã— hour â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if not _hm.empty:
    import numpy as np

    _days  = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    _hours = list(range(7, 24))

    # pivot into 7 Ã— 17 matrix; NaN where thin_cell or missing
    _z = np.full((7, 17), np.nan)
    _text = [[""] * 17 for _ in range(7)]
    for _, row in _hm.iterrows():
        r = int(row["DOW"]) - 1
        c = int(row["HR"]) - 7
        if not row["THIN_CELL"]:
            _z[r][c] = row["P50"]
        _text[r][c] = f"P50: {row['P50']} min<br>P90: {row['P90']} min<br>{row['PCT_WITHIN_60']}% within 60<br>n={int(row['LINES'])}"

    _fig_hm = go.Figure(go.Heatmap(
        z=_z,
        x=[f"{h:02d}:00" for h in _hours],
        y=_days,
        text=_text,
        hovertemplate="%{y} %{x}<br>%{text}<extra></extra>",
        colorscale="RdYlGn_r",
        zmid=40,
        colorbar=dict(title="TAT P50 (min)", thickness=14, len=0.8),
    ))
    _fig_hm.update_layout(
        title=dict(text="Median Dispensing Time by Day & Hour â€” Trailing 12 Months",
                   font=dict(size=13, color=C_NAVY), x=0, xanchor="left"),
        height=320, margin=dict(l=0, r=0, t=80, b=0),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(_fig_hm, use_container_width=True)
    st.caption("Grey = fewer than 30 observations Â· Red = slow Â· Green = fast")

