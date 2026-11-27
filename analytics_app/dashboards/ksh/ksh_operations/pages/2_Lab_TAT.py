"""
KSH V3 â€” Lab TAT Â· Detail Page
Gold only: RPT_KSH_V3_LAB_TAT + RPT_KSH_V3_LAB_UTILISATION.
LAB: Strong (88.2%). RADIOLOGY: 28% timestamp coverage â€” P50 indicative.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from ksh_operations.dashboard.theme import apply_theme, render_sidebar, section_header, page_header, cl, COLORS
from ksh_operations.dashboard.queries import q_lab_monthly, q_lab_utilisation, q_lab_utilisation_trend, q_lab_completion, q_lab_modality, q_xray_coverage_trend

apply_theme()
render_sidebar(active="lab")
page_header("Lab & Radiology", subtitle="Order-to-result turnaround Â· monthly grain")

_AX = dict(gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE"))
def _ax(title="", **kw): return {**_AX, "title": title, **kw}

C_BLUE  = COLORS["primary"]
C_NAVY  = COLORS["dark"]
C_TEAL  = COLORS["secondary"]

# â”€â”€ Load â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_lm = q_lab_monthly()
_lu = q_lab_utilisation()
_lt = q_lab_utilisation_trend()
_lc = q_lab_completion()
try:
    _lmod = q_lab_modality()
    _lmod.columns = [c.upper() for c in _lmod.columns]
except Exception:
    _lmod = pd.DataFrame()
try:
    _xray = q_xray_coverage_trend()
    _xray.columns = [c.upper() for c in _xray.columns]
except Exception:
    _xray = pd.DataFrame()
_lc.columns = [c.upper() for c in _lc.columns]
_lcr = _lc.iloc[0] if not _lc.empty else None
_lt.columns = [c.upper() for c in _lt.columns]
_lm.columns = [c.upper() for c in _lm.columns]
_lu.columns = [c.upper() for c in _lu.columns]
lab_m = _lm[(_lm["TEST_TYPE"] == "LAB")       & (_lm["IS_PARTIAL_MONTH"] == False)].copy()
rad_m = _lm[(_lm["TEST_TYPE"] == "RADIOLOGY") & (_lm["IS_PARTIAL_MONTH"] == False)].copy()

_l    = lab_m.iloc[-1] if not lab_m.empty  else None
_lp   = lab_m.iloc[-2] if len(lab_m) >= 2  else None
_r    = rad_m.iloc[-1] if not rad_m.empty  else None
_rp   = rad_m.iloc[-2] if len(rad_m) >= 2  else None
_u    = _lu.iloc[0]    if not _lu.empty    else None
_up   = _lu.iloc[1]    if len(_lu) >= 2    else None

def _get(row, col):
    if row is None: return None
    v = row.get(col)
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v

def _delta(cur, prior, col, rounding=0):
    c, p = _get(cur, col), _get(prior, col)
    if c is None or p is None: return None
    d = float(c) - float(p)
    return round(d, rounding) if rounding else int(round(d))

def _delta_html(val, unit="", good="down"):
    if val is None: return ""
    if good == "neutral":
        color = "#6B8CAE"
    else:
        color = "#16A34A" if ((val < 0) if good == "down" else (val > 0)) else "#E11D48"
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

def _sig(text):
    st.markdown(
        f'<div style="font-size:11px;color:#6B8CAE;margin-top:-6px;margin-bottom:14px">{text}</div>',
        unsafe_allow_html=True,
    )

def _fmt(v, suffix=""): return f"{v}{suffix}" if v is not None else "â€”"

# â”€â”€ ROW 1 â€” LAB Baseline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
section_header("Lab At a Glance", margin_top=8)
_sig("Lab coverage 88.2% Â· Radiology 28% Â· monthly grain Â· Strong unless noted")

c1, c2, c3, c4 = st.columns(4)
_card(c1, "Median TAT",
    _fmt(int(_get(_l, "ORDER_TO_PUBLISHED_P50")), " min") if _get(_l, "ORDER_TO_PUBLISHED_P50") else "â€”",
    "order â†’ result Â· latest month",
    delta=_delta_html(_delta(_l, _lp, "ORDER_TO_PUBLISHED_P50", rounding=0), unit=" min vs prior mo", good="down"))
_card(c2, "Lab Orders",
    f"{int(_get(_l, 'ORDER_COUNT')):,}" if _get(_l, "ORDER_COUNT") is not None else "â€”",
    "total orders Â· latest month",
    delta=_delta_html(_delta(_l, _lp, "ORDER_COUNT"), unit=" orders vs prior mo", good="neutral"))
_card(c3, "Within 60 Min",
    _fmt(_get(_l, "PCT_WITHIN_60MIN"), "%") if _get(_l, "PCT_WITHIN_60MIN") else "â€”",
    "resulted â‰¤60 min Â· latest month",
    delta=_delta_html(_delta(_l, _lp, "PCT_WITHIN_60MIN", rounding=1), unit="pp vs prior mo", good="up"))
_card(c4, "Over 1 Day",
    f"{int(_get(_l, 'OVER_1D_COUNT')):,}" if _get(_l, "OVER_1D_COUNT") is not None else "â€”",
    "results taking >24 hrs Â· latest month",
    delta=_delta_html(_delta(_l, _lp, "OVER_1D_COUNT"), unit=" orders vs prior mo", good="down"))

if _l is not None:
    _o   = _get(_l, "ORDER_COUNT")
    _o1d = _get(_l, "OVER_1D_COUNT")
    _p60 = _get(_l, "PCT_WITHIN_60MIN")
    if _o and _p60:
        _f_vals   = [int(_o)]
        _f_labels = ["Orders Placed"]
        if _o1d is not None:
            _f_vals.append(int(_o) - int(_o1d))
            _f_labels.append("Resulted â‰¤24 hrs")
        _f_vals.append(int(round(int(_o) * float(_p60) / 100)))
        _f_labels.append("Within 60 Min")
        fig_funnel = go.Figure(go.Funnel(
            y=_f_labels, x=_f_vals,
            textinfo="value+percent initial",
            marker_color=[C_NAVY, C_TEAL, "#16A34A"][:len(_f_vals)],
        ))
        fig_funnel.update_layout(**cl(
            height=180, margin=dict(l=20, r=20, t=10, b=10),
        ))
        _fc, _ = st.columns([2, 1])
        _fc.plotly_chart(fig_funnel, use_container_width=True)


# â”€â”€ Ordering rate â€” Lab & Radiology â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
section_header("Ordering Rate", margin_top=16)

if not _lt.empty:
    _lr = _lt.sort_values("MONTH").copy()
    _lr["MONTH_LABEL"] = _lr["MONTH"].astype(str).str[:7]
    _lr["LAB_RATE"] = (_lr["LAB_ORDERS"] / _lr["OPD_VISITS"] * 100).where(_lr["OPD_VISITS"] > 0).round(1)
    _lr["RAD_RATE"] = (_lr["RADIOLOGY_ORDERS"] / _lr["OPD_VISITS"] * 100).where(_lr["OPD_VISITS"] > 0).round(1)

    or1, or2 = st.columns(2)
    fig_lab_r = go.Figure(go.Bar(
        x=_lr["MONTH_LABEL"], y=_lr["LAB_RATE"],
        marker_color=C_NAVY,
        hovertemplate="%{x}<br>%{y} per 100 visits<extra></extra>",
    ))
    fig_lab_r.update_layout(**cl(
        height=240, margin=dict(l=50, r=20, t=10, b=50),
        yaxis=_ax("Orders per 100 visits"),
        xaxis=_ax(tickangle=-45, nticks=12),
        showlegend=False,
    ))
    or1.markdown("**Lab â€” orders per 100 OPD visits**")
    or1.plotly_chart(fig_lab_r, use_container_width=True)

    fig_rad_r = go.Figure(go.Bar(
        x=_lr["MONTH_LABEL"], y=_lr["RAD_RATE"],
        marker_color=C_TEAL,
        hovertemplate="%{x}<br>%{y} per 100 visits<extra></extra>",
    ))
    fig_rad_r.update_layout(**cl(
        height=240, margin=dict(l=50, r=20, t=10, b=50),
        yaxis=_ax("Orders per 100 visits"),
        xaxis=_ax(tickangle=-45, nticks=12),
        showlegend=False,
    ))
    or2.markdown("**Radiology â€” orders per 100 OPD visits**")
    or2.plotly_chart(fig_rad_r, use_container_width=True)
    st.caption("Rising = more orders per patient Â· falling = fewer Â· flat = stable utilisation Â· Indicative.")

# â”€â”€ Radiology â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
section_header("Radiology", margin_top=16)
zc1, zc2 = st.columns(2)
_card(zc1, "Radiology / 100 OPD",
    _fmt(_get(_u, "RAD_PER_100_OPD")) if _get(_u, "RAD_PER_100_OPD") is not None else "â€”",
    "imaging orders per 100 OPD visits Â· latest complete month Â· Strong",
    delta=_delta_html(_delta(_u, _up, "RAD_PER_100_OPD", rounding=1), unit=" vs prior mo", good="neutral"))
_card(zc2, "Radiology P50",
    _fmt(int(_get(_r, "ORDER_TO_PUBLISHED_P50")), " min") if _get(_r, "ORDER_TO_PUBLISHED_P50") else "â€”",
    "order â†’ result Â· ultrasound only Â· Indicative (28% coverage)",
    delta=_delta_html(_delta(_r, _rp, "ORDER_TO_PUBLISHED_P50", rounding=0), unit=" min vs prior mo", good="down"))

if not _xray.empty:
    _xray["MONTH_LABEL"] = _xray["MONTH"].astype(str).str[:7]
    _xray["GAP"] = _xray["ORDERS"] - _xray["RESULTED_ORDERS"]
    st.markdown("**X-Ray â€” Results Gap**")
    fig_xray = go.Figure(go.Bar(
        x=_xray["MONTH_LABEL"], y=_xray["GAP"],
        marker_color="#E11D48",
        hovertemplate="%{x}<br>%{y} orders without a result<extra></extra>",
    ))
    fig_xray.update_layout(**cl(
        height=240, margin=dict(l=50, r=20, t=10, b=50),
        yaxis=_ax("Orders without result"),
        xaxis=_ax(tickangle=-45, nticks=12),
        showlegend=False,
    ))
    st.plotly_chart(fig_xray, use_container_width=True)
    st.caption("X-ray orders placed with no result recorded Â· zero = full coverage Â· a rising bar = widening documentation gap")

if not _lmod.empty:
    st.markdown("**Imaging by Modality**")
    _months = sorted(_lmod["MONTH"].unique(), reverse=True)
    _cur  = _lmod[_lmod["MONTH"] == _months[0]].set_index("MODALITY")
    _prev = _lmod[_lmod["MONTH"] == _months[1]].set_index("MODALITY") if len(_months) > 1 else None

    def _cell(val, prev_val, fmt, good="neutral"):
        """Value + coloured MoM delta span for HTML table cell."""
        if prev_val is None or pd.isna(prev_val):
            return f"<td style='padding:6px 12px'>{fmt(val)}</td>"
        d = val - prev_val
        if d == 0:
            delta_html = ""
        else:
            up = d > 0
            if good == "neutral":   color = "#6B8CAE"
            elif good == "up":      color = "#16A34A" if up else "#E11D48"
            else:                   color = "#E11D48" if up else "#16A34A"
            arrow = "â–²" if up else "â–¼"
            delta_html = (f'<span style="font-size:10px;color:{color};margin-left:4px">'
                          f'{arrow}{abs(round(d, 1))}</span>')
        return f"<td style='padding:6px 12px'>{fmt(val)}{delta_html}</td>"

    _mod_label = str(_months[0])[:7]
    _th = "style='padding:6px 12px;text-align:left;font-size:11px;color:#6B8CAE;border-bottom:1px solid #EBF3FB'"
    html_rows = ""
    for mod in _cur.index:
        c = _cur.loc[mod]
        p = _prev.loc[mod] if (_prev is not None and mod in _prev.index) else None
        pv = lambda col: (float(p[col]) if not pd.isna(p[col]) else None) if p is not None else None
        def _null(v): return v is None or (isinstance(v, float) and pd.isna(v))
        tat = c["MEDIAN_TAT_MIN"]; pct = c["PCT_WITHIN_60_MIN"]
        html_rows += (
            f"<tr style='border-bottom:1px solid #F1F5F9'>"
            f"<td style='padding:6px 12px;font-weight:600'>{mod}</td>"
            + _cell(int(c["ORDERS"]), pv("ORDERS") and int(pv("ORDERS")), lambda v: f"{v:,}", "neutral")
            + (f"<td style='padding:6px 12px;color:#9CA3AF'>â€”</td>" if _null(tat)
               else _cell(int(tat), pv("MEDIAN_TAT_MIN") and int(pv("MEDIAN_TAT_MIN")), lambda v: f"{v} min", "down"))
            + (f"<td style='padding:6px 12px;color:#9CA3AF'>â€”</td>" if _null(pct)
               else _cell(float(pct), pv("PCT_WITHIN_60_MIN"), lambda v: f"{v}%", "up"))
            + "</tr>"
        )
    st.markdown(
        f'<table style="width:100%;border-collapse:collapse;font-size:13px">'
        f'<thead><tr>'
        f'<th {_th}>Modality</th><th {_th}>Orders</th>'
        f'<th {_th}>Median TAT</th><th {_th}>% â‰¤60 Min</th>'
        f'</tr></thead><tbody>{html_rows}</tbody></table>',
        unsafe_allow_html=True,
    )
    st.caption(f"{_mod_label} Â· X-Ray TAT shown as â€” (results not entered in system) Â· coloured arrows = MoM change Â· Indicative (28% radiology coverage)")

# â”€â”€ Data Quality â€” All Time â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
section_header("Data Quality â€” All Time", margin_top=28)
dq1, dq2 = st.columns(2)
_card(dq1, "Result Completion Rate",
    _fmt(_get(_lcr, "COMPLETION_RATE"), "%") if _get(_lcr, "COMPLETION_RATE") is not None else "â€”",
    f"{int(_get(_lcr, 'WITH_RESULT')):,} orders with a result on file Â· all-time Â· Strong" if _get(_lcr, "WITH_RESULT") else "all-time Â· Strong")
_card(dq2, "Orders Without Result",
    _fmt(_get(_lcr, "WITHOUT_RESULT_PCT"), "%") if _get(_lcr, "WITHOUT_RESULT_PCT") is not None else "â€”",
    f"{int(_get(_lcr, 'WITHOUT_RESULT')):,} orders never resulted Â· all-time Â· Strong" if _get(_lcr, "WITHOUT_RESULT") else "all-time Â· Strong")
st.caption("All-time counts since system go-live Â· a rising % without result indicates a data entry or workflow gap Â· not comparable to monthly metrics above.")

