"""
Care Pathway Completion — pages/8_dropoff.py
Source: mart_pathway_analysis + rpt_ortho_patient_journey · V2 only · Feb 2025 – present
Rule A mandatory: source_system = 'EMR_V2' for all queries.

Decision question: What prevents patients progressing through the care pathway?

Analytical chain:
  1. Scale       — How large is the problem? (KPI cards)
  2. Persistence — Is it getting better or worse? (monthly trend)
  3. Priority    — Which pathways contribute most? (service line + impact share)
  4. Pathway     — How are patients moving through the system? (Sankey)
  5. Mechanism   — Where in the pathway do patients exit? (ghost stage + dept hourly charts)
  6. Implication — What should operations do? (two distinct recommendations)

Methodological note:
  pathway_complete = 1 is the authoritative "received care" signal (mart_pathway_analysis).
  cons_ts IS NOT NULL measures ConsTime recording rate only — do not use as care proxy.
  True OPD incomplete rate: ~4–6% across the stable baseline (March 2025 – June 2026).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="Care Pathway Completion · SPH Ortho",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from facility_operations.dashboard.theme import (
    apply_theme, render_sidebar, page_header, COLORS, cl,
)
from facility_operations.dashboard.queries import (
    q_dropoff_kpis, q_dropoff_sankey_v2,
    q_dropoff_monthly_trend, q_dropoff_service_line,
    q_dropoff_volume_corr,
    q_dropoff_stage_responsibility, q_dropoff_dept_breakdown,
    q_dropoff_ghost_stage, q_dropoff_dept_hourly,
)

apply_theme()
render_sidebar("dropoff")


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


def _caption(text):
    st.markdown(
        f'<div style="font-size:9px;color:#9BAEC8;margin-top:-6px;margin-bottom:4px">{text}</div>',
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


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

kpi_df       = q_dropoff_kpis()
trend_df     = q_dropoff_monthly_trend()
svc_df       = q_dropoff_service_line()
sankey_df    = q_dropoff_sankey_v2()
corr_df      = q_dropoff_volume_corr()
stage_df     = q_dropoff_stage_responsibility()
dept_df      = q_dropoff_dept_breakdown()
ghost_df     = q_dropoff_ghost_stage()
dept_hr_df   = q_dropoff_dept_hourly()

# Derive service-line month label from query result — stays in sync with
# what the SQL actually returned, not a separate date calculation.
_svc_month_label = (pd.Timestamp.today() - pd.DateOffset(months=1)).strftime("%b %Y")
if not svc_df.empty:
    svc_df.columns = svc_df.columns.str.upper()
    _ref = svc_df.iloc[0].get("REF_MONTH")
    if _ref is not None and pd.notna(_ref):
        _svc_month_label = pd.to_datetime(_ref).strftime("%b %Y")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════

page_header(
    "Care Pathway Completion",
    period="Feb 2025 – present",
    mode="live",
)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — KPI CARDS: SCALE
# ══════════════════════════════════════════════════════════════════════════════

if not kpi_df.empty:
    _r       = kpi_df.iloc[0]
    _opd_inc = _safe_float(_r["OPD_INCOMPLETE_PCT"])
    _opd_n   = _safe_int(_r["OPD_V2_N"])
    _rcv     = _safe_float(_r["RECEIVED_CARE_PCT"])

    def _kpi(title, value, sub=None, sub_color=None):
        _sub_html = (
            f'<div style="font-size:12px;color:{sub_color or COLORS["muted"]};margin-top:6px">{sub}</div>'
            if sub else ""
        )
        st.markdown(
            f'<div style="background:#F8FBFE;border:1px solid #D6E4F0;border-radius:8px;'
            f'padding:28px 24px;text-align:center;min-height:200px;display:flex;'
            f'flex-direction:column;justify-content:center">'
            f'<div style="font-size:11px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
            f'letter-spacing:2px;margin-bottom:10px">{title}</div>'
            f'<div style="font-size:32px;font-weight:800;color:#003467;line-height:1.1">{value}</div>'
            f'{_sub_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

    _opd_col = COLORS["danger"] if _opd_inc >= 15 else (COLORS["warning"] if _opd_inc >= 8 else COLORS["success"])
    _rcv_col = COLORS["success"] if _rcv    >= 95 else (COLORS["warning"] if _rcv     >= 90 else COLORS["danger"])

    # Dept breakdown — top depts for post-registration exits (Walk-In excluded)
    _dept_rows = []
    if not dept_df.empty:
        dept_df.columns = dept_df.columns.str.upper()
        for _, _dr in dept_df.head(3).iterrows():
            _dept_rows.append({
                "name": str(_dr["DEPT"]).title(),
                "n":    int(_dr["N"]),
                "pct":  _safe_float(_dr["PCT"]),
            })

    k1, k2, k3 = st.columns(3)

    with k1:
        _kpi("OPD Without Recorded Care", f"{_opd_inc:.1f}%",
             sub=f"{_rcv:.1f}% received care", sub_color=_rcv_col)

    with k2:
        _mom_rendered = False
        if not trend_df.empty and len(trend_df) >= 3:
            _td_k = trend_df.copy()
            _td_k["VISIT_MONTH"] = pd.to_datetime(_td_k["VISIT_MONTH"])
            _td_k = _td_k.sort_values("VISIT_MONTH").reset_index(drop=True)
            # Use last complete month — anchored to same reference as service line query
            # (_svc_month_label derived from MAX(visit_date) with DAY<25 guard).
            # This correctly steps back when the latest data month is partial.
            _ref_complete = pd.to_datetime(_svc_month_label, format="%b %Y")
            _last_mo = _td_k.iloc[-1]["VISIT_MONTH"]
            if _last_mo > _ref_complete:
                _curr = _td_k.iloc[-2]
                _prev = _td_k.iloc[-3]
            else:
                _curr = _td_k.iloc[-1]
                _prev = _td_k.iloc[-2]
            _curr_pct = _safe_float(_curr["INCOMPLETE_PCT"])
            _prev_pct = _safe_float(_prev["INCOMPLETE_PCT"])
            _delta    = _curr_pct - _prev_pct
            _m_l      = _curr["VISIT_MONTH"].strftime("%b %Y")
            _m2_l     = _prev["VISIT_MONTH"].strftime("%b %Y")
            _arr  = "↓" if _delta < 0 else ("↑" if _delta > 0 else "→")
            _cc   = COLORS["success"] if _delta < 0 else (COLORS["danger"] if _delta > 0 else COLORS["muted"])
            st.markdown(
                f'<div style="background:#F8FBFE;border:1px solid #D6E4F0;border-radius:8px;'
                f'padding:28px 24px;text-align:center;min-height:200px;display:flex;'
                f'flex-direction:column;justify-content:center">'
                f'<div style="font-size:11px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
                f'letter-spacing:2px;margin-bottom:10px">{_m_l} Drop-off</div>'
                f'<div style="font-size:32px;font-weight:800;color:#003467;line-height:1.1">{_curr_pct:.1f}%</div>'
                f'<div style="font-size:12px;font-weight:700;color:{_cc};margin-top:6px">'
                f'{_arr} {abs(_delta):.1f}pp vs {_m2_l}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            _mom_rendered = True
        if not _mom_rendered:
            _kpi("Drop-off Rate", "—", "Insufficient monthly data")

    with k3:
        _dept_rows_html = ""
        for _i, _row in enumerate(_dept_rows):
            _bar_w = max(4, int(_row["pct"] * 0.9))
            _border = "border-bottom:1px solid #EBF3FB;" if _i < len(_dept_rows) - 1 else ""
            _dept_rows_html += (
                f'<div style="padding:8px 0;{_border}">'
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px">'
                f'<span style="font-size:12px;font-weight:600;color:#003467">{_row["name"]}</span>'
                f'<span style="font-size:12px;font-weight:700;color:#003467">{_row["pct"]:.0f}%'
                f'<span style="font-size:10px;font-weight:400;color:#9BAEC8"> · {_row["n"]:,}</span></span>'
                f'</div>'
                f'<div style="background:#EBF3FB;border-radius:3px;height:4px">'
                f'<div style="background:{COLORS["warning"]};border-radius:3px;height:4px;width:{_bar_w}%"></div>'
                f'</div>'
                f'</div>'
            )
        if not _dept_rows_html:
            _dept_rows_html = '<div style="font-size:12px;color:#9BAEC8">No data</div>'
        st.markdown(
            f'<div style="background:#F8FBFE;border:1px solid #D6E4F0;border-radius:8px;'
            f'padding:28px 24px;min-height:200px">'
            f'<div style="font-size:11px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
            f'letter-spacing:2px;margin-bottom:14px">Where Patients Leave</div>'
            f'{_dept_rows_html}'
            f'<div style="font-size:9px;color:#9BAEC8;margin-top:10px">'
            f'Share of all incomplete visits · Walk-In excluded · volume, not rate</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="margin-bottom:24px"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SANKEY — PATHWAY: HOW ARE PATIENTS LOST?
# ══════════════════════════════════════════════════════════════════════════════

_act("Pathway: How Are Patients Moving Through the System?")

if not sankey_df.empty:
    _sr     = sankey_df.iloc[0]
    _tot    = max(_safe_int(_sr["TOTAL"]), 1)
    _to_c   = _safe_int(_sr["TO_CONSULT"])
    _to_a   = _safe_int(_sr["TO_ANCILLARY"])
    _to_d   = _safe_int(_sr["TO_DROPOFF"])
    _c_thr  = _safe_int(_sr["CONSULT_THEATRE_DIRECT"])
    _c_adm  = _safe_int(_sr["CONSULT_ADMITTED"])
    _c_ph   = _safe_int(_sr["CONSULT_PHARMACY"])
    _c_opd  = _safe_int(_sr["CONSULT_OPD_EXIT"])
    _a_adm  = _safe_int(_sr["ANCILLARY_ADMITTED"])
    _a_exit = _safe_int(_sr["ANCILLARY_EXIT"])
    _ad_thr = _safe_int(_sr["ADMITTED_THEATRE"])
    _ad_dis = _safe_int(_sr["ADMITTED_DISCHARGE"])

    def _pp(n): return f"{100 * n / _tot:.1f}%"
    def _lbl(name, n): return f"{name}  {n:,}"

    _labels = [
        _lbl("All Arrivals",       _tot),
        _lbl("Consultation",       _to_c),
        _lbl("Ancillary Pathway",  _to_a),
        _lbl("No Care",            _to_d),
        _lbl("Pharmacy",           _c_ph),
        _lbl("Admitted",           _c_adm + _a_adm),
        _lbl("Theatre",            _c_thr + _ad_thr),
        _lbl("Ward Discharge",     _ad_dis),
        _lbl("OPD Discharge",      _c_opd + _c_ph + _a_exit),
    ]
    _node_x = [0.001, 0.32, 0.32, 0.32, 0.64, 0.64, 0.999, 0.999, 0.999]
    _node_y = [0.50,  0.14, 0.58, 0.92, 0.05, 0.40, 0.12,  0.55,  0.86 ]
    _node_colors = [
        "#0072CE", "#1E8449", "#1A6BA0", "#C0392B",
        "#0BB99F", "#E67E22", "#6C3483", "#117A65", "#27AE60",
    ]
    _hover_node = [
        f"<b>All Arrivals</b><br>{_tot:,} visits<extra></extra>",
        f"<b>Consultation</b><br>{_to_c:,} · {_pp(_to_c)}<br>ConsTime recorded<extra></extra>",
        f"<b>Ancillary Pathway</b><br>{_to_a:,} · {_pp(_to_a)}<br>Care delivered, no ConsTime (Physio / Procedures / CWC)<extra></extra>",
        f"<b>No Care</b><br>{_to_d:,} · {_pp(_to_d)}<br>pathway_complete = 0 — no care signal<extra></extra>",
        f"<b>Pharmacy</b><br>{_c_ph:,} · {_pp(_c_ph)}<br>Consultation + dispensed medication<extra></extra>",
        f"<b>Admitted</b><br>{_c_adm + _a_adm:,} · {_pp(_c_adm + _a_adm)}<br>Moved to inpatient<extra></extra>",
        f"<b>Theatre</b><br>{_c_thr + _ad_thr:,} · {_pp(_c_thr + _ad_thr)}<br>Surgical procedure<extra></extra>",
        f"<b>Ward Discharge</b><br>{_ad_dis:,}<br>Discharged after inpatient stay<extra></extra>",
        f"<b>OPD Discharge</b><br>{_c_opd + _c_ph + _a_exit:,}<br>Completed OPD — went home<extra></extra>",
    ]
    _edges = [
        (0, 1, _to_c,   "rgba(30,132,73,0.18)"),
        (0, 2, _to_a,   "rgba(26,107,160,0.18)"),
        (0, 3, _to_d,   "rgba(192,57,43,0.60)"),
        (1, 4, _c_ph,   "rgba(11,185,159,0.25)"),
        (1, 5, _c_adm,  "rgba(230,126,34,0.25)"),
        (1, 6, _c_thr,  "rgba(108,52,131,0.25)"),
        (1, 8, _c_opd,  "rgba(39,174,96,0.18)"),
        (2, 5, _a_adm,  "rgba(230,126,34,0.25)"),
        (2, 8, _a_exit, "rgba(39,174,96,0.18)"),
        (4, 8, _c_ph,   "rgba(39,174,96,0.18)"),
        (5, 6, _ad_thr, "rgba(108,52,131,0.25)"),
        (5, 7, _ad_dis, "rgba(17,122,101,0.22)"),
    ]
    _src, _tgt, _val, _link_colors = zip(*_edges)

    fig_sk = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(
            pad=20, thickness=22,
            line=dict(color="#FFFFFF", width=0.8),
            label=_labels, color=_node_colors,
            x=_node_x, y=_node_y,
            customdata=_hover_node,
            hovertemplate="%{customdata}",
        ),
        link=dict(
            source=list(_src), target=list(_tgt), value=list(_val),
            color=list(_link_colors),
            hovertemplate="%{source.label} → %{target.label}<br>%{value:,} visits<extra></extra>",
        ),
    ))
    fig_sk.update_layout(**cl(
        height=500,
        font=dict(size=13, color="#1E3A55", family="Inter, Arial, sans-serif"),
        margin=dict(l=10, r=10, t=16, b=10),
    ))
    st.plotly_chart(fig_sk, use_container_width=True)

    st.markdown(
        f'<div style="display:flex;gap:0;margin-top:4px;border:1px solid #D6E4F0;'
        f'border-radius:8px;overflow:hidden;font-size:12px">'
        f'<div style="flex:1;padding:12px 16px;background:#F0FBF8;border-right:1px solid #D6E4F0">'
        f'<div style="font-size:9px;font-weight:700;color:#0BB99F;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px">Consultation</div>'
        f'<div style="font-size:20px;font-weight:800;color:#003467">{_to_c:,}</div>'
        f'<div style="color:#6B8CAE;margin-top:2px">{_pp(_to_c)} of arrivals</div>'
        f'</div>'
        f'<div style="flex:1;padding:12px 16px;background:#F4F8FC;border-right:1px solid #D6E4F0">'
        f'<div style="font-size:9px;font-weight:700;color:#1A6BA0;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px">Ancillary Pathway</div>'
        f'<div style="font-size:20px;font-weight:800;color:#003467">{_to_a:,}</div>'
        f'<div style="color:#6B8CAE;margin-top:2px">{_pp(_to_a)} · Physio / Proc / CWC</div>'
        f'</div>'
        f'<div style="flex:1;padding:12px 16px;background:#F4F8FC;border-right:1px solid #D6E4F0">'
        f'<div style="font-size:9px;font-weight:700;color:#0BB99F;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px">+ Pharmacy</div>'
        f'<div style="font-size:20px;font-weight:800;color:#003467">{_c_ph:,}</div>'
        f'<div style="color:#6B8CAE;margin-top:2px">{_pp(_c_ph)} · consult + dispense</div>'
        f'</div>'
        f'<div style="flex:1;padding:12px 16px;background:#FEF9F0;border-right:1px solid #D6E4F0">'
        f'<div style="font-size:9px;font-weight:700;color:#E67E22;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px">Admitted</div>'
        f'<div style="font-size:20px;font-weight:800;color:#003467">{_c_adm + _a_adm:,}</div>'
        f'<div style="color:#6B8CAE;margin-top:2px">{_pp(_c_adm + _a_adm)} · moved to ward</div>'
        f'</div>'
        f'<div style="flex:1;padding:12px 16px;background:#FDF2F8">'
        f'<div style="font-size:9px;font-weight:700;color:#C0392B;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px">No Care</div>'
        f'<div style="font-size:20px;font-weight:800;color:#C0392B">{_to_d:,}</div>'
        f'<div style="color:#6B8CAE;margin-top:2px">{_pp(_to_d)} · zero care signal</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — MONTHLY TREND: PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

_act("Trend: Is the Problem Persisting?")

if not trend_df.empty:
    _td = trend_df.copy()
    _td["VISIT_MONTH"] = pd.to_datetime(_td["VISIT_MONTH"])
    _peak_idx = _td["INCOMPLETE_PCT"].idxmax()
    _peak_mo  = _td.loc[_peak_idx, "VISIT_MONTH"]
    _peak_pct = _td.loc[_peak_idx, "INCOMPLETE_PCT"]

    fig_tr = go.Figure()
    fig_tr.add_trace(go.Scatter(
        x=_td["VISIT_MONTH"].tolist(),
        y=_td["INCOMPLETE_PCT"].tolist(),
        fill="tozeroy",
        fillcolor="rgba(192,57,43,0.08)",
        line=dict(color=COLORS["danger"], width=2),
        mode="lines+markers",
        marker=dict(size=5, color=COLORS["danger"]),
        name="Without Recorded Care",
        hovertemplate="<b>%{x|%b %Y}</b><br>Without Recorded Care: %{y:.2f}%<extra></extra>",
    ))
    fig_tr.add_trace(go.Scatter(
        x=_td["VISIT_MONTH"].tolist(),
        y=_td["GHOST_PCT"].tolist(),
        line=dict(color="#9BAEC8", width=1.2, dash="dot"),
        mode="lines",
        name="Post-registration only",
        hovertemplate="<b>%{x|%b %Y}</b><br>Post-reg: %{y:.2f}%<extra></extra>",
    ))
    fig_tr.add_annotation(
        x=_peak_mo, y=_peak_pct,
        text=f"Peak {_peak_pct:.1f}%",
        showarrow=True, arrowhead=2, arrowcolor=COLORS["danger"],
        font=dict(size=10, color=COLORS["danger"]),
        bgcolor="white", bordercolor=COLORS["danger"], borderwidth=1,
        ay=-30,
    )
    fig_tr.update_layout(**cl(
        height=300,
        showlegend=True,
        legend=dict(orientation="h", x=0, y=1.12, font=dict(size=10)),
        xaxis=dict(gridcolor="#EBF3FB", tickformat="%b %y"),
        yaxis=dict(title="Incomplete Rate (%)", gridcolor="#EBF3FB", rangemode="tozero"),
        margin=dict(l=0, r=10, t=36, b=20),
    ))
    st.plotly_chart(fig_tr, use_container_width=True)
    _caption(
        "Incomplete = pathway_complete = 0 (authoritative). "
        "Dotted line = post-registration ghost sub-rate. "
        "ConsTime recording rate excluded — Issue 86 (Feb–May 2026) distorts it."
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SERVICE LINE + IMPACT SHARE: PRIORITISE WHERE TO FOCUS
# ══════════════════════════════════════════════════════════════════════════════

_act(f"Where: Which Pathways Contribute Most?  ·  {_svc_month_label}")

if not svc_df.empty:
    _sd = svc_df.copy()
    _sd = _sd[_sd["INCOMPLETE_PCT"].notna()].sort_values("INCOMPLETE_PCT", ascending=True)
    _total_incomplete = max(_sd["INCOMPLETE_N"].sum(), 1)

    _bar_colors = [
        COLORS["danger"]  if p >= 15 else
        COLORS["warning"] if p >= 8  else
        COLORS["primary"]
        for p in _sd["INCOMPLETE_PCT"]
    ]
    _share_list = [
        round(n / _total_incomplete * 100, 1) for n in _sd["INCOMPLETE_N"]
    ]
    _bar_text = [
        f"{p:.1f}% · {n:,} visits · {s:.0f}% of total incomplete"
        for p, n, s in zip(_sd["INCOMPLETE_PCT"], _sd["TOTAL_VISITS"], _share_list)
    ]

    fig_svc = go.Figure(go.Bar(
        x=_sd["INCOMPLETE_PCT"].tolist(),
        y=_sd["DEPT"].tolist(),
        orientation="h",
        marker_color=_bar_colors,
        text=[f"{p:.1f}%" for p in _sd["INCOMPLETE_PCT"]],
        textposition="outside",
        textfont=dict(size=10),
        customdata=list(zip(
            _sd["TOTAL_VISITS"].tolist(),
            _sd["INCOMPLETE_N"].tolist(),
            _share_list,
        )),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Incomplete rate: %{x:.1f}%<br>"
            "Total visits: %{customdata[0]:,}<br>"
            "Incomplete n: %{customdata[1]:,}<br>"
            "Share of all incomplete: %{customdata[2]:.1f}%<extra></extra>"
        ),
    ))
    _x_max = max(_sd["INCOMPLETE_PCT"].max(), 1)
    fig_svc.update_layout(**cl(
        height=max(260, len(_sd) * 38),
        showlegend=False,
        xaxis=dict(title="Incomplete Rate (%)", range=[0, _x_max * 1.4], gridcolor="#EBF3FB"),
        yaxis=dict(autorange="reversed", gridcolor="#EBF3FB"),
        margin=dict(l=0, r=80, t=10, b=20),
    ))
    st.plotly_chart(fig_svc, use_container_width=True)
    _caption(
        f"{_svc_month_label} · Incomplete rate within each pathway (% of that dept's visits). "
        "Red ≥15%, amber ≥8%. Min 10 visits. Hover for share of total incomplete care. "
        "A dept can lead by rate but not by volume if it handles fewer patients overall."
    )

    # ── Top 2 impact summary strip ────────────────────────────────────────────
    _top2 = svc_df.head(2).reset_index(drop=True)
    if not _top2.empty:
        _strip_parts = []
        _strip_colors = [("#C0392B", "#FDF2F8", "#F5B7B1"), ("#E67E22", "#FEF9F0", "#F0C580")]
        for _i, (_row, (_fc, _bg, _bc)) in enumerate(zip(
            _top2.itertuples(), _strip_colors
        )):
            _d_share = round(_row.INCOMPLETE_N / _total_incomplete * 100, 1)
            _strip_parts.append(
                f'<div style="flex:1;padding:14px 18px;background:{_bg};'
                f'border-right:1px solid #D6E4F0;border-left:3px solid {_bc}">'
                f'<div style="font-size:9px;font-weight:700;color:{_fc};text-transform:uppercase;'
                f'letter-spacing:1.5px;margin-bottom:4px">{_row.DEPT}</div>'
                f'<div style="display:flex;gap:20px;align-items:baseline">'
                f'<span style="font-size:22px;font-weight:800;color:#003467">{_row.INCOMPLETE_PCT:.1f}%</span>'
                f'<span style="font-size:12px;color:#6B8CAE">incomplete rate</span>'
                f'</div>'
                f'<div style="font-size:12px;color:#1E3A55;margin-top:4px">'
                f'{_row.INCOMPLETE_N:,} of {_row.TOTAL_VISITS:,} visits · '
                f'<b>{_d_share:.0f}% of all incomplete care this month</b></div>'
                f'</div>'
            )
        st.markdown(
            f'<div style="display:flex;gap:0;margin-top:8px;border:1px solid #D6E4F0;'
            f'border-radius:8px;overflow:hidden">{"".join(_strip_parts)}</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — WHERE IN THE PATHWAY: PRE-SERVICE EXIT MECHANISM
# ══════════════════════════════════════════════════════════════════════════════

_act("How: Where in the Pathway Do Patients Exit?")

# ── Finding card: 100% exit before any service ────────────────────────────────
st.markdown(
    '<div style="background:#F0F4FA;border:1px solid #D6E4F0;border-left:4px solid #003467;'
    'border-radius:6px;padding:16px 20px;margin-bottom:20px">'
    '<div style="font-size:9px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
    'letter-spacing:1.5px;margin-bottom:8px">Key finding — March 2025 to present</div>'
    '<div style="font-size:14px;color:#1E3A55;line-height:1.7">'
    'In <b>100% of incomplete visits</b>, no service was ordered before exit — no lab, '
    'imaging, or pharmacy request was placed. Patients are registered to their dept in the '
    'EMR but the clinical encounter is never opened. The dropout point is at '
    '<b>registration</b>, not mid-pathway.'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Ghost stage breakdown by dept ─────────────────────────────────────────────
if not ghost_df.empty:
    ghost_df.columns = ghost_df.columns.str.upper()
    _ghost_parts = []
    for _row in ghost_df.itertuples():
        _ghost_parts.append(
            f'<div style="flex:1;padding:14px 18px;border-right:1px solid #D6E4F0">'
            f'<div style="font-size:9px;font-weight:700;color:#003467;text-transform:uppercase;'
            f'letter-spacing:1.5px;margin-bottom:8px">{_row.DEPT}</div>'
            f'<div style="font-size:12px;color:#1E3A55;line-height:1.9">'
            f'<span style="font-weight:700;color:#C0392B">{_row.PURE_GHOST_PCT:.0f}%</span>'
            f' exited before triage<br>'
            f'<span style="font-weight:700;color:#E67E22">{_row.POST_TRIAGE_GHOST_PCT:.0f}%</span>'
            f' triaged but not seen<br>'
            f'<span style="color:#6B8CAE;font-size:11px">{_row.INCOMPLETE_N:,} incomplete visits'
            f' (Mar 2025+)</span>'
            f'</div>'
            f'</div>'
        )
    st.markdown(
        f'<div style="display:flex;gap:0;border:1px solid #D6E4F0;border-radius:8px;'
        f'overflow:hidden;margin-bottom:8px">{"".join(_ghost_parts)}</div>',
        unsafe_allow_html=True,
    )
    _caption("Baseline: March 2025 – present. Walk-In excluded (registration channel).")

# ── Hourly incomplete rate charts: Physio + Pharmacy ──────────────────────────
if not dept_hr_df.empty:
    dept_hr_df.columns = dept_hr_df.columns.str.upper()

    def _hourly_chart(dept_name, title, caption_text, hr_min=7, hr_max=22):
        _dh = dept_hr_df[
            (dept_hr_df["DEPT"] == dept_name) &
            (dept_hr_df["ARRIVAL_HOUR"] >= hr_min) &
            (dept_hr_df["ARRIVAL_HOUR"] <= hr_max)
        ].copy()
        if _dh.empty:
            return
        _dh["HOUR_LABEL"] = _dh["ARRIVAL_HOUR"].apply(lambda h: f"{h:02d}:00")
        _colors = [
            COLORS["danger"]  if p >= 10 else
            COLORS["warning"] if p >= 5  else
            COLORS["primary"]
            for p in _dh["INCOMPLETE_PCT"]
        ]
        _fig = go.Figure(go.Bar(
            x=_dh["HOUR_LABEL"].tolist(),
            y=_dh["INCOMPLETE_PCT"].tolist(),
            marker_color=_colors,
            text=[f"{p:.1f}%" for p in _dh["INCOMPLETE_PCT"]],
            textposition="outside",
            textfont=dict(size=9),
            customdata=_dh["TOTAL_N"].tolist(),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Incomplete rate: %{y:.1f}%<br>"
                "Total arrivals: %{customdata:,}<extra></extra>"
            ),
        ))
        _fig.update_layout(**cl(
            height=260,
            showlegend=False,
            xaxis=dict(title="Arrival hour", gridcolor="#EBF3FB"),
            yaxis=dict(title="Incomplete rate (%)", gridcolor="#EBF3FB", rangemode="tozero"),
            margin=dict(l=0, r=10, t=30, b=20),
        ))
        st.markdown(
            f'<div style="font-size:11px;font-weight:600;color:#1E3A55;margin:20px 0 4px 0">'
            f'{title}</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_fig, use_container_width=True)
        _caption(caption_text)

    _hourly_chart(
        "PHYSIOTHERAPY CLINIC",
        "Physiotherapy Clinic — Arrival Time as a Contributing Factor",
        "Incomplete rate rises through the day: 2% at 8am → 10% by 3pm → 12% by 6pm. "
        "Time of arrival correlates with incomplete rate, though other factors may also be in play. "
        "Red ≥10%, amber ≥5%.",
    )
    _hourly_chart(
        "PHARMACY",
        "Pharmacy — Arrival Time as a Contributing Factor",
        "Near-zero all day, then 11.9% (29 patients) at 8pm — consistent with reduced "
        "after-hours dispensing capacity, though other factors may also be in play. Red ≥10%, amber ≥5%.",
        hr_min=7, hr_max=22,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — OPERATIONAL IMPLICATION
# ══════════════════════════════════════════════════════════════════════════════

_act("Operational Implication")

st.markdown(
    '<div style="display:flex;gap:12px;flex-wrap:wrap">'

    # Block 1 — universal finding
    '<div style="flex:1;min-width:260px;background:#FEF9F0;border:1px solid #F0C580;'
    'border-left:4px solid #E67E22;border-radius:6px;padding:16px 18px">'
    '<div style="font-size:9px;font-weight:700;color:#E67E22;text-transform:uppercase;'
    'letter-spacing:1.5px;margin-bottom:8px">Registration handoff — all high-risk depts</div>'
    '<div style="font-size:12px;color:#1E3A55;line-height:1.8">'
    'Every incomplete visit exits before any service is ordered. The failure is at the '
    'registration-to-encounter step, not inside the clinical process. '
    '<b>Action:</b> confirm that every EMR registration triggers an encounter opening '
    'in the clinician\'s worklist. Unmatched registrations (arrived but no encounter '
    'opened within 30 min) should alert the coordinator.'
    '</div>'
    '</div>'

    # Block 2 — Physio scheduling
    '<div style="flex:1;min-width:260px;background:#FEF9F0;border:1px solid #F0C580;'
    'border-left:4px solid #E67E22;border-radius:6px;padding:16px 18px">'
    '<div style="font-size:9px;font-weight:700;color:#E67E22;text-transform:uppercase;'
    'letter-spacing:1.5px;margin-bottom:8px">Physiotherapy — afternoon scheduling cap</div>'
    '<div style="font-size:12px;color:#1E3A55;line-height:1.8">'
    'Incomplete rate doubles between 8am (2%) and 2pm (9%) and continues rising. '
    'Afternoon registrations are outpacing clinic capacity. '
    '<b>Action:</b> set a hard booking cap for arrival slots after 12:00 that matches '
    'measured throughput. Patients beyond capacity should be offered a scheduled '
    'return appointment rather than a same-day registration that goes unfulfilled.'
    '</div>'
    '</div>'

    '</div>',
    unsafe_allow_html=True,
)
