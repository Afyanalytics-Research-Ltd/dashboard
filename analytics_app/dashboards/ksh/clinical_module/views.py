"""
views.py — Afya Clinical Analytics
=====================================
Render functions only. All SQL lives in queries.py.
Each function calls queries.load_* and builds charts.

  render_tab1_operations
  render_tab2_segmentation
  render_tab3_retention
  render_tab4_disease_burden
  render_tab5_workload
  render_clinician_view
"""

import os as _os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as _stcomp

import ksh.clinical_module.queries as Q

_PATIENT_LIST_COMPONENT = _stcomp.declare_component(
    "patient_list",
    path=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "patient_list_component"),
)
from ksh.clinical_module.ui_template import AFYA_BLUE, TEAL, COOL_BLUE, ORANGE, CORAL, PURPLE, GRAY, MUTED, BG_LIGHT, BORDER, GREEN
from ksh.clinical_module.charts import (
    line_chart, bar_chart, hbar_chart, stacked_bar, stacked_area,
    funnel_chart, heatmap, scatter, donut, table_fig, bullet, sparkline,
    BURDEN_COLORS, LIFECYCLE_COLORS,
)


# ─── SHARED HELPERS ───────────────────────────────────────────────────────────

def _H():
    return st.session_state.get("helpers", {})

def _gap(px=10):   _H().get("gap",      lambda px=10: None)(px)
def _sh(t, mt=0):  _H().get("sh",       lambda t, mt=0: None)(t, mt)
def _kpi(l, v, s="", color=AFYA_BLUE):
    _H().get("kpi_card", lambda l, v, s="", c=AFYA_BLUE: None)(l, v, sub=s, color=color)
def _pc(fig):      _H().get("pc",       lambda f: st.plotly_chart(f, use_container_width=True))(fig)
def _note(t, w=False): _H().get("note", lambda t,warn=False: None)(t, w)
def _n(v):
    return _H().get("fmt_num", lambda v: str(v))(v)

def _p(v, d=1):
    return _H().get("fmt_pct", lambda v, d=1: str(v))(v, d)

def _k(v):
    return _H().get("fmt_kes", lambda v: str(v))(v)


# ══════════════════════════════════════════════════════════════════════════════
# WAIT-TIME CHART HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _render_wait_time_chart(df: pd.DataFrame):
    """Two-panel P90 wait-time chart — pure HTML/CSS, no external dependencies."""
    import math as _m

    def _fh(hrs):
        """Format decimal hours → '2 hr 15 min' plain string."""
        if hrs is None: return "—"
        try: hrs = float(hrs)
        except: return "—"
        if _m.isnan(hrs): return "—"
        mins = round(hrs * 60)
        h, m = divmod(mins, 60)
        if h == 0: return f"{m} min"
        if m == 0: return f"{h} hr"
        return f"{h} hr {m} min"

    def _safe(row, col):
        v = row.get(col)
        if v is None: return None
        try:
            f = float(v)
            return None if _m.isnan(f) else f
        except: return None

    def _classify(p90, tgt):
        if p90 is None or tgt is None: return None
        if p90 <= tgt: return "within"
        if p90 <= tgt * 1.5: return "warn"
        return "breach"

    S_COL  = {"within": "#1D9E75", "warn": "#EF9F27", "breach": "#E24B4A"}
    S_BDGE = {"within": "Within target", "warn": "Approaching limit", "breach": "Breaching target"}
    S_BG   = {"within": "#E1F5EE", "warn": "#FEF3C7", "breach": "#FEE2E2"}
    S_TXT  = {"within": "#0F6E56", "warn": "#854F0B", "breach": "#A32D2D"}

    def _desc(vtype, p90, tgt, status):
        vt = "inpatients" if vtype == "Inpatient" else "outpatients"
        p90f, tgtf = _fh(p90), _fh(tgt)
        if status == "breach":
            x = round(float(p90) / float(tgt), 1)
            return (f"Most {vt} move through this stage without significant delay. "
                    f"But 1 in 10 waited more than {p90f} — {x}× longer than the {tgtf} target.")
        if status == "warn":
            return (f"Most {vt} are seen within the target window. "
                    f"But 1 in 10 waited over {p90f} — the tail is beginning to stretch.")
        return f"9 in 10 {vt} passed through this stage within {p90f}."

    def _exc_line(pct, tgt, vtype):
        if pct is None or tgt is None: return ""
        vt = "inpatients" if vtype == "Inpatient" else "outpatients"
        pf = float(pct)
        c = "#A32D2D" if pf > 20 else ("#854F0B" if pf >= 5 else "#0F6E56")
        return (f'<div style="font-size:11px;margin-top:3px;">'
                f'<span style="color:{c};font-weight:600;">'
                f'{pf:.1f}% of {vt} waited longer than the {_fh(tgt)} target</span></div>')

    def _gap_row(name, is_last):
        bb = "" if is_last else "border-bottom:0.5px solid rgba(0,0,0,0.06);"
        return (f'<div style="padding:10px 0;{bb}">'
                f'<div style="border:0.5px dashed #d1d5db;border-radius:8px;padding:8px 10px;">'
                f'<div style="font-style:italic;font-size:11px;color:#9ca3af;">{name}</div>'
                f'<div style="font-size:10px;color:#9ca3af;margin-top:3px;">'
                f'Timestamps not recorded in system — this stage cannot currently be measured'
                f'</div></div></div>')

    def _bar_row(name, p90, tgt, status, pct_exc, vtype, x_max, is_last):
        bb   = "" if is_last else "border-bottom:0.5px solid rgba(0,0,0,0.06);"
        bar_w = min(100, round(float(p90) / x_max * 100, 1))
        col   = S_COL.get(status, "#6b7280") if status else "#6b7280"
        tick  = ""
        if tgt is not None:
            tp = min(98, round(float(tgt) / x_max * 100, 1))
            tick = (f'<div style="position:absolute;left:{tp}%;top:-5px;bottom:-5px;'
                    f'width:2px;background:#7F77DD;border-radius:1px;z-index:2;"></div>'
                    f'<div style="position:absolute;left:{tp}%;top:-18px;'
                    f'transform:translateX(-50%);font-size:9px;color:#534AB7;'
                    f'white-space:nowrap;z-index:2;">{_fh(tgt)}</div>')
        badge = ""
        if status:
            badge = (f'<span style="font-size:10px;padding:2px 7px;border-radius:20px;'
                     f'background:{S_BG[status]};color:{S_TXT[status]};font-weight:600;">'
                     f'{S_BDGE[status]}</span>')
        desc_html = (f'<div style="font-size:11px;color:#6b7280;margin-top:4px;">'
                     f'{_desc(vtype, p90, tgt, status)}</div>') if status else ""
        exc_html  = _exc_line(pct_exc, tgt, vtype) if pct_exc is not None else ""
        return (
            f'<div style="padding:10px 0;{bb}">'
            f'  <div style="display:flex;justify-content:space-between;align-items:center;">'
            f'    <span style="font-size:12px;font-weight:500;color:#111827;">{name}</span>'
            f'    <div style="display:flex;align-items:center;gap:6px;">'
            f'      <span style="font-size:12px;font-weight:600;color:#111827;">{_fh(p90)}</span>'
            f'      {badge}</div></div>'
            f'  <div style="position:relative;height:10px;background:#f5f5f3;'
            f'border-radius:4px;margin:16px 0 6px;">'
            f'    <div style="width:{bar_w}%;height:100%;background:{col};border-radius:4px;"></div>'
            f'    {tick}</div>'
            f'  {desc_html}{exc_html}'
            f'</div>'
        )

    def _panel(row, vtype):
        tr_tgt = 0.5 if vtype == "Inpatient" else 0.25
        co_tgt = 1.0
        total  = _safe(row, "total_visits")
        total_s = f"{int(total):,}" if total is not None else "—"

        p90_tr  = _safe(row, "p90_hrs_to_triage")
        p90_co  = _safe(row, "p90_hrs_triage_to_consult")
        p90_lab = _safe(row, "p90_hrs_consult_to_lab")
        ptr_rec = _safe(row, "pct_triage_recorded")  or 0
        pco_rec = _safe(row, "pct_consult_recorded") or 0
        plab_rec= _safe(row, "pct_lab_recorded")     or 0
        pexc_tr = _safe(row, "pct_exceeding_triage_target")
        pexc_co = _safe(row, "pct_exceeding_consult_target")

        gap_tr  = ptr_rec  < 50
        gap_co  = pco_rec  < 50
        gap_lab = plab_rec < 50 or p90_lab is None

        s_tr = None if gap_tr  else _classify(p90_tr,  tr_tgt)
        s_co = None if gap_co  else _classify(p90_co,  co_tgt)

        p90s = [v for v in [p90_tr, p90_co] + ([p90_lab] if not gap_lab else []) if v]
        x_max = max(_m.ceil(max(p90s)) * 1.1, 0.5) if p90s else 2.0
        xd    = _m.ceil(x_max)
        scale = (f"Scale: 0–{xd} hr · {total_s} visits"
                 if vtype == "Inpatient"
                 else f"Scale: 0–{xd} hr · independent scale · {total_s} visits")

        # Verdict
        statuses = [s for s in [s_tr, s_co] if s is not None]
        vt_l = "inpatients" if vtype == "Inpatient" else "outpatients"
        if "breach" in statuses:
            ws = "Arrival → Triage" if s_tr == "breach" else "Triage → Consultation"
            wp = pexc_tr if s_tr == "breach" else pexc_co
            wps = f"{float(wp):.0f}%" if wp is not None else "a significant share"
            vbg, vbd = "#FFF0F0", "#E24B4A"
            vtx = (f'<strong>Action needed.</strong> {ws} is the critical failure point — '
                   f'{wps} of {vt_l} are waiting beyond acceptable limits.')
        elif "warn" in statuses:
            ws = "Arrival → Triage" if s_tr == "warn" else "Triage → Consultation"
            wp = pexc_tr if s_tr == "warn" else pexc_co
            wps = f"{float(wp):.0f}%" if wp is not None else "some"
            vbg, vbd = "#FAEEDA", "#EF9F27"
            vtx = (f'<strong>Watch closely.</strong> {ws} is approaching its limit — '
                   f'{wps} of {vt_l} are waiting longer than the target.')
        elif statuses:
            vbg, vbd = "#E1F5EE", "#1D9E75"
            vtx = '<strong>On track.</strong> All stages are currently within target wait times.'
        else:
            vbg, vbd = "#f8fafc", "#d1d5db"
            vtx = 'Insufficient timestamp data to assess wait-time status.'

        rows = ""
        rows += (_gap_row("Arrival → Triage", False)
                 if gap_tr  else _bar_row("Arrival → Triage",       p90_tr,  tr_tgt, s_tr,  pexc_tr, vtype, x_max, False))
        rows += (_gap_row("Triage → Consultation", False)
                 if gap_co  else _bar_row("Triage → Consultation",  p90_co,  co_tgt, s_co,  pexc_co, vtype, x_max, False))
        rows += (_gap_row("Consult → Lab Result", False)
                 if gap_lab else _bar_row("Consult → Lab Result",   p90_lab, None,   None,  None,    vtype, x_max, False))
        rows += _gap_row("Lab Result → Discharge", True)

        return (
            f'<div style="flex:1;background:#fff;border:0.5px solid rgba(0,0,0,0.12);'
            f'border-radius:12px;padding:16px;min-width:0;">'
            f'  <div style="margin-bottom:12px;">'
            f'    <div style="font-size:14px;font-weight:600;color:#111827;">{vtype}</div>'
            f'    <div style="font-size:10px;color:#9ca3af;margin-top:2px;">{scale}</div>'
            f'  </div>'
            f'  <div style="padding:8px 10px;background:{vbg};border-left:3px solid {vbd};'
            f'border-radius:0 6px 6px 0;font-size:11px;color:#374151;margin-bottom:12px;">'
            f'    {vtx}</div>'
            f'  {rows}'
            f'</div>'
        )

    # ── Assemble panels ───────────────────────────────────────────────────
    panels, gap_names = [], []
    for vtype in ["Inpatient", "Outpatient"]:
        sub = df[df["visit_type"] == vtype]
        if sub.empty: continue
        row = sub.iloc[0].to_dict()
        panels.append(_panel(row, vtype))
        ptr = float(row.get("pct_triage_recorded")  or 0)
        pco = float(row.get("pct_consult_recorded") or 0)
        plb = float(row.get("pct_lab_recorded")     or 0)
        if ptr < 50 and "Arrival → Triage"      not in gap_names: gap_names.append("Arrival → Triage")
        if pco < 50 and "Triage → Consultation" not in gap_names: gap_names.append("Triage → Consultation")
        if plb < 50 and "Consult → Lab Result"  not in gap_names: gap_names.append("Consult → Lab Result")
    if "Lab Result → Discharge" not in gap_names:
        gap_names.append("Lab Result → Discharge")

    legend = (
        '<div style="display:flex;flex-wrap:wrap;gap:14px;margin-bottom:12px;">'
        + "".join(
            f'<div style="display:flex;align-items:center;gap:5px;font-size:10px;color:#374151;">'
            f'<div style="width:10px;height:10px;border-radius:2px;background:{c};"></div>{lbl}</div>'
            for c, lbl in [("#1D9E75","Within target"),("#EF9F27","Approaching limit"),("#E24B4A","Breaching target")]
        )
        + '<div style="display:flex;align-items:center;gap:5px;font-size:10px;color:#374151;">'
          '<div style="width:2px;height:14px;background:#7F77DD;border-radius:1px;margin:0 3px;"></div>Target</div>'
        + '</div>'
    )

    n = len(gap_names)
    footer = ""
    if gap_names:
        sw = "stage" if n == 1 else "stages"
        iw = "is" if n == 1 else "are"
        it = "it" if n == 1 else "them"
        sl = ", ".join(gap_names)
        footer = (f'<div style="margin-top:14px;padding:10px 14px;background:#f8fafc;'
                  f'border-radius:8px;font-size:11px;color:#6b7280;">'
                  f'{n} {sw} {iw} not shown above because the timestamps needed to measure {it} '
                  f'are not currently recorded in the system. These gaps mean the full patient '
                  f'journey cannot yet be measured end to end. Recommend adding result receipt '
                  f'timestamps at: <strong>{sl}</strong>.</div>')

    html = (
        '<div style="background:#f5f5f3;padding:12px;'
        'font-family:system-ui,-apple-system,sans-serif;">'
        + legend
        + '<div style="display:flex;gap:12px;">' + "".join(panels) + '</div>'
        + footer + '</div>'
    )
    _stcomp.html(html, height=620, scrolling=False)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def render_tab1_operations(filters: dict, run_query):
    # ── KPI ROW ───────────────────────────────────────────────────────────
    _sh("Key Metrics")
    try:
        df = Q.load_tab1_kpis(filters, run_query)
        df_cost = Q.load_avg_admission_cost_full(filters, run_query)
        if not df.empty:
            row  = df.iloc[0]
            crow = df_cost.iloc[0] if not df_cost.empty else {}
            ip_cost = crow.get("avg_ip_cost_full") or row.get("avg_admission_cost")
            op_cost = crow.get("avg_op_cost_full") or row.get("avg_op_cost")
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            with c1: _kpi("Total Visits",      _n(row.get("total_visits")))
            with c2: _kpi("Inpatient",         _n(row.get("inpatient_visits")),
                          str(_p(row.get("inpatient_pct")) or "—") + " of visits", TEAL)
            with c3: _kpi("Outpatient",        _n(row.get("outpatient_visits")))
            with c4: _kpi("Discharges",        _n(row.get("total_discharges")), color=TEAL)
            with c5: _kpi("Active Admissions", _n(row.get("active_admissions")), color=ORANGE)
            with c6: _kpi("Avg Admission Cost",_k(ip_cost),
                          f"Outpatient avg: {_k(op_cost)}")
    except Exception as e:
        st.warning(f"KPIs: {e}")

    _gap(16)

    # ── SECTION A: SERVICE GROWTH ─────────────────────────────────────────
    _sh("A — Service Growth by Type", mt=8)
    _note("Monthly visit volume by service type. Each panel uses its own scale.")
    try:
        df = Q.load_service_growth(filters, run_query)
        if not df.empty:
            df["outpatient"] = (
                df.get("outpatient_with_lab", pd.Series(0, index=df.index))
                + df.get("outpatient_rx_only", pd.Series(0, index=df.index))
                + df.get("consult_only", pd.Series(0, index=df.index))
            )
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["visit_month"], y=df["outpatient"],
                name="Outpatient", mode="lines+markers",
                line=dict(color=AFYA_BLUE, width=2), marker=dict(size=4),
            ))
            fig.add_trace(go.Scatter(
                x=df["visit_month"], y=df["inpatient"],
                name="Inpatient", mode="lines+markers",
                line=dict(color=TEAL, width=2), marker=dict(size=4),
            ))
            fig.update_layout(
                height=300, margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(showgrid=False),
                yaxis=dict(
                    title="Visits", showgrid=True, gridcolor="#EBF3FB",
                    rangemode="tozero", fixedrange=False,
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
            )
            _pc(fig)
    except Exception as e:
        st.warning(f"Service growth: {e}")

    _gap(12)

    # ── SECTION A2: WARD BREAKDOWN ────────────────────────────────────────
    _sh("Ward Breakdown", mt=8)

    # Ward summary: % share of admissions
    try:
        df_wb = Q.load_ward_breakdown(filters, run_query)
        if not df_wb.empty:
            total_adm = df_wb["admissions"].sum()
            if total_adm > 0:
                df_wb["pct_share"] = (df_wb["admissions"] / total_adm * 100).round(1)
            else:
                df_wb["pct_share"] = 0.0
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Ward Admission Share (%)**")
                _pc(hbar_chart(
                    df_wb.head(12), x="pct_share", y="ward",
                    x_label="% of Total Admissions", color=AFYA_BLUE, height=320,
                ))
            with c2:
                st.markdown("**Ward Summary**")
                _pc(table_fig(
                    df_wb[["ward", "admissions", "pct_share", "avg_los_days",
                            "avg_admission_cost"]].head(12),
                    col_labels={"ward": "Ward", "admissions": "Admissions",
                                "pct_share": "Share %", "avg_los_days": "Avg LOS (d)",
                                "avg_admission_cost": "Avg Cost"},
                    fmt={"avg_admission_cost": "KES", "pct_share": "pct"},
                    height=320,
                ))
    except Exception as e:
        st.warning(f"Ward share: {e}")

    _gap(12)

    # Ward 1: Admission volume — raw actuals (thin) + DB rolling avg (thick) per ward
    try:
        df_trend = Q.load_ward_admission_trend(filters, run_query)
        if not df_trend.empty:
            df_trend = df_trend.sort_values(["ward", "visit_month"])

            # Tier classification: top 3 by avg monthly admissions = Tier 1 (solid)
            avg_by_ward = df_trend.groupby("ward")["admissions"].mean().sort_values(ascending=False)
            ward_order = list(avg_by_ward.index)
            tier1 = set(ward_order[:3])
            _WARD_COLORS = [AFYA_BLUE, TEAL, ORANGE, CORAL, PURPLE, GRAY]
            color_map = {w: _WARD_COLORS[i % len(_WARD_COLORS)] for i, w in enumerate(ward_order)}

            has_avg_col = "admissions_3mo_avg" in df_trend.columns

            # Title + subtitle
            st.markdown(
                '<div style="font-size:15px;font-weight:600;color:#111827;margin-bottom:3px;">'
                'Admission Growth by Ward — Top 6</div>'
                '<div style="font-size:11px;color:#6b7280;margin-bottom:8px;">'
                '3-month rolling trend. Top 3 wards solid · lower-volume wards dashed.</div>',
                unsafe_allow_html=True,
            )
            _mode = st.radio(
                "view", ["Absolute volume", "Growth index"],
                horizontal=True, label_visibility="collapsed",
                key="ward_growth_mode",
            )
            use_index = _mode == "Growth index"

            fig = go.Figure()
            growth_data = {}

            for ward in ward_order:
                wd = df_trend[df_trend["ward"] == ward].sort_values("visit_month")
                x         = wd["visit_month"].tolist()
                y_raw     = wd["admissions"].tolist()
                y_avg     = ([None if pd.isna(v) else float(v) for v in wd["admissions_3mo_avg"]]
                             if has_avg_col else [None] * len(y_raw))

                first_raw = next((v for v in y_raw if v and v > 0), 1)
                last_raw  = next((v for v in reversed(y_raw) if v and v > 0), first_raw)
                g = round((last_raw - first_raw) / first_raw * 100) if first_raw else 0
                growth_data[ward] = g
                g_str = f"+{g}%" if g >= 0 else f"{g}%"

                if use_index:
                    base = first_raw or 1
                    y_plot_raw = [round(v / base * 100, 1) for v in y_raw]
                    y_plot_avg = [round(v / base * 100, 1) if v is not None else None
                                  for v in y_avg]
                else:
                    y_plot_raw = y_raw
                    y_plot_avg = y_avg

                is_t1  = ward in tier1
                col    = color_map[ward]
                dash   = "solid" if is_t1 else "dash"
                y_lbl  = "Index" if use_index else "Admissions"

                # Fill leading Nones so the line starts from the first data point
                y_filled = [r if a is None else a
                            for a, r in zip(y_plot_avg, y_plot_raw)]
                # Rolling-avg line only
                fig.add_trace(go.Scatter(
                    x=x, y=y_filled,
                    name=f"{ward}  {g_str}",
                    legendgroup=ward,
                    showlegend=True,
                    mode="lines",
                    line=dict(color=col, width=2.5 if is_t1 else 1.8, dash=dash),
                    connectgaps=False,
                    hovertemplate=f"<b>{ward}</b><br>%{{x|%b %Y}}: %{{y:.0f}} {y_lbl}<extra></extra>",
                ))

            if use_index:
                fig.add_hline(y=100, line_dash="dot", line_color="rgba(0,0,0,0.12)",
                              line_width=1)

            fig.update_layout(
                height=310,
                paper_bgcolor="#fff", plot_bgcolor="#fff",
                margin=dict(l=0, r=10, t=6, b=0),
                font=dict(size=11, color="#374151"),
                xaxis=dict(
                    showgrid=False, showline=False,
                    tickfont=dict(size=10, color="#9ca3af"),
                    tickformat="%b %y",
                ),
                yaxis=dict(
                    showgrid=True, gridcolor="rgba(0,0,0,0.05)", showline=False,
                    tickfont=dict(size=10, color="#9ca3af"),
                    title=dict(
                        text="Index (base month = 100)" if use_index else "Admissions",
                        font=dict(size=10, color="#9ca3af"),
                    ),
                    rangemode="tozero" if not use_index else "normal",
                ),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    font=dict(size=10, color="#374151"),
                    bgcolor="rgba(0,0,0,0)",
                    traceorder="normal",
                ),
                hovermode="x unified",
            )
            _pc(fig)

            # Insight strip — fastest ward overall and fastest outside Tier 1
            sorted_g = sorted(growth_data.items(), key=lambda kv: kv[1], reverse=True)
            if sorted_g:
                fw, fg = sorted_g[0]
                non_t1 = [(w, g) for w, g in sorted_g if w not in tier1]
                parts  = [
                    f'Fastest overall: <strong style="color:{color_map[fw]}">{fw}</strong>'
                    f' {"+" if fg >= 0 else ""}{fg}%'
                ]
                if non_t1:
                    nw, ng = non_t1[0]
                    tail = " — growing despite lower volume." if ng > 5 else " — modest growth, watch trend."
                    parts.append(
                        f'Fastest outside Tier 1: <strong style="color:{color_map[nw]}">{nw}</strong>'
                        f' {"+" if ng >= 0 else ""}{ng}%{tail}'
                    )
                st.markdown(
                    '<div style="margin-top:6px;padding:8px 12px;background:#f8fafc;'
                    'border-radius:8px;font-size:11px;color:#6b7280;">'
                    + " · ".join(parts) + "</div>",
                    unsafe_allow_html=True,
                )
    except Exception as e:
        st.warning(f"Ward trend: {e}")

    _gap(12)

    # Ward 2: Discharge latency vs patient count
    try:
        df_lat = Q.load_ward_discharge_latency(filters, run_query)
        if not df_lat.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Discharge Latency vs Patient Volume**")
                _note("Which ward takes longest to discharge after the decision is made.")
                _pc(scatter(
                    df_lat, x="patient_count", y="avg_discharge_latency_hrs",
                    label_col="ward",
                    x_label="Patients Discharged", y_label="Avg Discharge Latency (hrs)",
                    height=300,
                ))
            with c2:
                st.markdown("**Discharge Latency Ranking**")
                _pc(hbar_chart(
                    df_lat.sort_values("avg_discharge_latency_hrs", ascending=False).head(12),
                    x="avg_discharge_latency_hrs", y="ward",
                    x_label="Avg Latency (hrs)", color=ORANGE, height=300,
                ))
    except Exception as e:
        st.warning(f"Discharge latency: {e}")

    _gap(12)

    # Ward 3: Monthly admissions (bars) + Avg duration (line) — combined dual-axis
    try:
        df_active = Q.load_ward_active_vs_hours(filters, run_query)
        if not df_active.empty:
            bar_col = "total_admissions" if "total_admissions" in df_active.columns else "active_admissions"
            _note("Average admission time is calculated from completed admissions only. Ongoing admissions are excluded.")

            # Correlation annotation
            valid = df_active[[bar_col, "avg_admission_hours"]].dropna()
            if len(valid) >= 3:
                corr = valid[bar_col].corr(valid["avg_admission_hours"])
                if corr > 0.3:
                    corr_text = f"Higher-admission months correlate with longer average admission times (r = {corr:.2f}) — possible capacity pressure."
                elif corr < -0.3:
                    corr_text = f"Higher-admission months correlate with shorter average admission times (r = {corr:.2f}) — possible early discharge pressure."
                else:
                    corr_text = f"Admission volume shows no consistent relationship with average admission time (r = {corr:.2f})."
            else:
                corr_text = ""

            st.markdown("**Monthly Admissions vs Average Admission Time**")
            if corr_text:
                _note(corr_text)

            fig_a = go.Figure()
            fig_a.add_trace(go.Bar(
                x=df_active["visit_month"], y=df_active[bar_col],
                name="Admissions", marker_color=AFYA_BLUE,
                yaxis="y1", opacity=0.75,
            ))
            fig_a.add_trace(go.Scatter(
                x=df_active["visit_month"], y=df_active["avg_admission_hours"],
                name="Avg Admission Time (hrs)", mode="lines+markers",
                line=dict(color=ORANGE, width=2), marker=dict(size=5),
                yaxis="y2",
            ))
            fig_a.update_layout(
                height=300, margin=dict(l=0, r=60, t=10, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(title="Admissions", color=AFYA_BLUE,
                           showgrid=True, gridcolor="#EBF3FB"),
                yaxis2=dict(title="Avg Admission Time (hrs)", color=ORANGE,
                            overlaying="y", side="right", showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
            )
            _pc(fig_a)
    except Exception as e:
        st.warning(f"Active admissions: {e}")

    _gap(12)

    # Ward 4: Cost per ward vs patient volumes — bubble chart with quadrants
    try:
        df_cost = Q.load_ward_cost_volume(filters, run_query)
        if not df_cost.empty:
            st.markdown("**Cost per Admission vs Volume by Ward**")
            _note(
                "Bubble size = total revenue. Cost = invoice line items per admitted visit. "
                "Quadrant lines = median admissions and median avg cost.",
                w=True,
            )
            med_x = df_cost["admissions"].median()
            med_y = df_cost["avg_admission_cost"].median()

            # Normalise bubble size: 10–50px range
            rev = df_cost["total_revenue"].fillna(0)
            rev_range = max(rev.max() - rev.min(), 1)
            bubble_sizes = ((rev - rev.min()) / rev_range * 40 + 10).tolist()

            fig_cost = go.Figure()

            # Quadrant reference lines
            fig_cost.add_shape(type="line", x0=med_x, x1=med_x,
                               y0=0, y1=1, yref="paper",
                               line=dict(color=GRAY, width=1, dash="dot"))
            fig_cost.add_shape(type="line", x0=0, x1=1, xref="paper",
                               y0=med_y, y1=med_y,
                               line=dict(color=GRAY, width=1, dash="dot"))

            # Quadrant labels
            quad_style = dict(xref="paper", yref="paper", showarrow=False,
                              font=dict(size=9, color=MUTED), bgcolor="rgba(255,255,255,0.7)")
            fig_cost.add_annotation(x=0.02, y=0.98, xanchor="left", yanchor="top",
                                    text="<b>Investigate</b><br><i>Inefficiency or niche premium?</i>",
                                    **quad_style)
            fig_cost.add_annotation(x=0.98, y=0.98, xanchor="right", yanchor="top",
                                    text="<b>Revenue engine</b><br><i>Protect and optimise</i>",
                                    **quad_style)
            fig_cost.add_annotation(x=0.02, y=0.02, xanchor="left", yanchor="bottom",
                                    text="<b>Underutilised</b><br><i>Capacity opportunity</i>",
                                    **quad_style)
            fig_cost.add_annotation(x=0.98, y=0.02, xanchor="right", yanchor="bottom",
                                    text="<b>Operational workhorse</b><br><i>Efficiency model</i>",
                                    **quad_style)

            fig_cost.add_trace(go.Scatter(
                x=df_cost["admissions"],
                y=df_cost["avg_admission_cost"],
                mode="markers+text",
                text=df_cost["ward"],
                textposition="top center",
                textfont=dict(size=9, family="Montserrat", color=COOL_BLUE),
                marker=dict(
                    size=bubble_sizes,
                    color=AFYA_BLUE,
                    opacity=0.75,
                    line=dict(color="white", width=0.5),
                ),
                customdata=df_cost[["total_revenue", "revenue_per_patient"]].values,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Admissions: %{x:,}<br>"
                    "Avg Cost: KES %{y:,.0f}<br>"
                    "Total Revenue: KES %{customdata[0]:,.0f}<br>"
                    "Revenue/Patient: KES %{customdata[1]:,.0f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))

            fig_cost.update_layout(
                height=380,
                margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Admissions", showgrid=True,
                           gridcolor="#EBF3FB", zeroline=False),
                yaxis=dict(title="Avg Cost per Admission (KES)", showgrid=True,
                           gridcolor="#EBF3FB", zeroline=False,
                           tickformat=",.0f"),
            )
            _pc(fig_cost)
    except Exception as e:
        st.warning(f"Ward cost/volume: {e}")

    _gap(12)

    # Ward 5: Top-3 ward operational summary table
    try:
        df_ws = Q.load_top_ward_summary(filters, run_query)
        if not df_ws.empty:
            st.markdown("**What Are the Busiest Wards Treating?**")
            _note(
                "Top 3 wards by admissions. Pressure = recent 3-month avg vs prior 3-month avg. "
                "Seasonality: CV > 0.3 = seasonal pattern, < 0.15 = consistent."
            )

            for _c in ("admissions", "pressure_pct", "cv_seasonality", "top_condition_pct",
                       "new_pct", "returning_pct", "top_payer_pct",
                       "avg_los_days", "investigation_rate_pct",
                       "patients_per_clinician", "clinicians"):
                if _c in df_ws.columns:
                    df_ws[_c] = pd.to_numeric(df_ws[_c], errors="coerce")

            rows = []
            for _, r in df_ws.iterrows():
                pressure_pct = r.get("pressure_pct", 0) or 0
                pressure_sig = r.get("pressure_signal", "Stable")
                pressure_arrow = {"Rising": "↑", "Easing": "↓", "Stable": "→"}.get(
                    pressure_sig, "→"
                )
                cv = float(r.get("cv_seasonality") or 0)
                seasonality = "Seasonal" if cv > 0.30 else ("Consistent" if cv < 0.15 else "Mixed")

                top_cond     = r.get("top_condition") or "—"
                cond_pct     = r.get("top_condition_pct") or 0
                cond_pattern = r.get("condition_pattern") or ""
                cond_str     = f"{top_cond} ({int(cond_pct)}%)"
                if cond_pattern == "Varies":
                    cond_str += " + others"

                payer      = r.get("top_payer") or "—"
                payer_pct  = r.get("top_payer_pct") or 0
                payer_str  = f"{payer} ({int(payer_pct)}%)"

                los = r.get("avg_los_days")
                los_str = f"{float(los):.1f} days" if pd.notna(los) and los else "—"

                inv_rate = r.get("investigation_rate_pct")
                inv_str  = f"{int(inv_rate)}%" if pd.notna(inv_rate) and inv_rate else "—"

                ratio = r.get("patients_per_clinician")
                clin  = r.get("clinicians")
                ratio_str = (f"{float(ratio):.0f}:1  ({int(clin)} clinicians)"
                             if pd.notna(ratio) and ratio else "—")

                rows.append({
                    "Ward":              r.get("ward", "—"),
                    "Admissions":        f"{int(r.get('admissions', 0)):,}",
                    "Pressure":          f"{pressure_sig} {pressure_arrow} ({int(pressure_pct):+d}%)",
                    "Seasonality":       seasonality,
                    "Top Condition":     cond_str,
                    "New / Returning":   f"{int(r.get('new_pct', 0))}% new · {int(r.get('returning_pct', 0))}% returning",
                    "Top Payer":         payer_str,
                    "Avg Admission":     los_str,
                    "Investigation Rate": inv_str,
                    "Pts / Clinician":   ratio_str,
                })

            _pc(table_fig(
                pd.DataFrame(rows),
                col_labels={},
                height=max(200, len(rows) * 52 + 60),
            ))
    except Exception as e:
        st.warning(f"Ward summary: {e}")

    _gap(12)

    # ── DIAGNOSIS COST OUTLIERS ───────────────────────────────────────────
    _sh("Which Diagnoses Are Driving Disproportionate Cost?", mt=8)
    _note(
        "A ratio above 1.0 means this diagnosis consumes more cost share than its "
        "patient volume justifies. Use as a flag — deeper drug and investigation "
        "breakdowns belong in inpatient analytics."
    )
    try:
        df_dc = Q.load_diagnosis_cost_outliers(filters, run_query)
        if not df_dc.empty:
            for _c in ("visit_count", "avg_cost_per_case", "total_cost",
                       "volume_share_pct", "cost_share_pct", "cost_volume_ratio"):
                df_dc[_c] = pd.to_numeric(df_dc[_c], errors="coerce")

            overall_avg_cost = float(
                (df_dc["total_cost"].sum() / df_dc["visit_count"].sum())
                if df_dc["visit_count"].sum() > 0 else 0
            )

            def _ratio_color(r):
                if r >= 2.0:   return CORAL
                if r >= 1.5:   return ORANGE
                return TEAL

            colors = [_ratio_color(r) for r in df_dc["cost_volume_ratio"]]
            sizes  = df_dc["total_cost"]
            s_min, s_max = float(sizes.min()), float(sizes.max())
            s_range = s_max - s_min or 1
            bubble_sizes = [12 + (float(v) - s_min) / s_range * 28
                            for v in sizes]

            c1, c2 = st.columns([3, 2])
            with c1:
                fig_dc = go.Figure()
                fig_dc.add_trace(go.Scatter(
                    x=df_dc["visit_count"],
                    y=df_dc["avg_cost_per_case"],
                    mode="markers+text",
                    marker=dict(
                        color=colors,
                        size=bubble_sizes,
                        opacity=0.82,
                        line=dict(color="white", width=1),
                    ),
                    text=df_dc["diagnosis_group"],
                    textposition="top center",
                    textfont=dict(size=8, color=MUTED),
                    customdata=df_dc[["cost_volume_ratio", "cost_share_pct",
                                      "volume_share_pct", "total_cost"]].values,
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Visits: %{x:,}<br>"
                        "Avg cost/case: %{y:,.0f}<br>"
                        "Cost share: %{customdata[1]:.1f}%<br>"
                        "Volume share: %{customdata[2]:.1f}%<br>"
                        "Ratio: %{customdata[0]:.2f}x<extra></extra>"
                    ),
                ))
                # Overall avg cost reference line
                fig_dc.add_shape(
                    type="line",
                    xref="paper", yref="y",
                    x0=0, x1=1,
                    y0=overall_avg_cost, y1=overall_avg_cost,
                    line=dict(color=GRAY, width=1.5, dash="dash"),
                )
                fig_dc.add_annotation(
                    xref="paper", yref="y",
                    x=1.01, y=overall_avg_cost,
                    text="Avg cost",
                    xanchor="left", yanchor="middle",
                    showarrow=False,
                    font=dict(size=8, color=GRAY),
                )
                fig_dc.update_layout(
                    height=380,
                    margin=dict(l=0, r=60, t=20, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(title="Visit Count", showgrid=False),
                    yaxis=dict(
                        title="Avg Cost per Case",
                        showgrid=True, gridcolor="#EBF3FB",
                    ),
                )
                _pc(fig_dc)
                _note(
                    "Bubble size = total cost. "
                    "Red = ratio ≥ 2×  ·  Orange = 1.5–2×  ·  Teal = below 1.5×. "
                    "Diagnoses above the dashed line cost more per case than the average."
                )

            with c2:
                tbl = df_dc[["diagnosis_group", "visit_count",
                              "avg_cost_per_case", "cost_volume_ratio"]].copy()
                tbl["cost_volume_ratio"] = tbl["cost_volume_ratio"].apply(
                    lambda v: f"{v:.2f}×" if pd.notna(v) else "—"
                )
                tbl["avg_cost_per_case"] = tbl["avg_cost_per_case"].apply(
                    lambda v: f"{int(v):,}" if pd.notna(v) else "—"
                )
                tbl["visit_count"] = tbl["visit_count"].apply(
                    lambda v: f"{int(v):,}" if pd.notna(v) else "—"
                )
                _pc(table_fig(
                    tbl.head(12).rename(columns={
                        "diagnosis_group":   "Diagnosis",
                        "visit_count":       "Visits",
                        "avg_cost_per_case": "Avg Cost",
                        "cost_volume_ratio": "Ratio",
                    }),
                    col_labels={},
                    height=380,
                ))
    except Exception as e:
        st.warning(f"Diagnosis cost outliers: {e}")

    _gap(12)

    # ── SECTION B: SPIKE & DIP DETECTION ─────────────────────────────────
    _sh("B — When did volume behave unusually?", mt=8)

    try:
        df_vol = Q.load_monthly_volume_anomalies(filters, run_query)
        if not df_vol.empty:
            df_vol = df_vol.copy()
            df_vol["month_label"] = pd.to_datetime(
                df_vol["visit_month"], errors="coerce"
            ).dt.strftime("%Y-%m")

            # Drop the first partial month from chart — it skews visuals
            df_chart = df_vol[df_vol["month_type"] != "Partial"].copy()

            avg_vol = float(df_vol["avg_vol"].iloc[0])
            sd_vol  = float(df_vol["sd_vol"].iloc[0])

            spikes = df_chart[df_chart["month_type"] == "Spike"]
            dips   = df_chart[df_chart["month_type"] == "Dip"]

            # Fallback: no months cross 1.0 SD → use top-2 / bottom-2
            use_fallback = spikes.empty and dips.empty
            if use_fallback:
                spikes = df_chart.nlargest(2, "total_visits").copy()
                spikes["month_type"] = "Highest"
                dips = df_chart.nsmallest(2, "total_visits").copy()
                dips["month_type"] = "Lowest"
                _note(
                    "No months exceeded ±1 SD from the mean. "
                    "Showing the 2 highest and 2 lowest volume months instead."
                )

            # ── Time-series bar chart ──────────────────────────────────────
            type_color = {"Spike": ORANGE, "Dip": CORAL,
                          "Highest": ORANGE, "Lowest": CORAL, "Normal": AFYA_BLUE}
            bar_colors = [type_color.get(t, AFYA_BLUE) for t in df_chart["month_type"]]

            fig_vol = go.Figure()

            # ±1 SD shaded band (normal range)
            fig_vol.add_shape(
                type="rect", xref="paper", yref="y",
                x0=0, x1=1,
                y0=avg_vol - sd_vol, y1=avg_vol + sd_vol,
                fillcolor=GRAY, opacity=0.10, line_width=0,
            )

            # Spike threshold — upper edge of normal range
            fig_vol.add_hline(
                y=avg_vol + sd_vol,
                line=dict(color=ORANGE, width=1.5, dash="dash"),
                annotation_text=f"Spike threshold  ({avg_vol + sd_vol:,.0f})",
                annotation_position="right",
                annotation_font=dict(size=9, color=ORANGE),
            )

            # Dip threshold — lower edge of normal range
            fig_vol.add_hline(
                y=avg_vol - sd_vol,
                line=dict(color=CORAL, width=1.5, dash="dash"),
                annotation_text=f"Dip threshold  ({avg_vol - sd_vol:,.0f})",
                annotation_position="right",
                annotation_font=dict(size=9, color=CORAL),
            )

            fig_vol.add_trace(go.Bar(
                x=df_chart["month_label"],
                y=df_chart["total_visits"],
                marker_color=bar_colors,
                hovertemplate=(
                    "<b>%{x}</b><br>Visits: %{y:,}<extra></extra>"
                ),
                showlegend=False,
            ))

            # Legend patches (manual)
            for label, color in [("Spike / Highest", ORANGE),
                                  ("Dip / Lowest", CORAL),
                                  ("Normal", AFYA_BLUE)]:
                fig_vol.add_trace(go.Bar(
                    x=[None], y=[None], name=label,
                    marker_color=color, showlegend=True,
                ))

            fig_vol.update_layout(
                height=300,
                margin=dict(l=0, r=60, t=10, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(showgrid=False),
                yaxis=dict(title="Total Visits", showgrid=True,
                           gridcolor="#EBF3FB"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
                barmode="overlay",
            )
            _pc(fig_vol)

            # ── Summary tables with diagnosis context ─────────────────────
            spike_label = "Highest Volume Months" if use_fallback else "Spike Months"
            dip_label   = "Lowest Volume Months"  if use_fallback else "Dip Months"

            def _fmt_mom(row):
                if row.get("first_in_range") == 1:
                    return "prior period outside range"
                v = row.get("mom_pct")
                return f"{v:+.1f}%" if v is not None else "—"

            # Diagnosis context: top diagnosis + new/returning per anomalous month
            try:
                df_dx = Q.load_diagnosis_by_month(filters, run_query)
            except Exception:
                df_dx = pd.DataFrame()

            normal_months = df_chart[df_chart["month_type"] == "Normal"]["visit_month"].tolist()
            if not df_dx.empty and normal_months:
                baseline_new = (df_dx[df_dx["visit_month"].isin(normal_months)]
                                .groupby("visit_month")["new_patients"].sum().mean())
                baseline_ret = (df_dx[df_dx["visit_month"].isin(normal_months)]
                                .groupby("visit_month")["returning_patients"].sum().mean())
            else:
                baseline_new = baseline_ret = None

            def _enrich(df_in):
                rows = []
                for _, r in df_in.iterrows():
                    m = r["visit_month"]
                    mom = _fmt_mom(r)
                    if not df_dx.empty:
                        mdf = df_dx[df_dx["visit_month"] == m]
                        top_dx = (mdf.groupby("diagnosis_group")["visit_count"]
                                  .sum().idxmax() if not mdf.empty else "—")
                        new_pt = int(mdf["new_patients"].sum())
                        ret_pt = int(mdf["returning_patients"].sum())
                    else:
                        top_dx, new_pt, ret_pt = "—", None, None
                    rows.append({
                        "month_label":   r["month_label"],
                        "total_visits":  int(r["total_visits"]),
                        "z_score":       r["z_score"],
                        "mom_display":   mom,
                        "top_diagnosis": top_dx,
                        "new_patients":  new_pt,
                        "returning_pts": ret_pt,
                    })
                return pd.DataFrame(rows)

            col_labels = {
                "month_label":   "Month",
                "total_visits":  "Visits",
                "z_score":       "Z-Score",
                "mom_display":   "MoM Change",
                "top_diagnosis": "Top Diagnosis",
                "new_patients":  "New Patients",
                "returning_pts": "Returning Patients",
            }

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**{spike_label}**")
                tdf = _enrich(spikes)
                _pc(table_fig(tdf, col_labels=col_labels,
                              height=max(120, len(tdf) * 36 + 60)))
            with c2:
                st.markdown(f"**{dip_label}**")
                tdf = _enrich(dips)
                _pc(table_fig(tdf, col_labels=col_labels,
                              height=max(120, len(tdf) * 36 + 60)))

            if baseline_new is not None:
                _note(
                    f"Baseline avg (normal months): "
                    f"{baseline_new:,.0f} new patients / month, "
                    f"{baseline_ret:,.0f} returning patients / month."
                )

    except Exception as e:
        st.warning(f"Volume anomalies: {e}")

    _gap(12)

    # ── SECTION C: PATIENT WAIT TIMES ────────────────────────────────────
    _sh("C — Patient Wait Times by Stage", mt=8)
    _note(
        "Shows the time that 1 in 10 of the slowest patients waited at each stage "
        "(P90). Each panel is on its own scale. Targets are fixed clinical benchmarks."
    )
    try:
        df_jt = Q.load_journey_times(filters, run_query)
        if not df_jt.empty:
            _render_wait_time_chart(df_jt)
    except Exception as e:
        st.warning(f"Journey times: {e}")

    _gap(12)

    # Lab / investigation turnaround by clinical discipline
    _sh("Lab & Investigation Turnaround by Clinical Discipline", mt=8)
    try:
        _note(
            "Grouped by clinical discipline from procedure_discipline — "
            "Haematology, Clinical Chemistry, Microbiology & Infectious Disease, "
            "Immunology & Serology, Endocrinology & Hormones, Pathology & Cytology, "
            "Radiology & Imaging, Imaging & Diagnostics."
        )
        df_lab = Q.load_lab_turnaround_by_discipline(filters, run_query)

        # Fallback: discipline query empty → use simpler by-type query
        _disc_mode = True
        if df_lab.empty:
            df_lab = Q.load_lab_turnaround_by_test(filters, run_query)
            _disc_mode = False
            if not df_lab.empty:
                df_lab = df_lab.rename(columns={"test_type": "discipline"})
                _note("Discipline grouping returned no data — showing results by investigation type.")

        if not df_lab.empty:
            # Coerce Snowflake Decimal columns to float
            for _c in ("test_count", "avg_turnaround_hrs", "median_turnaround_hrs", "result_rate_pct"):
                if _c in df_lab.columns:
                    df_lab[_c] = pd.to_numeric(df_lab[_c], errors="coerce")

            # ── per-discipline summary (weighted collapse to discipline level) ──
            rows = []
            for disc, g in df_lab.groupby("discipline"):
                total = float(g["test_count"].sum())
                avg_tat = g["avg_turnaround_hrs"].dropna()
                rows.append({
                    "discipline":            disc,
                    "test_count":            int(total),
                    "avg_turnaround_hrs":    round(
                        float((g["avg_turnaround_hrs"].fillna(0) * g["test_count"]).sum()) / total, 2
                    ) if total and not avg_tat.empty else None,
                    "median_turnaround_hrs": round(float(g["median_turnaround_hrs"].median()), 2)
                                             if g["median_turnaround_hrs"].notna().any() else None,
                    "result_rate_pct":       round(
                        float((g["result_rate_pct"].fillna(0) * g["test_count"]).sum()) / total, 1
                    ) if total else None,
                })
            disc_summary = pd.DataFrame(rows).reset_index(drop=True)
            has_tat = disc_summary["avg_turnaround_hrs"].notna().any()
            sort_col = "avg_turnaround_hrs" if has_tat else "test_count"
            disc_summary = disc_summary.sort_values(sort_col, ascending=False).reset_index(drop=True)

            # Colour map for disciplines (procedure_discipline values)
            _disc_colors = {
                "Haematology":                    AFYA_BLUE,
                "Clinical Chemistry":             TEAL,
                "Microbiology & Infectious Disease": ORANGE,
                "Immunology & Serology":          PURPLE,
                "Endocrinology & Hormones":       GREEN,
                "Pathology & Cytology":           CORAL,
                "Radiology & Imaging":            GRAY,
                "Imaging & Diagnostics":          COOL_BLUE,
                "Other / Unclassified":           "#AAAAAA",
            }
            bar_colors = [_disc_colors.get(d, AFYA_BLUE) for d in disc_summary["discipline"]]

            c1, c2 = st.columns(2)
            with c1:
                if has_tat:
                    x_vals = disc_summary["avg_turnaround_hrs"]
                    x_title = "Avg Turnaround (hrs)"
                    txt = disc_summary["avg_turnaround_hrs"].apply(
                        lambda v: f"{v:.1f}h" if pd.notna(v) else ""
                    )
                else:
                    x_vals = disc_summary["test_count"]
                    x_title = "Tests ordered"
                    txt = disc_summary["test_count"].apply(lambda v: f"{int(v):,}")
                    _note("Result timestamps not recorded — showing test volume by discipline.")

                fig_disc = go.Figure(go.Bar(
                    y=disc_summary["discipline"],
                    x=x_vals,
                    orientation="h",
                    marker_color=bar_colors,
                    text=txt,
                    textposition="outside",
                ))
                fig_disc.update_layout(
                    height=320,
                    xaxis_title=x_title,
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=0, r=40, t=20, b=20),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                )
                _pc(fig_disc)
            with c2:
                tbl_cols = {
                    "discipline":            "Discipline",
                    "test_count":            "Tests",
                    "avg_turnaround_hrs":    "Avg Hrs",
                    "median_turnaround_hrs": "Median Hrs",
                    "result_rate_pct":       "Result %",
                }
                _pc(table_fig(
                    disc_summary,
                    col_labels=tbl_cols,
                    fmt={"result_rate_pct": "pct"},
                    height=320,
                ))

            if _disc_mode:
                # ── drill-down: per-discipline x investigation_type breakdown ─
                _gap(8)
                selected_disc = st.selectbox(
                    "Drill into discipline",
                    disc_summary["discipline"].tolist(),
                    key="lab_disc_select",
                )
                df_drill = (
                    df_lab[df_lab["discipline"] == selected_disc]
                    .sort_values("avg_turnaround_hrs" if has_tat else "test_count",
                                 ascending=False)
                    .reset_index(drop=True)
                )
                if not df_drill.empty:
                    _x_col = "avg_turnaround_hrs" if has_tat else "test_count"
                    df_chart = df_drill.dropna(subset=[_x_col]).reset_index(drop=True).copy()
                    df_chart["test_name"] = df_chart["test_name"].apply(
                        lambda v: (v[:42] + "…") if isinstance(v, str) and len(v) > 43 else v
                    )
                    _tbl_cols = {
                        "test_name":             "Test",
                        "test_count":            "Count",
                        "avg_turnaround_hrs":    "Avg Hrs",
                        "median_turnaround_hrs": "Median Hrs",
                        "result_rate_pct":       "Result %",
                    }
                    df_tbl = df_drill[[c for c in _tbl_cols if c in df_drill.columns]].copy()
                    d1, d2 = st.columns(2)
                    with d1:
                        _pc(hbar_chart(
                            df_chart,
                            x=_x_col,
                            y="test_name",
                            x_label="Avg Turnaround (hrs)" if has_tat else "Tests ordered",
                            color=_disc_colors.get(selected_disc, AFYA_BLUE),
                            height=max(220, len(df_chart) * 30 + 60),
                            show_text=True,
                        ))
                    with d2:
                        _pc(table_fig(
                            df_tbl,
                            col_labels=_tbl_cols,
                            fmt={"result_rate_pct": "pct"},
                            height=max(220, len(df_tbl) * 30 + 60),
                        ))
        else:
            _note("No investigation data found for the selected period.")
    except Exception as e:
        st.warning(f"Lab turnaround: {e}")

    _gap(12)

    # ── SECTION D: PEAK DEMAND ────────────────────────────────────────────
    _sh("D — Peak Demand: When, Who & Why", mt=8)

    # D1: Hour × Day heatmap with actual values
    try:
        df_hm = Q.load_peak_demand_heatmap(filters, run_query)
        if not df_hm.empty:
            day_order = ["Monday","Tuesday","Wednesday","Thursday",
                         "Friday","Saturday","Sunday"]
            hmap_sel = st.radio(
                "Visit type", ["Total", "Outpatient", "Inpatient"],
                horizontal=True, key="heatmap_type",
            )
            z_col = {"Total": "visit_count",
                     "Outpatient": "outpatient_count",
                     "Inpatient": "inpatient_count"}[hmap_sel]

            # Build heatmap with annotated values
            pivot = df_hm.pivot_table(
                index="day_name", columns="hour_of_day",
                values=z_col, aggfunc="sum", fill_value=0,
            )
            pivot = pivot.reindex([d for d in day_order if d in pivot.index])

            fig_hm = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[f"{h:02d}:00" for h in pivot.columns],
                y=list(pivot.index),
                colorscale="Blues",
                text=pivot.values,
                texttemplate="%{text:,}",
                textfont=dict(size=8),
                hovertemplate="<b>%{y} %{x}</b><br>Visits: %{z:,}<extra></extra>",
                colorbar=dict(thickness=12),
            ))
            fig_hm.update_layout(
                height=280, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Hour of Day (EAT)"),
                yaxis=dict(title="Day", autorange="reversed"),
            )
            _pc(fig_hm)

            peak = df_hm.loc[df_hm["visit_count"].idxmax()]
            _note(
                f"Busiest slot: {peak['day_name']} at {int(peak['hour_of_day']):02d}:00 "
                f"({int(peak['visit_count']):,} total · "
                f"{int(peak['outpatient_count']):,} OP · "
                f"{int(peak['inpatient_count']):,} IP)."
            )
    except Exception as e:
        st.warning(f"Heatmap: {e}")

    _gap(12)

    # D2: Monthly peaks — which months and what contributed
    try:
        df_mo = Q.load_peak_demand_monthly(filters, run_query)
        if not df_mo.empty:
            st.markdown("**Monthly Volume: Peak Months & Composition**")
            c1, c2 = st.columns(2)
            with c1:
                fig_mb = go.Figure()
                colors_op = [ORANGE if p else AFYA_BLUE for p in df_mo["is_peak_month"]]
                fig_mb.add_trace(go.Bar(
                    x=df_mo["visit_month"], y=df_mo["outpatient_visits"],
                    name="Outpatient", marker_color=colors_op, opacity=0.9,
                ))
                fig_mb.add_trace(go.Bar(
                    x=df_mo["visit_month"], y=df_mo["inpatient_visits"],
                    name="Inpatient", marker_color=TEAL,
                ))
                fig_mb.update_layout(
                    barmode="stack", height=280,
                    margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(title="Month"),
                    yaxis=dict(title="Visits", showgrid=True, gridcolor="#EBF3FB"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1),
                )
                _note("Orange bars = peak months (>1.0 SD above mean).")
                _pc(fig_mb)
            with c2:
                st.markdown("**What Brought Patients in Anomalous Months?**")
                peak_rows = df_mo[df_mo["is_peak_month"] == 1].copy()
                fallback = peak_rows.empty
                if fallback:
                    peak_rows = df_mo.nlargest(2, "total_visits").copy()

                # Enrich with top diagnosis group for each highlighted month
                try:
                    df_dx = Q.load_diagnosis_by_month(filters, run_query)
                    if not df_dx.empty:
                        top_dx = (
                            df_dx.sort_values("visit_count", ascending=False)
                            .drop_duplicates("visit_month")[["visit_month", "diagnosis_group"]]
                            .rename(columns={"diagnosis_group": "Top Diagnosis"})
                        )
                        peak_rows = peak_rows.merge(top_dx, on="visit_month", how="left")
                except Exception:
                    pass

                tbl_cols = ["visit_month", "total_visits", "z_score",
                            "communicable_pct", "ncd_pct"]
                tbl_rename = {
                    "visit_month": "Month", "total_visits": "Visits",
                    "z_score": "Z-Score",
                    "communicable_pct": "Communicable %", "ncd_pct": "NCD %",
                }
                if "Top Diagnosis" in peak_rows.columns:
                    tbl_cols.append("Top Diagnosis")

                _pc(table_fig(
                    peak_rows[tbl_cols].rename(columns=tbl_rename),
                    col_labels={},
                    fmt={"Communicable %": "pct", "NCD %": "pct"},
                    height=260,
                ))

                if fallback:
                    _note("No statistical spike in selected period — showing top 2 months by volume.")
                else:
                    top = peak_rows.sort_values("communicable_pct", ascending=False).iloc[0]
                    if pd.to_numeric(top["communicable_pct"], errors="coerce") > 30:
                        _note(
                            f"Peak in {str(top['visit_month'])[:7]} was "
                            f"{top['communicable_pct']:.0f}% communicable disease — "
                            "likely outbreak driven.",
                            w=True,
                        )
    except Exception as e:
        st.warning(f"Monthly peaks: {e}")

    _gap(12)

    # ── SECTION E: PATIENT FLOW UNDER PRESSURE ───────────────────────────
    _sh("E — Patient Flow Under Pressure", mt=8)
    _note(
        "How does operational performance change between peak and quiet days? "
        "Do night-shift patients convert to admissions differently, and does volume stress "
        "affect how quickly patients are seen, investigated, and discharged?"
    )

    # E1: Night-shift A&E → morning admission conversion
    _gap(8)
    st.markdown("**E1 — Night-Shift A&E: How Many Convert to Morning Admissions?**")
    try:
        df_night = Q.load_night_ae_conversion(filters, run_query)
        if not df_night.empty:
            for _c in ("total_visits", "admitted", "conversion_rate_pct",
                       "morning_admit_pct", "avg_wait_to_admit_hrs",
                       "insurance_pct", "surgery_pct"):
                if _c in df_night.columns:
                    df_night[_c] = pd.to_numeric(df_night[_c], errors="coerce")

            cols = st.columns(len(df_night))
            for col, (_, row) in zip(cols, df_night.iterrows()):
                shift = row.get("shift", "")
                conv  = row.get("conversion_rate_pct", 0)
                morn  = row.get("morning_admit_pct", 0)
                wait  = row.get("avg_wait_to_admit_hrs", 0)
                total = row.get("total_visits", 0)
                adm   = row.get("admitted", 0)
                card_color = AFYA_BLUE if "Night" in str(shift) else TEAL

                with col:
                    st.markdown(
                        f'<div style="border-radius:10px;overflow:hidden;margin-bottom:12px;'
                        f'box-shadow:0 1px 4px rgba(0,0,0,0.08);">'
                        f'<div style="background:{card_color};padding:8px 14px;">'
                        f'<span style="color:white;font-size:12px;font-weight:700;letter-spacing:0.5px;">{shift}</span>'
                        f'</div>'
                        f'<div style="background:#FAFCFF;padding:12px 14px;">'
                        f'<div style="display:flex;align-items:baseline;gap:6px;margin-bottom:6px;">'
                        f'<span style="font-size:36px;font-weight:800;color:{card_color};line-height:1;">{_n(conv)}%</span>'
                        f'<span style="font-size:11px;color:#888;">admission rate</span>'
                        f'</div>'
                        f'<div style="font-size:11px;color:#666;margin-bottom:8px;">'
                        f'{int(total or 0):,} visits &rarr; {int(adm or 0):,} admitted'
                        f'</div>'
                        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">'
                        f'<div style="background:#EEF4FF;border-radius:6px;padding:6px 8px;text-align:center;">'
                        f'<div style="font-size:15px;font-weight:700;color:#333;">{_n(wait)} hrs</div>'
                        f'<div style="font-size:10px;color:#888;">avg wait</div>'
                        f'</div>'
                        f'<div style="background:#EEF4FF;border-radius:6px;padding:6px 8px;text-align:center;">'
                        f'<div style="font-size:15px;font-weight:700;color:#333;">{_n(morn)}%</div>'
                        f'<div style="font-size:10px;color:#888;">next morning</div>'
                        f'</div>'
                        f'</div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

            night_row = df_night[df_night["shift"].str.contains("Night", na=False)]
            if not night_row.empty:
                nr = night_row.iloc[0]
                day_conv = df_night[~df_night["shift"].str.contains("Night", na=False)]["conversion_rate_pct"].values
                day_conv_val = _n(day_conv[0]) if len(day_conv) else "—"
                _note(
                    f"Night: {_n(nr.get('conversion_rate_pct'))}% admission rate vs {day_conv_val}% day. "
                    f"{_n(nr.get('morning_admit_pct'))}% of night admissions recorded next morning. "
                    f"Avg wait: {_n(nr.get('avg_wait_to_admit_hrs'))} hrs."
                )
        else:
            _note("No admission data available for the selected period.", w=True)
    except Exception as e:
        st.warning(f"Night A&E conversion: {e}")

    _gap(12)

    # E2: Service delivery times — peak vs normal vs quiet days
    st.markdown("**E2 — Are Services Slower on Peak Days?**")
    _note(
        "Each visit is assigned to the load tier of its day. "
        "Peak = top 25% busiest days. Quiet = bottom 25%. Times are median minutes from arrival."
    )
    try:
        df_svc = Q.load_peak_day_service_times(filters, run_query)
        if not df_svc.empty:
            for _c in ("median_mins_to_triage", "p90_mins_to_triage",
                       "median_mins_to_consult", "p90_mins_to_consult",
                       "median_mins_to_inv", "p90_mins_to_inv",
                       "median_mins_to_rx", "p90_mins_to_rx",
                       "vitals_rate_pct", "notes_rate_pct", "inv_rate_pct", "rx_rate_pct",
                       "avg_clinicians_on_day", "avg_patients_per_clinician", "total_visits"):
                if _c in df_svc.columns:
                    df_svc[_c] = pd.to_numeric(df_svc[_c], errors="coerce")

            tier_order  = ["Peak (top 25%)", "Normal", "Quiet (bottom 25%)"]
            tier_colors = {"Peak (top 25%)": CORAL, "Normal": TEAL, "Quiet (bottom 25%)": COOL_BLUE}

            c_left, c_right = st.columns([1.1, 0.9])
            with c_left:
                # Grouped bar: median wait times by tier
                metrics = [
                    ("median_mins_to_triage",  "Triage (vitals)"),
                    ("median_mins_to_consult",  "Doctor consult"),
                    ("median_mins_to_inv",      "First investigation"),
                    ("median_mins_to_rx",       "Prescription dispensed"),
                ]
                fig_svc = go.Figure()
                tiers_present = [t for t in tier_order if t in df_svc["load_tier"].values]
                x_labels = [m[1] for m in metrics]

                for tier in tiers_present:
                    row = df_svc[df_svc["load_tier"] == tier].iloc[0]
                    y_vals = [
                        None if pd.isna(row.get(m[0])) or row.get(m[0]) is None
                        else float(row.get(m[0]))
                        for m in metrics
                    ]
                    fig_svc.add_trace(go.Bar(
                        name=tier,
                        x=x_labels,
                        y=y_vals,
                        marker_color=tier_colors.get(tier, GRAY),
                        text=[f"{int(v)}m" if v is not None else "" for v in y_vals],
                        textposition="outside",
                        hovertemplate=(
                            f"<b>{tier}</b><br>"
                            "%{x}<br>"
                            "Median: %{y:.0f} min<br>"
                            "<extra></extra>"
                        ),
                    ))

                fig_svc.update_layout(
                    barmode="group",
                    height=320,
                    margin=dict(l=0, r=20, t=30, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(title="Median minutes", rangemode="tozero"),
                    legend=dict(orientation="h", y=1.05, xanchor="right", x=1),
                )
                _pc(fig_svc)

            with c_right:
                # Summary table: all tiers with rate metrics + clinician ratio
                disp_cols = ["load_tier", "total_visits", "avg_patients_per_clinician",
                             "vitals_rate_pct", "notes_rate_pct", "inv_rate_pct", "rx_rate_pct",
                             "median_mins_to_triage", "median_mins_to_consult",
                             "median_mins_to_inv", "median_mins_to_rx"]
                df_svc_disp = df_svc[[c for c in disp_cols if c in df_svc.columns]].copy()
                _pc(table_fig(
                    df_svc_disp,
                    col_labels={
                        "load_tier": "Load Tier", "total_visits": "Visits",
                        "avg_patients_per_clinician": "Patients / Clinician",
                        "vitals_rate_pct": "Vitals %",
                        "notes_rate_pct": "Notes %",
                        "inv_rate_pct": "Investigation %",
                        "rx_rate_pct": "Prescription %",
                        "median_mins_to_triage": "Triage (min)",
                        "median_mins_to_consult": "Consult (min)",
                        "median_mins_to_inv": "Investigation (min)",
                        "median_mins_to_rx": "Prescription (min)",
                    },
                    fmt={"vitals_rate_pct": "pct", "notes_rate_pct": "pct",
                         "inv_rate_pct": "pct", "rx_rate_pct": "pct"},
                    height=220,
                ))

                # Clinician ratio on peak vs quiet — call it out
                peak_row  = df_svc[df_svc["load_tier"].str.startswith("Peak")]
                quiet_row = df_svc[df_svc["load_tier"].str.startswith("Quiet")]
                if not peak_row.empty and not quiet_row.empty:
                    pr = peak_row.iloc[0].get("avg_patients_per_clinician", None)
                    qr = quiet_row.iloc[0].get("avg_patients_per_clinician", None)
                    if pd.notna(pr) and pd.notna(qr) and qr > 0:
                        ratio_diff = ((float(pr) - float(qr)) / float(qr)) * 100
                        _note(
                            f"Clinician load: {_n(pr)} patients per clinician on peak days "
                            f"vs {_n(qr)} on quiet days — "
                            f"{'a {:.0f}% higher load when volume spikes.'.format(ratio_diff) if ratio_diff > 0 else 'no meaningful difference.'}"
                        )
        else:
            _note("Insufficient data to compute service time tiers.", w=True)
    except Exception as e:
        st.warning(f"Peak day service times: {e}")

    _gap(12)

    # E3: Off-peak investigation over-ordering by hour
    st.markdown("**E3 — Are Off-peak Clinicians Over-ordering Investigations?**")
    _note(
        "Blue bars show what % of visits had an investigation ordered each hour. "
        "The red line shows visit volume. Where the bar stays high but the line drops, "
        "clinicians are ordering investigations on a disproportionately high share of a smaller patient load."
    )
    try:
        df_inv_hr = Q.load_offpeak_investigation_pattern(filters, run_query)
        if not df_inv_hr.empty:
            for _c in ("hour_eat", "total_visits", "inv_rate_pct", "avg_inv_per_visit"):
                if _c in df_inv_hr.columns:
                    df_inv_hr[_c] = pd.to_numeric(df_inv_hr[_c], errors="coerce")

            hour_labels = [f"{int(h):02d}:00" for h in df_inv_hr["hour_eat"]]

            fig_inv_hr = go.Figure()
            # Bars: investigation rate % by hour
            fig_inv_hr.add_trace(go.Bar(
                x=hour_labels,
                y=df_inv_hr["inv_rate_pct"],
                name="Investigation rate %",
                marker_color=AFYA_BLUE,
                hovertemplate="<b>%{x}</b><br>Investigation rate: %{y:.1f}%<extra></extra>",
            ))
            # Line: visit volume — the denominator that reveals whether high rate is driven by low volume
            fig_inv_hr.add_trace(go.Scatter(
                x=hour_labels,
                y=df_inv_hr["total_visits"],
                mode="lines+markers",
                line=dict(color=CORAL, width=2),
                marker=dict(size=5),
                name="Visit volume",
                yaxis="y2",
                hovertemplate="<b>%{x}</b><br>Visits: %{y:,}<extra></extra>",
            ))
            # Night-shift shading — shade first 7 and last 4 bars
            night_indices = list(range(0, 7)) + list(range(20, 24))
            for hr_idx in night_indices:
                if hr_idx < len(hour_labels):
                    fig_inv_hr.add_vrect(
                        x0=hour_labels[hr_idx], x1=hour_labels[hr_idx],
                        fillcolor=AFYA_BLUE, opacity=0.06,
                        layer="below", line_width=0,
                    )
            fig_inv_hr.update_layout(
                height=300,
                margin=dict(l=0, r=60, t=20, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Hour of Day (EAT)", tickangle=-45),
                yaxis=dict(title="Investigation rate %", rangemode="tozero"),
                yaxis2=dict(title="Visit volume", overlaying="y", side="right",
                            rangemode="tozero", showgrid=False),
                legend=dict(orientation="h", y=1.07, xanchor="right", x=1),
            )
            _pc(fig_inv_hr)

            # Flag any off-peak hours with above-average rate
            overall_avg = float(df_inv_hr["inv_rate_pct"].mean())
            night_df = df_inv_hr[
                (df_inv_hr["hour_eat"] >= 20) | (df_inv_hr["hour_eat"] <= 6)
            ]
            if not night_df.empty:
                night_avg = float(night_df["inv_rate_pct"].mean())
                diff = night_avg - overall_avg
                if diff > 5:
                    _note(
                        f"Night-shift investigation rate ({night_avg:.1f}%) is "
                        f"{diff:.1f} percentage points above the all-hours average ({overall_avg:.1f}%). "
                        f"This pattern is worth reviewing — it may reflect shift-end ordering to clear queues.",
                        w=True,
                    )
                else:
                    _note(
                        f"Night-shift investigation rate ({night_avg:.1f}%) is close to "
                        f"the all-hours average ({overall_avg:.1f}%) — no clear over-ordering signal."
                    )
        else:
            _note("No investigation data for the selected period.", w=True)
    except Exception as e:
        st.warning(f"Off-peak investigation pattern: {e}")

    # E3 drill-down: IP vs OP split + top investigation types
    _gap(8)
    c1, c2 = st.columns([1, 1.2])

    with c1:
        st.markdown("**Where is it happening — Inpatient or Outpatient?**")
        try:
            df_ipop = Q.load_offpeak_ipop_split(filters, run_query)
            if not df_ipop.empty:
                df_ipop["inv_count"]   = pd.to_numeric(df_ipop["inv_count"],   errors="coerce")
                df_ipop["visit_count"] = pd.to_numeric(df_ipop["visit_count"], errors="coerce")
                fig_ipop = go.Figure()
                for vtype, color in [("Outpatient", TEAL), ("Inpatient", AFYA_BLUE)]:
                    sub = df_ipop[df_ipop["visit_type"] == vtype]
                    if not sub.empty:
                        fig_ipop.add_trace(go.Bar(
                            name=vtype,
                            x=sub["shift_type"],
                            y=sub["inv_count"],
                            marker_color=color,
                            text=[f"{int(float(v)):,}" if pd.notna(v) else "" for v in sub["inv_count"]],
                            textposition="outside",
                            hovertemplate=f"<b>{vtype}</b><br>%{{x}}<br>Investigations: %{{y:,}}<extra></extra>",
                        ))
                fig_ipop.update_layout(
                    barmode="group", height=300,
                    margin=dict(l=0, r=20, t=20, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(title="Investigations ordered", rangemode="tozero"),
                    legend=dict(orientation="h", y=1.08, xanchor="right", x=1),
                )
                _pc(fig_ipop)
            else:
                _note("No IP/OP split data available.", w=True)
        except Exception as e:
            st.warning(f"IP/OP split: {e}")

    with c2:
        st.markdown("**Which procedure types are being ordered off-peak?**")
        try:
            df_types = Q.load_offpeak_top_investigations(filters, run_query)
            if not df_types.empty:
                df_types["inv_count"] = pd.to_numeric(df_types["inv_count"], errors="coerce")
                top5_discs = (
                    df_types.groupby("discipline")["inv_count"]
                    .sum().nlargest(5).index.tolist()
                )
                df_types  = df_types[df_types["discipline"].isin(top5_discs)]
                disc_order = (
                    df_types.groupby("discipline")["inv_count"]
                    .sum().sort_values().index.tolist()
                )
                fig_types = go.Figure()
                for vtype, color in [("Outpatient", TEAL), ("Inpatient", AFYA_BLUE)]:
                    sub = df_types[df_types["visit_type"] == vtype]
                    if sub.empty:
                        continue
                    sub = sub.set_index("discipline").reindex(disc_order).reset_index()
                    fig_types.add_trace(go.Bar(
                        name=vtype,
                        y=sub["discipline"],
                        x=sub["inv_count"],
                        orientation="h",
                        marker_color=color,
                        text=[f"{int(float(v)):,}" if pd.notna(v) else "" for v in sub["inv_count"]],
                        textposition="outside",
                        hovertemplate=f"<b>%{{y}}</b><br>{vtype}: %{{x:,}} off-peak<extra></extra>",
                    ))
                fig_types.update_layout(
                    barmode="group",
                    height=320,
                    margin=dict(l=0, r=80, t=20, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(title="Investigations ordered (off-peak)", rangemode="tozero"),
                    legend=dict(orientation="h", y=1.08, xanchor="right", x=1),
                )
                _pc(fig_types)
            else:
                _note("No off-peak investigation disciplines found.", w=True)
        except Exception as e:
            st.warning(f"Top investigations: {e}")

    _gap(12)

    # E4: Discharge timing vs peak admission hours
    st.markdown("**E4 — When Do Discharges Happen? Does It Coincide with Admission Peaks?**")
    _note(
        "Based on actual discharge timestamps. "
        "A discharge cluster during peak admission hours creates a bottleneck — beds "
        "are unavailable precisely when new admissions are arriving."
    )
    try:
        df_disc = Q.load_discharge_timing(filters, run_query)
        df_hm_adm = Q.load_peak_demand_heatmap(filters, run_query)

        if not df_disc.empty:
            for _c in ("discharge_count", "day_num", "hour_eat"):
                if _c in df_disc.columns:
                    df_disc[_c] = pd.to_numeric(df_disc[_c], errors="coerce")

            day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

            c_left, c_right = st.columns([1, 1])
            with c_left:
                st.markdown("<small>**Discharges by hour × day**</small>",
                            unsafe_allow_html=True)
                disc_pivot = df_disc.pivot_table(
                    index="day_name", columns="hour_eat",
                    values="discharge_count", aggfunc="sum", fill_value=0
                ).reindex(day_order)

                fig_disc_hm = go.Figure(go.Heatmap(
                    z=disc_pivot.values,
                    x=disc_pivot.columns.tolist(),
                    y=disc_pivot.index.tolist(),
                    colorscale=[[0, "#F0F4FF"], [1, AFYA_BLUE]],
                    hovertemplate="<b>%{y} %{x}:00</b><br>Discharge events: %{z:,}<extra></extra>",
                    colorbar=dict(thickness=12),
                    text=disc_pivot.values,
                    texttemplate="%{text}",
                    textfont=dict(size=9),
                ))
                fig_disc_hm.update_layout(
                    height=240, margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(title="Hour of Day (EAT)"),
                    yaxis=dict(title="Day", autorange="reversed"),
                )
                _pc(fig_disc_hm)

            with c_right:
                # Hourly discharge count across all days
                disc_by_hour = (
                    df_disc.groupby("hour_eat")["discharge_count"].sum().reset_index()
                )
                fig_disc_hr = go.Figure()
                fig_disc_hr.add_trace(go.Bar(
                    x=disc_by_hour["hour_eat"],
                    y=disc_by_hour["discharge_count"],
                    name="Discharge events",
                    marker_color=TEAL,
                    hovertemplate="<b>%{x}:00</b><br>Discharges: %{y:,}<extra></extra>",
                ))

                # Overlay admission peak hours from heatmap if available
                if not df_hm_adm.empty and "hour_of_day" in df_hm_adm.columns:
                    adm_by_hour = (
                        df_hm_adm.groupby("hour_of_day")["inpatient_count"]
                        .sum().reset_index()
                    )
                    peak_adm_hr = int(adm_by_hour.loc[adm_by_hour["inpatient_count"].idxmax(),
                                                       "hour_of_day"])
                    fig_disc_hr.add_vline(
                        x=peak_adm_hr,
                        line=dict(color=CORAL, width=2, dash="dash"),
                        annotation_text=f"Peak admissions ({peak_adm_hr:02d}:00)",
                        annotation_font=dict(size=9, color=CORAL),
                        annotation_position="top right",
                    )

                fig_disc_hr.update_layout(
                    height=240,
                    margin=dict(l=0, r=20, t=20, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(title="Hour (EAT)", dtick=2),
                    yaxis=dict(title="Discharge events", rangemode="tozero"),
                    showlegend=False,
                )
                _pc(fig_disc_hr)

            # Insight: check if discharge peak overlaps with admission peak
            disc_peak_hr = int(disc_by_hour.loc[disc_by_hour["discharge_count"].idxmax(), "hour_eat"])
            _peak_adm_hr = locals().get("peak_adm_hr", None)
            _note(
                f"Discharge peak: {disc_peak_hr:02d}:00 EAT. "
                + (f"This coincides with peak admission hours ({_peak_adm_hr:02d}:00) — "
                   f"bed turnover is being compressed at the moment of highest demand."
                   if _peak_adm_hr is not None and abs(disc_peak_hr - _peak_adm_hr) <= 3 else
                   (f"Discharge activity ({disc_peak_hr:02d}:00) and admission peaks "
                    f"({_peak_adm_hr:02d}:00) appear offset — bed turnover is not a direct bottleneck."
                    if _peak_adm_hr is not None else
                    f"Discharge peak is at {disc_peak_hr:02d}:00 EAT."))
            )
        else:
            _note("No completed admission records available to estimate discharge timing.", w=True)
    except Exception as e:
        st.warning(f"Discharge timing: {e}")

    _gap(12)

    # ── SECTION F: FORECAST ───────────────────────────────────────────────
    _sh("F — Where is patient volume headed in the next 6 months?", mt=8)
    _note(
        "Base scenario is the model midline. Conservative and stretch scenarios represent "
        "the forecast range. Use the base scenario for planning and the range to "
        "stress-test capacity assumptions."
    )
    try:
        df = Q.load_encounter_forecast(filters, run_query)
        if not df.empty:
            df["visit_month"] = pd.to_datetime(df["visit_month"])
            for _c in ("actual_visits", "actual_outpatient", "actual_inpatient"):
                df[_c] = pd.to_numeric(df[_c], errors="coerce")

            last_month = df["visit_month"].max()
            mo_display = filters.get("months_back") or {
                "Last 12 months": 12, "Last 6 months": 6, "Last 90 days": 3,
            }.get(filters.get("date_range", "Last 12 months"), 12)
            cutoff = last_month - pd.DateOffset(months=mo_display - 1)
            df_act = df[df["visit_month"] >= cutoff].copy()

            median_vol = df["actual_visits"].median()
            df_valid = df[df["actual_visits"] > median_vol * 0.1]
            last3 = df_valid.tail(3) if len(df_valid) >= 3 else df.tail(3)

            op_trend = float(last3["actual_outpatient"].mean())
            ip_trend = float(last3["actual_inpatient"].mean())
            op_std   = float(df_valid["actual_outpatient"].std() or op_trend * 0.1)
            ip_std   = float(df_valid["actual_inpatient"].std()  or ip_trend * 0.1)

            n_fut = 6
            future_months = [last_month + pd.DateOffset(months=i) for i in range(1, n_fut + 1)]

            panel_defs = [
                {
                    "act_col":      "actual_outpatient",
                    "trend":        op_trend,
                    "std":          op_std,
                    "capacity":     float(df["actual_outpatient"].mean()),
                    "color":        AFYA_BLUE,
                    "title":        "Outpatient Encounter Forecast",
                    "xref":         "x",
                    "yref":         "y",
                    "ydomref":      "y domain",
                },
                {
                    "act_col":      "actual_inpatient",
                    "trend":        ip_trend,
                    "std":          ip_std,
                    "capacity":     float(df["actual_inpatient"].mean()),
                    "color":        TEAL,
                    "title":        "Inpatient Encounter Forecast",
                    "xref":         "x2",
                    "yref":         "y2",
                    "ydomref":      "y2 domain",
                },
            ]

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Outpatient Encounter Forecast",
                                 "Inpatient Encounter Forecast"],
                horizontal_spacing=0.10,
            )

            for col_idx, pd_ in enumerate(panel_defs, start=1):
                show_leg  = col_idx == 1
                act_col   = pd_["act_col"]
                color     = pd_["color"]
                xref      = pd_["xref"]
                yref      = pd_["yref"]
                ydomref   = pd_["ydomref"]
                trend     = pd_["trend"]
                std_      = pd_["std"]
                cap_val   = pd_["capacity"]
                conservative = max(0.0, trend - std_)
                stretch      = trend + std_

                last_actual_y = float(df_act[act_col].iloc[-1]) if not df_act.empty else trend

                # ── Actuals ──────────────────────────────────────────
                fig.add_trace(go.Scatter(
                    x=df_act["visit_month"], y=df_act[act_col],
                    name="Actuals (12 mo.)", mode="lines",
                    line=dict(color=TEAL, width=2.5),
                    showlegend=show_leg,
                ), row=1, col=col_idx)

                # ── Forecast-start vertical line ──────────────────────
                fig.add_shape(
                    type="line",
                    xref=xref, yref=ydomref,
                    x0=last_month, x1=last_month, y0=0, y1=1,
                    line=dict(color=PURPLE, width=1.5, dash="dash"),
                )
                fig.add_annotation(
                    xref=xref, yref=ydomref,
                    x=last_month, y=0.97,
                    text="Forecast starts here",
                    showarrow=False,
                    font=dict(size=8, color=PURPLE),
                    xanchor="right",
                )

                # ── Capacity reference line ───────────────────────────
                x_start = df_act["visit_month"].min() if not df_act.empty else last_month
                fig.add_shape(
                    type="line",
                    xref=xref, yref=yref,
                    x0=x_start, x1=future_months[-1],
                    y0=cap_val, y1=cap_val,
                    line=dict(color=GRAY, width=1.5, dash="dot"),
                )
                fig.add_annotation(
                    xref=xref, yref=yref,
                    x=x_start, y=cap_val,
                    text=f"  Capacity baseline — to be confirmed ({int(cap_val):,})",
                    xanchor="left", yanchor="bottom",
                    showarrow=False,
                    font=dict(size=8, color=GRAY),
                )

                # ── Three scenario lines ──────────────────────────────
                scenario_specs = [
                    ("Conservative", conservative, "dash",  1.5, 0.55),
                    ("Base",          trend,        "solid", 2.5, 1.0),
                    ("Stretch",       stretch,      "dash",  1.5, 0.55),
                ]
                for sc_name, sc_val, sc_dash, sc_w, sc_op in scenario_specs:
                    sc_color = color if sc_name == "Base" else MUTED
                    # Bridge from last actual to first forecast point
                    fig.add_trace(go.Scatter(
                        x=[last_month, future_months[0]],
                        y=[last_actual_y, sc_val],
                        mode="lines",
                        line=dict(color=sc_color, width=1, dash="dot"),
                        showlegend=False, hoverinfo="skip",
                    ), row=1, col=col_idx)
                    fig.add_trace(go.Scatter(
                        x=future_months,
                        y=[sc_val] * n_fut,
                        name=sc_name,
                        mode="lines",
                        line=dict(color=sc_color, width=sc_w, dash=sc_dash),
                        opacity=sc_op,
                        showlegend=show_leg,
                        hovertemplate=f"{sc_name}: %{{y:,.0f}}<extra></extra>",
                    ), row=1, col=col_idx)
                    # End-of-line label
                    fig.add_annotation(
                        xref=xref, yref=yref,
                        x=future_months[-1], y=sc_val,
                        text=f"  {sc_name} ({int(sc_val):,})",
                        xanchor="left", yanchor="middle",
                        showarrow=False,
                        font=dict(size=8,
                                  color=color if sc_name == "Base" else MUTED),
                    )

                # ── Capacity pressure callout ─────────────────────────
                if trend > cap_val * 1.05:
                    crossing_month = future_months[0].strftime("%b %Y")
                    fig.add_annotation(
                        xref=xref, yref=yref,
                        x=future_months[0], y=trend,
                        text=f"⚠ Projected capacity pressure — {crossing_month}",
                        showarrow=True, arrowhead=2, arrowcolor=CORAL,
                        font=dict(size=8, color=CORAL),
                        ax=0, ay=-35,
                    )

            fig.update_layout(
                height=400,
                margin=dict(l=0, r=130, t=55, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.06,
                    xanchor="right", x=1,
                    font=dict(size=10),
                    bgcolor="rgba(0,0,0,0)",
                ),
            )
            for c in [1, 2]:
                fig.update_yaxes(
                    title_text="Visits", row=1, col=c,
                    showgrid=True, gridcolor="#EBF3FB", zeroline=False,
                    tickfont=dict(color=MUTED, size=10),
                )
            fig.update_xaxes(
                showgrid=False, tickfont=dict(color=MUTED, size=10),
                tickformat="%b %Y",
            )
            _pc(fig)
            _note(
                "Solid teal line = last 12 months actuals. "
                "Dashed lines = 6-month forecast scenarios."
            )

            # ── Four KPI cards ────────────────────────────────────────
            latest = df_act.iloc[-1] if not df_act.empty else df.iloc[-1]
            op_cons = int(max(0, op_trend - op_std))
            op_base = int(op_trend)
            op_str  = int(op_trend + op_std)
            ip_cons = int(max(0, ip_trend - ip_std))
            ip_base = int(ip_trend)
            ip_str  = int(ip_trend + ip_std)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                _kpi("Last Month Outpatient", _n(latest.get("actual_outpatient")))
            with c2:
                _kpi("Last Month Inpatient",  _n(latest.get("actual_inpatient")),
                     color=TEAL)
            with c3:
                _kpi("Next Month OP Forecast", _n(op_base),
                     s=f"Range: {op_cons:,} – {op_str:,}",
                     color=AFYA_BLUE)
            with c4:
                _kpi("Next Month IP Forecast", _n(ip_base),
                     s=f"Range: {ip_cons:,} – {ip_str:,}",
                     color=TEAL)
    except Exception as e:
        st.warning(f"Forecast: {e}")

    _gap(12)



# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PATIENT DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════════════════════

def render_tab2_segmentation(filters: dict, run_query):
    # ── KPI ROW ───────────────────────────────────────────────────────────
    _sh("Patient Overview")
    try:
        df = Q.load_seg_kpis(filters, run_query)
        if not df.empty:
            row = df.iloc[0]
            c1,c2,c3,c4,c5 = st.columns(5)
            with c1: _kpi("Total Patients",  _n(row.get("total_patients")))
            with c2: _kpi("Chronic",         _n(row.get("chronic_patients")),
                          str(_p(row.get("chronic_rate_pct")) or "—") + " of patients",
                          "#854F0B")
            with c3: _kpi("Repeat Patients", _n(row.get("repeat_patients")),
                          str(_p(row.get("repeat_rate_pct")) or "—") + " repeat rate",
                          "#0F6E56")
            with c4: _kpi("Single Visit",    _n(row.get("single_visit")),
                          f'{_p(100 - float(row.get("repeat_rate_pct") or 0), 1)} of patients',
                          GRAY)
            with c5: _kpi("Avg Visits / Pt", str(row.get("avg_visits", "—")),
                          "per patient", GRAY)
    except Exception as e:
        st.warning(f"Seg KPIs: {e}")

    _gap(16)

    # ── A: AGE GROUP & GENDER DISTRIBUTION ───────────────────────────────
    _sh("A — Age group & gender distribution", mt=8)
    try:
        df = Q.load_demographics_age_sex(filters, run_query)
        if not df.empty:
            gender_filter = st.radio(
                "Filter by gender", ["All", "Female", "Male"],
                horizontal=True, key="demo_gender_filter",
            )
            df_f = df.copy()
            if gender_filter == "Female":
                df_f = df_f[df_f["sex"].isin(["F", "FEMALE"])]
            elif gender_filter == "Male":
                df_f = df_f[df_f["sex"].isin(["M", "MALE"])]

            c1, c2, c3 = st.columns(3)

            # Sort order by volume — shared across panels 1 and 2
            _age_ord = (df_f.groupby("age_group")["total"]
                        .sum().sort_values(ascending=False).index.tolist())

            with c1:
                st.markdown(
                    '<div style="font-size:11px;font-weight:500;color:#1a1a18;margin-bottom:2px;">Patients by age group</div>'
                    '<div style="font-size:10px;color:#888780;margin-bottom:8px;">Volume per cohort</div>',
                    unsafe_allow_html=True)
                _av = (df_f.groupby("age_group")["total"].sum()
                       .reindex(_age_ord).reset_index())
                fig_av = go.Figure(go.Bar(
                    x=_av["total"], y=_av["age_group"], orientation="h",
                    marker_color="#378ADD", marker_line_width=0,
                    hovertemplate="<b>%{y}</b><br>%{x:,} patients<extra></extra>",
                ))
                fig_av.update_layout(
                    height=220, margin=dict(l=0, r=10, t=4, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)",
                               nticks=4, tickfont=dict(size=9, color="#888780")),
                    yaxis=dict(showgrid=False, tickfont=dict(size=9, color="#888780")),
                )
                _pc(fig_av)

            with c2:
                st.markdown(
                    '<div style="font-size:11px;font-weight:500;color:#1a1a18;margin-bottom:2px;">Chronic vs non-chronic by age</div>'
                    '<div style="font-size:10px;color:#888780;margin-bottom:4px;">Which age groups carry the highest chronic burden?</div>',
                    unsafe_allow_html=True)
                st.markdown(
                    '<div style="display:flex;gap:10px;margin-bottom:6px;">'
                    '<span style="display:flex;align-items:center;gap:4px;font-size:10px;color:#5f5e5a;">'
                    '<span style="width:10px;height:10px;border-radius:2px;background:#E24B4A;display:inline-block;"></span>Chronic</span>'
                    '<span style="display:flex;align-items:center;gap:4px;font-size:10px;color:#5f5e5a;">'
                    '<span style="width:10px;height:10px;border-radius:2px;background:#D3D1C7;display:inline-block;"></span>Non-chronic</span>'
                    '</div>',
                    unsafe_allow_html=True)
                _ch = (df_f.groupby("age_group")
                       .agg(chronic=("chronic", "sum"), non_chronic=("non_chronic", "sum"))
                       .reindex(_age_ord).reset_index())
                fig_chr = go.Figure()
                fig_chr.add_trace(go.Bar(
                    x=_ch["chronic"], y=_ch["age_group"], orientation="h",
                    name="Chronic", marker_color="#E24B4A", marker_line_width=0,
                    hovertemplate="<b>%{y}</b><br>Chronic: %{x:,}<extra></extra>",
                ))
                fig_chr.add_trace(go.Bar(
                    x=_ch["non_chronic"], y=_ch["age_group"], orientation="h",
                    name="Non-chronic", marker_color="#D3D1C7", marker_line_width=0,
                    hovertemplate="<b>%{y}</b><br>Non-chronic: %{x:,}<extra></extra>",
                ))
                fig_chr.update_layout(
                    barmode="stack", barnorm="percent",
                    height=200, margin=dict(l=0, r=10, t=4, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    showlegend=False,
                    xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)",
                               ticksuffix="%", nticks=4,
                               tickfont=dict(size=9, color="#888780")),
                    yaxis=dict(showgrid=False, tickfont=dict(size=9, color="#888780")),
                )
                _pc(fig_chr)
                _ch["chronic"]     = pd.to_numeric(_ch["chronic"],     errors="coerce")
                _ch["non_chronic"] = pd.to_numeric(_ch["non_chronic"], errors="coerce")
                _ch["pct"] = (_ch["chronic"]
                              / (_ch["chronic"] + _ch["non_chronic"]).replace(0, float("nan"))
                              * 100)
                _top2 = _ch.dropna(subset=["pct"]).nlargest(2, "pct")
                if len(_top2) >= 2:
                    _t1, _t2 = _top2.iloc[0], _top2.iloc[1]
                    st.markdown(
                        f'<div style="background:#f5f5f3;border-radius:8px;padding:6px 9px;'
                        f'font-size:10px;color:#5f5e5a;margin-top:6px;line-height:1.5;">'
                        f'<strong style="color:#1a1a18;">{_t1["age_group"]}</strong> and '
                        f'<strong style="color:#1a1a18;">{_t2["age_group"]}</strong> carry the '
                        f'highest chronic burden at '
                        f'<strong style="color:#1a1a18;">{_t1["pct"]:.0f}%</strong> and '
                        f'<strong style="color:#1a1a18;">{_t2["pct"]:.0f}%</strong> respectively</div>',
                        unsafe_allow_html=True)

            with c3:
                st.markdown(
                    '<div style="font-size:11px;font-weight:500;color:#1a1a18;margin-bottom:2px;">Gender distribution</div>'
                    '<div style="font-size:10px;color:#888780;margin-bottom:8px;">Overall split across all visits</div>',
                    unsafe_allow_html=True)
                _gt = df.groupby("sex")["total"].sum().reset_index()
                _gt["sex_clean"] = _gt["sex"].map(
                    {"F": "Female", "FEMALE": "Female", "M": "Male", "MALE": "Male"}
                ).fillna("Other")
                _ga = _gt.groupby("sex_clean")["total"].sum().reset_index()
                _pc(donut(
                    labels=_ga["sex_clean"].tolist(),
                    values=_ga["total"].tolist(),
                    color_map={"Female": "#7F77DD", "Male": "#1D9E75"},
                    height=200,
                ))
                _g_tot = float(_ga["total"].sum() or 1)
                _fem = float(_ga.loc[_ga["sex_clean"] == "Female", "total"].sum())
                _fem_pct = round(_fem / _g_tot * 100, 1)
                st.markdown(
                    f'<div style="background:#f5f5f3;border-radius:8px;padding:6px 9px;'
                    f'font-size:10px;color:#5f5e5a;margin-top:6px;line-height:1.5;">'
                    f'Female patients make up <strong style="color:#1a1a18;">{_fem_pct}%</strong> '
                    f'of visits. Female chronic patients are proportionally higher — '
                    f'driven by maternal and NCD conditions.</div>',
                    unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Demographics: {e}")

    _gap(12)

    # ── B: AGE COHORT GROWTH & PATIENT MIX ──────────────────────────────────
    _sh("B — Age cohort growth and patient mix", mt=8)
    st.markdown(
        '<div style="font-size:11px;color:#6b7280;margin-top:-4px;margin-bottom:10px;">'
        'Left: growth index — each cohort normalised to 100 at its first recorded month '
        '&nbsp;·&nbsp; Right: patient mix share each month</div>',
        unsafe_allow_html=True,
    )
    try:
        df = Q.load_cohort_forecast(filters, run_query)
        if not df.empty:
            pivot = df.pivot_table(index="visit_month", columns="age_cohort",
                                   values="patient_count", aggfunc="sum", fill_value=0)
            _cohort_colors = {
                "Senior (65+)":        "#888780",
                "Older Adult (55–64)": "#D85A30",
                "Middle Age (45–54)":  "#EF9F27",
                "Adult (35–44)":       "#1D9E75",
                "Young Adult (25–34)": "#378ADD",
                "Youth (18–24)":       "#7F77DD",
                "Adolescent (13–17)":  "#2D2D2A",
                "Child (5–12)":        "#9B59B6",
                "Toddler (0–4)":       "#D4537E",
            }
            cohorts = [c for c in _cohort_colors if c in pivot.columns]
            months  = pivot.index.tolist()

            _col_l, _col_r = st.columns([1, 1])

            # ── LEFT: Growth index ─────────────────────────────────────────
            with _col_l:
                idx_df = pivot[cohorts].copy().astype(float)
                for _c in cohorts:
                    _s = idx_df[_c]
                    _first = _s[_s > 0].iloc[0] if (_s > 0).any() else None
                    idx_df[_c] = (_s / _first * 100).where(_s > 0, other=None) if _first else None

                _growth_pct = {}
                for _c in cohorts:
                    _sv = idx_df[_c].dropna()
                    _growth_pct[_c] = round(float(_sv.iloc[-1]) - 100, 1) if len(_sv) >= 2 else 0.0

                # Legend with growth %
                _leg_l = '<div style="display:flex;flex-wrap:wrap;gap:5px 9px;margin-bottom:5px;">'
                for _c in cohorts:
                    _g = _growth_pct.get(_c, 0)
                    _gc = "#1D9E75" if _g >= 0 else "#E24B4A"
                    _gs = "+" if _g >= 0 else ""
                    _leg_l += (
                        f'<span style="display:flex;align-items:center;gap:3px;font-size:10px;color:#374151;">'
                        f'<span style="width:8px;height:8px;border-radius:50%;'
                        f'background:{_cohort_colors[_c]};flex-shrink:0;display:inline-block;"></span>'
                        f'{_c.split(" (")[0]}&nbsp;<span style="color:{_gc};font-weight:600;">{_gs}{_g}%</span>'
                        f'</span>'
                    )
                _leg_l += '</div>'
                st.markdown(_leg_l, unsafe_allow_html=True)

                fig_idx = go.Figure()
                for _c in cohorts:
                    fig_idx.add_trace(go.Scatter(
                        x=months, y=idx_df[_c].tolist(),
                        mode="lines", name=_c,
                        line=dict(color=_cohort_colors[_c], width=2),
                        connectgaps=False, showlegend=False,
                        hovertemplate=f"<b>{_c}</b><br>%{{x|%b %Y}}: %{{y:.1f}}<extra></extra>",
                    ))
                fig_idx.add_hline(y=100, line_dash="dot", line_color="#cccccc", line_width=1)
                fig_idx.update_layout(
                    height=220,
                    margin=dict(l=0, r=10, t=6, b=30),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(showgrid=False, tickformat="%b %y", tickangle=-30, title=None),
                    yaxis=dict(title="Index", showgrid=True, gridcolor="#f0f0f0", zeroline=False),
                )
                _pc(fig_idx)

                # Insight strip
                if _growth_pct:
                    _fast = max(_growth_pct, key=_growth_pct.get)
                    _fg   = _growth_pct[_fast]
                    _fgs  = "+" if _fg >= 0 else ""
                    st.markdown(
                        f'<div style="background:#f0f9f4;border-left:3px solid #1D9E75;'
                        f'padding:7px 10px;border-radius:0 6px 6px 0;font-size:11px;'
                        f'color:#374151;margin-top:6px;">'
                        f'<b>{_fast.split(" (")[0]}</b> is the fastest-growing cohort '
                        f'({_fgs}{_fg}% vs. baseline)</div>',
                        unsafe_allow_html=True,
                    )

            # ── RIGHT: Patient mix (100% stacked bar) ─────────────────────
            with _col_r:
                _row_totals = pivot[cohorts].sum(axis=1).replace(0, float("nan"))
                _share_df   = pivot[cohorts].div(_row_totals, axis=0) * 100

                # Legend (no growth %)
                _leg_r = '<div style="display:flex;flex-wrap:wrap;gap:5px 9px;margin-bottom:5px;">'
                for _c in cohorts:
                    _leg_r += (
                        f'<span style="display:flex;align-items:center;gap:3px;font-size:10px;color:#374151;">'
                        f'<span style="width:8px;height:8px;border-radius:50%;'
                        f'background:{_cohort_colors[_c]};flex-shrink:0;display:inline-block;"></span>'
                        f'{_c.split(" (")[0]}'
                        f'</span>'
                    )
                _leg_r += '</div>'
                st.markdown(_leg_r, unsafe_allow_html=True)

                fig_mix = go.Figure()
                for _c in reversed(cohorts):
                    fig_mix.add_trace(go.Bar(
                        x=months, y=_share_df[_c].tolist(),
                        name=_c, marker_color=_cohort_colors[_c],
                        showlegend=False,
                        hovertemplate=f"<b>{_c}</b><br>%{{x|%b %Y}}: %{{y:.1f}}%<extra></extra>",
                    ))
                fig_mix.update_layout(
                    barmode="stack",
                    height=220,
                    margin=dict(l=0, r=10, t=6, b=30),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(showgrid=False, tickformat="%b %y", tickangle=-30, title=None),
                    yaxis=dict(title="Share", range=[0, 100], showgrid=True,
                               gridcolor="#f0f0f0", ticksuffix="%"),
                )
                _pc(fig_mix)

                # Insight strip: dominant cohort in latest month
                if not _share_df.empty and months:
                    _latest = _share_df.iloc[-1]
                    _dom     = _latest.idxmax()
                    _dom_pct = round(float(_latest.max()), 1)
                    _lm      = pd.to_datetime(months[-1]).strftime("%b %Y")
                    st.markdown(
                        f'<div style="background:#eff6ff;border-left:3px solid #378ADD;'
                        f'padding:7px 10px;border-radius:0 6px 6px 0;font-size:11px;'
                        f'color:#374151;margin-top:6px;">'
                        f'In {_lm}, <b>{_dom.split(" (")[0]}</b> is the largest cohort '
                        f'at <b>{_dom_pct}%</b> of total patients</div>',
                        unsafe_allow_html=True,
                    )
    except Exception as e:
        st.warning(f"Cohort growth: {e}")

    _gap(12)

    # ── C: NEW VS RETURNING — VOLUMES, AGE & VISIT TYPE ──────────────────
    _sh("C — New vs returning patient trends", mt=8)
    try:
        df_nvr = Q.load_new_vs_returning(filters, run_query)
        if not df_nvr.empty:
            for _c in ("total_patients", "new_patients", "returning_patients"):
                df_nvr[_c] = pd.to_numeric(df_nvr[_c], errors="coerce")
            df_nvr = df_nvr.sort_values("visit_month").reset_index(drop=True)
            _rolling3 = df_nvr["new_patients"].rolling(3, min_periods=3).mean()

            _c1, _c2 = st.columns(2)

            # ── Row 1 Left: Acquisition bar + rolling avg line ────────────
            with _c1:
                st.markdown(
                    '<div style="font-size:11px;font-weight:500;color:#1a1a18;margin-bottom:2px;">New patient acquisition trend</div>'
                    '<div style="font-size:10px;color:#888780;margin-bottom:4px;">Monthly new patients with 3-month rolling average</div>',
                    unsafe_allow_html=True)
                st.markdown(
                    '<div style="display:flex;gap:10px;margin-bottom:6px;">'
                    '<span style="display:flex;align-items:center;gap:4px;font-size:10px;color:#5f5e5a;">'
                    '<span style="width:10px;height:10px;border-radius:2px;background:rgba(239,159,39,0.55);display:inline-block;"></span>New patients</span>'
                    '<span style="display:flex;align-items:center;gap:6px;font-size:10px;color:#5f5e5a;">'
                    '<span style="display:inline-block;width:18px;border-top:2px dashed #378ADD;"></span>3-mo avg</span>'
                    '</div>', unsafe_allow_html=True)
                fig_acq = go.Figure()
                fig_acq.add_trace(go.Bar(
                    x=df_nvr["visit_month"], y=df_nvr["new_patients"],
                    name="New", marker_color="rgba(239,159,39,0.55)", marker_line_width=0,
                    hovertemplate="<b>%{x|%b %Y}</b><br>New: %{y:,}<extra></extra>",
                ))
                fig_acq.add_trace(go.Scatter(
                    x=df_nvr["visit_month"], y=_rolling3,
                    mode="lines", name="3-mo avg",
                    line=dict(color="#378ADD", width=2, dash="dash"),
                    connectgaps=True,
                    hovertemplate="<b>%{x|%b %Y}</b><br>3-mo avg: %{y:.0f}<extra></extra>",
                ))
                fig_acq.update_layout(
                    height=180, margin=dict(l=0, r=10, t=4, b=30),
                    plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                    xaxis=dict(showgrid=False, tickformat="%b %y", tickangle=-30,
                               tickfont=dict(size=9, color="#888780")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)",
                               tickfont=dict(size=9, color="#888780"), rangemode="tozero"),
                )
                _pc(fig_acq)
                _nvr_v = df_nvr.dropna(subset=["new_patients"])
                if len(_nvr_v) > 0:
                    _mi = int(_nvr_v["new_patients"].idxmin())
                    _mm = pd.to_datetime(_nvr_v.loc[_mi, "visit_month"]).strftime("%b %Y")
                    _mn = int(_nvr_v.loc[_mi, "new_patients"])
                    _ra = _rolling3.iloc[_mi] if _mi < len(_rolling3) else None
                    if _ra and not pd.isna(_ra) and _ra > 0:
                        _bp = round((1 - _mn / _ra) * 100)
                        _ret_m = int(_nvr_v.loc[_mi, "returning_patients"] or 0)
                        _ins_txt = ("Returning patients partially compensated."
                                    if _ret_m > _mn else "Returning patients also declined that month.")
                        _acq_ins = (f"{_mm} dip: <strong style='color:#1a1a18;'>{_mn:,} new patients</strong>"
                                    f" — {_bp}% below rolling average. {_ins_txt}")
                    else:
                        _acq_ins = f"Lowest month: <strong style='color:#1a1a18;'>{_mm}</strong> with {_mn:,} new patients."
                    st.markdown(
                        f'<div style="background:#f5f5f3;border-radius:8px;padding:6px 9px;'
                        f'font-size:10px;color:#5f5e5a;margin-top:6px;line-height:1.5;">{_acq_ins}</div>',
                        unsafe_allow_html=True)

            # ── Row 1 Right: Dual line new vs returning ───────────────────
            with _c2:
                st.markdown(
                    '<div style="font-size:11px;font-weight:500;color:#1a1a18;margin-bottom:2px;">New vs returning volumes</div>'
                    '<div style="font-size:10px;color:#888780;margin-bottom:4px;">Are they moving together or diverging?</div>',
                    unsafe_allow_html=True)
                st.markdown(
                    '<div style="display:flex;gap:10px;margin-bottom:6px;">'
                    '<span style="display:flex;align-items:center;gap:4px;font-size:10px;color:#5f5e5a;">'
                    '<span style="width:8px;height:8px;border-radius:50%;background:#1D9E75;display:inline-block;"></span>New</span>'
                    '<span style="display:flex;align-items:center;gap:4px;font-size:10px;color:#5f5e5a;">'
                    '<span style="width:8px;height:8px;border-radius:50%;background:#7F77DD;display:inline-block;"></span>Returning</span>'
                    '</div>', unsafe_allow_html=True)
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(
                    x=df_nvr["visit_month"], y=df_nvr["new_patients"],
                    mode="lines", name="New",
                    line=dict(color="#1D9E75", width=2), connectgaps=False,
                    hovertemplate="<b>%{x|%b %Y}</b><br>New: %{y:,}<extra></extra>",
                ))
                fig_vol.add_trace(go.Scatter(
                    x=df_nvr["visit_month"], y=df_nvr["returning_patients"],
                    mode="lines", name="Returning",
                    line=dict(color="#7F77DD", width=2), connectgaps=False,
                    hovertemplate="<b>%{x|%b %Y}</b><br>Returning: %{y:,}<extra></extra>",
                ))
                fig_vol.update_layout(
                    height=180, margin=dict(l=0, r=10, t=4, b=30),
                    plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                    xaxis=dict(showgrid=False, tickformat="%b %y", tickangle=-30,
                               tickfont=dict(size=9, color="#888780")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)",
                               tickfont=dict(size=9, color="#888780"), rangemode="tozero"),
                )
                _pc(fig_vol)
                _nt = df_nvr["new_patients"].sum()
                _rt = df_nvr["returning_patients"].sum()
                if _nt > 0 and _rt > 0:
                    _vol_ins = ("New patients consistently outnumber returning. "
                                if _nt > _rt else "Returning patients outnumber new — strong loyalty signal. ")
                    _cr = df_nvr[["new_patients", "returning_patients"]].corr().iloc[0, 1]
                    _vol_ins += ("New and returning volumes move together — driven by the same seasonal or operational signals."
                                 if _cr > 0.7 else
                                 "New and returning volumes diverge — acquisition and retention have different drivers.")
                    st.markdown(
                        f'<div style="background:#f5f5f3;border-radius:8px;padding:6px 9px;'
                        f'font-size:10px;color:#5f5e5a;margin-top:6px;line-height:1.5;">{_vol_ins}</div>',
                        unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"New vs returning: {e}")

    _gap(10)

    # ── C Row 2: Age profile + care intensity ─────────────────────────────
    try:
        df_dist = Q.load_visit_distribution(filters, run_query)
        if not df_dist.empty:
            for _c in ("patient_count", "visit_count"):
                df_dist[_c] = pd.to_numeric(df_dist[_c], errors="coerce")

            _BM = {
                "Toddler (0–4)":       "Paediatric (<18)",
                "Child (5–12)":        "Paediatric (<18)",
                "Adolescent (13–17)":  "Paediatric (<18)",
                "Youth (18–24)":       "Young Adult (18–34)",
                "Young Adult (25–34)": "Young Adult (18–34)",
                "Adult (35–44)":       "Adult (35–54)",
                "Middle Age (45–54)":  "Adult (35–54)",
                "Older Adult (55–64)": "Senior (55+)",
                "Senior (65+)":        "Senior (55+)",
            }
            df_dist["age_bucket"] = df_dist["age_group"].map(_BM).fillna(df_dist["age_group"])
            _BO = ["Senior (55+)", "Adult (35–54)", "Young Adult (18–34)", "Paediatric (<18)"]

            _c3, _c4 = st.columns(2)

            # ── Row 2 Left: Age profile grouped H-bar ────────────────────
            with _c3:
                st.markdown(
                    '<div style="font-size:11px;font-weight:500;color:#1a1a18;margin-bottom:2px;">Age profile — new vs returning</div>'
                    '<div style="font-size:10px;color:#888780;margin-bottom:4px;">Which age groups are acquiring vs returning?</div>',
                    unsafe_allow_html=True)
                st.markdown(
                    '<div style="display:flex;gap:10px;margin-bottom:6px;">'
                    '<span style="display:flex;align-items:center;gap:4px;font-size:10px;color:#5f5e5a;">'
                    '<span style="width:10px;height:10px;border-radius:2px;background:rgba(29,158,117,0.75);display:inline-block;"></span>New</span>'
                    '<span style="display:flex;align-items:center;gap:4px;font-size:10px;color:#5f5e5a;">'
                    '<span style="width:10px;height:10px;border-radius:2px;background:rgba(127,119,221,0.75);display:inline-block;"></span>Returning</span>'
                    '</div>', unsafe_allow_html=True)
                _anr = (df_dist.groupby(["age_bucket", "patient_type"])["patient_count"]
                        .sum().reset_index())
                fig_anr = go.Figure()
                for _pt, _cl in [("New", "rgba(29,158,117,0.75)"),
                                  ("Returning", "rgba(127,119,221,0.75)")]:
                    _sb = (_anr[_anr["patient_type"] == _pt]
                           .set_index("age_bucket")["patient_count"]
                           .reindex(_BO).reset_index())
                    fig_anr.add_trace(go.Bar(
                        name=_pt, y=_sb["age_bucket"], x=_sb["patient_count"],
                        orientation="h", marker_color=_cl, marker_line_width=0,
                        hovertemplate=f"<b>%{{y}}</b><br>{_pt}: %{{x:,}}<extra></extra>",
                    ))
                fig_anr.update_layout(
                    barmode="group", height=180, margin=dict(l=0, r=10, t=4, b=30),
                    plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                    xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)",
                               nticks=4, tickfont=dict(size=9, color="#888780")),
                    yaxis=dict(showgrid=False, tickfont=dict(size=9, color="#888780")),
                )
                _pc(fig_anr)
                _nrp = _anr.pivot_table(index="age_bucket", columns="patient_type",
                                        values="patient_count", fill_value=0).reset_index()
                if "New" in _nrp.columns and "Returning" in _nrp.columns:
                    _nrp["ret_ratio"] = _nrp["Returning"] / (_nrp["New"] + 0.1)
                    _tr = _nrp.nlargest(2, "ret_ratio")["age_bucket"].tolist()
                    _trs = " and ".join([f'<strong style="color:#1a1a18;">{b}</strong>' for b in _tr])
                    st.markdown(
                        f'<div style="background:#f5f5f3;border-radius:8px;padding:6px 9px;'
                        f'font-size:10px;color:#5f5e5a;margin-top:6px;line-height:1.5;">'
                        f'{_trs} skew toward returning — consistent with chronic disease management.</div>',
                        unsafe_allow_html=True)

            # ── Row 2 Right: Care intensity cards + grouped bar + signal ──
            with _c4:
                st.markdown(
                    '<div style="font-size:11px;font-weight:500;color:#1a1a18;margin-bottom:2px;">Care intensity — inpatient escalation rate</div>'
                    '<div style="font-size:10px;color:#888780;margin-bottom:6px;">Returning patients escalate to inpatient at a higher rate</div>',
                    unsafe_allow_html=True)
                _ip_n = float(df_dist.loc[(df_dist["visit_type"] == "Inpatient") &
                                          (df_dist["patient_type"] == "New"), "patient_count"].sum() or 0)
                _op_n = float(df_dist.loc[(df_dist["visit_type"] == "Outpatient") &
                                          (df_dist["patient_type"] == "New"), "patient_count"].sum() or 0)
                _ip_r = float(df_dist.loc[(df_dist["visit_type"] == "Inpatient") &
                                          (df_dist["patient_type"] == "Returning"), "patient_count"].sum() or 0)
                _op_r = float(df_dist.loc[(df_dist["visit_type"] == "Outpatient") &
                                          (df_dist["patient_type"] == "Returning"), "patient_count"].sum() or 0)
                _rn = _ip_n / (_ip_n + _op_n) * 100 if (_ip_n + _op_n) > 0 else 0
                _rr = _ip_r / (_ip_r + _op_r) * 100 if (_ip_r + _op_r) > 0 else 0
                st.markdown(
                    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;">'
                    f'<div style="background:#E1F5EE;border-radius:8px;padding:8px 10px;">'
                    f'<div style="font-size:10px;font-weight:500;color:#0F6E56;">New patients</div>'
                    f'<div style="font-size:20px;font-weight:500;color:#0F6E56;">{_rn:.1f}%</div>'
                    f'<div style="font-size:10px;color:#0F6E56;">inpatient rate</div>'
                    f'<div style="font-size:10px;color:#0F6E56;margin-top:3px;">{int(_ip_n):,} IP · {int(_op_n):,} OP</div>'
                    f'</div>'
                    f'<div style="background:#FCEBEB;border-radius:8px;padding:8px 10px;">'
                    f'<div style="font-size:10px;font-weight:500;color:#A32D2D;">Returning patients</div>'
                    f'<div style="font-size:20px;font-weight:500;color:#A32D2D;">{_rr:.1f}%</div>'
                    f'<div style="font-size:10px;color:#A32D2D;">inpatient rate</div>'
                    f'<div style="font-size:10px;color:#A32D2D;margin-top:3px;">{int(_ip_r):,} IP · {int(_op_r):,} OP</div>'
                    f'</div></div>',
                    unsafe_allow_html=True)
                fig_care = go.Figure()
                for _pt, _cl in [("New", "rgba(29,158,117,0.75)"),
                                  ("Returning", "rgba(127,119,221,0.75)")]:
                    _iv = _ip_n if _pt == "New" else _ip_r
                    _ov = _op_n if _pt == "New" else _op_r
                    fig_care.add_trace(go.Bar(
                        name=_pt, x=["Outpatient", "Inpatient"], y=[_ov, _iv],
                        marker_color=_cl, marker_line_width=0,
                        hovertemplate=f"<b>%{{x}}</b><br>{_pt}: %{{y:,}}<extra></extra>",
                    ))
                fig_care.update_layout(
                    barmode="group", height=90, margin=dict(l=0, r=10, t=4, b=0),
                    plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                    xaxis=dict(showgrid=False, tickfont=dict(size=9, color="#888780")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)",
                               tickfont=dict(size=9, color="#888780")),
                )
                _pc(fig_care)
                _delta_ip = abs(_rr - _rn)
                st.markdown(
                    f'<div style="background:#FAEEDA;border-left:3px solid #EF9F27;'
                    f'border-radius:0 8px 8px 0;padding:6px 10px;font-size:10px;'
                    f'color:#633806;line-height:1.5;margin-top:8px;">'
                    f'<strong>Chronic disease signal:</strong> Returning patients are '
                    f'<strong>{_delta_ip:.1f}pp</strong> more likely to be admitted. '
                    f'As the returning base grows, inpatient demand will grow disproportionately.</div>',
                    unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Care intensity: {e}")

    _gap(12)

    # ── D: PAYER HABIT SWITCH — NEW → RETURNING ──────────────────────────
    _sh("D — Do payer habits change when new patients return?", mt=8)
    st.markdown(
        '<div style="font-size:10px;color:#5f5e5a;line-height:1.5;margin-bottom:10px;">'
        'Many new patients enter as cash payers during an emergency. If they return on '
        'corporate insurance or NHIF/SHA, your clinical experience converted an emergency '
        'visitor into a long-term enrolled client. The chart maps payer type on first visit '
        '→ payer type on return visit.'
        '</div>',
        unsafe_allow_html=True,
    )
    try:
        df_sk = Q.load_payer_switch_sankey(filters, run_query)
        if not df_sk.empty:
            df_sk["patient_count"] = pd.to_numeric(df_sk["patient_count"], errors="coerce")

            # Build Sankey nodes: source payers (left) → target payers (right)
            payer_types = ["Cash", "NHIF / SHA", "Insurance"]
            source_labels = [f"{p} (first visit)" for p in payer_types]
            target_labels = [f"{p} (returning)" for p in payer_types]
            all_labels = source_labels + target_labels

            node_colors = {
                "Cash":       ORANGE,
                "NHIF / SHA": TEAL,
                "Insurance":  AFYA_BLUE,
            }
            node_color_list = (
                [node_colors.get(p, GRAY) for p in payer_types] +
                [node_colors.get(p, GRAY) for p in payer_types]
            )

            link_sources, link_targets, link_values, link_colors = [], [], [], []
            for _, row in df_sk.iterrows():
                src = row["source_payer"]
                tgt = row["target_payer"]
                cnt = row["patient_count"]
                if src not in payer_types or tgt not in payer_types:
                    continue
                si = payer_types.index(src)           # 0–2 (left side)
                ti = payer_types.index(tgt) + 3       # 3–5 (right side)
                link_sources.append(si)
                link_targets.append(ti)
                link_values.append(float(cnt or 0))
                # Colour by source payer, semi-transparent (rgba required by Plotly Sankey)
                base = node_colors.get(src, GRAY)
                r, g, b = int(base[1:3], 16), int(base[3:5], 16), int(base[5:7], 16)
                link_colors.append(f"rgba({r},{g},{b},0.45)")

            fig_sk = go.Figure(go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=20, thickness=24,
                    label=all_labels,
                    color=node_color_list,
                    hovertemplate="%{label}<br>%{value:,} patients<extra></extra>",
                ),
                link=dict(
                    source=link_sources,
                    target=link_targets,
                    value=link_values,
                    color=link_colors,
                    hovertemplate=(
                        "%{source.label} → %{target.label}<br>"
                        "%{value:,} patients<extra></extra>"
                    ),
                ),
            ))
            fig_sk.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="white",
                font=dict(size=12),
            )
            _pc(fig_sk)

            # Highlight key conversion: Cash → Insurance
            cash_upgrades = df_sk[
                (df_sk["source_payer"] == "Cash") &
                (df_sk["target_payer"].isin(["Insurance", "NHIF / SHA"]))
            ]["patient_count"].sum()
            cash_stays = df_sk[
                (df_sk["source_payer"] == "Cash") &
                (df_sk["target_payer"] == "Cash")
            ]["patient_count"].sum()
            if cash_upgrades > 0 or cash_stays > 0:
                total_cash_first = float(cash_upgrades + cash_stays or 1)
                upgrade_pct = cash_upgrades / total_cash_first * 100
                _upg_ins = (
                    f"<strong>{int(cash_upgrades):,} patients</strong> who first visited as cash payers "
                    f"returned on insurance or NHIF/SHA — {upgrade_pct:.0f}% of all cash-first returning patients. "
                    + ("Strong loyalty conversion from emergency to enrolled."
                       if upgrade_pct > 20 else
                       "Most cash-first patients continue paying out-of-pocket on return visits.")
                )
                st.markdown(
                    f'<div style="background:#f5f5f3;border-radius:8px;padding:6px 9px;'
                    f'font-size:10px;color:#5f5e5a;margin-top:8px;line-height:1.5;">{_upg_ins}</div>',
                    unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="background:#fff8ed;border-left:3px solid #EF9F27;border-radius:0 8px 8px 0;'
                'padding:6px 10px;font-size:10px;color:#633806;margin-top:8px;">'
                'No payer switch data available — patients may not have enough return visits '
                'in the selected period, or payer information is incomplete.</div>',
                unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Payer switch: {e}")

    _gap(12)

    # ── E: PAYER MIX BY AGE GROUP ─────────────────────────────────────────
    _sh("E — Payer mix by age group", mt=8)
    try:
        df_pm = Q.load_payer_mix(filters, run_query)
        if not df_pm.empty:
            df_pm["payer_bucket"] = df_pm["payer_type"].map({
                "Cash":       "Cash",
                "NHIF / SHA": "Insurance / NHIF",
                "Insurance":  "Insurance / NHIF",
            }).fillna("Insurance / NHIF")
            _pm_agg = (df_pm.groupby(["age_group", "payer_bucket"])["total_visits"]
                       .sum().reset_index())
            _pm_pivot = _pm_agg.pivot_table(
                index="age_group", columns="payer_bucket",
                values="total_visits", fill_value=0)
            _age_alpha = sorted(_pm_pivot.index.tolist())
            _pm_pivot = _pm_pivot.reindex(_age_alpha)

            st.markdown(
                '<div style="display:flex;gap:10px;margin-bottom:6px;">'
                '<span style="display:flex;align-items:center;gap:4px;font-size:10px;color:#5f5e5a;">'
                '<span style="width:10px;height:10px;border-radius:2px;background:#378ADD;display:inline-block;"></span>Insurance / NHIF</span>'
                '<span style="display:flex;align-items:center;gap:4px;font-size:10px;color:#5f5e5a;">'
                '<span style="width:10px;height:10px;border-radius:2px;background:rgba(239,159,39,0.8);display:inline-block;"></span>Cash</span>'
                '</div>',
                unsafe_allow_html=True)
            fig_pm = go.Figure()
            for _seg, _clr in [("Insurance / NHIF", "#378ADD"),
                                ("Cash", "rgba(239,159,39,0.8)")]:
                if _seg in _pm_pivot.columns:
                    fig_pm.add_trace(go.Bar(
                        name=_seg, x=_age_alpha,
                        y=_pm_pivot[_seg].tolist(),
                        marker_color=_clr, marker_line_width=0,
                        hovertemplate=f"<b>%{{x}}</b><br>{_seg}: %{{y:,}}<extra></extra>",
                    ))
            fig_pm.update_layout(
                barmode="stack", height=220, margin=dict(l=0, r=10, t=6, b=30),
                plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                xaxis=dict(showgrid=False, tickangle=-30,
                           tickfont=dict(size=9, color="#888780")),
                yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)",
                           tickfont=dict(size=9, color="#888780")),
            )
            _pc(fig_pm)
            if "Cash" in _pm_pivot.columns and "Insurance / NHIF" in _pm_pivot.columns:
                _tot = _pm_pivot["Cash"] + _pm_pivot["Insurance / NHIF"]
                _cash_r = (_pm_pivot["Cash"] / _tot.replace(0, float("nan")) * 100).dropna()
                _ins_r  = (_pm_pivot["Insurance / NHIF"] / _tot.replace(0, float("nan")) * 100).dropna()
                _mc = _cash_r.idxmax() if len(_cash_r) > 0 else None
                _mi = _ins_r.idxmax()  if len(_ins_r) > 0 else None
                if _mc and _mi:
                    st.markdown(
                        f'<div style="background:#f5f5f3;border-radius:8px;padding:6px 9px;'
                        f'font-size:10px;color:#5f5e5a;margin-top:6px;line-height:1.5;">'
                        f'<strong style="color:#1a1a18;">{_mc}</strong> cohort is most cash-heavy — '
                        f'likely parents paying out of pocket. '
                        f'<strong style="color:#1a1a18;">{_mi}</strong> cohort has the highest insurance uptake.</div>',
                        unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Payer mix: {e}")

    _gap(12)

    # ── F: PATIENT REVENUE INTELLIGENCE ─────────────────────────────────
    _sh("F — Patient revenue intelligence", mt=8)
    try:
        df_mx = Q.load_revenue_profile_matrix(filters, run_query)
        if not df_mx.empty:
            for _c in ("total_revenue", "revenue_share_pct", "ip_pct",
                       "cash_pct", "nhif_pct", "corp_pct", "visit_count"):
                df_mx[_c] = pd.to_numeric(df_mx[_c], errors="coerce")

            _max_rev = float(df_mx["total_revenue"].max() or 1)

            # Risk classification per mockup spec
            def _risk_type(row):
                ip   = float(row.get("ip_pct")   or 0)
                cash = float(row.get("cash_pct") or 0)
                nhif = float(row.get("nhif_pct") or 0)
                corp = float(row.get("corp_pct") or 0)
                ins  = nhif + corp
                tier = str(row.get("pareto_tier", ""))
                if cash > 60 and ip < 5:
                    return "liquidity"
                if ip > 0 and cash > 25 and tier in ("Top 10%", "Top 11–20%"):
                    return "leak"
                if ins > 60:
                    return "insurance"
                if 40 <= ins <= 70 and 10 <= ip <= 40:
                    return "mixed"
                return "insurance"

            _RC = {
                "mixed":     {"border": "#1D9E75", "bdg_bg": "#E1F5EE", "bdg_txt": "#0F6E56",
                               "label": "Mixed profile",        "desc": "Balanced payer mix. Protect volume."},
                "leak":      {"border": "#E24B4A", "bdg_bg": "#FCEBEB", "bdg_txt": "#A32D2D",
                               "label": "Data leak",            "desc": "High-value claims going to cash. Audit billing."},
                "liquidity": {"border": "#EF9F27", "bdg_bg": "#FAEEDA", "bdg_txt": "#854F0B",
                               "label": "Liquidity engine",     "desc": "Rapid cash flow. Protect throughput."},
                "insurance": {"border": "#7F77DD", "bdg_bg": "#EEEDFE", "bdg_txt": "#534AB7",
                               "label": "Insurance cycle risk", "desc": "Revenue delayed by claims. Monitor receivables."},
            }
            _TIER_STYLE = {
                "Top 10%":        {"bg": "#FCEBEB", "txt": "#A32D2D"},
                "Top 11–20%":     {"bg": "#FAEEDA", "txt": "#854F0B"},
                "Middle 21–50%":  {"bg": "#f1efe8", "txt": "#5f5e5a"},
                "Mid 21–50%":     {"bg": "#f1efe8", "txt": "#5f5e5a"},
                "Bottom 50%":     {"bg": "#f5f5f3", "txt": "#888780"},
            }

            # ── Legend ──────────────────────────────────────────────────
            _leg_html = '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px;">'
            for _rk, _rv in _RC.items():
                _leg_html += (
                    f'<span style="display:flex;align-items:center;gap:5px;font-size:10px;color:#5f5e5a;">'
                    f'<span style="width:3px;height:16px;border-radius:1px;background:{_rv["border"]};flex-shrink:0;display:inline-block;"></span>'
                    f'<strong style="color:{_rv["bdg_txt"]};">{_rv["label"]}</strong>'
                    f' — {_rv["desc"]}'
                    f'</span>'
                )
            _leg_html += '</div>'

            # ── Column headers ───────────────────────────────────────────
            _hdr_html = (
                '<div style="display:grid;grid-template-columns:140px 1fr 80px 48px 100px 120px;'
                'gap:6px;padding:0 0 6px;border-bottom:0.5px solid rgba(0,0,0,0.10);margin-bottom:4px;">'
                + ''.join([
                    f'<div style="font-size:9px;font-weight:500;color:#888780;'
                    f'text-transform:uppercase;letter-spacing:0.04em;{"text-align:center;" if i > 1 else ""}">{h}</div>'
                    for i, h in enumerate(
                        ["Condition", "Revenue (KES)", "Pareto tier", "IP %", "Payer mix", "Commercial risk"]
                    )
                ])
                + '</div>'
            )

            # ── Rows ─────────────────────────────────────────────────────
            _rows_html = ""
            for _, row in df_mx.iterrows():
                _rk  = _risk_type(row)
                _rv  = _RC[_rk]
                _rev = float(row.get("total_revenue") or 0)
                _shr = float(row.get("revenue_share_pct") or 0)
                _bw  = round(_rev / _max_rev * 100)
                _ip  = int(float(row.get("ip_pct") or 0))
                _cas = int(float(row.get("cash_pct") or 0))
                _ins = min(100, int(float(row.get("nhif_pct") or 0) + float(row.get("corp_pct") or 0)))
                _tier = str(row.get("pareto_tier", ""))
                _ts  = _TIER_STYLE.get(_tier, {"bg": "#f5f5f3", "txt": "#888780"})
                _rev_lbl = (f"KES {_rev/1e6:.1f}M · {_shr:.1f}%"
                            if _rev >= 1e6 else f"KES {_rev:,.0f} · {_shr:.1f}%")
                _ip_clr = "#A32D2D" if _ip > 20 else ("#854F0B" if _ip > 10 else "#1a1a18")
                _ip_arr = " ▲" if _ip > 20 else ""
                _rev_bar = (
                    f'<div style="flex:1;position:relative;height:14px;background:#f5f5f3;border-radius:3px;overflow:hidden;">'
                    f'<div style="position:absolute;left:0;top:0;bottom:0;width:{_bw}%;background:{_rv["border"]};opacity:0.75;border-radius:3px;"></div>'
                    + (f'<span style="position:absolute;right:4px;top:50%;transform:translateY(-50%);font-size:9px;font-weight:500;color:#fff;white-space:nowrap;">{_rev_lbl}</span>'
                       if _bw > 32 else '')
                    + '</div>'
                    + (f'<span style="font-size:9px;font-weight:500;color:{_rv["border"]};white-space:nowrap;margin-left:4px;">{_rev_lbl}</span>'
                       if _bw <= 32 else '')
                )
                _rows_html += (
                    f'<div style="display:grid;grid-template-columns:140px 1fr 80px 48px 100px 120px;'
                    f'gap:6px;align-items:center;padding:5px 0;'
                    f'border-bottom:0.5px solid rgba(0,0,0,0.06);">'
                    f'<div style="font-size:10px;color:#1a1a18;line-height:1.3;'
                    f'border-left:3px solid {_rv["border"]};padding-left:6px;">'
                    f'{row.get("condition","")}</div>'
                    f'<div style="display:flex;align-items:center;">{_rev_bar}</div>'
                    f'<div style="text-align:center;">'
                    f'<span style="font-size:9px;font-weight:500;padding:2px 5px;border-radius:20px;'
                    f'background:{_ts["bg"]};color:{_ts["txt"]};white-space:nowrap;">{_tier}</span></div>'
                    f'<div style="font-size:10px;font-weight:500;text-align:center;color:{_ip_clr};">{_ip}%{_ip_arr}</div>'
                    f'<div style="display:flex;flex-direction:column;gap:2px;">'
                    f'<div style="height:6px;background:#f5f5f3;border-radius:3px;position:relative;overflow:hidden;">'
                    f'<div style="position:absolute;left:0;top:0;bottom:0;width:{_ins}%;background:#378ADD;border-radius:3px 0 0 3px;"></div>'
                    f'<div style="position:absolute;top:0;bottom:0;left:{_ins}%;width:{_cas}%;background:rgba(239,159,39,0.75);"></div>'
                    f'</div>'
                    f'<div style="display:flex;justify-content:space-between;margin-top:1px;">'
                    f'<span style="font-size:8px;color:#378ADD;">Ins {_ins}%</span>'
                    f'<span style="font-size:8px;color:#854F0B;">Cash {_cas}%</span>'
                    f'</div></div>'
                    f'<div style="text-align:center;">'
                    f'<span style="font-size:9px;font-weight:500;padding:3px 6px;border-radius:4px;'
                    f'background:{_rv["bdg_bg"]};color:{_rv["bdg_txt"]};line-height:1.3;display:inline-block;">'
                    f'{_rv["label"]}</span></div>'
                    f'</div>'
                )

            _html_f = (
                f'<div style="font-family:system-ui,sans-serif;padding:0;">'
                f'{_leg_html}{_hdr_html}{_rows_html}'
                f'</div>'
            )
            _tbl_h = len(df_mx) * 42 + 120
            _stcomp.html(_html_f, height=_tbl_h, scrolling=False)
        else:
            st.markdown(
                '<div style="background:#fff8ed;border-left:3px solid #EF9F27;border-radius:0 8px 8px 0;'
                'padding:6px 10px;font-size:10px;color:#633806;">No revenue data available for the selected period.</div>',
                unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Revenue intelligence: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PATIENT FLOW & RETENTION
# ══════════════════════════════════════════════════════════════════════════════

def _build_tier_table(df: pd.DataFrame) -> str:
    """HTML table for visit-tier LTFU revenue risk (Priority 2)."""
    rows = ""
    for _, r in df.iterrows():
        rar = float(r.get("revenue_at_risk") or 0)
        arv = float(r.get("avg_rev_per_visit") or 0)
        shr = float(r.get("risk_share_pct") or 0)
        tier = str(r.get("engagement_tier", ""))
        clr = "#E24B4A" if tier.startswith("1") else "#1D9E75"
        rows += (
            f'<tr>'
            f'<td style="padding:8px 12px;font-size:11px;font-weight:600;color:{clr};">{tier}</td>'
            f'<td style="padding:8px 12px;font-size:11px;text-align:right;">'
            f'{int(r.get("ltfu_patients") or 0):,}</td>'
            f'<td style="padding:8px 12px;font-size:11px;text-align:right;">'
            f'KES {arv:,.0f}</td>'
            f'<td style="padding:8px 12px;font-size:11px;text-align:right;">'
            f'KES {rar/1e6:.1f}M</td>'
            f'<td style="padding:8px 12px;font-size:11px;text-align:right;font-weight:600;">'
            f'{shr:.0f}%</td>'
            f'</tr>'
        )
    return (
        '<style>*{box-sizing:border-box;margin:0;padding:0;font-family:Montserrat,-apple-system,sans-serif;}'
        'table{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;overflow:hidden;}'
        'thead{background:#f8fafc;}th{padding:8px 12px;font-size:9px;font-weight:700;'
        'letter-spacing:0.08em;text-transform:uppercase;color:#888780;text-align:left;}'
        'tr:not(:last-child){border-bottom:1px solid #f3f4f6;}'
        'tr:hover{background:#f0f7ff;}</style>'
        '<table><thead><tr>'
        '<th>Visit Tier</th><th style="text-align:right">LTFU Patients</th>'
        '<th style="text-align:right">Avg Rev/Visit</th>'
        '<th style="text-align:right">Revenue at Risk</th>'
        '<th style="text-align:right">% of Total</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )


def _build_ddrr_table(df: pd.DataFrame) -> str:
    """HTML table for demographic-diagnosis revenue risk (Priority 3)."""
    _payer_badge = {
        "Cash":        ("rgba(239,159,39,0.15)", "#854F0B"),
        "NHIF / SHA":  ("rgba(55,138,221,0.15)", "#1E5FA5"),
        "Insurance":   ("rgba(55,138,221,0.15)", "#1E5FA5"),
    }
    rows = ""
    for i, r in df.iterrows():
        rar = float(r.get("revenue_at_risk") or 0)
        arv = float(r.get("avg_rev_per_visit") or 0)
        ltfu = int(r.get("ltfu_patients") or 0)
        payer = str(r.get("payer", ""))
        pbg, ptxt = _payer_badge.get(payer, ("rgba(100,100,100,0.1)", "#374151"))
        rar_str = f"KES {rar/1e6:.1f}M" if rar >= 1e6 else f"KES {rar:,.0f}"
        tier_col = "#E24B4A" if rar >= 1e6 else ("#EF9F27" if rar >= 5e5 else "#1D9E75")
        rows += (
            f'<tr>'
            f'<td style="padding:8px 12px;font-size:11px;">{r.get("age_group","")}</td>'
            f'<td style="padding:8px 12px;font-size:11px;">{r.get("gender","")}</td>'
            f'<td style="padding:8px 12px;font-size:11px;">{r.get("condition","")}</td>'
            f'<td style="padding:8px 12px;">'
            f'<span style="background:{pbg};color:{ptxt};font-size:9px;font-weight:700;'
            f'padding:2px 7px;border-radius:12px;">{payer}</span></td>'
            f'<td style="padding:8px 12px;font-size:11px;text-align:right;">{ltfu:,}</td>'
            f'<td style="padding:8px 12px;font-size:11px;text-align:right;'
            f'font-weight:700;color:{tier_col};">{rar_str}</td>'
            f'</tr>'
        )
    return (
        '<style>*{box-sizing:border-box;margin:0;padding:0;font-family:Montserrat,-apple-system,sans-serif;}'
        'table{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;overflow:hidden;}'
        'thead{background:#f8fafc;}th{padding:8px 12px;font-size:9px;font-weight:700;'
        'letter-spacing:0.08em;text-transform:uppercase;color:#888780;text-align:left;}'
        'tr:not(:last-child){border-bottom:1px solid #f3f4f6;}'
        'tr:hover{background:#f0f7ff;}</style>'
        '<table><thead><tr>'
        '<th>Age Group</th><th>Gender</th><th>Condition</th><th>Payer</th>'
        '<th style="text-align:right">LTFU Patients</th>'
        '<th style="text-align:right">Revenue at Risk</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )


def render_tab3_retention(filters: dict, run_query):

    # ── lifecycle display helpers ──────────────────────────────────────────────
    _LC_STRIP = {"1. Active (≤90 days)":    "Active",
                 "2. Lapsing (91–180 days)": "Lapsing",
                 "3. LTFU (>180 days)":      "LTFU >180d"}
    _LC_COL   = {"Active":    "#1D9E75",
                 "Lapsing":   "#EF9F27",
                 "LTFU >180d": "#E24B4A",
                 "Active (≤90d)":   "#1D9E75",
                 "Lapsing (91-180d)":"#EF9F27",
                 "LTFU (>180d)":    "#E24B4A"}

    # ── load lifecycle once — used by both KPI strip and Section A ────────────
    _df_lc_raw = pd.DataFrame()
    try:
        _df_lc_raw = Q.load_lifecycle(filters, run_query)
        _df_lc_raw["_label"] = (_df_lc_raw["lifecycle_status"]
                                .map(_LC_STRIP)
                                .fillna(_df_lc_raw["lifecycle_status"]))
        _df_lc_raw["patient_count"] = pd.to_numeric(_df_lc_raw["patient_count"], errors="coerce")
    except Exception:
        pass

    def _lc_count(label: str) -> int:
        if _df_lc_raw.empty:
            return 0
        rows = _df_lc_raw[_df_lc_raw["_label"] == label]
        return int(rows["patient_count"].sum()) if not rows.empty else 0

    # ── KPI STRIP ─────────────────────────────────────────────────────────────
    try:
        df_k  = Q.load_retention_kpis(filters, run_query)
        df_rr = Q.load_revenue_at_risk(filters, run_query)
        if not df_k.empty:
            row = df_k.iloc[0]
            rr  = float(row.get("retention_rate_pct") or 0)
            rr_row = df_rr.iloc[0] if not df_rr.empty else {}
            # Derive correct counts from lifecycle (based on last visit gap from today)
            _ltfu_count    = _lc_count("LTFU >180d")
            _chronic_total = int(_df_lc_raw["patient_count"].sum()) if not _df_lc_raw.empty else int(row.get("chronic_patients") or 0)
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                _kpi("Chronic Patients", _n(_chronic_total),
                     "under active management", AFYA_BLUE)
            with c2:
                _kpi("Retained (90d)", _n(row.get("retained_patients")),
                     str(_p(rr)) + " retention rate",
                     TEAL if rr >= 60 else CORAL)
            with c3:
                _kpi("LTFU >180d", _n(_ltfu_count),
                     "no visit in 180+ days", CORAL)
            with c4:
                _kpi("Recoverable Revenue",
                     _k(rr_row.get("lapsing_revenue_recoverable")),
                     "lapsing patients (31–90d)", ORANGE)
            with c5:
                _pc(bullet(rr, 60, "Retention vs 60%", format="pct", height=100))
    except Exception as e:
        st.warning(f"Retention KPIs: {e}")

    _gap(16)

    # ── SECTION A — LIFECYCLE OVERVIEW ────────────────────────────────────────
    _sh("A — Patient Lifecycle Overview", mt=8)
    try:
        df_lc = _df_lc_raw.copy()
        if not df_lc.empty:
            df_lc = df_lc.sort_values("_label", ascending=True)

            c1, c2 = st.columns(2)
            with c1:
                fig_lca = go.Figure(go.Bar(
                    x=df_lc["_label"],
                    y=df_lc["patient_count"],
                    marker_color=[_LC_COL.get(l, GRAY) for l in df_lc["_label"]],
                    text=[f"{int(v):,}" for v in df_lc["patient_count"]],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>%{y:,} patients<extra></extra>",
                ))
                fig_lca.update_layout(
                    height=280, margin=dict(l=0, r=20, t=20, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(title="Patients", rangemode="tozero"),
                    xaxis=dict(title=""),
                    showlegend=False,
                )
                _pc(fig_lca)
            with c2:
                _pc(donut(
                    labels=df_lc["_label"].tolist(),
                    values=df_lc["patient_count"].tolist(),
                    color_map=_LC_COL,
                    height=280,
                ))

            # Signal: LTFU share
            total_lc = float(df_lc["patient_count"].sum() or 1)
            ltfu_row = df_lc[df_lc["_label"] == "LTFU >180d"]
            if not ltfu_row.empty:
                ltfu_pct = float(ltfu_row["patient_count"].iloc[0]) / total_lc * 100
                lapsing_row = df_lc[df_lc["_label"] == "Lapsing"]
                lapsing_pct = (float(lapsing_row["patient_count"].iloc[0]) / total_lc * 100
                               if not lapsing_row.empty else 0)
                _note(
                    f"{ltfu_pct:.0f}% of chronic patients are LTFU (>180 days since last visit); "
                    f"a further {lapsing_pct:.0f}% are lapsing (91–180 days). "
                    + ("Combined dropout exceeds 50% — retention intervention is critical."
                       if ltfu_pct + lapsing_pct > 50 else
                       "Combined dropout below 50% — monitor lapsing cohort to prevent further loss."),
                    w=ltfu_pct + lapsing_pct > 50,
                )
    except Exception as e:
        st.warning(f"Lifecycle: {e}")

    _gap(16)

    # ── SECTION B — WHO DROPS OUT? ────────────────────────────────────────────
    _sh("B — Who Drops Out? LTFU Rate by Age, Payer & Sex", mt=8)
    try:
        df_cor = Q.load_ltfu_correlation(filters, run_query)
        if not df_cor.empty:
            for _c in ("ltfu_rate_pct", "total", "ltfu", "retained"):
                df_cor[_c] = pd.to_numeric(df_cor[_c], errors="coerce")

            _factor_order = ["Age Group", "Payer", "Sex"]
            factors = [f for f in _factor_order if f in df_cor["factor"].values]
            if not factors:
                factors = df_cor["factor"].unique().tolist()

            cols_b = st.columns(len(factors))
            for col, factor in zip(cols_b, factors):
                sub = df_cor[df_cor["factor"] == factor].sort_values("ltfu_rate_pct", ascending=True)
                with col:
                    st.markdown(
                        f'<p style="font-size:9px;font-weight:700;letter-spacing:0.08em;'
                        f'color:#888780;text-transform:uppercase;margin-bottom:6px;">'
                        f'{factor}</p>',
                        unsafe_allow_html=True,
                    )
                    fig_b = go.Figure(go.Bar(
                        x=sub["ltfu_rate_pct"],
                        y=sub["dimension"],
                        orientation="h",
                        marker_color=[
                            "#E24B4A" if v >= 60 else "#EF9F27" if v >= 40 else "#1D9E75"
                            for v in sub["ltfu_rate_pct"]
                        ],
                        text=[f"{v:.0f}%" for v in sub["ltfu_rate_pct"]],
                        textposition="outside",
                        hovertemplate="<b>%{y}</b><br>LTFU rate: %{x:.1f}%<br>"
                                      "<extra></extra>",
                    ))
                    fig_b.add_vline(x=60, line=dict(color="#E24B4A", width=1, dash="dot"))
                    fig_b.update_layout(
                        height=max(180, len(sub) * 38 + 60),
                        margin=dict(l=0, r=60, t=10, b=20),
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(title="LTFU %", range=[0, 110], ticksuffix="%"),
                        yaxis=dict(title=""),
                        showlegend=False,
                    )
                    _pc(fig_b)

            worst = df_cor.loc[df_cor["ltfu_rate_pct"].idxmax()]
            _note(
                f"Highest dropout: {worst['dimension']} ({worst['factor']}) — "
                f"{worst['ltfu_rate_pct']:.0f}% LTFU rate across {int(worst['total'] or 0):,} patients.",
                w=float(worst["ltfu_rate_pct"]) >= 60,
            )
    except Exception as e:
        st.warning(f"LTFU correlation: {e}")

    _gap(16)

    # ── SECTION C — REVENUE AT RISK ───────────────────────────────────────────
    _sh("C — Revenue at Risk from Dropout", mt=8)
    try:
        df_rar2 = Q.load_revenue_at_risk(filters, run_query)
        if not df_rar2.empty:
            row_r = df_rar2.iloc[0]
            c1, c2, c3 = st.columns(3)
            with c1:
                _kpi("Chronic LTFU Patients",
                     _n(row_r.get("chronic_ltfu")),
                     "permanently lost (>180d)", CORAL)
            with c2:
                _kpi("Annual Revenue at Risk",
                     _k(row_r.get("chronic_ltfu_revenue_at_risk")),
                     "from LTFU chronic patients", CORAL)
            with c3:
                _kpi("Recoverable Revenue",
                     _k(row_r.get("lapsing_revenue_recoverable")),
                     "lapsing cohort (31–90d)", ORANGE)
    except Exception as e:
        st.warning(f"Revenue at risk: {e}")

    _gap(16)

    # ── PRIORITY 1 — LAPSING PATIENTS ─────────────────────────────────────────
    _sh("Priority 1 — Lapsing Patients (31–90 Days)", mt=8)
    try:
        df_cc = Q.load_cost_dropout_correlation(filters, run_query)
        if not df_cc.empty:
            for _c in ("patient_count", "avg_invoice_size", "avg_inv_wait_hrs", "avg_rx_cost"):
                df_cc[_c] = pd.to_numeric(df_cc[_c], errors="coerce")

            _lapsing_masks = df_cc["lifecycle"].str.contains("Lapsing", case=False, na=False)
            lapsing_df = df_cc[_lapsing_masks]
            total_lapsing = int(lapsing_df["patient_count"].sum() or 0)

            c1, c2 = st.columns([1, 2])
            with c1:
                _kpi("Lapsing Patients", _n(total_lapsing),
                     "91–180 days without return", ORANGE)
                _gap(8)
                df_rar3 = Q.load_revenue_at_risk(filters, run_query)
                if not df_rar3.empty:
                    _kpi("Recoverable Revenue",
                         _k(df_rar3.iloc[0].get("lapsing_revenue_recoverable")),
                         "if re-engaged this month", ORANGE)

            with c2:
                if not lapsing_df.empty:
                    lapsing_by_payer = (lapsing_df.groupby("payer")["patient_count"]
                                        .sum().reset_index()
                                        .sort_values("patient_count", ascending=True))
                    _payer_col = {
                        "Cash":               "rgba(239,159,39,0.85)",
                        "Insurance / Corporate": "#378ADD",
                        "NHIF / SHA":         "#378ADD",
                    }
                    fig_p1 = go.Figure(go.Bar(
                        x=lapsing_by_payer["patient_count"],
                        y=lapsing_by_payer["payer"],
                        orientation="h",
                        marker_color=[_payer_col.get(p, GRAY) for p in lapsing_by_payer["payer"]],
                        text=[f"{int(v):,}" for v in lapsing_by_payer["patient_count"]],
                        textposition="outside",
                        hovertemplate="<b>%{y}</b><br>%{x:,} lapsing patients<extra></extra>",
                    ))
                    fig_p1.update_layout(
                        height=200, margin=dict(l=0, r=60, t=20, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(title="Patients", rangemode="tozero"),
                        yaxis=dict(title=""),
                        showlegend=False,
                    )
                    _pc(fig_p1)

            # Signal strip: cash vs insured lapsing split
            if not lapsing_df.empty:
                cash_lap = int(lapsing_df[lapsing_df["payer"] == "Cash"]["patient_count"].sum() or 0)
                if total_lapsing > 0:
                    cash_pct = cash_lap / total_lapsing * 100
                    _note(
                        f"{cash_pct:.0f}% of lapsing patients are cash-paying. "
                        + ("Cash patients face no appointment reminders via insurer — "
                           "proactive outreach (SMS/call) is the primary re-engagement lever."
                           if cash_pct > 50 else
                           "Insured lapsing patients may be reachable via their insurer's care coordination channel."),
                        w=cash_pct > 60,
                    )
    except Exception as e:
        st.warning(f"Priority 1 lapsing: {e}")

    _gap(16)

    # ── PRIORITY 2 — LOW-ENGAGEMENT LTFU ─────────────────────────────────────
    _sh("Priority 2 — 1–2 Visit LTFU Patients", mt=8)
    try:
        df_le = Q.load_low_engagement_revenue_risk(filters, run_query)
        if not df_le.empty:
            for _c in ("ltfu_patients", "avg_rev_per_visit", "revenue_at_risk"):
                df_le[_c] = pd.to_numeric(df_le[_c], errors="coerce")
            total_risk_le = float(df_le["revenue_at_risk"].sum() or 1)
            df_le["risk_share_pct"] = (df_le["revenue_at_risk"] / total_risk_le * 100).round(1)

            low_eng = df_le[df_le["engagement_tier"] == "1–2 Visits"]
            c1, c2 = st.columns([1, 2])
            with c1:
                if not low_eng.empty:
                    _kpi("1–2 Visit LTFU",
                         _n(low_eng["ltfu_patients"].iloc[0]),
                         f"{low_eng['risk_share_pct'].iloc[0]:.0f}% of total revenue at risk",
                         CORAL)
                    _gap(8)
                    _kpi("Revenue at Risk",
                         _k(low_eng["revenue_at_risk"].iloc[0]),
                         "low-engagement chronic LTFU", CORAL)
            with c2:
                _stcomp.html(
                    _build_tier_table(df_le),
                    height=len(df_le) * 34 + 52,
                    scrolling=False,
                )

            if not low_eng.empty:
                le_share = float(low_eng["risk_share_pct"].iloc[0] or 0)
                _note(
                    f"{le_share:.0f}% of 180-day LTFU revenue risk comes from patients with only 1–2 visits. "
                    + ("A significant share — investigate whether intake quality, cost transparency, "
                       "or wait times at first visits deterred return."
                       if le_share > 30 else
                       "Most revenue risk is from established patients who lapsed, not first-time visitors."),
                    w=le_share > 30,
                )
    except Exception as e:
        st.warning(f"Priority 2 low-engagement: {e}")

    _gap(16)

    # ── PRIORITY 3 — DEMOGRAPHIC-DIAGNOSIS REVENUE RISK TABLE ────────────────
    _sh("Priority 3 — Demographic-Diagnosis Revenue at Risk", mt=8)
    try:
        df_ddrr = Q.load_demographic_diagnosis_revenue_risk(filters, run_query)
        if not df_ddrr.empty:
            for _c in ("ltfu_patients", "avg_rev_per_visit", "revenue_at_risk"):
                df_ddrr[_c] = pd.to_numeric(df_ddrr[_c], errors="coerce")

            top = df_ddrr.iloc[0]
            _note(
                f"Highest concentration: {top.get('gender','')} aged {top.get('age_group','')} "
                f"with {top.get('condition','')} ({top.get('payer','')}) — "
                f"{int(top['ltfu_patients']):,} LTFU patients, "
                f"est. KES {float(top['revenue_at_risk'])/1e6:.1f}M at risk annually.",
                w=True,
            )
            _stcomp.html(
                _build_ddrr_table(df_ddrr.head(20)),
                height=len(df_ddrr.head(20)) * 34 + 52,
                scrolling=False,
            )
    except Exception as e:
        st.warning(f"Priority 3 demographic-diagnosis: {e}")

    _gap(16)

    # ── SECTION E — RETAINED PATIENT SERVICE USAGE ───────────────────────────
    _sh("E — What Are Retained Patients Coming Back For?", mt=8)
    try:
        df_fp = Q.load_retained_patient_footprint(filters, run_query)
        if not df_fp.empty:
            _svc_cols   = ["consult_rate_pct", "investigation_rate_pct", "rx_rate_pct"]
            _svc_labels = ["Consultations", "Lab / Investigations", "Prescriptions"]
            _svc_colors = [AFYA_BLUE, "#7F77DD", TEAL]
            for _c in ["retained_patients"] + _svc_cols:
                df_fp[_c] = pd.to_numeric(df_fp[_c], errors="coerce")

            fig_e = go.Figure()
            for svc, label, color in zip(_svc_cols, _svc_labels, _svc_colors):
                fig_e.add_trace(go.Bar(
                    name=label,
                    x=df_fp["payer"],
                    y=df_fp[svc],
                    marker_color=color,
                    text=[f"{v:.0f}%" for v in df_fp[svc]],
                    textposition="outside",
                    hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.1f}}%<extra></extra>",
                ))
            fig_e.update_layout(
                barmode="group", height=300,
                margin=dict(l=0, r=20, t=20, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(title="% of Retained Visits", rangemode="tozero", ticksuffix="%"),
                xaxis=dict(title=""),
                legend=dict(orientation="h", y=1.12, xanchor="right", x=1),
            )
            _pc(fig_e)

            for _, row in df_fp.iterrows():
                pharm_pct = float(row.get("pharmacy_only_pct") or 0)
                if pharm_pct > 30:
                    _note(
                        f"{row['payer']}: {pharm_pct:.0f}% of retained visits are pharmacy-only — "
                        "patients collecting medication without clinical review.",
                        w=pharm_pct > 40,
                    )
    except Exception as e:
        st.warning(f"Retained patient footprint: {e}")

    _gap(16)

    # ── SECTION F — COST & WAIT SIGNALS ──────────────────────────────────────
    _sh("F — Do Costs & Wait Times Drive Dropout?", mt=8)
    try:
        df_cc2 = Q.load_cost_dropout_correlation(filters, run_query)
        if not df_cc2.empty:
            for _c in ("patient_count", "avg_invoice_size", "avg_inv_wait_hrs", "avg_rx_cost"):
                df_cc2[_c] = pd.to_numeric(df_cc2[_c], errors="coerce")
            # Negative wait times are data artefacts (result recorded before order) — treat as missing
            df_cc2.loc[df_cc2["avg_inv_wait_hrs"] < 0, "avg_inv_wait_hrs"] = float("nan")

            _lc_order  = ["Active (≤90d)", "Lapsing (91-180d)", "LTFU (>180d)"]
            _lc_colors_f = {"Active (≤90d)": "#1D9E75", "Lapsing (91-180d)": "#EF9F27",
                            "LTFU (>180d)": "#E24B4A"}

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    '<p style="font-size:9px;font-weight:700;letter-spacing:0.08em;'
                    'color:#888780;text-transform:uppercase;margin-bottom:6px;">'
                    'Avg invoice size by payer &amp; lifecycle</p>',
                    unsafe_allow_html=True,
                )
                fig_f1 = go.Figure()
                for lc in _lc_order:
                    sub = df_cc2[df_cc2["lifecycle"] == lc]
                    if not sub.empty:
                        fig_f1.add_trace(go.Bar(
                            name=lc, x=sub["payer"], y=sub["avg_invoice_size"],
                            marker_color=_lc_colors_f.get(lc, GRAY),
                            text=[f"KES {v:,.0f}" if pd.notna(v) else "" for v in sub["avg_invoice_size"]],
                            textposition="outside",
                            hovertemplate=f"<b>%{{x}}</b><br>{lc}: KES %{{y:,.0f}}<extra></extra>",
                        ))
                fig_f1.update_layout(
                    barmode="group", height=280,
                    margin=dict(l=0, r=20, t=10, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(title="Avg Invoice (KES)", rangemode="tozero"),
                    legend=dict(orientation="h", y=1.12, xanchor="right", x=1),
                )
                _pc(fig_f1)

            with c2:
                st.markdown(
                    '<p style="font-size:9px;font-weight:700;letter-spacing:0.08em;'
                    'color:#888780;text-transform:uppercase;margin-bottom:6px;">'
                    'Investigation wait time (hrs) by lifecycle</p>',
                    unsafe_allow_html=True,
                )
                insured_df2 = df_cc2[df_cc2["payer"].str.contains("Insurance|Corporate|NHIF",
                                                                   case=False, na=False)]
                if not insured_df2.empty and insured_df2["avg_inv_wait_hrs"].notna().any():
                    fig_f2 = go.Figure()
                    for lc in _lc_order:
                        sub = insured_df2[insured_df2["lifecycle"] == lc]
                        if not sub.empty:
                            fig_f2.add_trace(go.Bar(
                                name=lc, x=[lc], y=sub["avg_inv_wait_hrs"].values,
                                marker_color=_lc_colors_f.get(lc, GRAY),
                                text=[f"{v:.1f}h" if pd.notna(v) else "" for v in sub["avg_inv_wait_hrs"].values],
                                textposition="outside",
                            ))
                    fig_f2.update_layout(
                        barmode="group", height=280,
                        margin=dict(l=0, r=20, t=10, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        yaxis=dict(title="Avg Wait (Hours)", rangemode="tozero"),
                        showlegend=False,
                    )
                    _pc(fig_f2)
                else:
                    st.info("Investigation wait time data not available for insured patients.")

            # Cost correlation signal
            _cash = df_cc2[df_cc2["payer"] == "Cash"].set_index("lifecycle")
            if "LTFU (>180d)" in _cash.index and "Active (≤90d)" in _cash.index:
                _li = float(_cash.loc["LTFU (>180d)", "avg_invoice_size"] or 0)
                _ai = float(_cash.loc["Active (≤90d)", "avg_invoice_size"] or 0)
                if _li > 0 and _ai > 0:
                    _dp = (_li - _ai) / _ai * 100
                    _note(
                        f"Cash LTFU patients had avg invoices of KES {_li:,.0f} vs KES {_ai:,.0f} for "
                        f"active cash patients ({'higher' if _dp > 0 else 'lower'} by {abs(_dp):.0f}%). "
                        + ("Higher bills appear to correlate with permanent dropout among cash-paying chronic patients."
                           if _dp > 20 else
                           "Invoice size alone does not appear to be the primary dropout driver for cash patients."),
                        w=_dp > 20,
                    )
    except Exception as e:
        st.warning(f"Cost / wait signals: {e}")

    _gap(8)

    # Surge follow-up sub-panel
    try:
        df_sf = Q.load_insured_surge_followup(filters, run_query)
        if not df_sf.empty:
            df_sf["avg_days_to_next_visit"] = pd.to_numeric(df_sf["avg_days_to_next_visit"], errors="coerce")
            df_sf["is_surge_month"]         = pd.to_numeric(df_sf["is_surge_month"],         errors="coerce")

            st.markdown(
                '<p style="font-size:9px;font-weight:700;letter-spacing:0.08em;'
                'color:#888780;text-transform:uppercase;margin:12px 0 6px;">'
                'Insured patient return gaps — surge vs normal months</p>',
                unsafe_allow_html=True,
            )
            fig_sf = go.Figure(go.Bar(
                x=df_sf["visit_month"],
                y=df_sf["avg_days_to_next_visit"],
                marker_color=["#E24B4A" if s == 1 else TEAL for s in df_sf["is_surge_month"]],
                hovertemplate="<b>%{x|%b %Y}</b><br>Avg days to next visit: %{y:.0f}<extra></extra>",
            ))
            for _, row_sf in df_sf[df_sf["is_surge_month"] == 1].iterrows():
                fig_sf.add_annotation(
                    x=row_sf["visit_month"],
                    y=float(row_sf["avg_days_to_next_visit"] or 0) + 3,
                    text="SURGE", showarrow=False,
                    font=dict(size=8, color="#E24B4A"),
                )
            fig_sf.update_layout(
                height=240, margin=dict(l=0, r=20, t=10, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(title="Avg Days to Next Visit", rangemode="tozero"),
                xaxis=dict(title="Month"),
                showlegend=False,
            )
            _pc(fig_sf)

            _surge  = df_sf[df_sf["is_surge_month"] == 1]
            _normal = df_sf[df_sf["is_surge_month"] == 0]
            if not _surge.empty and not _normal.empty:
                _sa = float(_surge["avg_days_to_next_visit"].mean() or 0)
                _na = float(_normal["avg_days_to_next_visit"].mean() or 0)
                _note(
                    f"Avg return gap: {_sa:.0f} days during surge months vs {_na:.0f} days normally "
                    f"({'longer' if _sa > _na else 'shorter'} by {abs(_sa - _na):.0f} days). "
                    + ("Surge periods appear to delay insured follow-ups — investigate whether "
                       "clinics are rebooking or patients are walking away."
                       if _sa > _na * 1.15 else
                       "No significant pattern between surge months and insured follow-up frequency."),
                    w=_sa > _na * 1.15,
                )
    except Exception as e:
        st.warning(f"Surge follow-up: {e}")




# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DISEASE BURDEN
# ══════════════════════════════════════════════════════════════════════════════

def _ov_ip_color(v: float) -> str:
    v = float(v or 0)
    return "#E24B4A" if v > 20 else "#EF9F27" if v >= 10 else "#1D9E75"

def _ov_ip_bg(v: float) -> str:
    v = float(v or 0)
    return "rgba(228,75,74,0.08)" if v > 20 else "rgba(239,159,39,0.08)" if v >= 10 else "rgba(29,158,117,0.08)"

def _ra3(arr: list) -> list:
    """3-month rolling average; use partial window for the first two positions."""
    out = []
    for i, v in enumerate(arr):
        if i == 0:
            out.append(arr[0])
        elif i == 1:
            out.append(round((arr[0] + arr[1]) / 2))
        else:
            a, b, c = arr[i-2], arr[i-1], arr[i]
            out.append(round((a + b + c) / 3) if None not in (a, b, c) else v)
    return out

_IP_LEGEND_HTML = (
    '<div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin-bottom:10px;">'
    '<span style="font-size:9px;font-weight:700;text-transform:uppercase;'
    'letter-spacing:.08em;color:#888780;">IP tier:</span>'
    '<span style="display:flex;align-items:center;gap:5px;font-size:10px;">'
    '<span style="display:inline-block;width:3px;height:14px;background:#E24B4A;border-radius:1px;"></span>'
    'High IP% (&gt;20%) — chronic, high retention value</span>'
    '<span style="display:flex;align-items:center;gap:5px;font-size:10px;">'
    '<span style="display:inline-block;width:3px;height:14px;background:#EF9F27;border-radius:1px;"></span>'
    'Mid IP% (10–20%) — mixed</span>'
    '<span style="display:flex;align-items:center;gap:5px;font-size:10px;">'
    '<span style="display:inline-block;width:3px;height:14px;background:#1D9E75;border-radius:1px;"></span>'
    'Low IP% (&lt;10%) — outpatient dominated, lower margin</span>'
    '</div>'
)

_HTML_BASE = (
    '<!DOCTYPE html><html><head><meta charset="utf-8">'
    '<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap" rel="stylesheet">'
    '<style>*{{box-sizing:border-box;margin:0;padding:0;}}'
    'body{{background:#fff;font-family:Montserrat,-apple-system,sans-serif;padding:2px 0;}}'
    '</style></head><body>{}</body></html>'
)


def _sec_c_html(df: pd.DataFrame) -> str:
    if df.empty:
        return _HTML_BASE.format('<p style="padding:12px;font-size:11px;color:#9ca3af;">No data.</p>')
    max_v = float(df["total_visits"].max() or 1)
    rows = ""
    for _, r in df.iterrows():
        ip   = float(r.get("ip_pct") or 0)
        op_v = int(r.get("outpatient_visits") or 0)
        ip_v = int(r.get("inpatient_visits") or 0)
        ip_col = _ov_ip_color(ip)
        label  = str(r.get("burden_group", ""))
        disp   = label[:44] + "…" if len(label) > 46 else label
        op_w   = int(op_v / max_v * 380)
        ip_w   = int(ip_v / max_v * 380)
        rows += (
            f'<div style="display:flex;align-items:center;gap:8px;padding:6px 10px;'
            f'border-left:3px solid {ip_col};border-bottom:0.5px solid rgba(0,0,0,0.06);">'
            f'<div style="flex:0 0 220px;font-size:11px;font-weight:500;overflow:hidden;'
            f'text-overflow:ellipsis;white-space:nowrap;" title="{label}">{disp}</div>'
            f'<div style="flex:1;display:flex;align-items:center;">'
            f'<div style="width:{op_w}px;height:14px;background:#1D9E75;border-radius:3px 0 0 3px;flex-shrink:0;"></div>'
            f'<div style="width:{ip_w}px;height:14px;background:#E24B4A;border-radius:0 3px 3px 0;flex-shrink:0;"></div>'
            f'</div>'
            f'<div style="flex:0 0 108px;font-size:9px;color:#888780;text-align:right;">'
            f'{op_v:,} OP · {ip_v:,} IP</div>'
            f'<div style="flex:0 0 44px;font-size:10px;font-weight:700;color:{ip_col};text-align:right;">'
            f'{ip:.0f}% IP</div>'
            f'</div>\n'
        )
    return _HTML_BASE.format(_IP_LEGEND_HTML + rows)


def _sec_d_html(df: pd.DataFrame) -> str:
    if df.empty:
        return _HTML_BASE.format('<p style="padding:12px;font-size:11px;color:#9ca3af;">No data.</p>')
    max_r = float(df["recent_90d_visits"].max() or 1)
    max_g = float(df["mom_growth_pct"].abs().max() or 1)
    hdr = (
        '<div style="display:grid;grid-template-columns:2fr 1.8fr 1fr 54px;gap:8px;'
        'padding:6px 10px 6px 13px;background:#f8fafc;border-bottom:1px solid rgba(0,0,0,0.08);">'
        + "".join(
            f'<div style="font-size:9px;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:.08em;color:#888780;{"text-align:center;" if h=="IP %" else ""}">{h}</div>'
            for h in ["Condition", "Recent 90d volume", "MoM growth", "IP %"]
        )
        + '</div>'
    )
    rows = ""
    n = len(df)
    for i, (_, r) in enumerate(df.iterrows()):
        ip     = float(r.get("inpatient_pct") or 0)
        recent = int(r.get("recent_90d_visits") or 0)
        prior  = int(r.get("prior_90d_visits") or 0)
        growth = float(r.get("mom_growth_pct") or 0)
        ip_col = _ov_ip_color(ip)
        label  = str(r.get("condition", ""))
        small  = recent < 50
        vol_w  = int(recent / max_r * 100)
        grw_w  = int(abs(growth) / max_g * 100)
        bar_c  = "rgba(239,159,39,0.6)" if small else "#1D9E75"
        txt_c  = "#854F0B" if small else "#0F6E56"
        g_str  = f"+{growth:.0f}%" if growth >= 0 else f"{growth:.0f}%"
        sb_div = ('<div style="font-size:9px;color:#888780;margin-top:2px;">small base — watch</div>'
                  if small else "")
        bb     = "" if i == n - 1 else "border-bottom:0.5px solid rgba(0,0,0,0.06);"
        rows += (
            f'<div style="display:grid;grid-template-columns:2fr 1.8fr 1fr 54px;gap:8px;'
            f'padding:7px 10px;border-left:3px solid {ip_col};{bb}">'
            f'<div style="padding-left:6px;">'
            f'<div style="font-size:11px;font-weight:500;">'
            f'{label[:48]}{"…" if len(label)>50 else ""}</div>'
            f'{sb_div}</div>'
            f'<div>'
            f'<div style="background:#378ADD;height:8px;border-radius:3px;'
            f'width:{vol_w}%;margin-bottom:3px;"></div>'
            f'<div style="font-size:9px;color:#888780;">'
            f'{recent:,} visits (vs {prior:,} prior 90d)</div></div>'
            f'<div>'
            f'<div style="background:{bar_c};height:8px;border-radius:3px;'
            f'width:{grw_w}%;margin-bottom:3px;"></div>'
            f'<div style="font-size:9px;font-weight:500;color:{txt_c};">{g_str}</div></div>'
            f'<div style="font-size:11px;font-weight:700;color:{ip_col};text-align:center;">'
            f'{ip:.0f}%</div>'
            f'</div>\n'
        )
    return _HTML_BASE.format(hdr + rows)


def _sec_e_html(df: pd.DataFrame) -> str:
    if df.empty:
        return _HTML_BASE.format('<p style="padding:12px;font-size:11px;color:#9ca3af;">No data.</p>')
    _pb = {
        "Cash":                   ("background:#FAEEDA;color:#633806", "Cash"),
        "NHIF / SHA":             ("background:#E6F1FB;color:#185FA5", "NHIF / SHA"),
        "Insurance / Corporate":  ("background:#E6F1FB;color:#185FA5", "Insurance / Corp"),
    }
    hdr = (
        '<div style="display:grid;grid-template-columns:2.2fr 1.4fr 1.2fr 1fr 70px;gap:8px;'
        'padding:6px 10px 6px 13px;background:#f8fafc;border-bottom:1px solid rgba(0,0,0,0.08);">'
        + "".join(
            f'<div style="font-size:9px;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:.08em;color:#888780;{"text-align:right;" if h=="90d visits" else ""}">{h}</div>'
            for h in ["Condition", "Primary demographic", "Visit type split", "Primary payer", "90d visits"]
        )
        + '</div>'
    )
    rows = ""
    n = len(df)
    for i, (_, r) in enumerate(df.iterrows()):
        ip     = float(r.get("ip_pct") or 0)
        op     = float(r.get("op_pct") or 100 - ip)
        ip_col = _ov_ip_color(ip)
        ip_bg  = _ov_ip_bg(ip)
        label  = str(r.get("condition", ""))
        age    = str(r.get("primary_age_group", "—"))
        gender = str(r.get("primary_gender", "—"))
        payer  = str(r.get("primary_payer", "—"))
        visits = int(r.get("total_visits") or 0)
        pstyle, plabel = _pb.get(payer, ("background:#f3f4f6;color:#374151", payer[:20]))
        op_w   = min(int(op * 0.6), 60)
        ip_w   = min(int(ip * 0.6), 12)
        bb     = "" if i == n - 1 else "border-bottom:0.5px solid rgba(0,0,0,0.06);"
        rows += (
            f'<div style="display:grid;grid-template-columns:2.2fr 1.4fr 1.2fr 1fr 70px;gap:8px;'
            f'padding:7px 10px;border-left:3px solid {ip_col};{bb}background:{ip_bg};">'
            f'<div style="font-size:11px;font-weight:500;padding-left:8px;overflow:hidden;'
            f'text-overflow:ellipsis;white-space:nowrap;">{label[:48]}{"…" if len(label)>50 else ""}</div>'
            f'<div style="font-size:10px;color:#5f5e5a;">{age} · {gender}</div>'
            f'<div style="display:flex;flex-direction:column;gap:2px;">'
            f'<div style="display:flex;">'
            f'<div style="width:{op_w}px;height:8px;background:#1D9E75;border-radius:3px 0 0 3px;"></div>'
            f'<div style="width:{ip_w}px;height:8px;background:#E24B4A;border-radius:0 3px 3px 0;"></div>'
            f'</div>'
            f'<div style="font-size:9px;color:#888780;">{op:.0f}% OP · {ip:.0f}% IP</div></div>'
            f'<div><span style="{pstyle};font-size:9px;font-weight:700;padding:2px 7px;'
            f'border-radius:12px;display:inline-block;">{plabel}</span></div>'
            f'<div style="font-size:11px;font-weight:500;text-align:right;">{visits:,}</div>'
            f'</div>\n'
        )
    return _HTML_BASE.format(hdr + rows)


def render_tab4_disease_burden(filters: dict, run_query):
    st_a, st_b, st_c, st_d, st_e = st.tabs([
        "Overview", "NCD & Chronic", "RMNCH",
        "Communicable & HIV", "Mental Health & Psychiatric",
    ])

    # ── OVERVIEW TAB ─────────────────────────────────────────────────────────
    with st_a:

        # ── SECTION A — KPIs ─────────────────────────────────────────────────
        _sh("A — Disease Burden Overview")
        try:
            c1, c2, c3, c4, c5 = st.columns(5)
            df_kpi = Q.load_burden_kpis(filters, run_query)
            if not df_kpi.empty:
                row = df_kpi.iloc[0]
                with c1:
                    _kpi("Diagnosed Visits", _n(row.get("total_diagnosed")), "Last 12 months")
                with c2:
                    _kpi("Comorbidity Rate", _p(row.get("comorbidity_rate_pct")),
                         "Patients with 2+ conditions", ORANGE)
                with c3:
                    _kpi("NCD Share", _p(row.get("ncd_share_pct")), "Of all diagnosed visits")
                with c4:
                    _kpi("Communicable Share", _p(row.get("communicable_share_pct")),
                         "Of all diagnosed visits")
            try:
                df_leak = Q.load_ncd_leakage_kpi(filters, run_query)
                if not df_leak.empty:
                    lr = df_leak.iloc[0]
                    leak = float(lr.get("estimated_leakage_kes") or 0)
                    undetected = int(lr.get("undetected_ncd_patients") or 0)
                    leak_fmt = f"KES {leak/1e6:.1f}M" if leak >= 1e6 else f"KES {leak:,.0f}"
                    with c5:
                        st.markdown(
                            f'<div style="border:0.5px solid rgba(228,75,74,0.3);border-radius:8px;'
                            f'padding:10px 12px;background:#fff;">'
                            f'<div style="font-size:9px;font-weight:700;text-transform:uppercase;'
                            f'letter-spacing:.08em;color:#888780;margin-bottom:4px;">NCD Billing Leakage</div>'
                            f'<div style="font-size:20px;font-weight:700;color:#A32D2D;">{leak_fmt}</div>'
                            f'<div style="font-size:10px;color:#888780;margin-top:3px;">'
                            f'{undetected:,} pts with elevated vitals · no NCD code</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
            except Exception:
                pass
        except Exception as e:
            st.warning(f"Section A KPIs: {e}")

        _gap(16)

        # ── SECTION B — TOP 5 DISEASE GROUPS GROWTH ──────────────────────────
        _sh("B — Top 5 Disease Groups — Visit Growth Over Time", mt=8)
        try:
            df_bt = Q.load_burden_trend(filters, run_query)
            if not df_bt.empty:
                df_bt["visit_count"] = pd.to_numeric(df_bt["visit_count"], errors="coerce")
                df_bt["visit_month"] = pd.to_datetime(df_bt["visit_month"])
                top5_groups = (df_bt.groupby("burden_group")["visit_count"]
                               .sum().nlargest(5).index.tolist())
                df_top5 = df_bt[df_bt["burden_group"].isin(top5_groups)].copy()
                _B5C = ["#378ADD", "#E24B4A", "#1D9E75", "#7F77DD", "#EF9F27"]

                # Per-group legend metadata
                _leg_items = []
                for i, grp in enumerate(top5_groups):
                    sub = df_top5[df_top5["burden_group"] == grp].sort_values("visit_month")
                    counts = [float(v) for v in sub["visit_count"].tolist()]
                    first = counts[0] if counts else 1
                    last  = counts[-1] if counts else first
                    eg    = round((last - first) / max(first, 1) * 100)
                    _leg_items.append((grp, _B5C[i % 5], eg))

                # Toggle
                _b_mode = st.radio(
                    "",
                    options=["Growth index", "Absolute visits"],
                    index=0,
                    horizontal=True,
                    key="_burden_ov_b_mode",
                    label_visibility="collapsed",
                )

                # Legend row (dots + name + end growth %)
                leg_html = '<div style="display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-bottom:6px;">'
                for grp, col, eg in _leg_items:
                    eg_col = "#0F6E56" if eg >= 0 else "#A32D2D"
                    eg_str = f"+{eg}%" if eg >= 0 else f"{eg}%"
                    leg_html += (
                        f'<span style="display:flex;align-items:center;gap:4px;font-size:10px;">'
                        f'<span style="width:8px;height:8px;border-radius:50%;background:{col};'
                        f'display:inline-block;flex-shrink:0;"></span>'
                        f'{grp}&nbsp;<span style="color:{eg_col};font-weight:700;">{eg_str}</span></span>'
                    )
                leg_html += '</div>'
                st.markdown(leg_html, unsafe_allow_html=True)

                # Build chart
                fig_b = go.Figure()
                for i, (grp, col, _) in enumerate(_leg_items):
                    sub = df_top5[df_top5["burden_group"] == grp].sort_values("visit_month")
                    months = sub["visit_month"].tolist()
                    counts = [float(v) for v in sub["visit_count"].tolist()]

                    if _b_mode == "Growth index":
                        first = counts[0] if counts else 1
                        raw_idx = [round(c / max(first, 1) * 100) for c in counts]
                        y_vals = _ra3(raw_idx)
                    else:
                        y_vals = _ra3([int(c) for c in counts])

                    fig_b.add_trace(go.Scatter(
                        x=months, y=y_vals,
                        name=grp, mode="lines",
                        line=dict(color=col, width=2),
                        connectgaps=False,
                        hovertemplate=f"<b>{grp}</b><br>%{{x|%b %Y}}: %{{y}}<extra></extra>",
                    ))

                if _b_mode == "Growth index":
                    fig_b.add_hline(y=100, line_dash="dot",
                                    line_color="rgba(0,0,0,0.15)", line_width=1)
                    y_title = "Index (first month = 100)"
                else:
                    y_title = "Visits"

                fig_b.update_layout(
                    height=320, margin=dict(l=0, r=20, t=10, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(title=y_title, rangemode="tozero",
                               gridcolor="rgba(0,0,0,0.06)",
                               tickfont=dict(size=9, color="#888780")),
                    xaxis=dict(title="", tickfont=dict(size=9, color="#888780")),
                    showlegend=False,
                )
                _pc(fig_b)

                # Insight
                fastest = max(_leg_items, key=lambda x: x[2])
                slowest = min(_leg_items, key=lambda x: x[2])
                _note(
                    f"Fastest growing: {fastest[0]} "
                    f"{'+'if fastest[2]>=0 else ''}{fastest[2]}%  ·  "
                    f"Slowest: {slowest[0]} "
                    f"{'+'if slowest[2]>=0 else ''}{slowest[2]}%"
                )
        except Exception as e:
            st.warning(f"Section B: {e}")

        _gap(16)

        # ── SECTION C — TOP DIAGNOSES IP/OP SPLIT ────────────────────────────
        _sh("C — Visit Volume by Diagnosis — Inpatient Share", mt=8)
        st.caption(
            "A surge in a low IP% condition is high-volume but low-margin. "
            "A surge in a high IP% condition signals chronic disease load and long-term retention value."
        )
        try:
            df_td = Q.load_top_diagnoses_ip_op(filters, run_query)
            if not df_td.empty:
                for _c in ("total_visits", "inpatient_visits", "outpatient_visits", "ip_pct", "op_pct"):
                    df_td[_c] = pd.to_numeric(df_td[_c], errors="coerce")
                df_td = df_td.sort_values("total_visits", ascending=False)

                _stcomp.html(
                    _sec_c_html(df_td),
                    height=len(df_td) * 34 + 70,
                    scrolling=False,
                )

                # Insight
                top_ip  = df_td.nlargest(2, "ip_pct")
                top_vol = df_td.nlargest(2, "total_visits")
                _hi_names  = " & ".join(top_ip["burden_group"].tolist())
                _vol_names = " & ".join(top_vol["burden_group"].tolist())
                _note(
                    f"{_hi_names} have the highest inpatient share — "
                    "these conditions drive the most long-term retention value. "
                    f"{_vol_names} dominate by volume but are overwhelmingly outpatient."
                )
            else:
                df_td2 = Q.load_top_diagnoses(filters, run_query)
                if not df_td2.empty:
                    _pc(hbar_chart(df_td2, x="visit_count", y="disease_group",
                                   color=AFYA_BLUE, x_label="Visits", height=320))
        except Exception as e:
            st.warning(f"Section C: {e}")

        _gap(16)

        # ── SECTION D — EMERGING MID-TIER DIAGNOSES ──────────────────────────
        _sh("D — Emerging Mid-Tier Diagnoses", mt=8)
        st.caption(
            "Conditions outside the top 5 showing the highest month-over-month growth. "
            "Volume bar shows absolute visit count — growth rate alone is misleading at small bases."
        )
        try:
            df_em = Q.load_emerging_diagnoses_90d(filters, run_query)
            if not df_em.empty:
                for _c in ("recent_90d_visits", "prior_90d_visits", "mom_growth_pct", "inpatient_pct"):
                    df_em[_c] = pd.to_numeric(df_em[_c], errors="coerce")
                df_em = df_em.head(10)

                _stcomp.html(
                    _sec_d_html(df_em),
                    height=len(df_em) * 48 + 50,
                    scrolling=False,
                )

                # Insight
                _small = df_em[df_em["recent_90d_visits"] < 50].sort_values(
                    "mom_growth_pct", ascending=False)
                _solid = df_em[df_em["recent_90d_visits"] >= 50].sort_values(
                    "mom_growth_pct", ascending=False)
                if not _small.empty and not _solid.empty:
                    _sb = _small.iloc[0]
                    _ss = _solid.iloc[0]
                    _note(
                        f"{_sb['condition']} shows the highest growth at "
                        f"+{_sb['mom_growth_pct']:.0f}% but on a base of only "
                        f"{int(_sb['recent_90d_visits']):,} visits — watch but do not overweight. "
                        f"{_ss['condition']} at +{_ss['mom_growth_pct']:.0f}% growth on "
                        f"{int(_ss['recent_90d_visits']):,} visits is the more clinically significant signal."
                    )
                elif not _solid.empty:
                    _ss = _solid.iloc[0]
                    _note(
                        f"Fastest emerging: {_ss['condition']} at "
                        f"+{_ss['mom_growth_pct']:.0f}% growth on "
                        f"{int(_ss['recent_90d_visits']):,} visits."
                    )
        except Exception as e:
            st.warning(f"Section D: {e}")

        _gap(16)

        # ── SECTION E — DISEASE INTELLIGENCE MATRIX ──────────────────────────
        _sh("E — Disease Intelligence Matrix", mt=8)
        st.caption(
            "Conditions ranked by visit volume — IP tier colour on the left border matches "
            "the encoding in Sections C and D. Trend column removed; data not reliable enough to display."
        )
        try:
            df_dim = Q.load_disease_intelligence_matrix(filters, run_query)
            if not df_dim.empty:
                for _c in ("total_visits", "ip_pct", "op_pct"):
                    if _c in df_dim.columns:
                        df_dim[_c] = pd.to_numeric(df_dim[_c], errors="coerce")
                df_dim = df_dim.sort_values("total_visits", ascending=False)

                _stcomp.html(
                    _sec_e_html(df_dim),
                    height=len(df_dim) * 44 + 50,
                    scrolling=False,
                )
        except Exception as e:
            st.warning(f"Section E: {e}")

    # ── NCD & CHRONIC TAB ────────────────────────────────────────────────────
    with st_b:
        _sh("B — NCD & Chronic Disease")

        # ── B1: KPI ROW ──────────────────────────────────────────────────────
        try:
            df_bkpi = Q.load_ncd_kpis(filters, run_query)
            if not df_bkpi.empty:
                row = df_bkpi.iloc[0]
                comorb_pct = float(row.get("comorbidity_rate_pct") or 0)
                htn_pct    = float(row.get("controlled_htn_pct") or 0)
                c1, c2, c3, c4 = st.columns(4)
                with c1: _kpi("NCD Patients",    _n(row.get("ncd_patients")), color=AFYA_BLUE)
                with c2: _kpi("Comorbidity Rate", _p(comorb_pct),
                               "patients with 2+ NCDs",
                               ORANGE if comorb_pct >= 20 else AFYA_BLUE)
                with c3: _kpi("Controlled HTN",   _p(htn_pct),
                               "avg BP <140/90 — benchmark 60%",
                               TEAL if htn_pct >= 60 else CORAL)
                with c4: _kpi("Undetected NCD",  _n(row.get("undetected_ncd_patients")),
                               "elevated vitals, no NCD code", CORAL)
        except Exception as e:
            st.warning(f"B1: {e}")

        _gap(12)

        # ── B2: TOP NCDs RANKED + GENDER ─────────────────────────────────────
        _sh("Top NCDs — Patient Volume by Condition & Gender", mt=8)
        _note("Diabetes and Endocrine & Metabolic are consolidated — they overlap heavily in ICD10 coding and represent the same patient risk profile.")
        try:
            df_rk = Q.load_ncd_ranked_with_gender(filters, run_query)
            if not df_rk.empty:
                df_rk["patient_count"] = pd.to_numeric(df_rk["patient_count"], errors="coerce")
                cond_totals = (df_rk.groupby("ncd_group")["patient_count"]
                               .sum().sort_values(ascending=False))
                top_conds = cond_totals.head(10).index.tolist()
                df_rk_top = df_rk[df_rk["ncd_group"].isin(top_conds)].copy()
                c1, c2 = st.columns([1.4, 0.6])
                with c1:
                    st.markdown("**Patient count by NCD group and gender:**")
                    pivot_g = (df_rk_top.pivot_table(
                        index="ncd_group", columns="gender",
                        values="patient_count", aggfunc="sum", fill_value=0
                    ).reindex(top_conds))
                    color_map_g = {"FEMALE": PURPLE, "MALE": AFYA_BLUE, "F": PURPLE,
                                   "M": AFYA_BLUE, "Unknown": GRAY}
                    fig_rk = go.Figure()
                    for g in pivot_g.columns.tolist():
                        fig_rk.add_trace(go.Bar(
                            x=pivot_g[g].values, y=pivot_g.index.tolist(),
                            name=g, orientation="h",
                            marker_color=color_map_g.get(g, TEAL),
                            hovertemplate=f"<b>%{{y}}</b><br>{g}: %{{x:,}}<extra></extra>",
                        ))
                    fig_rk.update_layout(
                        barmode="stack", height=340,
                        margin=dict(l=0, r=20, t=10, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(title="Patients", rangemode="tozero"),
                        yaxis=dict(title="", autorange="reversed"),
                        legend=dict(orientation="h", y=1.08, xanchor="right", x=1),
                    )
                    _pc(fig_rk)
                with c2:
                    st.markdown("**Totals:**")
                    _pc(table_fig(
                        cond_totals.head(10).reset_index().rename(columns={
                            "ncd_group": "Condition", "patient_count": "Patients"
                        }),
                        col_labels={}, height=340,
                    ))
        except Exception as e:
            st.warning(f"B2: {e}")

        _gap(12)

        # ── B3: NCD COMPLEXITY ────────────────────────────────────────────────
        _sh("NCD Complexity — Simple vs Multi-Morbidity Cases", mt=8)
        _note("Patients with 2+ NCDs are 3-4x more expensive to manage. The share of complex cases drives case management staffing and chronic care protocol design.")
        try:
            df_cx = Q.load_ncd_complexity_distribution(filters, run_query)
            if not df_cx.empty:
                df_cx["patient_count"]      = pd.to_numeric(df_cx["patient_count"],      errors="coerce")
                df_cx["pct_of_ncd_patients"] = pd.to_numeric(df_cx["pct_of_ncd_patients"], errors="coerce")
                c1, c2 = st.columns(2)
                with c1:
                    _pc(donut(
                        labels=df_cx["ncd_complexity"].tolist(),
                        values=df_cx["patient_count"].tolist(),
                        color_map={
                            "1 NCD": TEAL, "2 NCDs": ORANGE,
                            "3 NCDs": CORAL, "4+ NCDs (Complex)": PURPLE,
                        },
                        height=280,
                    ))
                with c2:
                    st.dataframe(
                        df_cx[["ncd_complexity", "patient_count", "pct_of_ncd_patients"]].rename(
                            columns={"ncd_complexity": "Complexity",
                                     "patient_count": "Patients",
                                     "pct_of_ncd_patients": "% of NCD Patients"}
                        ),
                        use_container_width=True, hide_index=True, height=220,
                        column_config={
                            "% of NCD Patients": st.column_config.NumberColumn(format="%.1f%%")
                        },
                    )
                    complex_pct = float(
                        df_cx.loc[df_cx["ncd_complexity"] != "1 NCD",
                                  "pct_of_ncd_patients"].sum()
                    )
                    _note(
                        f"{complex_pct:.0f}% of NCD patients carry 2+ conditions. "
                        "These patients need integrated management protocols.",
                        w=(complex_pct > 30),
                    )
        except Exception as e:
            st.warning(f"B3: {e}")

        _gap(12)

        # ── B4: NCD BY AGE GROUP HEATMAP ─────────────────────────────────────
        _sh("NCD Distribution by Age Group", mt=8)
        _note("Dark cells = high patient concentration. Use this to design age-targeted care pathways.")
        try:
            df_hm = Q.load_ncd_age_heatmap(filters, run_query)
            if not df_hm.empty:
                df_hm["patient_count"] = pd.to_numeric(df_hm["patient_count"], errors="coerce")
                pivot_hm = df_hm.pivot_table(
                    index="age_group", columns="chronic_condition",
                    values="patient_count", aggfunc="sum", fill_value=0
                )
                age_order = ["Under 18", "18-34", "35-49", "50-64", "65+"]
                pivot_hm = pivot_hm.reindex([a for a in age_order if a in pivot_hm.index])
                fig_hm = go.Figure(go.Heatmap(
                    z=pivot_hm.values,
                    x=pivot_hm.columns.tolist(),
                    y=pivot_hm.index.tolist(),
                    colorscale=[[0, "#EBF3FB"], [0.5, "#0072CE"], [1, "#003F88"]],
                    hovertemplate="<b>%{y} — %{x}</b><br>Patients: %{z:,}<extra></extra>",
                    text=pivot_hm.values, texttemplate="%{text}",
                    textfont=dict(size=10, color="white"),
                ))
                fig_hm.update_layout(
                    height=300, margin=dict(l=0, r=20, t=20, b=60),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(tickangle=-30, title=""),
                    yaxis=dict(title="Age Group"),
                )
                _pc(fig_hm)
        except Exception as e:
            st.warning(f"B4: {e}")

        _gap(12)

        # ── B5: COMORBIDITY PAIRS ─────────────────────────────────────────────
        _sh("Comorbidity Pipeline — Top Condition Pairs & Progression Speed", mt=8)
        _note("How quickly do patients develop a second NCD? Diabetes & Endocrine/Metabolic consolidated into one group.")
        try:
            df_cp = Q.load_chronic_comorbidity_pairs(filters, run_query)
            if not df_cp.empty:
                df_cp["avg_days_between_diagnoses"] = pd.to_numeric(
                    df_cp["avg_days_between_diagnoses"], errors="coerce")
                df_cp["patient_count"] = pd.to_numeric(df_cp["patient_count"], errors="coerce")
                c1, c2 = st.columns([1.3, 0.7])
                with c1:
                    _pc(hbar_chart(
                        df_cp.head(10), x="patient_count", y="condition_pair",
                        x_label="Patients", color=AFYA_BLUE, height=320, show_text=True,
                    ))
                with c2:
                    _pc(table_fig(
                        df_cp[["condition_pair", "patient_count",
                               "avg_days_between_diagnoses"]].head(10),
                        col_labels={"condition_pair": "Pair",
                                    "patient_count": "Patients",
                                    "avg_days_between_diagnoses": "Avg Days to 2nd Dx"},
                        height=320,
                    ))
                fastest = df_cp.dropna(subset=["avg_days_between_diagnoses"])
                if not fastest.empty:
                    f_row = fastest.loc[fastest["avg_days_between_diagnoses"].idxmin()]
                    _note(
                        f"Fastest progression: {f_row['condition_pair']} — avg "
                        f"{int(f_row['avg_days_between_diagnoses'])} days to second diagnosis. "
                        "This pair needs a co-management protocol."
                    )
        except Exception as e:
            st.warning(f"B5: {e}")

        _gap(12)

        # ── B6: HTN CONTROLLED vs UNCONTROLLED ───────────────────────────────
        _sh("HTN: Controlled vs Uncontrolled — Annual Visits & Doctors Seen", mt=8)
        _note("Controlled patients follow a consistent single-doctor cadence. Uncontrolled patients exhibit doctor-shopping and erratic visit gaps.")
        try:
            df_sc = Q.load_htn_scatter_data(filters, run_query)
            if not df_sc.empty:
                for _c in ("avg_annual_visits", "unique_doctors"):
                    df_sc[_c] = pd.to_numeric(df_sc[_c], errors="coerce")
                controlled   = df_sc[df_sc["htn_status"] == "Controlled"]
                uncontrolled = df_sc[df_sc["htn_status"] == "Uncontrolled"]
                fig_sc = go.Figure()
                for grp, color, name in [
                    (controlled,   TEAL,  "Controlled (BP <140/90)"),
                    (uncontrolled, CORAL, "Uncontrolled (BP >=140/90)"),
                ]:
                    if not grp.empty:
                        fig_sc.add_trace(go.Scatter(
                            x=grp["avg_annual_visits"], y=grp["unique_doctors"],
                            mode="markers", name=name,
                            marker=dict(color=color, size=8, opacity=0.7),
                            hovertemplate=(f"<b>{name}</b><br>Annual visits: %{{x:.1f}}<br>"
                                           "Unique doctors: %{y}<extra></extra>"),
                        ))
                fig_sc.update_layout(
                    height=300, margin=dict(l=0, r=20, t=20, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(title="Avg Annual Visits", rangemode="tozero"),
                    yaxis=dict(title="Unique Doctors Seen", rangemode="tozero"),
                    legend=dict(orientation="h", y=1.08, xanchor="right", x=1),
                )
                _pc(fig_sc)
                if not controlled.empty and not uncontrolled.empty:
                    ctrl_docs   = float(controlled["unique_doctors"].median() or 0)
                    unctrl_docs = float(uncontrolled["unique_doctors"].median() or 0)
                    _note(
                        f"Median unique doctors: {ctrl_docs:.0f} (Controlled) vs "
                        f"{unctrl_docs:.0f} (Uncontrolled). "
                        + ("Uncontrolled patients see significantly more doctors — "
                           "care fragmentation is likely driving poor BP management."
                           if unctrl_docs > ctrl_docs * 1.3
                           else "Doctor frequency is similar; investigate medication adherence.")
                    )
            else:
                df_htn = Q.load_htn_controlled(filters, run_query)
                if not df_htn.empty:
                    c1, c2 = st.columns(2)
                    with c1:
                        _pc(donut(labels=df_htn["htn_status"].tolist(),
                                  values=df_htn["patient_count"].tolist(),
                                  color_map={"Controlled": TEAL, "Uncontrolled": CORAL,
                                             "No BP Recorded": GRAY},
                                  height=260))
                    with c2:
                        _pc(table_fig(df_htn,
                                      col_labels={"htn_status": "Status",
                                                  "patient_count": "Patients",
                                                  "avg_systolic": "Avg Systolic"},
                                      height=180))
        except Exception as e:
            st.warning(f"B6: {e}")

        _gap(12)

        # ── B7: UNCONTROLLED HTN PROFILE ──────────────────────────────────────
        _sh("Uncontrolled HTN — Who Are They & Why?", mt=8)
        _note("Age, payer type, comorbidity burden, medication use, and investigation rate for HTN patients. Reveals whether uncontrolled status is linked to demographics, multi-morbidity, or medication gaps.")
        try:
            df_hp = Q.load_htn_uncontrolled_profile(filters, run_query)
            if not df_hp.empty:
                for _c in ("patient_count", "avg_investigations", "avg_visits", "avg_systolic"):
                    if _c in df_hp.columns:
                        df_hp[_c] = pd.to_numeric(df_hp[_c], errors="coerce")

                uc = df_hp[df_hp["htn_status"] == "Uncontrolled"]
                if not uc.empty:
                    total_uc = int(uc["patient_count"].sum())
                    on_rx_n  = df_hp[
                        (df_hp["htn_status"] == "Uncontrolled") &
                        (pd.to_numeric(df_hp["on_antihypertensive"], errors="coerce") == 1)
                    ]["patient_count"].sum()
                    on_rx_pct = on_rx_n / max(total_uc, 1) * 100
                    avg_inv = ((uc["avg_investigations"] * uc["patient_count"]).sum()
                               / max(total_uc, 1))
                    k1, k2, k3 = st.columns(3)
                    with k1: _kpi("Uncontrolled HTN",       _n(total_uc), color=CORAL)
                    with k2: _kpi("On Antihypertensive Rx", f"{on_rx_pct:.0f}%",
                                  "prescribed despite uncontrolled BP",
                                  ORANGE if on_rx_pct > 40 else CORAL)
                    with k3: _kpi("Avg Investigations",      f"{avg_inv:.1f}",
                                  "investigations per patient", AFYA_BLUE)
                    _gap(8)

                c1, c2 = st.columns(2)
                with c1:
                    age_grp = (df_hp.groupby(["htn_status", "age_group"])["patient_count"]
                               .sum().reset_index())
                    fig_age = go.Figure()
                    for status, color in [("Controlled", TEAL), ("Uncontrolled", CORAL),
                                          ("No BP Recorded", GRAY)]:
                        sub = age_grp[age_grp["htn_status"] == status]
                        if not sub.empty:
                            fig_age.add_trace(go.Bar(
                                x=sub["age_group"], y=sub["patient_count"],
                                name=status, marker_color=color,
                                hovertemplate=f"<b>%{{x}}</b><br>{status}: %{{y:,}}<extra></extra>",
                            ))
                    fig_age.update_layout(
                        barmode="group", height=260,
                        margin=dict(l=0, r=0, t=30, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(title="Age Group"),
                        yaxis=dict(title="Patients", rangemode="tozero"),
                        legend=dict(orientation="h", y=1.1, xanchor="right", x=1),
                        title=dict(text="By Age Group", font=dict(size=12)),
                    )
                    _pc(fig_age)

                with c2:
                    pay_grp = (df_hp.groupby(["htn_status", "payer"])["patient_count"]
                               .sum().reset_index())
                    fig_pay = go.Figure()
                    for status, color in [("Controlled", TEAL), ("Uncontrolled", CORAL),
                                          ("No BP Recorded", GRAY)]:
                        sub = pay_grp[pay_grp["htn_status"] == status]
                        if not sub.empty:
                            fig_pay.add_trace(go.Bar(
                                x=sub["payer"], y=sub["patient_count"],
                                name=status, marker_color=color,
                                hovertemplate=f"<b>%{{x}}</b><br>{status}: %{{y:,}}<extra></extra>",
                            ))
                    fig_pay.update_layout(
                        barmode="group", height=260,
                        margin=dict(l=0, r=0, t=30, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(title="Payer"),
                        yaxis=dict(title="Patients", rangemode="tozero"),
                        legend=dict(orientation="h", y=1.1, xanchor="right", x=1),
                        title=dict(text="By Payer Type", font=dict(size=12)),
                    )
                    _pc(fig_pay)

                _gap(8)
                comorb_grp = (
                    df_hp.groupby(["htn_status", "comorbidity_group"])
                    .apply(lambda d: pd.Series({
                        "patients": d["patient_count"].sum(),
                        "avg_inv":  ((d["avg_investigations"] * d["patient_count"]).sum()
                                    / max(d["patient_count"].sum(), 1)),
                        "on_rx_pct": (d.loc[
                            pd.to_numeric(d["on_antihypertensive"], errors="coerce") == 1,
                            "patient_count"
                        ].sum() / max(d["patient_count"].sum(), 1) * 100),
                    }), include_groups=False)
                    .reset_index()
                )
                st.dataframe(
                    comorb_grp.rename(columns={
                        "htn_status": "HTN Status", "comorbidity_group": "Comorbidity",
                        "patients": "Patients", "avg_inv": "Avg Investigations",
                        "on_rx_pct": "On Antihypertensive %",
                    }),
                    use_container_width=True, hide_index=True,
                    height=min(400, len(comorb_grp) * 35 + 50),
                    column_config={
                        "Avg Investigations":    st.column_config.NumberColumn(format="%.1f"),
                        "On Antihypertensive %": st.column_config.NumberColumn(format="%.0f%%"),
                    },
                )
        except Exception as e:
            st.warning(f"B7: {e}")

        _gap(12)

        # ── B8: PRESCRIPTION WITHOUT CLINICAL ASSESSMENT ──────────────────────
        _sh("Prescription Without Clinical Assessment — Documentation Gap", mt=8)
        _note(
            "Chronic visits where a prescription was issued but NO vitals were recorded AND "
            "NO clinical note exists. May indicate a documentation gap or a true pharmacy-only "
            "refill without clinical review. Either way it is a governance risk."
        )
        try:
            df_ph = Q.load_chronic_pharmacy_only(filters, run_query)
            if not df_ph.empty:
                for _c in ("patient_count", "pharmacy_only_pct", "avg_annual_revenue"):
                    if _c in df_ph.columns:
                        df_ph[_c] = pd.to_numeric(df_ph[_c], errors="coerce")
                if "pharmacy_only_visits" in df_ph.columns:
                    total_gap = int(pd.to_numeric(
                        df_ph["pharmacy_only_visits"], errors="coerce"
                    ).sum())
                    if total_gap > 0:
                        _note(
                            f"{total_gap:,} visits have a prescription but no recorded vitals or "
                            "clinical note. Investigate whether these are genuine assessments with "
                            "missing documentation or true unreviewed refills.",
                            w=True,
                        )
                _pc(table_fig(
                    df_ph[["payer", "condition", "patient_count",
                            "pharmacy_only_pct", "avg_annual_revenue"]].head(20),
                    col_labels={
                        "payer": "Payer", "condition": "Condition",
                        "patient_count": "Patients",
                        "pharmacy_only_pct": "Gap Visit %",
                        "avg_annual_revenue": "Avg Annual Rev (KES)",
                    },
                    fmt={"pharmacy_only_pct": "pct", "avg_annual_revenue": "num"},
                    height=380,
                ))
        except Exception as e:
            st.warning(f"B8: {e}")

        _gap(12)

        # ── B9: CHRONIC CARE QUALITY & REVENUE MATRIX — DATA DRIVEN ──────────
        _sh("Chronic Care Quality & Revenue Matrix", mt=8)
        _note("Data-driven view of actual top NCD conditions. All metrics from live data. Trend = % change last 6 months vs prior 6 months.")
        try:
            df_qmx = Q.load_chronic_care_matrix(filters, run_query)
            if not df_qmx.empty:
                for _c in ("patient_count", "trend_pct", "ip_rate_pct",
                           "avg_visits_per_patient", "investigations_per_visit",
                           "avg_revenue_per_patient", "controlled_pct"):
                    if _c in df_qmx.columns:
                        df_qmx[_c] = pd.to_numeric(df_qmx[_c], errors="coerce")

                def _trend_icon(v):
                    if pd.isna(v): return "—"
                    return f"↑ {abs(v):.0f}%" if v > 5 else f"↓ {abs(v):.0f}%" if v < -5 else f"→ {v:.0f}%"

                disp_qmx = pd.DataFrame({
                    "Condition":       df_qmx["condition"],
                    "Patients":        df_qmx["patient_count"].fillna(0).astype(int),
                    "6-Mo Trend":      df_qmx["trend_pct"].apply(_trend_icon),
                    "IP Rate %":       df_qmx["ip_rate_pct"].fillna(0).round(1),
                    "Top Payer":       df_qmx.get("top_payer",
                                           pd.Series(["-"] * len(df_qmx))).fillna("-"),
                    "Visits/Patient":  df_qmx["avg_visits_per_patient"].fillna(0).round(1),
                    "Inv/Visit":       df_qmx["investigations_per_visit"].fillna(0).round(2),
                    "Avg Rev/Patient": df_qmx["avg_revenue_per_patient"].fillna(0).round(0),
                    "Controlled %":    df_qmx["controlled_pct"].apply(
                                           lambda v: f"{v:.0f}%" if pd.notna(v) else "-"),
                })
                st.dataframe(
                    disp_qmx, use_container_width=True, hide_index=True,
                    height=min(560, len(disp_qmx) * 38 + 50),
                    column_config={
                        "IP Rate %":      st.column_config.NumberColumn(format="%.1f%%"),
                        "Visits/Patient": st.column_config.NumberColumn(format="%.1f"),
                        "Inv/Visit":      st.column_config.NumberColumn(format="%.2f"),
                        "Avg Rev/Patient": st.column_config.NumberColumn(format="KES %,.0f"),
                    },
                )
        except Exception as e:
            st.warning(f"B9: {e}")

        _gap(12)

        # ── B10: UNDETECTED NCD RISK PATIENTS ────────────────────────────────
        _sh("Undetected NCD Risk — Elevated Vitals, No Chronic Diagnosis", mt=8)
        _note(
            "Patients with systolic BP >= 140 or blood sugar >= 10 on at least 2 separate visits, "
            "with no NCD diagnosis code. Highest clinical and billing risk."
        )
        try:
            df_ev = Q.load_elevated_vitals_no_ncd_patients(filters, run_query)
            if df_ev.empty:
                st.success("No undetected NCD risk patients found in this period.")
            else:
                for _c in ("visit_count", "latest_systolic",
                           "latest_blood_sugar", "days_since_last_visit"):
                    if _c in df_ev.columns:
                        df_ev[_c] = pd.to_numeric(df_ev[_c], errors="coerce")
                _note(
                    f"{len(df_ev):,} patients flagged. Assign to a chronic disease nurse for "
                    "screening within 30 days.",
                    w=True,
                )
                _pc(table_fig(
                    df_ev[["patient", "visit_count", "latest_systolic",
                           "latest_blood_sugar", "days_since_last_visit", "payer"]].head(40),
                    col_labels={
                        "patient": "Patient", "visit_count": "Flagged Visits",
                        "latest_systolic": "Latest Systolic",
                        "latest_blood_sugar": "Latest Blood Sugar",
                        "days_since_last_visit": "Days Since Last Visit",
                        "payer": "Payer",
                    },
                    height=min(520, len(df_ev.head(40)) * 30 + 60),
                ))
        except Exception as e:
            st.warning(f"B10: {e}")


    # ── RMNCH TAB ─────────────────────────────────────────────────────────────
    with st_c:
        _sh("C — RMNCH: Maternal, Child & Reproductive Health")

        # ── C0: Facility Profile Overview ─────────────────────────────────────
        _sh("Facility Profile — What Does This Hospital Do?", mt=8)
        _note("How visits are distributed across maternal care categories. Reveals whether the facility is primarily delivery-focused, ANC-focused, or balanced.")
        try:
            df_fac = Q.load_anc_vs_delivery_pnc(filters, run_query)
            if not df_fac.empty:
                total_visits = int(df_fac["visit_count"].sum() or 1)
                c1, c2, c3, c4 = st.columns(4)
                for col, cat, color in [
                    (c1, "Antenatal Care (ANC)", AFYA_BLUE),
                    (c2, "Delivery",              PURPLE),
                    (c3, "Postnatal Care (PNC)",  TEAL),
                    (c4, "Family Planning",        ORANGE),
                ]:
                    row_f = df_fac[df_fac["care_category"] == cat]
                    v = int(row_f["visit_count"].sum()) if not row_f.empty else 0
                    pct = v / total_visits * 100
                    with col:
                        _kpi(cat, _n(v), f"{pct:.1f}% of RMNCH visits", color)
                _gap(8)
                c1, c2 = st.columns(2)
                with c1:
                    _pc(bar_chart(df_fac, x="care_category", y="visit_count",
                                  color=PURPLE, y_label="Visits", height=260, show_text=True))
                with c2:
                    _pc(donut(labels=df_fac["care_category"].tolist(),
                              values=df_fac["visit_count"].tolist(),
                              height=260))
        except Exception as e:
            st.warning(f"Facility profile: {e}")

        _gap(12)

        # ── C1: ANC Retention Curve ────────────────────────────────────────────
        _sh("ANC Pathway — Retention Curve", mt=8)
        _note("How many women are retained through each ANC visit stage. Steep drops indicate the exact stage where patients defect.")
        try:
            df_anc = Q.load_anc_funnel(filters, run_query)
            if not df_anc.empty:
                row = df_anc.iloc[0]
                anc_vals = [int(row.get("anc1") or 0), int(row.get("anc2") or 0),
                            int(row.get("anc3") or 0), int(row.get("anc4") or 0)]
                stages = ["ANC 1", "ANC 2", "ANC 3", "ANC 4"]

                c1, c2 = st.columns([1.3, 0.7])
                with c1:
                    retention_pcts = [100.0] + [
                        anc_vals[i] / anc_vals[0] * 100 if anc_vals[0] > 0 else 0
                        for i in range(1, 4)
                    ]
                    drop_pcts = [
                        (anc_vals[i-1] - anc_vals[i]) / anc_vals[i-1] * 100
                        if anc_vals[i-1] > 0 else 0
                        for i in range(1, 4)
                    ]
                    fig_anc = go.Figure()
                    fig_anc.add_trace(go.Scatter(
                        x=stages, y=retention_pcts, mode="lines+markers+text",
                        line=dict(color=AFYA_BLUE, width=3),
                        marker=dict(size=12, color=AFYA_BLUE),
                        text=[f"{v:.0f}%" for v in retention_pcts],
                        textposition="top center",
                        fill="tozeroy", fillcolor="rgba(0,114,206,0.13)",
                        hovertemplate="<b>%{x}</b><br>Retention: %{y:.1f}%<extra></extra>",
                    ))
                    fig_anc.update_layout(
                        height=280, margin=dict(l=0, r=20, t=30, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        yaxis=dict(title="% Retained from ANC1", range=[0, 120]),
                        xaxis=dict(title=""), showlegend=False,
                    )
                    _pc(fig_anc)
                with c2:
                    worst_drop_idx = drop_pcts.index(max(drop_pcts)) if drop_pcts else 0
                    _kpi("ANC4 Completion", _p(row.get("anc4_completion_pct")),
                         "completing all 4 visits",
                         TEAL if float(row.get("anc4_completion_pct") or 0) >= 50 else CORAL)
                    _gap(8)
                    _kpi("Biggest Drop-off",
                         f"ANC{worst_drop_idx+1} → ANC{worst_drop_idx+2}",
                         f"{max(drop_pcts):.0f}% patient loss at this stage", CORAL)
        except Exception as e:
            st.warning(f"ANC funnel: {e}")

        _gap(12)

        # ── C2: ANC Dropout by Payer ───────────────────────────────────────────
        _sh("Why Is ANC Completion Low? — Dropout by Payer", mt=8)
        _note("ANC4 completion rates split by how patients pay. Cash patients typically drop out earlier due to unexpected billing costs at each visit.")
        try:
            df_pay = Q.load_anc_dropout_by_payer(filters, run_query)
            if not df_pay.empty:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**ANC4 Completion Rate by Payer**")
                    _pc(bar_chart(df_pay, x="payer_type", y="anc4_completion_pct",
                                  color_map={"Cash": CORAL, "NHIF / SHA": TEAL,
                                             "Insurance / Corporate": AFYA_BLUE},
                                  y_label="ANC4 Completion %", y_format="pct",
                                  height=260, show_text=True))
                with c2:
                    st.markdown("**Funnel — Patients Reaching Each Stage**")
                    fig_pay = go.Figure()
                    colors = {"Cash": CORAL, "NHIF / SHA": TEAL, "Insurance / Corporate": AFYA_BLUE}
                    for _, pr in df_pay.iterrows():
                        clr = colors.get(pr["payer_type"], AFYA_BLUE)
                        fig_pay.add_trace(go.Bar(
                            name=pr["payer_type"],
                            x=["ANC1", "ANC2", "ANC3", "ANC4"],
                            y=[pr["total_anc1_patients"], pr["reached_anc2"],
                               pr["reached_anc3"], pr["reached_anc4"]],
                            marker_color=clr,
                        ))
                    fig_pay.update_layout(
                        barmode="group", height=260,
                        margin=dict(l=0, r=0, t=30, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        yaxis_title="Patients",
                    )
                    _pc(fig_pay)
        except Exception as e:
            st.warning(f"ANC dropout by payer: {e}")

        _gap(12)

        # ── C3: Same vs Different Patients at Each Stage ───────────────────────
        _sh("Are the Same Patients Coming Back for ANC?", mt=8)
        _note("Patient cohort tracking: how many of the women who came for ANC1 are the same ones seen at ANC2, 3, 4. Breaks down by payer type and age group.")
        try:
            df_coh = Q.load_anc_patient_cohort_profile(filters, run_query)
            if not df_coh.empty:
                c1, c2 = st.columns(2)
                with c1:
                    journey_summary = df_coh.groupby("anc_journey")["patient_count"].sum().reset_index()
                    journey_summary = journey_summary.sort_values("anc_journey")
                    _pc(bar_chart(journey_summary, x="anc_journey", y="patient_count",
                                  color=AFYA_BLUE, y_label="Patients", height=280, show_text=True))
                with c2:
                    payer_summary = df_coh.groupby("payer_type")["patient_count"].sum().reset_index()
                    completed = df_coh[df_coh["stages_completed"] >= 4].groupby(
                        "payer_type")["patient_count"].sum().reset_index()
                    completed.columns = ["payer_type", "completed"]
                    merged = payer_summary.merge(completed, on="payer_type", how="left").fillna(0)
                    merged["completion_pct"] = (merged["completed"] / merged["patient_count"] * 100).round(1)
                    _pc(bar_chart(merged, x="payer_type", y="completion_pct",
                                  color_map={"Cash": CORAL, "NHIF / SHA": TEAL,
                                             "Insurance / Corporate": AFYA_BLUE},
                                  y_label="ANC4 Completion %", y_format="pct",
                                  height=280, show_text=True))
        except Exception as e:
            st.warning(f"ANC patient cohort: {e}")

        _gap(12)

        # ── C4: High-Risk Pregnancy ────────────────────────────────────────────
        _sh("High-Risk Pregnancies — Detection & Profile", mt=8)
        _note("Three risk detection methods: clinical ICD10 diagnosis flags, maternal age (adolescent <18 or AME 35+), and repeated elevated BP in vitals. A patient may have multiple risk factors.")
        try:
            df_hr = Q.load_high_risk_pregnancy_profile(filters, run_query)
            if not df_hr.empty:
                total_hr = int(df_hr["patient_count"].sum() or 0)
                _kpi("High-Risk ANC Patients Detected", _n(total_hr),
                     "across all risk categories", CORAL)
                _gap(8)
                _pc(hbar_chart(df_hr.head(12), x="patient_count", y="risk_type",
                               x_label="Patients", color=CORAL, height=320, show_text=True))
        except Exception as e:
            st.warning(f"High-risk pregnancy profile: {e}")

        _gap(8)
        _note("Patient-level high-risk list — sorted by clinical severity, then age risk, then days since last ANC visit.")
        try:
            df_hrp = Q.load_high_risk_pregnancy_patients(filters, run_query)
            if not df_hrp.empty:
                show_cols = ["patient", "age_group", "payer_type", "anc_visits",
                             "days_since_last_anc", "risk_flags"]
                col_labels = {"patient": "Patient", "age_group": "Age",
                              "payer_type": "Payer", "anc_visits": "ANC Visits",
                              "days_since_last_anc": "Days Since ANC", "risk_flags": "Risk Flags"}
                _pc(table_fig(df_hrp[show_cols].head(30), col_labels=col_labels,
                              height=min(600, len(df_hrp.head(30)) * 28 + 60)))
        except Exception as e:
            st.warning(f"High-risk patient list: {e}")

        _gap(12)

        # ── C5: Illnesses in Pregnant Women ───────────────────────────────────
        _sh("Comorbidities in Pregnant Women", mt=8)
        _note("Other conditions diagnosed in patients who also had ANC visits. Identifies illnesses co-occurring with pregnancy that may compound risk.")
        try:
            df_com = Q.load_pregnancy_comorbidities(filters, run_query)
            if not df_com.empty:
                _pc(hbar_chart(df_com.head(15), x="patient_count", y="condition_group",
                               x_label="Pregnant Patients Affected", color=ORANGE,
                               height=min(400, len(df_com.head(15)) * 26 + 60), show_text=True))
        except Exception as e:
            st.warning(f"Pregnancy comorbidities: {e}")

        _gap(12)

        # ── C6: PNC Profile ────────────────────────────────────────────────────
        _sh("Postnatal Care (PNC) — Who Is Coming Back?", mt=8)
        _note("PNC visits by payer and age group. Low PNC volume relative to deliveries signals a retention gap after birth.")
        try:
            df_pnc = Q.load_pnc_profile(filters, run_query)
            if not df_pnc.empty:
                total_pnc = int(df_pnc["visit_count"].sum() or 0)
                total_pnc_pts = int(df_pnc["patient_count"].sum() or 0)
                c1, c2, c3 = st.columns(3)
                with c1: _kpi("PNC Visits", _n(total_pnc), color=TEAL)
                with c2: _kpi("PNC Patients", _n(total_pnc_pts), color=TEAL)
                _gap(8)
                c1, c2 = st.columns(2)
                with c1:
                    pnc_age = df_pnc.groupby("age_group")["visit_count"].sum().reset_index()
                    _pc(bar_chart(pnc_age, x="age_group", y="visit_count",
                                  color=TEAL, y_label="PNC Visits", height=240, show_text=True))
                with c2:
                    pnc_pay = df_pnc.groupby("payer_type")["visit_count"].sum().reset_index()
                    _pc(donut(labels=pnc_pay["payer_type"].tolist(),
                              values=pnc_pay["visit_count"].tolist(),
                              color_map={"Cash": CORAL, "NHIF / SHA": TEAL,
                                         "Insurance / Corporate": AFYA_BLUE},
                              height=240))
        except Exception as e:
            st.warning(f"PNC profile: {e}")

        _gap(12)

        # ── C7: Deliveries by Age ──────────────────────────────────────────────
        _sh("Deliveries by Maternal Age Group", mt=8)
        try:
            df_del = Q.load_deliveries_by_age(filters, run_query)
            if not df_del.empty:
                _pc(bar_chart(df_del, x="maternal_age_group", y="delivery_count",
                              color=PURPLE, y_label="Deliveries", height=260, show_text=True))
        except Exception as e:
            st.warning(f"Deliveries by age: {e}")

        _gap(12)

        # ── C8: Under-5 Profile ────────────────────────────────────────────────
        _sh("Under-5 Children — What Brought Them?", mt=8)
        _note("Visit category breakdown for children under 5, by age bucket. Shows whether admissions cluster in a specific age group or illness category.")
        try:
            df_u5 = Q.load_under5_profile(filters, run_query)
            if not df_u5.empty:
                total_u5 = int(df_u5["visit_count"].sum() or 0)
                total_admitted = int(df_u5["admitted_count"].sum() or 0)
                c1, c2, c3 = st.columns(3)
                with c1: _kpi("Under-5 Visits", _n(total_u5), color=AFYA_BLUE)
                with c2: _kpi("Admissions", _n(total_admitted),
                              f"{total_admitted / total_u5 * 100:.1f}% admission rate" if total_u5 else "",
                              CORAL if total_admitted / max(total_u5, 1) > 0.1 else TEAL)
                _gap(8)
                cat_summary = df_u5.groupby("visit_category").agg(
                    visit_count=("visit_count", "sum"),
                    admitted_count=("admitted_count", "sum"),
                ).reset_index().sort_values("visit_count", ascending=False)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Visits by Reason**")
                    _pc(bar_chart(cat_summary, x="visit_category", y="visit_count",
                                  color=AFYA_BLUE, y_label="Visits", height=300, show_text=True))
                with c2:
                    st.markdown("**Admission Rate by Category**")
                    cat_summary["admission_rate_pct"] = (
                        cat_summary["admitted_count"] / cat_summary["visit_count"].replace(0, 1) * 100
                    ).round(1)
                    _pc(hbar_chart(cat_summary.sort_values("admission_rate_pct"),
                                   x="admission_rate_pct", y="visit_category",
                                   x_label="Admission Rate %", y_format="pct",
                                   color=CORAL, height=300, show_text=True))
                _gap(8)
                age_summary = df_u5.groupby("age_bucket")["visit_count"].sum().reset_index()
                st.markdown("**Visits by Age Bucket**")
                _pc(bar_chart(age_summary, x="age_bucket", y="visit_count",
                              color=TEAL, y_label="Visits", height=220, show_text=True))
        except Exception as e:
            st.warning(f"Under-5 profile: {e}")

        _gap(12)

        # ── C9: Adolescent Reproductive Health ────────────────────────────────
        _sh("Adolescent Reproductive Health (Age 10–19)", mt=8)
        _note("What brings adolescents to the hospital in reproductive health terms: family planning, ANC/pregnancy, abortion, PID, STI, PNC, and delivery. Separated by sex and payer.")
        try:
            df_adol = Q.load_adolescent_rh_profile(filters, run_query)
            if not df_adol.empty:
                cat_tot = df_adol.groupby("rh_category").agg(
                    visit_count=("visit_count", "sum"),
                    patient_count=("patient_count", "sum"),
                ).reset_index().sort_values("patient_count", ascending=False)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Adolescent RH — Visit Category**")
                    _pc(hbar_chart(cat_tot, x="patient_count", y="rh_category",
                                   x_label="Patients", color=PURPLE,
                                   height=min(380, len(cat_tot) * 34 + 60), show_text=True))
                with c2:
                    st.markdown("**By Payer Type**")
                    pay_tot = df_adol.groupby("payer_type")["patient_count"].sum().reset_index()
                    _pc(donut(labels=pay_tot["payer_type"].tolist(),
                              values=pay_tot["patient_count"].tolist(),
                              color_map={"Cash": CORAL, "NHIF / SHA": TEAL,
                                         "Insurance / Corporate": AFYA_BLUE},
                              height=280))
        except Exception as e:
            st.warning(f"Adolescent RH: {e}")

        _gap(12)

        # ── C10: RMNCH Revenue — Segment Share & Trend ────────────────────────
        _sh("RMNCH Revenue — Segment Share & Monthly Trend", mt=8)
        _note("Revenue from Maternal, Paediatric, and Adolescent RH segments. The RMNCH Care & Capital Matrix below is built from actual query data.")
        try:
            df_rev = Q.load_rmnch_revenue_trend(filters, run_query)
            if not df_rev.empty:
                seg_summary = df_rev.groupby("rmnch_segment").agg(
                    visit_count=("visit_count", "sum"),
                    patient_count=("patient_count", "sum"),
                    revenue=("revenue", "sum"),
                ).reset_index()
                total_rev = float(seg_summary["revenue"].sum() or 1)
                seg_summary["rev_share_pct"] = (seg_summary["revenue"] / total_rev * 100).round(1)

                # KPIs
                c1, c2, c3, c4 = st.columns(4)
                with c1: _kpi("Total RMNCH Revenue", _k(total_rev), color=PURPLE)
                for col, seg, color in [(c2, "Maternal", AFYA_BLUE),
                                        (c3, "Paediatric (<12y)", TEAL),
                                        (c4, "Adolescent RH", ORANGE)]:
                    row_s = seg_summary[seg_summary["rmnch_segment"] == seg]
                    rev_s = float(row_s["revenue"].sum()) if not row_s.empty else 0
                    pct_s = float(row_s["rev_share_pct"].sum()) if not row_s.empty else 0
                    with col:
                        _kpi(seg, _k(rev_s), f"{pct_s:.1f}% of RMNCH revenue", color)

                _gap(12)

                # RMNCH Care & Capital Matrix — from live data
                st.markdown("**RMNCH Care & Capital Matrix**")
                matrix_rows = []
                for seg in ["Maternal", "Paediatric (<12y)", "Adolescent RH"]:
                    row_s = seg_summary[seg_summary["rmnch_segment"] == seg]
                    if row_s.empty:
                        continue
                    r = row_s.iloc[0]
                    matrix_rows.append({
                        "Segment": seg,
                        "Patients": _n(r["patient_count"]),
                        "Visits": _n(r["visit_count"]),
                        "Revenue": _k(r["revenue"]),
                        "% Share": f"{r['rev_share_pct']:.1f}%",
                    })
                if matrix_rows:
                    mdf = pd.DataFrame(matrix_rows)
                    _pc(table_fig(mdf, height=min(240, len(mdf) * 40 + 60)))

                _gap(12)

                # Monthly revenue trend
                st.markdown("**Monthly Revenue Trend by RMNCH Segment**")
                if "visit_month" in df_rev.columns:
                    df_rev["visit_month"] = pd.to_datetime(df_rev["visit_month"])
                    monthly = df_rev.sort_values("visit_month")
                    seg_colors = {"Maternal": AFYA_BLUE, "Paediatric (<12y)": TEAL,
                                  "Adolescent RH": ORANGE}
                    fig_trend = go.Figure()
                    for seg, grp in monthly.groupby("rmnch_segment"):
                        fig_trend.add_trace(go.Scatter(
                            x=grp["visit_month"], y=grp["revenue"],
                            mode="lines+markers", name=seg,
                            line=dict(color=seg_colors.get(seg, PURPLE), width=2),
                            hovertemplate=f"<b>{seg}</b><br>%{{x|%b %Y}}<br>KES %{{y:,.0f}}<extra></extra>",
                        ))
                    fig_trend.update_layout(
                        height=300, margin=dict(l=0, r=0, t=30, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        yaxis_title="Revenue (KES)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )
                    _pc(fig_trend)
        except Exception as e:
            st.warning(f"RMNCH revenue: {e}")

    # ── COMMUNICABLE & HIV TAB ────────────────────────────────────────────────
    with st_d:
        _sh("D — Communicable Disease & HIV")

        # ── D0: Disease KPI Snapshot ───────────────────────────────────────────
        _sh("Disease Case Counts — TB, Malaria, URTI, Typhoid, Enteric, HIV", mt=8)
        _note("Total patients and visits per disease in the selected period, with how many required inpatient admission.")
        try:
            df_kpi = Q.load_disease_kpi_snapshot(filters, run_query)
            if not df_kpi.empty:
                for col_set in [df_kpi.iloc[i:i+3] for i in range(0, len(df_kpi), 3)]:
                    cols = st.columns(3)
                    for col, (_, row_k) in zip(cols, col_set.iterrows()):
                        adm_pct = float(row_k.get("admission_rate_pct") or 0)
                        with col:
                            _kpi(
                                str(row_k["disease_label"]),
                                f"{_n(row_k['patient_count'])} patients",
                                f"{_n(row_k['visit_count'])} visits · {adm_pct:.0f}% admitted",
                                CORAL if adm_pct > 15 else AFYA_BLUE,
                            )
                    _gap(4)
        except Exception as e:
            st.warning(f"Disease KPIs: {e}")

        _gap(12)

        # ── D1: Who Does Each Disease Affect? ─────────────────────────────────
        _sh("Who Does Each Disease Affect? — Age & Sex Breakdown", mt=8)
        _note("Each bar shows patients by age group. The two bars side-by-side compare Female (F) vs Male (M). Paediatric = Under 5 + 5-17 combined. Use this to identify whether a disease disproportionately hits a specific demographic.")
        try:
            df_dem = Q.load_disease_demographics(filters, run_query)
            if not df_dem.empty:
                diseases = df_dem["disease_label"].unique().tolist()
                n_cols = min(3, len(diseases))
                rows_d = [diseases[i:i+n_cols] for i in range(0, len(diseases), n_cols)]
                for row_diseases in rows_d:
                    cols_d = st.columns(n_cols)
                    for col_d, dis in zip(cols_d, row_diseases):
                        sub_d = df_dem[df_dem["disease_label"] == dis].copy()
                        with col_d:
                            st.markdown(f"**{dis}**")
                            age_sex = sub_d.groupby(["age_group", "sex"])["patient_count"].sum().reset_index()
                            fig_ds = go.Figure()
                            sex_colors = {"F": PURPLE, "FEMALE": PURPLE,
                                          "M": AFYA_BLUE, "MALE": AFYA_BLUE,
                                          "Unknown": GRAY}
                            for sx in age_sex["sex"].unique():
                                sub_sx = age_sex[age_sex["sex"] == sx]
                                fig_ds.add_trace(go.Bar(
                                    name=sx,
                                    x=sub_sx["age_group"],
                                    y=sub_sx["patient_count"],
                                    marker_color=sex_colors.get(sx, GRAY),
                                    showlegend=(dis == diseases[0]),
                                ))
                            fig_ds.update_layout(
                                barmode="group", height=200,
                                margin=dict(l=0, r=0, t=10, b=30),
                                plot_bgcolor="white", paper_bgcolor="white",
                                yaxis_title="Patients",
                                xaxis=dict(tickangle=-20),
                                legend=dict(orientation="h", y=-0.35),
                            )
                            _pc(fig_ds)
        except Exception as e:
            st.warning(f"Disease demographics: {e}")

        _gap(12)

        # ── D2: Monthly Trend — Spike vs Sustained ─────────────────────────────
        _sh("Monthly Case Trend — Spike or Sustained? (Typhoid Focus)", mt=8)
        _note("Use this to determine whether the high typhoid inpatient count was a one-time outbreak spike or a persistent pattern. A single tall bar indicates a spike; multiple high months indicate endemic spread.")
        try:
            df_trend = Q.load_disease_monthly_trend(filters, run_query)
            if not df_trend.empty:
                df_trend["visit_month"] = pd.to_datetime(df_trend["visit_month"])
                selected_diseases = ["Typhoid", "Malaria", "URTI", "TB", "Enteric / GI"]
                disease_colors = {
                    "Typhoid": "#D97706", "Malaria": TEAL, "URTI": AFYA_BLUE,
                    "TB": CORAL, "Enteric / GI": PURPLE, "HIV": GRAY,
                }
                c1, c2 = st.columns([2, 1])
                with c1:
                    fig_tr = go.Figure()
                    for dis in selected_diseases:
                        sub_tr = df_trend[df_trend["disease_label"] == dis].sort_values("visit_month")
                        if not sub_tr.empty:
                            fig_tr.add_trace(go.Scatter(
                                x=sub_tr["visit_month"], y=sub_tr["visit_count"],
                                name=dis, mode="lines+markers",
                                line=dict(color=disease_colors.get(dis, GRAY), width=2),
                                hovertemplate=f"<b>{dis}</b> — %{{x|%b %Y}}: %{{y:,}} visits<extra></extra>",
                            ))
                    fig_tr.update_layout(
                        height=320, margin=dict(l=0, r=0, t=20, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        yaxis_title="Visits",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )
                    _pc(fig_tr)
                with c2:
                    typhoid_monthly = df_trend[df_trend["disease_label"] == "Typhoid"].copy()
                    if not typhoid_monthly.empty:
                        mean_v = typhoid_monthly["visit_count"].mean()
                        max_v = typhoid_monthly["visit_count"].max()
                        spike_months = typhoid_monthly[
                            typhoid_monthly["visit_count"] > mean_v * 1.5
                        ]
                        _kpi("Typhoid Monthly Avg", f"{mean_v:.0f} visits", color=ORANGE)
                        _gap(8)
                        _kpi("Peak Month", f"{max_v:.0f} visits",
                             f"{len(spike_months)} month(s) above 1.5× average",
                             CORAL if len(spike_months) <= 2 else ORANGE)
                        _gap(8)
                        verdict = "Spike (1–2 high months)" if len(spike_months) <= 2 else "Sustained / Endemic"
                        st.markdown(
                            f'<div style="background:#FFF8E7;border-left:4px solid #E69A00;'
                            f'padding:10px;border-radius:6px;font-size:12px">'
                            f'<b>Pattern:</b> {verdict}</div>',
                            unsafe_allow_html=True,
                        )
        except Exception as e:
            st.warning(f"Monthly trend: {e}")

        _gap(12)

        # ── D3: TB-HIV Co-infection ────────────────────────────────────────────
        _sh("TB & HIV — Co-infection Analysis", mt=8)
        _note("The female/male counts below are HIV patients (B20 diagnosis) by sex — they show who has HIV, not who has TB. The co-infection section below shows specifically which TB patients also have an HIV diagnosis, and whether HIV was tested or detected via clinical markers.")
        try:
            df_coinfect = Q.load_tb_hiv_coinfection(filters, run_query)
            if not df_coinfect.empty:
                row_c = df_coinfect.iloc[0]
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1: _kpi("TB Patients", _n(row_c.get("tb_patients")), color=CORAL)
                with c2: _kpi("HIV Patients", _n(row_c.get("hiv_patients")), color=ORANGE)
                with c3: _kpi("TB+HIV Co-infected", _n(row_c.get("tb_hiv_coinfected")),
                               f"{float(row_c.get('coinfection_rate_pct') or 0):.1f}% of TB patients",
                               CORAL if float(row_c.get("coinfection_rate_pct") or 0) > 10 else TEAL)
                with c4: _kpi("HIV Test Coverage",
                               _p(row_c.get("hiv_test_coverage_pct")),
                               "of TB patients had HIV test done",
                               TEAL if float(row_c.get("hiv_test_coverage_pct") or 0) >= 80 else CORAL)
                with c5: _kpi("TB with Fever Signal", _n(row_c.get("tb_with_fever")),
                               "elevated temp (>37.5°C) in vitals", ORANGE)
        except Exception as e:
            st.warning(f"TB-HIV co-infection: {e}")

        _gap(8)
        try:
            df_hiv = Q.load_hiv_profile(filters, run_query)
            if not df_hiv.empty:
                row_h = df_hiv.iloc[0]
                _note(f"HIV patient profile: {_n(row_h.get('hiv_patients'))} total — "
                      f"{_n(row_h.get('female'))} Female · {_n(row_h.get('male'))} Male · "
                      f"{_n(row_h.get('paediatric'))} Paediatric")
        except Exception as e:
            st.warning(f"HIV profile: {e}")

        _gap(12)

        # ── D4: Malaria — Who It Affects + Lab Accuracy ────────────────────────
        _sh("Malaria — Who Is Most Affected & Test Accuracy", mt=8)
        _note("Left: malaria patients by age group and sex. Right: how often is malaria tested vs clinically diagnosed without a test (false positive risk). Diagnosis without a test = clinical-only.")
        try:
            df_dem2 = Q.load_disease_demographics(filters, run_query)
            df_mal_lab = Q.load_malaria_lab_accuracy(filters, run_query)
            c1, c2 = st.columns(2)
            with c1:
                if not df_dem2.empty:
                    mal_dem = df_dem2[df_dem2["disease_label"] == "Malaria"].copy()
                    if not mal_dem.empty:
                        st.markdown("**Malaria Patients by Age Group**")
                        age_mal = mal_dem.groupby("age_group")["patient_count"].sum().reset_index().sort_values("patient_count", ascending=False)
                        _pc(bar_chart(age_mal, x="age_group", y="patient_count",
                                      color=TEAL, y_label="Patients", height=260, show_text=True))
            with c2:
                if not df_mal_lab.empty:
                    row_ml = df_mal_lab.iloc[0]
                    tested = int(row_ml.get("visits_with_test") or 0)
                    not_tested = int(row_ml.get("no_test_done") or 0)
                    resulted = int(row_ml.get("test_resulted") or 0)
                    ordered_only = int(row_ml.get("test_ordered_only") or 0)
                    st.markdown("**Malaria Test Coverage**")
                    _kpi("Test Rate", _p(row_ml.get("test_rate_pct")),
                         f"{tested:,} tested · {not_tested:,} no test (clinical-only dx)",
                         TEAL if float(row_ml.get("test_rate_pct") or 0) >= 70 else CORAL)
                    _gap(8)
                    if tested > 0:
                        _kpi("Result Rate", _p(row_ml.get("result_rate_pct")),
                             f"{resulted:,} resulted · {ordered_only:,} ordered but not resulted",
                             TEAL if float(row_ml.get("result_rate_pct") or 0) >= 80 else ORANGE)
                        _gap(8)
                        _pc(donut(
                            labels=["Test Resulted", "Test Ordered Only", "No Test Done"],
                            values=[resulted, ordered_only, not_tested],
                            color_map={"Test Resulted": TEAL,
                                       "Test Ordered Only": ORANGE,
                                       "No Test Done": CORAL},
                            height=200,
                        ))
        except Exception as e:
            st.warning(f"Malaria: {e}")

        _gap(12)

        # ── D5: Admissions per Disease ─────────────────────────────────────────
        _sh("Which Diseases Lead to Admission?", mt=8)
        _note("Admission rate per disease. TB and Typhoid typically have higher admission rates than URTI. High admission rates in diseases that are normally outpatient indicate severity or delayed presentation.")
        try:
            df_kpi2 = Q.load_disease_kpi_snapshot(filters, run_query)
            if not df_kpi2.empty:
                df_kpi2["admission_rate_pct"] = pd.to_numeric(df_kpi2["admission_rate_pct"], errors="coerce")
                c1, c2 = st.columns(2)
                with c1:
                    _pc(hbar_chart(
                        df_kpi2.sort_values("admission_rate_pct"),
                        x="admission_rate_pct", y="disease_label",
                        x_label="Admission Rate %", y_format="pct",
                        color=CORAL, height=280, show_text=True,
                    ))
                with c2:
                    _pc(bar_chart(
                        df_kpi2.sort_values("admitted_count", ascending=False),
                        x="disease_label", y="admitted_count",
                        color=ORANGE, y_label="Admitted Patients", height=280, show_text=True,
                    ))
        except Exception as e:
            st.warning(f"Disease admissions: {e}")

        _gap(12)

        # ── D6: Comorbidities per Communicable Disease ─────────────────────────
        _sh("What Other Conditions Co-occur with Communicable Diseases?", mt=8)
        _note("Top 5 comorbidities found in patients who also have each communicable diagnosis. Helps identify compound risk and treatment complexity.")
        try:
            df_comorb = Q.load_communicable_comorbidities(filters, run_query)
            if not df_comorb.empty:
                diseases_c = df_comorb["disease_label"].unique().tolist()
                cols_c = st.columns(min(3, len(diseases_c)))
                for col_c, dis_c in zip(cols_c, diseases_c[:3]):
                    sub_c = df_comorb[df_comorb["disease_label"] == dis_c]
                    with col_c:
                        st.markdown(f"**{dis_c} — Top Comorbidities**")
                        _pc(hbar_chart(sub_c, x="patient_count", y="comorbidity",
                                       x_label="Patients", color=PURPLE,
                                       height=min(240, len(sub_c) * 36 + 60), show_text=True))
                if len(diseases_c) > 3:
                    cols_c2 = st.columns(min(3, len(diseases_c) - 3))
                    for col_c2, dis_c2 in zip(cols_c2, diseases_c[3:6]):
                        sub_c2 = df_comorb[df_comorb["disease_label"] == dis_c2]
                        with col_c2:
                            st.markdown(f"**{dis_c2} — Top Comorbidities**")
                            _pc(hbar_chart(sub_c2, x="patient_count", y="comorbidity",
                                           x_label="Patients", color=PURPLE,
                                           height=min(240, len(sub_c2) * 36 + 60), show_text=True))
        except Exception as e:
            st.warning(f"Communicable comorbidities: {e}")

        _gap(12)

        # ── D7: Surge Pattern Indicator ────────────────────────────────────────
        _sh("Seasonal Surge Pattern — Malaria, URTI & Typhoid", mt=8)
        _note("Each disease keeps its own color. A red triangle marker (▲) appears on top of a bar when that disease exceeded 1.5× its period average that month — the label shows which disease surged.")
        try:
            df_surge = Q.load_disease_monthly_trend(filters, run_query)
            if not df_surge.empty:
                df_surge["visit_month"] = pd.to_datetime(df_surge["visit_month"])
                surge_diseases = ["Malaria", "URTI", "Typhoid"]
                surge_df = df_surge[df_surge["disease_label"].isin(surge_diseases)].copy()
                if not surge_df.empty:
                    means = surge_df.groupby("disease_label")["visit_count"].mean()
                    surge_df["is_surge"] = surge_df.apply(
                        lambda r: r["visit_count"] > means.get(r["disease_label"], 0) * 1.5, axis=1
                    )
                    surge_clrs = {"Malaria": TEAL, "URTI": AFYA_BLUE, "Typhoid": ORANGE}
                    fig_sg = go.Figure()

                    # bars — each disease keeps its own colour always
                    for dis_s in surge_diseases:
                        sub_s = surge_df[surge_df["disease_label"] == dis_s].sort_values("visit_month")
                        if sub_s.empty:
                            continue
                        fig_sg.add_trace(go.Bar(
                            name=dis_s,
                            x=sub_s["visit_month"],
                            y=sub_s["visit_count"],
                            marker_color=surge_clrs.get(dis_s, GRAY),
                            hovertemplate=f"<b>{dis_s}</b> — %{{x|%b %Y}}: %{{y:,}} visits<extra></extra>",
                        ))

                    # surge markers — red ▲ per disease, offset so they don't overlap
                    offsets = {"Malaria": -0.27, "URTI": 0, "Typhoid": 0.27}
                    first_surge = True
                    for dis_s in surge_diseases:
                        sub_s = surge_df[
                            (surge_df["disease_label"] == dis_s) & surge_df["is_surge"]
                        ].sort_values("visit_month")
                        if sub_s.empty:
                            continue
                        fig_sg.add_trace(go.Scatter(
                            name="Surge flag",
                            x=sub_s["visit_month"],
                            y=sub_s["visit_count"],
                            mode="markers+text",
                            marker=dict(symbol="triangle-up", size=12,
                                        color="#DC2626", line=dict(color="white", width=1)),
                            text=[dis_s] * len(sub_s),
                            textposition="top center",
                            textfont=dict(size=9, color="#DC2626"),
                            showlegend=first_surge,
                            legendgroup="surge",
                            hovertemplate=f"<b>SURGE: {dis_s}</b><br>%{{x|%b %Y}} — %{{y:,}} visits<br>(>1.5× average)<extra></extra>",
                        ))
                        first_surge = False

                    fig_sg.update_layout(
                        barmode="group", height=320,
                        margin=dict(l=0, r=0, t=30, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        yaxis_title="Visits", xaxis_title="",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )
                    _pc(fig_sg)

                    # surge summary table
                    surge_rows = surge_df[surge_df["is_surge"]].copy()
                    if not surge_rows.empty:
                        surge_rows["Month"] = surge_rows["visit_month"].dt.strftime("%b %Y")
                        surge_rows["vs Average"] = surge_rows.apply(
                            lambda r: f"{r['visit_count'] / max(means.get(r['disease_label'], 1), 1):.1f}×", axis=1
                        )
                        surge_rows = surge_rows.rename(columns={
                            "disease_label": "Disease", "visit_count": "Visits"
                        })[["Disease", "Month", "Visits", "vs Average"]].sort_values("Month")
                        st.markdown("**Surge months (>1.5× average):**")
                        _pc(table_fig(surge_rows, height=min(240, len(surge_rows) * 32 + 50)))
        except Exception as e:
            st.warning(f"Surge pattern: {e}")

        _gap(12)

        # ── D8: Unified Pipeline Matrix (from live data) ───────────────────────
        _sh("Unified Acute & Communicable Pipeline Matrix", mt=8)
        _note("Built from actual Snowflake data: lab confirmation rate, inpatient admission rate, primary demographic, top comorbidity, and primary payer per disease.")
        try:
            df_cpm = Q.load_communicable_pipeline_matrix(filters, run_query)
            if not df_cpm.empty:
                for _c in ("quarterly_visits", "lab_confirmation_pct", "inpatient_admission_pct"):
                    if _c in df_cpm.columns:
                        df_cpm[_c] = pd.to_numeric(df_cpm[_c], errors="coerce")

                rows_html = ""
                for _, r in df_cpm.iterrows():
                    ip_pct = float(r.get("inpatient_admission_pct") or 0)
                    ip_color = "#DC2626" if ip_pct > 15 else "#16A34A" if ip_pct < 5 else "#D97706"
                    rows_html += (
                        f'<tr>'
                        f'<td style="padding:7px 8px;font-weight:600;font-size:12px">{r.get("disease_group","")}</td>'
                        f'<td style="padding:7px 8px;font-size:11px;text-align:right">{int(r.get("quarterly_visits", 0) or 0):,}</td>'
                        f'<td style="padding:7px 8px;font-size:11px">{r.get("primary_age_sex","")}</td>'
                        f'<td style="padding:7px 8px;font-size:11px;text-align:right">'
                        f'{float(r.get("lab_confirmation_pct", 0) or 0):.0f}%</td>'
                        f'<td style="padding:7px 8px;font-size:11px;text-align:right;color:{ip_color};font-weight:700">'
                        f'{ip_pct:.0f}%</td>'
                        f'<td style="padding:7px 8px;font-size:11px">{r.get("primary_comorbidity","—")}</td>'
                        f'<td style="padding:7px 8px;font-size:11px">{r.get("primary_payer","")}</td>'
                        f'</tr>'
                    )
                st.markdown(
                    '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-family:sans-serif">'
                    '<thead><tr style="background:#E05C2D;color:white">'
                    '<th style="padding:8px;text-align:left">Disease</th>'
                    '<th style="padding:8px;text-align:right">90d Visits</th>'
                    '<th style="padding:8px;text-align:left">Primary Demographic</th>'
                    '<th style="padding:8px;text-align:right">Lab Confirm %</th>'
                    '<th style="padding:8px;text-align:right">IP Admission %</th>'
                    '<th style="padding:8px;text-align:left">Top Comorbidity</th>'
                    '<th style="padding:8px;text-align:left">Primary Payer</th>'
                    '</tr></thead><tbody>' + rows_html + '</tbody></table></div>',
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.warning(f"Pipeline matrix: {e}")

    # ── MENTAL HEALTH TAB ─────────────────────────────────────────────────────
    with st_e:
        _sh("E — Mental Health & Psychiatric Analytics")

        try:
            df_mhk = Q.load_mh_kpis(filters, run_query)
            if not df_mhk.empty:
                row = df_mhk.iloc[0]
                c1, c2, c3 = st.columns(3)
                with c1: _kpi("MH Patients", _n(row.get("total_mh_patients")))
                with c2: _kpi("Inpatient Share", _p(row.get("inpatient_share_pct")),
                               "admitted for MH", ORANGE)
                with c3: _kpi("MH Visits", _n(row.get("total_mh_visits")))
        except Exception as e:
            st.warning(f"E1: {e}")

        _gap(12)

        # Grouped Demographic Illness Matrix
        _sh("Mental Health — Diagnostic Breakdown by Age & Condition", mt=8)
        _note("Each age group is broken into specific diagnostic sub-segments: Depression/Anxiety vs Substance Abuse vs Psychotic Disorders vs Dementia.")
        try:
            df_mhd = Q.load_mh_diagnostic_breakdown(filters, run_query)
            if not df_mhd.empty:
                df_mhd["patient_count"] = pd.to_numeric(df_mhd["patient_count"], errors="coerce")
                fig_mhd = go.Figure()
                mh_cat_colors = {
                    "Depression & Anxiety": AFYA_BLUE,
                    "Substance & Alcohol": CORAL,
                    "Psychotic Disorders": PURPLE,
                    "Dementia / Organic Brain": ORANGE,
                    "Other Mental Health": MUTED,
                }
                for cat, color in mh_cat_colors.items():
                    sub = df_mhd[df_mhd["mh_category"] == cat]
                    if not sub.empty:
                        age_grp = sub.groupby("age_group")["patient_count"].sum().reset_index()
                        fig_mhd.add_trace(go.Bar(
                            name=cat, x=age_grp["age_group"], y=age_grp["patient_count"],
                            marker_color=color,
                            hovertemplate=f"<b>%{{x}}</b><br>{cat}: %{{y:,}}<extra></extra>",
                        ))
                fig_mhd.update_layout(
                    barmode="stack", height=300,
                    margin=dict(l=0, r=20, t=20, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(title="Patients", rangemode="tozero"),
                    legend=dict(orientation="h", y=-0.25, xanchor="center", x=0.5),
                )
                _pc(fig_mhd)
            else:
                df_mhas = Q.load_mh_by_age_sex(filters, run_query)
                if not df_mhas.empty:
                    pivot = df_mhas.pivot_table(index="age_group", columns="sex",
                                                values="patient_count", aggfunc="sum", fill_value=0)
                    _pc(bar_chart(pivot.reset_index(), x="age_group", y=list(pivot.columns),
                                  color_map={"F": "#7b5ea7", "FEMALE": "#7b5ea7",
                                             "M": TEAL, "MALE": TEAL},
                                  y_label="Patients", height=280))
        except Exception as e:
            st.warning(f"E2: {e}")

        _gap(12)

        # Comorbidity profile: standalone vs comorbid
        _sh("Standalone vs Comorbid Mental Health — Who Has a Hidden Primary Condition?", mt=8)
        _note("A patient with Depression secondary to Hypertension is a chronic patient who needs integrated care, not just psychiatric medication.")
        try:
            df_mhc = Q.load_mh_comorbidity_profile(filters, run_query)
            if not df_mhc.empty:
                for _c in ("standalone_patients", "comorbid_patients", "standalone_pct"):
                    df_mhc[_c] = pd.to_numeric(df_mhc[_c], errors="coerce")

                c1, c2 = st.columns([1.2, 0.8])
                with c1:
                    fig_mhc = go.Figure()
                    fig_mhc.add_trace(go.Bar(
                        name="Standalone", x=df_mhc["mh_category"],
                        y=df_mhc["standalone_patients"], marker_color=TEAL,
                        hovertemplate="<b>%{x}</b><br>Standalone: %{y:,}<extra></extra>",
                    ))
                    fig_mhc.add_trace(go.Bar(
                        name="Comorbid (has primary NCD/other)", x=df_mhc["mh_category"],
                        y=df_mhc["comorbid_patients"], marker_color=CORAL,
                        hovertemplate="<b>%{x}</b><br>Comorbid: %{y:,}<extra></extra>",
                    ))
                    fig_mhc.update_layout(
                        barmode="group", height=280,
                        margin=dict(l=0, r=20, t=20, b=60),
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(tickangle=-20),
                        yaxis=dict(title="Patients", rangemode="tozero"),
                        legend=dict(orientation="h", y=1.05, xanchor="right", x=1),
                    )
                    _pc(fig_mhc)
                with c2:
                    _pc(table_fig(
                        df_mhc[["mh_category", "standalone_patients", "comorbid_patients",
                                "standalone_pct", "top_comorbidity"]],
                        col_labels={"mh_category": "Condition", "standalone_patients": "Standalone",
                                    "comorbid_patients": "Comorbid",
                                    "standalone_pct": "Standalone %",
                                    "top_comorbidity": "Top Co-Condition"},
                        fmt={"standalone_pct": "pct"},
                        height=280,
                    ))
        except Exception as e:
            st.warning(f"E3: {e}")

        _gap(12)

        # Monthly trend by MH category
        _sh("Mental Health Visit Growth — Monthly Trend by Category", mt=8)
        try:
            df_mht = Q.load_mh_monthly_trend(filters, run_query)
            if not df_mht.empty:
                df_mht["visit_count"] = pd.to_numeric(df_mht["visit_count"], errors="coerce")
                pivot_mht = df_mht.pivot_table(
                    index="visit_month", columns="mh_category",
                    values="visit_count", aggfunc="sum", fill_value=0
                ).reset_index()
                fig_mht = go.Figure()
                for col in [c for c in pivot_mht.columns if c != "visit_month"]:
                    fig_mht.add_trace(go.Scatter(
                        x=pivot_mht["visit_month"], y=pivot_mht[col],
                        name=col, mode="lines+markers",
                        hovertemplate=f"<b>{col}</b> — %{{x|%b %Y}}: %{{y:,}}<extra></extra>",
                    ))
                fig_mht.update_layout(
                    height=280, margin=dict(l=0, r=20, t=20, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(title="Visits", rangemode="tozero"),
                    legend=dict(orientation="h", y=-0.25, xanchor="center", x=0.5),
                )
                _pc(fig_mht)
        except Exception as e:
            st.warning(f"E4: {e}")

        _gap(12)

        # Mental Health Care & Synergy Matrix — built from live query data
        _sh("Mental Health Care & Synergy Matrix", mt=8)
        _note("Built from actual data: comorbidity split from load_mh_comorbidity_profile, trend direction from load_mh_monthly_trend.")
        try:
            df_mhc2 = Q.load_mh_comorbidity_profile(filters, run_query)
            df_mht2 = Q.load_mh_monthly_trend(filters, run_query)
            if not df_mhc2.empty:
                # compute trend direction per category from monthly data
                trend_map = {}
                if not df_mht2.empty:
                    df_mht2["visit_month"] = pd.to_datetime(df_mht2["visit_month"])
                    for cat_t, grp_t in df_mht2.groupby("mh_category"):
                        grp_t = grp_t.sort_values("visit_month")
                        if len(grp_t) >= 2:
                            first_h = grp_t["visit_count"].iloc[:len(grp_t)//2].mean()
                            last_h = grp_t["visit_count"].iloc[len(grp_t)//2:].mean()
                            trend_map[cat_t] = (
                                "Rising" if last_h > first_h * 1.15
                                else "Declining" if last_h < first_h * 0.85
                                else "Stable"
                            )

                rows_html_mh = ""
                for _, r_mh in df_mhc2.iterrows():
                    cat_mh = str(r_mh.get("mh_category", ""))
                    standalone = int(r_mh.get("standalone_patients") or 0)
                    comorbid   = int(r_mh.get("comorbid_patients") or 0)
                    total_mh   = standalone + comorbid or 1
                    sa_pct = round(standalone / total_mh * 100)
                    co_pct = 100 - sa_pct
                    trend_label = trend_map.get(cat_mh, "—")
                    trend_color = "#16A34A" if trend_label == "Rising" else "#DC2626" if trend_label == "Declining" else "#D97706"
                    top_comorbid = str(r_mh.get("top_comorbidity") or "—")
                    rows_html_mh += (
                        f'<tr>'
                        f'<td style="padding:7px 8px;font-weight:600;font-size:12px">{cat_mh}</td>'
                        f'<td style="padding:7px 8px;font-size:11px;color:{trend_color};font-weight:700">{trend_label}</td>'
                        f'<td style="padding:7px 8px;font-size:11px">{sa_pct}% Standalone · {co_pct}% Comorbid</td>'
                        f'<td style="padding:7px 8px;font-size:11px">{standalone:,} / {comorbid:,}</td>'
                        f'<td style="padding:7px 8px;font-size:11px">{top_comorbid}</td>'
                        f'</tr>'
                    )
                st.markdown(
                    '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-family:sans-serif">'
                    '<thead><tr style="background:#7b5ea7;color:white">'
                    '<th style="padding:8px;text-align:left">Condition</th>'
                    '<th style="padding:8px;text-align:left">Trend</th>'
                    '<th style="padding:8px;text-align:left">Standalone vs Comorbid %</th>'
                    '<th style="padding:8px;text-align:right">Standalone / Comorbid Count</th>'
                    '<th style="padding:8px;text-align:left">Top Co-Condition</th>'
                    '</tr></thead><tbody>' + rows_html_mh + '</tbody></table></div>',
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.warning(f"MH Synergy Matrix: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — CLINICAL WORKLOAD & QUALITY
# ══════════════════════════════════════════════════════════════════════════════

def render_tab5_workload(filters: dict, run_query):
    _sh("Clinical Workload & Quality")
    _note("Scoped for Head of Clinical. "
          "Shortcut rate, BP omission, unplanned 72h returns.")

    # ── QUALITY SIGNALS SCORECARD ─────────────────────────────────────────────
    _sh("Clinical Quality Signals — Bubble View", mt=12)
    _note(
        "Bubble size = patient volume. Position = rate. "
        "Top-right bubbles are highest priority: high volume AND high risk rate."
    )
    try:
        df_sc  = Q.load_shortcut_rate(filters, run_query)
        df_bp  = Q.load_bp_omission_rate(filters, run_query)
        df_72  = Q.load_return_72h(filters, run_query)

        panels = [
            (df_sc,  "shortcut_rate_pct",  "chronic_visits",  "Shortcut Rate", ORANGE,
             "Single Dx on chronic patients", "Shortcut Rate %"),
            (df_bp,  "omission_pct",       "htn_visits",      "BP Omission",   CORAL,
             "HTN visits without BP recorded", "Omission %"),
            (df_72,  "return_72h_pct",     "total_visits",    "72h Return",    PURPLE,
             "Unplanned return within 72h", "Return Rate %"),
        ]

        cols3 = st.columns(3)
        for col, (df_p, rate_col, vol_col, title, color, subtitle, y_label) in zip(cols3, panels):
            with col:
                if df_p.empty:
                    st.info(f"No data: {title}")
                    continue
                df_p = df_p.copy()
                df_p[rate_col] = pd.to_numeric(df_p[rate_col], errors="coerce")
                df_p[vol_col]  = pd.to_numeric(df_p[vol_col],  errors="coerce")
                df_p = df_p.reset_index(drop=True)
                df_p["label"] = [f"C{i+1}" for i in range(len(df_p))]
                median_rate = df_p[rate_col].median()

                fig_q = go.Figure()
                fig_q.add_trace(go.Scatter(
                    x=df_p[vol_col],
                    y=df_p[rate_col],
                    mode="markers+text",
                    text=df_p["label"],
                    textposition="top center",
                    textfont=dict(size=9),
                    marker=dict(
                        size=df_p[vol_col].apply(
                            lambda v: max(10, min(40, v / df_p[vol_col].max() * 36 + 8))
                            if df_p[vol_col].max() > 0 else 12
                        ),
                        color=[CORAL if v > median_rate * 1.5
                               else ORANGE if v > median_rate
                               else TEAL
                               for v in df_p[rate_col]],
                        opacity=0.85,
                        line=dict(width=1, color="white"),
                    ),
                    customdata=df_p[["clinician", vol_col, rate_col]].values,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        f"Volume: %{{customdata[1]:.0f}}<br>"
                        f"{y_label}: %{{customdata[2]:.1f}}%<extra></extra>"
                    ),
                ))
                fig_q.add_hline(y=median_rate,
                                line=dict(color=GRAY, width=1, dash="dot"),
                                annotation_text=f"median {median_rate:.0f}%",
                                annotation_font=dict(size=8, color=GRAY))
                fig_q.update_layout(
                    height=280,
                    margin=dict(l=0, r=10, t=30, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    title=dict(text=f"<b>{title}</b><br><sup>{subtitle}</sup>",
                               font=dict(size=11), x=0),
                    xaxis=dict(title="Patient Volume", rangemode="tozero"),
                    yaxis=dict(title=y_label, rangemode="tozero"),
                    showlegend=False,
                )
                _pc(fig_q)

                # Top offender note
                worst_row = df_p.loc[df_p[rate_col].idxmax()]
                if float(worst_row[rate_col] or 0) > median_rate * 1.5:
                    _note(
                        f"{worst_row['clinician']}: {worst_row[rate_col]:.0f}% {y_label.lower()} "
                        f"({int(worst_row[vol_col] or 0):,} patients)."
                    )
    except Exception as e:
        st.warning(f"Quality signals: {e}")

    _gap(12)

    # ── SECTION F: CLINICIAN DOCUMENTATION RATES ─────────────────────────────
    _sh("F — Documentation Rates by Clinician — Ordered by Patient Load", mt=8)
    _note(
        "Clinicians with the highest load do not always have the lowest documentation rates "
        "— but the relationship is worth watching."
    )
    try:
        df = Q.load_clinician_load(filters, run_query)
        if not df.empty:
            for _c in ("vitals_rate_pct", "notes_rate_pct", "vitals_rate_new_pct",
                       "vitals_rate_returning_pct", "new_visit_pct",
                       "new_visits", "returning_visits", "total_visits",
                       "avg_daily_patients", "days_worked"):
                if _c in df.columns:
                    df[_c] = pd.to_numeric(df[_c], errors="coerce")

            df = df.reset_index(drop=True)
            df["label"] = ["Clinician " + str(i + 1) for i in range(len(df))]
            df15 = df.head(15).copy()

            zero_mask = (df15["vitals_rate_pct"].fillna(0) == 0) & (df15["notes_rate_pct"].fillna(0) == 0)
            has_zero = bool(zero_mask.any())

            non_zero = df15[~zero_mask]
            med_vitals = non_zero["vitals_rate_pct"].median() if not non_zero.empty else None
            med_notes  = non_zero["notes_rate_pct"].median()  if not non_zero.empty else None

            hover_new = [
                f"<b>{row.label}</b><br>"
                f"Vitals (new patients): {row.vitals_rate_new_pct:.0f}%<br>"
                f"New patients: {int(row.new_visits or 0):,}<extra></extra>"
                for _, row in df15.iterrows()
            ]
            hover_ret = [
                f"<b>{row.label}</b><br>"
                f"Vitals (returning/follow-up): {row.vitals_rate_returning_pct:.0f}%<br>"
                f"Returning patients: {int(row.returning_visits or 0):,}<extra></extra>"
                for _, row in df15.iterrows()
            ]
            hover_notes = [
                f"<b>{row.label}</b><br>"
                f"Notes documented: {row.notes_rate_pct:.0f}%<br>"
                f"Total visits: {int(row.total_visits or 0):,}<extra></extra>"
                for _, row in df15.iterrows()
            ]

            fig_doc = go.Figure()
            fig_doc.add_trace(go.Bar(
                y=df15["label"], x=df15["vitals_rate_new_pct"],
                name="Vitals — new patients", orientation="h",
                marker_color=TEAL,
                text=[f"{v:.0f}%" if pd.notna(v) else "—" for v in df15["vitals_rate_new_pct"]],
                textposition="outside", cliponaxis=False,
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_new,
            ))
            fig_doc.add_trace(go.Bar(
                y=df15["label"], x=df15["vitals_rate_returning_pct"],
                name="Vitals — returning/follow-up", orientation="h",
                marker_color=COOL_BLUE,
                text=[f"{v:.0f}%" if pd.notna(v) else "—" for v in df15["vitals_rate_returning_pct"]],
                textposition="outside", cliponaxis=False,
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_ret,
            ))
            fig_doc.add_trace(go.Bar(
                y=df15["label"], x=df15["notes_rate_pct"],
                name="Notes documented", orientation="h",
                marker_color=AFYA_BLUE,
                text=[f"{v:.0f}%" if pd.notna(v) else "—" for v in df15["notes_rate_pct"]],
                textposition="outside", cliponaxis=False,
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_notes,
            ))

            if med_vitals is not None:
                fig_doc.add_vline(
                    x=med_vitals,
                    line=dict(color=TEAL, width=1.5, dash="dot"),
                    annotation_text=f"Vitals median ({med_vitals:.0f}%)",
                    annotation_position="top right",
                    annotation_font=dict(size=9, color=TEAL),
                )
            if med_notes is not None:
                fig_doc.add_vline(
                    x=med_notes,
                    line=dict(color=AFYA_BLUE, width=1.5, dash="dot"),
                    annotation_text=f"Notes median ({med_notes:.0f}%)",
                    annotation_position="top left",
                    annotation_font=dict(size=9, color=AFYA_BLUE),
                )

            fig_doc.update_layout(
                barmode="group",
                height=max(420, len(df15) * 36),
                margin=dict(l=0, r=90, t=30, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Rate %", range=[0, 130],
                           showgrid=True, gridcolor="#EBF3FB"),
                yaxis=dict(autorange="reversed"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
            )
            _pc(fig_doc)
            _note(
                "Vitals rates are split by visit type. A low rate on returning/follow-up patients "
                "is often expected — clinicians managing chronic patients may not re-record vitals "
                "every visit. The new-patient vitals rate is the more meaningful quality signal.",
            )

            if has_zero:
                _note(
                    "Some clinicians show 0% on all rates — this likely reflects a data capture gap "
                    "rather than clinical behaviour.",
                    w=True,
                )

            _gap(8)
            st.markdown("**Full Load Summary**")
            df15_disp = df15[["label", "days_worked", "total_visits",
                               "avg_daily_patients", "new_visit_pct",
                               "vitals_rate_new_pct", "vitals_rate_returning_pct",
                               "notes_rate_pct"]].copy()
            df15_disp["visit_mix"] = df15["new_visit_pct"].apply(
                lambda x: f"{x:.0f}% new · {100 - x:.0f}% returning" if pd.notna(x) else "—"
            )
            _pc(table_fig(
                df15_disp[["label", "days_worked", "total_visits",
                            "avg_daily_patients", "visit_mix",
                            "vitals_rate_new_pct", "vitals_rate_returning_pct",
                            "notes_rate_pct"]],
                col_labels={
                    "label": "Clinician", "days_worked": "Days",
                    "total_visits": "Visits", "avg_daily_patients": "Avg/Day",
                    "visit_mix": "Visit Mix",
                    "vitals_rate_new_pct": "Vitals % (New)",
                    "vitals_rate_returning_pct": "Vitals % (Return)",
                    "notes_rate_pct": "Notes %",
                },
                fmt={"vitals_rate_new_pct": "pct", "vitals_rate_returning_pct": "pct",
                     "notes_rate_pct": "pct"},
                height=360,
            ))
    except Exception as e:
        st.warning(f"Clinician documentation rates: {e}")

    _gap(12)

    # ── CHRONIC LTFU RATES BY CLINICIAN ───────────────────────────────────────
    _sh("Chronic Patient LTFU Rate by Clinician", mt=8)
    _note(
        "% of each clinician's chronic patients who crossed 180 days without returning. "
        "High rates may reflect gaps in follow-up scheduling or patient satisfaction. Min 5 chronic patients."
    )
    try:
        df_cl = Q.load_clinician_ltfu_rate(filters, run_query)
        if not df_cl.empty:
            df_cl["ltfu_rate_pct"] = pd.to_numeric(df_cl["ltfu_rate_pct"], errors="coerce")
            df_cl["ltfu_count"] = pd.to_numeric(df_cl["ltfu_count"], errors="coerce")
            df_cl = df_cl.reset_index(drop=True)
            df_cl["clinician_label"] = [f"Clinician {i+1}" for i in range(len(df_cl))]

            c1, c2 = st.columns([1.4, 0.6])
            with c1:
                fig_cl = go.Figure(go.Bar(
                    x=df_cl["ltfu_rate_pct"],
                    y=df_cl["clinician_label"],
                    orientation="h",
                    marker_color=[CORAL if v > 60 else ORANGE if v > 40 else TEAL
                                  for v in df_cl["ltfu_rate_pct"]],
                    text=[f"{v:.0f}%" for v in df_cl["ltfu_rate_pct"]],
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>LTFU Rate: %{x:.1f}%<extra></extra>",
                ))
                fig_cl.add_vline(x=50, line=dict(color=GRAY, width=1.5, dash="dash"),
                                 annotation_text="50% reference",
                                 annotation_font=dict(size=9))
                fig_cl.update_layout(
                    height=max(300, len(df_cl) * 28),
                    margin=dict(l=0, r=60, t=20, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(title="% of Chronic Patients → 180d LTFU", range=[0, 115]),
                    yaxis=dict(title=""),
                )
                _pc(fig_cl)
            with c2:
                _pc(table_fig(
                    df_cl[["clinician_label", "chronic_patients_seen",
                            "ltfu_count", "ltfu_rate_pct"]].head(20),
                    col_labels={
                        "clinician_label": "Clinician",
                        "chronic_patients_seen": "Chronic Seen",
                        "ltfu_count": "LTFU",
                        "ltfu_rate_pct": "LTFU %",
                    },
                    fmt={"ltfu_rate_pct": "pct"},
                    height=420,
                ))
            worst = df_cl.iloc[0]
            _note(
                f"Highest LTFU rate: {worst['clinician_label']} — "
                f"{worst['ltfu_rate_pct']:.0f}% of their {int(worst['chronic_patients_seen']):,} "
                "chronic patients crossed 180 days. Review follow-up scheduling practices."
            )
    except Exception as e:
        st.warning(f"Clinician LTFU rate: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CLINICIAN VIEW
# ══════════════════════════════════════════════════════════════════════════════

def render_clinician_view(filters: dict, run_query):
    if not filters.get("schema") and not filters.get("clinician_id"):
        st.markdown(
            '<div style="background:#FFF8E7;border-left:4px solid #F59E0B;border-radius:4px;'
            'padding:10px 14px;font-size:12px;color:#92400E;margin-bottom:12px">'
            '⚠ Select a hospital schema and optionally your clinician ID in the sidebar.</div>',
            unsafe_allow_html=True,
        )

    try:
        df_pr = Q.load_priority_patients(filters, run_query)
    except Exception as e:
        st.warning(f"Priority patients: {e}")
        return

    if df_pr.empty:
        st.info("No patients found for the selected filters.")
        return

    for _c in ("days_since_last_visit", "is_chronic", "has_undetected_ncd",
               "had_op_to_ip", "unique_clinicians"):
        if _c in df_pr.columns:
            df_pr[_c] = pd.to_numeric(df_pr[_c], errors="coerce").fillna(0)

    # Deduplicate: one row per patient, highest priority kept
    _prio_ord = {"HIGH": 0, "MEDIUM": 1, "MONITOR": 2}
    df_pr["_ps"] = df_pr["priority_flag"].map(_prio_ord).fillna(3)
    df_pr = (df_pr.sort_values(["patient", "_ps", "days_since_last_visit"],
                               ascending=[True, True, False])
                  .drop_duplicates(subset=["patient"], keep="first")
                  .drop(columns=["_ps"])
                  .reset_index(drop=True))

    # ── BUILD PATIENT LIST FOR COMPONENT ─────────────────────────────────────
    def _sig(row):
        parts = []
        if row.get("had_op_to_ip"):                   parts.append("OP→IP")
        if row.get("has_undetected_ncd"):              parts.append("NCD undetected")
        if row.get("days_since_last_visit", 0) >= 90: parts.append("Long gap")
        if row.get("unique_clinicians", 1) >= 3:      parts.append("Irregular visits")
        return parts[0] if parts else ""

    patients_list = [
        {
            "id":        str(r["patient"]),
            "priority":  ("high"   if str(r.get("priority_flag")) == "HIGH"   else
                          "medium" if str(r.get("priority_flag")) == "MEDIUM" else "monitor"),
            "condition": str(r.get("primary_condition") or "Not recorded"),
            "days":      int(float(r.get("days_since_last_visit") or 0)),
            "signal":    _sig(r),
        }
        for _, r in df_pr.iterrows()
    ]

    # ── TWO-COLUMN LAYOUT ────────────────────────────────────────────────────
    left_col, right_col = st.columns([1, 2.5], gap="small")

    with left_col:
        clicked_id = _PATIENT_LIST_COMPONENT(
            patients=patients_list,
            selected_id=st.session_state.get("_pat_sel"),
            default=None,
            key="patient_list_comp",
            height=640,
        )
        if clicked_id and clicked_id != st.session_state.get("_pat_sel"):
            st.session_state["_pat_sel"] = clicked_id

    selected     = st.session_state.get("_pat_sel")
    sel_row_data = None
    if selected:
        _sel = df_pr[df_pr["patient"].astype(str) == str(selected)]
        sel_row_data = _sel.iloc[0] if not _sel.empty else None

    with right_col:
        if not selected or sel_row_data is None:
            st.markdown(
                '<div style="display:flex;flex-direction:column;align-items:center;'
                'justify-content:center;height:400px;color:#9ca3af;text-align:center;gap:8px">'
                '<div style="font-size:36px">👤</div>'
                '<div style="font-size:12px;font-weight:600">Select a patient</div>'
                '<div style="font-size:10px">Click any row on the left to view their clinical record</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            sel_schema = str(sel_row_data.get("source_schema") or filters.get("schema") or "")
            if sel_schema:
                _render_patient_card(selected, sel_schema, run_query, priority_row=sel_row_data)
            else:
                st.warning("No schema resolved for this patient — check source_schema column.")

    if False:
        html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:#f8fafc;font-family:'Montserrat',-apple-system,sans-serif;height:100vh;overflow:hidden;}}
.layout{{display:flex;height:100vh;overflow:hidden;}}
/* Left panel */
.lp{{width:260px;flex-shrink:0;background:#fff;border-right:1px solid #e5e7eb;display:flex;flex-direction:column;height:100%;}}
.lp-hdr{{padding:10px 12px 6px;border-bottom:1px solid #f3f4f6;}}
.lp-ttl{{font-size:9px;font-weight:800;letter-spacing:2px;text-transform:uppercase;color:#0072CE;margin-bottom:4px;}}
.lp-cnt{{font-size:10px;color:#9ca3af;margin-bottom:5px;}}
.srch{{width:100%;border:1px solid #e5e7eb;border-radius:6px;padding:5px 9px;font-size:11px;font-family:inherit;outline:none;color:#374151;}}
.srch:focus{{border-color:#0072CE;box-shadow:0 0 0 2px rgba(0,114,206,0.1);}}
.pills{{display:flex;gap:4px;padding:6px 12px;border-bottom:1px solid #f3f4f6;flex-wrap:wrap;}}
.pill{{font-size:9px;padding:2px 8px;border-radius:20px;border:1px solid #e5e7eb;cursor:pointer;font-weight:600;color:#6b7280;background:#fff;user-select:none;}}
.pill.a-all{{background:#0072CE;color:#fff;border-color:#0072CE;}}
.pill.a-high{{background:#ef4444;color:#fff;border-color:#ef4444;}}
.pill.a-med{{background:#f59e0b;color:#fff;border-color:#f59e0b;}}
.pill.a-mon{{background:#10b981;color:#fff;border-color:#10b981;}}
.pat-list{{flex:1;overflow-y:auto;}}
.pat-row{{display:flex;align-items:center;gap:7px;padding:7px 12px;cursor:pointer;border-bottom:1px solid #f9fafb;border-left:3px solid transparent;transition:background 0.1s;}}
.pat-row:hover{{background:#f0f7ff;}}
.pat-row.sel{{background:#EBF5FB;border-left-color:#0072CE;}}
.pdot{{width:7px;height:7px;border-radius:50%;flex-shrink:0;}}
.d-hi{{background:#ef4444;}}.d-med{{background:#f59e0b;}}.d-mon{{background:#10b981;}}
.pinf{{flex:1;min-width:0;}}
.pid{{font-size:11px;font-weight:700;color:#111827;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.pcnd{{font-size:9px;color:#9ca3af;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:1px;}}
.prt{{text-align:right;flex-shrink:0;}}
.pd{{font-size:10px;font-weight:700;color:#374151;}}
.pd.hi{{color:#ef4444;}}
.psig{{font-size:8px;color:#713f12;background:#fef9c3;padding:1px 5px;border-radius:10px;display:inline-block;margin-top:2px;}}
/* Right panel */
.rp{{flex:1;overflow-y:auto;background:#f8fafc;padding:14px;height:100%;}}
.empty-st{{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;color:#9ca3af;text-align:center;gap:8px;}}
.empty-ic{{font-size:36px;}}
.empty-tx{{font-size:12px;font-weight:600;}}
.empty-sub{{font-size:10px;}}
/* Patient card */
.dhdr{{padding:14px 16px;background:#fff;border:1px solid #e5e7eb;border-radius:10px;margin-bottom:12px;}}
.dname-row{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;}}
.dname{{font-size:15px;font-weight:700;color:#111827;}}
.dlv{{font-size:13px;font-weight:700;color:#b91c1c;text-align:right;}}
.dld{{font-size:10px;color:#9ca3af;text-align:right;margin-top:2px;}}
.dtags{{display:flex;gap:5px;flex-wrap:wrap;margin:5px 0;}}
.tag{{font-size:10px;padding:2px 8px;border-radius:20px;background:#f3f4f6;color:#374151;border:1px solid #e5e7eb;}}
.tag-hi{{background:#fee2e2;color:#991b1b;border-color:#fca5a5;}}
.tag-med{{background:#fef3c7;color:#92400e;border-color:#fcd34d;}}
.tag-mon{{background:#d1fae5;color:#065f46;border-color:#6ee7b7;}}
.sigpill{{display:inline-flex;align-items:center;gap:4px;background:#fef9c3;color:#713f12;font-size:10px;padding:2px 8px;border-radius:20px;font-weight:600;}}
.dsec{{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px 16px;margin-bottom:12px;}}
.dslabel{{font-size:9px;font-weight:700;letter-spacing:0.08em;color:#9ca3af;text-transform:uppercase;margin-bottom:10px;display:flex;align-items:center;gap:6px;}}
.dslabel::after{{content:'';flex:1;height:1px;background:#f3f4f6;}}
.met3{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:10px;}}
.mc{{background:#f8fafc;border-radius:8px;padding:9px 12px;border:1px solid #f3f4f6;}}
.mlabel{{font-size:9px;color:#9ca3af;margin-bottom:3px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;}}
.mval{{font-size:15px;font-weight:700;color:#111827;}}
.msub{{font-size:9px;color:#9ca3af;margin-top:2px;}}
.tl-legend{{display:flex;gap:12px;margin-top:8px;flex-wrap:wrap;}}
.vleg{{display:flex;align-items:center;gap:4px;font-size:10px;color:#6b7280;}}
.vleg-dot{{width:8px;height:8px;border-radius:50%;}}
.vleg-ring{{width:8px;height:8px;border-radius:50%;border:2px solid #3b82f6;background:transparent;}}
.esc-group{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:6px;}}
.esc-pill{{font-size:10px;padding:3px 10px;border-radius:20px;border:1px solid #e5e7eb;color:#6b7280;background:#f9fafb;}}
.esc-hi{{background:#fee2e2;color:#991b1b;border-color:#fca5a5;font-weight:600;}}
.esc-am{{background:#fef9c3;color:#713f12;border-color:#fde047;font-weight:600;}}
.esc-ok{{background:#d1fae5;color:#065f46;border-color:#6ee7b7;}}
.esc-det{{font-size:10px;color:#6b7280;margin-top:4px;}}
.radar-wrap{{display:grid;grid-template-columns:160px 1fr;gap:12px;align-items:start;}}
.radar-solo{{display:flex;flex-direction:column;align-items:center;}}
.radar-label{{font-size:9px;color:#9ca3af;margin-bottom:5px;text-align:center;}}
.spark-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;}}
.spark-card{{background:#f8fafc;border-radius:8px;padding:8px 10px;border:1px solid #f3f4f6;}}
.spark-name{{font-size:9px;color:#9ca3af;margin-bottom:2px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;}}
.spark-val{{font-size:16px;font-weight:700;color:#111827;}}
.spark-unit{{font-size:9px;color:#9ca3af;}}
.spark-ok{{font-size:9px;color:#059669;margin-top:2px;font-weight:600;}}
.spark-warn{{font-size:9px;color:#dc2626;margin-top:2px;font-weight:600;}}
.med-tl{{position:relative;padding-left:18px;}}
.med-tl-line{{position:absolute;left:6px;top:0;bottom:0;width:1px;background:#e5e7eb;}}
.med-ev{{position:relative;margin-bottom:14px;}}
.med-ev-dot{{width:10px;height:10px;border-radius:50%;position:absolute;left:-21px;top:2px;border:2px solid #fff;}}
.ev-active{{background:#10b981;outline:2px solid #10b981;}}
.ev-stopped{{background:#9ca3af;outline:2px solid #9ca3af;}}
.ev-changed{{background:#f59e0b;outline:2px solid #f59e0b;}}
.med-name{{font-size:11px;font-weight:600;color:#111827;}}
.med-meta{{font-size:10px;color:#9ca3af;margin-top:2px;}}
.mbadge{{font-size:9px;padding:1px 7px;border-radius:20px;margin-left:6px;font-weight:600;}}
.mb-a{{background:#d1fae5;color:#065f46;}}.mb-s{{background:#f3f4f6;color:#9ca3af;}}.mb-c{{background:#fef9c3;color:#713f12;}}
.alert-s{{background:#fef9c3;border-left:3px solid #f59e0b;border-radius:0 6px 6px 0;padding:6px 10px;font-size:10px;color:#713f12;margin-bottom:10px;display:flex;align-items:center;gap:6px;font-weight:600;}}
.lab-row{{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #f3f4f6;font-size:11px;}}
.lab-ok{{color:#059669;font-weight:600;}}.lab-warn{{color:#dc2626;font-weight:600;}}
</style></head><body>
<div class="layout">
  <div class="lp">
    <div class="lp-hdr">
      <div class="lp-ttl">Priority Patients</div>
      <div class="lp-cnt" id="lpCnt"></div>
      <input class="srch" type="text" placeholder="Search patient or condition…" id="srchInput" oninput="renderList()">
    </div>
    <div class="pills">
      <span class="pill a-all" data-f="all" onclick="setFilter('all',this)">All</span>
      <span class="pill" data-f="high" onclick="setFilter('high',this)">HIGH</span>
      <span class="pill" data-f="medium" onclick="setFilter('medium',this)">MEDIUM</span>
      <span class="pill" data-f="monitor" onclick="setFilter('monitor',this)">MONITOR</span>
    </div>
    <div class="pat-list" id="patList"></div>
  </div>
  <div class="rp" id="rPanel">
    <div id="emptyState" class="empty-st">
      <div class="empty-ic">👤</div>
      <div class="empty-tx">Select a patient</div>
      <div class="empty-sub">Click any row on the left to view their clinical record</div>
    </div>
    <div id="cardWrap" style="display:none">
      <div class="dhdr">
        <div class="dname-row">
          <div>
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
              <span class="dname" id="dId"></span>
              <span class="tag" id="pTag"></span>
            </div>
            <div class="dtags" id="dTags"></div>
            <span class="sigpill" id="dSig" style="display:none"></span>
          </div>
          <div>
            <div class="dlv" id="dDays"></div>
            <div class="dld" id="dDate"></div>
          </div>
        </div>
      </div>
      <div class="dsec">
        <div class="dslabel">1 — Visit cadence</div>
        <div class="met3">
          <div class="mc"><div class="mlabel">Total visits</div><div class="mval" id="vTotal"></div></div>
          <div class="mc"><div class="mlabel">Outpatient</div><div class="mval" id="vOP"></div></div>
          <div class="mc"><div class="mlabel">Inpatient</div><div class="mval" id="vIP"></div></div>
        </div>
        <div class="met3">
          <div class="mc"><div class="mlabel">First seen</div><div class="mval" style="font-size:11px" id="vFirst"></div></div>
          <div class="mc"><div class="mlabel">Avg gap</div><div class="mval" id="vAvgGap"></div></div>
          <div class="mc"><div class="mlabel">Frequency</div><div class="mval" style="font-size:11px" id="vFreq"></div><div class="msub" id="vFreqSub"></div></div>
        </div>
        <div style="font-size:9px;color:#9ca3af;margin-bottom:7px;margin-top:2px;">Visit purpose per date</div>
        <canvas id="visitCanvas" height="100" style="width:100%;display:block;"></canvas>
        <div class="tl-legend">
          <span class="vleg"><span class="vleg-dot" style="background:#10b981;"></span>Outpatient</span>
          <span class="vleg"><span class="vleg-ring"></span>Inpatient</span>
          <span class="vleg"><span class="vleg-dot" style="background:#7c3aed;width:7px;height:7px;"></span>Diagnosis</span>
          <span class="vleg"><span class="vleg-dot" style="background:#f59e0b;width:7px;height:7px;"></span>Follow-up</span>
          <span class="vleg"><span class="vleg-dot" style="background:#9ca3af;width:7px;height:7px;"></span>Meds pickup</span>
        </div>
      </div>
      <div class="dsec">
        <div class="dslabel">Escalation gap (OP → IP, same condition)</div>
        <div class="esc-group" id="escGroup"></div>
        <div class="esc-det" id="escDet"></div>
      </div>
      <div class="dsec">
        <div class="dslabel">2 — Illness history</div>
        <div style="font-size:9px;color:#9ca3af;margin-bottom:7px;">Each row = one condition · Each dot = occurrence date</div>
        <canvas id="illCanvas" style="width:100%;display:block;"></canvas>
      </div>
      <div class="dsec">
        <div class="dslabel">3 — Vitals</div>
        <div class="radar-wrap">
          <div class="radar-solo">
            <div class="radar-label">Current vs reference range</div>
            <canvas id="vitRadar" width="150" height="150"></canvas>
          </div>
          <div>
            <div style="font-size:9px;color:#9ca3af;margin-bottom:7px;">Trend over visits</div>
            <div class="spark-grid" id="sparkGrid"></div>
          </div>
        </div>
      </div>
      <div class="dsec">
        <div class="dslabel">4 — Labs &amp; haemogram</div>
        <div class="radar-wrap">
          <div class="radar-solo">
            <div class="radar-label">Haemogram vs reference</div>
            <canvas id="haeRadar" width="150" height="150"></canvas>
          </div>
          <div id="labDetail" style="padding-top:12px;width:100%;"></div>
        </div>
      </div>
      <div class="dsec">
        <div class="dslabel">5 — Medication timeline</div>
        <div class="alert-s" id="medAlert" style="display:none;">⚠ <span id="medAlertTxt"></span></div>
        <div class="med-tl"><div class="med-tl-line"></div><div id="medEvents"></div></div>
      </div>
    </div>
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
const patientsData = {patients_json};
const patientData  = {patient_json};
const selectedId   = {sel_id_js};

const PURPOSE_COLORS={{'diagnosis':'#7c3aed','follow-up':'#f59e0b','meds':'#9ca3af'}};
const PURPOSE_ROWS={{'diagnosis':0,'follow-up':1,'meds':2}};
const PURPOSE_LABELS=['Diagnosis','Follow-up','Meds pickup'];

let curFilter='all';
function setFilter(f,el){{
  curFilter=f;
  document.querySelectorAll('.pill').forEach(p=>{{
    const cls={{'all':'a-all','high':'a-high','medium':'a-med','monitor':'a-mon'}};
    p.className='pill'+(p.dataset.f===f?' '+cls[f]:'');
  }});
  renderList();
}}

function renderList(){{
  const q=(document.getElementById('srchInput').value||'').toLowerCase();
  const ul=document.getElementById('patList');ul.innerHTML='';
  const filtered=patientsData.filter(p=>{{
    if(curFilter!=='all'&&p.priority!==curFilter)return false;
    if(q&&!p.id.toLowerCase().includes(q)&&!p.condition.toLowerCase().includes(q))return false;
    return true;
  }});
  document.getElementById('lpCnt').textContent=filtered.length+' patient'+(filtered.length!==1?'s':'');
  filtered.forEach(p=>{{
    const row=document.createElement('div');
    row.className='pat-row'+(p.id===selectedId?' sel':'');
    row.innerHTML=
      `<span class="pdot d-${{p.priority}}"></span>`+
      `<div class="pinf"><div class="pid">Patient ${{p.id}}</div><div class="pcnd">${{p.condition}}</div></div>`+
      `<div class="prt"><div class="pd${{p.days>=90?' hi':''}}">${{p.days}}d</div>`+
      (p.signal?`<span class="psig">${{p.signal}}</span>`:'')+
      `</div>`;
    row.onclick=()=>{{
      const base=window.parent.location.href.split('?')[0];
      window.parent.location.href=base+'?_patient='+encodeURIComponent(p.id);
    }};
    ul.appendChild(row);
  }});
}}

function renderDetail(){{
  if(!patientData||!patientData.id)return;
  const patient=patientData;
  document.getElementById('emptyState').style.display='none';
  document.getElementById('cardWrap').style.display='block';
  document.getElementById('dId').textContent='Patient '+patient.id;
  const pTag=document.getElementById('pTag');
  const pCls={{'high':'tag-hi','medium':'tag-med','monitor':'tag-mon'}};
  pTag.textContent=patient.priority.charAt(0).toUpperCase()+patient.priority.slice(1)+' priority';
  pTag.className='tag '+(pCls[patient.priority]||'');
  document.getElementById('dTags').innerHTML=
    ['<span class="tag">'+patient.gender+'</span>',
     '<span class="tag">'+patient.age+'</span>',
     '<span class="tag">'+patient.condition+'</span>'].join('');
  const sigEl=document.getElementById('dSig');
  if(patient.signal){{sigEl.textContent='⏰ '+patient.signal;sigEl.style.display='inline-flex';}}
  document.getElementById('dDays').textContent=patient.days+'d ago';
  document.getElementById('dDate').textContent=patient.date;
  document.getElementById('vTotal').textContent=patient.visits.length;
  document.getElementById('vOP').textContent=patient.visits.filter(v=>v.type==='OP').length;
  document.getElementById('vIP').textContent=patient.visits.filter(v=>v.type==='IP').length;
  document.getElementById('vFirst').textContent=patient.firstSeen;
  document.getElementById('vAvgGap').textContent=patient.avgGap;
  document.getElementById('vFreq').textContent=patient.freq;
  document.getElementById('vFreqSub').textContent=patient.freqSub;
  const eg=document.getElementById('escGroup');
  const ed=document.getElementById('escDet');
  if(patient.escalations&&patient.escalations.length>0){{
    const b={{'0–15 days':[],'15–30 days':[],'> 30 days':[]}};
    patient.escalations.forEach(e=>{{if(b[e.bucket])b[e.bucket].push(e);}});
    const cls={{'0–15 days':'esc-hi','15–30 days':'esc-am','> 30 days':'esc-ok'}};
    Object.entries(b).forEach(([k,arr])=>{{
      if(arr.length){{const s=document.createElement('span');s.className='esc-pill '+cls[k];s.textContent=k+': '+arr.length;eg.appendChild(s);}}
    }});
    ed.innerHTML=patient.escalations.map(e=>`<div style="margin-bottom:3px;">· ${{e.dx}} — OP ${{e.opDate}} → IP ${{e.ipDate}} (${{e.gap}}d)</div>`).join('');
  }}else{{
    const s=document.createElement('span');s.className='esc-pill esc-ok';s.textContent='No escalations recorded';eg.appendChild(s);
  }}
  let vitChart=null,haeChart=null;
  const vit=patient.vitals;
  vitChart=drawRadar('vitRadar',['BP Sys','BP Dia','Heart rate','Blood sugar'],
    [vit.bp_sys.at(-1),vit.bp_dia.at(-1),vit.hr.at(-1),vit.sugar.at(-1)],
    [90,60,60,3.9],[120,80,100,5.6],vitChart);
  const sg=document.getElementById('sparkGrid');sg.innerHTML='';
  [{{name:'BP Systolic',unit:'mmHg',data:vit.bp_sys,min:90,max:120}},
   {{name:'BP Diastolic',unit:'mmHg',data:vit.bp_dia,min:60,max:80}},
   {{name:'Heart rate',unit:'bpm',data:vit.hr,min:60,max:100}},
   {{name:'Blood sugar',unit:'mmol/L',data:vit.sugar,min:3.9,max:5.6}}
  ].forEach(s=>{{
    const valid=s.data.filter(v=>v!==null);
    const last=valid.length?valid.at(-1):null;
    const inR=last!==null&&last>=s.min&&last<=s.max;
    const cid='sp_'+s.name.replace(/\s/g,'_');
    const card=document.createElement('div');card.className='spark-card';
    card.innerHTML=`<div class="spark-name">${{s.name}}</div><div class="spark-val">${{last!==null?last:'—'}}</div><div class="spark-unit">${{s.unit}}</div><div class="${{inR?'spark-ok':'spark-warn'}}">${{last===null?'No data':inR?'✓ In range':'⚠ Out of range'}}</div><canvas id="${{cid}}" width="90" height="32"></canvas>`;
    sg.appendChild(card);
    setTimeout(()=>{{const c=document.getElementById(cid);if(c)drawSparkline(c,s.data,s.min,s.max);}},120);
  }});
  const hae=patient.haemo;
  const hLabels=['WBC','RBC','Hgb','Platelets','MCV','MCHC'];
  const hData=[hae.wbc,hae.rbc,hae.hgb,hae.plt,hae.mcv,hae.mchc];
  const hMin=[4.0,4.5,12.0,150,80,32],hMax=[11.0,5.5,16.0,400,100,36];
  if(hData.some(v=>v!==null&&v!==undefined)){{
    haeChart=drawRadar('haeRadar',hLabels,hData,hMin,hMax,haeChart);
    document.getElementById('labDetail').innerHTML=hLabels.map((l,i)=>{{
      const v=hData[i];if(v===null||v===undefined)return '';
      const ok=v>=hMin[i]&&v<=hMax[i];
      return `<div class="lab-row"><span style="color:#6b7280;">${{l}}</span><span class="${{ok?'lab-ok':'lab-warn'}}">${{v}} ${{ok?'✓':'!'}}</span></div>`;
    }}).join('');
  }}else{{
    document.getElementById('haeRadar').parentElement.innerHTML='<div style="font-size:10px;color:#9ca3af;text-align:center;padding:24px 0;">No haemogram data recorded</div>';
    document.getElementById('labDetail').innerHTML='<div style="font-size:10px;color:#9ca3af;padding:24px 0;">Lab values not available</div>';
  }}
  const ma=document.getElementById('medAlert');
  if(patient.medChanges>0){{
    ma.style.display='flex';
    document.getElementById('medAlertTxt').textContent=patient.medChanges+' medication change'+(patient.medChanges>1?'s':'')+' detected — verify vitals stabilised after each switch.';
  }}
  const me=document.getElementById('medEvents');me.innerHTML='';
  patient.meds.slice().reverse().forEach(m=>{{
    const div=document.createElement('div');div.className='med-ev';
    const ec=m.status==='active'?'ev-active':m.status==='stopped'?'ev-stopped':'ev-changed';
    const bc=m.status==='active'?'mb-a':m.status==='stopped'?'mb-s':'mb-c';
    const lbl=m.status==='active'?'Active':m.status==='stopped'?'Stopped':'Changed → '+m.change;
    div.innerHTML=`<div class="med-ev-dot ${{ec}}"></div><div class="med-name">${{m.name}}<span class="mbadge ${{bc}}">${{lbl}}</span></div><div class="med-meta">${{m.date}}</div>`;
    me.appendChild(div);
  }});
}}

function drawVisitScatter(){{
  const canvas=document.getElementById('visitCanvas');
  if(!canvas||!patientData.visits.length)return;
  const W=canvas.parentElement.clientWidth||380;canvas.width=W;
  const ROWS=3,ROW_H=26,PAD_T=8,PAD_B=28,PAD_L=76,PAD_R=12;
  const H=PAD_T+ROWS*ROW_H+PAD_B;canvas.height=H;
  const ctx=canvas.getContext('2d');ctx.clearRect(0,0,W,H);
  const gridC='rgba(0,0,0,0.06)',textC='#9ca3af',trackW=W-PAD_L-PAD_R;
  const allMs=patientData.visits.map(v=>v.dateMs);
  const tMin=Math.min(...allMs),tMax=Math.max(...allMs),span=tMax-tMin||1;
  function xPos(ms){{return patientData.visits.length===1?PAD_L+trackW/2:PAD_L+((ms-tMin)/span)*trackW;}}
  ctx.font='9px Montserrat,sans-serif';
  PURPOSE_LABELS.forEach((lbl,ri)=>{{
    const y=PAD_T+ri*ROW_H+ROW_H/2;
    ctx.fillStyle=textC;ctx.textAlign='right';ctx.textBaseline='middle';ctx.fillText(lbl,PAD_L-8,y);
    ctx.strokeStyle=gridC;ctx.lineWidth=0.5;ctx.setLineDash([3,3]);
    ctx.beginPath();ctx.moveTo(PAD_L,y);ctx.lineTo(W-PAD_R,y);ctx.stroke();ctx.setLineDash([]);
  }});
  const datePositions=[];
  patientData.visits.forEach(v=>{{
    const ri=PURPOSE_ROWS[v.purpose]??1,x=xPos(v.dateMs),y=PAD_T+ri*ROW_H+ROW_H/2;
    const col=PURPOSE_COLORS[v.purpose]||'#9ca3af';
    if(v.type==='IP'){{
      ctx.strokeStyle='#3b82f6';ctx.lineWidth=2;ctx.beginPath();ctx.arc(x,y,7,0,Math.PI*2);ctx.stroke();
      ctx.fillStyle='rgba(59,130,246,0.12)';ctx.fill();
      ctx.fillStyle='#3b82f6';ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.font='600 8px Montserrat,sans-serif';ctx.fillText('IP',x,y);
    }}else{{ctx.fillStyle=col;ctx.beginPath();ctx.arc(x,y,6,0,Math.PI*2);ctx.fill();}}
    datePositions.push({{x,ms:v.dateMs}});
  }});
  const merged=[];
  datePositions.forEach(dp=>{{if(!merged.find(m=>Math.abs(m.x-dp.x)<20))merged.push(dp);}});
  ctx.font='9px Montserrat,sans-serif';ctx.fillStyle=textC;ctx.textAlign='center';ctx.textBaseline='top';
  merged.forEach(dp=>{{ctx.fillText(new Intl.DateTimeFormat('en-GB',{{day:'numeric',month:'short'}}).format(new Date(dp.ms)),dp.x,H-PAD_B+6);}});
  ctx.strokeStyle=gridC;ctx.lineWidth=0.5;ctx.setLineDash([]);
  ctx.beginPath();ctx.moveTo(PAD_L,H-PAD_B+2);ctx.lineTo(W-PAD_R,H-PAD_B+2);ctx.stroke();
}}

function drawIllnessScatter(){{
  const ills=patientData.illnesses;
  if(!ills||!ills.length)return;
  const canvas=document.getElementById('illCanvas');if(!canvas)return;
  const W=canvas.parentElement.clientWidth||380;canvas.width=W;
  const ROW_H=28,PAD_T=8,PAD_B=28,PAD_L=110,PAD_R=12;
  const H=PAD_T+ills.length*ROW_H+PAD_B;canvas.height=H;
  const ctx=canvas.getContext('2d');ctx.clearRect(0,0,W,H);
  const gridC='rgba(0,0,0,0.06)',textC='#9ca3af',trackW=W-PAD_L-PAD_R;
  const allMs=ills.flatMap(il=>il.dates);
  const tMin=Math.min(...allMs),tMax=Math.max(...allMs),span=tMax-tMin||1;
  function xPos(ms){{return allMs.length===1?PAD_L+trackW/2:PAD_L+((ms-tMin)/span)*trackW;}}
  const ILL_COLORS=['#10b981','#7c3aed','#ef4444','#f59e0b','#3b82f6','#ec4899'];
  ctx.font='9px Montserrat,sans-serif';
  const datePositions=[];
  ills.forEach((ill,ri)=>{{
    const y=PAD_T+ri*ROW_H+ROW_H/2,col=ILL_COLORS[ri%ILL_COLORS.length];
    ctx.fillStyle=textC;ctx.textAlign='right';ctx.textBaseline='middle';
    const nm=ill.name.length>16?ill.name.slice(0,15)+'…':ill.name;
    ctx.fillText(nm,PAD_L-8,y);
    ctx.strokeStyle=gridC;ctx.lineWidth=0.5;ctx.setLineDash([3,3]);
    ctx.beginPath();ctx.moveTo(PAD_L,y);ctx.lineTo(W-PAD_R,y);ctx.stroke();ctx.setLineDash([]);
    if(ill.dates.length>1){{
      const xs=ill.dates.map(xPos);ctx.strokeStyle=col;ctx.globalAlpha=0.2;ctx.lineWidth=1.5;
      ctx.beginPath();ctx.moveTo(xs[0],y);xs.slice(1).forEach(x=>ctx.lineTo(x,y));ctx.stroke();ctx.globalAlpha=1;
    }}
    ill.dates.forEach((ms,di)=>{{
      const x=xPos(ms);ctx.fillStyle=col;ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);ctx.fill();
      if(di===0){{ctx.strokeStyle='rgba(255,255,255,0.8)';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);ctx.stroke();}}
      datePositions.push({{x,ms}});
    }});
  }});
  const merged=[];
  datePositions.forEach(dp=>{{if(!merged.find(m=>Math.abs(m.x-dp.x)<22))merged.push(dp);}});
  ctx.font='9px Montserrat,sans-serif';ctx.fillStyle=textC;ctx.textAlign='center';ctx.textBaseline='top';
  merged.forEach(dp=>{{ctx.fillText(new Intl.DateTimeFormat('en-GB',{{day:'numeric',month:'short'}}).format(new Date(dp.ms)),dp.x,H-PAD_B+6);}});
  ctx.strokeStyle=gridC;ctx.lineWidth=0.5;ctx.beginPath();ctx.moveTo(PAD_L,H-PAD_B+2);ctx.lineTo(W-PAD_R,H-PAD_B+2);ctx.stroke();
}}

function drawRadar(canvasId,labels,data,refMin,refMax,existing){{
  const ctx=document.getElementById(canvasId).getContext('2d');
  if(existing)existing.destroy();
  const norm=data.map((v,i)=>{{
    if(v===null||v===undefined)return 0;
    const mid=(refMin[i]+refMax[i])/2,range=refMax[i]-refMin[i];
    return Math.max(0,Math.round(100-Math.min(100,Math.abs(v-mid)/(range*0.5)*100)));
  }});
  return new Chart(ctx,{{type:'radar',data:{{labels,datasets:[{{data:norm,backgroundColor:'rgba(16,185,129,0.15)',borderColor:'#10b981',borderWidth:1.5,pointBackgroundColor:'#10b981',pointRadius:3}}]}},options:{{responsive:false,scales:{{r:{{min:0,max:100,ticks:{{display:false}},grid:{{color:'rgba(0,0,0,0.07)'}},angleLines:{{color:'rgba(0,0,0,0.07)'}},pointLabels:{{color:'#9ca3af',font:{{size:9,family:'Montserrat,sans-serif'}}}}}}}},plugins:{{legend:{{display:false}}}}}}}});
}}

function drawSparkline(canvas,data,refMin,refMax){{
  const ctx=canvas.getContext('2d');const w=canvas.width,h=canvas.height;
  ctx.clearRect(0,0,w,h);
  const valid=data.filter(v=>v!==null&&v!==undefined);
  if(valid.length<2)return;
  const mn=Math.min(...valid)*0.95,mx=Math.max(...valid)*1.05;
  const pts=valid.map((v,i)=>{{return {{x:i*(w/(valid.length-1)),y:h-(((v-mn)/(mx-mn))*h*0.8+h*0.1)}};}});
  const inR=valid.every(v=>v>=refMin&&v<=refMax);
  ctx.strokeStyle=inR?'#10b981':'#ef4444';ctx.lineWidth=1.5;ctx.lineJoin='round';ctx.lineCap='round';
  ctx.beginPath();pts.forEach((p,i)=>i===0?ctx.moveTo(p.x,p.y):ctx.lineTo(p.x,p.y));ctx.stroke();
  pts.forEach(p=>{{ctx.beginPath();ctx.arc(p.x,p.y,2.5,0,Math.PI*2);ctx.fillStyle=inR?'#10b981':'#ef4444';ctx.fill();}});
}}

renderList();
renderDetail();
setTimeout(()=>{{if(patientData&&patientData.id){{drawVisitScatter();drawIllnessScatter();}}}},80);
</script></body></html>"""


def _sec_header(icon: str, title: str, subtitle: str = ""):
    sub = f'<span style="font-size:11px;color:#6B8CAE;font-weight:400;margin-left:8px">{subtitle}</span>' if subtitle else ""
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin:16px 0 10px 0;'
        f'padding-bottom:8px;border-bottom:1px solid #E8F0FA">'
        f'<span style="font-size:16px">{icon}</span>'
        f'<span style="font-size:13px;font-weight:700;color:#111827">{title}</span>'
        f'{sub}</div>',
        unsafe_allow_html=True,
    )


def _stat_card(label: str, value: str, note: str = "", color: str = "#0072CE",
               full_width: bool = False):
    w = "100%" if full_width else "auto"
    st.markdown(
        f'<div style="background:#F8FAFC;border:1px solid #E8F0FA;border-radius:8px;'
        f'padding:12px 16px;min-width:100px;width:{w}">'
        f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:1px;color:#6B8CAE;margin-bottom:4px">{label}</div>'
        f'<div style="font-size:22px;font-weight:800;color:{color};line-height:1">{value}</div>'
        + (f'<div style="font-size:10px;color:#9CA3AF;margin-top:3px">{note}</div>' if note else "")
        + f'</div>',
        unsafe_allow_html=True,
    )


def _build_patient_obj(patient_id: str, source_schema: str, run_query, priority_row=None) -> dict:
    def _fmt(dt):
        try:    return pd.Timestamp(dt).strftime("%d %b %Y")
        except: return "—"
    def _ms(dt):
        try:    return int(pd.Timestamp(dt).timestamp() * 1000)
        except: return 0

    obj = {
        "id": patient_id, "priority": "monitor", "condition": "—", "gender": "—",
        "age": "—", "days": 0, "date": "—", "signal": "", "firstSeen": "—",
        "avgGap": "—", "freq": "Once", "freqSub": "—", "medChanges": 0,
        "visits": [], "escalations": [], "illnesses": [],
        "vitals": {"bp_sys": [], "bp_dia": [], "hr": [], "sugar": []},
        "haemo": {"wbc": None, "rbc": None, "hgb": None, "plt": None, "mcv": None, "mchc": None},
        "meds": [],
    }
    if priority_row is not None:
        flag = str(priority_row.get("priority_flag", "MONITOR"))
        obj["priority"] = "high" if flag == "HIGH" else "medium" if flag == "MEDIUM" else "monitor"
        obj["condition"] = str(priority_row.get("primary_condition") or "—")
        obj["gender"]    = str(priority_row.get("gender") or "—")
        obj["age"]       = str(priority_row.get("age_group") or "—")
        days = int(float(priority_row.get("days_since_last_visit") or 0))
        obj["days"] = days
        sigs = []
        if days >= 90:                                              sigs.append("Long gap")
        if priority_row.get("has_undetected_ncd"):                 sigs.append("NCD undetected")
        if float(priority_row.get("unique_clinicians") or 0) >= 3: sigs.append("Irregular visits")
        obj["signal"] = sigs[0] if sigs else ""
    try:
        df_cad = Q.load_patient_visit_cadence(patient_id, source_schema, run_query)
        if not df_cad.empty:
            df_cad["visit_date"] = pd.to_datetime(df_cad["visit_date"], errors="coerce").dt.floor("D")
            df_cad["gap_days"]   = pd.to_numeric(df_cad["gap_days"], errors="coerce")
            df_cad = df_cad.sort_values("visit_date").reset_index(drop=True)
            total  = len(df_cad)
            gaps   = df_cad["gap_days"].dropna()
            ag     = float(gaps.mean()) if len(gaps) > 0 else None
            obj["firstSeen"] = _fmt(df_cad["visit_date"].min())
            obj["date"]      = _fmt(df_cad["visit_date"].max())
            if total == 1:
                obj["avgGap"] = "—"; obj["freq"] = "Once"; obj["freqSub"] = "1 visit"
            else:
                obj["avgGap"]  = f"{ag:.0f}d" if ag else "—"
                obj["freq"]    = ("Every ~1–2 wk" if ag and ag < 14 else
                                  "Every ~2–4 wk" if ag and ag < 30 else
                                  "Every ~1–2 mo" if ag and ag < 60 else
                                  "Every ~2–3 mo" if ag and ag < 90 else "Every ~3+ mo")
                obj["freqSub"] = f"{total} visits"
            cond = obj["condition"]
            for idx, row in df_cad.iterrows():
                dt = row["visit_date"]
                if pd.isna(dt): continue
                vt   = "IP" if str(row.get("visit_type", "")).lower() == "inpatient" else "OP"
                purp = "diagnosis" if idx == 0 else "follow-up"
                obj["visits"].append({"type": vt, "purpose": purp,
                                      "dateMs": _ms(dt), "dateStr": _fmt(dt), "dx": cond})
    except Exception:
        pass
    try:
        df_ill = Q.load_patient_illness_history(patient_id, source_schema, run_query)
        if not df_ill.empty:
            df_ill["visit_date"] = pd.to_datetime(df_ill["visit_date"], errors="coerce")
            df_ill = df_ill.sort_values("visit_date")
            ill_map: dict = {}
            for _, row in df_ill.iterrows():
                grp = str(row.get("disease_group") or "Unspecified")
                dt  = row["visit_date"]
                if pd.isna(dt): continue
                ill_map.setdefault(grp, []).append(_ms(dt))
            obj["illnesses"] = [{"name": n, "dates": sorted(d)} for n, d in ill_map.items() if d]
            op_v = df_ill[df_ill["visit_type"] == "Outpatient"]
            ip_v = df_ill[df_ill["visit_type"] == "Inpatient"]
            for _, ip_row in ip_v.iterrows():
                ip_dt = ip_row["visit_date"]; ip_dx = str(ip_row.get("disease_group") or "")
                if pd.isna(ip_dt): continue
                cands = op_v[(op_v["visit_date"] < ip_dt) &
                             (op_v["visit_date"] >= ip_dt - pd.Timedelta(days=90)) &
                             (op_v["disease_group"].astype(str) == ip_dx)]
                if not cands.empty:
                    op_dt = cands.iloc[-1]["visit_date"]
                    gap   = int((ip_dt - op_dt).days)
                    obj["escalations"].append({
                        "dx": ip_dx, "opDate": _fmt(op_dt), "ipDate": _fmt(ip_dt), "gap": gap,
                        "bucket": ("0–15 days" if gap <= 15 else "15–30 days" if gap <= 30 else "> 30 days"),
                    })
    except Exception:
        pass
    try:
        df_vit = Q.load_patient_vitals_trend(patient_id, source_schema, run_query)
        if not df_vit.empty:
            df_vit = df_vit.sort_values("reading_rank", ascending=False)
            def _f(v): return float(v) if v is not None and not pd.isnull(v) else None
            obj["vitals"]["bp_sys"] = [_f(v) for v in df_vit["bp_systolic"]]
            obj["vitals"]["bp_dia"] = [_f(v) for v in df_vit["bp_diastolic"]]
            obj["vitals"]["sugar"]  = [_f(v) for v in df_vit["blood_sugar"]]
            obj["vitals"]["hr"]     = [None] * len(df_vit)
    except Exception:
        pass
    try:
        df_med = Q.load_patient_medication_change_timeline(patient_id, source_schema, run_query)
        if not df_med.empty:
            df_med = df_med.sort_values("prescription_date")
            meds_js: list = []; med_changes = 0
            for _, row in df_med.iterrows():
                drug   = str(row.get("drug_name") or "Unknown")
                prev   = row.get("prev_drug")
                is_new = int(row.get("is_new_drug") or 0)
                date_s = _fmt(pd.to_datetime(row.get("prescription_date"), errors="coerce"))
                if is_new and prev and str(prev) != drug:
                    for m in reversed(meds_js):
                        if m["name"] == str(prev) and m["status"] == "active":
                            m["status"] = "changed"; m["change"] = drug
                            med_changes += 1; break
                meds_js.append({"name": drug, "date": date_s, "status": "active", "change": None})
            obj["meds"]       = meds_js
            obj["medChanges"] = med_changes
    except Exception:
        pass
    return obj


def _render_patient_card(patient_id: str, source_schema: str, run_query, priority_row=None):
    import json
    import streamlit.components.v1 as _components

    def _fmt(dt):
        try:
            return pd.Timestamp(dt).strftime("%d %b %Y")
        except Exception:
            return "—"

    def _ms(dt):
        try:
            return int(pd.Timestamp(dt).timestamp() * 1000)
        except Exception:
            return 0

    # ── Build patient object ─────────────────────────────────────────────────
    obj = {
        "id": patient_id,
        "priority": "monitor",
        "condition": "—",
        "gender": "—",
        "age": "—",
        "days": 0,
        "date": "—",
        "signal": "",
        "firstSeen": "—",
        "avgGap": "—",
        "freq": "Once",
        "freqSub": "—",
        "medChanges": 0,
        "visits": [],
        "escalations": [],
        "illnesses": [],
        "vitals": {"bp_sys": [], "bp_dia": [], "hr": [], "sugar": []},
        "haemo": {"wbc": None, "rbc": None, "hgb": None,
                  "plt": None, "mcv": None, "mchc": None},
        "meds": [],
    }

    if priority_row is not None:
        flag = str(priority_row.get("priority_flag", "MONITOR"))
        obj["priority"] = ("high"   if flag == "HIGH"   else
                           "medium" if flag == "MEDIUM" else "monitor")
        obj["condition"] = str(priority_row.get("primary_condition") or "—")
        obj["gender"]    = str(priority_row.get("gender")    or "—")
        obj["age"]       = str(priority_row.get("age_group") or "—")
        days = int(float(priority_row.get("days_since_last_visit") or 0))
        obj["days"] = days
        sigs = []
        if days >= 90:
            sigs.append("Long gap")
        if priority_row.get("has_undetected_ncd"):
            sigs.append("NCD undetected")
        if float(priority_row.get("unique_clinicians") or 0) >= 3:
            sigs.append("Irregular visits")
        obj["signal"] = sigs[0] if sigs else ""

    # Visit cadence
    try:
        df_cad = Q.load_patient_visit_cadence(patient_id, source_schema, run_query)
        if not df_cad.empty:
            df_cad["visit_date"] = pd.to_datetime(
                df_cad["visit_date"], errors="coerce").dt.floor("D")
            df_cad["gap_days"] = pd.to_numeric(df_cad["gap_days"], errors="coerce")
            df_cad = df_cad.sort_values("visit_date").reset_index(drop=True)
            total  = len(df_cad)
            gaps   = df_cad["gap_days"].dropna()
            ag     = float(gaps.mean()) if len(gaps) > 0 else None
            obj["firstSeen"] = _fmt(df_cad["visit_date"].min())
            obj["date"]      = _fmt(df_cad["visit_date"].max())
            if total == 1:
                obj["avgGap"] = "—"; obj["freq"] = "Once"; obj["freqSub"] = "1 visit"
            else:
                obj["avgGap"]  = f"{ag:.0f}d" if ag else "—"
                obj["freq"]    = ("Every ~1–2 wk"  if ag and ag < 14  else
                                  "Every ~2–4 wk"  if ag and ag < 30  else
                                  "Every ~1–2 mo"  if ag and ag < 60  else
                                  "Every ~2–3 mo"  if ag and ag < 90  else
                                  "Every ~3+ mo")
                obj["freqSub"] = f"{total} visits"
            cond = obj["condition"]
            for idx, row in df_cad.iterrows():
                dt = row["visit_date"]
                if pd.isna(dt): continue
                vt  = "IP" if str(row.get("visit_type","")).lower() == "inpatient" else "OP"
                purp = "diagnosis" if idx == 0 else "follow-up"
                obj["visits"].append({"type": vt, "purpose": purp,
                                      "dateMs": _ms(dt), "dateStr": _fmt(dt), "dx": cond})
    except Exception:
        pass

    # Illness history + escalations
    try:
        df_ill = Q.load_patient_illness_history(patient_id, source_schema, run_query)
        if not df_ill.empty:
            df_ill["visit_date"] = pd.to_datetime(df_ill["visit_date"], errors="coerce")
            df_ill = df_ill.sort_values("visit_date")
            ill_map: dict = {}
            for _, row in df_ill.iterrows():
                grp = str(row.get("disease_group") or "Unspecified")
                dt  = row["visit_date"]
                if pd.isna(dt): continue
                ill_map.setdefault(grp, []).append(_ms(dt))
            obj["illnesses"] = [{"name": n, "dates": sorted(d)}
                                 for n, d in ill_map.items() if d]
            op_v = df_ill[df_ill["visit_type"] == "Outpatient"]
            ip_v = df_ill[df_ill["visit_type"] == "Inpatient"]
            for _, ip_row in ip_v.iterrows():
                ip_dt = ip_row["visit_date"]; ip_dx = str(ip_row.get("disease_group") or "")
                if pd.isna(ip_dt): continue
                cands = op_v[
                    (op_v["visit_date"] < ip_dt) &
                    (op_v["visit_date"] >= ip_dt - pd.Timedelta(days=90)) &
                    (op_v["disease_group"].astype(str) == ip_dx)
                ]
                if not cands.empty:
                    op_dt = cands.iloc[-1]["visit_date"]
                    gap   = int((ip_dt - op_dt).days)
                    obj["escalations"].append({
                        "dx":     ip_dx,
                        "opDate": _fmt(op_dt),
                        "ipDate": _fmt(ip_dt),
                        "gap":    gap,
                        "bucket": ("0–15 days"  if gap <= 15 else
                                   "15–30 days" if gap <= 30 else "> 30 days"),
                    })
    except Exception:
        pass

    # Vitals
    try:
        df_vit = Q.load_patient_vitals_trend(patient_id, source_schema, run_query)
        if not df_vit.empty:
            df_vit = df_vit.sort_values("reading_rank", ascending=False)
            def _f(v): return float(v) if v is not None and not pd.isnull(v) else None
            obj["vitals"]["bp_sys"] = [_f(v) for v in df_vit["bp_systolic"]]
            obj["vitals"]["bp_dia"] = [_f(v) for v in df_vit["bp_diastolic"]]
            obj["vitals"]["sugar"]  = [_f(v) for v in df_vit["blood_sugar"]]
            obj["vitals"]["hr"]     = [None] * len(df_vit)
    except Exception:
        pass

    # Medications
    try:
        df_med = Q.load_patient_medication_change_timeline(patient_id, source_schema, run_query)
        if not df_med.empty:
            df_med = df_med.sort_values("prescription_date")
            meds_js: list = []; med_changes = 0
            for _, row in df_med.iterrows():
                drug = str(row.get("drug_name") or "Unknown")
                prev = row.get("prev_drug")
                is_new = int(row.get("is_new_drug") or 0)
                date_s = _fmt(pd.to_datetime(row.get("prescription_date"), errors="coerce"))
                if is_new and prev and str(prev) != drug:
                    for m in reversed(meds_js):
                        if m["name"] == str(prev) and m["status"] == "active":
                            m["status"] = "changed"; m["change"] = drug
                            med_changes += 1; break
                meds_js.append({"name": drug, "date": date_s, "status": "active", "change": None})
            obj["meds"]       = meds_js
            obj["medChanges"] = med_changes
    except Exception:
        pass

    patient_json = json.dumps(obj, ensure_ascii=False, default=str)

    # ── HTML template ────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:#f8fafc;font-family:'Montserrat',-apple-system,sans-serif;color:#1a1a2e;padding:0 0 24px 0;}}
.dhdr{{padding:14px 16px;background:#fff;border:1px solid #e5e7eb;border-radius:10px;margin-bottom:12px;}}
.dname-row{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;}}
.dname{{font-size:15px;font-weight:700;color:#111827;}}
.dlv{{font-size:13px;font-weight:700;color:#b91c1c;text-align:right;}}
.dld{{font-size:10px;color:#9ca3af;text-align:right;margin-top:2px;}}
.dtags{{display:flex;gap:5px;flex-wrap:wrap;margin:5px 0;}}
.tag{{font-size:10px;padding:2px 8px;border-radius:20px;background:#f3f4f6;color:#374151;border:1px solid #e5e7eb;}}
.tag-hi{{background:#fee2e2;color:#991b1b;border-color:#fca5a5;}}
.tag-med{{background:#fef3c7;color:#92400e;border-color:#fcd34d;}}
.tag-mon{{background:#d1fae5;color:#065f46;border-color:#6ee7b7;}}
.sigpill{{display:inline-flex;align-items:center;gap:4px;background:#fef9c3;color:#713f12;font-size:10px;padding:2px 8px;border-radius:20px;font-weight:600;}}
.dsec{{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px 16px;margin-bottom:12px;}}
.dslabel{{font-size:9px;font-weight:700;letter-spacing:0.08em;color:#9ca3af;text-transform:uppercase;margin-bottom:10px;display:flex;align-items:center;gap:6px;}}
.dslabel::after{{content:'';flex:1;height:1px;background:#f3f4f6;}}
.met3{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:10px;}}
.mc{{background:#f8fafc;border-radius:8px;padding:9px 12px;border:1px solid #f3f4f6;}}
.mlabel{{font-size:9px;color:#9ca3af;margin-bottom:3px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;}}
.mval{{font-size:15px;font-weight:700;color:#111827;}}
.msub{{font-size:9px;color:#9ca3af;margin-top:2px;}}
.tl-legend{{display:flex;gap:12px;margin-top:8px;flex-wrap:wrap;}}
.vleg{{display:flex;align-items:center;gap:4px;font-size:10px;color:#6b7280;}}
.vleg-dot{{width:8px;height:8px;border-radius:50%;}}
.vleg-ring{{width:8px;height:8px;border-radius:50%;border:2px solid #3b82f6;background:transparent;}}
.esc-group{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:6px;}}
.esc-pill{{font-size:10px;padding:3px 10px;border-radius:20px;border:1px solid #e5e7eb;color:#6b7280;background:#f9fafb;}}
.esc-hi{{background:#fee2e2;color:#991b1b;border-color:#fca5a5;font-weight:600;}}
.esc-am{{background:#fef9c3;color:#713f12;border-color:#fde047;font-weight:600;}}
.esc-ok{{background:#d1fae5;color:#065f46;border-color:#6ee7b7;}}
.esc-det{{font-size:10px;color:#6b7280;margin-top:4px;}}
.radar-wrap{{display:grid;grid-template-columns:160px 1fr;gap:12px;align-items:start;}}
.radar-solo{{display:flex;flex-direction:column;align-items:center;}}
.radar-label{{font-size:9px;color:#9ca3af;margin-bottom:5px;text-align:center;}}
.spark-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;}}
.spark-card{{background:#f8fafc;border-radius:8px;padding:8px 10px;border:1px solid #f3f4f6;}}
.spark-name{{font-size:9px;color:#9ca3af;margin-bottom:2px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;}}
.spark-val{{font-size:16px;font-weight:700;color:#111827;}}
.spark-unit{{font-size:9px;color:#9ca3af;}}
.spark-ok{{font-size:9px;color:#059669;margin-top:2px;font-weight:600;}}
.spark-warn{{font-size:9px;color:#dc2626;margin-top:2px;font-weight:600;}}
.med-tl{{position:relative;padding-left:18px;}}
.med-tl-line{{position:absolute;left:6px;top:0;bottom:0;width:1px;background:#e5e7eb;}}
.med-ev{{position:relative;margin-bottom:14px;}}
.med-ev-dot{{width:10px;height:10px;border-radius:50%;position:absolute;left:-21px;top:2px;border:2px solid #fff;}}
.ev-active{{background:#10b981;outline:2px solid #10b981;}}
.ev-stopped{{background:#9ca3af;outline:2px solid #9ca3af;}}
.ev-changed{{background:#f59e0b;outline:2px solid #f59e0b;}}
.med-name{{font-size:11px;font-weight:600;color:#111827;}}
.med-meta{{font-size:10px;color:#9ca3af;margin-top:2px;}}
.mbadge{{font-size:9px;padding:1px 7px;border-radius:20px;margin-left:6px;font-weight:600;}}
.mb-a{{background:#d1fae5;color:#065f46;}}
.mb-s{{background:#f3f4f6;color:#9ca3af;}}
.mb-c{{background:#fef9c3;color:#713f12;}}
.alert-s{{background:#fef9c3;border-left:3px solid #f59e0b;border-radius:0 6px 6px 0;padding:6px 10px;font-size:10px;color:#713f12;margin-bottom:10px;display:flex;align-items:center;gap:6px;font-weight:600;}}
.lab-row{{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #f3f4f6;font-size:11px;}}
.lab-ok{{color:#059669;font-weight:600;}}
.lab-warn{{color:#dc2626;font-weight:600;}}
</style></head><body>
<div class="dhdr">
  <div class="dname-row">
    <div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
        <span class="dname" id="dId"></span>
        <span class="tag" id="pTag"></span>
      </div>
      <div class="dtags" id="dTags"></div>
      <span class="sigpill" id="dSig" style="display:none"></span>
    </div>
    <div>
      <div class="dlv" id="dDays"></div>
      <div class="dld" id="dDate"></div>
    </div>
  </div>
</div>

<div class="dsec">
  <div class="dslabel">1 — Visit cadence</div>
  <div class="met3">
    <div class="mc"><div class="mlabel">Total visits</div><div class="mval" id="vTotal"></div></div>
    <div class="mc"><div class="mlabel">Outpatient</div><div class="mval" id="vOP"></div></div>
    <div class="mc"><div class="mlabel">Inpatient</div><div class="mval" id="vIP"></div></div>
  </div>
  <div class="met3">
    <div class="mc"><div class="mlabel">First seen</div><div class="mval" style="font-size:11px" id="vFirst"></div></div>
    <div class="mc"><div class="mlabel">Avg gap</div><div class="mval" id="vAvgGap"></div></div>
    <div class="mc"><div class="mlabel">Frequency</div><div class="mval" style="font-size:11px" id="vFreq"></div><div class="msub" id="vFreqSub"></div></div>
  </div>
  <div style="font-size:9px;color:#9ca3af;margin-bottom:7px;margin-top:2px;">Visit purpose per date</div>
  <canvas id="visitCanvas" height="100" style="width:100%;display:block;"></canvas>
  <div class="tl-legend">
    <span class="vleg"><span class="vleg-dot" style="background:#10b981;"></span>Outpatient</span>
    <span class="vleg"><span class="vleg-ring"></span>Inpatient</span>
    <span class="vleg"><span class="vleg-dot" style="background:#7c3aed;width:7px;height:7px;"></span>Diagnosis</span>
    <span class="vleg"><span class="vleg-dot" style="background:#f59e0b;width:7px;height:7px;"></span>Follow-up</span>
    <span class="vleg"><span class="vleg-dot" style="background:#9ca3af;width:7px;height:7px;"></span>Meds pickup</span>
  </div>
</div>

<div class="dsec">
  <div class="dslabel">Escalation gap (OP → IP, same condition)</div>
  <div class="esc-group" id="escGroup"></div>
  <div class="esc-det" id="escDet"></div>
</div>

<div class="dsec">
  <div class="dslabel">2 — Illness history</div>
  <div style="font-size:9px;color:#9ca3af;margin-bottom:7px;">Each row = one condition · Each dot = occurrence date</div>
  <canvas id="illCanvas" style="width:100%;display:block;"></canvas>
</div>

<div class="dsec">
  <div class="dslabel">3 — Vitals</div>
  <div class="radar-wrap">
    <div class="radar-solo">
      <div class="radar-label">Current vs reference range</div>
      <canvas id="vitRadar" width="150" height="150"></canvas>
    </div>
    <div>
      <div style="font-size:9px;color:#9ca3af;margin-bottom:7px;">Trend over visits</div>
      <div class="spark-grid" id="sparkGrid"></div>
    </div>
  </div>
</div>

<div class="dsec">
  <div class="dslabel">4 — Labs &amp; haemogram</div>
  <div class="radar-wrap">
    <div class="radar-solo">
      <div class="radar-label">Haemogram vs reference</div>
      <canvas id="haeRadar" width="150" height="150"></canvas>
    </div>
    <div id="labDetail" style="padding-top:12px;width:100%;"></div>
  </div>
</div>

<div class="dsec">
  <div class="dslabel">5 — Medication timeline</div>
  <div class="alert-s" id="medAlert" style="display:none;">
    ⚠ <span id="medAlertTxt"></span>
  </div>
  <div class="med-tl"><div class="med-tl-line"></div><div id="medEvents"></div></div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
const patient = {patient_json};

const PURPOSE_COLORS={{'diagnosis':'#7c3aed','follow-up':'#f59e0b','meds':'#9ca3af'}};
const PURPOSE_ROWS={{'diagnosis':0,'follow-up':1,'meds':2}};
const PURPOSE_LABELS=['Diagnosis','Follow-up','Meds pickup'];

// ── Header ────────────────────────────────────────────────────────────────
document.getElementById('dId').textContent='Patient '+patient.id;
const pTag=document.getElementById('pTag');
const pCls={{'high':'tag-hi','medium':'tag-med','monitor':'tag-mon'}};
pTag.textContent=patient.priority.charAt(0).toUpperCase()+patient.priority.slice(1)+' priority';
pTag.className='tag '+(pCls[patient.priority]||'');
document.getElementById('dTags').innerHTML=
  ['<span class="tag">'+patient.gender+'</span>',
   '<span class="tag">'+patient.age+'</span>',
   '<span class="tag">'+patient.condition+'</span>'].join('');
const sigEl=document.getElementById('dSig');
if(patient.signal){{sigEl.textContent='⏰ '+patient.signal;sigEl.style.display='inline-flex';}}
document.getElementById('dDays').textContent=patient.days+'d ago';
document.getElementById('dDate').textContent=patient.date;

// ── Visit cadence metrics ─────────────────────────────────────────────────
document.getElementById('vTotal').textContent=patient.visits.length;
document.getElementById('vOP').textContent=patient.visits.filter(v=>v.type==='OP').length;
document.getElementById('vIP').textContent=patient.visits.filter(v=>v.type==='IP').length;
document.getElementById('vFirst').textContent=patient.firstSeen;
document.getElementById('vAvgGap').textContent=patient.avgGap;
document.getElementById('vFreq').textContent=patient.freq;
document.getElementById('vFreqSub').textContent=patient.freqSub;

// ── Visit scatter ─────────────────────────────────────────────────────────
function drawVisitScatter(){{
  const canvas=document.getElementById('visitCanvas');
  const W=canvas.parentElement.clientWidth||380;
  canvas.width=W;
  const ROWS=3,ROW_H=26,PAD_T=8,PAD_B=28,PAD_L=76,PAD_R=12;
  const H=PAD_T+ROWS*ROW_H+PAD_B; canvas.height=H;
  const ctx=canvas.getContext('2d'); ctx.clearRect(0,0,W,H);
  const gridC='rgba(0,0,0,0.06)'; const textC='#9ca3af';
  const trackW=W-PAD_L-PAD_R;
  const allMs=patient.visits.map(v=>v.dateMs);
  const tMin=Math.min(...allMs),tMax=Math.max(...allMs),span=tMax-tMin||1;
  function xPos(ms){{return patient.visits.length===1?PAD_L+trackW/2:PAD_L+((ms-tMin)/span)*trackW;}}
  ctx.font='9px Montserrat,sans-serif';
  PURPOSE_LABELS.forEach((lbl,ri)=>{{
    const y=PAD_T+ri*ROW_H+ROW_H/2;
    ctx.fillStyle=textC;ctx.textAlign='right';ctx.textBaseline='middle';
    ctx.fillText(lbl,PAD_L-8,y);
    ctx.strokeStyle=gridC;ctx.lineWidth=0.5;ctx.setLineDash([3,3]);
    ctx.beginPath();ctx.moveTo(PAD_L,y);ctx.lineTo(W-PAD_R,y);ctx.stroke();
    ctx.setLineDash([]);
  }});
  const datePositions=[];
  patient.visits.forEach(v=>{{
    const ri=PURPOSE_ROWS[v.purpose]??1;
    const x=xPos(v.dateMs),y=PAD_T+ri*ROW_H+ROW_H/2;
    const col=PURPOSE_COLORS[v.purpose]||'#9ca3af';
    if(v.type==='IP'){{
      ctx.strokeStyle='#3b82f6';ctx.lineWidth=2;
      ctx.beginPath();ctx.arc(x,y,7,0,Math.PI*2);ctx.stroke();
      ctx.fillStyle='rgba(59,130,246,0.12)';ctx.fill();
      ctx.fillStyle='#3b82f6';ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.font='600 8px Montserrat,sans-serif';ctx.fillText('IP',x,y);
    }}else{{
      ctx.fillStyle=col;ctx.beginPath();ctx.arc(x,y,6,0,Math.PI*2);ctx.fill();
    }}
    datePositions.push({{x,ms:v.dateMs}});
  }});
  const merged=[];
  datePositions.forEach(dp=>{{if(!merged.find(m=>Math.abs(m.x-dp.x)<20))merged.push(dp);}});
  ctx.font='9px Montserrat,sans-serif';ctx.fillStyle=textC;ctx.textAlign='center';ctx.textBaseline='top';
  merged.forEach(dp=>{{
    ctx.fillText(new Intl.DateTimeFormat('en-GB',{{day:'numeric',month:'short'}}).format(new Date(dp.ms)),dp.x,H-PAD_B+6);
  }});
  ctx.strokeStyle=gridC;ctx.lineWidth=0.5;ctx.setLineDash([]);
  ctx.beginPath();ctx.moveTo(PAD_L,H-PAD_B+2);ctx.lineTo(W-PAD_R,H-PAD_B+2);ctx.stroke();
}}

// ── Illness scatter ───────────────────────────────────────────────────────
function drawIllnessScatter(){{
  const ills=patient.illnesses;
  if(!ills||!ills.length)return;
  const canvas=document.getElementById('illCanvas');
  const W=canvas.parentElement.clientWidth||380;
  canvas.width=W;
  const ROW_H=28,PAD_T=8,PAD_B=28,PAD_L=110,PAD_R=12;
  const H=PAD_T+ills.length*ROW_H+PAD_B; canvas.height=H;
  const ctx=canvas.getContext('2d'); ctx.clearRect(0,0,W,H);
  const gridC='rgba(0,0,0,0.06)'; const textC='#9ca3af';
  const trackW=W-PAD_L-PAD_R;
  const allMs=ills.flatMap(il=>il.dates);
  const tMin=Math.min(...allMs),tMax=Math.max(...allMs),span=tMax-tMin||1;
  function xPos(ms){{return allMs.length===1?PAD_L+trackW/2:PAD_L+((ms-tMin)/span)*trackW;}}
  const ILL_COLORS=['#10b981','#7c3aed','#ef4444','#f59e0b','#3b82f6','#ec4899'];
  ctx.font='9px Montserrat,sans-serif';
  const datePositions=[];
  ills.forEach((ill,ri)=>{{
    const y=PAD_T+ri*ROW_H+ROW_H/2; const col=ILL_COLORS[ri%ILL_COLORS.length];
    ctx.fillStyle=textC;ctx.textAlign='right';ctx.textBaseline='middle';
    const nm=ill.name.length>16?ill.name.slice(0,15)+'…':ill.name;
    ctx.fillText(nm,PAD_L-8,y);
    ctx.strokeStyle=gridC;ctx.lineWidth=0.5;ctx.setLineDash([3,3]);
    ctx.beginPath();ctx.moveTo(PAD_L,y);ctx.lineTo(W-PAD_R,y);ctx.stroke();
    ctx.setLineDash([]);
    if(ill.dates.length>1){{
      const xs=ill.dates.map(xPos);
      ctx.strokeStyle=col;ctx.globalAlpha=0.2;ctx.lineWidth=1.5;
      ctx.beginPath();ctx.moveTo(xs[0],y);xs.slice(1).forEach(x=>ctx.lineTo(x,y));ctx.stroke();
      ctx.globalAlpha=1;
    }}
    ill.dates.forEach((ms,di)=>{{
      const x=xPos(ms);
      ctx.fillStyle=col;ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);ctx.fill();
      if(di===0){{ctx.strokeStyle='rgba(255,255,255,0.8)';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);ctx.stroke();}}
      datePositions.push({{x,ms}});
    }});
  }});
  const merged=[];
  datePositions.forEach(dp=>{{if(!merged.find(m=>Math.abs(m.x-dp.x)<22))merged.push(dp);}});
  ctx.font='9px Montserrat,sans-serif';ctx.fillStyle=textC;ctx.textAlign='center';ctx.textBaseline='top';
  merged.forEach(dp=>{{ctx.fillText(new Intl.DateTimeFormat('en-GB',{{day:'numeric',month:'short'}}).format(new Date(dp.ms)),dp.x,H-PAD_B+6);}});
  ctx.strokeStyle=gridC;ctx.lineWidth=0.5;ctx.beginPath();ctx.moveTo(PAD_L,H-PAD_B+2);ctx.lineTo(W-PAD_R,H-PAD_B+2);ctx.stroke();
}}

// ── Radar helper ──────────────────────────────────────────────────────────
let vitChart=null,haeChart=null;
function drawRadar(canvasId,labels,data,refMin,refMax,existing){{
  const ctx=document.getElementById(canvasId).getContext('2d');
  if(existing)existing.destroy();
  const norm=data.map((v,i)=>{{
    if(v===null||v===undefined)return 0;
    const mid=(refMin[i]+refMax[i])/2,range=refMax[i]-refMin[i];
    return Math.max(0,Math.round(100-Math.min(100,Math.abs(v-mid)/(range*0.5)*100)));
  }});
  return new Chart(ctx,{{type:'radar',data:{{labels,datasets:[{{data:norm,backgroundColor:'rgba(16,185,129,0.15)',borderColor:'#10b981',borderWidth:1.5,pointBackgroundColor:'#10b981',pointRadius:3}}]}},options:{{responsive:false,scales:{{r:{{min:0,max:100,ticks:{{display:false}},grid:{{color:'rgba(0,0,0,0.07)'}},angleLines:{{color:'rgba(0,0,0,0.07)'}},pointLabels:{{color:'#9ca3af',font:{{size:9,family:'Montserrat,sans-serif'}}}}}}}},plugins:{{legend:{{display:false}}}}}}}});
}}

// ── Sparkline helper ──────────────────────────────────────────────────────
function drawSparkline(canvas,data,refMin,refMax){{
  const ctx=canvas.getContext('2d'); const w=canvas.width,h=canvas.height;
  ctx.clearRect(0,0,w,h);
  const valid=data.filter(v=>v!==null&&v!==undefined);
  if(valid.length<2)return;
  const mn=Math.min(...valid)*0.95,mx=Math.max(...valid)*1.05;
  const pts=valid.map((v,i)=>{{return {{x:i*(w/(valid.length-1)),y:h-(((v-mn)/(mx-mn))*h*0.8+h*0.1)}};}});
  const inR=valid.every(v=>v>=refMin&&v<=refMax);
  ctx.strokeStyle=inR?'#10b981':'#ef4444';ctx.lineWidth=1.5;ctx.lineJoin='round';ctx.lineCap='round';
  ctx.beginPath();pts.forEach((p,i)=>i===0?ctx.moveTo(p.x,p.y):ctx.lineTo(p.x,p.y));ctx.stroke();
  pts.forEach(p=>{{ctx.beginPath();ctx.arc(p.x,p.y,2.5,0,Math.PI*2);ctx.fillStyle=inR?'#10b981':'#ef4444';ctx.fill();}});
}}

// ── Escalations ───────────────────────────────────────────────────────────
const eg=document.getElementById('escGroup');
const ed=document.getElementById('escDet');
if(patient.escalations&&patient.escalations.length>0){{
  const b={{'0–15 days':[],'15–30 days':[],'> 30 days':[]}};
  patient.escalations.forEach(e=>{{if(b[e.bucket])b[e.bucket].push(e);}});
  const cls={{'0–15 days':'esc-hi','15–30 days':'esc-am','> 30 days':'esc-ok'}};
  Object.entries(b).forEach(([k,arr])=>{{
    if(arr.length){{const s=document.createElement('span');s.className='esc-pill '+cls[k];s.textContent=k+': '+arr.length;eg.appendChild(s);}}
  }});
  ed.innerHTML=patient.escalations.map(e=>`<div style="margin-bottom:3px;">· ${{e.dx}} — OP ${{e.opDate}} → IP ${{e.ipDate}} (${{e.gap}}d)</div>`).join('');
}}else{{
  const s=document.createElement('span');s.className='esc-pill esc-ok';s.textContent='No escalations recorded';eg.appendChild(s);
}}

// ── Vitals ────────────────────────────────────────────────────────────────
const vit=patient.vitals;
vitChart=drawRadar('vitRadar',['BP Sys','BP Dia','Heart rate','Blood sugar'],
  [vit.bp_sys.at(-1),vit.bp_dia.at(-1),vit.hr.at(-1),vit.sugar.at(-1)],
  [90,60,60,3.9],[120,80,100,5.6],vitChart);
const sg=document.getElementById('sparkGrid');sg.innerHTML='';
[{{name:'BP Systolic',unit:'mmHg',data:vit.bp_sys,min:90,max:120}},
 {{name:'BP Diastolic',unit:'mmHg',data:vit.bp_dia,min:60,max:80}},
 {{name:'Heart rate',unit:'bpm',data:vit.hr,min:60,max:100}},
 {{name:'Blood sugar',unit:'mmol/L',data:vit.sugar,min:3.9,max:5.6}}
].forEach(s=>{{
  const valid=s.data.filter(v=>v!==null);
  const last=valid.length?valid.at(-1):null;
  const inR=last!==null&&last>=s.min&&last<=s.max;
  const cid='sp_'+s.name.replace(/\s/g,'_');
  const card=document.createElement('div');card.className='spark-card';
  card.innerHTML=`<div class="spark-name">${{s.name}}</div><div class="spark-val">${{last!==null?last:'—'}}</div><div class="spark-unit">${{s.unit}}</div><div class="${{inR?'spark-ok':'spark-warn'}}">${{last===null?'No data':inR?'✓ In range':'⚠ Out of range'}}</div><canvas id="${{cid}}" width="90" height="32"></canvas>`;
  sg.appendChild(card);
  setTimeout(()=>{{const c=document.getElementById(cid);if(c)drawSparkline(c,s.data,s.min,s.max);}},120);
}});

// ── Haemogram ─────────────────────────────────────────────────────────────
const hae=patient.haemo;
const hLabels=['WBC','RBC','Hgb','Platelets','MCV','MCHC'];
const hData=[hae.wbc,hae.rbc,hae.hgb,hae.plt,hae.mcv,hae.mchc];
const hMin=[4.0,4.5,12.0,150,80,32],hMax=[11.0,5.5,16.0,400,100,36];
const hHasData=hData.some(v=>v!==null&&v!==undefined);
if(hHasData){{
  haeChart=drawRadar('haeRadar',hLabels,hData,hMin,hMax,haeChart);
  document.getElementById('labDetail').innerHTML=hLabels.map((l,i)=>{{
    const v=hData[i];if(v===null||v===undefined)return '';
    const ok=v>=hMin[i]&&v<=hMax[i];
    return `<div class="lab-row"><span style="color:#6b7280;">${{l}}</span><span class="${{ok?'lab-ok':'lab-warn'}}">${{v}} ${{ok?'✓':'!'}} </span></div>`;
  }}).join('');
}}else{{
  document.getElementById('haeRadar').parentElement.innerHTML='<div style="font-size:10px;color:#9ca3af;text-align:center;padding:24px 0;">No haemogram data recorded</div>';
  document.getElementById('labDetail').innerHTML='<div style="font-size:10px;color:#9ca3af;padding:24px 0;">Lab values not available</div>';
}}

// ── Medications ───────────────────────────────────────────────────────────
const ma=document.getElementById('medAlert');
if(patient.medChanges>0){{
  ma.style.display='flex';
  document.getElementById('medAlertTxt').textContent=patient.medChanges+' medication change'+(patient.medChanges>1?'s':'')+' detected — verify vitals stabilised after each switch.';
}}
const me=document.getElementById('medEvents');me.innerHTML='';
patient.meds.slice().reverse().forEach(m=>{{
  const div=document.createElement('div');div.className='med-ev';
  const ec=m.status==='active'?'ev-active':m.status==='stopped'?'ev-stopped':'ev-changed';
  const bc=m.status==='active'?'mb-a':m.status==='stopped'?'mb-s':'mb-c';
  const lbl=m.status==='active'?'Active':m.status==='stopped'?'Stopped':`Changed → ${{m.change}}`;
  div.innerHTML=`<div class="med-ev-dot ${{ec}}"></div><div class="med-name">${{m.name}}<span class="mbadge ${{bc}}">${{lbl}}</span></div><div class="med-meta">${{m.date}}</div>`;
  me.appendChild(div);
}});

// ── Draw canvases after layout ────────────────────────────────────────────
setTimeout(()=>{{drawVisitScatter();drawIllnessScatter();}},80);
</script></body></html>"""

    n_ill = len(obj.get("illnesses", []))
    card_height = 1500 + n_ill * 30
    _components.html(html, height=card_height, scrolling=True)

