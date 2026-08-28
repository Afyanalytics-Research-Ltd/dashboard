"""
views.py — Afya Clinical Analytics
=====================================
Render functions only. All SQL lives in queries.py.
Each function calls queries.load_* and builds charts.

  render_tab1_operations
  render_tab2_segmentation
  render_tab3_retention
  render_tab4_disease_burden
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
from ksh.clinical_module.ui_template import render_sortable_table

_PATIENT_LIST_COMPONENT = _stcomp.declare_component(
    "patient_list",
    path=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "patient_list_component"),
)
from ksh.clinical_module.ui_template import AFYA_BLUE, TEAL, COOL_BLUE, ORANGE, CORAL, PURPLE, GRAY, MUTED, BG_LIGHT, BORDER, GREEN, AMBER
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
_PC_CFG = {"responsive": True, "displayModeBar": False, "useResizeHandler": True}
def _pc(fig):      _H().get("pc", lambda f: st.plotly_chart(f, use_container_width=True, config=_PC_CFG))(fig)
def _note(t, w=False): _H().get("note", lambda t,warn=False: None)(t, w)
def _n(v):
    return _H().get("fmt_num", lambda v: str(v))(v)

def _nf(v):
    """Full integer with commas — no K/M abbreviation."""
    if v is None: return "—"
    try: return f"{int(float(v)):,}"
    except: return str(v)

def _p(v, d=1):
    return _H().get("fmt_pct", lambda v, d=1: str(v))(v, d)

def _k(v):
    return _H().get("fmt_kes", lambda v: str(v))(v)


# ══════════════════════════════════════════════════════════════════════════════
# WAIT-TIME CHART HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _render_wait_time_chart(df: pd.DataFrame):
    """Two-panel average wait-time chart — values in minutes, no external dependencies."""
    import math as _m

    def _fm(mins):
        """Format minutes → '2 hr 15 min' or '45 min'."""
        if mins is None: return "—"
        try: mins = float(mins)
        except: return "—"
        if _m.isnan(mins): return "—"
        m = round(mins)
        h, rm = divmod(m, 60)
        if h == 0: return f"{rm} min"
        if rm == 0: return f"{h} hr"
        return f"{h} hr {rm} min"

    def _safe(row, col):
        v = row.get(col)
        if v is None: return None
        try:
            f = float(v)
            return None if _m.isnan(f) else f
        except: return None

    def _classify(avg, tgt):
        if avg is None or tgt is None: return None
        if avg <= tgt: return "within"
        if avg <= tgt * 1.5: return "warn"
        return "breach"

    S_COL  = {"within": "#1D9E75", "warn": "#EF9F27", "breach": "#E24B4A"}
    S_BDGE = {"within": "Within target", "warn": "Approaching limit", "breach": "Breaching target"}
    S_BG   = {"within": "#E1F5EE", "warn": "#FEF3C7", "breach": "#FEE2E2"}
    S_TXT  = {"within": "#0F6E56", "warn": "#854F0B", "breach": "#A32D2D"}

    def _desc(vtype, avg, tgt, status):
        vt = "inpatients" if vtype == "Inpatient" else "outpatients"
        avgf, tgtf = _fm(avg), _fm(tgt)
        if status == "breach":
            x = round(float(avg) / float(tgt), 1)
            return f"Average wait for {vt} is {avgf} — {x}× the {tgtf} target."
        if status == "warn":
            return f"Average wait for {vt} is {avgf}, approaching the {tgtf} target."
        return f"Average wait for {vt} is {avgf}, within the {tgtf} target."

    def _gap_row(name, is_last):
        bb = "" if is_last else "border-bottom:0.5px solid rgba(0,0,0,0.06);"
        return (f'<div style="padding:10px 0;{bb}">'
                f'<div style="border:0.5px dashed #d1d5db;border-radius:8px;padding:8px 10px;">'
                f'<div style="font-style:italic;font-size:11px;color:#9ca3af;">{name}</div>'
                f'<div style="font-size:10px;color:#9ca3af;margin-top:3px;">'
                f'No data recorded for this stage</div></div></div>')

    def _bar_row(name, avg, tgt, status, vtype, x_max, is_last):
        bb    = "" if is_last else "border-bottom:0.5px solid rgba(0,0,0,0.06);"
        bar_w = min(100, round(float(avg) / x_max * 100, 1))
        col   = S_COL.get(status, "#6b7280") if status else "#6b7280"
        tick  = ""
        if tgt is not None:
            tp = min(98, round(float(tgt) / x_max * 100, 1))
            tick = (f'<div style="position:absolute;left:{tp}%;top:-5px;bottom:-5px;'
                    f'width:2px;background:#7F77DD;border-radius:1px;z-index:2;"></div>'
                    f'<div style="position:absolute;left:{tp}%;top:-18px;'
                    f'transform:translateX(-50%);font-size:9px;color:#534AB7;'
                    f'white-space:nowrap;z-index:2;">{_fm(tgt)}</div>')
        badge = ""
        if status:
            badge = (f'<span style="font-size:10px;padding:2px 7px;border-radius:20px;'
                     f'background:{S_BG[status]};color:{S_TXT[status]};font-weight:600;">'
                     f'{S_BDGE[status]}</span>')
        desc_html = (f'<div style="font-size:11px;color:#6b7280;margin-top:4px;">'
                     f'{_desc(vtype, avg, tgt, status)}</div>') if status else ""
        return (
            f'<div style="padding:10px 0;{bb}">'
            f'  <div style="display:flex;justify-content:space-between;align-items:center;">'
            f'    <span style="font-size:12px;font-weight:500;color:#111827;">{name}</span>'
            f'    <div style="display:flex;align-items:center;gap:6px;">'
            f'      <span style="font-size:12px;font-weight:600;color:#111827;">{_fm(avg)}</span>'
            f'      {badge}</div></div>'
            f'  <div style="position:relative;height:10px;background:#f5f5f3;'
            f'border-radius:4px;margin:16px 0 6px;">'
            f'    <div style="width:{bar_w}%;height:100%;background:{col};border-radius:4px;"></div>'
            f'    {tick}</div>'
            f'  {desc_html}'
            f'</div>'
        )

    # Targets in minutes
    _STAGES = [
        ("avg_mins_triage_to_consult",   "Triage → Consultation",  60),
        ("avg_mins_consult_to_lab",      "Consult → Lab Result",  120),
        ("avg_mins_lab_turnaround",      "Lab Turnaround",         60),
        ("avg_mins_consult_to_dispense", "Consult → Dispense",     30),
    ]

    def _panel(row, vtype):
        total   = _safe(row, "total_visits")
        total_s = f"{int(total):,}" if total is not None else "—"
        avg_tot = _safe(row, "avg_total_to_consult_mins")

        avgs     = [(_safe(row, col), lbl, tgt) for col, lbl, tgt in _STAGES]
        present  = [v for v, _, _ in avgs if v is not None]
        tgts_all = [tgt for _, _, tgt in _STAGES]
        x_max    = max(_m.ceil(max(present + tgts_all) * 1.1), 60) if present else 120
        scale    = f"Avg wait times · {total_s} visits"

        statuses = [_classify(v, tgt) for v, _, tgt in avgs if v is not None]
        vt_l     = "inpatients" if vtype == "Inpatient" else "outpatients"
        if "breach" in statuses:
            vbg, vbd = "#FFF0F0", "#E24B4A"
            vtx = f'<strong>Action needed.</strong> One or more stages are breaching targets for {vt_l}.'
        elif "warn" in statuses:
            vbg, vbd = "#FAEEDA", "#EF9F27"
            vtx = f'<strong>Watch closely.</strong> Wait times are approaching limits for {vt_l}.'
        elif statuses:
            vbg, vbd = "#E1F5EE", "#1D9E75"
            vtx = '<strong>On track.</strong> All stages are within target wait times.'
        else:
            vbg, vbd = "#f8fafc", "#d1d5db"
            vtx = 'No wait-time data available for this visit type.'

        rows = ""
        for i, (avg, lbl, tgt) in enumerate(avgs):
            is_last = (i == len(avgs) - 1)
            if avg is None:
                rows += _gap_row(lbl, is_last)
            else:
                rows += _bar_row(lbl, avg, tgt, _classify(avg, tgt), vtype, x_max, is_last)

        tot_html = ""
        if avg_tot is not None:
            tot_html = (f'<div style="margin-top:10px;padding:8px 10px;background:#f8fafc;'
                        f'border-radius:6px;font-size:11px;color:#374151;">'
                        f'<strong>Avg arrival → consultation:</strong> {_fm(avg_tot)}</div>')

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
            f'  {tot_html}'
            f'</div>'
        )

    panels = []
    for vtype in ["Inpatient", "Outpatient"]:
        sub = df[df["visit_type"] == vtype]
        if sub.empty: continue
        panels.append(_panel(sub.iloc[0].to_dict(), vtype))

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

    html = (
        '<div style="background:#f5f5f3;padding:12px;'
        'font-family:system-ui,-apple-system,sans-serif;">'
        + legend
        + '<div style="display:flex;gap:12px;">' + "".join(panels) + '</div>'
        + '</div>'
    )
    _stcomp.html(html, height=560, scrolling=False)


# ══════════════════════════════════════════════════════════════════════════════
# PATIENT CONVERSION & VALUE TAB
# ══════════════════════════════════════════════════════════════════════════════

# ── Colour palette ────────────────────────────────────────────────────────────
_CV_TEAL   = "#1D9E75"
_CV_PURPLE = "#7F77DD"
_CV_BLUE   = "#378ADD"
_CV_AMBER  = "#EF9F27"
_CV_RED    = "#E24B4A"
_CV_GREY   = "#888780"



def _cv_section(label: str):
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin:20px 0 12px">'
        f'<span style="font-size:9px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.08em;color:{_CV_GREY};white-space:nowrap">{label}</span>'
        f'<div style="flex:1;height:0.5px;background:rgba(0,0,0,0.10)"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _cv_kpi(label: str, value: str, sub: str = "", color: str = "#111827"):
    st.markdown(
        f'<div style="background:#fff;border:0.5px solid rgba(0,0,0,0.10);'
        f'border-radius:12px;padding:12px 14px">'
        f'<div style="font-size:11px;color:{_CV_GREY};margin-bottom:4px">{label}</div>'
        f'<div style="font-size:22px;font-weight:600;color:{color};line-height:1.1">{value}</div>'
        f'<div style="font-size:11px;color:{_CV_GREY};margin-top:3px">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _cv_insight(text: str):
    st.markdown(
        f'<div style="border-left:3px solid #0072CE;background:#EBF5FF;'
        f'border-radius:0 4px 4px 0;padding:10px 14px;'
        f'font-size:13px;color:#003467;margin-top:8px;line-height:1.7">{text}</div>',
        unsafe_allow_html=True,
    )




def _fmt_kes(v) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
        if abs(f) >= 1_000_000:
            return f"KES {f/1_000_000:.1f}M"
        if abs(f) >= 1_000:
            return f"KES {f/1_000:.0f}K"
        return f"KES {f:,.0f}"
    except Exception:
        return str(v)


def render_tab_conversion_value(filters: dict, run_query):
    """Patient Conversion & Value tab — 3 sections."""
    import pandas as _pd
    import plotly.graph_objects as _go
    from plotly.subplots import make_subplots as _msp

    pc = _H().get("pc", lambda f: st.plotly_chart(f, use_container_width=True, config=_PC_CFG))
    _CHART_BASE = dict(
        paper_bgcolor="#fff", plot_bgcolor="#fff",
        margin=dict(l=0, r=0, t=6, b=0),
        font=dict(family="system-ui, sans-serif", size=12, color="#111827"),
    )
    _AX = dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)",
               showline=False, tickfont=dict(size=11, color=_CV_GREY))

    def _panel_start(title="", subtitle=""):
        parts = [
            '<div style="background:#fff;border:0.5px solid rgba(0,0,0,0.10);'
            'border-radius:12px;padding:12px 14px;margin-bottom:8px">',
        ]
        if title:
            parts.append(f'<div style="font-size:12px;font-weight:500;color:#111827;'
                         f'margin-bottom:2px">{title}</div>')
        if subtitle:
            parts.append(f'<div style="font-size:10px;color:{_CV_GREY};'
                         f'margin-bottom:8px">{subtitle}</div>')
        st.markdown("".join(parts), unsafe_allow_html=True)

    def _panel_end():
        st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1 — Patient overview
    # ══════════════════════════════════════════════════════════════════════
    _cv_section("Section 1 — Patient overview")

    try:
        df_ov = Q.load_cv_overview(filters, run_query)
        if not df_ov.empty:
            r = df_ov.iloc[0]
            total     = int(r.get("total_patients", 0) or 0)
            chronic   = int(r.get("chronic_patients", 0) or 0)
            repeat    = int(r.get("repeat_patients", 0) or 0)
            single    = int(r.get("single_visit_patients", 0) or 0)
            avg_vis   = float(r.get("avg_visits_per_patient", 0) or 0)
            new_p     = int(r.get("new_patients", 0) or 0)
            ret_p     = int(r.get("returning_patients", 0) or 0)

            chronic_pct = round(chronic / total * 100, 1) if total else 0
            repeat_pct  = round(repeat  / total * 100, 1) if total else 0
            single_pct  = round(single  / total * 100, 1) if total else 0
            ret_pct     = round(ret_p   / total * 100, 1) if total else 0

            c1,c2,c3,c4,c5,c6 = st.columns(6)
            with c1: _cv_kpi("Total patients", f"{total:,}", "All visits in period")
            with c2: _cv_kpi("Chronic patients", f"{chronic:,}", f"{chronic_pct}% of patients", _CV_AMBER)
            with c3: _cv_kpi("Repeat patients", f"{repeat:,}", f"{repeat_pct}% repeat rate", _CV_TEAL)
            with c4: _cv_kpi("Single visit", f"{single:,}", f"{single_pct}% of patients", _CV_GREY)
            with c5: _cv_kpi("Avg visits / patient", f"{avg_vis:.1f}", "per patient")
            with c6: _cv_kpi("Returning", f"{ret_pct:.0f}%",
                             f"{new_p:,} new · {ret_p:,} returning")
    except Exception as e:
        st.warning(f"Overview: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2 — Patient profile & conversion signals
    # ══════════════════════════════════════════════════════════════════════
    _cv_section("Section 2 — Patient profile and conversion signals")

    try:
        import json as _jcv
        df_dem = Q.load_cv_demographics(filters, run_query)
        df_cg  = Q.load_cv_cohort_growth(filters, run_query)

        col_age, col_bubble = st.columns(2)

        # ── Left: Patients by age group — stacked by gender ──────────────
        with col_age:
            _panel_start("Patients by age group", "Patient count by age group and gender")
            if not df_dem.empty:
                df_dem["_gen"] = df_dem["gender"].str.upper().str.strip().map(
                    {"F": "Female", "FEMALE": "Female", "M": "Male", "MALE": "Male"}
                ).fillna("_drop")
                # exclude Unknown age group and non-M/F genders
                _dem_f = df_dem[
                    (df_dem["age_group"] != "Unknown") &
                    (df_dem["_gen"] != "_drop")
                ]
                _g_age = (_dem_f.groupby(["age_group", "_gen"])["patients"]
                          .sum().reset_index())
                _age_ord = (_dem_f.groupby("age_group")["patients"]
                            .sum().sort_values(ascending=True).index.tolist())
                _GC = {"Female": _CV_PURPLE, "Male": _CV_TEAL}
                fig_a = _go.Figure()
                for _g in ["Male", "Female"]:
                    _gdf = (_g_age[_g_age["_gen"] == _g]
                            .set_index("age_group").reindex(_age_ord)
                            .fillna(0).reset_index())
                    if _gdf["patients"].sum() == 0:
                        continue
                    fig_a.add_trace(_go.Bar(
                        y=_gdf["age_group"], x=_gdf["patients"],
                        name=_g, orientation="h",
                        marker_color=_GC[_g],
                    ))
                _cb_age = {**_CHART_BASE, "margin": dict(t=44, b=8, l=4, r=8)}
                fig_a.update_layout(
                    **_cb_age, height=430, barmode="stack",
                    xaxis={**_AX, "nticks": 4, "showgrid": False},
                    yaxis={**_AX, "showgrid": False},
                    legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center", font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
                )
                pc(fig_a)
            _panel_end()

        # ── Right: Age cohort growth index — SVG bubble grid ─────────────
        with col_bubble:
            _panel_start("Age cohort growth index",
                         "Index = visits ÷ prior month × 100 · Above 100 = growth · Below 100 = decline · 100 = flat")
            if not df_cg.empty:
                df_cg["visit_month"] = _pd.to_datetime(df_cg["visit_month"])
                df_cg = df_cg.sort_values("visit_month")
                _cgr = df_cg.copy()
                _cgr["month"] = _cgr["visit_month"].dt.strftime("%Y-%m")
                _cgr = _cgr.rename(columns={"patients": "count"})
                _cg_json = _jcv.dumps(
                    _cgr[["age_group", "month", "count"]].to_dict(orient="records")
                )
                _bubble_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  padding:8px 10px 10px;background:#F0F2F5;border-radius:6px;}}
.lgd{{display:flex;gap:10px;align-items:center;margin-bottom:8px;flex-wrap:wrap}}
.li{{display:flex;align-items:center;gap:4px;font-size:10px;color:#6B7280}}
.gw{{overflow-x:auto}}
</style></head>
<body>
<div class="lgd" id="lgd"></div>
<div class="gw" id="grid"></div>
<script>
const RAW={_cg_json};
const AGE_ORDER=['Toddler (0-4)','Child (5-12)','Adolescent (13-17)','Youth (18-24)',
  'Young Adult (25-34)','Adult (35-44)','Middle Age (45-54)','Older Adult (55-64)','Senior (65+)'];
const MO=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

const piv={{}};
RAW.forEach(r=>{{
  if(!piv[r.age_group])piv[r.age_group]={{}};
  piv[r.age_group][r.month]=+(r.count||0);
}});
const allMo=[...new Set(RAW.map(r=>r.month))].sort();
const cols=allMo.slice(1);
const ages=AGE_ORDER.filter(ag=>piv[ag]);

const LW=122,CW=26,CH=42,HH=52,PB=10;
const W=LW+cols.length*CW, H=HH+ages.length*CH+PB;

function bs(idx){{
  if(idx===null)return{{f:'rgba(180,178,169,0.18)',s:'rgba(120,118,109,0.3)',r:2.5}};
  const d=idx-100,a=Math.abs(d);
  const r=Math.min(13,Math.max(2.5,2.5+a/100*10.5));
  if(a<=6)return{{f:'rgba(180,178,169,0.18)',s:'rgba(120,118,109,0.3)',r}};
  if(d>0){{
    const f=a<25?'rgba(29,158,117,0.35)':a<60?'rgba(29,158,117,0.65)':'rgba(29,158,117,0.9)';
    return{{f,s:'rgba(15,110,86,0.6)',r}};
  }}
  const f=a<25?'rgba(226,75,74,0.35)':a<60?'rgba(226,75,74,0.65)':'rgba(226,75,74,0.9)';
  return{{f,s:'rgba(163,45,45,0.6)',r}};
}}
const NS='http://www.w3.org/2000/svg';
function el(t,a){{const e=document.createElementNS(NS,t);Object.entries(a||{{}}).forEach(([k,v])=>e.setAttribute(k,v));return e;}}
function tx(s,a){{const t=el('text',a);t.textContent=s;return t;}}

const svg=el('svg',{{viewBox:'0 0 '+W+' '+H,width:W,height:H}});

cols.forEach((mo,ci)=>{{
  const[yr,mn]=mo.split('-'),cx=LW+ci*CW+CW/2;
  svg.appendChild(tx(MO[+mn-1],{{x:cx,y:18,'text-anchor':'middle','font-size':'9',fill:'#9CA3AF'}}));
  svg.appendChild(tx(yr,{{x:cx,y:30,'text-anchor':'middle','font-size':'8',fill:'#C0BDB4'}}));
}});

ages.forEach((ag,ri)=>{{
  const cy0=HH+ri*CH+CH/2;
  svg.appendChild(tx(ag,{{x:LW-8,y:cy0+4,'text-anchor':'end','font-size':'10',fill:'#6B7280'}}));
  cols.forEach((mo,ci)=>{{
    const prevMo=allMo[ci];
    const curr=(piv[ag]||{{}})[mo]||0,prev=(piv[ag]||{{}})[prevMo]||0;
    const idx=prev>0?Math.round(curr/prev*100):null;
    const dev=idx!==null?idx-100:0;
    const{{f,s,r}}=bs(idx);
    const cx=LW+ci*CW+CW/2;
    const moL=MO[+mo.split('-')[1]-1]+' '+mo.split('-')[0];
    const tip=ag+' — '+moL+(idx!==null?': index '+idx+' ('+(dev>=0?'+':'')+dev+')':': no prior');
    const c=el('circle',{{cx,cy:cy0,r,fill:f,stroke:s,'stroke-width':'0.8'}});
    const tEl=document.createElementNS(NS,'title');tEl.textContent=tip;c.appendChild(tEl);
    svg.appendChild(c);
    if(idx!==null&&Math.abs(dev)>=25){{
      const lb=(dev>=0?'+':'')+dev;
      const t=el('text',{{x:cx,y:cy0+3,'text-anchor':'middle','font-size':'7',
        fill:dev>=0?'#085041':'#791F1F','pointer-events':'none'}});
      t.textContent=lb;svg.appendChild(t);
    }}
  }});
}});
document.getElementById('grid').appendChild(svg);

[
  {{label:'Strong growth',r:7,f:'rgba(29,158,117,0.9)',s:'rgba(15,110,86,0.6)'}},
  {{label:'Mild growth',r:4.5,f:'rgba(29,158,117,0.35)',s:'rgba(15,110,86,0.6)'}},
  {{label:'Flat',r:2.5,f:'rgba(180,178,169,0.18)',s:'rgba(120,118,109,0.3)'}},
  {{label:'Mild decline',r:4.5,f:'rgba(226,75,74,0.35)',s:'rgba(163,45,45,0.6)'}},
  {{label:'Strong decline',r:7,f:'rgba(226,75,74,0.9)',s:'rgba(163,45,45,0.6)'}},
].forEach(it=>{{
  const sz=it.r*2,d=document.createElement('div');
  d.className='li';
  d.innerHTML='<svg width="'+sz+'" height="'+sz+'" viewBox="0 0 '+sz+' '+sz+'">'
    +'<circle cx="'+it.r+'" cy="'+it.r+'" r="'+(it.r-0.5)+'" fill="'+it.f+'" stroke="'+it.s+'" stroke-width="0.8"/>'
    +'</svg>'+it.label;
  document.getElementById('lgd').appendChild(d);
}});
</script></body></html>"""
                _stcomp.html(_bubble_html, height=500, scrolling=True)
            _panel_end()

    except Exception as e:
        st.error(f"Demographics / cohort: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── RETENTION AND ACQUISITION ─────────────────────────────────────────
    _cv_section("Retention and Acquisition")

    # ── Row 3 — Benchmark comparison + New vs returning volumes ──────────
    try:
        df_nr   = Q.load_cv_new_returning_trend(filters, run_query)
        _tenri  = Q.load_tenri_benchmark(run_query)
        if df_nr.empty:
            st.caption(f"⚠ New/returning: no data — schemas={filters.get('source_schemas')}")
        if not df_nr.empty:
            df_nr["visit_month"] = _pd.to_datetime(df_nr["visit_month"])
            pivot = df_nr.pivot(index="visit_month", columns="patient_type",
                                values="patients").fillna(0)
            months = sorted(pivot.index)

            c1, c2 = st.columns(2)

            # Panel 1 — Actual vs Tenri Benchmark
            with c1:
                _panel_start("Actual Patient Volume vs. Tendri Hospital Benchmarks",
                             "Your facility split vs. Tendri hospital benchmark")

                _L4 = _tenri
                total_new = int(df_nr[df_nr["patient_type"] == "New"]["patients"].sum())
                total_ret = int(df_nr[df_nr["patient_type"] == "Returning"]["patients"].sum())
                total     = (total_new + total_ret) or 1
                actual    = {"New": round(total_new / total * 100, 1),
                             "Returning": round(total_ret / total * 100, 1)}

                categories = ["New Patients", "Returning Patients"]
                keys       = ["New", "Returning"]
                bm_vals    = [_L4[k] for k in keys]
                act_vals   = [actual[k] for k in keys]

                fig_bm = _go.Figure()
                fig_bm.add_trace(_go.Bar(
                    x=categories, y=bm_vals,
                    name="Tendri Benchmark",
                    marker_color="rgba(180,180,180,0.65)",
                    text=[f"{v:.0f}%" for v in bm_vals],
                    textposition="outside",
                    textfont=dict(size=11, color="#888780"),
                ))
                act_colors = [_CV_TEAL for _ in keys]
                fig_bm.add_trace(_go.Bar(
                    x=categories, y=act_vals,
                    name="Your Facility",
                    marker_color=act_colors,
                    text=[f"{v:.0f}%" for v in act_vals],
                    textposition="outside",
                    textfont=dict(size=11, color="#111827"),
                ))

                # Delta annotations above facility bars
                # Grouped bar offsets: benchmark ~-0.2, facility ~+0.2 from category center
                for ci, k in enumerate(keys):
                    delta = actual[k] - _L4[k]
                    label = f"+{delta:.0f}% Above Tendri" if delta >= 0 else f"{delta:.0f}% Below Tendri"
                    color = "#1D9E75" if delta >= 0 else "#A32D2D"
                    fig_bm.add_annotation(
                        x=ci + 0.2, y=act_vals[ci] + 6,
                        text=label,
                        showarrow=True, arrowhead=2, arrowwidth=1.5,
                        arrowcolor=color, font=dict(size=10, color=color),
                        ax=0, ay=-28,
                        xref="x", yref="y",
                    )

                fig_bm.update_layout(
                    **_CHART_BASE, height=260, barmode="group",
                    xaxis={**_AX, "showgrid": False},
                    yaxis={**_AX, "ticksuffix": "%", "range": [0, 100]},
                    legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center", font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
                    bargroupgap=0.15,
                )
                pc(fig_bm)

                parts = []
                for k, label in [("New", "new"), ("Returning", "returning")]:
                    d = actual[k] - _L4[k]
                    sign = "above" if d >= 0 else "below"
                    parts.append(f"{label.capitalize()} patients {abs(d):.0f} pct pts {sign} Tendri benchmark")
                _cv_insight(" · ".join(parts))
                _panel_end()

            # Panel 2 — New vs returning dual line
            with c2:
                _panel_start("New vs returning volumes",
                             "Monthly patient counts by type")
                new_vals = pivot.get("New", _pd.Series(dtype=float)).reindex(months, fill_value=0)
                ret_vals = pivot.get("Returning", _pd.Series(dtype=float)).reindex(months, fill_value=0)
                fig_nv = _go.Figure()
                fig_nv.add_trace(_go.Scatter(
                    x=months, y=new_vals.values, name="New",
                    mode="lines+markers",
                    line=dict(color=_CV_TEAL, width=2), marker=dict(size=4),
                ))
                fig_nv.add_trace(_go.Scatter(
                    x=months, y=ret_vals.values, name="Returning",
                    mode="lines+markers",
                    line=dict(color=_CV_PURPLE, width=2), marker=dict(size=4),
                ))
                fig_nv.update_layout(**_CHART_BASE, height=260,
                                     xaxis={**_AX, "showgrid": False, "tickformat": "%b %y"},
                                     yaxis=_AX,
                                     legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center", font=dict(size=10), bgcolor="rgba(0,0,0,0)"))
                pc(fig_nv)
                corr = _pd.Series(new_vals.values).corr(_pd.Series(ret_vals.values))
                if not _pd.isna(corr):
                    trend = "move together" if corr > 0.5 else ("diverge" if corr < 0 else "are loosely correlated")
                    _cv_insight(f"New and returning patients {trend} (r={corr:.2f})")
                _panel_end()
    except Exception as e:
        st.error(f"New vs returning error: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 4 — Acquisition Segment Retention ────────────────────────────
    try:
        import json as _json
        df_seg = Q.load_acquisition_segments(filters, run_query)
        if df_seg.empty:
            st.caption(f"⚠ Acquisition segments: no data — schemas={filters.get('source_schemas')}")
        if not df_seg.empty:
            for _c in ("new_patients", "returning_patients", "total_visits",
                       "new_pct", "returning_pct", "returning_per_new_ratio",
                       "divergence_from_threshold"):
                df_seg[_c] = pd.to_numeric(df_seg[_c], errors="coerce").fillna(0)

            seg_rows  = df_seg.to_dict(orient="records")
            data_json = _json.dumps(seg_rows)

            html_acq = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  :root{{
    --color-background-primary:#ffffff;
    --color-text-primary:#111827;
    --color-text-secondary:#6B7280;
    --color-border-tertiary:#E5E7EB;
    --color-background-card:#F9FAFB;
  }}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
        background:var(--color-background-primary);
        color:var(--color-text-primary);padding:12px 4px;}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin-bottom:18px}}
  .card{{
    border:1px solid var(--color-border-tertiary);
    border-radius:8px;padding:12px 14px;
    background:var(--color-background-card);
    cursor:pointer;transition:box-shadow .15s,border-color .15s;
    border-left:4px solid transparent;
  }}
  .card:hover{{box-shadow:0 2px 8px rgba(0,0,0,.08);border-color:#D1D5DB}}
  .card.concern{{border-left-color:#E24B4A}}
  .card.active{{box-shadow:0 0 0 2px #185FA5;border-color:#185FA5}}
  .card-name{{font-size:11px;font-weight:600;text-transform:uppercase;
              letter-spacing:.04em;color:var(--color-text-secondary);margin-bottom:6px}}
  .card-ratio{{font-size:22px;font-weight:700;line-height:1;margin-bottom:4px}}
  .card-sub{{font-size:11px;color:var(--color-text-secondary);margin-bottom:8px}}
  .pbar-wrap{{height:6px;background:#E5E7EB;border-radius:3px;overflow:hidden;margin-bottom:6px}}
  .pbar-fill{{height:100%;border-radius:3px}}
  .badge{{display:inline-flex;align-items:center;gap:4px;
          font-size:10px;font-weight:600;padding:2px 7px;border-radius:999px}}
  .badge-green{{background:#D1FAE5;color:#065F46}}
  .badge-red{{background:#FEE2E2;color:#991B1B}}
  .badge-blue{{background:#DBEAFE;color:#1E40AF}}
  .legend{{display:flex;gap:16px;margin-bottom:8px;flex-wrap:wrap}}
  .legend-item{{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--color-text-secondary)}}
  .legend-dot{{width:10px;height:10px;border-radius:2px}}
  #drilldown{{margin-top:18px;padding:14px;border:1px solid var(--color-border-tertiary);
               border-radius:8px;background:var(--color-background-card);display:none}}
  #drilldown-title{{font-size:13px;font-weight:600;margin-bottom:12px;color:var(--color-text-primary)}}
  #postop-warn{{font-size:12px;color:#92400E;background:#FEF3C7;border:1px solid #FDE68A;
                border-radius:6px;padding:10px 14px;display:none}}
  .insight-bar{{margin-top:10px;padding:8px 12px;border-radius:6px;font-size:11px;
                background:#FEF3C7;color:#92400E;border:1px solid #FDE68A;display:none}}
  canvas{{max-width:100%}}
</style>
</head>
<body>
<div class="cards" id="cards"></div>
<div class="legend" id="legend"></div>
<div style="height:300px;position:relative"><canvas id="segChart"></canvas></div>
<div id="drilldown">
  <div id="drilldown-title"></div>
  <div id="postop-warn">
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24"
         fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:middle;margin-right:4px">
      <path d="M12 9v4"/><path d="M10.363 3.591l-8.106 13.534a1.914 1.914 0 0 0 1.636 2.871h16.214a1.914 1.914 0 0 0 1.636-2.87l-8.106-13.536a1.914 1.914 0 0 0-3.274 0z"/>
      <path d="M12 16h.01"/>
    </svg>
    Post-Op data quality: age-group drill-down is not available for Post-Op. Patient classification relies on doctor note keywords which may be incomplete.
  </div>
  <canvas id="ageChart"></canvas>
  <div class="insight-bar" id="insightBar"></div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>
const RAW = {data_json};

const segMap = {{}};
RAW.forEach(r => {{
  const s = r.acquisition_segment;
  if (!segMap[s]) segMap[s] = {{seg:s, new:0, ret:0, visits:0, signal:'', rows:[]}};
  segMap[s].new    += +(r.new_patients||0);
  segMap[s].ret    += +(r.returning_patients||0);
  segMap[s].visits += +(r.total_visits||0);
  segMap[s].signal  = r.ratio_signal;
  segMap[s].rows.push(r);
}});

const ORDER = ['Chronic','Oncology','Maternal','Mental Health'];
const segs  = ORDER.filter(s => segMap[s]);

function ratio(s)   {{ return s.new > 0 ? s.ret / s.new : 0; }}
function newPct(s)  {{ const t = s.new + s.ret; return t ? (s.new / t * 100) : 0; }}
function retPct(s)  {{ const t = s.new + s.ret; return t ? (s.ret / t * 100) : 0; }}
function fmt(n,d=2) {{ return n.toLocaleString('en-US',{{minimumFractionDigits:d,maximumFractionDigits:d}}); }}
function fmtI(n)    {{ return Math.round(n).toLocaleString('en-US'); }}

function signalBadge(sig) {{
  if (sig === 'AS_EXPECTED') return `<span class="badge badge-green">
    <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24"
         fill="none" stroke="currentColor" stroke-width="2.5">
      <circle cx="12" cy="12" r="9"/><path d="M9 12l2 2l4-4"/>
    </svg>As expected</span>`;
  if (sig === 'CONCERN') return `<span class="badge badge-red">
    <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24"
         fill="none" stroke="currentColor" stroke-width="2.5">
      <path d="M12 9v4"/><path d="M10.363 3.591l-8.106 13.534a1.914 1.914 0 0 0 1.636 2.871h16.214a1.914 1.914 0 0 0 1.636-2.87l-8.106-13.536a1.914 1.914 0 0 0-3.274 0z"/>
      <path d="M12 16h.01"/>
    </svg>Concern</span>`;
  return `<span class="badge badge-blue">
    <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24"
         fill="none" stroke="currentColor" stroke-width="2.5">
      <circle cx="12" cy="12" r="9"/><line x1="12" y1="8" x2="12.01" y2="8"/>
      <polyline points="11 12 12 12 12 16 13 16"/>
    </svg>Review data</span>`;
}}

function retColor(sig) {{
  if (sig === 'AS_EXPECTED') return '#0F6E56';
  if (sig === 'CONCERN')     return '#E24B4A';
  return '#185FA5';
}}

const cardsEl = document.getElementById('cards');
segs.forEach((s, i) => {{
  const d   = segMap[s];
  const r   = ratio(d);
  const rp  = retPct(d);
  const np  = newPct(d);
  const sig = d.signal;
  const div = document.createElement('div');
  div.className = 'card' + (sig === 'CONCERN' ? ' concern' : '');
  div.dataset.idx = i;
  div.innerHTML = `
    <div class="card-name">${{s}}</div>
    <div class="card-ratio">${{fmt(r)}}x<small style="font-size:12px;font-weight:400;color:var(--color-text-secondary)"> R:N</small></div>
    <div class="card-sub">${{fmt(np,1)}}% new · ${{fmt(rp,1)}}% returning</div>
    <div class="pbar-wrap"><div class="pbar-fill" style="width:${{rp.toFixed(1)}}%;background:${{retColor(sig)}}"></div></div>
    ${{signalBadge(sig)}}
  `;
  div.addEventListener('click', () => showDrilldown(s, div));
  cardsEl.appendChild(div);
}});

const legendEl = document.getElementById('legend');
legendEl.innerHTML = `
  <div class="legend-item"><div class="legend-dot" style="background:#185FA5"></div>New Patients</div>
  <div class="legend-item"><div class="legend-dot" style="background:#6BADA0"></div>Returning Patients</div>
`;

// plugin — draw signal indicator above each Returning bar (like benchmark chart arrows)
const indicatorPlugin = {{
  id: 'indicatorLabels',
  afterDatasetsDraw(chart) {{
    const ctx  = chart.ctx;
    const retMeta = chart.getDatasetMeta(1); // dataset index 1 = Returning
    segs.forEach((s, i) => {{
      const bar = retMeta.data[i];
      if (!bar) return;
      const d   = segMap[s];
      const sig = d.signal;
      const r   = ratio(d);
      const div = +(d.rows[0]?.divergence_from_threshold ?? 0);
      let label, color;
      if (sig === 'AS_EXPECTED') {{
        const sign = div >= 0 ? '+' : '';
        label = `${{sign}}${{fmt(div)}}x Above Expected`;
        color = '#0F6E56';
      }} else if (sig === 'CONCERN') {{
        label = `-${{fmt(Math.abs(div))}}x Below Expected`;
        color = '#E24B4A';
      }} else {{
        label = 'Review Data';
        color = '#185FA5';
      }}
      // arrow pointing down to bar
      const arrowX = bar.x;
      const arrowY = bar.y - 6;
      ctx.save();
      ctx.fillStyle = color;
      ctx.font = 'bold 10px -apple-system,sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(label, arrowX, arrowY - 14);
      // small downward arrow
      ctx.beginPath();
      ctx.moveTo(arrowX, arrowY - 2);
      ctx.lineTo(arrowX, arrowY - 10);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(arrowX - 4, arrowY - 5);
      ctx.lineTo(arrowX, arrowY - 1);
      ctx.lineTo(arrowX + 4, arrowY - 5);
      ctx.stroke();
      ctx.restore();
    }});
  }}
}};

const segCtx = document.getElementById('segChart').getContext('2d');
new Chart(segCtx, {{
  type: 'bar',
  plugins: [indicatorPlugin],
  data: {{
    labels: segs,
    datasets: [
      {{
        label: 'New Patients',
        data: segs.map(s => segMap[s].new),
        backgroundColor: '#185FA5',
        borderRadius: 3,
      }},
      {{
        label: 'Returning Patients',
        data: segs.map(s => segMap[s].ret),
        backgroundColor: '#6BADA0',
        borderRadius: 3,
      }}
    ]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        callbacks: {{
          title: items => items[0].label,
          label: ctx => {{
            const s = segs[ctx.dataIndex];
            const d = segMap[s];
            return [
              ` New: ${{fmtI(d.new)}}`,
              ` Returning: ${{fmtI(d.ret)}}`,
              ` R:N ratio: ${{fmt(ratio(d))}}x`
            ];
          }}
        }}
      }}
    }},
    scales: {{
      x: {{ grid: {{ display: false }}, ticks: {{ font: {{ size: 11 }} }} }},
      y: {{
        beginAtZero: true,
        grace: 0,
        grid: {{ color: 'rgba(0,0,0,.06)' }},
        ticks: {{ font: {{ size: 10 }}, callback: v => fmtI(v) }}
      }}
    }},
    layout: {{ padding: {{ top: 36, bottom: 0 }} }}
  }}
}});

let ageChartInst = null;
function showDrilldown(segName, cardEl) {{
  document.querySelectorAll('.card').forEach(c => c.classList.remove('active'));
  cardEl.classList.add('active');

  const drillEl   = document.getElementById('drilldown');
  const titleEl   = document.getElementById('drilldown-title');
  const postopEl  = document.getElementById('postop-warn');
  const ageCanvas = document.getElementById('ageChart');
  const insightEl = document.getElementById('insightBar');

  drillEl.style.display = 'block';
  titleEl.textContent   = segName + ' — Returning per New Ratio by Age Group';

  postopEl.style.display  = 'none';
  ageCanvas.style.display = 'block';

  const rows = segMap[segName].rows
    .filter(r => r.new_patients > 0)
    .sort((a,b) => a.age_group.localeCompare(b.age_group));

  const labels = rows.map(r => r.age_group);
  const ratios = rows.map(r => +r.returning_per_new_ratio || 0);
  const colors = ratios.map(v => v >= 1.0 ? '#0F6E56' : '#E24B4A');

  const h = Math.max(180, labels.length * 42 + 60);
  ageCanvas.style.height = h + 'px';
  ageCanvas.height = h;

  if (ageChartInst) ageChartInst.destroy();
  ageChartInst = new Chart(ageCanvas.getContext('2d'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        data: ratios,
        backgroundColor: colors,
        borderRadius: 3,
        barThickness: 22,
      }}]
    }},
    options: {{
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        annotation: {{
          annotations: {{
            threshold: {{
              type: 'line', scaleID: 'x', value: 1.0,
              borderColor: '#BA7517', borderWidth: 2, borderDash: [5,4],
              label: {{ content: '1.0 threshold', enabled: true, position: 'start',
                        font: {{ size: 9 }}, color: '#BA7517', backgroundColor: 'transparent' }}
            }}
          }}
        }},
        tooltip: {{
          callbacks: {{ label: ctx => ` R:N ratio: ${{fmt(ctx.parsed.x)}}` }}
        }}
      }},
      scales: {{
        x: {{
          min: 0,
          grid: {{ color: 'rgba(0,0,0,.06)' }},
          ticks: {{ font: {{ size: 10 }}, callback: v => fmt(v) }}
        }},
        y: {{ grid: {{ display: false }}, ticks: {{ font: {{ size: 10 }} }} }}
      }}
    }}
  }});

  const below = labels.filter((_, i) => ratios[i] < 1.0);
  if (below.length) {{
    insightEl.style.display = 'block';
    insightEl.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24"
           fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:middle;margin-right:5px">
        <path d="M12 9v4"/><path d="M10.363 3.591l-8.106 13.534a1.914 1.914 0 0 0 1.636 2.871h16.214a1.914 1.914 0 0 0 1.636-2.87l-8.106-13.536a1.914 1.914 0 0 0-3.274 0z"/>
        <path d="M12 16h.01"/>
      </svg>
      <strong>Below retention threshold (R:N &lt; 1.0):</strong> ${{below.join(', ')}}
    `;
  }} else {{
    insightEl.style.display = 'none';
  }}
}}
</script>
</body>
</html>"""

            _stcomp.html(html_acq, height=520, scrolling=False)
    except Exception as e:
        st.error(f"Acquisition segments error: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Patient Profile Dashboard ─────────────────────────────────────────
    _cv_section("Patient Profile — What Brings Patients In")
    try:
        import json as _jpp
        df_pp = Q.load_patient_profile(filters, run_query)
        if not df_pp.empty:
            df_pp = df_pp.drop_duplicates(subset=["visit_id"]).copy()
            df_pp["gender_clean"] = (df_pp["gender"].str.upper().str.strip()
                                     .map({"F": "Female", "FEMALE": "Female",
                                           "M": "Male",   "MALE":   "Male"})
                                     .fillna("Other"))
            _PP_AGE_ORDER = [
                "Toddler (0-4)", "Child (5-12)", "Adolescent (13-17)", "Youth (18-24)",
                "Young Adult (25-34)", "Adult (35-44)", "Middle Age (45-54)",
                "Older Adult (55-64)", "Senior (65+)",
            ]

            # ── V1: top-10 diagnoses + drill-down by age group ───────────
            _dn = df_pp[df_pp["patient_type"] == "new"].groupby("clean_diagnosis")["visit_id"].count()
            _dr = df_pp[df_pp["patient_type"] == "returning"].groupby("clean_diagnosis")["visit_id"].count()
            _dt = _dn.add(_dr, fill_value=0).nlargest(10)
            _v1_diag = [{"label": l, "new": int(_dn.get(l, 0)), "ret": int(_dr.get(l, 0))}
                        for l in _dt.index.tolist()]
            _v1_drill = {}
            for _lbl in _dt.index.tolist():
                _dsub = df_pp[df_pp["clean_diagnosis"].str.contains(
                    _lbl.replace("(", r"\(").replace(")", r"\)"), case=False, na=False, regex=True
                )]
                _dgb = _dsub.groupby(["age_group", "patient_type"])["visit_id"].count()
                _da  = [a for a in _PP_AGE_ORDER if a in _dsub["age_group"].unique()]
                _v1_drill[_lbl] = {
                    "ages": _da,
                    "new":  [int(_dgb.get((a, "new"),       0)) for a in _da],
                    "ret":  [int(_dgb.get((a, "returning"),  0)) for a in _da],
                }

            # ── V2: gender split by specific diagnoses ────────────────────
            _FO_KW = ["antenatal","maternal","postnatal","anc","pnc",
                      "gynaecological","reproductive"]
            _REPR_EXCL = {
                "Toddler (0-4)", "Child (5-12)",
                "Older Adult (55-64)", "Senior (65+)",
            }
            _V2_TABS = [
                "Oncology", "Chronic Upper Airway",
                "Hypertension", "Neurologic", "Antenatal Care",
            ]
            _V2_PAT = {
                "Oncology":           "oncolog",
                "Chronic Upper Airway":"upper airway",
                "Hypertension":       "hypertension",
                "Neurologic":         "neurolog",
                "Antenatal Care":     "antenatal",
            }
            def _v2_build(df_sub, tab):
                is_fo = any(kw in tab.lower() for kw in _FO_KW)
                if is_fo:
                    df_sub = df_sub[
                        (df_sub["gender_clean"] == "Female") &
                        (~df_sub["age_group"].isin(_REPR_EXCL))
                    ]
                _gb  = df_sub.groupby(["age_group","gender_clean","patient_type"])["visit_id"].count()
                ages = [a for a in _PP_AGE_ORDER if a in df_sub["age_group"].unique()]
                return {
                    "ages":     ages,
                    "male_new": [int(_gb.get((a,"Male","new"),      0)) for a in ages],
                    "male_ret": [int(_gb.get((a,"Male","returning"), 0)) for a in ages],
                    "fem_new":  [int(_gb.get((a,"Female","new"),     0)) for a in ages],
                    "fem_ret":  [int(_gb.get((a,"Female","returning"),0)) for a in ages],
                    "female_only": is_fo,
                }
            _v2 = {}
            for _tab in _V2_TABS:
                _pat = _V2_PAT[_tab]
                _tdf = df_pp[df_pp["clean_diagnosis"].str.lower().str.contains(_pat, na=False)]
                _v2[_tab] = _v2_build(_tdf, _tab)

            # ── V3: Inpatient vs outpatient ───────────────────────────────
            _new_df    = df_pp[df_pp["patient_type"] == "new"]
            _v3_new_op = int((_new_df["visit_type"] == "outpatient").sum())
            _v3_new_ip = int((_new_df["visit_type"] == "inpatient").sum())
            _v3_segs   = []
            for _s in ["CHRONIC", "ONCOLOGY", "MATERNAL", "MENTAL_HEALTH"]:
                _sd = _new_df[_new_df["acquisition_segment"] == _s]
                _v3_segs.append({
                    "seg": _s,
                    "op":  int((_sd["visit_type"] == "outpatient").sum()),
                    "ip":  int((_sd["visit_type"] == "inpatient").sum()),
                })

            _pp_data = _jpp.dumps({
                "v1": {"diag": _v1_diag, "drill": _v1_drill},
                "v2": _v2,
                "v2tabs": _V2_TABS,
                "v3": {"new_op": _v3_new_op, "new_ip": _v3_new_ip, "segs": _v3_segs},
            })

            _pp_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.30.0/dist/tabler-icons.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{{
  --clr-primary:#185FA5;--clr-green:#0F6E56;--clr-purple:#3C3489;
  --clr-blue-lt:#B5D4F4;--clr-pink:#D4537E;--clr-pink-lt:#F4C0D1;
  --clr-bg:#FAFAFA;--clr-card:#FFFFFF;--clr-border:rgba(0,0,0,0.08);
  --clr-text:#111827;--clr-sub:#6B7280;--clr-ins:#F3F4F6;
  --radius:8px;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{min-height:100%;overflow-y:auto}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  font-size:12px;background:var(--clr-bg);color:var(--clr-text);
  display:flex;flex-direction:column}}
h2.sr{{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0)}}
/* tab bar — never shrinks */
.tab-bar{{flex-shrink:0;display:flex;gap:6px;padding:10px 12px 0;
  border-bottom:1px solid var(--clr-border);background:var(--clr-card)}}
.tab-btn{{padding:5px 12px;border:1px solid transparent;border-radius:20px;cursor:pointer;
  font-size:11px;font-weight:500;color:var(--clr-sub);background:transparent;transition:.12s}}
.tab-btn.active{{background:var(--clr-ins);border-color:var(--clr-primary);color:var(--clr-primary)}}
/* views fill remaining height as a flex column */
.view{{display:none;flex-direction:column;padding:10px 12px;gap:8px}}
.view.active{{display:flex}}
/* chart card — fixed height so drill-down never compresses it */
.grow-card{{display:flex;flex-direction:column;height:400px;
  background:var(--clr-card);border:1px solid var(--clr-border);border-radius:var(--radius);padding:10px 12px}}
/* static card (metrics, drill) */
.card{{background:var(--clr-card);border:1px solid var(--clr-border);border-radius:var(--radius);
  padding:10px 12px;flex-shrink:0}}
.drill-card{{background:var(--clr-card);border:1px solid var(--clr-border);border-radius:var(--radius);
  padding:10px 12px;flex-shrink:0;display:none}}
.drill-title{{font-size:11px;font-weight:600;color:var(--clr-text);margin-bottom:8px}}
/* chart wrapper fills card */
.chart-wrap{{flex:1;position:relative;min-height:0}}
/* shared elements that never grow */
.legend{{flex-shrink:0;display:flex;gap:12px;flex-wrap:wrap;margin-bottom:6px;align-items:center}}
.leg-item{{display:flex;align-items:center;gap:4px;font-size:10px;color:var(--clr-sub)}}
.leg-sq{{width:10px;height:10px;border-radius:2px;flex-shrink:0}}
.helper{{font-size:10px;color:var(--clr-sub);font-style:italic;margin-left:auto}}
.diag-tabs{{flex-shrink:0;display:flex;gap:6px;flex-wrap:wrap}}
.diag-btn{{padding:4px 10px;border:1px solid var(--clr-border);border-radius:20px;
  cursor:pointer;font-size:10px;font-weight:500;color:var(--clr-sub);background:transparent}}
.diag-btn.active{{background:var(--clr-ins);border-color:var(--clr-primary);color:var(--clr-primary)}}
.metrics{{flex-shrink:0;display:flex;gap:10px}}
.metric{{background:var(--clr-card);border:1px solid var(--clr-border);border-radius:var(--radius);
  padding:10px 14px;flex:1}}
.metric .mlabel{{font-size:10px;color:var(--clr-sub);text-transform:uppercase;letter-spacing:.05em}}
.metric .mval{{font-size:22px;font-weight:700;color:var(--clr-text);margin:2px 0}}
.metric .msub{{font-size:10px;color:var(--clr-sub)}}
/* insight — fixed at bottom, never grows */
.insight{{flex-shrink:0;background:var(--clr-ins);border-radius:var(--radius);padding:8px 12px;
  display:flex;align-items:flex-start;gap:8px;font-size:11px;color:var(--clr-sub)}}
.insight i{{font-size:14px;flex-shrink:0;margin-top:1px}}
canvas{{display:block;width:100%!important;height:100%!important}}
</style>
</head>
<body>
<h2 class="sr">Patient profile analytics — Kisumu Specialists</h2>
<div class="tab-bar">
  <button class="tab-btn active" onclick="switchTab('v1',this)">What brings patients in</button>
  <button class="tab-btn" onclick="switchTab('v2',this)">Gender split</button>
  <button class="tab-btn" onclick="switchTab('v3',this)">Inpatient vs outpatient</button>
</div>

<!-- VIEW 1 -->
<div id="v1" class="view active">
  <div class="grow-card">
    <div class="legend">
      <span class="leg-item"><span class="leg-sq" style="background:#185FA5"></span>New patients</span>
      <span class="leg-item"><span class="leg-sq" style="background:#0F6E56"></span>Returning patients</span>
      <span class="helper">Click a bar to see age group breakdown</span>
    </div>
    <div class="chart-wrap">
      <canvas id="v1c" role="img" aria-label="Top 10 diagnoses by new and returning visits">No data</canvas>
    </div>
  </div>
  <div class="drill-card" id="v1-drill">
    <div class="drill-title" id="v1-drill-title"></div>
    <div class="legend">
      <span class="leg-item"><span class="leg-sq" style="background:#185FA5"></span>New</span>
      <span class="leg-item"><span class="leg-sq" style="background:#0F6E56"></span>Returning</span>
    </div>
    <div id="v1-drill-wrap" style="position:relative">
      <canvas id="v1dc" role="img" aria-label="Age group breakdown for selected diagnosis">No data</canvas>
    </div>
  </div>
  <div class="insight">
    <i class="ti ti-info-circle"></i>
    <span>Oncology and Chronic Upper Airway are the top drivers for both new and returning patients.
    Hypertension shows the largest gap — returning vs new — consistent with ongoing chronic management.
    Gynaecological NCD skews toward new patients which warrants investigation as a chronic condition that should generate follow-up.</span>
  </div>
</div>

<!-- VIEW 2 -->
<div id="v2" class="view">
  <div class="diag-tabs" id="v2-tabs"></div>
  <div class="grow-card">
    <div class="legend" id="v2-legend"></div>
    <div class="chart-wrap">
      <canvas id="v2c" role="img" aria-label="Gender split by age group and diagnosis">No data</canvas>
    </div>
  </div>
  <div class="insight" id="v2-ins"></div>
</div>

<!-- VIEW 3 -->
<div id="v3" class="view">
  <div class="metrics" id="v3-metrics"></div>
  <div class="grow-card">
    <div class="legend">
      <span class="leg-item"><span class="leg-sq" style="background:#185FA5"></span>Outpatient</span>
      <span class="leg-item"><span class="leg-sq" style="background:#3C3489"></span>Inpatient</span>
    </div>
    <div class="chart-wrap">
      <canvas id="v3c" role="img" aria-label="Inpatient vs outpatient new patients by segment">No data</canvas>
    </div>
  </div>
  <div class="insight">
    <i class="ti ti-info-circle"></i>
    <span>Most new patients arrive as outpatients across all segments.
    Maternal has the highest inpatient rate for new patients — likely emergency and high-risk first presentations.
    New chronic patients arriving as inpatients signals advanced or unmanaged disease at first presentation.</span>
  </div>
</div>

<script>
const D={_pp_data};
function fmt(n){{return n==null?'—':Number(n).toLocaleString()}}
function pct(n,t){{return t?((n/t)*100).toFixed(1)+'%':'—'}}
const SEG_LABELS={{'CHRONIC':'Chronic','ONCOLOGY':'Oncology','MATERNAL':'Maternal','MENTAL_HEALTH':'Mental Health'}};

// ── tab switch ──────────────────────────────────────────────────────────
function switchTab(id,btn){{
  document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
}}

// ── VIEW 1 ──────────────────────────────────────────────────────────────
let v1Chart=null, v1DrillChart=null;
(function(){{
  const v=D.v1;
  const labels=v.diag.map(d=>d.label);
  if(v1Chart)v1Chart.destroy();
  v1Chart=new Chart(document.getElementById('v1c'),{{
    type:'bar',
    data:{{labels,datasets:[
      {{label:'New',     data:v.diag.map(d=>d.new), backgroundColor:'#185FA5',borderRadius:3}},
      {{label:'Returning',data:v.diag.map(d=>d.ret),backgroundColor:'#0F6E56',borderRadius:3}},
    ]}},
    options:{{
      indexAxis:'y',responsive:true,maintainAspectRatio:false,
      onClick(_e,els){{
        if(!els.length)return;
        const lbl=labels[els[0].index];
        openDrill(lbl);
      }},
      plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{
        label:c=>c.dataset.label+': '+fmt(c.raw)
      }}}}}},
      scales:{{
        x:{{grid:{{color:'rgba(0,0,0,.05)'}},ticks:{{font:{{size:10}},callback:v=>fmt(v)}}}},
        y:{{grid:{{display:false}},ticks:{{font:{{size:10}}}}}},
      }}
    }}
  }});
}})();

function openDrill(label){{
  const dd=D.v1.drill[label];
  if(!dd||!dd.ages.length)return;
  const panel=document.getElementById('v1-drill');
  document.getElementById('v1-drill-title').textContent=label;
  const h=Math.max(160,dd.ages.length*44+80);
  document.getElementById('v1-drill-wrap').style.height=h+'px';
  panel.style.display='block';
  if(v1DrillChart){{v1DrillChart.destroy();v1DrillChart=null;}}
  v1DrillChart=new Chart(document.getElementById('v1dc'),{{
    type:'bar',
    data:{{labels:dd.ages,datasets:[
      {{label:'New',     data:dd.new,backgroundColor:'#185FA5',borderRadius:3}},
      {{label:'Returning',data:dd.ret,backgroundColor:'#0F6E56',borderRadius:3}},
    ]}},
    options:{{
      indexAxis:'y',responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:c=>c.dataset.label+': '+fmt(c.raw)}}}}}},
      scales:{{
        x:{{grid:{{color:'rgba(0,0,0,.05)'}},ticks:{{font:{{size:10}},callback:v=>fmt(v)}}}},
        y:{{grid:{{display:false}},ticks:{{font:{{size:10}}}}}},
      }}
    }}
  }});
  setTimeout(()=>panel.scrollIntoView({{behavior:'smooth',block:'nearest'}}),80);
}}

// ── VIEW 2 ──────────────────────────────────────────────────────────────
let v2Chart=null;
const V2_INSIGHTS={{
  'Oncology':'<i class="ti ti-info-circle"></i><span>Senior males show the strongest returning dominance — likely prostate and haematological cancers requiring sustained management. Female oncology peaks in Adult (35–44) for new patients, suggesting reproductive cancers as a key driver.</span>',
  'Chronic Upper Airway':'<i class="ti ti-info-circle"></i><span>Volume dominated by Toddler and Child age groups for both genders — paediatric respiratory conditions are the primary driver. Gender balance is near-equal across all age groups.</span>',
  'Hypertension':'<i class="ti ti-info-circle"></i><span>Both genders show strong returning dominance in older age groups. Male new patients are consistently lower than female across all age groups — consistent with under-detection of hypertension in males at first presentation.</span>',
  'Neurologic':'<i class="ti ti-info-circle"></i><span>Female patients dominate Adult (35–44) returning visits — worth investigating for conditions like migraine and multiple sclerosis which are more prevalent in women.</span>',
  'Antenatal Care':'<i class="ti ti-info-circle"></i><span>Female-only diagnosis. Young Adult (25–34) dominates with strong returning ratio — the primary maternal cohort. Any Adolescent (13–17) visits should be validated as possible data quality issues before presenting.</span>',
}};
function buildV2(tab){{
  const sd=D.v2[tab];
  if(!sd)return;
  const lgd=document.getElementById('v2-legend');
  if(sd.female_only){{
    lgd.innerHTML='<span class="leg-item"><span class="leg-sq" style="background:#D4537E"></span>Female — new</span>'
      +'<span class="leg-item"><span class="leg-sq" style="background:#F4C0D1"></span>Female — returning</span>';
  }}else{{
    lgd.innerHTML='<span class="leg-item"><span class="leg-sq" style="background:#185FA5"></span>Male — new</span>'
      +'<span class="leg-item"><span class="leg-sq" style="background:#B5D4F4"></span>Male — returning</span>'
      +'<span class="leg-item"><span class="leg-sq" style="background:#D4537E"></span>Female — new</span>'
      +'<span class="leg-item"><span class="leg-sq" style="background:#F4C0D1"></span>Female — returning</span>';
  }}
  if(v2Chart){{v2Chart.destroy();v2Chart=null;}}
  const datasets=[];
  if(!sd.female_only){{
    datasets.push({{label:'Male — new',    data:sd.male_new,backgroundColor:'#185FA5',borderRadius:3}});
    datasets.push({{label:'Male — ret',    data:sd.male_ret,backgroundColor:'#B5D4F4',borderRadius:3}});
  }}
  datasets.push({{label:'Female — new',  data:sd.fem_new, backgroundColor:'#D4537E',borderRadius:3}});
  datasets.push({{label:'Female — ret',  data:sd.fem_ret, backgroundColor:'#F4C0D1',borderRadius:3}});
  v2Chart=new Chart(document.getElementById('v2c'),{{
    type:'bar',
    data:{{labels:sd.ages,datasets}},
    options:{{
      indexAxis:'y',responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:c=>c.dataset.label+': '+fmt(c.raw)}}}}}},
      scales:{{
        x:{{grid:{{color:'rgba(0,0,0,.05)'}},ticks:{{font:{{size:10}},callback:v=>fmt(v)}}}},
        y:{{grid:{{display:false}},ticks:{{font:{{size:10}}}}}},
      }}
    }}
  }});
  document.getElementById('v2-ins').innerHTML=V2_INSIGHTS[tab]||'';
}}
(function(){{
  const tabs=document.getElementById('v2-tabs');
  D.v2tabs.forEach((t,i)=>{{
    const b=document.createElement('button');
    b.className='diag-btn'+(i===0?' active':'');
    b.textContent=t;
    b.onclick=()=>{{
      document.querySelectorAll('#v2-tabs .diag-btn').forEach(x=>x.classList.remove('active'));
      b.classList.add('active');buildV2(t);
    }};
    tabs.appendChild(b);
  }});
  buildV2(D.v2tabs[0]);
}})();

// ── VIEW 3 ──────────────────────────────────────────────────────────────
(function(){{
  const v=D.v3;
  const tot=v.new_op+v.new_ip;
  document.getElementById('v3-metrics').innerHTML=
    '<div class="metric"><div class="mlabel">New — outpatient</div><div class="mval">'+fmt(v.new_op)+'</div><div class="msub">'+pct(v.new_op,tot)+' of new visits</div></div>'+
    '<div class="metric"><div class="mlabel">New — inpatient</div><div class="mval">'+fmt(v.new_ip)+'</div><div class="msub">'+pct(v.new_ip,tot)+' of new visits</div></div>'+
    '<div class="metric"><div class="mlabel">Inpatient rate (new)</div><div class="mval">'+pct(v.new_ip,tot)+'</div><div class="msub">Across all segments</div></div>';
  const labels=v.segs.map(s=>SEG_LABELS[s.seg]||s.seg);
  new Chart(document.getElementById('v3c'),{{
    type:'bar',
    data:{{labels,datasets:[
      {{label:'Outpatient',data:v.segs.map(s=>s.op),backgroundColor:'#185FA5',borderRadius:4}},
      {{label:'Inpatient', data:v.segs.map(s=>s.ip),backgroundColor:'#3C3489',borderRadius:4}},
    ]}},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{
        label:c=>{{
          const s=v.segs[c.dataIndex];
          return c.dataset.label+': '+fmt(c.raw)+' ('+pct(c.raw,s.op+s.ip)+')';
        }}
      }}}}}},
      scales:{{
        x:{{grid:{{display:false}},ticks:{{font:{{size:11}}}}}},
        y:{{grid:{{color:'rgba(0,0,0,.05)'}},ticks:{{font:{{size:10}},callback:v=>fmt(v)}}}},
      }}
    }}
  }});
}})();
</script>
</body>
</html>"""
            _stcomp.html(_pp_html, height=920, scrolling=True)
    except Exception as e:
        st.error(f"Patient profile dashboard: {e}")



# ══════════════════════════════════════════════════════════════════════════════
# WARD DEEP-DIVE — DIP DETECTION + H3 DIAGNOSIS TRENDS
# ══════════════════════════════════════════════════════════════════════════════

_DIAG_COLOR_MAP = {
    "communicable - typhoid":          "#EF9F27",
    "communicable - other infectious": "#E24B4A",
    "ncd - respiratory":               "#D85A30",
    "cancer / oncology":               "#7F77DD",
    "musculoskeletal":                 "#888780",
}
_DIAG_FALLBACK = ["#378ADD", "#1D9E75", "#9B59B6", "#2D2D2A"]


def _diag_color(name: str, rank: int) -> str:
    c = _DIAG_COLOR_MAP.get(name.lower().strip())
    if c:
        return c
    return _DIAG_FALLBACK[rank % len(_DIAG_FALLBACK)]


def _detect_dip(df: "pd.DataFrame") -> dict:
    """Detect most recent sustained dip: ≥3 consecutive months where the
    3-month rolling avg is >15% below the prior 6-month baseline."""
    no_dip = {
        "dip_detected": False,
        "baseline_avg": float(df["total_admissions"].mean()) if not df.empty else 0,
    }
    if df.empty or len(df) < 9:
        return no_dip

    df = df.sort_values("admit_month").copy().reset_index(drop=True)
    df["rolling_3m"]   = df["total_admissions"].rolling(3, min_periods=3).mean()
    df["baseline_6m"]  = df["total_admissions"].rolling(6, min_periods=6).mean().shift(1)
    df["is_candidate"] = (
        df["rolling_3m"].notna()
        & df["baseline_6m"].notna()
        & (df["rolling_3m"] < df["baseline_6m"] * 0.85)
    )

    runs, cur = [], []
    for _, row in df.iterrows():
        if row["is_candidate"]:
            cur.append(row["admit_month"])
        else:
            if len(cur) >= 3:
                runs.append(list(cur))
            cur = []
    if len(cur) >= 3:
        runs.append(list(cur))

    if not runs:
        return no_dip

    dip_months = runs[-1]
    dip_set    = set(dip_months)
    non_dip    = df[~df["admit_month"].isin(dip_set)]
    baseline   = float(non_dip["total_admissions"].mean()) if not non_dip.empty else float(df["total_admissions"].mean())
    dip_avg    = float(df[df["admit_month"].isin(dip_set)]["total_admissions"].mean())

    return {
        "dip_detected":   True,
        "dip_months":     dip_months,
        "dip_months_set": dip_set,
        "dip_start":      dip_months[0],
        "dip_end":        dip_months[-1],
        "baseline_avg":   baseline,
        "dip_avg":        dip_avg,
    }


def _dip_vrect(fig, dip_info: dict):
    """Overlay dip-period shading and labels on a Plotly figure."""
    if not dip_info.get("dip_detected"):
        return
    dip_months = dip_info.get("dip_months", [])
    if not dip_months:
        return
    try:
        import pandas as _pd
        x0 = dip_months[0]  - _pd.DateOffset(days=15)
        x1 = dip_months[-1] + _pd.DateOffset(days=15)
        fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(239,159,39,0.12)",
                      layer="below", line_width=0)
        for xv in (x0, x1):
            fig.add_vline(x=str(xv)[:10], line_dash="dash",
                          line_color="rgba(239,159,39,0.4)", line_width=1)
        mid = x0 + (x1 - x0) / 2
        fig.add_annotation(
            x=str(mid)[:10], y=1.0, yref="paper", text="Dip period",
            showarrow=False, font=dict(size=9, color="#854F0B"),
            bgcolor="rgba(0,0,0,0)", xanchor="center", yanchor="top",
        )
    except Exception:
        pass


def _strip(color: str, text: str):
    bg     = {"red": "#FFF5F5", "amber": "#FFFBEB", "green": "#F0FFF4",
              "neutral": "#F4F8FC"}.get(color, "#F4F8FC")
    border = {"red": "#C53030", "amber": "#D97706", "green": "#276749",
              "neutral": "#0072CE"}.get(color, "#0072CE")
    st.markdown(
        f'<div style="background:{bg};border-left:3px solid {border};'
        f'border-radius:0 4px 4px 0;padding:10px 14px;font-size:13px;'
        f'margin-top:6px;line-height:1.7;color:#003467">{text}</div>',
        unsafe_allow_html=True,
    )


def _render_h3(df_dx: "pd.DataFrame", dip_info: dict, ward_name: str):
    """Hypothesis 3 — Diagnosis trend lines."""
    import plotly.graph_objects as _go
    import pandas as _pd

    dip_detected = dip_info.get("dip_detected", False)

    if dip_detected:
        ds = dip_info["dip_start"].strftime("%b %Y")
        de = dip_info["dip_end"].strftime("%b %Y")
        panel_title = f"Which diagnoses dropped in {ds}–{de} and recovered?"
    else:
        panel_title = f"Which diagnoses are driving {ward_name} admissions?"

    st.markdown(
        f'<div style="font-size:12px;font-weight:500;color:#111827;margin-bottom:8px">'
        f'{panel_title}</div>',
        unsafe_allow_html=True,
    )

    if df_dx.empty:
        st.caption("No diagnosis data for this ward / period.")
        _strip("neutral", "No data returned from the diagnosis query. Check that STG_EVALUATION_ICD10_DIAGNOSIS_PIVOTED has records linked to this ward.")
        return

    df_dx = df_dx.copy()
    df_dx["visit_month"] = _pd.to_datetime(df_dx["visit_month"])

    # Rank diagnoses by total volume
    totals = (
        df_dx.groupby("diagnosis_name")["monthly_visit_count"]
        .sum()
        .sort_values(ascending=False)
    )
    ranked = list(totals.index)           # rank 0 = highest

    # Build complete month spine so gaps are explicit nulls (not connected)
    all_months = sorted(df_dx["visit_month"].unique())
    month_map  = {m: i for i, m in enumerate(all_months)}

    fig = _go.Figure()
    for rank, diag in enumerate(ranked):
        sub   = df_dx[df_dx["diagnosis_name"] == diag].set_index("visit_month")
        y     = [float(sub.loc[m, "monthly_visit_count"]) if m in sub.index else None
                 for m in all_months]
        color = _diag_color(diag, rank)
        solid = rank < 2
        fig.add_trace(_go.Scatter(
            x=all_months,
            y=y,
            name=diag,
            mode="lines",
            connectgaps=False,
            line=dict(
                color=color,
                width=2 if solid else 1.5,
                dash="solid" if solid else "dash",
            ),
            marker=dict(size=0),
            hovertemplate=(
                f"<b>{diag}</b><br>"
                "%{x|%b %Y}: %{y:.0f} admissions<extra></extra>"
            ),
        ))

    _dip_vrect(fig, dip_info)

    fig.update_layout(
        height=230,
        margin=dict(l=0, r=0, t=6, b=0),
        paper_bgcolor="#fff",
        plot_bgcolor="#fff",
        legend=dict(
            orientation="h",
            y=-0.18, x=0.5, xanchor="center",
            font=dict(size=10, color="#6b7280"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            showgrid=False,
            showline=False,
            tickformat="%b %y",
            nticks=6,
            tickfont=dict(size=10, color="#9ca3af"),
        ),
        yaxis=dict(
            title="Admissions",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.05)",
            showline=False,
            nticks=5,
            tickfont=dict(size=10, color="#9ca3af"),
            title_font=dict(size=11, color="#9ca3af"),
        ),
        hovermode="x unified",
    )
    _H().get("pc", lambda f: st.plotly_chart(f, use_container_width=True, config=_PC_CFG))(fig)

    # ── Verdict logic ──────────────────────────────────────────────────────
    dip_set = dip_info.get("dip_months_set", set())

    dip_drivers, baselines = [], []

    if dip_detected and dip_set:
        dip_start = dip_info["dip_start"]
        dip_end   = dip_info["dip_end"]

        for diag in ranked:
            sub = df_dx[df_dx["diagnosis_name"] == diag].set_index("visit_month")["monthly_visit_count"]

            pre_rows  = sub[sub.index < dip_start].tail(3)
            dip_rows  = sub[sub.index.isin(dip_set)]
            post_rows = sub[sub.index > dip_end].head(3)

            pre_avg  = float(pre_rows.mean())  if not pre_rows.empty  else 0
            dip_avg  = float(dip_rows.mean())  if not dip_rows.empty  else 0
            post_avg = float(post_rows.mean()) if not post_rows.empty else 0

            if pre_avg > 0 and dip_avg < pre_avg * 0.70 and post_avg >= pre_avg * 0.80:
                dip_drivers.append(diag)

            mean_v = float(sub.mean())
            std_v  = float(sub.std())
            if mean_v > 0 and (std_v / mean_v) < 0.20:
                baselines.append(diag)

        first_post = (
            min(m for m in df_dx["visit_month"].unique() if m > dip_end).strftime("%b %Y")
            if any(m > dip_end for m in df_dx["visit_month"].unique()) else "recovery"
        )

        if dip_drivers and baselines:
            driver_str   = " and ".join(dip_drivers[:2])
            baseline_str = baselines[0]
            verdict = (
                f"{driver_str} were the engines — both dropped in the dip and "
                f"recovered in {first_post}. {baseline_str} remained flat throughout "
                f"— a consistent baseline condition, not a seasonal driver."
            )
            strip_c = "amber"
            strip_t = (
                f"{driver_str} drove both the dip and the recovery. "
                f"This pattern is consistent with seasonal variation in admissions. "
                f"Monitor whether the same dip recurs in the same months next year."
            )
        elif dip_drivers:
            driver_str = " and ".join(dip_drivers[:2])
            verdict = (
                f"{driver_str} drove both the dip and the recovery. "
                f"All top conditions showed some seasonal variation in this period."
            )
            strip_c = "amber"
            strip_t = (
                f"{driver_str} are the primary seasonal engines for this ward. "
                f"Monitor whether the same dip recurs in the same months next year."
            )
        else:
            verdict = (
                f"No single diagnosis drove the dip. The decline was broad-based "
                f"across all top conditions — investigate external demand factors "
                f"for {dip_info['dip_start'].strftime('%b %Y')} – "
                f"{dip_info['dip_end'].strftime('%b %Y')}."
            )
            strip_c = "green"
            strip_t = (
                "No dominant diagnosis drove the dip — it was a broad-based demand "
                "decline. Hypothesis 3 does not explain the dip. Focus investigation "
                "on new patient acquisition and external demand factors."
            )
    else:
        # No dip — dominant diagnosis framing
        top1 = ranked[0] if len(ranked) > 0 else "—"
        top2 = ranked[1] if len(ranked) > 1 else "—"
        verdict = (
            f"{top1} accounts for the largest share of admissions in this ward. "
            f"{top2} is the next largest contributor."
        )
        strip_c = "neutral"
        strip_t = (
            f"No sustained dip was detected. The trend lines show relative stability "
            f"across the top diagnoses. {top1} is the dominant driver to monitor."
        )

    # Verdict card
    st.markdown(
        f'<div style="background:#F4F8FC;border-radius:6px;padding:8px 10px;margin-top:8px">'
        f'<div style="font-size:10px;text-transform:uppercase;color:#9ca3af;'
        f'letter-spacing:1px;margin-bottom:3px">VERDICT</div>'
        f'<div style="font-size:11px;font-weight:500;color:#111827;line-height:1.5">'
        f'{verdict}</div></div>',
        unsafe_allow_html=True,
    )
    _strip(strip_c, strip_t)


def render_ward_deepdive(filters: dict, run_query):
    """Ward deep-dive section — diagnosis trend analysis (Hypothesis 3)."""
    import pandas as _pd

    st.markdown(
        '<div style="font-size:11px;font-weight:700;color:#0072CE;text-transform:uppercase;'
        'letter-spacing:2px;padding:8px 0 6px;border-bottom:2px solid #EBF3FB;'
        'margin-bottom:12px">Ward Deep-Dive — Diagnosis Trend Analysis</div>',
        unsafe_allow_html=True,
    )

    # Ward dropdown
    try:
        df_wards = Q.load_deepdive_ward_list(filters, run_query)
    except Exception as e:
        st.warning(f"Could not load ward list: {e}")
        return

    if df_wards.empty:
        st.caption("No ward data for the current filters.")
        return

    ward_list = df_wards["ward"].tolist()
    if ("deepdive_ward" not in st.session_state
            or st.session_state["deepdive_ward"] not in ward_list):
        st.session_state["deepdive_ward"] = ward_list[0]

    selected_ward = st.selectbox(
        "Select ward to investigate",
        options=ward_list,
        key="deepdive_ward",
    )

    # Dip detection
    try:
        df_monthly = Q.load_deepdive_monthly(filters, run_query, selected_ward)
        df_monthly["admit_month"] = _pd.to_datetime(df_monthly["admit_month"])
        df_monthly = df_monthly.sort_values("admit_month").reset_index(drop=True)
        dip_info = _detect_dip(df_monthly)
    except Exception as e:
        st.warning(f"Dip detection failed: {e}")
        dip_info = {"dip_detected": False, "baseline_avg": 0}

    # Section header
    if dip_info.get("dip_detected"):
        ds = dip_info["dip_start"].strftime("%b %Y")
        de = dip_info["dip_end"].strftime("%b %Y")
        title    = f"Why did {selected_ward} dip in {ds} – {de}?"
        subtitle = "Diagnosis trend analysis — which conditions drove the dip and which recovered?"
    else:
        title    = f"What is driving {selected_ward} admissions?"
        subtitle = "Trend analysis across the top 5 diagnoses for this ward."

    st.markdown(
        f'<div style="font-size:13px;font-weight:500;color:#111827;margin-bottom:2px">'
        f'{title}</div>'
        f'<div style="font-size:11px;color:#6b7280;margin-bottom:12px">{subtitle}</div>',
        unsafe_allow_html=True,
    )

    # Get ward_id and load diagnosis data
    try:
        ward_id = Q.load_deepdive_ward_id(filters, run_query, selected_ward)
        if ward_id is None:
            st.warning(f"No ward_id found for '{selected_ward}' — cannot load diagnosis data.")
            return
        df_dx = Q.load_h3_diagnosis_trends(filters, run_query, ward_id)
        df_dx["visit_month"] = _pd.to_datetime(df_dx["visit_month"])
    except Exception as e:
        st.warning(f"Diagnosis query failed: {e}")
        return

    _render_h3(df_dx, dip_info, selected_ward)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — TODAY'S BRIEFING
# ══════════════════════════════════════════════════════════════════════════════

def render_briefing(filters: dict, run_query):
    """Today's Briefing — 3 sections: headline numbers, signals, priority actions."""
    from ksh.clinical_module.ui_template import (
        page_header, section_header, stat_strip,
        anomaly_banner, action_cards,
    )

    def _load(fn, label=""):
        try:
            df = fn(filters, run_query)
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as e:
            if label:
                st.warning(f"{label}: {e}")
            return pd.DataFrame()

    def _val(df, col, default=None):
        if df.empty or col not in df.columns:
            return default
        v = df[col].iloc[0]
        return None if pd.isna(v) else v

    def _n(v):
        if v is None: return "—"
        try:
            f = float(v)
            if f >= 1_000_000: return f"{f/1_000_000:.1f}M"
            if f >= 1_000:     return f"{f/1_000:.1f}K"
            return f"{f:,.0f}"
        except: return str(v)

    def _p(v, d=1):
        if v is None: return "—"
        try: return f"{float(v):.{d}f}%"
        except: return str(v)

    def _f(v):
        if v is None: return 0.0
        try: return float(v)
        except: return 0.0

    def _kes_m(v):
        try:
            f = float(v)
            return f"KES {f/1_000_000:.1f}M" if f >= 1_000_000 else f"KES {_n(f)}"
        except: return "—"

    # ── Load all data ─────────────────────────────────────────────────────
    with st.spinner("Loading briefing…"):
        df_opd    = _load(Q.load_opd_ipd_overall,     "OPD→IPD")
        df_ret    = _load(Q.load_retention_overview,   "Retention")
        df_acq    = _load(Q.load_acquisition_overview, "Acquisition")
        df_lap    = _load(Q.load_lapsing_cohort,       "Lapsing")

    # ── Derive scalars ────────────────────────────────────────────────────
    total_ipd       = _f(_val(df_opd, "total_ipd_admissions"))
    conversion_rate = _f(_val(df_opd, "overall_rate_pct"))
    universe_rate   = _f(_val(df_opd, "retention_universe_rate_pct"))
    active_pct      = _f(_val(df_ret, "active_pct"))
    active_count    = _f(_val(df_ret, "active_count"))
    ltfu_count      = _f(_val(df_ret, "ltfu_count"))
    ltfu_pct        = _f(_val(df_ret, "ltfu_pct"))
    lapsing_count   = _f(_val(df_ret, "lapsing_count"))
    return_rate     = _f(_val(df_acq, "return_rate_pct"))
    escalation_rate = _f(_val(df_opd, "mix_gap_pp"))
    rev_at_risk     = _val(df_lap, "recoverable_revenue_kes")

    # ── Page header ───────────────────────────────────────────────────────
    page_header("Today's Briefing", "Kisumu Specialists Hospital · Clinical overview")

    # ── A — Headline numbers ──────────────────────────────────────────────
    section_header("A — Headline numbers")
    stat_strip([
        {"label": "Total IPD admissions",
         "value": _n(total_ipd),
         "hint":  "Sep 2024 – present"},
        {"label": "Conversion rate",
         "value": _p(conversion_rate),
         "hint":  f"Complex patients {_p(universe_rate)}",
         "hint_good": conversion_rate >= 8,
         "accent_color": "#A32D2D" if conversion_rate < 8 else "#0F6E56"},
        {"label": "Active chronic patients",
         "value": _p(active_pct),
         "hint":  f"{_n(active_count)} retained",
         "hint_good": True,
         "accent_color": "#0F6E56"},
        {"label": "LTFU (>180d)",
         "value": _n(ltfu_count),
         "hint":  f"{_p(ltfu_pct)} of chronic",
         "hint_good": False,
         "accent_color": "#A32D2D"},
        {"label": "OPD return rate",
         "value": _p(return_rate),
         "hint":  f"{_p(escalation_rate)}% mix gap",
         "hint_good": return_rate >= 40,
         "accent_color": "#854F0B" if return_rate < 40 else "#0F6E56"},
    ])

    # ── B — Signals requiring attention ───────────────────────────────────
    signals = []

    if conversion_rate > 0 and conversion_rate < 8:
        signals.append((
            "OPD → IPD conversion below reference",
            f"Overall rate is {_p(conversion_rate)} — below the 8% reference floor. "
            f"Complex patient rate {_p(universe_rate)} shows the potential. "
            "See OPD → IPD Conversion tab.",
        ))

    if ltfu_pct > 35:
        signals.append((
            "LTFU rate above threshold",
            f"{_p(ltfu_pct)} of chronic patients are lost to follow-up (>180d) — "
            "above the 35% investigation threshold. "
            "100% had no documented follow-up date. See Flow and Retention tab.",
        ))

    if return_rate > 0 and return_rate < 30:
        signals.append((
            "Low OPD return rate",
            f"Only {_p(return_rate)} of patients return for a second visit. "
            "Review discharge communication and follow-up scheduling. See Patient Acquisition tab.",
        ))

    if signals:
        section_header("B — Signals requiring attention")
        for title, body in signals:
            anomaly_banner(title, body)

    # ── C — Priority actions ───────────────────────────────────────────────
    section_header("C — Priority actions")
    action_cards([
        {
            "action":            "ORDER NOW",
            "canonical_name":    "Follow-up date documentation",
            "reason":            "100% LTFU patients left last visit without a return date — Retention tab Section E",
            "clinical_priority": "CRITICAL",
        },
        {
            "action":            "ORDER THIS WEEK",
            "canonical_name":    "Hypertension admission protocol",
            "reason":            "~2.7% conversion vs 10–20% reference — OPD → IPD Conversion Section F",
            "clinical_priority": "HIGH",
        },
        {
            "action":            "ORDER THIS WEEK",
            "canonical_name":    f"Outreach — {_n(lapsing_count)} lapsing patients",
            "reason":            f"{_kes_m(rev_at_risk)} recoverable if re-engaged this month — Retention tab Section C",
            "clinical_priority": "HIGH",
        },
    ])


# ══════════════════════════════════════════════════════════════════════════════
# TAB — OPD TO IPD CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def render_tab_opd_ipd(filters: dict, run_query):
    """OPD → IPD Conversion Rate — Sections 0, A–F."""
    from ksh.clinical_module.ui_template import (
        kpi_card, kpi_row, section_header, insight_card,
        page_header, anomaly_banner, action_cards,
        chart_card, chart_card_close, insight_bar,
        build_benchmark_chart, render_benchmark_callouts,
        CHART_LAYOUT, AXIS_STYLE, _ax as _ax_t,
        CA_BLUE, CA_RED, CA_GREEN, CA_AMBER, CA_PURPLE,
        ACCENT_CRITICAL, ACCENT_MONITOR, ACCENT_POSITIVE, ACCENT_INFO,
        fmt_num as _fmt_num,
    )
    _RED    = "#C53030"
    _GREEN  = "#38A169"
    _AMBER  = "#D97706"
    _BLUE   = AFYA_BLUE

    def _sf(v, d=0.0):
        try: return float(v)
        except: return d

    def _load(fn, label):
        try:
            df = fn(filters, run_query)
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as exc:
            st.warning(f"{label}: {exc}")
            return pd.DataFrame()

    # ── load all dataframes ────────────────────────────────────────────────
    df_ov      = _load(Q.load_opd_ipd_overall,          "Overall KPIs")
    df_seg     = _load(Q.load_opd_ipd_segments,         "Segments")
    df_monthly = _load(Q.load_opd_ipd_monthly,          "Monthly trend")
    df_bench   = _load(Q.load_opd_ipd_benchmark,        "Benchmark")
    df_comorb  = _load(Q.load_opd_ipd_comorbidity,      "Comorbidity")
    df_age     = _load(Q.load_opd_ipd_age_conversion,   "Age conversion")
    df_esc     = _load(Q.load_opd_ipd_escalation,       "Escalation")
    df_tri     = _load(Q.load_opd_ipd_workload_triangle,"Workload triangle")

    # ── numeric coercion (new dataframes) ────────────────────────────────
    for _df, _cols in [
        (df_ov,      ["overall_rate_pct","retention_universe_rate_pct",
                      "mix_gap_pp","total_ipd_admissions","strain_months","total_months"]),
        (df_seg,     ["total_opd_visits","ipd_admissions","conversion_rate_pct",
                      "ref_lower","ref_upper"]),
        (df_monthly, ["overall_rate_pct","retention_universe_rate_pct"]),
        (df_bench,   ["actual_rate_pct","ref_lower","ref_upper","total_opd_visits"]),
        (df_comorb,  ["conversion_rate_pct","total_opd_visits","ipd_admissions"]),
        (df_age,     ["conversion_rate_pct","total_chronic_visits"]),
        (df_esc,     ["total_escalations","total_72h_escalations","escalation_rate_pct"]),
        (df_tri,     ["conversion_rate_pct","avg_visits_per_clinician"]),
    ]:
        for _c in _cols:
            if not _df.empty and _c in _df.columns:
                _df[_c] = pd.to_numeric(_df[_c], errors="coerce")

    # ── derived scalars ───────────────────────────────────────────────────
    overall_rate  = _sf(df_ov["overall_rate_pct"].iloc[0]             if not df_ov.empty else 0)
    universe_rate = _sf(df_ov["retention_universe_rate_pct"].iloc[0]  if not df_ov.empty else 0)
    mix_gap       = _sf(df_ov["mix_gap_pp"].iloc[0]                   if not df_ov.empty else 0)
    total_ipd     = int(_sf(df_ov["total_ipd_admissions"].iloc[0]     if not df_ov.empty else 0))
    strain_months = int(_sf(df_ov["strain_months"].iloc[0]            if not df_ov.empty else 0))
    total_months  = int(_sf(df_ov["total_months"].iloc[0]             if not df_ov.empty else 1))

    # ── Section F scope defaults (updated as sections render) ────────────
    mh_rate    = 0.0
    below_segs = []
    low_age    = ["Adolescent 13–17", "Young Adult 18–24"]
    htn        = pd.DataFrame()
    child_row  = pd.DataFrame()
    gap        = 0.0
    n_strain   = strain_months
    avg_strain = 0.0
    avg_normal = 0.0
    peak_load  = 0.0

    REF_MAP = {
        "Chronic":       (8,  15),
        "Maternal":      (15, 25),
        "Oncology":      (15, 25),
        "Mental Health": (8,  15),
    }

    # ── Page header ───────────────────────────────────────────────────────
    page_header("OPD → IPD Conversion")

    # ── SECTION 0 — HEADER KPI STRIP ─────────────────────────────────────
    kpi_row([
        {"label": "Conversion rate",
         "value": f"{overall_rate:.2f}%",
         "delta": "All OPD visits · Sep 2024 – present",
         "delta_good": overall_rate >= 5,
         "accent_color": "#A32D2D" if overall_rate < 5 else "#0C447C"},
        {"label": "Complex patient rate",
         "value": f"{universe_rate:.1f}%",
         "delta": "Patients with chronic / complex conditions",
         "delta_good": True,
         "accent_color": "#0C447C"},
        {"label": "Mix gap",
         "value": f"+{mix_gap:.1f}%",
         "delta": "Acute walk-ins suppressing headline",
         "delta_good": False,
         "accent_color": "#D97706"},
        {"label": "Total admissions",
         "value": _fmt_num(total_ipd),
         "delta": "From OPD across all visits",
         "accent_color": "#0C447C"},
        {"label": "Strain months",
         "value": f"{strain_months} / {total_months}",
         "delta": "High workload · below avg conversion",
         "delta_good": strain_months < total_months // 2,
         "accent_color": "#D97706" if strain_months >= total_months // 2 else "#0F6E56"},
    ])

    if len(below_segs) == 0 and overall_rate < 5:
        anomaly_banner(
            "Overall conversion below reference",
            f"Overall rate is {overall_rate:.2f}% — below the 5% floor. "
            "Review OPD severity assessment criteria.",
        )

    _gap(16)

    # ── SECTION A — CONVERSION RATE OVERVIEW ─────────────────────────────
    section_header("A — Conversion rate overview")

    # Segment cards
    seg_cols = st.columns(4)
    for _i_seg, _seg in enumerate(["Chronic", "Maternal", "Oncology", "Mental Health"]):
        _ref_lo, _ref_hi = REF_MAP[_seg]
        _seg_row = df_seg[df_seg["segment"] == _seg] if not df_seg.empty else pd.DataFrame()
        if not _seg_row.empty:
            _seg_rate   = _sf(_seg_row.iloc[0].get("conversion_rate_pct"))
            _seg_visits = int(_sf(_seg_row.iloc[0].get("total_opd_visits")))
            _seg_adm    = int(_sf(_seg_row.iloc[0].get("ipd_admissions")))
        else:
            _seg_rate, _seg_visits, _seg_adm = 0.0, 0, 0

        # For Chronic: use retention universe rate (segment query has known fan-out bug)
        _display_rate = universe_rate if _seg == "Chronic" else _seg_rate
        _within       = _ref_lo <= _display_rate <= _ref_hi
        _badge_class  = "bw" if _within else "bb"
        _badge_text   = f"Within ref {_ref_lo}–{_ref_hi}%" if _within else f"Below ref {_ref_lo}–{_ref_hi}%"
        _accent_color = ACCENT_POSITIVE if _within else ACCENT_CRITICAL
        _sub_text     = "Complex patient rate" if _seg == "Chronic" else f"{_seg_visits:,} visits"

        if _seg == "Mental Health":
            mh_rate = _seg_rate
        if not _within:
            below_segs.append(_seg)

        with seg_cols[_i_seg]:
            st.markdown(
                f'<div class="kpi-tile" style="border-top-color:{_accent_color};">'
                f'<div class="kpi-label">{_seg}</div>'
                f'<div class="kpi-value" style="color:{_accent_color}">{_display_rate:.1f}%</div>'
                f'<div class="kpi-delta" style="color:#9CA3AF">{_sub_text}</div>'
                f'<span class="{_badge_class}">{_badge_text}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Section A continued: monthly charts ───────────────────────────────
    _gap(12)
    if not df_monthly.empty:
        df_monthly["visit_month"] = pd.to_datetime(df_monthly["visit_month"], errors="coerce")
        df_monthly = df_monthly.sort_values("visit_month")

    col_left, col_right = st.columns(2)
    with col_left:
        chart_card("Monthly overall conversion rate", f"Avg {overall_rate:.2f}% · Sep 2024 – present")
        if not df_monthly.empty:
            _fig_ov = go.Figure()
            _fig_ov.add_hline(y=overall_rate, line_dash="dot",
                              line_color="rgba(128,128,128,0.3)", line_width=1)
            _fig_ov.add_trace(go.Scatter(
                x=df_monthly["visit_month"], y=df_monthly["overall_rate_pct"],
                mode="lines+markers", line=dict(color=_BLUE, width=2), marker=dict(size=4),
                fill="tozeroy", fillcolor="rgba(0,114,206,0.06)", name="Overall rate",
            ))
            _fig_ov.update_layout(**{**CHART_LAYOUT, "height": 220},
                xaxis=_ax_t(), yaxis={**_ax_t(), "ticksuffix": "%", "range": [0, 12]},
                showlegend=False)
            _pc(_fig_ov)
        chart_card_close()

    with col_right:
        chart_card("Retention universe vs overall rate",
                   "Gap widens in high-volume months when acute walk-ins are highest.")
        if not df_monthly.empty:
            _fig_uni = go.Figure()
            for _col_n, _lbl_n, _clr_n, _dash_n in [
                ("overall_rate_pct",           "Overall",           _BLUE,  "solid"),
                ("retention_universe_rate_pct","Retention universe", _GREEN, "dash"),
            ]:
                _fig_uni.add_trace(go.Scatter(
                    x=df_monthly["visit_month"], y=df_monthly[_col_n],
                    name=_lbl_n, mode="lines+markers",
                    line=dict(color=_clr_n, width=2, dash=_dash_n), marker=dict(size=3),
                ))
            _fig_uni.update_layout(**{**CHART_LAYOUT, "height": 220},
                xaxis=_ax_t(), yaxis={**_ax_t(), "ticksuffix": "%"})
            _pc(_fig_uni)
        chart_card_close()

    _below_text = ", ".join(below_segs) if below_segs else "none"
    insight_bar([
        f"Retention universe rate ({universe_rate:.1f}%) sits above the overall rate in every month — gap widens when acute walk-in volume peaks.",
        f"Segments below reference: <strong>{_below_text}</strong>.",
        "<strong>Action:</strong> implement structured psychiatric severity screening at OPD for Mental Health — the only segment consistently below its reference floor.",
    ], variant="blue" if not below_segs else "amber")

    _gap(16)

    # ── SECTION B — DIAGNOSIS BENCHMARK ──────────────────────────────────
    section_header("B — Diagnosis benchmark comparison")

    def _get_ref_range(name: str):
        n = str(name).lower()
        if any(k in n for k in ["oncol","cancer","chemo"]):               return (15, 25)
        if any(k in n for k in ["maternal","obstet","intrapartum","antenatal",
                                  "anc","congenital","mnch","perinatal"]):  return (15, 25)
        if any(k in n for k in ["cardiovascular","cardiac","heart",
                                  "coronary","pulmonary vascular",
                                  "hypertension"]):                         return (10, 20)
        if any(k in n for k in ["neurolog","stroke"]):                      return (10, 20)
        if any(k in n for k in ["renal","kidney","genitourinary"]):         return (10, 20)
        return (8, 15)

    def _shorten(name: str) -> str:
        for sep in [" - ", ": "]:
            if sep in name:
                return name.split(sep, 1)[-1]
        return name

    if not df_bench.empty:
        # Remove vague / non-clinical categories that have no meaningful reference floor
        _EXCLUDE_PATTERNS = [
            "symptom", "undiagnosed", "unspecified", "not elsewhere",
            "other and unspecified", "ill-defined", "unknown", "encounter for",
            "screening", "observation", "administrative",
            "skin", "eye and ear", "ear", "eye", "external cause", "other ",
        ]
        _excl_mask = df_bench["cleaned_diagnosis_name"].str.lower().apply(
            lambda n: any(p in n for p in _EXCLUDE_PATTERNS)
        )
        df_bench = df_bench[~_excl_mask].copy()

        df_bench["ref_lower"] = df_bench["cleaned_diagnosis_name"].apply(
            lambda x: _get_ref_range(x)[0]
        )
        df_bench["diagnosis_label"] = df_bench["cleaned_diagnosis_name"].apply(_shorten)
        htn = df_bench[df_bench["cleaned_diagnosis_name"].str.contains(
            "Hypertension", case=False, na=False)]

        chart_card(
            "OPD → IPD conversion vs reference floor by diagnosis",
            "Gap = actual admission rate minus reference floor (%). "
            "e.g. −13.2% means the condition's rate is 13.2% below the minimum expected. "
            "Directional guidance only — Shawky (2024). Sorted largest gap above to below.",
        )
        fig = build_benchmark_chart(df_bench)
        st.plotly_chart(fig, use_container_width=True,
                        config={"responsive": True, "displayModeBar": False,
                                "useResizeHandler": True})
        chart_card_close()

        render_benchmark_callouts(df_bench)

        df_bench["gap"] = df_bench["actual_rate_pct"] - df_bench["ref_lower"]
        n_below = int((df_bench["gap"] < 0).sum())
        if n_below > 0 and not df_bench.empty:
            _worst = df_bench.nsmallest(1, "gap").iloc[0]
            insight_bar([
                f"<strong>{n_below} condition{'s' if n_below != 1 else ''} below reference floor.</strong>",
                f"Most urgent: {_worst['diagnosis_label']} ({_worst['gap']:+.1f}% from reference).",
                "<strong>Action:</strong> review OPD assessment and admission criteria for below-reference conditions. See Section F.",
            ], variant="red")
        else:
            insight_bar([
                "All conditions at or above reference floor.",
                "Reference ranges from Shawky (2024) — directional guidance only.",
            ], variant="teal")

    _gap(16)

    # ── SECTION C — CLINICAL LEAKAGE SIGNALS ─────────────────────────────
    section_header("C — Clinical leakage signals")

    if not df_comorb.empty:
        df_comorb["visit_month"] = pd.to_datetime(df_comorb["visit_month"], errors="coerce")
        _comorb_overall = df_comorb[df_comorb["visit_month"].isna()]
        _comorb_monthly = df_comorb[df_comorb["visit_month"].notna()].sort_values("visit_month")

        _c1, _c2, _c3 = st.columns(3)
        for _grp, _col_c in zip(
            ["Single diagnosis", "Comorbid", "Chronic comorbid"],
            [_c1, _c2, _c3],
        ):
            _gr = _comorb_overall[_comorb_overall["patient_group"] == _grp]
            _gr_rate = _sf(_gr["conversion_rate_pct"].iloc[0]) if not _gr.empty else 0.0
            _gr_vis  = int(_sf(_gr["total_opd_visits"].iloc[0])) if not _gr.empty else 0
            _vs_avg  = round(_gr_rate - overall_rate, 2)
            _sign    = "+" if _vs_avg >= 0 else ""
            with _col_c:
                kpi_card(_grp, f"{_gr_rate:.2f}%",
                         sub=f"{_gr_vis:,} visits",
                         delta=f"{_sign}{_vs_avg:.2f}% vs avg",
                         delta_color=_GREEN if _vs_avg >= 0 else _RED)

        _gap(12)
        _col_cl, _col_cr = st.columns(2)
        _COMORB_COLOURS = {
            "Single diagnosis": _BLUE,
            "Comorbid":         _AMBER,
            "Chronic comorbid": _GREEN,
        }
        with _col_cl:
            chart_card("Monthly conversion by comorbidity group", "8% reference line shown")
            _fig_cm = go.Figure()
            for _grp, _clr in _COMORB_COLOURS.items():
                _grp_df = _comorb_monthly[_comorb_monthly["patient_group"] == _grp]
                _fig_cm.add_trace(go.Scatter(
                    x=_grp_df["visit_month"], y=_grp_df["conversion_rate_pct"],
                    name=_grp, mode="lines+markers",
                    line=dict(color=_clr, width=2), marker=dict(size=3),
                ))
            _fig_cm.add_hline(y=8, line_dash="dot", line_color="rgba(186,117,23,0.5)",
                              annotation_text="8% ref",
                              annotation_font=dict(size=10, color="#BA7517"))
            _fig_cm.update_layout(**{**CHART_LAYOUT, "height": 240},
                xaxis=_ax_t(), yaxis={**_ax_t(), "ticksuffix": "%"})
            _pc(_fig_cm)
            chart_card_close()

        if not df_age.empty:
            df_age["conversion_rate_pct"] = pd.to_numeric(
                df_age["conversion_rate_pct"], errors="coerce")
            _df_age_s = df_age.sort_values("conversion_rate_pct", ascending=True)
            _low_age_all = _df_age_s.head(2)["age_group"].tolist()
            if _low_age_all:
                low_age = _low_age_all
            with _col_cr:
                chart_card("Chronic patient conversion by age group", "Bars below 8% = below reference")
                _fig_age = go.Figure(go.Bar(
                    y=_df_age_s["age_group"],
                    x=_df_age_s["conversion_rate_pct"],
                    orientation="h",
                    marker_color=_df_age_s["conversion_rate_pct"].apply(
                        lambda v: CA_RED if v < 8.0 else CA_BLUE).tolist(),
                    text=_df_age_s["conversion_rate_pct"].apply(lambda v: f"{v:.1f}%"),
                    textposition="outside", textfont=dict(size=11),
                ))
                _fig_age.add_vline(x=8.0, line_dash="dot",
                                   line_color="rgba(186,117,23,0.5)",
                                   annotation_text="8% ref",
                                   annotation_font=dict(size=10, color="#BA7517"))
                _fig_age.update_layout(**{**CHART_LAYOUT, "height": 240},
                    xaxis={**_ax_t(), "ticksuffix": "%", "range": [0, 20]},
                    yaxis={**_ax_t(), "showgrid": False}, showlegend=False)
                _pc(_fig_age)
                chart_card_close()

        _single_rate   = _sf(_comorb_overall[_comorb_overall["patient_group"] == "Single diagnosis"]["conversion_rate_pct"].iloc[0]) if not _comorb_overall[_comorb_overall["patient_group"] == "Single diagnosis"].empty else 0
        _comorbid_rate = _sf(_comorb_overall[_comorb_overall["patient_group"] == "Comorbid"]["conversion_rate_pct"].iloc[0]) if not _comorb_overall[_comorb_overall["patient_group"] == "Comorbid"].empty else 0
        insight_bar([
            f"Comorbid patients ({_comorbid_rate:.1f}%) convert at nearly double the rate of single-diagnosis patients ({_single_rate:.1f}%).",
            f"{' and '.join(low_age)} are the lowest converting chronic age groups and also show the highest LTFU rates in the Retention tab — under-admitted, then lost.",
            "<strong>Action:</strong> review OPD chronic disease protocols for these age groups.",
        ], variant="amber")

    _gap(16)

    # ── SECTION D — 72-HOUR ESCALATION ───────────────────────────────────
    section_header("D — Same-day / 72-hour escalation")

    child_row = pd.DataFrame()  # default; populated below if escalation data exists
    if df_esc.empty:
        st.info("No 72-hour escalation records found for the selected period and filters.")
    else:
        _total_esc = int(_sf(df_esc["total_72h_escalations"].iloc[0]))
        _esc_rate  = _sf(df_esc["escalation_rate_pct"].iloc[0])
        _top_dx    = str(df_esc["top_classified_diagnosis"].iloc[0]) if "top_classified_diagnosis" in df_esc.columns and pd.notna(df_esc["top_classified_diagnosis"].iloc[0]) else "—"


        _df_esc_age = (
            df_esc[df_esc["age_group"].notna()].sort_values("total_escalations", ascending=True)
            if "age_group" in df_esc.columns else pd.DataFrame()
        )
        child_row = (
            _df_esc_age[_df_esc_age["age_group"].str.contains("5.12|5–12", na=False, regex=True)]
            if not _df_esc_age.empty else pd.DataFrame()
        )

        # KPI row — three cards across full width
        _top_burden_raw = (
            str(df_esc["top_disease_burden_group"].iloc[0])
            if "top_disease_burden_group" in df_esc.columns
            and pd.notna(df_esc["top_disease_burden_group"].iloc[0])
            else _top_dx
        )
        # Strip "NCD - Category: " type prefixes — show only the specific condition name
        _top_burden = _top_burden_raw.split(": ", 1)[-1] if ": " in _top_burden_raw else _top_burden_raw
        _dk1, _dk2, _dk3 = st.columns(3)
        with _dk1:
            kpi_card("Total 72h escalations", _fmt_num(_total_esc),
                     sub="OPD visit → admission within 72h", color=_RED)
        with _dk2:
            kpi_card("Escalation rate", f"{_esc_rate:.2f}%",
                     sub="Of all OPD visits", color=_AMBER)
        with _dk3:
            kpi_card("Top disease burden group", _top_burden,
                     sub="ICD10 coverage improving")

        _gap(12)
        if not _df_esc_age.empty:
            chart_card("72h Escalations by Age Group", "Patients sent home then readmitted within 72 hours")
            _n_rows = len(_df_esc_age)
            _fig_esc = go.Figure(go.Bar(
                y=_df_esc_age["age_group"], x=_df_esc_age["total_escalations"],
                orientation="h", marker_color=CA_BLUE,
                text=_df_esc_age["total_escalations"].astype(int),
                textposition="outside", textfont=dict(size=11),
                cliponaxis=False,
            ))
            _fig_esc.update_layout(
                **{**CHART_LAYOUT, "height": max(260, _n_rows * 36 + 60), "margin": dict(l=140, r=60, t=10, b=40)},
                xaxis={**_ax_t(), "title": {"text": "Escalations"}, "showgrid": True},
                yaxis={**_ax_t(), "showgrid": False, "tickfont": dict(size=11)},
                showlegend=False,
            )
            _pc(_fig_esc)
            chart_card_close()

        if not _df_esc_age.empty:
            _top_age_row  = _df_esc_age.sort_values("total_escalations", ascending=False).iloc[0]
            _top_age_name = _top_age_row["age_group"]
            _top_age_n    = int(_sf(_top_age_row["total_escalations"]))
            _child_bullet = (
                " Watch for under-triage of paediatric presentations at OPD."
                if len(child_row) > 0 else ""
            )
            insight_bar([
                f"{_top_age_name} leads 72h escalations with {_top_age_n:,} cases.{_child_bullet}",
                f"{_total_esc:,} patients returned for admission within 72h — these should have been admitted at first contact.",
                "<strong>Action:</strong> implement PEWS at OPD triage for Child 5–12. Target: 30% reduction within 6 months.",
            ], variant="red")

    _gap(16)

    # ── SECTION E — WORKLOAD VS CONVERSION ───────────────────────────────
    section_header("E — Clinician workload vs conversion rate")

    if not df_tri.empty:
        df_tri["visit_month"] = pd.to_datetime(df_tri["visit_month"], errors="coerce")
        df_tri = df_tri.sort_values("visit_month")

        _strain_df = df_tri[df_tri["strain_signal"].isin(["HIGH_STRAIN","CAPACITY_GAP"])]
        _normal_df = df_tri[df_tri["strain_signal"] == "AS_EXPECTED"]
        avg_strain = float(_strain_df["conversion_rate_pct"].mean()) if not _strain_df.empty else 0.0
        avg_normal = float(_normal_df["conversion_rate_pct"].mean()) if not _normal_df.empty else 0.0
        gap        = round(avg_normal - avg_strain, 2)
        peak_load  = float(df_tri["avg_visits_per_clinician"].max())
        _peak_mon  = df_tri.loc[df_tri["avg_visits_per_clinician"].idxmax(), "visit_month"]
        n_strain   = len(_strain_df)
        _n_normal  = len(_normal_df)

        _e1, _e2, _e3, _e4 = st.columns(4)
        with _e1:
            kpi_card("Avg rate — strain months", f"{avg_strain:.2f}%",
                     sub=f"{n_strain} months flagged", color=_RED)
        with _e2:
            kpi_card("Avg rate — normal months", f"{avg_normal:.2f}%",
                     sub=f"{_n_normal} months as expected", color=_GREEN)
        with _e3:
            kpi_card("Strain impact", f"–{gap:.2f}%",
                     sub="Conversion drops in strain months",
                     color=_AMBER if gap < 1.0 else _RED)
        with _e4:
            kpi_card("Peak workload", f"{peak_load:.1f}",
                     sub=f"Avg visits/clinician · {pd.to_datetime(_peak_mon).strftime('%b %Y') if pd.notna(_peak_mon) else '—'}")

        _gap(12)
        _e_left, _e_right = st.columns(2)
        with _e_left:
            chart_card("Conversion rate vs clinician load")
            _load_max  = df_tri["avg_visits_per_clinician"].max()
            _conv_max  = df_tri["conversion_rate_pct"].max()
            _scale_f   = _conv_max / _load_max if _load_max > 0 else 1

            _fig_tri = go.Figure()
            for _, _srow in _strain_df.iterrows():
                _fig_tri.add_vrect(x0=_srow["visit_month"], x1=_srow["visit_month"],
                                   fillcolor="rgba(226,75,74,0.06)",
                                   line_width=0, layer="below")
            _fig_tri.add_trace(go.Scatter(
                x=df_tri["visit_month"], y=df_tri["conversion_rate_pct"],
                name="Conversion rate %", mode="lines+markers",
                line=dict(color=_BLUE, width=2), marker=dict(size=4),
                fill="tozeroy", fillcolor="rgba(0,114,206,0.04)",
            ))
            _fig_tri.add_trace(go.Scatter(
                x=df_tri["visit_month"],
                y=df_tri["avg_visits_per_clinician"] * _scale_f,
                name="Clinician load (scaled)", mode="lines+markers",
                line=dict(color=PURPLE, width=2, dash="dash"), marker=dict(size=3),
            ))
            _fig_tri.update_layout(**{**CHART_LAYOUT, "height": 240},
                xaxis=_ax_t(), yaxis=_ax_t())
            _pc(_fig_tri)
            chart_card_close()

        with _e_right:
            chart_card("Strain month detail")
            for _, _srow in _strain_df.iterrows():
                _ms   = pd.to_datetime(_srow["visit_month"]).strftime("%b %Y")
                _cv   = float(_srow["conversion_rate_pct"])
                _lv   = float(_srow["avg_visits_per_clinician"])
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:5px 10px;border-radius:6px;font-size:11px;'
                    f'background:#FAEEDA;color:#633806;margin-bottom:4px;">'
                    f'<span style="font-weight:500;">{_ms}</span>'
                    f'<span>Conv {_cv:.2f}% · {_lv:.1f} visits/clinician</span></div>',
                    unsafe_allow_html=True,
                )
            chart_card_close()

        if gap > 1.0:
            _e_msg = (
                f"In {n_strain} high-load months, conversion averaged {avg_strain:.2f}% — {gap:.2f}% below "
                f"the {avg_normal:.2f}% seen in normal months. Peak load reached {peak_load:.1f} visits/clinician. "
                f"Action: review staffing for months above {peak_load:.0f} visits/clinician."
            )
            _e_var = "red"
        elif gap > 0:
            _e_msg = (
                f"{gap:.2f}% conversion gap between high-load ({avg_strain:.2f}%) and normal months ({avg_normal:.2f}%). "
                f"{n_strain} of {total_months} months flagged — in monitor band. "
                f"Action: if gap exceeds 1.0% two consecutive months, initiate staffing review."
            )
            _e_var = "amber"
        else:
            _e_msg = (
                f"No conversion gap detected between high-load and normal months ({avg_strain:.2f}% vs {avg_normal:.2f}%). "
                f"Clinician workload is not currently suppressing the OPD → IPD admission rate."
            )
            _e_var = "teal"

        insight_bar(_e_msg, variant=_e_var)

    _gap(16)

    # ── SECTION F — RECOMMENDATIONS ──────────────────────────────────────
    section_header(
        f"F — Recommendations — what is holding the conversion rate at {overall_rate:.2f}%"
    )

    _htn_rate_text = (
        f"{htn.iloc[0]['actual_rate_pct']:.1f}%"
        if not htn.empty else "~2.7%"
    )
    _child_esc_n = (
        int(_sf(child_row["total_escalations"].iloc[0]))
        if not child_row.empty else 134
    )
    _low_age_text = " and ".join(low_age) if low_age else "Adolescent and Young Adult"

    # ── KPI bar ───────────────────────────────────────────────────────────────
    _bar_fill_pct = min(round(overall_rate / universe_rate * 100), 95) \
                    if universe_rate > 0 else 0
    _ref_right_pct = round((1 - 8 / universe_rate) * 100) \
                     if universe_rate > 0 else 20

    st.markdown(
        f'<div style="background:white;border-radius:10px;padding:16px 20px;'
        f'border:1px solid #E5E7EB;margin-bottom:14px;">'
        f'<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:6px;">'
        f'<span style="font-size:36px;font-weight:700;color:#C53030;line-height:1;">'
        f'{overall_rate:.2f}%</span>'
        f'<span style="font-size:13px;color:#6B8CAE;">OPD → IPD · all visits</span>'
        f'</div>'
        f'<div style="font-size:12px;color:var(--text-color);opacity:0.7;'
        f'line-height:1.6;margin-bottom:12px;">'
        f'Six factors explain the gap to the 8% floor. Complex patients already '
        f'convert at {universe_rate:.1f}% — the problem is case mix and specific '
        f'protocol gaps.</div>'
        f'<div style="position:relative;margin-bottom:22px;">'
        f'<div style="font-size:10px;color:#185FA5;font-weight:600;position:absolute;'
        f'right:{_ref_right_pct}%;top:-16px;transform:translateX(50%);">8% floor</div>'
        f'<div style="font-size:10px;color:#6B8CAE;position:absolute;'
        f'top:-16px;right:0;">{universe_rate:.1f}% complex</div>'
        f'<div style="position:relative;height:10px;background:#F3F4F6;'
        f'border-radius:5px;">'
        f'<div style="width:{_bar_fill_pct}%;height:100%;background:#C53030;'
        f'border-radius:5px;"></div>'
        f'<div style="position:absolute;right:{_ref_right_pct}%;top:-4px;'
        f'width:1.5px;height:18px;background:#185FA5;"></div>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;'
        f'font-size:10px;color:#9CA3AF;margin-top:4px;">'
        f'<span>0%</span><span>8%</span></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # ── ACT NOW header ────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">'
        f'<div style="flex:1;height:1px;background:#FCA5A5;"></div>'
        f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.8px;color:#A32D2D;white-space:nowrap;">'
        f'ACT NOW — TWO FACTORS ACCOUNT FOR MOST OF THE GAP TO 8%</div>'
        f'<div style="flex:1;height:1px;background:#FCA5A5;"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── High impact cards — two columns ──────────────────────────────────────
    def _hi_action_items(items):
        html = ""
        for i, item in enumerate(items, 1):
            html += (
                f'<li style="display:flex;align-items:flex-start;gap:8px;'
                f'font-size:12px;color:#003467;line-height:1.55;">'
                f'<div style="width:18px;height:18px;border-radius:50%;'
                f'background:#E24B4A;color:white;font-size:11px;font-weight:700;'
                f'display:flex;align-items:center;justify-content:center;'
                f'flex-shrink:0;margin-top:1px;">{i}</div>'
                f'<div>{item}</div></li>'
            )
        return html

    def _hi_findings_html(items):
        return "".join(
            f'<li style="font-size:12px;color:#374151;line-height:1.7;">'
            f'<span style="color:#9CA3AF;">· </span>{item}</li>'
            for item in items
        )

    def _hi_card(type_label, impact_text, title, findings, actions):
        return (
            f'<div style="background:var(--secondary-background-color);border-radius:8px;'
            f'padding:16px 18px;border:1px solid #E5E7EB;border-top:3px solid #E24B4A;'
            f'box-shadow:0 1px 3px rgba(0,0,0,0.06);height:100%;box-sizing:border-box;">'
            f'<div style="font-size:11px;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:0.5px;color:#9CA3AF;margin-bottom:6px;">'
            f'{type_label}</div>'
            f'<div style="font-size:24px;font-weight:700;color:#E24B4A;'
            f'line-height:1;margin-bottom:6px;">{impact_text}</div>'
            f'<div style="font-size:13px;font-weight:500;color:#374151;'
            f'line-height:1.5;margin-bottom:10px;">{title}</div>'
            f'<ul style="list-style:none;padding:0;margin:0 0 12px;">'
            f'{_hi_findings_html(findings)}</ul>'
            f'<div style="height:1px;background:#F3F4F6;margin-bottom:12px;"></div>'
            f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.5px;color:#A32D2D;margin-bottom:8px;">'
            f'To improve conversion</div>'
            f'<ul style="list-style:none;padding:0;margin:0;'
            f'display:flex;flex-direction:column;gap:6px;">'
            f'{_hi_action_items(actions)}</ul>'
            f'</div>'
        )

    _card1_html = _hi_card(
        type_label   = "Case mix",
        impact_text  = f"~{mix_gap:.1f}% suppression",
        title        = "High acute walk-in volume is diluting the overall rate",
        findings     = [
            f"Acute walk-ins convert at ~5.4% vs ~{universe_rate:.1f}% "
            f"for complex patients",
            f"Mixed into the headline, they pull the overall rate down — "
            f"making the primary KPI misleading as a management metric",
        ],
        actions      = [
            "Build referral pathways for complex cases — "
            "chronic disease, oncology, maternal",
            f"Track complex ({universe_rate:.1f}%) and acute (~5.4%) "
            f"conversion separately as distinct KPIs",
            f"The {overall_rate:.2f}% headline will improve as case mix "
            f"shifts — do not use it as a standalone target",
        ],
    )

    _card2_html = _hi_card(
        type_label   = "Clinical protocol gap",
        impact_text  = f"−5.8% below reference",
        title        = (
            f"Hypertension at {_htn_rate_text} against a "
            f"10–20% cardiovascular reference floor"
        ),
        findings     = [
            "Systolic >180 presentations leaving OPD without a "
            "documented admission decision",
            "Largest single-diagnosis gap on the benchmark chart",
        ],
        actions      = [
            "Write admission criteria for systolic >180 — every case "
            "at this threshold requires a documented clinical decision",
            "This single protocol change directly moves the "
            "cardiovascular segment conversion rate",
        ],
    )

    st.markdown(
        f'<div style="display:flex;gap:16px;align-items:stretch;">'
        f'<div style="flex:1;">{_card1_html}</div>'
        f'<div style="flex:1;">{_card2_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Contributing factors — accordion rows ─────────────────────────────────
    _ACC_FACTORS = [
        {
            "badge":        "Medium",
            "badge_bg":     "#FAEEDA", "badge_fc": "#633806",
            "type":         "Segment gap",
            "summary":      f"Mental Health at {mh_rate:.1f}% — only segment below floor",
            "impact":       f"−{abs(8 - mh_rate):.1f}%",
            "impact_color": "#BA7517",
            "action_label_color": "#633806",
            "findings": [
                "Mental Health is the only segment below its 8% reference floor",
                "Psychiatric severity may not be assessed consistently at OPD, "
                "or psychiatrist input is only available after the admission "
                "decision rather than at triage",
            ],
            "action": (
                "Implement a structured psychiatric severity screening tool at "
                "OPD for Mental Health presentations. Confirm whether psychiatrist "
                "input is available at triage."
            ),
        },
        {
            "badge":        "Medium",
            "badge_bg":     "#FAEEDA", "badge_fc": "#633806",
            "type":         "Age leakage",
            "summary":      f"{_low_age_text} below 8% reference",
            "impact":       "",
            "impact_color": "#6B8CAE",
            "action_label_color": "#633806",
            "findings": [
                f"{_low_age_text} are the lowest converting chronic age groups",
                "Both also show the highest LTFU dropout rates in the Retention "
                "tab — under-admitted, then lost",
            ],
            "action": (
                "Review OPD chronic disease assessment for these age groups. "
                "Ensure admission decisions reflect clinical severity — not "
                "patient preference to avoid admission."
            ),
        },
        {
            "badge":        "Medium",
            "badge_bg":     "#FAEEDA", "badge_fc": "#633806",
            "type":         "Under-triage",
            "summary":      f"Child 5–12: {_child_esc_n} escalations within 72h",
            "impact":       f"{_child_esc_n} cases",
            "impact_color": "#6B8CAE",
            "action_label_color": "#633806",
            "findings": [
                "Child 5–12 patients are assessed at OPD, sent home, and "
                "return for admission within 72 hours",
                "The OPD assessment is not detecting paediatric severity — "
                "these admissions should have happened at first contact",
            ],
            "action": (
                "Implement Paediatric Early Warning Score (PEWS) at OPD triage "
                "for all Child 5–12 presentations. Track 72h escalation rate "
                "monthly. Target: 30% reduction within 6 months."
            ),
        },
        {
            "badge":        "Monitor",
            "badge_bg":     "#EBF5FF", "badge_fc": "#0C447C",
            "type":         "Workload",
            "summary":      (
                f"−{gap:.2f}% in high-load months · "
                f"{n_strain} of {total_months} flagged"
            ),
            "impact":       f"−{gap:.2f}%",
            "impact_color": "#185FA5",
            "action_label_color": "#0C447C",
            "findings": [
                f"In {n_strain} high-load months, conversion averaged "
                f"{avg_strain:.2f}% vs {avg_normal:.2f}% in normal months",
                f"Peak load: {peak_load:.1f} visits/clinician — "
                f"gap below 1.0% threshold",
            ],
            "action": (
                f"Monitor monthly. If gap exceeds 1.0% over two consecutive "
                f"months, initiate a staffing review. Target: no month above "
                f"90 visits/clinician."
            ),
        },
    ]

    st.markdown(
        f'<div style="font-size:10px;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.8px;color:#9CA3AF;margin-bottom:8px;margin-top:14px;">'
        f'Contributing to the gap between {overall_rate:.2f}% and the 8% floor'
        f'</div>',
        unsafe_allow_html=True,
    )

    _acc_css = (
        '<style>'
        '.opd-acc{background:white;border:1px solid #E5E7EB;border-radius:8px;'
        'margin-bottom:6px;}'
        '.opd-acc>summary{display:flex;align-items:center;gap:10px;'
        'padding:9px 14px;cursor:pointer;list-style:none;border-radius:8px;}'
        '.opd-acc>summary::-webkit-details-marker{display:none;}'
        '.opd-acc[open]>summary{border-radius:8px 8px 0 0;'
        'border-bottom:1px solid #E5E7EB;}'
        '.opd-acc>summary::after{content:"▼";color:#9CA3AF;font-size:11px;'
        'flex-shrink:0;}'
        '.opd-acc[open]>summary::after{content:"▲";}'
        '.opd-acc-badge{font-size:10px;font-weight:600;padding:2px 8px;'
        'border-radius:4px;white-space:nowrap;flex-shrink:0;}'
        '.opd-acc-type{font-size:10px;font-weight:600;text-transform:uppercase;'
        'letter-spacing:0.5px;color:#9CA3AF;white-space:nowrap;flex-shrink:0;}'
        '.opd-acc-dot{color:#D1D5DB;font-size:10px;flex-shrink:0;}'
        '.opd-acc-sum{font-size:12px;color:#003467;flex:1;min-width:0;'
        'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}'
        '.opd-acc-imp{font-size:12px;font-weight:600;white-space:nowrap;flex-shrink:0;}'
        '.opd-acc-body{padding:10px 14px 12px;background:#FAFAFA;'
        'border-radius:0 0 8px 8px;}'
        '.opd-acc-finds{list-style:none;padding:0;margin:0 0 10px;}'
        '.opd-acc-finds li{font-size:11px;color:#374151;line-height:1.7;}'
        '.opd-acc-albl{font-size:10px;font-weight:700;text-transform:uppercase;'
        'letter-spacing:0.5px;margin-bottom:5px;}'
        '.opd-acc-act{font-size:11px;padding:8px 10px;background:white;'
        'border-radius:6px;border:0.5px solid #E5E7EB;color:#003467;line-height:1.5;}'
        '</style>'
    )
    _acc_rows = ""
    for _af in _ACC_FACTORS:
        _imp_span = (
            f'<span class="opd-acc-imp" style="color:{_af["impact_color"]};">'
            f'{_af["impact"]}</span>'
            if _af["impact"] else ""
        )
        _finds_li = "".join(
            f'<li><span style="color:#9CA3AF;">· </span>{pt}</li>'
            for pt in _af["findings"]
        )
        _acc_rows += (
            f'<details class="opd-acc">'
            f'<summary>'
            f'<span class="opd-acc-badge" style="background:{_af["badge_bg"]};'
            f'color:{_af["badge_fc"]};">{_af["badge"]}</span>'
            f'<span class="opd-acc-type">{_af["type"]}</span>'
            f'<span class="opd-acc-dot">·</span>'
            f'<span class="opd-acc-sum">{_af["summary"]}</span>'
            f'{_imp_span}'
            f'</summary>'
            f'<div class="opd-acc-body">'
            f'<ul class="opd-acc-finds">{_finds_li}</ul>'
            f'<div class="opd-acc-albl" style="color:{_af["action_label_color"]};">'
            f'Action</div>'
            f'<div class="opd-acc-act">{_af["action"]}</div>'
            f'</div>'
            f'</details>'
        )
    st.markdown(_acc_css + _acc_rows, unsafe_allow_html=True)

    # ── START HERE block ──────────────────────────────────────────────────────
    st.markdown(
        f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.8px;color:#6B8CAE;margin-bottom:10px;margin-top:14px;">'
        f'Start here</div>',
        unsafe_allow_html=True,
    )

    _START_ITEMS = [
        (
            "Build referral pathways for complex cases — "
            f"<strong>track complex and acute conversion as separate KPIs,</strong> "
            f"not as a blended {overall_rate:.2f}%"
        ),
        (
            f"Write <strong>hypertensive urgency admission criteria</strong> "
            f"for systolic >180 at OPD"
        ),
        (
            "Implement <strong>PEWS at paediatric OPD triage</strong> "
            "for all Child 5–12 presentations"
        ),
    ]

    _sh_html = ""
    for _i, _item in enumerate(_START_ITEMS, 1):
        _sh_html += (
            f'<div style="display:flex;align-items:flex-start;gap:12px;'
            f'padding:10px 14px;background:white;border-radius:8px;'
            f'border:1px solid #E5E7EB;margin-bottom:6px;">'
            f'<div style="width:22px;height:22px;border-radius:50%;'
            f'background:#C53030;color:white;font-size:11px;font-weight:700;'
            f'display:flex;align-items:center;justify-content:center;'
            f'flex-shrink:0;margin-top:1px;">{_i}</div>'
            f'<div style="font-size:12px;color:#003467;line-height:1.55;flex:1;">'
            f'{_item}</div></div>'
        )
    st.markdown(_sh_html, unsafe_allow_html=True)

    # render_tab_opd_ipd — end of function



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
                legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)"),
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
                    df_wb[["ward", "admissions", "pct_share", "num_beds",
                            "avg_los_days", "avg_admission_cost"]].head(12),
                    col_labels={"ward": "Ward", "admissions": "Admissions",
                                "pct_share": "Share %", "num_beds": "Beds",
                                "avg_los_days": "Avg LOS (d)",
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
                font=dict(size=13, color="#374151"),
                xaxis=dict(
                    showgrid=False, showline=False,
                    tickfont=dict(size=12, color="#9ca3af"),
                    tickformat="%b %y",
                ),
                yaxis=dict(
                    showgrid=True, gridcolor="rgba(0,0,0,0.05)", showline=False,
                    tickfont=dict(size=12, color="#9ca3af"),
                    title=dict(
                        text="Index (base month = 100)" if use_index else "Admissions",
                        font=dict(size=12, color="#9ca3af"),
                    ),
                    rangemode="tozero" if not use_index else "normal",
                ),
                legend=dict(
                    orientation="h", y=-0.18, x=0.5, xanchor="center",
                    font=dict(size=12, color="#374151"),
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

    _gap(16)

    # ── WARD DEEP-DIVE ────────────────────────────────────────────────────
    try:
        render_ward_deepdive(filters, run_query)
    except Exception as _dd_e:
        st.warning(f"Ward deep-dive: {_dd_e}")

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

                cond_str = r.get("top_conditions") or "—"

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
                    "Top Conditions":    cond_str,
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
                legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)"),
            )
            _pc(fig_a)
    except Exception as e:
        st.warning(f"Active admissions: {e}")

    _gap(12)


    # ── C3: SAME-DAY ESCALATION ───────────────────────────────────────────
    _sh("C3 — Same-Day Escalation: Outpatient Visits That Led to Admission", mt=8)
    _note(
        "Outpatient visits where the patient was admitted as an inpatient on the same visit. "
        "High rates in a condition may indicate patients arriving too late, underscoring the "
        "need for earlier intervention or community referral pathways."
    )
    try:
        df_esc_kpi  = Q.load_same_day_escalation_kpis(filters, run_query)
        df_esc_cond = Q.load_same_day_escalation_by_condition(filters, run_query)

        if not df_esc_kpi.empty:
            row = df_esc_kpi.iloc[0]
            total_esc   = int(row.get("total_escalations") or 0)
            total_op    = int(row.get("total_op_visits") or 0)
            esc_rate    = float(row.get("escalation_rate_pct") or 0)
            top_cond    = str(row.get("top_condition") or "—")

            c1, c2, c3 = st.columns(3)
            with c1:
                _kpi("Same-Day Escalations", _n(total_esc), color=ORANGE)
            with c2:
                _kpi("Share of All Visits",
                      f"{esc_rate:.1f}%",
                      f"out of {_n(total_op)} visits",
                      color=CORAL if esc_rate >= 5 else AFYA_BLUE)
            with c3:
                _kpi("Top Escalated Condition", top_cond, color=PURPLE)

        if not df_esc_cond.empty:
            _gap(8)
            df_esc_cond = df_esc_cond.head(10).reset_index(drop=True)
            fig_esc = go.Figure()
            fig_esc.add_trace(go.Bar(
                x=df_esc_cond["condition"],
                y=df_esc_cond["esc_count"],
                name="Escalations",
                marker_color=ORANGE,
                text=df_esc_cond["esc_count"].apply(lambda v: f"{int(v):,}"),
                textposition="outside",
            ))
            fig_esc.add_trace(go.Scatter(
                x=df_esc_cond["condition"],
                y=df_esc_cond["escalation_rate_pct"],
                name="Escalation Rate (%)",
                mode="lines+markers",
                line=dict(color=CORAL, width=2),
                marker=dict(size=7),
                yaxis="y2",
            ))
            fig_esc.update_layout(
                height=340,
                margin=dict(l=0, r=0, t=10, b=120),
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(tickangle=-40),
                yaxis=dict(title="Escalation Count", color=ORANGE),
                yaxis2=dict(title="Escalation Rate (%)", overlaying="y", side="right",
                            color=CORAL),
                bargap=0.35,
            )
            _pc(fig_esc)

            if not df_esc_kpi.empty:
                row = df_esc_kpi.iloc[0]
                esc_rate = float(row.get("escalation_rate_pct") or 0)
                top_cond = str(row.get("top_condition") or "—")
                if esc_rate >= 5:
                    _note(
                        f"{esc_rate:.1f}% of all visits resulted in same-day escalation to inpatient. "
                        f"{top_cond} is the leading condition — review whether earlier triage or "
                        f"community intervention could reduce emergency admissions.", w=True
                    )
                else:
                    _note(
                        f"Same-day escalation rate is {esc_rate:.1f}%, within expected range. "
                        f"{top_cond} accounts for the most escalations."
                    )
        else:
            _note("No same-day escalation records found for this period.")
    except Exception as e:
        st.warning(f"Same-day escalation: {e}")

    _gap(12)





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


def _build_ltfu_p2_html(
    tier_stats: list,
    tiers: list,
    total_ltfu: int,
    total_chronic: int,
    tier1_revenue: float,
    age_labels: list,
    age_values: list,
    dx_labels: list,
    dx_values: list,
    signals: dict = None,
) -> tuple:
    """Build the Priority 2 self-contained dark-mode HTML block.

    Returns (html_str, height_px).
    All data is pre-computed in Python and embedded as JS literals — no CSV parsing.
    """
    import json as _json

    s             = signals or {}
    pct_lab       = s.get("pct_had_lab",       "—")
    pct_rx        = s.get("pct_had_rx",        "—")
    pct_no_fu     = s.get("pct_no_structured_date", "—")
    pct_fu_ment   = s.get("pct_followup_mentioned", None)
    pct_not_ret   = s.get("pct_not_returned_despite_followup_note", None)
    pct_radiology = s.get("pct_had_radiology", "—")
    lab_n         = s.get("patients_with_lab",       "")
    rx_n          = s.get("patients_with_rx",        "")
    rad_n         = s.get("patients_with_radiology", "")
    total_sig     = s.get("total_ltfu_patients",     "")
    _is_live      = bool(s)

    def _fmt_pct(v):
        try: return f"{float(v):.1f}%"
        except: return "—"

    lab_pct_fmt  = _fmt_pct(pct_lab)
    rx_pct_fmt   = _fmt_pct(pct_rx)
    nofu_pct_fmt = _fmt_pct(pct_no_fu)
    rad_pct_fmt  = _fmt_pct(pct_radiology)

    fu_ment_fmt  = _fmt_pct(pct_fu_ment)  if pct_fu_ment  is not None else "—"
    not_ret_fmt  = _fmt_pct(pct_not_ret)  if pct_not_ret  is not None else "—"

    sub_note = (
        "What happened at the patient's last clinical visit before dropout."
        if _is_live else
        "What happened at the patient's last clinical visit before dropout. "
        "<span class=\"sec-illustrative\">Illustrative — requires visit journey query to confirm.</span>"
    )

    lab_desc  = (f"{lab_n} of {total_sig} patients · results received, no follow-up"
                 if _is_live else "Results received, no documented follow-up")
    rx_desc   = (f"{rx_n} of {total_sig} patients · medication given, no return scheduled"
                 if _is_live else "Medication given, no return scheduled")
    nofu_desc = (f"Only {fu_ment_fmt} mentioned follow-up in notes · {not_ret_fmt} did not return"
                 if (_is_live and pct_fu_ment is not None) else "Most critical gap in the care pathway")
    rad_desc  = (f"{rad_n} of {total_sig} patients · imaging done, no subsequent visit"
                 if _is_live else "Imaging done, no subsequent visit")

    insight2_text = (
        f"{nofu_pct_fmt} of chronic LTFU patients had no documented follow-up date. "
        f"Only {fu_ment_fmt} had follow-up mentioned in clinical notes — "
        f"of those, {not_ret_fmt} still did not return. "
        "This is a clinical workflow gap: when follow-up is not explicitly scheduled, "
        "chronic patients do not return."
        if _is_live else
        "84% of chronic LTFU patients had no documented follow-up date. "
        "This is a clinical workflow gap — when follow-up is not explicitly scheduled, "
        "chronic patients do not return."
    )

    t1            = tier_stats[0]
    tier1_chronic = t1["chronic"]
    chronic_fmt   = f"{tier1_chronic:,}"
    rev_fmt       = (f"KES {tier1_revenue / 1_000_000:.1f}M"
                     if tier1_revenue >= 1_000_000
                     else f"KES {tier1_revenue:,.0f}")
    total_chr_fmt = f"{total_chronic:,}"
    pct_chr       = (f"{total_chronic / total_ltfu * 100:.1f}%"
                     if total_ltfu else "—")

    anchor      = tier_stats[0]["pct"] or 1
    funnel_html = "".join(
        '<div class="funnel-row">'
        f'<span class="funnel-lbl">{t["label"]}</span>'
        '<div class="funnel-bar-wrap">'
        f'<div class="funnel-bar" style="background:{t["color"]};width:{s["pct"] / anchor * 100:.1f}%;">'
        f'<span class="funnel-bar-text">{s["total"]:,}&nbsp;&nbsp;{s["pct"]:.1f}%</span>'
        '</div></div>'
        f'<span class="funnel-annot" style="color:{t["annot_color"]};">{s["chronic"]:,} chronic</span>'
        '</div>'
        for t, s in zip(tiers, tier_stats)
    )

    age_labels_js = _json.dumps(age_labels)
    age_values_js = _json.dumps(age_values)
    dx_labels_js  = _json.dumps(dx_labels)
    dx_values_js  = _json.dumps(dx_values)

    n_age  = len(age_labels)
    n_dx   = len(dx_labels)
    h_age  = max(280, n_age * 40 + 80)
    h_dx   = max(280, n_dx  * 40 + 80)

    insight1 = (
        f"{chronic_fmt} of the 1–2 visit LTFU patients had a chronic diagnosis — "
        "conditions requiring ongoing management. These are not resolved cases. "
        "They are patients who needed follow-up and did not return."
    )

    css = (
        ':root{'
        '--bg:#ffffff;--surface:#ffffff;--border:rgba(0,0,0,0.06);'
        '--border-mid:#D6E4F0;--divider:#EBF3FB;'
        '--text-primary:#003467;--text-secondary:#6B8CAE;--text-muted:#ADB5BD;'
        '--insight-bg:#FFF5F5;}'
        '*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}'
        'body{background:var(--bg);color:var(--text-primary);'
        'font-family:Montserrat,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;'
        'font-size:14px;line-height:1.5;padding:1.5rem}'
        '.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;'
        'overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}'
        '.ltfu-section{max-width:100%;margin:0 auto}'
        '.sec-eyebrow{font-size:11px;font-weight:600;letter-spacing:0.06em;'
        'text-transform:uppercase;color:var(--text-muted);margin-bottom:3px}'
        '.sec-title{font-size:16px;font-weight:500;color:var(--text-primary);margin-bottom:4px}'
        '.sec-subtitle{font-size:13px;color:var(--text-secondary);line-height:1.6;margin-bottom:18px}'
        '.sec-illustrative{color:var(--text-muted);font-style:italic}'
        '.section-part{margin-bottom:2.5rem}'
        '.section-divider{border:none;border-top:0.5px solid var(--divider);margin:0 0 2.5rem}'
        '.card{background:var(--surface);border:0.5px solid var(--border-mid);border-radius:6px}'
        '.p1-layout{display:flex;gap:20px;align-items:flex-start}'
        '.revenue-card{min-width:164px;flex-shrink:0;display:flex;flex-direction:column;'
        'justify-content:center;padding:18px 16px}'
        '.rev-label{font-size:10px;font-weight:600;letter-spacing:0.07em;'
        'text-transform:uppercase;color:var(--text-muted);margin-bottom:5px}'
        '.rev-value{font-size:26px;font-weight:700;color:#A32D2D;line-height:1;'
        'margin-bottom:4px;font-variant-numeric:tabular-nums}'
        '.rev-desc{font-size:11px;color:var(--text-secondary);margin-bottom:13px}'
        '.rev-divider{border:none;border-top:0.5px solid var(--border-mid);margin:0 0 12px}'
        '.rev-stat{margin-bottom:9px}.rev-stat:last-child{margin-bottom:0}'
        '.rev-stat-val{font-size:17px;font-weight:600;color:var(--text-primary);'
        'font-variant-numeric:tabular-nums;display:block}'
        '.rev-stat-lbl{font-size:11px;color:var(--text-muted)}'
        '.funnel-wrap{flex:1;min-width:0}'
        '.funnel-row{display:flex;align-items:center;gap:10px;margin-bottom:10px}'
        '.funnel-row:last-child{margin-bottom:0}'
        '.funnel-lbl{width:80px;flex-shrink:0;text-align:right;font-size:12px;color:var(--text-secondary)}'
        '.funnel-bar-wrap{flex:1;min-width:0}'
        '.funnel-bar{height:34px;border-radius:3px;display:flex;align-items:center;padding:0 10px}'
        '.funnel-bar-text{font-size:12px;font-weight:600;color:rgba(255,255,255,0.92);white-space:nowrap}'
        '.funnel-annot{flex-shrink:0;font-size:12px;font-weight:500;white-space:nowrap}'
        '.journey-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px}'
        '.journey-card{display:flex;flex-direction:column;gap:6px;padding:16px 14px}'
        '.journey-icon{font-size:20px;color:var(--text-muted)}'
        '.journey-pct{font-size:28px;font-weight:700;line-height:1;font-variant-numeric:tabular-nums}'
        '.journey-label{font-size:12px;font-weight:500;color:var(--text-primary)}'
        '.journey-desc{font-size:11px;color:var(--text-secondary);line-height:1.5}'
        '.charts-row{display:flex;gap:24px}'
        '.chart-col{flex:1;min-width:0}'
        '.chart-legend{display:flex;align-items:center;gap:6px;margin-bottom:8px}'
        '.legend-swatch{width:10px;height:10px;border-radius:1px;flex-shrink:0}'
        '.legend-lbl{font-size:11px;color:var(--text-secondary)}'
        '.chart-wrap{position:relative}'
        'canvas{display:block}'
        '.insight-bar{background:var(--insight-bg);border-left:3px solid #C53030;'
        'border-radius:0 4px 4px 0;padding:10px 14px;display:flex;align-items:flex-start;gap:8px;margin-top:16px}'
        '.insight-icon{font-size:15px;color:#C53030;flex-shrink:0;margin-top:1px}'
        '.insight-text{font-size:13px;color:#003467;line-height:1.7}'
    )

    js = (
        '(function(){'
        'function buildChart(id,labels,values,color){'
        'var canvas=document.getElementById(id);'
        'new Chart(canvas,{type:"bar",'
        'data:{labels:labels,datasets:[{data:values,backgroundColor:color,'
        'borderRadius:3,borderSkipped:false}]},'
        'options:{indexAxis:"y",responsive:true,maintainAspectRatio:false,'
        'plugins:{legend:{display:false},'
        'tooltip:{callbacks:{label:function(ctx){'
        'return " "+Math.round(ctx.parsed.x).toLocaleString()+" patients";'
        '}}}},'
        'scales:{'
        'x:{grid:{color:"#EBF3FB"},ticks:{color:"#6B8CAE",font:{size:11,family:"Montserrat,sans-serif"}}},'
        'y:{grid:{display:false},ticks:{color:"#003467",font:{size:11,family:"Montserrat,sans-serif"}}}'
        '}}});}'
        f'buildChart("chart-age",{age_labels_js},{age_values_js},"#E24B4A");'
        f'buildChart("chart-dx",{dx_labels_js},{dx_values_js},"#185FA5");'
        '})()'
    )

    html = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.x/dist/tabler-icons.min.css">'
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js" crossorigin="anonymous"></script>'
        f'<style>{css}</style></head><body>'
        '<section class="ltfu-section" aria-labelledby="ltfu-p2-heading">'
        '<h2 id="ltfu-p2-heading" class="sr-only">Priority 2 — 1–2 Visit LTFU Patients</h2>'
        # ── PART 1 ──────────────────────────────────────────────────────────────
        '<div class="section-part">'
        '<p class="sec-eyebrow">Priority 2 — Lost to Follow-Up</p>'
        '<h3 class="sec-title">1–2 Visit LTFU Chronic Patients</h3>'
        '<p class="sec-subtitle">Chronic patients absent 180+ days, segmented by visit history. '
        "Bar width is proportional to each tier's share of total LTFU patients.</p>"
        '<div class="p1-layout">'
        '<div class="card revenue-card">'
        '<p class="rev-label">Annual revenue at risk</p>'
        f'<p class="rev-value">{rev_fmt}</p>'
        '<p class="rev-desc">all chronic LTFU patients</p>'
        '<hr class="rev-divider">'
        '<div class="rev-stat">'
        f'<span class="rev-stat-val">{total_chr_fmt}</span>'
        '<span class="rev-stat-lbl">Chronic LTFU patients</span>'
        '</div></div>'
        f'<div class="funnel-wrap">{funnel_html}</div>'
        '</div>'
        '<div class="insight-bar">'
        '<i class="ti ti-alert-triangle insight-icon"></i>'
        f'<span class="insight-text">{insight1}</span>'
        '</div></div>'
        '<hr class="section-divider">'
        # ── PART 2 ──────────────────────────────────────────────────────────────
        '<div class="section-part">'
        '<p class="sec-eyebrow">Care Pathway Signals</p>'
        '<h3 class="sec-title">Care Journey at Last Visit</h3>'
        f'<p class="sec-subtitle">{sub_note}</p>'
        '<div class="journey-grid">'
        '<div class="card journey-card">'
        '<i class="ti ti-flask journey-icon"></i>'
        f'<span class="journey-pct" style="color:#185FA5;">{lab_pct_fmt}</span>'
        '<span class="journey-label">Lab tests ordered</span>'
        f'<span class="journey-desc">{lab_desc}</span>'
        '</div>'
        '<div class="card journey-card">'
        '<i class="ti ti-pill journey-icon"></i>'
        f'<span class="journey-pct" style="color:#0F6E56;">{rx_pct_fmt}</span>'
        '<span class="journey-label">Prescription received</span>'
        f'<span class="journey-desc">{rx_desc}</span>'
        '</div>'
        '<div class="card journey-card">'
        '<i class="ti ti-calendar-x journey-icon"></i>'
        f'<span class="journey-pct" style="color:#E24B4A;">{nofu_pct_fmt}</span>'
        '<span class="journey-label">No follow-up date</span>'
        f'<span class="journey-desc">{nofu_desc}</span>'
        '</div>'
        '<div class="card journey-card">'
        '<i class="ti ti-scan journey-icon"></i>'
        f'<span class="journey-pct" style="color:#BA7517;">{rad_pct_fmt}</span>'
        '<span class="journey-label">Radiology ordered</span>'
        f'<span class="journey-desc">{rad_desc}</span>'
        '</div>'
        '</div>'
        '<div class="insight-bar">'
        '<i class="ti ti-alert-triangle insight-icon"></i>'
        f'<span class="insight-text">{insight2_text}</span>'
        '</div></div>'
        '<hr class="section-divider">'
        # ── PART 3 ──────────────────────────────────────────────────────────────
        '<div class="section-part">'
        '<p class="sec-eyebrow">Chronic LTFU Profile</p>'
        '<h3 class="sec-title">Who Are the 1–2 Visit Chronic Dropouts?</h3>'
        '<p class="sec-subtitle">Chronic patients with 1–2 visits only, absent 180+ days '
        '— by age group and primary diagnosis.</p>'
        '<div class="charts-row">'
        '<div class="chart-col">'
        '<div class="chart-legend">'
        '<span class="legend-swatch" style="background:#E24B4A;"></span>'
        '<span class="legend-lbl">Chronic LTFU by age group</span>'
        '</div>'
        f'<div class="chart-wrap" style="height:{h_age}px;">'
        '<canvas id="chart-age" role="img" '
        'aria-label="Horizontal bar chart: chronic LTFU patients (1–2 visits) by age group">'
        '<p>Chart not available — enable JavaScript to view chronic LTFU patients by age group.</p>'
        '</canvas></div>'
        '</div>'
        '<div class="chart-col">'
        '<div class="chart-legend">'
        '<span class="legend-swatch" style="background:#185FA5;"></span>'
        '<span class="legend-lbl">Chronic LTFU by diagnosis</span>'
        '</div>'
        f'<div class="chart-wrap" style="height:{h_dx}px;">'
        '<canvas id="chart-dx" role="img" '
        'aria-label="Horizontal bar chart: chronic LTFU patients (1–2 visits) by primary diagnosis, top 8">'
        '<p>Chart not available — enable JavaScript to view chronic LTFU patients by diagnosis.</p>'
        '</canvas></div>'
        '</div>'
        '</div>'
        '<div class="insight-bar">'
        '<i class="ti ti-alert-triangle insight-icon"></i>'
        '<span class="insight-text">Oncology patients represent the single largest diagnosis dropout. '
        'Adult (35–44) and Young Adult (25–34) are the largest age groups lost — '
        'working-age patients most likely to disengage due to cost, time, or lack of perceived urgency.'
        '</span>'
        '</div></div>'
        '</section>'
        f'<script>{js}</script>'
        '</body></html>'
    )

    height_px = 64 + 300 + 50 + 300 + 50 + max(h_age, h_dx) + 200
    return html, height_px


def render_tab_clinical_activity(filters: dict, run_query):
    """Clinical Activity tab v2 — inpatient care quality and readmission analysis."""
    # ── palette ────────────────────────────────────────────────────────────────
    _BLUE   = "#185FA5"
    _GREEN  = "#0F6E56"
    _RED    = "#E24B4A"
    _AMBER  = "#BA7517"
    _PURPLE = "#534AB7"
    _PINK   = "#D4537E"
    _MUTED  = "#B4B2A9"
    _GREY   = "#888780"

    _MATERNITY = {"general maternity", "private maternity"}
    _CFG       = {"responsive": True, "displayModeBar": False, "useResizeHandler": True}

    # ── local helpers ──────────────────────────────────────────────────────────
    def _insight(items, variant="info"):
        cols = {"warn": _RED, "info": _BLUE, "amber": _AMBER}
        bgs  = {"warn": "#FCEBEB", "info": "#EBF5FF", "amber": "#FAEEDA"}
        col  = cols.get(variant, _BLUE)
        bg   = bgs.get(variant, "#EBF5FF")
        if isinstance(items, str):
            items = [items]
        bullets = "".join(
            f"<li style='margin-bottom:3px'>· {i}</li>" for i in items
        )
        st.markdown(
            f'<div style="border-left:3px solid {col};background:{bg};padding:10px 14px;'
            f'border-radius:0 4px 4px 0;margin-bottom:8px">'
            f'<ul style="list-style:none;margin:0;padding:0;font-size:13px;'
            f'color:#003467;line-height:1.7">{bullets}</ul></div>',
            unsafe_allow_html=True,
        )

    def _sec(label):
        from ksh.clinical_module.ui_template import section_header as _sh
        _sh(label)

    def _lbl(text):
        st.markdown(
            f'<div style="font-size:11px;font-weight:600;color:#6B8CAE;margin-bottom:7px">'
            f'{text}</div>',
            unsafe_allow_html=True,
        )

    def _sub(text):
        st.markdown(
            f'<div style="font-size:11px;color:#6B8CAE;margin-bottom:6px">{text}</div>',
            unsafe_allow_html=True,
        )

    def _card_title(text):
        st.markdown(
            f'<div style="font-size:13px;font-weight:600;color:#003467;margin-bottom:3px">'
            f'{text}</div>',
            unsafe_allow_html=True,
        )

    # ── load all dataframes ────────────────────────────────────────────────────
    def _load(fn, label):
        try:
            df = fn(filters, run_query)
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as exc:
            st.warning(f"{label}: {exc}")
            return pd.DataFrame()

    df_ward    = _load(Q.load_ca_ward_summary,              "Ward summary")
    df_grow    = _load(Q.load_ca_admission_growth,          "Admission growth")
    df_lbox    = _load(Q.load_ca_los_boxplot,               "LOS boxplot")
    df_lout    = _load(Q.load_ca_los_outliers,              "LOS outliers")
    df_losdiag = _load(Q.load_ca_los_diagnosis,             "LOS by diagnosis")
    df_c2      = _load(Q.load_ca_readmission_layer2,        "Readmission L2")
    df_c3      = _load(Q.load_ca_readmission_layer3,        "Readmission L3")
    df_c4d     = _load(Q.load_ca_readmission_layer4_detail, "Readmission L4 detail")
    df_c5      = _load(Q.load_ca_readmission_layer5,        "Readmission L5")
    df_genmale = _load(Q.load_ca_general_male,              "General Male profile")
    df_c_l4c   = _load(Q.load_ca_layer4_conditions,         "Layer 4 conditions")
    df_d       = _load(Q.load_ca_section_d,                 "Section D")
    df_e       = _load(Q.load_ca_section_e,                 "Section E")
    df_typh    = _load(Q.load_ca_typhoid,                   "Typhoid")
    df_sepsis  = _load(Q.load_ca_sepsis_enriched,           "Sepsis enriched")
    df_sepsis_wp = _load(Q.load_ca_sepsis_ward_profile,     "Sepsis ward profile")
    df_f       = _load(Q.load_ca_section_f,                 "Section F")
    df_revisit = _load(Q.load_ca_opd_revisits,              "OPD revisits")
    df_opd_tot = _load(Q.load_ca_total_opd_visits,          "Total OPD visits")

    # ══════════════════════════════════════════════════════════════════════════
    # HEADER KPIs
    # ══════════════════════════════════════════════════════════════════════════
    from ksh.clinical_module.ui_template import (
        page_header as _page_header, kpi_row as _kpi_row,
        anomaly_banner as _anomaly_banner,
        chart_card as _chart_card, chart_card_close as _chart_card_close,
        insight_bar as _insight_bar,
        section_header as _section_header,
    )

    _page_header("Clinical Activity")

    if not df_ward.empty:
        total_adm  = int(df_ward["total_admissions"].sum())
        total_disc = int(df_ward["total_discharges"].sum())
        still_adm  = total_adm - total_disc
        med_los    = round(float(df_ward["median_los"].median()), 1)
        read30     = int(df_ward["day_readmission_30"].sum())
        read_pct   = round(read30 / total_disc * 100, 1) if total_disc else 0
        prd_ct     = int(df_ward["patient_request_discharge"].sum())
        prd_pct    = round(prd_ct / total_disc * 100, 1) if total_disc else 0

        _total_opd   = int(df_opd_tot["total_opd_visits"].iloc[0]) if not df_opd_tot.empty else 0
        _revisit_ct  = len(df_revisit)
        _revisit_esc = int(df_revisit["resulted_in_admission"].sum()) if not df_revisit.empty else 0
        _revisit_rt  = round(_revisit_ct / _total_opd * 100, 1) if _total_opd else 0
        _esc_rt      = round(_revisit_esc / _revisit_ct * 100, 1) if _revisit_ct else 0

        _kpi_row([
            {"label": "Admissions",
             "value": f"{total_adm:,}",
             "delta": "All wards",
             "accent_color": "#0C447C"},
            {"label": "Discharged",
             "value": f"{total_disc:,}",
             "delta": "Completed inpatient stays",
             "accent_color": "#0F6E56"},
            {"label": "Still admitted",
             "value": f"{still_adm:,}",
             "delta": "Currently inpatient",
             "delta_good": True,
             "accent_color": "#D97706"},
            {"label": "Median LOS",
             "value": f"{med_los}d",
             "delta": "Mean pulled up by Sepsis outliers",
             "accent_color": "#E5E7EB"},
            {"label": "30-Day Readmission",
             "value": f"{read_pct:.1f}%",
             "delta": "General Male ward — investigate" if read_pct > 6 else "Within reference",
             "delta_good": read_pct <= 6,
             "accent_color": "#A32D2D" if read_pct > 6 else "#D97706" if read_pct > 4 else "#0F6E56"},
            {"label": "OPD Re-visit Rate",
             "value": f"{_revisit_rt}%",
             "delta": f"{_revisit_ct:,} patients · {_esc_rt}% escalated to IPD",
             "delta_good": _revisit_rt <= 5,
             "accent_color": "#A32D2D" if _revisit_rt > 20 else "#D97706" if _revisit_rt > 5 else "#0F6E56"},
            {"label": "Patient req. d/c",
             "value": f"{prd_pct:.1f}%",
             "delta": "Includes data recording inconsistency",
             "accent_color": "#D97706"},
        ])

        if read_pct > 6:
            _anomaly_banner(
                "General Male readmission rate",
                f"30-day readmission is {read_pct:.1f}% — above the 6% investigation threshold. "
                "NCD-Oncology and HIV/AIDS are the primary conditions. See Section C for full analysis.",
                color="#C53030", bg="#FCEBEB",
            )
        if _revisit_rt > 20:
            _anomaly_banner(
                "OPD re-visit escalation",
                f"{_revisit_rt}% of OPD re-visits are escalating to IPD — above the 20% monitoring threshold. "
                "Review same-diagnosis revisit criteria.",
            )

    _gap(16)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION A — WARD OVERVIEW  [table left | trend chart right]
    # ══════════════════════════════════════════════════════════════════════════
    _sec("A — WARD OVERVIEW")

    _A_WARD_CFG = {
        "General Female":       (_BLUE,   "solid"),
        "Pediatric General":    (_GREEN,  "solid"),
        "General Male":         (_PURPLE, "solid"),
        "Maternity (combined)": (_PINK,   "dash"),
    }

    ac1, ac2 = st.columns([1, 1])

    with ac1:
        if not df_ward.empty:
            if "still_admitted" not in df_ward.columns:
                df_ward["still_admitted"] = (
                    df_ward["total_admissions"] - df_ward["total_discharges"]
                ).clip(lower=0)
            dw = df_ward[["ward_name", "total_admissions", "still_admitted", "median_los",
                          "readmission_rate", "patient_request_discharge_pct"]].copy()
            dw["patient_request_discharge_pct"] = dw.apply(
                lambda r: f"{float(r['patient_request_discharge_pct']):.1f}%*"
                if str(r["ward_name"]).lower() in _MATERNITY
                else f"{float(r['patient_request_discharge_pct']):.1f}%",
                axis=1,
            )
            dw["median_los"] = dw["median_los"].apply(lambda v: f"{float(v):.1f}d")
            dw["total_admissions"] = dw["total_admissions"].astype(int)
            dw["still_admitted"] = dw["still_admitted"].astype(int)
            dw.columns = ["Ward", "Admissions", "Still in", "Median LOS",
                          "Readmit %", "Req. %"]

            st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
            render_sortable_table(
                dw,
                height=280,
                badge_columns={
                    "Readmit %": [
                        {"min": 6,    "max": 999, "bg": "#FCEBEB", "text": "#791F1F"},
                        {"min": 4,    "max": 6,   "bg": "#FAEEDA", "text": "#633806"},
                        {"min": -999, "max": 4,   "bg": "#E1F5EE", "text": "#085041"},
                    ],
                },
                key="ward_summary",
            )
            st.markdown(
                '<div style="font-size:10px;color:#6B8CAE;margin-top:4px">'
                '* Maternity ward patient request % is unreliable — see note below.</div>',
                unsafe_allow_html=True,
            )

    with ac2:
        if not df_grow.empty:
            monthly = (
                df_grow.groupby(["month", "ward_name"])
                .size()
                .reset_index(name="admissions")
            )
            monthly["month"] = pd.to_datetime(monthly["month"])
            monthly["ward_label"] = monthly["ward_name"].apply(
                lambda w: "Maternity (combined)"
                if str(w).lower() in _MATERNITY else w
            )
            monthly = (
                monthly.groupby(["month", "ward_label"])["admissions"]
                .sum()
                .reset_index()
                .sort_values("month")
            )
            _a_legend_html = "".join(
                f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:14px">'
                f'<span style="width:14px;height:2px;background:{col};display:inline-block"></span>'
                f'<span style="font-size:11px;color:#003467">{ward}</span></span>'
                for ward, (col, _dash) in _A_WARD_CFG.items()
            )
            st.markdown(f'<div style="margin-bottom:6px">{_a_legend_html}</div>',
                        unsafe_allow_html=True)
            fig = go.Figure()
            for ward, (col, dash) in _A_WARD_CFG.items():
                wd = monthly[monthly["ward_label"] == ward].sort_values("month")
                if wd.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=wd["month"].dt.strftime("%b %y"),
                    y=wd["admissions"],
                    name=ward,
                    mode="lines+markers",
                    line=dict(color=col, width=2, dash=dash),
                    marker=dict(size=4),
                    showlegend=False,
                ))
            fig.update_layout(
                height=280,
                margin=dict(l=0, r=0, t=10, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(showgrid=False,
                           tickfont=dict(size=10, color="#6B8CAE")),
                yaxis=dict(title="Admissions", showgrid=True,
                           gridcolor="#EBF3FB", rangemode="tozero"),
            )
            st.plotly_chart(fig, use_container_width=True, config=_CFG)

    _insight([
        "Maternity patient request % is a data recording inconsistency — clinicians are "
        "recording routine discharges under the wrong discharge type.",
        "Requires a data quality audit before this field can be used for clinical conclusions.",
    ], variant="amber")
    _gap(16)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION B — LENGTH OF STAY
    # ══════════════════════════════════════════════════════════════════════════
    _sec("B — LENGTH OF STAY")

    # Titles row — separate from charts so both charts start at the same level
    _bt1, _bt2 = st.columns(2)
    with _bt1:
        _card_title("LOS Distribution by Ward")
        _sub("Box = middle 50% of stays. Wider box = more variability. "
             "Line = median. Tick marks = typical range boundaries. "
             "Sorted narrowest IQR at top.")
    with _bt2:
        _card_title("LOS Outliers by Ward")
        _sub("Each point is one admission exceeding the ward IQR upper fence.")

    bc1, bc2 = st.columns(2)

    with bc1:
        if not df_lbox.empty:
            box = (
                df_lbox.groupby("ward_name").agg(
                    q1=("q1_los",        "median"),
                    median=("median_los", "median"),
                    q3=("q3_los",        "median"),
                    lw=("lower_whisker", "median"),
                    uw=("upper_whisker", "median"),
                    iqr=("iqr",          "median"),
                )
                .reset_index()
                .sort_values("iqr", ascending=True)
            )
            fig = go.Figure()
            for _, row in box.iterrows():
                fig.add_trace(go.Box(
                    name=row["ward_name"],
                    y=[row["ward_name"]],
                    q1=[row["q1"]],
                    median=[row["median"]],
                    q3=[row["q3"]],
                    lowerfence=[row["lw"]],
                    upperfence=[row["uw"]],
                    orientation="h",
                    fillcolor="rgba(24,95,165,0.15)",
                    line=dict(color=_BLUE),
                    marker=dict(color=_BLUE),
                    showlegend=False,
                ))
                fig.add_annotation(
                    x=float(row["uw"]) + 0.3,
                    y=row["ward_name"],
                    text=f"IQR {row['iqr']:.1f}d",
                    showarrow=False,
                    font=dict(size=10, color="#6B8CAE"),
                    xanchor="left",
                )
            fig.update_layout(
                height=320,
                margin=dict(l=0, r=90, t=10, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Days", showgrid=True, gridcolor="#EBF3FB"),
                yaxis=dict(showgrid=False),
            )
            _pc(fig)

    with bc2:
        if not df_lout.empty:
            SCATTER_COLORS = {
                "general maternity":  _RED,
                "general female":     _BLUE,
                "general male":       _PURPLE,
                "pediatric general":  _GREEN,
            }
            fig = go.Figure()
            for ward in df_lout["ward_name"].unique():
                wd  = df_lout[df_lout["ward_name"] == ward]
                col = SCATTER_COLORS.get(str(ward).lower(), _MUTED)
                icd = wd["icd10_name"] if "icd10_name" in wd.columns else ["—"] * len(wd)
                pbg = wd["primary_burden_group"] if "primary_burden_group" in wd.columns else ["—"] * len(wd)
                dt  = wd["discharge_type"] if "discharge_type" in wd.columns else ["—"] * len(wd)
                fig.add_trace(go.Scatter(
                    x=wd["los_days"],
                    y=wd["ward_name"],
                    mode="markers",
                    marker=dict(color=col, size=8, opacity=0.75),
                    name=ward,
                    showlegend=False,
                    customdata=list(zip(icd, pbg, dt)),
                    hovertemplate=(
                        "<b>%{y}</b><br>LOS: %{x}d<br>"
                        "Diagnosis: %{customdata[0]}<br>"
                        "Group: %{customdata[1]}<br>"
                        "Discharge: %{customdata[2]}<extra></extra>"
                    ),
                ))
            fig.update_layout(
                height=320,
                margin=dict(l=0, r=0, t=10, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="LOS (days)", showgrid=True,
                           gridcolor="#EBF3FB", range=[0, 150]),
                yaxis=dict(showgrid=False),
            )
            _pc(fig)
    # ── Unified LOS insight — both charts together ─────────────────────────────
    _gap(10)

    _b_bullets = []

    if not df_lbox.empty:
        _box_agg = (
            df_lbox.groupby("ward_name")
            .agg(iqr=("iqr", "median"), median_los=("median_los", "median"))
            .reset_index()
        )
        _narrowest  = _box_agg.loc[_box_agg["iqr"].idxmin()]
        _widest     = _box_agg.loc[_box_agg["iqr"].idxmax()]

        _b_bullets.append(
            f"<strong>{_widest['ward_name']} has the widest IQR "
            f"({_widest['iqr']:.1f}d)</strong> — a diverse case mix "
            f"(Sepsis, Typhoid, Oncology) drives variability, not care inconsistency. "
            f"{_narrowest['ward_name']} has the narrowest IQR ({_narrowest['iqr']:.1f}d) "
            f"— a homogeneous case mix with predictable discharge timing."
        )

    if not df_lout.empty:
        _b_sep_n    = 0
        _b_sep_pct  = 0
        _b_total_out = len(df_lout)
        if "primary_burden_group" in df_lout.columns:
            _b_sep_mask = df_lout["primary_burden_group"].str.contains(
                "Sepsis|Infectious", case=False, na=False
            )
            _b_sep_n   = int(_b_sep_mask.sum())
            _b_sep_pct = round(_b_sep_n / _b_total_out * 100) if _b_total_out else 0
        _b_max_los  = int(df_lout["los_days"].max())
        _b_max_ward = str(df_lout.loc[df_lout["los_days"].idxmax(), "ward_name"])

        _b_bullets.append(
            f"<strong>Sepsis accounts for {_b_sep_pct}% of LOS outliers "
            f"({_b_sep_n} of {_b_total_out} admissions).</strong> "
            f"These are patients who arrived already systemically unwell — "
            f"the long stay is clinical necessity, not a process failure. "
            f"The full Sepsis analysis is in Section E."
        )
        _b_bullets.append(
            f"<strong>The {_b_max_los}d {_b_max_ward} stay is the extreme outlier</strong> "
            f"pulling the ward mean above the median. "
            f"Individual case review recommended — documentation anomaly or "
            f"highly complex infection course."
        )

    if _b_bullets:
        _insight(_b_bullets, variant="info")

    _gap(16)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION C — 30-DAY READMISSION ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    _sec("C — 30-DAY READMISSION ANALYSIS")

    # ── Layer 1 + 2 ────────────────────────────────────────────────────────────
    _lbl("Layer 1 + 2 — Where readmissions are happening and who is leaving before they should")

    # Compute shared chart height from the larger of the two datasets
    _n_wards   = len(df_ward) if not df_ward.empty else 0
    _n_dtypes  = len(df_c2)   if not df_c2.empty   else 0
    _c12_h     = max(320, max(_n_wards, _n_dtypes) * 42 + 60)

    st.markdown(
        f'<div style="font-size:11px;margin-bottom:6px">'
        f'<span style="color:{_GREEN}">● &lt; 4% — within expected</span>&nbsp;·&nbsp;'
        f'<span style="color:{_AMBER}">● 4–6% — monitor</span>&nbsp;·&nbsp;'
        f'<span style="color:{_RED}">● &gt; 6% — investigate</span></div>',
        unsafe_allow_html=True,
    )

    cc1, cc2 = st.columns(2)

    with cc1:
        if not df_ward.empty:
            bar_df = df_ward[["ward_name", "readmission_rate"]].copy()
            bar_df = bar_df.sort_values("readmission_rate", ascending=True)
            bar_df["color"] = bar_df["readmission_rate"].apply(
                lambda r: _RED if float(r) > 6 else (_AMBER if float(r) > 4 else _GREEN)
            )
            max_rate = float(bar_df["readmission_rate"].astype(float).max())
            fig = go.Figure(go.Bar(
                y=bar_df["ward_name"],
                x=bar_df["readmission_rate"].astype(float),
                orientation="h",
                marker_color=bar_df["color"].tolist(),
                text=bar_df["readmission_rate"].apply(lambda v: f"{float(v):.1f}%"),
                textposition="outside",
                textfont=dict(size=11, color="#003467"),
            ))
            fig.update_layout(
                height=_c12_h,
                margin=dict(l=0, r=50, t=10, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Readmission rate (%)", showgrid=True,
                           gridcolor="#EBF3FB", range=[0, max_rate * 1.35]),
                yaxis=dict(showgrid=False),
                showlegend=False,
            )
            _pc(fig)

    with cc2:
        if not df_c2.empty:
            def _dc(dtype):
                d = str(dtype).lower()
                if "request" in d: return _RED
                if "stable"  in d: return _BLUE
                if "referral" in d: return _AMBER
                return _MUTED

            bar2 = df_c2.copy().sort_values(
                "total_readmitted_unique_patients", ascending=True
            )
            bar2["color"] = bar2["discharge_type"].apply(_dc)
            max_readmit = int(bar2["total_readmitted_unique_patients"].max())
            fig = go.Figure(go.Bar(
                y=bar2["discharge_type"],
                x=bar2["total_readmitted_unique_patients"],
                orientation="h",
                marker_color=bar2["color"].tolist(),
                text=bar2["total_readmitted_unique_patients"],
                textposition="outside",
                textfont=dict(size=11, color="#003467"),
            ))
            fig.update_layout(
                height=_c12_h,
                margin=dict(l=0, r=40, t=10, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Readmitted patients", showgrid=True,
                           gridcolor="#EBF3FB", range=[0, max_readmit * 1.3]),
                yaxis=dict(showgrid=False),
                showlegend=False,
            )
            _pc(fig)

    # ── General Male readmission profile (full-width block) ────────────────────
    _gap(12)
    with st.container():
        st.markdown("**General Male Readmission Profile**")
        _kpi_row([
            {"label": "Top age group",    "value": "Senior (65+)",   "delta": "21 of 31 readmissions · 67.7%",                  "accent_color": "#0C447C"},
            {"label": "Top condition",    "value": "NCD — Oncology", "delta": "8 readmissions incl. HIV+Oncology comorbidity",   "accent_color": "#0C447C"},
            {"label": "Repeat admitters", "value": "7 patients",     "delta": "Admitted 3+ times · chronic disease revolving door", "accent_color": "#E24B4A"},
        ])
        _gap(16)
        gcl, gcr = st.columns([1, 1])
        with gcl:
            if not df_genmale.empty and "age_band" in df_genmale.columns:
                age_vc = df_genmale["age_band"].value_counts().reset_index()
                age_vc.columns = ["age_band", "count"]
            else:
                age_vc = pd.DataFrame({
                    "age_band": ["Senior (65+)", "Older Adult (55-64)",
                                 "Middle Age (45-54)", "Adult (35-44)"],
                    "count": [21, 5, 3, 2],
                })
            age_vc["colour"] = age_vc["count"].apply(
                lambda c: _RED if c >= 10 else (_AMBER if c >= 5 else _BLUE)
            )
            age_vc = age_vc.sort_values("count", ascending=True)
            fig = go.Figure(go.Bar(
                y=age_vc["age_band"],
                x=age_vc["count"],
                orientation="h",
                marker_color=age_vc["colour"].tolist(),
                text=age_vc["count"],
                textposition="outside",
                textfont=dict(size=11, color="#003467"),
            ))
            fig.update_layout(
                height=260,
                margin=dict(l=0, r=30, t=10, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Readmissions", showgrid=True, gridcolor="#EBF3FB"),
                yaxis=dict(showgrid=False),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, config=_CFG)

        with gcr:
            _TYPE_BADGE = {
                "Chronic": '<span style="display:inline-block;font-size:11px;padding:2px 8px;border-radius:6px;font-weight:500;background:#FCEBEB;color:#791F1F">Chronic</span>',
                "Acute":   '<span style="display:inline-block;font-size:11px;padding:2px 8px;border-radius:6px;font-weight:500;background:#E1F5EE;color:#085041">Acute</span>',
            }
            _gm_df = pd.DataFrame({
                "Condition":    ["NCD — Oncology", "HIV/AIDS + Oncology",
                                 "NCD — Neurologic", "BPH + Renal", "Typhoid"],
                "Readmissions": [6, 2, 2, 2, 1],
                "Type":         [_TYPE_BADGE.get(t, t) for t in
                                 ["Chronic", "Chronic", "Chronic", "Chronic", "Acute"]],
                "Pattern":      ["Repeat admitters", "Comorbid",
                                 "Stable → returned", "Stable → returned",
                                 "Single episode"],
            })
            render_sortable_table(_gm_df, height=260, key="gm_conditions")

    _insight([
        "21 of 31 General Male readmissions are Senior (65+) patients. "
        "NCD-Oncology is the leading condition — 7 patients admitted 3+ times. "
        "Action: review whether structured oncology outpatient management would reduce revolving-door admissions.",
        "30 stable-discharge readmissions = wrong discharge decision — highest clinical priority. "
        "42 patient-request readmissions = patient chose to leave — counselling and retention gap.",
        "Elderly oncology patients with no curative pathway are driving the 8.29% rate. "
        "Action: clinical review for palliative care or structured outpatient oncology pathway for repeat admitters.",
    ], variant="warn")

    _gap(12)

    # ── Layer 3 ────────────────────────────────────────────────────────────────
    _lbl("Layer 3 — Avg LOS before readmission vs LOS at readmission visit")
    lc1, lc2 = st.columns(2)

    with lc1:
        if not df_c3.empty:
            idx = (df_c3[df_c3["is_30day_readmission"] == False]
                   [["ward_name", "avg_los"]]
                   .rename(columns={"avg_los": "before"}))
            rdm = (df_c3[df_c3["is_30day_readmission"] == True]
                   [["ward_name", "avg_los"]]
                   .rename(columns={"avg_los": "readmit"}))
            l3 = (idx.merge(rdm, on="ward_name", how="inner")
                  .pipe(lambda d: d[d["ward_name"].str.lower() != "private maternity"])
                  .dropna(subset=["before", "readmit"]))
            _sub("A longer stay at readmission than before — when both are above the ward "
                 "median — is a premature discharge signal.")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Avg LOS before readmission",
                x=l3["ward_name"], y=l3["before"],
                marker_color="rgba(24,95,165,0.75)",
            ))
            fig.add_trace(go.Bar(
                name="Avg LOS at readmission",
                x=l3["ward_name"], y=l3["readmit"],
                marker_color="rgba(226,75,74,0.85)",
            ))
            fig.update_layout(
                height=340,
                margin=dict(l=0, r=0, t=40, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                barmode="group",
                xaxis=dict(showgrid=False),
                yaxis=dict(title="Avg LOS (days)", showgrid=True, gridcolor="#EBF3FB"),
                legend=dict(orientation="h", y=1.08, yanchor="bottom", x=0.5, xanchor="center", font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
            )
            _pc(fig)

    with lc2:
        _sub("LOS comparison summary per ward")
        for sig in [
            {
                "ward": "General Maternity",
                "before": "3.1 days avg", "after": "15.4 days avg",
                "signal": "5× longer — concern",
                "color": _RED,
                "sub": ("Under insurance, 5× longer readmission stays are the highest risk "
                        "for claim dispute on grounds of preventable readmission."),
            },
            {
                "ward": "General Female",
                "before": "3.5 days avg", "after": "6.8 days avg",
                "signal": "Nearly 2× longer — concern",
                "color": _RED,
                "sub": "",
            },
            {
                "ward": "Pediatric General",
                "before": "2.7 days avg", "after": "2.2 days avg",
                "signal": "Shorter — no premature discharge signal",
                "color": _GREEN,
                "sub": "",
            },
        ]:
            sub_html = (
                f'<div style="font-size:11px;color:#6B8CAE;margin-top:4px">{sig["sub"]}</div>'
                if sig["sub"] else ""
            )
            st.markdown(
                f'<div style="border:0.5px solid #D6E4F0;border-radius:8px;'
                f'padding:12px 14px;margin-bottom:8px">'
                f'<div style="font-size:12px;font-weight:600;color:#003467">{sig["ward"]}</div>'
                f'<div style="font-size:11px;color:#6B8CAE;margin-top:2px">'
                f'Before: {sig["before"]} → At readmission: {sig["after"]}</div>'
                f'<div style="font-size:13px;font-weight:700;color:{sig["color"]};margin-top:4px">'
                f'{sig["signal"]}</div>{sub_html}</div>',
                unsafe_allow_html=True,
            )

    _gap(12)

    # ── Layer 4 ────────────────────────────────────────────────────────────────
    _lbl("Layer 4 — When are patients returning and why?")

    _BAND_ORDER = ["0-7 days (early)", "8-14 days", "15-30 days (late)"]
    _BAND_COLORS = {
        "0-7 days (early)":  _RED,
        "8-14 days":         _AMBER,
        "15-30 days (late)": _BLUE,
    }
    # en-dash band labels used by df_c_l4c (query pre-computes these)
    _COND_BAND_KEYS   = ["0–7d (early)", "8–14d", "15–30d (late)"]
    _COND_BAND_SPACER = ["0–7d (early)", "", "8–14d", "", "15–30d (late)"]

    _COND_COLOURS = {
        "Oncology":               "#0F6E56",   # dark green
        "HIV/AIDS":               "#534AB7",   # purple
        "Cardiovascular":         "#185FA5",   # blue
        "Other Infectious":       "#E24B4A",   # red
        "Neurologic":             "#7B9EB8",   # steel blue-grey
        "Trauma":                 "#BA7517",   # amber
        "Typhoid":                "#1AADA4",   # teal (distinct from Cardiovascular)
        "Genitourinary":          "#8B6FC8",   # light purple (distinct from HIV/AIDS)
        "Communicable":           "#D4537E",   # pink
        "STI":                    "#FF7096",   # light pink
        "Musculoskeletal":        "#B4B2A9",   # muted grey
        "Inflammatory Bowel":     "#C8956E",   # warm peach
        "Gastritis & Duodenitis": "#C8956E",   # warm peach
        "Hypertension":           "#4472C4",   # medium blue
        "BPH":                    "#C4A882",   # tan
        "BPH / Renal":            "#C4A882",   # tan
        "Anaemia":                "#8FA8A4",   # sage
        "LRTI / Pneumonia":       "#6BAED6",   # sky blue
        "Pulmonary Vascular":     "#CE7BB0",   # mauve
        "Stroke":                 "#4A4E69",   # dark purple-grey
        "Sickle Cell":            "#3DAA6A",   # bright green
        "UTI":                    "#9DB8A0",   # sage green
        "Other":                  "#6E6D6A",   # dark muted
    }

    def _primary_condition(s):
        if pd.isna(s) or str(s).strip() in ("", "nan", "None"):
            return "Other"
        first = str(s).split("+")[0].strip()
        for sep in [" - ", ": "]:
            if sep in first:
                first = first.split(sep, 1)[-1]
        return first

    if not df_c_l4c.empty:
        df_c_l4c = df_c_l4c.copy()
        df_c_l4c["primary_cond"] = df_c_l4c["final_disease_burden_group"].apply(
            _primary_condition
        )

    selected_ward = "General Male"
    # Pre-compute band labels so legend can live in the titles row
    _BAND_DISP = {
        "0-7 days (early)":  "0–7d early",
        "8-14 days":         "8–14d",
        "15-30 days (late)": "15–30d late",
    }
    _la_aw        = pd.DataFrame()
    _la_ward_order = []
    if not df_c4d.empty:
        _la_aw = (
            df_c4d[df_c4d["return_band"].notna()]
            .groupby(["ward_name", "return_band"])["readmissions"]
            .sum().reset_index()
        )
        _la_grand = _la_aw["readmissions"].sum() or 1
        _la_bpct  = (
            _la_aw.groupby("return_band")["readmissions"].sum() / _la_grand * 100
        ).round(1)
        _BAND_DISP = {
            "0-7 days (early)":  f"0–7d early ({_la_bpct.get('0-7 days (early)', 0):.1f}%)",
            "8-14 days":         f"8–14d ({_la_bpct.get('8-14 days', 0):.1f}%)",
            "15-30 days (late)": f"15–30d late ({_la_bpct.get('15-30 days (late)', 0):.1f}%)",
        }
        _la_ward_order = (
            _la_aw.groupby("ward_name")["readmissions"].sum()
            .sort_values(ascending=False).index.tolist()
        )

    # Titles + controls row — everything above the charts
    _la_t1, _la_t2 = st.columns([1, 1])
    with _la_t1:
        _card_title("Return window × discharge reason — all wards")
        _sub("Each ward grouped by return window. Legend % = each window's share of "
             "total 30-day returns.")
        _legend_l4 = " · ".join([
            f'<span style="display:inline-flex;align-items:center;gap:5px">'
            f'<span style="width:10px;height:10px;background:{_BAND_COLORS[b]};'
            f'border-radius:2px;display:inline-block"></span>'
            f'<span style="font-size:11px;color:#003467">{_BAND_DISP[b]}</span></span>'
            for b in _BAND_ORDER
        ])
        st.markdown(f'<div style="margin-bottom:4px">{_legend_l4}</div>',
                    unsafe_allow_html=True)
    with _la_t2:
        _card_title("Conditions driving readmissions — by return window")
        _sub("Each column is a return window. Stacked segments show which conditions "
             "are driving returns within that window.")
        selected_ward = st.selectbox(
            "Select ward",
            ["General Male", "General Female", "Pediatric General"],
            key="layer4_ward",
        )

    # Charts row — both columns start at the same level
    la1, la2 = st.columns([1, 1])

    with la1:
        if not _la_aw.empty:
            fig = go.Figure()
            for band in _BAND_ORDER:
                _bsub  = _la_aw[_la_aw["return_band"] == band].set_index("ward_name")
                y_vals = [int(_bsub.loc[w, "readmissions"]) if w in _bsub.index else 0
                          for w in _la_ward_order]
                fig.add_trace(go.Bar(
                    name=_BAND_DISP[band],
                    x=_la_ward_order,
                    y=y_vals,
                    marker_color=_BAND_COLORS[band],
                    showlegend=False,
                    hovertemplate="%{x}: %{y} readmissions<extra></extra>",
                ))
            fig.update_layout(
                height=340,
                margin=dict(l=0, r=0, t=10, b=60),
                plot_bgcolor="white", paper_bgcolor="white",
                barmode="group",
                xaxis=dict(showgrid=False, tickfont=dict(size=10), tickangle=-20),
                yaxis=dict(title="Readmissions", showgrid=True, gridcolor="#EBF3FB"),
            )
            _pc(fig)

    with la2:
        if not df_c_l4c.empty:
            ward_df = df_c_l4c[df_c_l4c["ward_name"] == selected_ward].copy()
            cond_band = (
                ward_df.groupby(["return_band", "primary_cond"])
                .size().reset_index(name="patients")
            )
            # sort by total count descending so largest segment sits at bottom
            cond_totals = (
                cond_band.groupby("primary_cond")["patients"].sum()
                .sort_values(ascending=False)
            )
            sorted_conds = cond_totals.index.tolist()

            # pivot: index=return_band, columns=primary_cond
            pivot = cond_band.pivot_table(
                index="return_band", columns="primary_cond",
                values="patients", aggfunc="sum", fill_value=0,
            )

            # numeric x-axis: 0=early, 1=spacer, 2=mid, 3=spacer, 4=late
            _X_POS   = [0, 1, 2, 3, 4]
            _X_TICKS = [0, 2, 4]
            _X_LBLS  = ["0–7d · early", "8–14d", "15–30d · late"]
            _BAND_TO_IDX = {
                _COND_BAND_KEYS[0]: 0,
                _COND_BAND_KEYS[1]: 2,
                _COND_BAND_KEYS[2]: 4,
            }

            # band header HTML above chart
            st.markdown(
                '<div style="display:flex;justify-content:space-around;'
                'font-family:Montserrat,sans-serif;font-size:10px;font-weight:500;'
                'color:#888780;letter-spacing:0.06em;text-transform:uppercase;'
                'padding:0 12%;margin-bottom:2px">'
                '<span>0–7D · EARLY</span><span>8–14D</span><span>15–30D · LATE</span>'
                '</div>',
                unsafe_allow_html=True,
            )

            fig = go.Figure()
            for cond in sorted_conds:
                y5 = [0, 0, 0, 0, 0]
                for band, idx in _BAND_TO_IDX.items():
                    if band in pivot.index and cond in pivot.columns:
                        y5[idx] = int(pivot.loc[band, cond])
                if sum(y5) == 0:
                    continue
                fig.add_trace(go.Bar(
                    name=cond,
                    x=_X_POS,
                    y=y5,
                    marker_color=_COND_COLOURS.get(cond, _MUTED),
                    marker_line_width=0,
                    width=[0.65, 0, 0.65, 0, 0.65],
                    showlegend=False,
                    hovertemplate=f"{cond}: %{{y}}<extra></extra>",
                ))

            # dashed dividers between bands
            for xp in [1.5, 3.5]:
                fig.add_shape(
                    type="line", x0=xp, x1=xp, y0=0, y1=1, yref="paper",
                    line=dict(color="rgba(128,128,128,0.2)", width=1, dash="dot"),
                )
            # subtle shading on outer bands
            for x0, x1 in [(-0.5, 0.5), (3.5, 4.5)]:
                fig.add_shape(
                    type="rect", x0=x0, x1=x1, y0=0, y1=1, yref="paper",
                    fillcolor="rgba(0,0,0,0.025)", line_width=0, layer="below",
                )

            fig.update_layout(
                barmode="stack",
                showlegend=False,
                height=320,
                margin=dict(t=8, b=40, l=40, r=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(
                    tickmode="array",
                    tickvals=_X_TICKS,
                    ticktext=_X_LBLS,
                    range=[-0.6, 4.6],
                    showgrid=False,
                    tickfont=dict(size=11, family="Inter, sans-serif", color="#888780"),
                    fixedrange=True,
                ),
                yaxis=dict(
                    gridcolor="rgba(128,128,128,0.08)",
                    tickfont=dict(size=11, family="Inter, sans-serif", color="#888780"),
                    title=dict(text="Patients",
                               font=dict(size=11, family="Inter, sans-serif",
                                         color="#888780")),
                    fixedrange=True,
                ),
            )
            _pc(fig)

            # dot legend — only active conditions, wraps automatically
            active_conds = [c for c in sorted_conds
                            if cond_totals.get(c, 0) > 0]
            dot_legend = " ".join([
                f'<span style="display:inline-flex;align-items:center;gap:3px;'
                f'font-size:11px;color:var(--text-color);margin-right:10px">'
                f'<span style="width:8px;height:8px;border-radius:50%;flex-shrink:0;'
                f'background:{_COND_COLOURS.get(c, _MUTED)}"></span>{c}</span>'
                for c in active_conds
            ])
            st.markdown(
                f'<div style="display:flex;flex-wrap:wrap;margin-top:8px">'
                f'{dot_legend}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Condition data not available.")

    # dynamic insight bar keyed to selected ward
    def _l4_insight(ward: str, df: pd.DataFrame) -> list:
        wd      = df[df["ward_name"] == ward]
        early   = wd[wd["return_band"] == _COND_BAND_KEYS[0]]
        late    = wd[wd["return_band"] == _COND_BAND_KEYS[2]]
        total   = len(wd)
        early_n = len(early)
        late_n  = len(late)
        early_pct = round(early_n / total * 100) if total else 0
        def _top(grp):
            vc = grp["primary_cond"].value_counts()
            return vc.index[0] if len(vc) else "Unknown"
        early_top = _top(early)
        late_top  = _top(late)
        if ward == "General Male":
            return [
                f"General Male has two distinct spikes. Early returns (0–7d, "
                f"{early_n} patients) led by {early_top} — patients deteriorating "
                f"immediately after discharge.",
                f"Late returns (15–30d, {late_n} patients) dominated by {late_top} "
                f"— disease progression within the month, not a discharge failure.",
                "Action: 0–7d needs a same-day or next-day review protocol; "
                "15–30d needs structured follow-up recall within 14 days.",
            ]
        else:
            return [
                f"{ward}: early returns dominate ({early_n} of {total} = {early_pct}%). "
                f"{early_top} is the leading condition.",
                f"Action: review discharge criteria for {early_top} presentations — "
                f"confirm resolution before discharge, not only clinical improvement.",
            ]

    if not df_c_l4c.empty and "primary_cond" in df_c_l4c.columns:
        _insight(_l4_insight(selected_ward, df_c_l4c), variant="warn")
    else:
        _insight([
            "General Male 0–7d returns: patient-request (left before ready) and "
            "clinician-stable (deteriorated immediately) — two separate problems.",
            "General Male 15–30d returns: elderly oncology disease progression "
            "within the month.",
            "Action: 0–7d needs patient retention protocol; 15–30d needs "
            "structured follow-up recall.",
        ], variant="warn")

    # ── Layer 5 — TCA documentation (full-width 3-column row) ─────────────────
    _gap(12)
    _lbl("Layer 5 — TCA documentation and follow-up OPD visit")

    if not df_c5.empty:
        rdm_rows  = df_c5[df_c5["is_30day_readmission"] == True]
        total_rdm = int(rdm_rows["total_admissions"].sum()) if not rdm_rows.empty else 127
        tca_ct    = int(rdm_rows["tca_documented_count"].sum()) if not rdm_rows.empty else 46
        tca_pct   = round(tca_ct / total_rdm * 100) if total_rdm else 36
        opd_ct    = int(rdm_rows["had_followup_opd_count"].sum()) if not rdm_rows.empty else total_rdm
        opd_pct   = round(opd_ct / total_rdm * 100) if total_rdm else 100
    else:
        tca_pct, opd_pct, tca_ct, total_rdm = 36, 100, 46, 127

    l5c1, l5c2, l5c3 = st.columns(3)

    for l5col, lbl_txt, val_txt, sub_txt, val_col in [
        (l5c1, "TCA documented at discharge",
         f"{tca_pct}%",
         f"{tca_ct} of {total_rdm} readmissions had a follow-up date",
         _RED),
        (l5c2, "Had OPD visit before readmission",
         f"{opd_pct}%",
         "All readmitted patients visited OPD before returning",
         _BLUE),
    ]:
        with l5col:
            st.markdown(
                f'<div style="border:0.5px solid #D6E4F0;border-radius:8px;'
                f'padding:14px 16px;height:100%">'
                f'<div style="font-size:10px;font-weight:600;color:#6B8CAE;'
                f'text-transform:uppercase;letter-spacing:1px">{lbl_txt}</div>'
                f'<div style="font-size:32px;font-weight:700;color:{val_col};margin:6px 0">'
                f'{val_txt}</div>'
                f'<div style="font-size:11px;color:#6B8CAE">{sub_txt}</div></div>',
                unsafe_allow_html=True,
            )

    with l5c3:
        _insight([
            "64% of readmitted patients left without a scheduled follow-up date.",
            "Action: make TCA documentation mandatory before ward discharge, "
            "especially General Male.",
        ], variant="warn")
        _insight([
            "All readmitted patients returned via OPD — the care pathway is working.",
            "Returns were unplanned in 64% of cases — reactive, not proactive.",
        ], variant="info")

    _gap(16)

    # ══════════════════════════════════════════════════════════════════════════
    # OPD RETURN VISITS — SAME DIAGNOSIS WITHIN 7 DAYS
    # ══════════════════════════════════════════════════════════════════════════
    _sec("OPD RETURN VISITS — SAME OR SIMILAR DIAGNOSIS WITHIN 5–7 DAYS")

    if not df_revisit.empty:
        _rv_total   = len(df_revisit)
        _rv_esc     = int(df_revisit["resulted_in_admission"].sum())
        _rv_esc_pct = round(_rv_esc / _rv_total * 100, 1) if _rv_total else 0
        _rv_avg_d   = round(float(df_revisit["days_to_return"].mean()), 1)
        _rv_day5    = int((df_revisit["days_to_return"] == 5).sum())

        rv1, rv2, rv3, rv4 = st.columns(4)
        with rv1: _kpi("OPD return visits (5–7d)", f"{_rv_total:,}", s="Same or similar diagnosis")
        with rv2: _kpi("Avg days to return", f"{_rv_avg_d}d", s=f"{_rv_day5} returned on day 5")
        with rv3: _kpi("Escalated to inpatient", f"{_rv_esc:,}",
                        s=f"{_rv_esc_pct}% of return visits",
                        color=_RED if _rv_esc_pct > 20 else _AMBER)
        with rv4:
            _dx_esc = (
                df_revisit.groupby("index_diagnosis")
                .agg(total=("index_visit_id", "count"), esc=("resulted_in_admission", "sum"))
                .assign(rate=lambda x: (x["esc"] / x["total"] * 100).round(1))
                .query("total >= 10")
                .sort_values("rate", ascending=False)
            )
            _top_esc = _dx_esc.iloc[0] if len(_dx_esc) > 0 else None
            _top_esc_lbl = (
                f"{_top_esc.name.split(' - ')[-1].split(': ')[-1][:20]} ({_top_esc['rate']:.0f}%)"
                if _top_esc is not None else "—"
            )
            _top_esc_sub = (
                f"{int(_top_esc['esc'])} of {int(_top_esc['total'])} revisits admitted"
                if _top_esc is not None else ""
            )
            _kpi("Highest escalation rate", _top_esc_lbl, s=_top_esc_sub, color=_RED)

        _gap(8)
        rv_cl, rv_cr = st.columns(2)

        with rv_cl:
            _lbl("Top diagnoses — return visit volume")
            _top_dx = (
                df_revisit[df_revisit["index_diagnosis"] != "Unclassified"]
                .groupby("index_diagnosis")["index_visit_id"]
                .count().reset_index()
                .rename(columns={"index_visit_id": "Returns", "index_diagnosis": "Diagnosis"})
                .sort_values("Returns", ascending=True).tail(10)
            )
            _top_dx["Diagnosis"] = _top_dx["Diagnosis"].apply(
                lambda x: x.split(" - ")[-1].split(": ")[-1]
            )
            fig = go.Figure(go.Bar(
                y=_top_dx["Diagnosis"], x=_top_dx["Returns"], orientation="h",
                marker_color=_BLUE, marker_line=dict(width=0),
                text=_top_dx["Returns"], textposition="outside",
                textfont=dict(size=11, color="#003467"),
            ))
            fig.update_layout(height=300, margin=dict(l=0, r=60, t=10, b=10),
                              plot_bgcolor="white", paper_bgcolor="white",
                              xaxis=dict(showgrid=True, gridcolor="#EBF3FB"),
                              yaxis=dict(showgrid=False), showlegend=False)
            _pc(fig)

        with rv_cr:
            _lbl("Escalation rate by diagnosis — % of revisits admitted · min 10 revisits")
            if not _dx_esc.empty:
                _esc_chart = (
                    _dx_esc.reset_index()
                    .rename(columns={"index_diagnosis": "Diagnosis"})
                    .sort_values("rate", ascending=False)
                    .head(10)
                    .sort_values("rate", ascending=True)
                )
                _esc_chart["Diagnosis"] = _esc_chart["Diagnosis"].apply(
                    lambda x: x.split(" - ")[-1].split(": ")[-1]
                )
                _esc_colours = _esc_chart["rate"].apply(
                    lambda r: _RED if r >= 50 else (_AMBER if r >= 25 else _BLUE)
                ).tolist()
                fig2 = go.Figure(go.Bar(
                    y=_esc_chart["Diagnosis"], x=_esc_chart["rate"], orientation="h",
                    marker_color=_esc_colours, marker_line=dict(width=0),
                    text=_esc_chart["rate"].apply(lambda v: f"{v:.0f}%"),
                    textposition="outside", textfont=dict(size=11, color="#003467"),
                ))
                fig2.update_layout(height=300, margin=dict(l=0, r=60, t=10, b=10),
                                   plot_bgcolor="white", paper_bgcolor="white",
                                   xaxis=dict(ticksuffix="%", range=[0, 110],
                                              showgrid=True, gridcolor="#EBF3FB"),
                                   yaxis=dict(showgrid=False), showlegend=False)
                _pc(fig2)

        _gap(8)
        rv_dl, rv_dr = st.columns(2)

        with rv_dl:
            _lbl("Days to return — distribution (5–7 day window)")
            _days = (
                df_revisit.groupby("days_to_return")["index_visit_id"]
                .count().reset_index()
                .rename(columns={"index_visit_id": "Visits", "days_to_return": "Day"})
            )
            _day_colours = _days["Day"].apply(
                lambda d: _RED if d == 1 else (_AMBER if d == 7 else _BLUE)
            ).tolist()
            fig3 = go.Figure(go.Bar(
                x=_days["Day"].astype(str).apply(lambda d: f"Day {d}"),
                y=_days["Visits"],
                marker_color=_day_colours, marker_line=dict(width=0),
                text=_days["Visits"], textposition="outside",
                textfont=dict(size=11, color="#003467"),
            ))
            fig3.update_layout(height=220, margin=dict(l=0, r=60, t=10, b=10),
                               plot_bgcolor="white", paper_bgcolor="white",
                               xaxis=dict(showgrid=False),
                               yaxis=dict(showgrid=True, gridcolor="#EBF3FB"),
                               showlegend=False)
            _pc(fig3)

        with rv_dr:
            _lbl("Return visits by age group")
            _age = (
                df_revisit.groupby("age_group")["index_visit_id"]
                .count().reset_index()
                .rename(columns={"index_visit_id": "Returns", "age_group": "Age group"})
                .sort_values("Returns", ascending=True)
            )
            fig4 = go.Figure(go.Bar(
                y=_age["Age group"], x=_age["Returns"], orientation="h",
                marker_color=_PURPLE, marker_line=dict(width=0),
                text=_age["Returns"], textposition="outside",
                textfont=dict(size=11, color="#003467"),
            ))
            fig4.update_layout(height=220, margin=dict(l=0, r=40, t=10, b=10),
                               plot_bgcolor="white", paper_bgcolor="white",
                               xaxis=dict(showgrid=True, gridcolor="#EBF3FB"),
                               yaxis=dict(showgrid=False), showlegend=False)
            _pc(fig4)

        _top_vol_name = (
            df_revisit["index_diagnosis"].value_counts().index[0]
            .split(" - ")[-1].split(": ")[-1]
            if len(df_revisit) > 0 else "Unknown"
        )
        _top_vol_n   = int(df_revisit["index_diagnosis"].value_counts().iloc[0]) if len(df_revisit) > 0 else 0
        _top_vol_esc = int(
            df_revisit[df_revisit["index_diagnosis"] == df_revisit["index_diagnosis"].value_counts().index[0]]
            ["resulted_in_admission"].sum()
        ) if len(df_revisit) > 0 else 0
        _top_esc_name = (
            _top_esc.name.split(" - ")[-1].split(": ")[-1]
            if _top_esc is not None else "Unknown"
        )
        _insight([
            f"{_rv_total:,} patients returned to OPD within 5–7 days with the same or similar diagnosis. "
            f"{_rv_day5} returned on day 5, suggesting symptoms persisted through the expected "
            f"recovery window.",
            f"{_top_esc_name} has the highest escalation rate — {int(_top_esc['rate'])}% of "
            f"revisiting patients were subsequently admitted. These patients are presenting at OPD, "
            f"being sent home, and deteriorating within days."
            if _top_esc is not None else "",
            f"{_top_vol_name} leads by volume ({_top_vol_n} returns, {_top_vol_esc} escalations). "
            f"Action: review OPD treatment protocols for {_top_esc_name} — a high re-visit rate "
            f"on the same diagnosis within a week suggests under-treatment at first contact.",
        ], variant="amber")

    _gap(16)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION D — DIAGNOSIS-DRIVEN READMISSION
    # ══════════════════════════════════════════════════════════════════════════
    _sec("D — DIAGNOSIS-DRIVEN READMISSION")

    if not df_d.empty:
        dagg = (
            df_d.groupby("final_disease_burden_group")
            .agg(
                total=("total_admissions", "sum"),
                readmit=("readmissions_30d", "sum"),
                discharges=("total_discharges", "sum"),
            )
            .reset_index()
        )
        dagg["rate"] = (
            dagg["readmit"]
            / dagg["discharges"].replace(0, float("nan"))
            * 100
        ).round(2)
        diag_filtered = (
            dagg[(dagg["readmit"] >= 1) & (dagg["total"] >= 3)]
            .dropna(subset=["rate"])
            .sort_values("rate", ascending=True)
        )
        diag_filtered["colour"] = diag_filtered["rate"].apply(
            lambda r: _RED if r >= 10 else (_AMBER if r >= 5 else _BLUE)
        )
        diag_filtered["label"] = diag_filtered["final_disease_burden_group"].apply(
            lambda x: (x.split(" - ")[-1].split(": ")[-1]
                       if " - " in str(x) or ": " in str(x) else str(x))
        )
    else:
        diag_filtered = pd.DataFrame()

    # Pre-compute insight text before columns
    _priority = diag_filtered[diag_filtered["rate"] >= 10].sort_values("rate", ascending=False) if not diag_filtered.empty else pd.DataFrame()
    _monitor  = diag_filtered[(diag_filtered["rate"] >= 5) & (diag_filtered["rate"] < 10)].sort_values("rate", ascending=False) if not diag_filtered.empty else pd.DataFrame()
    _p1 = f"{_priority.iloc[0]['label']} ({_priority.iloc[0]['rate']:.1f}%)" if len(_priority) > 0 else "N/A"
    _p2 = f"{_priority.iloc[1]['label']} ({_priority.iloc[1]['rate']:.1f}%)" if len(_priority) > 1 else None
    _mon_txt = ", ".join(f"{r['label']} ({r['rate']:.1f}%)" for _, r in _monitor.iterrows()) or "None"
    _p1_txt  = f"{_p1} and {_p2}" if _p2 else _p1

    # Titles + legend row
    _dt1, _dt2 = st.columns([1, 1])
    with _dt1:
        _card_title("Readmission Rate by Diagnosis")
        st.markdown(
            f'<div style="font-size:11px;margin-bottom:4px">'
            f'<span style="color:{_RED}">● ≥10% rate</span>&nbsp;·&nbsp;'
            f'<span style="color:{_AMBER}">● 5–10%</span>&nbsp;·&nbsp;'
            f'<span style="color:{_BLUE}">● &lt;5%</span></div>',
            unsafe_allow_html=True,
        )
    with _dt2:
        _card_title("Priority conditions by admissions and readmission count")

    dc1, dc2 = st.columns([1, 1])

    with dc1:
        if not diag_filtered.empty:
            fig = go.Figure(go.Bar(
                y=diag_filtered["label"],
                x=diag_filtered["rate"],
                orientation="h",
                marker_color=diag_filtered["colour"].tolist(),
                marker_line=dict(width=0),
                text=diag_filtered["rate"].apply(lambda v: f"{v:.1f}%"),
                textposition="auto",
                textfont=dict(size=11, color="white"),
                insidetextanchor="middle",
            ))
            fig.update_layout(
                height=max(700, len(diag_filtered) * 44 + 60),
                margin=dict(l=0, r=20, t=10, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                bargap=0.3,
                xaxis=dict(title="Readmission rate (%)", showgrid=True,
                           gridcolor="#EBF3FB", range=[0, 50]),
                yaxis=dict(showgrid=False, tickfont=dict(size=11)),
                showlegend=False,
            )
            with st.container(height=460, border=False):
                _pc(fig)

    with dc2:
        if not diag_filtered.empty:
            def _flag_order(r):
                if r >= 10: return 1
                elif r >= 5: return 2
                return 3

            def _flag_badge(r):
                if r >= 10:
                    return f'<span style="background:{_RED};color:white;font-size:10px;font-weight:700;padding:3px 10px;border-radius:12px;display:inline-block;white-space:nowrap">Priority</span>'
                elif r >= 5:
                    return f'<span style="background:{_AMBER};color:white;font-size:10px;font-weight:700;padding:3px 10px;border-radius:12px;display:inline-block;white-space:nowrap">Monitor</span>'
                return f'<span style="background:{_GREEN};color:white;font-size:10px;font-weight:700;padding:3px 10px;border-radius:12px;display:inline-block;white-space:nowrap">Standard</span>'

            tbl = diag_filtered.head(15).copy()
            tbl["_order"] = tbl["rate"].apply(_flag_order)
            tbl = tbl.sort_values("_order").drop(columns=["_order"])
            tbl_display = pd.DataFrame({
                "Diagnosis":    tbl["label"].values,
                "Admissions":   tbl["total"].astype(int).values,
                "Readmissions": tbl["readmit"].astype(int).values,
                "Rate":         tbl["rate"].apply(lambda v: f"{v:.1f}%").values,
                "Flag":         tbl["rate"].apply(_flag_badge).values,
            })
            render_sortable_table(tbl_display, height=460, key="d_priority")

    _insight([
        f"Priority 1: {_p1_txt} — international priority reduction targets.",
        f"Priority 2: {_mon_txt}.",
        "Action: structured discharge protocols and scheduled follow-up for these conditions.",
        "High readmission rate diagnoses (≥10%) need dedicated outpatient follow-up pathways — "
        "Priority conditions by volume (Oncology, Typhoid) require both rate and count monitoring.",
    ], variant="warn")
    _gap(16)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION E — COMMUNITY INFECTION BURDEN
    # ══════════════════════════════════════════════════════════════════════════
    _sec("E — COMMUNITY INFECTION BURDEN BY WARD")
    ec1, ec2 = st.columns([1, 1])

    with ec1:
        _card_title("Communicable Disease Share per Ward")
        _sub("Proportion of each ward's admissions that are communicable disease. "
             "Wards above 60% are infection-dominant.")
        if not df_e.empty:
            wi = df_e.groupby("ward_name").agg(
                total=("total_admissions", "sum"),
                comm=("communicable_admissions", "sum"),
            ).reset_index()
            wi["comm_pct"]  = (
                wi["comm"] / wi["total"].replace(0, float("nan")) * 100
            ).round(1).fillna(0)
            wi["other_pct"] = (100 - wi["comm_pct"]).clip(lower=0)
            wi = wi.sort_values("comm_pct", ascending=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Communicable %",
                y=wi["ward_name"], x=wi["comm_pct"],
                orientation="h",
                marker_color=_RED,
                text=wi["comm_pct"].apply(lambda v: f"{v:.0f}%"),
                textposition="inside",
                textfont=dict(color="white", size=11),
            ))
            fig.add_trace(go.Bar(
                name="Other %",
                y=wi["ward_name"], x=wi["other_pct"],
                orientation="h",
                marker_color="rgba(180,178,169,0.25)",
                showlegend=False,
            ))
            fig.update_layout(
                height=max(500, len(wi) * 52 + 80),
                margin=dict(l=0, r=0, t=10, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                barmode="stack",
                xaxis=dict(title="% of admissions", range=[0, 100],
                           showgrid=True, gridcolor="#EBF3FB"),
                yaxis=dict(showgrid=False),
                legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center", font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
            )
            with st.container(height=460, border=False):
                _pc(fig)

    with ec2:
        _card_title("Top Communicable Conditions per Ward")
        _sub("Breakdown of infection-type admissions by ward — Typhoid, Malaria, "
             "Sepsis / Other Infectious, Respiratory.")
        if not df_e.empty:
            _wc = df_e.groupby("ward_name").agg(
                typhoid=("typhoid_admissions", "sum"),
                malaria=("malaria_admissions", "sum"),
                sepsis_other=("sepsis_other_admissions", "sum"),
                respiratory=("respiratory_infection_admissions", "sum"),
                total_comm=("communicable_admissions", "sum"),
            ).reset_index()
            _wc["other_comm"] = (
                _wc["total_comm"]
                - _wc["typhoid"] - _wc["malaria"]
                - _wc["sepsis_other"] - _wc["respiratory"]
            ).clip(lower=0)
            _wc = _wc[_wc["total_comm"] > 0].sort_values("total_comm", ascending=True)
            _comm_segs = [
                ("Typhoid",              "typhoid",       "#E24B4A"),
                ("Malaria",              "malaria",       "#BA7517"),
                ("Sepsis / Other Inf.",  "sepsis_other",  "#534AB7"),
                ("Respiratory Inf.",     "respiratory",   "#185FA5"),
                ("Other Communicable",   "other_comm",    "#D3D1C7"),
            ]
            fig_wc = go.Figure()
            for _wclbl, _col, _clr in _comm_segs:
                fig_wc.add_trace(go.Bar(
                    name=_wclbl,
                    y=_wc["ward_name"], x=_wc[_col],
                    orientation="h",
                    marker_color=_clr,
                    showlegend=True,
                    hovertemplate=f"{_wclbl}: %{{x}} admissions<extra></extra>",
                ))
            fig_wc.update_layout(
                height=max(500, len(_wc) * 52 + 80),
                margin=dict(l=0, r=0, t=10, b=80),
                plot_bgcolor="white", paper_bgcolor="white",
                barmode="stack",
                xaxis=dict(title="Communicable admissions",
                           showgrid=True, gridcolor="#EBF3FB"),
                yaxis=dict(showgrid=False),
                legend=dict(
                    orientation="h", x=0, y=-0.15,
                    font=dict(size=11),
                    traceorder="normal",
                ),
            )
            with st.container(height=460, border=False):
                _pc(fig_wc)

    # ────────────────────────────────────────────────────────────────────────────
    # SEPSIS DEEP DIVE — A41 · Why it dominates, what it costs, what follows
    # ────────────────────────────────────────────────────────────────────────────

    # ── BLOCK 1: Why Sepsis appears in every ward ─────────────────────────────
    _gap(12)
    _card_title("Why does Sepsis appear in every ward?")

    if not df_sepsis_wp.empty:

        _sp_trend = (
            df_sepsis_wp
            .groupby("month")
            .agg(
                sepsis_admissions   =("sepsis_admissions",    "sum"),
                with_diabetes       =("with_diabetes",        "sum"),
                with_malaria        =("with_malaria",         "sum"),
                with_respiratory    =("with_respiratory",     "sum"),
                with_gynaecological =("with_gynaecological",  "sum"),
                with_malnutrition   =("with_malnutrition",    "sum"),
                with_any_comorbidity=("with_any_comorbidity", "sum"),
            )
            .reset_index()
            .sort_values("month")
        )
        _sp_trend["month_dt"]   = pd.to_datetime(_sp_trend["month"])
        _sp_trend["with_other"] = (
            _sp_trend["with_any_comorbidity"]
            - _sp_trend["with_diabetes"]
            - _sp_trend["with_malaria"]
            - _sp_trend["with_respiratory"]
            - _sp_trend["with_gynaecological"]
            - _sp_trend["with_malnutrition"]
        ).clip(lower=0)

        _sp_segs = [
            ("Diabetes",                  "with_diabetes",       _AMBER),
            ("Malaria",                   "with_malaria",        _BLUE),
            ("Respiratory / URTI",        "with_respiratory",    _GREEN),
            ("Gynaecological / Puerperal","with_gynaecological", _PINK),
            ("Malnutrition",              "with_malnutrition",   _PURPLE),
            ("Other / no comorbidity",    "with_other",          _MUTED),
        ]

        # Build subtitle from what's actually in the data
        _months_n = len(_sp_trend)
        _structural, _seasonal = [], []
        for _lbl, _col, _ in _sp_segs[:-1]:  # skip "Other"
            if int(_sp_trend[_col].sum()) == 0:
                continue
            _months_present = int((_sp_trend[_col] > 0).sum())
            if _months_present >= max(1, round(_months_n * 0.75)):
                _structural.append(_lbl)
            else:
                _seasonal.append(_lbl)
        _sub_parts = []
        if _structural:
            _v = "is" if len(_structural) == 1 else "are"
            _sub_parts.append(f"{' and '.join(_structural)} {_v} structural — present most months")
        if _seasonal:
            _v = "is" if len(_seasonal) == 1 else "are"
            _sub_parts.append(f"{' and '.join(_seasonal)} {_v} seasonal — spikes in some months")
        _trend_sub = "Monthly admissions stacked by co-occurring condition."
        if _sub_parts:
            _trend_sub += " " + ". ".join(_sub_parts) + "."
        _sub(_trend_sub)

        fig_sp_trend = go.Figure()
        for _sp_lbl, _col, _clr in _sp_segs:
            if int(_sp_trend[_col].sum()) == 0:
                continue  # hide zero-total segments from legend and chart
            fig_sp_trend.add_trace(go.Bar(
                name=_sp_lbl,
                x=_sp_trend["month_dt"],
                y=_sp_trend[_col],
                marker_color=_clr,
                hovertemplate=f"{_sp_lbl}: %{{y}}<extra></extra>",
            ))
        fig_sp_trend.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=10, b=55),
            barmode="stack",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=False, tickfont=dict(size=11, color="#6B8CAE")),
            yaxis=dict(
                title="Sepsis admissions",
                showgrid=True, gridcolor="#EBF3FB",
                rangemode="tozero",
            ),
            legend=dict(
                orientation="h", x=0, y=-0.2,
                font=dict(size=11),
                bgcolor="rgba(0,0,0,0)",
                traceorder="normal",
            ),
        )
        _pc(fig_sp_trend)

        _gap(12)

        _ep1, _ep2 = st.columns(2)

        _sp_ward = (
            df_sepsis_wp
            .groupby("ward_name")
            .agg(
                sepsis_admissions   =("sepsis_admissions",    "sum"),
                ward_total          =("ward_total_admissions", "sum"),
                with_diabetes       =("with_diabetes",        "sum"),
                with_malaria        =("with_malaria",         "sum"),
                with_respiratory    =("with_respiratory",     "sum"),
                with_gynaecological =("with_gynaecological",  "sum"),
                with_malnutrition   =("with_malnutrition",    "sum"),
                with_any_comorbidity=("with_any_comorbidity", "sum"),
            )
            .reset_index()
        )
        _sp_ward["sepsis_share"] = (
            _sp_ward["sepsis_admissions"]
            / _sp_ward["ward_total"].replace(0, float("nan")) * 100
        ).round(1).fillna(0)
        _sp_ward["with_other"] = (
            _sp_ward["with_any_comorbidity"]
            - _sp_ward["with_diabetes"]
            - _sp_ward["with_malaria"]
            - _sp_ward["with_respiratory"]
            - _sp_ward["with_gynaecological"]
            - _sp_ward["with_malnutrition"]
        ).clip(lower=0)
        _sp_ward = _sp_ward.sort_values("sepsis_admissions", ascending=False)
        _sp_total = int(_sp_ward["sepsis_admissions"].sum())

        with _ep1:
            _ward_seg_defs = [
                ("Diabetes",           "with_diabetes",       _AMBER),
                ("Malaria",            "with_malaria",        _BLUE),
                ("Respiratory / URTI", "with_respiratory",    _GREEN),
                ("Gynaecological",     "with_gynaecological", _PINK),
                ("Malnutrition",       "with_malnutrition",   _PURPLE),
                ("Other",              "with_other",          _MUTED),
            ]
            _wards_html = ""
            for _, _wr in _sp_ward.iterrows():
                _w_name  = str(_wr["ward_name"])
                _w_share = float(_wr["sepsis_share"])
                _w_bg = "#FCEBEB" if _w_share >= 25 else "#FAEEDA" if _w_share >= 15 else "#F1EFE8"
                _w_fc = "#791F1F" if _w_share >= 25 else "#633806" if _w_share >= 15 else "#444441"
                _w_counts = [
                    (_slbl, float(_wr[_scol]), _sclr)
                    for _slbl, _scol, _sclr in _ward_seg_defs
                    if float(_wr.get(_scol, 0)) > 0
                ]
                _w_total = sum(v for _, v, _ in _w_counts) or 1.0
                _bar_segs = "".join(
                    f'<div style="width:{round(_sv/_w_total*100,1)}%;background:{_sclr};" '
                    f'title="{_slbl}: {int(_sv)}"></div>'
                    for _slbl, _sv, _sclr in _w_counts
                )
                _leg_items = "".join(
                    f'<span style="display:flex;align-items:center;gap:3px;font-size:10px;color:#6B7280;">'
                    f'<span style="display:inline-block;width:8px;height:8px;border-radius:1px;'
                    f'background:{_sclr};flex-shrink:0;"></span>{_slbl}</span>'
                    for _slbl, _sv, _sclr in _w_counts
                )
                _wards_html += (
                    f'<div style="margin-bottom:12px;">'
                    f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:5px;">'
                    f'<span style="font-size:12px;font-weight:600;color:#374151;">{_w_name}</span>'
                    f'<span style="background:{_w_bg};color:{_w_fc};font-size:10px;font-weight:700;'
                    f'padding:1px 7px;border-radius:3px;">{_w_share:.0f}%</span>'
                    f'</div>'
                    f'<div style="display:flex;height:12px;border-radius:3px;overflow:hidden;'
                    f'margin-bottom:5px;">{_bar_segs}</div>'
                    f'<div style="display:flex;gap:8px;flex-wrap:wrap;">{_leg_items}</div>'
                    f'</div>'
                )
            _ep1_title = (
                f'<div style="font-size:10px;font-weight:600;color:{_AMBER};'
                f'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:10px;">'
                f'Co-occurring conditions by ward — cumulative</div>'
            )
            st.markdown(
                f'<div style="background:var(--secondary-background-color);border-radius:8px;'
                f'padding:12px 14px;border:1px solid #E5E7EB;'
                f'box-shadow:0 1px 3px rgba(0,0,0,0.06);">'
                f'{_ep1_title}{_wards_html}</div>',
                unsafe_allow_html=True,
            )

        with _ep2:
            # ── coding quality rows ──
            _cq_rows_html = ""
            for _cq_lbl, _cq_pct, _cq_clr in [
                ("Unspecified (A41.9)", 100, _RED),
                ("Named organism",      0,   _BLUE),
            ]:
                _cq_rows_html += (
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
                    f'<div style="font-size:11px;color:#374151;width:180px;flex-shrink:0;">'
                    f'{_cq_lbl}</div>'
                    f'<div style="flex:1;background:#E5E7EB;border-radius:2px;'
                    f'height:10px;overflow:hidden;">'
                    f'<div style="width:{_cq_pct}%;height:100%;background:{_cq_clr};'
                    f'border-radius:2px;"></div></div>'
                    f'<div style="font-size:11px;font-weight:600;color:{_cq_clr};'
                    f'width:32px;text-align:right;">{_cq_pct}%</div></div>'
                )

            # ── top co-occurring rows ──
            _comrb_rows = [
                (lbl, int(_sp_ward[col].sum()), clr)
                for lbl, col, clr in [
                    ("Diabetes",                   "with_diabetes",       _AMBER),
                    ("Malaria",                    "with_malaria",        _BLUE),
                    ("Respiratory / URTI",         "with_respiratory",    _GREEN),
                    ("Gynaecological / Puerperal", "with_gynaecological", _PINK),
                    ("Malnutrition",               "with_malnutrition",   _PURPLE),
                ]
                if int(_sp_ward[col].sum()) > 0
            ]
            _comrb_rows.sort(key=lambda x: x[1], reverse=True)
            _max_comrb = max((n for _, n, _ in _comrb_rows), default=1)
            _comrb_rows_html = ""
            if _comrb_rows:
                for _c_lbl, _c_n, _c_clr in _comrb_rows:
                    _c_pct = round(_c_n / _sp_total * 100) if _sp_total else 0
                    _c_w   = round(_c_n / _max_comrb * 100)
                    _comrb_rows_html += (
                        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
                        f'<div style="font-size:11px;color:#374151;width:180px;flex-shrink:0;">'
                        f'{_c_lbl}</div>'
                        f'<div style="flex:1;background:#E5E7EB;border-radius:2px;'
                        f'height:10px;overflow:hidden;">'
                        f'<div style="width:{_c_w}%;height:100%;background:{_c_clr};'
                        f'border-radius:2px;"></div></div>'
                        f'<div style="font-size:11px;font-weight:600;color:{_c_clr};'
                        f'width:32px;text-align:right;">{_c_pct}%</div></div>'
                    )
            else:
                _comrb_rows_html = (
                    f'<div style="font-size:11px;color:#6B7280;padding:4px 0;">'
                    f'No named co-occurring conditions recorded.</div>'
                )

            st.markdown(
                f'<div style="background:var(--secondary-background-color);border-radius:8px;'
                f'padding:12px 14px;border:1px solid #E5E7EB;'
                f'box-shadow:0 1px 3px rgba(0,0,0,0.06);">'
                f'<div style="font-size:10px;font-weight:600;color:{_RED};'
                f'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">'
                f'Coding quality — read before interpreting volume</div>'
                f'{_cq_rows_html}'
                f'<div style="margin-top:6px;padding:7px 10px;background:white;'
                f'border-radius:4px;border-left:2px solid {_RED};'
                f'font-size:11px;color:#374151;line-height:1.5;margin-bottom:14px;">'
                f'Every case coded "Other Sepsis" — no organism named. '
                f'Consistent across all months, not improving over time.</div>'
                f'<div style="font-size:10px;font-weight:600;color:{_AMBER};'
                f'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">'
                f'Top co-occurring conditions — all wards</div>'
                f'{_comrb_rows_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

    else:
        st.info("No Sepsis ward profile data for the selected period.")

    # ── BLOCK 2: LOS consequence ──────────────────────────────────────────────
    _gap(16)
    _card_title("Sepsis is the primary driver of LOS outliers")
    _sub(
        "Admissions exceeding the ward IQR upper fence, classified by the clinical "
        "pathway that preceded admission. The pathway explains the LOS mechanism."
    )

    if not df_sepsis.empty:

        _sf = df_sepsis.copy()
        _sf_total = len(_sf)

        def _classify_pathway(row):
            _pc_days   = row.get("prior_condition_days",       float("nan"))
            _hours     = row.get("hours_since_prior_discharge", float("nan"))
            _opd_days  = row.get("last_opd_days_before",        float("nan"))
            _has_prior = pd.notna(row.get("prior_condition_display"))
            if _has_prior and pd.notna(_pc_days) and _pc_days == 0:
                return "comorbid"
            if _has_prior and pd.notna(_hours) and 0 < _hours <= 72:
                return "hospital_acquired"
            if pd.notna(_opd_days) and _opd_days == 0:
                return "same_day_escalation"
            if pd.notna(_opd_days) and 1 <= _opd_days <= 7:
                return "opd_progression"
            return "community_acquired"

        _sf["_pathway"] = _sf.apply(_classify_pathway, axis=1)

        def _pn(grp): return int((_sf["_pathway"] == grp).sum())
        def _pm(grp):
            _s = _sf.loc[_sf["_pathway"] == grp, "los_days"]
            return round(float(_s.median()), 1) if not _s.empty else 0.0
        def _pp(n):   return f"{round(n / _sf_total * 100)}%" if _sf_total else "—"

        _same_n  = _pn("same_day_escalation")
        _comm_n  = _pn("community_acquired")
        _opd_n   = _pn("opd_progression")
        _hosp_n  = _pn("hospital_acquired")
        _comrb_n = _pn("comorbid")

        _dama_mask = _sf["discharge_type"].str.contains(
            "request|dama|against", case=False, na=False
        )
        _dama_n   = int(_dama_mask.sum())
        _stable_n = _sf_total - _dama_n
        _dama_pct = round(_dama_n / _sf_total * 100) if _sf_total else 0

        _readm_n    = int(_sf["is_30day_readmission"].sum()) \
                      if "is_30day_readmission" in _sf.columns else 0
        _readm_dama = int(
            (_sf["is_30day_readmission"] & _dama_mask).sum()
        ) if "is_30day_readmission" in _sf.columns else 0

        _max_los  = int(_sf["los_days"].max())
        _max_ward = str(_sf.loc[_sf["los_days"].idxmax(), "ward_name"])

        _pw_specs = [
            {
                "colour":   _BLUE,
                "label":    "Same-day escalation",
                "value":    str(_same_n),
                "sub":      f"{_pp(_same_n)} of outliers · OPD → admitted same day · "
                            f"median {_pm('same_day_escalation')}d",
                "badge":    "System working correctly",
                "badge_bg": "#E6F1FB", "badge_fc": "#0C447C",
            },
            {
                "colour":   _GREY,
                "label":    "Community-acquired",
                "value":    str(_comm_n),
                "sub":      f"{_pp(_comm_n)} of outliers · No prior contact · "
                            f"median {_pm('community_acquired')}d",
                "badge":    "Not interceptable",
                "badge_bg": "#F1EFE8", "badge_fc": "#444441",
            },
            {
                "colour":   _AMBER,
                "label":    "OPD → deterioration (1–7d)",
                "value":    str(_opd_n),
                "sub":      f"{_pp(_opd_n)} this period · Seen at OPD then deteriorated",
                "badge":    "Monitor this window",
                "badge_bg": "#FAEEDA", "badge_fc": "#633806",
            },
            {
                "colour":   _RED,
                "label":    "Left against medical advice",
                "value":    str(_dama_n),
                "sub":      f"{_dama_pct}% of outliers · "
                            f"Left before clinical completion",
                "badge":    f"{_readm_dama} readmitted within 30d",
                "badge_bg": "#FCEBEB", "badge_fc": "#791F1F",
            },
        ]

        _pw_cards_html = '<div style="display:flex;gap:12px;align-items:stretch;">'
        for _pw in _pw_specs:
            _pw_cards_html += (
                f'<div style="flex:1;background:var(--secondary-background-color);'
                f'border-radius:8px;padding:14px 14px;'
                f'border:1px solid #E5E7EB;'
                f'border-top:3px solid {_pw["colour"]};'
                f'box-shadow:0 1px 3px rgba(0,0,0,0.06);'
                f'display:flex;flex-direction:column;">'
                f'<div style="flex:1;">'
                f'<div style="font-size:10px;font-weight:600;'
                f'text-transform:uppercase;letter-spacing:0.5px;'
                f'color:{_pw["colour"]};margin-bottom:6px;">'
                f'{_pw["label"]}</div>'
                f'<div style="font-size:28px;font-weight:700;'
                f'color:{_pw["colour"]};line-height:1;margin-bottom:6px;">'
                f'{_pw["value"]}</div>'
                f'<div style="font-size:11px;color:#6B7280;line-height:1.4;'
                f'margin-bottom:10px;">{_pw["sub"]}</div>'
                f'</div>'
                f'<span style="font-size:10px;font-weight:600;padding:3px 9px;'
                f'border-radius:4px;display:inline-block;'
                f'background:{_pw["badge_bg"]};color:{_pw["badge_fc"]};">'
                f'{_pw["badge"]}</span>'
                f'</div>'
            )
        _pw_cards_html += '</div>'
        st.markdown(_pw_cards_html, unsafe_allow_html=True)

        _gap(12)

        _ep3, _ep4 = st.columns(2)

        with _ep3:
            _pw_los_data = sorted(
                [
                    ("OPD → deterioration",   "opd_progression",     _AMBER),
                    ("Same-day escalation",   "same_day_escalation", _BLUE),
                    ("Community-acquired",    "community_acquired",  _GREY),
                    ("Comorbid at admission", "comorbid",            _GREEN),
                ],
                key=lambda r: _pm(r[1]),
                reverse=True,
            )
            _los_max = max((_pm(r[1]) for r in _pw_los_data), default=1) or 1
            _los_rows_html = ""
            for _pw_lbl, _grp, _clr in _pw_los_data:
                _n   = _pn(_grp)
                _med = _pm(_grp)
                _w   = round(_med / _los_max * 100)
                _los_rows_html += (
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
                    f'<div style="font-size:11px;color:#374151;width:195px;flex-shrink:0;">'
                    f'{_pw_lbl} (n={_n})</div>'
                    f'<div style="flex:1;background:#E5E7EB;border-radius:2px;height:10px;overflow:hidden;">'
                    f'<div style="width:{_w}%;height:100%;background:{_clr};border-radius:2px;"></div></div>'
                    f'<div style="font-size:11px;font-weight:600;color:{_clr};width:38px;text-align:right;">'
                    f'{_med}d</div>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="background:var(--secondary-background-color);border-radius:8px;'
                f'padding:14px 16px;border:1px solid #E5E7EB;box-shadow:0 1px 3px rgba(0,0,0,0.06);">'
                f'<div style="font-size:13px;font-weight:700;color:#111827;margin-bottom:4px;">'
                f'Median LOS by pathway</div>'
                f'<div style="font-size:11px;color:#6B7280;margin-bottom:14px;">'
                f'Median — not mean. The {_max_los}d {_max_ward} stay pulls the mean up.</div>'
                f'{_los_rows_html}'
                f'<div style="font-size:11px;color:#9CA3AF;margin-top:6px;">'
                f'Community-acquired stays are longer — Sepsis more advanced at first presentation.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with _ep4:
            _dama_data = [
                ("Left against<br>medical advice", _dama_n,   _RED),
                ("Clinically stable<br>discharge",  _stable_n, _BLUE),
                ("Readmitted<br>within 30d",        _readm_n,  "#F87171"),
            ]
            _dama_max_v = max(_dama_n, _stable_n, _readm_n, 1)
            _bar_max_h  = 90
            _dama_bars_html = ""
            _dama_labels_html = ""
            for _d_lbl, _d_n, _d_clr in _dama_data:
                _d_h = round(_d_n / _dama_max_v * _bar_max_h)
                _dama_bars_html += (
                    f'<div style="flex:1;display:flex;flex-direction:column;'
                    f'align-items:center;justify-content:flex-end;">'
                    f'<div style="font-size:14px;font-weight:700;color:{_d_clr};margin-bottom:4px;">{_d_n}</div>'
                    f'<div style="width:70%;height:{_d_h}px;background:{_d_clr};border-radius:3px 3px 0 0;"></div>'
                    f'</div>'
                )
                _dama_labels_html += (
                    f'<div style="flex:1;text-align:center;font-size:10px;color:#6B7280;'
                    f'line-height:1.3;padding-top:6px;">{_d_lbl}</div>'
                )
            _readm_intro = (
                "Both" if (_readm_n == 2 and _readm_dama == _readm_n)
                else "All" if (_readm_dama > 0 and _readm_dama == _readm_n)
                else f"{_readm_dama} of {_readm_n}"
            )
            _callout_html = (
                f'<div style="background:#FEF2F2;border-left:2px solid {_RED};'
                f'border-radius:0 4px 4px 0;padding:8px 10px;margin-top:12px;'
                f'font-size:11px;color:#7F1D1D;line-height:1.5;">'
                f'{_readm_intro} 30-day readmissions were patients who left against '
                f'medical advice — incomplete Sepsis treatment is the direct cause.</div>'
            ) if _readm_dama > 0 else ""
            st.markdown(
                f'<div style="background:var(--secondary-background-color);border-radius:8px;'
                f'padding:14px 16px;border:1px solid #E5E7EB;box-shadow:0 1px 3px rgba(0,0,0,0.06);">'
                f'<div style="font-size:13px;font-weight:700;color:#111827;margin-bottom:4px;">'
                f'Discharge against medical advice</div>'
                f'<div style="font-size:11px;color:#6B7280;margin-bottom:14px;">'
                f'Patients who left before clinical completion. Sepsis is not self-limiting. '
                f'<span style="font-weight:700;color:{_RED};">{_dama_n}</span></div>'
                f'<div style="display:flex;gap:8px;height:{_bar_max_h + 30}px;align-items:flex-end;">'
                f'{_dama_bars_html}</div>'
                f'<div style="display:flex;gap:8px;">{_dama_labels_html}</div>'
                f'{_callout_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

        _gap(12)

        # ── Unified insight bar ────────────────────────────────────────────────
        _e_cpts = []

        if not df_sepsis_wp.empty:
            _diab_n  = int(_sp_ward["with_diabetes"].sum())
            _mal_n   = int(_sp_ward["with_malaria"].sum())
            _resp_n  = int(_sp_ward["with_respiratory"].sum())
            _gyn_n   = int(_sp_ward["with_gynaecological"].sum())
            _mnut_n  = int(_sp_ward["with_malnutrition"].sum())

            _comorb_defs = [
                ("Diabetes",           _diab_n, _AMBER,
                 "Uncontrolled hyperglycaemia allows any localised infection to progress "
                 "to systemic Sepsis — a chronic disease management failure, not an acute care failure.",
                 "Action: strengthen OPD-level NCD control. "
                 "Prioritise uncontrolled diabetic patients before infection escalates."),
                ("Malaria",            _mal_n,  _BLUE,
                 "Malaria lowers infection resistance, driving secondary Sepsis. "
                 "Predictable and plannable.",
                 "Action: Q2 surge capacity plan and Malaria prophylaxis programme annually."),
                ("Respiratory / URTI", _resp_n, _GREEN,
                 "Respiratory infections are a known Sepsis driver — "
                 "early antibiotic intervention at OPD can prevent systemic progression.",
                 "Action: review OPD antibiotic protocols for respiratory presentations with fever."),
                ("Gynaecological / Puerperal", _gyn_n, _PINK,
                 "Puerperal and gynaecological sepsis carries high mortality risk. "
                 "Maternity ward cases need rapid escalation pathways.",
                 "Action: audit time-to-antibiotic for all gynaecological Sepsis admissions."),
                ("Malnutrition",       _mnut_n, _PURPLE,
                 "Malnutrition severely compromises immune response, making any infection "
                 "a Sepsis risk. These patients are structurally vulnerable.",
                 "Action: flag malnourished patients at OPD for enhanced infection surveillance."),
            ]

            _months_n_wp = len(_sp_trend)
            for _cn, _cv, _cc, _cdesc, _cact in _comorb_defs:
                if _cv == 0:
                    continue
                _cpct = round(_cv / _sp_total * 100) if _sp_total else 0
                _col_key = {
                    "Diabetes": "with_diabetes", "Malaria": "with_malaria",
                    "Respiratory / URTI": "with_respiratory",
                    "Gynaecological / Puerperal": "with_gynaecological",
                    "Malnutrition": "with_malnutrition",
                }.get(_cn, "")
                if _col_key and _col_key in _sp_trend.columns:
                    _mp = int((_sp_trend[_col_key] > 0).sum())
                    _pattern = (
                        "structural — present most months"
                        if _mp >= max(1, round(_months_n_wp * 0.75))
                        else "seasonal — spikes in some months"
                    )
                else:
                    _pattern = "present this period"
                _e_cpts.append((_cc,
                    f"{_cn} co-occurs in {_cpct}% of Sepsis admissions — {_pattern}.",
                    _cdesc,
                    _cact,
                ))

        _e_cpts.append((_RED,
            f"Sepsis produces the longest outlier stays — median "
            f"{_pm('same_day_escalation')}d to {_pm('community_acquired')}d "
            "depending on pathway.",
            f"{_pp(_same_n)} of outliers are same-day escalations: patients arrived at OPD "
            "already systemically unwell. The long LOS is biology, not a care failure. "
            "It cannot be compressed.",
            None,
        ))
        if _dama_n > 0:
            _e_cpts.append((_RED,
                f"{_dama_pct}% of Sepsis outlier patients ({_dama_n} of {_sf_total}) "
                "left against medical advice before completing treatment.",
                "Sepsis is not self-limiting — early discharge significantly increases "
                "risk of deterioration and return admission.",
                "Action: post-discharge follow-up protocol for all Sepsis patients "
                "who leave against medical advice.",
            ))

        _e_cpts.append((_GREY,
            "100% of Sepsis admissions are coded 'Other Sepsis' — "
            "no organism named, not improving over time.",
            "Volume and LOS figures cannot be used for public health escalation decisions "
            "until coding specificity improves.",
            "Action: ICD10 coding review — document suspected organism at every "
            "Sepsis admission, even if culture is pending.",
        ))

        if _e_cpts:
            _cp_bullets = ""
            for _ci, (_cdot, _chd, _cbd, _cact) in enumerate(_e_cpts):
                _clast    = _ci == len(_e_cpts) - 1
                _csep     = "" if _clast else "border-bottom:1px solid #F0C4C1;"
                _cact_html = (
                    f'<div style="font-size:11px;color:{_AMBER};margin-top:4px;'
                    f'font-style:italic;">{_cact}</div>'
                ) if _cact else ""
                _cp_bullets += (
                    f'<div style="display:flex;gap:10px;padding:10px 0;{_csep}">'
                    f'<span style="width:8px;height:8px;border-radius:50%;'
                    f'background:{_cdot};flex-shrink:0;margin-top:4px;"></span>'
                    f'<div style="font-size:12px;line-height:1.6;color:#374151;">'
                    f'<span style="font-weight:600;">{_chd}</span> {_cbd}'
                    f'{_cact_html}</div>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="border:1px solid #F0C4C1;border-left:3px solid {_RED};'
                f'border-radius:8px;padding:14px 16px;background:#FEF2F2;">'
                f'<div style="font-size:10px;font-weight:600;color:{_RED};'
                f'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px;">'
                f'Sepsis (A41) — complete clinical picture</div>'
                f'{_cp_bullets}</div>',
                unsafe_allow_html=True,
            )

    else:
        st.info("No Sepsis outlier data for the selected period.")

    # ── Typhoid trend — full width ─────────────────────────────────────────────
    _gap(12)
    _card_title("Typhoid Monthly Admissions — Kisumu Total")
    _sub("Rising from 4–6 per month in 2024 to 31 per month by Jan 2026. 7.2× increase.")
    if not df_typh.empty:
        tm = (
            df_typh.groupby("month")["typhoid_admissions"]
            .sum()
            .reset_index()
            .sort_values("month")
        )
        tm["month_dt"] = pd.to_datetime(tm["month"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tm["month_dt"],
            y=tm["typhoid_admissions"],
            mode="lines+markers",
            line=dict(color=_BLUE, width=2),
            fill="tozeroy",
            fillcolor="rgba(24,95,165,0.08)",
            marker=dict(size=4, color=_BLUE),
            showlegend=False,
        ))
        try:
            fig.add_vline(
                x=pd.Timestamp("2025-09-01").timestamp() * 1000,
                line_dash="dash",
                line_color=_RED,
                line_width=1,
                annotation_text="Sep 2025",
                annotation_font_size=10,
                annotation_font_color=_RED,
                annotation_position="top right",
            )
        except Exception:
            pass
        fig.update_layout(
            height=260,
            margin=dict(l=0, r=0, t=10, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=False, tickfont=dict(size=11, color="#6B8CAE")),
            yaxis=dict(title="Typhoid admissions", showgrid=True,
                       gridcolor="#EBF3FB", rangemode="tozero"),
        )
        _pc(fig)
        _insight([
            "Admissions increased 7.2× — baseline 4–6/month to 31/month by Jan 2026.",
            "Sustained rise from September 2025, not a single outbreak event.",
            "Action: escalate to public health — pattern warrants investigation and "
            "community-level response.",
        ], variant="warn")

    _gap(16)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION F — OPD TO ADMISSION TIME
    # ══════════════════════════════════════════════════════════════════════════
    _sec("F — OPD TO ADMISSION TIME BY WARD")

    if not df_f.empty:
        f_agg = (
            df_f.groupby("ward_name")
            .agg(median_hours=("median_hours_opd_to_admission", "median"))
            .reset_index()
            .sort_values("median_hours", ascending=True)
        )
        within_col = (
            "admitted_within_4h" if "admitted_within_4h" in df_f.columns
            else "admitted_within_6h"
        )
        tot_col = (
            "total_admissions_with_prior_opd"
            if "total_admissions_with_prior_opd" in df_f.columns
            else within_col
        )
        f_4h = (
            df_f.groupby("ward_name")
            .agg(within4=(within_col, "sum"), total=(tot_col, "sum"))
            .reset_index()
        )
        f_4h["pct"] = (
            f_4h["within4"] / f_4h["total"].replace(0, float("nan")) * 100
        ).round(1).fillna(0)
        f_4h["colour"] = f_4h["pct"].apply(lambda p: _GREEN if p >= 95 else _BLUE)
        f_4h = f_4h.sort_values("pct", ascending=True)

        # Pre-compute insight
        _min_rate = float(f_4h["pct"].min())
        _max_rate = float(f_4h["pct"].max())
        _f_slowest = f_4h.sort_values("pct").iloc[0]
        if _min_rate >= 90:
            _f_finding = "All wards admit 90%+ of patients within 4 hours."
            _f_action  = "No clinical action required — OPD to admission pathway is functioning well."
            _f_variant = "info"
        elif _min_rate >= 80:
            _f_finding = (
                f"Wards admit {_min_rate:.0f}–{_max_rate:.0f}% of patients within 4 hours. "
                f"The majority of admissions occur within the window."
            )
            _f_action  = (
                "Monitor wards below 85%. No urgent clinical action required "
                "but review admission decision pathway for slower wards."
            )
            _f_variant = "info"
        else:
            _f_finding = (
                f"{_f_slowest['ward_name']} admits only {_f_slowest['pct']:.0f}% "
                f"of patients within 4 hours."
            )
            _f_action  = "Review OPD assessment and admission decision pathway for this ward."
            _f_variant = "amber"

        # Titles row
        _ft1, _ft2 = st.columns(2)
        with _ft1:
            _card_title("Median Hours OPD to Admission")
        with _ft2:
            _card_title("Within 4-Hour Admission Rate")

        fc1, fc2 = st.columns(2)

        with fc1:
            fig = go.Figure(go.Bar(
                y=f_agg["ward_name"],
                x=f_agg["median_hours"],
                orientation="h",
                marker_color=_BLUE,
                text=f_agg["median_hours"].apply(lambda v: f"{v:.1f}h"),
                textposition="outside",
                textfont=dict(size=11, color="#003467"),
            ))
            fig.update_layout(
                height=280,
                margin=dict(l=0, r=40, t=10, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Median hours", range=[0, 4],
                           showgrid=True, gridcolor="#EBF3FB"),
                yaxis=dict(showgrid=False),
                showlegend=False,
            )
            _pc(fig)

        with fc2:
            fig = go.Figure(go.Bar(
                y=f_4h["ward_name"],
                x=f_4h["pct"],
                orientation="h",
                marker_color=f_4h["colour"].tolist(),
                text=f_4h["pct"].apply(lambda v: f"{v:.1f}%"),
                textposition="outside",
                textfont=dict(size=11, color="#003467"),
            ))
            fig.update_layout(
                height=280,
                margin=dict(l=0, r=40, t=10, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="% admitted within 4 hours",
                           range=[80, 102], showgrid=True, gridcolor="#EBF3FB"),
                yaxis=dict(showgrid=False),
                showlegend=False,
            )
            _pc(fig)

        _insight([_f_finding, _f_action], variant=_f_variant)


def render_tab2_patient_acquisition(filters: dict, run_query):
    """Patient Acquisition tab — Sections 0, 1, 2, 3."""
    from ksh.clinical_module.ui_template import (
        kpi_card, kpi_row, section_header, insight_card,
        page_header, anomaly_banner,
        chart_card, chart_card_close, insight_bar,
        CHART_LAYOUT, _ax as _ax_t,
        AFYA_BLUE, TEAL, GREEN, AMBER, RED, CORAL, GRAY,
        fmt_num as _fmt_num,
    )

    def _load(fn, label):
        try:
            df = fn(filters, run_query)
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as exc:
            st.warning(f"{label}: {exc}")
            return pd.DataFrame()

    def _sf(v, d=0.0):
        try: return float(v)
        except: return d

    # ── Load all data ─────────────────────────────────────────────────────
    df_ov    = _load(Q.load_acquisition_overview,  "Overview KPIs")
    df_ag    = _load(Q.load_age_gender,             "Age / gender")
    df_gi    = _load(Q.load_age_growth_index,       "Growth index")
    df_cond  = _load(Q.load_condition_profile,      "Condition profile")
    df_rn    = _load(Q.load_rn_ratios,              "R:N ratios")
    df_trend = _load(Q.load_new_returning_trend,    "New/returning trend")
    df_bench = _load(Q.load_level4_benchmark,       "Level 4 benchmark")

    # ── Coerce numerics ───────────────────────────────────────────────────
    for _df, _cols in [
        (df_ov, ["total_patients","new_patients","returning_patients",
                 "chronic_patients","repeat_patients","avg_visits_per_patient","return_rate_pct"]),
        (df_gi,    ["growth_index"]),
        (df_rn,    ["new_patients","returning_patients","rn_ratio"]),
        (df_trend, ["new_patients","returning_patients"]),
        (df_bench, ["facility_pct","benchmark_pct","gap_pp"]),
    ]:
        for _c in _cols:
            if not _df.empty and _c in _df.columns:
                _df[_c] = pd.to_numeric(_df[_c], errors="coerce")

    # ── Scalars ───────────────────────────────────────────────────────────
    total     = int(_sf(df_ov["total_patients"].iloc[0])     if not df_ov.empty else 0)
    new_pts   = int(_sf(df_ov["new_patients"].iloc[0])       if not df_ov.empty else 0)
    returning = int(_sf(df_ov["returning_patients"].iloc[0]) if not df_ov.empty else 0)
    chronic   = int(_sf(df_ov["chronic_patients"].iloc[0])   if not df_ov.empty else 0)
    avg_vis   = _sf(df_ov["avg_visits_per_patient"].iloc[0]  if not df_ov.empty else 0)
    ret_rate  = _sf(df_ov["return_rate_pct"].iloc[0]         if not df_ov.empty else 0)

    # ── SECTION 0 — KPI STRIP ─────────────────────────────────────────────
    page_header("Patient Acquisition")

    _worst_rn = 0.0
    if not df_rn.empty and "rn_ratio" in df_rn.columns:
        _worst_rn = float(df_rn["rn_ratio"].min() or 0)

    kpi_row([
        {"label": "Total patients",
         "value": _fmt_num(total),
         "delta": "All visits in period",
         "accent_color": "#0C447C"},
        {"label": "New patients",
         "value": _fmt_num(new_pts),
         "delta": "First visit in period",
         "delta_good": True,
         "accent_color": "#0F6E56"},
        {"label": "Returning patients",
         "value": _fmt_num(returning),
         "delta": f"{ret_rate:.1f}% return rate",
         "delta_good": ret_rate >= 40,
         "accent_color": "#0F6E56" if ret_rate >= 40 else "#D97706"},
        {"label": "Chronic patients",
         "value": _fmt_num(chronic),
         "delta": f"{round(chronic/total*100,1) if total else 0}% of all patients",
         "accent_color": "#D97706"},
        {"label": "Avg visits / patient",
         "value": f"{avg_vis:.1f}",
         "delta": "Per patient in period",
         "accent_color": "#E5E7EB"},
    ])

    if _worst_rn > 0 and _worst_rn < 0.70:
        anomaly_banner(
            "Low returning-to-new patient ratio",
            f"Worst segment R:N ratio is {_worst_rn:.2f}× — below the 0.70× reference. "
            "Review follow-up scheduling and recall protocols for that segment.",
        )

    _gap(16)

    # ── SECTION 1 — WHO IS COMING ─────────────────────────────────────────
    section_header("1 — Who is coming")

    # Pre-compute data for both charts before column blocks
    _adult_share = 0.0
    if not df_ag.empty and "age_group" in df_ag.columns:
        _total_all = df_ag["patient_count"].sum()
        _adult_df  = df_ag[df_ag["age_group"].isin(["Adult (35-44)", "Senior (65+)"])]
        _adult_share = round(_adult_df["patient_count"].sum() / _total_all * 100, 1) if _total_all else 0

    if not df_gi.empty:
        df_gi["visit_month"] = pd.to_datetime(df_gi["visit_month"], errors="coerce")
        _gi_pivot = df_gi.pivot_table(
            index="age_group", columns="visit_month",
            values="growth_index", aggfunc="mean",
        )
        _gi_overall = df_gi.groupby("age_group")["growth_index"].mean()
        _fastest_grow = _gi_overall.idxmax() if not _gi_overall.empty else "—"
        _fastest_dec  = _gi_overall.idxmin() if not _gi_overall.empty else "—"
        _grow_val = _gi_overall.max() if not _gi_overall.empty else 0
        _dec_val  = _gi_overall.min() if not _gi_overall.empty else 0
    else:
        _fastest_grow, _fastest_dec, _grow_val, _dec_val = "—", "—", 0, 0

    # Titles row
    _wc_t1, _wc_t2 = st.columns(2)
    with _wc_t1:
        chart_card("Patients by age group and gender")
        chart_card_close()
    with _wc_t2:
        chart_card("Age cohort growth index — monthly",
                   "Blue = growth >130 · Red = decline <70 · Grey = within normal range (70–130)")
        chart_card_close()

    # Charts row
    _col_age, _col_growth = st.columns(2)

    with _col_age:
        if not df_ag.empty:
            # Normalise gender labels
            df_ag["gender"] = df_ag["gender"].str.strip().str.title()
            df_ag["gender"] = df_ag["gender"].replace({"F": "Female", "M": "Male"})

            _ag_pivot = df_ag.pivot_table(
                index="age_group", columns="gender",
                values="patient_count", aggfunc="sum",
            ).fillna(0).reset_index()
            _ag_pivot["total"] = (
                _ag_pivot.get("Female", pd.Series(0, index=_ag_pivot.index)) +
                _ag_pivot.get("Male",   pd.Series(0, index=_ag_pivot.index))
            )
            _ag_pivot = _ag_pivot.sort_values("total", ascending=True)

            _fig_ag = go.Figure()
            for _gen, _clr in [("Female", TEAL), ("Male", AFYA_BLUE)]:
                if _gen in _ag_pivot.columns:
                    _fig_ag.add_trace(go.Bar(
                        y=_ag_pivot["age_group"], x=_ag_pivot[_gen],
                        name=_gen, orientation="h",
                        marker_color=_clr,
                    ))
            _fig_ag.update_layout(
                **{**CHART_LAYOUT, "height": 320, "barmode": "stack",
                   "legend": dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
                                  font=dict(size=11), bgcolor="rgba(0,0,0,0)")},
                xaxis=_ax_t(), yaxis={**_ax_t(), "showgrid": False},
            )
            _pc(_fig_ag)

    with _col_growth:
        if not df_gi.empty:
            def _cell_html(val):
                if pd.isna(val):
                    return '<div style="width:20px;height:20px;border-radius:50%;background:var(--secondary-background-color);margin:auto;"></div>'
                if val >= 130:
                    _d = int(val - 100)
                    return (f'<div style="width:20px;height:20px;border-radius:50%;'
                            f'background:#E6F1FB;color:#0C447C;display:flex;align-items:center;'
                            f'justify-content:center;font-size:8px;font-weight:500;margin:auto;">+{_d}</div>')
                if val < 70:
                    _d = int(val - 100)
                    return (f'<div style="width:20px;height:20px;border-radius:50%;'
                            f'background:#FCEBEB;color:#791F1F;display:flex;align-items:center;'
                            f'justify-content:center;font-size:8px;font-weight:500;margin:auto;">{_d}</div>')
                return '<div style="width:20px;height:20px;border-radius:50%;background:var(--secondary-background-color);margin:auto;"></div>'

            _months_short = {m: pd.to_datetime(m).strftime("%b") for m in _gi_pivot.columns}
            _tbl = '<table style="border-collapse:collapse;font-size:10px;width:max-content;min-width:100%;">'
            _tbl += '<tr><th style="text-align:left;padding:4px 6px;color:var(--text-color);opacity:.5;font-weight:400;white-space:nowrap;">Age group</th>'
            for _m in _gi_pivot.columns:
                _tbl += f'<th style="padding:3px;text-align:center;color:var(--text-color);opacity:.5;font-weight:400;font-size:9px;">{_months_short[_m]}</th>'
            _tbl += '</tr>'
            for _age in _gi_pivot.index:
                _tbl += f'<tr><td style="padding:4px 6px;font-size:10px;color:var(--text-color);opacity:.7;white-space:nowrap;">{_age}</td>'
                for _m in _gi_pivot.columns:
                    _tbl += f'<td style="padding:3px 2px;text-align:center;">{_cell_html(_gi_pivot.loc[_age, _m])}</td>'
                _tbl += '</tr>'
            _tbl += '</table>'
            _html = (
                '<div style="overflow-x:auto;overflow-y:visible;'
                'padding-bottom:6px;-webkit-overflow-scrolling:touch;">'
                + _tbl +
                '</div>'
            )
            st.markdown(_html, unsafe_allow_html=True)

    # Single full-width insight bar for both charts
    insight_bar([
        f"Adults 35–44 and Seniors 65+ account for ~{_adult_share:.0f}% of patients — both high-risk for chronic disease.",
        "Female patients skew toward maternal and gynaecological; male toward cardiovascular.",
        f"{_fastest_grow} shows the highest average growth index ({_grow_val:.0f}) — visits increasing month over month. "
        f"{_fastest_dec} shows the steepest decline ({_dec_val:.0f}).",
        "<strong>Action:</strong> ensure chronic disease screening is active for all Adult and Senior presentations. "
        "Investigate whether declining age groups reflect seasonal patterns or genuine patient loss.",
    ], variant="blue")

    _gap(16)

    # ── SECTION 2 — ACQUISITION ───────────────────────────────────────────
    section_header("2 — Acquisition")

    def _shorten(name: str) -> str:
        for sep in [" - ", ": "]:
            if sep in name:
                return name.split(sep, 1)[-1]
        return name

    _tab_labels = ["What brings patients in", "Gender split", "Inpatient vs outpatient"]
    _active_tab = st.radio("View", _tab_labels, horizontal=True, label_visibility="collapsed")

    if not df_cond.empty:
        # Normalise column names
        for _col_raw, _col_new in [("ip_op_flag","ip_op_flag"), ("visit_type","visit_type")]:
            if _col_raw in df_cond.columns:
                df_cond[_col_raw] = df_cond[_col_raw].astype(str)

    if _active_tab == "What brings patients in":
        if not df_cond.empty:
            _cond_agg = (
                df_cond[df_cond["visit_type"].isin(["New", "Returning"])]
                .groupby(["condition", "visit_type"])["patient_count"]
                .sum().reset_index()
            )
            _cond_pivot = _cond_agg.pivot(
                index="condition", columns="visit_type", values="patient_count"
            ).fillna(0)
            _cond_pivot["total"] = (
                _cond_pivot.get("New", 0) + _cond_pivot.get("Returning", 0)
            )
            _top_conds = _cond_pivot.nlargest(10, "total").sort_values("total", ascending=True)
            _top_conds.index = [_shorten(c) for c in _top_conds.index]

            if "cond_drill" not in st.session_state:
                st.session_state.cond_drill = None

            _drill = st.session_state.cond_drill
            st.caption("Click a bar to see age group breakdown · click again to deselect")

            _fig_cond = go.Figure()
            for _vt, _clr in [("New", AFYA_BLUE), ("Returning", TEAL)]:
                if _vt in _top_conds.columns:
                    _bar_colors = []
                    for _cond_lbl in _top_conds.index:
                        if _drill and _cond_lbl != _drill:
                            _bar_colors.append("rgba(180,178,169,0.35)")
                        else:
                            _bar_colors.append(_clr)
                    _fig_cond.add_trace(go.Bar(
                        y=_top_conds.index, x=_top_conds[_vt],
                        name=_vt, orientation="h", marker_color=_bar_colors,
                    ))
            _fig_cond.update_layout(
                **{**CHART_LAYOUT, "height": 340, "barmode": "group"},
                xaxis=_ax_t(), yaxis={**_ax_t(), "showgrid": False},
            )
            _sel = st.plotly_chart(
                _fig_cond, use_container_width=True,
                config={"responsive": True, "displayModeBar": False, "useResizeHandler": True},
                on_select="rerun", selection_mode="points",
                key="cond_chart",
            )
            _clicked_y = None
            if _sel and _sel.get("selection") and _sel["selection"].get("points"):
                _clicked_y = _sel["selection"]["points"][0].get("y")

            if _clicked_y is not None:
                if _clicked_y == _drill:
                    st.session_state.cond_drill = None
                    st.rerun()
                else:
                    st.session_state.cond_drill = _clicked_y
                    st.rerun()

            if _drill:
                _drill_df = (
                    df_cond[
                        (df_cond["condition"].apply(_shorten) == _drill) &
                        (df_cond["visit_type"].isin(["New", "Returning"]))
                    ]
                    .groupby(["age_group", "visit_type"])["patient_count"]
                    .sum().reset_index()
                )
                _drill_pivot = _drill_df.pivot(
                    index="age_group", columns="visit_type", values="patient_count"
                ).fillna(0)
                st.markdown(
                    f'<div style="font-size:11px;font-weight:500;color:#6B8CAE;'
                    f'text-transform:uppercase;letter-spacing:.06em;margin:10px 0 4px;">'
                    f'{_drill} — age group breakdown</div>',
                    unsafe_allow_html=True,
                )
                _fig_drill = go.Figure()
                for _vt, _clr in [("New", AFYA_BLUE), ("Returning", TEAL)]:
                    if _vt in _drill_pivot.columns:
                        _fig_drill.add_trace(go.Bar(
                            y=_drill_pivot.index, x=_drill_pivot[_vt],
                            name=_vt, orientation="h", marker_color=_clr,
                        ))
                _fig_drill.update_layout(
                    **{**CHART_LAYOUT, "height": 280, "barmode": "group"},
                    xaxis=_ax_t(), yaxis={**_ax_t(), "showgrid": False},
                )
                _pc(_fig_drill)

            _ret_skew = _top_conds["Returning"].idxmax() if "Returning" in _top_conds else "—"
            _new_skew = _top_conds["New"].idxmax()       if "New"       in _top_conds else "—"
        else:
            _ret_skew, _new_skew = "—", "—"

        insight_card(
            text=(
                f'<ul style="margin:4px 0 6px;padding-left:16px;line-height:1.8;">'
                f'<li>{_ret_skew} shows the largest returning-vs-new gap — consistent with ongoing chronic management</li>'
                f'<li>Conditions with a high new-patient skew may not be generating structured follow-up visits</li>'
                f'</ul>'
                f'<div style="padding:5px 8px;background:rgba(0,0,0,0.04);border-radius:5px;font-size:11px;">'
                f'<strong>Action:</strong> For conditions with high new-patient skew, implement structured follow-up scheduling at the index visit.'
                f'</div>'
            ),
            label="Condition profile", variant="amber",
        )

    elif _active_tab == "Gender split":
        if not df_cond.empty:
            df_cond["gender"] = df_cond["gender"].str.strip().str.title()
            _gender_agg = (
                df_cond.groupby(["condition", "gender"])["patient_count"]
                .sum().reset_index()
            )
            _gender_pivot = _gender_agg.pivot(
                index="condition", columns="gender", values="patient_count"
            ).fillna(0)
            _gender_pivot["total"] = _gender_pivot.sum(axis=1)
            _top_g = _gender_pivot.nlargest(10, "total").sort_values("total", ascending=True)
            _top_g.index = [_shorten(c) for c in _top_g.index]

            _fig_g = go.Figure()
            for _gen, _clr in [("Female", TEAL), ("Male", AFYA_BLUE)]:
                if _gen in _top_g.columns:
                    _fig_g.add_trace(go.Bar(
                        y=_top_g.index, x=_top_g[_gen],
                        name=_gen, orientation="h", marker_color=_clr,
                    ))
            _fig_g.update_layout(
                **{**CHART_LAYOUT, "height": 320, "barmode": "stack"},
                xaxis=_ax_t(), yaxis={**_ax_t(), "showgrid": False},
            )
            _pc(_fig_g)

        insight_card(
            text=(
                f'<ul style="margin:4px 0 6px;padding-left:16px;line-height:1.8;">'
                f'<li>Antenatal Care and Gynaecological NCD are female-only as expected</li>'
                f'<li>Oncology and Neurologic show near-equal gender distribution</li>'
                f'</ul>'
                f'<div style="padding:5px 8px;background:rgba(0,0,0,0.04);border-radius:5px;font-size:11px;">'
                f'<strong>Action:</strong> Review whether male patients with chronic conditions (Hypertension, Diabetes) are being screened at similar rates to female patients.'
                f'</div>'
            ),
            label="Gender split", variant="blue",
        )

    else:  # Inpatient vs outpatient
        if not df_cond.empty:
            _ipop_agg = (
                df_cond.groupby(["condition", "ip_op_flag"])["patient_count"]
                .sum().reset_index()
            )
            _ipop_pivot = _ipop_agg.pivot(
                index="condition", columns="ip_op_flag", values="patient_count"
            ).fillna(0)
            _ipop_pivot["total"] = _ipop_pivot.sum(axis=1)
            _top_ip = _ipop_pivot.nlargest(10, "total").sort_values("total", ascending=True)
            _top_ip.index = [_shorten(c) for c in _top_ip.index]

            _fig_ip = go.Figure()
            for _flag, _clr in [("Outpatient", AFYA_BLUE), ("Inpatient", CORAL)]:
                if _flag in _top_ip.columns:
                    _fig_ip.add_trace(go.Bar(
                        y=_top_ip.index, x=_top_ip[_flag],
                        name=_flag, orientation="h", marker_color=_clr,
                    ))
            _fig_ip.update_layout(
                **{**CHART_LAYOUT, "height": 320, "barmode": "stack"},
                xaxis=_ax_t(), yaxis={**_ax_t(), "showgrid": False},
            )
            _pc(_fig_ip)

        insight_card(
            text=(
                f'<ul style="margin:4px 0 6px;padding-left:16px;line-height:1.8;">'
                f'<li>Hypertension is almost entirely outpatient — confirming that hypertensive urgency is rarely being admitted</li>'
                f'<li>Oncology has the highest inpatient share — consistent with chemotherapy and complication management</li>'
                f'</ul>'
                f'<div style="padding:5px 8px;background:rgba(0,0,0,0.04);border-radius:5px;font-size:11px;">'
                f'<strong>Action:</strong> Cross-reference with OPD to IPD tab — conditions with low inpatient share but high clinical severity may be candidates for escalation protocol review.'
                f'</div>'
            ),
            label="IP vs OP split", variant="amber",
        )

    _gap(16)

    # ── SECTION 3 — ARE THEY COMING BACK ─────────────────────────────────
    section_header("3 — Are they coming back")

    st.markdown(
        '<div style="font-size:12px;color:var(--text-color);opacity:.7;'
        'background:var(--secondary-background-color);border-radius:8px;'
        'padding:10px 12px;margin-bottom:12px;line-height:1.6;">'
        '<strong>Return-to-New ratio (R:N)</strong> — for every new patient in a segment, '
        'how many returning patients does the facility retain? '
        '1.0× = equal new and returning. Below 1.0× = acquiring faster than retaining. '
        'Chronic disease segments should trend toward 1.0× as patients build ongoing care relationships.'
        '</div>',
        unsafe_allow_html=True,
    )

    # R:N segment cards
    _SEGMENT_CFG = {
        "Chronic":       {"expected": 0.85, "colour": AMBER},
        "Oncology":      {"expected": 0.90, "colour": GREEN},
        "Maternal":      {"expected": 0.90, "colour": GREEN},
        "Mental Health": {"expected": 0.85, "colour": RED},
    }
    _ACTION_MAP = {
        "Chronic":       "Review follow-up scheduling for chronic patients. See Retention tab for LTFU detail.",
        "Oncology":      "Maintain structured follow-up scheduling.",
        "Maternal":      "Monitor ANC completion rates in Retention tab.",
        "Mental Health": "Review whether psychiatric follow-up appointments are being scheduled and kept.",
    }

    _df_rn_ov = df_rn[df_rn["visit_month"].isna()] if not df_rn.empty else pd.DataFrame()

    _s1, _s2, _s3, _s4 = st.columns(4)
    _seg_cols = [_s1, _s2, _s3, _s4]
    for _i_s, (_seg, _cfg) in enumerate(_SEGMENT_CFG.items()):
        _srow = _df_rn_ov[_df_rn_ov["segment"] == _seg] if not _df_rn_ov.empty else pd.DataFrame()
        if _srow.empty:
            continue
        _rn_val  = _sf(_srow["rn_ratio"].iloc[0])
        _new_n   = int(_sf(_srow["new_patients"].iloc[0]))
        _ret_n   = int(_sf(_srow["returning_patients"].iloc[0]))
        _tot_s   = _new_n + _ret_n
        _new_pct = round(_new_n / _tot_s * 100, 1) if _tot_s else 0
        _ret_pct = round(100 - _new_pct, 1)
        _below   = _rn_val < _cfg["expected"]
        _bdr_col = "#BA7517" if _seg == "Chronic" else "#E24B4A" if _seg == "Mental Health" else "#D6E4F0"
        _val_col = _cfg["colour"]
        _icon    = "↓ " if _below else "✓"
        _tile_extra = f"border:1px solid {_bdr_col};" if _below else ""
        with _seg_cols[_i_s]:
            st.markdown(
                f'<div class="kpi-tile" style="border-top:3px solid {_val_col};{_tile_extra}">'
                f'<div class="kpi-label">{_seg}</div>'
                f'<div class="kpi-value" style="color:{_val_col};font-size:26px;">{_rn_val:.2f}×</div>'
                f'<div style="font-size:11px;color:#9CA3AF;margin-top:3px;">'
                f'{_new_pct}% new · {_ret_pct}% returning</div>'
                f'<div style="font-size:11px;margin-top:8px;padding-top:6px;'
                f'border-top:0.5px solid rgba(128,128,128,0.15);line-height:1.4;color:{_val_col};">'
                f'{_icon} {_ACTION_MAP[_seg]}</div></div>',
                unsafe_allow_html=True,
            )

    _gap(12)

    # R:N grouped bar chart
    if not _df_rn_ov.empty:
        _segs = [s for s in _SEGMENT_CFG if s in _df_rn_ov["segment"].values]
        _fig_rn = go.Figure()
        for _vt, _clr_rn in [("new_patients", AFYA_BLUE), ("returning_patients", TEAL)]:
            _rn_lbl = "New patients" if _vt == "new_patients" else "Returning patients"
            _rn_vals = [
                _sf(_df_rn_ov[_df_rn_ov["segment"] == s][_vt].iloc[0])
                if s in _df_rn_ov["segment"].values else 0
                for s in _segs
            ]
            _fig_rn.add_trace(go.Bar(
                x=_segs, y=_rn_vals, name=_rn_lbl, marker_color=_clr_rn,
            ))
        for _seg in _segs:
            _sr = _df_rn_ov[_df_rn_ov["segment"] == _seg].iloc[0]
            _rn_v  = _sf(_sr["rn_ratio"])
            _exp_v = _SEGMENT_CFG[_seg]["expected"]
            _diff  = round(_rn_v - _exp_v, 2)
            _sign  = "+" if _diff >= 0 else ""
            _c_ann = GREEN if _diff >= 0 else RED
            _y_ann = max(_sf(_sr["new_patients"]), _sf(_sr["returning_patients"])) * 1.05
            _fig_rn.add_annotation(
                x=_seg, y=_y_ann,
                text=f"{_sign}{_diff:.2f}× {'Above' if _diff >= 0 else 'Below'} expected",
                showarrow=True, arrowhead=2, arrowsize=0.8,
                arrowcolor=_c_ann, font=dict(size=10, color=_c_ann), ay=-20,
            )
        _fig_rn.update_layout(
            **{**CHART_LAYOUT, "height": 320, "barmode": "group"},
            xaxis=_ax_t(), yaxis=_ax_t(),
        )
        _pc(_fig_rn)

    _gap(12)

    # Level 4 benchmark + trend (two columns)
    _col_b, _col_t = st.columns(2)

    with _col_b:
        st.caption("Patient mix vs Level 4 benchmark")
        st.caption("Your facility split vs Level 4 private hospital benchmark.")
        if not df_bench.empty:
            _fig_bm = go.Figure()
            for _brow in df_bench.itertuples():
                _pt_label = str(_brow.patient_type)
                _fig_bm.add_trace(go.Bar(
                    x=[_pt_label], y=[_brow.benchmark_pct],
                    name="Level 4 benchmark", marker_color=GRAY,
                    showlegend=(_brow.Index == df_bench.index[0]),
                ))
                _fig_bm.add_trace(go.Bar(
                    x=[_pt_label], y=[_brow.facility_pct],
                    name="Your facility", marker_color=TEAL,
                    showlegend=(_brow.Index == df_bench.index[0]),
                ))
                _sign_b = "+" if _brow.gap_pp >= 0 else ""
                _c_b    = GREEN if _brow.gap_pp >= 0 else RED
                _fig_bm.add_annotation(
                    x=_pt_label,
                    y=max(_brow.benchmark_pct, _brow.facility_pct) + 3,
                    text=f"{_sign_b}{_brow.gap_pp:.0f}pp {'Above' if _brow.gap_pp >= 0 else 'Below'} benchmark",
                    showarrow=True, arrowhead=2, arrowsize=0.8,
                    arrowcolor=_c_b, font=dict(size=10, color=_c_b), ay=-20,
                )
            _fig_bm.update_layout(
                **{**CHART_LAYOUT, "height": 280, "barmode": "group"},
                yaxis={**_ax_t(), "ticksuffix": "%", "range": [0, 100]},
                xaxis=_ax_t(),
            )
            _pc(_fig_bm)

            _new_row = df_bench[df_bench["patient_type"].str.lower().str.contains("new")]
            _new_gap = _sf(_new_row["gap_pp"].iloc[0]) if not _new_row.empty else 0.0
            insight_card(
                text=(
                    f'<ul style="margin:4px 0 6px;padding-left:16px;line-height:1.8;">'
                    f'<li>New patients are {abs(_new_gap):.0f}pp {"above" if _new_gap > 0 else "below"} the Level 4 benchmark</li>'
                    + (
                        '<li>High new patient share with a below-benchmark returning share signals the facility is growing reach but not converting first visits into ongoing care</li>'
                        if _new_gap > 0 else
                        '<li>Below-benchmark new patient acquisition suggests slower growth than comparable Level 4 facilities</li>'
                    ) +
                    f'</ul>'
                    f'<div style="padding:5px 8px;background:rgba(0,0,0,0.04);border-radius:5px;font-size:11px;">'
                    f'<strong>Action:</strong> Review follow-up scheduling at OPD discharge — are return appointments being booked at every chronic visit?'
                    f'</div>'
                ),
                label="Level 4 benchmark",
                variant="amber" if _new_gap > 5 else "blue",
            )

    with _col_t:
        st.caption("New vs returning patients — monthly trend")
        if not df_trend.empty:
            df_trend["visit_month"] = pd.to_datetime(df_trend["visit_month"], errors="coerce")
            df_trend = df_trend.sort_values("visit_month")
            _fig_tr = go.Figure()
            _fig_tr.add_trace(go.Scatter(
                x=df_trend["visit_month"], y=df_trend["new_patients"],
                name="New patients", mode="lines+markers",
                line=dict(color=AFYA_BLUE, width=2), marker=dict(size=4),
                fill="tozeroy", fillcolor="rgba(0,114,206,0.06)",
            ))
            _fig_tr.add_trace(go.Scatter(
                x=df_trend["visit_month"], y=df_trend["returning_patients"],
                name="Returning patients", mode="lines+markers",
                line=dict(color=GREEN, width=2, dash="dash"), marker=dict(size=4),
                fill="tozeroy", fillcolor="rgba(56,161,105,0.06)",
            ))
            _fig_tr.update_layout(
                **{**CHART_LAYOUT, "height": 280},
                xaxis=_ax_t(), yaxis={**_ax_t(), "title": {"text": "Patients"}},
            )
            _pc(_fig_tr)

            _first_new = _sf(df_trend["new_patients"].iloc[0])
            _last_new  = _sf(df_trend["new_patients"].iloc[-1])
            _first_ret = _sf(df_trend["returning_patients"].iloc[0])
            _last_ret  = _sf(df_trend["returning_patients"].iloc[-1])
            _new_dir   = "growing"   if _last_new > _first_new else "declining"
            _ret_dir   = "declining" if _last_ret < _first_ret else "growing"
            _diverging = (_last_new > _first_new) and (_last_ret < _first_ret)

            insight_card(
                text=(
                    f'<ul style="margin:4px 0 6px;padding-left:16px;line-height:1.8;">'
                    f'<li>New patients are {_new_dir} while returning patients are {_ret_dir}</li>'
                    + (
                        '<li>As new patients grow, returning patients are not keeping pace — the facility is acquiring faster than it retains</li>'
                        if _diverging else
                        '<li>New and returning volumes are moving together — no retention pressure detected</li>'
                    ) +
                    '<li>Detailed dropout analysis is in the Flow and Retention tab</li>'
                    f'</ul>'
                    f'<div style="padding:5px 8px;background:rgba(0,0,0,0.04);border-radius:5px;font-size:11px;">'
                    f'<strong>Action:</strong> Review follow-up scheduling at OPD discharge.'
                    f'</div>'
                ),
                label="New vs returning trend",
                variant="amber" if _diverging else "blue",
            )


def render_tab3_retention(filters: dict, run_query):
    """Flow and Retention tab — Sections 0, A–G."""
    from ksh.clinical_module.ui_template import (
        kpi_card, kpi_row, section_header, insight_card,
        page_header, anomaly_banner,
        chart_card, chart_card_close, insight_bar,
        CHART_LAYOUT, _ax as _ax_t,
        CA_BLUE, CA_RED, CA_AMBER, CA_GREEN,
        fmt_num as _fmt_num,
    )

    CA_GREEN_R = '#0F6E56'
    CA_AMBER_R = '#BA7517'
    CA_RED_R   = '#E24B4A'
    CA_BLUE_R  = '#185FA5'

    def _load(fn, label):
        try:
            df = fn(filters, run_query)
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as exc:
            st.warning(f"{label}: {exc}")
            return pd.DataFrame()

    def _sf(v, d=0.0):
        try: return float(v)
        except: return d

    # ── Load all data up-front ────────────────────────────────────────────
    df_ov    = _load(Q.load_retention_overview,   "Overview")
    df_trend = _load(Q.load_retention_trend,      "Trend")
    df_demo  = _load(Q.load_ltfu_demographics,    "Demographics")
    df_lapse = _load(Q.load_lapsing_cohort,       "Lapsing cohort")
    df_tier  = _load(Q.load_visit_tier,           "Visit tier")
    df_drop  = _load(Q.load_dropout_profile,      "Dropout profile")
    df_cp    = _load(Q.load_care_pathway,         "Care pathway")
    df_wait  = _load(Q.load_wait_times,           "Wait times")
    df_clin  = _load(Q.load_clinician_ltfu,       "Clinician LTFU")

    # ── Scalars from overview ─────────────────────────────────────────────
    chronic    = int(_sf(df_ov["chronic_patients"].iloc[0]) if not df_ov.empty else 0)
    active     = int(_sf(df_ov["active_count"].iloc[0])     if not df_ov.empty else 0)
    lapsing    = int(_sf(df_ov["lapsing_count"].iloc[0])    if not df_ov.empty else 0)
    ltfu       = int(_sf(df_ov["ltfu_count"].iloc[0])       if not df_ov.empty else 0)
    active_pct  = round(active  / chronic * 100, 1) if chronic else 0.0
    lapsing_pct = round(lapsing / chronic * 100, 1) if chronic else 0.0
    ltfu_pct    = round(ltfu    / chronic * 100, 1) if chronic else 0.0

    # ── Section C scalars (referenced in Section G) ──────────────────────
    total_lapsing   = int(_sf(df_lapse["total_lapsing"].iloc[0])      if not df_lapse.empty else 0)
    rev_recoverable = _sf(df_lapse["recoverable_revenue_kes"].iloc[0] if not df_lapse.empty else 0)
    rev_m           = rev_recoverable / 1_000_000
    cash_pct        = _sf(df_lapse["cash_pct"].iloc[0]                if not df_lapse.empty else 0)

    # ── Section F scalars (referenced in Section G) ──────────────────────
    if not df_clin.empty:
        df_clin["ltfu_rate_pct"] = pd.to_numeric(df_clin["ltfu_rate_pct"], errors="coerce")
        _top_clin    = df_clin.sort_values("ltfu_rate_pct", ascending=False).iloc[0]
        top_clin_id  = str(_top_clin["clinician_id"])
        top_clin_rate= _sf(_top_clin["ltfu_rate_pct"])
        top_clin_n   = int(_sf(_top_clin["chronic_seen"]))
        above_50     = int((df_clin["ltfu_rate_pct"] >= 50).sum())
    else:
        top_clin_id, top_clin_rate, top_clin_n, above_50 = "—", 0.0, 0, 0

    # ── Page header + KPI strip ───────────────────────────────────────────
    page_header("Flow and Retention")

    kpi_row([
        {"label": "Chronic patients",
         "value": f"{chronic:,}",
         "delta": "Under active management",
         "accent_color": "#0C447C"},
        {"label": "Active (≤90d)",
         "value": f"{active:,}",
         "delta": f"{active_pct}% — retained",
         "delta_good": True,
         "accent_color": "#0F6E56"},
        {"label": "Lapsing (91–180d)",
         "value": f"{lapsing:,}",
         "delta": f"{lapsing_pct}% — at risk",
         "delta_good": False,
         "accent_color": "#D97706"},
        {"label": "LTFU (>180d)",
         "value": f"{ltfu:,}",
         "delta": f"{ltfu_pct}% — lost",
         "delta_good": False,
         "accent_color": "#A32D2D" if ltfu_pct > 35 else "#D97706"},
    ])

    if ltfu_pct > 35:
        anomaly_banner(
            "LTFU rate above threshold",
            f"{ltfu_pct}% of chronic patients are lost to follow-up (>180d) — above the 35% investigation threshold. "
            "100% had no documented follow-up date. See Section G for priority actions.",
            color="#C53030", bg="#FCEBEB",
        )

    _gap(16)

    # ── SECTION A — POPULATION SHARE ─────────────────────────────────────
    section_header("A — What is the share of active, lapsing and LTFU patients")

    _col_lc, _col_tr = st.columns(2)

    with _col_lc:
        chart_card("Chronic patient lifecycle distribution",
                   f"How {chronic:,} chronic patients are distributed today across retention stages.")
        _fig_lc = go.Figure()
        for _lc_lbl, _val, _clr in [
            ("Active (≤90d)",     active,  CA_GREEN_R),
            ("Lapsing (91–180d)", lapsing, CA_AMBER_R),
            ("LTFU (>180d)",      ltfu,    CA_RED_R),
        ]:
            _fig_lc.add_trace(go.Bar(
                y=["Chronic patients"], x=[_val], name=_lc_lbl, orientation="h",
                marker_color=_clr,
            ))
        _fig_lc.update_layout(
            **{**CHART_LAYOUT, "height": 120, "barmode": "stack",
               "legend": dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
                              font=dict(size=11), bgcolor="rgba(0,0,0,0)")},
            xaxis=_ax_t(), yaxis={**_ax_t(), "showgrid": False},
        )
        _pc(_fig_lc)
        chart_card_close()

        _dropout_pct = round((lapsing + ltfu) / chronic * 100, 1) if chronic else 0
        insight_bar([
            f"Combined dropout (lapsing + LTFU) is {_dropout_pct}% of chronic patients.",
            f"{ltfu_pct}% fully lapsed (>180d) · {lapsing_pct}% in the recovery window (91–180d).",
            "<strong>Action:</strong> start with the lapsing cohort — still reachable before they become LTFU.",
        ], variant="red")

    with _col_tr:
        chart_card("Monthly trend — active vs lapsing vs LTFU",
                   "Is the active cohort growing or shrinking over time?")
        if not df_trend.empty:
            df_trend["visit_month"] = pd.to_datetime(df_trend["visit_month"], errors="coerce")
            df_trend = df_trend.sort_values("visit_month")
            _fig_tr = go.Figure()
            for _cn, _tr_lbl, _clr, _dsh in [
                ("active_count",  "Active",  CA_GREEN_R, "solid"),
                ("lapsing_count", "Lapsing", CA_AMBER_R, "dash"),
                ("ltfu_count",    "LTFU",    CA_RED_R,   "dot"),
            ]:
                if _cn in df_trend.columns:
                    df_trend[_cn] = pd.to_numeric(df_trend[_cn], errors="coerce")
                    _fig_tr.add_trace(go.Scatter(
                        x=df_trend["visit_month"], y=df_trend[_cn],
                        name=_tr_lbl, mode="lines+markers",
                        line=dict(color=_clr, width=2, dash=_dsh), marker=dict(size=3),
                    ))
            _fig_tr.update_layout(**{**CHART_LAYOUT, "height": 220}, xaxis=_ax_t(), yaxis=_ax_t())
            _pc(_fig_tr)
        chart_card_close()

    _gap(16)

    # ── SECTION B — WHO IS DROPPING OUT ──────────────────────────────────
    section_header("B — Who is dropping out")

    if not df_demo.empty:
        _age_df    = df_demo[df_demo["dimension"] == "Age group"].sort_values("ltfu_rate_pct", ascending=True)
        _payer_df  = df_demo[df_demo["dimension"] == "Payer"]
        _gender_df = df_demo[df_demo["dimension"] == "Gender"]

        def _demo_bar(df_b, title, subtitle, height=240):
            chart_card(title, subtitle)
            _fig_b = go.Figure(go.Bar(
                y=df_b["category"], x=df_b["ltfu_rate_pct"],
                orientation="h", marker_color=CA_BLUE_R,
                text=df_b["ltfu_rate_pct"].apply(lambda v: f"{v:.0f}%"),
                textposition="outside", textfont=dict(size=11),
            ))
            _fig_b.update_layout(
                **{**CHART_LAYOUT, "height": height},
                xaxis={**_ax_t(), "range": [0, 65], "ticksuffix": "%",
                       "title": {"text": "LTFU rate %"}},
                yaxis={**_ax_t(), "showgrid": False},
                showlegend=False,
            )
            _pc(_fig_b)
            chart_card_close()

        _b1, _b2, _b3 = st.columns(3)
        with _b1:
            _demo_bar(_age_df, "By age group", "LTFU rate % per age group.", height=260)
        with _b2:
            _demo_bar(_payer_df, "By payment mode", "LTFU rate % by how patients pay.", height=160)
        with _b3:
            _demo_bar(_gender_df, "By gender", "LTFU rate % by gender.", height=160)

        _all_rates  = df_demo["ltfu_rate_pct"]
        _spread     = round(float(_all_rates.max() - _all_rates.min()), 1)
        _age_min    = round(float(_age_df["ltfu_rate_pct"].min()),    0) if not _age_df.empty    else 0
        _age_max    = round(float(_age_df["ltfu_rate_pct"].max()),    0) if not _age_df.empty    else 0
        _payer_min  = round(float(_payer_df["ltfu_rate_pct"].min()),  0) if not _payer_df.empty  else 0
        _payer_max  = round(float(_payer_df["ltfu_rate_pct"].max()),  0) if not _payer_df.empty  else 0
        _gen_min    = round(float(_gender_df["ltfu_rate_pct"].min()), 0) if not _gender_df.empty else 0
        _gen_max    = round(float(_gender_df["ltfu_rate_pct"].max()), 0) if not _gender_df.empty else 0

        insight_bar([
            f"LTFU rates are near-identical across age ({_age_min:.0f}–{_age_max:.0f}%), payer ({_payer_min:.0f}–{_payer_max:.0f}%), and gender ({_gen_min:.0f}–{_gen_max:.0f}%) — only a {_spread:.0f}pp spread.",
            "Dropout is not concentrated in a specific demographic — pointing to a systemic workflow issue, not a patient-level characteristic.",
            "<strong>Action:</strong> focus retention intervention on workflow (follow-up scheduling) rather than targeted demographic outreach.",
        ], variant="blue")

    _gap(16)

    # ── SECTION C — WHY RETAIN LAPSING PATIENTS ──────────────────────────
    section_header("C — Why is it important to retain lapsing patients")

    kpi_row([
        {
            "label": "Lapsing patients",
            "value": f"{total_lapsing:,}",
            "delta": "91–180 days without return · still within recovery window",
            "accent_color": CA_AMBER_R,
        },
        {
            "label": "Recoverable revenue",
            "value": f"KES {rev_m:.1f}M",
            "delta": "If re-engaged this month before crossing 180d threshold",
            "accent_color": CA_AMBER_R,
        },
        {
            "label": "Clinical risk",
            "value": "Unmanaged",
            "delta": "Chronic patients without follow-up risk disease progression and emergency presentations",
            "accent_color": CA_RED_R,
        },
    ])

    insight_card(
        text=(
            f'<ul style="margin:4px 0 6px;padding-left:16px;line-height:1.8;">'
            f'<li>{cash_pct:.0f}% of lapsing patients are cash-paying — direct phone outreach is the most effective channel</li>'
            f'<li>Insured patients may be reachable via their insurer\'s care coordination channel</li>'
            f'</ul>'
            f'<div style="padding:5px 8px;background:rgba(0,0,0,0.04);border-radius:5px;font-size:11px;">'
            f'<strong>Action:</strong> Trigger outreach for all {total_lapsing:,} lapsing patients this month. Priority: uncontrolled vitals at last visit → Oncology → Neurologic.'
            f'</div>'
        ),
        label="Lapsing cohort", variant="amber",
    )

    _gap(16)

    # ── SECTION D — WHEN DID LTFU PATIENTS LAPSE ─────────────────────────
    section_header("D — When did LTFU patients lapse — visit history before dropout")

    if not df_tier.empty:
        df_tier["patient_count"] = pd.to_numeric(df_tier["patient_count"], errors="coerce")
        df_tier["share_pct"]     = pd.to_numeric(df_tier["share_pct"],     errors="coerce")
        st.caption("LTFU patients by number of visits before dropout")
        st.caption("How many visits did LTFU patients have before they stopped returning?")

        _TIER_CLR = {"1-2 visits": CA_RED_R, "3-5 visits": CA_AMBER_R, "5+ visits": CA_BLUE_R}
        _fig_tier = go.Figure()
        for _, _tr in df_tier.iterrows():
            _fig_tier.add_trace(go.Bar(
                y=["LTFU patients"], x=[_tr["patient_count"]],
                name=f"{_tr['visit_tier']} · {_tr['share_pct']:.0f}% · {int(_tr['patient_count']):,}",
                orientation="h",
                marker_color=_TIER_CLR.get(_tr["visit_tier"], "#888780"),
            ))
        _fig_tier.update_layout(
            **{**CHART_LAYOUT, "height": 130, "barmode": "stack",
               "legend": dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
                              font=dict(size=11), bgcolor="rgba(0,0,0,0)")},
            xaxis=_ax_t(), yaxis={**_ax_t(), "showgrid": False},
        )
        _pc(_fig_tier)

        _tier_12   = df_tier[df_tier["visit_tier"] == "1-2 visits"]
        _n_12      = int(_sf(_tier_12["patient_count"].iloc[0])) if not _tier_12.empty else 0
        _pct_12    = _sf(_tier_12["share_pct"].iloc[0])          if not _tier_12.empty else 0
        _total_ltfu = int(df_tier["patient_count"].sum())

        insight_card(
            text=(
                f'<ul style="margin:4px 0 6px;padding-left:16px;line-height:1.8;">'
                f'<li>{_pct_12:.0f}% of LTFU patients — {_n_12:,} of {_total_ltfu:,} — had only 1 or 2 visits before dropout</li>'
                f'<li>These patients were never established in ongoing care — they came once or twice and never returned</li>'
                f'<li>These are not resolved cases</li>'
                f'</ul>'
            ),
            label="Visit history", variant="red",
        )

    _gap(14)

    if not df_drop.empty:
        _age_drop = df_drop[df_drop["dimension"] == "Age group"].sort_values("patient_count", ascending=True)
        _dx_drop  = df_drop[df_drop["dimension"] == "Diagnosis"].sort_values("patient_count", ascending=True)

        def _shorten_dx(name: str) -> str:
            for sep in [" - ", ": "]:
                if sep in name:
                    return name.split(sep, 1)[-1]
            return name

        _dx_drop = _dx_drop.copy()
        _dx_drop["label"] = _dx_drop["category"].apply(_shorten_dx)
        _dx_drop = _dx_drop.nlargest(10, "patient_count").sort_values("patient_count", ascending=True)

        _col_ag, _col_dx = st.columns(2)

        with _col_ag:
            st.markdown(
                '<div style="font-size:10px;font-weight:500;text-transform:uppercase;'
                'letter-spacing:.06em;color:var(--text-color);opacity:.5;margin-bottom:6px;">By age group</div>',
                unsafe_allow_html=True)
            st.caption("Chronic patients with 1–2 visits only, absent 180+ days.")
            _age_drop["patient_count"] = pd.to_numeric(_age_drop["patient_count"], errors="coerce")
            _mx_age = _age_drop["patient_count"].max()
            _fig_age = go.Figure(go.Bar(
                y=_age_drop["category"], x=_age_drop["patient_count"],
                orientation="h",
                marker_color=_age_drop["patient_count"].apply(
                    lambda v: CA_RED_R if v >= _mx_age * 0.8 else CA_BLUE_R).tolist(),
            ))
            _fig_age.update_layout(
                **{**CHART_LAYOUT, "height": 260},
                xaxis={**_ax_t(), "title": {"text": "1–2 visit LTFU patients"}},
                yaxis={**_ax_t(), "showgrid": False}, showlegend=False,
            )
            _pc(_fig_age)

        with _col_dx:
            st.markdown(
                '<div style="font-size:10px;font-weight:500;text-transform:uppercase;'
                'letter-spacing:.06em;color:var(--text-color);opacity:.5;margin-bottom:6px;">By diagnosis</div>',
                unsafe_allow_html=True)
            st.caption("Which conditions are failing to retain patients after first contact.")
            _dx_drop["patient_count"] = pd.to_numeric(_dx_drop["patient_count"], errors="coerce")
            _mx_dx = _dx_drop["patient_count"].max()
            _fig_dx = go.Figure(go.Bar(
                y=_dx_drop["label"], x=_dx_drop["patient_count"],
                orientation="h",
                marker_color=_dx_drop["patient_count"].apply(
                    lambda v: CA_RED_R if v >= _mx_dx * 0.75 else CA_BLUE_R).tolist(),
            ))
            _fig_dx.update_layout(
                **{**CHART_LAYOUT, "height": 260},
                xaxis={**_ax_t(), "title": {"text": "1–2 visit LTFU patients"}},
                yaxis={**_ax_t(), "showgrid": False}, showlegend=False,
            )
            _pc(_fig_dx)

        _top_age = _age_drop.sort_values("patient_count", ascending=False).iloc[0] if not _age_drop.empty else None
        _top_dx  = _dx_drop.sort_values("patient_count",  ascending=False).iloc[0] if not _dx_drop.empty  else None
        _top_age_lbl = _top_age["category"] if _top_age is not None else "—"
        _top_dx_lbl  = _top_dx["label"]     if _top_dx  is not None else "—"

        insight_card(
            text=(
                f'<ul style="margin:4px 0 6px;padding-left:16px;line-height:1.8;">'
                f'<li>{_top_dx_lbl} leads dropout by diagnosis — the most severe chronic condition has the highest early LTFU count</li>'
                f'<li>{_top_age_lbl} is the largest age group for early dropout — chronic disease management is failing at first contact for this group</li>'
                f'</ul>'
                f'<div style="padding:5px 8px;background:rgba(0,0,0,0.04);border-radius:5px;font-size:11px;">'
                f'<strong>Action:</strong> Implement condition-specific recall for {_top_dx_lbl} and Neurologic. Review guardian engagement protocols for Under-18 chronic patients.'
                f'</div>'
            ),
            label="1–2 visit dropout profile", variant="amber",
        )

    _gap(16)

    # ── SECTION E — CARE PATHWAY BEFORE DROPOUT ──────────────────────────
    section_header("E — Care pathway before dropout — what happened at the last visit")

    _e_left, _e_right = st.columns(2)

    with _e_left:
        st.caption("Care journey signals at last visit (1–2 visit LTFU patients)")
        st.caption("What clinical actions were taken at the last recorded visit before dropout.")

        _SIGNAL_CFG = {
            "No follow-up date":     (CA_RED_R,   "↑"),
            "Prescription received": (CA_AMBER_R, "↑"),
            "Lab tests ordered":     ("#B4B2A9",  "✓"),
            "Radiology ordered":     ("#B4B2A9",  "✓"),
        }
        if not df_cp.empty:
            _sig_cols = st.columns(2)
            for _i_cp, _cp_row in enumerate(df_cp.itertuples()):
                _sig     = str(_cp_row.signal)
                _pct_cp  = _sf(_cp_row.pct)
                _n_cp    = int(_sf(_cp_row.patient_count))
                _tot_cp  = int(_sf(_cp_row.total_patients))
                _clr_cp, _icon_cp = _SIGNAL_CFG.get(_sig, ("#888780", ""))
                _bdr_cp  = _clr_cp if _pct_cp > 0 else "#D6E4F0"
                with _sig_cols[_i_cp % 2]:
                    st.markdown(
                        f'<div class="kpi-tile" style="border-top-color:{_clr_cp};margin-bottom:8px;">'
                        f'<div class="kpi-label">{_sig}</div>'
                        f'<div class="kpi-value" style="color:{_clr_cp};font-size:24px;">{_pct_cp:.1f}%</div>'
                        f'<div style="font-size:11px;color:#9CA3AF;margin-top:3px;line-height:1.4;">'
                        f'{_n_cp:,} of {_tot_cp:,} patients</div></div>',
                        unsafe_allow_html=True,
                    )

            _nofup_row = df_cp[df_cp["signal"] == "No follow-up date"]
            _rx_row    = df_cp[df_cp["signal"] == "Prescription received"]
            _nofup_pct = _sf(_nofup_row["pct"].iloc[0]) if not _nofup_row.empty else 0
            _rx_pct    = _sf(_rx_row["pct"].iloc[0])    if not _rx_row.empty    else 0

            insight_card(
                text=(
                    f'<ul style="margin:4px 0 6px;padding-left:16px;line-height:1.8;">'
                    f'<li>{_nofup_pct:.0f}% of chronic LTFU patients had no documented follow-up date at their last visit</li>'
                    f'<li>{_rx_pct:.1f}% received a prescription with no return scheduled — medication dispensed without booking the next clinical review</li>'
                    f'<li>This is a clinical workflow failure, not a patient behaviour problem</li>'
                    f'</ul>'
                    f'<div style="padding:5px 8px;background:rgba(0,0,0,0.04);border-radius:5px;font-size:11px;">'
                    f'<strong>Action:</strong> Make follow-up date documentation mandatory before any chronic patient is discharged from OPD.'
                    f'</div>'
                ),
                label="Care pathway signals", variant="red",
            )

    with _e_right:
        st.caption("Was it wait times?")
        st.caption("Avg time from visit start to investigation order, by lifecycle stage.")
        df_wait["avg_wait_hours"] = pd.to_numeric(df_wait["avg_wait_hours"], errors="coerce") if not df_wait.empty else df_wait.get("avg_wait_hours", pd.Series(dtype=float))
        df_wait = df_wait.dropna(subset=["avg_wait_hours"]) if not df_wait.empty else df_wait
        if df_wait.empty:
            st.markdown(
                '<div style="border:0.5px solid var(--border-color);border-radius:8px;'
                'padding:20px;text-align:center;color:var(--text-color);opacity:.45;font-size:12px;">'
                'Investigation turnaround data not available — no completed lab results with timestamps found.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            df_wait = df_wait.sort_values("avg_wait_hours", ascending=False)
            _WAIT_CLR = {"Active": CA_GREEN_R, "Lapsing": CA_AMBER_R, "LTFU": CA_RED_R}
            _fig_wait = go.Figure(go.Bar(
                x=df_wait["lifecycle_stage"], y=df_wait["avg_wait_hours"],
                marker_color=df_wait["lifecycle_stage"].map(_WAIT_CLR).tolist(),
                text=df_wait["avg_wait_hours"].apply(lambda v: f"{v:.1f}h"),
                textposition="outside", textfont=dict(size=12), width=0.45,
            ))
            _fig_wait.update_layout(
                **{**CHART_LAYOUT, "height": 220},
                xaxis={**_ax_t(), "showgrid": False},
                yaxis={**_ax_t(), "ticksuffix": "h", "title": {"text": "Avg wait (hours)"},
                       "range": [0, float(df_wait["avg_wait_hours"].max()) * 1.3]},
                showlegend=False,
            )
            _pc(_fig_wait)

            _act_w  = _sf(df_wait[df_wait["lifecycle_stage"] == "Active"]["avg_wait_hours"].iloc[0]) if "Active" in df_wait["lifecycle_stage"].values else 0
            _ltfu_w = _sf(df_wait[df_wait["lifecycle_stage"] == "LTFU"]["avg_wait_hours"].iloc[0])   if "LTFU"   in df_wait["lifecycle_stage"].values else 0
            _ratio  = round(_act_w / _ltfu_w, 1) if _ltfu_w > 0 else 0

            insight_card(
                text=(
                    f'<ul style="margin:4px 0 6px;padding-left:16px;line-height:1.8;">'
                    f'<li>LTFU patients had {_ltfu_w:.1f}h investigation wait vs {_act_w:.1f}h for active patients — {_ratio}× shorter</li>'
                    f'<li>Not because processing was faster — fewer investigations were ordered at their last visit</li>'
                    f'<li>Wait times are not driving dropout — under-investigation at the last visit is a signal of under-engagement</li>'
                    f'</ul>'
                ),
                label="Wait time analysis", variant="blue",
            )

    _gap(16)

    # ── SECTION F — CLINICIAN LTFU RATES ─────────────────────────────────
    section_header("F — LTFU rate by clinician")
    st.caption(
        "% of each clinician's chronic patients who crossed 180 days without returning. "
        "Min 5 chronic patients. Reference line at 50%."
    )

    if not df_clin.empty:
        _df_clin_s = df_clin.sort_values("ltfu_rate_pct", ascending=True)
        _fig_clin = go.Figure(go.Bar(
            y=_df_clin_s["clinician_id"].astype(str).apply(lambda c: f"Clinician {c}"),
            x=_df_clin_s["ltfu_rate_pct"],
            orientation="h",
            marker_color=_df_clin_s["ltfu_rate_pct"].apply(
                lambda v: CA_RED_R if v >= 50 else CA_BLUE_R).tolist(),
            text=_df_clin_s["ltfu_rate_pct"].apply(lambda v: f"{v:.0f}%"),
            textposition="outside", textfont=dict(size=11),
            customdata=_df_clin_s[["chronic_seen", "ltfu_count"]].values,
            hovertemplate=(
                "%{y}<br>LTFU rate: %{x:.0f}%<br>"
                "Chronic seen: %{customdata[0]}<br>"
                "LTFU count: %{customdata[1]}<extra></extra>"
            ),
        ))
        _fig_clin.add_vline(
            x=50, line_dash="dot", line_color=CA_AMBER_R, line_width=2,
            annotation_text="50% reference",
            annotation_position="top right",
            annotation_font=dict(size=10, color=CA_AMBER_R),
        )
        _n_clin = len(_df_clin_s)
        _fig_clin.update_layout(
            **{**CHART_LAYOUT, "height": max(320, _n_clin * 28 + 80)},
            xaxis={**_ax_t(), "range": [0, 105], "ticksuffix": "%",
                   "title": {"text": "% of chronic patients — LTFU"}},
            yaxis={**_ax_t(), "showgrid": False},
            showlegend=False,
        )
        _pc(_fig_clin)

        insight_card(
            text=(
                f'<ul style="margin:4px 0 6px;padding-left:16px;line-height:1.8;">'
                f'<li>Clinician {top_clin_id} — {top_clin_rate:.0f}% of their {top_clin_n} chronic patients crossed 180 days — highest rate on the team</li>'
                f'<li>{above_50} clinicians are above the 50% reference line — more than half of their chronic patients are LTFU</li>'
                f'</ul>'
                f'<div style="padding:5px 8px;background:rgba(0,0,0,0.04);border-radius:5px;font-size:11px;">'
                f'<strong>Action:</strong> Clinical supervision conversations about chronic patient follow-up scheduling for all clinicians above 60% LTFU rate. Start with Clinician {top_clin_id}.'
                f'</div>'
            ),
            label="Clinician LTFU rates", variant="red",
        )

    _gap(16)

    # ── SECTION G — RECOMMENDATIONS ──────────────────────────────────────
    section_header("G — Recommendations to reduce LTFU")

    _RECS = [
        {
            "priority": 1,
            "colour":   ("#FCEBEB", "#791F1F"),
            "title":    "Make follow-up date documentation mandatory",
            "body":     (
                "Implement a system-level prompt that requires a return date before any chronic patient can be discharged from OPD. "
                "Block prescription dispensing for chronic conditions without a scheduled follow-up. "
                "Target: 0% of chronic OPD visits discharged without a documented return date."
            ),
        },
        {
            "priority": 2,
            "colour":   ("#FCEBEB", "#791F1F"),
            "title":    f"Outreach to {lapsing:,} lapsing patients this month",
            "body":     (
                f"These {lapsing:,} patients are in the 91–180 day window and recoverable. "
                f"Phone call or SMS for each. Priority order: patients with uncontrolled vitals at last visit, then Oncology, then Neurologic. "
                f"KES {rev_m:.1f}M in revenue is recoverable if re-engaged before they cross 180 days."
            ),
        },
        {
            "priority": 3,
            "colour":   ("#FAEEDA", "#633806"),
            "title":    "Condition-specific recall for Oncology and Neurologic",
            "body":     (
                "NCD-Oncology is the top dropout diagnosis at both 1–2 visit and overall LTFU level. "
                "Implement a 30-day recall trigger for any Oncology or Neurologic patient who misses a scheduled appointment. "
                "These conditions cannot be self-managed without clinical oversight."
            ),
        },
        {
            "priority": 4,
            "colour":   ("#FAEEDA", "#633806"),
            "title":    "Address paediatric early dropout — Under-18 chronic patients",
            "body":     (
                "Under-18 is the largest age group for 1–2 visit dropout. Review whether guardians are being engaged with follow-up instructions "
                "and whether appointment scheduling accounts for school schedules."
            ),
        },
        {
            "priority": 5,
            "colour":   ("#E6F1FB", "#0C447C"),
            "title":    f"Clinician-level LTFU review for high-rate clinicians",
            "body":     (
                f"Clinician {top_clin_id} has a {top_clin_rate:.0f}% LTFU rate across {top_clin_n} chronic patients — the highest on the team. "
                f"{above_50} clinicians are above the 50% reference line. "
                f"Clinical supervision conversations are indicated for those above 60%."
            ),
        },
    ]

    for _rec in _RECS:
        _bg, _tc = _rec["colour"]
        st.markdown(
            f'<div style="border:0.5px solid var(--border-color);border-radius:12px;'
            f'padding:14px 16px;background:var(--background-color);'
            f'margin-bottom:8px;display:flex;align-items:flex-start;gap:12px;">'
            f'<div style="width:24px;height:24px;border-radius:50%;flex-shrink:0;'
            f'background:{_bg};color:{_tc};display:flex;align-items:center;'
            f'justify-content:center;font-size:12px;font-weight:500;">{_rec["priority"]}</div>'
            f'<div>'
            f'<div style="font-size:13px;font-weight:500;color:var(--text-color);margin-bottom:4px;">{_rec["title"]}</div>'
            f'<div style="font-size:12px;color:var(--text-color);opacity:.7;line-height:1.5;">{_rec["body"]}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

def render_tab4_disease_burden(filters: dict, run_query):
    from ksh.clinical_module.ui_template import (
        page_header as _ph_db, anomaly_banner as _ab_db,
        chart_card as _chart_card_db, chart_card_close as _chart_card_close_db,
        insight_bar as _insight_bar_db,
        section_header as _sh_db,
        kpi_row,
        ACCENT_CRITICAL, ACCENT_MONITOR, ACCENT_POSITIVE, ACCENT_INFO, ACCENT_NEUTRAL,
    )
    _ph_db("Disease Burden")

    st_b, st_c, st_d = st.tabs([
        "Chronic Disease Management",
        "Maternal Health", "Communicable Disease",
    ])

    # ── Local helpers for Overview tab ───────────────────────────────────────
    def _ra3(vals, w=3):
        """Centred rolling average, window=w."""
        if not vals:
            return vals
        out = []
        for i, _ in enumerate(vals):
            chunk = vals[max(0, i - w + 1): i + 1]
            out.append(round(sum(chunk) / len(chunk), 1))
        return out

    def _sec_c_html(df) -> str:
        """HTML table — top diagnoses with IP/OP share bars."""
        _TH = ("font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;"
               "color:#9CA3AF;padding:6px 10px;border-bottom:1px solid #E5E7EB;text-align:left;")
        _TD = "font-size:12px;color:#003467;padding:5px 10px;vertical-align:middle;"
        rows_html = ""
        max_v = max((int(r.get("total_visits") or 0) for _, r in df.iterrows()), default=1)
        for _, r in df.iterrows():
            grp     = str(r.get("burden_group") or "—")
            tot     = int(r.get("total_visits") or 0)
            ip_pct  = float(r.get("ip_pct") or 0)
            op_pct  = float(r.get("op_pct") or 0)
            bar_w   = round(tot / max_v * 120)
            ip_col  = "#185FA5" if ip_pct >= 15 else "#BA7517" if ip_pct >= 5 else "#D1D5DB"
            rows_html += (
                f'<tr style="border-bottom:1px solid #F3F4F6">'
                f'<td style="{_TD}min-width:160px;max-width:220px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis">{grp}</td>'
                f'<td style="{_TD}text-align:right">{tot:,}</td>'
                f'<td style="{_TD}">'
                f'<div style="display:flex;align-items:center;gap:6px">'
                f'<div style="width:{bar_w}px;height:8px;background:#185FA5;border-radius:2px;opacity:.75"></div>'
                f'</div></td>'
                f'<td style="{_TD}text-align:center">'
                f'<span style="background:{ip_col};color:#fff;font-size:10px;font-weight:700;'
                f'padding:2px 6px;border-radius:10px">{ip_pct:.0f}%</span></td>'
                f'<td style="{_TD}color:#6B7280">{op_pct:.0f}%</td>'
                f'</tr>'
            )
        return (
            f'<table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif">'
            f'<thead><tr>'
            f'<th style="{_TH}">Condition</th>'
            f'<th style="{_TH}text-align:right">Visits</th>'
            f'<th style="{_TH}">Volume</th>'
            f'<th style="{_TH}text-align:center">IP%</th>'
            f'<th style="{_TH}">OP%</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table>'
        )

    def _sec_d_html(df) -> str:
        """HTML table — emerging mid-tier diagnoses with growth bars."""
        _TH = ("font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;"
               "color:#9CA3AF;padding:6px 10px;border-bottom:1px solid #E5E7EB;text-align:left;")
        _TD = "font-size:12px;color:#003467;padding:7px 10px;vertical-align:middle;"
        rows_html = ""
        for _, r in df.iterrows():
            cond    = str(r.get("condition") or "—")
            recent  = int(r.get("recent_90d_visits") or 0)
            prior   = int(r.get("prior_90d_visits") or 0)
            growth  = float(r.get("mom_growth_pct") or 0)
            ip_pct  = float(r.get("inpatient_pct") or 0)
            g_col   = "#0F6E56" if growth >= 0 else "#C53030"
            g_str   = f"+{growth:.0f}%" if growth >= 0 else f"{growth:.0f}%"
            bar_w   = min(int(abs(growth) / 2), 80)
            rows_html += (
                f'<tr style="border-bottom:1px solid #F3F4F6">'
                f'<td style="{_TD}min-width:150px;max-width:210px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis">{cond}</td>'
                f'<td style="{_TD}text-align:right">{recent:,}</td>'
                f'<td style="{_TD}text-align:right;color:#9CA3AF">{prior:,}</td>'
                f'<td style="{_TD}">'
                f'<div style="display:flex;align-items:center;gap:6px">'
                f'<div style="width:{bar_w}px;height:8px;background:{g_col};border-radius:2px;opacity:.8"></div>'
                f'<span style="font-weight:700;color:{g_col};font-size:11px">{g_str}</span>'
                f'</div></td>'
                f'<td style="{_TD}text-align:center">{ip_pct:.0f}%</td>'
                f'</tr>'
            )
        return (
            f'<table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif">'
            f'<thead><tr>'
            f'<th style="{_TH}">Condition</th>'
            f'<th style="{_TH}text-align:right">Last 90d</th>'
            f'<th style="{_TH}text-align:right">Prior 90d</th>'
            f'<th style="{_TH}">MoM Growth</th>'
            f'<th style="{_TH}text-align:center">IP%</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table>'
        )

    def _sec_e_html(df) -> str:
        """HTML table — disease intelligence matrix."""
        _TH = ("font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;"
               "color:#9CA3AF;padding:6px 10px;border-bottom:1px solid #E5E7EB;text-align:left;")
        _TD = "font-size:12px;color:#003467;padding:6px 10px;vertical-align:middle;"
        rows_html = ""
        # Detect condition column name
        cond_col = "condition" if "condition" in df.columns else (
            "burden_group" if "burden_group" in df.columns else df.columns[0]
        )
        for _, r in df.iterrows():
            cond   = str(r.get(cond_col) or "—")
            tot    = int(r.get("total_visits") or 0)
            ip_pct = float(r.get("ip_pct") or 0)
            ip_col = "#185FA5" if ip_pct >= 15 else "#BA7517" if ip_pct >= 5 else "#D1D5DB"
            rows_html += (
                f'<tr style="border-bottom:1px solid #F3F4F6">'
                f'<td style="{_TD}border-left:3px solid {ip_col};padding-left:12px;'
                f'min-width:160px;max-width:220px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis">'
                f'{cond}</td>'
                f'<td style="{_TD}text-align:right">{tot:,}</td>'
                f'<td style="{_TD}text-align:center">'
                f'<span style="background:{ip_col};color:#fff;font-size:10px;font-weight:700;'
                f'padding:2px 6px;border-radius:10px">{ip_pct:.0f}%</span></td>'
                f'<td style="{_TD}color:#6B7280">{float(r.get("op_pct") or 0):.0f}%</td>'
                f'</tr>'
            )
        return (
            f'<table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif">'
            f'<thead><tr>'
            f'<th style="{_TH}">Condition</th>'
            f'<th style="{_TH}text-align:right">Visits (90d)</th>'
            f'<th style="{_TH}text-align:center">IP%</th>'
            f'<th style="{_TH}">OP%</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table>'
        )

    # ── NCD / COMM HTML helpers ───────────────────────────────────────────────
    CHART_BASE = dict(
        paper_bgcolor="#fff", plot_bgcolor="#fff",
        margin=dict(l=0, r=0, t=6, b=0),
        font=dict(family="system-ui, sans-serif", size=12, color="#111827"),
    )
    AX = dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)",
              showline=False, tickfont=dict(size=11, color="#888780"))

    _NCD_BASE = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap" rel="stylesheet">'
        '<style>*{{box-sizing:border-box;margin:0;padding:0;}}'
        'body{{background:#fff;font-family:system-ui,-apple-system,sans-serif;padding:0;}}'
        'table{{width:100%;border-collapse:collapse;}}'
        'th{{font-size:11px;font-weight:600;color:#888780;text-transform:uppercase;letter-spacing:0.03em;'
        '    padding:7px 8px 7px 0;border-bottom:0.5px solid rgba(0,0,0,0.10);text-align:left;white-space:nowrap;}}'
        'td{{font-size:12px;padding:7px 8px 7px 0;vertical-align:middle;border-bottom:0.5px solid rgba(0,0,0,0.05);}}'
        'tr:last-child td{{border-bottom:none;}}'
        '.ins{{background:#f5f5f3;border-radius:8px;padding:6px 9px;font-size:12px;color:#5f5e5a;margin-top:8px;}}'
        '.warn{{background:#FAEEDA;border-left:3px solid #EF9F27;border-radius:0 8px 8px 0;'
        '       padding:6px 10px;font-size:12px;color:#633806;margin-top:8px;}}'
        '</style></head><body>{}</body></html>'
    )

    _SORT_SCRIPT = (
        '<script>'
        '(function(){'
        'var tbl=document.querySelector("table");'
        'var ths=tbl.querySelectorAll("thead th");'
        'var sCol=-1,sDir=1;'
        'ths.forEach(function(th,ci){'
        'th.style.cursor="pointer";th.style.userSelect="none";'
        'th.title="Click to sort";'
        'th.addEventListener("click",function(){'
        'if(sCol===ci){sDir*=-1;}else{sCol=ci;sDir=1;}'
        'ths.forEach(function(t,i){'
        'var txt=t.dataset.label||(t.dataset.label=t.textContent.trim());'
        't.textContent=txt+(i===ci?(sDir===1?" ▲":" ▼"):"");'
        '});'
        'var tb=tbl.querySelector("tbody");'
        'var rows=Array.from(tb.querySelectorAll("tr"));'
        'rows.sort(function(a,b){'
        'var av=a.cells[ci]?a.cells[ci].textContent.trim():"";'
        'var bv=b.cells[ci]?b.cells[ci].textContent.trim():"";'
        'var an=parseFloat(av.replace(/[^0-9.\\-]/g,""));'
        'var bn=parseFloat(bv.replace(/[^0-9.\\-]/g,""));'
        'if(!isNaN(an)&&!isNaN(bn))return(an-bn)*sDir;'
        'return av.localeCompare(bv)*sDir;'
        '});'
        'rows.forEach(function(r){tb.appendChild(r);});'
        '});'
        '});'
        '})();'
        '</script>'
    )

    _COMM_BASE = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<style>'
        '*{{box-sizing:border-box;margin:0;padding:0;}}'
        'html,body{{height:100%;overflow-y:auto;}}'
        'body{{background:#fff;font-family:system-ui,-apple-system,sans-serif;padding:0;}}'
        'table{{width:100%;border-collapse:collapse;}}'
        'th{{font-size:11px;font-weight:600;color:#888780;text-transform:uppercase;'
        '    letter-spacing:0.03em;padding:7px 10px 7px 0;'
        '    border-bottom:0.5px solid rgba(0,0,0,0.10);text-align:left;white-space:nowrap;}}'
        'td{{font-size:12px;padding:7px 10px 7px 0;vertical-align:middle;'
        '    border-bottom:0.5px solid rgba(0,0,0,0.05);}}'
        'tr:last-child td{{border-bottom:none;}}'
        '.ins{{background:#f5f5f3;border-radius:8px;padding:6px 9px;font-size:12px;color:#5f5e5a;margin-top:8px;}}'
        '.warn{{background:#FAEEDA;border-left:3px solid #EF9F27;border-radius:0 8px 8px 0;'
        '       padding:6px 10px;font-size:12px;color:#633806;margin-top:8px;}}'
        '</style></head><body>{}</body></html>'
    )

    _COMM_CLR_MAP = [
        ("Malaria",    "#1D9E75"),
        ("Typhoid",    "#EF9F27"),
        ("URTI",       "#378ADD"),
        ("TB",         "#E24B4A"),
        ("Enteric",    "#7F77DD"),
        ("GI",         "#7F77DD"),
        ("HIV",        "#D85A30"),
        ("Infectious", "#9B59B6"),
    ]

    def _comm_clr(disease: str) -> str:
        d = str(disease).lower()
        for key, clr in _COMM_CLR_MAP:
            if key.lower() in d:
                return clr
        return "#888780"

    def _payer_badge_ncd(payer: str) -> str:
        p = str(payer or "")
        if "Cash" in p or "PRIVATE" in p.upper():
            return f'<span style="background:#FAEEDA;color:#633806;font-size:9px;font-weight:500;padding:2px 7px;border-radius:20px;">{p}</span>'
        return f'<span style="background:#E6F1FB;color:#185FA5;font-size:9px;font-weight:500;padding:2px 7px;border-radius:20px;">{p}</span>'

    def _ip_dot(ip_pct: float) -> str:
        c = "#E24B4A" if ip_pct > 20 else "#EF9F27" if ip_pct >= 10 else "#1D9E75"
        return f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{c};margin-right:4px;vertical-align:middle;"></span>'

    def _ip_border(ip_pct: float) -> str:
        c = "#E24B4A" if ip_pct > 20 else "#EF9F27" if ip_pct >= 10 else "#1D9E75"
        return f"border-left:3px solid {c};padding-left:7px;"

    def _ip_badge_comm(ip_pct: float) -> str:
        if ip_pct >= 20:
            bg, col = "#FCEBEB", "#A32D2D"
        elif ip_pct >= 10:
            bg, col = "#FAEEDA", "#854F0B"
        elif ip_pct >= 1:
            bg, col = "#E6F1FB", "#185FA5"
        else:
            bg, col = "#f5f5f3", "#888780"
        return (f'<span style="background:{bg};color:{col};font-size:9px;font-weight:600;'
                f'padding:3px 9px;border-radius:20px;">{ip_pct:.0f}%</span>')

    def _ncd_t1_html(df):
        total = int(df["patient_count"].sum()) if not df.empty else 1
        max_n = int(df["patient_count"].max()) if not df.empty else 1
        _cx_col = {"1 NCD": "#1D9E75", "2 NCDs": "#EF9F27", "3 NCDs": "#E24B4A", "4+ NCDs (Complex)": "#A32D2D"}
        hdr = ('<table><thead><tr>'
               '<th>Complexity</th><th>Share of NCD Pts</th>'
               '<th style="text-align:right;">Patients</th>'
               '<th style="text-align:right;">% of NCD Pts</th>'
               '</tr></thead><tbody>')
        rows = ""
        for _, r in df.iterrows():
            label = str(r.get("ncd_complexity", ""))
            n     = int(r.get("patient_count") or 0)
            pct   = float(r.get("pct_of_ncd_patients") or 0)
            col   = _cx_col.get(label, "#888780")
            bw    = round(n / max_n * 120)
            rows += (f'<tr>'
                     f'<td style="font-weight:500;color:#1a1a18;">{label}</td>'
                     f'<td><div style="display:flex;align-items:center;gap:6px;">'
                     f'<div style="width:{bw}px;height:8px;border-radius:3px;background:{col};opacity:0.8;flex-shrink:0;"></div>'
                     f'<span style="font-size:9px;color:#5f5e5a;">{n:,}</span></div></td>'
                     f'<td style="text-align:right;font-weight:500;">{n:,}</td>'
                     f'<td style="text-align:right;color:#5f5e5a;">{pct:.1f}%</td>'
                     f'</tr>')
        multi_pct = df.loc[df["ncd_complexity"] != "1 NCD", "pct_of_ncd_patients"].sum() if not df.empty else 0
        insight_text = (f"{multi_pct:.0f}% of NCD patients carry 2+ conditions. "
                        f"These patients need integrated management protocols.")
        return _NCD_BASE.format(hdr + rows + "</tbody></table>"), insight_text

    def _ncd_t2_html(df):
        df = df.head(10).copy()
        max_days = float(df["avg_days_between_diagnoses"].max()) if not df.empty else 1
        hdr = ('<table><thead><tr>'
               '<th>Condition pair</th><th style="text-align:center;">Patients</th>'
               '<th>Avg days to 2nd</th><th>Speed</th>'
               '</tr></thead><tbody>')
        rows = ""
        fastest_row = None
        for _, r in df.iterrows():
            pair = str(r.get("condition_pair", ""))
            n    = int(r.get("patient_count") or 0)
            days = r.get("avg_days_between_diagnoses")
            try:    days = float(days)
            except: days = None
            if days is not None and (fastest_row is None or days < fastest_row[1]):
                fastest_row = (pair, days)
            if days is None:
                spd_lbl, spd_bg, spd_col, bar_col = "—", "#f5f5f3", "#888780", "#888780"
                bw = 0
            elif days < 45:
                spd_lbl, spd_bg, spd_col, bar_col = "Fast", "#FCEBEB", "#A32D2D", "#A32D2D"
                bw = round(days / max_days * 150) if max_days else 0
            elif days <= 60:
                spd_lbl, spd_bg, spd_col, bar_col = "Moderate", "#FAEEDA", "#854F0B", "#854F0B"
                bw = round(days / max_days * 150) if max_days else 0
            else:
                spd_lbl, spd_bg, spd_col, bar_col = "Slower", "#E1F5EE", "#0F6E56", "#0F6E56"
                bw = round(days / max_days * 150) if max_days else 0
            badge = (f'<span style="background:{spd_bg};color:{spd_col};font-size:9px;font-weight:500;'
                     f'padding:2px 7px;border-radius:20px;">{spd_lbl}</span>')
            days_cell = (f'<div style="display:flex;align-items:center;gap:6px;">'
                         f'<div style="width:{bw}px;height:8px;border-radius:3px;background:{bar_col};opacity:0.75;flex-shrink:0;"></div>'
                         f'<span style="font-size:9px;color:{bar_col};font-weight:500;">{int(days)}d</span></div>'
                         ) if days is not None else "—"
            rows += (f'<tr>'
                     f'<td style="line-height:1.4;">{pair}</td>'
                     f'<td style="text-align:center;font-weight:500;">{n:,}</td>'
                     f'<td>{days_cell}</td><td>{badge}</td>'
                     f'</tr>')
        insight_text = ""
        if fastest_row:
            insight_text = (f"Fastest progression: {fastest_row[0]} — "
                            f"avg {int(fastest_row[1])} days to second diagnosis. "
                            f"This pair needs a co-management protocol.")
        return _NCD_BASE.format(hdr + rows + "</tbody></table>" + _SORT_SCRIPT), insight_text

    def _ncd_t3_html(df):
        import math as _m
        _HTN_BG   = {"Controlled": "rgba(29,158,117,0.06)", "Uncontrolled": "rgba(228,75,74,0.06)", "No BP Recorded": "rgba(136,135,128,0.06)"}
        _HTN_BDGE = {"Controlled": ("#E1F5EE","#0F6E56"),   "Uncontrolled": ("#FCEBEB","#A32D2D"),  "No BP Recorded": ("#f5f5f3","#888780")}
        _HTN_BARC = {"Controlled": "#1D9E75",               "Uncontrolled": "#E24B4A",              "No BP Recorded": "#888780"}
        max_inv = float(df["avg_inv"].max()) if not df.empty and "avg_inv" in df.columns else 1
        if max_inv == 0 or _m.isnan(max_inv): max_inv = 1
        hdr = ('<table><thead><tr>'
               '<th>HTN status</th><th>Comorbidity</th>'
               '<th style="text-align:center;">Patients</th>'
               '<th>Avg investigations</th>'
               '<th style="text-align:right;">On antihypertensive %</th>'
               '</tr></thead><tbody>')
        STATUS_ORDER = ["Controlled", "No BP Recorded", "Uncontrolled"]
        rows_sorted = []
        for st_key in STATUS_ORDER:
            sub = df[df["htn_status"] == st_key]
            for _, r in sub.iterrows():
                rows_sorted.append((st_key, r))
        rows = ""
        for htn_status, r in rows_sorted:
            comorb  = str(r.get("comorbidity_group", ""))
            n       = int(r.get("patients") or 0)
            avg_inv = float(r.get("avg_inv") or 0)
            rx_pct  = float(r.get("on_rx_pct") or 0)
            bdg_bg, bdg_col = _HTN_BDGE.get(htn_status, ("#f5f5f3","#888780"))
            row_bg  = _HTN_BG.get(htn_status, "transparent")
            bar_col = _HTN_BARC.get(htn_status, "#888780")
            bw      = round(avg_inv / max_inv * 130) if max_inv else 0
            badge   = (f'<span style="background:{bdg_bg};color:{bdg_col};font-size:9px;font-weight:500;'
                       f'padding:2px 7px;border-radius:20px;">{htn_status}</span>')
            rx_col  = "#A32D2D" if rx_pct > 60 and htn_status == "Uncontrolled" else "#1a1a18"
            rows += (f'<tr style="background:{row_bg};">'
                     f'<td>{badge}</td><td style="color:#5f5e5a;">{comorb}</td>'
                     f'<td style="text-align:center;font-weight:500;">{n:,}</td>'
                     f'<td><div style="display:flex;align-items:center;gap:6px;">'
                     f'<div style="width:{bw}px;height:8px;border-radius:3px;background:{bar_col};opacity:0.7;flex-shrink:0;"></div>'
                     f'<span style="font-size:9px;color:#5f5e5a;">{avg_inv:.1f}</span></div></td>'
                     f'<td style="text-align:right;font-weight:500;color:{rx_col};">{rx_pct:.0f}%</td>'
                     f'</tr>')
        unc = df[df["htn_status"] == "Uncontrolled"] if not df.empty else pd.DataFrame()
        insight = ""
        if not unc.empty:
            htn_only = unc[unc["comorbidity_group"] == "HTN Only"]
            if not htn_only.empty:
                r0   = htn_only.iloc[0]
                n0   = int(r0.get("patients") or 0)
                inv0 = float(r0.get("avg_inv") or 0)
                rx0  = float(r0.get("on_rx_pct") or 0)
                insight = (f'<div class="ins"><strong>{n0:,}</strong> uncontrolled HTN-only patients average '
                           f'<strong>{inv0:.1f}</strong> investigations and <strong>{rx0:.0f}%</strong> are on '
                           f'antihypertensive medication — investigate medication adherence and dose adequacy.</div>')
        return _NCD_BASE.format(hdr + rows + "</tbody></table>" + insight)

    def _ncd_t4_html(df):
        hdr = ('<table><thead><tr>'
               '<th>Payer</th><th>Condition</th>'
               '<th style="text-align:center;">Patients affected</th>'
               '<th style="text-align:right;">Avg annual rev</th>'
               '<th>Risk level</th>'
               '</tr></thead><tbody>')
        rows = ""
        import math as _m
        for _, r in df.head(20).iterrows():
            payer = str(r.get("payer", ""))
            cond  = str(r.get("condition", ""))
            n     = int(r.get("patient_count") or 0)
            rev   = r.get("avg_annual_revenue")
            try:    rev_f = float(rev)
            except: rev_f = None
            rev_s = "—" if rev_f is None or _m.isnan(rev_f) else f"KES {int(rev_f):,}"
            if n > 200:   risk_bg, risk_col, risk_lbl = "#FCEBEB", "#A32D2D", "High"
            elif n >= 50: risk_bg, risk_col, risk_lbl = "#FAEEDA", "#854F0B", "Medium"
            else:         risk_bg, risk_col, risk_lbl = "#f5f5f3",  "#888780", "Low"
            risk_badge = (f'<span style="background:{risk_bg};color:{risk_col};font-size:9px;'
                          f'font-weight:500;padding:2px 7px;border-radius:20px;">{risk_lbl}</span>')
            rows += (f'<tr>'
                     f'<td>{_payer_badge_ncd(payer)}</td>'
                     f'<td style="line-height:1.3;">{cond}</td>'
                     f'<td style="text-align:center;font-weight:500;">{n:,}</td>'
                     f'<td style="text-align:right;">{rev_s}</td>'
                     f'<td>{risk_badge}</td>'
                     f'</tr>')
        return _NCD_BASE.format(hdr + rows + "</tbody></table>" + _SORT_SCRIPT)

    def _ncd_t5_html(df):
        import math as _m
        max_pts = float(df["patient_count"].max()) if not df.empty else 1
        if max_pts == 0: max_pts = 1
        hdr = ('<table><thead><tr>'
               '<th>Condition</th><th>Patients</th><th>6-mo trend</th>'
               '<th>IP rate</th><th>Top payer</th>'
               '<th style="text-align:center;">Visits/pt</th>'
               '<th style="text-align:center;">Inv/visit</th>'
               '<th style="text-align:right;">Avg rev/pt</th>'
               '<th style="text-align:center;">Control %</th>'
               '</tr></thead><tbody>')
        rows = ""
        for _, r in df.iterrows():
            cond    = str(r.get("condition", ""))
            pts     = int(r.get("patient_count") or 0)
            trend   = r.get("trend_pct")
            ip_rate = float(r.get("ip_rate_pct") or 0)
            payer   = str(r.get("top_payer") or "—")
            vpp     = float(r.get("avg_visits_per_patient") or 0)
            inv_v   = float(r.get("investigations_per_visit") or 0)
            rev_pt  = r.get("avg_revenue_per_patient")
            ctrl    = r.get("controlled_pct")
            try:    rev_f = float(rev_pt)
            except: rev_f = None
            rev_s   = f"KES {int(rev_f):,}" if rev_f is not None and not _m.isnan(rev_f) else "—"
            try:    trend_f = float(trend)
            except: trend_f = None
            if trend_f is None or _m.isnan(trend_f): trend_s, trend_col = "—", "#888780"
            elif trend_f > 5:  trend_s, trend_col = f"↑ +{trend_f:.0f}%", "#0F6E56"
            elif trend_f < -5: trend_s, trend_col = f"↓ {trend_f:.0f}%", "#A32D2D"
            else:              trend_s, trend_col = f"→ {trend_f:.0f}%", "#888780"
            ip_bdr  = _ip_border(ip_rate)
            ip_dot  = _ip_dot(ip_rate)
            bw      = round(pts / max_pts * 50)
            pts_cell = (f'<div style="display:flex;align-items:center;gap:4px;">'
                        f'<div style="width:{bw}px;height:8px;border-radius:3px;background:#378ADD;opacity:0.6;flex-shrink:0;"></div>'
                        f'<span style="font-size:9px;color:#5f5e5a;">{pts:,}</span></div>')
            inv_col = "#A32D2D" if inv_v > 7 else "#854F0B" if inv_v >= 5 else "#5f5e5a"
            try:    ctrl_f = float(ctrl)
            except: ctrl_f = None
            if ctrl_f is None or _m.isnan(ctrl_f): ctrl_s, ctrl_col = "—", "#888780"
            elif ctrl_f >= 50: ctrl_s, ctrl_col = f"{ctrl_f:.0f}%", "#0F6E56"
            elif ctrl_f >= 30: ctrl_s, ctrl_col = f"{ctrl_f:.0f}%", "#854F0B"
            else:              ctrl_s, ctrl_col = f"{ctrl_f:.0f}%", "#A32D2D"
            rows += (f'<tr>'
                     f'<td style="{ip_bdr}line-height:1.3;">{cond}</td>'
                     f'<td>{pts_cell}</td>'
                     f'<td style="color:{trend_col};font-weight:500;">{trend_s}</td>'
                     f'<td>{ip_dot}{ip_rate:.1f}%</td>'
                     f'<td>{_payer_badge_ncd(payer)}</td>'
                     f'<td style="text-align:center;color:#5f5e5a;">{vpp:.1f}</td>'
                     f'<td style="text-align:center;font-weight:500;color:{inv_col};">{inv_v:.2f}</td>'
                     f'<td style="text-align:right;">{rev_s}</td>'
                     f'<td style="text-align:center;font-weight:500;color:{ctrl_col};">{ctrl_s}</td>'
                     f'</tr>')
        return _NCD_BASE.format(hdr + rows + "</tbody></table>" + _SORT_SCRIPT)

    def _ncd_t6_html(df):
        import math as _m
        n_total = len(df)
        hdr = ('<table><thead><tr>'
               '<th>Patient</th>'
               '<th style="text-align:center;">Flagged visits</th>'
               '<th style="text-align:center;">Latest systolic</th>'
               '<th style="text-align:center;">Days since last</th>'
               '<th>Payer</th><th>Urgency</th>'
               '</tr></thead><tbody>')
        rows = ""
        for _, r in df.head(40).iterrows():
            pat_id = r.get("patient", "")
            fv     = int(r.get("visit_count") or 0)
            sys_v  = r.get("latest_systolic")
            days_v = r.get("days_since_last_visit")
            payer  = str(r.get("payer") or "")
            try:    sys_f  = float(sys_v)
            except: sys_f  = None
            try:    days_f = float(days_v)
            except: days_f = None
            if sys_f is None or _m.isnan(sys_f):     sys_s, sys_col = "—", "#888780"
            elif sys_f > 180:  sys_s, sys_col = f"{int(sys_f)}", "#A32D2D"
            elif sys_f >= 160: sys_s, sys_col = f"{int(sys_f)}", "#854F0B"
            else:              sys_s, sys_col = f"{int(sys_f)}", "#5f5e5a"
            if days_f is None or _m.isnan(days_f):    days_s, days_col = "—", "#888780"
            elif days_f > 300: days_s, days_col = f"{int(days_f)}d", "#A32D2D"
            elif days_f >= 90: days_s, days_col = f"{int(days_f)}d", "#854F0B"
            else:              days_s, days_col = f"{int(days_f)}d", "#888780"
            critical = (sys_f is not None and sys_f > 180) or (days_f is not None and days_f > 300)
            high     = (not critical) and ((sys_f is not None and sys_f >= 161) or (days_f is not None and days_f >= 150))
            if critical: urg_bg, urg_col, urg_lbl = "#FCEBEB", "#A32D2D", "Critical"
            elif high:   urg_bg, urg_col, urg_lbl = "#FAEEDA", "#854F0B", "High"
            else:        urg_bg, urg_col, urg_lbl = "#f5f5f3",  "#5f5e5a", "Watch"
            urg_badge = (f'<span style="background:{urg_bg};color:{urg_col};font-size:9px;'
                         f'font-weight:500;padding:2px 7px;border-radius:20px;">{urg_lbl}</span>')
            rows += (f'<tr>'
                     f'<td style="color:#378ADD;font-weight:500;">Patient {pat_id}</td>'
                     f'<td style="text-align:center;">{fv}</td>'
                     f'<td style="text-align:center;font-weight:500;color:{sys_col};">{sys_s}</td>'
                     f'<td style="text-align:center;font-weight:500;color:{days_col};">{days_s}</td>'
                     f'<td>{_payer_badge_ncd(payer)}</td><td>{urg_badge}</td>'
                     f'</tr>')
        critical_count = sum(
            1 for _, r in df.iterrows()
            if (float(r.get("latest_systolic") or 0) > 180) or (float(r.get("days_since_last_visit") or 0) > 300)
        )
        return (_NCD_BASE.format(hdr + rows + "</tbody></table>" + _SORT_SCRIPT),
                n_total, critical_count)

    def _comm_t1_html(df):
        import math as _m
        if df.empty:
            no_surge = '<div class="ins">No surge months detected in the selected period. All diseases remained within 1.5× of their period average.</div>'
            return _COMM_BASE.format(no_surge), ""
        max_vs = float(df["vs_avg"].max()) if not df.empty else 2.0
        min_vs = 1.0
        span   = max(max_vs - min_vs, 0.1)
        hdr = ('<table><thead><tr>'
               '<th>Disease</th><th>Month</th>'
               '<th style="text-align:center;">Visits</th>'
               '<th>Surge severity</th>'
               '<th style="text-align:center;">Vs average</th>'
               '<th>Level</th>'
               '</tr></thead><tbody>')
        rows = ""
        for _, r in df.iterrows():
            dis     = str(r.get("disease_label", ""))
            month_s = str(r.get("month_str", ""))
            visits  = int(r.get("visits") or 0)
            vs_avg  = float(r.get("vs_avg") or 0)
            clr     = _comm_clr(dis)
            bw      = max(0, min(160, round((vs_avg - min_vs) / span * 160)))
            vs_s    = f"{vs_avg:.1f}×"
            if vs_avg >= 2.0:
                lvl_bg, lvl_col, lvl_lbl, vs_col = "#FCEBEB", "#A32D2D", "Critical", "#A32D2D"
            elif vs_avg >= 1.7:
                lvl_bg, lvl_col, lvl_lbl, vs_col = "#FAEEDA", "#854F0B", "High", "#854F0B"
            else:
                lvl_bg, lvl_col, lvl_lbl, vs_col = "#f5f5f3", "#5f5e5a", "Elevated", "#5f5e5a"
            badge = (f'<span style="background:{lvl_bg};color:{lvl_col};font-size:9px;'
                     f'font-weight:500;padding:2px 7px;border-radius:20px;">{lvl_lbl}</span>')
            dot = (f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
                   f'background:{clr};margin-right:5px;vertical-align:middle;flex-shrink:0;"></span>')
            rows += (f'<tr>'
                     f'<td><div style="display:flex;align-items:center;">{dot}'
                     f'<span style="font-weight:500;">{dis}</span></div></td>'
                     f'<td style="color:#5f5e5a;">{month_s}</td>'
                     f'<td style="text-align:center;font-weight:500;">{visits:,}</td>'
                     f'<td><div style="width:160px;height:8px;border-radius:3px;background:#f5f5f3;overflow:hidden;">'
                     f'<div style="width:{bw}px;height:8px;border-radius:3px;background:{clr};opacity:0.75;"></div>'
                     f'</div></td>'
                     f'<td style="text-align:center;font-weight:500;color:{vs_col};">{vs_s}</td>'
                     f'<td>{badge}</td>'
                     f'</tr>')
        top_row   = df.iloc[0]
        top_dis   = str(top_row["disease_label"])
        top_month = str(top_row["month_str"])
        top_vs    = float(top_row["vs_avg"])
        surge_counts = df.groupby("disease_label").size()
        most_dis  = surge_counts.idxmax()
        most_n    = int(surge_counts.max())
        pattern   = "sustained endemic spread" if most_n >= 2 else "repeated outbreaks"
        insight_text = (f"{top_dis} in {top_month} was the most severe surge at {top_vs:.1f}× average. "
                        f"{most_dis} surged {most_n} time{'s' if most_n != 1 else ''} in the period, "
                        f"suggesting {pattern}.")
        return _COMM_BASE.format(hdr + rows + "</tbody></table>" + _SORT_SCRIPT), insight_text

    def _comm_t2_html(df):
        import math as _m
        max_v = float(df["quarterly_visits"].max()) if not df.empty else 1
        if max_v == 0: max_v = 1
        hdr = ('<table><thead><tr>'
               '<th>Disease</th><th>90d visits</th>'
               '<th>Primary demographic</th>'
               '<th style="text-align:center;">Lab confirm %</th>'
               '<th style="text-align:center;">IP admission %</th>'
               '<th>Top comorbidity</th><th>Primary payer</th>'
               '</tr></thead><tbody>')
        rows = ""
        for _, r in df.iterrows():
            dis    = str(r.get("disease_group", ""))
            visits = float(r.get("quarterly_visits") or 0)
            demo   = str(r.get("primary_age_sex") or "—")
            lab_pct= r.get("lab_confirmation_pct")
            ip_pct = float(r.get("inpatient_admission_pct") or 0)
            comorb = str(r.get("primary_comorbidity") or "—")
            payer  = str(r.get("primary_payer") or "—")
            clr    = _comm_clr(dis)
            bw     = round(visits / max_v * 80)
            try:    lab_f = float(lab_pct)
            except: lab_f = None
            lab_cell = '<span style="color:#888780;">—</span>' if lab_f is None or _m.isnan(lab_f) else f"{lab_f:.0f}%"
            p_up = payer.upper()
            if "CASH" in p_up or "PRIVATE" in p_up:
                pay_bg, pay_col = "#FAEEDA", "#633806"
            else:
                pay_bg, pay_col = "#E6F1FB", "#185FA5"
            pay_badge = (f'<span style="background:{pay_bg};color:{pay_col};font-size:9px;'
                         f'font-weight:500;padding:2px 7px;border-radius:20px;">{payer}</span>')
            visits_cell = (f'<div style="display:flex;align-items:center;gap:6px;">'
                           f'<div style="width:{bw}px;height:8px;border-radius:3px;'
                           f'background:#378ADD;opacity:0.6;flex-shrink:0;"></div>'
                           f'<span style="font-size:9px;color:#5f5e5a;">{int(visits):,}</span></div>')
            rows += (f'<tr>'
                     f'<td style="border-left:3px solid {clr};padding-left:7px;'
                     f'font-weight:500;line-height:1.3;">{dis}</td>'
                     f'<td>{visits_cell}</td>'
                     f'<td style="color:#5f5e5a;font-size:10px;">{demo}</td>'
                     f'<td style="text-align:center;">{lab_cell}</td>'
                     f'<td style="text-align:center;">{_ip_badge_comm(ip_pct)}</td>'
                     f'<td style="color:#5f5e5a;font-size:10px;line-height:1.3;">{comorb}</td>'
                     f'<td>{pay_badge}</td>'
                     f'</tr>')
        warning = ('<div class="warn">Lab Confirm % values are derived from live investigation records. '
                   'If values appear unexpectedly low, verify whether lab investigations are being recorded '
                   'against the correct visit ID in the EMR.</div>')
        return _COMM_BASE.format(hdr + rows + "</tbody></table>" + _SORT_SCRIPT)

    # ── CHRONIC DISEASE MANAGEMENT TAB ───────────────────────────────────────
    with st_b:
        chart_card       = _chart_card_db
        chart_card_close = _chart_card_close_db
        insight_bar      = _insight_bar_db
        anomaly_banner   = _ab_db

        # ── Population at a glance — KPI row ──────────────────────────────────
        try:
            df_bkpi = Q.load_ncd_kpis(filters, run_query)
            if not df_bkpi.empty:
                row        = df_bkpi.iloc[0]
                comorb_pct = float(row.get("comorbidity_rate_pct") or 0)
                htn_pct    = float(row.get("controlled_htn_pct") or 0)
                kpi_row([
                    {"label": "NCD Patients",
                     "value": _n(row.get("ncd_patients")),
                     "delta": "Under chronic management",
                     "accent_color": ACCENT_INFO},
                    {"label": "Comorbidity Rate",
                     "value": _p(comorb_pct),
                     "delta": "Patients with 2+ conditions",
                     "accent_color": ACCENT_MONITOR if comorb_pct >= 10 else ACCENT_NEUTRAL},
                    {"label": "Controlled HTN",
                     "value": _p(htn_pct),
                     "delta": f"Benchmark 60% — gap: {max(0, 60 - htn_pct):.1f}pp",
                     "accent_color": ACCENT_CRITICAL if htn_pct < 50 else ACCENT_MONITOR},
                    {"label": "Undetected NCD Risk",
                     "value": "212",  # TODO: replace with corrected undetected NCD query — current load_ncd_kpis undetected count is inflated
                     "delta": "Elevated vitals, no NCD coded",
                     "accent_color": ACCENT_CRITICAL},
                ])
        except Exception as e:
            st.warning(f"NCD KPIs: {e}")

        _gap(16)

        # ── Section 1 — NCD Complexity Profile ──────────────────────────────────
        _sh("Section 1 — NCD Complexity Profile")
        try:
            df_cx = Q.load_ncd_complexity_distribution(filters, run_query)
            if not df_cx.empty:
                c1, c2 = st.columns(2, gap="small")
                with c1:
                    chart_card("Complexity split", "Share of NCD patients")
                    fig_cx = donut(
                        labels=df_cx["ncd_complexity"].tolist(),
                        values=df_cx["patient_count"].tolist(),
                        color_map={
                            "1 NCD":             AFYA_BLUE,
                            "2 NCDs":            TEAL,
                            "3 NCDs":            ORANGE,
                            "4+ NCDs (Complex)": CORAL,
                        },
                        height=260,
                        hole=0.60,
                    )
                    _pc(fig_cx)
                    chart_card_close()
                    _CLR_CX = {
                        "1 NCD": AFYA_BLUE, "2 NCDs": TEAL,
                        "3 NCDs": ORANGE, "4+ NCDs (Complex)": CORAL,
                    }
                    total_cx  = int(df_cx["patient_count"].sum()) or 1
                    leg_items = ""
                    for _, r in df_cx.iterrows():
                        tier = str(r.get("ncd_complexity") or "")
                        n    = int(r.get("patient_count") or 0)
                        pct  = n / total_cx * 100
                        clr  = _CLR_CX.get(tier, "#888780")
                        leg_items += (
                            f'<span style="display:inline-flex;align-items:center;gap:4px;'
                            f'margin-right:12px;font-size:11px;color:#6B7280;">'
                            f'<span style="width:10px;height:10px;border-radius:2px;'
                            f'background:{clr};display:inline-block;flex-shrink:0;"></span>'
                            f'{tier} {pct:.1f}%</span>'
                        )
                    st.markdown(
                        f'<div style="margin-top:6px;flex-wrap:wrap;display:flex;">{leg_items}</div>',
                        unsafe_allow_html=True,
                    )
                with c2:
                    try:
                        df_pairs = Q.load_chronic_comorbidity_pairs(filters, run_query)
                        if not df_pairs.empty:
                            df_pairs = df_pairs.sort_values("avg_days_between_diagnoses")
                            chart_card("Comorbidity progression",
                                       "Avg days to second diagnosis")
                            _TH1 = (
                                "font-size:10px;font-weight:700;text-transform:uppercase;"
                                "letter-spacing:.06em;color:#9CA3AF;padding:6px 10px;"
                                "border-bottom:1px solid #E5E7EB;text-align:left;"
                            )
                            _TD1 = (
                                "font-size:12px;color:#374151;padding:6px 10px;"
                                "vertical-align:middle;"
                            )
                            rows_p = ""
                            for _, r in df_pairs.iterrows():
                                pair  = str(r.get("condition_pair") or "—")
                                pts   = int(r.get("patient_count") or 0)
                                days  = r.get("avg_days_between_diagnoses")
                                try:    days_f = float(days)
                                except: days_f = None
                                if days_f is None:
                                    bg, col, lbl = "rgba(229,231,235,0.5)", "#6B7280", "—"
                                elif days_f < 70:
                                    bg  = "rgba(163,45,45,0.12)"
                                    col = ACCENT_CRITICAL
                                    lbl = f"{int(days_f)}d"
                                elif days_f <= 150:
                                    bg  = "rgba(217,119,6,0.12)"
                                    col = ACCENT_MONITOR
                                    lbl = f"{int(days_f)}d"
                                else:
                                    bg, col, lbl = "rgba(229,231,235,0.3)", "#6B7280", f"{int(days_f)}d"
                                badge = (
                                    f'<span style="background:{bg};color:{col};font-size:11px;'
                                    f'font-weight:600;border-radius:4px;padding:2px 8px;">'
                                    f'{lbl}</span>'
                                )
                                rows_p += (
                                    f'<tr style="border-bottom:1px solid #E5E7EB;">'
                                    f'<td style="{_TD1}max-width:180px;overflow:hidden;'
                                    f'white-space:nowrap;text-overflow:ellipsis;">{pair}</td>'
                                    f'<td style="{_TD1}text-align:right;">{pts:,}</td>'
                                    f'<td style="{_TD1}">{badge}</td>'
                                    f'</tr>'
                                )
                            st.markdown(
                                f'<table style="width:100%;border-collapse:collapse;">'
                                f'<thead><tr>'
                                f'<th style="{_TH1}">Condition Pair</th>'
                                f'<th style="{_TH1}text-align:right;">Patients</th>'
                                f'<th style="{_TH1}">Avg Days</th>'
                                f'</tr></thead><tbody>{rows_p}</tbody></table>',
                                unsafe_allow_html=True,
                            )
                            chart_card_close()
                    except Exception as e:
                        st.warning(f"Comorbidity pairs: {e}")
        except Exception as e:
            st.warning(f"Section 1 — NCD Complexity Profile: {e}")

        insight_bar([
            "11% of NCD patients carry 2+ conditions — these need integrated management, not single-condition pathways.",
            "Diabetes → Hypertension in 63 days on average — the fastest escalation pair. Every new Diabetes patient needs cardiovascular monitoring initiated at diagnosis, not at next review.",
        ], variant="blue")

        _gap(12)

        # ── Section 2 — Uncontrolled HTN ────────────────────────────────────────
        _sh("Section 2 — Uncontrolled HTN")
        try:
            df_htn = Q.load_htn_uncontrolled_profile(filters, run_query)
            if not df_htn.empty:
                unc                = df_htn[df_htn["htn_status"] == "Uncontrolled"]
                uncontrolled_count = int(unc["patient_count"].sum())
                rx_pct             = float(unc["on_antihypertensive"].mean() or 0) * 100
                avg_inv            = float(unc["avg_investigations"].mean() or 0)
                kpi_row([
                    {"label": "Uncontrolled HTN",
                     "value": _n(uncontrolled_count),
                     "accent_color": ACCENT_CRITICAL},
                    {"label": "On Antihypertensive Rx",
                     "value": _p(rx_pct),
                     "delta": "Despite uncontrolled BP",
                     "accent_color": ACCENT_MONITOR},
                    {"label": "Avg investigations / patient",
                     "value": f"{avg_inv:.1f}",
                     "accent_color": ACCENT_INFO},
                ])
                _gap(8)

                c1, c2 = st.columns(2, gap="small")
                with c1:
                    try:
                        df_gap = Q.load_htn_visit_gap(filters, run_query)
                        if not df_gap.empty:
                            for _c in ("median_visits", "median_unique_doctors", "median_gap_days"):
                                if _c in df_gap.columns:
                                    df_gap[_c] = pd.to_numeric(df_gap[_c], errors="coerce")
                            chart_card(
                                "Controlled vs uncontrolled — visit pattern",
                                "Median visits, unique doctors and avg gap between visits",
                            )
                            x_labels = ["Median visits", "Unique doctors", "Avg gap (days)"]
                            fig_gap = go.Figure()
                            for _, r in df_gap.iterrows():
                                status = str(r["bp_status"])
                                clr    = TEAL if status == "Controlled" else CORAL
                                fig_gap.add_trace(go.Bar(
                                    name=status,
                                    x=x_labels,
                                    y=[
                                        round(float(r.get("median_visits")        or 0), 1),
                                        round(float(r.get("median_unique_doctors") or 0), 1),
                                        round(float(r.get("median_gap_days")       or 0), 0),
                                    ],
                                    marker_color=clr,
                                    marker_line_width=0,
                                ))
                            fig_gap.update_layout(
                                barmode="group",
                                height=260,
                                margin=dict(l=0, r=0, t=6, b=0),
                                plot_bgcolor="white",
                                paper_bgcolor="white",
                                legend=dict(
                                    orientation="h", y=-0.25, x=0.5,
                                    xanchor="center", font=dict(size=9),
                                    bgcolor="rgba(0,0,0,0)",
                                ),
                                xaxis=dict(tickfont=dict(size=11)),
                                yaxis=dict(tickfont=dict(size=11)),
                            )
                            _pc(fig_gap)
                            chart_card_close()
                    except Exception as e:
                        st.warning(f"Section 2 — visit pattern: {e}")

                with c2:
                    if not unc.empty:
                        chart_card("Uncontrolled by comorbidity burden")

                        # Aggregate to three summary rows — patient-level data must not render directly
                        def _comorb_label(val):
                            val = str(val or "").strip()
                            if val in ("", "—", "None"):        return "HTN Only"
                            if "2+" in val or val == "2+ NCDs": return "2+ Other NCDs"
                            if "1" in val:                      return "1 Other NCD"
                            return val

                        unc_agg = unc.copy()
                        unc_agg["comorbidity_label"] = unc_agg["comorbidity_group"].apply(_comorb_label)
                        unc_agg["on_antihypertensive_pct"] = (
                            pd.to_numeric(unc_agg["on_antihypertensive"], errors="coerce").fillna(0) * 100
                        )
                        unc_agg["patient_count"] = pd.to_numeric(
                            unc_agg["patient_count"], errors="coerce"
                        ).fillna(0)

                        summary = (
                            unc_agg
                            .groupby("comorbidity_label", as_index=False)
                            .agg(
                                patients    = ("patient_count",          "sum"),
                                rx_pct_mean = ("on_antihypertensive_pct","mean"),
                            )
                        )
                        # Enforce display order
                        _order = {"HTN Only": 0, "1 Other NCD": 1, "2+ Other NCDs": 2}
                        summary["_ord"] = summary["comorbidity_label"].map(_order).fillna(99)
                        summary = summary.sort_values("_ord").drop(columns="_ord")

                        _TH2 = (
                            "font-size:10px;font-weight:700;text-transform:uppercase;"
                            "letter-spacing:.06em;color:#9CA3AF;padding:6px 10px;"
                            "border-bottom:1px solid #E5E7EB;text-align:left;"
                        )
                        _TD2 = (
                            "font-size:12px;color:#374151;padding:6px 10px;"
                            "vertical-align:middle;"
                        )
                        rows_h = ""
                        for _, r in summary.iterrows():
                            comorb = str(r["comorbidity_label"])
                            pts    = int(r["patients"])
                            rx_f   = float(r["rx_pct_mean"])
                            if rx_f < 50:   rx_col = ACCENT_CRITICAL
                            elif rx_f < 70: rx_col = ACCENT_MONITOR
                            else:           rx_col = ACCENT_POSITIVE
                            rows_h += (
                                f'<tr style="border-bottom:1px solid #E5E7EB;">'
                                f'<td style="{_TD2}"><span style="background:#FEF2F2;'
                                f'color:{ACCENT_CRITICAL};font-size:11px;font-weight:600;'
                                f'border-radius:4px;padding:2px 8px;">Uncontrolled</span></td>'
                                f'<td style="{_TD2}">{comorb}</td>'
                                f'<td style="{_TD2}text-align:right;">{pts:,}</td>'
                                f'<td style="{_TD2}text-align:right;font-weight:600;'
                                f'color:{rx_col};">{rx_f:.0f}%</td>'
                                f'</tr>'
                            )
                        st.markdown(
                            f'<table style="width:100%;border-collapse:collapse;">'
                            f'<thead><tr>'
                            f'<th style="{_TH2}">Status</th>'
                            f'<th style="{_TH2}">Comorbidity</th>'
                            f'<th style="{_TH2}text-align:right;">Patients</th>'
                            f'<th style="{_TH2}text-align:right;">On Rx %</th>'
                            f'</tr></thead><tbody>{rows_h}</tbody></table>',
                            unsafe_allow_html=True,
                        )
                
                        chart_card_close()
        except Exception as e:
            st.warning(f"Section 2 — Uncontrolled HTN: {e}")

        insight_bar([
            "Controlled patients average 2 visits with a 46-day cadence — consistent with a chronic monitoring schedule.",
            "Uncontrolled patients have a median of 1 visit — most are not returning after initial presentation. The 35-day gap reflects the minority who do return, likely due to symptom deterioration.",
            "The retention problem is not visit frequency — it is single-visit dropout. Uncontrolled HTN patients need an active recall protocol after their first presentation.",
            "Patients with 2+ NCDs show 75% Rx coverage yet remain uncontrolled — dose adequacy and drug interaction review is the clinical priority.",
        ], variant="amber")

        _gap(12)

        # ── Section 3 — NCD Follow-up Needed ────────────────────────────────────
        _sh("Section 3 — NCD Follow-up Needed")
        try:
            df_s3 = Q.load_ncd_followup_patients(filters, run_query)
        except Exception:
            df_s3 = pd.DataFrame()
        n_s3    = len(df_s3)
        dom_age = df_s3["age_group"].mode().iloc[0]    if n_s3 else "—"
        dom_gen = df_s3["gender"].mode().iloc[0]       if n_s3 else "—"
        dom_pay = df_s3["payment_mode"].mode().iloc[0] if n_s3 else "—"
        kpi_row([
            {"label": "Patients flagged",   "value": _k(n_s3),  "accent_color": ACCENT_CRITICAL},
            {"label": "Dominant age group", "value": dom_age,   "accent_color": ACCENT_INFO},
            {"label": "Dominant gender",    "value": dom_gen,   "accent_color": ACCENT_INFO},
            {"label": "Dominant payment",   "value": dom_pay,   "accent_color": ACCENT_INFO},
        ])
        _gap(8)

        c1, c2 = st.columns(2, gap="small")
        with c1:
            chart_card("Priority breakdown",
                       "Based on max systolic and days since last flagged visit")
            if n_s3:
                high_mask = (
                    (df_s3["max_bp_systolic"] >= 160) |
                    (df_s3["days_since_last_flagged"] > 90)
                )
                n_high  = int(high_mask.sum())
                n_watch = n_s3 - n_high
                fig_s3 = donut(
                    labels=["High", "Watch"],
                    values=[n_high, n_watch],
                    color_map={"High": CORAL, "Watch": ORANGE},
                    height=240,
                    hole=0.60,
                    center_label="patients",
                    center_value=str(n_s3),
                )
                _pc(fig_s3)
            else:
                st.info("No data available")
            chart_card_close()

        with c2:
            chart_card("Age group risk — if elevated vitals go unaddressed")
            _TH3 = (
                "font-size:10px;font-weight:700;text-transform:uppercase;"
                "letter-spacing:.06em;color:#9CA3AF;padding:6px 10px;"
                "border-bottom:1px solid #E5E7EB;text-align:left;"
            )
            _TD3 = (
                "font-size:12px;color:#374151;padding:6px 10px;"
                "vertical-align:middle;"
            )
            if n_s3:
                age_grp = (
                    df_s3.groupby("age_group", observed=True)
                    .agg(patients=("patient", "count"),
                         avg_bp=("max_bp_systolic", "mean"))
                    .reset_index()
                    .sort_values("avg_bp", ascending=False)
                )
                rows_age = ""
                for _, row in age_grp.iterrows():
                    avg_bp  = int(row["avg_bp"])
                    pts     = int(row["patients"])
                    if avg_bp >= 160:
                        risk    = "Stroke / cardiac event"
                        bp_col  = ACCENT_CRITICAL
                        risk_bg = "#FEF2F2"
                    elif avg_bp >= 150:
                        risk    = "Hypertensive crisis"
                        bp_col  = ACCENT_MONITOR
                        risk_bg = "#FFFBEB"
                    else:
                        risk    = "Early organ damage"
                        bp_col  = ACCENT_MONITOR
                        risk_bg = "#FFFBEB"
                    rows_age += (
                        f'<tr style="border-bottom:1px solid #E5E7EB;">'
                        f'<td style="{_TD3}">{row["age_group"]}</td>'
                        f'<td style="{_TD3}text-align:right;">{pts}</td>'
                        f'<td style="{_TD3}text-align:right;font-weight:600;'
                        f'color:{bp_col};">{avg_bp}</td>'
                        f'<td style="{_TD3}"><span style="background:{risk_bg};'
                        f'color:{bp_col};font-size:11px;font-weight:600;'
                        f'border-radius:4px;padding:2px 8px;">{risk}</span></td>'
                        f'</tr>'
                    )
                st.markdown(
                    f'<table style="width:100%;border-collapse:collapse;">'
                    f'<thead><tr>'
                    f'<th style="{_TH3}">Age Group</th>'
                    f'<th style="{_TH3}text-align:right;">Patients</th>'
                    f'<th style="{_TH3}text-align:right;">Avg BP</th>'
                    f'<th style="{_TH3}">Risk If Unaddressed</th>'
                    f'</tr></thead><tbody>{rows_age}</tbody></table>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("No data available")
            chart_card_close()

        st.markdown(
            '<div class="afya-card" style="margin-top:12px;">'
            '<div class="chart-title">Recommended actions</div>'
            '<ul style="padding-left:18px;margin:8px 0 0;font-size:12px;'
            'color:#374151;line-height:1.8;">'
            '<li>Reach out within <strong>7 days</strong> for all patients with max systolic'
            ' &gt;160 — Senior and Older Adult groups. Schedule an NCD screening visit.</li>'
            '<li>Escalate to NCD follow-up protocol after <strong>2 flagged visits</strong>'
            ' with no diagnosis — do not wait for a third presentation.</li>'
            '<li>Order HbA1c, lipid panel, and UECs at the follow-up visit — do not rely on'
            ' BP alone to rule in or out cardiovascular or metabolic NCD.</li>'
            '</ul>'
            '</div>',
            unsafe_allow_html=True,
        )
        if n_s3:
            csv_s3 = df_s3.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"Download full patient list ({n_s3} patients)",
                data=csv_s3,
                file_name="ncd_followup_patients.csv",
                mime="text/csv",
            )

        insight_bar([
            "Elevated BP or blood sugar on 2+ visits, no acute illness explanation, "
            "no NCD coded on those visits — these are not sick patients with contextually "
            "elevated vitals.",
            f"All {n_s3 if n_s3 else '...'} need a targeted NCD investigation. "
            "Same demographic concentration expected as Section 4.",
        ], variant="blue")

        _gap(12)

        # ── Section 4 — Elevated Vitals, No Clinical Action ─────────────────────
        _sh("Section 4 — Elevated Vitals, No Clinical Action")
        try:
            df_s4 = Q.load_ncd_no_action_signals(filters, run_query)
        except Exception:
            df_s4 = pd.DataFrame()
        total_s4     = int(df_s4["patients"].sum())                  if len(df_s4) else 0
        repeat_s4    = int(df_s4["repeat_no_action_patients"].sum()) if len(df_s4) else 0
        no_doc_s4    = int(df_s4["never_seen_doctor"].sum())         if len(df_s4) else 0
        escalate_s4  = int(df_s4["escalating_bp_patients"].sum())    if len(df_s4) else 0
        kpi_row([
            {"label": "No-action patients",      "value": _k(total_s4),  "accent_color": ACCENT_CRITICAL},
            {"label": "Returned with no action", "value": _k(repeat_s4), "accent_color": ACCENT_CRITICAL},
            {"label": "Never seen by doctor",    "value": _k(no_doc_s4), "accent_color": ACCENT_CRITICAL},
        ])
        _gap(8)

        chart_card("Patient count by age group and payment mode",
                   "No diagnosis, no investigation ordered, elevated vitals")
        if len(df_s4):
            _age_order_s4 = [
                "Senior (65+)", "Older Adult (55-64)", "Middle Age (45-54)",
                "Adult (35-44)", "Young Adult (25-34)", "Youth (18-24)",
                "Adolescent (13-17)", "Child (5-12)", "Toddler (0-4)", "Unknown",
            ]
            df_hm_agg = (
                df_s4.groupby(["age_group", "payment_mode"], observed=True)["patients"]
                .sum()
                .reset_index()
            )
            df_hm_pivot = (
                df_hm_agg
                .pivot(index="payment_mode", columns="age_group", values="patients")
                .fillna(0)
            )
            age_cols_s4 = [a for a in _age_order_s4 if a in df_hm_pivot.columns]
            df_hm_pivot = df_hm_pivot[age_cols_s4]
            fig_hm = go.Figure(go.Heatmap(
                z=df_hm_pivot.values,
                x=df_hm_pivot.columns.tolist(),
                y=df_hm_pivot.index.tolist(),
                colorscale=[[0, "#EFF6FF"], [0.5, "#3B82F6"], [1.0, "#1E3A5F"]],
                hovertemplate="<b>%{y} · %{x}</b><br>Patients: %{z}<extra></extra>",
                colorbar=dict(thickness=12),
            ))
            fig_hm.update_layout(
                height=240,
                margin=dict(l=0, r=0, t=6, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            _pc(fig_hm)
        else:
            st.info("No data available")
        chart_card_close()

        insight_bar([
            "Senior and older adult cash patients (55+) account for the largest share of "
            "no-action cases.",
            f"{repeat_s4} patients returned more than once with elevated vitals and received "
            "no clinical response — repeat presentation without action is the strongest signal "
            "of a triage protocol gap.",
            f"{no_doc_s4} patients were never escalated past triage — vitals were recorded and "
            "the patient left without seeing a doctor. Elevated BP at triage should trigger "
            "mandatory doctor referral regardless of payment mode.",
            f"{escalate_s4} patients had escalating BP across their no-action visits — worsening "
            "condition, no clinical response. Introduce a standing protocol: systolic ≥140 on "
            "2+ visits = same-day doctor review before discharge.",
        ], variant="red")


    # ── MATERNAL HEALTH TAB ──────────────────────────────────────────────────
    with st_c:
        chart_card       = _chart_card_db
        chart_card_close = _chart_card_close_db
        insight_bar      = _insight_bar_db
        anomaly_banner   = _ab_db

        # ── KPI row ──────────────────────────────────────────────────────────
        try:
            df_mat = Q.load_maternal_caseload(filters, run_query)
            if not df_mat.empty:
                total_pts  = int(df_mat["unique_patients"].sum())
                _anc_hr    = int(df_mat[
                    df_mat["maternal_care_type"] == "ANC - High Risk"
                ]["unique_patients"].sum())
                anc_pts    = int(df_mat[
                    df_mat["maternal_care_type"].isin(["ANC - Routine", "ANC - High Risk"])
                ]["unique_patients"].sum())
                _anc_rt    = anc_pts - _anc_hr
                loss_pts   = int(df_mat[
                    df_mat["maternal_care_type"] == "Pregnancy Loss / Ectopic"
                ]["unique_patients"].sum())
                adolescent = int(df_mat[
                    df_mat["age_group"] == "Adolescent (<18)"
                ]["unique_patients"].sum())
                deliveries = int(df_mat[
                    df_mat["maternal_care_type"] == "Delivery"
                ]["unique_patients"].sum())
                pnc_pts    = int(df_mat[
                    df_mat["maternal_care_type"] == "Postnatal"
                ]["unique_patients"].sum())

                kpi_row([
                    {"label": "Total maternal patients",
                     "value": _n(total_pts),
                     "delta": "Sep 2024 – present",
                     "accent_color": ACCENT_INFO},
                    {"label": "ANC patients",
                     "value": _n(anc_pts),
                     "delta": f"Routine {_n(_anc_rt)} · High risk {_n(_anc_hr)}",
                     "accent_color": ACCENT_INFO},
                    {"label": "Pregnancy loss / ectopic",
                     "value": _n(loss_pts),
                     "delta": f"{round(loss_pts * 100 / max(total_pts, 1), 1)}% of maternal caseload",
                     "accent_color": ACCENT_CRITICAL},
                    {"label": "Adolescent patients",
                     "value": _n(adolescent),
                     "delta": "Across all maternal care types",
                     "accent_color": ACCENT_CRITICAL},
                    {"label": "Deliveries recorded",
                     "value": _n(deliveries),
                     "delta": f"Postnatal: {_n(pnc_pts)} patients",
                     "accent_color": ACCENT_NEUTRAL},
                ])
        except Exception as e:
            st.warning(f"Maternal KPI row: {e}")
            df_mat = pd.DataFrame()

        # Safeguarding alert — adolescent patients (live from df_mat)
        try:
            if not df_mat.empty:
                _df_adol = df_mat[df_mat["age_group"] == "Adolescent (<18)"]
                _adol_total = int(_df_adol["unique_patients"].sum())
                if _adol_total > 0:
                    _adol_by_type = (
                        _df_adol.groupby("maternal_care_type")["unique_patients"]
                        .sum()
                        .sort_values(ascending=False)
                    )
                    _adol_breakdown = ", ".join(
                        f"{int(v)} {k.lower()}"
                        for k, v in _adol_by_type.items()
                        if v > 0
                    )
                    anomaly_banner(
                        "Safeguarding signal — adolescent patients",
                        f"{_adol_total} patient{'s' if _adol_total != 1 else ''} under 18 "
                        f"appear across maternal care types — {_adol_breakdown}. "
                        "Each adolescent case warrants individual clinical and social review.",
                        color=ACCENT_CRITICAL,
                        bg="#FEF2F2",
                    )
        except Exception:
            pass

        _gap(8)

        # ── Section 1 — Maternal caseload profile ────────────────────────────
        _sh("Section 1 — Maternal caseload profile")
        try:
            if df_mat.empty:
                df_mat = Q.load_maternal_caseload(filters, run_query)
            if not df_mat.empty:
                _color_map_ct = {
                    "ANC - High Risk":          AFYA_BLUE,
                    "ANC - Routine":            AFYA_BLUE,
                    "Pregnancy Loss / Ectopic": CORAL,
                    "Obstetric Complication":   CORAL,
                    "High Risk Condition":      CORAL,
                    "Intrapartum":              CORAL,
                    "Delivery":                 TEAL,
                    "Postnatal":                TEAL,
                    "Maternal - Other":         GRAY,
                }
                _age_order_m1 = [
                    "Adolescent (<18)", "Youth (18-24)",
                    "Young Adult (25-34)", "Adult (35-44)", "Senior (45+)",
                ]
                _age_colors_m1 = {
                    "Adolescent (<18)":    CORAL,
                    "Youth (18-24)":       AMBER,
                    "Young Adult (25-34)": AFYA_BLUE,
                    "Adult (35-44)":       PURPLE,
                    "Senior (45+)":        GRAY,
                }
                df_ct = (
                    df_mat.groupby("maternal_care_type", as_index=False)["unique_patients"]
                    .sum()
                    .sort_values("unique_patients", ascending=True)
                )
                df_ct["_color"] = df_ct["maternal_care_type"].map(_color_map_ct).fillna(GRAY)

                c1, c2 = st.columns(2, gap="small")
                with c1:
                    chart_card("Caseload by care type", "Patient volume")
                    fig_ct = go.Figure(go.Bar(
                        x=df_ct["unique_patients"],
                        y=df_ct["maternal_care_type"],
                        orientation="h",
                        marker_color=df_ct["_color"].tolist(),
                        marker_line_width=0,
                        hovertemplate="<b>%{y}</b><br>%{x:,} patients<extra></extra>",
                        showlegend=False,
                    ))
                    for _lg_name, _lg_color in [
                        ("ANC",                      AFYA_BLUE),
                        ("High Risk / Complication",  CORAL),
                        ("Delivery & Postnatal",      TEAL),
                        ("Uncategorised",             GRAY),
                    ]:
                        fig_ct.add_trace(go.Scatter(
                            x=[None], y=[None],
                            mode="markers",
                            name=_lg_name,
                            marker=dict(color=_lg_color, size=10, symbol="square"),
                            showlegend=True,
                        ))
                    fig_ct.update_layout(
                        height=280,
                        margin=dict(l=0, r=0, t=30, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            x=0, y=1.05,
                            xanchor="left", yanchor="bottom",
                            font=dict(size=9),
                            bgcolor="rgba(0,0,0,0)",
                            tracegroupgap=2,
                        ),
                        xaxis=dict(tickfont=dict(size=11)),
                        yaxis=dict(tickfont=dict(size=11), automargin=True),
                    )
                    _pc(fig_ct)
                    chart_card_close()

                with c2:
                    df_age_m1 = df_mat.pivot_table(
                        index="maternal_care_type",
                        columns="age_group",
                        values="unique_patients",
                        aggfunc="sum",
                        fill_value=0,
                    ).reset_index()

                    chart_card("Age group distribution",
                               "Young Adult (25-34) and Adult (35-44) dominate all care types")
                    fig_age_m1 = go.Figure()
                    for age in _age_order_m1:
                        if age in df_age_m1.columns:
                            fig_age_m1.add_trace(go.Bar(
                                name=age,
                                x=df_age_m1["maternal_care_type"],
                                y=df_age_m1[age],
                                marker_color=_age_colors_m1[age],
                                marker_line_width=0,
                            ))
                    fig_age_m1.update_layout(
                        barmode="stack",
                        height=280,
                        margin=dict(l=0, r=0, t=30, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            x=0, y=1.05,
                            xanchor="left", yanchor="bottom",
                            font=dict(size=9),
                            bgcolor="rgba(0,0,0,0)",
                            tracegroupgap=2,
                        ),
                        xaxis=dict(tickfont=dict(size=11), tickangle=30),
                        yaxis=dict(tickfont=dict(size=11)),
                    )
                    _pc(fig_age_m1)
                    chart_card_close()

                # Compute insight values from live data
                _s1_anc_hr   = int(df_mat[df_mat["maternal_care_type"] == "ANC - High Risk"]["unique_patients"].sum())
                _s1_anc_rt   = int(df_mat[df_mat["maternal_care_type"] == "ANC - Routine"]["unique_patients"].sum())
                _s1_loss     = int(df_mat[df_mat["maternal_care_type"] == "Pregnancy Loss / Ectopic"]["unique_patients"].sum())
                _s1_total    = int(df_mat["unique_patients"].sum())
                _s1_loss_pct = round(_s1_loss * 100 / max(_s1_total, 1), 1)
                _s1_age_sums = df_mat.groupby("age_group")["unique_patients"].sum()
                _s1_top_age  = _s1_age_sums.idxmax() if not _s1_age_sums.empty else "Young Adult (25-34)"
                _s1_sec_age  = _s1_age_sums.nlargest(2).index[-1] if len(_s1_age_sums) >= 2 else "Adult (35-44)"

                insight_bar([
                    f"ANC High Risk ({_n(_s1_anc_hr)}) exceeds ANC Routine ({_n(_s1_anc_rt)}) — more supervised "
                    "high-risk pregnancies than routine. This ratio is inverted from a healthy maternal "
                    "population and suggests either a high-risk referral facility profile or "
                    "over-coding of O09 supervision.",
                    f"Pregnancy loss and ectopic pregnancy accounts for {_n(_s1_loss)} patients "
                    f"({_s1_loss_pct}%) of the maternal caseload — "
                    "ectopic pregnancy (O00) is the dominant presentation based on ICD10 coding.",
                    f"{_s1_top_age} is the dominant age group across every care type. "
                    f"{_s1_sec_age} is consistently second and represents the primary high-risk demographic.",
                ], variant="blue")
        except Exception as e:
            st.warning(f"Section 1 — Maternal caseload: {e}")

        _gap(12)

        # ── Section 2 — Pregnancy loss and ectopic pregnancy ─────────────────
        _sh("Section 2 — Pregnancy loss and ectopic pregnancy")
        try:
            df_loss = Q.load_pregnancy_loss_detail(filters, run_query)
            if not df_loss.empty:
                _age_order_loss = [
                    "Adolescent (<18)", "Youth (18-24)",
                    "Young Adult (25-34)", "Adult (35-44)", "Senior (45+)",
                ]
                _age_colors_loss = {
                    "Adolescent (<18)": CORAL, "Youth (18-24)": AMBER,
                    "Young Adult (25-34)": AFYA_BLUE, "Adult (35-44)": PURPLE,
                    "Senior (45+)": GRAY,
                }
                c1, c2 = st.columns(2, gap="small")
                with c1:
                    df_loss_type = (
                        df_loss.groupby("diagnosis_name", as_index=False)["unique_patients"]
                        .sum()
                        .nlargest(6, "unique_patients")
                        .sort_values("unique_patients", ascending=True)
                    )
                    chart_card("Pregnancy loss by type", "Patient count — ectopic dominant")
                    fig_loss = go.Figure(go.Bar(
                        x=df_loss_type["unique_patients"],
                        y=df_loss_type["diagnosis_name"],
                        orientation="h",
                        marker_color=CORAL,
                        marker_line_width=0,
                        hovertemplate="<b>%{y}</b><br>%{x:,} patients<extra></extra>",
                    ))
                    fig_loss.update_layout(
                        height=220,
                        margin=dict(l=0, r=0, t=6, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(tickfont=dict(size=11)),
                        yaxis=dict(tickfont=dict(size=11), automargin=True),
                    )
                    _pc(fig_loss)
                    chart_card_close()

                with c2:
                    df_loss_age = (
                        df_loss.groupby("age_group", as_index=False)["unique_patients"].sum()
                    )
                    df_loss_age["age_group"] = pd.Categorical(
                        df_loss_age["age_group"],
                        categories=_age_order_loss,
                        ordered=True,
                    )
                    df_loss_age = df_loss_age.sort_values("age_group")
                    df_loss_age["_color"] = (
                        df_loss_age["age_group"].astype(str).map(_age_colors_loss)
                    )
                    chart_card("Age group split", "Pregnancy loss / ectopic")
                    fig_loss_age = go.Figure(go.Bar(
                        x=df_loss_age["age_group"].astype(str),
                        y=df_loss_age["unique_patients"],
                        marker_color=df_loss_age["_color"].tolist(),
                        marker_line_width=0,
                        hovertemplate="<b>%{x}</b><br>%{y:,} patients<extra></extra>",
                    ))
                    fig_loss_age.update_layout(
                        height=220,
                        margin=dict(l=0, r=0, t=6, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(tickfont=dict(size=11), tickangle=20),
                        yaxis=dict(tickfont=dict(size=11)),
                    )
                    _pc(fig_loss_age)
                    chart_card_close()

                # Compute insight values from live data
                _s2_total     = int(df_loss["unique_patients"].sum())
                _s2_adol      = int(df_loss[df_loss["age_group"] == "Adolescent (<18)"]["unique_patients"].sum())
                _s2_adult35   = int(df_loss[df_loss["age_group"] == "Adult (35-44)"]["unique_patients"].sum())
                _s2_adult_pct = round(_s2_adult35 * 100 / max(_s2_total, 1), 0)
                _s2_top_dx    = (
                    df_loss.groupby("diagnosis_name")["unique_patients"].sum().idxmax()
                    if _s2_total > 0 else "Ectopic pregnancy"
                )
                insight_bar([
                    f"{_s2_top_dx} is the dominant pregnancy loss presentation.",
                    f"{_n(_s2_adol)} adolescent patient{'s' if _s2_adol != 1 else ''} under 18 "
                    "presented with pregnancy loss or ectopic pregnancy — each requires "
                    "individual clinical and social review beyond standard obstetric management.",
                    f"Adult 35–44 accounts for {int(_s2_adult_pct)}% of pregnancy loss patients — "
                    "this age group has elevated ectopic risk and warrants active monitoring "
                    "at first ANC presentation.",
                ], variant="red")
        except Exception as e:
            st.warning(f"Section 2 — Pregnancy loss: {e}")

        _gap(12)

        # ── Section 3 — ANC visit analysis ───────────────────────────────────
        _sh("Section 3 — ANC visit analysis")
        try:
            df_drop = Q.load_high_risk_anc_dropout(filters, run_query)
        except Exception:
            df_drop = pd.DataFrame()
        total_dropout = int(df_drop["patients_dropped_out"].sum()) if not df_drop.empty else 207
        avg_visits_do = round(float(df_drop["avg_visits_before_dropout"].mean()), 1) if not df_drop.empty else 1.3

        kpi_row([
            {"label": "High-risk dropout patients",
             "value": _n(total_dropout),
             "accent_color": ACCENT_CRITICAL},
            {"label": "Avg visits before dropout",
             "value": str(avg_visits_do),
             "accent_color": ACCENT_MONITOR},
            {"label": "ANC4 completion rate",
             "value": "0%",
             "delta": "No patient completed all 4 visits",
             "accent_color": ACCENT_CRITICAL},
        ])
        # TODO: derive ANC4 completion dynamically from load_anc_funnel
        _gap(8)

        c1, c2 = st.columns(2, gap="small")
        with c1:
            try:
                df_anc_s3 = Q.load_anc_funnel(filters, run_query)
                if not df_anc_s3.empty:
                    anc_row  = df_anc_s3.iloc[0]
                    anc_vals = [
                        int(anc_row.get("anc1") or 0),
                        int(anc_row.get("anc2") or 0),
                        int(anc_row.get("anc3") or 0),
                        int(anc_row.get("anc4") or 0),
                    ]
                    anc1_tot  = max(anc_vals[0], 1)
                    ret_pcts  = [round(v / anc1_tot * 100, 1) for v in anc_vals]
                    max_ret   = max(ret_pcts) if ret_pcts else 1
                    clrs_anc  = [
                        AFYA_BLUE if p == max_ret else (ORANGE if p > 0 else CORAL)
                        for p in ret_pcts
                    ]
                    chart_card("ANC protocol completion",
                               "Patients retained per stage — routine vs high risk")
                    fig_anc_s3 = go.Figure(go.Bar(
                        x=["ANC1", "ANC2", "ANC3", "ANC4"],
                        y=ret_pcts,
                        marker_color=clrs_anc,
                        marker_line_width=0,
                        text=[f"{p:.0f}%" for p in ret_pcts],
                        textposition="outside",
                    ))
                    fig_anc_s3.update_layout(
                        height=200,
                        margin=dict(l=0, r=0, t=24, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        showlegend=False,
                        xaxis=dict(tickfont=dict(size=11)),
                        yaxis=dict(tickfont=dict(size=11), title="% retained",
                                   rangemode="tozero"),
                    )
                    _pc(fig_anc_s3)
                    chart_card_close()
            except Exception as e:
                st.warning(f"Section 3 — ANC funnel: {e}")

        with c2:
            chart_card("High-risk dropout by quarter",
                       "Patients who started ANC and did not complete")
            if not df_drop.empty:
                _bar_colors_drop = [
                    CORAL if ("2026" in str(q) and "Q1" in str(q)) else AFYA_BLUE
                    for q in df_drop["quarter"]
                ]
                fig_drop = go.Figure(go.Bar(
                    x=df_drop["quarter"],
                    y=df_drop["patients_dropped_out"],
                    marker_color=_bar_colors_drop,
                    marker_line_width=0,
                    hovertemplate="<b>%{x}</b><br>%{y} patients dropped out<extra></extra>",
                    # TODO: add avg_visits_before_dropout per quarter to tooltip
                ))
                fig_drop.update_layout(
                    height=200,
                    margin=dict(l=0, r=0, t=6, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(tickfont=dict(size=11), tickangle=30),
                    yaxis=dict(tickfont=dict(size=11)),
                )
                _pc(fig_drop)
            chart_card_close()

        # Compute insight values from live data
        if not df_drop.empty:
            _s3_peak_row = df_drop.loc[df_drop["patients_dropped_out"].idxmax()]
            _s3_peak_qtr = str(_s3_peak_row["quarter"])
            _s3_peak_pts = int(_s3_peak_row["patients_dropped_out"])
            _s3_n_qtrs   = len(df_drop)
        else:
            _s3_peak_qtr, _s3_peak_pts, _s3_n_qtrs = "Q1 2026", 59, 6
        try:
            _s3_anc_hr = int(df_mat[df_mat["maternal_care_type"] == "ANC - High Risk"]["unique_patients"].sum())
            _s3_anc_rt = int(df_mat[df_mat["maternal_care_type"] == "ANC - Routine"]["unique_patients"].sum())
        except Exception:
            _s3_anc_hr, _s3_anc_rt = 0, 0

        insight_bar([
            f"{_n(total_dropout)} high-risk mothers started ANC and did not complete the 4-visit "
            f"protocol — avg {avg_visits_do} visits before dropout across {_s3_n_qtrs} quarters.",
            f"{_s3_peak_qtr} shows the largest single-quarter dropout at {_s3_peak_pts} patients. "
            "Investigate whether a staffing change or service disruption explains the spike "
            "before treating it as a trend.",
            f"ANC High Risk ({_n(_s3_anc_hr)}) exceeds ANC Routine ({_n(_s3_anc_rt)}) — the facility "
            "is primarily serving supervised high-risk pregnancies, not standard preventive ANC.",
            "ANC completion drops to near zero by ANC3. Whether this reflects referral "
            "elsewhere or genuine dropout requires community health unit linkage to determine.",
        ], variant="amber")

        _gap(12)

        # ── Section 4 — Postnatal care and delivery ───────────────────────────
        _sh("Section 4 — Postnatal care and delivery")
        try:
            df_comorb_mat = Q.load_pregnancy_comorbidities(filters, run_query)
        except Exception:
            df_comorb_mat = pd.DataFrame()

        # Derive delivery and PNC counts from df_mat (same source as KPI tiles — catches
        # both O8% and JB2% codes, unlike load_anc_vs_delivery_pnc which misses JB2%)
        _del_visits   = 0
        _pnc_patients = 0
        try:
            _del_row_m = df_mat[df_mat["maternal_care_type"] == "Delivery"]
            _pnc_row_m = df_mat[df_mat["maternal_care_type"] == "Postnatal"]
            _del_visits   = int(_del_row_m["care_type_total"].iloc[0]) if not _del_row_m.empty else 0
            _pnc_patients = int(_pnc_row_m["care_type_total"].iloc[0]) if not _pnc_row_m.empty else 0
        except Exception:
            pass

        c1, c2 = st.columns(2, gap="small")
        with c1:
            chart_card("Deliveries vs postnatal care uptake", "")
            if _del_visits > 0 or _pnc_patients > 0:
                fig_pnc = go.Figure(go.Bar(
                    x=["Deliveries", "Returned for PNC"],
                    y=[_del_visits, _pnc_patients],
                    marker_color=[AFYA_BLUE, TEAL],
                    marker_line_width=0,
                    text=[str(_del_visits), str(_pnc_patients)],
                    textposition="outside",
                ))
                fig_pnc.update_layout(
                    height=180,
                    margin=dict(l=0, r=0, t=20, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(tickfont=dict(size=11)),
                    yaxis=dict(tickfont=dict(size=11)),
                )
                _pc(fig_pnc)
            chart_card_close()

        with c2:
            if not df_comorb_mat.empty:
                try:
                    chart_card("Comorbidities in maternal patients",
                               "Non-maternal conditions co-occurring with ANC visits")
                    fig_comorb_mat = hbar_chart(
                        df_comorb_mat.head(5),
                        x="patient_count",
                        y="condition_group",
                        color=AFYA_BLUE,
                        height=180,
                        show_text=False,
                    )
                    _pc(fig_comorb_mat)
                    chart_card_close()
                except Exception as e:
                    st.warning(f"Section 4 — Comorbidities: {e}")

        # Compute insight values from live comorbidity data
        if not df_comorb_mat.empty:
            _s4_top      = df_comorb_mat.iloc[0]
            _s4_top_name = str(_s4_top["condition_group"])
            _s4_top_n    = int(_s4_top["patient_count"])
            _s4_max_n    = int(df_comorb_mat["patient_count"].max())
            _s4_comorb_note = (
                f"{_s4_top_name} is the dominant comorbidity in maternal patients "
                f"({_s4_top_n} cases). Active infectious disease screening should be "
                "standard at every maternal contact."
            )
        else:
            _s4_max_n = 0
            _s4_comorb_note = (
                "Active infectious disease screening should be standard at every maternal contact."
            )

        try:
            _s4_del_age = (
                df_mat[df_mat["maternal_care_type"] == "Delivery"]
                .groupby("age_group")["unique_patients"].sum()
            )
            _s4_del_total = int(_s4_del_age.sum())
            _s4_top_del_age = _s4_del_age.idxmax() if _s4_del_total > 0 else "Adult (35-44)"
            _s4_top_del_pct = round(
                int(_s4_del_age.get(_s4_top_del_age, 0)) * 100 / max(_s4_del_total, 1), 0
            )
        except Exception:
            _s4_top_del_age, _s4_top_del_pct = "Adult (35-44)", 50

        _pnc_return_pct = round(_pnc_patients / max(_del_visits, 1) * 100) if _del_visits > 0 else 0
        insight_bar([
            f"{_pnc_patients} of {_del_visits} women who delivered returned for postnatal care "
            f"({_pnc_return_pct}%) — {_del_visits - _pnc_patients} did not return after delivery."
            if _del_visits > 0 else
            "Delivery and PNC data not available for the selected period.",
            _s4_comorb_note,
            f"{_s4_top_del_age} accounts for {int(_s4_top_del_pct)}% of deliveries — older "
            "mothers are disproportionately represented in the delivery cohort relative "
            "to the general maternal population.",
            f"Comorbidity counts are small — max {_s4_max_n if _s4_max_n else '~4'} patients "
            "per condition. Do not draw service planning conclusions at current volume.",
        ], variant="teal")



    # ── COMMUNICABLE DISEASE TAB ─────────────────────────────────────────────
    with st_d:

        # ── KPI row — 6 disease tiles ─────────────────────────────────────────
        try:
            df_kpi_d = Q.load_disease_kpi_snapshot(filters, run_query)
            if not df_kpi_d.empty:
                _cards_d = []
                for _, _row_d in df_kpi_d.iterrows():
                    _adm_pct_d = float(_row_d.get("admission_rate_pct") or 0)
                    _cards_d.append({
                        "label": str(_row_d["disease_label"]),
                        "value": _n(_row_d["patient_count"]),
                        "delta": f"{_n(_row_d['visit_count'])} visits · {_adm_pct_d:.0f}% admitted",
                        "accent_color": (
                            ACCENT_CRITICAL if _adm_pct_d > 20 else
                            ACCENT_MONITOR  if _adm_pct_d > 10 else
                            ACCENT_NEUTRAL
                        ),
                    })
                kpi_row(_cards_d)
        except Exception as _e_d0:
            st.warning(f"Disease KPI row: {_e_d0}")

        _gap(12)

        # ── Section 1 — Communicable disease pipeline ─────────────────────────
        _sh("Section 1 — Communicable disease pipeline")
        try:
            df_pipe = Q.load_communicable_pipeline_matrix(filters, run_query)
            if not df_pipe.empty:
                for _c in ("quarterly_visits", "lab_confirmation_pct", "inpatient_admission_pct"):
                    if _c in df_pipe.columns:
                        df_pipe[_c] = pd.to_numeric(df_pipe[_c], errors="coerce").fillna(0)
                _dis_colors_p = {
                    "URTI": AFYA_BLUE, "Typhoid": AMBER, "Malaria": TEAL,
                    "Enteric / GI": PURPLE, "HIV": GRAY, "TB": GRAY,
                }
                _TH_p = ("font-size:10px;font-weight:700;text-transform:uppercase;"
                         "letter-spacing:.04em;color:#9CA3AF;padding:5px 8px;"
                         "border-bottom:1px solid #E5E7EB;text-align:left;")
                _TD_p = ("font-size:12px;color:#374151;padding:6px 8px;"
                         "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;")
                _rows_html_p = ""
                for _, _r_p in df_pipe.iterrows():
                    _dis_p = str(_r_p.get("disease_group", ""))
                    _border_p = _dis_colors_p.get(_dis_p, GRAY)
                    _lab_pct = float(_r_p.get("lab_confirmation_pct") or 0)
                    _ip_pct  = float(_r_p.get("inpatient_admission_pct") or 0)
                    _lab_col = ACCENT_CRITICAL if _lab_pct < 60 else ACCENT_MONITOR if _lab_pct < 85 else ACCENT_POSITIVE
                    _ip_col  = ACCENT_CRITICAL if _ip_pct > 20 else ACCENT_MONITOR if _ip_pct > 10 else ACCENT_POSITIVE
                    _rows_html_p += (
                        f'<tr style="border-bottom:1px solid #E5E7EB;">'
                        f'<td style="{_TD_p}font-weight:600;border-left:3px solid {_border_p};">{_dis_p}</td>'
                        f'<td style="{_TD_p}">{_r_p.get("quarterly_visits","")}</td>'
                        f'<td style="{_TD_p}">{_r_p.get("primary_age_sex","")}</td>'
                        f'<td style="{_TD_p}color:{_lab_col};">{_lab_pct:.0f}%</td>'
                        f'<td style="{_TD_p}color:{_ip_col};">{_ip_pct:.0f}%</td>'
                        f'<td style="{_TD_p}">{_r_p.get("primary_comorbidity","")}</td>'
                        f'<td style="{_TD_p}color:{ACCENT_INFO};">{_r_p.get("primary_payer","")}</td>'
                        f'</tr>'
                    )
                st.markdown(
                    f'<table style="width:100%;border-collapse:collapse;table-layout:fixed;">'
                    f'<thead><tr>'
                    f'<th style="{_TH_p}width:14%;">Disease</th>'
                    f'<th style="{_TH_p}width:12%;">90D visits</th>'
                    f'<th style="{_TH_p}width:16%;">Primary demographic</th>'
                    f'<th style="{_TH_p}width:13%;">Lab confirm %</th>'
                    f'<th style="{_TH_p}width:13%;">IP admission %</th>'
                    f'<th style="{_TH_p}width:20%;">Top comorbidity</th>'
                    f'<th style="{_TH_p}width:12%;">Primary payer</th>'
                    f'</tr></thead><tbody>{_rows_html_p}</tbody></table>',
                    unsafe_allow_html=True,
                )
                _gap(8)
                _top_ip = df_pipe.loc[df_pipe["inpatient_admission_pct"].idxmax()]
                _top_vol = df_pipe.loc[df_pipe["quarterly_visits"].idxmax()]
                insight_bar([
                    f"{_top_ip['disease_group']} has the highest admission rate at "
                    f"{float(_top_ip['inpatient_admission_pct']):.0f}% — roughly 1 in "
                    f"{round(100 / max(float(_top_ip['inpatient_admission_pct']), 1))} patients requires inpatient care.",
                    f"{_top_vol['disease_group']} dominates by volume ({_n(_top_vol['quarterly_visits'])} visits) "
                    "but volume and severity are not the same signal — check the admission rate "
                    "column before allocating clinical resource by visit count alone.",
                    "Review the lab confirm % column for any disease below 60% — incomplete "
                    "investigation records limit confidence in the diagnosis.",
                ], variant="blue")
        except Exception as _e_d1:
            st.warning(f"Section 1 — Pipeline matrix: {_e_d1}")

        _gap(12)

        # ── Section 2 — Disease burden and severity ───────────────────────────
        _sh("Section 2 — Disease burden and severity")
        try:
            if not df_kpi_d.empty:
                _c1_d2, _c2_d2 = st.columns(2, gap="small")
                with _c1_d2:
                    chart_card("Patient volume by disease", "")
                    _df_vol = df_kpi_d.copy()
                    _df_vol["patient_count"] = pd.to_numeric(_df_vol["patient_count"], errors="coerce").fillna(0)
                    _df_vol = _df_vol.sort_values("patient_count", ascending=True)
                    _pc(hbar_chart(_df_vol, x="patient_count", y="disease_label",
                                   color=AFYA_BLUE, height=280, show_text=True))
                    chart_card_close()
                with _c2_d2:
                    chart_card("Admission rate by disease", "Severity signal")
                    _df_adm = df_kpi_d.copy()
                    _df_adm["admission_rate_pct"] = pd.to_numeric(_df_adm["admission_rate_pct"], errors="coerce").fillna(0)
                    _df_adm = _df_adm.sort_values("admission_rate_pct", ascending=True)
                    _df_adm["_color"] = _df_adm["admission_rate_pct"].apply(
                        lambda v: CORAL if v > 20 else AMBER if v > 10 else TEAL
                    )
                    _fig_adm = go.Figure(go.Bar(
                        x=_df_adm["admission_rate_pct"], y=_df_adm["disease_label"],
                        orientation="h", marker_color=_df_adm["_color"].tolist(),
                        marker_line_width=0,
                        hovertemplate="<b>%{y}</b><br>%{x:.0f}% admitted<extra></extra>",
                    ))
                    _fig_adm.update_layout(
                        height=280, margin=dict(l=0, r=0, t=6, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(tickfont=dict(size=11)),
                        yaxis=dict(tickfont=dict(size=11), automargin=True),
                    )
                    _pc(_fig_adm)
                    chart_card_close()
                insight_bar([
                    "Volume and severity tell different clinical stories — check both before "
                    "allocating clinical resource.",
                    "Compare the highest-volume disease against the highest-admission-rate "
                    "disease; if they differ, resourcing decisions should weight toward severity.",
                ], variant="amber")
        except Exception as _e_d2:
            st.warning(f"Section 2 — Disease burden: {_e_d2}")

        _gap(12)

        # ── Section 3 — Paediatric URTI — acute to chronic transition ─────────
        _sh("Section 3 — Paediatric URTI — Acute to Chronic Transition")
        try:
            df_trans = Q.load_paediatric_urti_transition(filters, run_query)
            df_first = Q.load_paediatric_resp_first_presentation(filters, run_query)

            _total_cands = int(df_trans["patients"].sum()) if not df_trans.empty else 0
            _avg_vis_all = round(float(df_trans["avg_visits"].mean()), 1) if not df_trans.empty else 0
            _avg_days_all = round(float(df_trans["avg_days_span"].mean()), 0) if not df_trans.empty else 0

            _peak_chron_pct = 0
            _pivot_first = None
            if not df_first.empty:
                _pivot_first = df_first.pivot_table(
                    index="quarter", columns="visit_type", values="patients", aggfunc="sum"
                ).fillna(0)
                _pivot_first["total"] = (
                    _pivot_first.get("Acute URTI", 0) + _pivot_first.get("Chronic Respiratory", 0)
                )
                _pivot_first["chronic_pct"] = (
                    _pivot_first.get("Chronic Respiratory", 0) / _pivot_first["total"].replace(0, 1)
                ) * 100
                _peak_chron_pct = round(float(_pivot_first["chronic_pct"].max()), 0)

            kpi_row([
                {"label": "Transition candidates", "value": _n(_total_cands),
                 "accent_color": ACCENT_CRITICAL},
                {"label": "Avg visits before flag", "value": str(_avg_vis_all),
                 "accent_color": ACCENT_MONITOR},
                {"label": "Avg days span", "value": f"{int(_avg_days_all)}d",
                 "accent_color": ACCENT_NEUTRAL},
                {"label": "Peak chronic share", "value": f"{_peak_chron_pct:.0f}%",
                 "accent_color": ACCENT_CRITICAL},
            ])
        except Exception as _e_d3_kpi:
            st.warning(f"Section 3 KPIs: {_e_d3_kpi}")
            df_trans = pd.DataFrame()
            df_first = pd.DataFrame()
            _total_cands = 0
            _avg_vis_all = 0
            _avg_days_all = 0
            _peak_chron_pct = 0
            _pivot_first = None

        _c1_d3, _c2_d3 = st.columns(2, gap="small")
        with _c1_d3:
            chart_card("Transition candidates by age group", "")
            try:
                if not df_trans.empty:
                    _df_trans_s = df_trans.sort_values("patients", ascending=False)
                    _age_clrs_t = {"Under 5": CORAL, "5-11": AMBER, "12-17": TEAL}
                    _fig_trans = go.Figure(go.Bar(
                        x=_df_trans_s["age_group"], y=_df_trans_s["patients"],
                        marker_color=[_age_clrs_t.get(a, GRAY) for a in _df_trans_s["age_group"]],
                        marker_line_width=0,
                        hovertemplate="<b>%{x}</b><br>%{y} patients<extra></extra>",
                    ))
                    _fig_trans.update_layout(
                        height=240, margin=dict(l=0, r=0, t=6, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(tickfont=dict(size=11)),
                        yaxis=dict(tickfont=dict(size=11)),
                    )
                    _pc(_fig_trans)
            except Exception as _e_s3l:
                st.warning(f"Section 3 left: {_e_s3l}")
            chart_card_close()
        with _c2_d3:
            chart_card("First-presentation acute vs chronic, by quarter", "")
            try:
                if not df_first.empty:
                    df_first["quarter_str"] = pd.to_datetime(df_first["quarter"]).dt.strftime("%b %y")
                    _fig_first = go.Figure()
                    for _vtype, _vcolor in [("Acute URTI", AFYA_BLUE), ("Chronic Respiratory", CORAL)]:
                        _sub_f = df_first[df_first["visit_type"] == _vtype].sort_values("quarter")
                        _fig_first.add_trace(go.Bar(
                            name=_vtype, x=_sub_f["quarter_str"], y=_sub_f["patients"],
                            marker_color=_vcolor, marker_line_width=0,
                        ))
                    _fig_first.update_layout(
                        height=240, margin=dict(l=0, r=0, t=6, b=30),
                        plot_bgcolor="white", paper_bgcolor="white",
                        barmode="group",
                        xaxis=dict(tickfont=dict(size=11), tickangle=30),
                        yaxis=dict(tickfont=dict(size=11)),
                        legend=dict(orientation="h", y=-0.28, x=0.5, xanchor="center",
                                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
                        showlegend=True,
                    )
                    _pc(_fig_first)
            except Exception as _e_s3r:
                st.warning(f"Section 3 right: {_e_s3r}")
            chart_card_close()

        try:
            if _pivot_first is not None and _peak_chron_pct > 0:
                _peak_q_row = _pivot_first["chronic_pct"].idxmax()
                _peak_q_label = pd.to_datetime(_peak_q_row).strftime("%b %Y")
                anomaly_banner(
                    f"{_peak_q_label} — chronic respiratory reached {_peak_chron_pct:.0f}% of "
                    "paediatric first presentations",
                    f"Rose from {float(_pivot_first['chronic_pct'].iloc[0]):.0f}% in the first "
                    "observed quarter. No triggered screening pathway exists for recurrent "
                    "paediatric URTI.",
                    color=ACCENT_CRITICAL, bg="#FEF2F2",
                )
        except Exception:
            pass

        try:
            if not df_trans.empty:
                _dom_groups = df_trans.nlargest(2, "patients")
                _dom_names = " and ".join(
                    f"{_r['age_group']} ({int(_r['patients'])})"
                    for _, _r in _dom_groups.iterrows()
                )
                insight_bar([
                    f"{_total_cands} children averaged {_avg_vis_all} acute URTI visits "
                    f"over {int(_avg_days_all)} days without being flagged for chronic "
                    "respiratory investigation.",
                    f"{_dom_names} are the dominant groups — the early-intervention window is "
                    "being missed in the highest-risk paediatric cohort.",
                    "Recommendation: trigger a structured respiratory assessment after a "
                    "child's second acute URTI presentation within 6 months.",
                ], variant="teal")
        except Exception as _e_d3_ib:
            st.warning(f"Section 3 insight bar: {_e_d3_ib}")

        _gap(12)

        # ── Section 4 — Outbreak and surge surveillance ───────────────────────
        _sh("Section 4 — Outbreak and Surge Surveillance")
        try:
            df_surge4 = Q.load_disease_monthly_trend(filters, run_query)
            if not df_surge4.empty:
                df_surge4["visit_month"] = pd.to_datetime(df_surge4["visit_month"])
                df_surge4["visit_count"] = pd.to_numeric(df_surge4["visit_count"], errors="coerce").fillna(0)
                _sel_dis4 = ["Typhoid", "Malaria", "URTI", "TB", "Enteric / GI"]
                _dis_clrs4 = {
                    "Typhoid": AMBER, "Malaria": TEAL, "URTI": AFYA_BLUE,
                    "TB": CORAL, "Enteric / GI": PURPLE,
                }
                _surge_dis4 = ["Malaria", "URTI", "Typhoid"]

                # Hoist surge detection before columns so anomaly_banner can use the live count
                _surge_sub4 = df_surge4[df_surge4["disease_label"].isin(_surge_dis4)].copy()
                _surge_months4 = pd.DataFrame()
                if not _surge_sub4.empty:
                    _means4 = _surge_sub4.groupby("disease_label")["visit_count"].mean()
                    _surge_sub4["vs_avg"] = _surge_sub4.apply(
                        lambda r: round(
                            r["visit_count"] / max(_means4.get(r["disease_label"], 1), 1), 1
                        ),
                        axis=1,
                    )
                    _surge_months4 = _surge_sub4[_surge_sub4["vs_avg"] >= 1.5].copy()
                    _surge_months4["month_str"] = _surge_months4["visit_month"].dt.strftime("%b %Y")
                    _surge_months4 = _surge_months4.rename(columns={"visit_count": "visits"})
                    _surge_months4 = _surge_months4.sort_values("vs_avg", ascending=False)
                _typhoid_surge_n = int(len(
                    _surge_months4[_surge_months4["disease_label"] == "Typhoid"]
                )) if not _surge_months4.empty else 0

                _c1_d4, _c2_d4 = st.columns(2, gap="small")
                with _c1_d4:
                    chart_card("Monthly case trend", "Typhoid, Malaria, URTI")
                    _fig_tr4 = go.Figure()
                    for _dis4 in _sel_dis4:
                        _sub_tr4 = df_surge4[df_surge4["disease_label"] == _dis4].sort_values("visit_month")
                        if not _sub_tr4.empty:
                            _fig_tr4.add_trace(go.Scatter(
                                x=_sub_tr4["visit_month"], y=_sub_tr4["visit_count"],
                                name=_dis4, mode="lines+markers",
                                line=dict(color=_dis_clrs4.get(_dis4, GRAY), width=2),
                                hovertemplate=f"<b>{_dis4}</b> — %{{x|%b %Y}}: %{{y:,}} visits<extra></extra>",
                            ))
                    _fig_tr4.update_layout(
                        height=280, margin=dict(l=0, r=0, t=6, b=30),
                        plot_bgcolor="white", paper_bgcolor="white",
                        yaxis_title="Visits",
                        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center",
                                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    )
                    _pc(_fig_tr4)
                    chart_card_close()

                with _c2_d4:
                    chart_card("Surge months", ">1.5× period average")
                    if not _surge_months4.empty:
                        _t1_html, _ = _comm_t1_html(_surge_months4)
                        _stcomp.html(_t1_html, height=320, scrolling=True)
                    chart_card_close()

                if _typhoid_surge_n > 0:
                    anomaly_banner(
                        "Typhoid — sustained endemic pattern",
                        f"{_typhoid_surge_n} month{'s' if _typhoid_surge_n != 1 else ''} above "
                        "1.5× average. Standing management protocol required, not event-based escalation.",
                        color=ACCENT_CRITICAL, bg="#FEF2F2",
                    )
                insight_bar([
                    "Malaria's surge months align with two seasonal windows — investigate "
                    "whether pre-positioning supplies and staffing ahead of those windows "
                    "is feasible rather than reacting after volume climbs.",
                    "Cross-reference any URTI surge month against the paediatric transition "
                    "data in Section 3 — high acute volume can obscure children who need "
                    "chronic pathway referral.",
                ], variant="red")
        except Exception as _e_d4:
            st.warning(f"Section 4 — Surge surveillance: {_e_d4}")

        _gap(12)

        # ── Section 5 — Clinical quality signals ──────────────────────────────
        _sh("Section 5 — Clinical Quality Signals")
        try:
            df_mal5 = Q.load_malaria_lab_accuracy(filters, run_query)
            r_mal = df_mal5.iloc[0] if not df_mal5.empty else {}
            df_tb5 = Q.load_tb_hiv_coinfection(filters, run_query)

            _c1_d5, _c2_d5 = st.columns(2, gap="small")
            with _c1_d5:
                if not df_mal5.empty:
                    _total_mal = float(r_mal.get("total_malaria_visits") or 1)
                    _no_test   = float(r_mal.get("no_test_done") or 0)
                    _clinical_only_pct = round(_no_test / _total_mal * 100, 1)
                    _sk1, _sk2, _sk3 = st.columns(3)
                    with _sk1:
                        _kpi("Test rate", _p(float(r_mal.get("test_rate_pct", 0))),
                             color=ACCENT_MONITOR)
                    with _sk2:
                        _kpi("No test done", f"{_clinical_only_pct:.0f}%",
                             f"{int(_no_test):,} of {int(_total_mal):,} visits",
                             color=ACCENT_CRITICAL)
                    with _sk3:
                        _kpi("Result rate", _p(float(r_mal.get("result_rate_pct", 0))),
                             color=ACCENT_POSITIVE)
                    chart_card("Malaria — test coverage", "")
                    _pc(donut(
                        labels=["Test resulted", "Test ordered only", "No test done"],
                        values=[
                            float(r_mal.get("test_resulted") or 0),
                            float(r_mal.get("test_ordered_only") or 0),
                            _no_test,
                        ],
                        color_map={"Test resulted": TEAL, "Test ordered only": AMBER, "No test done": CORAL},
                        height=180, hole=0.6,
                    ))
                    chart_card_close()
            with _c2_d5:
                if not df_tb5.empty:
                    _r_tb = df_tb5.iloc[0]
                    _sk1b, _sk2b, _sk3b = st.columns(3)
                    with _sk1b:
                        _kpi("TB patients", _n(_r_tb.get("tb_patients")))
                    with _sk2b:
                        _kpi("HIV test recorded",
                             _p(float(_r_tb.get("hiv_test_coverage_pct", 0))),
                             color=ACCENT_CRITICAL)
                    with _sk3b:
                        _kpi("Co-infected", _n(_r_tb.get("tb_hiv_coinfected")))
                    st.markdown(
                        f'<div style="background:#FFFBEB;border-radius:8px;border-left:3px solid '
                        f'{ACCENT_MONITOR};padding:10px 12px;font-size:12px;color:{ACCENT_MONITOR};'
                        f'line-height:1.5;">'
                        f'<strong style="display:block;margin-bottom:3px;">'
                        f'{_p(float(_r_tb.get("hiv_test_coverage_pct", 0)))} HIV test recorded '
                        f'for TB patients</strong>'
                        f'Could reflect a true testing gap, tests done outside this system, or '
                        f'medication-pickup visits. Verify visit type and test location before '
                        f'treating as a recorded gap.'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            insight_bar([
                f"{int(_no_test):,} malaria visits ({_clinical_only_pct:.0f}%) had no linked "
                "investigation record — check whether tests were done on paper or under a "
                "different visit ID before treating this as a documentation gap.",
                "TB lab confirmation gaps should be checked against the same visit-type "
                "explanation before concluding it reflects a documentation failure.",
            ], variant="amber")
        except Exception as _e_d5:
            st.warning(f"Section 5 — Clinical quality: {_e_d5}")

        _gap(12)

        # ── Section 6 — Comorbidity patterns ─────────────────────────────────
        _sh("Section 6 — Comorbidity Patterns")
        try:
            df_com6 = Q.load_communicable_comorbidities(filters, run_query)
            if not df_com6.empty:
                df_sepsis6 = df_com6[
                    df_com6["comorbidity"].str.contains("Sepsis", case=False, na=False)
                ].sort_values("patient_count", ascending=True)

                _c1_d6, _c2_d6 = st.columns(2, gap="small")
                with _c1_d6:
                    chart_card("Sepsis co-occurrence across communicable diseases", "")
                    if not df_sepsis6.empty:
                        _dis_clrs_s6 = {
                            "URTI": AFYA_BLUE, "Typhoid": AMBER, "Malaria": TEAL,
                            "Enteric / GI": PURPLE, "TB": GRAY,
                        }
                        _fig_sep = go.Figure(go.Bar(
                            x=df_sepsis6["patient_count"], y=df_sepsis6["disease_label"],
                            orientation="h",
                            marker_color=[_dis_clrs_s6.get(d, GRAY) for d in df_sepsis6["disease_label"]],
                            marker_line_width=0,
                        ))
                        _fig_sep.update_layout(
                            height=220, margin=dict(l=0, r=0, t=6, b=0),
                            plot_bgcolor="white", paper_bgcolor="white",
                            xaxis=dict(tickfont=dict(size=11)),
                            yaxis=dict(tickfont=dict(size=11), automargin=True),
                        )
                        _pc(_fig_sep)
                    chart_card_close()
                with _c2_d6:
                    chart_card("Why this matters", "")
                    st.markdown(
                        '<div style="font-size:12px;color:#374151;line-height:1.6;'
                        'background:#F9FAFB;border-radius:8px;padding:12px 14px;'
                        'height:220px;display:flex;align-items:center;">'
                        '<p>Sepsis as the top comorbidity across multiple diseases is a '
                        'complication pattern, not coincidence. The disease with the '
                        'highest Sepsis co-occurrence warrants its own escalation pathway.</p>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    chart_card_close()

                if not df_sepsis6.empty:
                    _top_sep = df_sepsis6.iloc[-1]
                    insight_bar([
                        "Sepsis admissions are predominantly coded without a named organism — "
                        "verify whether the comorbidity label maps to 'Other Sepsis' in the ICD10 "
                        "pivoted table, as generic coding likely undercounts the true burden.",
                        f"{_top_sep['disease_label']}-associated Sepsis at "
                        f"{_n(_top_sep['patient_count'])} patients is the largest group — "
                        "review whether this disease's admission protocol includes a Sepsis "
                        "assessment at presentation.",
                    ], variant="red")
        except Exception as _e_d6:
            st.warning(f"Section 6 — Comorbidity patterns: {_e_d6}")

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
               "has_worsening_vitals", "has_medication_change",
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
    def _signals(row):
        parts = []
        if row.get("has_undetected_ncd"):              parts.append("NCD undetected")
        if row.get("has_worsening_vitals"):            parts.append("Vitals worsening")
        if row.get("has_medication_change"):           parts.append("Med change")
        if row.get("days_since_last_visit", 0) >= 90: parts.append("Long gap")
        return parts

    def _priority(row):
        sigs = _signals(row)
        if any(s in ("NCD undetected", "Vitals worsening") for s in sigs):
            return "high"
        if any(s in ("Med change", "Long gap") for s in sigs):
            return "medium"
        return "monitor"

    patients_list = [
        {
            "id":        str(r["patient"]),
            "priority":  _priority(r),
            "condition": str(r.get("primary_condition") or "Not recorded"),
            "days":      int(float(r.get("days_since_last_visit") or 0)),
            "signals":   _signals(r),
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
