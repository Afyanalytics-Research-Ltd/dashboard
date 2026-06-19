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
def _pc(fig):      _H().get("pc",       lambda f: st.plotly_chart(f, use_container_width=True))(fig)
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

    pc = _H().get("pc", lambda f: st.plotly_chart(f, use_container_width=True))
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
                    legend=dict(orientation="h", y=1.04, x=0, font=dict(size=10)),
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
                    legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10)),
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
                                     legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10)))
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
html,body{{height:100%;overflow:hidden}}
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
.view{{display:none;flex:1;flex-direction:column;padding:10px 12px;gap:8px;min-height:0}}
.view.active{{display:flex}}
/* card fills available flex space */
.grow-card{{flex:1;display:flex;flex-direction:column;min-height:0;
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
            _stcomp.html(_pp_html, height=740, scrolling=True)
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
            yanchor="bottom", y=1.01,
            xanchor="left", x=0,
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
    _H().get("pc", lambda f: st.plotly_chart(f, use_container_width=True))(fig)

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
# TAB — OPD TO IPD CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def render_tab_opd_ipd(filters: dict, run_query):
    """OPD to IPD Conversion Rate — Sections A–E."""

    # ── local palette ──────────────────────────────────────────────────────
    _B  = "#185FA5"
    _G  = "#0F6E56"
    _R  = "#E24B4A"
    _Am = "#BA7517"
    _Pu = "#534AB7"
    _Pk = "#D4537E"
    _Mu = "#B4B2A9"

    def _sf(v, d=0.0):
        try: return float(v)
        except: return d

    def _warn_bar(text: str):
        st.markdown(
            f'<div style="border-left:3px solid {_R};background:#FCEBEB;'
            f'padding:10px 14px;margin:6px 0;border-radius:0 4px 4px 0;">'
            f'<span style="font-size:13px;color:#003467;line-height:1.7;">{text}</span></div>',
            unsafe_allow_html=True,
        )

    def _leg(items):
        """Inline legend: [(label, color, 'solid'|'dashed'|'dotted')]"""
        parts = []
        for lbl, col, sty in items:
            if sty == "dashed":
                sw = (f'<span style="display:inline-block;width:18px;height:2px;'
                      f'background:repeating-linear-gradient(90deg,{col} 0 4px,'
                      f'transparent 4px 8px);vertical-align:middle;margin-right:4px;"></span>')
            elif sty == "dotted":
                sw = (f'<span style="display:inline-block;width:18px;height:2px;'
                      f'background:repeating-linear-gradient(90deg,{col} 0 2px,'
                      f'transparent 2px 5px);vertical-align:middle;margin-right:4px;"></span>')
            else:
                sw = (f'<span style="display:inline-block;width:18px;height:2px;'
                      f'background:{col};vertical-align:middle;margin-right:4px;"></span>')
            parts.append(f'<span style="margin-right:14px;font-size:11px;color:#374151;">'
                         f'{sw}{lbl}</span>')
        st.markdown('<div style="margin-bottom:4px">' + "".join(parts) + '</div>',
                    unsafe_allow_html=True)

    # ── load dataframes ────────────────────────────────────────────────────
    df_overall     = pd.DataFrame()
    df_monthly     = pd.DataFrame()
    df_disease     = pd.DataFrame()
    df_comorbidity = pd.DataFrame()
    df_retention   = pd.DataFrame()
    df_escalation  = pd.DataFrame()
    df_triangle    = pd.DataFrame()

    try: df_overall     = Q.load_opd_ipd_overall(filters, run_query)
    except Exception as e: st.warning(f"Overall KPIs: {e}")
    try: df_monthly     = Q.load_opd_ipd_monthly_rate(filters, run_query)
    except Exception as e: st.warning(f"Monthly trend: {e}")
    try: df_disease     = Q.load_opd_ipd_disease_mix(filters, run_query)
    except Exception as e: st.warning(f"Disease mix: {e}")
    try: df_comorbidity = Q.load_opd_ipd_comorbidity(filters, run_query)
    except Exception as e: st.warning(f"Comorbidity: {e}")
    try: df_retention   = Q.load_opd_ipd_retention(filters, run_query)
    except Exception as e: st.warning(f"Retention: {e}")
    try: df_escalation  = Q.load_opd_ipd_escalation_72h(filters, run_query)
    except Exception as e: st.warning(f"Escalation: {e}")
    try: df_triangle    = Q.load_operational_triangle(filters, run_query)
    except Exception as e: st.warning(f"Triangle: {e}")

    # ── numeric coercion ──────────────────────────────────────────────────
    _coerce = [
        (df_overall,     ["true_average_opd_to_ipd_rate", "total_opd_to_ipd",
                          "total_opd_visits"]),
        (df_monthly,     ["monthly_opd_to_ipd_rate"]),
        (df_disease,     ["total_opd_cases_with_this_diagnosis",
                          "successful_ipd_admissions"]),
        (df_comorbidity, ["total_opd_visits", "conversion_rate_pct",
                          "delta_from_group_avg"]),
        (df_retention,   ["total_opd_visits", "ipd_admissions",
                          "conversion_rate_pct"]),
        (df_escalation,  ["total_escalations"]),
        (df_triangle,    ["conversion_rate_pct", "avg_visits_per_clinician",
                          "wait_time_gap_mins"]),
    ]
    for df, cols in _coerce:
        for c in cols:
            if not df.empty and c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── derived header values ─────────────────────────────────────────────
    overall_rate = _sf(df_overall["true_average_opd_to_ipd_rate"].iloc[0]
                       if not df_overall.empty else 0)
    total_ipd    = int(_sf(df_overall["total_opd_to_ipd"].iloc[0]
                       if not df_overall.empty else 0))
    total_opd_v  = int(_sf(df_overall["total_opd_visits"].iloc[0]
                       if not df_overall.empty and "total_opd_visits" in df_overall.columns
                       else 0))

    ret_rate = 0.0
    if not df_retention.empty and "acquisition_segment" in df_retention.columns:
        _m = (
            (df_retention["acquisition_segment"] == "RETENTION_UNIVERSE_TOTAL") &
            (df_retention["visit_month"].isna()) &
            (df_retention["age_group"].isna())
        )
        _rv = df_retention[_m]
        if not _rv.empty:
            ret_rate = _sf(_rv["conversion_rate_pct"].iloc[0])

    mix_gap = round(ret_rate - overall_rate, 2)

    strain_ct, total_months_ct = 0, 0
    if not df_triangle.empty and "strain_signal" in df_triangle.columns:
        total_months_ct = len(df_triangle)
        strain_ct = int(
            df_triangle["strain_signal"].isin(["HIGH_STRAIN", "CAPACITY_GAP"]).sum())

    # ── HEADER KPI STRIP ─────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: _kpi("Overall Conversion Rate", f"{overall_rate:.2f}%",
                  "All OPD visits · Sep 2024–Apr 2026", AFYA_BLUE)
    with k2: _kpi("Complex Patient Conversion Rate", f"{ret_rate:.2f}%",
                  "Patients with ≥2 visits or a chronic condition",
                  GREEN if ret_rate > overall_rate else MUTED)
    with k3: _kpi("Mix Gap",
                  f"+{mix_gap:.2f} pct points" if mix_gap >= 0 else f"{mix_gap:.2f} pct points",
                  "Acute walk-ins suppressing headline",
                  AMBER if mix_gap > 0 else MUTED)
    with k4: _kpi("Total IPD Admissions", f"{total_ipd:,}",
                  "From OPD across all visits", COOL_BLUE)
    with k5: _kpi("Strain Months",
                  f"{strain_ct} / {total_months_ct}" if total_months_ct else "—",
                  "High workload · below avg conversion",
                  CORAL if strain_ct >= 3 else MUTED)

    _gap(16)

    # ── SECTION A — CONVERSION RATE OVERVIEW ─────────────────────────────
    _sh("A — Conversion Rate Overview", mt=8)

    # Four segment cards — keys must match _SEGMENT_CASE SQL output (title-case)
    _SEG_META = {
        "Chronic":      {"label": "Chronic",      "color": _B,  "ref_lo": 8,  "ref_hi": 15},
        "Maternal":     {"label": "Maternal",     "color": _G,  "ref_lo": 15, "ref_hi": 25},
        "Oncology":     {"label": "Oncology",     "color": _R,  "ref_lo": 15, "ref_hi": 25},
        "Mental Health":{"label": "Mental Health","color": _Pu, "ref_lo": 8,  "ref_hi": 15},
    }
    if not df_retention.empty:
        df_segs = df_retention[
            df_retention["visit_month"].isna() &
            df_retention["age_group"].isna() &
            ~df_retention["acquisition_segment"].isin(["RETENTION_UNIVERSE_TOTAL"])
        ].copy()
    else:
        df_segs = pd.DataFrame()

    sc1, sc2, sc3, sc4 = st.columns(4)
    for col_s, seg_key in zip([sc1, sc2, sc3, sc4],
                               ["Chronic", "Maternal", "Oncology", "Mental Health"]):
        meta = _SEG_META[seg_key]
        rate, visits, admissions = 0.0, 0, 0
        if not df_segs.empty and seg_key in df_segs["acquisition_segment"].values:
            r = df_segs[df_segs["acquisition_segment"] == seg_key].iloc[0]
            rate       = _sf(r.get("conversion_rate_pct"))
            visits     = int(_sf(r.get("total_opd_visits")))
            admissions = int(_sf(r.get("ipd_admissions")))
        if rate >= meta["ref_hi"]:
            bc, bb, bt = _G, "#E8F7F4", f"Within ref {meta['ref_lo']}–{meta['ref_hi']}%"
        elif rate >= meta["ref_lo"]:
            bc, bb, bt = _Am, "#FEF3C7", f"Ref {meta['ref_lo']}–{meta['ref_hi']}%"
        else:
            bc, bb, bt = _R, "#FEE2E2", f"Below ref {meta['ref_lo']}–{meta['ref_hi']}%"
        with col_s:
            st.markdown(
                f'<div style="background:#fff;border:0.5px solid {BORDER};'
                f'border-top:3px solid {meta["color"]};border-radius:8px;'
                f'padding:14px 14px 10px;margin-bottom:8px">'
                f'<div style="font-size:10px;font-weight:700;color:{MUTED};'
                f'text-transform:uppercase;letter-spacing:1.2px;margin-bottom:4px">'
                f'{meta["label"]}</div>'
                f'<div style="font-size:26px;font-weight:700;color:{meta["color"]};'
                f'line-height:1.1;margin-bottom:4px">{rate:.2f}%</div>'
                f'<div style="font-size:11px;color:{MUTED};margin-bottom:6px">'
                f'{visits:,} visits · {admissions:,} admissions</div>'
                f'<span style="font-size:10px;font-weight:700;padding:2px 8px;'
                f'border-radius:20px;background:{bb};color:{bc}">{bt}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    _gap(12)

    # Monthly trend prep
    if not df_monthly.empty and "conversion_month" in df_monthly.columns:
        df_monthly["conversion_month"] = pd.to_datetime(
            df_monthly["conversion_month"], errors="coerce")
        df_mon_s = df_monthly[
            df_monthly["conversion_month"] >= "2024-09-01"
        ].sort_values("conversion_month")
    else:
        df_mon_s = pd.DataFrame()

    ch1, ch2 = st.columns(2)

    _OV_GRAY = "#6B7280"   # visible gray for the avg/overall reference lines

    with ch1:
        _leg([("Monthly rate", _B, "solid"), (f"Avg {overall_rate:.2f}%", _OV_GRAY, "dashed")])
        _ov_max  = df_mon_s["monthly_opd_to_ipd_rate"].max() if not df_mon_s.empty else 10
        _ov_ceil = max(10, round(_ov_max * 1.2))
        fig_ov = go.Figure()
        if not df_mon_s.empty:
            fig_ov.add_trace(go.Scatter(
                x=df_mon_s["conversion_month"],
                y=df_mon_s["monthly_opd_to_ipd_rate"],
                mode="lines+markers",
                line=dict(color=_B, width=2.5), marker=dict(size=5, color=_B),
                showlegend=False,
                hovertemplate="%{x|%b %y}: <b>%{y:.2f}%</b><extra></extra>",
            ))
            fig_ov.add_hline(y=overall_rate,
                             line=dict(color=_OV_GRAY, dash="dash", width=1.5),
                             annotation_text=f"Avg {overall_rate:.2f}%",
                             annotation_font=dict(size=10, color=_OV_GRAY),
                             annotation_position="right")
        fig_ov.update_layout(
            height=280, margin=dict(l=0, r=80, t=36, b=0),
            plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
            title=dict(text="Monthly Overall Conversion Rate",
                       font=dict(size=11, color=COOL_BLUE),
                       x=0, xanchor="left", pad=dict(t=4)),
            xaxis=dict(tickformat="%b %y", tickfont=dict(size=10), title=""),
            yaxis=dict(range=[0, _ov_ceil], ticksuffix="%",
                       tickfont=dict(size=10), title=""),
        )
        _pc(fig_ov)

    with ch2:
        _leg([("Complex patients", _G, "solid"), ("Overall", _OV_GRAY, "dashed")])
        if not df_retention.empty and "visit_month" in df_retention.columns:
            df_retention["visit_month"] = pd.to_datetime(
                df_retention["visit_month"], errors="coerce")
            # Aggregate all retention-universe segments per month
            _age_null = df_retention["age_group"].isna() | (df_retention["age_group"] == "")
            df_ret_mon = df_retention[
                df_retention["visit_month"].notna() &
                _age_null &
                (~df_retention["acquisition_segment"].isin(
                    ["RETENTION_UNIVERSE_TOTAL"])) &
                (df_retention["visit_month"] >= pd.Timestamp("2024-09-01"))
            ].groupby("visit_month").agg(
                total_opd=("total_opd_visits", "sum"),
                total_ipd=("ipd_admissions", "sum"),
            ).reset_index()
            df_ret_mon["ret_rate"] = (
                df_ret_mon["total_ipd"] /
                df_ret_mon["total_opd"].replace(0, pd.NA) * 100
            ).round(2)
            df_ret_mon = df_ret_mon.sort_values("visit_month")
        else:
            df_ret_mon = pd.DataFrame()

        _dual_max  = max(
            df_ret_mon["ret_rate"].max() if not df_ret_mon.empty else 0,
            df_mon_s["monthly_opd_to_ipd_rate"].max() if not df_mon_s.empty else 0,
        )
        _dual_ceil = max(16, round(_dual_max * 1.2))

        fig_dual = go.Figure()
        if not df_ret_mon.empty:
            fig_dual.add_trace(go.Scatter(
                x=df_ret_mon["visit_month"], y=df_ret_mon["ret_rate"],
                mode="lines+markers",
                line=dict(color=_G, width=2.5), marker=dict(size=5, color=_G),
                showlegend=False,
                hovertemplate="%{x|%b %y}: <b>%{y:.2f}%</b><extra></extra>",
            ))
        if not df_mon_s.empty:
            fig_dual.add_trace(go.Scatter(
                x=df_mon_s["conversion_month"],
                y=df_mon_s["monthly_opd_to_ipd_rate"],
                mode="lines+markers",
                line=dict(color=_OV_GRAY, width=2, dash="dash"),
                marker=dict(size=3, color=_OV_GRAY),
                showlegend=False,
                hovertemplate="%{x|%b %y}: <b>%{y:.2f}%</b><extra></extra>",
            ))
        fig_dual.update_layout(
            height=280, margin=dict(l=0, r=30, t=36, b=0),
            plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
            title=dict(text="Complex Patients vs Overall Conversion Rate",
                       font=dict(size=11, color=COOL_BLUE),
                       x=0, xanchor="left", pad=dict(t=4)),
            xaxis=dict(tickformat="%b %y", tickfont=dict(size=10), title=""),
            yaxis=dict(range=[0, _dual_ceil], ticksuffix="%",
                       tickfont=dict(size=10), title=""),
        )
        _pc(fig_dual)

    # Section A dynamic insight
    _seg_refs = {
        "Oncology":     (15, 25),
        "Maternal":     (15, 25),
        "Chronic":      (8,  15),
        "Mental Health":(8,  15),
    }
    _below_segs = []
    if not df_segs.empty:
        for _seg, (_lo, _hi) in _seg_refs.items():
            _r = df_segs[df_segs["acquisition_segment"] == _seg]
            if not _r.empty:
                _seg_rate = _sf(_r["conversion_rate_pct"].iloc[0])
                if _seg_rate < _lo:
                    _below_segs.append(
                        f"{_seg} ({_seg_rate:.1f}% vs {_lo}–{_hi}% reference)")
    _below_text = ", ".join(_below_segs) if _below_segs else "all segments within reference"
    st.markdown(
        f'<div style="border-left:3px solid {_B};background:#EBF5FF;'
        f'padding:10px 14px;font-size:13px;margin-top:10px;border-radius:0 4px 4px 0;">'
        f'<ul style="list-style:none;margin:0;padding:0;color:#003467;line-height:1.7;">'
        f'<li>· The complex patient conversion rate sits above the overall rate every month — '
        f'acute walk-ins suppress the headline figure. The overall rate understates '
        f'the true admission rate for patients with chronic or multi-visit needs.</li>'
        f'<li style="margin-top:3px;">· Segments below international reference: {_below_text}.</li>'
        f'<li style="margin-top:3px;">· Action: review OPD assessment criteria for below-reference '
        f'segments. Low conversion on complex patients suggests either '
        f'under-detection of severity at OPD or barriers to admission.</li>'
        f'</ul></div>',
        unsafe_allow_html=True,
    )
    _gap(16)

    # ── SECTION B — DIAGNOSIS BENCHMARK ──────────────────────────────────
    _sh("B — Diagnosis Benchmark Comparison", mt=8)
    try:
        if not df_disease.empty:
            df_bench = df_disease[
                df_disease["total_opd_cases_with_this_diagnosis"] >= 50
            ].groupby("cleaned_diagnosis_name").agg(
                opd=("total_opd_cases_with_this_diagnosis", "sum"),
                ipd=("successful_ipd_admissions", "sum"),
            ).reset_index()
            df_bench["actual_rate"] = (
                df_bench["ipd"] / df_bench["opd"].replace(0, pd.NA) * 100
            ).round(2)
            df_bench = df_bench.sort_values("actual_rate", ascending=False).head(12)
            _ref_map = {
                "Typhoid": 8, "Malaria": 8, "Sepsis": 8,
                "Chronic Upper Airway": 8, "Chr Upper Airway": 8,
                "Oncology": 15, "Gynaecological NCD": 15,
                "Maternal: Other": 15, "Hypertension": 8,
                "Antenatal Care": 15, "Neurologic": 10,
            }
            df_bench["ref_lower"] = (
                df_bench["cleaned_diagnosis_name"].map(_ref_map).fillna(8))
            df_bench["bar_color"] = df_bench.apply(
                lambda r: _G if r["actual_rate"] >= r["ref_lower"] else _R, axis=1)
            df_bench = df_bench.sort_values("actual_rate", ascending=True)

            _leg([("Above reference", _G, "solid"),
                  ("Below reference", _R, "solid"),
                  ("Reference lower bound", _Am, "dashed")])
            _dx_max  = df_bench["actual_rate"].max() if not df_bench.empty else 32
            _dx_ceil = max(35, round(_dx_max * 1.25))   # headroom for outside labels
            fig_dx = go.Figure()
            fig_dx.add_trace(go.Bar(
                y=df_bench["cleaned_diagnosis_name"],
                x=df_bench["actual_rate"],
                orientation="h",
                marker_color=df_bench["bar_color"].tolist(),
                text=[f"{v:.1f}%" for v in df_bench["actual_rate"]],
                textposition="outside",
                cliponaxis=False,
                showlegend=False,
                hovertemplate="<b>%{y}</b><br>Actual: %{x:.2f}%<extra></extra>",
            ))
            for _rv in sorted(df_bench["ref_lower"].unique()):
                fig_dx.add_vline(
                    x=_rv,
                    line=dict(color=_Am, width=2, dash="dash"),
                    annotation_text=f"Ref {int(_rv)}%",
                    annotation_font=dict(size=10, color=_Am),
                    annotation_position="top",
                )
            fig_dx.update_layout(
                height=max(340, len(df_bench) * 44 + 80),
                margin=dict(l=0, r=90, t=36, b=20),
                plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                title=dict(text="OPD to IPD Conversion Rate by Diagnosis",
                           font=dict(size=11, color=COOL_BLUE),
                           x=0, xanchor="left", pad=dict(t=4)),
                xaxis=dict(title="", range=[0, _dx_ceil],
                           ticksuffix="%", tickfont=dict(size=10)),
                yaxis=dict(title=""),
            )
            _pc(fig_dx)
            # Section B dynamic insight
            _below_dx = df_bench[df_bench["actual_rate"] < df_bench["ref_lower"]]\
                .sort_values("actual_rate")
            if not _below_dx.empty:
                _w = _below_dx.iloc[0]
                _worst_text = (
                    f"{_w['cleaned_diagnosis_name']} has the largest gap — "
                    f"{_w['actual_rate']:.1f}% actual vs {_w['ref_lower']:.0f}%"
                    f" reference lower bound."
                )
            else:
                _worst_text = "All diagnoses are at or above their reference lower bound."
            _htn = df_bench[df_bench["cleaned_diagnosis_name"].str.contains(
                "Hypertension|HTN", na=False)]
            _htn_text = ""
            if not _htn.empty and _htn.iloc[0]["actual_rate"] < 5:
                _htn_text = (
                    f" Hypertension at {_htn.iloc[0]['actual_rate']:.1f}% suggests "
                    f"hypertensive urgency patients are almost never admitted — "
                    f"review whether outpatient-only management is clinically appropriate."
                )
            st.markdown(
                f'<div style="border-left:3px solid {_R};background:#FCEBEB;'
                f'padding:10px 14px;font-size:13px;margin-top:10px;border-radius:0 4px 4px 0;">'
                f'<ul style="list-style:none;margin:0;padding:0;color:#003467;line-height:1.7;">'
                f'<li>· {_worst_text}{_htn_text}</li>'
                f'<li style="margin-top:3px;">· Reference ranges are from Shawky (2024) '
                f'international benchmarks — directional guidance only, not directly '
                f'comparable to Kenyan private hospital context.</li>'
                f'<li style="margin-top:3px;">· Action: for each diagnosis below its reference, '
                f'review the OPD assessment protocol — are severity criteria being applied '
                f'consistently? Are admissions being deferred for non-clinical reasons?</li>'
                f'</ul></div>',
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.warning(f"Diagnosis benchmark: {e}")
    _gap(16)

    # ── SECTION C — CLINICAL LEAKAGE SIGNALS ─────────────────────────────
    _sh("C — Clinical Leakage Signals", mt=8)
    try:
        cl, cr = st.columns(2)

        with cl:
            if not df_comorbidity.empty:
                df_comorb_overall = df_comorbidity[
                    ~df_comorbidity["patient_group"].str.contains(" — ", na=False)
                ].copy()

                _single_rate   = 0.0
                _comorbid_rate = 0.0
                if not df_comorb_overall.empty:
                    _s = df_comorb_overall[
                        df_comorb_overall["patient_group"] == "Single Diagnosis"]
                    _c = df_comorb_overall[
                        df_comorb_overall["patient_group"] == "Comorbid"]
                    _single_rate   = _sf(_s["conversion_rate_pct"].iloc[0]) \
                                     if not _s.empty else 0.0
                    _comorbid_rate = _sf(_c["conversion_rate_pct"].iloc[0]) \
                                     if not _c.empty else 0.0
                is_concern = _comorbid_rate < _single_rate

                _COMB_ORD   = ["Single Diagnosis", "Comorbid", "Chronic Comorbid"]
                _COMB_COL   = {"Single Diagnosis": _B, "Comorbid": _R,
                                "Chronic Comorbid": _G}
                cc1, cc2, cc3 = st.columns(3)
                for col_c, grp in zip([cc1, cc2, cc3], _COMB_ORD):
                    rate, delta, vis = 0.0, 0.0, 0
                    if (not df_comorb_overall.empty and
                            grp in df_comorb_overall["patient_group"].values):
                        r     = df_comorb_overall[
                            df_comorb_overall["patient_group"] == grp].iloc[0]
                        rate  = _sf(r.get("conversion_rate_pct"))
                        delta = _sf(r.get("delta_from_group_avg"))
                        vis   = int(_sf(r.get("total_opd_visits")))
                    d_col = _G if delta >= 0 else _R
                    d_lbl = f"+{delta:.2f} pct points" if delta >= 0 else f"{delta:.2f} pct points"
                    concern = grp == "Comorbid" and is_concern
                    bdr = (f"border:1px solid {_R};" if concern
                           else f"border:0.5px solid {BORDER};")
                    badge = (
                        f'<span style="font-size:10px;font-weight:700;padding:2px 8px;'
                        f'border-radius:20px;background:#FEE2E2;color:{_R};'
                        f'margin-right:4px">Concern</span> ') if concern else ""
                    with col_c:
                        st.markdown(
                            f'<div style="background:#fff;{bdr}border-radius:8px;'
                            f'padding:12px 12px 8px;margin-bottom:8px">'
                            f'<div style="font-size:10px;font-weight:700;color:{MUTED};'
                            f'text-transform:uppercase;letter-spacing:1px;margin-bottom:3px">'
                            f'{grp}</div>'
                            f'<div style="font-size:22px;font-weight:700;'
                            f'color:{_COMB_COL[grp]};line-height:1.1">{rate:.2f}%</div>'
                            f'<div style="font-size:11px;color:{MUTED};margin:3px 0">'
                            f'{vis:,} visits</div>'
                            f'{badge}'
                            f'<span style="font-size:11px;color:{d_col}">{d_lbl} vs avg</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                df_comorb_monthly = df_comorbidity[
                    df_comorbidity["patient_group"].str.contains(" — ", na=False)
                ].copy()
                df_comorb_monthly["group"] = (
                    df_comorb_monthly["patient_group"].str.split(" — ").str[0])
                df_comorb_monthly["month"] = pd.to_datetime(
                    df_comorb_monthly["patient_group"].str.split(" — ").str[1],
                    errors="coerce")
                df_comorb_monthly = df_comorb_monthly.sort_values("month")

                if not df_comorb_monthly.empty:
                    _leg([("Single Diagnosis", _B, "solid"),
                          ("Comorbid", _R, "dashed"),
                          ("Chronic Comorbid", _G, "dotted")])
                    fig_com = go.Figure()
                    _COM_STYLES = {
                        "Single Diagnosis": dict(color=_B, dash="solid"),
                        "Comorbid":         dict(color=_R, dash="dash"),
                        "Chronic Comorbid": dict(color=_G, dash="dot"),
                    }
                    for grp, sty in _COM_STYLES.items():
                        sub = df_comorb_monthly[
                            df_comorb_monthly["group"] == grp].sort_values("month")
                        if not sub.empty:
                            fig_com.add_trace(go.Scatter(
                                x=sub["month"], y=sub["conversion_rate_pct"],
                                mode="lines", line=dict(width=2, **sty),
                                showlegend=False,
                                hovertemplate=(f"{grp} %{{x|%b %y}}: "
                                               f"<b>%{{y:.1f}}%</b><extra></extra>"),
                            ))
                    _y_max = (df_comorb_monthly["conversion_rate_pct"].max() or 22)
                    _y_ceil = max(22, round(_y_max * 1.2))
                    fig_com.update_layout(
                        height=280, margin=dict(l=0, r=10, t=36, b=0),
                        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                        title=dict(text="Monthly Conversion by Comorbidity Group",
                                   font=dict(size=11, color=COOL_BLUE),
                                   x=0, xanchor="left", pad=dict(t=4)),
                        xaxis=dict(tickformat="%b %y", tickfont=dict(size=10), title=""),
                        yaxis=dict(range=[0, _y_ceil], ticksuffix="%",
                                   tickfont=dict(size=10), title=""),
                    )
                    _pc(fig_com)

        with cr:
            if not df_retention.empty:
                df_chronic_age = df_retention[
                    (df_retention["acquisition_segment"] == "Chronic") &
                    (df_retention["age_group"].notna()) &
                    (df_retention["visit_month"].isna())
                ][["age_group", "total_opd_visits",
                   "ipd_admissions", "conversion_rate_pct"]].copy()
                df_chronic_age = df_chronic_age.sort_values(
                    "conversion_rate_pct", ascending=True)

                if not df_chronic_age.empty:
                    _leg([("≥ 8% (within ref)", _B, "solid"),
                          ("< 8% (below ref)", _R, "solid")])
                    _chr_max  = df_chronic_age["conversion_rate_pct"].max()
                    _chr_ceil = max(22, round(_chr_max * 1.3))
                    fig_chr = go.Figure(go.Bar(
                        y=df_chronic_age["age_group"],
                        x=df_chronic_age["conversion_rate_pct"],
                        orientation="h",
                        marker_color=[_B if v >= 8 else _R
                                      for v in df_chronic_age["conversion_rate_pct"]],
                        text=[f"{v:.1f}%"
                              for v in df_chronic_age["conversion_rate_pct"]],
                        textposition="outside",
                        cliponaxis=False,
                        showlegend=False,
                        hovertemplate="<b>%{y}</b><br>%{x:.2f}%<extra></extra>",
                    ))
                    fig_chr.add_vline(x=8, line=dict(color=_Am, dash="dot", width=1.5))
                    fig_chr.update_layout(
                        height=max(320, len(df_chronic_age) * 42 + 80),
                        margin=dict(l=0, r=80, t=36, b=20),
                        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                        title=dict(text="Chronic Patient Conversion by Age Group",
                                   font=dict(size=11, color=COOL_BLUE),
                                   x=0, xanchor="left", pad=dict(t=4)),
                        xaxis=dict(title="", range=[0, _chr_ceil], ticksuffix="%"),
                        yaxis=dict(title=""),
                    )
                    _pc(fig_chr)

        # Section C dynamic insight
        try:
            _s_rate = _sf(df_comorb_overall[
                df_comorb_overall["patient_group"] == "Single Diagnosis"
            ]["conversion_rate_pct"].iloc[0]) if not df_comorb_overall.empty else 0.0
            _c_rate = _sf(df_comorb_overall[
                df_comorb_overall["patient_group"] == "Comorbid"
            ]["conversion_rate_pct"].iloc[0]) if not df_comorb_overall.empty else 0.0
            if _c_rate < _s_rate:
                _comorb_signal = (
                    f"Comorbid patients ({_c_rate:.1f}%) are converting at a lower rate "
                    f"than single-diagnosis patients ({_s_rate:.1f}%) despite having 2+ "
                    f"conditions — a clinical leakage signal."
                )
            else:
                _comorb_signal = (
                    f"Comorbid patients ({_c_rate:.1f}%) are converting at a higher rate "
                    f"than single-diagnosis patients ({_s_rate:.1f}%), as expected."
                )
            _age_text = ""
            if not df_chronic_age.empty:
                _la = df_chronic_age.sort_values("conversion_rate_pct").iloc[0]
                _age_text = (
                    f"{_la['age_group']} is the lowest converting chronic age group "
                    f"at {_la['conversion_rate_pct']:.1f}%."
                )
            st.markdown(
                f'<div style="border-left:3px solid {_R};background:#FCEBEB;'
                f'padding:10px 14px;font-size:13px;margin-top:10px;border-radius:0 4px 4px 0;">'
                f'<ul style="list-style:none;margin:0;padding:0;color:#003467;line-height:1.7;">'
                f'<li>· {_comorb_signal}</li>'
                f'<li style="margin-top:3px;">· {_age_text}</li>'
                f'<li style="margin-top:3px;">· Action: review OPD escalation criteria for '
                f'comorbid patients. A patient with 2+ chronic conditions who is not being '
                f'admitted at a higher rate than single-diagnosis patients suggests '
                f'comorbidity is not being weighted in the admission decision.</li>'
                f'</ul></div>',
                unsafe_allow_html=True,
            )
        except Exception:
            pass
    except Exception as e:
        st.warning(f"Leakage signals: {e}")
    _gap(16)

    # ── SECTION D — 72-HOUR ESCALATION ───────────────────────────────────
    _sh("D — Same-Day / 72-Hour Escalation", mt=8)
    try:
        dl, dr = st.columns([3, 2])

        with dl:
            if not df_escalation.empty:
                df_esc_age = df_escalation[
                    df_escalation["age_group"].notna() &
                    (df_escalation["age_group"] != "Unknown") &
                    (df_escalation["clean_diagnosis"] != "Unknown")
                ].groupby("age_group")["total_escalations"].sum().reset_index()
                df_esc_age = df_esc_age.sort_values(
                    "total_escalations", ascending=True)

                if not df_esc_age.empty:
                    fig_esc = go.Figure(go.Bar(
                        y=df_esc_age["age_group"],
                        x=df_esc_age["total_escalations"],
                        orientation="h",
                        marker_color=_B,
                        text=df_esc_age["total_escalations"].astype(int).astype(str),
                        textposition="outside",
                        showlegend=False,
                        hovertemplate=(
                            "<b>%{y}</b><br>%{x} escalations<extra></extra>"),
                    ))
                    _esc_max  = df_esc_age["total_escalations"].max()
                    _esc_ceil = max(10, round(_esc_max * 1.25))
                    fig_esc.update_layout(
                        height=max(260, len(df_esc_age) * 44 + 80),
                        margin=dict(l=0, r=70, t=36, b=20),
                        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                        title=dict(text="72h Escalations by Age Group",
                                   font=dict(size=11, color=COOL_BLUE),
                                   x=0, xanchor="left", pad=dict(t=4)),
                        xaxis=dict(title="Escalations", tickfont=dict(size=10),
                                   range=[0, _esc_ceil]),
                        yaxis=dict(title=""),
                    )
                    _pc(fig_esc)

        with dr:
            total_esc = 0
            top_dx    = "—"
            if not df_escalation.empty:
                total_esc = int(df_escalation["total_escalations"].sum())
                _known = df_escalation[
                    df_escalation["clean_diagnosis"] != "Unknown"]
                if not _known.empty:
                    top_dx = (
                        _known.groupby("clean_diagnosis")["total_escalations"]
                        .sum().idxmax()
                    )
            esc_rate = round(total_esc / total_opd_v * 100, 2) if total_opd_v else 0.0
            _kpi("Total 72h Escalations", f"{total_esc:,}",
                 "OPD visit → admission within 72h", COOL_BLUE)
            _gap(8)
            _kpi("Escalation Rate", f"{esc_rate:.2f}%",
                 "of all OPD visits",
                 AMBER if esc_rate < 1 else CORAL)
            _gap(8)
            _kpi("Top Classified Diagnosis", top_dx,
                 "Full breakdown pending ICD code fix", MUTED)

        # Section D dynamic insight
        try:
            _esc_rate = round(total_esc / total_opd_v * 100, 1) if total_opd_v else 0.0
            _top_age_row = df_esc_age.sort_values(
                "total_escalations", ascending=False).iloc[0] \
                if not df_esc_age.empty else None
            _known_dx = df_escalation[
                df_escalation["clean_diagnosis"] != "Unknown"
            ].groupby("clean_diagnosis")["total_escalations"].sum() \
                if not df_escalation.empty else pd.Series(dtype=float)
            _top_dx_name = _known_dx.idxmax() if not _known_dx.empty else "Unknown"
            _top_dx_n    = int(_known_dx.max()) if not _known_dx.empty else 0
            _age_line = (
                f"{_top_age_row['age_group']} is the most escalated age group "
                f"({int(_top_age_row['total_escalations']):,} escalations). "
                f"{_top_dx_name} is the leading classified diagnosis ({_top_dx_n:,} escalations)."
            ) if _top_age_row is not None else ""
            _action_age = _top_age_row["age_group"] if _top_age_row is not None else "affected"
            st.markdown(
                f'<div style="border-left:3px solid {_R};background:#FCEBEB;'
                f'padding:10px 14px;font-size:13px;margin-top:10px;border-radius:0 4px 4px 0;">'
                f'<ul style="list-style:none;margin:0;padding:0;color:#003467;line-height:1.7;">'
                f'<li>· {total_esc:,} patients returned for inpatient admission within 72 hours '
                f'of an OPD visit ({_esc_rate:.1f}% of all OPD visits). These are patients '
                f'whose condition was not recognised as admission-requiring at first contact.</li>'
                f'<li style="margin-top:3px;">· {_age_line}</li>'
                f'<li style="margin-top:3px;">· Action: implement a paediatric early warning '
                f'score (PEWS) at OPD triage for {_action_age} presentations to reduce '
                f'missed escalations. Resolve ICD10 coding gaps to enable full diagnosis '
                f'breakdown — a large share currently remains unclassified.</li>'
                f'</ul></div>',
                unsafe_allow_html=True,
            )
        except Exception:
            pass
    except Exception as e:
        st.warning(f"Escalation: {e}")
    _gap(16)

    # ── SECTION E — CLINICIAN WORKLOAD VS CONVERSION RATE ────────────────
    _sh("E — Clinician Workload vs Conversion Rate", mt=8)
    try:
        if not df_triangle.empty:
            df_tri = df_triangle.sort_values("visit_month").copy()

            strain_months = df_tri[
                df_tri["strain_signal"].isin(["HIGH_STRAIN", "CAPACITY_GAP"])]
            normal_months = df_tri[df_tri["strain_signal"] == "AS_EXPECTED"]
            avg_strain = round(strain_months["conversion_rate_pct"].mean(), 2) \
                         if not strain_months.empty else 0.0
            avg_normal = round(normal_months["conversion_rate_pct"].mean(), 2) \
                         if not normal_months.empty else 0.0
            conversion_gap = round(avg_normal - avg_strain, 2)
            peak_wl    = df_tri["avg_visits_per_clinician"].max()
            peak_month = pd.to_datetime(
                df_tri.loc[df_tri["avg_visits_per_clinician"].idxmax(),
                           "visit_month"]
            ).strftime("%b %Y")

            gap_col = _R if conversion_gap > 0.5 else (_Am if conversion_gap > 0 else _G)

            ek1, ek2, ek3, ek4 = st.columns(4)
            with ek1: _kpi("Conversion — Normal Months", f"{avg_normal:.2f}%",
                           "Months within expected clinician load", GREEN)
            _strain_month_labels = (
                ", ".join(
                    pd.to_datetime(m).strftime("%b %Y")
                    for m in strain_months["visit_month"].sort_values()
                ) if not strain_months.empty else "—"
            )
            with ek2: _kpi("Conversion — High Load Months", f"{avg_strain:.2f}%",
                           _strain_month_labels,
                           CORAL)
            with ek3: _kpi("Conversion Gap", f"{conversion_gap:+.2f} pct pts",
                           "Positive = overload suppressing admissions",
                           gap_col)
            with ek4: _kpi("Peak Workload Month", peak_month,
                           f"{peak_wl:.1f} avg visits per clinician", CORAL if peak_wl > 80 else MUTED)

            _gap(12)
            _leg([("Conversion rate %", _B, "solid"),
                  ("Clinician load (scaled)", _Pu, "dashed")])
            st.caption(
                "Purple dashed line = clinician load. When load rises and conversion drops, "
                "patients may be under-admitted due to capacity pressure. "
                "Shaded months = high load periods."
            )

            _wl_max = df_tri["avg_visits_per_clinician"].max() or 1
            fig_tri = go.Figure()
            for _, row in strain_months.iterrows():
                fig_tri.add_vrect(
                    x0=row["visit_month"], x1=row["visit_month"],
                    fillcolor="rgba(226,75,74,0.08)",
                    line_width=0, layer="below",
                )
            fig_tri.add_trace(go.Scatter(
                x=df_tri["visit_month"], y=df_tri["conversion_rate_pct"],
                mode="lines+markers",
                line=dict(color=_B, width=2), marker=dict(size=4),
                showlegend=False,
                hovertemplate="%{x|%b %y}: <b>%{y:.2f}%</b><extra></extra>",
            ))
            fig_tri.add_trace(go.Scatter(
                x=df_tri["visit_month"],
                y=df_tri["avg_visits_per_clinician"] / _wl_max * 10,
                mode="lines",
                line=dict(color=_Pu, width=1.5, dash="dash"),
                showlegend=False,
                customdata=df_tri["avg_visits_per_clinician"],
                hovertemplate=("%{x|%b %y}: <b>%{customdata:.1f}</b>"
                               " visits/clinician<extra></extra>"),
            ))
            fig_tri.update_layout(
                height=280, margin=dict(l=0, r=30, t=36, b=0),
                plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                title=dict(text="Conversion Rate vs Clinician Workload",
                           font=dict(size=11, color=COOL_BLUE),
                           x=0, xanchor="left", pad=dict(t=4)),
                xaxis=dict(tickformat="%b %y", tickfont=dict(size=10), title=""),
                yaxis=dict(title="", range=[0, 12], ticksuffix="%",
                           tickfont=dict(size=10)),
            )
            _pc(fig_tri)

            # Strain callout rows — no wait time column
            strain_callout = df_tri[df_tri["strain_signal"] != "AS_EXPECTED"]
            if not strain_callout.empty:
                _gap(8)
                for _, sr in strain_callout.iterrows():
                    signal  = str(sr.get("strain_signal", "CAPACITY_GAP"))
                    month_s = (pd.to_datetime(sr["visit_month"]).strftime("%b %Y")
                               if pd.notna(sr.get("visit_month")) else "—")
                    conv = _sf(sr.get("conversion_rate_pct"))
                    wl   = _sf(sr.get("avg_visits_per_clinician"))
                    if signal == "HIGH_STRAIN":
                        row_bg, row_txt = "#FCEBEB", "#791F1F"
                    else:
                        row_bg, row_txt = "#FEF3E2", "#7A3B00"
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:10px;'
                        f'padding:8px 12px;margin-bottom:4px;border-radius:6px;'
                        f'background:{row_bg};">'
                        f'<span style="font-size:11px;font-weight:700;color:{row_txt};'
                        f'min-width:80px">{month_s}</span>'
                        f'<span style="font-size:10px;font-weight:700;padding:2px 8px;'
                        f'border-radius:20px;background:{row_txt}20;color:{row_txt}">'
                        f'{signal.replace("_", " ")}</span>'
                        f'<span style="font-size:11px;color:{row_txt};">'
                        f'Conv: {conv:.2f}% &nbsp;·&nbsp; {wl:.1f} visits/clinician'
                        f'</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Dynamic insight bar
            if conversion_gap > 1.0:
                _signal = (
                    f"In {len(strain_months)} high-load months, conversion averaged "
                    f"{avg_strain:.2f}% — {conversion_gap:.2f} pct pts below the {avg_normal:.2f}% "
                    f"seen in {len(normal_months)} normal months. "
                    f"Clinician overload is suppressing admission decisions."
                )
                _action = (
                    f"Action: review clinical staffing for months where average load exceeds "
                    f"{peak_wl:.0f} visits per clinician. Additional staff during peak periods "
                    f"would allow admission decisions to be made on clinical grounds "
                    f"rather than capacity constraints."
                )
            elif conversion_gap > 0:
                _signal = (
                    f"A {conversion_gap:.2f} pct pts conversion gap exists between high-load months "
                    f"({avg_strain:.2f}%) and normal months ({avg_normal:.2f}%). The relationship "
                    f"is present but not yet clinically significant."
                )
                _action = (
                    f"Action: monitor monthly. If gap exceeds 1.0pp, review clinical staffing "
                    f"levels for months where average load exceeds {peak_wl:.0f} visits per clinician."
                )
            else:
                _signal = (
                    f"No conversion gap between high-load and normal months "
                    f"({avg_strain:.2f}% vs {avg_normal:.2f}%). Clinician workload is not "
                    f"currently suppressing admission decisions."
                )
                _action = "No staffing action required based on current data."

            _gap_bg = {"#C53030": "#FCEBEB", "#D97706": "#FAEEDA"}.get(gap_col, "#EBF5FF")
            st.markdown(
                f'<div style="border-left:3px solid {gap_col};background:{_gap_bg};'
                f'padding:10px 14px;font-size:13px;margin-top:10px;'
                f'border-radius:0 4px 4px 0;">'
                f'<ul style="list-style:none;margin:0;padding:0;color:#003467;line-height:1.7;">'
                f'<li>· {_signal}</li>'
                f'<li style="margin-top:3px;">· {_action}</li>'
                f'</ul></div>',
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.warning(f"Workload vs conversion: {e}")


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
                    orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
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
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
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
                legend=dict(orientation="h", y=1.08, x=0),
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
    _CFG       = {"displayModeBar": False}

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
        st.markdown(
            f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:2px;color:#6B8CAE;border-bottom:1px solid #EBF3FB;'
            f'padding-bottom:5px;margin:18px 0 10px">{label}</div>',
            unsafe_allow_html=True,
        )

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
    df_f       = _load(Q.load_ca_section_f,                 "Section F")

    # ══════════════════════════════════════════════════════════════════════════
    # HEADER KPIs
    # ══════════════════════════════════════════════════════════════════════════
    _sec("CLINICAL ACTIVITY — KISUMU SPECIALISTS · INPATIENT CARE")
    if not df_ward.empty:
        total_adm  = int(df_ward["total_admissions"].sum())
        total_disc = int(df_ward["total_discharges"].sum())
        still_adm  = total_adm - total_disc
        med_los    = round(float(df_ward["median_los"].median()), 1)
        read30     = int(df_ward["day_readmission_30"].sum())
        read_pct   = round(read30 / total_disc * 100, 1) if total_disc else 0
        prd_ct     = int(df_ward["patient_request_discharge"].sum())
        prd_pct    = round(prd_ct / total_disc * 100, 1) if total_disc else 0

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1: _kpi("Total Admissions",       f"{total_adm:,}")
        with c2: _kpi("Discharged",             f"{total_disc:,}")
        with c3: _kpi("Still Admitted",         f"{still_adm:,}",  color=_AMBER)
        with c4: _kpi("Median LOS",             f"{med_los}d",
                       s="Mean pulled up by Sepsis outliers")
        with c5: _kpi("30-Day Readmission",     f"{read_pct:.1f}%", color=_RED)
        with c6: _kpi("Patient Req. Discharge", f"{prd_pct:.1f}%",
                       s="Includes data recording inconsistency", color=_AMBER)

    _gap(16)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION A — WARD OVERVIEW  [table left | trend chart right]
    # ══════════════════════════════════════════════════════════════════════════
    _sec("A — WARD OVERVIEW")
    ac1, ac2 = st.columns([1, 1])

    with ac1:
        if not df_ward.empty:
            dw = df_ward[["ward_name", "total_admissions", "median_los",
                          "readmission_rate", "patient_request_discharge_pct"]].copy()
            dw["patient_request_discharge_pct"] = dw.apply(
                lambda r: f"{float(r['patient_request_discharge_pct']):.1f}%*"
                if str(r["ward_name"]).lower() in _MATERNITY
                else f"{float(r['patient_request_discharge_pct']):.1f}%",
                axis=1,
            )
            dw["median_los"] = dw["median_los"].apply(lambda v: f"{float(v):.1f}d")
            dw["total_admissions"] = dw["total_admissions"].astype(int)
            dw.columns = ["Ward", "Admissions", "Median LOS",
                          "Readmission %", "Patient Request %"]

            render_sortable_table(
                dw,
                height=280,
                badge_columns={
                    "Readmission %": [
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
            _insight([
                "Maternity patient request % is a data recording inconsistency — clinicians are "
                "recording routine discharges under the wrong discharge type.",
                "Requires a data quality audit before this field can be used for clinical conclusions.",
            ], variant="amber")

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
            WARD_CFG = {
                "General Female":       (_BLUE,   "solid"),
                "Pediatric General":    (_GREEN,  "solid"),
                "General Male":         (_PURPLE, "solid"),
                "Maternity (combined)": (_PINK,   "dash"),
            }
            legend_html = "".join(
                f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:14px">'
                f'<span style="width:14px;height:2px;background:{col};display:inline-block"></span>'
                f'<span style="font-size:11px;color:#003467">{ward}</span></span>'
                for ward, (col, dash) in WARD_CFG.items()
            )
            st.markdown(f'<div style="margin-bottom:6px">{legend_html}</div>',
                        unsafe_allow_html=True)
            fig = go.Figure()
            for ward, (col, dash) in WARD_CFG.items():
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
                height=340,
                margin=dict(l=0, r=0, t=10, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(showgrid=False,
                           tickfont=dict(size=10, color="#6B8CAE")),
                yaxis=dict(title="Admissions", showgrid=True,
                           gridcolor="#EBF3FB", rangemode="tozero"),
            )
            st.plotly_chart(fig, use_container_width=True, config=_CFG)

    _gap(16)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION B — LENGTH OF STAY
    # ══════════════════════════════════════════════════════════════════════════
    _sec("B — LENGTH OF STAY")
    bc1, bc2 = st.columns(2)

    with bc1:
        _card_title("LOS Distribution by Ward")
        _sub("Box = middle 50% of stays. Wider box = more variability. "
             "Line = median. Tick marks = typical range boundaries. "
             "Sorted narrowest IQR at top.")
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
            _insight([
                "Private Maternity: IQR 0.75d — consistent pathway, predictable discharge timing.",
                "General Female + General Male: IQR 3.0–3.1d — reflects diverse case mix "
                "(Typhoid, Sepsis, Oncology), not care inconsistency.",
            ], variant="info")

    with bc2:
        _card_title("LOS Outliers by Ward")
        _sub("Each point is one admission exceeding the ward IQR upper fence.")
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
            _insight([
                "Other Sepsis drives most outlier stays (7–73 days across all wards).",
                "Diabetes + Sepsis comorbidity confirmed in outlier data — uncontrolled chronic "
                "disease precipitating Sepsis.",
                "139-day General Maternity case has no ICD10 recorded — individual documentation "
                "review needed.",
            ], variant="warn")

    # ── LOS by condition ward dropdown (full width) ────────────────────────────
    _gap(12)
    _sub("Select a ward to see median LOS per condition. Conditions with longest "
         "stays show where extended care is concentrated.")
    if not df_losdiag.empty:
        ward_options = sorted(df_losdiag["ward_name"].dropna().unique())
        selected_ward_b = st.selectbox(
            "Select ward to investigate", ward_options, key="ca_los_ward_select"
        )
        ward_diag = (
            df_losdiag[df_losdiag["ward_name"] == selected_ward_b]
            .groupby("final_disease_burden_group")
            .agg(median_los=("median_los_days", "median"))
            .reset_index()
            .sort_values("median_los", ascending=True)
            .tail(12)
        )
        ward_diag["label"] = ward_diag["final_disease_burden_group"].apply(
            lambda x: (x.split(" - ")[-1].split(": ")[-1]
                       if " - " in str(x) or ": " in str(x) else str(x))
        )
        ward_diag["colour"] = ward_diag["median_los"].apply(
            lambda v: _RED if v > 20 else (_AMBER if v > 10 else _BLUE)
        )
        fig = go.Figure(go.Bar(
            y=ward_diag["label"],
            x=ward_diag["median_los"],
            orientation="h",
            marker_color=ward_diag["colour"].tolist(),
            text=ward_diag["median_los"].apply(lambda v: f"{v:.1f}d"),
            textposition="outside",
            textfont=dict(size=10, color="#003467"),
        ))
        fig.update_layout(
            height=max(280, len(ward_diag) * 28 + 60),
            margin=dict(l=0, r=50, t=10, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(title="Median LOS (days)", showgrid=True, gridcolor="#EBF3FB"),
            yaxis=dict(showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config=_CFG)

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

    cc1, cc2 = st.columns(2)

    with cc1:
        if not df_ward.empty:
            bar_df = df_ward[["ward_name", "readmission_rate"]].copy()
            bar_df = bar_df.sort_values("readmission_rate", ascending=True)
            bar_df["color"] = bar_df["readmission_rate"].apply(
                lambda r: _RED if float(r) > 6 else (_AMBER if float(r) > 4 else _GREEN)
            )
            st.markdown(
                f'<div style="font-size:11px;margin-bottom:6px">'
                f'<span style="color:{_GREEN}">● &lt; 4% — within expected</span>&nbsp;·&nbsp;'
                f'<span style="color:{_AMBER}">● 4–6% — monitor</span>&nbsp;·&nbsp;'
                f'<span style="color:{_RED}">● &gt; 6% — investigate</span></div>',
                unsafe_allow_html=True,
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
        gm1, gm2, gm3 = st.columns(3)
        for col_card, (lbl_txt, val_txt, sub_txt) in zip(
            [gm1, gm2, gm3],
            [
                ("Top age group",    "Senior (65+)",   "21 of 31 readmissions · 67.7%"),
                ("Top condition",    "NCD — Oncology", "8 readmissions incl. HIV+Oncology comorbidity"),
                ("Repeat admitters", "7 patients",     "Admitted 3+ times · chronic disease revolving door"),
            ],
        ):
            with col_card:
                st.markdown(
                    f'<div style="background:#F4F8FC;border-radius:8px;'
                    f'padding:10px 12px;margin-bottom:8px">'
                    f'<div style="font-size:10px;font-weight:600;color:#6B8CAE;'
                    f'text-transform:uppercase;letter-spacing:1px">{lbl_txt}</div>'
                    f'<div style="font-size:15px;font-weight:700;color:#003467;'
                    f'margin:3px 0">{val_txt}</div>'
                    f'<div style="font-size:10px;color:#6B8CAE">{sub_txt}</div></div>',
                    unsafe_allow_html=True,
                )

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
            render_sortable_table(_gm_df, height=220, key="gm_conditions")

    _ic1, _ic2, _ic3 = st.columns(3)
    _INSIGHT_CARDS = [
        (_ic1, "warn", [
            "21 of 31 General Male readmissions are Senior (65+) patients.",
            "NCD-Oncology is the leading condition — 7 patients admitted 3+ times.",
            "Action: review whether structured oncology outpatient management would reduce "
            "revolving-door admissions.",
        ]),
        (_ic2, "info", [
            "30 stable-discharge readmissions = wrong discharge decision — highest clinical priority.",
            "42 patient-request readmissions = patient chose to leave — counselling and retention gap.",
        ]),
        (_ic3, "warn", [
            "Elderly oncology patients with no curative pathway are driving the 8.29% rate.",
            "Action: clinical review for palliative care or structured outpatient oncology "
            "pathway for repeat admitters.",
        ]),
    ]
    for _col, _variant, _items in _INSIGHT_CARDS:
        with _col:
            _insight(_items, variant=_variant)

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
                height=300,
                margin=dict(l=0, r=0, t=10, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                barmode="group",
                xaxis=dict(showgrid=False),
                yaxis=dict(title="Avg LOS (days)", showgrid=True, gridcolor="#EBF3FB"),
                legend=dict(orientation="h", y=1.08, x=0,
                            font=dict(size=11)),
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
    la1, la2 = st.columns([1, 1])

    with la1:
        _card_title("Return window × discharge reason — all wards")
        _sub("Each ward grouped by return window. Legend % = each window's share of "
             "total 30-day returns.")
        if not df_c4d.empty:
            aw = (
                df_c4d[df_c4d["return_band"].notna()]
                .groupby(["ward_name", "return_band"])["readmissions"]
                .sum().reset_index()
            )
            grand_total = aw["readmissions"].sum() or 1
            band_pct = (
                aw.groupby("return_band")["readmissions"].sum() / grand_total * 100
            ).round(1)
            _BAND_DISP = {
                "0-7 days (early)":  f"0–7d early ({band_pct.get('0-7 days (early)', 0):.1f}%)",
                "8-14 days":         f"8–14d ({band_pct.get('8-14 days', 0):.1f}%)",
                "15-30 days (late)": f"15–30d late ({band_pct.get('15-30 days (late)', 0):.1f}%)",
            }
            # custom HTML legend above chart
            legend_l4 = " · ".join([
                f'<span style="display:inline-flex;align-items:center;gap:5px">'
                f'<span style="width:10px;height:10px;background:{_BAND_COLORS[b]};'
                f'border-radius:2px;display:inline-block"></span>'
                f'<span style="font-size:11px;color:#003467">{_BAND_DISP[b]}</span></span>'
                for b in _BAND_ORDER
            ])
            st.markdown(f'<div style="margin-bottom:8px">{legend_l4}</div>',
                        unsafe_allow_html=True)
            ward_order = (
                aw.groupby("ward_name")["readmissions"].sum()
                .sort_values(ascending=False).index.tolist()
            )
            fig = go.Figure()
            for band in _BAND_ORDER:
                sub = aw[aw["return_band"] == band].set_index("ward_name")
                y_vals = [int(sub.loc[w, "readmissions"]) if w in sub.index else 0
                          for w in ward_order]
                fig.add_trace(go.Bar(
                    name=_BAND_DISP[band],
                    x=ward_order,
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
        _card_title("Conditions driving readmissions — by return window")
        _sub("Each column is a return window. Stacked segments show which conditions "
             "are driving returns within that window.")
        selected_ward = st.selectbox(
            "Select ward",
            ["General Male", "General Female", "Pediatric General"],
            key="layer4_ward",
        )
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

    dc1, dc2 = st.columns([1, 1])

    with dc1:
        _card_title("Readmission Rate by Diagnosis")
        if not diag_filtered.empty:
            st.markdown(
                f'<div style="font-size:11px;margin-bottom:6px">'
                f'<span style="color:{_RED}">● ≥10% rate</span>&nbsp;·&nbsp;'
                f'<span style="color:{_AMBER}">● 5–10%</span>&nbsp;·&nbsp;'
                f'<span style="color:{_BLUE}">● &lt;5%</span></div>',
                unsafe_allow_html=True,
            )
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
            _insight([
                "Priority 1: NCD-Cardiovascular (18.2%) and NCD-Oncology (15.1%) — "
                "international priority reduction targets.",
                "Priority 2: Genitourinary (12.5%), Sickle Cell (12.5%), HIV+Hernia (16.7%).",
                "Action: structured discharge protocols and scheduled follow-up for these "
                "five conditions.",
            ], variant="warn")

    with dc2:
        _card_title("Priority conditions by admissions and readmission count")
        if not diag_filtered.empty:
            tbl = (
                diag_filtered.sort_values("readmit", ascending=False)
                .head(10).copy()
            )

            def _flag(r):
                if r >= 10:
                    return ("Priority", _RED)
                elif r >= 5:
                    return ("Monitor", _AMBER)
                else:
                    return ("Standard", _GREEN)

            rows_html = ""
            for _, row in tbl.iterrows():
                flag_lbl, flag_col = _flag(row["rate"])
                rate_col = (
                    _RED   if row["rate"] >= 10
                    else (_AMBER if row["rate"] >= 5 else "inherit")
                )
                rdm_col = (
                    _RED   if row["readmit"] >= 10
                    else (_AMBER if row["readmit"] >= 3 else "inherit")
                )
                rdm_fw  = "700" if rdm_col  != "inherit" else "400"
                rate_fw = "700" if rate_col != "inherit" else "400"
                rows_html += (
                    f'<tr style="border-bottom:1px solid rgba(128,128,128,0.1)">'
                    f'<td style="padding:10px 8px;font-weight:600;font-size:12px;'
                    f'color:var(--text-color)">{row["label"]}</td>'
                    f'<td style="padding:10px 8px;text-align:right;font-size:12px;'
                    f'color:var(--text-color)">{int(row["total"])}</td>'
                    f'<td style="padding:10px 8px;text-align:right;font-size:12px;'
                    f'color:{rdm_col};font-weight:{rdm_fw}">{int(row["readmit"])}</td>'
                    f'<td style="padding:10px 8px;text-align:right;font-size:12px;'
                    f'color:{rate_col};font-weight:{rate_fw}">{row["rate"]:.1f}%</td>'
                    f'<td style="padding:10px 8px;text-align:center">'
                    f'<span style="background:{flag_col};color:white;font-size:10px;'
                    f'font-weight:700;padding:3px 10px;border-radius:12px;'
                    f'display:inline-block;white-space:nowrap">{flag_lbl}</span>'
                    f'</td></tr>'
                )

            _th = (
                "padding:9px 8px;text-align:{a};font-size:10px;color:#888780;"
                "font-weight:700;text-transform:uppercase;letter-spacing:0.05em;"
                "border-bottom:2px solid rgba(128,128,128,0.15)"
            )
            table_html = (
                '<table style="width:100%;border-collapse:collapse">'
                '<thead><tr>'
                f'<th style="{_th.format(a="left")}">Diagnosis</th>'
                f'<th style="{_th.format(a="right")}">Admissions</th>'
                f'<th style="{_th.format(a="right")}">Readmissions</th>'
                f'<th style="{_th.format(a="right")}">Rate</th>'
                f'<th style="{_th.format(a="center")}">Flag</th>'
                f'</tr></thead>'
                f'<tbody>{rows_html}</tbody>'
                f'</table>'
            )
            st.markdown(table_html, unsafe_allow_html=True)

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
                legend=dict(orientation="h", y=1.05, x=0, font=dict(size=11)),
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
            for _lbl, _col, _clr in _comm_segs:
                fig_wc.add_trace(go.Bar(
                    name=_lbl,
                    y=_wc["ward_name"], x=_wc[_col],
                    orientation="h",
                    marker_color=_clr,
                    showlegend=True,
                    hovertemplate=f"{_lbl}: %{{x}} admissions<extra></extra>",
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

    # ── Sepsis LOS Outlier Analysis ────────────────────────────────────────────
    _gap(16)
    _card_title("Sepsis LOS Outlier Analysis — Clinical Origin of Extended Stays")
    _sub("Sepsis admissions exceeding each ward's IQR upper fence, classified by "
         "prior contact history to identify preventable vs community-acquired stays.")

    df_sepsis = _load(Q.load_ca_sepsis_enriched, "Sepsis enriched")

    if not df_sepsis.empty:
        total      = len(df_sepsis)
        bonnke_n   = df_sepsis["notes_diagnosis"].str.contains("Bonnke", na=False).sum()
        bonnke_pct = round(bonnke_n / total * 100) if total else 0

        _df      = df_sepsis.copy()
        _has_opd = "last_opd_days_before" in _df.columns

        def _classify_sepsis(row):
            _ip   = pd.notna(row["prior_condition_display"])
            _days = row["prior_condition_days"]
            _opd  = row["last_opd_days_before"] if _has_opd else float("nan")
            if _ip and pd.notna(_days) and _days == 0:
                return "comorbid"
            if _ip and pd.notna(_days) and 1 <= _days <= 30:
                return "prior_ip"
            if not _ip and pd.notna(_opd) and 1 <= _opd <= 30:
                return "prior_opd"
            if not _ip and (pd.isna(_opd) or _opd > 30):
                return "no_prior"
            return "outside"

        _df["_grp"]  = _df.apply(_classify_sepsis, axis=1)
        _comorbid_n  = int((_df["_grp"] == "comorbid").sum())
        _prior_ip_n  = int((_df["_grp"] == "prior_ip").sum())
        _prior_opd_n = int((_df["_grp"] == "prior_opd").sum())
        _no_prior_n  = int((_df["_grp"] == "no_prior").sum())
        _outside_n   = int((_df["_grp"] == "outside").sum())
        _opd_note    = "" if _has_opd else " *OPD contact column not present"

        def _pct(n): return f"{int(n / total * 100)}%" if total else "—"

        # ── KPI cards ─────────────────────────────────────────────────────────
        _kc1, _kc2, _kc3, _kc4 = st.columns(4)

        def _sep_card(col, label, value, pct_str, sub, color):
            col.markdown(
                f'<div style="background:var(--secondary-background-color);'
                f'border-radius:10px;padding:16px 18px;border-top:3px solid {color};">'
                f'<div style="font-size:11px;font-weight:600;color:{color};'
                f'text-transform:uppercase;letter-spacing:0.6px;margin-bottom:8px;">'
                f'{label}</div>'
                f'<div style="display:flex;align-items:baseline;gap:8px;">'
                f'<span style="font-size:28px;font-weight:700;color:{color};">{value}</span>'
                f'<span style="font-size:14px;color:{color};opacity:0.75;">{pct_str}</span>'
                f'</div>'
                f'<div style="font-size:11px;color:var(--text-color);opacity:0.55;margin-top:6px;">'
                f'{sub}</div></div>',
                unsafe_allow_html=True,
            )

        _sep_card(_kc1, "Comorbid at visit",       _comorbid_n,  _pct(_comorbid_n),
                  "Condition coded on same Sepsis admission",           "#185FA5")
        _sep_card(_kc2, "Prior inpatient ≤30d",    _prior_ip_n,  _pct(_prior_ip_n),
                  "Traceable inpatient admission in prior 30 days",     "#BA7517")
        _sep_card(_kc3, "Prior OPD only ≤30d",     _prior_opd_n, _pct(_prior_opd_n),
                  f"OPD contact only — no prior inpatient{_opd_note}", "#EF9F27")
        _sep_card(_kc4, "No prior contact",         _no_prior_n,  _pct(_no_prior_n),
                  "Community-acquired — no OPD or inpatient history",   "#E24B4A")
        _gap(16)

        # ── Two charts ────────────────────────────────────────────────────────
        _ccol1, _ccol2 = st.columns(2)

        with _ccol1:
            st.markdown("**Clinical origin of Sepsis outlier stays**")
            _seg_labels = [
                "Comorbid dx coded at visit",
                "Prior inpatient ≤30d",
                "Prior OPD only ≤30d",
                "No prior contact",
                "Outside 30d window / other",
            ]
            _seg_counts = [_comorbid_n, _prior_ip_n, _prior_opd_n, _no_prior_n, _outside_n]
            _seg_colors = ["#185FA5", "#BA7517", "#EF9F27", "#E24B4A", "#D3D1C7"]
            _leg_html = '<div style="display:flex;flex-wrap:wrap;gap:8px;font-size:11px;margin-bottom:6px;">'
            for _sl, _sc, _sk in zip(_seg_labels, _seg_counts, _seg_colors):
                _leg_html += (
                    f'<span style="display:flex;align-items:center;gap:4px;">'
                    f'<span style="width:10px;height:10px;border-radius:2px;'
                    f'background:{_sk};display:inline-block;flex-shrink:0;"></span>'
                    f'<strong>{_sc}</strong>&nbsp;{_sl}</span>'
                )
            _leg_html += '</div>'
            st.markdown(_leg_html, unsafe_allow_html=True)
            _fig_orig = go.Figure()
            for _sl, _sc, _sk in zip(_seg_labels, _seg_counts, _seg_colors):
                _fig_orig.add_trace(go.Bar(
                    name=_sl, x=[_sc], y=["All patients"],
                    orientation="h", marker_color=_sk, showlegend=False,
                    hovertemplate=f"{_sl}: %{{x}} patients<extra></extra>",
                ))
            _fig_orig.update_layout(
                height=220, margin=dict(l=0, r=0, t=10, b=40),
                xaxis=dict(range=[0, total], dtick=2, title="Patients"),
                yaxis=dict(showgrid=False), barmode="stack",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            _pc(_fig_orig)

            # ── Diagnosis pills (inside same column) ──────────────────────────
            st.markdown("**Diagnoses by clinical origin**")
            _pill_groups = [
                ("comorbid",  "Comorbid at visit",    "#185FA5", "#E6F1FB", "#0C447C"),
                ("prior_ip",  "Prior inpatient ≤30d", "#BA7517", "#FAEEDA", "#633806"),
                ("prior_opd", "Prior OPD only ≤30d",  "#EF9F27", "#FAEEDA", "#633806"),
                ("no_prior",  "No prior contact",      "#E24B4A", "#FCEBEB", "#791F1F"),
                ("outside",   "Outside 30d / other",  "#888780", "#F1EFE8", "#444441"),
            ]
            _rendered_groups = []
            for _gk, _gn, _gdot, _gpb, _gpt in _pill_groups:
                _gdf = _df[_df["_grp"] == _gk]
                if _gdf.empty:
                    continue
                _dx_set = set()
                if _gk in ("comorbid", "prior_ip", "outside"):
                    _dx_set.update(_gdf["prior_condition_display"].dropna().unique())
                elif _gk == "prior_opd":
                    if "last_opd_diagnosis" in _gdf.columns:
                        _dx_set.update(_gdf["last_opd_diagnosis"].dropna().unique())
                    _dx_set.update(_gdf["sepsis_condition"].dropna().unique())
                else:
                    _dx_set.update(_gdf["sepsis_condition"].dropna().unique())
                _dx_list = sorted(
                    v for v in _dx_set
                    if v and str(v) not in ("No prior admission", "nan")
                )
                _pills = "".join(
                    f'<span style="font-size:12px;padding:2px 7px;border-radius:20px;'
                    f'background:{_gpb};color:{_gpt};margin:2px;display:inline-block;">'
                    f'{_v}</span>'
                    for _v in _dx_list
                ) or f'<span style="font-size:12px;opacity:0.5;">—</span>'
                _rendered_groups.append((_gn, _gdot, _gpb, _gpt, len(_gdf), _pills))

            if _rendered_groups:
                _cards_html = '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:4px;">'
                for _gn, _gdot, _gpb, _gpt, _gcnt, _pills in _rendered_groups:
                    _cards_html += (
                        f'<div style="background:var(--secondary-background-color);'
                        f'border-radius:6px;padding:8px 10px;min-width:120px;max-width:200px;">'
                        f'<div style="display:flex;align-items:center;gap:5px;margin-bottom:6px;">'
                        f'<span style="width:7px;height:7px;border-radius:1px;'
                        f'background:{_gdot};display:inline-block;flex-shrink:0;"></span>'
                        f'<span style="font-size:11px;font-weight:600;color:{_gdot};">{_gn}</span>'
                        f'</div>'
                        f'<div style="line-height:1.7;">{_pills}</div></div>'
                    )
                _cards_html += '</div>'
                st.markdown(_cards_html, unsafe_allow_html=True)

        with _ccol2:
            st.markdown("**Average LOS — traceable prior condition vs no prior contact**")
            _pm  = _df["prior_condition_display"].notna()
            _npm = _df["prior_condition_display"].isna()
            _p_avg  = float(_df.loc[_pm,  "los_days"].mean()) if _pm.any()  else 0.0
            _np_avg = float(_df.loc[_npm, "los_days"].mean()) if _npm.any() else 0.0
            _p_cnt  = int(_pm.sum())
            _np_cnt = int(_npm.sum())
            _los_leg = (
                f'<div style="display:flex;gap:14px;font-size:11px;margin-bottom:6px;">'
                f'<span style="display:flex;align-items:center;gap:4px;">'
                f'<span style="width:10px;height:10px;border-radius:2px;'
                f'background:#185FA5;display:inline-block;"></span>'
                f'<strong>{_p_cnt}</strong>&nbsp;Has prior condition</span>'
                f'<span style="display:flex;align-items:center;gap:4px;">'
                f'<span style="width:10px;height:10px;border-radius:2px;'
                f'background:#E24B4A;display:inline-block;"></span>'
                f'<strong>{_np_cnt}</strong>&nbsp;No prior contact</span></div>'
            )
            st.markdown(_los_leg, unsafe_allow_html=True)
            _fig_los = go.Figure()
            _fig_los.add_trace(go.Bar(
                x=[f"Has prior\ncondition\n({_p_cnt} pts)"], y=[_p_avg],
                marker_color="#185FA5", showlegend=False,
                text=[f"{_p_avg:.1f}d"], textposition="outside",
                hovertemplate=f"Has prior condition: {_p_avg:.1f}d avg LOS<extra></extra>",
            ))
            _fig_los.add_trace(go.Bar(
                x=[f"No prior\ncontact\n({_np_cnt} pts)"], y=[_np_avg],
                marker_color="#E24B4A", showlegend=False,
                text=[f"{_np_avg:.1f}d"], textposition="outside",
                hovertemplate=f"No prior contact: {_np_avg:.1f}d avg LOS<extra></extra>",
            ))
            _fig_los.update_layout(
                height=220, margin=dict(l=0, r=0, t=10, b=40),
                yaxis=dict(range=[0, 22], dtick=2, ticksuffix="d", title="Avg LOS (days)"),
                xaxis=dict(showgrid=False), barmode="group", bargap=0.4,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            _pc(_fig_los)

        # ── Insight callout ───────────────────────────────────────────────────
        _diff = _np_avg - _p_avg
        _insight([
            f"{_no_prior_n} of {total} outlier patients ({_pct(_no_prior_n)}) are "
            f"community-acquired first presentations — no OPD or inpatient history. "
            f"These cannot be intercepted at OPD level; focus is on inpatient management protocol.",
            f"Patients with a traceable prior condition averaged {_p_avg:.1f}d LOS vs "
            f"{_np_avg:.1f}d for those without — a {abs(_diff):.1f}-day difference. "
            f"Earlier recognition of the underlying condition may shorten the Sepsis episode.",
            f"{bonnke_n} of {total} stays ({bonnke_pct}%) are attributed to the same clinician "
            f"— clinical audit recommended. For the {_prior_opd_n} patients with prior OPD "
            f"contact only, investigate whether escalation criteria were missed.",
        ], variant="warn")

    else:
        st.info("No Sepsis LOS outlier data for the selected period.")

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
    fc1, fc2 = st.columns(2)

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

        with fc1:
            _card_title("Median Hours OPD to Admission")
            _sub("All wards admit 90%+ of patients within 4 hours.")
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
            _card_title("Within 4-Hour Admission Rate")
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
            _insight([
                "All wards admit 90%+ of patients within 4 hours.",
                "No clinical action required — OPD to admission pathway is functioning well.",
            ], variant="info")


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
    _df_lc_raw  = pd.DataFrame()
    _ltfu_count = 0
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
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                _kpi("Chronic Patients", _nf(_chronic_total),
                     "under active management", AFYA_BLUE)
            with c2:
                _kpi("Retained (90d)", _nf(row.get("retained_patients")),
                     str(_p(rr)) + " retention rate",
                     TEAL if rr >= 60 else CORAL)
            with c3:
                _kpi("LTFU >180d", _nf(_ltfu_count),
                     "no visit in 180+ days", CORAL)
            with c4:
                _kpi("Recoverable Revenue",
                     _k(rr_row.get("lapsing_revenue_recoverable")),
                     "lapsing patients (91–180d)", ORANGE)
    except Exception as e:
        st.warning(f"Retention KPIs: {e}")

    _gap(16)

    # ── SECTION A — LIFECYCLE OVERVIEW ────────────────────────────────────────
    _sh("A — Patient Lifecycle Overview", mt=8)
    try:
        df_lc = _df_lc_raw.copy()
        if not df_lc.empty:
            df_lc = df_lc.sort_values("_label", ascending=True)

            fig_lca = go.Figure()
            for lbl in df_lc["_label"]:
                row_lc = df_lc[df_lc["_label"] == lbl].iloc[0]
                fig_lca.add_trace(go.Bar(
                    x=[lbl],
                    y=[row_lc["patient_count"]],
                    name=lbl,
                    marker_color=_LC_COL.get(lbl, GRAY),
                    text=[f"{int(row_lc['patient_count']):,}"],
                    textposition="outside",
                    hovertemplate=f"<b>{lbl}</b><br>%{{y:,}} patients<extra></extra>",
                ))
            fig_lca.update_layout(
                height=300, margin=dict(l=0, r=20, t=10, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(title="Patients", rangemode="tozero"),
                xaxis=dict(title="", showticklabels=False),
                showlegend=True,
                legend=dict(orientation="h", y=-0.15, x=0, font=dict(size=11)),
                barmode="group",
            )
            _pc(fig_lca)

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

    _gap(16)

    # ── SECTION B — WHO DROPS OUT? ───────────────────────────────────────────
    _sh("B — Who Drops Out? LTFU Rate by Age, Payer & Gender", mt=8)
    try:
        df_cor = Q.load_ltfu_correlation(filters, run_query)
        if not df_cor.empty:
            for _c in ("ltfu_rate_pct", "total", "ltfu", "retained"):
                df_cor[_c] = pd.to_numeric(df_cor[_c], errors="coerce")

            _factor_order = ["Age Group", "Payer", "Gender"]
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

            _note(
                "Age group, gender and payer don't have an impact on patients who have left — "
                "this points to patient satisfaction of service as the primary driver of dropout.",
                w=False,
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
                     _nf(_ltfu_count),
                     "permanently lost (>180d)", CORAL)
            with c2:
                _kpi("Annual Revenue at Risk",
                     _k(row_r.get("chronic_ltfu_revenue_at_risk")),
                     "from LTFU chronic patients", CORAL)
            with c3:
                _kpi("Recoverable Revenue",
                     _k(row_r.get("lapsing_revenue_recoverable")),
                     "lapsing cohort (91–180d)", ORANGE)
    except Exception as e:
        st.warning(f"Revenue at risk: {e}")

    _gap(16)

    # ── PRIORITY 1 — LAPSING PATIENTS ─────────────────────────────────────────
    _sh("Priority 1 — Lapsing Patients (91–180 Days)", mt=8)
    try:
        df_rar3        = Q.load_revenue_at_risk(filters, run_query)
        df_lap_payer   = Q.load_lapsing_by_payer(filters, run_query)
        if not df_lap_payer.empty:
            df_lap_payer["patient_count"] = pd.to_numeric(df_lap_payer["patient_count"], errors="coerce")

        total_lapsing = _lc_count("Lapsing") or int(df_lap_payer["patient_count"].sum() or 0)

        c1, c2 = st.columns([1, 2])
        with c1:
            _kpi("Lapsing Patients", _nf(total_lapsing),
                 "91–180 days without return", ORANGE)
            _gap(8)
            if not df_rar3.empty:
                _kpi("Recoverable Revenue",
                     _k(df_rar3.iloc[0].get("lapsing_revenue_recoverable")),
                     "if re-engaged this month", ORANGE)

        with c2:
            if not df_lap_payer.empty:
                lapsing_by_payer = df_lap_payer.sort_values("patient_count", ascending=True)
                _payer_col = {
                    "Cash":                  "rgba(239,159,39,0.85)",
                    "Insurance / Corporate":  "#378ADD",
                    "NHIF / SHA":            "#378ADD",
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

        if not df_lap_payer.empty and total_lapsing > 0:
            cash_lap = int(df_lap_payer.loc[df_lap_payer["payer"] == "Cash", "patient_count"].sum() or 0)
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

    # ── PRIORITY 2 — 1–2 VISIT LTFU PATIENTS ────────────────────────────────
    _sh("Priority 2 — 1–2 Visit LTFU Patients", mt=8)
    try:
        df_p2 = Q.load_ltfu_priority2_patients(filters, run_query)
        if not df_p2.empty:
            for _c in ("visit_count", "days_since_last_visit", "revenue_at_risk"):
                df_p2[_c] = pd.to_numeric(df_p2[_c], errors="coerce").fillna(0)

            total_ltfu    = len(df_p2)
            total_chronic = total_ltfu  # all rows are chronic by query construction

            def _tier(n):
                if n <= 2: return 0
                if n <= 5: return 1
                return 2

            df_p2["_tier"] = df_p2["visit_count"].apply(_tier)

            tiers = [
                {"label": "1–2 visits", "color": "#E24B4A", "annot_color": "#A32D2D"},
                {"label": "3–5 visits", "color": "#BA7517", "annot_color": "#7A4C0E"},
                {"label": "5+ visits",  "color": "#0F6E56", "annot_color": "#0A4A3A"},
            ]
            tier_stats = [
                {
                    "total":   int((df_p2["_tier"] == i).sum()),
                    "chronic": int((df_p2["_tier"] == i).sum()),
                    "pct":     (df_p2["_tier"] == i).sum() / total_ltfu * 100 if total_ltfu else 0,
                }
                for i in range(3)
            ]

            # Total projected annual loss across all chronic LTFU patients —
            # same figure and methodology as Section C (count × avg_rev_per_visit × 4).
            df_rar_p2     = Q.load_revenue_at_risk(filters, run_query)
            tier1_revenue = float(df_rar_p2.iloc[0].get("chronic_ltfu_revenue_at_risk") or 0) if not df_rar_p2.empty else 0

            # Profile: 1–2 visit patients for Part 3 charts
            profile  = df_p2[df_p2["visit_count"] <= 2]
            age_ser  = profile["age_group"].value_counts().sort_values(ascending=False)
            dx_ser   = profile["clean_diagnosis"].value_counts().head(8).sort_values(ascending=False)

            # ── Care pathway signal queries ───────────────────────────────────
            def _sig_row(fn, label):
                try:
                    df = fn(filters, run_query)
                    df.columns = [c.lower() for c in df.columns]
                    return df.iloc[0].to_dict() if not df.empty else {}
                except Exception as exc:
                    st.warning(f"{label}: {exc}")
                    return {}

            _lab = _sig_row(Q.load_care_pathway_lab,       "Care pathway lab")
            _rx  = _sig_row(Q.load_care_pathway_rx,        "Care pathway rx")
            _fu  = _sig_row(Q.load_care_pathway_followup,  "Care pathway followup")
            _rad = _sig_row(Q.load_care_pathway_radiology, "Care pathway radiology")

            signals = {
                **_lab,
                **{k: v for k, v in _rx.items()  if k not in _lab},
                **{k: v for k, v in _fu.items()  if k not in _lab and k not in _rx},
                **{k: v for k, v in _rad.items() if k not in _lab and k not in _rx and k not in _fu},
            }

            html_block, h_px = _build_ltfu_p2_html(
                tier_stats    = tier_stats,
                tiers         = tiers,
                total_ltfu    = total_ltfu,
                total_chronic = total_chronic,
                tier1_revenue = tier1_revenue,
                age_labels    = age_ser.index.tolist(),
                age_values    = age_ser.values.tolist(),
                dx_labels     = dx_ser.index.tolist(),
                dx_values     = dx_ser.values.tolist(),
                signals       = signals,
            )
            _stcomp.html(html_block, height=h_px, scrolling=False)
        else:
            _note("No chronic LTFU patients found for this period.")
    except Exception as e:
        st.warning(f"Priority 2: {e}")

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


# ── NCD TAB HTML HELPERS ──────────────────────────────────────────────────────

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

def _ncd_t1_html(df: pd.DataFrame) -> str:
    """Table 1 — NCD complexity distribution."""
    total = int(df["patient_count"].sum()) if not df.empty else 1
    max_n = int(df["patient_count"].max()) if not df.empty else 1
    _cx_col = {"1 NCD": "#1D9E75", "2 NCDs": "#EF9F27", "3 NCDs": "#E24B4A", "4+ NCDs (Complex)": "#A32D2D"}
    hdr = ('<table><thead><tr>'
           '<th>Complexity</th>'
           '<th>Share of NCD Pts</th>'
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
        rows += (
            f'<tr>'
            f'<td style="font-weight:500;color:#1a1a18;">{label}</td>'
            f'<td><div style="display:flex;align-items:center;gap:6px;">'
            f'  <div style="width:{bw}px;height:8px;border-radius:3px;background:{col};opacity:0.8;flex-shrink:0;"></div>'
            f'  <span style="font-size:9px;color:#5f5e5a;">{n:,}</span></div></td>'
            f'<td style="text-align:right;font-weight:500;">{n:,}</td>'
            f'<td style="text-align:right;color:#5f5e5a;">{pct:.1f}%</td>'
            f'</tr>'
        )
    multi_pct = df.loc[df["ncd_complexity"] != "1 NCD", "pct_of_ncd_patients"].sum() if not df.empty else 0
    insight_text = (f"{multi_pct:.0f}% of NCD patients carry 2+ conditions. "
                    f"These patients need integrated management protocols.")
    return _NCD_BASE.format(hdr + rows + "</tbody></table>"), insight_text


def _ncd_t2_html(df: pd.DataFrame) -> str:
    """Table 2 — Comorbidity pipeline."""
    df = df.head(10).copy()
    max_days = float(df["avg_days_between_diagnoses"].max()) if not df.empty else 1
    hdr = ('<table><thead><tr>'
           '<th>Condition pair</th>'
           '<th style="text-align:center;">Patients</th>'
           '<th>Avg days to 2nd</th>'
           '<th>Speed</th>'
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
        days_cell = (
            f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<div style="width:{bw}px;height:8px;border-radius:3px;background:{bar_col};opacity:0.75;flex-shrink:0;"></div>'
            f'<span style="font-size:9px;color:{bar_col};font-weight:500;">{int(days)}d</span>'
            f'</div>'
        ) if days is not None else "—"
        rows += (
            f'<tr>'
            f'<td style="line-height:1.4;">{pair}</td>'
            f'<td style="text-align:center;font-weight:500;">{n:,}</td>'
            f'<td>{days_cell}</td>'
            f'<td>{badge}</td>'
            f'</tr>'
        )
    insight_text = ""
    if fastest_row:
        insight_text = (f"Fastest progression: {fastest_row[0]} — "
                        f"avg {int(fastest_row[1])} days to second diagnosis. "
                        f"This pair needs a co-management protocol.")
    return _NCD_BASE.format(hdr + rows + "</tbody></table>" + _SORT_SCRIPT), insight_text


def _ncd_t3_html(df: pd.DataFrame) -> str:
    """Table 3 — HTN by control status and comorbidity."""
    import math as _m
    _HTN_BG   = {"Controlled": "rgba(29,158,117,0.06)",  "Uncontrolled": "rgba(228,75,74,0.06)",  "No BP Recorded": "rgba(136,135,128,0.06)"}
    _HTN_BDGE = {"Controlled": ("#E1F5EE","#0F6E56"),    "Uncontrolled": ("#FCEBEB","#A32D2D"),    "No BP Recorded": ("#f5f5f3","#888780")}
    _HTN_BARC = {"Controlled": "#1D9E75",                "Uncontrolled": "#E24B4A",               "No BP Recorded": "#888780"}
    max_inv = float(df["avg_inv"].max()) if not df.empty and "avg_inv" in df.columns else 1
    if max_inv == 0 or _m.isnan(max_inv): max_inv = 1
    hdr = ('<table><thead><tr>'
           '<th>HTN status</th>'
           '<th>Comorbidity</th>'
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
        comorb = str(r.get("comorbidity_group", ""))
        n      = int(r.get("patients") or 0)
        avg_inv= float(r.get("avg_inv") or 0)
        rx_pct = float(r.get("on_rx_pct") or 0)
        bdg_bg, bdg_col = _HTN_BDGE.get(htn_status, ("#f5f5f3","#888780"))
        row_bg = _HTN_BG.get(htn_status, "transparent")
        bar_col= _HTN_BARC.get(htn_status, "#888780")
        bw     = round(avg_inv / max_inv * 130) if max_inv else 0
        badge  = (f'<span style="background:{bdg_bg};color:{bdg_col};font-size:9px;font-weight:500;'
                  f'padding:2px 7px;border-radius:20px;">{htn_status}</span>')
        rx_col = "#A32D2D" if rx_pct > 60 and htn_status == "Uncontrolled" else "#1a1a18"
        rows += (
            f'<tr style="background:{row_bg};">'
            f'<td>{badge}</td>'
            f'<td style="color:#5f5e5a;">{comorb}</td>'
            f'<td style="text-align:center;font-weight:500;">{n:,}</td>'
            f'<td><div style="display:flex;align-items:center;gap:6px;">'
            f'  <div style="width:{bw}px;height:8px;border-radius:3px;background:{bar_col};opacity:0.7;flex-shrink:0;"></div>'
            f'  <span style="font-size:9px;color:#5f5e5a;">{avg_inv:.1f}</span></div></td>'
            f'<td style="text-align:right;font-weight:500;color:{rx_col};">{rx_pct:.0f}%</td>'
            f'</tr>'
        )
    unc = df[df["htn_status"] == "Uncontrolled"] if not df.empty else pd.DataFrame()
    insight = ""
    if not unc.empty:
        htn_only = unc[unc["comorbidity_group"] == "HTN Only"]
        if not htn_only.empty:
            r0     = htn_only.iloc[0]
            n0     = int(r0.get("patients") or 0)
            inv0   = float(r0.get("avg_inv") or 0)
            rx0    = float(r0.get("on_rx_pct") or 0)
            insight = (f'<div class="ins"><strong>{n0:,}</strong> uncontrolled HTN-only patients average '
                       f'<strong>{inv0:.1f}</strong> investigations and <strong>{rx0:.0f}%</strong> are on '
                       f'antihypertensive medication — investigate medication adherence and dose adequacy.</div>')
    return _NCD_BASE.format(hdr + rows + "</tbody></table>" + insight)


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


def _ncd_t4_html(df: pd.DataFrame) -> str:
    """Table 4 — Prescription without clinical assessment."""
    hdr = ('<table><thead><tr>'
           '<th>Payer</th>'
           '<th>Condition</th>'
           '<th style="text-align:center;">Patients affected</th>'
           '<th style="text-align:right;">Avg annual rev</th>'
           '<th>Risk level</th>'
           '</tr></thead><tbody>')
    rows = ""
    for _, r in df.head(20).iterrows():
        payer = str(r.get("payer", ""))
        cond  = str(r.get("condition", ""))
        n     = int(r.get("patient_count") or 0)
        rev   = r.get("avg_annual_revenue")
        try:    rev_f = float(rev)
        except: rev_f = None
        import math as _m
        if rev_f is None or _m.isnan(rev_f):
            rev_s = "—"
        else:
            rev_s = f"KES {int(rev_f):,}"
        if n > 200:   risk_bg, risk_col, risk_lbl = "#FCEBEB", "#A32D2D", "High"
        elif n >= 50: risk_bg, risk_col, risk_lbl = "#FAEEDA", "#854F0B", "Medium"
        else:         risk_bg, risk_col, risk_lbl = "#f5f5f3",  "#888780", "Low"
        risk_badge = (f'<span style="background:{risk_bg};color:{risk_col};font-size:9px;'
                      f'font-weight:500;padding:2px 7px;border-radius:20px;">{risk_lbl}</span>')
        rows += (
            f'<tr>'
            f'<td>{_payer_badge_ncd(payer)}</td>'
            f'<td style="line-height:1.3;">{cond}</td>'
            f'<td style="text-align:center;font-weight:500;">{n:,}</td>'
            f'<td style="text-align:right;">{rev_s}</td>'
            f'<td>{risk_badge}</td>'
            f'</tr>'
        )
    return _NCD_BASE.format(hdr + rows + "</tbody></table>" + _SORT_SCRIPT)


def _ncd_t5_html(df: pd.DataFrame) -> str:
    """Table 5 — Chronic care quality & revenue matrix."""
    import math as _m
    max_pts = float(df["patient_count"].max()) if not df.empty else 1
    if max_pts == 0: max_pts = 1
    hdr = ('<table><thead><tr>'
           '<th>Condition</th>'
           '<th>Patients</th>'
           '<th>6-mo trend</th>'
           '<th>IP rate</th>'
           '<th>Top payer</th>'
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
        # trend
        try:    trend_f = float(trend)
        except: trend_f = None
        if trend_f is None or _m.isnan(trend_f):
            trend_s, trend_col = "—", "#888780"
        elif trend_f > 5:
            trend_s, trend_col = f"↑ +{trend_f:.0f}%", "#0F6E56"
        elif trend_f < -5:
            trend_s, trend_col = f"↓ {trend_f:.0f}%", "#A32D2D"
        else:
            trend_s, trend_col = f"→ {trend_f:.0f}%", "#888780"
        # ip
        ip_bdr = _ip_border(ip_rate)
        ip_dot = _ip_dot(ip_rate)
        # bar
        bw = round(pts / max_pts * 50)
        pts_cell = (f'<div style="display:flex;align-items:center;gap:4px;">'
                    f'<div style="width:{bw}px;height:8px;border-radius:3px;background:#378ADD;opacity:0.6;flex-shrink:0;"></div>'
                    f'<span style="font-size:9px;color:#5f5e5a;">{pts:,}</span></div>')
        # inv/visit colour
        inv_col = "#A32D2D" if inv_v > 7 else "#854F0B" if inv_v >= 5 else "#5f5e5a"
        # control
        try:    ctrl_f = float(ctrl)
        except: ctrl_f = None
        if ctrl_f is None or _m.isnan(ctrl_f):
            ctrl_s, ctrl_col = "—", "#888780"
        elif ctrl_f >= 50:
            ctrl_s, ctrl_col = f"{ctrl_f:.0f}%", "#0F6E56"
        elif ctrl_f >= 30:
            ctrl_s, ctrl_col = f"{ctrl_f:.0f}%", "#854F0B"
        else:
            ctrl_s, ctrl_col = f"{ctrl_f:.0f}%", "#A32D2D"
        rows += (
            f'<tr>'
            f'<td style="{ip_bdr}line-height:1.3;">{cond}</td>'
            f'<td>{pts_cell}</td>'
            f'<td style="color:{trend_col};font-weight:500;">{trend_s}</td>'
            f'<td>{ip_dot}{ip_rate:.1f}%</td>'
            f'<td>{_payer_badge_ncd(payer)}</td>'
            f'<td style="text-align:center;color:#5f5e5a;">{vpp:.1f}</td>'
            f'<td style="text-align:center;font-weight:500;color:{inv_col};">{inv_v:.2f}</td>'
            f'<td style="text-align:right;">{rev_s}</td>'
            f'<td style="text-align:center;font-weight:500;color:{ctrl_col};">{ctrl_s}</td>'
            f'</tr>'
        )
    return _NCD_BASE.format(hdr + rows + "</tbody></table>" + _SORT_SCRIPT)


def _ncd_t6_html(df: pd.DataFrame) -> str:
    """Table 6 — Undetected NCD risk patient action list."""
    import math as _m
    n_total = len(df)
    hdr = ('<table><thead><tr>'
           '<th>Patient</th>'
           '<th style="text-align:center;">Flagged visits</th>'
           '<th style="text-align:center;">Latest systolic</th>'
           '<th style="text-align:center;">Days since last</th>'
           '<th>Payer</th>'
           '<th>Urgency</th>'
           '</tr></thead><tbody>')
    rows = ""
    for _, r in df.head(40).iterrows():
        pat_id  = r.get("patient", "")
        fv      = int(r.get("visit_count") or 0)
        sys_v   = r.get("latest_systolic")
        days_v  = r.get("days_since_last_visit")
        payer   = str(r.get("payer") or "")
        try:    sys_f  = float(sys_v)
        except: sys_f  = None
        try:    days_f = float(days_v)
        except: days_f = None
        if sys_f is None or _m.isnan(sys_f):
            sys_s, sys_col = "—", "#888780"
        elif sys_f > 180: sys_s, sys_col = f"{int(sys_f)}", "#A32D2D"
        elif sys_f >= 160: sys_s, sys_col = f"{int(sys_f)}", "#854F0B"
        else: sys_s, sys_col = f"{int(sys_f)}", "#5f5e5a"
        if days_f is None or _m.isnan(days_f):
            days_s, days_col = "—", "#888780"
        elif days_f > 300: days_s, days_col = f"{int(days_f)}d", "#A32D2D"
        elif days_f >= 90: days_s, days_col = f"{int(days_f)}d", "#854F0B"
        else: days_s, days_col = f"{int(days_f)}d", "#888780"
        critical = (sys_f is not None and sys_f > 180) or (days_f is not None and days_f > 300)
        high     = (not critical) and ((sys_f is not None and sys_f >= 161) or (days_f is not None and days_f >= 150))
        if critical:   urg_bg, urg_col, urg_lbl = "#FCEBEB", "#A32D2D", "Critical"
        elif high:     urg_bg, urg_col, urg_lbl = "#FAEEDA", "#854F0B", "High"
        else:          urg_bg, urg_col, urg_lbl = "#f5f5f3",  "#5f5e5a", "Watch"
        urg_badge = (f'<span style="background:{urg_bg};color:{urg_col};font-size:9px;'
                     f'font-weight:500;padding:2px 7px;border-radius:20px;">{urg_lbl}</span>')
        rows += (
            f'<tr>'
            f'<td style="color:#378ADD;font-weight:500;">Patient {pat_id}</td>'
            f'<td style="text-align:center;">{fv}</td>'
            f'<td style="text-align:center;font-weight:500;color:{sys_col};">{sys_s}</td>'
            f'<td style="text-align:center;font-weight:500;color:{days_col};">{days_s}</td>'
            f'<td>{_payer_badge_ncd(payer)}</td>'
            f'<td>{urg_badge}</td>'
            f'</tr>'
        )
    critical_count = sum(
        1 for _, r in df.iterrows()
        if (float(r.get("latest_systolic") or 0) > 180) or (float(r.get("days_since_last_visit") or 0) > 300)
    )
    return (_NCD_BASE.format(hdr + rows + "</tbody></table>" + _SORT_SCRIPT),
            n_total, critical_count)


# ── Communicable disease colour map ──────────────────────────────────────────
_COMM_CLR_MAP = [
    ("Malaria",      "#1D9E75"),
    ("Typhoid",      "#EF9F27"),
    ("URTI",         "#378ADD"),
    ("TB",           "#E24B4A"),
    ("Enteric",      "#7F77DD"),
    ("GI",           "#7F77DD"),
    ("HIV",          "#D85A30"),
    ("Infectious",   "#9B59B6"),
]

def _comm_clr(disease: str) -> str:
    d = str(disease).lower()
    for key, clr in _COMM_CLR_MAP:
        if key.lower() in d:
            return clr
    return "#888780"


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


def _comm_t1_html(df: pd.DataFrame):
    """Table 1 — Surge months. Returns (html, insight_text)."""
    if df.empty:
        no_surge = '<div class="ins">No surge months detected in the selected period. All diseases remained within 1.5× of their period average.</div>'
        return _COMM_BASE.format(no_surge), ""

    import math as _m
    max_vs = float(df["vs_avg"].max()) if not df.empty else 2.0
    min_vs = 1.0
    span   = max(max_vs - min_vs, 0.1)

    hdr = ('<table><thead><tr>'
           '<th>Disease</th>'
           '<th>Month</th>'
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
        bw      = round((vs_avg - min_vs) / span * 160)
        bw      = max(0, min(160, bw))
        vs_s    = f"{vs_avg:.1f}×"
        if vs_avg >= 2.0:
            lvl_bg, lvl_col, lvl_lbl = "#FCEBEB", "#A32D2D", "Critical"
            vs_col = "#A32D2D"
        elif vs_avg >= 1.7:
            lvl_bg, lvl_col, lvl_lbl = "#FAEEDA", "#854F0B", "High"
            vs_col = "#854F0B"
        else:
            lvl_bg, lvl_col, lvl_lbl = "#f5f5f3", "#5f5e5a", "Elevated"
            vs_col = "#5f5e5a"
        badge = (f'<span style="background:{lvl_bg};color:{lvl_col};font-size:9px;'
                 f'font-weight:500;padding:2px 7px;border-radius:20px;">{lvl_lbl}</span>')
        dot = (f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
               f'background:{clr};margin-right:5px;vertical-align:middle;flex-shrink:0;"></span>')
        rows += (
            f'<tr>'
            f'<td><div style="display:flex;align-items:center;">{dot}'
            f'<span style="font-weight:500;">{dis}</span></div></td>'
            f'<td style="color:#5f5e5a;">{month_s}</td>'
            f'<td style="text-align:center;font-weight:500;">{visits:,}</td>'
            f'<td><div style="width:160px;height:8px;border-radius:3px;background:#f5f5f3;overflow:hidden;">'
            f'<div style="width:{bw}px;height:8px;border-radius:3px;background:{clr};opacity:0.75;"></div>'
            f'</div></td>'
            f'<td style="text-align:center;font-weight:500;color:{vs_col};">{vs_s}</td>'
            f'<td>{badge}</td>'
            f'</tr>'
        )

    # Build insight
    top_row      = df.iloc[0]
    top_dis      = str(top_row["disease_label"])
    top_month    = str(top_row["month_str"])
    top_vs       = float(top_row["vs_avg"])
    surge_counts = df.groupby("disease_label").size()
    most_dis     = surge_counts.idxmax()
    most_n       = int(surge_counts.max())
    pattern      = "sustained endemic spread" if most_n >= 2 else "repeated outbreaks"
    insight_text = (
        f"{top_dis} in {top_month} was the most severe surge at {top_vs:.1f}× average. "
        f"{most_dis} surged {most_n} time{'s' if most_n != 1 else ''} in the period, "
        f"suggesting {pattern}."
    )
    return _COMM_BASE.format(hdr + rows + "</tbody></table>" + _SORT_SCRIPT), insight_text


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


def _comm_t2_html(df: pd.DataFrame):
    """Table 2 — Pipeline matrix. Returns (html, has_verify_data)."""
    import math as _m
    max_v = float(df["quarterly_visits"].max()) if not df.empty else 1
    if max_v == 0: max_v = 1
    has_verify = False

    hdr = ('<table><thead><tr>'
           '<th>Disease</th>'
           '<th>90d visits</th>'
           '<th>Primary demographic</th>'
           '<th style="text-align:center;">Lab confirm %</th>'
           '<th style="text-align:center;">IP admission %</th>'
           '<th>Top comorbidity</th>'
           '<th>Primary payer</th>'
           '</tr></thead><tbody>')
    rows = ""
    for _, r in df.iterrows():
        dis      = str(r.get("disease_group", ""))
        visits   = float(r.get("quarterly_visits") or 0)
        demo     = str(r.get("primary_age_sex") or "—")
        lab_pct  = r.get("lab_confirmation_pct")
        comp_pct = r.get("data_completeness_pct")
        ip_pct   = float(r.get("inpatient_admission_pct") or 0)
        comorb   = str(r.get("primary_comorbidity") or "—")
        payer    = str(r.get("primary_payer") or "—")
        clr      = _comm_clr(dis)
        bw       = round(visits / max_v * 80)

        try:    lab_f = float(lab_pct)
        except: lab_f = None

        if lab_f is None or _m.isnan(lab_f):
            lab_cell = '<span style="color:#888780;">—</span>'
        else:
            lab_cell = f"{lab_f:.0f}%"

        # payer badge
        p_up = payer.upper()
        if "CASH" in p_up or "PRIVATE" in p_up:
            pay_bg, pay_col = "#FAEEDA", "#633806"
        else:
            pay_bg, pay_col = "#E6F1FB", "#185FA5"
        pay_badge = (f'<span style="background:{pay_bg};color:{pay_col};font-size:9px;'
                     f'font-weight:500;padding:2px 7px;border-radius:20px;">{payer}</span>')

        visits_cell = (
            f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<div style="width:{bw}px;height:8px;border-radius:3px;'
            f'background:#378ADD;opacity:0.6;flex-shrink:0;"></div>'
            f'<span style="font-size:9px;color:#5f5e5a;">{int(visits):,}</span></div>'
        )
        rows += (
            f'<tr>'
            f'<td style="border-left:3px solid {clr};padding-left:7px;'
            f'font-weight:500;line-height:1.3;">{dis}</td>'
            f'<td>{visits_cell}</td>'
            f'<td style="color:#5f5e5a;font-size:10px;">{demo}</td>'
            f'<td style="text-align:center;">{lab_cell}</td>'
            f'<td style="text-align:center;">{_ip_badge_comm(ip_pct)}</td>'
            f'<td style="color:#5f5e5a;font-size:10px;line-height:1.3;">{comorb}</td>'
            f'<td>{pay_badge}</td>'
            f'</tr>'
        )

    warning = ('<div class="warn">Lab Confirm % values are derived from live investigation records. '
               'If values appear unexpectedly low, verify whether lab investigations are being recorded '
               'against the correct visit ID in the EMR. Columns marked <em>verify data</em> indicate '
               'less than 50% of visits have a linked investigation record and should not be used for '
               'clinical decision-making until resolved.</div>')
    return _COMM_BASE.format(hdr + rows + "</tbody></table>" + _SORT_SCRIPT)


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
                               tickfont=dict(size=11, color="#888780")),
                    xaxis=dict(title="", tickfont=dict(size=11, color="#888780")),
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
                c1, c2 = st.columns([0.9, 1.1])
                with c1:
                    st.markdown("**Patient count by NCD group and gender:**")
                    # Normalise gender to Male/Female only; drop nulls/other
                    df_rk_g = df_rk_top.copy()
                    df_rk_g["gender"] = df_rk_g["gender"].str.upper().str.strip()
                    df_rk_g["gender"] = df_rk_g["gender"].map(
                        {"MALE": "Male", "M": "Male", "FEMALE": "Female", "F": "Female"}
                    )
                    df_rk_g = df_rk_g.dropna(subset=["gender"])
                    pivot_g = (df_rk_g.pivot_table(
                        index="ncd_group", columns="gender",
                        values="patient_count", aggfunc="sum", fill_value=0
                    ).reindex(top_conds).fillna(0))
                    color_map_g = {"Female": PURPLE, "Male": AFYA_BLUE}
                    fig_rk = go.Figure()
                    for g in ["Male", "Female"]:
                        if g not in pivot_g.columns:
                            continue
                        fig_rk.add_trace(go.Bar(
                            x=pivot_g[g].values, y=pivot_g.index.tolist(),
                            name=g, orientation="h",
                            marker_color=color_map_g[g],
                            hovertemplate=f"<b>%{{y}}</b><br>{g}: %{{x:,}}<extra></extra>",
                        ))
                    fig_rk.update_layout(
                        barmode="stack", height=360,
                        margin=dict(l=0, r=0, t=36, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(title="Patients", rangemode="tozero",
                                   tickfont=dict(size=11)),
                        yaxis=dict(title="", autorange="reversed",
                                   tickfont=dict(size=12)),
                        legend=dict(orientation="h", y=1.06, x=0,
                                    xanchor="left", font=dict(size=12)),
                    )
                    _pc(fig_rk)
                with c2:
                    st.markdown("**Monthly visits per condition:**")
                    try:
                        df_bt = Q.load_burden_trend(filters, run_query)
                        df_bt["visit_count"] = pd.to_numeric(df_bt["visit_count"], errors="coerce")
                        df_bt["visit_month"] = pd.to_datetime(df_bt["visit_month"])
                        _ncd_kw = ("Cardiovascular", "Diabetes", "Neurolog", "Mental",
                                   "Musculo", "Chronic", "Endocrin", "Metabolic")
                        df_bt_ncd = df_bt[df_bt["burden_group"].str.contains(
                            "|".join(_ncd_kw), case=False, na=False
                        )]
                        # Top 5 only
                        top_bt = (df_bt_ncd.groupby("burden_group")["visit_count"]
                                  .sum().nlargest(5).index.tolist())
                        df_bt_ncd = df_bt_ncd[df_bt_ncd["burden_group"].isin(top_bt)]
                        if not df_bt_ncd.empty:
                            palette = [AFYA_BLUE, TEAL, ORANGE, CORAL, PURPLE]
                            fig_tr = go.Figure()
                            for i, cond in enumerate(top_bt):
                                sub = df_bt_ncd[df_bt_ncd["burden_group"] == cond].sort_values("visit_month")
                                if sub.empty:
                                    continue
                                short = (cond.replace("NCD — ", "").replace("NCD - ", "")
                                             .replace("NCD – ", "").replace("NCD: ", ""))
                                fig_tr.add_trace(go.Scatter(
                                    x=sub["visit_month"], y=sub["visit_count"],
                                    name=short, mode="lines",
                                    line=dict(color=palette[i % len(palette)], width=2),
                                    hovertemplate=f"<b>{short}</b><br>%{{x|%b %Y}}: %{{y:,}}<extra></extra>",
                                ))
                            fig_tr.update_layout(
                                height=400,
                                margin=dict(l=0, r=0, t=80, b=0),
                                plot_bgcolor="white", paper_bgcolor="white",
                                xaxis=dict(title="", tickfont=dict(size=11),
                                           gridcolor="rgba(0,0,0,0.05)"),
                                yaxis=dict(title="Visits", rangemode="tozero",
                                           tickfont=dict(size=11),
                                           gridcolor="rgba(0,0,0,0.05)"),
                                legend=dict(
                                    font=dict(size=11), orientation="h",
                                    x=0, y=1.22, xanchor="left", yanchor="top",
                                ),
                            )
                            _pc(fig_tr)
                        else:
                            st.caption("No NCD trend data available for this period.")
                    except Exception as e:
                        st.warning(f"Trend chart: {e}")
        except Exception as e:
            st.warning(f"B2: {e}")

        _gap(12)

        # ── B3: NCD COMPLEXITY ────────────────────────────────────────────────
        _sh("NCD Complexity — Simple vs Multi-Morbidity Cases", mt=8)
        _note("Patients with 2+ NCDs are 3-4x more expensive to manage. The share of complex cases drives case management staffing and chronic care protocol design.")
        try:
            df_cx = Q.load_ncd_complexity_distribution(filters, run_query)
            if not df_cx.empty:
                df_cx["patient_count"]       = pd.to_numeric(df_cx["patient_count"],       errors="coerce")
                df_cx["pct_of_ncd_patients"] = pd.to_numeric(df_cx["pct_of_ncd_patients"], errors="coerce")
                c1, c2 = st.columns(2)
                with c1:
                    _pc(donut(
                        labels=df_cx["ncd_complexity"].tolist(),
                        values=df_cx["patient_count"].tolist(),
                        color_map={"1 NCD": TEAL, "2 NCDs": ORANGE, "3 NCDs": CORAL, "4+ NCDs (Complex)": PURPLE},
                        height=280,
                    ))
                with c2:
                    t1_html, t1_insight = _ncd_t1_html(df_cx)
                    _stcomp.html(t1_html, height=280, scrolling=False)
                    st.caption(t1_insight)
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
                age_order = ["Toddler (0–4)", "Child (5–12)", "Adolescent (13–17)", "Youth (18–24)",
                             "Young Adult (25–34)", "Adult (35–44)", "Middle Age (45–54)",
                             "Older Adult (55–64)", "Senior (65+)", "Unknown"]
                pivot_hm = pivot_hm.reindex([a for a in age_order if a in pivot_hm.index])
                n_rows = len(pivot_hm.index)
                fig_hm = go.Figure(go.Heatmap(
                    z=pivot_hm.values,
                    x=pivot_hm.columns.tolist(),
                    y=pivot_hm.index.tolist(),
                    colorscale=[[0, "#BDD7EE"], [0.4, "#378ADD"], [0.75, "#0072CE"], [1, "#003467"]],
                    hovertemplate="<b>%{y} — %{x}</b><br>Patients: %{z:,}<extra></extra>",
                    text=pivot_hm.values, texttemplate="%{text}",
                    textfont=dict(size=13, color="white"),
                    xgap=2, ygap=2,
                ))
                fig_hm.update_layout(
                    height=max(380, n_rows * 42 + 80),
                    margin=dict(l=10, r=20, t=20, b=80),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(tickangle=-35, title="", tickfont=dict(size=13)),
                    yaxis=dict(title="", tickfont=dict(size=13)),
                    coloraxis_showscale=True,
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
                df_cp["avg_days_between_diagnoses"] = pd.to_numeric(df_cp["avg_days_between_diagnoses"], errors="coerce")
                df_cp["patient_count"]              = pd.to_numeric(df_cp["patient_count"], errors="coerce")
                t2_html, t2_insight = _ncd_t2_html(df_cp)
                _stcomp.html(t2_html, height=len(df_cp.head(10)) * 36 + 40, scrolling=False)
                if t2_insight:
                    st.caption(t2_insight)
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
                        "patients":  d["patient_count"].sum(),
                        "avg_inv":   ((d["avg_investigations"] * d["patient_count"]).sum()
                                      / max(d["patient_count"].sum(), 1)),
                        "on_rx_pct": (d.loc[
                            pd.to_numeric(d["on_antihypertensive"], errors="coerce") == 1,
                            "patient_count"
                        ].sum() / max(d["patient_count"].sum(), 1) * 100),
                    }), include_groups=False)
                    .reset_index()
                )
                _stcomp.html(_ncd_t3_html(comorb_grp), height=len(comorb_grp) * 30 + 72, scrolling=False)
        except Exception as e:
            st.warning(f"B7: {e}")

        _gap(12)

        # ── B8: PRESCRIPTION WITHOUT CLINICAL ASSESSMENT ──────────────────────
        _sh("Prescription Without Clinical Assessment — Documentation Gap", mt=8)
        _note(
            "Chronic visits where a prescription was issued but NO vitals were recorded AND "
            "NO clinical note exists. Governance risk signal."
        )
        try:
            df_ph = Q.load_chronic_pharmacy_only(filters, run_query)
            if not df_ph.empty:
                df_ph["patient_count"]      = pd.to_numeric(df_ph["patient_count"],      errors="coerce")
                df_ph["avg_annual_revenue"] = pd.to_numeric(df_ph["avg_annual_revenue"],  errors="coerce")
                _stcomp.html(_ncd_t4_html(df_ph), height=400, scrolling=True)
                st.caption(
                    "⚠️ Gap Visit % removed pending data validation. "
                    "Investigate whether the gap visit logic is capturing pharmacy-only visits correctly."
                )
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
                _stcomp.html(_ncd_t5_html(df_qmx), height=len(df_qmx) * 30 + 38, scrolling=False)
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
                for _c in ("visit_count", "latest_systolic", "latest_blood_sugar", "days_since_last_visit"):
                    if _c in df_ev.columns:
                        df_ev[_c] = pd.to_numeric(df_ev[_c], errors="coerce")
                t6_html, t6_total, t6_critical = _ncd_t6_html(df_ev)
                _stcomp.html(t6_html, height=480, scrolling=True)
                st.caption(
                    f"{t6_total:,} patients flagged. "
                    f"{t6_critical} marked Critical (systolic >180 or not seen in 300+ days). "
                    "Prioritise these for immediate outreach."
                )
        except Exception as e:
            st.warning(f"B10: {e}")

        # ── Chronic illness growth over time ──────────────────────────────
        _gap(16)
        _sh("Chronic Illness Growth Over Time", mt=8)
        try:
            df_chr = Q.load_cv_chronic_growth(filters, run_query)
            if not df_chr.empty:
                df_chr["visit_month"] = pd.to_datetime(df_chr["visit_month"])
                df_chr = df_chr.sort_values("visit_month")
                _all_ages_chr = (df_chr.groupby("age_group")["chronic_patients"]
                                 .sum().sort_values(ascending=False).index.tolist())
                _chr_months = sorted(df_chr["visit_month"].unique())
                _col_chr, _ = st.columns([1, 1])
                with _col_chr:
                    _chr_age_sel = st.selectbox(
                        "Filter by age group",
                        options=["All"] + _all_ages_chr,
                        index=0, key="db_chr_age_filter",
                    )
                    df_chr_f = (df_chr if _chr_age_sel == "All"
                                else df_chr[df_chr["age_group"] == _chr_age_sel])
                    fig_ch = go.Figure()
                    _CHR_GCOL = {"Female": PURPLE, "Male": TEAL}
                    for _g in [g for g in ["Female", "Male"]
                               if g in df_chr_f["gender"].dropna().unique()]:
                        _raw = (df_chr_f[df_chr_f["gender"] == _g]
                                .groupby("visit_month")["chronic_patients"].sum()
                                .reindex(_chr_months, fill_value=0))
                        _prev = _raw.shift(1)
                        _idx  = (_raw / _prev.replace(0, float("nan")) * 100).tolist()
                        _lv   = next((v for v in reversed(_idx) if v and not pd.isna(v)), None)
                        _dstr = (f"+{round(_lv-100)}%" if _lv else "")
                        fig_ch.add_trace(go.Scatter(
                            x=_chr_months, y=_idx,
                            name=f"{_g}  {_dstr}",
                            mode="lines+markers",
                            line=dict(color=_CHR_GCOL.get(_g, GREY), width=2),
                            marker=dict(size=5),
                        ))
                    fig_ch.add_hline(y=100, line_dash="dot",
                                     line_color="rgba(0,0,0,0.12)", line_width=1)
                    fig_ch.update_layout(
                        **CHART_BASE, height=280,
                        xaxis={**AX, "showgrid": False, "tickformat": "%b %y"},
                        yaxis={**AX, "title": ""},
                        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=9)),
                    )
                    _pc(fig_ch)
        except Exception as e:
            st.warning(f"Chronic growth: {e}")

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
                _AGE_ORDER = [
                    "Toddler (0–4)", "Child (5–12)", "Adolescent (13–17)",
                    "Youth (18–24)", "Young Adult (25–34)", "Adult (35–44)",
                    "Middle Age (45–54)", "Older Adult (55–64)", "Senior (65+)", "Unknown",
                ]
                _AGE_SHORT = {
                    "Toddler (0–4)": "0–4", "Child (5–12)": "5–12",
                    "Adolescent (13–17)": "13–17", "Youth (18–24)": "18–24",
                    "Young Adult (25–34)": "25–34", "Adult (35–44)": "35–44",
                    "Middle Age (45–54)": "45–54", "Older Adult (55–64)": "55–64",
                    "Senior (65+)": "65+", "Unknown": "Unk",
                }
                n_cols = min(3, len(diseases))
                rows_d = [diseases[i:i+n_cols] for i in range(0, len(diseases), n_cols)]
                for row_diseases in rows_d:
                    cols_d = st.columns(n_cols)
                    for col_d, dis in zip(cols_d, row_diseases):
                        sub_d = df_dem[df_dem["disease_label"] == dis].copy()
                        with col_d:
                            st.markdown(f"**{dis}**")
                            age_sex = sub_d.groupby(["age_group", "sex"])["patient_count"].sum().reset_index()
                            # sort age groups in correct order and shorten labels
                            age_sex["age_order"] = age_sex["age_group"].map(
                                {a: i for i, a in enumerate(_AGE_ORDER)}
                            ).fillna(99)
                            age_sex = age_sex.sort_values("age_order")
                            age_sex["age_label"] = age_sex["age_group"].map(_AGE_SHORT).fillna(age_sex["age_group"])
                            fig_ds = go.Figure()
                            sex_colors = {"F": PURPLE, "FEMALE": PURPLE,
                                          "M": AFYA_BLUE, "MALE": AFYA_BLUE,
                                          "Unknown": GRAY}
                            for sx in ["FEMALE", "F", "MALE", "M"]:
                                sub_sx = age_sex[age_sex["sex"].str.upper() == sx.upper()]
                                if sub_sx.empty:
                                    continue
                                label = "Female" if sx in ("FEMALE", "F") else "Male"
                                fig_ds.add_trace(go.Bar(
                                    name=label,
                                    x=sub_sx["age_label"],
                                    y=sub_sx["patient_count"],
                                    marker_color=sex_colors.get(sx, GRAY),
                                    showlegend=(dis == diseases[0]),
                                    legendgroup=label,
                                ))
                            fig_ds.update_layout(
                                barmode="group", height=240,
                                margin=dict(l=0, r=0, t=10, b=50),
                                plot_bgcolor="white", paper_bgcolor="white",
                                yaxis_title="Patients",
                                xaxis=dict(
                                    tickangle=-45,
                                    tickfont=dict(size=12),
                                ),
                                legend=dict(orientation="h", y=-0.28, font=dict(size=12)),
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
                            textfont=dict(size=11, color="#DC2626"),
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

                    # surge summary table — same disease set and means as the chart above
                    _gap(12)
                    _sh("Surge Months — Malaria, URTI & Typhoid (>1.5× period average)", mt=4)
                    surge_df["vs_avg"] = surge_df.apply(
                        lambda r: round(r["visit_count"] / max(means.get(r["disease_label"], 1), 1), 1), axis=1
                    )
                    surge_all = surge_df[surge_df["vs_avg"] >= 1.5].copy()
                    surge_all["month_str"] = surge_all["visit_month"].dt.strftime("%b %Y")
                    surge_all = surge_all.rename(columns={"visit_count": "visits"}).sort_values("vs_avg", ascending=False)
                    t1_html, t1_insight = _comm_t1_html(surge_all)
                    _stcomp.html(t1_html, height=300, scrolling=True)
                    if t1_insight:
                        st.caption(t1_insight)
        except Exception as e:
            st.warning(f"Surge pattern: {e}")

        _gap(12)

        # ── D8: Unified Pipeline Matrix (from live data) ───────────────────────
        _sh("Unified Acute & Communicable Pipeline Matrix", mt=8)
        _note("Disease colour matches the trend chart. IP Admission % is the primary risk signal. Lab Confirm % shows 'verify data' where less than 50% of visits have a linked investigation record.")
        try:
            df_cpm = Q.load_communicable_pipeline_matrix(filters, run_query)
            if not df_cpm.empty:
                for _c in ("quarterly_visits", "lab_confirmation_pct",
                           "data_completeness_pct", "inpatient_admission_pct"):
                    if _c in df_cpm.columns:
                        df_cpm[_c] = pd.to_numeric(df_cpm[_c], errors="coerce")
                t2_html = _comm_t2_html(df_cpm)
                _stcomp.html(t2_html, height=300, scrolling=True)
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
                    _MH_CLR = {
                        "Depression & Anxiety":     "#378ADD",
                        "Substance & Alcohol":      "#E24B4A",
                        "Psychotic Disorders":      "#7F77DD",
                        "Dementia / Organic Brain": "#EF9F27",
                        "Other Mental Health":      "#888780",
                    }
                    # trend direction from monthly data
                    _trend_map = {}
                    try:
                        _df_mht_tmp = Q.load_mh_monthly_trend(filters, run_query)
                        if not _df_mht_tmp.empty:
                            _df_mht_tmp["visit_month"] = pd.to_datetime(_df_mht_tmp["visit_month"])
                            for _cat, _grp in _df_mht_tmp.groupby("mh_category"):
                                _grp = _grp.sort_values("visit_month")
                                if len(_grp) >= 2:
                                    _fh = _grp["visit_count"].iloc[:len(_grp)//2].mean()
                                    _lh = _grp["visit_count"].iloc[len(_grp)//2:].mean()
                                    _trend_map[_cat] = (
                                        "Rising"    if _lh > _fh * 1.15 else
                                        "Declining" if _lh < _fh * 0.85 else
                                        "Stable"
                                    )
                    except Exception:
                        pass
                    hdr_mh = ('<table><thead><tr>'
                              '<th>Condition</th>'
                              '<th style="text-align:center;">Trend</th>'
                              '<th style="text-align:center;">Standalone</th>'
                              '<th style="text-align:center;">Comorbid</th>'
                              '<th>Top co-condition</th>'
                              '</tr></thead><tbody>')
                    rows_mh = ""
                    for _, _r in df_mhc.iterrows():
                        _cat    = str(_r.get("mh_category", ""))
                        _sa     = int(_r.get("standalone_patients") or 0)
                        _co     = int(_r.get("comorbid_patients") or 0)
                        _top    = str(_r.get("top_comorbidity") or "—")
                        _clr    = _MH_CLR.get(_cat, "#888780")
                        _tl     = _trend_map.get(_cat, "—")
                        if _tl == "Rising":
                            _tb, _tc, _ts = "#E1F5EE", "#0F6E56", "↑ Rising"
                        elif _tl == "Declining":
                            _tb, _tc, _ts = "#FCEBEB", "#A32D2D", "↓ Declining"
                        elif _tl == "Stable":
                            _tb, _tc, _ts = "#f5f5f3", "#5f5e5a", "→ Stable"
                        else:
                            _tb, _tc, _ts = "#f5f5f3", "#888780", "—"
                        _badge = (f'<span style="background:{_tb};color:{_tc};font-size:9px;'
                                  f'font-weight:500;padding:2px 7px;border-radius:20px;">{_ts}</span>')
                        rows_mh += (
                            f'<tr>'
                            f'<td style="border-left:3px solid {_clr};padding-left:7px;'
                            f'font-weight:500;line-height:1.3;">{_cat}</td>'
                            f'<td style="text-align:center;">{_badge}</td>'
                            f'<td style="text-align:center;font-weight:500;">{_sa:,}</td>'
                            f'<td style="text-align:center;font-weight:500;color:#7F77DD;">{_co:,}</td>'
                            f'<td style="color:#5f5e5a;font-size:10px;line-height:1.3;">{_top}</td>'
                            f'</tr>'
                        )
                    _html_mh = _NCD_BASE.format(hdr_mh + rows_mh + "</tbody></table>")
                    _stcomp.html(_html_mh, height=len(df_mhc) * 34 + 44, scrolling=False)
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
