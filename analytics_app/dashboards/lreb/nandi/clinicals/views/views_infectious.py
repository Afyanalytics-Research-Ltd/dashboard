"""
views_infectious.py: Tab 3, Infectious disease.

No SQL here: every number comes from queries.q3_*. This file shapes DataFrames into Plotly charts and
ui_template cards. Sections: A burden at a glance · B children under 5 · C malaria · D pneumonia ·
E sepsis · F urinary tract infection · G gonococcal infection.

Data-quality issues (coding errors, documentation gaps) belong on the Data Quality tab unless they limit
a specific finding, in which case they appear as a note on that finding.
"""
from __future__ import annotations

import re
import textwrap

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import nandi_queries as q
import nandi_ui_template as ui

BLUE, TEAL, AMBER, RED, GREY = ui.BLUE, ui.PRIMARY, ui.AMBER, ui.CRITICAL, ui.NEUTRAL
BLUE2 = "#4A7FB5"
PINK = "#D6336C"
LGREY = "#B8BCC4"
DKAMBER = "#B45309"
AGES = ["Toddler (0-4)", "Child (5-12)", "Adolescent (13-17)", "Youth (18-24)", "Young Adult (25-34)", "Adult (35-44)", "Middle Age (45-54)", "55+"]
AGE_SHORT = ["0–4", "5–12", "13–17", "18–24", "25–34", "35–44", "45–54", "55+"]
DISEASE_COL = {"Malaria": AMBER, "Pneumonia": BLUE, "Sepsis": RED, "UTI": TEAL, "Gonococcal": DKAMBER}

_CSS = f"""
<style>
.chain {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin:0 0 14px; }}
.chain .cs {{ background:#fff; border:1px solid {ui.BORDER}; border-top:3px solid {RED}; border-radius:8px; padding:14px 16px; font-size:12.5px; line-height:1.4; display:flex; flex-direction:column; gap:3px; box-shadow:0 1px 3px rgba(0,0,0,.04); }}
.chain .ct {{ font-size:10.5px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:{RED}; }}
.chain b {{ font-size:26px; line-height:1.15; color:{RED}; }}
@media(max-width:1100px) {{ .chain {{ grid-template-columns:1fr 1fr; }} }}
@media(max-width:620px) {{ .chain {{ grid-template-columns:1fr; }} }}
.afya-why.red div {{ border-left-color:{RED}; }}
.afya-link {{ display:flex; justify-content:space-between; align-items:center; gap:12px; background:#EEF4FB; border:1px solid #C9D8EA; border-left:3px solid {BLUE}; border-radius:0 6px 6px 0; padding:10px 16px; font-size:13px; margin-top:12px; }}
.afya-link a {{ color:#2E8B3A; font-weight:700; text-decoration:none; white-space:nowrap; }}
.afya-stat {{ border-left:3px solid {BLUE}; padding:4px 10px; margin-bottom:8px; font-size:12px; }}
.afya-stat b {{ display:block; font-size:22px; line-height:1.2; }}
.afya-tiles {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-top:6px; }}
.afya-tiles div {{ border-radius:6px; padding:12px; font-size:12px; font-weight:600; }}
.afya-tiles b {{ display:block; font-size:28px; line-height:1.2; }}
.afya-tiles i {{ display:block; font-weight:400; color:{ui.TEXT_MUTED}; }}
.afya-limit {{ font-size:11px; color:#92400E; background:#FEF3E0; border-radius:4px; padding:4px 8px; margin-top:8px; display:inline-block; }}
.afya-flow {{ display:grid; grid-template-columns:minmax(0,.9fr) 26px minmax(0,1.3fr) 26px minmax(0,1fr); align-items:stretch; row-gap:12px; margin-top:6px; }}
.afya-flow .fh {{ font-size:10.5px; font-weight:700; letter-spacing:.07em; text-transform:uppercase; color:{ui.TEXT_MUTED}; padding:0 2px 2px; }}
.afya-flow .fb {{ border:1px solid #C9D8EA; background:#F4F8FD; border-radius:8px; padding:11px 14px; line-height:1.3; display:flex; flex-direction:column; justify-content:center; min-height:62px; }}
.afya-flow .fb .t {{ font-size:13px; font-weight:700; color:{ui.TEXT}; }}
.afya-flow .fb .s {{ font-size:11.5px; color:{ui.TEXT_MUTED}; margin-top:2px; }}
.afya-flow .fb .n {{ font-size:22px; font-weight:700; color:{BLUE}; line-height:1.15; }}
.afya-flow .fb.end {{ border-color:transparent; color:#fff; }}
.afya-flow .fb.end .t {{ color:#fff; font-size:12px; font-weight:600; opacity:.95; }}
.afya-flow .fb.end .n {{ color:#fff; font-size:24px; }}
.afya-flow .fb.end .s {{ color:#fff; opacity:.85; }}
.afya-flow .ar {{ display:flex; align-items:center; justify-content:center; color:#9CA3AF; font-size:18px; }}
@media(max-width:820px) {{ .afya-flow {{ grid-template-columns:1fr; row-gap:4px; }} .afya-flow .fh {{ display:none; }} .afya-flow .ar {{ transform:rotate(90deg); height:16px; }} .afya-flow .fb:nth-child(5n+1) {{ margin-top:16px; }} }}
.bub {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); justify-items:center; align-items:start; gap:22px 12px; padding:14px 4px 10px; }}
.bub .bi {{ display:flex; flex-direction:column; align-items:center; width:100%; max-width:130px; text-align:center; }}
.bub .bc {{ border-radius:50%; display:flex; align-items:center; justify-content:center; color:#fff; font-weight:700; border:3px solid #fff; box-shadow:0 0 0 1px rgba(0,0,0,.06); }}
.bub .bn {{ margin-top:8px; font-size:12px; line-height:1.3; color:{ui.TEXT}; font-weight:600; }}
.bub .bv {{ font-size:11.5px; color:{ui.TEXT_MUTED}; }}
.afya-cap {{ font-size:12px; line-height:1.5; color:{ui.TEXT_MUTED}; margin-top:10px; padding-bottom:10px; }}
.afya-body {{ font-size:13px; line-height:1.55; color:{ui.TEXT}; margin-top:8px; }}
.waffle {{ display:grid; grid-template-columns:repeat(10,20px); gap:3px; }}
.waffle i {{ width:20px; height:20px; border-radius:3px; display:block; }}
</style>
"""


# ============================================================ helpers
def _n(x) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except (TypeError, ValueError):
        return "n/a"


def _f(x, default=0.0) -> float:
    try:
        return default if pd.isna(x) else float(x)
    except (TypeError, ValueError):
        return default


def _pct(a, b, d=0) -> str:
    return f"{(100 * _f(a) / b):.{d}f}%" if b else "n/a"


def _col(df: pd.DataFrame, name: str, default=0):
    return df[name] if name in df.columns else pd.Series([default] * len(df), index=df.index)


def _sum(df: pd.DataFrame, name: str) -> float:
    return float(pd.to_numeric(_col(df, name), errors="coerce").fillna(0).sum()) if not df.empty else 0.0


def _legend_html(items) -> str:
    """Legend drawn as HTML above the chart (items: [(name, colour)]). It always sits on top and wraps, whatever the chart width."""
    return ('<div style="display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12px;color:#374151;margin:2px 0 6px">'
            + "".join(f'<span style="display:inline-flex;align-items:center;gap:6px"><i style="width:11px;height:11px;border-radius:2px;background:{c};display:inline-block"></i>{n}</span>'
                      for n, c in items) + "</div>")


def _plot(fig: go.Figure, key: str) -> None:
    meta = fig.layout.meta
    if isinstance(meta, dict) and meta.get("legend_html"):
        st.markdown(meta["legend_html"], unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config=ui.PC_CFG, key=key)


def _eq(first: float, n: int) -> list[str]:
    """Column widths: one wide label column, then n equal columns."""
    return [f"{first}%"] + [f"{(100 - first) / n:.2f}%"] * n


def _cap(text: str) -> None:
    """Small muted note under a chart, in the template's note size (st.caption renders far larger)."""
    st.markdown(f'<div class="afya-cap">{text}</div>', unsafe_allow_html=True)


def _empty(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        _cap("No data returned for this card.")
        return True
    return False


def _hbar(labels, values, colours, texts=None, height=240):
    fig = ui.bar_chart(list(labels), list(values), list(colours) if not isinstance(colours, str) else [colours] * len(labels), texts, height=height)
    fig.update_layout(margin=dict(l=10, r=120, t=6, b=6))      # room for the "1,606 · 65.1%" labels
    fig.update_yaxes(automargin=True)
    return fig


def _wrap(label: str, width: int = 24) -> str:
    return "<br>".join(textwrap.wrap(str(label), width, break_long_words=False)) or str(label)


def _sb(labels, series, height=300, suffix="") -> go.Figure:
    """Horizontal stacked bars: legend on top in the same order as the stack, thin white gaps between segments."""
    fig = ui.stacked_bar_chart(labels, series, height=height, xsuffix=suffix)
    fig.update_traces(marker_line_color="#fff", marker_line_width=1.5)
    fig.update_layout(showlegend=False, meta=dict(legend_html=_legend_html([(n, c) for n, _, c in series])),
                      margin=dict(l=10, r=20, t=6, b=10))
    fig.update_yaxes(automargin=True)
    return fig


def _single_bar(label: str, series, height: int = 120) -> go.Figure:
    """One horizontal stacked bar with the legend on top and no axis."""
    fig = go.Figure()
    for name, v, colour in series:
        fig.add_trace(go.Bar(y=[label], x=[v], orientation="h", name=name, marker_color=colour, text=[_n(v)], textposition="inside", insidetextanchor="middle",
                             textfont=dict(color="#fff", size=12), hovertemplate=f"{name}: %{{x:,}}<extra></extra>"))
    ui._finish(fig, height, barmode="stack", showlegend=False, meta=dict(legend_html=_legend_html([(n, c) for n, _, c in series])), margin=dict(l=10, r=10, t=4, b=6))
    fig.update_xaxes(visible=False, showgrid=False)
    fig.update_yaxes(visible=False)
    return fig


def _lolli(labels, values, colour, texts, height=None):
    """Horizontal lollipop: a thin stem and a dot, first label on top."""
    labels, values = list(labels), list(values)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=values, orientation="h", marker_color=colour, width=0.06, hoverinfo="skip", showlegend=False))
    labels = [_wrap(l, 26) for l in labels]
    fig.data[0].y = labels
    fig.add_trace(go.Scatter(y=labels, x=values, mode="markers+text", marker=dict(size=13, color=colour), text=texts, textposition="middle right",
                             textfont=dict(size=12, color=ui.TEXT), cliponaxis=False, hovertemplate="%{y}: %{x:,}<extra></extra>", showlegend=False))
    ui._finish(fig, height or max(180, 44 * len(labels) + 20), margin=dict(l=10, r=110, t=6, b=6))
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.update_xaxes(visible=False, showgrid=False, range=[0, max(values + [1]) * 1.08])
    return fig


def _columns(x, y, colours, texts, height=260, ysuffix=""):
    fig = go.Figure(go.Bar(x=list(x), y=list(y), marker_color=colours, text=texts, textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}: %{y:,}<extra></extra>"))
    ui._finish(fig, height, showlegend=False, margin=dict(l=10, r=10, t=24, b=10))
    fig.update_yaxes(showgrid=True, gridcolor=ui.SURFACE_2, ticksuffix=ysuffix, range=[0, max([_f(v) for v in y] + [1]) * 1.18])
    fig.update_xaxes(showgrid=False, type="category")
    return fig


def _stack_cols(x, series, height=300, ysuffix=""):
    fig = go.Figure()
    for name, vals, colour in series:
        fig.add_trace(go.Bar(x=list(x), y=list(vals), name=name, marker_color=colour, hovertemplate=f"%{{x}}<br>{name}: %{{y:,}}<extra></extra>"))
    ui._finish(fig, height, barmode="stack")
    fig.update_yaxes(showgrid=True, gridcolor=ui.SURFACE_2, ticksuffix=ysuffix)
    fig.update_xaxes(showgrid=False, type="category")
    return fig


def _ward_card(title: str, disease: str, colour: str, wards: pd.DataFrame, note: str = "Visits · % of that disease’s visits (Nandi residents, 2024–2025)", height: int = 280) -> None:
    with ui.chart_card(title, note=note):
        d = wards[wards.disease == disease].sort_values("visits", ascending=False).head(5) if not wards.empty else wards
        if _empty(d):
            return
        cols = [colour] + [_mix(colour, 0.7)] * (len(d) - 1)
        _plot(_hbar(d.ward, d.visits, cols, [f"{_n(v)} · {p:.1f}%" for v, p in zip(d.visits, d.pct_of_disease_visits)], height=height), f"ward-{disease}")


def _mix(hex_colour: str, a: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a:.2f})"


def _age_sums(df: pd.DataFrame, col: str, group="age_group") -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(0).groupby(df[group]).sum()


BUCKETS = [("Gonococcal", r"gonococc"), ("Pneumonia", r"pneumonia"), ("Gastritis", r"gastritis"), ("UTI", r"urinary tract"),
           ("Malaria", r"malaria|plasmodium"), ("Sepsis", r"sepsis"), ("Gastroenteritis", r"gastroenteritis|rotavirus"), ("Tonsillitis", r"tonsil"),
           ("Upper respiratory", r"upper respiratory"), ("Typhoid", r"typhoid"), ("TB", r"tubercul")]


def _bucket(dx: str) -> str:
    for name, pat in BUCKETS:
        if re.search(pat, str(dx), re.I):
            return name
    return str(dx)


# ============================================================ data
def _load() -> dict[str, pd.DataFrame]:
    names = ["infectious_patients", "malaria_species_year", "malaria_age", "malaria_rain", "pneumonia_age", "sepsis_types", "toddler_sepsis_codx",
             "puerperal_sepsis", "uti_sex_age", "uti_female_complicated", "uti_male_codx", "toddler_uti", "top_by_year", "top_by_age", "comorbidity",
             "toddler_anaemia", "gonococcal", "gonococcal_pregnancy", "ward_top5", "ward_infectious", "toddler_sepsis_age", "toddler_diarrhoea",
             "malaria_pregnancy", "malaria_anaemia", "anaemia_malaria", "artemisinin", "mild_sepsis_codx", "uti_codx"]
    out = {}
    for n in names:
        df = getattr(q, "q3_" + n)().copy()
        for c in df.columns:            # Snowflake NUMBERs can arrive as Decimal/object: make them plain numbers
            if df[c].dtype == object:
                try:
                    df[c] = pd.to_numeric(df[c])
                except (ValueError, TypeError):
                    pass
        out[n] = df
    return out


# ============================================================ A · burden at a glance
def _kpis(D) -> list[dict]:
    pts = D["infectious_patients"]
    allr = pts[pts.period.astype(str) == "All years"] if not pts.empty else pts
    cards = [dict(label="Patients with infectious disease", value=_n(allr.infectious_patients.iloc[0]) if len(allr) else "n/a",
                  sub=f"{allr.pct_of_patients.iloc[0]:.1f}% of {_n(allr.all_patients.iloc[0])} patients" if len(allr) else "", accent="blue")]
    rain, pn, sp, uti, gon, ma = D["malaria_rain"], D["pneumonia_age"], D["sepsis_types"], D["uti_sex_age"], D["gonococcal"], D["malaria_age"]

    def delta(df, col, ycol="visit_year"):
        if df.empty:
            return ""
        s = pd.to_numeric(df[col], errors="coerce").fillna(0).groupby(df[ycol]).sum()
        if 2024 in s.index and 2025 in s.index and s[2024]:
            d = (s[2025] / s[2024] - 1) * 100
            return f" · 2024→25 {ui.arrow_html(round(d))}"
        return ""

    cards.append(dict(label="Malaria", value=_n(_sum(ma, "total_malaria_visits")), sub="visits" + delta(rain, "malaria_visits"), accent="amber"))
    cards.append(dict(label="Pneumonia", value=_n(_sum(pn, "pneumonia_visits")), sub="visits" + delta(pn, "pneumonia_visits"), accent="blue"))
    sev = _sum(sp[sp.sepsis_type.isin(["Severe sepsis", "Septic shock"])], "visits") if not sp.empty else 0
    cards.append(dict(label="Sepsis", value=_n(_sum(sp, "visits")), sub=f"visits · {_n(sev)} severe or shock", accent="crit"))
    fem = _sum(uti[uti.sex == "F"], "visits") if not uti.empty else 0
    cards.append(dict(label="UTI", value=_n(_sum(uti, "visits")), sub=f"visits · {_pct(fem, _sum(uti, 'visits'))} women", accent="teal"))
    cards.append(dict(label="Gonococcal", value=_n(_sum(gon, "visits")), sub="visits" + delta(gon, "visits"), accent="amber"))
    return cards


def _top_conditions(D):
    t = D["top_by_year"]
    if t.empty:
        return pd.DataFrame()
    t = t.copy()
    t["bucket"] = t.diagnosis.map(_bucket)
    p = t.pivot_table(index="bucket", columns="visit_year", values="occurrences", aggfunc="sum").fillna(0)
    for y in (2024, 2025):
        if y not in p.columns:
            p[y] = 0
    p["tot"] = p[2024] + p[2025]
    return p.sort_values("tot", ascending=False).head(6)


def _dumbbell(p: pd.DataFrame) -> go.Figure:
    labels = list(p.index)
    xmax = float(max(p[2024].max(), p[2025].max(), 1))
    chg = [((b / a - 1) * 100) if a else None for a, b in zip(p[2024], p[2025])]
    col = [ui.TEXT_MUTED if c is None else RED if c <= -30 else AMBER if c < 0 else TEAL for c in chg]
    fig = go.Figure()
    for lab, a, b in zip(labels, p[2024], p[2025]):
        fig.add_trace(go.Scatter(x=[a, b], y=[lab, lab], mode="lines", line=dict(color="#CBD5E1", width=3), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=list(p[2024]), y=labels, mode="markers", name="2024", marker=dict(size=13, color=LGREY), hovertemplate="%{y} 2024: %{x:,}<extra></extra>"))
    fig.add_trace(go.Scatter(x=list(p[2025]), y=labels, mode="markers", name="2025", marker=dict(size=13, color=AMBER), hovertemplate="%{y} 2025: %{x:,}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[xmax * 1.22] * len(labels), y=labels, mode="text", text=[f"<b>{c:+.0f}%</b>" if c is not None else "" for c in chg], textfont=dict(size=13, color=col),
                             hoverinfo="skip", showlegend=False))
    ui._finish(fig, 330, margin=dict(l=10, r=10, t=34, b=10), legend=dict(orientation="h", x=0, y=1.1))
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.update_xaxes(range=[0, xmax * 1.32])
    return fig


def _who_gets_what(D) -> None:
    rows = []
    ma, pn, uti, sp, gon = D["malaria_age"], D["pneumonia_age"], D["uti_sex_age"], D["sepsis_types"], D["gonococcal"]
    series = {"Pneumonia": _age_sums(pn, "pneumonia_visits"), "UTI": _age_sums(uti, "visits"), "Sepsis": _age_sums(sp, "visits"),
              "Malaria": _age_sums(ma, "total_malaria_visits")}
    for name, s in series.items():
        tot = s.sum()
        rows.append((name, [100 * float(s.get(a, 0)) / tot if tot else 0 for a in AGES]))
    g = _age_sums(gon, "visits")
    gt = g.sum()
    bands = [None, None, "Adolescent (<18)", "Youth (18-24)", "Young Adult (25-34)", "Adult (35-44)", "Middle Age (45-54)", "55+"]
    rows.append(("Gonococcal", [None if b is None else 100 * float(g.get(b, 0)) / gt if gt else 0 for b in bands]))
    cols = [("", "")] + [(a, a) for a in AGE_SHORT]
    out = []
    for name, vals in rows:
        cells = [(name, "hl")]
        for v in vals:
            cells.append(("·", "", "color:#9CA3AF") if v is None else (f"{v:.0f}%", "", ui.heat(v, 0, 25)))
        out.append(cells)
    ui.html_table(cols, out, compact=True, widths=_eq(16, 8), min_width=420)
    _cap("Share of each condition’s visits by age group. Gonococcal under-18s are one band, shown under 13–17.")


def _where_stack(D) -> None:
    w = D["ward_infectious"]
    if _empty(w):
        return
    cols = ["malaria", "pneumonia", "sepsis", "uti", "gonococcal"]
    names = ["Malaria", "Pneumonia", "Sepsis", "UTI", "Gonococcal"]
    tot = w[cols].sum(axis=1).replace(0, 1)
    ser = [(n, list(100 * w[c] / tot), DISEASE_COL[n]) for c, n in zip(cols, names)]
    _plot(_sb(list(w.ward), ser, height=330, suffix="%"), "where-stack")


def _comorbidity(D) -> None:
    c = D["comorbidity"]
    if _empty(c):
        return
    cols = [("Condition", ""), ("Visits", ""), ("Anaemia", ""), ("High BP", ""), ("Diabetes", ""), ("Kidney", ""), ("Heart failure", ""), ("HIV", ""), ("Sickle cell", "")]
    keys = ["with_anaemia", "with_htn", "with_dm", "with_ckd", "with_heart_failure", "with_hiv", "with_sickle_cell"]
    rows = []
    for r in c.itertuples():
        tot = max(1, _f(r.total_visits))
        cells = [(r.infectious_condition, "hl"), _n(r.total_visits)]
        for k in keys:
            v = _f(getattr(r, k, 0))
            cells.append((f"{_n(v)}" if v else "0", "", ui.heat(100 * v / tot, 0, 5) if v else "color:#9CA3AF"))
        rows.append(cells)
    ui.html_table(cols, rows, compact=True, widths=_eq(21, 8), min_width=640)
    _cap("Visits that also carry each condition. Shade = share of that infection’s visits (darker = more).")


def _section_a(D) -> None:
    ui.section_header("Section A — Burden at a glance")
    left, right = st.columns([3, 2])
    p = _top_conditions(D)
    with left:
        with ui.chart_card("Top infectious conditions, 2024 vs 2025", note="Label = % change from 2024 to 2025."):
            if not _empty(p):
                _plot(_dumbbell(p), "dumbbell")
    with right:
        with ui.chart_card("Who gets what"):
            _who_gets_what(D)
    l2, r2 = st.columns(2)
    with l2:
        with ui.chart_card("Where they come from: top 5 wards", note="Share of each ward’s infectious visits, split by disease."):
            _where_stack(D)
    with r2:
        with ui.chart_card("What infections travel together"):
            _comorbidity(D)
    if not p.empty:
        lead = ", ".join(p.index[:3])
        drops = {k: (p.loc[k, 2025] / p.loc[k, 2024] - 1) * 100 for k in p.index if p.loc[k, 2024]}
        big = [k for k, v in drops.items() if v < -30]
        held = [k for k, v in drops.items() if v > -10]
        gon = drops.get("Gonococcal")
        txt = f"<strong>{lead} lead both years.</strong> "
        if big:
            txt += f"{' and '.join(big[:2])} fell by more than 30%"
            txt += f" while {' and '.join(held[:2])} held, so most of that fall is probably the 2025 export gap outside Emgwen. " if held else ". "
        if gon is not None and gon < -30:
            txt += f"<strong>Gonococcal infection fell {abs(gon):.0f}%</strong>, which the gap alone may not explain."
        ui.insight_bar(txt, "blue")


# ============================================================ B · children under 5
def _toddler_sepsis(D) -> pd.DataFrame:
    s = D["sepsis_types"]
    return s[s.age_group.astype(str).str.startswith("Toddler")] if not s.empty else s


BANDS = ["< 1 month", "1-11 months", "1 yr", "2 yr", "3 yr", "4 yr"]
BAND_LABEL = {"< 1 month": "< 1 month", "1-11 months": "1–11 months", "1 yr": "1 yr", "2 yr": "2 yrs", "3 yr": "3 yrs", "4 yr": "4 yrs"}


def _band_totals(D) -> pd.Series:
    t = D["toddler_sepsis_age"]
    if t.empty:
        return pd.Series(dtype=float)
    s = pd.to_numeric(t.visits, errors="coerce").fillna(0).groupby(t.age_band).sum()
    return s.reindex(BANDS).fillna(0)


def _codx(df: pd.DataFrame, pattern: str, col="occurrences") -> float:
    if df.empty:
        return 0.0
    m = df[df.co_diagnosis.astype(str).str.contains(pattern, case=False, regex=True)]
    return _sum(m, col)


def _chain(D) -> None:
    ts = _toddler_sepsis(D)
    allsep = _sum(D["sepsis_types"], "visits")
    tsep = _sum(ts, "visits")
    bands = _band_totals(D)
    peak = BAND_LABEL.get(bands.idxmax(), "") if len(bands) and bands.max() else ""
    cod = D["toddler_sepsis_codx"]
    urt = _codx(cod, "upper respiratory tract")
    ida = _codx(cod, "iron deficiency")
    spl = _codx(cod, "spleen")
    wd = D["ward_top5"]
    ks = wd[(wd.disease == "Toddler sepsis") & (wd.ward == "Kapsabet")] if not wd.empty else wd
    outside = 100 - float(ks.pct_of_disease_visits.iloc[0]) if len(ks) else None
    an = D["toddler_anaemia"]
    an_tot, an_sickle = _sum(an, "toddler_anaemia_visits"), _sum(an, "with_sickle_cell")
    art = D["artemisinin"]
    art_t = _sum(art[art.age_group.astype(str).str.startswith("Toddler")], "distinct_visits") if not art.empty else 0
    cards = [
        ("1 · They arrive with", _n(tsep), f"sepsis visits{', peaking at ' + peak + ' (' + _n(bands.max()) + ')' if peak else ''}"),
        ("2 · Carrying", _n(urt), f"upper respiratory infections, plus {_n(ida)} iron-deficiency anaemia and {_n(spl)} spleen abscess"),
        ("3 · From", f"{outside:.0f}%" if outside is not None else "n/a", "come from outside Kapsabet ward: " + ", ".join(wd[(wd.disease == "Toddler sepsis") & (wd.ward != "Kapsabet")].ward.head(4)) if len(ks) else ""),
        ("4 · Missed", f"{_n(an_sickle)} of {_n(an_tot)}", f"anaemic toddler visits carry a sickle-cell code, and {_n(art_t)} toddlers have resistant malaria"),
    ]
    html = "".join(f'<div class="cs"><span class="ct">{t}</span><b>{v}</b><span>{s}</span></div>' for t, v, s in cards)
    ui.insight_bar(f"<strong>Under-fives are {_pct(tsep, allsep)} of all sepsis visits.</strong> {_n(tsep)} of {_n(allsep)}. "
                   "Most arrive carrying a chest infection, often already anaemic, from outside Kapsabet ward, and the hospital sees them late.", "red")
    st.markdown(f'<div class="chain">{html}</div>', unsafe_allow_html=True)


def _bubbles(D) -> list[tuple[str, float]]:
    top = D["top_by_age"]
    items: list[tuple[str, float]] = []
    if not top.empty:
        t = top[top.age_group.astype(str).str.startswith("Toddler")].sort_values("occurrences", ascending=False).head(5)
        items = [(re.sub(r"(, unspecified| without specification of infectious agent|, site unspecified)", "", str(r.diagnosis)), _f(r.occurrences)) for r in t.itertuples()]
    names = " ".join(n.lower() for n, _ in items)
    extras = [("Anaemia", _sum(D["toddler_anaemia"], "toddler_anaemia_visits"), "anaemia"), ("UTI", _sum(D["toddler_uti"], "visits"), "urinary"),
              ("Malaria", _sum(D["malaria_age"][D["malaria_age"].age_group.astype(str).str.startswith("Toddler")], "total_malaria_visits") if not D["malaria_age"].empty else 0, "malaria")]
    for n, v, key in extras:
        if key not in names and v:
            items.append((n, v))
    return sorted(items, key=lambda x: -x[1])


def _bubble_chart(items) -> None:
    """Circle area = visits. Plain HTML so labels can never overlap, and the circles wrap on narrow screens."""
    pal = [BLUE, BLUE2, TEAL, AMBER, PINK, "#7C3AED", GREY, LGREY]
    mx = max([v for _, v in items] + [1])
    html = ""
    for (name, v), col in zip(items, pal):
        d = max(34, round(112 * (v / mx) ** 0.5))
        html += (f'<div class="bi"><div class="bc" style="width:{d}px;height:{d}px;background:{col};font-size:{13 if d > 60 else 11}px">{_n(v)}</div>'
                 f'<div class="bn">{name}</div></div>')
    st.markdown(f'<div class="bub">{html}</div>', unsafe_allow_html=True)


def _waffle(ts: pd.DataFrame) -> None:
    order = [("Mild sepsis", BLUE), ("Sepsis without septic shock", BLUE2), ("Severe sepsis", AMBER), ("Septic shock", RED)]
    counts = [_sum(ts[ts.sepsis_type == n], "visits") if not ts.empty else 0 for n, _ in order]
    tot = sum(counts) or 1
    raw = [100 * c / tot for c in counts]
    sq = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: -(raw[i] - sq[i]))[: 100 - sum(sq)]:
        sq[i] += 1
    sq = [max(s, 1) if c else 0 for s, c in zip(sq, counts)]
    cells = "".join(f'<i style="background:{col}"></i>' * n for (_, col), n in zip(order, sq))
    labels = "".join(f'<div style="margin-bottom:10px"><span style="font-size:24px;font-weight:700;color:{col}">{100 * c / tot:.0f}%</span><br><span style="font-size:12px;font-weight:600">{n.replace("Sepsis without septic shock", "without shock").replace(" sepsis", "").lower()} · {_n(c)}</span></div>'
                     for (n, col), c in zip(order, counts))
    st.markdown(f'<div style="display:flex;gap:28px;align-items:flex-start"><div class="waffle">{cells}</div><div>{labels}</div></div>', unsafe_allow_html=True)


def _anaemia_cards(D) -> None:
    an = D["toddler_anaemia"]
    tot, sickle, iso = _sum(an, "toddler_anaemia_visits"), _sum(an, "with_sickle_cell"), _sum(an, "isolated_anaemia")
    pct = {int(r.visit_year): _f(r.pct_isolated) for r in an.itertuples()} if not an.empty else {}
    left, mid = st.columns(2)
    with left:
        with ui.chart_card("Toddler anaemia: is it sickle cell?", note="Anaemic toddler visits by what else was coded. Overlaps possible."):
            if not _empty(an):
                rows = an[an.visit_year.isin([2024, 2025])].sort_values("visit_year")
                labels = [f"{int(r.visit_year)} · {_n(r.toddler_anaemia_visits)} visits" for r in rows.itertuples()]
                ser = [("Isolated", list(rows.isolated_anaemia), AMBER), ("With sepsis", list(rows.with_sepsis), RED),
                       ("With pneumonia", list(rows.with_pneumonia), BLUE), ("With malaria", list(rows.with_malaria), "#7C3AED")]
                _plot(_sb(labels, ser, height=250), "anaemia")
    with mid:
        with ui.chart_card("What the investigation found"):
            st.markdown(
                f'<div class="afya-stat" style="border-color:{RED}"><b style="color:{RED}">{_n(sickle)} of {_n(tot)}</b>toddler anaemia visits carry a sickle-cell code</div>'
                f'<div class="afya-stat" style="border-color:{RED}"><b style="color:{RED}">{_n(iso)}</b>isolated: no malaria, sepsis or pneumonia'
                f'{(" · " + format(pct.get(2024, 0), ".0f") + "% in 2024, " + format(pct.get(2025, 0), ".0f") + "% in 2025") if pct else ""}</div>'
                f'<div class="afya-stat" style="border-color:{RED}"><b style="color:{RED}">3–5 a year</b>is the minimum we’d expect to see coded, given 15–20% trait prevalence in parts of western Kenya</div>'
                '<div class="afya-note">Kenya child anaemia is 42.8% (World Bank, 2019), so these are the visible tip. Recommendation: routine sickle-cell screening for every anaemic toddler.</div>',
                unsafe_allow_html=True)


def _toddler_uti_card(D) -> None:
    uti = D["toddler_uti"]
    with ui.chart_card("Toddler UTI", note=f"{_n(_sum(uti, 'visits'))} visits"):
        if not _empty(uti):
            f = _sum(uti[uti.sex == "F"], "visits")
            m = _sum(uti[uti.sex == "M"], "visits")
            _plot(_single_bar("Toddlers", [("Female", f, TEAL), ("Male", m, BLUE)]), "tuti")
            _cap("Male toddler UTI is a red flag for reflux or posterior urethral valves. Fever is rarely coded, so imaging decisions are hard to audit.")


def _section_b(D) -> None:
    ui.section_header("Section B — Children under 5, the urgent story")
    _chain(D)
    ts = _toddler_sepsis(D)
    items = _bubbles(D)
    l, r = st.columns([2, 1])
    with l:
        with ui.chart_card("What toddlers come in with", note="Circle area = visits, 0–4 years. Conditions overlap, so don’t add them."):
            if items:
                _bubble_chart(items)
    with r:
        with ui.chart_card("Why sepsis leads", note="In toddlers, sepsis isn’t a late complication of something else. It is the #1 and #2 diagnosis."):
            if items:
                top3 = items[:3]
                _plot(_lolli([n for n, _ in top3], [v for _, v in top3], BLUE, [_n(v) for _, v in top3], height=230), "rank")
            top = D["top_by_age"]
            if not top.empty:
                ad = top[top.age_group.astype(str).str.startswith("Adolescent")]
                g = _sum(ad[ad.diagnosis.str.contains("gastritis", case=False)], "occurrences")
                p = _sum(ad[ad.diagnosis.str.contains("pneumonia", case=False)], "occurrences")
                if g > p > 0:
                    _cap("Adolescents are the turning point: gastritis overtakes pneumonia.")
    l, r = st.columns(2)
    bands = _band_totals(D)
    with l:
        with ui.chart_card("Toddler sepsis by year of age", note="All years. Sepsis peaks at 1–11 months, then falls with each year of age."):
            if not bands.empty:
                mx = bands.max()
                _plot(_columns([BAND_LABEL[b] for b in bands.index], list(bands.values), [RED if v == mx else "#D98C8C" for v in bands.values], [_n(v) for v in bands.values], height=300), "tage")
    with r:
        _ward_card("Where toddler sepsis comes from: top 5 wards", "Toddler sepsis", RED, D["ward_top5"], height=300)
    l, r = st.columns(2)
    with l:
        with ui.chart_card("Toddler sepsis on arrival", note=f"Share of {_n(_sum(ts, 'visits'))} toddler sepsis visits by type."):
            _waffle(ts)
    with r:
        with ui.chart_card("What toddler sepsis travels with", note="Co-diagnoses, visits (% of toddler sepsis co-diagnoses)"):
            cod = D["toddler_sepsis_codx"]
            if not _empty(cod):
                c = cod.head(5)
                _plot(_lolli(c.co_diagnosis, c.occurrences, BLUE, [f"{_n(o)} · {p:.1f}%" for o, p in zip(c.occurrences, c.pct)], height=270), "codx-tsep")
    cod = D["toddler_sepsis_codx"]
    if not cod.empty:
        urt, spl, ida, rota, asp = (_codx(cod, p) for p in ("upper respiratory tract", "spleen", "iron deficiency", "rotavirus", "aspergillosis"))
        ui.insight_bar(f"<strong>The pathway is legible: a chest infection that wasn’t cleared, in a child with weak reserves.</strong> Upper respiratory infection leads ({_n(urt)}). "
                       f"Spleen abscess ({_n(spl)}) means bacteria reached the bloodstream, iron-deficiency anaemia ({_n(ida)}) means the child had little reserve, and rotavirus ({_n(rota)}) is vaccine-preventable. "
                       f"{'The ' + _n(asp) + ' tonsillar aspergillosis cases point to underlying immunosuppression and need individual review.' if asp else ''}", "red")
    _anaemia_cards(D)
    l, r = st.columns(2)
    with l:
        _toddler_uti_card(D)
    with r:
        with ui.chart_card("Toddler malaria", note="Falciparum vs malariae in under-fives."):
            ma = D["malaria_age"]
            tm = ma[ma.age_group.astype(str).str.startswith("Toddler")] if not ma.empty else ma
            if not _empty(tm):
                pf, pm = _sum(tm, "p_falciparum"), _sum(tm, "p_malariae")
                _plot(_single_bar("Toddlers", [("P. falciparum", pf, RED), ("P. malariae", pm, BLUE)]), "tmal")
                art = D["artemisinin"]
                art_t = _sum(art[art.age_group.astype(str).str.startswith("Toddler")], "distinct_visits") if not art.empty else 0
                sp = D["malaria_species_year"]
                viv = sp[(sp.species == "P. vivax") & sp.visit_year.isin([2024, 2025])] if not sp.empty else sp
                st.markdown(f'<div class="afya-insight" style="background:#FFF5F5;border-left:3px solid {RED};margin:6px 0"><b>{_n(art_t)} toddler cases of artemisinin-resistant falciparum.</b> First-line treatment may fail in under-fives.</div>'
                            + (f'<div class="afya-insight" style="background:#FEF3E0;border-left:3px solid {AMBER};margin:6px 0"><b>P. vivax: {_n(_sum(viv, "visits"))} cases in 2024–25.</b> Rare in the highlands. Each needs a G6PD test before primaquine.</div>' if _sum(viv, "visits") else ""),
                            unsafe_allow_html=True)
    with ui.chart_card("Toddler diarrhoeal disease", note="Toddler diarrhoeal visits by year. The 2025 drop is partly the export gap."):
        dd = D["toddler_diarrhoea"]
        if not _empty(dd):
            yrs = [f"{int(y)}{'*' if int(y) in (2023, 2026) else ''}" for y in dd.visit_year]
            _plot(_stack_cols(yrs, [("Rotavirus", list(dd.rotavirus), BLUE), ("Campylobacter", list(dd.campylobacter), AMBER), ("Other gastroenteritis", list(dd.gastroenteritis_other), LGREY)], height=300), "diar")
            _cap(f"Rotavirus is {_pct(_sum(dd, 'rotavirus'), _sum(dd, 'toddler_visits'))} of {_n(_sum(dd, 'toddler_visits'))} toddler diarrhoeal visits. * partial year.")
    bands = _band_totals(D)
    dd = D["toddler_diarrhoea"]
    an = D["toddler_anaemia"]
    items = [
        ("Catch it earlier", f"Sepsis peaks at {BAND_LABEL.get(bands.idxmax(), '1–11 months') if len(bands) and bands.max() else '1–11 months'}. A chest infection that isn’t cleared in a baby is the lead-in. Point IMCI follow-up at the outer wards."),
        ("Prevent what is preventable", f"Rotavirus is {_pct(_sum(dd, 'rotavirus'), _sum(dd, 'toddler_visits'))} of toddler diarrhoea ({_n(_sum(dd, 'rotavirus'))} visits) and has a vaccine. Malaria is mostly falciparum, with resistant cases to escalate."),
        ("Test for sickle cell", f"{_n(_sum(an, 'isolated_anaemia'))} anaemic toddler visits have no malaria, sepsis or pneumonia alongside. Screen every anaemic under-five."),
    ]
    st.markdown('<div class="afya-why red">' + "".join(f"<div><b>{t}</b>{b}</div>" for t, b in items) + "</div>", unsafe_allow_html=True)


# ============================================================ C · malaria
def _species_year(D) -> go.Figure | None:
    sp = D["malaria_species_year"]
    if sp.empty:
        return None
    yrs = sorted(sp.visit_year.unique())
    g = lambda names: [_sum(sp[(sp.visit_year == y) & sp.species.isin(names)], "visits") for y in yrs]  # noqa: E731
    lab = [f"{int(y)}{'*' if int(y) in (2023, 2026) else ''}" for y in yrs]
    fig = _stack_cols(lab, [("P. falciparum", g(["P. falciparum", "Artemisinin-resistant Pf"]), RED), ("P. malariae", g(["P. malariae"]), BLUE),
                            ("Other / unspecified", g(["P. vivax", "Unspecified"]), LGREY)], height=300)
    return fig


def _butterfly(D) -> go.Figure | None:
    ma = D["malaria_age"]
    if ma.empty:
        return None
    ma = ma.set_index("age_group").reindex([a for a in AGES if a in set(ma.age_group)])
    labels = [AGE_SHORT[AGES.index(a)] for a in ma.index]
    pf, pm = pd.to_numeric(ma.p_falciparum).fillna(0), pd.to_numeric(ma.p_malariae).fillna(0)
    art = pd.to_numeric(ma.artemisinin_resistant).fillna(0)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=-pf, orientation="h", name="P. falciparum", marker_color=RED, text=[_n(v) for v in pf], textposition="outside", customdata=pf,
                         hovertemplate="%{y} · P. falciparum: %{customdata:,}<extra></extra>", cliponaxis=False))
    fig.add_trace(go.Bar(y=labels, x=pm, orientation="h", name="P. malariae", marker_color=BLUE, text=[_n(v) for v in pm], textposition="outside",
                         hovertemplate="%{y} · P. malariae: %{x:,}<extra></extra>", cliponaxis=False))
    ar = [(l, -(p + 22 + 12 * k)) for l, p, a in zip(labels, pf, art) for k in range(int(a))]
    if ar:
        fig.add_trace(go.Scatter(y=[a[0] for a in ar], x=[a[1] for a in ar], mode="markers", marker=dict(symbol="diamond", size=11, color=AMBER), name="◆ artemisinin-resistant case",
                                 hoverinfo="skip"))
    ui._finish(fig, 340, barmode="overlay")
    mx = max(list(pf) + list(pm) + [1]) * 1.25
    fig.update_xaxes(range=[-mx, mx], showticklabels=False, showgrid=False)
    fig.update_yaxes(autorange="reversed")
    return fig


def _malaria_pregnancy(D) -> None:
    mp = D["malaria_pregnancy"]
    if _empty(mp):
        return
    g = mp.groupby("age_group").agg(f=("female_malaria_visits", "sum"), p=("with_pregnancy", "sum")).reindex([a for a in AGES if a in set(mp.age_group)])
    labels = [AGE_SHORT[AGES.index(a)] for a in g.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=list(g.f), orientation="h", marker_color="#F1C48A", text=[f"{_n(f)}" + (f" · {_n(p)} pregnant" if p else "") for f, p in zip(g.f, g.p)],
                         textposition="outside", cliponaxis=False, hoverinfo="skip", name="Female malaria visits"))
    fig.add_trace(go.Bar(y=labels, x=[max(p * 6, 0) if p else 0 for p in g.p], orientation="h", marker_color=RED, name="Pregnancy-coded (bar not to scale)", hoverinfo="skip"))
    ui._finish(fig, 260, barmode="overlay", showlegend=False, margin=dict(l=10, r=90, t=6, b=6))
    fig.update_xaxes(range=[0, float(g.f.max() or 1) * 1.45])
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(visible=False, showgrid=False)
    _plot(fig, "mpreg")
    tot, preg, pf = _sum(mp, "female_malaria_visits"), _sum(mp, "with_pregnancy"), _sum(mp, "pregnant_falciparum")
    st.markdown(f'<span class="afya-limit">Only {_n(preg)} of {_n(tot)} female malaria visits ({_pct(preg, tot, 1)}) carry a pregnancy code, {_n(pf)} of them falciparum. Probably under-coded: the pregnancy sits on the obstetric visit, not the malaria one.</span>', unsafe_allow_html=True)


def _donut_pair(a_pct, a_lab, b_pct, b_lab) -> go.Figure:
    fig = go.Figure()
    for (p, col), dom in zip(((a_pct, RED), (b_pct, AMBER)), ([0.0, 0.46], [0.54, 1.0])):
        fig.add_trace(go.Pie(values=[p, max(100 - p, 0)], hole=0.7, marker=dict(colors=[col, "#E5E7EB"]), textinfo="none", sort=False, direction="clockwise",
                             hoverinfo="skip", showlegend=False, domain=dict(x=dom, y=[0, 1])))
    ui._finish(fig, 230, showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
               annotations=[dict(text=f"<b>{a_pct:.1f}%</b><br><span style='font-size:10.5px;color:{ui.TEXT_MUTED}'>{a_lab}</span>", x=0.23, y=0.5, xanchor="center", yanchor="middle", showarrow=False, font=dict(size=20)),
                            dict(text=f"<b>{b_pct:.1f}%</b><br><span style='font-size:10.5px;color:{ui.TEXT_MUTED}'>{b_lab}</span>", x=0.77, y=0.5, xanchor="center", yanchor="middle", showarrow=False, font=dict(size=20))])
    return fig


def _malaria_anaemia(D) -> None:
    ma, am = D["malaria_anaemia"], D["anaemia_malaria"]
    if ma.empty or am.empty:
        _empty(pd.DataFrame())
        return
    t1 = ma[ma.age_group.astype(str).str.startswith("Toddler")]
    t2 = am[am.age_group.astype(str).str.startswith("Toddler")]
    a = _f(t1.pct_malaria_with_anaemia.iloc[0]) if len(t1) else 0
    b = _f(t2.pct_anaemia_with_malaria.iloc[0]) if len(t2) else 0
    _plot(_donut_pair(a, "toddler malaria<br>with anaemia", b, "toddler anaemia<br>with malaria"), "manae")
    tm, tw = _sum(ma, "malaria_visits"), _sum(ma, "malaria_with_anaemia")
    _cap(f"Toddlers: {_n(_sum(t1, 'malaria_with_anaemia'))} of {_n(_sum(t1, 'malaria_visits'))} malaria visits carry anaemia; {_n(_sum(t2, 'anaemia_with_malaria'))} of {_n(_sum(t2, 'anaemia_visits'))} anaemia visits carry malaria.")
    child = am[am.age_group.astype(str).str.startswith("Child")]
    _cap(f"Overall only {_n(tw)} of {_n(tm)} malaria visits ({_pct(tw, tm, 1)}) carry anaemia. The link is strongest in toddlers and in anaemic children aged 5–12 "
               f"({_f(child.pct_anaemia_with_malaria.iloc[0]) if len(child) else 0:.1f}% have malaria), but still weak. Anaemia is probably untested on malaria visits, or not coded.")


def _artemisinin(D) -> None:
    ar = D["artemisinin"]
    yrs = [2023, 2024, 2025, 2026]
    vals = [int(_sum(ar[ar.visit_year == y], "distinct_visits")) if not ar.empty else 0 for y in yrs]
    fig = _columns([f"{y}{'*' if y in (2023, 2026) else ''}" for y in yrs], vals, [RED] * 4, [str(v) for v in vals], height=330)
    _plot(fig, "artem")


def _artemisinin_insight(D) -> None:
    ar = D["artemisinin"]
    yrs = [2023, 2024, 2025, 2026]
    vals = [int(_sum(ar[ar.visit_year == y], "distinct_visits")) if not ar.empty else 0 for y in yrs]
    if not ar.empty:
        ages = ar.groupby("age_group").distinct_visits.sum()
        order = [a for a in AGES if a in ages.index]
        lab = ", ".join(f"{int(ages[a])} {AGE_SHORT[AGES.index(a)]}" for a in order)
        a24, a25 = vals[1], vals[2]
        mult = f"{a25 / a24:.0f}×" if a24 else "n/a"
        ui.insight_bar(f"<strong>Resistant malaria rose {mult} in a year: {a24} visits in 2024, {a25} in 2025.</strong> Artemisinin is first-line treatment for falciparum, and the {sum(vals)} cases span ages ({lab}). "
                       "Falciparum is also gaining share and moving to younger patients. This needs clinical validation, escalation to county health management, and a report to KEMRI and the national malaria programme.", "red")


def _rain(D) -> None:
    r = D["malaria_rain"]
    if _empty(r):
        return
    r = r[r.visit_year.isin([2023, 2024, 2025])]
    yrs = [str(int(y)) for y in r.visit_year]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.18, subplot_titles=("Rainfall, Eldoret station (mm per year)", "Malaria visits per 1,000 OPD visits"))
    rain = [_f(v) for v in r.annual_rainfall_mm]
    fig.add_trace(go.Bar(x=yrs, y=[v if v else 0 for v in rain], marker_color=BLUE, text=[f"{v:,.0f} mm" if v else "no data" for v in rain], textposition="outside", cliponaxis=False,
                         hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Bar(x=yrs, y=list(r.malaria_per_1000_opd), marker_color=RED, text=[f"{_f(v):.2f}" for v in r.malaria_per_1000_opd], textposition="outside", cliponaxis=False,
                         hoverinfo="skip"), row=2, col=1)
    ui._finish(fig, 440, showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
    fig.update_yaxes(showgrid=True, gridcolor=ui.SURFACE_2, rangemode="tozero")
    fig.update_xaxes(showgrid=False, type="category")
    fig.update_annotations(font=dict(size=12, color=ui.TEXT), xanchor="left", x=0)
    _plot(fig, "rain")


def _rain_insight(D) -> None:
    r_ = D["malaria_rain"]
    sp = D["malaria_species_year"]
    if r_.empty or sp.empty:
        return

    def v(y, s):
        return _sum(sp[(sp.visit_year == y) & (sp.species == s)], "visits")
    ma = D["malaria_age"]
    t = ma[ma.age_group.astype(str).str.startswith("Toddler")]
    r23, r24 = r_[r_.visit_year == 2023], r_[r_.visit_year == 2024]
    rain_txt = ""
    if len(r23) and len(r24) and _f(r23.annual_rainfall_mm.iloc[0]) and _f(r24.annual_rainfall_mm.iloc[0]):
        a, b = _f(r23.annual_rainfall_mm.iloc[0]), _f(r24.annual_rainfall_mm.iloc[0])
        rain_txt = (f" 2024 had {abs(b / a - 1) * 100:.0f}% {'less' if b < a else 'more'} rain than 2023 ({b:,.0f} vs {a:,.0f} mm) and a "
                    f"{'higher' if _f(r24.malaria_per_1000_opd.iloc[0]) > _f(r23.malaria_per_1000_opd.iloc[0]) else 'lower'} malaria rate ({_f(r24.malaria_per_1000_opd.iloc[0]):.2f} vs {_f(r23.malaria_per_1000_opd.iloc[0]):.2f}).")
    pf25, pm25 = v(2025, "P. falciparum"), v(2025, "P. malariae")
    r25 = r_[r_.visit_year == 2025]
    no_rain_25 = (not len(r25)) or not _f(r25.annual_rainfall_mm.iloc[0])
    ui.insight_bar(f"<strong>The danger is the species, not the volume, and rain doesn’t explain it.</strong> Falciparum {'overtook' if pf25 > pm25 else 'trails'} malariae in 2025 ({_n(pf25)} vs {_n(pm25)}) "
                   f"and dominates under-fives ({_n(_sum(t, 'p_falciparum'))} of {_n(_sum(t, 'total_malaria_visits'))} visits).{rain_txt} Resistance in children would be a KEMRI and national malaria programme report, not a dashboard note. "
                   + ('<span class="afya-limit">Limitation: the rainfall station has been offline since Dec 2024, so there is no 2025 rainfall. A monthly correlation needs visit dates and a resumed feed.</span>' if no_rain_25 else
                      '<span class="afya-limit">Check the rainfall feed: annual totals this high suggest more than one station or duplicated days are being summed.</span>' if max([_f(v) for v in r_.annual_rainfall_mm] + [0]) > 3000 else ""), "red")


def _section_c(D) -> None:
    ui.section_header("Section C — Malaria")
    l, r = st.columns(2)
    with l:
        with ui.chart_card("Species by year", note="* partial year. Falciparum overtook malariae in 2025."):
            fig = _species_year(D)
            if fig:
                _plot(fig, "species")
    with r:
        with ui.chart_card("Species by age group", note="◆ = artemisinin-resistant case"):
            fig = _butterfly(D)
            if fig:
                _plot(fig, "butterfly")
    l, r = st.columns(2)
    with l:
        with ui.chart_card("Youth 18–24: how many are pregnant?", note="Women, all years, by age. Pregnancy-coded visits are labelled in red."):
            _malaria_pregnancy(D)
    with r:
        with ui.chart_card("Malaria and anaemia", note="Falciparum destroys red cells, so we’d expect the two to travel together."):
            _malaria_anaemia(D)
    l, r = st.columns([3, 2])
    with l:
        with ui.chart_card("Artemisinin resistance over time", note="Resistant falciparum cases per year (the data has no finer date). A rise is the signal to escalate. * partial year."):
            _artemisinin(D)
    with r:
        _ward_card("Where malaria comes from: top 5 wards", "Malaria", AMBER, D["ward_top5"], height=330)
    _artemisinin_insight(D)
    with ui.chart_card("Rainfall and malaria: what the season predicts", note="Two stacked panels on a shared year axis, not a dual axis: rain first, then malaria rate."):
        _rain(D)
    _rain_insight(D)


# ============================================================ D · pneumonia
def _section_d(D) -> None:
    ui.section_header("Section D — Pneumonia")
    pn = D["pneumonia_age"]
    tot = _sum(pn, "pneumonia_visits")
    st.markdown(f'<div class="afya-link"><span><b>Pneumonia is the largest infectious burden ({_n(tot)} visits).</b> The full respiratory picture, including asthma and COPD, sits on the Respiratory tab.</span>'
                '<a href="?tab=tab_resp" target="_self">Open Respiratory →</a></div>', unsafe_allow_html=True)
    s = _age_sums(pn, "pneumonia_visits").reindex(AGES).fillna(0)
    share = [100 * v / tot if tot else 0 for v in s.values]
    top2 = sorted(range(len(share)), key=lambda i: -share[i])[:2]
    l, r = st.columns([3, 2])
    with l:
        with ui.chart_card("Pneumonia by age group", note="% of pneumonia visits. Red = the two peaks."):
            if not _empty(pn):
                _plot(_columns(AGE_SHORT, [round(v, 1) for v in share], [RED if i in top2 else BLUE2 for i in range(8)], [f"{v:.1f}%" for v in share], height=340), "pn-age")
    with r:
        with ui.chart_card("What pneumonia travels with", note="Visits that also carry each condition, 2024–2025"):
            if not pn.empty:
                items = [("Anaemia", _sum(pn, "with_anaemia")), ("Diabetes", _sum(pn, "with_dm")), ("Heart failure", _sum(pn, "with_heart_failure")),
                         ("Asthma / COPD", _sum(pn, "with_respiratory_ncd")), ("Malnutrition", _sum(pn, "with_malnutrition")), ("HIV", _sum(pn, "with_hiv"))]
                items.sort(key=lambda x: -x[1])
                _plot(_lolli([n for n, _ in items], [v for _, v in items], BLUE, [_n(v) for _, v in items], height=340), "pn-codx")
                if _sum(pn, "with_hiv") == 0:
                    st.markdown('<span class="afya-limit">0 HIV co-diagnoses: a documentation gap, not absence.</span>', unsafe_allow_html=True)
    l, r = st.columns([3, 2])
    with l:
        with ui.chart_card("What is behind the peaks", note="Severity and co-diagnoses by age group, 2024–2025. Darker = higher within the column. ★ = peak age groups."):
            if not pn.empty:
                g = pn.groupby("age_group")
                sev = (g.severe_problem.sum() / g.pneumonia_visits.sum() * 100).reindex(AGES)
                cols = [("Age", ""), ("Visits", ""), ("Severe %", ""), ("Diabetes", ""), ("Heart failure", ""), ("Asthma/COPD", ""), ("Anaemia", ""), ("Malnutrition", "")]
                keys = ["with_dm", "with_heart_failure", "with_respiratory_ncd", "with_anaemia", "with_malnutrition"]
                sums = {k: _age_sums(pn, k).reindex(AGES).fillna(0) for k in keys}
                rows = []
                for i, a in enumerate(AGES):
                    cells = [(AGE_SHORT[i] + (" ★" if i in top2 else ""), "hl"), _n(s[a]), (f"{sev[a]:.1f}%", "", ui.heat(sev[a], 0, max(20, sev.max()), RED))]
                    for k in keys:
                        v = sums[k][a]
                        cells.append((_n(v), "", ui.heat(v, 0, sums[k].max() or 1, BLUE) if v else "color:#9CA3AF"))
                    rows.append(cells)
                ui.html_table(cols, rows, compact=True, widths=_eq(13, 7), min_width=560)
    with r:
        _ward_card("Where pneumonia comes from: top 5 wards", "Pneumonia", BLUE, D["ward_top5"], height=380)
    if not pn.empty:
        names = lambda i: AGE_SHORT[i]  # noqa: E731
        old = AGES.index("55+")
        dm55, hf55 = _sum(pn[pn.age_group == "55+"], "with_dm"), _sum(pn[pn.age_group == "55+"], "with_heart_failure")
        sev_hi = sev.idxmax()
        ui.insight_bar(f"<strong>Two peaks, two stories.</strong> {names(top2[0])} ({share[top2[0]]:.1f}%) and {names(top2[1])} ({share[top2[1]]:.1f}%) carry the most visits. "
                       f"At 55+ the co-diagnoses are the heaviest: {_n(dm55)} diabetes and {_n(hf55)} heart failure. "
                       f"The highest severity is in the {AGE_SHORT[AGES.index(sev_hi)]} group ({sev[sev_hi]:.1f}% severe), which fits late care-seeking in working-age adults.", "amber")


# ============================================================ E · sepsis
def _section_e(D) -> None:
    ui.section_header("Section E — Sepsis")
    sp = D["sepsis_types"]
    tot = _sum(sp, "visits")
    l, r = st.columns([3, 2])
    with l:
        with ui.chart_card("Sepsis by type", note=f"Share of {_n(tot)} sepsis visits (corrected classification)"):
            if not _empty(sp):
                t = pd.to_numeric(sp.visits, errors="coerce").fillna(0).groupby(sp.sepsis_type).sum().sort_values(ascending=False)
                colours = {"Mild sepsis": BLUE, "Sepsis without septic shock": BLUE2, "Severe sepsis": AMBER, "Septic shock": RED, "Puerperal sepsis": PINK}
                _plot(_hbar(t.index, t.values, [colours.get(i, GREY) for i in t.index], [f"{_n(v)} · {100 * v / tot:.1f}%" for v in t.values], height=300), "sep-types")
    with r:
        _ward_card("Where sepsis comes from: top 5 wards", "Sepsis", RED, D["ward_top5"], height=300)
    l, r = st.columns(2)
    with l:
        with ui.chart_card("What accompanies mild sepsis", note="Co-diagnoses on mild sepsis visits, with how many are toddlers."):
            m = D["mild_sepsis_codx"]
            if not _empty(m):
                m = m.head(6)
                _plot(_lolli(m.co_diagnosis.str.slice(0, 40), m.visits, BLUE, [f"{_n(v)} · {_n(t)} toddlers" for v, t in zip(m.visits, m.toddler_visits)], height=290), "mild-codx")
                odd = m_all = D["mild_sepsis_codx"]
                if not odd.empty and odd.co_diagnosis.str.contains("aortic aneurysm", case=False).any():
                    st.markdown('<span class="afya-limit">Aortic aneurysm and essential hypertension appear on mild sepsis visits with no toddlers: a coding error, logged on the <a href="?tab=tab_dataquality" target="_self" style="color:#2E8B3A;font-weight:700;text-decoration:none">Data quality tab</a>.</span>', unsafe_allow_html=True)
    with r:
        with ui.chart_card("Sepsis and anaemia", note="Sepsis visits that carry an anaemia diagnosis"):
            c = D["comorbidity"]
            s = c[c.infectious_condition == "Sepsis"] if not c.empty else c
            if not _empty(s):
                a, tdl = _f(s.with_anaemia.iloc[0]), _f(s.anaemia_in_toddlers.iloc[0])
                fig = go.Figure(go.Pie(values=[tdl, max(a - tdl, 0)], hole=0.68, marker=dict(colors=[AMBER, "#CBD5E1"]), textinfo="none", sort=False, hoverinfo="skip", showlegend=False))
                ui._finish(fig, 230, showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
                           annotations=[dict(text=f"<b>{_n(a)}</b><br>sepsis visits<br>with anaemia", showarrow=False, font=dict(size=13))])
                _plot(fig, "sanae")
                _cap(f"{_n(tdl)} are toddlers; {_n(max(a - tdl, 0))} are other ages.")
    pu = D["puerperal_sepsis"]
    with ui.chart_card("Puerperal sepsis by year", note="* partial year"):
        if not _empty(pu):
            y = pd.to_numeric(pu.visits, errors="coerce").fillna(0).groupby(pu.visit_year).sum()
            fig = go.Figure(go.Scatter(x=[f"{int(i)}{'*' if int(i) in (2023, 2026) else ''}" for i in y.index], y=list(y.values), mode="lines+markers+text", line=dict(color=RED, width=3),
                                       marker=dict(size=9, color=RED), fill="tozeroy", fillcolor="rgba(163,45,45,.12)", text=[_n(v) for v in y.values], textposition="top center", cliponaxis=False))
            ui._finish(fig, 260, showlegend=False, margin=dict(l=10, r=10, t=24, b=10))
            fig.update_yaxes(showgrid=True, gridcolor=ui.SURFACE_2, rangemode="tozero")
            fig.update_xaxes(type="category")
            _plot(fig, "puerp")
    if not pu.empty:
        ad = pu[pu.age_group.astype(str).str.startswith("Adolescent")]
        sev = _sum(pu, "severe_cases")
        ui.insight_bar(f"<strong>All cases are women.</strong> {_n(sev)} severe case(s) were in girls aged 13–17"
                       f"{', a teenage-pregnancy complication and a safeguarding signal' if _sum(ad, 'severe_cases') else ''}. "
                       "A fall after 2024 is the export gap, not better maternal outcomes.", "red")
    if tot:
        tsep = _sum(_toddler_sepsis(D), "visits")
        m = D["mild_sepsis_codx"]
        top = m.iloc[0] if not m.empty else None
        c = D["comorbidity"]
        s = c[c.infectious_condition == "Sepsis"] if not c.empty else c
        ui.insight_bar(f"<strong>Most sepsis is mild, and it rides on a chest infection.</strong> "
                       + (f"{top.co_diagnosis} is the top partner of mild sepsis ({_n(top.visits)} visits, {_n(top.toddler_visits)} of them toddlers). " if top is not None else "")
                       + (f"{_n(s.anaemia_in_toddlers.iloc[0])} of the {_n(s.with_anaemia.iloc[0])} sepsis visits with anaemia are toddlers." if len(s) else ""), "red")


# ============================================================ F · UTI
def _uti_cell(u1: pd.DataFrame, sex: str, age: str | None, col: str) -> float:
    if u1.empty:
        return 0.0
    m = u1[u1.sex == sex]
    if age:
        m = m[m.age_group == age]
    return _sum(m, col)


def _flow_row(a: tuple[str, str], b: tuple[str, str, str], c: tuple[str, str, str], colour: str) -> str:
    """a = (group, visits); b = (label, number, note); c = (outcome, number, note)."""
    return (f'<div class="fb"><span class="t">{a[0]}</span><span class="s">{a[1]}</span></div><div class="ar">→</div>'
            f'<div class="fb"><span class="s" style="margin:0 0 2px">{b[0]}</span><span class="n">{b[1]}</span><span class="s">{b[2]}</span></div><div class="ar">→</div>'
            f'<div class="fb end" style="background:{colour}"><span class="t">{c[0]}</span><span class="n">{c[1]}</span><span class="s">{c[2]}</span></div>')


def _uti_paths(D) -> None:
    u1, fc, tu = D["uti_codx"], D["uti_female_complicated"], D["toddler_uti"]
    ya, old = "Young Adult (25-34)", "55+"
    ya_v = _uti_cell(u1, "F", ya, "uti_visits")
    old_row = fc[fc.age_group == old] if not fc.empty else fc
    comp = _f(old_row.pct_complicated.iloc[0]) if len(old_row) else 0
    men = _sum(u1[u1.sex == "M"], "uti_visits") if not u1.empty else 0
    tod = _sum(tu, "visits")
    boys = _sum(tu[tu.sex == "M"], "visits") if not tu.empty else 0
    dm55, ckd55 = _uti_cell(u1, "F", old, "with_diabetes"), _uti_cell(u1, "F", old, "with_ckd")
    gc_m, pr_m = _uti_cell(u1, "M", None, "with_gonococcal"), _uti_cell(u1, "M", None, "with_prostate")
    rows = [
        _flow_row(("Women 25–34", f"{_n(ya_v)} UTI visits"), ("Pregnancy co-coded", _n(_uti_cell(u1, "F", ya, "with_pregnancy")), "visits"), ("Went on to sepsis", _n(_uti_cell(u1, "F", ya, "with_sepsis")), "visits"), TEAL),
        _flow_row(("Women 55+", f"{_n(_uti_cell(u1, 'F', old, 'uti_visits'))} UTI visits"), ("Diabetes or kidney disease", _n(dm55 + ckd55), f"{_n(dm55)} diabetes · {_n(ckd55)} CKD"), ("Complicated UTI", f"{comp:.1f}%", "of visits"), AMBER),
        _flow_row(("Men", f"{_n(men)} UTI visits"), ("Gonococcal or prostate", _n(gc_m + pr_m), f"{_n(gc_m)} gonococcal · {_n(pr_m)} prostate"), ("Went on to sepsis", _n(_uti_cell(u1, "M", None, "with_sepsis")), "visits"), BLUE),
        _flow_row(("Toddlers", f"{_n(tod)} UTI visits"), ("Boys", _n(boys), f"{_pct(boys, tod)} of toddler UTI"), ("Sepsis · renal imaging", _n(_sum(tu, "progressed_sepsis")), "visits"), RED),
    ]
    st.markdown('<div class="afya-flow"><div class="fh">Who</div><div></div><div class="fh">What travels with it</div><div></div><div class="fh">Where it ends</div>' + "".join(rows) + "</div>", unsafe_allow_html=True)
    _cap("Gonococcal in men likely means an STI presenting as a UTI, not a true ascending infection.")


def _uti_matrix(D) -> None:
    u1 = D["uti_codx"]
    if _empty(u1):
        return
    nf, nm = _uti_cell(u1, "F", None, "uti_visits"), _uti_cell(u1, "M", None, "uti_visits")
    spec = [("Pregnancy", "with_pregnancy", True, False), ("Gonococcal", "with_gonococcal", True, True), ("Diabetes", "with_diabetes", True, True), ("Anaemia", "with_anaemia", True, True),
            ("Sepsis", "with_sepsis", True, True), ("Heart failure", "with_heart_failure", True, True), ("Prostate hyperplasia", "with_prostate", False, True),
            ("Chronic kidney disease", "with_ckd", True, True), ("HIV", "with_hiv", True, True)]
    rows = []
    for name, col, f_ok, m_ok in spec:
        cells = [(name, "hl")]
        for sex, ok, n in (("F", f_ok, nf), ("M", m_ok, nm)):
            if not ok:
                cells.append(("n/a", "", "color:#9CA3AF;background:#F5F6FA"))
                continue
            v = _uti_cell(u1, sex, None, col)
            pc = 100 * v / n if n else 0
            cells.append((f"{_n(v)} · {pc:.1f}%" if v else "0", "", ui.heat(pc, 0, 3, TEAL) if v else "color:#9CA3AF"))
        rows.append(cells)
    ui.html_table([("", ""), (f"Women ({_n(nf)})", ""), (f"Men ({_n(nm)})", "")], rows, compact=True, widths=["34%", "33%", "33%"])
    if _uti_cell(u1, "F", None, "with_hiv") + _uti_cell(u1, "M", None, "with_hiv") == 0:
        st.markdown('<span class="afya-limit">HIV: zero recorded in both sexes. A documentation gap rather than absence.</span>', unsafe_allow_html=True)


def _section_f(D) -> None:
    ui.section_header("Section F — Urinary tract infection")
    uti, fc = D["uti_sex_age"], D["uti_female_complicated"]
    l, r = st.columns(2)
    with l:
        with ui.chart_card("Who it hits", note="Height = % female. Bubble area = visits."):
            if not _empty(uti):
                g = uti.pivot_table(index="age_group", columns="sex", values="visits", aggfunc="sum").reindex(AGES).fillna(0)
                tot = g.sum(axis=1)
                fem = 100 * g.get("F", 0) / tot.replace(0, 1)
                top3 = set(tot.sort_values(ascending=False).index[:3])
                fig = go.Figure(go.Scatter(x=AGE_SHORT, y=list(fem), mode="markers+text", text=[f"{p:.0f}%" if a in top3 else "" for a, p in zip(AGES, fem)], textposition="top center",
                                           marker=dict(size=list(tot), sizemode="area", sizeref=2.0 * max(list(tot) + [1]) / (46 ** 2), sizemin=6, color=TEAL, opacity=0.7),
                                           hovertemplate="%{x}: %{y:.0f}% female<extra></extra>", cliponaxis=False))
                ui._finish(fig, 300, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
                fig.update_yaxes(showgrid=True, gridcolor=ui.SURFACE_2, ticksuffix="%", range=[40, 100])
                _plot(fig, "uti-who")
    with r:
        with ui.chart_card("Complicated UTI by age, women", note="% of female UTI visits with pregnancy, diabetes, kidney disease or a catheter"):
            if not _empty(fc):
                fcg = fc.set_index("age_group").reindex([a for a in AGES if a in set(fc.age_group)])
                vals = [_f(v) for v in fcg.pct_complicated]
                _plot(_columns([AGE_SHORT[AGES.index(a)] for a in fcg.index], vals, [RED if v >= 5 else BLUE2 for v in vals], [f"{v:.1f}%" for v in vals], height=300, ysuffix="%"), "uti-comp")
    with ui.chart_card("Where UTI leads: four pathways", note="Each group has a different mechanism."):
        _uti_paths(D)
    l, r = st.columns(2)
    with l:
        with ui.chart_card("UTI against other conditions", note="UTI visits carrying each diagnosis, with % of that sex’s UTI visits."):
            _uti_matrix(D)
    with r:
        _ward_card("Where UTI comes from: top 5 wards", "UTI", TEAL, D["ward_top5"], height=330)
    u1 = D["uti_codx"]
    if not u1.empty:
        yp = _uti_cell(u1, "F", "Young Adult (25-34)", "with_pregnancy")
        ys = _uti_cell(u1, "F", "Young Adult (25-34)", "with_sepsis")
        ui.insight_bar(f"<strong>UTI is four different problems.</strong> In women 25–34 it is a pregnancy risk: {_n(yp)} co-coded, {_n(ys)} progressed to sepsis. "
                       f"In women 55+ it is a diabetes risk ({_n(_uti_cell(u1, 'F', '55+', 'with_diabetes'))} co-coded). In men it is partly an STI misclassified as UTI "
                       f"({_n(_uti_cell(u1, 'M', None, 'with_gonococcal'))} gonococcal) and partly prostate ({_n(_uti_cell(u1, 'M', None, 'with_prostate'))}). In toddler boys it is a structural red flag.", "teal")


# ============================================================ G · gonococcal
GBANDS = ["Adolescent (<18)", "Youth (18-24)", "Young Adult (25-34)", "Adult (35-44)", "Middle Age (45-54)", "55+"]
GLABEL = ["Under 18", "18–24", "25–34", "35–44", "45–54", "55+"]


def _pyr(groups, female, male, height=320) -> go.Figure:
    mx = max(list(female) + list(male) + [1]) * 1.25
    fig = go.Figure()
    fig.add_trace(go.Bar(y=groups, x=[-v for v in female], orientation="h", name="Female", marker_color=TEAL, text=[_n(v) for v in female], textposition="outside", customdata=female,
                         hovertemplate="%{y} · Female: %{customdata:,}<extra></extra>", cliponaxis=False))
    fig.add_trace(go.Bar(y=groups, x=male, orientation="h", name="Male", marker_color=BLUE, text=[_n(v) for v in male], textposition="outside",
                         hovertemplate="%{y} · Male: %{x:,}<extra></extra>", cliponaxis=False))
    ui._finish(fig, height, barmode="overlay")
    fig.update_xaxes(range=[-mx, mx], showticklabels=False, showgrid=False)
    fig.update_yaxes(autorange="reversed")
    return fig


def _section_g(D) -> None:
    ui.section_header("Section G — Gonococcal infection")
    gon, gp = D["gonococcal"], D["gonococcal_pregnancy"]
    yv = pd.to_numeric(gon.visits, errors="coerce").fillna(0).groupby(gon.visit_year).sum() if not gon.empty else pd.Series(dtype=float)
    with ui.chart_card("Visits by year", note="Lighter bars are partial years and aren’t comparable."):
        if not yv.empty:
            cols = [AMBER if int(y) in (2024, 2025) else _mix(AMBER, 0.4) for y in yv.index]
            _plot(_columns([f"{int(y)}{'*' if int(y) in (2023, 2026) else ''}" for y in yv.index], list(yv.values), cols, [_n(v) for v in yv.values], height=300), "gyear")
    if 2024 in yv.index and 2025 in yv.index and yv[2024]:
        d = (yv[2025] / yv[2024] - 1) * 100
        ui.insight_bar(f"<strong>The {abs(d):.0f}% {'fall' if d < 0 else 'rise'} {'can’t be called treatment success yet' if d < 0 else 'needs watching'}.</strong> Visits went from {_n(yv[2024])} to {_n(yv[2025])} between two full years, "
                       "but the 2025 export gap outside Emgwen may account for part of it. Nandi county DHIS2 STI surveillance is needed as a benchmark before presenting this as improved outcomes.", "amber")
    l, r = st.columns([3, 2])
    with l:
        with ui.chart_card("Age and sex", note="All years. Women carry the larger share in every adult age group."):
            if not _empty(gon):
                f = [_sum(gon[(gon.age_group == b) & (gon.sex == "F")], "visits") for b in GBANDS]
                m = [_sum(gon[(gon.age_group == b) & (gon.sex == "M")], "visits") for b in GBANDS]
                _plot(_pyr(GLABEL, f, m), "gpyr")
    with r:
        with ui.chart_card("Under 18, 2024"):
            u = gon[(gon.visit_year == 2024) & (gon.age_group == "Adolescent (<18)")] if not gon.empty else gon
            if not _empty(u):
                uf, um = _sum(u[u.sex == "F"], "visits"), _sum(u[u.sex == "M"], "visits")
                st.markdown(f'<div style="font-size:34px;font-weight:700;color:{RED};line-height:1.1">{_n(uf + um)}</div><b>patients under 18</b> with gonococcal infection', unsafe_allow_html=True)
                _plot(_single_bar("Under 18", [("Female", uf, TEAL), ("Male", um, BLUE)]), "gsafe")
                st.markdown('<div class="afya-body">Boys are included, so this isn’t only an adolescent pregnancy issue. Both need <b>county child-protection reporting</b> alongside clinical management.</div>', unsafe_allow_html=True)
    l, r = st.columns(2)
    with l:
        with ui.chart_card("What gonococcal infection leads to", note="Co-diagnoses on gonococcal visits."):
            if not gon.empty:
                items = [("Pregnancy co-coded", _sum(gp, "visits") or _sum(gon, "with_pregnancy")), ("Sepsis", _sum(gon, "with_sepsis")),
                         ("Severe, in pregnancy", _sum(gp, "severe_cases")), ("Pelvic inflammatory disease", _sum(gon, "with_pid")), ("HIV", _sum(gon, "with_hiv"))]
                _plot(_hbar([n for n, _ in items], [v for _, v in items], AMBER, [_n(v) if v else "0 recorded" for _, v in items], height=230), "gcomp")
                if _sum(gon, "with_hiv") == 0:
                    st.markdown(f'<span class="afya-limit">Limitation: 0 HIV co-diagnoses across {_n(_sum(gon, "visits"))} visits, against ~3.8% local prevalence. This is a documentation gap, not absence.</span>', unsafe_allow_html=True)
    with r:
        with ui.chart_card("Gonococcal infection in pregnancy", note="Obstetric visits code pregnancy first, so true overlap is higher."):
            if not _empty(gp):
                ya24 = gon[(gon.visit_year == 2024) & (gon.sex == "F") & (gon.age_group == "Young Adult (25-34)")] if not gon.empty else gon
                st.markdown(
                    f'<div class="afya-tiles"><div style="border-top:3px solid {AMBER};background:#FEF3E0"><b style="color:{AMBER}">{_n(_sum(gp, "visits"))}</b>visits with both<i>2024–2025</i></div>'
                    f'<div style="border-top:3px solid {RED};background:#FFF5F5"><b style="color:{RED}">{_n(_sum(gp, "severe_cases"))}</b>severe cases<i>pregnant women</i></div>'
                    f'<div style="border-top:3px solid {RED};background:#FFF5F5"><b style="color:{RED}">{_n(_sum(ya24, "with_sepsis"))}</b>went on to sepsis<i>women 25–34, 2024</i></div></div>'
                    f'<span class="afya-limit">Obstetric visits code the pregnancy first, so true overlap is higher than {_n(_sum(gp, "visits"))}.</span>', unsafe_allow_html=True)
    l, r = st.columns(2)
    with l:
        with ui.chart_card("Where they live: Nandi or outside", note="Share of gonococcal visits by county of residence."):
            if not _empty(gon):
                nd = [_sum(gon[gon.age_group == b], "from_nandi") for b in GBANDS]
                ot = [_sum(gon[gon.age_group == b], "from_outside_nandi") for b in GBANDS]
                tot = [a + b or 1 for a, b in zip(nd, ot)]
                _plot(_sb(GLABEL, [("Nandi", [100 * a / t for a, t in zip(nd, tot)], TEAL), ("Outside Nandi", [100 * b / t for b, t in zip(ot, tot)], LGREY)], height=300, suffix="%"), "gorig")
                _cap(f"{100 * sum(nd) / max(1, sum(nd) + sum(ot)):.0f}% of visits are from Nandi residents: local transmission, not a transient or highway pattern.")
    with r:
        _ward_card("Where gonococcal infection comes from: top 5 wards", "Gonococcal", AMBER, D["ward_top5"], height=300)
    sep13 = _sum(gon[(gon.visit_year == 2024) & (gon.sex == "F") & (gon.age_group == "Young Adult (25-34)")], "with_sepsis") if not gon.empty else 0
    u18 = _sum(gon[(gon.visit_year == 2024) & (gon.age_group == "Adolescent (<18)")], "visits") if not gon.empty else 0
    ui.insight_bar(f"<strong>A local epidemic signal that reaches pregnancy and sepsis.</strong> {_n(sep13)} gonococcal visits in women 25–34 also carried sepsis in 2024, and "
                   f"{_n(_sum(gp, 'severe_cases'))} pregnant patients were severe, so the ascending-infection pathway is active. {_n(u18)} patients under 18 need county child-protection reporting alongside treatment.", "amber")


def _hdr_n(D) -> str:
    pts = D["infectious_patients"]
    allr = pts[pts.period.astype(str) == "All years"] if not pts.empty else pts
    return _n(allr.infectious_patients.iloc[0]) if len(allr) else "n/a"


# ============================================================ page
def render() -> None:
    ui.inject_css()
    st.markdown(_CSS, unsafe_allow_html=True)
    if q.DEMO:
        ui.demo_banner()
    _top = st.container()
    with ui.loading("Infectious disease"):
        D = _load()
    with _top:
        ui.page_header("Infectious disease", "What infection brings people to Kapsabet CRH, who they are, where they come from, and how late they arrive", icon="infectious", stat=(_hdr_n(D), "infectious disease patients"))
    ui.kpi_row(_kpis(D))
    ui.insight_bar("<strong>Infection reaches Kapsabet late, and it starts early in life.</strong> Sepsis is the leading toddler diagnosis, pneumonia is the biggest burden overall and peaks in working-age adults "
                   "and the elderly, and young women carry the UTI and gonococcal load. The hospital is the endpoint of a care-seeking gap that starts upstream.", "red")
    _section_a(D)
    _section_b(D)
    _section_c(D)
    _section_d(D)
    _section_e(D)
    _section_f(D)
    _section_g(D)
    ui.synthesis("Kapsabet isn’t failing these patients at the point of care. It receives them after a failure upstream: toddlers whose chest infections turn into sepsis, a falciparum shift with resistance to watch, "
                 "women whose UTI and STI infections run into pregnancy, and working-age adults whose pneumonia arrives severe. The evidence is here, but the fix is community-level: earlier recognition, vaccination, "
                 "sickle-cell screening and better documentation.")
