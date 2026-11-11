"""
views_ncd.py: Tab 4, NCDs.

No SQL here: every number comes from queries.q4_*. This file shapes DataFrames into Plotly charts, heatmap tables,
ward maps (template/geo_engine.js, kind "ncd") and ui_template cards. Sections: A burden at a glance · B patients who
keep coming back · C women vs men · D hypertension and pregnancy · E patterns that do not fit · F sickle cell disease.

Visuals are heatmap tables, horizontal bars (grouped, stacked and pyramid), KPI cards and the ward maps.

Data-quality issues (coding errors, documentation gaps) belong on the Data Quality tab unless they limit a specific
finding, in which case they appear as a note on that finding.
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import nandi_geo_map as geo
import nandi_queries as q
import nandi_ui_template as ui
import nandi_views_geo as gv
import nandi_views_infectious as vi

BLUE, TEAL, AMBER, RED = ui.BLUE, ui.PRIMARY, ui.AMBER, ui.CRITICAL
LGREY = "#CBD5E1"
DGREY = "#9CA3AF"
MUTED = ui.TEXT_MUTED
AGES = vi.AGES
AS = ["0-4", "5-12", "13-17", "18-24", "25-34", "35-44", "45-54", "55+"]
SHORT = {"Chronic kidney disease": "Kidney disease", "Sickle cell disease": "Sickle cell", "Stroke / cerebrovascular disease": "Stroke"}
KEEP = ["Diabetes", "Asthma", "Cancer", "Heart failure", "Hypertension"]   # the five condition cards in the KPI row
_n, _f, _cap, _plot, _eq = vi._n, vi._f, vi._cap, vi._plot, vi._eq

_CSS = f"""
<style>
.afya-kpis.k4 {{ grid-template-columns:repeat(4,1fr); margin-bottom:12px; }}
.afya-kpis.k2 {{ grid-template-columns:repeat(2,1fr); margin-bottom:12px; }}
@media(max-width:1100px) {{ .afya-kpis.k4 {{ grid-template-columns:repeat(2,1fr); }} }}
.ncd-lg {{ display:flex; gap:14px; flex-wrap:wrap; font-size:11px; color:{MUTED}; margin-top:6px; }}
.ncd-lg i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; }}
.ncd-sub {{ font-size:11px; font-style:italic; color:{MUTED}; margin:-4px 0 8px; }}
</style>
"""


# ============================================================ helpers
def _s(c: str) -> str:
    return SHORT.get(c, c)


def _lc(c: str) -> str:
    return _s(c).lower()


def _kpi(cards: list[dict], cls: str = "") -> None:
    """Same card as ui.kpi_row, with a column-count class (k4 / k2) for rows that are not six wide."""
    html = "".join(
        f'<div class="afya-kpi" style="border-top-color:{ui.ACCENTS.get(c.get("accent", "blue"), BLUE)}">'
        f'<div class="afya-kpi-label">{c["label"]}</div><div class="afya-kpi-value">{c["value"]}</div>'
        f'<div class="afya-kpi-sub">{c.get("sub", "")}</div></div>' for c in cards)
    st.markdown(f'<div class="afya-kpis {cls}">{html}</div>', unsafe_allow_html=True)


def _legend(items: list[tuple[str, str]]) -> None:
    st.markdown('<div class="ncd-lg">' + "".join(f'<span><i style="background:{c}"></i>{t}</span>' for t, c in items) + "</div>", unsafe_allow_html=True)


def _words(p: float) -> str:
    """'a third' for 30 to 36%, 'a quarter' for 22 to 28%, 'half' for 46 to 54%; otherwise the number."""
    for lo, hi, w in ((22, 28, "a quarter"), (30, 36, "a third"), (46, 54, "half")):
        if lo <= p <= hi:
            return w
    return f"{p:.0f}%"


def _signed(p: float) -> str:
    return "flat" if abs(p) < 0.5 else f"{p:+.0f}%"


def _cell(text, style: str = "", title: str = "") -> tuple:
    t = f'<span title="{title}">{text}</span>' if title else text
    return (t, "", style)


def _dot() -> tuple:
    return ("·", "", "color:#9CA3AF")


def _num(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = pd.to_numeric(df[c])
            except (ValueError, TypeError):
                pass
    return df


def _load() -> dict[str, pd.DataFrame]:
    names = ["headline", "prevalence", "timeseries", "comorbidity", "anaemia_sign", "anaemia_subtypes", "ward_patients", "ward_recurrent",
             "condition_return", "returning", "baseline", "female_gap", "htn_pregnancy", "htn_chronic", "heart_failure", "diabetes_type",
             "stroke", "sickle_crisis", "sickle_trait", "sickle_returning"]
    return {n: _num(getattr(q, "q4_" + n)().copy()) for n in names}


class _Data:
    """The handful of frames and lookups the sections share, computed once per render."""

    def __init__(self, D: dict[str, pd.DataFrame]):
        self.D = D
        p = D["prevalence"]
        self.prev = p
        self.burden = p.groupby("condition_name").patients.sum().sort_values(ascending=False) if not p.empty else pd.Series(dtype=float)
        self.heat = (p.pivot_table(index="condition_name", columns="age_group_at_first_diagnosis", values="patients", aggfunc="sum")
                     .reindex(columns=AGES).fillna(0)) if not p.empty else pd.DataFrame()
        ts = D["timeseries"]
        self.ts = ts.pivot_table(index="condition_name", columns="visit_year", values="patients_seen", aggfunc="sum").fillna(0) if not ts.empty else pd.DataFrame()
        for y in (2023, 2024, 2025, 2026):
            if y not in self.ts.columns:
                self.ts[y] = 0
        h = D["headline"]
        h = h.assign(period=h.period.astype(str)) if not h.empty else h
        self.hl = {r.period: r for r in h.itertuples()} if not h.empty else {}
        b = D["baseline"]
        self.base_age = {}
        if not b.empty:
            for a, g in b.groupby("age_group"):
                self.base_age[a] = 100 * _f(g[g.sex == "F"].distinct_patients.sum()) / max(1.0, _f(g.distinct_patients.sum()))
            self.base_all = 100 * _f(b[b.sex == "F"].distinct_patients.sum()) / max(1.0, _f(b.distinct_patients.sum()))
        else:
            self.base_all = 0.0

    def year(self, c: str, y: int) -> float:
        return _f(self.ts.loc[c, y]) if c in self.ts.index else 0.0

    def change(self, c: str) -> float | None:
        a, b = self.year(c, 2024), self.year(c, 2025)
        return (b / a - 1) * 100 if a else None

    def fem_share(self, c: str, age: str | None = None) -> float | None:
        d = self.prev[self.prev.condition_name == c]
        if age:
            d = d[d.age_group_at_first_diagnosis == age]
        t = _f(d.patients.sum())
        return 100 * _f(d[d.sex == "F"].patients.sum()) / t if t else None


# ============================================================ charts
def _year_bars(S: _Data) -> go.Figure:
    order = sorted(S.burden.index, key=lambda c: -(S.year(c, 2024) + S.year(c, 2025)))
    labels = [_s(c) for c in order]
    xmax = max([max(S.year(c, 2024), S.year(c, 2025), S.year(c, 2026)) for c in order] + [1])
    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=[S.year(c, 2024) for c in order], orientation="h", name="2024", marker_color=LGREY, hovertemplate="%{y} 2024: %{x:,}<extra></extra>"))
    fig.add_trace(go.Bar(y=labels, x=[S.year(c, 2025) for c in order], orientation="h", name="2025", marker_color=BLUE, hovertemplate="%{y} 2025: %{x:,}<extra></extra>"))
    fig.add_trace(go.Bar(y=labels, x=[S.year(c, 2026) for c in order], orientation="h", name="2026 year-to-date",
                         marker=dict(color="rgba(156,163,175,.55)", pattern=dict(shape="/", fgcolor=DGREY, bgcolor="#fff", size=6), line=dict(color=DGREY, width=0.5)),
                         hovertemplate="%{y} 2026 so far: %{x:,}<extra></extra>"))
    chg = [S.change(c) for c in order]
    col = [MUTED if d is None else RED if d <= -30 else AMBER if d < -0.5 else TEAL if d > 0.5 else MUTED for d in chg]
    fig.add_trace(go.Scatter(x=[xmax * 1.16] * len(order), y=labels, mode="text", text=[f"<b>{_signed(d)}</b>" if d is not None else "" for d in chg],
                             textfont=dict(size=12, color=col), hoverinfo="skip", showlegend=False))
    ui._finish(fig, 560, barmode="group", bargap=0.22, bargroupgap=0.04, margin=dict(l=10, r=10, t=34, b=10), legend=dict(orientation="h", x=0, y=1.07, traceorder="normal"))
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.update_xaxes(range=[0, xmax * 1.26], showticklabels=False, showgrid=False)
    return fig


def _stack100(labels, series: list[tuple[str, list, str, str]], height: int) -> go.Figure:
    """Horizontal 100% stacked bars. series: [(name, counts, colour, label colour)]. Each bar is one group; n= on the right."""
    labels = list(labels)
    tot = [sum(_f(s[1][i]) for s in series) for i in range(len(labels))]
    fig = go.Figure()
    for name, vals, colour, tcol in series:
        pct = [100 * _f(v) / t if t else 0 for v, t in zip(vals, tot)]
        fig.add_trace(go.Bar(y=labels, x=pct, orientation="h", name=name, marker_color=colour, marker_line_color="#fff", marker_line_width=1.5,
                             text=[f"{p:.0f}%" if p >= 11 else "" for p in pct], textposition="inside", insidetextanchor="middle", textfont=dict(color=tcol, size=11),
                             customdata=[int(_f(v)) for v in vals], hovertemplate=f"%{{y}}<br>{name}: %{{customdata:,}} patients (%{{x:.0f}}%)<extra></extra>"))
    fig.add_trace(go.Scatter(x=[101] * len(labels), y=labels, mode="text", text=[f"n={int(t):,}" for t in tot], textposition="middle right",
                             textfont=dict(size=11, color=MUTED), hoverinfo="skip", showlegend=False, cliponaxis=False))
    ui._finish(fig, height, barmode="stack", margin=dict(l=10, r=60, t=40, b=10), legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", traceorder="normal"))
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.update_xaxes(range=[0, 112], showticklabels=False, showgrid=False)
    return fig


# ============================================================ maps
def _ward_geo(df: pd.DataFrame) -> dict:
    """{(ward, sub_county): dict(lat, lon, rings)} from the BOUNDARY_GEOJSON / latitude / longitude columns the map queries return."""
    out = {}
    if df.empty or "latitude" not in df.columns:
        return out
    for r in df.drop_duplicates(["ward", "sub_county"]).itertuples():
        if pd.isna(r.latitude) or pd.isna(r.longitude):
            continue
        out[(r.ward, r.sub_county)] = dict(lat=round(float(r.latitude), 5), lon=round(float(r.longitude), 5), rings=gv._rings(r.boundary_geojson))
    return out


def _burden_payload(S: _Data) -> dict | None:
    df = S.D["ward_patients"]
    if df.empty:
        return None
    G = _ward_geo(df)
    conds = ["All NCDs"] + [c for c in S.burden.index]
    wards = {k: dict(w=k[0], s=k[1], **g, v={}) for k, g in G.items()}
    unplaced: dict[str, int] = {}
    for r in df.itertuples():
        k = (r.ward, r.sub_county)
        if k in wards:
            wards[k]["v"][r.condition_name] = int(r.patients)
        else:
            unplaced[r.condition_name] = unplaced.get(r.condition_name, 0) + int(r.patients)
    return dict(mode="burden", hospital=dict(lat=q.HOSPITAL_LAT, lon=q.HOSPITAL_LON), conds=conds, labels=["All NCDs"] + [_s(c) for c in conds[1:]],
                wards=list(wards.values()), unplaced=unplaced, h=560)


def _recurrent_payload(S: _Data) -> dict | None:
    df = S.D["ward_recurrent"]
    if df.empty:
        return None
    G = _ward_geo(df)
    wards = []
    for r in df.itertuples():
        g = G.get((r.ward, r.sub_county))
        if g is None:
            continue
        wards.append(dict(w=r.ward, s=r.sub_county, pat=int(r.ncd_patients), both=int(r.seen_both_years), p=float(r.pct_seen_both_years), **g))
    return dict(mode="recur", hospital=dict(lat=q.HOSPITAL_LAT, lon=q.HOSPITAL_LON), wards=wards, thr=20, h=560)


# ============================================================ header, KPIs
def _kpis(S: _Data) -> list[dict]:
    allr = S.hl.get("All years")
    cards = [dict(label="Patients with an NCD", value=_n(allr.ncd_patients) if allr else "n/a",
                  sub=f"{allr.pct_ncd:.1f}% of {_n(allr.all_patients)} patients" if allr else "", accent="blue")]
    acc = {"Diabetes": "blue", "Asthma": "amber", "Cancer": "crit", "Heart failure": "crit", "Hypertension": "teal"}
    for c in KEEP:
        d = S.change(c)
        chg = "" if d is None else ui.arrow_html(round(d) if abs(d) >= 0.5 else 0, flat="flat")
        cards.append(dict(label=_s(c), value=_n(S.burden.get(c, 0)), sub=f"patients · 2024→25 {chg or 'flat'}", accent=acc[c]))
    return cards


def _top_insight(S: _Data) -> None:
    a, b = S.hl.get("2024"), S.hl.get("2025")
    allr = S.hl.get("All years")
    share = 100 * _f(S.burden.get("Diabetes", 0)) / allr.ncd_patients if allr and allr.ncd_patients else 0
    head = (f"<strong>NCDs are {a.pct_ncd:.0f}% of Kapsabet patients, {'flat' if abs(a.pct_ncd - b.pct_ncd) < 0.5 else 'changing'} between 2024 and 2025, "
            f"and diabetes alone carries {_words(share)} of them.</strong>") if a is not None and b is not None and allr else "<strong>NCDs at Kapsabet CRH.</strong>"
    ui.insight_bar(f"{head} Heart failure, stroke, COPD and cancer are 55+ diseases; epilepsy and sickle cell start in childhood. Kapsabet ward supplies the most patients in every condition, "
                   "and almost no one comes from outside Nandi. 2026 is shown year-to-date and 2023 is a partial year, so neither is used for year-on-year change.", "blue")


# ============================================================ A · burden at a glance
def _who_gets_what(S: _Data) -> None:
    rows = []
    for c in S.burden.index:
        v = S.heat.loc[c] if c in S.heat.index else pd.Series([0] * 8, index=AGES)
        tot = _f(v.sum())
        cells = [(_s(c), "hl")]
        for a in AGES:
            n = _f(v.get(a, 0))
            p = 100 * n / tot if tot else 0
            cells.append(_dot() if n == 0 else _cell(f"{p:.0f}%", ui.heat(p, 0, 40), f"{c}, {a}: {int(n)} patients ({p:.0f}%)"))
        rows.append(cells)
    ui.html_table([("", "")] + [(a, a) for a in AS], rows, compact=True, widths=_eq(20, 8), min_width=420)
    _cap("Share of each condition’s patients by age group at first diagnosis. Darker = bigger share.")


def _comorbidity(S: _Data) -> None:
    c = S.D["comorbidity"]
    if vi._empty(c):
        return
    pairs = [("Hypertension + diabetes", "hypertension_and_diabetes"), ("Hypertension + heart failure", "hypertension_and_heart_failure"),
             ("Diabetes + kidney disease", "diabetes_and_ckd"), ("Hypertension + kidney disease", "hypertension_and_ckd"),
             ("Diabetes + heart failure", "diabetes_and_heart_failure"), ("Hypertension + epilepsy", "hypertension_and_epilepsy")]
    mx = max([_f(c.groupby("age_group")[k].sum().max()) for _, k in pairs] + [1])
    rows = []
    for label, k in pairs:
        by = c.groupby("age_group")[k].sum()
        sex = {(r.age_group, r.sex): _f(getattr(r, k)) for r in c.itertuples()}
        cells = [(label, "hl")]
        for a in AGES:
            n = _f(by.get(a, 0))
            cells.append(_dot() if n == 0 else _cell(f"{int(n)}", ui.heat(n, 0, mx), f"{label}, {a}: {int(n)} patients ({int(sex.get((a, 'F'), 0))} F, {int(sex.get((a, 'M'), 0))} M)"))
        cells.append((f"{int(_f(by.sum()))}", "hl"))
        rows.append(cells)
    ui.html_table([("Pair", "")] + [(a, a) for a in AS] + [("All", "")], rows, compact=True, widths=_eq(30, 9), min_width=480)
    _cap("Patients carrying both conditions, by age group. Treat as a minimum: most visits record a single diagnosis.")


def _anaemia(S: _Data) -> None:
    a = S.D["anaemia_sign"]
    if vi._empty(a):
        return
    a = a.sort_values("pct_with_anaemia", ascending=False)
    cols = [RED if c == "Sickle cell disease" else BLUE for c in a.condition_name]
    fig = vi._hbar([_s(c) for c in a.condition_name], list(a.pct_with_anaemia), cols, [f"{p:.1f}%" for p in a.pct_with_anaemia], height=330)
    fig.update_xaxes(range=[0, 12])
    _plot(fig, "ncd-anaemia")
    iron = S.D["anaemia_subtypes"]
    iv = _f(iron[iron.anaemia_subtype.astype(str).str.startswith("Iron deficiency")].total_visits.sum()) if not iron.empty else 0
    others = a[a.condition_name != "Sickle cell disease"].pct_with_anaemia.max()
    _cap(f"Under {int(-(-others // 1)) if others else 5}% for every condition except sickle cell. Iron-deficiency anaemia ({_n(iv)} visits) is nutritional and not counted as an NCD sign.")


def _burden_insight(S: _Data) -> None:
    ok = [c for c in S.burden.index if S.year(c, 2024) >= 30 and S.year(c, 2025) >= 30]
    ch = {c: S.change(c) for c in ok}
    lead = sorted(S.burden.index, key=lambda c: -(S.year(c, 2024) + S.year(c, 2025)))[:3]
    up = sorted([c for c in ok if ch[c] and ch[c] > 0.5], key=lambda c: -ch[c])[:3]
    down = sorted([c for c in ok if ch[c] and ch[c] < -0.5], key=lambda c: ch[c])[:3]
    a, b = S.hl.get("2024"), S.hl.get("2025")
    names = lambda cs: ", ".join(_lc(c) for c in cs[:-1]) + " and " + _lc(cs[-1]) if len(cs) > 1 else (_lc(cs[0]) if cs else "")  # noqa: E731
    fmtc = lambda cs: ", ".join(f"{_lc(c)} ({ch[c]:+.0f}%)" for c in cs[:-1]) + " and " + f"{_lc(cs[-1])} ({ch[cs[-1]]:+.0f}%)" if len(cs) > 1 else (f"{_lc(cs[0])} ({ch[cs[0]]:+.0f}%)" if cs else "")  # noqa: E731
    share = (f" NCD patients are a steady {a.pct_ncd:.1f}% of all patients in both 2024 and 2025 ({_n(a.ncd_patients)} of {_n(a.all_patients)} and {_n(b.ncd_patients)} of {_n(b.all_patients)})."
             if a is not None and b is not None and abs(a.pct_ncd - b.pct_ncd) < 0.5 else "")
    ui.insight_bar(f"<strong>{names([_s(c) for c in lead]).capitalize()} lead both years.</strong> {fmtc(up).capitalize()} grew in 2025 while {fmtc(down)} fell.{share} "
                   "Part of any fall is the 2025 export gap, and COPD and sickle cell are small counts.", "blue")


def _section_a(S: _Data, maps: dict) -> None:
    ui.section_header("Section A — Burden at a glance")
    l, r = st.columns([3, 2])
    with l:
        with ui.chart_card("Patients per condition: 2024 vs 2025",
                           note="Each condition has three bars: 2024, 2025, and 2026 so far (hatched, year not finished, never compared). Label = % change 2024 to 2025. "
                                + (f"All NCDs together: {_n(S.hl['2024'].ncd_patients)} patients in 2024, {_n(S.hl['2025'].ncd_patients)} in 2025 "
                                   f"({_signed((S.hl['2025'].ncd_patients / S.hl['2024'].ncd_patients - 1) * 100)}), {_n(S.hl['2026'].ncd_patients)} so far in 2026." if all(k in S.hl for k in ("2024", "2025", "2026")) else "")):
            if not S.burden.empty:
                _plot(_year_bars(S), "ncd-years")
    with r:
        with ui.chart_card("Who gets what"):
            if not S.heat.empty:
                _who_gets_what(S)
    with ui.chart_card("Where they come from", note="Patients by ward (Nandi residents, 2024-2025). Pick a condition."):
        if maps.get("burden"):
            p = S.prev
            n_in, n_out = _f(p.from_nandi.sum()), _f(p.from_outside_nandi.sum())
            if n_in + n_out:
                maps["burden"]["outside"] = f"Outside Nandi: {_n(n_out)} of {_n(n_in + n_out)} NCD patients with a recorded county ({100 * n_out / (n_in + n_out):.1f}%)"
            geo.render_geo("ncd", maps["burden"], height=705, scrolling=False)
        else:
            vi._cap("No ward data returned for this card.")
    l, r = st.columns(2)
    with l:
        with ui.chart_card("What NCDs travel together"):
            _comorbidity(S)
    with r:
        with ui.chart_card("Is anaemia a sign here?", note="% of each condition’s patients who also carry an anaemia code."):
            _anaemia(S)
    _burden_insight(S)


# ============================================================ B · patients who keep coming back
def _returning(S: _Data) -> pd.DataFrame:
    r = S.D["returning"]
    if r.empty:
        return r
    r = r.assign(v=r.patients * r.avg_total_visits, pct=r.patients * r.avg_pct_visits_ncd)
    g = r.groupby(["age_group", "sex"]).agg(patients=("patients", "sum"), v=("v", "sum"), pct=("pct", "sum"), comorbid=("comorbid_patients", "sum")).reset_index()
    g["avg_visits"] = g.v / g.patients
    g["pct_ncd"] = g.pct / g.patients
    return g


def _ret_table(g: pd.DataFrame, col: str, fmt, lo: float, hi: float, colour: str) -> None:
    rows = []
    for a in AGES:
        cells = [(a, "hl")]
        for sx, nm in (("F", "women"), ("M", "men")):
            d = g[(g.age_group == a) & (g.sex == sx)]
            if d.empty:
                cells.append(_dot())
            else:
                v = _f(d[col].iloc[0])
                cells.append(_cell(fmt(v), ui.heat(v, lo, hi, colour), f"{a} {nm}: {fmt(v)}"))
        rows.append(cells)
    ui.html_table([("", ""), ("Women", ""), ("Men", "")], rows, compact=True, widths=_eq(40, 2), min_width=300)


def _recur_stats(S: _Data) -> dict:
    m2, rc = S.D["ward_recurrent"], S.D["condition_return"]
    out = dict(n=0, both=0, kap=0, wmax=0.0)
    if not m2.empty:
        out["n"], out["both"] = int(m2.ncd_patients.sum()), int(m2.seen_both_years.sum())
        out["kap"] = int(m2[m2.ward == "Kapsabet"].seen_both_years.sum())
        big = m2[m2.ncd_patients >= 20]
        out["wmax"] = float(big.pct_seen_both_years.max()) if not big.empty else 0.0
    out["cond"] = {}
    if not rc.empty:
        for r in rc[rc.age_band == "All ages"].itertuples():
            out["cond"][r.condition_name] = float(r.pct_seen_both_years)
    return out


def _cond_return(S: _Data) -> None:
    rc = S.D["condition_return"]
    if vi._empty(rc):
        return
    bands = ["Under 18", "18-54", "55+", "All ages"]
    allr = rc[rc.age_band == "All ages"].sort_values("pct_seen_both_years", ascending=False)
    rows = []
    for c in allr.condition_name:
        cells = [(_s(c), "hl")]
        for b in bands:
            d = rc[(rc.condition_name == c) & (rc.age_band == b)]
            if d.empty:
                cells.append(_dot())
                continue
            r = d.iloc[0]
            cells.append(_cell(f"{r.pct_seen_both_years:.1f}%", ("opacity:.4;" if r.patients < 15 else "") + ui.heat(r.pct_seen_both_years, 0, 7, AMBER),
                               f"{c}, {b}: {int(r.seen_both_years)} of {int(r.patients)} patients seen in both years ({r.pct_seen_both_years}%)"))
        rows.append(cells)
    ui.html_table([("", "")] + [(b, b) for b in bands], rows, compact=True, widths=_eq(32, 4), min_width=380)


def _section_b(S: _Data, maps: dict) -> None:
    ui.section_header("Section B — Patients who keep coming back")
    st.markdown('<div class="ncd-sub">Returning = seen in more than one calendar year, with at least one NCD visit.</div>', unsafe_allow_html=True)
    g, st_ = _returning(S), _recur_stats(S)
    n = _f(g.patients.sum()) if not g.empty else 0
    allr = S.hl.get("All years")
    outside = int(_f(S.D["returning"][S.D["returning"].origin == "Outside Nandi"].patients.sum())) if not S.D["returning"].empty else 0
    ratio = (allr.ncd_patients / n) if n and allr else 0
    _kpi([dict(label="Returning patients", value=_n(n), sub=f"about 1 in {round(ratio)} NCD patients" if ratio else "", accent="blue"),
          dict(label="Average visits", value=f"{_f((g.avg_visits * g.patients).sum()) / n:.1f}" if n else "n/a", sub="per returning patient", accent="teal"),
          dict(label="NCD share of visits", value=f"{_f((g.pct_ncd * g.patients).sum()) / n:.0f}%" if n else "n/a", sub="of a returning patient's visits", accent="amber"),
          dict(label="From outside Nandi", value=_n(outside), sub=f"of {_n(n)} returning patients", accent="crit")], "k4")
    l, r = st.columns(2)
    with l:
        with ui.chart_card("Which conditions come back", note="% of each condition’s patients seen in both 2024 and 2025, by age band. Faded = fewer than 15 patients. Hover for counts."):
            _cond_return(S)
    with r:
        with ui.chart_card("Where recurrent patients come from", note="Share of each ward’s NCD patients seen in both 2024 and 2025. Hover for counts."):
            if maps.get("recur"):
                geo.render_geo("ncd", maps["recur"], height=655, scrolling=False)
            else:
                vi._cap("No ward data returned for this card.")
    if st_["n"]:
        c = st_["cond"]
        ui.insight_bar(f"<strong>Chronic care is not continuing year to year: only {_n(st_['both'])} of {_n(st_['n'])} NCD patients ({100 * st_['both'] / st_['n']:.0f}%) are seen in both 2024 and 2025.</strong> "
                       f"Diabetes, the biggest chronic load, returns at {c.get('Diabetes', 0):.1f}%; hypertension at {c.get('Hypertension', 0):.1f}%; heart failure at {c.get('Heart failure', 0):.1f}%. "
                       f"Of the {_n(st_['both'])}, {_n(st_['kap'])} are Kapsabet residents, and no ward with 20+ patients rises above {st_['wmax']:.0f}%. "
                       "These conditions need lifelong follow-up, so either patients collect their care elsewhere (pharmacies, other facilities) or this register does not link them. Both are worth knowing. "
                       "The 2025 export gap limits how far this can be pushed, so read it as what the register can show, not proof that patients are lost. "
                       f"The {_n(n)} below count anyone seen across 2023 to 2026 with any NCD visit, which is why that figure is higher.", "amber")
    l, r = st.columns(2)
    with l:
        with ui.chart_card("Returning patients", note="Patients by age and sex."):
            if not g.empty:
                _ret_table(g, "patients", lambda v: f"{int(v)}", 0, 100, BLUE)
    with r:
        with ui.chart_card("NCD share of their visits", note="Average % of a returning patient’s visits that are NCD-related."):
            if not g.empty:
                _ret_table(g, "pct_ncd", lambda v: f"{v:.0f}%", 40, 75, BLUE)
    l, r = st.columns(2)
    with l:
        with ui.chart_card("Average visits per returning patient"):
            if not g.empty:
                _ret_table(g, "avg_visits", lambda v: f"{v:.1f}", 2, 6, TEAL)
    with r:
        with ui.chart_card("Returning patients with 2+ conditions", note="Coded comorbid patients. A floor, not a total."):
            if not g.empty:
                _ret_table(g, "comorbid", lambda v: f"{int(v)}", 0, 12, RED)
    if not g.empty:
        def share(ages):
            d = g[g.age_group.isin(ages)]
            return _f((d.pct_ncd * d.patients).sum()) / max(1.0, _f(d.patients.sum()))
        top = g.sort_values("pct_ncd", ascending=False).iloc[0]
        cm = g[(g.age_group == top.age_group) & (g.sex == top.sex)].comorbid.iloc[0]
        who = "Men" if top.sex == "M" else "Women"
        ui.insight_bar(f"<strong>Older chronic patients come back for NCD care; younger ones mix it with everything else.</strong> NCD visits are about {share(['Toddler (0-4)']):.0f}% of a toddler’s visits "
                       f"and {share(['55+']):.0f}% at 55+. {who} aged {AS[AGES.index(top.age_group)]} are the most NCD-heavy group at {top.pct_ncd:.0f}% and carry the most comorbidity ({int(cm)} of {int(top.patients)}).", "blue")


# ============================================================ C · women vs men
def _section_c(S: _Data) -> None:
    ui.section_header("Section C — Women vs men")
    gp = S.D["female_gap"]
    fs = {c: S.fem_share(c) for c in S.burden.index}
    fs = {c: v for c, v in fs.items() if v is not None}
    srt = sorted(fs, key=lambda c: fs[c])
    pick = srt[:2] + srt[-2:][::-1] if len(srt) >= 4 else srt
    overall = ", ".join(f"{_lc(c)} {fs[c]:.0f}%" for c in pick)
    note = (f"Female share minus the hospital baseline for that age group, in percentage points (overall female share: {overall}; hospital {S.base_all:.0f}%). "
            "Hover for the actual %. Faded = fewer than 15 patients.")
    with ui.chart_card("Where women and men depart from the hospital norm, by age", note=note):
        if not vi._empty(gp):
            rows = []
            for c in gp.drop_duplicates("condition_name").condition_name:
                cells = [(_s(c), "hl")]
                for a in AGES:
                    d = gp[(gp.condition_name == c) & (gp.age_group == a)]
                    if d.empty:
                        cells.append(_dot())
                        continue
                    r = d.iloc[0]
                    n = _f(S.heat.loc[c, a]) if c in S.heat.index else 0
                    cells.append(_cell(f"{r.gap_vs_baseline_pts:+.0f}".replace("+0", "0") if round(r.gap_vs_baseline_pts) == 0 else f"{r.gap_vs_baseline_pts:+.0f}",
                                       ("opacity:.4;" if n < 15 else "") + ui.diverge(r.gap_vs_baseline_pts, 0, 40),
                                       f"{c}, {a}: {r.pct_female_this_condition:.0f}% female vs {r.pct_female_baseline_opd:.0f}% baseline, {r.gap_vs_baseline_pts:+.1f} points ({int(n)} patients)"))
                rows.append(cells)
            ui.html_table([("", "")] + [(a, a) for a in AS], rows, compact=True, widths=_eq(18, 8), min_width=520)
            _legend([("more women than baseline", TEAL), ("more men than baseline", AMBER)])
    ckd = "Chronic kidney disease"
    f_ckd, f_ckd55 = S.fem_share(ckd), S.fem_share(ckd, "55+")
    htn25, hf25 = S.fem_share("Hypertension", "Young Adult (25-34)"), S.fem_share("Heart failure", "Young Adult (25-34)")
    if None not in (f_ckd, f_ckd55, htn25, hf25):
        ui.insight_bar(f"<strong>Kidney disease and cancer skew male; asthma, hypertension and heart failure skew female.</strong> Kidney disease is {f_ckd:.0f}% female overall and {f_ckd55:.0f}% at 55+ "
                       f"against {S.base_all:.0f}% for the hospital. At 25-34, hypertension is {htn25:.0f}% female and heart failure {hf25:.0f}% female.", "amber")


# ============================================================ D · hypertension and pregnancy
def _section_d(S: _Data) -> None:
    ui.section_header("Section D — Hypertension and pregnancy")
    hp = S.D["htn_pregnancy"]
    d = hp[hp.age_group.isin(AGES[3:])].set_index("age_group").reindex(AGES[3:]).fillna(0) if not hp.empty else hp
    l, r = st.columns(2)
    with l:
        with ui.chart_card("Women’s hypertension visits", note="Visits by age group. Pregnancy-related = pre-eclampsia, gestational, “complicating pregnancy”. Those are excluded from hypertension counts elsewhere on this tab."):
            if not vi._empty(d):
                rowsH = [("All hypertension visits", "total_hypertension_visits_f", BLUE), ("Pregnancy-related", "pregnancy_related_htn", AMBER), ("Any pregnancy context", "any_pregnancy_context", AMBER)]
                rows = []
                for lab, col, colour in rowsH:
                    mx = max(_f(d[col].max()), 1)
                    rows.append([(lab, "hl")] + [(_dot() if not _f(d[col].get(a, 0)) else _cell(f"{int(_f(d[col][a]))}", ui.heat(_f(d[col][a]), 0, mx, colour))) for a in AGES[3:]])
                ui.html_table([("", "")] + [(a, a) for a in AS[3:]], rows, compact=True, widths=_eq(34, 5), min_width=380)
    with r:
        with ui.chart_card("Pregnancy-related share", note="% of women’s hypertension visits that are pregnancy-related."):
            if not vi._empty(d):
                pct = [100 * _f(d.pregnancy_related_htn[a]) / max(1.0, _f(d.total_hypertension_visits_f[a])) for a in AGES[3:]]
                fig = vi._hbar(AGES[3:], pct, AMBER, [f"{p:.0f}%" for p in pct], height=240)
                fig.update_xaxes(range=[0, 40])
                _plot(fig, "ncd-pregshare")
    hc = S.D["htn_chronic"]
    with ui.chart_card("Chronic hypertension only, by age and sex", note="Patients, pregnancy-related cases removed. Women left, men right."):
        if not vi._empty(hc):
            f = [_f(hc[(hc.age_group == a) & (hc.sex == "F")].patients_chronic_hypertension_only.sum()) for a in AGES[3:]]
            m = [_f(hc[(hc.age_group == a) & (hc.sex == "M")].patients_chronic_hypertension_only.sum()) for a in AGES[3:]]
            _plot(vi._pyr(AS[3:], f, m, height=280), "ncd-htnpyr")
    if not hp.empty:
        sh = {a: 100 * _f(d.pregnancy_related_htn[a]) / max(1.0, _f(d.total_hypertension_visits_f[a])) for a in AGES[3:]}
        first = max(sh, key=sh.get)
        zero = [a for a in AGES[3:] if sh[a] == 0]
        ui.insight_bar(f"<strong>{_words(sh[first]).capitalize()} of young women’s hypertension visits are pregnancy-related"
                       f"{', and none after 45' if 'Middle Age (45-54)' in zero and '55+' in zero else ''}.</strong> With them removed, chronic hypertension is still mostly women at every age, "
                       "so pregnancy alone does not explain the female excess.", "teal")


# ============================================================ E · patterns that do not fit
def _section_e(S: _Data) -> None:
    ui.section_header("Section E — Patterns that do not fit")
    st.markdown('<div class="ncd-sub">Three conditions behave differently from what we would expect. Each shows what the data does say, what it rules out, and what to check.</div>', unsafe_allow_html=True)
    hf = S.D["heart_failure"]
    ya = "Young Adult (25-34)"
    if not hf.empty:
        fw = _f(hf[(hf.age_group == ya) & (hf.sex == "F")].patients.sum())
        mn = _f(hf[(hf.age_group == ya) & (hf.sex == "M")].patients.sum())
        fshare = 100 * fw / max(1.0, fw + mn)
        base = S.base_age.get(ya, S.base_all)
        preg = int(_f(hf.with_pregnancy_term.sum()))
        _kpi([dict(label="Women aged 25-34", value=_n(fw), sub="heart failure patients", accent="teal"), dict(label="Men aged 25-34", value=_n(mn), sub="heart failure patients", accent="blue"),
              dict(label="Female share", value=f"{fshare:.0f}%", sub=f"hospital baseline {base:.0f}%", accent="amber"), dict(label="Pregnancy terms", value=_n(preg), sub="at any age" if preg == 0 else "across all ages", accent="crit")], "k4")
    l, r = st.columns(2)
    with l:
        with ui.chart_card("Heart failure by age and sex", note="Patients. Women left, men right. One spike breaks an otherwise 55+ disease."):
            if not vi._empty(hf):
                f = [_f(hf[(hf.age_group == a) & (hf.sex == "F")].patients.sum()) for a in AGES]
                m = [_f(hf[(hf.age_group == a) & (hf.sex == "M")].patients.sum()) for a in AGES]
                _plot(vi._pyr(AGES, f, m, height=360), "ncd-hfpyr")
    with r:
        with ui.chart_card("What else is coded with heart failure", note="Patients with each code, by age." + (" Pregnancy terms: none at any age." if not hf.empty and not _f(hf.with_pregnancy_term.sum()) else "")):
            if not hf.empty:
                rows = []
                for lab, col in (("Pregnancy term", "with_pregnancy_term"), ("Cardiomyopathy", "with_cardiomyopathy"), ("Hypertension", "with_hypertension")):
                    cells = [(lab, "hl")]
                    for a in AGES:
                        n = _f(hf[hf.age_group == a][col].sum())
                        cells.append(_cell(f"{int(n)}", ui.heat(n, 0, 20)) if n else ("0", "", "color:#9CA3AF"))
                    rows.append(cells)
                ui.html_table([("", "")] + [(a, a) for a in AS], rows, compact=True, widths=_eq(20, 8), min_width=420)
    if not hf.empty:
        card = int(_f(hf.with_cardiomyopathy[hf.age_group == ya].sum()))
        ui.insight_bar(f"<strong>Heart failure is a 55+ disease, with one exception: women aged 25-34.</strong> There are {_n(fw)} women and {_n(mn)} {'man' if mn == 1 else 'men'} in that group, {fshare:.0f}% female against a hospital baseline "
                       f"of {base:.0f}%. We can rule out the obvious explanations in the data: no pregnancy or postpartum terms{', no cardiomyopathy' if not card else ''}, no rheumatic or valve codes. "
                       f"What is left is a small group worth a clinical record review, {_n(fw + mn)} patients, before it is read as real disease.", "amber")
    dt = S.D["diabetes_type"]
    with ui.chart_card("Diabetes type mix by age group", note="Each bar = 100% of that age group’s diabetes patients with a type coded. Type 1 falls with age, as expected, but never reaches zero."):
        if not vi._empty(dt):
            t = {k: [_f(dt[(dt.diabetes_type == k) & (dt.age_group == a)].patients.sum()) for a in AGES] for k in ("Type 1", "Type 2", "Type not specified")}
            _plot(_stack100(AGES, [("Type 1", t["Type 1"], AMBER, "#fff"), ("Type 2", t["Type 2"], BLUE, "#fff"), ("Type not specified", t["Type not specified"], LGREY, ui.TEXT)], 380), "ncd-dmtype")
    if not dt.empty:
        pct = [100 * t["Type 1"][i] / max(1.0, t["Type 1"][i] + t["Type 2"][i]) for i in range(4)]
        ui.insight_bar(f"<strong>Type 1 diabetes does not fade out in adults.</strong> It is {min(pct):.0f}% to {max(pct):.0f}% of typed diabetes under 25, which fits a childhood-onset disease. "
                       f"But {_n(t['Type 1'][4] + t['Type 1'][5])} adults aged 25-44 and {_n(t['Type 1'][7])} patients aged 55+ also carry a Type 1 code. Some are long-standing Type 1; some are likely Type 2 coded as Type 1. "
                       "The register has no insulin or lab field to tell them apart, so a coding review is the check.", "amber")
    sk = S.D["stroke"]
    l, r = st.columns(2)
    with l:
        with ui.chart_card("Stroke by age group", note="Patients, with how many also carry a sickle cell code."):
            c = "Stroke / cerebrovascular disease"
            if not sk.empty and c in S.heat.index:
                st_tot = _f(sk.stroke_patients.sum())
                row1 = [("Stroke patients", "hl")] + [(_dot() if not _f(S.heat.loc[c, a]) else _cell(f"{int(S.heat.loc[c, a])}", ui.heat(_f(S.heat.loc[c, a]), 0, _f(S.heat.loc[c].max())))) for a in AGES] + [(_n(st_tot), "hl")]
                sick = _f(sk.also_sickle_cell.sum())
                row2 = [("Also sickle cell", "hl")] + [(("0", "", "") if not sick else _dot()) for _ in AGES] + [(_n(sick), "hl")]
                ui.html_table([("", "")] + [(a, a) for a in AS] + [("All", "")], [row1, row2], compact=True, widths=_eq(18, 9), min_width=460)
    with r:
        with ui.chart_card("Stroke patients with sickle cell"):
            if not sk.empty:
                young = _f(sk[sk.age_band.astype(str).str.contains("18-44")].stroke_patients.sum())
                _kpi([dict(label="Stroke, aged 18-44", value=_n(young), sub="patients", accent="blue"),
                      dict(label="Also sickle cell", value=f"{_n(sick)} of {_n(st_tot)}", sub="stroke patients", accent="crit")], "k2")
    if not sk.empty:
        ui.insight_bar(f"<strong>Stroke starts at 18, and {'none of it is sickle cell' if not sick else 'some of it is sickle cell'}.</strong> Stroke patients aged 18-44 are {_n(young)}, and "
                       f"{'none' if not sick else _n(sick)} of the {_n(st_tot)} stroke patients carries a sickle cell code. Stroke in children disappeared once default code pairs were excluded, so it was a coding artefact, not a finding.", "teal")


# ============================================================ F · sickle cell disease
def _section_f(S: _Data) -> None:
    ui.section_header("Section F — Sickle cell disease")
    sc, trait, sr = S.D["sickle_crisis"], S.D["sickle_trait"], S.D["sickle_returning"]
    c = "Sickle cell disease"
    total = _f(sc.patients.sum()) if not sc.empty else 0
    under25 = _f(sc[sc.age_group.isin(AGES[:4])].patients.sum()) if not sc.empty else 0
    carriers = _f(trait.carriers_identified.sum()) if not trait.empty else 0
    d = S.change(c)
    _kpi([dict(label="2024", value=_n(S.year(c, 2024)), sub="sickle cell patients", accent="blue"),
          dict(label="2025", value=_n(S.year(c, 2025)), sub="patients" + (f" · {'down' if d < 0 else 'up'} {abs(d):.0f}%" if d is not None else ""), accent="blue"),
          dict(label="2026 so far", value=_n(S.year(c, 2026)), sub="year not finished", accent="neutral"),
          dict(label="All years", value=_n(total), sub=f"{_n(under25)} under 25 · {_n(carriers)} carriers (trait) reported separately", accent="crit")], "k4")
    grp = [("0-4", [0]), ("5-12", [1]), ("13-17", [2]), ("18-24", [3]), ("25+", [4, 5, 6, 7])]
    with ui.chart_card("How sickle cell patients arrive, by age group",
                       note="Each bar = 100% of that age group’s patients, split by whether the visit was coded as a crisis. Under 25 most are crisis; after 25 there are almost no patients at all."):
        if not vi._empty(sc):
            def cnt(status, idx):
                return sum(_f(sc[(sc.crisis_status == status) & (sc.age_group == AGES[i])].patients.sum()) for i in idx)
            other = [s for s in sc.crisis_status.unique() if s not in ("With crisis", "Without crisis")]
            ser = [("With crisis", [cnt("With crisis", ix) for _, ix in grp], RED, "#fff"), ("Without crisis", [cnt("Without crisis", ix) for _, ix in grp], BLUE, "#fff"),
                   ("Unspecified", [sum(cnt(o, ix) for o in other) for _, ix in grp], LGREY, ui.TEXT)]
            _plot(_stack100([g for g, _ in grp], ser, 330), "ncd-sickle")
    if not sc.empty:
        crisis = _f(sc[sc.crisis_status == "With crisis"].patients.sum())
        ch = _f(sc[(sc.crisis_status == "With crisis") & (sc.age_group == AGES[1])].patients.sum())
        child = _f(sc[sc.age_group == AGES[1]].patients.sum())
        old = _f(sc[~sc.age_group.isin(AGES[:4])].patients.sum())
        multi = _f(sr.multi_year_patients.sum()) if not sr.empty else 0
        seen = _f(sr.patients.sum()) if not sr.empty else 0
        ui.insight_bar(f"<strong>Half of sickle cell patients arrive in crisis, and almost all of them are under 25.</strong> {_n(crisis)} of {_n(total)} patients were coded with a crisis, {_n(ch)} of the {_n(child)} aged 5-12. "
                       "A crisis is the acute, painful emergency of the disease, so Kapsabet meets sickle cell mostly at its worst moments, not in routine care. That matters for two reasons: "
                       f"only {_n(multi)} of {_n(seen)} patients are seen in more than one year, so there is little sign of steady follow-up, and only {_n(old)} patients are older than 24, "
                       "so the adults with sickle cell are either not diagnosed or not reaching this hospital. Numbers are small, so read the pattern, not each cell.", "red")


# ============================================================ page
def render() -> None:
    ui.inject_css()
    st.markdown(vi._CSS, unsafe_allow_html=True)
    st.markdown(_CSS, unsafe_allow_html=True)
    if q.DEMO:
        ui.demo_banner()
    _top = st.container()
    with ui.loading("NCDs"):
        S = _Data(_load())
        maps = dict(burden=_burden_payload(S), recur=_recurrent_payload(S))
    with _top:
        ui.page_header("NCDs", "Which non-communicable diseases bring people to Kapsabet CRH, who has them, where they come from, who keeps coming back, and where the data does not behave as expected", icon="ncd", stat=(_n(S.hl["All years"].ncd_patients) if S.hl.get("All years") is not None else "n/a", "patients with an NCD"))
    ui.kpi_row(_kpis(S))
    _top_insight(S)
    _section_a(S, maps)
    _section_b(S, maps)
    _section_c(S)
    _section_d(S)
    _section_e(S)
    _section_f(S)
    a, allr = S.hl.get("2024"), S.hl.get("All years")
    st_ = _recur_stats(S)
    ui.synthesis(f"Kapsabet sees NCDs mostly in people over 55 and almost entirely from within Nandi, at a steady {a.pct_ncd:.0f}% of patients. "
                 f"Most come once: only {100 * st_['both'] / max(1, st_['n']):.0f}% are seen in both 2024 and 2025, even for conditions that need lifelong follow-up. "
                 "Three patterns break the expected shape: heart failure in young women, Type 1 diabetes in adults, and sickle cell that is seen in crisis and then disappears after youth. "
                 "They are carried to the <a href='?tab=tab_dataquality' target='_self' style='color:#2E8B3A;font-weight:700;text-decoration:none'>Data quality tab</a> as items to check, not conclusions." if a is not None else "")
