"""
views_mental.py: Tab 5, Psychiatric Conditions.

No SQL here: every number comes from queries.q5_* (mental_health_queries_v3.sql). This file shapes DataFrames into heatmap tables,
a pyramid, the ward map (template/geo_engine.js, kind "ncd" in burden mode) and ui_template cards.
Sections: A burden at a glance · B who is affected · C where they come from · D other illnesses alongside ·
E patients who keep coming back · F patterns that do not fit.

Definition: ICD-11 chapter 6 diagnoses, counted by patient, default code pair excluded (see queries.py, Tab 5 block).
Data-quality items (the 6E20 perinatal code, wards that do not match a boundary, the 2025 export gap, ICD coverage) belong
on the Data Quality tab unless they limit a specific finding, in which case they appear as a note on that finding.
"""
from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import nandi_geo_map as geo
import nandi_queries as q
import nandi_ui_template as ui
import nandi_views_geo as gv
import nandi_views_infectious as vi

BLUE, TEAL, AMBER, RED = ui.BLUE, ui.PRIMARY, ui.AMBER, ui.CRITICAL
MUTED = ui.TEXT_MUTED
AGES = vi.AGES
AS = ["0-4", "5-12", "13-17", "18-24", "25-34", "35-44", "45-54", "55+"]
ANY = "Any mental health"
CONDS = ["Schizophrenia", "Other psychotic disorders", "Bipolar disorder", "Alcohol & substance use", "Anxiety", "Dementia", "Depression"]
EXTRA = ["Trauma & stress-related"]          # added by the v3 definition; shown only when the data has it
SHORT = {"Other psychotic disorders": "Other psychotic", "Alcohol & substance use": "Alcohol & substance", "Bipolar disorder": "Bipolar",
         "Trauma & stress-related": "Trauma & stress"}
KPI_CONDS = ["Schizophrenia", "Other psychotic disorders", "Bipolar disorder", "Alcohol & substance use", "Dementia"]
NCDS = [("Hypertension", "with_hypertension"), ("Diabetes", "with_diabetes"), ("Epilepsy", "with_epilepsy"), ("Heart failure", "with_heart_failure"),
        ("Kidney disease", "with_ckd"), ("Asthma", "with_asthma"), ("Stroke", "with_stroke")]
BANDS = ["Under 18", "18-54", "55+"]
_n, _f, _cap, _plot, _eq = vi._n, vi._f, vi._cap, vi._plot, vi._eq

_CSS = f"""
<style>
.psy-lg {{ display:flex; gap:14px; flex-wrap:wrap; font-size:11px; color:{MUTED}; margin-top:6px; }}
.psy-lg i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; }}
.psy-kv {{ font-size:20px; font-weight:600; margin:2px 0 8px; color:{ui.TEXT}; }}
.psy-p {{ font-size:13px; line-height:1.5; color:{ui.TEXT}; margin:0 0 8px; }}
.psy-p b {{ font-weight:700; }}
.psy-kp3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:0 0 12px; }}
@media(max-width:1100px) {{ .psy-kp3 {{ grid-template-columns:1fr; }} }}
</style>
"""


# ============================================================ helpers
def _s(c: str) -> str:
    return SHORT.get(c, c)


def _label(c: str) -> tuple:
    return (f"<b>{c}</b>", "hl") if c == ANY else (_s(c), "hl")


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


def _pct_change(a: float, b: float) -> float | None:
    return None if not a else (b - a) / a * 100


def _arrow(c: str, S: "_Data") -> str:
    """Coloured arrow and % change 2024 to 2025 for the KPI cards (more patients = more burden, so up is red)."""
    p = _pct_change(S.n(c, "2024"), S.n(c, "2025"))
    r = 0 if p is None else round(p)
    return f'{ui.arrow_html(r)} vs 2024'



def _load() -> dict[str, pd.DataFrame]:
    names = ["yearly", "age_sex", "comorbidity", "wards", "returning", "oddities"]
    D = {n: _num(getattr(q, "q5_" + n)().copy()) for n in names}
    D["baseline"] = _num(q.q4_baseline().copy())
    return D


class _Data:
    """The frames and lookups the sections share, computed once per render."""

    def __init__(self, D: dict[str, pd.DataFrame]):
        self.D = D
        y = D["yearly"]
        self.y: dict[tuple, tuple] = {}
        if not y.empty:
            for r in y.itertuples():
                self.y[(r.condition_name, str(r.period))] = (_f(r.patients), _f(r.pct_of_patients), _f(r.all_patients))
        present = set(D["yearly"].condition_name) if not D["yearly"].empty else set()
        self.conds = CONDS + [c for c in EXTRA if c in present]
        # age x sex counts per condition
        self.age: dict[str, dict[str, list[float]]] = {}
        a = D["age_sex"]
        if not a.empty:
            for r in a.itertuples():
                if r.age_group_at_first_diagnosis not in AGES or r.sex not in ("F", "M"):
                    continue
                d = self.age.setdefault(r.condition_name, {"F": [0.0] * 8, "M": [0.0] * 8})
                d[r.sex][AGES.index(r.age_group_at_first_diagnosis)] += _f(r.patients)
        # hospital female share by age group
        self.base = [None] * 8
        b = D["baseline"]
        if not b.empty:
            for i, g in enumerate(AGES):
                f = _f(b[(b.age_group == g) & (b.sex == "F")].distinct_patients.sum())
                m = _f(b[(b.age_group == g) & (b.sex == "M")].distinct_patients.sum())
                self.base[i] = 100 * f / (f + m) if f + m else None
        self.cm = {r.condition_name: r for r in D["comorbidity"].itertuples()} if not D["comorbidity"].empty else {}
        self.rec: dict[str, dict[str, tuple]] = {}
        if not D["returning"].empty:
            for r in D["returning"].itertuples():
                self.rec.setdefault(r.condition_name, {})[r.age_band] = (_f(r.patients), _f(r.seen_both_years), _f(r.total_visits))

    def n(self, c: str, p: str) -> float:
        return self.y.get((c, p), (0, 0, 0))[0]

    def pct(self, c: str, p: str) -> float:
        return self.y.get((c, p), (0, 0, 0))[1]

    def allp(self, p: str) -> float:
        return self.y.get((ANY, p), (0, 0, 0))[2]

    def outside_nandi(self) -> float:
        a = self.D["age_sex"]
        if a.empty:
            return 0.0
        d = a[a.condition_name == ANY]
        if d.empty:
            d = a[a.condition_name.isin(self.conds)]
        return _f(d.from_outside_nandi.sum())


# ============================================================ maps
def _ward_payload(S: _Data) -> dict | None:
    df = S.D["wards"]
    if df.empty or "latitude" not in df.columns:
        return None
    G = {}
    for r in df.drop_duplicates(["ward", "sub_county"]).itertuples():
        if pd.isna(r.latitude) or pd.isna(r.longitude):
            continue
        G[(r.ward, r.sub_county)] = dict(lat=round(float(r.latitude), 5), lon=round(float(r.longitude), 5), rings=gv._rings(r.boundary_geojson))
    wards = {k: dict(w=k[0], s=k[1], **g, v={}) for k, g in G.items()}
    unplaced: dict[str, int] = {}
    for r in df.itertuples():
        k = (r.ward, r.sub_county)
        if k in wards:
            wards[k]["v"][r.condition_name] = int(r.patients)
        else:
            unplaced[r.condition_name] = unplaced.get(r.condition_name, 0) + int(r.patients)
    conds = [ANY] + S.conds
    return dict(mode="burden", all=ANY, noun="mental health", hospital=dict(lat=q.HOSPITAL_LAT, lon=q.HOSPITAL_LON), conds=conds,
                labels=[ANY] + [_s(c) for c in S.conds], wards=list(wards.values()), unplaced=unplaced, h=560)


# ============================================================ header, KPIs
def _kpis(S: _Data) -> list[dict]:
    allr = S.n(ANY, "All years")
    cards = [dict(label=ANY, value=_n(allr), sub=f"{_arrow(ANY, S)} · {S.pct(ANY, 'All years'):.1f}% of {_n(S.allp('All years'))} patients", accent="blue")]
    for c in KPI_CONDS:
        cards.append(dict(label=_s(c) if c != "Alcohol & substance use" else c, value=_n(S.n(c, "All years")), sub=f"{_arrow(c, S)} · patients", accent="blue"))
    return cards


def _top_insight(S: _Data) -> None:
    a, b = S.pct(ANY, "2024"), S.pct(ANY, "2025")
    psy = S.n("Schizophrenia", "All years") + S.n("Other psychotic disorders", "All years")
    tot = S.n(ANY, "All years")
    da = S.n("Depression", "All years") + S.n("Anxiety", "All years")
    ui.insight_bar(f"<strong>Mental health is {b:.1f}% of Kapsabet patients in 2025, up from {a:.1f}% in 2024, and psychosis carries most of it.</strong> "
                   f"Schizophrenia and other psychotic disorders together are {_n(psy)} of {_n(tot)} patients, nearly three in four patients with a mental health diagnosis. "
                   f"Depression and anxiety are almost absent ({_n(da)} patients between them), which is far lower than community levels and points to under-diagnosis, not low need. "
                   "2026 is year-to-date and 2023 is a partial year, so neither is used for year-on-year change.", "blue")


# ============================================================ A · burden at a glance
def _year_heat(S: _Data) -> None:
    periods = ["2023", "2024", "2025", "2026"]
    heads = [("", "")] + [(p + (" (partial)" if p == "2023" else " so far" if p == "2026" else ""), p) for p in periods]
    mx = max([S.pct(c, p) for c in S.conds for p in periods] + [0.1])
    rows = []
    for c in [ANY] + S.conds:
        cells = [_label(c)]
        for p in periods:
            n, pc, tot = S.y.get((c, p), (0, 0, 0))
            if not n:
                cells.append(_dot())
                continue
            hi = 3.4 if c == ANY else mx * 1.05
            fade = "opacity:.45;" if p in ("2023", "2026") else ""
            cells.append(_cell(f"{pc:.2f}%", fade + ui.heat(pc, 0, hi), f"{c}, {p}: {int(n)} patients ({pc:.2f}% of {int(tot):,})"))
        rows.append(cells)
    ui.html_table(heads, rows, compact=True, widths=_eq(26, 4), min_width=420)
    _cap(f"Patients seen rose by {'under 1' if abs(S.allp('2025') / S.allp('2024') - 1) * 100 < 1 else format(abs(S.allp('2025') / S.allp('2024') - 1) * 100, '.0f')}% between 2024 and 2025 ({_n(S.allp('2024'))} to {_n(S.allp('2025'))}), "
         "so the jump is in mental health diagnoses, not in how many people came." if S.allp("2024") else "")


def _who_gets_what(S: _Data) -> None:
    rows = []
    for c in S.conds:
        d = S.age.get(c)
        if not d:
            continue
        t = [f + m for f, m in zip(d["F"], d["M"])]
        tot = sum(t)
        cells = [(_s(c), "hl")]
        for i, a in enumerate(AGES):
            cells.append(_dot() if not t[i] else _cell(f"{100 * t[i] / tot:.0f}%", ui.heat(100 * t[i] / tot, 0, 40), f"{c}, {a}: {int(t[i])} of {int(tot)} patients"))
        rows.append(cells)
    ui.html_table([("", "")] + [(a, a) for a in AS], rows, compact=True, widths=_eq(26, 8), min_width=420)
    _cap("Share of each condition’s patients by age group at first diagnosis. Darker = bigger share.")


def _growth_insight(S: _Data) -> None:
    def g(c):
        p = _pct_change(S.n(c, "2024"), S.n(c, "2025"))
        return 0 if p is None else round(p)
    dd = g("Dementia")
    dem_txt = "did not move" if dd == 0 else f"moved {dd}%"
    ui.insight_bar(f"<strong>Patients with a mental health diagnosis grew {g(ANY)}% in one year ({_n(S.n(ANY, '2024'))} to {_n(S.n(ANY, '2025'))}) while total patients were flat.</strong> "
                   f"Other psychotic disorders went from {_n(S.n('Other psychotic disorders', '2024'))} to {_n(S.n('Other psychotic disorders', '2025'))}, schizophrenia and bipolar grew {g('Schizophrenia')}% and {g('Bipolar disorder')}%, "
                   f"dementia {dem_txt} ({_n(S.n('Dementia', '2024'))} and {_n(S.n('Dementia', '2025'))}). "
                   "Part of this may be better recording, not more illness: the code set and clinician habits can change between years. "
                   "The 2025 export gap and small counts for anxiety and depression limit how far it can be read.", "amber")


def _section_a(S: _Data) -> None:
    ui.section_header("Section A — Burden at a glance")
    l, r = st.columns([3, 2])
    with l:
        with ui.chart_card("Share of all patients with each condition, by year",
                           note="% of that year’s Kapsabet patients carrying the diagnosis. Faded columns are partial years (2023, 2026 so far). Hover for patient counts."):
            _year_heat(S)
    with r:
        with ui.chart_card("Who gets what"):
            _who_gets_what(S)
    _growth_insight(S)


# ============================================================ B · who is affected
def _sex_heat(S: _Data) -> None:
    rows = []
    for c in S.conds:
        d = S.age.get(c)
        if not d:
            continue
        cells = [(_s(c), "hl")]
        for i, a in enumerate(AGES):
            f, m = d["F"][i], d["M"][i]
            n = f + m
            base = S.base[i]
            if not n or base is None:
                cells.append(_dot())
                continue
            sh = 100 * f / n
            cells.append(_cell(f"{sh:.0f}%", ("opacity:.4;" if n < 15 else "") + ui.diverge(sh, base, 35),
                               f"{c}, {a}: {int(f)} women, {int(m)} men ({sh:.0f}% women; hospital norm {base:.0f}%)"))
        rows.append(cells)
    ui.html_table([("", "")] + [(a, a) for a in AS], rows, compact=True, widths=_eq(26, 8), min_width=420)
    st.markdown('<div class="psy-lg"><span><i style="background:%s"></i>More women than the hospital norm</span><span><i style="background:%s"></i>More men than the hospital norm</span></div>' % (TEAL, AMBER),
                unsafe_allow_html=True)


def _pyramid(S: _Data) -> None:
    d = S.age.get("Schizophrenia")
    if not d:
        _cap("No data returned for this card.")
        return
    f, m = d["F"], d["M"]
    mx = max(f + m + [1])
    fig = go.Figure()
    fig.add_trace(go.Bar(y=AGES, x=[-v for v in f], orientation="h", name="Women", marker_color=TEAL, customdata=f, text=[int(v) if v else "" for v in f],
                         textposition="outside", cliponaxis=False, hovertemplate="%{y} · Women: %{customdata:,.0f}<extra></extra>"))
    fig.add_trace(go.Bar(y=AGES, x=m, orientation="h", name="Men", marker_color=BLUE, text=[int(v) if v else "" for v in m],
                         textposition="outside", cliponaxis=False, hovertemplate="%{y} · Men: %{x:,.0f}<extra></extra>"))
    ui._finish(fig, 340, barmode="overlay", margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom"))
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.update_xaxes(range=[-mx * 1.25, mx * 1.25], showticklabels=False, showgrid=False)
    _plot(fig, "psy-pyramid")


def _sex_insight(S: _Data) -> None:
    d = S.age.get("Schizophrenia", {"F": [0] * 8, "M": [0] * 8})
    ui.insight_bar(f"<strong>Schizophrenia flips by age: men lead between 25 and 44, women lead from 45 up.</strong> In the 25-34 group there are {_n(d['M'][4])} men and {_n(d['F'][4])} women; "
                   f"at 55+ there are {_n(d['F'][7])} women and {_n(d['M'][7])} men. Bipolar disorder is more even, with women ahead in nearly every adult group. "
                   "Expected pattern for men is earlier onset; the late female excess may be women kept in care longer or recorded differently, and cannot be told apart in this data.", "amber")


def _section_b(S: _Data) -> None:
    ui.section_header("Section B — Who is affected")
    l, r = st.columns(2)
    with l:
        with ui.chart_card("Women vs men, by condition and age",
                           note="Female share of each condition’s patients, compared with the hospital’s own mix at that age. Green = more women than usual, amber = fewer. Faded = fewer than 15 patients."):
            _sex_heat(S)
    with r:
        with ui.chart_card("Schizophrenia: who, by age and sex", note="Patients by age group at first diagnosis. Left = women, right = men."):
            _pyramid(S)
    _sex_insight(S)


# ============================================================ C · where they come from
def _section_c(S: _Data, payload: dict | None) -> None:
    ui.section_header("Section C — Where they come from")
    with ui.chart_card("Where they come from", note="Patients by ward (Nandi residents, 2024-2025). Pick a condition. Every ward is drawn from its real boundary."):
        if payload:
            out = S.outside_nandi()
            payload["outside"] = f"Outside Nandi: {_n(out)} patients with a recorded county"
            geo.render_geo("ncd", payload, height=705, scrolling=False)
        else:
            _cap("No ward data returned for this card.")
    df = S.D["wards"]
    if not df.empty:
        a = df[df.condition_name == ANY].sort_values("patients", ascending=False)
        if not a.empty:
            tot, top = _f(a.patients.sum()), a.iloc[0]
            nxt = _f(a.iloc[1].patients) if len(a) > 1 else 0
            share = 100 * _f(top.patients) / tot if tot else 0
            word = "Nine in ten" if 87 <= share <= 93 else f"{share:.0f}%"
            ui.insight_bar(f"<strong>{word} mental health patients with a recorded ward live in {top.ward} ward.</strong> {_n(top.patients)} of {_n(tot)} patients live in {top.ward}, the next ward has {_n(nxt)}. "
                           f"Almost no one comes from outside Nandi ({_n(S.outside_nandi())} patients). The hospital’s catchment for mental health is the town itself; the outer wards either use other facilities "
                           "or do not reach one. A ward-by-ward rate is not possible without population figures here.", "blue")


# ============================================================ D · other illnesses alongside
def _section_d(S: _Data) -> None:
    ui.section_header("Section D — Other illnesses alongside")
    with ui.chart_card("Physical NCDs carried by mental health patients", note="% of each condition’s patients who also carry the NCD. Darker = higher %. Hover for patient counts."):
        rows = []
        for c in S.conds:
            r = S.cm.get(c)
            if r is None:
                continue
            tot = _f(r.patients)
            cells = [(_s(c), "hl")]
            for name, col in NCDS:
                v = _f(getattr(r, col, 0))
                cells.append(_dot() if not v else _cell(f"{100 * v / tot:.1f}%", ui.heat(100 * v / tot, 0, 16), f"{c} with {name}: {int(v)} of {int(tot)} patients"))
            cells.append((_n(tot), "hl"))
            rows.append(cells)
        ui.html_table([("", "")] + [(n, n) for n, _ in NCDS] + [("Patients", "")], rows, compact=True, widths=_eq(18, len(NCDS) + 1), min_width=560)
        _cap("Treat as a minimum: most visits record a single diagnosis, so a second condition is only seen when the clinician wrote both.")
    sch, per, dem, psy = S.cm.get("Schizophrenia"), S.cm.get("Other psychotic disorders"), S.cm.get("Dementia"), None
    if sch is not None and per is not None and dem is not None:
        ui.insight_bar(f"<strong>Physical NCDs are rare next to mental illness, except stroke in dementia ({100 * _f(dem.with_stroke) / _f(dem.patients):.0f}%).</strong> Everything else is under 5%: "
                       f"epilepsy sits highest in psychotic disorders ({_n(per.with_epilepsy)} of {_n(per.patients)}, {100 * _f(per.with_epilepsy) / _f(per.patients):.1f}%), "
                       f"diabetes in schizophrenia is {_n(sch.with_diabetes)} of {_n(sch.patients)} ({100 * _f(sch.with_diabetes) / _f(sch.patients):.1f}%) and hypertension {_n(sch.with_hypertension)} "
                       f"({100 * _f(sch.with_hypertension) / _f(sch.patients):.1f}%). People with serious mental illness usually carry more diabetes and heart disease than average, so these low figures "
                       "suggest physical illness is not being recorded at the mental health visit, not that it is absent.", "amber")


# ============================================================ E · patients who keep coming back
def _band_sum(S: _Data, c: str) -> tuple[float, float, float]:
    v = [S.rec.get(c, {}).get(b, (0, 0, 0)) for b in BANDS]
    return sum(x[0] for x in v), sum(x[1] for x in v), sum(x[2] for x in v)


def _section_e(S: _Data) -> None:
    ui.section_header("Section E — Patients who keep coming back")
    st.markdown(f'<div class="afya-note" style="margin:-4px 0 8px">Returning = seen in both 2024 and 2025, with at least one mental health visit.</div>', unsafe_allow_html=True)
    n, both, vis = _band_sum(S, ANY)
    sn, sb, _ = _band_sum(S, "Schizophrenia")
    cards = [("Seen in both 2024 and 2025", f"{_n(both)} of {_n(n)}", f"{100 * both / n:.1f}% of mental health patients" if n else "", AMBER),
             ("Schizophrenia", f"{_n(sb)} of {_n(sn)}", f"{100 * sb / sn:.1f}% return, the highest of any condition" if sn else "", BLUE),
             ("Average visits per patient", f"{vis / n:.1f}" if n else "n/a", "across 2024-2025", BLUE)]
    html = "".join(f'<div class="afya-kpi" style="border-top-color:{col}"><div class="afya-kpi-label">{a}</div><div class="afya-kpi-value">{b}</div><div class="afya-kpi-sub">{c}</div></div>' for a, b, c, col in cards)
    st.markdown(f'<div class="psy-kp3">{html}</div>', unsafe_allow_html=True)
    with ui.chart_card("Which conditions come back", note="% of each condition’s patients seen in both years, by age band. Faded = fewer than 15 patients. Hover for counts."):
        rows = []
        for c in [ANY] + S.conds:
            if c not in S.rec:
                continue
            cells = [_label(c)]
            for b in BANDS:
                p, s, _v = S.rec[c].get(b, (0, 0, 0))
                cells.append(_dot() if not p else _cell(f"{100 * s / p:.1f}%", ("opacity:.4;" if p < 15 else "") + ui.heat(100 * s / p, 0, 8), f"{c}, {b}: {int(s)} of {int(p)} patients seen in both years"))
            tp, ts, _tv = _band_sum(S, c)
            cells.append(_cell(f"{100 * ts / tp:.1f}%", ui.heat(100 * ts / tp, 0, 8), f"{c}: {int(ts)} of {int(tp)}") if tp else _dot())
            rows.append(cells)
        ui.html_table([("", "")] + [(b, b) for b in BANDS] + [("All", "")], rows, compact=True, widths=_eq(26, 4), min_width=420)
    bp, bs, _bv = _band_sum(S, "Bipolar disorder")
    ui.insight_bar(f"<strong>Only {_n(both)} of {_n(n)} mental health patients ({100 * both / n:.0f}%) are seen in both 2024 and 2025, even though these are long-term conditions.</strong> "
                   f"Schizophrenia returns at {100 * sb / sn:.1f}% ({_n(sb)} of {_n(sn)}), bipolar at {100 * bs / bp:.1f}% ({_n(bs)} of {_n(bp)}), and most other conditions at under 2%. "
                   f"Patients average {vis / n:.1f} visits. Schizophrenia and bipolar need continuous treatment, so either patients collect it elsewhere or this register does not link them across visits. "
                   "Both are worth knowing. The 2025 export gap limits how far this can be pushed, so read it as what the register can show, not proof that patients are lost." if n and sn and bp else "",
                   "amber")


# ============================================================ F · patterns that do not fit
def _odd_card(title: str, colour: str, headline: str, seen: str, rules: str, check: str) -> None:
    key = "card-" + re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40]
    st.markdown(f"<style>div.st-key-{key} {{{{ border-top:3px solid {colour} !important; }}}}</style>".replace("{{", "{").replace("}}", "}"), unsafe_allow_html=True)
    with ui.chart_card(title):
        st.markdown(f'<div class="psy-kv" style="border-top:0">{headline}</div>'
                    f'<div class="psy-p"><b>What the data says.</b> {seen}</div>'
                    f'<div class="psy-p"><b>What it rules out.</b> {rules}</div>'
                    f'<div class="psy-p"><b>What to check.</b> {check}</div>', unsafe_allow_html=True)


def _section_f(S: _Data) -> None:
    ui.section_header("Section F — Patterns that do not fit")
    st.markdown('<div class="afya-note" style="margin:-4px 0 8px">Counts that go against what the condition normally looks like. Listed to be checked, not concluded.</div>', unsafe_allow_html=True)
    o = S.D["oddities"]
    if o.empty:
        _cap("No data returned for this section.")
        return

    def tot(name, **kw):
        d = o[o.oddity == name]
        for k, v in kw.items():
            d = d[d[k] == v] if not isinstance(v, (list, tuple)) else d[d[k].isin(v)]
        return int(_f(d.patients.sum()))
    s_all = tot("Schizophrenia under 13")
    s_m = tot("Schizophrenia under 13", sex="M")
    d_s = o[o.oddity == "Schizophrenia under 13"]
    s_only = int(_f(d_s[d_s.icd11_codes.astype(str).str.strip() == "6A20"].patients.sum()))
    a_all, a_m = tot("Alcohol / substance, under 18"), tot("Alcohol / substance, under 18", sex="M")
    a_teen, a_kid = tot("Alcohol / substance, under 18", age_band="13-17"), tot("Alcohol / substance, under 18", age_band="Under 13")
    m_all, m_m = tot("Dementia under 55"), tot("Dementia under 55", sex="M")
    d_d = o[o.oddity == "Dementia under 55"]
    head = int(_f(d_d[d_d.icd11_codes.astype(str).str.startswith("6D85.7")].patients.sum()))
    ftd = int(_f(d_d[d_d.icd11_codes.astype(str).str.startswith("6D83")].patients.sum()))
    down = int(_f(d_d[d_d.icd11_codes.astype(str).str.startswith("6D85.9")].patients.sum()))
    young = tot("Dementia under 55", age_band=["Under 13", "13-17"])
    c1, c2, c3 = st.columns(3)
    with c1:
        _odd_card("Schizophrenia in children under 13", RED, f"{s_all} patients, {'all boys' if s_all == s_m else f'{s_m} boys'}",
                  f"Schizophrenia before adolescence is <b>very rare</b>, and <b>{'every one of the ' + str(s_all) + ' is male' if s_all == s_m else str(s_m) + ' of the ' + str(s_all) + ' are male'}</b>. "
                  f"<b>{s_only} of the {s_all}</b> have schizophrenia as their <b>only diagnosis</b>.",
                  "<b>Not a text-matching slip</b>: the diagnosis is the ICD-11 code <b>6A20</b> itself.",
                  "Check the <b>registered age</b> (a parent or carer age, or a day-and-month swap), and whether a <b>different code</b> was meant, such as a childhood developmental or psychotic condition.")
    with c2:
        _odd_card("Alcohol and substance use under 18", AMBER, f"{a_all} patients, {a_m} boys",
                  f"<b>{a_teen} are 13-17 and {a_kid} are under 13</b>. <b>Cocaine, MDMA and psychostimulant</b> codes appear in the under-13s.",
                  "<b>Not a loose text match</b>: the codes name <b>specific drugs</b>, not a general substance disorder.",
                  "Check <b>ages</b> and whether a <b>parent or guardian</b> was recorded as the patient; confirm any real cases with the clinic.")
    with c3:
        _odd_card("Dementia before age 55", AMBER, f"{m_all} patients, {m_m} men",
                  f"<b>Head-injury dementia ({head}), frontotemporal dementia ({ftd}) and Down syndrome dementia ({down})</b> are real early-onset causes. <b>The rest need checking.</b>",
                  f"<b>Not all {m_all} are errors</b>: early onset is clinically possible for these causes.",
                  f"Check the <b>cases in children and teenagers ({young} of the {m_all})</b>, and the entries that also carry a <b>perinatal code</b>.")


# ============================================================ page
def render() -> None:
    ui.inject_css()
    st.markdown(vi._CSS, unsafe_allow_html=True)
    st.markdown(_CSS, unsafe_allow_html=True)
    if q.DEMO:
        ui.demo_banner()
    _top = st.container()
    with ui.loading("Psychiatric Conditions"):
        S = _Data(_load())
        payload = _ward_payload(S)
    with _top:
        ui.page_header("Psychiatric Conditions", "Which psychiatric conditions bring people to Kapsabet CRH, who has them, where they come from, who keeps coming back, and where the data does not behave as expected", icon="mental", stat=(_n(S.n(ANY, "All years")), "patients with a mental health diagnosis"))
    ui.kpi_row(_kpis(S))
    _top_insight(S)
    _section_a(S)
    _section_b(S)
    _section_c(S, payload)
    _section_d(S)
    _section_e(S)
    _section_f(S)
    n, both, vis = _band_sum(S, ANY)
    ui.synthesis(f"Kapsabet sees mental health mostly as psychosis in adults, almost all from Kapsabet town, and the diagnosed share grew from {S.pct(ANY, '2024'):.1f}% to {S.pct(ANY, '2025'):.1f}% in a year. "
                 f"Depression and anxiety are close to missing, physical illness alongside is rarely written down, and only {100 * both / max(1, n):.0f}% of patients return in a second year. "
                 "Three counts break the expected shape: schizophrenia in children under 13, substance use under 18, and dementia under 55. "
                 "They are carried to the <a href='?tab=tab_dataquality' target='_self' style='color:#2E8B3A;font-weight:700;text-decoration:none'>Data quality tab</a> as items to check, not conclusions.")
