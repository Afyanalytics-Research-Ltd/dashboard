"""
views_maternal.py: Tab 7, Maternal & Obstetric.

No SQL here: every number comes from queries.q7_* (maternal_queries.sql: Q7.1b, Q7.3, Q7.4, Q7.8, Q7.9, Q7.10b, Q7.11, Q7.16 to Q7.19).
This file shapes DataFrames into KPI cards, category cards, stacked bars, one comorbidity table, the ward map
(template/geo_engine.js, kind "ncd" in burden mode) and ui_template cards.
Sections: A maternal conditions at a glance · B pregnancy loss · C hypertensive disorders · D what else they have ·
E teenage pregnancy · F where they come from · G who comes back.

Definition: ICD-11 chapter 18 (JA, JB) diagnoses, counted by patient, default code pair excluded (see queries.py, Tab 7 block).
Data-quality items (Q7.0c code dictionary, Q7.0d sex and age check, Q7.6, Q7.7, the JA10 "abdominal pregnancy" jump) belong on the
Data Quality tab and are not run here.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import nandi_geo_map as geo
import nandi_queries as q
import nandi_ui_template as ui
import nandi_views_geo as gv
import nandi_views_infectious as vi

BLUE, TEAL, AMBER, RED = ui.BLUE, ui.PRIMARY, ui.AMBER, ui.CRITICAL
MUTED = ui.TEXT_MUTED
_n, _f, _cap, _eq = vi._n, vi._f, vi._cap, vi._eq

ANY = "Any maternal"
CATS = ["Pregnancy loss", "Infections in pregnancy", "Maternal complications", "High-risk pregnancy", "Postpartum complications",
        "Labour & delivery", "ANC (Antenatal care)", "Maternal medical conditions", "Other"]
MAP_CATS = [c for c in CATS if c != "Other"]
CATN = {"ANC (Antenatal care)": "Antenatal care"}
# (category key or None, card title, colour, the conditions in the card)
GROUPS = [
    ("Pregnancy loss", "Pregnancy loss", "#0C447C", ["Spontaneous abortion", "Ectopic & threatened abortion", "Abdominal/ectopic pregnancy", "Other pregnancy loss"]),
    ("Infections in pregnancy", "Infections in pregnancy", "#0F6E56", ["GU infection (general)", "GU infection (site-specific)"]),
    ("Maternal complications", "Maternal complications", "#8B6BB5", ["Hyperemesis gravidarum", "Hypertensive disorder", "Obstetric haemorrhage"]),
    ("High-risk pregnancy", "High-risk pregnancy", "#4B9CA8", ["Fetal/delivery-problem care", "Previous uterine scar"]),
    ("Postpartum complications", "Postpartum complications", "#B08968", ["Puerperal infection & sepsis", "Puerperal mastitis"]),
    ("Labour & delivery", "Labour & delivery", "#6B93C4", ["Delivery", "Labour/delivery complication"]),
    (None, "Antenatal & other", "#6B7280", ["Antenatal screening finding", "Anaemia & other obstetric condition"]),
]
CONDN = {"Ectopic & threatened abortion": "Threatened abortion & ectopic", "Spontaneous abortion": "Spontaneous / unspecified abortion",
         "Abdominal/ectopic pregnancy": "Abdominal pregnancy (as coded)", "GU infection (general)": "Genitourinary infection",
         "GU infection (site-specific)": "Genitourinary infection, site given", "Fetal/delivery-problem care": "Fetal problems & high-risk care",
         "Labour/delivery complication": "Labour complication", "Anaemia & other obstetric condition": "Anaemia & other obstetric"}
AGEC = ["#E0A458", "#6B93C4", "#0C447C", "#0F6E56", "#9CA3AF"]
LOSS_BANDS = ["Under 18", "18-24", "25-34", "35-44", "45+"]
LOSS_TYPES = [("JA05", "Spontaneous / unspecified abortion"), ("JA00", "Threatened / other abortion"), ("JA10", "Abdominal pregnancy (as coded)"),
              ("JA01", "Ectopic"), ("JA02", "Molar"), ("Other loss code", "Other loss codes")]
HTN_TYPES = ["Pre-eclampsia", "Pre-existing hypertension / renal", "Gestational hypertension", "Superimposed pre-eclampsia", "Eclampsia"]
COMORB = [("Diabetes", "with_diabetes"), ("Anaemia", "with_anaemia"), ("Malaria", "with_malaria"), ("HIV", "with_hiv"),
          ("UTI", "with_uti"), ("Hypertension", "with_hypertension"), ("Sepsis", "with_sepsis")]
# Cells where the category is defined by that diagnosis, so the figure is not a finding.
DEFINING = {("Maternal medical conditions", "Diabetes"), ("Maternal medical conditions", "Anaemia"), ("Maternal complications", "Hypertension"),
            ("Postpartum complications", "Sepsis"), ("Postpartum complications", "UTI"), ("Infections in pregnancy", "UTI")}
HATCH = "background:repeating-linear-gradient(45deg,#F3F4F6,#F3F4F6 4px,#E5E7EB 4px,#E5E7EB 8px);color:#9CA3AF;font-weight:600"
RBANDS = ["Under 18", "18-24", "25-34", "35+"]
TEEN_GROUPS = ["13-15", "16-17", "18-19"]

_CSS = f"""
<style>
.mt-kp3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:0 0 12px; }}
.mt-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }}
@media(max-width:1100px) {{ .mt-grid {{ grid-template-columns:repeat(2,1fr); }} .mt-kp3 {{ grid-template-columns:1fr; }} }}
.mt-card {{ background:#fff; border:1px solid {ui.BORDER}; border-radius:8px; padding:14px 14px 8px; box-shadow:0 1px 3px rgba(0,0,0,.04); }}
.mt-card.wide {{ grid-column:span 2; }}
.mt-card .ct {{ font-size:13px; font-weight:600; }}
.mt-card .cs {{ font-size:11px; color:{MUTED}; margin-bottom:8px; min-height:15px; }}
.mt-item {{ padding:3px 0 6px; }}
.mt-item .r1 {{ display:flex; justify-content:space-between; font-size:12.5px; }}
.mt-item .tr {{ height:9px; background:#F0F1F5; border-radius:5px; margin-top:3px; }}
.mt-item .tr i {{ display:block; height:9px; border-radius:5px; }}
.mt-st {{ display:grid; grid-template-columns:var(--lw,210px) 1fr 44px; gap:8px; align-items:center; padding:4px 0; font-size:12.5px; }}
.mt-st .bar {{ display:flex; height:20px; border-radius:4px; overflow:hidden; min-width:4px; }}
.mt-st .bar div {{ color:#fff; font-size:10px; font-weight:600; display:flex; align-items:center; justify-content:center; }}
.mt-st b {{ text-align:right; }}
.mt-db {{ display:grid; grid-template-columns:150px 1fr 100px; align-items:center; gap:8px; height:34px; font-size:12px; }}
.mt-db .tk {{ position:relative; height:14px; }}
.mt-db .tk .ln {{ position:absolute; left:0; right:0; top:6px; height:2px; background:#E5E7EB; }}
.mt-db .tk .sg {{ position:absolute; top:6px; height:3px; }}
.mt-db .tk i {{ position:absolute; top:1px; width:12px; height:12px; border-radius:50%; }}
.mt-lg {{ display:flex; gap:14px; flex-wrap:wrap; font-size:11px; color:{MUTED}; margin-top:8px; }}
.mt-lg i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; }}
.mt-bars {{ display:grid; grid-template-columns:160px 1fr 40px; gap:8px; align-items:center; padding:3px 0; font-size:12px; }}
.mt-bars .b {{ height:14px; background:#F0F1F5; border-radius:0 6px 6px 0; }}
.mt-bars .b i {{ display:block; height:100%; border-radius:0 6px 6px 0; }}
.mt-bars .v {{ text-align:right; font-weight:600; }}
</style>
"""


# ============================================================ helpers
def _cn(k: str) -> str:
    return CONDN.get(k, k)


def _catn(k: str) -> str:
    return CATN.get(k, k)


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


def _r(a: float, b: float) -> int:
    p = _pct_change(a, b)
    return 0 if p is None else round(p)


def _arrow(r: int, up_is: str = "bad") -> str:
    return ui.arrow_html(r, up_is, flat="0%")


def _moved(r: int) -> str:
    return f"{'rose' if r > 0 else 'fell' if r < 0 else 'held'} {abs(r)}%" if r else "held flat"


def _legend(names: list[str], cols: list[str]) -> str:
    return '<div class="mt-lg">' + "".join(f'<span><i style="background:{c}"></i>{n}</span>' for n, c in zip(names, cols)) + "</div>"


def _stack_rows(rows: list[tuple[str, list[float]]], names: list[str], lw: str = "210px") -> str:
    """One bar per row; bar length = row total, segments = age band (or any split)."""
    mx = max([sum(v) for _, v in rows] + [1])
    out = ""
    for label, v in rows:
        t = sum(v)
        tip = f"{label}: " + ", ".join(f"{names[i]} {int(x)}" for i, x in enumerate(v))
        seg = "".join(f'<div style="width:{100 * x / t:.1f}%;background:{AGEC[i]}">{int(x) if 100 * x / t >= 9 else ""}</div>' for i, x in enumerate(v) if x and t)
        out += (f'<div class="mt-st" style="--lw:{lw}" title="{tip}"><span>{label}</span>'
                f'<div class="bar" style="width:{100 * t / mx:.1f}%">{seg}</div><b>{int(t)}</b></div>')
    return out


def _load() -> dict[str, pd.DataFrame]:
    names = ["yearly", "conditions", "age", "loss", "htn", "comorbidity", "teen", "teen_share", "wards", "returning", "diabetes"]
    return {n: _num(getattr(q, "q7_" + n)().copy()) for n in names}


class _Data:
    """The frames and lookups the sections share, computed once per render."""

    def __init__(self, D: dict[str, pd.DataFrame]):
        self.D = D
        self.y: dict[tuple, tuple] = {}
        for r in D["yearly"].itertuples():
            self.y[(r.condition_name, str(r.period))] = (_f(r.patients), _f(r.pct_of_patients), _f(r.all_patients))
        self.c: dict[tuple, float] = {}
        for r in D["conditions"].itertuples():
            self.c[(r.condition_name, str(r.period))] = _f(r.patients)
        self.loss: dict[str, dict[str, float]] = {}
        for r in D["loss"].itertuples():
            self.loss.setdefault(r.age_band, {})[r.loss_code] = _f(r.patients)
        self.htn: dict[str, dict[str, float]] = {}
        for r in D["htn"].itertuples():
            self.htn.setdefault(r.hypertensive_type, {})[r.age_band] = _f(r.patients)
        self.cm = {r.condition_name: [_f(r.patients)] + [_f(getattr(r, col, 0)) for _, col in COMORB] for r in D["comorbidity"].itertuples()}
        self.teen: dict[str, dict[str, float]] = {}
        for r in D["teen"].itertuples():
            self.teen.setdefault(str(int(r.visit_year)), {})[r.age_group] = _f(r.patients)
        self.t9 = {str(int(r.visit_year)): (_f(r.teen_maternal_patients), _f(r.teen_opd_patients), _f(r.pct_of_teen_patients)) for r in D["teen_share"].itertuples()}
        self.rec: dict[str, dict[str, tuple]] = {}
        for r in D["returning"].itertuples():
            self.rec.setdefault(r.condition_name, {})[r.age_band] = (_f(r.patients), _f(r.seen_both_years), _f(r.more_than_one_visit), _f(r.total_visits))
        self.dm = {r.diabetes_type: _f(r.patients) for r in D["diabetes"].itertuples()}

    def n(self, c: str, p: str) -> float:
        return self.y.get((c, p), (0, 0, 0))[0]

    def pct(self, c: str, p: str) -> float:
        return self.y.get((c, p), (0, 0, 0))[1]

    def cn(self, k: str, p: str) -> float:
        return self.c.get((k, p), 0.0)

    def chg(self, c: str) -> int:
        return _r(self.n(c, "2024"), self.n(c, "2025"))

    def rec_sum(self, c: str, i: int) -> float:
        return sum(row[i] for row in self.rec.get(c, {}).values())


# ============================================================ header, KPIs
def _kpis(S: _Data) -> None:
    t24, t25 = S.t9.get("2024", (0, 0, 0))[0], S.t9.get("2025", (0, 0, 0))[0]
    cards = [
        ("Maternal patients, 2025", _n(S.n(ANY, "2025")),
         f"{_arrow(S.chg(ANY), 'neutral')} vs 2024 ({_n(S.n(ANY, '2024'))} to {_n(S.n(ANY, '2025'))}), {S.pct(ANY, '2025'):.1f}% of all patients", BLUE),
        ("Pregnancy-loss patients, 2025", _n(S.n("Pregnancy loss", "2025")),
         f"{_arrow(S.chg('Pregnancy loss'))} vs 2024 ({_n(S.n('Pregnancy loss', '2024'))} to {_n(S.n('Pregnancy loss', '2025'))})", AMBER),
        ("Teen maternal patients, 2025", _n(t25), f"{_arrow(_r(t24, t25))} vs 2024 ({_n(t24)} to {_n(t25)}), ages 13-19", BLUE),
    ]
    html = "".join(f'<div class="afya-kpi" style="border-top-color:{col}"><div class="afya-kpi-label">{a}</div><div class="afya-kpi-value">{b}</div><div class="afya-kpi-sub">{c}</div></div>'
                   for a, b, c, col in cards)
    st.markdown(f'<div class="mt-kp3">{html}</div>', unsafe_allow_html=True)


def _top_insight(S: _Data) -> None:
    flat = abs(S.chg(ANY)) <= 2
    named = [c for c in CATS if c not in ("Other", "ANC (Antenatal care)") and S.n(c, "2024") >= 50]
    up = max(named, key=S.chg, default=None)
    dn = min(named, key=S.chg, default=None)
    lead = (f"Maternal patients held flat at about {_n(round(S.n(ANY, '2025'), -1))} a year ({S.pct(ANY, 'All years'):.1f}% of all patients)" if flat
            else f"Maternal patients {_moved(S.chg(ANY))} in 2025 ({_n(S.n(ANY, '2024'))} to {_n(S.n(ANY, '2025'))})")
    tail = ""
    if up and dn and up != dn:
        tail = f", but the mix moved: {up.lower()} {_moved(S.chg(up))} while {dn.lower()} {_moved(S.chg(dn))}"
    ui.insight_bar(f"<strong>{lead}{tail}.</strong>", "blue")


# ============================================================ A · conditions at a glance
def _item(S: _Data, k: str, col: str, mx: float) -> str:
    v, a, b = S.cn(k, "All years"), S.cn(k, "2024"), S.cn(k, "2025")
    r = _r(a, b)
    ac = ui.trend_color(r, "good" if k.startswith("ANC") else "bad")
    tip = f"{_cn(k)}: {_n(v)} patients; 2024 {_n(a)} to 2025 {_n(b)}"
    return (f'<div class="mt-item" title="{tip}"><div class="r1"><span>{_cn(k)}</span><span><b>{_n(v)}</b> '
            f'<b style="color:{ac};margin-left:6px;font-size:11.5px">{"&#9650; +" if r > 0 else "&#9660; "}{r}%</b></span></div>'
            f'<div class="tr"><i style="width:{max(1.5, 100 * v / mx):.1f}%;background:{col}"></i></div></div>')


def _dumbbell(S: _Data) -> str:
    cats = [ANY] + [c for c in CATS if c != "Other"]
    mx = max([S.pct(c, p) for c in cats for p in ("2024", "2025")] + [0.1]) * 1.1
    rows = ""
    for c in cats:
        a, b = S.pct(c, "2024"), S.pct(c, "2025")
        col = AMBER if b > a else TEAL
        top = c == ANY
        tip = f"{c}: {a:.2f}% in 2024 ({_n(S.n(c, '2024'))}) to {b:.2f}% in 2025 ({_n(S.n(c, '2025'))})"
        lo, hi = min(a, b), max(a, b)
        rows += (f'<div class="mt-db" title="{tip}"><span style="{"font-weight:700" if top else ""}">{"Any maternal" if top else _catn(c)}</span>'
                 f'<div class="tk"><div class="ln"></div><div class="sg" style="left:{100 * lo / mx:.1f}%;width:{100 * (hi - lo) / mx:.1f}%;background:{col}"></div>'
                 f'<i style="left:calc({100 * a / mx:.1f}% - 6px);background:#9CA3AF"></i><i style="left:calc({100 * b / mx:.1f}% - 6px);background:{col}"></i></div>'
                 f'<span style="font-size:11px;color:{MUTED};text-align:right">{a:.2f}% &rarr; {b:.2f}%</span></div>')
    return (f'<div class="mt-card wide"><div class="ct">Share of all patients with each category, 2024 vs 2025</div>'
            f'<div class="cs">% of that year’s patients. A patient can be in more than one category. Amber = grew, green = fell.</div>{rows}</div>')


def _section_a(S: _Data) -> None:
    ui.section_header("Section A — Maternal conditions at a glance")
    st.markdown('<div class="afya-note" style="margin:-4px 0 8px">One card per maternal category: patients by condition, all years. Arrow = change in patients, 2024 to 2025. Bars share one scale. Hover for counts.</div>',
                unsafe_allow_html=True)
    shown = {k for _, _, _, items in GROUPS for k in items}
    mx = max([S.cn(k, "All years") for k in shown] + [1])
    cards = ""
    for key, title, col, items in GROUPS:
        sub = f"{_n(S.n(key, 'All years'))} patients · {_arrow(S.chg(key), 'good' if key.startswith('ANC') else 'bad')} vs 2024" if key else ""
        cards += f'<div class="mt-card"><div class="ct" style="color:{col}">{title}</div><div class="cs">{sub}</div>{"".join(_item(S, k, col, mx) for k in items)}</div>'
    st.markdown(f'<div class="mt-grid">{cards}{_dumbbell(S)}</div>', unsafe_allow_html=True)
    rest = sorted({k for (k, p) in S.c if p == "All years"} - shown, key=lambda k: -S.cn(k, "All years"))
    if rest:
        _cap("Smaller conditions not shown: " + ", ".join(f"{_cn(k).lower()} {_n(S.cn(k, 'All years'))}" for k in rest) + ".")
    cats = [c for c in CATS if c != "Other" and S.n(c, "2024") >= 50]
    big = max(cats, key=lambda c: S.n(c, "All years"), default=None)
    grow = [c for c in cats if S.chg(c) > 0]
    fast = sorted([k for k in shown if S.cn(k, "2024") >= 5], key=lambda k: -_r(S.cn(k, "2024"), S.cn(k, "2025")))[:2]
    if big:
        rising = ("the only category still growing" if grow == [big] else
                  f"one of only {len(grow)} still growing, with {', '.join(g.lower() + ' (+' + str(S.chg(g)) + '%)' for g in grow if g != big)}" if len(grow) > 1 and big in grow else
                  f"not the one growing fastest")
        txt = f"{big} is the largest category and {rising}."
        if len(fast) == 2:
            txt += (f" {_cn(fast[0])} ({_n(S.cn(fast[0], '2024'))} to {_n(S.cn(fast[0], '2025'))}) and {_cn(fast[1]).lower()} "
                    f"({_n(S.cn(fast[1], '2024'))} to {_n(S.cn(fast[1], '2025'))}) rose fastest and look like recording changes; both go to the <a href='?tab=tab_dataquality' target='_self' style='color:#2E8B3A;font-weight:700;text-decoration:none'>Data quality tab</a>.")
        ui.insight_bar(f"<strong>{txt}</strong>", "amber")


# ============================================================ B · pregnancy loss
def _section_b(S: _Data) -> None:
    ui.section_header("Section B — Pregnancy loss")
    names = LOSS_BANDS
    rows = [(label, [S.loss.get(b, {}).get(code, 0) for b in names]) for code, label in LOSS_TYPES]
    with ui.chart_card("Pregnancy-loss patients by type and age band",
                       note="Bar length = patients with that code; colours = age band. By ICD-11 code, all years. A patient can carry more than one code. Hover for counts."):
        st.markdown(_legend(names, AGEC) + _stack_rows(rows, names), unsafe_allow_html=True)
    tot = {b: S.loss.get(b, {}).get("Any pregnancy loss", 0) for b in names}
    top2 = sorted(tot, key=lambda b: -tot[b])[:2]
    typ = {label: sum(v) for label, v in rows}
    ect, mol = typ.get("Ectopic", 0), typ.get("Molar", 0)
    lead = sorted([l for l in typ if l.startswith(("Spontaneous", "Threatened"))], key=lambda l: -typ[l])
    ui.insight_bar(f"<strong>Loss peaks at {top2[0]} ({_n(tot[top2[0]])} patients) and {top2[1]} ({_n(tot[top2[1]])}). "
                   f"Most is {' and '.join(l.split(' /')[0].lower() + (' abortion' if 'abortion' not in l else '') for l in lead).replace(' abortion abortion', ' abortion')}; "
                   f"ectopic is only {_n(ect)} patients and molar {_n(mol)}.</strong>", "amber")


# ============================================================ C · hypertensive disorders
def _section_c(S: _Data) -> None:
    ui.section_header("Section C — Hypertensive disorders of pregnancy")
    names = LOSS_BANDS
    rows = [(t, [S.htn.get(t, {}).get(b, 0) for b in names]) for t in HTN_TYPES]
    with ui.chart_card("Hypertensive-disorder patients by type and age band",
                       note="Bar length = patients with that type; colours = age band. Women, all years. A patient can have more than one type. Hover for counts."):
        st.markdown(_legend(names, AGEC) + _stack_rows(rows, names, "230px"), unsafe_allow_html=True)
    anyd = S.htn.get("Any hypertensive disorder", {})
    tot = sum(anyd.values())
    if tot:
        peak = max(names, key=lambda b: anyd.get(b, 0))
        pe = sum(S.htn.get("Pre-eclampsia", {}).values())
        under = anyd.get("Under 18", 0)
        ui.insight_bar(f"<strong>Hypertensive disorders peak at {peak} ({_n(anyd[peak])} of {_n(tot)} women with an age), the pre-eclampsia pattern. "
                       f"Pre-eclampsia is {_n(pe)} of {_n(tot)}{' and leads at every age' if all(S.htn.get('Pre-eclampsia', {}).get(b, 0) >= max(S.htn.get(t, {}).get(b, 0) for t in HTN_TYPES) for b in names if anyd.get(b, 0)) else ''}"
                       f"{'; none are under 18' if not under else ''}.</strong>", "amber")


# ============================================================ D · what else they have
def _section_d(S: _Data) -> None:
    ui.section_header("Section D — What else they have")
    rows = []
    for c in [ANY] + CATS:
        v = S.cm.get(c)
        if not v:
            continue
        N = v[0]
        cells = [(f"<b>Any maternal</b>" if c == ANY else _catn(c), "hl")]
        for i, (name, _) in enumerate(COMORB):
            cnt = v[i + 1]
            p = 100 * cnt / N if N else 0
            if (c, name) in DEFINING:
                cells.append((f"{p:.1f}%", "", HATCH))
                cells[-1] = (f'<span title="{c}: {int(cnt)} of {int(N)} with {name}; this category is defined by that diagnosis">{p:.1f}%</span>', "", HATCH)
            elif cnt:
                cells.append((f'<span title="{c}: {int(cnt)} of {int(N):,} patients with {name}">{p:.1f}%</span>', "", ("opacity:.4;" if N < 15 else "") + ui.heat(p, 0, 10)))
            else:
                cells.append((f'<span title="{c}: none of {int(N):,} with {name}">0</span>', "", "color:#9CA3AF"))
        cells.append((_n(N), "", f"color:{MUTED}"))
        rows.append(cells)
    with ui.chart_card("Comorbidity by maternal category", note="% of each category’s patients with the condition also recorded, any visit. Hatched = the category is defined by that diagnosis, "
                                                                 "so the figure is not a finding. Hover for counts."):
        ui.html_table([("", "")] + [(n, n) for n, _ in COMORB] + [("Patients", "")], rows, compact=True, widths=_eq(18, len(COMORB) + 1), min_width=620)
    with ui.chart_card("Diabetes among maternal patients: which type", note=f"{_n(sum(S.dm.values()))} patients with a diabetes diagnosis, by type recorded."):
        mx = max(list(S.dm.values()) + [1])
        order = [("Pre-existing type 2", "Pre-existing Type 2", TEAL), ("Type unspecified", "Diabetes, unspecified type", "#6B93C4"),
                 ("Pre-existing type 1", "Pre-existing Type 1", BLUE), ("Gestational", "Gestational diabetes", AMBER)]
        st.markdown("".join(f'<div class="mt-bars" title="{lab}: {_n(S.dm.get(key, 0))} patients"><span>{lab}</span><div class="b"><i style="width:{100 * S.dm.get(key, 0) / mx:.1f}%;background:{col}"></i></div>'
                            f'<span class="v">{_n(S.dm.get(key, 0))}</span></div>' for lab, key, col in order), unsafe_allow_html=True)
    st.markdown('<div class="afya-link"><span>Diabetes in the wider population: type 2 is the largest NCD burden.</span><a href="?tab=tab_ncds" target="_self">NCDs →</a></div>', unsafe_allow_html=True)
    st.markdown('<div class="afya-link"><span>Urinary tract infection across all patients, including pregnancy.</span><a href="?tab=tab_infectious" target="_self">Infectious disease →</a></div>', unsafe_allow_html=True)

    def pc(c: str, name: str) -> float:
        v = S.cm.get(c)
        i = [n for n, _ in COMORB].index(name) + 1
        return 100 * v[i] / v[0] if v and v[0] else 0.0

    uti_cats = ["Pregnancy loss", "High-risk pregnancy", "Labour & delivery"]
    an_cats = [c for c in ("Pregnancy loss", "Infections in pregnancy", "Maternal complications", "High-risk pregnancy") if c in S.cm]
    if all(c in S.cm for c in uti_cats) and an_cats and ANY in S.cm:
        u = [pc(c, "UTI") for c in uti_cats]
        a = [pc(c, "Anaemia") for c in an_cats]
        mixed = S.dm.get("Pre-existing Type 2", 0) + S.dm.get("Pre-existing Type 1", 0) >= S.dm.get("Gestational diabetes", 0) + S.dm.get("Diabetes, unspecified type", 0)
        hiv = int(S.cm[ANY][[n for n, _ in COMORB].index("HIV") + 1])
        ui.insight_bar(f"<strong>Take out each category’s own defining diagnosis and comorbidity is thin: UTI in {min(u):.0f}-{max(u):.0f}% of loss, high-risk and labour patients, "
                       f"anaemia in {min(a):.0f}-{max(a):.0f}%{', and diabetes is mostly pre-existing' if mixed else ''}.</strong> "
                       f"HIV is coded in {_n(hiv)} of {_n(S.cm[ANY][0])} maternal patients, a coding gap and not absence.", "amber")


# ============================================================ E · teenage pregnancy
def _section_e(S: _Data) -> None:
    ui.section_header("Section E — Teenage pregnancy")
    yrs = ["2023", "2024", "2025", "2026"]
    lab = lambda y: y + (" (partial)" if y == "2023" else " so far" if y == "2026" else "")  # noqa: E731
    l, r = st.columns(2)
    with l:
        with ui.chart_card("Teen maternal patients by year", note="Ages 13-19, by age group. Faded = partial year."):
            html = ""
            for y in yrs:
                v = [S.teen.get(y, {}).get(g, 0) for g in TEEN_GROUPS]
                t = sum(v)
                if not t:
                    continue
                segs = "".join(f'<div style="width:{100 * x / t:.1f}%;background:{["#9DB8D9", BLUE, TEAL][i]};color:#fff;font-size:10px;font-weight:600;display:flex;align-items:center;justify-content:center">'
                               f'{round(100 * x / t) if 100 * x / t >= 11 else ""}{"%" if 100 * x / t >= 11 else ""}</div>' for i, x in enumerate(v) if x)
                tip = f"{lab(y)}: " + ", ".join(f"{g} {int(v[i])}" for i, g in enumerate(TEEN_GROUPS))
                html += (f'<div class="mt-st" style="--lw:110px;{"opacity:.5;" if y in ("2023", "2026") else ""}" title="{tip}"><span>{lab(y)}</span>'
                         f'<div class="bar" style="width:100%;background:#F0F1F5">{segs}</div><span style="text-align:right;color:{MUTED}">n={int(t)}</span></div>')
            st.markdown(html + _legend(TEEN_GROUPS, ["#9DB8D9", BLUE, TEAL]), unsafe_allow_html=True)
    with r:
        with ui.chart_card("Teen maternal patients as % of all teen patients", note="Teen OPD patients shown as the denominator."):
            html = ""
            for y in yrs:
                if y not in S.t9:
                    continue
                m, o, p = S.t9[y]
                part = y in ("2023", "2026")
                html += (f'<div style="{"opacity:.5" if part else ""}"><div class="mt-bars" style="grid-template-columns:110px 1fr 56px" title="{lab(y)}: {int(m)} of {int(o):,} teen patients">'
                         f'<span>{lab(y)}</span><div class="b"><i style="width:{min(100, 100 * p / 2):.1f}%;background:{AMBER}"></i></div><span class="v">{p:.2f}%</span></div>'
                         f'<div style="font-size:10.5px;color:{MUTED};margin:-2px 0 4px 118px">{int(m)} of {int(o):,} teen patients</div></div>')
            st.markdown(html, unsafe_allow_html=True)
    t24, t25 = S.t9.get("2024"), S.t9.get("2025")
    if t24 and t25:
        und = lambda y: S.teen.get(y, {}).get("13-15", 0) + S.teen.get(y, {}).get("16-17", 0)  # noqa: E731
        old = lambda y: S.teen.get(y, {}).get("18-19", 0)  # noqa: E731
        mr, orr = _r(t24[0], t25[0]), _r(t24[1], t25[1])
        txt = (f"Teen maternal patients {_moved(mr)} in 2025 while teen OPD patients {_moved(orr)}, and all maternal patients {_moved(S.chg(ANY))}. "
               f"Under-18s went from {_n(und('2024'))} to {_n(und('2025'))}; 18-19s from {_n(old('2024'))} to {_n(old('2025'))}.")
        if mr < orr - 3 and abs(S.chg(ANY)) <= 3:
            txt = f"The 2025 fall is not just reporting: {txt}"
        ui.insight_bar(f"<strong>{txt}</strong>", "amber")


# ============================================================ F · where they come from
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
    return dict(mode="burden", all=ANY, noun="maternal", hospital=dict(lat=q.HOSPITAL_LAT, lon=q.HOSPITAL_LON), conds=[ANY] + MAP_CATS,
                labels=[ANY] + [_catn(c) for c in MAP_CATS], wards=list(wards.values()), unplaced=unplaced, h=560)


def _section_f(S: _Data, payload: dict | None) -> None:
    ui.section_header("Section F — Where they come from")
    with ui.chart_card("Patients by ward", note="Nandi residents, 2024-2025. Pick a category. Darker = more patients. Hover a ward for its count."):
        if payload:
            geo.render_geo("ncd", payload, height=705, scrolling=False)
        else:
            _cap("No ward data returned for this card.")
    df = S.D["wards"]
    if not df.empty and payload:
        a = df[df.condition_name == ANY].sort_values("patients", ascending=False)
        if not a.empty:
            tot, top = _f(a.patients.sum()), a.iloc[0]
            nxt = [r for r in a.iloc[1:].itertuples() if (r.ward, r.sub_county) in {(w["w"], w["s"]) for w in payload["wards"]}][:2]
            unplaced = int(payload["unplaced"].get(ANY, 0))
            more = f"; {nxt[0].ward} ({_n(nxt[0].patients)}) and {nxt[1].ward} ({_n(nxt[1].patients)}) come next" if len(nxt) == 2 else ""
            ui.insight_bar(f"<strong>{top.ward} ward alone holds {100 * _f(top.patients) / tot:.0f}% of maternal patients{more}, and {_n(unplaced)} patients cannot be placed on the map.</strong>", "amber")


# ============================================================ G · who comes back
def _section_g(S: _Data) -> None:
    ui.section_header("Section G — Who comes back")
    N, both, more = S.rec_sum(ANY, 0), S.rec_sum(ANY, 1), S.rec_sum(ANY, 2)
    anc, ancr = S.rec_sum("ANC (Antenatal care)", 0), S.rec_sum("ANC (Antenatal care)", 2)
    cards = [("Seen more than once", f"{100 * more / N:.1f}%" if N else "n/a", f"{_n(more)} of {_n(N)} maternal patients, 2024-2025", BLUE),
             ("Seen in both 2024 and 2025", _n(both), f"{100 * both / N:.1f}% of maternal patients" if N else "", AMBER),
             ("Antenatal patients who came back", f"{_n(ancr)} of {_n(anc)}", "no antenatal follow-up visible" if not ancr else "of all antenatal patients", AMBER)]
    html = "".join(f'<div class="afya-kpi" style="border-top-color:{col}"><div class="afya-kpi-label">{a}</div><div class="afya-kpi-value">{b}</div><div class="afya-kpi-sub">{c}</div></div>' for a, b, c, col in cards)
    st.markdown(f'<div class="mt-kp3">{html}</div>', unsafe_allow_html=True)
    cats = [c for c in CATS if c != "Other" and c in S.rec]
    cats.sort(key=lambda c: (-S.rec_sum(c, 2), -(S.rec_sum(c, 2) / max(S.rec_sum(c, 0), 1))))
    mx = max([S.rec_sum(c, 2) for c in cats] + [1])
    rows = ""
    for c in cats:
        v = [S.rec[c].get(b, (0, 0, 0, 0))[2] for b in RBANDS]
        t, Nc = sum(v), S.rec_sum(c, 0)
        tip = f"{c}: " + ", ".join(f"{b} {int(S.rec[c].get(b, (0, 0, 0, 0))[2])} of {int(S.rec[c].get(b, (0, 0, 0, 0))[0])}" for b in RBANDS) + " patients came back"
        seg = "".join(f'<div style="width:{100 * x / t:.1f}%;background:{AGEC[i]};color:#fff;font-size:10px;font-weight:600;display:flex;align-items:center;justify-content:center">{int(x) if 100 * x / t >= 12 else ""}</div>'
                      for i, x in enumerate(v) if x and t)
        rows += (f'<div class="mt-st" style="grid-template-columns:190px 1fr 150px;--lw:190px" title="{tip}"><span>{_catn(c)}</span>'
                 f'<div class="bar" style="width:{max(100 * t / mx, 3) if t else 0:.1f}%">{seg}</div>'
                 f'<span style="text-align:right"><b>{int(t)}</b> <span style="color:{MUTED}">of {_n(Nc)} · {100 * t / Nc if Nc else 0:.1f}%</span></span></div>')
    with ui.chart_card("Patients seen more than once, by category and age band",
                       note="Bar length = patients who came back; colours = age band. The label on the right shows them as a share of all the category’s patients. 2024-2025. Hover for counts."):
        st.markdown(_legend(RBANDS, AGEC) + rows, unsafe_allow_html=True)
    if N:
        ui.insight_bar(f"<strong>Maternal care here is almost all one-off: {100 * more / N:.1f}% of patients are seen more than once"
                       f"{', and none of the ' + _n(anc) + ' antenatal patients came back' if anc and not ancr else ''}.</strong>", "amber")


# ============================================================ page
def render() -> None:
    ui.inject_css()
    st.markdown(vi._CSS, unsafe_allow_html=True)
    st.markdown(_CSS, unsafe_allow_html=True)
    if q.DEMO:
        ui.demo_banner()
    _top = st.container()
    with ui.loading("Maternal"):
        S = _Data(_load())
        payload = _ward_payload(S)
    with _top:
        ui.page_header("Maternal & Obstetric", "Which maternal conditions bring women to Kapsabet CRH, who they are, where they come from, what else they have, and who comes back", icon="maternal", stat=(_n(S.n(ANY, "All years")), "maternal patients"))
    _kpis(S)
    _top_insight(S)
    _section_a(S)
    _section_b(S)
    _section_c(S)
    _section_d(S)
    _section_e(S)
    _section_f(S, payload)
    _section_g(S)
