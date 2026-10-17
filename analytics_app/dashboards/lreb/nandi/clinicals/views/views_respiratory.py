"""
views_respiratory.py: Tab 6, Respiratory.

No SQL here: every number comes from queries.q6_* (respiratory_queries.sql, Q6.1 to Q6.5 and Q6.7). This file shapes DataFrames
into KPI cards, a bar chart, heatmap tables, age-by-sex panels, the ward map (template/geo_engine.js, kind "ncd" in burden mode)
and ui_template cards.
Sections: A burden at a glance · B who has each burden · C pneumonia: how severe, and what else they have ·
D where they come from · E who comes back.

Definition: ICD-11 chapter 12 (CA, CB) diagnoses, counted by patient, default code pair excluded (see queries.py, Tab 6 block).
Data-quality items (the fall in severe pneumonia, near-zero HIV coding, the 2024 aspergillosis jump, reused ICD-11 codes) belong on
the Data Quality tab; aspergillosis (Q6.6) is not run here.
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
AGES = vi.AGES
AS = ["0-4", "5-12", "13-17", "18-24", "25-34", "35-44", "45-54", "55+"]
ANY = "Any respiratory"
CONDS = ["Pneumonia", "Upper respiratory infection", "Acute tonsillitis", "Asthma", "Allergic rhinitis", "Pleural disease", "COPD", "Other respiratory"]
PANELS = ["Pneumonia", "Upper respiratory infection", "Acute tonsillitis", "Asthma", "Allergic rhinitis", "Pleural disease", "COPD"]
SHORT = {"Upper respiratory infection": "Upper respiratory", "Acute tonsillitis": "Tonsillitis"}
RBANDS = ["Under 5", "5-17", "18-54", "55+"]
COMORB = [("HIV", "with_hiv"), ("TB", "with_tb"), ("Malnutrition", "with_malnutrition"), ("Anaemia", "with_anaemia"), ("Malaria", "with_malaria"),
          ("Hypertension", "with_hypertension"), ("Diabetes", "with_diabetes"), ("Heart failure", "with_heart_failure")]
_n, _f, _cap, _eq = vi._n, vi._f, vi._cap, vi._eq

_CSS = f"""
<style>
.afya-kpis:has(.afya-kpi:nth-child(4):last-child) {{ grid-template-columns:repeat(4,1fr); }}
.rs-kp3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:0 0 12px; }}
.rs-bars {{ display:grid; grid-template-columns:130px 1fr 54px 64px; gap:8px; align-items:center; padding:6px 0; font-size:12px; }}
.rs-bars .b {{ height:16px; background:#F0F1F5; border-radius:0 6px 6px 0; }}
.rs-bars .b i {{ display:block; height:100%; border-radius:0 6px 6px 0; }}
.rs-bars .v {{ text-align:right; font-variant-numeric:tabular-nums; font-weight:600; }}
.rs-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }}
@media(max-width:1100px) {{ .rs-grid {{ grid-template-columns:repeat(2,1fr); }} .rs-kp3 {{ grid-template-columns:1fr; }} }}
.rs-panel {{ background:#fff; border:1px solid {ui.BORDER}; border-radius:8px; padding:12px 12px 8px; box-shadow:0 1px 3px rgba(0,0,0,.04); }}
.rs-panel .pt {{ display:flex; justify-content:space-between; align-items:baseline; font-size:13px; font-weight:600; color:{ui.TEXT}; }}
.rs-panel .ps {{ font-size:11px; color:{MUTED}; margin-bottom:6px; }}
.rs-row {{ display:grid; grid-template-columns:1fr 38px 1fr; align-items:center; gap:4px; height:19px; font-size:10.5px; }}
.rs-row span {{ text-align:center; color:{MUTED}; }}
.rs-row .l {{ display:flex; justify-content:flex-end; }}
.rs-row i {{ display:block; height:12px; }}
.rs-lg {{ display:flex; gap:14px; flex-wrap:wrap; font-size:11px; color:{MUTED}; margin-top:8px; }}
.rs-lg i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; }}
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


def _chg(S: "_Data", c: str) -> int:
    p = _pct_change(S.n(c, "2024"), S.n(c, "2025"))
    return 0 if p is None else round(p)


def _arrow_txt(r: int) -> str:
    return ui.arrow_html(r)


def _up(S: "_Data", c: str) -> str:
    return f"{_arrow_txt(_chg(S, c))} vs 2024 ({_n(S.n(c, '2024'))} to {_n(S.n(c, '2025'))})"


def _load() -> dict[str, pd.DataFrame]:
    names = ["yearly", "age_sex", "wards", "returning", "severity", "comorbidity"]
    return {n: _num(getattr(q, "q6_" + n)().copy()) for n in names}


class _Data:
    """The frames and lookups the sections share, computed once per render."""

    def __init__(self, D: dict[str, pd.DataFrame]):
        self.D = D
        self.y: dict[tuple, tuple] = {}
        y = D["yearly"]
        if not y.empty:
            for r in y.itertuples():
                self.y[(r.condition_name, str(r.period))] = (_f(r.patients), _f(r.pct_of_patients), _f(r.all_patients))
        self.age: dict[str, dict[str, list[float]]] = {}
        a = D["age_sex"]
        if not a.empty:
            for r in a.itertuples():
                if r.age_group_at_first_diagnosis not in AGES or r.sex not in ("F", "M"):
                    continue
                d = self.age.setdefault(r.condition_name, {"F": [0.0] * 8, "M": [0.0] * 8})
                d[r.sex][AGES.index(r.age_group_at_first_diagnosis)] += _f(r.patients)
        self.rec: dict[str, dict[str, tuple]] = {}
        if not D["returning"].empty:
            for r in D["returning"].itertuples():
                self.rec.setdefault(r.condition_name, {})[r.age_band] = (_f(r.patients), _f(r.seen_both_years), _f(r.more_than_one_visit), _f(r.total_visits))
        self.cm: dict[str, dict[str, list[float]]] = {}
        if not D["comorbidity"].empty:
            for r in D["comorbidity"].itertuples():
                self.cm.setdefault(r.condition_name, {})[r.age_band] = [_f(r.patients)] + [_f(getattr(r, col, 0)) for _, col in COMORB]
        self.sev: dict[tuple, tuple] = {}
        if not D["severity"].empty:
            for r in D["severity"].itertuples():
                self.sev[(int(r.visit_year), r.age_band)] = (_f(r.pneumonia_patients), _f(r.severe_patients), _f(r.with_hiv_code))

    def n(self, c: str, p: str) -> float:
        return self.y.get((c, p), (0, 0, 0))[0]

    def pct(self, c: str, p: str) -> float:
        return self.y.get((c, p), (0, 0, 0))[1]

    def tot(self, c: str) -> list[float]:
        d = self.age.get(c)
        return [f + m for f, m in zip(d["F"], d["M"])] if d else [0.0] * 8

    def cm_sum(self, c: str, bands: list[str] | None = None) -> list[float]:
        v = [0.0] * (len(COMORB) + 1)
        for b, row in self.cm.get(c, {}).items():
            if bands is None or b in bands:
                v = [x + y for x, y in zip(v, row)]
        return v

    def rec_sum(self, c: str, bands: list[str] | None = None) -> tuple[float, float, float, float]:
        v = [0.0] * 4
        for b, row in self.rec.get(c, {}).items():
            if bands is None or b in bands:
                v = [x + y for x, y in zip(v, row)]
        return tuple(v)  # type: ignore[return-value]


# ============================================================ header, KPIs
def _kpis(S: _Data) -> list[dict]:
    p24, p25 = S.pct(ANY, "2024"), S.pct(ANY, "2025")
    named = [c for c in CONDS if c != "Other respiratory"]
    grow = sorted([c for c in named if _chg(S, c) > 0], key=lambda c: -_chg(S, c))
    big = sorted([c for c in named if S.n(c, "2024") >= 500], key=lambda c: _chg(S, c))
    cards = [
        dict(label="Respiratory patients, 2025", value=_n(S.n(ANY, "2025")), sub=_up(S, ANY), accent="blue"),
        dict(label="Share of all patients, 2025", value=f"{p25:.1f}%",
             sub=f'{ui.arrow_html(round(p25 - p24, 1), pts=True, dec=1)} vs 2024 ({p24:.1f}% to {p25:.1f}%)', accent="blue"),
    ]
    if grow:
        c = grow[0]
        cards.append(dict(label="Only burden still growing" if len(grow) == 1 else "Fastest-growing burden", value=f"{_s(c)} {_n(S.n(c, '2025'))}", sub=_up(S, c), accent="amber"))
    else:
        cards.append(dict(label="Burdens still growing", value="None", sub="every burden fell or held between 2024 and 2025", accent="amber"))
    if big:
        c = big[0]
        cards.append(dict(label="Biggest fall among large burdens", value=f"{_s(c)} {_n(S.n(c, '2025'))}", sub=_up(S, c), accent="blue"))
    return cards


def _top_insight(S: _Data) -> None:
    named = [c for c in CONDS if c != "Other respiratory"]
    grow = [c for c in named if _chg(S, c) > 0]
    flat = abs(S.y.get((ANY, "2025"), (0, 0, 1))[2] / max(1, S.y.get((ANY, "2024"), (0, 0, 1))[2]) - 1) < 0.02
    if len(grow) == 1:
        fall = f"every burden but {_s(grow[0]).lower()} fell in 2025"
    elif not grow:
        fall = "every burden fell in 2025"
    else:
        fall = f"{len(named) - len(grow)} of {len(named)} burdens fell in 2025"
    share = S.pct(ANY, "All years")
    lead = "One in five Kapsabet patients has" if 17.5 <= share <= 22.5 else f"{share:.0f}% of Kapsabet patients have"
    ui.insight_bar(f"<strong>{lead} a respiratory diagnosis, and {fall} "
                   f"{'while total patients stayed flat' if flat else 'against a change in total patients'}.</strong>", "blue")


# ============================================================ A · burden at a glance
def _league(S: _Data) -> None:
    order = sorted([c for c in CONDS if c != "Other respiratory"], key=lambda c: -S.n(c, "All years")) + ["Other respiratory"]
    mx = max(S.n(ANY, "All years"), 1)
    rows = ""
    for c in [ANY] + order:
        v, r, top = S.n(c, "All years"), _chg(S, c), c == ANY
        col = ui.trend_color(r)
        tip = f"{c}: {_n(v)} patients, {S.pct(c, 'All years'):.1f}% of all patients; 2024 {_n(S.n(c, '2024'))} to 2025 {_n(S.n(c, '2025'))}"
        rows += (f'<div class="rs-bars" title="{tip}"><span style="{"font-weight:700" if top else ""}">{_s(c)}</span>'
                 f'<div class="b"><i style="width:{max(0.4, 100 * v / mx):.1f}%;background:{BLUE if top else "#6B93C4"}"></i></div>'
                 f'<span class="v">{_n(v)}</span><span class="v" style="color:{col}">{"&#9650; +" if r > 0 else "&#9660; "}{r}%</span></div>')
    st.markdown(rows, unsafe_allow_html=True)


def _year_heat(S: _Data) -> None:
    periods = ["2023", "2024", "2025", "2026"]
    heads = [("", "")] + [(p + (" (partial)" if p == "2023" else " so far" if p == "2026" else ""), p) for p in periods]
    mx = max([S.pct(c, p) for c in CONDS for p in periods] + [0.1])
    rows = []
    for c in [ANY] + CONDS:
        cells = [_label(c)]
        for p in periods:
            n, pc, tot = S.y.get((c, p), (0, 0, 0))
            if not n:
                cells.append(_dot())
                continue
            hi = 21 if c == ANY else mx * 1.05
            fade = "opacity:.45;" if p in ("2023", "2026") else ""
            cells.append(_cell(f"{pc:.2f}%", fade + ui.heat(pc, 0, hi), f"{c}, {p}: {int(n)} patients ({pc:.2f}% of {int(tot):,})"))
        rows.append(cells)
    ui.html_table(heads, rows, compact=True, widths=_eq(26, 4), min_width=420)


def _section_a(S: _Data) -> None:
    ui.section_header("Section A — Burden at a glance")
    l, r = st.columns(2)
    with l:
        with ui.chart_card("Patients by burden, all years", note="Arrow = change in patients, 2024 to 2025. Hover for share of all patients."):
            _league(S)
    with r:
        with ui.chart_card("Share of all patients with each burden, by year", note="% of that year’s patients. Faded = partial year (2023, 2026 so far)."):
            _year_heat(S)


# ============================================================ B · who has each burden
def _panel(S: _Data, c: str) -> str:
    d = S.age.get(c)
    if not d:
        return ""
    mx = max(d["F"] + d["M"] + [1])
    t = sum(S.tot(c))
    wf = 100 * sum(d["F"]) / t if t else 0
    rows = ""
    for i, a in enumerate(AS):
        tip = f"{c}, {AGES[i]}: {int(d['F'][i])} women, {int(d['M'][i])} men"
        rows += (f'<div class="rs-row" title="{tip}"><div class="l"><i style="width:{100 * d["F"][i] / mx:.1f}%;background:{TEAL};border-radius:3px 0 0 3px"></i></div>'
                 f'<span>{a}</span><div><i style="width:{100 * d["M"][i] / mx:.1f}%;background:{BLUE};border-radius:0 3px 3px 0"></i></div></div>')
    r = _chg(S, c)
    col = ui.trend_color(r)
    return (f'<div class="rs-panel"><div class="pt"><span>{c}</span><span style="font-size:11px;color:{col}">{"&#9650; +" if r > 0 else "&#9660; "}{r}%</span></div>'
            f'<div class="ps">{_n(t)} patients · {wf:.0f}% women</div>{rows}</div>')


def _section_b(S: _Data) -> None:
    ui.section_header("Section B — Who has each burden")
    st.markdown('<div class="afya-note" style="margin:-4px 0 8px">Patients by age group at first diagnosis. Left = women, right = men. Each panel is scaled to its own largest bar.</div>', unsafe_allow_html=True)
    panels = "".join(_panel(S, c) for c in PANELS)
    other = (f'<div class="rs-panel" style="background:#F9FAFB;display:flex;align-items:center;color:{MUTED};font-size:12px">'
             f'Other respiratory ({_n(S.n("Other respiratory", "All years"))} patients) is a mixed group, so it has no panel.</div>')
    st.markdown(f'<div class="rs-grid">{panels}{other}</div>'
                f'<div class="rs-lg"><span><i style="background:{TEAL}"></i>Women</span><span><i style="background:{BLUE}"></i>Men</span></div>', unsafe_allow_html=True)
    ui.insight_bar("<strong>Upper respiratory infection is a child’s diagnosis; asthma, pleural disease and COPD are adults’; "
                   "women lead asthma and tonsillitis from the teens up.</strong>", "amber")


# ============================================================ C · pneumonia: severity and comorbidity
def _severity(S: _Data) -> None:
    yrs = [2023, 2024, 2025, 2026]
    heads = [("", "")] + [(str(y) + (" (partial)" if y == 2023 else " so far" if y == 2026 else ""), str(y)) for y in yrs]
    rows = []
    for b in ["5 and over", "Under 5"]:
        sev = [(_label_b(b, "severe"))]
        hiv = [(f"{b}: HIV code", "", "font-weight:400")]
        for y in yrs:
            n, sv, hv = S.sev.get((y, b), (0, 0, 0))
            if not n:
                sev.append(_dot())
                hiv.append(_dot())
                continue
            p = 100 * sv / n
            fade = "opacity:.45;" if y in (2023, 2026) or sv < 15 else ""
            sev.append(_cell(f"{p:.1f}%", fade + ui.heat(p, 0, 25), f"{b}, {y}: {int(sv)} severe of {int(n)} pneumonia patients"))
            hiv.append(_cell(f"{int(hv)} of {int(n):,}", f"color:{MUTED}", f"{b}, {y}: {int(hv)} with an HIV code"))
        rows += [sev, hiv]
    ui.html_table(heads, rows, compact=True, widths=_eq(24, 4), min_width=420)


def _label_b(b: str, what: str) -> tuple:
    return (f"{b}: {what}", "hl")


def _comorb_row(label, v: list[float], bold: bool) -> list:
    n = v[0]
    cells = [(f"<b>{label}</b>" if bold and label == ANY else label, "hl" if bold else "", "" if bold else "font-weight:400")]
    for i, (name, _) in enumerate(COMORB):
        c = v[i + 1]
        if c:
            p = 100 * c / n
            cells.append(_cell(f"{p:.1f}%", ("opacity:.4;" if n < 15 else "") + ui.heat(p, 0, 12), f"{label}: {int(c)} of {int(n):,} patients with {name}"))
        else:
            cells.append(_cell("0", "color:#9CA3AF", f"{label}: none of {int(n):,} patients with {name}"))
    cells.append((_n(n), "", f"color:{MUTED}"))
    return cells


def _comorb_burden(S: _Data) -> None:
    rows = []
    for c in [ANY] + CONDS:
        if c not in S.cm:
            continue
        rows.append(_comorb_row(_s(c), S.cm_sum(c), c == ANY) if c != ANY else _comorb_row(ANY, S.cm_sum(c), True))
    ui.html_table([("", "")] + [(n, n) for n, _ in COMORB] + [("Patients", "")], rows, compact=True, widths=_eq(16, len(COMORB) + 1), min_width=620)


def _comorb_pneumonia(S: _Data) -> None:
    rows = [_comorb_row(b, S.cm["Pneumonia"][b], True) for b in RBANDS if b in S.cm.get("Pneumonia", {})]
    ui.html_table([("Age band", "")] + [(n, n) for n, _ in COMORB] + [("Patients", "")], rows, compact=True, widths=_eq(16, len(COMORB) + 1), min_width=620)


def _section_c(S: _Data) -> None:
    ui.section_header("Section C — Pneumonia: how severe, and what else they have")
    st.markdown(f'<div class="afya-link"><span><b>Pneumonia is the largest respiratory burden ({_n(S.n("Pneumonia", "All years"))} patients).</b> Its age peaks, severity by age group and top wards sit on the Infectious disease tab.</span>'
                '<a href="?tab=tab_infectious" target="_self">Open Infectious disease →</a></div>', unsafe_allow_html=True)
    with ui.chart_card("Pneumonia severity and HIV coding", note="% of pneumonia patients flagged severe, and patients with an HIV code, by year and age band. Faded = partial year."):
        _severity(S)
    with ui.chart_card("What else they have: comorbidity by burden", note="% of each burden’s patients with the condition also recorded. Hover for counts."):
        _comorb_burden(S)
    with ui.chart_card("Pneumonia comorbidity by age band", note="% of pneumonia patients in each age band with the condition also recorded. Hover for counts."):
        _comorb_pneumonia(S)
    copd, pn55 = S.cm_sum("COPD"), S.cm_sum("Pneumonia", ["55+"])
    ar, pn1854 = S.cm_sum("Allergic rhinitis"), S.cm_sum("Pneumonia", ["18-54"])
    hf = [i for i, (n, _) in enumerate(COMORB) if n == "Heart failure"][0] + 1
    ml = [i for i, (n, _) in enumerate(COMORB) if n == "Malaria"][0] + 1
    hv = [i for i, (n, _) in enumerate(COMORB) if n == "HIV"][0] + 1
    s23, s25 = S.sev.get((2023, "5 and over"), (0, 0, 0)), S.sev.get((2025, "5 and over"), (0, 0, 0))
    if copd[0] and pn55[0] and ar[0] and pn1854[0] and s23[0] and s25[0]:
        ui.insight_bar(f"<strong>Heart failure travels with COPD ({100 * copd[hf] / copd[0]:.0f}% of COPD patients) and with pneumonia over 55 ({100 * pn55[hf] / pn55[0]:.1f}%), "
                       f"and malaria shows in {100 * ar[ml] / ar[0]:.1f}% of allergic rhinitis patients.</strong> HIV is coded in only {int(pn1854[hv])} of {_n(pn1854[0])} pneumonia patients aged 18-54, "
                       f"and severe pneumonia fell from {100 * s23[1] / s23[0]:.0f}% to {100 * s25[1] / s25[0]:.0f}%; both look like recording changes, so they go to the <a href='?tab=tab_dataquality' target='_self' style='color:#2E8B3A;font-weight:700;text-decoration:none'>Data quality tab</a>.", "amber")


# ============================================================ D · where they come from
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
    return dict(mode="burden", all=ANY, noun="respiratory", hospital=dict(lat=q.HOSPITAL_LAT, lon=q.HOSPITAL_LON), conds=[ANY] + CONDS,
                labels=[ANY] + [_s(c) for c in CONDS], wards=list(wards.values()), unplaced=unplaced, h=560)


def _outside(S: _Data) -> float:
    a = S.D["age_sex"]
    if a.empty:
        return 0.0
    d = a[a.condition_name == ANY]
    return _f(d.from_outside_nandi.sum()) if not d.empty else 0.0


def _section_d(S: _Data, payload: dict | None) -> None:
    ui.section_header("Section D — Where they come from")
    with ui.chart_card("Patients by ward", note="Nandi residents, 2024-2025. Pick a burden. Darker = more patients. Hover a ward for its count."):
        if payload:
            payload["outside"] = f"Outside Nandi: {_n(_outside(S))} patients with a recorded county (all years)"
            geo.render_geo("ncd", payload, height=705, scrolling=False)
        else:
            _cap("No ward data returned for this card.")
    df = S.D["wards"]
    if not df.empty and payload:
        a = df[df.condition_name == ANY].sort_values("patients", ascending=False)
        if not a.empty:
            tot, top = _f(a.patients.sum()), a.iloc[0]
            placed = sum(1 for w in payload["wards"] if w["v"].get(ANY))
            unplaced = int(payload["unplaced"].get(ANY, 0))
            ui.insight_bar(f"<strong>{top.ward} ward alone holds {100 * _f(top.patients) / tot:.0f}% of respiratory patients; the other {max(placed - 1, 0)} wards are far paler, "
                           f"and {_n(unplaced)} patients cannot be placed on the map.</strong>", "amber")


# ============================================================ E · who comes back
def _section_e(S: _Data) -> None:
    ui.section_header("Section E — Who comes back")
    n, both, more, vis = S.rec_sum(ANY)
    un, _ub, um, _uv = S.rec_sum(ANY, ["Under 5"])
    on, _ob, om, _ov = S.rec_sum(ANY, ["5-17", "18-54", "55+"])
    cards = [("Seen in both 2024 and 2025", _n(both), f"{100 * both / n:.1f}% of {_n(n)} respiratory patients" if n else "", AMBER),
             ("Seen more than once: under 5 vs older", f"{100 * um / un:.1f}% vs {100 * om / on:.1f}%" if un and on else "n/a", "under-5s come back most", BLUE),
             ("Average visits per patient", f"{vis / n:.2f}" if n else "n/a", "across 2024-2025", BLUE)]
    html = "".join(f'<div class="afya-kpi" style="border-top-color:{col}"><div class="afya-kpi-label">{a}</div><div class="afya-kpi-value">{b}</div><div class="afya-kpi-sub">{c}</div></div>' for a, b, c, col in cards)
    st.markdown(f'<div class="rs-kp3">{html}</div>', unsafe_allow_html=True)
    with ui.chart_card("% of patients seen more than once, by burden and age band", note="2024-2025. Faded = fewer than 15 patients. Hover for counts."):
        rows = []
        for c in [ANY] + CONDS:
            if c not in S.rec:
                continue
            cells = [_label(c)]
            for b in RBANDS:
                p, s, m, _v = S.rec[c].get(b, (0, 0, 0, 0))
                cells.append(_dot() if not p else _cell(f"{100 * m / p:.1f}%", ("opacity:.4;" if p < 15 else "") + ui.heat(100 * m / p, 0, 7),
                                                       f"{c}, {b}: {int(m)} of {int(p)} patients seen more than once; {int(s)} seen in both years"))
            tp, _ts, tm, _tv = S.rec_sum(c)
            cells.append(_cell(f"{100 * tm / tp:.1f}%", ui.heat(100 * tm / tp, 0, 7), f"{c}: {int(tm)} of {int(tp)} patients") if tp else _dot())
            rows.append(cells)
        ui.html_table([("", "")] + [(b, b) for b in RBANDS] + [("All", "")], rows, compact=True, widths=_eq(26, 5), min_width=420)
    cand = []
    for c in CONDS:
        if c == "Other respiratory":
            continue
        for b, (p, _s2, m, _v) in S.rec.get(c, {}).items():
            if p >= 15:
                cand.append((100 * m / p, c, b))
    cand.sort(reverse=True)
    if n and len(cand) >= 2:
        def phrase(c, b):
            return f"{'under-5s' if b == 'Under 5' else 'patients aged ' + b} with {c.lower()}"
        ui.insight_bar(f"<strong>Respiratory illness is mostly a one-off: only {100 * more / n:.1f}% of patients are seen more than once. "
                       f"{phrase(cand[0][1], cand[0][2]).capitalize()} and {phrase(cand[1][1], cand[1][2])} come back most.</strong>", "amber")


# ============================================================ page
def render() -> None:
    ui.inject_css()
    st.markdown(vi._CSS, unsafe_allow_html=True)
    st.markdown(_CSS, unsafe_allow_html=True)
    if q.DEMO:
        ui.demo_banner()
    _top = st.container()
    with ui.loading("Respiratory"):
        S = _Data(_load())
        payload = _ward_payload(S)
    with _top:
        ui.page_header("Respiratory", "Which respiratory conditions bring people to Kapsabet CRH, who has them, where they come from, who comes back, and what looks off", icon="respiratory", stat=(_n(S.n(ANY, "All years")), "respiratory patients"))
    ui.kpi_row(_kpis(S))
    _top_insight(S)
    _section_a(S)
    _section_b(S)
    _section_c(S)
    _section_d(S, payload)
    _section_e(S)
