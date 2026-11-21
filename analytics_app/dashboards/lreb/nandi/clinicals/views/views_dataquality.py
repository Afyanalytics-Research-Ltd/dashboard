"""
views_dataquality.py: Tab 11, Data Quality.

No SQL here: every number comes from queries.q11_* (dataquality_queries.sql: Q11.1 to Q11.8b and the malnutrition note).
Sections: A coverage · B is reporting getting thinner · C patient-record checks · D issue register.

THE ISSUE REGISTER IS PARTLY HAND-MAINTAINED. Lines whose size comes from a query on this tab (the whole-dataset items, malnutrition,
the ward match) are computed live. The lines for the other tabs (Workload, Injury, Maternal, Respiratory, Psychiatric, NCDs,
Infectious) quote the results those tabs already show, as plain text in REGISTER below. When one of those tabs' results change,
update its line here. Severity is a judgement: Fix at source = changes a headline number; Check = looks wrong, needs the facility; Note = worth knowing.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import nandi_queries as q
import nandi_ui_template as ui
import nandi_views_infectious as vi

BLUE, TEAL, AMBER, RED = ui.BLUE, ui.PRIMARY, ui.AMBER, ui.CRITICAL
MUTED = ui.TEXT_MUTED
LIGHT, GREY = "#8FB0D4", "#CBD5E1"
_n, _f = vi._n, vi._f

_CSS = f"""
<style>
.dq-kp4 {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:0 0 12px; }}
@media(max-width:1100px) {{ .dq-kp4 {{ grid-template-columns:repeat(2,1fr); }} }}
.dq-lg {{ display:flex; gap:14px; flex-wrap:wrap; font-size:11px; color:{MUTED}; margin-top:8px; }}
.dq-lg i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; }}
.dq-sb {{ display:flex; height:26px; border-radius:6px; overflow:hidden; gap:2px; margin:8px 0; }}
.dq-sb i {{ display:flex; align-items:center; justify-content:center; color:#fff; font-size:11px; font-weight:600; font-style:normal; min-width:2px; }}
.dq-cap {{ font-size:11px; color:{MUTED}; margin-top:8px; }}
.dq-hb {{ display:grid; grid-template-columns:100px 1fr 140px; gap:10px; align-items:center; padding:6px 0; font-size:12.5px; }}
.dq-hb .b {{ height:16px; background:#F0F1F5; border-radius:0 6px 6px 0; }}
.dq-hb .b i {{ display:block; height:100%; border-radius:0 4px 4px 0; }}
.dq-mc .v {{ font-size:26px; font-weight:700; }}
.dq-mc .d {{ font-size:12px; color:#374151; margin-bottom:6px; }}
.dq-chip {{ display:inline-block; font-size:10.5px; font-weight:700; padding:2px 8px; border-radius:99px; white-space:nowrap; }}
.dq-hi {{ background:#FBE4E4; color:#A32D2D; }} .dq-md {{ background:#FEF3E0; color:#B45309; }} .dq-lo {{ background:#EEF0F4; color:#6B7280; }}
.dq-rl {{ display:flex; gap:10px; align-items:center; font-size:12px; color:{MUTED}; flex-wrap:wrap; margin:0 0 10px; }}
details.dq-g {{ background:#fff; border:1px solid {ui.BORDER}; border-radius:8px; margin-bottom:10px; }}
details.dq-g > summary {{ cursor:pointer; list-style:none; display:flex; justify-content:space-between; align-items:center; padding:12px 16px; font-weight:600; font-size:13.5px; }}
details.dq-g > summary::-webkit-details-marker {{ display:none; }}
details.dq-g > summary .n {{ display:flex; gap:6px; align-items:center; font-weight:400; }}
.dq-it, .dq-ith {{ display:grid; grid-template-columns:96px 1.4fr 1.6fr 1.2fr; gap:12px; padding:10px 16px; border-top:1px solid {ui.BORDER}; font-size:12.5px; align-items:start; }}
.dq-ith {{ padding:6px 16px; font-size:10px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:{MUTED}; }}
.dq-it .h {{ font-weight:600; }} .dq-it .e, .dq-it .s {{ color:#374151; }}
@media(max-width:1000px) {{ .dq-it, .dq-ith {{ grid-template-columns:1fr; }} .dq-ith {{ display:none; }} }}
</style>
"""

HI, MD, LO = "hi", "md", "lo"
SEV = {HI: ("Fix at source", "dq-hi"), MD: ("Check", "dq-md"), LO: ("Note", "dq-lo")}

# Hand-maintained register lines for the other tabs. (severity, issue, what it affects, size). See the module docstring.
REGISTER_STATIC = {
    "Workload": [
        (HI, "A duplicate rollup row equals every other clinician combined", "Doubles every contact total if left in", "115,232 with it, 57,616 without. Excluded. Ask the facility what the record is"),
        (MD, "37 of 147 clinicians recorded no contacts in 8 months", "The roster is not a reliable picture of who works", "25% of the roster. Leave, new hires or misattributed contacts cannot be told apart"),
    ],
    "Injury and trauma": [
        (HI, "Cause of injury is almost never recorded", "Cannot say how people are getting hurt", "3 of 2,361 fracture patients"),
        (MD, "Assault-by-transport codes (PB5Z and similar) are kept out of road traffic", "Road traffic may be understated", "35 PB5Z patients in 2025 alone"),
        (MD, "Road traffic severity looks undercoded", "Severity of road injuries", "1 of 173 flagged severe in 2024, 0 of 150 in 2025"),
        (MD, "Wound sepsis share fell", "Whether wound care improved or sepsis is coded less", "2.5% in 2024, 1.1% in 2025"),
        (MD, "No assault-context code at all in 2023", "The sexual and gender-based violence trend", "0 of 3,446 patients in 2023, then in use from 2024"),
    ],
    "Maternal": [
        (MD, "Abdominal pregnancy and antenatal screening findings jumped", "Look like recording changes, not real change", "37 to 83 and 8 to 48 patients"),
        (MD, "Pregnancy codes on men and on girls and older women", "Sex and age on maternal records", "77 male patients; 24 female patients under 13 or over 50"),
        (LO, "Some ICD-11 codes carry two different texts", "Counting by diagnosis text", 'For example JA00.1 is "Threatened abortion" for 213 patients and "Induced abortion" for 1'),
    ],
    "Respiratory": [
        (MD, "Severe pneumonia fell sharply", "Whether care improved or severity is recorded less", "22% to 9%"),
        (MD, "HIV is almost never coded alongside pneumonia", "Comorbidity picture", "7 of 4,033 pneumonia patients aged 18 to 54"),
        (LO, "2024 aspergillosis jump and reused ICD-11 codes", "Small counts of a rare condition", "Not shown on the Respiratory tab"),
    ],
    "Psychiatric conditions": [
        (MD, "Counts that go against how the conditions normally look", "Schizophrenia under 13, substance use under 18, dementia under 55", "Listed to be checked, not concluded"),
        (MD, "Only 3% of patients are seen in both 2024 and 2025", "Either treatment is collected elsewhere or the register does not link visits", "57 of 1,887 patients"),
        (LO, "Perinatal code 6E20 appears in the mental health list", "How the code is being applied", "Check with the facility"),
    ],
    "NCDs": [
        (MD, "Patterns that break the expected shape", "Heart failure in young women, Type 1 diabetes in adults, sickle cell that disappears after youth", "Listed to be checked, not concluded"),
        (MD, "Only 2% of NCD patients are seen in both 2024 and 2025", "Lifelong conditions with almost no follow-up on record", "112 of 5,251 patients"),
    ],
}


# ============================================================ helpers
def _num(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = pd.to_numeric(df[c])
            except (ValueError, TypeError):
                pass
    return df


def _legend(items: list[tuple[str, str]]) -> str:
    return '<div class="dq-lg">' + "".join(f'<span><i style="background:{c}"></i>{n}</span>' for n, c in items) + "</div>"


def _kpi_html(cards: list[tuple]) -> str:
    return "".join(f'<div class="afya-kpi" style="border-top-color:{col}"><div class="afya-kpi-label">{a}</div><div class="afya-kpi-value">{b}</div><div class="afya-kpi-sub">{c}</div></div>'
                   for a, b, c, col in cards)


def _load() -> dict[str, pd.DataFrame]:
    names = ["diag", "geo", "year", "tracking", "dupes", "gender", "malnutrition", "complete", "ward_match", "ward_list"]
    return {n: _num(getattr(q, "q11_" + n)().copy()) for n in names}


class _Data:
    def __init__(self, D: dict[str, pd.DataFrame]):
        d = D["diag"].iloc[0] if not D["diag"].empty else None
        g = (lambda c: _f(d[c])) if d is not None else (lambda c: 0.0)
        self.total, self.nodx, self.trunc, self.unres, self.aaa = g("total_visits"), g("no_diagnosis"), g("truncated_dx"), g("unresolved_dx"), g("aaa_default_pair")
        self.pct_nodx = g("pct_no_diagnosis")
        geo = D["geo"].groupby("geo_confidence")["visits"].sum().to_dict() if not D["geo"].empty else {}
        self.geo = {k: _f(v) for k, v in geo.items()}
        self.geo_total = sum(self.geo.values())
        self.years = [(str(int(r.visit_year)), _f(r.visits), str(r.completeness_flag)) for r in D["year"].sort_values("visit_year").itertuples()]
        self.comp = {str(int(r.visit_year)): {"no_dx": _f(r.pct_no_diagnosis), "multi": _f(r.pct_multi_diagnosis_of_coded), "trunc": _f(r.pct_truncated)} for r in D["complete"].itertuples()}
        t = D["tracking"]
        self.named = next((_f(r.visits) for r in t.itertuples() if "NAMELESS" not in str(r.tracking_classification).upper()), 0.0)
        self.nameless = next((_f(r.visits) for r in t.itertuples() if "NAMELESS" in str(r.tracking_classification).upper()), 0.0)
        dp = D["dupes"]
        self.dup_v = next((_f(r.visits) for r in dp.itertuples() if "DUPLICATE" in str(r.possible_duplicate_flag).upper() and "DISTINCT" not in str(r.possible_duplicate_flag).upper()), 0.0)
        self.dup_p = next((_f(r.unique_patients) for r in dp.itertuples() if "DUPLICATE" in str(r.possible_duplicate_flag).upper() and "DISTINCT" not in str(r.possible_duplicate_flag).upper()), 0.0)
        self.dist_v = next((_f(r.visits) for r in dp.itertuples() if "DISTINCT" in str(r.possible_duplicate_flag).upper()), 0.0)
        gd = D["gender"]
        self.gd_p = next((_f(r.patients) for r in gd.itertuples() if str(r.gender_disparity_flag) in ("1", "1.0", "True")), 0.0)
        self.gd_v = next((_f(r.visits) for r in gd.itertuples() if str(r.gender_disparity_flag) in ("1", "1.0", "True")), 0.0)
        self.gd_all = _f(gd["patients"].sum()) if not gd.empty else 0.0
        mal = D["malnutrition"]
        self.mal_total = _f(mal["visits_with_malnutrition_code"].sum()) if not mal.empty else 0.0
        self.mal_u5 = next((_f(r.visits_with_malnutrition_code) for r in mal.itertuples() if "UNDER" in str(r.age_band).upper()), 0.0)
        wm = D["ward_match"]
        nm = next((r for r in wm.itertuples() if str(r.match_status).startswith("No")), None)
        self.nm = (_f(nm.visits), _f(nm.patients), _f(nm.pct_of_visits)) if nm is not None else (0.0, 0.0, 0.0)
        wl = D["ward_list"]
        self.wl_top = (str(wl.iloc[0]["ward"]), _f(wl.iloc[0]["patients"])) if not wl.empty else None
        subs = {str(s).strip().lower() for s in wl["sub_county"]} if not wl.empty else set()
        self.tind = "tindiret" in subs and "tinderet" in subs


# ============================================================ KPIs and headline
def _kpis(S: _Data) -> None:
    coded = S.total - S.nodx
    matched = S.geo.get("High", 0) + S.geo.get("Medium", 0)
    cards = [
        ("Visits in the register", _n(S.total), "2023 to 2026 (2023 and 2026 are partial years)", BLUE),
        ("Visits with a diagnosis", f"{100 * coded / max(S.total, 1):.1f}%", f"{_n(coded)} coded · {_n(S.nodx)} with none", AMBER),
        ("Residence matched", f"{100 * matched / max(S.geo_total, 1):.1f}%", "High or medium confidence", BLUE),
        ("Diagnoses unresolved", f"{100 * S.unres / max(S.total, 1):.1f}%", f"{_n(S.unres)} visits after the ICD-11 lookup", BLUE),
    ]
    st.markdown(f'<div class="dq-kp4">{_kpi_html(cards)}</div>', unsafe_allow_html=True)


def _top_insight(S: _Data) -> None:
    yrs = sorted(S.comp)
    last, prev, base = (yrs[-1], yrs[-2], yrs[1]) if len(yrs) >= 3 else (None, None, None)
    tail = ""
    if last:
        tail = (f" The no-diagnosis share is getting worse ({S.comp[base]['no_dx']:.0f}% in {base}, {S.comp[prev]['no_dx']:.0f}% in {prev}, {S.comp[last]['no_dx']:.0f}% so far in {last}), "
                "so a fall in any disease count may be thinner recording, not fewer patients.")
    ui.insight_bar(f"<strong>Two things limit every page: {S.pct_nodx:.0f}% of visits carry no diagnosis, and 2025 has an export gap.</strong>{tail}", "amber")


# ============================================================ A · coverage
def _diag_bar(S: _Data) -> str:
    coded = S.total - S.nodx
    segs = [("Coded, complete", coded - S.trunc, BLUE), ("Coded, truncated", S.trunc, LIGHT), ("No diagnosis", S.nodx, AMBER)]
    bar = '<div class="dq-sb">' + "".join(f'<i style="width:{100 * v / max(S.total, 1):.2f}%;background:{c}" title="{n}: {_n(v)}">{100 * v / max(S.total, 1):.0f}%</i>' for n, v, c in segs) + "</div>"
    return (bar + _legend([(f"{n} · {_n(v)}", c) for n, v, c in segs])
            + f'<div class="dq-cap">Also: {_n(S.aaa)} visits carry the default code pair (AAA) and are left out of every disease count; {_n(S.unres)} visits have a diagnosis that could not be resolved.</div>')


def _geo_bars(S: _Data) -> str:
    order = [("High", BLUE), ("Medium", LIGHT), ("UNMATCHED", AMBER), ("Low", GREY), ("NOT_A_PLACE", GREY)]
    labs = {"UNMATCHED": "Unmatched", "NOT_A_PLACE": "Not a place"}
    mx = max([S.geo.get(k, 0) for k, _ in order] + [1])
    out = ""
    for k, c in order:
        v = S.geo.get(k, 0)
        out += (f'<div class="dq-hb"><span>{labs.get(k, k)}</span><div class="b"><i style="width:{max(100 * v / mx, 1.5):.1f}%;background:{c}"></i></div>'
                f'<span><b>{_n(v)}</b> · {100 * v / max(S.geo_total, 1):.1f}%</span></div>')
    return out


def _year_cols(S: _Data) -> str:
    if not S.years:
        return ""
    W, H, pb, pt = 420, 190, 34, 26
    mx = max(v for _, v, _ in S.years) * 1.04
    bw = 60
    gap = (W - 20 - bw * len(S.years)) / max(len(S.years) - 1, 1)
    s = (f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Visits per year"><defs><pattern id="dqh" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
         f'<rect width="6" height="6" fill="{AMBER}"/><line x1="0" y1="0" x2="0" y2="6" stroke="#fff" stroke-width="2.5"/></pattern></defs>')
    for i, (y, v, flag) in enumerate(S.years):
        part, gapyr = flag.upper().startswith("PARTIAL"), "EXPORT GAP" in flag.upper()
        fill = "url(#dqh)" if gapyr else (GREY if part else BLUE)
        x, h = 10 + i * (bw + gap), (H - pb - pt) * v / mx
        yy = H - pb - h
        s += (f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bw}" height="{h:.1f}" rx="3" fill="{fill}"><title>{y}: {_n(v)} visits. {flag}</title></rect>'
              f'<text x="{x + bw / 2:.1f}" y="{yy - 6:.1f}" text-anchor="middle" font-size="11.5" font-weight="600">{_n(v)}</text>'
              f'<text x="{x + bw / 2:.1f}" y="{H - pb + 16}" text-anchor="middle" font-size="12">{y}{" (part)" if part else ""}</text>')
    return s + "</svg>" + _legend([("Partial year", GREY), ("Full year", BLUE), ("Full year, export gap", AMBER)])


def _section_a(S: _Data) -> None:
    ui.section_header("Section A — Coverage")
    with ui.chart_card("Visits by diagnosis status", note=f"Share of all {_n(S.total)} visits."):
        st.markdown(_diag_bar(S), unsafe_allow_html=True)
    l, r = st.columns(2)
    with l:
        with ui.chart_card("Residence match", note="Visits by how confidently the residence was placed. Bar = visits."):
            st.markdown(_geo_bars(S), unsafe_allow_html=True)
    with r:
        with ui.chart_card("Visits per year", note="Partial years are grey. 2025 is a complete year with an export gap."):
            st.markdown(_year_cols(S), unsafe_allow_html=True)


# ============================================================ B · reporting over time
def _lines(S: _Data) -> str:
    yrs = sorted(S.comp)
    if len(yrs) < 2:
        return ""
    W, H, pl, pr, pt, pb = 900, 280, 70, 230, 24, 36
    series = [("No diagnosis", "no_dx", AMBER), ("More than one diagnosis", "multi", BLUE), ("Truncated diagnosis", "trunc", TEAL)]
    hi = max(40.0, max(S.comp[y][k] for y in yrs for _, k, _ in series) * 1.1)
    X = lambda i: pl + i * (W - pl - pr) / (len(yrs) - 1)
    Yf = lambda v: H - pb - (H - pb - pt) * v / hi
    s = f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Reporting completeness by year">'
    for g in range(0, int(hi) + 1, 10):
        s += f'<line x1="{pl}" x2="{W - pr}" y1="{Yf(g):.1f}" y2="{Yf(g):.1f}" stroke="#E5E7EB"/><text x="{pl - 8}" y="{Yf(g) + 4:.1f}" text-anchor="end" font-size="11" fill="#6B7280">{g}%</text>'
    partial = {y for y, _, f in S.years if f.upper().startswith("PARTIAL")}
    for i, y in enumerate(yrs):
        s += f'<text x="{X(i):.1f}" y="{H - pb + 20}" text-anchor="middle" font-size="12">{y}{" (part)" if y in partial else ""}</text>'
    for n, k, c in series:
        pts = [(X(i), Yf(S.comp[y][k])) for i, y in enumerate(yrs)]
        last_full = max((i for i, y in enumerate(yrs) if y not in partial), default=len(yrs) - 1)
        s += f'<polyline fill="none" stroke="{c}" stroke-width="2.4" points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts[:last_full + 1]) + '"/>'
        if last_full < len(yrs) - 1:
            s += f'<polyline fill="none" stroke="{c}" stroke-width="2.4" stroke-dasharray="5 4" points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts[last_full:]) + '"/>'
        for i, (a, b) in enumerate(pts):
            below = k == "trunc" or (k == "no_dx" and i == 2)
            s += (f'<circle cx="{a:.1f}" cy="{b:.1f}" r="4" fill="{c}"><title>{n}, {yrs[i]}: {S.comp[yrs[i]][k]:.1f}%</title></circle>'
                  f'<text x="{a:.1f}" y="{b + (18 if below else -9):.1f}" text-anchor="middle" font-size="11" font-weight="600" fill="{c}">{S.comp[yrs[i]][k]:.1f}%</text>')
        s += f'<text x="{pts[-1][0] + 10:.1f}" y="{pts[-1][1] + 4:.1f}" font-size="11.5" fill="{c}" font-weight="600">{n}</text>'
    return s + "</svg>"


def _section_b(S: _Data) -> None:
    ui.section_header("Section B — Is reporting getting thinner?")
    with ui.chart_card("Reporting completeness by year", note="Share of visits. Partial years are dashed. 'More than one diagnosis' is out of visits that have at least one diagnosis."):
        st.markdown(_lines(S), unsafe_allow_html=True)
    yrs = sorted(S.comp)
    if len(yrs) >= 3:
        b, l = yrs[1], yrs[-1]
        c = S.comp
        ui.insight_bar(
            f"<strong>Coverage is thinning while coding is getting cleaner.</strong> Visits with no diagnosis rose from {c[b]['no_dx']:.0f}% to {c[l]['no_dx']:.0f}% and visits with a second diagnosis fell from "
            f"{c[b]['multi']:.0f}% to {c[l]['multi']:.0f}% since {b}, while truncated diagnoses fell from {c[b]['trunc']:.0f}% to {c[l]['trunc']:.0f}%. "
            "This is the evidence behind the recording-change notes on the disease pages.", "blue")


# ============================================================ C · patient-record checks
def _mini_card(title: str, note: str, big: str, desc: str, segs: list[tuple[str, float, str]]) -> str:
    bar = '<div class="dq-sb">' + "".join(f'<i style="width:{max(w, 0.4):.2f}%;background:{c}">{t}</i>' for t, w, c in segs) + "</div>"
    return f'<div class="dq-mc"><div class="v">{big}</div><div class="d">{desc}</div>{bar}</div>'


def _section_c(S: _Data) -> None:
    ui.section_header("Section C — Patient-record checks")
    c1, c2, c3 = st.columns(3)
    tv = max(S.named + S.nameless, 1)
    with c1:
        with ui.chart_card("Patients with no name recorded", note="Tracking classification"):
            st.markdown(_mini_card("", "", f"{_n(S.nameless)} visits", f"Out of {_n(tv)}. Every other visit has a named patient, so this is not a problem.",
                                   [(f"Named patient {100 * S.named / tv:.0f}%", 100 * S.named / tv, BLUE), ("", 100 * S.nameless / tv, AMBER)]), unsafe_allow_html=True)
    dv = max(S.dup_v + S.dist_v, 1)
    with c2:
        with ui.chart_card("Possible duplicate patient records", note="Flagged for review"):
            st.markdown(_mini_card("", "", f"{_n(S.dup_v)} visits", f"{100 * S.dup_v / dv:.0f}% of visits, about {_n(round(S.dup_p, -2))} patients. Some patients also have visits that are not flagged, so the two counts overlap.",
                                   [(f"Distinct {100 * S.dist_v / dv:.0f}%", 100 * S.dist_v / dv, BLUE), ("", 100 * S.dup_v / dv, AMBER)]), unsafe_allow_html=True)
    with c3:
        with ui.chart_card("Gender disparity flag", note="Set by the cleaning pipeline"):
            tot = max(S.gd_all, 1)
            st.markdown(_mini_card("", "", f"{_n(S.gd_p)} patients", f"{100 * S.gd_p / tot:.1f}% of patients and {_n(S.gd_v)} visits are flagged.",
                                   [(f"Not flagged {100 * (1 - S.gd_p / tot):.1f}%", 100 * (1 - S.gd_p / tot), BLUE), ("", 100 * S.gd_p / tot, AMBER)]), unsafe_allow_html=True)
    ui.insight_bar(f"<strong>One visit in {max(round(dv / max(S.dup_v, 1)), 1)} is flagged as a possible duplicate patient.</strong> Until those are reviewed, patient counts can be slightly high and return-visit rates slightly off.", "amber")


# ============================================================ D · issue register
def _register(S: _Data) -> list[tuple[str, list[tuple]]]:
    yrs = {y: v for y, v, _ in S.years}
    partial = [(y, v) for y, v, f in S.years if f.upper().startswith("PARTIAL")]
    ratio = 100 * yrs.get("2025", 0) / yrs["2024"] if yrs.get("2024") else 0
    last = sorted(S.comp)[-1] if S.comp else None
    whole = [
        (HI, "2025 export gap outside Emgwen", "Every 2024 to 2025 comparison", f"2025 has {ratio:.0f}% of 2024's visits"),
        (HI, "No diagnosis on a large and growing share of visits", "Every disease count, which only sees coded visits",
         f"{_n(S.nodx)} visits ({S.pct_nodx:.0f}%)" + (f"; {S.comp[last]['no_dx']:.0f}% so far in {last}" if last else "")),
    ]
    if partial:
        whole.append((LO, f"{' and '.join(y for y, _ in partial)} are partial years", "Trends that include them", " and ".join(_n(v) for _, v in partial) + " visits"))
    infectious = [
        (MD, "Aortic aneurysm and essential hypertension on mild sepsis visits with no toddlers", "A coding error", "Seen on the Infectious tab"),
        (MD, "Gonococcal infection fell by almost half", "The export gap alone may not explain it", "-48% from 2024 to 2025"),
        (MD, "Malnutrition is almost never coded", "Under-5 nutrition cannot be read from this register", f"{_n(S.mal_total)} visits in total, {_n(S.mal_u5)} under 5"),
    ]
    geo = [(MD, "Some wards do not match a boundary", "Those patients cannot be placed on any map",
            f"{_n(S.nm[0])} visits, {_n(S.nm[1])} patients ({S.nm[2]:.1f}%)" + (f". {S.wl_top[0]} is the largest, {_n(S.wl_top[1])} patients" if S.wl_top else ""))]
    if S.tind:
        geo.append((MD, "Tindiret and Tinderet are spelled both ways as a sub-county", "Splits one area in two and breaks boundary matching", "Both spellings appear in the ward list"))
    geo.append((LO, "Residence could not be matched on some visits", "Maps and distance bands",
                f"{_n(S.geo.get('UNMATCHED', 0))} visits unmatched ({100 * S.geo.get('UNMATCHED', 0) / max(S.geo_total, 1):.1f}%), {_n(S.geo.get('NOT_A_PLACE', 0))} not a place"))
    out = [("The whole dataset", whole)]
    out += [(k, REGISTER_STATIC[k]) for k in ("Workload", "Injury and trauma", "Maternal", "Respiratory", "Psychiatric conditions", "NCDs")]
    out += [("Infectious disease", infectious), ("Geography", geo)]
    return out


def _group_html(title: str, items: list[tuple], open_: bool) -> str:
    cnt = {k: sum(1 for i in items if i[0] == k) for k in (HI, MD, LO)}
    chips = "".join(f'<span class="dq-chip {SEV[k][1]}">{cnt[k]} {SEV[k][0].lower()}</span>' for k in (HI, MD, LO) if cnt[k])
    rows = '<div class="dq-ith"><span>Severity</span><span>Issue</span><span>What it affects</span><span>Size</span></div>'
    for sev, a, b, c in items:
        rows += f'<div class="dq-it"><span><span class="dq-chip {SEV[sev][1]}">{SEV[sev][0]}</span></span><span class="h">{a}</span><span class="e">{b}</span><span class="s">{c}</span></div>'
    return f'<details class="dq-g"{" open" if open_ else ""}><summary><span>{title}</span><span class="n">{chips}<span style="color:{MUTED};font-size:12px;margin-left:6px">Open ▾</span></span></summary>{rows}</details>'


def _section_d(S: _Data) -> None:
    ui.section_header("Section D — Issue register")
    reg = _register(S)
    allitems = [i for _, items in reg for i in items]
    n = {k: sum(1 for i in allitems if i[0] == k) for k in (HI, MD, LO)}
    st.markdown(
        f'<div class="dq-rl"><span><b>{len(allitems)} items</b> carried from the other pages:</span>'
        + "".join(f'<span class="dq-chip {SEV[k][1]}">{n[k]} {SEV[k][0].lower()}</span>' for k in (HI, MD, LO))
        + "<span>Fix at source = changes a headline number. Check = looks wrong, needs the facility. Note = worth knowing.</span></div>"
        + "".join(_group_html(t, items, i == 0) for i, (t, items) in enumerate(reg)), unsafe_allow_html=True)


def render() -> None:
    ui.inject_css()
    st.markdown(vi._CSS, unsafe_allow_html=True)
    st.markdown(_CSS, unsafe_allow_html=True)
    if q.DEMO:
        ui.demo_banner()
    _top = st.container()
    with ui.loading("Data quality"):
        S = _Data(_load())
    with _top:
        ui.page_header("Data Quality", "How far the numbers on the other pages can be trusted, and the issues each page is carrying.", icon="dataquality", stat=(_n(S.total), "visits in the register"))
    _kpis(S)
    _top_insight(S)
    _section_a(S)
    _section_b(S)
    _section_c(S)
    _section_d(S)
