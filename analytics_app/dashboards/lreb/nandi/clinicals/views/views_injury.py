"""
views_injury.py: Tab 8, Injury & Trauma.

No SQL here: every number comes from queries.q8_* (injury_queries.sql: Q8.1, Q8.4 to Q8.15; Q8.2 and Q8.3 are internal and not run).
Sections: A injury burden · B severity and complications · C where they come from · D burns · E sexual and gender-based violence (SGBV).

Definitions: patients, not visits. Injury types are matched on diagnosis text one diagnosis slot at a time (see the Tab 8 block in queries.py).
SGBV figures are grouped and any cell under 5 is hidden in the query itself. Q8.2/Q8.3 (internal detail), Q8.0 and Q8.16 (code dictionaries),
the cause-of-injury gap (Q8.9 is shown only as a link line), the 2023 zero in the assault-context code and the assault-by-transport codes
belong on the Data Quality tab.
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
_n, _f, _cap = vi._n, vi._f, vi._cap

YEARS = ["2023", "2024", "2025", "2026"]
PARTIAL = {"2023", "2026"}
TYPES = [("fractures", "Fractures"), ("dislocation_sprain", "Dislocation / sprain"), ("open_wounds", "Open wounds"), ("burns", "Burns"), ("rta", "Road traffic")]
YC = {"2023": "#D5DEEA", "2024": "#9DB8D9", "2025": BLUE, "2026": "#D5DEEA"}
AGE_BANDS = ["Under 5", "5-12", "13-17", "18-34", "35-54", "55+"]
AGE_COL = ["#D9A55B", "#B45F06", "#6F94C8", BLUE, TEAL, "#9CA3AF"]
AGE_MAP = {"Under 5": 0, "5-12": 1, "13-17": 2, "18-24": 3, "25-34": 3, "35-44": 4, "45-54": 4, "55+": 5}
TYPE_KEYS = {"Fractures": "Fractures", "Dislocation / sprain": "Dislocation / sprain", "Open wounds": "Open wounds", "Burns": "Burns", "RTA": "Road traffic"}
MAP_CONDS = ["Any injury", "Fractures", "Dislocation / sprain", "Open wounds", "Burns", "RTA"]
MAP_LABELS = ["Any injury", "Fractures", "Dislocation / sprain", "Open wounds", "Burns", "Road traffic"]

_CSS = f"""
<style>
.in-kp3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:0 0 12px; }}
@media(max-width:1100px) {{ .in-kp3 {{ grid-template-columns:1fr; }} }}
.in-lg {{ display:flex; gap:14px; flex-wrap:wrap; font-size:11px; color:{MUTED}; margin-top:8px; }}
.in-lg i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; }}
.in-grp {{ display:flex; gap:18px; align-items:flex-end; }}
.in-grp > div {{ flex:1; min-width:0; }}
.in-cols {{ display:flex; gap:3px; align-items:flex-end; justify-content:center; }}
.in-col {{ flex:1; display:flex; flex-direction:column; justify-content:flex-end; align-items:center; height:100%; }}
.in-col span {{ font-size:10px; color:{MUTED}; }}
.in-col i {{ display:block; width:100%; border-radius:3px 3px 0 0; }}
.in-st {{ display:grid; grid-template-columns:var(--lw,150px) 1fr 56px; gap:8px; align-items:center; padding:4px 0; font-size:12.5px; }}
.in-st .bar {{ display:flex; height:22px; border-radius:4px; overflow:hidden; min-width:4px; }}
.in-st .bar div {{ color:#fff; font-size:10px; font-weight:600; display:flex; align-items:center; justify-content:center; }}
.in-st b {{ text-align:right; }}
.in-top {{ display:grid; grid-template-columns:minmax(110px,38%) 1fr 40px; gap:8px; align-items:center; padding:4px 0; font-size:11.5px; }}
.in-top .bar {{ display:flex; height:16px; border-radius:3px; overflow:hidden; }}
.in-top .v {{ text-align:right; font-weight:600; }}
.in-top .a {{ text-align:right; color:{MUTED}; }}
.in-vc {{ flex:1; text-align:center; }}
.in-vc .h {{ display:flex; flex-direction:column; justify-content:flex-end; align-items:center; }}
.in-vc .h b {{ font-size:12px; }}
.in-vc .h i {{ display:block; width:60%; border-radius:3px 3px 0 0; }}
.in-vc .y {{ font-size:11.5px; margin-top:4px; }}
.in-vc .s {{ font-size:10.5px; color:{MUTED}; }}
.in-row {{ display:flex; gap:10px; }}
.in-pyr {{ display:grid; grid-template-columns:var(--lw,130px) 1fr 1fr; gap:0 6px; align-items:center; padding:3px 0; font-size:12px; }}
.in-pyr .l {{ display:flex; justify-content:flex-end; align-items:center; gap:4px; height:16px; }}
.in-pyr .r {{ display:flex; align-items:center; gap:4px; height:16px; border-left:1px solid #9CA3AF; }}
.in-pyr i {{ display:block; height:14px; }}
.in-pyr span.n {{ font-size:11px; color:{MUTED}; }}
.in-bars {{ display:grid; grid-template-columns:150px 1fr 40px; gap:8px; align-items:center; padding:3px 0; font-size:12px; }}
.in-bars .b {{ height:14px; background:#F0F1F5; border-radius:0 6px 6px 0; }}
.in-bars .b i {{ display:block; height:100%; border-radius:0 6px 6px 0; }}
.in-bars .v {{ text-align:right; font-weight:600; }}
.in-origin {{ display:grid; grid-template-columns:repeat(3,1fr); gap:18px; }}
@media(max-width:900px) {{ .in-origin {{ grid-template-columns:1fr; }} }}
.in-note {{ border:1px solid {ui.BORDER}; border-left:3px solid {AMBER}; border-radius:0 6px 6px 0; background:#fff; padding:9px 14px; font-size:12px; margin:0 0 12px; }}
</style>
"""


# ============================================================ helpers
def _num(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = pd.to_numeric(df[c])
            except (ValueError, TypeError):
                pass
    return df


def _r(a: float, b: float) -> int:
    return 0 if not a else round((b - a) / a * 100)


def _arrow(r: int, up_is: str = "bad") -> str:
    return ui.arrow_html(r, up_is, flat="0%")


def _moved(r: int) -> str:
    return f"{'rose' if r > 0 else 'fell' if r < 0 else 'held'} {abs(r)}%" if r else "held flat"


def _yl(y: str) -> str:
    return f"{y} (part)" if y in PARTIAL else y


def _legend(names: list[str], cols: list[str]) -> str:
    return '<div class="in-lg">' + "".join(f'<span><i style="background:{c}"></i>{n}</span>' for n, c in zip(names, cols)) + "</div>"


def _kpi_html(cards: list[tuple]) -> str:
    return "".join(f'<div class="afya-kpi" style="border-top-color:{col}"><div class="afya-kpi-label">{a}</div><div class="afya-kpi-value">{b}</div><div class="afya-kpi-sub">{c}</div></div>'
                   for a, b, c, col in cards)


def _load() -> dict[str, pd.DataFrame]:
    names = ["yearly", "top", "age", "severity", "sepsis", "origin", "wards", "burns", "mechanism",
             "sgbv_cat", "sgbv_sexage", "sgbv_trend", "sgbv_repeat", "sgbv_mh"]
    return {n: _num(getattr(q, "q8_" + n)().copy()) for n in names}


class _Data:
    """Frames and lookups the sections share, computed once per render."""

    def __init__(self, D: dict[str, pd.DataFrame]):
        self.D = D
        self.y = {str(int(r.visit_year)): {c: _f(getattr(r, c)) for c in ("all_patients", "any_injury", "fractures", "open_wounds", "burns", "rta", "dislocation_sprain")}
                  for r in D["yearly"].itertuples()}
        self.top = [(r.condition, _f(r.patients), _f(r.female), _f(r.male), _f(r.mean_age)) for r in D["top"].itertuples()
                    if "epilepsy" not in str(r.condition).lower()][:10]
        self.age: dict[str, list[float]] = {}
        for r in D["age"].itertuples():
            self.age.setdefault(r.injury_type, [0.0] * 6)[AGE_MAP.get(r.age_band, 5)] += _f(r.patients)
        self.sev = {(r.condition, str(int(r.visit_year))): (_f(r.severe_patients), _f(r.patients)) for r in D["severity"].itertuples()}
        self.sep = {str(int(r.visit_year)): (_f(r.with_sepsis), _f(r.open_wound_patients), _f(r.pct_with_sepsis)) for r in D["sepsis"].itertuples()}
        self.origin = {r.origin: {k: _f(getattr(r, k)) for k in ("fractures", "open_wounds", "burns")} for r in D["origin"].itertuples()}
        self.burns = [(r.age_group, _f(r.female), _f(r.male)) for r in D["burns"].itertuples()]
        m = D["mechanism"]
        self.mech = (_f(m["with_any_external_cause_code"].sum()), _f(m["fracture_patients"].sum())) if not m.empty and "with_any_external_cause_code" in m.columns else (0.0, 0.0)
        self.sgcat = {r.broad_category: (None if pd.isna(r.total_patients) else _f(r.total_patients)) for r in D["sgbv_cat"].itertuples()}
        self.sgsa = {(r.sex, r.age_band): (None if pd.isna(r.patients) else _f(r.patients)) for r in D["sgbv_sexage"].itertuples() if r.broad_category == "Sexual violence"}
        self.tr = {str(int(r.visit_year)): {c: _f(getattr(r, c)) for c in ("all_patients", "any_sgbv_patients", "sexual_violence", "minors", "adults")} for r in D["sgbv_trend"].itertuples()}
        rp, mh = D["sgbv_repeat"], D["sgbv_mh"]
        self.repeat = (_f(rp.iloc[0]["repeat_presentation_patients"]), _f(rp.iloc[0]["sgbv_patients"])) if not rp.empty else (0.0, 0.0)
        self.mh = (_f(mh.iloc[0]["with_any_mental_health_code"]), _f(mh.iloc[0]["total_sgbv_patients"])) if not mh.empty else (0.0, 0.0)

    def v(self, y: str, k: str) -> float:
        return self.y.get(y, {}).get(k, 0.0)

    def chg(self, k: str) -> int:
        return _r(self.v("2024", k), self.v("2025", k))


def _cols(items: list[tuple], mx: float, color: str, h: int = 96) -> str:
    """Small column chart. items: (year, value, label, sub-label)."""
    out = ""
    for y, v, txt, sub in items:
        part = y in PARTIAL
        out += (f'<div class="in-vc" title="{_yl(y)}: {sub}" style="{"opacity:.55" if part else ""}"><div class="h" style="height:{h + 22}px"><b>{txt}</b>'
                f'<i style="height:{max(v / mx * h, 2) if v > 0 else 0:.0f}px;background:{color}"></i></div><div class="y">{_yl(y)}</div><div class="s">{sub}</div></div>')
    return f'<div class="in-row">{out}</div>'


def _pyr(rows: list[tuple[str, float | None, float | None]], mx: float, lw: str = "130px", bar_h: int = 14) -> str:
    out = ""
    for lab, f, m in rows:
        fv = "<5" if f is None else _n(f)
        mv = "<5" if m is None else _n(m)
        out += (f'<div class="in-pyr" style="--lw:{lw}" title="{lab}: {fv} female, {mv} male"><span>{lab}</span>'
                f'<div class="l"><span class="n">{fv if f != 0 else ""}</span><i style="width:{100 * (f or 0) / mx:.1f}%;height:{bar_h}px;background:{TEAL};border-radius:4px 0 0 4px"></i></div>'
                f'<div class="r"><i style="width:{100 * (m or 0) / mx:.1f}%;height:{bar_h}px;background:{BLUE};border-radius:0 4px 4px 0"></i><span class="n">{mv if m != 0 else ""}</span></div></div>')
    return out


def _sex_legend() -> str:
    return _legend(["Female", "Male"], [TEAL, BLUE])


def _short(c: str) -> str:
    return (c.replace("Acute headache associated with traumatic injury", "Traumatic headache").replace("Supracondylar fracture of humerus", "Supracondylar fracture, humerus")
            .replace("Open wound of knee or lower leg", "Open wound, knee or lower leg").replace("Fracture of ", "Fracture, "))


# ============================================================ header, KPIs
def _kpis(S: _Data) -> None:
    a, b = S.y.get("2025", {}), S.y.get("2024", {})
    sg = S.sgcat.get("All SGBV patients") or 0
    sx = S.sgcat.get("Sexual violence")
    cards = [
        ("Injury patients, 2025", _n(a.get("any_injury", 0)),
         f"{_arrow(S.chg('any_injury'))} vs 2024 ({_n(b.get('any_injury', 0))}), {100 * a.get('any_injury', 0) / max(a.get('all_patients', 1), 1):.1f}% of all patients", BLUE),
        ("Fracture patients, 2025", _n(a.get("fractures", 0)), f"{_arrow(S.chg('fractures'))} vs 2024 ({_n(b.get('fractures', 0))})", AMBER),
        ("SGBV patients, all years", _n(sg), f"{_n(sx) if sx is not None else 'n/a'} sexual violence · {_n(S.repeat[0])} seen again", BLUE),
    ]
    st.markdown(f'<div class="in-kp3">{_kpi_html(cards)}</div>', unsafe_allow_html=True)


def _top_insight(S: _Data) -> None:
    a24, a25 = S.v("2024", "any_injury"), S.v("2025", "any_injury")
    f24, f25 = S.v("2024", "fractures"), S.v("2025", "fractures")
    tot = _r(a24, a25)
    tail = ""
    if tot < 0 and (f24 - f25) >= (a24 - a25) > 0:
        tail = f", and fractures account for {_n(f24 - f25)} of the {_n(a24 - a25)} fewer patients ({S.chg('fractures'):+d}%)"
    elif S.chg("fractures"):
        tail = f", while fractures {_moved(S.chg('fractures'))}"
    sp = S.chg("dislocation_sprain")
    ui.insight_bar(f"<strong>Injury patients {_moved(tot)} in 2025 ({_n(a24)} to {_n(a25)}){tail}. Dislocations and sprains {_moved(sp)}.</strong>", "blue")


# ============================================================ A · injury burden
def _by_year(S: _Data) -> str:
    mx = max([S.v(y, k) for y in YEARS for k, _ in TYPES] + [1])
    h = 230
    out = ""
    for k, lab in TYPES:
        cols = ""
        for y in YEARS:
            v = S.v(y, k)
            hatch = "background-image:repeating-linear-gradient(135deg,transparent 0 4px,rgba(255,255,255,.55) 4px 6px);" if y in PARTIAL else ""
            cols += (f'<div class="in-col" title="{lab}, {_yl(y)}: {_n(v)} patients"><span>{int(v)}</span>'
                     f'<i style="height:{max(v / mx * h, 2) if v else 0:.0f}px;background-color:{YC[y]};{hatch}"></i></div>')
        out += f'<div><div class="in-cols" style="height:{h + 18}px">{cols}</div><div style="text-align:center;font-size:11.5px;margin-top:6px;font-weight:600">{lab}</div></div>'
    return _legend([_yl(y) for y in YEARS], [YC[y] for y in YEARS]) + f'<div class="in-grp">{out}</div>'


def _top_conditions(S: _Data) -> str:
    if not S.top:
        return ""
    mx = max(r[1] for r in S.top)
    out = ""
    for c, p, f, m, a in S.top:
        tip = f"{c}: {_n(p)} patients ({_n(f)} female, {_n(m)} male), mean age {a:.0f}"
        out += (f'<div class="in-top" title="{tip}"><span>{_short(c)}</span><div class="bar" style="width:{100 * p / mx:.1f}%">'
                f'<div style="width:{100 * f / max(p, 1):.1f}%;background:{TEAL}"></div><div style="width:{100 * m / max(p, 1):.1f}%;background:{BLUE}"></div></div>'
                f'<span class="v">{_n(p)}</span></div>')
    return _sex_legend() + out


def _by_age(S: _Data) -> str:
    ts = ["Fractures", "Dislocation / sprain", "Open wounds", "Burns", "RTA"]
    rows = [(TYPE_KEYS[t], S.age.get(t, [0.0] * 6)) for t in ts]
    mx = max([sum(v) for _, v in rows] + [1])
    out = ""
    for lab, v in rows:
        t = sum(v)
        tip = f"{lab}: " + ", ".join(f"{AGE_BANDS[i]} {int(x)}" for i, x in enumerate(v))
        seg = "".join(f'<div style="width:{100 * x / t:.1f}%;background:{AGE_COL[i]}">{int(x) if 100 * x / mx >= 4 else ""}</div>' for i, x in enumerate(v) if x and t)
        out += (f'<div class="in-st" style="--lw:150px" title="{tip}"><span>{lab}</span><div class="bar" style="width:{max(100 * t / mx, 3):.1f}%">{seg}</div><b>{_n(t)}</b></div>')
    return out


def _section_a(S: _Data) -> None:
    ui.section_header("Section A — Injury burden")
    l, r = st.columns([3, 2])
    with l:
        with ui.chart_card("Injury patients by type and year",
                           note="Patients per year. A patient can appear under more than one type. 2023 and 2026 are partial years (faded). Hover for counts."):
            st.markdown(_by_year(S), unsafe_allow_html=True)
    with r:
        with ui.chart_card("Top 10 injury conditions", note="Patients, all years. Bar split by sex. Hover for the sex split and mean age."):
            st.markdown(_top_conditions(S), unsafe_allow_html=True)
    with ui.chart_card("Injury patients by type and age band",
                       note="Bar length = patients with that type; colours = age band at first injury visit. All years. Hover for counts."):
        st.markdown(_legend(AGE_BANDS, AGE_COL) + _by_age(S), unsafe_allow_html=True)
    any_, tot = S.mech
    pct = 100 * any_ / tot if tot else 0
    shown = "under 1%" if 0 < pct < 1 else f"{pct:.0f}%"
    msg = f"A cause of injury is recorded for {shown} of fracture patients ({_n(any_)} of {_n(tot)}), so we cannot say how people are getting hurt."
    st.markdown(f'<div class="afya-link"><span>{msg}</span><a href="?tab=tab_dataquality" target="_self">Data quality →</a></div>', unsafe_allow_html=True)
    # insight
    top = S.top[0] if S.top else None
    fr = S.age.get("Fractures", [0.0] * 6)
    kids = (fr[0] + fr[1]) / sum(fr) * 100 if sum(fr) else 0
    rta = S.age.get("RTA", [0.0] * 6)
    rta_top = max(range(6), key=lambda i: rta[i]) if sum(rta) else None
    parts = []
    if top:
        parts.append(f"{_short(top[0])} is the single largest condition ({_n(top[1])} patients).")
    parts.append(f"Children under 13 are {kids:.0f}% of fracture patients")
    if rta_top is not None:
        parts[-1] += f", and road traffic injuries peak at {AGE_BANDS[rta_top]} ({_n(rta[rta_top])} of {_n(sum(rta))})."
    else:
        parts[-1] += "."
    ui.insight_bar("<strong>" + " ".join(parts) + "</strong>", "amber")


# ============================================================ B · severity and complications
def _section_b(S: _Data) -> None:
    ui.section_header("Section B — Severity and complications")

    def sv(cond: str):
        out = []
        for y in YEARS:
            s, n = S.sev.get((cond, y), (0, 0))
            out.append((y, 100 * s / n if n else 0.0, f"{100 * s / n:.1f}%" if n else "0%", f"{int(s)} of {_n(n)}"))
        return out

    c1, c2, c3 = st.columns(3)
    with c1:
        with ui.chart_card("Fractures: share flagged severe", note="% of fracture patients each year."):
            st.markdown(_cols(sv("Fractures"), 10, BLUE), unsafe_allow_html=True)
    with c2:
        with ui.chart_card("Road traffic injuries: share flagged severe", note="Few patients in some years, so read the counts as well as the %."):
            rt = sv("RTA")
            st.markdown(_cols(rt, max(10, max(x[1] for x in rt)), MUTED), unsafe_allow_html=True)
    with c3:
        with ui.chart_card("Open wounds: share with sepsis", note="% of open-wound patients each year."):
            sp = [(y, S.sep.get(y, (0, 0, 0))[2], f"{S.sep.get(y, (0, 0, 0))[2]:.1f}%", f"{int(S.sep.get(y, (0, 0, 0))[0])} of {_n(S.sep.get(y, (0, 0, 0))[1])}") for y in YEARS]
            st.markdown(_cols(sp, max(4, max(x[1] for x in sp)), RED), unsafe_allow_html=True)
    f = {y: S.sev.get(("Fractures", y), (0, 0)) for y in YEARS}
    p = {y: (100 * f[y][0] / f[y][1] if f[y][1] else 0) for y in YEARS}
    direction = "rose" if p["2025"] > p["2024"] else "fell" if p["2025"] < p["2024"] else "held"
    txt = f"Fracture severity {direction} from {p['2024']:.1f}% in 2024 to {p['2025']:.1f}% in 2025 and is {p['2026']:.1f}% so far in 2026."
    zero = [y for y in ("2024", "2025") if (S.sev.get(("RTA", y), (0, 0))[1] >= 20 and S.sev.get(("RTA", y), (0, 0))[0] == 0)]
    if zero:
        txt += f" No severe road traffic patient in {zero[0]} ({S.sev[('RTA', zero[0])][1]:.0f} patients) looks like severity undercoding, not a real zero."
    s24, s25 = S.sep.get("2024", (0, 0, 0))[2], S.sep.get("2025", (0, 0, 0))[2]
    if s25 < s24:
        txt += f" Sepsis among wound patients fell from {s24:.1f}% to {s25:.1f}%, which may be undercoding."
    ui.insight_bar(f"<strong>{txt}</strong>", "amber")


# ============================================================ C · where they come from
def _origin(S: _Data) -> str:
    parts = [("From Nandi", BLUE), ("Unknown", "#CBD5E1"), ("Outside Nandi", AMBER)]
    out = ""
    for lab, k in (("Fractures", "fractures"), ("Open wounds", "open_wounds"), ("Burns", "burns")):
        v = [S.origin.get(p, {}).get(k, 0) for p, _ in parts]
        t = sum(v)
        tip = f"{lab}: " + ", ".join(f"{p} {int(v[i])}" for i, (p, _) in enumerate(parts))
        seg = "".join(f'<div style="width:{100 * x / t:.1f}%;background:{parts[i][1]};color:{"#111827" if i == 1 else "#fff"};font-size:10px;font-weight:600;display:flex;align-items:center;justify-content:center">'
                      f'{round(100 * x / t) if 100 * x / t >= 8 else ""}{"%" if 100 * x / t >= 8 else ""}</div>' for i, x in enumerate(v) if x and t)
        out += (f'<div title="{tip}"><div style="font-size:12px;margin-bottom:4px"><b>{lab}</b> <span style="color:{MUTED}">n={_n(t)}</span></div>'
                f'<div style="display:flex;height:22px;border-radius:4px;overflow:hidden;background:#F0F1F5">{seg}</div></div>')
    return _legend(["From Nandi", "County not recorded", "Outside Nandi"], [BLUE, "#CBD5E1", AMBER]) + f'<div class="in-origin">{out}</div>'


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
    return dict(mode="burden", all="Any injury", noun="injury", hospital=dict(lat=q.HOSPITAL_LAT, lon=q.HOSPITAL_LON), conds=MAP_CONDS,
                labels=MAP_LABELS, wards=list(wards.values()), unplaced=unplaced, h=560)


def _section_c(S: _Data, payload: dict | None) -> None:
    ui.section_header("Section C — Where they come from")
    with ui.chart_card("Where patients live", note="Share of patients by county of residence, all years."):
        st.markdown(_origin(S), unsafe_allow_html=True)
    with ui.chart_card("Injury patients by ward", note="Nandi residents, 2024-2025. Pick an injury type. Darker = more patients. Hover a ward for its count."):
        if payload:
            geo.render_geo("ncd", payload, height=705, scrolling=False)
        else:
            _cap("No ward data returned for this card.")
    df = S.D["wards"]
    if df.empty or not payload:
        return
    a = df[df.condition_name == "Any injury"].sort_values("patients", ascending=False)
    if a.empty:
        return
    placed = {(w["w"], w["s"]) for w in payload["wards"]}
    a_placed = a[[(r.ward, r.sub_county) in placed for r in a.itertuples()]]
    tot = _f(a_placed.patients.sum())
    top = a_placed.iloc[0] if not a_placed.empty else a.iloc[0]
    nxt = list(a_placed.iloc[1:3].itertuples())
    more = f"; {nxt[0].ward} ({_n(nxt[0].patients)}) and {nxt[1].ward} ({_n(nxt[1].patients)}) come next" if len(nxt) == 2 else ""
    unplaced = int(payload["unplaced"].get("Any injury", 0))
    fr = S.origin.get("From Nandi", {}).get("fractures", 0)
    fr_t = sum(v.get("fractures", 0) for v in S.origin.values())
    out6 = S.origin.get("Outside Nandi", {}).get("fractures", 0)
    unk = S.origin.get("Unknown", {}).get("fractures", 0)
    ui.insight_bar(f"<strong>{top.ward} ward alone holds {100 * _f(top.patients) / tot:.0f}% of injury patients who can be placed ({_n(top.patients)} of {_n(tot)}){more}, "
                   f"and {_n(unplaced)} patients cannot be placed on the map. About {100 * fr / max(fr_t, 1):.0f}% of fracture patients live in Nandi, "
                   f"{_n(out6)} are recorded from outside it, and {100 * unk / max(fr_t, 1):.0f}% have no county.</strong>", "amber")


# ============================================================ D · burns
def _section_d(S: _Data) -> None:
    ui.section_header("Section D — Burns")
    l, r = st.columns(2)
    with l:
        with ui.chart_card("Burn patients by age and sex", note="Patients, all years. Female left, male right. Hover for counts."):
            mx = max([max(f, m) for _, f, m in S.burns] + [1])
            st.markdown(_sex_legend() + _pyr(S.burns, mx), unsafe_allow_html=True)
    with r:
        with ui.chart_card("Burns: share flagged severe", note="% of burn patients each year. Small numbers."):
            items = []
            for y in YEARS:
                s, n = S.sev.get(("Burns", y), (0, 0))
                items.append((y, 100 * s / n if n else 0.0, f"{100 * s / n:.1f}%" if n else "0%", f"{int(s)} of {_n(n)}"))
            st.markdown(_cols(items, max(10, max(x[1] for x in items)), AMBER), unsafe_allow_html=True)
    tot = sum(f + m for _, f, m in S.burns)
    if tot:
        tod = S.burns[0]
        t = tod[1] + tod[2]
        adults = [(a, f, m) for a, f, m in S.burns[3:] if f + m >= 10 and f >= 2 * m]
        skew = ""
        if adults:
            a, f, m = max(adults, key=lambda x: x[1] / (x[1] + x[2]))
            skew = f" The one adult skew is {a.split('(')[-1].rstrip(')')}, where {_n(f)} of {_n(f + m)} burn patients are female."
        even = "girls and boys about equally" if abs(tod[1] - tod[2]) <= 0.1 * t else ("more boys" if tod[2] > tod[1] else "more girls")
        ui.insight_bar(f"<strong>{tod[0].split(' (')[0]}s are {_n(t)} of {_n(tot)} burn patients ({100 * t / tot:.0f}%), {even}.{skew}</strong>", "amber")


# ============================================================ E · SGBV
def _section_e(S: _Data) -> None:
    ui.section_header("Section E — Sexual and gender-based violence (SGBV)")
    st.markdown('<div class="in-note">Counts are grouped, and any cell under 5 is hidden, to protect patients. Categories are by assault-context code and do not describe every case.</div>', unsafe_allow_html=True)
    l, r = st.columns(2)
    with l:
        with ui.chart_card("SGBV patients by category", note=f"Patients, all years, {_n(S.sgcat.get('All SGBV patients') or 0)} in total."):
            cats = [("Sexual violence", "Sexual violence"), ("Physical violence", "Physical violence"), ("Altercation/dispute", "Altercation / dispute"), ("Neglect", "Neglect"), ("Other/unspecified", "Other / unspecified")]
            mx = max([v or 0 for v in S.sgcat.values() if v is not None] + [1])
            rows = ""
            for i, (k, lab) in enumerate(cats):
                if k not in S.sgcat:
                    continue
                v = S.sgcat[k]
                txt = "<5" if v is None else _n(v)
                w = 0 if v is None else 100 * v / mx
                rows += (f'<div class="in-bars" title="{lab}: {txt} patients"><span>{lab}</span><div class="b"><i style="width:{max(w, 1.5) if v else 0:.1f}%;background:{AMBER if i == 0 else BLUE}"></i></div><span class="v">{txt}</span></div>')
            st.markdown(rows, unsafe_allow_html=True)
    with r:
        with ui.chart_card("Sexual violence by sex and age", note='Patients, all years. "<5" = hidden small cell.'):
            bands = [("Under 18", "Minor (<18)"), ("18-34", "Adult (18-34)"), ("35 and over", "35+")]
            rows = [(lab, S.sgsa.get(("F", k)), S.sgsa.get(("M", k))) for lab, k in bands]
            mx = max([v for _, f, m in rows for v in (f, m) if v] + [1])
            st.markdown(_sex_legend() + _pyr(rows, mx, "90px", 18), unsafe_allow_html=True)
    l, r = st.columns([3, 2])
    with l:
        with ui.chart_card("Sexual-violence patients by year",
                           note="Patients, with the rate per 1,000 patients seen below each year. 2026 is a partial year (faded)."):
            mx = max([S.tr.get(y, {}).get("sexual_violence", 0) for y in YEARS] + [1])
            h = 120
            out = ""
            for y in YEARS:
                t = S.tr.get(y, {})
                sv, all_ = t.get("sexual_violence", 0), t.get("all_patients", 0)
                rate = 1000 * sv / all_ if all_ else 0
                zero = not sv
                sub = (f'<span style="color:{RED};font-weight:700">code not in use</span>' if zero else f"{rate:.2f} per 1,000")
                tip = f"{_yl(y)}: {_n(sv)} sexual-violence patients ({_n(t.get('minors', 0))} under 18, {_n(t.get('adults', 0))} adults), {rate:.2f} per 1,000 of {_n(all_)} patients"
                out += (f'<div class="in-vc" title="{tip}" style="{"opacity:.6" if y == "2026" else ""}"><div class="h" style="height:{h + 22}px"><b>{int(sv)}</b>'
                        f'<div style="width:56%;display:flex;flex-direction:column;border-radius:3px 3px 0 0;overflow:hidden"><div style="height:{t.get("adults", 0) / mx * h:.0f}px;background:{BLUE}"></div>'
                        f'<div style="height:{t.get("minors", 0) / mx * h:.0f}px;background:{AMBER}"></div></div></div><div class="y">{_yl(y)}</div><div class="s">{sub}</div></div>')
            st.markdown(_legend(["Under 18", "18 and over"], [AMBER, BLUE]) + f'<div class="in-row" style="margin-top:6px">{out}</div>', unsafe_allow_html=True)
    with r:
        rep, repn = S.repeat
        mh, mhn = S.mh
        cards = [("Seen again for SGBV", f"{_n(rep)} of {_n(repn)}", "no patient came back with a new assault-context code" if not rep else "patients came back with a new assault-context code", AMBER),
                 ("With any mental-health code", f"{_n(mh)} of {_n(mhn)}", f"{100 * mh / mhn:.1f}% of SGBV patients" + ("; support after assault is not being recorded" if mhn and mh / mhn < 0.05 else "") if mhn else "", RED)]
        st.markdown('<div style="display:grid;gap:12px">' + _kpi_html(cards) + "</div>", unsafe_allow_html=True)
    # insight
    sx, tot = S.sgcat.get("Sexual violence"), S.sgcat.get("All SGBV patients")
    fm, mm = S.sgsa.get(("F", "Minor (<18)")), S.sgsa.get(("M", "Minor (<18)"))
    parts = []
    if sx and tot:
        parts.append(f"Sexual violence is {_n(sx)} of {_n(tot)} SGBV patients")
        if fm and mm:
            parts[-1] += f", and {_n(fm)} are females under 18 against {_n(mm)} males under 18 ({fm / mm:.0f} to 1)."
        else:
            parts[-1] += "."
    r = {y: 1000 * S.tr[y]["sexual_violence"] / S.tr[y]["all_patients"] for y in S.tr if S.tr[y]["all_patients"]}
    if all(y in r for y in ("2024", "2025", "2026")) and r["2026"] > r["2025"]:
        parts.append(f"The code is not fading: 2026 so far runs at {r['2026']:.2f} per 1,000 patients against {r['2024']:.2f} in 2024 and {r['2025']:.2f} in 2025, "
                     f"and 2023's zero means the code was not in use.")
    mh, mhn = S.mh
    tail = f" Only {_n(mh)} of {_n(mhn)} SGBV patients has any mental-health code, a service or coding gap to raise with the facility." if mhn and mh / mhn < 0.05 else ""
    if parts:
        ui.insight_bar("<strong>" + " ".join(parts) + "</strong>" + tail, "amber")


# ============================================================ page
def render() -> None:
    ui.inject_css()
    st.markdown(vi._CSS, unsafe_allow_html=True)
    st.markdown(_CSS, unsafe_allow_html=True)
    if q.DEMO:
        ui.demo_banner()
    _top = st.container()
    with ui.loading("Injury & trauma"):
        S = _Data(_load())
        payload = _ward_payload(S)
    with _top:
        ui.page_header("Injury & Trauma", "How many people come in injured, what the injuries are, who and where they are, how severe, and the sexual and gender-based violence (SGBV) picture", icon="injury", stat=(_n(S.y.get("2025", {}).get("any_injury", 0)), "injury patients, 2025"))
    _kpis(S)
    _top_insight(S)
    _section_a(S)
    _section_b(S)
    _section_c(S, payload)
    _section_d(S)
    _section_e(S)
