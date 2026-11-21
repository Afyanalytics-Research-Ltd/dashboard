"""
views_summary.py: Tab 1, Summary.

No SQL here. Q1.1 and Q1.2 (summary_queries.sql) feed Section A; every attention card in Section B reads a query that another
tab already runs, so each result is cached once and shared with that tab (see summary_queries.sql for the mapping).
Insights are hardcoded sentences with numbers computed from the data (step 1). Step 2 will swap the sentences for generated ones.
Sections: A the hospital at a glance · B needs attention (each card links to its tab) · C burden mix · D jump to a tab.
Q1.5 (keyword disease categories) is not used: its counts would not match the tab pages.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import nandi_queries as q
import nandi_ui_template as ui

BLUE, AMBER = ui.BLUE, ui.AMBER
MUTED, GREEN, GOLD, BROWN = ui.TEXT_MUTED, ui.GREEN, ui.GOLD, ui.BROWN
COL = ui._PH_COLORS


def _n(x) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except (TypeError, ValueError):
        return "n/a"


STATUS = {"Act": ("#A32D2D", "#FDECEC"), "Check": ("#B45309", "#FEF3E0"), "Watch": ("#0C447C", "#EAF1FA")}
ORDER = {"Act": 0, "Check": 1, "Watch": 2}

_CSS = f"""
<style>
.sm-kp {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }}
.sm-kp .afya-kpi {{ margin:0; }}
@media(max-width:1000px) {{ .sm-kp {{ grid-template-columns:1fr 1fr; }} }}
.sm-yrs {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; align-items:end; text-align:center; padding-top:6px; }}
.sm-yb {{ border-radius:4px 4px 0 0; margin:4px auto 0; width:58%; }}
.sm-yv {{ font-size:13px; font-weight:700; }}
.sm-yl {{ font-size:12px; margin-top:6px; font-weight:600; }}
.sm-yt {{ font-size:10.5px; color:{MUTED}; min-height:14px; }}
.sm-so {{ font-size:12.5px; margin-top:12px; padding-top:10px; border-top:1px solid {ui.BORDER}; color:#374151; line-height:1.5; }}
.sm-lg {{ font-size:12px; color:{MUTED}; margin:0 0 10px; display:flex; gap:16px; flex-wrap:wrap; }}
.sm-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:12px; }}
@media(max-width:900px) {{ .sm-grid {{ grid-template-columns:1fr; }} }}
a.sm-cd {{ display:block; text-decoration:none !important; color:inherit !important; background:#fff; border:1px solid {ui.BORDER};
          border-left:4px solid var(--c); border-radius:8px; padding:14px 16px; transition:box-shadow .15s, transform .15s; }}
a.sm-cd:hover {{ box-shadow:0 4px 14px rgba(0,0,0,.10); transform:translateY(-2px); }}
.sm-cdh {{ display:flex; justify-content:space-between; align-items:center; }}
.sm-tb {{ display:flex; gap:6px; align-items:center; font-size:12px; font-weight:600; color:var(--c); }}
.sm-tb svg {{ width:16px; height:16px; fill:none; stroke:var(--c); stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round; }}
.sm-st {{ font-size:10.5px; font-weight:700; letter-spacing:.05em; text-transform:uppercase; padding:3px 9px; border-radius:999px; }}
.sm-big {{ font-size:26px; font-weight:750; margin-top:10px; letter-spacing:-.01em; color:{ui.TEXT}; }}
.sm-sb {{ font-size:12px; color:{MUTED}; margin-top:1px; }}
.sm-ins {{ font-size:12.5px; line-height:1.5; margin-top:9px; color:#1f2937; }}
.sm-go {{ font-size:12px; font-weight:700; color:{GREEN}; margin-top:10px; }}
.sm-bars {{ display:grid; grid-template-columns:170px 1fr 64px; gap:9px 12px; align-items:center; font-size:12.5px; }}
.sm-bt {{ background:#EEF1F5; border-radius:4px; height:16px; }}
.sm-bt i {{ display:block; height:16px; background:{BLUE}; border-radius:4px; }}
.sm-bv {{ text-align:right; font-weight:600; }}
@media(max-width:700px) {{ .sm-bars {{ grid-template-columns:110px 1fr 54px; }} }}
.sm-jps {{ display:flex; flex-wrap:wrap; gap:8px; }}
a.sm-jp {{ display:flex; gap:7px; align-items:center; font-size:12.5px; font-weight:600; text-decoration:none !important; color:var(--c) !important;
          background:#fff; border:1px solid {ui.BORDER}; border-radius:999px; padding:6px 12px; }}
a.sm-jp:hover {{ background:#fafafa; }}
a.sm-jp svg {{ width:16px; height:16px; fill:none; stroke:var(--c); stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round; }}
</style>
"""


def _svg(icon: str) -> str:
    return f'<svg viewBox="0 0 24 24">{ui._ICONS[icon]}</svg>'


# ------------------------------------------------------------------ data
def _load() -> dict:
    names = ["q1_headline_kpis", "q1_year", "q11_diag", "q11_complete", "q8_sgbv_mh", "q3_gonococcal", "q9_summary", "q9_concentration",
             "q5_yearly", "q6_severity", "q11_ward_match", "q7_yearly", "q4_sickle_returning",
             "q3_infectious_patients", "q4_headline", "q6_yearly", "q8_yearly"]
    return {n: getattr(q, n)() for n in names}


def _ok(df) -> bool:
    return df is not None and not df.empty


def _pat(df, cond, period) -> float | None:
    """Patients for one condition in one period, from the q5/q6/q7 yearly frames."""
    if not _ok(df):
        return None
    r = df[(df.condition_name == cond) & (df.period.astype(str) == str(period))]
    return float(r.patients.iloc[0]) if len(r) else None


# ------------------------------------------------------------------ cards
def _cards(D: dict) -> list[dict]:
    out: list[dict] = []

    def add(status, icon, tab, name, big, sub, insight):
        out.append(dict(status=status, icon=icon, tab=tab, name=name, big=big, sub=sub, ins=insight))

    # 1 Data Quality: no diagnosis
    d, c = D["q11_diag"], D["q11_complete"]
    if _ok(d) and _ok(c):
        by = {int(r.visit_year): r.pct_no_diagnosis for r in c.itertuples()}
        ys = sorted(by)
        trail = ", ".join(f"{by[y]:.1f}% in {y}" for y in ys[:-1]) + f" and {by[ys[-1]]:.1f}% so far in {ys[-1]}"
        add("Act", "dataquality", "tab_dataquality", "Data Quality", f"{d.pct_no_diagnosis.iloc[0]:.1f}%", "of visits have no diagnosis",
            f"{trail.capitalize()}. Every disease count is a floor, and a fall in any of them may be thinner recording, not fewer patients.")

    # 2 Injury: SGBV mental-health coding
    m = D["q8_sgbv_mh"]
    if _ok(m) and m.total_sgbv_patients.iloc[0]:
        add("Act", "injury", "tab_injury", "Injury & Trauma", f"{_n(m.with_any_mental_health_code.iloc[0])} of {_n(m.total_sgbv_patients.iloc[0])}",
            "SGBV patients has a mental-health code",
            "Almost no SGBV patient is coded for mental health. That is a service or coding gap to raise with the facility.")

    # 3 Infectious: gonococcal under 18
    g = D["q3_gonococcal"]
    if _ok(g):
        u18 = g[g.age_group.astype(str).str.contains("<18")].visits.sum()
        add("Act", "infectious", "tab_infectious", "Infectious disease", _n(u18), "gonococcal visits in patients under 18",
            "Each one needs county child-protection reporting. Sepsis is also the leading diagnosis in toddlers.")

    # 4 Data Quality: 2025 vs 2024 visits
    y = D["q1_year"]
    if _ok(y):
        v = {int(r.visit_year): r.visits for r in y.itertuples()}
        if 2024 in v and 2025 in v and v[2024]:
            ch = 100 * (v[2025] / v[2024] - 1)
            add("Check", "dataquality", "tab_dataquality", "Data Quality", f"{ch:+.0f}%".replace("-", "−"), "visits in 2025 against 2024",
                "2025 is a complete year with a known export gap. Treat 2025 trends as understated until the facility confirms what is missing.")

    # 5 Workload: zero-contact clinicians
    s, cn = D["q9_summary"], D["q9_concentration"]
    if _ok(s):
        top = ""
        if _ok(cn):
            t = cn[cn.group_label.astype(str).str.startswith("Top 10")]
            if len(t):
                top = f"The top 10 clinicians hold {t.pct_of_total.iloc[0]:.0f}% of all contacts. "
        add("Check", "workload", "tab_workload", "Clinician Workload", f"{_n(s.zero_load_clinicians.iloc[0])} of {_n(s.total_clinicians.iloc[0])}",
            "clinicians recorded no contacts", top + "The facility cannot yet see its own clinical workload from this table.")

    # 6 Psychiatric: depression + anxiety
    mh = D["q5_yearly"]
    dep, anx, anyc = _pat(mh, "Depression", "All years"), _pat(mh, "Anxiety", "All years"), _pat(mh, "Any mental health", "All years")
    if dep is not None and anx is not None and anyc:
        add("Check", "mental", "tab_mental", "Psychiatric Conditions", f"{_n(dep + anx)} of {_n(anyc)}", "patients carry depression or anxiety",
            "Far below community levels. This points to under-diagnosis, not low need.")

    # 7 Respiratory: severe pneumonia share
    sv = D["q6_severity"]
    if _ok(sv):
        t = sv.groupby("visit_year")[["pneumonia_patients", "severe_patients"]].sum()
        if 2023 in t.index and 2025 in t.index and t.loc[2023, "pneumonia_patients"] and t.loc[2025, "pneumonia_patients"]:
            a = 100 * t.loc[2023, "severe_patients"] / t.loc[2023, "pneumonia_patients"]
            b = 100 * t.loc[2025, "severe_patients"] / t.loc[2025, "pneumonia_patients"]
            add("Check", "respiratory", "tab_resp", "Respiratory", f"{a:.0f}% → {b:.0f}%", "of pneumonia patients were severe, 2023 to 2025",
                "A fall this steep in two years looks like a recording change, not better outcomes.")

    # 8 Geography: ward boundary match
    w = D["q11_ward_match"]
    if _ok(w):
        r = w[w.match_status.astype(str).str.contains("No boundary")]
        if len(r):
            add("Check", "geo", "tab_geo", "Population & Geography", f"{r.pct_of_visits.iloc[0]:.1f}%", "of visits have no ward boundary match",
                f"{_n(r.visits.iloc[0])} visits are missing from the map. Tinderet is spelled two ways in the sub-county field, which is part of it.")

    # 9 Maternal: pregnancy loss
    my = D["q7_yearly"]
    l24, l25, a24, a25 = (_pat(my, "Pregnancy loss", 2024), _pat(my, "Pregnancy loss", 2025), _pat(my, "Any maternal", 2024), _pat(my, "Any maternal", 2025))
    if l24 and l25 is not None and a24 and a25 is not None:
        ch = 100 * (l25 / l24 - 1)
        flat = abs(a25 / a24 - 1) <= 0.03
        tail = (f"Maternal patients overall held flat ({_n(a24)} to {_n(a25)}), so the change is in pregnancy loss, not in volume." if flat
                else f"Maternal patients overall moved from {_n(a24)} to {_n(a25)}.")
        add("Watch", "maternal", "tab_maternal", "Maternal & Obstetric", f"{ch:+.0f}%", f"pregnancy-loss patients, {_n(l24)} to {_n(l25)}", tail)

    # 10 NCDs: sickle cell
    sk = D["q4_sickle_returning"]
    if _ok(sk):
        add("Watch", "ncd", "tab_ncds", "NCDs", _n(sk.patients.sum()),
            f"sickle cell patients, {_n(sk.multi_year_patients.sum())} seen in more than one year",
            "Seen mostly in crisis, with little sign of follow-up care.")
    return sorted(out, key=lambda c: ORDER[c["status"]])


def _card_html(c: dict) -> str:
    fg, bg = STATUS[c["status"]]
    return (f'<a class="sm-cd" href="?tab={c["tab"]}" target="_self" style="--c:{COL[c["icon"]]}">'
            f'<div class="sm-cdh"><span class="sm-tb">{_svg(c["icon"])}{c["name"]}</span>'
            f'<span class="sm-st" style="color:{fg};background:{bg}">{c["status"]}</span></div>'
            f'<div class="sm-big">{c["big"]}</div><div class="sm-sb">{c["sub"]}</div><div class="sm-ins">{c["ins"]}</div>'
            f'<div class="sm-go">Open {c["name"]} →</div></a>')


# ------------------------------------------------------------------ sections
def _section_a(D: dict) -> None:
    k, y = D["q1_headline_kpis"], D["q1_year"]
    ui.section_header("Section A: The hospital at a glance")
    if _ok(k):
        r = k.iloc[0]
        ui.kpi_row([
            dict(label="Clinical visits", value=_n(r.total_visits), sub=f"{int(r.earliest_year)} to {int(r.latest_year)}", accent="blue"),
            dict(label="Unique patients", value=_n(r.unique_patients), sub="across all years", accent="blue"),
            dict(label="Female patients", value=f"{r.pct_female:.1f}%", sub="of visits", accent="blue"),
            dict(label="Under 5", value=f"{r.pct_under5:.1f}%", sub="of visits", accent="blue"),
        ])
        st.markdown("<style>.afya-kpis{grid-template-columns:repeat(4,1fr) !important}</style>", unsafe_allow_html=True)
    if not _ok(y):
        return
    yrs = [(int(r.visit_year), float(r.visits), float(r.unique_patients)) for r in y.itertuples()]
    lo, hi, mx = min(a for a, _, _ in yrs), max(a for a, _, _ in yrs), max(b for _, b, _ in yrs) or 1
    cols = ""
    for yr, v, _ in yrs:
        partial = yr in (lo, hi)
        gap = yr == 2025
        color = AMBER if gap else BLUE
        op = "opacity:.45;" if partial else ""
        tag = "Partial, to date" if yr == hi else "Partial year" if yr == lo else "Export gap" if gap else ""
        cols += (f'<div><div class="sm-yv">{_n(v)}</div><div class="sm-yb" style="height:{150 * v / mx:.0f}px;background:{color};{op}"></div>'
                 f'<div class="sm-yl">{yr}</div><div class="sm-yt">{tag}</div></div>')
    so = ""
    d = {a: (b, c) for a, b, c in yrs}
    if 2024 in d and 2025 in d and d[2024][0]:
        drop = 100 * (1 - d[2025][0] / d[2024][0])
        pch = 100 * (d[2025][1] / d[2024][1] - 1) if d[2024][1] else 0
        if drop > 5 and abs(pch) <= 3:
            so = (f"<b>2025 looks like a fall, but part of it is missing data.</b> Visits dropped {drop:.0f}% from 2024 while unique patients held steady "
                  f"({_n(d[2024][1])} to {_n(d[2025][1])}), so fewer visits per patient, not fewer patients.")
        elif drop > 5:
            so = (f"<b>Visits dropped {drop:.0f}% from 2024 to 2025</b> and unique patients moved from {_n(d[2024][1])} to {_n(d[2025][1])}. "
                  "2025 has a known export gap, so read the fall with care.")
    with ui.chart_card("Visits by year", note="2023 and 2026 are partial years. 2025 is complete but has a known export gap."):
        st.markdown(f'<div class="sm-yrs">{cols}</div>' + (f'<div class="sm-so">{so}</div>' if so else ""), unsafe_allow_html=True)


def _section_b(cards: list[dict]) -> None:
    ui.section_header("Section B: Needs attention")
    lg = ('<div class="sm-lg"><span><b style="color:#A32D2D">Act</b> a decision or report is needed</span>'
          '<span><b style="color:#B45309">Check</b> looks wrong, confirm with the facility</span>'
          '<span><b style="color:#0C447C">Watch</b> worth knowing</span></div>')
    st.markdown(lg + '<div class="sm-grid">' + "".join(_card_html(c) for c in cards) + "</div>", unsafe_allow_html=True)


def _section_c(D: dict) -> None:
    ui.section_header("Section C: Burden mix")
    inf, nc = D["q3_infectious_patients"], D["q4_headline"]
    rows = []
    if _ok(inf):
        r = inf[inf.period.astype(str) == "2025"]
        if len(r):
            rows.append(("Infectious disease", float(r.infectious_patients.iloc[0])))
    if _ok(D["q6_yearly"]):
        v = _pat(D["q6_yearly"], "Any respiratory", 2025)
        if v is not None:
            rows.append(("Respiratory", v))
    if _ok(nc):
        r = nc[nc.period.astype(str) == "2025"]
        if len(r):
            rows.append(("NCDs", float(r.ncd_patients.iloc[0])))
    if _ok(D["q8_yearly"]):
        r = D["q8_yearly"][D["q8_yearly"].visit_year == 2025]
        if len(r):
            rows.append(("Injury & Trauma", float(r.any_injury.iloc[0])))
    v = _pat(D["q5_yearly"], "Any mental health", 2025)
    if v is not None:
        rows.append(("Psychiatric Conditions", v))
    v = _pat(D["q7_yearly"], "Any maternal", 2025)
    if v is not None:
        rows.append(("Maternal & Obstetric", v))
    if not rows:
        return
    rows.sort(key=lambda t: -t[1])
    mx = rows[0][1] or 1
    bars = "".join(f'<div>{n}</div><div class="sm-bt"><i style="width:{100 * v / mx:.1f}%"></i></div><div class="sm-bv">{_n(v)}</div>' for n, v in rows)
    top2 = [r for r in rows if r[0] in ("Infectious disease", "Respiratory")]
    so = ""
    if len(top2) == 2 and rows[0][0] in ("Infectious disease", "Respiratory") and rows[1][0] in ("Infectious disease", "Respiratory"):
        small = " and ".join(f"{_n(v)} for {n.lower()}" for n, v in rows[-2:])
        so = (f"<b>Infection and respiratory illness are the bulk of what the hospital sees.</b> Infectious disease reached {_n(dict(rows)['Infectious disease'])} "
              f"patients in 2025 and respiratory {_n(dict(rows)['Respiratory'])}, against {small} care.")
    with ui.chart_card("Patients by disease group, 2025", note="Groups overlap (pneumonia counts as both infectious and respiratory), so do not add them up."):
        st.markdown(f'<div class="sm-bars">{bars}</div>' + (f'<div class="sm-so">{so}</div>' if so else ""), unsafe_allow_html=True)


def _section_d() -> None:
    ui.section_header("Section D: Jump to a tab")
    items = [("geo", "tab_geo", "Population & Geography"), ("infectious", "tab_infectious", "Infectious disease"), ("ncd", "tab_ncds", "NCDs"),
             ("mental", "tab_mental", "Psychiatric Conditions"), ("respiratory", "tab_resp", "Respiratory"), ("maternal", "tab_maternal", "Maternal & Obstetric"),
             ("injury", "tab_injury", "Injury & Trauma"), ("workload", "tab_workload", "Clinician Workload"), ("dataquality", "tab_dataquality", "Data Quality")]
    st.markdown('<div class="sm-jps">' + "".join(
        f'<a class="sm-jp" href="?tab={t}" target="_self" style="--c:{COL[i]}">{_svg(i)}{n}</a>' for i, t, n in items) + "</div>", unsafe_allow_html=True)


def _top_insight(D: dict, cards: list[dict]) -> None:
    act = sum(c["status"] == "Act" for c in cards)
    chk = sum(c["status"] == "Check" for c in cards)
    d, s = D["q11_diag"], D["q9_summary"]
    lim = []
    if _ok(d):
        lim.append(f"no diagnosis on {d.pct_no_diagnosis.iloc[0]:.0f}% of visits")
    lim.append("a gap in the 2025 export")
    if _ok(s):
        lim.append(f"{_n(s.zero_load_clinicians.iloc[0])} clinicians with no recorded contacts")
    words = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight"}
    head = (f"{words.get(act, act)} finding{'s' if act != 1 else ''} {'need' if act != 1 else 'needs'} action and "
            f"{words.get(chk, chk).lower()} {'need' if chk != 1 else 'needs'} checking with the facility.")
    lim_txt = (', '.join(lim[:-1]) + ' and ' + lim[-1]) if len(lim) > 1 else lim[0]
    ui.insight_bar(f"<strong>{head}</strong> {lim_txt[0].upper() + lim_txt[1:]} limit how far every number on the other pages "
                   "can be trusted. Open any card below to go to its tab.", "amber")


def render() -> None:
    ui.inject_css()
    st.markdown(_CSS, unsafe_allow_html=True)
    if q.DEMO:
        ui.demo_banner()
    _top = st.container()
    with ui.loading("Summary"):
        D = _load()
        cards = _cards(D)
    k = D["q1_headline_kpis"]
    with _top:
        ui.page_header("Summary", "What needs attention at Kapsabet CRH, and where to look next.", icon="summary",
                       stat=(_n(k.total_visits.iloc[0]) if _ok(k) else "n/a",
                             f"clinical visits, {int(k.earliest_year.iloc[0])} – {int(k.latest_year.iloc[0])}" if _ok(k) else "clinical visits"))
    _top_insight(D, cards)
    _section_a(D)
    _section_b(cards)
    _section_c(D)
    _section_d()
