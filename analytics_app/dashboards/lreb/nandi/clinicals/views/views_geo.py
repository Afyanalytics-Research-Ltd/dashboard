"""
views_geo.py: Tab 2, Population & Geography.

No SQL here: every number comes from queries.q2_* / q1_*. This file only shapes
DataFrames into the payloads the map engine (template/geo_engine.js) and the
Plotly charts expect, then lays the page out with ui_template helpers.

Sections: A where patients come from · B who comes from other counties ·
C distance from the hospital by disease group · D disease burden by place ·
E catchment shift 2024 to 2025 · F population profile.
Geographic data confidence and the issue register live on the Data Quality tab.
"""
from __future__ import annotations

import json
import re

import pandas as pd
import streamlit as st

import nandi_geo_map as geo
import nandi_queries as q
import nandi_ui_template as ui

CATS_ORDER = ["Infectious disease", "Respiratory", "Injury & trauma", "NCD", "Maternal & obstetric", "Mental health", "Malaria"]
BAND_EDGES = [2, 5, 10, 20, 50, 100]
BAND_LABELS = ["0–2 km", "2–5 km", "5–10 km", "10–20 km", "20–50 km", "50–100 km", "100+ km"]
NANDI_SUB_OK = {"emgwen", "chesumei", "mosop", "nandihills", "aldai", "tinderettindiret"}
MIN_PLACE_VISITS = 30  # below this a share is not shown on a map


def _nk(s) -> str:
    return re.sub(r"[^a-z]", "", str(s).lower())


def _sub(name):
    """Tinderet and Tindiret are one sub-county spelled two ways."""
    k = _nk(name)
    return "Tinderet / Tindiret" if ("tinderet" in k or "tindiret" in k) else name


def _num(x, default=0.0):
    try:
        return default if pd.isna(x) else float(x)
    except (TypeError, ValueError):
        return default


# ============================================================ payload builders
def _counties(profile: pd.DataFrame) -> list[dict]:
    out = []
    for r in profile.itertuples():
        out.append(dict(name=r.county, v=int(r.visits), p=int(r.patients), v24=int(_num(r.visits_2024)), v25=int(_num(r.visits_2025)),
                        f=None if pd.isna(r.pct_female) else float(r.pct_female), age=_num(r.mean_age),
                        inj=int(_num(r.injury_visits)), acu=int(_num(r.acute_illness_visits)), ncd=int(_num(r.ncd_visits))))
    return out


def _places(place: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Sub-county and ward rows. Tinderet/Tindiret merged; non-Nandi sub-counties pass through."""
    if place.empty:
        return [], []
    df = place.copy()
    df["sub_county"] = df["sub_county"].map(lambda s: _sub(s) if pd.notna(s) else s)
    nandi = df["county"].map(_nk) == "nandi"
    df = df[~nandi | df["sub_county"].map(_nk).isin(NANDI_SUB_OK) | df["level"].eq("county")]
    subs = df[df.level == "sub"].copy()
    subs["fw"] = subs["pct_female"].fillna(0) * subs["visits"]
    g = subs.groupby(["county", "sub_county"], as_index=False).agg(visits=("visits", "sum"), pat=("unique_patients", "sum"),
                                                                  v24=("visits_2024", "sum"), v25=("visits_2025", "sum"), fw=("fw", "sum"))
    subs_out = [dict(county=r.county, n=r.sub_county, v=int(r.visits), p=int(r.pat), v24=int(r.v24), v25=int(r.v25),
                     f=round(r.fw / r.visits, 1) if r.visits else None) for r in g.itertuples()]
    wards = df[df.level == "ward"].groupby(["county", "sub_county", "ward"], as_index=False).agg(
        visits=("visits", "sum"), pat=("unique_patients", "sum"), v24=("visits_2024", "sum"), v25=("visits_2025", "sum"))
    wards_out = [dict(county=r.county, sub=r.sub_county, n=r.ward, v=int(r.visits), p=int(r.pat), v24=int(r.v24), v25=int(r.v25)) for r in wards.itertuples()]
    return subs_out, wards_out


def _category_payload(cat: pd.DataFrame) -> tuple[dict, dict]:
    """Share of each place's visits that fall in each disease category, at county / sub / ward level, plus ward LQ."""
    if cat.empty:
        return {}, {}
    df = cat.copy()
    df["sub_county"] = df["sub_county"].map(_sub)
    nandi = df["county"].map(_nk) == "nandi"
    df = df[~nandi | df["sub_county"].map(_nk).isin(NANDI_SUB_OK) | df["sub_county"].eq("(none)")]
    areas = df.drop_duplicates(["county", "sub_county", "ward"])[["county", "sub_county", "ward", "area_visits"]]
    out: dict = {c: {"county": {}, "sub": {}, "ward": {}} for c in CATS_ORDER}
    # county
    ca = areas.groupby("county").area_visits.sum()
    cc = df.groupby(["category", "county"]).category_visits.sum()
    for (cat_name, county), n in cc.items():
        if cat_name in out and ca.get(county, 0):
            out[cat_name]["county"][_nk(county)] = dict(pct=round(n * 100 / ca[county], 1), visits=int(ca[county]))
    # sub
    real = df[df.sub_county != "(none)"]
    sa = areas[areas.sub_county != "(none)"].groupby(["county", "sub_county"]).area_visits.sum()
    sc = real.groupby(["category", "county", "sub_county"]).category_visits.sum()
    for (cat_name, county, sub), n in sc.items():
        if cat_name in out and sa.get((county, sub), 0):
            out[cat_name]["sub"][f"{_nk(county)}|{sub}"] = dict(pct=round(n * 100 / sa[(county, sub)], 1), visits=int(sa[(county, sub)]))
    # ward
    realw = real[real.ward != "(none)"]
    for r in realw.itertuples():
        if r.category in out and r.area_visits:
            out[r.category]["ward"][f"{_nk(r.county)}|{r.sub_county}|{r.ward}"] = dict(pct=round(r.category_visits * 100 / r.area_visits, 1), visits=int(r.area_visits))
    # location quotient: ward share / Nandi share, wards with 100+ visits
    lq: dict = {c: [] for c in CATS_ORDER}
    for c in CATS_ORDER:
        base = out[c]["county"].get("nandi", {}).get("pct")
        if not base:
            continue
        rows = [(k.split("|")[2], d["pct"] / base) for k, d in out[c]["ward"].items() if k.startswith("nandi|") and d["visits"] >= 100]
        lq[c] = [[w, round(v, 2)] for w, v in sorted(rows, key=lambda x: -x[1])[:5]]
    return out, lq


def _dp(ring, tol):
    """Douglas-Peucker on an open polyline of [lon, lat] points."""
    keep = [False] * len(ring)
    keep[0] = keep[-1] = True
    stack = [(0, len(ring) - 1)]
    while stack:
        a, b = stack.pop()
        (x1, y1), (x2, y2) = ring[a], ring[b]
        dx, dy = x2 - x1, y2 - y1
        norm = (dx * dx + dy * dy) ** 0.5 or 1e-12
        md, mi = 0.0, -1
        for i in range(a + 1, b):
            d = abs(dy * (ring[i][0] - x1) - dx * (ring[i][1] - y1)) / norm
            if d > md:
                md, mi = d, i
        if md > tol and mi > 0:
            keep[mi] = True
            stack += [(a, mi), (mi, b)]
    return [p for p, k in zip(ring, keep) if k]


def _simplify(ring, tol=0.0015):
    """Simplify a polygon ring. A closed ring has first == last, which makes the Douglas-Peucker
    base line degenerate (it would collapse the ring to two points), so split it at the farthest vertex."""
    if len(ring) < 8:
        return ring
    if ring[0] == ring[-1]:
        far = max(range(1, len(ring) - 1), key=lambda i: (ring[i][0] - ring[0][0]) ** 2 + (ring[i][1] - ring[0][1]) ** 2)
        return _dp(ring[: far + 1], tol)[:-1] + _dp(ring[far:], tol)
    return _dp(ring, tol)


def _rings(geojson) -> list:
    try:
        g = json.loads(geojson) if isinstance(geojson, str) else geojson
        if g.get("type") == "Feature":
            g = g["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"] if g["type"] == "MultiPolygon" else []
        return [[[round(x, 4), round(y, 4)] for x, y in _simplify(p[0])] for p in polys if p]
    except Exception:  # noqa: BLE001
        return []


def _bounds(b: pd.DataFrame) -> dict:
    """{'sub': {county: {sub: {rings}}}, 'ward': {county: {sub: {ward: {rings}}}}}, keys lower-cased letters only."""
    out = {"sub": {}, "ward": {}}
    for r in b.itertuples():
        rings = _rings(r.boundary_geojson)
        if not rings:
            continue
        if str(r.geo_level).upper() == "SUB_COUNTY":
            out["sub"].setdefault(_nk(r.county), {})[_nk(r.geo_name)] = dict(rings=rings)
        elif str(r.geo_level).upper() == "WARD":
            out["ward"].setdefault(_nk(r.county), {}).setdefault(_nk(r.parent_name), {})[_nk(r.geo_name)] = dict(rings=rings)
    return out


def _weighted_median(km, w):
    pairs = sorted(zip(km, w))
    tot = sum(w)
    run = 0
    for k, x in pairs:
        run += x
        if run >= tot / 2:
            return k
    return None


def _rings_payload(loc: pd.DataFrame) -> dict:
    if loc.empty:
        return dict(bands=BAND_LABELS, edges=BAND_EDGES, cats=["All visits"] + CATS_ORDER, byCat={}, locs=[])
    cats = ["All visits"] + CATS_ORDER
    by = {}
    for c in cats:
        d = loc[loc.category == c]
        tot = d.visits.sum()
        if not tot:
            continue
        idx = pd.cut(d.km, [-1] + BAND_EDGES + [1e9], labels=False)
        pct = [round(float(d.visits[idx == i].sum()) * 100 / tot, 1) for i in range(len(BAND_LABELS))]
        by[c] = dict(pct=pct, median=round(float(_weighted_median(d.km, d.visits)), 1))
    locs = {}
    for r in loc.itertuples():
        o = locs.setdefault(r.locality, dict(n=r.locality, lat=float(r.latitude), lon=float(r.longitude), km=float(r.km), v={}))
        o["v"][r.category] = int(r.visits)
    return dict(bands=BAND_LABELS, edges=BAND_EDGES, cats=cats, byCat=by, locs=list(locs.values()))


def _growth(g: pd.DataFrame) -> tuple[list[dict], float]:
    if g.empty:
        return [], 0.0
    d = g[g.locality.str.lower() != "kapsabet"].copy()
    d["tot"] = d.visits_2024 + d.visits_2025
    d = d.sort_values("tot", ascending=False).head(9)
    d["pct"] = (d.visits_2025 - d.visits_2024) * 100 / d.visits_2024.where(d.visits_2024 > 0)
    med = float(d.pct.median())
    out = []
    for r in d.itertuples():
        pts = int(round(r.pct - med))
        out.append(dict(n=r.locality, v24=int(r.visits_2024), v25=int(r.visits_2025), pts=pts,
                        watch=bool(pts >= 2 and _num(r.share_sub_2025) > _num(r.share_sub_2024))))
    out.sort(key=lambda x: -x["pts"])
    return out, round(med)


@st.cache_data(ttl=3600, show_spinner=False)
def build_payload() -> dict:
    profile, place = q.q2_county_profile(), q.q2_place_summary()
    subs, wards = _places(place)
    catv, lq = _category_payload(q.q2_category_by_place())
    growth, gmed = _growth(q.q2_locality_growth())
    kpi = q.q1_headline_kpis()
    bounds = _bounds(q.q2_boundaries()) if not q.q2_boundaries().empty else {}
    return dict(hospital=dict(lat=q.HOSPITAL_LAT, lon=q.HOSPITAL_LON), counties=_counties(profile), subs=subs, wards=wards,
                cats=CATS_ORDER, catv=catv, lq=lq, rings=_rings_payload(q.q2_locality_distance()), growth=growth, growthMedian=gmed,
                femaleBaseline=float(kpi.pct_female.iloc[0]) if not kpi.empty else 57.0, bounds=bounds)


# ============================================================ page text helpers
def _fmt(n) -> str:
    return f"{int(round(n)):,}"


def _readings(profile: pd.DataFrame) -> list[tuple[str, str]]:
    """The three 'why' cards, written from the data so the numbers never go stale."""
    d = profile[(profile.county != "Nandi") & (profile.visits >= 30)].copy()
    if d.empty:
        return [("Not enough volume", "No other county has 30+ visits yet, so no pattern is read.")]
    d["rpp"] = d.visits / d.patients
    d["ncd_pct"] = d.ncd_visits * 100 / d.visits
    d["acute_inj_pct"] = (d.acute_illness_visits + d.injury_visits) * 100 / d.visits
    out = []
    a = d.sort_values(["ncd_pct", "rpp"], ascending=False).iloc[0]
    out.append(("Chronic follow-up pull", f"{a.county}: mean age {a.mean_age:.0f}, {a.rpp:.1f} visits per patient and the highest NCD share ({a.ncd_pct:.0f}%). Looks like repeat care for ongoing conditions."))
    b = d.sort_values("acute_inj_pct", ascending=False).iloc[0]
    out.append(("Acute and injury referrals", f"{b.county}: mean age {b.mean_age:.0f}, {b.acute_illness_visits * 100 / b.visits:.0f}% acute illness and {b.injury_visits * 100 / b.visits:.0f}% injury. Consistent with nearby emergency or referral demand."))
    m = d.sort_values("pct_female").iloc[0]
    out.append(("Who is missing from the mix", f"{m.county}: only {m.pct_female:.0f}% female against a 57% hospital baseline. Consistent with working-age men attending while away from home."))
    return out


# ============================================================ render
def render() -> None:
    ui.inject_css()
    if q.DEMO:
        ui.demo_banner()
    _top = st.container()

    with ui.loading("Population & Geography"):
        origin, profile, place = q.q2_origin(), q.q2_county_profile(), q.q2_place_summary()
        kpi, burden = q.q1_headline_kpis(), q.q1_disease_category_summary()
        P = build_payload()
    with _top:
        ui.page_header("Population & Geography", "Where Kapsabet CRH's patients come from across Kenya, what they come for, and how the catchment moved", icon="geo", stat=(_fmt(kpi.unique_patients.iloc[0]) if not kpi.empty else "n/a", "unique patients"))

    # ---- KPI strip
    nandi_o = origin[origin.patient_origin == "Nandi County"]
    other_o = origin[origin.patient_origin == "Other Kenyan county"]
    subs = pd.DataFrame(P["subs"])
    wards = pd.DataFrame(P["wards"])
    emg = subs[(subs.county == "Nandi") & (subs.n == "Emgwen")].v.sum() / max(1, subs[subs.county == "Nandi"].v.sum()) * 100 if not subs.empty else 0
    topw = wards[wards.county == "Nandi"].sort_values("v", ascending=False).head(1) if not wards.empty else wards
    topw_share = topw.v.iloc[0] * 100 / max(1, wards[wards.county == "Nandi"].v.sum()) if len(topw) else 0
    tc = profile[profile.county != "Nandi"].sort_values("visits", ascending=False).head(1)
    top_b = burden.sort_values("visits", ascending=False).head(2)
    ui.kpi_row([
        dict(label="Unique patients", value=_fmt(kpi.unique_patients.iloc[0]) if not kpi.empty else "n/a", sub=f"{_fmt(kpi.total_visits.iloc[0])} visits, {int(kpi.earliest_year.iloc[0])}–{int(kpi.latest_year.iloc[0])}" if not kpi.empty else "", accent="blue"),
        dict(label="Nandi-origin share", value=f"{nandi_o.pct_of_total.iloc[0]:.2f}%" if len(nandi_o) else "n/a", sub="Of all visits", accent="teal"),
        dict(label="Emgwen share", value=f"{emg:.1f}%", sub="Of matched Nandi visits", accent="blue"),
        dict(label="Top ward", value=topw.n.iloc[0] if len(topw) else "n/a", sub=f"{topw_share:.1f}% of matched visits", accent="blue"),
        dict(label="From other counties", value=_fmt(other_o.visits.iloc[0]) if len(other_o) else "0",
             sub=f"{other_o.pct_of_total.iloc[0]:.2f}% of visits" + (f" · {tc.county.iloc[0]} leads ({int(tc.visits.iloc[0])})" if len(tc) else "") if len(other_o) else "", accent="amber"),
        dict(label="Largest burden", value=top_b.category.iloc[0] if len(top_b) else "n/a",
             sub=(f"{_fmt(top_b.visits.iloc[0])} visits · {top_b.category.iloc[1]} {_fmt(top_b.visits.iloc[1])}" if len(top_b) > 1 else ""), accent="amber"),
    ])

    # ---- A
    ui.section_header("Section A — Where patients come from")
    left, right = st.columns([1.25, 1])
    with left:
        geo.render_geo("catchment", P, height=770)
    with right:
        order = ["Nandi County", "Residence unmatched", "Other Kenyan county", "International"]
        o = origin.set_index("patient_origin")
        vals = [float(o.pct_of_total.get(k, 0)) + (float(o.pct_of_total.get("Noise / unresolvable", 0)) if k == "Residence unmatched" else 0) for k in order]
        with ui.chart_card("Patient origin", "Q2.5"):
            st.plotly_chart(ui.bar_chart(order, vals, [ui.PRIMARY, ui.AMBER, ui.BLUE, ui.NEUTRAL], [f"{v:.2f}%" for v in vals], height=312), use_container_width=True, config=ui.PC_CFG)
        with ui.chart_card("Top ward share of Nandi visits", "N1a"):
            nw = wards[wards.county == "Nandi"].sort_values("v", ascending=False).head(6) if not wards.empty else wards
            tot = max(1, wards[wards.county == "Nandi"].v.sum()) if not wards.empty else 1
            if len(nw):
                st.plotly_chart(ui.bar_chart(list(nw.n), list(nw.v * 100 / tot), [ui.PRIMARY] + [ui.BLUE] * (len(nw) - 1), [f"{v:.1f}%" for v in nw.v * 100 / tot], height=312), use_container_width=True, config=ui.PC_CFG)
    with st.expander("Map data check: what detail each county has"):
        bsub, bward = P["bounds"].get("sub", {}), P["bounds"].get("ward", {})
        chk = []
        for cname in sorted({r["name"] for r in P["counties"]}, key=lambda c: -next(r["v"] for r in P["counties"] if r["name"] == c)):
            ns = int((subs.county == cname).sum()) if not subs.empty else 0
            nw = int((wards.county == cname).sum()) if not wards.empty else 0
            chk.append(dict(county=cname, sub_counties=ns, wards=nw,
                            real_sub_shapes=len(bsub.get(_nk(cname), {})), real_ward_shapes=sum(len(v) for v in bward.get(_nk(cname), {}).values())))
        st.dataframe(pd.DataFrame(chk), hide_index=True, use_container_width=True)
        st.caption("Sub-counties and wards only appear when a county has rows here. Shapes are real when real_*_shapes is filled, otherwise a schematic division of the county. "
                   "Pick a county in the map's 'Go to' list, or click it, to see its sub-counties and wards.")
    ui.insight_bar(f"<strong>The catchment is Kapsabet, with a thin tail across western Kenya.</strong> {topw_share:.0f}% of matched visits come from one ward and "
                   f"{nandi_o.pct_of_total.iloc[0] if len(nandi_o) else 0:.0f}% from Nandi, but {_fmt(other_o.visits.iloc[0]) if len(other_o) else 0} visits came from other counties, mostly the neighbours. "
                   "Switch the map to <i>Main reason</i> to see what each county comes for.")

    # ---- B
    ui.section_header("Section B — Who comes from other counties, and why")
    with ui.chart_card("Who they are", "Q2.6"):
        d = profile[profile.county != "Nandi"].sort_values("visits", ascending=False).head(10)
        base_f = float(kpi.pct_female.iloc[0]) if not kpi.empty else 57.0
        vmax, pmax = max(1, d.visits.max()), max(1, d.patients.max())
        rpps = d.visits / d.patients.clip(lower=1)
        inj = d.injury_visits * 100 / d.visits
        ncd = d.ncd_visits * 100 / d.visits
        rows = []
        for r in d.itertuples():
            rpp = r.visits / max(1, r.patients)
            ij, nc = r.injury_visits * 100 / r.visits, r.ncd_visits * 100 / r.visits
            rows.append([(r.county, "hl"),
                         (_fmt(r.visits), "", ui.heat(r.visits, 0, vmax, ui.BLUE)),
                         (_fmt(r.patients), "", ui.heat(r.patients, 0, pmax, ui.BLUE)),
                         (f"{rpp:.1f}", "", ui.heat(rpp, 1, max(2.5, rpps.max()), ui.AMBER)),
                         (f"{r.mean_age:.1f}", "", ui.heat(r.mean_age, 20, 55, ui.NEUTRAL)),
                         (f"{r.pct_female:.1f}%", "", ui.diverge(r.pct_female, base_f, 25)),
                         (f"{ij:.0f}%", "", ui.heat(ij, 0, max(10, inj.max()), ui.CRITICAL)),
                         (f"{nc:.0f}%", "", ui.heat(nc, 0, max(10, ncd.max()), ui.BLUE))])
        ui.html_table([("County", ""), ("Visits", ""), ("Patients", ""), ("Visits per patient", ""), ("Mean age", ""), ("% female", ""), ("% injury", ""), ("% NCD", "")], rows)
        st.caption(f"Shading: blue = more, amber = more repeat visits, red = more injury; % female is teal above the {base_f:.0f}% hospital baseline and amber below it.")
    ui.why_cards(_readings(profile))
    ui.insight_bar("<strong>These are readings, not proof.</strong> The data holds age, sex, repeat visits and diagnosis. It does not hold referral source, transport or the nearest "
                   "alternative facility. Ask the hospital if a referral-in field exists; if it does, it goes here.", "amber")

    # ---- C
    ui.section_header("Section C — Distance from the hospital, by disease group")
    geo.render_geo("rings", P, height=1040)
    dist = q.q2_other_county_distance()
    with ui.chart_card("Beyond Nandi: how far other-county patients travel, and for what", "N2d", new=True,
                       note="Bars = % of that county's visits. Distance is county centre to Kapsabet CRH."):
        d = dist[dist.visits >= 5].copy() if not dist.empty else dist
        if len(d):
            d["_km"] = d.km_from_hospital.fillna(1e9)
            d = d.sort_values("_km")  # counties without a located centre go last, labelled "distance n/a"
            lab = [f"{r.county} · {int(r.km_from_hospital)} km" if pd.notna(r.km_from_hospital) else f"{r.county} · distance n/a" for r in d.itertuples()]
            inj, acu, ncd = (list(d.injury_visits * 100 / d.visits), list(d.acute_illness_visits * 100 / d.visits), list(d.ncd_visits * 100 / d.visits))
            oth = [max(0.0, 100 - a - b - c) for a, b, c in zip(inj, acu, ncd)]  # every bar fills to 100%, so a county with no classified visits still shows
            ser = [("Injury", inj, ui.CRITICAL), ("Acute illness", acu, ui.AMBER), ("NCD", ncd, ui.BLUE), ("Other / not classified", oth, "#B8BDC7")]
            st.plotly_chart(ui.stacked_bar_chart(lab, ser, height=max(220, 36 * len(d) + 80), xsuffix="%"), use_container_width=True, config=ui.PC_CFG)
            d = d.dropna(subset=["km_from_hospital"])
            near, far = d[d.km_from_hospital < 100], d[d.km_from_hospital >= 100]
            p = lambda s, t: f"{s * 100 / t:.1f}%" if t else "n/a"  # noqa: E731
            st.markdown(
                f"**What changes with distance.** Counties within 100 km ({_fmt(near.visits.sum())} visits): NCD {p(near.ncd_visits.sum(), near.visits.sum())}, acute illness {p(near.acute_illness_visits.sum(), near.visits.sum())}, injury {p(near.injury_visits.sum(), near.visits.sum())}. "
                f"Counties beyond 100 km ({_fmt(far.visits.sum())} visits): NCD {p(far.ncd_visits.sum(), far.visits.sum())}, acute illness {p(far.acute_illness_visits.sum(), far.visits.sum())}, injury {p(far.injury_visits.sum(), far.visits.sum())}. "
                "Small numbers, so treat this as a pointer, not a finding.")
    ui.why_cards([("Convenience", "Short-reach groups look like walk-in primary care: most patients live within 2 km."),
                  ("Specialty or referral pull", "Long-reach groups suggest patients pass nearer facilities to reach this hospital."),
                  ("How we test it", "Compare age, sex and visits per patient by distance band. Older, repeat, far-away patients point to chronic-care pull.")])
    R = P["rings"]["byCat"]
    if "Injury & trauma" in R and "Infectious disease" in R and R["Infectious disease"]["median"]:
        ratio = R["Injury & trauma"]["median"] / R["Infectious disease"]["median"]
        ui.insight_bar(f"<strong>Reach differs by disease.</strong> Median distance is {R['Injury & trauma']['median']} km for injury and trauma against {R['Infectious disease']['median']} km for infectious disease "
                       f"({ratio:.1f}× further). If this holds on the real data, the hospital acts as a referral centre for those groups, not only a local clinic.")

    # ---- D
    ui.section_header("Section D — Disease burden by place")
    geo.render_geo("disease", P, height=820)
    ui.insight_bar("<strong>Read shading as share of each place's visits, not volume.</strong> Remote areas have small denominators after the 2025 export gap, so any place under "
                   f"{MIN_PLACE_VISITS} visits is greyed out and each tooltip shows the visit count.", "amber")

    # ---- E
    ui.section_header("Section E — Catchment shift, 2024 → 2025")
    ns = subs[subs.county == "Nandi"] if not subs.empty else subs
    if len(ns):
        ns = ns.assign(chg=(ns.v25 - ns.v24) * 100 / ns.v24.where(ns.v24 > 0))
        worst = ns.sort_values("chg").iloc[0]
        emg_row = ns[ns.n == "Emgwen"]
        st.markdown(
            f'<div style="background:#fff;border:1px solid {ui.BORDER};border-left:4px solid {ui.CRITICAL};border-radius:0 8px 8px 0;padding:16px 18px;margin-bottom:14px">'
            f'<div style="font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:{ui.TEXT_MUTED}">Critical data gap</div>'
            f'<div style="font-size:28px;font-weight:700;color:{ui.CRITICAL}">{worst.chg:.1f}%</div>'
            f'<div style="font-size:12px;margin-top:4px">{worst.n} visits fell from {_fmt(worst.v24)} in 2024 to {_fmt(worst.v25)} in 2025</div>'
            f'<div style="font-size:11px;color:{ui.TEXT_MUTED};margin-top:6px">Almost every non-Emgwen sub-county fell by a similar margin'
            + (f" while Emgwen moved {emg_row.chg.iloc[0]:+.1f}%" if len(emg_row) else "")
            + ". A simultaneous drop this size is an export failure, not a clinical signal.</div></div>", unsafe_allow_html=True)
    geo.render_geo("shift", P, height=1170)
    ui.insight_bar("⚠ <strong>Do not compare places across years</strong> until the non-Emgwen feeds are validated (the 2025 export gap). The Kapsabet locality itself grows because the "
                   "other localities stop reporting, which is why Emgwen's share looks like it rose.", "amber")

    # ---- F
    ui.section_header("Section F — Population profile")
    c1, c2 = st.columns(2)
    with c1:
        with ui.chart_card("Age–sex pyramid (visits)", "Q1.3"):
            pyr = q.q1_age_sex()
            if len(pyr):
                st.plotly_chart(ui.pyramid_chart(list(pyr.age_group), list(pyr.female), list(pyr.male), height=574), use_container_width=True, config=ui.PC_CFG)
    with c2:
        geo.render_geo("female", P, height=640)
    base = P["femaleBaseline"]
    ui.insight_bar(f"Women attend more than men in every age group above 13. <strong>Every sex split in the disease pages is compared with {base:.1f}%, not 50/50.</strong>")
    ui.synthesis("The hospital serves a hyper-local population, with a small but patterned inflow from neighbouring counties. Some disease groups reach further than others, which is the test "
                 "that separates a local clinic from a referral centre. The 2025 collapse in every other place is an export gap and is flagged on each year comparison. "
                 "Geo-confidence and the issue register now live on the <a href='?tab=tab_dataquality' target='_self' style='color:#2E8B3A;font-weight:700;text-decoration:none'>Data quality tab</a>.")
