"""
views_workload.py: Tab 9, Clinician Workload.

No SQL here: every number comes from queries.q9_* (workload_queries.sql: Q9.1, Q9.3, Q9.4, Q9.6, Q9.7, Q9.8, Q9.9).
Source table HOSPITALS.NANDI_CLEAN.CLINICIANS_WORKLOAD covers January to August 2024 only. One duplicate rollup row
(it equalled every other clinician combined) is excluded inside every query.
Sections: A contacts by month · B how the workload is spread · C unusually heavy load · D zero and part-year load.

Q9.2 (inflated monthly trend), Q9.4b/c/d (duplicate-row proof) and Q9.5 (fence, already returned by Q9.4) are not run here.
The duplicate row and the 37 clinicians with no contacts belong on the Data Quality tab.
Q9.10 (clinician contacts vs OPD visits by month) is not built: the OPD table has no month breakdown.
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
MONTHS = ["jan_2024", "feb_2024", "mar_2024", "apr_2024", "may_2024", "jun_2024", "jul_2024", "aug_2024"]
MLAB = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"]
FULL = {"Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April", "May": "May", "Jun": "June", "Jul": "July", "Aug": "August"}

_CSS = f"""
<style>
.wk-kp3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:0 0 12px; }}
@media(max-width:1100px) {{ .wk-kp3 {{ grid-template-columns:1fr; }} }}
.wk-lg {{ display:flex; gap:14px; flex-wrap:wrap; font-size:11px; color:{MUTED}; margin-top:8px; }}
.wk-lg i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; }}
.wk-cols {{ display:flex; gap:10px; align-items:flex-end; }}
.wk-col {{ flex:1; min-width:0; text-align:center; }}
.wk-col .h {{ display:flex; flex-direction:column; justify-content:flex-end; align-items:center; }}
.wk-col .h b {{ font-size:12px; }}
.wk-col .h i {{ display:block; width:78%; max-width:70px; border-radius:3px 3px 0 0; }}
.wk-col .m {{ font-size:12px; margin-top:5px; }}
.wk-col .s {{ font-size:11px; color:{MUTED}; }}
.wk-band {{ display:grid; grid-template-columns:130px 1fr 240px; gap:10px; align-items:center; padding:7px 0; font-size:12.5px; }}
@media(max-width:900px) {{ .wk-band {{ grid-template-columns:90px 1fr; }} .wk-band .t {{ grid-column:1 / -1; }} }}
.wk-band .b {{ height:22px; background:#F0F1F5; border-radius:0 6px 6px 0; }}
.wk-band .b i {{ display:block; height:100%; border-radius:0 4px 4px 0; }}
.wk-band .t b {{ font-weight:700; }}
.wk-band .t span {{ color:{MUTED}; font-size:11.5px; }}
.wk-top {{ display:grid; grid-template-columns:78px 1fr 52px; gap:10px; align-items:center; padding:8px 0; font-size:12px; }}
.wk-top .id {{ font-family:ui-monospace,monospace; font-size:11.5px; }}
.wk-top .b {{ position:relative; height:16px; }}
.wk-top .b i {{ display:block; height:100%; border-radius:0 4px 4px 0; }}
.wk-top .b u {{ position:absolute; top:-4px; bottom:-4px; border-left:2px dashed {RED}; text-decoration:none; }}
.wk-top .v {{ text-align:right; font-weight:600; }}
.wk-mini {{ display:grid; grid-template-columns:72px 1fr; gap:8px; align-items:end; padding:6px 0; border-bottom:1px solid {ui.BORDER}; }}
.wk-mini:last-of-type {{ border-bottom:0; }}
.wk-mini .id {{ font-size:12px; font-weight:600; }}
.wk-mini .id div {{ font-size:11px; color:{MUTED}; font-weight:400; }}
.wk-mini .c {{ display:flex; gap:3px; align-items:flex-end; height:46px; }}
.wk-mini .c i {{ flex:1; display:block; border-radius:2px 2px 0 0; background:{BLUE}; }}
.wk-ends {{ display:flex; justify-content:space-between; font-size:11px; color:{MUTED}; margin-top:6px; }}
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


def _truthy(v) -> bool:
    return v is True or str(v).strip().upper() == "TRUE"


def _r(a: float, b: float) -> int:
    return 0 if not a else round((b - a) / a * 100)


def _legend(items: list[tuple[str, str]]) -> str:
    return '<div class="wk-lg">' + "".join(f'<span><i style="background:{c}"></i>{n}</span>' for n, c in items) + "</div>"


def _kpi_html(cards: list[tuple]) -> str:
    return "".join(f'<div class="afya-kpi" style="border-top-color:{col}"><div class="afya-kpi-label">{a}</div><div class="afya-kpi-value">{b}</div><div class="afya-kpi-sub">{c}</div></div>'
                   for a, b, c, col in cards)


def _load() -> dict[str, pd.DataFrame]:
    names = ["summary", "top", "concentration", "monthly_outliers", "monthly", "bands", "months_active"]
    return {n: _num(getattr(q, "q9_" + n)().copy()) for n in names}


class _Data:
    def __init__(self, D: dict[str, pd.DataFrame]):
        s = D["summary"].iloc[0] if not D["summary"].empty else None
        g = (lambda c: _f(s[c])) if s is not None else (lambda c: 0.0)
        self.total, self.clin, self.active, self.zero = g("total_contacts"), g("total_clinicians"), g("active_clinicians"), g("zero_load_clinicians")
        self.mean, self.median = g("mean_contacts_active"), g("median_contacts_active")
        m = D["monthly"].sort_values("month_order")
        self.m = [(str(r.month_name)[:3], _f(r.total_contacts), _f(r.active_clinicians), _f(r.contacts_per_active_clinician)) for r in m.itertuples()]
        self.bands = [(r.contact_band, _f(r.clinicians), _f(r.contacts), _f(r.pct_of_contacts)) for r in D["bands"].sort_values("band_order").itertuples()]
        c = D["concentration"]
        self.top10 = next((_f(r.pct_of_total) for r in c.itertuples() if str(r.group_label).startswith("Top 10")), 0.0)
        self.top10_n = next((_f(r.contacts) for r in c.itertuples() if str(r.group_label).startswith("Top 10")), 0.0)
        t = D["top"].sort_values("total_contacts", ascending=False)
        self.fence = _f(t["upper_fence"].iloc[0]) if not t.empty else 0.0
        self.top = [(str(r.clinician_id)[:8], _f(r.total_contacts), _truthy(r.is_statistical_outlier)) for r in t.head(10).itertuples()]
        self.flag = [(str(r["clinician_id"])[:8], [_f(r[k]) for k in MONTHS], _f(r["total_contacts"])) for _, r in t.iterrows() if _truthy(r["is_statistical_outlier"])]
        self.mo = D["monthly_outliers"]
        self.ma = [(int(r.months_active), _f(r.clinicians), _f(r.contacts)) for r in D["months_active"].sort_values("months_active").itertuples()]


# ============================================================ KPIs
def _kpis(S: _Data) -> None:
    cards = [
        ("Clinician contacts, Jan to Aug 2024", _n(S.total), "8 months, duplicate row excluded", BLUE),
        ("Clinicians", _n(S.clin), f"{_n(S.active)} active · {_n(S.zero)} ({100 * S.zero / max(S.clin, 1):.0f}%) with no contacts", AMBER),
        ("Median contacts per active clinician", _n(S.median), f"over 8 months · the mean is {_n(S.mean)}, pulled up by a few busy clinicians", BLUE),
    ]
    st.markdown(f'<div class="wk-kp3">{_kpi_html(cards)}</div>', unsafe_allow_html=True)


def _top_insight(S: _Data) -> None:
    big = next(((c, p) for n, c, _, p in S.bands if n.startswith('"1,000') or n.startswith("1,000")), None)
    tail = f" and {_n(big[0])} clinicians ({100 * big[0] / max(S.clin, 1):.0f}% of the roster) carry {big[1]:.0f}% of all contacts" if big else ""
    ui.insight_bar(f"<strong>{100 * S.zero / max(S.clin, 1):.0f}% of clinicians recorded no contacts{tail}.</strong> This facility does not yet have a complete view of its own clinical workload.", "amber")


# ============================================================ A · contacts by month
def _month_cols(S: _Data) -> str:
    if not S.m:
        return ""
    mx = max(v for _, v, _, _ in S.m) or 1
    lo = min(range(len(S.m)), key=lambda i: S.m[i][1])
    hi = max(range(len(S.m)), key=lambda i: S.m[i][1])
    h = 190
    out = ""
    for i, (lab, v, a, p) in enumerate(S.m):
        col = AMBER if i == lo else (BLUE if i == hi else LIGHT)
        out += (f'<div class="wk-col" title="{lab} 2024: {_n(v)} contacts, {_n(a)} active clinicians"><div class="h" style="height:{h + 22}px"><b>{_n(v)}</b>'
                f'<i style="height:{v / mx * h:.0f}px;background:{col}"></i></div><div class="m">{lab}</div><div class="s">{_n(a)}</div></div>')
    return f'<div class="wk-cols">{out}</div><div class="wk-lg">Row under each month: clinicians with at least one contact</div>'


def _per_line(S: _Data) -> str:
    if not S.m:
        return ""
    W, H, pl, pr, pt, pb = 420, 250, 36, 14, 28, 36
    hi = max(140.0, max(p for *_, p in S.m) * 1.1)
    lo = min(range(len(S.m)), key=lambda i: S.m[i][1])
    pts = [(pl + i * (W - pl - pr) / (len(S.m) - 1), H - pb - (H - pb - pt) * p / hi) for i, (_, _, _, p) in enumerate(S.m)]
    s = f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Contacts per active clinician by month">'
    for g in (0, 50, 100):
        y = H - pb - (H - pb - pt) * g / hi
        s += f'<line x1="{pl}" x2="{W - pr}" y1="{y:.1f}" y2="{y:.1f}" stroke="#E5E7EB"/><text x="{pl - 6}" y="{y + 4:.1f}" text-anchor="end" font-size="10.5" fill="#6B7280">{g}</text>'
    s += '<polyline fill="none" stroke="#0C447C" stroke-width="2.2" points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '"/>'
    for i, (x, y) in enumerate(pts):
        lab, v, a, p = S.m[i]
        s += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{5 if i == lo else 3.5}" fill="{AMBER if i == lo else BLUE}"><title>{lab}: {p:.0f} contacts per active clinician</title></circle>'
        if i in (0, lo, len(S.m) - 2):
            s += f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-size="11.5" font-weight="600">{p:.0f}</text>'
        s += f'<text x="{x:.1f}" y="{H - pb + 18}" text-anchor="middle" font-size="11.5">{lab}</text>'
    return s + "</svg>"


def _section_a(S: _Data) -> None:
    ui.section_header("Section A — Contacts by month")
    l, r = st.columns([3, 2])
    with l:
        with ui.chart_card("Contacts by month, 2024", note="Clinician contacts per month. The lowest month is highlighted, the peak is dark blue. Hover for active clinicians."):
            st.markdown(_month_cols(S), unsafe_allow_html=True)
    with r:
        with ui.chart_card("Contacts per active clinician", note="Contacts divided by clinicians with at least one contact that month."):
            st.markdown(_per_line(S), unsafe_allow_html=True)
    if len(S.m) < 3:
        return
    lo = min(range(len(S.m)), key=lambda i: S.m[i][1])
    hi = max(range(len(S.m)), key=lambda i: S.m[i][1])
    if lo == 0:
        return
    pl_, cl = S.m[lo - 1], S.m[lo]
    drop, d_act, d_per = -_r(pl_[1], cl[1]), _r(pl_[2], cl[2]), _r(pl_[3], cl[3])
    why = ("so the drop is mostly lower activity per clinician, not just absences" if abs(d_per) > abs(d_act) else "so the drop is mostly fewer clinicians recording work")
    ui.insight_bar(
        f"<strong>{FULL.get(cl[0], cl[0])} is the month that suffered: {_n(cl[1])} contacts, {drop}% below {FULL.get(pl_[0], pl_[0])}.</strong> "
        f"Clinicians recording work went from {_n(pl_[2])} to {_n(cl[2])} and each recorded {_n(cl[3])} contacts, down from {_n(pl_[3])}, {why}. "
        f"{FULL.get(S.m[hi][0], S.m[hi][0])} is the peak at {_n(S.m[hi][1])} with {_n(S.m[hi][2])} clinicians active.", "blue")


# ============================================================ B · spread
def _bands_html(S: _Data) -> str:
    mx = max([c for _, c, _, _ in S.bands] + [1])
    out = ""
    for n, c, _, p in S.bands:
        col = GREY if n.startswith("0") else (AMBER if "1,000" in n else BLUE)
        out += (f'<div class="wk-band"><span>{n}</span><div class="b"><i style="width:{100 * c / mx:.1f}%;background:{col}"></i></div>'
                f'<div class="t"><b>{_n(c)}</b> clinicians <span>· {p:.1f}% of contacts</span></div></div>')
    return out


def _section_b(S: _Data) -> None:
    ui.section_header("Section B — How the workload is spread")
    with ui.chart_card("Clinicians by 8-month contact band", note="Bar = clinicians in the band. Text = that band's share of all contacts."):
        st.markdown(_bands_html(S), unsafe_allow_html=True)
    big = next(((c, p) for n, c, _, p in S.bands if "1,000" in n), None)
    small = [(c, p) for n, c, _, p in S.bands if n.startswith("0") or n.startswith("1-99")]
    if big and small:
        sc, sp = sum(c for c, _ in small), sum(p for _, p in small)
        ui.insight_bar(f"<strong>The 10 busiest clinicians hold {S.top10:.0f}% of all contacts, and the {_n(big[0])} with 1,000 or more hold {big[1]:.0f}%.</strong> "
                       f"The {_n(sc)} clinicians with under 100 contacts (including the {_n(S.zero)} with none) together hold {'under 3' if sp < 3 else f'{sp:.0f}'}%.", "amber")


# ============================================================ C · heavy load
def _top_html(S: _Data) -> str:
    if not S.top:
        return ""
    mx = max(v for _, v, _ in S.top) * 1.06
    out = ""
    for cid, v, flagged in S.top:
        out += (f'<div class="wk-top" title="{cid}: {_n(v)} contacts"><span class="id">{cid}</span><div class="b"><i style="width:{100 * v / mx:.1f}%;background:{AMBER if flagged else LIGHT}"></i>'
                f'<u style="left:{100 * S.fence / mx:.1f}%"></u></div><span class="v">{_n(v)}</span></div>')
    return out + _legend([("Above the statistical fence", AMBER), (f"Fence = {_n(S.fence)} contacts", RED), ("Within the fence", LIGHT)])


def _minis(S: _Data) -> str:
    mx = max([max(v) for _, v, _ in S.flag] + [1])
    out = ""
    for cid, v, tot in S.flag:
        bars = "".join(f'<i style="height:{max(100 * x / mx, 2):.0f}%" title="{MLAB[i]}: {_n(x)}"></i>' for i, x in enumerate(v))
        out += f'<div class="wk-mini"><div class="id">{cid}<div>{_n(tot)} in 8 months</div></div><div class="c">{bars}</div></div>'
    return out + '<div class="wk-ends"><span>Jan</span><span>Aug (same scale for all)</span></div>'


def _pattern_counts(S: _Data) -> tuple[int, int, int]:
    """(flagged clinicians topping the monthly fence in 4+ months, other single-month spikers, of those how many fade to almost nothing)."""
    if S.mo.empty:
        return 0, 0, 0
    ids = {c for c, _, _ in S.flag}
    per = S.mo.drop_duplicates("clinician_id")
    sustained = sum(1 for r in per.itertuples() if str(r.clinician_id)[:8] in ids and _f(r.months_flagged) >= 4)
    singles = [r.clinician_id for r in per.itertuples() if str(r.pattern) == "Single month" and str(r.clinician_id)[:8] not in ids]
    fade = 0
    for cid in singles:
        c = S.mo[S.mo.clinician_id == cid].sort_values("month_order")["contacts"].astype(float).tolist()
        if len(c) >= 8 and max(c) > 0 and c[6] <= 0.25 * max(c) and c[7] <= 0.25 * max(c):
            fade += 1
    return sustained, len(singles), fade


def _section_c(S: _Data) -> None:
    ui.section_header("Section C — Clinicians carrying an unusually heavy load")
    l, r = st.columns([3, 2])
    with l:
        with ui.chart_card("Top 10 clinicians by total contacts", note="Contacts, Jan to Aug 2024. Clinician IDs shortened to 8 characters."):
            st.markdown(_top_html(S), unsafe_allow_html=True)
    with r:
        with ui.chart_card(f"The {len(S.flag)} above the fence, month by month", note="Contacts per month. Same scale for all."):
            st.markdown(_minis(S), unsafe_allow_html=True)
    if not S.flag:
        return
    tots = [t for _, _, t in S.flag]
    sus, singles, fade = _pattern_counts(S)
    msg = (f"<strong>{len(S.flag)} clinicians sit above the statistical fence ({_n(S.fence)}), at {_n(min(tots))} to {_n(max(tots))} contacts each.</strong> "
           f"{sus} of them top the monthly fence in 4 or more months, so this is a sustained load worth a staffing conversation. ")
    if singles:
        msg += f"{singles} other clinicians top the monthly fence in just one month"
        msg += f", and {fade} of those drop to almost nothing afterwards, which looks like people leaving or being reassigned." if fade else "."
    ui.insight_bar(msg, "amber")


# ============================================================ D · zero and part-year load
def _active_cols(S: _Data) -> str:
    if not S.ma:
        return ""
    mx = max(c for _, c, _ in S.ma) or 1
    h = 150
    out = ""
    for m, c, _ in S.ma:
        col = GREY if m == 0 else (AMBER if m <= 5 else BLUE)
        out += (f'<div class="wk-col" title="{_n(c)} clinicians active in {m} of 8 months"><div class="h" style="height:{h + 22}px"><b>{_n(c)}</b>'
                f'<i style="height:{max(c / mx * h, 2):.0f}px;background:{col}"></i></div><div class="m">{m}</div></div>')
    return f'<div class="wk-cols">{out}</div><div class="wk-ends" style="justify-content:center">Months active, out of 8</div>' + _legend([("Never active", GREY), ("1 to 5 months", AMBER), ("6 to 8 months", BLUE)])


def _section_d(S: _Data) -> None:
    ui.section_header("Section D — Zero and part-year load")
    with ui.chart_card("Clinicians by number of months active", note=f"Months with at least one contact, out of 8. {_n(S.clin)} clinicians."):
        st.markdown(_active_cols(S), unsafe_allow_html=True)
    full = next(((c, k) for m, c, k in S.ma if m == 8), None)
    part = sum(c for m, c, _ in S.ma if 1 <= m <= 5)
    if full:
        ui.insight_bar(f"<strong>Only {_n(full[0])} clinicians were active in all 8 months, yet they recorded {100 * full[1] / max(S.total, 1):.0f}% of all contacts.</strong> "
                       f"Another {_n(part)} were active for 1 to 5 months, and {_n(S.zero)} never recorded a contact. The data cannot say whether these are new hires, leave, or contacts attributed to someone else.", "blue")


def render() -> None:
    ui.inject_css()
    st.markdown(vi._CSS, unsafe_allow_html=True)
    st.markdown(_CSS, unsafe_allow_html=True)
    if q.DEMO:
        ui.demo_banner()
    _top = st.container()
    with ui.loading("Workload"):
        S = _Data(_load())
    with _top:
        ui.page_header("Clinician Workload", "How many patient contacts clinicians recorded, and how evenly the work is spread. January to August 2024 only.", icon="workload", stat=(_n(S.total), "clinician contacts"))
    st.markdown('<div class="afya-link" style="margin-bottom:12px"><span>The workload table covers January to August 2024 only. One duplicate rollup row that equalled every other clinician combined was removed from every number on this page.</span>'
                '<a href="?tab=tab_dataquality" target="_self">Data quality →</a></div>', unsafe_allow_html=True)
    _kpis(S)
    _top_insight(S)
    _section_a(S)
    _section_b(S)
    _section_c(S)
    _section_d(S)
