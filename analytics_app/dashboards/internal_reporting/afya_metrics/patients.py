"""Patient counts tab: Patient Records and UPDs per source, filtered by when data reached the warehouse.

One period filter at the top drives everything below it: the two totals, the chart's shaded period and
the per-source table. Patient Records are blue and UPDs teal wherever they appear.
"""

from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

from . import hmis as H
from . import theme as T
from .analysis import fmt

MEASURES = ("PATIENT_RECORDS", "UPDS")
RECEIPT = {"PATIENT_RECORDS": "RECORDS", "UPDS": "UPDS"}
PERIODS = ("All time", "This quarter", "Last 30 days", "Custom")


def _span(a, b):
    return f"{a:%d %b} – {b:%d %b %Y}" if a.year == b.year else f"{a:%d %b %Y} – {b:%d %b %Y}"


def _hmis_status(parts, error):
    """Where the HMIS figures come from: each environment's latest capture, and whether it has gone quiet."""
    if error is not None:
        return f"HMIS unavailable: {error.__class__.__name__}"
    out = []
    for label, (view, _) in parts.items():
        when = f"{view['captured']:%d %b %Y, %H:%M}" if hasattr(view["captured"], "strftime") else view["captured"]
        stale = f" <b class='x-flag-t'>(no new run for over {H.STALE_HOURS}h)</b>" if view["stale"] else ""
        out.append(f"{label} captured <b>{when}</b>{stale}")
    return " · ".join(out)


def _period(receipts, updated, hmis_line=""):
    """Period picker and the as-of line. Returns (start, end), or None for all time."""
    dated = [r["DATE"] for r in receipts if r["DATE"]]
    today = date.today()
    lo = min(dated) if dated else today
    with st.container(key="pc_filter"):
        row = st.container(horizontal=True, vertical_alignment="bottom", gap="small")
        choice = row.segmented_control("Data received", PERIODS, default="All time", key="pc_period",
                                       disabled=not dated) or "All time"
        rng = None
        if choice == "This quarter":
            rng = (date(today.year, 3 * ((today.month - 1) // 3) + 1, 1), today)
        elif choice == "Last 30 days":
            rng = (today - timedelta(days=29), today)
        elif choice == "Custom":
            with row.container(key="pc_custom", width=250):
                picked = st.date_input("Between", value=(lo, today), min_value=lo, max_value=today,
                                       format="DD/MM/YYYY", label_visibility="collapsed")
            picked = tuple(picked) if isinstance(picked, (tuple, list)) else (picked,)
            rng = (picked[0], picked[-1]) if picked else (lo, today)
    asof = " · ".join(x for x in (f"Warehouse updated <b>{updated:%d %b %Y}</b>" if updated else "", hmis_line) if x)
    T.html(f'<div class="x-asof pc-asof"><span>{asof}</span></div>')
    return rng


def _received(rows, receipts, rng):
    """Each source's figures counting only what arrived in the period (undated records always count)."""
    got = {r["SOURCE"]: dict.fromkeys(MEASURES, 0) for r in rows}
    for x in receipts:
        if x["SOURCE"] in got and (not x["DATE"] or rng[0] <= x["DATE"] <= rng[1]):
            for m in MEASURES:
                got[x["SOURCE"]][m] += x[RECEIPT[m]]
    return [{**r, **got[r["SOURCE"]]} for r in rows]


def _tile(label, colour, value, meta, primary=False):
    return T.tile(f'<i class="x-mk" style="background:{colour}"></i>{label}', fmt(value), meta=meta, primary=primary)


def _chart(receipts, rng):
    """Running total of Patient Records by the date they reached the warehouse."""
    dated = pd.DataFrame([x for x in receipts if x["DATE"]], columns=["DATE", "RECORDS"])
    if dated.empty:
        return
    daily = dated.groupby("DATE", as_index=False)["RECORDS"].sum().sort_values("DATE")
    daily["TOTAL"] = daily["RECORDS"].cumsum()
    end = max(date.today(), daily["DATE"].iloc[-1])
    if daily["DATE"].iloc[-1] < end:   # carry the line to today
        daily = pd.concat([daily, pd.DataFrame({"DATE": [end], "RECORDS": [0], "TOTAL": [daily["TOTAL"].iloc[-1]]})])
    daily["DATE"] = pd.to_datetime(daily["DATE"])
    first, last = daily["DATE"].iloc[0], daily["DATE"].iloc[-1]

    months = pd.date_range(first.to_period("M").to_timestamp(), last, freq="MS")
    step = max(1, -(-len(months) // 8))   # at most ~8 month labels
    ticks = [alt.DateTime(year=m.year, month=m.month, date=1) for m in months[::step]]
    axis = dict(labelColor=T.MUTED, labelFont=T.FONT, labelFontSize=11)
    x = alt.X("DATE:T", title=None, scale=alt.Scale(padding=6),
              axis=alt.Axis(values=ticks, format="%b %Y", grid=False, domainColor=T.BORDER, tickColor=T.BORDER, **axis))
    y = alt.Y("TOTAL:Q", title=None, scale=alt.Scale(zero=True),
              axis=alt.Axis(format="~s", tickCount=4, gridColor=T.DIVIDE, domain=False, ticks=False, **axis))
    hover = alt.selection_point(nearest=True, on="pointerover", fields=["DATE"], empty=False, clear="pointerout")
    base = alt.Chart(daily).encode(x=x)
    shaded = rng and (pd.Timestamp(rng[0]) > first or pd.Timestamp(rng[1]) < last)
    layers = []
    if shaded:
        layers.append(alt.Chart(pd.DataFrame({"a": [pd.Timestamp(rng[0])], "b": [pd.Timestamp(rng[1])]}))
                      .mark_rect(color=T.REC, opacity=0.08).encode(x="a:T", x2="b:T"))
    layers += [
        base.mark_line(interpolate="step-after", strokeWidth=2, color=T.REC).encode(y=y),
        alt.Chart(daily.tail(1)).mark_text(   # current total, labelled at the line's end
            align="right", dy=-12, font=T.FONT, fontSize=12, fontWeight=700, color=T.INK).encode(
            x=x, y="TOTAL:Q", text=alt.Text("TOTAL:Q", format=",")),
        base.mark_rule(color=T.SOFT).encode(opacity=alt.condition(hover, alt.value(1), alt.value(0))),
        base.mark_point(size=80, filled=True, color=T.REC, stroke="white", strokeWidth=2).encode(
            y="TOTAL:Q", opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=[alt.Tooltip("DATE:T", title="Date received", format="%d %b %Y"),
                     alt.Tooltip("RECORDS:Q", title="Records received that day", format=","),
                     alt.Tooltip("TOTAL:Q", title="Total so far", format=",")]).add_params(hover),
    ]
    note = f" · shaded: {_span(*rng)}" if shaded else ""
    with st.container(key="pc_chart"):
        T.html(f'<div class="x-chart-hd"><div><div class="x-chart-t">'
               f'<i class="x-mk" style="background:{T.REC}"></i>Patient Records received</div>'
               f'<div class="x-sub">Running total by the date data reached the warehouse{note}</div></div></div>')
        st.altair_chart(alt.layer(*layers).properties(height=220, background="transparent",
                                                     padding={"left": 0, "right": 4, "top": 18, "bottom": 0})
                        .configure_view(strokeWidth=0), width="stretch", theme=None)




def _filters(env_options):
    """Search (narrows the table only) and the Environment filter (picks what feeds the table and every total;
    read before the totals in afya_internal_metrics)."""
    with st.container(key="pc_tbl_filters"):
        c_q, c_env = st.columns([2.2, 2], vertical_alignment="bottom")
        q = c_q.text_input("Search sources", key="pc_q", placeholder="Type part of a source or facility name")
        c_env.multiselect("Environment", env_options, key=H.ENV_KEY, placeholder="Choose environments")
    return q


def _sources_table(listed, q):
    """The sources as a grid: click a column header to sort; search narrows it, and the total line below follows
    what is shown."""
    picked = [r for r in listed if not q or q.strip().lower() in r["SOURCE"].lower()]
    if not picked:
        T.html('<div class="x-empty" style="padding:28px 24px;margin-top:6px">No sources match.</div>')
        return
    top = max(max(r["UPDS"] for r in picked), 1)
    flagged = any(r.get("FLAGS") for r in picked)
    data = [{"Source": r["SOURCE"], "Environment": r.get("ENV") or H.WAREHOUSE_ENV,
             "Patient Records": None if r.get("HMIS") and not r.get("VISITS_POSTED") else r["PATIENT_RECORDS"],
             "UPDs": r["UPDS"], **({"Flags": "; ".join(r.get("FLAGS", []))} if flagged else {})} for r in picked]
    st.dataframe(data, hide_index=True, width="stretch", height=min(38 + 35 * len(data), 520), key="pc_tbl",
                 column_config={
                     "Source": st.column_config.TextColumn("Source", width="large"),
                     "Environment": st.column_config.TextColumn("Environment", width="small"),
                     "Patient Records": st.column_config.NumberColumn("Patient Records", format="localized"),
                     "UPDs": st.column_config.ProgressColumn("UPDs", format="localized", min_value=0, max_value=top),
                 })
    recs = sum(r["PATIENT_RECORDS"] for r in picked)
    upds = sum(r["UPDS"] for r in picked)
    T.html(f'<div class="x-tbl-total"><span>Total · {len(picked)} source{"s" if len(picked) != 1 else ""}'
           f'{" shown" if len(picked) != len(listed) else ""}</span>'
           f'<span><i class="x-mk" style="background:{T.REC}"></i>{fmt(recs)} Patient Records</span>'
           f'<span><i class="x-mk" style="background:{T.UPD}"></i>{fmt(upds)} <span class="x-acr">UPDs</span></span>'
           f'</div>')


def render(rows, updated, receipts, hmis_parts=None, hmis_error=None, env_options=(H.WAREHOUSE_ENV,), overlap=()):
    hmis_parts = hmis_parts or {}
    hmis_line = _hmis_status(hmis_parts, hmis_error)
    if not rows:
        _period(receipts, updated, hmis_line)
        _filters(list(env_options))
        T.html('<div class="x-empty"><b>No patient counts yet</b>They appear after the next count.</div>')
        return

    rng = _period(receipts, updated, hmis_line)
    shown = _received(rows, receipts, rng) if rng else rows
    total = {m: sum(r[m] for r in shown) for m in MEASURES}
    when = f"received {_span(*rng)}" if rng else None
    n_hmis = sum(1 for r in rows if r.get("HMIS"))
    n_wh = len(rows) - n_hmis
    upd_meta = "unique patients, counted within each source"
    if n_hmis:
        upd_meta = f"patients across {len(rows)} sources, {n_hmis} of them from the HMIS"

    T.html('<div class="x-grid two">'
           + _tile("Patient Records", T.REC, total["PATIENT_RECORDS"],
                   when or (f"visit records across {len(rows)} sources, {n_hmis} of them from the HMIS" if n_hmis
                            else f"held in the warehouse, across {n_wh} sources"))
           + _tile('<span class="x-acr">UPDs</span>', T.UPD, total["UPDS"],
                   f"new unique patients, {when}" if when else upd_meta,
                   primary=True)
           + '</div>')

    _chart(receipts, rng)

    q = _filters(list(env_options))
    if overlap:
        T.html(f'<div class="x-note" style="margin:0 2px 8px"><b class="x-flag-t">{" and ".join(overlap)} are both '
               f'selected:</b> they can hold the same facilities, so the totals may count them twice.</div>')
    listed = [r for r in shown if r["PATIENT_RECORDS"] or r["UPDS"]] if rng else shown
    listed = sorted(listed, key=lambda r: (not r.get("FLAGS"), -max(r["PATIENT_RECORDS"], r["UPDS"])))
    if listed:
        _sources_table(listed, q)
    else:
        T.html('<div class="x-empty" style="padding:28px 24px">No data was received in this period.</div>')
    hidden = len(shown) - len(listed)
    if hidden:
        T.html(f'<div class="x-note">{hidden} other source{"s" if hidden != 1 else ""} received no data in this period.</div>')
    dropped = sum(view["dropped"] for view, _ in hmis_parts.values())
    if dropped:
        T.html(f'<div class="x-note">{dropped} HMIS rows skipped (another counting method, or zero patients).</div>')
