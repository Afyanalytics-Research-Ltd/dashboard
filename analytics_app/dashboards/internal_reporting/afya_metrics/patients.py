"""Patient counts tab: Patient Records and UPDs per source, filtered by when data reached the warehouse.

One period filter at the top drives everything below it: the two totals, the chart's shaded period and
the per-source table. Patient Records are blue and UPDs teal wherever they appear.
"""

from datetime import date, timedelta
from html import escape

import altair as alt
import pandas as pd
import streamlit as st

from . import theme as T
from .analysis import fmt, pct

MEASURES = ("PATIENT_RECORDS", "UPDS")
RECEIPT = {"PATIENT_RECORDS": "RECORDS", "UPDS": "UPDS"}
PERIODS = ("All time", "This quarter", "Last 30 days", "Custom")


def _span(a, b):
    return f"{a:%d %b} – {b:%d %b %Y}" if a.year == b.year else f"{a:%d %b %Y} – {b:%d %b %Y}"


def _period(receipts, updated):
    """Period picker plus the as-of line. Returns (start, end), or None for all time."""
    dated = [r["DATE"] for r in receipts if r["DATE"]]
    today = date.today()
    lo = min(dated) if dated else today
    with st.container(key="pc_filter"):
        c_pick, c_asof = st.columns([3, 1], vertical_alignment="bottom")
        row = c_pick.container(horizontal=True, vertical_alignment="bottom", gap="small")
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
        asof = f"Updated <b>{updated:%d %b %Y}</b>" if updated else ""
        c_asof.markdown(f'<div class="x-asof">{asof}</div>', unsafe_allow_html=True)
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


def render(rows, updated, receipts):
    if not rows:
        T.html('<div class="x-empty"><b>No patient counts yet</b>They appear after the next count.</div>')
        return

    rng = _period(receipts, updated)
    shown = _received(rows, receipts, rng) if rng else rows
    total = {m: sum(r[m] for r in shown) for m in MEASURES}
    when = f"received {_span(*rng)}" if rng else None

    T.html('<div class="x-grid two">'
           + _tile("Patient Records", T.REC, total["PATIENT_RECORDS"],
                   when or f"held in the warehouse, across {len(rows)} sources")
           + _tile('<span class="x-acr">UPDs</span>', T.UPD, total["UPDS"],
                   f"new unique patients, {when}" if when else "unique patients, counted within each source",
                   primary=True)
           + '</div>')

    _chart(receipts, rng)

    listed = [r for r in shown if r["PATIENT_RECORDS"] or r["UPDS"]] if rng else shown
    top = max([r["PATIENT_RECORDS"] for r in listed] or [1]) or 1
    body = "".join(
        f'<tr><td>{escape(r["SOURCE"])}</td><td class="x-num">{fmt(r["PATIENT_RECORDS"])}</td>'
        f'<td class="x-num">{fmt(r["UPDS"])}</td><td class="x-barcell"><div class="x-hbar">'
        f'<i style="width:{pct(r["PATIENT_RECORDS"], top):.1f}%;background:{T.REC};opacity:.35"></i>'
        f'<i style="width:{pct(r["UPDS"], top):.1f}%;background:{T.UPD}"></i></div></td></tr>'
        for r in sorted(listed, key=lambda r: -r["PATIENT_RECORDS"]))
    if not listed:
        body = '<tr><td colspan="4" class="x-dim">No data was received in this period.</td></tr>'
    T.html(f'<table class="x-table"><thead><tr><th>Source</th>'
           f'<th class="x-num"><i class="x-mk" style="background:{T.REC}"></i>Patient Records</th>'
           f'<th class="x-num"><i class="x-mk" style="background:{T.UPD}"></i><span class="x-acr">UPDs</span></th>'
           f'<th class="x-barcell"></th></tr></thead>'
           f'<tbody>{body}</tbody><tfoot><tr><td>Total</td><td class="x-num">{fmt(total["PATIENT_RECORDS"])}</td>'
           f'<td class="x-num">{fmt(total["UPDS"])}</td><td></td></tr></tfoot></table>')
    hidden = len(shown) - len(listed)
    if hidden:
        T.html(f'<div class="x-note">{hidden} other source{"s" if hidden != 1 else ""} received no data in this period.</div>')
