"""Inventory Control — shrinkage, losses, dispensing without stock and prescription fulfilment."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import config as C, insights, ui


def render(ctx: dict) -> None:
    ctl, as_of = ctx["control"], ctx["as_of"]
    if ctl.empty:
        ui.empty()
        return
    c1, _ = st.columns([2, 3])
    months = c1.segmented_control("Period", [3, 6, 12], default=6, format_func=lambda m: f"Last {m} months",
                                  key="ctl_period") or 6
    last_full = (as_of.to_period("M") - 1).to_timestamp()
    start = last_full - pd.DateOffset(months=months - 1)
    prev_start = start - pd.DateOffset(months=months)
    cur = ctl[(ctl["month"] >= start) & (ctl["month"] <= last_full)]
    prev = ctl[(ctl["month"] >= prev_start) & (ctl["month"] < start)]

    def total(df, metric, col="value_kes"):
        return df.loc[df["metric"] == metric, col].sum()

    def shrink(df):
        return total(df, "Stock-count losses") - total(df, "Stock-count gains")

    def fill(df):
        return insights.fill_rate(total(df, insights.DISPENSED, "n_events"), total(df, insights.NOT_DISPENSED, "n_events"))

    def covered(metric_prefix: str) -> bool:
        """A previous-period comparison is only fair if the previous window has data in every month."""
        m = prev.loc[prev["metric"].str.startswith(metric_prefix) & (prev["n_events"].fillna(0) > 0), "month"]
        return m.nunique() >= months

    s_now, s_prev = shrink(cur), (shrink(prev) if covered("Stock-count") else None)
    f_now, f_prev = fill(cur), (fill(prev) if covered("Prescriptions") else float("nan"))
    raw_now = total(cur, "Stock-count net loss before re-sync netting")
    first_count = ctl.loc[(ctl["metric"] == "Stock-count losses") & (ctl["n_events"].fillna(0) > 0), "month"].min()
    no_cmp = f"no comparison: stock counts start {first_count:%b %Y}" if pd.notna(first_count) else "no stock counts recorded"
    store_view = bool(ctx["f"].stores)
    t_in, t_out = total(cur, "Transfers in", "qty"), total(cur, "Transfers out", "qty")
    ui.kpi_row([
        ui.kpi("Net shrinkage", ui.kes(s_now),
               f"previous {months} mo: {ui.kes(s_prev)}" if s_prev is not None else "no comparison yet",
               tip=("Stock counted below the system figure (losses − gains), at cost. "
                    + ("" if s_prev is not None else no_cmp[0].upper() + no_cmp[1:] + ".")),
               tone=("bad" if s_now > s_prev else "good") if s_prev is not None else "",
               accent="warning" if s_now > 0 else "good"),
        ui.kpi("Recorded losses", ui.kes(total(cur, "Losses (negative adjustments + write-offs)")),
               f"manual adjustments {ui.kes(total(cur, 'Manual adjustments (net, excl. re-sync reversals)'))}",
               tip="Write-offs and loss postings. Manual adjustments exclude staff undoing a system re-sync."),
        ui.kpi("Dispensed from negative stock", ui.num(total(cur, "Dispenses from negative stock", "n_events")),
               f"{ui.kes(total(cur, 'Dispenses from negative stock'))} at selling price",
               tone="warn" if total(cur, "Dispenses from negative stock", "n_events") else "",
               tip="Prescription lines dispensed when the system showed no stock (facility-wide)."),
        ui.kpi("Prescription fill rate", ui.pct(f_now),
               f"{ui.num(total(cur, insights.RX_OUT_OF_STOCK, 'n_events'))} lost to stockouts "
               f"({ui.kes(total(cur, insights.RX_OUT_OF_STOCK))})",
               tone="bad" if pd.notna(f_prev) and f_now < f_prev - 0.02 else "",
               tip="Dispensed ÷ (dispensed + not dispensed); cancelled and stopped prescriptions are left out. "
                   + (f"{(f_now - f_prev) * 100:+.1f} points vs the previous {months} months." if pd.notna(f_prev)
                      else "")),
        ui.kpi("Transfers in vs out", f"{ui.num(t_in)} / {ui.num(t_out)}", "units",
               tip="Selected stores: stores send and receive differently." if store_view
               else "Facility-wide, so in and out should balance."),
    ])
    st.caption(f"{start:%b %Y} – {last_full:%b %Y} · store-level figures (product filters don't apply here)")
    with st.expander("How these figures are calculated"):
        st.markdown(
            "- **Shrinkage** compares each stock count with the system figure before any system adjustment posted "
            "earlier that day (a system re-sync isn't a physical loss); system adjustments posted after the count are "
            f"ignored. Counting every count adjustment would give {ui.kes(raw_now)}.\n"
            "- **Manual adjustments** leave out staff undoing a system re-sync from the same month.\n"
            "- **Fill rate** = dispensed ÷ (dispensed + not dispensed); cancelled and stopped prescriptions are left out.\n"
            "- Values are at today's unit cost.")

    left, right = st.columns(2, gap="large")
    with left:
        ui.section("Net shrinkage by month", "KES, by store")
        sc = ctl[ctl["metric"].isin(["Stock-count losses", "Stock-count gains"]) & (ctl["month"] >= prev_start)
                 & (ctl["month"] <= last_full)].copy()
        sc["net"] = sc["value_kes"].where(sc["metric"] == "Stock-count losses", -sc["value_kes"])
        p = sc.groupby(["month", "store_name"])["net"].sum().unstack(fill_value=0)
        fig = go.Figure([go.Bar(x=p.index, y=p[c], name=c) for c in p.columns])
        fig.update_layout(barmode="relative", colorway=[C.PRIMARY, C.AMBER, "#60A5FA", C.PURPLE, C.RED, C.GREY, C.INK])
        ui.plot(ui.style(fig, 320))
    with right:
        ui.section("Prescription outcomes by month", "Bars: prescriptions by outcome · line: fill rate")
        rx = ctl[ctl["metric"].str.startswith("Prescriptions") & (ctl["month"] >= prev_start)
                 & (ctl["month"] <= last_full)]
        p = rx.groupby(["month", "metric"])["n_events"].sum().unstack(fill_value=0)
        colors = {"Prescriptions dispensed": C.PRIMARY_LIGHT, "Prescriptions not dispensed": C.AMBER,
                  "Prescriptions cancelled": C.GREY, "Prescriptions stopped": "#60A5FA"}
        fig = go.Figure([go.Bar(x=p.index, y=p[c], name=c.replace("Prescriptions ", "").capitalize(),
                                marker_color=colors.get(c)) for c in p.columns])
        if "Prescriptions dispensed" in p:
            fig.add_trace(go.Scatter(x=p.index, y=insights.fill_rate(p[insights.DISPENSED], p.get(
                insights.NOT_DISPENSED, 0 * p[insights.DISPENSED])), yaxis="y2",
                                     name="Fill rate", line=dict(color=C.INK)))
        fig.update_layout(barmode="stack", yaxis2=dict(overlaying="y", side="right", tickformat=".0%", range=[0, 1]))
        ui.plot(ui.style(fig, 320))

    ui.section("By store", "Selected period, KES at cost")
    tab = cur[cur["store_id"].notna()].pivot_table(index="store_name", columns="metric", values="value_kes",
                                                    aggfunc="sum", fill_value=0)
    keep = [c for c in ["Stock-count losses", "Stock-count gains", "Losses (negative adjustments + write-offs)",
                        "Manual adjustments (net, excl. re-sync reversals)"] if c in tab.columns]
    if keep:
        tab = tab[keep].assign(**{"Net shrinkage": tab.get("Stock-count losses", 0) - tab.get("Stock-count gains", 0)})
        tab = tab.sort_values("Net shrinkage", ascending=False).reset_index()
        st.dataframe(tab.round(0), hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(c, format="localized") for c in tab.columns
                                    if c != "store_name"} | {"store_name": "Store"})
        ui.download(tab, "ksh_v3_control_by_store")
