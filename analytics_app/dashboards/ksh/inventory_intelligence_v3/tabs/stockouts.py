"""Stockout Watch — what is out now, how often items run out, and how stockouts actually end."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import config as C, insights, ui


def render(ctx: dict) -> None:
    as_of = ctx["as_of"]
    so = ctx["stockouts"].copy()
    c1, c2 = st.columns([2, 2])
    demand_only = c1.toggle("Only items with recent demand", value=True,
                            help="Hide items that hit zero but are no longer used (not a service gap).")
    excl_sys = c2.toggle("Exclude stockouts caused by system adjustments", value=True,
                         help="Episodes that began on a system (bookkeeping) adjustment rather than real use.")
    demand = ctx["stock"][["store_id", "product_id", "adc_90d"]]
    so = so.merge(demand, on=["store_id", "product_id"], how="left")
    so["adc_90d"] = so["adc_90d"].fillna(0)
    if demand_only:
        so = so[so["adc_90d"] > 0]
    sys_count = int(so["started_on_system_adjustment"].sum())
    if excl_sys:
        so = so[~so["started_on_system_adjustment"]]
    if so.empty:
        ui.empty()
        return

    # 'Out now' uses the shared definition (also requires the current stock position to be <= 0).
    ongoing = insights.ongoing_stockouts(ctx, demand_only=demand_only, excl_system=excl_sys).copy()
    ongoing["units_unmet"] = ongoing["adc_90d"] * ongoing["days_out"]
    year = so[so["stockout_start_date"] > as_of - pd.Timedelta(days=365)]
    repeat = year.groupby(["store_id", "product_id"], as_index=False).agg(   # by ID: 40 names are shared
        store_name=("store_name", "first"), product_name=("product_name", "first"),
        clinical_tier=("clinical_tier", "first"), episodes=("episode_no", "count"),
        total_days_out=("days_out", "sum"), last_start=("stockout_start_date", "max"))
    repeat = repeat[repeat["episodes"] >= C.REPEAT_STOCKOUT_EPISODES].sort_values("episodes", ascending=False)
    ended = year[~year["is_ongoing"]]       # same 12-month window as the "How stockouts end" chart
    # Wards are restocked by transfer from the pharmacy/main store, so a transfer in is a physical restock too.
    physical = ended["restocked_by"].isin(["Receipt", "Transfer in"]).mean() if len(ended) else float("nan")

    av = ctx["availability"]
    bt = av["by_tier"]
    _, lost, lost_value = insights.fill_rate_window(ctx["control"], as_of, 6)
    ui.kpi_row([
        ui.kpi("Availability", ui.pct(av["rate"]),
               " · ".join(f"{t.lower()} {ui.pct(bt[t])}" for t in ("CRITICAL", "HIGH") if t in bt) or "last 90 days",
               tone="bad" if av["rate"] < 0.9 else "good", accent="warning" if av["rate"] < 0.9 else "good",
               tip=f"Share of days, over the last {C.AVAILABILITY_DAYS}, that the {ui.num(av['n_items'])} store items in "
                   f"use had stock on hand. {ui.pct(av['in_stock_now'])} are in stock today."),
        ui.kpi("Out now", ui.num(len(ongoing)),
               f"{ongoing['clinical_tier'].isin(['CRITICAL', 'HIGH']).sum()} high-priority",
               tone="bad" if len(ongoing) else "good", accent="critical" if len(ongoing) else "good"),
        ui.kpi("Median days out", ui.num(ongoing["days_out"].median()) if len(ongoing) else "—", "items out now"),
        ui.kpi("Repeat stockouts", ui.num(len(repeat)),
               f"out {C.REPEAT_STOCKOUT_EPISODES}+ times in 12 months", tone="warn" if len(repeat) else ""),
        ui.kpi("Lost to stockouts", ui.kes(lost_value) if not ctx.get("product_filtered") else "—",
               f"{ui.num(lost)} prescriptions, 6 months" if not ctx.get("product_filtered") else "not by product",
               tone="bad" if lost_value and not ctx.get("product_filtered") else "",
               tip="Prescriptions not dispensed because the item was out of stock in that store, valued at selling "
                   "price (a minimum: some lines have no price). Last 6 complete months; store-level, so product "
                   "filters don't apply."),
    ])

    left, right = st.columns([3, 2], gap="large")
    with left:
        t_av, t_new = st.tabs(["Availability by month", "New stockouts by month"])
        with t_av:
            mo = av["monthly"]
            if mo.empty:
                ui.empty("No availability history at the current filters.")
            else:
                colors = {"All items in use": C.PRIMARY, "Critical": C.RED, "High": C.AMBER}
                fig = go.Figure([go.Scatter(x=mo.index, y=mo[c], name=c, mode="lines+markers",
                                            line=dict(color=colors.get(c))) for c in mo.columns])
                fig.update_layout(yaxis=dict(tickformat=".0%", range=[max(0, mo.min().min() - 0.05), 1]))
                ui.plot(ui.style(fig, 280))
                st.caption(f"Share of days in stock, items in use · from {pd.Timestamp(C.RELIABLE_LEDGER_FROM):%b %Y} "
                           "(earlier stock records have gaps)")
        with t_new:
            last_full = (as_of.to_period("M") - 1).to_timestamp()       # the current month is incomplete
            y = year.assign(month=year["stockout_start_date"].dt.to_period("M").dt.to_timestamp())
            y = y[y["month"] <= last_full]
            t = y.groupby(["month", "product_class"]).size().unstack(fill_value=0)
            fig = go.Figure([go.Bar(x=t.index, y=t[c], name=c) for c in t.columns])
            fig.update_layout(barmode="stack", colorway=[C.PRIMARY, "#60A5FA", C.AMBER, C.PURPLE, C.GREY, C.RED])
            ui.plot(ui.style(fig, 280))
    with right:
        ui.section("How stockouts end", f"Last 12 months · {ui.pct(physical)} ended with a delivery or transfer (green)")
        e = year[~year["is_ongoing"]]["restocked_by"].value_counts()
        fig = go.Figure(go.Bar(x=e.values, y=e.index, orientation="h",
                               marker_color=[C.PRIMARY if i in ("Receipt", "Transfer in") else C.GREY
                                             for i in e.index]))
        ui.plot(ui.style(fig, 300, legend=False))

    ui.section("Out of stock now", "Most critical first, then longest out")
    ongoing = ongoing.sort_values(["clinical_tier", "days_out"], ascending=[True, False])
    cols = ["product_name", "store_name", "clinical_tier", "product_class", "stockout_start_date", "days_out",
            "adc_90d", "units_unmet", "episodes_for_item"]
    st.dataframe(ongoing[cols], hide_index=True, use_container_width=True, height=380, column_config={
        "product_name": "Product", "store_name": "Store", "clinical_tier": "Tier", "product_class": "Class",
        "stockout_start_date": st.column_config.DateColumn("Out since"),
        "days_out": st.column_config.NumberColumn("Days out", format="%d"),
        "adc_90d": st.column_config.NumberColumn("Use / day", format="%.2f"),
        "units_unmet": st.column_config.NumberColumn("Unmet units", format="%.0f",
                                                     help="Average daily use × days out: demand that couldn't be served."),
        "episodes_for_item": st.column_config.NumberColumn("Episodes (all time)", format="%d")})
    ui.download(ongoing[cols], "ksh_v3_out_of_stock")

    ui.section("Repeat stockouts", f"Out {C.REPEAT_STOCKOUT_EPISODES}+ times in 12 months: candidates for a higher reorder point")
    if repeat.empty:
        ui.empty("No repeat stockouts at the current filters.")
    else:
        st.dataframe(repeat.drop(columns=["store_id", "product_id"]), hide_index=True, use_container_width=True, column_config={
            "store_name": "Store", "product_name": "Product", "clinical_tier": "Tier",
            "episodes": st.column_config.NumberColumn("Episodes", format="%d"),
            "total_days_out": st.column_config.NumberColumn("Days out (total)", format="%d"),
            "last_start": st.column_config.DateColumn("Last stockout")})
