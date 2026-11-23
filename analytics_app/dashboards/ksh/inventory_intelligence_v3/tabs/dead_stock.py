"""Dead & excess stock — money tied up in stock that isn't moving (dead/slow) or that far exceeds use (excess)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import config as C, ui


def render(ctx: dict) -> None:
    c1, c2 = st.columns([2, 3])
    idle_days = c1.slider("Idle for at least (days)", 30, 365, C.DEAD_DAYS, step=15,
                          help="Days since the item last left that store (used, dispensed or transferred out).")
    blind = ctx.get("no_usage_stores", set())
    include_blind = c2.toggle(f"Include stores with no recorded usage ({', '.join(sorted(blind)) or 'none'})",
                              value=False, disabled=not blind,
                              help="These stores' consumption isn't captured in the inventory system, so all their "
                                   "stock appears idle. Excluded by default so the figures reflect real idleness.")
    held = ctx["stock"][ctx["stock"]["qty_on_hand"] > 0].copy()
    if not include_blind:
        held = held[~held["store_name"].isin(blind)]
    if held.empty:
        ui.empty()
        return
    held["idle_days"] = held["days_since_last_outflow"]       # used OR transferred out
    never = held["last_outflow_date"].isna()
    dead = held[never | (held["idle_days"] >= idle_days)].copy()
    slow = held[~never & (held["idle_days"] >= C.SLOW_DAYS) & (held["idle_days"] < idle_days)]

    # Where else is this product used? (best release route: transfer, not write-off)
    users = ctx["stock_all"][ctx["stock_all"]["adc_90d"] > 0]
    best = users.sort_values("adc_90d", ascending=False).drop_duplicates("product_id").set_index("product_id")
    dead["used_in"] = dead["product_id"].map(best["store_name"])
    dead["used_in_rate"] = dead["product_id"].map(best["adc_90d"])
    other = dead["used_in"].notna() & (dead["used_in"] != dead["store_name"])
    never = dead["last_outflow_date"].isna()                # aligned to `dead` from here on
    dead["recommended_action"] = np.select(
        [other, never | (dead["idle_days"] >= 180), dead["idle_days"] >= 90],
        ["Transfer to " + dead["used_in"].fillna(""), "Review: return to supplier or write off",
         "Stop reordering; offer to other stores"], "Reduce next order")
    dead["months_to_clear_elsewhere"] = np.where(other, dead["qty_on_hand"] / (dead["used_in_rate"] * 30), np.nan)
    # Only what the receiving store would use within 6 months is realistically releasable by transfer.
    dead["releasable_value"] = np.where(
        other, np.minimum(dead["qty_on_hand"], dead["used_in_rate"] * 182) * dead["unit_cost"], 0.0)
    dead["idle_label"] = np.where(never, "No movement out since go-live", dead["idle_days"].fillna(0).astype(int).astype(str) + " days")

    # Excess: still in use, but holding more than EXCESS_COVER_DAYS of cover (the part above that line is the excess).
    ex = held[(held["adc_90d"] > 0) & (held["days_of_stock"] > C.EXCESS_COVER_DAYS)].copy()
    ex["excess_qty"] = np.floor(ex["qty_on_hand"] - ex["adc_90d"] * C.EXCESS_COVER_DAYS)
    ex["excess_value"] = ex["excess_qty"] * ex["unit_cost"]
    # Days of stock on hand = stock value ÷ value used per day (at unit cost), by product class.
    all_use =ctx["stock"][ctx["stock"]["adc_90d"] > 0]
    if not include_blind:
        all_use = all_use[~all_use["store_name"].isin(blind)]
    use_val = (all_use["adc_90d"] * all_use["unit_cost"]).groupby(all_use["product_class"]).sum()
    doh = (held.groupby("product_class")["stock_value"].sum() / use_val.replace(0, np.nan)).dropna().sort_values()
    doh_all = held["stock_value"].sum() / use_val.sum() if use_val.sum() else float("nan")

    total = held["stock_value"].sum()
    ui.kpi_row([
        ui.kpi(f"Idle {idle_days}+ days", ui.kes(dead["stock_value"].sum()),
               f"{ui.num(len(dead))} items · {ui.pct(dead['stock_value'].sum() / total if total else 0)} of stock value",
               tone="bad", accent="warning",
               tip=f"{never.sum()} of these items have not moved since go-live; {dead['unit_cost'].isna().sum()} have no "
                   "cost, so the value is a minimum." + ("" if include_blind else " Stores that record usage only.")),
        ui.kpi(f"Slow ({C.SLOW_DAYS}–{idle_days} days)", ui.kes(slow["stock_value"].sum()), f"{len(slow)} items"),
        ui.kpi("Releasable by transfer", ui.kes(dead["releasable_value"].sum()),
               f"{other.sum()} items used elsewhere", tone="good",
               tip="Idle stock another store uses, capped at 6 months of that store's use."),
        ui.kpi("Excess stock", ui.kes(ex["excess_value"].sum()), f"{len(ex)} items in use, over {C.EXCESS_COVER_DAYS} days",
               tone="warn" if len(ex) else "",
               tip=f"Items still being used but holding more than {C.EXCESS_COVER_DAYS} days of stock; the value is the "
                   "part above that line. Reduce or pause orders rather than write off."),
        ui.kpi("Days of stock on hand", ui.num(doh_all),
               " · ".join(f"{k.split()[0].lower()} {ui.num(v)}" for k, v in doh.sort_values(ascending=False).head(2).items()),
               tip="Stock value ÷ value used per day (last 90 days, at unit cost): how long today's stock would last."),
    ])

    left, right = st.columns(2, gap="large")
    with left:
        t_store, t_class = st.tabs(["Idle value by store", "Idle value by therapeutic class"])
        with t_store:
            v = dead.groupby("store_name")["stock_value"].sum().sort_values()
            ui.plot(ui.style(go.Figure(go.Bar(x=v.values, y=v.index, orientation="h", marker_color=C.AMBER)),
                             60 + 32 * len(v), legend=False))
        with t_class:
            v = dead.groupby(ui.class_label(dead))["stock_value"].sum().nlargest(10).sort_values()
            ui.plot(ui.style(go.Figure(go.Bar(x=v.values, y=v.index, orientation="h", marker_color=C.GREY)),
                             60 + 32 * len(v), legend=False))
    with right:
        ui.section("Days of stock on hand, by product class", "How long today's stock would last at the current rate")
        if doh.empty:
            ui.empty("No items in use at the current filters.")
        else:
            fig = go.Figure(go.Bar(x=doh.values, y=doh.index, orientation="h", marker_color=C.PRIMARY,
                                   text=[f"{x:,.0f} d" for x in doh.values], textposition="outside"))
            fig.add_vline(x=C.EXCESS_COVER_DAYS, line_dash="dot", line_color=C.GREY)
            ui.plot(ui.style(fig, 60 + 40 * len(doh), legend=False))

    ui.section("Idle stock", "Highest value first, with a suggested action")
    cols = ["product_name", "store_name", "product_class", "qty_on_hand", "stock_value", "idle_label",
            "recommended_action", "months_to_clear_elsewhere", "clinical_tier"]
    st.dataframe(dead.sort_values("stock_value", ascending=False)[cols].round({"stock_value": 0}), hide_index=True,
                 use_container_width=True, height=440, column_config={
                     "product_name": "Product", "store_name": "Store", "product_class": "Class",
                     "qty_on_hand": st.column_config.NumberColumn("On hand", format="%d"),
                     "stock_value": st.column_config.NumberColumn("Value (KES)", format="localized"),
                     "idle_label": "Last moved out", "recommended_action": st.column_config.TextColumn("Recommended action", width="medium"),
                     "months_to_clear_elsewhere": st.column_config.NumberColumn("Months to use up there", format="%.1f"),
                     "clinical_tier": "Tier"})
    ui.download(dead[cols], "ksh_v3_idle_stock")

    eq = ctx.get("equipment")
    eq = eq[eq["qty_on_hand"] > 0] if eq is not None else eq
    if eq is not None and len(eq):
        with st.expander(f"Equipment recorded as stock: {ui.kes(eq['stock_value'].sum())} ({len(eq)} items), "
                         "not counted above"):
            st.caption("Durable devices and spare parts held in the HMIS as inventory (analyst-reviewed list). "
                       "Kept out of stock value, idle and excess; move them to the asset register at source.")
            st.dataframe(eq.sort_values("stock_value", ascending=False)[
                ["product_name", "store_name", "qty_on_hand", "unit_cost", "stock_value"]].round({"stock_value": 0}),
                hide_index=True, use_container_width=True, column_config={
                    "product_name": "Item", "store_name": "Store",
                    "qty_on_hand": st.column_config.NumberColumn("Qty", format="%d"),
                    "unit_cost": st.column_config.NumberColumn("Unit cost (KES)", format="localized"),
                    "stock_value": st.column_config.NumberColumn("Value (KES)", format="localized")})

    ui.section("Excess stock", f"In use, but more than {C.EXCESS_COVER_DAYS} days of cover · largest excess value first")
    if ex.empty:
        ui.empty("No excess stock at the current filters.")
    else:
        ecols = ["product_name", "store_name", "clinical_tier", "qty_on_hand", "adc_90d", "days_of_stock",
                 "excess_qty", "excess_value"]
        st.dataframe(ex.sort_values("excess_value", ascending=False)[ecols].round({"excess_value": 0}),
                     hide_index=True, use_container_width=True, height=360, column_config={
                         "product_name": st.column_config.TextColumn("Product", width="medium"), "store_name": "Store",
                         "clinical_tier": "Tier", "qty_on_hand": st.column_config.NumberColumn("On hand", format="%d"),
                         "adc_90d": st.column_config.NumberColumn("Use / day", format="%.2f"),
                         "days_of_stock": st.column_config.NumberColumn("Days of cover", format="%.0f"),
                         "excess_qty": st.column_config.NumberColumn(f"Above {C.EXCESS_COVER_DAYS} days", format="%d"),
                         "excess_value": st.column_config.NumberColumn("Excess value (KES)", format="localized")})
        ui.download(ex[ecols], "ksh_v3_excess_stock")
