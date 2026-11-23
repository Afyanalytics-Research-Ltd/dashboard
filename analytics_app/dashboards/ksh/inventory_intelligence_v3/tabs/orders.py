"""Order Workbench — what to order, how much, and what can be moved instead of bought."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import config as C, engines, ui


def _items(ctx) -> pd.DataFrame:
    s = ctx["stock"]
    if s.empty:
        return s
    # Clip each store's negative balance BEFORE summing: one store's negative must not cancel another's real stock.
    s = s.assign(qty_on_hand=s["qty_on_hand"].clip(lower=0))
    g = s.groupby("product_id", as_index=False).agg(
        product_name=("product_name", "first"), product_class=("product_class", "first"),
        therapeutic_class=("therapeutic_class", "first"), therapeutic_subclass=("therapeutic_subclass", "first"),
        clinical_tier=("clinical_tier", "first"), unit_std=("unit_std", "first"),
        qty_on_hand=("qty_on_hand", "sum"), adc_90d=("adc_90d", "sum"), unit_cost=("unit_cost", "median"),
        stores=("store_name", lambda x: ", ".join(sorted(set(x)))))
    g = g.merge(ctx["demand"][["product_id", "ewm_daily", "std_daily_90d", "trend", "demand_type", "confidence"]],
                on="product_id", how="left")
    g = g.merge(ctx["seasonal"][["product_id", "seasonal_multiplier", "calendar_disease"]], on="product_id", how="left")
    lt_per, lt_fac = ctx["lead_times"]
    return engines.attach_lead_time(g, lt_per, lt_fac)


def order_plan(ctx: dict, level: str = "95%", cover: int = C.TARGET_COVER_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(plan, need): the replenishment plan for items in use, and the items that need an order now. Shared by the
    Order Workbench and the Briefing so their figures always agree."""
    items = _items(ctx)
    if items.empty:
        return items, items
    plan = engines.replenishment(items, C.SERVICE_LEVEL_Z[level or "95%"], cover)
    plan = plan[plan["planning_daily_demand"] > 0]
    need = plan[plan["at_or_below_rop"] | (plan["suggested_order_qty"] > 0) & (plan["days_of_stock"] < C.DOS_LOW)]
    return plan, need


def render(ctx: dict) -> None:
    c1, c2, c3 = st.columns([1.2, 1.2, 2])
    level = c1.segmented_control("Service level", list(C.SERVICE_LEVEL_Z), default="95%", key="sl",
                                 help="Probability of not running out during the lead time. Higher = more safety stock.")
    cover = c2.slider("Days of cover to order for", 14, 60, C.TARGET_COVER_DAYS, step=7)
    scope = c3.segmented_control("Show", ["Needs ordering", "All items with demand"], default="Needs ordering",
                                 key="order_scope")
    detail = c3.toggle("Show planning detail", value=False, key="order_detail",
                       help="Adds safety stock, reorder point, lead-time basis, seasonality, trend and forecast confidence.")
    plan, need = order_plan(ctx, level, cover)
    if plan.empty:
        ui.empty("No items with recorded demand at the current filters, so there is nothing to plan.")
        return
    view = need if scope != "All items with demand" else plan
    _, lt_fac = ctx["lead_times"]

    ui.kpi_row([
        ui.kpi("At or below reorder point", ui.num(plan["at_or_below_rop"].sum()),
               f"of {len(plan)} items in use", tone="bad" if plan["at_or_below_rop"].any() else "good"),
        ui.kpi("Will run out before delivery", ui.num(plan["stockout_before_delivery"].sum()),
               "stock lasts less than the lead time", tone="bad" if plan["stockout_before_delivery"].any() else "good"),
        ui.kpi("Suggested order value", ui.kes(need["order_value"].sum()),
               f"{need['order_value'].isna().sum()} items have no cost" if need["order_value"].isna().any() else "at unit cost",
               tip="Items needing an order, valued at unit cost."),
        ui.kpi("Critical/high-tier to order", ui.num(need["clinical_tier"].isin(["CRITICAL", "HIGH"]).sum()),
               tone="warn" if need["clinical_tier"].isin(["CRITICAL", "HIGH"]).any() else ""),
        ui.kpi("Typical lead time", f"{lt_fac['median']:.0f} days",
               f"items capped at {C.LEAD_TIME_CAP_X_MEDIAN * lt_fac['median']:.0f} days",
               tip=f"Median days from running low to the next delivery, over {lt_fac['n']:,} events in the selected "
                   "stores."),
    ])

    ui.section("Order list", "Most urgent first: items that will run out before a delivery could arrive")
    view = view.sort_values(["stockout_before_delivery", "clinical_tier", "days_of_stock"],
                            ascending=[False, True, True])
    all_cols = ["product_name", "clinical_tier", "product_class", "qty_on_hand", "planning_daily_demand",
                "days_of_stock", "lead_time_days", "lead_time_source", "safety_stock", "reorder_point",
                "suggested_order_qty", "order_value", "seasonal_multiplier", "calendar_disease", "trend", "confidence",
                "stores"]
    core = ["product_name", "clinical_tier", "qty_on_hand", "planning_daily_demand", "days_of_stock",
            "suggested_order_qty", "order_value", "lead_time_days", "stores"]
    cols = all_cols if detail else core
    st.dataframe(view[cols].round({"order_value": 0}), hide_index=True, use_container_width=True, height=440,
                 column_config={
        "product_name": st.column_config.TextColumn("Product", width="medium"), "clinical_tier": "Tier",
        "product_class": "Class", "qty_on_hand": st.column_config.NumberColumn("On hand", format="%d"),
        "planning_daily_demand": st.column_config.NumberColumn("Demand / day", format="%.2f"),
        "days_of_stock": st.column_config.ProgressColumn("Days left", format="%.0f", min_value=0, max_value=60),
        "lead_time_days": st.column_config.NumberColumn("Lead time (d)", format="%.0f"),
        "lead_time_source": "Lead-time basis",
        "safety_stock": st.column_config.NumberColumn("Safety stock", format="%.0f"),
        "reorder_point": st.column_config.NumberColumn("Reorder point", format="%.0f"),
        "suggested_order_qty": st.column_config.NumberColumn("Order qty", format="%d"),
        "order_value": st.column_config.NumberColumn("Order value (KES)", format="localized"),
        "seasonal_multiplier": st.column_config.NumberColumn("Seasonal ×", format="%.2f"),
        "calendar_disease": "Seasonal driver", "trend": "Trend", "confidence": "Forecast confidence",
        "stores": "Held in"})
    ui.download(view[all_cols], "ksh_v3_order_list", "Download order list")

    left, right = st.columns(2, gap="large")
    with left:
        ui.section("Order value by therapeutic class")
        v = need.groupby(ui.class_label(need))["order_value"].sum().sort_values()
        if len(v) > 12:                      # keep the bars summing to the KPI: fold the tail into "Other"
            v = pd.concat([pd.Series({"Other classes": v.iloc[:-11].sum()}), v.iloc[-11:]]).sort_values()
        if v.sum() > 0:
            ui.plot(ui.style(go.Figure(go.Bar(x=v.values, y=v.index, orientation="h", marker_color=C.PRIMARY)),
                             height=60 + 28 * len(v), legend=False))
        else:
            ui.empty("No priced items to order.")
    with right:
        rd = ctx["redistribution"]
        ui.section("Move before you buy", "Shortages another store can cover from its surplus")
        if rd.empty:
            ui.empty("No transfer opportunities at the current filters.")
        else:
            st.dataframe(rd.round({"transfer_value": 0}), hide_index=True, use_container_width=True, height=300,
                         column_config={
                "product_id": None, "product_class": None, "product_name": "Product", "clinical_tier": "Tier",
                "from_store": "From", "from_on_hand": st.column_config.NumberColumn("From has", format="%d"),
                "from_days_of_stock": st.column_config.NumberColumn("From days", format="%.0f"),
                "to_store": "To", "to_status": "To status",
                "to_days_of_stock": st.column_config.NumberColumn("To days", format="%.1f"),
                "transfer_qty": st.column_config.NumberColumn("Move qty", format="%d"),
                "transfer_value": st.column_config.NumberColumn("Value (KES)", format="localized"),
                "days_cover_after": st.column_config.NumberColumn("Cover after (d)", format="%.0f")})
            ui.download(rd, "ksh_v3_transfers", "Download transfer list")

    with st.expander("How these numbers are calculated"):
        st.markdown(f"""
- **Demand / day**: exponentially-weighted daily consumption (adaptive span {C.EWM_SPAN_MIN}–{C.EWM_SPAN_MAX} days),
  adjusted for the 30-day trend (±15% threshold) and the **seasonal multiplier** (Kisumu disease calendar blended
  with this facility's own month-of-year pattern for the item's therapeutic subclass). Falls back to the 90-day average.
- **Lead time**: days from a low-stock point to the next **recorded receipt** for the item (median of at least
  {C.MIN_LT_OBSERVATIONS} observations), else the facility median ({lt_fac['median']:.0f} days from {lt_fac['n']} events).
  V3 has no purchase orders, so this includes reaction time. An item's lead time is capped at
  {C.LEAD_TIME_CAP_X_MEDIAN:.0f}× the typical wait ({C.LEAD_TIME_CAP_X_MEDIAN * lt_fac['median']:.0f} days) and its
  variability at the facility's typical spread ({lt_fac['std']:.0f} days): items waiting longer are restocked only every
  few months, which reflects reordering habits, not supplier speed.
- **Safety stock** = z × √(LT·σ²demand + demand²·σ²LT), z for the chosen service level. **Reorder point** = demand × LT + safety stock.
- Negative system stock is treated as zero available. Values use unit cost (store cost, else the item's median cost).
""")
