"""Today's Briefing — what needs attention now, in priority order."""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import ai, config as C, insights, ui
from . import orders


def render(ctx: dict) -> None:
    stock, as_of = ctx["stock"], ctx["as_of"]
    act = stock[stock["priority_action"] == "ORDER NOW"]
    at_risk = stock[stock["stock_status"].isin(insights.AT_RISK) & (stock["adc_90d"] > 0)]
    ongoing = insights.ongoing_stockouts(ctx)
    held = stock[stock["qty_on_hand"] > 0]
    tracked = held[~held["store_name"].isin(ctx.get("no_usage_stores", set()))]
    dead = tracked[tracked["last_outflow_date"].isna() | (tracked["days_since_last_outflow"] >= C.DEAD_DAYS)]
    cons = ctx["consumption"]
    chg = cons["last"] / cons["prev"] - 1 if cons["prev"] else float("nan")

    blind_value = held.loc[held["store_name"].isin(ctx.get("no_usage_stores", set())), "stock_value"].sum()
    tracked_value = tracked["stock_value"].sum()
    blind_names = ", ".join(sorted(ctx.get("no_usage_stores", set())))
    eq = ctx.get("equipment")
    eq_value = float(eq.loc[eq["qty_on_hand"] > 0, "stock_value"].sum()) if eq is not None and len(eq) else 0.0
    # Shortages: one card. The breakdown adds up to the headline (at zero or below + under 7 days + critical early).
    n_zero = int((act["qty_on_hand"] <= 0).sum())
    n_week = int(((act["qty_on_hand"] > 0) & (act["stock_status"] == "Critical")).sum())
    n_early = len(act) - n_zero - n_week
    n_crit, n_high = int((act["clinical_tier"] == "CRITICAL").sum()), int((act["clinical_tier"] == "HIGH").sum())
    plan, need = orders.order_plan(ctx)
    fill, lost, lost_value = insights.fill_rate_window(ctx["control"], as_of, 6)
    use_value = float((tracked["adc_90d"] * tracked["unit_cost"]).sum() +
                      (stock.loc[(stock["qty_on_hand"] <= 0) & ~stock["store_name"].isin(ctx.get("no_usage_stores", set())),
                                 "adc_90d"] * stock.loc[(stock["qty_on_hand"] <= 0) & ~stock["store_name"].isin(
                                     ctx.get("no_usage_stores", set())), "unit_cost"]).sum())
    doh = tracked_value / use_value if use_value else float("nan")
    ui.kpi_row([
        ui.kpi("Shortages to act on", ui.num(len(act)),
               f"{ui.num(len(ongoing))} out of stock · {n_crit} critical",
               tone="bad" if len(act) else "good", accent="critical" if len(act) else "good",
               tip=(f"Store items in use that need ordering now: {n_zero} at zero or below "
                    f"({ui.num(len(ongoing))} genuinely out of stock; the rest hit zero through a bookkeeping "
                    f"adjustment or show negative stock), {n_week} with under 7 days left"
                    + (f", and {n_early} critical drugs ordered early" if n_early else "")
                    + f". {n_crit} are critical-tier and {n_high} high-priority. The same drug in two stores counts twice.")),
        ui.kpi("Suggested order value", ui.kes(need["order_value"].sum()) if len(need) else "—",
               "to restock to 30 days' cover",
               tip=f"The full Order Workbench list ({ui.num(len(need))} products, including items at their reorder point "
                   "that are not yet urgent), at 95% service level and 30 days of cover, valued at unit cost."),
        ui.kpi("Prescription fill rate", ui.pct(fill), f"{ui.num(lost)} lost to stockouts ({ui.kes(lost_value)})",
               tone="bad" if fill == fill and fill < 0.8 else "",
               tip="Last 6 complete months. Dispensed ÷ (dispensed + not dispensed); cancelled and stopped "
                   "prescriptions are left out. Lost to stockouts = not dispensed while the item was out of stock; "
                   "value at selling price (a minimum: some lines have no price)."),
        # Headline = stores whose usage is recorded; stores with no recorded usage (e.g. the Laboratory, whose
        # quantities and pack costs don't reconcile to its ledger) are shown separately, never mixed in.
        ui.kpi("Stock value", ui.kes(tracked_value),
               f"{ui.pct(dead['stock_value'].sum() / tracked_value if tracked_value else 0)} idle {C.DEAD_DAYS}+ days",
               tip=(f"Medical stock at cost, in stores that record usage: about {ui.num(doh)} days of use at the current "
                    "rate. " + (f"A further {ui.kes(blind_value)} sits in {blind_names}, which record no usage, so it can't "
                                "be verified. " if blind_value else "")
                    + (f"Equipment recorded as stock ({ui.kes(eq_value)}) is listed separately on Dead & Excess."
                       if eq_value else ""))),
        ui.kpi("Consumption, 30 days", ui.num(cons["last"]),
               f"{ui.pct(chg, signed=True)} vs previous 30 days" if pd.notna(chg) else "",
               tone="warn" if pd.notna(chg) and abs(chg) >= 0.1 else "",
               tip="Units used (dispensed + ward use, net of returns); mixed units such as tablets, vials and ml."),
    ])

    facts = {"stock_as_of": f"{as_of:%d %b %Y}", "usage_to": f"{ctx['usage_end']:%d %b %Y}", "filters": ctx["f"].describe(
        dict(zip(ctx["stores"]["store_id"], ctx["stores"]["store_name"]))),
        "kpis": {"shortages_to_act_on": int(len(act)), "out_of_stock_now": int(len(ongoing)),
                 "under_7_days": n_week, "critical_tier_shortages": n_crit,
                 "suggested_order_value_kes": round(float(need["order_value"].sum())) if len(need) else 0,
                 "prescription_fill_rate_6m": round(float(fill), 3) if fill == fill else None,
                 "prescriptions_lost_to_stockouts_6m": int(lost),
                 "lost_to_stockouts_value_kes_6m": round(lost_value),
                 "days_of_stock_on_hand": round(doh) if doh == doh else None,
                 "stock_value_kes_stores_recording_usage": round(float(tracked_value)),
                 "stock_value_kes_stores_without_usage_records": round(float(blind_value)),
                 "idle_stock_value_kes": round(float(dead["stock_value"].sum()))},
        "insights": [{k: i[k] for k in ("severity", "title", "detail")} for i in ctx["insights"]]}
    text, is_ai = ai.briefing_summary(json.dumps(facts, default=str))
    text = " ".join(line.strip().lstrip("-•* ").strip() for line in text.replace("**", "").splitlines() if line.strip())
    st.markdown(f'<div class="ai-box"><div class="ai-tag">{"AI summary" if is_ai else "Summary"}</div>'
                f'<div class="ai-text">{ui.esc(text)}</div></div>', unsafe_allow_html=True)

    ui.section("Needs attention", "Most urgent first")
    if not ctx["insights"]:
        ui.empty("No exceptions at the current filters.")
    cols = st.columns(2, gap="medium")
    for n, i in enumerate(ctx["insights"]):
        with cols[n % 2]:
            st.markdown(ui.insight_card(i), unsafe_allow_html=True)
            st.button(f"Open {i['tab']} →", key=f"open_{n}", type="tertiary", on_click=ui.open_tab, args=(i["tab"],))

    ui.section("Do today", "Store shortages, most critical first. Some can be covered by a transfer instead of an order.")
    top = act.sort_values(["priority_score", "days_of_stock"], ascending=[False, True], na_position="first").head(15)
    if top.empty:
        ui.empty("No items need ordering now at the current filters.")
    else:
        rd = ctx["redistribution"]
        movable = set(zip(rd["product_name"], rd["to_store"])) if len(rd) else set()
        top = top.assign(days_of_stock=top["days_of_stock"].where(top["qty_on_hand"] > 0, 0.0),
                         why=top.apply(insights.why_flagged, axis=1),
                         transfer=[("Yes — from another store" if (p, s_) in movable else "")
                                   for p, s_ in zip(top["product_name"], top["store_name"])])
        show = top[["product_name", "store_name", "clinical_tier", "stock_status", "qty_on_hand", "days_of_stock",
                    "adc_90d", "transfer", "why"]]
        st.dataframe(show, hide_index=True, use_container_width=True, column_config={
            "product_name": st.column_config.TextColumn("Product", width="medium"), "store_name": "Store",
            "clinical_tier": "Tier", "stock_status": "Status",
            "qty_on_hand": st.column_config.NumberColumn("On hand", format="%d"),
            "days_of_stock": st.column_config.NumberColumn("Days left", format="%.1f"),
            "adc_90d": st.column_config.NumberColumn("Use / day", format="%.2f"),
            "transfer": "Transfer possible",
            "why": st.column_config.TextColumn("Why it's flagged", width="large")})
        st.button("Open the full order list →", key="open_orders", type="tertiary", on_click=ui.open_tab,
                  args=("Order Workbench",))

    ui.section("Stock health by store", "Number of items in each status")
    if len(stock):
        g = stock.groupby(["store_name", "stock_status"]).size().unstack(fill_value=0)
        g = g.reindex(columns=[s for s in C.STATUS_ORDER if s in g.columns])
        g = g.loc[g.sum(axis=1).sort_values().index]
        fig = go.Figure([go.Bar(y=g.index, x=g[s], name=s, orientation="h", marker_color=C.STATUS_COLORS[s])
                         for s in g.columns])
        fig.update_layout(barmode="stack")
        ui.plot(ui.style(fig, height=60 + 34 * len(g)))
