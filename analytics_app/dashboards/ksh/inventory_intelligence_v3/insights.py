"""Deterministic, decision-oriented insights. Every insight is computed from the data with stated rules;
the AI layer only rephrases these facts, it never invents numbers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

AT_RISK = ("Critical", "Stocked out", "Negative")


def redistribution(stock: pd.DataFrame, exclude_donors: set | None = None) -> pd.DataFrame:
    """Same product: one active store short (critical / out, with demand), another holding surplus.
    Suggest moving enough to give the short store 30 days while the donor keeps 60 days of its own use.
    A donor's spare stock is allocated once (largest need first), and stores whose usage is not recorded
    (their 'surplus' can't be verified) never donate."""
    s = stock.copy()
    s["avail"] = s["qty_on_hand"].clip(lower=0)
    recv = s[s["stock_status"].isin(AT_RISK) & (s["adc_90d"] > 0)].copy()
    recv["need"] = np.ceil(recv["adc_90d"] * C.DOS_LOW - recv["avail"])
    don = s[(s["avail"] > 0) & ((s["days_of_stock"] > C.REDISTRIBUTE_SURPLUS_DOS) |
                                (s["stock_status"] == "No recent demand"))].copy()
    if exclude_donors:
        don = don[~don["store_name"].isin(exclude_donors)]
    don["spare"] = np.floor(don["avail"] - don["adc_90d"] * C.REDISTRIBUTE_KEEP_DAYS)
    don = don[don["spare"] > 0]
    if recv.empty or don.empty:
        return pd.DataFrame()
    pairs = recv.merge(don, on="product_id", suffixes=("_to", "_from"))
    pairs = pairs[pairs["store_id_to"] != pairs["store_id_from"]]
    if pairs.empty:
        return pd.DataFrame()
    # 'need' exists only on the receiver side and 'spare' only on the donor side, so they carry no suffix.
    # One donor per receiver (the one with most spare); then allocate each donor's spare cumulatively.
    pairs = pairs.sort_values("spare", ascending=False).drop_duplicates(["product_id", "store_id_to"])
    pairs = pairs.sort_values(["product_id", "store_id_from", "need"], ascending=[True, True, False])
    given_before = pairs.groupby(["product_id", "store_id_from"])["need"].cumsum() - pairs["need"]
    pairs["transfer_qty"] = np.minimum(pairs["need"], (pairs["spare"] - given_before).clip(lower=0))
    pairs = pairs[pairs["transfer_qty"] > 0]
    pairs["transfer_value"] = pairs["transfer_qty"] * pairs["unit_cost_from"]
    pairs["days_cover_after"] = (pairs["avail_to"] + pairs["transfer_qty"]) / pairs["adc_90d_to"]
    return pairs.rename(columns={
        "product_name_to": "product_name", "clinical_tier_to": "clinical_tier",
        "product_class_to": "product_class", "store_name_from": "from_store", "store_name_to": "to_store",
        "days_of_stock_to": "to_days_of_stock", "stock_status_to": "to_status", "avail_from": "from_on_hand",
        "days_of_stock_from": "from_days_of_stock"})[[
        "product_id", "product_name", "product_class", "clinical_tier", "from_store", "from_on_hand",
        "from_days_of_stock", "to_store", "to_status", "to_days_of_stock", "transfer_qty", "transfer_value",
        "days_cover_after"]].sort_values(["clinical_tier", "transfer_value"], ascending=[True, False])


DISPENSED, NOT_DISPENSED = "Prescriptions dispensed", "Prescriptions not dispensed"
RX_OUT_OF_STOCK = "Unfilled prescriptions: item out of stock"


def fill_rate(dispensed, not_dispensed):
    """THE fill-rate definition: dispensed ÷ (dispensed + not dispensed). Cancelled and
    stopped prescriptions are outside the rate: the pharmacy was never asked to fill them. Works on numbers or Series."""
    den = dispensed + not_dispensed
    return dispensed / den.replace(0, float("nan")) if hasattr(den, "replace") else (dispensed / den if den else float("nan"))


def fill_rate_window(ctl: pd.DataFrame, as_of: pd.Timestamp, months: int = 6) -> tuple[float, float, float]:
    """(fill rate, prescriptions lost to stockouts, their value at selling price) over the last `months` complete
    months — the Inventory Control tab's default window, so every tab shows the same figure."""
    last_full = (as_of.to_period("M") - 1).to_timestamp()
    w = ctl[(ctl["month"] >= last_full - pd.DateOffset(months=months - 1)) & (ctl["month"] <= last_full)]
    n = w.groupby("metric")["n_events"].sum()
    lost_value = float(w.loc[w["metric"] == RX_OUT_OF_STOCK, "value_kes"].sum())
    return fill_rate(n.get(DISPENSED, 0.0), n.get(NOT_DISPENSED, 0.0)), float(n.get(RX_OUT_OF_STOCK, 0.0)), lost_value


def ongoing_stockouts(ctx: dict, demand_only: bool = True, excl_system: bool = True) -> pd.DataFrame:
    """THE definition of 'out of stock now', shared by Briefing, Stockout Watch and the insights: an open
    episode at a store that (a) still has zero/negative stock in the current stock position, (b) uses the item
    (demand_only) and (c) did not start on a bookkeeping (system) adjustment (excl_system)."""
    so = ctx["stockouts"]
    st_ = ctx["stock"][["store_id", "product_id", "adc_90d", "qty_on_hand"]]
    o = so[so["is_ongoing"]].merge(st_, on=["store_id", "product_id"], how="left")
    o["adc_90d"] = o["adc_90d"].fillna(0)
    o = o[o["qty_on_hand"].fillna(0) <= 0]        # ledger says out AND the current stock position agrees
    if demand_only:
        o = o[o["adc_90d"] > 0]
    if excl_system:
        o = o[~o["started_on_system_adjustment"]]
    return o


def _kes(x: float) -> str:
    return f"KES {x / 1e6:,.1f}M" if abs(x) >= 1e6 else f"KES {x:,.0f}"


def _names(df: pd.DataFrame, n: int = 3) -> str:
    names = df["product_name"].dropna().astype(str).str.title().head(n).tolist()
    return ", ".join(names) + ("…" if len(df) > n else "")


def build(ctx: dict) -> list[dict]:
    """ctx keys: stock, stockouts, control, demand, anomalies, seasonal, redistribution, as_of, consumption."""
    out: list[dict] = []
    stock, as_of = ctx["stock"], ctx["as_of"]
    demand_items = stock[stock["adc_90d"] > 0]

    # 1. Clinically important items at risk
    hi = demand_items[demand_items["clinical_tier"].isin(["CRITICAL", "HIGH"]) &
                      demand_items["stock_status"].isin(AT_RISK)].sort_values(
        ["clinical_tier", "days_of_stock"], na_position="first")
    if len(hi):
        crit = (hi["clinical_tier"] == "CRITICAL").sum()
        out.append(dict(severity="critical", tab="Order Workbench", metric=len(hi),
                        title=f"{len(hi)} high-priority medicines are out or below {C.DOS_CRITICAL} days",
                        detail=f"{crit} critical-tier. Most urgent: {_names(hi)}."))

    # 2. New stockouts
    ongoing = ongoing_stockouts(ctx)
    new = ongoing[ongoing["stockout_start_date"] > as_of - pd.Timedelta(days=C.NEW_STOCKOUT_DAYS)]
    if len(new):
        out.append(dict(severity="critical" if (new["clinical_tier"] != "STANDARD").any() else "warning",
                        tab="Stockout Watch", metric=len(new),
                        title=f"{len(new)} items ran out in the last {C.NEW_STOCKOUT_DAYS} days",
                        detail=f"Still out as of {as_of:%d %b}. Includes {_names(new.sort_values('clinical_tier'))}."))

    # 3. Demand spikes on thin stock
    an = ctx["anomalies"]
    if len(an):
        sp = an[(an["direction"] == "Spike") & (an["change_pct"] >= 0.5)].merge(
            stock.groupby("product_id", as_index=False).agg(product_name=("product_name", "first"),
                                                             qty=("qty_on_hand", "sum"), adc=("adc_90d", "sum")),
            on="product_id")
        sp = sp[(sp["qty"] / sp["recent_daily"].replace(0, np.nan)) < C.DOS_LOW]
        if len(sp):
            out.append(dict(severity="warning", tab="Demand Insights", metric=len(sp),
                            title=f"{len(sp)} items show a demand spike while stock is thin",
                            detail=f"Use in the last {C.ANOMALY_RECENT_DAYS} days is at least 50% above normal "
                                   f"and covers < {C.DOS_LOW} days at the new rate: {_names(sp)}."))

    # 4. Seasonal outlook
    se = ctx["seasonal"]
    if len(se):
        hot = se[se["calendar_disease"] != ""].merge(
            demand_items.groupby("product_id", as_index=False).agg(
                product_name=("product_name", "first"), qty=("qty_on_hand", "sum"), adc=("adc_90d", "sum")),
            on="product_id")
        if len(hot):
            hot["dos_seasonal"] = hot["qty"].clip(lower=0) / (hot["adc"] * hot["seasonal_multiplier"])
            short = hot[hot["dos_seasonal"] < C.DOS_LOW]
            diseases = ", ".join(sorted(hot["calendar_disease"].unique()))
            out.append(dict(severity="warning" if len(short) else "info", tab="Demand Insights", metric=len(short),
                            title=f"Seasonal peak ahead: {diseases}",
                            detail=(f"{len(short)} affected items will have < {C.DOS_LOW} days of cover at "
                                    f"peak-season demand: {_names(short)}." if len(short)
                                    else "Affected items have enough cover for the expected uplift.")))

    # 5. Redistribution
    rd = ctx["redistribution"]
    if len(rd):
        out.append(dict(severity="info", tab="Order Workbench", metric=len(rd),
                        title=f"{len(rd)} shortages can be covered by moving stock between stores",
                        detail=f"Worth KES {rd['transfer_value'].sum():,.0f} at cost, no purchase needed. "
                               f"E.g. {rd.iloc[0]['product_name'].title()}: {rd.iloc[0]['from_store']} → "
                               f"{rd.iloc[0]['to_store']} ({rd.iloc[0]['transfer_qty']:,.0f} units)."))

    # 6. Dead stock (stores that record usage only — elsewhere every item looks idle)
    no_usage = ctx.get("no_usage_stores", set())
    held_all = stock[stock["qty_on_hand"] > 0]
    held = held_all[~held_all["store_name"].isin(no_usage)]
    dead = held[held["last_outflow_date"].isna() | (held["days_since_last_outflow"] >= C.DEAD_DAYS)]
    total_val, dead_val = held["stock_value"].sum(), dead["stock_value"].sum()
    if dead_val > 0:
        top10 = dead.nlargest(10, "stock_value")["stock_value"].sum()
        out.append(dict(severity="warning", tab="Dead & Excess", metric=dead_val,
                        title=f"KES {dead_val:,.0f} tied up in stock idle for {C.DEAD_DAYS}+ days",
                        detail=f"{dead_val / total_val:.0%} of stock value in stores that record usage; the top 10 "
                               f"items are {top10 / dead_val:.0%} of it — a short list to act on."))
    blind = held_all[held_all["store_name"].isin(no_usage)]
    if blind["stock_value"].sum() > 0:
        out.append(dict(severity="info", tab="Dead & Excess", metric=blind["stock_value"].sum(),
                        title=f"{', '.join(sorted(blind['store_name'].unique()))}: "
                              f"{_kes(blind['stock_value'].sum())} of stock with no usage recorded",
                        detail="These stores record no usage or transfers out in the inventory system, so their stock "
                               "can't be tracked for days-of-stock or idleness — a data-capture gap to raise."))

    # 7. Stock-count shrinkage trend (last 3 full months vs the 3 before)
    ctl = ctx["control"]
    if len(ctl) and not ctx.get("product_filtered"):   # control figures are store-level; product filters can't apply
        last_full = (as_of.to_period("M") - 1).to_timestamp()
        def window(metric, start, end):
            w = ctl[(ctl["metric"] == metric) & (ctl["month"] > start) & (ctl["month"] <= end)]
            return w["value_kes"].sum()
        a, b = last_full - pd.DateOffset(months=3), last_full - pd.DateOffset(months=6)
        net_now = window("Stock-count losses", a, last_full) - window("Stock-count gains", a, last_full)
        net_before = window("Stock-count losses", b, a) - window("Stock-count gains", b, a)
        if net_now > 0:
            trend = (f"up from KES {net_before:,.0f}" if net_now > net_before else f"down from KES {net_before:,.0f}")
            out.append(dict(severity="warning" if net_now > net_before else "info", tab="Inventory Control",
                            metric=net_now, title=f"Net stock-count shrinkage KES {net_now:,.0f} in the last 3 months",
                            detail=f"Counted stock below the system figure, {trend} in the previous 3 months."))

        rx = ctl[ctl["metric"].str.startswith("Prescriptions")]
        if len(rx):
            m = rx.groupby(["month", "metric"])["n_events"].sum().unstack(fill_value=0)
            m = m[m.index <= last_full]
            if len(m) >= 2 and "Prescriptions dispensed" in m:
                rate = fill_rate(m["Prescriptions dispensed"], m.get(NOT_DISPENSED, 0 * m["Prescriptions dispensed"]))
                cur, prev = rate.iloc[-1], rate.iloc[-2]
                if abs(cur - prev) >= 0.03:
                    out.append(dict(severity="warning" if cur < prev else "positive", tab="Inventory Control",
                                    metric=cur, title=f"Prescription fill rate {cur:.0%} in {m.index[-1]:%b %Y}",
                                    detail=f"{'Down' if cur < prev else 'Up'} from {prev:.0%} the month before."))

    # 8. Consumption momentum
    cons = ctx["consumption"]
    if cons["prev"] > 0:
        chg = cons["last"] / cons["prev"] - 1
        if abs(chg) >= 0.10:
            out.append(dict(severity="info", tab="Demand Insights", metric=chg,
                            title=f"Consumption {'up' if chg > 0 else 'down'} {abs(chg):.0%} vs the previous 30 days",
                            detail=f"{cons['last']:,.0f} units in the last 30 days vs {cons['prev']:,.0f} "
                                   "(mixed units: tablets, vials, ml…)."))

    # 9. Data quality that changes how numbers should be read
    nocost = held[held["unit_cost"].isna()]
    if len(nocost):
        out.append(dict(severity="info", tab="Dead & Excess", metric=len(nocost),
                        title=f"{len(nocost)} stocked items have no unit cost",
                        detail="Their value is excluded from KES totals, so value figures are a floor."))

    rank = {"critical": 0, "warning": 1, "positive": 2, "info": 3}
    return sorted(out, key=lambda i: rank[i["severity"]])


def why_flagged(row: pd.Series) -> str:
    """Plain-language reason for an item's status/priority (no AI needed)."""
    parts = []
    if row.get("stock_status") == "Negative":
        parts.append("system stock is below zero — dispensed without a recorded receipt")
    elif row.get("stock_status") == "Stocked out":
        parts.append("no stock on hand")
    elif pd.notna(row.get("days_of_stock")):
        parts.append(f"{row['days_of_stock']:.0f} days left at {row.get('adc_90d', 0):.1f}/day (90-day average)")
    if row.get("clinical_tier") in ("CRITICAL", "HIGH"):
        parts.append(f"{row['clinical_tier'].lower()}-priority class ({row.get('therapeutic_subclass')})")
    return "; ".join(parts).capitalize()
