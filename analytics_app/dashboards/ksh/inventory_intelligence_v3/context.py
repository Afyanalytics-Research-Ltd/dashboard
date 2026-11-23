"""Builds everything the tabs need for one filter selection, once (cached per selection)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from . import config as C
from . import data, engines, insights


def _demand_set(daily: pd.DataFrame, col: str, products: pd.DataFrame, usage_end: pd.Timestamp,
                as_of: pd.Timestamp, dark_days) -> dict:
    """Demand engines on one consumption basis: matrix, profile (EWM, trend, forecast), anomalies, seasonality and
    the 30-day consumption totals."""
    m = engines.daily_matrix(daily, col, usage_end)
    profile = engines.demand_profile(m, usage_end)
    sel = products[products["product_id"].isin(profile["product_id"])] if len(profile) else products.iloc[0:0]
    seasonal = engines.seasonality(sel, engines.monthly_rates(daily, col, dark_days), as_of) if len(sel) else         pd.DataFrame(columns=["product_id", "seasonal_multiplier", "calendar_disease"])
    return dict(col=col, matrix=m, demand=profile, anomalies=engines.anomalies(m), seasonal=seasonal,
                consumption={"last": float(m.iloc[-30:].values.sum()) if len(m) else 0.0,
                             "prev": float(m.iloc[-60:-30].values.sum()) if len(m) else 0.0})


@st.cache_data(ttl=3600, show_spinner=False)
def build(key: tuple) -> dict:
    f = data.Filters(*[list(k) if isinstance(k, tuple) else k for k in key])
    d = data.load_all()
    as_of = data.as_of(d)              # stock date (idle days, stockout durations, "out as of")
    usage_end = data.usage_end(d)      # last complete day: every usage / trend / forecast window ends here
    products = d["products"]
    col = data.stream_column(f)

    daily_store = d["daily"][data.store_mask(d["daily"], f)]
    daily = daily_store[data.product_mask(daily_store, f, products)]

    # Stock status, priorities, orders, stockouts and insights always use ALL usage — that is what empties the
    # shelf. The consumption-basis switch only changes the Demand Insights analysis.
    physical = d["stock"]
    # Equipment recorded as stock (analyst-reviewed list in STAGING) is reported on its own, never as medical stock:
    # it is left out of stock value, idle, excess, ordering and stockouts everywhere below.
    is_eq = physical["is_equipment"].fillna(False).astype(bool)
    equipment = data.filtered(physical[is_eq], f, products)
    physical = physical[~is_eq]
    stock_all = physical
    # Last outbound movement (used OR transferred out) per store x product — idleness must count transfers,
    # otherwise a distribution store (Main Store) looks idle by design.
    out_rows = d["daily"][(d["daily"]["consumption"] > 0) | (d["daily"]["transfer_out_qty"] > 0)]
    last_out = out_rows.groupby(["store_id", "product_id"], as_index=False)["movement_date"].max().rename(
        columns={"movement_date": "last_outflow_date"})
    stock_all = stock_all.merge(last_out, on=["store_id", "product_id"], how="left")
    stock_all["days_since_last_outflow"] = (as_of - stock_all["last_outflow_date"]).dt.days
    stock = data.filtered(stock_all, f, products)
    stockouts = data.filtered(d["stockouts"], f, products)
    control = d["control"][data.store_mask(d["control"], f) | d["control"]["store_id"].isna()]

    # engines (selected stores + product filters)
    # Days with no ledger rows anywhere (e.g. 6–16 Oct 2025, a source-wide gap) are missing data, not zero use.
    all_days = pd.date_range(d["daily"]["movement_date"].min(), as_of, freq="D")
    dark_days = all_days.difference(pd.DatetimeIndex(d["daily"]["movement_date"].unique()))

    ops = _demand_set(daily, "consumption", products, usage_end, as_of, dark_days)          # operational
    analysis = ops if col == "consumption" else _demand_set(daily, col, products, usage_end, as_of, dark_days)
    lt_per, lt_fac = engines.lead_times(daily_store)

    # Stores whose usage is not captured in the inventory ledger (e.g. the Laboratory: large stock, no
    # consumption movements). Their stock always looks "idle", so idle-stock figures exclude them by default.
    yr = d["daily"][(d["daily"]["movement_date"] > as_of - pd.Timedelta(days=365)) &
                    ((d["daily"]["consumption"] > 0) | (d["daily"]["transfer_out_qty"] > 0))]
    usage_days = yr.groupby("store_id")["movement_date"].nunique()
    active = d["stores"][d["stores"]["is_active_store"]]
    no_usage = active[~active["store_id"].isin(usage_days[usage_days >= 30].index)]

    # Redistribution looks across all active stores on PHYSICAL cover (all outflows), never donating from a
    # store whose usage isn't recorded; keep pairs touching the selected stores.
    rd_src = data.filtered(physical, data.Filters(classes=f.classes, ther_classes=f.ther_classes,
                                                   tiers=f.tiers, search=f.search), products)
    rd = insights.redistribution(rd_src, exclude_donors=set(no_usage["store_name"]))
    if len(rd) and f.stores:
        names = set(d["stores"].loc[d["stores"]["store_id"].isin(f.stores), "store_name"])
        rd = rd[rd["from_store"].isin(names) | rd["to_store"].isin(names)]

    ctx = dict(as_of=as_of, usage_end=usage_end, f=f, equipment=equipment, products=products, stores=d["stores"], stock=stock, stock_all=stock_all,
               stockouts=stockouts, control=control, dark_days=dark_days,
               product_filtered=bool(f.classes or f.ther_classes or f.tiers or f.search),
               daily=daily, matrix=ops["matrix"], demand=ops["demand"], anomalies=ops["anomalies"],
               seasonal=ops["seasonal"], consumption=ops["consumption"], analysis=analysis,
               lead_times=(lt_per, lt_fac), redistribution=rd,
               no_usage_stores=set(no_usage["store_name"]))
    # Availability of the items currently in use (used in the last 90 days) in the selected stores and products.
    in_use = stock[stock["adc_90d"] > 0][["store_id", "product_id", "clinical_tier"]]
    start = min(usage_end - pd.Timedelta(days=C.AVAILABILITY_DAYS - 1), pd.Timestamp(C.RELIABLE_LEDGER_FROM))
    mat = engines.in_stock_matrix(d["daily"], in_use, start, usage_end)
    tier = in_use.set_index(["store_id", "product_id"])["clinical_tier"]
    last90 = mat.loc[usage_end - pd.Timedelta(days=C.AVAILABILITY_DAYS - 1):]
    per_item = last90.mean()
    by_tier = per_item.groupby(tier.reindex(per_item.index).values).mean() if len(per_item) else pd.Series(dtype=float)
    last_full = (as_of.to_period("M") - 1).to_timestamp()
    trend = mat.loc[pd.Timestamp(C.RELIABLE_LEDGER_FROM):last_full + pd.offsets.MonthEnd(0)]
    monthly = pd.DataFrame({"All items in use": trend.mean(axis=1)})
    for t in ("CRITICAL", "HIGH"):
        cols = [c for c in trend.columns if tier.get(c) == t]
        if cols:
            monthly[t.title()] = trend[cols].mean(axis=1)
    monthly = monthly.resample("MS").mean()
    ctx["availability"] = dict(rate=float(per_item.mean()) if len(per_item) else float("nan"),
                               by_tier=by_tier, monthly=monthly, n_items=int(len(in_use)),
                               in_stock_now=float((stock.loc[stock["adc_90d"] > 0, "qty_on_hand"] > 0).mean())
                               if len(in_use) else float("nan"))
    ctx["insights"] = insights.build(ctx)
    return ctx
