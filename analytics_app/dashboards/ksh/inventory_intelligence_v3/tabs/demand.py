"""Demand Insights — trends, movers, anomalies, ABC, forecasts and the seasonal outlook."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import config as C, engines, ui


def _abc(ctx) -> pd.DataFrame:
    d, end = ctx["daily"], ctx["usage_end"]
    d = d[d["movement_date"] <= end]
    col = {"Dispensing only": "net_dispensing_qty", "Ward use only": "net_consumable_qty"}.get(ctx["f"].stream, "consumption")
    y = d[d["movement_date"] > end - pd.Timedelta(days=365)].groupby("product_id", as_index=False)[col].sum()
    y = y[y[col] > 0].rename(columns={col: "use_365d"})
    cost = ctx["stock_all"].groupby("product_id")["unit_cost"].median()
    y = y.merge(ctx["products"][["product_id", "product_name", "product_class", "therapeutic_class"]], on="product_id")
    y["unit_cost"] = y["product_id"].map(cost)
    y["value_365d"] = y["use_365d"] * y["unit_cost"]
    y = y.sort_values("value_365d", ascending=False)
    y["cum_share"] = y.groupby("product_class")["value_365d"].cumsum() / y.groupby("product_class")["value_365d"].transform("sum")
    y["abc"] = np.select([y["value_365d"].isna() | (y["value_365d"] <= 0), y["cum_share"] <= 0.7, y["cum_share"] <= 0.9],
                         ["Not priced", "A", "B"], "C")
    return y


def render(ctx: dict) -> None:
    # This tab follows the "Demand view" switch; every other tab always uses all usage.
    ana = ctx["analysis"]
    m, prof, as_of = ana["matrix"], ana["demand"], ctx["as_of"]
    if m.empty or prof.empty:
        ui.empty()
        return
    named = prof.merge(ctx["products"][["product_id", "product_name", "product_class", "clinical_tier",
                                        "therapeutic_subclass"]], on="product_id")
    an = ana["anomalies"].merge(ctx["products"][["product_id", "product_name", "product_class"]], on="product_id")
    d = ctx["daily"]
    end = ctx["usage_end"]
    d = d[d["movement_date"] <= end]                       # usage windows end on the last complete day
    l30 = d[d["movement_date"] > end - pd.Timedelta(days=30)]
    disp, ward = l30["net_dispensing_qty"].sum(), l30["net_consumable_qty"].sum()
    cons = ana["consumption"]
    if ctx["f"].stream != "All consumption":
        st.caption(f"Demand view: {ctx['f'].stream.lower()}. Stock status, orders and stockouts on the other tabs "
                   "always use all usage.")
    chg = cons["last"] / cons["prev"] - 1 if cons["prev"] else float("nan")
    # A fall to near zero is often a stockout, not lower demand: flag items out of stock in any selected store.
    out_now = set(ctx["stock"].loc[ctx["stock"]["stock_status"].isin(["Stocked out", "Negative"]), "product_id"])
    OUT, STOP = "Out in a selected store", "Stopped moving, stock on hand"
    named["supply"] = np.select([named["product_id"].isin(out_now), named["use_30d"] <= 0], [OUT, STOP], "")
    an["supply"] = np.select([an["product_id"].isin(out_now),
                              (an["direction"] == "Drop") & (an["recent_daily"] <= 0)], [OUT, STOP], "")
    rising = named[(named["trend"] == "UP") & (named["use_prev_30d"] >= 30)]
    falling = named[(named["trend"] == "DOWN") & (named["use_prev_30d"] >= 30)]
    supply_falls = (falling["supply"] == OUT).sum()
    stopped = (falling["supply"] == STOP).sum()

    ui.kpi_row([
        ui.kpi("Consumption, 30 days", ui.num(cons["last"]), f"{ui.pct(chg, signed=True)} vs previous 30 days",
               tip="Units used, net of returns; mixed units such as tablets, vials and ml."),
        ui.kpi("Dispensed vs ward use", ui.pct(disp / (disp + ward) if disp + ward else float("nan")),
               f"dispensed · {ui.pct(ward / (disp + ward) if disp + ward else float('nan'))} ward use"),
        ui.kpi("Rising items", ui.num(len(rising)), "up 15%+ vs previous 30 days", tone="warn" if len(rising) else ""),
        ui.kpi("Falling items", ui.num(len(falling)), f"{supply_falls} of them out of stock",
               tip=f"Down 15%+ vs the previous 30 days. {supply_falls} are out of stock in a store (a supply problem, "
                   f"not lower demand); {stopped} stopped moving although stock is on hand (check for a substitute)."),
        ui.kpi("Unusual demand", ui.num(len(an)), f"last {C.ANOMALY_RECENT_DAYS} days",
               tone="warn" if len(an) else "",
               tip=f"Use in the last {C.ANOMALY_RECENT_DAYS} days far from the previous 90 days "
                   f"(at least {C.ANOMALY_Z:.0f} standard deviations)."),
    ])

    ui.section("Weekly consumption", "Last 52 weeks, net of returns")
    w = d[d["movement_date"] > end - pd.Timedelta(days=364)].set_index("movement_date")[
        ["net_dispensing_qty", "net_consumable_qty"]].resample("W").sum()
    w = w[w.index <= end]          # weeks end on Sunday: keep only weeks complete by the last complete day
    fig = go.Figure()
    if ctx["f"].stream != "Ward use only":
        fig.add_trace(go.Scatter(x=w.index, y=w["net_dispensing_qty"], name="Dispensing", stackgroup="one",
                                 line=dict(color=C.STREAM_COLORS["Dispensing"])))
    if ctx["f"].stream != "Dispensing only":
        fig.add_trace(go.Scatter(x=w.index, y=w["net_consumable_qty"], name="Ward use", stackgroup="one",
                                 line=dict(color=C.STREAM_COLORS["Ward use"])))
    ui.plot(ui.style(fig, 300))

    t1, t2, t3 = st.tabs(["Movers & anomalies", "ABC (value)", "Forecast explorer"])
    with t1:
        a, b = st.columns(2, gap="large")
        cfg = {"product_name": st.column_config.TextColumn("Product", width="medium"),
               "use_30d": st.column_config.NumberColumn("Last 30 d", format="%.0f"),
               "use_prev_30d": st.column_config.NumberColumn("Prev 30 d", format="%.0f"),
               "change_30d": st.column_config.NumberColumn("Change", format="percent"),
               "supply": "Note"}
        with a:
            ui.section("Biggest risers", "Items used 30+ times in the previous 30 days")
            st.dataframe(rising.nlargest(15, "change_30d")[list(cfg)], hide_index=True,
                         use_container_width=True, column_config=cfg)
        with b:
            ui.section("Biggest fallers", "Check the Note column: a stockout is a supply problem, not lower demand")
            st.dataframe(falling.nsmallest(15, "change_30d")[list(cfg)], hide_index=True,
                         use_container_width=True, column_config=cfg)
        ui.section("Unusual demand", "Last 14 days compared with the 90 days before")
        if an.empty:
            ui.empty("No unusual demand at the current filters.")
        else:
            st.dataframe(an[["product_name", "product_class", "direction", "recent_per_active_day",
                             "baseline_per_active_day", "change_pct", "z", "supply"]], hide_index=True,
                         use_container_width=True,
                         column_config={"product_name": "Product", "product_class": "Class", "direction": "Signal",
                                        "supply": "Note",
                                        "recent_per_active_day": st.column_config.NumberColumn("Recent / day", format="%.1f"),
                                        "baseline_per_active_day": st.column_config.NumberColumn("Normal / day", format="%.1f"),
                                        "change_pct": st.column_config.NumberColumn("Change", format="percent"),
                                        "z": st.column_config.NumberColumn("z-score", format="%.1f")})
    with t2:
        abc = _abc(ctx)
        if abc.empty:
            ui.empty()
        else:
            ui.section("ABC within each product class",
                       "Last 365 days of use × unit cost. A = top 70% of value, B = next 20%, C = last 10%.")
            summ = abc.groupby(["product_class", "abc"]).agg(items=("product_id", "count"),
                                                             value=("value_365d", "sum")).reset_index()
            summ["value_share"] = summ["value"] / summ.groupby("product_class")["value"].transform("sum")
            st.dataframe(summ, hide_index=True, use_container_width=True, column_config={
                "product_class": "Class", "abc": "ABC", "items": "Items",
                "value": st.column_config.NumberColumn("Value (KES)", format="localized"),
                "value_share": st.column_config.NumberColumn("Share of class value", format="percent")})
            classes = [c for c in C.CLASS_ORDER if c in set(abc["product_class"])]
            pick = st.selectbox("Pareto for", classes, index=0)
            p = abc[(abc["product_class"] == pick) & (abc["abc"] != "Not priced")].head(40)
            fig = go.Figure(go.Bar(x=p["product_name"].str.title().str[:28], y=p["value_365d"],
                                   marker_color=p["abc"].map({"A": C.PRIMARY, "B": C.AMBER, "C": C.GREY})))
            fig.add_trace(go.Scatter(x=p["product_name"].str.title().str[:28], y=p["cum_share"], yaxis="y2",
                                     name="Cumulative share", line=dict(color=C.INK)))
            fig.update_layout(yaxis2=dict(overlaying="y", side="right", tickformat=".0%", range=[0, 1.02]))
            ui.plot(ui.style(fig, 360, legend=False))
    with t3:
        opts = named.sort_values("use_30d", ascending=False)
        labels = (opts["product_name"].str.title() + "  ·  " + opts["product_class"].fillna("")).tolist()
        choice = st.selectbox("Product (busiest first)", range(len(opts)), format_func=lambda i: labels[i])
        row = opts.iloc[choice]
        seas = ana["seasonal"].set_index("product_id")
        mult = float(seas["seasonal_multiplier"].get(row["product_id"], 1.0))
        disease = seas["calendar_disease"].get(row["product_id"], "") if len(seas) else ""
        ui.kpi_row([   # same basis as the chart line: EWM × trend × seasonal
            ui.kpi("Forecast, next 30 days", ui.num(row["forecast_30d"] * mult),
                   f"90% range {ui.num(row['forecast_30d_low'] * mult)}–{ui.num(row['forecast_30d_high'] * mult)}"
                   + (f" · incl. seasonal × {mult:.2f}" if abs(mult - 1) >= 0.005 else "")),
            ui.kpi("Demand pattern", str(row["demand_type"]).title(), f"confidence {row['confidence'].lower()}"),
            ui.kpi("Trend", row["trend"].title(), f"{ui.pct(row['change_30d'], signed=True)} vs previous 30 days"),
            ui.kpi("Seasonal outlook", f"× {mult:.2f}", disease or "no calendar peak in the next 30 days"),
        ])
        hist = m[row["product_id"]].iloc[-182:]
        fc = engines.forecast_path(hist, row, 90, avg_days=7, mult=mult)
        wk = hist.resample("W").mean().iloc[:-1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wk.index, y=wk.values, name="Actual (weekly avg / day)", line=dict(color=C.PRIMARY)))
        fig.add_trace(go.Scatter(x=list(fc["date"]) + list(fc["date"][::-1]), y=list(fc["high"]) + list(fc["low"][::-1]),
                                 fill="toself", fillcolor="rgba(15,110,86,0.12)", line=dict(width=0),
                                 name="90% range (weekly avg)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fc["date"], y=fc["mean"], name="Forecast (× seasonal)",
                                 line=dict(color=C.PRIMARY, dash="dash")))
        ui.plot(ui.style(fig, 320))

    ui.section("Seasonal outlook, next 30 days", "Kisumu disease calendar (MoH IDSR) × this facility's own pattern")
    months = {(as_of + pd.Timedelta(days=k)).month for k in range(1, 31)}
    peaks = [dz for dz in C.DISEASE_CALENDAR if months & set(dz.peak_months)]
    if not peaks:
        st.caption("No calendar peaks in the next 30 days.")
    se = ctx["seasonal"].merge(ctx["stock"].groupby("product_id", as_index=False).agg(
        product_name=("product_name", "first"), qty=("qty_on_hand", "sum"), adc=("adc_90d", "sum")), on="product_id")
    for dz in peaks:
        hit = se[se["calendar_disease"] == dz.name].copy()
        hit["days_at_peak"] = hit["qty"].clip(lower=0) / (hit["adc"] * hit["seasonal_multiplier"]).replace(0, np.nan)
        short = hit[hit["days_at_peak"] < C.DOS_LOW]
        applied = hit["seasonal_multiplier"].median() if len(hit) else 1.0
        with st.expander(f"{dz.name} — calendar peak × {dz.multiplier:.2f}, applied × {applied:.2f} (median, blended "
                         f"with this facility's pattern) · {len(hit)} items · "
                         f"{len(short)} with < {C.DOS_LOW} days at peak demand", expanded=bool(len(short))):
            st.caption(dz.note)
            st.dataframe(hit.sort_values("days_at_peak")[["product_name", "qty", "adc", "seasonal_multiplier",
                                                          "days_at_peak"]],
                         hide_index=True, use_container_width=True, column_config={
                             "product_name": "Product", "qty": st.column_config.NumberColumn("On hand", format="%d"),
                             "adc": st.column_config.NumberColumn("Use / day now", format="%.1f"),
                             "seasonal_multiplier": st.column_config.NumberColumn("Expected ×", format="%.2f"),
                             "days_at_peak": st.column_config.NumberColumn("Days at peak demand", format="%.0f")})
