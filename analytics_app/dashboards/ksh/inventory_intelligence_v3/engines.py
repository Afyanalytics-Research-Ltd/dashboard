"""Analytical engines over the daily consumption data.

  demand_profile   EWM demand, Syntetos-Boylan demand type, trend, 30-day forecast + band, confidence
  anomalies        last-14-day rate vs prior-90-day non-zero baseline, |z| >= 2
  lead_times       low-stock -> next real receipt, measured per item (V3: receipts are real ledger events)
  seasonality      Kisumu disease calendar blended with the facility's own seasonal index
  replenishment    safety stock, reorder point, order-up-to quantity at a chosen service level

All take the daily store x product fact (already filtered) and the data's own as-of date.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from . import config as C


# ── shared ───────────────────────────────────────────────────────────────────
def daily_matrix(daily: pd.DataFrame, value_col: str, as_of: pd.Timestamp, days: int = 730) -> pd.DataFrame:
    """Dates x products, zero-filled, summed over the selected stores, for products used in the last year."""
    start = as_of - pd.Timedelta(days=days - 1)
    d = daily[(daily["movement_date"] >= start) & (daily["movement_date"] <= as_of)]
    recent = d[d["movement_date"] > as_of - pd.Timedelta(days=365)]
    keep = recent.loc[recent[value_col] > 0, "product_id"].unique()
    d = d[d["product_id"].isin(keep)]
    if d.empty:
        return pd.DataFrame(index=pd.date_range(start, as_of, freq="D"))
    m = d.pivot_table(index="movement_date", columns="product_id", values=value_col, aggfunc="sum")
    m = m.reindex(pd.date_range(start, as_of, freq="D")).fillna(0.0)
    return m.clip(lower=0)          # net returns can make a day negative; demand is never negative


# ── demand ───────────────────────────────────────────────────────────────────
PROFILE_COLS = ["product_id", "first_use", "history_days", "ewm_daily", "avg_daily_90d", "std_daily_90d", "use_30d",
                "use_prev_30d", "change_30d", "trend", "adi", "cv2", "demand_type", "forecast_30d",
                "forecast_30d_low", "forecast_30d_high", "confidence"]
ANOMALY_COLS = ["product_id", "recent_daily", "recent_per_active_day", "baseline_per_active_day",
                "baseline_active_days", "z", "change_pct", "direction"]
LT_COLS = ["product_id", "lt_median", "lt_std", "lt_obs"]


def demand_profile(m: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    if m.empty or m.shape[1] == 0:
        return pd.DataFrame(columns=PROFILE_COLS)
    nonzero = m > 0
    first_use = nonzero.idxmax()
    n_days = ((as_of - first_use).dt.days + 1).clip(lower=1)
    nz_days = nonzero.sum()

    span = (n_days // 4).clip(C.EWM_SPAN_MIN, C.EWM_SPAN_MAX)
    ewm = pd.Series(index=m.columns, dtype=float)
    for s in span.unique():
        cols = span.index[span == s]
        ewm[cols] = m[cols].ewm(span=int(s), adjust=False).mean().iloc[-1]

    last90 = m.iloc[-90:]
    avg90 = last90.mean()
    std90 = last90.std(ddof=0)
    l30, p30 = m.iloc[-30:].sum(), m.iloc[-60:-30].sum()
    chg = (l30 - p30) / p30.replace(0, np.nan)
    trend = np.where(chg > C.TREND_THRESHOLD, "UP", np.where(chg < -C.TREND_THRESHOLD, "DOWN", "STABLE"))

    last = m.iloc[-(len(m) // 7) * 7:][-364:]
    weekly = last.groupby(np.arange(len(last)) // 7).sum()
    wk_nz = weekly.where(weekly > 0)
    cv2 = (wk_nz.std(ddof=0) / wk_nz.mean()) ** 2
    adi = n_days / nz_days.replace(0, np.nan)
    smooth_freq = adi < C.SB_ADI_THRESHOLD
    variable = cv2.fillna(0) >= C.SB_CV2_THRESHOLD
    dtype = np.where(smooth_freq, np.where(variable, "ERRATIC", "SMOOTH"),
                     np.where(variable, "LUMPY", "INTERMITTENT"))

    out = pd.DataFrame({
        "product_id": m.columns, "first_use": first_use.values, "history_days": n_days.values,
        "ewm_daily": ewm.values, "avg_daily_90d": avg90.values, "std_daily_90d": std90.values,
        "use_30d": l30.values, "use_prev_30d": p30.values, "change_30d": chg.values,
        "trend": trend, "adi": adi.values, "cv2": cv2.values, "demand_type": dtype,
    })
    mult = out["trend"].map(C.TREND_MULT)
    out["forecast_30d"] = out["ewm_daily"] * 30 * mult
    band = C.FORECAST_Z * out["std_daily_90d"] * math.sqrt(30)
    out["forecast_30d_low"] = (out["forecast_30d"] - band).clip(lower=0)
    out["forecast_30d_high"] = out["forecast_30d"] + band
    out["confidence"] = np.select(
        [(out["history_days"] < C.MIN_DAYS_MEDIUM_CONFIDENCE) | (out["demand_type"] == "LUMPY"),
         (out["demand_type"] == "SMOOTH") & (out["history_days"] >= C.MIN_DAYS_HIGH_CONFIDENCE)],
        ["LOW", "HIGH"], "MEDIUM")
    short = out["history_days"] < C.MIN_DAYS_FOR_FORECAST
    out.loc[short, ["forecast_30d", "forecast_30d_low", "forecast_30d_high"]] = np.nan
    out.loc[short, "demand_type"] = "INSUFFICIENT HISTORY"
    return out


def forecast_path(series: pd.Series, profile_row: pd.Series, horizon: int = 90, avg_days: int = 7,
                  mult: float = 1.0) -> pd.DataFrame:
    """Flat EWM-based forecast for one product's chart. The band is for an `avg_days`-day average
    (σ/√avg_days), so it is on the same scale as a weekly-average actuals line."""
    idx = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    d = float(profile_row["ewm_daily"]) * C.TREND_MULT.get(profile_row["trend"], 1.0) * mult
    sd = float(profile_row["std_daily_90d"]) * mult / np.sqrt(max(avg_days, 1))
    return pd.DataFrame({"date": idx, "mean": d,
                         "low": max(d - C.FORECAST_Z * sd, 0.0), "high": d + C.FORECAST_Z * sd})


# ── anomalies ────────────────────────────────────────────────────────────────
def anomalies(m: pd.DataFrame) -> pd.DataFrame:
    if m.shape[0] < C.ANOMALY_RECENT_DAYS + C.ANOMALY_BASELINE_DAYS or m.shape[1] == 0:
        return pd.DataFrame(columns=ANOMALY_COLS)
    recent = m.iloc[-C.ANOMALY_RECENT_DAYS:]
    base = m.iloc[-(C.ANOMALY_RECENT_DAYS + C.ANOMALY_BASELINE_DAYS):-C.ANOMALY_RECENT_DAYS]
    base_nz = base.where(base > 0)
    mu, sd, n_nz = base_nz.mean(), base_nz.std(ddof=0), base_nz.count()
    r = recent.mean()
    # compare like with like: recent rate per active day vs baseline rate per active day
    r_nz = recent.where(recent > 0).mean().fillna(0)
    z = (r_nz - mu) / sd.replace(0, np.nan)
    out = pd.DataFrame({"product_id": m.columns, "recent_daily": r.values, "recent_per_active_day": r_nz.values,
                        "baseline_per_active_day": mu.values, "baseline_active_days": n_nz.values, "z": z.values})
    out["change_pct"] = out["recent_per_active_day"] / out["baseline_per_active_day"] - 1
    out = out[(out["baseline_active_days"] >= 5) & (out["z"].abs() >= C.ANOMALY_Z)]
    out["direction"] = np.where(out["z"] > 0, "Spike", "Drop")
    return out.sort_values("z", key=np.abs, ascending=False)


# ── availability ─────────────────────────────────────────────────────────────
def in_stock_matrix(daily: pd.DataFrame, keys: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Dates x (store_id, product_id): True when the item had stock on hand at the end of that day. A day's stock is
    the closing of the last ledger movement on or before it (carried forward); NaN before an item's first movement."""
    if keys.empty:
        return pd.DataFrame(index=pd.date_range(start, end))
    d = daily.merge(keys[["store_id", "product_id"]], on=["store_id", "product_id"])
    d = d[d["movement_date"] <= end]
    if d.empty:
        return pd.DataFrame(index=pd.date_range(start, end))
    piv = d.pivot_table(index="movement_date", columns=["store_id", "product_id"], values="day_closing", aggfunc="last")
    piv = piv.reindex(pd.date_range(min(piv.index.min(), start), end)).ffill().loc[start:end]
    return (piv > 0).where(piv.notna())


# ── lead times ───────────────────────────────────────────────────────────────
def lead_times(daily: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Days from a low-stock point to the next recorded receipt, per store x product, summarised per product."""
    d = daily[["store_id", "product_id", "movement_date", "day_closing", "receipt_qty"]].dropna(
        subset=["day_closing"]).sort_values(["store_id", "product_id", "movement_date"])
    default = {"median": C.DEFAULT_LEAD_TIME_DAYS, "std": C.DEFAULT_LEAD_TIME_STD,
               "p90": C.DEFAULT_LEAD_TIME_DAYS * 2, "n": 0}
    if d.empty:
        return pd.DataFrame(columns=LT_COLS), default
    pos_median = d[d["day_closing"] > 0].groupby(["store_id", "product_id"])["day_closing"].median()
    d = d.join(pos_median.rename("pos_median"), on=["store_id", "product_id"])
    d["low"] = d["day_closing"] <= np.maximum(1, 0.2 * d["pos_median"].fillna(0))
    prev_low = d.groupby(["store_id", "product_id"])["low"].shift(fill_value=False)
    events = d[d["low"] & ~prev_low][["store_id", "product_id", "movement_date"]]
    rcpt = d[d["receipt_qty"] > 0][["store_id", "product_id", "movement_date"]].rename(
        columns={"movement_date": "receipt_date"})
    if events.empty or rcpt.empty:
        return pd.DataFrame(columns=LT_COLS), default
    j = pd.merge_asof(events.sort_values("movement_date"), rcpt.sort_values("receipt_date"),
                      left_on="movement_date", right_on="receipt_date", by=["store_id", "product_id"],
                      direction="forward", allow_exact_matches=False)
    j["lt"] = (j["receipt_date"] - j["movement_date"]).dt.days
    obs = j[(j["lt"] >= 1) & (j["lt"] <= 180)]
    # Robust spread (1.4826 x median absolute deviation): the gaps have a long tail of 100+ day waits
    # (items left at zero), which would otherwise inflate safety stock for everything.
    def robust_sd(s: pd.Series) -> float:
        return float(1.4826 * (s - s.median()).abs().median())
    facility = {"median": float(obs["lt"].median()) if len(obs) else C.DEFAULT_LEAD_TIME_DAYS,
                "std": robust_sd(obs["lt"]) if len(obs) > 1 else C.DEFAULT_LEAD_TIME_STD,
                "p90": float(obs["lt"].quantile(0.9)) if len(obs) else C.DEFAULT_LEAD_TIME_DAYS * 2,
                "n": int(len(obs))}
    per = obs.groupby("product_id")["lt"].agg(lt_median="median", lt_std=robust_sd, lt_obs="count").reset_index()
    return per, facility


def attach_lead_time(df: pd.DataFrame, per: pd.DataFrame, facility: dict) -> pd.DataFrame:
    per = per.reindex(columns=LT_COLS).astype({"product_id": df["product_id"].dtype, "lt_median": float,
                                               "lt_std": float, "lt_obs": float})
    out = df.merge(per, on="product_id", how="left")
    measured = out["lt_obs"].fillna(0) >= C.MIN_LT_OBSERVATIONS
    # An item's gap can be months long when it was simply left low; that is reordering delay, not supplier lead
    # time. Cap the item's lead time at 2x the facility's typical wait and its spread at the facility's typical
    # spread, so erratic waits can't inflate safety stock.
    cap = C.LEAD_TIME_CAP_X_MEDIAN * facility["median"]
    capped = measured & ((out["lt_median"] > cap) | (out["lt_std"] > facility["std"]))
    out["lead_time_days"] = np.where(measured, np.minimum(out["lt_median"], cap), facility["median"])
    out["lead_time_std"] = np.where(measured, np.minimum(out["lt_std"].fillna(facility["std"]), facility["std"]),
                                    facility["std"])
    out["lead_time_source"] = np.select(
        [capped, measured, facility["n"] > 0],
        [f"Measured, capped (≤ {cap:.0f} d, spread ≤ {facility['std']:.0f} d)", "Measured (this item)",
         "Facility median"], "Default")
    return out


# ── seasonality ──────────────────────────────────────────────────────────────
def _horizon_months(as_of: pd.Timestamp, days: int = 30) -> set:
    return {d.month for d in pd.date_range(as_of + pd.Timedelta(days=1), periods=days, freq="D")}


def seasonality(products: pd.DataFrame, m_monthly: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Per product: next-30-day demand multiplier = blend of the disease calendar and the facility's own
    month-of-year index for the product's therapeutic subclass (V2: alpha 0.40 with two years of data)."""
    months = _horizon_months(as_of)
    text = (products["therapeutic_subclass"].fillna("") + " | " + products["therapeutic_class"].fillna("")
            + " | " + products["product_name"].fillna("")).str.lower()
    cal_mult = pd.Series(1.0, index=products.index)
    cal_name = pd.Series("", index=products.index)
    for dz in C.DISEASE_CALENDAR:
        if not months & set(dz.peak_months):
            continue
        # Keywords match at the start of a word ("macrolide" -> "macrolides"); a trailing space in the
        # keyword means whole word only ("ors " must not match "inhibitors").
        pattern = "|".join(r"\b" + re.escape(k.strip()) + (r"\b" if k.endswith(" ") else "") for k in dz.keywords)
        hit = text.str.contains(pattern, regex=True)
        better = hit & (dz.multiplier > cal_mult)
        cal_mult[better] = dz.multiplier
        cal_name[better] = dz.name

    idx = pd.Series(1.0, index=products.index)
    years = 0
    if not m_monthly.empty:
        mm = m_monthly.copy()
        mm["month_of_year"] = mm["month"].dt.month
        years = mm["month"].dt.year.nunique() - 1
        mm = mm.merge(products[["product_id", "therapeutic_subclass"]], on="product_id", how="left")
        mm = mm[mm["therapeutic_subclass"].fillna("Unclassified") != "Unclassified"]
    if not m_monthly.empty and not mm.empty:
        sub = mm.groupby(["therapeutic_subclass", "month"])["daily_rate"].sum().unstack(fill_value=0.0)
        sub = sub.reindex(columns=pd.date_range(mm["month"].min(), mm["month"].max(), freq="MS"), fill_value=0.0)
        sub = sub.stack().rename("daily_rate").rename_axis(["therapeutic_subclass", "month"]).reset_index()
        sub["moy"] = sub["month"].dt.month
        overall = sub.groupby("therapeutic_subclass")["daily_rate"].mean()
        by_moy = sub.groupby(["therapeutic_subclass", "moy"])["daily_rate"].agg(["mean", "count"]).reset_index()
        by_moy = by_moy[by_moy["count"] >= 2]
        by_moy = by_moy[by_moy["moy"].isin(months)]
        by_moy["index"] = by_moy["mean"] / by_moy["therapeutic_subclass"].map(overall)
        sub_idx = by_moy.groupby("therapeutic_subclass")["index"].mean().clip(0.7, 1.8)
        idx = products["therapeutic_subclass"].map(sub_idx).fillna(1.0)
    alpha = {1: 0.20, 2: 0.40, 3: 0.60}.get(max(years, 0), 0.75 if years > 3 else 0.0)
    out = products[["product_id"]].copy()
    out["calendar_multiplier"] = cal_mult.values
    out["calendar_disease"] = cal_name.values
    out["facility_index"] = np.asarray(idx, dtype=float)
    out["seasonal_multiplier"] = ((1 - alpha) * out["calendar_multiplier"] + alpha * out["facility_index"]).clip(0.7, 1.8)
    return out


def monthly_rates(daily: pd.DataFrame, value_col: str, dark_days=None) -> pd.DataFrame:
    """Product x calendar month average daily consumption (complete months only). `dark_days` (dates with no
    ledger rows anywhere in the source) are excluded from each month's day count."""
    d = daily[["movement_date", "product_id", value_col]].copy()
    d["month"] = d["movement_date"].dt.to_period("M").dt.to_timestamp()
    last_full = (d["movement_date"].max() + pd.Timedelta(days=1)).to_period("M").to_timestamp() - pd.offsets.MonthBegin(1)
    d = d[d["month"] <= last_full]
    g = d.groupby(["product_id", "month"], as_index=False)[value_col].sum()
    days = g["month"].dt.days_in_month
    if dark_days is not None and len(dark_days):
        dark = pd.Series(1, index=pd.DatetimeIndex(dark_days)).groupby(
            pd.DatetimeIndex(dark_days).to_period("M").to_timestamp()).sum()
        days = days - g["month"].map(dark).fillna(0)
    g["daily_rate"] = g[value_col] / days.clip(lower=1)
    return g


# ── replenishment ────────────────────────────────────────────────────────────
def replenishment(items: pd.DataFrame, z: float, cover_days: int = C.TARGET_COVER_DAYS) -> pd.DataFrame:
    """items: product-level rows with qty_on_hand, unit_cost, ewm_daily, std_daily_90d, adc_90d,
    seasonal_multiplier, lead_time_days, lead_time_std."""
    out = items.copy()
    for c in ("ewm_daily", "std_daily_90d", "adc_90d", "seasonal_multiplier", "lead_time_days", "lead_time_std",
              "qty_on_hand", "unit_cost"):        # empty merges arrive as object dtype
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    base = out["ewm_daily"].where(out["ewm_daily"].notna(), out["adc_90d"]).fillna(0)
    trend = out["trend"].map(C.TREND_MULT).fillna(1.0) if "trend" in out else 1.0
    out["planning_daily_demand"] = base * trend * out["seasonal_multiplier"].fillna(1.0)
    d, sd = out["planning_daily_demand"], out["std_daily_90d"].fillna(0)
    lt, slt = out["lead_time_days"], out["lead_time_std"]
    out["safety_stock"] = z * np.sqrt(lt * sd ** 2 + d ** 2 * slt ** 2)
    out["reorder_point"] = d * lt + out["safety_stock"]
    available = out["qty_on_hand"].clip(lower=0)
    out["order_up_to"] = d * (lt + cover_days) + out["safety_stock"]
    out["suggested_order_qty"] = np.ceil((out["order_up_to"] - available).clip(lower=0))
    out.loc[d <= 0, "suggested_order_qty"] = 0
    out["order_value"] = out["suggested_order_qty"] * out["unit_cost"]
    out["days_of_stock"] = np.where(d > 0, available / d, np.nan)
    out["at_or_below_rop"] = (available <= out["reorder_point"]) & (d > 0)
    out["stockout_before_delivery"] = (out["days_of_stock"] < lt) & (d > 0)
    return out
