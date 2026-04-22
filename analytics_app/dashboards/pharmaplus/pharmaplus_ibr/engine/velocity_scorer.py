"""
engine/velocity_scorer.py
Computes sales velocity per SKU per branch.
Also detects trend acceleration/deceleration for predictive redistribution.
"""

import pandas as pd
import numpy as np
from datetime import timedelta


def compute_velocity(
    dispensing: pd.DataFrame,
    window_days: int = 30,
) -> pd.DataFrame:
    """
    Computes average daily units sold per (store_id, product_id) over the last N days.

    Returns:
        store_id, product_id,
        units_last_30d, avg_daily_units,
        velocity_score (0–100, normalised within each product across branches),
        revenue_last_30d
    """
    cutoff = dispensing["date"].max() - timedelta(days=window_days)
    recent = dispensing[dispensing["date"] >= cutoff].copy()

    vel = (
        recent
        .groupby(["store_id", "product_id"])
        .agg(
            units_last_30d=("qty_dispensed", "sum"),
            revenue_last_30d=("total_sales_value", "sum"),
        )
        .reset_index()
    )
    vel["avg_daily_units"] = vel["units_last_30d"] / window_days

    # Normalize velocity score 0–100 within each product
    def normalize(x):
        mn, mx = x.min(), x.max()
        if mx == mn:
            return pd.Series([50.0] * len(x), index=x.index)
        return ((x - mn) / (mx - mn)) * 100

    vel["velocity_score"] = (
        vel.groupby("product_id")["avg_daily_units"]
        .transform(normalize)
    )

    return vel


def compute_trend(
    dispensing: pd.DataFrame,
    short_window: int = 14,
    long_window: int = 28,
) -> pd.DataFrame:
    """
    Detects momentum: is this SKU accelerating or decelerating at this branch?

    Trend = (units in last 14d / units in prior 14d) - 1
        > +0.2  → accelerating (growing)
        < -0.2  → decelerating (slowing)
        else    → stable

    Returns:
        store_id, product_id,
        units_short, units_prior,
        trend_pct, trend_label
    """
    today = dispensing["date"].max()
    short_start = today - timedelta(days=short_window)
    long_start = today - timedelta(days=long_window)

    short = dispensing[dispensing["date"] > short_start].groupby(
        ["store_id", "product_id"]
    )["qty_dispensed"].sum().reset_index(name="units_short")

    prior = dispensing[
        (dispensing["date"] > long_start) & (dispensing["date"] <= short_start)
    ].groupby(["store_id", "product_id"])["qty_dispensed"].sum().reset_index(name="units_prior")

    trend = short.merge(prior, on=["store_id", "product_id"], how="outer").fillna(0)

    def calc_trend(row):
        if row["units_prior"] == 0:
            return 1.0 if row["units_short"] > 0 else 0.0
        return (row["units_short"] - row["units_prior"]) / row["units_prior"]

    trend["trend_pct"] = trend.apply(calc_trend, axis=1)

    def label(t):
        if t > 0.20:
            return "accelerating"
        elif t < -0.20:
            return "decelerating"
        return "stable"

    trend["trend_label"] = trend["trend_pct"].apply(label)
    return trend