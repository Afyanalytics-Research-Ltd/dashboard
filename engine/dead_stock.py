"""
engine/dead_stock.py

Flags SKUs at each branch by days-since-last-sale.
Tiers are now CATEGORY-AWARE — thresholds differ by business category:

    Pharma:                 WATCH 30d / ALERT 60d / DEAD 90d
    Beauty & Cosmetics:     WATCH 20d / ALERT 40d / DEAD 60d
    Vitamins & Supplements: WATCH 25d / ALERT 50d / DEAD 75d
    Body Building:          WATCH 20d / ALERT 40d / DEAD 60d
    Non-Pharma:             WATCH 15d / ALERT 30d / DEAD 45d

Outputs KSh value at risk per tier.
"""

import pandas as pd
import numpy as np
from datetime import date

from engine.catalogue_matcher import (
    get_category_tiers,
    DEAD_STOCK_TIERS_BY_CATEGORY,
)

# ── Legacy flat tiers kept as fallback for unmatched products ─────────────────
TIERS = {
    "WATCH": (30, 60),
    "ALERT": (60, 90),
    "DEAD":  (90, None),
}

TIER_ORDER = {"WATCH": 1, "ALERT": 2, "DEAD": 3}


def compute_last_sale(dispensing: pd.DataFrame) -> tuple[pd.DataFrame, date]:
    """
    Returns (DataFrame, today_date).
    DataFrame cols: store_id, product_id, last_sale_date, days_frozen
    """
    today = dispensing["date"].max().date()

    last_sale = (
        dispensing
        .groupby(["store_id", "product_id"])["date"]
        .max()
        .reset_index()
        .rename(columns={"date": "last_sale_date"})
    )
    last_sale["last_sale_date"] = last_sale["last_sale_date"].dt.date
    last_sale["days_frozen"] = (today - last_sale["last_sale_date"]).apply(lambda x: x.days)
    return last_sale, today


def assign_tier(days: int, internal_category: str = "Pharma") -> str | None:
    """
    Category-aware tier assignment.
    Falls back to legacy pharma thresholds if category is unrecognised.
    """
    thresholds = get_category_tiers(internal_category)
    watch_d = thresholds["WATCH"]
    alert_d = thresholds["ALERT"]
    dead_d  = thresholds["DEAD"]

    if days >= dead_d:
        return "DEAD"
    elif days >= alert_d:
        return "ALERT"
    elif days >= watch_d:
        return "WATCH"
    return None


def flag_dead_stock(
    dispensing: pd.DataFrame,
    inventory: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """
    Core dead stock flagging function — now category-aware.

    products DataFrame is expected to have an 'internal_category' column
    (added by loader.py after catalogue matching). Falls back gracefully
    to 'Pharma' thresholds if the column is absent.

    Returns one row per (store_id, product_id) crossing the WATCH threshold.
    Columns:
        store_id, product_id, product_name, category_name, internal_category,
        unit_of_measure, last_sale_date, days_frozen, tier,
        qty_on_hand, unit_cost, ksh_at_risk,
        shelf_life_days, analysis_date,
        watch_threshold, alert_threshold, dead_threshold  ← new: visible thresholds
    """
    last_sale_df, today = compute_last_sale(dispensing)

    # Drop deprecated items frozen longer than 2 years
    last_sale_df = last_sale_df[last_sale_df["days_frozen"] <= 730].copy()

    # Merge product metadata early so we have category for tier assignment
    prod_cols = [
        "product_id", "product_name", "category_name",
        "unit_of_measure", "shelf_life_days",
    ]
    if "internal_category" in products.columns:
        prod_cols.append("internal_category")

    last_sale_df = last_sale_df.merge(
        products[prod_cols],
        on="product_id",
        how="left",
    )

    if "internal_category" not in last_sale_df.columns:
        last_sale_df["internal_category"] = "Pharma"
    last_sale_df["internal_category"] = last_sale_df["internal_category"].fillna("Pharma")

    # Category-aware tier assignment
    last_sale_df["tier"] = last_sale_df.apply(
        lambda r: assign_tier(r["days_frozen"], r["internal_category"]),
        axis=1,
    )

    # Attach visible thresholds for transparency in the UI
    def get_thresholds(cat):
        t = get_category_tiers(cat)
        return t["WATCH"], t["ALERT"], t["DEAD"]

    threshold_cols = last_sale_df["internal_category"].apply(
        lambda c: pd.Series(get_thresholds(c), index=["watch_threshold", "alert_threshold", "dead_threshold"])
    )
    last_sale_df = pd.concat([last_sale_df, threshold_cols], axis=1)

    flagged = last_sale_df[last_sale_df["tier"].notna()].copy()

    # Join inventory
    inv_cols = ["store_id", "product_id", "qty_on_hand", "unit_cost", "total_inventory_value"]
    flagged = flagged.merge(inventory[inv_cols], on=["store_id", "product_id"], how="left")

    # Only flag where there's stock on hand
    flagged = flagged[flagged["qty_on_hand"] > 0].copy()

    flagged["ksh_at_risk"] = flagged["total_inventory_value"].fillna(
        flagged["qty_on_hand"] * flagged["unit_cost"]
    )

    # Sort by severity
    flagged["tier_order"] = flagged["tier"].map(TIER_ORDER)
    flagged = flagged.sort_values(["tier_order", "ksh_at_risk"], ascending=[False, False])
    flagged = flagged.drop(columns=["tier_order"])
    flagged["analysis_date"] = today

    return flagged.reset_index(drop=True)


def dead_stock_summary(dead_df: pd.DataFrame) -> dict:
    """
    Returns high-level KPIs for the header ticker.
    Now also breaks down frozen capital by internal_category.
    """
    by_tier = dead_df.groupby("tier")["ksh_at_risk"].agg(
        ksh_at_risk="sum", sku_count="count"
    ).to_dict("index")

    by_category = {}
    if "internal_category" in dead_df.columns:
        by_category = dead_df.groupby("internal_category")["ksh_at_risk"].agg(
            ksh_at_risk="sum", sku_count="count"
        ).to_dict("index")

    return {
        "total_frozen_ksh":   dead_df["ksh_at_risk"].sum(),
        "total_skus":         len(dead_df),
        "by_tier":            by_tier,
        "by_category":        by_category,
        "branches_affected":  dead_df["store_id"].nunique(),
    }