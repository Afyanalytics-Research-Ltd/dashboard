"""
engine/predictive.py

Proactive rebalancing — flags SKUs decelerating at one branch while
accelerating at another, BEFORE they hit the dead stock threshold.

Quality filters applied before surfacing a flag:
  - Source velocity score > 20 (product actually moves at source)
  - Source trend < -35% (meaningful deceleration, not noise)
  - Destination velocity score > 40 (real demand at destination)
  - Destination trend = accelerating only
  - Product must be in top 40% of KSh value at source branch
    (focus on high-value items worth acting on)
  - One flag per product — best destination only

Result: a short, high-confidence list of 20-60 flags rather than 1,000+
"""

import pandas as pd
from data.loader import BRANCH_NAMES


def flag_proactive_transfers(
    trend:                   pd.DataFrame,
    velocity:                pd.DataFrame,
    dead_stock_product_ids:  set,
    inventory:               pd.DataFrame = None,
    min_source_velocity:     float = 20.0,
    min_dest_velocity:       float = 40.0,
    decel_threshold:         float = -0.35,   # -35% — meaningful, not noise
    max_flags:               int   = 60,
) -> pd.DataFrame:
    """
    Returns proactive redistribution recommendations.
    One row per (source_branch, product) — best destination only.
    """
    # Exclude already-dead SKUs
    trend_active = trend[~trend["product_id"].isin(dead_stock_product_ids)].copy()

    # Decelerating at source: trend < -35%, velocity > 20
    vel_lookup = velocity.set_index(["store_id", "product_id"])

    def _get_vel(store_id, product_id, col):
        try:
            return float(vel_lookup.loc[(store_id, product_id), col])
        except (KeyError, TypeError):
            return 0.0

    trend_active["src_vel"] = trend_active.apply(
        lambda r: _get_vel(r["store_id"], r["product_id"], "velocity_score"), axis=1
    )

    decelerating = trend_active[
        (trend_active["trend_label"] == "decelerating") &
        (trend_active["trend_pct"] <= decel_threshold) &
        (trend_active["src_vel"] >= min_source_velocity)
    ][["store_id", "product_id", "trend_pct", "src_vel"]].copy()

    # Accelerating at destination: trend = accelerating, velocity > 40
    accelerating = trend_active[
        trend_active["trend_label"] == "accelerating"
    ][["store_id", "product_id", "trend_pct"]].copy()
    accelerating["dest_vel"] = accelerating.apply(
        lambda r: _get_vel(r["store_id"], r["product_id"], "velocity_score"), axis=1
    )
    accelerating = accelerating[accelerating["dest_vel"] >= min_dest_velocity]

    if decelerating.empty or accelerating.empty:
        return pd.DataFrame()

    # Cross-join on product_id
    merged = decelerating.merge(
        accelerating,
        on="product_id",
        suffixes=("_source", "_dest"),
    )
    merged = merged[merged["store_id_source"] != merged["store_id_dest"]]

    if merged.empty:
        return pd.DataFrame()

    # KSh value filter — only flag high-value items if inventory provided
    if inventory is not None and not inventory.empty:
        inv_val = inventory.groupby(["store_id", "product_id"])["total_inventory_value"].sum()
        merged["inv_ksh"] = merged.apply(
            lambda r: float(inv_val.get((int(r["store_id_source"]), r["product_id"]), 0)),
            axis=1,
        )
        # Keep top 40% by value
        threshold = merged["inv_ksh"].quantile(0.60)
        merged = merged[merged["inv_ksh"] >= threshold]

    if merged.empty:
        return pd.DataFrame()

    # Keep best destination per (source, product): highest dest velocity
    merged = (
        merged.sort_values("dest_vel", ascending=False)
        .drop_duplicates(subset=["store_id_source", "product_id"])
        .reset_index(drop=True)
    )

    rows = []
    for _, row in merged.iterrows():
        src  = int(row["store_id_source"])
        dest = int(row["store_id_dest"])
        pid  = row["product_id"]

        rows.append({
            "source_store_id":      src,
            "source_branch_name":   BRANCH_NAMES.get(src, f"Branch {src}"),
            "dest_store_id":        dest,
            "dest_branch_name":     BRANCH_NAMES.get(dest, f"Branch {dest}"),
            "product_id":           pid,
            "source_trend_pct":     round(float(row["trend_pct_source"]) * 100, 1),
            "dest_trend_pct":       round(float(row["trend_pct_dest"])   * 100, 1),
            "source_velocity_score":round(float(row["src_vel"]),  1),
            "dest_velocity_score":  round(float(row["dest_vel"]), 1),
            "urgency":              "high" if row["trend_pct_source"] < -0.50 else "medium",
            "inv_ksh":              round(float(row.get("inv_ksh", 0)), 2),
            "flag_type":            "proactive",
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["urgency", "source_trend_pct"], ascending=[True, True])
    return df.head(max_flags).reset_index(drop=True)