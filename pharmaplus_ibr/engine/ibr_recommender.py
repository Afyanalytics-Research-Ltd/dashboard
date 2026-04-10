"""
engine/ibr_recommender.py

Decision engine for dead stock recovery.
Makes ONE recommendation per frozen SKU — not a list of options.

Decision logic:
    1. MARKDOWN   — price is materially above market (>15% SKU match, >25% benchmark)
                    Action: mark down to market price. No transfer.
    2. TRANSFER   — demand-driven freeze, destination branch has genuine velocity (>40)
                    and trend is stable or accelerating.
                    Action: transfer with full logistics detail.
    3. BUNDLE     — no single destination has enough velocity, but cumulative demand
                    across branches justifies a split transfer + promotional bundle.
    4. REVIEW     — no viable action. Flag for write-off or supplier return negotiation.

Recovery estimates are conservative:
    Transfer:  65% of gross inventory value (transit loss + unsold residual)
    Markdown:  80% of gross inventory value (discount + clearance cost)
    Bundle:    55% of gross inventory value
"""

import pandas as pd
import numpy as np
from itertools import groupby
from data.loader import BRANCH_NAMES
from engine.price_signal import _is_benchmark_unreliable

BASE_TRANSIT_KSH = 1500
COST_PER_KM      = 8.0

BRANCH_DISTANCES = {
    (1, 2): 15,  (1, 3): 12,  (1, 4): 490, (1, 5): 25,
    (2, 3): 6,   (2, 4): 495, (2, 5): 12,
    (3, 4): 490, (3, 5): 18,
    (4, 5): 500,
}
_d = {}
for (a, b), dist in BRANCH_DISTANCES.items():
    _d[(a, b)] = dist
    _d[(b, a)] = dist
BRANCH_DISTANCES = _d

TRANSIT_DAYS = {
    (1, 2): 1, (1, 3): 1, (1, 4): 2, (1, 5): 1,
    (2, 3): 1, (2, 4): 2, (2, 5): 1,
    (3, 4): 2, (3, 5): 1, (4, 5): 2,
}
for (a, b), d in list(TRANSIT_DAYS.items()):
    TRANSIT_DAYS[(b, a)] = d

SAFETY_MARGIN_DAYS = 14

# Recovery rate assumptions
RECOVERY_RATE = {
    "TRANSFER": 0.65,
    "MARKDOWN": 0.80,
    "BUNDLE":   0.55,
    "REVIEW":   0.10,
}

# Minimum velocity score for a destination to qualify as a transfer target
MIN_DEST_VELOCITY = 40.0

# Price signal thresholds for markdown recommendation
PRICE_DRIVEN_PCT_SKU_MATCH  = 15.0   # >15% above market on direct SKU match
PRICE_DRIVEN_PCT_BENCHMARK  = 25.0   # >25% above market on category benchmark


def _transit_cost(src: int, dest: int) -> float:
    dist = BRANCH_DISTANCES.get((src, dest), 150)
    return BASE_TRANSIT_KSH + dist * COST_PER_KM


def _transit_days(src: int, dest: int) -> int:
    return TRANSIT_DAYS.get((src, dest), TRANSIT_DAYS.get((dest, src), 2))


def _shelf_viable(days_frozen: int, shelf_life_days: int, t_days: int) -> str:
    remaining = shelf_life_days - t_days
    if remaining >= SAFETY_MARGIN_DAYS * 3:
        return "safe"
    elif remaining >= SAFETY_MARGIN_DAYS:
        return "borderline"
    return "do_not_move"


def _days_to_clear(qty: float, avg_daily: float):
    if avg_daily <= 0:
        return None
    return round(qty / avg_daily, 1)


def _is_price_driven(ps_row) -> tuple[bool, float]:
    """
    Returns (is_price_driven, pct_above_market).
    Uses stricter threshold for category benchmark estimates.
    """
    if ps_row is None:
        return False, 0.0
    signal = ps_row.get("price_signal", "unknown")
    pct    = float(ps_row.get("price_vs_market_pct") or 0)
    method = ps_row.get("match_method", "")
    threshold = PRICE_DRIVEN_PCT_SKU_MATCH if method == "sku_match" else PRICE_DRIVEN_PCT_BENCHMARK
    return (signal == "above" and pct >= threshold), pct


def _markdown_price(current_price: float, pct_above: float) -> float:
    """Target markdown price = market price with a 5% undercut to drive clearance."""
    if pct_above <= 0 or current_price <= 0:
        return current_price
    market_price = current_price / (1 + pct_above / 100)
    return round(market_price * 0.95, 2)


def build_recommendations(
    dead_stock:    pd.DataFrame,
    velocity:      pd.DataFrame,
    trend:         pd.DataFrame,
    price_signal:  pd.DataFrame = None,
    min_velocity_score: float   = MIN_DEST_VELOCITY,
) -> pd.DataFrame:
    """
    Makes one recommendation per (store_id, product_id).

    Returns one row per SKU with columns:
        source_store_id, source_branch_name,
        product_id, product_name, category_name, internal_category,
        tier, days_frozen, qty_on_hand, ksh_at_risk,
        recommendation_type  (TRANSFER | MARKDOWN | BUNDLE | REVIEW)
        dest_store_id, dest_branch_name,       ← TRANSFER/BUNDLE only
        transit_cost_ksh, transit_days,         ← TRANSFER only
        shelf_viability,                         ← TRANSFER only
        dest_velocity_score, dest_trend_label,  ← TRANSFER only
        days_to_clear_at_dest,                  ← TRANSFER only
        markdown_target_price,                  ← MARKDOWN only
        pct_above_market,                        ← MARKDOWN only
        estimated_recovery_ksh,
        recovery_rationale,
        rank (1 = best destination, for TRANSFER only)
    """
    if dead_stock.empty:
        return pd.DataFrame()

    vel_lookup   = velocity.set_index(["store_id", "product_id"])
    trend_lookup = trend.set_index(["store_id", "product_id"])

    # Price signal lookup
    ps_lookup = {}
    if price_signal is not None and not price_signal.empty:
        ps_lookup = price_signal.set_index("product_id").to_dict("index")

    all_branches = list(
        set(dead_stock["store_id"].unique()) | set(velocity["store_id"].unique())
    )
    dead_pairs = set(zip(dead_stock["store_id"], dead_stock["product_id"]))

    rows = []

    for _, item in dead_stock.iterrows():
        src         = int(item["store_id"])
        pid         = item["product_id"]
        qty         = float(item["qty_on_hand"])
        ksh         = float(item["ksh_at_risk"])
        frozen_days = int(item["days_frozen"])
        shelf_life  = float(item.get("shelf_life_days", 730))
        pp_price    = float(item.get("unit_cost", 0) or 0) * 1.3  # approx retail from cost

        ps_row      = ps_lookup.get(pid)
        price_driven, pct_above = _is_price_driven(ps_row)

        # If the price signal is based on a category benchmark AND the product
        # is in a category where benchmarks are unreliable (luxury, premium devices),
        # do not suggest a markdown — we don't have enough data to set a target price
        if price_driven and ps_row:
            method = ps_row.get("match_method", "")
            if method == "category_benchmark" and _is_benchmark_unreliable(
                item.get("product_name", "")
            ):
                price_driven = False  # fall through to transfer/review logic

        base_row = {
            "source_store_id":    src,
            "source_branch_name": BRANCH_NAMES.get(src, f"Branch {src}"),
            "product_id":         pid,
            "product_name":       item.get("product_name", str(pid)),
            "category_name":      item.get("category_name", "—"),
            "internal_category":  item.get("internal_category", "Pharma"),
            "tier":               item["tier"],
            "days_frozen":        frozen_days,
            "qty_on_hand":        int(qty),
            "ksh_at_risk":        round(ksh, 2),
            "pct_above_market":   round(pct_above, 1),
            "price_signal":       ps_row.get("price_signal", "unknown") if ps_row else "unknown",
            "freeze_hypothesis":  ps_row.get("freeze_hypothesis", "unknown") if ps_row else "unknown",
            "competitor_promo":   bool(ps_row.get("competitor_promo_active", False)) if ps_row else False,
            "primary_competitor": ps_row.get("primary_competitor", "") if ps_row else "",
            "match_method":       ps_row.get("match_method", "") if ps_row else "",
        }

        # ── STEP 1: Markdown if price-driven ─────────────────────────────────
        if price_driven:
            pp_price_actual = float(ps_row.get("pharmaplus_price_kes") or pp_price or 0)
            target_price    = _markdown_price(pp_price_actual, pct_above)
            recovery        = round(ksh * RECOVERY_RATE["MARKDOWN"], 2)
            rows.append({
                **base_row,
                "recommendation_type":  "MARKDOWN",
                "dest_store_id":        src,
                "dest_branch_name":     "Same branch",
                "transit_cost_ksh":     0.0,
                "transit_days":         0,
                "shelf_viability":      "safe",
                "dest_velocity_score":  0.0,
                "dest_avg_daily_units": 0.0,
                "dest_trend_label":     "—",
                "days_to_clear_at_dest":None,
                "markdown_target_price":target_price,
                "estimated_recovery_ksh": recovery,
                "recovery_rationale":   f"Mark down from KSh {pp_price_actual:,.0f} to KSh {target_price:,.0f} to match market",
                "low_roi":              False,
                "rank":                 1,
            })
            continue

        # ── STEP 2: Find transfer candidates ─────────────────────────────────
        candidates = []
        for dest in all_branches:
            if dest == src or (dest, pid) in dead_pairs:
                continue
            try:
                vel_row   = vel_lookup.loc[(dest, pid)]
                v_score   = float(vel_row["velocity_score"])
                avg_daily = float(vel_row["avg_daily_units"])
            except KeyError:
                continue

            if v_score < min_velocity_score:
                continue

            try:
                t_label = str(trend_lookup.loc[(dest, pid), "trend_label"])
            except KeyError:
                t_label = "stable"

            # Never transfer to decelerating destinations
            if t_label == "decelerating":
                continue

            t_cost    = _transit_cost(src, dest)
            t_days    = _transit_days(src, dest)
            viability = _shelf_viable(frozen_days, shelf_life, t_days)

            if viability == "do_not_move":
                continue

            net_saving = ksh - t_cost
            if net_saving <= 0:
                continue

            candidates.append({
                "dest_store_id":        dest,
                "dest_branch_name":     BRANCH_NAMES.get(dest, f"Branch {dest}"),
                "transit_cost_ksh":     round(t_cost, 2),
                "transit_days":         t_days,
                "shelf_viability":      viability,
                "dest_velocity_score":  round(v_score, 1),
                "dest_avg_daily_units": round(avg_daily, 3),
                "dest_trend_label":     t_label,
                "days_to_clear_at_dest": _days_to_clear(qty, avg_daily),
                "net_saving_ksh":       round(net_saving, 2),
            })

        # ── STEP 3: Best transfer candidate ──────────────────────────────────
        if candidates:
            # Rank: accelerating destinations first, then by velocity
            candidates.sort(key=lambda x: (
                0 if x["dest_trend_label"] == "accelerating" else 1,
                -x["dest_velocity_score"],
            ))

            best = candidates[0]
            recovery = round(ksh * RECOVERY_RATE["TRANSFER"], 2)
            rows.append({
                **base_row,
                "recommendation_type":  "TRANSFER",
                **best,
                "markdown_target_price": np.nan,
                "estimated_recovery_ksh": recovery,
                "recovery_rationale":   (
                    f"Transfer {int(qty)} units to {best['dest_branch_name']} "
                    f"· {best['dest_trend_label']} demand · clears in "
                    f"{best['days_to_clear_at_dest'] or '?'}d"
                ),
                "low_roi":  best["net_saving_ksh"] < (ksh * 0.3),
                "rank":     1,
            })
            continue

        # ── STEP 4: No single good destination — try bundle across branches ──
        # Bundle = split transfer to 2+ branches if combined velocity justifies it
        bundle_branches = []
        for dest in all_branches:
            if dest == src or (dest, pid) in dead_pairs:
                continue
            try:
                vel_row   = vel_lookup.loc[(dest, pid)]
                v_score   = float(vel_row["velocity_score"])
                avg_daily = float(vel_row["avg_daily_units"])
            except KeyError:
                continue
            # Lower bar for bundle — velocity > 15
            if v_score < 15:
                continue
            try:
                t_label = str(trend_lookup.loc[(dest, pid), "trend_label"])
            except KeyError:
                t_label = "stable"
            if t_label == "decelerating":
                continue
            bundle_branches.append((dest, v_score, avg_daily))

        if len(bundle_branches) >= 2:
            bundle_branches.sort(key=lambda x: -x[1])
            top_two   = bundle_branches[:2]
            dest_names = " + ".join(BRANCH_NAMES.get(b[0], str(b[0])).replace("PharmaPlus ", "") for b in top_two)
            recovery  = round(ksh * RECOVERY_RATE["BUNDLE"], 2)
            rows.append({
                **base_row,
                "recommendation_type":  "BUNDLE",
                "dest_store_id":        top_two[0][0],
                "dest_branch_name":     dest_names,
                "transit_cost_ksh":     round(_transit_cost(src, top_two[0][0]) * 1.5, 2),
                "transit_days":         max(_transit_days(src, b[0]) for b in top_two),
                "shelf_viability":      "safe",
                "dest_velocity_score":  round(sum(b[1] for b in top_two) / 2, 1),
                "dest_avg_daily_units": round(sum(b[2] for b in top_two), 3),
                "dest_trend_label":     "stable",
                "days_to_clear_at_dest": _days_to_clear(qty, sum(b[2] for b in top_two)),
                "markdown_target_price": np.nan,
                "estimated_recovery_ksh": recovery,
                "recovery_rationale":   f"Split transfer across {dest_names} — no single branch has sufficient demand",
                "low_roi":              False,
                "rank":                 1,
            })
            continue

        # ── STEP 5: No viable action — flag for review ───────────────────────
        recovery = round(ksh * RECOVERY_RATE["REVIEW"], 2)
        rows.append({
            **base_row,
            "recommendation_type":  "REVIEW",
            "dest_store_id":        src,
            "dest_branch_name":     "—",
            "transit_cost_ksh":     0.0,
            "transit_days":         0,
            "shelf_viability":      "—",
            "dest_velocity_score":  0.0,
            "dest_avg_daily_units": 0.0,
            "dest_trend_label":     "—",
            "days_to_clear_at_dest": None,
            "markdown_target_price": np.nan,
            "estimated_recovery_ksh": recovery,
            "recovery_rationale":   "No branch has demand · consider write-off or supplier return",
            "low_roi":              True,
            "rank":                 1,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("estimated_recovery_ksh", ascending=False)
    return df.reset_index(drop=True)


def recommendations_summary(recs: pd.DataFrame) -> dict:
    if recs.empty:
        return {
            "total_recoverable_ksh": 0,
            "transfer_ksh": 0, "transfer_count": 0,
            "markdown_ksh": 0, "markdown_count": 0,
            "bundle_ksh":   0, "bundle_count":   0,
            "review_count": 0,
        }

    by_type = recs.groupby("recommendation_type")["estimated_recovery_ksh"].agg(
        ksh="sum", count="count"
    ).to_dict("index")

    def _get(t, k): return by_type.get(t, {}).get(k, 0)

    return {
        "total_recoverable_ksh": recs["estimated_recovery_ksh"].sum(),
        "transfer_ksh":    _get("TRANSFER", "ksh"),
        "transfer_count":  _get("TRANSFER", "count"),
        "markdown_ksh":    _get("MARKDOWN", "ksh"),
        "markdown_count":  _get("MARKDOWN", "count"),
        "bundle_ksh":      _get("BUNDLE",   "ksh"),
        "bundle_count":    _get("BUNDLE",   "count"),
        "review_count":    int(_get("REVIEW", "count")),
    }