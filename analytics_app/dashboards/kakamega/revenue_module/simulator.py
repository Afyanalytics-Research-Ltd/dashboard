"""
simulator.py
------------
What-if scenario engine ONLY.

This module does NOT generate fake revenue data. Real revenue numbers are
queried from Snowflake via queries.py + data_layer.py. This module exists to
let business users explore "what if" levers — operational changes that would
move revenue — using sensitivity arithmetic seeded from the live Snowflake
baseline (avg daily revenue, current mix, current AR ageing).

Public API:
    simulate_levers(baseline_daily_revenue, horizon_days, levers, ar_overdue)
        -> DataFrame with the marginal and cumulative uplift per lever.

    elasticity_grid(baseline_daily_revenue, horizon_days, lever_grid)
        -> DataFrame for heatmap-style two-lever sensitivity exploration.

    payer_mix_shift(payer_perf_df, target_cash_share)
        -> projected revenue & DSO if cash share is dialled up.
"""



import numpy as np
import pandas as pd


DEFAULT_LEVERS = {
    # name -> (% uplift on horizon revenue, short rationale)
    "Reduce no-shows by 20%":               (0.045, "Recover slots via SMS reminders + deposit-on-booking"),
    "Lift ARPV via cross-sell (Pharmacy)":  (0.060, "Bundle scripts with consults; capture leakage to outside chemists"),
    "Cut M-Pesa downtime to <0.1%":         (0.012, "Add Pesapal failover; eliminates 3-day outage tail"),
    "Recover 50% of 90+ day AR":            (0.030, "Dedicated insurance follow-up team + automated chasers"),
    "Open Eldoret Sat half-day":            (0.018, "Captures weekend demand currently lost to competitors"),
    "Insurance pre-auth turnaround -1d":    (0.022, "Reduces patient drop-off at consult-to-procedure step"),
    "Tighten discount policy (-30%)":       (0.014, "Cap front-desk discretion; require manager sign-off >5%"),
}


def simulate_levers(
    baseline_daily_revenue: float,
    horizon_days: int = 90,
    levers: dict | None = None,
) -> pd.DataFrame:
    """
    Apply each lever as an independent % uplift on horizon revenue.

    Returns columns: lever, uplift_pct, uplift_value, cumulative, rationale.
    """
    if levers is None:
        levers = DEFAULT_LEVERS

    horizon_rev = baseline_daily_revenue * horizon_days
    rows, cumulative = [], horizon_rev
    for name, payload in levers.items():
        if isinstance(payload, tuple):
            pct, rationale = payload
        else:
            pct, rationale = float(payload), ""
        uplift = horizon_rev * pct
        cumulative += uplift
        rows.append({
            "lever":        name,
            "uplift_pct":   pct * 100,
            "uplift_value": round(uplift, 2),
            "cumulative":   round(cumulative, 2),
            "rationale":    rationale,
        })
    return pd.DataFrame(rows)


def elasticity_grid(
    baseline_daily_revenue: float,
    horizon_days: int = 90,
    arpv_deltas:    np.ndarray | None = None,
    volume_deltas:  np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Two-lever grid: ARPV change × visit-volume change.

    Returns long-format DataFrame for heatmap rendering with columns:
        arpv_delta_pct, volume_delta_pct, projected_revenue, uplift_pct
    """
    if arpv_deltas is None:
        arpv_deltas   = np.linspace(-0.10, 0.20, 7)        # -10% .. +20%
    if volume_deltas is None:
        volume_deltas = np.linspace(-0.10, 0.20, 7)

    horizon_rev = baseline_daily_revenue * horizon_days
    rows = []
    for a in arpv_deltas:
        for v in volume_deltas:
            projected = horizon_rev * (1 + a) * (1 + v)
            rows.append({
                "arpv_delta_pct":    round(a * 100, 1),
                "volume_delta_pct":  round(v * 100, 1),
                "projected_revenue": round(projected, 2),
                "uplift_pct":        round((projected / horizon_rev - 1) * 100, 2),
            })
    return pd.DataFrame(rows)


def payer_mix_shift(
    payer_perf: pd.DataFrame,
    target_cash_share: float = 0.45,
) -> dict:
    """
    Compute projected metrics if cash share is shifted up to `target_cash_share`.

    Cash collects in 0 days (DSO=0), insurance has its current weighted DSO.
    Returns dict with projected DSO, projected collection_rate, working_capital_freed.
    """
    df = payer_perf.copy()
    df["is_cash"] = df["payer_name"].str.contains("Cash", case=False, na=False)
    total_billed = df["billed"].sum()

    cur_cash_share = df.loc[df["is_cash"], "billed"].sum() / total_billed
    cur_weighted_dso = (df["billed"] * df["avg_dso"]).sum() / total_billed

    # New billed split
    new_cash    = total_billed * target_cash_share
    new_ins     = total_billed * (1 - target_cash_share)

    # Weighted DSO post-shift (cash=0, insurance keeps its weighted DSO)
    ins_only = df[~df["is_cash"]]
    ins_dso  = (ins_only["billed"] * ins_only["avg_dso"]).sum() / max(ins_only["billed"].sum(), 1)

    new_weighted_dso = (1 - target_cash_share) * ins_dso

    # Working capital freed = (old DSO - new DSO) * billed/365
    wc_freed = max(0, cur_weighted_dso - new_weighted_dso) * total_billed / 365

    cur_collection = df["collected"].sum() / total_billed
    new_collection = (
        target_cash_share * 1.00
        + (1 - target_cash_share) * (ins_only["collected"].sum() / max(ins_only["billed"].sum(), 1))
    )

    return {
        "current_cash_share":    cur_cash_share,
        "target_cash_share":     target_cash_share,
        "current_weighted_dso":  cur_weighted_dso,
        "projected_weighted_dso": new_weighted_dso,
        "current_collection":    cur_collection,
        "projected_collection":  new_collection,
        "working_capital_freed": wc_freed,
    }