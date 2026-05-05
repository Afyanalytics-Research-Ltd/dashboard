"""
whatif.py
---------
What-if scenario engine. Real revenue numbers are queried from Snowflake
via queries.py + data_layer.py — this module only computes operational
sensitivity arithmetic on top of those live baselines.
"""



import numpy as np
import pandas as pd


DEFAULT_LEVERS = {
    "Reduce no-shows by 20%":               (0.045, "SMS reminders + deposit-on-booking"),
    "Lift ARPV via cross-sell (Pharmacy)":  (0.060, "Bundle scripts with consults"),
    "Cut M-Pesa downtime to <0.1%":         (0.012, "Pesapal failover"),
    "Recover 50% of 90+ day AR":            (0.030, "Dedicated insurance follow-up"),
    "Open Eldoret Sat half-day":            (0.018, "Captures lost weekend demand"),
    "Insurance pre-auth turnaround -1d":    (0.022, "Reduces consult-to-procedure drop-off"),
    "Tighten discount policy (-30%)":       (0.014, "Manager sign-off above 5%"),
}


def simulate_levers(baseline_daily_revenue, horizon_days=90, levers=None):
    if levers is None:
        levers = DEFAULT_LEVERS
    horizon_rev = baseline_daily_revenue * horizon_days
    rows, cumulative = [], horizon_rev
    for name, payload in levers.items():
        pct, rationale = payload if isinstance(payload, tuple) else (float(payload), "")
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


def elasticity_grid(baseline_daily_revenue, horizon_days=90,
                    arpv_deltas=None, volume_deltas=None):
    if arpv_deltas   is None: arpv_deltas   = np.linspace(-0.10, 0.20, 7)
    if volume_deltas is None: volume_deltas = np.linspace(-0.10, 0.20, 7)
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


def payer_mix_shift(payer_perf: pd.DataFrame, target_cash_share: float = 0.45) -> dict:
    df = payer_perf.copy()
    df["is_cash"] = df["payer_name"].str.contains("Cash", case=False, na=False)
    total_billed = df["billed"].sum()
    cur_cash_share   = df.loc[df["is_cash"], "billed"].sum() / total_billed
    cur_weighted_dso = (df["billed"] * df["avg_dso"]).sum() / total_billed
    ins_only = df[~df["is_cash"]]
    ins_dso  = (ins_only["billed"] * ins_only["avg_dso"]).sum() / max(ins_only["billed"].sum(), 1)
    new_weighted_dso = (1 - target_cash_share) * ins_dso
    wc_freed = max(0, cur_weighted_dso - new_weighted_dso) * total_billed / 365
    cur_collection = df["collected"].sum() / total_billed
    new_collection = (
        target_cash_share * 1.00
        + (1 - target_cash_share) * (ins_only["collected"].sum() / max(ins_only["billed"].sum(), 1))
    )
    return {
        "current_cash_share":     cur_cash_share,
        "target_cash_share":      target_cash_share,
        "current_weighted_dso":   cur_weighted_dso,
        "projected_weighted_dso": new_weighted_dso,
        "current_collection":     cur_collection,
        "projected_collection":   new_collection,
        "working_capital_freed":  wc_freed,
    }