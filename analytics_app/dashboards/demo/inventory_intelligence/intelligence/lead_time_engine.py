"""
Lead time inference engine — dual-source.

Tenri: actual receipt timestamps from inventory_batch_purchases.
KSH:   receipt events detected from upward SOH jumps in fact_dispensing.

Both sources produce the same output: a lead-time distribution
(mean, std, n_observations) per product and facility-wide fallback.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from intelligence.config import (
    DEFAULT_LEAD_TIME_DAYS,
    MAX_LEAD_TIME_DAYS,
    MIN_LEAD_TIME_DAYS,
    MIN_LEAD_TIME_OBSERVATIONS,
)


class LeadTimeEngine:
    """
    Infers replenishment lead times from observed receipt events.
    Provides product-level estimates with fallback to facility-wide distribution.
    """

    def __init__(self, facility: str) -> None:
        self.facility = facility
        self._lead_times: Optional[pd.DataFrame] = None   # cols: product_id, lead_time_days
        self._facility_mean: float = float(DEFAULT_LEAD_TIME_DAYS)
        self._facility_std: float = float(DEFAULT_LEAD_TIME_DAYS) * 0.3
        self._fitted = False
        self._source: str = "default"

    

    # ── KSH: fit from SOH jump detection in dispensing history ───────────────

    def fit_kisumu(self, dispensing_df: pd.DataFrame) -> "LeadTimeEngine":
        """
        Detect receipt events as upward SOH jumps between consecutive dispenses.
        Lead time = days between SOH dropping to critical level and next receipt.
        Required columns: product_id, dispensed_at, soh_after_raw, soh_after_display
        """
        if dispensing_df.empty:
            return self

        tmp = dispensing_df.copy()
        tmp.columns = tmp.columns.str.lower()
        df = tmp[["product_id", "dispensed_at", "soh_after_raw", "soh_after_display"]].copy()
        df["dispensed_at"] = pd.to_datetime(df["dispensed_at"])
        df = df.sort_values(["product_id", "dispensed_at"])

        records = []
        for pid, grp in df.groupby("product_id"):
            grp = grp.reset_index(drop=True)
            grp["prev_soh"] = grp["soh_after_raw"].shift(1)

            soh_median = grp["soh_after_display"].median()
            low_threshold = max(1.0, soh_median * 0.2)
            jump_threshold = max(10.0, soh_median * 0.3)

            grp["soh_jump"] = grp["soh_after_raw"] - grp["prev_soh"].fillna(
                grp["soh_after_raw"]
            )

            low_dates = grp.loc[grp["soh_after_raw"] <= low_threshold, "dispensed_at"].tolist()
            receipt_dates = grp.loc[grp["soh_jump"] >= jump_threshold, "dispensed_at"].tolist()

            for low_date in low_dates:
                future = [r for r in receipt_dates if r > low_date]
                if future:
                    lt = (future[0] - low_date).days
                    if MIN_LEAD_TIME_DAYS <= lt <= MAX_LEAD_TIME_DAYS:
                        records.append({"product_id": pid, "lead_time_days": lt})

        if records:
            self._lead_times = pd.DataFrame(records)
            self._source = "soh_jump_detection"
            self._fitted = True
            self._compute_facility_stats()

        return self

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_lead_time(self, product_id: Optional[str] = None) -> tuple[float, float, int]:
        """
        Return (mean_days, std_days, n_observations) for a product.
        Falls back to facility-wide distribution if product has too few observations.
        Falls back to config defaults if engine not fitted.
        """
        if not self._fitted or self._lead_times is None or self._lead_times.empty:
            return self._facility_mean, self._facility_std, 0

        if product_id is not None:
            prod = self._lead_times[
                self._lead_times["product_id"] == product_id
            ]["lead_time_days"]
            if len(prod) >= MIN_LEAD_TIME_OBSERVATIONS:
                return float(prod.mean()), float(prod.std()), len(prod)

        return self._facility_mean, self._facility_std, len(self._lead_times)

    def get_all_product_lead_times(self) -> pd.DataFrame:
        """DataFrame with product-level lead time stats (product_id, mean, std, n)."""
        if not self._fitted or self._lead_times is None:
            return pd.DataFrame(columns=["product_id", "lt_mean", "lt_std", "lt_n"])

        stats = (
            self._lead_times.groupby("product_id")["lead_time_days"]
            .agg(lt_mean="mean", lt_std="std", lt_n="count")
            .reset_index()
        )
        stats["lt_std"] = stats["lt_std"].fillna(stats["lt_mean"] * 0.3)
        return stats

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute_facility_stats(self) -> None:
        if self._lead_times is not None and not self._lead_times.empty:
            vals = self._lead_times["lead_time_days"]
            self._facility_mean = float(vals.mean())
            self._facility_std = float(vals.std()) if len(vals) > 1 else self._facility_mean * 0.3
