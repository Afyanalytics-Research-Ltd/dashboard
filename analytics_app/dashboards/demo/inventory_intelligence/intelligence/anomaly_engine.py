"""
Consumption anomaly detection engine.
Compares a product's recent demand against its own facility-calibrated baseline
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from intelligence.config import (
    ANOMALY_BASELINE_DAYS,
    ANOMALY_RECENT_DAYS,
    ANOMALY_Z_THRESHOLD,
)


@dataclass
class AnomalyResult:
    product_id: str
    canonical_name: str
    is_anomaly: bool
    direction: str          # UP | DOWN | NONE
    magnitude_pct: float    # % deviation from baseline mean
    z_score: float
    baseline_avg_daily: float
    recent_avg_daily: float
    message: str


class AnomalyEngine:
    """
    Detects consumption anomalies using z-score comparison of recent demand
    against the product's own historical baseline within the same facility.
    """

    def __init__(self, facility: str) -> None:
        self.facility = facility
        self._baseline: Optional[pd.DataFrame] = None  # (product_id, date, qty)
        self._fitted = False

    def fit(self, dispensing_df: pd.DataFrame) -> "AnomalyEngine":
        """
        Establish per-product baselines from dispensing history.
        Required columns: product_id, dispensed_at, quantity_dispensed
        Optional: canonical_name
        """
        df = dispensing_df.copy()
        df.columns = df.columns.str.lower()
        df["dispensed_at"] = pd.to_datetime(df["dispensed_at"])
        df["date"] = df["dispensed_at"].dt.normalize()

        # Daily totals per product — the baseline distribution
        self._baseline = (
            df.groupby(["product_id", "date"])["quantity_dispensed"]
            .sum()
            .reset_index()
        )

        if "canonical_name" in df.columns:
            self._names = (
                df[["product_id", "canonical_name"]]
                .drop_duplicates()
                .set_index("product_id")["canonical_name"]
                .to_dict()
            )
        else:
            self._names = {}

        self._fitted = True
        return self

    def detect(self, product_id: str, as_of_date: Optional[pd.Timestamp] = None) -> Optional[AnomalyResult]:
        """
        Detect anomaly for a single product.
        Compares the last ANOMALY_RECENT_DAYS against the prior ANOMALY_BASELINE_DAYS.
        """
        if not self._fitted or self._baseline is None:
            raise RuntimeError("Call fit() before detect()")

        prod = self._baseline[self._baseline["product_id"] == product_id].copy()
        if prod.empty:
            return None

        prod = prod.sort_values("date")
        ref_date = as_of_date or prod["date"].max()

        recent_start = ref_date - pd.Timedelta(days=ANOMALY_RECENT_DAYS)
        baseline_start = ref_date - pd.Timedelta(days=ANOMALY_RECENT_DAYS + ANOMALY_BASELINE_DAYS)

        recent = prod[(prod["date"] > recent_start) & (prod["date"] <= ref_date)]
        baseline = prod[(prod["date"] > baseline_start) & (prod["date"] <= recent_start)]

        if len(baseline) < 7:
            return None

        # Fill in zero-dispense days so averages reflect true daily demand
        recent_daily = recent["quantity_dispensed"].sum() / ANOMALY_RECENT_DAYS
        baseline_vals = baseline["quantity_dispensed"].values
        baseline_daily = float(np.mean(baseline_vals))
        baseline_std = float(np.std(baseline_vals)) if len(baseline_vals) > 1 else 0.0

        if baseline_std == 0 or baseline_daily == 0:
            return None

        z = (recent_daily - baseline_daily) / baseline_std
        magnitude_pct = ((recent_daily - baseline_daily) / baseline_daily * 100
                         if baseline_daily > 0 else 0.0)

        is_anomaly = abs(z) >= ANOMALY_Z_THRESHOLD
        direction = "NONE"
        if is_anomaly:
            direction = "UP" if z > 0 else "DOWN"

        name = self._names.get(product_id, product_id)
        message = _build_message(name, is_anomaly, direction, magnitude_pct, z)

        return AnomalyResult(
            product_id=product_id,
            canonical_name=name,
            is_anomaly=is_anomaly,
            direction=direction,
            magnitude_pct=round(magnitude_pct, 1),
            z_score=round(z, 2),
            baseline_avg_daily=round(baseline_daily, 2),
            recent_avg_daily=round(recent_daily, 2),
            message=message,
        )

    def detect_all(self, as_of_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Return anomaly results for all products with sufficient history."""
        if not self._fitted or self._baseline is None:
            return pd.DataFrame()

        rows = []
        for pid in self._baseline["product_id"].unique():
            result = self.detect(pid, as_of_date)
            if result:
                rows.append(result.__dict__)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        return df[df["is_anomaly"]].sort_values("magnitude_pct", key=abs, ascending=False)


def build_anomaly_context(
    dispensing_df: pd.DataFrame,
    soh_df: pd.DataFrame,
    anomaly_row: dict,
    anomalies_df: pd.DataFrame,
    baseline_days: int = ANOMALY_BASELINE_DAYS,
    recent_days: int = ANOMALY_RECENT_DAYS,
) -> dict:
    """
    Compute enriched context for a single anomaly for use in the data panel + AI prompt.
    Returns a dict with:
      - daily_series: DataFrame(date, qty) for the full baseline+recent window
      - spike_start: date when consumption first exceeded 1.5x baseline (or None)
      - current_soh, days_at_current_rate, days_at_normal_rate: stock impact
      - correlated_drugs: other drugs in same class with same-direction anomaly
    """
    name = anomaly_row["canonical_name"]
    direction = anomaly_row.get("direction", "UP")
    baseline_avg = float(anomaly_row.get("baseline_avg_daily", 0))
    recent_avg   = float(anomaly_row.get("recent_avg_daily", 0))

    # ── Daily consumption series ───────────────────────────────────────────────
    df = dispensing_df.copy()
    df.columns = df.columns.str.lower()
    df["dispensed_at"] = pd.to_datetime(df["dispensed_at"])

    drug_df = df[df["canonical_name"] == name].copy()
    drug_df["date"] = drug_df["dispensed_at"].dt.normalize()

    # Use the facility-wide last dispensing date as ref_date, not the per-drug
    # last date. This ensures drugs that stocked out mid-period (e.g. CIPROFLOXACIN
    # last dispensed in March) show the subsequent zero-dispensing weeks on the
    # chart rather than silently truncating — giving the pharmacist the full picture.
    ref_date     = pd.to_datetime(df["dispensed_at"].max())
    window_days  = baseline_days + recent_days
    window_start = ref_date - pd.Timedelta(days=window_days)
    recent_start = ref_date - pd.Timedelta(days=recent_days)

    daily = (
        drug_df[drug_df["dispensed_at"] >= window_start]
        .groupby("date")["quantity_dispensed"]
        .sum()
        .reset_index()
    )
    daily.columns = ["date", "qty"]
    daily["date"] = pd.to_datetime(daily["date"])

    # Fill missing dates with 0 so chart is continuous
    full_range = pd.date_range(start=window_start.normalize(), end=ref_date.normalize(), freq="D")
    daily = daily.set_index("date").reindex(full_range, fill_value=0).reset_index()
    daily.columns = ["date", "qty"]

    # ── Spike start: first day in recent window where qty >= 1.5x baseline ─────
    threshold = baseline_avg * 1.5
    spike_start = None
    if baseline_avg > 0:
        recent_rows = daily[daily["date"] > recent_start]
        hits = recent_rows[recent_rows["qty"] >= threshold]
        if not hits.empty:
            spike_start = hits["date"].iloc[0]

    # ── Spike type classification ──────────────────────────────────────────────
    recent_qtys = daily[daily["date"] > recent_start]["qty"].values
    spike_type  = _classify_spike(recent_qtys, baseline_avg, direction)

    # safe_order_adc: the ADC to use for ordering, shielded from distortion.
    # TRANSIENT/DECLINING → use baseline (the spike is noise, not signal).
    # SUSTAINED           → use recent_avg (genuine demand has shifted).
    if spike_type in ("TRANSIENT", "DECLINING"):
        safe_order_adc = baseline_avg
    else:
        safe_order_adc = recent_avg

    # ── Stock impact ───────────────────────────────────────────────────────────
    _soh = soh_df.copy()
    _soh.columns = _soh.columns.str.lower()
    soh_match = _soh[_soh["canonical_name"] == name]
    current_soh = float(soh_match["current_soh"].iloc[0]) if not soh_match.empty else 0.0

    this_class = ""
    if not soh_match.empty and "therapeutic_class" in _soh.columns:
        this_class = str(soh_match["therapeutic_class"].iloc[0] or "")

    days_at_current = round(current_soh / recent_avg)    if recent_avg   > 0 else None
    days_at_normal  = round(current_soh / baseline_avg)  if baseline_avg > 0 else None

    # ── Proper order quantity (workbench formula, using safe ADC) ──────────────
    # Uses same logic as the Command Center decision cards:
    # cover (30d + lead time) at safe rate, minus current SOH.
    # safe_order_adc shields the quantity from spike distortion.
    from intelligence.config import DEFAULT_LEAD_TIME_DAYS as _LT
    proper_order_qty = max(0, int((30 + _LT) * safe_order_adc - current_soh))

    # ── Correlated spikes: same therapeutic class, same direction ─────────────
    correlated: list[str] = []
    if not anomalies_df.empty and this_class:
        _soh_cls = _soh[["canonical_name", "therapeutic_class"]].drop_duplicates()
        others = anomalies_df[
            (anomalies_df["canonical_name"] != name) &
            (anomalies_df["direction"] == direction)
        ].merge(_soh_cls, on="canonical_name", how="left")
        correlated = (
            others[others["therapeutic_class"] == this_class]["canonical_name"].tolist()[:2]
        )

    return {
        "name":                 name,
        "direction":            direction,
        "baseline_avg":         baseline_avg,
        "recent_avg":           recent_avg,
        "magnitude_pct":        float(anomaly_row.get("magnitude_pct", 0)),
        "z_score":              float(anomaly_row.get("z_score", 0)),
        "daily_series":         daily,
        "ref_date":             ref_date,
        "recent_start":         recent_start,
        "spike_start":          spike_start,
        "spike_type":           spike_type,        # TRANSIENT | DECLINING | SUSTAINED
        "safe_order_adc":       safe_order_adc,    # ADC shielded from spike distortion
        "proper_order_qty":     proper_order_qty,  # workbench formula qty at safe ADC
        "current_soh":          current_soh,
        "days_at_current_rate": days_at_current,
        "days_at_normal_rate":  days_at_normal,
        "correlated_drugs":     correlated,
        "therapeutic_class":    this_class,
        "baseline_days":        baseline_days,
        "recent_days":          recent_days,
    }


def _classify_spike(
    recent_qtys: "np.ndarray",
    baseline_avg: float,
    direction: str,
) -> str:
    """
    Classify a consumption anomaly into one of three types:

    TRANSIENT  — spike concentrated in 1–3 days then returned to baseline.
                 Likely a bulk dispensing event or data entry error.
                 → Do NOT adjust standing order. Investigate the event date.

    DECLINING  — spike is actively reversing (second half of recent window
                 substantially lower than first half).
                 → Hold order decision. Monitor for 5 more days.

    SUSTAINED  — consumption consistently elevated across the full window.
                 Likely a genuine demand shift (outbreak, new patients, protocol change).
                 → Adjust order quantity based on new demand rate.
    """
    if len(recent_qtys) < 3 or baseline_avg <= 0:
        return "SUSTAINED"

    n          = len(recent_qtys)
    mean_qty   = float(np.mean(recent_qtys)) if np.mean(recent_qtys) > 0 else 1.0
    peak       = float(np.max(recent_qtys))
    tail_avg   = float(np.mean(recent_qtys[-4:])) if n >= 4 else mean_qty
    first_half = float(np.mean(recent_qtys[: n // 2]))
    second_half= float(np.mean(recent_qtys[n // 2 :]))

    if direction == "DOWN":
        # For drops: RECOVERING means consumption trending back toward baseline
        if first_half > 0 and second_half > first_half * 1.2:
            return "DECLINING"   # recovering (we reuse DECLINING label — trend reverting)
        return "SUSTAINED"

    # UP spikes ────────────────────────────────────────────────────────────────
    # TRANSIENT: one blow-out day carries most of the volume, tail back at baseline
    if peak / mean_qty >= 3.0 and tail_avg <= baseline_avg * 1.5:
        return "TRANSIENT"

    # DECLINING: clear downward trend within the recent window (≥30% drop half-to-half)
    if first_half > 0 and second_half < first_half * 0.70:
        return "DECLINING"

    return "SUSTAINED"


def _build_message(
    name: str,
    is_anomaly: bool,
    direction: str,
    magnitude_pct: float,
    z: float,
) -> str:
    if not is_anomaly:
        return "Consumption within normal range."
    verb = "above" if direction == "UP" else "below"
    return (
        f"Consumption is {abs(magnitude_pct):.0f}% {verb} its 90-day baseline "
        f"(z={z:.1f}) — review before placing a standard order."
    )
