"""
Per-facility demand forecasting engine.
Trained exclusively on the facility's own dispensing history — no cross-facility training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from intelligence.config import (
    EWM_SPAN_MAX,
    EWM_SPAN_MIN,
    MIN_DAYS_FOR_FORECAST,
    MIN_MONTHS_HIGH_CONFIDENCE,
    MIN_MONTHS_MEDIUM_CONFIDENCE,
    MIN_DAYS_HIGH_CONFIDENCE,
    MIN_DAYS_MEDIUM_CONFIDENCE,
    SB_ADI_THRESHOLD,
    SB_CV2_THRESHOLD,
    TREND_THRESHOLD,
    DOS_CRITICAL,
    DOS_LOW,
)


@dataclass
class DemandForecast:
    product_id: str
    canonical_name: str
    avg_daily_units: float
    std_daily_units: float
    cv: float                  # std / ewm_avg over zero-filled series (used in SS formula)
    demand_type: str           # SMOOTH | ERRATIC | INTERMITTENT | LUMPY (Syntetos-Boylan)
    adi: float                 # average demand interval — calendar days per non-zero dispense
    cv_nz: float               # CV of non-zero dispensing quantities only
    forecast_30d: float
    forecast_60d: float
    forecast_90d: float
    ci_lower_30d: float
    ci_upper_30d: float
    confidence: str            # HIGH | MEDIUM | LOW
    data_months: int
    seasonality_detected: bool
    trend_direction: str       # UP | DOWN | STABLE


def confidence_score(n_days: int, demand_type: str) -> str:
    """
    Derive forecast confidence tier from data sufficiency and Syntetos-Boylan demand type.

    HIGH:   SMOOTH  + ≥90 days
    MEDIUM: SMOOTH  + 30–89 days
            ERRATIC + ≥30 days  (frequent demand — EWM trend is still useful)
            INTERMITTENT + ≥30 days  (consistent quantity per event)
    LOW:    LUMPY   (infrequent AND variable — genuinely unpredictable regardless of history)
            Any type + <30 days  (insufficient history)

    Parameters
    ----------
    n_days      : int — number of calendar days spanned by dispensing history
    demand_type : str — SMOOTH | ERRATIC | INTERMITTENT | LUMPY
    """
    if n_days < MIN_DAYS_MEDIUM_CONFIDENCE:
        return "LOW"
    if demand_type == "LUMPY":
        return "LOW"
    if demand_type == "SMOOTH" and n_days >= MIN_DAYS_HIGH_CONFIDENCE:
        return "HIGH"
    return "MEDIUM"


class DemandEngine:
    """
    Facility-isolated demand forecasting via exponential weighted moving averages.
    Adapts EWM span to data availability; derives calibrated thresholds from
    the facility's own observed operating patterns.
    """

    def __init__(self, facility: str) -> None:
        self.facility = facility
        self._daily: Optional[pd.DataFrame] = None
        self._monthly: Optional[pd.DataFrame] = None
        self._fitted = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, dispensing_df: pd.DataFrame) -> "DemandEngine":
        """
        Train on facility dispensing records.
        Required columns: product_id, dispensed_at, quantity_dispensed
        """
        df = dispensing_df.copy()
        df.columns = df.columns.str.lower()
        df["dispensed_at"] = pd.to_datetime(df["dispensed_at"])
        df["date"] = df["dispensed_at"].dt.normalize()

        self._daily = (
            df.groupby(["product_id", "date"])["quantity_dispensed"]
            .sum()
            .reset_index()
        )

        df["month"] = df["dispensed_at"].dt.to_period("M")
        self._monthly = (
            df.groupby(["product_id", "month"])["quantity_dispensed"]
            .sum()
            .reset_index()
            .rename(columns={"quantity_dispensed": "units"})
        )

        self._fitted = True
        return self

    # ── Forecasting ───────────────────────────────────────────────────────────

    def forecast(self, product_id: str, canonical_name: str = "") -> Optional[DemandForecast]:
        if not self._fitted:
            raise RuntimeError("Call fit() before forecast()")

        prod = self._daily[self._daily["product_id"] == product_id].copy()
        if prod.empty or len(prod) < MIN_DAYS_FOR_FORECAST:
            return None

        # Fill date gaps so EWM sees a continuous series
        idx = pd.date_range(prod["date"].min(), prod["date"].max(), freq="D")
        prod = (
            prod.set_index("date")["quantity_dispensed"]
            .reindex(idx, fill_value=0)
            .reset_index()
        )
        prod.columns = ["date", "quantity_dispensed"]

        units = prod["quantity_dispensed"].values
        n = len(units)
        data_months = max(1, n // 30)

        # Adaptive span: more history → shorter span (more reactive to recent patterns)
        span = max(EWM_SPAN_MIN, min(EWM_SPAN_MAX, n // 4))
        ewm_avg = float(pd.Series(units).ewm(span=span, adjust=False).mean().iloc[-1])
        ewm_avg = max(0.0, ewm_avg)

        # std over zero-filled daily series — correct denominator for safety stock formula
        std = float(pd.Series(units).std()) if len(units) > 1 else ewm_avg * 0.3
        std = max(std, 0.0)
        cv  = std / ewm_avg if ewm_avg > 0 else 0.0

        # ── Syntetos-Boylan demand classification ────────────────────────────
        # ADI on daily data: avg calendar days between non-zero dispensing events
        nonzero_daily = units[units > 0]
        n_nonzero = len(nonzero_daily)
        if n_nonzero >= 2:
            adi = n / n_nonzero
        else:
            adi = float(n)

        # CV_nz on weekly aggregated quantities: absorbs prescription batch-size noise
        # (30/90-day supplies inflate daily CV without reflecting true demand variability)
        weekly_qty = prod.set_index("date")["quantity_dispensed"].resample("W").sum()
        weekly_nonzero = weekly_qty[weekly_qty > 0]
        if len(weekly_nonzero) >= 2 and weekly_nonzero.mean() > 0:
            cv_nz = float(weekly_nonzero.std() / weekly_nonzero.mean())
        else:
            cv_nz = 0.0
        cv2_nz = cv_nz ** 2

        if adi < SB_ADI_THRESHOLD:
            demand_type = "ERRATIC" if cv2_nz >= SB_CV2_THRESHOLD else "SMOOTH"
        else:
            demand_type = "LUMPY"   if cv2_nz >= SB_CV2_THRESHOLD else "INTERMITTENT"

        # Trend detection: last 30d mean vs prior 30d mean
        trend = "STABLE"
        if n >= 60:
            r30, p30 = np.mean(units[-30:]), np.mean(units[-60:-30])
            if p30 > 0:
                chg = (r30 - p30) / p30
                if chg > TREND_THRESHOLD:
                    trend = "UP"
                elif chg < -TREND_THRESHOLD:
                    trend = "DOWN"

        mult = {"UP": 1.10, "DOWN": 0.95, "STABLE": 1.00}[trend]
        f30 = max(0.0, ewm_avg * 30 * mult)
        f60 = max(0.0, ewm_avg * 60 * mult)
        f90 = max(0.0, ewm_avg * 90 * mult)

        z = 1.645  # 95% CI
        ci_lo = max(0.0, f30 - z * std * np.sqrt(30))
        ci_hi = f30 + z * std * np.sqrt(30)

        confidence = confidence_score(n_days=n, demand_type=demand_type)

        # Seasonality: high monthly CV (>25%) indicates seasonal variation
        seasonality = False
        if self._monthly is not None:
            pm = self._monthly[self._monthly["product_id"] == product_id]
            if len(pm) >= 6 and pm["units"].mean() > 0:
                seasonality = (pm["units"].std() / pm["units"].mean()) > 0.25

        return DemandForecast(
            product_id=product_id,
            canonical_name=canonical_name,
            avg_daily_units=round(ewm_avg, 3),
            std_daily_units=round(std, 3),
            cv=round(cv, 3),
            demand_type=demand_type,
            adi=round(adi, 2),
            cv_nz=round(cv_nz, 3),
            forecast_30d=round(f30, 1),
            forecast_60d=round(f60, 1),
            forecast_90d=round(f90, 1),
            ci_lower_30d=round(ci_lo, 1),
            ci_upper_30d=round(ci_hi, 1),
            confidence=confidence,
            data_months=data_months,
            seasonality_detected=seasonality,
            trend_direction=trend,
        )

    def forecast_all(self, dispensing_df: pd.DataFrame) -> pd.DataFrame:
        """Return one forecast row per product."""
        df_lower = dispensing_df.copy()
        df_lower.columns = df_lower.columns.str.lower()
        if "canonical_name" in df_lower.columns:
            meta = df_lower[["product_id", "canonical_name"]].drop_duplicates()
        else:
            meta = df_lower[["product_id"]].drop_duplicates().assign(canonical_name="")

        rows = []
        for _, row in meta.iterrows():
            raw_name = row.get("canonical_name", "")
            name = (
                str(raw_name).strip()
                if raw_name is not None and pd.notna(raw_name) and str(raw_name).strip()
                else str(row.get("product_id", ""))
            )
            f = self.forecast(row["product_id"], name)
            if f:
                rows.append(f.__dict__)
        return pd.DataFrame(rows)

    # ── Calibration ───────────────────────────────────────────────────────────

    def get_calibrated_thresholds(self, soh_df: pd.DataFrame) -> dict:
        """
        Derive facility-specific DOS alert thresholds from its own operating patterns.
        Uses p10 and p25 of observed DOS distribution as the critical/low boundaries.
        Falls back to global config defaults if data is insufficient.
        """
        defaults = {"critical": DOS_CRITICAL, "low": DOS_LOW}
        df = soh_df.copy()
        df.columns = df.columns.str.lower()
        if df.empty or "days_of_stock" not in df.columns:
            return defaults

        active = df.dropna(subset=["days_of_stock"])
        active = active[active["days_of_stock"] > 0]
        if len(active) < 10:
            return defaults

        p10 = float(np.percentile(active["days_of_stock"], 10))
        p25 = float(np.percentile(active["days_of_stock"], 25))

        return {
            "critical": max(DOS_CRITICAL, round(p10)),
            "low": max(DOS_LOW, round(p25)),
        }
