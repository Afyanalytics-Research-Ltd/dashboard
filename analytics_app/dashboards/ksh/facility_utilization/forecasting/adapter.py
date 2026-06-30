"""
Forecast contract adapter — KSH Prophet.

Reads the raw daily forecast_cache.json produced by the Django retrain pipeline
and emits the unified forecast contract consumed by dashboard.py Page 6.

Responsibilities:
  - Aggregate daily Prophet rows to monthly grain (sum of point estimates).
  - Surface wmape + coverage metrics stored in the cache at retrain time.
  - Emit contract_version, model_source, model_family, grain — so the dashboard
    is a pure renderer with no model-identification logic.

CI note:
  Monthly confidence intervals are NOT computed. Daily yhat_lower/yhat_upper
  are not additively coherent at monthly aggregation (requires variance, which
  Prophet does not expose cleanly). low_90/high_90 are set to None. The
  renderer must not interpolate or substitute.

Contract schema (contract_version=1):
  {
    "contract_version": 1,
    "model_source":     "prophet_ksh",
    "model_family":     "prophet",
    "model_philosophy": "probabilistic",
    "generated_at":     str (ISO-8601 UTC, from cache),
    "model_version":    str,
    "grain":            "monthly",
    "metrics": {
      "wmape":    float | None,   # weighted MAPE %, from retrain evaluation
      "coverage": float | None,   # interval coverage %, from retrain evaluation
      "mape":     None
    },
    "model_context": {
      "grain_warning":      str,
      "comparability_note": str
    },
    "forecast": [
      {
        "facility":      "KSH",
        "ward":          "Facility",
        "forecast_month": "YYYY-MM-DD",  # first day of month
        "month_offset":  int,            # 1 = next calendar month
        "point":         float,          # sum of daily point estimates
        "low_90":        None,
        "high_90":       None
      }, ...
    ]
  }
"""

from __future__ import annotations

import calendar as _calendar
import json
from datetime import datetime, timezone, date as _date
from pathlib import Path

import pandas as pd

_CONTRACT_VERSION = 1
_MODEL_SOURCE     = "prophet_ksh"
_MODEL_FAMILY     = "prophet"
_MODEL_PHILOSOPHY = "probabilistic"

_GRAIN_WARNING      = (
    "WMAPE reflects daily-grain prediction error — "
    "not comparable to monthly Holt MAPE."
)
_COMPARABILITY_NOTE = (
    "Coverage is the primary quality signal for probabilistic models."
)


def build_contract(cache_path: Path) -> dict:
    """
    Read daily forecast_cache.json and return the unified forecast contract.

    Parameters
    ----------
    cache_path : Path
        Absolute path to forecast_cache.json (ml_platform/forecast_cache.json).

    Returns
    -------
    dict conforming to contract_version=1.

    Raises
    ------
    FileNotFoundError  — cache not yet generated (retrain not run).
    ValueError         — cache schema_version mismatch or malformed JSON.
    """
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Forecast cache not found at {cache_path}. "
            "POST /forecast/retrain/ to generate."
        )

    with open(cache_path, encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw.get("forecast"), list) or not raw["forecast"]:
        raise ValueError("forecast_cache.json: 'forecast' key missing or empty.")

    df = pd.DataFrame(raw["forecast"])
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])

    # Reference: today UTC
    _today = datetime.now(timezone.utc).date()
    if _today.month == 12:
        _first_next_month = _date(_today.year + 1, 1, 1)
    else:
        _first_next_month = _date(_today.year, _today.month + 1, 1)

    # Only use strictly future forecast dates (today + 1 onwards)
    df = df[df["forecast_date"].dt.date > _today].copy()
    if df.empty:
        raise ValueError(
            "Forecast cache contains no future dates — the horizon has expired. "
            "POST /forecast/retrain/ to refresh (horizon_days=180 recommended)."
        )

    _forecast_end = df["forecast_date"].dt.date.max()

    # Aggregate to monthly sums (low/high are conservative bounds — wider than true monthly CI)
    df["forecast_month"] = df["forecast_date"].dt.to_period("M").dt.to_timestamp()
    _agg_cols = {"point": ("point", "sum")}
    if "low_90" in df.columns:
        _agg_cols["low_approx"]  = ("low_90", "sum")
        _agg_cols["high_approx"] = ("high_90", "sum")
    monthly = (
        df.groupby("forecast_month", as_index=False)
        .agg(**_agg_cols)
        .sort_values("forecast_month")
        .reset_index(drop=True)
    )
    monthly["forecast_month"] = pd.to_datetime(monthly["forecast_month"])

    def _last_day(ts: pd.Timestamp) -> _date:
        return _date(ts.year, ts.month, _calendar.monthrange(ts.year, ts.month)[1])

    monthly["_first"] = monthly["forecast_month"].apply(lambda ts: ts.date())
    monthly["_last"]  = monthly["forecast_month"].apply(_last_day)

    # Keep only complete future months:
    #   - first of month must be >= first day of next calendar month (not the current month)
    #   - last day of month must be within the forecast range (no partial tail months)
    monthly = monthly[
        (monthly["_first"] >= _first_next_month) &
        (monthly["_last"]  <= _forecast_end)
    ].reset_index(drop=True)

    if monthly.empty:
        raise ValueError(
            "No complete future months in forecast cache. "
            "POST /forecast/retrain/ with a longer horizon (horizon_days=180)."
        )

    monthly["point"] = monthly["point"].round(1)

    # month_offset relative to current calendar month (1 = next month)
    _current_period = pd.Period(_today, freq="M")
    monthly["month_offset"] = monthly["forecast_month"].apply(
        lambda ts: (pd.Period(ts, freq="M") - _current_period).n
    )

    forecast_rows = [
        {
            "facility":       "KSH",
            "ward":           "Facility",
            "forecast_month": row["forecast_month"].strftime("%Y-%m-%d"),
            "month_offset":   int(row["month_offset"]),
            "point":          float(row["point"]),
            "low_90":         None,
            "high_90":        None,
            "low_approx":     round(float(row["low_approx"]), 0) if "low_approx" in row else None,
            "high_approx":    round(float(row["high_approx"]), 0) if "high_approx" in row else None,
        }
        for _, row in monthly.iterrows()
    ]

    wmape    = raw.get("wmape")
    coverage = raw.get("coverage")

    return {
        "contract_version": _CONTRACT_VERSION,
        "model_source":     _MODEL_SOURCE,
        "model_family":     _MODEL_FAMILY,
        "model_philosophy": _MODEL_PHILOSOPHY,
        "generated_at":     raw.get("generated_at"),
        "model_version":    raw.get("model_version", "unknown"),
        "grain":            "monthly",
        "metrics": {
            "wmape":    wmape,
            "coverage": coverage,
            "mape":     None,
        },
        "model_context": {
            "grain_warning":      _GRAIN_WARNING,
            "comparability_note": _COMPARABILITY_NOTE,
        },
        "forecast": forecast_rows,
    }
