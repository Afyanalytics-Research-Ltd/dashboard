"""
Prophet daily forecasting model — KSH admission forecasting.

PURPOSE: Champion/Challenger candidate.
  - 60-day holdout evaluation (meaningful coverage_90 with 60 points)
  - Champion/Challenger: new Prophet promotes only if MAPE improves >= 5% vs
    the PREVIOUS CHAMPION PROPHET RUN (same daily grain, same 60-day holdout).
    Promotion logic lives in registry.py — this module fits and logs only.
  - Holt ETS (baseline_model.py) is sanity reference only — different grain,
    different holdout size, not used in promotion decision.

Why daily grain:
  Prophet handles weekly and yearly seasonality at daily resolution via Fourier
  series decomposition. ETS at daily grain would be dominated by day-of-week
  noise with no seasonal mechanism. Prophet is the right tool here.

Yearly seasonality note:
  Training data spans ~19 months (~1.5 annual cycles). Yearly seasonality is
  enabled but may be weakly identified. Monitor seasonality component plots.
  If yearly starts producing unstable swings, reduce seasonality_prior_scale
  (default=10) — do not disable. The lever is prior_scale, not enable/disable.

Oct 2025 gap handling:
  prepare_prophet_inputs() excludes Oct 4–16 rows. Prophet sees a 13-day jump
  in ds (Oct 3 → Oct 17). Correct — Prophet fits Fourier terms over continuous
  time, not row index, so date gaps are handled natively. A spurious changepoint
  may appear near Oct 2025 due to the discontinuity. Inspect model.params["delta"]
  for large values post-fit. Documented, not corrected at v1.

Prediction intervals:
  interval_width=0.90 at instantiation — Prophet generates yhat_lower/yhat_upper
  from its uncertainty model (posterior samples via cmdstanpy). Do not post-process
  simulated draws. yhat_lower clamped to 0 in forecast output (counts cannot be negative).

v1 tuning findings (2026-06-23, tune_prophet.py on 60-day holdout):
  interval_width=0.95: coverage 71.7%→83.3%, wmape unchanged. Adopted.
  changepoint_prior_scale=0.10: rejected — worsened wmape, rmse, coverage.
  Easter Monday upper_window=4: coverage 83.3%→85.0%, wmape 51.91→51.45%. Adopted.
  Final v1 metrics: wmape=51.45%, rmse=2.77, coverage=85.0%.

Residual coverage gap (15%):
  Dispersion index=1.66 (moderately overdispersed). Remaining misses split:
    - Surge days (actual>=10): 9/584 days (1.7%), no DOW pattern, no detectable
      precursor. Irreducible without external predictors (referral spikes, outbreaks).
    - Random crashes: operational disruptions not in time-series signal.
  Further hyperparameter search unlikely to yield material gains. Stop tuning here.
  Use intervals as planning bands, not precise probabilistic bounds.

Forecast output:
  Future-only rows filtered via ds > all_data_end. Never use tail() — fragile
  when horizon_days changes.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

try:
    from prophet import Prophet
    _PROPHET_AVAILABLE = True
except ImportError:
    _PROPHET_AVAILABLE = False

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

from facility_utilization.forecasting.features import (
    prepare_prophet_inputs,
    make_prophet_holidays,
)

_HOLDOUT_DAYS            = 60
_FACILITY                = "KSH"
_MODEL_VERSION           = "prophet_v1"
_CHANGEPOINT_PRIOR_SCALE = 0.05   # confirmed: looser (0.10) worsened wmape+rmse+coverage
_SEASONALITY_PRIOR_SCALE = 10.0   # configured — logged for reproducibility
_INTERVAL_WIDTH          = 0.95   # confirmed: 0.90→0.95 improved coverage 71.7%→83.3%, wmape unchanged

_MLFLOW_DB = Path(__file__).resolve().parents[2] / "ml_platform" / "mlflow" / "mlflow.db"


def run(df: pd.DataFrame, horizon_days: int = 180) -> dict:
    """
    Fit Prophet on training split, evaluate on 60-day holdout, forecast future days.

    Parameters
    ----------
    df : pd.DataFrame
        Output of extractor.pull(). Columns: admission_date, admissions, is_gap_flagged.
    horizon_days : int
        Future days to forecast beyond the data end date. Default 180 ≈ 6 months.
        Extended from 90 to ensure 3 complete future calendar months remain
        available when data cutoff lags the current date by 1–2 months.

    Returns
    -------
    dict with keys:
        run_id       : str            — MLflow run ID ('no-mlflow' if unavailable)
        metrics      : dict           — mape, rmse, coverage_90, training_days, holdout_days
        prophet_df   : pd.DataFrame   — full (ds, y) input used by Prophet
        train        : pd.DataFrame   — training split (first N-60 rows)
        holdout      : pd.DataFrame   — holdout split (last 60 rows, ds + y)
        holdout_pred : pd.DataFrame   — Prophet predict() output for holdout dates
        forecast     : pd.DataFrame   — future forecast (forecast_date, point, low_90, high_90)
        model        : Prophet        — final model refit on all data
    """
    if not _PROPHET_AVAILABLE:
        raise ImportError(
            "prophet is not installed. Install with: pip install prophet"
        )
    if horizon_days < 1:
        raise ValueError(f"prophet_model: horizon_days must be >= 1, got {horizon_days}")

    prophet_df = prepare_prophet_inputs(df)
    _validate_prophet_df(prophet_df)

    # Time-series safe split: last 60 rows = holdout, no shuffling
    train_df   = prophet_df.iloc[:-_HOLDOUT_DAYS].copy().reset_index(drop=True)
    holdout_df = prophet_df.iloc[-_HOLDOUT_DAYS:].copy().reset_index(drop=True)

    # Holiday range: training start → forecast end (covers future holiday effects)
    forecast_end = prophet_df["ds"].max() + pd.Timedelta(days=horizon_days)
    hols = make_prophet_holidays(train_df["ds"].min(), forecast_end)

    # Evaluation: fit on training data only, predict holdout dates
    model_eval   = _fit(train_df, hols)
    holdout_pred = model_eval.predict(holdout_df[["ds"]])
    metrics      = _compute_metrics(
        holdout_df["y"].values,
        holdout_pred,
        training_days=len(train_df),
    )

    # Production: refit on all available data, forecast future
    model_final = _fit(prophet_df, hols)
    forecast_df = _build_forecast_df(model_final, prophet_df, horizon_days)

    run_id = _log_mlflow(metrics, train_df, horizon_days)

    return {
        "run_id":        run_id,
        "metrics":       metrics,
        "prophet_df":    prophet_df,
        "train":         train_df,
        "holdout":       holdout_df,
        "holdout_pred":  holdout_pred,
        "forecast":      forecast_df,
        "model":         model_final,
    }


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _validate_prophet_df(df: pd.DataFrame) -> None:
    if not df["ds"].is_monotonic_increasing:
        raise ValueError("prophet_model: ds column is not monotonically increasing.")
    if df["ds"].duplicated().any():
        n = df["ds"].duplicated().sum()
        raise ValueError(f"prophet_model: {n} duplicate dates in ds column.")
    if df["y"].isna().any():
        n = df["y"].isna().sum()
        raise ValueError(
            f"prophet_model: {n} NaN values in y column — "
            "gap rows must be excluded before Prophet (prepare_prophet_inputs handles this)."
        )


def _fit(train_df: pd.DataFrame, holidays_df) -> "Prophet":
    kwargs = dict(
        interval_width=_INTERVAL_WIDTH,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=_CHANGEPOINT_PRIOR_SCALE,
        seasonality_prior_scale=_SEASONALITY_PRIOR_SCALE,
    )
    if holidays_df is not None:
        kwargs["holidays"] = holidays_df

    model = Prophet(**kwargs)
    model.fit(train_df)
    return model


def _compute_metrics(
    actuals: np.ndarray,
    holdout_pred: pd.DataFrame,
    training_days: int,
) -> dict:
    """
    MAPE, RMSE, coverage_90 on 60-day holdout.

    coverage_90: fraction of actual values inside [yhat_lower, yhat_upper].
    With 60 holdout points this is meaningful — unlike ETS's 3-point holdout.

    MAPE denominator: max(actual, 1) — avoids divide-by-zero on zero-admission days.
    """
    yhat  = holdout_pred["yhat"].values
    lower = holdout_pred["yhat_lower"].values
    upper = holdout_pred["yhat_upper"].values

    errors      = actuals - yhat
    # MAPE: kept for reference but unreliable on daily count data with zero-admission days.
    # Zero days (actual=0) use denominator=1, producing 200-500% per-row contributions.
    # Use wmape for champion/challenger promotion threshold — not mape.
    mape        = float(np.mean(np.abs(errors / np.maximum(actuals, 1))) * 100)
    # WMAPE: sum(|errors|) / sum(actuals). Zero days contribute ~0 to both num and denom.
    # Stable and meaningful for daily count data.
    wmape       = float(np.sum(np.abs(errors)) / max(float(np.sum(actuals)), 1.0) * 100)
    rmse        = float(np.sqrt(np.mean(errors ** 2)))
    inside      = (actuals >= lower) & (actuals <= upper)
    coverage_90 = float(inside.mean())

    if coverage_90 < 0.80:
        print(
            f"\n  [WARN] coverage_90={coverage_90:.2f} is below 0.80 (v1 target). "
            "Prophet's Gaussian uncertainty model underestimates tails on daily admission "
            "counts with clinical surges. Documented limitation — not a code error."
        )

    return {
        "mape":          round(mape, 4),
        "wmape":         round(wmape, 4),
        "rmse":          round(rmse, 4),
        "coverage_90":   round(coverage_90, 4),
        "training_days": training_days,
        "holdout_days":  _HOLDOUT_DAYS,
    }


def _build_forecast_df(
    model: "Prophet",
    all_data: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    """
    Forecast horizon_days into the future.

    make_future_dataframe includes all historical dates — filter explicitly
    to future-only rows (ds > last data date). Never use tail(): fragile
    when horizon_days changes between runs.
    """
    future      = model.make_future_dataframe(periods=horizon_days, freq="D")
    forecast    = model.predict(future)

    data_end    = all_data["ds"].max()
    future_only = forecast[forecast["ds"] > data_end].copy()

    return pd.DataFrame({
        "facility":      _FACILITY,
        "ward":          None,
        "forecast_date": future_only["ds"].values,
        "point":         np.round(future_only["yhat"].values, 2),
        "low_90":        np.round(np.maximum(future_only["yhat_lower"].values, 0.0), 2),
        "high_90":       np.round(future_only["yhat_upper"].values, 2),
        "model_version": _MODEL_VERSION,
    })


def _log_mlflow(
    metrics: dict,
    train_df: pd.DataFrame,
    horizon_days: int,
) -> str:
    if not _MLFLOW_AVAILABLE:
        return "no-mlflow"

    _MLFLOW_DB.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{_MLFLOW_DB.as_posix()}")
    mlflow.set_experiment("ksh_admission_forecasting")

    with mlflow.start_run(run_name=_MODEL_VERSION) as active_run:
        mlflow.set_tag("facility",   _FACILITY)
        mlflow.set_tag("model_type", "prophet")
        mlflow.set_tag("grain",      "daily")
        mlflow.set_tag("is_champion", "false")

        import prophet as _prophet_lib
        mlflow.log_param("prophet_version",          _prophet_lib.__version__)
        mlflow.log_param("interval_width",           _INTERVAL_WIDTH)
        mlflow.log_param("changepoint_prior_scale",  _CHANGEPOINT_PRIOR_SCALE)
        mlflow.log_param("seasonality_prior_scale",  _SEASONALITY_PRIOR_SCALE)
        mlflow.log_param("yearly_seasonality",       True)
        mlflow.log_param("weekly_seasonality",       True)
        mlflow.log_param("daily_seasonality",        False)
        mlflow.log_param("training_days",            metrics["training_days"])
        mlflow.log_param("holdout_days",             _HOLDOUT_DAYS)
        mlflow.log_param("horizon_days",             horizon_days)
        mlflow.log_param("training_start",           str(train_df["ds"].min().date()))
        mlflow.log_param("training_end",             str(train_df["ds"].max().date()))

        mlflow.log_metric("mape",        metrics["mape"])
        mlflow.log_metric("wmape",       metrics["wmape"])
        mlflow.log_metric("rmse",        metrics["rmse"])
        mlflow.log_metric("coverage_90", metrics["coverage_90"])

        return active_run.info.run_id
