"""
Drift monitoring for KSH admission forecasting — observability only.

PURPOSE: Detect regime change in the daily admission time series between the
training window and the most recent holdout window after each retrain. Output
is telemetry — it informs the decision to retrain but must never trigger it.

Conceptual note:
  Prophet is univariate — admissions is both input and target. What this module
  measures is distributional shift in a single time series, more precisely
  "regime change detection in a stochastic process" than classical ML data drift.
  Evidently is used for structured HTML visualisation only. Metrics are computed
  via scipy (KS test) which is version-stable and independent of Evidently internals.

Metric choice — KS statistic (D):
  D = max absolute difference between the two empirical CDFs. Bounded 0–1.
  Direct effect size — interpretable without statistical context.
  More stable than 1–p_value under sample size changes.
  _DRIFT_THRESHOLD = 0.15: corresponds to p ≈ 0.10 for n=60 vs n=524.
  Threshold tied to operational tolerance, not statistics purity — adjust
  as retrain history accumulates.

Pipeline isolation:
  generate() is called from views._retrain_worker() inside a best-effort
  try/except. Any exception here is logged and swallowed — it must never
  propagate to the critical path (forecast cache write).

  Within generate() itself: scipy metrics always run. Evidently HTML is
  additionally best-effort — its failure returns report_id=None without
  raising.

Output schema (schema_version=1):
  {
    "schema_version": 1,
    "run_id":         str,
    "generated_at":   str (ISO-8601 UTC),
    "window":         {"type": "rolling_holdout", "size_days": 60},
    "reference_rows": int,
    "current_rows":   int,
    "drift_detected": bool,
    "drift_score":    float,  # KS statistic D, 0–1, higher = more drift
    "drift_type":     "none" | "distribution_shift",
    "diagnostics":    {"p_value": float},
    "report_id":      str | None  # filename only — resolve via ml_platform/evidently/reports/
  }
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

_LOG = logging.getLogger(__name__)

try:
    from evidently.core.report import Report
    from evidently.presets import DataDriftPreset
    from evidently import Dataset, DataDefinition
    _EVIDENTLY_AVAILABLE = True
except ImportError:
    _EVIDENTLY_AVAILABLE = False

_HOLDOUT_DAYS    = 60
_SCHEMA_VERSION  = 1

# KS statistic D threshold. D ≈ 0.15 → p ≈ 0.10 for n=60 vs n=524.
# Adjust upward (more conservative) if false-positive rate is too high
# as retrain frequency increases.
_DRIFT_THRESHOLD = 0.15

import os as _os
_ML_PLATFORM = Path(_os.environ.get("ML_PLATFORM_PATH", "ml_platform"))
_REPORTS_DIR = _ML_PLATFORM / "evidently" / "reports"


def generate(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    run_id: str,
) -> dict:
    """
    Compute drift metrics and generate HTML report for one retrain cycle.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split — columns ds, y. From prophet_model.run()["train"].
    holdout_df : pd.DataFrame
        Holdout split — columns ds, y. From prophet_model.run()["holdout"].
    run_id : str
        MLflow run ID — ties this report to the experiment run.

    Returns
    -------
    dict conforming to schema_version=1.
    """
    ref = train_df[["y"]].rename(columns={"y": "target"}).reset_index(drop=True)
    cur = holdout_df[["y"]].rename(columns={"y": "target"}).reset_index(drop=True)

    # KS statistic: stable effect size independent of Evidently version.
    # D = max |CDF_ref(x) - CDF_cur(x)|, bounded 0–1.
    ks_stat, p_value  = ks_2samp(ref["target"].values, cur["target"].values)
    drift_score       = round(float(ks_stat), 4)
    drift_detected    = drift_score >= _DRIFT_THRESHOLD

    # HTML report — Evidently visualisation, best-effort.
    report_id = _save_evidently_html(ref, cur, run_id)

    return {
        "schema_version": _SCHEMA_VERSION,
        "run_id":         run_id,
        "generated_at":   _now_iso(),
        "window":         {"type": "rolling_holdout", "size_days": _HOLDOUT_DAYS},
        "reference_rows": len(ref),
        "current_rows":   len(cur),
        "drift_detected": drift_detected,
        "drift_score":    drift_score,
        "drift_type":     "distribution_shift" if drift_detected else "none",
        "diagnostics":    {"p_value": round(float(p_value), 4)},
        "report_id":      report_id,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_evidently_html(ref: pd.DataFrame, cur: pd.DataFrame, run_id: str) -> str | None:
    """
    Generate Evidently HTML visualisation report.
    Returns filename (e.g. "<run_id>.html") on success, None on failure.
    Failure is non-fatal — logged and returned as None.
    """
    if not _EVIDENTLY_AVAILABLE:
        _LOG.warning("drift: evidently not installed — HTML report skipped")
        return None
    try:
        dd       = DataDefinition(numerical_columns=["target"])
        ref_ds   = Dataset.from_pandas(ref, data_definition=dd)
        cur_ds   = Dataset.from_pandas(cur, data_definition=dd)
        report   = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(current_data=cur_ds, reference_data=ref_ds)
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{run_id}.html"
        snapshot.save_html(str(_REPORTS_DIR / filename))
        return filename
    except Exception as exc:
        _LOG.warning("drift: HTML report failed (non-critical): %s", exc)
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
