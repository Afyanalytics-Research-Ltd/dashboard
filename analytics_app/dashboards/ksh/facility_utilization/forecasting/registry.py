"""
Champion/Challenger registry — KSH Prophet admission forecasting.

Promotion rules:
  - Only Prophet vs previous Prophet. Holt ETS (baseline_model.py) is NEVER
    a candidate — different grain (monthly vs daily), different holdout size (3 vs 60),
    not a valid apples-to-apples comparison.
  - Promotion metric: WMAPE (weighted MAPE = sum|errors| / sum(actuals)).
    MAPE is numerically unstable on daily count data with zero-admission days —
    zeros produce 200–500% per-row contributions regardless of fit quality.
  - Promotion threshold: new WMAPE must improve by >= PROMOTION_THRESHOLD (default 5%)
    relative to current champion WMAPE.
    Formula: (champion_wmape - new_wmape) / champion_wmape >= 0.05
  - First prophet run: promotes automatically (no existing champion to compare against).
  - Champion tag: MLflow tag is_champion='true'. Only one champion at a time.

PostgreSQL model_runs table:
  Defined in ml_platform/postgres/schema.sql — has is_champion column with a
  partial unique index ensuring only one champion. Not yet wired here: sync added
  in Phase 4 when FastAPI and the Postgres container are live. MLflow is the
  source of truth until then.
"""

import os
from pathlib import Path

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

_ML_PLATFORM      = Path(os.environ.get("ML_PLATFORM_PATH", "ml_platform"))
_MLFLOW_DB        = _ML_PLATFORM / "mlflow" / "mlflow.db"
_EXPERIMENT_NAME  = "ksh_admission_forecasting"
_CHAMPION_TAG_KEY = "is_champion"
_MODEL_TYPE_KEY   = "model_type"
_PROPHET_TYPE     = "prophet"

PROMOTION_THRESHOLD = 0.05   # 5% relative WMAPE improvement required


def evaluate(
    metrics: dict,
    run_id: str,
    promotion_threshold: float = PROMOTION_THRESHOLD,
) -> dict:
    """
    Run Champion/Challenger evaluation for a completed Prophet run.

    Finds the current champion in MLflow (if any), compares WMAPE, and promotes
    the new run if it clears the threshold. Demotes the previous champion atomically
    (tag update — not a transaction, but MLflow is the local store so collision risk
    is negligible in single-pipeline operation).

    Parameters
    ----------
    metrics : dict
        Output of prophet_model.run()["metrics"]. Must contain key 'wmape'.
    run_id : str
        MLflow run_id from prophet_model.run()["run_id"].
        If 'no-mlflow', evaluation is skipped and {'promoted': False} is returned.
    promotion_threshold : float
        Minimum relative WMAPE improvement (default 0.05 = 5%).

    Returns
    -------
    dict with keys:
        promoted         : bool   — True if this run became champion
        champion_run_id  : str    — run_id of the current champion after evaluation
        reason           : str    — human-readable promotion outcome
        new_wmape        : float  — WMAPE of this run
        champion_wmape   : float | None — WMAPE of previous champion (None if first run)
    """
    new_wmape = metrics.get("wmape")
    if new_wmape is None:
        raise ValueError(
            "registry.evaluate: metrics dict missing 'wmape'. "
            "Ensure prophet_model._compute_metrics() ran with WMAPE support."
        )

    if run_id == "no-mlflow":
        return {
            "promoted":        False,
            "champion_run_id": None,
            "reason":          "mlflow_unavailable",
            "new_wmape":       new_wmape,
            "champion_wmape":  None,
        }

    if not _MLFLOW_AVAILABLE:
        return {
            "promoted":        False,
            "champion_run_id": None,
            "reason":          "mlflow_unavailable",
            "new_wmape":       new_wmape,
            "champion_wmape":  None,
        }

    client = _get_client()
    champion = _find_champion(client)

    if champion is None:
        _promote(client, run_id)
        return {
            "promoted":        True,
            "champion_run_id": run_id,
            "reason":          "first_prophet_run",
            "new_wmape":       round(new_wmape, 4),
            "champion_wmape":  None,
        }

    champion_wmape = champion.get("wmape")

    if champion_wmape is None:
        # Older champion logged before WMAPE was added — promote unconditionally.
        _demote(client, champion["run_id"])
        _promote(client, run_id)
        return {
            "promoted":        True,
            "champion_run_id": run_id,
            "reason":          "champion_predates_wmape_metric",
            "new_wmape":       round(new_wmape, 4),
            "champion_wmape":  None,
        }

    improvement = (champion_wmape - new_wmape) / max(champion_wmape, 1e-9)

    if improvement >= promotion_threshold:
        _demote(client, champion["run_id"])
        _promote(client, run_id)
        return {
            "promoted":        True,
            "champion_run_id": run_id,
            "reason":          f"wmape_improved_{improvement:.1%}",
            "new_wmape":       round(new_wmape, 4),
            "champion_wmape":  round(champion_wmape, 4),
        }

    return {
        "promoted":        False,
        "champion_run_id": champion["run_id"],
        "reason":          f"insufficient_improvement_{improvement:.1%}",
        "new_wmape":       round(new_wmape, 4),
        "champion_wmape":  round(champion_wmape, 4),
    }


def get_champion() -> dict | None:
    """
    Return current champion run info, or None if no champion exists.

    Useful for FastAPI /forecast endpoint to identify which run to load.
    """
    if not _MLFLOW_AVAILABLE:
        return None
    return _find_champion(_get_client())


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _get_client() -> "MlflowClient":
    _MLFLOW_DB.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{_MLFLOW_DB.as_posix()}")
    return MlflowClient()


def _find_champion(client: "MlflowClient") -> dict | None:
    """
    Search MLflow for the current Prophet champion run.
    Returns None if no experiment exists or no champion tagged.
    """
    experiment = client.get_experiment_by_name(_EXPERIMENT_NAME)
    if experiment is None:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"tags.{_CHAMPION_TAG_KEY} = 'true' "
            f"and tags.{_MODEL_TYPE_KEY} = '{_PROPHET_TYPE}'"
        ),
        max_results=1,
    )

    if not runs:
        return None

    r = runs[0]
    return {
        "run_id":      r.info.run_id,
        "wmape":       r.data.metrics.get("wmape"),
        "rmse":        r.data.metrics.get("rmse"),
        "coverage_90": r.data.metrics.get("coverage_90"),
    }


def _promote(client: "MlflowClient", run_id: str) -> None:
    client.set_tag(run_id, _CHAMPION_TAG_KEY, "true")


def _demote(client: "MlflowClient", run_id: str) -> None:
    client.set_tag(run_id, _CHAMPION_TAG_KEY, "false")
