"""
Django views — KSH Admission Forecasting API.

Endpoints:
  GET  /forecast/                — return cached forecast (last retrain output)
  GET  /forecast/health/         — check MLflow DB + cache present, last retrain status
  POST /forecast/retrain/        — trigger background Prophet retrain (CSRF-exempt)
  GET  /forecast/retrain/status/ — return last retrain run status
  GET  /forecast/drift/          — return latest drift report summary (observability only)

Cache design:
  Retrain is expensive (~30-60s: Snowflake pull + two Prophet fits). GET /forecast/
  reads a pre-written JSON cache, not recompute on demand. Cache is written atomically
  (tmp -> fsync -> os.replace) to prevent partial-read corruption.

ML_PLATFORM_PATH env var:
  Set this to the absolute path of the ml_platform/ directory on the server.
  Both this service and the Streamlit dashboard read from that location.
  Defaults to ml_platform/ relative to the repo root if not set.
"""

import json
import logging
import os
import threading
import traceback as _traceback
from datetime import datetime, timezone
from pathlib import Path

_LOG = logging.getLogger(__name__)

from django.http import HttpResponse, HttpResponseForbidden, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required
from django.shortcuts import render

_ML_PLATFORM   = Path(os.environ.get("ML_PLATFORM_PATH",
                       str(Path(__file__).resolve().parent.parent / "ml_platform")))
_CACHE_FILE    = _ML_PLATFORM / "forecast_cache.json"
_STATUS_FILE   = _ML_PLATFORM / "retrain_status.json"
_DRIFT_FILE    = _ML_PLATFORM / "evidently" / "latest_drift.json"
_LOCK_FILE     = _ML_PLATFORM / "retrain.lock"
_MLFLOW_DB     = _ML_PLATFORM / "mlflow" / "mlflow.db"

_CACHE_SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _acquire_lock() -> bool:
    _ML_PLATFORM.mkdir(parents=True, exist_ok=True)
    try:
        with open(_LOCK_FILE, "x") as f:
            f.write(str(os.getpid()))
        return True
    except FileExistsError:
        if not _lock_is_live():
            _release_lock()
            try:
                with open(_LOCK_FILE, "x") as f:
                    f.write(str(os.getpid()))
                return True
            except FileExistsError:
                return False
        return False


def _lock_is_live() -> bool:
    try:
        pid = int(_LOCK_FILE.read_text().strip())
    except (ValueError, FileNotFoundError, OSError):
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _release_lock() -> None:
    _LOCK_FILE.unlink(missing_ok=True)


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_status(status: str, **kwargs) -> None:
    _atomic_write(_STATUS_FILE, {"status": status, **kwargs})


def _retrain_worker() -> None:
    """
    Background thread: pull -> fit -> evaluate -> cache -> status (critical path).
    Drift report runs after, isolated — its failure never aborts the pipeline.
    """
    started_at = _now_iso()
    _write_status("running", started_at=started_at)
    result = None

    try:
        from analytics_app.dashboards.ksh.facility_utilization.forecasting.extractor import pull
        from analytics_app.dashboards.ksh.facility_utilization.forecasting.prophet_model import run
        from analytics_app.dashboards.ksh.facility_utilization.forecasting.registry import evaluate

        df     = pull()
        result = run(df)
        evaluate(result["metrics"], result["run_id"])

        fdf = result["forecast"]
        records = [
            {
                "facility":      row["facility"],
                "ward":          row["ward"],
                "forecast_date": str(row["forecast_date"])[:10],
                "point":         float(row["point"]),
                "low_90":        float(row["low_90"]),
                "high_90":       float(row["high_90"]),
                "model_version": row["model_version"],
            }
            for _, row in fdf.iterrows()
        ]

        model_version = fdf["model_version"].iloc[0] if len(fdf) else "unknown"
        _atomic_write(_CACHE_FILE, {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "model_version":  model_version,
            "generated_at":   _now_iso(),
            "wmape":          result["metrics"]["wmape"],
            "coverage":       round(result["metrics"]["coverage_90"] * 100, 1),
            "forecast":       records,
        })

        _write_status(
            "success",
            started_at=started_at,
            completed_at=_now_iso(),
            run_id=result["run_id"],
        )

    except Exception as exc:
        _write_status(
            "error",
            started_at=started_at,
            completed_at=_now_iso(),
            error=str(exc),
            traceback=_traceback.format_exc(),
        )
    finally:
        _release_lock()

    if result is not None:
        try:
            from analytics_app.dashboards.ksh.facility_utilization.forecasting.drift import generate as drift_generate
            drift_summary = drift_generate(result["train"], result["holdout"], result["run_id"])
            _atomic_write(_DRIFT_FILE, drift_summary)
        except Exception as exc:
            _LOG.warning("drift: non-critical failure (pipeline unaffected): %s", exc)


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def forecast(request):
    if not _CACHE_FILE.exists():
        return JsonResponse(
            {"error": "No forecast cache yet. POST to /forecast/retrain/ to generate one."},
            status=404,
        )
    try:
        with open(_CACHE_FILE, encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return JsonResponse({"error": f"Cache read error: {exc}"}, status=500)

    if payload.get("schema_version") != _CACHE_SCHEMA_VERSION:
        return JsonResponse(
            {
                "error":         "Cache schema_version mismatch. Retrain to refresh.",
                "cache_version": payload.get("schema_version"),
                "expected":      _CACHE_SCHEMA_VERSION,
            },
            status=409,
        )
    return JsonResponse(payload)


def health(request):
    cache_ok  = _CACHE_FILE.exists()
    mlflow_ok = _MLFLOW_DB.exists()

    retrain_info = None
    if _STATUS_FILE.exists():
        try:
            with open(_STATUS_FILE, encoding="utf-8") as f:
                retrain_info = json.load(f)
        except Exception:
            retrain_info = {"error": "status file unreadable"}

    ok = cache_ok and mlflow_ok
    return JsonResponse(
        {
            "healthy":      ok,
            "cache":        "ok" if cache_ok else "missing",
            "mlflow_db":    "ok" if mlflow_ok else "missing",
            "last_retrain": retrain_info,
        },
        status=200 if ok else 503,
    )


@csrf_exempt
@require_POST
def retrain(request):
    if not _acquire_lock():
        return JsonResponse(
            {"status": "already_running", "message": "Retrain already in progress."},
            status=409,
        )
    t = threading.Thread(target=_retrain_worker, daemon=True)
    t.start()
    return JsonResponse(
        {
            "status":  "accepted",
            "message": "Retrain started. Poll /forecast/retrain/status/ for progress.",
        },
        status=202,
    )


def retrain_status(request):
    if not _STATUS_FILE.exists():
        return JsonResponse({"status": "never_run"})
    try:
        with open(_STATUS_FILE, encoding="utf-8") as f:
            return JsonResponse(json.load(f))
    except Exception as exc:
        return JsonResponse({"error": f"Status read error: {exc}"}, status=500)


def drift(request):
    if not _DRIFT_FILE.exists():
        return JsonResponse(
            {"error": "No drift report yet. POST to /forecast/retrain/ to generate one."},
            status=404,
        )
    try:
        with open(_DRIFT_FILE, encoding="utf-8") as f:
            return JsonResponse(json.load(f))
    except Exception as exc:
        return JsonResponse({"error": f"Drift read error: {exc}"}, status=500)


@login_required
def admin_monitor(request):
    """Internal admin page — staff/superuser only."""
    if not (request.user.is_staff or request.user.is_superuser):
        return HttpResponseForbidden("Admin access only.")

    # Cache info
    cache_info = None
    if _CACHE_FILE.exists():
        try:
            with open(_CACHE_FILE, encoding="utf-8") as f:
                raw = json.load(f)
            cache_info = {
                "model_version": raw.get("model_version", "—"),
                "generated_at":  raw.get("generated_at", "—"),
                "wmape":         raw.get("wmape", "—"),
                "coverage":      raw.get("coverage", "—"),
                "record_count":  len(raw.get("forecast", [])),
            }
        except Exception:
            pass

    # Last retrain status
    retrain_info = None
    if _STATUS_FILE.exists():
        try:
            with open(_STATUS_FILE, encoding="utf-8") as f:
                retrain_info = json.load(f)
        except Exception:
            retrain_info = {"status": "unreadable"}

    # Drift summary
    drift_info = None
    if _DRIFT_FILE.exists():
        try:
            with open(_DRIFT_FILE, encoding="utf-8") as f:
                drift_info = json.load(f)
        except Exception:
            pass

    # MLflow recent runs (graceful — mlflow may not be installed)
    mlflow_runs = None
    if _MLFLOW_DB.exists():
        try:
            import mlflow as _mlflow
            _mlflow.set_tracking_uri(f"sqlite:///{_MLFLOW_DB}")
            df = _mlflow.search_runs(
                experiment_names=["ksh_admission_forecast"],
                order_by=["start_time DESC"],
                max_results=5,
                output_format="pandas",
            )
            mlflow_runs = []
            for _, row in df.iterrows():
                wmape    = row.get("metrics.wmape")
                cov      = row.get("metrics.coverage_90")
                mlflow_runs.append({
                    "run_id":   str(row.get("run_id", ""))[:14],
                    "wmape":    round(float(wmape), 4) if wmape is not None else "—",
                    "coverage": round(float(cov) * 100, 1) if cov is not None else "—",
                    "started":  str(row.get("start_time", ""))[:19],
                    "status":   str(row.get("status", "—")),
                })
        except Exception:
            mlflow_runs = None

    # Does the Evidently HTML report file exist?
    has_drift_report = False
    if drift_info and drift_info.get("report_id"):
        has_drift_report = (_ML_PLATFORM / "evidently" / "reports" / drift_info["report_id"]).exists()

    context = {
        "cache_ok":         _CACHE_FILE.exists(),
        "cache_info":       cache_info,
        "mlflow_ok":        _MLFLOW_DB.exists(),
        "retrain_info":     retrain_info,
        "drift_info":       drift_info,
        "mlflow_runs":      mlflow_runs,
        "has_drift_report": has_drift_report,
    }
    return render(request, "forecasting/admin_monitor.html", context)


@login_required
def drift_report(request):
    """Serve the latest Evidently HTML drift report (staff/superuser only)."""
    if not (request.user.is_staff or request.user.is_superuser):
        return HttpResponseForbidden("Admin access only.")

    reports_dir = _ML_PLATFORM / "evidently" / "reports"
    report_path = None

    # Prefer the report_id recorded in the drift JSON
    if _DRIFT_FILE.exists():
        try:
            with open(_DRIFT_FILE, encoding="utf-8") as f:
                drift = json.load(f)
            rid = drift.get("report_id")
            if rid:
                candidate = reports_dir / rid
                if candidate.exists():
                    report_path = candidate
        except Exception:
            pass

    # Fallback: newest .html in reports dir
    if report_path is None and reports_dir.exists():
        html_files = sorted(reports_dir.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
        if html_files:
            report_path = html_files[0]

    if report_path is None:
        return HttpResponse(
            "<h2>No drift report found</h2>"
            "<p>Trigger a retrain to generate one. "
            "Evidently must be installed and the retrain must complete successfully.</p>",
            status=404,
            content_type="text/html",
        )

    try:
        content = report_path.read_text(encoding="utf-8")
    except OSError as exc:
        return HttpResponse(f"Error reading report: {exc}", status=500, content_type="text/plain")

    return HttpResponse(content, content_type="text/html")
