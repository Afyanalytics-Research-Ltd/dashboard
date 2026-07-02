"""
snapshot_writer.py — AI Chat Layer

Reads metrics_registry.json + schema_catalog.json, queries each metric from its
gold table applying all mandatory and conditional filters, computes current values,
runs inline validation, and writes metrics_snapshot.json.

Invoked via: manage.py refresh_snapshot

Dispatch is data-driven: every registry entry declares fetch_strategy,
compute_strategy, history_months, data_start, and gate_denominator_column.
No business rules are hardcoded in this module.
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .snowflake_db import run_query_df

logger = logging.getLogger(__name__)

_DIR          = Path(__file__).parent
REGISTRY_PATH = _DIR / "metrics_registry.json"
CATALOG_PATH  = _DIR / "schema_catalog.json"
SNAPSHOT_PATH = _DIR / "metrics_snapshot.json"

# Warning codes that indicate unusable data → validation = "failed"
_FATAL_CODES = {"no_data", "duplicate_periods"}


# ─── Loaders ──────────────────────────────────────────────────────────────────

def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _alias_map(catalog: dict) -> dict:
    return {t["alias"]: t for t in catalog["tables"]}


def _date_col(entry: dict) -> str:
    for col, meta in entry["columns"].items():
        if meta.get("type") == "DATE":
            return col
    raise ValueError(f"No DATE column in catalog entry '{entry['alias']}'")


def _target_facility(metric: dict, alias_map: dict) -> str:
    """Derive target facility from schema_catalog facility_scope + table_filter."""
    entry = alias_map.get(metric["table"], {})
    if entry.get("facility_scope") == "KSH_ONLY":
        return "KISUMU_CLEAN"
    tf = metric.get("table_filter") or {}
    return tf.get("facility", "KISUMU_CLEAN")


# ─── SQL helpers ───────────────────────────────────────────────────────────────

def _filter_clauses(table_filter: Optional[dict], skip: set = None) -> list:
    if not table_filter:
        return []
    skip = skip or set()
    clauses = []
    for col, val in table_filter.items():
        if col in skip:
            continue
        if col == "discharge_type":
            clauses.append("discharge_type ILIKE '%patient request%'")
        else:
            clauses.append(f"{col} = '{val}'")
    return clauses


def _mandatory_clauses(entry: dict) -> list:
    return [f"{f['column']} {f['operator']}" for f in entry.get("mandatory_filters", [])]


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names — Snowflake returns uppercase by default."""
    df.columns = df.columns.str.lower()
    return df


# ─── Fetch functions ──────────────────────────────────────────────────────────
# All signatures: (metric, entry, history_months) → DataFrame

def _fetch_standard(metric: dict, entry: dict, history_months: int) -> pd.DataFrame:
    """
    One row per month after filters — rpt_ward_los, rpt_lab_monthly,
    rpt_cd12_monthly_rate, rpt_dialysis_ops, rpt_admission_tat, rpt_imaging_ops.
    Reads data_start and gate_denominator_column from metric if set.
    """
    table = entry["table"]
    dc    = _date_col(entry)
    col   = metric["column"]

    select = [f"{dc} AS month", f"{col} AS value"]
    gate_col = metric.get("gate_denominator_column")
    if gate_col:
        select.append(f"{gate_col} AS gate_count")

    where = [f"{dc} < DATE_TRUNC('month', CURRENT_DATE)"]
    where.extend(_mandatory_clauses(entry))
    where.extend(_filter_clauses(metric.get("table_filter")))
    data_start = metric.get("data_start")
    if data_start:
        where.append(f"{dc} >= '{data_start}'")

    sql = (
        f"SELECT {', '.join(select)} "
        f"FROM {table} "
        f"WHERE {' AND '.join(where)} "
        f"ORDER BY {dc} DESC LIMIT {history_months}"
    )
    return _norm(run_query_df(sql))


def _fetch_theatre(metric: dict, entry: dict, history_months: int) -> pd.DataFrame:
    """
    rpt_theatre_utilization — grain is session_month × theatre_name × booking_status.
    GROUP BY month and recompute completion_rate_pct from session sums.
    """
    table = entry["table"]
    dc    = _date_col(entry)

    where = [f"{dc} < DATE_TRUNC('month', CURRENT_DATE)"]
    where.extend(_mandatory_clauses(entry))
    where.extend(_filter_clauses(metric.get("table_filter")))

    sql = (
        f"SELECT {dc} AS month, "
        f"  SUM(completed_sessions) AS completed, "
        f"  SUM(total_sessions) AS total, "
        f"  100.0 * SUM(completed_sessions) / NULLIF(SUM(total_sessions), 0) AS value "
        f"FROM {table} "
        f"WHERE {' AND '.join(where)} "
        f"GROUP BY {dc} "
        f"ORDER BY {dc} DESC LIMIT {history_months}"
    )
    return _norm(run_query_df(sql))


def _fetch_ward_traffic(metric: dict, entry: dict, history_months: int) -> pd.DataFrame:
    """
    rpt_bed_occupancy — grain is admission_month × facility × ward_name × ward_category.
    GROUP BY month to roll up ward_names within a ward_category.
    """
    table = entry["table"]
    dc    = _date_col(entry)
    col   = metric["column"]

    where = [
        f"{dc} < DATE_TRUNC('month', CURRENT_DATE)",
        "ward_name IS NOT NULL",
    ]
    where.extend(_filter_clauses(metric.get("table_filter")))

    sql = (
        f"SELECT {dc} AS month, SUM({col}) AS value "
        f"FROM {table} "
        f"WHERE {' AND '.join(where)} "
        f"GROUP BY {dc} "
        f"ORDER BY {dc} DESC LIMIT {history_months}"
    )
    return _norm(run_query_df(sql))


def _fetch_patient_request(metric: dict, entry: dict, history_months: int) -> pd.DataFrame:
    """
    rpt_readmissions — patient_request_pct is not pre-computed in gold.
    CASE WHEN across discharge_type rows per month per ward_category.
    gate_count = total_admissions for min_gate enforcement downstream.
    """
    table = entry["table"]
    dc    = _date_col(entry)

    where = [
        f"{dc} < DATE_TRUNC('month', CURRENT_DATE)",
        "discharge_type IS NOT NULL",
    ]
    # Apply facility + ward_category; discharge_type goes into CASE WHEN, not WHERE
    where.extend(_filter_clauses(metric.get("table_filter"), skip={"discharge_type"}))

    sql = (
        f"SELECT "
        f"  {dc} AS month, "
        f"  100.0 * SUM(CASE WHEN discharge_type ILIKE '%patient request%' THEN total_admissions ELSE 0 END) "
        f"    / NULLIF(SUM(total_admissions), 0) AS value, "
        f"  SUM(total_admissions) AS gate_count "
        f"FROM {table} "
        f"WHERE {' AND '.join(where)} "
        f"GROUP BY {dc} "
        f"ORDER BY {dc} DESC LIMIT {history_months}"
    )
    return _norm(run_query_df(sql))


def _fetch_doctor_concentration(metric: dict, entry: dict, history_months: int) -> pd.DataFrame:
    table = entry["table"]
    dc    = _date_col(entry)

    sql = (
        f"SELECT {dc} AS month, "
        f"  100.0 * MAX(evaluations) / NULLIF(SUM(evaluations), 0) AS value "
        f"FROM {table} "
        f"WHERE {dc} >= '2024-01-01' "
        f"  AND {dc} < DATE_TRUNC('month', CURRENT_DATE) "
        f"GROUP BY {dc} "
        f"ORDER BY {dc} DESC LIMIT {history_months}"
    )
    return _norm(run_query_df(sql))


def _fetch_doctor_per_person(metric: dict, entry: dict, history_months: int) -> pd.DataFrame:
    """Evaluations per doctor per month — for burnout_relative and workload_absolute."""
    table = entry["table"]
    dc    = _date_col(entry)

    sql = (
        f"SELECT {dc} AS month, username, evaluations "
        f"FROM {table} "
        f"WHERE {dc} >= '2024-01-01' "
        f"  AND {dc} < DATE_TRUNC('month', CURRENT_DATE) "
        f"ORDER BY {dc} DESC LIMIT {history_months * 10}"
    )
    return _norm(run_query_df(sql))


def _fetch_dialysis(metric: dict, entry: dict, history_months: int) -> pd.DataFrame:
    """history_months comes from registry (24 for dialysis_idle)."""
    table = entry["table"]
    dc    = _date_col(entry)

    sql = (
        f"SELECT {dc} AS month, sessions_billed "
        f"FROM {table} "
        f"WHERE is_partial_month = FALSE "
        f"  AND {dc} < DATE_TRUNC('month', CURRENT_DATE) "
        f"ORDER BY {dc} DESC LIMIT {history_months}"
    )
    return _norm(run_query_df(sql))


# ─── Strategy maps (populated after function definitions) ─────────────────────

FETCH_STRATEGIES = {
    "standard":                     _fetch_standard,
    "theatre_grouped":              _fetch_theatre,
    "ward_traffic_grouped":         _fetch_ward_traffic,
    "patient_request_pct":          _fetch_patient_request,
    "doctor_concentration_grouped": _fetch_doctor_concentration,
    "doctor_per_person":            _fetch_doctor_per_person,
    "dialysis_idle":                _fetch_dialysis,
}


# ─── Compute functions ────────────────────────────────────────────────────────
# All signatures: (df, metric) → result dict

def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _to_history(df: pd.DataFrame, extra_cols: list = None) -> list:
    rows = []
    for _, row in df.iterrows():
        m = row["month"]
        month_str = m.strftime("%Y-%m-%d") if hasattr(m, "strftime") else str(m)
        entry = {"month": month_str, "value": _safe_float(row["value"])}
        for ec in (extra_cols or []):
            if ec in row.index:
                entry[ec] = _safe_float(row[ec])
        rows.append(entry)
    return rows


def _compute_standard(df: pd.DataFrame, metric: dict) -> dict:
    if df.empty:
        return {
            "current_value":         None,
            "trailing_3mo_avg":      None,
            "prior_3mo_avg":         None,
            "history":               [],
            "data_months_available": 0,
        }

    # Include gate_count in history if present (patient_request + cd12)
    extra_cols = ["gate_count"] if "gate_count" in df.columns else None
    history    = _to_history(df, extra_cols=extra_cols)
    values     = [h["value"] for h in history if h["value"] is not None]

    current      = values[0] if values else None
    trailing_avg = round(sum(values[:3]) / len(values[:3]), 4) if values else None
    prior_vals   = values[1:4]
    prior_avg    = round(sum(prior_vals) / len(prior_vals), 4) if prior_vals else None

    return {
        "current_value":         current,
        "trailing_3mo_avg":      trailing_avg,
        "prior_3mo_avg":         prior_avg,
        "history":               history,
        "data_months_available": len(values),
    }


def _compute_theatre(df: pd.DataFrame, metric: dict) -> dict:
    """
    Session-weighted trailing averages for theatre completion rate.
    Arithmetic mean of monthly rates diverges from the dashboard when
    session volumes vary — use SUM(completed)/SUM(total) across the window.
    """
    if df.empty:
        return {
            "current_value":         None,
            "trailing_3mo_avg":      None,
            "prior_3mo_avg":         None,
            "history":               [],
            "data_months_available": 0,
        }

    history = _to_history(df)
    values  = [h["value"] for h in history if h["value"] is not None]
    current = values[0] if values else None

    def _weighted_avg(rows):
        c = sum(int(r.get("completed", 0) or 0) for r in rows)
        t = sum(int(r.get("total",     0) or 0) for r in rows)
        return round(100.0 * c / t, 4) if t else None

    raw = df.to_dict("records")
    trailing_avg = _weighted_avg(raw[:3])
    prior_avg    = _weighted_avg(raw[1:4])

    return {
        "current_value":         current,
        "trailing_3mo_avg":      trailing_avg,
        "prior_3mo_avg":         prior_avg,
        "history":               history,
        "data_months_available": len(values),
    }


def _compute_doctor_per_person(df: pd.DataFrame, metric: dict) -> dict:
    """
    Uniform top-level contract: current_value = worst-case doctor.
    Per-doctor detail lives in `entities` (not `per_doctor`) so the builder
    reads current_value for every metric without branching on compute_strategy.

    current_value semantics by unit:
      pct_of_personal_avg → max pct_of_personal_avg (burnout: highest relative load)
      count               → max evaluations (workload: heaviest-loaded doctor)
    """
    _empty = {
        "current_value":         None,
        "trailing_3mo_avg":      None,
        "prior_3mo_avg":         None,
        "history":               [],
        "data_months_available": 0,
        "entities":              {},
    }
    if df.empty:
        return _empty

    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month", ascending=False)

    entities = {}
    for username, grp in df.groupby("username"):
        grp    = grp.sort_values("month", ascending=False).reset_index(drop=True)
        evals  = [int(v) for v in grp["evaluations"].tolist()]
        months = [m.strftime("%Y-%m-%d") for m in grp["month"].tolist()]

        current    = evals[0] if evals else None
        prior_vals = evals[1:4]
        prior_avg  = round(sum(prior_vals) / len(prior_vals), 1) if prior_vals else None
        pct_of_avg = round(100.0 * current / prior_avg, 1) if (current and prior_avg) else None

        entities[str(username)] = {
            "current_value":       current,
            "personal_3mo_avg":    prior_avg,
            "pct_of_personal_avg": pct_of_avg,
            "history":             [{"month": m, "value": v} for m, v in zip(months, evals)],
        }

    unit = metric.get("unit", "")
    if unit == "pct_of_personal_avg":
        pcts      = [e["pct_of_personal_avg"] for e in entities.values() if e["pct_of_personal_avg"] is not None]
        top_value = _safe_float(max(pcts)) if pcts else None
    else:
        vals      = [e["current_value"] for e in entities.values() if e["current_value"] is not None]
        top_value = _safe_float(max(vals)) if vals else None

    return {
        "current_value":         top_value,
        "trailing_3mo_avg":      None,
        "prior_3mo_avg":         None,
        "history":               [],
        "data_months_available": df["month"].nunique(),
        "entities":              entities,
    }


def _compute_dialysis_idle(df: pd.DataFrame, metric: dict) -> dict:
    if df.empty:
        return {
            "current_value":         None,
            "last_active_month":     None,
            "history":               [],
            "data_months_available": 0,
        }

    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month", ascending=False).reset_index(drop=True)

    today       = datetime.utcnow()
    last_active = None
    for _, row in df.iterrows():
        sb = row.get("sessions_billed")
        if sb is not None and int(sb) > 0:
            last_active = row["month"]
            break

    history_months = metric.get("history_months", 6)
    if last_active is None:
        months_idle = history_months  # idle for at least the full lookback window
    else:
        months_idle = (today.year - last_active.year) * 12 + (today.month - last_active.month)

    history = [
        {"month": row["month"].strftime("%Y-%m-%d"), "value": int(row["sessions_billed"])}
        for _, row in df.iterrows()
    ]

    return {
        "current_value":         months_idle,
        "last_active_month":     last_active.strftime("%Y-%m-%d") if last_active else None,
        "history":               history[:6],
        "data_months_available": len(df),
    }


COMPUTE_STRATEGIES = {
    "standard":          _compute_standard,
    "theatre":           _compute_theatre,
    "doctor_per_person": _compute_doctor_per_person,
    "dialysis_idle":     _compute_dialysis_idle,
}


# ─── Inline validation ────────────────────────────────────────────────────────

def _make_validation(warnings: list) -> dict:
    if any(w in _FATAL_CODES for w in warnings):
        status = "failed"
    elif warnings:
        status = "passed_with_warnings"
    else:
        status = "passed"
    return {"validation": status, "warnings": warnings}


def _validate(metric: dict, result: dict) -> dict:
    """
    Inline validation before snapshot write.
    Returns {"validation": "passed"|"passed_with_warnings"|"failed", "warnings": [...]}

    Warning codes
    ─────────────
    no_data               — query returned nothing
    stale_data            — latest month is > 2 months before today
    insufficient_history  — data_months_available < alerting.requires_data_months
    duplicate_periods     — same month appears more than once (grain issue)
    value_out_of_range    — pct > 100 or negative count/days where impossible
    """
    compute_strategy = metric.get("compute_strategy", "standard")
    warnings = []

    # ── doctor_per_person ────────────────────────────────────────────────────
    if compute_strategy == "doctor_per_person":
        if not result.get("entities"):
            warnings.append("no_data")
        elif result.get("data_months_available", 0) < metric["alerting"].get("requires_data_months", 1):
            warnings.append("insufficient_history")
        return _make_validation(warnings)

    # ── dialysis_idle ────────────────────────────────────────────────────────
    if compute_strategy == "dialysis_idle":
        if result.get("data_months_available", 0) == 0:
            warnings.append("no_data")
        return _make_validation(warnings)

    # ── standard ─────────────────────────────────────────────────────────────
    history = result.get("history", [])

    # 1. No data
    if not history or result.get("current_value") is None:
        warnings.append("no_data")
        return _make_validation(warnings)

    # 2. Freshness — latest month should be last complete month (≤ 2 months lag)
    try:
        latest = datetime.strptime(history[0]["month"], "%Y-%m-%d")
        today  = datetime.utcnow()
        months_lag = (today.year - latest.year) * 12 + (today.month - latest.month)
        if months_lag > 2:
            warnings.append("stale_data")
    except (ValueError, KeyError):
        pass

    # 3. Insufficient history
    requires    = metric["alerting"].get("requires_data_months", 1)
    data_months = result.get("data_months_available", 0)
    if data_months < requires:
        warnings.append("insufficient_history")

    # 4. Duplicate periods
    months_seen = [h["month"] for h in history]
    if len(months_seen) != len(set(months_seen)):
        warnings.append("duplicate_periods")

    # 5. Value sanity
    current = result.get("current_value")
    if current is not None:
        unit = metric.get("unit", "")
        if unit == "pct" and (current < 0 or current > 100):
            warnings.append("value_out_of_range")
        elif unit in ("count", "days", "sessions") and current < 0:
            warnings.append("value_out_of_range")

    return _make_validation(warnings)


# ─── Dispatcher ───────────────────────────────────────────────────────────────

def _fetch_and_compute(metric: dict, alias_map: dict) -> dict:
    mid             = metric["metric_id"]
    alias           = metric["table"]
    fetch_strategy  = metric.get("fetch_strategy", "standard")
    compute_strategy = metric.get("compute_strategy", "standard")
    history_months  = metric.get("history_months", 6)

    entry = alias_map.get(alias)
    if not entry:
        return {
            "fetch_ok":   False,
            "error":      f"alias '{alias}' not found in schema_catalog",
            "validation": "failed",
            "warnings":   ["catalog_miss"],
        }

    fetcher  = FETCH_STRATEGIES.get(fetch_strategy)
    computer = COMPUTE_STRATEGIES.get(compute_strategy)

    if not fetcher:
        return {
            "fetch_ok":   False,
            "error":      f"unknown fetch_strategy '{fetch_strategy}'",
            "validation": "failed",
            "warnings":   ["unknown_strategy"],
        }
    if not computer:
        return {
            "fetch_ok":   False,
            "error":      f"unknown compute_strategy '{compute_strategy}'",
            "validation": "failed",
            "warnings":   ["unknown_strategy"],
        }

    try:
        df              = fetcher(metric, entry, history_months)
        row_count       = len(df)
        result          = computer(df, metric)
        validation      = _validate(metric, result)
        result.update(validation)
        result["fetch_ok"]  = True
        result["row_count"] = row_count
        return result

    except Exception as exc:
        logger.error("snapshot_writer: %s failed — %s", mid, exc, exc_info=True)
        return {
            "fetch_ok":   False,
            "error":      str(exc),
            "validation": "failed",
            "warnings":   ["fetch_exception"],
        }


# ─── Entry point ──────────────────────────────────────────────────────────────

def write_snapshot() -> dict:
    """
    Build and write metrics_snapshot.json. Returns the snapshot dict.
    Iterates over facilities derived from schema_catalog.facility_slugs.
    All metric fetches are attempted independently — failures are isolated.
    """
    registry = _load(REGISTRY_PATH)
    catalog  = _load(CATALOG_PATH)
    am       = _alias_map(catalog)

    snapshot: dict = {
        "schema_version": "1.0",
        "generated_at":   datetime.utcnow().isoformat() + "Z",
    }

    # Group metrics by target facility (derived from schema_catalog, not hardcoded)
    facility_metrics: dict[str, list] = {}
    for metric in registry["metrics"]:
        facility = _target_facility(metric, am)
        facility_metrics.setdefault(facility, []).append(metric)

    for facility, metrics in facility_metrics.items():
        bucket = snapshot[facility] = {"metrics": {}, "fetch_errors": [], "validation_warnings": []}

        for metric in metrics:
            mid = metric["metric_id"]
            logger.info("snapshot_writer: [%s] fetching %s", facility, mid)

            computed = _fetch_and_compute(metric, am)
            bucket["metrics"][mid] = {
                "metric_id":        mid,
                "label":            metric["label"],
                "domain":           metric["domain"],
                "unit":             metric["unit"],
                "alerting_enabled": metric["alerting"]["enabled"],
                **computed,
            }

            if not computed.get("fetch_ok"):
                bucket["fetch_errors"].append(mid)
                logger.warning("snapshot_writer: [%s] %s — FAILED: %s", facility, mid, computed.get("error"))
            else:
                v_status = computed.get("validation", "passed")
                v_warns  = computed.get("warnings", [])
                if v_status != "passed":
                    bucket["validation_warnings"].append(
                        {"metric_id": mid, "validation": v_status, "warnings": v_warns}
                    )
                logger.info(
                    "snapshot_writer: [%s] %s — %s%s",
                    facility, mid, v_status,
                    f" {v_warns}" if v_warns else "",
                )

    with open(SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, default=str)

    for facility, bucket in snapshot.items():
        if not isinstance(bucket, dict) or "metrics" not in bucket:
            continue
        n_total = len(bucket["metrics"])
        n_err   = len(bucket["fetch_errors"])
        n_warn  = len(bucket["validation_warnings"])
        logger.info(
            "snapshot_writer: [%s] done — %d/%d OK, %d failed, %d with warnings",
            facility, n_total - n_err, n_total, n_err, n_warn,
        )

    return snapshot
