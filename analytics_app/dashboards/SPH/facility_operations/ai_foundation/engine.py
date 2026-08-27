from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from ai_foundation.contracts import (
    InvestigationCard,
    InvestigationStep,
    MetricDefinition,
    MetricState,
    OperationalProblem,
    PrioritisedProblem,
    ProblemSignature,
    StepResult,
    Trigger,
    TriggerResult,
)

def _cursor_to_df(conn, sql: str, params=None) -> pd.DataFrame:
    cur = conn.cursor()
    try:
        cur.execute(sql, params)
        cols = [c[0].lower() for c in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)
    finally:
        cur.close()


_SENTINEL = MetricState(
    value=0.0,
    baseline=0.0,
    change=None,
    n=0,
    freshness_date=date.min,
    status="QUERY_FAILED",
)

_OPS = {
    "gt":  lambda a, b: a > b,
    "lt":  lambda a, b: a < b,
    "gte": lambda a, b: a >= b,
    "lte": lambda a, b: a <= b,
}


def _hours_since(d: date) -> float:
    return (datetime.utcnow().date() - d).total_seconds() / 3600


def populate_metric_state(defn: MetricDefinition, get_connection=None) -> MetricState:
    if get_connection is None:
        from dashboard.db import get_connection as _gc
        get_connection = _gc

    try:
        conn = get_connection()
        try:
            value_df = _cursor_to_df(conn, defn.value_query)
            baseline_df = _cursor_to_df(conn, defn.baseline_query)
        finally:
            conn.close()
    except Exception:
        return _SENTINEL

    # Step 2: validate columns
    if not {"value", "n", "freshness_date"}.issubset(set(value_df.columns.str.lower())):
        return _SENTINEL
    if "baseline" not in baseline_df.columns.str.lower().tolist():
        return _SENTINEL

    try:
        row = value_df.iloc[0]
        raw_value = float(row["value"])
        raw_n = int(row["n"])
        raw_freshness = row["freshness_date"]
        if isinstance(raw_freshness, str):
            raw_freshness = date.fromisoformat(raw_freshness)
        elif hasattr(raw_freshness, "date"):
            raw_freshness = raw_freshness.date()
        raw_baseline = float(baseline_df.iloc[0]["baseline"])
    except Exception:
        return _SENTINEL

    change = (raw_value - raw_baseline) / raw_baseline if raw_baseline != 0 else None

    # Step 4: minimum sample
    if raw_n < defn.minimum_sample:
        return MetricState(
            value=raw_value, baseline=raw_baseline, change=change,
            n=raw_n, freshness_date=raw_freshness, status="INSUFFICIENT_SAMPLE",
        )

    # Step 5: freshness
    if _hours_since(raw_freshness) > defn.freshness_requirement_hours:
        return MetricState(
            value=raw_value, baseline=raw_baseline, change=change,
            n=raw_n, freshness_date=raw_freshness, status="STALE",
        )

    return MetricState(
        value=raw_value, baseline=raw_baseline, change=change,
        n=raw_n, freshness_date=raw_freshness, status="FRESH",
    )


def evaluate_trigger(state: MetricState, trigger: Trigger) -> TriggerResult:
    if state.status != "FRESH":
        return TriggerResult.NOT_APPLICABLE

    left_val = getattr(state, trigger.left_ref)
    if left_val is None:
        return TriggerResult.NOT_APPLICABLE

    if trigger.threshold_type == "relative":
        right_val = getattr(state, trigger.threshold_ref)
        if right_val is None:
            return TriggerResult.NOT_APPLICABLE
    else:
        right_val = trigger.threshold_val

    return TriggerResult.FIRE if _OPS[trigger.operator](left_val, right_val) else TriggerResult.DO_NOT_FIRE


def _execute_step(step: InvestigationStep, state: MetricState, conn, cohort=None) -> StepResult:
    if step.query is None:
        return StepResult(
            step_id=step.step_id,
            status="SUPPORTED",
            evidence={
                "value": state.value,
                "baseline": state.baseline,
                "delta_mins": state.value - state.baseline,
                "change": state.change,
                "n": state.n,
                "freshness_date": str(state.freshness_date),
            },
            cohort=None,
        )

    try:
        params = (cohort,) if step.uses_cohort else None
        df = _cursor_to_df(conn, step.query, params=params)
    except Exception:
        return StepResult(step_id=step.step_id, status="INSUFFICIENT_DATA", evidence=None, cohort=None)

    if df.empty:
        return StepResult(step_id=step.step_id, status="NOT_FOUND", evidence=None, cohort=None)

    if step.step_id == "dept_attribution":
        return _attribution_result(step, df, state)
    if step.step_id == "temporal_pattern":
        return _temporal_result(step, df, cohort)
    if step.step_id == "mechanism_test":
        return _mechanism_result(step, df, cohort)
    if step.step_id == "downstream_pharmacy":
        return _downstream_result(step, df, cohort)
    if step.step_id == "downstream_incomplete_care":
        return _incomplete_care_result(step, df, cohort)

    # generic fallback for future steps
    return StepResult(
        step_id=step.step_id,
        status="SUPPORTED",
        evidence={"rows": df.to_dict(orient="records")},
        cohort=None,
    )


def _attribution_result(step: InvestigationStep, df: pd.DataFrame, state: MetricState) -> StepResult:
    departments = [
        {
            "dept": row["dept"],
            "visits": int(row["visits"]),
            "valid_n": int(row["valid_n"]),
            "p50_mins": int(row["p50_mins"]) if row["p50_mins"] is not None else None,
        }
        for _, row in df.iterrows()
    ]
    top = departments[0]
    above_baseline = (top["p50_mins"] or 0) > state.baseline
    coverage_pct = round(top["valid_n"] / top["visits"] * 100, 1) if top["visits"] > 0 else None

    return StepResult(
        step_id=step.step_id,
        status="SUPPORTED" if above_baseline else "NOT_FOUND",
        evidence={
            "departments": departments,
            "top_attribution": top["dept"],
            "anchor": "MAX(visit_date) from rpt_ortho_patient_journey",
            "limitation": (
                f"{top['dept']}: {coverage_pct}% timestamp coverage "
                f"({top['valid_n']}/{top['visits']} visits)"
            ),
        },
        cohort=top["dept"],
    )


def _temporal_result(step: InvestigationStep, df: pd.DataFrame, cohort) -> StepResult:
    from collections import defaultdict

    cells = [
        {
            "dow_num": int(row["dow_num"]),
            "day_name": str(row["day_name"]),
            "hour_of_day": int(row["hour_of_day"]),
            "visit_count": int(row["visit_count"]),
            "median_wait_mins": int(row["median_wait_mins"]) if pd.notna(row["median_wait_mins"]) else None,
        }
        for _, row in df.iterrows()
    ]
    valid_cells = [c for c in cells if c["median_wait_mins"] is not None]
    if not valid_cells:
        return StepResult(step_id=step.step_id, status="NOT_FOUND", evidence=None, cohort=None)

    dow_medians: dict = defaultdict(list)
    for c in valid_cells:
        dow_medians[(c["dow_num"], c["day_name"])].append(c["median_wait_mins"])
    peak_dow_key = max(dow_medians, key=lambda k: sum(dow_medians[k]) / len(dow_medians[k]))
    peak_dow_avg = round(sum(dow_medians[peak_dow_key]) / len(dow_medians[peak_dow_key]), 0)

    hour_medians: dict = defaultdict(list)
    for c in valid_cells:
        hour_medians[c["hour_of_day"]].append(c["median_wait_mins"])
    peak_hour_key = max(hour_medians, key=lambda k: sum(hour_medians[k]) / len(hour_medians[k]))
    peak_hour_avg = round(sum(hour_medians[peak_hour_key]) / len(hour_medians[peak_hour_key]), 0)

    anchor = step.meta.get("temporal_anchor", "MAX(visit_date) from rpt_ortho_patient_journey")
    return StepResult(
        step_id=step.step_id,
        status="SUPPORTED",
        evidence={
            "cohort_dept": cohort,
            "cells": cells,
            "peak_dow": {"dow_num": peak_dow_key[0], "day_name": peak_dow_key[1], "avg_median_wait_mins": peak_dow_avg},
            "peak_hour": {"hour_of_day": peak_hour_key, "avg_median_wait_mins": peak_hour_avg},
            "anchor": anchor,
        },
        cohort=None,
    )


def run_card(card: InvestigationCard, state: MetricState, get_connection=None) -> list[StepResult]:
    if get_connection is None:
        from dashboard.db import get_connection as _gc
        get_connection = _gc

    results: list[StepResult] = []

    try:
        conn = get_connection()
    except Exception:
        return [
            StepResult(step_id=s.step_id, status="INSUFFICIENT_DATA", evidence=None, cohort=None)
            for s in card.steps
        ]

    try:
        cohort = None
        for step in card.steps:
            result = _execute_step(step, state, conn, cohort=cohort)
            results.append(result)
            if result.cohort is not None:
                cohort = result.cohort
    finally:
        conn.close()

    return results


def _extract_supported_mechanisms(mech_result: StepResult) -> str:
    if mech_result.status not in ("SUPPORTED", "NOT_FOUND") or mech_result.evidence is None:
        return "UNCONFIRMED"
    mechs = mech_result.evidence.get("mechanisms", {})
    supported = sorted(k for k, v in mechs.items() if v.get("status") == "SUPPORTED")
    return "+".join(supported) if supported else "UNCONFIRMED"


def build_problem(
    card: InvestigationCard,
    state: MetricState,
    step_results: list[StepResult],
) -> OperationalProblem | None:
    """Build an OperationalProblem from validated step results. Returns None if attribution is not SUPPORTED."""
    by_id = {r.step_id: r for r in step_results}

    attr = by_id.get("dept_attribution")
    if attr is None or attr.status != "SUPPORTED" or attr.evidence is None:
        return None

    attribution = attr.evidence.get("top_attribution", "UNKNOWN")
    cohort = attr.cohort or attribution

    temporal = by_id.get("temporal_pattern")
    if temporal and temporal.status == "SUPPORTED" and temporal.evidence:
        peak_dow = temporal.evidence.get("peak_dow", {})
        peak_hour = temporal.evidence.get("peak_hour", {})
        day_name = peak_dow.get("day_name", "UNKNOWN")
        hour = peak_hour.get("hour_of_day", 0)
        temporal_pattern = f"{day_name}/{hour:02d}:00"
    else:
        temporal_pattern = "UNKNOWN"

    mech = by_id.get("mechanism_test")
    mechanism = _extract_supported_mechanisms(mech) if mech else "UNCONFIRMED"

    sig = ProblemSignature(
        attribution=attribution,
        temporal_pattern=temporal_pattern,
        cohort=cohort,
        mechanism=mechanism,
    )

    return OperationalProblem(
        card_id=card.id,
        metric_id=card.trigger_metric_id,
        signature=sig,
        metric_state=state,
        step_results=step_results,
    )


def prioritise_problems(problems: list[OperationalProblem]) -> list[PrioritisedProblem]:
    """Score each problem and return ranked list (highest score first).

    Inputs are deterministic: severity + impact_domain from card config,
    magnitude from MetricState.change. No LLM involvement.
    """
    from ai_foundation.registry import CARD_REGISTRY, IMPACT_WEIGHTS, SEVERITY_WEIGHTS

    ranked: list[PrioritisedProblem] = []
    for p in problems:
        card = CARD_REGISTRY.get(p.metric_id)
        if card is None:
            continue
        sw = SEVERITY_WEIGHTS[card.severity]
        iw = IMPACT_WEIGHTS[card.impact_domain]
        change = p.metric_state.change
        magnitude = round(abs(change), 4) if change is not None else 0.0
        score = round(sw * iw * magnitude, 4)
        ranked.append(PrioritisedProblem(
            problem=p,
            severity_weight=sw,
            impact_weight=iw,
            magnitude=magnitude,
            priority_score=score,
        ))

    ranked.sort(key=lambda x: x.priority_score, reverse=True)
    return ranked


def group_problems(problems: list[OperationalProblem]) -> list[OperationalProblem]:
    """Exact-match dedup by ProblemSignature. v1: no fuzzy matching."""
    seen: dict[tuple, OperationalProblem] = {}
    for p in problems:
        s = p.signature
        key = (s.attribution, s.temporal_pattern, s.cohort, s.mechanism)
        if key not in seen:
            seen[key] = p
    return list(seen.values())


def _mechanism_result(step: InvestigationStep, df: pd.DataFrame, cohort) -> StepResult:
    from ai_foundation.registry import MECHANISM_PEAK_WINDOW_HOURS, MECHANISM_VOLUME_SPIKE_RATIO

    peak_window = tuple(step.meta.get("peak_window_hours", MECHANISM_PEAK_WINDOW_HOURS))
    capacity_reason = step.meta.get(
        "capacity_not_applicable_reason",
        "No Physiotherapy staffing data in schema — SHIFTS covers OPD and Pharmacy only",
    )
    scheduling_reason = step.meta.get(
        "scheduling_not_applicable_reason",
        "No appointments/clinic session table at source — DATA_GAPS.md §3",
    )

    hourly = [
        {
            "hour_of_day": int(row["hour_of_day"]),
            "total_arrivals": int(row["total_arrivals"]),
            "days_observed": int(row["days_observed"]),
            "avg_daily_arrivals": float(row["avg_daily_arrivals"]),
            "overall_avg": float(row["overall_avg_hourly_arrivals"]),
            "volume_ratio": float(row["volume_ratio"]) if pd.notna(row["volume_ratio"]) else None,
        }
        for _, row in df.iterrows()
    ]

    peak_rows = [h for h in hourly if h["hour_of_day"] in peak_window]
    peak_avg = (
        sum(h["avg_daily_arrivals"] for h in peak_rows) / len(peak_rows)
        if peak_rows else 0.0
    )
    overall_avg = hourly[0]["overall_avg"] if hourly else 0.0
    peak_ratio = round(peak_avg / overall_avg, 2) if overall_avg > 0 else None

    volume_status = (
        "SUPPORTED" if (peak_ratio is not None and peak_ratio >= MECHANISM_VOLUME_SPIKE_RATIO)
        else "NOT_FOUND"
    )

    return StepResult(
        step_id=step.step_id,
        status=volume_status,
        evidence={
            "mechanisms": {
                "volume": {
                    "status": volume_status,
                    "peak_window_hours": list(peak_window),
                    "peak_window_avg_daily_arrivals": round(peak_avg, 1),
                    "overall_avg_hourly_arrivals": overall_avg,
                    "peak_ratio": peak_ratio,
                    "threshold": MECHANISM_VOLUME_SPIKE_RATIO,
                    "hourly_data": hourly,
                },
                "capacity": {
                    "status": "NOT_APPLICABLE",
                    "reason": capacity_reason,
                },
                "scheduling": {
                    "status": "NOT_APPLICABLE",
                    "reason": scheduling_reason,
                },
            },
            "cohort_dept": cohort,
        },
        cohort=None,
    )


_DOWNSTREAM_MIN_VISITS = 10


def _downstream_result(step: InvestigationStep, df: pd.DataFrame, cohort) -> StepResult:
    from ai_foundation.registry import DOWNSTREAM_PHARMACY_ELEVATION_RATIO

    row = df.iloc[0]
    cohort_visit_n = int(row["cohort_visit_n"])

    if cohort_visit_n == 0:
        return StepResult(
            step_id=step.step_id,
            status="NOT_APPLICABLE",
            evidence={
                "cohort_dept": cohort,
                "cohort_visit_n": 0,
                "reason": f"{cohort} has 0 pharmacy records in the 28-day window — patient pathway does not include pharmacy",
            },
            cohort=None,
        )

    if cohort_visit_n < _DOWNSTREAM_MIN_VISITS:
        return StepResult(
            step_id=step.step_id,
            status="INSUFFICIENT_DATA",
            evidence={
                "cohort_dept": cohort,
                "cohort_visit_n": cohort_visit_n,
                "reason": f"{cohort_visit_n} cohort visits have pharmacy records — minimum {_DOWNSTREAM_MIN_VISITS} required",
            },
            cohort=None,
        )

    cohort_p50 = float(row["cohort_p50_mins"]) if pd.notna(row["cohort_p50_mins"]) else None
    baseline_p50 = float(row["baseline_p50_mins"]) if pd.notna(row["baseline_p50_mins"]) else None

    if cohort_p50 is None:
        return StepResult(
            step_id=step.step_id,
            status="NOT_APPLICABLE",
            evidence={
                "cohort_dept": cohort,
                "cohort_visit_n": cohort_visit_n,
                "reason": "No valid pharmacy TAT records for cohort in the 28-day window",
            },
            cohort=None,
        )

    downstream_ratio = (
        round(cohort_p50 / baseline_p50, 2)
        if (baseline_p50 is not None and baseline_p50 > 0) else None
    )
    downstream_elevated = (
        downstream_ratio is not None and downstream_ratio >= DOWNSTREAM_PHARMACY_ELEVATION_RATIO
    )

    return StepResult(
        step_id=step.step_id,
        status="SUPPORTED",
        evidence={
            "cohort_dept": cohort,
            "cohort_pharm_p50_mins": int(cohort_p50),
            "baseline_pharm_p50_mins": int(baseline_p50) if baseline_p50 is not None else None,
            "downstream_ratio": downstream_ratio,
            "downstream_elevated": downstream_elevated,
            "elevation_threshold": DOWNSTREAM_PHARMACY_ELEVATION_RATIO,
            "cohort_visit_n": cohort_visit_n,
            "cohort_item_n": int(row["cohort_item_n"]),
            "baseline_visit_n": int(row["baseline_visit_n"]),
        },
        cohort=None,
    )


def _incomplete_care_result(step: InvestigationStep, df: pd.DataFrame, cohort) -> StepResult:
    from ai_foundation.registry import DOWNSTREAM_INCOMPLETE_ELEVATION_RATIO

    row = df.iloc[0]
    cohort_visits = int(row["cohort_visits"])

    if cohort_visits == 0:
        return StepResult(
            step_id=step.step_id,
            status="NOT_APPLICABLE",
            evidence={
                "cohort_dept": cohort,
                "cohort_visits": 0,
                "reason": f"{cohort} has 0 visits in the 28-day window",
            },
            cohort=None,
        )

    if cohort_visits < _DOWNSTREAM_MIN_VISITS:
        return StepResult(
            step_id=step.step_id,
            status="INSUFFICIENT_DATA",
            evidence={
                "cohort_dept": cohort,
                "cohort_visits": cohort_visits,
                "reason": f"{cohort_visits} cohort visits — minimum {_DOWNSTREAM_MIN_VISITS} required",
            },
            cohort=None,
        )

    cohort_incomplete_pct = float(row["cohort_incomplete_pct"]) if pd.notna(row["cohort_incomplete_pct"]) else None
    baseline_incomplete_pct = float(row["baseline_incomplete_pct"]) if pd.notna(row["baseline_incomplete_pct"]) else None

    incomplete_ratio = (
        round(cohort_incomplete_pct / baseline_incomplete_pct, 2)
        if (cohort_incomplete_pct is not None and baseline_incomplete_pct and baseline_incomplete_pct > 0)
        else None
    )
    downstream_elevated = (
        incomplete_ratio is not None and incomplete_ratio >= DOWNSTREAM_INCOMPLETE_ELEVATION_RATIO
    )

    return StepResult(
        step_id=step.step_id,
        status="SUPPORTED",
        evidence={
            "cohort_dept": cohort,
            "cohort_incomplete_pct": cohort_incomplete_pct,
            "baseline_incomplete_pct": baseline_incomplete_pct,
            "incomplete_ratio": incomplete_ratio,
            "downstream_elevated": downstream_elevated,
            "elevation_threshold": DOWNSTREAM_INCOMPLETE_ELEVATION_RATIO,
            "cohort_visits": cohort_visits,
            "cohort_incomplete_n": int(row["cohort_incomplete_n"]),
            "baseline_visits": int(row["baseline_visits"]),
            "baseline_incomplete_n": int(row["baseline_incomplete_n"]),
        },
        cohort=None,
    )
