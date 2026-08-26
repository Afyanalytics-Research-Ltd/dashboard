from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class MetricDefinition(BaseModel):
    metric_id: str
    value_query: str          # must return exactly: value (float), n (int), freshness_date (date)
    baseline_query: str       # must return exactly: baseline (float) — single scalar
    freshness_requirement_hours: int
    minimum_sample: int


class MetricState(BaseModel):
    value: float
    baseline: float
    change: float | None      # (value - baseline) / baseline; None when baseline == 0
    n: int
    freshness_date: date
    status: Literal["FRESH", "STALE", "INSUFFICIENT_SAMPLE", "QUERY_FAILED"]


class Trigger(BaseModel):
    metric_id: str
    left_ref: Literal["value", "baseline", "change"] = "value"
    operator: Literal["gt", "lt", "gte", "lte"]
    threshold_type: Literal["relative", "absolute"]
    threshold_ref: Literal["value", "baseline", "change"] | None = None
    threshold_val: float | None = None

    @model_validator(mode="after")
    def validate_threshold(self) -> "Trigger":
        if self.threshold_type == "relative":
            if self.threshold_ref is None:
                raise ValueError("relative trigger requires threshold_ref")
            if self.threshold_val is not None:
                raise ValueError("relative trigger must not have threshold_val")
        else:
            if self.threshold_val is None:
                raise ValueError("absolute trigger requires threshold_val")
            if self.threshold_ref is not None:
                raise ValueError("absolute trigger must not have threshold_ref")
        if self.threshold_ref is not None and self.left_ref == self.threshold_ref:
            raise ValueError("left_ref and threshold_ref must not be the same field")
        return self


class TriggerResult(str, Enum):
    FIRE = "FIRE"
    DO_NOT_FIRE = "DO_NOT_FIRE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class StepResult(BaseModel):
    step_id: str
    status: Literal["SUPPORTED", "NOT_FOUND", "NOT_APPLICABLE", "INSUFFICIENT_DATA"]
    evidence: dict | None
    cohort: Any | None        # propagated to downstream steps; None for steps 1–2


class InvestigationStep(BaseModel):
    step_id: str
    purpose: str
    query: str | None         # None = reads MetricState directly (quantify step)
    uses_cohort: bool = False  # True = query expects one %s bind param (cohort value from prior step)
    meta: dict = Field(default_factory=dict)  # card-specific handler config (peak_window_hours, reason strings, etc.)


class InvestigationCard(BaseModel):
    id: str
    trigger_metric_id: str
    steps: list[InvestigationStep]
    severity: Literal["Critical", "Warning", "Info"]
    impact_domain: Literal["patient_flow", "clinical_safety", "capacity", "efficiency", "data_quality"]
    sample_label: str = "records"
    scope_note: str | None = None


class ProblemSignature(BaseModel):
    attribution: str       # department name from attribution step
    temporal_pattern: str  # canonical "{DayName}/{HH}:00" — peak DOW + peak hour for exact matching
    cohort: str            # cohort key propagated from attribution
    mechanism: str         # SUPPORTED mechanism(s) joined with "+"; "UNCONFIRMED" if none supported


class OperationalProblem(BaseModel):
    card_id: str
    metric_id: str
    signature: ProblemSignature
    metric_state: MetricState
    step_results: list[StepResult]


class PrioritisedProblem(BaseModel):
    problem: OperationalProblem   # reference — no second copy of evidence
    severity_weight: int
    impact_weight: int
    magnitude: float              # abs(metric_state.change)
    priority_score: float         # severity_weight x impact_weight x magnitude


class OperationalBriefing(BaseModel):
    what: str       # metric elevation, magnitude, baseline period
    where: str      # attributed department + key stats + any data limitation
    when: str       # temporal pattern — day, hour, window
    mechanism: str  # SUPPORTED mechanisms + NOT_APPLICABLE items with reasons
    downstream: str # evidenced downstream effects for cohort
    unknowns: str   # what cannot be established and why
    action: str     # recommended action grounded only in SUPPORTED evidence
