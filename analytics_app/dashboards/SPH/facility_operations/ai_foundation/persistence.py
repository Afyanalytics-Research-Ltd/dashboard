"""Persistence contract for the AI Foundation Intelligence run.

This module is the interface between the runner and the dashboard.

Runner   → writes IntelligenceRun as JSON to a known path
Dashboard → reads JSON with json.load(); does NOT import this module

The dashboard reads raw dict keys — PersistedBriefing field names are the public API.
Any rename here is a breaking change for the dashboard reader.

Status semantics:
  ok               — pipeline ran, trigger fired, synthesis succeeded
  no_trigger       — pipeline ran cleanly, no metric crossed threshold
  synthesis_failed — pipeline ran, trigger fired, LLM unavailable; evidence_payload is the fallback
  pipeline_failed  — DB/query error before analysis completed; problems list is empty
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class PersistedSignature(BaseModel):
    attribution: str       # department name from attribution step
    temporal_pattern: str  # "{DayName}/{HH}:00" canonical form
    cohort: str
    mechanism: str         # SUPPORTED mechs joined "+"; "UNCONFIRMED" if none


class PersistedBriefing(BaseModel):
    what: str
    where: str
    when: str
    mechanism: str
    downstream: str
    unknowns: str
    action: str


class PersistedProblem(BaseModel):
    metric_id: str
    priority_score: float
    severity: str | None = None          # from InvestigationCard: "Critical" | "Warning" | "Info"
    impact_domain: str | None = None     # from InvestigationCard: "patient_flow" | "clinical_safety" | ...
    signature: PersistedSignature
    briefing: PersistedBriefing | None  # None when synthesis_failed or pipeline_failed
    evidence_payload: str | None         # None only when pipeline_failed


class IntelligenceRun(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    run_ts: str                          # ISO 8601 datetime string — stamped by runner after return
    status: Literal["ok", "no_trigger", "synthesis_failed", "pipeline_failed"]
    problems: list[PersistedProblem]     # empty when no_trigger or pipeline_failed
