#!/usr/bin/env python3
"""
canary.py — N-record canary runs for the load pipelines: write a few records
through the real path into the real tables, read them back, and fail loudly
if any did not land intact — before the full run is started.

USED BY
  from_json_mappings_to_snowflake.py   Afya Extraction → <NAME>_RAW.EVENTS_RAW
  dags/document_pipeline.py            Drive PDFs → LandingAI → Snowflake table
  snowflake_to_v3_migration.py         <NAME>_CLEAN → V3 gateway
  dags/canary_end_to_end_dag.py        the three in sequence

  Each pipeline takes a `canary` count (0 = off; CLI --canary N, DAG param
  canary). It writes at most that many records per target table into the
  SAME schemas / tables a full run uses, then checks every one of them on the
  destination. The full run afterwards skips (V3, documents) or replaces
  (EVENTS_RAW) what the canary already wrote, so nothing lands twice.

ONE TARGET = ONE TABLE / MODEL (TargetResult)
  expected     records the canary wrote to it
  landed       of those, found on the destination when read back
  missing      record ids not found
  mismatched   {field: [record ids]} stored empty or different from what was sent
  duplicates   record ids stored more than once
  errors       anything that stopped the write or the read-back
  untested     why nothing could be checked (empty source, every record
               waiting on a parent …) — reported, not a failure on its own

  status: failed (any missing / mismatched / duplicate / error), untested, passed.
  A report fails when any target failed, or when every target is untested
  (the canary checked nothing).

RESULT (CanaryReport)
  output/canary/<pipeline>/<run_id>.json, plus <facility>_latest.json (what
  the end-to-end DAG reads). In Airflow, finish_in_airflow() also pushes the
  `pipeline_reconciliation` XCom (PIPELINE_MONITORING_README.md contract) and
  fails the task with the summary. The task failure is what alerts: the
  central failure callback (config/airflow_local_settings.py) sends it to
  Sentry like any other failed task. Messages carry table, field and record
  ids only — never values.
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

log = logging.getLogger("canary")

# the repo root's output/ (this file lives in the repo root), whatever the working directory
CANARY_DIR = Path(os.getenv("CANARY_DIR") or Path(__file__).resolve().parent / "output" / "canary")
EXAMPLES = 5                                         # record ids quoted per problem in the summary

PASSED, FAILED, UNTESTED = "passed", "failed", "untested"


class CanaryFailed(RuntimeError):
    """A canary found records that did not land intact (the message is the report summary)."""


def canary_size(value) -> int:
    """A `canary` param / flag → records per target (0 = no canary).

    Accepts an int or its text ("3"); None and "" mean off. Raises ValueError
    for anything negative or not a whole number, so a typo in a trigger form
    fails the run instead of silently running the full load."""
    if value in (None, ""):
        return 0
    if isinstance(value, bool):
        raise ValueError(f"canary must be a whole number of records, got {value!r}")
    try:
        n = int(str(value).strip())
    except ValueError:
        raise ValueError(f"canary must be a whole number of records, got {value!r}") from None
    if n < 0:
        raise ValueError(f"canary must be 0 (off) or a positive number of records, got {n}")
    return n


# ─── VALUE COMPARISON ────────────────────────────────────────────────────

_DATETIME = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}:\d{2})(?::(\d{2}))?)?")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


def normalise(value):
    """A value in the form both sides of a comparison agree on.

    Destinations hand values back in their own spelling: Snowflake VARIANT
    numbers as text, V3 booleans as 1/0, decimals as "12.00", timestamps with
    a "T", fractional seconds and a zone. Normalised:
      None / "" / whitespace        → None
      bool, numbers, numeric text   → int when whole, else Decimal
      date / datetime / ISO text    → "YYYY-MM-DD[ HH:MM:SS]" (fraction and zone dropped;
                                      midnight → the bare date)
      dict / list                   → canonical JSON of the normalised members
      a uuid                        → lower case
      other text                    → stripped text"""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, Decimal)):
        return _number(value)
    if isinstance(value, (datetime, date)):
        value = value.isoformat()
    if isinstance(value, dict):
        return json.dumps({k: normalise(v) for k, v in value.items()}, sort_keys=True, default=str)
    if isinstance(value, (list, tuple)):
        return json.dumps([normalise(v) for v in value], sort_keys=True, default=str)
    text = str(value).strip()
    if not text:
        return None
    m = _DATETIME.match(text)
    if m and len(text) <= 35:
        day, hm, sec = m.groups()
        # midnight = the bare date: a date column is often handed back as a timestamp
        return day if not hm or (hm == "00:00" and (sec or "00") == "00") else f"{day} {hm}:{sec or '00'}"
    if re.fullmatch(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", text):
        return _number(text)
    if _UUID.match(text):
        return text.lower()
    return text


def _number(value):
    """int when the number is whole, else a Decimal — so 12, 12.0 and "12.00" compare equal."""
    try:
        d = Decimal(str(value))
    except InvalidOperation:
        return str(value)
    return int(d) if d == d.to_integral_value() else d.normalize()


def compare_record(sent: dict, stored: dict, *, skip=()) -> tuple[list[str], list[str]]:
    """(fields stored differently or emptied, fields not returned at all) for one record.

    Only fields the canary sent with a value are checked — a field sent empty
    has nothing to lose. `skip` names fields the destination legitimately
    rewrites (its own id, the destination facility …)."""
    mismatched, absent = [], []
    for key, value in sent.items():
        if key in skip:
            continue
        want = normalise(value)
        if want is None:
            continue
        if key not in stored:
            absent.append(key)
        elif normalise(stored[key]) != want:
            mismatched.append(key)
    return mismatched, absent


# ─── RESULTS ─────────────────────────────────────────────────────────────

@dataclass
class TargetResult:
    """The canary outcome for one destination table / model (see the module docstring)."""
    target: str
    expected: int = 0
    landed: int = 0
    ids: list[str] = field(default_factory=list)                 # the canary's record ids, as sent
    missing: list[str] = field(default_factory=list)
    mismatched: dict[str, list[str]] = field(default_factory=dict)
    duplicates: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)              # worth knowing, not a failure
    untested: str | None = None

    @property
    def status(self) -> str:
        """failed, untested or passed — in that order of precedence."""
        if self.errors or self.missing or self.mismatched or self.duplicates:
            return FAILED
        if self.untested or not self.expected:
            return UNTESTED
        return PASSED

    def mismatch(self, field_name: str, record_id) -> None:
        """Record that `field_name` of record `record_id` did not land as sent."""
        self.mismatched.setdefault(field_name, []).append(str(record_id))

    def summary(self) -> str:
        """One line: status, counts, and the first record ids behind each problem."""
        head = f"{self.status.upper():8s} {self.target}: {self.landed}/{self.expected} landed"
        parts = []
        if self.untested and self.status == UNTESTED:
            parts.append(f"untested — {self.untested}")
        if self.missing:
            parts.append(f"{len(self.missing)} missing (ids {_ex(self.missing)})")
        for name, ids in sorted(self.mismatched.items()):
            parts.append(f"{name} not stored as sent on {len(ids)} (ids {_ex(ids)})")
        if self.duplicates:
            parts.append(f"{len(self.duplicates)} stored more than once (ids {_ex(self.duplicates)})")
        parts += [f"error: {e}" for e in self.errors[:EXAMPLES]]
        return head + (" · " + " · ".join(parts) if parts else "")

    def to_dict(self) -> dict:
        """JSON-safe form (status included) for reports and XCom."""
        return {**asdict(self), "status": self.status}

    @classmethod
    def from_dict(cls, d: dict) -> "TargetResult":
        """Inverse of to_dict (the derived status is ignored)."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _ex(ids) -> str:
    ids = list(ids)
    return ", ".join(str(i) for i in ids[:EXAMPLES]) + (" …" if len(ids) > EXAMPLES else "")


@dataclass
class CanaryReport:
    """Every target of one canary run, with the verdict and where it was saved."""
    pipeline: str
    size: int
    facility: str = ""
    run_id: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    targets: list[TargetResult] = field(default_factory=list)
    finished_at: str = ""

    def add(self, target: TargetResult) -> TargetResult:
        """Add one target's result; returns it."""
        self.targets.append(target)
        return target

    @property
    def failed(self) -> list[TargetResult]:
        return [t for t in self.targets if t.status == FAILED]

    @property
    def tested_nothing(self) -> bool:
        return all(t.status == UNTESTED for t in self.targets)

    @property
    def ok(self) -> bool:
        """True when no target failed and at least one was actually checked."""
        return not self.failed and not self.tested_nothing

    def summary(self) -> str:
        """The multi-line verdict printed by the CLI and raised by a failed task."""
        counts = {s: sum(1 for t in self.targets if t.status == s) for s in (PASSED, FAILED, UNTESTED)}
        verdict = "PASSED" if self.ok else ("FAILED — nothing could be checked" if not self.failed else "FAILED")
        lines = [f"Canary {self.pipeline}{f' [{self.facility}]' if self.facility else ''} · "
                 f"{self.size} record(s) per target · run {self.run_id}: {verdict} "
                 f"({counts[PASSED]} passed, {counts[FAILED]} failed, {counts[UNTESTED]} untested)"]
        order = {FAILED: 0, UNTESTED: 1, PASSED: 2}
        lines += [f"  {t.summary()}" for t in sorted(self.targets, key=lambda t: (order[t.status], t.target))]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """JSON-safe form of the whole report."""
        return {"pipeline": self.pipeline, "facility": self.facility, "run_id": self.run_id,
                "size": self.size, "ok": self.ok, "finished_at": self.finished_at,
                "targets": [t.to_dict() for t in self.targets]}

    def reconciliation(self) -> dict:
        """The `pipeline_reconciliation` XCom payload (PIPELINE_MONITORING_README.md):
        one entry per target, counts and ids only."""
        return {"canary": True, "run_id": self.run_id, "targets": [
            {"target": t.target, "expected_target_rows": t.expected, "loaded_target_rows": t.landed,
             "duplicate_count": len(t.duplicates), "mismatched_fields": sorted(t.mismatched),
             "canary_status": t.status} for t in self.targets]}

    def save(self, directory: Path | str | None = None) -> Path:
        """Write <dir>/<pipeline>/<run_id>.json and <facility>_latest.json; returns the first.

        Written atomically (temp file + rename), so a reader never sees half
        a report."""
        self.finished_at = self.finished_at or datetime.now(timezone.utc).isoformat()
        out = Path(directory or CANARY_DIR) / self.pipeline
        out.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), indent=1, default=str)
        path = out / f"{self.run_id}.json"
        for p in (path, out / f"{self.facility or 'all'}_latest.json"):
            tmp = p.with_suffix(f".{uuid.uuid4().hex}.tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, p)
        log.info("canary report saved: %s", path)
        return path

    def raise_if_failed(self) -> None:
        """Raise CanaryFailed with the summary unless the report is ok."""
        if not self.ok:
            raise CanaryFailed(self.summary())


def load_latest(pipeline: str, facility: str, directory: Path | str | None = None) -> dict | None:
    """The newest saved report of `pipeline` for `facility` (as saved by
    CanaryReport.save), or None when there is none."""
    path = Path(directory or CANARY_DIR) / pipeline / f"{facility or 'all'}_latest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def finish_in_airflow(report: CanaryReport, context) -> dict:
    """End a canary task: log and save the report, push `pipeline_reconciliation`,
    and fail the task (AirflowFailException, no retry) when the canary failed —
    the central failure callback then alerts Sentry. Returns the report dict."""
    try:
        from airflow.sdk.exceptions import AirflowFailException
    except ImportError:   # older Airflow
        from airflow.exceptions import AirflowFailException

    text = report.summary()
    (log.info if report.ok else log.error)("%s", text)
    report.save()
    context["ti"].xcom_push(key="pipeline_reconciliation", value=report.reconciliation())
    if not report.ok:
        raise AirflowFailException(text)
    return report.to_dict()
