"""
chat_logger.py — Structured event logger for the KSH AI Chat Layer.

All events are written as one JSON object per line (JSONL).
Callers use log_event(event_type, payload) only — file paths are internal.

Event types:
    sql_request    — SQL generation, validation, and execution audit
    context_gap    — Question matched no context documents
    context_skip   — Context file skipped due to prompt budget
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

_CHAT = Path(__file__).resolve().parent

_LOG_FILES: dict = {
    "sql_request":  _CHAT / "sql_requests.jsonl",
    "context_gap":  _CHAT / "chat_gaps.jsonl",
    "context_skip": _CHAT / "context_budget.jsonl",
}

_lock = threading.Lock()


def log_event(event_type: str, payload: dict) -> None:
    """
    Append one structured event to the appropriate JSONL log file.

    Injects 'ts' (UTC ISO-8601) if absent. Silently ignores unknown
    event_type values. Thread-safe for single-process deployments.
    """
    path = _LOG_FILES.get(event_type)
    if path is None:
        return
    if "ts" not in payload:
        payload = {**payload, "ts": datetime.now(timezone.utc).isoformat()}
    line = json.dumps(payload, default=str) + "\n"
    try:
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
    except OSError:
        pass
