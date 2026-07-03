"""
router.py — Question routing for the KSH AI Chat Layer.

Additive routing — both flags are independent:
  use_snapshot  True when the question matches a tracked metric in the registry.
  use_sql       True when the question likely needs a database query — either
                because there is no snapshot match, or because the question has
                signals that the snapshot cannot answer (time-specific dates,
                cross-ward comparisons, breakdowns, totals).

sql_executor receives use_sql and makes the final call. It returns
not_required when snapshot + context already answer the question.

Both paths always return context_keys — the relevant context/*.md files
to include for interpretation alongside whichever data path is used.

Note: context matching uses rglob (includes investigations/) so all 18
context files are reachable. The previous views.py glob("*.md") only
reached top-level files.
"""

import json
import re
import threading
from pathlib import Path

_CHAT = Path(__file__).resolve().parent        # chat/
_PRIVATE = _CHAT.parent                        # facility_utilization/

# ---------------------------------------------------------------------------
# Registry cache — stable build artifact, never changes at runtime
# ---------------------------------------------------------------------------
_registry_cache: dict | None = None
_registry_lock = threading.Lock()


def _get_registry() -> dict:
    global _registry_cache
    if _registry_cache is not None:
        return _registry_cache
    with _registry_lock:
        if _registry_cache is None:
            _registry_cache = json.loads(
                (_CHAT / "metrics_registry.json").read_text(encoding="utf-8")
            )
    return _registry_cache


# ---------------------------------------------------------------------------
# Snapshot cache — refreshed by `refresh_snapshot` management command;
# reloaded here when the file changes on disk (mtime guard)
# ---------------------------------------------------------------------------
_snap_cache: dict = {}
_snap_mtime: float = 0.0
_snap_lock = threading.Lock()


def _get_snapshot() -> dict:
    global _snap_cache, _snap_mtime
    path = _CHAT / "metrics_snapshot.json"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    if mtime <= _snap_mtime:
        return _snap_cache
    with _snap_lock:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return _snap_cache
        if mtime <= _snap_mtime:
            return _snap_cache
        try:
            _snap_cache = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return _snap_cache
        _snap_mtime = mtime
    return _snap_cache


# ---------------------------------------------------------------------------
# Context keyword index — file stem → keywords list, mtime-cached.
# Uses rglob so context/investigations/*.md files are included.
# ---------------------------------------------------------------------------
_ctx_kw_index: list = []   # [(keywords: list[str], stem: str), ...]
_ctx_kw_mtime: float = 0.0
_ctx_kw_lock = threading.Lock()


def _get_ctx_kw_index() -> list:
    global _ctx_kw_index, _ctx_kw_mtime
    ctx_dir = _PRIVATE / "context"
    try:
        latest = max(
            (f.stat().st_mtime for f in ctx_dir.rglob("*.md")), default=0.0
        )
    except OSError:
        latest = 0.0
    if latest <= _ctx_kw_mtime:
        return _ctx_kw_index
    with _ctx_kw_lock:
        try:
            latest = max(
                (f.stat().st_mtime for f in ctx_dir.rglob("*.md")), default=0.0
            )
        except OSError:
            latest = 0.0
        if latest <= _ctx_kw_mtime:
            return _ctx_kw_index
        index = []
        for f in sorted(ctx_dir.rglob("*.md")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.startswith("**keywords:**"):
                    kwds = [
                        k.strip().lower()
                        for k in line.replace("**keywords:**", "").split(",")
                        if k.strip()
                    ]
                    index.append((kwds, str(f.relative_to(ctx_dir))))
                    break
        _ctx_kw_index = index
        _ctx_kw_mtime = latest
    return _ctx_kw_index


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def _score(text: str, keywords: list) -> int:
    """Count keyword/phrase matches (case-insensitive, handles simple s-plurals only)."""
    return sum(
        1 for kw in keywords
        if re.search(r"\b" + re.escape(kw.lower()) + r"s?\b", text)
    )


# Signals that suggest the question needs a database query even when a
# snapshot match exists. Router sets use_sql=True on any match; sql_executor
# makes the final call and returns not_required if the snapshot answers it.
_SQL_SIGNALS = re.compile(
    r"\b("
    r"last\s+year|last\s+month|last\s+quarter"
    r"|last\s+\d+\s+months?|last\s+\d+\s+years?"
    r"|over\s+the\s+last|over\s+the\s+past|in\s+the\s+last|in\s+the\s+past"
    r"|past\s+\d+\s+months?|past\s+\d+\s+years?"
    r"|january|february|march|april|may|june|july|august|september|october|november|december"
    r"|q[1-4]\b|quarter"
    r"|in\s+\d{4}|since\s+\d{4}|\d{4}\s*[-–]\s*\d{4}"
    r"|trend|over\s+time|history|historical"
    r"|compare|versus|\bvs\b|breakdown"
    r"|each\s+ward|which\s+ward|by\s+ward|per\s+ward|ward.level|ward\s+by\s+ward"
    r"|each\s+doctor|by\s+doctor|per\s+doctor"
    r"|how\s+many\s+total|total\s+across|across\s+all|sum\s+of"
    r"|highest|lowest|most|least|ranking|rank"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def route(question: str) -> dict:
    """
    Route an incoming question to the appropriate answer paths.

    Returns a dict with:
        use_snapshot  True when question matches a tracked metric in registry.
        use_sql       True when question likely needs a database query.
                      sql_executor makes the final call — it can return
                      not_required if snapshot + context already answer it.
        matches       list of registry hits, sorted by score desc.
                      Each item: {metric_id, score, registry, snapshot}
                      snapshot is None if not present in the snapshot file.
                      Check snapshot["fetch_ok"] before citing values.
        context_keys  list of context file stems sorted by relevance desc.
                      Empty list means no keyword overlap found — caller
                      should fall back to including all context or notices.
    """
    q = question.lower()
    registry = _get_registry()
    snap_metrics = _get_snapshot().get("KISUMU_CLEAN", {}).get("metrics", {})
    kw_index = _get_ctx_kw_index()

    # Registry match → snapshot candidates
    matches = []
    for entry in registry["metrics"]:
        sc = _score(q, entry["keywords"])
        if sc > 0:
            matches.append({
                "metric_id": entry["metric_id"],
                "score": sc,
                "registry": entry,
                "snapshot": snap_metrics.get(entry["metric_id"]),
            })
    matches.sort(key=lambda x: (-x["score"], x["metric_id"]))
    # When a specific match exists (score > 1), drop lower-scoring noise.
    # Generic questions (all score 1) keep the full set for broad coverage.
    if matches and matches[0]["score"] > 1:
        top = matches[0]["score"]
        matches = [m for m in matches if m["score"] == top]

    # Context keyword match → relevant files
    ctx_scores: dict = {}
    for kwds, stem in kw_index:
        sc = _score(q, kwds)
        if sc > 0:
            ctx_scores[stem] = ctx_scores.get(stem, 0) + sc
    context_keys = sorted(ctx_scores, key=lambda k: (-ctx_scores[k], k))

    use_snapshot = bool(matches)
    use_sql = (not matches) or bool(_SQL_SIGNALS.search(question))

    return {
        "use_snapshot": use_snapshot,
        "use_sql":      use_sql,
        "matches":      matches,
        "context_keys": context_keys,
    }
