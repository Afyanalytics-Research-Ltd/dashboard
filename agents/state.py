from __future__ import annotations

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

# Cap how many {"role", "content"} turns persist in a thread's checkpoint —
# bounds state size for long-running conversations. This is independent of
# how many of those turns nodes.py actually sends to the LLM per call (fewer,
# for prompt-cost reasons); see nodes._recent_history.
MAX_HISTORY_MESSAGES = 40


def _append_and_trim(existing: list[dict], new: list[dict]) -> list[dict]:
    return (existing + new)[-MAX_HISTORY_MESSAGES:]


class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    question: str           # original natural-language question
    user_id: str            # identifier for the requesting user
    user_phone: Optional[str]  # WhatsApp-capable phone e.g. "254712345678"
    callback_url: str       # POST target for async Phase-2 result delivery
    thread_id: str          # LangGraph thread ID (mirrors configurable key)

    # ── Classification ─────────────────────────────────────────────────────
    matched_metric: Optional[dict]      # {id, name, cube_query, filters}
    classification_confidence: float    # 0.0–1.0; below threshold → fallback

    # ── Execution ──────────────────────────────────────────────────────────
    cube_query: Optional[dict]   # final Cube query dict sent to /load
    raw_result: Optional[dict]   # raw Cube API response

    # ── Output ─────────────────────────────────────────────────────────────
    formatted_result: Optional[dict]  # shaped response returned to caller

    # ── Control flow ───────────────────────────────────────────────────────
    is_resumed: bool             # True after human completes the loop
    fallback_reason: Optional[str]
    resume_data: Optional[dict]  # payload passed by human via Command(resume=…)

    # ── Conversation memory ─────────────────────────────────────────────────
    # Rolling {"role": "user"/"assistant", "content": str} history for this
    # thread. thread_id must be stable across turns (see api.py) for this to
    # actually accumulate — a fresh thread_id per request starts empty every
    # time. Uses an additive reducer so each node's partial return appends
    # rather than replacing the whole list.
    messages: Annotated[list[dict[str, str]], _append_and_trim]

    # Most recent metric that WAS successfully matched in this thread, kept
    # around as a deterministic anchor for elliptical follow-ups ("and for
    # last month?"). Unlike matched_metric (reset to None at the start of
    # every turn so each question gets reclassified from scratch), this key
    # is only ever written by classify_intent when it finds a match, and is
    # simply omitted from every turn's initial_state — so it survives turns
    # where classification fails, rather than being cleared alongside it.
    last_matched_metric: Optional[dict]

    # ── Planner / retrieval pipeline ─────────────────────────────────────────
    # Output of plan_intent — the LLM's structured interpretation of the
    # question (subject, candidate_terms, time/filter hints, confidence)
    # BEFORE any grounding against real schema happens. Reset every turn,
    # like matched_metric — not sticky.
    intent_plan: Optional[dict]

    # Output of retrieve_candidates — ranked list of candidate
    # measures/dimensions/metrics/glossary entries from agents/retrieval.py.
    # Empty list [] (not None) signals "nothing relevant found", which is
    # what routes the graph to fallback_notifier.
    retrieval_candidates: Optional[list]

    # Output of validate_query (agents/schema_validation.py) — which
    # user-requested filters got dropped and why, so explain_result can
    # surface a caveat instead of silently discarding a constraint. Also
    # used by generate_cube_query to record caveats of its own (e.g. "this
    # metric has no date field") before validate_query runs.
    validation_report: Optional[dict]

    # ── Derived metrics (Phase 4) ─────────────────────────────────────────────
    # Output of compose_derived_metric when no single catalog/measure hit
    # answers the question directly, but 2+ retrieved measures on the SAME
    # cube can be combined (see agents/derived_metrics.py + agents/expr_eval.py
    # for why this must be same-cube/one-query, never a client-side merge of
    # separate queries). None when the resolved metric is a plain catalog
    # entry. Distinct from matched_metric: matched_metric holds a
    # (possibly synthesized) catalog-shaped entry whenever there IS a result
    # to run/chart; derived_metric additionally carries the expression,
    # variables, and computed field name needed to explain and chart it.
    derived_metric: Optional[dict]

    # Set by compose_derived_metric when a derived metric would need a cube
    # join that doesn't exist yet — consumed by agents/schema_writer.py in
    # the same node call (not a separate graph node, since the join must
    # exist before generate_cube_query can reference the joined measure).
    # Left populated after the fact only for logging/inspection.
    pending_join_writes: Optional[list]