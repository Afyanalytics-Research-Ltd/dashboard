"""
Result Explainer node — replaces the old format_result.

Same core summarization behavior as before, PLUS a templated recovery-
message branch for when generate_cube_query/validate_query couldn't
produce a usable query at all (no result to summarize, so the message is
built directly from retrieval_candidates rather than asking an LLM to
describe data that doesn't exist).
"""

from __future__ import annotations

import json
import logging

from . import charts
from .nodes import _openai, _recent_history
from .state import AgentState

logger = logging.getLogger(__name__)

# Candidates below this score aren't worth mentioning in a recovery message —
# mirrors the RELEVANT_FLOOR used by the graph's routing function, but kept
# as its own constant since "worth summarizing" and "worth routing on" are
# related but separate judgment calls.
RECOVERY_MENTION_FLOOR = 0.4

# Defense-in-depth cap on how many rows get serialized into the summarization
# prompt. generate_cube_query already tries to avoid over-broad group-bys,
# but a metric can still legitimately return hundreds of rows (e.g. grouped
# by month over years of data) — asking an LLM to read AND reproduce
# hundreds of rows verbatim in one JSON response is slow, costly, and risks
# an oversized/failed completion for no benefit, since the summary only
# ever needs a few sentences plus a representative sample.
MAX_ROWS_IN_PROMPT = 50


def _capped_raw_result(raw: dict) -> dict:
    rows = raw.get("data") or []
    if len(rows) <= MAX_ROWS_IN_PROMPT:
        return raw
    capped = dict(raw)
    capped["data"] = rows[:MAX_ROWS_IN_PROMPT]
    capped["_truncated_note"] = (
        f"Showing the first {MAX_ROWS_IN_PROMPT} of {len(rows)} rows — "
        f"mention in your summary that this is a partial view if it matters for the answer."
    )
    return capped

FORMAT_SYSTEM = """\
You are a data analyst assistant. Given a user question and a raw Cube.dev API
response, produce a concise, helpful answer.

CRITICAL — never fabricate: state only what the raw result actually shows.
Do not invent, estimate, or adjust a number to match the phrasing or implied
time range of the question. If the raw result doesn't actually reflect a date
range, breakdown, or filter the question asked for — including when a caveat
below says so — say that plainly (e.g. "This is our all-time total; a
date-filtered version isn't available for this metric") rather than
presenting an unfiltered figure as if it were period-specific.

CRITICAL — never compute a derived value yourself: if the question asks for
a rate, ratio, percentage, or difference, and the raw result contains only
the separate component numbers (not that computed value as its own field),
report the component numbers plainly and say the computed figure itself
isn't directly available — do NOT divide/subtract them yourself, even
simple arithmetic. Only state a rate/ratio/percentage/difference number
when it is already present as its own field in the raw result (this means
it was computed by a verified calculation step, not guessed by you).
{grounding_notes}{derivation_note}
Respond with valid JSON only:
{{
  "summary": "<1–3 sentence plain-English answer>",
  "data": <the relevant rows/values from the raw result, cleaned up>,
  "metric_name": "<human-readable metric name>"
}}
"""


def _grounding_notes(state: AgentState) -> str:
    """
    Caveats generate_cube_query/validate_query already know about (e.g. "this
    metric has no date field, so the requested range wasn't applied") — fed
    to the LLM up front so it can phrase the summary honestly, rather than
    appending a correction after the fact to a summary that already made a
    false claim.
    """
    warnings = (state.get("validation_report") or {}).get("warnings") or []
    if not warnings:
        return ""
    bullets = "\n".join(f"- {w}" for w in warnings)
    return f"\nCaveats about this result — your summary MUST reflect these honestly:\n{bullets}\n"


def _derivation_note(state: AgentState) -> str:
    """
    When the resolved metric was composed by compose_derived_metric (not a
    predefined catalog entry), tell the LLM how it was computed so the
    summary can say so briefly — e.g. "computed as total discount ÷ total
    dispensing value" — instead of presenting a derived figure as if it
    were an existing, named metric.
    """
    derived = state.get("derived_metric")
    if not derived:
        return ""
    return (
        f"\nThis metric was NOT a predefined metric. It was computed as:\n"
        f"  {derived.get('name')} = {derived.get('expression')}\n"
        f"  ({derived.get('explanation', '')})\n"
        f"Briefly mention in the summary that this is a computed figure and how (one clause is enough).\n"
    )


def explain_result(state: AgentState) -> dict:
    raw = state.get("raw_result")
    metric = state.get("matched_metric")

    if raw is None or metric is None:
        return _recovery_message(state)

    question = state["question"]
    prompt = FORMAT_SYSTEM.format(grounding_notes=_grounding_notes(state), derivation_note=_derivation_note(state))
    raw_for_prompt = _capped_raw_result(raw)

    response = _openai().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            *_recent_history(state, include_current=False),
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Metric: {metric['name']}\n\n"
                    f"Raw result:\n{json.dumps(raw_for_prompt, indent=2)}"
                ),
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    formatted = json.loads(response.choices[0].message.content)
    formatted["metric_id"] = metric["id"]
    formatted["thread_id"] = state["thread_id"]
    formatted["row_count"] = len(raw.get("data") or [])
    # Offer a chart rather than building one eagerly — most results are small
    # enough that the text summary already suffices, and building unwanted
    # charts wastes compute. The actual image is only rendered later, on
    # confirmation, via charts.get_chart_for_thread(thread_id).
    formatted["chart_offer"] = charts.is_heavy(raw)

    dropped = (state.get("validation_report") or {}).get("dropped_filters") or []
    if dropped:
        note = "; ".join(f"{d['member']} ({d['reason']})" for d in dropped)
        formatted["summary"] = (
            f"{formatted.get('summary', '')}\n\n(Note: I ignored an unsupported filter — {note}.)"
        )

    return {
        "formatted_result": formatted,
        "messages": [{"role": "assistant", "content": formatted.get("summary", "")}],
    }


def _recovery_message(state: AgentState) -> dict:
    """
    Templated (not LLM-hallucinated) message for when there's no real
    result to summarize — generate_cube_query found nothing usable,
    validate_query stripped the query down to no valid measures, or
    compose_derived_metric attempted (but didn't complete) a cross-cube
    join this turn. Building this from state directly (rather than asking
    an LLM to describe a result that doesn't exist) avoids fabricating a
    plausible-sounding but baseless answer.
    """
    pending_joins = state.get("pending_join_writes") or []
    if pending_joins:
        info = pending_joins[0]
        if info.get("written"):
            summary = (
                "I found related figures in two different data sources and connected them "
                "just now for questions like this. Please ask again in a moment while the "
                "connection takes effect."
            )
        else:
            summary = (
                "I found related figures in two different data sources, but combining them "
                "safely wasn't possible right now"
                + (f" ({info['reason']})" if info.get("reason") else "")
                + ", so I've flagged this to our analytics team to model properly."
            )
        formatted = {
            "summary": summary, "data": None, "metric_name": None, "metric_id": None,
            "thread_id": state.get("thread_id", ""), "row_count": 0, "chart_offer": False,
        }
        logger.info("explain_result: pending-join recovery message, written=%s", info.get("written"))
        return {
            "formatted_result": formatted,
            "messages": [{"role": "assistant", "content": summary}],
        }

    candidates = [
        c for c in (state.get("retrieval_candidates") or [])
        if c.get("score", 0) >= RECOVERY_MENTION_FLOOR
    ]

    if candidates:
        labels = ", ".join(c.get("label") or c.get("field") or c.get("id") for c in candidates[:3])
        summary = (
            f"I found a few things that might be related — {labels} — but I'm not confident "
            f"enough to compute an answer automatically. Could you rephrase, or tell me which "
            f"one you meant?"
        )
    else:
        summary = (
            "I couldn't find anything in our data that matches this question. "
            "Our analytics team may need to add a new metric for this."
        )

    formatted = {
        "summary": summary,
        "data": None,
        "metric_name": None,
        "metric_id": None,
        "thread_id": state.get("thread_id", ""),
        "row_count": 0,
        "chart_offer": False,
    }

    logger.info("explain_result: recovery message, %d candidates mentioned", len(candidates[:3]))

    return {
        "formatted_result": formatted,
        "messages": [{"role": "assistant", "content": summary}],
    }
