"""
Intent & Metric Planner + Semantic Retriever nodes.

  plan_intent          — LLM: natural-language question -> structured intent_plan.
                          Does NOT pick an exact metric id (that's retrieval's
                          job) — only extracts what the user is asking for.
  retrieve_candidates  — embeddings-based search (agents/retrieval.py) over
                          measures/dimensions/metrics/glossary entries, using
                          the plan's candidate_terms plus the raw question.
"""

from __future__ import annotations

import json
import logging

from . import retrieval
from .nodes import _openai, _recent_history, _context_hint
from .state import AgentState

logger = logging.getLogger(__name__)

RETRIEVE_TOP_K = 8

PLAN_SYSTEM = """\
You are an intent planner for a hospital analytics platform. Read the user's
question (and any prior conversation turns) and produce a structured plan
describing what they're asking for. Do NOT try to pick an exact predefined
metric yourself — a separate retrieval step searches for candidates by
meaning, and a separate query builder constructs the actual Cube query.
{context_hint}
Respond with valid JSON only — no markdown, no explanation:
{{
  "subject": "<short phrase naming the business concept being asked about>",
  "metric_type": "<one of: direct, ratio, difference, sum_of, rate_over_time, unknown>",
  "candidate_terms": ["<phrase 1>", "<phrase 2>", "..."],
  "time_range": {{"operator": "<inDateRange|beforeDate|afterDate|none>", "start": "<YYYY-MM-DD or null>", "end": "<YYYY-MM-DD or null>"}},
  "group_by_hints": ["<free-text dimension concept>"],
  "filter_hints": [{{"concept": "<e.g. facility>", "value": "<e.g. Kisumu>", "operator": "<e.g. equals>"}}],
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence>"
}}

Rules:
- candidate_terms: 1-4 short phrases capturing the business concept(s) in the
  question, used to search measures/dimensions/glossary by meaning. For a
  compound ask like "discount rate", include both "discount" and "discount
  rate" (and, if it's a ratio, the implied denominator like "dispensing
  total") so retrieval can find each half separately.
- metric_type "ratio"/"difference"/"sum_of" is a HINT for a later stage, not
  a hard classification — leave it "direct" if unsure.
- time_range/filter_hints are free text at this stage — exact Cube field
  names are resolved later by validation, not here.
- If the question is a follow-up with no new subject of its own (e.g. "and
  for last month?", "what about the past two months?"), resolve "subject"
  and "candidate_terms" from the conversation so far — see the note above
  about the most recently matched metric, if any.
- confidence reflects whether you understood the QUESTION itself, not
  whether a matching metric exists — a well-understood but uncatalogued
  question should still get a reasonably high confidence.
"""


def plan_intent(state: AgentState) -> dict:
    prompt = PLAN_SYSTEM.format(context_hint=_context_hint(state))

    response = _openai().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            *_recent_history(state),
            {"role": "user", "content": state["question"]},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    plan: dict = json.loads(response.choices[0].message.content)

    logger.info(
        "plan_intent: subject=%r metric_type=%s confidence=%.2f",
        plan.get("subject"), plan.get("metric_type"), float(plan.get("confidence", 0.0)),
    )

    return {
        "intent_plan": plan,
        "messages": [{"role": "user", "content": state["question"]}],
    }


def retrieve_candidates(state: AgentState) -> dict:
    plan = state.get("intent_plan") or {}
    query_texts = [state["question"]] + list(plan.get("candidate_terms") or [])

    candidates = retrieval.retrieve_many(query_texts, top_k=RETRIEVE_TOP_K)

    logger.info(
        "retrieve_candidates: %d candidates, top=%s",
        len(candidates),
        {"id": candidates[0]["id"], "score": candidates[0]["score"]} if candidates else None,
    )

    return {"retrieval_candidates": candidates}
