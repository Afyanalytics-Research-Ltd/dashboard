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

from django.utils import timezone

from . import retrieval
from .nodes import _openai, _recent_history, _context_hint
from .state import AgentState

logger = logging.getLogger(__name__)

RETRIEVE_TOP_K = 30

PLAN_SYSTEM = """\
You are an intent planner for a hospital analytics platform. Read the user's
question (and any prior conversation turns) and produce a structured plan
describing what they're asking for. Do NOT try to pick an exact predefined
metric yourself — a separate retrieval step searches for candidates by
meaning, and a separate query builder constructs the actual Cube query.

Today's date is {today}. Resolve any relative date phrase that names a
SPECIFIC calendar period ("this month", "today", "this quarter", "year to
date", "in April") against THIS date, not any date mentioned in example data
or prior turns — a question with no explicit year almost always means the
current one.

"Most recent" / "latest" / "last N months" phrases are DIFFERENT — they mean
the most recent period the DATA actually has, not a period computed from
today's date. The reporting data can lag today by weeks or months, so a
question like "the most recent month" or "the last 6 months" must NOT be
turned into a calendar range anchored on today (e.g. today minus one month) —
doing so silently matches zero rows whenever the data is older than that
guess. Use time_range.operator "latest" for these instead (see below); it
defers to whatever period is actually freshest in the data.
{context_hint}
Respond with valid JSON only — no markdown, no explanation:
{{
  "subject": "<short phrase naming the business concept being asked about>",
  "metric_type": "<one of: direct, ratio, difference, sum_of, rate_over_time, unknown>",
  "candidate_terms": ["<phrase 1>", "<phrase 2>", "..."],
  "time_range": {{"operator": "<inDateRange|beforeDate|afterDate|latest|none>", "start": "<YYYY-MM-DD or null>", "end": "<YYYY-MM-DD or null>", "periods": "<int, only for 'latest' — how many of the most-recent periods to return>"}},
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
- group_by_hints from IMPLICIT breakdowns, not just explicit "by X" phrasing:
  a yes/no question of the shape "are there <entities> where <condition>"
  (e.g. "are there WARDS where insured admissions exceed cash admissions",
  "are there INSURERS with over 1M in overpayments", "are there MONTHS
  where the rate exceeded 40%") is always implicitly asking to check the
  condition PER <entity> — add that entity as a group_by_hint even though
  the question never says "broken down by" or "by ward". Skipping this
  means the query aggregates everything into one total, and a per-entity
  comparison becomes impossible to answer at all.
- time_range/filter_hints are free text at this stage — exact Cube field
  names are resolved later by validation, not here.
- time_range.operator "latest": use for "the most recent month/period",
  "currently", "latest", or "the last/past N months/weeks" trend questions —
  anything asking for the freshest available data rather than a specific
  named calendar period. Leave start/end null and set "periods" to how many
  of the most-recent periods to return: 1 for a single "most recent X"
  value, or N for "the last N months" (default 6 if the question says
  "recently"/"lately" with no number, since that is this platform's
  standard trend window). Do not also compute start/end dates for this
  operator — periods alone drives it.
- If the question is a follow-up with no new subject of its own (e.g. "and
  for last month?", "what about the past two months?"), resolve "subject"
  and "candidate_terms" from the conversation so far — see the note above
  about the most recently matched metric, if any.
- confidence reflects whether you understood the QUESTION itself, not
  whether a matching metric exists — a well-understood but uncatalogued
  question should still get a reasonably high confidence.
- When the subject names a specific KIND or CATEGORY of a broader
  countable thing — "CT/Angio sessions", "malaria cases", "insured
  admissions", "chronic drugs", "antibiotic prescriptions" — ALSO extract
  the qualifying word as a filter_hint, even
  though the phrase reads as one natural noun phrase with no separate
  category word like "type" spelled out (unlike "Private WARDS", where
  "ward" itself signals the category). concept: the general category this
  qualifies, worded to lexically overlap the REAL field name it will be
  matched against, not just a generic label for the subject matter — for a
  drug/medication category (e.g. "antimalarials", "ARVs", "antibiotics",
  "opioids", "chronic drugs"), use concept "therapeutic class" or "drug
  class", NOT the generic word "medication" or "drug" (real fields are
  named therapeutic_class/therapeutic_subclass — "medication" shares no
  word with either, so a concept of "medication" fails to match a field
  that's an exact conceptual fit). Other examples: "modality", "diagnosis",
  "payment mode" — value: the specific qualifier (e.g. "CT / Angio",
  "malaria", "insured", "Antimalarials"). Getting this right matters:
  skipping it, or picking a concept word that won't lexically match any
  real field, means the query runs against EVERY kind of the broader thing
  instead of just the one asked about, and the answer looks specific when
  it silently isn't.
- filter_hints operator: prefer "contains" over "equals" whenever the value
  is a short qualifying word rather than a known exact code — real field
  values are often longer labels that merely include that word (e.g. "Private
  wards" → the real ward names/categories are likely "Private Male", "Private
  Female", "Private / Amenity", not the bare string "Private"). "equals" is
  only safe for values you're confident are the field's exact, complete value
  (e.g. a boolean flag, or a value copied verbatim from prior conversation
  turns/example data). Getting this wrong means a real, matching row gets
  silently filtered out because "Private" != "Private Male".
- filter_hints for a NUMERIC THRESHOLD comparison — "rate ABOVE 10%",
  "conversion rate BELOW 5%", "OVER 100 evaluations", "MORE THAN KES 1M",
  "EXCEEDS", "dropped BELOW 50%" — use operator "gt"/"gte"/"lt"/"lte" (never
  "equals"/"contains" for these) and set value to the bare number with no
  unit or symbol (e.g. "10" not "10%" or "above 10%"). concept should name
  the RATE/COUNT field itself (e.g. "readmission rate", "evaluations",
  "overpayment amount"), not the entity being filtered (that's a separate
  group_by_hint — see the "are there <entities> where <condition>" rule
  above; a threshold question almost always needs one of these too, e.g.
  "are there DISCHARGE PATHWAYS with a rate above 10%" needs group_by_hint
  "discharge pathway" AND filter_hint {{concept: "readmission rate", value:
  "10", operator: "gt"}}). A question with TWO thresholds on the SAME
  measure family (e.g. "over 100 evaluations BUT a conversion rate below
  5%") needs two separate filter_hints entries, one per threshold.
"""


def plan_intent(state: AgentState) -> dict:
    prompt = PLAN_SYSTEM.format(
        today=timezone.now().strftime("%Y-%m-%d"),
        context_hint=_context_hint(state),
    )

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
        "plan_intent: subject=%r metric_type=%s confidence=%.2f time_range=%s group_by_hints=%s filter_hints=%s",
        plan.get("subject"), plan.get("metric_type"), float(plan.get("confidence", 0.0)),
        plan.get("time_range"), plan.get("group_by_hints"), plan.get("filter_hints"),
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
