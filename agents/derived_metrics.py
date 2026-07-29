"""
Derived Metric Generator (Phase 4) — compose_derived_metric node.

Runs when retrieve_candidates didn't find a single confident, directly-
resolvable hit, but the results look combinable into a derived
(ratio/difference) metric. Three paths, tried in order of how much they
can be trusted:

  1. A curated glossary "formula" entry (catalog/glossary.yaml) — hand-
     verified same-cube variables, used as-is. No LLM call, no ordering
     ambiguity (a discount-rate formula is already written as
     discount / total, not the other way round).

  2. 2+ measure-shaped candidates on the SAME cube — a small LLM call
     decides which two and how to combine them, since getting the order
     wrong (a/b vs b/a) silently produces a different, wrong number and a
     bare heuristic has no way to know which the question actually wants.
     The returned expression is validated by agents/expr_eval.py's
     whitelist before ever being trusted, regardless of what the LLM says.

  3. 2+ measure-shaped candidates on two DIFFERENT cubes — routed through
     agents/schema_writer.py's auto-join pipeline (Phase 5). Never combined
     client-side across separate queries (see module docstrings on
     expr_eval.py / schema_writer.py for why). Nothing is computed THIS
     turn even when a join is written — the write may not have taken
     effect yet — so this always exits without a derived_metric, leaving a
     pending_join_writes record for explain_result to describe honestly.
"""

from __future__ import annotations

import json
import logging

from . import expr_eval, schema_writer
from .nodes import _openai
from .nodes_query import RELEVANT_FLOOR, find_catalog_metric_containing_measure
from .state import AgentState

logger = logging.getLogger(__name__)

# Only chase a cross-cube join when the planner explicitly flagged this as
# a combination ask — the auto-join pipeline mutates the shared Cube schema,
# so it's reserved for asks that clearly want a computed figure, not
# triggered speculatively just because two cubes happened to score well.
_CROSS_CUBE_METRIC_TYPES = {"ratio", "difference", "sum_of", "rate_over_time"}

DERIVE_SYSTEM = """\
You compose a derived metric from two candidate measures already found in
the same Cube.dev cube. Given the user's question and the two measures
below, decide the correct arithmetic combination.

Question: {question}

Candidate measures (in this exact cube):
  a = {field_a} ({label_a}): {desc_a}
  b = {field_b} ({label_b}): {desc_b}

Respond with valid JSON only:
{{
  "computable": <true or false>,
  "expression": "<arithmetic expression using ONLY the names a and b, and
    the operators + - * /, e.g. \\"a / b\\" or \\"a - b\\">",
  "computed_field_name": "<short snake_case name for the computed field>",
  "name": "<short human-readable name for this derived metric>",
  "explanation": "<one sentence: what this computes, e.g. 'discount as a
    share of dispensing value, computed as discount / dispensing_total'>"
}}

Rules:
- Only use "a" and "b" as variable names in "expression" — never the raw
  field names shown above.
- If the question doesn't actually call for combining these two measures
  (e.g. they're unrelated to what was asked), set "computable" to false.
- Order matters: "a / b" and "b / a" are different numbers — pick the
  order that matches what the question is actually asking for (e.g.
  "discount rate" means discount / total, not total / discount).
"""


def _candidate_cube(candidate: dict) -> str | None:
    if candidate.get("cube"):
        return candidate["cube"]
    field = candidate.get("field")
    if field and "." in field:
        return field.split(".", 1)[0]
    return None


def _measure_shaped(candidate: dict) -> bool:
    if candidate["source"] == "measure" and candidate.get("field"):
        return True
    if candidate["source"] == "glossary" and candidate.get("field") and not candidate.get("formula"):
        return True
    return False


def _base_query_for(field: str) -> dict:
    """
    Reuse a curated catalog metric's own dimensions/timeDimensions when one
    already covers this field (same reasoning as
    nodes_query.find_catalog_metric_containing_measure) — otherwise fall
    back to a bare, dimension-less query.
    """
    base = find_catalog_metric_containing_measure(field)
    if base:
        bq = base.get("cube_query") or {}
        return {
            "dimensions": list(bq.get("dimensions") or []),
            "timeDimensions": list(bq.get("timeDimensions") or []),
        }
    return {"dimensions": [], "timeDimensions": []}


def compose_derived_metric(state: AgentState) -> dict:
    candidates = state.get("retrieval_candidates") or []
    relevant = [c for c in candidates if c.get("score", 0) >= RELEVANT_FLOOR]

    formula_hits = [c for c in relevant if c["source"] == "glossary" and c.get("formula") and c.get("variables")]
    if formula_hits:
        derived = _build_from_glossary_formula(formula_hits[0])
        if derived:
            return _finalize(derived)

    measure_candidates = [c for c in relevant if _measure_shaped(c)]
    by_cube: dict[str, list[dict]] = {}
    for c in measure_candidates:
        cube = _candidate_cube(c)
        if cube:
            by_cube.setdefault(cube, []).append(c)

    same_cube_group = next(
        (group for group in by_cube.values() if len({c["field"] for c in group}) >= 2), None
    )
    if same_cube_group:
        derived = _compose_same_cube(state, same_cube_group)
        if derived:
            return _finalize(derived)

    if len(by_cube) >= 2:
        cross_cube_result = _attempt_cross_cube(state, by_cube)
        if cross_cube_result:
            return cross_cube_result  # pending_join_writes only — nothing computed this turn

    logger.info("compose_derived_metric: nothing composable from %d relevant candidates", len(relevant))
    return {}


def _finalize(derived: dict) -> dict:
    matched_metric = {"id": derived["id"], "name": derived["name"], "cube_query": derived["cube_query"]}
    return {
        "derived_metric": derived,
        "matched_metric": matched_metric,
        "last_matched_metric": matched_metric,
    }


def _build_from_glossary_formula(candidate: dict) -> dict | None:
    variables: dict = candidate.get("variables") or {}
    formula = candidate.get("formula")
    if not variables or not formula:
        return None

    fields = list(variables.values())
    cubes = {f.split(".", 1)[0] for f in fields if "." in f}
    if len(cubes) != 1:
        logger.warning(
            "compose_derived_metric: glossary formula '%s' spans cubes %s — refusing "
            "(catalog/glossary.yaml formulas are meant to be same-cube; this looks like a curation bug)",
            candidate.get("glossary_term"), cubes,
        )
        return None

    try:
        expr_eval.evaluate(formula, {k: 1.0 for k in variables})
    except expr_eval.ExprEvalError as exc:
        logger.error(
            "compose_derived_metric: glossary formula '%s' failed the safety check — %s",
            candidate.get("glossary_term"), exc,
        )
        return None

    base_cube = cubes.pop()
    base_query = _base_query_for(fields[0])
    query = {
        "measures": sorted(set(fields)),
        "dimensions": base_query["dimensions"],
        "timeDimensions": base_query["timeDimensions"],
        "filters": [],
        "limit": 500,
    }

    field_name = f"computed_{candidate['glossary_term'].replace(' ', '_')}"
    return {
        "id": f"derived:{candidate['glossary_term']}",
        "name": candidate.get("label") or candidate["glossary_term"],
        "expression": formula,
        "variables": variables,
        "base_cube": base_cube,
        "joined_cubes": [],
        "cube_query": query,
        "computed_field_name": field_name,
        "explanation": candidate.get("description", ""),
        "source_glossary_term": candidate["glossary_term"],
        "confidence": candidate.get("score", 0.0),
    }


def _compose_same_cube(state: AgentState, group: list[dict]) -> dict | None:
    by_field: dict[str, dict] = {}
    for c in sorted(group, key=lambda c: -c["score"]):
        by_field.setdefault(c["field"], c)
    top_two = list(by_field.values())[:2]
    if len(top_two) < 2:
        return None
    cand_a, cand_b = top_two

    prompt = DERIVE_SYSTEM.format(
        question=state["question"],
        field_a=cand_a["field"], label_a=cand_a.get("label", ""), desc_a=cand_a.get("description", ""),
        field_b=cand_b["field"], label_b=cand_b.get("label", ""), desc_b=cand_b.get("description", ""),
    )
    response = _openai().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    parsed = json.loads(response.choices[0].message.content)
    if not parsed.get("computable") or not parsed.get("expression"):
        logger.info("compose_derived_metric: LLM judged %s/%s not computable for this question", cand_a["field"], cand_b["field"])
        return None

    try:
        expr_eval.evaluate(parsed["expression"], {"a": 1.0, "b": 1.0})
    except expr_eval.ExprEvalError as exc:
        logger.warning("compose_derived_metric: LLM-proposed expression rejected by safety check — %s", exc)
        return None

    base_cube = _candidate_cube(cand_a)
    base_query = _base_query_for(cand_a["field"])
    query = {
        "measures": sorted({cand_a["field"], cand_b["field"]}),
        "dimensions": base_query["dimensions"],
        "timeDimensions": base_query["timeDimensions"],
        "filters": [],
        "limit": 500,
    }

    field_name = parsed.get("computed_field_name") or "computed_value"
    return {
        "id": f"derived:{field_name}",
        "name": parsed.get("name") or field_name,
        "expression": parsed["expression"],
        "variables": {"a": cand_a["field"], "b": cand_b["field"]},
        "base_cube": base_cube,
        "joined_cubes": [],
        "cube_query": query,
        "computed_field_name": field_name,
        "explanation": parsed.get("explanation", ""),
        "source_glossary_term": None,
        "confidence": min(cand_a["score"], cand_b["score"]),
    }


def _attempt_cross_cube(state: AgentState, by_cube: dict[str, list[dict]]) -> dict | None:
    plan = state.get("intent_plan") or {}
    if plan.get("metric_type") not in _CROSS_CUBE_METRIC_TYPES:
        return None

    ranked_cubes = sorted(by_cube.items(), key=lambda kv: -max(c["score"] for c in kv[1]))
    if len(ranked_cubes) < 2:
        return None
    (base_cube, base_group), (target_cube, target_group) = ranked_cubes[:2]

    base_candidate = max(base_group, key=lambda c: c["score"])
    target_candidate = max(target_group, key=lambda c: c["score"])
    agg_types = {
        base_candidate["field"]: base_candidate.get("cube_measure_type", ""),
        target_candidate["field"]: target_candidate.get("cube_measure_type", ""),
    }

    result = schema_writer.attempt_auto_join(
        base_cube=base_cube,
        target_cube=target_cube,
        agg_types=agg_types,
        question=state["question"],
        confidence=min(base_candidate["score"], target_candidate["score"]),
    )
    logger.info(
        "compose_derived_metric: cross-cube join attempt %s <-> %s -> written=%s",
        base_cube, target_cube, result.get("written"),
    )
    return {"pending_join_writes": [result]}
