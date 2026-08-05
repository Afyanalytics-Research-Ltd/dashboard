"""
Cube Query Generator + Schema Validator + Cube API execution nodes.

  generate_cube_query — deterministic (no LLM): resolves the top retrieval
                        candidate into a matched_metric (first turn this
                        happens for a given question), then translates the
                        intent plan's free-text filter/time hints into Cube
                        filter shape. Does NOT validate field names — that's
                        validate_query's job, kept separate so it's
                        independently testable/reusable.
  validate_query      — wraps agents/schema_validation.py against the
                        resolved metric's own real fields.
  execute_query       — unchanged from the original nodes.py body: facility
                        scoping, date-filter promotion, Cube API call.
"""

from __future__ import annotations

import logging
from copy import deepcopy

import httpx

from . import expr_eval, schema_validation
from .catalog import get_all, get_by_id
from .cube_client import run_query
from .facility import inject_facility_filter, resolve_facility
from .state import AgentState

logger = logging.getLogger(__name__)

# ── Retrieval-routing thresholds ──────────────────────────────────────────────
# Calibrated against real questions run through the live embeddings index
# (text-embedding-3-small cosine similarity) during rollout: genuinely
# unrelated questions ("what's the weather tomorrow") scored ~0.19-0.20;
# legitimate-but-loosely-worded questions ("how many patients last month")
# scored ~0.45-0.58; clean glossary-covered questions scored ~0.60-0.82.
# Never observed a score near 1.0 even for excellent matches — these are
# short-phrase-vs-question similarities, not near-duplicate text.
RELEVANT_FLOOR = 0.45          # below this, a candidate doesn't count as "found something"
CONFIDENT_SINGLE_SCORE = 0.58   # at/above this + a directly-resolvable source → answer directly

# Sources that can resolve straight to a query without derived-metric
# composition: a catalog metric, a bare Cube measure, or a glossary entry
# that maps directly to one field (not a multi-measure formula — those need
# compose_derived_metric).
_DIRECTLY_RESOLVABLE_SOURCES = {"metric", "measure"}


def is_directly_resolvable(candidate: dict) -> bool:
    if candidate["source"] in _DIRECTLY_RESOLVABLE_SOURCES:
        return True
    if candidate["source"] == "glossary":
        return bool(candidate.get("field")) and not candidate.get("formula")
    return False


def find_first_resolvable(candidates: list[dict], min_score: float) -> dict | None:
    """
    Scan the RANKED candidate list for the first one that's both above
    min_score and directly resolvable — NOT just candidates[0]. A
    dimension or formula-only glossary hit can easily outrank a perfectly
    good directly-resolvable candidate a few places down (embeddings don't
    know which sources are "answerable"), so checking only the top entry
    routed real, answerable questions to the human fallback unnecessarily.
    """
    for candidate in candidates:
        if candidate.get("score", 0) >= min_score and is_directly_resolvable(candidate):
            return candidate
    return None


def find_catalog_metric_containing_measure(field: str) -> dict | None:
    """
    A "measure" or "glossary" retrieval hit names a bare Cube field — but a
    fully-curated catalog/metrics.yaml entry for that same measure may
    already exist, with real dimensions/timeDimensions attached (e.g.
    glossary "dialysis sessions" maps_to rpt_dialysis.count, and the
    catalog's own dialysis_session_count entry already declares
    rpt_dialysis's session_month time dimension). Preferring that richer
    entry over a bare single-measure synthesis is what lets date/dimension
    filters actually work for these cases, instead of silently having
    nothing to attach to.
    """
    for metric in get_all():
        if field in ((metric.get("cube_query") or {}).get("measures") or []):
            return deepcopy(metric)
    return None


def _synthesize_single_measure_metric(candidate: dict) -> dict:
    """
    Build a matched_metric-shaped dict from a bare "measure" retrieval
    candidate whose field isn't covered by any curated catalog entry (see
    find_catalog_metric_containing_measure, tried first). Known
    simplification: a retrieval candidate only carries its own field, not
    the rest of its cube's dimensions/time dimensions, so the synthesized
    metric has none — filter/time hints simply won't have anything to
    attach to for these truly uncovered-cube cases (schema_validation
    safely drops them rather than guessing), which is a safe degradation,
    not a crash.
    """
    field = candidate["field"]
    return {
        "id": f"measure:{field}",
        "name": candidate.get("label") or field,
        "cube_query": {
            "measures": [field],
            "dimensions": [],
            "timeDimensions": [],
            "filters": [],
            "limit": 500,
        },
    }


def _resolve_single_field(candidate: dict) -> dict:
    field = candidate["field"]
    return find_catalog_metric_containing_measure(field) or _synthesize_single_measure_metric(candidate)


def _resolve_matched_metric(state: AgentState) -> dict | None:
    """Resolve retrieval_candidates' best directly-resolvable pick into a
    matched_metric, unless one is already set (e.g. by the resume path's
    re_classify, or by compose_derived_metric's synthesized entry)."""
    if state.get("matched_metric"):
        return state["matched_metric"]

    candidates = state.get("retrieval_candidates") or []
    relevant = [c for c in candidates if c.get("score", 0) >= RELEVANT_FLOOR]
    top = find_first_resolvable(relevant, CONFIDENT_SINGLE_SCORE)
    if not top:
        return None

    if top["source"] == "metric" and top.get("metric_id"):
        base = get_by_id(top["metric_id"])
        return deepcopy(base) if base else None
    if top["source"] in ("measure", "glossary"):
        # A glossary entry that maps directly to one measure resolves the
        # same way a bare measure hit does — is_directly_resolvable already
        # excluded formula-shaped glossary entries (those need
        # compose_derived_metric).
        return _resolve_single_field(top)
    return None


def generate_cube_query(state: AgentState) -> dict:
    matched_metric = _resolve_matched_metric(state)
    if not matched_metric:
        logger.warning("generate_cube_query: no usable metric resolved from retrieval_candidates")
        return {"matched_metric": None, "cube_query": None}

    update: dict = {}
    if not state.get("matched_metric"):
        update["matched_metric"] = matched_metric
        update["last_matched_metric"] = matched_metric

    plan = state.get("intent_plan") or {}
    query = dict(matched_metric["cube_query"])
    filters = list(query.get("filters") or [])
    warnings: list[str] = []

    # Only break the result out by a dimension the user actually asked to
    # see (plan.group_by_hints) — NOT every dimension the catalog metric
    # happens to declare. Blindly copying the catalog's full dimension list
    # into every query (the original behavior) means a metric whose
    # dimensions include a near-unique column (e.g. a patient/visit ID) — or
    # even just an unrelated demographic breakdown — returns hundreds of
    # barely-aggregated rows, or an unwanted breakdown, for a plain "how
    # many X" total question. Observed both ways: fact_dispensing_metrics'
    # dimensions include sk_patient_id/sk_visit_id and blew a plain count
    # question up to the full 500-row limit; inpatient_admissions_metrics'
    # sex/source_schema dimensions turned "how many patients, past six
    # months" into an unrequested sex breakdown.
    all_dimensions = list(query.get("dimensions") or [])
    group_by_hints = [h.strip().lower() for h in (plan.get("group_by_hints") or []) if h and h.strip()]
    selected_dimensions = []
    if group_by_hints:
        for dimension in all_dimensions:
            field_name = dimension.rsplit(".", 1)[-1].lower()
            if any(hint in field_name or field_name in hint for hint in group_by_hints):
                selected_dimensions.append(dimension)
        if not selected_dimensions and len(all_dimensions) == 1:
            # The user clearly asked for SOME breakdown (group_by_hints is
            # non-empty) but no hint word lexically overlaps the metric's
            # one dimension at all (e.g. hint "doctor" vs the real column
            # "username" — no shared substring either way). With only one
            # possible axis to break out by, using it is a safe default:
            # the ambiguity that makes guessing risky with 2+ candidates
            # doesn't exist when there's exactly one.
            selected_dimensions = list(all_dimensions)
    query["dimensions"] = selected_dimensions

    # Same reasoning applies to the catalog's timeDimension: it always
    # carries a granularity (e.g. "month"), so including it unconditionally
    # turns EVERY question into a per-period time series — a plain "how many
    # drugs have been dispensed" (no date range mentioned at all) returned
    # 72 monthly rows instead of one total, because the catalog's own
    # timeDimension+granularity was forwarded regardless of intent. Only
    # keep granularity when the plan actually wants a time breakdown; when a
    # date range is needed only to FILTER (not bucket), keep the dimension
    # bare (no granularity) so Cube applies it as a plain range filter.
    all_time_dims = list(query.get("timeDimensions") or [])
    wants_time_breakdown = any(
        any(kw in hint for kw in ("month", "time", "trend", "date", "week", "year", "day"))
        for hint in group_by_hints
    )
    query["timeDimensions"] = []
    if all_time_dims and wants_time_breakdown:
        query["timeDimensions"] = [dict(all_time_dims[0])]

    time_range = plan.get("time_range") or {}
    operator = time_range.get("operator")
    if operator and operator != "none":
        time_members = schema_validation.valid_time_members_for(matched_metric)
        values = [v for v in (time_range.get("start"), time_range.get("end")) if v]
        if time_members and values:
            # Anchor on the metric's own real time dimension rather than
            # trusting the plan to have named the right field — validation
            # would drop a mismatched name anyway, so just use the one that
            # actually exists.
            member = sorted(time_members)[0]
            filters.append({"member": member, "operator": operator, "values": values})
            logger.info(
                "generate_cube_query: applying time filter member=%s operator=%s values=%s",
                member, operator, values,
            )
            if not query["timeDimensions"]:
                # Bare entry (no granularity) purely so _promote_date_filters
                # (execute_query) has something to attach dateRange to —
                # without granularity Cube returns one aggregated total for
                # the range, not a per-period breakdown.
                query["timeDimensions"] = [{"dimension": member}]
        else:
            # This metric has no date field to filter on at all — the query
            # below will run unfiltered (all-time). explain_result MUST be
            # told this explicitly, or the summarization LLM has no way to
            # know the user's requested date range wasn't actually applied
            # and may fabricate a plausible-sounding but baseless number to
            # match the question's framing (observed: identical unfiltered
            # data for "last month" and "the past two months" produced two
            # different invented totals).
            warnings.append(
                f"The user asked for a specific date range, but "
                f"'{matched_metric.get('name') or matched_metric.get('id')}' has no date "
                f"field to filter on — the result below covers all available data, not "
                f"just the requested period."
            )
            logger.warning(
                "generate_cube_query: metric=%s has no date field — time_range %s dropped, running unfiltered",
                matched_metric.get("id"), time_range,
            )

    allowed_members = schema_validation.valid_members_for(matched_metric)
    for hint in plan.get("filter_hints") or []:
        concept = (hint.get("concept") or "").strip().lower()
        value = hint.get("value")
        if not concept or not value:
            continue
        # allowed_members is a set (unordered) and a broad concept like "ward"
        # can substring-match MULTIPLE real fields (ward_name AND
        # ward_category both contain "ward") — picking whichever the set
        # happens to yield first is non-deterministic and can silently
        # filter on the wrong field (observed: "General Female" — a ward
        # NAME value — applied to ward_category instead of ward_name,
        # matching zero rows). Collect every match, then break the tie
        # deterministically: prefer a field ending in "_name" (this
        # codebase's convention for "the specific one of these", e.g.
        # ward_name/facility_name), else fall back to sorted order so the
        # choice is at least stable across runs.
        candidates = [
            member for member in allowed_members
            if (field_name := member.rsplit(".", 1)[-1].lower())
            and (concept in field_name or field_name in concept)
        ]
        if candidates:
            candidates.sort(key=lambda m: (not m.rsplit(".", 1)[-1].lower().endswith("_name"), m))
            filters.append({
                "member": candidates[0],
                "operator": hint.get("operator", "equals"),
                "values": [value],
            })
        else:
            # No field on this metric matches the concept at all (e.g. the
            # user said "Private wards" but the resolved metric has no ward
            # dimension whatsoever) — this filter is silently unattachable,
            # not just ambiguous. Unlike the date-range drop above, this
            # path previously had NO warning at all, so explain_result had
            # no way to know "Private" was never actually applied — the
            # summarizer would then confidently claim a filtered result that
            # was really an unfiltered total (observed: "the Private wards
            # had 81 admissions" for a metric with no concept of ward).
            warnings.append(
                f"The user asked to filter by '{value}' (concept: {concept}), but "
                f"'{matched_metric.get('name') or matched_metric.get('id')}' has no matching "
                f"field for this — the result below is NOT filtered by {concept}."
            )
            logger.warning(
                "generate_cube_query: metric=%s has no field matching filter concept=%r value=%r — dropped",
                matched_metric.get("id"), concept, value,
            )

    query["filters"] = filters
    update["cube_query"] = query
    if warnings:
        update["validation_report"] = {"dropped_filters": [], "warnings": warnings}
    return update


def validate_query(state: AgentState) -> dict:
    query = state.get("cube_query")
    metric = state.get("matched_metric")

    # Preserve any warnings generate_cube_query already recorded (e.g. "this
    # metric has no date field") — this node's own return would otherwise
    # silently replace them, since validation_report has no merge reducer.
    warnings = list((state.get("validation_report") or {}).get("warnings") or [])

    if not query or not metric:
        warnings.append("no query to validate")
        return {
            "cube_query": None,
            "validation_report": {"dropped_filters": [], "warnings": warnings},
        }

    allowed_members = schema_validation.valid_members_for(metric)
    time_members = schema_validation.valid_time_members_for(metric)
    result = schema_validation.validate_query(query, allowed_members=allowed_members, time_members=time_members)

    validated_query = result["query"]
    if not validated_query.get("measures"):
        warnings.append("no valid measures remained after validation")
        logger.warning("validate_query: query has no valid measures left for metric %s", metric.get("id"))

    return {
        "cube_query": validated_query,
        "validation_report": {"dropped_filters": result["dropped_filters"], "warnings": warnings},
    }


def execute_query(state: AgentState) -> dict:
    metric = state["matched_metric"]
    query = dict(state["cube_query"])  # shallow copy — don't mutate state

    user_facility = state.get("user_facility") or resolve_facility(state["user_id"])
    if user_facility:
        query = inject_facility_filter(query, user_facility)  # ← filter added here

    filters = query.get("filters", [])
    if filters:
        query, filters = schema_validation.promote_date_filters(query, filters)
        query["filters"] = filters

    logger.info("execute_query: metric=%s query=%s", metric["id"], query)

    try:
        result = run_query(query)
    except httpx.HTTPStatusError as exc:
        logger.error("Cube API error: %s — %s", exc.response.status_code, exc.response.text)
        raise
    except httpx.TimeoutException:
        logger.error("Cube API timed out for metric %s", metric["id"])
        raise

    derived = state.get("derived_metric")
    if derived:
        _inject_computed_field(result, derived)

    return {
        "cube_query": query,
        "raw_result": result,
    }


def _inject_computed_field(result: dict, derived: dict) -> None:
    """
    Mutates each row of result["data"] in place, adding the derived
    metric's computed field — division happens client-side per row, but
    ONLY across measures that came back in the SAME query/row (Cube already
    did the grouping), which is the one case where this is mathematically
    correct without Cube's own SQL-level calculated measures. A row missing
    a base value, or dividing by zero, gets None for the computed field
    rather than failing the whole query.
    """
    variables: dict[str, str] = derived.get("variables") or {}
    expression: str = derived.get("expression", "")
    field_name = derived.get("computed_field_name", "computed")
    rows = result.get("data") or []

    skipped = 0
    for row in rows:
        bound = {}
        for var, measure_key in variables.items():
            value = row.get(measure_key)
            if value is None:
                continue
            try:
                bound[var] = float(value)
            except (TypeError, ValueError):
                continue
        if len(bound) < len(variables):
            row[field_name] = None
            skipped += 1
            continue
        try:
            row[field_name] = expr_eval.evaluate(expression, bound)
        except expr_eval.ExprEvalError:
            row[field_name] = None
            skipped += 1

    if skipped:
        logger.info("execute_query: %d/%d rows missing data for derived field '%s'", skipped, len(rows), field_name)
