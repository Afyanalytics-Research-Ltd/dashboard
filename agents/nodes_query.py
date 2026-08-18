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
from .cube_client import fetch_meta, run_query
from .facility import (
    FACILITY_DIMENSION_NAMES,
    inject_facility_filter,
    resolve_facility,
    resolve_facility_filter_value,
)
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

# Glue words that show up in a filter concept or a field name without being
# what actually relates them (e.g. "ward type" and "admission_type" both
# contain "type", but that's not evidence they're the same axis) — stripped
# before the whole-word-overlap fallback in _field_matches_concept, so two
# fields don't false-positive match on nothing but a shared suffix.
_GENERIC_CONCEPT_WORDS = {"type", "category", "mode", "status", "name", "code", "group", "kind"}


def _concept_words(concept: str) -> set[str]:
    words = {w for w in concept.lower().replace("_", " ").split() if w}
    distinctive = words - _GENERIC_CONCEPT_WORDS
    return distinctive or words  # an all-generic concept still needs something to match on


def _field_matches_concept(field_name: str, concept: str) -> bool:
    """
    True if a metric's field named field_name is a plausible target for a
    free-text filter concept (e.g. "ward type", "modality"). Two tiers:

    1. Plain substring, either direction — handles the common case where
       concept and field name are basically the same word (concept
       "modality" / field "modality", or concept "facility" / field
       "facility_name").
    2. Whole-word overlap on the concept's DISTINCTIVE words (glue words
       like "type"/"category" stripped first — see _GENERIC_CONCEPT_WORDS)
       against the field name's own underscore-separated words. Lets an LLM
       -extracted concept like "ward type" still match a field actually
       named "ward_category" or "ward_name" (no shared substring at all
       otherwise), without matching an unrelated field that merely happens
       to share the same glue word (e.g. "admission_type").
    """
    concept = concept.strip().lower()
    if not concept or not field_name:
        return False
    if concept in field_name or field_name in concept:
        return True
    return bool(_concept_words(concept) & set(field_name.split("_")))


# This codebase's own naming convention for a record's reporting PERIOD —
# nearly every cube's intended date-range-filter target is named this way
# (dispensing_month, admission_month, revenue_month, invoice_month, ...) —
# checked ahead of "_date" and any bare event timestamp when a metric
# declares more than one time dimension, so a plain "in April 2026" question
# lands on the period field rather than a per-record event timestamp (e.g.
# fact_patient_dispensing has dispensing_month AND first_dispensed_at/
# last_dispensed_at — the latter two are per-patient lifetime markers, not
# what "in April 2026" means for an aggregate question).
_TIME_MEMBER_SUFFIX_PRIORITY = ("_month", "_date", "_at")


def _pick_time_member(time_members: set[str]) -> str:
    """
    Choose which real time dimension a date-range filter should target when
    a metric declares more than one. Tries each suffix tier in
    _TIME_MEMBER_SUFFIX_PRIORITY in turn — the first tier with any match
    wins, sorted alphabetically within that tier for a stable choice among
    ties (e.g. two "_month" fields). Falls back to plain alphabetical order
    only if nothing matches any tier at all.
    """
    for suffix in _TIME_MEMBER_SUFFIX_PRIORITY:
        matches = sorted(m for m in time_members if m.rsplit(".", 1)[-1].lower().endswith(suffix))
        if matches:
            return matches[0]
    return sorted(time_members)[0]


def _find_measure_for_qualifier(metric: dict, value: str) -> str | None:
    """
    Some "kind of X" filter values name a dedicated MEASURE on the same
    cube rather than something to filter a dimension by — e.g. "chronic"
    (from "chronic drugs") isn't a WHERE-able category, it's
    fact_patient_dispensing.has_chronic_drug, a pre-aggregated 0/1 flag
    literally titled "chronic". Same shape as the "CT/Angio sessions" bug
    (the qualifier IS the measure, not a filter on top of a generic one) —
    but worse when missed, since Cube can't filter a SUM by a string at
    all: it emits a HAVING ... ILIKE clause on the aggregated number,
    which is never true and silently returns the wrong (usually zero or
    nonsensical) rows rather than erroring.

    Checked BEFORE attempting to attach a dimension filter for a hint — a
    hit here means don't filter at all, select this measure instead.
    """
    value_lower = str(value).strip().lower()
    if not value_lower:
        return None
    for measure in (metric.get("cube_query") or {}).get("measures") or []:
        field_name = measure.rsplit(".", 1)[-1].lower()
        if _field_matches_concept(field_name, value_lower):
            return measure
    return None


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


_cube_meta_cache: dict[str, dict] | None = None


def _cube_meta_for(cube_name: str) -> dict | None:
    """Live Cube /meta entry for one cube, fetched once per process.

    Used to backfill a synthesized single-measure metric's dimensions —
    see _synthesize_single_measure_metric for why this matters.
    """
    global _cube_meta_cache
    if _cube_meta_cache is None:
        try:
            meta = fetch_meta()
        except Exception:
            logger.exception("_cube_meta_for: fetch_meta() failed — proceeding with no cube metadata")
            _cube_meta_cache = {}
        else:
            _cube_meta_cache = {c["name"]: c for c in meta.get("cubes", [])}
    return _cube_meta_cache.get(cube_name)


def _synthesize_single_measure_metric(candidate: dict) -> dict:
    """
    Build a matched_metric-shaped dict from a bare "measure" retrieval
    candidate whose field isn't covered by any curated catalog entry (see
    find_catalog_metric_containing_measure, tried first).

    Pulls the REST of the measure's own cube's dimensions/timeDimensions
    from Cube's live /meta — a retrieval candidate only carries its own
    field, so without this the synthesized metric had no filterable fields
    at all, meaning every filter_hint/time_range got silently dropped even
    when the cube genuinely has a matching facility or date column
    (observed: rpt_admission_tat.p50_tat_min resolved this way, correctly
    matching the question at 0.67, but a "most recent month" time_range was
    dropped as "no date field" despite the cube's own tat_month column
    existing — the metric just never carried it). Falls back to no
    dimensions at all if the live cube metadata can't be fetched, which is
    the previous (safe, non-crashing) behavior.
    """
    field = candidate["field"]
    cube_name = candidate.get("cube") or (field.split(".", 1)[0] if "." in field else None)

    dimensions: list[str] = []
    time_dimensions: list[dict] = []
    cube_meta = _cube_meta_for(cube_name) if cube_name else None
    if cube_meta:
        for dim in cube_meta.get("dimensions") or []:
            name = dim.get("name")
            if not name:
                continue
            if dim.get("type") == "time":
                time_dimensions.append({"dimension": name})
            else:
                dimensions.append(name)

    return {
        "id": f"measure:{field}",
        "name": candidate.get("label") or field,
        "cube_query": {
            "measures": [field],
            "dimensions": dimensions,
            "timeDimensions": time_dimensions,
            "filters": [],
            "limit": 500,
        },
    }


def _resolve_single_field(candidate: dict) -> dict:
    field = candidate["field"]
    return find_catalog_metric_containing_measure(field) or _synthesize_single_measure_metric(candidate)


def _resolve_candidate_metric(candidate: dict) -> dict | None:
    """Turn one directly-resolvable retrieval candidate into a full
    matched_metric dict. Factored out of _resolve_matched_metric so
    _find_metric_supporting_concepts can resolve several candidates to
    check their fields, not just whichever one ends up chosen."""
    if candidate["source"] == "metric" and candidate.get("metric_id"):
        base = get_by_id(candidate["metric_id"])
        return deepcopy(base) if base else None
    if candidate["source"] in ("measure", "glossary"):
        # A glossary entry that maps directly to one measure resolves the
        # same way a bare measure hit does — is_directly_resolvable already
        # excluded formula-shaped glossary entries (those need
        # compose_derived_metric).
        return _resolve_single_field(candidate)
    return None


def _find_metric_supporting_concepts(
    candidates: list[dict], concepts: list[str], candidate_terms: list[str],
) -> dict | None:
    """
    When the user named filter concept(s) (e.g. "ward" for "Private wards"),
    a directly-resolvable candidate with no field for it is a worse pick
    than a slightly-lower-ranked one that does — top score alone can't
    distinguish "answerable" from "merely resolvable" here, the same gap
    find_first_resolvable already closes for resolvability itself (see its
    own docstring). Scans candidates in ranked order and returns the first
    resolved metric whose fields cover EVERY named concept.

    Having a matching FIELD isn't enough on its own, though — plenty of
    tables incidentally have a ward/facility/payment column without being
    what the question is actually about (observed: a "Private wards"
    question, ranked-first candidate landed on a table with a ward_name
    column but zero Private-ward rows, because a different table entirely
    was the real subject). So when the plan also named candidate_terms
    (what the user's asking to MEASURE, e.g. "revenue", "admission count"),
    additionally require EVERY one of them to match one of the metric's own
    MEASURE names — cheap corroboration that this metric doesn't just
    happen to have the filter field, it also measures ALL of what was
    asked about, not merely one generic "count" that happens to lexically
    overlap one of the terms while missing the others entirely (observed:
    a bare row-count measure satisfied "admission count" on its own but the
    same table had nothing answering "revenue" at all).

    Returns None if no candidate in range satisfies both — callers fall
    back to the plain top-ranked pick, so a genuinely inapplicable/mistyped
    concept (nothing on ANY candidate matches it) doesn't block an answer
    outright.
    """
    for candidate in candidates:
        if not is_directly_resolvable(candidate):
            continue
        metric = _resolve_candidate_metric(candidate)
        if not metric:
            continue
        cube_query = metric.get("cube_query") or {}
        field_names = {m.rsplit(".", 1)[-1].lower() for m in schema_validation.valid_members_for(metric)}
        if not all(any(_field_matches_concept(name, concept) for name in field_names) for concept in concepts):
            continue
        terms = [term.strip().lower() for term in candidate_terms if term and term.strip()]
        if terms:
            measure_names = {m.rsplit(".", 1)[-1].lower() for m in (cube_query.get("measures") or [])}
            all_terms_covered = all(
                any(_field_matches_concept(name, term) for name in measure_names)
                for term in terms
            )
            if not all_terms_covered:
                continue
        return metric
    return None


def _resolve_matched_metric(state: AgentState) -> dict | None:
    """Resolve retrieval_candidates' best directly-resolvable pick into a
    matched_metric, unless one is already set (e.g. by the resume path's
    re_classify, or by compose_derived_metric's synthesized entry)."""
    if state.get("matched_metric"):
        return state["matched_metric"]

    candidates = state.get("retrieval_candidates") or []
    relevant = [c for c in candidates if c.get("score", 0) >= RELEVANT_FLOOR]

    plan = state.get("intent_plan") or {}
    concepts = [hint["concept"].strip().lower() for hint in plan.get("filter_hints") or [] if hint.get("concept")]
    if concepts:
        preferred = _find_metric_supporting_concepts(relevant, concepts, plan.get("candidate_terms") or [])
        if preferred:
            return preferred

    top = find_first_resolvable(relevant, CONFIDENT_SINGLE_SCORE)
    return _resolve_candidate_metric(top) if top else None


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
            # _field_matches_concept (word-overlap, not just substring) —
            # a naive substring check missed real synonym pairs like hint
            # "discharge pathway" vs field "discharge_type" (no substring
            # relationship either direction, despite meaning the same
            # thing), silently dropping the group-by and aggregating every
            # discharge type into one row for "are there discharge
            # pathways where..." questions.
            if any(_field_matches_concept(field_name, hint) for hint in group_by_hints):
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
    if operator == "latest":
        # "Most recent" / "last N months" — the freshest period the DATA
        # actually has, not a calendar range computed from today (see
        # PLAN_SYSTEM's note on this: reporting data can lag today by
        # months, so a today-anchored guess silently matches zero rows).
        # Cube has no "give me whatever's newest" filter primitive, so this
        # is expressed as order-by-time-desc + limit instead of a WHERE
        # clause — the same shape the ground-truth SQL for these questions
        # uses (ORDER BY <month> DESC LIMIT N), never a date filter.
        time_members = schema_validation.valid_time_members_for(matched_metric)
        if time_members:
            member = _pick_time_member(time_members)
            try:
                periods = max(1, int(time_range.get("periods") or 1))
            except (TypeError, ValueError):
                periods = 1
            query["timeDimensions"] = [{"dimension": member, "granularity": "month"}]
            query["order"] = {member: "desc"}
            query["limit"] = periods
            logger.info(
                "generate_cube_query: 'latest' time_range — member=%s periods=%d (order-by-desc, no date filter)",
                member, periods,
            )
        else:
            warnings.append(
                f"The user asked for the most recent period, but "
                f"'{matched_metric.get('name') or matched_metric.get('id')}' has no date "
                f"field to order by — the result below covers all available data, not "
                f"just the most recent period."
            )
            logger.warning(
                "generate_cube_query: metric=%s has no date field — 'latest' time_range dropped, running unfiltered",
                matched_metric.get("id"),
            )
    elif operator and operator != "none":
        time_members = schema_validation.valid_time_members_for(matched_metric)
        values = [v for v in (time_range.get("start"), time_range.get("end")) if v]
        if time_members and values:
            # Anchor on the metric's own real time dimension rather than
            # trusting the plan to have named the right field — validation
            # would drop a mismatched name anyway, so just use the one that
            # actually exists. When more than one does, _pick_time_member
            # prefers the reporting-period field over a per-record event
            # timestamp (see its own docstring).
            member = _pick_time_member(time_members)
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
    qualifier_measures: list[str] = []
    for hint in plan.get("filter_hints") or []:
        concept = (hint.get("concept") or "").strip().lower()
        value = hint.get("value")
        if not concept or not value:
            continue

        measure_for_value = _find_measure_for_qualifier(matched_metric, value)
        if measure_for_value:
            logger.info(
                "generate_cube_query: qualifier value=%r matched measure %s directly — "
                "using it instead of a filter", value, measure_for_value,
            )
            if measure_for_value not in qualifier_measures:
                qualifier_measures.append(measure_for_value)
            continue

        # allowed_members is a set (unordered) and a broad concept like "ward"
        # can substring-match MULTIPLE real fields (ward_name AND
        # ward_category both contain "ward") — picking whichever the set
        # happens to yield first is non-deterministic and can silently
        # filter on the wrong field. Collect every match, then break the tie
        # deterministically based on the VALUE's own shape:
        #   - A multi-word value (e.g. "General Female") reads as a specific
        #     ward's full name — prefer a field ending in "_name" (observed:
        #     applying it to ward_category instead matched zero rows, since
        #     ward_category's values are broader tiers, not ward names).
        #   - A single bare qualifier word (e.g. "Private", "General") reads
        #     as a category/tier, not one specific ward's name — prefer a
        #     field ending in "_category" instead (observed: "Private" as a
        #     ward_NAME contains-filter over-matched "Private Maternity" —
        #     itself filed under ward_CATEGORY "Maternity", not "Private" —
        #     inflating the total; ward_category's own "Private / Amenity"
        #     value is the actual curated grouping the word refers to).
        # Whichever loses this tie-break is still kept as a fallback so a
        # metric with only one of the two fields still gets filtered.
        # A filter concept naming the facility axis needs its value resolved
        # through the same alias table as row-level security (e.g. "KSH" ->
        # ["KISUMU", "KISUMU_CLEAN"]) rather than filtered as literal text —
        # the real column never contains "KSH". Checked BEFORE the generic
        # concept-word matching below (not derived from its output) because
        # "facility" as a concept word has NO lexical overlap with
        # "source_schema" at all (fact_* tables' facility axis) — the
        # generic _field_matches_concept check alone would never surface it
        # as a candidate, so a resolvable facility value on a fact_* cube
        # was silently dropped every time (observed: every pharmacy/
        # dispensing question with an explicit facility mention lost that
        # filter, aggregating across every facility with no caveat other
        # than a vague "not filtered by facility" note).
        facility_candidate = next(
            (
                member for member in allowed_members
                if member.rsplit(".", 1)[-1].lower() in FACILITY_DIMENSION_NAMES
                and resolve_facility_filter_value(value, member)
            ),
            None,
        )

        is_multi_word_value = len(str(value).split()) > 1
        preferred_suffix = "_name" if is_multi_word_value else "_category"
        fallback_suffix = "_category" if is_multi_word_value else "_name"
        candidates = [
            member for member in allowed_members
            if (field_name := member.rsplit(".", 1)[-1].lower())
            and _field_matches_concept(field_name, concept)
        ]
        if facility_candidate:
            resolved_values = resolve_facility_filter_value(value, facility_candidate)
            if resolved_values:
                filters.append({
                    "member": facility_candidate,
                    "operator": "equals",
                    "values": resolved_values,
                })
                continue

        if candidates:
            def _tie_break(member: str) -> tuple[int, str]:
                field_name = member.rsplit(".", 1)[-1].lower()
                if field_name.endswith(preferred_suffix):
                    rank = 0
                elif field_name.endswith(fallback_suffix):
                    rank = 1
                else:
                    rank = 2
                return (rank, member)

            candidates.sort(key=_tie_break)
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

    if qualifier_measures:
        query["measures"] = qualifier_measures
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
