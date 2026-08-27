"""
Schema validation for Cube queries.

Generalized out of nodes.py so both the fixed-catalog path (classify_intent /
re_classify) and the retrieval-driven pipeline (validate_query) share one
source of truth for "is this filter/query actually valid against the real
schema" — rather than each guessing independently at what Cube will accept.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Valid Cube.dev filter operators
VALID_CUBE_OPERATORS = {
    "equals", "notEquals", "contains", "notContains", "startsWith", "endsWith",
    "gt", "gte", "lt", "lte", "set", "notSet",
    "inDateRange", "notInDateRange", "beforeDate", "afterDate",
}

DATE_OPERATORS = {"inDateRange", "notInDateRange", "beforeDate", "afterDate"}


def valid_time_members_for(metric: dict) -> set[str]:
    """The real timeDimension names declared for this metric — the only
    fields a date-range-shaped operator is allowed to target."""
    query = metric.get("cube_query") or {}
    return {td.get("dimension") for td in (query.get("timeDimensions") or []) if td.get("dimension")}


def valid_members_for(metric: dict) -> set[str]:
    """
    The real measure/dimension/timeDimension names declared for this metric
    in the catalog — the only members a filter is allowed to reference.
    Grounds filter validation in the actual schema instead of just checking
    "looks like Cube.member" shape, which a hallucinated cube name (e.g. one
    copied from an unrelated example) would still pass.
    """
    query = metric.get("cube_query") or {}
    members = set(query.get("measures") or []) | set(query.get("dimensions") or [])
    members |= valid_time_members_for(metric)
    return members


def validate_filters(
    filters: list[dict],
    allowed_members: set[str] | None = None,
    time_members: set[str] | None = None,
) -> list[dict]:
    """
    Strip out any filters that are malformed, use unsupported operators, or
    (when allowed_members is given) reference a field that isn't actually
    part of the matched metric's cube_query. That last check is what catches
    an LLM hallucinating a cube name it was never shown (e.g. copying an
    unrelated few-shot example) — the "Cube.member" shape alone doesn't rule
    that out, and Cube.js returns a hard 400 rather than ignoring it.

    time_members additionally restricts date-range-shaped operators
    (inDateRange/notInDateRange/beforeDate/afterDate) to actual time
    dimensions — a member merely being an *allowed* field (e.g. a facility
    string like source_schema) doesn't mean it's date-typed. Applying a
    date-range op to a non-time column makes Cube try to cast that column's
    real values (e.g. a facility code like "TENRI") to a timestamp, which
    fails at query time rather than at validation time — and, worse, can
    collide with and suppress the separate facility-scoping filter that
    also targets that same field (see facility.inject_facility_filter).
    """
    valid = []
    for f in filters:
        member = f.get("member", "")
        operator = f.get("operator", "")
        values = f.get("values")

        if not member or not operator:
            logger.warning("validate_filters: dropping filter missing member/operator: %s", f)
            continue
        # Member must be in CubeName.fieldName format — bare words like "date" are invalid
        if "." not in member:
            logger.warning(
                "validate_filters: dropping filter — member '%s' is not in Cube.member format", member
            )
            continue
        if allowed_members is not None and member not in allowed_members:
            logger.warning(
                "validate_filters: dropping filter — '%s' isn't a field on this metric (allowed: %s)",
                member, sorted(allowed_members),
            )
            continue
        if operator not in VALID_CUBE_OPERATORS:
            logger.warning("validate_filters: dropping filter with invalid operator '%s': %s", operator, f)
            continue
        if operator in DATE_OPERATORS and time_members is not None and member not in time_members:
            logger.warning(
                "validate_filters: dropping date-range filter — '%s' isn't a time dimension on this metric (time fields: %s)",
                member, sorted(time_members),
            )
            continue
        # notSet / set don't require values
        if operator not in ("set", "notSet") and not values:
            logger.warning("validate_filters: dropping filter missing values: %s", f)
            continue

        valid.append({"member": member, "operator": operator, "values": values or []})
    return valid


def promote_date_filters(query: dict, filters: list[dict]) -> tuple[dict, list[dict]]:
    """
    Date range filters whose member matches an existing timeDimension should be
    expressed as timeDimension.dateRange, NOT as a separate filter — Cube rejects
    the latter when a timeDimension for that member already exists.

    Returns (updated_query, remaining_filters).
    """
    time_dims = [dict(td) for td in query.get("timeDimensions", [])]
    remaining = []

    for f in filters:
        if f["operator"] not in DATE_OPERATORS:
            remaining.append(f)
            continue

        member = f["member"]
        converted = False
        if f["operator"] == "inDateRange":
            for td in time_dims:
                if td.get("dimension") == member:
                    td["dateRange"] = f["values"]  # e.g. ["2023-09-01", "2023-09-30"]
                    converted = True
                    logger.info("promote_date_filters: moved %s filter to timeDimension.dateRange", member)
                    break

        if not converted:
            # Either no matching timeDimension, or a date operator (beforeDate/
            # afterDate/notInDateRange) that isn't a single [start, end] range and
            # so can't be expressed as dateRange — keep it as a regular filter
            # instead of silently discarding it (previously: any operator matching
            # an existing timeDimension's member was marked "matched" and dropped
            # here even when it was never actually applied anywhere).
            remaining.append(f)

    updated_query = {**query, "timeDimensions": time_dims}
    return updated_query, remaining


def validate_query(query: dict, allowed_members: set[str], time_members: set[str]) -> dict:
    """
    Validate a full Cube query dict against a known set of allowed fields —
    used by the retrieval-driven pipeline's validate_query node, where
    "allowed_members"/"time_members" come from live Cube metadata rather
    than a fixed catalog entry. Drops invalid filters and reports what was
    dropped, so the caller (explain_result) can surface a caveat instead of
    silently losing user-requested constraints.

    Returns {"query": <query with filters replaced by only the valid ones>,
             "dropped_filters": [{"member", "operator", "reason"}, ...]}
    """
    original_filters = query.get("filters") or []
    kept = validate_filters(original_filters, allowed_members=allowed_members, time_members=time_members)
    kept_keys = {(f["member"], f["operator"], tuple(f["values"])) for f in kept}

    dropped = []
    for f in original_filters:
        key = (f.get("member"), f.get("operator"), tuple(f.get("values") or []))
        if key not in kept_keys:
            reason = "not a valid field/operator for this query"
            if f.get("operator") in DATE_OPERATORS and f.get("member") not in time_members:
                reason = "date-range operator on a non-time field"
            elif f.get("member") not in allowed_members:
                reason = "field not part of this metric"
            dropped.append({"member": f.get("member"), "operator": f.get("operator"), "reason": reason})

    validated_query = {**query, "filters": kept}
    return {"query": validated_query, "dropped_filters": dropped}
