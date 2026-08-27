"""Facility-based scoping for the warehouse Snowflake query interface.

Reuses the same canonical facility registry that already scopes the
natural-language chatbot (agents/facility.py) so both surfaces enforce
identical facility boundaries instead of maintaining two alias lists.

A user with no resolvable facility (e.g. a client-wide Client Admin with no
single Facility linked) gets unrestricted access — this mirrors
resolve_facility_from_user()'s existing "None = unrestricted" contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agents.facility import FACILITY_REGISTRY, resolve_facility_from_user

# The ONE schema per facility that raw-SQL users may query. Deliberately
# excludes every *_RAW / *_V3_RAW schema (unvalidated ingestion data) and
# also excludes REPORTING — REPORTING pools every facility's rows behind a
# single source_schema/facility column, and a raw-SQL query has no reliable
# way to be forced to filter it correctly, so facility-scoped users are kept
# to their own CLEAN schema only. REPORTING-level analysis goes through the
# NL chatbot / Cube semantic layer instead, which already enforces row-level
# scoping there (see inject_facility_filter() in agents/facility.py).
#
# TENRI's live/active cleaned data lives in the bare "TENRI" schema (240
# tables, verified live against Snowflake) rather than a "TENRI_CLEAN" schema
# (1 table — legacy/unused) — so it's listed without the usual "_CLEAN" suffix.
FACILITY_CLEAN_SCHEMA: dict[str, str] = {
    "KISUMU": "KISUMU_CLEAN",
    "KAKAMEGA": "KAKAMEGA_CLEAN",
    "LODWAR": "LODWAR_CLEAN",
    "TENRI": "TENRI",
}


class FacilityScopeError(Exception):
    """Raised when a query or table-browse request reaches outside the caller's facility scope."""


@dataclass(frozen=True)
class FacilityScope:
    facility_key: str
    clean_schema: str

    @property
    def allowed_schemas(self) -> frozenset[str]:
        return frozenset({self.clean_schema})


def get_facility_scope(user) -> FacilityScope | None:
    """Return this user's FacilityScope, or None if they're unrestricted.

    Unrestricted covers: unauthenticated callers (handled upstream by the
    view's own auth/permission checks — this function just won't scope them),
    and any authenticated user whose facility can't be resolved to one of the
    facilities we have a known CLEAN schema for (e.g. a Client Admin with no
    single Facility linked, or a facility outside the known registry).
    """
    key = resolve_facility_from_user(user)
    if not key or key not in FACILITY_REGISTRY or key not in FACILITY_CLEAN_SCHEMA:
        return None

    return FacilityScope(facility_key=key, clean_schema=FACILITY_CLEAN_SCHEMA[key])


# Matches a schema-qualified table reference after FROM/JOIN, with an
# optional leading database qualifier (e.g. HOSPITALS.REPORTING.RPT_X or
# "REPORTING"."RPT_X"). Deliberately does NOT match bare/unqualified table
# names — those resolve against the connection's default session schema
# (PUBLIC, per SnowflakeClient._connect()), which holds no facility data, so
# they fail naturally in Snowflake rather than silently leaking another
# facility's rows.
_SCHEMA_TABLE_RE = re.compile(
    r'(?:FROM|JOIN)\s+(?:"?[A-Za-z0-9_]+"?\.)?"?(?P<schema>[A-Za-z0-9_]+)"?\."?(?P<table>[A-Za-z0-9_]+)"?',
    re.IGNORECASE,
)


def filter_tables_for_scope(tables: list[dict], scope: FacilityScope | None) -> list[dict]:
    """Filter a get_tables()-style record list down to the caller's allowed schemas.

    Table records are expected to carry a schema name under either
    "SCHEMA_NAME" (SnowflakeClient.get_tables()'s raw column) or
    "schema_name" (post-serialization). Unrestricted callers get everything.
    """
    if scope is None:
        return tables
    allowed = scope.allowed_schemas
    return [
        t for t in tables
        if str(t.get("SCHEMA_NAME") or t.get("schema_name") or "").upper() in allowed
    ]


def validate_query_scope(sql: str, scope: FacilityScope | None) -> None:
    """Raise FacilityScopeError if ``sql`` reaches outside the caller's facility scope.

    Every schema-qualified table reference in the query must be this
    facility's CLEAN schema — anything else (another facility's CLEAN
    schema, any *_RAW schema, REPORTING, STAGING, etc.) is rejected outright.
    A no-op when ``scope`` is None (unrestricted callers).

    This is a pragmatic, regex-based guard (consistent with this module's
    existing BLOCKED_KEYWORDS validation style in services/snowflake.py) —
    not a substitute for real Snowflake-side role/warehouse separation, but
    it stops the normal case of browsing or querying into a schema outside
    the caller's own facility.
    """
    if scope is None:
        return

    for match in _SCHEMA_TABLE_RE.finditer(sql):
        schema = match.group("schema").upper()
        if schema not in scope.allowed_schemas:
            raise FacilityScopeError(
                f"Your account is scoped to the '{scope.facility_key}' facility "
                f"({scope.clean_schema} only) — this query references schema "
                f"'{schema}', which isn't permitted."
            )
