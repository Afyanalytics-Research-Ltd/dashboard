"""
agent/facility.py
─────────────────
Resolves a user_id / phone_number → facility and injects the correct
schema filter into any Cube query automatically.

Resolution order (first match wins):
  1. UserProfile.facility  looked up by username / email      (Django DB)
  2. UserProfile.facility  looked up by phone_number          (Django DB)
  3. USER_FACILITY_MAP in settings                            (static fallback)
  4. USER_DOMAIN_MAP  in settings                            (domain fallback)
  5. None → no filter, user sees all facilities

Why two filter values per facility?
  - fact_* cubes use  source_schema  e.g. "KISUMU_CLEAN"
  - rpt_*  cubes use  facility       e.g. "KISUMU"

The injector detects which dimension exists in the cube and picks the
right value automatically.

Facility.name → canonical key mapping is configured via FACILITY_SCHEMA_MAP
in settings.  Defaults are provided for the four known facilities.
"""

from __future__ import annotations

import logging

from django.conf import settings

logger = logging.getLogger(__name__)


# ── Canonical facility registry ───────────────────────────────────────────────
# Maps canonical facility key → filter values for each cube family.

FACILITY_REGISTRY: dict[str, dict[str, str | list[str]]] = {
    # source_schema AND facility both carry a list, not a single string: raw
    # ingestion has left some rows under inconsistent naming (e.g.
    # lowercase "kisumu", or a "_CLEAN"-suffixed value in the "facility"
    # column of some rpt_* tables) alongside the normalized value — an
    # "equals" filter with only the canonical value silently misses those
    # rows and undercounts, or returns zero rows outright (observed:
    # rpt_bed_occupancy.facility is literally "KISUMU_CLEAN" for this
    # facility's rows, not "KISUMU" — a KISUMU user's BTR/BTI questions
    # were being scoped to a facility value that matches nothing at all).
    # Cube's "equals" operator with multiple values compiles to SQL
    # IN(...), so listing every known alias here is enough — no operator
    # change needed. TENRI's "facility" column matches its bare name
    # exactly (confirmed against live data), so it's left as a single value.
    "KISUMU": {
        # Three different conventions across cube families all target this
        # same "source_schema" axis: rpt_bed_occupancy-style tables use
        # "KISUMU_CLEAN", canonical_product_taxonomy/fact_dispensing use
        # lowercase "kisumu", and the clinical-domain tables (rpt_opd_ipd,
        # rpt_case_mix, rpt_clinical_activity, etc.) use bare uppercase
        # "KISUMU" with no suffix at all. Missing any one of the three
        # means every question against that cube family silently drops the
        # facility filter (observed: the entire 100-question clinical test
        # suite lost its KISUMU scoping this way — bare "KISUMU" wasn't in
        # this list, so it never matched the source_schema value those
        # tables actually store).
        "source_schema": ["KISUMU_CLEAN", "kisumu", "KISUMU"],
        # canonical_product_taxonomy.facility uses lowercase "kisumu" —
        # a different convention than the rpt_* tables' "KISUMU_CLEAN" —
        # so both must be listed or that cube's facility filter matches
        # nothing (observed: every canonical_product_taxonomy-joined
        # question about KSH silently returned zero rows).
        "facility":      ["KISUMU", "KISUMU_CLEAN", "kisumu"],
    },
    "KAKAMEGA": {
        "source_schema": ["KAKAMEGA_CLEAN", "kakamega", "KAKAMEGA"],
        "facility":      ["KAKAMEGA", "KAKAMEGA_CLEAN", "kakamega"],
    },
    "LODWAR": {
        "source_schema": ["LODWAR_CLEAN", "lodwar", "LODWAR"],
        "facility":      ["LODWAR", "LODWAR_CLEAN", "lodwar"],
    },
    "TENRI": {
        "source_schema": ["TENRI", "tenri"],
        "facility":      "TENRI",
    },
    "SPH": {
            "source_schema": ["SPH", "sph"],
            "facility":      ["SPH"],
    }
}

# Dimension names that represent the facility/schema axis in Cube
FACILITY_DIMENSION_NAMES = {"source_schema", "facility"}


# ── Keyword matching ──────────────────────────────────────────────────────────
# Maps canonical facility key → keywords to icontains-match against Facility.name.
# Order matters — more specific keywords first to avoid false matches.

FACILITY_KEYWORDS: dict[str, list[str]] = {
    "KISUMU":   ["kisumu", "ksh"],
    "KAKAMEGA": ["kakamega"],
    "LODWAR":   ["lodwar", "turkana"],
    "TENRI":    ["tenri"],
}


def _match_by_keywords(facility_name: str) -> str | None:
    """
    Case-insensitive substring match of facility_name against FACILITY_KEYWORDS.

    Examples:
        "Kisumu County Referral Hospital"  → "KISUMU"
        "KAKAMEGA TEACHING HOSPITAL"       → "KAKAMEGA"
        "Lodwar County Referral"           → "LODWAR"
        "Tenri Hospital Kenya"             → "TENRI"
    """
    name_lower = facility_name.lower()
    for key, keywords in FACILITY_KEYWORDS.items():
        for kw in keywords:
            if kw in name_lower:
                logger.debug(
                    "_match_by_keywords: '%s' matched keyword '%s' → %s",
                    facility_name, kw, key,
                )
                return key
    return None


# ── UserProfile DB lookup ─────────────────────────────────────────────────────

def _profile_to_facility_key(profile) -> str | None:
    """
    Extract the canonical facility key from a UserProfile instance.

    Resolution order:
      1. Keyword icontains match on Facility.name  (e.g. "kisumu" → KISUMU)
      2. Facility fields: schema_name / source_schema / code
      3. Exact match via FACILITY_SCHEMA_MAP in settings (legacy override)
    """
    if not profile or not profile.facility:
        return None

    facility_obj = profile.facility
    facility_name: str = getattr(facility_obj, "name", "") or ""

    # 1. Keyword icontains — works for any facility name containing the keyword
    if facility_name:
        key = _match_by_keywords(facility_name)
        if key:
            return key

    # 2. Dedicated schema field on the Facility model (if it exists)
    for field in ("reporting_source_schema", "schema_name", "source_schema", "code"):
        val = getattr(facility_obj, field, None)
        if val:
            candidate = str(val).upper().replace("_CLEAN", "")
            if candidate in FACILITY_REGISTRY:
                return candidate
            # Also try keyword match on the field value
            key = _match_by_keywords(str(val))
            if key:
                return key

    # 3. Explicit override map in settings (escape hatch for unusual names)
    schema_map: dict = getattr(settings, "FACILITY_SCHEMA_MAP", {})
    if facility_name in schema_map:
        return schema_map[facility_name].upper()

    logger.warning(
        "_profile_to_facility_key: could not map facility '%s' to a canonical key. "
        "Add a keyword to FACILITY_KEYWORDS or an entry to FACILITY_SCHEMA_MAP.",
        facility_name,
    )
    return None


def resolve_facility_from_user(user) -> str | None:
    """
    Resolve facility directly from a Django User object (request.user).

    This is the preferred method when the request is authenticated —
    avoids the string-based lookup entirely and goes straight to the profile.

    Args:
        user: A Django User instance (request.user).

    Returns:
        Canonical facility key e.g. "KISUMU", or None if unresolvable.
    """
    try:
        from authentication.models import UserProfile  # adjust import path if needed

        profile = (
            UserProfile.objects
            .select_related("facility")
            .get(user=user)
        )
        key = _profile_to_facility_key(profile)
        if key:
            logger.info(
                "resolve_facility_from_user: user=%s → %s (facility='%s')",
                user.username, key,
                getattr(profile.facility, "name", ""),
            )
        else:
            logger.warning(
                "resolve_facility_from_user: user=%s has no resolvable facility",
                user.username,
            )
        return key
    except Exception as exc:
        logger.debug("resolve_facility_from_user(%s): %s", getattr(user, "username", user), exc)
        return None


def _lookup_by_username_or_email(user_id: str) -> str | None:
    """Try to find UserProfile by Django username or email."""
    try:
        from authentication.models import UserProfile  # adjust import path if needed

        profile = (
            UserProfile.objects
            .select_related("facility")
            .filter(user__username=user_id)
            .first()
            or
            UserProfile.objects
            .select_related("facility")
            .filter(user__email=user_id)
            .first()
        )
        return _profile_to_facility_key(profile)
    except Exception as exc:
        logger.debug("_lookup_by_username_or_email(%s): %s", user_id, exc)
        return None


def _lookup_by_phone(phone: str) -> str | None:
    """
    Find UserProfile by phone_number.

    Handles WhatsApp chat_id format ("254700701209@s.whatsapp.net")
    by stripping the suffix before querying.
    """
    if not phone:
        return None

    # Strip WhatsApp suffix
    clean_phone = phone.split("@")[0]

    try:
        from authentication.models import UserProfile  # adjust import path if needed

        profile = (
            UserProfile.objects
            .select_related("facility")
            .filter(phone_number=clean_phone)
            .first()
            or
            # Some systems store with country prefix; try both
            UserProfile.objects
            .select_related("facility")
            .filter(phone_number__endswith=clean_phone[-9:])
            .first()
        )
        return _profile_to_facility_key(profile)
    except Exception as exc:
        logger.debug("_lookup_by_phone(%s): %s", phone, exc)
        return None


# ── Public resolution function ────────────────────────────────────────────────

def resolve_facility(user_id: str, phone: str | None = None) -> str | None:
    """
    Return the canonical facility key (e.g. "KISUMU") for this user,
    or None if the user has no facility restriction.

    Args:
        user_id:  Django username, email, or WhatsApp chat_id
                  (e.g. "254700701209@s.whatsapp.net").
        phone:    Optional raw phone number for an additional lookup pass.

    Resolution order:
        1. DB lookup by username / email  → UserProfile.facility
        2. DB lookup by phone_number      → UserProfile.facility
        3. USER_FACILITY_MAP in settings  → static explicit map
        4. USER_DOMAIN_MAP  in settings   → email domain suffix
        5. None                           → unrestricted
    """
    # 1. DB: username or email
    key = _lookup_by_username_or_email(user_id)
    if key:
        logger.info("resolve_facility: %s → %s (DB username/email)", user_id, key)
        return key

    # 2. DB: phone number (covers WhatsApp chat_id and raw phone)
    key = _lookup_by_phone(user_id) or _lookup_by_phone(phone)
    if key:
        logger.info("resolve_facility: %s → %s (DB phone)", user_id, key)
        return key

    # 3. Static explicit map
    explicit_map: dict = getattr(settings, "USER_FACILITY_MAP", {})
    if user_id in explicit_map:
        key = explicit_map[user_id].upper()
        logger.info("resolve_facility: %s → %s (settings USER_FACILITY_MAP)", user_id, key)
        return key

    # 4. Email domain map
    if "@" in user_id:
        domain = user_id.split("@", 1)[1].lower()
        domain_map: dict = getattr(settings, "USER_DOMAIN_MAP", {})
        for dom, fac in domain_map.items():
            if domain == dom.lower() or domain.endswith(f".{dom.lower()}"):
                key = fac.upper()
                logger.info(
                    "resolve_facility: %s → %s (domain match %s)", user_id, key, dom
                )
                return key

    logger.debug("resolve_facility: %s → unrestricted", user_id)
    return None


# ── Free-text facility filter resolution ──────────────────────────────────────
# Used by generate_cube_query (agents/nodes_query.py) when the intent planner
# extracts a facility mention as a filter_hint (e.g. concept="facility",
# value="KSH") from the question text itself — distinct from the row-level
# security path above, which only fires when the CALLING USER is scoped to a
# single facility. A multi-facility user (e.g. a Client Administrator covering
# both KSH and TENRI) asking about one facility BY NAME has no user_facility
# to inject, so without this the filter_hint's raw text ("KSH") was applied
# verbatim as a literal equals/contains filter — which never matches the real
# column value ("KISUMU_CLEAN") and silently returns null/empty results.


def resolve_facility_filter_value(raw_value: str, dimension_field: str) -> list[str] | None:
    """
    Resolve a free-text facility mention (e.g. "KSH", "Kisumu", "Tenri
    Hospital") to the exact DB-value list for the given Cube dimension
    (e.g. "rpt_bed_occupancy.facility" or "fact_x.source_schema"), using the
    same keyword table and alias list as the row-level-security path.

    Returns None if raw_value doesn't match any known facility keyword, so
    the caller can fall back to filtering on the literal text instead.
    """
    key = _match_by_keywords(str(raw_value))
    if not key or key not in FACILITY_REGISTRY:
        return None

    dim_suffix = dimension_field.rsplit(".", 1)[-1]
    values = FACILITY_REGISTRY[key].get(dim_suffix)
    if not values:
        return None
    return values if isinstance(values, list) else [values]


# ── Live cube schema cache (for verifying Pass 2's naming-convention guess) ────
# Same lazy-load + explicit-reload shape as agents/retrieval.py's embeddings
# index cache — cube dimension lists change rarely (only via the Semantic
# Layer Configuration approve flow or a manual schema edit), so a per-process
# cache avoids an extra Cube /meta round trip on every single query.

_cube_dimensions_cache: dict[str, set[str]] | None = None


def _cube_dimension_names(cube_name: str) -> set[str]:
    global _cube_dimensions_cache
    if _cube_dimensions_cache is None:
        from . import cube_client

        meta = cube_client.fetch_meta()
        _cube_dimensions_cache = {
            cube["name"]: {d["name"].rsplit(".", 1)[-1] for d in cube.get("dimensions", [])}
            for cube in meta.get("cubes", [])
        }
    return _cube_dimensions_cache.get(cube_name, set())


def reload_cube_dimensions_cache() -> None:
    """Bust the cache — call after a cube's dimensions change (e.g. a new
    measure/dimension approved via the Semantic Layer Configuration page)."""
    global _cube_dimensions_cache
    _cube_dimensions_cache = None


# ── Query filter injection ────────────────────────────────────────────────────

def inject_facility_filter(query: dict, facility_key: str) -> dict:
    """
    Inject a facility filter into a Cube query dict.

    Detects whether the cube uses `source_schema` (fact_ tables) or
    `facility` (rpt_ tables) and inserts the correct filter value.

    If the cube has no facility dimension, the query is returned unchanged.
    If a facility filter is already present, it is not duplicated.
    """
    if facility_key not in FACILITY_REGISTRY:
        logger.warning(
            "inject_facility_filter: unknown facility key '%s' — skipping",
            facility_key,
        )
        return query

    schema_map = FACILITY_REGISTRY[facility_key]

    # Derive cube prefix from the first member name
    all_members: list[str] = (
        query.get("measures", [])
        + query.get("dimensions", [])
        + [td["dimension"] for td in query.get("timeDimensions", [])]
    )
    if not all_members:
        return query

    cube_prefix = all_members[0].rsplit(".", 1)[0]

    # Find which facility dimension this cube actually exposes.
    # Strategy (in priority order):
    #   1. The dimension is explicitly listed in query["dimensions"] — most reliable.
    #   2. Cube naming convention: rpt_* → "facility", fact_* → "source_schema".
    # The old "dim_suffix in FACILITY_DIMENSION_NAMES" fallback matched both keys
    # unconditionally and always chose the first one (source_schema), which broke
    # rpt_ cubes that only expose "facility".
    matched_dim: str | None = None
    matched_value: str | list[str] | None = None

    query_dims = set(query.get("dimensions", []))

    # Pass 1 — prefer a dimension that is explicitly selected in this query
    for dim_suffix, filter_value in schema_map.items():
        candidate = f"{cube_prefix}.{dim_suffix}"
        if candidate in query_dims:
            matched_dim = candidate
            matched_value = filter_value
            break

    # Pass 2 — infer from cube naming convention when no dimension was selected.
    # The naming convention is a good GUESS, not a guarantee — some rpt_*
    # cubes (e.g. rpt_doctor_performance: only username/visit_month) have no
    # facility dimension at all. Guessing one anyway sends Cube a filter on a
    # field that doesn't exist, which it rejects with a hard 400 — this
    # crashed real questions in production before this check existed. Verify
    # against the cube's own live schema before committing to the guess.
    if not matched_dim:
        real_dims = _cube_dimension_names(cube_prefix)
        if cube_prefix.startswith("rpt_") and "facility" in real_dims:
            matched_dim = f"{cube_prefix}.facility"
            matched_value = schema_map.get("facility")
        elif cube_prefix.startswith("fact_") and "source_schema" in real_dims:
            matched_dim = f"{cube_prefix}.source_schema"
            matched_value = schema_map.get("source_schema")

    if not matched_dim or not matched_value:
        logger.debug(
            "inject_facility_filter: no facility dimension resolved for %s — skipping",
            cube_prefix,
        )
        return query

    # Row-level security: ALWAYS enforce the correct facility filter on this
    # dimension, replacing any filter already targeting it rather than
    # skipping when one is merely present. An existing filter on this exact
    # field is not necessarily the facility scope — it could be an
    # unrelated (even hallucinated) LLM-extracted filter that happens to
    # target the same column, which would otherwise silently suppress row-
    # level scoping entirely rather than just being redundant with it.
    existing = [f for f in query.get("filters", []) if f.get("member") != matched_dim]

    logger.info(
        "inject_facility_filter: %s = %s  (facility=%s)",
        matched_dim, matched_value, facility_key,
    )

    values = matched_value if isinstance(matched_value, list) else [matched_value]

    query = dict(query)
    query["filters"] = existing + [
        {"member": matched_dim, "operator": "equals", "values": values}
    ]
    return query