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

FACILITY_REGISTRY: dict[str, dict[str, str]] = {
    "KISUMU": {
        "source_schema": "KISUMU_CLEAN",
        "facility":      "KISUMU",
    },
    "KAKAMEGA": {
        "source_schema": "KAKAMEGA_CLEAN",
        "facility":      "KAKAMEGA",
    },
    "LODWAR": {
        "source_schema": "LODWAR_CLEAN",
        "facility":      "LODWAR",
    },
    "TENRI": {
        "source_schema": "TENRI",
        "facility":      "TENRI",
    },
}

# Dimension names that represent the facility/schema axis in Cube
FACILITY_DIMENSION_NAMES = {"source_schema", "facility"}


# ── Keyword matching ──────────────────────────────────────────────────────────
# Maps canonical facility key → keywords to icontains-match against Facility.name.
# Order matters — more specific keywords first to avoid false matches.

FACILITY_KEYWORDS: dict[str, list[str]] = {
    "KISUMU":   ["kisumu"],
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
    for field in ("schema_name", "source_schema", "code"):
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
    matched_value: str | None = None

    query_dims = set(query.get("dimensions", []))

    # Pass 1 — prefer a dimension that is explicitly selected in this query
    for dim_suffix, filter_value in schema_map.items():
        candidate = f"{cube_prefix}.{dim_suffix}"
        if candidate in query_dims:
            matched_dim = candidate
            matched_value = filter_value
            break

    # Pass 2 — infer from cube naming convention when no dimension was selected
    if not matched_dim:
        if cube_prefix.startswith("rpt_"):
            matched_dim = f"{cube_prefix}.facility"
            matched_value = schema_map.get("facility")
        elif cube_prefix.startswith("fact_"):
            matched_dim = f"{cube_prefix}.source_schema"
            matched_value = schema_map.get("source_schema")

    if not matched_dim or not matched_value:
        logger.debug(
            "inject_facility_filter: no facility dimension resolved for %s — skipping",
            cube_prefix,
        )
        return query

    # Don't double-inject
    existing = query.get("filters", [])
    for f in existing:
        if f.get("member") == matched_dim:
            logger.debug(
                "inject_facility_filter: %s filter already present — skipping",
                matched_dim,
            )
            return query

    logger.info(
        "inject_facility_filter: %s = %s  (facility=%s)",
        matched_dim, matched_value, facility_key,
    )

    query = dict(query)
    query["filters"] = existing + [
        {"member": matched_dim, "operator": "equals", "values": [matched_value]}
    ]
    return query