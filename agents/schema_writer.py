"""
Cross-cube auto-join writer (Phase 5).

Only invoked by agents/derived_metrics.py when a derived-metric ask spans
two cubes with no existing join. Per explicit user decision this ships
with NO per-request human review gate — but every join is still verified
against live data (Snowflake cardinality check) before being written, and
every attempt (written or refused) emails the analytics team for
visibility, since this mutates the shared Cube schema every user's queries
run against.

Cube YAML syntax used here (relationship enum, sql templating, calculated
measure shape) is verified against Cube's current documentation — not
guessed:
  https://docs.cube.dev/reference/data-modeling/joins
  https://docs.cube.dev/docs/data-modeling/measures
Only one_to_one / one_to_many / many_to_one are valid Cube join
relationships — many_to_many has no direct-join representation in Cube at
all, which is *why* refusing that cardinality (see would_fan_out_beyond_tolerance)
isn't just caution, it's a hard modeling requirement.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml
from django.conf import settings
from django.core.mail import send_mail

logger = logging.getLogger(__name__)

CUBES_DIR = Path(__file__).resolve().parent.parent / "model" / "cubes"

# Preferred join-key names, in priority order, based on naming conventions
# actually observed across every cube in this repo's model/cubes/*.yml.
_PREFERRED_KEY_NAMES = (
    "patient_id", "sk_patient_id", "composite_patient_id",
    "facility", "facility_key", "source_schema",
)

# A join fans out the "one" side whenever the OTHER side is the "many" side
# of a one_to_many/many_to_one relationship. Fan-out-sensitive aggregations
# (sum, count) on the fanned-out side double/triple-count — these are the
# aggregation TYPES that make a fan-out unsafe, not every aggregation.
_FAN_OUT_SENSITIVE_AGG_TYPES = {"sum", "count"}


class SchemaWriterError(Exception):
    pass


def _cube_yaml_path(cube_name: str) -> Path:
    return CUBES_DIR / f"{cube_name}.yml"


def _load_cube_def(cube_name: str) -> dict | None:
    """Parse a cube's own YAML file and return its cube dict (name/dimensions/
    measures/joins/sql_table), or None if the file doesn't exist."""
    path = _cube_yaml_path(cube_name)
    if not path.exists():
        return None
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    for cube in data.get("cubes") or []:
        if cube.get("name") == cube_name:
            return cube
    return None


def find_candidate_join_key(base_cube: str, target_cube: str) -> str | None:
    """
    Look for a dimension name that exists (same name) on BOTH cubes'
    dimension lists — a pure name-match against the already-parsed YAML,
    no LLM involved. Returns the matched dimension name, or None if no
    plausible key exists on both sides, in which case the caller MUST
    abandon the cross-cube derived metric rather than guess a join.
    """
    base = _load_cube_def(base_cube)
    target = _load_cube_def(target_cube)
    if not base or not target:
        return None

    base_dims = {d["name"] for d in (base.get("dimensions") or []) if d.get("name")}
    target_dims = {d["name"] for d in (target.get("dimensions") or []) if d.get("name")}
    common = base_dims & target_dims
    if not common:
        return None

    for preferred in _PREFERRED_KEY_NAMES:
        if preferred in common:
            return preferred
    # Fall back to any other shared dimension name, deterministically
    # (sorted) so repeated calls for the same cube pair pick the same key.
    return sorted(common)[0]


def _table_for_cube(cube_name: str) -> str:
    """
    Cube name -> Snowflake table name, per the sql_table convention observed
    across every cube in this repo: "REPORTING"."<UPPERCASE_CUBE_NAME>".
    """
    return f'"REPORTING"."{cube_name.upper()}"'


def check_join_cardinality(base_cube: str, target_cube: str, key_column: str) -> dict:
    """
    Classify the join on key_column as one_to_one / one_to_many /
    many_to_one / many_to_many by comparing row count vs. distinct-key
    count on each side, via a REAL query against live Snowflake data —
    never inferred from naming alone.
    """
    from warehouse.services.snowflake import SnowflakeClient

    client = SnowflakeClient()
    col = key_column.upper()

    def _counts(cube_name: str) -> tuple[int, int]:
        table = _table_for_cube(cube_name)
        sql = (
            f'SELECT COUNT(*) AS TOTAL, COUNT(DISTINCT "{col}") AS DISTINCT_KEYS '
            f'FROM {table} WHERE "{col}" IS NOT NULL'
        )
        df = client.query(sql, max_rows=1)
        row = df.iloc[0]
        return int(row["TOTAL"]), int(row["DISTINCT_KEYS"])

    base_total, base_distinct = _counts(base_cube)
    target_total, target_distinct = _counts(target_cube)

    base_unique = base_total == base_distinct
    target_unique = target_total == target_distinct

    if base_unique and target_unique:
        cardinality = "one_to_one"
    elif base_unique and not target_unique:
        cardinality = "one_to_many"       # base is the "one" side
    elif not base_unique and target_unique:
        cardinality = "many_to_one"       # base is the "many" side
    else:
        cardinality = "many_to_many"

    return {
        "cardinality": cardinality,
        "base_total": base_total,
        "base_distinct": base_distinct,
        "target_total": target_total,
        "target_distinct": target_distinct,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def would_fan_out_beyond_tolerance(cardinality: str, agg_types: dict[str, str]) -> bool:
    """
    Returns True (refuse the join) when:
      - cardinality is "many_to_many" — Cube has no direct-join
        representation for this at all, so there is nothing safe to write.
      - cardinality is "one_to_many" or "many_to_one" AND any measure
        referenced from the fanned-out ("one") side has a fan-out-sensitive
        aggregation type (sum/count) — summing/counting across a fan-out
        join double/triple-counts those values once per matching row on
        the other side.
    """
    if cardinality == "many_to_many":
        return True
    if cardinality in ("one_to_many", "many_to_one"):
        return any(agg_type in _FAN_OUT_SENSITIVE_AGG_TYPES for agg_type in agg_types.values())
    return False


def _notify_analytics_team_of_join(join_info: dict, refused: bool) -> None:
    analytics_email = getattr(settings, "ANALYTICS_TEAM_EMAIL", "analytics@example.com")
    from_email = getattr(settings, "DEFAULT_FROM_EMAIL", "noreply@example.com")

    status = "REFUSED" if refused else "AUTO-WRITTEN"
    subject = f"[Analytics Auto-Join {status}] {join_info['base_cube']} <-> {join_info['target_cube']} on {join_info['key_column']}"
    body = (
        f"An analytics agent {'considered but refused' if refused else 'automatically wrote'} "
        f"a cross-cube join.\n\n"
        f"Base cube    : {join_info['base_cube']}\n"
        f"Target cube  : {join_info['target_cube']}\n"
        f"Join key     : {join_info['key_column']}\n"
        f"Cardinality  : {join_info['cardinality']['cardinality']} "
        f"(base {join_info['cardinality']['base_total']} rows / "
        f"{join_info['cardinality']['base_distinct']} distinct; "
        f"target {join_info['cardinality']['target_total']} rows / "
        f"{join_info['cardinality']['target_distinct']} distinct)\n"
        f"Triggering question: {join_info.get('question', 'unknown')}\n\n"
        + (
            "This join was NOT written — the cardinality would silently fan out "
            "one or more sum/count measures. Please model this relationship by hand "
            "(e.g. a pre-aggregation or an associative cube) if it's needed.\n"
            if refused else
            f"Written to model/cubes/{join_info['target_cube']}.yml — review at your "
            f"convenience; this is already live.\n"
        )
    )
    try:
        send_mail(subject, body, from_email, [analytics_email], fail_silently=False)
        logger.info("schema_writer: notified analytics team (%s) of join %s<->%s",
                    status, join_info["base_cube"], join_info["target_cube"])
    except Exception as exc:
        logger.error("schema_writer: failed to send join notification email — %s", exc)


def _resolve_join_direction(base_cube: str, target_cube: str, cardinality: str) -> tuple[str, str, str]:
    """
    Cube's join convention: the "many" side declares the join, pointing at
    the "one" side, labeled many_to_one (from the declaring cube's own
    perspective). Getting this backwards doesn't error — Cube would accept
    it — it just silently mislabels the relationship, which is exactly the
    kind of mistake the cardinality check exists to prevent, so this is
    computed explicitly rather than assumed to always be "target owns it".

    Returns (owner_cube, referenced_cube, relationship) — owner_cube is
    whichever file actually gets the joins: entry.
    """
    if cardinality == "one_to_many":
        # base is the "one" side, target is the "many" side.
        return target_cube, base_cube, "many_to_one"
    if cardinality == "many_to_one":
        # base is the "many" side, target is the "one" side.
        return base_cube, target_cube, "many_to_one"
    if cardinality == "one_to_one":
        # Either side works; declare on target pointing at base.
        return target_cube, base_cube, "one_to_one"
    raise SchemaWriterError(f"cannot resolve join direction for cardinality {cardinality!r}")


def _splice_join_into_yaml(
    owner_cube: str, referenced_cube: str, relationship: str, key_column: str, confidence: float, question: str,
) -> None:
    """
    Insert a joins: entry into owner_cube's own YAML file via a targeted
    text splice (not a full parse/re-dump — pyyaml drops every existing
    comment on re-dump, and every cube file here has real, load-bearing
    header/pre_aggregations comments).

    Verified Cube syntax (see module docstring for the doc URLs):
      joins:
        - name: <other_cube>
          relationship: many_to_one | one_to_many | one_to_one
          sql: "{CUBE}.key = {other_cube}.key"
    """
    path = _cube_yaml_path(owner_cube)
    text = path.read_text()

    timestamp = datetime.now(timezone.utc).isoformat()
    block = (
        f"    joins:\n"
        f"      # Auto-added by derived-metric agent on {timestamp}\n"
        f"      # Confidence: {confidence:.2f} | Join key: {key_column}\n"
        f"      # Triggered by question: {question!r}\n"
        f"      - name: {referenced_cube}\n"
        f"        relationship: {relationship}\n"
        f'        sql: "{{CUBE}}.\\"{key_column.upper()}\\" = {{{referenced_cube}}}.\\"{key_column.upper()}\\""\n'
    )

    if re.search(r"^\s*joins:\s*\[\]\s*$", text, re.MULTILINE):
        new_text = re.sub(r"^\s*joins:\s*\[\]\s*$", block.rstrip("\n"), text, count=1, flags=re.MULTILINE)
    elif re.search(r"^\s*joins:\s*$", text, re.MULTILINE):
        # Empty `joins:` block with no `[]` shorthand — splice right after it.
        new_text = re.sub(r"^\s*joins:\s*$", block.rstrip("\n"), text, count=1, flags=re.MULTILINE)
    else:
        raise SchemaWriterError(f"could not find a 'joins:' key to splice into {path}")

    path.write_text(new_text)
    logger.info(
        "schema_writer: wrote join %s -> %s (relationship=%s, key=%s) into %s",
        owner_cube, referenced_cube, relationship, key_column, path,
    )


def attempt_auto_join(
    base_cube: str,
    target_cube: str,
    agg_types: dict[str, str],
    question: str,
    confidence: float = 0.0,
) -> dict:
    """
    Full pipeline for one cross-cube join attempt: find a candidate key,
    verify its cardinality against live Snowflake data, and either write
    the join (fully automatic, no review gate — per explicit product
    decision) or refuse and notify the analytics team either way.

    agg_types: {"<alias>": "sum"|"count"|"avg"|...} — the Cube measure
    `type` of each base measure the derived metric would reference, used
    to decide whether a one_to_many/many_to_one fan-out is safe.

    Returns {"written": bool, "key_column": str|None, "cardinality": dict|None,
             "reason": str|None}.
    """
    key_column = find_candidate_join_key(base_cube, target_cube)
    if not key_column:
        logger.info("schema_writer: no candidate join key found for %s <-> %s", base_cube, target_cube)
        return {"written": False, "key_column": None, "cardinality": None, "reason": "no shared dimension name found"}

    cardinality = check_join_cardinality(base_cube, target_cube, key_column)
    join_info = {
        "base_cube": base_cube, "target_cube": target_cube, "key_column": key_column,
        "cardinality": cardinality, "question": question,
    }

    if would_fan_out_beyond_tolerance(cardinality["cardinality"], agg_types):
        logger.warning(
            "schema_writer: refusing join %s <-> %s on %s — cardinality=%s would fan out",
            base_cube, target_cube, key_column, cardinality["cardinality"],
        )
        _notify_analytics_team_of_join(join_info, refused=True)
        return {
            "written": False, "key_column": key_column, "cardinality": cardinality,
            "reason": f"cardinality {cardinality['cardinality']} would fan out a sum/count measure",
        }

    try:
        owner_cube, referenced_cube, relationship = _resolve_join_direction(
            base_cube, target_cube, cardinality["cardinality"]
        )
        _splice_join_into_yaml(owner_cube, referenced_cube, relationship, key_column, confidence, question)
    except Exception as exc:
        logger.error("schema_writer: failed to write join %s <-> %s — %s", base_cube, target_cube, exc)
        return {"written": False, "key_column": key_column, "cardinality": cardinality, "reason": str(exc)}

    _notify_analytics_team_of_join(join_info, refused=False)
    return {"written": True, "key_column": key_column, "cardinality": cardinality, "reason": None}
