"""
Semantic Layer Configuration settings page — service layer.

Backs agents/views.py (the settings-page UI) and the CLI management
commands (agents/management/commands/generate_metrics.py,
rebuild_embeddings.py). Four independent operations:

  list_live_cube_fields()       — live Cube /meta cross-referenced against
                                   which cubes already have a curated
                                   MetricDefinition, for the "live vs
                                   curated" view.
  generate_missing_metrics()    — additive-only LLM drafting of
                                   MetricDefinition rows for cubes with none
                                   yet (reuses catalog/generate_metrics.py's
                                   prompt, never touches a cube that already
                                   has an entry — unlike that script's own
                                   CLI mode, which regenerates its whole
                                   output file from scratch every run).
  rebuild_embeddings()          — adapts catalog/build_embeddings.py to read
                                   metrics from the DB instead of YAML;
                                   glossary terms still come from
                                   catalog/glossary.yaml (out of scope for
                                   this feature — only metrics.yaml moved).
  validate_column_exists()      — live Snowflake column-existence probe,
                                   the safety net that justifies staging
                                   PendingCubeMeasure for approval instead of
                                   auto-writing it the way schema_writer.py
                                   auto-writes joins.
  write_pending_measure_to_yaml() — the approval action: splices a new
                                   measure into an existing cube's
                                   model/cubes/<cube>.yml (Cube hot-reloads,
                                   CUBEJS_DEV_MODE=true — no restart needed).
  sync_cube_schemas_from_snowflake() — introspects every REPORTING table an
                                   existing cube maps to and auto-writes
                                   whatever measures/dimensions Cube doesn't
                                   expose yet — no approval step, unlike the
                                   single-measure Propose/Approve flow above
                                   (explicit product decision; see that
                                   function's own docstring for why).
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from django.conf import settings
from django.core.mail import send_mail

from . import cube_client
from .catalog import get_all
from .models import MetricDefinition, PendingCubeMeasure

logger = logging.getLogger(__name__)

CUBES_DIR = Path(__file__).resolve().parent.parent / "model" / "cubes"


# ── live vs. curated ────────────────────────────────────────────────────────

def _covered_cube_names() -> set[str]:
    """Cube names referenced by ANY existing MetricDefinition's measures."""
    covered: set[str] = set()
    for metric in get_all():
        for field in (metric.get("cube_query") or {}).get("measures") or []:
            if "." in field:
                covered.add(field.split(".", 1)[0])
    return covered


def _parse_cube_measures(cube_name: str) -> list[dict]:
    """Every measure's full definition (name/sql/type/title/description) on
    cube_name's own model/cubes/<cube_name>.yml, in one file read — backs
    both list_live_cube_fields()'s per-measure Edit buttons and
    get_cube_measure_definition()'s single-measure lookup. Empty list if
    the file doesn't exist / won't parse (live-only, not-yet-modeled cube,
    or a read error) rather than raising — this only ever feeds a display,
    never a write."""
    path = CUBES_DIR / f"{cube_name}.yml"
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return []
    for cube in data.get("cubes") or []:
        if cube.get("name") == cube_name:
            return [
                {
                    "name": m.get("name", ""),
                    "sql": m.get("sql", "") or "",
                    "type": m.get("type", "") or "",
                    "title": m.get("title", "") or "",
                    "description": m.get("description", "") or "",
                }
                for m in (cube.get("measures") or [])
                if m.get("name")
            ]
    return []


def list_live_cube_fields() -> list[dict]:
    """
    Live Cube schema (via /meta), one entry per cube, flagged with whether a
    curated MetricDefinition already covers it — powers the "what's live
    but not yet catalogued" section of the settings page. measure_details
    carries each measure's full sql/type/title/description (read from the
    cube's own YAML — Cube's /meta doesn't expose a measure's SQL) so the
    page can offer an Edit button per measure, not just per cube.
    """
    meta = cube_client.fetch_meta()
    covered = _covered_cube_names()

    return [
        {
            "name": cube["name"],
            "measures": [m["name"] for m in cube.get("measures", [])],
            "dimensions": [d["name"] for d in cube.get("dimensions", [])],
            "has_metric_definition": cube["name"] in covered,
            "measure_details": _parse_cube_measures(cube["name"]),
        }
        for cube in meta.get("cubes", [])
    ]


# ── generate missing metrics (additive only) ────────────────────────────────

def generate_missing_metrics(user) -> dict:
    """
    Drafts one MetricDefinition per live cube that has NO existing
    MetricDefinition covering it yet — never touches a cube that already
    has one. Reuses catalog/generate_metrics.py's SYSTEM_PROMPT/
    generate_metric/build_cube_query (same one-LLM-call-per-cube shape),
    just persists to the DB instead of overwriting the whole metrics.yaml
    file on every run.

    Returns {"created": [metric_id, ...], "skipped": [cube_name, ...],
             "failed": [cube_name, ...]}.
    """
    from catalog.generate_metrics import SKIP_CUBES, build_cube_query, generate_metric
    from .nodes import _openai

    meta = cube_client.fetch_meta()
    cubes = meta.get("cubes", [])
    covered = _covered_cube_names()
    existing_ids = set(MetricDefinition.objects.values_list("metric_id", flat=True))

    client = _openai()
    created: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    for cube in cubes:
        name = cube["name"]
        if name in covered or name in SKIP_CUBES or not cube.get("measures"):
            skipped.append(name)
            continue

        llm = generate_metric(client, cube, "gpt-4o-mini")
        if llm is None:
            failed.append(name)
            continue

        metric_id = llm.get("id") or name
        base_id = metric_id
        suffix = 1
        while metric_id in existing_ids:
            suffix += 1
            metric_id = f"{base_id}_{suffix}"
        existing_ids.add(metric_id)

        MetricDefinition.objects.create(
            metric_id=metric_id,
            name=llm.get("name") or name,
            description=llm.get("description") or "",
            cube_query=build_cube_query(llm),
            created_by=user,
            updated_by=user,
        )
        created.append(metric_id)

    logger.info(
        "generate_missing_metrics: created=%d skipped=%d failed=%d",
        len(created), len(skipped), len(failed),
    )
    return {"created": created, "skipped": skipped, "failed": failed}


# ── rebuild embeddings ──────────────────────────────────────────────────────

def rebuild_embeddings() -> dict:
    """
    Adapts catalog/build_embeddings.py's collect_entries()/embed_all() to
    read metrics from the DB (agents.catalog.get_all()) instead of parsing
    catalog/metrics.yaml. Glossary terms still come from
    catalog/glossary.yaml on disk — out of scope for this feature.

    Synchronous — this makes one OpenAI embeddings call per batch of ~100
    entries, matching the existing button-triggered sync pattern used
    elsewhere in this app (no task queue available).
    """
    from openai import OpenAI

    from catalog.build_embeddings import DEFAULT_GLOSSARY, DEFAULT_OUTPUT, collect_entries, embed_all

    meta = cube_client.fetch_meta()
    cubes = meta.get("cubes", [])
    metrics = get_all()

    with open(DEFAULT_GLOSSARY, "r") as f:
        glossary_terms = (yaml.safe_load(f) or {}).get("terms", [])

    entries = collect_entries(cubes, metrics, glossary_terms)

    api_key = getattr(settings, "OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    vectors = embed_all(client, entries)

    ids = np.array([e["id"] for e in entries], dtype=object)
    sources = np.array([e["source"] for e in entries], dtype=object)
    metadata_json = json.dumps([e["metadata"] for e in entries])

    np.savez_compressed(
        DEFAULT_OUTPUT,
        vectors=vectors,
        ids=ids,
        sources=sources,
        metadata=np.array(metadata_json),
    )

    counts = {src: sum(1 for e in entries if e["source"] == src)
              for src in ("metric", "measure", "dimension", "glossary")}
    logger.info("rebuild_embeddings: %s -> %s", counts, DEFAULT_OUTPUT)
    return counts


# ── validate + write a pending cube measure ─────────────────────────────────

_QUOTED_COLUMN_RE = re.compile(r'"([A-Z0-9_]+)"')

# Cube's calculated-measure templating — a measure's sql: can reference
# another measure/dimension on the SAME cube by name in curly braces (e.g.
# "{total_admissions} / NULLIF({bed_count}, 0)"), which Cube resolves at
# query time. "{CUBE}" is Cube's own keyword for "this cube's SQL table
# alias" (as in {CUBE}."COLUMN") — not a member reference, so excluded.
# https://cube.dev/docs/product/data-modeling/concepts/calculated-measures
_MEMBER_REF_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _load_cube_member_names(cube_name: str) -> set[str] | None:
    """All measure + dimension names currently defined on cube_name's own
    model/cubes/<cube_name>.yml, or None if the file doesn't exist / won't
    parse. Mirrors schema_writer._load_cube_def's read-only parse."""
    path = CUBES_DIR / f"{cube_name}.yml"
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return None
    for cube in data.get("cubes") or []:
        if cube.get("name") == cube_name:
            names = {d["name"] for d in (cube.get("dimensions") or []) if d.get("name")}
            names |= {m["name"] for m in (cube.get("measures") or []) if m.get("name")}
            return names
    return None


def validate_column_exists(cube_name: str, sql_expression: str) -> tuple[bool, str]:
    """
    Validates a proposed measure's sql_expression before it's allowed to be
    written to the live cube schema. Two shapes, two different checks —
    there's no single mechanical rule that covers both, so the expression's
    shape decides which applies:

      1. A raw column reference (e.g. {CUBE}."TOTAL_ADMISSIONS", or a CASE
         expression built on one) — probed against live Snowflake. This is
         the safety net that motivated staging PendingCubeMeasure for
         approval instead of auto-writing it like schema_writer.py's joins
         (a hand-typed measure has no automated cardinality-style check).

      2. A calculated measure (Cube's "{member}" templating, e.g.
         "{total_admissions} / NULLIF({bed_count}, 0)") — has no raw column
         to probe at all, so instead every referenced name is checked
         against the SAME cube's own already-defined measures/dimensions.
         Catches a typo'd or not-yet-added reference, which is the
         equivalent mistake this measure shape can actually make.

    Table naming for (1) mirrors schema_writer._table_for_cube exactly:
    "REPORTING"."<CUBE_NAME_UPPER>".
    """
    if not sql_expression.strip():
        return True, "no sql expression to validate (type=count needs none)"

    member_refs = sorted(set(_MEMBER_REF_RE.findall(sql_expression)) - {"CUBE"})
    quoted_match = _QUOTED_COLUMN_RE.search(sql_expression)

    if member_refs and not quoted_match:
        known = _load_cube_member_names(cube_name)
        if known is None:
            return False, f"could not read model/cubes/{cube_name}.yml to verify referenced members"
        missing = [m for m in member_refs if m not in known]
        if missing:
            return False, (
                f"calculated measure references {missing} which "
                f"{'is' if len(missing) == 1 else 'are'} not (yet) a measure/dimension on "
                f"{cube_name} — add and approve {'it' if len(missing) == 1 else 'them'} first."
            )
        return True, f"calculated measure — verified {member_refs} all exist on {cube_name}"

    if not quoted_match:
        return False, f"could not find a quoted \"COLUMN_NAME\" in {sql_expression!r}"

    column = quoted_match.group(1)
    table = f'"REPORTING"."{cube_name.upper()}"'

    from warehouse.services.snowflake import SnowflakeClient

    try:
        SnowflakeClient().query(f'SELECT "{column}" FROM {table} LIMIT 0')
    except Exception as exc:
        return False, f'column "{column}" not found on {table} — {exc}'
    return True, f'column "{column}" verified on {table}'


def _yaml_scalar(value: str) -> str:
    """A properly YAML-escaped one-line string, without a full document dump
    (which would drop every comment in the target file — see
    schema_writer._splice_join_into_yaml's docstring for why this repo
    never re-dumps a whole cube file).

    Two yaml.safe_dump quirks corrected here, both confirmed by actually
    parsing spliced output back, not just eyeballing it:

    1. It appends a "...\\n" document-end marker even for a bare scalar
       (e.g. dumping "Bed Turnover Rate" produces "Bed Turnover
       Rate\\n...\\n") — stripped below, or it ends up as a literal stray
       "..." line in the cube file.
    2. It line-wraps any scalar past its default width=80 — harmless for a
       standalone document, but this string gets spliced in at a much
       deeper indentation than PyYAML assumed when it decided where to
       break the line, so the continuation line lands at the wrong column
       and produces INVALID YAML (confirmed: a real CASE-expression measure
       spliced this way made the whole cube file fail to parse). width=
       set far past any realistic SQL expression length disables wrapping.
    """
    dumped = yaml.safe_dump(value, default_flow_style=True, width=1_000_000).strip()
    if dumped.endswith("..."):
        dumped = dumped[: -len("...")].rstrip()
    return dumped


def write_pending_measure_to_yaml(pending: PendingCubeMeasure) -> tuple[bool, str]:
    """
    Splice pending's measure into model/cubes/<cube_name>.yml's `measures:`
    block, as the FIRST item right after the `measures:` line — every cube
    file already has at least one measure (count), so (unlike
    schema_writer's `joins: []` marker) there's no empty list to replace.

    Refuses (returns False, leaves the row `pending`) if the target file
    doesn't exist, or if `measures:` doesn't appear EXACTLY once — today
    every cube's `pre_aggregations:` block is an empty placeholder, so a
    stray nested `measures:` key can't collide, but this guard is what
    keeps that safe if that ever changes, rather than silently splicing
    into the wrong spot.
    """
    path = CUBES_DIR / f"{pending.cube_name}.yml"
    if not path.exists():
        return False, f"model/cubes/{pending.cube_name}.yml does not exist"

    try:
        text = path.read_text()
    except OSError as exc:
        return False, f"could not read {path.name}: {exc}"

    pattern = r"^(?P<indent>[ \t]*)measures:[ \t]*$"
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    if len(matches) != 1:
        return False, (
            f"expected exactly one 'measures:' key in {path.name}, found "
            f"{len(matches)} — refusing to guess; insert this measure by hand."
        )

    indent = matches[0].group("indent")
    item_indent = indent + "  "
    timestamp = datetime.now(timezone.utc).isoformat()

    lines = [f"{item_indent}- name: {pending.measure_name}"]
    if pending.sql_expression.strip():
        lines.append(f"{item_indent}  sql: {_yaml_scalar(pending.sql_expression)}")
    lines.append(f"{item_indent}  type: {pending.measure_type}")
    if pending.title.strip():
        lines.append(f"{item_indent}  title: {_yaml_scalar(pending.title)}")
    if pending.description.strip():
        lines.append(f"{item_indent}  description: {_yaml_scalar(pending.description)}")

    replacement_block = (
        f"{indent}measures:\n"
        f"{item_indent}# Added via Semantic Layer Configuration settings page on {timestamp}\n"
        f"{item_indent}# Requested by: {pending.requested_by}\n"
        + "\n".join(lines)
    )

    # Callable replacement (not a plain string) — re.sub interprets
    # backslash/group escapes in a string replacement, and these fields are
    # free-typed user text that could otherwise be misparsed.
    new_text = re.sub(pattern, lambda _m: replacement_block, text, count=1, flags=re.MULTILINE)

    try:
        path.write_text(new_text)
    except OSError as exc:
        # The most common real-world cause: the Cube container (running as
        # root inside Docker) touched this bind-mounted file at some point,
        # leaving it root-owned while the app runs as a regular user — an
        # ownership mismatch, not a code bug. Surface it as a normal
        # (ok=False, message) result like every other refusal above, rather
        # than letting an unhandled OSError 500 the request and roll back
        # the DB status change inside ApproveCubeMeasureView's transaction
        # without telling the reviewer why.
        return False, (
            f"could not write {path.name}: {exc}. This usually means the file "
            f"is owned by a different user (e.g. root, from the Cube container "
            f"writing through the bind mount) — check `ls -la` and `chown` it "
            f"to the user this app runs as."
        )

    logger.info(
        "write_pending_measure_to_yaml: wrote %s.%s into %s",
        pending.cube_name, pending.measure_name, path,
    )
    _notify_analytics_team_of_measure(pending)
    return True, f"wrote {pending.cube_name}.{pending.measure_name} to {path}"


def _notify_analytics_team_of_measure(pending: PendingCubeMeasure) -> None:
    analytics_email = getattr(settings, "ANALYTICS_TEAM_EMAIL", "analytics@example.com")
    from_email = getattr(settings, "DEFAULT_FROM_EMAIL", "noreply@example.com")

    verb = "edited" if pending.action == PendingCubeMeasure.ACTION_EDIT else "new"
    subject = f"[Agent Config] {verb.capitalize()} measure written: {pending.cube_name}.{pending.measure_name}"
    body = (
        f"A {verb} Cube measure was approved and written to the live schema.\n\n"
        f"Cube        : {pending.cube_name}\n"
        f"Measure     : {pending.measure_name} (type={pending.measure_type})\n"
        f"SQL         : {pending.sql_expression or '(none — count measure)'}\n"
        f"Requested by: {pending.requested_by}\n"
        f"Reviewed by : {pending.reviewed_by}\n\n"
        f"Written to model/cubes/{pending.cube_name}.yml — already live "
        f"(Cube runs in dev mode and hot-reloads schema files).\n"
    )
    try:
        send_mail(subject, body, from_email, [analytics_email], fail_silently=False)
    except Exception as exc:
        logger.error("write_pending_measure_to_yaml: failed to send notification email — %s", exc)


def get_cube_measure_definition(cube_name: str, measure_name: str) -> dict | None:
    """Current sql/type/title/description for one existing measure on
    cube_name — used to pre-fill the Edit Measure form and to confirm the
    measure actually exists before an edit gets staged. None if not found."""
    return next(
        (m for m in _parse_cube_measures(cube_name) if m["name"] == measure_name),
        None,
    )


def _line_indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def _find_named_item_span(text: str, block_key: str, item_name: str) -> tuple[int, int, str] | None:
    """
    Locate the [start, end) line range of one `- name: <item_name>` list
    item within block_key's ("measures"/"dimensions") block.

    A single regex can match where an item STARTS easily enough, but not
    reliably where it ENDS — that's either the next sibling item at the
    same indent, or the block itself ending (a shallower-indented key, or
    end of file) — so this scans line-by-line in two passes instead:
    first bound the block itself, then bound the one named item inside it.
    Returns None if the block or the named item isn't found, or the item
    name appears more than once (refuse rather than guess, same posture as
    every other splice in this module).
    """
    lines = text.splitlines(keepends=True)
    block_re = re.compile(rf"^(?P<indent>[ \t]*){re.escape(block_key)}:[ \t]*$")

    block_start = None
    block_indent = ""
    for i, line in enumerate(lines):
        m = block_re.match(line)
        if m:
            block_start = i
            block_indent = m.group("indent")
            break
    if block_start is None:
        return None

    block_end = len(lines)
    for j in range(block_start + 1, len(lines)):
        if lines[j].strip() == "":
            continue
        if len(_line_indent(lines[j])) <= len(block_indent):
            block_end = j
            break

    item_re = re.compile(rf"^(?P<indent>[ \t]*)- name: {re.escape(item_name)}[ \t]*$")
    item_matches = [
        (k, item_re.match(lines[k]).group("indent"))
        for k in range(block_start + 1, block_end)
        if item_re.match(lines[k])
    ]
    if len(item_matches) != 1:
        return None
    item_start, item_indent = item_matches[0]

    item_end = block_end
    for k in range(item_start + 1, block_end):
        if lines[k].strip() == "":
            continue
        if len(_line_indent(lines[k])) <= len(item_indent):
            item_end = k
            break

    return item_start, item_end, item_indent


def _replace_field_in_cube_yaml(
    cube_name: str,
    block_key: str,
    field_name: str,
    sql_expression: str,
    field_type: str,
    title: str,
    description: str,
) -> tuple[bool, str]:
    """
    Replace an EXISTING `- name: field_name` item's sql/type/title/
    description within block_key's block in model/cubes/<cube_name>.yml —
    the edit counterpart to write_pending_measure_to_yaml's insert-new-item
    splice. Refuses (changes nothing) if the file doesn't exist, or the
    named item doesn't appear in that block exactly once (see
    _find_named_item_span). Any comment lines directly above the old item
    are NOT preserved — they describe the old definition, which this
    replaces; this is a known, low-stakes limitation (a lost comment, not a
    correctness issue).
    """
    path = CUBES_DIR / f"{cube_name}.yml"
    if not path.exists():
        return False, f"model/cubes/{cube_name}.yml does not exist"

    try:
        text = path.read_text()
    except OSError as exc:
        return False, f"could not read {path.name}: {exc}"

    span = _find_named_item_span(text, block_key, field_name)
    if span is None:
        return False, (
            f"could not find exactly one '{field_name}' item in {cube_name}.yml's "
            f"'{block_key}:' block — refusing to guess."
        )
    start, end, item_indent = span

    lines = [f"{item_indent}- name: {field_name}"]
    if sql_expression.strip():
        lines.append(f"{item_indent}  sql: {_yaml_scalar(sql_expression)}")
    lines.append(f"{item_indent}  type: {field_type}")
    if title.strip():
        lines.append(f"{item_indent}  title: {_yaml_scalar(title)}")
    if description.strip():
        lines.append(f"{item_indent}  description: {_yaml_scalar(description)}")
    new_block = "\n".join(lines) + "\n"

    all_lines = text.splitlines(keepends=True)
    new_text = "".join(all_lines[:start]) + new_block + "".join(all_lines[end:])

    try:
        path.write_text(new_text)
    except OSError as exc:
        return False, (
            f"could not write {path.name}: {exc}. This usually means the file "
            f"is owned by a different user (e.g. root, from the Cube container "
            f"writing through the bind mount) — check `ls -la` and `chown` it "
            f"to the user this app runs as."
        )

    logger.info(
        "_replace_field_in_cube_yaml: replaced %s.%s in %s block of %s",
        cube_name, field_name, block_key, path,
    )
    return True, f"replaced {cube_name}.{field_name} in {path}"


def write_pending_measure_edit_to_yaml(pending: PendingCubeMeasure) -> tuple[bool, str]:
    """The edit counterpart to write_pending_measure_to_yaml — replaces an
    EXISTING measure's definition instead of inserting a new one."""
    ok, msg = _replace_field_in_cube_yaml(
        pending.cube_name, "measures", pending.measure_name,
        pending.sql_expression, pending.measure_type, pending.title, pending.description,
    )
    if not ok:
        return ok, msg
    _notify_analytics_team_of_measure(pending)
    return ok, msg


# ── Cube schema auto-sync from Snowflake's REPORTING schema ────────────────
#
# Everything above (validate_column_exists / write_pending_measure_to_yaml)
# is the human-in-the-loop path: one measure, typed by a person, checked
# before writing. This is the opposite: introspect Snowflake's
# INFORMATION_SCHEMA.COLUMNS directly for every REPORTING table an existing
# cube maps to, and auto-write every real column Cube doesn't expose yet —
# no PendingCubeMeasure staging, no approval click. Explicit user decision,
# matching the precedent schema_writer.py already set for auto-joins: no
# per-request review gate, but every run is logged and emails the analytics
# team. Motivated by a pattern repeated all session: rpt_bed_occupancy and
# fact_inpatient_admissions were both missing measures/dimensions that had
# been sitting in the warehouse the whole time — the catalog can only ever
# describe what Cube already exposes, so no amount of regenerating it would
# have caught these; the gap was always one layer down, in model/cubes/*.yml
# itself.

# Identifier-shaped columns are skipped outright even when numeric — a
# patient/visit ID is neither a useful measure (summing IDs is meaningless)
# nor a useful dimension in this context. Decided explicitly with the user.
_ID_LIKE_RE = re.compile(r"^ID$|_ID$|_KEY$|_CODE$", re.IGNORECASE)

# A numeric column whose NAME already reads as an average/rate/percentage
# gets type "avg" instead of "sum" — re-summing an already-averaged value
# across grouped rows double-counts it, regardless of its raw SQL type.
# Decided explicitly with the user (name-assisted on top of the data type).
_AVG_NAME_RE = re.compile(r"^AVG_|_RATE$|_PCT$|PERCENT", re.IGNORECASE)

# Snowflake's INFORMATION_SCHEMA.COLUMNS.DATA_TYPE values are a small fixed
# vocabulary (confirmed against real REPORTING tables, not guessed) — VARCHAR/
# CHAR/STRING all normalize to "TEXT", NUMBER/DECIMAL/INT/BIGINT all
# normalize to "NUMBER", so exact-match sets are enough; no prefix guessing.
_TIME_DATA_TYPES = {"DATE", "DATETIME", "TIME", "TIMESTAMP_NTZ", "TIMESTAMP_TZ", "TIMESTAMP_LTZ"}
_NUMERIC_DATA_TYPES = {"NUMBER", "FLOAT"}
_STRING_DATA_TYPES = {"TEXT", "BOOLEAN"}
# VARIANT/OBJECT/ARRAY/GEOGRAPHY/GEOMETRY/BINARY (semi-structured or
# otherwise not meaningfully aggregatable/groupable in Cube) fall through to
# the final `return None` below — skipped and reported, never guessed at.


def _classify_column(name: str, data_type: str) -> tuple[str, str] | None:
    """
    Decide whether a raw Snowflake column becomes a Cube measure, dimension,
    or time dimension, or should be skipped entirely.

    Returns (kind, cube_type) where kind is "measure" / "dimension" / "time",
    or None to skip. kind="measure" cube_type is "sum" or "avg"; kind=
    "dimension" cube_type is "string" or "boolean"; kind="time" cube_type is
    always "time".
    """
    upper_type = (data_type or "").strip().upper()
    upper_name = name.strip().upper()

    if _ID_LIKE_RE.search(upper_name):
        return None

    if upper_type in _TIME_DATA_TYPES:
        return ("time", "time")

    if upper_type in _NUMERIC_DATA_TYPES:
        if _AVG_NAME_RE.search(upper_name):
            return ("measure", "avg")
        return ("measure", "sum")

    if upper_type in _STRING_DATA_TYPES:
        return ("dimension", "boolean" if upper_type == "BOOLEAN" else "string")

    return None


def _splice_new_fields_into_cube_yaml(
    cube_name: str,
    new_measures: list[dict],
    new_dimensions: list[dict],
) -> tuple[bool, str]:
    """
    Batch-splice one or more new measures and/or dimensions into
    model/cubes/<cube_name>.yml in a single file write. Generalizes
    write_pending_measure_to_yaml's proven single-measure splice (same
    "insert as first item right after the block's key line" mechanics, same
    _yaml_scalar escaping, same width=1_000_000 fix for long expressions —
    all confirmed by actually parsing spliced output back, see that
    function's docstring for the bugs each one fixed) to cover BOTH the
    `measures:` and `dimensions:` blocks in one pass, since a schema-sync
    run typically finds both kinds of gap on the same cube at once.

    Each dict: {"name": str, "sql": str, "type": str}. Refuses (returns
    False, changes nothing) if a block being written into doesn't appear in
    the file EXACTLY once — same "refuse rather than guess" guard as the
    single-measure version, applied independently per block.
    """
    path = CUBES_DIR / f"{cube_name}.yml"
    if not path.exists():
        return False, f"model/cubes/{cube_name}.yml does not exist"
    if not new_measures and not new_dimensions:
        return True, "nothing to add"

    try:
        text = path.read_text()
    except OSError as exc:
        return False, f"could not read {path.name}: {exc}"

    timestamp = datetime.now(timezone.utc).isoformat()

    def _block_lines(fields: list[dict], item_indent: str) -> list[str]:
        lines = []
        for field in fields:
            lines.append(f"{item_indent}- name: {field['name']}")
            if field.get("sql"):
                lines.append(f"{item_indent}  sql: {_yaml_scalar(field['sql'])}")
            lines.append(f"{item_indent}  type: {field['type']}")
        return lines

    for block_key, fields in (("measures", new_measures), ("dimensions", new_dimensions)):
        if not fields:
            continue
        pattern = rf"^(?P<indent>[ \t]*){block_key}:[ \t]*$"
        matches = list(re.finditer(pattern, text, re.MULTILINE))
        if len(matches) != 1:
            return False, (
                f"expected exactly one '{block_key}:' key in {path.name}, found "
                f"{len(matches)} — refusing to guess; add these by hand: "
                f"{[f['name'] for f in fields]}"
            )
        indent = matches[0].group("indent")
        item_indent = indent + "  "
        replacement_block = (
            f"{indent}{block_key}:\n"
            f"{item_indent}# Added by sync_cube_schemas_from_snowflake on {timestamp}\n"
            + "\n".join(_block_lines(fields, item_indent))
        )
        # Callable replacement, not a plain string — re.sub interprets
        # backslash/group escapes in a string replacement (see
        # write_pending_measure_to_yaml for the same reasoning).
        text = re.sub(pattern, lambda _m, rb=replacement_block: rb, text, count=1, flags=re.MULTILINE)

    try:
        path.write_text(text)
    except OSError as exc:
        return False, (
            f"could not write {path.name}: {exc}. This usually means the file "
            f"is owned by a different user (e.g. root, from the Cube container "
            f"writing through the bind mount) — check `ls -la` and `chown` it "
            f"to the user this app runs as."
        )

    logger.info(
        "_splice_new_fields_into_cube_yaml: wrote %d measure(s), %d dimension(s) into %s",
        len(new_measures), len(new_dimensions), path,
    )
    return True, f"added {len(new_measures)} measure(s), {len(new_dimensions)} dimension(s) to {cube_name}"


def sync_cube_schemas_from_snowflake(dry_run: bool = False) -> dict:
    """
    For every existing model/cubes/*.yml, introspect its REPORTING table's
    real columns via Snowflake and add whatever measures/dimensions/time
    dimensions Cube doesn't expose yet — no PendingCubeMeasure staging, no
    approval click (explicit user decision; see module docstring above).

    Scope boundary: only touches cubes that ALREADY have a YAML file. A
    REPORTING table with no corresponding cube file yet is reported under
    "unmodeled_tables", never auto-created — matches the Propose/Approve
    flow's existing "no brand-new cube definitions" boundary.

    dry_run=True computes and returns exactly what WOULD be added, without
    writing anything or sending the notification email — a preview.

    Returns {"cubes_updated": [...], "fields_added": {cube: [...]},
             "skipped_unclassified": {cube: [...]}, "errors": {cube: msg},
             "dry_run": bool}.
    """
    from warehouse.services.snowflake import SnowflakeClient

    client = SnowflakeClient()
    cubes_updated: list[str] = []
    fields_added: dict[str, list[str]] = {}
    skipped_unclassified: dict[str, list[str]] = {}
    errors: dict[str, str] = {}

    for path in sorted(CUBES_DIR.glob("*.yml")):
        cube_name = path.stem
        try:
            existing = _load_cube_member_names(cube_name)
            if existing is None:
                errors[cube_name] = f"could not parse {path.name}"
                continue

            columns_df = client.get_columns("REPORTING", cube_name.upper())
            if columns_df.empty:
                # No such table/view in Snowflake at all — not this cube's
                # fault, just nothing to sync (e.g. a cube modeling something
                # other than a 1:1 REPORTING table, if one ever exists).
                continue

            new_measures: list[dict] = []
            new_dimensions: list[dict] = []
            unclassified: list[str] = []

            for _, row in columns_df.iterrows():
                col_name = str(row["COLUMN_NAME"])
                lower_name = col_name.lower()
                if lower_name in existing:
                    continue  # already declared under this name — never touch it

                classified = _classify_column(col_name, str(row["DATA_TYPE"]))
                if classified is None:
                    if not _ID_LIKE_RE.search(col_name.upper()):
                        unclassified.append(col_name)  # a real gap worth reporting, not an intentional exclusion
                    continue

                kind, cube_type = classified
                field = {
                    "name": lower_name,
                    "sql": f'{{CUBE}}."{col_name.upper()}"',
                    "type": cube_type,
                }
                if kind == "measure":
                    new_measures.append(field)
                elif kind == "dimension":
                    new_dimensions.append(field)
                else:  # "time"
                    new_dimensions.append(field)

            if unclassified:
                skipped_unclassified[cube_name] = unclassified

            if not new_measures and not new_dimensions:
                continue

            added_names = [f["name"] for f in new_measures + new_dimensions]
            if dry_run:
                fields_added[cube_name] = added_names
                cubes_updated.append(cube_name)
                continue

            ok, msg = _splice_new_fields_into_cube_yaml(cube_name, new_measures, new_dimensions)
            if not ok:
                errors[cube_name] = msg
                continue

            fields_added[cube_name] = added_names
            cubes_updated.append(cube_name)

        except Exception as exc:
            # One cube's Snowflake hiccup must not abort the other ~80.
            logger.exception("sync_cube_schemas_from_snowflake: failed on %s", cube_name)
            errors[cube_name] = str(exc)

    summary = {
        "cubes_updated": cubes_updated,
        "fields_added": fields_added,
        "skipped_unclassified": skipped_unclassified,
        "errors": errors,
        "dry_run": dry_run,
    }
    logger.info(
        "sync_cube_schemas_from_snowflake: dry_run=%s cubes_updated=%d errors=%d",
        dry_run, len(cubes_updated), len(errors),
    )
    if not dry_run and cubes_updated:
        _notify_analytics_team_of_schema_sync(summary)
    return summary


def _notify_analytics_team_of_schema_sync(summary: dict) -> None:
    analytics_email = getattr(settings, "ANALYTICS_TEAM_EMAIL", "analytics@example.com")
    from_email = getattr(settings, "DEFAULT_FROM_EMAIL", "noreply@example.com")

    lines = [
        f"Auto-synced {len(summary['cubes_updated'])} cube(s) from Snowflake's REPORTING schema.\n",
    ]
    for cube in summary["cubes_updated"]:
        lines.append(f"  {cube}: added {', '.join(summary['fields_added'][cube])}")
    if summary["skipped_unclassified"]:
        lines.append("\nColumns found but not classified (unrecognized/semi-structured type — review manually):")
        for cube, cols in summary["skipped_unclassified"].items():
            lines.append(f"  {cube}: {', '.join(cols)}")
    if summary["errors"]:
        lines.append("\nCubes that could not be synced:")
        for cube, err in summary["errors"].items():
            lines.append(f"  {cube}: {err}")

    subject = f"[Semantic Layer Config] Cube schema sync — {len(summary['cubes_updated'])} cube(s) updated"
    try:
        send_mail(subject, "\n".join(lines), from_email, [analytics_email], fail_silently=False)
    except Exception as exc:
        logger.error("sync_cube_schemas_from_snowflake: failed to send notification email — %s", exc)
