"""Field rules — the transform grammar shared by the mapper, the review UI,
the exporter and the sync engine. It is the grammar the existing
``from_json_mappings_to_snowflake`` loader understands, so exported
mappings.json files run there unchanged.

kinds
  copied                       value as is
  worked out                   {"steps": [{"op": "lowercase"}, …]}
  codes converted              {"map": {"0": 1, "1": 0}, "default": …}
  looked up in another table   {"table": "departments", "key": "id", "value": "name"}
                               = departments.name of the row whose id = this value
  one column divided           {"part": 1 | "rest", "delimiter": " "}
  constant                     {"value": …}  (connect-studio only; not in the V3 loader)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

KINDS = ["copied", "worked out", "codes converted", "looked up in another table", "one column divided", "constant"]
LOADER_KINDS = set(KINDS) - {"constant"}


def slug(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


STEPS = {
    "lowercase": lambda s: s.lower(),
    "uppercase": lambda s: s.upper(),
    "trim": lambda s: s.strip(),
    "slug": slug,
    "first_word": lambda s: (s.split() or [""])[0],
    "rest_words": lambda s: " ".join(s.split()[1:]),
    "title_case": lambda s: s.title(),
}


def validate(field: dict, source_tables: dict[str, dict]) -> str | None:
    """Why a suggested field is unusable (None = fine). ``source_tables`` = {key: table}."""
    kind = field.get("kind")
    rule = field.get("rule") or {}
    if kind not in KINDS:
        return f"unknown kind {kind!r}"
    if kind != "constant":
        table = source_tables.get(field.get("source_table") or "")
        if not table:
            return "source table not in the snapshot"
        cols = {c["name"] for c in table["columns"]}
        if field.get("source") not in cols:
            return f"source column {field.get('source')!r} does not exist"
    if kind == "worked out":
        steps = rule.get("steps") or []
        if not steps or any(str(s.get("op", "")).lower() not in STEPS for s in steps if isinstance(s, dict)):
            return "worked out needs steps from: " + ", ".join(STEPS)
    elif kind == "codes converted":
        if not isinstance(rule.get("map"), dict) or not rule["map"]:
            return "codes converted needs a non-empty map"
    elif kind == "looked up in another table":
        other = source_tables.get(rule.get("table") or "")
        if not other:
            return f"lookup table {rule.get('table')!r} does not exist"
        ocols = {c["name"] for c in other["columns"]}
        if rule.get("key") not in ocols or rule.get("value") not in ocols:
            return f"lookup columns {rule.get('key')!r}/{rule.get('value')!r} not in {rule['table']}"
    elif kind == "one column divided":
        part = rule.get("part")
        if not (part == "rest" or (isinstance(part, int) and part >= 1) or (isinstance(part, str) and part.isdigit())):
            return "one column divided needs part = 1, 2, … or \"rest\""
    elif kind == "constant":
        if "value" not in rule:
            return "constant needs a value"
    return None


def normalize(field: dict) -> dict:
    """Coerce an AI/human field into the canonical shape."""
    rule = field.get("rule")
    if isinstance(rule, str):
        import json
        try:
            rule = json.loads(rule) if rule.strip() else {}
        except ValueError:
            rule = {}
    rule = dict(rule or {})
    kind = field.get("kind") or "copied"
    # tolerate the loader's flat lookup spelling and a nested {"lookup": {...}}
    if kind == "looked up in another table" and "lookup" in rule and isinstance(rule["lookup"], dict):
        lk = rule.pop("lookup")
        rule.update({"table": lk.get("table"), "key": lk.get("match") or lk.get("key"), "value": lk.get("value")})
    if kind == "one column divided" and isinstance(rule.get("part"), str) and rule["part"].isdigit():
        rule["part"] = int(rule["part"])
    if kind == "worked out" and isinstance(rule.get("steps"), list):
        rule["steps"] = [s if isinstance(s, dict) else {"op": str(s)} for s in rule["steps"]]
    if kind == "copied":
        rule = {}
    rule.pop("column", None)
    field["rule"] = rule
    field["kind"] = kind
    transform = str(field.get("transform") or "").strip().lower()
    field["transform"] = transform if transform in STEPS else ""
    return field


def apply(field: dict, value: Any, lookups: dict | None = None) -> Any:
    """Run one field's rule on one value (mirrors the loader's _apply_rule)."""
    kind = field.get("kind")
    rule = field.get("rule") or {}
    if kind == "constant":
        return rule.get("value")
    if value is None:
        return None
    if kind == "looked up in another table":
        table = (lookups or {}).get((rule.get("table"), rule.get("key"), rule.get("value")), {})
        value = table.get(str(value))
    if "part" in rule and isinstance(value, str):
        pieces = value.split(rule.get("delimiter") or None)
        part = rule["part"]
        if part == "rest":
            value = (rule.get("delimiter") or " ").join(pieces[1:]) or None
        else:
            value = pieces[int(part) - 1] if 0 < int(part) <= len(pieces) else None
    if "map" in rule and value is not None:
        table = {str(k).lower(): v for k, v in rule["map"].items()}
        value = table.get(str(value).strip().lower(), rule.get("default", value))
    steps = list(rule.get("steps") or [])
    if field.get("transform"):
        steps.append({"op": field["transform"]})
    for step in steps:
        op = STEPS.get(str(step.get("op", "")).lower())
        if op and isinstance(value, str):
            value = op(value)
    return value


def describe(field: dict) -> str:
    """A one-line human summary of a field's rule."""
    kind = field.get("kind")
    rule = field.get("rule") or {}
    src = field.get("source") or ""
    if kind == "copied":
        return src
    if kind == "worked out":
        return f"{src} → " + " → ".join(s.get("op", "?") for s in rule.get("steps", []))
    if kind == "codes converted":
        pairs = ", ".join(f"{k}→{v}" for k, v in list(rule.get("map", {}).items())[:4])
        return f"{src}: {pairs}{' …' if len(rule.get('map', {})) > 4 else ''}"
    if kind == "looked up in another table":
        return f"{rule.get('table')}.{rule.get('value')} where {rule.get('key')} = {src}"
    if kind == "one column divided":
        return f"{src} split on {rule.get('delimiter') or 'space'!r}, part {rule.get('part')}"
    if kind == "constant":
        return f"= {rule.get('value')!r}"
    return src
