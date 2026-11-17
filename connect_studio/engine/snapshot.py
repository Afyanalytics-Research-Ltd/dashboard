"""Schema snapshots: one normalized shape for every source and target.

A snapshot is plain JSON, so it can be stored, diffed, uploaded and handed to
the mapper without caring where it came from::

    {
      "format": "connect-studio.snapshot/1",
      "dialect": "postgresql", "database": "hmis",
      "extracted_at": "...", "extractor": "dbatlas 0.1.0",
      "tables": [{
        "key": "patients", "schema": "public", "name": "patients", "kind": "table",
        "rows": 1200, "comment": null, "model": null, "service": null,
        "primary_key": ["id"],
        "columns": [{
          "name": "othernames", "type": "VARCHAR(255)", "family": "text",
          "nullable": true, "pk": false, "unique": false, "default": null, "comment": null,
          "allowed_values": [],
          "profile": {"null_pct": 0.4, "distinct": 1180, "min": null, "max": null,
                      "samples": ["Stephen Ombaka", ...], "top": [{"v": "Jane", "n": 3}], "pii": []}
        }]
      }],
      "relationships": [{"from": "visits", "from_columns": ["patient_id"],
                         "to": "patients", "to_columns": ["id"],
                         "confidence": 1.0, "kind": "foreign_key"}],
      "stats": {"tables": 4, "columns": 18, "relationships": 3, "rows": 5000}
    }

Inputs handled here:
  * a dbatlas ``Database`` graph (every live connector ends up as one)
  * an uploaded schema document: this snapshot format, dbatlas JSON, a
    mappings.json (its target side), JSON Schema, ``{"tables": [...]}`` /
    ``{"tables": {...}}`` / ``{table: {column: type}}`` shapes, or SQL DDL
    in any dialect sqlglot reads.
Relationships the document does not declare are inferred with dbatlas.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Iterable

FORMAT = "connect-studio.snapshot/1"
SAMPLE_LIMIT = 8
TOP_LIMIT = 15
VALUE_MAX_LEN = 80
DEFAULT_SCHEMAS = {"main", "public", "default", "dbo", ""}


class SchemaDocumentError(ValueError):
    """An uploaded schema document could not be understood."""


# ─── types ────────────────────────────────────────────────────────────────

_FAMILIES = [
    ("boolean", r"\b(bool|boolean|bit)\b"),
    ("uuid", r"\b(uuid|uniqueidentifier|guid)\b"),
    ("json", r"\b(json|jsonb|variant|object|struct|map|array|super)\b"),
    ("datetime", r"(timestamp|datetime|date-time|timestamptz|smalldatetime)"),
    ("date", r"\bdate\b"),
    ("time", r"\btime\b"),
    ("decimal", r"(decimal|numeric|money|real|float|double|number\(\d+,\s*[1-9]|\bnumber\b(?!\(\d+(,\s*0)?\)))"),
    ("integer", r"(int|serial|long|short|byte|integer|number\(\d+(,\s*0)?\))"),
    ("binary", r"(blob|binary|bytea|varbinary|image)"),
    ("text", r"(char|text|string|clob|enum|set|citext|varchar|nvarchar|name)"),
]


def type_family(data_type: str | None) -> str:
    """Coarse family for a SQL / JSON type name — what type compatibility and icons use."""
    t = str(data_type or "").strip().lower()
    if not t:
        return "unknown"
    if t in ("tinyint(1)",):
        return "boolean"
    for family, pattern in _FAMILIES:
        if re.search(pattern, t):
            return family
    return "other"


COMPATIBLE = {
    "integer": {"integer", "decimal", "text", "boolean"},
    "decimal": {"decimal", "integer", "text"},
    "text": {"text", "unknown", "other", "uuid", "json", "date", "datetime", "integer", "decimal", "boolean", "time"},
    "boolean": {"boolean", "integer", "text"},
    "date": {"date", "datetime", "text"},
    "datetime": {"datetime", "date", "text"},
    "time": {"time", "text", "datetime"},
    "uuid": {"uuid", "text"},
    "json": {"json", "text"},
    "binary": {"binary", "text"},
}


def compatible(source_family: str, target_family: str) -> bool:
    if "unknown" in (source_family, target_family) or "other" in (source_family, target_family):
        return True
    return source_family in COMPATIBLE.get(target_family, {target_family})


# ─── helpers ──────────────────────────────────────────────────────────────

def _clip(value: Any) -> str:
    text = str(value)
    return text if len(text) <= VALUE_MAX_LEN else text[: VALUE_MAX_LEN - 1] + "…"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def empty_snapshot(**meta) -> dict:
    return {"format": FORMAT, "dialect": meta.get("dialect", ""), "database": meta.get("database", ""),
            "extracted_at": now_iso(), "extractor": meta.get("extractor", "connect-studio"),
            "tables": [], "relationships": [], "warnings": [], "stats": {}}


def table_index(snapshot: dict) -> dict[str, dict]:
    return {t["key"]: t for t in snapshot.get("tables", [])}


def column_index(table: dict) -> dict[str, dict]:
    return {c["name"]: c for c in table.get("columns", [])}


def _assign_keys(tables: list[dict]) -> None:
    """Plain table name as the key unless two schemas share it (then schema.name)."""
    counts: dict[str, int] = {}
    for t in tables:
        counts[t["name"].lower()] = counts.get(t["name"].lower(), 0) + 1
    for t in tables:
        dup = counts[t["name"].lower()] > 1
        schema = t.get("schema") or ""
        t["key"] = f"{schema}.{t['name']}" if dup and schema else t["name"]


def finalize(snapshot: dict, *, infer: bool = True) -> dict:
    """Keys, primary keys, inferred relationships, stats and a fingerprint."""
    tables = snapshot.setdefault("tables", [])
    for t in tables:
        t.setdefault("schema", "")
        t.setdefault("kind", "table")
        t.setdefault("comment", None)
        t.setdefault("model", None)
        t.setdefault("service", None)
        t.setdefault("rows", None)
        for i, c in enumerate(t.setdefault("columns", []), start=1):
            c.setdefault("type", "")
            c["family"] = c.get("family") or type_family(c["type"])
            c.setdefault("nullable", True)
            c.setdefault("pk", False)
            c.setdefault("unique", False)
            c.setdefault("default", None)
            c.setdefault("comment", None)
            c.setdefault("allowed_values", [])
            c.setdefault("position", i)
            c.setdefault("profile", None)
        t["primary_key"] = t.get("primary_key") or [c["name"] for c in t["columns"] if c.get("pk")]
        for c in t["columns"]:
            if c["name"] in t["primary_key"]:
                c["pk"] = True
    if not all(t.get("key") for t in tables):
        _assign_keys(tables)
    tables.sort(key=lambda t: (t.get("schema") or "", t["name"].lower()))

    snapshot.setdefault("relationships", [])
    _resolve_relationship_keys(snapshot)
    rels = snapshot["relationships"]  # (re-read: resolving replaces the list)
    if infer:
        try:
            inferred = infer_relationships(snapshot)
        except Exception as e:  # inference is a bonus, never fatal
            snapshot.setdefault("warnings", []).append(f"relationship inference skipped: {e}")
            inferred = []
        seen = {(r["from"], tuple(r["from_columns"])) for r in rels}
        for r in inferred:
            if (r["from"], tuple(r["from_columns"])) not in seen:
                rels.append(r)
                seen.add((r["from"], tuple(r["from_columns"])))
    for t in tables:
        fk_cols = {}
        for r in rels:
            if r["from"] == t["key"]:
                for col, tcol in zip(r["from_columns"], r["to_columns"]):
                    fk_cols[col] = {"table": r["to"], "column": tcol, "confidence": r["confidence"], "kind": r["kind"]}
        for c in t["columns"]:
            c["references"] = fk_cols.get(c["name"]) or c.get("references") or None

    snapshot["stats"] = {
        "tables": len(tables),
        "columns": sum(len(t["columns"]) for t in tables),
        "relationships": len(rels),
        "declared_relationships": sum(1 for r in rels if r["kind"] != "inferred"),
        "rows": sum(int(t["rows"] or 0) for t in tables),
        "profiled": any(c.get("profile") for t in tables for c in t["columns"]),
    }
    snapshot["fingerprint"] = fingerprint(snapshot)
    snapshot["format"] = FORMAT
    return snapshot


def _resolve_relationship_keys(snapshot: dict) -> None:
    """Relationships may name tables as schema.name or bare names; map both to keys."""
    by_key = table_index(snapshot)
    by_name: dict[str, str] = {}
    for t in snapshot["tables"]:
        by_name.setdefault(t["name"].lower(), t["key"])
        by_name[f"{t.get('schema') or ''}.{t['name']}".lower()] = t["key"]
    keep = []
    for r in snapshot.get("relationships", []):
        src = r["from"] if r["from"] in by_key else by_name.get(str(r["from"]).lower())
        dst = r["to"] if r["to"] in by_key else by_name.get(str(r["to"]).lower())
        if not src or not dst:
            continue
        r["from"], r["to"] = src, dst
        r.setdefault("confidence", 1.0)
        r.setdefault("kind", "foreign_key")
        if not r.get("to_columns"):
            r["to_columns"] = by_key[dst]["primary_key"][:len(r["from_columns"])] or ["id"]
        keep.append(r)
    snapshot["relationships"] = keep


def fingerprint(snapshot: dict) -> str:
    """Structure-only hash: same tables/columns/types ⇒ same fingerprint."""
    shape = [(t["key"], [(c["name"], c["type"]) for c in t["columns"]]) for t in snapshot.get("tables", [])]
    return hashlib.sha256(json.dumps(shape, sort_keys=True).encode()).hexdigest()[:16]


# ─── dbatlas ↔ snapshot ───────────────────────────────────────────────────

def from_dbatlas(database, *, dialect: str | None = None, extractor: str | None = None) -> dict:
    """Normalize a dbatlas ``Database`` graph (introspected + profiled + analyzed)."""
    try:
        from dbatlas import __version__ as dbatlas_version
    except Exception:  # pragma: no cover
        dbatlas_version = "?"
    snap = empty_snapshot(dialect=dialect or database.dialect, database=database.name,
                          extractor=extractor or f"dbatlas {dbatlas_version}")
    for table in database.tables:
        profile = table.profile
        col_profiles = {p.column_name: p for p in (profile.columns if profile else ())}
        columns = []
        for col in table.columns:
            p = col_profiles.get(col.name)
            columns.append({
                "name": col.name, "type": col.data_type, "nullable": col.nullable,
                "pk": col.primary_key, "unique": col.unique, "default": col.default,
                "comment": col.comment, "position": col.ordinal_position,
                "profile": _profile(p) if p else None,
            })
        rows = profile.row_count if profile else table.estimated_row_count
        snap["tables"].append({
            "schema": table.schema_name, "name": table.name, "kind": table.type,
            "rows": rows, "comment": table.comment, "columns": columns,
            "primary_key": list(table.primary_key_columns),
        })
    names = {(t.schema_name, t.name): t for t in database.tables}
    for r in database.relationships:
        snap["relationships"].append({
            "from": f"{r.source_schema}.{r.source_table}" if (r.source_schema, r.source_table) in names else r.source_table,
            "from_columns": list(r.source_columns),
            "to": f"{r.target_schema}.{r.target_table}" if (r.target_schema, r.target_table) in names else r.target_table,
            "to_columns": list(r.target_columns),
            "confidence": round(float(r.confidence), 3),
            "kind": "foreign_key" if r.discovered_by in ("constraint", "foreign_key") else "inferred",
            "description": r.description,
        })
    _assign_keys(snap["tables"])
    # dbatlas already ran its inference, so finalize() needn't repeat it
    return finalize(snap, infer=False)


def _profile(p) -> dict:
    out = {
        "rows": p.row_count,
        "null_pct": round(p.null_percent, 2),
        "distinct": p.distinct_count,
        "min": _clip(p.min_value) if p.min_value is not None else None,
        "max": _clip(p.max_value) if p.max_value is not None else None,
        "avg": round(p.average, 4) if p.average is not None else None,
        "samples": [_clip(v) for v in list(p.sample_values)[:SAMPLE_LIMIT]],
        "top": [{"v": _clip(v.value), "n": v.count} for v in list(p.most_common_values)[:TOP_LIMIT]],
        "pii": sorted({f.kind for f in p.pii}),
    }
    return out


def to_dbatlas(snapshot: dict):
    """A dbatlas Database graph for a snapshot — so dbatlas inference runs on uploaded schemas too."""
    from dbatlas.models import Column, Database, Schema, Table

    by_schema: dict[str, list] = {}
    for t in snapshot.get("tables", []):
        declared = set(t.get("primary_key") or []) | {c["name"] for c in t.get("columns", []) if c.get("pk")}
        # schema docs often omit keys: a lone "id" column is the key for inference purposes
        implied = set() if declared else {c["name"] for c in t.get("columns", []) if c["name"].lower() == "id"}
        cols = tuple(
            Column(name=c["name"], data_type=c.get("type") or "unknown", ordinal_position=i,
                   nullable=bool(c.get("nullable", True)), primary_key=c["name"] in (declared or implied),
                   unique=bool(c.get("unique")))
            for i, c in enumerate(t.get("columns", []), start=1))
        schema = t.get("schema") or "main"
        by_schema.setdefault(schema, []).append(Table(name=t["name"], schema_name=schema, columns=cols))
    return Database(name=snapshot.get("database") or "schema", dialect=snapshot.get("dialect") or "generic",
                    schemas=tuple(Schema(name=s, tables=tuple(ts)) for s, ts in by_schema.items()))


def infer_relationships(snapshot: dict) -> list[dict]:
    """dbatlas' deterministic relationship inference, on any snapshot."""
    from dbatlas.inference.relationships import infer_relationships as dbatlas_infer

    db = to_dbatlas(snapshot)
    keys = {((t.get("schema") or "main"), t["name"]): t["key"] for t in snapshot["tables"]}
    out = []
    for r in dbatlas_infer(db):
        src = keys.get((r.source_schema, r.source_table))
        dst = keys.get((r.target_schema, r.target_table))
        if src and dst:
            out.append({"from": src, "from_columns": list(r.source_columns), "to": dst,
                        "to_columns": list(r.target_columns), "confidence": round(float(r.confidence), 3),
                        "kind": "inferred", "description": r.description})
    return out


# ─── uploaded schema documents ────────────────────────────────────────────

def parse_schema_document(content: bytes | str, filename: str = "schema.json", *, dialect: str | None = None) -> dict:
    """Any supported schema document → finalized snapshot. Raises SchemaDocumentError."""
    text = content.decode("utf-8-sig", errors="replace") if isinstance(content, bytes) else content
    name = (filename or "").lower()
    stripped = text.strip()
    if not stripped:
        raise SchemaDocumentError("The file is empty.")
    if name.endswith((".sql", ".ddl")) or (not stripped.startswith(("{", "[")) and re.search(r"create\s+table", stripped, re.I)):
        snap = parse_ddl(text, dialect=dialect)
    else:
        try:
            doc = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise SchemaDocumentError(f"Not valid JSON: {e.msg} (line {e.lineno}, column {e.colno}).") from None
        snap = parse_schema_json(doc)
    if not snap["tables"]:
        raise SchemaDocumentError("No tables found in the document.")
    return finalize(snap)


def detect_document_kind(doc: Any) -> str:
    if isinstance(doc, dict):
        if doc.get("format") == FORMAT:
            return "snapshot"
        if isinstance(doc.get("schemas"), list) and "dialect" in doc:
            return "dbatlas"
        if isinstance(doc.get("fields"), list) and isinstance(doc.get("tables"), list) and "connection" in doc:
            return "mappings"
        if any(k in doc for k in ("$defs", "definitions")) or (doc.get("type") == "object" and isinstance(doc.get("properties"), dict)):
            return "json-schema"
        if "tables" in doc:
            return "tables"
        return "table-map"
    if isinstance(doc, list):
        return "tables"
    return "unknown"


def parse_schema_json(doc: Any) -> dict:
    kind = detect_document_kind(doc)
    if kind == "snapshot":
        snap = {**doc, "tables": [dict(t, columns=[dict(c) for c in t.get("columns", [])]) for t in doc.get("tables", [])],
                "relationships": [dict(r) for r in doc.get("relationships", [])]}
        return snap
    if kind == "dbatlas":
        from dbatlas.models import Database
        try:
            database = Database.model_validate(doc)
        except Exception as e:
            raise SchemaDocumentError(f"Looks like dbatlas JSON but does not validate: {e}") from None
        snap = from_dbatlas(database)
        snap["extractor"] = "uploaded dbatlas.json"
        return snap
    snap = empty_snapshot(extractor=f"uploaded schema ({kind})")
    if kind == "mappings":
        _from_mappings(doc, snap)
    elif kind == "json-schema":
        _from_json_schema(doc, snap)
    elif kind == "tables":
        tables = doc.get("tables") if isinstance(doc, dict) else doc
        if isinstance(tables, dict):
            for tname, spec in tables.items():
                snap["tables"].append(_table_from_spec(spec, tname, snap))
        elif isinstance(tables, list):
            for spec in tables:
                snap["tables"].append(_table_from_spec(spec, None, snap))
        else:
            raise SchemaDocumentError('"tables" must be a list or an object.')
        for r in (doc.get("relationships") or doc.get("foreign_keys") or []) if isinstance(doc, dict) else []:
            rel = _relationship_from_spec(r)
            if rel:
                snap["relationships"].append(rel)
        if isinstance(doc, dict):
            snap["database"] = str(doc.get("name") or doc.get("database") or "")
            snap["dialect"] = str(doc.get("dialect") or "")
    elif kind == "table-map":
        for tname, spec in doc.items():
            if isinstance(spec, (dict, list)):
                snap["tables"].append(_table_from_spec(spec, tname, snap))
    else:
        raise SchemaDocumentError("Unrecognized schema document — expected a JSON object or list of tables.")
    snap["tables"] = [t for t in snap["tables"] if t and t.get("name")]
    return snap


def _from_mappings(doc: dict, snap: dict) -> None:
    """The target side of an existing mappings.json — reuse a known-good target schema."""
    tables: dict[str, dict] = {}
    for t in doc.get("tables", []):
        name = t.get("v3 table") or t.get("v3 model", "").split("\\")[-1]
        if not name:
            continue
        tables.setdefault(name, {"schema": t.get("v3 service") or "", "name": name, "model": t.get("v3 model"),
                                 "service": t.get("v3 service"), "columns": [], "_seen": set()})
    for f in doc.get("fields", []):
        name = f.get("v3_table") or (f.get("model") or "").split("\\")[-1]
        if not name or not f.get("field"):
            continue
        t = tables.setdefault(name, {"schema": "", "name": name, "model": f.get("model"), "service": None,
                                     "columns": [], "_seen": set()})
        if f["field"] not in t["_seen"]:
            t["_seen"].add(f["field"])
            t["columns"].append({"name": f["field"], "type": "", "pk": f["field"] == "id"})
    for t in tables.values():
        t.pop("_seen", None)
        snap["tables"].append(t)
    snap["database"] = str((doc.get("connection") or {}).get("name") or "")
    snap["extractor"] = "uploaded mappings.json (target side)"


def _from_json_schema(doc: dict, snap: dict) -> None:
    defs = doc.get("$defs") or doc.get("definitions") or {}
    if not defs and isinstance(doc.get("properties"), dict):
        # a root object whose properties are tables (arrays/objects of rows)
        for tname, prop in doc["properties"].items():
            item = prop.get("items") if prop.get("type") == "array" else prop
            if isinstance(item, dict) and isinstance(item.get("properties"), dict):
                defs[tname] = item
        if not defs:
            defs = {doc.get("title") or "record": doc}
    for tname, spec in defs.items():
        if not isinstance(spec, dict) or not isinstance(spec.get("properties"), dict):
            continue
        required = set(spec.get("required") or [])
        cols = []
        for cname, prop in spec["properties"].items():
            ptype = prop.get("type", "")
            nullable = cname not in required
            if isinstance(ptype, list):
                nullable = nullable or "null" in ptype
                ptype = next((p for p in ptype if p != "null"), "string")
            fmt = prop.get("format")
            type_name = {"date-time": "timestamp", "date": "date", "time": "time", "uuid": "uuid"}.get(fmt, ptype or "string")
            if ptype == "integer":
                type_name = "integer"
            elif ptype == "number":
                type_name = "decimal"
            elif ptype == "boolean":
                type_name = "boolean"
            elif ptype in ("object", "array"):
                type_name = "json"
            cols.append({"name": cname, "type": type_name, "nullable": nullable, "comment": prop.get("description"),
                         "allowed_values": [_clip(v) for v in (prop.get("enum") or [])][:50],
                         "pk": bool(prop.get("primaryKey") or prop.get("x-primary-key"))})
        snap["tables"].append({"name": spec.get("title") or tname, "schema": "", "comment": spec.get("description"),
                               "columns": cols})


_TYPE_KEYS = ("type", "data_type", "datatype", "dtype", "sql_type", "column_type")


def _table_from_spec(spec: Any, name: str | None, snap: dict) -> dict:
    if isinstance(spec, list):
        spec = {"columns": spec}
    if not isinstance(spec, dict):
        raise SchemaDocumentError(f"Table {name or '?'} must be an object or a list of columns.")
    if name and spec and all(isinstance(v, str) for v in spec.values()):
        # {"patients": {"id": "int", "name": "text"}} — a bare column map ("name" is a column here)
        spec = {"columns": dict(spec)}
    tname = str(spec.get("name") or spec.get("table") or spec.get("table_name") or spec.get("title") or name or "").strip()
    if not tname:
        raise SchemaDocumentError("A table is missing its name.")
    raw_cols = spec.get("columns") or spec.get("fields") or spec.get("properties") or {}
    if not (isinstance(raw_cols, (dict, list))) or (not raw_cols and not any(k in spec for k in ("columns", "fields", "properties"))):
        # {"patients": {"id": "int", "name": "text"}} — the spec itself is the column map
        raw_cols = {k: v for k, v in spec.items() if k not in ("name", "schema", "description", "comment", "model", "service")}
    cols = []
    items = raw_cols.items() if isinstance(raw_cols, dict) else ((None, c) for c in raw_cols)
    for cname, c in items:
        col = _column_from_spec(c, cname)
        if col:
            cols.append(col)
    pk = spec.get("primary_key") or spec.get("primaryKey") or spec.get("pk") or []
    if isinstance(pk, str):
        pk = [pk]
    for fk in spec.get("foreign_keys") or spec.get("foreignKeys") or spec.get("relationships") or []:
        rel = _relationship_from_spec(fk, default_from=tname)
        if rel:
            snap["relationships"].append(rel)
    for col in cols:
        ref = col.pop("_ref", None)
        if ref:
            snap["relationships"].append({"from": tname, "from_columns": [col["name"]], "to": ref[0],
                                          "to_columns": [ref[1]] if ref[1] else [], "confidence": 1.0,
                                          "kind": "foreign_key"})
    return {"schema": str(spec.get("schema") or ""), "name": tname, "comment": spec.get("description") or spec.get("comment"),
            "model": spec.get("model"), "service": spec.get("service"), "rows": spec.get("rows"),
            "columns": cols, "primary_key": list(pk)}


def _parse_ref(ref: Any) -> tuple[str, str | None] | None:
    if not ref:
        return None
    if isinstance(ref, dict):
        table = ref.get("table") or ref.get("ref_table") or ref.get("references")
        column = ref.get("column") or ref.get("ref_column") or (ref.get("columns") or [None])[0]
        return (str(table), column) if table else None
    text = str(ref).strip()
    m = re.match(r"^([\w.\"`\[\]]+?)\s*\(\s*([\w\"`\[\]]+)\s*\)$", text)
    if m:
        return m.group(1).strip('"`[]'), m.group(2).strip('"`[]')
    if "." in text:
        table, column = text.rsplit(".", 1)
        return table, column
    return text, None


def _column_from_spec(c: Any, cname: str | None) -> dict | None:
    if isinstance(c, str):
        if cname is None:
            # "id" or "id int" or "id:int"
            parts = re.split(r"[:\s]+", c.strip(), maxsplit=1)
            return {"name": parts[0], "type": parts[1] if len(parts) > 1 else ""}
        return {"name": cname, "type": c}
    if not isinstance(c, dict):
        return {"name": cname, "type": ""} if cname else None
    name = str(c.get("name") or c.get("column") or c.get("column_name") or cname or "").strip()
    if not name:
        return None
    ctype = next((c[k] for k in _TYPE_KEYS if c.get(k)), "")
    if isinstance(ctype, list):  # JSON-schema style ["string", "null"]
        ctype = next((p for p in ctype if p != "null"), "")
    nullable = c.get("nullable")
    if nullable is None and "null" in c:
        nullable = bool(c["null"])
    if nullable is None and "required" in c:
        nullable = not bool(c["required"])
    if nullable is None and "not_null" in c:
        nullable = not bool(c["not_null"])
    allowed = c.get("allowed_values") or c.get("enum") or c.get("values") or c.get("accepted_values") or []
    col = {
        "name": name, "type": str(ctype),
        "nullable": True if nullable is None else bool(nullable),
        "pk": bool(c.get("primary_key") or c.get("pk") or c.get("primary") or c.get("primaryKey")),
        "unique": bool(c.get("unique")),
        "default": c.get("default"),
        "comment": c.get("description") or c.get("comment"),
        "allowed_values": [_clip(v) for v in allowed][:50] if isinstance(allowed, list) else [],
    }
    ref = _parse_ref(c.get("references") or c.get("foreign_key") or c.get("fk") or c.get("ref"))
    if ref:
        col["_ref"] = ref
    return col


def _relationship_from_spec(r: Any, default_from: str | None = None) -> dict | None:
    if not isinstance(r, dict):
        return None
    src = r.get("from") or r.get("table") or r.get("source_table") or default_from
    cols = r.get("from_columns") or r.get("columns") or r.get("column") or r.get("source_columns")
    ref = r.get("references") or r.get("to") or r.get("ref_table") or r.get("target_table")
    to_cols = r.get("to_columns") or r.get("ref_columns") or r.get("ref_column") or r.get("target_columns")
    if isinstance(ref, dict):
        to_cols = to_cols or ref.get("columns") or ref.get("column")
        ref = ref.get("table")
    elif isinstance(ref, str) and "(" in ref:
        parsed = _parse_ref(ref)
        if parsed:
            ref, to_cols = parsed[0], to_cols or parsed[1]
    if not (src and cols and ref):
        return None
    as_list = lambda v: [v] if isinstance(v, str) else list(v or [])
    return {"from": str(src), "from_columns": as_list(cols), "to": str(ref), "to_columns": as_list(to_cols),
            "confidence": float(r.get("confidence", 1.0)), "kind": r.get("kind") or "foreign_key"}


# ─── SQL DDL ──────────────────────────────────────────────────────────────

def parse_ddl(sql: str, dialect: str | None = None) -> dict:
    """CREATE TABLE statements (any sqlglot dialect) → snapshot."""
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError as e:  # pragma: no cover
        raise SchemaDocumentError("SQL DDL needs sqlglot installed.") from e
    snap = empty_snapshot(dialect=dialect or "", extractor="uploaded SQL DDL (sqlglot)")
    statements = None
    errors = []
    for d in ([dialect] if dialect else []) + [None, "mysql", "postgres", "tsql", "snowflake", "sqlite"]:
        try:
            statements = sqlglot.parse(sql, read=d, error_level=sqlglot.ErrorLevel.IGNORE)
            if any(isinstance(s, exp.Create) for s in statements if s is not None):
                snap["dialect"] = d or snap["dialect"] or "ansi"
                break
        except Exception as e:  # try the next dialect
            errors.append(f"{d or 'default'}: {e}")
            statements = None
    if not statements:
        raise SchemaDocumentError("Could not parse the SQL: " + "; ".join(errors[:2]))
    for stmt in statements:
        if not isinstance(stmt, exp.Create) or str(stmt.args.get("kind") or "").upper() not in ("TABLE", "VIEW"):
            continue
        schema_exp = stmt.this
        table_exp = schema_exp.this if isinstance(schema_exp, exp.Schema) else schema_exp
        if not isinstance(table_exp, exp.Table):
            continue
        tname = table_exp.name
        tschema = table_exp.db or ""
        cols, pk, comment = [], [], None
        if isinstance(schema_exp, exp.Schema):
            for item in schema_exp.expressions:
                if isinstance(item, exp.ColumnDef):
                    col = {"name": item.name, "type": item.args["kind"].sql() if item.args.get("kind") else "",
                           "nullable": True, "pk": False, "unique": False}
                    for cons in item.args.get("constraints") or []:
                        kind = cons.args.get("kind") if isinstance(cons, exp.ColumnConstraint) else cons
                        if isinstance(kind, exp.NotNullColumnConstraint) and not kind.args.get("allow_null"):
                            col["nullable"] = False
                        elif isinstance(kind, exp.PrimaryKeyColumnConstraint):
                            col["pk"] = True
                            col["nullable"] = False
                        elif isinstance(kind, exp.UniqueColumnConstraint):
                            col["unique"] = True
                        elif isinstance(kind, exp.DefaultColumnConstraint):
                            col["default"] = kind.this.sql() if kind.this is not None else None
                        elif isinstance(kind, exp.CommentColumnConstraint):
                            col["comment"] = kind.this.name if kind.this is not None else None
                        elif isinstance(kind, exp.Reference):
                            ref_table, ref_cols = _ddl_reference(kind)
                            if ref_table:
                                snap["relationships"].append({"from": tname, "from_columns": [item.name], "to": ref_table,
                                                              "to_columns": ref_cols, "confidence": 1.0, "kind": "foreign_key"})
                    if "enum" in col["type"].lower():
                        col["allowed_values"] = re.findall(r"'((?:[^']|'')*)'", col["type"])[:50]
                    cols.append(col)
                elif isinstance(item, exp.PrimaryKey):
                    pk = [e.name for e in item.expressions]
                elif isinstance(item, exp.ForeignKey):
                    ref = item.args.get("reference")
                    ref_table, ref_cols = _ddl_reference(ref) if ref is not None else (None, [])
                    if ref_table:
                        snap["relationships"].append({"from": tname, "from_columns": [e.name for e in item.expressions],
                                                      "to": ref_table, "to_columns": ref_cols, "confidence": 1.0,
                                                      "kind": "foreign_key"})
                elif isinstance(item, exp.Constraint):
                    for sub in item.expressions:
                        if isinstance(sub, exp.PrimaryKey):
                            pk = [e.name for e in sub.expressions]
                        elif isinstance(sub, exp.ForeignKey):
                            ref = sub.args.get("reference")
                            ref_table, ref_cols = _ddl_reference(ref) if ref is not None else (None, [])
                            if ref_table:
                                snap["relationships"].append({"from": tname, "from_columns": [e.name for e in sub.expressions],
                                                              "to": ref_table, "to_columns": ref_cols, "confidence": 1.0,
                                                              "kind": "foreign_key"})
        for prop in (stmt.args.get("properties").expressions if stmt.args.get("properties") else []):
            if isinstance(prop, exp.SchemaCommentProperty):
                comment = prop.this.name if prop.this is not None else None
        snap["tables"].append({"schema": tschema, "name": tname, "kind": str(stmt.args.get("kind")).lower(),
                               "columns": cols, "primary_key": pk, "comment": comment})
    if not snap["tables"]:
        raise SchemaDocumentError("No CREATE TABLE statements found in the SQL.")
    return snap


def _ddl_reference(ref) -> tuple[str | None, list[str]]:
    from sqlglot import exp
    target = ref.this
    if isinstance(target, exp.Schema):
        table = target.this
        return (table.name if isinstance(table, exp.Table) else str(table)), [e.name for e in target.expressions]
    if isinstance(target, exp.Table):
        return target.name, []
    return None, []


# ─── diff ─────────────────────────────────────────────────────────────────

def diff(old: dict | None, new: dict) -> dict:
    """What changed between two snapshots of the same source (schema drift)."""
    if not old:
        return {"first": True, "added_tables": [t["key"] for t in new.get("tables", [])], "removed_tables": [],
                "changed_tables": [], "summary": "first snapshot"}
    a, b = table_index(old), table_index(new)
    added = sorted(set(b) - set(a))
    removed = sorted(set(a) - set(b))
    changed = []
    for key in sorted(set(a) & set(b)):
        ca, cb = column_index(a[key]), column_index(b[key])
        add_c = sorted(set(cb) - set(ca))
        rem_c = sorted(set(ca) - set(cb))
        typ = [{"column": c, "from": ca[c]["type"], "to": cb[c]["type"]}
               for c in sorted(set(ca) & set(cb)) if (ca[c]["type"] or "").lower() != (cb[c]["type"] or "").lower()]
        if add_c or rem_c or typ:
            changed.append({"table": key, "added_columns": add_c, "removed_columns": rem_c, "type_changes": typ})
    bits = []
    if added:
        bits.append(f"+{len(added)} tables")
    if removed:
        bits.append(f"−{len(removed)} tables")
    if changed:
        bits.append(f"{len(changed)} changed")
    return {"first": False, "added_tables": added, "removed_tables": removed, "changed_tables": changed,
            "summary": ", ".join(bits) or "no changes"}


def subset(snapshot: dict, table_keys: Iterable[str] | None) -> dict:
    """A copy with only the given tables (and the relationships between them)."""
    if not table_keys:
        return snapshot
    keep = set(table_keys)
    out = dict(snapshot)
    out["tables"] = [t for t in snapshot["tables"] if t["key"] in keep]
    out["relationships"] = [r for r in snapshot.get("relationships", []) if r["from"] in keep and r["to"] in keep]
    return out
