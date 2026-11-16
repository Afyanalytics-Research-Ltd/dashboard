"""Mapping document → mappings.json, coverage stats and validation.

Two export flavors:
  generic  neutral keys ("target table", "target_table") — for any target
  afya     the exact shape of dags/jsons/*_data_mappings.json ("v3 model",
           "v3 table", "model", "v3_table" …) so
           pipelines/from_json_mappings_to_snowflake.py runs it unchanged
"""

from __future__ import annotations

import json
from collections import defaultdict

from . import rules
from .snapshot import now_iso

LOW_CONFIDENCE = 0.7


def summary(doc: dict, target: dict | None = None) -> dict:
    """Per-target-table and overall coverage."""
    per: dict[str, dict] = {}
    totals_by_table = {t["key"]: len(t["columns"]) for t in (target or {}).get("tables", [])}
    for p in doc.get("tables", []):
        per[p["target"]] = {"target": p["target"], "source": p.get("source"), "columns": totals_by_table.get(p["target"], 0),
                            "mapped": 0, "approved": 0, "rejected": 0, "review": 0, "unmapped": 0,
                            "confidence": p.get("confidence"), "status": p.get("status")}
    for f in doc.get("fields", []):
        s = per.get(f["target_table"])
        if not s:
            continue
        if f["status"] == "rejected":
            s["rejected"] += 1
            continue
        s["mapped"] += 1
        if f["status"] == "approved":
            s["approved"] += 1
        elif f["confidence"] < LOW_CONFIDENCE:
            s["review"] += 1
    for u in doc.get("unmapped_fields", []):
        if u["target_table"] in per:
            per[u["target_table"]]["unmapped"] += 1
    for s in per.values():
        if not s["columns"]:
            s["columns"] = s["mapped"] + s["unmapped"] + s["rejected"]
        s["coverage"] = round(s["mapped"] / s["columns"], 3) if s["columns"] else 0.0
        s["approved_pct"] = round(s["approved"] / s["mapped"], 3) if s["mapped"] else 0.0
    tables = list(per.values())
    mapped = sum(s["mapped"] for s in tables)
    columns = sum(s["columns"] for s in tables)
    approved = sum(s["approved"] for s in tables)
    return {
        "tables": tables,
        "target_tables": len(tables),
        "matched_tables": sum(1 for s in tables if s["source"]),
        "columns": columns, "mapped": mapped, "approved": approved,
        "review": sum(s["review"] for s in tables),
        "unmapped": sum(s["unmapped"] for s in tables),
        "rejected": sum(s["rejected"] for s in tables),
        "coverage": round(mapped / columns, 3) if columns else 0.0,
        "approved_pct": round(approved / mapped, 3) if mapped else 0.0,
        "ai_fields": sum(1 for f in doc.get("fields", []) if "ai" in str(f.get("made_by"))),
        "human_fields": sum(1 for f in doc.get("fields", []) if f.get("made_by") == "human"),
    }


def _rule_text(f: dict) -> str:
    rule = dict(f.get("rule") or {})
    if f["kind"] == "copied" and not rule:
        return ""
    if f["kind"] == "worked out" and not rule.get("steps") and f.get("transform"):
        rule["steps"] = [{"op": f["transform"]}]
    if f.get("source"):
        rule["column"] = f["source"]
    return json.dumps(rule, ensure_ascii=False, separators=(",", ":"))


def _special(f: dict, model: str) -> dict | None:
    kind_type = {"codes converted": "codes converted", "one column divided": "divide",
                 "looked up in another table": "cross-field", "constant": "constant", "worked out": "worked out"}
    t = kind_type.get(f["kind"])
    if not t:
        return None
    return {"type": t, "v3_model": model, "v3_field": f["target"], "source_table": f["source_table"],
            "source_columns": [f["source"]] if f.get("source") else [], "rule": _rule_text(f),
            "why": f.get("why") or "", "made_by": f.get("made_by") or "", "examples": []}


def export(doc: dict, *, source_meta: dict, target_meta: dict, target: dict | None = None, source: dict | None = None,
           flavor: str = "generic", include_suggested: bool = True, connection_id: int | None = None) -> dict:
    """The mappings.json document."""
    tgt_tables = {t["key"]: t for t in (target or {}).get("tables", [])}
    src_tables = {t["key"]: t for t in (source or {}).get("tables", [])}
    fields_by_table = defaultdict(list)
    for f in doc.get("fields", []):
        if f["status"] == "rejected" or (not include_suggested and f["status"] != "approved"):
            continue
        fields_by_table[f["target_table"]].append(f)
    afya = flavor == "afya"
    connection = {
        "id": connection_id if connection_id is not None else source_meta.get("id"),
        "name": source_meta.get("name"),
        "exported_at": now_iso(),
        "generated_by": "Connect Studio (dbatlas + Groq)",
        "source": {k: source_meta.get(k) for k in ("id", "name", "kind")},
        "target": {k: target_meta.get(k) for k in ("id", "name", "kind")},
    }
    tables, fields, special = [], [], []
    for p in doc.get("tables", []):
        if not p.get("source"):
            continue
        tt = tgt_tables.get(p["target"], {})
        model = tt.get("model") or p["target"]
        tfields = fields_by_table.get(p["target"], [])
        all_ok = tfields and all(f["status"] == "approved" for f in tfields)
        found_by = f"{p.get('made_by', 'ai')}: {p.get('reason') or ''}".strip(": ")
        rows = (src_tables.get(p["source"]) or {}).get("rows") or 0
        if afya:
            entry = {"v3 model": model, "v3 table": tt.get("name") or p["target"], "v3 service": tt.get("service") or tt.get("schema") or "",
                     "v3 table found by": found_by, "source table": p["source"], "rows": rows,
                     "status": "done" if all_ok else "review"}
        else:
            entry = {"source table": p["source"], "target table": p["target"], "target model": tt.get("model"),
                     "target schema": tt.get("schema") or "", "found by": found_by, "rows": rows,
                     "status": "done" if all_ok else "review", "confidence": p.get("confidence")}
        if p.get("where"):
            entry["where"] = p["where"]
        tables.append(entry)
        for f in tfields:
            if afya and f["kind"] not in rules.LOADER_KINDS:
                s = _special(f, model)
                if s:
                    special.append(s)
                continue
            if afya:
                fields.append({"model": model, "v3_table": tt.get("name") or p["target"], "table": f["source_table"],
                               "column": f.get("source") or "", "field": f["target"], "kind": f["kind"],
                               "rule": _rule_text(f), "transform": f.get("transform") or "", "why": f.get("why") or "",
                               "confidence": f.get("confidence"), "status": f["status"], "made_by": f.get("made_by")})
            else:
                fields.append({"table": f["source_table"], "column": f.get("source") or "", "target_table": p["target"],
                               "field": f["target"], "kind": f["kind"], "rule": _rule_text(f),
                               "transform": f.get("transform") or "", "why": f.get("why") or "",
                               "confidence": f.get("confidence"), "status": f["status"], "made_by": f.get("made_by")})
            if f["kind"] not in ("copied",):
                s = _special(f, model)
                if s:
                    special.append(s)
    not_mapped = {u["source_table"]: u.get("reason") or "not used" for u in doc.get("unused_sources", [])}
    out = {"connection": connection, "tables": tables, "fields": fields, "special_actions": special, "not_mapped": not_mapped,
           "unmapped_fields": [{"table": u["target_table"], "field": u["target"], "reason": u["reason"]}
                               for u in doc.get("unmapped_fields", [])]}
    if not afya:
        out = {"format": "mappings/1", **out}
    return out


def validate_export(exported: dict, flavor: str = "generic") -> list[str]:
    """The checks the V3 loader's _check_shape makes (afya) or their generic equivalent."""
    problems = []
    conn = exported.get("connection")
    if not isinstance(conn, dict) or "id" not in conn:
        problems.append('"connection" must be an object with an "id"')
    elif flavor == "afya":
        try:
            int(conn["id"])
        except (TypeError, ValueError):
            problems.append(f'connection "id" must be a number for the V3 loader, got {conn.get("id")!r}')
    tables = exported.get("tables")
    if not isinstance(tables, list) or not tables:
        problems.append('"tables" must be a non-empty list — match at least one target table to a source')
    else:
        keys = ("source table", "v3 model") if flavor == "afya" else ("source table", "target table")
        for i, t in enumerate(tables):
            for key in keys:
                if not isinstance(t.get(key), str) or not t[key].strip():
                    problems.append(f'tables[{i}] needs a "{key}"')
    for i, f in enumerate(exported.get("fields", [])):
        need_target = f.get("v3_table") or f.get("model") if flavor == "afya" else f.get("target_table")
        if not f.get("table") or not need_target or not f.get("field"):
            problems.append(f"fields[{i}] needs a source table, a field and a target table")
    return problems[:50]
