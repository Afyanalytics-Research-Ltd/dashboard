"""Source snapshot + target snapshot → a mapping document, with Groq.

Pipeline
  1. rule-based table candidates   name similarity + column overlap
  2. AI table matching             Groq picks the source table (or none) for every
                                   target table, with a reason and an optional row filter
  3. rule-based field matches      exact / synonym names with compatible types
  4. AI field mapping              per table pair, in parallel: Groq maps every
                                   target column using the allowed rule kinds
                                   (copy, steps, code maps, lookups, splits, constants)
  5. validation                    every suggestion is checked against the real
                                   snapshots; invalid ones are dropped with a reason
Without a Groq key the rule-based results are returned on their own.

The mapping document (``connect-studio.mapping/1``) is what the review UI
edits; ``document.export()`` turns it into a mappings.json.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import Any, Callable

from . import llm, rules
from .snapshot import compatible, now_iso

FORMAT = "connect-studio.mapping/1"
Progress = Callable[..., None]

# ─── names ────────────────────────────────────────────────────────────────

SYNONYMS = [
    {"dob", "date_of_birth", "birth_date", "birthdate", "birthday"},
    {"phone", "mobile", "telephone", "phone_number", "mobile_number", "tel", "cell", "msisdn", "contact"},
    {"sex", "gender"},
    {"surname", "last_name", "lastname", "family_name"},
    {"first_name", "firstname", "given_name", "fname"},
    {"middle_name", "middlename", "other_names", "othernames"},
    {"created_at", "created", "date_created", "created_on", "creation_date", "inserted_at"},
    {"updated_at", "updated", "date_updated", "modified", "modified_at", "last_modified", "updated_on"},
    {"deleted_at", "deleted", "date_deleted"},
    {"email", "email_address", "mail"},
    {"address", "physical_address", "street", "address_line1"},
    {"amount", "total", "value", "price", "cost"},
    {"quantity", "qty", "count", "units"},
    {"description", "desc", "notes", "remarks", "comment"},
    {"name", "title", "label"},
    {"status", "state"},
    {"code", "ref", "reference", "number", "no"},
    {"national_id", "id_number", "nid", "national_id_number", "id_no"},
    {"user_id", "created_by", "staff_id", "employee_id"},
    {"visit_id", "encounter_id", "admission_id"},
    {"patient_id", "client_id", "person_id"},
    {"facility_id", "hospital_id", "clinic_id", "branch_id"},
]
_SYN = {}
for group in SYNONYMS:
    for w in group:
        _SYN.setdefault(w, set()).update(group)

PREFIXES = re.compile(r"^(tbl|tb|t|core|inv|inp|fin|hr|app|dim|fact|stg|src|raw|v\d?)_")


def norm(name: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name or ""))
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()
    return s


def singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def table_norm(name: str) -> str:
    n = norm(name.split(".")[-1])
    n = PREFIXES.sub("", n)
    return "_".join(singular(p) for p in n.split("_") if p)


def name_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ta, tb = set(a.split("_")), set(b.split("_"))
    jac = len(ta & tb) / len(ta | tb) if ta | tb else 0.0
    return max(jac, SequenceMatcher(None, a, b).ratio() * 0.95)


def col_match(src_name: str, tgt_name: str) -> float:
    a, b = norm(src_name), norm(tgt_name)
    if a == b:
        return 1.0
    if b in _SYN.get(a, ()):  # synonym
        return 0.9
    if singular(a) == singular(b):
        return 0.95
    return 0.0


# ─── 1. table candidates ─────────────────────────────────────────────────

def table_candidates(source: dict, target: dict, top: int = 6) -> dict[str, list[tuple[str, float]]]:
    out = {}
    src = [(t["key"], table_norm(t["name"]), {norm(c["name"]) for c in t["columns"]}) for t in source["tables"]]
    for tt in target["tables"]:
        tn = table_norm(tt["name"])
        tcols = {norm(c["name"]) for c in tt["columns"]} - {"id", "created_at", "updated_at", "deleted_at"}
        scored = []
        for key, sn, scols in src:
            ns = name_sim(tn, sn)
            overlap = len(tcols & scols) / len(tcols) if tcols else 0.0
            score = 0.55 * ns + 0.45 * overlap
            if score > 0.15:
                scored.append((key, round(score, 3)))
        scored.sort(key=lambda x: -x[1])
        out[tt["key"]] = scored[:top]
    return out


# ─── prompt rendering ────────────────────────────────────────────────────

def _mask(value: str) -> str:
    return re.sub(r"[A-Za-z]", "x", re.sub(r"\d", "9", value))


def _col_line(c: dict, *, samples: str = "full", target: bool = False) -> str:
    bits = [f"{c['name']} {c.get('type') or c.get('family') or ''}".strip()]
    flags = []
    if c.get("pk"):
        flags.append("PK")
    if not c.get("nullable", True) and not c.get("pk"):
        flags.append("NOT NULL")
    ref = c.get("references")
    if ref:
        flags.append(f"FK→{ref['table']}.{ref['column']}")
    if c.get("comment"):
        flags.append(f"“{str(c['comment'])[:60]}”")
    if c.get("allowed_values"):
        flags.append("allowed: " + " | ".join(map(str, c["allowed_values"][:15])))
    p = c.get("profile") or {}
    if p and samples != "none":
        if p.get("null_pct") is not None and p["null_pct"] >= 1:
            flags.append(f"{p['null_pct']:.0f}% null")
        pii = p.get("pii") or []
        top = p.get("top") or []
        distinct = p.get("distinct")
        mask = samples == "masked" or (samples == "mask_pii" and pii)
        if top and distinct is not None and distinct <= 15:
            vals = [(_mask(t["v"]) if mask else t["v"]) for t in top[:15]]
            flags.append("values: " + " | ".join(vals))
        elif p.get("samples"):
            vals = [(_mask(v) if mask else v) for v in p["samples"][:4]]
            flags.append("e.g. " + " | ".join(vals))
        if pii:
            flags.append("PII:" + ",".join(pii))
    if flags:
        bits.append("[" + "; ".join(flags) + "]")
    return " ".join(bits)


def _table_brief(t: dict, max_cols: int = 30) -> str:
    cols = ", ".join(c["name"] for c in t["columns"][:max_cols])
    more = f", …+{len(t['columns']) - max_cols}" if len(t["columns"]) > max_cols else ""
    rows = f" ({t['rows']} rows)" if t.get("rows") is not None else ""
    label = t["key"]
    if t.get("model"):
        label += f" [model {t['model']}]"
    if t.get("comment"):
        label += f" “{str(t['comment'])[:60]}”"
    return f"- {label}{rows}: {cols}{more}"


TABLE_SYSTEM = """You map database schemas. Given TARGET tables and SOURCE tables, choose for every target table the single
source table whose rows become that target's rows, or null when no source table fits. A source table may feed several
targets (then usually with a row filter). Prefer the rule-based candidates but override them when the columns say otherwise.
Return JSON: {"matches": [{"target": "<target key>", "source": "<source key>" | null, "confidence": 0.0-1.0,
"reason": "<short why>", "where": null | {"column": "<source column>", "in": ["<value>", ...]}}]}
Use exact keys from the lists. Be decisive and honest about confidence."""

FIELD_SYSTEM = """You write column mappings from one SOURCE table into one TARGET table (an ETL migration).
For EVERY target column decide how it is filled from the source, using only these kinds:
- "copied": the source column value as is.
- "worked out": text steps. rule {"steps":[{"op":"lowercase"}]} ops: lowercase, uppercase, trim, slug, first_word, rest_words, title_case.
- "codes converted": value map. rule {"map":{"<source value>":"<target value>"}, "default": <target value or null>}.
  Use it when source codes/labels differ from the target's allowed values; map every source value you can see.
- "looked up in another table": rule {"table":"<source table>","key":"<column in that table>","value":"<column in that table>"}
  = that table's <value> for the row whose <key> equals this row's source column (follow FKs, e.g. a department id → department name).
- "one column divided": rule {"part": 1 | 2 | "rest", "delimiter": " "} — e.g. a full name split into first / middle names.
- "constant": rule {"value": <fixed value>} — only when the target needs a fixed value no source column provides.
Rules: "source" is a column of the SOURCE table (null only for "constant"). Never invent columns. Keep ids/keys/FK columns
copied so links survive. Map semantically (othernames→first_name via split, dob→date_of_birth, sex codes→gender values).
Code maps: send every source value to the semantically closest allowed target value — finished states (served, done,
closed, discharged) → the completed/closed value, not-started states (new, pending, queued) → the waiting/open value;
never send opposite states to the same value. Booleans: map 0/1, yes/no, available/unavailable to true/false.
Lookups: when the target needs a value the source table only reaches through another table (a code → its name, or a
payment's invoice_no → that invoice's patient_id), use "looked up in another table" with a table from RELATED or OTHER
SOURCE TABLES — prefer that over leaving a column unmapped.
If nothing fits a target column, put it in "unmapped" with a reason. confidence: 0.95+ certain, 0.7-0.9 likely, below 0.6 a guess.
Return JSON: {"fields":[{"target":"<target column>","source":"<source column>"|null,"kind":"...","rule":{...}|null,
"transform":"","confidence":0.0-1.0,"why":"<short reason>"}],"unmapped":[{"target":"<target column>","reason":"..."}]}"""


# ─── 2. AI table matching ─────────────────────────────────────────────────

def match_tables(source: dict, target: dict, candidates: dict, *, use_ai: bool, usage: llm.Usage,
                 progress: Progress, chunk: int = 18) -> list[dict]:
    src_keys = {t["key"] for t in source["tables"]}
    results: dict[str, dict] = {}
    # rule-based baseline
    for tt in target["tables"]:
        cands = candidates.get(tt["key"]) or []
        best = cands[0] if cands else None
        results[tt["key"]] = {
            "target": tt["key"], "source": best[0] if best and best[1] >= 0.5 else None,
            "confidence": round(best[1], 2) if best and best[1] >= 0.5 else 0.0,
            "reason": "name and column overlap" if best and best[1] >= 0.5 else "no similar source table",
            "made_by": "rule", "where": None, "candidates": [{"source": k, "score": s} for k, s in cands],
        }
    if not use_ai:
        return list(results.values())

    src_index = {t["key"]: t for t in source["tables"]}
    many = len(source["tables"]) > 120
    targets = target["tables"]
    for i in range(0, len(targets), chunk):
        part = targets[i:i + chunk]
        progress(msg=f"AI matching tables {i + 1}–{i + len(part)} of {len(targets)}…", phase="tables",
                 pct=8 + 17 * (i + len(part)) / max(len(targets), 1))
        lines = ["TARGET TABLES:"]
        involved = set()
        for tt in part:
            cands = candidates.get(tt["key"]) or []
            involved.update(k for k, _ in cands)
            lines.append(_table_brief(tt) + ("  | candidates: " + ", ".join(f"{k} ({s:.2f})" for k, s in cands) if cands else "  | candidates: none"))
        lines.append("")
        lines.append("SOURCE TABLES:")
        for st in source["tables"]:
            if many and st["key"] not in involved:
                continue
            lines.append(_table_brief(st, 25))
        if many:
            others = sorted(src_keys - involved)
            lines.append("OTHER SOURCE TABLES (names only): " + ", ".join(others[:600]))
        try:
            data = llm.chat_json(TABLE_SYSTEM, "\n".join(lines), usage=usage, max_tokens=6000)
        except Exception as e:
            progress(msg=f"AI table matching failed for this batch — keeping rule-based matches ({str(e)[:140]})", level="warning")
            continue
        for m in data.get("matches") or []:
            tkey = m.get("target")
            if tkey not in results:
                continue
            skey = m.get("source")
            if skey and skey not in src_keys:
                # tolerate schema-less names
                skey = next((k for k in src_keys if k.split(".")[-1] == str(skey).split(".")[-1]), None)
                if not skey:
                    continue
            where = m.get("where")
            if where and (not isinstance(where, dict) or not where.get("column") or not isinstance(where.get("in"), list)
                          or (skey and where["column"] not in {c["name"] for c in src_index[skey]["columns"]})):
                where = None
            results[tkey].update({"source": skey, "confidence": _conf(m.get("confidence")), "reason": str(m.get("reason") or "")[:300],
                                  "made_by": "ai", "where": where})
    return list(results.values())


def _conf(v: Any, default: float = 0.6) -> float:
    try:
        return max(0.0, min(1.0, round(float(v), 2)))
    except (TypeError, ValueError):
        return default


# ─── 3 + 4. field mapping ─────────────────────────────────────────────────

def rule_field_matches(src_table: dict, tgt_table: dict) -> dict[str, dict]:
    out = {}
    for tc in tgt_table["columns"]:
        best, best_score = None, 0.0
        for sc in src_table["columns"]:
            s = col_match(sc["name"], tc["name"])
            if s > best_score and compatible(sc.get("family", "unknown"), tc.get("family", "unknown")):
                best, best_score = sc, s
        if best is not None and best_score >= 0.9:
            out[tc["name"]] = {"target": tc["name"], "source": best["name"], "kind": "copied", "rule": {}, "transform": "",
                               "confidence": 0.97 if best_score == 1.0 else 0.85,
                               "why": "same name" if best_score == 1.0 else "synonym / similar name", "made_by": "rule"}
    return out


def _related_tables(source: dict, table_key: str, limit: int = 8) -> list[tuple[dict, dict]]:
    idx = {t["key"]: t for t in source["tables"]}
    out = []
    for r in source.get("relationships", []):
        if r["from"] == table_key and r["to"] in idx:
            out.append((idx[r["to"]], r))
    return out[:limit]


def map_fields(source: dict, target: dict, pair: dict, *, use_ai: bool, usage: llm.Usage, samples: str = "full") -> dict:
    src_idx = {t["key"]: t for t in source["tables"]}
    tgt = next(t for t in target["tables"] if t["key"] == pair["target"])
    src = src_idx[pair["source"]]
    base = rule_field_matches(src, tgt)
    fields: dict[str, dict] = {k: dict(v) for k, v in base.items()}
    unmapped: dict[str, str] = {}
    notes = []
    if use_ai:
        lines = [f"TARGET TABLE {tgt['key']}" + (f" [model {tgt['model']}]" if tgt.get("model") else "") + ":"]
        lines += ["  " + _col_line(c, samples=samples, target=True) for c in tgt["columns"]]
        lines.append("")
        rows = f" ({src['rows']} rows)" if src.get("rows") is not None else ""
        lines.append(f"SOURCE TABLE {src['key']}{rows}:")
        lines += ["  " + _col_line(c, samples=samples) for c in src["columns"]]
        rel = _related_tables(source, src["key"])
        if rel:
            lines.append("")
            lines.append("RELATED SOURCE TABLES (usable for lookups):")
            for t, r in rel:
                lines.append(f"  {t['key']} via {src['key']}.{','.join(r['from_columns'])} → {t['key']}.{','.join(r['to_columns'])}"
                             f" ({r['kind']}): " + ", ".join(_col_line(c, samples='none') for c in t["columns"][:20]))
        related_keys = {t["key"] for t, _ in rel} | {src["key"]}
        others = [t for t in source["tables"] if t["key"] not in related_keys][:30]
        if others:
            lines.append("")
            lines.append("OTHER SOURCE TABLES (no declared link, but usable for lookups):")
            for t in others:
                cols = ", ".join(c["name"] + (" PK" if c.get("pk") else "") for c in t["columns"][:14])
                lines.append(f"  {t['key']}: {cols}")
        if pair.get("where"):
            lines.append(f"\nOnly source rows where {pair['where']['column']} in {pair['where']['in']} feed this target.")
        if base:
            lines.append("\nRULE-BASED MATCHES (confirm or improve): " + ", ".join(f"{k}←{v['source']}" for k, v in base.items()))
        try:
            data = llm.chat_json(FIELD_SYSTEM, "\n".join(lines), usage=usage, max_tokens=min(12000, 1500 + 220 * len(tgt["columns"])))
        except Exception as e:
            data = {}
            notes.append(f"AI field mapping failed — rule-based matches only ({str(e)[:160]})")
        tcols = {c["name"] for c in tgt["columns"]}
        for f in data.get("fields") or []:
            tname = f.get("target")
            if tname not in tcols:
                continue
            cand = rules.normalize({"target": tname, "source": f.get("source"), "kind": f.get("kind") or "copied",
                                    "rule": f.get("rule"), "transform": f.get("transform"),
                                    "confidence": _conf(f.get("confidence")), "why": str(f.get("why") or "")[:300],
                                    "made_by": "ai", "source_table": src["key"]})
            problem = rules.validate(cand, src_idx)
            if problem:
                if tname not in fields:
                    unmapped[tname] = f"AI suggestion rejected: {problem}"
                continue
            prev = fields.get(tname)
            if prev and prev["source"] == cand.get("source") and cand["kind"] == "copied":
                prev["confidence"] = max(prev["confidence"], cand["confidence"], 0.97)
                prev["why"] = cand["why"] or prev["why"]
                prev["made_by"] = "rule+ai"
            else:
                cand.pop("source_table", None)
                fields[tname] = cand
        for u in data.get("unmapped") or []:
            if u.get("target") in tcols and u["target"] not in fields:
                unmapped[u["target"]] = str(u.get("reason") or "no matching source column")[:300]
    for c in tgt["columns"]:
        if c["name"] not in fields and c["name"] not in unmapped:
            unmapped[c["name"]] = "no matching source column"
    return {"fields": fields, "unmapped": unmapped, "notes": notes}


# ─── orchestration ────────────────────────────────────────────────────────

def generate(source: dict, target: dict, *, use_ai: bool = True, samples: str = "full", workers: int = 2,
             progress: Progress = lambda **k: None, only_targets: list[str] | None = None,
             fixed_pairs: dict[str, str | None] | None = None) -> dict:
    """Build a mapping document. ``fixed_pairs`` = {target: source} decisions to keep (re-map runs)."""
    t0 = time.time()
    usage = llm.Usage()
    ai_on = use_ai
    if use_ai:
        try:
            llm._client()
        except llm.LLMUnavailable as e:
            progress(msg=str(e), level="warning")
            ai_on = False
    tgt = target if not only_targets else {**target, "tables": [t for t in target["tables"] if t["key"] in set(only_targets)]}
    progress(msg=f"Scoring {len(tgt['tables'])} target × {len(source['tables'])} source tables…", phase="tables", pct=5)
    cands = table_candidates(source, tgt)
    pairs = match_tables(source, tgt, cands, use_ai=ai_on and not fixed_pairs, usage=usage, progress=progress)
    if fixed_pairs:
        for p in pairs:
            if p["target"] in fixed_pairs:
                p["source"] = fixed_pairs[p["target"]]
                p["made_by"] = "human"
                p["confidence"] = 1.0 if p["source"] else 0.0
    matched = [p for p in pairs if p["source"]]
    progress(msg=f"Matched {len(matched)} of {len(pairs)} target tables. Mapping columns…", phase="fields", pct=26)

    fields_out, unmapped_out, notes = [], [], []
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers if ai_on else 1)) as pool:
        futures = {pool.submit(map_fields, source, target, p, use_ai=ai_on, usage=usage, samples=samples): p for p in matched}
        for fut in as_completed(futures):
            p = futures[fut]
            done += 1
            try:
                res = fut.result()
            except Exception as e:  # pragma: no cover - defensive
                notes.append(f"{p['target']}: {e}")
                continue
            for tname, f in res["fields"].items():
                fields_out.append({
                    "id": f"{p['target']}::{tname}", "target_table": p["target"], "target": tname,
                    "source_table": p["source"], "source": f.get("source"), "kind": f["kind"], "rule": f.get("rule") or {},
                    "transform": f.get("transform") or "", "confidence": f["confidence"], "why": f.get("why") or "",
                    "made_by": f.get("made_by") or "ai", "status": "suggested",
                })
            for tname, reason in res["unmapped"].items():
                unmapped_out.append({"target_table": p["target"], "target": tname, "reason": reason})
            for note in res["notes"]:
                notes.append(f"{p['target']}: {note}")
                progress(msg=f"{p['target']}: {note}", level="warning")
            mapped = len(res["fields"])
            total = mapped + len(res["unmapped"])
            progress(msg=f"{p['source']} → {p['target']}: {mapped}/{total} columns", pct=26 + 70 * done / max(len(matched), 1),
                     phase="fields", detail={"target": p["target"], "mapped": mapped, "total": total})

    tgt_order = {t["key"]: i for i, t in enumerate(target["tables"])}
    col_order = {(t["key"], c["name"]): i for t in target["tables"] for i, c in enumerate(t["columns"])}
    fields_out.sort(key=lambda f: (tgt_order.get(f["target_table"], 0), col_order.get((f["target_table"], f["target"]), 0)))
    for p in pairs:
        if not p["source"]:
            t = next(t for t in target["tables"] if t["key"] == p["target"])
            for c in t["columns"]:
                unmapped_out.append({"target_table": p["target"], "target": c["name"], "reason": "target table has no source"})
    used = {p["source"] for p in matched}
    unused = [{"source_table": t["key"], "rows": t.get("rows"), "reason": "no target table chose it"}
              for t in source["tables"] if t["key"] not in used]
    for p in pairs:
        p["status"] = "suggested"
    progress(msg="Mapping ready for review.", pct=100, phase="done")
    return {
        "format": FORMAT, "generated_at": now_iso(),
        "settings": {"ai": ai_on, "samples": samples, "model": usage.model if ai_on else None},
        "tables": pairs, "fields": fields_out, "unmapped_fields": unmapped_out, "unused_sources": unused,
        "notes": notes, "usage": {**usage.as_dict(), "wall_seconds": round(time.time() - t0, 1)},
    }


def remap_table(doc: dict, source: dict, target: dict, target_table: str, source_table: str | None, *,
                use_ai: bool = True, samples: str = "full", keep_human: bool = True) -> dict:
    """Re-run field mapping for one target table (after the user changes its source)."""
    usage = llm.Usage()
    ai_on = use_ai
    try:
        if use_ai:
            llm._client()
    except llm.LLMUnavailable:
        ai_on = False
    pair = next((p for p in doc["tables"] if p["target"] == target_table), None)
    if pair is None:
        raise ValueError(f"unknown target table {target_table}")
    if pair.get("source") != source_table:  # same source = a plain re-run: keep the AI's match + reason
        pair.update({"source": source_table, "made_by": "human", "confidence": 1.0 if source_table else 0.0,
                     "reason": "chosen in review"})
    human = {f["target"]: f for f in doc["fields"] if f["target_table"] == target_table and f["made_by"] == "human"} if keep_human else {}
    doc["fields"] = [f for f in doc["fields"] if f["target_table"] != target_table]
    doc["unmapped_fields"] = [u for u in doc["unmapped_fields"] if u["target_table"] != target_table]
    doc["notes"] = [n for n in doc.get("notes", []) if not n.startswith(f"{target_table}: ")]
    if source_table:
        res = map_fields(source, target, pair, use_ai=ai_on, usage=usage, samples=samples)
        doc["notes"] += [f"{target_table}: {n}" for n in res["notes"]]
        for tname, f in res["fields"].items():
            if tname in human:
                continue
            doc["fields"].append({"id": f"{target_table}::{tname}", "target_table": target_table, "target": tname,
                                  "source_table": source_table, "source": f.get("source"), "kind": f["kind"],
                                  "rule": f.get("rule") or {}, "transform": f.get("transform") or "",
                                  "confidence": f["confidence"], "why": f.get("why") or "",
                                  "made_by": f.get("made_by") or "ai", "status": "suggested"})
        for tname, reason in res["unmapped"].items():
            if tname not in human:
                doc["unmapped_fields"].append({"target_table": target_table, "target": tname, "reason": reason})
    else:
        t = next(t for t in target["tables"] if t["key"] == target_table)
        for c in t["columns"]:
            if c["name"] not in human:
                doc["unmapped_fields"].append({"target_table": target_table, "target": c["name"], "reason": "target table has no source"})
    doc["fields"].extend(human.values())
    used = {p["source"] for p in doc["tables"] if p["source"]}
    doc["unused_sources"] = [{"source_table": t["key"], "rows": t.get("rows"), "reason": "no target table chose it"}
                             for t in source["tables"] if t["key"] not in used]
    u = doc.setdefault("usage", {})
    for k, v in usage.as_dict().items():
        if isinstance(v, (int, float)):
            u[k] = round((u.get(k) or 0) + v, 1)
    return doc
