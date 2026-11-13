#!/usr/bin/env python3
"""
from_json_mappings_to_snowflake.py — Afya Extraction → Snowflake in the V3 data
structure, driven by nothing but a connection's mappings export
(e.g. silverwood_data_mappings.json).

INPUT   one mappings JSON: {"connection": {"id": …}, "tables": [source table →
        V3 model], "fields": [source column → V3 field, per table/model]} —
        given as a file path, as the JSON text itself, or (from Python / an
        Airflow trigger conf) as the parsed object. See read_mappings().

FLOW
  1. NAME     GET /connections/<id> → the connection's name ("silverwood") →
              Snowflake schemas <NAME>_RAW and <NAME>_CLEAN (created if missing;
              CLEAN is left for the JSON flattening step).
  2. EXTRACT  every source table via POST /gateway {connection_id, namespace:
              <source table>, page, per_page} — parallel page windows until a
              page is shorter than the size the gateway reports (it caps at 500);
              401 → re-login, 429/5xx/timeouts → retry with backoff.
  3. MAP      each source row → one V3 record per V3 model the table maps to:
              {<V3 field>: value} using the mapping's source column, or the V3
              field name when the gateway already returns it renamed; the source
              row id is kept as "id". Secrets (password hashes, tokens) are
              stripped. Columns the mapping doesn't cover are not carried over.
  4. LAND     <NAME>_RAW.EVENTS_RAW — the same layout the V2→V3 pipeline uses
              (facility_id, ingested_at, module_source, source_table, namespace,
              payload), with source_table = the V3 table (reception_patient,
              patientevaluation_vital, …) and namespace = the V3 model. Each
              table is replaced on every run (new load COPYed in, older loads of
              that table deleted after it succeeds), so re-runs never duplicate.
  5. REPORT   rows per V3 table and how many of the mapped fields carry data;
              every table's result is also logged in <NAME>_RAW.EXTRACTION_RUNS.

Then flatten into <NAME>_CLEAN with the existing step
(snowflake_flatten_clean DAG, facility = <name>).

CANARY  (--canary N, DAG param canary; see canary.py and CANARY_README.md)
  Only the first N records per V3 table are extracted (page by page, until
  each V3 table has N rows past its "where" filter) and landed in the same
  EVENTS_RAW as a full run — but as an upsert: rows of the same record ids
  from earlier loads are replaced and nothing else in the table is touched
  (no table replace, no clear when the source is empty, no prune). Each load
  is then read back and compared with what was sent: every record present,
  every value as sent, no record id stored twice. A failed check fails the
  table (exit 1 / failed task) and the report goes to
  output/canary/from_json_mappings_to_snowflake/. The full run afterwards
  replaces each table as usual, canary rows included, so nothing is doubled.

USAGE
  python from_json_mappings_to_snowflake.py                             # default mappings file
  python from_json_mappings_to_snowflake.py --mappings other.json
  python from_json_mappings_to_snowflake.py --mappings '{"connection": {"id": 35}, "tables": [...], "fields": [...]}'
  cat other.json | python from_json_mappings_to_snowflake.py --mappings -   # JSON on stdin
  python from_json_mappings_to_snowflake.py --list                      # what the file covers
  python from_json_mappings_to_snowflake.py --tables person patient_visit
  python from_json_mappings_to_snowflake.py --dry-run                   # extract + map, write nothing
  python from_json_mappings_to_snowflake.py --canary 3                  # 3 records per V3 table, checked

ENV  (repo .env, or Airflow — see dags/from_json_mappings_to_snowflake_dag.py)
  AFYA_EXTRACTION_BASE_URL (default https://afyapi.afyaanalytics.ai/api)
  AFYA_EXTRACTION_USERNAME / AFYA_EXTRACTION_PASSWORD
  SNOWFLAKE_* / AWS_*  — as for the other loaders
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import requests

# Shared plumbing from the main loader: Snowflake client, S3 client + stage,
# JSON encoder, secret stripping. Importing it also loads the repo .env.
from facility_to_snowflake_fast_resume import (
    S3_BUCKET, SF_DB, SF_FILE_FORMAT, SF_STAGE,
    SnowflakeClient, _dumps_bytes, _s3_client, _strip_secrets,
)

log = logging.getLogger("from_json_mappings_to_snowflake")

ROOT = Path(__file__).resolve().parent
DEFAULT_MAPPINGS = ROOT / "silverwood_data_mappings.json"
S3_PREFIX = "raw/extraction"


def slug(name: str) -> str:
    """"St Peters Orthopedics" → "st_peters_orthopedics"."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


# ─── MAPPINGS ────────────────────────────────────────────────────────────

def _rule(f: dict) -> dict:
    """A field's "rule" (JSON text or object) — {} when there is none."""
    rule = f.get("rule")
    if isinstance(rule, dict):
        rule = dict(rule)
    else:
        try:
            rule = json.loads(rule) if rule else {}
        except (TypeError, ValueError):
            log.warning("%s.%s → %s: unreadable rule %r — copied as is", f.get("table"), f.get("column"), f.get("field"), rule)
            return {}
    # newer exports write a lookup flat: {"table": "invoices", "key": "id",
    # "value": "patient_id", "column": "invoice_no"} = invoices.patient_id of
    # the row whose id equals this row's invoice_no
    if "lookup" not in rule and {"table", "key", "value"} <= rule.keys():
        rule["lookup"] = {"table": rule.pop("table"), "match": rule.pop("key"), "value": rule.pop("value")}
    # the field's "transform" (lowercase …) is one more text step
    transform = str(f.get("transform") or "").strip().lower()
    if transform in _STEPS:
        rule["steps"] = [*rule.get("steps", []), {"op": transform}]
    return rule


_STEPS = {"lowercase": lambda s: s.lower(), "uppercase": lambda s: s.upper(), "trim": lambda s: s.strip(),
          "slug": lambda s: slug(s),
          # "Stephen Ombaka Murono" → first_word "Stephen" · rest_words "Ombaka Murono"
          "first_word": lambda s: (s.split() or [""])[0], "rest_words": lambda s: " ".join(s.split()[1:]),
          "title_case": lambda s: s.title()}


# Lookup rules read a value from another source table:
#   {"lookup": {"table": "facility_employee", "match": "user_id", "value": "staff_number"}, "column": "id"}
# = facility_employee.staff_number of the row whose user_id equals this row's id.
# Tables are fetched once per process (load_lookups) before rows are mapped.
_LOOKUPS: dict[tuple[str, str, str], dict[str, object]] = {}
_lookup_lock = threading.Lock()


def _lookup_key(spec: dict) -> tuple[str, str, str]:
    return spec["table"], spec["match"], spec["value"]


def load_lookups(models: list["V3Model"], client: "ExtractionClient") -> None:
    for m in models:
        for f in m.fields:
            spec = _rule(f).get("lookup")
            if not spec:
                continue
            key = _lookup_key(spec)
            with _lookup_lock:
                if key in _LOOKUPS:
                    continue
            rows = client.fetch_all(spec["table"])
            found = {str(r.get(spec["match"])): r.get(spec["value"]) for r in rows
                     if r.get(spec["match"]) not in (None, "")}
            with _lookup_lock:
                _LOOKUPS[key] = found
            log.info("  lookup %s.%s by %s: %d rows", spec["table"], spec["value"], spec["match"], len(found))


def _lookup(spec: dict, value):
    if value in (None, ""):
        return None
    return _LOOKUPS.get(_lookup_key(spec), {}).get(str(value))


def _apply_rule(rule: dict, value):
    """Apply a mapping rule to one value:
      {"map": {"0": 1, "1": 0}, "default": …}  code conversion (unknown codes → default, else unchanged)
      {"steps": [{"op": "lowercase"}, …]}     text steps: lowercase, uppercase, trim, slug
      {"part": 1 | "rest", "delimiter": " "}  one column divided: the nth piece (from 1), or all after the first"""
    if not rule or value is None:
        return value
    if "part" in rule and isinstance(value, str):
        pieces = value.split(rule.get("delimiter") or None)
        part = rule["part"]
        if part == "rest":
            value = (rule.get("delimiter") or " ").join(pieces[1:]) or None
        else:
            value = pieces[int(part) - 1] if 0 < int(part) <= len(pieces) else None
    if "map" in rule:
        table = {str(k).lower(): v for k, v in rule["map"].items()}
        value = table.get(str(value).strip().lower(), rule.get("default", value))
    for step in rule.get("steps", []):
        op = _STEPS.get(str(step.get("op", "")).lower())
        if op is None:
            log.warning("unknown rule step %r — skipped", step)
        elif isinstance(value, str):
            value = op(value)
    return value


@dataclass
class V3Model:
    """One V3 table fed by one source table, with its field mappings."""
    source_table: str
    namespace: str                                     # App\Models\Reception\Patient
    v3_table: str = ""                                 # "v3 table" in the mappings: patients, visits …
    service: str = ""                                  # "v3 service": reception-service …
    fields: list[dict] = field(default_factory=list)   # {"column", "field", ...}
    where: dict | None = None                          # table filter: {"column": c, "in": [values]}

    def accepts(self, row: dict) -> bool:
        """Rows this V3 table takes (a table entry's "where": e.g. only the
        Drug / Consumable rows of a services list go to inventory products)."""
        if not self.where:
            return True
        value = row.get(self.where["column"])
        allowed = {str(v).lower() for v in self.where.get("in", [])}
        return value is not None and str(value).lower() in allowed

    @property
    def module(self) -> str:
        """module_source in EVENTS_RAW: the V3 service, else the model's module."""
        if self.service:
            return self.service
        parts = [p for p in self.namespace.split("\\") if p not in ("", "App", "Models")]
        return parts[0] if len(parts) > 1 else ""

    @property
    def table(self) -> str:
        """source_table in EVENTS_RAW = the CLEAN view name after flattening:
        the V3 table from the mappings (exports without one: derived from the
        model, App\\Models\\Reception\\Patient → reception_patient)."""
        if self.v3_table:
            return slug(self.v3_table)
        parts = [p for p in self.namespace.split("\\") if p not in ("", "App", "Models")]
        return slug("_".join(parts)) or slug(self.source_table)

    def to_v3(self, row: dict) -> dict:
        """Source row → V3 record: each V3 field from its source column, or from
        the V3 name itself when the gateway already returned it renamed."""
        out = {"id": row.get("id")}
        for f in self.fields:
            target, column = f["field"], f["column"]
            if target == "id":
                continue                      # the source id stays the record's id
            rule = _rule(f)
            column = rule.get("column") or column
            if ((rule.keys() & {"lookup", "map", "part"}) and target != column
                    and row.get(column) is None and row.get(target) is not None):
                # the source column isn't sent but the extraction tool worked
                # this field out itself (its lookup / code conversion / divide
                # rules) — applying the rule again would e.g. look a patient
                # id up as an invoice number. When the source column IS sent,
                # the rule is applied here: the tool stores rules posted
                # through its API as plain copies.
                value = _apply_rule({"steps": rule.get("steps", [])}, row[target])
            else:
                # the source column, else a name the extraction tool already
                # renamed it to: those listed under "also", then the V3 field
                # name itself (last — the tool may fill that name from a
                # different column of its own)
                value = next((row.get(k) for k in (column, *f.get("also", ()), target)
                              if row.get(k) is not None), None)
                if "lookup" in rule:
                    value = _lookup(rule["lookup"], value)
                value = _apply_rule(rule, value)
            # several columns mapped to one field (e.g. visit_id → visit AND
            # id → visit): the first one, in mapping order, with a value wins
            if out.get(target) is None:
                out[target] = value
        return out

    @property
    def shared_targets(self) -> dict[str, list[str]]:
        """V3 fields that more than one source column maps to."""
        by_target = defaultdict(list)
        for f in self.fields:
            by_target[f["field"]].append(f["column"])
        return {t: cols for t, cols in by_target.items() if len(cols) > 1}


class MappingsError(ValueError):
    """The mappings can't be used: file missing, not JSON, or not shaped like a mappings export."""


INLINE_LABEL = "inline mappings JSON"


def looks_like_json(text: str) -> bool:
    """True for JSON text ({…}, […] or a JSON-encoded string "…") — anything else is a file path."""
    s = text.lstrip("﻿ \t\r\n")
    return s[:1] in ("{", "[", '"')


def parse_json_text(text: str, label: str = INLINE_LABEL):
    """JSON text → its value. A JSON-encoded JSON text ("{\\"connection\\": …}", as it arrives when
    pasted escaped into a trigger conf) is decoded once more; a JSON string that isn't JSON (a quoted
    file path) comes back as that str."""
    try:
        doc = json.loads(text.lstrip("﻿").strip())
    except json.JSONDecodeError as e:
        raise MappingsError(f"{label} is not valid JSON: {e.msg} (line {e.lineno}, column {e.colno})") from None
    if isinstance(doc, str) and looks_like_json(doc) and doc.lstrip("﻿ \t\r\n")[:1] != '"':
        return parse_json_text(doc, label)
    return doc


def _check_shape(doc, label: str) -> None:
    """What Mappings.load relies on, checked up front so a bad input fails with a clear message
    instead of a KeyError half-way through."""
    if not isinstance(doc, dict):
        raise MappingsError(f'{label}: expected a JSON object {{"connection": …, "tables": […], "fields": […]}}, '
                            f"got {type(doc).__name__}")
    conn = doc.get("connection")
    if not isinstance(conn, dict) or "id" not in conn:
        raise MappingsError(f'{label}: "connection" must be an object with an "id"')
    if isinstance(conn["id"], bool):
        raise MappingsError(f'{label}: connection "id" must be a number, got {conn["id"]!r}')
    try:
        int(conn["id"])
    except (TypeError, ValueError):
        raise MappingsError(f'{label}: connection "id" must be a number, got {conn["id"]!r}') from None
    tables = doc.get("tables")
    if not isinstance(tables, list) or not tables:
        raise MappingsError(f'{label}: "tables" must be a non-empty list of source table → V3 model entries')
    for i, t in enumerate(tables):
        if not isinstance(t, dict):
            raise MappingsError(f"{label}: tables[{i}] must be an object, got {type(t).__name__}")
        for key in ("source table", "v3 model"):
            if not isinstance(t.get(key), str) or not t[key].strip():
                raise MappingsError(f'{label}: tables[{i}] needs a "{key}"')
    fields = doc.get("fields", [])
    if not isinstance(fields, list):
        raise MappingsError(f'{label}: "fields" must be a list')
    for i, f in enumerate(fields):
        if not isinstance(f, dict):
            raise MappingsError(f"{label}: fields[{i}] must be an object, got {type(f).__name__}")
        if not f.get("table") or not (f.get("v3_table") or f.get("model")) or not f.get("field"):
            raise MappingsError(f'{label}: fields[{i}] needs "table", "field" and "v3_table" or "model"')


def read_mappings(source) -> tuple[dict, str]:
    """The mappings document and a label for messages, from any of:
      a file path (Path or str)     silverwood_data_mappings.json, /abs/path.json
      JSON text                     '{"connection": {"id": 35}, "tables": [...], "fields": [...]}'
                                    (also JSON-encoded twice, or a JSON string holding a path)
      an already-parsed dict        e.g. an Airflow trigger conf object
    Raises MappingsError with a readable message for anything unusable."""
    if isinstance(source, (bytes, bytearray)):
        try:
            source = bytes(source).decode("utf-8-sig")
        except UnicodeDecodeError:
            raise MappingsError("Mappings bytes are not UTF-8 text") from None
    if isinstance(source, (dict, list)):               # a list fails the shape check with a clear message
        doc, label = source, INLINE_LABEL
    elif isinstance(source, str):
        text = source.lstrip("﻿").strip()
        if not text:
            raise MappingsError("No mappings given: pass a mappings JSON file path or the JSON itself")
        if looks_like_json(text):
            doc, label = parse_json_text(text), INLINE_LABEL
            if isinstance(doc, str):                   # a quoted file path: "silverwood_data_mappings.json"
                if not doc.strip():
                    raise MappingsError("No mappings given: the JSON string is empty")
                return read_mappings(Path(doc.strip()))
        else:
            return read_mappings(Path(text))
    elif isinstance(source, os.PathLike):
        path = Path(source).expanduser()
        if path.is_dir():
            raise MappingsError(f"Mappings path is a directory, not a JSON file: {path}")
        if not path.is_file():
            raise MappingsError(f"Mappings file not found: {path}")
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            raise MappingsError(f"Mappings file {path.name} is not UTF-8 text") from None
        doc, label = parse_json_text(text, f"Mappings file {path.name}"), path.name
    else:
        raise MappingsError(f"Mappings must be a file path, JSON text or a dict, not {type(source).__name__}")
    _check_shape(doc, label)
    return doc, label


@dataclass
class Mappings:
    connection_id: int
    exported_name: str
    models: list[V3Model]
    label: str = ""                                    # file name, or INLINE_LABEL — for messages

    @classmethod
    def load(cls, source) -> "Mappings":
        """source: a mappings file path, the mappings JSON text, or the parsed dict (see read_mappings)."""
        doc, label = read_mappings(source)
        # A field belongs to a table entry by (source table, V3 table); fields
        # exported without a v3_table by (source table, model), where model is
        # the full namespace or, in older exports, its last part.
        fields = defaultdict(list)
        for f in doc.get("fields", []):
            fields[(f["table"], f.get("v3_table") or f["model"])].append(f)

        def fields_for(t: dict) -> list[dict]:
            src, ns = t["source table"], t["v3 model"]
            return (fields.get((src, t.get("v3 table") or "\0"), []) + fields.get((src, ns), [])
                    + fields.get((src, ns.split("\\")[-1]), []))

        models = [V3Model(t["source table"], t["v3 model"], t.get("v3 table") or "", t.get("v3 service") or "",
                          fields_for(t), t.get("where")) for t in doc["tables"]]
        dupes = sorted(n for n, c in Counter(m.table for m in models).items() if c > 1)
        if dupes:
            raise MappingsError(f"{label}: several entries map to the same V3 table: {dupes}")
        for mdl in models:
            for target, cols in mdl.shared_targets.items():
                log.warning("%s → %s: %s all map to %r — the first with a value is used",
                            mdl.source_table, mdl.table, ", ".join(cols), target)
        conn = doc["connection"]
        return cls(int(conn["id"]), str(conn.get("name") or ""), models, label)

    @property
    def source_tables(self) -> list[str]:
        return sorted({m.source_table for m in self.models})

    def models_for(self, source_table: str) -> list[V3Model]:
        return [m for m in self.models if m.source_table == source_table]


# ─── EXTRACTION API ──────────────────────────────────────────────────────

class ExtractionClient:
    """Afya Extraction API for one connection: login, metadata, paginated reads."""

    TOKEN_TTL = 45 * 60

    def __init__(self, connection_id: int, *, per_page: int = 500, page_workers: int = 4,
                 timeout: int = 180, max_retries: int = 6):
        self.base = (os.getenv("AFYA_EXTRACTION_BASE_URL") or "https://afyapi.afyaanalytics.ai/api").rstrip("/")
        self.username = (os.getenv("AFYA_EXTRACTION_USERNAME") or "").strip()
        self.password = (os.getenv("AFYA_EXTRACTION_PASSWORD") or "").strip()
        if not self.username or not self.password:
            raise RuntimeError("Set AFYA_EXTRACTION_USERNAME / AFYA_EXTRACTION_PASSWORD")
        self.connection_id = connection_id
        self.per_page, self.page_workers = per_page, max(1, page_workers)
        self.timeout, self.max_retries = timeout, max_retries
        self._session = requests.Session()
        self._token: tuple[str, float] | None = None
        self._lock = threading.Lock()

    def _bearer(self, force: bool = False) -> str:
        with self._lock:
            if force or not self._token or time.time() - self._token[1] > self.TOKEN_TTL:
                r = self._session.post(f"{self.base}/auth/login", timeout=60,
                                       json={"username": self.username, "password": self.password})
                r.raise_for_status()
                body = r.json()
                token = body.get("token") or (body.get("data") or {}).get("token")
                if not token:
                    raise RuntimeError(f"Extraction login returned no token: {str(body)[:200]}")
                self._token = (token, time.time())
            return self._token[0]

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._bearer()}", "Accept": "application/json"}

    def connection(self) -> dict:
        """The connection record (name, facility_id, status …)."""
        r = self._session.get(f"{self.base}/connections/{self.connection_id}", headers=self._headers(), timeout=60)
        r.raise_for_status()
        return r.json().get("data") or {}

    def fetch_page(self, source_table: str, page: int) -> tuple[list[dict], int]:
        """(rows, page size the gateway actually used — it caps per_page at 500)."""
        body = {"connection_id": self.connection_id, "namespace": source_table, "page": page, "per_page": self.per_page}
        wait = 5
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self._session.post(f"{self.base}/gateway", json=body, headers=self._headers(), timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt == self.max_retries:
                    raise
                log.warning("  %s p%d: %s — retry in %ss", source_table, page, type(e).__name__, wait)
                time.sleep(wait); wait = min(wait * 2, 120)
                continue
            if r.status_code == 401:
                self._bearer(force=True)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == self.max_retries:
                    r.raise_for_status()
                delay = int(r.headers.get("Retry-After") or wait)
                log.warning("  %s p%d: HTTP %s — retry in %ss", source_table, page, r.status_code, delay)
                time.sleep(delay); wait = min(wait * 2, 120)
                continue
            if not r.ok:
                raise RuntimeError(f"{source_table} page {page}: HTTP {r.status_code} {r.text[:200]}")
            payload = r.json()
            data = payload.get("data")
            if isinstance(data, dict):
                data = data.get("data")
            return (data if isinstance(data, list) else []), int(payload.get("per_page") or self.per_page)
        # only reached when every attempt got a 401: an empty page here would
        # read as "the table ended" — an empty or truncated load that looks fine
        raise RuntimeError(f"{source_table} page {page}: still HTTP 401 after {self.max_retries} "
                           f"re-logins — check AFYA_EXTRACTION_USERNAME / _PASSWORD")

    def iter_pages(self, source_table: str, max_pages: int | None = None):
        """Pages of rows one at a time, in order (generator), until a page is
        shorter than the gateway's page size or `max_pages` were read. For a
        canary, which stops as soon as it has its records instead of reading
        the whole table."""
        page = 1
        while max_pages is None or page <= max_pages:
            rows, page_size = self.fetch_page(source_table, page)
            yield rows
            if len(rows) < page_size:
                return
            page += 1

    def fetch_all(self, source_table: str) -> list[dict]:
        """Every row. No page count is reported, so pages are fetched in
        parallel windows until one is shorter than the gateway's page size."""
        rows, page = [], 1
        while True:
            window = list(range(page, page + self.page_workers))
            with ThreadPoolExecutor(max_workers=len(window)) as pool:
                results = dict(zip(window, pool.map(lambda p: self.fetch_page(source_table, p), window)))
            for p in window:
                page_rows, page_size = results[p]
                rows.extend(page_rows)
                if len(page_rows) < page_size:
                    return rows
            page += len(window)


# ─── SNOWFLAKE ───────────────────────────────────────────────────────────

def _sq(value) -> str:
    return str(value).replace("\\", "\\\\").replace("'", "''")


class Warehouse:
    """<NAME>_RAW.EVENTS_RAW (V2→V3 pipeline layout) + <NAME>_CLEAN + run log."""

    def __init__(self, name: str):
        self.name = slug(name)
        self.raw = f"{SF_DB}.{self.name.upper()}_RAW"
        self.clean = f"{SF_DB}.{self.name.upper()}_CLEAN"
        self.events = f"{self.raw}.EVENTS_RAW"
        self.runs = f"{self.raw}.EXTRACTION_RUNS"

    def ensure(self, sf: SnowflakeClient) -> None:
        sf.execute(f"CREATE SCHEMA IF NOT EXISTS {self.raw}", label="ensure_raw")
        sf.execute(f"CREATE SCHEMA IF NOT EXISTS {self.clean}", label="ensure_clean")
        sf.execute(f"""CREATE TABLE IF NOT EXISTS {self.events} (
            facility_id STRING NOT NULL, ingested_at TIMESTAMP_TZ NOT NULL, module_source STRING,
            source_table STRING, namespace STRING, payload VARIANT)""", label="ensure_events_raw")
        sf.execute(f"""CREATE TABLE IF NOT EXISTS {self.runs} (
            run_id STRING, connection_id NUMBER, source_table STRING, v3_tables STRING, status STRING,
            source_rows NUMBER, started_at TIMESTAMP_TZ, finished_at TIMESTAMP_TZ, error STRING)""",
                   label="ensure_runs")

    def replace_table(self, sf: SnowflakeClient, model: V3Model, records: list[dict], run_id: str) -> None:
        """Load this V3 table's records, then drop its older loads — atomic per table
        from a reader's point of view only after the COPY succeeded."""
        now = self._copy(sf, model, records, run_id)
        sf.execute(f"""DELETE FROM {self.events}
            WHERE facility_id = '{_sq(self.name)}' AND source_table = '{_sq(model.table)}'
              AND ingested_at < '{now.isoformat()}'::TIMESTAMP_TZ""", label=f"replace:{model.table}")

    def upsert_records(self, sf: SnowflakeClient, model: V3Model, records: list[dict], run_id: str) -> datetime:
        """Canary load: COPY these records in, then drop only the older rows of
        the SAME record ids — the rest of the table (an earlier full load) is
        kept, and re-running a canary never stores a record twice. Returns the
        load's ingested_at, which identifies its rows for the read-back."""
        now = self._copy(sf, model, records, run_id, suffix="canary")
        ids = sorted({str(r["id"]) for r in records if r.get("id") is not None})
        for i in range(0, len(ids), 1000):
            in_list = ", ".join(f"'{_sq(x)}'" for x in ids[i:i + 1000])
            sf.execute(f"""DELETE FROM {self.events}
                WHERE facility_id = '{_sq(self.name)}' AND source_table = '{_sq(model.table)}'
                  AND ingested_at < '{now.isoformat()}'::TIMESTAMP_TZ
                  AND payload:id::STRING IN ({in_list})""", label=f"canary_upsert:{model.table}")
        return now

    def read_back(self, sf: SnowflakeClient, model: V3Model, ingested_at: datetime) -> list[dict]:
        """The payloads of one load of this V3 table (its rows carry that ingested_at)."""
        cur = sf._conn.cursor()
        cur.execute(f"SELECT payload FROM {self.events} WHERE facility_id = %s AND source_table = %s "
                    f"AND ingested_at = %s::TIMESTAMP_TZ", (self.name, model.table, ingested_at.isoformat()))
        return [json.loads(r[0]) if isinstance(r[0], str) else r[0] for r in cur.fetchall()]

    def duplicate_ids(self, sf: SnowflakeClient, model: V3Model, ids: list[str]) -> list[str]:
        """Of these record ids, the ones stored more than once in this V3 table (any load)."""
        if not ids:
            return []
        cur = sf._conn.cursor()
        in_list = ", ".join(f"'{_sq(x)}'" for x in ids)
        cur.execute(f"SELECT payload:id::STRING FROM {self.events} WHERE facility_id = %s AND source_table = %s "
                    f"AND payload:id::STRING IN ({in_list}) GROUP BY 1 HAVING COUNT(*) > 1",
                    (self.name, model.table))
        return sorted(r[0] for r in cur.fetchall())

    def _copy(self, sf: SnowflakeClient, model: V3Model, records: list[dict], run_id: str,
              suffix: str = "") -> datetime:
        """Stage records as gzipped JSON lines in S3 and COPY them into EVENTS_RAW,
        all stamped with one ingested_at (returned)."""
        now = datetime.now(timezone.utc)
        name = f"{run_id}_{suffix}" if suffix else run_id
        key = f"{S3_PREFIX}/{self.name}/table={model.table}/dt={now.date().isoformat()}/{name}.jsonl.gz"
        buf = BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            for rec in records:
                gz.write(_dumps_bytes(rec) + b"\n")
        _s3_client().put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
        sf.execute(f"""
            COPY INTO {self.events} (facility_id, ingested_at, module_source, source_table, namespace, payload)
            FROM (SELECT '{_sq(self.name)}', '{now.isoformat()}'::TIMESTAMP_TZ, '{_sq(model.module)}',
                         '{_sq(model.table)}', '{_sq(model.namespace)}', PARSE_JSON($1)
                  FROM @{SF_STAGE})
            FILES = ('{_sq(key)}') FILE_FORMAT = (FORMAT_NAME = {SF_FILE_FORMAT}) ON_ERROR = 'ABORT_STATEMENT'
        """, label=f"copy:{model.table}")
        return now

    def clear_table(self, sf: SnowflakeClient, model: V3Model) -> None:
        sf.execute(f"DELETE FROM {self.events} WHERE facility_id = '{_sq(self.name)}' "
                   f"AND source_table = '{_sq(model.table)}'", label=f"clear:{model.table}")

    def prune(self, sf: SnowflakeClient, keep: set[str]) -> list[str]:
        """Drop tables no longer in the mappings (renamed / removed V3 tables),
        so flattening only builds views for the mappings' V3 tables."""
        cur = sf._conn.cursor()
        cur.execute(f"SELECT DISTINCT source_table FROM {self.events} WHERE facility_id = %s", (self.name,))
        stale = sorted(r[0] for r in cur.fetchall() if r[0] not in keep)
        if stale:
            in_list = ", ".join(f"'{_sq(t)}'" for t in stale)
            sf.execute(f"DELETE FROM {self.events} WHERE facility_id = '{_sq(self.name)}' "
                       f"AND source_table IN ({in_list})", label="prune")
            log.info("Removed %d table(s) no longer in the mappings: %s", len(stale), ", ".join(stale))
        return stale

    def log_run(self, sf: SnowflakeClient, run_id: str, connection_id: int, source_table: str,
                models: list[V3Model], status: str, rows: int, started: datetime, error: str | None = None) -> None:
        err = "NULL" if not error else f"'{_sq(error[:2000])}'"
        sf.execute(f"""INSERT INTO {self.runs} SELECT '{_sq(run_id)}', {connection_id}, '{_sq(source_table)}',
            '{_sq(",".join(m.table for m in models))}', '{status}', {rows},
            '{started.isoformat()}'::TIMESTAMP_TZ, CURRENT_TIMESTAMP(), {err}""", label=f"runlog:{source_table}")


# ─── PIPELINE ────────────────────────────────────────────────────────────

@dataclass
class TableResult:
    source_table: str
    status: str                       # loaded | empty | dry_run | canary | failed
    source_rows: int = 0
    v3: dict = field(default_factory=dict)   # v3 table -> {"rows", "fields": {field: non-null count}}
    error: str | None = None
    canary: list[dict] = field(default_factory=list)   # canary.TargetResult.to_dict() per V3 table


# Canary: pages read per source table at most while looking for N rows that
# pass each V3 table's "where" filter (500 rows a page).
CANARY_MAX_PAGES = int(os.getenv("CANARY_MAX_PAGES", "20"))


def canary_rows(client: ExtractionClient, source_table: str, models: list[V3Model], size: int,
                max_pages: int = CANARY_MAX_PAGES) -> list[dict]:
    """The first rows of a source table that give every V3 table it feeds
    `size` records — page by page, stopping as soon as they are there (or the
    table / max_pages ends). Secrets are stripped as in a full run."""
    rows: list[dict] = []
    for page in client.iter_pages(source_table, max_pages):
        rows.extend(_strip_secrets(r) for r in page)
        if all(sum(1 for r in rows if m.accepts(r)) >= size for m in models):
            break
    return rows


def verify_canary(wh: Warehouse, sf: SnowflakeClient, model: V3Model, records: list[dict],
                  ingested_at: datetime) -> "canary.TargetResult":
    """Read one canary load back from EVENTS_RAW and compare it with what was
    sent: every record there, every mapped value stored as sent, no record id
    stored twice."""
    import canary

    sent = [json.loads(_dumps_bytes(r)) for r in records]     # the encoder's own spelling of each value
    res = canary.TargetResult(model.table, expected=len(sent), ids=[str(r.get("id")) for r in sent])
    stored = wh.read_back(sf, model, ingested_at)
    by_id = {str(p.get("id")): p for p in stored}
    res.landed = sum(1 for r in sent if str(r.get("id")) in by_id) if by_id else min(len(stored), len(sent))
    for rec in sent:
        rid = str(rec.get("id"))
        got = by_id.get(rid)
        if got is None:
            res.missing.append(rid)
            continue
        mismatched, absent = canary.compare_record(rec, got)
        for f in mismatched + absent:
            res.mismatch(f, rid)
    res.duplicates = wh.duplicate_ids(sf, model, [x for x in res.ids if x != "None"])
    empty = sorted({f["field"] for f in model.fields} - {k for r in sent for k, v in r.items() if v not in (None, "")})
    if empty:
        res.notes.append(f"mapped fields empty in every canary record: {', '.join(empty)}")
    return res


def resolve_name(spec: Mappings, client: ExtractionClient) -> str:
    """The connection's name in the extraction tool (falls back to the name in
    the export if the API can't be reached)."""
    try:
        name = client.connection().get("name")
    except Exception as e:
        log.warning("Couldn't read connection %s from the extraction tool (%s) — using the export's name",
                    spec.connection_id, e)
        name = None
    name = name or spec.exported_name
    if not slug(name):
        raise SystemExit(f"No name for connection {spec.connection_id} — pass --name")
    return name


def process_table(spec: Mappings, wh: Warehouse, client: ExtractionClient, source_table: str, *,
                  run_id: str, dry_run: bool, canary: int = 0) -> TableResult:
    """Extract one source table, map it to its V3 model(s) and land them.

    canary > 0: only the first `canary` records per V3 table, added to the
    table (upsert by record id — nothing else in it is touched), then read
    back and checked; result.canary holds the check per V3 table and a failed
    check makes the result "failed". An empty source never clears the table."""
    import canary as cn

    started = datetime.now(timezone.utc)
    models = spec.models_for(source_table)
    sf = None if dry_run else SnowflakeClient()
    try:
        load_lookups(models, client)
        rows = (canary_rows(client, source_table, models, canary) if canary
                else [_strip_secrets(r) for r in client.fetch_all(source_table)])
        status = "dry_run" if dry_run else "canary" if canary else ("loaded" if rows else "empty")
        result = TableResult(source_table, status, len(rows))
        for m in models:
            records = [m.to_v3(r) for r in rows if m.accepts(r)]
            if canary:
                records = records[:canary]
            fields = sorted({f["field"] for f in m.fields})
            result.v3[m.table] = {"rows": len(records),
                                  "fields": {f: sum(1 for x in records if x.get(f) not in (None, "")) for f in fields}}
            if canary:
                result.canary.append(_canary_target(wh, sf, m, records, run_id, rows).to_dict())
            elif sf is not None:
                if records:
                    wh.replace_table(sf, m, records, run_id)
                else:
                    wh.clear_table(sf, m)          # source is empty now → V3 table empty too
        bad = [t for t in result.canary if t["status"] == cn.FAILED]
        if bad:
            result.status = "failed"
            result.error = "canary: " + "; ".join(cn.TargetResult.from_dict(t).summary() for t in bad)
        if sf is not None:
            wh.log_run(sf, run_id, spec.connection_id, source_table, models,
                       "canary_failed" if canary and bad else result.status, len(rows), started, result.error)
        log.info("  %-32s %6d rows → %s%s", source_table, len(rows), ", ".join(m.table for m in models),
                 f" (canary: {len(bad)} V3 table(s) failed)" if bad else " (canary)" if canary else "")
        return result
    except Exception as e:
        log.error("  %-32s FAILED: %s", source_table, e)
        if sf is not None:
            try:
                wh.log_run(sf, run_id, spec.connection_id, source_table, models, "failed", 0, started, str(e))
            except Exception:
                pass
        return TableResult(source_table, "failed", error=str(e),
                           canary=[cn.TargetResult(m.table, errors=[str(e)[:500]]).to_dict() for m in models]
                           if canary else [])
    finally:
        if sf is not None:
            sf.close()


def _canary_target(wh: Warehouse, sf: SnowflakeClient | None, model: V3Model, records: list[dict],
                   run_id: str, rows: list[dict]) -> "canary.TargetResult":
    """Land one V3 table's canary records and check them — or say why it can't be checked."""
    import canary

    if not records:
        why = ("the source table is empty" if not rows else
               f"none of the first {len(rows)} source rows pass its \"where\" filter")
        return canary.TargetResult(model.table, untested=why)
    if sf is None:
        return canary.TargetResult(model.table, expected=len(records), ids=[str(r.get("id")) for r in records],
                                   untested="dry run — nothing written")
    try:
        ingested_at = wh.upsert_records(sf, model, records, run_id)
        return verify_canary(wh, sf, model, records, ingested_at)
    except Exception as e:
        return canary.TargetResult(model.table, expected=len(records), ids=[str(r.get("id")) for r in records],
                                   errors=[f"{type(e).__name__}: {str(e)[:400]}"])


def canary_report(results: list[TableResult], *, name: str, run_id: str, size: int) -> "canary.CanaryReport":
    """Every V3 table's canary check from the per-source-table results, as one report."""
    import canary

    report = canary.CanaryReport("from_json_mappings_to_snowflake", size, facility=slug(name), run_id=run_id)
    for r in sorted(results, key=lambda r: r.source_table):
        for t in r.canary:
            report.add(canary.TargetResult.from_dict(t))
    return report


def print_report(results: list[TableResult]) -> None:
    print(f"\n  {'V3 table':42s} {'from source table':30s} {'rows':>6s}  mapped fields with data")
    for r in sorted(results, key=lambda r: r.source_table):
        if r.status == "failed":
            print(f"  {'—':42s} {r.source_table:30s} {'FAILED':>6s}  {(r.error or '')[:70]}")
            continue
        for table, info in sorted(r.v3.items()):
            filled = [f for f, n in info["fields"].items() if n]
            empty = [f for f, n in info["fields"].items() if not n]
            note = f"  (no data: {', '.join(empty)})" if info["rows"] and empty else ""
            print(f"  {table:42s} {r.source_table:30s} {info['rows']:>6}  {len(filled)}/{len(info['fields'])}{note}")


def run(mappings: Path | str | dict = DEFAULT_MAPPINGS, *, tables: list[str] | None = None, name: str | None = None,
        dry_run: bool = False, workers: int = 6, per_page: int = 500, page_workers: int = 4,
        canary: int = 0) -> list[TableResult]:
    """mappings: a mappings file path, the mappings JSON text, or the parsed dict.
    canary > 0: a canary run of that many records per V3 table (see process_table);
    its report is printed and saved, and its failures make tables "failed"."""
    spec = Mappings.load(mappings)
    unknown = sorted(set(tables or []) - set(spec.source_tables))
    if unknown:
        raise SystemExit(f"Not in {spec.label}: {unknown}")
    selected = [t for t in spec.source_tables if not tables or t in tables]
    client = ExtractionClient(spec.connection_id, per_page=per_page, page_workers=page_workers)
    wh = Warehouse(name or resolve_name(spec, client))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log.info("connection %s → %s / %s · %d source tables → %d V3 tables · run %s%s%s",
             spec.connection_id, wh.raw, wh.clean, len(selected),
             sum(1 for m in spec.models if m.source_table in selected), run_id, " (dry run)" if dry_run else "",
             f" · CANARY {canary} record(s) per V3 table" if canary else "")
    if not dry_run:
        with SnowflakeClient() as sf:
            wh.ensure(sf)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(process_table, spec, wh, client, t, run_id=run_id, dry_run=dry_run, canary=canary)
                   for t in selected]
        results = [f.result() for f in as_completed(futures)]
    print_report(results)
    if canary:
        report = canary_report(results, name=wh.name, run_id=run_id, size=canary)
        print("\n" + report.summary())
        report.save()
        if not report.ok and not dry_run and not any(r.status == "failed" for r in results):
            # every V3 table untested: nothing failed, but nothing was checked either
            results.append(TableResult("(canary)", "failed", error="the canary could not check any V3 table"))
        return results                      # a canary never prunes: it saw only a few rows per table
    if not dry_run and not tables and not any(r.status == "failed" for r in results):
        with SnowflakeClient() as sf:
            wh.prune(sf, {m.table for m in spec.models})
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mappings", default=str(DEFAULT_MAPPINGS),
                    help="The connection's mappings export: a JSON file path, the JSON itself "
                         "('{\"connection\": …}'), or - to read the JSON from stdin")
    ap.add_argument("--tables", nargs="+", help="Only these source tables")
    ap.add_argument("--name", help="Override the schema name (default: the connection's name in the extraction tool)")
    ap.add_argument("--list", action="store_true", help="Show what the mappings file covers and exit")
    ap.add_argument("--dry-run", action="store_true", help="Extract and map; write nothing")
    ap.add_argument("--canary", type=int, default=0, metavar="N",
                    help="Canary: land only the first N records per V3 table (upsert, nothing else touched), "
                         "read them back and exit 1 if any did not land intact. 0 = full run")
    ap.add_argument("--workers", type=int, default=int(os.getenv("EXTRACTION_WORKERS", "6")), help="Tables in parallel")
    ap.add_argument("--page-workers", type=int, default=int(os.getenv("EXTRACTION_PAGE_WORKERS", "4")))
    ap.add_argument("--per-page", type=int, default=int(os.getenv("EXTRACTION_PER_PAGE", "500")))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s · %(levelname)-7s · %(message)s", datefmt="%H:%M:%S")
    for noisy in ("snowflake.connector", "botocore", "urllib3", "facility_pipeline"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    source = sys.stdin.read() if args.mappings.strip() == "-" else args.mappings
    try:
        spec = Mappings.load(source)
    except MappingsError as e:
        raise SystemExit(f"error: {e}") from None
    if args.list:
        print(f"connection {spec.connection_id} (export name {spec.exported_name!r}) · {spec.label}")
        for m in sorted(spec.models, key=lambda m: (m.source_table, m.table)):
            print(f"  {m.source_table:32s} → {m.table:42s} {m.namespace}  ({len(m.fields)} fields)")
        return
    import canary

    try:
        size = canary.canary_size(args.canary)
    except ValueError as e:
        raise SystemExit(f"error: {e}") from None
    results = run(source, tables=args.tables, name=args.name, dry_run=args.dry_run,
                  workers=args.workers, per_page=args.per_page, page_workers=args.page_workers, canary=size)
    sys.exit(1 if any(r.status == "failed" for r in results) else 0)


if __name__ == "__main__":
    main()
