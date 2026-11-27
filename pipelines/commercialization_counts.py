#!/usr/bin/env python3
"""
Commercialization counts → the warehouse (Databend), database COMMERCIALIZATION.

Every run sends ONE call to the V3 core count gateway with the queries the
commercialization team asked for (Mercy, 2026-11-27):

    POST {core}/v1/gateway/counts
      { "tenants": "all", "labels": true, "queries": {
          "patients_by_county": reception/patient          group_by [county, facility_id]
          "diseases":           evaluation/visit_icd10_diagnoses
                                                         group_by [facility_id, icd10_variation_id]
                                                         distinct patient_id
          "notes":              evaluation/doctor_notes   group_by [facility_id], distinct patient_id } }

No rows and no PII leave the gateway — only counts per group, with the names
the gateway labels them with.

OUTPUT  (HOSPITALS.COMMERCIALIZATION → Databend database COMMERCIALIZATION)
  PATIENTS_BY_COUNTY        COUNTY, FACILITY_ID, FACILITY_NAME, PATIENT_COUNT
  DISEASES_BY_FACILITY      FACILITY_ID, FACILITY_NAME, ICD10_VARIATION_ID, ICD10_NAME, DISTINCT_PATIENTS
  DOCTOR_NOTES_BY_FACILITY  FACILITY_ID, FACILITY_NAME, DISTINCT_PATIENTS
  each also: RUN_ID, CAPTURED_AT (UTC), ENVIRONMENT, LABELS (every label the gateway returned, JSON)
  - the plain tables are append-only snapshots (one per run): history / trends
  - LATEST_<table> holds only the newest run (rewritten each run) - a TABLE, not a view, so
    tools like DBeaver read its columns instantly (Databend compiles a view to describe it)

ACCESS
  tenants:"all" needs a V3 account with PLATFORM STATISTICS access (granted on the V3 server).
  An ordinary organisation login gets 403 "Counting across organizations (tenants) needs platform
  statistics access" - the run then stops with that message (CountAccessError).
  One query failing (e.g. a model missing on that platform) doesn't stop the others; it is reported
  in the summary, and the run fails only when every query failed.

USAGE
  python commercialization_counts.py --platform prod [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv

from v3_to_v3_client import V3Client, GatewayError
from v3_to_snowflake_sync import snowflake_connect, _ident

log = logging.getLogger("commercialization_counts")

DATABASE = os.getenv("COMMERCIALIZATION_DATABASE", "HOSPITALS")
SCHEMA = os.getenv("COMMERCIALIZATION_SCHEMA", "COMMERCIALIZATION")
ENVIRONMENT = os.getenv("UPD_RAW_ENVIRONMENT", "prod")
LIMIT = 1000

# query name -> (gateway spec, table, count column)
QUERIES: dict[str, dict] = {
    "patients_by_county": {
        "spec": {"service": "reception", "model": "patient", "group_by": ["county", "facility_id"], "limit": LIMIT},
        "table": "PATIENTS_BY_COUNTY", "count_column": "PATIENT_COUNT",
    },
    "diseases": {
        "spec": {"service": "evaluation", "model": "visit_icd10_diagnoses",
                 "group_by": ["facility_id", "icd10_variation_id"], "distinct": "patient_id", "labels": True,
                 "limit": LIMIT},
        "table": "DISEASES_BY_FACILITY", "count_column": "DISTINCT_PATIENTS",
    },
    "notes": {
        "spec": {"service": "evaluation", "model": "doctor_notes", "group_by": ["facility_id"], "distinct": "patient_id"},
        "table": "DOCTOR_NOTES_BY_FACILITY", "count_column": "DISTINCT_PATIENTS",
    },
}

# table columns per group-by field: (value column, value type, label column or None)
FIELD_COLUMNS = {
    "county": ("COUNTY", "STRING", None),
    "facility_id": ("FACILITY_ID", "NUMBER", "FACILITY_NAME"),
    "icd10_variation_id": ("ICD10_VARIATION_ID", "NUMBER", "ICD10_NAME"),
}
COMMON_COLUMNS = [("RUN_ID", "STRING"), ("CAPTURED_AT", "TIMESTAMP_NTZ"), ("ENVIRONMENT", "STRING")]


class CountAccessError(RuntimeError):
    """The account can't count across organisations (no platform statistics access)."""


def request_body() -> dict:
    return {"tenants": "all", "labels": True, "queries": {q: dict(d["spec"]) for q, d in QUERIES.items()}}


def table_columns(query: str) -> list[tuple[str, str]]:
    d = QUERIES[query]
    cols = list(COMMON_COLUMNS)
    for f in d["spec"]["group_by"]:
        col, typ, label = FIELD_COLUMNS.get(f, (_ident(f), "STRING", None))
        cols.append((col, typ))
        if label:
            cols.append((label, "STRING"))
    cols += [(d["count_column"], "NUMBER"), ("LABELS", "STRING")]
    return cols


def _count_of(group: dict) -> int:
    for k in ("count", "distinct", "distinct_count", "value", "total"):
        v = group.get(k)
        if isinstance(v, (int, float)) or (isinstance(v, str) and v.strip().lstrip("-").isdigit()):
            return int(v)
    return 0


def _as_number(v):
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def rows_for(query: str, result: dict) -> list[dict]:
    """Rows for one successful query result ({data: {groups: [{key, label, count}]}})."""
    d = QUERIES[query]
    out = []
    for g in (result.get("data") or {}).get("groups", []):
        key, label = g.get("key") or {}, g.get("label") or {}
        row = {d["count_column"]: _count_of(g), "LABELS": json.dumps(label, sort_keys=True, default=str) if label else None}
        for f in d["spec"]["group_by"]:
            col, typ, label_col = FIELD_COLUMNS.get(f, (_ident(f), "STRING", None))
            v = key.get(f)
            row[col] = _as_number(v) if typ == "NUMBER" else (None if v is None else str(v))
            if label_col:
                lv = label.get(f)
                row[label_col] = lv if isinstance(lv, str) or lv is None else json.dumps(lv, default=str)
        out.append(row)
    return out


def fetch(client: V3Client) -> dict[str, dict]:
    """{query: result} from one /gateway/counts call. Raises CountAccessError when the account may not count
    across organisations, GatewayError for any other refusal of the whole call."""
    url = f"{client.urls['core'].rstrip('/')}/v1/gateway/counts"
    headers = client._headers("core")
    if client.organization_id is not None:          # the Model Gateway collection sends it on every call
        headers["X-Tenant-Id"] = str(client.organization_id)
    r = client._session.post(url, headers=headers, data=json.dumps(request_body()).encode(), timeout=300)
    try:
        j = r.json()
    except ValueError:
        j = {"message": r.text[:300]}
    message = str(j.get("message") or "")
    if r.status_code == 403 or ("platform statistics" in message.lower()):
        raise CountAccessError(
            f"V3 refused to count across organisations ({r.status_code}): {message or r.text[:200]}. The platform "
            f"account needs PLATFORM STATISTICS access on the V3 server (store it as Airflow Connection "
            f"afya_v3_<key>).")
    if not r.ok or not isinstance(j.get("data"), dict):
        raise GatewayError(f"gateway/counts {r.status_code}: {message or r.text[:300]}")
    results = j["data"].get("results") or {}
    # PROD (2026-11-12): core's fan-out to another service can be mis-routed while that service's own gateway
    # answers - retry a failed query there directly (same as patient_counts_to_snowflake).
    from patient_counts_to_snowflake import _direct_count
    for q, d in QUERIES.items():
        res = results.get(q) or {}
        if not res.get("success"):
            direct = _direct_count(client, headers, {**d["spec"], "labels": True})
            if direct.get("success"):
                log.warning("%s: core gateway failed (%s) - counted on %s's own gateway", q,
                            json.dumps(res)[:200], d["spec"]["service"])
                results[q] = direct
            else:
                results[q] = res or direct
    return results


def ensure_tables(conn) -> None:
    sch = f"{_ident(DATABASE)}.{_ident(SCHEMA)}"
    cur = conn.cursor()
    try:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {sch}")
        for q, d in QUERIES.items():
            cols = ", ".join(f"{c} {t}" for c, t in table_columns(q))
            for t in (d["table"], "LATEST_" + d["table"]):
                cur.execute(f"CREATE TABLE IF NOT EXISTS {sch}.{t} ({cols})")
    finally:
        cur.close()


def write(conn, query: str, rows: list[dict], run_id: str, captured_at: datetime) -> int:
    sch = f"{_ident(DATABASE)}.{_ident(SCHEMA)}"
    d = QUERIES[query]
    order = [c for c, _ in table_columns(query)]
    data = [tuple({**r, "RUN_ID": run_id, "CAPTURED_AT": captured_at, "ENVIRONMENT": ENVIRONMENT}.get(c) for c in order)
            for r in rows]
    sql = f"INSERT INTO {sch}.{{t}} ({', '.join(order)}) VALUES ({', '.join(['%s'] * len(order))})"
    cur = conn.cursor()
    try:
        if data:
            cur.executemany(sql.format(t=d["table"]), data)
        # LATEST_: only the newest run (this environment)
        cur.execute(f"DELETE FROM {sch}.LATEST_{d['table']} WHERE ENVIRONMENT = %s", (ENVIRONMENT,))
        if data:
            cur.executemany(sql.format(t="LATEST_" + d["table"]), data)
    finally:
        cur.close()
    return len(data)


def run(*, platform_key: str = "prod", dry_run: bool = False) -> dict:
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    run_id = uuid.uuid4().hex[:16]
    captured_at = datetime.now(timezone.utc).replace(tzinfo=None)
    client = V3Client(platform_key)
    client.login()
    results = fetch(client)

    per_query: dict[str, dict] = {}
    rows_by_query: dict[str, list[dict]] = {}
    for q in QUERIES:
        res = results.get(q) or {}
        if not res.get("success"):
            per_query[q] = {"ok": False, "error": str(res.get("message") or res.get("errors") or res)[:300]}
            log.error("%s failed: %s", q, per_query[q]["error"])
            continue
        data = res.get("data") or {}
        rows_by_query[q] = rows_for(q, res)
        per_query[q] = {"ok": True, "groups": len(rows_by_query[q]), "total": data.get("total"),
                        "truncated": bool(data.get("truncated"))}
        if data.get("truncated"):
            log.warning("%s hit the %d-group limit (groups_total=%s): some groups are missing", q, LIMIT,
                        data.get("groups_total"))
        log.info("%s: %d groups, total %s", q, len(rows_by_query[q]), data.get("total"))

    written = {}
    if dry_run:
        log.info("[dry-run] would write %s", {QUERIES[q]["table"]: len(r) for q, r in rows_by_query.items()})
    elif rows_by_query:
        conn = snowflake_connect(database=DATABASE, schema=SCHEMA)
        try:
            ensure_tables(conn)
            for q, rows in rows_by_query.items():
                written[QUERIES[q]["table"]] = write(conn, q, rows, run_id, captured_at)
            conn.commit()
        finally:
            conn.close()
        log.info("wrote %s to %s.%s (run %s)", written, DATABASE, SCHEMA, run_id)

    failed = [q for q, s in per_query.items() if not s["ok"]]
    summary = {"run_id": run_id, "environment": ENVIRONMENT, "captured_at": captured_at.isoformat(),
               "queries": per_query, "written": written, "failed": failed}
    if failed and len(failed) == len(QUERIES):
        raise GatewayError(f"every count query failed: {json.dumps(per_query)[:600]}")
    return summary


def _cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", default="prod", help="platform account key K (AFYA_<K>_*)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(json.dumps(run(platform_key=args.platform, dry_run=args.dry_run), indent=2, default=str))


if __name__ == "__main__":
    _cli()
