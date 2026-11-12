#!/usr/bin/env python3
"""
Patient record counts per organisation / facility / region → Snowflake
HOSPITALS.ANALYTICS.AFYA_HMIS_PATIENT_COUNTS.

SOURCE (migrate platform, native count gateway)
-----------------------------------------------
The "patient counts by organisation" endpoint is the Model-Gateway `count`
action with `tenants:"all"` (ModelGateway-Count collection, folders 3 & 5),
served on the MIGRATE platform (`*migrate.afyaanalytics.ai`). A single
platform-statistics account (AFYA_PLATFORM_* in pipelines/.env, key `platform`)
gets exact counts for every organisation and facility in ONE call — no row
scanning and no PII ever leaves the gateway:

    POST core/api/v1/gateway/counts
      { "tenants":"all", "labels":true,
        "queries": { "patients": { "service":"reception", "model":"patient",
                             "group_by":["organization_id","facility_id"] } } }

returns one group per (organization_id, facility_id) with `count` and a `label`
naming both. Summing a facility's org gives the organisation total; a group whose
facility_id is null is the org's "(unassigned)" bucket. (Production does NOT
have the count action — 422/404 — which is why this runs against migrate.)

REGION
------
Patient has no region column (not groupable), and the platform account can't
read facilities (superadmin-only). So region is resolved per organisation using
the per-org superadmin logins already in pipelines/.env (AFYA_<KEY>_* whose
URL_TEMPLATE points at the migrate platform): facility_id → region_id →
regions.name. Orgs we have no superadmin login for get region_id/name = null.

OUTPUT (one timestamped snapshot appended per run; latest captured_at = "now")
  scope 'facility'     : one row per (org, facility) by NAME, with region
  scope 'unassigned'   : patients with no facility, per org
  scope 'organization' : the org total (facility_id null)

USAGE
  python patient_counts_to_snowflake.py                 # all orgs, write
  python patient_counts_to_snowflake.py --dry-run       # compute only
  python patient_counts_to_snowflake.py --no-region     # skip region reads
  python patient_counts_to_snowflake.py --platform platform   # key override
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

log = logging.getLogger("patient_counts")

# ─── CONFIG ────────────────────────────────────────────────────────────────
SF_DATABASE = os.getenv("UPD_RAW_DATABASE", "HOSPITALS")
SF_SCHEMA   = os.getenv("UPD_RAW_SCHEMA",   "ANALYTICS")
SF_TABLE    = os.getenv("UPD_RAW_TABLE",    "AFYA_HMIS_PATIENT_COUNTS")

PLATFORM_KEY  = os.getenv("UPD_RAW_PLATFORM_KEY", "platform")
ENVIRONMENT   = os.getenv("UPD_RAW_ENVIRONMENT", "migrate")
COUNT_SERVICE = "reception"
PATIENT_MODEL = "patient"
VISIT_MODEL   = "visit"
UNASSIGNED    = "(unassigned)"
GROUP_BY      = ["organization_id", "facility_id"]


# ─── NATIVE COUNT (one call, all orgs + facilities) ─────────────────────────
def _groups(result: dict) -> dict[tuple, tuple[int, dict]]:
    """{(org_id, facility_id): (count, label)} from one /gateway/counts query result."""
    out: dict[tuple, tuple[int, dict]] = {}
    for g in result.get("data", {}).get("groups", []):
        key = g.get("key") or {}
        out[(key.get("organization_id"), key.get("facility_id"))] = (int(g.get("count") or 0), g.get("label") or {})
    return out


def _direct_count(client: V3Client, headers: dict, spec: dict) -> dict:
    """One /gateway/counts query sent straight to its service's gateway ({service}/v1/gateway, action count,
    tenants all). Returns a result shaped like a /gateway/counts entry: {success, data:{total, groups}}."""
    spec = dict(spec)
    service = spec.pop("service")
    body = {**spec, "action": "count", "tenants": "all"}
    r = client._session.post(f"{client.urls[service].rstrip('/')}/v1/gateway", headers=headers,
                             data=json.dumps(body).encode(), timeout=180)
    try:
        j = r.json()
    except ValueError:
        j = {"message": r.text[:200]}
    if r.ok and j.get("success", True) and isinstance(j.get("data"), dict):
        return {"success": True, "data": j["data"]}
    return {"success": False, "status": r.status_code, "message": j.get("message")}


def _names(client: V3Client, headers: dict) -> tuple[dict, dict]:
    """({organization_id: name}, {facility_id: name}) from core's own count gateway, which labels grouped
    organization_id / facility_id. Queries that stay inside core work even while core's fan-out to other
    services is broken. Organisations: facilities grouped by organization_id (every org has a facility).
    Facilities: service_destinations grouped by facility_id (every facility has them on PROD), departments
    as a backup. Best-effort: empty dicts if core refuses."""
    body = {"tenants": "all", "labels": True, "queries": {
        "organization_names": {"service": "core", "model": "facilities", "group_by": ["organization_id"], "limit": 1000},
        "facility_names": {"service": "core", "model": "service_destinations", "group_by": ["facility_id"], "limit": 1000},
        "facility_names_departments": {"service": "core", "model": "departments", "group_by": ["facility_id"], "limit": 1000},
    }}
    orgs: dict = {}
    facs: dict = {}
    try:
        r = client._session.post(f"{client.urls['core'].rstrip('/')}/v1/gateway/counts", headers=headers,
                                 data=json.dumps(body).encode(), timeout=180)
        results = (r.json().get("data") or {}).get("results", {}) if r.ok else {}
    except Exception as e:  # noqa: BLE001 - names are best-effort
        log.warning("core name lookup failed: %s", str(e)[:200])
        results = {}
    for q, field, target in (("organization_names", "organization_id", orgs),
                             ("facility_names", "facility_id", facs),
                             ("facility_names_departments", "facility_id", facs)):
        res = results.get(q) or {}
        for g in (res.get("data") or {}).get("groups", []) if res.get("success") else []:
            name = (g.get("label") or {}).get(field)
            key = (g.get("key") or {}).get(field)
            if name and key is not None:
                target.setdefault(key, name)
    log.info("names from core: %d organisations, %d facilities", len(orgs), len(facs))
    return orgs, facs


def fetch_org_facility_counts(client: V3Client) -> list[dict]:
    """Patient and visit counts per (organization_id, facility_id), named, for every org.
    Uses core /gateway/counts with tenants:"all" + labels (one call, three queries).
    Patients are required; the visit queries are best-effort (None if they fail).
    Returns a list of {org_id, org_name, facility_id, facility_name, count,
    visit_count, visits_this_month}."""
    body = {
        "tenants": "all",
        "labels": True,
        "queries": {
            "patients": {"service": COUNT_SERVICE, "model": PATIENT_MODEL, "group_by": GROUP_BY, "limit": 1000},
            "visits": {"service": COUNT_SERVICE, "model": VISIT_MODEL, "group_by": GROUP_BY, "limit": 1000},
            "visits_this_month": {"service": COUNT_SERVICE, "model": VISIT_MODEL, "group_by": GROUP_BY, "limit": 1000,
                                  "where": [{"field": "created_at", "op": "this_month"}]},
        },
    }
    url = f"{client.urls['core'].rstrip('/')}/v1/gateway/counts"
    headers = client._headers("core")
    if client.organization_id is not None:          # the Model Gateway collection sends it on every call
        headers["X-Tenant-Id"] = str(client.organization_id)
    r = client._session.post(url, headers=headers,
                             data=json.dumps(body).encode(), timeout=180)
    if not r.ok:
        raise GatewayError(f"gateway/counts {r.status_code}: {r.text[:300]}")
    results = (r.json().get("data") or {}).get("results", {})
    result = results.get("patients") or {}
    if not result.get("success"):
        # PROD (2026-11-12): core's fan-out to reception is mis-routed ("route api/v1/api/v1/gateway") while
        # reception's own gateway answers - count there directly, then name orgs/facilities from core-only queries.
        log.warning("gateway/counts patients failed (%s) - counting directly on %s's gateway",
                    json.dumps(result)[:200], COUNT_SERVICE)
        results = {q: _direct_count(client, headers, spec) for q, spec in body["queries"].items()}
        result = results["patients"]
        if not result.get("success"):
            raise GatewayError(f"gateway/counts patients failed and direct {COUNT_SERVICE} count failed: "
                               f"{json.dumps(result)[:300]}")
        org_names, fac_names = _names(client, headers)
        for res in results.values():
            for g in (res.get("data") or {}).get("groups", []) if res.get("success") else []:
                key = g.get("key") or {}
                g["label"] = {"organization_id": org_names.get(key.get("organization_id")),
                              "facility_id": fac_names.get(key.get("facility_id"))}
    patients = _groups(result)
    visits: dict[str, dict | None] = {}
    for q in ("visits", "visits_this_month"):
        res = results.get(q) or {}
        if res.get("success"):
            visits[q] = _groups(res)
        else:
            visits[q] = None
            log.warning("gateway/counts %s (visits) failed - visit columns left empty: %s", q, json.dumps(res)[:200])
    v_all, v_month = visits["visits"], visits["visits_this_month"]
    keys = set(patients) | set(v_all or {}) | set(v_month or {})
    out: list[dict] = []
    for k in keys:
        label = (patients.get(k) or (v_all or {}).get(k) or (v_month or {}).get(k) or (0, {}))[1]
        out.append({
            "org_id": k[0],
            "org_name": label.get("organization_id"),
            "facility_id": k[1],
            "facility_name": label.get("facility_id"),
            "count": patients.get(k, (0, {}))[0],
            "visit_count": None if v_all is None else v_all.get(k, (0, {}))[0],
            "visits_this_month": None if v_month is None else v_month.get(k, (0, {}))[0],
        })
    if not out:
        raise GatewayError("gateway/counts returned no groups")
    return out


# ─── REGION (per-org superadmin facility reads) ─────────────────────────────
def migrate_org_key_map() -> dict[int, str]:
    """{organization_id: AFYA_<KEY>} for every per-org superadmin login in the
    env whose URL_TEMPLATE points at the migrate platform."""
    out: dict[int, str] = {}
    for k, v in os.environ.items():
        if k.startswith("AFYA_") and k.endswith("_ORGANIZATION_ID"):
            base = k[len("AFYA_"):-len("_ORGANIZATION_ID")]
            tmpl = os.environ.get(f"AFYA_{base}_URL_TEMPLATE", "")
            if "migrate" in tmpl and os.environ.get(f"AFYA_{base}_USERNAME"):
                try:
                    out[int(v)] = base.lower()
                except ValueError:
                    pass
    return out


def region_map_for_orgs(org_ids: set[int]) -> dict[tuple[int, int], tuple[int | None, str | None]]:
    """(org_id, facility_id) → (region_id, region_name), using each org's own
    superadmin login. Orgs with no migrate login are skipped (region null)."""
    key_map = migrate_org_key_map()
    out: dict[tuple[int, int], tuple[int | None, str | None]] = {}
    for org_id in sorted(org_ids):
        key = key_map.get(org_id)
        if not key:
            continue
        try:
            c = V3Client(key)
            c.login()
            regs = {r.get("id"): r.get("name") for r in c.read_all("core", "regions")}
            for f in c.read_all("core", "facilities"):
                fid, rid = f.get("id"), f.get("region_id")
                if fid is not None:
                    out[(org_id, fid)] = (rid, regs.get(rid))
            log.info("region map: org %s via %s — %d facilities", org_id, key,
                     sum(1 for k in out if k[0] == org_id))
        except Exception as e:  # noqa: BLE001 - region is best-effort
            log.warning("region map: org %s (%s) failed: %s", org_id, key, str(e)[:160])
    return out


# ─── BUILD SNAPSHOT ROWS ────────────────────────────────────────────────────
def build_rows(counts: list[dict], regions: dict[tuple[int, int], tuple[int | None, str | None]],
               captured_at: datetime) -> list[dict]:
    rows: list[dict] = []
    org_totals: dict[int, int] = {}
    org_visits: dict[int, int | None] = {}
    org_visits_month: dict[int, int | None] = {}
    org_names: dict[int, str | None] = {}

    def add(acc: dict, org, v) -> None:              # org total; stays None if any part is unknown
        acc[org] = None if v is None or (org in acc and acc[org] is None) else acc.get(org, 0) + v

    for c in counts:
        org_id = c["org_id"]
        if c["org_name"]:
            org_names[org_id] = c["org_name"]
        else:
            org_names.setdefault(org_id, None)
        org_totals[org_id] = org_totals.get(org_id, 0) + c["count"]
        add(org_visits, org_id, c.get("visit_count"))
        add(org_visits_month, org_id, c.get("visits_this_month"))
        fid = c["facility_id"]
        rid, rname = regions.get((org_id, fid), (None, None)) if fid is not None else (None, None)
        rows.append({
            "ENVIRONMENT": ENVIRONMENT,
            "ORGANIZATION_ID": org_id,
            "ORGANIZATION_NAME": c["org_name"],
            "FACILITY_ID": fid,
            "FACILITY_NAME": (c["facility_name"] if fid is not None else UNASSIGNED),
            "REGION_ID": rid,
            "REGION_NAME": rname,
            "PATIENT_COUNT": c["count"],
            "VISIT_COUNT": c.get("visit_count"),
            "VISITS_THIS_MONTH": c.get("visits_this_month"),
            "SCOPE": "facility" if fid is not None else "unassigned",
            "COUNT_METHOD": "gateway_count",
            "CAPTURED_AT": captured_at,
        })

    for org_id, total in org_totals.items():
        rows.append({
            "ENVIRONMENT": ENVIRONMENT,
            "ORGANIZATION_ID": org_id,
            "ORGANIZATION_NAME": org_names.get(org_id),
            "FACILITY_ID": None,
            "FACILITY_NAME": None,
            "REGION_ID": None,
            "REGION_NAME": None,
            "PATIENT_COUNT": total,
            "VISIT_COUNT": org_visits.get(org_id),
            "VISITS_THIS_MONTH": org_visits_month.get(org_id),
            "SCOPE": "organization",
            "COUNT_METHOD": "gateway_count",
            "CAPTURED_AT": captured_at,
        })
    return rows


# ─── SNOWFLAKE ─────────────────────────────────────────────────────────────
COLUMNS = [
    ("CAPTURED_AT", "TIMESTAMP_NTZ"), ("RUN_ID", "STRING"), ("ENVIRONMENT", "STRING"),
    ("ORGANIZATION_ID", "NUMBER"), ("ORGANIZATION_NAME", "STRING"),
    ("FACILITY_ID", "NUMBER"), ("FACILITY_NAME", "STRING"),
    ("REGION_ID", "NUMBER"), ("REGION_NAME", "STRING"),
    ("PATIENT_COUNT", "NUMBER"), ("SCOPE", "STRING"), ("COUNT_METHOD", "STRING"),
    ("VISIT_COUNT", "NUMBER"), ("VISITS_THIS_MONTH", "NUMBER"),   # added 2026-11-12 (older rows: NULL)
]
_INSERT_ORDER = [c for c, _ in COLUMNS]
ADDED_COLUMNS = [("VISIT_COUNT", "NUMBER"), ("VISITS_THIS_MONTH", "NUMBER")]


def ensure_table(conn) -> None:
    fq = f"{_ident(SF_DATABASE)}.{_ident(SF_SCHEMA)}.{_ident(SF_TABLE)}"
    cols = ", ".join(f"{c} {t}" for c, t in COLUMNS)
    cur = conn.cursor()
    try:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {_ident(SF_DATABASE)}.{_ident(SF_SCHEMA)}")
        cur.execute(f"CREATE TABLE IF NOT EXISTS {fq} ({cols})")
        for col, typ in ADDED_COLUMNS:               # tables created before these columns existed
            cur.execute(f"ALTER TABLE {fq} ADD COLUMN IF NOT EXISTS {col} {typ}")
    finally:
        cur.close()


def write_rows(conn, rows: list[dict], run_id: str) -> int:
    if not rows:
        return 0
    fq = f"{_ident(SF_DATABASE)}.{_ident(SF_SCHEMA)}.{_ident(SF_TABLE)}"
    placeholders = ", ".join(["%s"] * len(_INSERT_ORDER))
    sql = f"INSERT INTO {fq} ({', '.join(_INSERT_ORDER)}) VALUES ({placeholders})"
    data = [tuple({**r, "RUN_ID": run_id}.get(c) for c in _INSERT_ORDER) for r in rows]
    cur = conn.cursor()
    try:
        cur.executemany(sql, data)
    finally:
        cur.close()
    return len(data)


# ─── RUN ───────────────────────────────────────────────────────────────────
def run(*, platform_key: str | None = None, with_region: bool = True,
        dry_run: bool = False, database: str = SF_DATABASE, schema: str = SF_SCHEMA) -> dict:
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    run_id = uuid.uuid4().hex[:16]
    captured_at = datetime.now(timezone.utc).replace(tzinfo=None)

    client = V3Client(platform_key or PLATFORM_KEY)
    client.login()
    counts = fetch_org_facility_counts(client)
    org_ids = {c["org_id"] for c in counts if c["org_id"] is not None}
    total_visits = None if any(c["visit_count"] is None for c in counts) else sum(c["visit_count"] for c in counts)
    log.info("counts: %d org/facility groups across %d organisations (total patients %d, visits %s)",
             len(counts), len(org_ids), sum(c["count"] for c in counts), total_visits)

    regions: dict = {}
    if with_region:
        regions = region_map_for_orgs(org_ids)
        covered = {k[0] for k in regions}
        log.info("region map covers %d/%d organisations", len(covered), len(org_ids))

    rows = build_rows(counts, regions, captured_at)

    written = 0
    if dry_run:
        log.info("[dry-run] would write %d rows to %s.%s.%s", len(rows), database, schema, SF_TABLE)
    else:
        conn = snowflake_connect(database=database, schema=schema)
        try:
            ensure_table(conn)
            written = write_rows(conn, rows, run_id)
            conn.commit()
        finally:
            conn.close()
        log.info("wrote %d rows to %s.%s.%s (run %s)", written, database, schema, SF_TABLE, run_id)

    return {
        "run_id": run_id, "environment": ENVIRONMENT,
        "organisations": len(org_ids), "groups": len(counts), "rows": len(rows),
        "written": written,
        "total_patients": sum(c["count"] for c in counts),
        "total_visits": total_visits,
        "regions_covered": len({k[0] for k in regions}),
    }


def _cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", default=None, help=f"platform account key (default {PLATFORM_KEY} → AFYA_PLATFORM_*)")
    ap.add_argument("--no-region", action="store_true", help="skip per-org region reads")
    ap.add_argument("--dry-run", action="store_true", help="compute but do not write to Snowflake")
    ap.add_argument("--database", default=SF_DATABASE)
    ap.add_argument("--schema", default=SF_SCHEMA)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    summary = run(platform_key=args.platform, with_region=not args.no_region,
                  dry_run=args.dry_run, database=args.database, schema=args.schema)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    _cli()
