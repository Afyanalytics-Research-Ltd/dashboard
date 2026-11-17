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
from datetime import date, datetime, timezone
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
# Visits live in the evaluation service (613k on PROD, incl. migrated history); reception's own
# "visit" model only holds the few visits opened at the V3 front desk since 2026-08.
VISIT_SERVICE = "evaluation"
VISIT_MODEL   = "visits"
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


def _platform_directory(client: V3Client, headers: dict) -> tuple[set, set]:
    """({(organization_id, facility_id)} for EVERY facility, {organization_id} for EVERY organisation) from
    core's own count gateway. The patient counts only return groups that have patients, so without this a
    facility (or organisation) with no patients yet would be missing from the snapshot. Best-effort."""
    facilities, orgs = set(), set()
    res = _direct_count(client, headers, {"service": "core", "model": "facilities", "limit": 1000,
                                          "group_by": ["organization_id", "id"]})
    if res.get("success"):
        if res["data"].get("truncated"):
            log.warning("facility directory truncated at 1000 - some empty facilities may be missing")
        for g in res["data"].get("groups", []):
            k = g.get("key") or {}
            if k.get("id") is not None:
                facilities.add((k.get("organization_id"), k.get("id")))
    else:
        log.warning("facility directory unavailable: %s %s", res.get("status"), res.get("message"))
    res = _direct_count(client, headers, {"service": "core", "model": "organizations", "limit": 1000, "group_by": ["id"]})
    if res.get("success"):
        orgs = {(g.get("key") or {}).get("id") for g in res["data"].get("groups", [])} - {None}
    else:
        log.warning("organisation directory unavailable: %s %s", res.get("status"), res.get("message"))
    log.info("platform directory: %d facilities, %d organisations", len(facilities), len(orgs))
    return facilities, orgs


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
            "visits": {"service": VISIT_SERVICE, "model": VISIT_MODEL, "group_by": GROUP_BY, "limit": 1000},
            "visits_this_month": {"service": VISIT_SERVICE, "model": VISIT_MODEL, "group_by": GROUP_BY, "limit": 1000,
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

    # Every facility and organisation on the platform gets a row, not only those with patients:
    # facilities with none get PATIENT_COUNT 0; an organisation with no facility gets an org total of 0.
    directory, all_orgs = _platform_directory(client, headers)
    org_names, fac_names = _names(client, headers)
    for row in out:                                  # fill names core's labels didn't give
        row["org_name"] = row["org_name"] or org_names.get(row["org_id"])
        if row["facility_id"] is not None:
            row["facility_name"] = row["facility_name"] or fac_names.get(row["facility_id"])
    zero_visits = lambda v: None if v is None else 0
    counted_facilities = {fac for _, fac in keys if fac is not None}   # facility ids are platform-wide
    empty = sorted((k for k in directory if k[1] not in counted_facilities), key=lambda k: (str(k[0]), str(k[1])))
    for org, fac in empty:
        out.append({"org_id": org, "org_name": org_names.get(org), "facility_id": fac, "facility_name": fac_names.get(fac),
                    "count": 0, "visit_count": zero_visits(v_all), "visits_this_month": zero_visits(v_month)})
    seen_orgs = {row["org_id"] for row in out}
    no_facility = sorted(all_orgs - seen_orgs, key=str)
    for org in no_facility:
        out.append({"org_id": org, "org_name": org_names.get(org), "facility_id": None, "facility_name": None,
                    "count": 0, "visit_count": zero_visits(v_all), "visits_this_month": zero_visits(v_month),
                    "org_only": True})
    log.info("rows: %d with data, %d empty facilities, %d organisations without facilities",
             len(keys), len(empty), len(no_facility))
    return out


# ─── MONTHLY HISTORY (new records per month since the first one) ────────────
# The hourly snapshots only start when this pipeline started; the count gateway can also bucket
# records by the month they were created, which gives the whole history. Summing those months in
# order = the running total at the end of each month (deleted records excluded, as in the snapshot).
# NB: created_at is when the record entered V3 - migrated patients count in their migration month.
HISTORY_TABLE = os.getenv("UPD_RAW_HISTORY_TABLE", "AFYA_HMIS_PATIENT_REGISTRATIONS_MONTHLY")
MONTH_KEY = "created_at:month"


def _gateway_headers(client: V3Client) -> dict:
    headers = client._headers("core")
    if client.organization_id is not None:          # the Model Gateway collection sends it on every call
        headers["X-Tenant-Id"] = str(client.organization_id)
    return headers


def _add_months(d: date, n: int) -> date:
    y, m = divmod(d.month - 1 + n, 12)
    return date(d.year + y, m + 1, 1)


def _monthly_groups(client: V3Client, headers: dict, service: str, model: str, start: date, end: date) -> dict:
    """{(org_id, facility_id, 'YYYY-MM'): count} for `model` records created in [start, end). The gateway
    returns at most 1000 groups, so a truncated window is split in two and asked again."""
    spec = {"service": service, "model": model, "limit": 1000,
            "group_by": GROUP_BY + [{"field": "created_at", "bucket": "month"}],
            "where": [{"field": "created_at", "op": ">=", "value": start.isoformat()},
                      {"field": "created_at", "op": "<", "value": end.isoformat()}]}
    res = _direct_count(client, headers, spec)
    if not res.get("success"):
        raise GatewayError(f"{model} per month {start}..{end}: {res.get('status')} {res.get('message')}")
    data = res["data"]
    if data.get("truncated"):
        months = (end.year - start.year) * 12 + end.month - start.month
        if months <= 1:
            raise GatewayError(f"{model} per month {start}: more than 1000 org/facility groups in one month")
        mid = _add_months(start, months // 2)
        return {**_monthly_groups(client, headers, service, model, start, mid),
                **_monthly_groups(client, headers, service, model, mid, end)}
    out = {}
    for g in data.get("groups", []):
        k = g.get("key") or {}
        out[(k.get("organization_id"), k.get("facility_id"), k.get(MONTH_KEY))] = int(g.get("count") or 0)
    return out


def fetch_monthly_history(client: V3Client, org_names: dict, fac_names: dict) -> list[dict]:
    """New patients (required) and visits (best-effort) per organisation / facility / month, from the
    first year that has patients up to the current month, asked one calendar year at a time."""
    headers = _gateway_headers(client)
    years_res = _direct_count(client, headers, {"service": COUNT_SERVICE, "model": PATIENT_MODEL, "limit": 1000,
                                                "group_by": [{"field": "created_at", "bucket": "year"}]})
    if not years_res.get("success"):
        raise GatewayError(f"patients per year: {years_res.get('status')} {years_res.get('message')}")
    years = sorted(int(y) for g in years_res["data"].get("groups", [])
                   if (y := (g.get("key") or {}).get("created_at:year")))
    if not years:
        return []
    today = datetime.now(timezone.utc).date()
    end_all = _add_months(date(today.year, today.month, 1), 1)
    patients: dict = {}
    visits: dict | None = {}
    for y in range(years[0], today.year + 1):
        start, end = date(y, 1, 1), min(date(y + 1, 1, 1), end_all)
        patients.update(_monthly_groups(client, headers, COUNT_SERVICE, PATIENT_MODEL, start, end))
        if visits is not None:
            try:
                visits.update(_monthly_groups(client, headers, VISIT_SERVICE, VISIT_MODEL, start, end))
            except GatewayError as e:
                log.warning("visit history failed - NEW_VISITS left empty: %s", str(e)[:200])
                visits = None
    rows = []
    for (org, fac, month) in sorted(set(patients) | set(visits or {}), key=lambda k: (str(k[2]), str(k[0]), str(k[1]))):
        if not month:                                   # records with no created_at: no month to put them in
            continue
        rows.append({
            "ENVIRONMENT": ENVIRONMENT,
            "MONTH": date(int(month[:4]), int(month[5:7]), 1),
            "ORGANIZATION_ID": org, "ORGANIZATION_NAME": org_names.get(org),
            "FACILITY_ID": fac, "FACILITY_NAME": fac_names.get(fac) if fac is not None else UNASSIGNED,
            "NEW_PATIENTS": patients.get((org, fac, month), 0),
            "NEW_VISITS": None if visits is None else visits.get((org, fac, month), 0),
        })
    log.info("monthly history: %d rows, %s .. %s, %d patients", len(rows),
             rows[0]["MONTH"] if rows else None, rows[-1]["MONTH"] if rows else None,
             sum(r["NEW_PATIENTS"] for r in rows))
    return rows


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
        if c.get("org_only"):                        # an organisation with no facility: org total row only
            continue
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


HISTORY_COLUMNS = [
    ("ENVIRONMENT", "STRING"), ("MONTH", "DATE"),
    ("ORGANIZATION_ID", "NUMBER"), ("ORGANIZATION_NAME", "STRING"),
    ("FACILITY_ID", "NUMBER"), ("FACILITY_NAME", "STRING"),
    ("NEW_PATIENTS", "NUMBER"), ("NEW_VISITS", "NUMBER"),
    ("REFRESHED_AT", "TIMESTAMP_NTZ"), ("RUN_ID", "STRING"),
]


def write_history(conn, rows: list[dict], run_id: str, refreshed_at: datetime) -> int:
    """Replace this environment's monthly history in one transaction (it is recomputed from source each
    run; the hourly snapshot table is append-only and never touched here)."""
    fq = f"{_ident(SF_DATABASE)}.{_ident(SF_SCHEMA)}.{_ident(HISTORY_TABLE)}"
    order = [c for c, _ in HISTORY_COLUMNS]
    cur = conn.cursor()
    try:
        cur.execute(f"CREATE TABLE IF NOT EXISTS {fq} ({', '.join(f'{c} {t}' for c, t in HISTORY_COLUMNS)})")
        cur.execute("BEGIN")
        cur.execute(f"DELETE FROM {fq} WHERE ENVIRONMENT = %s", (ENVIRONMENT,))
        if rows:
            cur.executemany(f"INSERT INTO {fq} ({', '.join(order)}) VALUES ({', '.join(['%s'] * len(order))})",
                            [tuple({**r, "RUN_ID": run_id, "REFRESHED_AT": refreshed_at}.get(c) for c in order) for r in rows])
        cur.execute("COMMIT")
    except Exception:
        cur.execute("ROLLBACK")
        raise
    finally:
        cur.close()
    return len(rows)


def ensure_views(conn) -> None:
    """Plot-ready views (recreated each run, cheap):
      <SF_TABLE>_DAILY             last snapshot of each day + increase since the previous day
      AFYA_HMIS_PATIENT_GROWTH_MONTHLY        per org/facility: new + running-total patients/visits per month
      AFYA_HMIS_PATIENT_GROWTH_MONTHLY_TOTAL  whole platform: new + running-total patients/visits per month"""
    sch = f"{_ident(SF_DATABASE)}.{_ident(SF_SCHEMA)}"
    snap, hist = f"{sch}.{_ident(SF_TABLE)}", f"{sch}.{_ident(HISTORY_TABLE)}"
    part = "s.ENVIRONMENT, s.SCOPE, s.ORGANIZATION_ID, s.FACILITY_ID"
    running = "OVER (PARTITION BY ENVIRONMENT, ORGANIZATION_ID, FACILITY_ID ORDER BY MONTH ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)"
    statements = [
        f"""CREATE OR REPLACE VIEW {sch}.{_ident(SF_TABLE + '_DAILY')} AS
            WITH last_run AS (SELECT ENVIRONMENT, TO_DATE(CAPTURED_AT) AS DAY, MAX(CAPTURED_AT) AS CAPTURED_AT
                              FROM {snap} GROUP BY 1, 2)
            SELECT s.ENVIRONMENT, l.DAY, s.CAPTURED_AT, s.SCOPE, s.ORGANIZATION_ID, s.ORGANIZATION_NAME,
                   s.FACILITY_ID, s.FACILITY_NAME, s.PATIENT_COUNT,
                   s.PATIENT_COUNT - LAG(s.PATIENT_COUNT) OVER (PARTITION BY {part} ORDER BY l.DAY) AS PATIENT_INCREASE,
                   s.VISIT_COUNT,
                   s.VISIT_COUNT - LAG(s.VISIT_COUNT) OVER (PARTITION BY {part} ORDER BY l.DAY) AS VISIT_INCREASE
            FROM {snap} s JOIN last_run l ON s.ENVIRONMENT = l.ENVIRONMENT AND s.CAPTURED_AT = l.CAPTURED_AT""",
        f"""CREATE OR REPLACE VIEW {sch}.AFYA_HMIS_PATIENT_GROWTH_MONTHLY AS
            SELECT ENVIRONMENT, MONTH, ORGANIZATION_ID, ORGANIZATION_NAME, FACILITY_ID, FACILITY_NAME,
                   NEW_PATIENTS, SUM(NEW_PATIENTS) {running} AS CUMULATIVE_PATIENTS,
                   NEW_VISITS, SUM(NEW_VISITS) {running} AS CUMULATIVE_VISITS, REFRESHED_AT
            FROM {hist}""",
        f"""CREATE OR REPLACE VIEW {sch}.AFYA_HMIS_PATIENT_GROWTH_MONTHLY_TOTAL AS
            SELECT ENVIRONMENT, MONTH, SUM(NEW_PATIENTS) AS NEW_PATIENTS,
                   SUM(SUM(NEW_PATIENTS)) OVER (PARTITION BY ENVIRONMENT ORDER BY MONTH) AS CUMULATIVE_PATIENTS,
                   SUM(NEW_VISITS) AS NEW_VISITS,
                   SUM(SUM(NEW_VISITS)) OVER (PARTITION BY ENVIRONMENT ORDER BY MONTH) AS CUMULATIVE_VISITS,
                   MAX(REFRESHED_AT) AS REFRESHED_AT
            FROM {hist} GROUP BY ENVIRONMENT, MONTH""",
    ]
    cur = conn.cursor()
    try:
        for sql in statements:
            cur.execute(sql)
    finally:
        cur.close()


# ─── RUN ───────────────────────────────────────────────────────────────────
def run(*, platform_key: str | None = None, with_region: bool = True, with_history: bool = True,
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

    # history is best-effort: a failure there never costs the hourly snapshot
    history: list[dict] = []
    history_error = None
    if with_history:
        try:
            org_names = {c["org_id"]: c["org_name"] for c in counts if c["org_name"]}
            fac_names = {c["facility_id"]: c["facility_name"] for c in counts if c["facility_id"] is not None and c["facility_name"]}
            history = fetch_monthly_history(client, org_names, fac_names)
        except Exception as e:  # noqa: BLE001
            history_error = f"{type(e).__name__}: {str(e)[:200]}"
            log.warning("monthly history skipped: %s", history_error)

    written = history_written = 0
    if dry_run:
        log.info("[dry-run] would write %d rows to %s.%s.%s and %d to %s", len(rows), database, schema, SF_TABLE,
                 len(history), HISTORY_TABLE)
    else:
        conn = snowflake_connect(database=database, schema=schema)
        try:
            ensure_table(conn)
            written = write_rows(conn, rows, run_id)
            conn.commit()
            log.info("wrote %d rows to %s.%s.%s (run %s)", written, database, schema, SF_TABLE, run_id)
            if history:
                history_written = write_history(conn, history, run_id, captured_at)
                log.info("refreshed %d rows in %s.%s.%s", history_written, database, schema, HISTORY_TABLE)
            try:
                ensure_views(conn)
            except Exception as e:  # noqa: BLE001 - views are a convenience
                log.warning("could not (re)create views: %s", str(e)[:200])
        finally:
            conn.close()

    return {
        "run_id": run_id, "environment": ENVIRONMENT,
        "organisations": len(org_ids), "groups": len(counts), "rows": len(rows),
        "written": written,
        "total_patients": sum(c["count"] for c in counts),
        "total_visits": total_visits,
        "regions_covered": len({k[0] for k in regions}),
        "history_rows": len(history), "history_written": history_written, "history_error": history_error,
    }


def _cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", default=None, help=f"platform account key (default {PLATFORM_KEY} → AFYA_PLATFORM_*)")
    ap.add_argument("--no-region", action="store_true", help="skip per-org region reads")
    ap.add_argument("--no-history", action="store_true", help="skip the monthly history refresh")
    ap.add_argument("--dry-run", action="store_true", help="compute but do not write to Snowflake")
    ap.add_argument("--database", default=SF_DATABASE)
    ap.add_argument("--schema", default=SF_SCHEMA)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    summary = run(platform_key=args.platform, with_region=not args.no_region, with_history=not args.no_history,
                  dry_run=args.dry_run, database=args.database, schema=args.schema)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    _cli()
