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
        "queries": { "pf": { "service":"reception", "model":"patient",
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
UNASSIGNED    = "(unassigned)"


# ─── NATIVE COUNT (one call, all orgs + facilities) ─────────────────────────
def fetch_org_facility_counts(client: V3Client) -> list[dict]:
    """Patient count per (organization_id, facility_id), named, for every org.
    Uses core /gateway/counts with tenants:"all" + labels. Returns a list of
    {org_id, org_name, facility_id, facility_name, count}."""
    body = {
        "tenants": "all",
        "labels": True,
        "queries": {"pf": {"service": COUNT_SERVICE, "model": PATIENT_MODEL,
                            "group_by": ["organization_id", "facility_id"]}},
    }
    url = f"{client.urls['core'].rstrip('/')}/v1/gateway/counts"
    r = client._session.post(url, headers=client._headers("core"),
                             data=json.dumps(body).encode(), timeout=180)
    if not r.ok:
        raise GatewayError(f"gateway/counts {r.status_code}: {r.text[:300]}")
    payload = r.json()
    result = (payload.get("data") or {}).get("results", {}).get("pf") or {}
    if not result.get("success"):
        raise GatewayError(f"gateway/counts pf failed: {json.dumps(result)[:300]}")
    out: list[dict] = []
    for g in result.get("data", {}).get("groups", []):
        key = g.get("key") or {}
        label = g.get("label") or {}
        out.append({
            "org_id": key.get("organization_id"),
            "org_name": label.get("organization_id"),
            "facility_id": key.get("facility_id"),
            "facility_name": label.get("facility_id"),
            "count": int(g.get("count") or 0),
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
    org_names: dict[int, str | None] = {}

    for c in counts:
        org_id = c["org_id"]
        org_names.setdefault(org_id, c["org_name"])
        org_totals[org_id] = org_totals.get(org_id, 0) + c["count"]
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
]
_INSERT_ORDER = [c for c, _ in COLUMNS]


def ensure_table(conn) -> None:
    fq = f"{_ident(SF_DATABASE)}.{_ident(SF_SCHEMA)}.{_ident(SF_TABLE)}"
    cols = ", ".join(f"{c} {t}" for c, t in COLUMNS)
    cur = conn.cursor()
    try:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {_ident(SF_DATABASE)}.{_ident(SF_SCHEMA)}")
        cur.execute(f"CREATE TABLE IF NOT EXISTS {fq} ({cols})")
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
    log.info("counts: %d org/facility groups across %d organisations (total patients %d)",
             len(counts), len(org_ids), sum(c["count"] for c in counts))

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
