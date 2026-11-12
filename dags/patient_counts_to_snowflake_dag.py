# dags/patient_counts_to_snowflake_dag.py
"""
Patient record counts per organisation / facility / region → HOSPITALS.ANALYTICS.AFYA_HMIS_PATIENT_COUNTS.

Every 15 minutes, one call to the platform's native count gateway
(core /gateway/counts, tenants:"all", group_by organization_id+facility_id,
labels) counts every organisation and facility exactly — no row scanning, no
PII. Region is resolved per organisation from the per-org superadmin logins in
pipelines/.env (facility_id → region_id → regions.name); orgs without a login
get a null region. A timestamped snapshot is appended, so the latest
captured_at is "now".

Platform account (key `prod` = PROD V3 by default since 2026-11-12, `platform` =
MIGRATE; the "platform" param picks another key K). Region is only resolved for
MIGRATE (it uses per-org MIGRATE logins). The first source that has it wins, and Airflow values REPLACE .env:

  1. Connection  afya_v3_<k>   (or the "connection" param)
       Login / Password       the platform account
       Host                   URL template, e.g. https://{service}migrate.afyaanalytics.ai/api/
                              (or Extra "url_template")
       Extra (optional)       {"organization_id": 1, "facility_id": 1,
                               "migration_key": "...", "urls": {"core": "https://…/api/"},
                               "environment": "migrate"}
  2. Variables   AFYA_<K>_USERNAME, AFYA_<K>_PASSWORD, AFYA_<K>_URL_TEMPLATE
                 (+ optional AFYA_<K>_ORGANIZATION_ID / _FACILITY_ID / _MIGRATION_KEY)
  3. pipelines/.env  AFYA_<K>_*  (previous behaviour)

The rows' ENVIRONMENT label is Extra "environment", else derived from the URL
(migrate / staging / prod). Only MIGRATE has the count gateway today.

See pipelines/patient_counts_to_snowflake.py.

    load  (single task: count → region → write)
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from airflow.sdk import Param, dag, task

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for pipelines_common
from pipelines_common import _connection, _variable, load_airflow_config, use_pipelines_dir

log = logging.getLogger(__name__)

DAG_ID = "patient_counts_to_snowflake"
DEFAULT_KEY = "prod"          # PROD V3 (Connection afya_v3_prod) since 2026-11-12; "platform" = MIGRATE
_ACCOUNT_FIELDS = ("USERNAME", "PASSWORD", "URL_TEMPLATE", "ORGANIZATION_ID", "FACILITY_ID", "MIGRATION_KEY")


def _environment_label(url_template: str) -> str:
    u = (url_template or "").lower()
    if "migrate" in u:
        return "migrate"
    if "staging" in u:
        return "staging"
    return "prod"


def apply_platform_account(key: str, conn_id: str | None = None) -> str:
    """Put the platform account for `key` into AFYA_<KEY>_* from Airflow (Connection, then Variables),
    replacing any .env values so the two never mix. Returns where it came from."""
    up = key.upper()
    conn_id = conn_id or f"afya_v3_{key}"
    conn = _connection(conn_id)
    has_vars = bool(_variable(f"AFYA_{up}_USERNAME"))
    if conn is None and not has_vars:
        return "pipelines/.env"

    # Airflow has this account: drop the .env copy first (also per-service *_URL overrides).
    for name in list(os.environ):
        if name.startswith(f"AFYA_{up}_") and (name[len(f"AFYA_{up}_"):] in _ACCOUNT_FIELDS or name.endswith("_URL")):
            os.environ.pop(name)
    os.environ.pop("UPD_RAW_ENVIRONMENT", None)

    if conn is not None:
        extra = conn.extra_dejson or {}
        template = (extra.get("url_template") or conn.host or "").strip()
        values = {"USERNAME": conn.login, "PASSWORD": conn.password, "URL_TEMPLATE": template,
                  "ORGANIZATION_ID": extra.get("organization_id"), "FACILITY_ID": extra.get("facility_id"),
                  "MIGRATION_KEY": extra.get("migration_key")}
        for field, value in values.items():
            if value not in (None, ""):
                os.environ[f"AFYA_{up}_{field}"] = str(value).strip()
        for svc, url in (extra.get("urls") or {}).items():
            os.environ[f"AFYA_{up}_{str(svc).upper()}_URL"] = str(url).strip()
        if extra.get("environment"):
            os.environ["UPD_RAW_ENVIRONMENT"] = str(extra["environment"]).strip()
        source = f"connection {conn_id}"
    else:
        load_airflow_config((key,))  # Variables AFYA_<K>_*
        source = f"variables AFYA_{up}_*"

    if not os.environ.get(f"AFYA_{up}_USERNAME") or not os.environ.get(f"AFYA_{up}_PASSWORD"):
        raise RuntimeError(f"{source}: platform account needs a login and password")
    if not os.environ.get(f"AFYA_{up}_URL_TEMPLATE") and not any(
            n.startswith(f"AFYA_{up}_") and n.endswith("_URL") for n in os.environ):
        raise RuntimeError(f"{source}: set the URL template (Connection Host / Extra url_template, "
                           f"or Variable AFYA_{up}_URL_TEMPLATE) so it never falls back to .env's URL")
    os.environ.setdefault("UPD_RAW_ENVIRONMENT", _environment_label(os.environ.get(f"AFYA_{up}_URL_TEMPLATE", "")))
    log.info("platform account %r: %s as %s, environment=%s", key, source,
             os.environ[f"AFYA_{up}_USERNAME"], os.environ["UPD_RAW_ENVIRONMENT"])
    return source


@dag(
    dag_id=DAG_ID,
    description="V3 patient counts per org/facility/region → HOSPITALS.ANALYTICS.AFYA_HMIS_PATIENT_COUNTS (snapshot every 15 min)",
    # every 15 minutes; meant to run on the PROD server only (keep it paused on staging/local:
    # they have no afya_v3_prod connection and would add duplicate snapshots to the same table)
    schedule="*/15 * * * *",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    tags=["v3", "snowflake", "counts", "hmis"],
    default_args={"owner": "data-eng", "retries": 1},
    params={
        "platform": Param("", type="string", title="Platform account key",
                          description="Account key K (default: UPD_RAW_PLATFORM_KEY or 'prod' = PROD V3; 'platform' = MIGRATE). "
                                      "Read from Connection afya_v3_<k>, else Variables AFYA_<K>_*, else .env."),
        "connection": Param("", type="string", title="Airflow connection id",
                            description="Optional: use this Connection instead of afya_v3_<k>."),
        "with_region": Param(True, type="boolean", title="Resolve region",
                             description="Read per-org facilities to fill region_id/name. Off = region null."),
        "dry_run": Param(False, type="boolean", title="Dry run (no Snowflake write)"),
        "monthly_history": Param(True, type="boolean", title="Refresh monthly history",
                                 description="Recompute new patients/visits per org/facility/month since the first "
                                             "record (AFYA_HMIS_PATIENT_REGISTRATIONS_MONTHLY) for running-total charts."),
        # This DAG's table lives in Snowflake: pin it there so the platform-wide
        # System Settings -> Data warehouse switch (afya_backends.toggles) can't
        # redirect the snapshot to Databend.
        "warehouse_backend": Param("snowflake", type="string", enum=["snowflake", "databend"],
                                   title="Warehouse", description="Where the snapshot is written (default Snowflake)."),
    },
)
def patient_counts_to_snowflake():

    @task
    def load(params: dict | None = None) -> dict:
        params = params or {}
        # highest-priority toggle override for this task process (afya_backends.toggles: AFYA_WAREHOUSE_BACKEND)
        os.environ["AFYA_WAREHOUSE_BACKEND"] = (params.get("warehouse_backend") or "snowflake").strip().lower()
        log.info("warehouse backend for this run: %s", os.environ["AFYA_WAREHOUSE_BACKEND"])
        key = ((params.get("platform") or "").strip()
               or (os.getenv("UPD_RAW_PLATFORM_KEY") or "").strip() or DEFAULT_KEY).lower()
        use_pipelines_dir()  # loads pipelines/.env (override=False)
        source = apply_platform_account(key, (params.get("connection") or "").strip() or None)
        import patient_counts_to_snowflake as P
        P.ENVIRONMENT = os.getenv("UPD_RAW_ENVIRONMENT", P.ENVIRONMENT)
        # region lookup uses per-org MIGRATE logins / MIGRATE org ids - meaningless (wrong) for other platforms
        with_region = bool(params.get("with_region", True)) and P.ENVIRONMENT == "migrate"
        if params.get("with_region", True) and not with_region:
            log.info("region lookup skipped: it only works for the migrate platform (this run: %s)", P.ENVIRONMENT)

        summary = P.run(
            platform_key=key,
            with_region=with_region,
            with_history=bool(params.get("monthly_history", True)),
            dry_run=bool(params.get("dry_run", False)),
        )
        summary["account_source"] = source
        log.info("done: %s", summary)
        return summary

    load()


patient_counts_to_snowflake()
