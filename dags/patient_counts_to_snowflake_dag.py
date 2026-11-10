# dags/patient_counts_to_snowflake_dag.py
"""
Patient record counts per organisation / facility / region → HOSPITALS.ANALYTICS.AFYA_HMIS_PATIENT_COUNTS.

Every hour, one call to the MIGRATE platform's native count gateway
(core /gateway/counts, tenants:"all", group_by organization_id+facility_id,
labels) counts every organisation and facility exactly — no row scanning, no
PII. Region is resolved per organisation from the per-org superadmin logins in
pipelines/.env (facility_id → region_id → regions.name); orgs without a login
get a null region. A timestamped snapshot is appended, so the latest
captured_at is "now".

See pipelines/patient_counts_to_snowflake.py. The platform account is
AFYA_PLATFORM_* (key `platform`).

    load  (single task: count → region → write)
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from airflow.sdk import Param, dag, task

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for pipelines_common
from pipelines_common import use_pipelines_dir

log = logging.getLogger(__name__)

DAG_ID = "patient_counts_to_snowflake"


@dag(
    dag_id=DAG_ID,
    description="V3 patient counts per org/facility/region → HOSPITALS.ANALYTICS.AFYA_HMIS_PATIENT_COUNTS (hourly snapshot)",
    schedule="0 * * * *",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    tags=["v3", "snowflake", "counts", "hmis"],
    default_args={"owner": "data-eng", "retries": 1},
    params={
        "platform": Param("", type="string", title="Platform account key",
                          description="Override the platform-statistics key (default: UPD_RAW_PLATFORM_KEY / AFYA_PLATFORM_*)."),
        "with_region": Param(True, type="boolean", title="Resolve region",
                             description="Read per-org facilities to fill region_id/name. Off = region null."),
        "dry_run": Param(False, type="boolean", title="Dry run (no Snowflake write)"),
    },
)
def patient_counts_to_snowflake():

    @task
    def load(params: dict | None = None) -> dict:
        params = params or {}
        use_pipelines_dir()
        import patient_counts_to_snowflake as P

        summary = P.run(
            platform_key=(params.get("platform") or "").strip() or None,
            with_region=bool(params.get("with_region", True)),
            dry_run=bool(params.get("dry_run", False)),
        )
        log.info("done: %s", summary)
        return summary

    load()


patient_counts_to_snowflake()
