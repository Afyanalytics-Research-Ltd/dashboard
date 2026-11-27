# dags/commercialization_counts_dag.py
"""
Commercialization counts → Databend database COMMERCIALIZATION (snapshot every 15 minutes).

One call to the V3 core count gateway (POST {core}/v1/gateway/counts, tenants:"all", labels) with the
commercialization team's three queries — patients per county & facility, distinct patients per facility &
ICD-10 diagnosis (visit_icd10_diagnoses), distinct patients with doctor notes per facility — appended as
timestamped snapshots, plus LATEST_* tables holding only the newest run. See pipelines/commercialization_counts.py.

Account: the platform key (param "platform", default "prod" → Airflow Connection afya_v3_prod, the same account
patient_counts_to_snowflake uses). It needs PLATFORM STATISTICS access on V3 (counting across organisations);
the URL template comes from the Connection, so a domain change is a Connection edit, not a code change
(dags/v3_platform_account.py).

Warehouse: Databend — this DAG's own warehouse_backend param defaults to "databend" and outranks
System Settings → Data warehouse.
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from airflow.sdk import Param, dag, task

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for pipelines_common
from pipelines_common import use_pipelines_dir
from v3_platform_account import apply_platform_account

log = logging.getLogger(__name__)

DAG_ID = "commercialization_counts"
DEFAULT_KEY = "prod"


@dag(
    dag_id=DAG_ID,
    description="V3 counts for commercialization (patients by county, diseases, doctor notes) → Databend COMMERCIALIZATION",
    schedule="*/15 * * * *",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    tags=["v3", "databend", "counts", "commercialization"],
    default_args={"owner": "data-eng", "retries": 1},
    params={
        "platform": Param("", type="string", title="Platform account key",
                          description="Account key K (default 'prod'). Read from Connection afya_v3_<k>, "
                                      "else Variables AFYA_<K>_*, else pipelines/.env."),
        "connection": Param("", type="string", title="Airflow connection id",
                            description="Optional: use this Connection instead of afya_v3_<k>."),
        "dry_run": Param(False, type="boolean", title="Dry run (no warehouse write)"),
        # Declared here, so config/airflow_local_settings.py leaves it alone and it outranks System Settings.
        "warehouse_backend": Param("databend", type="string", enum=["default", "snowflake", "databend"],
                                   title="Warehouse", description="Where the snapshots are written (default Databend)."),
    },
)
def commercialization_counts():

    @task
    def load(params: dict | None = None) -> dict:
        from airflow.exceptions import AirflowFailException
        params = params or {}
        backend = (params.get("warehouse_backend") or "databend").strip().lower()
        if backend in ("snowflake", "databend"):
            os.environ["AFYA_WAREHOUSE_BACKEND"] = backend
        key = ((params.get("platform") or "").strip() or DEFAULT_KEY).lower()
        use_pipelines_dir()  # loads pipelines/.env (override=False)
        source = apply_platform_account(key, (params.get("connection") or "").strip() or None)
        import commercialization_counts as C
        C.ENVIRONMENT = os.getenv("UPD_RAW_ENVIRONMENT", C.ENVIRONMENT)
        try:
            summary = C.run(platform_key=key, dry_run=bool(params.get("dry_run", False)))
        except C.CountAccessError as e:
            # a credential problem, not a bug: no retry, no stack trace
            raise AirflowFailException(f"{e} (account: {source})") from None
        summary["account_source"] = source
        if summary["failed"]:
            log.warning("some queries failed (the others were written): %s",
                        {q: summary["queries"][q]["error"] for q in summary["failed"]})
        log.info("done: %s", summary)
        return summary

    load()


commercialization_counts()
