# dags/warehouse_bisync_dag.py
"""
Two-way sync Snowflake <-> Databend (afya_backends/warehouse/bisync.py).

Every hour at :30, per base table in Snowflake HOSPITALS (all schemas) and
the matching Databend databases (HOSPITALS.<SCHEMA> <-> Databend <SCHEMA>):

    only on one side            -> copied across
    changed on one side only    -> other side replaced (staging + row-count check + swap)
    changed on both / unknown   -> keyed tables (ID/UUID/_ID): missing rows inserted both ways
                                   no key: reported as a conflict, left alone
    a copy never shrinks the target unless allow_shrink=true (it is reconciled instead)

While Snowflake refuses connections (the 2026-11-10 lock / invalid JWT) the
run is SKIPPED, not failed. State (fingerprints after the last sync):
pipelines/.warehouse_bisync_state.json. The run FAILS if any table failed to
sync; conflicts are logged and returned but don't fail it.
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import Param, dag, task

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for pipelines_common
from pipelines_common import PIPELINES_DIR, use_pipelines_dir

log = logging.getLogger(__name__)

DAG_ID = "warehouse_bisync"
STATE_FILE = PIPELINES_DIR / ".warehouse_bisync_state.json"


def _snowflake_conn(database: str):
    import snowflake.connector
    kw = dict(user=os.getenv("SNOWFLAKE_USER", "").strip(), account=os.getenv("SNOWFLAKE_ACCOUNT", "").strip(),
              warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "").strip() or None,
              role=os.getenv("SNOWFLAKE_ROLE", "").strip() or None, database=database, login_timeout=60,
              session_parameters={"QUERY_TAG": DAG_ID, "TIMEZONE": "UTC"})
    key_path = (os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH") or "").strip()
    if key_path:
        kw["private_key_file"] = key_path
    elif os.getenv("SNOWFLAKE_PASSWORD"):
        kw["password"] = os.getenv("SNOWFLAKE_PASSWORD").strip()
    else:
        raise RuntimeError("set SNOWFLAKE_PRIVATE_KEY_PATH or SNOWFLAKE_PASSWORD")
    return snowflake.connector.connect(**kw)


@dag(
    dag_id=DAG_ID,
    description="Two-way Snowflake HOSPITALS <-> Databend table sync (safe: staging+swap, never-shrink, keyed reconcile)",
    schedule="30 * * * *",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=6),
    tags=["snowflake", "databend", "sync", "warehouse"],
    default_args={"owner": "data-eng", "retries": 0, "execution_timeout": timedelta(hours=5)},
    params={
        "database": Param("HOSPITALS", type="string", title="Snowflake database"),
        "schemas": Param("", type="string", title="Schemas (comma-separated, empty = all)"),
        "tables": Param("", type="string", title="Only these tables (SCHEMA.TABLE or TABLE, comma-separated)"),
        "allow_shrink": Param(False, type="boolean", title="Allow a copy to shrink the target",
                              description="Off: a copy that would leave the target with fewer rows is reconciled instead."),
        "dry_run": Param(False, type="boolean", title="Dry run (plan only, no writes)"),
    },
)
def warehouse_bisync():

    @task
    def sync(params: dict | None = None) -> dict:
        from airflow.exceptions import AirflowSkipException
        params = params or {}
        use_pipelines_dir()
        from afya_backends.warehouse import bisync, dbapi

        database = (params.get("database") or "HOSPITALS").strip().upper()
        schemas = {s.strip().upper() for s in (params.get("schemas") or "").split(",") if s.strip()} or None
        only = {t.strip().upper() for t in (params.get("tables") or "").split(",") if t.strip()} or None
        try:
            sf_conn = _snowflake_conn(database)
        except Exception as e:  # noqa: BLE001 - e.g. 250001 system lock / JWT token is invalid
            raise AirflowSkipException(f"Snowflake unreachable, nothing synced: {str(e)[:300]}")

        sf = bisync.Side("snowflake", sf_conn, database, schemas)
        db = bisync.Side("databend", dbapi.Connection(database=database), database, schemas)
        # Big tables go Snowflake -> S3 parquet -> Databend server-side once
        # pipelines/setup_bisync_bulk.py has created the stage; until then (or if
        # it breaks) every table uses the row-streaming path.
        bulk = bisync.Bulk(
            stage=os.getenv("BISYNC_SNOWFLAKE_STAGE", "HOSPITALS.ANALYTICS.BISYNC_STAGE"),
            s3_url=os.getenv("BISYNC_S3_URL", "s3://afya-platform-backups-707559309638/_bisync_stage/"),
            key_id=(os.getenv("BACKUP_AWS_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID") or "").strip(),
            secret=(os.getenv("BACKUP_AWS_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip(),
            region=os.getenv("BACKUP_S3_REGION") or os.getenv("AWS_REGION") or "eu-north-1",
            min_rows=int(os.getenv("BISYNC_BULK_MIN_ROWS", "200000")))
        try:
            report = bisync.run(sf, db, STATE_FILE, allow_shrink=bool(params.get("allow_shrink")),
                                dry_run=bool(params.get("dry_run")), only=only, log=log.info, bulk=bulk)
        finally:
            sf_conn.close()
        for c in report["conflicts"]:
            log.warning("conflict: %s", c)
        if report["failed"]:
            raise RuntimeError(f"{len(report['failed'])} table(s) failed: {report['failed'][:10]}")
        return report

    sync()


warehouse_bisync()
