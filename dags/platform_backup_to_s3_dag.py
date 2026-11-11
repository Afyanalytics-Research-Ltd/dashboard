# dags/platform_backup_to_s3_dag.py
"""
Daily full platform backup -> s3://$BACKUP_S3_BUCKET/<source>/<YYYY-MM-DD>/

    preflight ─┬─ postgres   every DB on django / airflow / airflow-legacy / redash /
               │             superset / mlflow / marquez (pg_dump -Fc + globals)
               ├─ databend   per-table COPY INTO s3 (parquet) + DDL
               └─ snowflake  DDL every run; table data on Sundays (or snowflake_data=yes).
                             Skipped, not failed, while the account refuses connections.
    manifest   (all_done)    head-checks every object, writes manifest.json, FAILS the
                             run if any target failed - so Sentry/the UI shows it red.

Services whose hostname doesn't resolve on this host are skipped (see the
manifest's "skipped"). Retention = the bucket's 30-day lifecycle rule
(pipelines/platform_backup.py setup-bucket); versioning keeps re-runs safe.
Restore steps: pipelines/BACKUP_RESTORE.md.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import Param, dag, task

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for pipelines_common
from pipelines_common import use_pipelines_dir

log = logging.getLogger(__name__)

DAG_ID = "platform_backup_to_s3"
ENGINES = ["postgres", "databend", "snowflake"]


@dag(
    dag_id=DAG_ID,
    description="Daily backup of all Postgres, Databend and Snowflake data to S3 (30-day retention)",
    schedule="0 22 * * *",  # 22:00 UTC = 01:00 EAT
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=12),
    tags=["backup", "s3", "postgres", "databend", "snowflake"],
    default_args={"owner": "data-eng", "retries": 1, "retry_delay": timedelta(minutes=10),
                  "execution_timeout": timedelta(hours=8)},
    params={
        "engines": Param(ENGINES, type="array", items={"type": "string", "enum": ENGINES},
                         title="Engines to back up"),
        "snowflake_data": Param("auto", type="string", enum=["auto", "yes", "no"],
                                title="Snowflake table data",
                                description="auto = only on BACKUP_SNOWFLAKE_DATA_WEEKDAY (Sunday). DDL always."),
        "dry_run": Param(False, type="boolean", title="Dry run (list what would be uploaded)"),
    },
)
def platform_backup_to_s3():

    @task
    def preflight(params: dict | None = None) -> str:
        use_pipelines_dir()
        import platform_backup as B
        if not (params or {}).get("dry_run"):
            log.info("bucket ok: %s", B.check_bucket())
        prefix = B.run_prefix()
        log.info("backing up to s3://%s/%s/", B.BUCKET, prefix)
        return prefix

    def _engine(name: str):
        @task(task_id=name)
        def run(prefix: str, params: dict | None = None) -> list[dict]:
            params = params or {}
            if name not in (params.get("engines") or ENGINES):
                return [{"engine": name, "target": "*", "status": "skipped",
                         "reason": "not selected for this run", "objects": []}]
            use_pipelines_dir()
            import platform_backup as B
            kw = {}
            if name == "snowflake" and params.get("snowflake_data") in ("yes", "no"):
                kw["with_data"] = params["snowflake_data"] == "yes"
            results = B.ENGINES[name](prefix, dry_run=bool(params.get("dry_run")), **kw)
            for r in results:
                log.info("%s/%s: %s %s", r["engine"], r["target"], r["status"],
                         r.get("reason") or r.get("errors") or f"{len(r.get('objects', []))} objects")
            return results
        return run

    @task(trigger_rule="all_done", retries=0)
    def manifest(prefix: str, params: dict | None = None, **context) -> dict:
        use_pipelines_dir()
        import platform_backup as B
        ti = context["ti"]
        results = []
        for name in ENGINES:
            got = ti.xcom_pull(task_ids=name)
            if got is None:  # the task itself crashed / timed out: no results to trust
                results.append({"engine": name, "target": "*", "status": "failed", "objects": [],
                                "errors": ["task did not return results - see its log"]})
            else:
                results.extend(got)
        return B.write_manifest(prefix, results, dry_run=bool((params or {}).get("dry_run")))

    prefix = preflight()
    engine_tasks = [_engine(n)(prefix) for n in ENGINES]
    done = manifest(prefix)
    engine_tasks >> done


platform_backup_to_s3()
