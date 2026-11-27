# dags/databend_materialize_views_dag.py
"""
Databend: serve the platform's REPORTING / STAGING / BRONZE views as refreshed tables, so queries and column
metadata (DBeaver, dashboards) run at table speed. See afya_backends/warehouse/materialize.py.

Measured on prod 2026-11-27 before this: COUNT(*) of the RPT_KSH_V3_INV_CONTROL view 31 s (a table: 0.03 s),
one view's column list 5-8 s, all REPORTING columns 131-142 s - DBeaver timed out.

Hourly at :45 (after v3_to_warehouse_sync at :15 has loaded): views are refreshed dependencies first, each
from its <NAME>__SRC definition. Views people created themselves (owned by their role) are never touched.
A view Databend can't copy stays a view (warning); a table that fails to refresh fails the run (it is going
stale). Param `revert` turns everything back into plain views.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import Param, dag, get_current_context, task

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for pipelines_common
from pipelines_common import use_pipelines_dir

log = logging.getLogger(__name__)
DAG_ID = "databend_materialize_views"


@dag(
    dag_id=DAG_ID,
    schedule="45 * * * *",
    start_date=datetime(2026, 11, 27),
    catchup=False,
    max_active_runs=1,
    tags=["databend", "warehouse", "performance"],
    default_args={"owner": "data-eng", "retries": 1, "retry_delay": timedelta(minutes=5)},
    params={
        "databases": Param(["REPORTING", "STAGING", "BRONZE"], type="array", items={"type": "string"},
                           title="Databases whose platform views are served as tables"),
        "only": Param([], type="array", items={"type": "string"}, title="Only these (DB.NAME or NAME)"),
        "dry_run": Param(False, type="boolean", title="Dry run", description="List what would be done."),
        "revert": Param(False, type="boolean", title="Revert",
                        description="Turn every materialized view back into a plain view, then stop."),
    },
)
def databend_materialize_views():

    @task(execution_timeout=timedelta(minutes=55))
    def refresh() -> dict:
        from airflow.exceptions import AirflowFailException
        p = get_current_context()["params"]
        use_pipelines_dir()
        from afya_backends.warehouse import dbapi, materialize
        conn = dbapi.Connection(database="HOSPITALS")
        if p.get("revert"):
            done = materialize.revert(conn, p["databases"], log=log.info)
            log.info("reverted %d view(s)", len(done))
            return {"reverted": done}
        rep = materialize.run(conn, p["databases"], only=set(p.get("only") or []) or None,
                              dry_run=bool(p.get("dry_run")), log=log.info)
        slow = sorted(rep["seconds"].items(), key=lambda kv: -kv[1])[:10]
        log.info("slowest: %s", ", ".join(f"{k} {v:.0f}s" for k, v in slow))
        for f in rep["failed"]:
            log.warning("failed: %s", f)
        if rep.get("stale"):
            raise AirflowFailException(f"{len(rep['stale'])} table(s) could not be refreshed and are going stale "
                                       f"(their last copy is still served): {rep['stale']}")
        return {k: v for k, v in rep.items() if k != "seconds"}

    refresh()


databend_materialize_views()
