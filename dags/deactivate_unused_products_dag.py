# dags/deactivate_unused_products_dag.py
"""
Switch off (is_active = 0) an org's TEMPLATE inventory products (facility_id NULL) that nothing uses — no stock, batch,
scheme price, prescription, consumable, sale or order line — so facility product lists show only the hospital's own
items. Facility products are never touched. Reversible: every deactivated id + name is logged; set reactivate_ids to
put them back. Same code as pipelines/deactivate_unused_products.py.

  kerugoya_prod   Kerugoya Medical Centre, production org 357 (client wants only their Excel products; 2026-11-24)
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.exceptions import AirflowFailException
from airflow.sdk import Param, dag, get_current_context, task

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for pipelines_common
from pipelines_common import use_pipelines_dir

log = logging.getLogger(__name__)


@dag(
    dag_id="deactivate_unused_products",
    description="Deactivate unused template inventory products of an org (reversible) — dry_run first",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["v3", "inventory", "cleanup"],
    default_args={"owner": "data-eng", "retries": 0},
    params={
        "target": Param("kerugoya_prod", type="string", enum=["kerugoya_prod"], title="Target"),
        "dry_run": Param(True, type="boolean", title="Dry run", description="On: read and log the plan, change nothing."),
        "workers": Param(12, type="integer", minimum=1, maximum=32, title="Parallel V3 posts"),
        "reactivate_ids": Param([], type="array", items={"type": "integer"}, title="Undo: product ids to set active again"),
    },
)
def deactivate_unused_products():

    @task(execution_timeout=timedelta(hours=2))
    def run() -> dict:
        p = get_current_context()["params"]
        use_pipelines_dir([p["target"]])
        import deactivate_unused_products as D
        r = D.Run(p["target"], bool(p["dry_run"]), int(p["workers"]))
        if p["reactivate_ids"]:
            r.apply([{"id": int(i)} for i in p["reactivate_ids"]], 1)
        else:
            items = r.plan()
            if not r.dry_run:
                for n in range(0, len(items), 500):   # the undo list, in the task log
                    log.info("deactivating ids %d-%d: %s", n, n + 499, json.dumps([x["id"] for x in items[n:n + 500]]))
            r.apply(items, 0)
        out = {"plan": r.report["plan"], "result": r.report["result"]}
        log.info("result: %s", json.dumps(out))
        if r.report["failures"]:
            raise AirflowFailException(f"{len(r.report['failures'])} refused, e.g. {r.report['failures'][:5]}")
        return out

    run()


deactivate_unused_products()
