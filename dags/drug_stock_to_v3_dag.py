# dags/drug_stock_to_v3_dag.py
"""
A hospital's hand-counted drug stock sheet (item rows + batch rows, cost / selling / insurance prices) → a stock take
of one V3 store, through pipelines/drug_stock_to_v3.py (same code as the command line).

Data: dags/jsons/stock_takes/<target>.json, made with `python drug_stock_to_v3.py convert <xlsx> --target <target>`.
V3 login: Airflow connection afya_v3_<target>.

  chacawest_prod   Chacawest Healthcare, production org 131 / facility 129 — count of 2026-11-27 into Main store 288:
                   matched products renamed to the sheet + sheet prices, counted batches set, their other batches → 0

Run dry_run first. Re-running is safe (stable uuids; batches already holding the count are left alone).
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

TARGETS = ["chacawest_prod"]
STEPS = ["products", "stock", "expiry", "store_rows", "verify"]


@dag(
    dag_id="drug_stock_to_v3",
    description="Drug stock count sheet → V3 stock take (products, prices, batches) — dry_run first",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["v3", "migration", "inventory"],
    default_args={"owner": "data-eng", "retries": 0},
    params={
        "target": Param("chacawest_prod", type="string", enum=TARGETS, title="Target",
                        description="chacawest_prod = Chacawest Healthcare (org 131 / facility 129), Main store 288."),
        "dry_run": Param(True, type="boolean", title="Dry run", description="On: read and log the plan, post nothing."),
        "steps": Param(STEPS, type="array", items={"type": "string", "enum": STEPS}, title="Steps"),
        "workers": Param(8, type="integer", minimum=1, maximum=32, title="Parallel V3 posts"),
    },
)
def drug_stock_to_v3():

    @task(execution_timeout=timedelta(hours=2))
    def load() -> dict:
        p = get_current_context()["params"]
        use_pipelines_dir([p["target"]])
        import drug_stock_to_v3 as S
        report = S.Run(p["target"], bool(p["dry_run"]), int(p["workers"]), list(p["steps"])).load()
        for k, v in report["detail"].items():
            log.info("%s: %s", k, json.dumps(v, default=str))
        out = {"plan": report["plan"], "result": report["result"]}
        log.info("result: %s", json.dumps(out, default=str))
        if report.get("failures"):
            raise AirflowFailException(f"{len(report['failures'])} refused, e.g. {report['failures'][:5]}")
        return out

    load()


drug_stock_to_v3()
