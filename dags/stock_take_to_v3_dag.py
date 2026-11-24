# dags/stock_take_to_v3_dag.py
"""
Filled-in V3 stock-take templates (one spreadsheet per store) → V3 products, batches, stock, adjustment movements and
a completed stock take per store, through pipelines/stock_take_to_v3.py (same code as the command line).

Data: dags/jsons/stock_takes/<target>.json, made with `python stock_take_to_v3.py convert <folder> --target <target>`;
prices for new products come from dags/jsons/rate_cards/<target>.json. V3 login: Airflow connection afya_v3_<target>.

  kerugoya_prod   Kerugoya Medical Centre, production org 357 / facility 367 — counts of 2026-11-23:
                  pharmacy → Pharmacy (945); dental / optical → new "<Dental|Optical> Store - Kerugoya Medical Centre"

Missing products are created in the facility; counted rows get a batch + stock + adjustment movement; one completed
stock take per store records every line. Run dry_run first. Re-running is safe (stable uuids; batches already holding
the count are left alone). The gateway has no delete.
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

TARGETS = ["kerugoya_prod"]
STEPS = ["stores", "products", "stock"]


@dag(
    dag_id="stock_take_to_v3",
    description="Stock-take spreadsheets → V3 products, batches, stock + completed stock takes — dry_run first",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["v3", "migration", "inventory"],
    default_args={"owner": "data-eng", "retries": 0},
    params={
        "target": Param("kerugoya_prod", type="string", enum=TARGETS, title="Target",
                        description="kerugoya_prod = Kerugoya Medical Centre, production org 357 / facility 367."),
        "dry_run": Param(True, type="boolean", title="Dry run", description="On: read and log the plan, post nothing."),
        "steps": Param(STEPS, type="array", items={"type": "string", "enum": STEPS}, title="Steps"),
        "workers": Param(8, type="integer", minimum=1, maximum=32, title="Parallel V3 posts"),
    },
)
def stock_take_to_v3():

    @task(execution_timeout=timedelta(hours=2))
    def load() -> dict:
        p = get_current_context()["params"]
        use_pipelines_dir([p["target"]])
        import stock_take_to_v3 as S
        report = S.Run(p["target"], bool(p["dry_run"]), int(p["workers"]), list(p["steps"])).load()
        d = report["detail"]
        log.info("plan: %s", json.dumps(report["plan"], default=str))
        for n in d.get("new_products", []):
            log.info("new product %s [%s] cat %s sell %s cost %s track_expiry %s", n["generic_name"], n.get("code"),
                     n["category_id"], n["selling_price"], n["cost_price"], n["track_expiry"])
        for k in ("counted_without_expiry", "batches_recount", "batches_zeroed"):
            if d.get(k):
                log.info("%s: %s", k, d[k])
        out = {"plan": report["plan"], "result": report["result"]}
        log.info("result: %s", json.dumps(out, default=str))
        if report.get("failures"):
            raise AirflowFailException(f"{len(report['failures'])} refused, e.g. {report['failures'][:5]}")
        return out

    load()


stock_take_to_v3()
