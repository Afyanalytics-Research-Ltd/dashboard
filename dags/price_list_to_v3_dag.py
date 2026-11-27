# dags/price_list_to_v3_dag.py
"""
A hospital's procedure & service price list → V3 procedures + facility prices, through pipelines/price_list_to_v3.py
(Airflow driver; same code as the command line).

Data: dags/jsons/price_lists/<target>.json, made from the hospital's spreadsheet with
`python price_list_to_v3.py convert <xlsx> --target <target>`.
The V3 login comes from Airflow connection `afya_v3_<target>` (login / password; Extra {"organization_id": …,
"facility_id": …, "url_template": "https://{service}.afyaanalytics.ai/api/"}) or AFYA_<TARGET>_* in pipelines/.env.

  transmarigold_prod   Trans Marigold Hospital, production org 87 / facility 85 ("TMH procedure & servive list.xlsx")

The sheet is reconciled with what the org already has: a row whose procedure exists (by name) gets its facility price
set to the sheet's cash / insurance price; a row with no procedure gets a new one (code <PREFIX>-<row #>) and its
price. A sheet price of 0 / 0 never overwrites a real V3 price. Every change, with the price it replaced, is logged.
Run dry_run first: it reads and logs the plan, posts nothing. Re-running is safe (stable uuids, unchanged prices are
skipped).
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

TARGETS = ["transmarigold_prod", "fairview_prod"]


@dag(
    dag_id="price_list_to_v3",
    description="Hospital price list (spreadsheet) → V3 procedures + facility prices — dry_run first",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["v3", "migration", "prices"],
    default_args={"owner": "data-eng", "retries": 1, "retry_delay": timedelta(minutes=5)},
    params={
        "target": Param("transmarigold_prod", type="string", enum=TARGETS, title="Target",
                        description="transmarigold_prod = Trans Marigold Hospital (org 87 / facility 85); fairview_prod = Fairview Medical Centre (org 154 / facility 152)."),
        "dry_run": Param(True, type="boolean", title="Dry run", description="On: read and log the plan, post nothing."),
        "limit": Param(0, type="integer", minimum=0, title="At most N changes of each kind (0 = all; 3 = canary)"),
        "workers": Param(8, type="integer", minimum=1, maximum=32, title="Parallel V3 posts"),
    },
)
def price_list_to_v3():

    @task(execution_timeout=timedelta(hours=1))
    def load() -> dict:
        p = get_current_context()["params"]
        use_pipelines_dir([p["target"]])
        import price_list_to_v3 as P
        report = P.Run(p["target"], bool(p["dry_run"]), int(p["limit"]), int(p["workers"])).load()
        d = report["detail"]
        log.info("plan: %s", report["plan"])
        log.info("department → category: %s", d["department_category"])
        for u in d["price_updates"]:
            log.info("price row %s %s [%s]: %s → %s", u["_row"], u["name"], u["code"], u["before"], u["data"])
        for n in d["new_procedures"]:
            log.info("new procedure row %s %s: %s", n["_row"], n["code"], n["name"])
        for k in ("kept_v3_price_sheet_has_none", "matched_but_inactive"):
            if d[k]:
                log.info("%s: %s", k, d[k])
        out = {k: report.get(k) for k in ("plan", "procedures", "price_inserts", "price_updates")}
        log.info("result: %s", json.dumps(out, default=str))
        if report.get("failures"):
            raise AirflowFailException(f"{len(report['failures'])} refused, e.g. {report['failures'][:5]}")
        return out

    load()


price_list_to_v3()
