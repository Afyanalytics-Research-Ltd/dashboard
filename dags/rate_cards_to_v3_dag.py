# dags/rate_cards_to_v3_dag.py
"""
A hospital's per-payer rate cards + insurer/scheme list + item cost prices → V3, through
pipelines/rate_cards_to_v3.py (Airflow driver; same code as the command line).

Data: dags/jsons/rate_cards/<target>.json, made from the hospital's spreadsheets with
`python rate_cards_to_v3.py convert <folder> --target <target>`.
The V3 login comes from Airflow connection `afya_v3_<target>` (login / password; Extra {"organization_id": …,
"facility_id": …, "url_template": "https://{service}.afyaanalytics.ai/api/"}) or AFYA_<TARGET>_* in pipelines/.env.

  kerugoya_prod   Kerugoya Medical Centre, production org 357 / facility 367 (kerugoya.zip, 2026-11-24)

Steps (each its own task, in order):
  procedures      facility prices: cash = Cash card, insurance = "Other Insurance" card; missing items → new procedures
  schemes         the insurer/scheme list → core insurance_schemes under the existing V3 companies
  scheme_pricing  each insurer's own card → core scheme_pricing on its schemes, where it differs from the default
  product_prices  facility products' cost_price / selling_price from the cost and pharmacy sheets
Run dry_run first: it reads and logs the plan, posts nothing. Re-running is safe (stable uuids, unchanged rows skipped).
The gateway has no delete.
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

TARGETS = ["kerugoya_prod", "kerugoya_town_prod"]
STEPS = ["procedures", "schemes", "scheme_pricing", "product_prices", "inpatient_charges", "company_pricing", "verify",
         "sha_move", "copy_facility_prices", "catalog"]


def _run(step: str) -> dict:
    p = get_current_context()["params"]
    if step not in p["steps"]:
        log.info("step %s not selected — skipped", step)
        return {"skipped": True}
    use_pipelines_dir([p["target"]])
    import rate_cards_to_v3 as R
    report = R.Run(p["target"], bool(p["dry_run"]), int(p["limit"]), int(p["workers"]), [step]).load()
    d = report["detail"]
    log.info("plan: %s", json.dumps(report["plan"], default=str))
    if d.get("category_map"):
        log.info("sheet category → V3 category: %s", d["category_map"])
    for u in d.get("price_updates", []):
        log.info("price %s %s: %s → %s", u["_key"], u["name"], u["before"], u["data"])
    for n in d.get("new_procedures", []):
        log.info("new procedure %s: %s", n["code"], n["name"])
    for s in d.get("new_schemes", []):
        log.info("new scheme %s: %s (company %s)", s["code"], s["name"], s["company_id"])
    for u in d.get("scheme_price_updates", [])[:200]:
        log.info("scheme price %s: %s → %s", u["_key"], u["before"], u["data"])
    for u in d.get("product_price_updates", []):
        log.info("product %s %s: %s → %s", u["id"], u["name"], u["before"], u["data"])
    if d.get("matched_but_inactive"):
        log.info("matched_but_inactive: %s", d["matched_but_inactive"])
    out = {"plan": report["plan"], "result": report["result"]}
    log.info("result: %s", json.dumps(out, default=str))
    if report.get("failures"):
        raise AirflowFailException(f"{len(report['failures'])} refused, e.g. {report['failures'][:5]}")
    return out


@dag(
    dag_id="rate_cards_to_v3",
    description="Hospital rate cards (cash / insurers / SHA) + schemes + cost prices → V3 — dry_run first",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["v3", "migration", "prices"],
    default_args={"owner": "data-eng", "retries": 0},
    params={
        "target": Param("kerugoya_prod", type="string", enum=TARGETS, title="Target",
                        description="kerugoya_prod = Kerugoya Medical Centre (org 357 / facility 367); kerugoya_town_prod = Kerugoya Town Clinic (facility 368, mirror)."),
        "dry_run": Param(True, type="boolean", title="Dry run", description="On: read and log the plan, post nothing."),
        "steps": Param(STEPS, type="array", items={"type": "string", "enum": STEPS}, title="Steps"),
        "limit": Param(0, type="integer", minimum=0, title="At most N changes of each kind (0 = all; 3 = canary)"),
        "workers": Param(8, type="integer", minimum=1, maximum=32, title="Parallel V3 posts"),
    },
)
def rate_cards_to_v3():

    @task(execution_timeout=timedelta(hours=1))
    def procedures() -> dict:
        return _run("procedures")

    @task(execution_timeout=timedelta(hours=1))
    def schemes() -> dict:
        return _run("schemes")

    @task(execution_timeout=timedelta(hours=4))
    def scheme_pricing() -> dict:
        return _run("scheme_pricing")

    @task(execution_timeout=timedelta(hours=1))
    def product_prices() -> dict:
        return _run("product_prices")

    @task(execution_timeout=timedelta(hours=1))
    def inpatient_charges() -> dict:
        """ward charges (bed / nursing / review / meals per ward) as inpatient charges"""
        return _run("inpatient_charges")

    @task(execution_timeout=timedelta(hours=2))
    def company_pricing() -> dict:
        """each insurer's whole card as company prices (Procedures Pricing — <insurer> → Company Price)"""
        return _run("company_pricing")

    @task(execution_timeout=timedelta(hours=1), trigger_rule="all_done")
    def verify() -> dict:
        """read-only, record by record: facility prices + every insurer's company prices vs the sheets"""
        return _run("verify")

    @task(execution_timeout=timedelta(hours=1))
    def sha_move() -> dict:
        """default insurance columns (procedures + products) → SHA company prices, then cleared to 0"""
        return _run("sha_move")

    @task(execution_timeout=timedelta(hours=1))
    def catalog() -> dict:
        """service list = the sheets: names / categories (Cash card for procedures, SHA card for lab)"""
        return _run("catalog")

    @task(execution_timeout=timedelta(hours=1))
    def copy_facility_prices() -> dict:
        """procedure prices this facility lacks, copied from the source facility (level 5)"""
        return _run("copy_facility_prices")

    (procedures() >> catalog() >> schemes() >> scheme_pricing() >> product_prices() >> inpatient_charges()
     >> company_pricing() >> sha_move() >> copy_facility_prices() >> verify())


rate_cards_to_v3()
