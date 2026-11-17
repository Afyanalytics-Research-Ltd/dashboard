# dags/betacare_to_v3_dag.py
"""
Beta Care Hospital (Smart Hospital HMS) → V3 production org 96 / facility 94 through betacare_to_v3.py (Airflow driver;
same code as the command line). Reads the Snowflake mirror HOSPITALS.BETACARE_HOSPITAL_HISTORICAL_DATA
(load_betacare_to_snowflake.py) — nothing on the client's server.

The V3 login comes from Airflow connection `afya_v3_betacare_prod` (login / password; Extra
{"organization_id": 96, "facility_id": 94, "url_template": "https://{service}.afyaanalytics.ai/api/"}) or the
AFYA_BETACARE_PROD_* settings in pipelines/.env.

`base` runs first and alone (staff, insurers, patients, visits, admissions, catalogue, products — what everything else
links to), then four groups side by side, each one step after another:
  clinical   vitals → doctor_notes → diagnoses
  orders     investigations → lab_results
  pharmacy   stock → prescriptions
  money      payments → payment_details → invoices → appointments
`steps` (when not empty) runs just those betacare_to_v3 steps, in order, instead. `patients` (Smart Hospital patient ids)
limits a run to those patients and their visits, orders and money — the canary.

With only_missing (default) every step posts only rows whose uuid V3 doesn't have yet, so a run resumes where the last
one stopped. Rows are matched by stable uuid5s, so nothing is duplicated either way.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.exceptions import AirflowFailException
from airflow.sdk import Param, dag, get_current_context, task

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for pipelines_common
from pipelines_common import use_pipelines_dir

log = logging.getLogger(__name__)

DAG_ID = "betacare_to_v3"
BASE = ["users", "insurance", "patients", "patient_insurance", "visits", "catalogue", "procedure_prices", "products",
        "wards", "admissions", "ward_stays", "discharges"]
GROUPS = {
    "clinical": ["vitals", "doctor_notes", "diagnoses"],
    "orders": ["investigations", "lab_results"],
    "pharmacy": ["stock", "prescriptions"],
    "money": ["payments", "payment_details", "invoices", "appointments"],
}
TARGETS = ["betacare_prod"]


def step_names(raw) -> list[str]:
    """The trigger form's array box takes one item per line; a pasted '["a", "b"]' or 'a, b' arrives as one item."""
    items = raw if isinstance(raw, list) else [raw]
    return [s for item in items for s in re.split(r"[\s,\[\]\"']+", str(item)) if s]


def run_steps(steps: list[str], *, target: str, only_missing: bool, dry_run: bool, workers: int, limit: int,
              patients: list[str]) -> dict:
    """One betacare_to_v3 Run, the steps in order; an errored step fails the task."""
    # the mirror lives in Snowflake only: the run's warehouse is Snowflake unless the trigger form's
    # warehouse_backend explicitly says otherwise (the cluster policy exports that choice before the task starts)
    os.environ.setdefault("AFYA_WAREHOUSE_BACKEND", "snowflake")
    use_pipelines_dir([target])
    import betacare_to_v3 as B
    unknown = [s for s in steps if s not in B.STEPS]
    if unknown:
        raise AirflowFailException(f"unknown steps {unknown}; valid: {list(B.STEPS)}")
    run = B.Run(target, dry_run, limit, workers, only_missing)
    run.only_patients = set(patients) or None
    out = {}
    for s in steps:
        log.info("── %s", s)
        try:
            out[s] = getattr(run, s)()
        except Exception as e:
            log.exception("%s failed", s)
            raise AirflowFailException(f"{s}: {type(e).__name__}: {str(e)[:300]}") from e
        log.info("%s: %s", s, out[s])
        for model, rows in run.failures.items():
            if rows:
                log.warning("%s: %d refused, e.g. %s", model, len(rows), rows[:3])
        run.failures.clear()
    return out


@dag(
    dag_id=DAG_ID,
    description="Beta Care Hospital (Smart Hospital) → V3 production org 96 / facility 94 — resumable (only_missing)",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["v3", "migration", "betacare"],
    default_args={"owner": "data-eng", "retries": 2, "retry_delay": timedelta(minutes=10)},   # a retry resumes (only_missing)
    params={
        "target": Param("betacare_prod", type="string", enum=TARGETS, title="Target",
                        description="betacare_prod = production org 96 / facility 94."),
        "steps": Param([], type="array", items={"type": "string"}, title="Steps (instead of base + groups)",
                       description="Run just these betacare_to_v3 steps, in order. Empty = base, then the groups."),
        "base": Param(True, type="boolean", title="Run the base steps first"),
        "groups": Param(list(GROUPS), type="array", items={"type": "string", "enum": list(GROUPS)}, title="Groups"),
        "only_missing": Param(True, type="boolean", title="Only rows V3 doesn't have yet"),
        "workers": Param(16, type="integer", minimum=1, maximum=48, title="Parallel V3 posts per group"),
        "patients": Param([], type="array", items={"type": "string"}, title="Canary patients (Smart Hospital patient ids)",
                          description="Only these patients and their visits, orders and money — e.g. 9289. Empty = everyone."),
        "limit": Param(0, type="integer", minimum=0, title="At most N rows per step (0 = all; 3 = canary)"),
        "dry_run": Param(False, type="boolean", title="Dry run", description="On: read and count only."),
    },
)
def betacare_to_v3():

    def args() -> dict:
        p = get_current_context()["params"]
        return {"target": p["target"], "only_missing": bool(p["only_missing"]), "dry_run": bool(p["dry_run"]),
                "workers": int(p["workers"]), "limit": int(p["limit"]), "patients": step_names(p["patients"])}

    @task(execution_timeout=timedelta(hours=12))
    def chosen_steps() -> dict:
        p = get_current_context()["params"]
        if not p["steps"]:
            log.info("no steps listed — base + groups run")
            return {}
        return run_steps(step_names(p["steps"]), **args())

    @task(execution_timeout=timedelta(hours=12))
    def base(_) -> dict:
        p = get_current_context()["params"]
        if p["steps"] or not p["base"]:
            log.info("base skipped")
            return {}
        return run_steps(BASE, **args())

    def group_task(name: str):
        @task(task_id=name, execution_timeout=timedelta(hours=12))
        def _run(_) -> dict:
            p = get_current_context()["params"]
            if p["steps"] or name not in p["groups"]:
                log.info("%s not selected", name)
                return {}
            return run_steps(GROUPS[name], **args())
        return _run

    @task(trigger_rule="all_done")
    def summary(*results) -> None:
        bad = {}
        for r in results:
            for step, res in (r or {}).items():
                log.info("%-20s %s", step, res)
                if isinstance(res, dict) and (res.get("failed") or res.get("error")):
                    bad[step] = res.get("failed") or res.get("error")
        if bad:
            log.warning("refused / errored: %s", bad)
        failed = [n for n, r in zip(["chosen_steps", "base", *GROUPS], results) if r is None]
        if failed:
            raise AirflowFailException(f"tasks that failed (re-run; only_missing resumes them): {failed}")

    c = chosen_steps()
    b = base(c)
    summary(c, b, *[group_task(n)(b) for n in GROUPS])


betacare_to_v3()
