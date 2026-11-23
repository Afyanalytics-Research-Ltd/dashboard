# dags/v3_to_v3_all_orgs_dag.py
"""
Every organisation: the records a V3 clone (test*.afyaanalytics.ai) has that the target
(prod) lacks — surveyed, then copied — with ONE platform account per side
(Airflow driver for v3_to_v3_all_orgs.py; tail mode of v3_to_v3_migration per org).

  survey   read-only: per org, would_post (by the day created), held, would_patch
  migrate  posts them (rows CREATED in [since, until) only), fills empty links, relinks

Connections (login / password = the platform account, e.g. "analyst"):
  afya_v3_platform_source  Extra {"url_template": "https://test{service}.afyaanalytics.ai/api/",
                                  "migration_key": "…"}
  afya_v3_platform_target  Extra {"url_template": "https://{service}.afyaanalytics.ai/api/",
                                  "migration_key": "…"}
The summary (per org, per model) is saved in the state dir: all_orgs_<mode>_<time>.json.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

try:
    from airflow.sdk.exceptions import AirflowFailException
except ImportError:   # older Airflow 3.0.x
    from airflow.exceptions import AirflowFailException
from airflow.sdk import Param, dag, get_current_context, task

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for pipelines_common
from pipelines_common import _connection, use_pipelines_dir

log = logging.getLogger(__name__)


def _platform_env() -> None:
    for side, conn_id in (("SOURCE", "afya_v3_platform_source"), ("TARGET", "afya_v3_platform_target")):
        c = _connection(conn_id)
        if c is None:
            raise AirflowFailException(f"Connection {conn_id} is missing")
        extra = c.extra_dejson or {}
        os.environ[f"V3V3_PLATFORM_{side}_USERNAME"] = c.login or ""
        os.environ[f"V3V3_PLATFORM_{side}_PASSWORD"] = c.password or ""
        os.environ[f"V3V3_PLATFORM_{side}_URL_TEMPLATE"] = extra.get("url_template") or ""
        os.environ[f"V3V3_PLATFORM_{side}_MIGRATION_KEY"] = extra.get("migration_key") or ""


@dag(
    dag_id="v3_to_v3_all_orgs",
    description="Every org: records a V3 clone has that the target lacks — survey, then copy (one platform account)",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["v3", "migration"],
    default_args={"owner": "data-eng", "retries": 0},
    params={
        "mode": Param("survey", type="string", enum=["survey", "migrate"], title="Mode",
                      description="survey: read-only counts per org. migrate: post them."),
        "since": Param("2026-11-16", type="string", title="Rows created since (Nairobi time)"),
        "until": Param("", type="string", title="… until (exclusive, optional)"),
        "orgs": Param(["auto"], type="array", items={"type": ["string", "integer"]}, title="Orgs",
                      description="Org ids, or auto: every org with patients / visits / invoices created since `since`."),
        "exclude_orgs": Param([], type="array", items={"type": "integer"}, title="Exclude orgs"),
        "parallel_orgs": Param(2, type="integer", minimum=1, maximum=6, title="Orgs at once"),
        "workers": Param(6, type="integer", minimum=1, maximum=32, title="Parallel POSTs per model"),
        "read_workers": Param(4, type="integer", minimum=1, maximum=16, title="Parallel page reads per table"),
        "identity_links": Param(["inventory/"], type="array", items={"type": "string"}, title="Identity links"),
        "redo": Param(False, type="boolean", title="Re-check finished orgs",
                      description="By default a migrate skips orgs a previous run (same `since`) finished."),
        "read_mode": Param("filter", type="string", enum=["filter", "tail"], title="Reads",
                           description="filter: only rows updated since `since` on each side (needs the gateway's "
                                       "updated_since / after_id; `since` must be before the clone). tail: bisection."),
    },
)
def v3_to_v3_all_orgs():

    @task(execution_timeout=timedelta(hours=20))
    def run() -> dict:
        p = get_current_context()["params"]
        use_pipelines_dir([])
        _platform_env()
        import v3_to_v3_all_orgs as allorgs
        argv = [p["mode"], "--since", p["since"], "--orgs", *[str(o) for o in (p["orgs"] or ["auto"])],
                "--parallel-orgs", str(p["parallel_orgs"]), "--workers", str(p["workers"]),
                "--read-workers", str(p["read_workers"]), "--identity-links", *p["identity_links"],
                "--read-mode", p.get("read_mode") or "filter"]
        if p.get("until"):
            argv += ["--until", p["until"]]
        if p.get("redo"):
            argv += ["--redo"]
        if p.get("exclude_orgs"):
            argv += ["--exclude-orgs", *[str(o) for o in p["exclude_orgs"]]]
        rc = allorgs.main(argv)
        latest = max(allorgs.vv.STATE_ROOT.glob(f"all_orgs_{p['mode']}_*.json"), key=lambda f: f.stat().st_mtime)
        summary = json.loads(latest.read_text())
        log.info("totals %s; errors %s; summary %s", summary["totals"], summary["errors"], latest)
        for o in sorted(summary["orgs"], key=lambda o: -(o.get("would_post") or o.get("posted") or 0)):
            log.info("org %s: would_post %s held %s would_patch %s posted %s failed %s%s", o["org"],
                     o.get("would_post"), o.get("held"), o.get("would_patch"), o.get("posted"), o.get("failed"),
                     f" ERROR {o['error']}" if o.get("error") else "")
        if rc:
            raise AirflowFailException(f"{len(summary['errors'])} org(s) failed or rows were refused — see {latest}")
        return summary["totals"]

    run()


v3_to_v3_all_orgs()
