# dags/v1_to_v1_dag.py
"""
Old Collabmed V1 server → new V1 server: the records the old server took after the new one was cloned from it,
through pipelines/v1_to_v1.py (Airflow driver; same code as the command line — read its docstring).

  cttm   cttm.afyaanalytics.ai (old, cloned 2026-11-20 morning) → cttm.collabmed.net (live from 2026-11-21)

Logins: Airflow connections afya_v1_<target>_source / afya_v1_<target>_target (host = server URL, login, password),
or <TARGET>_V1_SOURCE_* / <TARGET>_V1_TARGET_* in pipelines/.env.

Records get new ids on the target (the target has used the same ids for its own records) and every reference to a
copied record is remapped. Re-running is safe: what is already on the target is recognised and skipped. Run dry_run
first, then limit = 1 (canary), then everything. A model the target hasn't enabled for writes
(MODEL_GATEWAY_WRITE_MODELS) is reported as blocked and fails the run, after everything else has been copied.
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

TARGETS = ["cttm"]
TABLES = ["reception_patients", "reception_patients_nok", "reception_patient_documents", "evaluation_visits",
          "evaluation_visit_metas", "evaluation_visit_destinations", "evaluation_vitals", "evaluation_doctor_notes",
          "evaluation_nursing_notes", "evaluation_investigations", "evaluation_investigation_results", "result_files",
          "evaluation_sensitivity"]


@dag(
    dag_id="v1_to_v1",
    description="Old V1 server → new V1 server: records taken after the clone, new ids + remapped links — dry_run first",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["v1", "migration"],
    default_args={"owner": "data-eng", "retries": 0},
    params={
        "target": Param("cttm", type="string", enum=TARGETS, title="Target",
                        description="cttm = cttm.afyaanalytics.ai → cttm.collabmed.net"),
        "dry_run": Param(True, type="boolean", title="Dry run", description="On: read, match and log the plan; post nothing."),
        "limit": Param(0, type="integer", minimum=0, title="At most N new rows per table (0 = all; 1 = canary)"),
        "tables": Param([], type="array", title="Only insert into these tables (empty = all)",
                        description="Other tables are still read so references can be mapped. One of: "
                                    + ", ".join(TABLES)),
        "preflight": Param(True, type="boolean", title="Preflight",
                           description="Stop if the source has new rows in a table this pipeline doesn't copy."),
        "files": Param(True, type="boolean", title="Result files",
                       description="Download result PDFs the target lacks (output/v1_to_v1/<target>/files + S3)."),
    },
)
def v1_to_v1():

    @task(execution_timeout=timedelta(hours=2), retries=0)
    def copy() -> dict:
        p = get_current_context()["params"]
        use_pipelines_dir([])
        import v1_to_v1 as V
        tables = [t for t in (p.get("tables") or []) if t]
        report = V.Run(p["target"], apply=not bool(p["dry_run"]), limit=int(p["limit"] or 0), tables=tables,
                       preflight=bool(p["preflight"]), files=bool(p["files"])).run()
        for t, st in report["tables"].items():
            log.info("%-34s %s", t, json.dumps(st, default=str))
        for r in report.get("reregistered", []):
            log.info("re-registered on the target, linked not copied: %s", r)
        log.info("files: %s", json.dumps(report.get("files"), default=str))
        log.info("report: %s", report.get("report_path"))
        summary = {k: report.get(k) for k in ("tables", "blocked", "to_enable", "files", "calls", "report_path")}
        if report.get("failures"):
            raise AirflowFailException(f"{len(report['failures'])} failure(s), e.g. {report['failures'][:3]}")
        if report.get("blocked") and not p["dry_run"]:
            raise AirflowFailException(f"the target refuses writes to {sorted(report['blocked'])} — "
                                       f"set {report.get('to_enable')} on the target server, then run again")
        return summary

    copy()


v1_to_v1()
