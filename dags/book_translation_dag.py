# dags/book_translation_dag.py
"""
Book translation pipeline (Kenyan-language ASR / TTS / MT on Kaggle) — the operational tools of the
book-translation-pipeline repo, run from Airflow. Logic: pipelines/book_translation/btp_airflow.py;
the tools themselves: pipelines/book_translation/tools/ (exported from that repo, see SOURCE_REVISION).

  btp_kaggle_scheduler_tick   every 30 min: record each finished Kaggle run, copy its models to Drive
                              (tools/kaggle_scheduler.py tick; monitor-only unless its state says otherwise)
  btp_kaggle_scheduler_admin  scheduler status / resume-schedule (clear a halt) / monitor-only
  btp_kaggle_status           Kaggle run status, notebook version, weekly GPU quota
  btp_kaggle_push             start a Kaggle GPU run (tools/kaggle_run.py push) — dry_run first;
                              refuses while a version is running or queued
  btp_kaggle_logs             a run's logs / run_summary.json / stage results
  btp_models_to_drive         copy finished models + reports to My Drive/btp_workspace/models_kaggle
  btp_colab_logs              Colab run logs / reports from My Drive/btp_workspace
  btp_colab_notebook_push     replace the Colab notebook on Drive with the exported one — dry_run first

Setup (Admin → Connections / Variables) — see btp_airflow.py:
  Connection kaggle_default        Login = Kaggle username, Password = API key, Extra {"api_token": "KGAT_..."}
  Variable  btp_gdrive_token_json  Google OAuth authorized_user JSON (Drive with btp_workspace)
"""
from __future__ import annotations

import importlib.util
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import Param, Variable, dag, get_current_context, task

try:  # Airflow 3 task SDK
    from airflow.sdk import Connection
except ImportError:  # pragma: no cover
    Connection = None

try:  # longer runs (model copies) get their own deadline instead of the cluster's 1 h default
    from airflow.sdk.definitions.deadline import AsyncCallback, DeadlineAlert, DeadlineReference
    from airflow_local_settings import sentry_deadline_callback

    def _deadline(hours: float):
        return DeadlineAlert(reference=DeadlineReference.DAGRUN_QUEUED_AT, interval=timedelta(hours=hours),
                             callback=AsyncCallback(sentry_deadline_callback))
except Exception:  # noqa: BLE001 -- no custom deadline: the cluster policy's 1 h applies
    def _deadline(hours: float):
        return None

log = logging.getLogger(__name__)

BTP_DIR = Path(os.getenv("PIPELINES_DIR", "/opt/airflow/pipelines")) / "book_translation"


def _ops():
    """pipelines/book_translation/btp_airflow.py, loaded by path (no sys.path changes)."""
    spec = importlib.util.spec_from_file_location("btp_airflow", BTP_DIR / "btp_airflow.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OPS = _ops()


def _kaggle_env() -> dict:
    if Connection is not None:
        c = Connection.get(OPS.KAGGLE_CONN_ID)
    else:  # pragma: no cover
        from airflow.hooks.base import BaseHook
        c = BaseHook.get_connection(OPS.KAGGLE_CONN_ID)
    return OPS.kaggle_env(c.login, c.password, c.extra_dejson)


def _drive_token() -> str:
    return Variable.get(OPS.GDRIVE_TOKEN_VARIABLE)


def _params() -> dict:
    return get_current_context()["params"]


COMMON = dict(start_date=datetime(2026, 11, 1), catchup=False, max_active_runs=1,
              default_args={"owner": "data-eng", "retries": 0})
TAGS = ["book-translation", "kaggle", "ml"]


# ─── scheduler tick (every 30 min) ───────────────────────────────────────
# Created paused on every server (staging too): unpause it on production only, once its state file is in
# place -- two schedulers (another server, or the old PC cron) would record and copy every run twice.
@dag(dag_id="btp_kaggle_scheduler_tick", schedule="*/30 * * * *", tags=TAGS, deadline=_deadline(4),
     is_paused_upon_creation=True,
     description="Every 30 min: record finished Kaggle runs, copy their models to Drive",
     default_args={"owner": "data-eng", "retries": 1, "retry_delay": timedelta(minutes=5)},
     start_date=datetime(2026, 11, 1), catchup=False, max_active_runs=1,
     params={"dry_run": Param(False, type="boolean", title="Dry run", description="Decide and log; launch/sync nothing")})
def btp_kaggle_scheduler_tick():
    @task(execution_timeout=timedelta(hours=3))
    def tick() -> str:
        return OPS.scheduler_tick(_kaggle_env(), _drive_token(), dry_run=bool(_params()["dry_run"]))[-2000:]
    tick()


@dag(dag_id="btp_kaggle_scheduler_admin", schedule=None, tags=TAGS, **COMMON,
     description="Kaggle scheduler: status / resume-schedule (clear a halt) / monitor-only",
     params={"action": Param("status", type="string", enum=OPS.SCHEDULER_ACTIONS, title="Action")})
def btp_kaggle_scheduler_admin():
    @task(execution_timeout=timedelta(minutes=10))
    def admin() -> str:
        return OPS.scheduler_admin(_params()["action"])[-2000:]
    admin()


# ─── Kaggle ──────────────────────────────────────────────────────────────
@dag(dag_id="btp_kaggle_status", schedule=None, tags=TAGS, **COMMON,
     description="Kaggle run status, notebook version and weekly GPU quota")
def btp_kaggle_status():
    @task(execution_timeout=timedelta(minutes=15))
    def status() -> str:
        return OPS.kaggle_status(_kaggle_env())[-2000:]
    status()


@dag(dag_id="btp_kaggle_push", schedule=None, tags=TAGS, **COMMON,
     description="Start a Kaggle GPU run of the pipeline — dry_run first; refuses while a run is active",
     params={
         "dry_run": Param(True, type="boolean", title="Dry run",
                          description="On: build the notebook + metadata and log them; start nothing."),
         "job": Param("full", type="string", enum=OPS.JOBS, title="Job",
                      description="full = train/continue all stages; smoke = quick end-to-end check; b = new language; book"),
         "langs": Param("a", type="string", title="Languages",
                        description="'a' = the 17 target languages, 'b:text', or a list like 'swh,kik'"),
         "resume": Param(True, type="boolean", title="Resume",
                         description="Continue from the saved workspace (the most advanced recent version)"),
         "deadline_hours": Param(11.0, type="number", minimum=0.25, maximum=11.5, title="Deadline (hours)"),
         "accelerator": Param("NvidiaTeslaT4", type="string", enum=OPS.ACCELERATORS, title="GPU"),
         "smoke_lang": Param("saq", type="string", title="Smoke-test language"),
         "b_lang": Param("guz", type="string", title="Pipeline B language"),
         "allow_overlap": Param(False, type="boolean", title="Allow overlap (dangerous)",
                                description="Push even while a version is running/queued — it would restore the OLDER workspace"),
     })
def btp_kaggle_push():
    @task(execution_timeout=timedelta(minutes=20))
    def push() -> str:
        p = _params()
        return OPS.kaggle_push(_kaggle_env(), job=p["job"], langs=p["langs"], resume=bool(p["resume"]),
                               deadline_hours=float(p["deadline_hours"]), accelerator=p["accelerator"],
                               smoke_lang=p["smoke_lang"], b_lang=p["b_lang"], dry_run=bool(p["dry_run"]),
                               allow_overlap=bool(p["allow_overlap"]))[-3000:]
    push()


@dag(dag_id="btp_kaggle_logs", schedule=None, tags=TAGS, **COMMON,
     description="A Kaggle run's logs, run_summary.json and stage results",
     params={"version": Param(0, type="integer", minimum=0, title="Notebook version (0 = newest)")})
def btp_kaggle_logs():
    @task(execution_timeout=timedelta(minutes=30))
    def logs() -> str:
        return OPS.kaggle_logs(_kaggle_env(), int(_params()["version"]) or None)[-3000:]
    logs()


@dag(dag_id="btp_models_to_drive", schedule=None, tags=TAGS, deadline=_deadline(4),
     description="Copy finished models + reports from Kaggle to My Drive/btp_workspace/models_kaggle",
     start_date=datetime(2026, 11, 1), catchup=False, max_active_runs=1,
     default_args={"owner": "data-eng", "retries": 1, "retry_delay": timedelta(minutes=10)},
     params={
         "version": Param(0, type="integer", minimum=0, title="Notebook version (0 = most advanced recent)"),
         "dry_run": Param(False, type="boolean", title="Dry run", description="List what would be copied"),
         "include_checkpoints": Param(False, type="boolean", title="Include checkpoints"),
     })
def btp_models_to_drive():
    @task(execution_timeout=timedelta(hours=3))
    def copy() -> str:
        p = _params()
        return OPS.models_to_drive(_kaggle_env(), _drive_token(), version=int(p["version"]) or None,
                                   dry_run=bool(p["dry_run"]), include_checkpoints=bool(p["include_checkpoints"]))[-2000:]
    copy()


# ─── Colab / Drive ───────────────────────────────────────────────────────
@dag(dag_id="btp_colab_logs", schedule=None, tags=TAGS, **COMMON,
     description="Colab run logs / reports from My Drive/btp_workspace",
     params={
         "ws": Param("btp_workspace", type="string", title="Workspace under My Drive"),
         "tail": Param(60, type="integer", minimum=1, maximum=2000, title="Lines per log"),
         "logs": Param(1, type="integer", minimum=1, maximum=20, title="Most recent logs"),
         "reports": Param(False, type="boolean", title="Also show reports"),
     })
def btp_colab_logs():
    @task(execution_timeout=timedelta(minutes=15))
    def colab_logs() -> str:
        p = _params()
        return OPS.colab_logs(_drive_token(), ws=p["ws"], tail=int(p["tail"]), logs=int(p["logs"]),
                              reports=bool(p["reports"]))[-3000:]
    colab_logs()


@dag(dag_id="btp_colab_notebook_push", schedule=None, tags=TAGS, **COMMON,
     description="Replace the Colab notebook on Drive with the exported one — dry_run first",
     params={
         "dry_run": Param(True, type="boolean", title="Dry run",
                          description="On: check the notebook and the Drive file; change nothing."),
         "file_id": Param(OPS.COLAB_NOTEBOOK_FILE_ID, type="string", title="Drive file id of the Colab notebook"),
     })
def btp_colab_notebook_push():
    @task(execution_timeout=timedelta(minutes=15))
    def push_notebook() -> str:
        p = _params()
        return OPS.push_colab_notebook(_drive_token(), file_id=p["file_id"], dry_run=bool(p["dry_run"]))[-2000:]
    push_notebook()


btp_kaggle_scheduler_tick()
btp_kaggle_scheduler_admin()
btp_kaggle_status()
btp_kaggle_push()
btp_kaggle_logs()
btp_models_to_drive()
btp_colab_logs()
btp_colab_notebook_push()
