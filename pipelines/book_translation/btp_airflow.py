"""
Run the book-translation-pipeline operational tools from Airflow (dags/book_translation_dag.py).

The tools in tools/ and the prebuilt notebooks in notebooks/ are a verbatim copy of the
book-translation-pipeline repo (tools/export_to_airflow.sh there; revision in SOURCE_REVISION).
This module only wires them to Airflow: credentials, temp files, logging, input checks. No Airflow
import here, so it is unit-tested without Airflow (pipelines/tests/test_book_translation.py).

Credentials (never written into the repo or the task log):
  Connection kaggle_default   Login = Kaggle username, Password = legacy API key,
                              Extra {"api_token": "KGAT_..."} (the new-style token; either works)
                              -> passed to the tools only as KAGGLE_USERNAME / KAGGLE_KEY / KAGGLE_API_TOKEN
  Variable btp_gdrive_token_json   Google OAuth "authorized_user" JSON for the Drive that holds
                              My Drive/btp_workspace -> written to a 0600 temp file for the task, deleted after

State: the Kaggle scheduler's state + log live in pipelines/state/book_translation/ (git-ignored,
bind-mounted, so they survive container restarts).

The `kaggle` package: used from the Airflow image if it has it; otherwise the tools run in a small venv in
the state folder (created on first use, `--system-site-packages`, so it adds only kaggle and its missing
dependencies and uses the image's versions of everything else). This keeps the Airflow image -- and every
running task on the cluster -- untouched when this DAG is deployed.
"""
from __future__ import annotations

import collections
import contextlib
import fcntl
import importlib.util
import json
import shutil
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE / "tools"
NOTEBOOKS = HERE / "notebooks"
STATE_DIR = Path(os.getenv("BTP_STATE_DIR", HERE.parent / "state" / "book_translation"))
SCHED_STATE = STATE_DIR / "scheduler.json"
VENV = STATE_DIR / "venv"
KAGGLE_REQUIREMENT = "kaggle==2.2.5"

KAGGLE_CONN_ID = "kaggle_default"
GDRIVE_TOKEN_VARIABLE = "btp_gdrive_token_json"
COLAB_NOTEBOOK_FILE_ID = "1rcR2SDLa6zWifQ7Cfo70cj2ZAUkzDdDK"   # same default as tools/push_notebook.py

JOBS = ["full", "smoke", "b", "book"]
ACCELERATORS = ["NvidiaTeslaT4", "NvidiaTeslaP100", "kaggle-default"]
SCHEDULER_ACTIONS = ["status", "resume-schedule", "monitor-only"]
_LANGS_RE = re.compile(r"^[a-z0-9]+(:[a-z]+)?(,[a-z0-9]+)*$")       # 'a', 'b:text', 'swh,kik'
_NOISE = ("file_cache is only supported with oauth2client",)

log = logging.getLogger(__name__)


class ToolError(RuntimeError):
    """A tool exited non-zero; the message carries its last output lines."""


# ─── credentials ─────────────────────────────────────────────────────────
def kaggle_env(login: str | None, password: str | None, extra: dict | str | None) -> dict[str, str]:
    """Environment for the Kaggle CLI/SDK from the kaggle_default connection."""
    if isinstance(extra, str):
        extra = json.loads(extra) if extra.strip() else {}
    extra = extra or {}
    token = extra.get("api_token") or extra.get("KAGGLE_API_TOKEN")
    if not login:
        raise ValueError(f"connection {KAGGLE_CONN_ID}: Login must be the Kaggle username")
    if not (password or token):
        raise ValueError(f"connection {KAGGLE_CONN_ID}: set Password (API key) or Extra api_token")
    env = {"KAGGLE_USERNAME": login}
    if password:
        env["KAGGLE_KEY"] = password
    if token:
        env["KAGGLE_API_TOKEN"] = token
    return env


@contextlib.contextmanager
def drive_token_file(token_json: str):
    """The Drive OAuth token as a private temp file for the tools' --token; removed afterwards."""
    try:
        data = json.loads(token_json)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Variable {GDRIVE_TOKEN_VARIABLE} is not JSON") from e
    if not isinstance(data, dict) or not data.get("refresh_token"):
        raise ValueError(f"Variable {GDRIVE_TOKEN_VARIABLE} must be an authorized_user token with a refresh_token")
    with tempfile.TemporaryDirectory(prefix="btp_gdrive_") as d:
        path = Path(d) / "gdrive_token.json"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        yield str(path)


# ─── the Python the tools run with ───────────────────────────────────────
def _has_kaggle(python: str) -> bool:
    # find_spec, not import: `import kaggle` tries to authenticate at import time
    code = "import importlib.util as u, sys; sys.exit(0 if u.find_spec('kaggle') else 1)"
    try:
        return subprocess.run([python, "-c", code], capture_output=True, timeout=120).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _venv_ready() -> bool:
    py, marker = VENV / "bin" / "python", VENV / ".btp_requirement"
    return (py.is_file() and marker.is_file() and marker.read_text().strip() == KAGGLE_REQUIREMENT
            and _has_kaggle(str(py)))


def _create_venv() -> None:
    shutil.rmtree(VENV, ignore_errors=True)
    steps = [[sys.executable, "-m", "venv", "--system-site-packages", str(VENV)],
             [str(VENV / "bin" / "python"), "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
              KAGGLE_REQUIREMENT]]
    for cmd in steps:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if r.returncode:
            shutil.rmtree(VENV, ignore_errors=True)
            raise ToolError(f"setting up {VENV} failed ({' '.join(cmd[-3:])}):\n{(r.stdout + r.stderr)[-1500:]}")
    (VENV / ".btp_requirement").write_text(KAGGLE_REQUIREMENT + "\n")


def tool_python() -> str:
    """The image's Python if it has `kaggle`, else the state-folder venv (created once, under a lock so
    two tasks never build it at the same time; rebuilt if the pin changes or the image's Python moved)."""
    if importlib.util.find_spec("kaggle") is not None:
        return sys.executable
    if _venv_ready():
        return str(VENV / "bin" / "python")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_DIR / "venv.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not _venv_ready():
            log.info("installing %s into %s (first use; a few minutes)", KAGGLE_REQUIREMENT, VENV)
            _create_venv()
    return str(VENV / "bin" / "python")


# ─── running a tool ──────────────────────────────────────────────────────
def run_tool(script: str, *args, env: dict[str, str] | None = None, keep: int = 400) -> str:
    """Run tools/<script> with tool_python(), streaming its output into the task log.
    Returns the last `keep` lines; raises ToolError on a non-zero exit."""
    cmd = [tool_python(), str(TOOLS / script), *map(str, args)]
    full_env = {**os.environ, **(env or {}), "BTP_SCHED_STATE": str(SCHED_STATE), "PYTHONUNBUFFERED": "1"}
    log.info("running tools/%s %s", script, " ".join(map(str, args)))
    tail: collections.deque[str] = collections.deque(maxlen=keep)
    proc = subprocess.Popen(cmd, cwd=TOOLS, env=full_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if any(n in line for n in _NOISE):
                continue
            log.info("%s", line)
            tail.append(line)
        rc = proc.wait()
    except BaseException:           # task killed / timed out: do not leave the tool running
        proc.kill()
        proc.wait()
        raise
    if rc:
        raise ToolError(f"tools/{script} exited {rc}:\n" + "\n".join(list(tail)[-25:]))
    return "\n".join(tail)


# ─── operations (one per DAG) ────────────────────────────────────────────
def kaggle_status(env: dict[str, str]) -> str:
    out = run_tool("kaggle_run.py", "status", env=env)
    if SCHED_STATE.is_file():
        out += "\n" + run_tool("kaggle_scheduler.py", "status", env=env)
    return out


def push_args(job: str, langs: str, resume: bool, deadline_hours: float, accelerator: str,
              smoke_lang: str = "saq", b_lang: str = "guz", dry_run: bool = True,
              allow_overlap: bool = False) -> list[str]:
    if job not in JOBS:
        raise ValueError(f"job must be one of {JOBS}")
    if not _LANGS_RE.match(langs or ""):
        raise ValueError(f"langs {langs!r}: use 'a', 'b:text' or a comma list like 'swh,kik'")
    if not 0.25 <= float(deadline_hours) <= 11.5:
        raise ValueError("deadline_hours must be between 0.25 and 11.5 (Kaggle kills runs at 12 h)")
    if accelerator not in ACCELERATORS:
        raise ValueError(f"accelerator must be one of {ACCELERATORS}")
    for name, v in (("smoke_lang", smoke_lang), ("b_lang", b_lang)):
        if not re.fullmatch(r"[a-z]{2,4}", v or ""):
            raise ValueError(f"{name} must be a 2-4 letter language code")
    args = ["push", "--job", job, "--langs", langs, "--deadline-hours", float(deadline_hours),
            "--smoke-lang", smoke_lang, "--b-lang", b_lang,
            "--accelerator", "" if accelerator == "kaggle-default" else accelerator]
    if resume:
        args.append("--resume")
    if dry_run:
        args.append("--dry-run")
    if allow_overlap:
        args.append("--allow-overlap")
    return args


def kaggle_push(env: dict[str, str], **params) -> str:
    return run_tool("kaggle_run.py", *push_args(**params), env=env)


def kaggle_logs(env: dict[str, str], version: int | None = None) -> str:
    with tempfile.TemporaryDirectory(prefix="btp_kaggle_logs_") as d:
        args = ["logs", "--out", d] + (["--version", int(version)] if version else [])
        return run_tool("kaggle_run.py", *args, env=env)


def models_to_drive(env: dict[str, str], token_json: str, version: int | None = None, dry_run: bool = False,
                    include_checkpoints: bool = False) -> str:
    with drive_token_file(token_json) as tok:
        args = ["--token", tok] + (["--version", int(version)] if version else [])
        args += (["--dry-run"] if dry_run else []) + (["--include-checkpoints"] if include_checkpoints else [])
        return run_tool("sync_models_to_drive.py", *args, env=env)


def colab_logs(token_json: str, ws: str = "btp_workspace", tail: int = 60, logs: int = 1,
               reports: bool = False) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*", ws or "") or ".." in ws:
        raise ValueError("ws must be a path under My Drive like btp_workspace or btp_workspace/smoke")
    with drive_token_file(token_json) as tok:
        args = ["--token", tok, "--ws", ws, "--tail", int(tail), "--logs", int(logs)]
        return run_tool("pull_logs.py", *args + (["--reports"] if reports else []))


def push_colab_notebook(token_json: str, file_id: str = COLAB_NOTEBOOK_FILE_ID, dry_run: bool = True,
                        drive_service=None) -> str:
    """Replace the Colab notebook on Drive with notebooks/colab_pipeline.ipynb (same file id, same link).
    dry_run: only check the notebook and that the Drive file is reachable."""
    nb_path = NOTEBOOKS / "colab_pipeline.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    if nb.get("nbformat") != 4 or not nb.get("cells"):
        raise ValueError(f"{nb_path} is not a notebook")
    if not re.fullmatch(r"[A-Za-z0-9_-]{10,}", file_id or ""):
        raise ValueError("file_id is not a Drive file id")
    if not dry_run:
        with drive_token_file(token_json) as tok:
            return run_tool("push_notebook.py", "--token", tok, "--file-id", file_id, "--notebook", nb_path)
    with drive_token_file(token_json) as tok:
        if drive_service is None:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
            drive_service = build("drive", "v3", credentials=Credentials.from_authorized_user_file(tok),
                                  cache_discovery=False)
        meta = drive_service.files().get(fileId=file_id, fields="id,name,modifiedTime,size").execute()
    msg = (f"dry run: would replace Drive file {meta.get('name')} ({file_id}, modified {meta.get('modifiedTime')}, "
           f"{meta.get('size')} bytes) with {nb_path.name} ({len(nb['cells'])} cells, {nb_path.stat().st_size} bytes)")
    log.info("%s", msg)
    return msg


def scheduler_tick(env: dict[str, str], token_json: str, dry_run: bool = False) -> str:
    """One tick of tools/kaggle_scheduler.py: records each finished Kaggle run and copies its models to Drive
    (monitor-only), or launches the next run (scheduling mode) -- whatever the state file says."""
    if not SCHED_STATE.is_file():
        # a missing state means phase resume_test_1 with launching enabled: never start GPU runs by accident
        raise ToolError(f"no scheduler state at {SCHED_STATE}: copy it from the machine that ran the scheduler "
                        "(~/.local/state/btp/scheduler.json), or trigger btp_kaggle_scheduler_admin with "
                        "action=monitor-only to start a monitor-only one")
    with drive_token_file(token_json) as tok:
        return run_tool("kaggle_scheduler.py", "tick", "--token", tok, *(["--dry-run"] if dry_run else []), env=env)


def scheduler_admin(action: str) -> str:
    if action not in SCHEDULER_ACTIONS:
        raise ValueError(f"action must be one of {SCHEDULER_ACTIONS}")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return run_tool("kaggle_scheduler.py", action)
