"""
dags/book_translation_dag.py + pipelines/book_translation/btp_airflow.py: credentials handling, tool
invocation, input checks, the scheduler's no-state guard, and that the exported tools work as a
tools-only copy (prebuilt notebook, no pipeline sources). The tools' own logic is tested in the
book-translation-pipeline repo.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BTP = ROOT / "pipelines" / "book_translation"
DAG_FILE = ROOT / "dags" / "book_translation_dag.py"

TOKEN = json.dumps({"type": "authorized_user", "client_id": "cid", "client_secret": "cs", "refresh_token": "rt"})


def _load_ops():
    spec = importlib.util.spec_from_file_location("btp_airflow_under_test", BTP / "btp_airflow.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def ops(tmp_path, monkeypatch):
    mod = _load_ops()
    monkeypatch.setattr(mod, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(mod, "SCHED_STATE", tmp_path / "state" / "scheduler.json")
    monkeypatch.setattr(mod, "VENV", tmp_path / "state" / "venv")
    return mod


@pytest.fixture
def fake_tools(ops, tmp_path, monkeypatch):
    """Replace tools/ with scripts that record argv, selected env and the --token file's state."""
    tools = tmp_path / "tools"
    tools.mkdir()
    body = textwrap.dedent("""
        import json, os, sys
        from pathlib import Path
        argv = sys.argv[1:]
        tok = argv[argv.index("--token") + 1] if "--token" in argv else None
        rec = {"script": Path(sys.argv[0]).name, "argv": argv,
               "env": {k: os.environ.get(k) for k in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN",
                                                      "BTP_SCHED_STATE")},
               "token": json.loads(Path(tok).read_text()) if tok else None,
               "token_mode": oct(os.stat(tok).st_mode & 0o777) if tok else None, "token_path": tok}
        print("file_cache is only supported with oauth2client<4.0.0")
        print("RECORD " + json.dumps(rec))
        sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
    """)
    for name in ("kaggle_run.py", "kaggle_scheduler.py", "sync_models_to_drive.py", "pull_logs.py",
                 "push_notebook.py"):
        (tools / name).write_text(body)
    monkeypatch.setattr(ops, "TOOLS", tools)
    monkeypatch.setattr(ops, "tool_python", lambda: sys.executable)
    return tools


def _record(out: str) -> dict:
    return json.loads(next(line for line in out.splitlines() if line.startswith("RECORD "))[7:])


ENV = {"KAGGLE_USERNAME": "someone", "KAGGLE_KEY": "k" * 32}


# ─── credentials ─────────────────────────────────────────────────────────
def test_kaggle_env_from_connection_fields(ops):
    assert ops.kaggle_env("u", "key", {"api_token": "KGAT_x"}) == {
        "KAGGLE_USERNAME": "u", "KAGGLE_KEY": "key", "KAGGLE_API_TOKEN": "KGAT_x"}
    assert ops.kaggle_env("u", None, '{"api_token": "KGAT_x"}') == {"KAGGLE_USERNAME": "u", "KAGGLE_API_TOKEN": "KGAT_x"}
    assert ops.kaggle_env("u", "key", "") == {"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "key"}


@pytest.mark.parametrize("login,password,extra", [(None, "k", None), ("u", None, None), ("u", "", {"other": 1})])
def test_kaggle_env_rejects_incomplete_connection(ops, login, password, extra):
    with pytest.raises(ValueError, match="kaggle_default"):
        ops.kaggle_env(login, password, extra)


def test_drive_token_file_is_private_and_removed(ops):
    with ops.drive_token_file(TOKEN) as p:
        assert json.loads(Path(p).read_text())["refresh_token"] == "rt"
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    assert not Path(p).exists() and not Path(p).parent.exists()


@pytest.mark.parametrize("bad", ["not json", "[]", json.dumps({"type": "authorized_user"}), None])
def test_drive_token_must_have_refresh_token(ops, bad):
    with pytest.raises(ValueError, match="btp_gdrive_token_json"):
        with ops.drive_token_file(bad):
            pass


def test_drive_token_file_removed_even_when_the_tool_fails(ops, fake_tools, monkeypatch):
    monkeypatch.setenv("FAKE_EXIT", "3")
    seen = {}
    real = ops.drive_token_file

    def spy(tok):
        cm = real(tok)

        class Wrap:
            def __enter__(self):
                seen["path"] = cm.__enter__()
                return seen["path"]

            def __exit__(self, *a):
                return cm.__exit__(*a)
        return Wrap()
    monkeypatch.setattr(ops, "drive_token_file", spy)
    with pytest.raises(ops.ToolError):
        ops.models_to_drive(ENV, TOKEN)
    assert not Path(seen["path"]).exists()


# ─── the tools' Python: image or state-folder venv ───────────────────────
def test_tool_python_uses_the_image_when_it_has_kaggle(ops, monkeypatch):
    monkeypatch.setattr(ops.importlib.util, "find_spec", lambda name: object() if name == "kaggle" else None)
    monkeypatch.setattr(ops, "_create_venv", lambda: pytest.fail("must not build a venv"))
    assert ops.tool_python() == sys.executable and not ops.VENV.exists()


def _no_kaggle_in_image(ops, monkeypatch, built):
    monkeypatch.setattr(ops.importlib.util, "find_spec", lambda name: None)

    def fake_create():
        (ops.VENV / "bin").mkdir(parents=True, exist_ok=True)
        (ops.VENV / "bin" / "python").write_text("")
        (ops.VENV / ".btp_requirement").write_text(ops.KAGGLE_REQUIREMENT + "\n")
        built.append(1)
    monkeypatch.setattr(ops, "_create_venv", fake_create)
    monkeypatch.setattr(ops, "_has_kaggle", lambda py: Path(py).is_file())


def test_tool_python_builds_the_venv_once(ops, monkeypatch):
    built = []
    _no_kaggle_in_image(ops, monkeypatch, built)
    py = ops.tool_python()
    assert py == str(ops.VENV / "bin" / "python") and built == [1]
    assert ops.tool_python() == py and built == [1], "second call must reuse the venv"
    assert (ops.STATE_DIR / "venv.lock").exists()


def test_tool_python_rebuilds_when_the_pin_changes_or_python_moved(ops, monkeypatch):
    built = []
    _no_kaggle_in_image(ops, monkeypatch, built)
    ops.tool_python()
    monkeypatch.setattr(ops, "KAGGLE_REQUIREMENT", "kaggle==9.9.9")
    ops.tool_python()
    assert built == [1, 1]
    (ops.VENV / "bin" / "python").unlink()          # e.g. the image's Python was upgraded
    ops.tool_python()
    assert built == [1, 1, 1]


def test_create_venv_failure_cleans_up_and_reports(ops, monkeypatch):
    def fake_run(cmd, **kw):
        ops.VENV.mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(cmd, 1, "", "No matching distribution found for kaggle==2.2.5")
    monkeypatch.setattr(ops.subprocess, "run", fake_run)
    with pytest.raises(ops.ToolError, match="No matching distribution"):
        ops._create_venv()
    assert not ops.VENV.exists()


def test_has_kaggle_checks_without_importing(ops):
    """`import kaggle` authenticates at import time; the check must only look the package up."""
    assert ops._has_kaggle(sys.executable) in (True, False)
    assert ops._has_kaggle("/nonexistent/python") is False


# ─── running tools ───────────────────────────────────────────────────────
def test_run_tool_passes_env_and_state_filters_noise(ops, fake_tools):
    out = ops.run_tool("kaggle_run.py", "status", env=ENV)
    rec = _record(out)
    assert rec["argv"] == ["status"] and rec["env"]["KAGGLE_USERNAME"] == "someone"
    assert rec["env"]["BTP_SCHED_STATE"] == str(ops.SCHED_STATE)
    assert "file_cache" not in out


def test_run_tool_failure_raises_with_output(ops, fake_tools, monkeypatch):
    monkeypatch.setenv("FAKE_EXIT", "2")
    with pytest.raises(ops.ToolError, match="exited 2") as e:
        ops.run_tool("kaggle_run.py", "status", env=ENV)
    assert "RECORD" in str(e.value)


def test_models_to_drive_arguments_and_token(ops, fake_tools):
    rec = _record(ops.models_to_drive(ENV, TOKEN, version=12, dry_run=True, include_checkpoints=True))
    assert rec["script"] == "sync_models_to_drive.py"
    assert rec["argv"][2:] == ["--version", "12", "--dry-run", "--include-checkpoints"]
    assert rec["token"]["refresh_token"] == "rt" and rec["token_mode"] == "0o600"
    assert not Path(rec["token_path"]).exists()
    assert _record(ops.models_to_drive(ENV, TOKEN))["argv"][2:] == []


def test_kaggle_logs_uses_a_temp_dir_and_version(ops, fake_tools):
    rec = _record(ops.kaggle_logs(ENV, 12))
    out_dir = rec["argv"][rec["argv"].index("--out") + 1]
    assert rec["argv"][-2:] == ["--version", "12"] and not Path(out_dir).exists()
    assert "--version" not in _record(ops.kaggle_logs(ENV, None))["argv"]


def test_kaggle_status_adds_scheduler_status_when_state_exists(ops, fake_tools):
    assert ops.kaggle_status(ENV).count("RECORD") == 1
    ops.SCHED_STATE.parent.mkdir(parents=True)
    ops.SCHED_STATE.write_text("{}")
    out = ops.kaggle_status(ENV)
    assert [json.loads(line[7:])["script"] for line in out.splitlines() if line.startswith("RECORD")] == [
        "kaggle_run.py", "kaggle_scheduler.py"]


# ─── push ────────────────────────────────────────────────────────────────
def _push(**kw):
    p = dict(job="full", langs="a", resume=True, deadline_hours=11.0, accelerator="NvidiaTeslaT4")
    p.update(kw)
    return p


def test_push_args_default_is_a_dry_run(ops):
    args = ops.push_args(**_push())
    assert args[:5] == ["push", "--job", "full", "--langs", "a"]
    assert "--resume" in args and "--dry-run" in args and "--allow-overlap" not in args
    assert args[args.index("--accelerator") + 1] == "NvidiaTeslaT4"


def test_push_args_real_push_and_kaggle_default_gpu(ops):
    args = ops.push_args(**_push(dry_run=False, resume=False, accelerator="kaggle-default", langs="swh,kik"))
    assert "--dry-run" not in args and "--resume" not in args
    assert args[args.index("--accelerator") + 1] == "" and args[args.index("--langs") + 1] == "swh,kik"


@pytest.mark.parametrize("bad", [dict(job="rm"), dict(langs="a; rm -rf /"), dict(langs=""),
                                 dict(deadline_hours=12), dict(deadline_hours=0.1), dict(accelerator="TPU"),
                                 dict(smoke_lang="x y"), dict(b_lang="")])
def test_push_args_rejects_bad_input(ops, bad):
    with pytest.raises(ValueError):
        ops.push_args(**_push(**bad))


@pytest.mark.parametrize("langs", ["a", "b:text", "swh", "swh,kik,luo"])
def test_push_args_accepts_language_forms(ops, langs):
    assert ops.push_args(**_push(langs=langs))[4] == langs


def test_kaggle_push_runs_kaggle_run_push(ops, fake_tools):
    rec = _record(ops.kaggle_push(ENV, **_push(allow_overlap=True)))
    assert rec["script"] == "kaggle_run.py" and rec["argv"][0] == "push" and "--allow-overlap" in rec["argv"]


# ─── scheduler ───────────────────────────────────────────────────────────
def test_tick_refuses_without_state(ops, fake_tools):
    """No state file = phase resume_test_1 with launching enabled: must never start a GPU run by accident."""
    with pytest.raises(ops.ToolError, match="no scheduler state"):
        ops.scheduler_tick(ENV, TOKEN)


def test_tick_with_state(ops, fake_tools):
    ops.SCHED_STATE.parent.mkdir(parents=True)
    ops.SCHED_STATE.write_text(json.dumps({"phase": "full", "monitor_only": True, "runs": []}))
    rec = _record(ops.scheduler_tick(ENV, TOKEN, dry_run=True))
    assert rec["script"] == "kaggle_scheduler.py" and rec["argv"][0] == "tick" and rec["argv"][-1] == "--dry-run"
    assert rec["token"]["refresh_token"] == "rt" and rec["env"]["KAGGLE_KEY"] == "k" * 32


def test_scheduler_admin(ops, fake_tools):
    assert _record(ops.scheduler_admin("monitor-only"))["argv"] == ["monitor-only"]
    assert ops.STATE_DIR.is_dir()
    with pytest.raises(ValueError):
        ops.scheduler_admin("tick")


# ─── Colab ───────────────────────────────────────────────────────────────
def test_colab_logs_arguments(ops, fake_tools):
    rec = _record(ops.colab_logs(TOKEN, ws="btp_workspace/smoke", tail=25, logs=2, reports=True))
    assert rec["argv"][2:] == ["--ws", "btp_workspace/smoke", "--tail", "25", "--logs", "2", "--reports"]


@pytest.mark.parametrize("ws", ["../x", "a b", "", "/abs"])
def test_colab_logs_rejects_bad_workspace(ops, ws):
    with pytest.raises(ValueError):
        ops.colab_logs(TOKEN, ws=ws)


class _FakeDrive:
    def __init__(self):
        self.calls = []

    def files(self):
        return self

    def get(self, **kw):
        self.calls.append(kw)
        return self

    def execute(self):
        return {"id": "f", "name": "colab_pipeline.ipynb", "modifiedTime": "2026-11-24T06:00:00Z", "size": "1"}


def test_colab_notebook_dry_run_changes_nothing(ops, fake_tools):
    d = _FakeDrive()
    msg = ops.push_colab_notebook(TOKEN, dry_run=True, drive_service=d)
    assert msg.startswith("dry run") and d.calls[0]["fileId"] == ops.COLAB_NOTEBOOK_FILE_ID


def test_colab_notebook_real_push_uses_the_exported_notebook(ops, fake_tools):
    rec = _record(ops.push_colab_notebook(TOKEN, dry_run=False))
    assert rec["script"] == "push_notebook.py"
    assert rec["argv"][rec["argv"].index("--notebook") + 1] == str(BTP / "notebooks" / "colab_pipeline.ipynb")


def test_colab_notebook_rejects_bad_file_id(ops):
    with pytest.raises(ValueError):
        ops.push_colab_notebook(TOKEN, file_id="x/../y", dry_run=True, drive_service=_FakeDrive())


# ─── the exported copy itself ────────────────────────────────────────────
def test_export_is_complete():
    for f in ("kaggle_run.py", "kaggle_scheduler.py", "sync_models_to_drive.py", "pull_logs.py",
              "push_notebook.py", "build_notebook.py"):
        assert (BTP / "tools" / f).is_file(), f
    for f in ("kaggle_pipeline.ipynb", "colab_pipeline.ipynb"):
        nb = json.loads((BTP / "notebooks" / f).read_text(encoding="utf-8"))
        assert nb["nbformat"] == 4 and nb["cells"]
    assert (BTP / "SOURCE_REVISION").read_text().startswith("book-translation-pipeline ")
    assert not (BTP / "btp").exists(), "only the tools are exported, not the pipeline sources"


def test_exported_tools_use_the_prebuilt_notebook():
    """The real exported kaggle_run.py, as Airflow runs it: no pipeline sources -> prebuilt notebook,
    with the parameters cell that `push` rewrites."""
    code = ("import json, kaggle_run as K; nb = K.kaggle_notebook(); "
            "nb = K.set_params(nb, {'JOB': 'full', 'LANGS': 'swh,kik', 'DEADLINE_HOURS': 0.75}); "
            "src = ''.join(next(c for c in nb['cells'] if 'JOB = ' in ''.join(c['source']))['source']); "
            "print(json.dumps([K.SLUG, \"JOB = 'full'\" in src, \"LANGS = 'swh,kik'\" in src]))")
    r = subprocess.run([sys.executable, "-c", code], cwd=BTP / "tools", capture_output=True, text=True,
                       env={**os.environ, "KAGGLE_USERNAME": "someone"})
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == ["adventproclaimer-multilingual-translator", True, True]


def test_exported_tools_compile():
    for f in (BTP / "tools").glob("*.py"):
        compile(f.read_text(encoding="utf-8"), str(f), "exec")


# ─── the DAG file ────────────────────────────────────────────────────────
EXPECTED_DAGS = {"btp_kaggle_scheduler_tick", "btp_kaggle_scheduler_admin", "btp_kaggle_status", "btp_kaggle_push",
                 "btp_kaggle_logs", "btp_models_to_drive", "btp_colab_logs", "btp_colab_notebook_push"}


def test_dag_file_declares_each_dag_once_and_calls_it():
    tree = ast.parse(DAG_FILE.read_text(encoding="utf-8"))
    ids, called = [], set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "dag":
            ids += [kw.value.value for kw in node.keywords if kw.arg == "dag_id"]
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            called.add(node.value.func.id)
    assert sorted(ids) == sorted(EXPECTED_DAGS) and EXPECTED_DAGS <= called


def test_dag_file_imports_against_the_stub(monkeypatch):
    from tests import airflow_stub  # noqa: F401  (installed by conftest)
    import airflow.sdk as sdk
    from airflow.models import Variable
    monkeypatch.setattr(sdk, "Variable", Variable, raising=False)
    monkeypatch.setenv("PIPELINES_DIR", str(ROOT / "pipelines"))
    spec = importlib.util.spec_from_file_location("book_translation_dag_under_test", DAG_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.OPS.KAGGLE_CONN_ID == "kaggle_default" and mod.BTP_DIR == ROOT / "pipelines" / "book_translation"
    for name in EXPECTED_DAGS:
        assert callable(getattr(mod, name))
