import ast
import threading
from pathlib import Path

import pytest

import facility_to_snowflake_fast_resume as facility_loader
import snowflake_to_v3_migration as snowflake_migration
import v2_to_v3_api_migration as v2v3


def test_facility_loader_progress_merges_parallel_completion_marks(tmp_path, monkeypatch):
    monkeypatch.setattr(facility_loader, "PROGRESS_FILE", tmp_path / ".progress.json")
    threads = [
        threading.Thread(
            target=facility_loader._mark_done,
            args=("facility", "run-1", {"module": "Core", "table": f"table_{n}"}, f"s3/{n}"),
        )
        for n in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    completed = facility_loader._load_progress()["facility"]["completed"]
    assert set(completed) == {f"core|table_{n}" for n in range(8)}


def test_facility_loader_fails_closed_on_corrupt_resume_state(tmp_path, monkeypatch):
    path = tmp_path / "page-state.json"
    path.write_text("{truncated")
    with pytest.raises(RuntimeError, match="Cannot safely resume page extraction"):
        facility_loader._load_page_state(path)

    monkeypatch.setattr(facility_loader, "PROGRESS_FILE", path)
    with pytest.raises(RuntimeError, match="Cannot safely resume"):
        facility_loader._load_progress()


def test_v2_migration_progress_is_persisted_and_corruption_is_not_ignored(tmp_path, monkeypatch):
    progress = tmp_path / ".migration_progress.json"
    monkeypatch.setattr(v2v3, "PROGRESS_FILE", progress)
    monkeypatch.setattr(v2v3, "DONE_FILE", tmp_path / ".migration_done.json")
    monkeypatch.setattr(v2v3, "_completed_jobs", set())
    monkeypatch.setattr(v2v3, "_permanently_done", set())

    v2v3._mark_done("run-1", "facility|namespace")
    assert v2v3._load_progress("run-1") == {"facility|namespace"}

    progress.write_text("{truncated")
    with pytest.raises(RuntimeError, match="Cannot safely resume"):
        v2v3._load_progress("run-1")


def test_snowflake_migration_does_not_replace_corrupt_uuid_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / ".migration_id_to_uuid.json"
    path.write_text("{truncated")
    monkeypatch.setattr(snowflake_migration, "ID_TO_UUID_FILE", path)
    monkeypatch.setattr(snowflake_migration, "_id_to_uuid", {})
    monkeypatch.setattr(snowflake_migration, "_alias_to_table", {})
    monkeypatch.setattr(snowflake_migration, "_uuid_to_id", {})

    with pytest.raises(RuntimeError, match="Cannot safely checkpoint UUID links"):
        snowflake_migration._add_uuids("visits", {1: "visit-1"})
    assert path.read_text() == "{truncated"


def test_airflow_loader_tasks_declare_retry_policies():
    path = Path(__file__).resolve().parents[2] / "dags" / "v2_facility_to_snowflake.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

    for name, retries in (("extract_load", 3), ("finalize", 2)):
        task = next(
            decorator for decorator in functions[name].decorator_list
            if isinstance(decorator, ast.Call) and
            (isinstance(decorator.func, ast.Name) and decorator.func.id == "task")
        )
        kwargs = {kw.arg: kw.value for kw in task.keywords}
        assert ast.literal_eval(kwargs["retries"]) == retries
        assert ast.literal_eval(kwargs["retry_exponential_backoff"]) is True
        assert "max_retry_delay" in kwargs
