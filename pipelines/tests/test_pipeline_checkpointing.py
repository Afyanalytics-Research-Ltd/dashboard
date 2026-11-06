import json
import multiprocessing
import threading

import pytest

from pipeline_checkpointing import CheckpointStore, retry_call, run_checkpointed, update_json_file


def _process_update_json(path, value):
    update_json_file(path, lambda state: {**state, str(value): value}, default={})


def _process_append_jsonl(root, value):
    CheckpointStore(root).append_jsonl("records", "job", {"value": value})


def test_pipeline_runs_have_isolated_resumable_checkpoints(tmp_path):
    first = CheckpointStore.for_pipeline(tmp_path, "aroha/import", "run/1")
    first.save("progress", "products", {"last_id": 25, "complete": False})

    resumed = CheckpointStore.for_pipeline(tmp_path, "aroha/import", "run/1")
    other_run = CheckpointStore.for_pipeline(tmp_path, "aroha/import", "run/2")
    other_pipeline = CheckpointStore.for_pipeline(tmp_path, "claims/import", "run/1")

    assert resumed.load("progress", "products") == {"last_id": 25, "complete": False}
    assert other_run.load("progress", "products") is None
    assert other_pipeline.load("progress", "products") is None
    assert other_pipeline.path_for("progress", "products") != \
        CheckpointStore.for_pipeline(tmp_path, "claims_import", "run/1").path_for("progress", "products")


def test_failed_checkpoint_write_preserves_last_good_state(tmp_path):
    store = CheckpointStore(tmp_path)
    store.save("progress", "job", {"cursor": 10})
    circular = []
    circular.append(circular)

    with pytest.raises(ValueError, match="Circular reference"):
        store.save("progress", "job", circular)

    assert store.load("progress", "job") == {"cursor": 10}
    assert not list(tmp_path.rglob("*.tmp"))


def test_jsonl_appends_are_thread_safe_and_complete(tmp_path):
    store = CheckpointStore(tmp_path)
    threads = [
        threading.Thread(target=lambda n=n: [store.append_jsonl("errors", "job", {"n": n, "i": i})
                                            for i in range(20)])
        for n in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    lines = store.path_for("errors", "job", "jsonl").read_text().splitlines()
    assert len(lines) == 160
    assert all(isinstance(json.loads(line), dict) for line in lines)


def test_json_updates_merge_across_processes(tmp_path):
    path = tmp_path / "progress.json"
    processes = [multiprocessing.Process(target=_process_update_json, args=(path, n)) for n in range(6)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert json.loads(path.read_text()) == {str(n): n for n in range(6)}


def test_jsonl_appends_remain_whole_across_processes(tmp_path):
    processes = [multiprocessing.Process(target=_process_append_jsonl, args=(tmp_path, n)) for n in range(6)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    path = CheckpointStore(tmp_path).path_for("records", "job", "jsonl")
    assert {json.loads(line)["value"] for line in path.read_text().splitlines()} == set(range(6))


def test_checkpointed_runner_resumes_after_interruption(tmp_path):
    store = CheckpointStore.for_pipeline(tmp_path, "sample", "run-1")
    completed = []

    def process(item):
        if item == 3:
            raise RuntimeError("interrupted")
        completed.append(item)

    with pytest.raises(RuntimeError, match="interrupted"):
        run_checkpointed(range(1, 5), process, store=store, job="table",
                         item_key=lambda item: str(item))

    assert completed == [1, 2]
    result = run_checkpointed(range(1, 5), completed.append, store=store, job="table",
                              item_key=lambda item: str(item))
    assert result == {"processed": 2, "skipped": 2, "completed": 4}
    assert completed == [1, 2, 3, 4]


def test_checkpoint_journal_ignores_only_a_truncated_last_record(tmp_path):
    store = CheckpointStore(tmp_path)
    store.append_jsonl("completed", "table", {"key": "1"})
    path = store.path_for("completed", "table", "jsonl")
    with path.open("ab") as fh:
        fh.write(b'{"key": "2"')

    assert store.completed_keys("completed", "table") == {"1"}
    with path.open("ab") as fh:
        fh.write(b"}\n")
    assert store.completed_keys("completed", "table") == {"1", "2"}


def test_retry_call_retries_only_classified_errors_and_caps_backoff():
    attempts = 0
    delays = []

    def operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TimeoutError("temporary")
        return "ok"

    result = retry_call(
        operation,
        attempts=4,
        retry_if=lambda exc: isinstance(exc, TimeoutError),
        base_delay=2,
        max_delay=3,
        jitter=0,
        sleep=delays.append,
    )

    assert result == "ok"
    assert attempts == 3
    assert delays == [2, 3]


def test_checkpointed_runner_retries_transient_errors_before_checkpoint(tmp_path):
    store = CheckpointStore.for_pipeline(tmp_path, "sample", "run-2")
    attempts = {}

    def process(item):
        attempts[item] = attempts.get(item, 0) + 1
        if attempts[item] == 1:
            raise TimeoutError("temporary")

    result = run_checkpointed(
        ["a"],
        process,
        store=store,
        job="table",
        item_key=str,
        attempts=2,
        retry_if=lambda exc: isinstance(exc, TimeoutError),
        base_delay=0,
        jitter=0,
    )

    assert result == {"processed": 1, "skipped": 0, "completed": 1}
    assert attempts == {"a": 2}


def test_retry_call_propagates_non_retryable_and_exhausted_errors():
    def permanent_error():
        raise ValueError("permanent")

    with pytest.raises(ValueError, match="permanent"):
        retry_call(permanent_error, attempts=3,
                   retry_if=lambda exc: isinstance(exc, TimeoutError), sleep=lambda _: None)

    attempts = 0

    def unavailable():
        nonlocal attempts
        attempts += 1
        raise TimeoutError("still down")

    with pytest.raises(TimeoutError, match="still down"):
        retry_call(unavailable, attempts=2, retry_if=lambda exc: isinstance(exc, TimeoutError),
                   sleep=lambda _: None)
    assert attempts == 2


def test_empty_or_corrupt_checkpoint_names_the_file(tmp_path):
    from pipeline_checkpointing import CorruptCheckpointError

    path = tmp_path / "state.json"
    path.write_text("")
    with pytest.raises(CorruptCheckpointError, match="state.json: file is empty"):
        update_json_file(path, lambda state: state, default={})
    path.write_text("{truncated")
    with pytest.raises(ValueError, match="Cannot safely resume from .*state.json"):
        CheckpointStore.load_path(path)
    assert path.read_text() == "{truncated"


def test_stale_temp_files_from_killed_writers_are_swept(tmp_path):
    import os
    import time

    import pipeline_checkpointing as pc

    target = tmp_path / "state.json"
    stale = tmp_path / ".state.json.abc123.tmp"
    fresh = tmp_path / ".state.json.def456.tmp"     # could be a live writer's
    other = tmp_path / ".other.json.ghi789.tmp"
    for tmp in (stale, fresh, other):
        tmp.write_text("partial")
    old = time.time() - pc.STALE_TMP_SECONDS - 60
    os.utime(stale, (old, old))
    os.utime(other, (old, old))

    pc.atomic_write_json(target, {"ok": True})

    assert not stale.exists()
    assert fresh.exists() and other.exists()
    assert pc.load_json(target) == {"ok": True}


def test_retry_wait_never_exceeds_max_delay_with_jitter():
    waits = []
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        if calls < 5:
            raise TimeoutError()
        return "ok"

    retry_call(operation, attempts=5, retry_if=lambda exc: isinstance(exc, TimeoutError),
               base_delay=1, max_delay=4, jitter=0.5, sleep=waits.append, random_value=lambda: 1.0)
    assert waits == [1.5, 3.0, 4.0, 4.0]
