"""Reusable file-backed checkpoints and bounded retries for pipeline jobs.

Use ``CheckpointStore.for_pipeline(root, pipeline, run_id)`` to isolate a
pipeline run. Persist only after the corresponding work has safely completed;
operations passed to ``retry_call`` must be idempotent or retry-safe.

    state = CheckpointStore.for_pipeline(".state", "my_pipeline", run_id)
    cursor = state.load("progress", "customers", {"after_id": 0})
    # process from cursor, then:
    state.save("progress", "customers", {"after_id": last_id})
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from collections.abc import Iterable
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")
_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()
# Temp files older than this are leftovers from a writer killed mid-write
# (a live writer replaces its temp file within milliseconds).
STALE_TMP_SECONDS = 3600
_SWEPT_PATHS: set[str] = set()
_SWEPT_GUARD = threading.Lock()

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None


def _component(value: str) -> str:
    if not value or "\x00" in value:
        raise ValueError("state namespace values must be non-empty and contain no NUL bytes")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")[:48] or "item"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"{slug}-{digest}"


class CorruptCheckpointError(RuntimeError, ValueError):
    """A checkpoint file exists but is not valid JSON.

    Subclasses ValueError so existing ``except ValueError`` handlers behave as
    they did with the bare ``json.JSONDecodeError`` this replaces.
    """


def load_json(path: str | Path, default=None):
    """Load a JSON checkpoint, or ``default`` when it does not exist.

    An empty or malformed file raises ``CorruptCheckpointError`` naming the
    file, rather than a bare JSONDecodeError, so the operator knows which
    state file to inspect or delete.
    """
    path = Path(path)
    if not path.exists():
        return default
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except ValueError as exc:
        problem = "file is empty" if not text.strip() else str(exc)
        raise CorruptCheckpointError(
            f"Cannot safely resume from {path}: {problem} — inspect it, or delete it "
            f"to rebuild that state from scratch"
        ) from exc


def _sweep_stale_tmp(path: Path) -> None:
    """Remove temp files a killed writer left beside ``path`` (once per process)."""
    key = str(path)
    with _SWEPT_GUARD:
        if key in _SWEPT_PATHS:
            return
        _SWEPT_PATHS.add(key)
    cutoff = time.time() - STALE_TMP_SECONDS
    for tmp in path.parent.glob(f".{glob.escape(path.name)}.*.tmp"):
        try:
            if tmp.stat().st_mtime < cutoff:
                tmp.unlink()
        except OSError:
            pass  # another process removed or replaced it first


def atomic_write_json(path: str | Path, data) -> None:
    """Durably replace a JSON file without exposing a partial checkpoint."""
    atomic_write_text(path, json.dumps(data, default=str))


def atomic_write_text(path: str | Path, text: str) -> None:
    """Durably replace a text file without exposing a partial checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _sweep_stale_tmp(path)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        if os.name == "posix":
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def _file_lock(path: Path):
    key = str(path.resolve())
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.setdefault(key, threading.RLock())
    with lock:
        if fcntl is None:
            yield
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_name(path.name + ".lock").open("a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)


def update_json_file(path: str | Path, update: Callable[[object], object], *, default=None):
    """Lock, load, update, and durably replace a JSON file as one transaction."""
    path = Path(path)
    with _file_lock(path):
        data = load_json(path, default)
        result = update(data)
        atomic_write_json(path, result)
        return result


class CheckpointStore:
    """Thread-safe JSON/JSONL storage scoped to a directory or pipeline run."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._lock = threading.RLock()

    @classmethod
    def for_pipeline(cls, root: str | Path, pipeline: str, run_id: str) -> "CheckpointStore":
        return cls(Path(root) / _component(pipeline) / _component(run_id))

    def path_for(self, kind: str, key: str, extension: str = "json") -> Path:
        if not re.fullmatch(r"[A-Za-z0-9]+", extension):
            raise ValueError(f"invalid checkpoint file extension: {extension!r}")
        return self.root / _component(kind) / f"{_component(key)}.{extension}"

    def load(self, kind: str, key: str, default=None):
        return self.load_path(self.path_for(kind, key), default)

    @staticmethod
    def load_path(path: str | Path, default=None):
        return load_json(path, default)

    def save(self, kind: str, key: str, data) -> None:
        self.save_path(self.path_for(kind, key), data)

    @staticmethod
    def save_path(path: str | Path, data) -> None:
        atomic_write_json(path, data)

    def append_jsonl(self, kind: str, key: str, record) -> None:
        self.append_jsonl_path(self.path_for(kind, key, "jsonl"), record)

    def append_jsonl_path(self, path: str | Path, record) -> None:
        path = Path(path)
        encoded = (json.dumps(record, default=str, separators=(",", ":")) + "\n").encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, _file_lock(path):
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o666)
            with os.fdopen(fd, "ab") as fh:
                fh.write(encoded)
                fh.flush()
                os.fsync(fh.fileno())

    def completed_keys(self, kind: str, key: str) -> set[str]:
        """Load successful work-item keys from an append-only completion journal.

        A truncated final line is ignored because its completion was not
        durably recorded; malformed complete lines are reported as corruption.
        """
        path = self.path_for(kind, key, "jsonl")
        if not path.exists():
            return set()
        data = path.read_bytes()
        lines = data.splitlines(keepends=True)
        if lines and not lines[-1].endswith(b"\n"):
            lines.pop()
        completed = set()
        for number, line in enumerate(lines, start=1):
            try:
                record = json.loads(line)
                item_key = record["key"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"invalid checkpoint journal {path} at line {number}") from exc
            if not isinstance(item_key, str) or not item_key:
                raise ValueError(f"invalid checkpoint key in {path} at line {number}")
            completed.add(item_key)
        return completed


def run_checkpointed(
    items: Iterable[T],
    process: Callable[[T], object],
    *,
    store: CheckpointStore,
    job: str,
    item_key: Callable[[T], str],
    attempts: int = 1,
    retry_if: Callable[[Exception], bool] | None = None,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: float = 0.1,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> dict[str, int]:
    """Run idempotent work items, checkpointing each success before continuing.

    On restart, completed keys are skipped. The checkpoint follows successful
    processing, so an interruption between processing and saving may replay one
    item; ``process`` must therefore be idempotent or otherwise retry-safe.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least one")
    if attempts > 1 and retry_if is None:
        raise ValueError("retry_if is required when attempts is greater than one")

    completed = store.completed_keys("completed", job)
    skipped = processed = 0
    for item in items:
        key = item_key(item)
        if not isinstance(key, str) or not key:
            raise ValueError("item_key must return a non-empty string")
        if key in completed:
            skipped += 1
            continue

        retry_call(
            lambda: process(item),
            attempts=attempts,
            retry_if=retry_if or (lambda _: False),
            base_delay=base_delay,
            max_delay=max_delay,
            jitter=jitter,
            on_retry=on_retry,
        )
        store.append_jsonl("completed", job, {"key": key, "completed_at": time.time()})
        completed.add(key)
        processed += 1
    return {"processed": processed, "skipped": skipped, "completed": len(completed)}


def retry_call(
    operation: Callable[[], T],
    *,
    attempts: int,
    retry_if: Callable[[Exception], bool],
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: float = 0.1,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    random_value: Callable[[], float] = random.random,
) -> T:
    """Retry explicitly classified transient errors with bounded backoff.

    ``retry_if`` is mandatory so callers decide which failures are transient.
    ``attempts`` includes the initial call; terminal exceptions are re-raised.
    No single wait exceeds ``max_delay``, jitter included.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least one")
    if base_delay < 0 or max_delay < 0:
        raise ValueError("retry delays must not be negative")
    if not 0 <= jitter <= 1:
        raise ValueError("jitter must be between zero and one")

    delay = min(base_delay, max_delay)
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt == attempts or not retry_if(exc):
                raise
            # jitter first, then clamp, so max_delay is a hard ceiling
            wait = min(max_delay, max(0.0, delay * (1 + jitter * (2 * random_value() - 1))))
            if on_retry is not None:
                on_retry(attempt, exc, wait)
            sleep(wait)
            delay = min(delay * 2, max_delay)
    raise RuntimeError("unreachable retry state")
