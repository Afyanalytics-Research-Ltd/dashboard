"""Background jobs: run engine work in a thread, stream progress + logs into the Job row.

The UI polls ``/api/jobs/<id>/?since=<n>`` for new log lines. Writes are
throttled (every ~0.7 s, or immediately for warnings/errors and phase changes).
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from typing import Callable

from django.db import close_old_connections, connection as db_connection
from django.utils import timezone

from .models import Job

logger = logging.getLogger(__name__)
MAX_LOG = 2000
_Thread = threading.Thread  # own reference so tests can run jobs inline without touching thread pools


class Cancelled(Exception):
    pass


class JobContext:
    def __init__(self, job: Job):
        self.job = job
        self._log: list[dict] = list(job.log or [])
        self._dirty = False
        self._last_flush = 0.0
        self._lock = threading.Lock()
        self._fields = {}

    def progress(self, pct: float | None = None, msg: str | None = None, level: str = "info", phase: str | None = None,
                 detail: dict | None = None) -> None:
        with self._lock:
            urgent = level in ("warning", "error")
            if pct is not None:
                self._fields["progress"] = round(max(0.0, min(100.0, float(pct))), 1)
            if phase and phase != self._fields.get("phase", self.job.phase):
                self._fields["phase"] = phase
                urgent = True
            if msg:
                entry = {"t": timezone.now().isoformat(timespec="milliseconds"), "level": level, "msg": str(msg)[:1000]}
                if detail:
                    entry["detail"] = detail
                self._log.append(entry)
                if len(self._log) > MAX_LOG:
                    self._log = self._log[:50] + [{"t": entry["t"], "level": "debug", "msg": "… older lines trimmed …"}] + self._log[-(MAX_LOG - 60):]
                if level != "debug":
                    self._fields["message"] = str(msg)[:500]
            self._dirty = True
        if urgent or time.time() - self._last_flush > 0.7:
            self.flush()

    __call__ = progress

    def flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            fields = dict(self._fields, log=list(self._log), heartbeat_at=timezone.now())
            self._dirty = False
            self._last_flush = time.time()
        Job.objects.filter(pk=self.job.pk).update(**fields)
        if Job.objects.filter(pk=self.job.pk, cancel_requested=True).exists():
            self.job.cancel_requested = True

    def cancelled(self) -> bool:
        if not self.job.cancel_requested:
            self.job.cancel_requested = Job.objects.filter(pk=self.job.pk, cancel_requested=True).exists()
        return self.job.cancel_requested

    def check(self) -> None:
        if self.cancelled():
            raise Cancelled()


def start(job: Job, fn: Callable[[JobContext], dict | None]) -> Job:
    """Run ``fn(ctx)`` in a daemon thread. Its return value becomes job.result."""

    def runner():
        close_old_connections()
        ctx = JobContext(job)
        Job.objects.filter(pk=job.pk).update(status="running", started_at=timezone.now(), heartbeat_at=timezone.now())
        try:
            result = fn(ctx) or {}
            # keep the job's own last message (its summary) as the headline; log "done" quietly
            ctx.progress(pct=100, msg="Done.", phase="done", level="debug")
            ctx.flush()
            Job.objects.filter(pk=job.pk).update(status="succeeded", result=result, finished_at=timezone.now(), progress=100)
        except Cancelled:
            ctx.progress(msg="Cancelled.", level="warning")
            ctx.flush()
            Job.objects.filter(pk=job.pk).update(status="cancelled", finished_at=timezone.now())
        except Exception as e:  # report every failure on the job
            logger.exception("connect-studio job %s failed", job.pk)
            ctx.progress(msg=str(e) or type(e).__name__, level="error")
            ctx.flush()
            Job.objects.filter(pk=job.pk).update(status="failed", error=f"{e}\n\n{traceback.format_exc()[-4000:]}",
                                                 finished_at=timezone.now())
        finally:
            db_connection.close()

    _Thread(target=runner, name=f"connect-studio-job-{job.pk}", daemon=True).start()
    return job


def recover_interrupted() -> int:
    """Jobs still 'running' from a previous process can never finish — mark them failed."""
    cutoff = timezone.now() - timezone.timedelta(minutes=5)
    return Job.objects.filter(status__in=("running", "queued"), heartbeat_at__lt=cutoff).update(
        status="failed", error="Interrupted — the server restarted while this job was running.", finished_at=timezone.now())
