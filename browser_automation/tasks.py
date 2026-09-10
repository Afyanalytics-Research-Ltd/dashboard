"""Celery task backing ScheduledBrowserTask (browser_automation/models.py).

Recurrence is handled entirely by django_celery_beat's DatabaseScheduler —
this is just the payload a PeriodicTask row points at, called with one
argument: the ScheduledBrowserTask's primary key.

Unlike agents/tasks.py's tasks, this one DOES wrap the work in try/except:
the whole point of a user-facing scheduled task is showing "did it run, did
it work" on the ScheduledBrowserTask row, not just relying on Celery's own
(not user-visible) FAILURE state — so failures are recorded to the row
before re-raising, keeping both that visibility AND Celery's normal
failure tracking/alerting.
"""

import asyncio
import json
import logging

from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task(bind=True)
def run_scheduled_browser_task(self, scheduled_task_id: int):
    from django.utils import timezone

    from . import mcp_client
    from .models import ScheduledBrowserTask

    try:
        task_obj = ScheduledBrowserTask.objects.get(pk=scheduled_task_id)
    except ScheduledBrowserTask.DoesNotExist:
        logger.warning(
            "run_scheduled_browser_task[%s]: ScheduledBrowserTask %s no longer exists",
            self.request.id, scheduled_task_id,
        )
        return

    if not task_obj.is_active:
        logger.info(
            "run_scheduled_browser_task[%s]: '%s' is inactive, skipping",
            self.request.id, task_obj.name,
        )
        return

    lines = [line.strip() for line in task_obj.instructions.splitlines() if line.strip()]
    steps = []
    for line in lines:
        tool_name, args = mcp_client.route_message(line)
        steps.append({"tool": tool_name, "args": args})

    logger.info(
        "run_scheduled_browser_task[%s]: running '%s' (%d step(s))",
        self.request.id, task_obj.name, len(steps),
    )

    try:
        results = asyncio.run(mcp_client.run_task(steps))
        status, detail = "success", json.dumps(results, default=str)[:5000]
    except Exception as exc:
        status, detail = "error", str(exc)[:5000]
        task_obj.last_run_at = timezone.now()
        task_obj.last_run_status = status
        task_obj.last_run_detail = detail
        task_obj.save(update_fields=["last_run_at", "last_run_status", "last_run_detail"])
        logger.exception(
            "run_scheduled_browser_task[%s]: '%s' failed", self.request.id, task_obj.name,
        )
        raise

    task_obj.last_run_at = timezone.now()
    task_obj.last_run_status = status
    task_obj.last_run_detail = detail
    task_obj.save(update_fields=["last_run_at", "last_run_status", "last_run_detail"])
    logger.info(
        "run_scheduled_browser_task[%s]: '%s' succeeded", self.request.id, task_obj.name,
    )
    return results
