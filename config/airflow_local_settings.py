"""Airflow-wide pipeline monitoring: Sentry setup plus cluster policies.

Airflow imports this module in every component (api-server, scheduler,
dag-processor, worker, triggerer) when it starts, so everything here applies
to every DAG and task without any DAG importing it:

* Sentry is initialized once per service from Airflow's ``[sentry]`` settings
  (``AIRFLOW__SENTRY__SENTRY_ON`` and ``SENTRY_DSN`` in docker-compose).
  Airflow 3.1's own Sentry integration only wraps the Airflow 2 task path, so
  it never captures task failures; this module takes its place.
* ``task_policy`` reports every task failure and checks the published row
  count of every terminal task.
* ``dag_policy`` gives each DAG run a one-hour deadline (the SLA).

The callbacks build their events with ``pipeline_monitoring`` (repo root,
mounted at ``PIPELINES_DIR``), which sends ERROR-level events to Sentry.
A failure here must never stop Airflow from starting, so Sentry setup only
logs its problems, and the callbacks import ``pipeline_monitoring`` lazily.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import sys
from datetime import timedelta
from typing import Any, Mapping

from airflow.configuration import conf
from airflow.sdk.definitions.deadline import (
    AsyncCallback,
    DeadlineAlert,
    DeadlineReference,
)

log = logging.getLogger(__name__)

DAG_RUN_DEADLINE = timedelta(hours=1)


def configure_sentry() -> bool:
    """Initialize the Sentry SDK for this Airflow service; return whether it is on."""
    if not conf.getboolean("sentry", "sentry_on", fallback=False):
        return False
    try:
        import sentry_sdk
        from sentry_sdk.integrations.logging import LoggingIntegration
    except ImportError:
        log.warning("[sentry] sentry_on is set but sentry-sdk is not installed")
        return False
    if sentry_sdk.is_initialized():
        return True

    dsn = (
        conf.get("sentry", "sentry_dsn", fallback="")
        or conf.get("sentry", "dsn", fallback="")
        or os.getenv("SENTRY_DSN", "")
    ).strip()
    if not dsn:
        log.warning("[sentry] sentry_on is set but no DSN is configured; alerts are off")
        return False
    try:
        sentry_sdk.init(
            dsn=dsn,
            # None lets the SDK fall back to SENTRY_ENVIRONMENT / SENTRY_RELEASE.
            environment=conf.get("sentry", "environment", fallback="") or None,
            release=conf.get("sentry", "release", fallback="") or None,
            send_default_pii=False,
            # Only pipeline_monitoring sends issues. Ordinary log lines can
            # carry patient rows, so they become neither breadcrumbs nor events.
            integrations=[LoggingIntegration(level=None, event_level=None)],
            auto_enabling_integrations=False,
        )
    except Exception:
        log.exception("Could not initialize Sentry; pipeline alerts are off")
        return False
    return True


def _pipeline_monitoring() -> Any:
    """Import pipeline_monitoring from the repo root mounted at PIPELINES_DIR."""
    pipelines_dir = os.getenv("PIPELINES_DIR")
    if pipelines_dir and pipelines_dir not in sys.path:
        sys.path.insert(0, pipelines_dir)
    import pipeline_monitoring

    return pipeline_monitoring


def _add_callback(task: Any, attribute: str, callback: Any) -> None:
    """Append a callback to a task without replacing ones the DAG set."""
    existing = getattr(task, attribute, None)
    if existing is None:
        setattr(task, attribute, callback)
    elif isinstance(existing, list):
        if callback not in existing:
            setattr(task, attribute, [*existing, callback])
    elif existing is not callback:
        setattr(task, attribute, [existing, callback])


WAREHOUSE_PARAM = "warehouse_backend"
WAREHOUSE_CHOICES = ("default", "snowflake", "databend")


def _add_warehouse_param(dag: Any) -> None:
    """Give every DAG a ``warehouse_backend`` trigger param.

    "default" follows the platform switch (System Settings -> Data
    warehouse, mirrored into the Airflow Variable afya_warehouse_backend);
    "snowflake"/"databend" force that warehouse for this run only.
    """
    params = getattr(dag, "params", None)
    if params is None or WAREHOUSE_PARAM in params:
        return
    try:
        from airflow.sdk import Param
    except ImportError:
        try:  # Airflow 2
            from airflow.models.param import Param
        except ImportError:
            log.warning("no Param class available; %s param not added", WAREHOUSE_PARAM)
            return
    try:
        params[WAREHOUSE_PARAM] = Param(
            "default",
            type="string",
            enum=list(WAREHOUSE_CHOICES),
            title="Warehouse",
            description="Where this run reads/writes warehouse data. 'default' follows the "
                        "dashboard's System Settings switch.",
        )
    except Exception:
        log.exception("could not add %s param to DAG %s", WAREHOUSE_PARAM, getattr(dag, "dag_id", "?"))


def apply_warehouse_param(context: Mapping[str, Any]) -> None:
    """Before a task runs: export the run's warehouse choice.

    afya_backends.toggles reads AFYA_WAREHOUSE_BACKEND first, so this pins
    every connect() in the task - including subprocesses it starts - to the
    run's choice.  "default" clears it so the platform switch applies.
    """
    params = context.get("params") or {}
    choice = str(params.get(WAREHOUSE_PARAM) or "default").strip().lower()
    if choice in ("snowflake", "databend"):
        os.environ["AFYA_WAREHOUSE_BACKEND"] = choice
        log.info("warehouse for this run: %s (DAG param)", choice)
    else:
        os.environ.pop("AFYA_WAREHOUSE_BACKEND", None)


def dag_policy(dag: Any) -> None:
    """Set a one-hour run deadline unless the DAG defines its own deadline;
    add the warehouse_backend trigger param."""
    if dag.deadline is None:
        dag.deadline = DeadlineAlert(
            reference=DeadlineReference.DAGRUN_QUEUED_AT,
            interval=DAG_RUN_DEADLINE,
            callback=AsyncCallback(sentry_deadline_callback),
        )
    _add_warehouse_param(dag)


def task_policy(task: Any) -> None:
    """Report every task failure; check row counts after terminal tasks succeed;
    apply the run's warehouse choice before the task executes."""
    _add_callback(task, "on_failure_callback", pipeline_failure_callback)
    if not getattr(task, "downstream_task_ids", set()):
        _add_callback(task, "on_success_callback", pipeline_row_count_callback)
    _add_callback(task, "on_execute_callback", apply_warehouse_param)


def pipeline_failure_callback(context: Mapping[str, Any]) -> None:
    """Send a failed task, with its traceback, to Sentry."""
    _pipeline_monitoring().airflow_failure_callback(context)


def pipeline_row_count_callback(context: Mapping[str, Any]) -> None:
    """Alert when a task explicitly reports zero output rows in XCom."""
    task_instance = context.get("task_instance") or context.get("ti")
    if task_instance is None:
        raise ValueError("Task success context does not contain a task instance")

    row_count = task_instance.xcom_pull(
        task_ids=task_instance.task_id,
        key="pipeline_row_count",
    )
    if row_count is None:
        return
    if isinstance(row_count, bool) or not isinstance(row_count, int):
        raise TypeError("pipeline_row_count XCom values must be integers")

    monitoring = _pipeline_monitoring()
    result = monitoring.check_row_count(
        f"{task_instance.dag_id}.{task_instance.task_id}.output_rows",
        row_count,
        minimum=1,
    )
    if result.passed:
        return
    monitor = monitoring.PipelineMonitor(
        pipeline=str(task_instance.dag_id),
        run_id=str(task_instance.run_id),
        tags={
            "monitor_type": "data_quality",
            "dag_id": task_instance.dag_id,
            "task_id": task_instance.task_id,
            "quality_check": result.name,
        },
    )
    monitor.quality(
        dataclasses.replace(result, message="output row count is zero"),
        stage=str(task_instance.task_id),
    )


def _report_deadline_miss(dag_id: str, run_id: str | None) -> None:
    """Send a breached-deadline event for one DAG run to Sentry."""
    monitor = _pipeline_monitoring().PipelineMonitor(
        pipeline=dag_id,
        run_id=run_id,
        tags={"monitor_type": "dag_deadline", "dag_id": dag_id},
    )
    monitor.emit(
        "deadline_missed",
        status="breached",
        message="DAG run exceeded its one-hour runtime deadline",
        data={"deadline_seconds": DAG_RUN_DEADLINE.total_seconds()},
    )


async def sentry_deadline_callback(**kwargs: Any) -> None:
    """Report a DAG run that exceeded the default runtime (runs in the triggerer)."""
    context = kwargs.get("context") or {}
    dag_run = context.get("dag_run") if isinstance(context, Mapping) else None
    dag_id = getattr(dag_run, "dag_id", None) or "airflow-unknown"
    run_id = getattr(dag_run, "run_id", None)
    # Sending and flushing block, so keep them off the triggerer's event loop.
    await asyncio.to_thread(
        _report_deadline_miss,
        str(dag_id),
        str(run_id) if run_id is not None else None,
    )


configure_sentry()
