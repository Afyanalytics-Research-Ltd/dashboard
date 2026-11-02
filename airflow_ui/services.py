"""
High-level Airflow service facade used by the DRF API and server-side code.

The browser UI talks to Airflow through ``proxy.py``; Python callers use
these helpers, which return plain dicts/lists and raise
:class:`~airflow_ui.client.AirflowError` on failure.
"""

import logging
import uuid
from typing import Any

from django.utils import timezone

from .client import AirflowError, encode_segment, get_client

logger = logging.getLogger(__name__)

__all__ = ['AirflowError', 'AirflowService']


class AirflowService:

    @staticmethod
    def get_dags(**filters) -> list[dict]:
        """Every DAG (all pages). ``filters`` are passed to ``GET /dags``."""
        return list(get_client().paginate('/dags', 'dags', **filters))

    @staticmethod
    def get_dag_details(dag_id: str) -> dict:
        return get_client().get_json(f'/dags/{encode_segment(dag_id)}/details')

    @staticmethod
    def get_dag_runs(dag_id: str, limit: int = 25, **filters) -> list[dict]:
        """Most recent ``limit`` runs, newest first. ``dag_id='~'`` = all DAGs."""
        page = get_client().get_json(
            f'/dags/{encode_segment(dag_id)}/dagRuns',
            limit=min(limit, 100), order_by='-run_after', **filters,
        )
        return page.get('dag_runs', [])

    @staticmethod
    def get_dag_run(dag_id: str, run_id: str) -> dict:
        return get_client().get_json(
            f'/dags/{encode_segment(dag_id)}/dagRuns/{encode_segment(run_id)}'
        )

    @staticmethod
    def trigger_dag(
        dag_id: str,
        conf: dict[str, Any] | None = None,
        run_id: str | None = None,
        note: str | None = None,
    ) -> dict:
        """Trigger a run "now" (``logical_date: null``) and return the created run."""
        payload = {
            'dag_run_id': run_id or f'manual__datahub__{uuid.uuid4().hex[:12]}',
            'logical_date': None,
            'conf': conf or {},
            'note': note or f'Triggered from Afya DataHub at {timezone.now():%Y-%m-%d %H:%M:%S}',
        }
        result = get_client().send_json('POST', f'/dags/{encode_segment(dag_id)}/dagRuns', payload)
        logger.info('DAG %s triggered: %s', dag_id, result.get('dag_run_id'))
        return result

    @staticmethod
    def set_paused(dag_id: str, paused: bool) -> dict:
        return get_client().send_json(
            'PATCH', f'/dags/{encode_segment(dag_id)}', {'is_paused': paused},
        )

    @staticmethod
    def get_task_instances(dag_id: str, run_id: str) -> list[dict]:
        page = get_client().get_json(
            f'/dags/{encode_segment(dag_id)}/dagRuns/{encode_segment(run_id)}/taskInstances',
            limit=100,
        )
        return page.get('task_instances', [])

    @staticmethod
    def get_dag_stats() -> list[dict]:
        return list(get_client().paginate('/dagStats', 'dags'))

    @staticmethod
    def health() -> dict:
        return get_client().get_json('/monitor/health')
