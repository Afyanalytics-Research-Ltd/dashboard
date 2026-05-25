"""
Airflow REST API service layer.

All external HTTP calls are centralised here so views stay thin and
the service methods can be mocked easily in tests.
"""

import logging
from typing import Any

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

# Module-level token cache — refreshed on 401.
_airflow_token: str | None = None

AIRFLOW_API_BASE = getattr(settings, 'AIRFLOW_BASE_URL', 'http://localhost:8080').rstrip('/') + '/api/v2'
AIRFLOW_AUTH_URL = getattr(settings, 'AIRFLOW_BASE_URL', 'http://localhost:8080').rstrip('/') + '/auth/token'
AIRFLOW_USERNAME = getattr(settings, 'AIRFLOW_USERNAME', 'airflow')
AIRFLOW_PASSWORD = getattr(settings, 'AIRFLOW_PASSWORD', 'airflow')

REQUEST_TIMEOUT = 15  # seconds


def get_airflow_token() -> str:
    """
    Obtain a fresh JWT from the Airflow login endpoint and cache it at
    module level.  Raises ``requests.HTTPError`` on failure.
    """
    global _airflow_token

    response = requests.post(
        AIRFLOW_AUTH_URL,
        json={'username': AIRFLOW_USERNAME, 'password': AIRFLOW_PASSWORD},
        headers={'Content-Type': 'application/json'},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    _airflow_token = response.json().get('access_token', '')
    logger.debug('Airflow JWT refreshed.')
    return _airflow_token


def _airflow_request(method: str, endpoint: str, data: dict | None = None) -> dict:
    """
    Make an authenticated request to the Airflow API.
    Transparently refreshes the token on a 401 response.
    """
    global _airflow_token

    if _airflow_token is None:
        get_airflow_token()

    url = f'{AIRFLOW_API_BASE}{endpoint}'
    headers = {'Authorization': f'Bearer {_airflow_token}'}

    try:
        response = requests.request(
            method, url, json=data, headers=headers, timeout=REQUEST_TIMEOUT
        )

        if response.status_code == 401:
            logger.info('Airflow token expired, refreshing…')
            get_airflow_token()
            headers = {'Authorization': f'Bearer {_airflow_token}'}
            response = requests.request(
                method, url, json=data, headers=headers, timeout=REQUEST_TIMEOUT
            )

        logger.debug('Airflow API %s %s → %s', method, url, response.status_code)

        try:
            return response.json()
        except Exception:
            return {'error': response.text, 'status_code': response.status_code}

    except requests.RequestException as exc:
        logger.error('Airflow API request failed: %s', exc)
        return {'error': str(exc)}


class AirflowService:
    """
    High-level service methods wrapping the Airflow REST API v2.
    Each method returns plain Python dicts / lists so callers are
    decoupled from requests internals.
    """

    @staticmethod
    def get_dags() -> list[dict]:
        """Return the full list of DAGs from Airflow."""
        result = _airflow_request('GET', '/dags')
        dags = result.get('dags', [])
        if not isinstance(dags, list):
            logger.warning('Unexpected dags payload: %s', result)
            return []
        return dags

    @staticmethod
    def get_dag_runs(dag_id: str, limit: int = 25) -> list[dict]:
        """Return the most recent ``limit`` runs for the given DAG."""
        result = _airflow_request('GET', f'/dags/{dag_id}/dagRuns?limit={limit}&order_by=-execution_date')
        runs = result.get('dag_runs', [])
        if not isinstance(runs, list):
            logger.warning('Unexpected dag_runs payload for %s: %s', dag_id, result)
            return []
        return runs

    @staticmethod
    def trigger_dag(dag_id: str) -> dict:
        """
        Trigger a manual DAG run.
        Returns the raw Airflow response dict (contains dag_run_id, state, etc.)
        """
        import uuid
        from django.utils import timezone

        now = timezone.now()
        payload = {
            'dag_run_id': f'manual__{uuid.uuid4()}',
            'logical_date': now.isoformat(),
            'conf': {},
            'note': f'Triggered from Afya DataHub at {now.strftime("%Y-%m-%d %H:%M:%S")}',
        }
        result = _airflow_request('POST', f'/dags/{dag_id}/dagRuns', data=payload)
        logger.info('DAG %s triggered. Response: %s', dag_id, result)
        return result

    @staticmethod
    def get_task_instances(dag_id: str, run_id: str) -> list[dict]:
        """Return all task instances for a specific DAG run."""
        result = _airflow_request(
            'GET',
            f'/dags/{dag_id}/dagRuns/{run_id}/taskInstances',
        )
        tasks = result.get('task_instances', [])
        if not isinstance(tasks, list):
            logger.warning('Unexpected task_instances payload: %s', result)
            return []
        return tasks

    @staticmethod
    def get_dag_details(dag_id: str) -> dict:
        """Return metadata for a single DAG."""
        return _airflow_request('GET', f'/dags/{dag_id}')
