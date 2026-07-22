"""
Thin wrapper around the Redash HTTP API for provisioning queries.

Mirrors the session/auth pattern already used by
``analytics_app/management/commands/provision_redash_facility.py`` — a
``requests.Session`` authenticated with the admin API key from settings.
"""

import time

import requests
from django.conf import settings

# Redash chart "type" is always CHART; the visual subtype lives in
# options.globalSeriesType. Maps our simple chart-type choices to it.
CHART_SERIES_TYPES = {
    'bar': 'column',
    'line': 'line',
    'pie': 'pie',
    'area': 'area',
    'scatter': 'scatter',
}


class RedashAPIError(Exception):
    """Raised when the Redash API rejects a request or is unreachable."""


def _session() -> requests.Session:
    api_key = settings.REDASH_ADMIN_API_KEY
    if not api_key:
        raise RedashAPIError(
            'REDASH_ADMIN_API_KEY is not set — add it to .env '
            '(Redash admin user -> Profile -> API Key).'
        )
    session = requests.Session()
    session.headers.update({'Authorization': f'Key {api_key}'})
    return session


def list_data_sources() -> list[dict]:
    """Return every Redash data source as dicts with at least 'id'/'name'/'type'."""
    try:
        resp = _session().get(f'{settings.REDASH_BASE_URL}/api/data_sources', timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RedashAPIError(f'Could not reach Redash: {exc}') from exc
    return resp.json()


def create_query(name: str, sql: str, data_source_id: int) -> dict:
    """Create a new query in Redash and return the created query object (includes 'id')."""
    payload = {'name': name, 'query': sql, 'data_source_id': data_source_id}
    try:
        resp = _session().post(f'{settings.REDASH_BASE_URL}/api/queries', json=payload, timeout=30)
    except requests.RequestException as exc:
        raise RedashAPIError(f'Could not reach Redash: {exc}') from exc
    if resp.status_code >= 400:
        raise RedashAPIError(f'Redash rejected the query ({resp.status_code}): {resp.text}')
    return resp.json()


def get_query(query_id: int) -> dict:
    """Return a query object, including its nested 'visualizations' list."""
    try:
        resp = _session().get(f'{settings.REDASH_BASE_URL}/api/queries/{query_id}', timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RedashAPIError(f'Could not reach Redash: {exc}') from exc
    return resp.json()


def create_dashboard(name: str) -> dict:
    """Create a new (draft) dashboard in Redash and return it (includes 'id'/'slug')."""
    try:
        resp = _session().post(f'{settings.REDASH_BASE_URL}/api/dashboards', json={'name': name}, timeout=15)
    except requests.RequestException as exc:
        raise RedashAPIError(f'Could not reach Redash: {exc}') from exc
    if resp.status_code >= 400:
        raise RedashAPIError(f'Redash rejected the dashboard ({resp.status_code}): {resp.text}')
    return resp.json()


def publish_dashboard(dashboard_id: int) -> dict:
    """Mark a dashboard as published (not a draft)."""
    try:
        resp = _session().post(
            f'{settings.REDASH_BASE_URL}/api/dashboards/{dashboard_id}',
            json={'is_draft': False},
            timeout=15,
        )
    except requests.RequestException as exc:
        raise RedashAPIError(f'Could not reach Redash: {exc}') from exc
    if resp.status_code >= 400:
        raise RedashAPIError(f'Redash rejected publishing the dashboard ({resp.status_code}): {resp.text}')
    return resp.json()


def create_widget(dashboard_id: int, visualization_id: int) -> dict:
    """Add a visualization as a widget on a dashboard."""
    payload = {
        'dashboard_id': dashboard_id,
        'visualization_id': visualization_id,
        'options': {},
        'text': '',
        'width': 1,
    }
    try:
        resp = _session().post(f'{settings.REDASH_BASE_URL}/api/widgets', json=payload, timeout=15)
    except requests.RequestException as exc:
        raise RedashAPIError(f'Could not reach Redash: {exc}') from exc
    if resp.status_code >= 400:
        raise RedashAPIError(f'Redash rejected the widget ({resp.status_code}): {resp.text}')
    return resp.json()


def share_dashboard(dashboard_id: int) -> dict:
    """Enable public sharing on a dashboard and return {'public_url', 'api_key'}."""
    try:
        resp = _session().post(f'{settings.REDASH_BASE_URL}/api/dashboards/{dashboard_id}/share', timeout=15)
    except requests.RequestException as exc:
        raise RedashAPIError(f'Could not reach Redash: {exc}') from exc
    if resp.status_code >= 400:
        raise RedashAPIError(f'Redash rejected sharing the dashboard ({resp.status_code}): {resp.text}')
    return resp.json()


def get_query_result(query_result_id: int) -> dict:
    """Return a query_result object (has 'query_result': {'data': {'columns', 'rows'}})."""
    try:
        resp = _session().get(f'{settings.REDASH_BASE_URL}/api/query_results/{query_result_id}', timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RedashAPIError(f'Could not reach Redash: {exc}') from exc
    return resp.json()


def refresh_query(query_id: int) -> dict:
    """Kick off a fresh execution of a query. Returns the job dict (has 'job': {'id': ...})."""
    try:
        resp = _session().post(f'{settings.REDASH_BASE_URL}/api/queries/{query_id}/refresh', timeout=15)
    except requests.RequestException as exc:
        raise RedashAPIError(f'Could not reach Redash: {exc}') from exc
    if resp.status_code >= 400:
        raise RedashAPIError(f'Redash rejected the refresh ({resp.status_code}): {resp.text}')
    return resp.json()


def get_job(job_id: str) -> dict:
    """Return a background job's status dict (has 'job': {'status', 'query_result_id'})."""
    try:
        resp = _session().get(f'{settings.REDASH_BASE_URL}/api/jobs/{job_id}', timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RedashAPIError(f'Could not reach Redash: {exc}') from exc
    return resp.json()


def get_query_columns(query_id: int, timeout_seconds: float = 45.0) -> list[str]:
    """Return the column names available for a query, for building a chart.

    Uses the query's cached result if there is one; otherwise triggers a
    fresh run and polls briefly (bounded by ``timeout_seconds``). Returns an
    empty list if no result becomes available in time — callers should treat
    that as "can't build a chart for this query yet".
    """
    query_obj = get_query(query_id)
    result_id = query_obj.get('latest_query_data_id')

    if result_id is None:
        try:
            job = refresh_query(query_id)
        except RedashAPIError:
            return []
        job_id = job.get('job', job).get('id')
        if not job_id:
            return []
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                job_status = get_job(job_id)
            except RedashAPIError:
                return []
            job_data = job_status.get('job', job_status)
            status = job_data.get('status')
            if status == 3:  # finished
                result_id = job_data.get('query_result_id')
                break
            if status == 4:  # failed
                return []
            time.sleep(1)
        if result_id is None:
            return []

    try:
        result = get_query_result(result_id)
    except RedashAPIError:
        return []
    columns = result.get('query_result', {}).get('data', {}).get('columns', [])
    return [c['name'] for c in columns]


def create_visualization(query_id: int, name: str, series_type: str, x_column: str, y_columns: list[str]) -> dict:
    """Create a new CHART visualization on a query and return it (includes 'id')."""
    column_mapping = {x_column: 'x'}
    column_mapping.update({y: 'y' for y in y_columns})
    options = {
        'globalSeriesType': CHART_SERIES_TYPES.get(series_type, 'column'),
        'series': {'stacking': None},
        'columnMapping': column_mapping,
    }
    payload = {'query_id': query_id, 'type': 'CHART', 'name': name, 'options': options}
    try:
        resp = _session().post(f'{settings.REDASH_BASE_URL}/api/visualizations', json=payload, timeout=15)
    except requests.RequestException as exc:
        raise RedashAPIError(f'Could not reach Redash: {exc}') from exc
    if resp.status_code >= 400:
        raise RedashAPIError(f'Redash rejected the chart ({resp.status_code}): {resp.text}')
    return resp.json()


def publish_query(query_id: int) -> dict:
    """Mark a query as published (not a draft) so it shows up in Redash's query list for users.

    Queries created via the API default to ``is_draft=True``, which hides
    them from the main query browser — this makes the synced/custom queries
    actually discoverable by anyone Redash gives access to the data source.
    """
    try:
        resp = _session().post(
            f'{settings.REDASH_BASE_URL}/api/queries/{query_id}',
            json={'is_draft': False},
            timeout=15,
        )
    except requests.RequestException as exc:
        raise RedashAPIError(f'Could not reach Redash: {exc}') from exc
    if resp.status_code >= 400:
        raise RedashAPIError(f'Redash rejected publishing the query ({resp.status_code}): {resp.text}')
    return resp.json()
