"""
Allowlist of Airflow REST API v2 routes the browser proxy may call.

Generated from the official Airflow 3.1 OpenAPI spec (``/openapi.json``).
``/auth/*`` is deliberately excluded: the proxy owns authentication.
"""

import re

ROUTES: dict[str, tuple[str, ...]] = {
    '/assets': ('GET',),
    '/assets/aliases': ('GET',),
    '/assets/aliases/{asset_alias_id}': ('GET',),
    '/assets/events': ('GET', 'POST'),
    '/assets/{asset_id}/materialize': ('POST',),
    '/assets/{asset_id}/queuedEvents': ('GET', 'DELETE'),
    '/assets/{asset_id}': ('GET',),
    '/dags/{dag_id}/assets/queuedEvents': ('GET', 'DELETE'),
    '/dags/{dag_id}/assets/{asset_id}/queuedEvents': ('GET', 'DELETE'),
    '/backfills': ('GET', 'POST'),
    '/backfills/{backfill_id}': ('GET',),
    '/backfills/{backfill_id}/pause': ('PUT',),
    '/backfills/{backfill_id}/unpause': ('PUT',),
    '/backfills/{backfill_id}/cancel': ('PUT',),
    '/backfills/dry_run': ('POST',),
    '/connections/{connection_id}': ('DELETE', 'GET', 'PATCH'),
    '/connections': ('GET', 'POST', 'PATCH'),
    '/connections/test': ('POST',),
    '/connections/defaults': ('POST',),
    '/dags/{dag_id}/dagRuns/{dag_run_id}': ('GET', 'DELETE', 'PATCH'),
    '/dags/{dag_id}/dagRuns/{dag_run_id}/upstreamAssetEvents': ('GET',),
    '/dags/{dag_id}/dagRuns/{dag_run_id}/clear': ('POST',),
    '/dags/{dag_id}/dagRuns': ('GET', 'POST'),
    '/dags/{dag_id}/dagRuns/{dag_run_id}/wait': ('GET',),
    '/dags/{dag_id}/dagRuns/list': ('POST',),
    '/dagSources/{dag_id}': ('GET',),
    '/dagStats': ('GET',),
    '/config': ('GET',),
    '/config/section/{section}/option/{option}': ('GET',),
    '/dagWarnings': ('GET',),
    '/dags': ('GET', 'PATCH'),
    '/dags/{dag_id}': ('GET', 'PATCH', 'DELETE'),
    '/dags/{dag_id}/details': ('GET',),
    '/dags/{dag_id}/favorite': ('POST',),
    '/dags/{dag_id}/unfavorite': ('POST',),
    '/eventLogs/{event_log_id}': ('GET',),
    '/eventLogs': ('GET',),
    '/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/links': ('GET',),
    '/importErrors/{import_error_id}': ('GET',),
    '/importErrors': ('GET',),
    '/jobs': ('GET',),
    '/plugins': ('GET',),
    '/plugins/importErrors': ('GET',),
    '/pools/{pool_name}': ('DELETE', 'GET', 'PATCH'),
    '/pools': ('GET', 'POST', 'PATCH'),
    '/providers': ('GET',),
    '/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/xcomEntries/{xcom_key}': ('GET', 'PATCH'),
    '/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/xcomEntries': ('GET', 'POST'),
    '/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}': ('GET', 'PATCH', 'DELETE'),
    '/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/listMapped': ('GET',),
    '/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/{map_index}/dependencies': ('GET',),
    '/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/dependencies': ('GET',),
    '/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/tries': ('GET',),
    '/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/{map_index}/tries': ('GET',),
    '/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/{map_index}': ('GET', 'PATCH'),
    '/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances': ('GET', 'PATCH'),
    '/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/list': ('POST',),
    '/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/tries/{task_try_number}': ('GET',),
    '/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/{map_index}/tries/{task_try_number}': ('GET',),
    '/dags/{dag_id}/clearTaskInstances': ('POST',),
    '/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/{map_index}/dry_run': ('PATCH',),
    '/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/dry_run': ('PATCH',),
    '/dags/{dag_id}/tasks': ('GET',),
    '/dags/{dag_id}/tasks/{task_id}': ('GET',),
    '/variables/{variable_key}': ('DELETE', 'GET', 'PATCH'),
    '/variables': ('GET', 'POST', 'PATCH'),
    '/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/logs/{try_number}': ('GET',),
    '/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/externalLogUrl/{try_number}': ('GET',),
    '/parseDagFile/{file_token}': ('PUT',),
    '/dagTags': ('GET',),
    '/dags/{dag_id}/dagVersions/{version_number}': ('GET',),
    '/dags/{dag_id}/dagVersions': ('GET',),
    '/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/{map_index}/hitlDetails': ('PATCH', 'GET'),
    '/dags/{dag_id}/dagRuns/{dag_run_id}/hitlDetails': ('GET',),
    '/monitor/health': ('GET',),
    '/version': ('GET',),
}


def _compile(template: str) -> re.Pattern:
    parts = re.split(r'\{[^/}]+\}', template)
    return re.compile('^' + '[^/]+'.join(re.escape(p) for p in parts) + '$')


_COMPILED = [(_compile(t), t, set(m)) for t, m in ROUTES.items()]
# Literal routes must win over templated ones (``/dags/~`` vs ``/dags/{dag_id}``).
_COMPILED.sort(key=lambda r: r[1].count('{'))


def match_route(path: str, method: str) -> tuple[str | None, bool]:
    """
    Return ``(template, method_allowed)`` for an encoded API path such as
    ``/dags/foo/dagRuns``. ``template`` is ``None`` when nothing matches.
    """
    method = method.upper()
    first = None
    for regex, template, methods in _COMPILED:
        if regex.match(path):
            if method in methods:
                return template, True
            first = first or template
    return first, False
