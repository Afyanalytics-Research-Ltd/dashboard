"""Where Cost Center / Optimizer data comes from.

Preferred: the toolkit views installed by ``snowflake/warehouse_studio/install.sql``
(owner's-rights views over ``SNOWFLAKE.ACCOUNT_USAGE`` that the app role may
read).  Fallback: ``SNOWFLAKE.ACCOUNT_USAGE`` directly, when the app role has
been granted IMPORTED PRIVILEGES on the SNOWFLAKE database.  Otherwise the
source is unavailable and the UI explains how to install the toolkit.
"""

from __future__ import annotations

import logging

from django.core.cache import cache

from ..conf import get_config
from . import executor, installer

logger = logging.getLogger(__name__)

AU = 'SNOWFLAKE.ACCOUNT_USAGE'

# logical name -> (toolkit relation, direct ACCOUNT_USAGE equivalent)
SOURCES: dict[str, tuple[str, str]] = {
    'query_history': ('WS_QUERY_HISTORY',
                      f"(SELECT * FROM {AU}.QUERY_HISTORY WHERE start_time >= DATEADD('day', -30, CURRENT_TIMESTAMP()))"),
    'query_attribution': ('WS_QUERY_ATTRIBUTION',
                          f"(SELECT * FROM {AU}.QUERY_ATTRIBUTION_HISTORY WHERE start_time >= DATEADD('day', -30, CURRENT_TIMESTAMP()))"),
    'warehouse_metering': ('WS_WAREHOUSE_METERING',
                           f"(SELECT * FROM {AU}.WAREHOUSE_METERING_HISTORY WHERE start_time >= DATEADD('day', -365, CURRENT_TIMESTAMP()))"),
    'metering_daily': ('WS_METERING_DAILY',
                       f"(SELECT * FROM {AU}.METERING_DAILY_HISTORY WHERE usage_date >= DATEADD('day', -365, CURRENT_DATE()))"),
    'warehouse_load': ('WS_WAREHOUSE_LOAD',
                       f"(SELECT * FROM {AU}.WAREHOUSE_LOAD_HISTORY WHERE start_time >= DATEADD('day', -30, CURRENT_TIMESTAMP()))"),
    'warehouse_events': ('WS_WAREHOUSE_EVENTS',
                         f"(SELECT * FROM {AU}.WAREHOUSE_EVENTS_HISTORY WHERE timestamp >= DATEADD('day', -30, CURRENT_TIMESTAMP()))"),
    'storage_daily': ('WS_STORAGE_DAILY',
                      f"(SELECT * FROM {AU}.STORAGE_USAGE WHERE usage_date >= DATEADD('day', -365, CURRENT_DATE()))"),
    'database_storage': ('WS_DATABASE_STORAGE',
                         f"(SELECT * FROM {AU}.DATABASE_STORAGE_USAGE_HISTORY WHERE usage_date >= DATEADD('day', -90, CURRENT_DATE()))"),
    'table_storage': ('WS_TABLE_STORAGE',
                      f"(SELECT id AS table_id, table_catalog || '.' || table_schema || '.' || table_name AS path_to_object, "
                      f"table_catalog AS database_name, table_schema AS schema_name, table_name, is_transient, table_created, "
                      f"active_bytes, time_travel_bytes, failsafe_bytes, retained_for_clone_bytes, "
                      f"active_bytes + time_travel_bytes + failsafe_bytes + retained_for_clone_bytes AS total_bytes, "
                      f"ROUND((time_travel_bytes + failsafe_bytes) / NULLIF(active_bytes, 0), 2) AS overhead_multiple "
                      f"FROM {AU}.TABLE_STORAGE_METRICS WHERE deleted = FALSE AND table_dropped IS NULL "
                      f"AND schema_dropped IS NULL AND catalog_dropped IS NULL)"),
    'task_history': ('WS_TASK_HISTORY',
                     f"(SELECT * FROM {AU}.TASK_HISTORY WHERE scheduled_time >= DATEADD('day', -30, CURRENT_TIMESTAMP()))"),
    'clustering_history': ('WS_CLUSTERING_HISTORY',
                           f"(SELECT * FROM {AU}.AUTOMATIC_CLUSTERING_HISTORY WHERE start_time >= DATEADD('day', -90, CURRENT_TIMESTAMP()))"),
}

PROBE_CACHE_KEY = 'ws:sources:probe'


class SourceUnavailable(Exception):
    """Neither the toolkit view nor ACCOUNT_USAGE is readable by the app role."""

    def __init__(self, source: str):
        cfg = get_config()
        super().__init__(
            f'{source.replace("_", " ").capitalize()} data is not available to the Studio\'s Snowflake role. '
            f'Install the cost & performance toolkit into {cfg.toolkit} '
            f'(python manage.py warehouse_studio_install) or grant the role IMPORTED PRIVILEGES on SNOWFLAKE.')
        self.source = source


def probe(force: bool = False) -> dict:
    """Which data paths exist.  Cached for five minutes."""
    if not force:
        cached = cache.get(PROBE_CACHE_KEY)
        if cached is not None:
            return cached
    cfg = get_config()
    result = {'toolkit': cfg.toolkit, 'toolkit_installed': False, 'toolkit_missing': [],
              'toolkit_procedures': {}, 'toolkit_relations': {}, 'account_usage': False, 'error': None}
    try:
        rows = executor.records(
            f'SELECT procedure_name AS name, \'PROCEDURE\' AS kind FROM "{cfg.toolkit_database}".INFORMATION_SCHEMA.PROCEDURES '
            f'WHERE procedure_schema = %(s)s UNION ALL '
            f'SELECT table_name, table_type FROM "{cfg.toolkit_database}".INFORMATION_SCHEMA.TABLES WHERE table_schema = %(s)s',
            {'s': cfg.toolkit_schema}, tag=executor.statement_tag('probe'))
        names = {r['name'] for r in rows}
        result['toolkit_procedures'] = {p: p in names for p in installer.EXPECTED_PROCEDURES}
        result['toolkit_relations'] = {r: r in names for r in installer.EXPECTED_RELATIONS}
        result['toolkit_missing'] = sorted([k for k, v in {**result['toolkit_procedures'], **result['toolkit_relations']}.items() if not v])
        result['toolkit_installed'] = not result['toolkit_missing']
    except Exception as exc:  # noqa: BLE001 — the probe must never take the page down
        result['error'] = str(getattr(exc, 'message', exc))[:300]
    try:
        executor.run(f'SELECT 1 FROM {AU}.WAREHOUSE_METERING_HISTORY LIMIT 1', tag=executor.statement_tag('probe'))
        result['account_usage'] = True
    except Exception:  # noqa: BLE001
        result['account_usage'] = False
    cache.set(PROBE_CACHE_KEY, result, 300)
    return result


def relation(source: str, state: dict | None = None) -> str:
    """SQL relation (table name or parenthesised subquery) for a logical source."""
    if source not in SOURCES:
        raise KeyError(source)
    state = state or probe()
    view, direct = SOURCES[source]
    if state.get('toolkit_relations', {}).get(view):
        return f'{get_config().toolkit}.{view}'
    if state.get('account_usage'):
        return direct
    raise SourceUnavailable(source)


def toolkit_object(name: str, state: dict | None = None) -> str:
    """Fully-qualified toolkit object, or :class:`SourceUnavailable` if not installed."""
    state = state or probe()
    known = {**state.get('toolkit_procedures', {}), **state.get('toolkit_relations', {})}
    if not known.get(name):
        raise SourceUnavailable(name.lower())
    return f'{get_config().toolkit}.{name}'


def invalidate() -> None:
    cache.delete(PROBE_CACHE_KEY)
