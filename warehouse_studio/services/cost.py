"""Cost Center analytics: credits, spend forecast, heatmaps, attribution, storage.

Every function returns plain JSON-able dicts.  Queries for one page run in
parallel (each on its own pooled session) and results are cached briefly so
flipping between tabs is instant.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

from django.core.cache import cache

from ..conf import credits_per_hour, get_config
from . import executor, sources


def clamp_days(days, default: int = 30, maximum: int = 365) -> int:
    try:
        days = int(days)
    except (TypeError, ValueError):
        return default
    return max(1, min(days, maximum))


GEN_KEY = 'ws:cost:generation'


def _cached(key: str, fn, seconds: int | None = None):
    gen = cache.get(GEN_KEY, 0)
    full = f'ws:cost:{gen}:' + hashlib.sha1(key.encode()).hexdigest()
    hit = cache.get(full)
    if hit is not None:
        return hit
    value = fn()
    cache.set(full, value, seconds if seconds is not None else get_config().cache_seconds)
    return value


def _parallel(jobs: dict) -> dict:
    """Run ``{name: callable}`` concurrently; a failing job yields ``{'error': msg}``."""
    out = {}
    with ThreadPoolExecutor(max_workers=min(len(jobs), max(1, get_config().pool_size - 1))) as pool:
        futures = {name: pool.submit(fn) for name, fn in jobs.items()}
        for name, fut in futures.items():
            try:
                out[name] = fut.result()
            except sources.SourceUnavailable as exc:
                out[name] = {'error': str(exc), 'unavailable': True}
            except Exception as exc:  # noqa: BLE001
                out[name] = {'error': getattr(exc, 'message', None) or str(exc)}
    return out


def _q(sql: str, params=None, kind: str = 'cost') -> list[dict]:
    return executor.records(sql, params, tag=executor.statement_tag(kind))


def _num(v) -> float:
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


# ----------------------------------------------------------------------------- forecasting

def month_forecast(daily: list[dict], today: dt.date | None = None) -> dict:
    """Project this month's credits from month-to-date plus the recent daily run-rate.

    ``daily`` = ``[{'day': 'YYYY-MM-DD', 'credits': float}]``.  The run-rate is
    the mean of the last 7 *complete* days (today is partial), which reacts to
    workload changes faster than a month average without one spike dominating.
    """
    today = today or dt.date.today()
    by_day = {}
    for row in daily:
        d = row['day'] if isinstance(row['day'], dt.date) else dt.date.fromisoformat(str(row['day'])[:10])
        by_day[d] = by_day.get(d, 0.0) + _num(row['credits'])
    month_start = today.replace(day=1)
    next_month = (month_start + dt.timedelta(days=32)).replace(day=1)
    days_in_month = (next_month - month_start).days
    prev_start = (month_start - dt.timedelta(days=1)).replace(day=1)
    mtd = sum(v for d, v in by_day.items() if month_start <= d <= today)
    last_month = sum(v for d, v in by_day.items() if prev_start <= d < month_start)
    complete = [today - dt.timedelta(days=i) for i in range(1, 8)]
    window = [by_day.get(d, 0.0) for d in complete]
    run_rate = sum(window) / len(window) if window else 0.0
    today_so_far = by_day.get(today, 0.0)
    remaining_days = (next_month - today).days - 1          # full days after today
    # Today: assume it finishes at the run-rate unless it already exceeded it.
    projected = mtd - today_so_far + max(today_so_far, run_rate) + run_rate * max(remaining_days, 0)
    # Same point last month, for an apples-to-apples pace comparison.
    same_day_last = prev_start + dt.timedelta(days=min(today.day, (month_start - prev_start).days) - 1)
    last_month_to_date = sum(v for d, v in by_day.items() if prev_start <= d <= same_day_last)
    return {
        'month_to_date': round(mtd, 4), 'last_month': round(last_month, 4),
        'last_month_to_date': round(last_month_to_date, 4), 'run_rate_per_day': round(run_rate, 4),
        'projected_month': round(projected, 4), 'days_in_month': days_in_month, 'day_of_month': today.day,
        'pace_vs_last_month': round((mtd - last_month_to_date) / last_month_to_date, 4) if last_month_to_date else None,
    }


# ----------------------------------------------------------------------------- overview

def overview(days: int = 30) -> dict:
    days = clamp_days(days)
    cfg = get_config()

    def build():
        state = sources.probe()
        md = lambda: sources.relation('metering_daily', state)  # noqa: E731
        jobs = {
            'daily_by_service': lambda: _q(
                f"SELECT usage_date AS day, service_type, SUM(credits_billed) AS credits, SUM(credits_used) AS used "
                f"FROM {md()} WHERE usage_date >= DATEADD('day', -%(d)s, CURRENT_DATE()) GROUP BY 1, 2 ORDER BY 1",
                {'d': days}),
            'month_daily': lambda: _q(
                f"SELECT usage_date AS day, SUM(credits_billed) AS credits FROM {md()} "
                f"WHERE usage_date >= DATE_TRUNC('month', DATEADD('month', -1, CURRENT_DATE())) GROUP BY 1 ORDER BY 1"),
            'daily_by_warehouse': lambda: _q(
                f"SELECT TO_DATE(start_time) AS day, warehouse_name, SUM(credits_used) AS credits, "
                f"SUM(credits_used_compute) AS compute, SUM(credits_used_cloud_services) AS cloud "
                f"FROM {sources.relation('warehouse_metering', state)} "
                f"WHERE start_time >= DATEADD('day', -%(d)s, CURRENT_TIMESTAMP()) GROUP BY 1, 2 ORDER BY 1",
                {'d': days}),
            'storage': lambda: _q(
                f"SELECT usage_date AS day, storage_bytes, stage_bytes, failsafe_bytes "
                f"FROM {sources.relation('storage_daily', state)} "
                f"WHERE usage_date >= DATEADD('day', -%(d)s, CURRENT_DATE()) ORDER BY 1", {'d': days}),
            'queries': lambda: _q(
                f"SELECT TO_DATE(start_time) AS day, COUNT(*) AS queries, "
                f"COUNT_IF(execution_status = 'FAIL') AS failed, "
                f"AVG(total_elapsed_time) AS avg_ms, MEDIAN(total_elapsed_time) AS p50_ms, "
                f"APPROX_PERCENTILE(total_elapsed_time, 0.95) AS p95_ms, "
                f"SUM(bytes_scanned) AS bytes_scanned, SUM(execution_time) / 1000 AS exec_seconds, "
                f"SUM(queued_overload_time) / 1000 AS queued_seconds, "
                f"SUM(bytes_spilled_to_local_storage + bytes_spilled_to_remote_storage) AS spilled_bytes, "
                f"COUNT(DISTINCT user_name) AS users "
                f"FROM {sources.relation('query_history', state)} "
                f"WHERE start_time >= DATEADD('day', -%(d)s, CURRENT_TIMESTAMP()) GROUP BY 1 ORDER BY 1",
                {'d': min(days, 30)}),
        }
        data = _parallel(jobs)
        month = data.get('month_daily')
        forecast = month_forecast(month) if isinstance(month, list) else None
        services = {}
        if isinstance(data.get('daily_by_service'), list):
            for r in data['daily_by_service']:
                services[r['service_type']] = services.get(r['service_type'], 0.0) + _num(r['credits'])
        warehouses = {}
        if isinstance(data.get('daily_by_warehouse'), list):
            for r in data['daily_by_warehouse']:
                warehouses[r['warehouse_name']] = warehouses.get(r['warehouse_name'], 0.0) + _num(r['credits'])
        storage_rows = data.get('storage') if isinstance(data.get('storage'), list) else []
        latest_storage = storage_rows[-1] if storage_rows else None
        total_credits = sum(services.values())
        storage_tb = (_num(latest_storage['storage_bytes']) + _num(latest_storage['stage_bytes'])
                      + _num(latest_storage['failsafe_bytes'])) / 1024 ** 4 if latest_storage else 0.0
        return {
            'days': days, 'credit_price_usd': cfg.credit_price_usd,
            'storage_price_per_tb_month_usd': cfg.storage_price_per_tb_month_usd,
            'total_credits': round(total_credits, 4),
            'total_usd': round(total_credits * cfg.credit_price_usd, 2),
            'services': [{'service_type': k, 'credits': round(v, 4)} for k, v in sorted(services.items(), key=lambda kv: -kv[1])],
            'warehouses': [{'warehouse_name': k, 'credits': round(v, 4)} for k, v in sorted(warehouses.items(), key=lambda kv: -kv[1])],
            'forecast': forecast,
            'storage_tb': round(storage_tb, 6),
            'storage_monthly_usd': round(storage_tb * cfg.storage_price_per_tb_month_usd, 2),
            **data,
        }

    return _cached(f'overview:{days}', build)


def heatmap(days: int = 30) -> dict:
    """Credits by weekday x hour-of-day (UTC) — when does the account spend?"""
    days = clamp_days(days, maximum=90)

    def build():
        rows = _q(
            f"SELECT DAYOFWEEKISO(start_time) AS dow, HOUR(start_time) AS hour, SUM(credits_used) AS credits, "
            f"COUNT(DISTINCT TO_DATE(start_time)) AS days_seen "
            f"FROM {sources.relation('warehouse_metering')} "
            f"WHERE start_time >= DATEADD('day', -%(d)s, CURRENT_TIMESTAMP()) GROUP BY 1, 2", {'d': days})
        return {'days': days, 'cells': rows, 'timezone': 'UTC'}

    return _cached(f'heatmap:{days}', build)


def warehouses(days: int = 30) -> dict:
    """Per-warehouse spend, utilisation, idle burn, load and settings."""
    days = clamp_days(days, maximum=30)
    cfg = get_config()

    def build():
        state = sources.probe()
        metering = sources.relation('warehouse_metering', state)
        qh = sources.relation('query_history', state)

        def usage():
            return _q(
                f"WITH m AS (SELECT warehouse_name, DATE_TRUNC('hour', start_time) AS h, "
                f"SUM(credits_used_compute) AS compute, SUM(credits_used_cloud_services) AS cloud "
                f"FROM {metering} WHERE start_time >= DATEADD('day', -%(d)s, CURRENT_TIMESTAMP()) GROUP BY 1, 2), "
                f"q AS (SELECT warehouse_name, DATE_TRUNC('hour', start_time) AS h, COUNT(*) AS queries, "
                f"SUM(execution_time) / 1000 AS exec_seconds, SUM(queued_overload_time) / 1000 AS queued_seconds, "
                f"SUM(bytes_spilled_to_remote_storage) AS remote_spill, COUNT_IF(execution_status = 'FAIL') AS failed "
                f"FROM {qh} WHERE start_time >= DATEADD('day', -%(d)s, CURRENT_TIMESTAMP()) "
                f"AND warehouse_size IS NOT NULL GROUP BY 1, 2) "
                f"SELECT m.warehouse_name, SUM(m.compute) AS compute_credits, SUM(m.cloud) AS cloud_credits, "
                f"COUNT_IF(m.compute > 0) AS active_hours, COALESCE(SUM(q.queries), 0) AS queries, "
                f"COALESCE(SUM(q.exec_seconds), 0) AS exec_seconds, COALESCE(SUM(q.queued_seconds), 0) AS queued_seconds, "
                f"COALESCE(SUM(q.remote_spill), 0) AS remote_spill_bytes, COALESCE(SUM(q.failed), 0) AS failed, "
                f"SUM(IFF(COALESCE(q.exec_seconds, 0) < 60, m.compute, 0)) AS idle_hour_credits, "
                f"COUNT_IF(COALESCE(q.exec_seconds, 0) < 60 AND m.compute > 0) AS idle_hours "
                f"FROM m LEFT JOIN q ON q.warehouse_name = m.warehouse_name AND q.h = m.h "
                f"GROUP BY 1 ORDER BY 2 DESC", {'d': days})

        def daily():
            return _q(
                f"SELECT warehouse_name, TO_DATE(start_time) AS day, SUM(credits_used) AS credits "
                f"FROM {metering} WHERE start_time >= DATEADD('day', -%(d)s, CURRENT_TIMESTAMP()) "
                f"GROUP BY 1, 2 ORDER BY 2", {'d': days})

        def load():
            return _q(
                f"SELECT warehouse_name, AVG(avg_running) AS avg_running, MAX(avg_running) AS peak_running, "
                f"AVG(avg_queued_load) AS avg_queued, MAX(avg_queued_load) AS peak_queued, "
                f"AVG(avg_queued_provisioning) AS avg_provisioning, AVG(avg_blocked) AS avg_blocked "
                f"FROM {sources.relation('warehouse_load', state)} "
                f"WHERE start_time >= DATEADD('day', -%(d)s, CURRENT_TIMESTAMP()) GROUP BY 1", {'d': min(days, 14)})

        def events():
            return _q(
                f"SELECT warehouse_name, COUNT_IF(event_name = 'RESUME_WAREHOUSE' AND event_state = 'COMPLETED') AS resumes, "
                f"COUNT_IF(event_name = 'SUSPEND_WAREHOUSE' AND event_state = 'COMPLETED') AS suspends, "
                f"COUNT_IF(event_name ILIKE '%%RESIZE%%') AS resizes "
                f"FROM {sources.relation('warehouse_events', state)} "
                f"WHERE timestamp >= DATEADD('day', -%(d)s, CURRENT_TIMESTAMP()) GROUP BY 1", {'d': days})

        data = _parallel({'usage': usage, 'daily': daily, 'load': load, 'events': events,
                          'settings': list_warehouses})
        settings_rows = data['settings'] if isinstance(data.get('settings'), list) else []
        by_name = {r['name']: r for r in settings_rows}
        extra = {k: {r['warehouse_name']: r for r in data[k]} if isinstance(data.get(k), list) else {}
                 for k in ('load', 'events')}
        spark = {}
        if isinstance(data.get('daily'), list):
            for r in data['daily']:
                spark.setdefault(r['warehouse_name'], []).append({'day': r['day'], 'credits': _num(r['credits'])})
        items = []
        names = set(by_name)
        if isinstance(data.get('usage'), list):
            names |= {r['warehouse_name'] for r in data['usage']}
        usage_by = {r['warehouse_name']: r for r in data['usage']} if isinstance(data.get('usage'), list) else {}
        for name in sorted(names, key=lambda n: -_num(usage_by.get(n, {}).get('compute_credits'))):
            u = usage_by.get(name, {})
            s = by_name.get(name, {})
            cph = credits_per_hour(s.get('size'))
            compute = _num(u.get('compute_credits'))
            busy_capacity = compute / cph * 3600 if cph else 0
            items.append({
                'name': name, 'settings': s, 'credits_per_hour': cph,
                'compute_credits': round(compute, 4), 'cloud_credits': round(_num(u.get('cloud_credits')), 4),
                'usd': round((compute + _num(u.get('cloud_credits'))) * cfg.credit_price_usd, 2),
                'active_hours': int(_num(u.get('active_hours'))), 'queries': int(_num(u.get('queries'))),
                'failed': int(_num(u.get('failed'))),
                'exec_seconds': round(_num(u.get('exec_seconds')), 1),
                'queued_seconds': round(_num(u.get('queued_seconds')), 1),
                'remote_spill_bytes': _num(u.get('remote_spill_bytes')),
                'idle_hour_credits': round(_num(u.get('idle_hour_credits')), 4),
                'idle_hours': int(_num(u.get('idle_hours'))),
                # Share of paid compute time spent executing (can exceed 1 with concurrency).
                'utilization': round(_num(u.get('exec_seconds')) / busy_capacity, 4) if busy_capacity else None,
                'load': extra['load'].get(name), 'events': extra['events'].get(name),
                'daily': spark.get(name, []),
            })
        errors = {k: v['error'] for k, v in data.items() if isinstance(v, dict) and 'error' in v}
        return {'days': days, 'credit_price_usd': cfg.credit_price_usd, 'warehouses': items, 'errors': errors}

    return _cached(f'warehouses:{days}', build)


def list_warehouses() -> list[dict]:
    """Warehouse settings + live state.  Toolkit procedure sees every warehouse; SHOW sees the role's."""
    state = sources.probe()
    if state.get('toolkit_procedures', {}).get('SP_WS_WAREHOUSES'):
        rows = executor.records(f'CALL {get_config().toolkit}.SP_WS_WAREHOUSES()', tag=executor.statement_tag('warehouses'))
    else:
        raw = executor.records('SHOW WAREHOUSES', tag=executor.statement_tag('warehouses'))
        rows = [{k: r.get(k) for k in ('name', 'state', 'type', 'size', 'min_cluster_count', 'max_cluster_count',
                                       'started_clusters', 'running', 'queued', 'auto_suspend', 'auto_resume',
                                       'resource_monitor', 'scaling_policy', 'created_on', 'resumed_on', 'owner',
                                       'comment')} for r in raw]
    for r in rows:
        try:
            r['auto_suspend'] = int(r['auto_suspend']) if r.get('auto_suspend') not in (None, '', 'null') else None
        except (TypeError, ValueError):
            r['auto_suspend'] = None
        r['credits_per_hour'] = credits_per_hour(r.get('size'))
    return rows


def resource_monitors() -> list[dict]:
    state = sources.probe()
    if state.get('toolkit_procedures', {}).get('SP_WS_RESOURCE_MONITORS'):
        return executor.records(f'CALL {get_config().toolkit}.SP_WS_RESOURCE_MONITORS()',
                                tag=executor.statement_tag('monitors'))
    raw = executor.records('SHOW RESOURCE MONITORS', tag=executor.statement_tag('monitors'))
    return [{'name': r.get('name'), 'credit_quota': _num(r.get('credit_quota')), 'used_credits': _num(r.get('used_credits')),
             'remaining_credits': _num(r.get('remaining_credits')), 'level': r.get('level'), 'frequency': r.get('frequency'),
             'start_time': r.get('start_time'), 'end_time': r.get('end_time'), 'notify_at': r.get('notify_at'),
             'suspend_at': r.get('suspend_at'), 'suspend_immediately_at': r.get('suspend_immediately_at')} for r in raw]


def attribution(days: int = 30, limit: int = 25) -> dict:
    """Compute credits attributed to users, roles, warehouses, databases and query tags."""
    days = clamp_days(days, maximum=30)
    limit = max(5, min(int(limit), 100))

    def build():
        state = sources.probe()
        rows = _q(
            f"SELECT GROUPING_ID(q.user_name, q.role_name, q.warehouse_name, q.database_name, q.query_tag) AS gid, "
            f"q.user_name, q.role_name, q.warehouse_name, q.database_name, q.query_tag, "
            f"SUM(a.credits_attributed_compute) AS credits, COUNT(*) AS queries, "
            f"SUM(q.execution_time) / 1000 AS exec_seconds "
            f"FROM {sources.relation('query_attribution', state)} a "
            f"JOIN {sources.relation('query_history', state)} q ON q.query_id = a.query_id "
            f"WHERE a.start_time >= DATEADD('day', -%(d)s, CURRENT_TIMESTAMP()) "
            f"GROUP BY GROUPING SETS ((q.user_name), (q.role_name), (q.warehouse_name), (q.database_name), (q.query_tag))",
            {'d': days})
        dims = {'user': 'user_name', 'role': 'role_name', 'warehouse': 'warehouse_name',
                'database': 'database_name', 'query_tag': 'query_tag'}
        out = {}
        for dim, col in dims.items():
            items = [{'name': r.get(col) if r.get(col) not in (None, '') else '(none)',
                      'credits': round(_num(r['credits']), 6), 'queries': int(_num(r['queries'])),
                      'exec_seconds': round(_num(r['exec_seconds']), 1)}
                     for r in rows if _is_dim_row(r, col, dims.values())]
            out[dim] = sorted(items, key=lambda x: -x['credits'])[:limit]
        return {'days': days, 'credit_price_usd': get_config().credit_price_usd, **out}

    return _cached(f'attribution:{days}:{limit}', build)


def _is_dim_row(row: dict, col: str, cols) -> bool:
    """Rows from GROUPING SETS: this dimension's row has every *other* dimension NULL.

    GROUPING_ID tells which columns were aggregated away; the dimension's own
    column may legitimately be NULL (e.g. queries with no database).
    """
    order = ['user_name', 'role_name', 'warehouse_name', 'database_name', 'query_tag']
    gid = int(_num(row.get('gid')))
    pos = order.index(col)
    # bit = 1 means "aggregated" (not part of this grouping set); MSB is the first argument.
    expected = sum(1 << (len(order) - 1 - i) for i in range(len(order)) if i != pos)
    return gid == expected


def top_queries(days: int = 7, by: str = 'credits', limit: int = 50) -> dict:
    """Most expensive query *shapes* (parameterized hash): one fix pays off on every run."""
    days = clamp_days(days, default=7, maximum=30)
    limit = max(5, min(int(limit), 200))
    order = {'credits': 'credits', 'elapsed': 'total_elapsed_ms', 'executions': 'executions',
             'bytes': 'bytes_scanned', 'spill': 'spilled_bytes'}.get(by, 'credits')

    def build():
        state = sources.probe()
        qh = sources.relation('query_history', state)
        try:
            attr = sources.relation('query_attribution', state)
            credits_sql = f"LEFT JOIN {attr} a ON a.query_id = q.query_id"
            credits_col = 'SUM(COALESCE(a.credits_attributed_compute, 0))'
        except sources.SourceUnavailable:
            credits_sql, credits_col = '', '0'
        rows = _q(
            f"SELECT q.query_parameterized_hash AS shape, COUNT(*) AS executions, {credits_col} AS credits, "
            f"SUM(q.total_elapsed_time) AS total_elapsed_ms, AVG(q.total_elapsed_time) AS avg_elapsed_ms, "
            f"APPROX_PERCENTILE(q.total_elapsed_time, 0.95) AS p95_elapsed_ms, SUM(q.bytes_scanned) AS bytes_scanned, "
            f"SUM(q.bytes_spilled_to_local_storage + q.bytes_spilled_to_remote_storage) AS spilled_bytes, "
            f"COUNT_IF(q.execution_status = 'FAIL') AS failed, ANY_VALUE(q.query_type) AS query_type, "
            f"ARRAY_AGG(DISTINCT q.user_name) WITHIN GROUP (ORDER BY q.user_name) AS users, "
            f"ANY_VALUE(q.warehouse_name) AS warehouse_name, MAX(q.start_time) AS last_run, "
            f"MAX_BY(q.query_id, q.total_elapsed_time) AS slowest_query_id, "
            f"ANY_VALUE(LEFT(q.query_text, 1200)) AS sample_text "
            f"FROM {qh} q {credits_sql} "
            f"WHERE q.start_time >= DATEADD('day', -%(d)s, CURRENT_TIMESTAMP()) AND q.warehouse_size IS NOT NULL "
            f"AND q.query_parameterized_hash IS NOT NULL "
            f"GROUP BY 1 ORDER BY {order} DESC NULLS LAST LIMIT %(n)s", {'d': days, 'n': limit})
        price = get_config().credit_price_usd
        for r in rows:
            r['usd'] = round(_num(r['credits']) * price, 4)
            if isinstance(r.get('users'), str):
                try:
                    r['users'] = json.loads(r['users'])
                except ValueError:
                    pass
        return {'days': days, 'by': by, 'credit_price_usd': price, 'queries': rows}

    return _cached(f'top:{days}:{order}:{limit}', build)


def storage(limit: int = 50) -> dict:
    """Account storage trend, per-database split and the heaviest tables."""
    limit = max(5, min(int(limit), 500))
    cfg = get_config()

    def build():
        state = sources.probe()
        data = _parallel({
            'daily': lambda: _q(
                f"SELECT usage_date AS day, storage_bytes, stage_bytes, failsafe_bytes "
                f"FROM {sources.relation('storage_daily', state)} "
                f"WHERE usage_date >= DATEADD('day', -90, CURRENT_DATE()) ORDER BY 1"),
            'databases': lambda: _q(
                f"SELECT database_name, average_database_bytes AS bytes, average_failsafe_bytes AS failsafe_bytes "
                f"FROM {sources.relation('database_storage', state)} "
                f"QUALIFY ROW_NUMBER() OVER (PARTITION BY database_name ORDER BY usage_date DESC) = 1 "
                f"ORDER BY bytes DESC"),
            'tables': lambda: _q(
                f"SELECT path_to_object, database_name, schema_name, table_name, is_transient, active_bytes, "
                f"time_travel_bytes, failsafe_bytes, retained_for_clone_bytes, total_bytes, overhead_multiple "
                f"FROM {sources.relation('table_storage', state)} ORDER BY total_bytes DESC NULLS LAST LIMIT %(n)s",
                {'n': limit}),
        })
        price = cfg.storage_price_per_tb_month_usd
        if isinstance(data.get('tables'), list):
            for t in data['tables']:
                t['est_monthly_usd'] = round(_num(t['total_bytes']) / 1024 ** 4 * price, 4)
                t['overhead_monthly_usd'] = round(
                    (_num(t['time_travel_bytes']) + _num(t['failsafe_bytes'])) / 1024 ** 4 * price, 4)
        return {'storage_price_per_tb_month_usd': price, **data}

    return _cached(f'storage:{limit}', build)


def clear_cache() -> None:
    """Invalidate every cached cost page (the Refresh button) by bumping the key generation."""
    sources.invalidate()
    try:
        cache.incr(GEN_KEY)
    except ValueError:
        cache.set(GEN_KEY, 1, None)
