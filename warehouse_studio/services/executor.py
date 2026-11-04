"""Query execution on Snowflake: async submit / poll / fetch / cancel / export.

Studio queries never block a web worker: ``submit`` starts the statement with
``cursor.execute_async`` and returns Snowflake's query id at once; the browser
polls ``status`` (``connection.get_query_status``) and then pages through
``results`` (``cursor.get_results_from_sfqid``).  Results are fetched once and
cached, so paging and exports don't re-run anything.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from django.core.cache import cache

from ..conf import get_config
from . import serialization as ser
from .connection import Context, StudioConnectionError, default_context, get_pool
from .sql_guard import error_position

logger = logging.getLogger(__name__)

QUERY_ID_RE = re.compile(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$')

RUNNING_STATES = {'RUNNING', 'QUEUED', 'RESUMING_WAREHOUSE', 'QUEUED_REPARING_WAREHOUSE', 'QUEUED_REPAIRING_WAREHOUSE',
                  'BLOCKED', 'NO_DATA', 'RESTARTED', 'ABORTING'}
ERROR_STATES = {'FAILED_WITH_ERROR', 'FAILED_WITH_INCIDENT', 'ABORTED', 'DISCONNECTED'}

RESULT_CACHE_SECONDS = 30 * 60


class QueryError(Exception):
    """A Snowflake error with the fields the editor needs to point at the problem."""

    def __init__(self, message: str, *, errno=None, sqlstate=None, query_id=None):
        super().__init__(message)
        self.message = message
        self.errno = errno
        self.sqlstate = sqlstate
        self.query_id = query_id
        self.position = error_position(message)

    def as_dict(self) -> dict:
        return {'message': self.message, 'errno': self.errno, 'sqlstate': self.sqlstate,
                'query_id': self.query_id, 'position': self.position}


def _wrap(exc: Exception, query_id: str | None = None) -> QueryError:
    if isinstance(exc, QueryError):
        return exc
    msg = getattr(exc, 'msg', None) or str(exc) or type(exc).__name__
    return QueryError(str(msg).strip(), errno=getattr(exc, 'errno', None),
                      sqlstate=getattr(exc, 'sqlstate', None),
                      query_id=getattr(exc, 'sfqid', None) or query_id)


def valid_query_id(query_id: str) -> str:
    if not query_id or not QUERY_ID_RE.match(str(query_id)):
        raise QueryError('Not a Snowflake query id.')
    return str(query_id).lower()


def statement_tag(kind: str, user: str = '') -> str:
    prefix = get_config().query_tag_prefix
    return f'{prefix}:{kind}:{user}'[:2000] if user else f'{prefix}:{kind}'


# ----------------------------------------------------------------------------- synchronous

@dataclass
class SyncResult:
    columns: list[dict]
    rows: list[list]
    query_id: str | None
    elapsed_ms: int

    def records(self) -> list[dict]:
        names = [c['name'].lower() for c in self.columns]
        return [dict(zip(names, r)) for r in self.rows]


TRANSIENT_ERRORS = re.compile(r'response ended prematurely|connection (?:reset|aborted|broken)|remote end closed|'
                              r'max retries exceeded|read timed out|temporarily unavailable|network is unreachable',
                              re.IGNORECASE)


def is_transient(exc: Exception) -> bool:
    return bool(TRANSIENT_ERRORS.search(f'{type(exc).__name__} {getattr(exc, "msg", "")} {exc}'))


def run(sql: str, params=None, *, context: Context | None = None, tag: str = '', max_rows: int | None = None,
        timeout: int | None = None, retries: int = 1) -> SyncResult:
    """Run one statement and wait for it.  For internal dashboard queries, not user SQL.

    These are always reads, so a dropped network connection is retried once
    on a fresh session (the failed session is discarded by the pool).
    """
    for attempt in range(retries + 1):
        try:
            return _run_once(sql, params, context=context, tag=tag, max_rows=max_rows, timeout=timeout)
        except QueryError as exc:
            if attempt >= retries or not is_transient(exc):
                raise
            logger.warning('Transient Snowflake error, retrying once: %s', exc.message)
            time.sleep(0.5)
    raise AssertionError('unreachable')  # pragma: no cover


def _run_once(sql: str, params=None, *, context: Context | None, tag: str, max_rows: int | None,
              timeout: int | None) -> SyncResult:
    pool = get_pool()
    context = context or default_context()
    t0 = time.monotonic()
    with pool.connection(context) as conn:
        cur = conn.cursor()
        try:
            try:
                cur.execute(sql, params, timeout=timeout,
                            _statement_params={'QUERY_TAG': tag or statement_tag('internal')})
            except Exception as exc:  # noqa: BLE001
                raise _wrap(exc, getattr(cur, 'sfqid', None)) from exc
            columns = ser.describe_columns(cur.description)
            raw = cur.fetchmany(max_rows) if max_rows else cur.fetchall()
            families = [c['family'] for c in columns]
            rows = [[(ser.plain(v) if families[i] == 'json' else ser.to_json_value(v, families[i]))
                     for i, v in enumerate(r)] for r in raw]
            return SyncResult(columns, rows, getattr(cur, 'sfqid', None), int((time.monotonic() - t0) * 1000))
        finally:
            cur.close()


def records(sql: str, params=None, **kwargs) -> list[dict]:
    return run(sql, params, **kwargs).records()


# ----------------------------------------------------------------------------- async

def submit(sql: str, *, context: Context | None = None, tag: str = '', params=None) -> dict:
    """Start ``sql`` without waiting.  Returns ``{query_id, context}``."""
    pool = get_pool()
    context = context or default_context()
    with pool.connection(context) as conn:
        cur = conn.cursor()
        try:
            cur.execute_async(sql, params, _statement_params={'QUERY_TAG': tag or statement_tag('worksheet')})
            query_id = cur.sfqid
        except StudioConnectionError:
            raise
        except Exception as exc:  # noqa: BLE001 — compile errors surface here
            raise _wrap(exc, getattr(cur, 'sfqid', None)) from exc
        finally:
            cur.close()
    if not query_id:
        raise QueryError('Snowflake did not return a query id.')
    return {'query_id': query_id, 'context': context.as_dict()}


def _status_name(st) -> str:
    return getattr(st, 'name', None) or str(st).split('.')[-1]


def status(query_id: str, *, context: Context | None = None) -> dict:
    """``{query_id, status, running, done, ok, error}`` for an async query."""
    query_id = valid_query_id(query_id)
    pool = get_pool()
    with pool.connection(context or default_context()) as conn:
        try:
            st = conn.get_query_status(query_id)
        except Exception as exc:  # noqa: BLE001
            raise _wrap(exc, query_id) from exc
        name = _status_name(st)
        error = None
        if name in ERROR_STATES:
            try:
                conn.get_query_status_throw_if_error(query_id)
                error = {'message': f'Query {name.lower().replace("_", " ")}.', 'position': None}
            except Exception as exc:  # noqa: BLE001
                error = _wrap(exc, query_id).as_dict()
    running = name in RUNNING_STATES
    return {'query_id': query_id, 'status': name, 'running': running,
            'done': not running, 'ok': name == 'SUCCESS', 'error': error}


def _cache_key(query_id: str) -> str:
    return f'ws:result:{query_id}'


def results(query_id: str, *, offset: int = 0, limit: int = 500, context: Context | None = None,
            max_rows: int | None = None) -> dict:
    """A page of a finished query's result set (fetched once, then served from cache)."""
    query_id = valid_query_id(query_id)
    offset = max(0, int(offset))
    limit = max(1, min(int(limit), 10_000))
    payload = cache.get(_cache_key(query_id))
    if payload is None:
        payload = _fetch_all(query_id, context=context, max_rows=max_rows or get_config().max_result_rows)
        cache.set(_cache_key(query_id), payload, RESULT_CACHE_SECONDS)
    page = payload['rows'][offset:offset + limit]
    return {**{k: v for k, v in payload.items() if k != 'rows'},
            'rows': page, 'offset': offset, 'limit': limit, 'returned': len(page)}


def _fetch_all(query_id: str, *, context: Context | None, max_rows: int) -> dict:
    pool = get_pool()
    t0 = time.monotonic()
    with pool.connection(context or default_context()) as conn:
        cur = conn.cursor()
        try:
            try:
                cur.get_results_from_sfqid(query_id)
                raw = cur.fetchmany(max_rows + 1)
            except Exception as exc:  # noqa: BLE001
                raise _wrap(exc, query_id) from exc
            columns = ser.describe_columns(cur.description)
            rowcount = cur.rowcount
        finally:
            cur.close()
    truncated = len(raw) > max_rows
    raw = raw[:max_rows]
    total = rowcount if isinstance(rowcount, int) and rowcount >= 0 else (None if truncated else len(raw))
    return {
        'query_id': query_id, 'columns': columns, 'rows': ser.rows_to_json(raw, columns),
        'total': total, 'fetched': len(raw), 'truncated': truncated, 'max_rows': max_rows,
        'fetch_ms': int((time.monotonic() - t0) * 1000),
    }


def cancel(query_id: str, *, context: Context | None = None) -> dict:
    query_id = valid_query_id(query_id)
    res = run('SELECT SYSTEM$CANCEL_QUERY(%s)', (query_id,), context=context, tag=statement_tag('cancel'))
    message = res.rows[0][0] if res.rows else ''
    return {'query_id': query_id, 'message': message}


def iter_export(query_id: str, *, context: Context | None = None, batch: int = 5_000, max_rows: int | None = None):
    """Yield ``(columns, rows_batch)`` straight from Snowflake for large exports."""
    query_id = valid_query_id(query_id)
    max_rows = max_rows or get_config().max_export_rows
    pool = get_pool()
    with pool.connection(context or default_context()) as conn:
        cur = conn.cursor()
        try:
            try:
                cur.get_results_from_sfqid(query_id)
            except Exception as exc:  # noqa: BLE001
                raise _wrap(exc, query_id) from exc
            columns = ser.describe_columns(cur.description)
            sent = 0
            while sent < max_rows:
                chunk = cur.fetchmany(min(batch, max_rows - sent))
                if not chunk:
                    break
                sent += len(chunk)
                yield columns, chunk
        finally:
            cur.close()


def forget(query_id: str) -> None:
    cache.delete(_cache_key(query_id))
