"""Object browser: databases → schemas → objects → columns, DDL and procedures.

Identifiers coming from the browser are validated and double-quoted; values
go through bind parameters.  Listings are cached for a minute so expanding
the tree feels instant.
"""

from __future__ import annotations

import hashlib
import re

from django.core.cache import cache

from . import executor
from .connection import Context

KINDS = ('TABLE', 'VIEW', 'MATERIALIZED_VIEW', 'DYNAMIC_TABLE', 'EXTERNAL_TABLE', 'EVENT_TABLE', 'PROCEDURE', 'FUNCTION',
         'STAGE', 'STREAM', 'TASK', 'PIPE', 'SEQUENCE', 'FILE_FORMAT')
DDL_TYPES = {'TABLE', 'VIEW', 'PROCEDURE', 'FUNCTION', 'STAGE', 'STREAM', 'TASK', 'PIPE', 'SEQUENCE', 'FILE_FORMAT',
             'SCHEMA', 'DATABASE', 'DYNAMIC_TABLE', 'MATERIALIZED_VIEW', 'EXTERNAL_TABLE', 'EVENT_TABLE'}


class CatalogError(ValueError):
    pass


def quote(name: str) -> str:
    name = str(name or '')
    if not name or len(name) > 255 or any(ord(c) < 32 for c in name):
        raise CatalogError(f'Invalid identifier: {name!r}')
    return '"' + name.replace('"', '""') + '"'


def _cached(key: str, fn, seconds: int = 60):
    full = 'ws:cat:' + hashlib.sha1(key.encode()).hexdigest()
    hit = cache.get(full)
    if hit is not None:
        return hit
    value = fn()
    cache.set(full, value, seconds)
    return value


def _q(sql, params=None, context: Context | None = None):
    return executor.records(sql, params, context=context, tag=executor.statement_tag('catalog'))


def databases(context: Context | None = None, refresh: bool = False) -> list[dict]:
    key = f'dbs:{context.role if context else ""}'
    fn = lambda: [{'name': r['name'], 'kind': r.get('kind') or 'STANDARD', 'owner': r.get('owner'),  # noqa: E731
                   'comment': r.get('comment'), 'created_on': r.get('created_on'), 'origin': r.get('origin')}
                  for r in _q('SHOW DATABASES', context=context)]
    if refresh:
        cache.delete('ws:cat:' + hashlib.sha1(key.encode()).hexdigest())
    return _cached(key, fn)


def schemas(database: str, context: Context | None = None) -> list[dict]:
    db = quote(database)
    return _cached(f'schemas:{database}:{context.role if context else ""}', lambda: [
        {'name': r['name'], 'owner': r.get('owner'), 'comment': r.get('comment'), 'created_on': r.get('created_on')}
        for r in _q(f'SHOW SCHEMAS IN DATABASE {db}', context=context) if r['name'] != 'INFORMATION_SCHEMA'])


def objects(database: str, schema: str, context: Context | None = None) -> dict:
    """Tables/views (with rows & bytes), procedures, functions, stages, tasks, streams, pipes."""
    db, sc = quote(database), quote(schema)

    def fetch_relations():
        return _q(
            f'SELECT table_name AS name, table_type, row_count, bytes, last_altered, comment, clustering_key, '
            f'is_transient, retention_time FROM {db}.INFORMATION_SCHEMA.TABLES WHERE table_schema = %s ORDER BY 1',
            (schema,), context)

    def fetch_routines(view, cols, col):
        try:
            return [{**r, 'params': parse_signature(r['argument_signature'])}
                    for r in _q(f'SELECT {cols} FROM {db}.INFORMATION_SCHEMA.{view} WHERE {col} = %s ORDER BY 1',
                                (schema,), context)]
        except executor.QueryError:
            return []

    def fetch_show(show):
        try:
            return _q(f'SHOW {show} IN SCHEMA {db}.{sc}', context=context)
        except executor.QueryError:
            return []

    def build():
        # Seven independent metadata queries: run them side by side (each on its
        # own pooled session) — serially they take ~10-15 s on a cold schema.
        from concurrent.futures import ThreadPoolExecutor

        jobs = {
            'relations': fetch_relations,
            'PROCEDURE': lambda: fetch_routines('PROCEDURES', 'procedure_name AS name, argument_signature, data_type, '
                                                'procedure_language AS language, comment', 'procedure_schema'),
            'FUNCTION': lambda: fetch_routines('FUNCTIONS', 'function_name AS name, argument_signature, data_type, '
                                               'function_language AS language, comment', 'function_schema'),
            **{kind: (lambda s=show: fetch_show(s)) for kind, show in
               (('STAGE', 'STAGES'), ('TASK', 'TASKS'), ('STREAM', 'STREAMS'), ('PIPE', 'PIPES'))},
        }
        from ..conf import get_config
        with ThreadPoolExecutor(max_workers=max(1, min(len(jobs), get_config().pool_size - 1))) as pool:
            futures = {k: pool.submit(fn) for k, fn in jobs.items()}
            results = {k: f.result() for k, f in futures.items()}   # relations errors propagate, as before

        out: dict[str, list] = {}
        for r in results['relations']:
            t = (r['table_type'] or '').upper()
            kind = {'BASE TABLE': 'TABLE', 'VIEW': 'VIEW', 'MATERIALIZED VIEW': 'MATERIALIZED_VIEW',
                    'EXTERNAL TABLE': 'EXTERNAL_TABLE', 'EVENT TABLE': 'EVENT_TABLE',
                    'DYNAMIC TABLE': 'DYNAMIC_TABLE'}.get(t, t.replace(' ', '_') or 'TABLE')
            out.setdefault(kind, []).append({'name': r['name'], 'rows': r['row_count'], 'bytes': r['bytes'],
                                             'last_altered': r['last_altered'], 'comment': r['comment'],
                                             'clustering_key': r['clustering_key'], 'transient': r['is_transient'] == 'YES',
                                             'retention_days': r['retention_time']})
        out['PROCEDURE'] = results['PROCEDURE']
        out['FUNCTION'] = results['FUNCTION']
        for kind in ('STAGE', 'TASK', 'STREAM', 'PIPE'):
            rows = results[kind]
            if rows:
                out[kind] = [{'name': r['name'], 'comment': r.get('comment'), 'state': r.get('state'),
                              'schedule': r.get('schedule'), 'type': r.get('type'), 'url': r.get('url'),
                              'warehouse': r.get('warehouse'), 'stale': r.get('stale')} for r in rows]
        return out

    return _cached(f'objects:{database}:{schema}:{context.role if context else ""}', build)


def columns(database: str, schema: str, table: str | None = None, context: Context | None = None) -> list[dict]:
    """Columns of one table, or of every table in a schema (for autocomplete)."""
    db = quote(database)
    params: tuple = (schema,)
    where = 'table_schema = %s'
    if table:
        where += ' AND table_name = %s'
        params = (schema, table)
    sql = (f'SELECT table_name, column_name, data_type, is_nullable, column_default, comment, ordinal_position, '
           f'character_maximum_length, numeric_precision, numeric_scale FROM {db}.INFORMATION_SCHEMA.COLUMNS '
           f'WHERE {where} ORDER BY table_name, ordinal_position')
    return _cached(f'cols:{database}:{schema}:{table}:{context.role if context else ""}',
                   lambda: _q(sql + (' LIMIT 20000' if not table else ''), params, context))


def ddl(object_type: str, name: str, context: Context | None = None) -> str:
    t = str(object_type or '').upper().replace(' ', '_')
    if t not in DDL_TYPES:
        raise CatalogError(f'Cannot show DDL for {object_type!r}.')
    if not re.fullmatch(r'[^\x00-\x1f]{1,800}', name or ''):
        raise CatalogError('Invalid object name.')
    rows = executor.run('SELECT GET_DDL(%s, %s, TRUE)', (t, name),
                        context=context, tag=executor.statement_tag('catalog')).rows
    return rows[0][0] if rows else ''


_SIG_RE = re.compile(r'^\((.*)\)$', re.S)


def parse_signature(signature: str | None) -> list[dict]:
    """``(A VARCHAR, B NUMBER DEFAULT 5)`` → ``[{name, type, has_default}]``."""
    if not signature:
        return []
    m = _SIG_RE.match(signature.strip())
    body = m.group(1) if m else signature
    parts, depth, cur = [], 0, ''
    for ch in body:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(cur)
            cur = ''
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    out = []
    for p in parts:
        tokens = p.strip().split()
        if not tokens:
            continue
        has_default = tokens[0].upper() == 'DEFAULT'
        if has_default:
            tokens = tokens[1:]
        if len(tokens) == 1:
            out.append({'name': f'ARG{len(out) + 1}', 'type': tokens[0].upper(), 'has_default': has_default})
        else:
            out.append({'name': tokens[0].strip('"'), 'type': ' '.join(tokens[1:]).upper(), 'has_default': has_default})
    return out


def bind_value(type_name: str, raw):
    """Python value + SQL placeholder for one procedure argument."""
    t = (type_name or 'VARCHAR').upper()
    if raw is None or raw == '':
        return None, '%s'
    if t.startswith(('NUMBER', 'INT', 'INTEGER', 'BIGINT', 'SMALLINT', 'DECIMAL', 'NUMERIC')):
        text = str(raw).strip()
        if not re.fullmatch(r'-?\d+(\.\d+)?', text):
            raise CatalogError(f'{raw!r} is not a number.')
        return (int(text) if '.' not in text else float(text)), '%s'
    if t.startswith(('FLOAT', 'DOUBLE', 'REAL')):
        try:
            return float(raw), '%s'
        except (TypeError, ValueError) as exc:
            raise CatalogError(f'{raw!r} is not a number.') from exc
    if t.startswith('BOOLEAN'):
        return str(raw).strip().lower() in ('1', 'true', 'yes', 'on'), '%s'
    if t.startswith(('ARRAY', 'OBJECT', 'VARIANT')):
        import json
        text = raw if isinstance(raw, str) else json.dumps(raw)
        try:
            json.loads(text)
        except ValueError as exc:
            raise CatalogError(f'{type_name} arguments must be valid JSON.') from exc
        return text, f'PARSE_JSON(%s)::{t.split("(")[0]}'
    return str(raw), '%s'


def build_procedure_call(database: str, schema: str, name: str, params: list[dict], values: list) -> tuple[str, tuple]:
    fq = '.'.join(quote(p) for p in (database, schema, name))
    binds, holders = [], []
    for i, p in enumerate(params):
        raw = values[i] if i < len(values) else None
        if (raw is None or raw == '') and p.get('has_default'):
            break  # trailing defaults may be omitted
        value, holder = bind_value(p['type'], raw)
        binds.append(value)
        holders.append(holder)
    return f'CALL {fq}({", ".join(holders)})', tuple(binds)
