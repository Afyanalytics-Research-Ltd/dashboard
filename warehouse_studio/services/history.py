"""Query history with filters, for the History page.

Two sources, chosen by ``scope``:
* ``live`` — ``INFORMATION_SCHEMA.QUERY_HISTORY`` (real time, last 7 days,
  queries the Studio's role can see).
* ``account`` — the toolkit's 30-day ``WS_QUERY_HISTORY`` view (45 min – 3 h
  latency, every user), joined to per-query attributed credits.
"""

from __future__ import annotations

import re

from ..conf import CREDITS_PER_HOUR, get_config
from . import executor, sources

COLUMNS = ('query_id, query_text, query_type, user_name, role_name, warehouse_name, warehouse_size, database_name, '
           'schema_name, execution_status, error_code, error_message, start_time, end_time, total_elapsed_time, '
           'compilation_time, execution_time, queued_overload_time, queued_provisioning_time, bytes_scanned, '
           'rows_produced, partitions_scanned, partitions_total, bytes_spilled_to_local_storage, '
           'bytes_spilled_to_remote_storage, query_tag')

# Columns ACCOUNT_USAGE has but the INFORMATION_SCHEMA table function does not.
NOT_IN_LIVE = ('partitions_scanned', 'partitions_total', 'bytes_spilled_to_local_storage',
               'bytes_spilled_to_remote_storage', 'bytes_sent_over_the_network', 'percentage_scanned_from_cache',
               'bytes_written')


def live_columns(columns: str) -> str:
    """Rewrite a column list for INFORMATION_SCHEMA.QUERY_HISTORY*, NULL-filling the missing ones."""
    out = []
    for col in (c.strip() for c in columns.split(',')):
        out.append(f'NULL AS {col}' if col in NOT_IN_LIVE else col)
    return ', '.join(out)


STATUSES = {'SUCCESS': "execution_status = 'SUCCESS'",
            'FAILED': "execution_status IN ('FAIL', 'FAILED_WITH_ERROR', 'FAILED_WITH_INCIDENT')",
            'RUNNING': "execution_status IN ('RUNNING', 'QUEUED', 'RESUMING_WAREHOUSE', 'BLOCKED')"}

SORTS = {'start': 'start_time DESC', 'duration': 'total_elapsed_time DESC NULLS LAST', 'bytes': 'bytes_scanned DESC NULLS LAST',
         'spill': '(COALESCE(bytes_spilled_to_local_storage, 0) + COALESCE(bytes_spilled_to_remote_storage, 0)) DESC',
         'queued': 'queued_overload_time DESC NULLS LAST', 'compile': 'compilation_time DESC NULLS LAST'}


def _cph_case() -> str:
    seen, whens = set(), []
    for label, cph in CREDITS_PER_HOUR.items():
        if label not in seen:
            seen.add(label)
            whens.append(f"WHEN '{label}' THEN {cph}")
    return f"(CASE UPPER(REPLACE(warehouse_size, ' ', '')) {' '.join(whens)} ELSE 0 END)"


def _where(f: dict, binds: dict, *, hours_col: str = 'start_time') -> list[str]:
    where = []
    hours = f.get('hours')
    try:
        hours = max(1, min(int(hours), 24 * 30)) if hours not in (None, '') else 24
    except (TypeError, ValueError):
        hours = 24
    binds['hours'] = hours
    where.append(f"{hours_col} >= DATEADD('hour', -%(hours)s, CURRENT_TIMESTAMP())")
    st = str(f.get('status') or '').upper()
    if st in STATUSES:
        where.append(STATUSES[st])
    for key, col in (('user', 'user_name'), ('warehouse', 'warehouse_name'), ('role', 'role_name'),
                     ('database', 'database_name'), ('type', 'query_type')):
        v = str(f.get(key) or '').strip()
        if v:
            binds[key] = v.upper()
            where.append(f'UPPER({col}) = %({key})s')
    q = str(f.get('q') or '').strip()
    if q:
        if re.fullmatch(r'[0-9a-fA-F-]{36}', q):
            binds['qid'] = q.lower()
            where.append('query_id = %(qid)s')
        else:
            binds['q'] = f'%{q[:300]}%'
            where.append('query_text ILIKE %(q)s')
    try:
        min_ms = int(f.get('min_ms') or 0)
    except (TypeError, ValueError):
        min_ms = 0
    if min_ms > 0:
        binds['min_ms'] = min_ms
        where.append('total_elapsed_time >= %(min_ms)s')
    if str(f.get('spilled') or '') in ('1', 'true'):
        where.append('(COALESCE(bytes_spilled_to_local_storage, 0) + COALESCE(bytes_spilled_to_remote_storage, 0)) > 0')
    if str(f.get('hide_system') or '1') in ('1', 'true'):
        where.append("query_type NOT IN ('SHOW', 'DESCRIBE', 'USE', 'ALTER_SESSION', 'BEGIN_TRANSACTION', 'COMMIT', 'ROLLBACK')")
    return where


def search(filters: dict) -> dict:
    scope = 'account' if filters.get('scope') == 'account' else 'live'
    try:
        limit = max(1, min(int(filters.get('limit') or 200), 1000))
    except (TypeError, ValueError):
        limit = 200
    order = SORTS.get(str(filters.get('sort') or 'start'), SORTS['start'])
    binds: dict = {'limit': limit}
    cfg = get_config()
    if scope == 'live':
        db = cfg.database or cfg.toolkit_database
        hours = filters.get('hours') or 24
        try:
            hours = max(1, min(int(hours), 24 * 7))
        except (TypeError, ValueError):
            hours = 24
        f = {**filters, 'hours': hours}
        where = _where(f, binds)
        rel = (f'TABLE("{db}".INFORMATION_SCHEMA.QUERY_HISTORY('
               f"END_TIME_RANGE_START => DATEADD('hour', -%(hours)s, CURRENT_TIMESTAMP()), RESULT_LIMIT => 10000))")
        if any(col in order for col in NOT_IN_LIVE):
            order = SORTS['start']
        where = [w for w in where if not any(col in w for col in NOT_IN_LIVE)]
        sql = (f'SELECT * FROM (SELECT {live_columns(COLUMNS)}, total_elapsed_time AS elapsed_ms, NULL AS credits, '
               f'execution_time / 3600000 * {_cph_case()} AS est_credits FROM {rel}) '
               f"WHERE {' AND '.join(where)} ORDER BY {order} LIMIT %(limit)s")
    else:
        state = sources.probe()
        qh = sources.relation('query_history', state)
        try:
            attr = sources.relation('query_attribution', state)
            join = f'LEFT JOIN (SELECT query_id AS a_qid, SUM(credits_attributed_compute) AS credits FROM {attr} GROUP BY 1) a ON a.a_qid = q.query_id'
            credits = 'a.credits'
        except sources.SourceUnavailable:
            join, credits = '', 'NULL'
        where = _where(filters, binds)
        sql = (f'SELECT {COLUMNS}, total_elapsed_time AS elapsed_ms, {credits} AS credits, '
               f'execution_time / 3600000 * {_cph_case()} AS est_credits FROM {qh} q {join} '
               f"WHERE {' AND '.join(where)} ORDER BY {order} LIMIT %(limit)s")
    rows = executor.records(sql, binds, tag=executor.statement_tag(f'history:{scope}'))
    price = cfg.credit_price_usd
    for r in rows:
        c = r.get('credits') if r.get('credits') is not None else r.get('est_credits')
        r['usd'] = round(float(c or 0) * price, 6)
        text = r.get('query_text') or ''
        r['query_text'] = text if len(text) <= 4000 else text[:4000] + '…'
    return {'scope': scope, 'rows': rows, 'limit': limit, 'credit_price_usd': price}
