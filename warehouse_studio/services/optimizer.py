"""The cost & performance toolkit as the Studio sees it.

``DETECTORS`` and ``REPORTS`` describe every stored procedure installed by
``snowflake/warehouse_studio/install.sql`` — parameters, defaults, limits and
what the procedure looks for — so the UI can render forms and the API can
validate input before anything reaches Snowflake.  Calls are built with bind
parameters only; no user input is ever spliced into SQL text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..conf import CREDITS_PER_HOUR, get_config
from . import executor, sources


@dataclass(frozen=True)
class Param:
    name: str
    label: str
    type: str                  # int | float | identifier | text
    default: object = None
    min: float | None = None
    max: float | None = None
    help: str = ''
    required: bool = False

    def coerce(self, raw):
        if raw in (None, ''):
            if self.required:
                raise ValueError(f'{self.label} is required.')
            return self.default
        if self.type == 'int':
            try:
                value = int(float(raw))
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{self.label} must be a whole number.') from exc
        elif self.type == 'float':
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{self.label} must be a number.') from exc
        elif self.type == 'identifier':
            value = str(raw).strip()
            if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_$]*(\.[A-Za-z_][A-Za-z0-9_$]*){0,2}', value):
                raise ValueError(f'{self.label} must look like DATABASE.SCHEMA.TABLE (unquoted).')
            return value
        else:
            value = str(raw)
            if len(value) > 2000:
                raise ValueError(f'{self.label} is too long.')
            return value
        if self.min is not None and value < self.min:
            raise ValueError(f'{self.label} must be at least {self.min:g}.')
        if self.max is not None and value > self.max:
            raise ValueError(f'{self.label} must be at most {self.max:g}.')
        return value

    def as_dict(self) -> dict:
        return {'name': self.name, 'label': self.label, 'type': self.type, 'default': self.default,
                'min': self.min, 'max': self.max, 'help': self.help, 'required': self.required}


LOOKBACK = Param('P_LOOKBACK_DAYS', 'Look-back (days)', 'int', 7, 1, 365, 'How far back in query history to look.')


@dataclass(frozen=True)
class Procedure:
    key: str
    procedure: str
    title: str
    category: str              # performance | cost | storage | concurrency
    icon: str
    summary: str
    detects: str
    fix: str
    params: tuple[Param, ...] = ()
    returns: str = 'status'    # status (detector JSON) | table | variant
    kind: str = 'detector'
    heavy: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)

    def fq(self) -> str:
        return f'{get_config().toolkit}.{self.procedure}'

    def as_dict(self) -> dict:
        return {'key': self.key, 'procedure': self.procedure, 'fq': self.fq(), 'title': self.title,
                'category': self.category, 'icon': self.icon, 'summary': self.summary, 'detects': self.detects,
                'fix': self.fix, 'params': [p.as_dict() for p in self.params], 'returns': self.returns,
                'kind': self.kind, 'heavy': self.heavy, 'tags': list(self.tags)}


DETECTORS: tuple[Procedure, ...] = (
    Procedure('join_explosion', 'SP_DETECT_JOIN_EXPLOSION', 'Join explosions', 'performance', 'diagram-3',
              'Cartesian joins hiding inside slow queries.',
              'Walks the operator profile (GET_QUERY_OPERATOR_STATS) of the slowest recent queries looking for '
              'CartesianJoin operators and the row multiple they produce.',
              'Count the join predicates — they should equal (number of tables − 1). Add the missing ON clause.',
              (Param('P_LOOKBACK_DAYS', 'Look-back (days)', 'int', 1, 1, 14, 'Operator stats are kept for 14 days.'),
               Param('P_MIN_ELAPSED_MS', 'Min elapsed (ms)', 'int', 60000, 0, 86_400_000, 'Only inspect queries slower than this.'),
               Param('P_MAX_QUERIES', 'Max queries to inspect', 'int', 200, 1, 2000, 'Each inspection is one profile lookup.')),
              heavy=True, tags=('profile', 'critical')),
    Procedure('long_compilation', 'SP_DETECT_LONG_COMPILATION', 'Long compilation', 'performance', 'cpu',
              'Queries that spent longer compiling than running.',
              'Successful queries whose compilation time is at least the threshold and exceeds execution time.',
              'Shorten the SQL: bind variables instead of literals, big IN lists into a temp-table join, flatten '
              'nested views. A bigger warehouse will not help — compilation runs in cloud services.',
              (LOOKBACK, Param('P_MIN_COMPILE_MS', 'Min compile (ms)', 'int', 1000, 0, 3_600_000))),
    Procedure('long_execution', 'SP_DETECT_LONG_EXECUTION', 'Long execution', 'performance', 'hourglass-split',
              'Slow queries, each with a diagnosis of why.',
              'Queries running longer than the threshold, classified as spill, queueing, poor pruning or compute bound.',
              'Follow the per-finding diagnosis: size up for spill, fix pruning for scans, inspect the profile otherwise.',
              (LOOKBACK, Param('P_MIN_EXECUTION_MS', 'Min execution (ms)', 'int', 60000, 0, 86_400_000))),
    Procedure('poor_pruning', 'SP_DETECT_POOR_PRUNING', 'Poor pruning', 'performance', 'funnel',
              'Filters that read most of a big table anyway.',
              'Queries scanning more than the ratio of partitions on tables with at least the partition threshold.',
              'Remove functions/casts around filter columns, prefer equality predicates, or cluster on the filter column.',
              (LOOKBACK, Param('P_MIN_PARTITIONS', 'Min table partitions', 'int', 1000, 1, 10_000_000),
               Param('P_MIN_RATIO', 'Min scan ratio', 'float', 0.5, 0.01, 1.0, 'Fraction of partitions scanned.'))),
    Procedure('spill_and_oom', 'SP_DETECT_SPILL_AND_OOM', 'Spill & out-of-memory', 'performance', 'droplet-half',
              'Memory pressure: local/remote spill and OOM failures.',
              'Queries that spilled to local or remote storage, or failed with a memory error.',
              'Remote spill: size up (often cheaper overall). High network bytes too: fix join-key skew.',
              (LOOKBACK,)),
    Procedure('queueing', 'SP_DETECT_QUEUEING', 'Queueing', 'concurrency', 'stoplights',
              'Warehouse-hours where queries waited for a slot.',
              'Hours with more than a minute of overload queueing per warehouse.',
              'Re-time batches, lower MAX_CONCURRENCY_LEVEL or split workloads. A little queueing is healthy.',
              (LOOKBACK,)),
    Procedure('blocked_transactions', 'SP_DETECT_BLOCKED_TRANSACTIONS', 'Blocked transactions', 'concurrency', 'lock',
              'Statements that waited on locks.',
              'Queries with transaction-blocked time (another transaction held the lock).',
              'Avoid shared log/control tables, batch DML, or segment the target table.',
              (LOOKBACK,)),
    Procedure('warehouse_waste', 'SP_DETECT_WAREHOUSE_WASTE', 'Idle warehouse burn', 'cost', 'fire',
              'Warehouses billing while doing (almost) nothing.',
              'Metered warehouse-hours with more than 0.05 credits but under five minutes of query execution.',
              'Set AUTO_SUSPEND = 60 and find sessions/BI tools keeping the warehouse awake.',
              (LOOKBACK,), tags=('money',)),
    Procedure('warehouse_settings', 'SP_DETECT_WAREHOUSE_SETTINGS', 'Warehouse settings', 'cost', 'sliders',
              'AUTO_SUSPEND unset or longer than 60 s.',
              'Every warehouse whose auto-suspend is missing, zero, or above 60 seconds.',
              'ALTER WAREHOUSE … SET AUTO_SUSPEND = 60 (billing is per second after the first minute).',
              (), tags=('money',)),
    Procedure('storage_overhead', 'SP_DETECT_STORAGE_OVERHEAD', 'Storage overhead', 'storage', 'hdd-stack',
              'Tables whose Time Travel + Fail-safe outweigh their data.',
              'Tables above the size threshold where (time-travel + fail-safe) / active bytes > 1.',
              'Lower DATA_RETENTION_TIME_IN_DAYS or make reloadable tables TRANSIENT; batch high-churn DML.',
              (Param('P_MIN_ACTIVE_GB', 'Min active size (GB)', 'float', 1.0, 0, 1_000_000),
               Param('P_RATE_PER_TB_MONTH', 'Storage $/TB-month', 'float', 23.0, 0, 1000))),
    Procedure('select_star', 'SP_DETECT_SELECT_STAR', 'SELECT *', 'performance', 'asterisk',
              'Slow SELECT * query shapes.',
              'Query shapes using SELECT * whose executions run longer than the threshold.',
              'Name only the columns you need — free to fix, compounds on every execution.',
              (LOOKBACK, Param('P_MIN_EXECUTION_MS', 'Min execution (ms)', 'int', 10000, 0, 86_400_000))),
    Procedure('repeat_offenders', 'SP_DETECT_REPEAT_OFFENDERS', 'Repeat offenders', 'cost', 'arrow-repeat',
              'The query shapes that cost the most in total.',
              'Top-N parameterized query shapes by total execution time, with at least N executions.',
              'Start here: a fix to one shape pays off on every execution.',
              (LOOKBACK, Param('P_MIN_EXECUTIONS', 'Min executions', 'int', 5, 1, 1_000_000),
               Param('P_TOP_N', 'Top N', 'int', 20, 1, 500)), tags=('money',)),
)

REPORTS: tuple[Procedure, ...] = (
    Procedure('full_analysis', 'SP_RUN_FULL_OPTIMIZATION_ANALYSIS', 'Full optimization analysis', 'performance',
              'clipboard2-pulse', 'Long-running, spilling, queued and poorly-pruned work in one table.',
              'Top-N of each problem class over the look-back window.', 'Open each query to see its profile.',
              (Param('P_DAYS_BACK', 'Look-back (days)', 'int', 7, 1, 365), Param('P_TOP_N', 'Top N per class', 'int', 5, 1, 100)),
              returns='table', kind='report'),
    Procedure('time_travel_tables', 'SP_FIND_HIGH_TIME_TRAVEL_TABLES', 'Time-travel heavy tables', 'storage', 'clock-history',
              'Tables carrying the most Time Travel storage.', 'Live tables ranked by time-travel bytes.',
              'Shorten retention or go TRANSIENT for reloadable data.',
              (Param('P_TOP_N', 'Top N', 'int', 20, 1, 500), Param('P_MIN_TIME_TRAVEL_GB', 'Min time travel (GB)', 'float', 1.0, 0, 1_000_000)),
              returns='table', kind='report'),
    Procedure('clustering_costs', 'SP_MONITOR_CLUSTERING_COSTS', 'Automatic clustering costs', 'cost', 'boxes',
              'Credits spent by automatic clustering per table.', 'AUTOMATIC_CLUSTERING_HISTORY rolled up per table.',
              'Suspend reclustering on tables whose queries no longer filter on the key.',
              (Param('P_DAYS_BACK', 'Look-back (days)', 'int', 7, 1, 365),), returns='table', kind='report'),
    Procedure('clustering_report', 'SP_CLUSTERING_REPORT', 'Clustering health', 'performance', 'bar-chart-steps',
              'Depth and overlap of a table\'s micro-partitions for a key.', 'SYSTEM$CLUSTERING_INFORMATION, summarised.',
              'Average depth near 1 is ideal; high depth means filters on this key cannot prune.',
              (Param('P_TABLE', 'Table', 'identifier', None, help='DATABASE.SCHEMA.TABLE', required=True),
               Param('P_KEY', 'Clustering key (optional)', 'text', None, help='e.g. (event_date) — blank uses the declared key')),
              returns='variant', kind='report'),
    Procedure('sos_cost', 'SP_ESTIMATE_SOS_COST', 'Search optimization estimate', 'cost', 'search',
              'What search optimization would cost to build and maintain.', 'SYSTEM$ESTIMATE_SEARCH_OPTIMIZATION_COSTS.',
              'Worth it for selective point-lookups on large tables that cannot be clustered for them.',
              (Param('P_TABLE_NAME', 'Table', 'identifier', None, help='DATABASE.SCHEMA.TABLE', required=True),),
              returns='variant', kind='report'),
    Procedure('run_all', 'SP_RUN_ALL_DETECTORS', 'Run every detector', 'performance', 'lightning-charge',
              'One batch across all twelve detectors.', 'Calls each detector; failures are logged, never fatal.',
              'Results land in the findings explorer.',
              (Param('P_LOOKBACK_DAYS', 'Look-back (days)', 'int', 2, 1, 365),), returns='status', kind='batch', heavy=True),
)

ALL = {p.key: p for p in DETECTORS + REPORTS}
HELPERS = ('SP_WS_WAREHOUSES', 'SP_WS_RESOURCE_MONITORS', 'SP_WS_QUERY_PROFILE', 'SP_WS_DETECTOR_SCHEDULE')


def callable_procedures() -> frozenset[str]:
    """Fully-qualified names that may be CALLed from a worksheet in read mode."""
    tk = get_config().toolkit
    return frozenset(f'{tk}.{p.procedure}' for p in ALL.values()) | frozenset(f'{tk}.{h}' for h in HELPERS)


def get(key: str) -> Procedure:
    try:
        return ALL[key]
    except KeyError as exc:
        raise ValueError(f'Unknown detector or report {key!r}.') from exc


def build_call(proc: Procedure, raw_params: dict | None) -> tuple[str, dict, dict]:
    """``(sql, binds, clean_params)`` for a CALL.  Raises ValueError with a friendly message."""
    raw_params = raw_params or {}
    clean, binds, placeholders = {}, {}, []
    for i, p in enumerate(proc.params):
        value = p.coerce(raw_params.get(p.name))
        clean[p.name] = value
        binds[f'p{i}'] = value
        placeholders.append(f'%(p{i})s')
    return f'CALL {proc.fq()}({", ".join(placeholders)})', binds, clean


def start(key: str, raw_params: dict | None, *, user: str = '') -> dict:
    """Submit a detector/report asynchronously.  Returns ``{query_id, sql, params}``."""
    proc = get(key)
    state = sources.probe()
    if not state.get('toolkit_procedures', {}).get(proc.procedure):
        raise sources.SourceUnavailable(proc.procedure.lower())
    sql, binds, clean = build_call(proc, raw_params)
    started = executor.submit(sql, params=binds, tag=executor.statement_tag(f'optimizer:{key}', user))
    return {**started, 'sql': sql, 'params': clean, 'procedure': proc.as_dict()}


def parse_outcome(proc: Procedure, result: dict) -> dict:
    """Interpret a finished CALL's result set for the UI."""
    import json

    rows = result.get('rows') or []
    if proc.returns == 'table':
        return {'type': 'table', 'columns': result.get('columns', []), 'rows': rows, 'total': result.get('total')}
    cell = rows[0][0] if rows and rows[0] else None
    if isinstance(cell, dict) and '$json' in cell:
        cell = cell['$json']
    if isinstance(cell, str):
        try:
            cell = json.loads(cell)
        except ValueError:
            pass
    return {'type': 'variant' if proc.returns == 'variant' else 'status', 'value': cell}


# ----------------------------------------------------------------------------- findings

SEVERITIES = ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')
SEVERITY_WEIGHT = {'CRITICAL': 10, 'HIGH': 4, 'MEDIUM': 1, 'LOW': 0.25}


def _cph_case(expr: str) -> str:
    """SQL CASE mapping a warehouse-size expression to credits/hour."""
    seen, whens = set(), []
    for label, cph in CREDITS_PER_HOUR.items():
        if label in seen:
            continue
        seen.add(label)
        whens.append(f"WHEN '{label}' THEN {cph}")
    return f"(CASE UPPER(REPLACE({expr}, ' ', '')) {' '.join(whens)} ELSE 1 END)"


def _est_credits_sql() -> str:
    # Warehouse-waste findings carry the hour's metered credits; query findings
    # carry wasted seconds on a warehouse of known size; storage has a $ figure.
    return (f"CASE WHEN f.detector = 'sp_detect_warehouse_waste' THEN f.detail:credits::FLOAT "
            f"WHEN f.wasted_seconds IS NOT NULL THEN f.wasted_seconds / 3600 * "
            f"{_cph_case('f.detail:warehouse_size::STRING')} ELSE 0 END")


def _filters(params: dict) -> tuple[str, dict]:
    where, binds = ['f.detected_at >= DATEADD(\'day\', -%(days)s, CURRENT_TIMESTAMP())'], {}
    try:
        binds['days'] = max(1, min(int(params.get('days') or 30), 365))
    except (TypeError, ValueError):
        binds['days'] = 30
    sev = [s.upper() for s in _as_list(params.get('severity')) if s.upper() in SEVERITIES]
    if sev:
        where.append('f.severity IN (' + ', '.join(f'%(sev{i})s' for i in range(len(sev))) + ')')
        binds.update({f'sev{i}': s for i, s in enumerate(sev)})
    det = [d for d in _as_list(params.get('detector')) if re.fullmatch(r'sp_[a-z_]+', d)]
    if det:
        where.append('f.detector IN (' + ', '.join(f'%(det{i})s' for i in range(len(det))) + ')')
        binds.update({f'det{i}': d for i, d in enumerate(det)})
    obj = [o.upper() for o in _as_list(params.get('object_type')) if o.upper() in ('QUERY', 'WAREHOUSE', 'TABLE')]
    if obj:
        where.append('f.object_type IN (' + ', '.join(f'%(obj{i})s' for i in range(len(obj))) + ')')
        binds.update({f'obj{i}': o for i, o in enumerate(obj)})
    q = str(params.get('q') or '').strip()
    if q:
        where.append('(f.object_name ILIKE %(q)s OR f.recommendation ILIKE %(q)s OR f.detail::STRING ILIKE %(q)s)')
        binds['q'] = f'%{q[:200]}%'
    ids = [int(i) for i in _as_list(params.get('ids')) if str(i).isdigit()]
    if ids:
        where.append('f.finding_id IN (' + ', '.join(str(i) for i in ids[:500]) + ')')
    return ' AND '.join(where), binds


def _as_list(value) -> list[str]:
    if value is None or value == '':
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    return [v.strip() for v in str(value).split(',') if v.strip()]


SORTS = {'wasted': 'f.wasted_seconds DESC NULLS LAST', 'recent': 'f.detected_at DESC', 'occurred': 'f.occurred_at DESC NULLS LAST',
         'severity': "CASE f.severity WHEN 'CRITICAL' THEN 0 WHEN 'HIGH' THEN 1 WHEN 'MEDIUM' THEN 2 ELSE 3 END, "
                     "f.wasted_seconds DESC NULLS LAST",
         'cost': 'est_credits DESC NULLS LAST', 'metric': 'f.metric_value DESC NULLS LAST'}


def findings(params: dict) -> dict:
    table = sources.toolkit_object('TUNING_FINDINGS')
    where, binds = _filters(params)
    try:
        limit = max(1, min(int(params.get('limit') or 50), 500))
        offset = max(0, int(params.get('offset') or 0))
    except (TypeError, ValueError):
        limit, offset = 50, 0
    order = SORTS.get(str(params.get('sort') or 'severity'), SORTS['severity'])
    binds.update({'limit': limit, 'offset': offset})
    rows = executor.records(
        f"SELECT f.finding_id, f.detector, f.severity, f.object_type, f.object_name, f.occurred_at, f.detected_at, "
        f"f.metric_name, f.metric_value, f.wasted_seconds, f.detail, f.recommendation, "
        f"{_est_credits_sql()} AS est_credits, COUNT(*) OVER () AS total_count "
        f"FROM {table} f WHERE {where} ORDER BY {order} LIMIT %(limit)s OFFSET %(offset)s",
        binds, tag=executor.statement_tag('findings'))
    price = get_config().credit_price_usd
    total = int(rows[0]['total_count']) if rows else 0
    for r in rows:
        r.pop('total_count', None)
        r['est_usd'] = round((r.get('est_credits') or 0) * price, 4)
    return {'findings': rows, 'total': total, 'limit': limit, 'offset': offset, 'credit_price_usd': price}


def summary(params: dict) -> dict:
    """Roll-ups for the Optimizer dashboard: by severity, detector, day, object; health score."""
    table = sources.toolkit_object('TUNING_FINDINGS')
    where, binds = _filters(params)
    rows = executor.records(
        f"SELECT f.detector, f.severity, f.object_type, TO_DATE(f.detected_at) AS day, COUNT(*) AS findings, "
        f"SUM(COALESCE(f.wasted_seconds, 0)) AS wasted_seconds, SUM({_est_credits_sql()}) AS est_credits, "
        f"COUNT(DISTINCT f.object_name) AS objects, MAX(f.detected_at) AS last_detected "
        f"FROM {table} f WHERE {where} GROUP BY 1, 2, 3, 4",
        binds, tag=executor.statement_tag('findings-summary'))
    price = get_config().credit_price_usd
    by_sev = {s: 0 for s in SEVERITIES}
    by_det: dict[str, dict] = {}
    by_day: dict[str, dict] = {}
    wasted = credits = 0.0
    for r in rows:
        n = int(r['findings'] or 0)
        sev = (r['severity'] or 'LOW').upper()
        by_sev[sev] = by_sev.get(sev, 0) + n
        d = by_det.setdefault(r['detector'], {'detector': r['detector'], 'findings': 0, 'wasted_seconds': 0.0,
                                              'est_credits': 0.0, 'severity': {}, 'last_detected': None})
        d['findings'] += n
        d['wasted_seconds'] += float(r['wasted_seconds'] or 0)
        d['est_credits'] += float(r['est_credits'] or 0)
        d['severity'][sev] = d['severity'].get(sev, 0) + n
        if r['last_detected'] and (d['last_detected'] is None or r['last_detected'] > d['last_detected']):
            d['last_detected'] = r['last_detected']
        day = by_day.setdefault(str(r['day']), {'day': str(r['day']), **{s: 0 for s in SEVERITIES}})
        day[sev] = day.get(sev, 0) + n
        wasted += float(r['wasted_seconds'] or 0)
        credits += float(r['est_credits'] or 0)
    for d in by_det.values():
        d['est_usd'] = round(d['est_credits'] * price, 2)
    return {
        'by_severity': by_sev,
        'by_detector': sorted(by_det.values(), key=lambda d: -d['wasted_seconds']),
        'by_day': sorted(by_day.values(), key=lambda d: d['day']),
        'total': sum(by_sev.values()), 'wasted_seconds': round(wasted, 1), 'est_credits': round(credits, 4),
        'est_usd': round(credits * price, 2), 'health_score': health_score(by_sev), 'credit_price_usd': price,
    }


def health_score(by_severity: dict) -> int:
    """0-100: 100 = nothing to fix.  Weighted severity load on a soft (log-like) curve.

    ``100 / (1 + load/25)`` keeps the score meaningful for both a handful of
    findings and hundreds of them, without ever reaching zero.
    """
    load = sum(SEVERITY_WEIGHT.get(k.upper(), 0) * int(v or 0) for k, v in by_severity.items())
    return int(round(100 / (1 + load / 25)))


def recent_runs(limit: int = 20) -> list[dict]:
    table = sources.toolkit_object('WS_DETECTOR_RUNS')
    return executor.records(
        f"SELECT run_id, batch_id, detector, started_at, finished_at, status, result, error_message, "
        f"DATEDIFF('millisecond', started_at, finished_at) AS duration_ms FROM {table} "
        f"ORDER BY started_at DESC LIMIT %(n)s", {'n': max(1, min(int(limit), 500))},
        tag=executor.statement_tag('detector-runs'))


def schedule(action: str = 'STATUS') -> dict:
    action = str(action or 'STATUS').upper()
    if action not in ('STATUS', 'RESUME', 'SUSPEND'):
        raise ValueError('Action must be STATUS, RESUME or SUSPEND.')
    proc = sources.toolkit_object('SP_WS_DETECTOR_SCHEDULE')
    res = executor.run(f'CALL {proc}(%s)', (action,), tag=executor.statement_tag('schedule'))
    value = res.rows[0][0] if res.rows else None
    return value if isinstance(value, dict) else {'raw': value}
