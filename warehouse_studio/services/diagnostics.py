"""Explain one query: what it cost, why it was slow, and what to do about it.

Snowsight shows a query's numbers; this module reads them.  ``diagnose``
applies the same rules as the toolkit's detectors to a single query's stats,
``analyze_profile`` turns GET_QUERY_OPERATOR_STATS output into a graph with
hot-spots, and ``preflight`` estimates a statement's cost *before* it runs
from ``EXPLAIN USING JSON``.
"""

from __future__ import annotations

import json
import re

from ..conf import credits_per_hour, get_config
from . import executor, sources

GB = 1024 ** 3


def _n(v) -> float:
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def insight(severity: str, title: str, detail: str, fix: str, metric: str = '', category: str = 'performance') -> dict:
    return {'severity': severity, 'title': title, 'detail': detail, 'fix': fix, 'metric': metric, 'category': category}


def fmt_bytes(n: float) -> str:
    n = _n(n)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB', 'PB'):
        if abs(n) < 1024 or unit == 'PB':
            return f'{n:.0f} {unit}' if unit == 'B' else f'{n:.1f} {unit}'
        n /= 1024
    return f'{n:.1f} PB'


def fmt_ms(ms: float) -> str:
    ms = _n(ms)
    if ms < 1000:
        return f'{ms:.0f} ms'
    s = ms / 1000
    if s < 60:
        return f'{s:.1f} s'
    m, s = divmod(s, 60)
    if m < 60:
        return f'{int(m)}m {int(s)}s'
    h, m = divmod(m, 60)
    return f'{int(h)}h {int(m)}m'


def diagnose(q: dict, operators: list[dict] | None = None) -> list[dict]:
    """Rules over a QUERY_HISTORY row (lower-case keys).  Most severe first."""
    out: list[dict] = []
    elapsed, compile_ms, exec_ms = _n(q.get('total_elapsed_time')), _n(q.get('compilation_time')), _n(q.get('execution_time'))
    queued = _n(q.get('queued_overload_time'))
    provisioning = _n(q.get('queued_provisioning_time')) + _n(q.get('queued_repair_time'))
    blocked = _n(q.get('transaction_blocked_time'))
    local, remote = _n(q.get('bytes_spilled_to_local_storage')), _n(q.get('bytes_spilled_to_remote_storage'))
    scanned, total = _n(q.get('partitions_scanned')), _n(q.get('partitions_total'))
    network = _n(q.get('bytes_sent_over_the_network'))
    status = str(q.get('execution_status') or '').upper()
    text = str(q.get('query_text') or '')
    size = q.get('warehouse_size')

    if status in ('FAIL', 'FAILED_WITH_ERROR') and 'memory' in str(q.get('error_message') or '').lower():
        out.append(insight('CRITICAL', 'Out of memory', 'The query failed because the warehouse ran out of memory.',
                           'Run it on a larger warehouse, or reduce the working set (filter earlier, fewer columns).'))
    if remote > 0:
        out.append(insight('CRITICAL', 'Remote spill', f'{fmt_bytes(remote)} spilled to remote storage — the slowest place data can go.',
                           'Size the warehouse up one or two steps. Removing remote spill often more than halves runtime, '
                           'so the bigger warehouse is frequently cheaper overall.', fmt_bytes(remote)))
    elif local > GB:
        out.append(insight('HIGH', 'Local spill', f'{fmt_bytes(local)} spilled to local disk.',
                           'Size up one step, or shrink the working set: project fewer columns, filter before joining, '
                           'avoid ORDER BY on very large sets.', fmt_bytes(local)))
    elif local > 0:
        out.append(insight('LOW', 'Minor local spill', f'{fmt_bytes(local)} spilled to local disk.',
                           'Usually harmless; watch it if it grows.', fmt_bytes(local)))
    if (local or remote) and network > GB:
        out.append(insight('HIGH', 'Possible join skew', f'{fmt_bytes(network)} moved between nodes alongside spill.',
                           'One join key may carry most rows. Check key distribution; sizing up masks skew, a rewrite fixes it.',
                           fmt_bytes(network)))
    if total >= 1000 and total and scanned / total > 0.8:
        out.append(insight('HIGH' if scanned / total > 0.95 else 'MEDIUM', 'Poor partition pruning',
                           f'Scanned {scanned:,.0f} of {total:,.0f} micro-partitions ({scanned / total:.0%}).',
                           'Filters are not eliminating data. Avoid functions/casts on filter columns, prefer equality and '
                           'range predicates on the clustering key, or cluster the table by what you filter on.',
                           f'{scanned / total:.0%}'))
    if compile_ms >= 1000 and compile_ms > exec_ms:
        out.append(insight('HIGH' if compile_ms > 10000 else 'MEDIUM', 'Compilation dominates',
                           f'Compiling took {fmt_ms(compile_ms)} vs {fmt_ms(exec_ms)} executing.',
                           'Shorten the SQL text: bind variables instead of literals, big IN lists into a temp table join, '
                           'flatten deeply nested views. A bigger warehouse will not help.', fmt_ms(compile_ms)))
    if queued > 1000 and elapsed and queued > elapsed * 0.3:
        out.append(insight('HIGH' if queued > elapsed * 0.6 else 'MEDIUM', 'Waited in the queue',
                           f'{fmt_ms(queued)} ({queued / elapsed:.0%} of the elapsed time) was spent queued behind other work.',
                           'This is concurrency, not query speed: re-time the batch, split workloads across warehouses, '
                           'or enable multi-cluster scaling (Enterprise).', fmt_ms(queued), 'concurrency'))
    if provisioning > 2000:
        out.append(insight('LOW', 'Warehouse start-up', f'{fmt_ms(provisioning)} waiting for the warehouse to resume.',
                           'Expected after auto-suspend. Only keep it warm if latency matters more than idle credits.',
                           fmt_ms(provisioning), 'cost'))
    if blocked > 0:
        out.append(insight('HIGH' if blocked > 300000 else 'MEDIUM', 'Blocked by a lock',
                           f'{fmt_ms(blocked)} waiting for another transaction to release a lock.',
                           'Avoid concurrent DML on one table (logs, control tables); batch the writes.', fmt_ms(blocked), 'concurrency'))
    if re.search(r'select\s+\*', text, re.IGNORECASE) and _n(q.get('bytes_scanned')) > GB:
        out.append(insight('LOW', 'SELECT *', f'Reads every column ({fmt_bytes(q.get("bytes_scanned"))} scanned).',
                           'Name only the columns you need — Snowflake stores columns separately, so unused ones are pure cost.'))
    rows = _n(q.get('rows_produced'))
    if rows > 1_000_000 and str(q.get('query_type') or '').upper() == 'SELECT':
        out.append(insight('LOW', 'Very large result', f'Returned {rows:,.0f} rows to the client.',
                           'Aggregate in Snowflake or unload with COPY INTO a stage instead of pulling rows to a client.'))
    cache_pct = _n(q.get('percentage_scanned_from_cache'))
    if cache_pct >= 0.8 and _n(q.get('bytes_scanned')) > 0:
        out.append(insight('INFO', 'Served from warehouse cache', f'{cache_pct:.0%} of scanned data came from the local disk cache.',
                           'Good: keeping similar queries on the same warehouse preserves this.', f'{cache_pct:.0%}', 'cost'))
    for op in operators or []:
        if op.get('operator_type') == 'CartesianJoin':
            out.append(insight('CRITICAL', 'Cartesian join', f'Operator #{op.get("operator_id")} joins every row to every row.',
                               'A join predicate is missing: count them — they should equal (tables − 1).'))
            break
    for op in operators or []:
        st = op.get('operator_statistics') or {}
        inp, outp = _n(st.get('input_rows')), _n(st.get('output_rows'))
        if op.get('operator_type') == 'Join' and inp > 0 and outp / inp > 10 and outp > 1_000_000:
            out.append(insight('HIGH', 'Exploding join', f'Join #{op.get("operator_id")} turns {inp:,.0f} rows into {outp:,.0f}.',
                               'Duplicate join keys multiply rows. De-duplicate one side or join on the full key.',
                               f'×{outp / inp:,.0f}'))
            break
    if not out and status == 'SUCCESS':
        out.append(insight('INFO', 'Looks healthy', 'No spill, pruning, queueing, compilation or locking problems detected.',
                           'Nothing to do.', category='performance'))
    if size is None and status == 'SUCCESS' and not _n(q.get('bytes_scanned')):
        out.append(insight('INFO', 'Metadata-only', 'Answered from metadata or the result cache — no warehouse compute used.',
                           'Free.', category='cost'))
    order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3, 'INFO': 4}
    out.sort(key=lambda i: order.get(i['severity'], 9))
    return out


def query_cost(q: dict, attributed_credits: float | None = None) -> dict:
    """Credits/$ for one query: attributed credits if known, else execution time × warehouse rate."""
    cph = credits_per_hour(q.get('warehouse_size'))
    estimated = _n(q.get('execution_time')) / 3_600_000 * cph
    credits = attributed_credits if attributed_credits is not None else estimated
    cloud = _n(q.get('credits_used_cloud_services'))
    price = get_config().credit_price_usd
    return {'credits': round(credits, 6), 'cloud_credits': round(cloud, 6), 'estimated': attributed_credits is None,
            'credits_per_hour': cph, 'usd': round((credits + cloud) * price, 4), 'credit_price_usd': price}


# ----------------------------------------------------------------------------- profile

def analyze_profile(rows: list[dict]) -> dict:
    """Shape operator stats for the graph: nodes, edges, % time, hottest path, totals."""
    nodes, edges = [], []
    for r in rows:
        stats = _maybe_json(r.get('operator_statistics')) or {}
        breakdown = _maybe_json(r.get('execution_time_breakdown')) or {}
        attrs = _maybe_json(r.get('operator_attributes')) or {}
        parents = _maybe_json(r.get('parent_operators')) or []
        node_id = f"{r.get('step_id', 1)}:{r.get('operator_id')}"
        nodes.append({
            'id': node_id, 'step': r.get('step_id'), 'operator_id': r.get('operator_id'),
            'type': r.get('operator_type'), 'stats': stats, 'breakdown': breakdown, 'attributes': attrs,
            'percent': _n(breakdown.get('overall_percentage')) * 100,
            'input_rows': stats.get('input_rows'), 'output_rows': stats.get('output_rows'),
            'label': _operator_label(r.get('operator_type'), attrs),
        })
        for p in parents or []:
            edges.append({'from': node_id, 'to': f"{r.get('step_id', 1)}:{p}", 'rows': stats.get('output_rows')})
    hottest = sorted(nodes, key=lambda n: -n['percent'])[:3]
    spill = sum(_n((n['stats'].get('spilling') or {}).get('bytes_spilled_local_storage'))
                + _n((n['stats'].get('spilling') or {}).get('bytes_spilled_remote_storage')) for n in nodes)
    return {'nodes': nodes, 'edges': edges, 'hottest': [n['id'] for n in hottest],
            'operators': len(nodes), 'steps': len({n['step'] for n in nodes}), 'spilled_bytes': spill}


def _operator_label(op_type, attrs: dict) -> str:
    if not isinstance(attrs, dict):
        return ''
    for key in ('table_name', 'join_condition', 'equality_join_condition', 'filter_condition', 'grouping_keys',
                'sort_keys', 'functions', 'input_expressions'):
        v = attrs.get(key)
        if v:
            v = ', '.join(map(str, v)) if isinstance(v, list) else str(v)
            return v[:160]
    return ''


def _maybe_json(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def load_profile(query_id: str) -> dict:
    query_id = executor.valid_query_id(query_id)
    state = sources.probe()
    if state.get('toolkit_procedures', {}).get('SP_WS_QUERY_PROFILE'):
        rows = executor.records(f'CALL {get_config().toolkit}.SP_WS_QUERY_PROFILE(%s)', (query_id,),
                                tag=executor.statement_tag('profile'))
    else:
        rows = executor.records('SELECT * FROM TABLE(GET_QUERY_OPERATOR_STATS(%s))', (query_id,),
                                tag=executor.statement_tag('profile'))
    return analyze_profile(rows)


# ----------------------------------------------------------------------------- query details

DETAIL_COLUMNS = (
    'query_id, query_text, query_type, query_tag, database_name, schema_name, user_name, role_name, '
    'warehouse_name, warehouse_size, warehouse_type, cluster_number, execution_status, error_code, error_message, '
    'start_time, end_time, total_elapsed_time, compilation_time, execution_time, queued_provisioning_time, '
    'queued_repair_time, queued_overload_time, transaction_blocked_time, bytes_scanned, percentage_scanned_from_cache, '
    'bytes_written, bytes_spilled_to_local_storage, bytes_spilled_to_remote_storage, bytes_sent_over_the_network, '
    'rows_produced, partitions_scanned, partitions_total, credits_used_cloud_services, query_parameterized_hash')


def load_query(query_id: str) -> dict | None:
    """One query's history row.

    The 30-day account view has the complete statistics (partitions, spill,
    network) but lags up to ~45 minutes; the INFORMATION_SCHEMA table function
    is real time but has fewer columns.  Try complete first, then real time.
    """
    from .history import live_columns

    query_id = executor.valid_query_id(query_id)
    cfg = get_config()
    try:
        rows = executor.records(
            f'SELECT {DETAIL_COLUMNS} FROM {sources.relation("query_history")} '
            f"WHERE query_id = %s AND start_time >= DATEADD('day', -30, CURRENT_TIMESTAMP())",
            (query_id,), tag=executor.statement_tag('query-detail'))
        if rows:
            return {**rows[0], 'source': 'account_usage', 'complete': True}
    except (sources.SourceUnavailable, executor.QueryError):
        pass
    db = cfg.database or cfg.toolkit_database
    for fn in ('QUERY_HISTORY_BY_USER', 'QUERY_HISTORY'):
        try:
            rows = executor.records(
                f'SELECT {live_columns(DETAIL_COLUMNS)} FROM TABLE("{db}".INFORMATION_SCHEMA.{fn}(RESULT_LIMIT => 10000)) '
                f'WHERE query_id = %s', (query_id,), tag=executor.statement_tag('query-detail'))
        except executor.QueryError:
            continue
        if rows:
            return {**rows[0], 'source': 'information_schema', 'complete': False}
    return None


def attributed_credits(query_id: str) -> float | None:
    try:
        rows = executor.records(
            f'SELECT SUM(credits_attributed_compute) AS c FROM {sources.relation("query_attribution")} WHERE query_id = %s',
            (query_id,), tag=executor.statement_tag('query-detail'))
    except Exception:  # noqa: BLE001
        return None
    value = rows[0]['c'] if rows else None
    return float(value) if value is not None else None


# ----------------------------------------------------------------------------- pre-flight (EXPLAIN)

# Rough scan throughput per credit-per-hour of warehouse (bytes/second) when
# no history is available.  ~100 MB/s on an X-Small, scaling with size.
DEFAULT_THROUGHPUT_PER_CPH = 100 * 1024 ** 2


def parse_explain(plan: dict | str) -> dict:
    plan = _maybe_json(plan) or {}
    stats = plan.get('GlobalStats') or {}
    ops = []
    for group in plan.get('Operations') or []:
        for op in group if isinstance(group, list) else [group]:
            ops.append({'id': op.get('id'), 'operation': op.get('operation'), 'objects': op.get('objects') or [],
                        'expressions': op.get('expressions') or [], 'parents': op.get('parentOperators') or [],
                        'partitions_assigned': op.get('partitionsAssigned'), 'partitions_total': op.get('partitionsTotal'),
                        'bytes_assigned': op.get('bytesAssigned'), 'alias': op.get('alias')})
    total, assigned = _n(stats.get('partitionsTotal')), _n(stats.get('partitionsAssigned'))
    warnings = []
    if any((o['operation'] or '').lower() == 'cartesianjoin' for o in ops):
        warnings.append({'severity': 'CRITICAL', 'text': 'The plan contains a Cartesian join — a join condition is missing.'})
    for o in ops:
        pt, pa = _n(o['partitions_total']), _n(o['partitions_assigned'])
        if (o['operation'] or '').lower() == 'tablescan' and pt >= 1000 and pa / pt > 0.8:
            warnings.append({'severity': 'HIGH', 'text': f"Full scan of {', '.join(o['objects']) or 'a table'}: "
                                                         f"{pa:,.0f}/{pt:,.0f} partitions — no filter is pruning it."})
    return {'partitions_total': total, 'partitions_assigned': assigned, 'bytes_assigned': _n(stats.get('bytesAssigned')),
            'prune_ratio': round(1 - assigned / total, 4) if total else None, 'operations': ops, 'warnings': warnings,
            'tables': sorted({obj for o in ops for obj in o['objects']})}


def estimate_cost(bytes_assigned: float, warehouse: dict | None, throughput_per_cph: float | None = None) -> dict:
    """Seconds / credits / $ for scanning ``bytes_assigned`` on ``warehouse`` (settings row)."""
    size = (warehouse or {}).get('size')
    cph = credits_per_hour(size) or 1.0
    tput = (throughput_per_cph or DEFAULT_THROUGHPUT_PER_CPH) * cph
    seconds = bytes_assigned / tput if tput else 0
    suspended = str((warehouse or {}).get('state') or '').upper() == 'SUSPENDED'
    billed = max(seconds, 60.0) if suspended else seconds    # a resume bills a minimum of 60 s
    credits = billed / 3600 * cph
    price = get_config().credit_price_usd
    return {'seconds': round(seconds, 2), 'billed_seconds': round(billed, 2), 'credits': round(credits, 6),
            'usd': round(credits * price, 4), 'warehouse_size': size, 'credits_per_hour': cph,
            'resume_minimum_applies': suspended and seconds < 60, 'throughput_bytes_per_second': round(tput)}


def preflight(sql: str, *, context, warehouses: list[dict] | None = None) -> dict:
    res = executor.run(f'EXPLAIN USING JSON {sql}', context=context, tag=executor.statement_tag('explain'))
    plan = res.rows[0][0] if res.rows else '{}'
    parsed = parse_explain(plan)
    wh = None
    if warehouses and context.warehouse:
        wh = next((w for w in warehouses if str(w.get('name', '')).upper() == context.warehouse.upper()), None)
    parsed['estimate'] = estimate_cost(parsed['bytes_assigned'], wh or {'size': 'X-Small', 'state': 'STARTED'})
    return parsed
