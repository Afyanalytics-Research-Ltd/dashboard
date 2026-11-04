"""Warehouse Studio JSON API (mounted at ``/warehouse-studio/api/``).

Every endpoint requires a logged-in superuser (the Studio sees account-wide
cost data).  Errors are JSON with a stable shape::

    {"error": {"message": "...", "code": "write|query|toolkit|connection|invalid|...", ...}}
"""

from __future__ import annotations

import csv
import functools
import io
import json
import logging

from asgiref.sync import sync_to_async
from django.core.exceptions import PermissionDenied
from django.core.handlers.asgi import ASGIRequest
from django.db import connections as db_connections
from django.http import Http404, HttpRequest, HttpResponseNotAllowed, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone

from core.models import AuditLog

from .conf import get_config
from .models import DetectorRun, FindingTriage, QueryRun, Worksheet
from .services import catalog, cost, diagnostics, executor, history, optimizer, sources, sql_guard
from .services import session as session_svc
from .services.connection import Context, StudioConnectionError, default_context

logger = logging.getLogger(__name__)


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400, code: str = 'invalid', **extra):
        super().__init__(message)
        self.status, self.code, self.extra = status, code, extra


def _json_default(o):
    if hasattr(o, 'isoformat'):
        return o.isoformat()
    return str(o)


def ok(data, status: int = 200) -> JsonResponse:
    return JsonResponse(data, status=status, safe=False, json_dumps_params={'default': _json_default})


def fail(message: str, status: int, code: str, **extra) -> JsonResponse:
    return JsonResponse({'error': {'message': message, 'code': code, **extra}}, status=status,
                        json_dumps_params={'default': _json_default})


def body(request: HttpRequest) -> dict:
    if not request.body:
        return {}
    try:
        data = json.loads(request.body)
    except ValueError as exc:
        raise ApiError('Request body is not valid JSON.') from exc
    if not isinstance(data, dict):
        raise ApiError('Request body must be a JSON object.')
    return data


def studio_api(methods=('GET',)):
    """Auth + method check + uniform error handling.

    The returned view is *async*: under ASGI (Daphne) Django would otherwise
    run every sync view on one shared thread, serialising all Studio calls —
    each of which waits on Snowflake for up to several seconds.  The handler
    itself stays synchronous and runs in the default thread pool, so pages
    that fire several requests at once get them answered concurrently.
    """

    def deco(fn):
        allowed = [m.upper() for m in methods]

        def pooled(request, *args, **kwargs):
            try:
                return _handle(request, *args, **kwargs)
            finally:
                # Worker threads are pooled: don't leave their DB connections open.
                db_connections.close_all()

        def _handle(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return fail('Sign in to use Warehouse Studio.', 401, 'auth')
            if not request.user.is_superuser:
                return fail('Warehouse Studio is limited to platform administrators.', 403, 'forbidden')
            try:
                return fn(request, *args, **kwargs)
            except ApiError as exc:
                return fail(str(exc), exc.status, exc.code, **exc.extra)
            except sql_guard.SqlGuardError as exc:
                return fail(str(exc), 400, exc.code, statement=exc.statement[:500])
            except executor.QueryError as exc:
                return fail(exc.message, 422, 'query', **{k: v for k, v in exc.as_dict().items() if k != 'message'})
            except sources.SourceUnavailable as exc:
                return fail(str(exc), 409, 'toolkit', source=exc.source, toolkit=get_config().toolkit)
            except StudioConnectionError as exc:
                return fail(str(exc), 503, 'connection')
            except (ValueError, catalog.CatalogError) as exc:
                return fail(str(exc), 400, 'invalid')
            except PermissionDenied as exc:
                return fail(str(exc) or 'Not allowed.', 403, 'forbidden')
            except Http404:
                return fail('Not found.', 404, 'not_found')
            except Exception:  # noqa: BLE001
                logger.exception('Warehouse Studio API error')
                return fail('Unexpected server error — see the server log.', 500, 'server')

        @functools.wraps(fn)
        async def view(request, *args, **kwargs):
            if request.method not in allowed:
                return HttpResponseNotAllowed(allowed)
            if isinstance(request, ASGIRequest):
                return await sync_to_async(pooled, thread_sensitive=False)(request, *args, **kwargs)
            # WSGI / test client: same thread as the request (keeps test transactions visible).
            return await sync_to_async(_handle, thread_sensitive=True)(request, *args, **kwargs)

        view.sync_handler = _handle   # for direct calls in tests / management code
        return view
    return deco


def _context(data: dict | None) -> Context:
    try:
        return Context.from_dict(data or {}, default=default_context())
    except ValueError as exc:
        raise ApiError(str(exc)) from exc


def _context_from_query(request) -> Context:
    return _context({k: request.GET.get(k) for k in ('role', 'warehouse', 'database', 'schema') if request.GET.get(k)})


def _int(value, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(int(value), hi))
    except (TypeError, ValueError):
        return default


def _ip(request) -> str | None:
    return request.META.get('REMOTE_ADDR')


# ============================================================================ session

@studio_api()
def session_view(request):
    return ok(session_svc.info(_context_from_query(request), refresh=request.GET.get('refresh') == '1'))


# ============================================================================ queries

@studio_api(['POST'])
def query_split(request):
    data = body(request)
    sql = str(data.get('sql') or '')
    return ok({'statements': [{'text': s.text, 'start': s.start, 'end': s.end, 'keyword': s.keyword}
                              for s in sql_guard.split_statements(sql)]})


@studio_api(['POST'])
def query_submit(request):
    data = body(request)
    sql = str(data.get('sql') or '')
    ctx = _context(data.get('context'))
    mode = sql_guard.WRITE if data.get('mode') == 'write' else sql_guard.READ
    cfg = get_config()
    worksheet = None
    if data.get('worksheet_id'):
        worksheet = Worksheet.objects.filter(pk=data['worksheet_id'], owner=request.user).first()
    try:
        checked = sql_guard.check(sql, mode=mode, allow_writes=cfg.allow_writes,
                                  callable_procedures=optimizer.callable_procedures(),
                                  database=ctx.database, schema=ctx.schema)
    except sql_guard.SqlGuardError as exc:
        QueryRun.objects.create(user=request.user, sql=sql[:100_000], context=ctx.as_dict(), status='BLOCKED',
                                error=str(exc), worksheet=worksheet, finished_at=timezone.now())
        raise
    statement, verdict = checked[0]
    run = QueryRun.objects.create(user=request.user, sql=statement.text[:100_000], context=ctx.as_dict(),
                                  worksheet=worksheet, kind=str(data.get('kind') or 'worksheet')[:20])
    try:
        started = executor.submit(statement.text, context=ctx,
                                  tag=executor.statement_tag('worksheet', request.user.get_username()))
    except executor.QueryError as exc:
        run.status, run.error, run.query_id = 'FAILED', exc.message, exc.query_id or ''
        run.finished_at = timezone.now()
        run.save()
        raise
    run.query_id, run.status = started['query_id'], 'RUNNING'
    run.save(update_fields=['query_id', 'status'])
    AuditLog.log(user=request.user, action='query', resource='Warehouse Studio',
                 detail=f'{verdict.kind} {verdict.keyword} {started["query_id"]}', ip_address=_ip(request))
    return ok({**started, 'run_id': run.id, 'statement': statement.text, 'kind': verdict.kind,
               'keyword': verdict.keyword}, status=202)


def _finish_run(request, query_id: str, *, status: str, rows=None, error: str = ''):
    run = QueryRun.objects.filter(user=request.user, query_id=query_id).order_by('-created_at').first()
    if run and run.status in ('SUBMITTED', 'RUNNING'):
        run.status = status
        run.rows = rows
        run.error = error[:5000]
        run.finished_at = timezone.now()
        run.elapsed_ms = int((run.finished_at - run.created_at).total_seconds() * 1000)
        run.save()


@studio_api()
def query_status(request, query_id):
    st = executor.status(query_id)
    if st['done']:
        if st['ok']:
            _finish_run(request, st['query_id'], status='SUCCESS')
        else:
            _finish_run(request, st['query_id'], status='CANCELLED' if st['status'] == 'ABORTED' else 'FAILED',
                        error=(st.get('error') or {}).get('message', ''))
    return ok(st)


@studio_api()
def query_results(request, query_id):
    res = executor.results(query_id, offset=_int(request.GET.get('offset'), 0, 0, 10_000_000),
                           limit=_int(request.GET.get('limit'), 500, 1, 10_000))
    run = QueryRun.objects.filter(user=request.user, query_id=res['query_id']).order_by('-created_at').first()
    if run and run.rows is None:
        run.rows = res.get('total') if res.get('total') is not None else res.get('fetched')
        if run.status in ('SUBMITTED', 'RUNNING'):
            run.status, run.finished_at = 'SUCCESS', timezone.now()
        run.save()
    return ok(res)


@studio_api(['POST'])
def query_cancel(request, query_id):
    res = executor.cancel(query_id)
    _finish_run(request, executor.valid_query_id(query_id), status='CANCELLED', error='Cancelled by user')
    return ok(res)


EXPORT_TYPES = {'csv': 'text/csv', 'tsv': 'text/tab-separated-values', 'json': 'application/json',
                'ndjson': 'application/x-ndjson'}


@studio_api()
def query_export(request, query_id):
    fmt = request.GET.get('format', 'csv')
    if fmt not in EXPORT_TYPES:
        raise ApiError('format must be csv, tsv, json or ndjson.')
    query_id = executor.valid_query_id(query_id)
    gen = executor.iter_export(query_id)
    first = next(gen, None)   # surface errors before the streaming response starts

    def stream():
        if first is None:
            if fmt == 'json':
                yield '[]'
            return
        columns, chunk = first
        names = [c['name'] for c in columns]
        if fmt in ('csv', 'tsv'):
            buf = io.StringIO()
            csv.writer(buf, delimiter='\t' if fmt == 'tsv' else ',').writerow(names)
            yield buf.getvalue()
        elif fmt == 'json':
            yield '['
        out, started = _encode(chunk, columns, names, fmt, False)
        yield out
        for _, rows in gen:
            out, started = _encode(rows, columns, names, fmt, started)
            yield out
        if fmt == 'json':
            yield ']'

    resp = StreamingHttpResponse(stream(), content_type=EXPORT_TYPES[fmt])
    resp['Content-Disposition'] = f'attachment; filename="query-{query_id[:8]}.{fmt}"'
    AuditLog.log(user=request.user, action='export', resource='Warehouse Studio', detail=f'{fmt} {query_id}',
                 ip_address=_ip(request))
    return resp


def _encode(rows, columns, names, fmt, started):
    from .services import serialization as ser
    if fmt in ('csv', 'tsv'):
        buf = io.StringIO()
        writer = csv.writer(buf, delimiter='\t' if fmt == 'tsv' else ',')
        for r in rows:
            writer.writerow(['' if v is None else (json.dumps(v, default=str) if isinstance(v, (dict, list)) else v)
                             for v in (ser.plain(x) if columns[i]['family'] == 'json' else ser.to_json_value(x, columns[i]['family'])
                                       for i, x in enumerate(r))])
        return buf.getvalue(), started
    parts = []
    for r in rows:
        rec = {names[i]: (ser.plain(x) if columns[i]['family'] == 'json' else ser.to_json_value(x, columns[i]['family']))
               for i, x in enumerate(r)}
        line = json.dumps(rec, default=str)
        if fmt == 'json':
            parts.append(('' if not started else ',') + line)
            started = True
        else:
            parts.append(line + '\n')
    return ''.join(parts), started


@studio_api()
def query_details(request, query_id):
    query_id = executor.valid_query_id(query_id)
    q = diagnostics.load_query(query_id)
    if not q:
        raise ApiError('Snowflake has no history for that query yet (it can take a few seconds) or the role cannot see it.',
                       404, 'not_found')
    operators = []
    profile_error = None
    try:
        operators = diagnostics.load_profile(query_id)['nodes']
        ops_for_rules = [{'operator_type': n['type'], 'operator_id': n['operator_id'], 'operator_statistics': n['stats']}
                         for n in operators]
    except Exception as exc:  # noqa: BLE001 — profile is optional (older than 14 days, no privilege, …)
        ops_for_rules = []
        profile_error = getattr(exc, 'message', None) or str(exc)
    credits = diagnostics.attributed_credits(query_id)
    return ok({'query': q, 'insights': diagnostics.diagnose(q, ops_for_rules),
               'cost': diagnostics.query_cost(q, credits), 'profile_available': bool(operators),
               'profile_error': profile_error})


@studio_api()
def query_profile(request, query_id):
    return ok(diagnostics.load_profile(query_id))


@studio_api(['POST'])
def query_explain(request):
    data = body(request)
    sql = str(data.get('sql') or '')
    ctx = _context(data.get('context'))
    (statement, verdict), = sql_guard.check(sql, mode=sql_guard.READ, allow_writes=False,
                                           callable_procedures=frozenset(), database=ctx.database, schema=ctx.schema)
    if verdict.keyword not in ('SELECT', 'WITH', 'VALUES'):
        raise ApiError('Cost preview works for SELECT statements.', code='explain')
    warehouses = []
    try:
        warehouses = cost.list_warehouses()
    except Exception:  # noqa: BLE001
        pass
    return ok(diagnostics.preflight(statement.text, context=ctx, warehouses=warehouses))


@studio_api()
def history_view(request):
    return ok(history.search(request.GET.dict()))


@studio_api()
def my_runs(request):
    limit = _int(request.GET.get('limit'), 100, 1, 1000)
    runs = QueryRun.objects.filter(user=request.user)
    if request.GET.get('worksheet'):
        runs = runs.filter(worksheet_id=request.GET['worksheet'])
    return ok({'runs': [r.as_dict() for r in runs[:limit]]})


# ============================================================================ worksheets

@studio_api(['GET', 'POST'])
def worksheets(request):
    if request.method == 'POST':
        data = body(request)
        ws = Worksheet.objects.create(owner=request.user, title=str(data.get('title') or 'Untitled worksheet')[:200],
                                      sql=str(data.get('sql') or ''), context=_context(data.get('context')).as_dict(),
                                      folder=str(data.get('folder') or '')[:120])
        return ok(ws.as_dict(), status=201)
    mine = Worksheet.objects.filter(owner=request.user)
    shared = Worksheet.objects.filter(shared=True).exclude(owner=request.user)
    return ok({'worksheets': [w.as_dict(include_sql=False) for w in mine],
               'shared': [w.as_dict(include_sql=False) for w in shared[:200]]})


@studio_api(['GET', 'PATCH', 'DELETE'])
def worksheet_detail(request, pk):
    ws = get_object_or_404(Worksheet, pk=pk)
    if ws.owner_id != request.user.id and not (ws.shared and request.method == 'GET'):
        raise PermissionDenied('That worksheet belongs to someone else.')
    if request.method == 'DELETE':
        ws.delete()
        return ok({'deleted': True})
    if request.method == 'PATCH':
        data = body(request)
        for field, limit in (('title', 200), ('folder', 120)):
            if field in data:
                setattr(ws, field, str(data[field] or '')[:limit] or ('Untitled worksheet' if field == 'title' else ''))
        if 'sql' in data:
            ws.sql = str(data['sql'] or '')
        if 'context' in data:
            ws.context = _context(data['context']).as_dict()
        for flag in ('starred', 'shared'):
            if flag in data:
                setattr(ws, flag, bool(data[flag]))
        ws.save()
    return ok(ws.as_dict())


# ============================================================================ catalog

@studio_api()
def catalog_databases(request):
    return ok({'databases': catalog.databases(_context_from_query(request), refresh=request.GET.get('refresh') == '1')})


@studio_api()
def catalog_schemas(request):
    return ok({'schemas': catalog.schemas(request.GET.get('database', ''), _context_from_query(request))})


@studio_api()
def catalog_objects(request):
    return ok({'objects': catalog.objects(request.GET.get('database', ''), request.GET.get('schema_name', ''),
                                          _context_from_query(request))})


@studio_api()
def catalog_columns(request):
    return ok({'columns': catalog.columns(request.GET.get('database', ''), request.GET.get('schema_name', ''),
                                          request.GET.get('table') or None, _context_from_query(request))})


@studio_api()
def catalog_ddl(request):
    return ok({'ddl': catalog.ddl(request.GET.get('type', ''), request.GET.get('name', ''), _context_from_query(request))})


@studio_api(['POST'])
def procedure_call(request):
    data = body(request)
    ctx = _context(data.get('context'))
    params = data.get('params') or []
    if not isinstance(params, list):
        raise ApiError('params must be a list.')
    sql, binds = catalog.build_procedure_call(str(data.get('database') or ''), str(data.get('schema') or ''),
                                              str(data.get('name') or ''), params, list(data.get('values') or []))
    fq = '.'.join(str(data.get(k) or '').upper() for k in ('database', 'schema', 'name'))
    cfg = get_config()
    write_ok = data.get('mode') == 'write' and cfg.allow_writes
    if fq not in optimizer.callable_procedures() and not write_ok:
        raise sql_guard.SqlGuardError(
            f'{fq} is not on the Studio allow-list; procedures can change data.'
            + (' Unlock write mode to run it.' if cfg.allow_writes else ' The Studio is read-only on this deployment.'),
            code='call')
    run = QueryRun.objects.create(user=request.user, sql=sql, context=ctx.as_dict(), kind='procedure')
    started = executor.submit(sql, context=ctx, params=binds,
                              tag=executor.statement_tag('procedure', request.user.get_username()))
    run.query_id, run.status = started['query_id'], 'RUNNING'
    run.save(update_fields=['query_id', 'status'])
    AuditLog.log(user=request.user, action='call', resource='Warehouse Studio', detail=f'{fq} {started["query_id"]}',
                 ip_address=_ip(request))
    return ok({**started, 'sql': sql}, status=202)


# ============================================================================ cost

@studio_api()
def cost_overview(request):
    return ok(cost.overview(request.GET.get('days', 30)))


@studio_api()
def cost_heatmap(request):
    return ok(cost.heatmap(request.GET.get('days', 30)))


@studio_api()
def cost_warehouses(request):
    return ok(cost.warehouses(request.GET.get('days', 30)))


@studio_api()
def cost_attribution(request):
    return ok(cost.attribution(request.GET.get('days', 30), _int(request.GET.get('limit'), 25, 5, 100)))


@studio_api()
def cost_top_queries(request):
    return ok(cost.top_queries(request.GET.get('days', 7), request.GET.get('by', 'credits'),
                               _int(request.GET.get('limit'), 50, 5, 200)))


@studio_api()
def cost_storage(request):
    return ok(cost.storage(_int(request.GET.get('limit'), 50, 5, 500)))


@studio_api()
def cost_monitors(request):
    return ok({'monitors': cost.resource_monitors()})


@studio_api(['POST'])
def cost_refresh(request):
    cost.clear_cache()
    return ok({'refreshed': True})


# ============================================================================ optimizer

@studio_api()
def optimizer_catalog(request):
    state = sources.probe(force=request.GET.get('refresh') == '1')
    installed = state.get('toolkit_procedures', {})
    def entry(p):
        return {**p.as_dict(), 'installed': bool(installed.get(p.procedure))}
    last = {}
    for run in DetectorRun.objects.order_by('-created_at')[:300]:
        last.setdefault(run.key, run.as_dict())
    return ok({'toolkit': state, 'detectors': [entry(p) for p in optimizer.DETECTORS],
               'reports': [entry(p) for p in optimizer.REPORTS], 'last_runs': last,
               'credit_price_usd': get_config().credit_price_usd,
               'install_command': f'python manage.py warehouse_studio_install --env-file <admin.env> '
                                  f'--database {get_config().toolkit_database} --schema {get_config().toolkit_schema}'})


@studio_api(['POST'])
def optimizer_run(request):
    data = body(request)
    key = str(data.get('key') or '')
    started = optimizer.start(key, data.get('params') or {}, user=request.user.get_username())
    run = DetectorRun.objects.create(user=request.user, key=key, params=started['params'], query_id=started['query_id'])
    QueryRun.objects.create(user=request.user, sql=started['sql'], query_id=started['query_id'], kind='optimizer',
                            status='RUNNING', context=started['context'])
    AuditLog.log(user=request.user, action='optimizer', resource='Warehouse Studio',
                 detail=f'{key} {started["query_id"]}', ip_address=_ip(request))
    return ok({'run': run.as_dict(), 'sql': started['sql']}, status=202)


@studio_api()
def optimizer_run_detail(request, run_id):
    run = get_object_or_404(DetectorRun, pk=run_id)
    if run.status == 'RUNNING':
        st = executor.status(run.query_id)
        if st['done']:
            run.finished_at = timezone.now()
            if st['ok']:
                proc = optimizer.get(run.key)
                res = executor.results(run.query_id, limit=10_000)
                run.outcome = optimizer.parse_outcome(proc, res)
                run.status = 'SUCCESS'
                _finish_run(request, run.query_id, status='SUCCESS', rows=res.get('total'))
                if proc.kind in ('detector', 'batch'):
                    cost.clear_cache()
            else:
                run.status = 'FAILED'
                run.error = (st.get('error') or {}).get('message', 'Failed')
                _finish_run(request, run.query_id, status='FAILED', error=run.error)
            run.save()
        else:
            return ok({**run.as_dict(), 'live_status': st['status']})
    return ok(run.as_dict())


@studio_api()
def optimizer_runs(request):
    runs = DetectorRun.objects.all()
    if request.GET.get('key'):
        runs = runs.filter(key=request.GET['key'])
    return ok({'runs': [r.as_dict() for r in runs[:_int(request.GET.get('limit'), 50, 1, 500)]]})


def _attach_triage(rows: list[dict]) -> list[dict]:
    ids = [r['finding_id'] for r in rows if r.get('finding_id') is not None]
    triage = {t.finding_id: t.as_dict() for t in FindingTriage.objects.filter(finding_id__in=ids)}
    for r in rows:
        r['triage'] = triage.get(r.get('finding_id'))
    return rows


@studio_api()
def optimizer_findings(request):
    params = {k: (request.GET.getlist(k) if len(request.GET.getlist(k)) > 1 else request.GET.get(k)) for k in request.GET}
    status_filter = params.pop('triage', None)
    if status_filter:
        statuses = status_filter if isinstance(status_filter, list) else str(status_filter).split(',')
        ids = list(FindingTriage.objects.filter(status__in=statuses).values_list('finding_id', flat=True))
        if 'open' in statuses:
            # "open" = no triage record or explicitly open; filter after fetch.
            params['triage_open'] = True
        elif not ids:
            return ok({'findings': [], 'total': 0, 'limit': 0, 'offset': 0})
        else:
            params['ids'] = ids
    data = optimizer.findings(params)
    _attach_triage(data['findings'])
    if params.get('triage_open'):
        wanted = set(status_filter if isinstance(status_filter, list) else str(status_filter).split(','))
        data['findings'] = [f for f in data['findings'] if (f['triage'] or {}).get('status', 'open') in wanted]
    return ok(data)


@studio_api()
def optimizer_summary(request):
    data = optimizer.summary(request.GET.dict())
    counts = {}
    for t in FindingTriage.objects.values_list('status', flat=True):
        counts[t] = counts.get(t, 0) + 1
    data['triage'] = counts
    return ok(data)


@studio_api()
def optimizer_batches(request):
    return ok({'runs': optimizer.recent_runs(_int(request.GET.get('limit'), 60, 1, 500))})


@studio_api(['GET', 'POST'])
def optimizer_schedule(request):
    action = 'STATUS' if request.method == 'GET' else str(body(request).get('action') or 'STATUS')
    result = optimizer.schedule(action)
    if request.method == 'POST':
        AuditLog.log(user=request.user, action='schedule', resource='Warehouse Studio', detail=action,
                     ip_address=_ip(request))
    return ok(result)


@studio_api(['POST'])
def optimizer_triage(request, finding_id):
    data = body(request)
    status = str(data.get('status') or 'open')
    if status not in FindingTriage.STATUSES:
        raise ApiError(f'status must be one of {", ".join(FindingTriage.STATUSES)}.')
    triage, _ = FindingTriage.objects.get_or_create(finding_id=int(finding_id))
    triage.status = status
    if 'note' in data:
        triage.note = str(data.get('note') or '')[:5000]
    if 'assignee' in data:
        triage.assignee = str(data.get('assignee') or '')[:150]
    triage.updated_by = request.user
    triage.save()
    return ok(triage.as_dict())


@studio_api(['POST'])
def optimizer_triage_bulk(request):
    data = body(request)
    status = str(data.get('status') or '')
    if status not in FindingTriage.STATUSES:
        raise ApiError(f'status must be one of {", ".join(FindingTriage.STATUSES)}.')
    ids = [int(i) for i in data.get('ids') or [] if str(i).isdigit()][:1000]
    for fid in ids:
        t, _ = FindingTriage.objects.get_or_create(finding_id=fid)
        t.status, t.updated_by = status, request.user
        if data.get('note'):
            t.note = str(data['note'])[:5000]
        t.save()
    return ok({'updated': len(ids), 'status': status})
