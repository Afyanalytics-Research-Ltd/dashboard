"""
Studio-only JSON endpoints that sit next to the API proxy.

    GET  health/                      instance, API key, login and catalog status
    GET  catalog/                     node + credential type index (gzip when accepted)
    GET  catalog/nodes/<type>/        full description(s) of one node type
    GET  catalog/credentials/<type>/  full description of one credential type
    GET  icons/<path>                 node/credential icons, cached
    POST run/<workflow_id>/           manual run through the optional n8n login
    POST webhook/                     call a production/test webhook from the server
    GET  snapshots/                   Studio's own workflow history (?workflow=, ?source=)
    GET  snapshots/<id>/              one snapshot with its workflow JSON
    GET  snapshots/deleted/           latest snapshot of every workflow deleted via Studio
    POST snapshots/capture/           back up every workflow now
"""

from __future__ import annotations

import json
import logging
import re
import time

import requests
from django.core.cache import cache
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.views.decorators.http import require_GET, require_POST

from . import catalog as catalog_mod
from . import conf
from .client import N8nError, get_client, get_session
from .models import WorkflowSnapshot
from .proxy import audit, error, guard
from .routes import N8N_SPEC_VERSION, operation_count

logger = logging.getLogger(__name__)

ICON_RE = re.compile(r'^(icons|types)/[\w@.\-/]+\.(svg|png|jpg|jpeg|gif|webp)$', re.I)
WEBHOOK_PATH_RE = re.compile(r'^[\w\-./:{}%@]+$')
WEBHOOK_METHODS = {'GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD'}
MAX_WEBHOOK_BODY = 1024 * 1024


def _json_body(request: HttpRequest) -> dict:
    if not request.body:
        return {}
    try:
        body = json.loads(request.body)
    except ValueError as exc:
        raise ValueError('Request body must be valid JSON.') from exc
    if not isinstance(body, dict):
        raise ValueError('Request body must be a JSON object.')
    return body


# ---------------------------------------------------------------------------- health

@require_GET
def health(request: HttpRequest) -> JsonResponse:
    denied = guard(request)
    if denied:
        return denied
    client = get_client()
    out = {
        'instance': client.health(),
        'apiUrl': conf.api_url(),
        'publicUrl': conf.public_url(),
        'specVersion': N8N_SPEC_VERSION,
        'operations': operation_count(),
        'login': {'configured': conf.login_configured()},
        'key': {'configured': bool(conf.api_key())},
    }
    if conf.api_key():
        t0 = time.monotonic()
        try:
            resp = client.request('GET', '/workflows', params={'limit': 1})
            out['key'].update(ok=resp.ok, status=resp.status_code, ms=round((time.monotonic() - t0) * 1000))
        except N8nError as exc:
            out['key'].update(ok=False, error=str(exc), code=exc.payload.get('code'))
    if conf.login_configured():
        try:
            session = get_session()
            if not session._logged_in:
                session.login()
            out['login']['ok'] = True
        except N8nError as exc:
            out['login'].update(ok=False, error=str(exc))
    cat = catalog_mod.get_catalog()
    out['catalog'] = {'source': cat.source, 'nodes': cat.node_count}
    return JsonResponse(out)


# ---------------------------------------------------------------------------- catalog

@require_GET
def catalog_index(request: HttpRequest) -> HttpResponse:
    denied = guard(request)
    if denied:
        return denied
    cat = catalog_mod.get_catalog(force=request.GET.get('refresh') == '1')
    if 'gzip' in request.headers.get('Accept-Encoding', ''):
        resp = HttpResponse(cat.index_gzip, content_type='application/json')
        resp['Content-Encoding'] = 'gzip'
    else:
        resp = HttpResponse(cat.index_json, content_type='application/json')
    resp['Cache-Control'] = 'private, max-age=600'
    resp['Vary'] = 'Accept-Encoding'
    return resp


@require_GET
def catalog_node(request: HttpRequest, node_type: str) -> HttpResponse:
    denied = guard(request)
    if denied:
        return denied
    raw = catalog_mod.get_catalog().node(node_type)
    if raw is None:
        raw = catalog_mod.bundled().node(node_type)
    if raw is None:
        return error(404, f'Unknown node type {node_type}. It may come from a community package '
                          'Studio has not seen yet — set N8N_STUDIO_EMAIL/PASSWORD for the live catalog.', 'unknown_node')
    resp = HttpResponse(raw, content_type='application/json')
    resp['Cache-Control'] = 'private, max-age=3600'
    return resp


@require_GET
def catalog_credential(request: HttpRequest, cred_type: str) -> HttpResponse:
    denied = guard(request)
    if denied:
        return denied
    raw = catalog_mod.get_catalog().credential(cred_type) or catalog_mod.bundled().credential(cred_type)
    if raw is None:
        return error(404, f'Unknown credential type {cred_type}.', 'unknown_credential')
    resp = HttpResponse(raw, content_type='application/json')
    resp['Cache-Control'] = 'private, max-age=3600'
    return resp


@require_GET
def icon(request: HttpRequest, path: str) -> HttpResponse:
    # Icons are public assets in n8n; still require a Studio user so this is no open proxy.
    denied = guard(request)
    if denied:
        return denied
    if not ICON_RE.match(path) or '..' in path:
        return error(400, 'Not an icon path.')
    key = f'n8n_studio:icon:{path}'
    hit = cache.get(key)
    if hit is None:
        try:
            r = requests.get(f'{conf.api_url()}/{path}', timeout=10)
        except requests.RequestException as exc:
            return error(502, f'Cannot fetch icon: {exc}')
        ctype = r.headers.get('Content-Type', '')
        if r.status_code != 200 or not ctype.startswith('image/'):
            return error(404, 'Icon not found.')
        hit = (r.content, ctype)
        cache.set(key, hit, 86400)
    resp = HttpResponse(hit[0], content_type=hit[1])
    resp['Cache-Control'] = 'private, max-age=86400'
    resp['X-Content-Type-Options'] = 'nosniff'
    resp['Content-Security-Policy'] = "default-src 'none'; style-src 'unsafe-inline'"
    return resp


# ---------------------------------------------------------------------------- manual run

@require_POST
def run_workflow(request: HttpRequest, workflow_id: str) -> JsonResponse:
    """Start a manual execution, like the editor's "Execute workflow" button.

    Body: {triggerName?, destinationNode?, workflowData?}. ``workflowData``
    (nodes/connections/pinData) runs unsaved edits; otherwise the saved workflow runs.
    """
    denied = guard(request)
    if denied:
        return denied
    if not conf.login_configured():
        return error(409, 'Manual runs need an n8n login: set N8N_STUDIO_EMAIL and N8N_STUDIO_PASSWORD '
                          '(the public API cannot start executions). Webhook-triggered workflows can be '
                          'run from the Webhook tab instead.', 'login_not_configured')
    try:
        body = _json_body(request)
    except ValueError as exc:
        return error(400, str(exc))
    try:
        workflow = body.get('workflowData')
        if not isinstance(workflow, dict):
            saved = get_client().request('GET', f'/workflows/{workflow_id}')
            if not saved.ok:
                return JsonResponse(saved.json() or {'message': 'Workflow not found.'}, status=saved.status_code)
            workflow = saved.json()
        workflow = {**workflow, 'id': workflow_id}
        payload: dict = {'workflowData': workflow}
        trigger = body.get('triggerName')
        if not trigger and not body.get('destinationNode'):
            trigger = _first_trigger(workflow)
        if trigger:
            payload['triggerToStartFrom'] = {'name': trigger}
        if body.get('destinationNode'):
            payload['destinationNode'] = {'nodeName': body['destinationNode'], 'mode': 'inclusive'}
        if not trigger and not body.get('destinationNode'):
            return error(400, 'This workflow has no trigger node to start from.', 'no_trigger')
        upstream = get_session().request('POST', f'/rest/workflows/{workflow_id}/run', json=payload)
    except N8nError as exc:
        return JsonResponse(exc.payload, status=exc.status_code)
    audit(request, 'query', f'/workflows/{workflow_id}/run', f'manual run {workflow_id} -> {upstream.status_code}')
    data = upstream.json() if upstream.content else {}
    if isinstance(data, dict) and 'data' in data and upstream.ok:
        data = data['data']
    return JsonResponse(data if isinstance(data, dict) else {'result': data}, status=upstream.status_code)


TRIGGER_HINTS = ('trigger', 'webhook', 'cron', 'schedule', 'formtrigger', 'chattrigger')


def _first_trigger(workflow: dict) -> str | None:
    nodes = [n for n in workflow.get('nodes') or [] if not n.get('disabled')]
    manual = [n for n in nodes if n.get('type', '').endswith('.manualTrigger')]
    if manual:
        return manual[0]['name']
    for n in nodes:
        if any(h in n.get('type', '').lower() for h in TRIGGER_HINTS):
            return n['name']
    return None


# ---------------------------------------------------------------------------- webhook tester

@require_POST
def webhook_call(request: HttpRequest) -> JsonResponse:
    """Call ``{N8N_API_URL}/webhook[-test]/<path>`` server-side (no CORS) and report the response."""
    denied = guard(request)
    if denied:
        return denied
    try:
        body = _json_body(request)
    except ValueError as exc:
        return error(400, str(exc))
    path = str(body.get('path') or '').strip('/')
    method = str(body.get('method') or 'POST').upper()
    if not path or not WEBHOOK_PATH_RE.match(path) or '..' in path:
        return error(400, 'Webhook path is missing or contains invalid characters.')
    if method not in WEBHOOK_METHODS:
        return error(400, f'Unsupported method {method}.')
    prefix = 'webhook-test' if body.get('mode') == 'test' else 'webhook-waiting' if body.get('mode') == 'waiting' else 'webhook'
    headers = {str(k): str(v) for k, v in (body.get('headers') or {}).items() if str(k).lower() not in ('host', 'content-length')}
    payload = body.get('body')
    data = None
    if payload is not None and method not in ('GET', 'HEAD'):
        data = payload if isinstance(payload, str) else json.dumps(payload)
        headers.setdefault('Content-Type', 'application/json')
        if len(data) > MAX_WEBHOOK_BODY:
            return error(413, 'Webhook body is larger than 1 MB.')
    url = f'{conf.api_url()}/{prefix}/{path}'
    t0 = time.monotonic()
    try:
        r = requests.request(method, url, params=body.get('query') or None, data=data, headers=headers,
                             timeout=conf.timeout(), allow_redirects=False)
    except requests.RequestException as exc:
        return error(502, f'Cannot reach n8n webhook: {exc}', 'unreachable')
    ms = round((time.monotonic() - t0) * 1000)
    audit(request, 'query', f'/{prefix}/{path}'[:100], f'{method} /{prefix}/{path} -> {r.status_code}')
    ctype = r.headers.get('Content-Type', '')
    text = r.content[:MAX_WEBHOOK_BODY].decode('utf-8', errors='replace')
    parsed = None
    if 'json' in ctype:
        try:
            parsed = json.loads(text)
        except ValueError:
            pass
    return JsonResponse({
        'url': f'{conf.webhook_url()}/{prefix}/{path}',
        'status': r.status_code,
        'ms': ms,
        'headers': dict(r.headers),
        'contentType': ctype,
        'json': parsed,
        'text': None if parsed is not None else text,
        'truncated': len(r.content) > MAX_WEBHOOK_BODY,
    })


# ---------------------------------------------------------------------------- snapshots

@require_GET
def snapshots(request: HttpRequest) -> JsonResponse:
    denied = guard(request)
    if denied:
        return denied
    qs = WorkflowSnapshot.objects.select_related('user')
    if request.GET.get('workflow'):
        qs = qs.filter(workflow_id=request.GET['workflow'])
    if request.GET.get('source'):
        qs = qs.filter(source=request.GET['source'])
    try:
        limit = max(1, min(int(request.GET.get('limit', 200)), 1000))
    except ValueError:
        limit = 200
    total = qs.count()
    return JsonResponse({'total': total, 'data': [s.as_dict() for s in qs[:limit]]})


@require_GET
def snapshot_detail(request: HttpRequest, pk: int) -> JsonResponse:
    denied = guard(request)
    if denied:
        return denied
    snap = WorkflowSnapshot.objects.select_related('user').filter(pk=pk).first()
    if not snap:
        return error(404, 'Snapshot not found.')
    return JsonResponse(snap.as_dict(with_data=True))


@require_GET
def snapshots_deleted(request: HttpRequest) -> JsonResponse:
    """Workflows deleted through Studio whose id no longer exists in n8n (the recycle bin)."""
    denied = guard(request)
    if denied:
        return denied
    latest = {}
    for snap in WorkflowSnapshot.objects.filter(source='deleted').select_related('user')[:2000]:
        latest.setdefault(snap.workflow_id, snap)
    alive = set()
    if latest and request.GET.get('check', '1') == '1':
        try:
            alive = {str(w['id']) for w in get_client().iterate('/workflows', excludePinnedData='true')}
        except N8nError as exc:
            return JsonResponse(exc.payload, status=exc.status_code)
    return JsonResponse({'data': [s.as_dict() for wid, s in latest.items() if wid not in alive]})


@require_POST
def snapshots_capture(request: HttpRequest) -> JsonResponse:
    denied = guard(request)
    if denied:
        return denied
    try:
        result = capture_all(user=request.user, source='manual')
    except N8nError as exc:
        return JsonResponse(exc.payload, status=exc.status_code)
    audit(request, 'create', 'snapshots', f'backup: {result["captured"]} captured of {result["workflows"]}')
    return JsonResponse(result)


def capture_all(*, user=None, source: str = 'backup') -> dict:
    client = get_client()
    captured = total = 0
    for summary in client.iterate('/workflows'):
        total += 1
        wf = summary if isinstance(summary.get('nodes'), list) else client.get_json(f'/workflows/{summary["id"]}')
        if WorkflowSnapshot.capture(wf, source=source, user=user):
            captured += 1
    return {'workflows': total, 'captured': captured, 'unchanged': total - captured}
