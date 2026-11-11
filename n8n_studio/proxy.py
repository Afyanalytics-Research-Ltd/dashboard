"""
Browser-facing proxy for the n8n public API.

The Automation Studio single-page app calls ``/automations/api/<path>`` (e.g.
``/automations/api/workflows?limit=50``); this view forwards it to
``{N8N_API_URL}/api/v1/<path>`` with the server-side API key, so the key never
reaches the browser.

Guard rails:
  * superusers only (session auth; Django CSRF applies to unsafe methods)
  * only routes in n8n's own OpenAPI spec are forwarded (``routes.py``)
  * JSON bodies are validated; multipart is only accepted for package import
  * every mutating call is written to the AuditLog
  * workflow creates/saves are snapshotted, and workflows are snapshotted
    before they are deleted, so nothing is lost beyond n8n's history window
"""

import json
import logging
from urllib.parse import quote

from django.http import HttpRequest, HttpResponse, JsonResponse

from core.models import AuditLog

from .client import N8nError, get_client
from .models import WorkflowSnapshot
from .routes import match_route

logger = logging.getLogger(__name__)

MAX_UPLOAD = 50 * 1024 * 1024
MULTIPART_TEMPLATES = {'/n8n-packages/import'}
AUDIT_ACTIONS = {'POST': 'update', 'PUT': 'update', 'PATCH': 'update', 'DELETE': 'delete'}
CREATE_TEMPLATES = {
    '/workflows', '/credentials', '/tags', '/variables', '/users', '/projects', '/data-tables',
    '/projects/{projectId}/folders', '/community-packages', '/roles', '/settings/log-streaming/destinations',
}
EXECUTE_TEMPLATES = {'/executions/{id}/retry', '/workflows/{id}/test-runs'}
WORKFLOW_SAVE_TEMPLATES = {'/workflows', '/workflows/{workflowId}'}


def error(status: int, message: str, code: str = 'studio') -> JsonResponse:
    # n8n reports errors as {"message": ...}; keep the same shape.
    return JsonResponse({'message': message, 'code': code}, status=status)


def guard(request: HttpRequest) -> JsonResponse | None:
    if not request.user.is_authenticated:
        return error(401, 'Your session has expired. Please sign in again.', 'session')
    if not request.user.is_superuser:
        return error(403, 'Superuser access is required for Automation Studio.', 'forbidden')
    return None


def audit(request: HttpRequest, action: str, resource_id: str, detail: str) -> None:
    try:
        AuditLog.log(
            user=request.user, action=action, resource='n8n', resource_id=resource_id[:100], detail=detail[:500],
            ip_address=request.META.get('REMOTE_ADDR'), user_agent=request.META.get('HTTP_USER_AGENT', ''),
        )
    except Exception as exc:  # auditing must never break the request
        logger.warning('Audit log failed for %s: %s', detail, exc)


def _audit_action(method: str, template: str) -> str:
    if template in CREATE_TEMPLATES and method == 'POST':
        return 'create'
    if template in EXECUTE_TEMPLATES:
        return 'query'
    return AUDIT_ACTIONS.get(method, 'update')


def _snapshot_before_delete(request: HttpRequest, workflow_id: str) -> None:
    try:
        resp = get_client().request('GET', f'/workflows/{quote(workflow_id, safe="")}')
        if resp.ok:
            WorkflowSnapshot.capture(resp.json() or {}, source='deleted', user=request.user, force=True,
                                     note='Captured automatically before the workflow was deleted')
    except Exception as exc:  # never block a delete on the safety net
        logger.warning('Could not snapshot workflow %s before delete: %s', workflow_id, exc)


def _snapshot_after_save(request: HttpRequest, template: str, upstream) -> None:
    try:
        saved = upstream.json() or {}
        source = request.headers.get('X-Studio-Snapshot') or ('create' if template == '/workflows' else 'save')
        if source not in dict(WorkflowSnapshot.SOURCES):
            source = 'save'
        WorkflowSnapshot.capture(saved, source=source, user=request.user,
                                 note=(request.headers.get('X-Studio-Note') or '')[:255])
    except Exception as exc:  # history must never break a save
        logger.warning('Could not snapshot saved workflow: %s', exc)


def n8n_proxy(request: HttpRequest, subpath: str) -> HttpResponse:
    """Forward ``/automations/api/<subpath>`` to ``{N8N_API_URL}/api/v1/<subpath>``."""
    denied = guard(request)
    if denied:
        return denied

    # Re-encode each decoded segment so ids/names with odd characters stay one segment.
    path = '/' + '/'.join(quote(s, safe='@:~') for s in subpath.split('/') if s)
    if any(seg in ('.', '..') for seg in path.split('/')):
        return error(400, 'Invalid path.')
    template, allowed = match_route(path, request.method)
    if template is None:
        return error(404, f'Unknown n8n API route: {path}', 'unknown_route')
    if not allowed:
        return error(405, f'{request.method} is not supported on {template}', 'method')

    kwargs = {}
    if template in MULTIPART_TEMPLATES and request.method == 'POST':
        upload = request.FILES.get('package')
        if not upload:
            return error(400, 'Attach the .n8np package file as "package".')
        if upload.size > MAX_UPLOAD:
            return error(413, 'Package is larger than 50 MB.')
        kwargs['files'] = {'package': (upload.name, upload.read(), upload.content_type or 'application/gzip')}
        kwargs['data'] = {k: v for k, v in request.POST.items()}
    elif request.method in ('POST', 'PUT', 'PATCH', 'DELETE') and request.body:
        try:
            json.loads(request.body)
        except ValueError:
            return error(400, 'Request body must be valid JSON.')
        kwargs['data'] = request.body

    if request.method == 'DELETE' and template == '/workflows/{workflowId}':
        _snapshot_before_delete(request, path.split('/')[2])

    params = [(k, v) for k, values in request.GET.lists() for v in values] or None
    try:
        upstream = get_client().request(request.method, path, params=params, **kwargs)
    except N8nError as exc:
        logger.error('n8n proxy %s %s failed: %s', request.method, path, exc)
        return JsonResponse(exc.payload, status=exc.status_code)

    if request.method != 'GET':
        audit(request, _audit_action(request.method, template), path, f'{request.method} {path} -> {upstream.status_code}')
        if upstream.ok and template in WORKFLOW_SAVE_TEMPLATES and request.method in ('POST', 'PUT'):
            _snapshot_after_save(request, template, upstream)

    response = HttpResponse(upstream.content, status=upstream.status_code, content_type=upstream.content_type)
    for key, value in upstream.headers.items():
        response[key] = value
    response['Cache-Control'] = 'no-store'
    return response
