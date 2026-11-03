"""
Browser-facing proxy for the Redash API.

The Analytics Studio single-page app calls ``/studio/proxy/<redash path>``
(e.g. ``/studio/proxy/api/queries``). This view forwards to Redash with the
server-side admin API key, so the key never reaches the browser.

Guard rails:
  * superusers only (session auth; Django CSRF applies to unsafe methods)
  * only routes present in Redash's own route table are forwarded (``routes.py``)
  * every mutating call is written to the AuditLog
"""

import json
import logging
from urllib.parse import quote

from django.core.cache import cache
from django.http import HttpRequest, HttpResponse, JsonResponse

from core.models import AuditLog

from .client import RedashError, get_client
from .models import QueryRevision
from .routes import match_route

logger = logging.getLogger(__name__)

AUDIT_ACTIONS = {'POST': 'update', 'DELETE': 'delete'}
KEY_OWNER_CACHE_KEY = 'redash_ui:key_owner'
CREATE_TEMPLATES = {
    '/api/queries', '/api/dashboards', '/api/alerts', '/api/visualizations', '/api/widgets',
    '/api/users', '/api/groups', '/api/data_sources', '/api/destinations', '/api/query_snippets',
}
EXECUTE_TEMPLATES = {'/api/query_results', '/api/queries/<query_id>/results', '/api/queries/<query_id>/refresh'}


def _error(status: int, message: str) -> JsonResponse:
    # Redash itself reports errors as {"message": ...}; keep the same shape.
    return JsonResponse({'message': message}, status=status)


def _audit(request: HttpRequest, path: str, template: str, status: int) -> None:
    if template in CREATE_TEMPLATES and request.method == 'POST':
        action = 'create'
    elif template in EXECUTE_TEMPLATES:
        action = 'query'
    else:
        action = AUDIT_ACTIONS.get(request.method, 'update')
    try:
        AuditLog.log(
            user=request.user,
            action=action,
            resource='redash',
            resource_id=path[:100],
            detail=f'{request.method} {path} -> {status}',
            ip_address=request.META.get('REMOTE_ADDR'),
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
        )
    except Exception as exc:  # auditing must never break the request
        logger.warning('Audit log failed for %s %s: %s', request.method, path, exc)


def _record_revision(request: HttpRequest, upstream) -> None:
    """Snapshot a query after a successful create/save, unless the text is unchanged."""
    try:
        sent = json.loads(request.body or b'{}')
        if not any(k in sent for k in ('query', 'name', 'description', 'data_source_id')):
            return  # e.g. tag, schedule or archive toggles
        saved = upstream.json() or {}
        if not saved.get('id'):
            return
        last = QueryRevision.objects.filter(query_id=saved['id']).first()
        if last and last.query_text == (saved.get('query') or '') and last.name == (saved.get('name') or ''):
            return
        QueryRevision.objects.create(
            query_id=saved['id'],
            version=saved.get('version'),
            name=(saved.get('name') or '')[:255],
            query_text=saved.get('query') or '',
            description=saved.get('description') or '',
            data_source_id=saved.get('data_source_id'),
            options=saved.get('options') or {},
            user=request.user,
        )
    except Exception as exc:  # history must never break a save
        logger.warning('Could not record query revision: %s', exc)


def query_history(request: HttpRequest, query_id: int) -> JsonResponse:
    """GET /studio/history/<query_id>/ — revisions recorded by Studio, newest first."""
    if not request.user.is_authenticated:
        return _error(401, 'Your session has expired. Please sign in again.')
    if not request.user.is_superuser:
        return _error(403, 'Superuser access is required for Analytics Studio.')
    revisions = QueryRevision.objects.filter(query_id=query_id).select_related('user')[:200]
    return JsonResponse({'query_id': query_id, 'revisions': [r.as_dict() for r in revisions]})


SELF_LOCKOUT_TEMPLATES = {'/api/users/<user_id>/regenerate_api_key', '/api/users/<user_id>/disable'}


def _key_owner_id() -> int | None:
    """Redash user id that owns REDASH_ADMIN_API_KEY (cached; None if unknown)."""
    owner = cache.get(KEY_OWNER_CACHE_KEY)
    if owner is None:
        try:
            owner = get_client().get_json('/api/session').get('user', {}).get('id')
        except RedashError:
            return None
        cache.set(KEY_OWNER_CACHE_KEY, owner, 3600)
    return owner


def _would_lock_out(path: str, template: str, method: str) -> bool:
    """Regenerating or disabling the key owner would break this proxy for everyone."""
    if template not in SELF_LOCKOUT_TEMPLATES or method != 'POST':
        return False
    user_id = path.split('/')[3]
    return user_id.isdigit() and int(user_id) == _key_owner_id()


def redash_proxy(request: HttpRequest, subpath: str) -> HttpResponse:
    """Forward ``/studio/proxy/<subpath>`` to ``{REDASH_BASE_URL}/<subpath>``."""
    if not request.user.is_authenticated:
        return _error(401, 'Your session has expired. Please sign in again.')
    if not request.user.is_superuser:
        return _error(403, 'Superuser access is required for Analytics Studio.')

    # Re-encode each decoded segment; keep '.' so "results.csv" stays a file type.
    path = '/' + '/'.join(quote(s, safe='.~') for s in subpath.split('/') if s)
    template, allowed = match_route(path, request.method)
    if template is None:
        return _error(404, f'Unknown Redash API route: {path}')
    if not allowed:
        return _error(405, f'{request.method} is not supported on {template}')
    if _would_lock_out(path, template, request.method):
        return _error(409, 'This user owns the API key Analytics Studio uses to talk to Redash. '
                           'Regenerating its key or disabling it would disconnect Studio for everyone — '
                           'do it from Redash itself and update REDASH_ADMIN_API_KEY afterwards.')

    body = None
    if request.method == 'POST' and request.body:
        try:
            json.loads(request.body)
        except ValueError:
            return _error(400, 'Request body must be valid JSON.')
        body = request.body

    params = [(k, v) for k, values in request.GET.lists() for v in values] or None
    try:
        upstream = get_client().request(request.method, path, params=params, data=body)
    except RedashError as exc:
        logger.error('Redash proxy %s %s failed: %s', request.method, path, exc)
        return JsonResponse(exc.payload, status=exc.status_code)

    if request.method != 'GET':
        _audit(request, path, template, upstream.status_code)
        if request.method == 'POST' and upstream.ok and template in ('/api/queries', '/api/queries/<query_id>'):
            _record_revision(request, upstream)

    response = HttpResponse(upstream.content, status=upstream.status_code, content_type=upstream.content_type)
    for key, value in upstream.headers.items():
        response[key] = value
    response['Cache-Control'] = 'no-store'
    return response
