"""
Browser-facing endpoints for Model Studio.

  POST /models/graphql       -> W&B /graphql (allowlisted operations only)
  GET  /models/file?url=...  -> W&B file downloads (only URLs on the W&B host)
  /models/views/...          -> saved workspace views (stored in Django)

All superuser-only; Django CSRF applies to unsafe methods; every mutation is
written to the AuditLog.
"""

import base64
import json
import logging
from urllib.parse import urlsplit, urlunsplit

from django.conf import settings
from django.core.cache import cache
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from core.models import AuditLog

from .client import WandbError, get_client
from .gql import check_document, mutation_inputs
from .models import SavedView

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 2 * 1024 * 1024


def _error(status: int, message: str) -> JsonResponse:
    # Same shape as GraphQL errors so the frontend has one error path.
    return JsonResponse({'errors': [{'message': message}]}, status=status)


def _superuser(view):
    def wrapped(request: HttpRequest, *args, **kwargs):
        if not request.user.is_authenticated:
            return _error(401, 'Your session has expired. Please sign in again.')
        if not request.user.is_superuser:
            return _error(403, 'Superuser access is required for Model Studio.')
        return view(request, *args, **kwargs)
    wrapped.__name__ = view.__name__
    wrapped.__doc__ = view.__doc__
    return wrapped


def _audit(request: HttpRequest, action: str, resource_id: str, detail: str) -> None:
    try:
        AuditLog.log(user=request.user, action=action, resource='wandb', resource_id=resource_id[:100],
                     detail=detail, ip_address=request.META.get('REMOTE_ADDR'),
                     user_agent=request.META.get('HTTP_USER_AGENT', ''))
    except Exception as exc:  # auditing must never break the request
        logger.warning('Audit log failed: %s', exc)


KEY_OWNER_CACHE = 'wandb_ui:key_owner'
LOCKOUT_MESSAGE = ('This would disconnect Model Studio: the W&B user and API key in WANDB_API_KEY are what Studio uses '
                   'to talk to W&B. Do it from W&B itself and update .env afterwards.')


def public_key_id(key: str) -> str:
    """
    Identifier of a W&B API key for server-side comparison. New-style keys are
    '<prefix>_v1_<id>_<secret>' (the id is everything before the last '_');
    classic 40-char keys are their own id — so this is SECRET for classic keys.
    """
    head, sep, _ = key.rpartition('_')
    return head if sep and '_v1_' in key else key


def displayable_key_id(key: str) -> str:
    """The key id only when it is not the secret itself (new-style keys); '' otherwise."""
    pid = public_key_id(key or '')
    return pid if pid and pid != key else ''


def _mentions(node_id: str, public_id: str) -> bool:
    """W&B node ids are (nested) base64 that embed the key's public id."""
    seen = node_id or ''
    for _ in range(3):
        if public_id in seen:
            return True
        try:
            padded = seen + '=' * (-len(seen) % 4)
            seen = base64.urlsafe_b64decode(padded.replace('+', '-').replace('/', '_')).decode('latin-1')
        except (ValueError, TypeError):
            return False
        seen = seen.split(':', 1)[-1] if ':' in seen[:40] else seen
    return public_id in seen


def _key_owner() -> dict | None:
    """{'user_id', 'public_id'} for the configured key. Only viewer.id is asked of W&B
    (viewer.apiKeys needs a recent web-UI login, so it can't be relied on)."""
    owner = cache.get(KEY_OWNER_CACHE)
    if owner is None:
        try:
            data = get_client().query('{ viewer { id } }')
        except WandbError:
            return None
        owner = {'user_id': (data.get('viewer') or {}).get('id'), 'public_id': public_key_id(settings.WANDB_API_KEY)}
        cache.set(KEY_OWNER_CACHE, owner, 3600)
    return owner


def _is_key_in_use(ref: dict, public_id: str) -> bool:
    return any(v and (v == public_id or _mentions(str(v), public_id)) for v in (ref.get('id'), ref.get('apiKeyId')))


def _lockout_reason(document: str, variables: dict | None) -> str | None:
    """Refuse mutations that would revoke Studio's own access (key, user, admin flag)."""
    risky = {'deleteApiKey', 'deleteApiKeys', 'deleteUser', 'updateUser', 'updateUserRole'}
    calls = [(name, args) for name, args in mutation_inputs(document, variables) if name in risky]
    if not calls:
        return None
    owner = _key_owner()
    if not owner:
        return 'Could not verify that this change is safe (W&B did not answer); try again in a moment.'
    user_id, public_id = owner['user_id'], owner['public_id']
    for name, args in calls:
        inp = args.get('input') if isinstance(args.get('input'), dict) else {}
        if name == 'deleteApiKey' and _is_key_in_use(inp, public_id):
            return LOCKOUT_MESSAGE
        if name == 'deleteApiKeys' and any(_is_key_in_use(r or {}, public_id) for r in inp.get('keys') or []):
            return LOCKOUT_MESSAGE
        if name in ('deleteUser', 'updateUserRole') and inp.get('id') == user_id:
            return LOCKOUT_MESSAGE
        if name == 'updateUser' and inp.get('admin') is False and inp.get('id') in (user_id, None):
            return LOCKOUT_MESSAGE
    return None


@require_POST
@_superuser
def graphql_proxy(request: HttpRequest) -> HttpResponse:
    if len(request.body) > MAX_BODY_BYTES:
        return _error(413, 'Request is too large.')
    try:
        payload = json.loads(request.body or b'{}')
    except ValueError:
        return _error(400, 'Request body must be JSON: {"query": "...", "variables": {...}}.')
    batch = payload if isinstance(payload, list) else [payload]
    if not batch or len(batch) > 20 or not all(isinstance(p, dict) for p in batch):
        return _error(400, 'Send one GraphQL request object (or a batch of up to 20).')

    mutations = []
    for item in batch:
        checked = check_document(item.get('query'))
        if not checked.ok:
            return _error(403 if 'not available' in checked.error else 400, checked.error)
        mutations.extend(checked.mutations)
        if checked.mutations:
            lockout = _lockout_reason(item.get('query'), item.get('variables'))
            if lockout:
                return _error(409, lockout)

    try:
        upstream = get_client().graphql(request.body, read_only=not mutations)
    except WandbError as exc:
        logger.error('W&B GraphQL proxy failed: %s', exc)
        return JsonResponse(exc.payload, status=exc.status_code)

    if mutations:
        action = 'delete' if any(m.startswith(('delete', 'purge')) for m in mutations) else (
            'create' if any(m.startswith('create') for m in mutations) else 'update')
        _audit(request, action, ','.join(mutations), f'GraphQL {", ".join(mutations)} -> {upstream.status_code}')

    resp = HttpResponse(upstream.content, status=upstream.status_code, content_type=upstream.content_type)
    resp['Cache-Control'] = 'no-store'
    return resp


FILE_PATH_PREFIXES = ('/files/', '/local-files/', '/artifacts/', '/artifactsV2/', '/storage')
LOOPBACK = {'localhost', '127.0.0.1', '::1'}


def _host_key(parts) -> tuple:
    """(host, port) with every loopback name folded together: 127.0.0.1:8090 == localhost:8090."""
    try:
        port = parts.port or (443 if parts.scheme == 'https' else 80)
    except ValueError:
        return ('invalid', None)
    host = (parts.hostname or '').lower()
    return ('loopback' if host in LOOPBACK else host, port)


def _allowed_file_url(url: str) -> str | None:
    """Map a W&B file URL (presigned links included) to the internal base URL; None if it isn't a W&B URL."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme not in ('http', 'https') or not parts.netloc:
        return None
    internal = urlsplit(settings.WANDB_BASE_URL)
    public = urlsplit(getattr(settings, 'WANDB_HOST_URL', '') or settings.WANDB_BASE_URL)
    if _host_key(parts) not in {_host_key(internal), _host_key(public)}:
        return None
    if not parts.path.startswith(FILE_PATH_PREFIXES):
        return None
    return urlunsplit((internal.scheme, internal.netloc, parts.path, parts.query, ''))


@require_GET
@_superuser
def file_proxy(request: HttpRequest) -> HttpResponse:
    """Stream a run/artifact file (images, logs, tables) through the server-side key."""
    target = _allowed_file_url(request.GET.get('url', ''))
    if not target:
        return _error(400, 'Only W&B file URLs can be fetched.')
    try:
        upstream = get_client().get_file(target, range_header=request.headers.get('Range'))
    except WandbError as exc:
        return JsonResponse(exc.payload, status=exc.status_code)
    resp = HttpResponse(upstream.content, status=upstream.status_code, content_type=upstream.content_type)
    for k, v in upstream.headers.items():
        if k.lower() != 'content-length':
            resp[k] = v
    if request.GET.get('download'):
        name = target.rstrip('/').rsplit('/', 1)[-1].split('?')[0] or 'file'
        resp['Content-Disposition'] = f'attachment; filename="{name}"'
    resp['Cache-Control'] = 'private, max-age=300'
    return resp


# ------------------------------------------------------------------ saved views

def _view_payload(request):
    try:
        data = json.loads(request.body or b'{}')
    except ValueError:
        return None, _error(400, 'Body must be JSON.')
    name = str(data.get('name', '')).strip()[:120]
    if not name:
        return None, _error(400, 'A view name is required.')
    if not isinstance(data.get('state', {}), dict):
        return None, _error(400, 'state must be an object.')
    return data, None


@require_http_methods(['GET', 'POST'])
@_superuser
def views_collection(request: HttpRequest, entity: str, project: str) -> JsonResponse:
    if request.method == 'GET':
        qs = SavedView.objects.filter(entity=entity, project=project).select_related('owner')
        qs = [v for v in qs if v.shared or v.owner_id == request.user.id]
        return JsonResponse({'views': [v.as_dict() for v in qs]})
    data, err = _view_payload(request)
    if err:
        return err
    view = SavedView.objects.create(entity=entity, project=project, name=data['name'], state=data.get('state', {}),
                                    shared=bool(data.get('shared', True)), owner=request.user)
    return JsonResponse(view.as_dict(), status=201)


@require_http_methods(['GET', 'POST', 'DELETE'])
@_superuser
def view_detail(request: HttpRequest, view_id: int) -> JsonResponse:
    view = SavedView.objects.filter(id=view_id).select_related('owner').first()
    if not view or (not view.shared and view.owner_id != request.user.id):
        return _error(404, 'Saved view not found.')
    if request.method == 'GET':
        return JsonResponse(view.as_dict())
    if request.method == 'DELETE':
        view.delete()
        return JsonResponse({'deleted': view_id})
    data, err = _view_payload(request)
    if err:
        return err
    view.name = data['name']
    view.state = data.get('state', view.state)
    view.shared = bool(data.get('shared', view.shared))
    view.save()
    return JsonResponse(view.as_dict())
