"""
Browser-facing proxy for the Airflow REST API v2.

The pipeline UI is a single-page app that calls ``/pipelines/api/v2/<path>``.
This view forwards those calls to Airflow with the server-side JWT so the
Airflow credentials never reach the browser.

Guard rails:
  * superusers only (session auth; Django CSRF applies to unsafe methods)
  * only routes present in the official spec are forwarded (``routes.py``)
  * every mutating call is written to the AuditLog
"""

import json
import logging

import requests
from django.conf import settings
from django.core.cache import cache
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.views.decorators.http import require_GET

from core.models import AuditLog

from .client import AirflowError, encode_segment, get_client
from .routes import match_route

logger = logging.getLogger(__name__)

SPEC_CACHE_KEY = 'airflow_ui:openapi'
SPEC_TTL_SECONDS = 60 * 60
PASSTHROUGH_ACCEPT = ('application/json', 'text/plain')
AUDIT_ACTIONS = {'POST': 'create', 'PUT': 'update', 'PATCH': 'update', 'DELETE': 'delete'}


def _error(status: int, detail: str, **extra) -> JsonResponse:
    return JsonResponse({'detail': detail, **extra}, status=status)


def _superuser_only(view):
    """JSON 401/403 instead of login redirects — the caller is fetch()."""
    def wrapped(request: HttpRequest, *args, **kwargs):
        if not request.user.is_authenticated:
            return _error(401, 'Your session has expired. Please sign in again.')
        if not request.user.is_superuser:
            return _error(403, 'Superuser access is required for pipeline management.')
        return view(request, *args, **kwargs)
    wrapped.__name__ = view.__name__
    wrapped.__doc__ = view.__doc__
    return wrapped


def _audit(request: HttpRequest, path: str, template: str, status: int) -> None:
    action = AUDIT_ACTIONS.get(request.method, 'update')
    if request.method == 'POST' and template.endswith('/dagRuns'):
        action = 'trigger'
    try:
        AuditLog.log(
            user=request.user,
            action=action,
            resource='airflow',
            resource_id=path[:100],
            detail=f'{request.method} /api/v2{path} -> {status}',
            ip_address=request.META.get('REMOTE_ADDR'),
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
        )
    except Exception as exc:  # auditing must never break the request
        logger.warning('Audit log failed for %s %s: %s', request.method, path, exc)


@_superuser_only
def api_proxy(request: HttpRequest, subpath: str) -> HttpResponse:
    """Forward ``/pipelines/api/v2/<subpath>`` to Airflow ``/api/v2/<subpath>``."""
    # Django hands us a decoded path; re-encode each segment so run ids with
    # ':' and '+' survive the trip.
    segments = [s for s in subpath.split('/') if s != '']
    path = '/' + '/'.join(encode_segment(s) for s in segments)

    template, allowed = match_route(path, request.method)
    if template is None:
        return _error(404, f'Unknown Airflow API route: {path}')
    if not allowed:
        return _error(405, f'{request.method} is not supported on {template}')

    accept = request.headers.get('Accept', 'application/json')
    accept = next((a for a in PASSTHROUGH_ACCEPT if a in accept), 'application/json')

    body = None
    if request.method in ('POST', 'PUT', 'PATCH') and request.body:
        try:
            json.loads(request.body)
        except ValueError:
            return _error(400, 'Request body must be valid JSON.')
        body = request.body

    # Keep repeated keys (``?tags=a&tags=b``, ``?state=failed&state=running``).
    params = [(k, v) for k, values in request.GET.lists() for v in values] or None

    try:
        upstream = get_client().request(
            request.method, path, params=params, data=body, accept=accept,
        )
    except AirflowError as exc:
        logger.error('Airflow proxy %s %s failed: %s', request.method, path, exc)
        return JsonResponse(exc.payload, status=exc.status_code)

    if request.method != 'GET':
        _audit(request, path, template, upstream.status_code)

    response = HttpResponse(
        upstream.content,
        status=upstream.status_code,
        content_type=upstream.content_type,
    )
    for key, value in upstream.headers.items():
        response[key] = value
    response['Cache-Control'] = 'no-store'
    return response


@require_GET
@_superuser_only
def api_spec(request: HttpRequest) -> HttpResponse:
    """The Airflow OpenAPI spec, trimmed to the v2 routes the proxy serves."""
    spec = cache.get(SPEC_CACHE_KEY)
    if spec is None:
        try:
            # The spec needs no auth.  Try the configured layout, then the other
            # one (root vs /airflow), so it works on either Airflow stack.
            client = get_client()
            resp = None
            for base in client.candidate_bases():
                resp = requests.get(f'{base}/openapi.json', timeout=30)
                if resp.status_code != 404:
                    break
            resp.raise_for_status()
            spec = resp.json()
        except (requests.RequestException, ValueError) as exc:
            return _error(502, f'Could not load the Airflow OpenAPI spec: {exc}')
        spec['paths'] = {
            p: ops for p, ops in spec.get('paths', {}).items()
            if p.startswith('/api/v2/') and not p.startswith('/api/v2/auth/')
        }
        cache.set(SPEC_CACHE_KEY, spec, SPEC_TTL_SECONDS)
    return JsonResponse(spec)
