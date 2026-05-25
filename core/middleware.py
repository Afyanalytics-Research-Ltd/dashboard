"""
Custom middleware for Afya DataHub.

- AuditLogMiddleware  : records mutating HTTP requests to AuditLog
- RequestLoggingMiddleware : lightweight request/response logging
"""

import logging
import time
from typing import Callable

from django.http import HttpRequest, HttpResponse

logger = logging.getLogger('core')

# Methods that modify state
_MUTATING_METHODS = frozenset({'POST', 'PUT', 'PATCH', 'DELETE'})

# URL prefixes to skip (admin, static, media, API schema)
_SKIP_PREFIXES = (
    '/static/',
    '/media/',
    '/api/v1/schema/',
    '/api/v1/docs/',
    '/api/v1/redoc/',
    '/favicon.ico',
)


def _get_client_ip(request: HttpRequest) -> str | None:
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        return x_forwarded_for.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR')


def _resource_from_path(path: str) -> str:
    """Derive a human-readable resource name from a URL path."""
    parts = [p for p in path.strip('/').split('/') if p]
    if not parts:
        return 'root'
    return parts[0].replace('-', '_').replace('_', ' ').title()


class AuditLogMiddleware:
    """
    Records POST / PUT / PATCH / DELETE requests as AuditLog entries.

    Logs are written after the response is generated so the status code
    is available.  Database imports are deferred to avoid circular imports
    during startup.
    """

    def __init__(self, get_response: Callable) -> None:
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        response = self.get_response(request)

        if request.method in _MUTATING_METHODS:
            self._maybe_log(request, response)

        return response

    def _maybe_log(self, request: HttpRequest, response: HttpResponse) -> None:
        path: str = request.path

        # Skip uninteresting paths
        if any(path.startswith(p) for p in _SKIP_PREFIXES):
            return

        # Only log for authenticated users
        user = getattr(request, 'user', None)
        if not (user and user.is_authenticated):
            return

        # Determine action from HTTP method
        method_to_action = {
            'POST': 'create',
            'PUT': 'update',
            'PATCH': 'update',
            'DELETE': 'delete',
        }
        action = method_to_action.get(request.method, 'update')

        resource = _resource_from_path(path)
        detail = f'{request.method} {path} → {response.status_code}'

        try:
            from .models import AuditLog  # local import to avoid circular deps
            AuditLog.log(
                user=user,
                action=action,
                resource=resource,
                resource_id='',
                detail=detail,
                ip_address=_get_client_ip(request),
                user_agent=request.META.get('HTTP_USER_AGENT', ''),
            )
        except Exception:
            logger.exception('AuditLogMiddleware: failed to write audit log entry')


class RequestLoggingMiddleware:
    """
    Logs each request with method, path, status code, and duration.
    Uses DEBUG level for read requests, INFO for mutations, WARNING for 4xx/5xx.
    """

    def __init__(self, get_response: Callable) -> None:
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        start = time.monotonic()
        response = self.get_response(request)
        duration_ms = (time.monotonic() - start) * 1000

        status = response.status_code
        msg = f'{request.method} {request.path} {status} ({duration_ms:.1f}ms)'

        if status >= 500:
            logger.error(msg)
        elif status >= 400:
            logger.warning(msg)
        elif request.method in _MUTATING_METHODS:
            logger.info(msg)
        else:
            logger.debug(msg)

        return response
