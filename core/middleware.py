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
    """Extract the real client IP address from the request, respecting proxies.

    When the platform runs behind a load balancer or reverse proxy (e.g.
    Nginx), the original client IP is forwarded in the ``X-Forwarded-For``
    header rather than ``REMOTE_ADDR``.  This function reads the first
    (leftmost) IP from ``X-Forwarded-For``, which represents the actual
    client, or falls back to ``REMOTE_ADDR`` for direct connections.

    Args:
        request: The incoming Django HTTP request.

    Returns:
        The client's IP address as a string (IPv4 or IPv6), or ``None``
        if the address cannot be determined.
    """
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        return x_forwarded_for.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR')


def _resource_from_path(path: str) -> str:
    """Derive a human-readable resource name from a URL path.

    Takes the first meaningful path segment and converts it to Title Case,
    replacing hyphens and underscores with spaces.  Used to populate the
    ``resource`` field of :class:`core.models.AuditLog` entries created by
    middleware.

    Examples:
        ``/api/v1/analytics/dashboards/`` → ``"Api"``
        ``/warehouse/spreadsheets/``      → ``"Warehouse"``
        ``/``                             → ``"root"``

    Args:
        path: The URL path from ``request.path``.

    Returns:
        A short, title-cased string identifying the resource area.
    """
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
        """Process the request: delegate to the next middleware, then log if needed.

        The response is generated first so that the HTTP status code is
        available when we write the audit log entry.

        Args:
            request: The incoming HTTP request.

        Returns:
            The HTTP response produced by the view or downstream middleware.
        """
        response = self.get_response(request)

        if request.method in _MUTATING_METHODS:
            self._maybe_log(request, response)

        return response

    def _maybe_log(self, request: HttpRequest, response: HttpResponse) -> None:
        """Conditionally write an :class:`AuditLog` entry for this request.

        Skips logging for:
        - Static/media files and API schema endpoints (not user actions).
        - Unauthenticated requests (no user to attribute the action to).

        Silently catches and logs any exception so a logging failure never
        breaks the user's request.

        Args:
            request: The original HTTP request.
            response: The HTTP response returned by the view.
        """
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
    """Lightweight request/response logger for operational monitoring.

    Writes a single log line per request containing the HTTP method, path,
    status code, and elapsed time.  The log level escalates with severity:
    - DEBUG for safe read requests (GET, HEAD, OPTIONS).
    - INFO  for mutating requests (POST, PUT, PATCH, DELETE).
    - WARNING for 4xx client errors.
    - ERROR   for 5xx server errors.

    Non-technical explanation:
        Every time someone interacts with the platform, this middleware
        writes a timestamped note like "GET /analytics/ 200 (45ms)" to the
        application log — similar to an access log in a web server.
        Operations teams use this to spot slow pages or unusual error rates.
    """

    def __init__(self, get_response: Callable) -> None:
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        """Time the request and log it after the response is ready.

        Args:
            request: The incoming HTTP request.

        Returns:
            The HTTP response produced by the view or downstream middleware.
        """
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
