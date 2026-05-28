"""
Core HTML views: system settings, error handlers.
"""

import logging
from typing import Any

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.decorators import method_decorator
from django.views import View
from django.views.generic import TemplateView

from .mixins import BreadcrumbMixin, LoggingMixin, SuperuserRequiredMixin
from .models import AuditLog, Client, Facility, Notification, SystemSettings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System Settings
# ---------------------------------------------------------------------------

@method_decorator(login_required, name='dispatch')
class SystemSettingsView(SuperuserRequiredMixin, BreadcrumbMixin, LoggingMixin, TemplateView):
    """Platform-wide system settings page (superusers only).

    Displays all :class:`SystemSettings` key/value pairs and accepts AJAX
    POST requests to update individual settings in-place without a full
    page reload.

    Access is restricted to Django superusers via
    :class:`core.mixins.SuperuserRequiredMixin`.

    Non-technical explanation:
        This is the admin control panel for the whole platform — like the
        settings menu on a phone that only the IT administrator can open.
        Superusers can read and change platform-wide configuration values
        here without touching any code.
    """

    template_name = 'core/settings.html'

    def get_breadcrumbs(self):
        """Return the breadcrumb trail for the settings page."""
        return [
            {'label': 'Home', 'url': '/analytics/'},
            {'label': 'System Settings', 'url': None},
        ]

    def get_context_data(self, **kwargs) -> dict[str, Any]:
        """Build the template context with all current system settings.

        Returns:
            A dict containing:
            - ``settings_list``: All :class:`SystemSettings` rows.
            - ``sidebar_section``: Highlights the correct sidebar link.
            - ``page_title``: Title shown in the browser tab.
        """
        context = super().get_context_data(**kwargs)
        context['settings_list'] = SystemSettings.objects.all()
        context['sidebar_section'] = 'settings'
        context['page_title'] = 'System Settings'
        return context

    def post(self, request: HttpRequest) -> JsonResponse:
        """Handle an AJAX request to create or update a system setting.

        Expects ``key``, ``value``, and optionally ``description`` in the
        POST body.  The ``value`` is parsed as JSON if possible so that
        booleans, numbers, and objects are stored correctly; otherwise it
        is stored as a plain string.

        Args:
            request: The incoming HTTP POST request.

        Returns:
            A JSON response with ``{"ok": true, "key": ..., "value": ...,
            "updated_at": ...}`` on success, or ``{"ok": false, "error":
            ...}`` with status 400 on validation failure.
        """
        key = request.POST.get('key', '').strip()
        value = request.POST.get('value', '')
        description = request.POST.get('description', '').strip()

        if not key:
            return JsonResponse({'ok': False, 'error': 'Key is required.'}, status=400)

        # Parse value as JSON if possible, else store as string
        import json
        try:
            parsed_value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            parsed_value = value

        setting = SystemSettings.set(key=key, value=parsed_value, user=request.user, description=description)
        self.audit_log('update', 'SystemSettings', resource_id=key, detail=f'Updated setting {key}')
        logger.info('Superuser %s updated system setting: %s', request.user.username, key)

        return JsonResponse({
            'ok': True,
            'key': setting.key,
            'value': setting.value,
            'updated_at': setting.updated_at.isoformat(),
        })


@method_decorator(login_required, name='dispatch')
class NotificationListView(BreadcrumbMixin, TemplateView):
    """In-app notification inbox for the currently logged-in user.

    Renders all notifications belonging to the request user, newest first.
    Login is required (enforced by the ``login_required`` decorator applied
    to ``dispatch``).

    Non-technical explanation:
        This is the bell icon / inbox page — showing all messages the
        platform has sent to the current user, such as "your export is
        ready" or "a new dashboard has been added for your facility."
    """

    template_name = 'core/notifications.html'

    def get_breadcrumbs(self):
        """Return the breadcrumb trail for the notifications page."""
        return [
            {'label': 'Home', 'url': '/analytics/'},
            {'label': 'Notifications', 'url': None},
        ]

    def get_context_data(self, **kwargs) -> dict[str, Any]:
        """Build the template context with the user's notifications.

        Returns:
            A dict containing:
            - ``notifications``: QuerySet of the user's notifications,
              ordered newest-first.
            - ``sidebar_section``: Highlights the correct sidebar link.
            - ``page_title``: Title shown in the browser tab.
        """
        context = super().get_context_data(**kwargs)
        context['notifications'] = Notification.objects.filter(user=self.request.user)
        context['sidebar_section'] = 'profile'
        context['page_title'] = 'Notifications'
        return context


@method_decorator(login_required, name='dispatch')
class MarkNotificationReadView(View):
    """Mark a single notification as read via an AJAX POST request.

    Ownership is enforced — users can only mark their own notifications,
    not those belonging to other accounts.
    """

    def post(self, request: HttpRequest, pk: int) -> JsonResponse:
        """Mark the notification identified by ``pk`` as read.

        Args:
            request: The incoming HTTP POST request.
            pk: Primary key of the :class:`Notification` to mark read.

        Returns:
            ``{"ok": true}`` on success, or 404 if the notification does
            not exist or belongs to a different user.
        """
        notification = get_object_or_404(Notification, pk=pk, user=request.user)
        notification.mark_read()
        return JsonResponse({'ok': True})


@method_decorator(login_required, name='dispatch')
class MarkAllNotificationsReadView(View):
    """Mark every unread notification for the current user as read in bulk.

    More efficient than calling :class:`MarkNotificationReadView` once per
    notification because it uses a single database UPDATE statement.
    """

    def post(self, request: HttpRequest) -> HttpResponse:
        """Mark all unread notifications as read and redirect back.

        Args:
            request: The incoming HTTP POST request.

        Returns:
            A redirect to the referring page (or ``/analytics/`` as
            fallback), with a success flash message indicating how many
            notifications were cleared.
        """
        count = Notification.objects.filter(user=request.user, is_read=False).update(is_read=True)
        messages.success(request, f'{count} notification(s) marked as read.')
        return redirect(request.META.get('HTTP_REFERER', '/analytics/'))


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

def error_403(request: HttpRequest, exception=None) -> HttpResponse:
    """Render the custom 403 Access Denied page.

    Called by Django when a view raises :class:`~django.core.exceptions.PermissionDenied`.
    Returns a styled HTML page instead of the default Django error page.

    Args:
        request: The request that triggered the permission error.
        exception: The exception instance (may be ``None``).

    Returns:
        An HTTP 403 response with the ``403.html`` template.
    """
    return render(request, '403.html', {
        'sidebar_section': '',
        'page_title': 'Access Denied',
    }, status=403)


def error_404(request: HttpRequest, exception=None) -> HttpResponse:
    """Render the custom 404 Page Not Found page.

    Called by Django when no URL pattern matches the requested path.

    Args:
        request: The request for the missing page.
        exception: The exception instance (may be ``None``).

    Returns:
        An HTTP 404 response with the ``404.html`` template.
    """
    return render(request, '404.html', {
        'sidebar_section': '',
        'page_title': 'Page Not Found',
    }, status=404)


def error_500(request: HttpRequest) -> HttpResponse:
    """Render the custom 500 Server Error page.

    Called by Django when an unhandled exception propagates out of a view.
    Does not accept an exception argument because Django 500 handlers are
    called after the exception has already been logged.

    Args:
        request: The request that caused the server error.

    Returns:
        An HTTP 500 response with the ``500.html`` template.
    """
    return render(request, '500.html', {
        'sidebar_section': '',
        'page_title': 'Server Error',
    }, status=500)
