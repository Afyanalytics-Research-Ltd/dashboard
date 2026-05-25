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
    """Platform-wide system settings page (superusers only)."""

    template_name = 'core/settings.html'

    def get_breadcrumbs(self):
        return [
            {'label': 'Home', 'url': '/analytics/'},
            {'label': 'System Settings', 'url': None},
        ]

    def get_context_data(self, **kwargs) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        context['settings_list'] = SystemSettings.objects.all()
        context['sidebar_section'] = 'settings'
        context['page_title'] = 'System Settings'
        return context

    def post(self, request: HttpRequest) -> JsonResponse:
        """Handle AJAX setting updates."""
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
    """In-app notification inbox."""

    template_name = 'core/notifications.html'

    def get_breadcrumbs(self):
        return [
            {'label': 'Home', 'url': '/analytics/'},
            {'label': 'Notifications', 'url': None},
        ]

    def get_context_data(self, **kwargs) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        context['notifications'] = Notification.objects.filter(user=self.request.user)
        context['sidebar_section'] = 'profile'
        context['page_title'] = 'Notifications'
        return context


@method_decorator(login_required, name='dispatch')
class MarkNotificationReadView(View):
    """Mark a single notification as read (AJAX-friendly)."""

    def post(self, request: HttpRequest, pk: int) -> JsonResponse:
        notification = get_object_or_404(Notification, pk=pk, user=request.user)
        notification.mark_read()
        return JsonResponse({'ok': True})


@method_decorator(login_required, name='dispatch')
class MarkAllNotificationsReadView(View):
    """Mark all of the current user's notifications as read."""

    def post(self, request: HttpRequest) -> HttpResponse:
        count = Notification.objects.filter(user=request.user, is_read=False).update(is_read=True)
        messages.success(request, f'{count} notification(s) marked as read.')
        return redirect(request.META.get('HTTP_REFERER', '/analytics/'))


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

def error_403(request: HttpRequest, exception=None) -> HttpResponse:
    return render(request, '403.html', {
        'sidebar_section': '',
        'page_title': 'Access Denied',
    }, status=403)


def error_404(request: HttpRequest, exception=None) -> HttpResponse:
    return render(request, '404.html', {
        'sidebar_section': '',
        'page_title': 'Page Not Found',
    }, status=404)


def error_500(request: HttpRequest) -> HttpResponse:
    return render(request, '500.html', {
        'sidebar_section': '',
        'page_title': 'Server Error',
    }, status=500)
