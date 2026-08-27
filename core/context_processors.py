"""
Template context processors for Afya DataHub.
"""

from django.conf import settings
from django.http import HttpRequest


def notifications(request: HttpRequest) -> dict:
    """
    Add notification data to every template context.

    Returns:
        unread_notification_count  – integer badge count
        recent_notifications       – last 5 unread notifications for the user
    """
    if not (request.user and request.user.is_authenticated):
        return {
            'unread_notification_count': 0,
            'recent_notifications': [],
        }

    try:
        from .models import Notification  # local import avoids startup issues
        unread_qs = Notification.objects.filter(user=request.user, is_read=False)
        count = unread_qs.count()
        recent = list(unread_qs.select_related('user').order_by('-created_at')[:5])
    except Exception:
        count = 0
        recent = []

    return {
        'unread_notification_count': count,
        'recent_notifications': recent,
    }


def brand_settings(request: HttpRequest) -> dict:
    """
    Expose brand colours and app metadata to all templates.
    """
    brand = getattr(settings, 'AFYA_BRAND', {})
    return {
        'BRAND': brand,
        'APP_NAME': brand.get('APP_NAME', 'Afya DataHub'),
        'APP_VERSION': brand.get('VERSION', ''),
    }


def open_tickets(request: HttpRequest) -> dict:
    """
    Expose the count of open support tickets to staff users, for the
    sidebar's "Support & Ticketing" badge (mirrors the notification bell
    badge). Always 0 for non-staff/anonymous users, so nobody sees a badge
    hinting at ticket volume unless they can actually act on it.
    """
    if not (request.user and request.user.is_authenticated and request.user.is_staff):
        return {'open_ticket_count': 0}

    try:
        from .models import Ticket
        count = Ticket.objects.filter(status=Ticket.STATUS_OPEN).count()
    except Exception:
        count = 0

    return {'open_ticket_count': count}


def module_access(request: HttpRequest) -> dict:
    """
    Expose each module's effective access for the current user, so templates
    (the sidebar in particular) can show/hide sections based on the same
    grant/revoke rules enforced server-side — see
    authentication.module_access.has_module_access.

    Returns:
        module_access – dict like {"warehouse": True, "analytics": True,
        "self_service": False} for the current user; all False for anonymous
        visitors.
    """
    if not (request.user and request.user.is_authenticated):
        return {'module_access': {}}

    try:
        from authentication.module_access import ALL_MODULES, has_module_access
        return {
            'module_access': {key: has_module_access(request.user, key) for key in ALL_MODULES}
        }
    except Exception:
        return {'module_access': {}}
