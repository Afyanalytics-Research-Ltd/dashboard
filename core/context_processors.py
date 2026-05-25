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
