"""
Core app DRF API URL patterns (registered via DRF Router).
"""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .api import (
    AuditLogViewSet,
    ClientViewSet,
    FacilityViewSet,
    NotificationViewSet,
    SystemSettingsViewSet,
)

router = DefaultRouter()
router.register(r'clients', ClientViewSet, basename='client')
router.register(r'facilities', FacilityViewSet, basename='facility')
router.register(r'audit-logs', AuditLogViewSet, basename='auditlog')
router.register(r'notifications', NotificationViewSet, basename='notification')
router.register(r'system-settings', SystemSettingsViewSet, basename='systemsettings')

urlpatterns = [
    path('', include(router.urls)),
]
