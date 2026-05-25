"""
DRF URL patterns for the authentication API.
Mounted at /api/v1/auth/ by the root URLconf.
"""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .api import (
    ChangePasswordView,
    NotificationViewSet,
    UserActivityView,
    UserProfileViewSet,
    UserViewSet,
)

router = DefaultRouter()
router.register(r'users', UserViewSet, basename='user')
router.register(r'profiles', UserProfileViewSet, basename='profile')
router.register(r'notifications', NotificationViewSet, basename='notification')

urlpatterns = [
    path('', include(router.urls)),
    path('password/change/', ChangePasswordView.as_view(), name='api-password-change'),
    path('activity/', UserActivityView.as_view(), name='api-user-activity'),
]
