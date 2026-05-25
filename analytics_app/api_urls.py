"""
Analytics app API URL patterns.
"""

from django.urls import path
from rest_framework.routers import DefaultRouter

from .api import DashboardStatsAPIView, DashboardSyncAPIView, DashboardViewSet

router = DefaultRouter()
router.register(r'dashboards', DashboardViewSet, basename='dashboard')

# Explicit paths must come BEFORE router URLs to avoid being caught by <pk> pattern
urlpatterns = [
    path('dashboards/sync/', DashboardSyncAPIView.as_view(), name='dashboard-sync'),
    path('dashboards/stats/', DashboardStatsAPIView.as_view(), name='dashboard-stats'),
] + router.urls
