"""
Core app HTML URL patterns.
"""

from django.urls import path

from . import views

app_name = 'core'

urlpatterns = [
    path('settings/', views.SystemSettingsView.as_view(), name='settings'),
    path('settings/permissions/', views.PermissionsView.as_view(), name='permissions'),
    path('notifications/', views.NotificationListView.as_view(), name='notifications'),
    path(
        'notifications/<int:pk>/read/',
        views.MarkNotificationReadView.as_view(),
        name='notification-mark-read',
    ),
    path(
        'notifications/mark-all-read/',
        views.MarkAllNotificationsReadView.as_view(),
        name='notifications-mark-all-read',
    ),
]
