"""
URL patterns for the authentication app (HTML views).
Namespace: 'authentication'
Mounted at /auth/ by the root URLconf.
"""

from django.urls import path

from .views import (
    LandingView,
    LoginView,
    LogoutView,
    MarkNotificationReadView,
    NotificationsListView,
    PasswordChangeView,
    ProfileView,
    SignupView,
    UserActivityView,
)

app_name = 'authentication'

urlpatterns = [
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('signup/', SignupView.as_view(), name='signup'),
    path('profile/', ProfileView.as_view(), name='profile'),
    path('password/change/', PasswordChangeView.as_view(), name='password_change'),
    path('notifications/', NotificationsListView.as_view(), name='notifications'),
    path('notifications/<int:pk>/read/', MarkNotificationReadView.as_view(), name='notification_read'),
    path('activity/', UserActivityView.as_view(), name='activity'),
]
