# accounts/urls.py

from django.urls import path
from .views import (
    login_view, logout_view, signup_view,
    profile_view, password_reset_view
)

urlpatterns = [
    path("login/", login_view, name="login"),
    path("logout/", logout_view, name="logout"),
    path("signup/", signup_view, name="signup"),
    path("profile/", profile_view, name="profile"),
    path("password-reset/", password_reset_view, name="password_reset"),
]