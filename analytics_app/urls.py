from django.urls import path
from . import views

urlpatterns = [
    path("", views.dashboard_list, name="dashboard_home"),
    path("dashboards/<slug:slug>/", views.dashboard_view, name="dashboard_view"),
]
