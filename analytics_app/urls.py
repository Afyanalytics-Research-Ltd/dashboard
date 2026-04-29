from django.urls import path
from . import views

urlpatterns = [
    # path("", views.dashboard_main, name = "dashboard_home"),
    path("", views.dashboard_list, name = "dashboard_home"),
    path("dashboard/list/", views.dashboard_list, name="dashboard_list"),
    path("dashboards/<slug:slug>/", views.dashboard_view, name="dashboard_view"),
]
