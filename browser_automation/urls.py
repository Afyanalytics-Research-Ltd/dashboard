"""HTML URL configuration for the browser-automation module (namespace='browserbase')."""

from django.urls import path

from . import views

app_name = "browserbase"

urlpatterns = [
    path("", views.ConsoleView.as_view(), name="console"),
    path("sessions/new/", views.CreateSessionView.as_view(), name="create_session"),
    path("sessions/<str:session_id>/", views.SessionDetailView.as_view(), name="session_detail"),
    path("sessions/<str:session_id>/end/", views.EndSessionView.as_view(), name="end_session"),
    path("mcp/navigate/", views.RunMCPTaskView.as_view(), name="mcp_navigate"),
    path("scheduled/", views.ScheduledTaskListView.as_view(), name="scheduled_tasks"),
    path("scheduled/new/", views.CreateScheduledTaskView.as_view(), name="create_scheduled_task"),
    path("scheduled/<int:pk>/edit/", views.UpdateScheduledTaskView.as_view(), name="update_scheduled_task"),
    path("scheduled/<int:pk>/delete/", views.DeleteScheduledTaskView.as_view(), name="delete_scheduled_task"),
    path("scheduled/<int:pk>/run/", views.RunScheduledTaskNowView.as_view(), name="run_scheduled_task"),
]
