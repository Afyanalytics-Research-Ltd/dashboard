from django.urls import path

from agents.api import RunAgentsView

urlpatterns = [
    path("run/", RunAgentsView.as_view(), name="agents-run"),
]
