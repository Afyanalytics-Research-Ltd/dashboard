"""
Semantic Layer Configuration settings-page HTML URL patterns.

Separate from agents/api_urls.py (the REST agent API, included unnamespaced
at /agents/ in the root urlconf) — this is the human-facing settings UI,
included at /settings/agents/.
"""

from django.urls import path

from . import views

app_name = "agents"

urlpatterns = [
    path("", views.AgentConfigurationView.as_view(), name="agent-configuration"),
    path("metrics/save/", views.MetricDefinitionSaveView.as_view(), name="metric-save"),
    path("measures/propose/", views.ProposeCubeMeasureView.as_view(), name="measure-propose"),
    path("measures/<int:pk>/approve/", views.ApproveCubeMeasureView.as_view(), name="measure-approve"),
    path("measures/<int:pk>/reject/", views.RejectCubeMeasureView.as_view(), name="measure-reject"),
    path("generate/", views.GenerateMetricsView.as_view(), name="generate-metrics"),
    path("rebuild-embeddings/", views.RebuildEmbeddingsView.as_view(), name="rebuild-embeddings"),
    path("sync-cube-schemas/", views.SyncCubeSchemasView.as_view(), name="sync-cube-schemas"),
]
