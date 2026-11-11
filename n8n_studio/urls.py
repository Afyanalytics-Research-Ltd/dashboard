"""
Automation Studio URL patterns (mounted at ``/automations/``; nginx already
proxies ``/n8n/`` to n8n itself).
"""

from django.urls import path, re_path

from . import endpoints, proxy, views

app_name = 'n8n_studio'

app = views.StudioAppView.as_view()

urlpatterns = [
    # Browser -> n8n public API proxy
    path('api/', proxy.n8n_proxy, {'subpath': ''}, name='proxy_root'),
    path('api/<path:subpath>', proxy.n8n_proxy, name='proxy'),

    # Studio endpoints
    path('studio/health/', endpoints.health, name='health'),
    path('studio/catalog/', endpoints.catalog_index, name='catalog'),
    path('studio/catalog/nodes/<str:node_type>/', endpoints.catalog_node, name='catalog_node'),
    path('studio/catalog/credentials/<str:cred_type>/', endpoints.catalog_credential, name='catalog_credential'),
    path('studio/icons/<path:path>', endpoints.icon, name='icon'),
    path('studio/run/<str:workflow_id>/', endpoints.run_workflow, name='run'),
    path('studio/webhook/', endpoints.webhook_call, name='webhook'),
    path('studio/snapshots/', endpoints.snapshots, name='snapshots'),
    path('studio/snapshots/deleted/', endpoints.snapshots_deleted, name='snapshots_deleted'),
    path('studio/snapshots/capture/', endpoints.snapshots_capture, name='snapshots_capture'),
    path('studio/snapshots/<int:pk>/', endpoints.snapshot_detail, name='snapshot'),

    # Single-page app
    path('', app, name='home'),
    re_path(r'^(?!api/|studio/).+$', app, name='app'),
]
