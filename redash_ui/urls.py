"""
Analytics Studio URL patterns (mounted at ``/redash/``).
"""

from django.urls import path, re_path

from . import proxy, views

app_name = 'redash_ui'

app = views.StudioAppView.as_view()

urlpatterns = [
    # Browser -> Redash API proxy
    path('proxy/', proxy.redash_proxy, {'subpath': ''}, name='proxy_root'),
    path('proxy/<path:subpath>', proxy.redash_proxy, name='proxy'),
    path('history/<int:query_id>/', proxy.query_history, name='query_history'),

    # Single-page app
    path('', app, name='home'),
    re_path(r'^(?!proxy/|history/).+$', app, name='app'),
]
