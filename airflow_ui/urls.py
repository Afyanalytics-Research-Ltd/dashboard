"""
Airflow pipeline UI URL patterns (mounted at ``/pipelines/``).
"""

from django.urls import path, re_path

from . import proxy, views

app_name = 'airflow'

app = views.PipelineAppView.as_view()

urlpatterns = [
    # Browser -> Airflow REST API v2 proxy
    path('api/spec/', proxy.api_spec, name='api_spec'),
    path('api/v2/', proxy.api_proxy, {'subpath': ''}, name='api_root'),
    path('api/v2/<path:subpath>', proxy.api_proxy, name='api_proxy'),

    # Single-page app: named routes kept for reverse() + a catch-all.
    path('', app, name='dashboard'),
    path('dags/<str:dag_id>/', app, name='dag_detail'),
    path('dags/<str:dag_id>/runs/<str:run_id>/', app, name='dag_run_detail'),
    re_path(r'^(?!api/).+$', app, name='app'),
]
