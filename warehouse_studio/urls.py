"""Warehouse Studio URL patterns (mounted at ``/warehouse-studio/``)."""

from django.urls import path, re_path

from . import api, views

app_name = 'warehouse_studio'

app = views.StudioAppView.as_view()
QID = '<str:query_id>'

api_patterns = [
    path('api/', api.session_view, name='api_root'),
    path('api/session/', api.session_view, name='api_session'),
    # Queries
    path('api/query/split/', api.query_split, name='api_query_split'),
    path('api/query/submit/', api.query_submit, name='api_query_submit'),
    path('api/query/explain/', api.query_explain, name='api_query_explain'),
    path(f'api/query/{QID}/status/', api.query_status, name='api_query_status'),
    path(f'api/query/{QID}/results/', api.query_results, name='api_query_results'),
    path(f'api/query/{QID}/cancel/', api.query_cancel, name='api_query_cancel'),
    path(f'api/query/{QID}/export/', api.query_export, name='api_query_export'),
    path(f'api/query/{QID}/details/', api.query_details, name='api_query_details'),
    path(f'api/query/{QID}/profile/', api.query_profile, name='api_query_profile'),
    path('api/history/', api.history_view, name='api_history'),
    path('api/runs/', api.my_runs, name='api_runs'),
    # Worksheets
    path('api/worksheets/', api.worksheets, name='api_worksheets'),
    path('api/worksheets/<uuid:pk>/', api.worksheet_detail, name='api_worksheet'),
    # Catalog & procedures
    path('api/catalog/databases/', api.catalog_databases, name='api_databases'),
    path('api/catalog/schemas/', api.catalog_schemas, name='api_schemas'),
    path('api/catalog/objects/', api.catalog_objects, name='api_objects'),
    path('api/catalog/columns/', api.catalog_columns, name='api_columns'),
    path('api/catalog/ddl/', api.catalog_ddl, name='api_ddl'),
    path('api/procedures/call/', api.procedure_call, name='api_procedure_call'),
    # Cost center
    path('api/cost/overview/', api.cost_overview, name='api_cost_overview'),
    path('api/cost/heatmap/', api.cost_heatmap, name='api_cost_heatmap'),
    path('api/cost/warehouses/', api.cost_warehouses, name='api_cost_warehouses'),
    path('api/cost/attribution/', api.cost_attribution, name='api_cost_attribution'),
    path('api/cost/top-queries/', api.cost_top_queries, name='api_cost_top_queries'),
    path('api/cost/storage/', api.cost_storage, name='api_cost_storage'),
    path('api/cost/monitors/', api.cost_monitors, name='api_cost_monitors'),
    path('api/cost/refresh/', api.cost_refresh, name='api_cost_refresh'),
    # Optimizer (the cost & performance stored procedures)
    path('api/optimizer/', api.optimizer_catalog, name='api_optimizer'),
    path('api/optimizer/run/', api.optimizer_run, name='api_optimizer_run'),
    path('api/optimizer/runs/', api.optimizer_runs, name='api_optimizer_runs'),
    path('api/optimizer/runs/<int:run_id>/', api.optimizer_run_detail, name='api_optimizer_run_detail'),
    path('api/optimizer/findings/', api.optimizer_findings, name='api_findings'),
    path('api/optimizer/findings/triage/', api.optimizer_triage_bulk, name='api_findings_triage_bulk'),
    path('api/optimizer/findings/<int:finding_id>/triage/', api.optimizer_triage, name='api_finding_triage'),
    path('api/optimizer/summary/', api.optimizer_summary, name='api_findings_summary'),
    path('api/optimizer/batches/', api.optimizer_batches, name='api_optimizer_batches'),
    path('api/optimizer/schedule/', api.optimizer_schedule, name='api_optimizer_schedule'),
]

urlpatterns = api_patterns + [
    path('', app, name='home'),
    re_path(r'^(?!api/).+$', app, name='app'),
]
