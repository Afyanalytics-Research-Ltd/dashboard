"""Connect Studio URL patterns (mounted at ``/connect/``)."""

from django.urls import path, re_path

from . import api, views

app_name = "connect_studio"

app = views.StudioAppView.as_view()

api_patterns = [
    path("api/meta/", api.meta, name="api_meta"),
    path("api/overview/", api.overview, name="api_overview"),
    path("api/import-api/", api.import_api, name="api_import_api"),
    path("api/sources/", api.sources, name="api_sources"),
    path("api/sources/<int:pk>/", api.source_detail, name="api_source"),
    path("api/sources/<int:pk>/files/", api.source_files, name="api_source_files"),
    path("api/sources/<int:pk>/files/<int:fid>/", api.source_file_delete, name="api_source_file"),
    path("api/sources/<int:pk>/free-space/", api.source_free_space, name="api_source_free_space"),
    path("api/sources/<int:pk>/uploads/", api.source_upload_start, name="api_source_upload_start"),
    path("api/uploads/<uuid:uid>/chunks/<int:index>/", api.upload_chunk, name="api_upload_chunk"),
    path("api/uploads/<uuid:uid>/", api.upload_finish, name="api_upload_finish"),
    path("api/sources/<int:pk>/test/", api.source_test, name="api_source_test"),
    path("api/sources/<int:pk>/extract/", api.source_extract, name="api_source_extract"),
    path("api/sources/<int:pk>/snapshots/", api.source_snapshots, name="api_source_snapshots"),
    path("api/snapshots/<int:pk>/", api.snapshot_detail, name="api_snapshot"),
    path("api/snapshots/<int:pk>/compare/", api.snapshot_compare, name="api_snapshot_compare"),
    path("api/mappings/", api.mappings, name="api_mappings"),
    path("api/mappings/<int:pk>/", api.mapping_detail, name="api_mapping"),
    path("api/mappings/<int:pk>/generate/", api.mapping_generate, name="api_mapping_generate"),
    path("api/mappings/<int:pk>/fields/", api.mapping_fields, name="api_mapping_fields"),
    path("api/mappings/<int:pk>/bulk/", api.mapping_bulk, name="api_mapping_bulk"),
    path("api/mappings/<int:pk>/table/", api.mapping_table, name="api_mapping_table"),
    path("api/mappings/<int:pk>/preview/", api.mapping_preview, name="api_mapping_preview"),
    path("api/mappings/<int:pk>/export/", api.mapping_export, name="api_mapping_export"),
    path("api/connections/", api.connections_view, name="api_connections"),
    path("api/connections/<int:pk>/", api.connection_detail, name="api_connection"),
    path("api/connections/<int:pk>/run/", api.connection_run, name="api_connection_run"),
    path("api/jobs/", api.jobs_view, name="api_jobs"),
    path("api/jobs/<int:pk>/", api.job_detail, name="api_job"),
    path("api/jobs/<int:pk>/cancel/", api.job_cancel, name="api_job_cancel"),
]

urlpatterns = api_patterns + [
    path("", app, name="home"),
    re_path(r"^(?!api/).+$", app, name="app"),
]
