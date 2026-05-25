"""HTML URL configuration for the warehouse module (namespace='warehouse')."""

from django.urls import path

from . import views

app_name = "warehouse"

urlpatterns = [
    # Home
    path("", views.WarehouseHomeView.as_view(), name="index"),

    # Snowflake query interface
    path("snowflake/", views.SnowflakeQueryView.as_view(), name="snowflake"),

    # Report missing data (AJAX or form POST)
    path("report-missing/", views.ReportMissingView.as_view(), name="report_missing"),

    # Spreadsheet detail
    path("<str:spreadsheet_id>/", views.SpreadsheetDetailView.as_view(), name="detail"),

    # Values
    path("<str:spreadsheet_id>/values/read/", views.read_values, name="read_values"),
    path("<str:spreadsheet_id>/values/update/", views.update_values, name="update_values"),
    path("<str:spreadsheet_id>/values/append/", views.append_values, name="append_values"),
    path("<str:spreadsheet_id>/values/clear/", views.clear_range, name="clear_range"),
    path("<str:spreadsheet_id>/values/batch/", views.batch_update_values, name="batch_update"),

    # Tabs
    path("<str:spreadsheet_id>/tabs/add/", views.add_tab, name="add_tab"),
    path("<str:spreadsheet_id>/tabs/rename/", views.rename_tab, name="rename_tab"),
    path("<str:spreadsheet_id>/tabs/delete/", views.delete_tab, name="delete_tab"),

    # Formatting
    path("<str:spreadsheet_id>/format/", views.format_cells, name="format_cells"),
    path("<str:spreadsheet_id>/freeze/", views.freeze_rows, name="freeze_rows"),

    # Rows
    path("<str:spreadsheet_id>/rows/delete/", views.delete_rows, name="delete_rows"),
    path("<str:spreadsheet_id>/rows/insert/", views.insert_rows, name="insert_rows"),

    # Sharing
    path("<str:spreadsheet_id>/share/", views.share, name="share"),
    path("<str:spreadsheet_id>/share/remove/", views.remove_permission, name="remove_permission"),

    # Delete
    path("<str:spreadsheet_id>/delete/", views.delete_spreadsheet, name="delete_spreadsheet"),
]
