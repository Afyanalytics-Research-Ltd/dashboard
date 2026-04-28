from django.urls import path
from . import views

urlpatterns = [
    path("snowflake/query/", views.snowflake_query_view, name="snowflake_query"),
    path("", views.index, name="warehouse_index"),
    path("<str:spreadsheet_id>/", views.detail, name="detail"),

    # Values
    path("<str:spreadsheet_id>/values/read/", views.read_values, name="read-values"),
    path("<str:spreadsheet_id>/values/update/", views.update_values, name="update-values"),
    path("<str:spreadsheet_id>/values/append/", views.append_values, name="append-values"),
    path("<str:spreadsheet_id>/values/clear/", views.clear_range, name="clear-range"),
    path("<str:spreadsheet_id>/values/batch/", views.batch_update_values, name="batch-update"),

    # Tabs
    path("<str:spreadsheet_id>/tabs/add/", views.add_tab, name="add-tab"),
    path("<str:spreadsheet_id>/tabs/rename/", views.rename_tab, name="rename-tab"),
    path("<str:spreadsheet_id>/tabs/delete/", views.delete_tab, name="delete-tab"),

    # Formatting
    path("<str:spreadsheet_id>/format/", views.format_cells, name="format-cells"),
    path("<str:spreadsheet_id>/freeze-rows/", views.freeze_rows, name="freeze-rows"),

    # Rows
    path("<str:spreadsheet_id>/rows/delete/", views.delete_rows, name="delete-rows"),
    path("<str:spreadsheet_id>/rows/insert/", views.insert_rows, name="insert-rows"),

    # Sharing
    path("<str:spreadsheet_id>/share/", views.share, name="share"),
    path("<str:spreadsheet_id>/share/remove/", views.remove_permission, name="remove-permission"),

    # Delete
    path("<str:spreadsheet_id>/delete/", views.delete_spreadsheet, name="delete-spreadsheet"),


]