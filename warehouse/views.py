from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from warehouse.services.snowflake import SnowflakeClient
import pandas as pd
from django.contrib import messages
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views.decorators.http import require_http_methods, require_POST

from .forms import (
    AddTabForm,
    AppendValuesForm,
    BatchUpdateForm,
    ClearRangeForm,
    CreateSpreadsheetForm,
    DeleteRowsForm,
    DeleteSpreadsheetForm,
    DeleteTabForm,
    FormatCellsForm,
    FreezeRowsForm,
    InsertRowsForm,
    OpenSpreadsheetForm,
    ReadValuesForm,
    RemovePermissionForm,
    RenameTabForm,
    ShareForm,
    UpdateValuesForm,
    format_table_text,
)
from .models import TrackedSpreadsheet
from .sheet_service import SheetsServiceError, get_service, hex_to_rgb01


def snowflake_query_view(request):

    df = None
    error = None
    query = ""

    if request.method == "POST":
        query = request.POST.get("query")

        try:
            sf = SnowflakeClient()

            # ⚠️ basic safety guard (optional but recommended)
            if "drop" in query.lower() or "delete" in query.lower():
                raise Exception("Dangerous query not allowed")

            df = sf.query(query)

        except Exception as e:
            error = str(e)

    return render(request, "snowflake_query.html", {
        "df": df.to_html(classes="table table-striped") if df is not None else None,
        "error": error,
        "query": query
    })


# ============================================================ helpers
def _detail_url(spreadsheet_id: str) -> str:
    return reverse("sheets:detail", args=[spreadsheet_id])


def _track(spreadsheet_id: str, title: str = "", web_view_link: str = "") -> None:
    """Record (or refresh) a row in our local index of touched sheets."""
    defaults = {}
    if title:
        defaults["title"] = title
    if web_view_link:
        defaults["web_view_link"] = web_view_link
    TrackedSpreadsheet.objects.update_or_create(
        spreadsheet_id=spreadsheet_id, defaults=defaults
    )


# ============================================================ home
@require_http_methods(["GET", "POST"])
def index(request: HttpRequest) -> HttpResponse:
    """Home page: create a new sheet, open an existing one, or pick a recent one."""
    create_form = CreateSpreadsheetForm()
    open_form = OpenSpreadsheetForm()

    if request.method == "POST":
        action = request.POST.get("action")
        if action == "create":
            create_form = CreateSpreadsheetForm(request.POST)
            if create_form.is_valid():
                try:
                    result = get_service().create_spreadsheet(
                        title=create_form.cleaned_data["title"],
                        sheet_titles=create_form.cleaned_data["sheet_titles"],
                    )
                except SheetsServiceError as e:
                    messages.error(request, f"Google Sheets API error: {e}")
                else:
                    sid = result["spreadsheetId"]
                    _track(
                        sid,
                        title=result.get("properties", {}).get("title", ""),
                        web_view_link=result.get("spreadsheetUrl", ""),
                    )
                    messages.success(request, f"Created “{create_form.cleaned_data['title']}”.")
                    return redirect(_detail_url(sid))
        elif action == "open":
            open_form = OpenSpreadsheetForm(request.POST)
            if open_form.is_valid():
                sid = open_form.cleaned_data["id_or_url"]
                return redirect(_detail_url(sid))

    recent = TrackedSpreadsheet.objects.all()[:25]
    return render(
        request,
        "sheets/index.html",
        {"create_form": create_form, "open_form": open_form, "recent": recent},
    )


# ============================================================ detail
def _detail_context(request: HttpRequest, spreadsheet_id: str, **overrides) -> dict:
    """Shared detail-page context: meta + tabs + permissions + every form."""
    svc = get_service()
    try:
        meta = svc.get_spreadsheet(spreadsheet_id)
    except SheetsServiceError as e:
        meta = {"error": str(e)}

    try:
        permissions = svc.list_permissions(spreadsheet_id)
    except SheetsServiceError:
        permissions = []

    tabs = [s["properties"] for s in meta.get("sheets", [])] if "sheets" in meta else []
    title = meta.get("properties", {}).get("title", "")
    web_view_link = meta.get("spreadsheetUrl", "")
    if title or web_view_link:
        _track(spreadsheet_id, title=title, web_view_link=web_view_link)

    ctx = {
        "spreadsheet_id": spreadsheet_id,
        "meta": meta,
        "title": title,
        "web_view_link": web_view_link,
        "tabs": tabs,
        "permissions": permissions,

        "read_form": ReadValuesForm(),
        "update_form": UpdateValuesForm(),
        "append_form": AppendValuesForm(),
        "clear_form": ClearRangeForm(),
        "batch_form": BatchUpdateForm(),
        "add_tab_form": AddTabForm(),
        "rename_tab_form": RenameTabForm(),
        "delete_tab_form": DeleteTabForm(),
        "format_form": FormatCellsForm(),
        "freeze_form": FreezeRowsForm(),
        "delete_rows_form": DeleteRowsForm(),
        "insert_rows_form": InsertRowsForm(),
        "share_form": ShareForm(),
        "delete_spreadsheet_form": DeleteSpreadsheetForm(),
        "read_result": None,
        "active_tab": "values",
    }
    ctx.update(overrides)
    return ctx


@require_http_methods(["GET"])
def detail(request: HttpRequest, spreadsheet_id: str) -> HttpResponse:
    return render(request, "sheets/detail.html", _detail_context(request, spreadsheet_id))


# ============================================================ values
@require_http_methods(["POST"])
def read_values(request: HttpRequest, spreadsheet_id: str) -> HttpResponse:
    form = ReadValuesForm(request.POST)
    overrides: dict = {"read_form": form, "active_tab": "values"}
    if form.is_valid():
        try:
            values = get_service().read_values(
                spreadsheet_id, form.cleaned_data["range"]
            )
        except SheetsServiceError as e:
            messages.error(request, f"Read failed: {e}")
        else:
            overrides["read_result"] = {
                "range": form.cleaned_data["range"],
                "values": values,
                "preview_text": format_table_text(values),
            }
            messages.success(request, f"Read {sum(len(r) for r in values)} cell(s).")
    return render(request, "sheets/detail.html", _detail_context(request, spreadsheet_id, **overrides))


@require_POST
def update_values(request: HttpRequest, spreadsheet_id: str) -> HttpResponse:
    form = UpdateValuesForm(request.POST)
    if form.is_valid():
        try:
            res = get_service().update_values(
                spreadsheet_id,
                range_a1=form.cleaned_data["range"],
                values=form.cleaned_data["values"],
                value_input_option=form.cleaned_data["value_input_option"],
            )
        except SheetsServiceError as e:
            messages.error(request, f"Update failed: {e}")
        else:
            messages.success(
                request,
                f"Updated {res.get('updatedCells', 0)} cell(s) in {res.get('updatedRange', form.cleaned_data['range'])}.",
            )
    else:
        messages.error(request, "Update form has errors. See below.")
        return render(
            request, "sheets/detail.html",
            _detail_context(request, spreadsheet_id, update_form=form, active_tab="values"),
        )
    return redirect(_detail_url(spreadsheet_id))


@require_POST
def append_values(request: HttpRequest, spreadsheet_id: str) -> HttpResponse:
    form = AppendValuesForm(request.POST)
    if form.is_valid():
        try:
            res = get_service().append_values(
                spreadsheet_id,
                range_a1=form.cleaned_data["range"],
                values=form.cleaned_data["values"],
                value_input_option=form.cleaned_data["value_input_option"],
                insert_data_option=form.cleaned_data["insert_data_option"],
            )
        except SheetsServiceError as e:
            messages.error(request, f"Append failed: {e}")
        else:
            updates = res.get("updates", {})
            messages.success(
                request,
                f"Appended {updates.get('updatedRows', 0)} row(s) to {updates.get('updatedRange', form.cleaned_data['range'])}.",
            )
    else:
        messages.error(request, "Append form has errors. See below.")
        return render(
            request, "sheets/detail.html",
            _detail_context(request, spreadsheet_id, append_form=form, active_tab="values"),
        )
    return redirect(_detail_url(spreadsheet_id))


@require_POST
def clear_range(request: HttpRequest, spreadsheet_id: str) -> HttpResponse:
    form = ClearRangeForm(request.POST)
    if form.is_valid():
        try:
            get_service().clear_values(spreadsheet_id, form.cleaned_data["range"])
        except SheetsServiceError as e:
            messages.error(request, f"Clear failed: {e}")
        else:
            messages.success(request, f"Cleared {form.cleaned_data['range']}.")
    else:
        messages.error(request, "Clear form invalid.")
    return redirect(_detail_url(spreadsheet_id))


@require_POST
def batch_update_values(request: HttpRequest, spreadsheet_id: str) -> HttpResponse:
    form = BatchUpdateForm(request.POST)
    if form.is_valid():
        try:
            res = get_service().batch_update_values(
                spreadsheet_id,
                data=form.cleaned_data["blocks"],
                value_input_option=form.cleaned_data["value_input_option"],
            )
        except SheetsServiceError as e:
            messages.error(request, f"Batch update failed: {e}")
        else:
            messages.success(
                request,
                f"Batch update OK ({res.get('totalUpdatedCells', 0)} cell(s) total).",
            )
    else:
        messages.error(request, "Batch update form has errors. See below.")
        return render(
            request, "sheets/detail.html",
            _detail_context(request, spreadsheet_id, batch_form=form, active_tab="batch"),
        )
    return redirect(_detail_url(spreadsheet_id))


# ============================================================ tabs
@require_POST
def add_tab(request: HttpRequest, spreadsheet_id: str) -> HttpResponse:
    form = AddTabForm(request.POST)
    if form.is_valid():
        try:
            get_service().add_sheet(spreadsheet_id, form.cleaned_data["title"])
        except SheetsServiceError as e:
            messages.error(request, f"Add tab failed: {e}")
        else:
            messages.success(request, f"Added tab “{form.cleaned_data['title']}”.")
    else:
        messages.error(request, "Add tab form invalid.")
    return redirect(_detail_url(spreadsheet_id))


@require_POST
def rename_tab(request: HttpRequest, spreadsheet_id: str) -> HttpResponse:
    form = RenameTabForm(request.POST)
    if form.is_valid():
        try:
            get_service().rename_sheet(
                spreadsheet_id,
                form.cleaned_data["sheet_id"],
                form.cleaned_data["new_title"],
            )
        except SheetsServiceError as e:
            messages.error(request, f"Rename failed: {e}")
        else:
            messages.success(request, f"Renamed tab to “{form.cleaned_data['new_title']}”.")
    else:
        messages.error(request, "Rename tab form invalid.")
    return redirect(_detail_url(spreadsheet_id))


@require_POST
def delete_tab(request: HttpRequest, spreadsheet_id: str) -> HttpResponse:
    form = DeleteTabForm(request.POST)
    if form.is_valid():
        try:
            get_service().delete_sheet(spreadsheet_id, form.cleaned_data["sheet_id"])
        except SheetsServiceError as e:
            messages.error(request, f"Delete tab failed: {e}")
        else:
            messages.success(request, "Deleted tab.")
    else:
        messages.error(request, "Delete tab form invalid.")
    return redirect(_detail_url(spreadsheet_id))


# ============================================================ formatting
@require_POST
def format_cells(request: HttpRequest, spreadsheet_id: str) -> HttpResponse:
    form = FormatCellsForm(request.POST)
    if form.is_valid():
        c = form.cleaned_data
        nf = None
        if c.get("number_format_type"):
            nf = {"type": c["number_format_type"]}
            if c.get("number_format_pattern"):
                nf["pattern"] = c["number_format_pattern"]
        try:
            get_service().format_cells(
                spreadsheet_id,
                sheet_id=c["sheet_id"],
                start_row=c["start_row"],
                end_row=c["end_row"],
                start_col=c["start_col"],
                end_col=c["end_col"],
                bold=c.get("bold") or None,
                italic=c.get("italic") or None,
                font_size=c.get("font_size") or None,
                background_rgb=hex_to_rgb01(c["background_hex"]) if c.get("background_hex") else None,
                foreground_rgb=hex_to_rgb01(c["foreground_hex"]) if c.get("foreground_hex") else None,
                horizontal_alignment=c.get("horizontal_alignment") or None,
                number_format=nf,
            )
        except SheetsServiceError as e:
            messages.error(request, f"Format failed: {e}")
        except ValueError as e:
            messages.error(request, str(e))
        else:
            messages.success(request, "Formatting applied.")
    else:
        messages.error(request, "Formatting form has errors. See below.")
        return render(
            request, "sheets/detail.html",
            _detail_context(request, spreadsheet_id, format_form=form, active_tab="format"),
        )
    return redirect(_detail_url(spreadsheet_id))


@require_POST
def freeze_rows(request: HttpRequest, spreadsheet_id: str) -> HttpResponse:
    form = FreezeRowsForm(request.POST)
    if form.is_valid():
        try:
            get_service().freeze_rows(
                spreadsheet_id,
                form.cleaned_data["sheet_id"],
                form.cleaned_data["row_count"],
            )
        except SheetsServiceError as e:
            messages.error(request, f"Freeze failed: {e}")
        else:
            messages.success(
                request,
                f"Frozen first {form.cleaned_data['row_count']} row(s).",
            )
    else:
        messages.error(request, "Freeze form invalid.")
    return redirect(_detail_url(spreadsheet_id))


# ============================================================ rows
@require_POST
def delete_rows(request: HttpRequest, spreadsheet_id: str) -> HttpResponse:
    form = DeleteRowsForm(request.POST)
    if form.is_valid():
        try:
            get_service().delete_rows(
                spreadsheet_id,
                sheet_id=form.cleaned_data["sheet_id"],
                start_row=form.cleaned_data["start_row"],
                end_row=form.cleaned_data["end_row"],
            )
        except SheetsServiceError as e:
            messages.error(request, f"Delete rows failed: {e}")
        else:
            messages.success(request, "Rows deleted.")
    else:
        messages.error(request, "Delete rows form has errors. See below.")
        return render(
            request, "sheets/detail.html",
            _detail_context(request, spreadsheet_id, delete_rows_form=form, active_tab="rows"),
        )
    return redirect(_detail_url(spreadsheet_id))


@require_POST
def insert_rows(request: HttpRequest, spreadsheet_id: str) -> HttpResponse:
    form = InsertRowsForm(request.POST)
    if form.is_valid():
        try:
            get_service().insert_rows(
                spreadsheet_id,
                sheet_id=form.cleaned_data["sheet_id"],
                start_row=form.cleaned_data["start_row"],
                count=form.cleaned_data["count"],
            )
        except SheetsServiceError as e:
            messages.error(request, f"Insert rows failed: {e}")
        else:
            messages.success(request, "Rows inserted.")
    else:
        messages.error(request, "Insert rows form has errors. See below.")
        return render(
            request, "sheets/detail.html",
            _detail_context(request, spreadsheet_id, insert_rows_form=form, active_tab="rows"),
        )
    return redirect(_detail_url(spreadsheet_id))


# ============================================================ sharing
@require_POST
def share(request: HttpRequest, spreadsheet_id: str) -> HttpResponse:
    form = ShareForm(request.POST)
    if form.is_valid():
        try:
            get_service().share(
                spreadsheet_id,
                email=form.cleaned_data["email"],
                role=form.cleaned_data["role"],
                notify=form.cleaned_data.get("notify", False),
            )
        except SheetsServiceError as e:
            messages.error(request, f"Share failed: {e}")
        else:
            messages.success(
                request,
                f"Shared with {form.cleaned_data['email']} as {form.cleaned_data['role']}.",
            )
    else:
        messages.error(request, "Share form has errors. See below.")
        return render(
            request, "sheets/detail.html",
            _detail_context(request, spreadsheet_id, share_form=form, active_tab="share"),
        )
    return redirect(_detail_url(spreadsheet_id))


@require_POST
def remove_permission(request: HttpRequest, spreadsheet_id: str) -> HttpResponse:
    form = RemovePermissionForm(request.POST)
    if form.is_valid():
        try:
            get_service().remove_permission(
                spreadsheet_id, form.cleaned_data["permission_id"]
            )
        except SheetsServiceError as e:
            messages.error(request, f"Remove failed: {e}")
        else:
            messages.success(request, "Permission removed.")
    return redirect(_detail_url(spreadsheet_id))


# ============================================================ delete
@require_POST
def delete_spreadsheet(request: HttpRequest, spreadsheet_id: str) -> HttpResponse:
    form = DeleteSpreadsheetForm(request.POST)
    if form.is_valid():
        try:
            get_service().delete_spreadsheet(spreadsheet_id)
        except SheetsServiceError as e:
            messages.error(request, f"Delete failed: {e}")
            return redirect(_detail_url(spreadsheet_id))
        TrackedSpreadsheet.objects.filter(spreadsheet_id=spreadsheet_id).delete()
        messages.success(request, "Spreadsheet deleted.")
        return redirect("sheets:index")
    messages.error(request, "Confirm the checkbox to delete.")
    return redirect(_detail_url(spreadsheet_id))