"""
Warehouse views — Google Sheets management and Snowflake query interface.

Access is restricted to Client Admin users (or superusers).  All mutating
operations return JSON so the frontend can handle responses via AJAX without
a full-page reload.
"""

import logging

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.core.mail import send_mail
from django.core.paginator import Paginator
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views import View
from django.views.decorators.http import require_POST

from authentication.module_access import MODULE_WAREHOUSE, has_module_access
from core.models import AuditLog

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
    SnowflakeQueryForm,
    UpdateValuesForm,
    format_table_text,
)
from .models import SnowflakeQueryLog, TrackedSpreadsheet
from .services.facility_scope import FacilityScopeError, get_facility_scope, validate_query_scope
from .services.snowflake import SnowflakeClient, SnowflakeQueryError
from .sheet_service import SheetsServiceError, get_service, hex_to_rgb01

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────── role helpers


def _is_warehouse_user(user) -> bool:
    """True if the user may access the warehouse.

    Default: superuser or Client Admin. A facility administrator can grant
    or revoke this per-user via the Permissions page — see
    authentication.module_access.has_module_access.
    """
    return has_module_access(user, MODULE_WAREHOUSE)


def _require_warehouse_user(request: HttpRequest) -> bool:
    return _is_warehouse_user(request.user)


# ─────────────────────────────────────────────────── shared helpers


def _track(
    spreadsheet_id: str,
    title: str = "",
    web_view_link: str = "",
    user=None,
) -> TrackedSpreadsheet:
    """Create-or-update the local TrackedSpreadsheet record."""
    defaults: dict = {}
    if title:
        defaults["title"] = title
    if web_view_link:
        defaults["web_view_link"] = web_view_link

    obj, created = TrackedSpreadsheet.objects.get_or_create(
        spreadsheet_id=spreadsheet_id,
        defaults={**defaults, "created_by": user},
    )
    if not created and defaults:
        for k, v in defaults.items():
            setattr(obj, k, v)
        obj.save(update_fields=list(defaults.keys()) + ["updated_at"])
    return obj


def _audit(request: HttpRequest, action: str, resource_id: str, detail: str = "") -> None:
    AuditLog.log(
        user=request.user,
        action=action,
        resource="TrackedSpreadsheet",
        resource_id=resource_id,
        detail=detail,
        ip_address=request.META.get("REMOTE_ADDR"),
        user_agent=request.META.get("HTTP_USER_AGENT", "")[:512],
    )


# ──────────────────────────────────────────── class-based views

class WarehouseHomeView(LoginRequiredMixin, UserPassesTestMixin, View):
    """Home page: spreadsheet list, create/open actions, Snowflake schema."""

    template_name = "warehouse/index.html"
    sidebar_section = "warehouse"

    def test_func(self) -> bool:
        return _is_warehouse_user(self.request.user)

    def get(self, request: HttpRequest) -> HttpResponse:
        return self._render(request, CreateSpreadsheetForm(), OpenSpreadsheetForm())

    def post(self, request: HttpRequest) -> HttpResponse:
        action = request.POST.get("action")
        create_form = CreateSpreadsheetForm()
        open_form = OpenSpreadsheetForm()

        if action == "create":
            create_form = CreateSpreadsheetForm(request.POST)
            if create_form.is_valid():
                try:
                    result = get_service().create_spreadsheet(
                        title=create_form.cleaned_data["title"],
                        sheet_titles=create_form.cleaned_data["sheet_titles"],
                    )
                except SheetsServiceError as exc:
                    logger.error("Create spreadsheet failed: %s", exc)
                    messages.error(request, f"Google Sheets API error: {exc}")
                else:
                    sid = result["spreadsheetId"]
                    sheet = _track(
                        sid,
                        title=result.get("properties", {}).get("title", ""),
                        web_view_link=result.get("spreadsheetUrl", ""),
                        user=request.user,
                    )
                    _audit(request, "create", sid, f"Created spreadsheet: {sheet.title}")
                    messages.success(
                        request,
                        f"Spreadsheet '{create_form.cleaned_data['title']}' created successfully.",
                    )
                    return redirect(reverse("warehouse:detail", args=[sid]))

        elif action == "open":
            open_form = OpenSpreadsheetForm(request.POST)
            if open_form.is_valid():
                sid = open_form.cleaned_data["id_or_url"]
                return redirect(reverse("warehouse:detail", args=[sid]))

        return self._render(request, create_form, open_form)

    def _render(
        self,
        request: HttpRequest,
        create_form: CreateSpreadsheetForm,
        open_form: OpenSpreadsheetForm,
    ) -> HttpResponse:
        # Server-side paginated recent spreadsheets
        qs = TrackedSpreadsheet.objects.select_related("created_by", "client").all()
        search_q = request.GET.get("q", "").strip()
        if search_q:
            qs = qs.filter(title__icontains=search_q)
        paginator = Paginator(qs, 10)
        page_number = request.GET.get("page", 1)
        recent_page = paginator.get_page(page_number)

        return render(
            request,
            self.template_name,
            {
                "create_form": create_form,
                "open_form": open_form,
                "recent_page": recent_page,
                "search_q": search_q,
                "sidebar_section": self.sidebar_section,
            },
        )


class SpreadsheetDetailView(LoginRequiredMixin, UserPassesTestMixin, View):
    """Detail view for a specific spreadsheet — tabbed interface."""

    template_name = "warehouse/detail.html"
    sidebar_section = "warehouse"

    def test_func(self) -> bool:
        return _is_warehouse_user(self.request.user)

    def get(self, request: HttpRequest, spreadsheet_id: str) -> HttpResponse:
        return render(
            request,
            self.template_name,
            self._build_context(request, spreadsheet_id),
        )

    def _build_context(
        self, request: HttpRequest, spreadsheet_id: str, **overrides
    ) -> dict:
        svc = get_service()
        meta: dict = {}
        permissions: list = []
        try:
            meta = svc.get_spreadsheet(spreadsheet_id)
        except SheetsServiceError as exc:
            meta = {"error": str(exc)}
            logger.warning("Detail: could not load %s: %s", spreadsheet_id, exc)

        try:
            permissions = svc.list_permissions(spreadsheet_id)
        except SheetsServiceError:
            permissions = []

        tabs = [s["properties"] for s in meta.get("sheets", [])] if "sheets" in meta else []
        title = meta.get("properties", {}).get("title", "")
        web_view_link = meta.get("spreadsheetUrl", "")

        if title or web_view_link:
            _track(spreadsheet_id, title=title, web_view_link=web_view_link, user=request.user)

        ctx = {
            "spreadsheet_id": spreadsheet_id,
            "meta": meta,
            "title": title,
            "web_view_link": web_view_link,
            "tabs": tabs,
            "permissions": permissions,
            "sidebar_section": self.sidebar_section,
            # forms
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
            "remove_permission_form": RemovePermissionForm(),
            "delete_spreadsheet_form": DeleteSpreadsheetForm(),
            "read_result": None,
            "active_tab": request.GET.get("tab", "values"),
        }
        ctx.update(overrides)
        return ctx


class SnowflakeQueryView(LoginRequiredMixin, UserPassesTestMixin, View):
    """Interactive Snowflake SQL query interface."""

    template_name = "warehouse/snowflake.html"
    sidebar_section = "warehouse"

    def test_func(self) -> bool:
        return _is_warehouse_user(self.request.user)

    def get(self, request: HttpRequest) -> HttpResponse:
        history = SnowflakeQueryLog.objects.filter(user=request.user)[:10]
        return render(request, self.template_name, {
            "form": SnowflakeQueryForm(),
            "history": history,
            "sidebar_section": self.sidebar_section,
        })

    def post(self, request: HttpRequest) -> HttpResponse:
        form = SnowflakeQueryForm(request.POST)
        history = SnowflakeQueryLog.objects.filter(user=request.user)[:10]
        result_html: str | None = None
        result_cols: list = []
        result_rows: list = []
        exec_stats: dict = {}
        error_msg: str | None = None

        if form.is_valid():
            sql = form.cleaned_data["query"]
            log = SnowflakeQueryLog.objects.create(
                user=request.user, query=sql, status="pending"
            )
            import time as _time
            t0 = _time.monotonic()
            try:
                scope = get_facility_scope(request.user)
                validate_query_scope(sql, scope)
                client = SnowflakeClient()
                df = client.query(sql, max_rows=10_000)
                elapsed_ms = int((_time.monotonic() - t0) * 1000)

                # Replace NaN/NaT with None for safe JSON serialisation
                import pandas as pd
                df = df.where(pd.notnull(df), None)

                result_cols = list(df.columns)
                result_rows = df.head(1000).values.tolist()
                exec_stats = {
                    "rows_returned": len(df),
                    "execution_time_ms": elapsed_ms,
                    "truncated": len(df) > 1000,
                }

                log.status = "success"
                log.rows_returned = len(df)
                log.execution_time_ms = elapsed_ms
                log.save(update_fields=["status", "rows_returned", "execution_time_ms"])

                AuditLog.log(
                    user=request.user,
                    action="query",
                    resource="Snowflake",
                    detail=f"Rows: {len(df)}, Time: {elapsed_ms}ms",
                    ip_address=request.META.get("REMOTE_ADDR"),
                )

            except (SnowflakeQueryError, Exception) as exc:
                elapsed_ms = int((_time.monotonic() - t0) * 1000)
                error_msg = str(exc)
                log.status = "error"
                log.error_message = error_msg
                log.execution_time_ms = elapsed_ms
                log.save(update_fields=["status", "error_message", "execution_time_ms"])
                logger.error("Snowflake query error: %s", exc)

        return render(request, self.template_name, {
            "form": form,
            "history": history,
            "result_cols": result_cols,
            "result_rows": result_rows,
            "exec_stats": exec_stats,
            "error_msg": error_msg,
            "sidebar_section": self.sidebar_section,
        })


class ReportMissingView(LoginRequiredMixin, UserPassesTestMixin, View):
    """Handle missing data reports (GET shows form, POST sends email)."""

    template_name = "warehouse/index.html"
    sidebar_section = "warehouse"

    def test_func(self) -> bool:
        return _is_warehouse_user(self.request.user)

    def post(self, request: HttpRequest) -> HttpResponse:
        if request.headers.get("x-requested-with") == "XMLHttpRequest":
            return self._handle_ajax(request)
        return self._handle_form(request)

    def _handle_ajax(self, request: HttpRequest) -> JsonResponse:
        schema_name = request.POST.get("schema_name", "").strip()
        table_name = request.POST.get("table_name", "").strip()
        description = request.POST.get("description", "").strip()

        if not schema_name or not table_name:
            return JsonResponse(
                {"success": False, "error": "Schema and table name are required."},
                status=400,
            )
        self._send_report(request, schema_name, table_name, description)
        return JsonResponse({"success": True})

    def _handle_form(self, request: HttpRequest) -> HttpResponse:
        schema_name = request.POST.get("schema_name", "").strip()
        table_name = request.POST.get("table_name", "").strip()
        description = request.POST.get("description", "").strip()

        if not schema_name or not table_name:
            messages.error(request, "Schema and table name are required.")
            return redirect(reverse("warehouse:index"))

        success, msg = self._send_report(request, schema_name, table_name, description)
        if success:
            messages.success(request, msg)
        else:
            messages.error(request, msg)
        return redirect(reverse("warehouse:index"))

    def _send_report(
        self,
        request: HttpRequest,
        schema_name: str,
        table_name: str,
        description: str,
    ) -> tuple[bool, str]:
        reporter = request.user.get_full_name() or request.user.username
        reported_at = timezone.now().strftime("%Y-%m-%d %H:%M UTC")
        subject = f"[Data Quality] Missing Data — {schema_name}.{table_name}"
        body = (
            f"Missing Data Report\n"
            f"===================\n\n"
            f"Reporter : {reporter}\n"
            f"Schema   : {schema_name}\n"
            f"Table    : {table_name}\n"
            f"Reported : {reported_at}\n\n"
            f"Description\n"
            f"-----------\n"
            f"{description or '(no description provided)'}\n"
        )
        try:
            send_mail(
                subject=subject,
                message=body,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[settings.DEFAULT_FROM_EMAIL],
                fail_silently=False,
            )
            AuditLog.log(
                user=request.user,
                action="trigger",
                resource="MissingDataReport",
                detail=f"{schema_name}.{table_name}",
                ip_address=request.META.get("REMOTE_ADDR"),
            )
            return True, f"Missing-data report for {schema_name}.{table_name} sent."
        except Exception as exc:
            logger.error("Failed to send missing-data report: %s", exc)
            return False, f"Failed to send report: {exc}"


# ─────────────────────────────────────────── FBV AJAX views

def _warehouse_post_only(view_func):
    """Decorator: require POST + warehouse access; return JSON errors otherwise."""
    from functools import wraps

    @wraps(view_func)
    def wrapper(request: HttpRequest, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({"error": "Authentication required."}, status=401)
        if not _is_warehouse_user(request.user):
            return JsonResponse({"error": "Permission denied."}, status=403)
        if request.method != "POST":
            return JsonResponse({"error": "Method not allowed."}, status=405)
        return view_func(request, *args, **kwargs)

    return wrapper


# ── values ────────────────────────────────────────────────────────────

@_warehouse_post_only
def read_values(request: HttpRequest, spreadsheet_id: str) -> JsonResponse:
    form = ReadValuesForm(request.POST)
    if not form.is_valid():
        return JsonResponse({"success": False, "errors": form.errors}, status=422)
    try:
        values = get_service().read_values(
            spreadsheet_id, form.cleaned_data["range_notation"]
        )
    except SheetsServiceError as exc:
        logger.error("read_values %s: %s", spreadsheet_id, exc)
        _audit(request, "read", spreadsheet_id, f"Error: {exc}")
        return JsonResponse({"success": False, "error": str(exc)}, status=400)

    _audit(request, "read", spreadsheet_id, f"Read {form.cleaned_data['range_notation']}")
    return JsonResponse({
        "success": True,
        "range": form.cleaned_data["range_notation"],
        "values": values,
        "preview_text": format_table_text(values),
        "row_count": len(values),
    })


@_warehouse_post_only
def update_values(request: HttpRequest, spreadsheet_id: str) -> JsonResponse:
    form = UpdateValuesForm(request.POST)
    if not form.is_valid():
        return JsonResponse({"success": False, "errors": form.errors}, status=422)
    try:
        res = get_service().update_values(
            spreadsheet_id,
            range_a1=form.cleaned_data["range_notation"],
            values=form.cleaned_data["values"],
            value_input_option=form.cleaned_data["value_input_option"],
        )
    except SheetsServiceError as exc:
        logger.error("update_values %s: %s", spreadsheet_id, exc)
        _audit(request, "update", spreadsheet_id, f"Error: {exc}")
        return JsonResponse({"success": False, "error": str(exc)}, status=400)

    _audit(request, "update", spreadsheet_id, f"Updated {res.get('updatedRange', '')}")
    return JsonResponse({
        "success": True,
        "updated_cells": res.get("updatedCells", 0),
        "updated_range": res.get("updatedRange", ""),
    })


@_warehouse_post_only
def append_values(request: HttpRequest, spreadsheet_id: str) -> JsonResponse:
    form = AppendValuesForm(request.POST)
    if not form.is_valid():
        return JsonResponse({"success": False, "errors": form.errors}, status=422)
    try:
        res = get_service().append_values(
            spreadsheet_id,
            range_a1=form.cleaned_data["range_notation"],
            values=form.cleaned_data["values"],
            value_input_option=form.cleaned_data["value_input_option"],
            insert_data_option=form.cleaned_data["insert_data_option"],
        )
    except SheetsServiceError as exc:
        logger.error("append_values %s: %s", spreadsheet_id, exc)
        _audit(request, "update", spreadsheet_id, f"Append error: {exc}")
        return JsonResponse({"success": False, "error": str(exc)}, status=400)

    updates = res.get("updates", {})
    _audit(request, "update", spreadsheet_id,
           f"Appended {updates.get('updatedRows', 0)} row(s)")
    return JsonResponse({
        "success": True,
        "updated_rows": updates.get("updatedRows", 0),
        "updated_range": updates.get("updatedRange", ""),
    })


@_warehouse_post_only
def clear_range(request: HttpRequest, spreadsheet_id: str) -> JsonResponse:
    form = ClearRangeForm(request.POST)
    if not form.is_valid():
        return JsonResponse({"success": False, "errors": form.errors}, status=422)
    try:
        get_service().clear_values(spreadsheet_id, form.cleaned_data["range_notation"])
    except SheetsServiceError as exc:
        logger.error("clear_range %s: %s", spreadsheet_id, exc)
        _audit(request, "update", spreadsheet_id, f"Clear error: {exc}")
        return JsonResponse({"success": False, "error": str(exc)}, status=400)

    _audit(request, "update", spreadsheet_id, f"Cleared {form.cleaned_data['range_notation']}")
    return JsonResponse({"success": True, "cleared_range": form.cleaned_data["range_notation"]})


@_warehouse_post_only
def batch_update_values(request: HttpRequest, spreadsheet_id: str) -> JsonResponse:
    form = BatchUpdateForm(request.POST)
    if not form.is_valid():
        return JsonResponse({"success": False, "errors": form.errors}, status=422)
    try:
        res = get_service().batch_update_values(
            spreadsheet_id,
            data=form.cleaned_data["multi_block"],
            value_input_option=form.cleaned_data["value_input_option"],
        )
    except SheetsServiceError as exc:
        logger.error("batch_update_values %s: %s", spreadsheet_id, exc)
        _audit(request, "update", spreadsheet_id, f"Batch error: {exc}")
        return JsonResponse({"success": False, "error": str(exc)}, status=400)

    total_cells = res.get("totalUpdatedCells", 0)
    _audit(request, "update", spreadsheet_id, f"Batch update: {total_cells} cells")
    return JsonResponse({"success": True, "total_updated_cells": total_cells})


# ── tabs ──────────────────────────────────────────────────────────────

@_warehouse_post_only
def add_tab(request: HttpRequest, spreadsheet_id: str) -> JsonResponse:
    form = AddTabForm(request.POST)
    if not form.is_valid():
        return JsonResponse({"success": False, "errors": form.errors}, status=422)
    try:
        get_service().add_sheet(spreadsheet_id, form.cleaned_data["tab_title"])
    except SheetsServiceError as exc:
        logger.error("add_tab %s: %s", spreadsheet_id, exc)
        _audit(request, "create", spreadsheet_id, f"Add tab error: {exc}")
        return JsonResponse({"success": False, "error": str(exc)}, status=400)

    _audit(request, "create", spreadsheet_id, f"Added tab: {form.cleaned_data['tab_title']}")
    return JsonResponse({"success": True, "tab_title": form.cleaned_data["tab_title"]})


@_warehouse_post_only
def rename_tab(request: HttpRequest, spreadsheet_id: str) -> JsonResponse:
    form = RenameTabForm(request.POST)
    if not form.is_valid():
        return JsonResponse({"success": False, "errors": form.errors}, status=422)
    try:
        get_service().rename_sheet(
            spreadsheet_id,
            form.cleaned_data["sheet_id"],
            form.cleaned_data["new_title"],
        )
    except SheetsServiceError as exc:
        logger.error("rename_tab %s: %s", spreadsheet_id, exc)
        _audit(request, "update", spreadsheet_id, f"Rename tab error: {exc}")
        return JsonResponse({"success": False, "error": str(exc)}, status=400)

    _audit(request, "update", spreadsheet_id, f"Renamed tab to: {form.cleaned_data['new_title']}")
    return JsonResponse({"success": True, "new_title": form.cleaned_data["new_title"]})


@_warehouse_post_only
def delete_tab(request: HttpRequest, spreadsheet_id: str) -> JsonResponse:
    form = DeleteTabForm(request.POST)
    if not form.is_valid():
        return JsonResponse({"success": False, "errors": form.errors}, status=422)
    try:
        get_service().delete_sheet(spreadsheet_id, form.cleaned_data["sheet_id"])
    except SheetsServiceError as exc:
        logger.error("delete_tab %s: %s", spreadsheet_id, exc)
        _audit(request, "delete", spreadsheet_id, f"Delete tab error: {exc}")
        return JsonResponse({"success": False, "error": str(exc)}, status=400)

    _audit(request, "delete", spreadsheet_id, f"Deleted tab id={form.cleaned_data['sheet_id']}")
    return JsonResponse({"success": True})


# ── formatting ────────────────────────────────────────────────────────

@_warehouse_post_only
def format_cells(request: HttpRequest, spreadsheet_id: str) -> JsonResponse:
    form = FormatCellsForm(request.POST)
    if not form.is_valid():
        return JsonResponse({"success": False, "errors": form.errors}, status=422)
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
    except (SheetsServiceError, ValueError) as exc:
        logger.error("format_cells %s: %s", spreadsheet_id, exc)
        _audit(request, "update", spreadsheet_id, f"Format error: {exc}")
        return JsonResponse({"success": False, "error": str(exc)}, status=400)

    _audit(request, "update", spreadsheet_id, "Applied cell formatting")
    return JsonResponse({"success": True})


@_warehouse_post_only
def freeze_rows(request: HttpRequest, spreadsheet_id: str) -> JsonResponse:
    form = FreezeRowsForm(request.POST)
    if not form.is_valid():
        return JsonResponse({"success": False, "errors": form.errors}, status=422)
    try:
        get_service().freeze_rows(
            spreadsheet_id,
            form.cleaned_data["sheet_id"],
            form.cleaned_data["row_count"],
        )
    except SheetsServiceError as exc:
        logger.error("freeze_rows %s: %s", spreadsheet_id, exc)
        _audit(request, "update", spreadsheet_id, f"Freeze error: {exc}")
        return JsonResponse({"success": False, "error": str(exc)}, status=400)

    _audit(request, "update", spreadsheet_id,
           f"Froze {form.cleaned_data['row_count']} rows on sheet {form.cleaned_data['sheet_id']}")
    return JsonResponse({"success": True, "row_count": form.cleaned_data["row_count"]})


# ── rows ──────────────────────────────────────────────────────────────

@_warehouse_post_only
def delete_rows(request: HttpRequest, spreadsheet_id: str) -> JsonResponse:
    form = DeleteRowsForm(request.POST)
    if not form.is_valid():
        return JsonResponse({"success": False, "errors": form.errors}, status=422)
    try:
        get_service().delete_rows(
            spreadsheet_id,
            sheet_id=form.cleaned_data["sheet_id"],
            start_row=form.cleaned_data["start_row"],
            end_row=form.cleaned_data["end_row"],
        )
    except SheetsServiceError as exc:
        logger.error("delete_rows %s: %s", spreadsheet_id, exc)
        _audit(request, "delete", spreadsheet_id, f"Delete rows error: {exc}")
        return JsonResponse({"success": False, "error": str(exc)}, status=400)

    count = form.cleaned_data["end_row"] - form.cleaned_data["start_row"]
    _audit(request, "delete", spreadsheet_id, f"Deleted {count} rows")
    return JsonResponse({"success": True, "rows_deleted": count})


@_warehouse_post_only
def insert_rows(request: HttpRequest, spreadsheet_id: str) -> JsonResponse:
    form = InsertRowsForm(request.POST)
    if not form.is_valid():
        return JsonResponse({"success": False, "errors": form.errors}, status=422)
    try:
        get_service().insert_rows(
            spreadsheet_id,
            sheet_id=form.cleaned_data["sheet_id"],
            start_row=form.cleaned_data["start_row"],
            count=form.cleaned_data["count"],
        )
    except SheetsServiceError as exc:
        logger.error("insert_rows %s: %s", spreadsheet_id, exc)
        _audit(request, "update", spreadsheet_id, f"Insert rows error: {exc}")
        return JsonResponse({"success": False, "error": str(exc)}, status=400)

    _audit(request, "update", spreadsheet_id,
           f"Inserted {form.cleaned_data['count']} rows at {form.cleaned_data['start_row']}")
    return JsonResponse({"success": True, "rows_inserted": form.cleaned_data["count"]})


# ── sharing ───────────────────────────────────────────────────────────

@_warehouse_post_only
def share(request: HttpRequest, spreadsheet_id: str) -> JsonResponse:
    form = ShareForm(request.POST)
    if not form.is_valid():
        return JsonResponse({"success": False, "errors": form.errors}, status=422)
    try:
        result = get_service().share(
            spreadsheet_id,
            email=form.cleaned_data["email"],
            role=form.cleaned_data["role"],
            notify=form.cleaned_data.get("notify", False),
        )
    except SheetsServiceError as exc:
        logger.error("share %s: %s", spreadsheet_id, exc)
        _audit(request, "share", spreadsheet_id, f"Share error: {exc}")
        return JsonResponse({"success": False, "error": str(exc)}, status=400)

    _audit(request, "share", spreadsheet_id,
           f"Shared with {form.cleaned_data['email']} as {form.cleaned_data['role']}")
    return JsonResponse({
        "success": True,
        "permission_id": result.get("id"),
        "email": form.cleaned_data["email"],
        "role": form.cleaned_data["role"],
    })


@_warehouse_post_only
def remove_permission(request: HttpRequest, spreadsheet_id: str) -> JsonResponse:
    form = RemovePermissionForm(request.POST)
    if not form.is_valid():
        return JsonResponse({"success": False, "errors": form.errors}, status=422)
    try:
        get_service().remove_permission(spreadsheet_id, form.cleaned_data["permission_id"])
    except SheetsServiceError as exc:
        logger.error("remove_permission %s: %s", spreadsheet_id, exc)
        _audit(request, "delete", spreadsheet_id, f"Remove permission error: {exc}")
        return JsonResponse({"success": False, "error": str(exc)}, status=400)

    _audit(request, "delete", spreadsheet_id,
           f"Removed permission {form.cleaned_data['permission_id']}")
    return JsonResponse({"success": True})


# ── delete spreadsheet ────────────────────────────────────────────────

@_warehouse_post_only
def delete_spreadsheet(request: HttpRequest, spreadsheet_id: str) -> JsonResponse:
    form = DeleteSpreadsheetForm(request.POST)
    if not form.is_valid():
        return JsonResponse({"success": False, "errors": form.errors}, status=422)
    try:
        get_service().delete_spreadsheet(spreadsheet_id)
    except SheetsServiceError as exc:
        logger.error("delete_spreadsheet %s: %s", spreadsheet_id, exc)
        _audit(request, "delete", spreadsheet_id, f"Delete error: {exc}")
        return JsonResponse({"success": False, "error": str(exc)}, status=400)

    TrackedSpreadsheet.objects.filter(spreadsheet_id=spreadsheet_id).delete()
    _audit(request, "delete", spreadsheet_id, "Spreadsheet permanently deleted")
    return JsonResponse({
        "success": True,
        "redirect_url": reverse("warehouse:index"),
    })
