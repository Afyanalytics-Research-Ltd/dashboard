"""Django admin configuration for the warehouse module."""

from django.contrib import admin
from django.utils.html import format_html

from .models import (
    Artifact,
    ChatMessage,
    Conversation,
    SnowflakeQueryLog,
    TrackedSpreadsheet,
    Workbook,
)


@admin.register(TrackedSpreadsheet)
class TrackedSpreadsheetAdmin(admin.ModelAdmin):
    list_display = [
        "title",
        "spreadsheet_id_short",
        "client",
        "created_by",
        "updated_at",
        "google_link",
    ]
    list_filter = ["client", "created_at"]
    search_fields = ["title", "spreadsheet_id", "client__name", "created_by__username"]
    readonly_fields = ["spreadsheet_id", "created_at", "updated_at", "google_link"]
    raw_id_fields = ["client", "created_by"]
    ordering = ["-updated_at"]
    date_hierarchy = "created_at"

    fieldsets = [
        ("Spreadsheet", {
            "fields": ["spreadsheet_id", "title", "web_view_link", "google_link"],
        }),
        ("Ownership", {
            "fields": ["client", "created_by"],
        }),
        ("Timestamps", {
            "fields": ["created_at", "updated_at"],
            "classes": ["collapse"],
        }),
    ]

    @admin.display(description="Spreadsheet ID")
    def spreadsheet_id_short(self, obj: TrackedSpreadsheet) -> str:
        sid = obj.spreadsheet_id
        return sid[:20] + "…" if len(sid) > 20 else sid

    @admin.display(description="Open in Google")
    def google_link(self, obj: TrackedSpreadsheet) -> str:
        if obj.web_view_link:
            return format_html(
                '<a href="{}" target="_blank" rel="noopener">Open in Google Sheets</a>',
                obj.web_view_link,
            )
        return "—"


@admin.register(SnowflakeQueryLog)
class SnowflakeQueryLogAdmin(admin.ModelAdmin):
    list_display = [
        "user",
        "status_badge",
        "rows_returned",
        "execution_time_ms",
        "created_at",
        "query_preview",
    ]
    list_filter = ["status", "user", "created_at"]
    search_fields = ["user__username", "query"]
    readonly_fields = ["user", "query", "status", "rows_returned",
                       "execution_time_ms", "error_message", "created_at"]
    ordering = ["-created_at"]
    date_hierarchy = "created_at"

    fieldsets = [
        ("Query", {
            "fields": ["user", "query"],
        }),
        ("Result", {
            "fields": ["status", "rows_returned", "execution_time_ms", "error_message"],
        }),
        ("Timestamp", {
            "fields": ["created_at"],
        }),
    ]

    def has_add_permission(self, request) -> bool:
        return False

    def has_change_permission(self, request, obj=None) -> bool:
        return False

    @admin.display(description="Status")
    def status_badge(self, obj: SnowflakeQueryLog) -> str:
        colors = {"success": "green", "error": "red", "pending": "orange"}
        color = colors.get(obj.status, "grey")
        return format_html(
            '<span style="color:{};font-weight:bold;">{}</span>',
            color,
            obj.get_status_display(),
        )

    @admin.display(description="Query preview")
    def query_preview(self, obj: SnowflakeQueryLog) -> str:
        q = obj.query.strip().replace("\n", " ")
        return q[:80] + "…" if len(q) > 80 else q


# ──────────────────────────────────────────── spreadsheet analyst

class ChatMessageInline(admin.TabularInline):
    model = ChatMessage
    extra = 0
    readonly_fields = ("role", "content", "tool_calls", "created_at")
    can_delete = False


@admin.register(Workbook)
class WorkbookAdmin(admin.ModelAdmin):
    list_display = ("original_name", "owner", "source_type", "uploaded_at", "has_loaded")
    list_filter = ("uploaded_at", "source_type")
    search_fields = ("original_name", "google_sheet_id")
    readonly_fields = ("overview", "load_error", "uploaded_at")

    @admin.display(boolean=True, description="loaded")
    def has_loaded(self, obj: Workbook) -> bool:
        return not obj.load_error


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ("__str__", "workbook", "owner", "updated_at")
    list_filter = ("updated_at",)
    inlines = [ChatMessageInline]
    readonly_fields = ("transcript", "created_at", "updated_at")


@admin.register(Artifact)
class ArtifactAdmin(admin.ModelAdmin):
    list_display = ("title", "kind", "conversation", "created_at")
    list_filter = ("kind", "created_at")
