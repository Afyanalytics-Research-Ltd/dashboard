from django.contrib import admin

from .models import Connection, Job, Mapping, Snapshot, Source


@admin.register(Source)
class SourceAdmin(admin.ModelAdmin):
    list_display = ("name", "kind", "status", "last_tested_at", "updated_at")
    list_filter = ("kind", "status")
    search_fields = ("name",)
    exclude = ("secrets",)


@admin.register(Snapshot)
class SnapshotAdmin(admin.ModelAdmin):
    list_display = ("id", "source", "fingerprint", "created_at")
    exclude = ("data",)


@admin.register(Mapping)
class MappingAdmin(admin.ModelAdmin):
    list_display = ("name", "source", "target", "status", "version", "updated_at")
    list_filter = ("status",)
    exclude = ("document",)


@admin.register(Connection)
class ConnectionAdmin(admin.ModelAdmin):
    list_display = ("name", "mapping", "destination", "mode", "schedule_minutes", "enabled", "last_status", "last_run_at")


@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = ("id", "kind", "title", "status", "progress", "created_at", "finished_at")
    list_filter = ("kind", "status")
    exclude = ("log",)
