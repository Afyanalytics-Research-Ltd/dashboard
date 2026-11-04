from django.contrib import admin

from .models import DetectorRun, FindingTriage, QueryRun, Worksheet


@admin.register(Worksheet)
class WorksheetAdmin(admin.ModelAdmin):
    list_display = ('title', 'owner', 'folder', 'starred', 'shared', 'updated_at')
    list_filter = ('starred', 'shared')
    search_fields = ('title', 'sql', 'owner__username')


@admin.register(QueryRun)
class QueryRunAdmin(admin.ModelAdmin):
    list_display = ('created_at', 'user', 'kind', 'status', 'query_id', 'rows', 'elapsed_ms')
    list_filter = ('kind', 'status')
    search_fields = ('sql', 'query_id', 'user__username')
    readonly_fields = [f.name for f in QueryRun._meta.fields]


@admin.register(FindingTriage)
class FindingTriageAdmin(admin.ModelAdmin):
    list_display = ('finding_id', 'status', 'assignee', 'updated_by', 'updated_at')
    list_filter = ('status',)


@admin.register(DetectorRun)
class DetectorRunAdmin(admin.ModelAdmin):
    list_display = ('created_at', 'key', 'user', 'status', 'query_id')
    list_filter = ('key', 'status')
