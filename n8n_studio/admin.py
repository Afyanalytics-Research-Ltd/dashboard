from django.contrib import admin

from .models import WorkflowSnapshot


@admin.register(WorkflowSnapshot)
class WorkflowSnapshotAdmin(admin.ModelAdmin):
    list_display = ('name', 'workflow_id', 'source', 'node_count', 'active', 'user', 'created_at')
    list_filter = ('source', 'active', 'created_at')
    search_fields = ('name', 'workflow_id', 'note')
    readonly_fields = [f.name for f in WorkflowSnapshot._meta.fields]
