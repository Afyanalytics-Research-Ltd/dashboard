"""
Django admin registrations for the agent metric catalog — a secondary/
fallback CRUD surface alongside the Semantic Layer Configuration settings page
(agents/views.py), same as every other app in this repo (see core/admin.py,
analytics_app/admin.py).
"""

from django.contrib import admin

from .models import MetricDefinition, PendingCubeMeasure


@admin.register(MetricDefinition)
class MetricDefinitionAdmin(admin.ModelAdmin):
    list_display = ('metric_id', 'name', 'is_active', 'updated_by', 'updated_at')
    list_filter = ('is_active',)
    search_fields = ('metric_id', 'name', 'description')
    readonly_fields = ('created_by', 'created_at', 'updated_by', 'updated_at')
    ordering = ('name',)

    fieldsets = (
        (None, {'fields': ('metric_id', 'name', 'description', 'cube_query', 'is_active')}),
        ('Meta', {'fields': ('created_by', 'created_at', 'updated_by', 'updated_at'), 'classes': ('collapse',)}),
    )

    def save_model(self, request, obj, form, change):
        if not change:
            obj.created_by = request.user
        obj.updated_by = request.user
        super().save_model(request, obj, form, change)


@admin.register(PendingCubeMeasure)
class PendingCubeMeasureAdmin(admin.ModelAdmin):
    list_display = ('cube_name', 'measure_name', 'measure_type', 'status', 'requested_by', 'requested_at')
    list_filter = ('status', 'measure_type')
    search_fields = ('cube_name', 'measure_name')
    readonly_fields = ('requested_by', 'requested_at', 'reviewed_by', 'reviewed_at')
    ordering = ('-requested_at',)

    fieldsets = (
        (None, {'fields': (
            'cube_name', 'measure_name', 'measure_type', 'sql_expression',
            'title', 'description', 'status', 'rejection_reason',
        )}),
        ('Review', {'fields': ('requested_by', 'requested_at', 'reviewed_by', 'reviewed_at'), 'classes': ('collapse',)}),
    )

    def save_model(self, request, obj, form, change):
        # Approving/rejecting from the admin doesn't run the
        # Snowflake-validation-then-YAML-splice pipeline that
        # agents/views.py's ApproveCubeMeasureView does — this is a fallback
        # for editing metadata (e.g. fixing a typo before review), not a
        # substitute for the "Approve" button on the settings page.
        if not change:
            obj.requested_by = request.user
        super().save_model(request, obj, form, change)
