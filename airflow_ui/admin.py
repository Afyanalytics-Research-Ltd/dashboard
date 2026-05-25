"""
Airflow UI admin registration.
"""

from django.contrib import admin

from .models import DAGSummary


@admin.register(DAGSummary)
class DAGSummaryAdmin(admin.ModelAdmin):
    list_display = (
        'dag_id', 'is_active', 'is_paused',
        'last_run_state', 'last_run_at',
        'total_runs', 'successful_runs', 'failed_runs',
        'success_rate_display', 'synced_at',
    )
    list_filter = ('is_active', 'is_paused', 'last_run_state')
    search_fields = ('dag_id', 'description')
    readonly_fields = ('synced_at',)
    ordering = ('dag_id',)

    @admin.display(description='Success Rate %')
    def success_rate_display(self, obj: DAGSummary) -> str:
        return f'{obj.success_rate}%'
