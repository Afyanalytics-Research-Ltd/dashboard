"""
Analytics app admin registration.
"""

from django.contrib import admin

from .models import Dashboard


@admin.register(Dashboard)
class DashboardAdmin(admin.ModelAdmin):
    list_display = ('name', 'client', 'category', 'is_active', 'view_count', 'created_at')
    list_filter = ('category', 'is_active', 'client')
    search_fields = ('name', 'description', 'slug')
    prepopulated_fields = {'slug': ('name',)}
    readonly_fields = ('view_count', 'created_at', 'updated_at')
    raw_id_fields = ('client', 'facility', 'created_by')
    ordering = ('order', 'name')
