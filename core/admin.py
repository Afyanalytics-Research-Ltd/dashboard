"""
Django Admin registrations for core models.
"""

from django.contrib import admin
from django.utils.html import format_html

from .models import AuditLog, Client, Facility, Notification, SystemSettings


@admin.register(Client)
class ClientAdmin(admin.ModelAdmin):
    list_display = ('name', 'slug', 'logo_preview', 'active_facilities_count', 'is_active', 'created_at')
    list_filter = ('is_active', 'created_at')
    search_fields = ('name', 'slug')
    prepopulated_fields = {'slug': ('name',)}
    readonly_fields = ('created_at', 'updated_at', 'logo_preview')
    ordering = ('name',)

    fieldsets = (
        (None, {'fields': ('name', 'slug', 'is_active')}),
        ('Branding', {'fields': ('logo', 'logo_preview')}),
        ('Timestamps', {'fields': ('created_at', 'updated_at'), 'classes': ('collapse',)}),
    )

    @admin.display(description='Logo')
    def logo_preview(self, obj: Client) -> str:
        if obj.logo:
            return format_html(
                '<img src="{}" style="height:40px;border-radius:4px;" />',
                obj.logo.url,
            )
        return '—'

    @admin.display(description='Facilities')
    def active_facilities_count(self, obj: Client) -> int:
        return obj.active_facilities_count


@admin.register(Facility)
class FacilityAdmin(admin.ModelAdmin):
    list_display = ('name', 'client', 'slug', 'is_active', 'created_at')
    list_filter = ('is_active', 'client', 'created_at')
    search_fields = ('name', 'slug', 'client__name')
    prepopulated_fields = {'slug': ('name',)}
    readonly_fields = ('created_at', 'updated_at')
    raw_id_fields = ('client',)
    ordering = ('client__name', 'name')

    fieldsets = (
        (None, {'fields': ('client', 'name', 'slug', 'is_active')}),
        ('Timestamps', {'fields': ('created_at', 'updated_at'), 'classes': ('collapse',)}),
    )


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'user', 'action', 'resource', 'resource_id', 'ip_address')
    list_filter = ('action', 'timestamp')
    search_fields = ('user__username', 'resource', 'resource_id', 'ip_address')
    readonly_fields = ('user', 'action', 'resource', 'resource_id', 'detail', 'ip_address', 'user_agent', 'timestamp')
    date_hierarchy = 'timestamp'
    ordering = ('-timestamp',)

    def has_add_permission(self, request) -> bool:
        return False

    def has_change_permission(self, request, obj=None) -> bool:
        return False

    def has_delete_permission(self, request, obj=None) -> bool:
        return request.user.is_superuser


@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ('title', 'user', 'notification_type', 'is_read', 'created_at')
    list_filter = ('notification_type', 'is_read', 'created_at')
    search_fields = ('title', 'message', 'user__username')
    readonly_fields = ('created_at',)
    raw_id_fields = ('user',)
    ordering = ('-created_at',)
    actions = ['mark_as_read']

    @admin.action(description='Mark selected notifications as read')
    def mark_as_read(self, request, queryset):
        updated = queryset.update(is_read=True)
        self.message_user(request, f'{updated} notification(s) marked as read.')


@admin.register(SystemSettings)
class SystemSettingsAdmin(admin.ModelAdmin):
    list_display = ('key', 'is_public', 'updated_by', 'updated_at', 'description_short')
    list_filter = ('is_public', 'updated_at')
    search_fields = ('key', 'description')
    readonly_fields = ('updated_at', 'updated_by')
    ordering = ('key',)

    fieldsets = (
        (None, {'fields': ('key', 'value', 'description', 'is_public')}),
        ('Meta', {'fields': ('updated_by', 'updated_at'), 'classes': ('collapse',)}),
    )

    @admin.display(description='Description')
    def description_short(self, obj: SystemSettings) -> str:
        return obj.description[:80] + '…' if len(obj.description) > 80 else obj.description

    def save_model(self, request, obj, form, change):
        obj.updated_by = request.user
        super().save_model(request, obj, form, change)
