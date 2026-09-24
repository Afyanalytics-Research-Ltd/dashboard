from django.contrib import admin

from .models import BrowserChatSession, ScheduledBrowserTask


@admin.register(ScheduledBrowserTask)
class ScheduledBrowserTaskAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'is_active', 'last_run_status', 'last_run_at')
    list_filter = ('is_active', 'last_run_status')
    search_fields = ('name', 'instructions')
    readonly_fields = ('last_run_at', 'last_run_status', 'last_run_detail', 'created_at', 'updated_at')
    ordering = ('-created_at',)


@admin.register(BrowserChatSession)
class BrowserChatSessionAdmin(admin.ModelAdmin):
    list_display = ('session_key', 'user', 'browserbase_session_id', 'started_at', 'last_activity')
    search_fields = ('session_key', 'browserbase_session_id')
    readonly_fields = ('session_key', 'started_at', 'last_activity')
    ordering = ('-started_at',)
