from django.contrib import admin

from .models import ChatMessage, ChatSession


class ChatMessageInline(admin.TabularInline):
    model = ChatMessage
    extra = 0
    readonly_fields = ('role', 'content', 'query_intent', 'created_at')
    can_delete = False


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ('title', 'user', 'session_key', 'is_active', 'started_at', 'last_activity')
    list_filter = ('is_active', 'started_at')
    search_fields = ('title', 'user__username', 'session_key')
    readonly_fields = ('session_key', 'started_at', 'last_activity')
    inlines = [ChatMessageInline]
