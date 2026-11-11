import uuid

from django.conf import settings
from django.db import models
from django_celery_beat.models import PeriodicTask


class BrowserChatSession(models.Model):
    """One browser-agent chat conversation.

    The live MCP connection + Browserbase browser session backing this chat
    only exists in-process for the lifetime of the WebSocket connection (see
    browser_automation/consumers.py) — it is NOT resumed across reconnects
    or page refreshes, unlike this row's message history, which is. A
    refreshed page starts a fresh browser session but still shows past
    messages.
    """

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='browser_chat_sessions'
    )
    session_key = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    # Populated once the first tool call in a connection returns one — lets
    # the chat link out to the console's live-view page (browserbase:session_detail).
    browserbase_session_id = models.CharField(max_length=64, blank=True)
    started_at = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-started_at']
        indexes = [models.Index(fields=['user', '-started_at'])]

    def __str__(self) -> str:
        return f'Browser chat {self.session_key} — {self.user}'


class BrowserChatMessage(models.Model):
    ROLE_USER = 'user'
    ROLE_ASSISTANT = 'assistant'
    ROLE_CHOICES = [
        (ROLE_USER, 'User'),
        (ROLE_ASSISTANT, 'Assistant'),
    ]

    session = models.ForeignKey(
        BrowserChatSession, on_delete=models.CASCADE, related_name='messages'
    )
    role = models.CharField(max_length=16, choices=ROLE_CHOICES)
    content = models.TextField()
    # Which MCP tool this turn resolved to (navigate/act/extract/observe),
    # blank for plain user messages and error replies.
    tool_name = models.CharField(max_length=32, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']
        indexes = [models.Index(fields=['session', 'created_at'])]

    def __str__(self) -> str:
        return f'[{self.role}] {self.content[:60]}'


class ScheduledBrowserTask(models.Model):
    """A browser automation that runs on a recurring schedule via Celery Beat.

    Each line of ``instructions`` is routed the same way a chat message is
    (see mcp_client.route_message) and run in order as ONE MCP session —
    tasks.run_scheduled_browser_task calls mcp_client.run_task(), the same
    one-shot "open a session, run these steps, close it" helper the
    console's Navigate button uses. Unlike the chat, there's no persistent
    session across runs — every scheduled run starts fresh (though it still
    shares the persistent Browserbase Context, so it stays logged into
    whatever the chat/console logged into by hand).

    The actual recurrence (the "every day at 4pm" part) lives on the linked
    django_celery_beat PeriodicTask/CrontabSchedule — this row holds what to
    run and the human-facing metadata; browser_automation/views.py keeps
    both in sync when a task is created/edited/deleted.
    """

    STATUS_CHOICES = [
        ('success', 'Success'),
        ('error', 'Error'),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='scheduled_browser_tasks'
    )
    name = models.CharField(max_length=120)
    instructions = models.TextField(
        help_text='One instruction per line, run in order on a single browser session each time.'
    )
    is_active = models.BooleanField(default=True)
    periodic_task = models.OneToOneField(
        PeriodicTask, on_delete=models.CASCADE, related_name='scheduled_browser_task',
    )
    last_run_at = models.DateTimeField(null=True, blank=True)
    last_run_status = models.CharField(max_length=20, choices=STATUS_CHOICES, blank=True)
    last_run_detail = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [models.Index(fields=['user', '-created_at'])]

    def __str__(self) -> str:
        return self.name
