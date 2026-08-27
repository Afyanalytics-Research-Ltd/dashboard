"""
Warehouse models: TrackedSpreadsheet, SnowflakeQueryLog, and the spreadsheet
analyst models (Workbook, Conversation, ChatMessage, Artifact).
"""

import uuid

from django.conf import settings
from django.db import models
from django.urls import reverse


class TrackedSpreadsheet(models.Model):
    """Local record of a spreadsheet the app has created or interacted with.

    The actual data lives on Google Drive / Sheets. This table is a
    convenience index so the web UI can list and link past sheets without
    having to remember IDs.
    """

    spreadsheet_id = models.CharField(max_length=255, unique=True)
    title = models.CharField(max_length=512, blank=True)
    web_view_link = models.URLField(blank=True)
    client = models.ForeignKey(
        'core.Client',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='spreadsheets',
        help_text='Client this spreadsheet belongs to (optional).',
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        on_delete=models.SET_NULL,
        related_name='spreadsheets',
        help_text='User who first tracked this spreadsheet.',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']
        verbose_name = 'Tracked Spreadsheet'
        verbose_name_plural = 'Tracked Spreadsheets'

    def __str__(self) -> str:
        return self.title or self.spreadsheet_id

    def get_absolute_url(self) -> str:
        from django.urls import reverse
        return reverse('warehouse:detail', kwargs={'spreadsheet_id': self.spreadsheet_id})


class SnowflakeQueryLog(models.Model):
    """Record of every SQL query executed against Snowflake from the UI."""

    STATUS_CHOICES = [
        ('success', 'Success'),
        ('error', 'Error'),
        ('pending', 'Pending'),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='snowflake_queries',
    )
    query = models.TextField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    rows_returned = models.PositiveIntegerField(default=0)
    execution_time_ms = models.PositiveIntegerField(default=0)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Snowflake Query Log'
        verbose_name_plural = 'Snowflake Query Logs'
        indexes = [
            models.Index(fields=['user', 'created_at'], name='wh_sqlog_user_created_idx'),
            models.Index(fields=['status', 'created_at'], name='wh_sqlog_status_created_idx'),
        ]

    def __str__(self) -> str:
        return (
            f"{self.user} — {self.created_at.strftime('%Y-%m-%d %H:%M')} "
            f"({self.status})"
        )


# ---------------------------------------------------------------------------
# Spreadsheet analyst — Workbook, Conversation, ChatMessage, Artifact
#
# The Django DB is the single source of truth for a conversation. The
# agent's in-memory kernel (warehouse/agent/session.py) is a cache in front
# of it — if a worker restarts, the transcript replays and nothing is lost
# except locally-defined variables, which the model simply recomputes.
# ---------------------------------------------------------------------------

def workbook_upload_path(instance: 'Workbook', filename: str) -> str:
    return f'warehouse/analyst/workbooks/{instance.id}/{filename}'


class Workbook(models.Model):
    """An uploaded spreadsheet queued for analysis."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='workbooks',
        null=True,
        blank=True,
    )
    file = models.FileField(upload_to=workbook_upload_path)
    original_name = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    #: Cached profile so the upload page can show the schema without reloading.
    overview = models.TextField(blank=True)
    load_error = models.TextField(blank=True)

    class Meta:
        ordering = ['-uploaded_at']
        verbose_name = 'Workbook'
        verbose_name_plural = 'Workbooks'

    def __str__(self) -> str:
        return self.original_name

    def get_absolute_url(self) -> str:
        return reverse('warehouse:analyst_workbook_detail', args=[self.id])


class Conversation(models.Model):
    """One chat thread against one workbook."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workbook = models.ForeignKey(
        Workbook, on_delete=models.CASCADE, related_name='conversations'
    )
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='analyst_conversations',
        null=True,
        blank=True,
    )
    title = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    #: LangChain messages serialised with `messages_to_dict`. Replayed on every
    #: turn so the agent has full context regardless of which worker serves it.
    transcript = models.JSONField(default=list, blank=True)

    class Meta:
        ordering = ['-updated_at']
        verbose_name = 'Analyst Conversation'
        verbose_name_plural = 'Analyst Conversations'

    def __str__(self) -> str:
        return self.title or f'Conversation {self.id}'

    def get_absolute_url(self) -> str:
        return reverse('warehouse:analyst_chat', args=[self.id])


class ChatMessage(models.Model):
    """A display-facing turn in an analyst conversation. Separate from
    `Conversation.transcript` on purpose: this is what the template renders,
    that is what the model replays."""

    ROLE_CHOICES = [('user', 'User'), ('assistant', 'Assistant'), ('error', 'Error')]

    conversation = models.ForeignKey(
        Conversation, on_delete=models.CASCADE, related_name='messages'
    )
    role = models.CharField(max_length=16, choices=ROLE_CHOICES)
    content = models.TextField()
    #: [{"name": "run_python", "args": {...}}, ...] - powers the "show work" panel.
    tool_calls = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at', 'id']
        verbose_name = 'Analyst Chat Message'
        verbose_name_plural = 'Analyst Chat Messages'

    def __str__(self) -> str:
        return f'{self.role}: {self.content[:60]}'


class Artifact(models.Model):
    """A chart, table or report the analyst agent produced."""

    KIND_CHOICES = [('chart', 'Chart'), ('table', 'Table'), ('report', 'Report')]

    conversation = models.ForeignKey(
        Conversation, on_delete=models.CASCADE, related_name='artifacts'
    )
    message = models.ForeignKey(
        ChatMessage,
        on_delete=models.CASCADE,
        related_name='artifacts',
        null=True,
        blank=True,
    )
    kind = models.CharField(max_length=16, choices=KIND_CHOICES)
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='warehouse/analyst/artifacts/%Y/%m/')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at', 'id']
        verbose_name = 'Analyst Artifact'
        verbose_name_plural = 'Analyst Artifacts'

    def __str__(self) -> str:
        return f'{self.kind}: {self.title}'

    @property
    def is_image(self) -> bool:
        return self.kind == 'chart'
