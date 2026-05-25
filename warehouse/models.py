"""
Warehouse models: TrackedSpreadsheet and SnowflakeQueryLog.
"""

from django.conf import settings
from django.db import models


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
