"""
Agents app models.

Two groups of models live here:
  - WhatsAppChatState / ConversationThread — just enough persistent state to
    make the WhatsApp channel conversational across separate webhook
    requests. Unlike the chat websocket consumer (which can hold state on
    its own instance for the life of the connection), a Django webhook view
    is stateless per-request — nothing survives between one WhatsApp message
    and the next unless it's written somewhere durable.
  - MetricDefinition / PendingCubeMeasure — the DB-backed metric catalog
    (replaces catalog/metrics.yaml as the source of truth, see
    agents/catalog.py) and the staged-approval queue for adding new Cube
    measures via the Semantic Layer Configuration settings page
    (agents/catalog_sync.py does the actual work; agents/views.py is the UI).
"""

from django.conf import settings
from django.db import models


class WhatsAppChatState(models.Model):
    """Tracks conversational state for one WhatsApp phone number.

    `thread_id` is the persistent LangGraph thread for this phone number —
    reused across every webhook POST (instead of minting a new one per
    message) so the graph's checkpointed conversation history survives
    between messages, and so a later "yes" / "can I get a graph" reply can
    find its way back to the right thread.
    """

    phone = models.CharField(max_length=32, unique=True, db_index=True)
    thread_id = models.CharField(max_length=64, blank=True)
    chart_offer_pending = models.BooleanField(default=False)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f'WhatsApp state for {self.phone}'


class ConversationThread(models.Model):
    """Tracks the persistent LangGraph thread_id for one REST API user.

    Mirrors WhatsAppChatState.thread_id for the /api/query/ channel: reused
    across requests so the same user's follow-up questions land in the same
    checkpointed thread instead of starting a memoryless one each time.
    """

    user_id = models.CharField(max_length=255, unique=True, db_index=True)
    thread_id = models.CharField(max_length=64)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f'Conversation thread for {self.user_id}'


class MetricDefinition(models.Model):
    """One curated metric the agent can resolve a question to.

    Same shape as a catalog/metrics.yaml entry (id/name/description/
    cube_query), now stored in the DB so it can be managed from the Agent
    Configuration settings page instead of hand-editing YAML. Read by
    agents/catalog.py's get_all()/get_by_id()/as_context() — those keep
    their exact prior signatures, so every consumer (nodes_query.py,
    nodes.py, derived_metrics.py) needed zero changes for this swap.
    """

    metric_id = models.SlugField(max_length=100, unique=True)
    name = models.CharField(max_length=200)
    description = models.TextField()
    cube_query = models.JSONField(
        default=dict,
        help_text='{"measures": [...], "dimensions": [...], "timeDimensions": [...], "filters": [...], "limit": 500}',
    )
    is_active = models.BooleanField(
        default=True,
        help_text='Inactive metrics are hidden from the agent without deleting the row.',
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL,
        related_name='metric_definitions_created',
    )
    updated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL,
        related_name='metric_definitions_updated',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['name']
        verbose_name = 'Metric Definition'
        verbose_name_plural = 'Metric Definitions'

    def __str__(self) -> str:
        return f'{self.metric_id} ({self.name})'


class PendingCubeMeasure(models.Model):
    """A proposed new measure on an EXISTING cube, staged for review.

    Unlike agents/schema_writer.py's auto-join writer (which validates a
    join's cardinality against live Snowflake data before writing it with no
    human review gate), a hand-typed measure has no equivalent automated
    safety check — so this stays "pending" until a superuser approves it.
    Approval (agents/catalog_sync.write_pending_measure_to_yaml) runs a live
    Snowflake column-existence probe, then splices the measure into
    model/cubes/<cube_name>.yml, which Cube (CUBEJS_DEV_MODE=true) hot-
    reloads with no restart needed.
    """

    STATUS_PENDING = 'pending'
    STATUS_WRITTEN = 'written'
    STATUS_REJECTED = 'rejected'
    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending review'),
        (STATUS_WRITTEN, 'Written to cube schema'),
        (STATUS_REJECTED, 'Rejected'),
    ]

    MEASURE_TYPE_CHOICES = [
        ('count', 'count'),
        ('sum', 'sum'),
        ('avg', 'avg'),
        ('min', 'min'),
        ('max', 'max'),
        ('countDistinct', 'countDistinct'),
        ('number', 'number'),
    ]

    ACTION_ADD = 'add'
    ACTION_EDIT = 'edit'
    ACTION_CHOICES = [
        (ACTION_ADD, 'Add new measure'),
        (ACTION_EDIT, 'Edit existing measure'),
    ]

    action = models.CharField(
        max_length=10, choices=ACTION_CHOICES, default=ACTION_ADD,
        help_text='"add" splices a new measure in; "edit" replaces an existing one by name.',
    )
    cube_name = models.CharField(
        max_length=100,
        help_text='Must match an existing model/cubes/<cube_name>.yml.',
    )
    measure_name = models.CharField(max_length=100, help_text='snake_case Cube measure name.')
    measure_type = models.CharField(max_length=20, choices=MEASURE_TYPE_CHOICES)
    sql_expression = models.CharField(
        max_length=500, blank=True,
        help_text='Cube sql: value, e.g. {CUBE}."SOME_COLUMN". Not required for type=count.',
    )
    title = models.CharField(max_length=200, blank=True)
    description = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING)
    requested_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL,
        related_name='pending_cube_measures_requested',
    )
    reviewed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL,
        related_name='pending_cube_measures_reviewed',
    )
    rejection_reason = models.TextField(blank=True)
    requested_at = models.DateTimeField(auto_now_add=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-requested_at']
        verbose_name = 'Pending Cube Measure'
        verbose_name_plural = 'Pending Cube Measures'

    def __str__(self) -> str:
        return f'{self.cube_name}.{self.measure_name} ({self.status})'
