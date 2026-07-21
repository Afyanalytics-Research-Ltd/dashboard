"""
Agents app models.

Just enough persistent state to make the WhatsApp channel conversational
across separate webhook requests. Unlike the chat websocket consumer (which
can hold state on its own instance for the life of the connection), a Django
webhook view is stateless per-request — nothing survives between one
WhatsApp message and the next unless it's written somewhere durable.
"""

from django.db import models


class WhatsAppChatState(models.Model):
    """Tracks the last analytics-agent result for one WhatsApp phone number.

    Lets a later "yes" / "can I get a graph" message find its way back to
    the right LangGraph thread_id, since each webhook POST has no memory
    of its own.
    """

    phone = models.CharField(max_length=32, unique=True, db_index=True)
    last_metric_thread_id = models.CharField(max_length=64, blank=True)
    chart_offer_pending = models.BooleanField(default=False)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f'WhatsApp state for {self.phone}'
