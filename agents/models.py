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
