import uuid

from django.contrib.auth import get_user_model
from django.db import models

User = get_user_model()


def chart_image_upload_path(instance, filename):
    return f'self_service/charts/{instance.session_id}/{filename}'


class ChatSession(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name='chat_sessions'
    )
    session_key = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    title = models.CharField(max_length=120, blank=True)
    # The LangGraph checkpointer's thread_id for this session — minted once,
    # on the first metric question, and reused for every turn after that so
    # the graph's own conversation memory (agents/state.py's `messages` +
    # `last_matched_metric`) actually accumulates instead of starting fresh
    # every message. See self_service/consumers.py:_run_agent.
    thread_id = models.CharField(max_length=64, blank=True)
    started_at = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['-started_at']
        indexes = [models.Index(fields=['user', '-started_at'])]

    def __str__(self):
        return f'Chat {self.session_key} — {self.user}'


class ChatMessage(models.Model):
    ROLE_USER = 'user'
    ROLE_ASSISTANT = 'assistant'
    ROLE_CHOICES = [
        (ROLE_USER, 'User'),
        (ROLE_ASSISTANT, 'Assistant'),
    ]

    session = models.ForeignKey(
        ChatSession, on_delete=models.CASCADE, related_name='messages'
    )
    role = models.CharField(max_length=16, choices=ROLE_CHOICES)
    content = models.TextField()
    query_intent = models.CharField(max_length=64, blank=True)
    # FileField (not ImageField) deliberately — no Pillow dependency needed,
    # the DB never has to introspect dimensions, only serve the bytes back.
    chart_image = models.FileField(upload_to=chart_image_upload_path, blank=True, null=True)
    chart_caption = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']
        indexes = [models.Index(fields=['session', 'created_at'])]

    def __str__(self):
        return f'[{self.role}] {self.content[:60]}'
