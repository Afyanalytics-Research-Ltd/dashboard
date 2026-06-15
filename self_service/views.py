import json

from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse
from django.views import View

from .models import ChatSession, ChatMessage
from .security import get_user_access_context


class ChatHistoryView(LoginRequiredMixin, View):
    """Return the last N messages from the user's most recent active session."""

    def get(self, request):
        session = (
            ChatSession.objects
            .filter(user=request.user, is_active=True)
            .order_by('-started_at')
            .first()
        )
        if not session:
            return JsonResponse({'messages': []})

        messages = list(
            session.messages
            .values('role', 'content', 'query_intent', 'created_at')
            .order_by('created_at')[:50]
        )
        for msg in messages:
            msg['created_at'] = msg['created_at'].isoformat()

        return JsonResponse({'messages': messages, 'session': str(session.session_key)})


class AccessContextView(LoginRequiredMixin, View):
    """Return the current user's access context (for debugging / UI hints)."""

    def get(self, request):
        ctx = get_user_access_context(request.user)
        # Serialise sets → lists for JSON
        ctx['denied_columns'] = list(ctx['denied_columns'])
        ctx['masked_columns'] = list(ctx['masked_columns'])
        ctx['client'] = str(ctx['client']) if ctx['client'] else None
        ctx['facility'] = str(ctx['facility']) if ctx['facility'] else None
        return JsonResponse(ctx)
