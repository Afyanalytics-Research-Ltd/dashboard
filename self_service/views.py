import json

from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.exceptions import ValidationError
from django.core.files.storage import default_storage
from django.db.models import Count
from django.http import JsonResponse
from django.views import View

from .models import ChatSession, ChatMessage
from .security import get_user_access_context


class ChatHistoryView(LoginRequiredMixin, View):
    """Return the last N messages of one session (?session=<key>), or the
    user's most recent active session when no key is given."""

    def get(self, request):
        session_key = request.GET.get('session', '').strip()

        if session_key:
            try:
                session = ChatSession.objects.get(session_key=session_key, user=request.user)
            except (ChatSession.DoesNotExist, ValueError, ValidationError):
                return JsonResponse({'messages': [], 'session': None}, status=404)
        else:
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
            .values('role', 'content', 'query_intent', 'chart_image', 'chart_caption', 'created_at')
            .order_by('created_at')[:50]
        )
        for msg in messages:
            msg['created_at'] = msg['created_at'].isoformat()
            chart_image = msg.pop('chart_image', '')
            msg['chart_url'] = default_storage.url(chart_image) if chart_image else None
            msg['chart_caption'] = msg.get('chart_caption') or ''

        return JsonResponse({
            'messages': messages,
            'session': str(session.session_key),
            'title': session.title,
        })


class ChatSessionListView(LoginRequiredMixin, View):
    """List the current user's past conversations, most recently active first."""

    def get(self, request):
        sessions = (
            ChatSession.objects
            .filter(user=request.user)
            .annotate(message_count=Count('messages'))
            .order_by('-last_activity')[:50]
        )

        data = []
        for session in sessions:
            last_message = session.messages.order_by('-created_at').first()
            data.append({
                'session_key': str(session.session_key),
                'title': session.title or 'New conversation',
                'preview': last_message.content[:100] if last_message else '',
                'started_at': session.started_at.isoformat(),
                'last_activity': session.last_activity.isoformat(),
                'message_count': session.message_count,
            })

        return JsonResponse({'sessions': data})


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
