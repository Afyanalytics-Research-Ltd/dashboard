"""
WebSocket consumer for the Self-Service Analytics chatbot.

Connection lifecycle:
  connect    → authenticate user, load access context, send welcome message
  receive    → detect intent, enforce RLS/CLS, return formatted response
  disconnect → log session end
"""

import json
import logging

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer

logger = logging.getLogger('self_service')


class AnalyticsChatConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        user = self.scope.get('user')
        if not user or not user.is_authenticated:
            await self.close(code=4001)
            return

        self.user = user
        self.session_obj = None

        # Load role-based access context from DB
        self.access_context = await database_sync_to_async(
            self._load_access_context
        )()

        # Persist a ChatSession record
        self.session_obj = await database_sync_to_async(
            self._create_session
        )()

        await self.accept()
        logger.info('Chat WS connected: user=%s role=%s', user.username, self.access_context['role'])

        await self._send_welcome()

    async def disconnect(self, close_code):
        if self.session_obj:
            await database_sync_to_async(self._close_session)()
        logger.info(
            'Chat WS disconnected: user=%s code=%s',
            getattr(self, 'user', '?'),
            close_code,
        )

    async def receive(self, text_data=None, bytes_data=None):
        if not text_data:
            return

        try:
            payload = json.loads(text_data)
        except json.JSONDecodeError:
            return

        query = str(payload.get('message', '')).strip()
        if not query:
            return

        # Acknowledge with typing indicator
        await self._send({'type': 'typing', 'status': True})

        # Process query synchronously (DB access inside handlers)
        response = await database_sync_to_async(self._process)(query)

        # Persist both sides of the exchange
        await database_sync_to_async(self._save_messages)(query, response)

        await self._send({
            'type': 'message',
            'role': 'assistant',
            'content': response['content'],
            'data': response.get('data'),
            'intent': response.get('intent', ''),
        })

    # ------------------------------------------------------------------
    # Internal helpers (run in sync thread via database_sync_to_async)
    # ------------------------------------------------------------------

    def _load_access_context(self):
        from .security import get_user_access_context
        return get_user_access_context(self.user)

    def _create_session(self):
        from .models import ChatSession
        return ChatSession.objects.create(user=self.user)

    def _close_session(self):
        self.session_obj.is_active = False
        self.session_obj.save(update_fields=['is_active', 'last_activity'])

    def _process(self, query):
        from .handlers import process_query
        return process_query(query, self.user, self.access_context)

    def _save_messages(self, user_text, response):
        from .models import ChatMessage
        ChatMessage.objects.bulk_create([
            ChatMessage(
                session=self.session_obj,
                role=ChatMessage.ROLE_USER,
                content=user_text,
            ),
            ChatMessage(
                session=self.session_obj,
                role=ChatMessage.ROLE_ASSISTANT,
                content=response['content'],
                query_intent=response.get('intent', ''),
            ),
        ])

    # ------------------------------------------------------------------
    # WebSocket send helpers
    # ------------------------------------------------------------------

    async def _send(self, data):
        await self.send(text_data=json.dumps(data))

    async def _send_welcome(self):
        ctx = self.access_context
        topics = ', '.join(ctx['allowed_topics'])
        await self._send({
            'type': 'message',
            'role': 'assistant',
            'content': (
                f"Hello! I'm your **Afya Analytics Assistant**.\n\n"
                f"As a *{ctx['role_display']}*, you can ask me about: **{topics}**.\n\n"
                f"Type **help** to see example questions, or just ask away!"
            ),
        })
