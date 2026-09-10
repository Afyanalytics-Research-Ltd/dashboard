"""WebSocket consumer for the browser-agent chat.

Connection lifecycle:
  connect    → auth (superuser only), resume/create a BrowserChatSession,
               open ONE MCP connection (mcp_client.open_session()) that stays
               open for the life of this WebSocket — the underlying
               Browserbase browser session lives exactly as long as this
               connection does, so the agent stays on the same page across
               turns instead of resetting every message.
  receive    → route the message text to an MCP tool call, persist both
               sides of the exchange, return the result
  disconnect → end the MCP session (releases the Browserbase session too)

Message routing (see mcp_client.route_message) is a simple, transparent set
of rules — not its own LLM call — so the mapping from what you type to
which MCP tool runs is predictable:
  "navigate:<url>" / "go to <url>" / "open <url>" / a bare URL → navigate
  "act:<instruction>"                                          → act
  "extract:<instruction>"                                      → extract
  "observe:<instruction>"                                      → observe
  anything else                                                → act
  (act/extract/observe are Stagehand's AI-driven tools — see mcp_client.py)
  The same routing backs scheduled browser tasks — see tasks.py.
"""

import json
import logging
from urllib.parse import parse_qs

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer

from . import mcp_client
from .mcp_client import BrowserMCPError, format_tool_reply, route_message

logger = logging.getLogger(__name__)


class BrowserAgentChatConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        self.open_session = None

        user = self.scope.get('user')
        if not user or not user.is_authenticated or not user.is_superuser:
            await self.close(code=4001)
            return

        self.user = user

        requested_key = self._requested_session_key()
        self.session_obj, is_new = await database_sync_to_async(self._get_or_create_session)(requested_key)

        try:
            self.open_session = await mcp_client.open_session()
            start_result = await self.open_session.call('start')
        except Exception as exc:
            logger.error('browser-agent chat: failed to open MCP session: %s', exc)
            await self.accept()
            await self._send({
                'type': 'message',
                'role': 'assistant',
                'content': f"Couldn't reach the browser-agent tool server: {exc}",
            })
            await self.close(code=4002)
            return

        if isinstance(start_result, dict) and start_result.get('sessionId'):
            self.open_session.browserbase_session_id = start_result['sessionId']
            await database_sync_to_async(self._set_browserbase_session_id)(start_result['sessionId'])

        await self.accept()
        logger.info(
            'Browser-agent chat WS connected: user=%s session=%s new=%s browserbase_session=%s',
            user.username, self.session_obj.session_key, is_new,
            self.open_session.browserbase_session_id,
        )
        await self._send({
            'type': 'session',
            'session_key': str(self.session_obj.session_key),
            'is_new': is_new,
            'browserbase_session_id': self.open_session.browserbase_session_id,
            'history': await database_sync_to_async(self._history)() if not is_new else [],
        })
        if is_new:
            await self._send({
                'type': 'message',
                'role': 'assistant',
                'content': (
                    "New browser session started. Tell me a URL to open, or what to do "
                    "once you're on a page — e.g. \"go to https://example.com\", "
                    "\"click the sign in button\", \"extract: the pricing table\"."
                ),
            })

    async def disconnect(self, close_code):
        if self.open_session is not None:
            await self.open_session.close()
        logger.info(
            'Browser-agent chat WS disconnected: user=%s code=%s',
            getattr(self, 'user', '?'), close_code,
        )

    def _requested_session_key(self):
        query_string = self.scope.get('query_string', b'').decode('utf-8', 'ignore')
        values = parse_qs(query_string).get('session')
        return values[0].strip() if values and values[0].strip() else None

    async def receive(self, text_data=None, bytes_data=None):
        if not text_data or self.open_session is None:
            return

        try:
            payload = json.loads(text_data)
        except json.JSONDecodeError:
            return

        message = str(payload.get('message', '')).strip()
        if not message:
            return

        await self._send({'type': 'typing', 'status': True})
        await database_sync_to_async(self._save_message)('user', message, '')

        tool_name, args = route_message(message)
        try:
            result = await self.open_session.call(tool_name, args)
            content = format_tool_reply(tool_name, result)
        except BrowserMCPError as exc:
            content = f"That didn't work: {exc}"
        except Exception:
            logger.exception('browser-agent chat: tool call failed')
            content = "I ran into an unexpected problem running that."

        await database_sync_to_async(self._save_message)('assistant', content, tool_name)
        await self._send({
            'type': 'message',
            'role': 'assistant',
            'content': content,
            'tool': tool_name,
        })

    # ------------------------------------------------------------------
    # DB helpers — run via database_sync_to_async, never called directly
    # ------------------------------------------------------------------

    def _get_or_create_session(self, session_key):
        from .models import BrowserChatSession

        if session_key:
            try:
                return BrowserChatSession.objects.get(session_key=session_key, user=self.user), False
            except (BrowserChatSession.DoesNotExist, ValueError):
                pass
        return BrowserChatSession.objects.create(user=self.user), True

    def _set_browserbase_session_id(self, browserbase_session_id):
        self.session_obj.browserbase_session_id = browserbase_session_id
        self.session_obj.save(update_fields=['browserbase_session_id'])

    def _save_message(self, role, content, tool_name):
        from .models import BrowserChatMessage
        BrowserChatMessage.objects.create(
            session=self.session_obj, role=role, content=content, tool_name=tool_name,
        )

    def _history(self):
        return [
            {'role': m.role, 'content': m.content, 'tool': m.tool_name}
            for m in self.session_obj.messages.all()
        ]

    async def _send(self, data):
        await self.send(text_data=json.dumps(data))
