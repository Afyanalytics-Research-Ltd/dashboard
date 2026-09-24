from django.urls import re_path

from . import consumers

websocket_urlpatterns = [
    re_path(r'^ws/browser-agent/chat/$', consumers.BrowserAgentChatConsumer.as_asgi()),
]
