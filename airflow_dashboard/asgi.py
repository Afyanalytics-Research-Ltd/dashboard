"""
Afya DataHub — ASGI entry point.

Handles both HTTP (via Django's ASGI app) and WebSocket (via Channels).
"""

import os

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.security.websocket import AllowedHostsOriginValidator
from django.core.asgi import get_asgi_application

import self_service.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'airflow_dashboard.settings')

django_asgi_app = get_asgi_application()

application = ProtocolTypeRouter({
    'http': django_asgi_app,
    'websocket': AllowedHostsOriginValidator(
        AuthMiddlewareStack(
            URLRouter(self_service.routing.websocket_urlpatterns)
        )
    ),
})
