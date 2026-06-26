from django.urls import re_path

from . import consumers

websocket_urlpatterns = [
    re_path(r'^ws/analytics/chat/$', consumers.AnalyticsChatConsumer.as_asgi()),
]
