from django.urls import path
from . import api

urlpatterns = [
    path("query/", api.query, name="api-query"),
    path("resume/", api.resume, name="api-resume"),
    path("visualize/", api.visualize, name="api-visualize"),
    path("whatsapp/", api.whatsapp_webhook, name="api-whatsapp"),
    path("whatsapp/messages", api.whatsapp_webhook, name="api-whatsapp-messages"),
    path("whatsapp/statuses", api.whatsapp_webhook, name="api-whatsapp-statuses"),
]