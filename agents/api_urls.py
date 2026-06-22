from django.urls import path
from . import api

urlpatterns = [
    path("query/", api.query, name="api-query"),
    path("resume/", api.resume, name="api-resume"),
]