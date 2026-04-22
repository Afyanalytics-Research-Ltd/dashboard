from django.urls import path
from .views import snowflake_query_view

urlpatterns = [
    path("snowflake/query/", snowflake_query_view, name="snowflake_query"),
]