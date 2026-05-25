"""DRF API URL configuration for the warehouse module."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .api import (
    SnowflakeQueryAPIView,
    SnowflakeQueryLogViewSet,
    SnowflakeTablesAPIView,
    SpreadsheetViewSet,
)

router = DefaultRouter()
router.register(r"spreadsheets", SpreadsheetViewSet, basename="spreadsheet")
router.register(r"snowflake/queries", SnowflakeQueryLogViewSet, basename="snowflake-query-log")

urlpatterns = [
    path("", include(router.urls)),
    path("snowflake/query/", SnowflakeQueryAPIView.as_view(), name="snowflake-query"),
    path("snowflake/tables/", SnowflakeTablesAPIView.as_view(), name="snowflake-tables"),
]
