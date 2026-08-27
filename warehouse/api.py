"""DRF API views for the warehouse module.

Endpoints:
    /api/v1/warehouse/spreadsheets/         — SpreadsheetViewSet
    /api/v1/warehouse/snowflake/query/      — SnowflakeQueryAPIView
    /api/v1/warehouse/snowflake/queries/    — SnowflakeQueryLogViewSet
    /api/v1/warehouse/snowflake/tables/     — SnowflakeTablesAPIView
"""

import logging
import time

import pandas as pd
from django.contrib.auth import get_user_model
from django_filters.rest_framework import DjangoFilterBackend
from drf_spectacular.utils import OpenApiParameter, extend_schema, extend_schema_view
from rest_framework import filters, mixins, permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import PermissionDenied
from rest_framework.response import Response
from rest_framework.views import APIView

from authentication.module_access import MODULE_WAREHOUSE, has_module_access
from core.models import AuditLog

from .models import SnowflakeQueryLog, TrackedSpreadsheet
from .serializers import (
    SnowflakeQueryLogSerializer,
    SnowflakeQuerySerializer,
    TrackedSpreadsheetSerializer,
)
from .services.facility_scope import FacilityScopeError, filter_tables_for_scope, get_facility_scope, validate_query_scope
from .services.snowflake import SnowflakeClient, SnowflakeQueryError
from .sheet_service import SheetsServiceError, get_service

logger = logging.getLogger(__name__)
User = get_user_model()


def _is_warehouse_user(user) -> bool:
    """True if the user may access the warehouse (see authentication.module_access)."""
    return has_module_access(user, MODULE_WAREHOUSE)


class IsWarehouseUser(permissions.BasePermission):
    """Allow only superusers and Client Admins."""

    message = "You must be a Client Admin to access the warehouse API."

    def has_permission(self, request, view) -> bool:
        return _is_warehouse_user(request.user)


# ────────────────────────────────────────────── SpreadsheetViewSet

@extend_schema_view(
    list=extend_schema(
        summary="List tracked spreadsheets",
        tags=["warehouse"],
    ),
    retrieve=extend_schema(
        summary="Get a tracked spreadsheet",
        tags=["warehouse"],
    ),
    create=extend_schema(
        summary="Track (create) a spreadsheet",
        tags=["warehouse"],
    ),
    destroy=extend_schema(
        summary="Remove a tracked spreadsheet from the index",
        tags=["warehouse"],
    ),
)
class SpreadsheetViewSet(
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    mixins.CreateModelMixin,
    mixins.DestroyModelMixin,
    viewsets.GenericViewSet,
):
    """CRUD for TrackedSpreadsheet (no update via API — update by re-opening)."""

    queryset = TrackedSpreadsheet.objects.select_related("client", "created_by").all()
    serializer_class = TrackedSpreadsheetSerializer
    permission_classes = [permissions.IsAuthenticated, IsWarehouseUser]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ["client"]
    search_fields = ["title", "spreadsheet_id"]
    ordering_fields = ["updated_at", "created_at", "title"]
    ordering = ["-updated_at"]

    @extend_schema(
        summary="Read cell values from a spreadsheet",
        tags=["warehouse"],
        parameters=[
            OpenApiParameter(
                "range",
                str,
                description="A1 notation range, e.g. Sheet1!A1:D20",
                required=True,
            )
        ],
    )
    @action(detail=True, methods=["get"], url_path="values")
    def values(self, request, pk=None) -> Response:
        """GET /spreadsheets/{pk}/values/?range=Sheet1!A1:Z100"""
        obj = self.get_object()
        range_a1 = request.query_params.get("range", "Sheet1!A1:Z100")
        try:
            data = get_service().read_values(obj.spreadsheet_id, range_a1)
        except SheetsServiceError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response({
            "spreadsheet_id": obj.spreadsheet_id,
            "range": range_a1,
            "values": data,
            "row_count": len(data),
        })

    @extend_schema(
        summary="Share spreadsheet with a user",
        tags=["warehouse"],
        request={
            "application/json": {
                "type": "object",
                "properties": {
                    "email": {"type": "string"},
                    "role": {"type": "string", "enum": ["reader", "commenter", "writer"]},
                    "notify": {"type": "boolean"},
                },
                "required": ["email", "role"],
            }
        },
    )
    @action(detail=True, methods=["post"], url_path="share")
    def share(self, request, pk=None) -> Response:
        """POST /spreadsheets/{pk}/share/"""
        obj = self.get_object()
        email = request.data.get("email", "").strip()
        role = request.data.get("role", "reader")
        notify = bool(request.data.get("notify", False))

        if not email:
            return Response({"error": "email is required."}, status=status.HTTP_400_BAD_REQUEST)
        if role not in {"reader", "commenter", "writer"}:
            return Response(
                {"error": "role must be one of: reader, commenter, writer."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            result = get_service().share(obj.spreadsheet_id, email=email, role=role, notify=notify)
        except SheetsServiceError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        AuditLog.log(
            user=request.user,
            action="share",
            resource="TrackedSpreadsheet",
            resource_id=obj.spreadsheet_id,
            detail=f"Shared with {email} as {role}",
        )
        return Response(result, status=status.HTTP_201_CREATED)


# ─────────────────────────────────────── SnowflakeQueryAPIView

@extend_schema(
    summary="Execute a Snowflake SELECT query",
    tags=["warehouse"],
    request=SnowflakeQuerySerializer,
    responses={
        200: {
            "type": "object",
            "properties": {
                "columns": {"type": "array", "items": {"type": "string"}},
                "rows": {"type": "array"},
                "row_count": {"type": "integer"},
                "execution_time_ms": {"type": "integer"},
                "query_log_id": {"type": "integer"},
            },
        }
    },
)
class SnowflakeQueryAPIView(APIView):
    """POST a SQL query; returns columns + rows as JSON."""

    permission_classes = [permissions.IsAuthenticated, IsWarehouseUser]

    def post(self, request) -> Response:
        serializer = SnowflakeQuerySerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_422_UNPROCESSABLE_ENTITY)

        sql = serializer.validated_data["query"]
        max_rows = serializer.validated_data["max_rows"]

        log = SnowflakeQueryLog.objects.create(
            user=request.user, query=sql, status="pending"
        )
        t0 = time.monotonic()
        try:
            scope = get_facility_scope(request.user)
            try:
                validate_query_scope(sql, scope)
            except FacilityScopeError as exc:
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                log.status = "error"
                log.error_message = str(exc)
                log.execution_time_ms = elapsed_ms
                log.save(update_fields=["status", "error_message", "execution_time_ms"])
                return Response({"error": str(exc)}, status=status.HTTP_403_FORBIDDEN)

            client = SnowflakeClient()
            df = client.query(sql, max_rows=max_rows)
            elapsed_ms = int((time.monotonic() - t0) * 1000)

            df = df.where(pd.notnull(df), None)
            columns = list(df.columns)
            rows = df.values.tolist()

            log.status = "success"
            log.rows_returned = len(df)
            log.execution_time_ms = elapsed_ms
            log.save(update_fields=["status", "rows_returned", "execution_time_ms"])

            AuditLog.log(
                user=request.user,
                action="query",
                resource="Snowflake",
                detail=f"API query: rows={len(df)}, ms={elapsed_ms}",
                ip_address=request.META.get("REMOTE_ADDR"),
            )
            return Response({
                "columns": columns,
                "rows": rows,
                "row_count": len(df),
                "execution_time_ms": elapsed_ms,
                "query_log_id": log.pk,
            })

        except (SnowflakeQueryError, Exception) as exc:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            log.status = "error"
            log.error_message = str(exc)
            log.execution_time_ms = elapsed_ms
            log.save(update_fields=["status", "error_message", "execution_time_ms"])
            logger.error("SnowflakeQueryAPIView error: %s", exc)
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)


# ──────────────────────────────────── SnowflakeQueryLogViewSet

@extend_schema_view(
    list=extend_schema(
        summary="List the current user's Snowflake query history",
        tags=["warehouse"],
    ),
    retrieve=extend_schema(
        summary="Get a single query log entry",
        tags=["warehouse"],
    ),
)
class SnowflakeQueryLogViewSet(
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    viewsets.GenericViewSet,
):
    """Read-only list of the authenticated user's query history."""

    serializer_class = SnowflakeQueryLogSerializer
    permission_classes = [permissions.IsAuthenticated, IsWarehouseUser]
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ["status"]
    ordering_fields = ["created_at", "execution_time_ms", "rows_returned"]
    ordering = ["-created_at"]

    def get_queryset(self):
        return SnowflakeQueryLog.objects.filter(user=self.request.user)


# ──────────────────────────────────── SnowflakeTablesAPIView

@extend_schema(
    summary="List all Snowflake tables in the configured schema",
    tags=["warehouse"],
    responses={
        200: {
            "type": "object",
            "properties": {
                "tables": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "schema_name": {"type": "string"},
                            "table_name": {"type": "string"},
                            "table_type": {"type": "string"},
                            "row_count": {"type": "integer"},
                            "bytes": {"type": "integer"},
                            "last_altered": {"type": "string"},
                        },
                    },
                },
                "count": {"type": "integer"},
            },
        }
    },
)
class SnowflakeTablesAPIView(APIView):
    """GET the full list of tables visible to the Snowflake service account."""

    permission_classes = [permissions.IsAuthenticated, IsWarehouseUser]

    def get(self, request) -> Response:
        try:
            client = SnowflakeClient()
            df = client.get_tables()
            df = df.where(pd.notnull(df), None)
            tables = df.to_dict(orient="records")
        except (SnowflakeQueryError, Exception) as exc:
            logger.error("SnowflakeTablesAPIView error: %s", exc)
            return Response({"error": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        scope = get_facility_scope(request.user)
        tables = filter_tables_for_scope(tables, scope)
        return Response({"tables": tables, "count": len(tables)})
