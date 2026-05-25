"""
Analytics app DRF API views.
"""

import logging

from django.db.models import Count
from drf_spectacular.utils import extend_schema, extend_schema_view
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.filters import OrderingFilter, SearchFilter
from rest_framework.response import Response
from rest_framework.views import APIView
from django_filters.rest_framework import DjangoFilterBackend

from core.models import AuditLog
from .models import Dashboard
from .serializers import DashboardListSerializer, DashboardSerializer
from .views import _sync_dashboards_for_client, _get_client_obj

logger = logging.getLogger(__name__)


class IsAuthenticatedOrSuperuserWrite(permissions.BasePermission):
    """Authenticated users can read; only superusers can write."""

    def has_permission(self, request, view):
        if not request.user or not request.user.is_authenticated:
            return False
        if request.method in permissions.SAFE_METHODS:
            return True
        return request.user.is_superuser


@extend_schema_view(
    list=extend_schema(
        description='List all dashboards for the current user (filtered by client).',
        tags=['analytics'],
    ),
    retrieve=extend_schema(
        description='Get full details for a single dashboard.',
        tags=['analytics'],
    ),
    create=extend_schema(
        description='Create a new dashboard (superuser only).',
        tags=['analytics'],
    ),
    update=extend_schema(
        description='Replace a dashboard record (superuser only).',
        tags=['analytics'],
    ),
    partial_update=extend_schema(
        description='Partially update a dashboard (superuser only).',
        tags=['analytics'],
    ),
    destroy=extend_schema(
        description='Delete a dashboard (superuser only).',
        tags=['analytics'],
    ),
)
class DashboardViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Dashboard resources.

    Non-superusers can only list/retrieve dashboards belonging to their client.
    Superusers have full CRUD access.
    """

    permission_classes = [IsAuthenticatedOrSuperuserWrite]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ['category', 'is_active']
    search_fields = ['name', 'description']
    ordering_fields = ['name', 'view_count', 'created_at', 'order']
    ordering = ['order', 'name']

    def get_serializer_class(self):
        if self.action == 'list':
            return DashboardListSerializer
        return DashboardSerializer

    def get_queryset(self):
        user = self.request.user
        qs = Dashboard.objects.select_related('client', 'facility', 'created_by')
        if user.is_superuser:
            return qs
        client_obj = _get_client_obj(user)
        if client_obj:
            return qs.filter(client=client_obj, is_active=True)
        return qs.none()

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)
        logger.info('Dashboard created via API by %s', self.request.user.username)


@extend_schema(
    description='Trigger a filesystem sync of dashboards for all active clients (superuser only).',
    tags=['analytics'],
    responses={200: {'description': 'Sync result summary'}},
)
class DashboardSyncAPIView(APIView):
    """POST: superuser-only dashboard filesystem sync."""

    permission_classes = [permissions.IsAdminUser]

    def post(self, request, *args, **kwargs):
        from core.models import Client
        results = {}
        for client in Client.objects.filter(is_active=True):
            res = _sync_dashboards_for_client(client.slug, client)
            results[client.name] = res
        logger.info('API dashboard sync triggered by %s', request.user.username)
        return Response({'ok': True, 'results': results})


@extend_schema(
    description='Return dashboard counts broken down by category.',
    tags=['analytics'],
    responses={200: {'description': 'Category counts'}},
)
class DashboardStatsAPIView(APIView):
    """GET: aggregate stats by category for the current user's client."""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, *args, **kwargs):
        user = request.user
        qs = Dashboard.objects.filter(is_active=True)
        if not user.is_superuser:
            client_obj = _get_client_obj(user)
            if client_obj:
                qs = qs.filter(client=client_obj)
            else:
                qs = qs.none()

        by_category = (
            qs.values('category')
            .annotate(count=Count('id'))
            .order_by('category')
        )
        return Response({
            'total': qs.count(),
            'by_category': list(by_category),
        })
