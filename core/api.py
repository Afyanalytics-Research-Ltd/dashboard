"""
DRF ViewSets for core models.
"""

import logging

from django.db.models import QuerySet
from drf_spectacular.utils import extend_schema, extend_schema_view
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response

from .models import AuditLog, Client, Facility, Notification, SystemSettings
from .permissions import IsAdminOrReadOnly, IsClientAdmin, IsSuperuser
from .serializers import (
    AuditLogSerializer,
    ClientDetailSerializer,
    ClientSerializer,
    FacilitySerializer,
    NotificationMarkReadSerializer,
    NotificationSerializer,
    SystemSettingsSerializer,
)

logger = logging.getLogger(__name__)


@extend_schema(tags=['core'])
@extend_schema_view(
    list=extend_schema(summary='List all clients'),
    retrieve=extend_schema(summary='Retrieve a client'),
    create=extend_schema(summary='Create a client'),
    update=extend_schema(summary='Update a client'),
    partial_update=extend_schema(summary='Partially update a client'),
    destroy=extend_schema(summary='Delete a client'),
)
class ClientViewSet(viewsets.ModelViewSet):
    queryset = Client.objects.all()
    serializer_class = ClientSerializer
    permission_classes = [IsAuthenticated, IsAdminOrReadOnly]
    search_fields = ['name', 'slug']
    filterset_fields = ['is_active']
    ordering_fields = ['name', 'created_at']
    ordering = ['name']

    def get_serializer_class(self):
        if self.action == 'retrieve':
            return ClientDetailSerializer
        return ClientSerializer

    @extend_schema(summary='List facilities for a client')
    @action(detail=True, methods=['get'], url_path='facilities')
    def facilities(self, request: Request, pk=None) -> Response:
        client = self.get_object()
        facilities = client.facilities.filter(is_active=True)
        serializer = FacilitySerializer(facilities, many=True, context={'request': request})
        return Response(serializer.data)


@extend_schema(tags=['core'])
@extend_schema_view(
    list=extend_schema(summary='List all facilities'),
    retrieve=extend_schema(summary='Retrieve a facility'),
    create=extend_schema(summary='Create a facility'),
    update=extend_schema(summary='Update a facility'),
    partial_update=extend_schema(summary='Partially update a facility'),
    destroy=extend_schema(summary='Delete a facility'),
)
class FacilityViewSet(viewsets.ModelViewSet):
    queryset = Facility.objects.select_related('client').all()
    serializer_class = FacilitySerializer
    permission_classes = [IsAuthenticated, IsAdminOrReadOnly]
    search_fields = ['name', 'slug', 'client__name']
    filterset_fields = ['is_active', 'client']
    ordering_fields = ['name', 'created_at', 'client__name']
    ordering = ['name']


@extend_schema(tags=['core'])
class AuditLogViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only access to audit logs. Superusers see all; others see only their own."""

    serializer_class = AuditLogSerializer
    permission_classes = [IsAuthenticated]
    filterset_fields = ['action', 'resource']
    search_fields = ['resource', 'resource_id', 'user__username']
    ordering_fields = ['timestamp', 'action', 'resource']
    ordering = ['-timestamp']

    def get_queryset(self) -> QuerySet:
        user = self.request.user
        qs = AuditLog.objects.select_related('user')
        if user.is_superuser:
            return qs
        return qs.filter(user=user)


@extend_schema(tags=['core'])
class NotificationViewSet(viewsets.ModelViewSet):
    """User's own notifications. POST is disabled — notifications are system-generated."""

    serializer_class = NotificationSerializer
    permission_classes = [IsAuthenticated]
    filterset_fields = ['is_read', 'notification_type']
    ordering_fields = ['created_at']
    ordering = ['-created_at']
    http_method_names = ['get', 'patch', 'delete', 'head', 'options']

    def get_queryset(self) -> QuerySet:
        return Notification.objects.filter(user=self.request.user)

    @extend_schema(
        summary='Mark multiple notifications as read',
        request=NotificationMarkReadSerializer,
        responses={200: NotificationSerializer(many=True)},
    )
    @action(detail=False, methods=['post'], url_path='mark-read')
    def mark_read(self, request: Request) -> Response:
        serializer = NotificationMarkReadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        ids = serializer.validated_data['ids']
        updated = (
            Notification.objects
            .filter(user=request.user, id__in=ids)
            .update(is_read=True)
        )
        logger.info('User %s marked %d notification(s) as read', request.user.username, updated)
        return Response({'marked_read': updated})

    @extend_schema(summary='Mark all notifications as read')
    @action(detail=False, methods=['post'], url_path='mark-all-read')
    def mark_all_read(self, request: Request) -> Response:
        updated = (
            Notification.objects
            .filter(user=request.user, is_read=False)
            .update(is_read=True)
        )
        return Response({'marked_read': updated})


@extend_schema(tags=['core'])
@extend_schema_view(
    list=extend_schema(summary='List system settings'),
    retrieve=extend_schema(summary='Retrieve a setting'),
    create=extend_schema(summary='Create a setting (superuser)'),
    update=extend_schema(summary='Update a setting (superuser)'),
    partial_update=extend_schema(summary='Partially update a setting (superuser)'),
    destroy=extend_schema(summary='Delete a setting (superuser)'),
)
class SystemSettingsViewSet(viewsets.ModelViewSet):
    serializer_class = SystemSettingsSerializer
    filterset_fields = ['is_public']
    search_fields = ['key', 'description']
    ordering_fields = ['key', 'updated_at']
    ordering = ['key']

    def get_permissions(self):
        if self.action in ('list', 'retrieve'):
            return [IsAuthenticated()]
        return [IsAuthenticated(), IsSuperuser()]

    def get_queryset(self) -> QuerySet:
        user = self.request.user
        if user.is_superuser:
            return SystemSettings.objects.all()
        return SystemSettings.objects.filter(is_public=True)

    def perform_create(self, serializer):
        serializer.save(updated_by=self.request.user)

    def perform_update(self, serializer):
        serializer.save(updated_by=self.request.user)
