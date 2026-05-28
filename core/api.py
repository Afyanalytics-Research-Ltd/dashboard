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
    """Full CRUD API for :class:`core.models.Client` records.

    Read operations (GET) are available to any authenticated user.
    Write operations (POST, PUT, PATCH, DELETE) require the
    ``Client Admin`` role or superuser status.

    Non-technical explanation:
        A "Client" is a healthcare organisation (e.g. a hospital network)
        using the platform.  This API lets admins create, update, or remove
        client records and lets any logged-in user look them up.

    Endpoints (prefixed ``/api/v1/core/clients/``):
        - ``GET  /``         — paginated list
        - ``POST /``         — create new client (admin only)
        - ``GET  /{id}/``    — retrieve a single client with facilities
        - ``PUT  /{id}/``    — full update (admin only)
        - ``PATCH/{id}/``    — partial update (admin only)
        - ``DELETE/{id}/``   — delete (admin only)
        - ``GET  /{id}/facilities/`` — active facilities for this client
    """

    queryset = Client.objects.all()
    serializer_class = ClientSerializer
    permission_classes = [IsAuthenticated, IsAdminOrReadOnly]
    search_fields = ['name', 'slug']
    filterset_fields = ['is_active']
    ordering_fields = ['name', 'created_at']
    ordering = ['name']

    def get_serializer_class(self):
        """Return a richer serializer (with nested facilities) for detail views.

        Returns:
            :class:`ClientDetailSerializer` for ``retrieve`` actions,
            :class:`ClientSerializer` for all other actions.
        """
        if self.action == 'retrieve':
            return ClientDetailSerializer
        return ClientSerializer

    @extend_schema(summary='List facilities for a client')
    @action(detail=True, methods=['get'], url_path='facilities')
    def facilities(self, request: Request, pk=None) -> Response:
        """Return the active facilities belonging to a specific client.

        Args:
            request: The incoming GET request.
            pk: Primary key of the :class:`Client`.

        Returns:
            A JSON list of :class:`FacilitySerializer` representations for
            all active (``is_active=True``) facilities under this client.
        """
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
    """Full CRUD API for :class:`core.models.Facility` records.

    Facilities are physical or virtual locations (clinics, wards, branches)
    that belong to a :class:`core.models.Client`.  Same permission model as
    :class:`ClientViewSet`: anyone authenticated can read; only Client Admins
    or superusers can write.

    Non-technical explanation:
        A "Facility" is a specific hospital, clinic, or health post under a
        client organisation.  This API exposes facility management — listing,
        creating, and updating them.
    """

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
        """Return audit logs scoped to the requesting user's access level.

        Superusers see every log entry on the platform; regular users see
        only their own records.

        Returns:
            A QuerySet of :class:`AuditLog` objects, newest first.
        """
        user = self.request.user
        qs = AuditLog.objects.select_related('user')
        if user.is_superuser:
            return qs
        return qs.filter(user=user)


@extend_schema(tags=['core'])
class NotificationViewSet(viewsets.ModelViewSet):
    """Read/manage the current user's own notifications via the REST API.

    POST (create) is intentionally disabled — notifications are created by
    the platform, not by users themselves.  Users can read, filter, and
    delete their notifications, and use the ``/mark-read/`` and
    ``/mark-all-read/`` actions to clear the unread count.

    Non-technical explanation:
        This API is what powers the notification bell in the UI — it fetches
        your messages and lets you mark them as read or delete old ones.
        Only your own notifications are ever returned; you cannot see
        anyone else's.
    """

    serializer_class = NotificationSerializer
    permission_classes = [IsAuthenticated]
    filterset_fields = ['is_read', 'notification_type']
    ordering_fields = ['created_at']
    ordering = ['-created_at']
    http_method_names = ['get', 'patch', 'delete', 'head', 'options']

    def get_queryset(self) -> QuerySet:
        """Return only the notifications belonging to the requesting user.

        Returns:
            A QuerySet of :class:`Notification` objects, newest first.
        """
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
    """REST API for platform-wide configuration settings.

    Read access (list/retrieve): any authenticated user, but non-superusers
    can only see settings flagged as ``is_public=True``.  Write operations
    are restricted to superusers only.

    Non-technical explanation:
        This API exposes the platform's settings store — like an online
        configuration panel.  Regular users can read public settings;
        only administrators can change anything.
    """

    serializer_class = SystemSettingsSerializer
    filterset_fields = ['is_public']
    search_fields = ['key', 'description']
    ordering_fields = ['key', 'updated_at']
    ordering = ['key']

    def get_permissions(self):
        """Return stricter permissions for write operations.

        Returns:
            For ``list`` and ``retrieve``: ``[IsAuthenticated]``.
            For all other actions: ``[IsAuthenticated, IsSuperuser]``.
        """
        if self.action in ('list', 'retrieve'):
            return [IsAuthenticated()]
        return [IsAuthenticated(), IsSuperuser()]

    def get_queryset(self) -> QuerySet:
        """Return settings visible to the requesting user.

        Superusers see all settings; regular users see only public ones.

        Returns:
            A QuerySet of :class:`SystemSettings` objects.
        """
        user = self.request.user
        if user.is_superuser:
            return SystemSettings.objects.all()
        return SystemSettings.objects.filter(is_public=True)

    def perform_create(self, serializer):
        """Save a new setting, recording the creating superuser as ``updated_by``.

        Args:
            serializer: The validated :class:`SystemSettingsSerializer`.
        """
        serializer.save(updated_by=self.request.user)

    def perform_update(self, serializer):
        """Update an existing setting, recording the editing superuser as ``updated_by``.

        Args:
            serializer: The validated :class:`SystemSettingsSerializer`.
        """
        serializer.save(updated_by=self.request.user)
