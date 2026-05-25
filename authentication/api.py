"""
DRF API views for the authentication app.

All views use drf-spectacular decorators for schema generation.
"""

import logging

from django.contrib.auth import get_user_model, update_session_auth_hash
from drf_spectacular.utils import (
    OpenApiParameter,
    OpenApiResponse,
    extend_schema,
    extend_schema_view,
)
from rest_framework import filters, mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from core.models import AuditLog, Notification
from core.permissions import IsClientAdmin, IsOwnerOrAdmin
from .models import UserProfile
from .serializers import (
    AuthNotificationSerializer,
    PasswordChangeSerializer,
    UserProfileSerializer,
    UserRegistrationSerializer,
    UserSerializer,
)

NotificationSerializer = AuthNotificationSerializer

logger = logging.getLogger(__name__)
User = get_user_model()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client_ip(request) -> str | None:
    xff = request.META.get('HTTP_X_FORWARDED_FOR')
    if xff:
        return xff.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR')


# ---------------------------------------------------------------------------
# UserViewSet
# ---------------------------------------------------------------------------

@extend_schema_view(
    list=extend_schema(
        summary='List users',
        description='Returns a paginated list of users. Write access requires Client Admin role.',
        tags=['Users'],
    ),
    retrieve=extend_schema(
        summary='Get user',
        tags=['Users'],
    ),
    create=extend_schema(
        summary='Create user (admin)',
        tags=['Users'],
    ),
    update=extend_schema(
        summary='Update user',
        tags=['Users'],
    ),
    partial_update=extend_schema(
        summary='Partial update user',
        tags=['Users'],
    ),
    destroy=extend_schema(
        summary='Delete user (admin)',
        tags=['Users'],
    ),
)
class UserViewSet(viewsets.ModelViewSet):
    """
    CRUD for User accounts.
    - List/Retrieve: any authenticated user (own record for non-admins)
    - Create/Update/Delete: Client Admin only
    """

    serializer_class = UserSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['username', 'email', 'first_name', 'last_name']
    ordering_fields = ['username', 'date_joined']
    ordering = ['-date_joined']

    def get_queryset(self):
        user = self.request.user
        if user.is_superuser or (hasattr(user, 'profile') and user.profile.is_client_admin):
            return User.objects.all()
        return User.objects.filter(pk=user.pk)

    def get_permissions(self):
        if self.action in ('create', 'destroy'):
            return [IsAuthenticated(), IsClientAdmin()]
        return [IsAuthenticated()]


# ---------------------------------------------------------------------------
# UserProfileViewSet
# ---------------------------------------------------------------------------

@extend_schema_view(
    list=extend_schema(summary='List profiles', tags=['Profiles']),
    retrieve=extend_schema(summary='Get profile', tags=['Profiles']),
    update=extend_schema(summary='Update profile', tags=['Profiles']),
    partial_update=extend_schema(summary='Partial update profile', tags=['Profiles']),
)
class UserProfileViewSet(
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    mixins.UpdateModelMixin,
    viewsets.GenericViewSet,
):
    """
    CRUD for UserProfiles.
    Includes /me/ action for the current user's own profile.
    """

    serializer_class = UserProfileSerializer
    permission_classes = [IsAuthenticated, IsOwnerOrAdmin]
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['user__username', 'user__email', 'phone_number', 'job_title']
    ordering_fields = ['created_at', 'updated_at']
    ordering = ['-created_at']

    def get_queryset(self):
        user = self.request.user
        if user.is_superuser or (hasattr(user, 'profile') and user.profile.is_client_admin):
            return UserProfile.objects.select_related('user', 'client', 'facility').all()
        return UserProfile.objects.select_related('user', 'client', 'facility').filter(user=user)

    @extend_schema(
        summary='Get current user profile',
        description='Returns the profile for the currently authenticated user.',
        tags=['Profiles'],
        responses={200: UserProfileSerializer},
    )
    @action(detail=False, methods=['get', 'patch'], url_path='me')
    def me(self, request):
        """Retrieve or partially update the current user's profile."""
        profile, _ = UserProfile.objects.get_or_create(user=request.user)
        if request.method == 'PATCH':
            serializer = self.get_serializer(profile, data=request.data, partial=True)
            serializer.is_valid(raise_exception=True)
            serializer.save()
            logger.info('User %s updated their profile via API', request.user.username)
            return Response(serializer.data)
        serializer = self.get_serializer(profile)
        return Response(serializer.data)


# ---------------------------------------------------------------------------
# ChangePasswordView
# ---------------------------------------------------------------------------

@extend_schema(
    summary='Change password',
    description='Changes the password for the currently authenticated user.',
    tags=['Auth'],
    request=PasswordChangeSerializer,
    responses={
        200: OpenApiResponse(description='Password changed successfully.'),
        400: OpenApiResponse(description='Validation error.'),
    },
)
class ChangePasswordView(APIView):
    """API endpoint for changing the current user's password."""

    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = PasswordChangeSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            user = serializer.save()
            update_session_auth_hash(request, user)
            AuditLog.log(
                user=request.user,
                action='update',
                resource='authentication.password',
                resource_id=str(request.user.pk),
                detail='Password changed via API',
                ip_address=_get_client_ip(request),
                user_agent=request.META.get('HTTP_USER_AGENT', ''),
            )
            logger.info('User %s changed password via API', request.user.username)
            return Response({'detail': 'Password changed successfully.'}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# ---------------------------------------------------------------------------
# NotificationViewSet
# ---------------------------------------------------------------------------

@extend_schema_view(
    list=extend_schema(
        summary='List notifications',
        description='Returns paginated notifications for the current user.',
        tags=['Notifications'],
        parameters=[
            OpenApiParameter('is_read', bool, description='Filter by read status'),
            OpenApiParameter('notification_type', str, description='Filter by type'),
        ],
    ),
    retrieve=extend_schema(summary='Get notification', tags=['Notifications']),
    destroy=extend_schema(summary='Delete notification', tags=['Notifications']),
)
class NotificationViewSet(
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    mixins.DestroyModelMixin,
    viewsets.GenericViewSet,
):
    """
    Read-only notification management for the current user.
    Provides /mark_read/ and /mark_all_read/ actions.
    """

    serializer_class = NotificationSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [filters.OrderingFilter]
    ordering_fields = ['created_at']
    ordering = ['-created_at']

    def get_queryset(self):
        qs = Notification.objects.filter(user=self.request.user)
        is_read = self.request.query_params.get('is_read')
        if is_read is not None:
            qs = qs.filter(is_read=is_read.lower() in ('true', '1'))
        ntype = self.request.query_params.get('notification_type')
        if ntype:
            qs = qs.filter(notification_type=ntype)
        return qs

    @extend_schema(
        summary='Mark notification as read',
        tags=['Notifications'],
        responses={200: OpenApiResponse(description='Marked as read.')},
    )
    @action(detail=True, methods=['post'], url_path='mark_read')
    def mark_read(self, request, pk=None):
        """Mark a single notification as read."""
        notification = self.get_object()
        notification.mark_read()
        logger.debug('Notification %s marked read by user %s', pk, request.user.username)
        return Response({'detail': 'Marked as read.', 'id': pk}, status=status.HTTP_200_OK)

    @extend_schema(
        summary='Mark all notifications as read',
        tags=['Notifications'],
        responses={200: OpenApiResponse(description='All notifications marked as read.')},
    )
    @action(detail=False, methods=['post'], url_path='mark_all_read')
    def mark_all_read(self, request):
        """Mark all of the current user's notifications as read."""
        updated = Notification.objects.filter(user=request.user, is_read=False).update(is_read=True)
        logger.info('User %s marked %d notifications read via API', request.user.username, updated)
        return Response({'detail': f'{updated} notification(s) marked as read.'}, status=status.HTTP_200_OK)


# ---------------------------------------------------------------------------
# UserActivityView (API)
# ---------------------------------------------------------------------------

@extend_schema(
    summary='My activity log',
    description='Returns a paginated list of audit log entries for the current user.',
    tags=['Auth'],
    parameters=[
        OpenApiParameter('action', str, description='Filter by action type'),
        OpenApiParameter('q', str, description='Search by resource name'),
    ],
)
class UserActivityView(APIView):
    """Returns the current user's audit log entries."""

    permission_classes = [IsAuthenticated]

    def get(self, request):
        from rest_framework.pagination import PageNumberPagination
        from rest_framework import serializers as drf_serializers

        class AuditLogSerializer(drf_serializers.ModelSerializer):
            action_display = drf_serializers.CharField(source='get_action_display', read_only=True)

            class Meta:
                model = AuditLog
                fields = ['id', 'action', 'action_display', 'resource', 'resource_id',
                          'detail', 'ip_address', 'timestamp']

        qs = AuditLog.objects.filter(user=request.user).order_by('-timestamp')
        q = request.query_params.get('q', '').strip()
        if q:
            qs = qs.filter(resource__icontains=q)
        action_filter = request.query_params.get('action', '').strip()
        if action_filter:
            qs = qs.filter(action=action_filter)

        paginator = PageNumberPagination()
        paginator.page_size = 25
        page = paginator.paginate_queryset(qs, request)
        serializer = AuditLogSerializer(page, many=True)
        return paginator.get_paginated_response(serializer.data)
