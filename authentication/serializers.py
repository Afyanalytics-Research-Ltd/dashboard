"""
DRF serializers for the authentication app.
"""

import logging

from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from rest_framework import serializers

from core.models import Notification
from .models import UserProfile

logger = logging.getLogger(__name__)
User = get_user_model()


# ---------------------------------------------------------------------------
# UserSerializer
# ---------------------------------------------------------------------------

class UserSerializer(serializers.ModelSerializer):
    """Read / write serializer for the built-in User model."""

    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'is_active', 'date_joined']
        read_only_fields = ['id', 'date_joined']


# ---------------------------------------------------------------------------
# UserProfileSerializer
# ---------------------------------------------------------------------------

class UserProfileSerializer(serializers.ModelSerializer):
    """Full profile serializer, includes nested User."""

    user = UserSerializer(read_only=True)
    role_display = serializers.CharField(source='get_role_display', read_only=True)
    display_name = serializers.CharField(read_only=True)
    initials = serializers.CharField(read_only=True)
    avatar_url = serializers.SerializerMethodField()
    client_name = serializers.SerializerMethodField()
    facility_name = serializers.SerializerMethodField()

    class Meta:
        model = UserProfile
        fields = [
            'id',
            'user',
            'phone_number',
            'client',
            'client_name',
            'facility',
            'facility_name',
            'job_title',
            'bio',
            'avatar',
            'avatar_url',
            'role',
            'role_display',
            'display_name',
            'initials',
            'is_verified',
            'last_login_ip',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['id', 'user', 'last_login_ip', 'created_at', 'updated_at']

    def get_avatar_url(self, obj) -> str | None:
        if obj.avatar:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.avatar.url)
            return obj.avatar.url
        return None

    def get_client_name(self, obj) -> str | None:
        return str(obj.client) if obj.client_id else None

    def get_facility_name(self, obj) -> str | None:
        return str(obj.facility) if obj.facility_id else None


# ---------------------------------------------------------------------------
# UserRegistrationSerializer
# ---------------------------------------------------------------------------

class UserRegistrationSerializer(serializers.Serializer):
    """Handles user registration via the API."""

    username = serializers.CharField(max_length=150)
    email = serializers.EmailField()
    first_name = serializers.CharField(max_length=150, required=False, default='')
    last_name = serializers.CharField(max_length=150, required=False, default='')
    phone_number = serializers.CharField(max_length=32, required=False, default='')
    password = serializers.CharField(write_only=True, min_length=8)
    confirm_password = serializers.CharField(write_only=True)

    def validate_username(self, value):
        if User.objects.filter(username=value).exists():
            raise serializers.ValidationError('A user with this username already exists.')
        return value

    def validate_email(self, value):
        email = value.lower()
        if User.objects.filter(email__iexact=email).exists():
            raise serializers.ValidationError('A user with this email already exists.')
        return email

    def validate(self, attrs):
        if attrs['password'] != attrs['confirm_password']:
            raise serializers.ValidationError({'confirm_password': 'Passwords do not match.'})
        try:
            validate_password(attrs['password'])
        except Exception as exc:
            raise serializers.ValidationError({'password': list(exc.messages)})
        return attrs

    def create(self, validated_data):
        validated_data.pop('confirm_password')
        phone = validated_data.pop('phone_number', '')
        user = User.objects.create_user(
            username=validated_data['username'],
            email=validated_data['email'],
            password=validated_data['password'],
            first_name=validated_data.get('first_name', ''),
            last_name=validated_data.get('last_name', ''),
        )
        if phone:
            try:
                user.profile.phone_number = phone
                user.profile.save(update_fields=['phone_number'])
            except Exception as exc:
                logger.warning('Could not save phone for user %s: %s', user.pk, exc)
        logger.info('New user registered via API: %s (pk=%s)', user.username, user.pk)
        return user


# ---------------------------------------------------------------------------
# PasswordChangeSerializer
# ---------------------------------------------------------------------------

class PasswordChangeSerializer(serializers.Serializer):
    """API password change: requires old password for verification."""

    old_password = serializers.CharField(write_only=True)
    new_password = serializers.CharField(write_only=True, min_length=8)
    confirm_password = serializers.CharField(write_only=True)

    def validate_old_password(self, value):
        user = self.context['request'].user
        if not user.check_password(value):
            raise serializers.ValidationError('Current password is incorrect.')
        return value

    def validate(self, attrs):
        if attrs['new_password'] != attrs['confirm_password']:
            raise serializers.ValidationError({'confirm_password': 'Passwords do not match.'})
        try:
            validate_password(attrs['new_password'], self.context['request'].user)
        except Exception as exc:
            raise serializers.ValidationError({'new_password': list(exc.messages)})
        return attrs

    def save(self, **kwargs):
        user = self.context['request'].user
        user.set_password(self.validated_data['new_password'])
        user.save()
        return user


# ---------------------------------------------------------------------------
# NotificationSerializer
# ---------------------------------------------------------------------------

class AuthNotificationSerializer(serializers.ModelSerializer):
    """Serializer for core.Notification (authentication module)."""

    notification_type_display = serializers.CharField(
        source='get_notification_type_display', read_only=True
    )

    class Meta:
        model = Notification
        fields = [
            'id',
            'title',
            'message',
            'notification_type',
            'notification_type_display',
            'is_read',
            'link',
            'created_at',
        ]
        read_only_fields = ['id', 'created_at']

# Alias for backwards compatibility
NotificationSerializer = AuthNotificationSerializer
