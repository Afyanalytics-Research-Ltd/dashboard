"""
DRF serializers for core models.
"""

from django.contrib.auth import get_user_model
from rest_framework import serializers

from .models import AuditLog, Client, Facility, Notification, SystemSettings

User = get_user_model()


class ClientSerializer(serializers.ModelSerializer):
    active_facilities_count = serializers.IntegerField(read_only=True)
    logo_url = serializers.SerializerMethodField()

    class Meta:
        model = Client
        fields = [
            'id', 'name', 'slug', 'logo', 'logo_url',
            'is_active', 'active_facilities_count',
            'created_at', 'updated_at',
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']

    def get_logo_url(self, obj: Client) -> str | None:
        if obj.logo:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.logo.url)
            return obj.logo.url
        return None


class FacilitySerializer(serializers.ModelSerializer):
    client_name = serializers.CharField(source='client.name', read_only=True)

    class Meta:
        model = Facility
        fields = [
            'id', 'client', 'client_name', 'name', 'slug',
            'is_active', 'created_at', 'updated_at',
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class FacilityNestedSerializer(serializers.ModelSerializer):
    """Compact serializer for embedding facilities inside a Client."""

    class Meta:
        model = Facility
        fields = ['id', 'name', 'slug', 'is_active']


class ClientDetailSerializer(ClientSerializer):
    """Client with embedded facilities list."""

    facilities = FacilityNestedSerializer(many=True, read_only=True)

    class Meta(ClientSerializer.Meta):
        fields = ClientSerializer.Meta.fields + ['facilities']


class AuditLogSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True, default='')

    class Meta:
        model = AuditLog
        fields = [
            'id', 'user', 'username', 'action', 'resource',
            'resource_id', 'detail', 'ip_address', 'timestamp',
        ]
        read_only_fields = fields  # audit logs are immutable via API


class NotificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Notification
        fields = [
            'id', 'title', 'message', 'notification_type',
            'is_read', 'link', 'created_at',
        ]
        read_only_fields = ['id', 'created_at']


class NotificationMarkReadSerializer(serializers.Serializer):
    """Minimal serializer for the mark-read action."""
    ids = serializers.ListField(
        child=serializers.IntegerField(min_value=1),
        allow_empty=False,
    )


class SystemSettingsSerializer(serializers.ModelSerializer):
    updated_by_username = serializers.CharField(
        source='updated_by.username',
        read_only=True,
        default='',
    )

    class Meta:
        model = SystemSettings
        fields = [
            'id', 'key', 'value', 'description', 'is_public',
            'updated_by', 'updated_by_username', 'updated_at',
        ]
        read_only_fields = ['id', 'updated_by', 'updated_at']

    def validate_key(self, value: str) -> str:
        return value.lower().replace(' ', '_')
