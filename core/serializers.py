"""
DRF serializers for core models.
"""

from django.contrib.auth import get_user_model
from rest_framework import serializers

from .models import AuditLog, Client, Facility, Notification, SystemSettings

User = get_user_model()


class ClientSerializer(serializers.ModelSerializer):
    """Serializer for :class:`core.models.Client` — the standard list/write representation.

    Adds two computed read-only fields:
    - ``active_facilities_count``: how many active facilities the client has.
    - ``logo_url``: an absolute URL to the logo image (or ``null``).

    Non-technical explanation:
        Converts a Client database record into a JSON object that the API
        can return to callers.  Includes a count of active branches and a
        direct link to the client's logo image.
    """

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
        """Build an absolute URL for the client's logo image.

        Uses the request object from the serializer context so the URL
        includes the correct scheme and hostname (e.g.
        ``https://app.afya.ai/media/clients/logos/pharmaplus.png``).

        Args:
            obj: The :class:`Client` being serialized.

        Returns:
            An absolute URL string if a logo exists, otherwise ``None``.
        """
        if obj.logo:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.logo.url)
            return obj.logo.url
        return None


class FacilitySerializer(serializers.ModelSerializer):
    """Serializer for :class:`core.models.Facility`.

    Adds ``client_name`` as a denormalised read-only field so callers can
    display the owning client's name without a second API call.

    Non-technical explanation:
        Converts a Facility record (e.g. "Kisumu Specialist Hospital") into
        a JSON object.  The ``client_name`` field tells callers which
        organisation this facility belongs to, without them having to look
        it up separately.
    """

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
    """Read-only serializer for :class:`core.models.AuditLog`.

    All fields are marked read-only because audit logs are an immutable
    historical record — they must never be modified through the API.
    The ``username`` field is denormalised for convenient display.

    Non-technical explanation:
        Converts an audit log entry into JSON for the API.  Because audit
        logs are like legal records, the API deliberately prevents anyone
        from changing or deleting them — you can only read them.
    """

    username = serializers.CharField(source='user.username', read_only=True, default='')

    class Meta:
        model = AuditLog
        fields = [
            'id', 'user', 'username', 'action', 'resource',
            'resource_id', 'detail', 'ip_address', 'timestamp',
        ]
        read_only_fields = fields  # audit logs are immutable via API


class NotificationSerializer(serializers.ModelSerializer):
    """Serializer for :class:`core.models.Notification`.

    Exposes the notification's text, type (info/success/warning/danger),
    read status, and optional navigation link.  The ``id`` and
    ``created_at`` fields are read-only; callers can PATCH ``is_read``
    to mark individual notifications as read.

    Non-technical explanation:
        Converts a notification (an in-app message) into JSON so the UI
        can display it in the notification bell dropdown.
    """

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
    """Serializer for :class:`core.models.SystemSettings`.

    Adds ``updated_by_username`` as a convenience read-only field.
    The ``key`` field is normalised to lowercase with underscores on write
    so that ``"Max Export Rows"`` and ``"max_export_rows"`` refer to the
    same setting.

    Non-technical explanation:
        Converts a platform configuration entry (key + value) into JSON.
        Includes the name of the last person who changed the setting so
        there is an audit trail of who touched what.
    """

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
        """Normalise the setting key to lowercase with underscores.

        Ensures that ``"Max Export Rows"``, ``"max export rows"``, and
        ``"max_export_rows"`` all resolve to the same key.

        Args:
            value: The raw key string submitted by the caller.

        Returns:
            The normalised key string, e.g. ``"max_export_rows"``.
        """
        return value.lower().replace(' ', '_')
