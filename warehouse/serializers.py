"""DRF serializers for the warehouse module."""

from django.contrib.auth import get_user_model
from rest_framework import serializers

from .models import SnowflakeQueryLog, TrackedSpreadsheet

User = get_user_model()


class TrackedSpreadsheetSerializer(serializers.ModelSerializer):
    created_by_username = serializers.CharField(
        source="created_by.username", read_only=True, default=""
    )
    client_name = serializers.CharField(
        source="client.name", read_only=True, default=""
    )
    absolute_url = serializers.SerializerMethodField()

    class Meta:
        model = TrackedSpreadsheet
        fields = [
            "id",
            "spreadsheet_id",
            "title",
            "web_view_link",
            "client",
            "client_name",
            "created_by",
            "created_by_username",
            "absolute_url",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_by", "created_at", "updated_at"]

    def get_absolute_url(self, obj: TrackedSpreadsheet) -> str:
        request = self.context.get("request")
        url = obj.get_absolute_url()
        if request:
            return request.build_absolute_uri(url)
        return url

    def create(self, validated_data: dict) -> TrackedSpreadsheet:
        request = self.context.get("request")
        if request and request.user.is_authenticated:
            validated_data["created_by"] = request.user
        return super().create(validated_data)


class SnowflakeQueryLogSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)
    status_display = serializers.CharField(source="get_status_display", read_only=True)

    class Meta:
        model = SnowflakeQueryLog
        fields = [
            "id",
            "user",
            "username",
            "query",
            "status",
            "status_display",
            "rows_returned",
            "execution_time_ms",
            "error_message",
            "created_at",
        ]
        read_only_fields = fields  # query logs are immutable via API


class SnowflakeQuerySerializer(serializers.Serializer):
    """Validates an incoming Snowflake query request."""

    query = serializers.CharField(
        max_length=100_000,
        trim_whitespace=True,
        error_messages={"required": "Please provide a SQL query."},
    )
    max_rows = serializers.IntegerField(
        min_value=1,
        max_value=10_000,
        default=10_000,
        required=False,
    )

    def validate_query(self, value: str) -> str:
        import re
        blocked = frozenset({
            'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE',
            'INSERT', 'UPDATE', 'GRANT', 'REVOKE',
        })
        pattern = re.compile(
            r'\b(' + '|'.join(blocked) + r')\b',
            re.IGNORECASE,
        )
        match = pattern.search(value)
        if match:
            raise serializers.ValidationError(
                f"The keyword '{match.group(0).upper()}' is not permitted. "
                "Only read-only SELECT queries are allowed."
            )
        return value
