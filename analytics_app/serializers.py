"""
Analytics app serializers.
"""

from rest_framework import serializers

from .models import Dashboard


class DashboardSerializer(serializers.ModelSerializer):
    """Full serializer including computed URL."""

    absolute_url = serializers.SerializerMethodField()

    class Meta:
        model = Dashboard
        fields = [
            'id', 'name', 'description', 'slug',
            'client', 'facility', 'category',
            'streamlit_url', 'thumbnail',
            'is_active', 'is_public',
            'view_count', 'order',
            'created_by', 'created_at', 'updated_at',
            'absolute_url',
        ]
        read_only_fields = ('id', 'view_count', 'created_at', 'updated_at', 'absolute_url')

    def get_absolute_url(self, obj: Dashboard) -> str:
        request = self.context.get('request')
        url = obj.get_absolute_url()
        if request:
            return request.build_absolute_uri(url)
        return url


class DashboardListSerializer(serializers.ModelSerializer):
    """Lighter serializer for list endpoints."""

    absolute_url = serializers.SerializerMethodField()
    category_display = serializers.CharField(source='get_category_display', read_only=True)

    class Meta:
        model = Dashboard
        fields = [
            'id', 'name', 'slug', 'category', 'category_display',
            'is_active', 'view_count', 'updated_at', 'absolute_url',
        ]

    def get_absolute_url(self, obj: Dashboard) -> str:
        request = self.context.get('request')
        url = obj.get_absolute_url()
        if request:
            return request.build_absolute_uri(url)
        return url
