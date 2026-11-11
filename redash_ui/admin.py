from django.contrib import admin

from .models import QueryRevision


@admin.register(QueryRevision)
class QueryRevisionAdmin(admin.ModelAdmin):
    list_display = ('query_id', 'version', 'name', 'user', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('name', 'query_text')
    readonly_fields = [f.name for f in QueryRevision._meta.fields]
