from django.contrib import admin

from .models import SavedView


@admin.register(SavedView)
class SavedViewAdmin(admin.ModelAdmin):
    list_display = ('name', 'entity', 'project', 'owner', 'shared', 'updated_at')
    list_filter = ('shared', 'entity')
    search_fields = ('name', 'project')
