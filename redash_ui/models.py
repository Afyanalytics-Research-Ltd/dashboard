"""
Analytics Studio models.

Redash keeps only the current text of a query; Studio records a revision
every time a query is created or saved through the proxy so users can
diff and restore earlier versions.
"""

from django.conf import settings
from django.db import models


class QueryRevision(models.Model):
    query_id = models.PositiveIntegerField(db_index=True)
    version = models.PositiveIntegerField(null=True, blank=True, help_text='Redash query version after the save')
    name = models.CharField(max_length=255, blank=True)
    query_text = models.TextField(blank=True)
    description = models.TextField(blank=True)
    data_source_id = models.PositiveIntegerField(null=True, blank=True)
    options = models.JSONField(default=dict, blank=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL, related_name='+')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at', '-id']
        verbose_name = 'Query revision'

    def __str__(self):
        return f'query {self.query_id} v{self.version}'

    def as_dict(self) -> dict:
        return {
            'id': self.id,
            'query_id': self.query_id,
            'version': self.version,
            'name': self.name,
            'query': self.query_text,
            'description': self.description,
            'data_source_id': self.data_source_id,
            'options': self.options,
            'user': self.user.get_username() if self.user else None,
            'created_at': self.created_at.isoformat(),
        }
