"""
Model Studio models.

W&B keeps workspace layouts inside its own frontend; Studio stores named,
shareable views itself (one of the most common W&B complaints is losing a
carefully arranged workspace).
"""

from django.conf import settings
from django.db import models


class SavedView(models.Model):
    entity = models.CharField(max_length=128)
    project = models.CharField(max_length=128)
    name = models.CharField(max_length=120)
    state = models.JSONField(default=dict, help_text='Filters, grouping, columns, panels, smoothing, x-axis…')
    shared = models.BooleanField(default=True, help_text='Visible to every Model Studio user')
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL, related_name='+')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['name']
        indexes = [models.Index(fields=['entity', 'project'])]

    def __str__(self):
        return f'{self.entity}/{self.project}: {self.name}'

    def as_dict(self) -> dict:
        return {
            'id': self.id,
            'entity': self.entity,
            'project': self.project,
            'name': self.name,
            'state': self.state,
            'shared': self.shared,
            'owner': self.owner.get_username() if self.owner else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
        }
