"""
Model Studio models.

W&B keeps workspace layouts inside its own frontend; Studio stores named,
shareable views itself (one of the most common W&B complaints is losing a
carefully arranged workspace).
"""

from django.conf import settings
from django.db import models


class WandbCredential(models.Model):
    """A user's own W&B API key (encrypted) and which W&B it belongs to — see connection.py."""

    SERVERS = [('local', "This platform's W&B server"), ('cloud', 'W&B Cloud (wandb.ai)')]

    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='wandb_credential')
    server = models.CharField(max_length=20, choices=SERVERS, default='local')
    api_key_enc = models.TextField(help_text='Fernet-encrypted API key; never sent to the browser')
    key_hint = models.CharField(max_length=80, blank=True, default='', help_text="Non-secret: the key's public id or last 4")
    wandb_user = models.CharField(max_length=128, blank=True, default='')
    wandb_entity = models.CharField(max_length=128, blank=True, default='')
    verified_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f'{self.user} → {self.get_server_display()} ({self.wandb_user or "?"})'

    def as_dict(self) -> dict:
        return {'server': self.server, 'server_label': self.get_server_display(), 'key_hint': self.key_hint,
                'wandb_user': self.wandb_user, 'wandb_entity': self.wandb_entity,
                'verified_at': self.verified_at.isoformat() if self.verified_at else None,
                'updated_at': self.updated_at.isoformat() if self.updated_at else None}


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
