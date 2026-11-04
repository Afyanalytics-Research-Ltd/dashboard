"""
Automation Studio models.

n8n's own workflow history is limited by licence (24 hours on community
editions), so Studio snapshots every workflow it creates, saves, restores or
deletes, plus periodic backups (``manage.py n8n_snapshot``). Snapshots are
de-duplicated by a hash of the workflow's meaningful content.
"""

import hashlib
import json

from django.conf import settings
from django.db import models

SNAPSHOT_FIELDS = ('name', 'nodes', 'connections', 'settings', 'pinData', 'description')


def content_hash(workflow: dict) -> str:
    body = {k: workflow.get(k) for k in SNAPSHOT_FIELDS}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


class WorkflowSnapshot(models.Model):
    SOURCES = [
        ('save', 'Saved in Studio'),
        ('create', 'Created in Studio'),
        ('restore', 'Restored in Studio'),
        ('deleted', 'Before delete'),
        ('backup', 'Scheduled backup'),
        ('manual', 'Manual backup'),
    ]

    workflow_id = models.CharField(max_length=64, db_index=True)
    version_id = models.CharField(max_length=64, blank=True)
    name = models.CharField(max_length=255, blank=True)
    source = models.CharField(max_length=16, choices=SOURCES, default='save')
    content_hash = models.CharField(max_length=64, db_index=True)
    node_count = models.PositiveIntegerField(default=0)
    active = models.BooleanField(default=False)
    data = models.JSONField(default=dict)
    note = models.CharField(max_length=255, blank=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name='+')
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ['-created_at', '-id']
        verbose_name = 'Workflow snapshot'

    def __str__(self):
        return f'{self.name or self.workflow_id} @ {self.created_at:%Y-%m-%d %H:%M}'

    @classmethod
    def capture(cls, workflow: dict, *, source: str = 'save', user=None, note: str = '', force: bool = False):
        """Store ``workflow`` unless its content equals the latest snapshot. Returns the snapshot or None."""
        wid = str(workflow.get('id') or '')
        if not wid or not isinstance(workflow.get('nodes'), list):
            return None
        digest = content_hash(workflow)
        if not force:
            last = cls.objects.filter(workflow_id=wid).only('content_hash').first()
            if last and last.content_hash == digest:
                return None
        return cls.objects.create(
            workflow_id=wid,
            version_id=str(workflow.get('versionId') or '')[:64],
            name=(workflow.get('name') or '')[:255],
            source=source,
            content_hash=digest,
            node_count=len(workflow.get('nodes') or []),
            active=bool(workflow.get('active')),
            data={k: workflow.get(k) for k in (*SNAPSHOT_FIELDS, 'tags', 'id', 'versionId')},
            note=note[:255],
            user=user if getattr(user, 'is_authenticated', False) else None,
        )

    def as_dict(self, with_data: bool = False) -> dict:
        out = {
            'id': self.id,
            'workflowId': self.workflow_id,
            'versionId': self.version_id,
            'name': self.name,
            'source': self.source,
            'hash': self.content_hash[:12],
            'nodeCount': self.node_count,
            'active': self.active,
            'note': self.note,
            'user': self.user.get_username() if self.user_id and self.user else None,
            'createdAt': self.created_at.isoformat(),
        }
        if with_data:
            out['data'] = self.data
        return out
