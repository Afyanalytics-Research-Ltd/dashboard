"""Warehouse Studio persistence (Django DB, not Snowflake).

* :class:`Worksheet` — saved SQL with its context, folders and stars.
* :class:`QueryRun` — every statement a user ran from the Studio (audit + "My runs").
* :class:`FindingTriage` — acknowledge / resolve / ignore toolkit findings with a
  note.  Snowsight has no triage workflow; findings stay in Snowflake and the
  team's decisions about them live here.
* :class:`DetectorRun` — detector/report invocations started from the Optimizer.
"""

import uuid

from django.conf import settings
from django.db import models


class Worksheet(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='ws_worksheets')
    title = models.CharField(max_length=200, default='Untitled worksheet')
    sql = models.TextField(blank=True, default='')
    context = models.JSONField(default=dict, blank=True)
    folder = models.CharField(max_length=120, blank=True, default='')
    starred = models.BooleanField(default=False)
    shared = models.BooleanField(default=False, help_text='Visible (read-only) to other Studio users.')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-starred', '-updated_at']

    def __str__(self):
        return self.title

    def as_dict(self, include_sql: bool = True) -> dict:
        data = {'id': str(self.id), 'title': self.title, 'context': self.context, 'folder': self.folder,
                'starred': self.starred, 'shared': self.shared, 'owner': self.owner.get_username(),
                'created_at': self.created_at.isoformat(), 'updated_at': self.updated_at.isoformat()}
        if include_sql:
            data['sql'] = self.sql
        else:
            data['preview'] = ' '.join(self.sql.split())[:160]
        return data


class QueryRun(models.Model):
    STATUS_CHOICES = [(s, s) for s in ('SUBMITTED', 'RUNNING', 'SUCCESS', 'FAILED', 'CANCELLED', 'BLOCKED')]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='ws_query_runs')
    query_id = models.CharField(max_length=64, blank=True, default='', db_index=True)
    worksheet = models.ForeignKey(Worksheet, null=True, blank=True, on_delete=models.SET_NULL, related_name='runs')
    sql = models.TextField()
    context = models.JSONField(default=dict, blank=True)
    kind = models.CharField(max_length=20, default='worksheet')  # worksheet | optimizer | procedure | explain
    status = models.CharField(max_length=12, choices=STATUS_CHOICES, default='SUBMITTED')
    rows = models.BigIntegerField(null=True, blank=True)
    elapsed_ms = models.BigIntegerField(null=True, blank=True)
    error = models.TextField(blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def as_dict(self) -> dict:
        return {'id': self.id, 'query_id': self.query_id, 'sql': self.sql, 'context': self.context, 'kind': self.kind,
                'status': self.status, 'rows': self.rows, 'elapsed_ms': self.elapsed_ms, 'error': self.error,
                'worksheet': str(self.worksheet_id) if self.worksheet_id else None,
                'created_at': self.created_at.isoformat(),
                'finished_at': self.finished_at.isoformat() if self.finished_at else None}


class FindingTriage(models.Model):
    STATUSES = ('open', 'acknowledged', 'in_progress', 'resolved', 'ignored')

    finding_id = models.BigIntegerField(unique=True)
    status = models.CharField(max_length=20, choices=[(s, s) for s in STATUSES], default='open')
    note = models.TextField(blank=True, default='')
    assignee = models.CharField(max_length=150, blank=True, default='')
    updated_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL, related_name='+')
    updated_at = models.DateTimeField(auto_now=True)

    def as_dict(self) -> dict:
        return {'finding_id': self.finding_id, 'status': self.status, 'note': self.note, 'assignee': self.assignee,
                'updated_by': self.updated_by.get_username() if self.updated_by_id else None,
                'updated_at': self.updated_at.isoformat()}


class DetectorRun(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='ws_detector_runs')
    key = models.CharField(max_length=60)
    params = models.JSONField(default=dict, blank=True)
    query_id = models.CharField(max_length=64, db_index=True)
    status = models.CharField(max_length=12, default='RUNNING')
    outcome = models.JSONField(null=True, blank=True)
    error = models.TextField(blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def as_dict(self) -> dict:
        return {'id': self.id, 'key': self.key, 'params': self.params, 'query_id': self.query_id, 'status': self.status,
                'outcome': self.outcome, 'error': self.error, 'user': self.user.get_username(),
                'created_at': self.created_at.isoformat(),
                'finished_at': self.finished_at.isoformat() if self.finished_at else None}
