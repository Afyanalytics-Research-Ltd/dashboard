"""Connect Studio models.

Source      a configured connector (database, files, API, schema file) — any role
SourceFile  uploads that belong to a source (CSV, SQLite, schema.json …)
Snapshot    one extraction of a source's schema (dbatlas metadata + profiles + relationships)
Mapping     source snapshot → target snapshot, the reviewable mapping document
Connection  a sync: mapping + destination + mode + schedule + cursor state
Job         background work (test, extract, map, remap, sync) with progress and logs
"""

from __future__ import annotations

import uuid
from pathlib import Path

from django.conf import settings as django_settings
from django.db import models
from django.utils import timezone


def _upload_to(instance, filename):
    return f"connect_studio/sources/{instance.source_id}/{uuid.uuid4().hex[:8]}_{Path(filename).name}"


class Source(models.Model):
    STATUS = [("draft", "Draft"), ("ok", "Connected"), ("error", "Error"), ("untested", "Untested")]

    name = models.CharField(max_length=200)
    kind = models.CharField(max_length=50)
    description = models.TextField(blank=True, default="")
    config = models.JSONField(default=dict, blank=True)          # non-secret settings
    secrets = models.TextField(blank=True, default="")            # Fernet-encrypted JSON of secret fields
    status = models.CharField(max_length=20, choices=STATUS, default="draft")
    last_test = models.JSONField(default=dict, blank=True)
    last_tested_at = models.DateTimeField(null=True, blank=True)
    last_error = models.TextField(blank=True, default="")
    created_by = models.ForeignKey(django_settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.name} ({self.kind})"

    @property
    def latest_snapshot(self):
        return self.snapshots.order_by("-created_at").first()

    def file_paths(self) -> list[Path]:
        return [Path(f.file.path) for f in self.files.all() if f.file]

    def workdir(self) -> Path:
        path = Path(django_settings.MEDIA_ROOT) / "connect_studio" / "work" / str(self.pk)
        path.mkdir(parents=True, exist_ok=True)
        return path


class SourceFile(models.Model):
    source = models.ForeignKey(Source, on_delete=models.CASCADE, related_name="files")
    file = models.FileField(upload_to=_upload_to, max_length=500)
    original_name = models.CharField(max_length=300)
    size = models.BigIntegerField(default=0)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["uploaded_at"]


class Snapshot(models.Model):
    source = models.ForeignKey(Source, on_delete=models.CASCADE, related_name="snapshots")
    data = models.JSONField()
    fingerprint = models.CharField(max_length=32, blank=True, default="")
    stats = models.JSONField(default=dict, blank=True)
    diff = models.JSONField(default=dict, blank=True)
    options = models.JSONField(default=dict, blank=True)
    duration_ms = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]


class Mapping(models.Model):
    STATUS = [("draft", "Draft"), ("generating", "Generating"), ("review", "In review"), ("approved", "Approved"), ("failed", "Failed")]

    name = models.CharField(max_length=200)
    source = models.ForeignKey(Source, on_delete=models.CASCADE, related_name="mappings_as_source")
    target = models.ForeignKey(Source, on_delete=models.CASCADE, related_name="mappings_as_target")
    source_snapshot = models.ForeignKey(Snapshot, null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    target_snapshot = models.ForeignKey(Snapshot, null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    document = models.JSONField(default=dict, blank=True)
    summary = models.JSONField(default=dict, blank=True)
    settings = models.JSONField(default=dict, blank=True)  # ai, samples, tables …
    status = models.CharField(max_length=20, choices=STATUS, default="draft")
    version = models.IntegerField(default=0)
    created_by = models.ForeignKey(django_settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]


class Connection(models.Model):
    MODES = [("full_refresh_overwrite", "Full refresh · overwrite"), ("full_refresh_append", "Full refresh · append"),
             ("incremental_append", "Incremental · append"), ("incremental_dedupe", "Incremental · deduped")]

    name = models.CharField(max_length=200)
    mapping = models.ForeignKey(Mapping, on_delete=models.CASCADE, related_name="connections")
    destination = models.ForeignKey(Source, on_delete=models.CASCADE, related_name="connections_as_destination")
    mode = models.CharField(max_length=40, choices=MODES, default="full_refresh_overwrite")
    schedule_minutes = models.IntegerField(default=0)  # 0 = manual
    enabled = models.BooleanField(default=True)
    next_run_at = models.DateTimeField(null=True, blank=True)
    settings = models.JSONField(default=dict, blank=True)  # tables, cursors, row_limit
    state = models.JSONField(default=dict, blank=True)     # per-table cursors
    last_status = models.CharField(max_length=20, blank=True, default="")
    last_run_at = models.DateTimeField(null=True, blank=True)
    last_rows = models.BigIntegerField(default=0)
    created_by = models.ForeignKey(django_settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def schedule_next(self, base=None):
        if self.schedule_minutes and self.enabled:
            self.next_run_at = (base or timezone.now()) + timezone.timedelta(minutes=self.schedule_minutes)
        else:
            self.next_run_at = None


class Job(models.Model):
    KINDS = [("test", "Connection test"), ("extract", "Schema extraction"), ("map", "AI mapping"),
             ("remap", "Re-map table"), ("sync", "Sync")]
    STATUS = [("queued", "Queued"), ("running", "Running"), ("succeeded", "Succeeded"), ("failed", "Failed"),
              ("cancelled", "Cancelled")]

    kind = models.CharField(max_length=20, choices=KINDS)
    status = models.CharField(max_length=20, choices=STATUS, default="queued")
    title = models.CharField(max_length=300, blank=True, default="")
    progress = models.FloatField(default=0)
    phase = models.CharField(max_length=40, blank=True, default="")
    message = models.CharField(max_length=500, blank=True, default="")
    log = models.JSONField(default=list, blank=True)
    result = models.JSONField(default=dict, blank=True)
    error = models.TextField(blank=True, default="")
    source = models.ForeignKey(Source, null=True, blank=True, on_delete=models.CASCADE, related_name="jobs")
    mapping = models.ForeignKey(Mapping, null=True, blank=True, on_delete=models.CASCADE, related_name="jobs")
    connection = models.ForeignKey(Connection, null=True, blank=True, on_delete=models.CASCADE, related_name="jobs")
    trigger = models.CharField(max_length=20, blank=True, default="manual")
    cancel_requested = models.BooleanField(default=False)
    created_by = models.ForeignKey(django_settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    heartbeat_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["status", "created_at"])]

    @property
    def duration(self) -> float | None:
        if not self.started_at:
            return None
        return ((self.finished_at or timezone.now()) - self.started_at).total_seconds()

    @property
    def stalled(self) -> bool:
        return self.status == "running" and bool(self.heartbeat_at) and (timezone.now() - self.heartbeat_at).total_seconds() > 180
