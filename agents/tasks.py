"""
Celery tasks backing the Semantic Layer Configuration settings page's
"Sync Cube Schema", "Generate Missing Metrics", and "Rebuild Embeddings"
buttons (agents/views.py) — in that pipeline order: sync brings
model/cubes/*.yml up to date with Snowflake's REPORTING schema, generate
drafts catalog entries for any cube with none yet, rebuild refreshes the
retrieval index so the new/updated entries are actually findable.

All three wrap the exact same agents/catalog_sync.py functions the buttons
used to call synchronously — the functions themselves are unchanged; only
how they're invoked changed (queued via Celery instead of blocking the
request). The management commands (agents/management/commands/) still call
catalog_sync directly for CLI/ops use, not through Celery.

A Django User instance isn't JSON-serializable (Celery's default task
argument serializer), so generate_missing_metrics_task takes a user_id and
re-fetches it inside the task, rather than passing the model instance.
"""

from __future__ import annotations

import logging

from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task(bind=True)
def generate_missing_metrics_task(self, user_id: int | None):
    from django.contrib.auth import get_user_model

    from . import catalog_sync

    user = None
    if user_id is not None:
        user = get_user_model().objects.filter(pk=user_id).first()

    result = catalog_sync.generate_missing_metrics(user)
    logger.info(
        "generate_missing_metrics_task[%s]: created=%d skipped=%d failed=%d",
        self.request.id, len(result["created"]), len(result["skipped"]), len(result["failed"]),
    )
    return result


@shared_task(bind=True)
def rebuild_embeddings_task(self):
    from . import catalog_sync

    counts = catalog_sync.rebuild_embeddings()
    logger.info("rebuild_embeddings_task[%s]: %s", self.request.id, counts)
    return counts


@shared_task(bind=True)
def sync_cube_schemas_task(self):
    from . import catalog_sync

    summary = catalog_sync.sync_cube_schemas_from_snowflake(dry_run=False)
    logger.info(
        "sync_cube_schemas_task[%s]: cubes_updated=%d errors=%d",
        self.request.id, len(summary["cubes_updated"]), len(summary["errors"]),
    )
    return summary
