"""
One-time data migration: import catalog/metrics.yaml into MetricDefinition
so the DB becomes the source of truth (see agents/catalog.py).

Guarded two ways:
  - no-op if MetricDefinition already has rows (safe to re-run / re-deploy)
  - no-op (not a crash) if catalog/metrics.yaml is missing or unreadable —
    a data migration that hard-crashes on a missing file would block every
    later migration on every environment where that file isn't present
    (e.g. once agents/catalog.py no longer needs it and someone moves it).
"""

from pathlib import Path

import yaml
from django.db import migrations

METRICS_YAML_PATH = Path(__file__).resolve().parent.parent.parent / "catalog" / "metrics.yaml"


def import_metrics(apps, schema_editor):
    MetricDefinition = apps.get_model("agents", "MetricDefinition")

    if MetricDefinition.objects.exists():
        return

    try:
        with open(METRICS_YAML_PATH, "r") as f:
            data = yaml.safe_load(f) or {}
    except (FileNotFoundError, OSError, yaml.YAMLError):
        return

    metrics = data.get("metrics") or []
    MetricDefinition.objects.bulk_create([
        MetricDefinition(
            metric_id=m["id"],
            name=m.get("name", m["id"]),
            description=m.get("description", ""),
            cube_query=m.get("cube_query") or {},
            is_active=True,
        )
        for m in metrics
        if m.get("id")
    ])


def noop_reverse(apps, schema_editor):
    pass  # importing is a one-way seed; reversing would delete hand-added rows too


class Migration(migrations.Migration):

    dependencies = [
        ("agents", "0003_metricdefinition_pendingcubemeasure"),
    ]

    operations = [
        migrations.RunPython(import_metrics, noop_reverse),
    ]
