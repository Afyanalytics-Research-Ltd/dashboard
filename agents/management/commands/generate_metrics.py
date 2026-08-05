"""
CLI wrapper around agents/catalog_sync.py's generate_missing_metrics(), for
ops/CI use without going through the Semantic Layer Configuration settings page.

Usage:
    python manage.py generate_metrics
"""

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Draft MetricDefinition rows for any live Cube with no existing metric (additive only)."

    def handle(self, *args, **options):
        from agents.catalog_sync import generate_missing_metrics

        User = get_user_model()
        system_user = User.objects.filter(is_superuser=True).order_by("pk").first()

        result = generate_missing_metrics(system_user)
        self.stdout.write(
            self.style.SUCCESS(
                f"Created {len(result['created'])} new metric(s): {result['created']}"
            )
        )
        self.stdout.write(f"Skipped (already catalogued): {len(result['skipped'])}")
        if result["failed"]:
            self.stdout.write(self.style.WARNING(f"Failed: {result['failed']}"))
