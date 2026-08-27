"""
CLI wrapper around agents/catalog_sync.py's rebuild_embeddings(), for ops/CI
use without going through the Semantic Layer Configuration settings page.

Usage:
    python manage.py rebuild_embeddings
"""

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Rebuild catalog/embeddings.npz from the current MetricDefinition rows + glossary.yaml."

    def handle(self, *args, **options):
        from agents.catalog_sync import rebuild_embeddings

        counts = rebuild_embeddings()
        self.stdout.write(self.style.SUCCESS(f"Rebuilt embeddings: {counts}"))
