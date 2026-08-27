"""
CLI wrapper around agents/catalog_sync.py's sync_cube_schemas_from_snowflake(),
for ops/CI use without going through the Semantic Layer Configuration
settings page.

Usage:
    python manage.py sync_cube_schemas              # writes for real
    python manage.py sync_cube_schemas --dry-run     # preview only, writes nothing
"""

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = (
        "Introspect Snowflake's REPORTING schema and write any measure/dimension an "
        "existing cube doesn't expose yet directly into model/cubes/*.yml. No review "
        "step by default — use --dry-run to preview first."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run", action="store_true",
            help="Compute and print what would be added per cube; write nothing.",
        )

    def handle(self, *args, **options):
        from agents.catalog_sync import sync_cube_schemas_from_snowflake

        summary = sync_cube_schemas_from_snowflake(dry_run=options["dry_run"])

        label = "Would update" if options["dry_run"] else "Updated"
        for cube in summary["cubes_updated"]:
            fields = ", ".join(summary["fields_added"][cube])
            self.stdout.write(self.style.SUCCESS(f"{label} {cube}: {fields}"))

        if summary["skipped_unclassified"]:
            self.stdout.write(self.style.WARNING("Unclassified columns (review manually):"))
            for cube, cols in summary["skipped_unclassified"].items():
                self.stdout.write(f"  {cube}: {', '.join(cols)}")

        if summary["errors"]:
            self.stdout.write(self.style.ERROR("Cubes that could not be synced:"))
            for cube, err in summary["errors"].items():
                self.stdout.write(f"  {cube}: {err}")

        self.stdout.write(
            self.style.SUCCESS(
                f"\n{label} {len(summary['cubes_updated'])} cube(s), "
                f"{len(summary['errors'])} error(s)."
            )
        )
