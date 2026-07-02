from django.core.management.base import BaseCommand

from analytics_app.dashboards.ksh.facility_utilization.chat.snapshot_writer import write_snapshot


class Command(BaseCommand):
    help = "Refresh metrics_snapshot.json from Snowflake gold tables."

    def handle(self, *args, **options):
        self.stdout.write("Refreshing metrics snapshot...")
        snapshot = write_snapshot()
        errors   = snapshot.get("KISUMU_CLEAN", {}).get("fetch_errors", [])
        n_total  = len(snapshot.get("KISUMU_CLEAN", {}).get("metrics", {}))

        if errors:
            self.stderr.write(
                self.style.WARNING(
                    f"Snapshot written: {n_total - len(errors)}/{n_total} OK. "
                    f"Failed: {errors}"
                )
            )
        else:
            self.stdout.write(
                self.style.SUCCESS(f"Snapshot written: {n_total} metrics, 0 errors.")
            )
