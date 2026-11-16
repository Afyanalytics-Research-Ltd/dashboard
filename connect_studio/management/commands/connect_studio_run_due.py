"""Start every scheduled Connect Studio sync that is due (for cron / Airflow when
the in-process scheduler is off), and wait for them to finish.

    python manage.py connect_studio_run_due
"""

import time

from django.core.management.base import BaseCommand

from connect_studio import services
from connect_studio.models import Job


class Command(BaseCommand):
    help = "Run scheduled Connect Studio syncs that are due."

    def handle(self, *args, **opts):
        ids = services.run_due()
        self.stdout.write(f"started {len(ids)} sync(s): {ids}")
        while ids and Job.objects.filter(pk__in=ids, status__in=("queued", "running")).exists():
            time.sleep(5)
        for j in Job.objects.filter(pk__in=ids):
            self.stdout.write(f"  job {j.pk} {j.title}: {j.status} {j.message}")
