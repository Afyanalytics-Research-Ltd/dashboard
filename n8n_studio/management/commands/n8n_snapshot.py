"""Back up every n8n workflow into Studio's snapshot history (unchanged ones are skipped).

Schedule it (cron or django_celery_beat) to keep history beyond n8n's licence window:
    python manage.py n8n_snapshot
"""

from django.core.management.base import BaseCommand, CommandError

from n8n_studio.client import N8nError
from n8n_studio.endpoints import capture_all


class Command(BaseCommand):
    help = 'Snapshot every n8n workflow into Automation Studio history.'

    def handle(self, *args, **options):
        try:
            result = capture_all(source='backup')
        except N8nError as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(self.style.SUCCESS(
            f"{result['workflows']} workflows: {result['captured']} captured, {result['unchanged']} unchanged"))
