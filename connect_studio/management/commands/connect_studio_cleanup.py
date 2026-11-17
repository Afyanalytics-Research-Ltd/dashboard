"""Clear Connect Studio's leftovers on this server's disk and abandoned uploads (what the scheduler does every 15 min).

    python manage.py connect_studio_cleanup                  # workspaces idle > 6 h, uploads abandoned > 24 h
    python manage.py connect_studio_cleanup --move-to-s3     # also move older local uploads to S3 (bucket must be set)
"""

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from connect_studio import services, storage
from connect_studio.models import SourceFile


class Command(BaseCommand):
    help = "Delete stale Connect Studio workspaces / abandoned uploads; optionally move local uploads to S3."

    def add_arguments(self, parser):
        parser.add_argument("--move-to-s3", action="store_true",
                            help="Upload files still stored locally to S3, then delete the local copy.")

    def handle(self, *args, move_to_s3=False, **opts):
        res = services.janitor()
        self.stdout.write(f"workspaces removed: {res['workspaces']} ({res['bytes'] / 1e6:,.1f} MB) · "
                          f"abandoned uploads aborted: {res['uploads']}")
        if not move_to_s3:
            return
        if not storage.enabled():
            raise CommandError("CONNECT_STUDIO_S3_BUCKET is not set.")
        moved = 0
        for f in SourceFile.objects.filter(storage="local"):
            if not f.file or not Path(f.file.path).exists():
                continue
            if services.source_busy(f.source):
                self.stdout.write(f"skipped {f.original_name}: source {f.source_id} has a running job")
                continue
            key = storage.upload_key(f.source_id, f"moved-{f.pk}", f.original_name)
            storage.put_file(Path(f.file.path), key)
            f.file.delete(save=False)
            f.storage, f.key = "s3", key
            f.save(update_fields=["storage", "key", "file"])
            moved += 1
        self.stdout.write(f"moved to S3: {moved} file(s)")
