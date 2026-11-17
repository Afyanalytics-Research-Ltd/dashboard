from django.apps import AppConfig


class ConnectStudioConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "connect_studio"
    verbose_name = "Connect Studio"

    def ready(self):
        # in-process scheduler for scheduled syncs (only under a web server;
        # CONNECT_STUDIO_SCHEDULER=0 turns it off — then use `manage.py connect_studio_run_due`)
        from . import services

        services.start_scheduler()
