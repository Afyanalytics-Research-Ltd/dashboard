from django.apps import AppConfig


class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'
    verbose_name = 'Core Platform'

    def ready(self) -> None:
        # System Settings is the source of truth for the warehouse / LLM
        # switches in every Django process (web, Celery).
        from . import backends
        backends.register()
