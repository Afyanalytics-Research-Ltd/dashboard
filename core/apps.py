from django.apps import AppConfig


class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'
    verbose_name = 'Core Platform'

    def ready(self) -> None:
        # Import signal handlers when the app is ready
        pass  # noqa: F401
