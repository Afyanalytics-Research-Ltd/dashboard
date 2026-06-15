from django.apps import AppConfig


class SelfServiceConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'self_service'
    verbose_name = 'Self-Service Analytics'
