from django.apps import AppConfig


class WandbUiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'wandb_ui'
    verbose_name = 'Model Studio (Weights & Biases)'
