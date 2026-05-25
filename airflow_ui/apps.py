from django.apps import AppConfig


class AirflowUiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'airflow_ui'
    verbose_name = 'Airflow Pipeline Monitor'
