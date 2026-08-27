"""
Celery application instance.

Backs the two background tasks in agents/tasks.py (Agent Configuration's
"Generate Missing Metrics" / "Rebuild Embeddings" buttons — both make
several slow LLM/embedding API calls, too slow to run inline in a request).

Broker/result-backend URLs come from Django settings (CELERY_BROKER_URL /
CELERY_RESULT_BACKEND in airflow_dashboard/settings.py), which default to
the `redis` service already defined in docker-compose.yaml /
docker-compose.dev.yaml — no new infrastructure to stand up.

Start a worker with:
    celery -A airflow_dashboard worker --loglevel=info
"""

import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "airflow_dashboard.settings")

app = Celery("airflow_dashboard")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
