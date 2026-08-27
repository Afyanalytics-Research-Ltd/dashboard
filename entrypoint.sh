#!/bin/bash
export DJANGO_SETTINGS_MODULE=airflow_dashboard.settings
export PYTHONUNBUFFERED=1
# source /root/.local/share/virtualenvs/brooks-insurance-*/bin/activate

echo "<<<<<<<< Collect Staticfiles>>>>>>>>>"
python3 manage.py collectstatic --noinput


# sleep 5
# echo "<<<<<<<< Database airflow_dashboard and Migrations Starts >>>>>>>>>"
# # Run database migrations
python3 manage.py makemigrations &
python3 manage.py migrate  &

# sleep 5
# echo "<<<<<<< Initializing the Database >>>>>>>>>>"
# echo " "
# python manage.py loaddata initialization.yaml
# echo " "
echo "<<<<<<<<<<<<<<<<<<<< START Celery >>>>>>>>>>>>>>>>>>>>>>>>"

# Backs Agent Configuration's "Generate Missing Metrics" / "Rebuild
# Embeddings" background tasks (agents/tasks.py). No periodic tasks exist
# yet, so no celery beat is started.
celery -A airflow_dashboard worker --loglevel=info &

# # # start celery beat
# celery -A airflow_dashboard beat --loglevel=info &

# sleep 5

echo "<<<<<<<<<<<<<<<<<<<< START Daphne (ASGI: HTTP + WebSocket) >>>>>>>>>>>>>>>>>>>>>>>>"
# manage.py runserver is Django's dev server — single-threaded and not
# meant for production. daphne is the real ASGI server this project
# already depends on (see airflow_dashboard/asgi.py, which routes both
# plain HTTP and the /ws/ Channels websocket through one `application`).
# --proxy-headers trusts X-Forwarded-Proto/-For from nginx (see
# nginx/snippets/proxy-headers.conf) so request.is_secure() and the
# client IP are correct behind the reverse proxy.
# `exec` replaces this shell process with daphne so Docker's SIGTERM on
# `docker stop` reaches it directly for a clean shutdown, instead of
# being swallowed by bash.
#exec daphne -b 0.0.0.0 -p 8000 --proxy-headers airflow_dashboard.asgi:application
python3 manage.py runserver 0.0.0.0:8000