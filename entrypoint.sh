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
# echo "<<<<<<<<<<<<<<<<<<<< START Celery >>>>>>>>>>>>>>>>>>>>>>>>"

# # # start Celery worker
# celery -A airflow_dashboard worker --loglevel=info &

# # # start celery beat
# celery -A airflow_dashboard beat --loglevel=info &

# sleep 5

# Start Gunicorn WSGI
#gunicorn --bind 0.0.0.0:8000 airflow_dashboard.wsgi --workers=2 &

# Start Daphne ASGI for WebSockets
#daphne -b 0.0.0.0 -p 8001 airflow_dashboard.asgi:application 

python3 manage.py runserver 0.0.0.0:8000