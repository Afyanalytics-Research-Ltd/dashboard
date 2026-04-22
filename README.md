cd datahub
python -m venv env
source env/bin/activate   # Windows: env\Scripts\activate

pip install -r requirements.txt

python manage.py migrate
python manage.py createsuperuser  # create an admin account
python manage.py runserver