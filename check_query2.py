import os, time
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "airflow_dashboard.settings")
import django
django.setup()
import requests
from django.conf import settings


session = requests.Session()
session.headers.update({"Authorization": "Key " + settings.REDASH_ADMIN_API_KEY})
base = settings.REDASH_BASE_URL

resp = session.post(f"{base}/api/queries/1/refresh")
job = resp.json().get("job", {})
job_id = job.get("id")
print("refresh started, job", job_id)

j = {}
for _ in range(15):
    r = session.get(f"{base}/api/jobs/{job_id}")
    j = r.json().get("job", {})
    print("status", j.get("status"), j.get("error"))
    if j.get("status") in (3, 4):
        break
    time.sleep(2)

qr_id = j.get("query_result_id")
if qr_id:
    r = session.get(f"{base}/api/query_results/{qr_id}")
    data = r.json().get("query_result", {}).get("data")
    print("SUCCESS rows:", data.get("rows") if data else None)
else:
    print("still failing, see error above")
