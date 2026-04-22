from django.contrib.auth.decorators import login_required


from django.shortcuts import render, redirect
from django.utils import timezone
import uuid
import requests

import os
from dotenv import load_dotenv
load_dotenv()

AIRFLOW_BASE_URL = "https://datapipelines.afyaanalytics.com/api/v2"
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME","default").strip()
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD","default").strip()

# Optional: store token globally
AIRFLOW_BEARER_TOKEN = None

def get_airflow_token():
    """Generate a JWT token from Airflow API"""
    global AIRFLOW_BEARER_TOKEN
    url = "https://datapipelines.afyaanalytics.com/auth/token"
    response = requests.post(
        url,
        json={"username": AIRFLOW_USERNAME, "password": AIRFLOW_PASSWORD},
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()
    AIRFLOW_BEARER_TOKEN = response.json().get("access_token")
    return AIRFLOW_BEARER_TOKEN

def airflow_request(method, endpoint, data=None):
    """Make an Airflow API request using Bearer token"""
    global AIRFLOW_BEARER_TOKEN

    if AIRFLOW_BEARER_TOKEN is None:
        get_airflow_token()  # generate token if not yet

    url = f"{AIRFLOW_BASE_URL}{endpoint}"
    headers = {"Authorization": f"Bearer {AIRFLOW_BEARER_TOKEN}"}

    response = requests.request(method, url, json=data, headers=headers)
    
    # Refresh token if expired (401)
    if response.status_code == 401:
        get_airflow_token()
        headers = {"Authorization": f"Bearer {AIRFLOW_BEARER_TOKEN}"}
        response = requests.request(method, url, json=data, headers=headers)

    print(response.status_code)
    try:
        return response.json()
    except:
        return {"error": response.text}
    
@login_required
def dashboard(request):
    dags = airflow_request("GET", "/dags").get("dags", [])
    return render(request, "dashboard.html", {"dags": dags})

@login_required
def dag_detail(request, dag_id):
    runs = airflow_request("GET", f"/dags/{dag_id}/dagRuns").get("dag_runs", [])
    success = sum(1 for r in runs if r["state"] == "success")
    failed = sum(1 for r in runs if r["state"] == "failed")
    return render(request, "dag_detail.html", {
        "dag_id": dag_id,
        "runs": runs,
        "success": success,
        "failed": failed
    })

@login_required
def trigger_dag(request, dag_id):
    if request.method == "POST":
        now = timezone.now()

        data = {
            "dag_run_id": f"manual__{str(uuid.uuid4())}",  # unique run ID
            "logical_date": now.isoformat(),          # required by Airflow
            "conf": {
                "facility": "afya_api_auth"
            },
            "note": f"Triggered by {request.user.username}"
        }
       
        airflow_request("POST", f"/dags/{dag_id}/dagRuns", data=data)
        
    return redirect("dag_detail", dag_id=dag_id)

@login_required
def dag_run_detail(request, dag_id, run_id):
    tasks = airflow_request(
        "GET",
        f"/dags/{dag_id}/dagRuns/{run_id}/taskInstances"
    ).get("task_instances", [])

    return render(request, "dag_run.html", {
        "run_id": run_id,
        "tasks": tasks
    })
