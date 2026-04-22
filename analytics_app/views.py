import os
from django.conf import settings
from django.shortcuts import render
from .models import Dashboard

EXCLUDED_FILES = {"__init__.py", "dynamic_file_loader.py"}

def dashboard_list(request):
    folder = os.path.join(settings.BASE_DIR, "analytics_app", "dashboards","pharmaplus")

    # 🔄 Sync filesystem → DB
    for file in os.listdir(folder):
        if file.endswith(".py") and file not in EXCLUDED_FILES:
            slug = file.replace(".py", "")
            name = slug.replace("_", " ").title()
            url = f"http://localhost:8501/?dashboard={slug}"

            Dashboard.objects.update_or_create(
                slug=slug,
                defaults={
                    "name": name,
                    "streamlit_url": url,
                    "is_active": True
                }
            )

    # 📊 Only render what's in DB
    dashboards = Dashboard.objects.filter(is_active=True)

    return render(request, "dashboards.html", {
        "dashboards": dashboards
    })