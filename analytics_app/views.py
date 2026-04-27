import os
from django.conf import settings
from django.shortcuts import render
from .models import Dashboard

EXCLUDED_FILES = {"__init__.py", "dynamic_file_loader.py"}

# views.py

def dashboard_view(request, slug):
    dashboard = Dashboard.objects.filter(slug=slug, is_active=True).latest('created_at')

    return render(request, "dashboard_iframe.html", {
        "dashboard": dashboard
    })


def dashboard_list(request):
    user = request.user
    client = user.profile.client or "default"

    folder = os.path.join(
        settings.BASE_DIR,
        "analytics_app",
        "dashboards",
        client.lower().replace(" ", "_")
    )

    # fallback if folder doesn't exist
    if not os.path.exists(folder):
        folder = os.path.join(
            settings.BASE_DIR,
            "analytics_app",
            "dashboards",
            "default"
        )


    # 🔄 Sync filesystem → DB
    for file in os.listdir(folder):
        if file.endswith(".py") and file not in EXCLUDED_FILES:
            slug = file.replace(".py", "")
            name = slug.replace("_", " ").title()
            url = f"http://localhost:8501/?dashboard={slug}"

            Dashboard.objects.update_or_create(
                client=client,
                slug=slug,
                defaults={
                    "name": name,
                    "streamlit_url": url,
                    "is_active": True
                }
            )

    # 📊 Only render what's in DB
    dashboards = Dashboard.objects.filter(is_active=True, client=client)

    return render(request, "dashboards.html", {
        "dashboards": dashboards
    })