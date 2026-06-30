from django.urls import path
from . import views

urlpatterns = [
    path("",                views.forecast,        name="forecast"),
    path("health/",         views.health,          name="forecast_health"),
    path("retrain/",        views.retrain,         name="forecast_retrain"),
    path("retrain/status/", views.retrain_status,  name="forecast_retrain_status"),
    path("drift/",          views.drift,           name="forecast_drift"),
]
