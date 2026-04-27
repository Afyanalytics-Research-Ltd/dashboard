
from django.urls import path
from . import views

urlpatterns = [
    path("", views.dashboard, name="datapipelines"),
    path("dag/<str:dag_id>/", views.dag_detail, name="dag_detail"),
    path("dag/<str:dag_id>/trigger/", views.trigger_dag, name="trigger_dag"),
    path("dag/<str:dag_id>/run/<str:run_id>/", views.dag_run_detail, name="dag_run_detail"),
]
