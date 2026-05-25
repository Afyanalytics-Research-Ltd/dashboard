"""
Airflow pipeline monitoring HTML URL patterns.
"""

from django.urls import path

from . import views

app_name = 'airflow'

urlpatterns = [
    path('', views.PipelineDashboardView.as_view(), name='dashboard'),
    path('dags/<str:dag_id>/', views.DAGDetailView.as_view(), name='dag_detail'),
    path('dags/<str:dag_id>/trigger/', views.TriggerDAGView.as_view(), name='trigger_dag'),
    path('dags/<str:dag_id>/runs/<str:run_id>/', views.DAGRunDetailView.as_view(), name='dag_run_detail'),
]
