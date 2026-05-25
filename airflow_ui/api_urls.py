"""
Airflow UI API URL patterns.
"""

from django.urls import path
from rest_framework.routers import DefaultRouter

from .api import (
    DAGListAPIView,
    DAGRunListAPIView,
    DAGSummaryViewSet,
    TriggerDAGAPIView,
)

router = DefaultRouter()
router.register(r'summaries', DAGSummaryViewSet, basename='dag-summary')

urlpatterns = router.urls + [
    path('dags/', DAGListAPIView.as_view(), name='api-dag-list'),
    path('dags/<str:dag_id>/runs/', DAGRunListAPIView.as_view(), name='api-dag-runs'),
    path('dags/<str:dag_id>/trigger/', TriggerDAGAPIView.as_view(), name='api-trigger-dag'),
]
