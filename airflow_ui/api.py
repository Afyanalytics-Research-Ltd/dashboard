"""
Airflow UI DRF API views.
"""

import logging

from drf_spectacular.utils import extend_schema, extend_schema_view
from rest_framework import permissions, status, viewsets
from rest_framework.response import Response
from rest_framework.views import APIView
from django_filters.rest_framework import DjangoFilterBackend

from .models import DAGSummary
from .serializers import (
    DAGRunSerializer,
    DAGSummarySerializer,
    TaskInstanceSerializer,
)
from .services import AirflowService

logger = logging.getLogger(__name__)


class IsSuperuser(permissions.BasePermission):
    """Allow access only to superusers."""

    message = 'Superuser access is required for pipeline management.'

    def has_permission(self, request, view):
        return bool(request.user and request.user.is_superuser)


@extend_schema(
    description='Return the list of all DAGs fetched live from the Airflow API.',
    tags=['pipelines'],
    responses={200: {'description': 'List of DAG objects from Airflow'}},
)
class DAGListAPIView(APIView):
    """GET: live DAG list from Airflow."""

    permission_classes = [IsSuperuser]

    def get(self, request, *args, **kwargs):
        try:
            dags = AirflowService.get_dags()
        except Exception as exc:
            logger.error('DAGListAPIView error: %s', exc)
            return Response({'error': str(exc)}, status=status.HTTP_502_BAD_GATEWAY)
        return Response({'count': len(dags), 'dags': dags})


@extend_schema(
    description='Return dag runs for a specific DAG fetched live from Airflow.',
    tags=['pipelines'],
    responses={200: {'description': 'List of DAG run objects'}},
)
class DAGRunListAPIView(APIView):
    """GET: live run list for a DAG."""

    permission_classes = [IsSuperuser]

    def get(self, request, dag_id: str, *args, **kwargs):
        limit = int(request.query_params.get('limit', 25))
        try:
            runs = AirflowService.get_dag_runs(dag_id, limit=limit)
        except Exception as exc:
            logger.error('DAGRunListAPIView error for %s: %s', dag_id, exc)
            return Response({'error': str(exc)}, status=status.HTTP_502_BAD_GATEWAY)
        serializer = DAGRunSerializer(data=runs, many=True)
        serializer.is_valid()
        return Response({'count': len(runs), 'dag_runs': serializer.data})


@extend_schema(
    description='Trigger a new DAG run via the Airflow API (superuser only).',
    tags=['pipelines'],
    responses={200: {'description': 'Trigger result from Airflow'}},
)
class TriggerDAGAPIView(APIView):
    """POST: trigger a DAG run."""

    permission_classes = [IsSuperuser]

    def post(self, request, dag_id: str, *args, **kwargs):
        try:
            result = AirflowService.trigger_dag(dag_id)
        except Exception as exc:
            logger.error('TriggerDAGAPIView error for %s: %s', dag_id, exc)
            return Response({'error': str(exc)}, status=status.HTTP_502_BAD_GATEWAY)

        if 'error' in result:
            return Response({'ok': False, 'error': result['error']}, status=status.HTTP_502_BAD_GATEWAY)

        from core.models import AuditLog
        AuditLog.log(
            user=request.user,
            action='trigger',
            resource='dag',
            resource_id=dag_id,
            detail=f'API trigger: {result.get("dag_run_id", "")}',
        )
        logger.info('API trigger: %s by %s', dag_id, request.user.username)
        return Response({'ok': True, 'result': result})


@extend_schema_view(
    list=extend_schema(description='List cached DAG summary records.', tags=['pipelines']),
    retrieve=extend_schema(description='Get a single cached DAG summary.', tags=['pipelines']),
    create=extend_schema(description='Create a DAG summary cache record.', tags=['pipelines']),
    update=extend_schema(description='Replace a DAG summary cache record.', tags=['pipelines']),
    partial_update=extend_schema(description='Partially update a DAG summary.', tags=['pipelines']),
    destroy=extend_schema(description='Delete a DAG summary cache record.', tags=['pipelines']),
)
class DAGSummaryViewSet(viewsets.ModelViewSet):
    """
    CRUD for the local DAGSummary cache.
    Superuser-only.
    """

    queryset = DAGSummary.objects.all()
    serializer_class = DAGSummarySerializer
    permission_classes = [IsSuperuser]
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['is_active', 'is_paused', 'last_run_state']
