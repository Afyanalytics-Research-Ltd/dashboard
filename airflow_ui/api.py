"""
Airflow UI DRF API views (``/api/v1/pipelines/``).

These are the token-authenticatable server endpoints. The browser UI uses
the full Airflow v2 surface through ``proxy.py`` instead.
"""

import logging

from django.utils import timezone
from django.utils.dateparse import parse_datetime
from django_filters.rest_framework import DjangoFilterBackend
from drf_spectacular.utils import extend_schema, extend_schema_view
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView

from core.models import AuditLog

from .models import DAGSummary
from .serializers import DAGRunSerializer, DAGSummarySerializer
from .services import AirflowError, AirflowService

logger = logging.getLogger(__name__)


class IsSuperuser(permissions.BasePermission):
    """Allow access only to superusers."""

    message = 'Superuser access is required for pipeline management.'

    def has_permission(self, request, view):
        return bool(request.user and request.user.is_superuser)


def _airflow_error(exc: AirflowError) -> Response:
    code = exc.status_code if 400 <= exc.status_code < 500 else status.HTTP_502_BAD_GATEWAY
    return Response({'error': str(exc), 'airflow': exc.payload}, status=code)


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
        except AirflowError as exc:
            logger.error('DAGListAPIView error: %s', exc)
            return _airflow_error(exc)
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
        try:
            limit = max(1, min(int(request.query_params.get('limit', 25)), 100))
        except ValueError:
            return Response({'error': 'limit must be an integer'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            runs = AirflowService.get_dag_runs(dag_id, limit=limit)
        except AirflowError as exc:
            logger.error('DAGRunListAPIView error for %s: %s', dag_id, exc)
            return _airflow_error(exc)
        serializer = DAGRunSerializer(data=runs, many=True)
        serializer.is_valid()
        return Response({'count': len(runs), 'dag_runs': serializer.data})


@extend_schema(
    description=(
        'Trigger a new DAG run via the Airflow API (superuser only). '
        'Optional JSON body: {"conf": {...}, "dag_run_id": "...", "note": "..."}.'
    ),
    tags=['pipelines'],
    responses={200: {'description': 'Trigger result from Airflow'}},
)
class TriggerDAGAPIView(APIView):
    """POST: trigger a DAG run."""

    permission_classes = [IsSuperuser]

    def post(self, request, dag_id: str, *args, **kwargs):
        conf = request.data.get('conf') or {}
        if not isinstance(conf, dict):
            return Response({'ok': False, 'error': 'conf must be a JSON object'},
                            status=status.HTTP_400_BAD_REQUEST)
        try:
            result = AirflowService.trigger_dag(
                dag_id, conf=conf,
                run_id=request.data.get('dag_run_id'),
                note=request.data.get('note'),
            )
        except AirflowError as exc:
            logger.error('TriggerDAGAPIView error for %s: %s', dag_id, exc)
            resp = _airflow_error(exc)
            resp.data['ok'] = False
            return resp

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
    sync=extend_schema(description='Refresh the cache from Airflow (dags + dagStats).',
                       tags=['pipelines'], request=None),
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

    @action(detail=False, methods=['post'])
    def sync(self, request):
        try:
            dags = AirflowService.get_dags()
            stats = {d['dag_id']: {s['state']: s['count'] for s in d.get('stats', [])}
                     for d in AirflowService.get_dag_stats()}
            last_runs = {r['dag_id']: r for r in AirflowService.get_dag_runs('~', limit=100)}
        except AirflowError as exc:
            return _airflow_error(exc)

        seen = []
        for dag in dags:
            dag_id = dag['dag_id']
            counts = stats.get(dag_id, {})
            last = last_runs.get(dag_id) or {}
            DAGSummary.objects.update_or_create(
                dag_id=dag_id,
                defaults={
                    'description': dag.get('description') or '',
                    'is_active': not dag.get('is_stale', False),
                    'is_paused': bool(dag.get('is_paused')),
                    'last_run_state': last.get('state') or '',
                    'last_run_at': parse_datetime(last['run_after']) if last.get('run_after') else None,
                    'total_runs': sum(counts.values()),
                    'successful_runs': counts.get('success', 0),
                    'failed_runs': counts.get('failed', 0),
                },
            )
            seen.append(dag_id)
        DAGSummary.objects.exclude(dag_id__in=seen).update(is_active=False)
        return Response({'synced': len(seen), 'at': timezone.now()})
