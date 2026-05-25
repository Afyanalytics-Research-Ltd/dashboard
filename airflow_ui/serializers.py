"""
Airflow UI serializers — for both cached DAGSummary and live API data.
"""

from rest_framework import serializers

from .models import DAGSummary


class DAGSummarySerializer(serializers.ModelSerializer):
    success_rate = serializers.FloatField(read_only=True)

    class Meta:
        model = DAGSummary
        fields = [
            'id', 'dag_id', 'description',
            'is_active', 'is_paused',
            'last_run_state', 'last_run_at',
            'total_runs', 'successful_runs', 'failed_runs',
            'success_rate', 'synced_at',
        ]
        read_only_fields = ('id', 'success_rate', 'synced_at')


class DAGRunSerializer(serializers.Serializer):
    """
    Represents a DAG run as returned by the Airflow REST API.
    Read-only (used for rendering live data).
    """
    dag_run_id = serializers.CharField()
    dag_id = serializers.CharField()
    state = serializers.CharField()
    start_date = serializers.CharField(allow_null=True, default=None)
    end_date = serializers.CharField(allow_null=True, default=None)
    execution_date = serializers.CharField(allow_null=True, default=None)
    logical_date = serializers.CharField(allow_null=True, default=None)
    run_type = serializers.CharField(allow_blank=True, default='')
    note = serializers.CharField(allow_blank=True, default='')


class TaskInstanceSerializer(serializers.Serializer):
    """
    Represents a task instance as returned by the Airflow REST API.
    Read-only.
    """
    task_id = serializers.CharField()
    dag_id = serializers.CharField()
    dag_run_id = serializers.CharField()
    state = serializers.CharField(allow_null=True, default=None)
    start_date = serializers.CharField(allow_null=True, default=None)
    end_date = serializers.CharField(allow_null=True, default=None)
    duration = serializers.FloatField(allow_null=True, default=None)
    try_number = serializers.IntegerField(default=1)
    max_tries = serializers.IntegerField(default=0)
    operator = serializers.CharField(allow_blank=True, default='')
