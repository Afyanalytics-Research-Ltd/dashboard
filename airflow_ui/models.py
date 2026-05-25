"""
Airflow UI models — local cache of DAG summary data.
"""

from django.db import models


class DAGSummary(models.Model):
    """
    Cached snapshot of a DAG's key metrics.
    Populated by syncing from the Airflow API rather than stored live.
    """

    dag_id = models.CharField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    is_paused = models.BooleanField(default=False)
    last_run_state = models.CharField(max_length=50, blank=True)
    last_run_at = models.DateTimeField(null=True, blank=True)
    total_runs = models.PositiveIntegerField(default=0)
    successful_runs = models.PositiveIntegerField(default=0)
    failed_runs = models.PositiveIntegerField(default=0)
    synced_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'DAG Summary'
        verbose_name_plural = 'DAG Summaries'
        ordering = ['dag_id']

    def __str__(self):
        return self.dag_id

    @property
    def success_rate(self) -> float:
        if self.total_runs == 0:
            return 0.0
        return round((self.successful_runs / self.total_runs) * 100, 1)
