"""
Airflow pipeline monitoring views.

All views are restricted to superusers via SuperuserRequiredMixin.
"""

import logging
import uuid

from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from django.http import HttpRequest
from django.shortcuts import redirect
from django.urls import reverse
from django.utils import timezone
from django.views.generic import ListView, TemplateView, View

from core.mixins import BreadcrumbMixin, LoggingMixin, SuperuserRequiredMixin
from core.models import AuditLog

from .models import DAGSummary
from .services import AirflowService

logger = logging.getLogger(__name__)

RUNS_PER_PAGE = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_badge(state: str) -> str:
    """Return a CSS class suffix for the given state string."""
    mapping = {
        'success': 'success',
        'failed': 'failed',
        'running': 'running',
        'queued': 'queued',
        'up_for_retry': 'queued',
        'skipped': 'paused',
    }
    return mapping.get(state, 'paused')


def _annotate_dags(dags: list[dict]) -> list[dict]:
    """Add a ``badge_class`` key to each DAG dict."""
    for dag in dags:
        state = dag.get('last_run_state') or ('paused' if dag.get('is_paused') else 'unknown')
        dag['badge_class'] = _state_badge(state)
    return dags


def _paginate(request: HttpRequest, queryset, per_page: int):
    paginator = Paginator(queryset, per_page)
    page_number = request.GET.get('page', 1)
    try:
        page_obj = paginator.page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)
    return paginator, page_obj


# ---------------------------------------------------------------------------
# Pipeline Dashboard (DAG list)
# ---------------------------------------------------------------------------

class PipelineDashboardView(
    SuperuserRequiredMixin, BreadcrumbMixin, TemplateView
):
    """Monitor page showing all Airflow DAGs."""

    template_name = 'airflow/pipelines.html'

    def get_breadcrumbs(self):
        return [
            {'label': 'Home', 'url': reverse('analytics:home')},
            {'label': 'Pipelines', 'url': None},
        ]

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)

        try:
            dags = AirflowService.get_dags()
        except Exception as exc:
            logger.error('Failed to fetch DAGs from Airflow: %s', exc)
            dags = []
            messages.warning(self.request, 'Could not reach the Airflow API. Showing cached data.')

        # Apply search
        q = self.request.GET.get('q', '').strip()
        if q:
            dags = [d for d in dags if q.lower() in d.get('dag_id', '').lower()]

        # Apply state filter
        state_filter = self.request.GET.get('state', '').strip()
        if state_filter == 'active':
            dags = [d for d in dags if not d.get('is_paused', False)]
        elif state_filter == 'paused':
            dags = [d for d in dags if d.get('is_paused', False)]

        # Annotate badge classes
        _annotate_dags(dags)

        # Stats
        total = len(dags)
        active = sum(1 for d in dags if not d.get('is_paused', False))

        # Pagination
        paginator, page_obj = _paginate(self.request, dags, per_page=10)

        ctx.update({
            'sidebar_section': 'pipelines',
            'page_obj': page_obj,
            'paginator': paginator,
            'is_paginated': paginator.num_pages > 1,
            'total_dags': total,
            'active_dags': active,
            'paused_dags': total - active,
            'current_q': q,
            'current_state': state_filter,
        })
        return ctx


# ---------------------------------------------------------------------------
# DAG Detail (run history)
# ---------------------------------------------------------------------------

class DAGDetailView(
    SuperuserRequiredMixin, BreadcrumbMixin, LoggingMixin, TemplateView
):
    """Run history for a single DAG."""

    template_name = 'airflow/dag_detail.html'

    def get_breadcrumbs(self):
        dag_id = self.kwargs.get('dag_id', '')
        return [
            {'label': 'Home', 'url': reverse('analytics:home')},
            {'label': 'Pipelines', 'url': reverse('airflow:dashboard')},
            {'label': dag_id, 'url': None},
        ]

    def get(self, request, *args, **kwargs):
        dag_id = self.kwargs['dag_id']
        try:
            AuditLog.log(
                user=request.user,
                action='read',
                resource='dag',
                resource_id=dag_id,
                detail=f'Viewed DAG detail: {dag_id}',
                ip_address=request.META.get('REMOTE_ADDR'),
                user_agent=request.META.get('HTTP_USER_AGENT', ''),
            )
        except Exception as exc:
            logger.warning('Audit log failed for DAG detail: %s', exc)
        return super().get(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        dag_id = self.kwargs['dag_id']

        try:
            runs = AirflowService.get_dag_runs(dag_id, limit=50)
        except Exception as exc:
            logger.error('Failed to fetch runs for %s: %s', dag_id, exc)
            runs = []
            messages.warning(self.request, f'Could not load runs for {dag_id}.')

        # State filter
        state_filter = self.request.GET.get('state', '').strip()
        if state_filter:
            runs = [r for r in runs if r.get('state') == state_filter]

        # Stats
        total = len(runs)
        successful = sum(1 for r in runs if r.get('state') == 'success')
        failed = sum(1 for r in runs if r.get('state') == 'failed')
        success_rate = round((successful / total * 100), 1) if total else 0

        # Pagination
        paginator, page_obj = _paginate(self.request, runs, per_page=RUNS_PER_PAGE)

        ctx.update({
            'sidebar_section': 'pipelines',
            'dag_id': dag_id,
            'page_obj': page_obj,
            'paginator': paginator,
            'is_paginated': paginator.num_pages > 1,
            'total_runs': total,
            'successful_runs': successful,
            'failed_runs': failed,
            'success_rate': success_rate,
            'current_state': state_filter,
        })
        return ctx


# ---------------------------------------------------------------------------
# Trigger DAG
# ---------------------------------------------------------------------------

class TriggerDAGView(SuperuserRequiredMixin, View):
    """POST-only view to trigger a DAG run."""

    def post(self, request: HttpRequest, dag_id: str):
        try:
            result = AirflowService.trigger_dag(dag_id)
            run_id = result.get('dag_run_id', 'unknown')
            if 'error' in result:
                messages.error(request, f'Failed to trigger {dag_id}: {result["error"]}')
                logger.error('Trigger failed for %s: %s', dag_id, result['error'])
            else:
                messages.success(request, f'DAG {dag_id} triggered — run ID: {run_id}')
                logger.info('Superuser %s triggered DAG %s (run_id=%s)',
                            request.user.username, dag_id, run_id)
                # Audit log
                AuditLog.log(
                    user=request.user,
                    action='trigger',
                    resource='dag',
                    resource_id=dag_id,
                    detail=f'Triggered DAG run: {run_id}',
                    ip_address=request.META.get('REMOTE_ADDR'),
                    user_agent=request.META.get('HTTP_USER_AGENT', ''),
                )
        except Exception as exc:
            messages.error(request, f'Error triggering {dag_id}: {exc}')
            logger.exception('Exception when triggering DAG %s', dag_id)

        return redirect('airflow:dag_detail', dag_id=dag_id)


# ---------------------------------------------------------------------------
# DAG Run Detail (task instances)
# ---------------------------------------------------------------------------

class DAGRunDetailView(
    SuperuserRequiredMixin, BreadcrumbMixin, LoggingMixin, TemplateView
):
    """Task instance table for a specific DAG run."""

    template_name = 'airflow/dag_run.html'

    def get_breadcrumbs(self):
        dag_id = self.kwargs.get('dag_id', '')
        run_id = self.kwargs.get('run_id', '')
        return [
            {'label': 'Home', 'url': reverse('analytics:home')},
            {'label': 'Pipelines', 'url': reverse('airflow:dashboard')},
            {'label': dag_id, 'url': reverse('airflow:dag_detail', kwargs={'dag_id': dag_id})},
            {'label': f'Run {run_id[:20]}…' if len(run_id) > 20 else f'Run {run_id}', 'url': None},
        ]

    def get(self, request, *args, **kwargs):
        dag_id = self.kwargs['dag_id']
        run_id = self.kwargs['run_id']
        try:
            AuditLog.log(
                user=request.user,
                action='read',
                resource='dag_run',
                resource_id=f'{dag_id}/{run_id}',
                detail=f'Viewed run detail: {run_id}',
                ip_address=request.META.get('REMOTE_ADDR'),
                user_agent=request.META.get('HTTP_USER_AGENT', ''),
            )
        except Exception as exc:
            logger.warning('Audit log failed for DAG run detail: %s', exc)
        return super().get(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        dag_id = self.kwargs['dag_id']
        run_id = self.kwargs['run_id']

        try:
            tasks = AirflowService.get_task_instances(dag_id, run_id)
        except Exception as exc:
            logger.error('Failed to fetch tasks for %s/%s: %s', dag_id, run_id, exc)
            tasks = []
            messages.warning(self.request, 'Could not load task instances.')

        # Stats
        total = len(tasks)
        successful = sum(1 for t in tasks if t.get('state') == 'success')
        failed = sum(1 for t in tasks if t.get('state') == 'failed')
        running = sum(1 for t in tasks if t.get('state') == 'running')

        ctx.update({
            'sidebar_section': 'pipelines',
            'dag_id': dag_id,
            'run_id': run_id,
            'tasks': tasks,
            'total_tasks': total,
            'successful_tasks': successful,
            'failed_tasks': failed,
            'running_tasks': running,
        })
        return ctx
