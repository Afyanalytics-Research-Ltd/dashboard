"""
Semantic Layer Configuration settings page — HTML views.

Same mixin stack and access gate as core/views.py's SystemSettingsView
(superuser-only), since this page can write to the live Cube schema and to
the DB-backed metric catalog every agent question runs against.

  AgentConfigurationView    — GET: list MetricDefinitions, live Cube fields
                              (via agents/catalog_sync.py), and pending
                              measure proposals.
  MetricDefinitionSaveView  — POST: create or update one MetricDefinition
                              (matched by metric_id).
  ProposeCubeMeasureView    — POST: stage a new PendingCubeMeasure.
  ApproveCubeMeasureView    — POST: validate against live Snowflake, splice
                              into model/cubes/<cube>.yml, mark written.
  RejectCubeMeasureView     — POST: mark a pending proposal rejected.
  SyncCubeSchemasView       — POST: queues a Celery task that introspects
                              Snowflake's REPORTING schema and writes any
                              measure/dimension an existing cube doesn't
                              expose yet directly into model/cubes/*.yml —
                              no approval step (agents/tasks.py).
  GenerateMetricsView       — POST: queues a Celery task that additively
                              LLM-drafts MetricDefinitions for any live
                              cube with none yet (agents/tasks.py).
  RebuildEmbeddingsView     — POST: queues a Celery task that rebuilds
                              catalog/embeddings.npz from the current DB +
                              glossary.yaml (agents/tasks.py).

All three used to run synchronously in the request (no Celery/RQ existed in
this repo) — now queued via Celery instead, since each makes several slow
LLM/embedding/Snowflake calls. Requires a running `celery -A
airflow_dashboard worker` process; see airflow_dashboard/celery.py. Run in
that order (sync → generate → rebuild) for a full catch-up: sync brings the
Cube schema up to date with the warehouse, generate catalogs whatever's now
exposed, rebuild makes it findable.
"""

from __future__ import annotations

import json
import logging

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from django.db import transaction
from django.db.models import Q
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views import View
from django.views.generic import TemplateView

from core.mixins import BreadcrumbMixin, LoggingMixin, SuperuserRequiredMixin

from . import catalog_sync
from .models import MetricDefinition, PendingCubeMeasure
from .tasks import generate_missing_metrics_task, rebuild_embeddings_task, sync_cube_schemas_task

logger = logging.getLogger(__name__)

_REDIRECT = "agents:agent-configuration"
_PAGE_SIZE = 20


def _paginate(request: HttpRequest, items, param_name: str, page_size: int = _PAGE_SIZE):
    """
    Same page/PageNotAnInteger/EmptyPage handling as core.mixins.PaginationMixin,
    but with a caller-chosen GET param name instead of a hardcoded "page" —
    this view has four independent lists on one page (metrics, live cube
    fields, pending proposals, reviewed proposals), so each needs its own
    param to paginate without clobbering the others. Works on both
    QuerySets and plain lists (list_live_cube_fields() returns a plain list
    — it's a live Cube /meta call, not a DB query).
    """
    paginator = Paginator(items, page_size)
    page_number = request.GET.get(param_name, 1)
    try:
        return paginator.page(page_number)
    except PageNotAnInteger:
        return paginator.page(1)
    except EmptyPage:
        return paginator.page(paginator.num_pages)


@method_decorator(login_required, name="dispatch")
class AgentConfigurationView(SuperuserRequiredMixin, BreadcrumbMixin, LoggingMixin, TemplateView):
    """Manage the AI agent's metric catalog and propose new Cube measures."""

    template_name = "agents/configuration.html"

    def get_breadcrumbs(self):
        return [
            {"label": "Home", "url": "/analytics/"},
            {"label": "Semantic Layer Configuration", "url": None},
        ]

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["sidebar_section"] = "agent-configuration"
        context["page_title"] = "Semantic Layer Configuration"
        request = self.request

        # --- Metric Definitions ---
        metrics_q = request.GET.get("metrics_q", "").strip()
        metrics_qs = MetricDefinition.objects.all()
        if metrics_q:
            metrics_qs = metrics_qs.filter(
                Q(metric_id__icontains=metrics_q)
                | Q(name__icontains=metrics_q)
                | Q(description__icontains=metrics_q)
            )
        metrics_page = _paginate(request, metrics_qs, "metrics_page")
        for m in metrics_page:
            # Precomputed here (not a template filter) so the edit modal gets
            # valid JSON back — {{ m.cube_query|safe }} would render Python's
            # dict repr (single-quoted), which json.loads() on save rejects
            # the moment someone edits a metric without touching this field.
            m.cube_query_json = json.dumps(m.cube_query)
        context["metrics_q"] = metrics_q
        context["metrics_page"] = metrics_page

        # --- Pending measure proposals ---
        pending_q = request.GET.get("pending_q", "").strip()
        pending_qs = PendingCubeMeasure.objects.filter(status=PendingCubeMeasure.STATUS_PENDING)
        if pending_q:
            pending_qs = pending_qs.filter(
                Q(cube_name__icontains=pending_q) | Q(measure_name__icontains=pending_q)
            )
        context["pending_q"] = pending_q
        context["pending_page"] = _paginate(request, pending_qs, "pending_page")

        # --- Reviewed proposals ---
        reviewed_q = request.GET.get("reviewed_q", "").strip()
        reviewed_qs = PendingCubeMeasure.objects.exclude(status=PendingCubeMeasure.STATUS_PENDING)
        if reviewed_q:
            reviewed_qs = reviewed_qs.filter(
                Q(cube_name__icontains=reviewed_q) | Q(measure_name__icontains=reviewed_q)
            )
        context["reviewed_q"] = reviewed_q
        context["reviewed_page"] = _paginate(request, reviewed_qs, "reviewed_page")

        # --- Live Cube schema (not a QuerySet — a live /meta call) ---
        cubes_q = request.GET.get("cubes_q", "").strip()
        try:
            live_cube_fields = catalog_sync.list_live_cube_fields()
            context["cube_fetch_error"] = None
        except Exception as exc:
            logger.exception("AgentConfigurationView: could not fetch live Cube meta")
            live_cube_fields = []
            context["cube_fetch_error"] = str(exc)

        if cubes_q:
            needle = cubes_q.lower()
            live_cube_fields = [
                c for c in live_cube_fields
                if needle in c["name"].lower()
                or any(needle in f.lower() for f in c["measures"])
                or any(needle in f.lower() for f in c["dimensions"])
            ]
        context["cubes_q"] = cubes_q
        context["cubes_page"] = _paginate(request, live_cube_fields, "cubes_page")

        return context


@method_decorator(login_required, name="dispatch")
class MetricDefinitionSaveView(SuperuserRequiredMixin, LoggingMixin, View):
    """Create or update one MetricDefinition, matched by metric_id."""

    def post(self, request: HttpRequest) -> HttpResponse:
        metric_id = request.POST.get("metric_id", "").strip()
        name = request.POST.get("name", "").strip()
        description = request.POST.get("description", "").strip()
        cube_query_raw = request.POST.get("cube_query", "").strip()
        is_active = request.POST.get("is_active") == "on"

        if not metric_id or not name:
            messages.error(request, "metric_id and name are required.")
            return redirect(_REDIRECT)

        try:
            cube_query = json.loads(cube_query_raw) if cube_query_raw else {}
        except json.JSONDecodeError as exc:
            messages.error(request, f"cube_query must be valid JSON: {exc}")
            return redirect(_REDIRECT)

        existed = MetricDefinition.objects.filter(metric_id=metric_id).exists()
        defaults = {
            "name": name,
            "description": description,
            "cube_query": cube_query,
            "is_active": is_active,
            "updated_by": request.user,
        }
        if not existed:
            defaults["created_by"] = request.user

        metric, created = MetricDefinition.objects.update_or_create(
            metric_id=metric_id, defaults=defaults,
        )
        self.audit_log(
            "update" if not created else "create",
            "MetricDefinition",
            resource_id=metric_id,
            detail=f"{'Created' if created else 'Updated'} metric {metric_id}",
        )
        messages.success(request, f"{'Created' if created else 'Updated'} metric '{metric.name}'.")
        return redirect(_REDIRECT)


@method_decorator(login_required, name="dispatch")
class ProposeCubeMeasureView(SuperuserRequiredMixin, LoggingMixin, View):
    """Stage a new measure, OR an edit to an existing one, on a cube for
    review — action="add" (default) or "edit" picks which."""

    def post(self, request: HttpRequest) -> HttpResponse:
        cube_name = request.POST.get("cube_name", "").strip()
        measure_name = request.POST.get("measure_name", "").strip()
        measure_type = request.POST.get("measure_type", "").strip()
        action = request.POST.get("action", PendingCubeMeasure.ACTION_ADD).strip()

        if not cube_name or not measure_name or not measure_type:
            messages.error(request, "cube_name, measure_name, and measure_type are required.")
            return redirect(_REDIRECT)

        if action == PendingCubeMeasure.ACTION_EDIT:
            if catalog_sync.get_cube_measure_definition(cube_name, measure_name) is None:
                messages.error(
                    request,
                    f"'{measure_name}' doesn't exist on {cube_name} yet — nothing to edit. "
                    f"Use 'Propose Measure' to add it instead.",
                )
                return redirect(_REDIRECT)

        pending = PendingCubeMeasure.objects.create(
            action=action,
            cube_name=cube_name,
            measure_name=measure_name,
            measure_type=measure_type,
            sql_expression=request.POST.get("sql_expression", "").strip(),
            title=request.POST.get("title", "").strip(),
            description=request.POST.get("description", "").strip(),
            requested_by=request.user,
        )
        self.audit_log(
            "create", "PendingCubeMeasure", resource_id=str(pending.pk),
            detail=f"Proposed {action} of {cube_name}.{measure_name}",
        )
        verb = "edit to" if action == PendingCubeMeasure.ACTION_EDIT else "new measure"
        messages.success(request, f"Proposed {verb} {cube_name}.{measure_name} — awaiting approval.")
        return redirect(_REDIRECT)


@method_decorator(login_required, name="dispatch")
class ApproveCubeMeasureView(SuperuserRequiredMixin, LoggingMixin, View):
    """
    Validate a pending measure against live Snowflake, splice it into the
    cube's YAML, mark it written.

    select_for_update() inside the transaction closes the concurrent-
    approval race (two admins approving the same row at once) on any
    backend that supports row locking. Note: SQLite (the dev DB here) has
    no row-locking support — Django silently runs a plain SELECT there
    instead of erroring — so this guard is a no-op in local dev and a real
    lock once deployed against Postgres; the in-transaction status check
    ("already reviewed?") still prevents a double-write either way.
    """

    def post(self, request: HttpRequest, pk: int) -> HttpResponse:
        with transaction.atomic():
            pending = get_object_or_404(
                PendingCubeMeasure.objects.select_for_update(), pk=pk
            )
            if pending.status != PendingCubeMeasure.STATUS_PENDING:
                messages.warning(request, f"That proposal was already {pending.status}.")
                return redirect(_REDIRECT)

            ok, msg = catalog_sync.validate_column_exists(pending.cube_name, pending.sql_expression)
            if not ok:
                messages.error(request, f"Column validation failed: {msg}")
                return redirect(_REDIRECT)

            write_fn = (
                catalog_sync.write_pending_measure_edit_to_yaml
                if pending.action == PendingCubeMeasure.ACTION_EDIT
                else catalog_sync.write_pending_measure_to_yaml
            )
            written, write_msg = write_fn(pending)
            if not written:
                messages.error(request, f"Could not write measure: {write_msg}")
                return redirect(_REDIRECT)

            pending.status = PendingCubeMeasure.STATUS_WRITTEN
            pending.reviewed_by = request.user
            pending.reviewed_at = timezone.now()
            pending.save(update_fields=["status", "reviewed_by", "reviewed_at"])

        self.audit_log(
            "update", "PendingCubeMeasure", resource_id=str(pk),
            detail=f"Approved {pending.cube_name}.{pending.measure_name}",
        )
        messages.success(
            request,
            f"{write_msg}. Cube runs in dev mode and will hot-reload it — "
            f"consider clicking 'Rebuild Embeddings' so retrieval can find it.",
        )
        return redirect(_REDIRECT)


@method_decorator(login_required, name="dispatch")
class RejectCubeMeasureView(SuperuserRequiredMixin, LoggingMixin, View):
    def post(self, request: HttpRequest, pk: int) -> HttpResponse:
        with transaction.atomic():
            pending = get_object_or_404(
                PendingCubeMeasure.objects.select_for_update(), pk=pk
            )
            if pending.status != PendingCubeMeasure.STATUS_PENDING:
                messages.warning(request, f"That proposal was already {pending.status}.")
                return redirect(_REDIRECT)

            pending.status = PendingCubeMeasure.STATUS_REJECTED
            pending.reviewed_by = request.user
            pending.reviewed_at = timezone.now()
            pending.rejection_reason = request.POST.get("rejection_reason", "").strip()
            pending.save(update_fields=["status", "reviewed_by", "reviewed_at", "rejection_reason"])

        self.audit_log(
            "update", "PendingCubeMeasure", resource_id=str(pk),
            detail=f"Rejected {pending.cube_name}.{pending.measure_name}",
        )
        messages.success(request, f"Rejected {pending.cube_name}.{pending.measure_name}.")
        return redirect(_REDIRECT)


@method_decorator(login_required, name="dispatch")
class SyncCubeSchemasView(SuperuserRequiredMixin, LoggingMixin, View):
    """
    Introspects Snowflake's REPORTING schema and writes any measure/
    dimension an existing cube doesn't expose yet straight into
    model/cubes/*.yml — no PendingCubeMeasure staging, no approval click
    (explicit product decision — see agents/catalog_sync.py's module
    docstring for why). Queues agents.tasks.sync_cube_schemas_task; run
    this before "Generate Missing Metrics" so newly-exposed columns get a
    chance to be catalogued in the same pass.
    """

    def post(self, request: HttpRequest) -> HttpResponse:
        try:
            task = sync_cube_schemas_task.delay()
        except Exception as exc:
            logger.exception("SyncCubeSchemasView: could not queue sync_cube_schemas_task")
            messages.error(request, f"Could not queue cube schema sync: {exc}")
            return redirect(_REDIRECT)

        self.audit_log("update", "CubeSchema", detail=f"Queued schema sync task {task.id}")
        messages.success(
            request,
            f"Queued (task {task.id}) — checking every cube against Snowflake's REPORTING "
            f"schema and writing anything missing directly to model/cubes/*.yml (no review "
            f"step). Check the Celery worker log for what got added per cube. Run 'Generate "
            f"Missing Metrics' next so new columns can be catalogued.",
        )
        return redirect(_REDIRECT)


@method_decorator(login_required, name="dispatch")
class GenerateMetricsView(SuperuserRequiredMixin, LoggingMixin, View):
    """
    Additive-only: drafts a MetricDefinition for every live cube with none
    yet. Queues agents.tasks.generate_missing_metrics_task instead of
    running inline — one LLM call per missing cube is too slow for a
    request/response cycle.
    """

    def post(self, request: HttpRequest) -> HttpResponse:
        try:
            task = generate_missing_metrics_task.delay(request.user.pk)
        except Exception as exc:
            # Most likely cause: no Celery worker / broker reachable — a
            # connection failure dispatching the task, not a failure of the
            # task's own logic (which hasn't run yet at this point).
            logger.exception("GenerateMetricsView: could not queue generate_missing_metrics_task")
            messages.error(request, f"Could not queue metric generation: {exc}")
            return redirect(_REDIRECT)

        self.audit_log("create", "MetricDefinition", detail=f"Queued generation task {task.id}")
        messages.success(
            request,
            f"Queued (task {task.id}) — this runs in the background against every live cube "
            f"with no metric yet; check the Celery worker log for the created/skipped/failed "
            f"counts. Rebuild Embeddings afterward so retrieval can find anything new.",
        )
        return redirect(_REDIRECT)


@method_decorator(login_required, name="dispatch")
class RebuildEmbeddingsView(SuperuserRequiredMixin, LoggingMixin, View):
    """Queues agents.tasks.rebuild_embeddings_task to rebuild
    catalog/embeddings.npz from the current DB + glossary.yaml."""

    def post(self, request: HttpRequest) -> HttpResponse:
        try:
            task = rebuild_embeddings_task.delay()
        except Exception as exc:
            logger.exception("RebuildEmbeddingsView: could not queue rebuild_embeddings_task")
            messages.error(request, f"Could not queue embeddings rebuild: {exc}")
            return redirect(_REDIRECT)

        self.audit_log("update", "EmbeddingsIndex", detail=f"Queued rebuild task {task.id}")
        messages.success(
            request,
            f"Queued (task {task.id}) — rebuilding the retrieval index in the background; "
            f"check the Celery worker log for the metric/measure/dimension/glossary counts.",
        )
        return redirect(_REDIRECT)
