"""
Analytics app views.
"""

import logging
import os

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from django.db.models import Q
from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse, reverse_lazy
from django.views.generic import (
    CreateView, DeleteView, DetailView,
    ListView, TemplateView, UpdateView, View,
)

from core.mixins import BreadcrumbMixin, LoggingMixin, StaffRequiredMixin, SuperuserRequiredMixin
from core.models import AuditLog, Facility
from warehouse.models import TrackedSpreadsheet
from warehouse.services.snowflake import SnowflakeClient, SnowflakeQueryError

from .forms import DashboardForm, DashboardSearchForm, ReportingQueryForm
from .models import Dashboard, ReportingQuery
from .services.redash import (
    RedashAPIError, create_dashboard, create_query, create_visualization, create_widget,
    get_query, get_query_columns, list_data_sources, publish_dashboard, publish_query, share_dashboard,
)

logger = logging.getLogger(__name__)

EXCLUDED_FILES = {'__init__.py', 'dynamic_file_loader.py'}


def _sync_dashboards_for_client(client_slug: str, client_obj) -> dict:
    """
    Scan analytics_app/dashboards/{client_slug}/ for .py files and
    create/update Dashboard records.  Returns counts.
    """
    folder = os.path.join(
        settings.BASE_DIR, 'analytics_app', 'dashboards', client_slug
    )
    logging.warning(f"{folder} --------->")
    if not os.path.isdir(folder):
        folder = os.path.join(
            settings.BASE_DIR, 'analytics_app', 'dashboards', 'default'
        )

    created = updated = deactivated = 0
    current_slugs: set[str] = set()


    if os.path.isdir(folder):
        for filename in os.listdir(folder):
            if not filename.endswith('.py') or filename in EXCLUDED_FILES:
                continue
            slug = filename[:-3]  # strip .py
            current_slugs.add(slug)
            name = slug.replace('_', ' ').title()
            url = f"{settings.STREAMLIT_BASE_URL}/?dashboard={slug}"

            _, was_created = Dashboard.objects.update_or_create(
                slug=slug,
                defaults={
                    'name': name,
                    'client': client_obj,
                    'streamlit_url': url,
                    'is_active': True,
                },
            )
            if was_created:
                created += 1
            else:
                updated += 1

    deactivated = (
        Dashboard.objects
        .filter(client=client_obj)
        .exclude(slug__in=current_slugs)
        .filter(streamlit_url__regex=r'^https?://(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\d+)?') #handle external urls 
        .update(is_active=False)
    )
    return {'created': created, 'updated': updated, 'deactivated': deactivated}


def _get_client_obj(user):
    """Return the Client FK for the user if profile has one, else None."""
    try:
        profile = user.profile
        logging.warning(f"<-----{profile}-------->")
        logging.warning(f"<-----{profile.client}-------->")
        client_name = profile.client.name if profile.client else profile.client
        if not client_name:
            return None
        from core.models import Client
        return Client.objects.filter(name__iexact=client_name).first()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

class HomeView(LoginRequiredMixin, TemplateView):
    template_name = 'home.html'

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        user = self.request.user
        client_obj = _get_client_obj(user)
        logging.warning(f"<------------------{client_obj}--------------------------------->")

        qs = Dashboard.objects.filter(is_active=True)
        if client_obj:
            qs = qs.filter(client=client_obj)
        elif not user.is_superuser:
            qs = qs.none()

        ctx.update({
            'sidebar_section': 'home',
            'total_dashboards': qs.count(),
            'total_spreadsheets': TrackedSpreadsheet.objects.filter().count(),
            'recent_dashboards': qs.order_by('-updated_at')[:5],
        })
        return ctx


# ---------------------------------------------------------------------------
# Dashboard list
# ---------------------------------------------------------------------------

class DashboardListView(LoginRequiredMixin, BreadcrumbMixin, ListView):
    model = Dashboard
    template_name = 'analytics/list.html'
    context_object_name = 'dashboards'
    paginate_by = 12

    def get_breadcrumbs(self):
        return [
            {'label': 'Home', 'url': reverse('analytics:home')},
            {'label': 'Analytics', 'url': None},
            {'label': 'Dashboards', 'url': None},
        ]

    def get_queryset(self):
        user = self.request.user
        client_obj = _get_client_obj(user)

        qs = Dashboard.objects.filter(is_active=True)
        if client_obj:
            qs = qs.filter(client=client_obj)
        elif not user.is_superuser:
            qs = qs.none()

        q = self.request.GET.get('q', '').strip()
        if q:
            qs = qs.filter(Q(name__icontains=q) | Q(description__icontains=q))

        category = self.request.GET.get('category', '').strip()
        if category:
            qs = qs.filter(category=category)

        return qs.select_related('client', 'facility')

    def get(self, request, *args, **kwargs):
        # Sync filesystem on list load
        user = request.user
        client_obj = _get_client_obj(user)
        if client_obj:
            try:
                client_slug = client_obj.slug
                _sync_dashboards_for_client(client_slug, client_obj)
            except Exception as exc:
                logger.warning('Dashboard sync failed: %s', exc)
        return super().get(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx['sidebar_section'] = 'analytics'
        ctx['search_form'] = DashboardSearchForm(self.request.GET or None)
        ctx['current_q'] = self.request.GET.get('q', '')
        ctx['current_category'] = self.request.GET.get('category', '')
        ctx['categories'] = Dashboard.CATEGORY_CHOICES
        ctx['total_count'] = self.get_queryset().count()
        return ctx


# ---------------------------------------------------------------------------
# Dashboard detail (viewer)
# ---------------------------------------------------------------------------

class DashboardDetailView(LoginRequiredMixin, BreadcrumbMixin, LoggingMixin, DetailView):
    model = Dashboard
    template_name = 'analytics/viewer.html'
    context_object_name = 'dashboard'

    def get_object(self, queryset=None):
        obj = get_object_or_404(Dashboard, slug=self.kwargs['slug'], is_active=True)
        # Client scoping for non-superusers
        if not self.request.user.is_superuser:
            client_obj = _get_client_obj(self.request.user)
            if obj.client and client_obj and obj.client != client_obj:
                from django.core.exceptions import PermissionDenied
                raise PermissionDenied('You do not have access to this dashboard.')
        return obj

    def get_breadcrumbs(self):
        return [
            {'label': 'Home', 'url': reverse('analytics:home')},
            {'label': 'Dashboards', 'url': reverse('analytics:dashboard_list')},
            {'label': self.object.name, 'url': None},
        ]

    def get(self, request, *args, **kwargs):
        response = super().get(request, *args, **kwargs)
        obj = self.object
        obj.increment_view_count()
        try:
            self.audit_log(
                action='read',
                resource='dashboard',
                resource_id=obj.slug,
                detail=f'Viewed dashboard: {obj.name}',
            )
        except Exception as exc:
            logger.warning('Audit log failed: %s', exc)
        return response

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx['sidebar_section'] = 'analytics'
        ctx['embed_url'] = self.object.get_embed_url(self.request.user)
        return ctx


# ---------------------------------------------------------------------------
# Dashboard sync (superuser)
# ---------------------------------------------------------------------------

class DashboardSyncView(SuperuserRequiredMixin, View):
    def post(self, request, *args, **kwargs):
        from core.models import Client
        results = []
        for client in Client.objects.filter(is_active=True):
            res = _sync_dashboards_for_client(client.slug, client)
            results.append(f"{client.name}: +{res['created']} created, ~{res['updated']} updated, -{res['deactivated']} deactivated")
        summary = '; '.join(results) if results else 'No active clients found.'
        messages.success(request, f'Sync complete — {summary}')
        logger.info('Superuser %s triggered dashboard sync: %s', request.user.username, summary)
        return redirect('analytics:dashboard_list')


# ---------------------------------------------------------------------------
# Dashboard CRUD (superuser)
# ---------------------------------------------------------------------------

class DashboardCreateView(SuperuserRequiredMixin, BreadcrumbMixin, CreateView):
    model = Dashboard
    form_class = DashboardForm
    template_name = 'analytics/dashboard_form.html'
    success_url = reverse_lazy('analytics:dashboard_list')

    def get_breadcrumbs(self):
        return [
            {'label': 'Home', 'url': reverse('analytics:home')},
            {'label': 'Dashboards', 'url': reverse('analytics:dashboard_list')},
            {'label': 'Create', 'url': None},
        ]

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx['sidebar_section'] = 'analytics'
        ctx['form_title'] = 'Create Dashboard'
        return ctx

    def form_valid(self, form):
        form.instance.created_by = self.request.user
        messages.success(self.request, f'Dashboard "{form.instance.name}" created.')
        logger.info('Superuser %s created dashboard: %s', self.request.user.username, form.instance.name)
        return super().form_valid(form)


class DashboardUpdateView(SuperuserRequiredMixin, BreadcrumbMixin, UpdateView):
    model = Dashboard
    form_class = DashboardForm
    template_name = 'analytics/dashboard_form.html'

    def get_success_url(self):
        return reverse('analytics:dashboard_view', kwargs={'slug': self.object.slug})

    def get_breadcrumbs(self):
        return [
            {'label': 'Home', 'url': reverse('analytics:home')},
            {'label': 'Dashboards', 'url': reverse('analytics:dashboard_list')},
            {'label': self.object.name, 'url': reverse('analytics:dashboard_view', kwargs={'slug': self.object.slug})},
            {'label': 'Edit', 'url': None},
        ]

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx['sidebar_section'] = 'analytics'
        ctx['form_title'] = f'Edit — {self.object.name}'
        return ctx

    def form_valid(self, form):
        messages.success(self.request, f'Dashboard "{form.instance.name}" updated.')
        logger.info('Superuser %s updated dashboard: %s', self.request.user.username, form.instance.name)
        return super().form_valid(form)


class DashboardDeleteView(SuperuserRequiredMixin, DeleteView):
    model = Dashboard
    template_name = 'analytics/dashboard_confirm_delete.html'
    success_url = reverse_lazy('analytics:dashboard_list')

    def form_valid(self, form):
        name = self.object.name
        messages.success(self.request, f'Dashboard "{name}" deleted.')
        logger.info('Superuser %s deleted dashboard: %s', self.request.user.username, name)
        return super().form_valid(form)


# ---------------------------------------------------------------------------
# Reporting queries (superuser) — semantic layer -> Redash sync
# ---------------------------------------------------------------------------

class ReportingQueryListView(SuperuserRequiredMixin, BreadcrumbMixin, ListView):
    model = ReportingQuery
    template_name = 'analytics/reporting_query_list.html'
    context_object_name = 'reporting_queries'
    paginate_by = 12

    def get_breadcrumbs(self):
        return [
            {'label': 'Home', 'url': reverse('analytics:home')},
            {'label': 'Dashboards', 'url': reverse('analytics:dashboard_list')},
            {'label': 'Reporting Queries', 'url': None},
        ]

    def get_queryset(self):
        qs = ReportingQuery.objects.select_related('facility').all()
        q = self.request.GET.get('q', '').strip()
        if q:
            qs = qs.filter(
                Q(name__icontains=q)
                | Q(source_table__icontains=q)
                | Q(redash_data_source_name__icontains=q)
                | Q(facility__name__icontains=q)
            )
        return qs

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx['current_q'] = self.request.GET.get('q', '')
        ctx['total_count'] = ReportingQuery.objects.count()
        return ctx


class ReportingQueryCreateView(SuperuserRequiredMixin, BreadcrumbMixin, View):
    """Superuser form: submit a custom SQL query, create it in Redash, and record it."""

    template_name = 'analytics/reporting_query_form.html'

    def get_breadcrumbs(self):
        return [
            {'label': 'Home', 'url': reverse('analytics:home')},
            {'label': 'Dashboards', 'url': reverse('analytics:dashboard_list')},
            {'label': 'Reporting Queries', 'url': reverse('analytics:reporting_query_list')},
            {'label': 'Add Query', 'url': None},
        ]

    def _data_source_choices(self):
        try:
            return [(ds['id'], f"{ds['name']} ({ds['type']})") for ds in list_data_sources()]
        except RedashAPIError as exc:
            messages.error(self.request, f'Could not load Redash data sources: {exc}')
            return []

    def get(self, request, *args, **kwargs):
        form = ReportingQueryForm(data_source_choices=self._data_source_choices())
        return render(request, self.template_name, {'form': form, 'breadcrumbs': self.get_breadcrumbs()})

    def post(self, request, *args, **kwargs):
        choices = self._data_source_choices()
        form = ReportingQueryForm(request.POST, data_source_choices=choices)
        if not form.is_valid():
            return render(request, self.template_name, {'form': form, 'breadcrumbs': self.get_breadcrumbs()})

        ds_id = int(form.cleaned_data['data_source_id'])
        ds_name = dict(choices).get(ds_id, '')
        try:
            created = create_query(
                name=form.cleaned_data['name'],
                sql=form.cleaned_data['sql_text'],
                data_source_id=ds_id,
            )
        except RedashAPIError as exc:
            messages.error(request, f'Redash rejected the query: {exc}')
            return render(request, self.template_name, {'form': form, 'breadcrumbs': self.get_breadcrumbs()})

        try:
            publish_query(created['id'])
        except RedashAPIError as exc:
            logger.warning('Could not publish Redash query id=%s: %s', created['id'], exc)

        ReportingQuery.objects.create(
            name=form.cleaned_data['name'],
            sql_text=form.cleaned_data['sql_text'],
            redash_query_id=created['id'],
            redash_data_source_id=ds_id,
            redash_data_source_name=ds_name,
            created_by=request.user,
        )
        messages.success(request, f"Query \"{form.cleaned_data['name']}\" created in Redash.")
        logger.info(
            'Superuser %s created Redash query "%s" (id=%s)',
            request.user.username, form.cleaned_data['name'], created['id'],
        )
        return redirect('analytics:reporting_query_list')


class ReportingQuerySyncView(SuperuserRequiredMixin, View):
    """Provision one facility-scoped, read-only Redash query per (table, facility).

    Loops over every ``core.Facility`` that has at least one assigned user,
    matches it to its Redash data source by the ``reporting-{client}-{facility}``
    naming convention from ``provision_redash_facility``, and — for every
    HOSPITALS.REPORTING table not already synced for that facility — creates
    a query filtered to that facility's rows via ``source_schema``. Skips
    (table, facility) pairs that already have a ``ReportingQuery`` row, so
    re-running this only fills gaps.
    """

    def post(self, request, *args, **kwargs):
        try:
            data_sources_by_name = {ds['name']: ds['id'] for ds in list_data_sources()}
        except RedashAPIError as exc:
            messages.error(request, f'Could not reach Redash: {exc}')
            return redirect('analytics:reporting_query_list')

        try:
            tables_df = SnowflakeClient().get_tables()
        except SnowflakeQueryError as exc:
            messages.error(request, f'Could not list Snowflake tables: {exc}')
            return redirect('analytics:reporting_query_list')

        reporting_tables = [
            row['TABLE_NAME'] for _, row in tables_df.iterrows()
            if row['SCHEMA_NAME'] == 'REPORTING'
        ]

        facilities = list(Facility.objects.filter(users__isnull=False).distinct().select_related('client'))
        if not facilities:
            messages.error(request, 'No facilities have an assigned user yet — nothing to sync.')
            return redirect('analytics:reporting_query_list')

        created_count = 0
        failed = []
        missing_data_sources = set()
        missing_source_schema = set()

        for facility in facilities:
            if not facility.reporting_source_schema:
                missing_source_schema.add(facility.name)
                continue

            ds_name = f'reporting-{facility.client.slug}-{facility.slug}'
            ds_id = data_sources_by_name.get(ds_name)
            if ds_id is None:
                missing_data_sources.add(ds_name)
                continue

            already_synced = set(
                ReportingQuery.objects
                .filter(facility=facility, source_table__in=[f'REPORTING.{t}' for t in reporting_tables])
                .values_list('source_table', flat=True)
            )

            for table in reporting_tables:
                source_table = f'REPORTING.{table}'
                if source_table in already_synced:
                    continue

                safe_source_schema = facility.reporting_source_schema.replace("'", "''")
                sql = (
                    f'SELECT * FROM HOSPITALS.REPORTING.{table} '
                    f"WHERE source_schema ILIKE '%{safe_source_schema}%'"
                )
                query_name = f'Reporting - {table} ({facility.name})'
                try:
                    created = create_query(name=query_name, sql=sql, data_source_id=ds_id)
                except RedashAPIError as exc:
                    failed.append(query_name)
                    logger.warning('Redash query creation failed for %s: %s', query_name, exc)
                    continue
                try:
                    publish_query(created['id'])
                except RedashAPIError as exc:
                    logger.warning('Could not publish Redash query id=%s: %s', created['id'], exc)

                ReportingQuery.objects.create(
                    name=query_name,
                    sql_text=sql,
                    redash_query_id=created['id'],
                    redash_data_source_id=ds_id,
                    redash_data_source_name=ds_name,
                    source_table=source_table,
                    facility=facility,
                    created_by=request.user,
                )
                created_count += 1

        if created_count:
            messages.success(
                request,
                f"Synced {created_count} facility-scoped quer{'y' if created_count == 1 else 'ies'} into Redash.",
            )
        if failed:
            messages.warning(request, f"Failed to create: {', '.join(failed)}")
        if missing_source_schema:
            messages.warning(
                request,
                f"No reporting_source_schema set for: {', '.join(sorted(missing_source_schema))} "
                '— set it on the Facility (Django admin) first.',
            )
        if missing_data_sources:
            messages.warning(
                request,
                f"No matching Redash data source for: {', '.join(sorted(missing_data_sources))} "
                '— provision that facility first.',
            )
        if not created_count and not failed:
            messages.info(request, 'Nothing to sync — every table is already synced for every reachable facility.')

        logger.info(
            'Superuser %s synced REPORTING schema to Redash: %d created, %d failed, '
            '%d facilities missing data sources, %d missing source_schema',
            request.user.username, created_count, len(failed),
            len(missing_data_sources), len(missing_source_schema),
        )
        return redirect('analytics:reporting_query_list')


class ReportingQueryPublishAllView(SuperuserRequiredMixin, View):
    """Mark every provisioned ReportingQuery as published (not a draft) in Redash."""

    def post(self, request, *args, **kwargs):
        queries = list(ReportingQuery.objects.all())
        published = 0
        failed = []
        for rq in queries:
            try:
                publish_query(rq.redash_query_id)
                published += 1
            except RedashAPIError as exc:
                failed.append(rq.name)
                logger.warning('Could not publish Redash query id=%s: %s', rq.redash_query_id, exc)

        if published:
            messages.success(request, f"Published {published} quer{'y' if published == 1 else 'ies'}.")
        if failed:
            messages.warning(request, f"Failed to publish: {', '.join(failed)}")
        if not queries:
            messages.info(request, 'No reporting queries to publish yet.')

        logger.info(
            'Superuser %s published all reporting queries: %d succeeded, %d failed',
            request.user.username, published, len(failed),
        )
        return redirect('analytics:reporting_query_list')


# ---------------------------------------------------------------------------
# Redash dashboard builder (superuser) — create + publish a Redash dashboard
# from existing Reporting Queries, then list it in the Dashboards page.
# ---------------------------------------------------------------------------

def _redash_dashboard_breadcrumbs():
    return [
        {'label': 'Home', 'url': reverse('analytics:home')},
        {'label': 'Dashboards', 'url': reverse('analytics:dashboard_list')},
        {'label': 'Create Redash Dashboard', 'url': None},
    ]


def _build_dashboard_chart_groups(request, query_ids):
    """Build the per-query chart_groups list for Step 2.

    Fetches each query's visualizations and columns — the columns lookup
    triggers a refresh+poll for any query without a cached result yet
    (bounded wait; see ``get_query_columns``). Every selected query is
    included regardless of outcome, so Step 2 always has something to show
    (even if that's just a "no cached result" note with a retry option).
    """
    selected = ReportingQuery.objects.filter(id__in=query_ids).select_related('facility')
    chart_groups = []
    for rq in selected:
        try:
            query_obj = get_query(rq.redash_query_id)
        except RedashAPIError as exc:
            messages.warning(request, f'Could not load charts for "{rq.name}": {exc}')
            continue
        visualizations = query_obj.get('visualizations') or []
        columns = get_query_columns(rq.redash_query_id)
        chart_groups.append({
            'reporting_query': rq,
            'visualizations': visualizations,
            'columns': columns,
        })
    return chart_groups


class RedashDashboardStep1View(StaffRequiredMixin, BreadcrumbMixin, View):
    """Step 1: name the dashboard and pick which Reporting Queries to pull charts from."""

    template_name = 'analytics/redash_dashboard_step1.html'

    def get_breadcrumbs(self):
        return _redash_dashboard_breadcrumbs()

    def _render_step1(self, request, prefill_name=''):
        reporting_queries = ReportingQuery.objects.select_related('facility').order_by('name')
        return render(request, self.template_name, {
            'reporting_queries': reporting_queries,
            'breadcrumbs': self.get_breadcrumbs(),
            'prefill_name': prefill_name,
        })

    def get(self, request, *args, **kwargs):
        return self._render_step1(request)

    def post(self, request, *args, **kwargs):
        name = request.POST.get('name', '').strip()
        query_ids = request.POST.getlist('query_ids')

        if not name or not query_ids:
            messages.error(request, 'Give the dashboard a name and select at least one query.')
            return self._render_step1(request, prefill_name=name)

        chart_groups = _build_dashboard_chart_groups(request, query_ids)
        if not chart_groups:
            messages.error(request, 'None of the selected queries have any charts to add.')
            return self._render_step1(request, prefill_name=name)

        return render(request, 'analytics/redash_dashboard_step2.html', {
            'dashboard_name': name,
            'query_ids': query_ids,
            'chart_groups': chart_groups,
            'chart_group_count': len(chart_groups),
            'redash_base_url': settings.REDASH_BASE_URL,
            'breadcrumbs': self.get_breadcrumbs(),
        })


class RedashDashboardRefreshQueriesView(StaffRequiredMixin, View):
    """Step 2: retry chart-building state for the same set of queries.

    Lets a query that had no cached result (and timed out during Step 1)
    get another refresh+poll attempt without leaving this page to run it
    in Redash directly. Queries that already succeeded just re-fetch their
    (already cached) columns quickly.
    """

    def post(self, request, *args, **kwargs):
        name = request.POST.get('name', '').strip()
        query_ids = request.POST.getlist('query_ids')

        if not name or not query_ids:
            messages.error(request, 'Missing dashboard name or queries — please start over.')
            return redirect('analytics:redash_dashboard_create')

        chart_groups = _build_dashboard_chart_groups(request, query_ids)
        if not chart_groups:
            messages.error(request, 'None of the selected queries have any charts to add.')
            return redirect('analytics:redash_dashboard_create')

        return render(request, 'analytics/redash_dashboard_step2.html', {
            'dashboard_name': name,
            'query_ids': query_ids,
            'chart_groups': chart_groups,
            'chart_group_count': len(chart_groups),
            'redash_base_url': settings.REDASH_BASE_URL,
            'breadcrumbs': _redash_dashboard_breadcrumbs(),
        })


class RedashDashboardFinalizeView(StaffRequiredMixin, View):
    """Step 2 submit: create the dashboard in Redash, add the picked charts,
    publish it, enable public sharing, and record it as a Dashboard so it
    shows up in the Dashboards list with its published Redash URL.
    """

    def post(self, request, *args, **kwargs):
        name = request.POST.get('name', '').strip()
        visualization_ids = request.POST.getlist('visualization_ids')

        group_count = int(request.POST.get('chart_group_count', 0) or 0)
        new_charts = []
        for i in range(group_count):
            chart_name = request.POST.get(f'new_chart_name_{i}', '').strip()
            if not chart_name:
                continue
            query_id = request.POST.get(f'new_chart_query_id_{i}')
            series_type = request.POST.get(f'new_chart_type_{i}', 'bar')
            x_column = request.POST.get(f'new_chart_x_{i}', '')
            y_columns = request.POST.getlist(f'new_chart_y_{i}')
            if query_id and x_column and y_columns:
                new_charts.append({
                    'query_id': int(query_id),
                    'name': chart_name,
                    'series_type': series_type,
                    'x_column': x_column,
                    'y_columns': y_columns,
                })

        if not name or (not visualization_ids and not new_charts):
            messages.error(request, 'Give the dashboard a name and select or build at least one chart.')
            return redirect('analytics:redash_dashboard_create')

        profile = getattr(request.user, 'profile', None)
        client = getattr(profile, 'client', None)
        facility = getattr(profile, 'facility', None)

        try:
            dashboard = create_dashboard(name)
        except RedashAPIError as exc:
            messages.error(request, f'Could not create the dashboard: {exc}')
            return redirect('analytics:redash_dashboard_create')

        dashboard_id = dashboard['id']
        added = 0

        for chart in new_charts:
            try:
                viz = create_visualization(
                    query_id=chart['query_id'],
                    name=chart['name'],
                    series_type=chart['series_type'],
                    x_column=chart['x_column'],
                    y_columns=chart['y_columns'],
                )
                create_widget(dashboard_id, viz['id'])
                added += 1
            except RedashAPIError as exc:
                logger.warning('Could not create new chart "%s": %s', chart['name'], exc)
                messages.warning(request, f"Could not create chart \"{chart['name']}\": {exc}")

        for viz_id in visualization_ids:
            try:
                create_widget(dashboard_id, int(viz_id))
                added += 1
            except RedashAPIError as exc:
                logger.warning('Could not add widget (viz_id=%s) to dashboard %s: %s', viz_id, dashboard_id, exc)

        if not added:
            messages.error(request, 'Could not add any charts to the dashboard — check Redash connectivity.')
            return redirect('analytics:redash_dashboard_create')

        try:
            publish_dashboard(dashboard_id)
        except RedashAPIError as exc:
            logger.warning('Could not publish dashboard %s: %s', dashboard_id, exc)

        try:
            share = share_dashboard(dashboard_id)
            public_url = share['public_url']
        except RedashAPIError as exc:
            messages.error(request, f'Dashboard created in Redash but could not be published publicly: {exc}')
            return redirect('analytics:dashboard_list')

        Dashboard.objects.create(
            name=name,
            redash_dashboard_url=public_url,
            client=client,
            facility=facility,
            created_by=request.user,
        )

        messages.success(
            request,
            f'Dashboard "{name}" created in Redash with {added} chart(s) and added to your Dashboards list.',
        )
        logger.info(
            'Superuser %s created Redash dashboard "%s" (id=%s) with %d widgets',
            request.user.username, name, dashboard_id, added,
        )
        return redirect('analytics:dashboard_list')
