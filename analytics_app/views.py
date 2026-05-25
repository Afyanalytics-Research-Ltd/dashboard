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
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse, reverse_lazy
from django.views.generic import (
    CreateView, DeleteView, DetailView,
    ListView, TemplateView, UpdateView, View,
)

from core.mixins import BreadcrumbMixin, LoggingMixin, SuperuserRequiredMixin
from core.models import AuditLog
from warehouse.models import TrackedSpreadsheet

from .forms import DashboardForm, DashboardSearchForm
from .models import Dashboard

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
    logging.warning(f"{folder} --------------------->")
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
        .update(is_active=False)
    )
    return {'created': created, 'updated': updated, 'deactivated': deactivated}


def _get_client_obj(user):
    """Return the Client FK for the user if profile has one, else None."""
    try:
        profile = user.profile
        client_name = profile.client
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
