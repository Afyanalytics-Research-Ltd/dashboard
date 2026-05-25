"""
Reusable Django view mixins for the Afya DataHub platform.
"""

import logging
from typing import Any

from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.core.exceptions import PermissionDenied
from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from django.http import HttpRequest

from authentication.roles import in_role
from .models import AuditLog

logger = logging.getLogger(__name__)


class BreadcrumbMixin:
    """
    Provides a `breadcrumbs` list in the template context.

    Override `get_breadcrumbs()` in your view to return a list of dicts:
        [{'label': 'Home', 'url': '/'}, {'label': 'Detail', 'url': None}]

    A `url` of None means the crumb is the current (active) page.
    """

    breadcrumbs: list[dict[str, str | None]] = []

    def get_breadcrumbs(self) -> list[dict[str, str | None]]:
        return self.breadcrumbs

    def get_context_data(self, **kwargs) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)  # type: ignore[misc]
        context['breadcrumbs'] = self.get_breadcrumbs()
        return context


class LoggingMixin:
    """
    Provides an `audit_log()` helper that CBVs can call to record actions
    in AuditLog without boilerplate.
    """

    def audit_log(
        self,
        action: str,
        resource: str,
        resource_id: str = '',
        detail: str = '',
    ) -> None:
        request: HttpRequest = self.request  # type: ignore[attr-defined]
        AuditLog.log(
            user=request.user,
            action=action,
            resource=resource,
            resource_id=resource_id,
            detail=detail,
            ip_address=_get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
        )


class RoleRequiredMixin(LoginRequiredMixin, UserPassesTestMixin):
    """
    CBV mixin: require the logged-in user to have at least one of the given roles.

    Usage:
        class MyView(RoleRequiredMixin, TemplateView):
            required_roles = [ROLE_CLIENT_ADMIN, ROLE_FACILITIES_ADMIN]
    """

    required_roles: list[str] = []
    raise_exception = True  # raise PermissionDenied instead of redirecting

    def test_func(self) -> bool:
        user = self.request.user  # type: ignore[attr-defined]
        if not self.required_roles:
            return user.is_authenticated
        return in_role(user, *self.required_roles)

    def handle_no_permission(self):
        user = getattr(self.request, 'user', None)
        if user and user.is_authenticated:
            raise PermissionDenied('You do not have permission to access this page.')
        return super().handle_no_permission()


class SuperuserRequiredMixin(LoginRequiredMixin, UserPassesTestMixin):
    """Restrict a view to superusers only."""

    raise_exception = True

    def test_func(self) -> bool:
        return self.request.user.is_superuser  # type: ignore[attr-defined]

    def handle_no_permission(self):
        user = getattr(self.request, 'user', None)
        if user and user.is_authenticated:
            raise PermissionDenied('Superuser access is required.')
        return super().handle_no_permission()


class PaginationMixin:
    """
    Adds paginator helpers to class-based list views.

    Set `paginate_by` on the view (default 20).
    Adds `page_obj` and `paginator` to context automatically via
    `paginate_queryset()`.
    """

    paginate_by: int = 20

    def paginate_queryset(self, queryset, page_size: int | None = None):
        page_size = page_size or self.paginate_by
        paginator = Paginator(queryset, page_size)
        page_number = self.request.GET.get('page', 1)  # type: ignore[attr-defined]
        try:
            page_obj = paginator.page(page_number)
        except PageNotAnInteger:
            page_obj = paginator.page(1)
        except EmptyPage:
            page_obj = paginator.page(paginator.num_pages)
        return paginator, page_obj

    def get_context_data(self, **kwargs) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)  # type: ignore[misc]
        queryset = context.get('object_list') or context.get('queryset')
        if queryset is not None:
            paginator, page_obj = self.paginate_queryset(queryset)
            context['paginator'] = paginator
            context['page_obj'] = page_obj
            context['is_paginated'] = page_obj.has_other_pages()
        return context


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_client_ip(request: HttpRequest) -> str | None:
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        return x_forwarded_for.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR')
