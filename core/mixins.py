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
    """Injects a ``breadcrumbs`` list into every template context.

    Override :meth:`get_breadcrumbs` in your view class to define the trail.
    The mixin calls it automatically from :meth:`get_context_data` so
    templates always have access to ``{{ breadcrumbs }}`` without extra
    boilerplate in each view.

    Non-technical explanation:
        Breadcrumbs are the "Home > Dashboards > KSH Revenue" navigation
        trail at the top of a page.  This mixin makes it easy for any view
        to provide that trail without repeating the same code everywhere.

    Example:
        .. code-block:: python

            class MyView(BreadcrumbMixin, TemplateView):
                def get_breadcrumbs(self):
                    return [
                        {'label': 'Home',       'url': '/'},
                        {'label': 'My Section', 'url': '/my/'},
                        {'label': 'Detail',     'url': None},  # current page
                    ]

    A ``url`` of ``None`` marks the crumb as the current (active) page,
    which templates typically render as plain text rather than a link.
    """

    breadcrumbs: list[dict[str, str | None]] = []

    def get_breadcrumbs(self) -> list[dict[str, str | None]]:
        """Return the breadcrumb trail for this view.

        Subclasses should override this method to provide a specific trail.

        Returns:
            A list of dicts, each with ``"label"`` (display text) and
            ``"url"`` (link target or ``None`` for the current page).
        """
        return self.breadcrumbs

    def get_context_data(self, **kwargs) -> dict[str, Any]:
        """Add ``breadcrumbs`` to the template context.

        Args:
            **kwargs: Forwarded to the parent ``get_context_data``.

        Returns:
            The context dict with ``breadcrumbs`` added.
        """
        context = super().get_context_data(**kwargs)  # type: ignore[misc]
        context['breadcrumbs'] = self.get_breadcrumbs()
        return context


class LoggingMixin:
    """Provides a one-line ``audit_log()`` helper for class-based views.

    Mix this into any CBV that needs to record user actions in the
    :class:`core.models.AuditLog` table.  The mixin handles extracting
    the user, IP address, and User-Agent from ``self.request`` so callers
    only need to supply the action and resource details.

    Non-technical explanation:
        Gives any view the ability to write an entry in the activity log
        (e.g. "User A exported the KSH revenue dashboard") without having
        to repeat the same setup code in every view.
    """

    def audit_log(
        self,
        action: str,
        resource: str,
        resource_id: str = '',
        detail: str = '',
    ) -> None:
        """Write an :class:`AuditLog` entry for the current request.

        Automatically extracts the user, IP address, and User-Agent from
        ``self.request`` so callers only need to provide the action and
        resource information.

        Args:
            action: What happened, e.g. ``"read"``, ``"update"``,
                ``"export"``.  Must be one of ``AuditLog.ACTION_CHOICES``.
            resource: The thing that was acted on, e.g. ``"dashboard"``,
                ``"authentication.userprofile"``.
            resource_id: Optional unique identifier of the specific object,
                e.g. the slug ``"ksh-revenue"`` or a primary key.
            detail: Free-text description for context, e.g.
                ``"Viewed dashboard: KSH Revenue"``.
        """
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
    """CBV mixin that restricts a view to users holding specific roles.

    Set ``required_roles`` on the view class to one or more role constants
    from :mod:`authentication.roles`.  Users who do not hold any of the
    listed roles receive a 403 Forbidden response.  Unauthenticated users
    are redirected to the login page.

    Non-technical explanation:
        Like a door that checks your badge before letting you in.  You
        specify which badge types are allowed (``required_roles``), and
        anyone without the right badge gets turned away.

    Usage:
        .. code-block:: python

            from authentication.roles import ROLE_CLIENT_ADMIN, ROLE_FACILITIES_ADMIN

            class ReportView(RoleRequiredMixin, TemplateView):
                required_roles = [ROLE_CLIENT_ADMIN, ROLE_FACILITIES_ADMIN]
    """

    required_roles: list[str] = []
    raise_exception = True  # raise PermissionDenied instead of redirecting

    def test_func(self) -> bool:
        """Return ``True`` if the user has at least one of the required roles.

        If ``required_roles`` is empty, any authenticated user passes.

        Returns:
            ``True`` if access should be granted; ``False`` to trigger
            :meth:`handle_no_permission`.
        """
        user = self.request.user  # type: ignore[attr-defined]
        if not self.required_roles:
            return user.is_authenticated
        return in_role(user, *self.required_roles)

    def handle_no_permission(self):
        """Raise 403 for authenticated users; redirect unauthenticated users to login.

        Raises:
            PermissionDenied: If the user is authenticated but lacks the
                required role.
        """
        user = getattr(self.request, 'user', None)
        if user and user.is_authenticated:
            raise PermissionDenied('You do not have permission to access this page.')
        return super().handle_no_permission()


class SuperuserRequiredMixin(LoginRequiredMixin, UserPassesTestMixin):
    """Restrict a view to Django superusers only.

    Any authenticated non-superuser receives a 403 Forbidden response.
    Unauthenticated visitors are redirected to the login page.

    Non-technical explanation:
        The highest-security door — only the platform administrator
        (superuser) can walk through.  Even Client Admins are turned away.
    """

    raise_exception = True

    def test_func(self) -> bool:
        """Return ``True`` only if the requesting user is a superuser.

        Returns:
            ``True`` if ``request.user.is_superuser`` is set.
        """
        return self.request.user.is_superuser  # type: ignore[attr-defined]

    def handle_no_permission(self):
        """Raise 403 for authenticated non-superusers; redirect others to login.

        Raises:
            PermissionDenied: If the user is authenticated but not a superuser.
        """
        user = getattr(self.request, 'user', None)
        if user and user.is_authenticated:
            raise PermissionDenied('Superuser access is required.')
        return super().handle_no_permission()


class StaffRequiredMixin(LoginRequiredMixin, UserPassesTestMixin):
    """Restrict a view to Django staff users (``is_staff=True``).

    Any authenticated non-staff user receives a 403 Forbidden response.
    Unauthenticated visitors are redirected to the login page. Less strict
    than :class:`SuperuserRequiredMixin` — staff status can be granted to
    trusted non-superuser accounts via the Django admin.
    """

    raise_exception = True

    def test_func(self) -> bool:
        """Return ``True`` only if the requesting user has ``is_staff`` set."""
        return self.request.user.is_staff  # type: ignore[attr-defined]

    def handle_no_permission(self):
        """Raise 403 for authenticated non-staff users; redirect others to login."""
        user = getattr(self.request, 'user', None)
        if user and user.is_authenticated:
            raise PermissionDenied('Staff access is required.')
        return super().handle_no_permission()


class PaginationMixin:
    """Adds Django-style pagination helpers to class-based views.

    Set ``paginate_by`` on the view class (default 20 items per page).
    :meth:`get_context_data` automatically adds ``paginator``, ``page_obj``,
    and ``is_paginated`` to the template context, matching Django's built-in
    :class:`~django.views.generic.list.MultipleObjectMixin` API.

    Non-technical explanation:
        Splits a long list of records (e.g. 500 audit log entries) into
        smaller pages (e.g. 20 per page) so the browser doesn't load
        everything at once — like page numbers at the bottom of a search
        results page.

    Usage:
        .. code-block:: python

            class MyListView(PaginationMixin, TemplateView):
                paginate_by = 15  # 15 items per page
    """

    paginate_by: int = 20

    def paginate_queryset(self, queryset, page_size: int | None = None):
        """Split ``queryset`` into pages and return the requested page.

        Handles edge cases gracefully:
        - If ``page`` is not an integer, returns page 1.
        - If ``page`` is beyond the last page, returns the last page.

        Args:
            queryset: Any pageable sequence (QuerySet, list, etc.).
            page_size: Items per page; defaults to ``self.paginate_by``.

        Returns:
            A tuple ``(paginator, page_obj)`` where ``page_obj`` is the
            :class:`~django.core.paginator.Page` for the current request.
        """
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
        """Inject pagination variables into the template context.

        Looks for ``object_list`` or ``queryset`` in the existing context
        (both names are used by different generic view base classes).

        Args:
            **kwargs: Forwarded to the parent ``get_context_data``.

        Returns:
            The context dict enriched with ``paginator``, ``page_obj``,
            and ``is_paginated``.
        """
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
