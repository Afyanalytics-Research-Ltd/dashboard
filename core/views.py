"""
Core HTML views: system settings, error handlers.
"""

import logging
from typing import Any

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.decorators import method_decorator
from django.views import View
from django.views.generic import TemplateView

from django.contrib.auth import get_user_model

from authentication.models import UserModuleGrant
from authentication.module_access import get_module_overrides
from authentication.roles import ROLE_CLIENT_ADMIN, ROLE_FACILITIES_ADMIN, ROLE_FACILITY_ADMIN

from .mixins import BreadcrumbMixin, LoggingMixin, RoleRequiredMixin, StaffRequiredMixin, SuperuserRequiredMixin
from .models import AuditLog, Client, Facility, Notification, SystemSettings, Ticket, TicketComment

User = get_user_model()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System Settings
# ---------------------------------------------------------------------------

@method_decorator(login_required, name='dispatch')
class SystemSettingsView(SuperuserRequiredMixin, BreadcrumbMixin, LoggingMixin, TemplateView):
    """Platform-wide system settings page (superusers only).

    Displays all :class:`SystemSettings` key/value pairs and accepts AJAX
    POST requests to update individual settings in-place without a full
    page reload.

    Access is restricted to Django superusers via
    :class:`core.mixins.SuperuserRequiredMixin`.

    Non-technical explanation:
        This is the admin control panel for the whole platform — like the
        settings menu on a phone that only the IT administrator can open.
        Superusers can read and change platform-wide configuration values
        here without touching any code.
    """

    template_name = 'core/settings.html'

    def get_breadcrumbs(self):
        """Return the breadcrumb trail for the settings page."""
        return [
            {'label': 'Home', 'url': '/analytics/'},
            {'label': 'System Settings', 'url': None},
        ]

    def get_context_data(self, **kwargs) -> dict[str, Any]:
        """Build the template context with all current system settings.

        Returns:
            A dict containing:
            - ``settings_list``: All :class:`SystemSettings` rows.
            - ``sidebar_section``: Highlights the correct sidebar link.
            - ``page_title``: Title shown in the browser tab.
        """
        context = super().get_context_data(**kwargs)
        context['settings_list'] = SystemSettings.objects.all()
        context['sidebar_section'] = 'settings'
        context['page_title'] = 'System Settings'
        return context

    def post(self, request: HttpRequest) -> JsonResponse:
        """Handle an AJAX request to create or update a system setting.

        Expects ``key``, ``value``, and optionally ``description`` in the
        POST body.  The ``value`` is parsed as JSON if possible so that
        booleans, numbers, and objects are stored correctly; otherwise it
        is stored as a plain string.

        Args:
            request: The incoming HTTP POST request.

        Returns:
            A JSON response with ``{"ok": true, "key": ..., "value": ...,
            "updated_at": ...}`` on success, or ``{"ok": false, "error":
            ...}`` with status 400 on validation failure.
        """
        key = request.POST.get('key', '').strip()
        value = request.POST.get('value', '')
        description = request.POST.get('description', '').strip()

        if not key:
            return JsonResponse({'ok': False, 'error': 'Key is required.'}, status=400)

        # Parse value as JSON if possible, else store as string
        import json
        try:
            parsed_value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            parsed_value = value

        setting = SystemSettings.set(key=key, value=parsed_value, user=request.user, description=description)
        self.audit_log('update', 'SystemSettings', resource_id=key, detail=f'Updated setting {key}')
        logger.info('Superuser %s updated system setting: %s', request.user.username, key)

        return JsonResponse({
            'ok': True,
            'key': setting.key,
            'value': setting.value,
            'updated_at': setting.updated_at.isoformat(),
        })


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------

@method_decorator(login_required, name='dispatch')
class PermissionsView(RoleRequiredMixin, BreadcrumbMixin, LoggingMixin, TemplateView):
    """Lets a facility/facilities/client administrator control what specific
    users under their scope can and cannot see.

    Two independent things can be assigned per user:
      - Module access (Warehouse / Analytics / AI Chatbot) — an explicit
        grant or revoke overriding the user's role-based default, stored as
        :class:`authentication.models.UserModuleGrant`.
      - Dashboard visibility — hiding a specific dashboard (that would
        otherwise be visible to the user's whole client) from one user,
        stored on :class:`analytics_app.models.Dashboard.hidden_from_users`.

    Which users an admin can manage:
      - Superuser: every user.
      - Client Admin / Facilities Admin: every user under the same Client.
      - Facility Admin: only users linked to the same Facility.

    Non-technical explanation:
        A roster page for administrators — for every person under your
        scope, flip switches for which parts of the platform they can open
        and which dashboards they're allowed to see, without touching the
        Django admin or writing any code.
    """

    template_name = 'core/permissions.html'
    required_roles = [ROLE_CLIENT_ADMIN, ROLE_FACILITIES_ADMIN, ROLE_FACILITY_ADMIN]

    def get_breadcrumbs(self):
        return [
            {'label': 'Home', 'url': '/analytics/'},
            {'label': 'Permissions', 'url': None},
        ]

    def _manageable_users(self, acting_user):
        """Return the queryset of users ``acting_user`` is allowed to manage."""
        base = User.objects.select_related('profile', 'profile__facility', 'profile__client')
        if acting_user.is_superuser:
            qs = base
        else:
            profile = getattr(acting_user, 'profile', None)
            if profile and profile.is_facilities_admin:  # Client Admin or Facilities Admin
                qs = base.filter(profile__client=profile.client) if profile.client else base.none()
            elif profile and profile.facility:
                qs = base.filter(profile__facility=profile.facility)
            else:
                qs = base.none()
        return qs.exclude(pk=acting_user.pk).order_by('username')

    def get_context_data(self, **kwargs) -> dict[str, Any]:
        import json

        from analytics_app.models import Dashboard

        context = super().get_context_data(**kwargs)
        acting_user = self.request.user
        users = list(self._manageable_users(acting_user))

        client_obj = getattr(getattr(acting_user, 'profile', None), 'client', None)
        if acting_user.is_superuser:
            dashboards = Dashboard.objects.filter(is_active=True)
        elif client_obj:
            dashboards = Dashboard.objects.filter(is_active=True, client=client_obj)
        else:
            dashboards = Dashboard.objects.none()
        dashboards = list(dashboards.order_by('name'))

        overrides_by_user = {u.pk: get_module_overrides(u) for u in users}

        # Build a fully pre-resolved (user, dashboard, visible) structure so the
        # template never needs a dict-lookup-by-variable-key.
        managed_users_data = []
        for u in users:
            hidden_ids = set(u.hidden_dashboards.values_list('pk', flat=True))
            managed_users_data.append({
                'user': u,
                'dashboards': [
                    {'dashboard': d, 'visible': d.pk not in hidden_ids}
                    for d in dashboards
                ],
            })

        context.update({
            'sidebar_section': 'permissions',
            'page_title': 'Permissions',
            'managed_users': users,
            'managed_users_data': managed_users_data,
            'module_choices': UserModuleGrant.MODULE_CHOICES,
            'overrides_by_user': overrides_by_user,
            'overrides_by_user_json': json.dumps({str(k): v for k, v in overrides_by_user.items()}),
            'dashboards': dashboards,
        })
        return context

    def post(self, request: HttpRequest, *args, **kwargs) -> JsonResponse:
        """Handle a single grant/revoke/toggle action via AJAX.

        Expects ``action`` (``set_module`` | ``clear_module`` |
        ``toggle_dashboard``) and ``user_id`` in the POST body, plus
        action-specific fields. The target user must fall within the
        acting admin's manageable scope, or the request is rejected.
        """
        action = request.POST.get('action', '').strip()
        target_user = get_object_or_404(
            self._manageable_users(request.user), pk=request.POST.get('user_id')
        )

        if action == 'set_module':
            module_key = request.POST.get('module_key', '').strip()
            if module_key not in dict(UserModuleGrant.MODULE_CHOICES):
                return JsonResponse({'ok': False, 'error': 'Unknown module.'}, status=400)
            is_granted = request.POST.get('is_granted') == 'true'
            UserModuleGrant.objects.update_or_create(
                user=target_user,
                module_key=module_key,
                defaults={'is_granted': is_granted, 'granted_by': request.user},
            )
            self.audit_log(
                'update', 'UserModuleGrant', resource_id=f'{target_user.username}:{module_key}',
                detail=f'{"Granted" if is_granted else "Revoked"} {module_key} for {target_user.username}',
            )
            return JsonResponse({'ok': True, 'module_key': module_key, 'is_granted': is_granted})

        if action == 'clear_module':
            module_key = request.POST.get('module_key', '').strip()
            UserModuleGrant.objects.filter(user=target_user, module_key=module_key).delete()
            self.audit_log(
                'update', 'UserModuleGrant', resource_id=f'{target_user.username}:{module_key}',
                detail=f'Cleared override, reverting to role default for {target_user.username}',
            )
            return JsonResponse({'ok': True, 'module_key': module_key, 'cleared': True})

        if action == 'toggle_dashboard':
            from analytics_app.models import Dashboard

            dashboard = get_object_or_404(Dashboard, pk=request.POST.get('dashboard_id'))
            client_obj = getattr(getattr(request.user, 'profile', None), 'client', None)
            if not request.user.is_superuser and dashboard.client != client_obj:
                return JsonResponse({'ok': False, 'error': 'Dashboard outside your scope.'}, status=403)

            hide = request.POST.get('hidden') == 'true'
            if hide:
                dashboard.hidden_from_users.add(target_user)
            else:
                dashboard.hidden_from_users.remove(target_user)
            self.audit_log(
                'update', 'Dashboard', resource_id=str(dashboard.pk),
                detail=f'{"Hid" if hide else "Unhid"} "{dashboard.name}" for {target_user.username}',
            )
            return JsonResponse({'ok': True, 'dashboard_id': dashboard.pk, 'hidden': hide})

        return JsonResponse({'ok': False, 'error': 'Unknown action.'}, status=400)


@method_decorator(login_required, name='dispatch')
class NotificationListView(BreadcrumbMixin, TemplateView):
    """In-app notification inbox for the currently logged-in user.

    Renders all notifications belonging to the request user, newest first.
    Login is required (enforced by the ``login_required`` decorator applied
    to ``dispatch``).

    Non-technical explanation:
        This is the bell icon / inbox page — showing all messages the
        platform has sent to the current user, such as "your export is
        ready" or "a new dashboard has been added for your facility."
    """

    template_name = 'core/notifications.html'

    def get_breadcrumbs(self):
        """Return the breadcrumb trail for the notifications page."""
        return [
            {'label': 'Home', 'url': '/analytics/'},
            {'label': 'Notifications', 'url': None},
        ]

    def get_context_data(self, **kwargs) -> dict[str, Any]:
        """Build the template context with the user's notifications.

        Returns:
            A dict containing:
            - ``notifications``: QuerySet of the user's notifications,
              ordered newest-first.
            - ``sidebar_section``: Highlights the correct sidebar link.
            - ``page_title``: Title shown in the browser tab.
        """
        context = super().get_context_data(**kwargs)
        context['notifications'] = Notification.objects.filter(user=self.request.user)
        context['sidebar_section'] = 'profile'
        context['page_title'] = 'Notifications'
        return context


@method_decorator(login_required, name='dispatch')
class MarkNotificationReadView(View):
    """Mark a single notification as read via an AJAX POST request.

    Ownership is enforced — users can only mark their own notifications,
    not those belonging to other accounts.
    """

    def post(self, request: HttpRequest, pk: int) -> JsonResponse:
        """Mark the notification identified by ``pk`` as read.

        Args:
            request: The incoming HTTP POST request.
            pk: Primary key of the :class:`Notification` to mark read.

        Returns:
            ``{"ok": true}`` on success, or 404 if the notification does
            not exist or belongs to a different user.
        """
        notification = get_object_or_404(Notification, pk=pk, user=request.user)
        notification.mark_read()
        return JsonResponse({'ok': True})


@method_decorator(login_required, name='dispatch')
class MarkAllNotificationsReadView(View):
    """Mark every unread notification for the current user as read in bulk.

    More efficient than calling :class:`MarkNotificationReadView` once per
    notification because it uses a single database UPDATE statement.
    """

    def post(self, request: HttpRequest) -> HttpResponse:
        """Mark all unread notifications as read and redirect back.

        Args:
            request: The incoming HTTP POST request.

        Returns:
            A redirect to the referring page (or ``/analytics/`` as
            fallback), with a success flash message indicating how many
            notifications were cleared.
        """
        count = Notification.objects.filter(user=request.user, is_read=False).update(is_read=True)
        messages.success(request, f'{count} notification(s) marked as read.')
        return redirect(request.META.get('HTTP_REFERER', '/analytics/'))


# ---------------------------------------------------------------------------
# Support & Ticketing
# ---------------------------------------------------------------------------

@method_decorator(login_required, name='dispatch')
class TicketCreateAPIView(View):
    """AJAX endpoint backing the three global ticket modals (Issue / Suggestion /
    New Feature) defined in ``base.html``, plus the "New Ticket" flow on the
    Support & Ticketing page itself. Any authenticated user may file a ticket.
    """

    def post(self, request: HttpRequest) -> JsonResponse:
        """Create a :class:`Ticket` from AJAX form data.

        Expects ``ticket_type`` (one of ``Ticket.TYPE_CHOICES``), ``subject``,
        ``description``, and optionally ``priority``, ``page_url``, and an
        ``attachment`` file.

        Returns:
            ``{"ok": true, "ticket": {...}}`` on success, or
            ``{"ok": false, "error": "..."}`` with status 400 on validation
            failure.
        """
        ticket_type = request.POST.get('ticket_type', '').strip()
        subject = request.POST.get('subject', '').strip()
        description = request.POST.get('description', '').strip()
        priority = request.POST.get('priority', Ticket.PRIORITY_MEDIUM).strip()
        page_url = request.POST.get('page_url', '').strip()[:500]

        if ticket_type not in dict(Ticket.TYPE_CHOICES):
            return JsonResponse({'ok': False, 'error': 'Please choose a valid ticket type.'}, status=400)
        if not subject:
            return JsonResponse({'ok': False, 'error': 'Subject is required.'}, status=400)
        if not description:
            return JsonResponse({'ok': False, 'error': 'Description is required.'}, status=400)
        if priority not in dict(Ticket.PRIORITY_CHOICES):
            priority = Ticket.PRIORITY_MEDIUM

        profile = getattr(request.user, 'profile', None)
        ticket = Ticket.objects.create(
            ticket_type=ticket_type,
            subject=subject[:200],
            description=description,
            priority=priority,
            page_url=page_url,
            attachment=request.FILES.get('attachment'),
            created_by=request.user,
            client=getattr(profile, 'client', None),
            facility=getattr(profile, 'facility', None),
        )

        AuditLog.log(
            user=request.user, action='create', resource='Ticket', resource_id=str(ticket.pk),
            detail=f'{ticket.get_ticket_type_display()}: {ticket.subject}',
            ip_address=request.META.get('REMOTE_ADDR'),
        )

        for staff_user in User.objects.filter(is_staff=True, is_active=True):
            Notification.send(
                staff_user,
                title=f'New {ticket.get_ticket_type_display().lower()}',
                message=f'{request.user.get_full_name() or request.user.username}: {ticket.subject}',
                notification_type='warning' if ticket_type == Ticket.TYPE_ISSUE else 'info',
                link='/core/support/',
            )

        return JsonResponse({
            'ok': True,
            'ticket': {
                'id': ticket.pk,
                'subject': ticket.subject,
                'ticket_type': ticket.ticket_type,
                'ticket_type_display': ticket.get_ticket_type_display(),
                'status': ticket.status,
                'status_display': ticket.get_status_display(),
                'created_at': ticket.created_at.strftime('%d %b %Y, %H:%M'),
            },
        })


@method_decorator(login_required, name='dispatch')
class SupportView(BreadcrumbMixin, LoggingMixin, TemplateView):
    """Support & Ticketing home page.

    Every authenticated user sees "My Tickets" — everything they've
    personally filed. Staff (``is_staff``) additionally see a Kanban board
    of every ticket in the system, grouped by status, with drag-and-drop
    status changes and an inline comment thread per ticket.

    Non-technical explanation:
        The help-desk page. Anyone can check on the issues/suggestions/
        feature requests they've submitted. The support team sees
        everyone's, laid out like a whiteboard with columns for Open, In
        Progress, Resolved, and Closed, so they can drag a card across as
        they work through it.
    """

    template_name = 'core/support.html'

    def get_breadcrumbs(self):
        return [
            {'label': 'Home', 'url': '/analytics/'},
            {'label': 'Support & Ticketing', 'url': None},
        ]

    def get_context_data(self, **kwargs) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        user = self.request.user

        my_tickets = list(
            Ticket.objects.filter(created_by=user).select_related('assigned_to')
        )

        context.update({
            'sidebar_section': 'support',
            'page_title': 'Support & Ticketing',
            'ticket_types': Ticket.TYPE_CHOICES,
            'priorities': Ticket.PRIORITY_CHOICES,
            'my_tickets': my_tickets,
            'is_support_staff': user.is_staff,
        })

        if user.is_staff:
            all_tickets = list(
                Ticket.objects.select_related('created_by', 'assigned_to', 'facility')
            )
            board_columns = []
            for status_key, status_label in Ticket.STATUS_CHOICES:
                tickets_in_col = [t for t in all_tickets if t.status == status_key]
                board_columns.append({
                    'key': status_key,
                    'label': status_label,
                    'tickets': tickets_in_col,
                    'count': len(tickets_in_col),
                })
            context.update({
                'status_choices': Ticket.STATUS_CHOICES,
                'board_columns': board_columns,
                'open_count': sum(1 for t in all_tickets if t.status == Ticket.STATUS_OPEN),
                'total_count': len(all_tickets),
            })

        return context


class _TicketAccessMixin:
    """Shared helper for the ticket detail/comment/status AJAX endpoints.

    A ticket is visible to its creator and to any staff user; everyone
    else gets a 404 (not a 403) so the existence of another user's ticket
    isn't leaked.
    """

    def _get_visible_ticket(self, request: HttpRequest, pk: int) -> Ticket:
        ticket = get_object_or_404(Ticket.objects.select_related('created_by', 'assigned_to'), pk=pk)
        if not request.user.is_staff and ticket.created_by_id != request.user.pk:
            from django.http import Http404
            raise Http404('Ticket not found.')
        return ticket


@method_decorator(login_required, name='dispatch')
class TicketDetailAPIView(_TicketAccessMixin, View):
    """Return a ticket's full detail plus its comment thread as JSON.

    Internal (staff-only) comments are omitted for non-staff viewers, even
    though non-staff viewers can only reach their own tickets in the first
    place — this keeps internal notes private even if a ticket is later
    reassigned or shared.
    """

    def get(self, request: HttpRequest, pk: int) -> JsonResponse:
        ticket = self._get_visible_ticket(request, pk)
        comments = ticket.comments.select_related('author')
        if not request.user.is_staff:
            comments = comments.filter(is_internal=False)

        return JsonResponse({
            'ok': True,
            'ticket': {
                'id': ticket.pk,
                'subject': ticket.subject,
                'description': ticket.description,
                'ticket_type': ticket.ticket_type,
                'ticket_type_display': ticket.get_ticket_type_display(),
                'status': ticket.status,
                'status_display': ticket.get_status_display(),
                'priority': ticket.priority,
                'priority_display': ticket.get_priority_display(),
                'page_url': ticket.page_url,
                'attachment_url': ticket.attachment.url if ticket.attachment else '',
                'created_by': ticket.created_by.get_full_name() or ticket.created_by.username if ticket.created_by else 'Unknown',
                'assigned_to': (ticket.assigned_to.get_full_name() or ticket.assigned_to.username) if ticket.assigned_to else None,
                'created_at': ticket.created_at.strftime('%d %b %Y, %H:%M'),
                'resolution_notes': ticket.resolution_notes,
            },
            'comments': [
                {
                    'id': c.pk,
                    'author': (c.author.get_full_name() or c.author.username) if c.author else 'Unknown',
                    'body': c.body,
                    'is_internal': c.is_internal,
                    'created_at': c.created_at.strftime('%d %b %Y, %H:%M'),
                }
                for c in comments
            ],
        })


@method_decorator(login_required, name='dispatch')
class TicketCommentAPIView(_TicketAccessMixin, View):
    """Add a comment to a ticket's thread. Only staff may mark a comment internal."""

    def post(self, request: HttpRequest, pk: int) -> JsonResponse:
        ticket = self._get_visible_ticket(request, pk)
        body = request.POST.get('body', '').strip()
        if not body:
            return JsonResponse({'ok': False, 'error': 'Comment cannot be empty.'}, status=400)

        is_internal = request.user.is_staff and request.POST.get('is_internal') == 'true'
        comment = TicketComment.objects.create(
            ticket=ticket, author=request.user, body=body, is_internal=is_internal,
        )

        if not is_internal and ticket.created_by_id and ticket.created_by_id != request.user.pk:
            Notification.send(
                ticket.created_by,
                title=f'New reply on: {ticket.subject}',
                message=body[:200],
                notification_type='info',
                link='/core/support/',
            )
        elif request.user.pk != getattr(ticket.created_by, 'pk', None) and ticket.assigned_to_id and ticket.assigned_to_id != request.user.pk:
            Notification.send(
                ticket.assigned_to,
                title=f'New comment on: {ticket.subject}',
                message=body[:200],
                notification_type='info',
                link='/core/support/',
            )

        return JsonResponse({
            'ok': True,
            'comment': {
                'id': comment.pk,
                'author': comment.author.get_full_name() or comment.author.username,
                'body': comment.body,
                'is_internal': comment.is_internal,
                'created_at': comment.created_at.strftime('%d %b %Y, %H:%M'),
            },
        })


@method_decorator(login_required, name='dispatch')
class TicketStatusAPIView(StaffRequiredMixin, LoggingMixin, View):
    """Staff-only: change a ticket's status — powers the Kanban board's
    drag-and-drop columns and the ticket detail panel's status dropdown."""

    def post(self, request: HttpRequest, pk: int) -> JsonResponse:
        ticket = get_object_or_404(Ticket, pk=pk)
        new_status = request.POST.get('status', '').strip()
        if new_status not in dict(Ticket.STATUS_CHOICES):
            return JsonResponse({'ok': False, 'error': 'Unknown status.'}, status=400)

        assigned_to_id = request.POST.get('assigned_to_id', '').strip()
        if assigned_to_id:
            ticket.assigned_to_id = int(assigned_to_id) if assigned_to_id.isdigit() else None
            ticket.save(update_fields=['assigned_to'])

        old_status_display = ticket.get_status_display()
        ticket.set_status(new_status, actor=request.user)

        self.audit_log(
            'update', 'Ticket', resource_id=str(ticket.pk),
            detail=f'{old_status_display} -> {ticket.get_status_display()}',
        )
        return JsonResponse({'ok': True, 'status': ticket.status, 'status_display': ticket.get_status_display()})


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

def error_403(request: HttpRequest, exception=None) -> HttpResponse:
    """Render the custom 403 Access Denied page.

    Called by Django when a view raises :class:`~django.core.exceptions.PermissionDenied`.
    Returns a styled HTML page instead of the default Django error page.

    Args:
        request: The request that triggered the permission error.
        exception: The exception instance (may be ``None``).

    Returns:
        An HTTP 403 response with the ``403.html`` template.
    """
    return render(request, '403.html', {
        'sidebar_section': '',
        'page_title': 'Access Denied',
    }, status=403)


def error_404(request: HttpRequest, exception=None) -> HttpResponse:
    """Render the custom 404 Page Not Found page.

    Called by Django when no URL pattern matches the requested path.

    Args:
        request: The request for the missing page.
        exception: The exception instance (may be ``None``).

    Returns:
        An HTTP 404 response with the ``404.html`` template.
    """
    return render(request, '404.html', {
        'sidebar_section': '',
        'page_title': 'Page Not Found',
    }, status=404)


def error_500(request: HttpRequest) -> HttpResponse:
    """Render the custom 500 Server Error page.

    Called by Django when an unhandled exception propagates out of a view.
    Does not accept an exception argument because Django 500 handlers are
    called after the exception has already been logged.

    Args:
        request: The request that caused the server error.

    Returns:
        An HTTP 500 response with the ``500.html`` template.
    """
    return render(request, '500.html', {
        'sidebar_section': '',
        'page_title': 'Server Error',
    }, status=500)
