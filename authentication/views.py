"""
Authentication views for Afya DataHub.

All views use class-based patterns where possible and follow these conventions:
- LoginRequiredMixin / login_required for protected views
- BreadcrumbMixin for consistent breadcrumb context
- LoggingMixin.audit_log() for important actions
- AuditLog.log() classmethod for audit trail entries
- logger = logging.getLogger(__name__) for application logging
"""

import logging

from django.contrib import messages
from django.contrib.auth import (
    authenticate,
    get_user_model,
    login,
    logout,
    update_session_auth_hash,
)
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.models import Group
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse, reverse_lazy
from django.views import View
from django.views.generic import ListView, TemplateView

from core.mixins import BreadcrumbMixin, LoggingMixin
from core.models import AuditLog, Notification
from .forms import LoginForm, PasswordChangeForm, ProfileUpdateForm, SignupForm
from .models import UserProfile
from .roles import DEFAULT_ROLE

logger = logging.getLogger(__name__)
User = get_user_model()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client_ip(request) -> str | None:
    """Extract real client IP, honouring X-Forwarded-For."""
    xff = request.META.get('HTTP_X_FORWARDED_FOR')
    if xff:
        return xff.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR')


# ---------------------------------------------------------------------------
# LandingView
# ---------------------------------------------------------------------------

class LandingView(TemplateView):
    """
    Public landing page shown to unauthenticated visitors.
    Authenticated users are redirected straight to analytics.
    """

    template_name = 'landing.html'

    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return redirect(reverse('analytics:home'))
        return super().dispatch(request, *args, **kwargs)


# ---------------------------------------------------------------------------
# LoginView
# ---------------------------------------------------------------------------

class LoginView(View):
    """
    Authenticates user, records login IP in profile, writes AuditLog entry,
    then redirects to analytics dashboard.
    """

    template_name = 'accounts/login.html'
    redirect_authenticated_to = reverse_lazy('analytics:home')

    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return redirect(self.redirect_authenticated_to)
        return super().dispatch(request, *args, **kwargs)

    def get(self, request):
        form = LoginForm(request)
        return render(request, self.template_name, {'form': form})

    def post(self, request):
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)

            # Record IP in profile
            ip = _get_client_ip(request)
            try:
                user.profile.last_login_ip = ip
                user.profile.save(update_fields=['last_login_ip', 'updated_at'])
            except Exception as exc:
                logger.warning('Could not update last_login_ip for user %s: %s', user.pk, exc)

            # Audit log
            AuditLog.log(
                user=user,
                action='login',
                resource='authentication',
                detail=f'User logged in from {ip}',
                ip_address=ip,
                user_agent=request.META.get('HTTP_USER_AGENT', ''),
            )

            logger.info('User %s logged in from %s', user.username, ip)

            # Honour ?next= param
            next_url = request.GET.get('next') or request.POST.get('next')
            if next_url and next_url.startswith('/'):
                return redirect(next_url)
            return redirect(self.redirect_authenticated_to)

        return render(request, self.template_name, {'form': form})


# ---------------------------------------------------------------------------
# LogoutView
# ---------------------------------------------------------------------------

class LogoutView(View):
    """Logs the user out, writes AuditLog, redirects to login page."""

    def post(self, request):
        if request.user.is_authenticated:
            AuditLog.log(
                user=request.user,
                action='logout',
                resource='authentication',
                detail='User logged out',
                ip_address=_get_client_ip(request),
                user_agent=request.META.get('HTTP_USER_AGENT', ''),
            )
            logger.info('User %s logged out', request.user.username)
            logout(request)

        return redirect(reverse('authentication:login'))

    # Allow GET-based logout (e.g. direct link) during development
    def get(self, request):
        return self.post(request)


# ---------------------------------------------------------------------------
# SignupView
# ---------------------------------------------------------------------------

class SignupView(View):
    """
    Creates a new User + UserProfile, assigns default Group role,
    writes AuditLog entry, then redirects to login with a success message.
    """

    template_name = 'accounts/signup.html'

    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return redirect(reverse('analytics:home'))
        return super().dispatch(request, *args, **kwargs)

    def get(self, request):
        form = SignupForm()
        return render(request, self.template_name, {'form': form})

    def post(self, request):
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save()

            # Save profile fields from form
            phone = form.cleaned_data.get('phone_number', '')
            job_title = form.cleaned_data.get('job_title', '')
            try:
                profile = user.profile
                profile.phone_number = phone
                profile.job_title = job_title
                profile.role = DEFAULT_ROLE
                profile.save(update_fields=['phone_number', 'job_title', 'role', 'updated_at'])
            except Exception as exc:
                logger.error('Failed to update profile for new user %s: %s', user.pk, exc)

            # Add user to the default Django Group
            try:
                group, _ = Group.objects.get_or_create(name=DEFAULT_ROLE)
                user.groups.add(group)
            except Exception as exc:
                logger.warning('Could not assign group to user %s: %s', user.pk, exc)

            # Audit log
            AuditLog.log(
                user=user,
                action='create',
                resource='authentication.user',
                resource_id=str(user.pk),
                detail=f'New user registered: {user.username}',
                ip_address=_get_client_ip(request),
                user_agent=request.META.get('HTTP_USER_AGENT', ''),
            )

            logger.info('New user registered: %s (pk=%s)', user.username, user.pk)
            messages.success(request, 'Account created successfully. Please log in.')
            return redirect(reverse('authentication:login'))

        return render(request, self.template_name, {'form': form})


# ---------------------------------------------------------------------------
# ProfileView
# ---------------------------------------------------------------------------

class ProfileView(LoginRequiredMixin, BreadcrumbMixin, LoggingMixin, View):
    """
    Displays and handles updates to the current user's profile.
    Handles avatar upload via multipart/form-data.
    """

    template_name = 'accounts/profile.html'
    login_url = reverse_lazy('authentication:login')
    sidebar_section = 'profile'

    def get_breadcrumbs(self):
        return [
            {'label': 'Home', 'url': reverse('analytics:home')},
            {'label': 'Profile', 'url': None},
        ]

    def _context(self, request, form=None):
        ctx = {
            'form': form or ProfileUpdateForm(user=request.user),
            'profile': request.user.profile,
            'sidebar_section': self.sidebar_section,
            'breadcrumbs': self.get_breadcrumbs(),
        }
        return ctx

    def get(self, request):
        return render(request, self.template_name, self._context(request))

    def post(self, request):
        form = ProfileUpdateForm(user=request.user, data=request.POST, files=request.FILES)
        if form.is_valid():
            form.save()
            self.audit_log(
                action='update',
                resource='authentication.userprofile',
                resource_id=str(request.user.pk),
                detail='User updated their profile',
            )
            logger.info('User %s updated their profile', request.user.username)
            messages.success(request, 'Profile updated successfully.')
            return redirect(reverse('authentication:profile'))

        messages.error(request, 'Please correct the errors below.')
        return render(request, self.template_name, self._context(request, form))


# ---------------------------------------------------------------------------
# PasswordChangeView
# ---------------------------------------------------------------------------

class PasswordChangeView(LoginRequiredMixin, BreadcrumbMixin, LoggingMixin, View):
    """Handles password change for authenticated users."""

    template_name = 'accounts/password_change.html'
    login_url = reverse_lazy('authentication:login')

    def get_breadcrumbs(self):
        return [
            {'label': 'Home', 'url': reverse('analytics:home')},
            {'label': 'Profile', 'url': reverse('authentication:profile')},
            {'label': 'Change Password', 'url': None},
        ]

    def _context(self, form=None):
        return {
            'form': form or PasswordChangeForm(self.request.user),
            'breadcrumbs': self.get_breadcrumbs(),
            'sidebar_section': 'profile',
        }

    def get(self, request):
        return render(request, self.template_name, self._context())

    def post(self, request):
        form = PasswordChangeForm(request.user, data=request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)

            self.audit_log(
                action='update',
                resource='authentication.password',
                resource_id=str(request.user.pk),
                detail='User changed their password',
            )
            logger.info('User %s changed their password', request.user.username)
            messages.success(request, 'Password changed successfully.')
            return redirect(reverse('authentication:profile'))

        return render(request, self.template_name, self._context(form))


# ---------------------------------------------------------------------------
# MarkNotificationReadView
# ---------------------------------------------------------------------------

class MarkNotificationReadView(LoginRequiredMixin, View):
    """
    AJAX POST endpoint. Marks a single notification as read.
    Returns JSON: {success: true} or error dict with status 400/404.
    """

    login_url = reverse_lazy('authentication:login')

    def post(self, request, pk):
        notification = get_object_or_404(Notification, pk=pk, user=request.user)
        notification.mark_read()
        logger.debug('Notification %s marked read by user %s', pk, request.user.username)
        return JsonResponse({'success': True, 'id': pk})

    def http_method_not_allowed(self, request, *args, **kwargs):
        return JsonResponse({'error': 'Method not allowed.'}, status=405)


# ---------------------------------------------------------------------------
# NotificationsListView
# ---------------------------------------------------------------------------

class NotificationsListView(LoginRequiredMixin, BreadcrumbMixin, ListView):
    """
    Paginated list of all notifications for the current user.
    Supports filtering by type and read status via query params.
    """

    template_name = 'accounts/notifications.html'
    context_object_name = 'notifications'
    paginate_by = 20
    login_url = reverse_lazy('authentication:login')

    def get_breadcrumbs(self):
        return [
            {'label': 'Home', 'url': reverse('analytics:home')},
            {'label': 'Notifications', 'url': None},
        ]

    def get_queryset(self):
        qs = Notification.objects.filter(user=self.request.user).order_by('-created_at')
        tab = self.request.GET.get('tab', 'all')
        if tab == 'unread':
            qs = qs.filter(is_read=False)
        elif tab in ('success', 'warning', 'danger', 'info'):
            qs = qs.filter(notification_type=tab)
        q = self.request.GET.get('q', '').strip()
        if q:
            qs = qs.filter(title__icontains=q) | qs.filter(message__icontains=q)
            qs = qs.distinct()
        return qs

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx['active_tab'] = self.request.GET.get('tab', 'all')
        ctx['search_query'] = self.request.GET.get('q', '')
        ctx['unread_count'] = Notification.objects.filter(
            user=self.request.user, is_read=False
        ).count()
        ctx['sidebar_section'] = 'profile'
        return ctx

    def post(self, request):
        """Mark all as read."""
        Notification.objects.filter(user=request.user, is_read=False).update(is_read=True)
        messages.success(request, 'All notifications marked as read.')
        return redirect(reverse('authentication:notifications'))


# ---------------------------------------------------------------------------
# UserActivityView
# ---------------------------------------------------------------------------

class UserActivityView(LoginRequiredMixin, BreadcrumbMixin, ListView):
    """
    Paginated audit-log view showing the current user's activity history.
    Supports search by action/resource.
    """

    template_name = 'accounts/activity.html'
    context_object_name = 'audit_logs'
    paginate_by = 25
    login_url = reverse_lazy('authentication:login')

    def get_breadcrumbs(self):
        return [
            {'label': 'Home', 'url': reverse('analytics:home')},
            {'label': 'My Activity', 'url': None},
        ]

    def get_queryset(self):
        qs = AuditLog.objects.filter(user=self.request.user).order_by('-timestamp')
        q = self.request.GET.get('q', '').strip()
        if q:
            qs = qs.filter(action__icontains=q) | qs.filter(resource__icontains=q)
            qs = qs.distinct()
        action = self.request.GET.get('action', '').strip()
        if action:
            qs = qs.filter(action=action)
        return qs

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx['search_query'] = self.request.GET.get('q', '')
        ctx['active_action'] = self.request.GET.get('action', '')
        ctx['action_choices'] = AuditLog.ACTION_CHOICES
        ctx['sidebar_section'] = 'profile'
        return ctx
