"""Browser-automation console — a native frontend for Browserbase sessions.

Mirrors Browserbase's own dashboard: a session list, a "New Session"
action, and a live-view page (iframe embedding Browserbase's own
debugger_fullscreen_url) with an End Session action. All access is
superuser-only since it spends the org's Browserbase usage/quota.
"""

import asyncio
import json
import logging
import uuid

from django.conf import settings
from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse
from django.utils import timezone
from django.views import View
from django.views.generic import TemplateView
from django_celery_beat.models import CrontabSchedule, PeriodicTask

from core.mixins import BreadcrumbMixin, SuperuserRequiredMixin
from core.models import AuditLog

from . import mcp_client, services
from .forms import ScheduledBrowserTaskForm
from .mcp_client import BrowserMCPError
from .models import ScheduledBrowserTask
from .services import BrowserbaseError
from .tasks import run_scheduled_browser_task

logger = logging.getLogger(__name__)


class ConsoleView(SuperuserRequiredMixin, BreadcrumbMixin, TemplateView):
    """Session list + "New Session" entry point."""

    template_name = "browser_automation/console.html"

    def get_breadcrumbs(self):
        return [
            {"label": "Home", "url": reverse("analytics:home")},
            {"label": "Browser Automation", "url": None},
        ]

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx["sidebar_section"] = "browserbase"
        ctx["configured"] = bool(
            settings.BROWSERBASE_API_KEY and settings.BROWSERBASE_PROJECT_ID
        )

        status_filter = self.request.GET.get("status", "").strip().upper() or None
        sessions, error = [], None
        if ctx["configured"]:
            try:
                sessions = services.list_sessions(status=status_filter)
            except BrowserbaseError as exc:
                error = str(exc)

        ctx.update({
            "sessions": sessions,
            "error": error,
            "current_status": status_filter or "",
            "status_choices": ["RUNNING", "PENDING", "ERROR", "TIMED_OUT", "COMPLETED"],
        })
        return ctx


class CreateSessionView(SuperuserRequiredMixin, View):
    """POST-only: start a new Browserbase session, then jump to its live view."""

    def post(self, request):
        region = request.POST.get("region", "").strip() or None
        keep_alive = request.POST.get("keep_alive") == "on"
        use_proxy = request.POST.get("use_proxy") == "on"

        try:
            session = services.create_session(region=region, keep_alive=keep_alive, use_proxy=use_proxy)
        except BrowserbaseError as exc:
            messages.error(request, f"Could not start a Browserbase session: {exc}")
            return redirect("browserbase:console")

        AuditLog.log(
            user=request.user,
            action="create",
            resource="Browserbase Session",
            resource_id=session.get("id", ""),
            detail=f"Started session in {session.get('region', 'default region')}",
            ip_address=request.META.get("REMOTE_ADDR"),
        )
        messages.success(request, "Session started.")
        return redirect("browserbase:session_detail", session_id=session["id"])


class SessionDetailView(SuperuserRequiredMixin, BreadcrumbMixin, TemplateView):
    """Live view of a single session (iframe) + its metadata."""

    template_name = "browser_automation/session_detail.html"

    def get_breadcrumbs(self):
        return [
            {"label": "Home", "url": reverse("analytics:home")},
            {"label": "Browser Automation", "url": reverse("browserbase:console")},
            {"label": self.kwargs["session_id"][:8], "url": None},
        ]

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx["sidebar_section"] = "browserbase"
        session_id = self.kwargs["session_id"]

        try:
            session = services.get_session(session_id)
        except BrowserbaseError as exc:
            ctx.update({"session": None, "error": str(exc)})
            return ctx

        live_urls = None
        live_error = None
        if session.get("status") == "RUNNING":
            try:
                live_urls = services.get_live_urls(session_id)
            except BrowserbaseError as exc:
                live_error = str(exc)

        ctx.update({
            "session": session,
            "live_urls": live_urls,
            "live_error": live_error,
            "error": None,
        })
        return ctx


class EndSessionView(SuperuserRequiredMixin, View):
    """POST-only: request Browserbase release the session."""

    def post(self, request, session_id):
        try:
            services.end_session(session_id)
        except BrowserbaseError as exc:
            messages.error(request, f"Could not end session: {exc}")
            return redirect("browserbase:session_detail", session_id=session_id)

        AuditLog.log(
            user=request.user,
            action="update",
            resource="Browserbase Session",
            resource_id=session_id,
            detail="Ended session",
            ip_address=request.META.get("REMOTE_ADDR"),
        )
        messages.success(request, "Session ended.")
        return redirect("browserbase:console")


class RunMCPTaskView(SuperuserRequiredMixin, View):
    """POST-only: run a one-off agent-tool task via the mcp-browserbase
    server (start a session, navigate, end it) — see mcp_client.py.

    This is a separate Browserbase session lifecycle from the one
    services.py manages: it's driven through MCP tool calls the way an LLM
    agent would use them, not the direct SDK the console's own session
    list/live-view uses.
    """

    def post(self, request):
        url = request.POST.get("url", "").strip()
        if not url:
            messages.error(request, "Enter a URL to navigate to.")
            return redirect("browserbase:console")

        try:
            result = asyncio.run(mcp_client.navigate(url))
        except BrowserMCPError as exc:
            messages.error(request, f"Agent browser task failed: {exc}")
            return redirect("browserbase:console")
        except Exception as exc:
            logger.error("mcp-browserbase task failed: %s", exc)
            messages.error(request, f"Could not reach the agent browser tool server: {exc}")
            return redirect("browserbase:console")

        AuditLog.log(
            user=request.user,
            action="trigger",
            resource="Browserbase MCP Task",
            detail=f"navigate({url}) -> {result}",
            ip_address=request.META.get("REMOTE_ADDR"),
        )
        messages.success(request, f"Agent navigated to {url} and ended the session.")
        return redirect("browserbase:console")


class ScheduledTaskListView(SuperuserRequiredMixin, BreadcrumbMixin, TemplateView):
    """List this user's scheduled browser tasks + the create form."""

    template_name = "browser_automation/scheduled_tasks.html"

    def get_breadcrumbs(self):
        return [
            {"label": "Home", "url": reverse("analytics:home")},
            {"label": "Browser Automation", "url": reverse("browserbase:console")},
            {"label": "Scheduled Tasks", "url": None},
        ]

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx["sidebar_section"] = "browserbase"
        ctx["tasks"] = (
            ScheduledBrowserTask.objects
            .filter(user=self.request.user)
            .select_related("periodic_task__crontab")
        )
        ctx["form"] = ScheduledBrowserTaskForm()
        return ctx


class CreateScheduledTaskView(SuperuserRequiredMixin, View):
    """POST-only: create a ScheduledBrowserTask + its backing PeriodicTask/CrontabSchedule."""

    def post(self, request):
        form = ScheduledBrowserTaskForm(request.POST)
        if not form.is_valid():
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field}: {error}")
            return redirect("browserbase:scheduled_tasks")

        data = form.cleaned_data
        crontab, _ = CrontabSchedule.objects.get_or_create(
            minute=str(data["minute"]),
            hour=str(data["hour"]),
            day_of_week="*",
            day_of_month="*",
            month_of_year="*",
            timezone=timezone.get_current_timezone_name(),
        )
        # PeriodicTask needs to exist before ScheduledBrowserTask (its
        # OneToOneField target), but its `kwargs` needs the ScheduledBrowserTask's
        # own pk — created with a placeholder, then patched once that pk exists.
        periodic_task = PeriodicTask.objects.create(
            name=f"browser-task-{uuid.uuid4()}",
            task="browser_automation.tasks.run_scheduled_browser_task",
            crontab=crontab,
            enabled=data["is_active"],
        )
        task_obj = ScheduledBrowserTask.objects.create(
            user=request.user,
            name=data["name"],
            instructions=data["instructions"],
            is_active=data["is_active"],
            periodic_task=periodic_task,
        )
        periodic_task.kwargs = json.dumps({"scheduled_task_id": task_obj.pk})
        periodic_task.save(update_fields=["kwargs"])

        AuditLog.log(
            user=request.user,
            action="create",
            resource="Scheduled Browser Task",
            resource_id=str(task_obj.pk),
            detail=f"'{task_obj.name}' at {data['hour']:02d}:{data['minute']:02d} daily",
            ip_address=request.META.get("REMOTE_ADDR"),
        )
        messages.success(request, f"Scheduled '{task_obj.name}' for {data['hour']:02d}:{data['minute']:02d} daily.")
        return redirect("browserbase:scheduled_tasks")


class UpdateScheduledTaskView(SuperuserRequiredMixin, View):
    """POST-only: edit an existing scheduled task's instructions/schedule/active state."""

    def post(self, request, pk):
        task_obj = get_object_or_404(ScheduledBrowserTask, pk=pk, user=request.user)
        form = ScheduledBrowserTaskForm(request.POST)
        if not form.is_valid():
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field}: {error}")
            return redirect("browserbase:scheduled_tasks")

        data = form.cleaned_data
        crontab, _ = CrontabSchedule.objects.get_or_create(
            minute=str(data["minute"]),
            hour=str(data["hour"]),
            day_of_week="*",
            day_of_month="*",
            month_of_year="*",
            timezone=timezone.get_current_timezone_name(),
        )
        task_obj.name = data["name"]
        task_obj.instructions = data["instructions"]
        task_obj.is_active = data["is_active"]
        task_obj.save(update_fields=["name", "instructions", "is_active", "updated_at"])

        periodic_task = task_obj.periodic_task
        periodic_task.crontab = crontab
        periodic_task.enabled = data["is_active"]
        periodic_task.save(update_fields=["crontab", "enabled"])

        AuditLog.log(
            user=request.user,
            action="update",
            resource="Scheduled Browser Task",
            resource_id=str(task_obj.pk),
            detail=f"'{task_obj.name}' at {data['hour']:02d}:{data['minute']:02d} daily",
            ip_address=request.META.get("REMOTE_ADDR"),
        )
        messages.success(request, f"Updated '{task_obj.name}'.")
        return redirect("browserbase:scheduled_tasks")


class DeleteScheduledTaskView(SuperuserRequiredMixin, View):
    """POST-only: delete a scheduled task and its backing PeriodicTask."""

    def post(self, request, pk):
        task_obj = get_object_or_404(ScheduledBrowserTask, pk=pk, user=request.user)
        name = task_obj.name
        periodic_task = task_obj.periodic_task
        task_obj.delete()
        periodic_task.delete()

        AuditLog.log(
            user=request.user,
            action="delete",
            resource="Scheduled Browser Task",
            resource_id=str(pk),
            detail=f"Deleted '{name}'",
            ip_address=request.META.get("REMOTE_ADDR"),
        )
        messages.success(request, f"Deleted '{name}'.")
        return redirect("browserbase:scheduled_tasks")


class RunScheduledTaskNowView(SuperuserRequiredMixin, View):
    """POST-only: queue an immediate run, outside its normal schedule (for testing)."""

    def post(self, request, pk):
        task_obj = get_object_or_404(ScheduledBrowserTask, pk=pk, user=request.user)
        run_scheduled_browser_task.delay(task_obj.pk)

        AuditLog.log(
            user=request.user,
            action="trigger",
            resource="Scheduled Browser Task",
            resource_id=str(task_obj.pk),
            detail=f"Manually ran '{task_obj.name}'",
            ip_address=request.META.get("REMOTE_ADDR"),
        )
        messages.success(request, f"Queued '{task_obj.name}' to run now — refresh in a moment to see the result.")
        return redirect("browserbase:scheduled_tasks")
