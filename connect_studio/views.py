"""Connect Studio — HTML shell for the single-page app (superusers only)."""

from django.contrib.auth.views import redirect_to_login
from django.urls import reverse
from django.views.generic import TemplateView

from core.mixins import SuperuserRequiredMixin


class StudioAppView(SuperuserRequiredMixin, TemplateView):
    template_name = "connect_studio/app.html"

    def handle_no_permission(self):
        if not self.request.user.is_authenticated:
            return redirect_to_login(self.request.get_full_path())
        return super().handle_no_permission()

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx.update({
            "sidebar_section": "connect_studio",
            "studio_config": {
                "root": reverse("connect_studio:home"),
                "api": reverse("connect_studio:api_meta").rsplit("meta/", 1)[0],
                "user": self.request.user.get_username(),
                "appName": "Connect Studio",
                "service": "Connect Studio",
            },
        })
        return ctx
