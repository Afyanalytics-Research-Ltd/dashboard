"""
Automation Studio (n8n) — HTML shell for the single-page app.

Every URL under ``/automations/`` that is not an API/data endpoint serves the
same shell; the JS router picks the page. Superusers only, like the proxy.
"""

from django.contrib.auth.views import redirect_to_login
from django.urls import reverse
from django.views.generic import TemplateView

from core.mixins import SuperuserRequiredMixin

from . import conf
from .routes import N8N_SPEC_VERSION, operation_count


class StudioAppView(SuperuserRequiredMixin, TemplateView):
    template_name = 'n8n_studio/app.html'

    def handle_no_permission(self):
        # The shared mixin sets raise_exception, which would 403 anonymous visitors too.
        if not self.request.user.is_authenticated:
            return redirect_to_login(self.request.get_full_path())
        return super().handle_no_permission()

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        root = reverse('n8n_studio:home')
        ctx.update({
            'sidebar_section': 'automations',
            'studio_config': {
                'root': root,
                'api': reverse('n8n_studio:proxy_root'),
                'studio': root + 'studio/',
                'n8nUrl': conf.public_url(),
                'webhookUrl': conf.webhook_url(),
                'user': self.request.user.get_username(),
                'appName': 'Automation Studio',
                'service': 'n8n',
                'configured': bool(conf.api_key()),
                'canRun': conf.login_configured(),
                'specVersion': N8N_SPEC_VERSION,
                'operations': operation_count(),
            },
        })
        return ctx
