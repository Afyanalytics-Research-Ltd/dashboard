"""
Analytics Studio (Redash) — HTML shell for the single-page app.

Every URL under ``/studio/`` except ``/studio/proxy/`` serves the same shell;
the JS router picks the page. Restricted to superusers, like the proxy.
"""

from django.conf import settings
from django.contrib.auth.views import redirect_to_login
from django.urls import reverse
from django.views.generic import TemplateView

from core.mixins import SuperuserRequiredMixin


class StudioAppView(SuperuserRequiredMixin, TemplateView):
    template_name = 'redash_ui/app.html'

    def handle_no_permission(self):
        # The shared mixin sets raise_exception, which would 403 anonymous visitors too.
        if not self.request.user.is_authenticated:
            return redirect_to_login(self.request.get_full_path())
        return super().handle_no_permission()

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        public_url = getattr(settings, 'REDASH_PUBLIC_URL', '') or settings.REDASH_BASE_URL
        ctx.update({
            'sidebar_section': 'studio',
            'studio_config': {
                'root': reverse('redash_ui:home'),
                'api': reverse('redash_ui:proxy_root'),
                'history': reverse('redash_ui:query_history', kwargs={'query_id': 0}).rsplit('0/', 1)[0],
                'redashUrl': public_url.rstrip('/'),
                'user': self.request.user.get_username(),
                'appName': 'Analytics Studio',
                'service': 'Redash',
                'configured': bool(settings.REDASH_ADMIN_API_KEY),
            },
        })
        return ctx
