"""Warehouse Studio — HTML shell for the single-page app.

Every URL under ``/warehouse-studio/`` except ``api/`` serves the same shell;
the JS router picks the page.  Superusers only (the Studio sees account-wide
cost data and can run the optimizer).
"""

from django.contrib.auth.views import redirect_to_login
from django.urls import reverse
from django.views.generic import TemplateView

from core.mixins import SuperuserRequiredMixin

from .conf import get_config


class StudioAppView(SuperuserRequiredMixin, TemplateView):
    template_name = 'warehouse_studio/app.html'

    def handle_no_permission(self):
        # The shared mixin sets raise_exception, which would 403 anonymous visitors too.
        if not self.request.user.is_authenticated:
            return redirect_to_login(self.request.get_full_path())
        return super().handle_no_permission()

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        cfg = get_config()
        ctx.update({
            'sidebar_section': 'warehouse_studio',
            'studio_config': {
                'root': reverse('warehouse_studio:home'),
                'api': reverse('warehouse_studio:api_root'),
                'user': self.request.user.get_username(),
                'appName': 'Warehouse Studio',
                'configured': cfg.configured,
                'toolkit': cfg.toolkit,
                'creditPrice': cfg.credit_price_usd,
                'storagePrice': cfg.storage_price_per_tb_month_usd,
                'allowWrites': cfg.allow_writes,
            },
        })
        return ctx
