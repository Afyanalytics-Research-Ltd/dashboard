"""
Airflow pipeline UI — HTML shell for the single-page app.

All rendering happens client-side (``static/airflow/``); every URL under
``/pipelines/`` (except ``/pipelines/api/``) serves the same shell and the
JS router picks the page. Views are restricted to superusers.
"""

from django.conf import settings
from django.contrib.auth.views import redirect_to_login
from django.urls import reverse
from django.views.generic import TemplateView

from core.mixins import SuperuserRequiredMixin


class PipelineAppView(SuperuserRequiredMixin, TemplateView):
    template_name = 'airflow/app.html'

    def handle_no_permission(self):
        # The shared mixin sets raise_exception, which would 403 anonymous
        # visitors too; send them to sign in so deep links survive login.
        if not self.request.user.is_authenticated:
            return redirect_to_login(self.request.get_full_path())
        return super().handle_no_permission()

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        public_url = getattr(settings, 'AIRFLOW_PUBLIC_URL', '') or settings.AIRFLOW_BASE_URL
        ctx.update({
            'sidebar_section': 'pipelines',
            'airflow_config': {
                'root': reverse('airflow:dashboard'),
                'api': reverse('airflow:api_root'),
                'spec': reverse('airflow:api_spec'),
                'airflowUrl': public_url.rstrip('/'),
                'user': self.request.user.get_username(),
            },
        })
        return ctx
