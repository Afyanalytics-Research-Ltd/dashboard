"""
Model Studio (Weights & Biases) — HTML shell for the single-page app.

Every URL under ``/models/`` except the API endpoints serves the same shell;
the JS router picks the page. Restricted to superusers, like the proxy.
"""

from django.contrib.auth.views import redirect_to_login
from django.urls import reverse
from django.views.generic import TemplateView

from core.mixins import SuperuserRequiredMixin

from .connection import browser_url, connection_for
from .proxy import connection_state, displayable_key_id


class ModelStudioView(SuperuserRequiredMixin, TemplateView):
    template_name = 'wandb_ui/app.html'

    def handle_no_permission(self):
        # The shared mixin sets raise_exception, which would 403 anonymous visitors too.
        if not self.request.user.is_authenticated:
            return redirect_to_login(self.request.get_full_path())
        return super().handle_no_permission()

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        conn = connection_for(self.request.user)      # the user's own W&B key, else the shared server key
        ctx.update({
            'sidebar_section': 'models',
            'models_config': {
                'root': reverse('wandb_ui:home'),
                'graphql': reverse('wandb_ui:graphql'),
                'file': reverse('wandb_ui:file'),
                'connectionApi': reverse('wandb_ui:connection'),
                'views': reverse('wandb_ui:views', kwargs={'entity': '_E_', 'project': '_P_'}),
                # '' when this platform's W&B is only reachable on the server itself (never a localhost link online)
                'wandbUrl': browser_url(conn.server, self.request),
                'user': self.request.user.get_username(),
                'appName': 'Model Studio',
                'service': 'W&B',
                'configured': conn.configured,
                # Non-secret id of the key Studio uses for this person, so the UI can label it "in use".
                'apiKeyId': displayable_key_id(conn.api_key),
                'connection': connection_state(self.request.user, self.request),
            },
        })
        return ctx
