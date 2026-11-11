"""Print the ROUTES table for routes.py from the running n8n's OpenAPI spec (after an n8n upgrade)."""

import requests
import yaml
from django.core.management.base import BaseCommand

from n8n_studio import conf


class Command(BaseCommand):
    help = "Print n8n_studio.routes.ROUTES regenerated from {N8N_API_URL}/api/v1/openapi.yml"

    def handle(self, *args, **options):
        spec = yaml.safe_load(requests.get(f'{conf.api_url()}/api/v1/openapi.yml', timeout=30).text)
        merged: dict[str, list[str]] = {}
        for path, ops in spec['paths'].items():
            key = path.replace('/workflows/{id}', '/workflows/{workflowId}') if path == '/workflows/{id}' else path
            methods = merged.setdefault(key, [])
            methods += [m.upper() for m in ('get', 'post', 'put', 'patch', 'delete') if m in ops and m.upper() not in methods]
        self.stdout.write(f"N8N_SPEC_VERSION = '{spec.get('info', {}).get('version', '?')}'\n")
        for path, methods in merged.items():
            inner = ', '.join(repr(m) for m in methods) + (',' if len(methods) == 1 else '')
            self.stdout.write(f"    '{path}': ({inner}),")
