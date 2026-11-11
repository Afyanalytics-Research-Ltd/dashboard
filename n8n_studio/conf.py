"""
Automation Studio settings, read lazily from Django settings / the environment.

    N8N_API_URL         server-side n8n base URL (inside docker: http://n8n-container:5678;
                        add the N8N_PATH prefix if n8n runs under one, e.g. .../n8n)
    N8N_API_KEY         public-API key (n8n → Settings → n8n API). Never sent to the browser.
    N8N_PUBLIC_URL      browser-facing n8n editor URL, for "open in n8n" links
    N8N_WEBHOOK_URL     public webhook base URL shown in the webhook panel
    N8N_TIMEOUT         seconds per upstream request (default 60)
    N8N_STUDIO_EMAIL /  optional n8n login used for the two things the public API
    N8N_STUDIO_PASSWORD cannot do: manual runs and the live node-type catalog.
"""

import os

from django.conf import settings


def _get(name: str, default: str = '') -> str:
    value = getattr(settings, name, None)
    if value is None:
        value = os.getenv(name, default)
    return str(value).strip().strip('"')


def api_url() -> str:
    return _get('N8N_API_URL', 'http://localhost:5679').rstrip('/')


def api_key() -> str:
    return _get('N8N_API_KEY')


def public_url() -> str:
    return (_get('N8N_PUBLIC_URL') or _get('N8N_EDITOR_BASE_URL') or api_url()).rstrip('/')


def webhook_url() -> str:
    return (_get('N8N_WEBHOOK_URL') or public_url()).rstrip('/')


def timeout() -> float:
    try:
        return float(_get('N8N_TIMEOUT', '60') or 60)
    except ValueError:
        return 60.0


def login_email() -> str:
    return _get('N8N_STUDIO_EMAIL')


def login_password() -> str:
    return _get('N8N_STUDIO_PASSWORD')


def login_configured() -> bool:
    return bool(login_email() and login_password())
