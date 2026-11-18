"""
Which W&B a Model Studio request talks to, and with which key.

Every user may save their own W&B API key (Model Studio ▸ W&B connection) for one of two servers:

  local   this platform's self-hosted W&B Server (WANDB_BASE_URL / WANDB_HOST_URL)
  cloud   Weights & Biases Cloud (api.wandb.ai / wandb.ai)

Without one, requests use the shared server key (WANDB_API_KEY) on the local server, as before. Only these two
servers can be chosen — never a free-form URL — so a saved connection can't point the dashboard's server-side
requests at an arbitrary host.

Keys are stored encrypted (Fernet; MODEL_STUDIO_KEY, else derived from SECRET_KEY) and never sent to the browser.
"""

from __future__ import annotations

import base64
import hashlib
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass

from cryptography.fernet import Fernet, InvalidToken
from django.conf import settings

from .client import WandbClient, WandbError


@dataclass(frozen=True)
class Server:
    key: str
    label: str
    api_url: str        # GraphQL + files, server-side
    host_url: str       # where people open W&B in a browser
    key_page: str       # where to get an API key


def servers() -> dict[str, Server]:
    local_api = settings.WANDB_BASE_URL.rstrip('/')
    local_host = (getattr(settings, 'WANDB_HOST_URL', '') or local_api).rstrip('/')
    cloud_api = (os.getenv('WANDB_CLOUD_API_URL') or 'https://api.wandb.ai').rstrip('/')
    cloud_host = (os.getenv('WANDB_CLOUD_HOST_URL') or 'https://wandb.ai').rstrip('/')
    return {
        'local': Server('local', "This platform's W&B server", local_api, local_host, f'{local_host}/authorize'),
        'cloud': Server('cloud', 'W&B Cloud (wandb.ai)', cloud_api, cloud_host, f'{cloud_host}/authorize'),
    }


@dataclass(frozen=True)
class Connection:
    server: Server
    api_key: str
    source: str                 # 'own' (the user's saved key) | 'shared' (WANDB_API_KEY) | 'none'
    wandb_user: str = ''

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


# ------------------------------------------------------------------ secrets at rest

def _fernet() -> Fernet:
    raw = os.getenv('MODEL_STUDIO_KEY') or f'model-studio::{settings.SECRET_KEY}'
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(raw.encode()).digest()))


def encrypt_key(api_key: str) -> str:
    return _fernet().encrypt(api_key.encode()).decode()


def decrypt_key(token: str) -> str:
    try:
        return _fernet().decrypt(token.encode()).decode()
    except (InvalidToken, ValueError):
        return ''


def key_hint(api_key: str) -> str:
    """Safe to show: a new-style key's public id, else the last 4 characters."""
    from .proxy import displayable_key_id
    return displayable_key_id(api_key) or (f'••••{api_key[-4:]}' if len(api_key) >= 8 else '••••')


# ------------------------------------------------------------------ resolving a request

def shared_connection() -> Connection:
    key = settings.WANDB_API_KEY
    return Connection(servers()['local'], key, 'shared' if key else 'none')


def connection_for(user) -> Connection:
    cred = getattr(user, 'wandb_credential', None) if getattr(user, 'is_authenticated', False) else None
    if cred is not None:
        key = decrypt_key(cred.api_key_enc)
        server = servers().get(cred.server)
        if key and server:
            return Connection(server, key, 'own', cred.wandb_user)
    return shared_connection()


_clients: OrderedDict[str, WandbClient] = OrderedDict()
_lock = threading.Lock()
MAX_CLIENTS = 64


def client_for(conn: Connection) -> WandbClient:
    """One HTTP client per (server, key) — reused across requests, a few dozen at most."""
    if conn.source == 'shared':
        from .client import get_client
        return get_client()
    ident = hashlib.sha256(f'{conn.server.api_url}\0{conn.api_key}'.encode()).hexdigest()
    with _lock:
        c = _clients.get(ident)
        if c is None:
            c = WandbClient(base_url=conn.server.api_url, api_key=conn.api_key, timeout=getattr(settings, 'WANDB_TIMEOUT', 60))
            _clients[ident] = c
            while len(_clients) > MAX_CLIENTS:
                _clients.popitem(last=False)
        else:
            _clients.move_to_end(ident)
        return c


def verify(server_key: str, api_key: str) -> dict:
    """Ask the chosen W&B who this key belongs to. Raises WandbError with a message for people."""
    server = servers().get(server_key)
    if not server:
        raise WandbError('Choose a W&B server.', status_code=400)
    api_key = (api_key or '').strip()
    if len(api_key) < 20 or any(ch.isspace() for ch in api_key):
        raise WandbError('That does not look like a W&B API key.', status_code=400)
    client = WandbClient(base_url=server.api_url, api_key=api_key, timeout=20, max_retries=1)
    try:
        resp = client.graphql(b'{"query": "{ viewer { id username entity email } }"}', read_only=True)
    except WandbError as exc:
        raise WandbError(f'Could not reach {server.label}: {exc}', status_code=502) from None
    payload = resp.json() or {}
    viewer = (payload.get('data') or {}).get('viewer')
    if resp.status_code in (401, 403) or not viewer:
        raise WandbError(f'{server.label} rejected this API key. Copy it again from {server.key_page}.', status_code=400)
    return {'id': viewer.get('id'), 'username': viewer.get('username') or '', 'entity': viewer.get('entity') or '',
            'email': viewer.get('email') or ''}
