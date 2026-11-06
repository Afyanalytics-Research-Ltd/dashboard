"""
HTTP clients for n8n.

``N8nClient`` talks to the public API (``/api/v1``) with ``X-N8N-API-KEY``.
``N8nSession`` logs in with an n8n user (optional) for the internal endpoints the
public API lacks: manual runs (``/rest/workflows/{id}/run``) and the node-type
catalog (``/types/nodes.json``). Both retry idempotent requests on connection
errors and 502/503/504, and return 4xx/5xx responses instead of raising.
"""

from __future__ import annotations

import json as jsonlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from . import conf

logger = logging.getLogger(__name__)

IDEMPOTENT_METHODS = {'GET', 'HEAD', 'OPTIONS', 'PUT', 'DELETE'}
RETRY_STATUSES = {502, 503, 504}
PASSTHROUGH_HEADERS = ('content-disposition',)


class N8nError(Exception):
    """n8n cannot be reached, is not configured, or rejected our credentials."""

    def __init__(self, message: str, status_code: int = 502, code: str = 'upstream'):
        super().__init__(message)
        self.status_code = status_code
        self.payload = {'message': message, 'code': code}


@dataclass
class N8nResponse:
    status_code: int
    content: bytes
    content_type: str = 'application/json'
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def json(self) -> Any:
        return jsonlib.loads(self.content) if self.content else None


def _send(session: requests.Session, method: str, url: str, *, max_retries: int, timeout: float,
          label: str, **kwargs) -> requests.Response:
    attempt = 0
    while True:
        try:
            resp = session.request(method, url, timeout=timeout, allow_redirects=False, **kwargs)
        except requests.RequestException as exc:
            if method in IDEMPOTENT_METHODS and attempt < max_retries:
                attempt += 1
                time.sleep(0.4 * attempt)
                continue
            raise N8nError(f'Cannot reach n8n at {label}: {exc}', code='unreachable') from exc
        if resp.status_code in RETRY_STATUSES and method in IDEMPOTENT_METHODS and attempt < max_retries:
            attempt += 1
            time.sleep(0.4 * attempt)
            continue
        return resp


def _wrap(resp: requests.Response) -> N8nResponse:
    return N8nResponse(
        status_code=resp.status_code,
        content=resp.content,
        content_type=resp.headers.get('Content-Type', 'application/json'),
        headers={k: v for k, v in resp.headers.items() if k.lower() in PASSTHROUGH_HEADERS},
    )


class N8nClient:
    """Public API client. Paths are relative to ``/api/v1``."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 60, max_retries: int = 2):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def request(self, method: str, path: str, *, params: Any = None, data: bytes | None = None,
                json: Any = None, content_type: str | None = None, files: Any = None) -> N8nResponse:
        if not self.api_key:
            raise N8nError('N8N_API_KEY is not configured on the server.', status_code=503, code='not_configured')
        method = method.upper()
        headers = {'X-N8N-API-KEY': self.api_key, 'Accept': 'application/json, application/gzip, */*'}
        if content_type and files is None:
            headers['Content-Type'] = content_type
        elif (data is not None or json is not None) and files is None:
            headers['Content-Type'] = 'application/json'
        url = f'{self.base_url}/api/v1{path}'
        resp = _send(self.session, method, url, params=params, data=data, json=json, files=files,
                     headers=headers, max_retries=self.max_retries, timeout=self.timeout, label=self.base_url)
        logger.debug('n8n %s %s -> %s', method, path, resp.status_code)
        if resp.status_code == 401:
            # n8n answers 401 for a missing/expired/revoked key on every route.
            raise N8nError('n8n rejected N8N_API_KEY (expired, revoked or wrong instance).', status_code=502, code='bad_key')
        return _wrap(resp)

    def get_json(self, path: str, **params) -> Any:
        resp = self.request('GET', path, params=params or None)
        if not resp.ok:
            raise N8nError(f'n8n returned {resp.status_code} for {path}', status_code=resp.status_code)
        return resp.json()

    def iterate(self, path: str, *, limit: int = 250, max_items: int = 10000, **params):
        """Yield every item of a cursor-paginated list (``{data, nextCursor}``)."""
        cursor, seen = None, 0
        while True:
            page = self.get_json(path, limit=limit, cursor=cursor, **params) if cursor else self.get_json(path, limit=limit, **params)
            for item in page.get('data') or []:
                yield item
                seen += 1
                if seen >= max_items:
                    return
            cursor = page.get('nextCursor')
            if not cursor:
                return

    def health(self) -> dict:
        """Liveness of the instance itself (no key needed)."""
        try:
            resp = _send(self.session, 'GET', f'{self.base_url}/healthz', max_retries=0,
                         timeout=min(self.timeout, 5), label=self.base_url)
            return {'reachable': True, 'ok': resp.ok, 'status': resp.status_code}
        except N8nError as exc:
            return {'reachable': False, 'ok': False, 'error': str(exc)}


class N8nSession:
    """Cookie session for n8n's internal REST API, logging in lazily (and again on 401)."""

    def __init__(self, base_url: str, email: str, password: str, timeout: float = 60):
        self.base_url = base_url.rstrip('/')
        self.email = email
        self.password = password
        self.timeout = timeout
        self.session = requests.Session()
        self._logged_in = False
        self._lock = threading.Lock()

    @property
    def configured(self) -> bool:
        return bool(self.email and self.password)

    def login(self) -> None:
        if not self.configured:
            raise N8nError('Set N8N_STUDIO_EMAIL and N8N_STUDIO_PASSWORD to enable manual runs.',
                           status_code=409, code='login_not_configured')
        resp = _send(self.session, 'POST', f'{self.base_url}/rest/login', max_retries=1, timeout=self.timeout,
                     label=self.base_url, json={'emailOrLdapLoginId': self.email, 'password': self.password})
        if resp.status_code != 200:
            self._logged_in = False
            raise N8nError(f'n8n login for {self.email} failed ({resp.status_code}). '
                           'Check N8N_STUDIO_EMAIL / N8N_STUDIO_PASSWORD (MFA users cannot be used).',
                           status_code=502, code='login_failed')
        # Behind HTTPS (N8N_PROTOCOL=https, production and staging) n8n marks its
        # n8n-auth cookie Secure. This session talks to n8n server-to-server over the
        # internal http://n8n-container URL, where requests silently withholds Secure
        # cookies, so every call after login was a 401 (live catalog, manual runs).
        if self.base_url.startswith('http://'):
            for cookie in self.session.cookies:
                cookie.secure = False
        self._logged_in = True

    def request(self, method: str, path: str, **kwargs) -> N8nResponse:
        method = method.upper()
        with self._lock:
            if not self._logged_in:
                self.login()
        url = f'{self.base_url}{path}'
        resp = _send(self.session, method, url, max_retries=1, timeout=self.timeout, label=self.base_url, **kwargs)
        if resp.status_code == 401:
            with self._lock:
                self.login()
            resp = _send(self.session, method, url, max_retries=1, timeout=self.timeout, label=self.base_url, **kwargs)
        return _wrap(resp)


_client: N8nClient | None = None
_session: N8nSession | None = None
_lock = threading.Lock()


def get_client() -> N8nClient:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = N8nClient(conf.api_url(), conf.api_key(), timeout=conf.timeout())
    return _client


def get_session() -> N8nSession:
    global _session
    if _session is None:
        with _lock:
            if _session is None:
                _session = N8nSession(conf.api_url(), conf.login_email(), conf.login_password(), timeout=conf.timeout())
    return _session


def reset_clients() -> None:
    global _client, _session
    _client = None
    _session = None
