"""
Low-level Airflow 3 REST API (v2) client.

Handles JWT authentication against ``/auth/token``, shares the token across
processes via the Django cache, transparently re-authenticates once on
``401``/``403`` and retries idempotent requests on transient failures.

Everything else in the app (the service facade, the browser proxy) goes
through :func:`get_client` so there is exactly one place that talks HTTP
to Airflow.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import requests
from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger(__name__)

TOKEN_CACHE_KEY = 'airflow_ui:jwt'
TOKEN_TTL_SECONDS = 23 * 60 * 60          # Airflow issues 24h tokens
IDEMPOTENT_METHODS = {'GET', 'HEAD', 'OPTIONS', 'PUT', 'DELETE'}
RETRY_STATUSES = {502, 503, 504}


class AirflowError(Exception):
    """Raised when Airflow cannot be reached or rejects a request."""

    def __init__(self, message: str, status_code: int = 502, payload: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload if payload is not None else {'detail': message}


@dataclass
class AirflowResponse:
    status_code: int
    content: bytes
    content_type: str = 'application/json'
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def json(self) -> Any:
        import json
        if not self.content:
            return None
        return json.loads(self.content)


def encode_segment(value: Any) -> str:
    """URL-encode a single path segment (run ids contain ``:`` and ``+``)."""
    return quote(str(value), safe='~')


class AirflowClient:
    """Thread-safe Airflow API client bound to one base URL + credentials."""

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        timeout: float = 30,
        max_retries: int = 2,
    ):
        self.base_url = base_url.rstrip('/')
        self.username = username
        self.password = password
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ auth

    @property
    def api_root(self) -> str:
        return f'{self.base_url}/api/v2'

    def _cache_key(self) -> str:
        return f'{TOKEN_CACHE_KEY}:{self.base_url}:{self.username}'

    def login(self) -> str:
        """Fetch a fresh JWT and store it in the shared cache."""
        try:
            resp = self.session.post(
                f'{self.base_url}/auth/token',
                json={'username': self.username, 'password': self.password},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise AirflowError(f'Cannot reach Airflow at {self.base_url}: {exc}') from exc

        if resp.status_code not in (200, 201):
            raise AirflowError(
                f'Airflow login failed ({resp.status_code})',
                status_code=502,
                payload={'detail': 'Airflow rejected the configured credentials.',
                         'upstream_status': resp.status_code},
            )
        token = resp.json().get('access_token', '')
        cache.set(self._cache_key(), token, TOKEN_TTL_SECONDS)
        logger.info('Airflow JWT refreshed for %s', self.base_url)
        return token

    def token(self, force: bool = False) -> str:
        if not force:
            cached = cache.get(self._cache_key())
            if cached:
                return cached
        with self._lock:
            if not force:
                cached = cache.get(self._cache_key())
                if cached:
                    return cached
            return self.login()

    # --------------------------------------------------------------- request

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Any = None,
        json: Any = None,
        data: bytes | None = None,
        accept: str = 'application/json',
        raw_path: bool = False,
    ) -> AirflowResponse:
        """
        Call ``/api/v2{path}``.  ``path`` must already be URL-encoded when
        ``raw_path`` is true; otherwise it is used as given.

        Returns the upstream response even for 4xx so callers (the proxy)
        can relay Airflow's ``detail`` messages.  Network failures and
        exhausted retries raise :class:`AirflowError`.
        """
        method = method.upper()
        url = f'{self.api_root}{path}'
        headers = {'Accept': accept}
        if json is not None or data is not None:
            headers['Content-Type'] = 'application/json'

        reauthed = False
        attempt = 0
        while True:
            headers['Authorization'] = f'Bearer {self.token()}'
            try:
                resp = self.session.request(
                    method, url, params=params, json=json, data=data,
                    headers=headers, timeout=self.timeout,
                )
            except requests.RequestException as exc:
                if method in IDEMPOTENT_METHODS and attempt < self.max_retries:
                    attempt += 1
                    time.sleep(0.5 * attempt)
                    continue
                raise AirflowError(f'Airflow request failed: {exc}') from exc

            if resp.status_code in (401, 403) and not reauthed and self._is_auth_failure(resp):
                reauthed = True
                self.token(force=True)
                continue

            if resp.status_code in RETRY_STATUSES and method in IDEMPOTENT_METHODS \
                    and attempt < self.max_retries:
                attempt += 1
                time.sleep(0.5 * attempt)
                continue

            logger.debug('Airflow %s %s -> %s', method, url, resp.status_code)
            return AirflowResponse(
                status_code=resp.status_code,
                content=resp.content,
                content_type=resp.headers.get('Content-Type', 'application/json'),
                headers={k: v for k, v in resp.headers.items()
                         if k.lower() in ('content-disposition',)},
            )

    @staticmethod
    def _is_auth_failure(resp: requests.Response) -> bool:
        """403 is also used for permission errors; only re-login for token problems."""
        if resp.status_code == 401:
            return True
        try:
            detail = str(resp.json().get('detail', ''))
        except ValueError:
            return False
        return 'token' in detail.lower() or 'not authenticated' in detail.lower()

    # ------------------------------------------------------------ helpers

    def get_json(self, path: str, **params) -> Any:
        resp = self.request('GET', path, params=params or None)
        if not resp.ok:
            raise AirflowError(_detail(resp), status_code=resp.status_code, payload=_payload(resp))
        return resp.json()

    def send_json(self, method: str, path: str, body: Any = None, **params) -> Any:
        resp = self.request(method, path, json=body, params=params or None)
        if not resp.ok:
            raise AirflowError(_detail(resp), status_code=resp.status_code, payload=_payload(resp))
        return resp.json()

    def paginate(self, path: str, key: str, page_size: int = 100, **params):
        """Yield every item of a list endpoint, following limit/offset."""
        offset = 0
        while True:
            page = self.get_json(path, limit=page_size, offset=offset, **params)
            items = page.get(key) or []
            yield from items
            offset += len(items)
            if not items or offset >= page.get('total_entries', 0):
                return


def _payload(resp: AirflowResponse) -> Any:
    try:
        return resp.json()
    except ValueError:
        return {'detail': resp.content.decode(errors='replace')[:500]}


def _detail(resp: AirflowResponse) -> str:
    payload = _payload(resp)
    if isinstance(payload, dict) and 'detail' in payload:
        return f'Airflow returned {resp.status_code}: {payload["detail"]}'
    return f'Airflow returned {resp.status_code}'


_client: AirflowClient | None = None
_client_lock = threading.Lock()


def get_client() -> AirflowClient:
    """Process-wide client built from Django settings."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = AirflowClient(
                    base_url=settings.AIRFLOW_BASE_URL,
                    username=settings.AIRFLOW_USERNAME,
                    password=settings.AIRFLOW_PASSWORD,
                    timeout=getattr(settings, 'AIRFLOW_TIMEOUT', 30),
                )
    return _client


def reset_client() -> None:
    """Drop the cached client (used by tests and after settings changes)."""
    global _client
    _client = None
