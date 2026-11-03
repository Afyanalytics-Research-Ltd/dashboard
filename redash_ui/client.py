"""
Low-level Redash HTTP client.

Authenticates with the admin user API key (``Authorization: Key …``) from
settings, so the key never reaches the browser. Idempotent requests are
retried on connection errors and 502/503/504.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

IDEMPOTENT_METHODS = {'GET', 'HEAD', 'OPTIONS', 'DELETE'}
RETRY_STATUSES = {502, 503, 504}
PASSTHROUGH_HEADERS = ('content-disposition',)


class RedashError(Exception):
    """Raised when Redash cannot be reached or is not configured."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code
        self.payload = {'message': message}


@dataclass
class RedashResponse:
    status_code: int
    content: bytes
    content_type: str = 'application/json'
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def json(self) -> Any:
        import json
        return json.loads(self.content) if self.content else None


class RedashClient:

    def __init__(self, base_url: str, api_key: str, timeout: float = 60, max_retries: int = 2):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Any = None,
        data: bytes | None = None,
        json: Any = None,
    ) -> RedashResponse:
        """Call ``{base_url}{path}``; 4xx/5xx responses are returned, not raised."""
        if not self.api_key:
            raise RedashError('REDASH_ADMIN_API_KEY is not configured on the server.', status_code=503)
        method = method.upper()
        headers = {'Authorization': f'Key {self.api_key}', 'Accept': 'application/json'}
        if data is not None or json is not None:
            headers['Content-Type'] = 'application/json'
        url = f'{self.base_url}{path}'
        attempt = 0
        while True:
            try:
                resp = self.session.request(method, url, params=params, data=data, json=json,
                                            headers=headers, timeout=self.timeout, allow_redirects=False)
            except requests.RequestException as exc:
                if method in IDEMPOTENT_METHODS and attempt < self.max_retries:
                    attempt += 1
                    time.sleep(0.5 * attempt)
                    continue
                raise RedashError(f'Cannot reach Redash at {self.base_url}: {exc}') from exc

            if resp.status_code in RETRY_STATUSES and method in IDEMPOTENT_METHODS and attempt < self.max_retries:
                attempt += 1
                time.sleep(0.5 * attempt)
                continue

            # Redash redirects to /login when the key is wrong.
            if resp.status_code in (301, 302) and '/login' in resp.headers.get('Location', ''):
                raise RedashError('Redash rejected the configured API key.', status_code=502)

            logger.debug('Redash %s %s -> %s', method, url, resp.status_code)
            return RedashResponse(
                status_code=resp.status_code,
                content=resp.content,
                content_type=resp.headers.get('Content-Type', 'application/json'),
                headers={k: v for k, v in resp.headers.items() if k.lower() in PASSTHROUGH_HEADERS},
            )

    def get_json(self, path: str, **params) -> Any:
        resp = self.request('GET', path, params=params or None)
        if not resp.ok:
            raise RedashError(f'Redash returned {resp.status_code} for {path}', status_code=resp.status_code)
        return resp.json()


_client: RedashClient | None = None
_lock = threading.Lock()


def get_client() -> RedashClient:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = RedashClient(
                    base_url=getattr(settings, 'REDASH_API_URL', '') or settings.REDASH_BASE_URL,
                    api_key=settings.REDASH_ADMIN_API_KEY,
                    timeout=getattr(settings, 'REDASH_TIMEOUT', 60),
                )
    return _client


def reset_client() -> None:
    global _client
    _client = None
