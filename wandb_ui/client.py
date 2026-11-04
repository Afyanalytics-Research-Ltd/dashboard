"""
Low-level W&B Server client.

W&B's API is GraphQL at ``{base}/graphql`` plus plain file downloads, both
authenticated with HTTP Basic ``api:<API key>``. The key stays server-side.
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

RETRY_STATUSES = {502, 503, 504}
PASSTHROUGH_HEADERS = ('content-disposition', 'content-length', 'last-modified', 'etag')


class WandbError(Exception):
    """W&B is unreachable or not configured."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code
        self.payload = {'errors': [{'message': message}]}


@dataclass
class WandbResponse:
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


class WandbClient:

    def __init__(self, base_url: str, api_key: str, timeout: float = 60, max_retries: int = 2):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def _auth(self):
        if not self.api_key:
            raise WandbError('WANDB_API_KEY is not configured on the server.', status_code=503)
        return ('api', self.api_key)

    def graphql(self, body: bytes, *, read_only: bool) -> WandbResponse:
        """POST a GraphQL document. Queries are retried on transient failures; mutations never."""
        return self._send('POST', f'{self.base_url}/graphql', data=body, retry=read_only,
                          headers={'Content-Type': 'application/json', 'Accept': 'application/json'})

    def get_file(self, url: str, *, range_header: str | None = None) -> WandbResponse:
        headers = {'Range': range_header} if range_header else {}
        return self._send('GET', url, headers=headers, retry=True, stream_limit=200 * 1024 * 1024)

    def _send(self, method, url, *, data=None, headers=None, retry=False, stream_limit=None) -> WandbResponse:
        auth = self._auth()
        attempt = 0
        while True:
            try:
                resp = self.session.request(method, url, data=data, headers=headers, auth=auth,
                                            timeout=self.timeout, allow_redirects=True)
            except requests.RequestException as exc:
                if retry and attempt < self.max_retries:
                    attempt += 1
                    time.sleep(0.5 * attempt)
                    continue
                raise WandbError(f'Cannot reach W&B at {self.base_url}: {exc}') from exc
            if resp.status_code in RETRY_STATUSES and retry and attempt < self.max_retries:
                attempt += 1
                time.sleep(0.5 * attempt)
                continue
            if stream_limit and len(resp.content) > stream_limit:
                raise WandbError('File is too large to preview here.', status_code=413)
            return WandbResponse(
                status_code=resp.status_code,
                content=resp.content,
                content_type=resp.headers.get('Content-Type', 'application/octet-stream'),
                headers={k: v for k, v in resp.headers.items() if k.lower() in PASSTHROUGH_HEADERS},
            )

    def query(self, document: str, variables: dict | None = None) -> dict:
        """Convenience for server-side callers: run a query and return ``data`` (raises on errors)."""
        import json
        resp = self.graphql(json.dumps({'query': document, 'variables': variables or {}}).encode(), read_only=True)
        payload = resp.json() or {}
        if not resp.ok or payload.get('errors'):
            msg = '; '.join(e.get('message', '') for e in payload.get('errors') or []) or f'HTTP {resp.status_code}'
            raise WandbError(f'W&B query failed: {msg}', status_code=resp.status_code if resp.status_code >= 400 else 502)
        return payload.get('data') or {}


_client: WandbClient | None = None
_lock = threading.Lock()


def get_client() -> WandbClient:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = WandbClient(
                    base_url=settings.WANDB_BASE_URL,
                    api_key=settings.WANDB_API_KEY,
                    timeout=getattr(settings, 'WANDB_TIMEOUT', 60),
                )
    return _client


def reset_client() -> None:
    global _client
    _client = None
