"""
Node-type and credential-type catalogs for the editor.

n8n serves these as ``/types/nodes.json`` / ``/types/credentials.json`` to logged-in
users only (the public API has no equivalent), so Studio ships a snapshot taken
from n8n 2.36.8 in ``data/*.json.gz`` and, when ``N8N_STUDIO_EMAIL/PASSWORD`` are
set, refreshes from the live instance (which also picks up community nodes).

The browser gets a small index (~60 KB gzipped) and fetches one node type's full
description (parameters, all versions) on demand.
"""

from __future__ import annotations

import gzip
import json
import logging
import threading
import time
from pathlib import Path

from . import conf
from .client import N8nError, get_session

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / 'data'
LIVE_TTL = 3600
INDEX_FIELDS = ('name', 'displayName', 'group', 'description', 'defaults', 'inputs', 'outputs',
                'iconUrl', 'icon', 'iconColor', 'subtitle', 'usableAsTool', 'polling', 'maxNodes',
                'documentationUrl', 'hidden', 'outputNames', 'inputNames', 'eventTriggerDescription')


def _versions(desc: dict) -> list:
    v = desc.get('version', 1)
    return v if isinstance(v, list) else [v]


def _is_trigger(desc: dict) -> bool:
    return 'trigger' in (desc.get('group') or []) or bool(desc.get('webhooks')) or bool(desc.get('polling')) \
        or desc.get('name', '').lower().endswith('trigger')


class Catalog:
    def __init__(self, nodes: list[dict], credentials: list[dict], source: str):
        self.source = source
        self.loaded_at = time.time()
        by_name: dict[str, list[dict]] = {}
        for desc in nodes:
            by_name.setdefault(desc['name'], []).append(desc)
        index = []
        self._nodes: dict[str, str] = {}
        for name, descs in by_name.items():
            descs.sort(key=lambda d: max(_versions(d)), reverse=True)
            latest = descs[0]
            entry = {k: latest.get(k) for k in INDEX_FIELDS if latest.get(k) is not None}
            entry['versions'] = sorted({v for d in descs for v in _versions(d)})
            entry['defaultVersion'] = latest.get('defaultVersion') or max(_versions(latest))
            entry['credentials'] = sorted({c['name'] for d in descs for c in (d.get('credentials') or [])})
            entry['trigger'] = _is_trigger(latest)
            entry['webhook'] = bool(latest.get('webhooks'))
            codex = latest.get('codex') or {}
            entry['codex'] = {k: codex.get(k) for k in ('categories', 'subcategories', 'alias') if codex.get(k)}
            index.append(entry)
            self._nodes[name] = json.dumps(descs, separators=(',', ':'))
        index.sort(key=lambda e: e.get('displayName') or e['name'])
        cred_index = []
        self._creds: dict[str, str] = {}
        for c in credentials:
            cred_index.append({
                'name': c['name'], 'displayName': c.get('displayName') or c['name'],
                'iconUrl': c.get('iconUrl'), 'documentationUrl': c.get('documentationUrl'),
                'supportedNodes': c.get('supportedNodes') or [], 'extends': c.get('extends') or [],
                'testable': bool(c.get('test')),
            })
            self._creds[c['name']] = json.dumps(c, separators=(',', ':'))
        cred_index.sort(key=lambda c: c['displayName'].lower())
        payload = {'source': source, 'loadedAt': self.loaded_at, 'nodes': index, 'credentials': cred_index}
        self.index_json = json.dumps(payload, separators=(',', ':')).encode()
        self.index_gzip = gzip.compress(self.index_json, 6)

    def node(self, name: str) -> str | None:
        return self._nodes.get(name)

    def credential(self, name: str) -> str | None:
        return self._creds.get(name)

    @property
    def node_count(self) -> int:
        return len(self._nodes)


def _read_gz(name: str) -> list:
    with gzip.open(DATA_DIR / name, 'rt', encoding='utf-8') as fh:
        return json.load(fh)


_bundled: Catalog | None = None
_live: Catalog | None = None
_live_failed_at = 0.0
_lock = threading.RLock()  # get_catalog() falls back to bundled() while holding it


def bundled() -> Catalog:
    global _bundled
    if _bundled is None:
        with _lock:
            if _bundled is None:
                _bundled = Catalog(_read_gz('nodes.json.gz'), _read_gz('credentials.json.gz'), 'bundled')
    return _bundled


def _fetch_live() -> Catalog:
    session = get_session()
    nodes = session.request('GET', '/types/nodes.json')
    creds = session.request('GET', '/types/credentials.json')
    if not (nodes.ok and creds.ok):
        raise N8nError(f'n8n type catalog returned {nodes.status_code}/{creds.status_code}')
    return Catalog(nodes.json(), creds.json(), 'live')


def get_catalog(force: bool = False) -> Catalog:
    """Live catalog when a login is configured (cached for an hour), else the bundled one."""
    global _live, _live_failed_at
    if not conf.login_configured():
        return bundled()
    now = time.time()
    if not force and _live and now - _live.loaded_at < LIVE_TTL:
        return _live
    if not force and now - _live_failed_at < 300:
        return _live or bundled()
    with _lock:
        try:
            _live = _fetch_live()
        except Exception as exc:  # any failure falls back to the snapshot
            logger.warning('Live n8n catalog unavailable, using bundled snapshot: %s', exc)
            _live_failed_at = now
            return _live or bundled()
    return _live


def reset() -> None:
    global _bundled, _live, _live_failed_at
    _bundled = _live = None
    _live_failed_at = 0.0
