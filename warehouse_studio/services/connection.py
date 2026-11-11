"""Snowflake connection pool for Warehouse Studio.

Opening a Snowflake session costs 1–2 s (TLS + auth), so sessions are pooled
and re-used.  A pooled session is bound to one *context* — role, warehouse,
database, schema — because those are session state; keeping one pool bucket
per context means a query never has to issue ``USE`` statements on a session
another request is also using.

Each checkout is exclusive: a connection is handed to exactly one caller at a
time and returned with :func:`ConnectionPool.connection` (a context manager).
Dead sessions (network drop, expired token) are discarded and transparently
replaced on the next checkout.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from ..conf import IDENT_RE, StudioConfig, get_config

logger = logging.getLogger(__name__)

APPLICATION_NAME = 'AfyaWarehouseStudio'


class StudioConnectionError(Exception):
    """Snowflake could not be reached or rejected the credentials."""


@dataclass(frozen=True)
class Context:
    """Session context a query runs in.  Empty strings mean "account default"."""

    role: str = ''
    warehouse: str = ''
    database: str = ''
    schema: str = ''

    @classmethod
    def from_dict(cls, data: dict | None, default: 'Context | None' = None) -> 'Context':
        data = data or {}
        base = default or cls()
        values = {}
        for key in ('role', 'warehouse', 'database', 'schema'):
            raw = data.get(key)
            value = (raw if raw is not None else getattr(base, key)) or ''
            value = str(value).strip()
            if value.startswith('"') and value.endswith('"') and len(value) >= 2:
                value = value[1:-1]
            if value and not _valid_object_name(value):
                raise ValueError(f'Invalid {key} name: {value!r}')
            values[key] = value
        return cls(**values)

    def as_dict(self) -> dict:
        return {'role': self.role, 'warehouse': self.warehouse, 'database': self.database, 'schema': self.schema}


def _valid_object_name(value: str) -> bool:
    """Context names may be quoted-identifier style but never contain quotes or control chars."""
    return bool(value) and len(value) <= 255 and '"' not in value and not any(ord(c) < 32 for c in value)


def default_context(config: StudioConfig | None = None) -> Context:
    config = config or get_config()
    return Context(role=config.role, warehouse=config.warehouse, database=config.database, schema=config.schema)


def _load_private_key(path: str, passphrase: str = ''):
    """Return the DER bytes of a PEM private key (what the connector expects)."""
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization

    with open(path, 'rb') as fh:
        key = serialization.load_pem_private_key(
            fh.read(), password=passphrase.encode() if passphrase else None, backend=default_backend())
    return key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption())


def connect_kwargs(config: StudioConfig, context: Context, query_tag: str = '') -> dict:
    """Keyword arguments for :func:`snowflake.connector.connect`."""
    if not config.configured:
        raise StudioConnectionError(
            'Snowflake is not configured: set SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER and a password or private key.')
    kwargs = {
        'account': config.account,
        'user': config.user,
        'application': APPLICATION_NAME,
        'client_session_keep_alive': True,
        'login_timeout': 30,
        'network_timeout': 120,
        'session_parameters': {
            'QUERY_TAG': query_tag or config.query_tag_prefix,
            'STATEMENT_TIMEOUT_IN_SECONDS': int(config.statement_timeout_seconds),
            # Async queries must survive the checkout that started them.
            'ABORT_DETACHED_QUERY': False,
            'TIMEZONE': 'UTC',
        },
    }
    for key in ('role', 'warehouse', 'database', 'schema'):
        value = getattr(context, key)
        if value:
            # Plain identifiers go through bare (Snowflake upper-cases them);
            # anything else is quoted so mixed-case names keep their case.
            kwargs[key] = value if IDENT_RE.match(value) else f'"{value}"'
    if config.private_key_path:
        kwargs['private_key'] = _load_private_key(config.private_key_path, config.private_key_passphrase)
    elif config.authenticator:
        kwargs['authenticator'] = config.authenticator
        if config.password:
            kwargs['password'] = config.password
    else:
        kwargs['password'] = config.password
    return kwargs


class ConnectionPool:
    """Thread-safe pool of Snowflake sessions, bucketed by :class:`Context`."""

    def __init__(self, config: StudioConfig | None = None, connector=None):
        self._config = config
        self._connector = connector
        self._lock = threading.Condition()
        self._idle: 'OrderedDict[Context, list]' = OrderedDict()
        self._busy = 0

    # -- plumbing -----------------------------------------------------------------

    @property
    def config(self) -> StudioConfig:
        return self._config or get_config()

    def _connect(self, context: Context):
        connector = self._connector
        if connector is None:
            # Snowflake or Databend, per the platform warehouse switch
            # (same .connect(**kwargs) interface as snowflake.connector).
            from afya_backends import warehouse as connector  # noqa: PLC0415
        try:
            conn = connector.connect(**connect_kwargs(self.config, context))
        except StudioConnectionError:
            raise
        except Exception as exc:  # connector raises many types; normalise
            logger.warning('Snowflake connect failed: %s', exc)
            raise StudioConnectionError(_clean_error(exc)) from exc
        conn._ws_created = time.monotonic()  # noqa: SLF001 — bookkeeping on the driver object
        return conn

    @staticmethod
    def _alive(conn) -> bool:
        try:
            return not conn.is_closed()
        except Exception:  # noqa: BLE001
            return False

    def _total_idle(self) -> int:
        return sum(len(v) for v in self._idle.values())

    def _evict_one(self) -> bool:
        """Close the least-recently-used idle session from any bucket."""
        for ctx in list(self._idle):
            bucket = self._idle[ctx]
            if bucket:
                conn = bucket.pop(0)
                if not bucket:
                    del self._idle[ctx]
                _quiet_close(conn)
                return True
        return False

    # -- public API ---------------------------------------------------------------

    def acquire(self, context: Context, timeout: float = 60.0):
        deadline = time.monotonic() + timeout
        with self._lock:
            while True:
                bucket = self._idle.get(context)
                while bucket:
                    conn = bucket.pop()
                    if not bucket:
                        self._idle.pop(context, None)
                    if self._alive(conn):
                        self._busy += 1
                        return conn
                    _quiet_close(conn)
                    bucket = self._idle.get(context)
                if self._busy + self._total_idle() < self.config.pool_size or self._evict_one():
                    self._busy += 1
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise StudioConnectionError('All Snowflake sessions are busy; try again in a moment.')
                self._lock.wait(remaining)
        try:
            return self._connect(context)
        except Exception:
            with self._lock:
                self._busy -= 1
                self._lock.notify()
            raise

    def release(self, context: Context, conn, *, discard: bool = False) -> None:
        with self._lock:
            self._busy = max(0, self._busy - 1)
            if discard or not self._alive(conn):
                _quiet_close(conn)
            else:
                self._idle.setdefault(context, []).append(conn)
                self._idle.move_to_end(context)
            self._lock.notify()

    @contextmanager
    def connection(self, context: Context | None = None) -> Iterator:
        context = context or default_context(self.config)
        conn = self.acquire(context)
        discard = False
        try:
            yield conn
        except Exception as exc:
            discard = _is_connection_failure(exc)
            raise
        finally:
            self.release(context, conn, discard=discard)

    def close_all(self) -> None:
        with self._lock:
            for bucket in self._idle.values():
                for conn in bucket:
                    _quiet_close(conn)
            self._idle.clear()

    def stats(self) -> dict:
        with self._lock:
            return {'busy': self._busy, 'idle': self._total_idle(), 'contexts': len(self._idle),
                    'size': self.config.pool_size}


def _quiet_close(conn) -> None:
    try:
        conn.close()
    except Exception:  # noqa: BLE001
        pass


def _is_connection_failure(exc: Exception) -> bool:
    name = type(exc).__name__
    text = str(exc).lower()
    return name in ('OperationalError', 'InterfaceError') or 'session no longer exists' in text \
        or 'authentication token has expired' in text or 'connection is closed' in text \
        or 'response ended prematurely' in text or 'connection reset' in text or 'connection aborted' in text \
        or 'remote end closed' in text or 'max retries exceeded' in text


def _clean_error(exc: Exception) -> str:
    msg = getattr(exc, 'msg', None) or str(exc)
    return str(msg).strip().splitlines()[0][:500] if msg else type(exc).__name__


_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def get_pool() -> ConnectionPool:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ConnectionPool()
        return _pool


def reset_pool(pool: ConnectionPool | None = None) -> None:
    """Swap the process-wide pool (tests inject a fake one)."""
    global _pool
    with _pool_lock:
        if _pool is not None and _pool is not pool:
            _pool.close_all()
        _pool = pool
