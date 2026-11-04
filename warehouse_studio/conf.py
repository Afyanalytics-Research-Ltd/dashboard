"""Warehouse Studio configuration.

Every value can be set in Django settings (``WAREHOUSE_STUDIO = {...}``) or
through an environment variable; settings win over the environment.  The
Snowflake identity defaults to the platform-wide ``SNOWFLAKE_*`` variables
the rest of the dashboard already uses, so no new secret is required.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from django.conf import settings

IDENT_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_$]*$')


def _env(name: str, default: str = '') -> str:
    return os.getenv(name, default).strip().strip('"').strip("'")


def _float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ''):
        return default
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


@dataclass(frozen=True)
class StudioConfig:
    account: str = ''
    user: str = ''
    password: str = ''
    private_key_path: str = ''
    private_key_passphrase: str = ''
    authenticator: str = ''
    role: str = ''
    warehouse: str = ''
    database: str = ''
    schema: str = ''
    # Where the cost & performance toolkit lives (install.sql target).
    toolkit_database: str = 'HOSPITALS'
    toolkit_schema: str = 'ANALYTICS'
    # Pricing used for every dollar figure.  Snowflake bills credits; the
    # contract price per credit differs per account, so it is configurable.
    credit_price_usd: float = 2.60
    storage_price_per_tb_month_usd: float = 23.0
    # Execution limits.
    max_result_rows: int = 10_000
    max_export_rows: int = 1_000_000
    statement_timeout_seconds: int = 900
    pool_size: int = 6
    allow_writes: bool = False
    query_tag_prefix: str = 'warehouse-studio'
    cache_seconds: int = 300
    extra: dict = field(default_factory=dict)

    @property
    def toolkit(self) -> str:
        """Fully-qualified toolkit schema, e.g. ``HOSPITALS.ANALYTICS``."""
        return f'{self.toolkit_database}.{self.toolkit_schema}'

    @property
    def configured(self) -> bool:
        return bool(self.account and self.user and (self.password or self.private_key_path or self.authenticator))


def get_config() -> StudioConfig:
    """Build the config from Django settings, falling back to environment variables."""
    s = dict(getattr(settings, 'WAREHOUSE_STUDIO', {}) or {})

    def pick(key: str, env_names: tuple[str, ...], default=''):
        if key in s and s[key] not in (None, ''):
            return s[key]
        for name in env_names:
            value = _env(name)
            if value:
                return value
        return default

    key_path = pick('private_key_path', ('WAREHOUSE_STUDIO_SNOWFLAKE_PRIVATE_KEY_PATH', 'SNOWFLAKE_PRIVATE_KEY_PATH'))
    if key_path and not os.path.isabs(key_path):
        key_path = os.path.join(str(getattr(settings, 'BASE_DIR', '.')), key_path)

    toolkit_db = str(pick('toolkit_database', ('WAREHOUSE_STUDIO_TOOLKIT_DATABASE',), 'HOSPITALS')).upper()
    toolkit_schema = str(pick('toolkit_schema', ('WAREHOUSE_STUDIO_TOOLKIT_SCHEMA',), 'ANALYTICS')).upper()
    if not IDENT_RE.match(toolkit_db) or not IDENT_RE.match(toolkit_schema):
        raise ValueError('WAREHOUSE_STUDIO toolkit database/schema must be plain identifiers')

    return StudioConfig(
        account=pick('account', ('WAREHOUSE_STUDIO_SNOWFLAKE_ACCOUNT', 'SNOWFLAKE_ACCOUNT')),
        user=pick('user', ('WAREHOUSE_STUDIO_SNOWFLAKE_USER', 'SNOWFLAKE_USER')),
        password=pick('password', ('WAREHOUSE_STUDIO_SNOWFLAKE_PASSWORD', 'SNOWFLAKE_PASSWORD')),
        private_key_path=key_path,
        private_key_passphrase=pick('private_key_passphrase', ('WAREHOUSE_STUDIO_SNOWFLAKE_PRIVATE_KEY_PASSPHRASE', 'SNOWFLAKE_PRIVATE_KEY_PASSPHRASE')),
        authenticator=pick('authenticator', ('WAREHOUSE_STUDIO_SNOWFLAKE_AUTHENTICATOR',)),
        role=pick('role', ('WAREHOUSE_STUDIO_SNOWFLAKE_ROLE', 'SNOWFLAKE_ROLE')),
        warehouse=pick('warehouse', ('WAREHOUSE_STUDIO_SNOWFLAKE_WAREHOUSE', 'SNOWFLAKE_WAREHOUSE')),
        database=pick('database', ('WAREHOUSE_STUDIO_SNOWFLAKE_DATABASE', 'SNOWFLAKE_DATABASE')),
        schema=pick('schema', ('WAREHOUSE_STUDIO_SNOWFLAKE_SCHEMA', 'SNOWFLAKE_SCHEMA')),
        toolkit_database=toolkit_db,
        toolkit_schema=toolkit_schema,
        credit_price_usd=_float(pick('credit_price_usd', ('WAREHOUSE_STUDIO_CREDIT_PRICE_USD',), 2.60), 2.60),
        storage_price_per_tb_month_usd=_float(pick('storage_price_per_tb_month_usd', ('WAREHOUSE_STUDIO_STORAGE_PRICE_PER_TB_USD',), 23.0), 23.0),
        max_result_rows=_int(pick('max_result_rows', ('WAREHOUSE_STUDIO_MAX_RESULT_ROWS',), 10_000), 10_000),
        max_export_rows=_int(pick('max_export_rows', ('WAREHOUSE_STUDIO_MAX_EXPORT_ROWS',), 1_000_000), 1_000_000),
        statement_timeout_seconds=_int(pick('statement_timeout_seconds', ('WAREHOUSE_STUDIO_STATEMENT_TIMEOUT',), 900), 900),
        pool_size=max(1, _int(pick('pool_size', ('WAREHOUSE_STUDIO_POOL_SIZE',), 6), 6)),
        allow_writes=_bool(pick('allow_writes', ('WAREHOUSE_STUDIO_ALLOW_WRITES',), False)),
        query_tag_prefix=str(pick('query_tag_prefix', ('WAREHOUSE_STUDIO_QUERY_TAG',), 'warehouse-studio')),
        cache_seconds=_int(pick('cache_seconds', ('WAREHOUSE_STUDIO_CACHE_SECONDS',), 300), 300),
    )


# Credits per hour for a standard warehouse of each size (Snowflake list).
CREDITS_PER_HOUR = {
    'X-SMALL': 1, 'XSMALL': 1, 'SMALL': 2, 'MEDIUM': 4, 'LARGE': 8,
    'X-LARGE': 16, 'XLARGE': 16, '2X-LARGE': 32, 'XXLARGE': 32, 'X2LARGE': 32,
    '3X-LARGE': 64, 'XXXLARGE': 64, 'X3LARGE': 64, '4X-LARGE': 128, 'X4LARGE': 128,
    '5X-LARGE': 256, 'X5LARGE': 256, '6X-LARGE': 512, 'X6LARGE': 512,
}

SIZE_ORDER = ['X-Small', 'Small', 'Medium', 'Large', 'X-Large', '2X-Large', '3X-Large', '4X-Large', '5X-Large', '6X-Large']


def credits_per_hour(size: str | None) -> float:
    """Credits/hour for a warehouse size label like ``X-Small`` (0 when unknown)."""
    if not size:
        return 0.0
    return float(CREDITS_PER_HOUR.get(str(size).strip().upper().replace(' ', ''), 0))
