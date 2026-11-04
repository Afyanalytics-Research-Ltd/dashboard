"""What the Studio is connected to: identity, context choices, account facts, toolkit status."""

from __future__ import annotations

import json

from django.core.cache import cache

from ..conf import get_config
from . import executor, sources
from .connection import Context, default_context, get_pool


def info(context: Context | None = None, refresh: bool = False) -> dict:
    cfg = get_config()
    key = f'ws:session:{context.as_dict() if context else ""}'
    if not refresh:
        hit = cache.get(key)
        if hit is not None:
            return {**hit, 'pool': get_pool().stats()}
    ctx = context or default_context()
    row = executor.records(
        'SELECT CURRENT_ACCOUNT() AS account_locator, CURRENT_ACCOUNT_NAME() AS account_name, '
        'CURRENT_ORGANIZATION_NAME() AS organization, CURRENT_REGION() AS region, CURRENT_USER() AS user, '
        'CURRENT_ROLE() AS role, CURRENT_WAREHOUSE() AS warehouse, CURRENT_DATABASE() AS database, '
        'CURRENT_SCHEMA() AS schema, CURRENT_VERSION() AS version, CURRENT_AVAILABLE_ROLES() AS roles, '
        'CURRENT_CLIENT() AS client', context=ctx, tag=executor.statement_tag('session'))[0]
    roles = row.pop('roles')
    if isinstance(roles, str):
        try:
            roles = json.loads(roles)
        except ValueError:
            roles = [roles]
    try:
        warehouses = [{'name': w['name'], 'size': w.get('size'), 'state': w.get('state'), 'type': w.get('type'),
                       'auto_suspend': w.get('auto_suspend')}
                      for w in executor.records('SHOW WAREHOUSES', context=ctx, tag=executor.statement_tag('session'))]
    except executor.QueryError:
        warehouses = []
    if refresh:
        sources.invalidate()
    probe = sources.probe(force=refresh)
    data = {
        'connected': True, **row, 'available_roles': sorted(roles or []), 'warehouses': warehouses,
        'default_context': default_context().as_dict(),
        'toolkit': probe, 'credit_price_usd': cfg.credit_price_usd,
        'storage_price_per_tb_month_usd': cfg.storage_price_per_tb_month_usd,
        'allow_writes': cfg.allow_writes, 'max_result_rows': cfg.max_result_rows,
        'statement_timeout_seconds': cfg.statement_timeout_seconds,
    }
    cache.set(key, data, 120)
    return {**data, 'pool': get_pool().stats()}
