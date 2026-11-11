"""Turn Snowflake cursor output into JSON-safe structures.

The connector returns Decimal, datetime/date/time (often tz-aware), bytes,
and VARIANT/OBJECT/ARRAY as JSON *strings*.  The browser needs plain JSON,
and the grid needs to know column types to align, format and profile them.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import math
import uuid
from decimal import Decimal

# snowflake.connector.constants.FIELD_TYPES order (type_code -> name).  Kept
# local so this module imports without the connector (tests, docs builds).
FIELD_TYPE_NAMES = {
    0: 'FIXED', 1: 'REAL', 2: 'TEXT', 3: 'DATE', 4: 'TIMESTAMP', 5: 'VARIANT', 6: 'TIMESTAMP_LTZ',
    7: 'TIMESTAMP_TZ', 8: 'TIMESTAMP_NTZ', 9: 'OBJECT', 10: 'ARRAY', 11: 'BINARY', 12: 'TIME',
    13: 'BOOLEAN', 14: 'GEOGRAPHY', 15: 'GEOMETRY', 16: 'VECTOR', 17: 'MAP', 18: 'FILE', 19: 'DECFLOAT',
}

NUMERIC = {'FIXED', 'REAL', 'DECFLOAT'}
TEMPORAL = {'DATE', 'TIMESTAMP', 'TIMESTAMP_LTZ', 'TIMESTAMP_TZ', 'TIMESTAMP_NTZ', 'TIME'}
SEMI = {'VARIANT', 'OBJECT', 'ARRAY', 'MAP', 'GEOGRAPHY', 'GEOMETRY', 'VECTOR', 'FILE'}

# JavaScript loses precision past 2**53; send big integers as strings.
_JS_SAFE = 2 ** 53 - 1


def type_name(type_code) -> str:
    if isinstance(type_code, str):
        return type_code.upper()
    return FIELD_TYPE_NAMES.get(type_code, 'TEXT')


def type_family(name: str) -> str:
    if name in NUMERIC:
        return 'number'
    if name in TEMPORAL:
        return 'time' if name == 'TIME' else ('date' if name == 'DATE' else 'timestamp')
    if name == 'BOOLEAN':
        return 'boolean'
    if name in SEMI:
        return 'json'
    if name == 'BINARY':
        return 'binary'
    return 'text'


def describe_columns(description) -> list[dict]:
    """Column metadata from ``cursor.description`` (ResultMetadata tuples or plain 7-tuples)."""
    cols = []
    seen: dict[str, int] = {}
    for i, d in enumerate(description or []):
        name = getattr(d, 'name', None) or d[0]
        code = getattr(d, 'type_code', None) if hasattr(d, 'type_code') else d[1]
        precision = getattr(d, 'precision', None) if hasattr(d, 'precision') else (d[4] if len(d) > 4 else None)
        scale = getattr(d, 'scale', None) if hasattr(d, 'scale') else (d[5] if len(d) > 5 else None)
        nullable = getattr(d, 'is_nullable', None) if hasattr(d, 'is_nullable') else (d[6] if len(d) > 6 else None)
        tname = type_name(code)
        if tname == 'FIXED' and scale:
            label = f'NUMBER({precision},{scale})'
        elif tname == 'FIXED':
            label = 'NUMBER' if not precision else f'NUMBER({precision},0)'
        else:
            label = tname
        # Duplicate column names (SELECT a.id, b.id) get a stable key suffix.
        key = name
        if name in seen:
            seen[name] += 1
            key = f'{name}_{seen[name]}'
        else:
            seen[name] = 0
        cols.append({
            'index': i, 'name': name, 'key': key, 'type': tname, 'label': label,
            'family': type_family(tname), 'precision': precision, 'scale': scale,
            'nullable': bool(nullable) if nullable is not None else True,
        })
    return cols


def to_json_value(value, family: str = 'text'):
    """One cell -> JSON-safe value."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if -_JS_SAFE <= value <= _JS_SAFE else str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            return str(value)
        if value == value.to_integral_value() and abs(value) <= _JS_SAFE:
            return int(value)
        # Keep exact decimals exact unless they round-trip through float.
        as_float = float(value)
        return as_float if Decimal(repr(as_float)) == value else str(value)
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, (dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, dt.timedelta):
        return value.total_seconds()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {'$binary': base64.b16encode(bytes(value)).decode()[:4096]}
    if isinstance(value, uuid.UUID):
        return str(value)
    if family == 'json' and isinstance(value, str):
        try:
            return {'$json': json.loads(value)}
        except (ValueError, TypeError):
            return value
    if isinstance(value, (list, tuple)):
        return [to_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_json_value(v) for k, v in value.items()}
    return str(value)


def rows_to_json(rows, columns: list[dict]) -> list[list]:
    families = [c['family'] for c in columns]
    out = []
    for row in rows:
        out.append([to_json_value(v, families[i] if i < len(families) else 'text') for i, v in enumerate(row)])
    return out


def records(cursor_rows, description) -> list[dict]:
    """Rows as dicts keyed by lower-case column name (for internal dashboards)."""
    names = [(getattr(d, 'name', None) or d[0]).lower() for d in description or []]
    cols = describe_columns(description)
    families = [c['family'] for c in cols]
    return [{names[i]: (plain(v) if families[i] == 'json' else to_json_value(v, families[i]))
             for i, v in enumerate(r)} for r in cursor_rows]


def plain(value):
    """Like :func:`to_json_value` but VARIANT stays a parsed object (no ``$json`` wrapper)."""
    v = to_json_value(value, 'json') if isinstance(value, str) else to_json_value(value)
    if isinstance(v, dict) and set(v) == {'$json'}:
        return v['$json']
    return v
