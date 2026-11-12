"""``write_pandas`` for both warehouses.

Same signature and return value as
``snowflake.connector.pandas_tools.write_pandas``.  On a Snowflake
connection it simply delegates.  On a Databend connection it creates the
table if asked (types inferred from the DataFrame like Snowflake's
auto_create_table), then bulk-loads the frame as CSV through Databend's
streaming-load endpoint - one HTTP upload per chunk, no row-by-row
INSERTs.
"""

from __future__ import annotations

import csv
import datetime as _dt
import decimal
import io
import json
import logging
import math
from typing import Any, Optional

from . import sql_translate as tr

logger = logging.getLogger(__name__)

# NULL travels as a token no real value contains, so "" stays an empty
# string and a literal "\N" stays "\N" (Databend's CSV reader doesn't
# unescape backslashes).
_NULL = "__AFYA_NULL_7f3c__"
_CSV_FORMAT = ("FILE_FORMAT = (type = CSV, field_delimiter = ',', quote = '\"', skip_header = 0, "
               f"null_display = '{_NULL}', empty_field_as = string)")


def _q(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _is_databend(conn) -> bool:
    return getattr(conn, "backend", None) == "databend"


def write_pandas(conn, df, table_name: str, database: Optional[str] = None, schema: Optional[str] = None,
                 chunk_size: Optional[int] = None, compression: str = "gzip", on_error: str = "abort_statement",
                 parallel: int = 4, quote_identifiers: bool = True, auto_create_table: bool = False,
                 create_temp_table: bool = False, overwrite: bool = False, table_type: str = "",
                 use_logical_type: Optional[bool] = None, **kwargs):
    if not _is_databend(conn):
        from snowflake.connector.pandas_tools import write_pandas as sf_write_pandas
        args = dict(database=database, schema=schema, chunk_size=chunk_size, compression=compression,
                    on_error=on_error, parallel=parallel, quote_identifiers=quote_identifiers,
                    auto_create_table=auto_create_table, create_temp_table=create_temp_table,
                    overwrite=overwrite, table_type=table_type, **kwargs)
        if use_logical_type is not None:
            args["use_logical_type"] = use_logical_type
        return sf_write_pandas(conn, df, table_name, **args)
    return _write_databend(conn, df, table_name, database, schema, chunk_size, quote_identifiers,
                           auto_create_table, create_temp_table, overwrite, table_type)


def _ident(name: str, quote_identifiers: bool) -> str:
    return _q(name if quote_identifiers else str(name).upper())


def _target(conn, table_name, database, schema, quote_identifiers):
    ctx = conn._ctx
    sf_db = database or ctx.database
    sf_schema = schema or ctx.schema
    if sf_schema:
        db = tr.map_database(sf_db if quote_identifiers or not sf_db else sf_db.upper(),
                             sf_schema if quote_identifiers else sf_schema.upper(), ctx.primary_database)
        if not quote_identifiers:
            db = db.upper()
        return db, f"{_q(db)}.{_ident(table_name, quote_identifiers)}"
    return None, _ident(table_name, quote_identifiers)


def infer_type(series) -> str:
    import pandas as pd
    from pandas.api import types as pt
    dt = series.dtype
    if pt.is_bool_dtype(dt):
        return "BOOLEAN"
    if pt.is_integer_dtype(dt):
        return "BIGINT"
    if pt.is_float_dtype(dt):
        return "DOUBLE"
    if pt.is_datetime64_any_dtype(dt):
        return "TIMESTAMP"
    if pt.is_timedelta64_dtype(dt):
        return "BIGINT"
    sample = series.dropna()
    if len(sample) == 0:
        return "VARCHAR"
    v = sample.iloc[0]
    if isinstance(v, (dict, list)):
        return "VARIANT"
    if isinstance(v, bool):
        return "BOOLEAN"
    if isinstance(v, int):
        return "BIGINT"
    if isinstance(v, float):
        return "DOUBLE"
    if isinstance(v, decimal.Decimal):
        return "DECIMAL(38, 9)"
    if isinstance(v, (_dt.datetime, pd.Timestamp)):
        return "TIMESTAMP"
    if isinstance(v, _dt.date):
        return "DATE"
    if isinstance(v, (bytes, bytearray)):
        return "BINARY"
    return "VARCHAR"


def _cell(v: Any) -> Any:
    if v is None:
        return _NULL
    try:
        import pandas as pd
        if v is pd.NaT or (not isinstance(v, (list, dict, str, bytes)) and pd.isna(v)):
            return _NULL
        if isinstance(v, pd.Timestamp):
            v = v.to_pydatetime()
    except (TypeError, ValueError, ImportError):
        pass
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            v = v.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(v, float) and (math.isnan(v)):
        return _NULL
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, _dt.datetime):
        if v.tzinfo is not None:
            v = v.astimezone(_dt.timezone.utc).replace(tzinfo=None)
        return v.isoformat(sep=" ")
    if isinstance(v, (_dt.date, _dt.time)):
        return v.isoformat()
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, default=str)
    if isinstance(v, (bytes, bytearray)):
        return bytes(v).hex()
    if isinstance(v, decimal.Decimal):
        return format(v, "f")
    return v


def _to_csv_bytes(df) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, quoting=csv.QUOTE_MINIMAL, lineterminator="\n", escapechar=None, doublequote=True)
    for row in df.itertuples(index=False, name=None):
        out = []
        w.writerow([_cell(v) for v in row])
    return buf.getvalue().encode("utf-8")


def _write_databend(conn, df, table_name, database, schema, chunk_size, quote_identifiers,
                    auto_create_table, create_temp_table, overwrite, table_type):
    db, target = _target(conn, table_name, database, schema, quote_identifiers)
    t = conn._t
    if db:
        t.run(f"CREATE DATABASE IF NOT EXISTS {_q(db)}")
    cols = [_ident(c, quote_identifiers) for c in df.columns]
    kind = (table_type or "").lower()
    if create_temp_table:
        kind = "temporary"
    if auto_create_table:
        defs = ", ".join(f"{c} {infer_type(df[name])}" for c, name in zip(cols, df.columns))
        prefix = {"temp": "TEMPORARY ", "temporary": "TEMPORARY ", "transient": "TRANSIENT "}.get(kind, "")
        if overwrite:
            t.run(f"CREATE OR REPLACE {prefix}TABLE {target} ({defs})")
        else:
            t.run(f"CREATE {prefix}TABLE IF NOT EXISTS {target} ({defs})")
    elif overwrite:
        t.run(f"TRUNCATE TABLE {target}")
    n = len(df)
    if n == 0:
        return True, 0, 0, []
    step = chunk_size or 200_000
    nchunks, loaded, output = 0, 0, []
    insert = f"INSERT INTO {target} ({', '.join(cols)}) FROM @_databend_load {_CSV_FORMAT}"
    for start in range(0, n, step):
        part = df.iloc[start:start + step]
        payload = _to_csv_bytes(part)
        resp = t.streaming_load(insert, f"chunk_{nchunks}.csv", payload)
        rows = int(((resp.get("stats") or {}).get("rows")) or len(part))
        loaded += rows
        output.append((f"chunk_{nchunks}.csv", "LOADED", rows, rows, 1, 0, None, None, None, None))
        nchunks += 1
    logger.info("write_pandas[databend]: %s rows into %s in %s chunk(s)", loaded, target, nchunks)
    return True, nchunks, loaded, output
