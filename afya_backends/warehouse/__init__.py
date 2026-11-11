"""One warehouse API for Snowflake and Databend.

Use this everywhere instead of ``snowflake.connector`` directly::

    from afya_backends import warehouse as afya_warehouse

    conn = afya_warehouse.connect(account=..., user=..., database="HOSPITALS", schema="RAW")
    cur = conn.cursor()
    cur.execute("SELECT * FROM HOSPITALS.STAGING.STG_VISITS WHERE id = %s", (5,))
    afya_warehouse.write_pandas(conn, df, "MY_TABLE", schema="RAW", auto_create_table=True)

``connect`` accepts exactly snowflake.connector.connect's keyword arguments.
Which warehouse it returns is decided at call time by the platform switch
(afya_backends.toggles: System Settings -> Data warehouse, or an Airflow
DAG run's ``warehouse_backend`` param), so flipping the switch needs no
restart and no code change.  Snowflake-only kwargs (account, role,
warehouse, private_key, authenticator ...) are ignored on Databend.

Importing this package is cheap (no sqlglot/pandas/requests until a
Databend connection or translation is actually used) - DAG files import it
at parse time.
"""

from __future__ import annotations

from typing import Optional

from .. import toggles

SNOWFLAKE = toggles.SNOWFLAKE
DATABEND = toggles.DATABEND


def backend() -> str:
    """The warehouse new connections will go to: "snowflake" or "databend"."""
    return toggles.warehouse_backend()


def is_databend() -> bool:
    return backend() == DATABEND


def connect(*args, backend: Optional[str] = None, **kwargs):
    """``snowflake.connector.connect`` for whichever warehouse is active."""
    chosen = toggles.normalize(toggles.WAREHOUSE, backend) if backend else toggles.warehouse_backend()
    if chosen == DATABEND:
        from .dbapi import Connection
        return Connection(**kwargs)
    import snowflake.connector
    return snowflake.connector.connect(*args, **kwargs)


def write_pandas(conn, df, table_name, *args, **kwargs):
    """``snowflake.connector.pandas_tools.write_pandas`` for either warehouse."""
    from .pandas_tools import write_pandas as _wp
    return _wp(conn, df, table_name, *args, **kwargs)


def translate(sql, ctx=None):
    from .sql_translate import translate as _t
    return _t(sql, ctx)


def translate_sql(sql, ctx=None):
    from .sql_translate import translate_sql as _t
    return _t(sql, ctx)


def is_databend_connection(conn) -> bool:
    return getattr(conn, "backend", None) == DATABEND


def connection_backend(conn) -> str:
    return DATABEND if is_databend_connection(conn) else SNOWFLAKE


def errors():
    """Exception classes to catch for either backend (Snowflake's base
    classes - the Databend ones subclass them)."""
    try:
        from snowflake.connector import errors as e
        return e
    except ImportError:  # pragma: no cover
        from . import dbapi as e
        return e


def __getattr__(name):
    if name in ("WarehouseHook", "DatabendHook"):
        from . import airflow_hook
        return getattr(airflow_hook, name)
    if name in ("Connection", "Cursor", "DatabendProgrammingError", "DatabendOperationalError"):
        from . import dbapi
        return getattr(dbapi, name)
    if name in ("Context", "TranslationError"):
        from . import sql_translate
        return getattr(sql_translate, name)
    raise AttributeError(name)
