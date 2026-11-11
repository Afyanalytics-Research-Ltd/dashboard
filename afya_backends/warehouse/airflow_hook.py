"""Drop-in replacement for Airflow's ``SnowflakeHook``.

    from afya_backends.warehouse.airflow_hook import WarehouseHook as SnowflakeHook

``WarehouseHook(snowflake_conn_id=...)`` returns the real SnowflakeHook when
the warehouse switch (or the DAG run's ``warehouse_backend`` param) says
Snowflake, and a :class:`DatabendHook` with the same methods otherwise.
The decision is made when the hook is created - i.e. inside the task, at
run time - never at DAG-parse time.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Optional

from .. import toggles

logger = logging.getLogger(__name__)


def WarehouseHook(*args, **kwargs):  # noqa: N802 - mimics a class
    if toggles.warehouse_backend() == toggles.DATABEND:
        return DatabendHook(*args, **kwargs)
    from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
    return SnowflakeHook(*args, **kwargs)


def fetch_all_handler(cursor):
    return cursor.fetchall() if cursor.description is not None else None


class DatabendHook:
    """The subset of SnowflakeHook/DbApiHook the DAGs use, on Databend."""

    conn_name_attr = "snowflake_conn_id"
    default_conn_name = "snowflake_default"
    hook_name = "Databend (Snowflake-compatible)"

    def __init__(self, *args, snowflake_conn_id: str = "snowflake_default", database: Optional[str] = None,
                 schema: Optional[str] = None, warehouse=None, role=None, session_parameters=None, **kwargs):
        self.snowflake_conn_id = snowflake_conn_id
        self.database = database
        self.schema = schema
        self.query_ids: list[str] = []
        self.last_description = None
        self._autocommit = True
        if self.database is None or self.schema is None:
            extra = self._connection_extra()
            self.database = self.database or extra.get("database")
            self.schema = self.schema or extra.get("schema")

    def _connection_extra(self) -> dict:
        try:
            from airflow.sdk import BaseHook
        except ImportError:  # pragma: no cover
            from airflow.hooks.base import BaseHook
        try:
            c = BaseHook.get_connection(self.snowflake_conn_id)
        except Exception:
            return {}
        extra = dict(c.extra_dejson or {})
        out = {"database": extra.get("database") or extra.get("extra__snowflake__database"),
               "schema": c.schema or extra.get("schema") or extra.get("extra__snowflake__schema")}
        return {k: v for k, v in out.items() if v}

    def get_conn(self):
        from .dbapi import Connection
        return Connection(database=self.database, schema=self.schema)

    def get_autocommit(self, conn) -> bool:
        return getattr(conn, "_autocommit", True)

    def set_autocommit(self, conn, autocommit: bool) -> None:
        conn.autocommit(autocommit)

    def run(self, sql, autocommit: bool = False, parameters=None, handler: Optional[Callable] = None,
            split_statements: bool = True, return_last: bool = True, return_dictionaries: bool = False):
        from .sql_translate import split_statements as _split
        statements = [sql] if isinstance(sql, str) else list(sql)
        if split_statements and isinstance(sql, str):
            statements = _split(sql)
        conn = self.get_conn()
        results = []
        try:
            conn.autocommit(autocommit)
            for stmt in statements:
                cur = conn.cursor()
                cur.execute(stmt, parameters)
                if cur.sfqid:
                    self.query_ids.append(cur.sfqid)
                self.last_description = cur.description
                if handler is not None:
                    res = handler(cur)
                    if return_dictionaries and res is not None and cur.description:
                        names = [d[0] for d in cur.description]
                        res = [dict(zip(names, r)) for r in res] if isinstance(res, list) else dict(zip(names, res))
                    results.append(res)
            if not autocommit:
                conn.commit()
        finally:
            conn.close()
        if handler is None:
            return None
        if return_last or (isinstance(sql, str) and len(statements) == 1):
            return results[-1] if results else None
        return results

    def get_records(self, sql, parameters=None, **_kw):
        return self.run(sql, parameters=parameters, handler=fetch_all_handler, autocommit=True)

    def get_first(self, sql, parameters=None, **_kw):
        return self.run(sql, parameters=parameters, handler=lambda c: c.fetchone(), autocommit=True)

    def get_pandas_df(self, sql, parameters=None, **_kw):
        conn = self.get_conn()
        try:
            cur = conn.cursor()
            cur.execute(sql, parameters)
            return cur.fetch_pandas_all()
        finally:
            conn.close()

    def get_df(self, sql, parameters=None, *, df_type: str = "pandas", **kw):
        return self.get_pandas_df(sql, parameters)

    def insert_rows(self, table: str, rows: Iterable, target_fields=None, commit_every: int = 1000,
                    replace: bool = False, **kwargs):
        rows = [tuple(r) for r in rows]
        if not rows:
            return
        cols = f" ({', '.join(target_fields)})" if target_fields else ""
        placeholders = ", ".join(["%s"] * len(rows[0]))
        conn = self.get_conn()
        try:
            cur = conn.cursor()
            cur.executemany(f"INSERT INTO {table}{cols} VALUES ({placeholders})", rows)
        finally:
            conn.close()

    def copy_file_to_stage(self, file_path: str, stage: str, file_name: Optional[str] = None, **_kw) -> None:
        """Upload a local file to a (Snowflake-named) stage on Databend, so
        a following ``... FROM @stage/<file>`` reads it."""
        from .dbapi import Connection
        from .sql_translate import _stage_name
        name = _stage_name("@" + stage)
        conn = self.get_conn()
        try:
            conn._ensure_stage(name)
            import os
            with open(file_path, "rb") as fh:
                conn._t.upload_to_stage(name, file_path, file_name or os.path.basename(file_path), fh)
        finally:
            conn.close()

    def test_connection(self):
        try:
            self.get_first("SELECT 1")
            return True, "Connection successfully tested (Databend)"
        except Exception as exc:
            return False, str(exc)

    def get_sqlalchemy_engine(self, *a, **kw):
        raise NotImplementedError("SQLAlchemy engines are not available for the Databend warehouse; "
                                  "use get_conn()/get_pandas_df() instead")

    def get_uri(self) -> str:
        from .dbapi import _settings_url
        return _settings_url()
