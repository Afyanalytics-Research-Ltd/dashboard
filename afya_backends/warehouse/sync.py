"""Copy schemas from Snowflake into Databend (structure + data).

For the objects only Snowflake can build (e.g. HOSPITALS.STAGING /
REPORTING, created inside Snowflake) and history loaded before the switch:
once Snowflake is reachable again, run

    python manage.py sync_snowflake_to_databend --schemas STAGING,REPORTING
    python manage.py sync_snowflake_to_databend --all            # every schema
    python manage.py sync_snowflake_to_databend --schemas RAW --tables EVENTS_RAW --replace

Per table: the Snowflake column list (INFORMATION_SCHEMA.COLUMNS) becomes a
Databend CREATE TABLE with mapped types; rows stream across with
``fetch_pandas_batches`` -> CSV streaming load.  Views are recreated from
their definition through the Snowflake->Databend translator, or - when the
definition can't be translated - materialised as a table.  Progress is
recorded in ``.snowflake_to_databend_sync.json`` (next to this file's
caller's working dir) so an interrupted run resumes; ``--replace`` reloads
tables already done.

Reads Snowflake through snowflake.connector directly (never the switch), and
writes Databend directly, so it works whichever way the switch points.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

from . import sql_translate as tr

logger = logging.getLogger(__name__)

STATE_FILE = Path(".snowflake_to_databend_sync.json")


def databend_type(data_type: str, precision, scale) -> str:
    t = (data_type or "").upper()
    if t in ("NUMBER", "DECIMAL", "NUMERIC", "FIXED"):
        p = int(precision or 38)
        s = int(scale or 0)
        if s == 0 and p <= 18:
            return "BIGINT"
        return f"DECIMAL({min(p, 76)}, {s})"
    if t in ("FLOAT", "FLOAT4", "FLOAT8", "DOUBLE", "DOUBLE PRECISION", "REAL"):
        return "DOUBLE"
    if t in ("TEXT", "VARCHAR", "STRING", "CHAR", "CHARACTER"):
        return "VARCHAR"
    if t == "BOOLEAN":
        return "BOOLEAN"
    if t == "DATE":
        return "DATE"
    if t.startswith("TIMESTAMP") or t == "DATETIME":
        return "TIMESTAMP"
    if t == "TIME":
        return "VARCHAR"
    if t in ("VARIANT", "OBJECT", "ARRAY", "MAP"):
        return "VARIANT"
    if t in ("BINARY", "VARBINARY"):
        return "BINARY"
    if t in ("GEOGRAPHY", "GEOMETRY"):
        return "VARCHAR"
    return "VARCHAR"


def _q(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


class Syncer:
    def __init__(self, sf_conn, db_conn, database: str = "HOSPITALS", batch_rows: int = 200_000,
                 replace: bool = False, state_file: Path = STATE_FILE, log: Callable[[str], None] = None):
        self.sf = sf_conn
        self.db = db_conn          # afya_backends.warehouse.dbapi.Connection
        self.database = database.upper()
        self.batch_rows = batch_rows
        self.replace = replace
        self.state_file = Path(state_file)
        self.log = log or logger.info
        self.state = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}

    def _save(self):
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, sort_keys=True))
        tmp.replace(self.state_file)

    def _sf(self, sql, params=None):
        cur = self.sf.cursor()
        cur.execute(sql, params)
        return cur

    def schemas(self) -> list[str]:
        rows = self._sf(f"SELECT SCHEMA_NAME FROM {self.database}.INFORMATION_SCHEMA.SCHEMATA "
                        "WHERE SCHEMA_NAME <> 'INFORMATION_SCHEMA' ORDER BY 1").fetchall()
        return [r[0] for r in rows]

    def objects(self, schema: str, only: Optional[Iterable[str]] = None):
        rows = self._sf(f"SELECT TABLE_NAME, TABLE_TYPE FROM {self.database}.INFORMATION_SCHEMA.TABLES "
                        "WHERE TABLE_SCHEMA = %s ORDER BY TABLE_TYPE, TABLE_NAME", (schema,)).fetchall()
        wanted = {t.upper() for t in only} if only else None
        return [(n, t) for n, t in rows if wanted is None or n.upper() in wanted]

    def columns(self, schema: str, table: str):
        return self._sf(f"SELECT COLUMN_NAME, DATA_TYPE, NUMERIC_PRECISION, NUMERIC_SCALE "
                        f"FROM {self.database}.INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = %s "
                        "AND TABLE_NAME = %s ORDER BY ORDINAL_POSITION", (schema, table)).fetchall()

    def sync_schema(self, schema: str, only=None) -> dict:
        target_db = tr.map_database(self.database, schema, tr.default_primary_database())
        self.db._t.run(f"CREATE DATABASE IF NOT EXISTS {_q(target_db)}")
        summary = {"tables": 0, "views": 0, "rows": 0, "skipped": 0, "failed": []}
        objs = self.objects(schema, only)
        # Tables first, then views (which may select from them).
        for name, kind in sorted(objs, key=lambda o: o[1] == "VIEW"):
            key = f"{schema}.{name}"
            if self.state.get(key, {}).get("done") and not self.replace:
                summary["skipped"] += 1
                continue
            try:
                if kind == "VIEW":
                    how = self.sync_view(schema, name, target_db)
                    summary["views"] += 1
                    self.state[key] = {"done": True, "kind": "view", "how": how, "at": time.time()}
                else:
                    n = self.sync_table(schema, name, target_db)
                    summary["tables"] += 1
                    summary["rows"] += n
                    self.state[key] = {"done": True, "kind": "table", "rows": n, "at": time.time()}
            except Exception as exc:  # keep going; report at the end
                logger.exception("sync %s failed", key)
                summary["failed"].append(f"{key}: {str(exc)[:200]}")
                self.state[key] = {"done": False, "error": str(exc)[:500], "at": time.time()}
            self._save()
        return summary

    def sync_table(self, schema: str, table: str, target_db: str, source_sql: Optional[str] = None) -> int:
        cols = self.columns(schema, table)
        if not cols:
            raise RuntimeError("no columns visible (permissions?)")
        defs = ", ".join(f"{_q(c[0])} {databend_type(c[1], c[2], c[3])}" for c in cols)
        target = f"{_q(target_db)}.{_q(table)}"
        self.db._t.run(f"CREATE OR REPLACE TABLE {target} ({defs})")
        cur = self.sf.cursor()
        cur.execute(source_sql or f'SELECT * FROM {self.database}.{_q(schema)}.{_q(table)}')
        from .pandas_tools import write_pandas
        total = 0
        for df in cur.fetch_pandas_batches():
            if df.empty:
                continue
            for start in range(0, len(df), self.batch_rows):
                part = df.iloc[start:start + self.batch_rows]
                write_pandas(self.db, part, table, database=self.database, schema=schema, quote_identifiers=True)
                total += len(part)
            self.log(f"  {schema}.{table}: {total:,} rows")
        sf_count = self._sf(f'SELECT COUNT(*) FROM {self.database}.{_q(schema)}.{_q(table)}').fetchone()[0] \
            if source_sql is None else total
        db_count = int(self.db._t.run(f"SELECT COUNT(*) FROM {target}").rows[0][0])
        if db_count != sf_count:
            raise RuntimeError(f"row count mismatch: snowflake={sf_count} databend={db_count}")
        self.log(f"✓ {schema}.{table}: {db_count:,} rows")
        return db_count

    def sync_view(self, schema: str, view: str, target_db: str) -> str:
        row = self._sf(f"SELECT VIEW_DEFINITION FROM {self.database}.INFORMATION_SCHEMA.VIEWS "
                       "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s", (schema, view)).fetchone()
        definition = row[0] if row else None
        if definition:
            try:
                ctx = tr.Context(database=self.database, schema=schema, primary_database=tr.default_primary_database())
                statements = [s for s in tr.translate(definition, ctx) if s.action == "sql"]
                if len(statements) == 1 and statements[0].sql.upper().startswith("CREATE"):
                    sql = statements[0].sql
                    if " OR REPLACE " not in sql.upper()[:40]:
                        sql = sql.replace("CREATE ", "CREATE OR REPLACE ", 1)
                    self.db._t.run(sql)
                    self.db._t.run(f"SELECT * FROM {_q(target_db)}.{_q(view)} LIMIT 0")
                    self.log(f"✓ view {schema}.{view} recreated")
                    return "view"
            except Exception as exc:
                self.log(f"  view {schema}.{view}: definition not portable ({str(exc)[:120]}); materialising")
                try:
                    self.db._t.run(f"DROP VIEW IF EXISTS {_q(target_db)}.{_q(view)}")
                except Exception:
                    pass
        n = self.sync_table(schema, view, target_db)
        self.log(f"✓ view {schema}.{view} materialised as a table ({n:,} rows)")
        return "materialised"
