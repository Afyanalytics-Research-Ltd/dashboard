"""Two-way table sync between Snowflake and Databend.

    from afya_backends.warehouse import bisync
    report = bisync.run(sf_side, db_side, state_path, allow_shrink=False, dry_run=False)

Each side is a ``Side``: a connection plus the Snowflake-style database it
represents (``HOSPITALS`` -> Databend databases named after the schema,
``OTHER`` -> ``OTHER__<SCHEMA>``; see sql_translate.map_database).  Only base
tables are synced; views are not.

Per table (SCHEMA.TABLE), against the state of the last successful sync:

    only on one side                   -> copy it across (full)
    only one side changed              -> replace the other side's copy (full)
    both changed / first sight & the
      row counts differ                -> RECONCILE: with a key column (ID, UUID, _ID)
                                          insert each side's missing keys into the
                                          other (never deletes, never overwrites);
                                          without one -> conflict, left alone
    unchanged                          -> nothing

A full copy loads a staging table, verifies its row count, then swaps it in,
so a failed copy never leaves a half-written table. It never SHRINKS the
target (fewer rows than it has now) unless allow_shrink=True: a copy that
would is reconciled instead. That stops e.g. a Databend table that was
written fresh during a Snowflake outage (3 rows) from wiping the
Snowflake history (millions of rows).

"Changed" = the table's (row count, last-modified) fingerprint differs from
the one recorded after the last sync. State is a JSON file; if it is lost,
every table is "first sight": equal row counts are adopted as in sync,
different ones reconciled - safe, just slower.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import sql_translate as tr
from .sync import databend_type

logger = logging.getLogger(__name__)

KEY_CANDIDATES = ("ID", "UUID", "_ID")
# Append-only tables (every run adds rows under a new key, nothing is updated):
# they are always reconciled by that key - only new keys copied, both ways -
# instead of being fully re-copied whenever they grow. Extend at run time with
# BISYNC_APPEND_KEYS="SCHEMA.TABLE=COLUMN,...".
APPEND_KEYS = {
    "ANALYTICS.AFYA_HMIS_PATIENT_COUNTS": "RUN_ID",   # a snapshot per run, every 15 min
}
STAGING_SUFFIX = "__BISYNC_TMP"
OLD_SUFFIX = "__BISYNC_OLD"
DATABEND_SKIP_DBS = {"system", "information_schema", "default", tr.COMPAT_DB.lower()}


def _q(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _bq(name: str) -> str:
    return "`" + str(name).replace("`", "``") + "`"


# ── column types ─────────────────────────────────────────────────────────
@dataclass
class Col:
    name: str
    kind: str          # neutral: NUMBER FLOAT TEXT BOOLEAN DATE TIMESTAMP VARIANT BINARY
    precision: Optional[int] = None
    scale: Optional[int] = None

    def snowflake(self) -> str:
        if self.kind == "NUMBER":
            p, s = min(int(self.precision or 38), 38), int(self.scale or 0)
            return f"NUMBER({p}, {min(s, p)})"
        return {"FLOAT": "FLOAT", "TEXT": "VARCHAR", "BOOLEAN": "BOOLEAN", "DATE": "DATE",
                "TIMESTAMP": "TIMESTAMP_NTZ", "VARIANT": "VARIANT", "BINARY": "BINARY"}.get(self.kind, "VARCHAR")

    def databend(self) -> str:
        sf = {"NUMBER": "NUMBER", "FLOAT": "FLOAT", "TEXT": "VARCHAR", "BOOLEAN": "BOOLEAN", "DATE": "DATE",
              "TIMESTAMP": "TIMESTAMP_NTZ", "VARIANT": "VARIANT", "BINARY": "BINARY"}.get(self.kind, "VARCHAR")
        return databend_type(sf, self.precision, self.scale)


def col_from_snowflake(name, data_type, precision, scale) -> Col:
    t = (data_type or "").upper()
    if t in ("NUMBER", "DECIMAL", "NUMERIC", "FIXED"):
        return Col(name, "NUMBER", precision, scale)
    if t in ("FLOAT", "FLOAT4", "FLOAT8", "DOUBLE", "DOUBLE PRECISION", "REAL"):
        return Col(name, "FLOAT")
    if t == "BOOLEAN":
        return Col(name, "BOOLEAN")
    if t == "DATE":
        return Col(name, "DATE")
    if t.startswith("TIMESTAMP") or t == "DATETIME":
        return Col(name, "TIMESTAMP")
    if t in ("VARIANT", "OBJECT", "ARRAY", "MAP"):
        return Col(name, "VARIANT")
    if t in ("BINARY", "VARBINARY"):
        return Col(name, "BINARY")
    return Col(name, "TEXT")


_DEC = re.compile(r"DECIMAL\((\d+),\s*(\d+)\)")


def col_from_databend(name, type_str) -> Col:
    t = (type_str or "").upper().strip()
    t = re.sub(r"^NULLABLE\((.*)\)$", r"\1", t).strip()
    t = re.sub(r"\s+NULL$", "", t).replace(" NOT NULL", "").strip()
    m = _DEC.match(t)
    if m:
        return Col(name, "NUMBER", int(m.group(1)), int(m.group(2)))
    if re.match(r"^U?INT(8|16|32|64)?$|^(TINYINT|SMALLINT|INTEGER|BIGINT)( UNSIGNED)?$", t):
        return Col(name, "NUMBER", 38, 0)
    if t in ("FLOAT", "FLOAT32", "FLOAT64", "DOUBLE", "REAL"):
        return Col(name, "FLOAT")
    if t in ("BOOLEAN", "BOOL"):
        return Col(name, "BOOLEAN")
    if t == "DATE":
        return Col(name, "DATE")
    if t.startswith("TIMESTAMP") or t == "DATETIME":
        return Col(name, "TIMESTAMP")
    if t.startswith(("VARIANT", "JSON", "ARRAY", "MAP", "TUPLE")):
        return Col(name, "VARIANT")
    if t.startswith("BINARY"):
        return Col(name, "BINARY")
    return Col(name, "TEXT")


# ── one side of the sync ─────────────────────────────────────────────────
@dataclass
class TableInfo:
    schema: str
    name: str
    rows: int
    modified: str

    @property
    def fingerprint(self) -> str:
        return f"{self.rows}|{self.modified}"


@dataclass
class Side:
    """``backend`` is "snowflake" or "databend"; ``conn`` a snowflake.connector
    connection or an afya_backends.warehouse.dbapi.Connection opened on
    ``database``."""
    backend: str
    conn: object
    database: str = "HOSPITALS"
    schemas: Optional[set] = None                 # limit to these schemas (upper-case)
    _primary: str = field(default_factory=tr.default_primary_database)
    # (SCHEMA, TABLE) upper-case -> this side's own spelling, filled by tables().
    # Tables are matched case-insensitively across sides, but each side's
    # statements must use its own spelling (Databend names are case-sensitive:
    # TENRI.ACTIVATIONS in Snowflake can be TENRI.activations in Databend).
    _names: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.backend}:{self.database}"

    def real(self, schema: str, table: str) -> tuple[str, str]:
        """This side's spelling of SCHEMA.TABLE (as given if this side doesn't have it)."""
        return self._names.get((str(schema).upper(), str(table).upper()), (schema, table))

    # raw SQL in the side's own dialect
    def run(self, sql: str, params=None) -> list:
        if self.backend == "databend":
            return [list(r) for r in self.conn._t.run(sql).rows]
        cur = self.conn.cursor()
        try:
            cur.execute(sql, params)
            try:
                return cur.fetchall()
            except Exception:
                return []
        finally:
            cur.close()

    def bend_db(self, schema: str) -> str:
        return tr.map_database(self.database, schema, self._primary)

    def fq(self, schema: str, table: str) -> str:
        if self.backend == "databend":
            return f"{_bq(self.bend_db(schema))}.{_bq(table)}"
        return f"{_q(self.database)}.{_q(schema)}.{_q(table)}"

    def tables(self) -> dict[tuple, TableInfo]:
        out = {}
        if self.backend == "snowflake":
            rows = self.run(f"SELECT TABLE_SCHEMA, TABLE_NAME, ROW_COUNT, TO_VARCHAR(LAST_ALTERED) "
                            f"FROM {_q(self.database)}.INFORMATION_SCHEMA.TABLES "
                            "WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_SCHEMA <> 'INFORMATION_SCHEMA'")
            for s, n, r, m in rows:
                out[(s.upper(), n.upper())] = TableInfo(s, n, int(r or 0), str(m))
        else:
            rows = self.run("SELECT database, name, num_rows, updated_on FROM system.tables "
                            "WHERE engine NOT IN ('VIEW', 'STREAM') AND is_external = false")
            prefix = f"{self.database.upper()}__"
            primary = self.database.upper() == self._primary.upper()
            for db, n, r, m in rows:
                if db.lower() in DATABEND_SKIP_DBS:
                    continue
                if primary and "__" not in db:
                    schema = db
                elif db.upper().startswith(prefix):
                    schema = db[len(prefix):]
                else:
                    continue
                out[(schema.upper(), n.upper())] = TableInfo(schema, n, int(r or 0), str(m))
        out = {k: v for k, v in out.items()
               if not k[1].endswith((STAGING_SUFFIX, OLD_SUFFIX))
               and (self.schemas is None or k[0] in self.schemas)}
        self._names = {k: (v.schema, v.name) for k, v in out.items()}
        return out

    def views(self) -> dict[tuple, str]:
        """{(SCHEMA, VIEW) upper: definition} - Snowflake; {(SCHEMA, VIEW) upper: ''} - Databend."""
        out = {}
        if self.backend == "snowflake":
            for s, n, d in self.run(f"SELECT TABLE_SCHEMA, TABLE_NAME, VIEW_DEFINITION "
                                    f"FROM {_q(self.database)}.INFORMATION_SCHEMA.VIEWS "
                                    "WHERE TABLE_SCHEMA <> 'INFORMATION_SCHEMA'"):
                out[(s.upper(), n.upper())] = d or ""
        else:
            primary = self.database.upper() == self._primary.upper()
            prefix = f"{self.database.upper()}__"
            for db, n in self.run("SELECT database, name FROM system.tables WHERE engine = 'VIEW'"):
                if db.lower() in DATABEND_SKIP_DBS:
                    continue
                if primary and "__" not in db:
                    out[(db.upper(), n.upper())] = ""
                elif db.upper().startswith(prefix):
                    out[(db[len(prefix):].upper(), n.upper())] = ""
        return {k: v for k, v in out.items() if self.schemas is None or k[0] in self.schemas}

    def columns(self, schema: str, table: str) -> list[Col]:
        if self.backend == "snowflake":
            rows = self.run(f"SELECT COLUMN_NAME, DATA_TYPE, NUMERIC_PRECISION, NUMERIC_SCALE "
                            f"FROM {_q(self.database)}.INFORMATION_SCHEMA.COLUMNS "
                            "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s ORDER BY ORDINAL_POSITION",
                            (schema, table))
            return [col_from_snowflake(*r) for r in rows]
        # DESC: Field, Type, Null, Default, Extra - in column order
        return [col_from_databend(r[0], r[1]) for r in self.run(f"DESC {self.fq(schema, table)}")]

    def info(self, schema: str, table: str) -> Optional[TableInfo]:
        """Current fingerprint of one table (None if it doesn't exist)."""
        def lit(s):
            return "'" + str(s).replace("'", "''") + "'"
        if self.backend == "snowflake":
            rows = self.run(f"SELECT TABLE_SCHEMA, TABLE_NAME, ROW_COUNT, TO_VARCHAR(LAST_ALTERED) "
                            f"FROM {_q(self.database)}.INFORMATION_SCHEMA.TABLES "
                            "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s", (schema, table))
        else:
            rows = self.run(f"SELECT database, name, num_rows, updated_on FROM system.tables "
                            f"WHERE database = {lit(self.bend_db(schema))} AND name = {lit(table)}")
        if not rows:
            return None
        return TableInfo(schema, table, int(rows[0][2] or 0), str(rows[0][3]))

    def count(self, schema: str, table: str) -> int:
        return int(self.run(f"SELECT COUNT(*) FROM {self.fq(schema, table)}")[0][0])

    def ensure_schema(self, schema: str) -> None:
        if self.backend == "databend":
            self.run(f"CREATE DATABASE IF NOT EXISTS {_bq(self.bend_db(schema))}")
        else:
            self.run(f"CREATE SCHEMA IF NOT EXISTS {_q(self.database)}.{_q(schema)}")

    def create_table(self, schema: str, table: str, cols: list[Col], *, variant_as_text: bool = False) -> None:
        if self.backend == "databend":
            defs = ", ".join(f"{_bq(c.name)} {c.databend()} NULL" for c in cols)
            self.run(f"CREATE OR REPLACE TABLE {self.fq(schema, table)} ({defs})")
        else:
            defs = ", ".join(f"{_q(c.name)} {'VARCHAR' if variant_as_text and c.kind == 'VARIANT' else c.snowflake()}"
                             for c in cols)
            self.run(f"CREATE OR REPLACE TABLE {self.fq(schema, table)} ({defs})")

    def drop(self, schema: str, table: str) -> None:
        self.run(f"DROP TABLE IF EXISTS {self.fq(schema, table)}")

    def swap_in(self, schema: str, staging: str, table: str, exists: bool) -> None:
        """Make ``staging`` become ``table``."""
        if self.backend == "snowflake":
            if exists:
                self.run(f"ALTER TABLE {self.fq(schema, table)} SWAP WITH {self.fq(schema, staging)}")
                self.drop(schema, staging)
            else:
                self.run(f"ALTER TABLE {self.fq(schema, staging)} RENAME TO {self.fq(schema, table)}")
            return
        if exists:
            old = table + OLD_SUFFIX
            self.drop(schema, old)
            self.run(f"RENAME TABLE {self.fq(schema, table)} TO {self.fq(schema, old)}")
            self.run(f"RENAME TABLE {self.fq(schema, staging)} TO {self.fq(schema, table)}")
            self.drop(schema, old)
        else:
            self.run(f"RENAME TABLE {self.fq(schema, staging)} TO {self.fq(schema, table)}")

    def read_batches(self, schema: str, table: str, columns: Optional[list[str]] = None,
                     where: str = ""):
        """``where``: a Snowflake-dialect condition (the Databend cursor translates it)."""
        cols = ", ".join(_q(c) for c in columns) if columns else "*"
        cond = f" WHERE {where}" if where else ""
        cur = self.conn.cursor()
        if self.backend == "databend":   # dbapi translates Snowflake-dialect SQL to this side's names
            cur.execute(f"SELECT {cols} FROM {_q(schema)}.{_q(table)}{cond}")
        else:
            cur.execute(f"SELECT {cols} FROM {self.fq(schema, table)}{cond}")
        yield from cur.fetch_pandas_batches()

    def write(self, schema: str, table: str, df) -> None:
        from . import pandas_tools
        kw = {"use_logical_type": True} if self.backend == "snowflake" else {}
        ok, _, nrows, _ = pandas_tools.write_pandas(self.conn, df, table, database=self.database, schema=schema,
                                                    quote_identifiers=True, **kw)[:4]
        if not ok:
            raise RuntimeError(f"write_pandas into {self.label}.{schema}.{table} reported failure")

    def keys(self, schema: str, table: str, key: str) -> set:
        # DISTINCT: an append-only snapshot table keyed by RUN_ID has ~1000 rows
        # per key; only the keys are needed.
        rows = self.run(f"SELECT DISTINCT {_q(key) if self.backend == 'snowflake' else _bq(key)} "
                        f"FROM {self.fq(schema, table)}")
        return {None if r[0] is None else str(r[0]) for r in rows}


# ── copy / reconcile ─────────────────────────────────────────────────────
def _prepare(df, cols: list[Col], target: Side):
    """Make a pandas batch loadable on the target."""
    import json as _json
    df = df[[c.name for c in cols if c.name in df.columns]]
    for c in cols:
        if c.name in df.columns and c.kind == "VARIANT":
            df[c.name] = df[c.name].map(
                lambda v: v if v is None or isinstance(v, str) else _json.dumps(v, default=str))
    return df


def _dst_name(name: str, dst: Side) -> str:
    # Databend folds unquoted identifiers to lower case; Snowflake to upper. An
    # all-lower-case Databend column would otherwise become a quoted "id" in
    # Snowflake that every query has to quote.
    return name.upper() if dst.backend == "snowflake" and name == name.lower() else name


def copy_table(src: Side, dst: Side, schema: str, table: str, *, dst_exists: bool, log: Callable) -> int:
    """Full copy src -> dst via a verified staging table + swap. Returns rows."""
    s_schema, s_table = src.real(schema, table)
    d_schema, d_table = dst.real(schema, table) if dst_exists else (s_schema, s_table)
    src_cols = src.columns(s_schema, s_table)
    if not src_cols:
        raise RuntimeError(f"{src.label}: no columns visible for {schema}.{table}")
    cols = [Col(_dst_name(c.name, dst), c.kind, c.precision, c.scale) for c in src_cols]
    rename = {s.name: d.name for s, d in zip(src_cols, cols)}
    dst.ensure_schema(d_schema)
    staging = d_table + STAGING_SUFFIX
    variant_cols = [c for c in cols if c.kind == "VARIANT"]
    sf_variant_fix = dst.backend == "snowflake" and bool(variant_cols)
    dst.create_table(d_schema, staging, cols, variant_as_text=sf_variant_fix)
    total = 0
    try:
        for df in src.read_batches(s_schema, s_table, [c.name for c in src_cols]):
            if df.empty:
                continue
            dst.write(d_schema, staging, _prepare(df.rename(columns=rename), cols, dst))
            total += len(df)
        if sf_variant_fix:   # loaded as text; parse into real VARIANT
            exprs = ", ".join(f"TRY_PARSE_JSON({_q(c.name)}) AS {_q(c.name)}" if c.kind == "VARIANT"
                              else _q(c.name) for c in cols)
            dst.run(f"CREATE OR REPLACE TABLE {dst.fq(d_schema, staging)} AS SELECT {exprs} "
                    f"FROM {dst.fq(d_schema, staging)}")
        got = dst.count(d_schema, staging)
        if got != total:
            raise RuntimeError(f"staging row count {got} != {total} read from {src.label}")
        dst.swap_in(d_schema, staging, d_table, dst_exists)
    except Exception:
        try:
            dst.drop(d_schema, staging)
        except Exception:  # noqa: BLE001
            pass
        raise
    log(f"  copied {schema}.{table} {src.backend} -> {dst.backend}: {total:,} rows")
    return total


@dataclass
class Bulk:
    """Server-side Snowflake -> Databend path for big tables: Snowflake unloads
    parquet to ``stage`` (an external stage on s3://bucket/prefix, via a storage
    integration - no AWS secret in Snowflake), Databend COPYs the files in and
    purges them. Set up once with pipelines/setup_bisync_bulk.py."""
    stage: str                      # e.g. HOSPITALS.ANALYTICS.BISYNC_STAGE
    s3_url: str                     # the stage's URL, e.g. s3://bucket/_bisync_stage/
    key_id: str
    secret: str
    region: str = "eu-north-1"
    min_rows: int = 200_000

    def available(self, sf: "Side") -> bool:
        try:
            sf.run(f"DESC STAGE {self.stage}")
            return bool(self.key_id and self.secret)
        except Exception as e:  # noqa: BLE001
            logger.warning("bulk path off - stage %s unusable (%s)", self.stage, str(e)[:150])
            return False


def bulk_copy_sf_to_db(sf: Side, db: Side, schema: str, table: str, bulk: Bulk, *, dst_exists: bool,
                       log: Callable) -> int:
    """Snowflake COPY INTO @stage (parquet) -> Databend COPY INTO staging -> verify -> swap."""
    import uuid
    d_schema, d_table = db.real(schema, table) if dst_exists else sf.real(schema, table)
    schema, table = sf.real(schema, table)
    cols = sf.columns(schema, table)
    if not cols:
        raise RuntimeError(f"{sf.label}: no columns visible for {schema}.{table}")
    path = f"{schema}/{table}/{uuid.uuid4().hex[:12]}/"
    # Plain parquet types only: text as VARCHAR (TIME/GEOGRAPHY too), VARIANT as JSON text,
    # every timestamp flavour as NTZ.
    exprs = []
    for c in cols:
        q = _q(c.name)
        if c.kind == "VARIANT":
            exprs.append(f"TO_JSON({q}) AS {q}")
        elif c.kind == "TEXT":
            exprs.append(f"TO_VARCHAR({q}) AS {q}")
        elif c.kind == "TIMESTAMP":
            exprs.append(f"{q}::TIMESTAMP_NTZ AS {q}")
        else:
            exprs.append(q)
    out = sf.run(f"COPY INTO @{bulk.stage}/{path} FROM (SELECT {', '.join(exprs)} FROM {sf.fq(schema, table)}) "
                 "FILE_FORMAT = (TYPE = PARQUET) HEADER = TRUE MAX_FILE_SIZE = 268435456")
    unloaded = int(sum(r[0] for r in out)) if out else 0

    db.ensure_schema(d_schema)
    staging, raw = d_table + STAGING_SUFFIX, d_table + STAGING_SUFFIX + "_RAW"
    variant = any(c.kind == "VARIANT" for c in cols)
    url = bulk.s3_url.rstrip("/") + "/" + path
    conn = (f"CONNECTION = (ACCESS_KEY_ID = '{bulk.key_id}', SECRET_ACCESS_KEY = '{bulk.secret}', "
            f"REGION = '{bulk.region}')")
    try:
        db.create_table(d_schema, staging, cols)
        load_into = staging
        if variant:  # parquet carries JSON text; load as text, then parse
            db.create_table(d_schema, raw, [Col(c.name, "TEXT" if c.kind == "VARIANT" else c.kind, c.precision,
                                                c.scale) for c in cols])
            load_into = raw
        if unloaded:
            # File, Rows_loaded, Errors_seen, First_error, First_error_line
            res = db.run(f"COPY INTO {db.fq(d_schema, load_into)} FROM '{url}' {conn} PATTERN = '.*[.]parquet' "
                         "FILE_FORMAT = (TYPE = PARQUET) PURGE = TRUE FORCE = TRUE")
            loaded = sum(int(r[1] or 0) for r in res)
            errs = [f"{r[0]}: {r[3]}" for r in res if r[3]]
            if loaded != unloaded or errs:
                raise RuntimeError(f"databend COPY loaded {loaded} of {unloaded} rows from {len(res)} file(s); "
                                   f"errors: {errs[:3]}")
        if variant:
            sel = ", ".join(f"PARSE_JSON({_bq(c.name)})" if c.kind == "VARIANT" else _bq(c.name) for c in cols)
            db.run(f"INSERT INTO {db.fq(d_schema, staging)} SELECT {sel} FROM {db.fq(d_schema, raw)}")
            db.drop(d_schema, raw)
        got = db.count(d_schema, staging)
        if got != unloaded:
            raise RuntimeError(f"staging row count {got} != {unloaded} unloaded from Snowflake")
        db.swap_in(d_schema, staging, d_table, dst_exists)
    except Exception as e:
        for t in (staging, raw):
            try:
                db.drop(d_schema, t)
            except Exception:  # noqa: BLE001
                pass
        raise RuntimeError(str(e).replace(bulk.secret, "***") if bulk.secret else str(e)) from None
    finally:
        try:   # PURGE removes loaded files; this catches a failed load
            sf.run(f"REMOVE @{bulk.stage}/{path}")
        except Exception:  # noqa: BLE001
            pass
    log(f"  bulk-copied {schema}.{table} snowflake -> databend: {unloaded:,} rows")
    return unloaded


TARGETED_READ_MAX_KEYS = 20_000   # above this, scan the source and filter instead
_IN_CHUNK = 500


def insert_missing(src: Side, dst: Side, schema: str, table: str, key: str, *, log: Callable) -> int:
    """Append src rows whose key is absent on dst (columns dst lacks are dropped).

    Keys are compared first (SELECT DISTINCT key on both sides), so an
    in-sync table costs two small queries, and when only a few keys are
    missing only their rows are read (WHERE key IN (...)). Rows with a NULL
    key can't be matched and are never copied.
    """
    s_schema, s_table = src.real(schema, table)
    d_schema, d_table = dst.real(schema, table)
    dst_cols = dst.columns(d_schema, d_table)
    dst_names = {c.name.upper(): c.name for c in dst_cols}
    src_cols = [c for c in src.columns(s_schema, s_table) if c.name.upper() in dst_names]
    src_key = next(c.name for c in src_cols if c.name.upper() == key.upper())
    have = dst.keys(d_schema, d_table, dst_names[key.upper()])
    missing = src.keys(s_schema, s_table, src_key) - have
    missing.discard(None)
    if not missing:
        return 0

    rename = {c.name: dst_names[c.name.upper()] for c in src_cols}
    out_cols = [Col(dst_names[c.name.upper()], c.kind) for c in src_cols]
    names = [c.name for c in src_cols]
    added = 0

    def load(batches, wanted):
        nonlocal added
        for df in batches:
            if df.empty:
                continue
            df = df[df[src_key].map(lambda v: None if v is None else str(v)).isin(wanted)]
            if df.empty:
                continue
            dst.write(d_schema, d_table, _prepare(df.rename(columns=rename), out_cols, dst))
            added += len(df)

    if len(missing) <= TARGETED_READ_MAX_KEYS:
        keys = sorted(missing)
        for i in range(0, len(keys), _IN_CHUNK):
            chunk = keys[i:i + _IN_CHUNK]
            lits = ", ".join("'" + k.replace("'", "''") + "'" for k in chunk)
            load(src.read_batches(s_schema, s_table, names, where=f"CAST({_q(src_key)} AS VARCHAR) IN ({lits})"),
                 set(chunk))
        # A key whose text form differs between the warehouses (e.g. a float)
        # isn't found by the targeted read: pick those up with a scan.
        left = missing - dst.keys(d_schema, d_table, dst_names[key.upper()])
        if left:
            load(src.read_batches(s_schema, s_table, names), left)
    else:
        load(src.read_batches(s_schema, s_table, names), missing)
    if added:
        log(f"  reconciled {schema}.{table} {src.backend} -> {dst.backend}: +{added:,} rows (by {key})")
    return added


def append_keys_from_env() -> dict:
    """BISYNC_APPEND_KEYS="SCHEMA.TABLE=COLUMN,..." merged over APPEND_KEYS."""
    import os
    out = dict(APPEND_KEYS)
    for item in (os.getenv("BISYNC_APPEND_KEYS") or "").split(","):
        if "=" in item:
            t, c = item.split("=", 1)
            out[t.strip().upper()] = c.strip().upper()
    return out


def _key_for(a: Side, b: Side, schema: str, table: str, append_keys: Optional[dict] = None) -> Optional[str]:
    names_a = {c.name.upper() for c in a.columns(*a.real(schema, table))}
    names_b = {c.name.upper() for c in b.columns(*b.real(schema, table))}
    override = (append_keys or {}).get(f"{schema}.{table}".upper())
    if override:
        return override if override in names_a and override in names_b else None
    return next((k for k in KEY_CANDIDATES if k in names_a and k in names_b), None)


# ── planning + run ───────────────────────────────────────────────────────
def plan(sf: dict, bend: dict, state: dict, *, allow_shrink: bool,
         append_keys: Optional[dict] = None) -> list[tuple]:
    """[(key, action)] with action in copy_sf_to_db, copy_db_to_sf, reconcile, adopt, noop.
    Pure - unit-testable without any warehouse."""
    append = {k.upper() for k in (append_keys or {})}
    out = []
    for key in sorted(set(sf) | set(bend)):
        a, b = sf.get(key), bend.get(key)
        st = state.get(".".join(key))
        if a and not b:
            out.append((key, "copy_sf_to_db"))
            continue
        if b and not a:
            out.append((key, "copy_db_to_sf"))
            continue
        if st is None:
            out.append((key, "adopt" if a.rows == b.rows else "reconcile"))
            continue
        a_changed, b_changed = a.fingerprint != st.get("sf"), b.fingerprint != st.get("db")
        if not a_changed and not b_changed:
            out.append((key, "noop"))
        elif ".".join(key) in append:
            out.append((key, "reconcile"))       # new keys only, never a full re-copy
        elif a_changed and not b_changed:
            out.append((key, "copy_sf_to_db" if allow_shrink or a.rows >= b.rows else "reconcile"))
        elif b_changed and not a_changed:
            out.append((key, "copy_db_to_sf" if allow_shrink or b.rows >= a.rows else "reconcile"))
        else:
            out.append((key, "reconcile"))
    return out


def run(sf: Side, db: Side, state_path: Path, *, allow_shrink: bool = False, dry_run: bool = False,
        only: Optional[set] = None, log: Callable[[str], None] = None, bulk: Optional[Bulk] = None,
        append_keys: Optional[dict] = None, sync_views: bool = True) -> dict:
    log = log or logger.info
    append_keys = append_keys_from_env() if append_keys is None else append_keys
    if bulk is not None and not dry_run and not bulk.available(sf):
        bulk = None
    log(f"bulk path: {'ON via ' + bulk.stage + f' (tables >= {bulk.min_rows:,} rows)' if bulk else 'off'}")
    state_path = Path(state_path)
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    sf_t, db_t = sf.tables(), db.tables()
    actions = plan(sf_t, db_t, state, allow_shrink=allow_shrink, append_keys=append_keys)
    if only:
        actions = [(k, a) for k, a in actions if ".".join(k) in only or k[1] in only]
    report = {"sf_tables": len(sf_t), "db_tables": len(db_t), "copied_sf_to_db": [], "copied_db_to_sf": [],
              "reconciled": [], "adopted": 0, "unchanged": 0, "conflicts": [], "failed": [], "rows": 0,
              "views_created": [], "views_replaced": [], "views_adopted": 0, "views_failed": []}

    def save():
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
        tmp.replace(state_path)

    for key, action in actions:
        schema, table = key
        name = f"{schema}.{table}"
        a, b = sf_t.get(key), db_t.get(key)
        s_schema, s_table = (a or b).schema, (a or b).name      # side-correct casing
        if action == "noop":
            report["unchanged"] += 1
            continue
        if dry_run:
            log(f"[dry-run] {name}: {action} (sf={a and a.rows} db={b and b.rows})")
            (report["conflicts"] if action == "reconcile" else report.setdefault("planned", [])).append(
                f"{name}: {action}")
            continue
        try:
            if action == "adopt":
                report["adopted"] += 1
            elif action == "copy_sf_to_db":
                if bulk is not None and a.rows >= bulk.min_rows:
                    report["rows"] += bulk_copy_sf_to_db(sf, db, s_schema, s_table, bulk,
                                                         dst_exists=b is not None, log=log)
                else:
                    report["rows"] += copy_table(sf, db, s_schema, s_table, dst_exists=b is not None, log=log)
                report["copied_sf_to_db"].append(name)
            elif action == "copy_db_to_sf":
                report["rows"] += copy_table(db, sf, s_schema, s_table, dst_exists=a is not None, log=log)
                report["copied_db_to_sf"].append(name)
            else:
                k = _key_for(sf, db, s_schema, s_table, append_keys)
                if not k:
                    report["conflicts"].append(f"{name}: both sides differ (sf={a.rows} db={b.rows}) and "
                                               f"no key column {KEY_CANDIDATES} - left alone")
                    continue
                n = insert_missing(sf, db, s_schema, s_table, k, log=log)
                n += insert_missing(db, sf, s_schema, s_table, k, log=log)
                report["rows"] += n
                report["reconciled"].append(f"{name} (+{n})")
            # record both sides' fingerprints as they are NOW (after our writes)
            fresh_a = sf.info(*sf.real(s_schema, s_table)) if action != "adopt" else a
            fresh_b = db.info(*db.real(s_schema, s_table)) if action != "adopt" else b
            state[name] = {"sf": fresh_a and fresh_a.fingerprint, "db": fresh_b and fresh_b.fingerprint,
                           "action": action, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            save()
        except Exception as e:  # noqa: BLE001 - keep going, report at the end
            logger.exception("bisync %s failed", name)
            report["failed"].append(f"{name}: {action}: {str(e)[:300]}")
    if sync_views:
        try:
            views_sf_to_db(sf, db, state, report, dry_run=dry_run, only=only, log=log)
        except Exception as e:  # noqa: BLE001 - views never fail the table sync
            logger.exception("bisync views failed")
            report["views_failed"].append(f"(view sync aborted): {str(e)[:300]}")
        if not dry_run:
            save()
    log(f"bisync: {json.dumps({k: (v if isinstance(v, int) else len(v)) for k, v in report.items()})}")
    return report


# ── views (Snowflake -> Databend, one way) ───────────────────────────────
VIEW_STATE_KEY = "__views__"
VIEW_PAUSE_S = 0.1          # between view DDLs - keep the catalog load trivial
_TEXT_AGG_ERR = re.compile(r"(sum|avg) does not support type '?String", re.I)
_MISSING_DEP_ERR = re.compile(r"Unknown (table|database)", re.I)


def _view_ddl(definition: str, schema: str, database: str, *, coerce: bool = False) -> str:
    ctx = tr.Context(database=database, schema=schema, primary_database=tr.default_primary_database(),
                     coerce_text_aggregates=coerce)
    stmts = [s for s in tr.translate(definition, ctx) if s.action == "sql"]
    if len(stmts) != 1 or not re.match(r"\s*CREATE\b", stmts[0].sql, re.I):
        raise RuntimeError(f"definition translated to {len(stmts)} statement(s), not one CREATE VIEW")
    sql = re.sub(r"^\s*CREATE\s+(OR\s+REPLACE\s+)?(SECURE\s+)?(RECURSIVE\s+)?VIEW\s+(IF\s+NOT\s+EXISTS\s+)?",
                 "CREATE OR REPLACE VIEW ", stmts[0].sql, count=1, flags=re.I)
    if not sql.upper().startswith("CREATE OR REPLACE VIEW"):
        raise RuntimeError("unexpected DDL: " + sql[:80])
    return sql


def views_sf_to_db(sf: Side, db: Side, state: dict, report: dict, *, dry_run: bool = False,
                   only: Optional[set] = None, log: Callable) -> None:
    """Create/refresh Databend views from Snowflake view definitions.

    Views hold no data, so this is DDL only. Per view (state keeps a hash of
    the Snowflake definition):
      missing in Databend          -> translate + create
      definition changed           -> translate + replace
      exists, never seen (adopt)   -> just record the hash
    Every new definition is compile-checked (SELECT * ... LIMIT 0, no rows
    read). A new view that doesn't compile is dropped; a replaced one gets its
    previous Databend definition back. Views that depend on a view created
    later in the run are retried (a few passes). Failures are reported, never
    raised; views deleted in Snowflake are left alone in Databend.
    """
    import hashlib
    vstate = state.setdefault(VIEW_STATE_KEY, {})
    sf_v, db_v = sf.views(), db.views()
    todo = []
    for key in sorted(sf_v):
        name = ".".join(key)
        if only and name not in only and key[1] not in only:
            continue
        h = hashlib.md5(sf_v[key].encode()).hexdigest()
        st = vstate.get(name) or {}
        exists = key in db_v
        if exists and st.get("hash") == h and st.get("ok", True):
            continue
        if exists and not st:
            vstate[name] = {"hash": h, "ok": True, "action": "adopt"}
            report["views_adopted"] += 1
            continue
        if not exists and st.get("hash") == h and st.get("ok") is False and not st.get("retry"):
            continue     # failed before with this exact definition; retried when something else changes
        todo.append((key, h, exists))
    if dry_run:
        for key, _, exists in todo:
            log(f"[dry-run] view {'.'.join(key)}: {'replace' if exists else 'create'}")
        return

    changed_any = False
    pending = todo
    for _pass in range(4):
        retry = []
        for key, h, exists in pending:
            schema, view = key
            name = ".".join(key)
            target_db = db.bend_db(schema)
            ref = f"{_bq(target_db)}.{_bq(view)}"
            old = None
            if exists:
                rows = db.run(f"SELECT view_query FROM system.views WHERE database = '{target_db}' "
                              f"AND name = '{view}'") if "'" not in target_db + view else []
                old = rows[0][0] if rows else None
            err = None
            for coerce in (False, True):
                try:
                    db.run(f"CREATE DATABASE IF NOT EXISTS {_bq(target_db)}")
                    db.run(_view_ddl(sf_v[key], schema, sf.database, coerce=coerce))
                    db.run(f"SELECT * FROM {ref} LIMIT 0")
                    err = None
                    break
                except Exception as e:  # noqa: BLE001
                    err = str(e)
                    if not _TEXT_AGG_ERR.search(err):
                        break
            time.sleep(VIEW_PAUSE_S)
            if err is None:
                vstate[name] = {"hash": h, "ok": True, "action": "replace" if exists else "create",
                                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                report["views_replaced" if exists else "views_created"].append(name)
                changed_any = True
                continue
            # undo: drop a new view, restore a replaced one
            try:
                if exists and old:
                    db.run(f"CREATE OR REPLACE VIEW {ref} AS {old}")
                elif not exists:
                    db.run(f"DROP VIEW IF EXISTS {ref}")
            except Exception:  # noqa: BLE001
                logger.warning("could not undo view %s", name, exc_info=True)
            vstate[name] = {"hash": h, "ok": False, "error": err[:300],
                            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            if _MISSING_DEP_ERR.search(err):
                retry.append((key, h, exists))
            else:
                report["views_failed"].append(f"{name}: {err[:200]}")
        if not retry or not changed_any:
            for key, _, _ in retry:
                report["views_failed"].append(f"{'.'.join(key)}: {vstate['.'.join(key)]['error'][:200]}")
            break
        pending, changed_any = retry, False
    else:   # passes used up with views still waiting on dependencies
        for key, _, _ in pending:
            report["views_failed"].append(f"{'.'.join(key)}: {vstate['.'.join(key)]['error'][:200]}")
    # Something new exists now -> give earlier failures one more chance next run.
    if report["views_created"] or report["views_replaced"]:
        for st in vstate.values():
            if st.get("ok") is False:
                st["retry"] = True
    else:
        for st in vstate.values():
            st.pop("retry", None)
    if report["views_created"] or report["views_replaced"] or report["views_failed"]:
        log(f"views: created {len(report['views_created'])}, replaced {len(report['views_replaced'])}, "
            f"failed {len(report['views_failed'])}")
