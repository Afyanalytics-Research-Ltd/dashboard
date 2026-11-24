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

"Changed" = the table's fingerprint differs from the one recorded after the
last sync: Databend (row count, updated_on); Snowflake (row count, bytes), read
with SHOW TABLES - metadata only, so an hour in which nothing changed never
resumes (and bills) a Snowflake warehouse. State is a JSON file; if it is
lost, every table is "first sight": equal row counts are adopted as in sync,
different ones reconciled - safe, just slower.

Incremental tables (V3_SYNC models: _TENANT_ID, _KEY, _LAST_LOADED_AT - every
insert/update/delete-flag of pipelines/v3_to_snowflake_sync.py stamps
_LAST_LOADED_AT): a Snowflake-side change copies only the rows loaded since
Databend's newest _LAST_LOADED_AT (minus an overlap) and MERGEs them by key,
instead of re-copying the whole table.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
# Schemas bisync never touches, either way. V3_SYNC is written straight to Databend by the V3 sync
# (pipelines/v3_to_snowflake_sync.py, warehouse backend = databend): Snowflake's copy is frozen and
# must not be copied back over it. Override with BISYNC_SKIP_SCHEMAS="A,B" ("none" = skip nothing).
SKIP_SCHEMAS = {"V3_SYNC"}


def skip_schemas_from_env() -> set:
    import os
    raw = os.getenv("BISYNC_SKIP_SCHEMAS")
    if raw is None:
        return set(SKIP_SCHEMAS)
    if raw.strip().lower() in ("", "none"):
        return set()
    return {x.strip().upper() for x in raw.split(",") if x.strip()}


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

    def show(self, sql: str) -> list[dict]:
        """Snowflake SHOW/DESC rows as dicts. These are answered from metadata by the cloud
        services layer: no warehouse is resumed, so they cost no warehouse credits and keep
        working while a resource monitor has the warehouse suspended."""
        cur = self.conn.cursor()
        try:
            cur.execute(sql)
            cols = [d[0].lower() for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
        finally:
            cur.close()

    @staticmethod
    def _sf_info(r: dict) -> TableInfo:
        return TableInfo(r["schema_name"], r["name"], int(r.get("rows") or 0), f"b{int(r.get('bytes') or 0)}")

    def tables(self) -> dict[tuple, TableInfo]:
        out = {}
        if self.backend == "snowflake":
            for r in self.show(f"SHOW TABLES IN DATABASE {_q(self.database)}"):
                if (r["schema_name"] == "INFORMATION_SCHEMA" or r.get("kind") not in ("TABLE", "TRANSIENT")
                        or r.get("is_external") == "Y"):
                    continue
                key = (r["schema_name"].upper(), r["name"].upper())
                # Names differing only in case (TENRI.ACTIVATIONS and TENRI."activations") share one key:
                # always keep the upper-case one. Taking whichever came last made the pick (and so the
                # fingerprint) flip between runs, and the table was re-copied every hour.
                if key in out and out[key].name == out[key].name.upper():
                    continue
                out[key] = self._sf_info(r)
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
        skip = skip_schemas_from_env()
        out = {k: v for k, v in out.items()
               if not k[1].endswith((STAGING_SUFFIX, OLD_SUFFIX))
               and (self.schemas is None or k[0] in self.schemas) and k[0] not in skip}
        self._names = {k: (v.schema, v.name) for k, v in out.items()}
        return out

    def views(self) -> dict[tuple, str]:
        """{(SCHEMA, VIEW) upper: definition} - Snowflake; {(SCHEMA, VIEW) upper: ''} - Databend."""
        out = {}
        if self.backend == "snowflake":
            for r in self.show(f"SHOW VIEWS IN DATABASE {_q(self.database)}"):
                if r["schema_name"] != "INFORMATION_SCHEMA":
                    out[(r["schema_name"].upper(), r["name"].upper())] = r.get("text") or ""
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
        skip = skip_schemas_from_env()
        return {k: v for k, v in out.items() if (self.schemas is None or k[0] in self.schemas) and k[0] not in skip}

    def columns(self, schema: str, table: str) -> list[Col]:
        if self.backend == "snowflake":   # DESC: metadata only, no warehouse
            out = []
            for r in self.show(f"DESC TABLE {self.fq(schema, table)}"):
                if (r.get("kind") or "COLUMN") != "COLUMN":
                    continue
                m = re.match(r"\s*([A-Za-z_ ]+?)\s*(?:\((\d+)(?:\s*,\s*(\d+))?\))?\s*$", str(r["type"]))
                t, p, s = (m.group(1), m.group(2), m.group(3)) if m else (str(r["type"]), None, None)
                out.append(col_from_snowflake(r["name"], t, int(p) if p else None, int(s) if s else None))
            return out
        # DESC: Field, Type, Null, Default, Extra - in column order
        return [col_from_databend(r[0], r[1]) for r in self.run(f"DESC {self.fq(schema, table)}")]

    def info(self, schema: str, table: str) -> Optional[TableInfo]:
        """Current fingerprint of one table (None if it doesn't exist)."""
        def lit(s):
            return "'" + str(s).replace("'", "''") + "'"
        if self.backend == "snowflake":   # SHOW ... LIKE is case-insensitive and '_' is a wildcard
            hit = [r for r in self.show(f"SHOW TABLES LIKE {lit(table)} IN SCHEMA {_q(self.database)}.{_q(schema)}")
                   if r["name"] == table]
            return self._sf_info(hit[0]) if hit else None
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

    def read_batches(self, schema: str, table: str, columns: Optional[list] = None,
                     where: str = ""):
        """``columns``: names, or Col objects (then Snowflake timestamps are range-guarded, see
        ``_sf_select``). ``where``: a Snowflake-dialect condition (the Databend cursor translates it)."""
        cond = f" WHERE {where}" if where else ""
        cur = self.conn.cursor()
        if self.backend == "databend":   # dbapi translates Snowflake-dialect SQL to this side's names
            cols = ", ".join(_q(getattr(c, "name", c)) for c in columns) if columns else "*"
            cur.execute(f"SELECT {cols} FROM {_q(schema)}.{_q(table)}{cond}")
        else:
            cols = ", ".join(_sf_select(c) for c in columns) if columns else "*"
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


# pandas / Arrow nanosecond timestamps span 1677-09-21 .. 2262-04-11. A value outside it (V3 data entry: LMP
# '0026-06-10', EXPIRY_DATE '0229-05-30', DOB '1657-12-31') makes the Snowflake connector hand that result chunk
# over as timestamp[us] while the others are timestamp[ns], and pyarrow refuses to join them ("Schema at index N
# was different", V3_SYNC.EVALUATION__PATIENT_META / INVENTORY__BATCH, 2026-11-18) — one bad row stopped the table.
# Such values can't be a real date in these tables: they are read as NULL (Snowflake keeps the original).
TS_SAFE_YEARS = (1678, 2261)


def _sf_select(c) -> str:
    """One column of a Snowflake SELECT; a TIMESTAMP outside the nanosecond range comes back as NULL."""
    name = getattr(c, "name", c)
    q = _q(name)
    if getattr(c, "kind", None) == "TIMESTAMP":
        lo, hi = TS_SAFE_YEARS
        return f"IFF(YEAR({q}) BETWEEN {lo} AND {hi}, {q}, NULL) AS {q}"
    return q


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
        for df in src.read_batches(s_schema, s_table, src_cols):
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
    # Snowflake otherwise writes each parquet file with the narrowest integer type its own rows fit (ID int16 in one
    # file, int32 in the next) and Databend's COPY rejects files whose schemas differ ("Schema at index N was
    # different"). Off: every file carries the column's declared type.
    sf.run("ALTER SESSION SET ENABLE_UNLOAD_PHYSICAL_TYPE_OPTIMIZATION = FALSE")
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
    names = src_cols                     # Col objects: Snowflake timestamps are range-guarded (_sf_select)
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


# ── incremental (changed rows only) ──────────────────────────────────────
# V3_SYNC model tables (pipelines/v3_to_snowflake_sync.py): identity (_TENANT_ID, _KEY); every MERGE insert/update
# and every delete-flag UPDATE sets _LAST_LOADED_AT = SYSDATE(), so "loaded since X" is exactly what changed.
INCREMENTAL_KEYS = ("_TENANT_ID", "_KEY")
INCREMENTAL_WATERMARK = "_LAST_LOADED_AT"
# Re-read this much before Databend's newest watermark: covers a Snowflake MERGE that started (and stamped
# SYSDATE) before the previous read but committed after it. Re-read rows are merged again - idempotent.
INCREMENTAL_OVERLAP = timedelta(hours=2)
# More changed rows than this share of the table -> a full (bulk) copy is cheaper than a MERGE.
INCREMENTAL_MAX_FRACTION = 0.5


class NotIncremental(Exception):
    """The table can't (or shouldn't) take the incremental path; do a full copy."""


def incremental_copy(src: Side, dst: Side, schema: str, table: str, *, src_rows: int,
                     log: Callable) -> tuple[int, int]:
    """Copy only the src rows loaded since dst's newest INCREMENTAL_WATERMARK (minus the overlap) and MERGE them
    into dst by INCREMENTAL_KEYS: new rows inserted, changed / delete-flagged rows updated. dst must be Databend.
    Returns (rows merged, dst row count after). Raises NotIncremental when a full copy is needed instead."""
    if dst.backend != "databend":
        raise NotIncremental("incremental target must be Databend")
    s_schema, s_table = src.real(schema, table)
    d_schema, d_table = dst.real(schema, table)
    src_cols, dst_cols = src.columns(s_schema, s_table), dst.columns(d_schema, d_table)
    s_up, d_up = {c.name.upper(): c for c in src_cols}, {c.name.upper(): c for c in dst_cols}
    needed = (*INCREMENTAL_KEYS, INCREMENTAL_WATERMARK)
    if not all(k in s_up and k in d_up for k in needed):
        raise NotIncremental(f"no {'/'.join(needed)} columns")
    if set(s_up) != set(d_up):     # V3 added (or dropped) a column: the full copy rebuilds the table
        raise NotIncremental(f"columns differ: {sorted(set(s_up) ^ set(d_up))[:5]}")
    target = dst.fq(d_schema, d_table)
    wm = dst.run(f"SELECT TO_STRING(MAX({_bq(d_up[INCREMENTAL_WATERMARK].name)})) FROM {target}")
    wm = wm[0][0] if wm else None
    if not wm or str(wm).upper() == "NULL":
        raise NotIncremental("target has no watermark yet")
    since = (datetime.fromisoformat(str(wm)[:19]) - INCREMENTAL_OVERLAP).strftime("%Y-%m-%d %H:%M:%S")
    where = f"{_q(s_up[INCREMENTAL_WATERMARK].name)} >= '{since}'"
    if src.backend == "snowflake":
        changed = int(src.run(f"SELECT COUNT(*) FROM {src.fq(s_schema, s_table)} WHERE {where}")[0][0])
    else:
        changed = int(src.run(f"SELECT COUNT(*) FROM {src.fq(s_schema, s_table)} "
                              f"WHERE {_bq(s_up[INCREMENTAL_WATERMARK].name)} >= '{since}'")[0][0])
    if changed > max(1000, INCREMENTAL_MAX_FRACTION * src_rows):
        raise NotIncremental(f"{changed:,} of {src_rows:,} rows changed - a full copy is cheaper")

    if changed:
        staging = d_table + STAGING_SUFFIX
        rename = {c.name: d_up[c.name.upper()].name for c in src_cols}
        out_cols = [Col(d_up[c.name.upper()].name, c.kind) for c in src_cols]
        names = [c.name for c in out_cols]
        keys = [d_up[k].name for k in INCREMENTAL_KEYS]
        stamp = d_up[INCREMENTAL_WATERMARK].name
        try:
            dst.run(f"CREATE OR REPLACE TABLE {dst.fq(d_schema, staging)} LIKE {target}")
            for df in src.read_batches(s_schema, s_table, src_cols, where=where):
                if not df.empty:
                    dst.write(d_schema, staging, _prepare(df.rename(columns=rename), out_cols, dst))
            on = " AND ".join(f"t.{_bq(k)} = s.{_bq(k)}" for k in keys)
            sets = ", ".join(f"{_bq(n)} = s.{_bq(n)}" for n in names if n not in keys)
            dst.run(f"MERGE INTO {target} AS t USING (SELECT * FROM {dst.fq(d_schema, staging)} "
                    f"QUALIFY ROW_NUMBER() OVER (PARTITION BY {', '.join(_bq(k) for k in keys)} "
                    f"ORDER BY {_bq(stamp)} DESC) = 1) AS s ON {on} "
                    f"WHEN MATCHED THEN UPDATE SET {sets} "
                    f"WHEN NOT MATCHED THEN INSERT ({', '.join(_bq(n) for n in names)}) "
                    f"VALUES ({', '.join('s.' + _bq(n) for n in names)})")
        finally:
            try:
                dst.drop(d_schema, staging)
            except Exception:  # noqa: BLE001
                pass
    got = dst.count(d_schema, d_table)
    log(f"  incremental {schema}.{table} {src.backend} -> {dst.backend}: {changed:,} changed row(s) merged "
        f"(since {since}), {got:,} rows")
    return changed, got


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
def _old_sf_fingerprint(recorded: Optional[str]) -> bool:
    """Recorded before Snowflake fingerprints became rows|b<bytes> (they were rows|LAST_ALTERED)."""
    return bool(recorded) and not re.fullmatch(r"\d+\|b\d+", recorded)


def _changed(info: TableInfo, recorded: Optional[str]) -> bool:
    # An old-style record can't be compared with a SHOW TABLES fingerprint: compare row counts once (run() then
    # stores the new fingerprint) instead of treating all ~2,700 tables as changed and re-copying everything.
    if _old_sf_fingerprint(recorded) and re.fullmatch(r"\d+\|b\d+", info.fingerprint):
        return str(info.rows) != recorded.split("|", 1)[0]
    return info.fingerprint != recorded


_QUOTA_ERR = re.compile(r"090073|resource monitor .* exceeded its quota|cannot be resumed", re.I)


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
        a_changed, b_changed = _changed(a, st.get("sf")), b.fingerprint != st.get("db")
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
              "incremental": [], "reconciled": [], "adopted": 0, "unchanged": 0, "conflicts": [], "failed": [],
              "rows": 0, "views_created": [], "views_replaced": [], "views_adopted": 0, "views_failed": []}
    quota_error = None

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
            st = state.get(name)
            if not dry_run and a and st and _old_sf_fingerprint(st.get("sf")) and st.get("sf") != a.fingerprint:
                st["sf"] = a.fingerprint          # migrate to the SHOW TABLES fingerprint (saved below)
            continue
        if quota_error:
            report["failed"].append(f"{name}: {action}: not attempted - Snowflake warehouse quota exhausted")
            continue
        if dry_run:
            log(f"[dry-run] {name}: {action} (sf={a and a.rows} db={b and b.rows})")
            (report["conflicts"] if action == "reconcile" else report.setdefault("planned", [])).append(
                f"{name}: {action}")
            continue
        try:
            incr = action == "copy_sf_to_db" and b is not None and _incremental(sf, db, s_schema, s_table, a, log,
                                                                                 report, name)
            if action == "adopt":
                report["adopted"] += 1
            elif incr:
                pass
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
            # record both sides' fingerprints as they are NOW (after our writes). Incremental: Snowflake's as
            # PLANNED - if V3 wrote more while we merged, the next run sees a change and picks those rows up.
            fresh_a = a if action == "adopt" or incr else sf.info(*sf.real(s_schema, s_table))
            fresh_b = db.info(*db.real(s_schema, s_table)) if action != "adopt" else b
            state[name] = {"sf": fresh_a and fresh_a.fingerprint, "db": fresh_b and fresh_b.fingerprint,
                           "action": "incremental" if incr else action,
                           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            save()
        except Exception as e:  # noqa: BLE001 - keep going, report at the end
            logger.exception("bisync %s failed", name)
            report["failed"].append(f"{name}: {action}: {str(e)[:300]}")
            if _QUOTA_ERR.search(str(e)):
                # Every further Snowflake read fails the same way; stop instead of failing table after table.
                quota_error = str(e)[:300]
    if quota_error:
        report["quota_exhausted"] = quota_error
    if not dry_run:
        save()                          # fingerprint migrations above, even when nothing was copied
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


def _incremental(sf: Side, db: Side, schema: str, table: str, a: TableInfo, log: Callable, report: dict,
                 name: str) -> bool:
    """Try the changed-rows-only path; True if it synced the table, False if a full copy is needed."""
    try:
        n, got = incremental_copy(sf, db, schema, table, src_rows=a.rows, log=log)
    except NotIncremental as e:
        if not str(e).startswith("no "):
            log(f"  {name}: full copy ({e})")
        return False
    if got != a.rows:
        now = sf.info(*sf.real(schema, table))
        if now is not None and now.rows == a.rows:
            # Snowflake didn't move while we merged, yet the counts differ: Databend has diverged (e.g. Snowflake
            # removed duplicate rows). The full copy below makes it an exact copy again.
            log(f"  {name}: databend has {got:,} rows after the merge, snowflake {a.rows:,} - full copy")
            return False
    report["rows"] += n
    report["incremental"].append(f"{name} (+{n})")
    return True


# ── views (Snowflake -> Databend, one way) ───────────────────────────────
VIEW_STATE_KEY = "__views__"
VIEW_PAUSE_S = 0.1          # between view DDLs - keep the catalog load trivial
# Bump when the view translation changes: every view is re-translated and replaced once (compile-checked; the
# previous definition is restored if the new one doesn't compile), so all views carry the current translation.
# Without a bump a view is only re-created when its Snowflake definition changes.
# 2: qualify unqualified names with the view's schema, single-use CTEs inlined, TO_TIMESTAMP(n, scale),
#    ASOF MATCH_CONDITION, FLATTEN `value` only where FLATTEN is read.
VIEW_TRANSLATOR_VERSION = 2
_TEXT_AGG_ERR = re.compile(r"(sum|avg) does not support type '?String", re.I)
_MISSING_DEP_ERR = re.compile(r"Unknown (table|database)", re.I)


# SHOW VIEWS' text carries the view COMMENT (INFORMATION_SCHEMA.VIEW_DEFINITION didn't); Databend's CREATE VIEW
# has no COMMENT clause, so it's dropped from the header (before the AS).
_VIEW_COMMENT = re.compile(r"(?is)^(\s*CREATE\b(?:(?!\bAS\b).)*?\bVIEW\b(?:(?!\bAS\b).)*?)"
                           r"\s+COMMENT\s*=\s*'(?:\\.|''|[^'\\])*'")


VIEW_REWRITES_DIR = Path(__file__).resolve().parent / "view_rewrites"


def _view_source(name: str, definition: str) -> str:
    """The Snowflake-dialect text to translate for view SCHEMA.NAME: a hand-written stand-in from
    view_rewrites/<SCHEMA.NAME>.sql when its `-- for_hash: <md5>` matches the current Snowflake definition (for
    views Databend can't plan as written), otherwise the Snowflake definition itself."""
    import hashlib
    f = VIEW_REWRITES_DIR / f"{name.upper()}.sql"
    if f.exists():
        text = f.read_text()
        m = re.search(r"--\s*for_hash:\s*([0-9a-f]{32})", text)
        if m and m.group(1) == hashlib.md5(definition.encode()).hexdigest():
            return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("--")).strip()
        logger.warning("view rewrite %s is stale (Snowflake definition changed) - translating Snowflake's", f.name)
    return definition


def _view_ddl(definition: str, schema: str, database: str, *, coerce: bool = False) -> str:
    definition = _VIEW_COMMENT.sub(r"\1", definition, count=1)
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
        if exists and st.get("hash") == h and st.get("ok", True) and st.get("tv", 1) >= VIEW_TRANSLATOR_VERSION:
            continue
        if exists and not st:
            vstate[name] = {"hash": h, "ok": True, "action": "adopt"}
            report["views_adopted"] += 1
            continue
        if (not exists and st.get("hash") == h and st.get("ok") is False and not st.get("retry")
                and st.get("tv") == VIEW_TRANSLATOR_VERSION):
            continue     # failed before with this exact definition + translator; retried when something changes
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
                    db.run(_view_ddl(_view_source(name, sf_v[key]), schema, sf.database, coerce=coerce))
                    db.run(f"SELECT * FROM {ref} LIMIT 0")
                    err = None
                    break
                except Exception as e:  # noqa: BLE001
                    err = str(e)
                    if not _TEXT_AGG_ERR.search(err):
                        break
            time.sleep(VIEW_PAUSE_S)
            prior = vstate.get(name) or {}
            # only the translator changed (same Snowflake definition, working view): a failure keeps the old view
            retranslate = exists and prior.get("hash") == h and prior.get("ok", True)
            if err is None:
                vstate[name] = {"hash": h, "ok": True, "action": "replace" if exists else "create",
                                "tv": VIEW_TRANSLATOR_VERSION,
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
            if retranslate and old:
                vstate[name] = {"hash": h, "ok": True, "action": "kept", "tv": VIEW_TRANSLATOR_VERSION,
                                "retranslate_error": err[:300],
                                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                report.setdefault("views_kept", []).append(name)
                continue
            vstate[name] = {"hash": h, "ok": False, "error": err[:300], "tv": VIEW_TRANSLATOR_VERSION,
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
