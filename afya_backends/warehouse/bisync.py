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

    @property
    def label(self) -> str:
        return f"{self.backend}:{self.database}"

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
        return out

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

    def read_batches(self, schema: str, table: str, columns: Optional[list[str]] = None):
        cols = ", ".join(_q(c) for c in columns) if columns else "*"
        cur = self.conn.cursor()
        if self.backend == "databend":   # dbapi translates Snowflake-dialect SQL to this side's names
            cur.execute(f"SELECT {cols} FROM {_q(schema)}.{_q(table)}")
        else:
            cur.execute(f"SELECT {cols} FROM {self.fq(schema, table)}")
        yield from cur.fetch_pandas_batches()

    def write(self, schema: str, table: str, df) -> None:
        from . import pandas_tools
        kw = {"use_logical_type": True} if self.backend == "snowflake" else {}
        ok, _, nrows, _ = pandas_tools.write_pandas(self.conn, df, table, database=self.database, schema=schema,
                                                    quote_identifiers=True, **kw)[:4]
        if not ok:
            raise RuntimeError(f"write_pandas into {self.label}.{schema}.{table} reported failure")

    def keys(self, schema: str, table: str, key: str) -> set:
        rows = self.run(f"SELECT {_q(key) if self.backend == 'snowflake' else _bq(key)} "
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
    src_cols = src.columns(schema, table)
    if not src_cols:
        raise RuntimeError(f"{src.label}: no columns visible for {schema}.{table}")
    cols = [Col(_dst_name(c.name, dst), c.kind, c.precision, c.scale) for c in src_cols]
    rename = {s.name: d.name for s, d in zip(src_cols, cols)}
    dst.ensure_schema(schema)
    staging = table + STAGING_SUFFIX
    variant_cols = [c for c in cols if c.kind == "VARIANT"]
    sf_variant_fix = dst.backend == "snowflake" and bool(variant_cols)
    dst.create_table(schema, staging, cols, variant_as_text=sf_variant_fix)
    total = 0
    try:
        for df in src.read_batches(schema, table, [c.name for c in src_cols]):
            if df.empty:
                continue
            dst.write(schema, staging, _prepare(df.rename(columns=rename), cols, dst))
            total += len(df)
        if sf_variant_fix:   # loaded as text; parse into real VARIANT
            exprs = ", ".join(f"TRY_PARSE_JSON({_q(c.name)}) AS {_q(c.name)}" if c.kind == "VARIANT"
                              else _q(c.name) for c in cols)
            dst.run(f"CREATE OR REPLACE TABLE {dst.fq(schema, staging)} AS SELECT {exprs} "
                    f"FROM {dst.fq(schema, staging)}")
        got = dst.count(schema, staging)
        if got != total:
            raise RuntimeError(f"staging row count {got} != {total} read from {src.label}")
        dst.swap_in(schema, staging, table, dst_exists)
    except Exception:
        try:
            dst.drop(schema, staging)
        except Exception:  # noqa: BLE001
            pass
        raise
    log(f"  copied {schema}.{table} {src.backend} -> {dst.backend}: {total:,} rows")
    return total


def insert_missing(src: Side, dst: Side, schema: str, table: str, key: str, *, log: Callable) -> int:
    """Append src rows whose key is absent on dst (columns dst lacks are dropped)."""
    dst_cols = dst.columns(schema, table)
    dst_names = {c.name.upper(): c.name for c in dst_cols}
    src_cols = [c for c in src.columns(schema, table) if c.name.upper() in dst_names]
    have = dst.keys(schema, table, dst_names[key.upper()])
    src_key = next(c.name for c in src_cols if c.name.upper() == key.upper())
    added = 0
    for df in src.read_batches(schema, table, [c.name for c in src_cols]):
        if df.empty:
            continue
        df = df[~df[src_key].map(lambda v: None if v is None else str(v)).isin(have)]
        if df.empty:
            continue
        df = df.rename(columns={c.name: dst_names[c.name.upper()] for c in src_cols})
        dst.write(schema, table, _prepare(df, [Col(dst_names[c.name.upper()], c.kind) for c in src_cols], dst))
        added += len(df)
    if added:
        log(f"  reconciled {schema}.{table} {src.backend} -> {dst.backend}: +{added:,} rows (by {key})")
    return added


def _key_for(a: Side, b: Side, schema: str, table: str) -> Optional[str]:
    names_a = {c.name.upper() for c in a.columns(schema, table)}
    names_b = {c.name.upper() for c in b.columns(schema, table)}
    return next((k for k in KEY_CANDIDATES if k in names_a and k in names_b), None)


# ── planning + run ───────────────────────────────────────────────────────
def plan(sf: dict, bend: dict, state: dict, *, allow_shrink: bool) -> list[tuple]:
    """[(key, action)] with action in copy_sf_to_db, copy_db_to_sf, reconcile, adopt, noop.
    Pure - unit-testable without any warehouse."""
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
        elif a_changed and not b_changed:
            out.append((key, "copy_sf_to_db" if allow_shrink or a.rows >= b.rows else "reconcile"))
        elif b_changed and not a_changed:
            out.append((key, "copy_db_to_sf" if allow_shrink or b.rows >= a.rows else "reconcile"))
        else:
            out.append((key, "reconcile"))
    return out


def run(sf: Side, db: Side, state_path: Path, *, allow_shrink: bool = False, dry_run: bool = False,
        only: Optional[set] = None, log: Callable[[str], None] = None) -> dict:
    log = log or logger.info
    state_path = Path(state_path)
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    sf_t, db_t = sf.tables(), db.tables()
    actions = plan(sf_t, db_t, state, allow_shrink=allow_shrink)
    if only:
        actions = [(k, a) for k, a in actions if ".".join(k) in only or k[1] in only]
    report = {"sf_tables": len(sf_t), "db_tables": len(db_t), "copied_sf_to_db": [], "copied_db_to_sf": [],
              "reconciled": [], "adopted": 0, "unchanged": 0, "conflicts": [], "failed": [], "rows": 0}

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
                report["rows"] += copy_table(sf, db, s_schema, s_table, dst_exists=b is not None, log=log)
                report["copied_sf_to_db"].append(name)
            elif action == "copy_db_to_sf":
                report["rows"] += copy_table(db, sf, s_schema, s_table, dst_exists=a is not None, log=log)
                report["copied_db_to_sf"].append(name)
            else:
                k = _key_for(sf, db, s_schema, s_table)
                if not k:
                    report["conflicts"].append(f"{name}: both sides differ (sf={a.rows} db={b.rows}) and "
                                               f"no key column {KEY_CANDIDATES} - left alone")
                    continue
                n = insert_missing(sf, db, s_schema, s_table, k, log=log)
                n += insert_missing(db, sf, s_schema, s_table, k, log=log)
                report["rows"] += n
                report["reconciled"].append(f"{name} (+{n})")
            # record both sides' fingerprints as they are NOW (after our writes)
            fresh_a = sf.info(s_schema, s_table) if action != "adopt" else a
            fresh_b = db.info(s_schema, s_table) if action != "adopt" else b
            state[name] = {"sf": fresh_a and fresh_a.fingerprint, "db": fresh_b and fresh_b.fingerprint,
                           "action": action, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            save()
        except Exception as e:  # noqa: BLE001 - keep going, report at the end
            logger.exception("bisync %s failed", name)
            report["failed"].append(f"{name}: {action}: {str(e)[:300]}")
    log(f"bisync: {json.dumps({k: (v if isinstance(v, int) else len(v)) for k, v in report.items()})}")
    return report
