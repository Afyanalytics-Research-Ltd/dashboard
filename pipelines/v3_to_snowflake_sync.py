#!/usr/bin/env python3
"""
v3_to_snowflake_sync.py — keep Snowflake up to date with V3, one gateway model at a time.

    V3 gateway (read, every page)  →  rows new/edited since the watermark  →  MERGE in batches  →  Snowflake

WHAT LANDS IN SNOWFLAKE  ({database}.{schema}, default HOSPITALS.V3_SYNC)
  One table per model, named <SERVICE>__<ALIAS> (reception:patient → RECEPTION__PATIENT), shaped like the
  V3 table: every V3 column under its own name (first_name → FIRST_NAME), in V3's order, typed — then the
  pipeline's own columns, all starting with "_":
    _TENANT_ID, _KEY             the row's identity (V3 organisation + id; uuid / content hash when there's no id)
    _ROW_HASH                    hash of the row; an unchanged row is never rewritten
    _IS_DELETED, _DELETED_AT     the row stopped coming back from V3 (deleted there); never physically removed
    _SOURCE, _FIRST_LOADED_AT, _LAST_LOADED_AT
    _RAW                         the V3 row exactly as received (VARIANT)
  plus _SYNC_STATE: one row per (tenant, model) — its watermark and the last run's counts / status / error.

  Column types: the gateway doesn't expose V3's MySQL types, so a new column's type is inferred from the
  values in the batch that first carries it (column_types): integers → NUMBER(38,0), decimals ("1500.00")
  → NUMBER(38,6), other numbers → FLOAT, true/false → BOOLEAN, ISO timestamps → TIMESTAMP_NTZ (UTC),
  dates → DATE, objects / arrays → VARIANT, anything else or mixed → STRING. Digit strings such as phone
  numbers stay STRING (leading zeros). A column only ever seen as null starts as TIMESTAMP_NTZ if named
  *_at, else STRING — and is retyped when real values arrive while it's still empty in Snowflake. Otherwise
  a type is fixed once created; a later value that doesn't fit lands as NULL in its column and is still in
  _RAW. A column V3 adds later is added to the table (ALTER TABLE ADD COLUMN) the first time a row carries it.

HOW "INCREMENTAL" WORKS
  The V3 gateway can only filter on equality, so it can't be asked for "rows changed since X". Each run
  therefore reads every page of a model, and the watermark decides what gets WRITTEN:
    * since     = watermark − lookback   (the overlap absorbs clock skew between V3 and us)
    * a row is sent to Snowflake when updated_at >= since, or when it has no updated_at at all
    * MERGE inserts new keys and updates a matched row only if its hash changed, so the overlap is free
    * the new watermark = min(newest updated_at seen, time the read started) — a row edited while the
      read was running is picked up next run instead of being skipped
    * the watermark is saved only after every batch of the model is in; a failed run leaves it where it
      was, so the next run re-sends the same window (MERGE makes that harmless)
  full_refresh ignores the watermark and sends every row (still hash-guarded); use it after edits made
  straight in the V3 database, which don't move updated_at.

DELETES
  Soft-deleted V3 rows stop coming back from the gateway. Since each run reads the whole model, keys
  that are in Snowflake but weren't read are flagged _IS_DELETED (and un-flagged if they come back).
  Guard: if more than max_delete_fraction of a tenant's live rows would be flagged at once, nothing is
  flagged and the result says why — a half-empty read must not look like a mass delete.

USAGE
  python v3_to_snowflake_sync.py --source gyno                              # every readable model
  python v3_to_snowflake_sync.py --source gyno --models reception:patient,finance:invoice
  python v3_to_snowflake_sync.py --source gyno --services reception --dry-run
  python v3_to_snowflake_sync.py --source gyno --models reception:patient --full-refresh
  python v3_to_snowflake_sync.py --source gyno --models reception:patient --limit 3   # canary

  --source is a V3Client key: AFYA_<KEY>_USERNAME / _PASSWORD / _URL_TEMPLATE … in .env (v3_to_v3_client.py).
  Snowflake: SNOWFLAKE_USER / ACCOUNT / WAREHOUSE / ROLE + SNOWFLAKE_PRIVATE_KEY_PATH (or SNOWFLAKE_PASSWORD).
  Airflow: dags/v3_to_snowflake_sync_dag.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

log = logging.getLogger("v3_to_snowflake_sync")

DEFAULT_DATABASE = os.getenv("V3_SYNC_DATABASE", "HOSPITALS")
DEFAULT_SCHEMA = os.getenv("V3_SYNC_SCHEMA", "V3_SYNC")
STATE_TABLE = "_SYNC_STATE"
DEFAULT_BATCH_SIZE = 5_000
DEFAULT_LOOKBACK = timedelta(minutes=60)
DEFAULT_MAX_DELETE_FRACTION = 0.2
TS_FORMAT = "%Y-%m-%d %H:%M:%S.%f"            # how timestamps travel to the staging table (naive UTC)


# ── models and tables ─────────────────────────────────────────────────────

@dataclass(frozen=True, order=True)
class Model:
    service: str
    alias: str

    @property
    def name(self) -> str:
        return f"{self.service}:{self.alias}"

    @property
    def table(self) -> str:
        return table_name(self.service, self.alias)


def _ident(part: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", part).strip("_").upper()


def table_name(service: str, alias: str) -> str:
    """reception:patient → RECEPTION__PATIENT (service first: core and dialysis both have `users`)."""
    return f"{_ident(service)}__{_ident(alias)}"


def parse_model(text: str) -> Model:
    service, sep, alias = text.strip().partition(":")
    if not sep or not service or not alias:
        raise ValueError(f"model {text!r} must look like service:alias, e.g. reception:patient")
    return Model(service.strip().lower(), alias.strip())


def parse_models(values: Iterable[str] | str | None) -> list[Model]:
    """Accepts a list, or one string of comma / whitespace separated service:alias pairs."""
    if not values:
        return []
    items = [values] if isinstance(values, str) else list(values)
    names = [s for item in items for s in re.split(r"[\s,\[\]\"']+", str(item)) if s]
    return sorted({parse_model(n) for n in names})


# ── rows ──────────────────────────────────────────────────────────────────

def parse_ts(value) -> datetime | None:
    """A V3 timestamp as naive UTC. V3 sends '2026-07-20T21:00:00.000000Z'; older rows may be
    '2026-07-20 21:00:00' (taken as UTC). Blank, zero and unparseable dates → None."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text or text.startswith("0000-00-00"):
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def row_hash(row: dict) -> str:
    """Content fingerprint, independent of key order."""
    canonical = json.dumps(row, sort_keys=True, default=str, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def row_key(row: dict, digest: str | None = None) -> str:
    """The row's identity within its tenant: id, else uuid, else its content hash."""
    if row.get("id") not in (None, ""):
        return str(row["id"])
    if row.get("uuid"):
        return f"uuid:{row['uuid']}"
    return f"hash:{digest or row_hash(row)}"


def is_changed(updated_at: datetime | None, since: datetime | None) -> bool:
    """Should this row be written? Everything on a first / full run; a row without updated_at always
    (we can't tell — the MERGE's hash check decides); otherwise only if edited since `since`."""
    return since is None or updated_at is None or updated_at >= since


def next_watermark(previous: datetime | None, newest_seen: datetime | None,
                   read_started: datetime) -> datetime | None:
    """Never past the moment the read started (a row edited mid-read gets its turn next run) and never
    backwards."""
    if newest_seen is None:
        return previous
    candidate = min(newest_seen, read_started)
    return max(previous, candidate) if previous else candidate


def _fmt(ts: datetime | None) -> str | None:
    return ts.strftime(TS_FORMAT) if ts else None


@dataclass
class StagedRow:
    row: dict                      # the V3 row (its columns shape the table)
    key: str
    updated_at: str | None
    row_hash: str
    data: str                      # the row as JSON — what travels to Snowflake

    @classmethod
    def from_row(cls, row: dict, updated_at: datetime | None = None) -> "StagedRow":
        digest = row_hash(row)
        return cls(row=row, key=row_key(row, digest),
                   updated_at=_fmt(updated_at if updated_at is not None else parse_ts(row.get("updated_at"))),
                   row_hash=digest,
                   data=json.dumps(row, default=str, ensure_ascii=False))

    def as_tuple(self) -> tuple:
        return self.key, self.updated_at, self.row_hash, self.data


# staging names start with "_" so they can't meet a V3 column (column_name strips leading underscores)
STAGE_COLUMNS = ("_SKEY", "_SUPDATED", "_SHASH", "_SDATA")


# ── V3 columns → Snowflake columns ────────────────────────────────────────

INT, DECIMAL, FLOAT = "NUMBER(38,0)", "NUMBER(38,6)", "FLOAT"
BOOLEAN, TIMESTAMP, DATE = "BOOLEAN", "TIMESTAMP_NTZ", "DATE"
VARIANT, STRING = "VARIANT", "STRING"

_DECIMAL_RE = re.compile(r"^-?\d+\.\d+$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$")


def value_type(value) -> str | None:
    """The Snowflake type one V3 value suggests (None for null)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return BOOLEAN
    if isinstance(value, int):
        return INT
    if isinstance(value, float):
        return FLOAT
    if isinstance(value, (dict, list)):
        return VARIANT
    text = str(value)
    if _TS_RE.match(text) and parse_ts(text):
        return TIMESTAMP
    if _DATE_RE.match(text) and parse_ts(text):
        return DATE
    if _DECIMAL_RE.match(text):
        return DECIMAL
    return STRING


def _widen(types: set[str]) -> str:
    if not types:
        return STRING
    if len(types) == 1:
        return next(iter(types))
    if types <= {INT, DECIMAL, FLOAT}:
        return FLOAT if FLOAT in types else DECIMAL
    if types <= {DATE, TIMESTAMP}:
        return TIMESTAMP
    if VARIANT in types:
        return VARIANT
    return STRING


def column_types(rows: Iterable[dict]) -> dict[str, str]:
    """{V3 column: Snowflake type} for these rows, columns in the order V3 sends them. A column that is
    null in every row is a timestamp if it's named like one (Laravel's *_at), else STRING for now."""
    seen: dict[str, set[str]] = {}
    for row in rows:
        for name, value in row.items():
            t = value_type(value)
            seen.setdefault(name, set())
            if t:
                seen[name].add(t)
    return {name: _widen(types) if types else (TIMESTAMP if name.endswith("_at") else STRING)
            for name, types in seen.items()}


def column_name(v3_name: str) -> str:
    """V3 first_name → FIRST_NAME (quoted when used, so V3 names like `order` or `group` are fine)."""
    return _ident(v3_name) or "COL"


def _cast(v3_name: str, sf_type: str) -> str:
    """SQL that turns the staged JSON row's field into its column's type; NULL when it doesn't fit.
    (_J is the parsed row; % is doubled because the statement also carries %(name)s parameters.)"""
    field = "_J['" + v3_name.replace("'", "''").replace("%", "%%") + "']"
    text = f"TO_VARCHAR({field})"
    if sf_type == VARIANT:
        return f"STRIP_NULL_VALUE({field})"           # JSON null → SQL NULL, so IS NULL works
    if sf_type == BOOLEAN:
        return f"TRY_TO_BOOLEAN({text})"
    if sf_type.startswith("NUMBER"):
        precision, scale = re.findall(r"\d+", sf_type) or ("38", "0")
        return f"TRY_TO_NUMBER({text}, {precision}, {scale})"
    if sf_type == FLOAT:
        return f"TRY_TO_DOUBLE({text})"
    if sf_type == TIMESTAMP:
        return f"CONVERT_TIMEZONE('UTC', TRY_TO_TIMESTAMP_TZ({text}))::TIMESTAMP_NTZ"
    if sf_type == DATE:
        return f"TRY_TO_DATE({text})"
    return text


META_COLUMNS = {                       # the pipeline's own columns, after V3's
    "_TENANT_ID": "NUMBER NOT NULL",
    "_KEY": "STRING NOT NULL",
    "_ROW_HASH": "STRING",
    "_IS_DELETED": "BOOLEAN DEFAULT FALSE",
    "_DELETED_AT": "TIMESTAMP_NTZ",
    "_SOURCE": "STRING",
    "_FIRST_LOADED_AT": "TIMESTAMP_NTZ",
    "_LAST_LOADED_AT": "TIMESTAMP_NTZ",
    "_RAW": "VARIANT",
}


# ── Snowflake ─────────────────────────────────────────────────────────────

def snowflake_connect(database: str = DEFAULT_DATABASE, schema: str = DEFAULT_SCHEMA):
    import snowflake.connector
    from afya_backends import warehouse as afya_warehouse
    kwargs = dict(user=os.getenv("SNOWFLAKE_USER", "").strip(),
                  account=os.getenv("SNOWFLAKE_ACCOUNT", "").strip(),
                  warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "").strip(),
                  role=os.getenv("SNOWFLAKE_ROLE", "").strip() or None,
                  database=database,
                  session_parameters={"TIMEZONE": "UTC", "QUERY_TAG": "v3_to_snowflake_sync"})
    key_path = (os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH") or "").strip()
    password = (os.getenv("SNOWFLAKE_PASSWORD") or "").strip()
    if key_path:
        kwargs["private_key_file"] = key_path
    elif password:
        kwargs["password"] = password
    else:
        raise RuntimeError("set SNOWFLAKE_PRIVATE_KEY_PATH or SNOWFLAKE_PASSWORD")
    return afya_warehouse.connect(**kwargs)


class SnowflakeSink:
    """Everything this pipeline does to Snowflake. Tables live in {database}.{schema}; staging tables are
    session TEMPORARY tables, so nothing is left behind."""

    def __init__(self, conn, database: str = DEFAULT_DATABASE, schema: str = DEFAULT_SCHEMA):
        self.conn = conn
        self.database = _ident(database)
        self.schema = _ident(schema)
        self.fq = f"{self.database}.{self.schema}"
        self._columns: dict[str, dict[str, str]] = {}
        self._filled: dict[str, set[str]] = {}         # STRING columns known to hold data (never retyped)

    def _exec(self, sql: str, params=None) -> list[tuple]:
        cur = self.conn.cursor()
        try:
            cur.execute(sql, params)
            try:
                return cur.fetchall()
            except Exception:              # DDL / DML without a result set
                return []
        finally:
            cur.close()

    # set-up ----------------------------------------------------------------
    def ensure_schema(self) -> None:
        self._exec(f"CREATE SCHEMA IF NOT EXISTS {self.fq}")
        self._exec(f"""
            CREATE TABLE IF NOT EXISTS {self.fq}.{STATE_TABLE} (
                TENANT_ID          NUMBER        NOT NULL,
                MODEL              STRING        NOT NULL,
                TARGET_TABLE       STRING,
                SOURCE             STRING,
                WATERMARK          TIMESTAMP_NTZ,
                LAST_STATUS        STRING,
                LAST_ERROR         STRING,
                LAST_RUN_STARTED   TIMESTAMP_NTZ,
                LAST_RUN_FINISHED  TIMESTAMP_NTZ,
                ROWS_READ          NUMBER,
                ROWS_SENT          NUMBER,
                ROWS_INSERTED      NUMBER,
                ROWS_UPDATED       NUMBER,
                ROWS_DELETED       NUMBER,
                PRIMARY KEY (TENANT_ID, MODEL))""")

    def table_columns(self, table: str) -> dict[str, str]:
        """{column: type} of an existing table, in table order ({} when it doesn't exist yet)."""
        if table not in self._columns:
            rows = self._exec(f"""
                SELECT COLUMN_NAME, DATA_TYPE, NUMERIC_PRECISION, NUMERIC_SCALE
                FROM {self.database}.INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = %(s)s AND TABLE_NAME = %(t)s ORDER BY ORDINAL_POSITION""",
                              {"s": self.schema, "t": table})
            self._columns[table] = {
                name: (f"NUMBER({p},{s})" if dtype == "NUMBER" else STRING if dtype == "TEXT" else dtype)
                for name, dtype, p, s in rows}
        return self._columns[table]

    def ensure_table(self, table: str, v3_columns: dict[str, str]) -> dict[str, str]:
        """Create the table shaped like the V3 one (its columns, then the pipeline's _ columns), or add the
        V3 columns it doesn't have yet. Returns the table's {column: type}."""
        known = self.table_columns(table)
        wanted: dict[str, str] = {}
        for v3_name, sf_type in v3_columns.items():
            wanted.setdefault(column_name(v3_name), sf_type)
        if not known:
            cols = [f'"{c}" {t}' for c, t in wanted.items()] + [f"{c} {t}" for c, t in META_COLUMNS.items()]
            self._exec(f"CREATE TABLE IF NOT EXISTS {self.fq}.{table} ({', '.join(cols)}, "
                       f"PRIMARY KEY (_TENANT_ID, _KEY))")
            self._columns.pop(table)
            return self.table_columns(table)
        # a column made STRING only because it was empty gets its real type once values show up — as long
        # as it is still empty in Snowflake (drop + add loses nothing; the column moves to the end)
        filled = self._filled.setdefault(table, set())
        for c, t in wanted.items():
            if known.get(c) == STRING and t != STRING and c not in filled:
                (non_null,), = self._exec(f'SELECT COUNT("{c}") FROM {self.fq}.{table}')
                if non_null:
                    filled.add(c)
                    continue
                self._exec(f'ALTER TABLE {self.fq}.{table} DROP COLUMN "{c}"')
                self._exec(f'ALTER TABLE {self.fq}.{table} ADD COLUMN "{c}" {t}')
                known.pop(c)
                known[c] = t
                log.info("%s: empty column %s retyped STRING → %s", table, c, t)
        added = {c: t for c, t in wanted.items() if c not in known}
        for c, t in added.items():
            self._exec(f'ALTER TABLE {self.fq}.{table} ADD COLUMN IF NOT EXISTS "{c}" {t}')
        if added:
            log.info("%s: new V3 column(s) added: %s", table, ", ".join(f"{c} {t}" for c, t in added.items()))
            known.update(added)
        return known

    # watermark / state ---------------------------------------------------------
    def get_watermark(self, tenant_id: int, model: str) -> datetime | None:
        try:
            rows = self._exec(f"SELECT WATERMARK FROM {self.fq}.{STATE_TABLE} "
                              f"WHERE TENANT_ID = %(t)s AND MODEL = %(m)s", {"t": tenant_id, "m": model})
        except Exception as e:                 # no state table yet (e.g. a dry run before the first real one)
            if "does not exist" in str(e):
                return None
            raise
        return rows[0][0] if rows else None

    def save_state(self, tenant_id: int, model: Model, source: str, *, watermark: datetime | None,
                   status: str, error: str | None, started: datetime, counts: dict) -> None:
        self._exec(f"""
            MERGE INTO {self.fq}.{STATE_TABLE} t
            USING (SELECT %(t)s TENANT_ID, %(m)s MODEL) s
               ON t.TENANT_ID = s.TENANT_ID AND t.MODEL = s.MODEL
            WHEN MATCHED THEN UPDATE SET
                TARGET_TABLE = %(tbl)s, SOURCE = %(src)s, WATERMARK = %(wm)s::TIMESTAMP_NTZ,
                LAST_STATUS = %(st)s, LAST_ERROR = %(err)s,
                LAST_RUN_STARTED = %(started)s::TIMESTAMP_NTZ, LAST_RUN_FINISHED = SYSDATE(),
                ROWS_READ = %(read)s, ROWS_SENT = %(sent)s, ROWS_INSERTED = %(ins)s,
                ROWS_UPDATED = %(upd)s, ROWS_DELETED = %(dele)s
            WHEN NOT MATCHED THEN INSERT (TENANT_ID, MODEL, TARGET_TABLE, SOURCE, WATERMARK, LAST_STATUS, LAST_ERROR,
                LAST_RUN_STARTED, LAST_RUN_FINISHED, ROWS_READ, ROWS_SENT, ROWS_INSERTED, ROWS_UPDATED, ROWS_DELETED)
            VALUES (%(t)s, %(m)s, %(tbl)s, %(src)s, %(wm)s::TIMESTAMP_NTZ, %(st)s, %(err)s,
                %(started)s::TIMESTAMP_NTZ, SYSDATE(), %(read)s, %(sent)s, %(ins)s, %(upd)s, %(dele)s)""",
                   {"t": tenant_id, "m": model.name, "tbl": model.table, "src": source, "wm": _fmt(watermark),
                    "st": status, "err": (error or None) and error[:4000], "started": _fmt(started),
                    "read": counts.get("read", 0), "sent": counts.get("sent", 0),
                    "ins": counts.get("inserted", 0), "upd": counts.get("updated", 0),
                    "dele": counts.get("deleted", 0)})

    # loading -------------------------------------------------------------------
    def _stage(self, name: str, columns: tuple[str, ...], rows: list[tuple]) -> str:
        """(Re)fill a session temp table with rows; returns its name."""
        from afya_backends.warehouse import write_pandas
        import pandas as pd
        stage = f"{self.fq}._STG_{name}"
        cols = ", ".join(f"{c} STRING" for c in columns)
        self._exec(f"CREATE TEMPORARY TABLE IF NOT EXISTS {stage} ({cols})")
        self._exec(f"TRUNCATE TABLE {stage}")
        if rows:
            ok, _, n, _ = write_pandas(self.conn, pd.DataFrame(rows, columns=list(columns)), f"_STG_{name}",
                                       database=self.database, schema=self.schema, quote_identifiers=False)
            if not ok or n != len(rows):
                raise RuntimeError(f"staging {name}: wrote {n} of {len(rows)} rows")
        return stage

    def merge(self, table: str, tenant_id: int, source: str, rows: list[StagedRow]) -> tuple[int, int]:
        """Upsert one batch. Returns (inserted, updated); an unchanged row counts as neither."""
        if not rows:
            return 0, 0
        columns = self.ensure_table(table, column_types(r.row for r in rows))
        v3_names = {}
        for r in rows:
            for name in r.row:
                v3_names.setdefault(column_name(name), name)
        v3_cols = [(c, t) for c, t in columns.items() if c not in META_COLUMNS]
        typed = "".join(f',\n                       {_cast(v3_names.get(c, c.lower()), t)} AS "{c}"'
                        for c, t in v3_cols)
        set_v3 = "".join(f'"{c}" = s."{c}", ' for c, _ in v3_cols)
        ins_cols = "".join(f'"{c}", ' for c, _ in v3_cols)
        ins_vals = "".join(f's."{c}", ' for c, _ in v3_cols)
        stage = self._stage(table, STAGE_COLUMNS, [r.as_tuple() for r in rows])
        result = self._exec(f"""
            MERGE INTO {self.fq}.{table} t
            USING (
                SELECT _SKEY, _SHASH, _J{typed}
                FROM (SELECT _SKEY, _SHASH, PARSE_JSON(_SDATA) _J FROM {stage}
                      QUALIFY ROW_NUMBER() OVER (PARTITION BY _SKEY ORDER BY _SUPDATED DESC NULLS LAST) = 1)
            ) s
               ON t._TENANT_ID = %(t)s AND t._KEY = s._SKEY
            WHEN MATCHED AND (t._ROW_HASH IS DISTINCT FROM s._SHASH OR t._IS_DELETED) THEN UPDATE SET
                {set_v3}_RAW = s._J, _ROW_HASH = s._SHASH, _IS_DELETED = FALSE, _DELETED_AT = NULL,
                _SOURCE = %(src)s, _LAST_LOADED_AT = SYSDATE()
            WHEN NOT MATCHED THEN INSERT ({ins_cols}_TENANT_ID, _KEY, _RAW, _ROW_HASH, _IS_DELETED, _SOURCE,
                _FIRST_LOADED_AT, _LAST_LOADED_AT)
            VALUES ({ins_vals}%(t)s, s._SKEY, s._J, s._SHASH, FALSE, %(src)s, SYSDATE(), SYSDATE())""",
                            {"t": tenant_id, "src": source})
        inserted, updated = (result[0][0], result[0][1]) if result else (0, 0)
        return int(inserted), int(updated)

    def flag_deleted(self, table: str, tenant_id: int, seen_keys: list[str],
                     max_fraction: float) -> tuple[int, int, str | None]:
        """Flag this tenant's rows whose key wasn't read as deleted, and un-flag ones that came back.
        Returns (flagged, restored, reason-if-skipped)."""
        stage = self._stage(f"{table}_SEEN", ("_SKEY",), [(k,) for k in seen_keys])
        missing = f"""
            FROM {self.fq}.{table} t
            WHERE t._TENANT_ID = %(t)s AND NOT t._IS_DELETED
              AND NOT EXISTS (SELECT 1 FROM {stage} s WHERE s._SKEY = t._KEY)"""
        (live, gone), = self._exec(f"""
            SELECT (SELECT COUNT(*) FROM {self.fq}.{table} WHERE _TENANT_ID = %(t)s AND NOT _IS_DELETED),
                   (SELECT COUNT(*) {missing})""", {"t": tenant_id})
        skipped = None
        flagged = 0
        if gone and live and gone / live > max_fraction:
            skipped = (f"{gone} of {live} live rows were not read back — over the {max_fraction:.0%} guard, "
                       f"so none were flagged deleted (check V3; rerun with a higher max_delete_fraction if real)")
        elif gone:
            self._exec(f"""
                UPDATE {self.fq}.{table} t SET _IS_DELETED = TRUE, _DELETED_AT = SYSDATE(), _LAST_LOADED_AT = SYSDATE()
                WHERE t._TENANT_ID = %(t)s AND NOT t._IS_DELETED
                  AND NOT EXISTS (SELECT 1 FROM {stage} s WHERE s._SKEY = t._KEY)""", {"t": tenant_id})
            flagged = int(gone)
        restored = self._exec(f"""
            UPDATE {self.fq}.{table} t SET _IS_DELETED = FALSE, _DELETED_AT = NULL, _LAST_LOADED_AT = SYSDATE()
            WHERE t._TENANT_ID = %(t)s AND t._IS_DELETED
              AND EXISTS (SELECT 1 FROM {stage} s WHERE s._SKEY = t._KEY)""", {"t": tenant_id})
        return flagged, int(restored[0][0]) if restored else 0, skipped


# ── one model ─────────────────────────────────────────────────────────────

@dataclass
class SyncOptions:
    full_refresh: bool = False
    lookback: timedelta = DEFAULT_LOOKBACK
    batch_size: int = DEFAULT_BATCH_SIZE
    detect_deletes: bool = True
    max_delete_fraction: float = DEFAULT_MAX_DELETE_FRACTION
    page_workers: int = 4
    limit: int = 0                 # stop after N rows read (canary): no deletes, watermark not moved
    dry_run: bool = False          # read and count only; nothing written


@dataclass
class ModelResult:
    model: str
    table: str
    tenant_id: int | None = None
    status: str = "ok"             # ok | dry_run | partial (limit) | failed
    read: int = 0
    sent: int = 0
    inserted: int = 0
    updated: int = 0
    deleted: int = 0
    restored: int = 0
    watermark_before: str | None = None
    watermark_after: str | None = None
    note: str | None = None
    error: str | None = None
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


RETRY_WAITS = (10, 30)             # seconds before the 2nd and 3rd attempt of a Snowflake write


def _with_retries(what: str, fn, *args):
    """Run a Snowflake write, retrying transient failures (e.g. a 403 / dropped upload to the internal
    stage). Safe because every write here is idempotent — a MERGE or a flag UPDATE."""
    for attempt, wait in enumerate((*RETRY_WAITS, None), start=1):
        try:
            return fn(*args)
        except Exception as e:
            if wait is None:
                raise
            log.warning("%s failed (attempt %d): %s — retrying in %ss", what, attempt, e, wait)
            time.sleep(wait)


def sync_model(client, sink, model: Model, *, source: str, opts: SyncOptions = SyncOptions(),
               clock: Callable[[], datetime] = _utcnow) -> ModelResult:
    """Read one model from V3 and bring its Snowflake table up to date. Never raises for the model's own
    errors: they come back as status 'failed' (and are recorded in _SYNC_STATE) with the watermark
    unchanged, so the next run retries the same window."""
    started = clock()
    res = ModelResult(model.name, model.table)
    stored: datetime | None = None
    tenant: int | None = None
    batch: list[StagedRow] = []
    seen: list[str] = []
    newest: datetime | None = None
    complete = True

    def flush():
        if batch and not opts.dry_run:
            ins, upd = _with_retries(f"{model.name} merge", sink.merge, model.table, tenant, source, batch)
            res.inserted += ins
            res.updated += upd
        batch.clear()

    try:
        client.token()
        tenant = res.tenant_id = client.organization_id
        stored = sink.get_watermark(tenant, model.name)
        since = None if (opts.full_refresh or stored is None) else stored - opts.lookback
        res.watermark_before = _fmt(stored)

        # the table is created by the first MERGE, from the columns its rows carry
        for row in client.read_all(model.service, model.alias, workers=opts.page_workers):
            res.read += 1
            updated_at = parse_ts(row.get("updated_at"))
            if updated_at and (newest is None or updated_at > newest):
                newest = updated_at
            if is_changed(updated_at, since):
                staged = StagedRow.from_row(row, updated_at)
                batch.append(staged)
                res.sent += 1
                key = staged.key
            else:
                key = row_key(row)
            if opts.detect_deletes:
                seen.append(key)
            if len(batch) >= opts.batch_size:
                flush()
            if opts.limit and res.read >= opts.limit:
                complete = False
                break
        flush()

        if opts.dry_run:
            res.status, res.watermark_after = "dry_run", res.watermark_before
        elif not complete:
            res.status, res.watermark_after = "partial", res.watermark_before
            res.note = f"stopped at limit {opts.limit}: deletes not checked, watermark not moved"
            sink.save_state(tenant, model, source, watermark=stored, status="partial", error=None,
                            started=started, counts=res.as_dict())
        else:
            sink.ensure_table(model.table, {})          # a model with no rows yet still gets its table
            if opts.detect_deletes:
                res.deleted, res.restored, res.note = _with_retries(
                    f"{model.name} delete check", sink.flag_deleted, model.table, tenant, seen,
                    opts.max_delete_fraction)
            wm = next_watermark(stored, newest, started)
            res.watermark_after = _fmt(wm)
            sink.save_state(tenant, model, source, watermark=wm, status="ok", error=res.note,
                            started=started, counts=res.as_dict())
    except Exception as e:
        res.status, res.error = "failed", f"{type(e).__name__}: {e}"
        res.watermark_after = res.watermark_before
        log.error("%s failed after %d rows read: %s", model.name, res.read, res.error)
        if not opts.dry_run and tenant is not None:
            try:
                sink.save_state(tenant, model, source, watermark=stored, status="failed", error=res.error,
                                started=started, counts=res.as_dict())
            except Exception:
                log.exception("%s: could not record the failure in %s", model.name, STATE_TABLE)
    res.seconds = round((clock() - started).total_seconds(), 1)
    log.info("%-40s read %7d  sent %7d  +%d ~%d -%d  wm %s → %s%s", model.name, res.read, res.sent,
             res.inserted, res.updated, res.deleted, res.watermark_before, res.watermark_after,
             f"  ({res.note})" if res.note else "")
    return res


# ── many models ───────────────────────────────────────────────────────────

def discover_models(client, services: Iterable[str]) -> list[Model]:
    """Every model the gateway will read on these services."""
    found = []
    for service in services:
        for entry in client.models(service):
            if "read" in (entry.get("operations") or ["read"]):
                found.append(Model(service, entry["alias"]))
    return sorted(found)


@contextmanager
def _sink(database: str, schema: str):
    conn = snowflake_connect(database, schema)
    try:
        yield SnowflakeSink(conn, database, schema)
    finally:
        conn.close()


def sync(source: str, models: list[Model], *, opts: SyncOptions = SyncOptions(), workers: int = 4,
         database: str = DEFAULT_DATABASE, schema: str = DEFAULT_SCHEMA, client=None,
         sink_factory=None) -> list[ModelResult]:
    """Sync these models for one V3 source, `workers` models at a time, each on its own Snowflake session.
    A failing model doesn't stop the others; it comes back with status 'failed'."""
    from v3_to_v3_client import V3Client
    client = client or V3Client(source)
    client.token()
    sink_factory = sink_factory or (lambda: _sink(database, schema))
    if not opts.dry_run:
        with sink_factory() as sink:
            sink.ensure_schema()

    def one(model: Model) -> ModelResult:
        try:
            with sink_factory() as sink:
                return sync_model(client, sink, model, source=source, opts=opts)
        except Exception as e:            # couldn't even open a Snowflake session
            log.error("%s failed: %s", model.name, e)
            return ModelResult(model.name, model.table, client.organization_id, status="failed",
                               error=f"{type(e).__name__}: {e}")

    with ThreadPoolExecutor(max(1, workers), thread_name_prefix="v3sync") as pool:
        return list(pool.map(one, models))


def summarize(results: list[ModelResult | dict]) -> str:
    rows = [r.as_dict() if isinstance(r, ModelResult) else r for r in results]
    lines = [f"{'model':40} {'status':8} {'read':>8} {'sent':>8} {'ins':>7} {'upd':>7} {'del':>6}  watermark"]
    for r in rows:
        lines.append(f"{r['model']:40} {r['status']:8} {r['read']:>8} {r['sent']:>8} {r['inserted']:>7} "
                     f"{r['updated']:>7} {r['deleted']:>6}  {r.get('watermark_after') or '-'}"
                     + (f"  ERROR {r['error']}" if r.get("error") else "")
                     + (f"  NOTE {r['note']}" if r.get("note") else ""))
    failed = sum(r["status"] == "failed" for r in rows)
    lines.append(f"{len(rows)} model(s), {failed} failed; "
                 f"{sum(r['inserted'] for r in rows)} inserted, {sum(r['updated'] for r in rows)} updated, "
                 f"{sum(r['deleted'] for r in rows)} flagged deleted")
    return "\n".join(lines)


def main(argv=None) -> int:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    from v3_to_v3_client import SERVICES, V3Client

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--source", required=True, help="V3Client key, e.g. gyno (AFYA_GYNO_* in .env)")
    p.add_argument("--models", default="", help="service:alias list; empty = every readable model")
    p.add_argument("--services", default=",".join(SERVICES), help="services to discover models on")
    p.add_argument("--full-refresh", action="store_true", help="ignore the watermark; send every row")
    p.add_argument("--lookback-minutes", type=int, default=int(DEFAULT_LOOKBACK.total_seconds() // 60))
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--no-deletes", action="store_true", help="don't flag rows V3 no longer returns")
    p.add_argument("--max-delete-fraction", type=float, default=DEFAULT_MAX_DELETE_FRACTION)
    p.add_argument("--workers", type=int, default=4, help="models synced at once")
    p.add_argument("--page-workers", type=int, default=4, help="pages read ahead per model")
    p.add_argument("--limit", type=int, default=0, help="stop each model after N rows (canary)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--database", default=DEFAULT_DATABASE)
    p.add_argument("--schema", default=DEFAULT_SCHEMA)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(threadName)s %(message)s")

    client = V3Client(a.source)
    models = parse_models(a.models) or discover_models(client, [s.strip() for s in a.services.split(",") if s.strip()])
    opts = SyncOptions(full_refresh=a.full_refresh, lookback=timedelta(minutes=a.lookback_minutes),
                       batch_size=a.batch_size, detect_deletes=not a.no_deletes,
                       max_delete_fraction=a.max_delete_fraction, page_workers=a.page_workers,
                       limit=a.limit, dry_run=a.dry_run)
    log.info("%s → %s.%s: %d model(s)%s", a.source, a.database, a.schema, len(models),
             " (dry run)" if a.dry_run else "")
    results = sync(a.source, models, opts=opts, workers=a.workers, database=a.database, schema=a.schema,
                   client=client)
    print(summarize(results))
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
