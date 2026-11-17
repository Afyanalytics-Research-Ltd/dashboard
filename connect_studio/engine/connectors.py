"""Connectors: one declarative registry for every kind of source and destination.

Each connector is a ``ConnectorSpec`` (what the UI renders: label, icon,
category, form fields, which roles it can play, which driver it needs) plus a
class that knows how to

  * ``test()``                      — prove the credentials work
  * ``discover(options, progress)`` — produce a schema snapshot (dbatlas does
                                      the introspection, profiling and
                                      relationship inference)
  * ``read(table, ...)``            — stream rows in batches (for syncs)
  * ``writer()``                    — a destination writer (SQL engines)

Adding a connector = one ``register(ConnectorSpec(...), cls)`` call; the UI
catalog, forms, validation and secret handling follow from the spec.

Families:
  SQL (SQLAlchemy)  postgres · mysql · mariadb · mssql · oracle · snowflake ·
                    sqlite · duckdb · any SQLAlchemy URL
  Staged (→ DuckDB) files (CSV/TSV/Excel/JSON/NDJSON/Parquet) · declarative
                    REST API · Afya Extraction API
  Schema only       schema.json / SQL DDL / dbatlas.json / mappings.json
"""

from __future__ import annotations

import base64
import importlib.util
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from . import snapshot as snap_mod

logger = logging.getLogger(__name__)

Progress = Callable[..., None]


def _noop(*_a, **_k):
    return None


class ConnectorError(Exception):
    """A connector could not do what was asked; the message is user-facing."""


# ─── specs ────────────────────────────────────────────────────────────────

@dataclass
class FieldSpec:
    name: str
    label: str
    type: str = "text"            # text|number|password|textarea|select|checkbox|code|file
    required: bool = False
    secret: bool = False
    placeholder: str = ""
    help: str = ""
    default: Any = None
    options: list | None = None   # [{"value","label"}] for select
    group: str = "Connection"     # form section
    advanced: bool = False
    show_if: dict | None = None   # {"field": "auth_type", "in": ["bearer"]}
    rows: int | None = None


@dataclass
class ConnectorSpec:
    key: str
    label: str
    category: str                 # Databases · Warehouses · Files · APIs · Schema
    icon: str                     # bootstrap-icons name
    color: str                    # brand-ish accent for the tile
    description: str
    fields: list[FieldSpec] = field(default_factory=list)
    roles: tuple[str, ...] = ("source", "target", "destination")
    driver: str | None = None     # importable module the connector needs
    pip: str | None = None        # what to install if the driver is missing
    uploads: tuple[str, ...] = () # accepted upload extensions (files/schema connectors)
    multi_upload: bool = False
    tagline: str = ""
    beta: bool = False

    def driver_ok(self) -> bool:
        return not self.driver or importlib.util.find_spec(self.driver) is not None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["driver_ok"] = self.driver_ok()
        d["roles"] = list(self.roles)
        d["uploads"] = list(self.uploads)
        return d

    @property
    def secret_fields(self) -> set[str]:
        return {f.name for f in self.fields if f.secret}


REGISTRY: dict[str, tuple[ConnectorSpec, type]] = {}


def register(spec: ConnectorSpec, cls: type) -> None:
    REGISTRY[spec.key] = (spec, cls)


def get_spec(kind: str) -> ConnectorSpec:
    try:
        return REGISTRY[kind][0]
    except KeyError:
        raise ConnectorError(f"Unknown connector type: {kind}") from None


def catalog() -> list[dict]:
    return [spec.to_dict() for spec, _ in REGISTRY.values()]


def create(kind: str, config: dict, *, files: list[Path] | None = None, workdir: Path | None = None) -> "Connector":
    spec, cls = REGISTRY.get(kind) or (None, None)
    if not spec:
        raise ConnectorError(f"Unknown connector type: {kind}")
    if not spec.driver_ok():
        raise ConnectorError(f"The {spec.label} driver is not installed on this server (pip install {spec.pip or spec.driver}).")
    return cls(spec, config or {}, files=files or [], workdir=workdir)


def validate_config(kind: str, config: dict, *, has_files: bool = False, existing_secrets: set[str] = frozenset()) -> list[str]:
    """Missing required fields (secrets already stored count as present)."""
    spec = get_spec(kind)
    problems = []
    for f in spec.fields:
        if not f.required or f.type == "file":
            continue
        if f.show_if and str(config.get(f.show_if["field"], "")) not in [str(v) for v in f.show_if["in"]]:
            continue
        if f.secret and f.name in existing_secrets:
            continue
        if config.get(f.name) in (None, ""):
            problems.append(f"{f.label} is required.")
    if spec.uploads and not has_files and not config.get("path"):
        problems.append("Upload at least one file.")
    return problems


# ─── base ─────────────────────────────────────────────────────────────────

DEFAULT_DISCOVER = {"profile": True, "sample_rows": 1000, "exact_counts": True, "max_tables": 400,
                    "schemas": [], "tables": []}


class Connector:
    can_read = True
    can_write = False

    def __init__(self, spec: ConnectorSpec, config: dict, *, files: list[Path], workdir: Path | None = None):
        self.spec = spec
        self.config = config
        self.files = [Path(f) for f in files]
        self.workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="connect-studio-"))
        self.workdir.mkdir(parents=True, exist_ok=True)

    def cfg(self, name: str, default: Any = None) -> Any:
        v = self.config.get(name)
        if v in (None, ""):
            fdef = next((f.default for f in self.spec.fields if f.name == name), None)
            return default if fdef is None else fdef
        return v

    # interface
    def test(self) -> dict:
        raise NotImplementedError

    def discover(self, options: dict | None = None, progress: Progress = _noop) -> dict:
        raise NotImplementedError

    def read(self, table: str, *, columns: list[str] | None = None, cursor_column: str | None = None,
             cursor_value: Any = None, batch_size: int = 1000, limit: int | None = None) -> Iterator[list[dict]]:
        raise ConnectorError(f"{self.spec.label} cannot be read from.")

    def writer(self) -> "SQLWriter":
        raise ConnectorError(f"{self.spec.label} cannot be a sync destination.")

    def close(self) -> None:
        pass


def _options(options: dict | None) -> dict:
    out = dict(DEFAULT_DISCOVER)
    out.update({k: v for k, v in (options or {}).items() if v is not None})
    out["sample_rows"] = max(10, min(int(out.get("sample_rows") or 1000), 20000))
    return out


def _csv_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in re.split(r"[,\n]", str(value)) if v.strip()]


# ─── DuckDB staging + dbatlas ─────────────────────────────────────────────

def _safe_ident(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name)).strip("_") or "t"
    return f"t_{s}" if s[0].isdigit() else s


def _frame(rows: list[dict]):
    """Rows (possibly nested JSON) → flat DataFrame; nested lists/objects become JSON text."""
    import pandas as pd

    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows, max_level=1, sep="__")
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda v: json.dumps(v, default=str) if isinstance(v, (list, dict)) else v)
    return df


def stage_frames(frames: dict[str, Any], path: Path) -> dict[str, str]:
    """Write DataFrames into a DuckDB file. Returns {staged_name: original_name}."""
    import duckdb

    names = {}
    con = duckdb.connect(str(path))
    try:
        for original, df in frames.items():
            staged = _safe_ident(original)
            while staged in names:
                staged += "_"
            names[staged] = original
            if df is None or df.empty and len(df.columns) == 0:
                con.execute(f'CREATE OR REPLACE TABLE "{staged}" (_empty VARCHAR)')
                continue
            con.register("_frame", df)
            con.execute(f'CREATE OR REPLACE TABLE "{staged}" AS SELECT * FROM _frame')
            con.unregister("_frame")
    finally:
        con.close()
    return names


def dbatlas_run(conn_type: str, *, url: str | None = None, database: str | None = None,
                schemas: list[str] | None = None, tables: list[str] | None = None,
                profile: bool = True, sample_rows: int = 1000, progress: Progress = _noop,
                exact_counts: bool = True, max_tables: int = 400):
    """dbatlas introspect → (per-table) profile → analyze, with progress callbacks."""
    from dbatlas import Project
    from dbatlas.adapters import create_adapter
    from dbatlas.config import ConnectionConfig, DbAtlasConfig, IncludeConfig, PIIConfig, ProfilingConfig

    cfg = DbAtlasConfig(
        connection=ConnectionConfig(type=conn_type, url=url, database=database),
        include=IncludeConfig(schemas=tuple(schemas or ()), tables=tuple(tables or ())),
        profiling=ProfilingConfig(enabled=profile, sample_rows=sample_rows, cache=False),
        pii=PIIConfig(detect=True),
    )
    project = Project(cfg)
    progress(msg="Introspecting tables, columns, keys and constraints…", phase="introspect")
    db = project.introspect()
    all_tables = db.tables
    if len(all_tables) > max_tables:
        progress(msg=f"{len(all_tables)} tables found — keeping the first {max_tables} (raise max tables to see more).", level="warning")
        keep = {(t.schema_name, t.name) for t in all_tables[:max_tables]}
        db = project._replace_tables(db, tuple(t for t in all_tables if (t.schema_name, t.name) in keep))
    progress(msg=f"Found {len(db.tables)} tables across {len([s for s in db.schemas if s.tables])} schema(s).")
    if profile or exact_counts:
        profiled = []
        n = len(db.tables)
        with create_adapter(cfg.connection) as adapter:
            for i, table in enumerate(db.tables, start=1):
                progress(pct=10 + 70 * i / max(n, 1), msg=f"Profiling {table.qualified_name}", level="debug", phase="profile")
                try:
                    profiled.append(table.with_profile(adapter.profile_table(table, sample_rows)))
                except Exception as e:  # one unreadable table must not sink the whole extraction
                    progress(msg=f"Could not profile {table.qualified_name}: {str(e)[:160]}", level="warning")
                    profiled.append(table)
        db = project._replace_tables(db, tuple(profiled))
    progress(msg="Inferring relationships and detecting PII…", phase="analyze", pct=85)
    return project.analyze(db)


def snapshot_from_staged(path: Path, names: dict[str, str], *, dialect: str, database: str,
                         options: dict, progress: Progress, extractor: str) -> dict:
    db = dbatlas_run("duckdb", database=str(path), profile=options["profile"], sample_rows=options["sample_rows"],
                     progress=progress, max_tables=options["max_tables"])
    snap = snap_mod.from_dbatlas(db, dialect=dialect, extractor=extractor)
    # restore original stream names (staging sanitized them)
    rename = {staged: original for staged, original in names.items()}
    for t in snap["tables"]:
        t["name"] = rename.get(t["name"], t["name"])
        t["schema"] = ""
        t.pop("key", None)
    for r in snap["relationships"]:
        r["from"] = rename.get(r["from"].split(".")[-1], r["from"])
        r["to"] = rename.get(r["to"].split(".")[-1], r["to"])
    snap["database"] = database
    return snap_mod.finalize(snap, infer=False)


# ─── SQL connectors (SQLAlchemy) ──────────────────────────────────────────

HOST_FIELDS = lambda port, db_label="Database", db_required=True: [  # noqa: E731
    FieldSpec("host", "Host", required=True, placeholder="db.example.com"),
    FieldSpec("port", "Port", type="number", default=port),
    FieldSpec("database", db_label, required=db_required),
    FieldSpec("username", "Username", required=True),
    FieldSpec("password", "Password", type="password", secret=True),
]
SCHEMA_FIELD = lambda default="", help_="Comma-separated. Blank = every non-system schema.": FieldSpec(  # noqa: E731
    "schemas", "Schemas", placeholder=default or "public, sales", default=default, help=help_, group="Scope")
TABLES_FIELD = FieldSpec("tables", "Only these tables", placeholder="patients, visits", group="Scope", advanced=True,
                         help="Comma-separated table names. Blank = all tables.")
SYSTEM_SCHEMAS = {"information_schema", "pg_catalog", "pg_toast", "sys", "mysql", "performance_schema",
                  "guest", "db_owner", "db_accessadmin", "db_securityadmin", "db_ddladmin", "db_backupoperator",
                  "db_datareader", "db_datawriter", "db_denydatareader", "db_denydatawriter", "ctxsys", "xdb",
                  "outln", "system", "account_usage"}


class SQLConnector(Connector):
    """Any SQLAlchemy-reachable database: dbatlas reads metadata + row counts,
    a row sample per table is profiled in DuckDB (dbatlas' SQL adapter only counts rows)."""

    can_write = True
    _engine = None

    # URL per kind
    def url(self):
        from sqlalchemy.engine import URL

        k = self.spec.key
        c = self.cfg
        if k == "sqlalchemy":
            return c("url")
        if k in ("sqlite", "duckdb"):
            path = str(self.files[0]) if self.files else c("path")
            if not path:
                raise ConnectorError("Upload a database file or give its path on the server.")
            return f"{k}:///{path}"
        if k == "snowflake":
            from snowflake.sqlalchemy import URL as SFURL
            return SFURL(account=c("account"), user=c("username"), password=c("password") or "",
                         database=c("database") or "", schema=(self._schemas() or [""])[0] or "",
                         warehouse=c("warehouse") or "", role=c("role") or "")
        driver = {"postgres": "postgresql+psycopg2", "mysql": "mysql+pymysql", "mariadb": "mysql+pymysql",
                  "mssql": "mssql+pyodbc", "oracle": "oracle+oracledb", "redshift": "postgresql+psycopg2",
                  "cockroach": "postgresql+psycopg2"}[k]
        query = {}
        if k == "mssql":
            query["driver"] = c("odbc_driver", "ODBC Driver 18 for SQL Server")
            if c("trust_cert", True):
                query["TrustServerCertificate"] = "yes"
        if k in ("postgres", "redshift", "cockroach") and c("sslmode"):
            query["sslmode"] = c("sslmode")
        if k == "oracle":
            query["service_name"] = c("database")
        return URL.create(driver, username=c("username"), password=c("password") or None, host=c("host"),
                          port=int(c("port")) if c("port") else None,
                          database=None if k == "oracle" else c("database"), query=query)

    def url_string(self) -> str:
        u = self.url()
        return u if isinstance(u, str) else u.render_as_string(hide_password=False)

    def engine(self):
        if self._engine is None:
            from sqlalchemy import create_engine

            connect_args = {}
            if self.spec.key == "snowflake" and self.cfg("private_key"):
                connect_args["private_key"] = _snowflake_private_key(self.cfg("private_key"), self.cfg("private_key_passphrase"))
            if self.spec.key in ("postgres", "redshift", "cockroach"):
                connect_args["connect_timeout"] = 15
            if self.spec.key in ("mysql", "mariadb"):
                connect_args["connect_timeout"] = 15
            self._engine = create_engine(self.url_string(), pool_pre_ping=True, connect_args=connect_args)
        return self._engine

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None

    def _schemas(self) -> list[str]:
        explicit = _csv_list(self.cfg("schemas"))
        if explicit:
            return explicit
        if self.spec.key in ("mysql", "mariadb"):
            return [self.cfg("database")] if self.cfg("database") else []
        return []

    def test(self) -> dict:
        from sqlalchemy import inspect, text

        t0 = time.time()
        try:
            with self.engine().connect() as conn:
                version = None
                q = {"postgres": "select version()", "redshift": "select version()", "cockroach": "select version()",
                     "mysql": "select version()", "mariadb": "select version()",
                     "mssql": "select @@version", "oracle": "select banner from v$version",
                     "snowflake": "select current_version()", "sqlite": "select sqlite_version()",
                     "duckdb": "select version()"}.get(self.spec.key)
                if q:
                    try:
                        version = str(conn.execute(text(q)).scalar())[:120]
                    except Exception:
                        version = None
                insp = inspect(conn)
                schemas = [s for s in insp.get_schema_names() if s.lower() not in SYSTEM_SCHEMAS and not s.lower().startswith("pg_")]
                wanted = self._schemas() or schemas[:1] or [None]
                tables = sum(len(insp.get_table_names(schema=s)) for s in wanted[:5])
        except Exception as e:
            raise ConnectorError(_clean_db_error(e)) from None
        return {"ok": True, "latency_ms": int((time.time() - t0) * 1000), "version": version,
                "message": f"Connected — {tables} tables in {', '.join(str(s) for s in wanted[:5] if s) or 'default schema'}",
                "schemas": schemas[:200]}

    def discover(self, options: dict | None = None, progress: Progress = _noop) -> dict:
        from sqlalchemy import inspect, select, table as sa_table, literal_column

        o = _options(options)
        schemas = o.get("schemas") or self._schemas()
        if not schemas and self.spec.key not in ("sqlite", "duckdb"):
            with self.engine().connect() as conn:
                schemas = [s for s in inspect(conn).get_schema_names()
                           if s.lower() not in SYSTEM_SCHEMAS and not s.lower().startswith(("pg_", "crdb_"))]
        tables = o.get("tables") or _csv_list(self.cfg("tables"))
        conn_type = {"sqlite": "sqlite", "duckdb": "duckdb"}.get(self.spec.key, "sqlalchemy")
        kwargs = {"database": str(self.files[0]) if self.files else self.cfg("path")} if conn_type != "sqlalchemy" else {"url": self.url_string()}
        if self.spec.key == "snowflake" and self.cfg("private_key"):
            # dbatlas builds its own engine from a URL; key-pair auth needs ours
            return self._discover_with_engine(o, schemas, tables, progress)
        try:
            db = dbatlas_run(conn_type, schemas=schemas, tables=tables, profile=False,
                             exact_counts=o["exact_counts"], sample_rows=o["sample_rows"], progress=progress,
                             max_tables=o["max_tables"], **kwargs)
        except Exception as e:
            raise ConnectorError(_clean_db_error(e)) from None
        snapshot = snap_mod.from_dbatlas(db, dialect=self.engine().dialect.name)
        if o["profile"]:
            self._profile_samples(snapshot, o, progress)
        return snapshot

    def _discover_with_engine(self, o, schemas, tables, progress) -> dict:
        """Metadata via SQLAlchemy inspector directly (for engines dbatlas can't build itself)."""
        from sqlalchemy import inspect, text

        snap = snap_mod.empty_snapshot(dialect=self.engine().dialect.name, database=self.cfg("database") or "",
                                       extractor="sqlalchemy inspector")
        with self.engine().connect() as conn:
            insp = inspect(conn)
            for schema in schemas or [None]:
                names = insp.get_table_names(schema=schema)
                if tables:
                    names = [n for n in names if n in tables]
                for name in names[: o["max_tables"]]:
                    progress(msg=f"Reading {schema}.{name}", level="debug")
                    pk = (insp.get_pk_constraint(name, schema=schema) or {}).get("constrained_columns") or []
                    cols = [{"name": c["name"], "type": str(c["type"]), "nullable": c.get("nullable", True),
                             "pk": c["name"] in pk, "default": str(c.get("default")) if c.get("default") else None,
                             "comment": c.get("comment")} for c in insp.get_columns(name, schema=schema)]
                    rows = None
                    if o["exact_counts"]:
                        try:
                            rows = conn.execute(text(f"select count(*) from {_q(self.engine(), schema, name)}")).scalar()
                        except Exception:
                            rows = None
                    snap["tables"].append({"schema": schema or "", "name": name, "columns": cols, "primary_key": pk, "rows": rows})
                    for fk in insp.get_foreign_keys(name, schema=schema):
                        snap["relationships"].append({"from": f"{schema}.{name}" if schema else name,
                                                      "from_columns": fk["constrained_columns"],
                                                      "to": f"{fk.get('referred_schema') or schema}.{fk['referred_table']}" if (fk.get('referred_schema') or schema) else fk["referred_table"],
                                                      "to_columns": fk["referred_columns"], "confidence": 1.0, "kind": "foreign_key"})
        snapshot = snap_mod.finalize(snap)
        if o["profile"]:
            self._profile_samples(snapshot, o, progress)
        return snapshot

    def _profile_samples(self, snapshot: dict, o: dict, progress: Progress) -> None:
        """Pull up to N rows per table, profile them with dbatlas in DuckDB, merge profiles back."""
        import pandas as pd
        from sqlalchemy import text

        frames, keymap = {}, {}
        n = len(snapshot["tables"])
        for i, t in enumerate(snapshot["tables"], start=1):
            progress(pct=40 + 35 * i / max(n, 1), msg=f"Sampling {t['key']}", level="debug", phase="sample")
            try:
                sql = _limit_sql(self.engine(), _q(self.engine(), t.get("schema") or None, t["name"]), o["sample_rows"])
                with self.engine().connect() as conn:
                    df = pd.read_sql(text(sql), conn)
                frames[t["key"]] = df
            except Exception as e:
                progress(msg=f"Could not sample {t['key']}: {_clean_db_error(e)[:160]}", level="warning")
        if not frames:
            return
        path = self.workdir / "sample.duckdb"
        if path.exists():
            path.unlink()
        names = stage_frames(frames, path)
        try:
            db = dbatlas_run("duckdb", database=str(path), profile=True, exact_counts=False,
                             sample_rows=o["sample_rows"], progress=lambda **k: None, max_tables=10_000)
        except Exception as e:
            progress(msg=f"Sample profiling failed: {e}", level="warning")
            return
        profiled = snap_mod.from_dbatlas(db)
        by_staged = {t["name"]: t for t in profiled["tables"]}
        for staged, original in names.items():
            src = by_staged.get(staged)
            dst = next((t for t in snapshot["tables"] if t["key"] == original), None)
            if not src or not dst:
                continue
            cols = snap_mod.column_index(src)
            for c in dst["columns"]:
                p = cols.get(c["name"], {}).get("profile")
                if p:
                    p["sampled"] = True
                    c["profile"] = p
        snapshot["stats"]["profiled"] = True
        snapshot["stats"]["sample_rows"] = o["sample_rows"]

    def read(self, table, *, columns=None, cursor_column=None, cursor_value=None, batch_size=1000, limit=None):
        from sqlalchemy import text

        schema, _, name = table.rpartition(".") if "." in table and self.spec.key not in ("sqlite", "duckdb") else ("", "", table)
        eng = self.engine()
        cols = ", ".join(_qi(eng, c) for c in columns) if columns else "*"
        sql = f"select {cols} from {_q(eng, schema or None, name)}"
        params = {}
        if cursor_column:
            if cursor_value is not None:
                sql += f" where {_qi(eng, cursor_column)} > :cursor"
                params["cursor"] = cursor_value
            sql += f" order by {_qi(eng, cursor_column)}"
        if limit:
            sql = _limit_sql(eng, f"({sql}) _q", limit) if not cursor_column else sql
        with eng.connect().execution_options(stream_results=True) as conn:
            result = conn.execute(text(sql), params)
            keys = list(result.keys())
            fetched = 0
            while True:
                chunk = result.fetchmany(batch_size)
                if not chunk:
                    break
                fetched += len(chunk)
                yield [dict(zip(keys, row)) for row in chunk]
                if limit and fetched >= limit:
                    break

    def writer(self) -> "SQLWriter":
        return SQLWriter(self.engine(), schema=(self._schemas() or [None])[0])


def _snowflake_private_key(pem: str, passphrase: str | None):
    from cryptography.hazmat.primitives import serialization

    key = serialization.load_pem_private_key(pem.encode(), password=passphrase.encode() if passphrase else None)
    return key.private_bytes(encoding=serialization.Encoding.DER, format=serialization.PrivateFormat.PKCS8,
                             encryption_algorithm=serialization.NoEncryption())


def _qi(engine, ident: str) -> str:
    return engine.dialect.identifier_preparer.quote(ident)


def _q(engine, schema: str | None, name: str) -> str:
    prep = engine.dialect.identifier_preparer
    return f"{prep.quote_schema(schema)}.{prep.quote(name)}" if schema else prep.quote(name)


def _limit_sql(engine, from_expr: str, n: int) -> str:
    name = engine.dialect.name
    if not from_expr.lstrip().startswith("("):
        from_expr = from_expr
    if name == "mssql":
        return f"select top {int(n)} * from {from_expr}"
    if name == "oracle":
        return f"select * from {from_expr} fetch first {int(n)} rows only"
    return f"select * from {from_expr} limit {int(n)}"


def _clean_db_error(e: Exception) -> str:
    msg = str(getattr(e, "orig", None) or e)
    msg = re.sub(r"\(Background on this error at: [^)]+\)", "", msg)
    msg = re.sub(r"\[SQL: .*", "", msg, flags=re.S)
    msg = re.sub(r"(password|pwd)=[^&\s;]+", r"\1=•••", msg, flags=re.I)
    return msg.strip().splitlines()[0][:400] if msg.strip() else type(e).__name__


# ─── SQL destination writer ───────────────────────────────────────────────

class SQLWriter:
    """Create/replace/append/merge tables in any SQLAlchemy destination."""

    def __init__(self, engine, schema: str | None = None):
        self.engine = engine
        self.schema = schema

    def _type(self, family: str, *, key: bool = False):
        from sqlalchemy import JSON, BigInteger, Boolean, Date, DateTime, Float, LargeBinary, Numeric, String, Text, Time

        dialect = self.engine.dialect.name
        if family == "integer":
            return BigInteger()
        if family == "decimal":
            return Float() if dialect in ("sqlite",) else Numeric(38, 10)
        if family == "boolean":
            return Boolean()
        if family == "date":
            return Date()
        if family == "datetime":
            return DateTime()
        if family == "time":
            return Time()
        if family == "json":
            return JSON() if dialect in ("postgresql", "mysql", "sqlite", "duckdb") else Text()
        if family == "binary":
            return LargeBinary()
        if family == "uuid":
            return String(36)
        return String(255) if key or dialect in ("mysql", "mssql", "oracle") and key else Text()

    def ensure_table(self, name: str, columns: list[dict], *, primary_key: list[str] | None = None, replace: bool = False):
        from sqlalchemy import Column, MetaData, Table, inspect

        md = MetaData(schema=self.schema)
        pk = set(primary_key or [])
        # autoincrement=False: synced keys come from the source (and DuckDB has no SERIAL types)
        table = Table(name, md, *[Column(c["name"], self._type(c.get("family") or "text", key=c["name"] in pk),
                                         primary_key=c["name"] in pk, nullable=c["name"] not in pk, autoincrement=False)
                                  for c in columns])
        with self.engine.begin() as conn:
            exists = inspect(conn).has_table(name, schema=self.schema)
            if exists and replace:
                table.drop(conn)
                exists = False
            if not exists:
                table.create(conn)
            else:
                # schema evolution: add new columns
                have = {c["name"] for c in inspect(conn).get_columns(name, schema=self.schema)}
                for c in table.columns:
                    if c.name not in have:
                        from sqlalchemy.schema import CreateColumn
                        ddl = CreateColumn(c).compile(dialect=self.engine.dialect)
                        conn.exec_driver_sql(f"ALTER TABLE {_q(self.engine, self.schema, name)} ADD {ddl}")
        return Table(name, MetaData(schema=self.schema), autoload_with=self.engine)

    def truncate(self, table) -> None:
        with self.engine.begin() as conn:
            conn.execute(table.delete())

    def write(self, table, rows: list[dict], *, merge_keys: list[str] | None = None) -> int:
        if not rows:
            return 0
        cols = {c.name for c in table.columns}
        clean = [{k: v for k, v in r.items() if k in cols} for r in rows]
        with self.engine.begin() as conn:
            if merge_keys:
                # portable upsert: delete the incoming keys, then insert
                from sqlalchemy import and_, tuple_
                keys = [table.c[k] for k in merge_keys if k in table.c]
                if keys:
                    values = list({tuple(r.get(k.name) for k in keys) for r in clean})
                    for i in range(0, len(values), 500):
                        chunk = values[i:i + 500]
                        if len(keys) == 1:
                            conn.execute(table.delete().where(keys[0].in_([v[0] for v in chunk])))
                        else:
                            conn.execute(table.delete().where(tuple_(*keys).in_(chunk)))
            conn.execute(table.insert(), clean)
        return len(clean)


# ─── staged connectors ────────────────────────────────────────────────────

class StagedConnector(Connector):
    """Sources whose rows come from somewhere other than SQL: stage a sample in
    DuckDB and let dbatlas introspect + profile + infer relationships there."""

    dialect = "duckdb"

    def streams(self) -> list[str]:
        raise NotImplementedError

    def iter_rows(self, stream: str, *, limit: int | None = None) -> Iterator[list[dict]]:
        raise NotImplementedError

    def discover(self, options=None, progress: Progress = _noop) -> dict:
        o = _options(options)
        wanted = set(o.get("tables") or [])
        streams = [s for s in self.streams() if not wanted or s in wanted][: o["max_tables"]]
        if not streams:
            raise ConnectorError("Nothing to read — no tables/streams were found.")
        frames = {}
        for i, s in enumerate(streams, start=1):
            progress(pct=5 + 45 * i / len(streams), msg=f"Reading a sample of {s}", phase="sample")
            rows: list[dict] = []
            try:
                for batch in self.iter_rows(s, limit=o["sample_rows"]):
                    rows.extend(batch)
                    if len(rows) >= o["sample_rows"]:
                        break
            except ConnectorError:
                raise
            except Exception as e:
                progress(msg=f"Could not read {s}: {str(e)[:200]}", level="warning")
                continue
            frames[s] = _frame(rows[: o["sample_rows"]])
        if not frames:
            raise ConnectorError("No stream could be read.")
        path = self.workdir / "stage.duckdb"
        if path.exists():
            path.unlink()
        names = stage_frames(frames, path)
        snapshot = snapshot_from_staged(path, names, dialect=self.dialect, database=self.database_label(),
                                        options={**o, "profile": True}, progress=progress,
                                        extractor=f"dbatlas on a staged sample ({self.spec.label})")
        # sampled counts aren't true row counts
        for t in snapshot["tables"]:
            t["rows_sampled"] = True
        snapshot["stats"]["sample_rows"] = o["sample_rows"]
        return snapshot

    def database_label(self) -> str:
        return self.spec.label

    def read(self, table, *, columns=None, cursor_column=None, cursor_value=None, batch_size=1000, limit=None):
        sent = 0
        for batch in self.iter_rows(table, limit=limit):
            flat = _frame(batch).to_dict(orient="records") if batch and any(isinstance(v, (dict, list)) for v in batch[0].values()) else batch
            if cursor_column and cursor_value is not None:
                flat = [r for r in flat if r.get(cursor_column) is not None and str(r.get(cursor_column)) > str(cursor_value)]
            if columns:
                flat = [{c: r.get(c) for c in columns} for r in flat]
            for i in range(0, len(flat), batch_size):
                yield flat[i:i + batch_size]
            sent += len(flat)
            if limit and sent >= limit:
                break


class FilesConnector(StagedConnector):
    """CSV · TSV · Excel (every sheet) · JSON · NDJSON · Parquet — one table per file/sheet."""

    def _tables(self) -> dict[str, tuple[Path, str | None]]:
        out: dict[str, tuple[Path, str | None]] = {}
        for f in self.files:
            stem = re.sub(r"[^0-9a-zA-Z_]+", "_", f.stem).strip("_").lower() or "file"
            if f.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
                import openpyxl
                wb = openpyxl.load_workbook(f, read_only=True)
                for sheet in wb.sheetnames:
                    out[f"{stem}__{re.sub(r'[^0-9a-zA-Z_]+', '_', sheet).strip('_').lower()}" if len(wb.sheetnames) > 1 else stem] = (f, sheet)
                wb.close()
            else:
                out[stem] = (f, None)
        return out

    def streams(self) -> list[str]:
        return list(self._tables())

    def database_label(self) -> str:
        return ", ".join(f.name for f in self.files[:3]) + (" …" if len(self.files) > 3 else "")

    def _frame_for(self, stream: str, limit: int | None):
        import pandas as pd

        path, sheet = self._tables()[stream]
        ext = path.suffix.lower()
        delim = self.cfg("delimiter") or None
        if ext in (".csv", ".tsv", ".txt"):
            return pd.read_csv(path, sep="\t" if ext == ".tsv" else (delim or None), engine="python", nrows=limit,
                               encoding=self.cfg("encoding") or "utf-8", encoding_errors="replace")
        if ext in (".xlsx", ".xlsm", ".xls"):
            return pd.read_excel(path, sheet_name=sheet, nrows=limit)
        if ext == ".parquet":
            df = pd.read_parquet(path)
            return df.head(limit) if limit else df
        if ext in (".ndjson", ".jsonl"):
            return pd.read_json(path, lines=True, nrows=limit)
        if ext == ".json":
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            rows = _records(data, self.cfg("records_path"))
            return _frame(rows[:limit] if limit else rows)
        raise ConnectorError(f"Unsupported file type: {path.name}")

    def iter_rows(self, stream, *, limit=None):
        import numpy as np

        df = self._frame_for(stream, limit)
        df = df.replace({np.nan: None})
        records = df.to_dict(orient="records")
        for i in range(0, len(records), 5000):
            yield records[i:i + 5000]

    def test(self) -> dict:
        if not self.files:
            raise ConnectorError("Upload at least one file.")
        streams = self.streams()
        return {"ok": True, "message": f"{len(self.files)} file(s) → {len(streams)} table(s): {', '.join(streams[:6])}",
                "streams": streams}


def _dig(data: Any, path: str | None) -> Any:
    if not path:
        return data
    for part in str(path).strip(".").split("."):
        if isinstance(data, dict):
            data = data.get(part)
        elif isinstance(data, list) and part.isdigit():
            data = data[int(part)] if int(part) < len(data) else None
        else:
            return None
    return data


def _records(data: Any, path: str | None = None) -> list[dict]:
    """The list of records in a JSON payload: an explicit dotted path, or auto-detected."""
    if path:
        found = _dig(data, path)
        if isinstance(found, list):
            return [r if isinstance(r, dict) else {"value": r} for r in found]
        if isinstance(found, dict):
            return [found]
        return []
    if isinstance(data, list):
        return [r if isinstance(r, dict) else {"value": r} for r in data]
    if isinstance(data, dict):
        for key in ("data", "results", "items", "records", "rows", "value", "entries", "objects", "hits"):
            v = data.get(key)
            if isinstance(v, list):
                return [r if isinstance(r, dict) else {"value": r} for r in v]
            if isinstance(v, dict):
                inner = _records(v)
                if inner:
                    return inner
        lists = [v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
        if lists:
            return max(lists, key=len)
        return [data]
    return []


class RestConnector(StagedConnector):
    """Declarative REST/JSON connector (Airbyte-builder style): base URL, auth,
    streams, records path and pagination — no code per API."""

    dialect = "rest"

    def database_label(self) -> str:
        return urlparse(self.cfg("base_url") or "").netloc or "REST API"

    def _streams(self) -> list[dict]:
        raw = self.cfg("streams") or ""
        out = []
        text = str(raw).strip()
        if text.startswith("["):
            try:
                for s in json.loads(text):
                    if s.get("name") and s.get("path"):
                        out.append({"name": s["name"], "path": s["path"], "records": s.get("records", s.get("records_path")),
                                    "params": s.get("params") or {}, "cursor": s.get("cursor"),
                                    "method": (s.get("method") or "GET").upper(), "body": s.get("body")})
                return out
            except (ValueError, AttributeError):
                raise ConnectorError("Streams JSON is not valid: expected [{\"name\", \"path\", \"records\"?}].") from None
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # name = [METHOD ]/path?x=1[ {json body}][ -> records.path]
            m = re.match(r"^([\w\-]+)\s*[=:]\s*(?:(GET|POST|PUT|PATCH)\s+)?(\S+)(?:\s+(\{.*\}))?(?:\s*->\s*(\S*))?$", line, re.I)
            if not m:
                path = line.split()[0]
                name = re.sub(r"[^0-9a-zA-Z_]+", "_", urlparse(path).path.strip("/").split("/")[-1] or "records")
                out.append({"name": name, "path": path, "records": None, "params": {}, "method": "GET", "body": None})
                continue
            body = None
            if m.group(4):
                try:
                    body = json.loads(m.group(4))
                except ValueError:
                    raise ConnectorError(f"Stream {m.group(1)}: the request body is not valid JSON.") from None
            out.append({"name": m.group(1), "path": m.group(3), "records": m.group(5), "params": {},
                        "method": (m.group(2) or "GET").upper(), "body": body})
        return out

    def _session(self):
        import requests

        s = requests.Session()
        s.headers.update({"Accept": "application/json", "User-Agent": "connect-studio/1.0"})
        auth = self.cfg("auth_type", "none")
        if auth == "bearer" and self.cfg("token"):
            s.headers["Authorization"] = f"Bearer {self.cfg('token')}"
        elif auth == "basic":
            s.auth = (self.cfg("username") or "", self.cfg("password") or "")
        elif auth == "api_key_header" and self.cfg("token"):
            s.headers[self.cfg("api_key_name") or "X-API-Key"] = self.cfg("token")
        elif auth == "login":
            s.headers["Authorization"] = f"Bearer {self._login_token(s)}"
        for line in str(self.cfg("headers") or "").splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                s.headers[k.strip()] = v.strip()
        return s

    _login_cache = None

    def _login_token(self, session, force=False) -> str:
        """auth_type=login: POST the login body, read the token from the response."""
        if self._login_cache and not force and time.time() - self._login_cache[1] < 40 * 60:
            return self._login_cache[0]
        try:
            body = json.loads(self.cfg("login_body") or "{}")
        except ValueError:
            raise ConnectorError("Login body is not valid JSON.") from None
        r = session.post(self._url(self.cfg("login_path") or "/login", {}, auth=False), json=body,
                         timeout=int(self.cfg("timeout", 60)))
        if not r.ok:
            detail = "an HTML page — check the base URL and login path" if r.text.lstrip()[:1] == "<" else r.text[:160]
            raise ConnectorError(f"Login failed: HTTP {r.status_code} from {urlparse(self.cfg('base_url') or '').netloc} ({detail})")
        try:
            data = r.json()
        except ValueError:
            raise ConnectorError("The login response is not JSON.") from None
        paths = [self.cfg("token_path")] if self.cfg("token_path") else \
            ["access_token", "token", "data.token", "data.access_token", "accessToken", "jwt"]
        token = next((v for v in (_dig(data, p) for p in paths) if isinstance(v, str) and v), None)
        if not token:
            raise ConnectorError("No token in the login response — set “Token path”.")
        self._login_cache = (token, time.time())
        return token

    def _url(self, path: str, extra: dict, auth: bool = True) -> str:
        base = self.cfg("base_url") or ""
        url = path if path.startswith(("http://", "https://")) else urljoin(base.rstrip("/") + "/", path.lstrip("/"))
        parts = urlparse(url)
        q = dict(parse_qsl(parts.query))
        if auth and self.cfg("auth_type") == "api_key_query" and self.cfg("token"):
            q[self.cfg("api_key_name") or "api_key"] = self.cfg("token")
        q.update({k: v for k, v in extra.items() if v is not None})
        return urlunparse(parts._replace(query=urlencode(q)))

    def _get(self, session, url: str):
        return self._send(session, "GET", url)

    def _send(self, session, method: str, url: str, body=None):
        for attempt in range(4):
            r = session.request(method, url, timeout=int(self.cfg("timeout", 60)),
                                **({"json": body} if body is not None and method != "GET" else {}))
            if r.status_code == 401 and self.cfg("auth_type") == "login" and attempt == 0:
                session.headers["Authorization"] = f"Bearer {self._login_token(session, force=True)}"
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(int(r.headers.get("Retry-After") or 2 ** attempt))
                continue
            if r.status_code in (401, 403):
                raise ConnectorError(f"HTTP {r.status_code} from {urlparse(url).netloc} — check the credentials.")
            if not r.ok:
                raise ConnectorError(f"HTTP {r.status_code} from {url.split('?')[0]}: {r.text[:160]}")
            try:
                return r, r.json()
            except ValueError:
                raise ConnectorError(f"{url.split('?')[0]} did not return JSON.") from None
        raise ConnectorError(f"{url.split('?')[0]} kept failing (rate limited or down).")

    def streams(self) -> list[str]:
        return [s["name"] for s in self._streams()]

    def iter_rows(self, stream, *, limit=None):
        spec = next((s for s in self._streams() if s["name"] == stream), None)
        if not spec:
            raise ConnectorError(f"Unknown stream {stream}")
        session = self._session()
        mode = self.cfg("pagination", "none")
        size = int(self.cfg("page_size") or 100)
        max_pages = int(self.cfg("max_pages") or 1000)
        page_param = self.cfg("page_param") or {"page": "page", "offset": "offset", "cursor": "cursor"}.get(mode, "page")
        size_param = self.cfg("size_param") or ("per_page" if mode == "page" else "limit")
        page, offset, cursor, url_next = int(self.cfg("start_page") or 1), 0, None, None
        sent = 0
        for _ in range(max_pages):
            paging = {}
            if mode == "page":
                paging = {page_param: page, size_param: size}
            elif mode == "offset":
                paging = {page_param: offset, size_param: size}
            elif mode == "cursor" and cursor:
                paging = {page_param: cursor}
            method = spec.get("method") or "GET"
            if method == "GET":
                url = url_next or self._url(spec["path"], {**(spec.get("params") or {}), **paging})
                resp, data = self._get(session, url)
            else:                                   # POST-style reads: paging goes into the JSON body
                raw = spec.get("body")
                body = {**(raw or {}), **paging} if isinstance(raw, dict) or raw is None else raw
                resp, data = self._send(session, method, url_next or self._url(spec["path"], dict(spec.get("params") or {})), body)
            rows = _records(data, spec.get("records") if spec.get("records") else self.cfg("records_path"))
            if rows:
                yield rows
            sent += len(rows)
            if limit and sent >= limit:
                return
            if mode == "none" or not rows:
                return
            if mode == "page":
                page += 1
                if len(rows) < size:
                    return
            elif mode == "offset":
                offset += len(rows)
                if len(rows) < size:
                    return
            elif mode == "cursor":
                cursor = _dig(data, self.cfg("cursor_path") or "next_cursor")
                if not cursor:
                    return
            elif mode == "link":
                url_next = resp.links.get("next", {}).get("url") or _dig(data, self.cfg("cursor_path") or "next")
                if not url_next:
                    return

    def test(self) -> dict:
        streams = self._streams()
        if not streams:
            raise ConnectorError("Add at least one stream (e.g. `patients = /patients -> data`).")
        t0 = time.time()
        first = streams[0]
        rows = next(self.iter_rows(first["name"], limit=5), [])
        return {"ok": True, "latency_ms": int((time.time() - t0) * 1000),
                "message": f"{first['name']}: {len(rows)} record(s) on the first page · {len(streams)} stream(s) configured",
                "preview": rows[:3], "streams": [s["name"] for s in streams]}


class AfyaExtractionConnector(StagedConnector):
    """The Afya Extraction API gateway: login, then paged reads per namespace."""

    dialect = "afya-extraction"
    _token = None

    def database_label(self) -> str:
        return f"connection {self.cfg('connection_id')}"

    def _base(self) -> str:
        return (self.cfg("base_url") or os.getenv("AFYA_EXTRACTION_BASE_URL") or "https://afyapi.afyaanalytics.ai/api").rstrip("/")

    def _auth(self, session):
        import requests  # noqa: F401

        if self._token and time.time() - self._token[1] < 40 * 60:
            return self._token[0]
        user = self.cfg("username") or os.getenv("AFYA_EXTRACTION_USERNAME", "")
        pwd = self.cfg("password") or os.getenv("AFYA_EXTRACTION_PASSWORD", "")
        if not user or not pwd:
            raise ConnectorError("No Extraction API credentials — fill them in or set AFYA_EXTRACTION_USERNAME / _PASSWORD.")
        r = session.post(f"{self._base()}/auth/login", json={"username": user, "password": pwd}, timeout=60)
        if not r.ok:
            raise ConnectorError(f"Extraction login failed: HTTP {r.status_code}")
        body = r.json()
        token = body.get("token") or (body.get("data") or {}).get("token")
        if not token:
            raise ConnectorError("Extraction login returned no token.")
        self._token = (token, time.time())
        return token

    def _session(self):
        import requests
        return requests.Session()

    def streams(self) -> list[str]:
        explicit = _csv_list(self.cfg("tables"))
        if explicit:
            return explicit
        s = self._session()
        r = s.get(f"{self._base()}/connections/{self.cfg('connection_id')}",
                  headers={"Authorization": f"Bearer {self._auth(s)}", "Accept": "application/json"}, timeout=60)
        data = (r.json() or {}).get("data") or {} if r.ok else {}
        for key in ("tables", "namespaces", "models", "entities"):
            v = data.get(key)
            if isinstance(v, list) and v:
                return [x if isinstance(x, str) else (x.get("name") or x.get("namespace")) for x in v]
        raise ConnectorError("The API does not list this connection's tables — enter them under “Tables”.")

    def iter_rows(self, stream, *, limit=None):
        s = self._session()
        page, per_page, sent = 1, 500, 0
        while True:
            body = {"connection_id": int(self.cfg("connection_id")), "namespace": stream, "page": page, "per_page": per_page}
            r = None
            for attempt in range(5):
                r = s.post(f"{self._base()}/gateway", json=body, timeout=180,
                           headers={"Authorization": f"Bearer {self._auth(s)}", "Accept": "application/json"})
                if r.status_code == 401:
                    self._token = None
                    continue
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(int(r.headers.get("Retry-After") or 2 ** (attempt + 1)))
                    continue
                break
            if r is None or not r.ok:
                raise ConnectorError(f"{stream} page {page}: HTTP {getattr(r, 'status_code', '?')}")
            payload = r.json()
            data = payload.get("data")
            if isinstance(data, dict):
                data = data.get("data")
            rows = data if isinstance(data, list) else []
            size = int(payload.get("per_page") or per_page)
            if rows:
                yield rows
            sent += len(rows)
            if (limit and sent >= limit) or len(rows) < size:
                return
            page += 1

    def test(self) -> dict:
        s = self._session()
        t0 = time.time()
        self._auth(s)
        r = s.get(f"{self._base()}/connections/{self.cfg('connection_id')}",
                  headers={"Authorization": f"Bearer {self._auth(s)}", "Accept": "application/json"}, timeout=60)
        if not r.ok:
            raise ConnectorError(f"Connection {self.cfg('connection_id')}: HTTP {r.status_code}")
        data = r.json().get("data") or {}
        return {"ok": True, "latency_ms": int((time.time() - t0) * 1000),
                "message": f"Logged in — connection “{data.get('name') or self.cfg('connection_id')}”"}


V3_SERVICES = ("core", "reception", "evaluation", "inventory", "finance", "theatre", "inpatient", "dialysis")
V3_DEFAULT_TEMPLATE = "https://{service}.afyaanalytics.ai/api/"


class AfyaPlatformConnector(StagedConnector):
    """Afya Platform (V3): one gateway per service, `POST {service}/v1/gateway`
    with action list / read; the tenant is the account's organization."""

    dialect = "afya-v3"
    PER_PAGE = 500                                     # the gateway's validation cap
    TOKEN_TTL = 50 * 60

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._token = None
        self._org = None
        self._facility = None
        self._streams: dict[str, tuple[str, str]] | None = None
        self._http = None

    # settings: the form, else AFYA_<KEY>_* from the environment (the V3 DAGs' convention)
    def _env(self, name: str) -> str:
        key = (self.cfg("env_key") or "").strip().upper()
        return (os.getenv(f"AFYA_{key}_{name}") or "").strip() if key else ""

    def _url(self, service: str) -> str:
        tmpl = self.cfg("url_template") or self._env("URL_TEMPLATE") or V3_DEFAULT_TEMPLATE
        return (self._env(f"{service.upper()}_URL") or tmpl.format(service=service)).rstrip("/")

    def _services(self) -> list[str]:
        return _csv_list(self.cfg("services")) or ["core", "reception", "evaluation", "finance", "inventory", "inpatient"]

    def database_label(self) -> str:
        return f"V3 organization {self._org}" if self._org else "Afya Platform (V3)"

    def _session(self):
        if self._http is None:
            import requests
            self._http = requests.Session()
        return self._http

    def _login(self) -> str:
        user = self.cfg("username") or self._env("USERNAME")
        pwd = self.cfg("password") or self._env("PASSWORD")
        if not user or not pwd:
            raise ConnectorError("No V3 credentials — fill in username/password, or a credential key whose "
                                 "AFYA_<KEY>_USERNAME / _PASSWORD are set on the server.")
        url = f"{self._url('core')}/v1/login"
        body = {"username": user, "password": pwd}
        fac = str(self.cfg("facility_id") or self._env("FACILITY_ID") or "").strip()
        r = self._session().post(url, json=body, timeout=30)
        if r.status_code != 200:
            raise ConnectorError(f"V3 login failed: HTTP {r.status_code} {r.text[:160]}")
        res = r.json()
        if res.get("step") == 2:                       # the account has several facilities: pick one
            allowed = res.get("allowed_facility_ids") or [f.get("id") for f in (res.get("facilities") or []) if f.get("id")]
            if fac and int(fac) not in allowed:
                raise ConnectorError(f"This account can't log into facility {fac} (allowed: {allowed}).")
            body["facility_id"] = int(fac) if fac else allowed[0]
            r = self._session().post(url, json=body, timeout=30)
            if r.status_code != 200:
                raise ConnectorError(f"V3 login (facility step) failed: HTTP {r.status_code} {r.text[:160]}")
            res = r.json()
        token = res.get("access_token")
        if not token:
            raise ConnectorError("V3 login returned no access_token.")
        u = res.get("user") or res.get("data") or {}
        self._org = (res.get("tenant") or {}).get("id") or u.get("organization_id")
        self._facility = (res.get("facility") or {}).get("id") or u.get("facility_id") or body.get("facility_id")
        return token

    def _bearer(self, force=False) -> str:
        if force or not self._token or time.time() - self._token[1] > self.TOKEN_TTL:
            self._token = (self._login(), time.time())
        return self._token[0]

    def _post(self, service: str, body: dict, *, timeout=120):
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        mkey = self.cfg("migration_key") or self._env("MIGRATION_KEY") or os.getenv("MODEL_GATEWAY_MIGRATION_KEY", "")
        if service in ("theatre", "dialysis") and mkey:
            headers["X-Migration-Key"] = mkey
        r, wait = None, 2
        for attempt in range(6):
            headers["Authorization"] = f"Bearer {self._bearer(force=attempt > 0 and r is not None and r.status_code == 401)}"
            try:
                r = self._session().post(f"{self._url(service)}/v1/gateway", json=body, headers=headers, timeout=timeout)
            except Exception as e:  # network blip
                if attempt == 5:
                    raise ConnectorError(f"{service}: {type(e).__name__}: {e}") from e
                time.sleep(wait); wait = min(wait * 2, 60)
                continue
            if r.status_code == 401:
                continue
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(int(r.headers.get("Retry-After") or wait)); wait = min(wait * 2, 60)
                continue
            return r
        return r

    def _catalog(self) -> dict[str, tuple[str, str]]:
        """stream name → (service, alias). Explicit `service:alias` list, else every readable model."""
        if self._streams is not None:
            return self._streams
        pairs: list[tuple[str, str]] = []
        explicit = _csv_list(self.cfg("models"))
        if explicit:
            for item in explicit:
                service, sep, alias = item.partition(":")
                if not sep:
                    raise ConnectorError(f"“{item}”: write models as service:alias, e.g. reception:patient")
                pairs.append((service.strip().lower(), alias.strip()))
        else:
            for service in self._services():
                r = self._post(service, {"action": "list"}, timeout=60)
                if r is None or not r.ok:
                    continue                           # service not deployed / not allowed for this account
                for e in (r.json().get("data") or []):
                    if e.get("alias") and "read" in (e.get("operations") or ["read"]):
                        pairs.append((service, e["alias"]))
            if not pairs:
                raise ConnectorError("No readable models found — check the services, or list models as service:alias.")
        aliases = [a for _, a in pairs]
        self._streams = {(a if aliases.count(a) == 1 else f"{s}__{a}"): (s, a) for s, a in pairs}
        return self._streams

    def streams(self) -> list[str]:
        return list(self._catalog())

    def iter_rows(self, stream, *, limit=None):
        service, alias = self._catalog().get(stream) or (None, None)
        if not service:
            raise ConnectorError(f"Unknown V3 stream {stream}")
        per_page = min(self.PER_PAGE, limit or self.PER_PAGE)
        page, sent, seen = 1, 0, set()
        while True:
            self._bearer()                             # the tenant id comes from the login
            r = self._post(service, {"action": "read", "model": alias, "source_tenant_id": self._org,
                                     "per_page": per_page, "page": page})
            if r is None or not r.ok:
                raise ConnectorError(f"{service}/{alias} page {page}: HTTP {getattr(r, 'status_code', '?')} "
                                     f"{(getattr(r, 'text', '') or '')[:160]}")
            j = r.json()
            rows = j.get("data") or []
            if isinstance(rows, dict):
                rows = rows.get("data") or []
            fresh = [x for x in rows if x.get("id") is None or x.get("id") not in seen]
            seen.update(x.get("id") for x in fresh if x.get("id") is not None)
            if fresh:
                yield fresh
            sent += len(fresh)
            last = (j.get("meta") or {}).get("last_page")
            if not rows or len(rows) < per_page or (limit and sent >= limit) or (last and page >= last):
                return
            page += 1

    # ── structure: the gateway's `describe` is the source of truth ──────────
    def describe(self, stream: str) -> dict:
        service, alias = self._catalog()[stream]
        r = self._post(service, {"action": "describe", "model": alias}, timeout=60)
        if r is None or not r.ok:
            raise ConnectorError(f"{service}/{alias} describe: HTTP {getattr(r, 'status_code', '?')}")
        return r.json().get("data") or {}

    @staticmethod
    def _v3_columns(desc: dict) -> list[dict]:
        cols = []
        pk = set(desc.get("primary_key") or [])
        for c in desc.get("columns") or []:
            t = c.get("type") or c.get("type_name") or ""
            enum = re.findall(r"'((?:[^'\\]|\\.)*)'", t) if re.match(r"(enum|set)\(", t, re.I) else []
            note = [x for x in (c.get("comment"), c.get("set_by") and f"set by {c['set_by']}",
                                c.get("required_on_insert") and "required on insert") if x]
            cols.append({"name": c["name"], "type": t, "nullable": bool(c.get("nullable", True)),
                         "pk": bool(c.get("primary")) or c["name"] in pk, "default": c.get("default"),
                         "comment": " · ".join(str(x) for x in note) or None, "allowed_values": enum[:200],
                         "_ref": c.get("references")})
        return cols

    def discover(self, options=None, progress: Progress = _noop) -> dict:
        """Tables, columns, types, keys and foreign keys from `describe` (complete even for empty
        models); a row sample adds profiles when profiling is on and “Sample rows” is enabled."""
        o = _options(options)
        wanted = set(o.get("tables") or [])
        streams = [s for s in self.streams() if not wanted or s in wanted][: o["max_tables"]]
        if not streams:
            raise ConnectorError("No models to extract.")
        # describe first: a model this gateway doesn't have is skipped, not fatal
        described, missing = {}, []
        for i, s in enumerate(streams, start=1):
            service, alias = self._catalog()[s]
            progress(pct=2 + 28 * i / len(streams), msg=f"Describing {service}:{alias}", phase="describe")
            try:
                described[s] = self.describe(s)
            except ConnectorError as e:
                missing.append(f"{service}:{alias}")
                progress(msg=f"{e} — skipped", level="warning")
        if not described:
            raise ConnectorError(f"None of the models exist on this gateway ({', '.join(missing[:8])}).")
        sampled = {}
        if o.get("profile", True) and self.cfg("sample_data", True) not in (False, "false", "0", 0):
            try:
                base = StagedConnector.discover(self, {**o, "tables": list(described)},
                                                progress=lambda pct=None, **kw: progress(pct=None if pct is None else 30 + pct * .65, **kw))
                sampled = {t["key"].lower(): t for t in base["tables"]}
            except ConnectorError as e:
                progress(msg=f"Row samples unavailable ({e}) — structure only.", level="warning")
        snap = snap_mod.empty_snapshot(dialect="mysql", database=self.database_label(),
                                       extractor="Afya V3 gateway describe" + (" + dbatlas profiles" if sampled else ""))
        if missing:
            snap["warnings"].append(f"Not on this gateway (skipped): {', '.join(missing)}")
        rels = []
        for s, desc in described.items():
            service, alias = self._catalog()[s]
            cols = self._v3_columns(desc)
            prof = {c["name"]: c for c in (sampled.get(s.lower()) or {}).get("columns", [])}
            known = {c["name"] for c in cols}
            for c in cols:
                if c["name"] in prof:
                    c["profile"] = prof[c["name"]].get("profile")
                if c["_ref"] and "." in str(c["_ref"]):
                    tt, tc = str(c["_ref"]).rsplit(".", 1)
                    rels.append({"from": s, "from_columns": [c["name"]], "to": tt, "to_columns": [tc],
                                 "confidence": 1.0, "kind": "foreign_key"})
                c.pop("_ref", None)
            for fk in desc.get("foreign_keys") or []:
                fc = fk.get("columns") or ([fk["column"]] if fk.get("column") else [])
                if fc and (fk.get("foreign_table") or fk.get("references_table")):
                    rels.append({"from": s, "from_columns": fc, "to": fk.get("foreign_table") or fk.get("references_table"),
                                 "to_columns": fk.get("foreign_columns") or fk.get("references_columns") or [], "confidence": 1.0,
                                 "kind": "foreign_key"})
            # accessor fields the API adds to every row (e.g. patient_full_name)
            for name, pc in prof.items():
                if name not in known:
                    cols.append({**pc, "comment": "computed by the API (not a table column)", "pk": False})
            count = desc.get("count")
            snap["tables"].append({"key": s, "name": desc.get("table") or alias, "schema": service, "kind": "table",
                                   "model": desc.get("class"), "service": service, "columns": cols,
                                   "rows": count if isinstance(count, int) else (sampled.get(s.lower()) or {}).get("rows"),
                                   "comment": f"{service}:{alias}", "primary_key": desc.get("primary_key") or []})
        if not snap["tables"]:
            raise ConnectorError("No model could be described.")
        # FKs name real tables (patients.id): map to keys, drop ones outside this extraction
        # same table name on several services (evaluation.visits / reception.visits): prefer the FK's own service
        by_name: dict[str, str] = {}
        by_svc = {(t["service"], t["name"].lower()): t["key"] for t in snap["tables"]}
        svc_of = {t["key"]: t["service"] for t in snap["tables"]}
        for t in snap["tables"]:
            by_name.setdefault(t["name"].lower(), t["key"])
        seen = set()
        for r in rels:
            name = str(r["to"]).lower()
            dst = by_svc.get((svc_of.get(r["from"]), name)) or by_name.get(name) \
                or (r["to"] if r["to"] in svc_of else None)
            k = (r["from"], tuple(r["from_columns"]))
            if dst and k not in seen:
                seen.add(k)
                snap["relationships"].append({**r, "to": dst})
        out = snap_mod.finalize(snap)
        out["stats"]["sample_rows"] = o["sample_rows"] if sampled else 0
        return out

    def test(self) -> dict:
        t0 = time.time()
        self._bearer()
        r = self._post("core", {"action": "list"}, timeout=60)
        n = len([e for e in (r.json().get("data") or []) if e.get("alias")]) if r is not None and r.ok else None
        return {"ok": True, "latency_ms": int((time.time() - t0) * 1000),
                "message": f"Logged in — organization {self._org}, facility {self._facility}"
                           + (f" · {n} core models" if n is not None else "")}


class SchemaFileConnector(Connector):
    """A schema with no data behind it: schema.json, SQL DDL, dbatlas.json or mappings.json."""

    can_read = False

    def _doc(self) -> tuple[bytes, str]:
        if self.files:
            return self.files[0].read_bytes(), self.files[0].name
        if self.cfg("schema_text"):
            text = self.cfg("schema_text")
            return text.encode(), "schema.sql" if re.search(r"create\s+table", text, re.I) and not text.strip().startswith(("{", "[")) else "schema.json"
        raise ConnectorError("Upload a schema file or paste one.")

    def test(self) -> dict:
        snap = self.discover()
        s = snap["stats"]
        return {"ok": True, "message": f"Parsed {s['tables']} tables · {s['columns']} columns · {s['relationships']} relationships"}

    def discover(self, options=None, progress: Progress = _noop) -> dict:
        content, name = self._doc()
        progress(msg=f"Parsing {name}…", phase="parse", pct=20)
        try:
            snap = snap_mod.parse_schema_document(content, name, dialect=self.cfg("dialect") or None)
        except snap_mod.SchemaDocumentError as e:
            raise ConnectorError(str(e)) from None
        progress(msg=f"Parsed {snap['stats']['tables']} tables; inferred relationships with dbatlas.", pct=90)
        if (options or {}).get("tables"):
            snap = snap_mod.subset(snap, options["tables"])
        return snap


# ─── registry ─────────────────────────────────────────────────────────────

_SSL = FieldSpec("sslmode", "SSL mode", type="select", group="Security", advanced=True, default="prefer",
                 options=[{"value": v, "label": v} for v in ("disable", "allow", "prefer", "require", "verify-ca", "verify-full")])

register(ConnectorSpec(
    "postgres", "PostgreSQL", "Databases", "database", "#336791",
    "Tables, views, keys and foreign keys from any PostgreSQL database.",
    HOST_FIELDS(5432) + [SCHEMA_FIELD("public"), TABLES_FIELD, _SSL],
    driver="psycopg2", pip="psycopg2-binary", tagline="Postgres 9.6+ · Supabase · RDS · Cloud SQL"), SQLConnector)
register(ConnectorSpec(
    "mysql", "MySQL", "Databases", "database", "#00758F",
    "Any MySQL 5.7+/8 server — the classic HMIS backend.",
    HOST_FIELDS(3306) + [TABLES_FIELD], driver="pymysql", pip="pymysql", tagline="MySQL · Aurora · PlanetScale"), SQLConnector)
register(ConnectorSpec(
    "mariadb", "MariaDB", "Databases", "database-fill", "#C0765A",
    "MariaDB over the MySQL protocol.", HOST_FIELDS(3306) + [TABLES_FIELD], driver="pymysql", pip="pymysql"), SQLConnector)
register(ConnectorSpec(
    "mssql", "SQL Server", "Databases", "server", "#CC2927",
    "Microsoft SQL Server / Azure SQL over ODBC.",
    HOST_FIELDS(1433) + [SCHEMA_FIELD("dbo"), TABLES_FIELD,
                         FieldSpec("odbc_driver", "ODBC driver", default="ODBC Driver 18 for SQL Server", advanced=True, group="Security"),
                         FieldSpec("trust_cert", "Trust server certificate", type="checkbox", default=True, advanced=True, group="Security")],
    driver="pyodbc", pip="pyodbc (+ msodbcsql18)"), SQLConnector)
register(ConnectorSpec(
    "oracle", "Oracle", "Databases", "hdd-stack", "#F80000",
    "Oracle Database via python-oracledb (thin mode — no client install).",
    HOST_FIELDS(1521, "Service name") + [SCHEMA_FIELD(), TABLES_FIELD], driver="oracledb", pip="oracledb"), SQLConnector)
register(ConnectorSpec(
    "redshift", "Amazon Redshift", "Warehouses", "bucket", "#8C4FFF",
    "Redshift clusters over the Postgres protocol.", HOST_FIELDS(5439) + [SCHEMA_FIELD("public"), TABLES_FIELD, _SSL],
    driver="psycopg2", pip="psycopg2-binary"), SQLConnector)
register(ConnectorSpec(
    "snowflake", "Snowflake", "Warehouses", "snow2", "#29B5E8",
    "Snowflake databases and schemas — password or key-pair auth.",
    [FieldSpec("account", "Account", required=True, placeholder="xy12345.eu-west-1", help="Account identifier (org-account or locator.region)."),
     FieldSpec("username", "User", required=True),
     FieldSpec("password", "Password", type="password", secret=True, help="Or use a private key below."),
     FieldSpec("private_key", "Private key (PEM)", type="textarea", secret=True, rows=4, advanced=True, group="Security"),
     FieldSpec("private_key_passphrase", "Key passphrase", type="password", secret=True, advanced=True, group="Security"),
     FieldSpec("warehouse", "Warehouse", placeholder="COMPUTE_WH"),
     FieldSpec("role", "Role"),
     FieldSpec("database", "Database", required=True),
     SCHEMA_FIELD("PUBLIC", "Schemas to read / write (first one is the default)."), TABLES_FIELD],
    driver="snowflake.sqlalchemy", pip="snowflake-sqlalchemy"), SQLConnector)
register(ConnectorSpec(
    "sqlite", "SQLite", "Databases", "file-earmark-binary", "#0F80CC",
    "Upload a .sqlite / .db file (or point at one on the server).",
    [FieldSpec("path", "Path on server", placeholder="/data/app.sqlite", advanced=True, help="Instead of uploading.")],
    uploads=(".sqlite", ".sqlite3", ".db"), tagline="Upload a file"), SQLConnector)
register(ConnectorSpec(
    "duckdb", "DuckDB", "Databases", "lightning-charge", "#E8B100",
    "A DuckDB database file — also a fast local sync destination.",
    [FieldSpec("path", "Path on server", placeholder="/data/warehouse.duckdb", help="Upload a file or give a server path (created if missing when used as a destination).")],
    uploads=(".duckdb", ".db"), driver="duckdb_engine", pip="duckdb-engine"), SQLConnector)
register(ConnectorSpec(
    "sqlalchemy", "Any SQL database", "Databases", "plug", "#64748B",
    "Anything with a SQLAlchemy dialect installed — BigQuery, Databricks, Trino, ClickHouse, Db2…",
    [FieldSpec("url", "SQLAlchemy URL", required=True, secret=True, placeholder="dialect+driver://user:pass@host:port/db"),
     SCHEMA_FIELD(), TABLES_FIELD], tagline="Bring your own dialect"), SQLConnector)

register(ConnectorSpec(
    "files", "Files", "Files", "file-earmark-spreadsheet", "#10B981",
    "CSV, TSV, Excel (every sheet), JSON, NDJSON or Parquet — each file or sheet becomes a table.",
    [FieldSpec("delimiter", "CSV delimiter", placeholder="auto", advanced=True, group="Parsing"),
     FieldSpec("encoding", "Encoding", default="utf-8", advanced=True, group="Parsing"),
     FieldSpec("records_path", "JSON records path", placeholder="data.items", advanced=True, group="Parsing",
               help="Where the list of records sits inside a .json file. Blank = auto-detect.")],
    roles=("source", "target"), uploads=(".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".xls", ".json", ".ndjson", ".jsonl", ".parquet"),
    multi_upload=True, tagline="Drag & drop"), FilesConnector)

register(ConnectorSpec(
    "rest_api", "REST API", "APIs", "globe2", "#6366F1",
    "Any JSON API, configured declaratively: auth, streams, records path and pagination.",
    [FieldSpec("base_url", "Base URL", required=True, placeholder="https://api.example.com/v1"),
     FieldSpec("auth_type", "Authentication", type="select", default="none", group="Authentication",
               options=[{"value": "none", "label": "None"}, {"value": "bearer", "label": "Bearer token"},
                        {"value": "basic", "label": "Basic (user / password)"},
                        {"value": "api_key_header", "label": "API key in header"},
                        {"value": "api_key_query", "label": "API key in query string"},
                        {"value": "login", "label": "Login request → bearer token"}]),
     FieldSpec("login_path", "Login path", placeholder="/auth/login", group="Authentication",
               show_if={"field": "auth_type", "in": ["login"]}),
     FieldSpec("login_body", "Login body (JSON)", type="code", rows=3, secret=True, group="Authentication",
               placeholder='{"username": "…", "password": "…"}', show_if={"field": "auth_type", "in": ["login"]}),
     FieldSpec("token_path", "Token path", placeholder="access_token", group="Authentication",
               help="Where the token is in the login response. Blank = auto-detect.",
               show_if={"field": "auth_type", "in": ["login"]}),
     FieldSpec("token", "Token / API key", type="password", secret=True, group="Authentication",
               show_if={"field": "auth_type", "in": ["bearer", "api_key_header", "api_key_query"]}),
     FieldSpec("api_key_name", "Key name", placeholder="X-API-Key", group="Authentication",
               show_if={"field": "auth_type", "in": ["api_key_header", "api_key_query"]}),
     FieldSpec("username", "Username", group="Authentication", show_if={"field": "auth_type", "in": ["basic"]}),
     FieldSpec("password", "Password", type="password", secret=True, group="Authentication", show_if={"field": "auth_type", "in": ["basic"]}),
     FieldSpec("headers", "Extra headers", type="textarea", rows=2, secret=True, group="Authentication", advanced=True,
               placeholder="X-Tenant: 42"),
     FieldSpec("streams", "Streams", type="code", required=True, rows=5, group="Streams",
               placeholder="patients = /patients -> data\nvisits = /visits?status=closed -> data.items",
               help="One per line: name = [POST ]path[ {json body}] -> records.path (records path optional). "
                    "Or import a Postman collection / OpenAPI file above."),
     FieldSpec("records_path", "Default records path", placeholder="data", group="Streams", advanced=True),
     FieldSpec("pagination", "Pagination", type="select", default="none", group="Pagination",
               options=[{"value": "none", "label": "None — one request"}, {"value": "page", "label": "Page number"},
                        {"value": "offset", "label": "Offset / limit"}, {"value": "cursor", "label": "Cursor in body"},
                        {"value": "link", "label": "Next link (header or body)"}]),
     FieldSpec("page_param", "Page / offset / cursor param", placeholder="page", group="Pagination",
               show_if={"field": "pagination", "in": ["page", "offset", "cursor"]}),
     FieldSpec("size_param", "Page size param", placeholder="per_page", group="Pagination",
               show_if={"field": "pagination", "in": ["page", "offset"]}),
     FieldSpec("page_size", "Page size", type="number", default=100, group="Pagination",
               show_if={"field": "pagination", "in": ["page", "offset"]}),
     FieldSpec("cursor_path", "Cursor / next path in body", placeholder="meta.next_cursor", group="Pagination",
               show_if={"field": "pagination", "in": ["cursor", "link"]}),
     FieldSpec("max_pages", "Max pages", type="number", default=1000, group="Pagination", advanced=True)],
    roles=("source", "target"), pip="requests", tagline="Low-code connector builder"), RestConnector)

register(ConnectorSpec(
    "afya_extraction", "Afya Extraction API", "APIs", "hospital", "#0BB99F",
    "Facility data through the Afya Extraction gateway (one namespace = one table).",
    [FieldSpec("connection_id", "Connection id", type="number", required=True, placeholder="59"),
     FieldSpec("tables", "Tables (namespaces)", type="textarea", rows=2, placeholder="patients, visits, invoices",
               help="Comma-separated. Blank = whatever the API lists for the connection."),
     FieldSpec("base_url", "Base URL", placeholder="https://afyapi.afyaanalytics.ai/api", advanced=True, group="Authentication"),
     FieldSpec("username", "Username", advanced=True, group="Authentication", help="Blank = the platform's AFYA_EXTRACTION_USERNAME."),
     FieldSpec("password", "Password", type="password", secret=True, advanced=True, group="Authentication")],
    roles=("source",)), AfyaExtractionConnector)

register(ConnectorSpec(
    "afya_platform", "Afya Platform (V3)", "APIs", "hospital", "#0E7C66",
    "Read an organization's data from the V3 model gateways (core, reception, evaluation, finance …).",
    [FieldSpec("username", "Username", group="Authentication", help="The V3 account — its organization is the tenant you read."),
     FieldSpec("password", "Password", type="password", secret=True, group="Authentication"),
     FieldSpec("facility_id", "Facility id", type="number", group="Authentication",
               help="Only for accounts on several facilities. Blank = the first one."),
     FieldSpec("env_key", "…or credential key", placeholder="WEMA_PROD", group="Authentication", advanced=True,
               help="Uses AFYA_<KEY>_USERNAME / _PASSWORD / _URL_TEMPLATE from the server's environment."),
     FieldSpec("url_template", "URL template", placeholder=V3_DEFAULT_TEMPLATE, group="Environment",
               help="{service} is replaced by core, reception, … Migrate: https://{service}migrate.afyaanalytics.ai/api/"),
     FieldSpec("services", "Services", placeholder="core, reception, evaluation, finance, inventory, inpatient",
               group="Models", help="Where to look for models when none are listed below."),
     FieldSpec("models", "Models", type="textarea", rows=3, group="Models",
               placeholder="reception:patient, evaluation:visits, evaluation:vitals, finance:invoice",
               help="service:alias, comma-separated. Blank = every readable model on the services above."),
     FieldSpec("sample_data", "Sample rows for profiles", type="checkbox", default=True, group="Models",
               help="Off = structure only (describe): no patient data is read — enough for a target schema."),
     FieldSpec("migration_key", "X-Migration-Key", type="password", secret=True, group="Environment", advanced=True,
               help="Only theatre / dialysis need it.")],
    roles=("source", "target"), pip="requests", tagline="V3 model gateway"), AfyaPlatformConnector)

register(ConnectorSpec(
    "schema_file", "Schema file", "Schema", "filetype-json", "#F59E0B",
    "Upload a target schema: schema.json, SQL DDL, JSON Schema, dbatlas.json — or reuse a mappings.json's target side.",
    [FieldSpec("dialect", "SQL dialect (for DDL)", type="select", default="", advanced=True, group="Parsing",
               options=[{"value": "", "label": "Auto-detect"}] + [{"value": d, "label": d} for d in
                        ("mysql", "postgres", "tsql", "snowflake", "sqlite", "oracle", "bigquery", "duckdb")]),
     FieldSpec("schema_text", "…or paste it", type="code", rows=6, advanced=True, group="Parsing")],
    roles=("target", "source"), uploads=(".json", ".sql", ".ddl", ".txt"), tagline="Upload schema.json / DDL"), SchemaFileConnector)
