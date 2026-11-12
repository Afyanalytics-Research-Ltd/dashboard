"""A Databend connection that behaves like a ``snowflake.connector`` one.

``connect()`` (see __init__.py) hands this out instead of a Snowflake
connection when the warehouse switch is on Databend.  Callers keep their
code unchanged: cursors, ``%s`` / ``%(name)s`` / ``?`` / ``:1`` parameters,
``fetchone/fetchmany/fetchall/fetch_pandas_all``, ``description``,
``rowcount``, ``sfqid``, ``execute_async`` + ``get_results_from_sfqid``,
transactions, ``DictCursor`` and Snowflake-style errors (``errno`` 2003 for
a missing object, so existing ``except ProgrammingError`` paths still fire).

Every statement goes through translate.py first, so callers keep writing
Snowflake SQL.

Transport is Databend's HTTP handler (POST /v1/query, then page through
``next_uri``), with session cookies enabled so temporary tables and
explicit transactions work across statements - one requests.Session per
connection.
"""

from __future__ import annotations

import datetime as _dt
import decimal
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from collections import namedtuple
from typing import Any, Iterable, Optional, Sequence

import requests

from . import sql_translate as tr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Errors - subclass the Snowflake connector's so existing except clauses work
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    from snowflake.connector import errors as _sf_errors
    _BaseError = _sf_errors.Error
    _BaseProgrammingError = _sf_errors.ProgrammingError
    _BaseDatabaseError = _sf_errors.DatabaseError
    _BaseOperationalError = _sf_errors.OperationalError
    _BaseInterfaceError = _sf_errors.InterfaceError
    _BaseNotSupportedError = _sf_errors.NotSupportedError
except Exception:  # snowflake-connector not installed
    class _BaseError(Exception):
        def __init__(self, msg=None, errno=None, sqlstate=None, sfqid=None, **_):
            super().__init__(msg)
            self.msg, self.errno, self.sqlstate, self.sfqid = msg, errno, sqlstate, sfqid
    _BaseProgrammingError = _BaseDatabaseError = _BaseOperationalError = _BaseError
    _BaseInterfaceError = _BaseNotSupportedError = _BaseError


class DatabendError(_BaseDatabaseError):
    pass


class DatabendProgrammingError(_BaseProgrammingError):
    pass


class DatabendOperationalError(_BaseOperationalError):
    pass


class DatabendInterfaceError(_BaseInterfaceError):
    pass


class DatabendNotSupportedError(_BaseNotSupportedError):
    pass


def _make_error(cls, msg, errno, sqlstate, qid=None):
    try:
        return cls(msg=msg, errno=errno, sqlstate=sqlstate, sfqid=qid)
    except TypeError:
        e = cls(msg)
        e.errno, e.sqlstate, e.sfqid = errno, sqlstate, qid
        return e


# Databend error code -> (Snowflake errno, sqlstate, exception class)
_UNKNOWN_OBJECT = re.compile(r"(Unknown (table|database|stage|view)|"
                             r"(table|database|view|stage)\s+[`'\"]?[\w.\"`]+[`'\"]?\s+does not exist)", re.I)
_UNKNOWN_COLUMN = re.compile(r"column\s+\S+\s+doesn't exist|unknown column|invalid identifier", re.I)


def _translate_error(err: dict, sql: str, qid: Optional[str]):
    code = err.get("code")
    msg = (err.get("message") or "").strip()
    head = f"[Databend {code}] "
    if code in (1025, 1003, 2201, 1029) or _UNKNOWN_OBJECT.search(msg):
        hint = ""
        if re.search(r'"(STAGING|REPORTING|ANALYTICS|SHARED)"', msg) or re.search(
                r"\b(STAGING|REPORTING|ANALYTICS)\b", sql or ""):
            hint = (" [Databend warehouse: this schema is built inside Snowflake and has not been synced "
                    "to Databend yet - run `python manage.py sync_snowflake_to_databend --schemas STAGING,"
                    "REPORTING` once Snowflake is reachable.]")
        return _make_error(DatabendProgrammingError,
                           head + "SQL compilation error: Object does not exist or not authorized. " + msg + hint,
                           2003, "42S02", qid)
    if _UNKNOWN_COLUMN.search(msg):
        return _make_error(DatabendProgrammingError,
                           head + "SQL compilation error: invalid identifier. " + msg, 904, "42000", qid)
    if code == 1005 or "SyntaxException" in msg:
        return _make_error(DatabendProgrammingError, head + "SQL compilation error: syntax error. " + msg,
                           1003, "42000", qid)
    if code in (2002, 2205) or "already exists" in msg:
        return _make_error(DatabendProgrammingError, head + "SQL compilation error: Object already exists. " + msg,
                           2002, "42710", qid)
    if code in (1006, 1065, 1046):
        return _make_error(DatabendProgrammingError, head + msg, 100038 if code == 1006 else 2000 + int(code),
                           "22018" if code == 1006 else "42000", qid)
    return _make_error(DatabendProgrammingError, head + msg, 100000 + int(code or 0), "XX000", qid)


# ---------------------------------------------------------------------------
# Snowflake-compatible metadata
# ---------------------------------------------------------------------------
ResultMetadata = namedtuple(
    "ResultMetadata",
    ["name", "type_code", "display_size", "internal_size", "precision", "scale", "is_nullable"],
)

# snowflake.connector.constants.FIELD_TYPES ids
TYPE_FIXED, TYPE_REAL, TYPE_TEXT, TYPE_DATE, TYPE_TIMESTAMP, TYPE_VARIANT = 0, 1, 2, 3, 4, 5
TYPE_TIMESTAMP_NTZ, TYPE_OBJECT, TYPE_ARRAY, TYPE_BINARY, TYPE_TIME, TYPE_BOOLEAN = 8, 9, 10, 11, 12, 13

_NULLABLE_RE = re.compile(r"^Nullable\((.*)\)$")
_DECIMAL_RE = re.compile(r"^Decimal\((\d+),\s*(\d+)\)$")


class _ColType:
    __slots__ = ("raw", "base", "nullable", "precision", "scale", "kind", "type_code")

    def __init__(self, raw: str):
        self.raw = raw
        t = raw.strip()
        self.nullable = False
        m = _NULLABLE_RE.match(t)
        if m:
            t, self.nullable = m.group(1), True
        if t.endswith(" NULL"):
            t, self.nullable = t[:-5], True
        self.base = t
        self.precision = self.scale = None
        if t.startswith(("Int", "UInt")):
            self.kind, self.type_code, self.precision, self.scale = "int", TYPE_FIXED, 38, 0
        elif t.startswith("Decimal"):
            dm = _DECIMAL_RE.match(t)
            self.precision, self.scale = (int(dm.group(1)), int(dm.group(2))) if dm else (38, 0)
            self.kind = "int" if self.scale == 0 else "decimal"
            self.type_code = TYPE_FIXED
        elif t.startswith("Float"):
            self.kind, self.type_code = "float", TYPE_REAL
        elif t == "Boolean":
            self.kind, self.type_code = "bool", TYPE_BOOLEAN
        elif t == "Date":
            self.kind, self.type_code = "date", TYPE_DATE
        elif t.startswith("Timestamp"):
            self.kind, self.type_code = "timestamp", TYPE_TIMESTAMP_NTZ
        elif t in ("Variant", "Bitmap", "Geometry", "Geography") or t.startswith(("Tuple", "Map")):
            self.kind, self.type_code = "json", TYPE_OBJECT if t.startswith(("Tuple", "Map")) else TYPE_VARIANT
        elif t.startswith("Array"):
            self.kind, self.type_code = "json", TYPE_ARRAY
        elif t == "Binary":
            self.kind, self.type_code = "binary", TYPE_BINARY
        elif t == "Null":
            self.kind, self.type_code = "null", TYPE_TEXT
        else:
            self.kind, self.type_code = "str", TYPE_TEXT

    def convert(self, v):
        if v is None:
            return None
        k = self.kind
        try:
            if k == "str" or k == "json":
                return v
            if k == "int":
                return int(v)
            if k == "decimal":
                return decimal.Decimal(v)
            if k == "float":
                return float(v)
            if k == "bool":
                return v in ("1", "true", "TRUE", "True")
            if k == "timestamp":
                return _parse_ts(v)
            if k == "date":
                return _dt.date.fromisoformat(v[:10])
            if k == "binary":
                return bytes.fromhex(v)
        except (ValueError, decimal.InvalidOperation):
            return v
        return v


def _parse_ts(v: str):
    s = v.strip()
    try:
        return _dt.datetime.fromisoformat(s)
    except ValueError:
        # "2026-01-01 10:00:00.000000 +0300" style (TIMESTAMP_TZ)
        try:
            return _dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f %z")
        except ValueError:
            return s


# ---------------------------------------------------------------------------
# Parameter binding
# ---------------------------------------------------------------------------
_PYFORMAT_NAMED = re.compile(r"%\((\w+)\)s")


def _quote_aware_scan(sql: str, on_char):
    """Call on_char(i) for each char outside quotes/comments; returns list of
    (index) matches the callback reports."""
    i, n = 0, len(sql)
    quote = None
    hits = []
    while i < n:
        c = sql[i]
        if quote:
            if c == "\\" and quote == "'":
                i += 2
                continue
            if c == quote:
                if i + 1 < n and sql[i + 1] == quote:
                    i += 2
                    continue
                quote = None
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
        elif c == "-" and sql.startswith("--", i):
            j = sql.find("\n", i)
            i = n if j < 0 else j
            continue
        elif c == "/" and sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        elif c == "$" and sql.startswith("$$", i):
            j = sql.find("$$", i + 2)
            i = n if j < 0 else j + 2
            continue
        else:
            r = on_char(i)
            if r:
                hits.append(r)
        i += 1
    return hits


def to_literal(v: Any) -> str:
    """Render a Python value as a Databend SQL literal."""
    if v is None:
        return "NULL"
    try:
        import numpy as np  # noqa
        if isinstance(v, np.generic):
            v = v.item()
    except ImportError:
        pass
    try:
        import pandas as pd  # noqa
        if v is pd.NaT:
            return "NULL"
        if isinstance(v, pd.Timestamp):
            v = v.to_pydatetime()
    except ImportError:
        pass
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "NULL"
        if math.isinf(v):
            return "'Infinity'::DOUBLE" if v > 0 else "'-Infinity'::DOUBLE"
        return repr(v)
    if isinstance(v, decimal.Decimal):
        if v.is_nan():
            return "NULL"
        return format(v, "f")
    if isinstance(v, _dt.datetime):
        if v.tzinfo is not None:
            v = v.astimezone(_dt.timezone.utc).replace(tzinfo=None)
        return "'" + v.isoformat(sep=" ") + "'"
    if isinstance(v, _dt.date):
        return "'" + v.isoformat() + "'"
    if isinstance(v, _dt.time):
        return "'" + v.isoformat() + "'"
    if isinstance(v, (bytes, bytearray, memoryview)):
        return "FROM_HEX('" + bytes(v).hex() + "')"
    if isinstance(v, (dict, list, tuple)):
        v = json.dumps(v, default=str)
    if isinstance(v, uuid.UUID):
        v = str(v)
    s = str(v)
    return "'" + s.replace("\\", "\\\\").replace("'", "''") + "'"


def bind(sql: str, params) -> str:
    """Bind ``params`` into ``sql`` the way snowflake.connector would.

    Supports pyformat (``%s``, ``%(name)s`` - the connector's default),
    qmark (``?``) and numeric (``:1``) styles.  With ``params is None`` the
    SQL is returned untouched (``%`` stays literal), like the connector.
    """
    if params is None:
        return sql
    if isinstance(params, dict):
        if not _PYFORMAT_NAMED.search(sql):
            raise DatabendProgrammingError("named parameters given but the SQL has no %(name)s placeholders")
        return _PYFORMAT_NAMED.sub(lambda m: to_literal(params[m.group(1)]), sql.replace("%%", "\x00")) \
            .replace("\x00", "%")
    params = list(params)
    if "%s" in sql:
        parts = sql.replace("%%", "\x00").split("%s")
        if len(parts) - 1 != len(params):
            raise DatabendProgrammingError(
                f"Not enough arguments for format string: {len(parts) - 1} placeholders, {len(params)} params")
        out = [parts[0]]
        for p, rest in zip(params, parts[1:]):
            out.append(to_literal(p))
            out.append(rest)
        return "".join(out).replace("\x00", "%")
    qpos = _quote_aware_scan(sql, lambda i: i if sql[i] == "?" else None)
    if qpos:
        if len(qpos) != len(params):
            raise DatabendProgrammingError(f"{len(qpos)} ? placeholders but {len(params)} parameters")
        out, last = [], 0
        for i, p in zip(qpos, params):
            out.append(sql[last:i])
            out.append(to_literal(p))
            last = i + 1
        out.append(sql[last:])
        return "".join(out)
    npos = _quote_aware_scan(sql, lambda i: i if sql[i] == ":" and i + 1 < len(sql) and sql[i + 1].isdigit()
                             and (i == 0 or sql[i - 1] != ":") else None)
    if npos:
        out, last = [], 0
        for i in npos:
            j = i + 1
            while j < len(sql) and sql[j].isdigit():
                j += 1
            out.append(sql[last:i])
            out.append(to_literal(params[int(sql[i + 1:j]) - 1]))
            last = j
        out.append(sql[last:])
        return "".join(out)
    if params:
        raise DatabendProgrammingError("parameters given but the SQL has no placeholders")
    return sql


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------

def _settings_url() -> str:
    url = os.getenv("DATABEND_HTTP_URL", "").strip()
    if not url:
        try:
            from django.conf import settings as dj
            url = getattr(dj, "DATABEND_HTTP_URL", "") or ""
        except Exception:
            url = ""
    return (url or "http://databend:8000").rstrip("/")


def _settings_auth():
    user = os.getenv("DATABEND_USER", "").strip()
    pw = os.getenv("DATABEND_PASSWORD", "").strip()
    if not user:
        try:
            from django.conf import settings as dj
            user = getattr(dj, "DATABEND_USER", "") or ""
            pw = pw or getattr(dj, "DATABEND_PASSWORD", "") or ""
        except Exception:
            pass
    return (user or "databend", pw)


# Session settings applied to every Databend session: Snowflake semantics.
SESSION_SETTINGS = {
    "timezone": "UTC",                       # TIMESTAMP_NTZ: no zone shifts
    "date_format_style": "Oracle",           # Snowflake's YYYY-MM-DD tokens
    "enable_auto_detect_datetime_format": "1",  # Snowflake AUTO input format
    "quoted_ident_case_sensitive": "1",
}


class _Result:
    __slots__ = ("columns", "types", "rows", "query_id", "affect", "stats", "session")

    def __init__(self):
        self.columns: list[str] = []
        self.types: list[_ColType] = []
        self.rows: list[list] = []
        self.query_id: Optional[str] = None
        self.affect = None
        self.stats = None
        self.session = None


class DatabendTransport:
    """One logical Databend session (cookies + session object)."""

    def __init__(self, url=None, user=None, password=None, timeout: float = 600,
                 settings: Optional[dict] = None):
        self.url = (url or _settings_url()).rstrip("/")
        auth = _settings_auth()
        self.http = requests.Session()
        self.http.auth = (user or auth[0], password if password is not None else auth[1])
        self.http.headers.update({
            "X-DATABEND-CLIENT-CAPS": "session_cookie",
            "Content-Type": "application/json",
            "User-Agent": "afya-backends/1.0",
        })
        self.timeout = timeout
        self.session: dict = {"settings": dict(SESSION_SETTINGS, **(settings or {}))}
        self.lock = threading.RLock()

    # -- low level ----------------------------------------------------------
    def _post_query(self, sql: str, wait: int = 30, extra: Optional[dict] = None) -> dict:
        body = {"sql": sql, "session": self.session,
                "pagination": {"wait_time_secs": wait, "max_rows_per_page": 50_000}}
        if extra:
            body.update(extra)
        last_exc = None
        for attempt in range(3):
            try:
                r = self.http.post(f"{self.url}/v1/query", data=json.dumps(body, default=str),
                                   timeout=(10, wait + 120))
                break
            except requests.ConnectionError as exc:  # server not reachable: safe to retry
                last_exc = exc
                time.sleep(1 + attempt * 2)
        else:
            raise _make_error(DatabendOperationalError,
                              f"Databend not reachable at {self.url}: {last_exc}", 250001, "08001")
        return self._json(r)

    def _get(self, uri: str) -> dict:
        last_exc = None
        for attempt in range(4):
            try:
                r = self.http.get(f"{self.url}{uri}", timeout=(10, 300))
                return self._json(r)
            except requests.ConnectionError as exc:
                last_exc = exc
                time.sleep(1 + attempt * 2)
        raise _make_error(DatabendOperationalError, f"lost connection to Databend: {last_exc}", 250003, "08006")

    @staticmethod
    def _json(r: requests.Response) -> dict:
        if r.status_code == 401:
            raise _make_error(DatabendOperationalError, "Databend authentication failed (check DATABEND_USER/"
                              "DATABEND_PASSWORD)", 390100, "08004")
        try:
            return r.json()
        except ValueError:
            raise _make_error(DatabendOperationalError,
                              f"Databend HTTP {r.status_code}: {r.text[:300]}", 250000 + r.status_code, "08000")

    def _absorb_session(self, resp: dict) -> None:
        s = resp.get("session")
        if s:
            # Keep our semantic settings even if the server echoes others.
            merged = dict(s)
            merged["settings"] = dict(SESSION_SETTINGS, **(s.get("settings") or {}))
            for k, v in SESSION_SETTINGS.items():
                merged["settings"].setdefault(k, v)
            self.session = merged

    # -- public ---------------------------------------------------------------
    def start(self, sql: str, extra: Optional[dict] = None) -> dict:
        """Submit without waiting for completion (execute_async)."""
        with self.lock:
            resp = self._post_query(sql, wait=0, extra=extra)
            self._absorb_session(resp)
            if resp.get("error"):
                raise _translate_error(resp["error"], sql, resp.get("id"))
            return resp

    def run(self, sql: str, extra: Optional[dict] = None, first: Optional[dict] = None) -> _Result:
        with self.lock:
            resp = first or self._post_query(sql, extra=extra)
            res = _Result()
            res.query_id = resp.get("id")
            rows: list = []
            schema_seen = False
            while True:
                self._absorb_session(resp)
                if resp.get("error"):
                    final = resp.get("final_uri")
                    if final:
                        try:
                            self.http.get(f"{self.url}{final}", timeout=10)
                        except requests.RequestException:
                            pass
                    raise _translate_error(resp["error"], sql, res.query_id)
                if resp.get("schema") and not schema_seen:
                    res.columns = [c["name"] for c in resp["schema"]]
                    res.types = [_ColType(c["type"]) for c in resp["schema"]]
                    schema_seen = True
                if resp.get("data"):
                    rows.extend(resp["data"])
                res.affect = resp.get("affect") or res.affect
                res.stats = resp.get("stats") or res.stats
                nxt = resp.get("next_uri")
                if not nxt:
                    break
                if nxt.endswith("/final"):
                    fin = self._get(nxt)
                    self._absorb_session(fin)
                    if fin.get("error"):
                        raise _translate_error(fin["error"], sql, res.query_id)
                    break
                resp = self._get(nxt)
            res.rows = rows
            res.session = self.session
            return res

    def status(self, query_id: str) -> dict:
        return self._get(f"/v1/query/{query_id}")

    def kill(self, query_id: str) -> None:
        try:
            self.http.get(f"{self.url}/v1/query/{query_id}/kill", timeout=10)
        except requests.RequestException:
            pass

    def upload_to_stage(self, stage: str, path: str, filename: str, data) -> dict:
        r = self.http.put(f"{self.url}/v1/upload_to_stage",
                          headers={"stage_name": stage, "Content-Type": None},
                          files={"upload": (filename, data)}, timeout=(10, 3600))
        out = self._json(r)
        if out.get("error"):
            raise _translate_error(out["error"], f"PUT {path}", None)
        return out

    def streaming_load(self, insert_sql: str, filename: str, data, content_type="text/csv") -> dict:
        headers = {"X-DATABEND-SQL": insert_sql, "Content-Type": None,
                   "X-DATABEND-QUERY-CONTEXT": json.dumps(self.session)}
        r = self.http.put(f"{self.url}/v1/streaming_load", headers=headers,
                          files={"upload": (filename, data, content_type)}, timeout=(10, 3600))
        out = self._json(r)
        if out.get("error"):
            raise _translate_error(out["error"], insert_sql, out.get("id"))
        return out

    def close(self) -> None:
        try:
            self.http.post(f"{self.url}/v1/session/logout", timeout=5, json={})
        except Exception:
            pass
        self.http.close()


# ---------------------------------------------------------------------------
# Connection / cursor
# ---------------------------------------------------------------------------
_AUTO_CREATE = os.getenv("AFYA_DATABEND_AUTO_CREATE_SCHEMAS", "1").strip().lower() not in ("0", "false", "no")
_DML_RE = re.compile(r"^\s*(?:INSERT|UPDATE|DELETE|MERGE|COPY|REPLACE)\b", re.I)
_DDL_RE = re.compile(r"^\s*(?:CREATE|DROP|ALTER|TRUNCATE|RENAME|COMMENT|UNDROP)\b", re.I)
_UNKNOWN_DB_RE = re.compile(r"Unknown database [`'\"]?([^`'\"\s]+)", re.I)
_SELECTISH = re.compile(r"^\s*(?:SELECT|WITH|SHOW|DESC|DESCRIBE|EXPLAIN|LIST)\b", re.I)


class Connection:
    """DB-API 2.0 connection with the snowflake.connector surface."""

    paramstyle = "pyformat"

    def __init__(self, database: Optional[str] = None, schema: Optional[str] = None,
                 user: Optional[str] = None, url: Optional[str] = None, password: Optional[str] = None,
                 autocommit: Optional[bool] = None, session_parameters: Optional[dict] = None,
                 login_timeout: Optional[float] = None, network_timeout: Optional[float] = None,
                 **_ignored):
        primary = tr.default_primary_database()
        self._ctx = tr.Context(database=(database or primary).upper() if database and not database.startswith('"') else (database or primary),
                               schema=_norm_ident(schema) if schema else None,
                               primary_database=primary)
        self._t = DatabendTransport(url=url, user=None, password=None,
                                    timeout=float(network_timeout or 600))
        self._closed = False
        self._autocommit = True if autocommit is None else bool(autocommit)
        self._in_txn = False
        self._pending: dict[str, dict] = {}
        self.messages: list = []
        self.backend = "databend"
        from . import compat
        compat.ensure(self._t)
        if self._ctx.schema:
            self._use_databend_db(self._ctx.databend_db(self._ctx.database, self._ctx.schema), create=_AUTO_CREATE)
        if session_parameters and str(session_parameters.get("TIMEZONE", "")).strip():
            logger.debug("Databend: ignoring TIMEZONE session parameter (sessions stay UTC for NTZ semantics)")

    # -- snowflake-ish attributes --------------------------------------------
    @property
    def database(self):
        return self._ctx.database

    @property
    def schema(self):
        return self._ctx.schema

    @property
    def account(self):
        return "DATABEND"

    @property
    def warehouse(self):
        return None

    @property
    def role(self):
        return None

    @property
    def user(self):
        return self._t.http.auth[0]

    @property
    def session_id(self):
        return id(self)

    def is_closed(self) -> bool:
        return self._closed

    @property
    def expired(self) -> bool:
        return self._closed

    # -- context / lifecycle ----------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        self.close()
        return False

    def close(self, retry: bool = True) -> None:
        if self._closed:
            return
        try:
            if self._in_txn:
                self._raw("ROLLBACK")
        except Exception:
            pass
        self._t.close()
        self._closed = True

    def cursor(self, cursor_class=None):
        if self._closed:
            raise _make_error(DatabendInterfaceError, "Connection is closed", 251001, "08003")
        as_dict = cursor_class is not None and "dict" in getattr(cursor_class, "__name__", "").lower()
        return Cursor(self, as_dict=as_dict)

    # -- transactions -------------------------------------------------------
    def autocommit(self, mode: bool) -> None:
        mode = bool(mode)
        if mode and self._in_txn:
            self.commit()
        self._autocommit = mode

    def commit(self) -> None:
        if self._in_txn:
            self._raw("COMMIT")
            self._in_txn = False

    def rollback(self) -> None:
        if self._in_txn:
            self._raw("ROLLBACK")
            self._in_txn = False

    def _raw(self, sql: str) -> _Result:
        return self._t.run(sql)

    # -- statement execution ---------------------------------------------
    def _use_databend_db(self, db: Optional[str], create: bool = False) -> bool:
        if not db:
            return False
        q = '"' + db.replace('"', '""') + '"'
        try:
            self._t.run(f"USE {q}")
            return True
        except DatabendProgrammingError:
            if create:
                self._t.run(f"CREATE DATABASE IF NOT EXISTS {q}")
                self._t.run(f"USE {q}")
                return True
            return False

    def _execute_statements(self, sql: str, cursor: "Cursor") -> _Result:
        """Translate and run; returns the result of the last statement."""
        statements = tr.translate(sql, self._ctx)
        result = _Result()
        if not statements:
            return result
        for st in statements:
            result = self._run_statement(st, cursor)
        return result

    def _run_statement(self, st: "tr.Statement", cursor: "Cursor") -> _Result:
        a = st.action
        if a == "noop":
            r = _Result()
            r.columns, r.types, r.rows = ["status"], [_ColType("String")], [["Statement executed successfully."]]
            return r
        if a == "use_database":
            db = st.params["database"]
            self._ctx = tr.Context(database=_norm_ident(db), schema="PUBLIC",
                                   primary_database=self._ctx.primary_database)
            self._use_databend_db(self._ctx.databend_db(self._ctx.database, "PUBLIC"), create=True)
            return _status_result(f"Statement executed successfully.")
        if a in ("use_schema", "create_schema"):
            schema = _norm_ident(st.params["schema"])
            db = _norm_ident(st.params.get("database")) if st.params.get("database") else self._ctx.database
            if a == "create_schema":
                self._run_sql(st.sql, cursor)
            self._ctx = tr.Context(database=db, schema=schema, primary_database=self._ctx.primary_database)
            ok = self._use_databend_db(self._ctx.databend_db(db, schema), create=False)
            if not ok and a == "use_schema":
                raise _make_error(DatabendProgrammingError,
                                  f"SQL compilation error: Object does not exist or not authorized. Schema '{schema}'",
                                  2003, "02000")
            return _status_result(f"Schema {schema} successfully {'created' if a == 'create_schema' else 'used'}.")
        if a == "truncate_if_exists":
            for t in st.params["tables"]:
                try:
                    self._run_sql(f"TRUNCATE TABLE {t}", cursor)
                except DatabendProgrammingError as e:
                    if getattr(e, "errno", None) != 2003:
                        raise
            return _status_result("Statement executed successfully.")
        if a == "swap":
            ta, tb = st.params["a"], st.params["b"]
            tmp = _sibling(ta, f"_AFYA_SWAP_{uuid.uuid4().hex[:8].upper()}")
            self._run_sql(f"ALTER TABLE {ta} RENAME TO {_last(tmp)}", cursor)
            self._run_sql(f"ALTER TABLE {tb} RENAME TO {_last(ta)}", cursor) if _same_db(ta, tb) else \
                self._run_sql(f"ALTER TABLE {tb} RENAME TO {_last(ta)}", cursor)
            self._run_sql(f"ALTER TABLE {tmp} RENAME TO {_last(tb)}", cursor)
            return _status_result("Statement executed successfully.")
        if a == "put":
            return self._put(st)
        if a == "remove":
            return self._run_sql(st.sql, cursor)
        if a == "copy":
            for stage in st.params.get("stages", []):
                self._ensure_stage(stage)
            for ff in st.params.get("file_formats", []):
                self._ensure_file_format(ff)
            return _copy_result_to_snowflake(self._run_sql(st.sql, cursor))
        return self._run_sql(st.sql, cursor)

    # -- stages / file formats (Snowflake objects that must exist on Databend)
    _known_stages: set = set()
    _known_formats: set = set()

    def _ensure_stage(self, name: str) -> None:
        if name == "~" or name in Connection._known_stages:
            return
        try:
            res = self._t.run(f"DESC STAGE {name}")
            if res.rows:
                Connection._known_stages.add(name)
                return
        except DatabendProgrammingError:
            pass
        url = external_stage_url(name)
        if url is None:  # internal stage (PUT target / table stage)
            self._t.run(f"CREATE STAGE IF NOT EXISTS {name}")
        else:
            self._t.run(f"CREATE STAGE IF NOT EXISTS {name} URL = '{url}' {_s3_connection_clause()}")
            logger.info("Databend: created external stage %s -> %s", name, url)
        Connection._known_stages.add(name)

    def _ensure_file_format(self, name: str) -> None:
        if name in Connection._known_formats:
            return
        spec = file_format_spec(name)
        self._t.run(f"CREATE FILE FORMAT IF NOT EXISTS {name} {spec}")
        Connection._known_formats.add(name)

    def _expand_hash_star(self, sql: str) -> str:
        marker = tr.HASH_STAR_MARKER + "('"
        while marker in sql:
            start = sql.index(marker)
            end = sql.index("')", start)
            table_ref = sql[start + len(marker): end].replace("''", "'")
            parts = re.findall(r'"((?:[^"]|"")*)"', table_ref)
            if len(parts) == 2:
                where = f"database = '{parts[0]}' AND \"table\" = '{parts[1]}'"
            else:
                where = f"database = CURRENT_DATABASE() AND \"table\" = '{parts[-1]}'"
            cols = [r[0] for r in self._t.run(f"SELECT name FROM system.columns WHERE {where}").rows]
            if not cols:
                raise _make_error(DatabendProgrammingError,
                                  f"SQL compilation error: Object '{table_ref}' does not exist or not authorized.",
                                  2003, "42S02")
            expr = ("XXHASH64(CONCAT_WS('\\x1f', "
                    + ", ".join(f"COALESCE(CAST(\"{c}\" AS VARCHAR), '\\\\N')" for c in cols) + "))")
            sql = sql[:start] + expr + sql[end + 2:]
        return sql

    def _run_sql(self, sql: str, cursor: "Cursor") -> _Result:
        res = self._run_sql_raw(sql, cursor)
        if _MERGE_RE.match(sql):
            return _merge_result_to_snowflake(sql, res)
        return res

    def _run_sql_raw(self, sql: str, cursor: "Cursor") -> _Result:
        if tr.HASH_STAR_MARKER in sql:
            sql = self._expand_hash_star(sql)
        if not self._autocommit:
            if _DDL_RE.match(sql) and self._in_txn:
                self._raw("COMMIT")  # Snowflake commits implicitly before DDL
                self._in_txn = False
            elif _DML_RE.match(sql) and not self._in_txn:
                self._raw("BEGIN")
                self._in_txn = True
        upper = sql.lstrip()[:12].upper()
        if upper.startswith(("BEGIN", "START")):
            self._in_txn = True
        elif upper.startswith(("COMMIT", "ROLLBACK", "ABORT")):
            self._in_txn = False
        try:
            return self._t.run(sql)
        except DatabendProgrammingError as e:
            m = _UNKNOWN_DB_RE.search(str(e))
            if _AUTO_CREATE and m and (_DDL_RE.match(sql) or _DML_RE.match(sql)):
                db = m.group(1).strip('"`')
                logger.info("Databend: creating missing database %s (Snowflake schema)", db)
                self._t.run('CREATE DATABASE IF NOT EXISTS "' + db.replace('"', '""') + '"')
                if self._ctx.schema and self._ctx.databend_db(self._ctx.database, self._ctx.schema) == db:
                    self._use_databend_db(db)
                return self._t.run(sql)
            raise

    def _put(self, st: "tr.Statement") -> _Result:
        import glob
        stage = st.params["stage"]
        if stage != "~":
            self._t.run(f"CREATE STAGE IF NOT EXISTS {stage}")
        sub = st.params.get("subpath", "").strip("/")
        rows = []
        paths = sorted(glob.glob(st.params["path"])) or [st.params["path"]]
        for p in paths:
            name = os.path.basename(p)
            target = f"{sub}/{name}" if sub else name
            with open(p, "rb") as fh:
                self._t.upload_to_stage(stage, p, target, fh)
            size = os.path.getsize(p)
            rows.append([name, target, size, size, "NONE", "NONE", "UPLOADED", ""])
        r = _Result()
        r.columns = ["source", "target", "source_size", "target_size", "source_compression",
                     "target_compression", "status", "message"]
        r.types = [_ColType("String")] * 2 + [_ColType("Int64")] * 2 + [_ColType("String")] * 4
        r.rows = [[str(x) for x in row] for row in rows]
        return r

    # -- snowflake.connector extras ----------------------------------------
    def execute_string(self, sql_text: str, remove_comments: bool = False, return_cursors: bool = True,
                       cursor_class=None, **_kw):
        cursors = []
        for stmt in tr.split_statements(sql_text):
            cur = self.cursor(cursor_class)
            cur.execute(stmt)
            cursors.append(cur)
        return cursors if return_cursors else []

    def execute_stream(self, stream, remove_comments: bool = False, cursor_class=None, **_kw):
        text = stream.read() if hasattr(stream, "read") else str(stream)
        for cur in self.execute_string(text, cursor_class=cursor_class):
            yield cur

    def get_query_status(self, sfqid: str):
        return _query_status(self, sfqid)

    def get_query_status_throw_if_error(self, sfqid: str):
        st = _query_status(self, sfqid)
        pend = self._pending.get(sfqid)
        if pend and pend.get("error"):
            raise _translate_error(pend["error"], pend.get("sql", ""), sfqid)
        return st

    def is_still_running(self, status) -> bool:
        return getattr(status, "name", str(status)) in ("RUNNING", "QUEUED", "RESUMING_WAREHOUSE")

    def is_an_error(self, status) -> bool:
        return getattr(status, "name", str(status)) in ("FAILED_WITH_ERROR", "ABORTED", "FAILED_WITH_INCIDENT")


def _query_status(conn: Connection, sfqid: str):
    try:
        from snowflake.connector.constants import QueryStatus
    except Exception:  # pragma: no cover
        QueryStatus = None
    pend = conn._pending.get(sfqid)
    state = "Succeeded"
    if pend is not None and not pend.get("done"):
        try:
            resp = conn._t.status(sfqid)
            state = resp.get("state") or "Running"
            if resp.get("error"):
                pend["error"] = resp["error"]
                state = "Failed"
        except Exception:
            state = "Running"
    elif pend is not None and pend.get("error"):
        state = "Failed"
    name = {"Starting": "RUNNING", "Running": "RUNNING", "Succeeded": "SUCCESS", "Failed": "FAILED_WITH_ERROR"} \
        .get(state, "RUNNING")
    return QueryStatus[name] if QueryStatus is not None else name


# Snowflake stages that point at S3.  Override/extend with
# AFYA_DATABEND_STAGES='{"stage_name": "s3://bucket/prefix/"}'.
DEFAULT_EXTERNAL_STAGES = {
    "facility_raw_stage": "s3://collabmedbucket/",
    "db_bucket": "s3://collabmedbucket/",
}
# Snowflake named file formats -> Databend equivalents.  JSON_FF is
# gzipped JSON lines read as one VARIANT per line ($1).
DEFAULT_FILE_FORMATS = {
    "json_ff": "TYPE = NDJSON COMPRESSION = AUTO",
}


def external_stage_url(name: str) -> Optional[str]:
    stages = dict(DEFAULT_EXTERNAL_STAGES)
    raw = os.getenv("AFYA_DATABEND_STAGES", "").strip()
    if raw:
        try:
            stages.update({k.lower(): v for k, v in json.loads(raw).items()})
        except ValueError:
            logger.warning("AFYA_DATABEND_STAGES is not valid JSON; ignoring")
    return stages.get(name.lower())


def file_format_spec(name: str) -> str:
    formats = dict(DEFAULT_FILE_FORMATS)
    raw = os.getenv("AFYA_DATABEND_FILE_FORMATS", "").strip()
    if raw:
        try:
            formats.update({k.lower(): v for k, v in json.loads(raw).items()})
        except ValueError:
            logger.warning("AFYA_DATABEND_FILE_FORMATS is not valid JSON; ignoring")
    return formats.get(name.lower(), "TYPE = NDJSON COMPRESSION = AUTO")


def _s3_connection_clause() -> str:
    key = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    secret = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
    region = (os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1").strip()
    if not (key and secret):
        raise _make_error(DatabendOperationalError,
                          "Databend needs AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY to create the S3 stage the "
                          "pipeline COPYs from (Snowflake's stage credentials are not visible to Databend).",
                          250002, "08000")
    esc = lambda s: s.replace("'", "''")
    return f"CONNECTION = (ACCESS_KEY_ID = '{esc(key)}', SECRET_ACCESS_KEY = '{esc(secret)}', REGION = '{esc(region)}')"


_MERGE_RE = re.compile(r"^\s*MERGE\b", re.I)


def _merge_result_to_snowflake(sql: str, res: _Result) -> _Result:
    """Snowflake's MERGE returns one count column per action kind in the
    statement, in the order inserted, updated, deleted.  Databend omits a
    column when (it decides) that action can't happen - e.g. only "number
    of rows inserted" when the target is empty - which breaks callers doing
    ``inserted, updated = result[0]``.  Rebuild the Snowflake shape."""
    upper = re.sub(r"'(?:[^']|'')*'", "''", sql.upper())
    kinds = []
    if re.search(r"\bTHEN\s+INSERT\b", upper):
        kinds.append("inserted")
    if re.search(r"\bTHEN\s+UPDATE\b", upper):
        kinds.append("updated")
    if re.search(r"\bTHEN\s+DELETE\b", upper):
        kinds.append("deleted")
    got = {}
    for i, c in enumerate(res.columns):
        m = re.search(r"rows (inserted|updated|deleted)", c, re.I)
        if m and res.rows:
            got[m.group(1).lower()] = res.rows[0][i]
    out = _Result()
    out.query_id, out.session, out.stats, out.affect = res.query_id, res.session, res.stats, res.affect
    out.columns = [f"number of rows {k}" for k in kinds]
    out.types = [_ColType("UInt64") for _ in kinds]
    out.rows = [[str(got.get(k, 0) or 0) for k in kinds]]
    return out


_SF_COPY_COLUMNS = ["file", "status", "rows_parsed", "rows_loaded", "error_limit", "errors_seen",
                    "first_error", "first_error_line", "first_error_character", "first_error_column_name"]


def _copy_result_to_snowflake(res: _Result) -> _Result:
    """Reshape Databend's COPY result (File, Rows_loaded, Errors_seen,
    First_error, First_error_line) into Snowflake's column layout, so code
    reading row[3] (rows_loaded) or the status column keeps working."""
    if not res.columns or res.columns[0].lower() != "file":
        return res
    idx = {c.lower(): i for i, c in enumerate(res.columns)}
    out = _Result()
    out.query_id, out.session, out.stats = res.query_id, res.session, res.stats
    out.columns = list(_SF_COPY_COLUMNS)
    out.types = [_ColType("String"), _ColType("String"), _ColType("Int64"), _ColType("Int64"), _ColType("Int64"),
                 _ColType("Int64"), _ColType("Nullable(String)"), _ColType("Nullable(Int64)"),
                 _ColType("Nullable(Int64)"), _ColType("Nullable(String)")]
    rows = []
    for r in res.rows:
        loaded = int(r[idx.get("rows_loaded", 1)] or 0)
        errors = int(r[idx.get("errors_seen", 2)] or 0)
        status = "LOADED" if errors == 0 else ("PARTIALLY_LOADED" if loaded else "LOAD_FAILED")
        rows.append([r[idx.get("file", 0)], status, str(loaded + errors), str(loaded), "1" if errors else "0",
                     str(errors), r[idx["first_error"]] if "first_error" in idx else None,
                     r[idx["first_error_line"]] if "first_error_line" in idx else None, None, None])
    out.rows = rows
    out.affect = {"copy_rows_loaded": sum(int(r[3]) for r in rows)}
    return out


def _status_result(msg: str) -> _Result:
    r = _Result()
    r.columns, r.types, r.rows = ["status"], [_ColType("String")], [[msg]]
    return r


def _norm_ident(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    name = str(name).strip()
    if name.startswith('"') and name.endswith('"'):
        return name[1:-1].replace('""', '"')
    return name.upper()


def _last(qualified: str) -> str:
    parts = re.findall(r'"(?:[^"]|"")*"|[^.]+', qualified)
    return parts[-1] if parts else qualified


def _sibling(qualified: str, new_name: str) -> str:
    parts = re.findall(r'"(?:[^"]|"")*"|[^.]+', qualified)
    parts[-1] = f'"{new_name}"'
    return ".".join(parts)


def _same_db(a: str, b: str) -> bool:
    pa = re.findall(r'"(?:[^"]|"")*"|[^.]+', a)
    pb = re.findall(r'"(?:[^"]|"")*"|[^.]+', b)
    return pa[:-1] == pb[:-1]


class _Sentinel(str):
    """A parameter stand-in that renders as a unique string literal."""

    def __new__(cls, i: int):
        return super().__new__(cls, f"\x01afya_p{i}\x01")


_SENTINEL_RE = re.compile(r"'\x01afya_p(\d+)\x01'")


_INSERT_VALUES_RE = re.compile(
    r"^\s*INSERT\s+(?:OVERWRITE\s+)?INTO\s+.+?\bVALUES\s*\((?P<tuple>.*)\)\s*;?\s*$",
    re.IGNORECASE | re.DOTALL,
)


class Cursor:
    """DB-API cursor with the snowflake.connector SnowflakeCursor surface."""

    arraysize = 1

    def __init__(self, conn: Connection, as_dict: bool = False):
        self.connection = conn
        self._as_dict = as_dict
        self._rows: list = []
        self._types: list[_ColType] = []
        self._cols: list[str] = []
        self._pos = 0
        self._rowcount: Optional[int] = None
        self._description = None
        self.sfqid: Optional[str] = None
        self.query: Optional[str] = None
        self.messages: list = []
        self._closed = False
        self.timestamp_output_format = None

    # -- props ----------------------------------------------------------------
    @property
    def description(self):
        return self._description

    @property
    def rowcount(self):
        return self._rowcount

    @property
    def query_id(self):
        return self.sfqid

    @property
    def sqlstate(self):
        return None

    def is_closed(self):
        return self._closed

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __iter__(self):
        while True:
            row = self.fetchone()
            if row is None:
                return
            yield row

    def close(self):
        self._closed = True
        return True

    # -- execute -------------------------------------------------------------
    def execute(self, command: str, params=None, *, timeout=None, _statement_params=None,
                num_statements=None, _no_results: bool = False, **_kw):
        if self._closed:
            raise _make_error(DatabendInterfaceError, "Cursor is closed", 251006, "HY010")
        if command is None or not str(command).strip():
            self._load(_Result())
            return None
        sql = bind(command, params)
        self.query = sql
        res = self.connection._execute_statements(sql, self)
        self._load(res)
        return self

    def executemany(self, command: str, seqparams):
        seqparams = list(seqparams)
        if not seqparams:
            self._rowcount = 0
            return self
        m = _INSERT_VALUES_RE.match(command)
        if m:
            # Translate the statement ONCE with sentinel string literals in
            # place of the parameters, then substitute each row's values
            # into the translated VALUES tuple: one multi-row INSERT per
            # chunk (what snowflake.connector does too).  Sentinels survive
            # any function rewrite around a placeholder (TO_TIMESTAMP_NTZ(%s)
            # -> TO_TIMESTAMP('...')), even one that duplicates it.
            first = seqparams[0]
            if isinstance(first, dict):
                sentinels = {k: _Sentinel(i) for i, k in enumerate(first)}
                keys = list(first)
            else:
                sentinels = [_Sentinel(i) for i in range(len(first))]
                keys = None
            stmts = tr.translate(bind(command, sentinels), self.connection._ctx)
            if len(stmts) == 1 and stmts[0].action == "sql":
                tsql = stmts[0].sql
                vpos = tsql.upper().rfind(" VALUES ")
                if vpos > 0 and tsql[vpos + 8:].lstrip().startswith("("):
                    prefix = tsql[: vpos] + " VALUES "
                    tuple_tpl = tsql[vpos + 8:].strip()
                    pieces = _SENTINEL_RE.split(tuple_tpl)
                    total = 0
                    chunk, size = [], 0
                    for p in seqparams:
                        vals = [p[k] for k in keys] if keys is not None else list(p)
                        out = []
                        for j, piece in enumerate(pieces):
                            out.append(to_literal(vals[int(piece)]) if j % 2 else piece)
                        tup = "".join(out)
                        chunk.append(tup)
                        size += len(tup)
                        if size > 4_000_000 or len(chunk) >= 20_000:
                            total += self._insert_chunk(prefix, chunk)
                            chunk, size = [], 0
                    if chunk:
                        total += self._insert_chunk(prefix, chunk)
                    self._rowcount = total
                    self._rows, self._pos = [], 0
                    self._description = [ResultMetadata("number of rows inserted", TYPE_FIXED, None, None, 38, 0,
                                                        False)]
                    return self
        total = 0
        for p in seqparams:
            self.execute(command, p)
            total += self._rowcount or 0
        self._rowcount = total
        return self

    def _insert_chunk(self, prefix: str, chunk: list) -> int:
        res = self.connection._run_sql(prefix + ",".join(chunk), self)
        self.sfqid = res.query_id
        if res.rows and res.rows[0]:
            try:
                return int(res.rows[0][0])
            except (TypeError, ValueError):
                return len(chunk)
        return len(chunk)

    def execute_async(self, command: str, params=None, *, _statement_params=None, **_kw):
        sql = bind(command, params)
        statements = tr.translate(sql, self.connection._ctx)
        conn = self.connection
        # Client-side actions and multi-statement text run synchronously;
        # the last plain statement is started asynchronously.
        for st in statements[:-1]:
            conn._run_statement(st, self)
        last = statements[-1] if statements else tr.Statement("noop")
        if last.action != "sql":
            res = conn._run_statement(last, self)
            qid = "afya-" + uuid.uuid4().hex
            conn._pending[qid] = {"done": True, "result": res, "sql": sql}
            self.sfqid = qid
            return {"queryId": qid}
        resp = conn._t.start(last.sql)
        qid = resp.get("id") or ("afya-" + uuid.uuid4().hex)
        conn._pending[qid] = {"done": False, "first": resp, "sql": last.sql}
        self.sfqid = qid
        self.query = sql
        return {"queryId": qid}

    def get_results_from_sfqid(self, sfqid: str):
        conn = self.connection
        pend = conn._pending.get(sfqid)
        if pend is None:
            raise _make_error(DatabendProgrammingError, f"Unknown query id {sfqid}", 2003, "02000")
        if not pend.get("done"):
            try:
                pend["result"] = conn._t.run(pend["sql"], first=pend["first"])
            except Exception as exc:
                pend["done"] = True
                pend["error"] = {"code": getattr(exc, "errno", 0), "message": str(exc)}
                raise
            pend["done"] = True
        if pend.get("error"):
            raise _translate_error(pend["error"], pend.get("sql", ""), sfqid)
        self._load(pend["result"])
        self.sfqid = sfqid
        return self

    # -- results -------------------------------------------------------------
    def _load(self, res: _Result):
        self._cols = res.columns
        self._types = res.types
        conv = [t.convert for t in res.types]
        rows = []
        for raw in res.rows:
            rows.append(tuple(c(v) for c, v in zip(conv, raw)))
        self._rows = rows
        self._pos = 0
        self.sfqid = res.query_id or self.sfqid
        if res.columns:
            self._description = [
                ResultMetadata(name, t.type_code, None, None, t.precision, t.scale, t.nullable)
                for name, t in zip(res.columns, res.types)
            ]
        else:
            self._description = None
        dml = res.columns[:1] and res.columns[0].startswith("number of rows")
        if isinstance(res.affect, dict) and "copy_rows_loaded" in res.affect:
            self._rowcount = res.affect["copy_rows_loaded"]
        elif dml:
            try:
                self._rowcount = sum(int(x) for x in (res.rows[0] if res.rows else []) if x is not None)
            except (TypeError, ValueError):
                self._rowcount = len(rows)
        else:
            self._rowcount = len(rows)

    def _shape(self, row):
        if row is None or not self._as_dict:
            return row
        return dict(zip(self._cols, row))

    def fetchone(self):
        if self._pos >= len(self._rows):
            return None
        row = self._rows[self._pos]
        self._pos += 1
        return self._shape(row)

    def fetchmany(self, size: Optional[int] = None):
        size = size or self.arraysize
        out = self._rows[self._pos: self._pos + size]
        self._pos += len(out)
        return [self._shape(r) for r in out]

    def fetchall(self):
        out = self._rows[self._pos:]
        self._pos = len(self._rows)
        return [self._shape(r) for r in out]

    def fetch_pandas_all(self, **_kw):
        return _to_pandas(self._cols, self._types, self._rows[self._pos:])

    def fetch_pandas_batches(self, **_kw):
        df = self.fetch_pandas_all()
        step = 100_000
        for i in range(0, max(len(df), 1), step):
            yield df.iloc[i:i + step]

    def fetch_arrow_all(self, **_kw):
        import pyarrow as pa
        return pa.Table.from_pandas(self.fetch_pandas_all(), preserve_index=False)

    def setinputsizes(self, *a):
        pass

    def setoutputsize(self, *a):
        pass


def _to_pandas(cols, types, rows):
    import pandas as pd
    if not cols:
        return pd.DataFrame()
    series = []
    for i, (name, t) in enumerate(zip(cols, types)):
        values = [r[i] for r in rows]
        k = t.kind
        if k == "int":
            if any(v is None for v in values):
                s = pd.Series([float(v) if v is not None else float("nan") for v in values], dtype="float64")
            else:
                s = pd.Series(values, dtype="int64")
        elif k in ("decimal", "float"):
            s = pd.Series([float(v) if v is not None else float("nan") for v in values], dtype="float64")
        elif k == "timestamp":
            s = pd.to_datetime(pd.Series(values, dtype="object"), errors="coerce")
        elif k == "bool":
            s = pd.Series(values, dtype="object" if any(v is None for v in values) else "bool")
        else:
            s = pd.Series(values, dtype="object")
        series.append(s)
    # Positional build keeps duplicate column names (SELECT a.ID, b.ID).
    df = pd.concat(series, axis=1, ignore_index=True) if series else pd.DataFrame()
    df.columns = list(cols)
    return df
