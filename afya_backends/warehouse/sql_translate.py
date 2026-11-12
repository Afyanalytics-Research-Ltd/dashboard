"""Snowflake SQL -> Databend SQL translation.

Every statement the platform sends to "the warehouse" is written in
Snowflake SQL (pipelines, dashboards, agents, Redash queries, dbt models).
When the warehouse toggle is on Databend, :func:`translate` rewrites each
statement so Databend runs it with the same meaning and the same result
shape (column names, NULL handling, date maths).

How it works
------------
1. The statement is parsed with sqlglot's Snowflake dialect into a tree.
2. Tree rewrites map Snowflake-only objects onto Databend:
   * ``DB.SCHEMA.TABLE`` -> Databend ``"SCHEMA"."TABLE"`` (Databend has no
     schema level: each Snowflake schema of the primary database becomes a
     Databend database of the same name; other Snowflake databases become
     ``"<DB>__<SCHEMA>"``).
   * ``INFORMATION_SCHEMA.*`` -> Snowflake-shaped compatibility views in
     ``"AFYA_SF_COMPAT"`` (see compat.py) so TABLE_SCHEMA/DATA_TYPE/... keep
     their Snowflake names and values.
   * Unquoted identifiers are upper-cased and every identifier is emitted
     quoted.  That reproduces Snowflake's identifier semantics exactly on
     Databend: ``select id`` resolves column ``ID``, ``"source_schema"``
     stays lower case, and result columns come back upper-case just like
     Snowflake returns them.
3. The tree is generated with the :class:`Databend` dialect below, which
   inherits Snowflake's generator and overrides only what Databend spells
   differently (verified one by one against databend-query v1.2.881; see
   afya_backends/tests/test_translate_live.py).
4. Statements with no Databend meaning (USE WAREHOUSE, GRANT, ALTER
   SESSION, CREATE STAGE ...) become no-ops; a few (TRUNCATE IF EXISTS,
   ALTER TABLE SWAP WITH, PUT, SHOW, DESCRIBE) become *actions* that the
   Databend connection carries out itself (see dbapi.py).

Date/time format strings are preserved verbatim: Databend's default
``date_format_style = Oracle`` understands Snowflake's YYYY-MM-DD HH24:MI:SS
tokens, so they are never re-written.

This module must not import Django or Airflow.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import sqlglot
from sqlglot import exp
from sqlglot.dialects.snowflake import Snowflake
from sqlglot.errors import ParseError, TokenError

logger = logging.getLogger(__name__)

COMPAT_DB = "AFYA_SF_COMPAT"

# Output columns of Snowflake's FLATTEN table function.  Databend's FLATTEN
# produces the same columns but lower-case, so references to them must not
# be upper-cased.
FLATTEN_COLUMNS = frozenset({"SEQ", "KEY", "PATH", "INDEX", "VALUE", "THIS"})

# Snowflake statements that have no Databend equivalent and no effect on
# data.  They become no-ops (an empty result), like they would be for any
# client that doesn't care about warehouses/roles/session tags.
_NOOP_COMMANDS = re.compile(
    r"^\s*(?:"
    r"USE\s+(?:WAREHOUSE|ROLE|SECONDARY\s+ROLES)\b"
    r"|ALTER\s+(?:SESSION|WAREHOUSE|USER|ACCOUNT|ROLE)\b"
    r"|(?:GRANT|REVOKE)\b"
    r"|CREATE\s+(?:OR\s+REPLACE\s+)?(?:ROLE|USER|WAREHOUSE|NETWORK\s+POLICY|AUTHENTICATION\s+POLICY"
    r"|ROW\s+ACCESS\s+POLICY|MASKING\s+POLICY|RESOURCE\s+MONITOR|TAG|FILE\s+FORMAT|INTEGRATION|"
    r"STORAGE\s+INTEGRATION|NOTIFICATION\s+INTEGRATION|API\s+INTEGRATION|SECURITY\s+INTEGRATION)\b"
    r"|CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP(?:ORARY)?\s+)?FILE\s+FORMAT\b"
    r"|DROP\s+(?:ROLE|USER|WAREHOUSE|FILE\s+FORMAT|ROW\s+ACCESS\s+POLICY|MASKING\s+POLICY)\b"
    r"|SHOW\s+(?:GRANTS|ROLES|USERS|WAREHOUSES|PARAMETERS|INTEGRATIONS)\b"
    r"|ALTER\s+TABLE\s+\S+\s+(?:ADD|DROP)\s+ROW\s+ACCESS\s+POLICY\b"
    r"|ALTER\s+TABLE\s+\S+\s+(?:SET|UNSET)\s+(?:DATA_RETENTION|CHANGE_TRACKING|COMMENT|TAG|MAX_DATA_EXTENSION)"
    r"|EXECUTE\s+TASK\b"
    r")",
    re.IGNORECASE,
)

_PUT_RE = re.compile(
    r"^\s*PUT\s+'?file://(?P<path>[^'\s]+)'?\s+(?P<stage>@\S+)(?P<opts>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_REMOVE_RE = re.compile(r"^\s*(?:REMOVE|RM)\s+(?P<stage>@\S+)", re.IGNORECASE)
_LIST_RE = re.compile(r"^\s*(?:LIST|LS)\s+(?P<stage>@\S+)", re.IGNORECASE)
_IDENTIFIER_RE = re.compile(r"\bIDENTIFIER\s*\(\s*'([^']+)'\s*\)", re.IGNORECASE)
# Statements kept as plain text (no parse) - Databend accepts them as is.
_PASSTHROUGH_RE = re.compile(r"^\s*(?:BEGIN(?:\s+TRANSACTION)?|START\s+TRANSACTION|COMMIT|ROLLBACK|ABORT)\s*;?\s*$", re.I)


@dataclass(frozen=True)
class Context:
    """Where unqualified names resolve, in Snowflake terms."""

    database: str = "HOSPITALS"        # current Snowflake database
    schema: Optional[str] = None       # current Snowflake schema
    primary_database: str = "HOSPITALS"

    def databend_db(self, sf_db: Optional[str], sf_schema: Optional[str]) -> Optional[str]:
        return map_database(sf_db or self.database, sf_schema, self.primary_database)


def default_primary_database() -> str:
    return (os.getenv("AFYA_WAREHOUSE_PRIMARY_DATABASE")
            or os.getenv("SNOWFLAKE_DATABASE") or "HOSPITALS").strip().upper() or "HOSPITALS"


def map_database(sf_db: Optional[str], sf_schema: Optional[str], primary: str) -> Optional[str]:
    """Snowflake (database, schema) -> Databend database name."""
    if not sf_schema:
        return None
    if sf_schema.upper() == "INFORMATION_SCHEMA":
        return COMPAT_DB
    if not sf_db or sf_db.upper() == primary.upper():
        return sf_schema
    return f"{sf_db}__{sf_schema}"


@dataclass
class Statement:
    """One translated statement.

    ``action`` is ``"sql"`` (run ``sql`` on Databend), ``"noop"`` (return an
    empty result), or one of the client-side actions handled by the Databend
    connection: ``"use_database"``, ``"use_schema"``, ``"put"``, ``"remove"``,
    ``"list"``, ``"truncate_if_exists"``, ``"swap"``, ``"create_schema"``.
    """

    action: str
    sql: str = ""
    params: dict = field(default_factory=dict)
    original: str = ""


class TranslationError(Exception):
    """The statement could not be translated (sqlglot could not parse it)."""


# ---------------------------------------------------------------------------
# Databend dialect: Snowflake generator + Databend spellings
# ---------------------------------------------------------------------------

def _func(name):
    return lambda self, e: self.func(name, *[a for a in e.args.values() if isinstance(a, exp.Expression)][:1])


def _ilike_sql(self, e: exp.ILike) -> str:
    right = e.expression
    left = self.sql(e, "this")
    if isinstance(right, (exp.Any, exp.All)):
        items = right.this.expressions if isinstance(right.this, (exp.Tuple, exp.Paren)) and hasattr(right.this, "expressions") else [right.this]
        if isinstance(right.this, exp.Paren):
            items = [right.this.this]
        joiner = " OR " if isinstance(right, exp.Any) else " AND "
        parts = [f"LOWER({left}) LIKE LOWER({self.sql(i)})" for i in items]
        return "(" + joiner.join(parts) + ")"
    escape = e.args.get("escape")
    sql = f"LOWER({left}) LIKE LOWER({self.sql(right)})"
    if escape:
        sql += f" ESCAPE {self.sql(escape)}"
    return sql


def _anonymous_sql(self, e: exp.Anonymous) -> str:
    name = (e.name or "").upper()
    args = e.expressions
    a = [self.sql(x) for x in args]
    if name in ("TO_TIMESTAMP_NTZ", "TO_TIMESTAMP_LTZ", "TO_TIMESTAMP_TZ"):
        return self.func("TO_TIMESTAMP", *args)
    if name in ("TRY_TO_TIMESTAMP_NTZ", "TRY_TO_TIMESTAMP_LTZ", "TRY_TO_TIMESTAMP_TZ", "TRY_TO_TIMESTAMP"):
        return self.func("TRY_TO_TIMESTAMP", *args)
    if name == HASH_STAR_MARKER:
        return f"{HASH_STAR_MARKER}({a[0]})"
    if name == "HASH":
        if len(a) == 1:
            return f"XXHASH64(CAST({a[0]} AS VARCHAR))"
        inner = ", ".join(f"COALESCE(CAST({x} AS VARCHAR), '\\\\N')" for x in a)
        return f"XXHASH64(CONCAT_WS('\\x1f', {inner}))"
    if name in ("AS_VARCHAR", "AS_CHAR"):
        return f"TRY_CAST({a[0]} AS VARCHAR)"
    if name in ("AS_DOUBLE", "AS_REAL", "AS_DECIMAL", "AS_NUMBER"):
        return f"TRY_CAST({a[0]} AS DOUBLE)"
    if name == "AS_INTEGER":
        return f"TRY_CAST({a[0]} AS BIGINT)"
    if name == "AS_BOOLEAN":
        return f"TRY_CAST({a[0]} AS BOOLEAN)"
    if name == "AS_DATE":
        return f"TRY_CAST({a[0]} AS DATE)"
    if name in ("AS_TIMESTAMP_NTZ", "AS_TIMESTAMP_LTZ", "AS_TIMESTAMP_TZ"):
        return f"TRY_CAST({a[0]} AS TIMESTAMP)"
    if name in ("AS_OBJECT", "AS_ARRAY", "TO_OBJECT", "TO_VARIANT"):
        return f"CAST({a[0]} AS VARIANT)" if name == "TO_VARIANT" else a[0]
    if name == "IS_VARCHAR":
        return f"IS_STRING({a[0]})"
    if name in ("IS_DECIMAL", "IS_DOUBLE", "IS_REAL"):
        return f"IS_FLOAT({a[0]})"
    if name == "NOW":
        return "NOW()"
    if name in ("SYSDATE", "GETDATE", "SYSTIMESTAMP", "LOCALTIMESTAMP"):
        return "NOW()"
    if name == "SPLIT_TO_TABLE":
        return self.func("UNNEST", exp.Anonymous(this="SPLIT", expressions=args))
    if name == "ARRAY_CONSTRUCT_COMPACT":
        return f"ARRAY_COMPACT(ARRAY_CONSTRUCT({', '.join(a)}))"
    if name == "ZEROIFNULL":
        return f"COALESCE({a[0]}, 0)"
    if name == "TIMEADD" or name == "TIMESTAMPADD":
        return self.func("DATEADD", *args)
    if name == "TIMEDIFF" or name == "TIMESTAMPDIFF":
        return self.func("DATEDIFF", *args)
    if name == "BITOR":
        return f"({a[0]} | {a[1]})"
    if name == "BITXOR":
        return f"({a[0]} ^ {a[1]})"
    if name == "CURRENT_SESSION":
        return "''"
    if name == "OBJECT_CONSTRUCT_KEEP_NULL":
        return self.func("OBJECT_CONSTRUCT_KEEP_NULL", *args)
    if name == "ARRAY_UNION_AGG":
        return f"TO_VARIANT(ARRAY_DISTINCT(ARRAY_FLATTEN(ARRAY_AGG(CAST({a[0]} AS ARRAY(VARIANT))))))"
    if name == "ARRAY_UNIQUE_AGG":
        return f"TO_VARIANT(ARRAY_DISTINCT(ARRAY_AGG({a[0]})))"
    if name in ("TRY_TO_TIMESTAMP_TZ", "TO_TIMESTAMP_TZ"):
        return self.func("TRY_TO_TIMESTAMP" if name.startswith("TRY") else "TO_TIMESTAMP", *args)
    if name == "STRIP_NULL_VALUE":
        return self.func("STRIP_NULL_VALUE", *args)
    return self.func(e.name, *args)


def _window_sql(self, e: exp.Window) -> str:
    fn = e.this
    if isinstance(fn, exp.Anonymous):
        name = fn.name.upper()
        if name == "RATIO_TO_REPORT":
            over = _over_clause(self, e, order=False)
            return f"({self.sql(fn.expressions[0])} / NULLIF(SUM({self.sql(fn.expressions[0])}) OVER ({over}), 0))"
        if name == "CONDITIONAL_TRUE_EVENT":
            over = _over_clause(self, e, frame="ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW")
            return f"SUM(CASE WHEN {self.sql(fn.expressions[0])} THEN 1 ELSE 0 END) OVER ({over})"
        if name == "CONDITIONAL_CHANGE_EVENT":
            over = _over_clause(self, e, frame="ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW")
            col = self.sql(fn.expressions[0])
            inner_over = _over_clause(self, e)
            return (f"SUM(CASE WHEN {col} IS DISTINCT FROM LAG({col}) OVER ({inner_over}) "
                    f"AND LAG({col}) OVER ({inner_over}) IS NOT NULL THEN 1 ELSE 0 END) OVER ({over})")
    return Snowflake.Generator.window_sql(self, e)


def _over_clause(self, e: exp.Window, order: bool = True, frame: Optional[str] = None) -> str:
    parts = []
    if e.args.get("partition_by"):
        parts.append("PARTITION BY " + ", ".join(self.sql(p) for p in e.args["partition_by"]))
    if order and e.args.get("order"):
        parts.append(self.sql(e, "order").strip())
    if frame and order and e.args.get("order"):
        parts.append(frame)
    elif e.args.get("spec") and order:
        parts.append(self.sql(e, "spec"))
    return " ".join(parts)


def _within_group_sql(self, e: exp.WithinGroup) -> str:
    fn = e.this
    order = e.expression
    if isinstance(fn, (exp.PercentileCont, exp.PercentileDisc)) and isinstance(order, exp.Order) and order.expressions:
        ordered = order.expressions[0]
        col = ordered.this if isinstance(ordered, exp.Ordered) else ordered
        desc = isinstance(ordered, exp.Ordered) and ordered.args.get("desc")
        p = fn.this
        p_sql = self.sql(p)
        if desc:
            p_sql = str(1 - float(p.name)) if isinstance(p, exp.Literal) else f"(1 - {p_sql})"
        name = "QUANTILE_CONT" if isinstance(fn, exp.PercentileCont) else "QUANTILE_DISC"
        return f"{name}({p_sql})({self.sql(col)})"
    if isinstance(fn, exp.ArrayAgg):
        # Databend's ARRAY_AGG has no WITHIN GROUP; ordering is applied by
        # aggregating over an ordered argument, which Databend preserves.
        return self.sql(fn)
    return f"{self.sql(fn)} WITHIN GROUP ({self.sql(order).strip()})"


def _group_concat_sql(self, e: exp.GroupConcat) -> str:
    this = e.this
    order = None
    if isinstance(this, exp.Order):
        order = this
        this = this.this
    distinct = ""
    if isinstance(this, exp.Distinct):
        distinct = "DISTINCT "
        this = this.expressions[0]
    sep = self.sql(e, "separator") or "''"
    sql = f"LISTAGG({distinct}{self.sql(this)}, {sep})"
    if order is not None and order.expressions:
        sql += f" WITHIN GROUP (ORDER BY {', '.join(self.sql(o) for o in order.expressions)})"
    return sql


def _convert_timezone_sql(self, e: exp.ConvertTimezone) -> str:
    src = e.args.get("source_tz")
    tgt = e.args.get("target_tz")
    ts = self.sql(e, "timestamp")
    tgt_sql = self.sql(tgt)
    if src is None or (isinstance(src, exp.Literal) and src.name.upper() in ("UTC", "ETC/UTC", "GMT")):
        return f"CONVERT_TIMEZONE({tgt_sql}, {ts})"
    src_sql = self.sql(src)
    # Snowflake 3-arg form: ts is wall-clock time in src. Turn it into the
    # UTC instant (subtract src's UTC offset at that moment), then convert.
    utc = f"DATEADD(SECOND, -DATEDIFF(SECOND, {ts}, CONVERT_TIMEZONE({src_sql}, {ts})), {ts})"
    return f"CONVERT_TIMEZONE({tgt_sql}, {utc})"


_DATE_PART_MAP = {
    "DAYOFWEEKISO": "ISODOW", "WEEKDAY_ISO": "ISODOW", "DOW_ISO": "ISODOW",
    "DAYOFYEAR": "DOY", "YEARDAY": "DOY", "DOY": "DOY",
    "WEEKOFYEAR": "WEEK", "WEEKISO": "WEEK", "WEEK_ISO": "WEEK", "WOY": "WEEK", "WK": "WEEK", "WEEK": "WEEK",
    "YEAROFWEEK": "YEAR", "YEAROFWEEKISO": "YEAR",
    "DAYOFMONTH": "DAY", "DAY": "DAY", "D": "DAY", "DD": "DAY", "DAYS": "DAY",
    "YEAR": "YEAR", "Y": "YEAR", "YY": "YEAR", "YYYY": "YEAR", "YR": "YEAR", "YEARS": "YEAR",
    "MONTH": "MONTH", "MM": "MONTH", "MON": "MONTH", "MONS": "MONTH", "MONTHS": "MONTH",
    "QUARTER": "QUARTER", "Q": "QUARTER", "QTR": "QUARTER", "QTRS": "QUARTER", "QUARTERS": "QUARTER",
    "HOUR": "HOUR", "H": "HOUR", "HH": "HOUR", "HR": "HOUR", "HOURS": "HOUR",
    "MINUTE": "MINUTE", "M": "MINUTE", "MI": "MINUTE", "MIN": "MINUTE", "MINUTES": "MINUTE",
    "SECOND": "SECOND", "S": "SECOND", "SEC": "SECOND", "SECONDS": "SECOND",
    "EPOCH": "EPOCH",
}


def _extract_sql(self, e: exp.Extract) -> str:
    unit = (e.this.name if e.this else "").upper()
    target = self.sql(e, "expression")
    if unit in ("DAYOFWEEK", "WEEKDAY", "DOW", "DW"):
        return f"(TO_DAY_OF_WEEK({target}) % 7)"
    if unit in ("EPOCH_SECOND", "EPOCH_SECONDS"):
        return f"TO_UNIX_TIMESTAMP({target})"
    if unit in ("EPOCH_MILLISECOND", "EPOCH_MILLISECONDS"):
        return f"CAST(EXTRACT(EPOCH FROM {target}) * 1000 AS BIGINT)"
    if unit in ("EPOCH_MICROSECOND", "EPOCH_MICROSECONDS"):
        return f"CAST(EXTRACT(EPOCH FROM {target}) * 1000000 AS BIGINT)"
    if unit in ("NANOSECOND", "NS", "NANOSECONDS"):
        return "0"
    mapped = _DATE_PART_MAP.get(unit, unit)
    return f"EXTRACT({mapped} FROM {target})"


_DATEADD_UNITS = {
    "Y": "YEAR", "YY": "YEAR", "YYY": "YEAR", "YYYY": "YEAR", "YR": "YEAR", "YEARS": "YEAR", "YRS": "YEAR",
    "MM": "MONTH", "MON": "MONTH", "MONS": "MONTH", "MONTHS": "MONTH",
    "D": "DAY", "DD": "DAY", "DAYS": "DAY", "DAYOFMONTH": "DAY",
    "W": "WEEK", "WK": "WEEK", "WEEKOFYEAR": "WEEK", "WOY": "WEEK", "WY": "WEEK", "WEEKS": "WEEK",
    "Q": "QUARTER", "QTR": "QUARTER", "QTRS": "QUARTER", "QUARTERS": "QUARTER",
    "H": "HOUR", "HH": "HOUR", "HR": "HOUR", "HOURS": "HOUR", "HRS": "HOUR",
    "M": "MINUTE", "MI": "MINUTE", "MIN": "MINUTE", "MINUTES": "MINUTE", "MINS": "MINUTE",
    "S": "SECOND", "SEC": "SECOND", "SECONDS": "SECOND", "SECS": "SECOND",
    "MS": "MILLISECOND", "MSEC": "MILLISECOND", "MILLISECONDS": "MILLISECOND",
    "US": "MICROSECOND", "USEC": "MICROSECOND", "MICROSECONDS": "MICROSECOND",
}


def _unit(e) -> str:
    u = e.args.get("unit")
    name = (u.name if u is not None else "DAY").upper().strip("'\"")
    return _DATEADD_UNITS.get(name, name)


def _dateadd_sql(self, e) -> str:
    unit = _unit(e)
    amount = self.sql(e, "expression")
    target = self.sql(e, "this")
    if unit in ("MILLISECOND", "MICROSECOND", "NANOSECOND"):
        # Databend's DATEADD stops at SECOND; sub-second steps go through
        # interval arithmetic.
        factor = {"MILLISECOND": "1000", "MICROSECOND": "1", "NANOSECOND": "0.001"}[unit]
        return (f"(CAST({target} AS TIMESTAMP) + CAST(CONCAT(CAST(CAST(({amount}) * {factor} AS BIGINT) AS VARCHAR),"
                f" ' microseconds') AS INTERVAL))")
    return f"DATEADD({unit}, {amount}, {target})"


def _datediff_sql(self, e) -> str:
    # Snowflake DATEDIFF counts unit *boundaries* crossed (DATEDIFF(day,
    # '01 23:00', '02 01:00') = 1).  Databend counts the same way for
    # calendar units; verified in test_translate_live.
    unit = _unit(e)
    return f"DATEDIFF({unit}, {self.sql(e, 'expression')}, {self.sql(e, 'this')})"


def _trunc_sql(self, e) -> str:
    unit = (e.args.get("unit").name if e.args.get("unit") is not None else "DAY").upper().strip("'\"")
    unit = _DATEADD_UNITS.get(unit, unit)
    return f"DATE_TRUNC({unit}, {self.sql(e, 'this')})"


def _date_from_parts_sql(self, e: exp.DateFromParts) -> str:
    y, m, d = (self.sql(e, k) for k in ("year", "month", "day"))
    base = f"TO_DATE(CONCAT(CAST({y} AS VARCHAR), '-01-01'))"
    return f"DATEADD(DAY, ({d}) - 1, DATEADD(MONTH, ({m}) - 1, {base}))"


def _timestamp_from_parts_sql(self, e: exp.TimestampFromParts) -> str:
    y, mo, d = (self.sql(e, k) for k in ("year", "month", "day"))
    h = self.sql(e, "hour") or "0"
    mi = self.sql(e, "min") or "0"
    s = self.sql(e, "sec") or "0"
    base = f"TO_TIMESTAMP(CONCAT(CAST({y} AS VARCHAR), '-01-01 00:00:00'))"
    out = f"DATEADD(MONTH, ({mo}) - 1, {base})"
    out = f"DATEADD(DAY, ({d}) - 1, {out})"
    out = f"DATEADD(HOUR, {h}, {out})"
    out = f"DATEADD(MINUTE, {mi}, {out})"
    return f"DATEADD(SECOND, {s}, {out})"


_WEEKDAYS = {"MO": "MONDAY", "TU": "TUESDAY", "WE": "WEDNESDAY", "TH": "THURSDAY",
             "FR": "FRIDAY", "SA": "SATURDAY", "SU": "SUNDAY"}


def _next_day_sql(self, e) -> str:
    day = e.expression
    if isinstance(day, exp.Literal):
        d = _WEEKDAYS.get(day.name.strip().upper()[:2], day.name.upper())
        return f"NEXT_DAY({self.sql(e, 'this')}, {d})"
    return self.func("NEXT_DAY", e.this, e.expression)


def _to_number_sql(self, e: exp.ToNumber) -> str:
    precision = e.args.get("precision")
    scale = e.args.get("scale")
    p = self.sql(precision) if precision is not None else "38"
    s = self.sql(scale) if scale is not None else "0"
    fn = "TRY_CAST" if e.args.get("safe") else "CAST"
    if e.args.get("format") is not None and precision is None and isinstance(e.args.get("format"), exp.Literal) \
            and e.args["format"].name.isdigit():
        # TO_NUMBER(x, 10, 2) parses as (format=10, precision=2)
        p, s = e.args["format"].name, self.sql(precision) if precision is not None else "0"
    return f"{fn}({self.sql(e, 'this')} AS DECIMAL({p}, {s}))"


def _ts_or_ds_to_date_sql(self, e: exp.TsOrDsToDate) -> str:
    fmt = e.args.get("format")
    name = "TRY_TO_DATE" if e.args.get("safe") else "TO_DATE"
    if fmt is not None:
        return self.func(name, e.this, exp.Literal.string(_strftime(self, fmt)))
    return self.func(name, e.this)


def _str_to_time_sql(self, e: exp.StrToTime) -> str:
    name = "TRY_TO_TIMESTAMP" if e.args.get("safe") else "TO_TIMESTAMP"
    return self.func(name, e.this, exp.Literal.string(_strftime(self, e.args["format"])))


def _strftime(self, fmt: exp.Expression) -> str:
    # sqlglot normalised the Snowflake format into Python strftime when it
    # parsed it; Databend accepts strftime directly (alongside Oracle style).
    return fmt.name if isinstance(fmt, exp.Literal) else self.sql(fmt)


def _time_to_str_sql(self, e: exp.TimeToStr) -> str:
    return self.func("TO_CHAR", e.this, exp.Literal.string(_strftime(self, e.args["format"])))


def _to_char_sql(self, e: exp.ToChar) -> str:
    fmt = e.args.get("format")
    if fmt is None:
        return f"CAST({self.sql(e, 'this')} AS VARCHAR)"
    if isinstance(fmt, exp.Literal) and '"' in fmt.name:
        # Snowflake: "..." inside a format is literal text.  Databend copies
        # the quotes through, so render the pieces separately.
        x = self.sql(e, "this")
        parts = []
        for i, piece in enumerate(fmt.name.split('"')):
            if not piece:
                continue
            lit = "'" + piece.replace("'", "''") + "'"
            parts.append(lit if i % 2 else f"TO_CHAR({x}, {lit})")
        return parts[0] if len(parts) == 1 else f"CONCAT({', '.join(parts)})"
    return self.func("TO_CHAR", e.this, fmt)


def _regexp_like_sql(self, e: exp.RegexpLike) -> str:
    # Snowflake anchors REGEXP_LIKE/RLIKE patterns at both ends; Databend
    # searches.  Anchor explicitly.
    pattern = e.expression
    if isinstance(pattern, exp.Literal):
        anchored = exp.Literal.string(f"^(?:{pattern.name})$")
    else:
        anchored = exp.func("CONCAT", exp.Literal.string("^(?:"), pattern, exp.Literal.string(")$"))
    flag = e.args.get("flag")
    if flag is not None and isinstance(flag, exp.Literal) and "i" in flag.name:
        return self.func("REGEXP_LIKE", e.this, anchored, flag)
    return self.func("REGEXP_LIKE", e.this, anchored)


def _array_contains_sql(self, e) -> str:
    # Snowflake ARRAY_CONTAINS(value, array); Databend's variant overload
    # is CONTAINS(array, value) with the value cast to VARIANT.
    value, array = e.expression, e.this
    if isinstance(e, exp.ArrayContains):
        value, array = e.expression, e.this
    return f"ARRAY_CONTAINS(CAST({self.sql(array)} AS VARIANT), CAST({self.sql(value)} AS VARIANT))"


def _array_to_string_sql(self, e) -> str:
    return f"ARRAY_TO_STRING(CAST({self.sql(e, 'this')} AS ARRAY(STRING NULL)), {self.sql(e, 'expression')})"


def _split_sql(self, e: exp.Split) -> str:
    # Snowflake SPLIT returns a 0-indexed (variant) ARRAY; Databend's SPLIT
    # returns a 1-indexed native array.  Convert so [0] still means "first".
    return f"CAST(SPLIT({self.sql(e, 'this')}, {self.sql(e, 'expression')}) AS VARIANT)"


def _initcap_sql(self, e) -> str:
    x = self.sql(e, "this")
    return (f"ARRAY_TO_STRING(ARRAY_TRANSFORM(SPLIT(LOWER({x}), ' '), w -> "
            f"CONCAT(UPPER(SUBSTR(w, 1, 1)), SUBSTR(w, 2))), ' ')")


def _corr_sql(self, e) -> str:
    a, b = self.sql(e, "this"), self.sql(e, "expression")
    return (f"((AVG({a} * {b}) - AVG({a}) * AVG({b})) / "
            f"NULLIF(STDDEV_POP({a}) * STDDEV_POP({b}), 0))")


def _regexp_count_sql(self, e) -> str:
    return f"ARRAY_LENGTH(REGEXP_EXTRACT_ALL({self.sql(e, 'this')}, {self.sql(e, 'expression')}))"


def _starts_with_sql(self, e) -> str:
    a, b = self.sql(e, "this"), self.sql(e, "expression")
    return f"(LEFT({a}, LENGTH({b})) = {b})"


def _ends_with_sql(self, e) -> str:
    a, b = self.sql(e, "this"), self.sql(e, "expression")
    return f"(RIGHT({a}, LENGTH({b})) = {b})"


def _str_position_sql(self, e: exp.StrPosition) -> str:
    sub = self.sql(e, "substr")
    s = self.sql(e, "this")
    pos = e.args.get("position")
    return f"LOCATE({sub}, {s}, {self.sql(pos)})" if pos is not None else f"LOCATE({sub}, {s})"


def _contains_sql(self, e) -> str:
    return f"(POSITION({self.sql(e, 'expression')} IN {self.sql(e, 'this')}) > 0)"


def _sha2_sql(self, e) -> str:
    length = e.args.get("length")
    return f"SHA2({self.sql(e, 'this')}, {self.sql(length) if length is not None else '256'})"


def _get_extract_sql(self, e) -> str:
    return f"GET({self.sql(e, 'this')}, {self.sql(e, 'expression')})"


def _datatype_sql(self, e: exp.DataType) -> str:
    t = e.this
    T = exp.DataType.Type
    if t in (T.TIMESTAMPNTZ, T.TIMESTAMPLTZ, T.TIMESTAMPTZ, T.TIMESTAMP, T.DATETIME, T.TIMESTAMP_S,
             T.TIMESTAMP_MS, T.TIMESTAMP_NS):
        return "TIMESTAMP"
    if t in (T.TIME, T.TIMETZ):
        return "VARCHAR"
    if t in (T.OBJECT, T.STRUCT, T.MAP, T.VARIANT, T.JSON):
        return "VARIANT"
    if t == T.ARRAY and not e.expressions:
        return "VARIANT"
    if t in (T.VARCHAR, T.NVARCHAR, T.CHAR, T.NCHAR, T.TEXT, T.NAME):
        return "VARCHAR"
    if t in (T.INT, T.SMALLINT, T.TINYINT, T.MEDIUMINT, T.UTINYINT, T.USMALLINT, T.UINT):
        return "BIGINT"
    if t == T.FLOAT:
        return "DOUBLE"
    if t in (T.DOUBLE,):
        return "DOUBLE"
    if t in (T.VARBINARY, T.BINARY, T.BLOB):
        return "BINARY"
    if t in (T.DECIMAL, T.BIGDECIMAL):
        if not e.expressions:
            return "DECIMAL(38, 0)"
    if t == T.GEOGRAPHY:
        return "VARCHAR"
    return Snowflake.Generator.datatype_sql(self, e)


_TRUE_WORDS = "('true', 't', 'yes', 'y', 'on', '1')"
_FALSE_WORDS = "('false', 'f', 'no', 'n', 'off', '0')"


def _to_boolean_sql(self, e) -> str:
    # Snowflake accepts yes/no/on/off/y/n/t/f; Databend's cast only true/false/1/0.
    x = f"LOWER(TRIM(CAST({self.sql(e, 'this')} AS VARCHAR)))"
    return (f"(CASE WHEN {x} IN {_TRUE_WORDS} THEN TRUE WHEN {x} IN {_FALSE_WORDS} THEN FALSE "
            f"ELSE NULL END)")


def _array_union_agg_sql(self, e) -> str:
    x = self.sql(e, "this")
    return f"TO_VARIANT(ARRAY_DISTINCT(ARRAY_FLATTEN(ARRAY_AGG(CAST({x} AS ARRAY(VARIANT))))))"


def _lateral_sql(self, e: exp.Lateral) -> str:
    this = e.this
    alias = e.args.get("alias")
    alias_sql = ""
    if alias is not None and alias.this is not None:
        alias_sql = f" AS {self.sql(alias.this)}"
    if isinstance(this, exp.Explode):
        return f"LATERAL {_flatten_call(self, this)}{alias_sql}"
    return f"LATERAL {self.sql(this)}{alias_sql}"


def _flatten_call(self, ex: exp.Explode) -> str:
    args = [ex.this] + list(ex.expressions or [])
    parts = []
    for a in args:
        if isinstance(a, exp.Kwarg):
            key = a.this.name.lower()
            parts.append(f"{key} => {self.sql(a.expression)}")
        else:
            parts.append(f"input => {self.sql(a)}")
    return f"FLATTEN({', '.join(parts)})"


def _table_from_rows_sql(self, e) -> str:
    inner = e.this
    alias = e.args.get("alias")
    alias_sql = f" AS {self.sql(alias.this)}" if alias is not None and alias.this is not None else ""
    if isinstance(inner, exp.Explode):
        return f"{_flatten_call(self, inner)}{alias_sql}"
    if type(inner).__name__ == "Generator":
        # TABLE(GENERATOR(ROWCOUNT => n)) -> NUMBERS(n) (column "number";
        # SEQ4()/SEQ8() map to it).
        rc = inner.args.get("rowcount")
        return f"NUMBERS({self.sql(rc) if rc is not None else '0'}){alias_sql}"
    if isinstance(inner, exp.Anonymous) and inner.name.upper() == "GENERATOR":
        rowcount = next((self.sql(k.expression) for k in inner.expressions
                         if isinstance(k, exp.Kwarg) and k.this.name.upper() == "ROWCOUNT"), "0")
        return f"NUMBERS({rowcount}){alias_sql}"
    return Snowflake.Generator.tablefromrows_sql(self, e) if hasattr(Snowflake.Generator, "tablefromrows_sql") \
        else f"TABLE({self.sql(inner)}){alias_sql}"


def _column_def_sql(self, e: exp.ColumnDef) -> str:
    keep = []
    for c in e.args.get("constraints") or []:
        kind = c.args.get("kind")
        if isinstance(kind, (exp.PrimaryKeyColumnConstraint, exp.UniqueColumnConstraint,
                             exp.AutoIncrementColumnConstraint, exp.GeneratedAsIdentityColumnConstraint,
                             exp.CollateColumnConstraint, exp.Reference)):
            continue
        if type(kind).__name__ in ("IdentityColumnConstraint", "ForeignKey", "TagColumnConstraint",
                                    "MaskingPolicyColumnConstraint", "ProjectionPolicyColumnConstraint"):
            continue
        keep.append(c)
    e = e.copy()
    e.set("constraints", keep)
    return Snowflake.Generator.columndef_sql(self, e)


def _schema_sql(self, e: exp.Schema) -> str:
    e = e.copy()
    e.set("expressions", [x for x in e.expressions
                          if not isinstance(x, (exp.PrimaryKey, exp.UniqueColumnConstraint, exp.ForeignKey,
                                                exp.Constraint))])
    return Snowflake.Generator.schema_sql(self, e)


class Databend(Snowflake):
    """Snowflake syntax in, Databend syntax out."""

    class Generator(Snowflake.Generator):
        TRANSFORMS = {
            **Snowflake.Generator.TRANSFORMS,
            exp.ILike: _ilike_sql,
            exp.Anonymous: _anonymous_sql,
            exp.Window: _window_sql,
            exp.WithinGroup: _within_group_sql,
            exp.GroupConcat: _group_concat_sql,
            exp.ConvertTimezone: _convert_timezone_sql,
            exp.Extract: _extract_sql,
            exp.DateAdd: _dateadd_sql,
            exp.TsOrDsAdd: _dateadd_sql,
            exp.TimestampAdd: _dateadd_sql,
            exp.DateDiff: _datediff_sql,
            exp.TimestampDiff: _datediff_sql,
            exp.TimestampTrunc: _trunc_sql,
            exp.DateTrunc: _trunc_sql,
            exp.DateFromParts: _date_from_parts_sql,
            exp.TimestampFromParts: _timestamp_from_parts_sql,
            exp.NextDay: _next_day_sql,
            exp.ToNumber: _to_number_sql,
            exp.TsOrDsToDate: _ts_or_ds_to_date_sql,
            exp.StrToTime: _str_to_time_sql,
            exp.TimeToStr: _time_to_str_sql,
            exp.ToChar: _to_char_sql,
            exp.RegexpLike: _regexp_like_sql,
            exp.ArrayContains: _array_contains_sql,
            exp.ArrayToString: _array_to_string_sql,
            exp.Split: _split_sql,
            exp.Initcap: _initcap_sql,
            exp.Corr: _corr_sql,
            exp.RegexpCount: _regexp_count_sql,
            exp.StartsWith: _starts_with_sql,
            exp.EndsWith: _ends_with_sql,
            exp.StrPosition: _str_position_sql,
            exp.Contains: _contains_sql,
            exp.SHA2: _sha2_sql,
            exp.GetExtract: _get_extract_sql,
            exp.DataType: _datatype_sql,
            exp.Lateral: _lateral_sql,
            exp.ColumnDef: _column_def_sql,
            exp.Schema: _schema_sql,
            exp.Dayname: lambda self, e: f"TO_CHAR({self.sql(e, 'this')}, 'DY')",
            exp.Monthname: lambda self, e: f"TO_CHAR({self.sql(e, 'this')}, 'Mon')",
            exp.Hour: lambda self, e: f"TO_HOUR({self.sql(e, 'this')})",
            exp.Minute: lambda self, e: f"TO_MINUTE({self.sql(e, 'this')})",
            exp.Second: lambda self, e: f"TO_SECOND({self.sql(e, 'this')})",
            exp.DayOfWeek: lambda self, e: f"(TO_DAY_OF_WEEK({self.sql(e, 'this')}) % 7)",
            exp.DayOfWeekIso: lambda self, e: f"TO_DAY_OF_WEEK({self.sql(e, 'this')})",
            exp.WeekOfYear: lambda self, e: f"WEEKOFYEAR({self.sql(e, 'this')})",
            exp.Week: lambda self, e: f"WEEKOFYEAR({self.sql(e, 'this')})",
            exp.YearOfWeek: lambda self, e: f"YEAR({self.sql(e, 'this')})",
            exp.ToDouble: lambda self, e: f"{'TRY_TO_FLOAT64' if e.args.get('safe') else 'TO_FLOAT64'}({self.sql(e, 'this')})",
            exp.Rand: lambda self, e: "RAND()",
            exp.Pow: lambda self, e: f"POW({self.sql(e, 'this')}, {self.sql(e, 'expression')})",
            exp.CurrentTimestamp: lambda self, e: "NOW()",
            exp.CurrentSchema: lambda self, e: "CURRENT_DATABASE()",
            exp.CurrentAccount: lambda self, e: "'DATABEND'",
            exp.CurrentWarehouse: lambda self, e: "'DATABEND'",
            exp.CurrentVersion: lambda self, e: "VERSION()",
            exp.LogicalAnd: lambda self, e: f"BOOL_AND({self.sql(e, 'this')})",
            exp.LogicalOr: lambda self, e: f"BOOL_OR({self.sql(e, 'this')})",
            exp.Hll: lambda self, e: f"APPROX_COUNT_DISTINCT({self.sql(e, 'this')})",
            # Databend has no 1-argument VARIANCE/VAR_SAMP/VAR_POP.
            exp.Variance: lambda self, e: f"POW(STDDEV_SAMP({self.sql(e, 'this')}), 2)",
            exp.VariancePop: lambda self, e: f"POW(STDDEV_POP({self.sql(e, 'this')}), 2)",
            exp.ArgMin: lambda self, e: f"ARG_MIN({self.sql(e, 'this')}, {self.sql(e, 'expression')})",
            exp.ArgMax: lambda self, e: f"ARG_MAX({self.sql(e, 'this')}, {self.sql(e, 'expression')})",
            exp.MD5: lambda self, e: f"MD5({self.sql(e, 'this')})",
            exp.SHA: lambda self, e: f"SHA1({self.sql(e, 'this')})",
            exp.Uuid: lambda self, e: "UUID()",
            exp.Strtok: lambda self, e: (f"SPLIT_PART({self.sql(e, 'this')}, "
                                         f"{self.sql(e, 'delimiter') or chr(39) + ' ' + chr(39)}, "
                                         f"{self.sql(e, 'part_index') or '1'})"),
            exp.StrtokToArray: lambda self, e: (f"CAST(SPLIT({self.sql(e, 'this')}, "
                                                f"{self.sql(e, 'expression') or chr(39) + ' ' + chr(39)}) AS VARIANT)"),
            exp.ArrayConcat: lambda self, e: self.func("ARRAY_CONCAT", e.this, *e.expressions),
            exp.JSONFormat: lambda self, e: f"TO_STRING(CAST({self.sql(e, 'this')} AS VARIANT))",
            exp.Collate: lambda self, e: self.sql(e, "this"),
            exp.BitwiseAnd: lambda self, e: f"({self.sql(e, 'this')} & {self.sql(e, 'expression')})",
            exp.Seq4: lambda self, e: "number",
            exp.TsOrDsToTime: lambda self, e: f"CAST({self.sql(e, 'this')} AS VARCHAR)",
            exp.TableFromRows: _table_from_rows_sql,
            exp.Array: lambda self, e: self.func("ARRAY_CONSTRUCT", *e.expressions),
            exp.ToBoolean: _to_boolean_sql,
            **({exp.ArrayUnionAgg: _array_union_agg_sql} if hasattr(exp, "ArrayUnionAgg") else {}),
        }
        # Nothing the base class adds may undo the quoting we rely on.
        TYPE_MAPPING = {**Snowflake.Generator.TYPE_MAPPING}


# ---------------------------------------------------------------------------
# Tree rewrites
# ---------------------------------------------------------------------------

def _ident(name: str) -> exp.Identifier:
    return exp.Identifier(this=name, quoted=True)


def _flatten_aliases(tree: exp.Expression) -> set[str]:
    aliases = set()
    for node in tree.find_all(exp.Lateral, exp.TableFromRows):
        alias = node.args.get("alias")
        if alias is not None and alias.this is not None:
            aliases.add(alias.this.name.upper())
        if isinstance(node, exp.Lateral) and alias is not None:
            alias.set("columns", None)
    return aliases


def _protect_flatten_columns(tree: exp.Expression) -> None:
    """Keep FLATTEN's output columns lower-case (Databend's spelling)."""
    # Databend's NUMBERS(n) table function (GENERATOR's stand-in) names its
    # column lower-case "number" too.
    if any(isinstance(t.this, exp.Anonymous) and t.this.name.upper() == "NUMBERS" for t in tree.find_all(exp.Table)) \
            or any(isinstance(t, exp.TableFromRows) for t in tree.find_all(exp.TableFromRows)):
        for col in tree.find_all(exp.Column):
            if col.name.upper() == "NUMBER" and not col.this.args.get("quoted"):
                col.set("this", exp.Identifier(this="number", quoted=True))
    has_flatten = any(True for _ in tree.find_all(exp.Explode))
    if not has_flatten:
        return
    aliases = _flatten_aliases(tree)
    for col in tree.find_all(exp.Column):
        name = col.name
        if name.upper() not in FLATTEN_COLUMNS or col.this.args.get("quoted"):
            continue
        table = col.table
        if (table and table.upper() in aliases) or not table:
            col.set("this", exp.Identifier(this=name.lower(), quoted=True))


def _map_tables(tree: exp.Expression, ctx: Context) -> None:
    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            continue
        catalog = table.args.get("catalog")
        db = table.args.get("db")
        cat_name = catalog.name if catalog is not None else None
        db_name = db.name if db is not None else None
        name = table.this.name
        if db_name is None and name.upper() in ("TABLES", "COLUMNS", "SCHEMATA", "VIEWS") \
                and False:
            continue
        if db_name and db_name.upper() == "INFORMATION_SCHEMA":
            table.set("catalog", None)
            table.set("db", _ident(COMPAT_DB))
            table.set("this", _ident(name.upper()))
            continue
        if db_name is None:
            continue  # resolves against the session's current database
        target = map_database(cat_name or ctx.database, db_name, ctx.primary_database)
        table.set("catalog", None)
        table.set("db", exp.Identifier(this=target, quoted=db.args.get("quoted", False)
                                       if target == db_name else True))
    # Columns qualified by db.schema.table.col: drop the catalog part.
    for col in tree.find_all(exp.Column):
        if col.args.get("catalog") is not None:
            col.set("catalog", None)


_VALUES_COL_RE = re.compile(r"^COLUMN(\d+)$", re.I)


def _rewrite_values_columns(tree: exp.Expression) -> None:
    """Snowflake names VALUES columns COLUMN1..n; Databend col0..n-1."""
    if not any(True for _ in tree.find_all(exp.Values)):
        return
    for col in tree.find_all(exp.Column):
        ident = col.this
        if not isinstance(ident, exp.Identifier) or ident.args.get("quoted"):
            continue
        m = _VALUES_COL_RE.match(ident.name)
        if m:
            col.set("this", exp.Identifier(this=f"col{int(m.group(1)) - 1}", quoted=True))


def _rewrite_union_ordinal_order(tree: exp.Expression) -> exp.Expression:
    """``a UNION b ORDER BY 1`` -> ``SELECT * FROM (a UNION b) ORDER BY 1``.

    Databend can only order a set operation by column name.
    """
    if not isinstance(tree, exp.SetOperation):
        return tree
    order = tree.args.get("order")
    if order is None or not any(isinstance(o.this, exp.Literal) for o in order.expressions):
        return tree
    limit, offset = tree.args.get("limit"), tree.args.get("offset")
    tree.set("order", None)
    tree.set("limit", None)
    tree.set("offset", None)
    outer = exp.select("*").from_(exp.Subquery(this=tree, alias=exp.TableAlias(this=exp.to_identifier("_afya_u"))))
    outer.set("order", order)
    if limit is not None:
        outer.set("limit", limit)
    if offset is not None:
        outer.set("offset", offset)
    return outer


def _rename_target_alias(tree: exp.Expression) -> None:
    """UPDATE/DELETE <table> t ...: same databend-query quirk as MERGE (a
    quoted target alias is lower-cased internally, then its columns -
    including unqualified SET targets - can't be resolved: "Can not find
    column X in table t").  Rename the alias to lower-case afya_tgt and
    rewrite every reference."""
    if not isinstance(tree, (exp.Update, exp.Delete)):
        return
    target = tree.this
    alias = target.args.get("alias") if isinstance(target, exp.Table) else None
    if alias is None or alias.this is None:
        return
    old = alias.this.name
    alias.set("this", exp.Identifier(this="afya_tgt", quoted=True))
    for col in tree.find_all(exp.Column):
        tbl = col.args.get("table")
        if tbl is not None and (tbl.name == old or (not tbl.args.get("quoted") and tbl.name.upper() == old.upper())):
            col.set("table", exp.Identifier(this="afya_tgt", quoted=True))
    if isinstance(tree, exp.Update):
        for eq in tree.expressions:
            if isinstance(eq, exp.EQ) and isinstance(eq.this, exp.Column) and eq.this.table:
                eq.set("this", exp.Column(this=eq.this.this))


def _rewrite_merge_set(tree: exp.Expression) -> None:
    """MERGE ... UPDATE SET t.col = ... -> SET col = ... (Databend rejects
    a qualifier on the assignment target; Snowflake accepts both)."""
    _rename_target_alias(tree)
    if not isinstance(tree, exp.Merge):
        return
    # databend-query 1.2.881 lower-cases a quoted MERGE target alias
    # internally ("T" -> t) and then can't resolve UPDATE SET columns
    # ("Can not find column NAME in table t").  A lower-case alias works,
    # so rename the target alias and every reference to it.
    target = tree.this
    alias = target.args.get("alias") if isinstance(target, exp.Table) else None
    if alias is not None and alias.this is not None:
        old = alias.this.name.upper()
        alias.set("this", exp.Identifier(this="afya_tgt", quoted=True))
        for col in tree.find_all(exp.Column):
            if col.table and col.table.upper() == old and not col.args["table"].args.get("quoted"):
                col.set("table", exp.Identifier(this="afya_tgt", quoted=True))
            elif col.table and col.table == old:
                col.set("table", exp.Identifier(this="afya_tgt", quoted=True))
    for upd in tree.find_all(exp.Update):
        for eq in upd.expressions:
            if isinstance(eq, exp.EQ) and isinstance(eq.this, exp.Column) and eq.this.table:
                eq.set("this", exp.Column(this=eq.this.this))
    for when in tree.find_all(exp.When):
        then = when.args.get("then")
        if isinstance(then, exp.Update):
            continue
        for eq in (then.expressions if then is not None and hasattr(then, "expressions") else []):
            if isinstance(eq, exp.EQ) and isinstance(eq.this, exp.Column) and eq.this.table:
                eq.set("this", exp.Column(this=eq.this.this))


def _rewrite_delete_using(tree: exp.Expression) -> exp.Expression:
    if not isinstance(tree, exp.Delete) or not tree.args.get("using"):
        return tree
    using = tree.args["using"]
    where = tree.args.get("where")
    sub = exp.select("1").from_(using[0])
    for extra in using[1:]:
        sub = sub.join(extra, join_type="cross")
    if where is not None:
        sub = sub.where(where.this)
    tree.set("using", None)
    tree.set("where", exp.Where(this=exp.Exists(this=sub)))
    return tree


def _strip_unsupported_properties(tree: exp.Expression) -> None:
    props = tree.args.get("properties") if isinstance(tree, exp.Create) else None
    if props is None:
        return
    keep = []
    for p in props.expressions:
        cls = type(p).__name__
        if cls in ("SecureProperty", "CopyGrantsProperty", "DataRetentionTimeProperty",
                   "ChangeTrackingProperty", "RowAccessProperty", "TagsProperty", "StabilityProperty",
                   "MaterializedProperty"):
            continue
        if isinstance(p, exp.Property) and str(p.name).upper() in (
                "DATA_RETENTION_TIME_IN_DAYS", "MAX_DATA_EXTENSION_TIME_IN_DAYS", "CHANGE_TRACKING",
                "DEFAULT_DDL_COLLATION", "ENABLE_SCHEMA_EVOLUTION", "STAGE_FILE_FORMAT", "STAGE_COPY_OPTIONS"):
            continue
        keep.append(p)
    props.set("expressions", keep)
    if not keep:
        tree.set("properties", None)


def _schema_ref(t: exp.Expression) -> tuple[Optional[str], str]:
    """(database, schema) of a schema reference like ``HOSPITALS.RAW``.

    sqlglot stores a schema name in Table.db (database in Table.catalog)
    for CREATE/DROP SCHEMA, but in Table.this for USE SCHEMA; take the last
    two name parts whichever slots they're in.
    """
    if isinstance(t, exp.Identifier):
        return None, _ident_name(t)
    parts = [t.args.get(k) for k in ("catalog", "db", "this")]
    names = [_ident_name(p) for p in parts if p is not None and _ident_name(p)]
    if not names:
        return None, ""
    return (names[-2] if len(names) > 1 else None), names[-1]


def _ident_name(i) -> str:
    if isinstance(i, exp.Identifier):
        return i.name if i.args.get("quoted") else i.name.upper()
    return getattr(i, "name", str(i)) or ""


def _rewrite_create(tree: exp.Create, ctx: Context) -> list[Statement]:
    kind = (tree.args.get("kind") or "").upper()
    if kind == "SCHEMA":
        sf_db, schema = _schema_ref(tree.this)
        target = map_database(sf_db or ctx.database, schema, ctx.primary_database)
        exists = " IF NOT EXISTS" if tree.args.get("exists") else ""
        if tree.args.get("replace"):
            return [Statement("create_schema", f'CREATE OR REPLACE DATABASE "{target}"', {"schema": schema, "database": sf_db})]
        return [Statement("create_schema", f'CREATE DATABASE{exists} "{target}"', {"schema": schema, "database": sf_db})]
    if kind == "DATABASE":
        return [Statement("use_database", "", {"database": tree.this.name, "create": True})]
    if kind in ("STAGE",):
        name = _stage_name(tree.this.sql())
        return [Statement("sql", f'CREATE {"OR REPLACE " if tree.args.get("replace") else ""}STAGE '
                                 f'{"IF NOT EXISTS " if tree.args.get("exists") else ""}{name}')]
    if kind in ("FUNCTION", "PROCEDURE", "TASK", "STREAM", "PIPE", "SEQUENCE", "ALERT", "DYNAMIC TABLE",
                "FILE FORMAT", "TAG", "ROLE", "USER", "WAREHOUSE"):
        if kind == "SEQUENCE":
            return [Statement("sql", f'CREATE SEQUENCE IF NOT EXISTS "{tree.this.name.upper()}"')]
        logger.warning("Databend: skipping Snowflake-only CREATE %s", kind)
        return [Statement("noop", original=tree.sql("snowflake"))]
    clone = tree.args.get("clone")
    if clone is not None:
        src = clone.this
        tree.set("clone", None)
        tree.set("expression", exp.select("*").from_(src.copy()))
    _strip_unsupported_properties(tree)
    if kind == "VIEW":
        props = tree.args.get("properties")
        if props is not None:
            props.set("expressions", [p for p in props.expressions
                                      if type(p).__name__ not in ("SecureProperty", "MaterializedProperty")])
    return []


def _stage_name(raw: str) -> str:
    """Snowflake stage reference -> Databend stage name."""
    raw = raw.strip().lstrip("@").strip('"')
    if raw.startswith("~"):
        return "~"
    if raw.startswith("%"):
        raw = "tblstage_" + raw[1:].split("/")[0].replace('"', "")
    raw = raw.split("/")[0]
    raw = raw.split(".")[-1]
    return re.sub(r"[^A-Za-z0-9_]", "_", raw).lower()


def _stage_ref(raw: str) -> str:
    raw = raw.strip()
    path = ""
    body = raw.lstrip("@")
    if "/" in body:
        body, path = body.split("/", 1)
        path = "/" + path
    return "@" + _stage_name("@" + body) + path


def _rewrite_alter(tree: exp.Alter, ctx: Context) -> Optional[list[Statement]]:
    kind = (tree.args.get("kind") or "").upper()
    if kind in ("SESSION", "WAREHOUSE", "USER", "ACCOUNT", "ROLE"):
        return [Statement("noop", original=tree.sql("snowflake"))]
    actions = tree.args.get("actions") or []
    for a in actions:
        cls = type(a).__name__
        if cls == "SwapTable":
            return [Statement("swap", "", {"a": tree.this.sql(dialect=Databend, identify=True),
                                            "b": a.this.sql(dialect=Databend, identify=True)})]
        if cls == "AlterRename":
            # Databend: ALTER TABLE db.t RENAME TO t2 (new name unqualified,
            # stays in db) - cross-database renames aren't supported.
            new = a.this
            if isinstance(new, exp.Table):
                new.set("db", None)
                new.set("catalog", None)
        if cls == "AlterSet" or cls == "AlterUnset":
            return [Statement("noop", original=tree.sql("snowflake"))]
    return None


def _rewrite_use(tree: exp.Use, ctx: Context) -> list[Statement]:
    kind = (tree.args.get("kind").name.upper() if tree.args.get("kind") is not None else "")
    t = tree.this
    if kind in ("WAREHOUSE", "ROLE", "SECONDARY ROLES"):
        return [Statement("noop", original=tree.sql("snowflake"))]
    if kind == "SCHEMA":
        sf_db, schema = _schema_ref(t)
        return [Statement("use_schema", "", {"schema": schema, "database": sf_db})]
    # USE DATABASE x / USE x (Snowflake: database, or db.schema)
    sf_db, name = _schema_ref(t)
    if sf_db is not None and kind != "DATABASE":
        return [Statement("use_schema", "", {"schema": name, "database": sf_db})]
    return [Statement("use_database", "", {"database": name})]


def _rewrite_show(tree: exp.Show, ctx: Context) -> list[Statement]:
    what = (tree.name or "").upper()
    like = tree.args.get("like")
    scope = tree.args.get("scope")
    scope_kind = (tree.args.get("scope_kind") or "").upper()
    target_db = None
    if scope is not None and isinstance(scope, exp.Table) and what != "COLUMNS":
        sf_db, name = _schema_ref(scope)
        if scope_kind == "DATABASE":
            target_db = None  # whole (primary) database
        else:
            target_db = map_database(sf_db or ctx.database, name, ctx.primary_database)
    elif scope is None and what in ("TABLES", "VIEWS", "OBJECTS") and ctx.schema:
        target_db = map_database(ctx.database, ctx.schema, ctx.primary_database)
    # Snowflake's SHOW ... LIKE is case-insensitive.
    like_sql = f" AND LOWER(name) LIKE LOWER({like.sql()})" if like is not None else ""
    q = lambda s: "'" + s.replace("'", "''") + "'"
    if what in ("TABLES", "VIEWS", "OBJECTS"):
        cond = f"database = {q(target_db)}" if target_db else \
            f"database NOT IN ('system', 'information_schema', '{COMPAT_DB}', 'default')"
        if what == "VIEWS":
            cond += " AND engine = 'VIEW'"
        elif what == "TABLES":
            cond += " AND engine <> 'VIEW'"
        sql = (f"SELECT created_on, name, '{ctx.primary_database}' AS database_name, database AS schema_name, "
               f"CASE WHEN engine = 'VIEW' THEN 'VIEW' ELSE 'TABLE' END AS kind, comment, "
               f"num_rows AS rows, data_size AS bytes, 'DATABEND' AS owner "
               f"FROM system.tables WHERE {cond}{like_sql.replace('name LIKE', 'name ILIKE') if False else like_sql} "
               f"ORDER BY schema_name, name")
        return [Statement("sql", sql)]
    if what in ("SCHEMAS", "DATABASES"):
        if what == "DATABASES":
            return [Statement("sql", f"SELECT NOW() AS created_on, '{ctx.primary_database}' AS name")]
        sql = (f"SELECT NOW() AS created_on, name, '{ctx.primary_database}' AS database_name FROM system.databases "
               f"WHERE name NOT IN ('system', 'information_schema', '{COMPAT_DB}'){like_sql} ORDER BY name")
        return [Statement("sql", sql)]
    if what == "COLUMNS":
        cond = ""
        if isinstance(scope, exp.Table):
            sf_db = _ident_name(scope.args["db"]) if scope.args.get("db") is not None else None
            sf_cat = _ident_name(scope.args["catalog"]) if scope.args.get("catalog") is not None else None
            if scope_kind in ("TABLE", "VIEW", ""):
                db = map_database(sf_cat or ctx.database, sf_db, ctx.primary_database) if sf_db else None
                cond = (f" WHERE \"TABLE_NAME\" = {q(_ident_name(scope.this))}"
                        + (f" AND \"TABLE_SCHEMA\" = {q(db)}" if db else " AND \"TABLE_SCHEMA\" = CURRENT_DATABASE()"))
            elif scope_kind == "SCHEMA":
                db = map_database(sf_db or ctx.database, _ident_name(scope.this), ctx.primary_database)
                cond = f" WHERE \"TABLE_SCHEMA\" = {q(db)}"
        sql = (f'SELECT "TABLE_NAME" AS table_name, "TABLE_SCHEMA" AS schema_name, "COLUMN_NAME" AS column_name, '
               f'"DATA_TYPE" AS data_type, "IS_NULLABLE" AS "null?", "COLUMN_DEFAULT" AS "default", '
               f"'COLUMN' AS kind FROM \"{COMPAT_DB}\".\"COLUMNS\"{cond}")
        return [Statement("sql", sql)]
    return [Statement("noop", original=tree.sql("snowflake"))]


def _rewrite_describe(tree: exp.Describe, ctx: Context) -> list[Statement]:
    t = tree.this
    if not isinstance(t, exp.Table):
        return [Statement("noop")]
    sf_db = t.args.get("catalog").name if t.args.get("catalog") is not None else None
    sf_schema = t.args.get("db").name if t.args.get("db") is not None else None
    db = map_database(sf_db or ctx.database, sf_schema, ctx.primary_database) if sf_schema else None
    q = lambda s: "'" + s.replace("'", "''") + "'"
    cond = f'"TABLE_NAME" = {q(t.name.upper() if not t.this.args.get("quoted") else t.name)}'
    if db:
        cond += f' AND "TABLE_SCHEMA" = {q(db)}'
    else:
        cond += ' AND "TABLE_SCHEMA" = CURRENT_DATABASE()'
    sql = (f'SELECT "COLUMN_NAME" AS name, "DATA_TYPE" AS type, \'COLUMN\' AS kind, "IS_NULLABLE" AS "null?", '
           f'"COLUMN_DEFAULT" AS "default", \'N\' AS "primary key", \'N\' AS "unique key", '
           f'NULL AS "check", NULL AS expression, "COMMENT" AS comment '
           f'FROM "{COMPAT_DB}"."COLUMNS" WHERE {cond} ORDER BY "ORDINAL_POSITION"')
    return [Statement("sql", sql)]


def _rewrite_copy(sql: str, ctx: Context) -> Statement:
    """COPY INTO <table> FROM @stage ... -> Databend COPY INTO.

    Snowflake and Databend COPY share their shape; what differs is stage
    naming (``@%TABLE`` table stages don't exist in Databend - PUT uploads
    them to a named stage instead), the table name mapping and a few
    file-format option spellings.
    """
    m = re.match(r"^\s*COPY\s+INTO\s+(?P<target>(?:\"(?:[^\"]|\"\")*\"|[^\s(\"])+)\s*(?P<cols>\([^)]*\))?\s*FROM\s+",
                 sql, re.IGNORECASE | re.DOTALL)
    if not m:
        raise TranslationError("unsupported COPY statement")
    target = m.group("target")
    cols = m.group("cols")
    body = sql[m.end():]
    if body.startswith("("):
        depth, i = 0, 0
        quote = None
        while i < len(body):
            c = body[i]
            if quote:
                if c == quote:
                    quote = None
            elif c in ("'", '"'):
                quote = c
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        src, rest = body[: i + 1], body[i + 1:]
    else:
        parts = body.split(None, 1)
        src, rest = parts[0], (" " + parts[1] if len(parts) > 1 else "")
    if target.startswith("@"):
        raise TranslationError("COPY INTO @stage (unload) is not supported on Databend")
    try:
        t = sqlglot.parse_one(target, read="snowflake", into=exp.Table)
    except (ParseError, TokenError):
        t = None
    if t is not None:
        from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
        t = normalize_identifiers(t, dialect="snowflake")
        tree = exp.select("*").from_(t)
        _map_tables(tree, ctx)
        target_sql = tree.args["from_"].this.sql(dialect=Databend, identify=True)
    else:
        target_sql = target
    cols_sql = ""
    if cols:
        names = [c.strip() for c in cols.strip()[1:-1].split(",") if c.strip()]
        cols_sql = " (" + ", ".join(n if n.startswith('"') else f'"{n.upper()}"' for n in names) + ")"
    stages = []
    if src.startswith("@"):
        src_sql = _stage_ref(src)
        stages.append(_stage_name(src))
    else:
        # (SELECT ... FROM @stage): translate the projection like any
        # SELECT, with the stage reference parked in a placeholder table
        # so sqlglot doesn't have to understand it.
        inner = src.strip()[1:-1]
        refs = {}

        def park(mm):
            key = f"AFYA_STAGE_{len(refs)}"
            refs[key] = mm.group(0)
            return key
        inner = re.sub(r"@[%~]?[\w.\"$]+(?:/[^\s)'\"]*)?", park, inner)
        inner_sql = _translate_one(inner, ctx)[0].sql
        for key, ref in refs.items():
            inner_sql = re.sub(rf'"?{key}"?', _stage_ref(ref), inner_sql, flags=re.I)
            stages.append(_stage_name(ref))
        src_sql = f"({inner_sql})"
    file_formats = []

    def fmt_name(mm):
        name = mm.group(1).strip("'\"").split(".")[-1].strip('"').lower()
        file_formats.append(name)
        return f"FORMAT_NAME = '{name}'"
    rest = re.sub(r"\bFORMAT_NAME\s*=\s*([\w.\"']+)", fmt_name, rest, flags=re.I)
    rest = re.sub(r"\bMATCH_BY_COLUMN_NAME\s*=\s*\w+", "", rest, flags=re.I)
    rest = re.sub(r"\bON_ERROR\s*=\s*'?(\w+)'?", lambda mm: f"ON_ERROR = {mm.group(1).lower()}"
                  if mm.group(1).upper() in ("CONTINUE", "ABORT_STATEMENT") else "", rest, flags=re.I)
    rest = re.sub(r"\bON_ERROR\s*=\s*abort_statement", "ON_ERROR = abort", rest, flags=re.I)
    rest = re.sub(r"\b(FIELD_OPTIONALLY_ENCLOSED_BY)\b", "QUOTE", rest, flags=re.I)
    rest = re.sub(r"\bSKIP_HEADER\b", "SKIP_HEADER", rest, flags=re.I)
    rest = re.sub(r"\bNULL_IF\s*=\s*\(([^)]*)\)", lambda mm: f"NULL_DISPLAY = {mm.group(1).split(',')[0].strip() or chr(39)*2}",
                  rest, flags=re.I)
    rest = re.sub(r"\bEMPTY_FIELD_AS_NULL\s*=\s*\w+", "", rest, flags=re.I)
    rest = re.sub(r"\bERROR_ON_COLUMN_COUNT_MISMATCH\s*=\s*\w+", "", rest, flags=re.I)
    rest = re.sub(r"\bCOMPRESSION\s*=\s*'?AUTO'?", "COMPRESSION = AUTO", rest, flags=re.I)
    rest = re.sub(r"\bBINARY_AS_TEXT\s*=\s*\w+", "", rest, flags=re.I)
    rest = re.sub(r"\bUSE_LOGICAL_TYPE\s*=\s*\w+", "", rest, flags=re.I)
    rest = re.sub(r"\bTIMESTAMP_FORMAT\s*=\s*'[^']*'", "", rest, flags=re.I)
    rest = re.sub(r"\bDATE_FORMAT\s*=\s*'[^']*'", "", rest, flags=re.I)
    rest = re.sub(r"\bFORCE\s*=\s*TRUE", "FORCE = TRUE", rest, flags=re.I)
    rest = re.sub(r"\bTYPE\s*=\s*'?JSON'?", "TYPE = NDJSON", rest, flags=re.I)
    rest = re.sub(r"\bSTRIP_OUTER_ARRAY\s*=\s*\w+", "", rest, flags=re.I)
    rest = re.sub(r";\s*$", "", rest.rstrip())
    # Tidy commas left behind by removed options inside FILE_FORMAT = (...).
    rest = re.sub(r"\(\s*,", "(", re.sub(r",\s*\)", ")", re.sub(r",\s*,", ",", rest)))
    return Statement("copy", f"COPY INTO {target_sql}{cols_sql} FROM {src_sql}{rest}",
                     {"stages": sorted(set(stages)), "file_formats": sorted(set(file_formats))})


def _preprocess(sql: str) -> str:
    # IDENTIFIER('db.schema.t') -> db.schema.t (parameters have been bound
    # into the text by now, see dbapi.py).
    return _IDENTIFIER_RE.sub(lambda m: m.group(1), sql)


def split_statements(sql: str) -> list[str]:
    """Split a multi-statement string on top-level semicolons."""
    out, buf, i, n = [], [], 0, len(sql)
    quote = None
    while i < n:
        c = sql[i]
        if quote:
            buf.append(c)
            if c == "\\" and quote == "'" and i + 1 < n:
                buf.append(sql[i + 1]); i += 2; continue
            if c == quote:
                if i + 1 < n and sql[i + 1] == quote:
                    buf.append(sql[i + 1]); i += 2; continue
                quote = None
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
        elif c == "$" and sql.startswith("$$", i):
            end = sql.find("$$", i + 2)
            end = n if end < 0 else end + 2
            buf.append(sql[i:end]); i = end; continue
        elif c == "-" and sql.startswith("--", i):
            end = sql.find("\n", i)
            end = n if end < 0 else end
            buf.append(sql[i:end]); i = end; continue
        elif c == "/" and sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            end = n if end < 0 else end + 2
            buf.append(sql[i:end]); i = end; continue
        if c == ";":
            stmt = "".join(buf).strip()
            if stmt and not _only_comments(stmt):
                out.append(stmt)
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    stmt = "".join(buf).strip()
    if stmt and not _only_comments(stmt):
        out.append(stmt)
    return out


def _only_comments(s: str) -> bool:
    s = re.sub(r"--[^\n]*", "", s)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    return not s.strip()


_cache_lock = threading.Lock()


@lru_cache(maxsize=4096)
def _translate_one_cached(sql: str, ctx: Context) -> tuple:
    return tuple(_translate_one(sql, ctx))


def translate(sql: str, ctx: Optional[Context] = None) -> list[Statement]:
    """Translate one or more ``;``-separated Snowflake statements."""
    ctx = ctx or Context(primary_database=default_primary_database(), database=default_primary_database())
    out: list[Statement] = []
    for stmt in split_statements(sql):
        if len(stmt) > 200_000:
            out.extend(_translate_one(stmt, ctx))   # don't fill the cache with bulk INSERTs
        else:
            out.extend(_translate_one_cached(stmt, ctx))
    return out


def translate_sql(sql: str, ctx: Optional[Context] = None) -> str:
    """Translate to a single SQL string (for display/EXPLAIN and Redash).

    Client-side actions are rendered as comments.
    """
    parts = []
    for st in translate(sql, ctx):
        if st.action == "sql":
            parts.append(st.sql)
        elif st.action in ("use_schema",) and st.params.get("schema"):
            parts.append(f'USE "{map_database(st.params.get("database") or (ctx or Context()).database, st.params["schema"], (ctx or Context()).primary_database)}"')
        else:
            parts.append(f"/* {st.action} */ SELECT 1 WHERE FALSE")
    return ";\n".join(parts)


def _translate_one(stmt: str, ctx: Context) -> list[Statement]:
    if _PASSTHROUGH_RE.match(stmt):
        return [Statement("sql", re.sub(r";\s*$", "", stmt.strip()), original=stmt)]
    if _NOOP_COMMANDS.match(stmt):
        return [Statement("noop", original=stmt)]
    m = _PUT_RE.match(stmt)
    if m:
        return [Statement("put", "", {"path": m.group("path"), "stage": _stage_name(m.group("stage")),
                                      "subpath": _stage_ref(m.group("stage")).split("/", 1)[1]
                                      if "/" in _stage_ref(m.group("stage")) else "",
                                      "overwrite": bool(re.search(r"OVERWRITE\s*=\s*TRUE", m.group("opts"), re.I))},
                          original=stmt)]
    m = _REMOVE_RE.match(stmt)
    if m:
        return [Statement("remove", f"REMOVE {_stage_ref(m.group('stage'))}", original=stmt)]
    m = _LIST_RE.match(stmt)
    if m:
        return [Statement("sql", f"LIST {_stage_ref(m.group('stage'))}", original=stmt)]
    if re.match(r"^\s*COPY\s+INTO\b", stmt, re.I):
        st = _rewrite_copy(stmt, ctx)
        st.original = stmt
        return [st]

    pre = _preprocess(stmt)
    try:
        tree = sqlglot.parse_one(pre, read="snowflake")
    except (ParseError, TokenError) as exc:
        raise TranslationError(f"could not parse statement for Databend: {exc}") from exc
    if tree is None:
        return [Statement("noop", original=stmt)]
    if isinstance(tree, exp.Command):
        head = (tree.name or "").upper()
        if head in ("GRANT", "REVOKE", "ALTER", "SHOW", "CALL", "EXECUTE"):
            if head == "CALL":
                raise TranslationError("stored procedures (CALL) are not available on Databend")
            return [Statement("noop", original=stmt)]
        # Unknown command: send as is and let Databend decide.
        return [Statement("sql", stmt, original=stmt)]
    if isinstance(tree, exp.Grant) or type(tree).__name__ in ("Revoke",):
        return [Statement("noop", original=stmt)]
    if isinstance(tree, exp.Use):
        return _rewrite_use(tree, ctx)
    if isinstance(tree, exp.Show):
        return _rewrite_show(tree, ctx)
    if isinstance(tree, exp.Describe):
        return _rewrite_describe(tree, ctx)
    if isinstance(tree, exp.TruncateTable) and tree.args.get("exists"):
        tables = tree.expressions
        tree.set("exists", False)
        _normalize_and_map(tree, ctx)
        return [Statement("truncate_if_exists", "",
                          {"tables": [t.sql(dialect=Databend, identify=True) for t in tables]}, original=stmt)]
    if isinstance(tree, exp.Alter):
        special = _rewrite_alter(tree, ctx)
        if special is not None:
            return special
    if isinstance(tree, exp.Drop) and (tree.args.get("kind") or "").upper() == "SCHEMA":
        sf_db, schema = _schema_ref(tree.this)
        target = map_database(sf_db or ctx.database, schema, ctx.primary_database)
        return [Statement("sql", f'DROP DATABASE {"IF EXISTS " if tree.args.get("exists") else ""}"{target}"',
                          original=stmt)]
    if isinstance(tree, exp.Drop) and (tree.args.get("kind") or "").upper() in (
            "STAGE",):
        return [Statement("sql", f'DROP STAGE IF EXISTS {_stage_name(tree.this.name)}', original=stmt)]
    if isinstance(tree, exp.Drop) and (tree.args.get("kind") or "").upper() in (
            "FUNCTION", "PROCEDURE", "TASK", "STREAM", "PIPE", "FILE FORMAT", "ROLE", "USER", "WAREHOUSE", "TAG"):
        return [Statement("noop", original=stmt)]
    if isinstance(tree, exp.Create):
        special = _rewrite_create(tree, ctx)
        if special:
            for s in special:
                s.original = stmt
            return special
    tree = _rewrite_delete_using(tree)
    tree = _rewrite_union_ordinal_order(tree)
    _rewrite_merge_set(tree)
    _normalize_and_map(tree, ctx)
    sql = tree.sql(dialect=Databend, identify=True)
    return [Statement("sql", sql, original=stmt)]


def _normalize_and_map(tree: exp.Expression, ctx: Context) -> None:
    from sqlglot.optimizer.normalize_identifiers import normalize_identifiers

    _protect_flatten_columns(tree)
    _rewrite_values_columns(tree)
    normalize_identifiers(tree, dialect="snowflake")
    _map_tables(tree, ctx)
    _rewrite_hash_star(tree)


HASH_STAR_MARKER = "AFYA_HASH_STAR"


def _rewrite_hash_star(tree: exp.Expression) -> None:
    """HASH(*) -> AFYA_HASH_STAR('<table>'); the Databend connection swaps
    the marker for a hash over that table's actual columns at run time
    (Databend can't hash a whole row)."""
    for fn in list(tree.find_all(exp.Anonymous)):
        if fn.name.upper() != "HASH" or len(fn.expressions) != 1 or not isinstance(fn.expressions[0], exp.Star):
            continue
        select = fn.find_ancestor(exp.Select)
        src = select.args.get("from_") if select is not None else None
        table = src.this if src is not None else None
        if not isinstance(table, exp.Table) or select.args.get("joins"):
            raise TranslationError("HASH(*) on Databend needs a single-table FROM")
        ref = exp.Table(this=table.this.copy(), db=table.args.get("db").copy() if table.args.get("db") else None)
        fn.replace(exp.Anonymous(this=HASH_STAR_MARKER,
                                 expressions=[exp.Literal.string(ref.sql(dialect=Databend, identify=True))]))
