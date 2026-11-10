"""One aggregate profile query per candidate: a single read of a handful of columns, exact counts, no sampling.

The template is shared; only the key / deletion / date expressions differ per dialect. 'sqlite' exists so the
tests run the same SQL shape locally; its functions (registered by the test warehouse) mirror the Snowflake
expressions below. Only aggregates come back - never a key value.
"""
from decimal import Decimal, InvalidOperation

from . import config as C

import re

FIELDS = ["rows", "rows_no_key", "keys", "singleton_keys", "top_key_records", "key_records", "rows_with_record_key",
          "key_record_pairs", "raw_key_values", "rows_key_equals_record", "rows_deleted", "date_min", "date_max"]
RECORD_FIELDS = ["distinct_records", "records_with_many_keys"]
_WS = "' ' || CHR(9) || CHR(10) || CHR(13) || CHR(160)"     # Snowflake TRIM characters, incl. non-breaking space

NUMERIC_MISSING = {Decimal(v) for v in C.MISSING_KEY_VALUES if v.lstrip("-").isdigit()}   # 0, -1, 99999, 999999
_NUMERIC_MISSING_SQL = ", ".join(str(int(d)) for d in sorted(NUMERIC_MISSING))


def q(name):
    return '"' + name.replace('"', '""') + '"'


def _sql_list(values):
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


# ------------------------------------------------------------------ Python reference semantics (used by sqlite)
def normalise_key(value, data_type):
    """Numbers (or numeric text) -> fixed 6-decimal text; other text -> trimmed, upper-case, zero padding after a
    letter removed (P0012 -> P12); sentinels -> None. Mirrors Snowflake.key below."""
    if value is None:
        return None
    if data_type == "TEXT":
        s = str(value).strip(" \t\n\r\u00a0")
        if s.upper() in C.MISSING_KEY_VALUES:
            return None
        try:
            d = Decimal(s)
        except InvalidOperation:
            return re.sub(r"([A-Z])0+([0-9])", r"\1\2", s.upper())
        if not d.is_finite():
            return re.sub(r"([A-Z])0+([0-9])", r"\1\2", s.upper())
    else:
        d = Decimal(str(value))
    if d in NUMERIC_MISSING:
        return None
    return f"{d:.6f}"


def raw_key(value, data_type):
    """The value as the counting refresh sees it: TRIM of spaces only (text keys; numbers have no variants)."""
    if value is None:
        return None
    return str(value).strip(" ") if data_type == "TEXT" else normalise_key(value, data_type)


def is_deleted(value):
    return str(value).strip() not in C.REFRESH_NOT_DELETED_VALUES if value is not None else False


# ------------------------------------------------------------------ dialects
class Snowflake:
    @staticmethod
    def key(col, data_type):
        if data_type == "TEXT":
            t = f"TRIM({col}, {_WS})"
            num = f"TRY_TO_NUMBER({t}, 38, 6)"
            return (f"IFF(UPPER({t}) IN ({_sql_list(C.MISSING_KEY_VALUES)}) OR {num} IN ({_NUMERIC_MISSING_SQL}), NULL, "
                    f"COALESCE(TO_VARCHAR({num}), REGEXP_REPLACE(UPPER({t}), '([A-Z])0+([0-9])', '\\\\1\\\\2')))")
        n = f"{col}::NUMBER(38, 6)"
        return f"IFF({n} IN ({_NUMERIC_MISSING_SQL}), NULL, TO_VARCHAR({n}))"

    @staticmethod
    def raw(col, data_type):
        return f"TRIM({col}::STRING)" if data_type == "TEXT" else Snowflake.key(col, data_type)

    @staticmethod
    def deleted(col):
        return f"TRIM(COALESCE({col}::STRING, '')) NOT IN ({_sql_list(C.REFRESH_NOT_DELETED_VALUES)})"

    @staticmethod
    def date(col):
        return f"TRY_TO_DATE({col}::STRING)"

    @staticmethod
    def relation(c):
        return ".".join(q(p) for p in (C.DATABASE, c.schema, c.name))


class SQLite:
    @staticmethod
    def key(col, data_type):
        return f"norm_key({col}, '{data_type}')"

    @staticmethod
    def raw(col, data_type):
        return f"raw_key({col}, '{data_type}')"

    @staticmethod
    def deleted(col):
        return f"is_deleted({col})"

    @staticmethod
    def date(col):
        return f"date({col})"

    @staticmethod
    def relation(c):
        return q(c.fqn)


DIALECTS = {"snowflake": Snowflake, "sqlite": SQLite}


def profile_statements(c, dialect="snowflake"):
    """Main per-person profile; plus, when the refresh would deduplicate on the record key (version column present),
    a check that each record ID belongs to one person."""
    D = DIALECTS[dialect]
    types = {n: t for n, t in c.columns}
    k = D.key(q(c.key), c.key_type)
    raw = D.raw(q(c.key), c.key_type)
    rk = D.key(q(c.record_key), types[c.record_key]) if c.record_key else "NULL"
    dele = D.deleted(q(c.deleted_column)) if c.deleted_column else "1 = 0"
    dt = D.date(q(c.date_column)) if c.date_column else "NULL"
    eff = "(CASE WHEN n_rk > 0 THEN nd_rk + n - n_rk ELSE n END)"       # records once versions collapse
    main = f"""SELECT SUM(n), SUM(CASE WHEN k IS NULL THEN n ELSE 0 END), COUNT(k),
       SUM(CASE WHEN k IS NOT NULL AND {eff} = 1 THEN 1 ELSE 0 END), MAX(CASE WHEN k IS NOT NULL THEN {eff} END),
       SUM(CASE WHEN k IS NOT NULL THEN {eff} ELSE 0 END),
       SUM(CASE WHEN k IS NOT NULL THEN n_rk ELSE 0 END), SUM(CASE WHEN k IS NOT NULL THEN nd_rk ELSE 0 END),
       SUM(CASE WHEN k IS NOT NULL THEN n_raw ELSE 0 END), SUM(n_eq), SUM(n_del), MIN(d_min), MAX(d_max)
FROM (SELECT {k} AS k, COUNT(*) AS n, COUNT({rk}) AS n_rk, COUNT(DISTINCT {rk}) AS nd_rk,
             COUNT(DISTINCT {raw}) AS n_raw,
             SUM(CASE WHEN {k} = {rk} THEN 1 ELSE 0 END) AS n_eq,
             SUM(CASE WHEN {dele} THEN 1 ELSE 0 END) AS n_del,
             MIN({dt}) AS d_min, MAX({dt}) AS d_max
      FROM {D.relation(c)}
      GROUP BY {k}) g"""
    if not (c.record_key and c.version_column):
        return [main]
    records = f"""SELECT COUNT(*), SUM(CASE WHEN nk > 1 THEN 1 ELSE 0 END)
FROM (SELECT {rk} AS rk, COUNT(DISTINCT {k}) AS nk FROM {D.relation(c)} WHERE {rk} IS NOT NULL GROUP BY {rk}) r"""
    return [main, records]


def profile_sql(c, dialect="snowflake"):
    return profile_statements(c, dialect)[0]


def parse(rows):
    """rows: one result row per statement from profile_statements, in order."""
    p = dict(zip(FIELDS, rows[0]))
    for f in FIELDS[:11]:
        p[f] = int(p[f] or 0)
    for f in ("date_min", "date_max"):
        p[f] = str(p[f]) if p[f] is not None else None
    if len(rows) > 1:
        p.update({f: int(v or 0) for f, v in zip(RECORD_FIELDS, rows[1])})
    return p
