"""Snowflake access for the model. Metadata is always readable; source data only with allow_data=True.

Every statement goes through one guard: SELECT/WITH (or EXPLAIN of one) only, no `*` projection, no write keyword.
Session: role DATAANALYSTS without secondary roles, per-statement timeout, query tag.
"""
import json
import re
import time

from .. import config as C
SESSION = ["USE ROLE DATAANALYSTS", "USE SECONDARY ROLES NONE",
           f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {C.STATEMENT_TIMEOUT_SECONDS}, "
           "PYTHON_CONNECTOR_QUERY_RESULT_FORMAT = 'JSON'"]


def _connect():
    import os
    import sys
    dash_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if dash_dir not in sys.path:
        sys.path.insert(0, dash_dir)
    from snowflake_service.snowflake_client import SnowflakeClient
    return SnowflakeClient().conn


def _ident(name):
    return name if name == name.upper() else '"' + name.replace('"', '""') + '"'
from .interpret import nominate

SOURCES = f"{C.DATABASE}.ANALYTICS.AFYA_PATIENT_COUNT_SOURCES"
TAG = "ALTER SESSION SET QUERY_TAG = 'afya_patient_discovery_model'"
WRITE = re.compile(r"\b(INSERT|UPDATE|DELETE|MERGE|CREATE|DROP|TRUNCATE|CALL|EXECUTE|GRANT|REVOKE|COPY|PUT|REMOVE|"
                   r"UNDROP|ALTER|COMMIT|BEGIN|ROLLBACK|USE)\b", re.I)
STAR = re.compile(r"SELECT\s+(DISTINCT\s+)?\*|,\s*\*\s*(,|FROM)|\.\*", re.I)
VIEW_ROWS = 1_000_000          # views have no ROW_COUNT: assume a large table when estimating cost


LITERAL = re.compile(r"'(?:[^'\\]|\\.|'')*'")


def check(sql):
    s = LITERAL.sub("''", sql.strip())     # string literals (e.g. regex patterns like '.*[A-Z].*') are data, not SQL
    if not re.match(r"(SELECT|WITH)\b", s, re.I) or WRITE.search(s) or STAR.search(s) or ";" in s:
        raise ValueError("refused: only single aggregate SELECT statements are allowed")


class Snowflake:
    def __init__(self, conn=None, allow_data=False, max_statements=None):
        self.conn = conn or _connect()
        self.allow_data, self.max_statements = allow_data, max_statements
        self.n_queries = self.n_meta = self.n_explain = 0
        cur = self.conn.cursor()
        for s in SESSION + [TAG]:
            cur.execute(s)
        cur.close()

    def _rows(self, sql, params=None):
        cur = self.conn.cursor()
        try:
            cur.execute(sql, params)
            names = [d[0].upper() for d in cur.description]
            return [dict(zip(names, r)) for r in cur.fetchall()]
        finally:
            cur.close()

    # ------------------------------------------------------------------ metadata (no source data)
    def catalogue(self, schemas=None, keep=(), lineage=True, progress=None):
        """Tables and views with columns; *_RAW stores are included (marked `store`) so views over them change
        fingerprint when the store changes. `schemas` limits the candidate objects; `keep` adds objects by name."""
        skip = ", ".join(f"'{s}'" for s in sorted(C.SKIP_SCHEMAS))
        self.n_meta += 1
        recs = self._rows(f"""
            SELECT t.TABLE_SCHEMA, t.TABLE_NAME, t.TABLE_TYPE, t.ROW_COUNT, t.LAST_ALTERED::STRING AS LAST_ALTERED,
                   ARRAY_AGG(OBJECT_CONSTRUCT('n', c.COLUMN_NAME, 't', c.DATA_TYPE))
                       WITHIN GROUP (ORDER BY c.ORDINAL_POSITION) AS COLS
            FROM {C.DATABASE}.INFORMATION_SCHEMA.TABLES t
            JOIN {C.DATABASE}.INFORMATION_SCHEMA.COLUMNS c
              ON c.TABLE_SCHEMA = t.TABLE_SCHEMA AND c.TABLE_NAME = t.TABLE_NAME
            WHERE t.TABLE_SCHEMA NOT IN ({skip})
            GROUP BY 1, 2, 3, 4, 5""")
        keep = {k.upper() for k in keep}
        out = []
        for r in recs:
            fqn = f"{C.DATABASE}.{r['TABLE_SCHEMA']}.{r['TABLE_NAME']}"
            store = r["TABLE_SCHEMA"].endswith(C.RAW_SCHEMA_SUFFIX)
            if schemas and not store and r["TABLE_SCHEMA"] not in schemas and fqn.upper() not in keep:
                continue
            cols = r["COLS"] if isinstance(r["COLS"], list) else json.loads(r["COLS"])
            o = dict(fqn=fqn, schema=r["TABLE_SCHEMA"], name=r["TABLE_NAME"], type=r["TABLE_TYPE"],
                     row_count=r["ROW_COUNT"], last_altered=r["LAST_ALTERED"], store=store,
                     columns=[(c["n"], _type(c["t"])) for c in cols], refs=[])
            if o["type"] != "BASE TABLE":
                o["row_estimate"] = VIEW_ROWS
            out.append(o)
        if lineage:
            views = [o for o in out if o["type"] != "BASE TABLE" and not o["store"]
                     and o["schema"] not in C.EXCLUDED_SCHEMAS and nominate(o["columns"], set())[0]]
            t0 = time.time()
            for i, o in enumerate(views):
                o["refs"] = self.lineage(o["fqn"])
                if progress and i % 50 == 49:
                    progress(f"lineage {i + 1}/{len(views)} ({time.time() - t0:.0f}s)")
        return out

    def lineage(self, fqn):
        db, sch, obj = fqn.split(".")
        self.n_meta += 1
        try:
            recs = self._rows(
                f"SELECT REFERENCED_DATABASE_NAME AS D, REFERENCED_SCHEMA_NAME AS S, REFERENCED_OBJECT_NAME AS O "
                f"FROM TABLE({db}.INFORMATION_SCHEMA.GET_OBJECT_REFERENCES("
                f"DATABASE_NAME => %s, SCHEMA_NAME => %s, OBJECT_NAME => %s))", (_ident(db), _ident(sch), _ident(obj)))
        except Exception:
            return None                                   # unreadable lineage is a question, not an error
        return sorted({f"{r['D']}.{r['S']}.{r['O']}" for r in recs})

    def registry(self):
        self.n_meta += 1
        return self._rows(f"SELECT SOURCE, UPPER(TRIM(SOURCE_OBJECT)) AS SOURCE_OBJECT, UPD_KEY, RECORD_KEY, "
                          f"DELETED_COLUMN, VERSION_COLUMN, RECEIVED_COLUMN, STATUS FROM {SOURCES}")

    def explain(self, sql):
        """Compiles a measurement statement; reads no table data."""
        check(sql)
        self.n_explain += 1
        cur = self.conn.cursor()
        try:
            cur.execute("EXPLAIN USING TEXT " + sql)
            cur.fetchall()
        finally:
            cur.close()

    # ------------------------------------------------------------------ the two discovery tables (the only writes)
    def fetch(self, sql):
        """Reads of the discovery tables (our own analytics tables, not source data)."""
        check(sql)
        self.n_meta += 1
        return self._rows(sql)

    def write(self, pairs):
        """(sql, params) pairs in one transaction, after store.check_writes: the two discovery tables, and the
        guarded insert-if-absent into the registry. Nothing else."""
        from .store import check_writes
        check_writes(pairs)
        cur = self.conn.cursor()
        try:
            cur.execute("BEGIN")
            for sql, params in pairs:
                cur.execute(sql, params)
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise
        finally:
            cur.close()

    # ------------------------------------------------------------------ source data (aggregates only)
    def query(self, sql):
        check(sql)
        if not self.allow_data:
            raise PermissionError("reading source data needs the approved profile command")
        if self.max_statements is not None and self.n_queries >= self.max_statements:
            raise PermissionError(f"statement limit {self.max_statements} reached")
        self.n_queries += 1
        cur = self.conn.cursor()
        try:
            cur.execute(sql)
            return cur.fetchall()
        finally:
            cur.close()


def _type(t):
    t = (t or "").upper()
    if t in ("NUMBER", "DECIMAL", "INTEGER", "BIGINT", "SMALLINT"):
        return "NUMBER"
    if t in ("FLOAT", "DOUBLE", "REAL"):
        return "FLOAT"
    return t
