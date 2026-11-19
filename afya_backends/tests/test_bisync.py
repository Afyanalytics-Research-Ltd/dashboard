"""warehouse_bisync: planning (pure) and copy/reconcile/views against a real Databend.

The live tests use two Databend "sides" (Snowflake-style databases mapped to
Databend databases <DB>__<SCHEMA>) so no Snowflake is needed; skipped when no
Databend is reachable (DATABEND_HTTP_URL).
"""

import os
import uuid

import pytest
import requests

from afya_backends.warehouse import bisync
from afya_backends.warehouse.bisync import TableInfo

URL = os.getenv("DATABEND_HTTP_URL", "http://localhost:8008")


# -- plan (pure) ----------------------------------------------------------------

def _ti(rows, mod="m1", schema="ANALYTICS", name="T"):
    return TableInfo(schema, name, rows, mod)


def test_plan_append_key_table_reconciles_instead_of_full_copy():
    key = ("ANALYTICS", "AFYA_HMIS_PATIENT_COUNTS")
    state = {"ANALYTICS.AFYA_HMIS_PATIENT_COUNTS": {"sf": "100|m1", "db": "100|m1"}}
    sf, db = {key: _ti(110, "m2")}, {key: _ti(100, "m1")}
    assert bisync.plan(sf, db, state, allow_shrink=False) == [(key, "copy_sf_to_db")]
    assert bisync.plan(sf, db, state, allow_shrink=False,
                       append_keys={"ANALYTICS.AFYA_HMIS_PATIENT_COUNTS": "RUN_ID"}) == [(key, "reconcile")]
    # unchanged stays a no-op
    assert bisync.plan({key: _ti(100)}, {key: _ti(100)}, state, allow_shrink=False,
                       append_keys={"ANALYTICS.AFYA_HMIS_PATIENT_COUNTS": "RUN_ID"}) == [(key, "noop")]


def test_append_keys_env(monkeypatch):
    monkeypatch.setenv("BISYNC_APPEND_KEYS", "reporting.snap=snap_id, x.y = run")
    keys = bisync.append_keys_from_env()
    assert keys["REPORTING.SNAP"] == "SNAP_ID" and keys["X.Y"] == "RUN"
    assert keys["ANALYTICS.AFYA_HMIS_PATIENT_COUNTS"] == "RUN_ID"


def test_side_real_names():
    side = bisync.Side("databend", None, "HOSPITALS")
    side._names = {("TENRI", "CAFETERIA_PRODUCTS"): ("TENRI", "cafeteria_products")}
    assert side.real("tenri", "CAFETERIA_PRODUCTS") == ("TENRI", "cafeteria_products")
    assert side.real("X", "Y") == ("X", "Y")


def test_plan_migrates_old_snowflake_fingerprints_by_row_count():
    """Fingerprints recorded as rows|LAST_ALTERED must not make every table look changed once Snowflake is read
    with SHOW TABLES (rows|b<bytes>): the first run compares row counts only."""
    key = ("V3_SYNC", "T")
    state = {"V3_SYNC.T": {"sf": "100|2026-11-18 14:34:40.758 Z", "db": "100|d1"}}
    db = {key: _ti(100, "d1", "V3_SYNC")}
    assert bisync.plan({key: _ti(100, "b4096", "V3_SYNC")}, db, state, allow_shrink=False) == [(key, "noop")]
    assert bisync.plan({key: _ti(120, "b4096", "V3_SYNC")}, db, state, allow_shrink=False) == [(key, "copy_sf_to_db")]
    # once migrated, bytes count: an UPDATE (same rows, new bytes) is a change
    state["V3_SYNC.T"]["sf"] = "100|b4096"
    assert bisync.plan({key: _ti(100, "b4096", "V3_SYNC")}, db, state, allow_shrink=False) == [(key, "noop")]
    assert bisync.plan({key: _ti(100, "b8192", "V3_SYNC")}, db, state, allow_shrink=False) == [(key, "copy_sf_to_db")]


class _ShowCursor:
    def __init__(self, results, log):
        self.results, self.log, self.description, self._rows = results, log, None, []

    def execute(self, sql, params=None):
        self.log.append(sql)
        cols, rows = next(v for k, v in self.results.items() if sql.startswith(k))
        self.description = [(c,) for c in cols]
        self._rows = rows

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class _ShowConn:
    def __init__(self, results):
        self.results, self.sql = results, []

    def cursor(self):
        return _ShowCursor(self.results, self.sql)


def test_snowflake_metadata_uses_show_and_desc_not_a_warehouse():
    """Listing tables/views and reading columns must not resume the warehouse (INFORMATION_SCHEMA does: an hourly
    no-op run was billed ~6 warehouse-minutes, and failed outright while the resource monitor blocked it)."""
    cols = ["created_on", "name", "database_name", "schema_name", "kind", "rows", "bytes", "is_external"]
    conn = _ShowConn({
        "SHOW TABLES IN DATABASE": (cols, [
            (None, "activations", "HOSPITALS", "TENRI", "TABLE", 247, 1024, "N"),
            (None, "ACTIVATIONS", "HOSPITALS", "TENRI", "TABLE", 240, 2048, "N"),
            (None, "cache", "HOSPITALS", "TENRI", "TABLE", 20, 512, "N"),
            (None, "EXT", "HOSPITALS", "RAW", "TABLE", 5, 0, "Y"),
            (None, "TMP", "HOSPITALS", "RAW", "TEMPORARY", 5, 0, "N"),
            (None, "T", "HOSPITALS", "RAW", "TRANSIENT", 7, 4096, "N"),
            (None, "TABLES", "HOSPITALS", "INFORMATION_SCHEMA", "TABLE", 1, 1, "N"),
        ]),
        "SHOW TABLES LIKE": (cols, [(None, "T", "HOSPITALS", "RAW", "TABLE", 9, 8192, "N"),
                                    (None, "TX", "HOSPITALS", "RAW", "TABLE", 1, 1, "N")]),
        "SHOW VIEWS IN DATABASE": (["name", "schema_name", "text"], [("V", "RPT", "create view V as select 1")]),
        "DESC TABLE": (["name", "type", "kind"], [("ID", "NUMBER(38,0)", "COLUMN"), ("AMT", "NUMBER(12, 2)", "COLUMN"),
                                                  ("NAME", "VARCHAR(16777216)", "COLUMN"),
                                                  ("AT", "TIMESTAMP_NTZ(9)", "COLUMN"), ("J", "VARIANT", "COLUMN"),
                                                  ("F", "FLOAT", "COLUMN")]),
    })
    side = bisync.Side("snowflake", conn, "HOSPITALS")
    t = side.tables()
    assert set(t) == {("TENRI", "ACTIVATIONS"), ("TENRI", "CACHE"), ("RAW", "T")}
    assert t[("TENRI", "ACTIVATIONS")].name == "ACTIVATIONS"         # upper-case spelling wins, every run
    assert t[("TENRI", "ACTIVATIONS")].fingerprint == "240|b2048"
    assert t[("TENRI", "CACHE")].name == "cache"                     # the only spelling is kept as is
    assert side.info("RAW", "T").fingerprint == "9|b8192"           # exact name, not the LIKE wildcard match
    assert side.views() == {("RPT", "V"): "create view V as select 1"}
    c = {x.name: x for x in side.columns("RAW", "T")}
    assert (c["ID"].kind, c["ID"].precision, c["ID"].scale) == ("NUMBER", 38, 0)
    assert (c["AMT"].precision, c["AMT"].scale) == (12, 2)
    assert [c[n].kind for n in ("NAME", "AT", "J", "F")] == ["TEXT", "TIMESTAMP", "VARIANT", "FLOAT"]
    assert all(s.startswith(("SHOW ", "DESC ")) for s in conn.sql)


def test_view_comment_clause_is_dropped_for_databend():
    """SHOW VIEWS text keeps the view COMMENT; Databend rejects COMMENT= on CREATE VIEW (70 prod views, 2026-11-19)."""
    d = ("CREATE OR REPLACE VIEW HOSPITALS.SILVERWOOD_EXTRACTION_MAPPED.CORE_FACILITY\n"
         "comment = 'V3 App\\\\Models\\\\Core\\\\Facility mapped from facility (connection 35); latest load only' \n"
         "AS\nSELECT id AS comment_id, 'x = ''y''' AS note FROM HOSPITALS.RAW.T")
    sql = bisync._view_ddl(d, "SILVERWOOD_EXTRACTION_MAPPED", "HOSPITALS")
    assert sql.upper().startswith("CREATE OR REPLACE VIEW") and "Facility mapped" not in sql
    assert "comment_id" in sql.lower()                       # body untouched
    plain = "create or replace view HOSPITALS.RPT.V as select 1 as comment"
    assert "comment" in bisync._view_ddl(plain, "RPT", "HOSPITALS").lower()


# -- live --------------------------------------------------------------------------

def _reachable():
    try:
        return requests.get(URL + "/health", timeout=3).status_code == 200
    except requests.RequestException:
        return False


live = pytest.mark.skipif(not _reachable(), reason="no Databend at DATABEND_HTTP_URL")


@pytest.fixture()
def sides(monkeypatch):
    monkeypatch.setenv("DATABEND_HTTP_URL", URL)
    from afya_backends.warehouse.dbapi import Connection
    tag = uuid.uuid4().hex[:8].upper()
    a_db, b_db = f"BSA{tag}", f"BSB{tag}"
    ca, cb = Connection(database=a_db), Connection(database=b_db)
    a = bisync.Side("databend", ca, a_db)
    b = bisync.Side("databend", cb, b_db)
    yield a, b
    for c, d in ((ca, a_db), (cb, b_db)):
        for (db_name,) in c._t.run(f"SELECT name FROM system.databases WHERE name LIKE '{d}__%'").rows:
            c._t.run(f'DROP DATABASE IF EXISTS "{db_name}"')
        c.close()


@live
def test_copy_uses_each_sides_own_table_spelling(sides):
    """TENRI.ACTIVATIONS vs TENRI.activations: the replace must rename the
    destination's own table (it failed hourly with 'Unknown table')."""
    a, b = sides
    a.run(f'CREATE DATABASE "{a.bend_db("TENRI")}"')
    b.run(f'CREATE DATABASE "{b.bend_db("TENRI")}"')
    a.run(f'CREATE TABLE {a.fq("TENRI", "ACTIVATIONS")} ("ID" INT, "V" VARCHAR)')
    a.run(f"INSERT INTO {a.fq('TENRI', 'ACTIVATIONS')} VALUES (1, 'x'), (2, 'y'), (3, 'z')")
    b.run(f'CREATE TABLE {b.fq("TENRI", "activations")} ("ID" INT, "V" VARCHAR)')
    b.run(f"INSERT INTO {b.fq('TENRI', 'activations')} VALUES (1, 'x')")
    a.tables(), b.tables()
    n = bisync.copy_table(a, b, "TENRI", "ACTIVATIONS", dst_exists=True, log=lambda m: None)
    assert n == 3
    assert b.count("TENRI", "activations") == 3
    names = {r[0] for r in b.run(f"SELECT name FROM system.tables WHERE database = '{b.bend_db('TENRI')}'")}
    assert names == {"activations"}            # no stray ACTIVATIONS / staging / old tables


@live
def test_reconcile_by_append_key_copies_only_new_runs_both_ways(sides):
    a, b = sides
    for s in (a, b):
        s.run(f'CREATE DATABASE "{s.bend_db("ANALYTICS")}"')
        s.run(f'CREATE TABLE {s.fq("ANALYTICS", "PC")} ("RUN_ID" VARCHAR, "N" INT)')
    a.run(f"INSERT INTO {a.fq('ANALYTICS', 'PC')} VALUES ('r1', 1), ('r1', 2), ('r2', 3), ('r2', 4)")
    b.run(f"INSERT INTO {b.fq('ANALYTICS', 'PC')} VALUES ('r1', 1), ('r1', 2), ('r0', 9)")
    a.tables(), b.tables()
    keys = {"ANALYTICS.PC": "RUN_ID"}
    assert bisync._key_for(a, b, "ANALYTICS", "PC", keys) == "RUN_ID"
    assert bisync.insert_missing(a, b, "ANALYTICS", "PC", "RUN_ID", log=lambda m: None) == 2   # r2
    assert bisync.insert_missing(b, a, "ANALYTICS", "PC", "RUN_ID", log=lambda m: None) == 1   # r0
    assert a.count("ANALYTICS", "PC") == b.count("ANALYTICS", "PC") == 5
    # in sync -> nothing read or written
    assert bisync.insert_missing(a, b, "ANALYTICS", "PC", "RUN_ID", log=lambda m: None) == 0


def _v3_table(side, rows):
    side.run(f'CREATE DATABASE IF NOT EXISTS "{side.bend_db("V3_SYNC")}"')
    side.run(f'CREATE TABLE {side.fq("V3_SYNC", "M")} ("_TENANT_ID" INT, "_KEY" VARCHAR, "NAME" VARCHAR, '
             f'"_IS_DELETED" BOOLEAN, "_LAST_LOADED_AT" TIMESTAMP)')
    if rows:
        side.run(f"INSERT INTO {side.fq('V3_SYNC', 'M')} VALUES " + ", ".join(rows))


@live
def test_incremental_merges_only_rows_loaded_since_the_watermark(sides):
    """V3_SYNC: new rows inserted, changed / delete-flagged rows updated by (_TENANT_ID, _KEY); older rows not read."""
    src, dst = sides
    _v3_table(src, ["(1, 'a', 'old-a', false, '2026-11-01 00:00:00')",        # unchanged, far before the watermark
                    "(1, 'b', 'new-b', false, '2026-11-18 12:00:00')",        # changed after the last sync
                    "(2, 'a', 'gone', true, '2026-11-18 12:00:00')",          # flagged deleted
                    "(2, 'c', 'brand-new', false, '2026-11-18 12:30:00')"])   # new
    _v3_table(dst, ["(1, 'a', 'old-a', false, '2026-11-01 00:00:00')",
                    "(1, 'b', 'old-b', false, '2026-11-18 10:00:00')",
                    "(2, 'a', 'gone', false, '2026-11-18 10:00:00')"])
    src.tables(), dst.tables()
    n, got = bisync.incremental_copy(src, dst, "V3_SYNC", "M", src_rows=4, log=lambda m: None)
    assert (n, got) == (3, 4)
    rows = {(r[0], r[1]): (r[2], r[3]) for r in dst.run(
        f'SELECT "_TENANT_ID", "_KEY", "NAME", "_IS_DELETED" FROM {dst.fq("V3_SYNC", "M")}')}
    assert rows[("1", "b")][0] == "new-b" and rows[("2", "c")][0] == "brand-new"
    assert str(rows[("2", "a")][1]).lower() in ("1", "true")
    names = {r[0] for r in dst.run(f"SELECT name FROM system.tables WHERE database = '{dst.bend_db('V3_SYNC')}'")}
    assert names == {"M"}                                               # staging dropped
    # nothing new -> only the overlap window is re-read, merged idempotently
    n, got = bisync.incremental_copy(src, dst, "V3_SYNC", "M", src_rows=4, log=lambda m: None)
    assert got == 4


@live
def test_incremental_falls_back_when_columns_differ_or_target_empty(sides):
    src, dst = sides
    _v3_table(src, ["(1, 'a', 'x', false, '2026-11-18 12:00:00')"])
    _v3_table(dst, [])
    src.tables(), dst.tables()
    with pytest.raises(bisync.NotIncremental, match="watermark"):
        bisync.incremental_copy(src, dst, "V3_SYNC", "M", src_rows=1, log=lambda m: None)
    src.run(f'ALTER TABLE {src.fq("V3_SYNC", "M")} ADD COLUMN "EXTRA" VARCHAR')
    with pytest.raises(bisync.NotIncremental, match="columns differ"):
        bisync.incremental_copy(src, dst, "V3_SYNC", "M", src_rows=1, log=lambda m: None)


@live
def test_views_created_replaced_and_restored(sides, monkeypatch):
    """views_sf_to_db with a fake Snowflake side that only lists definitions."""
    _, db = sides
    db.run(f'CREATE DATABASE "{db.bend_db("RAW")}"')
    db.run(f'CREATE TABLE {db.fq("RAW", "T")} ("ID" INT, "AMOUNT" VARCHAR)')
    db.run(f"INSERT INTO {db.fq('RAW', 'T')} VALUES (1, '2.5'), (1, '1')")

    class FakeSF:
        database = db.database
        defs = {}

        def views(self):
            return dict(self.defs)

    sf = FakeSF()
    d = db.database
    sf.defs = {
        ("RPT", "V_OK"): f"create or replace view {d}.RPT.V_OK as select id from {d}.RAW.T",
        ("RPT", "V_DEP"): f"create or replace view {d}.RPT.V_DEP as with deduped as (select distinct id from {d}.RPT.V_OK) select * from deduped",
        ("RPT", "V_SUM"): f"create or replace view {d}.RPT.V_SUM as select id, sum(amount) as total from {d}.RAW.T group by id",
        ("RPT", "V_BAD"): f"create or replace view {d}.RPT.V_BAD as select * from {d}.NOPE.MISSING",
    }
    monkeypatch.setattr(bisync, "VIEW_PAUSE_S", 0)
    state, report = {}, {"views_created": [], "views_replaced": [], "views_adopted": 0, "views_failed": []}
    bisync.views_sf_to_db(sf, db, state, report, log=lambda m: None)
    assert sorted(report["views_created"]) == ["RPT.V_DEP", "RPT.V_OK", "RPT.V_SUM"]   # V_DEP after V_OK
    assert [f.split(":")[0] for f in report["views_failed"]] == ["RPT.V_BAD"]
    assert float(db.run(f'SELECT "TOTAL" FROM {db.fq("RPT", "V_SUM")}')[0][0]) == 3.5   # text SUM coerced
    assert not db.run(f"SELECT 1 FROM system.tables WHERE database = '{db.bend_db('RPT')}' AND name = 'V_BAD'")

    # unchanged -> nothing to do; a broken new definition -> the old view comes back
    sf.defs[("RPT", "V_OK")] = f"create or replace view {d}.RPT.V_OK as select nope_col from {d}.RAW.T"
    report = {"views_created": [], "views_replaced": [], "views_adopted": 0, "views_failed": []}
    bisync.views_sf_to_db(sf, db, state, report, log=lambda m: None)
    assert report["views_replaced"] == [] and report["views_failed"][0].startswith("RPT.V_OK")
    assert len(db.run(f'SELECT * FROM {db.fq("RPT", "V_OK")}')) == 2       # previous definition restored


def test_bulk_unload_keeps_one_parquet_schema_across_files():
    """Snowflake's per-file integer narrowing (int16 in one file, int32 in another) broke Databend's COPY with
    'Schema at index 2 was different' (V3_SYNC.EVALUATION__PATIENT_META, INVENTORY__BATCH): the unload turns it off
    in the same session, before the COPY INTO @stage."""
    from afya_backends.warehouse.bisync import Bulk, Col

    class Fake:
        def __init__(self, backend):
            self.backend, self.sql = backend, []

        def real(self, schema, table):
            return schema, table

        def columns(self, schema, table):
            return [Col("ID", "NUMBER", 38, 0), Col("NAME", "TEXT", None, None)]

        def fq(self, schema, table):
            return f"{schema}.{table}"

        def run(self, sql, params=None):
            self.sql.append(sql)
            return []

        def ensure_schema(self, schema):
            pass

        def create_table(self, *a, **k):
            pass

        def count(self, *a):
            return 0

        def swap_in(self, *a):
            pass

        def drop(self, *a):
            pass

    sf, db = Fake("snowflake"), Fake("databend")
    bisync.bulk_copy_sf_to_db(sf, db, "V3_SYNC", "T", Bulk("S.A.STAGE", "s3://b/p/", "k", "s"), dst_exists=False,
                              log=lambda m: None)
    alter = next(i for i, s in enumerate(sf.sql) if "ENABLE_UNLOAD_PHYSICAL_TYPE_OPTIMIZATION = FALSE" in s)
    copy = next(i for i, s in enumerate(sf.sql) if s.startswith("COPY INTO @"))
    assert alter < copy


def test_snowflake_reads_guard_timestamps_outside_the_nanosecond_range():
    """V3 has dates like LMP '0026-06-10' / EXPIRY_DATE '0229-05-30': the Snowflake connector then returns that
    result chunk as timestamp[us] next to timestamp[ns] ones and pyarrow fails with 'Schema at index N was different'
    (V3_SYNC.EVALUATION__PATIENT_META, INVENTORY__BATCH, 2026-11-18). Snowflake reads return such values as NULL."""
    from afya_backends.warehouse.bisync import Col, Side

    assert bisync._sf_select(Col("LMP", "TIMESTAMP")) == 'IFF(YEAR("LMP") BETWEEN 1678 AND 2261, "LMP", NULL) AS "LMP"'
    assert bisync._sf_select(Col("ID", "NUMBER", 38, 0)) == '"ID"'
    assert bisync._sf_select("NAME") == '"NAME"'

    class Cur:
        def __init__(self, log):
            self.log = log

        def execute(self, sql, params=None):
            self.log.append(sql)

        def fetch_pandas_batches(self):
            return iter(())

    class Conn:
        def __init__(self):
            self.sql = []

        def cursor(self):
            return Cur(self.sql)

    for backend in ("snowflake", "databend"):
        side = Side.__new__(Side)
        side.backend, side.database, side.conn = backend, "HOSPITALS", Conn()
        list(side.read_batches("V3_SYNC", "T", [Col("ID", "NUMBER", 38, 0), Col("LMP", "TIMESTAMP")]))
        sql = side.conn.sql[0]
        if backend == "snowflake":
            assert 'IFF(YEAR("LMP") BETWEEN 1678 AND 2261, "LMP", NULL) AS "LMP"' in sql and sql.startswith('SELECT "ID", ')
        else:                                   # Databend reads are untouched (names only)
            assert sql.startswith('SELECT "ID", "LMP" FROM')
