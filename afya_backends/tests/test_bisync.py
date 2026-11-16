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
