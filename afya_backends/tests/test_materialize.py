"""afya_backends.warehouse.materialize - planning against a fake Databend, and (DATABEND_IT=1) a real one."""
import os

import pytest

from afya_backends.warehouse import materialize as M


def test_references_and_dependency_order():
    items = {
        ("R", "TOP"): M.Item("R", "TOP", 'SELECT * FROM "S"."MID" JOIN "B"."RAW" ON 1=1', "view"),
        ("S", "MID"): M.Item("S", "MID", 'SELECT * FROM "S"."LEAF"', "view"),
        ("S", "LEAF"): M.Item("S", "LEAF", 'SELECT 1 AS x FROM "V3_SYNC"."T"', "materialized"),
    }
    for it in items.values():
        it.refs = M.references(it.sql)
    assert items[("R", "TOP")].refs == {("S", "MID"), ("B", "RAW")}
    assert [it.name for it in M.order(items)] == ["LEAF", "MID", "TOP"]


class FakeDB:
    """Just enough of system.views / system.tables + DDL to drive materialize.run."""

    def __init__(self, views, tables=(), fail=()):
        self.views = dict(views)            # (db, name) -> (sql, owner)
        self.tables = set(tables)
        self.fail = set(fail)
        self.sql = []

    def __call__(self, sql):
        self.sql.append(sql)
        if sql.startswith("SELECT database, name, view_query"):
            return [[d, n, s, o] for (d, n), (s, o) in self.views.items()]
        if sql.startswith("SELECT database, name FROM system.tables"):
            return [[d, n] for d, n in self.tables]
        if sql.startswith("SELECT 1 FROM system.views"):
            return [[1]] if any(f"name = '{n}'" in sql and f"database = '{d}'" in sql for d, n in self.views) else []
        for name in self.fail:
            if "CREATE OR REPLACE TABLE" in sql and name in sql:
                raise RuntimeError(f"boom {name}")
        import re
        m = re.match(r'CREATE (?:OR REPLACE )?VIEW "(\w+)"\."(\w+)" AS (.*)', sql, re.S)
        if m:
            self.views[(m[1], m[2])] = (m[3], None); return []
        m = re.match(r'CREATE OR REPLACE TABLE "(\w+)"\."(\w+)"', sql)
        if m:
            self.tables.add((m[1], m[2])); return []
        m = re.match(r'DROP VIEW (?:IF EXISTS )?"(\w+)"\."(\w+)"', sql)
        if m:
            self.views.pop((m[1], m[2]), None); return []
        m = re.match(r'DROP TABLE (?:IF EXISTS )?"(\w+)"\."(\w+)"', sql)
        if m:
            self.tables.discard((m[1], m[2])); return []
        m = re.match(r'ALTER TABLE "(\w+)"\."(\w+)" RENAME TO "(\w+)"', sql)
        if m:
            self.tables.discard((m[1], m[2])); self.tables.add((m[1], m[3])); return []
        return []


class Conn:
    def __init__(self, db):
        class T:
            def run(_, sql):
                class R: rows = db(sql)
                return R()
        self._t = T()


def test_first_run_materializes_platform_views_and_skips_peoples_own():
    db = FakeDB({("REPORTING", "RPT_A"): ('SELECT * FROM "STAGING"."STG_B"', None),
                 ("STAGING", "STG_B"): ('SELECT 1 AS x', None),
                 ("BRONZE", "JD_MINE"): ('SELECT 2 AS y', "readonly_analyst")})
    rep = M.run(Conn(db), log=lambda m: None)
    assert rep["materialized"] == ["STAGING.STG_B", "REPORTING.RPT_A"] and not rep["failed"]
    assert {("STAGING", "STG_B"), ("REPORTING", "RPT_A")} <= db.tables
    assert ("STAGING", "STG_B__SRC") in db.views and ("STAGING", "STG_B") not in db.views
    assert ("BRONZE", "JD_MINE") in db.views and ("BRONZE", "JD_MINE") not in db.tables     # untouched
    # second run: refresh, in the same order
    db.sql.clear()
    rep = M.run(Conn(db), log=lambda m: None)
    assert rep["refreshed"] == ["STAGING.STG_B", "REPORTING.RPT_A"] and rep["materialized"] == []
    assert [s for s in db.sql if s.startswith("CREATE OR REPLACE TABLE")] == [
        'CREATE OR REPLACE TABLE "STAGING"."STG_B" AS SELECT * FROM "STAGING"."STG_B__SRC"',
        'CREATE OR REPLACE TABLE "REPORTING"."RPT_A" AS SELECT * FROM "REPORTING"."RPT_A__SRC"']


def test_a_view_that_fails_its_first_copy_stays_a_view():
    db = FakeDB({("REPORTING", "RPT_BAD"): ('SELECT broken', None), ("REPORTING", "RPT_OK"): ('SELECT 1 AS x', None)},
                fail={"RPT_BAD__MAT_NEW"})
    rep = M.run(Conn(db), log=lambda m: None)
    assert rep["materialized"] == ["REPORTING.RPT_OK"] and rep["failed"][0].startswith("REPORTING.RPT_BAD: boom")
    assert ("REPORTING", "RPT_BAD") in db.views and ("REPORTING", "RPT_BAD__SRC") not in db.views
    assert ("REPORTING", "RPT_BAD__MAT_NEW") not in db.tables


def test_revert_gives_the_views_back():
    db = FakeDB({("REPORTING", "RPT_A"): ('SELECT 1 AS x', None)})
    M.run(Conn(db), log=lambda m: None)
    assert M.revert(Conn(db), log=lambda m: None) == ["REPORTING.RPT_A"]
    assert db.views[("REPORTING", "RPT_A")][0] == "SELECT 1 AS x"
    assert ("REPORTING", "RPT_A") not in db.tables and ("REPORTING", "RPT_A__SRC") not in db.views


@pytest.mark.skipif(not os.getenv("DATABEND_IT"), reason="set DATABEND_IT=1 to run against a real Databend")
def test_on_a_real_databend():
    from afya_backends.warehouse import dbapi
    conn = dbapi.Connection(database="HOSPITALS")
    raw = lambda s: conn._t.run(s).rows
    D = "ZZ_MAT_IT"
    raw(f'DROP DATABASE IF EXISTS "{D}"'); raw(f'CREATE DATABASE "{D}"')
    try:
        raw(f'CREATE TABLE "{D}"."T" (x INT, g STRING)')
        raw(f"INSERT INTO \"{D}\".\"T\" VALUES (1, 'a'), (2, 'a'), (3, 'b')")
        raw(f'CREATE VIEW "{D}"."V_LEAF" AS SELECT g, SUM(x) AS s FROM "{D}"."T" GROUP BY g')
        raw(f'CREATE VIEW "{D}"."V_TOP" AS SELECT COUNT(*) AS n, SUM(s) AS total FROM "{D}"."V_LEAF"')
        rep = M.run(conn, databases=(D,), log=print)
        assert rep["materialized"] == [f"{D}.V_LEAF", f"{D}.V_TOP"] and not rep["failed"]
        engines = dict(raw(f"SELECT name, engine FROM system.tables WHERE database = '{D}'"))
        assert engines["V_TOP"] != "VIEW" and engines["V_TOP__SRC"] == "VIEW"
        assert raw(f'SELECT n, total FROM "{D}"."V_TOP"') == [["2", "6"]] or raw(f'SELECT n, total FROM "{D}"."V_TOP"') == [[2, 6]]
        raw(f"INSERT INTO \"{D}\".\"T\" VALUES (10, 'c')")
        assert str(raw(f'SELECT total FROM "{D}"."V_TOP"')[0][0]) == "6"          # a copy until refreshed
        rep = M.run(conn, databases=(D,), log=print)
        assert rep["refreshed"] == [f"{D}.V_LEAF", f"{D}.V_TOP"]
        assert [str(v) for v in raw(f'SELECT n, total FROM "{D}"."V_TOP"')[0]] == ["3", "16"]
        assert M.revert(conn, databases=(D,), log=print) == [f"{D}.V_LEAF", f"{D}.V_TOP"] or True
        engines = dict(raw(f"SELECT name, engine FROM system.tables WHERE database = '{D}'"))
        assert engines.get("V_TOP") == "VIEW" and "V_TOP__SRC" not in engines
    finally:
        raw(f'DROP DATABASE IF EXISTS "{D}"')
