"""
v3_to_snowflake_sync against REAL Snowflake: the MERGE / delete-flag / state SQL and write_pandas staging.

Opt-in (needs the .env's Snowflake credentials and CREATE SCHEMA on HOSPITALS):

    V3SYNC_SNOWFLAKE_IT=1 python -m pytest tests/test_v3_to_snowflake_sync_snowflake.py -v

Everything happens in a throwaway schema HOSPITALS.V3_SYNC_IT_<random>, dropped at the end. V3 is faked
(tests/v3sync_fakes.py), so no gateway is touched.
"""
from __future__ import annotations

import os
import uuid
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

import v3_to_snowflake_sync as S
from tests.v3sync_fakes import FakeV3Client, patient

pytestmark = pytest.mark.skipif(os.getenv("V3SYNC_SNOWFLAKE_IT") != "1",
                                reason="set V3SYNC_SNOWFLAKE_IT=1 to run against real Snowflake")

PATIENT = S.Model("reception", "patient")
T = PATIENT.table
NOW = datetime(2026, 10, 29, 12, 0, 0)


@pytest.fixture(scope="module")
def sink():
    from dotenv import load_dotenv
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")
    os.chdir(root)                                   # SNOWFLAKE_PRIVATE_KEY_PATH is relative to the repo
    schema = f"V3_SYNC_IT_{uuid.uuid4().hex[:8].upper()}"
    conn = S.snowflake_connect("HOSPITALS", schema)
    s = S.SnowflakeSink(conn, "HOSPITALS", schema)
    s.ensure_schema()
    try:
        yield s
    finally:
        conn.cursor().execute(f"DROP SCHEMA IF EXISTS HOSPITALS.{schema}")
        conn.close()


def run(client, sink, model=PATIENT, now=NOW, **opts):
    r = S.sync_model(client, sink, model, source="it", opts=S.SyncOptions(**opts), clock=lambda: now)
    assert r.status != "failed", r.error
    return r


def q(sink, sql, params=None):
    return sink._exec(sql.format(fq=sink.fq), params)


def test_full_lifecycle(sink):
    client = FakeV3Client({("reception", "patient"): [
        patient(i, f"2026-10-{i:02d}T08:00:00.000000Z", address={"town": f"T{i}"}) for i in range(1, 11)]})

    # 1. first load
    r = run(client, sink)
    assert (r.read, r.sent, r.inserted, r.updated) == (10, 10, 10, 0)
    (n, first, town, upd, ident), = q(sink, "SELECT COUNT(*), MAX(FIRST_NAME), "
                                            "MAX(IFF(ID = 3, ADDRESS:town::string, NULL)), MAX(UPDATED_AT), "
                                            "MAX(IFF(ID = 3, UUID, NULL)) FROM {fq}.RECEPTION__PATIENT")
    assert (n, first, town, upd) == (10, "P9", "T3", datetime(2026, 10, 10, 8))
    assert ident == "00000000-0000-0000-0000-000000000003"
    assert sink.get_watermark(6, PATIENT.name) == datetime(2026, 10, 10, 8)
    loaded_at = q(sink, "SELECT _KEY, _LAST_LOADED_AT FROM {fq}.RECEPTION__PATIENT")

    # 2. nothing changed: rows in the lookback window are sent but nothing is rewritten
    r = run(client, sink)
    assert (r.sent, r.inserted, r.updated) == (1, 0, 0)
    assert q(sink, "SELECT _KEY, _LAST_LOADED_AT FROM {fq}.RECEPTION__PATIENT") == loaded_at

    # 3. one edit, one new row
    client.edit("reception", "patient", 3, first_name="Edited", updated_at="2026-10-20T09:00:00.000000Z")
    client.rows("reception", "patient").append(patient(11, "2026-10-21T10:00:00.000000Z"))
    r = run(client, sink)
    assert (r.inserted, r.updated) == (1, 1)
    assert q(sink, "SELECT FIRST_NAME, UPDATED_AT FROM {fq}.RECEPTION__PATIENT WHERE ID = 3") == \
        [("Edited", datetime(2026, 10, 20, 9))]
    assert sink.get_watermark(6, PATIENT.name) == datetime(2026, 10, 21, 10)

    # 4. deleted in V3 → flagged; back again → un-flagged
    gone = client.delete("reception", "patient", 5)
    r = run(client, sink)
    assert r.deleted == 1
    assert q(sink, "SELECT _IS_DELETED, _DELETED_AT IS NOT NULL FROM {fq}.RECEPTION__PATIENT WHERE ID = 5") == \
        [(True, True)]
    client.rows("reception", "patient").append(gone)
    r = run(client, sink)
    assert (r.deleted, r.restored) == (0, 1)
    assert q(sink, "SELECT COUNT_IF(_IS_DELETED) FROM {fq}.RECEPTION__PATIENT") == [(0,)]

    # 5. the guard: half the rows vanish at once → nothing flagged, reason recorded
    saved = [client.delete("reception", "patient", i) for i in range(1, 7)]
    r = run(client, sink)
    assert r.deleted == 0 and "guard" in r.note
    assert q(sink, "SELECT LAST_STATUS, LAST_ERROR LIKE '%guard%', ROWS_READ FROM {fq}._SYNC_STATE "
                   "WHERE MODEL = 'reception:patient'") == [("ok", True, 5)]
    client.rows("reception", "patient").extend(saved)

    # 6. full refresh after an edit that didn't move updated_at
    client.edit("reception", "patient", 7, first_name="Fixed in SQL")
    assert run(client, sink).updated == 0
    r = run(client, sink, full_refresh=True)
    assert (r.sent, r.inserted, r.updated) == (11, 0, 1)


def test_failure_is_recorded_and_watermark_kept(sink):
    model = S.Model("finance", "invoice")
    client = FakeV3Client({("finance", "invoice"): [
        {"id": i, "amount": i * 100, "updated_at": f"2026-10-{i:02d}T00:00:00Z"} for i in range(1, 9)]})
    run(client, sink, model=model)
    client.edit("finance", "invoice", 8, amount=1, updated_at="2026-10-25T00:00:00Z")
    client.fail_on, client.fail_after = {("finance", "invoice")}, 4
    r = S.sync_model(client, sink, model, source="it", clock=lambda: NOW)
    assert r.status == "failed"
    (status, err, wm), = q(sink, "SELECT LAST_STATUS, LAST_ERROR, WATERMARK FROM {fq}._SYNC_STATE "
                                 "WHERE MODEL = 'finance:invoice'")
    assert status == "failed" and "gateway 500" in err and wm == datetime(2026, 10, 8)
    client.fail_on = set()
    r = run(client, sink, model=model)
    assert r.updated == 1 and sink.get_watermark(6, model.name) == datetime(2026, 10, 25)


def test_tenants_share_a_table_without_touching_each_other(sink):
    model = S.Model("core", "department")
    a = FakeV3Client({("core", "department"): [{"id": i, "name": f"A{i}"} for i in range(1, 4)]}, organization_id=101)
    b = FakeV3Client({("core", "department"): [{"id": i, "name": f"B{i}"} for i in range(1, 4)]}, organization_id=102)
    run(a, sink, model=model)
    run(b, sink, model=model)
    b.delete("core", "department", 1)
    run(b, sink, model=model, max_delete_fraction=1)
    assert q(sink, "SELECT _TENANT_ID, COUNT(*), COUNT_IF(_IS_DELETED) FROM {fq}.CORE__DEPARTMENT "
                   "GROUP BY 1 ORDER BY 1") == [(101, 3, 0), (102, 3, 1)]


def test_rows_without_ids_and_duplicate_keys_in_one_batch(sink):
    model = S.Model("core", "model_has_role")
    client = FakeV3Client({("core", "model_has_role"): [{"role_id": 1, "model_id": 9}, {"role_id": 2, "model_id": 9},
                                                        {"role_id": 1, "model_id": 9}]})   # an exact duplicate
    r = run(client, sink, model=model)
    assert r.inserted == 2                                    # the duplicate collapses in the staging QUALIFY
    assert q(sink, "SELECT COUNT(*), COUNT_IF(_KEY LIKE 'hash:%'), MIN(ROLE_ID), MAX(ROLE_ID) "
                   "FROM {fq}.CORE__MODEL_HAS_ROLE") == [(2, 2, 1, 2)]


def test_many_batches(sink):
    model = S.Model("reception", "visit")
    client = FakeV3Client({("reception", "visit"): [
        {"id": i, "patient": i % 997, "updated_at": f"2026-10-{1 + i % 28:02d}T00:00:00Z",
         "notes": "x" * (i % 50)} for i in range(1, 12_346)]})
    r = run(client, sink, model=model, batch_size=5000)
    assert (r.read, r.inserted) == (12_345, 12_345)
    assert q(sink, "SELECT COUNT(*), COUNT(DISTINCT _KEY), MAX(ID) FROM {fq}.RECEPTION__VISIT") == \
        [(12_345, 12_345, 12_345)]
    r = run(client, sink, model=model, full_refresh=True, batch_size=5000)
    assert (r.sent, r.inserted, r.updated) == (12_345, 0, 0)


def test_table_is_shaped_like_v3(sink):
    model = S.Model("finance", "payment")
    client = FakeV3Client({("finance", "payment"): [
        {"id": i, "key": f"K{i}", "order": i * 10, "amount": f"{i}50.25", "rate": 1.5, "paid": i % 2 == 0,
         "phone": "0712000000", "meta": {"till": i}, "paid_on": "2026-10-01",
         "created_at": "2026-10-01T05:00:00.000000Z", "updated_at": f"2026-10-0{i}T05:00:00.000000Z"}
        for i in range(1, 4)]})
    run(client, sink, model=model)
    cols = q(sink, "SELECT COLUMN_NAME, DATA_TYPE, NUMERIC_SCALE FROM HOSPITALS.INFORMATION_SCHEMA.COLUMNS "
                   "WHERE TABLE_SCHEMA = %(s)s AND TABLE_NAME = 'FINANCE__PAYMENT' ORDER BY ORDINAL_POSITION",
             {"s": sink.schema})
    names = [c[0] for c in cols]
    assert names[:11] == ["ID", "KEY", "ORDER", "AMOUNT", "RATE", "PAID", "PHONE", "META", "PAID_ON",
                          "CREATED_AT", "UPDATED_AT"]                     # V3's columns, V3's order
    assert names[11] == "_TENANT_ID" and names[-1] == "_RAW"
    types = {c[0]: (c[1], c[2]) for c in cols}
    assert types["ID"] == ("NUMBER", 0) and types["AMOUNT"] == ("NUMBER", 6) and types["RATE"][0] == "FLOAT"
    assert types["PAID"][0] == "BOOLEAN" and types["PHONE"][0] == "TEXT" and types["META"][0] == "VARIANT"
    assert types["PAID_ON"][0] == "DATE" and types["UPDATED_AT"][0] == "TIMESTAMP_NTZ"
    assert q(sink, 'SELECT "KEY", "ORDER", AMOUNT, RATE, PAID, PHONE, META:till::int, PAID_ON, CREATED_AT '
                   'FROM {fq}.FINANCE__PAYMENT WHERE ID = 2') == \
        [("K2", 20, Decimal("250.250000"), 1.5, True, "0712000000", 2, date(2026, 10, 1), datetime(2026, 10, 1, 5))]

    # V3 adds a column → the table gets it; a value that doesn't fit its type → NULL there, kept in _RAW
    client.rows("finance", "payment").append(
        {"id": 4, "key": "K4", "order": "not a number", "amount": "9.99", "rate": 1.0, "paid": True,
         "phone": "0700", "meta": None, "paid_on": None, "created_at": None,
         "updated_at": "2026-10-09T00:00:00Z", "receipt_no": "R-4"})
    assert run(client, sink, model=model).inserted == 1
    assert q(sink, 'SELECT "ORDER", _RAW:"order"::string, RECEIPT_NO, META FROM {fq}.FINANCE__PAYMENT '
                   'WHERE ID = 4') == [(None, "not a number", "R-4", None)]
    assert q(sink, "SELECT COUNT(RECEIPT_NO) FROM {fq}.FINANCE__PAYMENT") == [(1,)]


def test_empty_columns_get_their_type_when_values_arrive(sink):
    model = S.Model("finance", "refund")
    client = FakeV3Client({("finance", "refund"): [
        {"id": i, "approved_by": None, "approved_at": None, "reason": "x", "updated_at": "2026-10-01T00:00:00Z"}
        for i in range(1, 4)]})
    run(client, sink, model=model)

    def types():
        return dict(q(sink, "SELECT COLUMN_NAME, DATA_TYPE FROM HOSPITALS.INFORMATION_SCHEMA.COLUMNS "
                            "WHERE TABLE_SCHEMA = %(s)s AND TABLE_NAME = 'FINANCE__REFUND'", {"s": sink.schema}))
    assert types()["APPROVED_BY"] == "TEXT" and types()["APPROVED_AT"] == "TIMESTAMP_NTZ"
    client.edit("finance", "refund", 2, approved_by=7, approved_at="2026-10-05T08:00:00Z",
                updated_at="2026-10-05T08:00:00Z")
    assert run(client, sink, model=model).updated == 1
    assert types()["APPROVED_BY"] == "NUMBER"
    assert q(sink, "SELECT ID, APPROVED_BY, APPROVED_AT FROM {fq}.FINANCE__REFUND WHERE APPROVED_BY IS NOT NULL") \
        == [(2, 7, datetime(2026, 10, 5, 8))]
    client.edit("finance", "refund", 3, reason=None, updated_at="2026-10-06T00:00:00Z")
    client.edit("finance", "refund", 1, reason=5, updated_at="2026-10-06T00:00:00Z")
    run(client, sink, model=model)
    assert types()["REASON"] == "TEXT"                         # holds data: never retyped
    assert q(sink, "SELECT REASON FROM {fq}.FINANCE__REFUND ORDER BY ID") == [("5",), ("x",), (None,)]


def test_model_with_no_rows_still_gets_a_table(sink):
    model = S.Model("core", "empty_thing")
    r = run(FakeV3Client({("core", "empty_thing"): []}), sink, model=model)
    assert (r.status, r.read) == ("ok", 0)
    assert q(sink, "SELECT COUNT(*) FROM {fq}.CORE__EMPTY_THING") == [(0,)]
