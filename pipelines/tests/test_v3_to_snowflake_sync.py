"""
v3_to_snowflake_sync — the watermark / change-selection / delete / failure logic, against in-memory fakes
(tests/v3sync_fakes.py). The Snowflake SQL itself is exercised for real in
test_v3_to_snowflake_sync_snowflake.py (opt-in).
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta

import pytest

import v3_to_snowflake_sync as S
from tests.v3sync_fakes import FakeSink, FakeV3Client, patient

PATIENT = S.Model("reception", "patient")
T = "RECEPTION__PATIENT"
NOW = datetime(2026, 10, 29, 12, 0, 0)


@pytest.fixture(autouse=True)
def _no_retry_waits(monkeypatch):
    monkeypatch.setattr(S, "RETRY_WAITS", (0, 0))


def run(client, sink, model=PATIENT, now=NOW, **opts):
    return S.sync_model(client, sink, model, source="gyno", opts=S.SyncOptions(**opts), clock=lambda: now)


def tenant_with(n: int, day0: int = 1) -> FakeV3Client:
    """n patients, patient i last edited on October (day0 + i - 1) 2026."""
    return FakeV3Client({("reception", "patient"): [
        patient(i, f"2026-10-{day0 + i - 1:02d}T08:00:00.000000Z") for i in range(1, n + 1)]})


# ── pure helpers ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("2026-07-20T21:00:00.000000Z", datetime(2026, 7, 20, 21, 0)),
    ("2026-07-20T21:00:00Z", datetime(2026, 7, 20, 21, 0)),
    ("2026-07-21T00:30:00+03:00", datetime(2026, 7, 20, 21, 30)),        # converted to UTC
    ("2026-07-20 21:00:00", datetime(2026, 7, 20, 21, 0)),                # no zone: taken as UTC
    ("2026-07-20", datetime(2026, 7, 20)),
    (datetime(2026, 7, 20, 21, 0), datetime(2026, 7, 20, 21, 0)),
    (None, None), ("", None), ("   ", None), ("0000-00-00 00:00:00", None), ("not a date", None),
])
def test_parse_ts(raw, expected):
    assert S.parse_ts(raw) == expected


def test_row_hash_ignores_key_order_but_not_values():
    a = {"id": 1, "name": "A", "nested": {"x": 1, "y": [1, 2]}}
    b = {"nested": {"y": [1, 2], "x": 1}, "name": "A", "id": 1}
    assert S.row_hash(a) == S.row_hash(b)
    assert S.row_hash(a) != S.row_hash({**a, "name": "B"})
    assert S.row_hash(a) != S.row_hash({**a, "nested": {"x": 1, "y": [2, 1]}})


def test_row_key_prefers_id_then_uuid_then_content():
    assert S.row_key({"id": 42, "uuid": "u"}) == "42"
    assert S.row_key({"id": 0}) == "0"
    assert S.row_key({"id": None, "uuid": "abc"}) == "uuid:abc"
    no_ids = {"role_id": 1, "model_id": 9}
    assert S.row_key(no_ids) == f"hash:{S.row_hash(no_ids)}"


def test_table_and_model_names():
    assert S.table_name("reception", "patient") == "RECEPTION__PATIENT"
    assert S.table_name("evaluation", "lab-test.units") == "EVALUATION__LAB_TEST_UNITS"
    assert S.Model("core", "users").table != S.Model("dialysis", "users").table
    assert S.parse_model(" Reception:patient ") == PATIENT
    with pytest.raises(ValueError):
        S.parse_model("patient")


def test_parse_models_accepts_lists_and_pasted_strings():
    want = [S.Model("finance", "invoice"), PATIENT]
    assert S.parse_models(["reception:patient", "finance:invoice"]) == want
    assert S.parse_models('["reception:patient", "finance:invoice"]') == want
    assert S.parse_models(["reception:patient,finance:invoice reception:patient"]) == want
    assert S.parse_models([]) == [] and S.parse_models(None) == []


@pytest.mark.parametrize("updated, since, changed", [
    (datetime(2026, 10, 2), None, True),                     # first / full run: everything
    (None, datetime(2026, 10, 1), True),                     # no updated_at: can't tell, send it
    (datetime(2026, 10, 1), datetime(2026, 10, 1), True),    # boundary is inclusive
    (datetime(2026, 9, 30), datetime(2026, 10, 1), False),
])
def test_is_changed(updated, since, changed):
    assert S.is_changed(updated, since) is changed


def test_next_watermark_caps_at_read_start_and_never_goes_back():
    start = datetime(2026, 10, 29, 12)
    assert S.next_watermark(None, datetime(2026, 10, 5), start) == datetime(2026, 10, 5)
    assert S.next_watermark(None, datetime(2026, 10, 30), start) == start          # edited mid-read / clock ahead
    assert S.next_watermark(datetime(2026, 10, 10), datetime(2026, 10, 5), start) == datetime(2026, 10, 10)
    assert S.next_watermark(datetime(2026, 10, 10), None, start) == datetime(2026, 10, 10)
    assert S.next_watermark(None, None, start) is None


def test_staged_row_carries_the_whole_row():
    row = patient(7, nested={"a": [1, 2]})
    s = S.StagedRow.from_row(row)
    assert s.key == "7" and s.row is row and s.updated_at == "2026-10-01 08:00:00.000000"
    assert '"nested": {"a": [1, 2]}' in s.data and s.row_hash == S.row_hash(row)
    assert s.as_tuple() == ("7", s.updated_at, s.row_hash, s.data)


# ── tables shaped like V3 ─────────────────────────────────────────────────

@pytest.mark.parametrize("value, sf_type", [
    (None, None), (True, S.BOOLEAN), (False, S.BOOLEAN), (42, S.INT), (0, S.INT), (1.5, S.FLOAT),
    ("1500.00", S.DECIMAL), ("-0.25", S.DECIMAL),
    ("0712345678", S.STRING),                                   # phone: keeps its leading zero
    ("12", S.STRING), ("abc", S.STRING), ("", S.STRING),
    ("2026-07-20T21:00:00.000000Z", S.TIMESTAMP), ("2026-07-20 21:00:00", S.TIMESTAMP),
    ("2026-07-20T21:00:00+03:00", S.TIMESTAMP), ("2026-07-20", S.DATE),
    ("2026-13-45", S.STRING),                                   # looks like a date, isn't one
    ({"a": 1}, S.VARIANT), ([1, 2], S.VARIANT),
])
def test_value_type(value, sf_type):
    assert S.value_type(value) == sf_type


def test_column_types_follow_v3_order_and_widen_mixed_values():
    rows = [{"id": 1, "amount": "10.50", "qty": 1, "rate": 2, "seen": "2026-01-01", "phone": "0712", "meta": None,
             "tags": ["a"], "flag": True, "mixed": 5},
            {"id": 2, "amount": "3.00", "qty": 2.5, "rate": "1.25", "seen": "2026-01-02T10:00:00Z", "phone": "0733",
             "meta": None, "tags": "x", "flag": False, "mixed": "five"}]
    assert S.column_types(rows) == {
        "id": S.INT, "amount": S.DECIMAL, "qty": S.FLOAT, "rate": S.DECIMAL, "seen": S.TIMESTAMP,
        "phone": S.STRING, "meta": S.STRING, "tags": S.VARIANT, "flag": S.BOOLEAN, "mixed": S.STRING}
    assert list(S.column_types(rows)) == list(rows[0])             # V3's column order


def test_all_null_columns_named_like_timestamps_are_timestamps():
    assert S.column_types([{"deleted_at": None, "created_by": None, "insurance_company_id": None}]) == {
        "deleted_at": S.TIMESTAMP, "created_by": S.STRING, "insurance_company_id": S.STRING}


def test_column_names_are_v3_names_upper_cased():
    assert S.column_name("first_name") == "FIRST_NAME"
    assert S.column_name("order") == "ORDER"                       # reserved words: always quoted in SQL
    assert S.column_name("_secret") == "SECRET"                    # never collides with the pipeline's _ columns
    assert not any(S.column_name(n) in S.META_COLUMNS for n in ("_key", "_raw", "tenant_id", "_tenant_id"))


@pytest.mark.parametrize("sf_type, sql", [
    (S.STRING, "TO_VARCHAR(_J['first_name'])"),
    (S.INT, "TRY_TO_NUMBER(TO_VARCHAR(_J['first_name']), 38, 0)"),
    (S.DECIMAL, "TRY_TO_NUMBER(TO_VARCHAR(_J['first_name']), 38, 6)"),
    (S.FLOAT, "TRY_TO_DOUBLE(TO_VARCHAR(_J['first_name']))"),
    (S.BOOLEAN, "TRY_TO_BOOLEAN(TO_VARCHAR(_J['first_name']))"),
    (S.TIMESTAMP, "CONVERT_TIMEZONE('UTC', TRY_TO_TIMESTAMP_TZ(TO_VARCHAR(_J['first_name'])))::TIMESTAMP_NTZ"),
    (S.DATE, "TRY_TO_DATE(TO_VARCHAR(_J['first_name']))"),
    (S.VARIANT, "STRIP_NULL_VALUE(_J['first_name'])"),
])
def test_cast_sql(sf_type, sql):
    assert S._cast("first_name", sf_type) == sql


def test_cast_escapes_quotes_and_percent():
    assert S._cast("it's 100%", S.STRING) == "TO_VARCHAR(_J['it''s 100%%'])"


def test_table_gets_the_v3_columns_and_new_ones_are_added():
    client, sink = tenant_with(3), FakeSink()
    run(client, sink)
    assert list(sink.columns[T]) == ["ID", "UUID", "FIRST_NAME", "ORGANIZATION_ID", "CREATED_AT", "UPDATED_AT",
                                     "DELETED_AT"]
    assert sink.columns[T]["ID"] == S.INT and sink.columns[T]["UPDATED_AT"] == S.TIMESTAMP
    client.rows("reception", "patient").append(patient(4, "2026-10-28T08:00:00Z", blood_group="O+"))
    run(client, sink)
    assert list(sink.columns[T])[-1] == "BLOOD_GROUP"


# ── the real SnowflakeSink's DDL ──────────────────────────────────────────

def test_sink_creates_the_table_v3_columns_first_then_pipeline_columns():
    conn = RecordingConn(answers=[[], [], [("ID", "NUMBER", 38, 0), ("NAME", "TEXT", None, None)]])
    sink = S.SnowflakeSink(conn, "hospitals", "v3_sync")
    cols = sink.ensure_table(T, {"id": S.INT, "name": S.STRING})
    create = conn.sql[1][0]
    assert create.startswith('CREATE TABLE IF NOT EXISTS HOSPITALS.V3_SYNC.RECEPTION__PATIENT ("ID" NUMBER(38,0), '
                             '"NAME" STRING, _TENANT_ID NUMBER NOT NULL, _KEY STRING NOT NULL')
    assert create.endswith("_RAW VARIANT, PRIMARY KEY (_TENANT_ID, _KEY))")
    assert cols == {"ID": "NUMBER(38,0)", "NAME": S.STRING}


def test_sink_adds_only_missing_columns_and_caches():
    conn = RecordingConn(answers=[[("ID", "NUMBER", 38, 0), ("_KEY", "TEXT", None, None)]])
    sink = S.SnowflakeSink(conn)
    sink.ensure_table(T, {"id": S.INT, "blood_group": S.STRING})
    sink.ensure_table(T, {"id": S.INT, "blood_group": S.STRING})
    alters = [s for s, _ in conn.sql if s.startswith("ALTER")]
    assert alters == ['ALTER TABLE HOSPITALS.V3_SYNC.RECEPTION__PATIENT ADD COLUMN IF NOT EXISTS "BLOOD_GROUP" STRING']
    assert sum("INFORMATION_SCHEMA" in s for s, _ in conn.sql) == 1


def test_sink_retypes_a_string_column_only_while_it_is_empty():
    conn = RecordingConn(answers=[[("ID", "NUMBER", 38, 0), ("CREATED_BY", "TEXT", None, None),
                                   ("NOTES", "TEXT", None, None)],
                                  [(0,)], [], [],          # CREATED_BY: empty → drop + add as NUMBER
                                  [(12,)]])                # NOTES: has data → left alone
    sink = S.SnowflakeSink(conn)
    cols = sink.ensure_table(T, {"id": S.INT, "created_by": S.INT, "notes": S.INT})
    stmts = [s for s, _ in conn.sql]
    assert 'ALTER TABLE HOSPITALS.V3_SYNC.RECEPTION__PATIENT DROP COLUMN "CREATED_BY"' in stmts
    assert 'ALTER TABLE HOSPITALS.V3_SYNC.RECEPTION__PATIENT ADD COLUMN "CREATED_BY" NUMBER(38,0)' in stmts
    assert not any('DROP COLUMN "NOTES"' in s for s in stmts)
    assert cols == {"ID": "NUMBER(38,0)", "NOTES": S.STRING, "CREATED_BY": S.INT}   # retyped column moves last
    n = len(conn.sql)
    sink.ensure_table(T, {"notes": S.INT})                       # NOTES known to hold data: no re-check
    assert len(conn.sql) == n


# ── incremental behaviour ─────────────────────────────────────────────────

def test_first_run_loads_everything_and_sets_watermark():
    client, sink = tenant_with(10), FakeSink()
    r = run(client, sink)
    assert (r.status, r.read, r.sent, r.inserted, r.updated, r.deleted) == ("ok", 10, 10, 10, 0, 0)
    assert sink.live_keys(T) == {str(i) for i in range(1, 11)}
    assert r.watermark_before is None and sink.get_watermark(6, PATIENT.name) == datetime(2026, 10, 10, 8)
    assert sink.state[(6, PATIENT.name)]["status"] == "ok"


def test_rerun_with_no_changes_sends_only_the_lookback_window_and_writes_nothing():
    client, sink = tenant_with(10), FakeSink()
    run(client, sink)
    r = run(client, sink, lookback=timedelta(hours=1))
    # only patient 10 (edited at the watermark) falls in [watermark - 1h, ∞); unchanged, so not rewritten
    assert (r.read, r.sent, r.inserted, r.updated) == (10, 1, 0, 0)
    assert r.watermark_after == r.watermark_before == "2026-10-10 08:00:00.000000"


def test_new_and_edited_rows_are_the_only_ones_sent():
    client, sink = tenant_with(10), FakeSink()
    run(client, sink)
    client.edit("reception", "patient", 3, first_name="Edited", updated_at="2026-10-20T09:00:00.000000Z")
    client.rows("reception", "patient").append(patient(11, "2026-10-21T10:00:00.000000Z"))
    r = run(client, sink, lookback=timedelta(minutes=0))
    assert (r.read, r.sent, r.inserted, r.updated) == (11, 3, 1, 1)     # 3, 11, and 10 (at the watermark)
    assert sink.tables[T][(6, "3")]["row"].data.count('"Edited"') == 1
    assert sink.get_watermark(6, PATIENT.name) == datetime(2026, 10, 21, 10)


def test_lookback_catches_a_late_commit_with_an_older_timestamp():
    client, sink = tenant_with(10), FakeSink()
    run(client, sink)                                                     # watermark Oct 10 08:00
    # a transaction that committed after our read but stamped 30 min before the watermark
    client.edit("reception", "patient", 4, first_name="Late", updated_at="2026-10-10T07:30:00.000000Z")
    missed = run(client, sink, lookback=timedelta(minutes=0))
    assert missed.updated == 0                                            # no overlap: lost
    client.edit("reception", "patient", 4, first_name="Late2")
    caught = run(client, sink, lookback=timedelta(hours=1))
    assert caught.updated == 1


def test_edit_that_does_not_move_updated_at_needs_full_refresh():
    client, sink = tenant_with(10), FakeSink()
    run(client, sink)
    client.edit("reception", "patient", 2, first_name="Fixed in SQL")     # updated_at untouched
    assert run(client, sink).updated == 0
    r = run(client, sink, full_refresh=True)
    assert (r.sent, r.inserted, r.updated) == (10, 0, 1)                  # everything sent, only the edit written
    assert sink.get_watermark(6, PATIENT.name) == datetime(2026, 10, 10, 8)


def test_row_edited_during_the_read_is_not_skipped_next_time():
    client, sink = tenant_with(3), FakeSink()
    # patient 3's timestamp is after the moment the read started (edited mid-read, or V3's clock is ahead)
    client.edit("reception", "patient", 3, updated_at="2026-10-29T12:05:00.000000Z")
    run(client, sink, now=NOW)
    assert sink.get_watermark(6, PATIENT.name) == NOW                     # capped at read start, not 12:05


def test_rows_without_updated_at_are_always_sent_but_never_churn():
    client = FakeV3Client({("core", "department"): [{"id": i, "name": f"D{i}"} for i in range(1, 6)]})
    sink, dept = FakeSink(), S.Model("core", "department")
    first = run(client, sink, model=dept)
    second = run(client, sink, model=dept)
    assert (first.inserted, second.sent, second.inserted, second.updated) == (5, 5, 0, 0)
    assert sink.get_watermark(6, dept.name) is None
    client.edit("core", "department", 2, name="Renamed")
    assert run(client, sink, model=dept).updated == 1


def test_rows_without_ids_are_keyed_by_uuid_or_content():
    pivot = S.Model("core", "model_has_role")
    client = FakeV3Client({("core", "model_has_role"): [{"role_id": 1, "model_id": 9}, {"role_id": 2, "model_id": 9}]})
    sink = FakeSink()
    assert run(client, sink, model=pivot).inserted == 2
    client.tables[("core", "model_has_role")][0]["role_id"] = 3            # content-keyed: a new key, old one gone
    r = run(client, sink, model=pivot, max_delete_fraction=1)
    assert (r.inserted, r.updated, r.deleted) == (1, 0, 1)


# ── batching ──────────────────────────────────────────────────────────────

def test_rows_go_in_batches_of_batch_size():
    client, sink = tenant_with(10), FakeSink()
    r = run(client, sink, batch_size=3)
    assert [n for _, n in sink.merges] == [3, 3, 3, 1] and r.inserted == 10


# ── deletes ───────────────────────────────────────────────────────────────

def test_rows_v3_stops_returning_are_flagged_and_restored_when_they_return():
    client, sink = tenant_with(10), FakeSink()
    run(client, sink)
    gone = client.delete("reception", "patient", 5)
    r = run(client, sink)
    assert r.deleted == 1 and sink.deleted_keys(T) == {"5"} and len(sink.live_keys(T)) == 9
    client.rows("reception", "patient").append(gone)                      # restored in V3, timestamp unchanged
    r = run(client, sink)
    assert (r.deleted, r.restored) == (0, 1) and sink.deleted_keys(T) == set()


def test_delete_guard_refuses_a_mass_delete_but_still_moves_on():
    client, sink = tenant_with(10), FakeSink()
    run(client, sink)
    for i in range(1, 6):
        client.delete("reception", "patient", i)                          # half the rows "vanish"
    client.rows("reception", "patient").append(patient(11, "2026-10-25T08:00:00.000000Z"))
    r = run(client, sink, max_delete_fraction=0.2)
    assert r.status == "ok" and r.deleted == 0 and "guard" in r.note
    assert sink.deleted_keys(T) == set() and r.inserted == 1
    assert sink.state[(6, PATIENT.name)]["error"] == r.note               # visible in _SYNC_STATE
    assert run(client, sink, max_delete_fraction=0.6).deleted == 5        # when it's real, raise the guard


def test_detect_deletes_off_leaves_flags_alone():
    client, sink = tenant_with(4), FakeSink()
    run(client, sink)
    client.delete("reception", "patient", 1)
    assert run(client, sink, detect_deletes=False).deleted == 0 and sink.deleted_keys(T) == set()


def test_tenants_are_kept_apart():
    sink = FakeSink()
    a, b = tenant_with(3), tenant_with(2)
    b.organization_id = 7
    run(a, sink)
    run(b, sink)
    b.delete("reception", "patient", 1)
    run(b, sink, max_delete_fraction=1)
    assert sink.live_keys(T, 6) == {"1", "2", "3"} and sink.deleted_keys(T, 7) == {"1"}
    assert sink.get_watermark(6, PATIENT.name) != sink.get_watermark(7, PATIENT.name)


# ── failures never lose data ──────────────────────────────────────────────

def test_read_failure_keeps_the_watermark_and_the_next_run_catches_up():
    client, sink = tenant_with(10), FakeSink()
    run(client, sink)
    client.edit("reception", "patient", 2, first_name="X", updated_at="2026-10-22T08:00:00.000000Z")
    client.edit("reception", "patient", 9, first_name="Y", updated_at="2026-10-23T08:00:00.000000Z")
    client.fail_on, client.fail_after = {("reception", "patient")}, 5
    r = run(client, sink, batch_size=1)
    assert r.status == "failed" and "gateway 500" in r.error and r.read == 5
    assert sink.state[(6, PATIENT.name)]["status"] == "failed"
    assert sink.get_watermark(6, PATIENT.name) == datetime(2026, 10, 10, 8)     # not moved
    assert sink.deleted_keys(T) == set()                                        # partial read: no deletes
    client.fail_on = set()
    r = run(client, sink)
    assert r.status == "ok" and r.updated == 1                                  # patient 9 (2 went in before)
    assert sink.get_watermark(6, PATIENT.name) == datetime(2026, 10, 23, 8)


def test_a_transient_snowflake_error_is_retried():
    client, sink = tenant_with(10), FakeSink()
    sink.fail_merge_calls = {2}                                                 # e.g. a 403 on the stage upload
    r = run(client, sink, batch_size=4)
    assert (r.status, r.inserted) == ("ok", 10) and len(sink.merges) == 4      # batch 2 went in on its 2nd try


def test_merge_failure_keeps_the_watermark():
    client, sink = tenant_with(10), FakeSink()
    sink.fail_merge_calls = {2, 3, 4}                                           # batch 2 fails all 3 attempts
    r = run(client, sink, batch_size=4)
    assert r.status == "failed" and "warehouse suspended" in r.error and r.inserted == 4
    assert sink.get_watermark(6, PATIENT.name) is None
    sink.fail_merge_calls = set()
    r = run(client, sink, batch_size=4)
    assert (r.status, r.inserted, r.updated) == ("ok", 6, 0)                    # the first 4 aren't rewritten


def test_limit_is_a_canary_that_moves_nothing():
    client, sink = tenant_with(10), FakeSink()
    r = run(client, sink, limit=3)
    assert (r.status, r.read, r.inserted) == ("partial", 3, 3) and "limit" in r.note
    assert sink.get_watermark(6, PATIENT.name) is None and sink.deleted_keys(T) == set()
    run(client, sink)
    client.delete("reception", "patient", 10)
    assert run(client, sink, limit=3).deleted == 0                               # not a full read: no deletes


def test_dry_run_writes_nothing():
    client, sink = tenant_with(10), FakeSink()
    r = run(client, sink, dry_run=True)
    assert (r.status, r.read, r.sent) == ("dry_run", 10, 10)
    assert sink.merges == [] and sink.tables == {} and sink.state == {}


# ── many models ───────────────────────────────────────────────────────────

def test_sync_runs_every_model_and_isolates_failures():
    client = FakeV3Client({("reception", "patient"): [patient(1), patient(2)],
                           ("finance", "invoice"): [{"id": 1, "updated_at": "2026-10-01T00:00:00Z"}],
                           ("core", "department"): [{"id": 1}]},
                          fail_on={("finance", "invoice")})
    sink = FakeSink()

    @contextmanager
    def factory():
        yield sink

    models = S.discover_models(client, ["reception", "finance", "core"])
    assert [m.name for m in models] == ["core:department", "finance:invoice", "reception:patient"]
    results = S.sync("gyno", models, client=client, sink_factory=factory, workers=3)
    assert [(r.model, r.status) for r in results] == [("core:department", "ok"), ("finance:invoice", "failed"),
                                                      ("reception:patient", "ok")]
    assert sink.ensure_schema_calls == 1
    text = S.summarize(results)
    assert "3 model(s), 1 failed" in text and "ERROR RuntimeError: gateway 500" in text


def test_discover_models_skips_models_the_gateway_wont_read():
    class C(FakeV3Client):
        def models(self, service):
            return [{"alias": "patient", "operations": ["read"]}, {"alias": "audit", "operations": ["insert"]}]
    assert S.discover_models(C(), ["reception"]) == [PATIENT]


# ── the SQL SnowflakeSink sends (shape only; semantics in the Snowflake test) ──

class RecordingConn:
    def __init__(self, answers=()):
        self.sql: list[tuple[str, object]] = []
        self.answers = list(answers)

    def cursor(self):
        conn = self

        class Cur:
            def execute(self, sql, params=None):
                conn.sql.append((" ".join(sql.split()), params))
                self._rows = conn.answers.pop(0) if conn.answers else []

            def fetchall(self):
                return self._rows

            def close(self):
                pass
        return Cur()


def test_sink_merge_is_tenant_scoped_hash_guarded_and_reports_counts(monkeypatch):
    conn = RecordingConn(answers=[[(3, 2)]])
    sink = S.SnowflakeSink(conn, "hospitals", "v3_sync")
    monkeypatch.setattr(sink, "_stage", lambda name, cols, rows: f"HOSPITALS.V3_SYNC._STG_{name}")
    monkeypatch.setattr(sink, "ensure_table", lambda table, cols: {
        "ID": S.INT, "FIRST_NAME": S.STRING, "UPDATED_AT": S.TIMESTAMP, "LEGACY": S.STRING,
        **{c: t for c, t in S.META_COLUMNS.items()}})
    assert sink.merge(T, 6, "gyno", [S.StagedRow.from_row(patient(1))]) == (3, 2)
    sql, params = conn.sql[0]
    assert sql.startswith("MERGE INTO HOSPITALS.V3_SYNC.RECEPTION__PATIENT t")
    assert "t._TENANT_ID = %(t)s AND t._KEY = s._SKEY" in sql
    assert "t._ROW_HASH IS DISTINCT FROM s._SHASH OR t._IS_DELETED" in sql
    assert "QUALIFY ROW_NUMBER() OVER (PARTITION BY _SKEY" in sql
    assert """TRY_TO_NUMBER(TO_VARCHAR(_J['id']), 38, 0) AS "ID\"""" in sql
    assert """TO_VARCHAR(_J['legacy']) AS "LEGACY\"""" in sql          # a column V3 no longer sends: NULL
    assert '"ID" = s."ID", "FIRST_NAME" = s."FIRST_NAME"' in sql and "_RAW = s._J" in sql
    assert 'INSERT ("ID", "FIRST_NAME", "UPDATED_AT", "LEGACY", _TENANT_ID, _KEY, _RAW' in sql
    assert '"_TENANT_ID"' not in sql                                   # pipeline columns aren't treated as V3's
    assert params == {"t": 6, "src": "gyno"}


def test_sink_delete_guard_skips_the_update(monkeypatch):
    conn = RecordingConn(answers=[[(10, 5)], [(0, 0)]])          # 10 live, 5 not read back; then restore
    sink = S.SnowflakeSink(conn)
    monkeypatch.setattr(sink, "_stage", lambda name, cols, rows: f"X._STG_{name}")
    flagged, restored, note = sink.flag_deleted(T, 6, ["1", "2"], max_fraction=0.2)
    assert (flagged, restored) == (0, 0) and "5 of 10" in note
    assert not any("SET _IS_DELETED = TRUE" in s for s, _ in conn.sql)


def test_sink_flags_when_within_the_guard(monkeypatch):
    conn = RecordingConn(answers=[[(10, 1)], [(1, 0)], [(2, 0)]])
    sink = S.SnowflakeSink(conn)
    monkeypatch.setattr(sink, "_stage", lambda name, cols, rows: f"X._STG_{name}")
    assert sink.flag_deleted(T, 6, ["1"], max_fraction=0.2) == (1, 2, None)
    assert any("SET _IS_DELETED = TRUE" in s for s, _ in conn.sql)


def test_missing_state_table_means_no_watermark():
    class Boom(RecordingConn):
        def cursor(self):
            class Cur:
                def execute(self, sql, params=None):
                    raise RuntimeError("Object 'HOSPITALS.V3_SYNC._SYNC_STATE' does not exist or not authorized.")

                def close(self):
                    pass
            return Cur()
    assert S.SnowflakeSink(Boom()).get_watermark(6, "reception:patient") is None


# ── sources: discovery, isolation, one sync per organisation ──────────────

PROD_T = "https://{service}.afyaanalytics.ai/api/"


def test_discover_sources_takes_only_production_logins_and_honours_exclude():
    env = {
        "AFYA_ORG284_USERNAME": "a", "AFYA_ORG284_PASSWORD": "x", "AFYA_ORG284_URL_TEMPLATE": PROD_T,
        "AFYA_WEMA_PROD_USERNAME": "b", "AFYA_WEMA_PROD_PASSWORD": "x", "AFYA_WEMA_PROD_URL_TEMPLATE": PROD_T,
        "AFYA_GYNO_USERNAME": "c", "AFYA_GYNO_PASSWORD": "x",
        "AFYA_GYNO_URL_TEMPLATE": "https://{service}migrate.afyaanalytics.ai/api/",          # migrate: never
        "AFYA_OROKO_USERNAME": "d", "AFYA_OROKO_PASSWORD": "x",
        "AFYA_OROKO_URL_TEMPLATE": "https://{service}staging.afyaanalytics.ai/api/",          # staging: never
        "AFYA_ORG70_USERNAME": "e", "AFYA_ORG70_PASSWORD": "x",                              # no template → migrate default
        "AFYA_NOPW_USERNAME": "f", "AFYA_NOPW_URL_TEMPLATE": PROD_T,                          # no password
        "AFYA_SILVERWOOD_USERNAME": "g", "AFYA_SILVERWOOD_PASSWORD": "x", "AFYA_SILVERWOOD_URL_TEMPLATE": PROD_T,
        "V3_SYNC_EXCLUDE": "silverwood",
    }
    assert S.discover_sources(env) == ["org284", "wema_prod"]


@pytest.mark.parametrize("chosen, configured, expected", [
    (["gyno159"], "org284", ["gyno159"]),                 # the param wins
    ([], "org284, gyno159", ["org284", "gyno159"]),       # then V3_SYNC_SOURCES
    ([], "", ["org284", "wema_prod"]),                    # empty → discover
    ([], "auto", ["org284", "wema_prod"]),
    ([], "extra,auto", ["extra", "org284", "wema_prod"]),
    ([], "org284,ORG284,silverwood", ["org284"]),         # deduped, excluded
])
def test_configured_sources(chosen, configured, expected):
    env = {"AFYA_ORG284_USERNAME": "a", "AFYA_ORG284_PASSWORD": "x", "AFYA_ORG284_URL_TEMPLATE": PROD_T,
           "AFYA_WEMA_PROD_USERNAME": "b", "AFYA_WEMA_PROD_PASSWORD": "x", "AFYA_WEMA_PROD_URL_TEMPLATE": PROD_T,
           "V3_SYNC_SOURCES": configured, "V3_SYNC_EXCLUDE": "silverwood"}
    assert S.configured_sources(chosen, env) == expected


def _client_factory(orgs: dict[str, object]):
    class C:
        def __init__(self, name):
            self.name, self.facility_id = name, 1
            self.organization_id = None

        def token(self):
            outcome = orgs[self.name]
            if isinstance(outcome, Exception):
                raise outcome
            self.organization_id = outcome
            return "t"
    return C


def test_resolve_sources_isolates_bad_logins_and_collapses_one_organisation():
    factory = _client_factory({
        "org284": 284,
        "org250": RuntimeError("org250: account belongs to organization 284, expected 250"),   # merged into 284
        "silverwood": RuntimeError('silverwood: login failed 422 {"message":"Invalid credentials."}'),
        "kisumu": 284,                                                                      # second login, same org
        "wema_prod": 258,
        "orphan": RuntimeError("orphan: account belongs to organization 999, expected 1"),  # moved somewhere unsynced
    })
    ok, problems, notes = S.resolve_sources(["org284", "org250", "silverwood", "kisumu", "wema_prod", "orphan"], factory)
    assert [(s.name, s.tenant_id) for s in ok] == [("org284", 284), ("wema_prod", 258)]
    assert len(problems) == 2 and problems[0].startswith("silverwood:") and "organization 999" in problems[1]
    assert any(n.startswith("kisumu: organization 284 is already synced by org284") for n in notes)
    assert any(n.startswith("org250: login now lands in organization 284") for n in notes)


def test_unreadable_models_are_not_failures():
    e = RuntimeError('org284 core/category_filters read p1: 422 {"message":"Unknown column organization_id"}')
    assert S.unreadable(e, 0)
    assert not S.unreadable(e, 10)                                           # rows came first: a real failure
    assert not S.unreadable(RuntimeError("org284 core/x read p1: 429 slow down"), 0)
    assert not S.unreadable(RuntimeError("org284 core/x read p1: 500 boom"), 0)
    assert not S.unreadable(RuntimeError("org284 core/x read p7: 422 boom"), 0)

    class Refuses(FakeV3Client):
        def read_all(self, service, alias, per_page=500, workers=1):
            raise RuntimeError('gyno core/category_filters read p1: 422 {"message":"Unknown column"}')
            yield  # pragma: no cover
    sink = FakeSink()
    res = run(Refuses(), sink, S.Model("core", "category_filters"))
    assert res.status == "unreadable" and res.error is None and "refuses" in res.note
    assert sink.state[(6, "core:category_filters")]["status"] == "unreadable"
    assert "1 unreadable on V3" in S.summarize([res]) and "0 failed" in S.summarize([res])


# ── duplicates ────────────────────────────────────────────────────────────

def test_check_duplicates_counts_and_repairs_only_tables_that_have_them(monkeypatch):
    conn = RecordingConn(answers=[[("RECEPTION__PATIENT",), ("FINANCE__INVOICE",)],   # sync_tables
                                  [(0,)],                                            # patient: clean
                                  [(3,)], []])                                       # invoice: 3 extra → overwrite
    sink = S.SnowflakeSink(conn, "hospitals", "v3_sync")
    assert S.check_duplicates(sink) == {"FINANCE__INVOICE": 3}
    sqls = [s for s, _ in conn.sql]
    assert "LEFT(TABLE_NAME, 1) <> '_'" in sqls[0]                    # never _SYNC_STATE / staging
    assert sqls[1] == "SELECT COUNT(*) - COUNT(DISTINCT _TENANT_ID, _KEY) FROM HOSPITALS.V3_SYNC.RECEPTION__PATIENT"
    assert sqls[3].startswith("INSERT OVERWRITE INTO HOSPITALS.V3_SYNC.FINANCE__INVOICE SELECT * FROM "
                              "HOSPITALS.V3_SYNC.FINANCE__INVOICE QUALIFY ROW_NUMBER() OVER (PARTITION BY _TENANT_ID, _KEY")
    assert len(sqls) == 4                                             # clean table: no rewrite


def test_check_duplicates_report_only(monkeypatch):
    conn = RecordingConn(answers=[[(2,)]])
    assert S.check_duplicates(S.SnowflakeSink(conn), ["T"], repair=False) == {"T": 2}
    assert not any("OVERWRITE" in s for s, _ in conn.sql)


def test_one_key_per_tenant_survives_overlapping_runs_and_page_repeats():
    """The end-to-end no-duplicates contract on the fakes: a row read twice in one run (offset paging over a
    table being written), re-sent by the lookback overlap, and re-run in full keeps exactly one row."""
    client = tenant_with(3)
    client.rows("reception", "patient").append(dict(client.rows("reception", "patient")[0]))   # page repeat
    sink = FakeSink()
    run(client, sink)
    run(client, sink, now=NOW + timedelta(minutes=5))
    run(client, sink, now=NOW + timedelta(hours=3), full_refresh=True)
    assert sorted(sink.live_keys(T)) == ["1", "2", "3"]
    assert len(sink.tables[T]) == 3


def test_a_500_that_is_a_missing_table_on_v3_is_unreadable_not_failed():
    missing = ('aroha_prod inpatient/clinic read p1: 500 {"success":false,"message":"the database refused the row: '
               "Table 'erp_inpatient_db.clinics' doesn't exist\",\"error\":{\"type\":\"QueryException\",\"sqlstate\":\"42S02\"")
    assert S.unreadable(RuntimeError(missing), 0)
    assert not S.unreadable(RuntimeError("x inpatient/clinic read p1: 500 Server Error"), 0)     # plain 500: transient


def test_client_does_not_retry_a_schema_error_500(monkeypatch):
    import v3_to_v3_client as C
    calls = []

    class Resp:
        def __init__(self, code, text):
            self.status_code, self.text, self.ok = code, text, code < 400

    class Sess:
        def post(self, url, **kw):
            calls.append(url)
            return Resp(500, '{"error":{"type":"QueryException","sqlstate":"42S02"}}' if "inpatient" in url else "boom")

        def mount(self, *a):
            pass

    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    c = C.V3Client("t", urls={s: f"https://{s}.x/api/" for s in C.SERVICES}, username="u", password="p", session=Sess())
    monkeypatch.setattr(c, "_headers", lambda service: {})
    assert c.post("inpatient", {}, retry_500=True, retries=8).status_code == 500 and len(calls) == 1
    calls.clear()
    c.post("finance", {}, retry_500=True, retries=3)
    assert len(calls) == 3                                  # an ordinary 500 is still retried
    assert C.schema_error("Base table or view not found: 1146") and not C.schema_error("timeout")


def test_explicit_sources_must_still_be_production_logins():
    """Even a source named by hand is refused when it would read migrate / staging (e.g. a key with no
    URL template falls back to the client's migrate default)."""
    logged_in = []

    class C:
        def __init__(self, name):
            self.name, self.facility_id, self.organization_id = name, 1, None
            host = {"wema_prod": "", "gyno": "migrate", "oroko": "staging"}[name]
            self.urls = {"core": f"https://core{host}.afyaanalytics.ai/api/"}

        def token(self):
            logged_in.append(self.name)
            self.organization_id = {"wema_prod": 258}.get(self.name, 1)
            return "t"
    ok, problems, _ = S.resolve_sources(["wema_prod", "gyno", "oroko"], C)
    assert [s.name for s in ok] == ["wema_prod"] and logged_in == ["wema_prod"]      # never even logged in
    assert len(problems) == 2 and all("not a production login" in p for p in problems)
    assert "coremigrate.afyaanalytics.ai" in problems[0]
