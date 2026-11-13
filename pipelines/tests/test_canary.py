"""
Canary runs (canary.py and its use in the three load pipelines):

  * canary.py — the canary param, value normalisation, record comparison,
    target / report verdicts, the saved report and the reconciliation payload
  * from_json_mappings_to_snowflake — page-by-page sampling, the 401 fix, a
    canary table landing intact / not intact / with an empty source, and that
    a canary never replaces or clears a table
  * dags/document_pipeline.py — only the first records' pages are read, a
    possibly truncated record is left out, the file is never finished or
    moved, the load replaces only the canary's rows, and the read-back
    catches a value its column could not take
  * snowflake_to_v3_migration — records with parents in V3 are picked first,
    what was posted is kept, and the read-back from the gateway passes, finds
    a changed field, and finds a missing record

No network, no Snowflake, no V3.
"""
from __future__ import annotations

import json
import re

import pytest

import canary
import from_json_mappings_to_snowflake as fj
import snowflake_to_v3_migration as m
import v2_to_v3_api_migration as v2v3
from tests.helpers import load_dag_module

dp = load_dag_module("document_pipeline")


# ─── canary.py ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("value, n", [(None, 0), ("", 0), (0, 0), (3, 3), ("12", 12)])
def test_canary_size_accepts_counts(value, n):
    assert canary.canary_size(value) == n


@pytest.mark.parametrize("value", [-1, "three", True, "2.5"])
def test_canary_size_rejects_anything_else(value):
    with pytest.raises(ValueError, match="canary"):
        canary.canary_size(value)


@pytest.mark.parametrize("a, b", [
    (12, "12.00"), (True, 1), ("2024-01-01 10:00:00", "2024-01-01T10:00:00.000000Z"),
    ("2024-01-01", "2024-01-01T00:00:00.000000Z"), ({"b": 1, "a": "x"}, {"a": "x", "b": "1"}),
    (" text ", "text"), ("", None),
    ("3FEB8752-0000-0000-0000-00000000000C", "3feb8752-0000-0000-0000-00000000000c"),
])
def test_normalise_treats_destination_spellings_as_equal(a, b):
    assert canary.normalise(a) == canary.normalise(b)


def test_normalise_keeps_real_differences():
    assert canary.normalise("12.5") != canary.normalise(12)
    assert canary.normalise("2024-01-01 10:00") != canary.normalise("2024-01-02 10:00")


def test_compare_record_reports_changed_and_dropped_fields_only_for_sent_values():
    sent = {"id": 1, "name": "Ann", "age": 30, "note": None, "facility_id": 9, "ward": "A"}
    stored = {"id": 77, "name": "Ann", "age": "31", "facility_id": 4}
    mismatched, absent = canary.compare_record(sent, stored, skip={"id", "facility_id"})
    assert mismatched == ["age"]
    assert absent == ["ward"]                 # sent with a value, not stored; "note" was sent empty


def test_target_status_and_summary():
    ok = canary.TargetResult("patients", expected=3, landed=3, ids=["1", "2", "3"])
    assert ok.status == canary.PASSED
    bad = canary.TargetResult("visits", expected=3, landed=2, missing=["9"])
    bad.mismatch("visit_date", 4)
    assert bad.status == canary.FAILED
    assert "1 missing (ids 9)" in bad.summary() and "visit_date not stored as sent on 1 (ids 4)" in bad.summary()
    assert canary.TargetResult("empty", untested="the source table is empty").status == canary.UNTESTED
    assert canary.TargetResult.from_dict(bad.to_dict()).to_dict() == bad.to_dict()


def test_report_fails_on_a_failed_target_and_when_nothing_was_checked():
    rep = canary.CanaryReport("p", 3, targets=[canary.TargetResult("a", expected=3, landed=3)])
    assert rep.ok
    rep.add(canary.TargetResult("b", errors=["boom"]))
    assert not rep.ok and [t.target for t in rep.failed] == ["b"]
    with pytest.raises(canary.CanaryFailed, match="error: boom"):
        rep.raise_if_failed()
    empty = canary.CanaryReport("p", 3, targets=[canary.TargetResult("a", untested="x")])
    assert not empty.ok and "nothing could be checked" in empty.summary()


def test_report_saves_by_run_and_as_latest_and_publishes_counts(tmp_path):
    rep = canary.CanaryReport("pipe", 2, facility="silverwood", run_id="R1")
    rep.add(canary.TargetResult("patients", expected=2, landed=2, ids=["1", "2"], duplicates=[]))
    path = rep.save(tmp_path)
    assert path == tmp_path / "pipe" / "R1.json"
    latest = canary.load_latest("pipe", "silverwood", tmp_path)
    assert latest["ok"] and latest["targets"][0]["status"] == "passed" and latest["finished_at"]
    assert canary.load_latest("pipe", "other", tmp_path) is None
    rec = rep.reconciliation()["targets"][0]
    assert rec == {"target": "patients", "expected_target_rows": 2, "loaded_target_rows": 2,
                   "duplicate_count": 0, "mismatched_fields": [], "canary_status": "passed"}


class _TI:
    def __init__(self):
        self.pushed = {}

    def xcom_push(self, key, value):
        self.pushed[key] = value


def test_finish_in_airflow_pushes_reconciliation_and_fails_the_task(tmp_path, monkeypatch):
    monkeypatch.setattr(canary, "CANARY_DIR", tmp_path)
    ti = _TI()
    rep = canary.CanaryReport("pipe", 1, targets=[canary.TargetResult("t", expected=1, missing=["5"])])
    with pytest.raises(RuntimeError, match="1 missing"):   # the stub's AirflowFailException
        canary.finish_in_airflow(rep, {"ti": ti})
    assert ti.pushed["pipeline_reconciliation"]["targets"][0]["canary_status"] == "failed"
    assert list((tmp_path / "pipe").glob("*.json"))


# ─── JSON mappings → Snowflake ───────────────────────────────────────────

MAPPINGS = {
    "connection": {"id": 7, "name": "silverwood"},
    "tables": [{"source table": "person", "v3 model": "App\\Models\\Reception\\Patient", "v3 table": "patients"},
               {"source table": "services", "v3 model": "App\\Models\\Product", "v3 table": "products",
                "where": {"column": "kind", "in": ["Drug"]}}],
    "fields": [{"table": "person", "v3_table": "patients", "column": "fname", "field": "first_name"},
               {"table": "services", "v3_table": "products", "column": "label", "field": "name"}],
}


class FakeExtraction:
    """Pages of 2 rows per source table; records every page asked for."""
    per_page = 2

    def __init__(self, tables):
        self.tables, self.asked = tables, []

    def fetch_page(self, table, page):
        self.asked.append((table, page))
        rows = self.tables.get(table, [])
        return rows[(page - 1) * 2:page * 2], 2

    def iter_pages(self, table, max_pages=None):
        return fj.ExtractionClient.iter_pages(self, table, max_pages)

    def fetch_all(self, table):
        raise AssertionError("a canary must not read the whole table")


class FakeWarehouse(fj.Warehouse):
    """Keeps loads in memory; `corrupt` changes a field on the way in."""

    def __init__(self, corrupt=None):
        super().__init__("silverwood")
        self.loads, self.calls, self.corrupt = {}, [], corrupt or {}

    def upsert_records(self, sf, model, records, run_id):
        self.calls.append(("upsert", model.table))
        stored = [json.loads(fj._dumps_bytes({**r, **self.corrupt.get(model.table, {})})) for r in records]
        self.loads[model.table] = stored
        return fj.datetime.now(fj.timezone.utc)

    def read_back(self, sf, model, ingested_at):
        return self.loads.get(model.table, [])

    def duplicate_ids(self, sf, model, ids):
        return []

    def replace_table(self, *a, **k):
        raise AssertionError("a canary must not replace a table")

    def clear_table(self, *a, **k):
        raise AssertionError("a canary must not clear a table")

    def log_run(self, sf, run_id, connection_id, source_table, models, status, rows, started, error=None):
        self.calls.append(("log", source_table, status))


class _SF:
    def close(self):
        pass


@pytest.fixture
def spec(monkeypatch):
    monkeypatch.setattr(fj, "SnowflakeClient", _SF)
    return fj.Mappings.load(MAPPINGS)


PEOPLE = [{"id": i, "fname": f"P{i}", "password": "secret"} for i in range(1, 8)]


def test_json_canary_reads_only_the_pages_it_needs_and_lands_n_records(spec):
    client, wh = FakeExtraction({"person": PEOPLE}), FakeWarehouse()
    res = fj.process_table(spec, wh, client, "person", run_id="R", dry_run=False, canary=3)
    assert client.asked == [("person", 1), ("person", 2)]          # 4 rows read, 3 needed
    assert res.status == "canary" and res.v3["patients"]["rows"] == 3
    assert [r["id"] for r in wh.loads["patients"]] == [1, 2, 3]
    assert "password" not in wh.loads["patients"][0]
    (target,) = res.canary
    assert target["status"] == "passed" and target["ids"] == ["1", "2", "3"] and target["landed"] == 3
    assert ("log", "person", "canary") in wh.calls


def test_json_canary_keeps_reading_until_the_where_filter_has_n_rows(spec):
    rows = [{"id": i, "label": f"S{i}", "kind": "Drug" if i in (3, 6) else "Lab"} for i in range(1, 9)]
    client = FakeExtraction({"services": rows})
    res = fj.process_table(spec, FakeWarehouse(), client, "services", run_id="R", dry_run=False, canary=2)
    assert client.asked == [("services", 1), ("services", 2), ("services", 3)]
    assert res.canary[0]["ids"] == ["3", "6"]


def test_json_canary_flags_a_value_that_did_not_land_as_sent(spec):
    wh = FakeWarehouse(corrupt={"patients": {"first_name": None}})
    res = fj.process_table(spec, wh, FakeExtraction({"person": PEOPLE}), "person", run_id="R",
                           dry_run=False, canary=2)
    assert res.status == "failed" and "first_name not stored as sent on 2" in res.error
    assert res.canary[0]["mismatched"] == {"first_name": ["1", "2"]}
    assert ("log", "person", "canary_failed") in wh.calls


def test_json_canary_on_an_empty_source_is_untested_and_clears_nothing(spec):
    res = fj.process_table(spec, FakeWarehouse(), FakeExtraction({}), "person", run_id="R",
                           dry_run=False, canary=3)
    assert res.status == "canary" and res.canary[0]["status"] == "untested"
    assert res.canary[0]["untested"] == "the source table is empty"


def test_json_canary_report_collects_every_v3_table(spec):
    wh = FakeWarehouse()
    results = [fj.process_table(spec, wh, FakeExtraction({"person": PEOPLE}), "person", run_id="R",
                                dry_run=False, canary=1),
               fj.process_table(spec, wh, FakeExtraction({}), "services", run_id="R", dry_run=False, canary=1)]
    rep = fj.canary_report(results, name="Silverwood", run_id="R", size=1)
    assert rep.facility == "silverwood" and rep.ok
    assert sorted(t.target for t in rep.targets) == ["patients", "products"]


def test_fetch_page_raises_when_every_attempt_is_401(monkeypatch):
    monkeypatch.setenv("AFYA_EXTRACTION_USERNAME", "u")
    monkeypatch.setenv("AFYA_EXTRACTION_PASSWORD", "p")
    client = fj.ExtractionClient(7, max_retries=2)
    monkeypatch.setattr(client, "_bearer", lambda force=False: "tok")

    class R:
        status_code, ok = 401, False

    monkeypatch.setattr(client._session, "post", lambda *a, **k: R())
    with pytest.raises(RuntimeError, match="401"):
        client.fetch_page("person", 1)


# ─── document pipeline ───────────────────────────────────────────────────

SCHEMA = {"type": "object", "title": "Neo notes",
          "properties": {"patient_name": {"type": ["string", "null"]}, "weight": {"type": "number"}}}
PAGES = {
    "1": "# Outpatient Clinical Consultation Form\nPatient Demographics\n| Patient Name | John Otieno |",
    "2": "Progress notes, continued.",
    "3": "# Outpatient Clinical Consultation Form\nPatient Demographics\n| Patient Name | Mary Akinyi |",
    "4": "Lab results.",
}


class _Resp:
    ok, status_code = True, 200

    def __init__(self, body):
        self.body, self.text = body, json.dumps(body)

    def json(self):
        return self.body


def _landing_ai(calls):
    def post(url, headers=None, files=None, data=None, timeout=None):
        schema = json.loads(data["schema"])
        calls.append(files["markdown"][1])
        if "is_first_page_of_record" in schema["properties"]:
            return _Resp({"extraction": {"is_first_page_of_record": False}})
        return _Resp({"extraction": {"patient_name": "John Otieno", "weight": "about three kilos"}})
    return post


@pytest.fixture
def doc_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VA_API_KEY", "k")
    monkeypatch.setenv("SNOWFLAKE_DATABASE", "HOSPITALS")
    calls, moves = [], []
    monkeypatch.setattr(dp.requests, "post", _landing_ai(calls))
    monkeypatch.setattr(dp, "move_file", lambda *a: moves.append(a))
    monkeypatch.setattr(dp, "download_file", lambda *a, **k: pytest.fail("pages are cached"))
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "f1.json").write_text(json.dumps({
        "file_id": "f1", "filename": "batch.pdf", "total_pages": 4, "file_done": False,
        "pages": PAGES, "page_errors": {}, "page_meta": {}, "records": {}}), encoding="utf-8")

    def cfg(**params):
        return dp.config_from_params({"schema": SCHEMA, "state_dir": str(state_dir), "output_dir": str(tmp_path / "o"),
                                      "json_output_dir": str(tmp_path / "o" / "j"), **params})

    return {"cfg": cfg, "calls": calls, "moves": moves, "state": state_dir / "f1.json"}


META = {"id": "f1", "name": "batch.pdf", "mimeType": "application/pdf"}


def test_document_canary_defaults_to_one_file():
    assert dp.config_from_params({"schema": SCHEMA, "canary": 2}).max_files == 1
    assert dp.config_from_params({"schema": SCHEMA, "canary": 2, "max_files": 5}).max_files == 5
    assert dp.config_from_params({"schema": SCHEMA}).max_files == 0


def test_canary_records_leaves_out_a_record_cut_by_the_page_limit():
    recs = [{"page_numbers": [1, 2]}, {"page_numbers": [3]}]
    assert dp.canary_records(recs, pages_read=3, total_pages=4, size=5) == [{"page_numbers": [1, 2]}]
    assert dp.canary_records(recs, pages_read=3, total_pages=3, size=5) == recs   # the file ended there
    assert dp.canary_records(recs, pages_read=4, total_pages=9, size=1) == [{"page_numbers": [1, 2]}]


def test_document_canary_extracts_the_first_records_and_leaves_the_file_unfinished(doc_env):
    cfg = doc_env["cfg"](canary=1, max_pages_per_record=2)        # reads pages 1-3
    results = dp.process_one_file(None, META, "doneFolder", cfg)
    assert [(r["record_index"], r["page_range"], r["ok"]) for r in results] == [(1, "1-2", True)]
    assert not any("Lab results" in c for c in doc_env["calls"])   # page 4 never sent
    state = json.loads(doc_env["state"].read_text())
    assert state["canary_record_count"] == 1 and not state.get("file_done")
    assert "record_count" not in state and doc_env["moves"] == []
    assert dp.canary_record_keys(cfg, ["f1"]) == (["f1:1"], {})


def test_document_canary_counts_records_done_earlier(doc_env):
    cfg = doc_env["cfg"](canary=1, max_pages_per_record=2)
    dp.process_one_file(None, META, None, cfg)
    again = dp.process_one_file(None, META, None, cfg)
    assert again == [{"ok": True, "filename": "batch.pdf", "file_id": "f1", "record_index": 1,
                      "status": "already_done"}]


class CanarySnowflake:
    """Captures statements; answers the canary read-back with `stored`."""

    def __init__(self, stored):
        self.stored, self.sql = stored, []

    def fetchall(self, sql, params=None):
        self.sql.append(sql)
        if "SCHEMATA" in sql:
            return [(1,)]
        if "COLUMNS" in sql:
            cols = {**{c: ("TEXT", None) for c in dp.SF_META_COLUMNS}, "PATIENT_NAME": ("TEXT", None),
                    "WEIGHT": ("FLOAT", None)}
            return [(c, t, s) for c, (t, s) in cols.items()]
        if "RECORD_KEY IN" in sql:
            return self.stored
        raise AssertionError(sql)

    def execute(self, sql, params=None):
        self.sql.append(sql)
        return 1

    def executemany(self, sql, rows):
        self.sql.append(sql)
        return len(rows)

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


def test_document_canary_load_replaces_only_the_canary_rows(doc_env):
    cfg = doc_env["cfg"](canary=1, max_pages_per_record=2)
    dp.process_one_file(None, META, None, cfg)
    sf = CanarySnowflake([])
    out = dp.load_to_snowflake(cfg, ["f1"], sf=sf, record_keys=["f1:1"])
    assert out["rows"] == 1
    delete = next(s for s in sf.sql if s.startswith("DELETE"))
    assert "WHERE RECORD_KEY IN" in delete and ":record_key::VARCHAR" in delete


def test_document_canary_read_back_catches_a_value_its_column_could_not_take(doc_env):
    cfg = doc_env["cfg"](canary=1, max_pages_per_record=2)
    dp.process_one_file(None, META, None, cfg)
    # columns: PATIENT_NAME IS NULL, WEIGHT IS NULL ("about three kilos" didn't fit FLOAT)
    report = dp.verify_snowflake_canary(cfg, ["f1"], sf=CanarySnowflake([("f1:1", 1, False, True)]))
    (t,) = report.targets
    assert t.target == "HOSPITALS.NEO_NOTES.EXTRACTED_RECORDS" and t.landed == 1
    assert t.mismatched == {"weight": ["f1:1"]} and not report.ok


def test_document_canary_read_back_finds_missing_and_duplicate_rows(doc_env):
    cfg = doc_env["cfg"](canary=1, max_pages_per_record=2)
    dp.process_one_file(None, META, None, cfg)
    assert dp.verify_snowflake_canary(cfg, ["f1"], sf=CanarySnowflake([])).targets[0].missing == ["f1:1"]
    twice = [("f1:1", 2, False, False), ("f1:1", 2, False, False)]
    assert dp.verify_snowflake_canary(cfg, ["f1"], sf=CanarySnowflake(twice)).targets[0].duplicates == ["f1:1"]


def test_document_canary_passes_when_every_record_landed(doc_env):
    cfg = doc_env["cfg"](canary=1, max_pages_per_record=2)
    dp.process_one_file(None, META, None, cfg)
    report = dp.verify_snowflake_canary(cfg, ["f1"], sf=CanarySnowflake([("f1:1", 1, False, False)]))
    assert report.ok


# ─── Snowflake → V3 ──────────────────────────────────────────────────────

@pytest.fixture
def v3(monkeypatch):
    """A canary of 2 with the posting path's collaborators stubbed."""
    monkeypatch.setattr(m, "RECORD_LIMIT", 2)
    monkeypatch.setattr(m, "_canary_sent", {})
    monkeypatch.setattr(m, "_canary_posted", m.defaultdict(list))
    monkeypatch.setattr(m, "_canary_unresolvable", m.Counter())
    monkeypatch.setattr(m, "_canary_results", {})
    monkeypatch.setattr(m, "_CRITICAL_FK_FIELDS", {"visit": ["patient"]})
    monkeypatch.setattr(m, "_already_inserted", lambda *a, **k: False)
    monkeypatch.setattr(m, "_link_existing_by_field", lambda pending, **k: pending)
    monkeypatch.setattr(m, "_apply_v3_column_names", lambda payload, tk: payload)
    monkeypatch.setattr(m, "_with_stable_uuid", lambda payload, **k: payload)
    monkeypatch.setattr(m, "_store_uuid_mapping", lambda *a: None)
    monkeypatch.setattr(v2v3, "_mark_record_inserted", lambda *a: None)
    monkeypatch.setattr(v2v3, "_flush_record_progress", lambda: None)
    monkeypatch.setattr(v2v3, "_alias_to_service", {"visits": "evaluation"})
    monkeypatch.setattr(v2v3, "_gateway_model_meta", {"visits": {"facility": "facility_id"}})

    def remap(rec, tk, ns, report_all=False):
        missing = [] if rec.get("patient") in (None, 10, 11, 12) else ["patient"]
        return {**rec}, missing
    monkeypatch.setattr(m, "_remap_fks_via_uuid", remap)
    monkeypatch.setattr(v2v3, "_v3_alias", lambda ns: "visits")
    monkeypatch.setattr(m, "_ALIAS_OVERRIDE", {})
    monkeypatch.setattr(m, "_SERVICE_OVERRIDE", {})
    posted = {}   # uuid -> the V3 id "assigned": 500 + the number in the uuid
    monkeypatch.setattr(v2v3, "_post_to_v3_batch",
                        lambda ns, cfg, payload, **kw: posted.setdefault(payload["uuid"], 500 + int(payload["uuid"][1:])))
    return posted


RECS = [{"id": 1, "uuid": "u1", "patient": 99, "visit_date": "2024-01-01"},   # parent not in V3
        {"id": 2, "uuid": "u2", "patient": 10, "visit_date": "2024-01-02"},
        {"id": 3, "uuid": "u3", "patient": 11, "visit_date": "2024-01-03"},
        {"id": 4, "uuid": "u4", "patient": 12, "visit_date": "2024-01-04"}]


def _post(recs):
    return m.post_table_to_v3(r"App\Models\Visit", {"organization_id": 4}, [dict(r) for r in recs],
                              transform_key="visit", alias="visits", job_key="fac|sf:visits", dry_run=False)


def test_v3_canary_posts_n_records_whose_parents_are_in_v3(v3):
    assert _post(RECS) == 0
    assert sorted(v3) == ["u2", "u3"]                                 # u1's patient isn't in V3
    posted = m._canary_posted["fac|sf:visits"]
    assert [(p["key"], p["v3_id"], p["alias"], p["service"]) for p in posted] == [
        ("u2", 502, "visits", "evaluation"), ("u3", 503, "visits", "evaluation")]
    assert m._canary_unresolvable["fac|sf:visits"] == 1
    assert _post(RECS[3:]) == 0 and sorted(v3) == ["u2", "u3"]       # limit reached: nothing more


def _gateway(rows_by_page, total, calls):
    def post(service, body, timeout=30):
        calls.append(body["page"])
        if body["per_page"] == 1:
            return _Resp({"data": [], "meta": {"total": total}})
        return _Resp({"data": rows_by_page.get(body["page"], [])})
    return post


def test_v3_canary_read_back_passes_and_ignores_rewritten_fields(v3, monkeypatch):
    _post(RECS)
    stored = [{"id": 502, "uuid": "u2", "patient": 10, "visit_date": "2024-01-02T00:00:00Z", "facility_id": 4},
              {"id": 503, "uuid": "u3", "patient": 11, "visit_date": "2024-01-03 00:00:00"}]
    calls = []
    monkeypatch.setattr(v2v3, "_gateway_post", _gateway({3: [{"id": 1}], 2: stored}, 1100, calls))
    res = m._verify_canary("visits", "fac|sf:visits", {"organization_id": 4})
    assert res.status == canary.PASSED and res.landed == 2 and res.ids == ["u2", "u3"]
    assert calls == [1, 3, 2]                                         # newest pages first, stop when found


def test_v3_canary_read_back_flags_changed_and_missing_records(v3, monkeypatch):
    _post(RECS)
    stored = [{"id": 502, "uuid": "u2", "patient": 10, "visit_date": "2024-02-02"}]
    monkeypatch.setattr(v2v3, "_gateway_post", _gateway({1: stored}, 1, []))
    res = m._verify_canary("visits", "fac|sf:visits", {"organization_id": 4})
    assert res.missing == ["u3"] and res.mismatched == {"visit_date": ["u2"]}
    assert res.status == canary.FAILED


def test_v3_canary_finish_reports_untested_when_no_parent_is_in_v3(v3, monkeypatch):
    _post(RECS[:1])
    monkeypatch.setattr(v2v3, "_gateway_post", lambda *a, **k: pytest.fail("nothing to read back"))
    assert m._finish_canary("visits", "fac|sf:visits", {}, "visits", total=1, failed=0, waiting=0)
    res = m._canary_results["visits"]
    assert res["status"] == "untested" and "no record has its parents in V3 yet" in res["untested"]


def test_v3_canary_dead_letters_fail_the_table(v3, monkeypatch):
    monkeypatch.setattr(v2v3, "_gateway_post", _gateway({}, 0, []))
    assert not m._finish_canary("visits", "fac|sf:visits", {}, "visits", total=4, failed=2, waiting=0)
    assert "2 record(s) dead-lettered" in m._canary_results["visits"]["errors"][0]
    rep = m.canary_report("fac", "R1")
    assert rep.facility == "fac" and not rep.ok


def test_run_one_table_turns_a_failed_canary_or_a_crash_into_a_failed_table(v3, monkeypatch):
    monkeypatch.setattr(m, "_run_one_table", lambda f, t, dry_run: {"table": t, "status": "error", "detail": "boom"})
    out = m.run_one_table("fac", "visits", dry_run=False)
    assert out["status"] == "error" and out["canary"]["status"] == "failed"
    assert out["canary"]["errors"] == ["crashed: boom"]

    def ran(f, t, dry_run):
        m._canary_results[t] = canary.TargetResult(t, expected=2, landed=1, missing=["u3"]).to_dict()
        return {"table": t, "status": "ok", "detail": ""}
    monkeypatch.setattr(m, "_run_one_table", ran)
    out = m.run_one_table("fac", "visits", dry_run=False)
    assert out["status"] == "failed" and "1 missing (ids u3)" in out["detail"]

    monkeypatch.setattr(m, "_run_one_table", lambda f, t, dry_run: {"table": t, "status": "skipped",
                                                                    "detail": "not insertable in the gateway"})
    out = m.run_one_table("fac", "visits", dry_run=False)
    assert out["status"] == "skipped" and out["canary"]["untested"] == "not insertable in the gateway"


def test_remap_report_all_lists_every_unresolved_link_quietly(monkeypatch, caplog):
    monkeypatch.setattr(v2v3, "_FK_REMAP", {"tk": {"patient": "patient", "ward": "ward"}})
    monkeypatch.setattr(v2v3, "_NS_FK_REMAP", {})
    monkeypatch.setattr(m, "_CRITICAL_FK_FIELDS", {"tk": ["patient"]})
    monkeypatch.setattr(m, "_id_map_for", lambda alias: {})
    monkeypatch.setattr(m, "_table_for_alias", lambda alias: None)
    rec = {"patient": 5, "ward": 6}
    assert m._remap_fks_via_uuid(rec, "tk", "ns")[1] == ["patient"]
    caplog.clear()
    assert m._remap_fks_via_uuid(rec, "tk", "ns", report_all=True)[1] == ["patient", "ward"]
    assert not [r for r in caplog.records if "No V3 ID mapping" in r.getMessage()]


def test_canary_module_reaches_the_document_dag():
    assert dp._canary_module() is canary
    assert re.search(r"CANARY", dp.__doc__)
