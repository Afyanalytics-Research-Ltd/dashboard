"""
snowflake_to_v3_migration hardening (2026-10-29) — one test (or more) per fixed failure mode, each
written as the scenario that used to go wrong:

  * an optional link whose parent isn't in V3 went out as the RAW V2 id (pointing at an unrelated V3 row)
  * a parent's ids were looked up only in the first non-empty spelling of its alias (visit / visits)
  * records looked "already inserted" because a DIFFERENT model's map shared the other spelling
  * a failed uuid-column check was cached as "no uuid" → re-runs inserted duplicates
  * vitals whose visit lost its visit-map entry were dropped for good (and counted as posted)
  * the visit maps were never flushed by this pipeline (only every 500th entry)
  * every non-ok table was reported "failed / dead-lettered" (no retry) — even a lock or a Snowflake hiccup
  * a model missing from the gateway showed green
  * a dry run wrote the real state files (and could create Snowflake views)
  * the CLI exited 0 with failed tables
  * id→uuid file: whole-table overwrite across processes, unsynchronised alias snapshot

No network, no Snowflake (the chunked reader's SQL runs for real in
test_snowflake_to_v3_chunks_snowflake.py).
"""
from __future__ import annotations

import json
import sys
import threading

import pytest

import snowflake_to_v3_migration as m
import v2_to_v3_api_migration as v2v3

NS = r"App\Models\Child"
TK = "t_child"


@pytest.fixture
def fk(monkeypatch):
    """A child transform with one critical link (patient), one null-if-unresolved link (approver) and
    two plain optional links (ward, visit); fresh id map and registry."""
    monkeypatch.setitem(v2v3._FK_REMAP, TK, {"patient_id": "patient", "approved_by": "users",
                                             "ward_id": "ward", "visit_id": "visit"})
    monkeypatch.setitem(m._CRITICAL_FK_FIELDS, TK, ["patient_id"])
    monkeypatch.setitem(m._NULL_IF_UNRESOLVED, TK, {"approved_by"})
    monkeypatch.setattr(v2v3, "_id_map", {})
    monkeypatch.setattr(m, "_id_to_uuid", {})
    monkeypatch.setattr(m, "_alias_to_table", {})
    monkeypatch.setattr(m, "_unresolved_fk", m.Counter())
    return v2v3._id_map


# ── foreign keys ──────────────────────────────────────────────────────────

class TestForeignKeys:
    def test_unresolved_optional_link_is_sent_empty_never_as_the_v2_id(self, fk):
        fk["patient"] = {10: 9010}
        out, unresolved = m._remap_fks_via_uuid({"patient_id": 10, "ward_id": 33}, TK, NS)
        assert out["ward_id"] is None                    # was 33: V3 ward 33 is somebody else's ward
        assert out["patient_id"] == 9010 and unresolved == []
        assert m._unresolved_fk[(TK, "ward_id")] == 1    # counted for the table summary

    def test_unresolved_critical_link_holds_the_record(self, fk):
        out, unresolved = m._remap_fks_via_uuid({"patient_id": 10}, TK, NS)
        assert unresolved == ["patient_id"] and out["patient_id"] is None

    def test_null_if_unresolved_link_is_empty_and_not_counted(self, fk):
        out, _ = m._remap_fks_via_uuid({"approved_by": 5}, TK, NS)
        assert out["approved_by"] is None and not m._unresolved_fk

    def test_resolved_links_and_missing_values_untouched(self, fk):
        fk["ward"] = {33: 7033}
        out, unresolved = m._remap_fks_via_uuid({"ward_id": "33", "visit_id": None}, TK, NS)
        assert out == {"ward_id": 7033, "visit_id": None} and unresolved == []

    def test_parent_resolved_through_its_uuid(self, fk):
        m._id_to_uuid["wards"] = {33: "w-uuid"}
        m._alias_to_table["ward"] = "wards"
        fk["ward"] = {"w-uuid": 7033}
        assert m._remap_fks_via_uuid({"ward_id": 33}, TK, NS)[0]["ward_id"] == 7033

    def test_lookup_tries_every_spelling_registered_one_first(self, fk):
        # reception's own `visit` model got synced (by name) into `visit`; the visits job wrote `visits`.
        fk["visit"] = {325013: 111}                      # the WRONG model's row
        fk["visits"] = {325013: 222, 400: 444}
        m._alias_to_table["visits"] = "visits"           # this pipeline migrated the parent as `visits`
        out, _ = m._remap_fks_via_uuid({"visit_id": 325013}, TK, NS)
        assert out["visit_id"] == 222
        # a key only the non-empty `visit` map lacks used to be missed altogether
        assert m._remap_fks_via_uuid({"visit_id": 400}, TK, NS)[0]["visit_id"] == 444

    def test_lookup_falls_back_to_the_other_spelling(self, fk):
        fk["visits"] = {9: 99}
        assert m._lookup_v3_id("visit", 9) == 99 and m._lookup_v3_id("visit", 10) is None

    def test_a_non_empty_map_without_the_key_doesnt_end_the_search(self, fk):
        # the old lookup stopped at the first NON-EMPTY spelling, so a key only the other one had was lost
        fk["visit"] = {1: 11}
        fk["visits"] = {9: 99}
        assert m._lookup_v3_id("visit", 9) == 99 and m._lookup_v3_id("visit", 1) == 11


class TestAlreadyInserted:
    def test_only_this_models_own_map_counts(self, fk, monkeypatch):
        monkeypatch.setattr(v2v3, "_record_inserted", lambda job, key: False)
        monkeypatch.setattr(m, "_single_source", lambda alias: True)
        fk["visit"] = {325013: 111}                      # reception visits — a different V3 model
        assert not m._already_inserted("f|sf:visits", {"id": 325013, "uuid": "u"}, "visits")
        fk["visits"] = {325013: 222}
        assert m._already_inserted("f|sf:visits", {"id": 325013, "uuid": "u"}, "visits")


# ── uuid column check ─────────────────────────────────────────────────────

class Resp:
    def __init__(self, status, rows=()):
        self.status_code, self.ok, self._rows = status, 200 <= status < 300, list(rows)

    def json(self):
        return {"data": self._rows}


class TestModelHasUuid:
    @pytest.fixture(autouse=True)
    def _cache(self, monkeypatch):
        monkeypatch.setattr(m, "_alias_has_uuid", {})
        monkeypatch.setattr(v2v3, "v3_login_org_cfg", lambda: {"organization_id": 6})

    def test_gateway_error_assumes_uuid_and_does_not_cache(self, monkeypatch):
        monkeypatch.setattr(v2v3, "_gateway_post", lambda *a, **k: Resp(504))
        assert m._model_has_uuid("visits", "evaluation") is True
        assert "visits" not in m._alias_has_uuid          # was cached False → duplicates on re-run

    def test_exception_assumes_uuid_and_does_not_cache(self, monkeypatch):
        def boom(*a, **k):
            raise ConnectionError("reset")
        monkeypatch.setattr(v2v3, "_gateway_post", boom)
        assert m._model_has_uuid("visits", "evaluation") is True and not m._alias_has_uuid

    def test_real_answers_are_cached(self, monkeypatch):
        monkeypatch.setattr(v2v3, "_gateway_post", lambda *a, **k: Resp(200, [{"id": 1, "name": "x"}]))
        assert m._model_has_uuid("wards", "inpatient") is False and m._alias_has_uuid["wards"] is False
        monkeypatch.setattr(v2v3, "_gateway_post", lambda *a, **k: Resp(200, [{"id": 1, "uuid": "u"}]))
        assert m._model_has_uuid("beds", "inpatient") is True and m._alias_has_uuid["beds"] is True


# ── vitals split ──────────────────────────────────────────────────────────

class TestVitalsSplit:
    @pytest.fixture
    def split(self, tmp_path, monkeypatch):
        monkeypatch.setattr(v2v3, "VISIT_PATIENT_FILE", tmp_path / "vp.json")
        monkeypatch.setattr(v2v3, "VISIT_ADMISSION_FILE", tmp_path / "va.json")
        monkeypatch.setattr(v2v3, "_visit_patient_map", {1: 100})           # visit 1 → patient 100
        monkeypatch.setattr(v2v3, "_visit_admission_map", {2: 200})         # visit 2 → admission 200
        monkeypatch.setattr(v2v3, "transform_record", lambda r, tk, cfg, *a: dict(r))
        monkeypatch.setattr(v2v3, "_ensure_id_maps", lambda *a: None)
        posted = {}

        def fake_post(ns, cfg, records, *, transform_key, alias, job_key, dry_run):
            posted[transform_key] = [r["id"] for r in records]
            return 0
        monkeypatch.setattr(m, "post_table_to_v3", fake_post)
        # the facility's CLEAN data: visit 3 (patient 300, lost from the map), visit 4 (admitted, lost),
        # visit 5 (its patient unknown) — visit 9 was never extracted
        monkeypatch.setattr(m, "_source_visit_maps", lambda f: ({1: 100, 3: 300, 4: 400, 5: None}, {2: 200, 4: 40}))
        return posted

    def test_routes_heals_holds_and_skips(self, split):
        rows = [{"id": i, "visit_id": v} for i, v in ((11, 1), (12, 2), (13, 3), (14, 4), (15, 5), (19, 9))]
        failed, held, skipped = m._run_vitals_split({}, "f", rows, {}, "f|sf:vitals", "lbl", False)
        assert split["inpatient_vital"] == [12, 14]                      # 14 healed from admissions
        assert split[v2v3.OUTPATIENT_VITAL_TRANSFORM] == [11, 13]        # 13 healed from visits
        assert (failed, held, skipped) == (0, 1, 1)                      # 15 waits, 19 skipped
        assert v2v3._visit_patient_map[3] == 300 and v2v3._visit_admission_map[4] == 40

    def test_v2_column_name_and_string_ids(self, split):
        failed, held, skipped = m._run_vitals_split({}, "f", [{"id": 1, "visit": "1"}], {}, "j", "l", False)
        assert split[v2v3.OUTPATIENT_VITAL_TRANSFORM] == [1] and (failed, held, skipped) == (0, 0, 0)


def test_flush_visit_maps_merges_with_what_other_tasks_wrote(tmp_path, monkeypatch):
    vp = tmp_path / "vp.json"
    vp.write_text(json.dumps({"7": 70}))                                 # written by another task
    monkeypatch.setattr(v2v3, "VISIT_PATIENT_FILE", vp)
    monkeypatch.setattr(v2v3, "VISIT_ADMISSION_FILE", tmp_path / "va.json")
    monkeypatch.setattr(v2v3, "_visit_patient_map", {1: 10, 2: 20})      # 2 entries: below v2v3's 500 batch
    monkeypatch.setattr(v2v3, "_visit_admission_map", {})
    m.flush_visit_maps()
    assert json.loads(vp.read_text()) == {"1": 10, "2": 20, "7": 70}
    assert not (tmp_path / "va.json").exists()


# ── what a table run reports ──────────────────────────────────────────────

class TestRunOneTable:
    @pytest.fixture
    def run(self, monkeypatch):
        calls = {"flush": 0}
        monkeypatch.setattr(m, "_start", lambda facility, dry_run=False: ({}, set()))
        monkeypatch.setattr(m, "_select_entries", lambda *a, **k: ([{"table": "visits"}], [], []))
        monkeypatch.setattr(v2v3, "_record_progress_unflushed", 0)
        monkeypatch.setattr(v2v3, "_flush_id_map", lambda: None)
        monkeypatch.setattr(m, "flush_visit_maps", lambda: calls.__setitem__("flush", calls["flush"] + 1))
        monkeypatch.setattr(m, "_waiting", {})
        monkeypatch.setattr(m, "_errored", {})
        monkeypatch.setattr(m, "_not_registered", {})
        monkeypatch.setattr(m, "_unresolved_fk", m.Counter())

        def go(outcome, *, dry_run=False, before=None):
            def job(entry, facility, cfg, dry):
                if before:
                    before()
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome
            monkeypatch.setattr(m, "run_table_job", job)
            return m.run_one_table("f", "visits", dry_run=dry_run)
        go.calls = calls
        return go

    def test_ok(self, run):
        assert run(True) == {"table": "visits", "status": "ok", "detail": ""}
        assert run.calls["flush"] == 1                                    # visit maps written every table

    def test_ok_reports_links_sent_empty(self, run):
        r = run(True, before=lambda: m._unresolved_fk.update({("t", "ward_id"): 3}))
        assert r["status"] == "ok" and "t.ward_id': 3" in r["detail"]

    def test_transient_error_is_retryable_not_dead_letters(self, run):
        r = run(False, before=lambda: m._errored.__setitem__("visits", "Snowflake fetch failed: timeout"))
        assert r == {"table": "visits", "status": "error", "detail": "Snowflake fetch failed: timeout"}

    def test_dead_letters_are_failed(self, run):
        assert run(False)["status"] == "failed"

    def test_waiting(self, run):
        r = run(False, before=lambda: m._waiting.__setitem__("visits", "3 record(s) waiting"))
        assert r["status"] == "waiting"

    def test_model_not_in_gateway_is_skipped_not_green(self, run):
        r = run(True, before=lambda: m._not_registered.__setitem__("visits", "model not registered"))
        assert r["status"] == "skipped"

    def test_crash_is_error(self, run):
        assert run(RuntimeError("boom"))["status"] == "error"

    def test_dry_run_flushes_nothing(self, run, monkeypatch):
        monkeypatch.setattr(v2v3, "_flush_id_map", lambda: pytest.fail("dry run wrote the id map"))
        run(True, dry_run=True)
        assert run.calls["flush"] == 0


def test_table_busy_is_retryable(monkeypatch):
    monkeypatch.setattr(m, "_errored", {})

    def busy(table):
        raise m.TableBusy(f"{table} is being migrated by another task")
    monkeypatch.setattr(m, "table_lock", busy)
    assert m.run_table_job({"table": "visits"}, "f", {}, dry_run=False) is False
    assert "another task" in m._errored["visits"]


# ── dry run never touches real state ─────────────────────────────────────

def test_dry_run_reads_real_state_but_writes_a_throwaway_copy(tmp_path, monkeypatch):
    real = tmp_path / "state" / "fac"
    real.mkdir(parents=True)
    (real / ".migration_visit_patient.json").write_text(json.dumps({"1": 10}))
    (real / ".run_visits.lock").write_text("held")
    for name in ("STATE_ROOT", "PROGRESS_FILE", "RECORD_PROGRESS_FILE", "ID_MAP_FILE", "VISIT_PATIENT_FILE",
                 "VISIT_ADMISSION_FILE", "DONE_FILE"):
        monkeypatch.setattr(v2v3, name, getattr(v2v3, name))
    monkeypatch.setattr(m, "ID_TO_UUID_FILE", m.ID_TO_UUID_FILE)
    monkeypatch.setattr(v2v3, "STATE_ROOT", tmp_path / "state")

    m._dry_run_state("fac")
    copy = m.use_state_dir("fac")
    assert copy != real and json.loads((copy / ".migration_visit_patient.json").read_text()) == {"1": 10}
    assert not (copy / ".run_visits.lock").exists()                     # a lock is never copied
    v2v3._write_int_map(v2v3.VISIT_PATIENT_FILE, {2: 20})
    m._add_uuids("visits", {5: "u5"})
    assert json.loads((real / ".migration_visit_patient.json").read_text()) == {"1": 10}
    assert sorted(p.name for p in real.iterdir()) == [".migration_visit_patient.json", ".run_visits.lock"]


def test_dry_run_creates_no_clean_view(monkeypatch):
    monkeypatch.setattr(v2v3, "_permanently_done", set())
    monkeypatch.setattr(m, "ensure_clean_view", lambda *a: pytest.fail("dry run created a CLEAN view"))
    monkeypatch.setattr(m, "_v2_duplicate_codes", lambda *a: {})
    monkeypatch.setattr(m, "ensure_uuids_loaded", lambda *a: None)
    monkeypatch.setattr(m, "iter_clean_chunks", lambda *a: iter(()))
    entry = {"table": "visits", "v3": r"App\Models\Visit", "transform": "reception_visit"}
    assert m._run_table_job(entry, "f", {}, dry_run=True) is True


# ── CLI ───────────────────────────────────────────────────────────────────

def test_cli_exits_nonzero_when_tables_failed(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--facility", "f"])
    monkeypatch.setattr(m, "run_migration", lambda *a, **k: ["visits"])
    with pytest.raises(SystemExit) as e:
        m.main()
    assert e.value.code == 1


def test_cli_exits_zero_when_clean(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--facility", "f"])
    monkeypatch.setattr(m, "run_migration", lambda *a, **k: [])
    m.main()


# ── id → uuid registry file ───────────────────────────────────────────────

@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ID_TO_UUID_FILE", tmp_path / ".migration_id_to_uuid.json")
    monkeypatch.setattr(m, "_id_to_uuid", {})
    monkeypatch.setattr(m, "_alias_to_table", {})
    monkeypatch.setattr(m, "_uuid_to_id", {})
    return tmp_path / ".migration_id_to_uuid.json"


def test_registry_merges_per_table_across_processes(registry):
    m._add_uuids("visits", {1: "u1"})
    on_disk = json.loads(registry.read_text())
    on_disk["id_to_uuid"]["visits"]["2"] = "u2"          # another process registered visit 2 meanwhile
    registry.write_text(json.dumps(on_disk))
    m._add_uuids("visits", {3: "u3"})                   # this process never saw visit 2
    assert json.loads(registry.read_text())["id_to_uuid"]["visits"] == {"1": "u1", "2": "u2", "3": "u3"}


def test_registry_is_safe_under_parallel_pages(registry):
    errors = []

    def worker(n):
        try:
            for i in range(25):
                m._register_table(f"alias{n}_{i}", f"table{n}", [{"id": n * 1000 + i, "uuid": f"u{n}-{i}"}])
        except Exception as e:          # "dictionary changed size during iteration"
            errors.append(e)
    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    data = json.loads(registry.read_text())
    assert sum(len(v) for v in data["id_to_uuid"].values()) == 200 and len(data["alias_to_table"]) == 200
