"""
dags/docs_to_events_raw.py unit tests.

Tests cover the module-level helpers (check_null_names, run_setup, run_validate,
run_write, run_verify, _require_facility) via a fake cursor — no Snowflake connection required.

Integration tests (real Snowflake) live on the server.
"""
from __future__ import annotations

import pytest
from airflow.exceptions import AirflowFailException

from tests.helpers import load_dag_module

dag = load_dag_module("docs_to_events_raw")

FACILITY = "siaya"
SRC = "HOSPITALS.SIAYA_DOCS.EXTRACTED_FORMS"
DST = "HOSPITALS.SIAYA_RAW.EVENTS_RAW"
STORE_UUID = "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"


# ---------------------------------------------------------------------------
# Fake cursor
# ---------------------------------------------------------------------------

class FakeCursor:
    """Records every SQL string passed to execute(); supports configurable
    fetchone() results keyed by a substring of the SQL."""

    def __init__(self, fetchone_map: dict | None = None, rowcount: int = 5,
                 fail_on: str | None = None):
        self._fetchone_map = fetchone_map or {}
        self.rowcount = rowcount
        self.fail_on = fail_on
        self.executed: list[str] = []

    def execute(self, sql, params=None):
        self.executed.append(sql.strip())
        if self.fail_on and self.fail_on in sql:
            raise RuntimeError(f"Simulated failure on: {self.fail_on}")
        return self

    def fetchone(self):
        for key, value in self._fetchone_map.items():
            if any(key in s for s in self.executed[-1:]):
                return value
        return (0,)


# ---------------------------------------------------------------------------
# _require_facility
# ---------------------------------------------------------------------------

def test_require_facility_raises_on_empty_string():
    with pytest.raises(AirflowFailException, match="requires 'facility'"):
        dag._require_facility('')


def test_require_facility_raises_on_whitespace():
    with pytest.raises(AirflowFailException, match="requires 'facility'"):
        dag._require_facility('   ')


def test_require_facility_passes_for_valid_key():
    dag._require_facility('siaya')
    dag._require_facility('kisumu-v3')
    dag._require_facility('facility_name_01')


# ---------------------------------------------------------------------------
# check_null_names
# ---------------------------------------------------------------------------

def test_check_null_names_returns_zero_and_no_warning(caplog):
    cur = FakeCursor(fetchone_map={"patient_name::STRING IS NULL": (0,)})
    with caplog.at_level("WARNING"):
        result = dag.check_null_names(cur, SRC)
    assert result == 0
    assert "row(s)" not in caplog.text


def test_check_null_names_logs_warning_when_nulls_present(caplog):
    cur = FakeCursor(fetchone_map={"patient_name::STRING IS NULL": (3,)})
    with caplog.at_level("WARNING"):
        result = dag.check_null_names(cur, SRC)
    assert result == 3
    assert "3 row(s)" in caplog.text
    assert "held back" in caplog.text


# ---------------------------------------------------------------------------
# run_setup
# ---------------------------------------------------------------------------

def test_run_setup_creates_both_schemas_and_events_raw_table():
    cur = FakeCursor()
    dag.run_setup(cur, FACILITY, DST)
    assert any("SIAYA_DOCS" in s for s in cur.executed)
    assert any("SIAYA_RAW" in s for s in cur.executed)
    assert any("EVENTS_RAW" in s and "CREATE TABLE" in s for s in cur.executed)
    assert all("IF NOT EXISTS" in s for s in cur.executed)


def test_run_setup_uses_facility_key_for_schema_names():
    cur = FakeCursor()
    dag.run_setup(cur, 'kisumu_v3', 'HOSPITALS.KISUMU_V3_RAW.EVENTS_RAW')
    assert any("KISUMU_V3_DOCS" in s for s in cur.executed)
    assert any("KISUMU_V3_RAW" in s for s in cur.executed)


# ---------------------------------------------------------------------------
# run_validate
# ---------------------------------------------------------------------------

def test_run_validate_raises_when_empty():
    cur = FakeCursor(fetchone_map={"COUNT(*)": (0,)})
    with pytest.raises(Exception, match="GDRIVE_SOURCE_FOLDER_ID"):
        dag.run_validate(cur, SRC, FACILITY)


def test_run_validate_raises_when_table_does_not_exist():
    class MissingTableCursor(FakeCursor):
        def execute(self, sql, params=None):
            if "COUNT(*)" in sql:
                raise RuntimeError("SQL compilation error: Object 'HOSPITALS.SIAYA_DOCS.EXTRACTED_FORMS' does not exist or not authorized")
            return super().execute(sql, params)
    with pytest.raises(Exception, match="does not exist"):
        dag.run_validate(MissingTableCursor(), SRC, FACILITY)


def test_run_validate_returns_count():
    cur = FakeCursor(fetchone_map={"patient_name::STRING IS NULL": (0,), "COUNT(*)": (10,)})
    assert dag.run_validate(cur, SRC, FACILITY) == 10


def test_run_validate_calls_check_null_names(monkeypatch):
    calls = []
    monkeypatch.setattr(dag, "check_null_names", lambda cur, src: calls.append(True) or 0)
    cur = FakeCursor(fetchone_map={"COUNT(*)": (4,)})
    dag.run_validate(cur, SRC, FACILITY)
    assert calls == [True]


# ---------------------------------------------------------------------------
# run_write — SQL order and content
# ---------------------------------------------------------------------------

def test_run_write_returns_counts():
    cur = FakeCursor(rowcount=7)
    result = dag.run_write(cur, SRC, DST, FACILITY)
    assert result == {"deleted": 7, "patients": 7, "visits": 7, "vitals": 7, "doctor_notes": 7,
                      "inv_products": 0, "inv_stock_movements": 0}


def test_run_write_delete_before_inserts():
    cur = FakeCursor()
    dag.run_write(cur, SRC, DST, FACILITY)
    # BEGIN, DELETE, patient INSERT, visit INSERT, vitals INSERT, doctor_notes INSERT, COMMIT
    assert cur.executed[0] == "BEGIN"
    assert "DELETE" in cur.executed[1]
    assert "'patients'" in cur.executed[2]
    assert "'visits'" in cur.executed[3]
    assert "'vitals'" in cur.executed[4]
    assert "'doctor_notes'" in cur.executed[5]
    assert cur.executed[6] == "COMMIT"


def test_run_write_rolls_back_on_failure():
    cur = FakeCursor(fail_on="nurses_triage_notes")  # unique to visit INSERT
    with pytest.raises(RuntimeError):
        dag.run_write(cur, SRC, DST, FACILITY)
    assert "ROLLBACK" in cur.executed
    assert "COMMIT" not in cur.executed


def test_run_write_delete_filters_by_file_id():
    cur = FakeCursor()
    dag.run_write(cur, SRC, DST, FACILITY)
    delete_sql = cur.executed[1]   # executed[0] = BEGIN
    assert "payload:file_id::STRING" in delete_sql
    assert "module_source" not in delete_sql
    assert "source_table IN ('patients', 'visits', 'vitals', 'doctor_notes', 'inv_products', 'inv_stock_movements')" in delete_sql


def test_run_write_patient_payload_fields():
    cur = FakeCursor()
    dag.run_write(cur, SRC, DST, FACILITY)
    patient_sql = cur.executed[2]  # BEGIN, DELETE, patient INSERT
    for field in ("'id'", "'file_id'", "'last_name'", "'patient_no'", "'date_of_birth'",
                  "'age'", "'gender'", "'phone'", "'address'"):
        assert field in patient_sql, f"Missing field {field} in patient INSERT"
    assert "RECORD_KEY" in patient_sql
    assert "EXTRACTION:patient_name::STRING" in patient_sql
    assert "COALESCE(EXTRACTION:camp_file_number::STRING, EXTRACTION:patient_id_file_no::STRING)" in patient_sql
    assert "TRY_CAST(SPLIT_PART(EXTRACTION:age::STRING" in patient_sql
    assert "TRY_TO_DATE" in patient_sql


def test_run_write_visit_payload_fields():
    cur = FakeCursor()
    dag.run_write(cur, SRC, DST, FACILITY)
    visit_sql = cur.executed[3]    # BEGIN, DELETE, patient INSERT, visit INSERT
    for field in ("'id'", "'file_id'", "'patient_id'", "'date'", "'notes'"):
        assert field in visit_sql, f"Missing field {field} in visit INSERT"
    assert "RECORD_KEY || '_v'" in visit_sql
    assert "RECORD_KEY" in visit_sql
    assert "COALESCE(EXTRACTION:nurses_triage_notes::STRING, EXTRACTION:history_of_present_illness::STRING, EXTRACTION:doctors_notes::STRING)" in visit_sql
    assert "TRY_TO_DATE" in visit_sql


def test_run_write_vitals_payload_fields():
    cur = FakeCursor()
    dag.run_write(cur, SRC, DST, FACILITY)
    vitals_sql = cur.executed[4]   # BEGIN, DELETE, patient, visit, vitals
    for field in ("'visit_id'", "'patient_id'", "'temperature'", "'systolic_bp'",
                  "'diastolic_bp'", "'pulse'", "'weight'", "'height'", "'bmi'",
                  "'respiratory_rate'", "'oxygen_saturation'"):
        assert field in vitals_sql, f"Missing field {field} in vitals INSERT"
    assert "TRY_CAST" in vitals_sql
    assert "SPLIT_PART" in vitals_sql
    assert "EXTRACTION:blood_pressure::STRING" in vitals_sql
    assert "RECORD_KEY || '_vitals'" in vitals_sql
    assert "RECORD_KEY || '_v'" in vitals_sql


def test_run_write_doctor_notes_payload_fields():
    cur = FakeCursor()
    dag.run_write(cur, SRC, DST, FACILITY)
    dn_sql = cur.executed[5]       # BEGIN, DELETE, patient, visit, vitals, doctor_notes
    for field in ("'visit_id'", "'patient_id'", "'chief_complaints'",
                  "'history_of_present_illness'", "'examination'", "'surgical_history'", "'plan'"):
        assert field in dn_sql, f"Missing field {field} in doctor_notes INSERT"
    assert "EXTRACTION:chief_complaints" in dn_sql
    assert "RECORD_KEY || '_dn'" in dn_sql
    assert "RECORD_KEY || '_v'" in dn_sql


def test_run_write_delete_covers_vitals_and_doctor_notes():
    cur = FakeCursor()
    dag.run_write(cur, SRC, DST, FACILITY)
    delete_sql = cur.executed[1]
    assert "'vitals'" in delete_sql
    assert "'doctor_notes'" in delete_sql
    assert "'inv_products'" in delete_sql
    assert "'inv_stock_movements'" in delete_sql


def test_run_write_delete_does_not_filter_module_source():
    cur = FakeCursor()
    dag.run_write(cur, SRC, DST, FACILITY)
    delete_sql = cur.executed[1]
    assert "module_source" not in delete_sql


def test_run_write_rollback_on_vitals_insert_failure():
    cur = FakeCursor(fail_on="body_mass_index")
    with pytest.raises(RuntimeError):
        dag.run_write(cur, SRC, DST, FACILITY)
    assert "ROLLBACK" in cur.executed
    assert "COMMIT" not in cur.executed


def test_run_write_rollback_on_doctor_notes_insert_failure():
    cur = FakeCursor(fail_on="EXTRACTION:chief_complaints")
    with pytest.raises(RuntimeError):
        dag.run_write(cur, SRC, DST, FACILITY)
    assert "ROLLBACK" in cur.executed
    assert "COMMIT" not in cur.executed


def test_run_write_null_vitals_does_not_rollback():
    """0-row vitals INSERT is not fatal — only patients/visits must have rows."""
    counts = iter([5, 5, 5, 0, 5])  # deleted, patients, visits, vitals=0, doctor_notes

    class PartialVitalsCursor(FakeCursor):
        def __init__(self):
            self._fetchone_map = {}
            self.fail_on = None
            self.executed = []

        @property
        def rowcount(self):
            return next(counts, 0)

    cur = PartialVitalsCursor()
    result = dag.run_write(cur, SRC, DST, FACILITY)
    assert result["vitals"] == 0
    assert "COMMIT" in cur.executed
    assert "ROLLBACK" not in cur.executed


def test_visit_id_distinct_from_patient_id():
    cur = FakeCursor()
    dag.run_write(cur, SRC, DST, FACILITY)
    patient_sql = cur.executed[2]   # after BEGIN, DELETE
    visit_sql = cur.executed[3]
    assert "RECORD_KEY || '_v'" not in patient_sql
    assert "RECORD_KEY || '_v'" in visit_sql


def test_run_write_module_source_is_reception_service():
    cur = FakeCursor()
    dag.run_write(cur, SRC, DST, FACILITY)
    for sql in cur.executed[2:4]:  # patient INSERT + visit INSERT
        assert "'reception-service'" in sql


def test_run_write_raises_when_insert_produces_zero_rows():
    cur = FakeCursor(rowcount=0)
    with pytest.raises(Exception, match="0 rows"):
        dag.run_write(cur, SRC, DST, FACILITY)
    assert "ROLLBACK" in cur.executed
    assert "COMMIT" not in cur.executed


def test_run_write_raises_and_rolls_back_on_partial_failure():
    """visits INSERT produces 0 rows while patients succeeds — must ROLLBACK, not COMMIT."""
    counts = iter([5, 5, 0])  # deleted=5, patients=5, visits=0

    class PartialCursor(FakeCursor):
        def __init__(self):
            self._fetchone_map = {}
            self.fail_on = None
            self.executed = []

        @property
        def rowcount(self):
            return next(counts, 0)

    cur = PartialCursor()
    with pytest.raises(Exception, match="0 rows"):
        dag.run_write(cur, SRC, DST, FACILITY)
    assert "ROLLBACK" in cur.executed
    assert "COMMIT" not in cur.executed


def test_run_write_uses_facility_in_sql():
    cur = FakeCursor()
    dag.run_write(cur, SRC, DST, FACILITY)
    for sql in cur.executed[2:6]:  # patient, visit, vitals, doctor_notes INSERTs
        assert f"'{FACILITY}'" in sql


def test_run_write_evaluation_inserts_use_patient_evaluation_service():
    cur = FakeCursor()
    dag.run_write(cur, SRC, DST, FACILITY)
    for sql in cur.executed[4:6]:  # vitals INSERT + doctor_notes INSERT
        assert "'patient-evaluation-service'" in sql


# ---------------------------------------------------------------------------
# run_verify
# ---------------------------------------------------------------------------

class FakeVerifyCursor(FakeCursor):
    """Returns configurable per-source_table counts from the three verify GROUP BY queries."""

    def __init__(self, patient_count: int = 5, visit_count: int = 5,
                 vitals_count: int = 5, doctor_notes_count: int = 5,
                 inv_products_count: int = 0, inv_stock_movements_count: int = 0):
        super().__init__()
        # Support run_verify()'s NULL check: 0 nulls out of patient_count total → check passes
        self._fetchone_map = {"COUNT_IF": (0, patient_count)}
        self._reception = [("patients", patient_count), ("visits", visit_count)]
        self._evaluation = [("vitals", vitals_count), ("doctor_notes", doctor_notes_count)]
        self._inventory = [("inv_products", inv_products_count), ("inv_stock_movements", inv_stock_movements_count)]

    def fetchall(self):
        last_sql = self.executed[-1] if self.executed else ""
        if "patient-evaluation-service" in last_sql:
            return [(r[0], r[1]) for r in self._evaluation if r[1] > 0]
        if "inventory-service" in last_sql:
            return [(r[0], r[1]) for r in self._inventory if r[1] > 0]
        return [(r[0], r[1]) for r in self._reception if r[1] > 0]


def test_run_verify_passes_when_both_tables_present():
    cur = FakeVerifyCursor(patient_count=3, visit_count=3)
    result = dag.run_verify(cur, DST, FACILITY)
    assert result["patients"] == 3
    assert result["visits"] == 3


def test_run_verify_raises_when_patients_missing():
    cur = FakeVerifyCursor(patient_count=0, visit_count=4)
    with pytest.raises(Exception, match="patients"):
        dag.run_verify(cur, DST, FACILITY)


def test_run_verify_raises_when_visits_missing():
    cur = FakeVerifyCursor(patient_count=4, visit_count=0)
    with pytest.raises(Exception, match="visits"):
        dag.run_verify(cur, DST, FACILITY)


def test_run_verify_zero_vitals_does_not_raise():
    """0-row vitals/doctor_notes is acceptable — no clinical data extracted."""
    cur = FakeVerifyCursor(patient_count=4, visit_count=4, vitals_count=0, doctor_notes_count=0)
    result = dag.run_verify(cur, DST, FACILITY)
    assert result["patients"] == 4
    assert result.get("vitals", 0) == 0


def test_run_verify_returns_all_four_counts():
    cur = FakeVerifyCursor(patient_count=3, visit_count=3, vitals_count=3, doctor_notes_count=3)
    result = dag.run_verify(cur, DST, FACILITY)
    assert set(result.keys()) == {"patients", "visits", "vitals", "doctor_notes"}


def test_run_verify_sql_filters_by_facility_and_module_source():
    cur = FakeVerifyCursor()
    dag.run_verify(cur, DST, FACILITY)
    reception_sql = cur.executed[0]
    evaluation_sql = cur.executed[2]   # executed[1] is the NULL check fetchone query
    assert f"facility_id = '{FACILITY}'" in reception_sql
    assert "module_source = 'reception-service'" in reception_sql
    assert "source_table IN ('patients', 'visits')" in reception_sql
    assert f"facility_id = '{FACILITY}'" in evaluation_sql
    assert "module_source = 'patient-evaluation-service'" in evaluation_sql
    assert "source_table IN ('vitals', 'doctor_notes')" in evaluation_sql


# ---------------------------------------------------------------------------
# DAG constants sanity check
# ---------------------------------------------------------------------------

def test_dag_id():
    assert dag.DAG_ID == "docs_to_events_raw"


def test_module_source_is_reception_service():
    assert dag.MODULE_SOURCE == "reception-service"


# ---------------------------------------------------------------------------
# run_write — stock-taking (inv_products + inv_stock_movements)
# ---------------------------------------------------------------------------

def test_run_write_default_store_absent_skips_stock_inserts():
    cur = FakeCursor()
    result = dag.run_write(cur, SRC, DST, FACILITY)
    assert cur.executed[6] == "COMMIT"
    assert result["inv_products"] == 0
    assert result["inv_stock_movements"] == 0
    assert not any("inv_products" in s or "inv_stock_movements" in s
                   for s in cur.executed[2:6])


def test_run_write_default_store_present_includes_stock_inserts():
    cur = FakeCursor(rowcount=3)
    result = dag.run_write(cur, SRC, DST, FACILITY, STORE_UUID)
    # BEGIN, DELETE, patient, visit, vitals, dn, inv_products, inv_stock_movements, COMMIT
    assert cur.executed[8] == "COMMIT"
    assert result["inv_products"] == 3
    assert result["inv_stock_movements"] == 3


def test_run_write_inv_products_fields():
    cur = FakeCursor()
    dag.run_write(cur, SRC, DST, FACILITY, STORE_UUID)
    inv_products_sql = cur.executed[6]
    assert "'inv_products'" in inv_products_sql
    assert "LATERAL FLATTEN" in inv_products_sql
    assert "RECORD_KEY || '_p_'" in inv_products_sql
    assert "'id'" in inv_products_sql
    assert "'file_id'" in inv_products_sql
    assert "'generic_name'" in inv_products_sql
    assert "item_name" in inv_products_sql
    assert "'inventory-service'" in inv_products_sql


def test_run_write_inv_stock_movements_fields():
    cur = FakeCursor()
    dag.run_write(cur, SRC, DST, FACILITY, STORE_UUID)
    inv_sm_sql = cur.executed[7]
    assert "'inv_stock_movements'" in inv_sm_sql
    assert "LATERAL FLATTEN" in inv_sm_sql
    assert "RECORD_KEY || '_sm_'" in inv_sm_sql
    assert "'product_id'" in inv_sm_sql
    assert "'from_store_id'" in inv_sm_sql
    assert STORE_UUID in inv_sm_sql
    assert "'quantity'" in inv_sm_sql
    assert "'batch_no'" in inv_sm_sql
    assert "'expiry_date'" in inv_sm_sql
    assert "'opening_stock'" in inv_sm_sql
    assert "'inventory-service'" in inv_sm_sql


def test_run_write_rollback_on_inv_products_insert_failure():
    cur = FakeCursor(fail_on="item_name")  # unique to inv_products INSERT
    with pytest.raises(RuntimeError):
        dag.run_write(cur, SRC, DST, FACILITY, STORE_UUID)
    assert "ROLLBACK" in cur.executed
    assert "COMMIT" not in cur.executed


def test_run_write_rollback_on_inv_stock_movements_insert_failure():
    cur = FakeCursor(fail_on="from_store_id")  # unique to inv_stock_movements INSERT
    with pytest.raises(RuntimeError):
        dag.run_write(cur, SRC, DST, FACILITY, STORE_UUID)
    assert "ROLLBACK" in cur.executed
    assert "COMMIT" not in cur.executed


# ---------------------------------------------------------------------------
# run_verify — inventory counts
# ---------------------------------------------------------------------------

def test_run_verify_returns_inventory_counts():
    cur = FakeVerifyCursor(inv_products_count=4, inv_stock_movements_count=4)
    result = dag.run_verify(cur, DST, FACILITY)
    assert result.get("inv_products") == 4
    assert result.get("inv_stock_movements") == 4


def test_run_verify_zero_inventory_not_in_result():
    cur = FakeVerifyCursor()  # inv counts default to 0
    result = dag.run_verify(cur, DST, FACILITY)
    assert "inv_products" not in result
    assert "inv_stock_movements" not in result


def test_run_verify_sql_includes_inventory_query():
    cur = FakeVerifyCursor()
    dag.run_verify(cur, DST, FACILITY)
    inventory_sql = cur.executed[3]   # executed[0]=reception, [1]=null check, [2]=evaluation, [3]=inventory
    assert "inventory-service" in inventory_sql
    assert "inv_products" in inventory_sql
    assert "inv_stock_movements" in inventory_sql
    assert f"facility_id = '{FACILITY}'" in inventory_sql


# ---------------------------------------------------------------------------
# FP-5 — UUID validation for default_store
# ---------------------------------------------------------------------------

def test_run_write_uuid_validation_rejects_invalid_default_store():
    cur = FakeCursor()
    with pytest.raises(AirflowFailException, match="must be a UUID"):
        dag.run_write(cur, SRC, DST, FACILITY, 'not-a-uuid')


def test_run_write_uuid_validation_accepts_valid_uuid():
    cur = FakeCursor(rowcount=3)
    result = dag.run_write(cur, SRC, DST, FACILITY, STORE_UUID)
    assert result["inv_products"] == 3


def test_run_write_uuid_validation_skips_when_empty():
    cur = FakeCursor()
    result = dag.run_write(cur, SRC, DST, FACILITY, '')
    assert result["inv_products"] == 0  # empty default_store skips inventory, no UUID error


# ---------------------------------------------------------------------------
# run_verify — zero-tolerance NULL check
# ---------------------------------------------------------------------------

def test_run_verify_raises_when_all_patient_names_null():
    class AllNullNamesCursor(FakeVerifyCursor):
        def fetchone(self):
            if self.executed and "COUNT_IF" in self.executed[-1]:
                return (5, 5)  # 5 nulls out of 5 total — all NULL
            return super().fetchone()

    cur = AllNullNamesCursor(patient_count=5)
    with pytest.raises(AirflowFailException, match="last_name"):
        dag.run_verify(cur, DST, FACILITY)


def test_run_verify_null_check_queries_last_name_not_patient_name():
    cur = FakeVerifyCursor()
    dag.run_verify(cur, DST, FACILITY)
    null_check_sql = cur.executed[1]   # executed[0]=reception fetchone wait — actually the COUNT_IF query
    assert "last_name" in null_check_sql
    assert "patient_name" not in null_check_sql
