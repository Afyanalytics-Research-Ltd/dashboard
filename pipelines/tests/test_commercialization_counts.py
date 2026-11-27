"""commercialization_counts: the gateway body, row shaping, access errors, partial failures, writes (fakes)."""
import json
from datetime import datetime

import pytest

import commercialization_counts as C
from v3_to_v3_client import GatewayError


class Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body
        self.ok = 200 <= status < 300
        self.text = json.dumps(body)

    def json(self):
        return self._body


class Session:
    def __init__(self, resp, direct=None):
        self.resp, self.direct, self.calls = resp, direct, []

    def post(self, url, headers=None, data=None, timeout=None):
        self.calls.append((url, json.loads(data)))
        if url.endswith("/gateway/counts"):
            return self.resp
        return self.direct or Resp(500, {"message": "no direct gateway"})


class Client:
    organization_id = 7

    def __init__(self, resp, direct=None):
        self._session = Session(resp, direct)
        self.urls = {"core": "https://core.example/api/", "reception": "https://reception.example/api/",
                     "evaluation": "https://evaluation.example/api/"}

    def _headers(self, service):
        return {"Authorization": "Bearer t"}


OK = {"success": True, "data": {"results": {
    "patients_by_county": {"success": True, "data": {"total": 30, "truncated": False, "groups": [
        {"key": {"county": "Kisumu", "facility_id": 12}, "label": {"facility_id": "KSH Main"}, "count": 20},
        {"key": {"county": None, "facility_id": 13}, "label": {"facility_id": "Annex"}, "count": 10}]}},
    "diseases": {"success": True, "data": {"total": 5, "truncated": True, "groups_total": 1500, "groups": [
        {"key": {"facility_id": 12, "icd10_variation_id": 401}, "label": {"facility_id": "KSH Main",
                                                                          "icd10_variation_id": "A09 Diarrhoea"},
         "count": 5}]}},
    "notes": {"success": True, "data": {"total": 9, "groups": [
        {"key": {"facility_id": 12}, "label": {"facility_id": "KSH Main"}, "count": 9}]}},
}}}


def test_request_body_is_the_teams_query():
    b = C.request_body()
    assert b["tenants"] == "all" and b["labels"] is True
    assert b["queries"]["patients_by_county"] == {"service": "reception", "model": "patient",
                                                  "group_by": ["county", "facility_id"], "limit": 1000}
    assert b["queries"]["diseases"] == {"service": "evaluation", "model": "visit_icd10_diagnoses",
                                        "group_by": ["facility_id", "icd10_variation_id"], "distinct": "patient_id",
                                        "labels": True, "limit": 1000}
    assert b["queries"]["notes"] == {"service": "evaluation", "model": "doctor_notes", "group_by": ["facility_id"],
                                     "distinct": "patient_id"}


def test_rows_carry_keys_labels_and_counts():
    rows = C.rows_for("diseases", OK["data"]["results"]["diseases"])
    assert rows == [{"FACILITY_ID": 12, "FACILITY_NAME": "KSH Main", "ICD10_VARIATION_ID": 401,
                     "ICD10_NAME": "A09 Diarrhoea", "DISTINCT_PATIENTS": 5,
                     "LABELS": json.dumps({"facility_id": "KSH Main", "icd10_variation_id": "A09 Diarrhoea"}, sort_keys=True)}]
    county = C.rows_for("patients_by_county", OK["data"]["results"]["patients_by_county"])
    assert [(r["COUNTY"], r["FACILITY_ID"], r["PATIENT_COUNT"]) for r in county] == [("Kisumu", 12, 20), (None, 13, 10)]
    assert [c for c, _ in C.table_columns("patients_by_county")] == [
        "RUN_ID", "CAPTURED_AT", "ENVIRONMENT", "COUNTY", "FACILITY_ID", "FACILITY_NAME", "PATIENT_COUNT", "LABELS"]


def test_missing_platform_statistics_access_is_a_clear_error():
    client = Client(Resp(403, {"success": False, "message": "Forbidden. Counting across organizations (tenants) "
                                                             "needs platform statistics access, granted on the server."}))
    with pytest.raises(C.CountAccessError, match="PLATFORM STATISTICS access"):
        C.fetch(client)


def test_header_and_body_sent_to_core():
    client = Client(Resp(200, OK))
    C.fetch(client)
    url, body = client._session.calls[0]
    assert url == "https://core.example/api/v1/gateway/counts" and body == C.request_body()


class Cur:
    def __init__(self, log):
        self.log = log

    def execute(self, sql, params=None):
        self.log.append((sql, params))

    def executemany(self, sql, rows):
        self.log.append((sql, list(rows)))

    def close(self):
        pass


class Conn:
    def __init__(self):
        self.log, self.closed, self.committed = [], False, False

    def cursor(self):
        return Cur(self.log)

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


def _run(monkeypatch, resp, direct=None):
    conn = Conn()
    monkeypatch.setattr(C, "V3Client", lambda key: type("X", (Client,), {"login": lambda self: None})(resp, direct))
    monkeypatch.setattr(C, "snowflake_connect", lambda **k: conn)
    return C.run(platform_key="prod"), conn


def test_run_writes_snapshot_and_latest_tables(monkeypatch, caplog):
    summary, conn = _run(monkeypatch, Resp(200, OK))
    assert summary["failed"] == [] and summary["written"] == {
        "PATIENTS_BY_COUNTY": 2, "DISEASES_BY_FACILITY": 1, "DOCTOR_NOTES_BY_FACILITY": 1}
    sqls = [s for s, _ in conn.log]
    assert "CREATE SCHEMA IF NOT EXISTS HOSPITALS.COMMERCIALIZATION" in sqls
    assert any(s.startswith("INSERT INTO HOSPITALS.COMMERCIALIZATION.DISEASES_BY_FACILITY ") for s in sqls)
    assert any(s.startswith("DELETE FROM HOSPITALS.COMMERCIALIZATION.LATEST_DISEASES_BY_FACILITY") for s in sqls)
    assert any(s.startswith("INSERT INTO HOSPITALS.COMMERCIALIZATION.LATEST_DISEASES_BY_FACILITY ") for s in sqls)
    assert conn.committed and conn.closed
    assert summary["queries"]["diseases"]["truncated"] is True
    assert "hit the 1000-group limit" in caplog.text


def test_one_failed_query_doesnt_stop_the_others(monkeypatch):
    body = json.loads(json.dumps(OK))
    body["data"]["results"]["notes"] = {"success": False, "status": 422, "message": "Model [doctor_notes] is not registered."}
    summary, conn = _run(monkeypatch, Resp(200, body))
    assert summary["failed"] == ["notes"] and "not registered" in summary["queries"]["notes"]["error"]
    assert set(summary["written"]) == {"PATIENTS_BY_COUNTY", "DISEASES_BY_FACILITY"}


def test_failed_query_retried_on_the_services_own_gateway(monkeypatch):
    body = json.loads(json.dumps(OK))
    body["data"]["results"]["notes"] = {"success": False, "message": "route api/v1/api/v1/gateway not found"}
    direct = Resp(200, {"success": True, "data": {"total": 4, "groups": [{"key": {"facility_id": 12}, "count": 4}]}})
    summary, _ = _run(monkeypatch, Resp(200, body), direct)
    assert summary["failed"] == [] and summary["written"]["DOCTOR_NOTES_BY_FACILITY"] == 1


def test_every_query_failing_fails_the_run(monkeypatch):
    body = {"success": True, "data": {"results": {q: {"success": False, "message": "boom"} for q in C.QUERIES}}}
    with pytest.raises(GatewayError, match="every count query failed"):
        _run(monkeypatch, Resp(200, body))
