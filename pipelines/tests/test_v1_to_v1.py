"""v1_to_v1: old V1 server → new V1 server after a clone. A fake V1 data gateway (in memory) stands in for both
servers: keyset reads, model `ids` reads, model posts with autoincrement ids (all-or-nothing per request),
MODEL_GATEWAY_WRITE_MODELS (403 / 422)."""
from __future__ import annotations

import copy
import json

import pytest
import requests

import v1_to_v1 as V

R, E = V.R, V.E
MODEL_TABLE = {p["model"]: p["table"] for p in V.PLAN}
KEYED = {p["table"]: p["key"] for p in V.PLAN if p.get("key")}


class FakeServer(V.V1Gateway):
    """A V1 server: tables of rows; gateway calls answered in memory."""

    def __init__(self, name, tables=None, writable=None, reject=None):
        super().__init__(f"https://{name}.example", "u", "p", name=name, sleep=lambda s: None)
        self.db = {t: {} for t in [p["table"] for p in V.PLAN] + ["users", "finance_payments"]}
        for t, rows in (tables or {}).items():
            for r in rows:
                self.db.setdefault(t, {})[r[KEYED.get(t, "id")]] = dict(r)
        self.writable_models = set(MODEL_TABLE) if writable is None else set(writable)
        self.reject = reject or (lambda model, row: None)       # → error message to refuse a row
        self.posts = []

    def call(self, body, timeout=180):
        self.calls += 1
        a = body.get("action")
        if a == "tables":
            return {"tables": [{"table": t, "exists": True} for t in self.db]}
        if a == "describe":
            t = body["table"]
            key = KEYED.get(t, "id")
            return {"table": t, "primary_key": [key], "paging": "keyset",
                    "final_id": max(self.db[t], default=0)}
        if a == "get" and "table" in body:
            t, after, n = body["table"], int(body.get("after_id", 0)), int(body.get("per_page", 100))
            ids = sorted(i for i in self.db[t] if i > after)
            page = ids[:n]
            return {"success": True, "data": [copy.deepcopy(self.db[t][i]) for i in page],
                    "pagination": {"has_more_pages": len(ids) > n, "next_after_id": page[-1] if page else after}}
        if a == "get" and "namespace" in body:
            t = MODEL_TABLE[body["namespace"]]
            want = {int(i) for i in body["ids"]}
            return {"success": True, "data": [copy.deepcopy(r) for k, r in self.db[t].items() if int(k) in want]}
        if a == "post":
            m, data = body["namespace"], body["data"]
            if m not in self.writable_models:
                raise V.GatewayError("403", status=403, data={"writable": sorted(self.writable_models)})
            if not isinstance(data, list) or any(not isinstance(r, dict) or not r for r in data):
                raise V.GatewayError("422", status=422, data={"message": "Nothing was saved."})
            for r in data:
                err = self.reject(m, r)
                if err:
                    raise V.GatewayError(f"422 {err}", status=422, data={"message": "Nothing was saved."})
            t, ids = MODEL_TABLE[m], []
            key = KEYED.get(t)
            for r in data:
                row = dict(r)
                if key:
                    new = row[key]
                else:
                    new = max(self.db[t], default=0) + 1
                    row["id"] = new
                self.db[t][new] = row
                ids.append(new)
            self.posts.append((m, len(data)))
            return {"success": True, "created": len(ids), "ids": ids}
        raise AssertionError(body)


def pat(i, first, last, created, dob="1990-01-01", mobile="0700000000"):
    return {"id": i, "first_name": first, "middle_name": "", "last_name": last, "dob": dob, "mobile": mobile,
            "status": 1, "created_at": created, "updated_at": created, "deleted_at": None}


def visit(i, patient, created):
    return {"id": i, "patient": patient, "clinic": 1, "user": 5, "created_at": created, "updated_at": created,
            "deleted_at": None}


def inv(i, v, proc, created):
    return {"id": i, "visit": v, "procedure": proc, "price": "100.00", "created_at": created, "updated_at": created}


def res(i, investigation, created, text="ok"):
    return {"id": i, "investigation": investigation, "results": text, "created_at": created, "updated_at": created}


CLONE = {"reception_patients": 10, "reception_patient_documents": 0, "evaluation_visits": 100,
         "evaluation_visit_destinations": 0, "evaluation_vitals": 0, "evaluation_doctor_notes": 0,
         "evaluation_nursing_notes": 0, "evaluation_investigations": 1000, "evaluation_investigation_results": 500,
         "result_files": 0, "evaluation_sensitivity": 0}
OLD = "2026-11-19 10:00:00"
DAY = "2026-11-20 1{}:00:00"
LIVE = "2026-11-22 0{}:00:00"


def scenario():
    """Shared history up to the clone; then the source took patients 11-12, visits 101-103, investigations 1001-1004,
    results 501-502 — while the target gave ids 11, 101, 1001, 501 to its own records."""
    shared = {
        "reception_patients": [pat(i, f"P{i}", "OLD", OLD) for i in range(9, 11)],
        "evaluation_visits": [visit(100, 9, OLD)],
        "evaluation_investigations": [inv(1000, 100, 7, OLD)],
        "evaluation_investigation_results": [res(500, 1000, OLD)],
        "users": [{"id": 1, "created_at": OLD}],
    }
    src = copy.deepcopy(shared)
    src["reception_patients"] += [pat(11, "ANN", "NEW", DAY.format(0), dob="2000-02-02"),
                                  pat(12, "MABOR", "IDING", DAY.format(1), dob="1989-12-24", mobile="0703 794278")]
    src["evaluation_visits"] += [visit(101, 11, DAY.format(0)), visit(102, 9, DAY.format(1)),   # 102: an old patient
                                 visit(103, 12, DAY.format(2))]
    src["evaluation_investigations"] += [inv(1001, 101, 7, DAY.format(0)), inv(1002, 101, 8, DAY.format(0)),
                                         inv(1003, 102, 7, DAY.format(1)), inv(1004, 100, 9, DAY.format(2))]
    src["evaluation_investigation_results"] += [res(501, 1001, DAY.format(3)), res(502, 1000, DAY.format(4))]
    dst = copy.deepcopy(shared)
    dst["reception_patients"] += [pat(11, "LIVE", "ONE", LIVE.format(1)),
                                  pat(12, "Mabor", "Iding", LIVE.format(2), dob="1989-12-24")]   # re-registered
    dst["evaluation_visits"] += [visit(101, 11, LIVE.format(1))]
    dst["evaluation_investigations"] += [inv(1001, 101, 7, LIVE.format(1))]
    dst["evaluation_investigation_results"] += [res(501, 1001, LIVE.format(2))]
    return FakeServer("src", src), FakeServer("dst", dst)


@pytest.fixture
def target(monkeypatch, tmp_path):
    monkeypatch.setitem(V.TARGETS, "t", {"source_url": "https://src.example", "target_url": "https://dst.example",
                                         "source_conn": "x", "target_conn": "y", "golive": "2026-11-21 00:00:00",
                                         "clone_ids": dict(CLONE), "ignore": {"users"}})
    monkeypatch.setattr(V, "_s3_client", lambda: None)
    return tmp_path


def run(src, dst, out, **kw):
    kw.setdefault("files", False)
    return V.Run("t", src=src, dst=dst, out_dir=out, **kw).run()


def rows_after(server, t, k):
    return [r for i, r in sorted(server.db[t].items()) if i > k]


def test_dry_run_posts_nothing_and_plans(target):
    src, dst = scenario()
    before = copy.deepcopy(dst.db)
    rep = run(src, dst, target)
    assert dst.db == before and dst.posts == []
    t = rep["tables"]
    assert t["reception_patients"]["to_insert"] == 1                          # ANN; MABOR is re-registered
    assert t["reception_patients"]["already_on_target"] == 1
    assert rep["reregistered"] == [{"source_id": 12, "target_id": 12, "name": "MABOR IDING", "dob": "1989-12-24"}]
    # visit 101 (new patient ANN) waits for its patient; 102 (old patient) and 103 (re-registered MABOR) can go
    assert t["evaluation_visits"]["to_insert"] == 2 and t["evaluation_visits"]["waiting_for_parent"] == 1
    assert rep["failures"] == [] and (target / rep["report_path"].split("/")[-1]).exists()


def test_apply_copies_with_new_ids_and_remapped_links(target):
    src, dst = scenario()
    rep = run(src, dst, target, apply=True)
    assert rep["failures"] == [] and not rep["blocked"]
    # target's own records untouched
    assert dst.db["reception_patients"][11]["first_name"] == "LIVE"
    assert dst.db["evaluation_visits"][101]["patient"] == 11 and dst.db["evaluation_visits"][101]["created_at"] == LIVE.format(1)
    new_p = rows_after(dst, "reception_patients", 12)
    assert [(p["first_name"], p["created_at"]) for p in new_p] == [("ANN", DAY.format(0))]
    ann = new_p[0]["id"]
    v = {r["created_at"]: r for r in rows_after(dst, "evaluation_visits", 101)}
    assert v[DAY.format(0)]["patient"] == ann            # new patient → its new id
    assert v[DAY.format(1)]["patient"] == 9              # old patient → unchanged
    assert v[DAY.format(2)]["patient"] == 12             # re-registered → the target's own record
    invs = rows_after(dst, "evaluation_investigations", 1001)
    by = {(r["procedure"], r["created_at"]): r for r in invs}
    v_ann = v[DAY.format(0)]["id"]
    assert by[(7, DAY.format(0))]["visit"] == v_ann and by[(8, DAY.format(0))]["visit"] == v_ann
    assert by[(7, DAY.format(1))]["visit"] == v[DAY.format(1)]["id"]
    assert by[(9, DAY.format(2))]["visit"] == 100        # new investigation on a pre-clone visit
    results = {r["created_at"]: r for r in rows_after(dst, "evaluation_investigation_results", 501)}
    assert results[DAY.format(3)]["investigation"] == by[(7, DAY.format(0))]["id"]
    assert results[DAY.format(4)]["investigation"] == 1000
    assert len(invs) == 4 and len(results) == 2
    assert rep["idmap"]["reception_patients"] == {"11": ann, "12": 12}


def test_rerun_is_idempotent(target):
    src, dst = scenario()
    run(src, dst, target, apply=True)
    snap = copy.deepcopy(dst.db)
    rep = run(src, dst, target, apply=True)
    assert dst.db == snap
    assert all(st.get("to_insert", 0) == 0 for st in rep["tables"].values())
    assert rep["tables"]["evaluation_investigations"]["already_on_target"] == 4


def test_canary_limit_then_rest(target):
    src, dst = scenario()
    rep = run(src, dst, target, apply=True, limit=1)
    assert rep["tables"]["evaluation_investigations"]["inserted"] == 1
    assert rep["tables"]["evaluation_investigations"].get("held_back_by_limit", 0) >= 1
    rep2 = run(src, dst, target, apply=True)
    assert rep2["failures"] == []
    assert len(rows_after(dst, "evaluation_investigations", 1001)) == 4
    assert len(rows_after(dst, "evaluation_visits", 101)) == 3
    assert len(rows_after(dst, "reception_patients", 12)) == 1


def test_models_not_writable_are_blocked_but_rest_is_copied(target):
    src, dst = scenario()
    dst.writable_models = {R + "Patients", R + "NextOfKin"}
    rep = run(src, dst, target, apply=True)
    assert len(rows_after(dst, "reception_patients", 12)) == 1
    assert rows_after(dst, "evaluation_visits", 101) == []
    assert rep["blocked"][E + "Visit"] == 3
    assert "MODEL_GATEWAY_WRITE_MODELS=" in rep["to_enable"] and E + "Visit" in rep["to_enable"]
    # enabled later → a re-run finishes without duplicating the patient
    dst.writable_models = set(MODEL_TABLE)
    rep2 = run(src, dst, target, apply=True)
    assert not rep2["blocked"] and rep2["failures"] == []
    assert len(rows_after(dst, "reception_patients", 12)) == 1
    assert len(rows_after(dst, "evaluation_visits", 101)) == 3


def test_refused_batch_falls_back_to_rows_and_children_wait(target):
    src, dst = scenario()
    dst.reject = lambda m, r: "bad procedure" if m == E + "Investigations" and r.get("procedure") == 8 else None
    rep = run(src, dst, target, apply=True)
    st = rep["tables"]["evaluation_investigations"]
    assert st["inserted"] == 3 and st["failed"] == 1
    assert any(f["table"] == "evaluation_investigations" and f["ids"] == [1002] for f in rep["failures"])
    # result 501 belongs to 1001 (inserted) → copied
    assert len(rows_after(dst, "evaluation_investigation_results", 501)) == 2


def test_child_of_failed_parent_waits(target):
    src, dst = scenario()
    dst.reject = lambda m, r: "nope" if m == E + "Investigations" and r.get("procedure") == 7 and r["visit"] != 100 else None
    rep = run(src, dst, target, apply=True)
    assert rep["tables"]["evaluation_investigation_results"]["waiting_for_parent"] == 1   # 501 → 1001 failed
    assert rep["tables"]["evaluation_investigation_results"]["inserted"] == 1             # 502 → 1000 (pre-clone)


def test_preflight_stops_on_new_data_in_an_unplanned_table(target):
    src, dst = scenario()
    src.db["finance_payments"] = {1: {"id": 1, "created_at": OLD}, 2: {"id": 2, "created_at": DAY.format(0)}}
    dst.db["finance_payments"] = {1: {"id": 1, "created_at": OLD}, 2: {"id": 2, "created_at": LIVE.format(1)}}
    with pytest.raises(V.GatewayError, match="finance_payments"):
        run(src, dst, target, apply=True, preflight=True)
    assert dst.posts == []


def test_preflight_passes_when_unplanned_tables_match(target):
    src, dst = scenario()
    src.db["finance_payments"] = {1: {"id": 1, "created_at": OLD}}
    dst.db["finance_payments"] = {1: {"id": 1, "created_at": OLD}, 2: {"id": 2, "created_at": LIVE.format(1)}}
    rep = run(src, dst, target, preflight=True)
    assert rep["preflight"]["problems"] == [] and rep["preflight"]["checked"] >= 1


def test_tables_param_inserts_only_selected(target):
    src, dst = scenario()
    rep = run(src, dst, target, apply=True, tables=["reception_patients"])
    assert len(rows_after(dst, "reception_patients", 12)) == 1
    assert rows_after(dst, "evaluation_visits", 101) == []
    assert rep["tables"]["evaluation_visits"]["not_selected"] is True


def test_unknown_table_rejected(target):
    src, dst = scenario()
    with pytest.raises(SystemExit):
        V.Run("t", src=src, dst=dst, tables=["finance_payments"])


def test_same_second_siblings_are_matched_one_to_one(target):
    """Two investigations of one visit created in the same second must map to two different target rows."""
    src, dst = scenario()
    run(src, dst, target, apply=True)
    rep = run(src, dst, target)
    m = rep["idmap"]["evaluation_investigations"]
    assert len(set(m.values())) == len(m) == 4


def test_writability_probe_saves_nothing(target):
    _, dst = scenario()
    assert dst.writable(E + "Visit") == (True, [])
    dst.writable_models = {R + "Patients"}
    ok, enabled = dst.writable(E + "Visit")
    assert ok is False and enabled == [R + "Patients"]
    assert dst.posts == []


def test_norm_and_person_matching():
    assert V.norm("1987-12-30 00:00:00") == V.norm("1987-12-30")
    assert V.norm("36.50") == V.norm("36.5") and V.norm("2.0") == "2"
    a = pat(1, "Mabor ", "IDING", OLD, dob="1989-12-24")
    assert V.same_person(a, pat(2, "MABOR", "Iding", LIVE, dob="1989-12-24 00:00:00"))
    assert V.same_person(pat(1, "A", "B", OLD, dob="", mobile="0703 794278"), pat(2, "a", "b", OLD, dob="1", mobile="254703794278"))
    assert not V.same_person(a, pat(2, "MABOR", "IDING", LIVE, dob="1990-01-01", mobile=""))


# ── HTTP layer ──────────────────────────────────────────────────────────────────
class Resp:
    def __init__(self, status, body, headers=None):
        self.status_code, self._b, self.headers = status, body, headers or {"content-type": "application/json"}
        self.ok = status < 400
        self.text = json.dumps(body)

    def json(self):
        return self._b


class Session:
    def __init__(self, script):
        self.script, self.sent = list(script), []

    def post(self, url, json=None, timeout=None, headers=None):
        self.sent.append((url.rsplit("/", 1)[-1], json))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


LOGIN = Resp(200, {"success": {"token": "T"}})


def gw(script):
    return V.V1Gateway("https://x.example/", "u", "p", session=Session(script), sleep=lambda s: None)


def test_post_201_returns_ids():
    g = gw([LOGIN, Resp(201, {"success": True, "created": 2, "ids": [7, 8]})])
    assert g.post(R + "Patients", [{"a": 1}, {"a": 2}]) == [7, 8]


def test_post_id_count_mismatch_is_an_error():
    g = gw([LOGIN, Resp(201, {"success": True, "created": 1, "ids": [7]})])
    with pytest.raises(V.GatewayError, match="expected 2 ids"):
        g.post(R + "Patients", [{"a": 1}, {"a": 2}])


def test_429_waits_and_401_relogs():
    g = gw([LOGIN, Resp(429, {}, {"Retry-After": "1", "content-type": "application/json"}),
            Resp(401, {"message": "x"}), LOGIN, Resp(200, {"success": True, "data": []})])
    assert g.call({"action": "get", "table": "t"}) == {"success": True, "data": []}


def test_reads_retry_network_errors_but_posts_never_resend():
    g = gw([LOGIN, requests.ConnectionError("boom"), Resp(200, {"success": True, "data": []})])
    assert g.call({"action": "get", "table": "t"})["data"] == []
    g2 = gw([LOGIN, requests.ReadTimeout("slow")])
    with pytest.raises(V.GatewayError, match="ReadTimeout"):
        g2.post(R + "Patients", [{"a": 1}])
    assert len(g2.s.sent) == 2                         # login + the one post, never a second


def test_post_5xx_not_retried():
    g = gw([LOGIN, Resp(503, {"message": "down"})])
    with pytest.raises(V.GatewayError, match="503"):
        g.post(R + "Patients", [{"a": 1}])


def test_403_carries_writable_list():
    g = gw([LOGIN, Resp(403, {"success": False, "message": "not enabled", "writable": [R + "Patients"]})])
    assert g.writable(E + "Visit") == (False, [R + "Patients"])


def test_login_failure():
    g = gw([Resp(401, {"message": "bad"})])
    with pytest.raises(V.GatewayError, match="login failed"):
        g.call({"action": "tables"})


def test_files_downloaded_only_when_missing_on_target(target, monkeypatch):
    src, dst = scenario()
    src.db["result_files"] = {1: {"id": 1, "visit_id": 101, "investigation_id": 1001, "file_path": "/evaluation/files/a.pdf",
                                  "created_at": DAY.format(0)},
                              2: {"id": 2, "visit_id": 100, "investigation_id": 1000, "file_path": "/evaluation/files/b.pdf",
                                  "created_at": DAY.format(1)}}
    calls = []

    class R2:
        def __init__(self, code, ctype="application/pdf", content=b"%PDF"):
            self.status_code, self.headers, self.content = code, {"content-type": ctype}, content

    monkeypatch.setattr(V.requests, "head", lambda url, **k: calls.append(("head", url)) or
                        (R2(200) if url.endswith("b.pdf") else R2(404, "text/html")))
    monkeypatch.setattr(V.requests, "get", lambda url, **k: calls.append(("get", url)) or R2(200))
    rep = V.Run("t", src=src, dst=dst, out_dir=target, apply=True, files=True).run()
    f = rep["files"]
    assert f["needed"] == 2 and f["already_on_target"] == 1 and f["downloaded"] == 1
    assert (target / "files" / "evaluation" / "files" / "a.pdf").read_bytes() == b"%PDF"
    assert ("get", "https://src.example/evaluation/files/a.pdf") in calls
    # dry run: only lists
    calls.clear()
    rep = V.Run("t", src=src, dst=dst, out_dir=target, files=True).run()
    assert rep["files"]["missing_on_target"] == ["/evaluation/files/a.pdf"] and not [c for c in calls if c[0] == "get"]
