"""
v3_to_v3_migration against two in-memory gateways: the target gives every
row an id far from its source id, so any link that isn't remapped shows up.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

import v3_to_v3_links as L
import v3_to_v3_migration as vv


class FakeGateway:
    """Enough of V3Client for the migration: list / read / describe / insert
    (match_on uuid = update in place), one tenant per client."""

    def __init__(self, key, org, facility, models, id_start, unknown_cols=()):
        self.key, self.organization_id, self.facility_id = key, org, facility
        self.meta = models                   # "svc/alias" → {"tenant": col|None, "facility": col|None}
        self.rows: dict[str, list[dict]] = {m: [] for m in models}
        self._ids = itertools.count(id_start)
        self.unknown_cols = set(unknown_cols)
        self.inserts = 0

    def token(self):
        return "t"

    def models(self, svc):
        return [{"alias": m.split("/", 1)[1], "tenant": x.get("tenant"), "facility": x.get("facility"),
                 "operations": ["read", "insert", "describe"]} for m, x in self.meta.items()
                if m.startswith(svc + "/")]

    def describe(self, svc, alias):
        return {"excluded": ["id", "organization_id", "created_at", "updated_at", "deleted_at"]}

    def _visible(self, model):
        tenant = self.meta[model].get("tenant")
        return [r for r in self.rows[model] if not tenant or r.get(tenant) == self.organization_id]

    def read_page(self, svc, alias, page, per_page=500):
        rows = self._visible(f"{svc}/{alias}")
        return [dict(r) for r in rows[(page - 1) * per_page: page * per_page]], {"total": len(rows)}

    def read_all(self, svc, alias, per_page=500, workers=1):
        yield from (dict(r) for r in self._visible(f"{svc}/{alias}"))

    def insert(self, svc, alias, data, match_on="uuid"):
        model = f"{svc}/{alias}"
        bad = self.unknown_cols & set(data)
        if bad:
            return _Resp(422, {"message": f"insert rejected: Unknown column '{sorted(bad)[0]}' in 'field list'"})
        self.inserts += 1
        for r in self.rows[model]:
            if data.get("uuid") and r.get("uuid") == data["uuid"]:
                r.update(data)
                return _Resp(200, {"data": {"id": r["id"], "created": False}})
        row = {**data, "id": next(self._ids)}
        if self.meta[model].get("tenant"):
            row[self.meta[model]["tenant"]] = self.organization_id
        self.rows[model].append(row)
        return _Resp(201, {"data": {"id": row["id"], "created": True}})

    def add(self, model, **row):
        tenant = self.meta[model].get("tenant")
        row.setdefault("id", next(self._ids))
        if tenant:
            row.setdefault(tenant, self.organization_id)
        self.rows[model].append(row)
        return row["id"]


class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body
        self.ok = status < 400
        self.text = json.dumps(body)

    def json(self):
        return self._body


T = {"tenant": "organization_id"}
MODELS = {
    "core/departments": T, "core/users": T,
    "reception/patient": {"tenant": "organization_id", "facility": "facility_id"},
    "evaluation/visits": T, "evaluation/investigations": T,
    "evaluation/investigation_results": T,
    "evaluation/investigation_result_details": {"tenant": None},
    "evaluation/symptoms": {"tenant": None},
    "core/organizations": {"tenant": None},
}


def _pair(unknown_cols=()):
    src = FakeGateway("src", 4, 4, MODELS, id_start=1)
    dst = FakeGateway("dst", 250, 247, MODELS, id_start=90_000, unknown_cols=unknown_cols)
    src.add("core/organizations", uuid="o4", name="Kisumu")
    d1 = src.add("core/departments", uuid="d1", name="Lab", head_user_id=None)
    u1 = src.add("core/users", uuid="u1", username="ann", department_id=d1, created_by=None)
    u2 = src.add("core/users", uuid="u2", username="ben", department_id=d1, created_by=u1)
    src.rows["core/departments"][0]["head_user_id"] = u2          # departments ↔ users cycle
    p1 = src.add("reception/patient", uuid="p1", first_name="Amy", facility_id=4, created_by=u1)
    p2 = src.add("reception/patient", uuid="p2", first_name="Bo", facility_id=4, created_by=u2)
    v1 = src.add("evaluation/visits", uuid="v1", patient=p1, user=u1, status="open")
    v2 = src.add("evaluation/visits", uuid="v2", patient=p2, user=u2, status="closed")
    i1 = src.add("evaluation/investigations", uuid="i1", visit=v1, procedure_id=None, ordered_by=u1)
    i2 = src.add("evaluation/investigations", uuid="i2", visit=v2, procedure_id=None, ordered_by=u2)
    r1 = src.add("evaluation/investigation_results", uuid="r1", investigation_id=i1, published_by=u2, status="final")
    src.add("evaluation/investigation_result_details", uuid="rd1", investigation_result_id=r1, value="5.1")
    # another org's detail row — the model has no tenant column, the parent isn't org 4's
    src.add("evaluation/investigation_result_details", uuid="rdX", investigation_result_id=777, value="9")
    src.add("evaluation/symptoms", uuid="s-src-1", name="Fever")
    src.add("evaluation/symptoms", uuid="s-src-2", name="Rash")
    dst.add("evaluation/symptoms", uuid="s-dst-1", name="fever")   # target already has it (other uuid)
    return src, dst


def _run(src, dst, tmp_path, **kw):
    state = vv.State("src", "dst", root=tmp_path)
    plan = vv.build_plan(src)
    copier = vv.Copier(src, dst, state, plan, workers=2, **kw)
    order = sorted((m for m, x in plan["models"].items() if x["selected"]), key=lambda m: plan["models"][m]["tier"])
    results = {m: copier.copy_model(m) for m in order}
    for m in order:
        copier.relink(m)
    return state, plan, copier, order, results


def test_state_namespace_is_configurable_and_preserves_legacy_layout(tmp_path):
    state = vv.State.for_pipeline("billing/import", "daily/2026-11-05", root=tmp_path)
    state.save("checkpoint", "customers/table", {"cursor": "page-4"})
    assert state.dir == vv.CheckpointStore.for_pipeline(tmp_path, "billing/import", "daily/2026-11-05").root
    assert state.load("checkpoint", "customers/table", {}) == {"cursor": "page-4"}
    assert vv.State("src", "dst", root=tmp_path).dir == tmp_path / "v3v3_src_to_dst"
    assert vv.State("src", "dst", root=tmp_path)._path("id_map", "core/departments").name == \
        "core__departments.json"


def test_state_namespace_rejects_path_traversal(tmp_path):
    state = vv.State.for_pipeline("../outside", "../../run", root=tmp_path)
    assert state.dir.is_relative_to(tmp_path)


def test_plan_orders_parents_first_and_skips_the_org_itself():
    src, _ = _pair()
    plan = vv.build_plan(src)
    m = plan["models"]
    assert m["core/organizations"]["kind"] == "skip"
    assert m["evaluation/investigation_result_details"]["kind"] == "parent_rows"
    assert m["evaluation/symptoms"]["kind"] == "reference"
    assert m["reception/patient"]["tier"] < m["evaluation/visits"]["tier"] < m["evaluation/investigations"]["tier"]
    assert m["evaluation/investigation_results"]["tier"] < m["evaluation/investigation_result_details"]["tier"]


def test_audit_ignores_other_orgs_rows_of_a_model_without_tenant_column():
    src, _ = _pair()
    for k in range(50):                     # mostly other orgs' rows: they'd read as broken links
        src.add("evaluation/investigation_result_details", uuid=f"o{k}", investigation_result_id=5000 + k, value="1")
    plan = vv.build_plan(src)
    audit = vv.audit_links(src, plan)
    assert vv.print_plan(plan, audit) == []
    assert audit["evaluation/investigation_result_details"]["investigation_result_id"]["rate"] == 1.0


def test_v2_keyed_columns_of_a_source_without_a_v2_load_are_v3_ids():
    # test284 → prod284: no V2 map, so drug / marked_as_em_by … hold plain V3 ids (they were blanked)
    col = next(iter(L.V2_KEYED))
    assert vv.repair_v2({}, col[0], col[1], 5752302) == (5752302, False)
    assert vv.repair_v2({a: {} for a in set(L.V2_KEYED.values())}, col[0], col[1], 7) == (7, False)
    some = {a: {"500": 3} for a in set(L.V2_KEYED.values())}
    assert vv.repair_v2(some, col[0], col[1], 500) == (3, True)          # a V2-loaded source keeps the guard
    assert vv.repair_v2(some, col[0], col[1], 999) == (None, True)


def test_v2_ids_left_by_the_v2_load_are_repaired_through_its_map(tmp_path):
    src, dst = _pair()
    u1 = next(r["id"] for r in src.rows["core/users"] if r["uuid"] == "u1")
    visits = src.rows["evaluation/visits"]
    visits[0]["marked_as_em_by"] = 500          # V2 user 500 = source user u1
    visits[1]["marked_as_em_by"] = u1           # a V2 id that happens to equal a real V3 id: not in the map
    v2_map = {"users": {"500": u1}}
    state = vv.State("src", "dst", root=tmp_path)
    plan = vv.build_plan(src)
    audit = vv.audit_links(src, plan, v2_map=v2_map)
    assert audit["evaluation/visits"]["marked_as_em_by"]["rate"] == 0.5
    assert vv.print_plan(plan, audit) == []    # source data, not a rule problem
    copier = vv.Copier(src, dst, state, plan, workers=2, v2_map=v2_map)
    res = {m: copier.copy_model(m) for m in sorted(plan["models"], key=lambda m: plan["models"][m]["tier"] or 0)
           if plan["models"][m]["selected"]}
    usr = {r["uuid"]: r["id"] for r in dst.rows["core/users"]}
    got = {r["uuid"]: r.get("marked_as_em_by") for r in dst.rows["evaluation/visits"]}
    assert got == {"v1": usr["u1"], "v2": None}
    assert res["evaluation/visits"]["v2_ids_not_in_map"] == {"marked_as_em_by": 1}
    assert vv.verify_model(copier, "evaluation/visits")["ok"]


def test_full_copy_mirrors_every_row_and_link(tmp_path):
    src, dst = _pair()
    state, plan, copier, order, _ = _run(src, dst, tmp_path)
    for model in order:
        res = vv.verify_model(copier, model)
        assert res["ok"], (model, res)
    # links point at the TARGET's rows
    pat = {r["uuid"]: r["id"] for r in dst.rows["reception/patient"]}
    usr = {r["uuid"]: r["id"] for r in dst.rows["core/users"]}
    v1 = next(r for r in dst.rows["evaluation/visits"] if r["uuid"] == "v1")
    assert v1["patient"] == pat["p1"] and v1["user"] == usr["u1"] and v1["organization_id"] == 250
    assert all(r["facility_id"] == 247 for r in dst.rows["reception/patient"])
    # the cycle is closed by the relink pass
    dept = dst.rows["core/departments"][0]
    assert dept["head_user_id"] == usr["u2"]
    ben = next(r for r in dst.rows["core/users"] if r["uuid"] == "u2")
    assert ben["created_by"] == usr["u1"]


def test_other_orgs_rows_are_not_copied_and_reference_rows_are_linked(tmp_path):
    src, dst = _pair()
    _, _, _, _, results = _run(src, dst, tmp_path)
    details = dst.rows["evaluation/investigation_result_details"]
    assert [r["uuid"] for r in details] == ["rd1"]
    assert results["evaluation/investigation_result_details"]["foreign"] == 1
    symptoms = {r["name"]: r for r in dst.rows["evaluation/symptoms"]}
    assert set(symptoms) == {"fever", "Rash"}                 # Fever linked by name, Rash inserted
    assert results["evaluation/symptoms"] == {"read": 2, "linked": 1, "posted": 1, "failed": 0, "status": "done"}
    assert not dst.rows["core/organizations"]


def test_rerun_posts_nothing_new(tmp_path):
    src, dst = _pair()
    _run(src, dst, tmp_path)
    before = {m: len(r) for m, r in dst.rows.items()}
    _, _, _, _, again = _run(src, dst, tmp_path)
    assert {m: len(r) for m, r in dst.rows.items()} == before
    assert again["evaluation/visits"]["posted"] == 0 and again["evaluation/visits"]["already"] == 2


def test_child_waits_for_its_parent(tmp_path):
    src, dst = _pair()
    state = vv.State("src", "dst", root=tmp_path)
    plan = vv.build_plan(src)
    copier = vv.Copier(src, dst, state, plan, workers=1)
    res = copier.copy_model("evaluation/visits")                 # patients not copied yet
    assert res["held"] == 2 and res["status"] == "waiting" and not dst.rows["evaluation/visits"]
    copier.copy_model("core/departments")
    copier.copy_model("core/users")
    copier.copy_model("reception/patient")
    res = copier.copy_model("evaluation/visits")
    assert res["posted"] == 2 and res["held"] == 0


def test_resolve_window_keywords_dates_and_errors():
    from datetime import datetime
    now = datetime(2026, 11, 19, 1, 30, tzinfo=vv.SOURCE_TZ)
    y0 = datetime(2026, 11, 18, tzinfo=vv.SOURCE_TZ)
    assert vv.resolve_window("", "") is None
    assert vv.resolve_window("yesterday", "", now=now) == (y0, datetime(2026, 11, 19, tzinfo=vv.SOURCE_TZ))
    assert vv.resolve_window("2026-11-18", "", now=now) == (y0, datetime(2026, 11, 19, tzinfo=vv.SOURCE_TZ))
    assert vv.resolve_window("2026-11-17", "2026-11-18", now=now)[1] == datetime(2026, 11, 19, tzinfo=vv.SOURCE_TZ)
    assert vv.resolve_window("2026-11-18 12:00:00", "", now=now) == (y0.replace(hour=12), now)
    for bad in (("", "today"), ("nope", ""), ("today", "yesterday")):
        with pytest.raises(ValueError):
            vv.resolve_window(*bad, now=now)
    win = vv.resolve_window("yesterday", "", now=now)
    assert vv.in_window({"created_at": "2026-11-17T21:00:00.000000Z"}, win)       # 00:00 Nairobi
    assert not vv.in_window({"created_at": "2026-11-17T20:59:59.000000Z"}, win)
    assert vv.in_window({"created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-11-18 23:00:00"}, win)
    assert not vv.in_window({"name": "no timestamps"}, win)
    for zero in ("0000-00-00T00:00:00.000000Z", "0000-12-31T21:00:00.000000Z"):   # patient_meta on test284
        assert vv.to_wall_time(zero) == zero
    # placeholder dates: year 0 in UTC must not crash the copy
    assert not vv.in_window({"created_at": "0001-01-01 00:00:00", "updated_at": "0000-00-00 00:00:00"}, win)
    assert vv.in_window({"created_at": "0001-01-01 00:00:00", "updated_at": "2026-11-18 10:00:00"}, win)
    assert vv.in_window({"name": "no timestamps"}, None)


def test_windowed_copy_posts_only_the_windows_rows_and_links_them_to_older_ones(tmp_path):
    """The target is a clone of the source up to some point; the window's new
    rows are copied and link to the clone's (pre-existing) rows by uuid."""
    from datetime import datetime
    src, dst = _pair()
    for r in (row for rows in src.rows.values() for row in rows):
        r["created_at"] = r["updated_at"] = "2026-10-01T08:00:00.000000Z"
    _run(src, dst, tmp_path / "clone")                       # the clone
    old_p1 = next(r for r in dst.rows["reception/patient"] if r["uuid"] == "p1")
    old_p1["first_name"] = "Amy (edited on the target since)"   # target's own later edit
    p1 = next(r["id"] for r in src.rows["reception/patient"] if r["uuid"] == "p1")
    u1 = next(r["id"] for r in src.rows["core/users"] if r["uuid"] == "u1")
    new_ts = "2026-11-18T09:00:00.000000Z"
    p3 = src.add("reception/patient", uuid="p3", first_name="Cy", facility_id=4, created_by=u1,
                 created_at=new_ts, updated_at=new_ts)
    src.add("evaluation/visits", uuid="v3", patient=p1, user=u1, status="open", created_at=new_ts, updated_at=new_ts)
    src.add("evaluation/visits", uuid="v4", patient=p3, user=u1, status="open", created_at=new_ts, updated_at=new_ts)
    src.add("evaluation/visits", uuid="v-old", patient=p1, user=u1, status="old",       # not in the window
            created_at="2026-11-10T09:00:00.000000Z", updated_at="2026-11-10T09:00:00.000000Z")
    before = {m: len(r) for m, r in dst.rows.items()}
    win = vv.resolve_window("yesterday", "", now=datetime(2026, 11, 19, 10, tzinfo=vv.SOURCE_TZ))
    _, _, copier, order, res = _run(src, dst, tmp_path / "window", window=win)
    grew = {m: len(r) - before[m] for m, r in dst.rows.items() if len(r) != before[m]}
    assert grew == {"reception/patient": 1, "evaluation/visits": 2}
    assert res["evaluation/visits"]["posted"] == 2 and res["evaluation/visits"]["outside_window"] == 1
    assert res["evaluation/visits"]["already"] == 2 and res["evaluation/visits"]["status"] == "done"
    pat = {r["uuid"]: r["id"] for r in dst.rows["reception/patient"]}
    vis = {r["uuid"]: r for r in dst.rows["evaluation/visits"]}
    assert vis["v3"]["patient"] == pat["p1"] and vis["v4"]["patient"] == pat["p3"]
    for model in order:                       # the target's own later edit of p1 is outside the window
        assert vv.verify_model(copier, model)["ok"], model


def test_unreachable_service_is_a_gateway_error_so_the_plan_skips_it(monkeypatch):
    import requests
    from v3_to_v3_client import V3Client, GatewayError

    c = V3Client("x", urls={s: f"https://{s}.invalid/api/" for s in vv.SERVICES}, username="u", password="p")

    def boom(*a, **k):
        raise requests.exceptions.SSLError("Hostname mismatch")

    monkeypatch.setattr(c, "post", boom)
    with pytest.raises(GatewayError, match="inventory list: SSLError"):
        c.models("inventory")
    with pytest.raises(GatewayError, match="describe"):
        c.describe("inventory", "products")


@pytest.mark.parametrize("model", ["evaluation/patient_meta", "evaluation/visit_symptoms",
                                   "evaluation/examination_review_details"])
def test_patient_and_visit_owned_models_without_tenant_are_parent_rows(model):
    # a "reference" kind would post every org's rows (V3 refused 126 other-org patient_meta rows)
    assert vv.model_kind(model, {"tenant": None}, {"id", "uuid", "patient_id", "visit_id"}) == "parent_rows"


def test_shared_row_whose_parent_is_another_orgs_is_not_posted(tmp_path):
    # evaluation/subprocedures on test284: 622 other orgs' rows went to prod with procedure NULL
    src, dst = _pair()
    _run(src, dst, tmp_path)
    u1 = next(r["id"] for r in src.rows["core/users"] if r["uuid"] == "u1")
    src.add("evaluation/symptoms", uuid="s-ours", name="Cough", created_by=u1)
    src.add("evaluation/symptoms", uuid="s-theirs", name="Itch", created_by=987654)   # another org's user
    _, _, _, _, res = _run(src, dst, tmp_path)
    names = {r["name"] for r in dst.rows["evaluation/symptoms"]}
    assert "Cough" in names and "Itch" not in names
    assert res["evaluation/symptoms"]["foreign"] == 1 and res["evaluation/symptoms"]["posted"] == 1


def _cloned_pair(n_patients=40, before_clone=None):
    """src = test instance, dst = its clone (same ids / uuids / org), both since grown apart."""
    import copy
    src, _ = _pair()
    old = "2026-10-01T08:00:00.000000Z"
    u1 = next(r["id"] for r in src.rows["core/users"] if r["uuid"] == "u1")
    for k in range(n_patients):
        src.add("reception/patient", uuid=f"bulk{k}", first_name=f"B{k}", facility_id=4, created_by=u1)
    if before_clone:
        before_clone(src, u1)
    for r in (row for rows in src.rows.values() for row in rows):
        r["created_at"] = r["updated_at"] = old
    dst = FakeGateway("dst", 4, 4, MODELS, id_start=1)
    dst.rows = copy.deepcopy(src.rows)
    nxt = next(src._ids)
    src._ids, dst._ids = itertools.count(nxt), itertools.count(nxt)
    return src, dst, u1


def _count_reads(gw):
    gw.reads = 0
    orig = gw.read_page

    def counted(*a, **k):
        gw.reads += 1
        return orig(*a, **k)
    gw.read_page = counted


def test_tail_mode_reads_only_the_diverged_pages_and_links_to_the_clone(tmp_path, monkeypatch):
    from datetime import datetime
    monkeypatch.setattr(vv, "TAIL_PAGE", 4)
    src, dst, u1 = _cloned_pair()
    new_ts = "2026-11-18T09:00:00.000000Z"
    p1 = next(r["id"] for r in src.rows["reception/patient"] if r["uuid"] == "p1")
    # the target (prod) kept working: its own new rows take the same ids the test instance used
    dst.add("reception/patient", uuid="p-prod", first_name="Prod", facility_id=4, created_by=u1,
            created_at=new_ts, updated_at=new_ts)
    # the test instance: a new patient, and visits for an old and the new patient
    pn = src.add("reception/patient", uuid="p-new", first_name="New", facility_id=4, created_by=u1,
                 created_at=new_ts, updated_at=new_ts)
    src.add("evaluation/visits", uuid="v-old-pat", patient=p1, user=u1, status="open",
            created_at=new_ts, updated_at=new_ts)
    src.add("evaluation/visits", uuid="v-new-pat", patient=pn, user=u1, status="open",
            created_at=new_ts, updated_at=new_ts)
    _count_reads(src)
    _count_reads(dst)
    win = vv.resolve_window("yesterday", "", now=datetime(2026, 11, 19, 10, tzinfo=vv.SOURCE_TZ))
    state = vv.State("src", "dst", root=tmp_path)
    plan = vv.build_plan(src)
    src.reads = dst.reads = 0
    copier = vv.Copier(src, dst, state, plan, workers=2, window=win, tail=True)
    order = sorted((m for m, x in plan["models"].items() if x["selected"]), key=lambda m: plan["models"][m]["tier"])
    res = {m: copier.copy_model(m) for m in order}
    pat = {r["uuid"]: r["id"] for r in dst.rows["reception/patient"]}
    vis = {r["uuid"]: r for r in dst.rows["evaluation/visits"]}
    assert "p-new" in pat and pat["p-new"] != pn                    # posted under a new id (pn is prod's p-prod)
    assert vis["v-old-pat"]["patient"] == p1                         # before the divergence: same id on both
    assert vis["v-new-pat"]["patient"] == pat["p-new"]               # posted in this run: mapped
    assert vis["v-old-pat"]["user"] == u1
    assert res["reception/patient"]["posted"] == 1 and res["evaluation/visits"]["posted"] == 2
    assert len([r for r in dst.rows["reception/patient"] if r["uuid"] == "p-new"]) == 1
    pages = -(-len(src.rows["reception/patient"]) // 4)
    assert copier.tail_point("reception/patient")["same_pages"] >= pages - 2
    assert copier.tail_point("reception/patient")["clone_upto"] < pn
    assert src.reads + dst.reads < 4 * sum(-(-len(r) // 4) + 1 for r in src.rows.values())   # far from a full read
    for m in order:
        assert vv.verify_model(copier, m)["ok"], m
    # a re-run posts nothing
    again = vv.Copier(src, dst, state, plan, workers=2, window=win, tail=True)
    assert all(again.copy_model(m).get("posted", 0) == 0 for m in order)


def test_tail_mode_fills_links_an_earlier_run_left_empty_never_overwrites(tmp_path, monkeypatch):
    from datetime import datetime
    monkeypatch.setattr(vv, "TAIL_PAGE", 4)
    src, dst, u1 = _cloned_pair()
    new_ts = "2026-11-18T09:00:00.000000Z"
    p1 = next(r["id"] for r in src.rows["reception/patient"] if r["uuid"] == "p1")
    src.add("evaluation/visits", uuid="v-x", patient=p1, user=u1, status="open",
            created_at=new_ts, updated_at=new_ts)
    src.add("evaluation/visits", uuid="v-y", patient=p1, user=u1, status="open",
            created_at=new_ts, updated_at=new_ts)
    # an earlier run copied them (new ids, after prod's own new rows) with the user link
    # empty; on v-y prod has since set its own
    dst.add("evaluation/visits", uuid="v-prod", patient=p1, user=u1, status="open",
            created_at=new_ts, updated_at=new_ts)
    src.add("evaluation/visits", uuid="v-z", patient=p1, user=u1, status="open",
            created_at=new_ts, updated_at=new_ts)
    stamped = "2026-11-20T05:00:00.000000Z"          # an earlier patch stamped "now"
    dst.add("evaluation/visits", uuid="v-x", patient=p1, user=None, status="open",
            created_at=new_ts, updated_at=stamped)
    dst.add("evaluation/visits", uuid="v-y", patient=p1, user=777, status="open",
            created_at=new_ts, updated_at="2026-11-19T10:00:00.000000Z")    # prod's own later edit
    dst.add("evaluation/visits", uuid="v-z", patient=p1, user=u1, status="open",
            created_at=new_ts, updated_at=stamped)
    win = vv.resolve_window("yesterday", "", now=datetime(2026, 11, 19, 10, tzinfo=vv.SOURCE_TZ))
    plan = vv.build_plan(src)
    copier = vv.Copier(src, dst, vv.State("src", "dst", root=tmp_path), plan, workers=1, window=win, tail=True)
    for m in sorted((m for m, x in plan["models"].items() if x["selected"]), key=lambda m: plan["models"][m]["tier"]):
        res = copier.copy_model(m)
    vis = {r["uuid"]: r for r in dst.rows["evaluation/visits"]}
    original = vv.to_wall_time(new_ts)
    assert vis["v-x"]["user"] == u1                      # empty on the target: filled
    assert vis["v-x"]["updated_at"] == original          # … and the original record's updated_at, not "now"
    assert vis["v-y"]["user"] == 777                     # the target's own value: kept
    assert vis["v-y"]["updated_at"] == "2026-11-19T10:00:00.000000Z"   # its own later edit: untouched
    assert vis["v-z"]["updated_at"] == original          # only the stamp differed: restored
    visits = copier.state.load("report", "evaluation/visits", {})
    assert visits["posted"] == 0 and visits["links_filled"] == {"user": 1}
    assert visits["updated_at_restored"] == 2


def test_tail_mode_identity_links_for_a_model_the_source_cant_read(tmp_path, monkeypatch):
    from datetime import datetime
    monkeypatch.setattr(vv, "TAIL_PAGE", 4)
    src, dst, u1 = _cloned_pair()
    plan = vv.build_plan(src)
    plan["models"]["evaluation/visits"]["links"]["drug"] = "inventory/product"     # not in the plan
    win = vv.resolve_window("yesterday", "", now=datetime(2026, 11, 19, 10, tzinfo=vv.SOURCE_TZ))
    row = {"id": 1, "uuid": "z", "drug": 5753122, "created_at": "2026-11-18T09:00:00.000000Z"}
    plain = vv.Copier(src, dst, vv.State("a", "b", root=tmp_path), plan, window=win, tail=True)
    assert plain.translate("evaluation/visits", row)[0]["drug"] is None
    ident = vv.Copier(src, dst, vv.State("a", "b", root=tmp_path), plan, window=win, tail=True,
                      identity_links=["inventory/"])
    assert ident.translate("evaluation/visits", row)[0]["drug"] == 5753122
    with pytest.raises(ValueError):
        vv.Copier(src, dst, vv.State("a", "b", root=tmp_path), plan, tail=True)


def test_tail_never_posts_another_orgs_row_of_a_tenantless_table(tmp_path, monkeypatch):
    """investigation_result_details has no tenant column: the gateway returns every org's
    rows. A new row whose parent result is ANOTHER org's must not land in this org."""
    from datetime import datetime
    monkeypatch.setattr(vv, "TAIL_PAGE", 4)
    ids = {}

    def other_org_result_then_ours(src, u1):
        i1 = next(r["id"] for r in src.rows["evaluation/investigations"] if r["uuid"] == "i1")
        ids["theirs"] = src.add("evaluation/investigation_results", uuid="r-org99", investigation_id=i1,
                                published_by=u1, status="final", organization_id=99)
        for k in range(6):          # ours, after it: the divergence point lies beyond the other org's id
            ids[f"ours{k}"] = src.add("evaluation/investigation_results", uuid=f"r-own{k}", investigation_id=i1,
                                      published_by=u1, status="final")

    src, dst, u1 = _cloned_pair(before_clone=other_org_result_then_ours)
    new_ts = "2026-11-18T09:00:00.000000Z"
    src.add("evaluation/investigation_result_details", uuid="rd-theirs", investigation_result_id=ids["theirs"],
            value="7", created_at=new_ts, updated_at=new_ts)
    src.add("evaluation/investigation_result_details", uuid="rd-ours", investigation_result_id=ids["ours0"],
            value="5", created_at=new_ts, updated_at=new_ts)
    win = vv.resolve_window("yesterday", "", now=datetime(2026, 11, 19, 10, tzinfo=vv.SOURCE_TZ))
    plan = vv.build_plan(src)
    copier = vv.Copier(src, dst, vv.State("src", "dst", root=tmp_path), plan, workers=1, window=win, tail=True)
    for m in sorted((m for m, x in plan["models"].items() if x["selected"]), key=lambda m: plan["models"][m]["tier"]):
        copier.copy_model(m)
    assert copier.tail_point("evaluation/investigation_results")["clone_upto"] > ids["theirs"]
    details = {r["uuid"]: r for r in dst.rows["evaluation/investigation_result_details"]}
    assert "rd-ours" in details and details["rd-ours"]["investigation_result_id"] == ids["ours0"]
    assert "rd-theirs" not in details


def test_created_only_window_never_brings_back_a_row_the_target_deleted(tmp_path, monkeypatch):
    from datetime import datetime
    monkeypatch.setattr(vv, "TAIL_PAGE", 4)
    src, dst, u1 = _cloned_pair()
    p1 = next(r["id"] for r in src.rows["reception/patient"] if r["uuid"] == "p1")
    gone = next(r for r in dst.rows["evaluation/visits"] if r["uuid"] == "v1")
    dst.rows["evaluation/visits"].remove(gone)                  # prod deleted v1 after the clone …
    v1 = next(r for r in src.rows["evaluation/visits"] if r["uuid"] == "v1")
    v1["updated_at"] = "2026-11-18T09:00:00.000000Z"            # … while test edited it in the window
    src.add("evaluation/visits", uuid="v-new", patient=p1, user=u1, status="open",
            created_at="2026-11-18T09:00:00.000000Z", updated_at="2026-11-18T09:00:00.000000Z")
    win = vv.resolve_window("yesterday", "", now=datetime(2026, 11, 19, 10, tzinfo=vv.SOURCE_TZ))
    plan = vv.build_plan(src)
    copier = vv.Copier(src, dst, vv.State("src", "dst", root=tmp_path), plan, workers=1, window=win, tail=True,
                       window_cols=("created_at",))
    order = sorted((m for m, x in plan["models"].items() if x["selected"]), key=lambda m: plan["models"][m]["tier"])
    copier.map_only = True                                      # the survey: counts, writes nothing
    before = {m: len(r) for m, r in dst.rows.items()}
    res = {m: copier.copy_model(m) for m in order}["evaluation/visits"]
    assert {m: len(r) for m, r in dst.rows.items()} == before
    assert res["would_post"] == 1 and res["would_post_by_day"] == {"2026-11-18": 1}
    copier.map_only = False
    for m in order:
        copier.copy_model(m)
    uuids = {r["uuid"] for r in dst.rows["evaluation/visits"]}
    assert "v-new" in uuids and "v1" not in uuids


def test_platform_client_acts_for_the_configured_org_and_names_the_facility(monkeypatch):
    from v3_to_v3_client import V3Client, GatewayError
    for k, v in {"USERNAME": "analyst", "PASSWORD": "x", "URL_TEMPLATE": "https://{service}.example/api/",
                 "ORGANIZATION_ID": "53", "FACILITY_ID": "52", "PLATFORM": "1"}.items():
        monkeypatch.setenv(f"AFYA_PT_{k}", v)
    c = V3Client("pt")
    sent = []

    class _R:
        status_code, ok = 200, True

        def json(self):
            return {"access_token": "tok", "user": {"id": 5546, "organization_id": 89}, "tenant": {"id": 89}}

    monkeypatch.setattr(c._session, "post", lambda url, **kw: _R())
    monkeypatch.setattr(c, "post", lambda svc, body, **kw: sent.append(body) or _R())
    V3Client._platform_tokens.clear()
    assert c.token() == "tok" and c.organization_id == 53 and c.facility_id == 52   # not the login's org 89
    c.insert("evaluation", "visits", {"uuid": "u"})
    c.insert("evaluation", "visits", {"uuid": "w"}, facility_id=60)
    assert [(b["destination_tenant_id"], b["destination_facility_id"]) for b in sent] == [(53, 52), (53, 60)]
    monkeypatch.delenv("AFYA_PT_FACILITY_ID")
    with pytest.raises(GatewayError):
        V3Client("pt")                                           # a platform client must name the org it acts for


def test_tail_patient_number_clash_is_renamed_never_paired_with_another_patient(tmp_path, monkeypatch):
    """After a clone both instances hand out the same next patient numbers to DIFFERENT
    patients: the test patient goes in as '<number>-T' and keeps its own visits."""
    from datetime import datetime
    monkeypatch.setattr(vv, "TAIL_PAGE", 4)
    src, dst, u1 = _cloned_pair()
    real_insert = dst.insert

    def unique_patient_no(svc, alias, data, match_on="uuid"):
        if f"{svc}/{alias}" == "reception/patient" and data.get("patient_no"):
            for r in dst.rows["reception/patient"]:
                if r.get("patient_no") == data["patient_no"] and r.get("uuid") != data.get("uuid"):
                    return _Resp(422, {"message": f"insert rejected: Duplicate entry '{data['patient_no']}' for key "
                                                  "'patients.patients_patient_no_unique'"})
        return real_insert(svc, alias, data, match_on=match_on)

    dst.insert = unique_patient_no
    ts = "2026-11-19T09:00:00.000000Z"
    dst.add("reception/patient", uuid="p-prod", first_name="Other", patient_no="PAT001", facility_id=4,
            created_by=u1, created_at=ts, updated_at=ts)                       # prod's own new patient
    pt = src.add("reception/patient", uuid="p-test", first_name="Mine", patient_no="PAT001", facility_id=4,
                 created_by=u1, created_at=ts, updated_at=ts)                  # test's, same number
    src.add("evaluation/visits", uuid="v-test", patient=pt, user=u1, status="open", created_at=ts, updated_at=ts)
    win = vv.resolve_window("2026-11-16", "2026-11-20", now=datetime(2026, 11, 20, 10, tzinfo=vv.SOURCE_TZ))
    plan = vv.build_plan(src)
    copier = vv.Copier(src, dst, vv.State("src", "dst", root=tmp_path), plan, workers=1, window=win, tail=True,
                       window_cols=("created_at",))
    for m in sorted((m for m, x in plan["models"].items() if x["selected"]), key=lambda m: plan["models"][m]["tier"]):
        copier.copy_model(m)
    pats = {r["uuid"]: r for r in dst.rows["reception/patient"]}
    assert pats["p-test"]["patient_no"] == "PAT001-T" and pats["p-prod"]["patient_no"] == "PAT001"
    visit = next(r for r in dst.rows["evaluation/visits"] if r["uuid"] == "v-test")
    assert visit["patient"] == pats["p-test"]["id"] != pats["p-prod"]["id"]      # its own patient, not the other
    rep = copier.state.load("report", "reception/patient", {})
    assert rep.get("renamed") == 1 and not rep.get("paired_by_key")


def _with_filter(gw):
    """Give a FakeGateway the gateway's updated_since / after_id read (ids ascending)."""
    from v3_to_v3_client import FilterUnsupported  # noqa: F401  (the real client raises it when unsupported)

    def read_window(svc, alias, since, before=None, per_page=500, after_id=0):
        gw.window_reads = getattr(gw, "window_reads", 0) + 1
        lo = vv.parse_instant(since)
        for r in sorted(gw._visible(f"{svc}/{alias}"), key=lambda r: r["id"]):
            t = vv.parse_instant(r.get("updated_at"))
            if t is not None and t >= lo and r["id"] > after_id:
                yield dict(r)

    def org_has(svc, alias, row_id):
        return any(r["id"] == row_id for r in gw._visible(f"{svc}/{alias}"))

    gw.read_window, gw.org_has = read_window, org_has
    return gw


def test_filter_reads_copy_only_the_window_and_link_old_parents_as_is(tmp_path, monkeypatch):
    from datetime import datetime
    monkeypatch.setattr(vv, "TAIL_PAGE", 4)
    src, dst, u1 = _cloned_pair()
    _with_filter(src), _with_filter(dst)
    _count_reads(src), _count_reads(dst)
    ts = "2026-11-19T09:00:00.000000Z"
    p1 = next(r["id"] for r in src.rows["reception/patient"] if r["uuid"] == "p1")
    dst.add("reception/patient", uuid="p-prod", first_name="Prod", facility_id=4, created_by=u1,
            created_at=ts, updated_at=ts)                                   # prod's own new row (id collides)
    pn = src.add("reception/patient", uuid="p-new", first_name="New", facility_id=4, created_by=u1,
                 created_at=ts, updated_at=ts)
    src.add("evaluation/visits", uuid="v-old-pat", patient=p1, user=u1, status="open", created_at=ts, updated_at=ts)
    src.add("evaluation/visits", uuid="v-new-pat", patient=pn, user=u1, status="open", created_at=ts, updated_at=ts)
    # p2: prod deleted it after the clone; test then edited it (in the window by updated_at, created before)
    p2 = next(r for r in src.rows["reception/patient"] if r["uuid"] == "p2")
    p2["updated_at"] = ts
    dst.rows["reception/patient"] = [r for r in dst.rows["reception/patient"] if r["uuid"] != "p2"]
    src.add("evaluation/visits", uuid="v-of-p2", patient=p2["id"], user=u1, status="open", created_at=ts, updated_at=ts)
    win = vv.resolve_window("2026-11-16", "2026-11-20", now=datetime(2026, 11, 20, 10, tzinfo=vv.SOURCE_TZ))
    plan = vv.build_plan(src)
    src.reads = dst.reads = 0
    copier = vv.Copier(src, dst, vv.State("src", "dst", root=tmp_path), plan, workers=1, window=win, tail=True,
                       window_cols=("created_at",), filter_reads=True)
    for m in sorted((m for m, x in plan["models"].items() if x["selected"]), key=lambda m: plan["models"][m]["tier"]):
        copier.copy_model(m)
    assert src.reads == 0 and dst.reads == 0 and src.window_reads and dst.window_reads   # no page reads at all
    pats = {r["uuid"]: r["id"] for r in dst.rows["reception/patient"]}
    vis = {r["uuid"]: r for r in dst.rows["evaluation/visits"]}
    assert "p-new" in pats and pats["p-new"] != pn
    assert vis["v-old-pat"]["patient"] == p1                       # untouched by the window: same row on both
    assert vis["v-new-pat"]["patient"] == pats["p-new"]            # posted in this run: mapped
    assert "p2" not in pats                                        # deleted on prod: not brought back …
    assert "v-of-p2" not in vis                                    # … and its visit isn't linked to whatever has that id
    assert str(p2["id"]) in copier._unplaced("reception/patient")


def test_platform_rate_limit_waits_as_v3_says_and_succeeds(monkeypatch):
    from v3_to_v3_client import V3Client
    import v3_to_v3_client as vc
    for k, v in {"USERNAME": "a", "PASSWORD": "x", "URL_TEMPLATE": "https://{service}.example/api/",
                 "ORGANIZATION_ID": "1", "FACILITY_ID": "1", "PLATFORM": "1"}.items():
        monkeypatch.setenv(f"AFYA_RL_{k}", v)
    c = V3Client("rl")
    monkeypatch.setattr(c, "token", lambda: "t")
    slept = []
    monkeypatch.setattr(vc.time, "sleep", lambda s: slept.append(s))

    class R:
        def __init__(self, code, text):
            self.status_code, self.text, self.ok = code, text, code < 400

    answers = [R(429, '{"message":"Too many calls under a grant over every organization (60 a minute). '
                      'Try again in 7 s."}')] * 6 + [R(200, "{}")]
    monkeypatch.setattr(c._session, "post", lambda *a, **k: answers.pop(0))
    r = c.post("core", {"action": "read"}, retries=2)        # 6 rate waits don't use up the 2 attempts
    assert r.status_code == 200 and len(slept) == 6 and all(7 <= s <= 10 for s in slept)


def test_canary_posts_at_most_n_per_model(tmp_path):
    src, dst = _pair()
    _, _, _, order, results = _run(src, dst, tmp_path, record_limit=1)
    assert len(dst.rows["reception/patient"]) == 1
    assert results["reception/patient"]["status"] == "canary"


def test_users_get_a_random_password_and_must_reset(tmp_path):
    src, dst = _pair()
    _run(src, dst, tmp_path)
    pw = [r["password"] for r in dst.rows["core/users"]]
    assert all(len(p) >= 20 for p in pw) and len(set(pw)) == len(pw)
    assert all(r["enforce_password_reset"] == 1 for r in dst.rows["core/users"])


def test_column_the_target_lacks_is_dropped_not_fatal(tmp_path):
    src, dst = _pair(unknown_cols={"status"})
    _, _, copier, _, results = _run(src, dst, tmp_path)
    assert results["evaluation/visits"]["failed"] == 0 and len(dst.rows["evaluation/visits"]) == 2
    assert "status" not in dst.rows["evaluation/visits"][0]


def test_verify_catches_a_changed_field_and_a_missing_row(tmp_path):
    src, dst = _pair()
    _, _, copier, _, _ = _run(src, dst, tmp_path)
    dst.rows["evaluation/visits"][0]["status"] = "tampered"
    dst.rows["reception/patient"].pop()
    v = vv.verify_model(copier, "evaluation/visits")
    p = vv.verify_model(copier, "reception/patient")
    assert not v["ok"] and v["mismatched_fields"] == {"status": 1}
    assert not p["ok"] and p["missing"] == 1


@pytest.mark.parametrize("err, says", [
    ('403 {"success":false,"message":"Forbidden. This endpoint is restricted to superadmin."}', "superadmin role"),
    ('422 {"success":false,"message":"the database refused the row: Duplicate entry \'INV-9\' for key '
     '\'invoices_invoice_no_unique\' (SQL: insert …)"}', "'INV-9' in unique key invoices_invoice_no_unique"),
    ('422 {"success":false,"message":"the database refused the row: Column \'patient_id\' cannot be null"}',
     "requires patient_id"),
    ('422 {"message":"Cannot add or update a child row: a foreign key constraint fails (`x`.`visits`, CONSTRAINT '
     '`f` FOREIGN KEY (`patient`) REFERENCES `patients` (`id`))"}', "links to a patients row"),
    ('422 {"message":"insert rejected: Data too long for column \'notes\' at row 1"}', "notes is longer"),
    ("ConnectionError: Max retries exceeded (Connection refused)", "didn't answer"),
    ('500 {"message":"Server Error"}', "HTTP 5xx"),
])
def test_errors_are_explained(err, says):
    msg, meaning = vv.describe_error(err)
    assert says in meaning, (msg, meaning)
    assert "(SQL:" not in msg


def test_failed_model_says_why_in_words(tmp_path):
    src, dst = _pair()
    real = dst.insert

    def picky(svc, alias, data, match_on="uuid"):
        if alias == "patient" and data.get("first_name") == "Bo":
            return _Resp(422, {"success": False, "message": "the database refused the row: Column 'gender' "
                                                            "cannot be null (SQL: insert into patients …)"})
        return real(svc, alias, data, match_on)

    dst.insert = picky
    state, plan, copier, order, results = _run(src, dst, tmp_path)
    res = results["reception/patient"]
    text = vv.describe_result("reception/patient", res, state)
    assert res["status"] == "failed"
    assert "1 row(s) [insert] — the target requires gender" in text
    assert "V3 said: HTTP 422: the database refused the row: Column 'gender' cannot be null" in text
    assert "uuid p2" in text and "dead_letters" in text
    # its child is held, and says which parent is missing
    vis = results["evaluation/visits"]
    held_text = vv.describe_result("evaluation/visits", vis, state)
    assert "evaluation/visits.patient → reception/patient ids [" in held_text
    line = json.loads((state.dir / "dead_letters" / "reception__patient.jsonl").read_text().splitlines()[0])
    assert line["row"]["_source_id"] and "requires gender" in line["error"]


def test_sync_patches_fields_that_differ_and_explains_missing_rows(tmp_path):
    src, dst = _pair()
    _, _, copier, _, _ = _run(src, dst, tmp_path)
    dst.rows["evaluation/visits"][0]["status"] = "tampered"
    dst.rows["reception/patient"].pop()                       # p2 gone from the target
    res = vv.verify_model(copier, "evaluation/visits", collect=True)
    assert res["mismatched_rows"] == 1 and set(res["_patches"]) == {"v1"}
    s = vv.sync_fields(copier, "evaluation/visits", res["_patches"])
    assert s["patched"] == 1 and s["failed"] == 0
    assert vv.verify_model(copier, "evaluation/visits")["ok"]
    assert dst.rows["evaluation/visits"][0]["status"] == "open"
    p = vv.verify_model(copier, "reception/patient")
    assert p["missing"] == 1 and p["missing_why"] == {
        "not posted yet (new in the source since the copy, or a canary run)": 1}
    assert copier.copy_model("reception/patient")["posted"] == 1          # the reload puts it back
    assert vv.verify_model(copier, "reception/patient")["ok"]


class _ClockGateway(FakeGateway):
    """Like the real V3 instances: stores dates as the wall-clock text it is
    sent, serialises them in UTC ('…T21:00:00.000000Z' for a Nairobi
    midnight), and stamps its own facility whatever is sent."""

    def __init__(self, *a, stamp_facility=None, **kw):
        super().__init__(*a, **kw)
        self.stamp_facility = stamp_facility

    def insert(self, svc, alias, data, match_on="uuid"):
        import re
        stored = {}
        for k, v in data.items():
            m = re.fullmatch(r"(\d{4}-\d\d-\d\d)T(\d\d:\d\d:\d\d)(?:\.\d+)?Z", v) if isinstance(v, str) else None
            stored[k] = f"{m.group(1)} {m.group(2)}" if m else v      # a UTC text is taken as wall time
        if self.stamp_facility and "facility_id" in stored:
            stored["facility_id"] = self.stamp_facility
        return super().insert(svc, alias, stored, match_on)

    def _serialise(self, r):
        return {k: vv.from_wall_time(v) if k in ("dob", "visit_date") else v for k, v in r.items()}

    def read_page(self, svc, alias, page, per_page=500):
        rows, meta = super().read_page(svc, alias, page, per_page)
        return [self._serialise(r) for r in rows], meta

    def read_all(self, svc, alias, per_page=500, workers=1):
        yield from (self._serialise(r) for r in super().read_all(svc, alias))


def test_dates_survive_the_copy_and_a_shifted_copy_is_repaired(tmp_path):
    src, dst0 = _pair()
    dst = _ClockGateway("dst", 250, 247, MODELS, id_start=90_000, stamp_facility=999)
    dst.rows["evaluation/symptoms"] = dst0.rows["evaluation/symptoms"]
    for r in src.rows["reception/patient"]:
        r["dob"] = "2082-03-21T21:00:00.000000Z"                   # DATE 2082-03-22 in Nairobi
    state, plan, copier, order, _ = _run(src, dst, tmp_path)
    assert {r["dob"] for r in dst.rows["reception/patient"]} == {"2082-03-22"}    # stored as the right day
    v = vv.verify_model(copier, "reception/patient")
    assert v["ok"], v
    assert v["facility_differs"] == {"facility_id": 2}            # the target's own stamp: reported, not a mismatch
    # an older copy that posted the UTC text: a day early
    dst.rows["reception/patient"][0]["dob"] = "2082-03-21"
    v = vv.verify_model(copier, "reception/patient", collect=True)
    assert v["mismatched_fields"] == {"dob": 1}
    patch = next(iter(v["_patches"].values()))
    assert patch["dob"] == "2082-03-22" and "facility_id" not in patch
    vv.sync_fields(copier, "reception/patient", v["_patches"])
    assert vv.verify_model(copier, "reception/patient")["ok"]


def test_wall_time_round_trip():
    assert vv.to_wall_time("2082-03-21T21:00:00.000000Z") == "2082-03-22"
    assert vv.to_wall_time("2024-05-31T06:28:47.000000Z") == "2024-05-31 09:28:47"
    assert vv.to_wall_time("not a date") == "not a date" and vv.to_wall_time(5) == 5
    assert vv.from_wall_time("2024-05-31 09:28:47") == "2024-05-31T06:28:47.000000Z"
    assert vv._norm(vv.from_wall_time(vv.to_wall_time("2026-10-05T21:00:00.000000Z"))) == \
        vv._norm("2026-10-05T21:00:00.000000Z")


def test_norm_treats_formats_alike():
    assert vv._norm("2026-10-13T08:00:00.000000Z") == vv._norm("2026-10-13 08:00:00")
    assert vv._norm(True) == vv._norm(1) == vv._norm("1") == vv._norm("1.00")
    assert vv._norm("") is None and vv._norm(None) is None
    assert vv._norm({"a": 1}) == vv._norm('{"a": 1}')


# every id-like column org 4 really has is covered by a rule (snapshot of the
# source's columns: tests/fixtures/v3v3_org4_columns.json)
FIXTURE = Path(__file__).parent / "fixtures" / "v3v3_org4_columns.json"


@pytest.mark.skipif(not FIXTURE.exists(), reason="no column snapshot")
def test_every_id_column_of_org4_has_a_rule():
    cols = json.loads(FIXTURE.read_text())
    gaps = sorted(f"{m}.{c}" for m, cs in cols.items() for c in cs
                  if L.looks_like_id(c) and L.link_target(m, c) is None and c not in L.MODEL_LINKS.get(m, {}))
    assert gaps == []


@pytest.mark.skipif(not FIXTURE.exists(), reason="no column snapshot")
def test_link_targets_are_real_models():
    cols = json.loads(FIXTURE.read_text())
    for m, cs in cols.items():
        for c in cs:
            t = L.link_target(m, c)
            if t not in (None, L.FACILITY, L.DROP):
                assert t in cols, f"{m}.{c} → {t} is not a source model"


def test_canary_verify_compares_only_the_copied_rows(tmp_path):
    src, dst = _pair()
    _, _, copier, order, _ = _run(src, dst, tmp_path, record_limit=1)
    res = vv.verify_model(copier, "reception/patient", canary=True)
    assert res["ok"] and res["source"] == 1 and res["missing"] == 0
    full = vv.verify_model(copier, "reception/patient")
    assert not full["ok"] and full["missing"] == 1
    dst.rows["reception/patient"][0]["first_name"] = "changed"
    assert not vv.verify_model(copier, "reception/patient", canary=True)["ok"]


def test_crash_midway_resumes_without_duplicates_or_new_passwords(tmp_path):
    src, dst = _pair()
    state = vv.State("src", "dst", root=tmp_path)
    plan = vv.build_plan(src)
    copier = vv.Copier(src, dst, state, plan, workers=1)
    for m in ("core/departments", "core/users", "reception/patient"):
        copier.copy_model(m)
    # "crash": the id maps on disk are lost, the target keeps its rows
    import shutil
    shutil.rmtree(state.dir / "id_map")
    passwords = {r["uuid"]: r["password"] for r in dst.rows["core/users"]}
    before = {m: len(r) for m, r in dst.rows.items()}
    fresh = vv.Copier(src, dst, vv.State("src", "dst", root=tmp_path), plan, workers=1)
    for m in ("core/departments", "core/users", "reception/patient"):
        res = fresh.copy_model(m)
        assert res["posted"] == 0, (m, res)
    assert {m: len(r) for m, r in dst.rows.items()} == before
    assert {r["uuid"]: r["password"] for r in dst.rows["core/users"]} == passwords
    # the maps were rebuilt from the target, so children still link correctly
    res = fresh.copy_model("evaluation/visits")
    assert res["posted"] == 2 and res["held"] == 0
    for m in ("core/departments", "core/users"):
        fresh.relink(m)
    for m in ("core/departments", "core/users", "reception/patient", "evaluation/visits"):
        assert vv.verify_model(fresh, m)["ok"], m
