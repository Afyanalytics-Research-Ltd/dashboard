from collections import Counter, defaultdict
from itertools import count

import betacare_lab_templates as L
import betacare_to_v3 as b


def test_sheet_loads_every_test_and_parameter_in_order():
    tests = L.load_tests()
    assert len(tests) == 180 and sum(len(t["params"]) for t in tests) == 536
    tbc = tests[0]
    assert tbc["no"] == 1 and tbc["name"].startswith("TOTAL BLOOD COUNT (TBC)") and len(tbc["params"]) == 16
    assert [p["name"] for p in tbc["params"][:3]] == ["WBC", "LYM", "LYM%"]
    assert tbc["params"][0] == {"name": "WBC", "group": None, "unit": "10^3/uL", "range": "4.0 – 10.0"}


def test_mapping_points_at_real_sheet_tests_and_one_template_per_v3_test():
    nos = {t["no"]: t["name"] for t in L.load_tests()}
    assert set(L.TEST_PROCEDURES) <= set(nos)
    codes = [c for cs in L.TEST_PROCEDURES.values() for c in cs]
    assert len(codes) == len(set(codes)), "a V3 test can carry one template only"
    assert L.TEST_PROCEDURES[1] == ("BCH-C269",) and "3-Part" in nos[1]          # Full Haemogram: 3-part (the lab)
    assert "CRP" in nos[3] and L.TEST_PROCEDURES[3] == ("BCH-C260",)
    assert "TSH" not in str([nos[n] for n, cs in L.TEST_PROCEDURES.items() if "BCH-C268" in cs])  # FSH is not TSH


def test_parse_range():
    assert L.parse_range("4.0 – 10.0") == {"type": "range", "lower": 4.0, "upper": 10.0}
    assert L.parse_range(" 0-5 ") == {"type": "range", "lower": 0.0, "upper": 5.0}
    assert L.parse_range("< 5") == {"type": "text", "text_reference": "< 5"}
    graded = "Desirable: < 5.2\nMod. high: 5.2 – 6.2\nHigh: > 6.2"
    assert L.parse_range(graded) == {"type": "text", "text_reference": graded}
    assert L.parse_range(None) == {} and L.parse_range("  ") == {}
    assert L.parse_range("10 – 4")["type"] == "text"                               # reversed: keep as written


def test_result_type_and_payloads():
    assert L.result_type({"range": "1 – 2"}) == "number"
    assert L.result_type({"unit": "IU/mL", "range": "Negative: < 8"}) == "number"
    assert L.result_type({"name": "Colour"}) == "text"
    line = L.line_payload("u", 10, 20, 3, {"name": "HGB", "group": "Red cells", "unit": "g/dl", "range": "11.0 – 16.5"})
    assert line == {"uuid": "u", "procedure_id": 10, "subtest_id": 20, "sort_order": 3, "appear_bold": 0, "title": "Red cells",
                    "lab_min_range": 11.0, "lab_max_range": 16.5, "adult_minrange": 11.0, "adult_maxrange": 16.5}
    rr = L.range_payload("r", 10, 20, {"name": "HGB", "unit": "g/dl", "range": "11.0 – 16.5"}, 14968)
    assert rr["procedure_id"] == 20 and rr["parent_procedure_id"] == 10 and rr["lower"] == 11.0 and rr["age_group_id"] == 14968
    assert L.range_payload("r", 1, 1, {"name": "x", "range": None}, None) is None
    assert L.created_in({"created_at": "2026-11-17T15:40:00.000000Z"}, L.BAD_IMPORT)
    assert not L.created_in({"created_at": "2026-11-17T13:40:00.000000Z"}, L.BAD_IMPORT)


# ── the whole step against a fake V3 ──────────────────────────────────────────────────────────────────────────
class Resp:
    def __init__(self, ok=True):
        self.ok, self.status_code, self.text, self.content = ok, 200 if ok else 422, "", b"{}"

    def json(self):
        return {"success": self.ok}


class FakeV3:
    organization_id = 96

    def __init__(self, tables):
        self.t = defaultdict(list, tables)
        self.updates = []

    def read_all(self, svc, alias, workers=1, per_page=500):
        return [dict(r) for r in self.t[alias]]

    def post(self, svc, body, timeout=120):
        assert body["action"] == "update" and body["destination_tenant_id"] == 96
        self.updates.append((body["model"], tuple(body["ids"]), body["data"]))
        for r in self.t[body["model"]]:
            if r["id"] in body["ids"]:
                r.update(body["data"])
        return Resp()


class FakeRun:
    def __init__(self, v3, dry_run=False):
        self.c, self.dry_run, self.failures, self.posted = v3, dry_run, defaultdict(list), defaultdict(list)
        self.next_id = count(10_000)

    def post(self, svc, alias, payloads):
        payloads = list(payloads)
        if self.dry_run:
            return Counter(would_post=len(payloads))
        rows = self.c.t[alias]
        for p in payloads:
            self.posted[alias].append(p)
            hit = next((r for r in rows if r.get("uuid") == p["uuid"]), None)
            if hit:
                hit.update(p)
            else:
                rows.append({"id": next(self.next_id), **p})
        return Counter(ok=len(payloads))

    def ids_by_uuid(self, svc, alias, fresh=False):
        return {r["uuid"]: r["id"] for r in self.c.t[alias] if r.get("uuid") and not r.get("deleted_at")}


def v3_with_bad_import():
    proc = lambda i, code, name, **k: {"id": i, "uuid": f"p{i}", "code": code, "name": name, "category": 7, "active": True, **k}
    return FakeV3({
        "procedures": [proc(10, "BCH-C305", "URINALYSIS"), proc(11, "BCH-C269", "FULL HAEMOGRAM.COMPLETE BLOOD COUNT"),
                       proc(12, "BCH-C260", "C-CREATIVE PROTEIN LEVEL (C.R.P.)"), proc(99, "LAB-0001", "Something else"),
                       proc(50, "SWMC-20-10", ">20 – 10,000,000 copies/ml"), proc(51, "SWMC-C-PEPTIDE", "C-PEPTIDE"),
                       proc(52, "SWMC-USED", "Ordered once")],
        "template_labs": [
            {"id": 900, "uuid": "bad", "procedure_id": 12, "subtest_id": 12, "created_at": "2026-11-17T15:40:00.000000Z"},
            {"id": 901, "uuid": "own", "procedure_id": 10, "subtest_id": 99, "created_at": "2024-06-08T09:23:06.000000Z"},
            {"id": 902, "uuid": "other", "procedure_id": 99, "subtest_id": 99, "created_at": "2024-06-08T09:23:06.000000Z"}],
        "bio_reference_ranges": [
            {"id": 700, "uuid": "badr", "procedure_id": 99, "parent_procedure_id": 99, "created_at": "2026-11-17T15:41:00Z"},
            {"id": 701, "uuid": "keep", "procedure_id": 99, "parent_procedure_id": 99, "created_at": "2026-09-27T15:04:20Z"}],
        "age_groups": [{"id": 14962, "group": "child", "lower_value": 730, "upper_value": 2190},
                       {"id": 14968, "group": "both", "lower_value": 0, "upper_value": 146000}],
        "investigations": [{"id": 1, "procedure": 52}],
    })


def test_step_builds_templates_and_replaces_the_bad_import():
    v3 = v3_with_bad_import()
    run = FakeRun(v3)
    res = L.lab_templates(run, b.gid)
    retired = res["retired_test"]
    assert next(p for p in v3.t["procedures"] if p["id"] == retired)["active"] == 0
    lines = [r for r in v3.t["template_labs"] if r["procedure_id"] != retired]
    by_panel = defaultdict(list)
    for r in lines:
        by_panel[r["procedure_id"]].append(r)
    # CRP: one self-line; URINALYSIS: its 21 sheet parameters in order; Full Haemogram: the 3-part's 16
    assert [(r["subtest_id"], r["sort_order"]) for r in by_panel[12]] == [(12, 1)]
    assert len(by_panel[10]) == 21 and [r["sort_order"] for r in by_panel[10]] == list(range(1, 22))
    names = {r["id"]: r["name"] for r in v3.t["procedures"]}
    assert [names[r["subtest_id"]] for r in sorted(by_panel[11], key=lambda r: r["sort_order"])][:3] == ["WBC", "LYM", "LYM%"]
    # the bad import's line and the old line on a re-templated test are moved onto the retired test (the gateway can't
    # delete); an unrelated template is not touched; where each came from is reported, to undo it
    gone = {r["id"] for r in v3.t["template_labs"] if r["procedure_id"] == retired}
    assert gone == {900, 901} and all(r["subtest_id"] == retired for r in v3.t["template_labs"] if r["id"] in gone)
    assert {r["id"] for r in v3.t["bio_reference_ranges"] if r["parent_procedure_id"] == retired} == {700}
    assert res["retired_from"]["template_labs"] == {900: [12, 12], 901: [10, 99]}
    assert not any(r.get("deleted_at") for t in ("template_labs", "bio_reference_ranges") for r in v3.t[t])
    assert res["replaced_own_templates"] == [{"id": 10, "code": "BCH-C305", "old_lines": 1}]
    # SWMC: junk deactivated, the one named like an unmapped sheet test reused, an ordered one kept
    active = {r["id"]: r["active"] for r in v3.t["procedures"]}
    assert active[50] == 0 and active[51] is True and active[52] is True
    assert any(r["procedure_id"] == 51 for r in lines)                          # C-PEPTIDE templated on the reused test
    assert res["earlier import's tests kept active (ordered)"] == 1
    # ranges: numeric limits or text, all ages, both sexes
    rr = [r for r in v3.t["bio_reference_ranges"] if r["parent_procedure_id"] == 11]
    wbc = next(r for r in rr if r["test_name"] == "WBC")
    assert (wbc["type"], wbc["lower"], wbc["upper"], wbc["unit"], wbc["age_group_id"], wbc["gender"]) == \
           ("range", 4.0, 10.0, "10^3/uL", 14968, "Both")
    # mapped tests missing from this fake catalogue got a new test of their own, coded BCH-LT<no>
    assert any(r["code"] == "BCH-LT004" for r in v3.t["procedures"])            # ESR (its BCH-C266 isn't here)
    assert not any(r["code"] == "BCH-LT003" for r in v3.t["procedures"])        # CRP went onto BCH-C260
    assert not run.failures


def test_step_is_idempotent():
    v3 = v3_with_bad_import()
    L.lab_templates(FakeRun(v3), b.gid)
    n = (len(v3.t["procedures"]), len(v3.t["template_labs"]), len(v3.t["bio_reference_ranges"]), len(v3.t["subprocedures"]))
    res = L.lab_templates(FakeRun(v3), b.gid)
    assert n == (len(v3.t["procedures"]), len(v3.t["template_labs"]), len(v3.t["bio_reference_ranges"]), len(v3.t["subprocedures"]))
    assert res["template_labs_retired"] == {} and res["bio_reference_ranges_retired"] == {}
    assert res["procedures_deactivated"] == {}


def test_dry_run_writes_nothing():
    v3 = v3_with_bad_import()
    before = {k: [dict(r) for r in v] for k, v in v3.t.items()}
    res = L.lab_templates(FakeRun(v3, dry_run=True), b.gid)
    assert {k: v for k, v in v3.t.items() if v} == before and not v3.updates
    assert res["template_labs"]["would_post"] > 500 and res["template_labs_retired"] == {"would_update": 2}
    assert res["procedures_deactivated"] == {"would_update": 1}
    assert len(res["mapping"]) == 180


def test_step_is_opt_in():
    assert "lab_templates" in b.STEPS and "lab_templates" in b.OPT_IN_STEPS
