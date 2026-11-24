import gyno_lab_ranges as L

GID = lambda t, s: f"{t}/{s}"
PROCS = [{"id": 1, "code": "GYN-LAB-5", "category": 9}, {"id": 2, "code": "LAB-0105"}, {"id": 3, "code": "LAB-0101"},
         {"id": 4, "code": "GYN-LAB-88"}, {"id": 5, "code": "OGTT"}]
LINES = [{"procedure_id": 1, "subtest_id": 2}, {"procedure_id": 1, "subtest_id": 3}]
GROUPS = [{"id": 7, "group": "both", "lower_value": 0, "upper_value": 146000},
          {"id": 8, "group": "adult", "lower_value": 6570, "upper_value": 146000}]


def test_plan_only_parameters_on_the_template_and_adult_group():
    rows, cover, why = L.plan(PROCS, LINES, GROUPS, GID)
    fhg = [r for r in rows if r["parent_procedure_id"] == 1]
    assert {(r["procedure_id"], r["gender"]) for r in fhg} == {(2, "Male"), (2, "Female")}   # Hgb only: on the template
    assert all(r["age_group_id"] == 8 for r in rows)
    crp = [r for r in rows if r["parent_procedure_id"] == 4]
    assert crp == [{"uuid": "lab_range/GYN-LAB-88:-:Both", "procedure_id": 4, "parent_procedure_id": 4,
                    "is_normal_range": 1, "unit": "mg/L", "age_group_id": 8, "gender": "Both", "type": "range",
                    "lower": 0.0, "upper": 5.0}]
    assert cover[1] == {2} and why["parameter LAB-0124 not on GYN-LAB-5's template (skipped)"] == 1


def test_retire_covered_orphan_and_misattached_rows_keep_the_rest():
    rows, cover, _ = L.plan(PROCS, LINES, GROUPS, GID)
    existing = [{"id": 10, "uuid": "x", "procedure_id": 2, "parent_procedure_id": 1},      # Hgb on FHG: replaced
                {"id": 11, "uuid": "y", "procedure_id": 3, "parent_procedure_id": 1},      # granulocytes: kept
                {"id": 12, "uuid": "z", "procedure_id": None, "parent_procedure_id": 4},   # orphan on CRP: retired
                {"id": 13, "uuid": "w", "procedure_id": 5, "parent_procedure_id": 4},      # CRP row on another test
                {"id": 14, "uuid": "v", "procedure_id": 99, "parent_procedure_id": 98},    # other test: untouched
                {"id": 15, "uuid": rows[0]["uuid"], "procedure_id": 2, "parent_procedure_id": 1}]  # ours: kept
    on_panel = {1: {2, 3}}
    assert [r["id"] for r in L.to_retire(existing, cover, on_panel, {r["uuid"] for r in rows}, None)] == [10, 12, 13]
