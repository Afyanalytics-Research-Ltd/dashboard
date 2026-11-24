"""
gyno_lab_ranges.py — Gyno prod's (org 159) adult reference ranges from "Laboratory_Tests_Reference_Ranges.xlsx"
(sheet "Reference Ranges", 2026-11-24) → V3 evaluation/bio_reference_ranges; step `lab_ranges` of gyno_to_v3.py
(opt-in: run it by name with target gyno159).

Which V3 test each sheet test is: the lab's own GYN-LAB-* test that Gyno actually orders (counted from org 159's
investigations, 2026-11-24) — e.g. CBC / Full haemogram / FHG = GYN-LAB-5 "FHG" (4,199 orders). Tests are matched by
code, never by id. A panel's parameters are its existing template lines (evaluation/template_labs) — this step adds or
removes no lines and no tests; it only sets ranges (and fixes two wrong units).

Decisions taken with the user (2026-11-24):
  * where the sheet's limits disagree with its own text / flag columns: FBS & OGTT fasting 3.9–5.5 (limits said 5.3),
    prolactin female 5–25 (limit 35, flag 25), ALP 40–129 (limit 306 is a paediatric value), CRP 0–5 (text 0–10);
  * LH / FSH: male numeric; female as text listing the cycle phases (V3 ranges have sex and age, not cycle phase);
  * UECs, LFT, lipid profile (no ranges in the sheet) are left as they are;
  * total T3 / T4 (nmol/L), because the lab's TFT panel and T4 / T3 tests report nmol/L; the sheet's free T3 / free T4
    (pmol/L) have no test in use, so they are not loaded.
FHG is a 3-part analyser (granulocytes, no neutrophil / eosinophil / basophil split, no RDW): only the parameters it
reports get ranges; its absolute counts, granulocyte % and PCT keep the ranges the lab set.

What it replaces: for every (test, parameter) set here, the range rows already on it — and rows on these tests that
point at no parameter, or at a test outside the panel (e.g. FBS's range was attached to an inactive OGTT) — are moved
onto one inactive test, GYN-RETIRED-LABRANGES-20261124 (the gateway can't delete; deleted_at is protected). The report's
retired_from keeps each row's original procedure / parent, to undo it.
"""
from __future__ import annotations

import logging
from collections import Counter

log = logging.getLogger(__name__)

RETIRED_KEY = "lab-ranges-2026-11-24"
RETIRED_CODE = "GYN-RETIRED-LABRANGES-20261124"
RETIRED_NAME = "Retired: reference ranges replaced 2026-11-24"
RETIRED_NOTE = "Holds the bio reference ranges replaced by the 2026-11-24 reference-range sheet (gateway can't delete)."
LAB_CODE = "GYN-LAB-5"            # FHG — its category is the lab category


def num(lo, hi, gender="Both"):
    return {"gender": gender, "type": "range", "lower": lo, "upper": hi}


def below(v, gender="Both"):
    return {"gender": gender, "type": "less_greater", "lg_type": "<", "lg_value": v}


def text(t, gender="Both"):
    return {"gender": gender, "type": "text", "text_reference": t}


def sexes(male, female):
    return [num(*male, gender="Male"), num(*female, gender="Female")]


RBS_TEXT = ("No fixed reference interval; often <7.8 mmol/L in people without diabetes. "
            "≥11.1 mmol/L with classic symptoms is diabetes-range. Interpret clinically.")
LH_F = "Female: follicular 2.4–12.6; mid-cycle peak 14.0–95.6; luteal 1.0–11.4; postmenopausal 7.7–58.5 IU/L"
FSH_F = "Female: follicular 3.5–12.5; mid-cycle peak 4.7–21.5; luteal 1.7–7.7; postmenopausal 25.8–134.8 IU/L"
PROLACTIN = sexes((4.0, 15.0), (5.0, 25.0))

# test code → {parameter code (None = the test itself): (unit, [ranges])}
SPEC: dict[str, dict] = {
    "GYN-LAB-531": {                                            # OGTT (809 orders)
        "GYN-LAB-129": ("mmol/L", [num(3.9, 5.5)]),             # fasting
        "1HPG": ("mmol/L", [text("No universal reference interval for the 1-hour value in non-pregnant adults; "
                                 "interpret per protocol. Pregnancy OGTT uses different cut-offs.")]),
        "2HPG": ("mmol/L", [below(7.8)]),                       # 7.8–11.0 IGT; ≥11.1 diabetes-range
    },
    "GYN-LAB-129": {None: ("mmol/L", [num(3.9, 5.5)])},        # FBS
    "GYN-LAB-12": {None: ("mmol/L", [text(RBS_TEXT)])},        # RBS (6,869)
    "GYN-LAB-42": {None: ("mmol/L", [text(RBS_TEXT)])},        # RBS INPATIENT
    "GYN-LAB-9": {None: ("%", [num(4.0, 5.6)])},               # HBA1C (585)
    "GYN-LAB-5": {                                              # FHG = full haemogram / CBC (4,199)
        "LAB-0124": ("10^9/L", [num(4.0, 11.0)]),              # WBC
        "LAB-0121": ("10^6/μL", sexes((4.5, 5.9), (4.1, 5.1))),  # RBC (10^6/μL = 10^12/L)
        "LAB-0105": ("g/dL", sexes((13.5, 17.5), (12.0, 15.5))),  # Hgb (female: non-pregnant)
        "LAB-0104": ("%", sexes((41.0, 53.0), (36.0, 46.0))),  # Hct
        "LAB-01111": ("fL", [num(80.0, 100.0)]),               # MCV
        "MCH": ("pg", [num(27.0, 33.0)]),
        "LAB-011011": ("g/dL", [num(32.0, 36.0)]),             # MCHC
        "LP14597-6": ("10^9/L", [num(150.0, 450.0)]),          # platelets
        "LAB-0107": ("%", [num(20.0, 40.0)]),                  # lymphocytes %
        "LAB-0113": ("%", [num(2.0, 8.0)]),                    # monocytes %
    },
    "GYN-LAB-167": {None: ("mIU/L", [num(0.4, 4.0)])},         # TSH (1,249)
    "GYN-LAB-133": {                                            # TFTs (220)
        "LAB-0202": ("mIU/L", [num(0.4, 4.0)]),                # TSH
        "LAB-0080": ("nmol/L", [num(1.2, 2.8)]),               # total T3
        "LAB-0081": ("nmol/L", [num(60.0, 160.0)]),            # total T4
    },
    "GYN-LAB-33": {None: ("nmol/L", [num(60.0, 160.0)])},      # T4 (total)
    "GYN-LAB-32": {None: ("nmol/L", [num(1.2, 2.8)])},         # T3 (total)
    "GYN-LAB-34": {None: ("ng/mL", PROLACTIN)},                # PROLACTIN (MALE) — ordered for both sexes (1,167)
    "GYN-LAB-104": {None: ("ng/mL", PROLACTIN)},               # PROLACTIN
    "GYN-LAB-35": {None: ("IU/L", [num(1.7, 8.6, "Male"), text(LH_F, "Female")])},      # LH
    "GYN-LAB-37": {None: ("IU/L", [text(FSH_F, "Female")])},                           # FSH (FEMALE)
    "GYN-LAB-36": {None: ("IU/L", [num(1.5, 12.4, "Male")])},                          # FSH (MALE)
    "GYN-LAB-168": {None: ("ng/mL", [text("Age- and assay-dependent; no single adult reference interval. "
                                          "Not a stand-alone fertility test.")])},       # AMH
    "GYN-LAB-23": {None: ("mIU/mL", [below(5.0)])},            # Beta HCG, non-pregnant <5
    "GYN-LAB-88": {None: ("mg/L", [num(0.0, 5.0)])},           # CRP (standard)
    "GYN-LAB-110": {None: ("U/mL", [num(0.0, 35.0)])},         # CA 125 GYNO
    "GYN-LAB-79": {None: ("nmol/L", sexes((8.6, 29.0), (0.3, 1.9)))},  # TESTOSTERONE (total)
    "GYN-LAB-210": {None: ("U/L", [num(40.0, 129.0)])},        # ALP (adult)
}
# units V3 has wrong on the parameter itself (subprocedures.units): parameter code → unit
UNIT_FIXES = {"GYN-LAB-104": "ng/mL",   # was ng/L for a 5–35 range
              "MCH": "pg"}              # was empty


def range_payload(uid: str, panel: int, subtest: int, unit: str, r: dict, age_group: int | None) -> dict:
    d = {"uuid": uid, "procedure_id": subtest, "parent_procedure_id": panel, "is_normal_range": 1, "unit": unit,
         "age_group_id": age_group, "gender": r["gender"], "type": r["type"]}
    for k in ("lower", "upper", "lg_type", "lg_value", "text_reference"):
        if k in r:
            d[k] = r[k]
    return d


def plan(procs: list[dict], lines: list[dict], groups: list[dict], gid) -> tuple[list[dict], dict, Counter]:
    """The range rows to write, and per test the parameters they cover. Pure (tests/test_gyno_lab_ranges.py)."""
    by_code = {}
    for p in procs:
        if p.get("code"):
            by_code.setdefault(p["code"], p)
    adult = next((g["id"] for g in sorted(groups, key=lambda g: g["id"])
                  if str(g.get("group")).lower() == "adult" and float(g.get("lower_value") or 0) == 6570
                  and float(g.get("upper_value") or 0) >= 36500), None)
    on_panel: dict[int, set] = {}
    for ln in lines:
        on_panel.setdefault(ln.get("procedure_id"), set()).add(ln.get("subtest_id"))
    rows, cover, why = [], {}, Counter()
    for code, params in SPEC.items():
        panel = by_code.get(code)
        if not panel:
            why[f"test {code} not in V3 (skipped)"] += 1
            continue
        pid = panel["id"]
        cover[pid] = set()
        for pcode, (unit, ranges) in params.items():
            if pcode is None:
                sub = pid
            else:
                sp = by_code.get(pcode)
                if not sp or sp["id"] not in on_panel.get(pid, set()):
                    why[f"parameter {pcode} not on {code}'s template (skipped)"] += 1
                    continue
                sub = sp["id"]
            cover[pid].add(sub)
            for r in ranges:
                rows.append(range_payload(gid("lab_range", f"{code}:{pcode or '-'}:{r['gender']}"), pid, sub, unit, r,
                                          adult))
    return rows, cover, why


def to_retire(existing: list[dict], cover: dict[int, set], on_panel: dict[int, set], keep: set[str],
              retired: int | None) -> list[dict]:
    """Rows on the tests set here that the new ranges replace: on a covered parameter, or pointing at no parameter /
    a test outside the panel. Ranges of parameters the sheet doesn't cover are kept."""
    out, keep = [], {str(u).lower() for u in keep}
    for r in existing:
        if str(r.get("uuid")).lower() in keep or r.get("deleted_at"):
            continue
        panel = r.get("parent_procedure_id") or r.get("procedure_id")
        if panel not in cover or panel == retired:
            continue
        sub = r.get("procedure_id")
        valid = on_panel.get(panel, set()) | {panel}
        if sub in cover[panel] or sub is None or sub not in valid:
            out.append(r)
    return out


def lab_ranges(run, gid) -> dict:
    """Run on gyno_to_v3.Run (`run.c` = V3 client, `run.post`, `run.dry_run`)."""
    c, res = run.c, {}
    live = lambda rows: [r for r in rows if not r.get("deleted_at")]
    procs = live(c.read_all("evaluation", "procedures", workers=8))
    lines = live(c.read_all("evaluation", "template_labs", workers=4))
    groups = live(c.read_all("evaluation", "age_groups"))
    rows, cover, why = plan(procs, lines, groups, gid)
    on_panel: dict[int, set] = {}
    for ln in lines:
        on_panel.setdefault(ln.get("procedure_id"), set()).add(ln.get("subtest_id"))
    by_code = {p["code"]: p for p in procs if p.get("code")}
    lab_cat = (by_code.get(LAB_CODE) or {}).get("category")
    if not lab_cat:
        raise RuntimeError(f"{LAB_CODE} (FHG) not in V3 — is this org 159?")

    # the retired test first, so old rows can be moved onto it
    retired_uuid = gid("lab_retired", RETIRED_KEY)
    res["retired_test"] = dict(run.post("evaluation", "procedures", [{
        "uuid": retired_uuid, "code": RETIRED_CODE, "name": RETIRED_NAME, "category": lab_cat, "active": 0,
        "description": RETIRED_NOTE}]))
    retired = run.ids_by_uuid("evaluation", "procedures").get(retired_uuid)
    if retired is None and run.dry_run:
        retired = -1

    old = to_retire(live(c.read_all("evaluation", "bio_reference_ranges", workers=4)), cover, on_panel,
                    {r["uuid"] for r in rows}, retired)
    res["bio_reference_ranges"] = dict(run.post("evaluation", "bio_reference_ranges", rows))
    res["bio_reference_ranges_retired"] = dict(update_ids(run, "bio_reference_ranges", [r["id"] for r in old],
                                                          {"procedure_id": retired, "parent_procedure_id": retired}))
    res["retired_from"] = {r["id"]: [r.get("procedure_id"), r.get("parent_procedure_id"), r.get("type"), r.get("lower"),
                                     r.get("upper"), r.get("lg_type"), r.get("lg_value"), r.get("gender")] for r in old}

    # wrong units on the parameter itself
    fix = {by_code[cd]["id"]: u for cd, u in UNIT_FIXES.items() if cd in by_code}
    subs = [s for s in c.read_all("evaluation", "subprocedures", workers=8) if s.get("procedure") in fix]
    upd = Counter()
    for s in subs:
        if s.get("units") != fix[s["procedure"]]:
            upd.update(update_ids(run, "subprocedures", [s["id"]], {"units": fix[s["procedure"]]}))
    res["subprocedure_units"] = dict(upd)
    res["tests"] = {by_code_code(procs, pid): len(subs_) for pid, subs_ in cover.items()}
    return {**why, **res, "ranges_planned": len(rows)}


def by_code_code(procs, pid):
    return next((f"{p.get('code')} {p.get('name')}" for p in procs if p["id"] == pid), str(pid))


def update_ids(run, alias: str, ids: list[int], data: dict, chunk: int = 200) -> Counter:
    """The same change on many evaluation rows (gateway update by id). Dry run: only counted."""
    res = Counter()
    if run.dry_run:
        res["would_update"] = len(ids)
        return res
    for i in range(0, len(ids), chunk):
        part = ids[i:i + chunk]
        r = run.c.post("evaluation", {"action": "update", "model": alias, "destination_tenant_id": run.c.organization_id,
                                      "ids": part, "data": data}, timeout=120)
        ok = r.ok and (r.json() if r.content else {}).get("success", True)
        res["ok" if ok else "failed"] += len(part)
        if not ok:
            log.error("  evaluation/%s update %s…: %s %s", alias, part[:3], r.status_code, r.text[:300])
    return res
