"""
betacare_lab_templates.py — Beta Care Hospital's lab test templates → V3 (org 96), step `lab_templates` of
betacare_to_v3.py (opt-in: run it by name, a full run leaves it out).

Source: dags/jsons/betacare_lab_templates.json, made from the lab's "Lab_Tests_Templates (1).xlsx" (sheet "Flat List",
2026-11-18): 180 tests, each with its parameters in display order — section ("Subtest Group"), units, reference range.

What V3 renders a lab template from (the way the org's own templates, e.g. Full Haemogram, are built):
  * the test is a Lab procedure; each of its parameters is a Lab procedure too, with a subprocedures row (units,
    number / text result, parent = the test);
  * evaluation/template_labs: one line per parameter — procedure_id = the test, subtest_id = the parameter,
    sort_order, title = the section; a one-parameter test has a single line pointing at itself;
  * evaluation/bio_reference_ranges: the parameter's range — lower / upper for "a – b", else the text as written
    ("Desirable: < 5.2 / Mod. high: 5.2 – 6.2 / High: > 6.2" stays text), with the unit, for all ages and both sexes.

Which V3 test gets a template: TEST_PROCEDURES maps a sheet test to the Beta Care tests (BCH-C… = Smart Hospital
charges, the ones with Beta Care's prices) it is — hand-reviewed, one template per V3 test. A sheet test with no
Beta Care counterpart gets its own procedure (BCH-LT<no>, no price yet), or reuses the one an earlier import made
with exactly its name.

Replacing the 2026-11-17 import (an earlier copy of the sheet loaded badly between 15:30 and 16:00 UTC: tests
collapsed to one self-line, wrong parameters, reference-range text made into tests, codes SWMC-…):
  * every template line / reference range of that import, and any other line / range on a test this step
    templates, that this step didn't write is soft-deleted (deleted_at);
  * its SWMC- procedures not reused here are deactivated — never one that an investigation uses.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
# dags/jsons is where the repo keeps tracked data (pipelines/ ignores *.json); in Airflow it is /opt/airflow/dags/jsons
DATA = next((p for p in (_HERE.parent / "dags" / "jsons" / "betacare_lab_templates.json",
                         Path("/opt/airflow/dags/jsons/betacare_lab_templates.json"),
                         _HERE / "betacare_lab_templates.json") if p.exists()),
            _HERE.parent / "dags" / "jsons" / "betacare_lab_templates.json")

# sheet test no → Beta Care tests that ARE that test (reviewed by hand against org 96's catalogue, 2026-11-18)
TEST_PROCEDURES: dict[int, tuple[str, ...]] = {
    1: ("BCH-C269",),                        # TBC 3-part → FULL HAEMOGRAM.COMPLETE BLOOD COUNT (lab confirmed 3-part)
    3: ("BCH-C260",),                        # CRP → C-CREATIVE PROTEIN LEVEL (C.R.P.)
    4: ("BCH-C266",),                        # ESR
    7: ("BCH-C552",),                        # Anti-CCP → Anti-CCP plus
    8: ("BCH-C305",),                        # URINALYSIS
    11: ("BCH-C302",),                       # Troponin I → TROPONIN
    14: ("BCH-C306",),                       # urine m/c/s → URINE CULTURE AND SENSITIVITY
    16: ("BCH-C303", "BCH-C294"),            # U/E/C → UREA AND ELECTROLYTES, Renal Function Tests (UECs)
    17: ("BCH-C282",),                       # LFTs
    18: ("BCH-C251",),                       # BILIRUBIN LEVELS
    19: ("BCH-C262",),                       # CREATININE
    20: ("BCH-C248",),                       # ASOT
    22: ("BCH-C281", "BCH-L34"),             # LIPID PROFILE
    23: ("BCH-C300",),                       # TFTs → Thyroid Function Tests (TFTs) each @ Ksh. 1500
    26: ("BCH-C278",),                       # HBsAg rapid → HEPATITIS B SURFACE ANTIGEN
    27: ("BCH-C279",),                       # HCV rapid → HEPATITIS C
    28: ("BCH-C554",),                       # VITAMIN D3 → VITAMIN D Neo
    30: ("BCH-C295",),                       # RF
    33: ("BCH-C275",),                       # H. pylori antigen → HELICOBACTOR PYLORI - STOOL
    34: ("BCH-C296",),                       # stool SAT → SALMONELLA ANTIGE (STOOL)
    35: ("BCH-C285",),                       # FOB → OCCULT BLOOD (STOOL)
    36: ("BCH-C255",),                       # Brucella antigen (Abortus / Melitensis) → BRUCELLA TEST
    37: ("BCH-C549",),                       # CEA
    38: ("BCH-C550",),                       # CA 19-9
    39: ("BCH-C307", "BCH-C546"),            # VDRL → VDRL/RPR (both)
    40: ("BCH-C453",),                       # P24
    41: ("BCH-C258",),                       # CALCIUM LEVEL
    43: ("BCH-C256",),                       # B/S for MPS → BS FOR MPS
    44: ("BCH-C553",),                       # IgE → Total IgE
    49: ("BCH-C254",),                       # RBS → BLOOD SUGAR (RBS /FBS)
    57: ("BCH-C263", "BCH-L18"),             # D-DIMER
    59: ("BCH-C304",),                       # URIC ACID LEVEL
    60: ("BCH-C274",),                       # HbA1c
    63: ("BCH-C547",),                       # CK-MB
    66: ("BCH-C292", "BCH-C371"),            # PT / PTI / INR / APTT → PROTHROBIN (TIME/I.N.R.), PT
    70: ("BCH-C261",),                       # cortisol (AM) → CORTISOL LEVEL
    73: ("BCH-C277",),                       # Hepatitis A IgM → HEPATITIS A
    74: ("BCH-C250",),                       # BETA-H.C.G. LEVEL
    85: ("BCH-C297",),                       # STOOL FOR CULTURE AND SENSITIVITY
    89: ("BCH-C272",),                       # HVS m/c/s → H.V.S CULTURE AND SENSITIVITY
    102: ("BCH-C249",),                      # GRAM STAIN → ANY GRAM STAINING
    104: ("BCH-C299",),                      # TESTOSTERONE LEVEL
    107: ("BCH-C267",),                      # Estradiol (E2) → ESTROGEN
    108: ("BCH-C283",),                      # LH
    109: ("BCH-C268",),                      # FSH
    110: ("BCH-C291",),                      # PROLACTIN LEVEL
    115: ("BCH-L15",),                       # Sickling Test
    122: ("BCH-C290",),                      # PROGESTERONE
    123: ("BCH-L3",),                        # Hb variants → HEMOGLOBIN VARIANT ANALYSIS
    127: ("BCH-C551",),                      # CA125
    129: ("BCH-C301",),                      # TPHA
    132: ("BCH-C287", "BCH-L14"),            # PBF
    135: ("BCH-C286",),                      # PSA → P.S.A LEVELS
    142: ("BCH-C253",),                      # ABO and Rhesus → BLOOD GROUPING
    151: ("BCH-L7",),                        # Reticulocyte Count
    152: ("BCH-C264",),                      # DIRECT COOMBS TEST
    153: ("BCH-C280",),                      # INDIRECT COOMBS TEST
    156: ("BCH-C247",),                      # ALP → ALKALINE PHOSTASE
    157: ("BCH-C284", "BCH-L28"),            # OGTT
    171: ("BCH-C293",),                      # pus swab m/c/s → PUS SWAB FOR CULTURE AND SESITIVITY
    177: ("BCH-C372",),                      # COVID 19
    179: ("BCH-C445", "BCH-C289", "BCH-C522"),  # urine pregnancy test: "Photodynamic Therapy (PDT)" is Smart
                                                # Hospital's mislabelled PDT (ordered 93×), PREGNACY TEST, strip
}

BAD_IMPORT = ("2026-11-17T15:30:00", "2026-11-17T16:00:00")      # UTC window of the import being replaced
BAD_IMPORT_CODE = "SWMC-"
NEW_CODE = "BCH-LT"
NUM_RANGE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*[–—-]\s*(-?\d+(?:\.\d+)?)\s*$")


# ── the sheet ──────────────────────────────────────────────────────────────────────────────────────────────────
def load_tests(path: Path = DATA) -> list[dict]:
    tests = json.loads(path.read_text(encoding="utf-8"))["tests"]
    for t in tests:
        t["params"] = [p for p in t["params"] if p.get("name")]
    return [t for t in tests if t["params"]]


def norm(s) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).split())


def parse_range(text) -> dict:
    """'4.0 – 10.0' → numeric limits; anything else (one-sided, graded, gendered, words) → text as written."""
    t = (text or "").strip()
    if not t:
        return {}
    m = NUM_RANGE.match(t)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo <= hi:
            return {"type": "range", "lower": lo, "upper": hi}
    return {"type": "text", "text_reference": t}


def result_type(param: dict) -> str:
    return "number" if parse_range(param.get("range")).get("type") == "range" or param.get("unit") else "text"


# ── V3 rows ─────────────────────────────────────────────────────────────────────────────────────────────────────
def procedure_payload(uid: str, code: str, name: str, category, description=None) -> dict:
    return {"uuid": uid, "code": code[:100], "name": name[:255], "category": category, "active": 1,
            "description": description}


def subprocedure_payload(uid: str, procedure: int, parent: int | None, param: dict | None, independent: bool) -> dict:
    d = {"uuid": uid, "procedure": procedure, "parent": parent, "lab_result_options": "[]", "lab_default": "1",
         "lab_ordered_independently": 1 if independent else 0, "lab_multiple_orders_allowed": 0}
    if param:
        d["lab_result_type"] = result_type(param)
        if param.get("unit"):
            d["units"] = param["unit"]
    return d


def line_payload(uid: str, panel: int, subtest: int, k: int, param: dict) -> dict:
    d = {"uuid": uid, "procedure_id": panel, "subtest_id": subtest, "sort_order": k, "appear_bold": 0,
         "title": param.get("group")}
    r = parse_range(param.get("range"))
    if r.get("type") == "range":
        d.update(lab_min_range=r["lower"], lab_max_range=r["upper"], adult_minrange=r["lower"], adult_maxrange=r["upper"])
    return d


def range_payload(uid: str, panel: int, subtest: int, param: dict, age_group: int | None) -> dict | None:
    r = parse_range(param.get("range"))
    if not r:
        return None
    return {"uuid": uid, "procedure_id": subtest, "parent_procedure_id": panel, "test_name": param["name"][:255],
            "is_normal_range": 1, "gender": "Both", "unit": param.get("unit"), "age_group_id": age_group, **r}


def created_in(row: dict, window: tuple[str, str]) -> bool:
    return window[0] <= str(row.get("created_at") or "")[:19] < window[1]


# ── the step ────────────────────────────────────────────────────────────────────────────────────────────────────
def lab_templates(run, gid) -> dict:
    """Run on betacare_to_v3.Run (`run.c` = V3 client, `run.post`, `run.dry_run`); `gid` = its stable-uuid function."""
    c, res, why = run.c, {}, Counter()
    tests = load_tests()
    live = lambda rows: [r for r in rows if not r.get("deleted_at")]
    procs = live(c.read_all("evaluation", "procedures", workers=8))
    by_code = {}
    for p in procs:
        by_code.setdefault(p.get("code"), p)
    lab_cat = (by_code.get("BCH-C305") or {}).get("category")
    if not lab_cat:
        raise RuntimeError("BCH-C305 URINALYSIS not in V3: run the catalogue step first")

    # 1. which V3 test(s) each sheet test templates
    swmc = {norm(p["name"]): p for p in procs if str(p.get("code") or "").startswith(BAD_IMPORT_CODE)}
    targets: dict[int, list[dict]] = {}
    new_panels, mapping = [], []
    for t in tests:
        codes = TEST_PROCEDURES.get(t["no"], ())
        found = [by_code[cd] for cd in codes if cd in by_code]
        why["sheet test → Beta Care test"] += bool(found)
        missing = [cd for cd in codes if cd not in by_code]
        if missing:
            why["mapped Beta Care test not in V3"] += len(missing)
        if not found and norm(t["name"]) in swmc:
            found = [swmc[norm(t["name"])]]
            why["sheet test → earlier import's test (reused)"] += 1
        if not found:
            new_panels.append(procedure_payload(gid("lab_test", t["no"]), f"{NEW_CODE}{t['no']:03d}", t["name"], lab_cat,
                                                text_of(t)))
            why["sheet test → new test"] += 1
        targets[t["no"]] = found
        mapping.append({"no": t["no"], "test": t["name"], "params": len(t["params"]),
                        "v3": [f"{p.get('code')} {p.get('name')}" for p in found] or [f"new {NEW_CODE}{t['no']:03d}"]})
    # the parameters of multi-parameter tests are procedures of their own
    params = [procedure_payload(gid("lab_param", f"{t['no']}:{k}"), f"{NEW_CODE}{t['no']:03d}-{k:02d}", p["name"], lab_cat)
              for t in tests if len(t["params"]) > 1 for k, p in enumerate(t["params"], 1)]
    res["procedures"] = dict(run.post("evaluation", "procedures", new_panels + params))
    ids = run.ids_by_uuid("evaluation", "procedures", fresh=True)
    if run.dry_run:                     # nothing was created: stand-in ids so the rest can be counted
        fake = iter(range(-1, -10**6, -1))
        ids = {**{p["uuid"]: next(fake) for p in new_panels + params}, **ids}
    for t in tests:
        if not targets[t["no"]]:
            targets[t["no"]] = [{"id": ids[gid("lab_test", t["no"])], "code": f"{NEW_CODE}{t['no']:03d}", "name": t["name"]}]

    # 2. sub-tests, template lines, reference ranges
    groups = [g for g in live(c.read_all("evaluation", "age_groups")) if g.get("is_active", True)]
    all_ages = next((g["id"] for g in sorted(groups, key=lambda g: g["id"])
                     if str(g.get("group")).lower() == "both" and float(g.get("lower_value") or 0) <= 0
                     and float(g.get("upper_value") or 0) >= 36500), None)
    subs_have = {}
    panel_ids = {p["id"] for ps in targets.values() for p in ps}
    param_ids = {ids.get(gid("lab_param", f"{t['no']}:{k}")) for t in tests for k in range(1, len(t["params"]) + 1)}
    for s in c.read_all("evaluation", "subprocedures", workers=8):
        if s.get("procedure") in panel_ids | param_ids:
            subs_have.setdefault(s["procedure"], s)
    subs, lines, ranges = [], [], []
    for t in tests:
        multi = len(t["params"]) > 1
        for pn, panel in enumerate(targets[t["no"]]):
            pid = panel["id"]
            subs.append(subprocedure_payload(gid("lab_sub", f"panel:{pid}"), pid, pid if multi else None,
                                             None if multi else t["params"][0], True))
            for k, p in enumerate(t["params"], 1):
                st = ids.get(gid("lab_param", f"{t['no']}:{k}")) if multi else pid
                if multi and pn == 0:
                    subs.append(subprocedure_payload(gid("lab_sub", f"param:{t['no']}:{k}"), st, pid, p, False))
                lines.append(line_payload(gid("lab_line", f"{pid}:{k}"), pid, st, k, p))
                rp = range_payload(gid("lab_range", f"{pid}:{k}"), pid, st, p, all_ages)
                if rp:
                    ranges.append(rp)
    # a test / parameter that already has a subprocedures row keeps it, updated in place (one row per procedure)
    ours = {s["uuid"] for s in subs}
    subs_new = [s for s in subs if not (subs_have.get(s["procedure"]) and str(subs_have[s["procedure"]].get("uuid")).lower() not in ours)]
    subs_upd = [(subs_have[s["procedure"]]["id"], s) for s in subs if s not in subs_new]
    res["subprocedures"] = dict(run.post("evaluation", "subprocedures", subs_new))
    res["subprocedures_updated"] = dict(update_each(run, "evaluation", "subprocedures", subs_upd))
    res["template_labs"] = dict(run.post("evaluation", "template_labs", lines))
    res["bio_reference_ranges"] = dict(run.post("evaluation", "bio_reference_ranges", ranges))

    # 3. what this replaces: the old lines / ranges on these tests, and everything the bad import wrote
    keep_lines, keep_ranges = {r["uuid"] for r in lines}, {r["uuid"] for r in ranges}
    old_lines = [r for r in live(c.read_all("evaluation", "template_labs", workers=4))
                 if str(r.get("uuid")).lower() not in keep_lines
                 and (r.get("procedure_id") in panel_ids or created_in(r, BAD_IMPORT))]
    old_ranges = [r for r in live(c.read_all("evaluation", "bio_reference_ranges", workers=4))
                  if str(r.get("uuid")).lower() not in keep_ranges
                  and (r.get("parent_procedure_id") in panel_ids or r.get("procedure_id") in panel_ids
                       or created_in(r, BAD_IMPORT))]
    replaced = Counter(r["procedure_id"] for r in old_lines if not created_in(r, BAD_IMPORT))
    res["replaced_own_templates"] = [{"id": pid, "code": next((p.get("code") for p in procs if p["id"] == pid), None),
                                      "old_lines": n} for pid, n in replaced.most_common()]
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    res["template_labs_removed"] = dict(update_ids(run, "evaluation", "template_labs", [r["id"] for r in old_lines],
                                                   {"deleted_at": now}))
    res["bio_reference_ranges_removed"] = dict(update_ids(run, "evaluation", "bio_reference_ranges",
                                                          [r["id"] for r in old_ranges], {"deleted_at": now}))
    reused = {p["id"] for ps in targets.values() for p in ps}
    retire = [p for p in procs if str(p.get("code") or "").startswith(BAD_IMPORT_CODE) and p["id"] not in reused
              and p.get("active") not in (0, False, "0")]
    used = {r.get("procedure") for r in c.read_all("evaluation", "investigations", workers=8)}
    kept = [p for p in retire if p["id"] in used]
    why["earlier import's tests kept active (ordered)"] += len(kept)
    res["procedures_deactivated"] = dict(update_ids(run, "evaluation", "procedures",
                                                    [p["id"] for p in retire if p["id"] not in used], {"active": 0}))
    res["mapping"] = mapping
    res["all_ages_group"] = all_ages
    return {**why, **res}


def text_of(t: dict) -> str | None:
    bits = [f"Sample: {t['sample']}" if t.get("sample") else None, f"Method: {t['method']}" if t.get("method") else None]
    return " | ".join(b for b in bits if b) or None


def update_ids(run, svc: str, alias: str, ids: list[int], data: dict, chunk: int = 200) -> Counter:
    """The same change on many rows (gateway update by id). Dry run: only counted."""
    res = Counter()
    if run.dry_run:
        res["would_update"] = len(ids)
        return res
    for i in range(0, len(ids), chunk):
        part = ids[i:i + chunk]
        r = run.c.post(svc, {"action": "update", "model": alias, "destination_tenant_id": run.c.organization_id,
                             "ids": part, "data": data}, timeout=120)
        ok = r.ok and (r.json() if r.content else {}).get("success", True)
        res["ok" if ok else "failed"] += len(part)
        if not ok:
            run.failures[f"{svc}/{alias} update"].append({"ids": part[:5], "error": f"{r.status_code} {r.text[:300]}"})
            log.error("  %s/%s update %s…: %s %s", svc, alias, part[:3], r.status_code, r.text[:300])
    return res


def update_each(run, svc: str, alias: str, rows: list[tuple[int, dict]]) -> Counter:
    res = Counter()
    for rid, data in rows:
        if run.dry_run:
            res["would_update"] += 1
            continue
        d = {k: v for k, v in data.items() if k != "uuid" and v is not None}
        res.update(update_ids(run, svc, alias, [rid], d))
    return res
