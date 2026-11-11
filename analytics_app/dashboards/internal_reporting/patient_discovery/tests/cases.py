"""Scenario catalogue. Each case: the outcome that is correct for the business, plus reasons that must appear.

C* cases come from the original harness; A* are the independent reviewer's held-out cases (2026-11-05).
Where the expected outcome is a deliberate conservative choice ("review" where a person could also accept
"proposed"), the case says so in `why`.
"""
import random
from datetime import date

from .fixtures import VISIT_COLS, d, obj, visits


def case(cid, o, status, must=(), must_not=(), known=None, why=""):
    return dict(id=cid, obj=o, status=status, must=set(must), must_not=set(must_not), known=known, why=why)


def build():
    C = []
    add = lambda *a, **k: C.append(case(*a, **k))

    # ---- valid patient identifiers
    add("C01", obj("NEWFAC_CLEAN", "VISITS", VISIT_COLS, visits(1, 300)), "proposed")
    add("C01b", obj("DIALYSISFAC_CLEAN", "VISITS", VISIT_COLS, visits(2, 80, max_visits=74)), "proposed",
        why="chronic clinic: many rows per patient is legitimate")
    add("C01c", obj("GYNOFAC_HISTORICAL_DATA", "CONSULTANTSVISITS",
                    [("ID", "NUMBER"), ("PATIENTID", "TEXT"), ("VISIT_DATE", "DATE")],
                    [r[:3] for r in visits(3, 200)]), "proposed")
    add("C01d", obj("MFH2_HISTORICAL_DATA", "LIMPATIENTVISITDETAILS",
                    [("APTID", "NUMBER"), ("REGID", "NUMBER"), ("APTDATE", "DATE")],
                    [(r[0], int(r[1][1:]), r[2]) for r in visits(4, 300)]), "proposed")
    add("C01e", obj("TINYFAC_CLEAN", "VISITS", VISIT_COLS, visits(5, 10, max_visits=4)), "review", {"SMALL_TABLE"},
        why="too small to judge (pilots are 1-30 rows)")
    rows = visits(6, 300)
    add("C01f", obj("ACTIVEFAC_CLEAN", "VISITS",
                    [("ID", "NUMBER"), ("PATIENT_ID", "TEXT"), ("VISIT_DATE", "DATE"), ("IS_ACTIVE", "TEXT")],
                    [(r[0], r[1], r[2], "false" if r[3] else "true") for r in rows]), "review",
        {"ONLY_INVERTED_DELETION_FLAG"}, why="the refresh cannot use IS_ACTIVE as its deletion column")
    rng = random.Random(7)
    add("C01g", obj("FLOATFAC_HISTORICAL_DATA", "VISITS", [("ID", "NUMBER"), ("REGID", "FLOAT"), ("VISIT_DATE", "DATE")],
                    [(i, float(p) if rng.random() < .5 else p, d(rng))
                     for i, p in enumerate(p for p in range(1, 201) for _ in range(rng.randint(1, 6)))]), "proposed",
        must_not={"KEY_FORMAT_VARIANTS"})
    add("C01h", obj("QUOTEDFAC_CLEAN", "VISITS", [("ID", "NUMBER"), ("patient_id", "TEXT"), ("VISIT_DATE", "DATE")],
                    [r[:3] for r in visits(8, 300)]), "review", {"NEEDS_QUOTED_IDENTIFIER"})

    # ---- encounter-level identifiers
    rng = random.Random(9)
    add("C02", obj("CAMP2_CLEAN", "CAMP_VISITS", [("VISIT_ID", "TEXT"), ("PATIENT_ID", "TEXT"), ("VISIT_DATE", "DATE")],
                   [(f"V{i}", f"S{i + 5000}", d(rng)) for i in range(900)]), "review", {"MOSTLY_ONE_ROW_PER_PERSON"})
    rows = visits(10, 400)
    add("C02b", obj("FALLBACKFAC_CLEAN", "VISITS", [("VISIT_ID", "TEXT"), ("PATIENT_ID", "TEXT"), ("VISIT_DATE", "DATE")],
                    [(f"V{r[0]}", f"V{r[0]}" if i % 10 < 7 else r[1], r[2]) for i, r in enumerate(rows)]),
        "review", {"PERSON_KEY_EQUALS_RECORD_ID"})

    # ---- household-level identifiers
    rng = random.Random(11)
    hh = [(i, f"H{h:05d}", d(rng)) for i, h in enumerate(h for h in range(250) for _ in range(4 * rng.randint(1, 5)))]
    add("C03a", obj("COMMUNITY_CLEAN", "HOME_VISITS", [("VISIT_ID", "NUMBER"), ("PATIENT_NO", "TEXT"), ("VISIT_DATE", "DATE")],
                    hh), "review",
        known="a household number stored as the patient key looks exactly like a patient with several visits")
    add("C03b", obj("COMMUNITY2_CLEAN", "HOME_VISITS",
                    [("VISIT_ID", "NUMBER"), ("PATIENT_ID", "TEXT"), ("HOUSEHOLD_ID", "TEXT"), ("VISIT_DATE", "DATE")],
                    [(i, h, h, x) for i, h, x in hh]), "review", {"HOUSEHOLD_COLUMN_PRESENT"})
    rng = random.Random(12)
    add("C03c", obj("COMMUNITY3_CLEAN", "HOME_VISITS",
                    [("VISIT_ID", "NUMBER"), ("PATIENT_ID", "TEXT"), ("HOUSEHOLD_ID", "TEXT"), ("VISIT_DATE", "DATE")],
                    [(i, f"P{p}", f"H{p // 4}", d(rng)) for i, p in
                     enumerate(p for p in range(1000) for _ in range(rng.randint(1, 4)))]), "review",
        {"HOUSEHOLD_COLUMN_PRESENT"}, why="conservative: any household column means a person confirms the key")

    # ---- file-level identifiers
    rng = random.Random(13)
    fcols = [("ID", "NUMBER"), ("PATIENT_ID", "TEXT"), ("FILENAME", "TEXT"), ("VISIT_DATE", "DATE")]
    add("C04a", obj("CAMP3_CLEAN", "CAMP_VISITS", fcols,
                    [(f * 100 + i, str(i), f"file_{f}.pdf", d(rng)) for f in range(50) for i in range(1, 21)]),
        "review", {"FILE_SOURCED_ROWS"})
    add("C04b", obj("CAMP4_CLEAN", "CAMP_VISITS", fcols,
                    [(i, f"D{f}", f"file_{f}.pdf", d(rng)) for i, f in enumerate(f for f in range(600) for _ in range(2))]),
        "review", {"FILE_SOURCED_ROWS"})
    add("C04c", obj("CAMP5_CLEAN", "CAMP_VISITS", [("ID", "NUMBER"), ("PATIENT_ID", "TEXT"), ("VISIT_DATE", "DATE")],
                    [(i, f"DOC{i}", d(rng)) for i in range(1200)]), "review", {"MOSTLY_ONE_ROW_PER_PERSON"})

    # ---- high singleton rates
    rng = random.Random(14)
    walk = [f"W{i}" for i in range(1900)] + [f"P{p}" for p in range(100) for _ in range(3)]
    add("C05", obj("WALKINFAC_CLEAN", "VISITS", [("ID", "NUMBER"), ("PATIENT_ID", "TEXT"), ("VISIT_DATE", "DATE")],
                   [(i, k, d(rng)) for i, k in enumerate(walk)]), "review", {"MOSTLY_ONE_ROW_PER_PERSON"})

    # ---- duplicate records
    rows = visits(15, 300)
    dup = rows + rows[: len(rows) // 5]
    add("C06a", obj("DUPFAC_CLEAN", "VISITS", VISIT_COLS[:4], [r[:4] for r in dup]), "review", {"DUPLICATE_RECORDS"})
    add("C06b", obj("DUPFAC2_CLEAN", "VISITS", VISIT_COLS[:4], [r[:4] for r in dup], "VIEW"), "review",
        {"DUPLICATE_RECORDS"})
    add("C06c", obj("DOUBLEFAC_CLEAN", "VISITS", VISIT_COLS[:4], [r[:4] for r in rows + rows], "VIEW"), "review",
        {"DUPLICATE_RECORDS"})
    add("C06d", obj("DUPFAC3_CLEAN", "VISITS", VISIT_COLS, dup, "VIEW"), "proposed",
        why="ID + UPDATED_AT: the refresh keeps one row per ID, so duplicates collapse")

    # ---- null, blank and placeholder identifiers
    rng = random.Random(16)
    rows = visits(17, 300)
    junk = [None, "", "   ", "null", "0", "UNKNOWN", "N/A"]
    add("C07a", obj("NULLFAC_CLEAN", "VISITS", VISIT_COLS,
                    [(r[0], junk[i % 7] if rng.random() < .08 else r[1]) + r[2:] for i, r in enumerate(rows)]),
        "review", {"MISSING_PERSON_KEYS"})
    add("C07b", obj("NULLFAC2_CLEAN", "VISITS", VISIT_COLS, [(r[0], None if rng.random() < .25 else r[1]) + r[2:]
                                                              for r in rows]), "review", {"MISSING_PERSON_KEYS"})
    add("C07c", obj("NULLFAC3_CLEAN", "VISITS", VISIT_COLS, [(r[0], "999999" if rng.random() < .08 else r[1]) + r[2:]
                                                              for r in rows]), "review", {"MISSING_PERSON_KEYS"})
    add("C07d", obj("NULLFAC4_CLEAN", "VISITS", VISIT_COLS, [(r[0], "XX-0000" if rng.random() < .10 else r[1]) + r[2:]
                                                              for r in rows]), "review", {"DOMINANT_KEY_VALUE"},
        why="an unlisted placeholder")

    # ---- multiple versions of the same source
    add("C08a", obj("KISUMU_CLEAN", "ADMISSIONS", VISIT_COLS, [], "VIEW"), "ignored", {"SCHEMA_EXCLUDED"})
    add("C08b", obj("KISUMU_V3_CLEAN", "VISITS", VISIT_COLS, rows + rows, "VIEW",
                    refs=["HOSPITALS.KISUMU_V3_RAW.EVENTS_RAW"]), "review", {"SAME_FACILITY_AS_ACTIVE:HOSPITALS.KISUMU_V3_CLEAN.EVALUATION_VISITS"})
    add("C08c", obj("ORTHOPEDIC_CLEAN_V3", "VISITS", VISIT_COLS, visits(18, 300)), "review",
        {"COVERED_BY:HOSPITALS.STAGING.STG_VISITS"})
    add("C08d", obj("ORTHOPEDIC_CLEAN_V3", "PATIENTS", [("PATIENT_ID", "TEXT")], []), "ignored", {"EMPTY"})
    rows = visits(19, 300)
    add("C08e", obj("VERSIONFAC_CLEAN", "VISITS", VISIT_COLS, rows + [(r[0], r[1], r[2], None, "2026-10-01") for r in rows[:200]]),
        "proposed", why="in-place edits share a record ID; the refresh keeps the latest version")

    # ---- derived views and duplicate representations
    add("C09a", obj("KAKAMEGA_MARTS", "KAKAMEGA_VISITS_V", VISIT_COLS, visits(20, 300), "VIEW",
                    refs=["HOSPITALS.KAKAMEGA_CLEAN.EVALUATION_VISITS"]), "review",
        {"READS_ACTIVE_SOURCE:HOSPITALS.KAKAMEGA_CLEAN.EVALUATION_VISITS"})
    add("C09b", obj("ORTHOPEDIC_CLEAN", "BRZ_VISITS", [("ID", "TEXT"), ("PATIENT_ID", "TEXT")], [], "VIEW",
                    refs=["HOSPITALS.ORTHOPEDIC_CLEAN_V2.NEWPRESCRIPTION"]), "review",
        {"COVERED_BY:HOSPITALS.STAGING.STG_VISITS"})
    add("C09c", obj("SIAYA_MEDICAL_CAMP", "SIAYA_MEDICAL_CAMP_COMPLETE_DATA", [("VISIT_ID", "TEXT"), ("PATIENT_ID", "TEXT")],
                    [(f"V{i}", f"S{i}") for i in range(917)]), "ignored",
        {"PART_OF_COUNTED_FACILITY"}, why="raw twin of the counted Siaya table; not a separate source")
    add("C09g", obj("SIAYAFAC2_CLEAN", "VISITS_EXPORT_ALL", [("VISIT_ID", "TEXT"), ("PATIENT_ID", "TEXT")],
                    [(f"V{i}", f"S{i}") for i in range(917)]), "review",
        {"SAME_ROW_COUNT_AS_ACTIVE:HOSPITALS.SIAYA_MEDICAL_CAMP.SIAYA_MEDICAL_CAMP_COMPLETE_DATA_CLEANED"})
    add("C09d", obj("REPORTING", "FACT_VISITS", VISIT_COLS, visits(21, 300)), "review", {"DERIVED_LAYER"})
    add("C09e", obj("KAKAMEGA_EXPORTS", "VISITS", VISIT_COLS, visits(22, 300)), "review",
        {"SAME_FACILITY_AS_ACTIVE:HOSPITALS.KAKAMEGA_CLEAN.EVALUATION_VISITS"})
    add("C09f", obj("NEWCOUNTY2_CLEAN", "OPD_VISITS", VISIT_COLS, visits(23, 300), "VIEW",
                    refs=["HOSPITALS.ORTHOPEDIC_CLEAN_V2.NEWPRESCRIPTION"]), "review",
        {"SAME_FACILITY_AS_ACTIVE:HOSPITALS.STAGING.STG_VISITS"}, why="reads an SPH patient table")

    # ---- new datasets inside already-registered schemas
    add("C10a", obj("KAKAMEGA_CLEAN", "ANC_VISITS", VISIT_COLS, visits(24, 300), "VIEW",
                    refs=["HOSPITALS.KAKAMEGA_RAW.EVENTS_RAW"]), "review",
        {"SAME_FACILITY_AS_ACTIVE:HOSPITALS.KAKAMEGA_CLEAN.EVALUATION_VISITS"})
    add("C10b", obj("KAKAMEGA_CLEAN", "FINANCE_INVOICES", [("ID", "NUMBER"), ("PATIENT_ID", "TEXT")], [], "VIEW"),
        "ignored", {"SATELLITE"})
    add("C10c", obj("XANALIFE_CLEAN", "INPATIENT_ADMISSIONS", VISIT_COLS, visits(25, 300), "VIEW"), "review",
        {"SCHEMA_ON_HOLD"})
    add("C10d", obj("SIAYA_CLEAN", "PATIENTS", [("PATIENT_NO", "TEXT")], [("1",), ("2",)], "VIEW"), "ignored",
        {"SCHEMA_EXCLUDED"})
    add("C10e", obj("HANDWRITTEN_CLEAN", "VISITS_V2", VISIT_COLS, visits(26, 300), "VIEW"), "ignored", {"SCHEMA_EXCLUDED"})
    add("C10f", obj("ORTHOPEDIC_HOSPITAL_NAKURU_CLEAN", "VISITS", VISIT_COLS, visits(27, 300)), "proposed",
        why="a different orthopaedic hospital is not covered by the exact SPH rule")

    # ---- new datasets outside _CLEAN
    add("C11a", obj("OTHERFAC_HISTORICAL_DATA", "PATIENT_VISITS", VISIT_COLS, visits(28, 300)), "proposed")
    add("C11b", obj("STAGING", "STG_NEW_VISITS", VISIT_COLS, [], "VIEW"), "review", {"DERIVED_LAYER"})
    add("C11c", obj("EAGLEFAC_HISTORICAL_DATA", "PATIENT_VISIT", [("ID", "NUMBER"), ("MEMBERNO", "TEXT"), ("VISIT_DATE", "DATE")],
                    [r[:3] for r in visits(29, 300)]), "review", {"AMBIGUOUS_PERSON_KEY"})
    add("C11d", obj("TWINFAC_CLEAN", "VISITS", VISIT_COLS, visits(30, 300)), "review", {"SAME_FACILITY_CANDIDATES"})
    add("C11e", obj("TWINFAC_HISTORICAL_DATA", "PATIENT_VISITS", VISIT_COLS, visits(31, 300)), "review",
        {"SAME_FACILITY_CANDIDATES"})

    # ---- names that must never become proposals
    add("C12a", obj("NEWFAC_CLEAN", "ACCOUNTS", [("ID", "NUMBER"), ("PATIENT_DEPOSIT_ID", "NUMBER")], [(1, 1)]),
        "ignored", {"SATELLITE"})
    add("C12b", obj("NEWFAC_CLEAN", "CLINIC_VISITS", [("PATIENT_VISIT_NUMBER", "TEXT"), ("INPATIENT_NO", "TEXT")],
                    [("a", "b")]), "review", {"PERSON_KEY_NOT_RECOGNISED"})
    add("C12c", obj("CSV_CONNECTION_TEST_CLEAN", "VISITS", VISIT_COLS, [], "VIEW"), "ignored", {"TEST_OR_BACKUP_SCHEMA"})
    add("C12d", obj("NEWFAC_CLEAN", "VISITS__20261028", VISIT_COLS, [(1, "P", None, None, None)]), "review",
        {"LOOKS_LIKE_COPY"})
    add("C12e", obj("SMARTFAC_V3", "MPI_CLUSTERS", [("PATIENT_KEY", "TEXT")], [("x",)]), "ignored", {"IDENTITY_INDEX"})
    rng = random.Random(27)
    add("C12f", obj("NEWFAC_CLEAN", "RECEPTION_APPOINTMENTS", [("ID", "NUMBER"), ("PATIENT_ID", "TEXT"), ("APPT_DATE", "DATE")],
                    [(i, f"P{rng.randrange(400)}", d(rng, date(2026, 11, 6), 120)) for i in range(1500)]), "ignored",
        {"NOT_THE_FACILITY_VISIT_TABLE"}, why="the facility has a visits table; appointments are not attendances")
    add("C12h", obj("APPTONLY_CLEAN", "RECEPTION_APPOINTMENTS", [("ID", "NUMBER"), ("PATIENT_ID", "TEXT")], [(1, "a")]),
        "review", {"FACILITY_WITHOUT_VISIT_TABLE", "ROLE_SCHEDULING"})
    add("C12i", obj("KISUMU_V3_CLEAN", "INPATIENT_ADMISSION_TYPES", [("ID", "NUMBER"), ("NAME", "TEXT")], [], "VIEW"),
        "ignored", {"NO_PERSON_KEY"}, why="a lookup table whose name mentions admissions")
    add("C12l", obj("KISUMU_V3_CLEAN", "INPATIENT_ADMISSIONS", VISIT_COLS, visits(40, 300), "VIEW"), "ignored",
        {"ADMISSIONS_COVERED_BY_VISITS"}, why="D6: admitted patients at counted facilities all have visits")
    add("C12m", obj("REPORTING", "RPT_OPD_IPD", [("MONTH", "DATE"), ("VISITS", "NUMBER")], [], "VIEW"), "ignored",
        {"NO_PERSON_KEY"}, why="aggregate report in a derived layer")
    add("C12j", obj("KISUMU_V3_CLEAN", "RECEPTION_PATIENTS", [("ID", "NUMBER"), ("PATIENT_NO", "TEXT")], [], "VIEW"),
        "ignored", {"PART_OF_COUNTED_FACILITY"})
    add("C12k", obj("EAGLEFAC2_HISTORICAL_DATA", "PATIEN_REGISTRATION_BACKUP", [("ID", "NUMBER"), ("PATIENTID", "TEXT")],
                    [(1, "a")]), "ignored", {"COPY_OF_NON_ENCOUNTER_TABLE"})
    add("C12g", obj("MFH3_HISTORICAL_DATA", "LIMSTOCKBINCARDS", [("ID", "NUMBER"), ("REGID", "NUMBER")], [(1, 1)]),
        "review", {"FACILITY_WITHOUT_VISIT_TABLE"}, why="unrecognised single-token name, no visit table: a person looks")

    # ---- registry, lineage and profiling failures
    add("C13a", obj("KAKAMEGA_CLEAN", "EVALUATION_VISITS", VISIT_COLS, [], "VIEW",
                    refs=["HOSPITALS.KAKAMEGA_RAW.EVENTS_RAW"]), "active")
    add("C13b", obj("XANALIFE_CLEAN", "EVALUATION_VISITS", VISIT_COLS, [], "VIEW"), "ignored", {"REJECTED_IN_REGISTRY"})
    add("C14", obj("LINEFAIL_CLEAN", "VISITS", VISIT_COLS, visits(32, 300), "VIEW", lineage_error=True), "review",
        {"LINEAGE_UNKNOWN"})
    add("C15", obj("PROFFAIL_CLEAN", "VISITS", VISIT_COLS, visits(33, 300), profile_error=True), "review",
        {"PROFILE_FAILED"})

    # ---- held-out adversarial cases (A*)
    add("A01", obj("BUSIAFAC_CLEAN", "HIV_TEST_VISITS", VISIT_COLS, visits(101, 300)), "review", {"LOOKS_LIKE_COPY"},
        why="TEST in an object name: shown for review, never silently dropped")
    add("A02", obj("BUSIA2_CLEAN", "COMPILED_VISITS", VISIT_COLS, visits(102, 300)), "proposed")
    add("A03", obj("BUSIA3_CLEAN", "VISITS_COMBINED", VISIT_COLS, visits(103, 300)), "proposed",
        why="an OPD+IPD union of one facility is a single countable source")
    add("A04", obj("BUSIA4_CLEAN", "LABOUR_AND_DELIVERY", VISIT_COLS, visits(104, 300)), "review", {"ROLE_OTHER"})
    add("A05", obj("BUSIA5_CLEAN", "LAB_VISITS", VISIT_COLS, visits(105, 300)), "review", {"SERVICE_SUBSET_NAME"})
    rng = random.Random(106)
    add("A06", obj("XLFAC_CLEAN", "VISITS", VISIT_COLS,
                   [(i, str(p) if rng.random() < .5 else f"{p}.0", d(rng), None, d(rng))
                    for i, p in enumerate(p for p in range(1, 301) for _ in range(rng.randint(1, 8)))]), "review",
        {"KEY_FORMAT_VARIANTS"})
    rows = visits(107, 400)
    add("A07", obj("FB20_CLEAN", "VISITS", [("VISIT_ID", "TEXT"), ("PATIENT_ID", "TEXT"), ("VISIT_DATE", "DATE")],
                   [(f"V{x[0]}", f"V{x[0]}" if i % 10 < 2 else x[1], x[2]) for i, x in enumerate(rows)]), "review",
        {"PERSON_KEY_EQUALS_RECORD_ID"})
    rng = random.Random(108)
    rows, i = [], 0
    for p in range(300):
        for _ in range(rng.randint(1, 8)):
            m = rng.randrange(12)
            rows.append((i, f"P{p}", f"visits_2025_{m:02d}.csv", f"2025-{m + 1:02d}-15"))
            i += 1
    add("A08", obj("CSVFAC_CLEAN", "VISITS", fcols, rows), "review", {"FILE_SOURCED_ROWS"},
        why="conservative: file-sourced rows need a person to confirm the key is not per file")
    rows = visits(109, 300)
    add("A09", obj("VER2FAC_CLEAN", "VISITS", VISIT_COLS, rows + [(x[0], x[1], x[2], None, "2026-10-01") for x in rows]),
        "proposed")
    rows = visits(110, 300)
    add("A10", obj("NULLRK_CLEAN", "VISITS", VISIT_COLS[:4], [((None if i % 2 else x[0]),) + x[1:4] for i, x in enumerate(rows)]),
        "proposed")
    rows = visits(111, 300)
    add("A11", obj("VOIDFAC_CLEAN", "VISITS", [("ID", "NUMBER"), ("PATIENT_ID", "TEXT"), ("VISIT_DATE", "DATE"), ("VOIDED", "TEXT")],
                   [(x[0], x[1], x[2], "Y" if x[3] else "N") for x in rows]), "review", {"MOSTLY_DELETED"},
        why="the refresh's deletion test treats 'N' as deleted: approving as-is would count nothing")
    add("A12", obj("EMPTYV_CLEAN", "VISITS", VISIT_COLS, [], "VIEW"), "ignored", {"EMPTY"})
    add("A13", obj("PILOT2_CLEAN", "VISITS", VISIT_COLS, visits(113, 2, max_visits=2)), "review", {"SMALL_TABLE"})
    rng = random.Random(114)
    add("A14", obj("SMALLFB_CLEAN", "VISITS", [("VISIT_ID", "TEXT"), ("PATIENT_ID", "TEXT"), ("VISIT_DATE", "DATE")],
                   [(f"V{i}", f"V{i}", d(rng)) for i in range(400)]), "review", {"PERSON_KEY_EQUALS_RECORD_ID"})
    rng = random.Random(115)
    rows = visits(115, 150)
    add("A15", obj("PH8_CLEAN", "VISITS", VISIT_COLS, [(x[0], "-1" if rng.random() < .08 else x[1]) + x[2:] for x in rows]),
        "review", {"MISSING_PERSON_KEYS"})
    rows = visits(116, 200, max_visits=10) + [(100000 + k, "P_LONGSTAY", d(random.Random(k)), None, None) for k in range(400)]
    add("A16", obj("LONGSTAY_CLEAN", "IPVISIT", VISIT_COLS, rows), "review", {"DOMINANT_KEY_VALUE"},
        why="conservative: one key with 27% of rows is checked by a person")
    add("A17", obj("KAKAMEGA_LEGACY", "VISITS", VISIT_COLS, visits(117, 300)), "review",
        {"SAME_FACILITY_AS_ACTIVE:HOSPITALS.KAKAMEGA_CLEAN.EVALUATION_VISITS"})
    add("A18a", obj("TWIN2_CLEAN", "VISITS", VISIT_COLS, visits(118, 300)), "review", {"SAME_FACILITY_CANDIDATES"})
    add("A18b", obj("TWIN2_HISTORICAL_DATA", "PATIENT_VISITS", VISIT_COLS, visits(218, 300)), "review",
        {"SAME_FACILITY_CANDIDATES"})
    add("A19", obj("LOWERFAC_CLEAN", "visits", VISIT_COLS, visits(119, 300)), "review", {"NEEDS_QUOTED_IDENTIFIER"})
    add("A21", obj("NEWCOUNTY_CLEAN", "OPD_VISITS", VISIT_COLS, visits(121, 300), "VIEW",
                   refs=["HOSPITALS.NEWCOUNTY_RAW.VISITS_TBL", "HOSPITALS.REFERENCE.FACILITY_LOOKUP"]), "proposed",
        why="sharing a facility lookup table with SPH is not overlap")
    add("A22", obj("KCGH_CLEAN", "OPD_VISITS", VISIT_COLS, visits(122, 300), "VIEW", refs=["HOSPITALS.KAKAMEGA_RAW.EVENTS_RAW"]),
        "review", {"SAME_FACILITY_AS_ACTIVE:HOSPITALS.KAKAMEGA_CLEAN.EVALUATION_VISITS"})
    rows = visits(123, 300)
    add("A23", obj("HHV_CLEAN", "HOME_VISITS", [("ID", "NUMBER"), ("PATIENT_ID", "TEXT"), ("VISIT_DATE", "DATE"), ("HOUSEHOLD_NO", "TEXT")],
                   [(x[0], x[1], x[2], f"HV{x[0]}") for x in rows]), "review", {"HOUSEHOLD_COLUMN_PRESENT"},
        why="conservative, as C03c")
    add("A24", obj("FILEFAC_CLEAN", "OPD_REGISTER", [("ID", "NUMBER"), ("PATIENT_FILE_NO", "TEXT"), ("VISIT_DATE", "DATE")],
                   [x[:3] for x in visits(124, 300)]), "review", {"ROLE_REGISTER"})
    add("A25", obj("FNOFAC_CLEAN", "VISITS", [("ID", "NUMBER"), ("PATIENT_ID", "TEXT"), ("FILE_NO", "TEXT"), ("VISIT_DATE", "DATE")],
                   [(x[0], x[1], "F" + x[1], x[2]) for x in visits(125, 300)]), "proposed",
        why="FILE_NO is the patient's hospital file number, not a file signal")
    return C
