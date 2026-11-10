"""Synthetic warehouses for the warehouse model (deterministic; no real data).

replica_world()   the 13 counted sources and their neighbours, rebuilt at small scale from the documented aggregate
                  facts (scratch evidence pack, 2026-11-04/05). Column layouts, value formats and register tables
                  beyond the documented columns are replica inventions (REPLICA_INVENTIONS); since meaning now needs a
                  statement, no invention can produce an acceptance.
replica_statements(held_out)
                  what the model may know when one source is held out: no statement about the held-out object,
                  onboarding only for facilities with another counted source, facility policies only where their
                  justification survives the hold-out.
scenario_world()  synthetic adversarial and automation cases (hypothetical facilities, explicit statements).

EXPECTED_* are written from the evidence and the stated semantic rules, before running the model, as stages plus the
question codes (and a few reason fragments) that must appear. They are never generated from model output.
"""
import hashlib
import random
import sqlite3
from datetime import date, datetime, timedelta

from patient_discovery import config as C
from patient_discovery import profile_sql as P
from patient_discovery.model.concepts import Statement
from patient_discovery.model.measure import fmt, shape

NOW = datetime(2026, 11, 9)
LIVE, STOPPED = "2026-11-08 00:00:00", "2026-05-31 00:00:00"
EV_COLS = [("ID", "NUMBER"), ("PATIENT", "TEXT"), ("VISIT_DATE", "DATE"), ("DELETED_AT", "TEXT"),
           ("UPDATED_AT", "TEXT"), ("FACILITY_ID", "NUMBER"), ("VISIT_TYPE", "TEXT"), ("CREATED_BY", "TEXT")]


def obj(schema, name, columns, rows, typ="BASE TABLE", refs=None, last_altered=LIVE, lineage=True):
    if refs is None:
        refs = [f"HOSPITALS.{schema}_RAW.EVENTS_RAW"] if typ != "BASE TABLE" else []
    return dict(schema=schema, name=name, type=typ, columns=columns, rows=rows, refs=list(refs) if lineage else None,
                row_count=len(rows) if typ == "BASE TABLE" else None, last_altered=last_altered)


class Warehouse:
    """The model's warehouse interface over synthetic objects: catalogue() and query(sql)."""

    def __init__(self, objects):
        self.objs = {f"HOSPITALS.{o['schema']}.{o['name']}": o for o in objects}
        self.db = sqlite3.connect(":memory:")
        for name, n, fn in (("norm_key", 2, P.normalise_key), ("raw_key", 2, P.raw_key),
                            ("is_deleted", 1, P.is_deleted), ("shape", 1, shape), ("fmt", 1, fmt)):
            self.db.create_function(name, n, fn)
        self.n_queries = 0
        for fqn, o in self.objs.items():
            self.db.execute(f"CREATE TABLE {P.q(fqn)} ({', '.join(P.q(c) for c, _ in o['columns'])})")
            self.db.executemany(f"INSERT INTO {P.q(fqn)} VALUES ({', '.join('?' * len(o['columns']))})", o["rows"])

    def catalogue(self):
        return [dict(fqn=f, schema=o["schema"], name=o["name"], type=o["type"], columns=list(o["columns"]),
                     row_count=o["row_count"], last_altered=o["last_altered"], refs=o["refs"])
                for f, o in self.objs.items()]

    def query(self, sql):
        self.n_queries += 1
        return self.db.execute(sql).fetchall()


class FixtureAdapter:
    """The model's interface over the existing tests' SQLiteWarehouse (fixtures.py), for the cross-check."""

    def __init__(self, fx):
        self.fx, self.n_queries = fx, 0
        fx.db.create_function("shape", 1, shape)
        fx.db.create_function("fmt", 1, fmt)

    def catalogue(self):
        return [dict(fqn=f, schema=o["schema"], name=o["name"], type=o["type"], columns=list(o["columns"]),
                     row_count=o["row_count"], last_altered=o["last_altered"],
                     refs=None if o.get("lineage_error") else (o["refs"] if o["type"] != "BASE TABLE" else []))
                for f, o in self.fx.objs.items()]

    def query(self, sql):
        self.n_queries += 1
        return self.fx.db.execute(sql).fetchall()


# ------------------------------------------------------------------ statements
def source(fqn, key, record=None, deleted=None, version=None, received=None, name=None):
    return Statement(f"S:{name or fqn.split('.')[1]}", "source", fqn,
                     dict(source=name, key=key, record=record, deleted=deleted, version=version, received=received),
                     origin="registry")


def namespace(nid, meaning, facility, *members, origin="human"):
    return Statement(nid, "namespace", nid, dict(meaning=meaning, members=[list(m) for m in members],
                                                 scope=dict(facility=facility)), origin=origin)


def policies(onboard, separate=()):
    return [
        Statement("D2", "exclude_schemas", "schemas", {"schemas": sorted(C.EXCLUDED_SCHEMAS)}),
        Statement("D3", "hold_schemas", "schemas", {"schemas": sorted(C.HELD_SCHEMAS)}),
        Statement("D7", "covered_schemas", "schemas", {"schemas": dict(C.COVERED_SCHEMAS)},
                  premise={"metric": "covering_source_counted"}),
        Statement("D6", "event_class", "admission", {"action": "exclude"},
                  premise={"metric": "contribution_share", "max": 0.01}),
        Statement("P-REG", "event_class", "registration", {"action": "exclude"}),   # patients = patients with visits
        Statement("P-SUP", "event_class", "supporting", {"action": "exclude"}),
        Statement("P-SCH", "event_class", "scheduling", {"action": "exclude"}),
        Statement("P-DER", "derived_schemas", "schemas", {"schemas": sorted(C.DERIVED_SCHEMAS)}),
    ] + [Statement(f"P-SEP:{f}", "unlinked_populations", f, {"action": "count_separately"}) for f in sorted(separate)] \
      + [Statement(f"F:{f}", "facility", f, {"onboarded": True}) for f in sorted(onboard)]


# ------------------------------------------------------------------ data helpers
def geometric(rng, mean):
    k, p = 1, 1 / mean
    while rng.random() > p and k < 60:
        k += 1
    return k


def day(rng, start, days):
    return (start + timedelta(days=rng.randrange(days))).isoformat()


def ev_rows(seed, patients, mean, start, days, versions=0.0, deletes=0.0, first_id=1, extra=()):
    """Evaluation-EHR visits: unique ID per visit, PATIENT = register ID, some versions (same ID, later UPDATED_AT)."""
    rng, rows, vid = random.Random(seed), [], first_id
    for p in patients:
        for _ in range(geometric(rng, mean)):
            dt = day(rng, start, days)
            dele = "2025-06-01 10:00:00" if rng.random() < deletes else None
            rows.append((vid, p, dt, dele, f"{dt} 09:00:00", 1, rng.choice(("NEW", "REVISIT")), "clerk") + extra)
            if rng.random() < versions:
                rows.append((vid, p, dt, dele, f"{dt} 15:30:00", 1, "REVISIT", "clerk") + extra)
            vid += 1
    return rows


def simple_visits(seed, patients, mean, start=date(2025, 1, 1), days=600, first_id=1):
    rng, rows, vid = random.Random(seed), [], first_id
    for p in patients:
        for _ in range(geometric(rng, mean)):
            rows.append((vid, p, day(rng, start, days)))
            vid += 1
    return rows


def register(values, extra, prefix_no="F"):
    return [(v, f"{prefix_no}{i:06d}", "2020-01-01") for i, v in enumerate(list(values) + list(extra))]


REG_COLS = [("ID", "TEXT"), ("FILE_NO", "TEXT"), ("REGISTERED_ON", "DATE")]


# ------------------------------------------------------------------ the 13 counted sources (replicas)
SOURCES = {
    "KSH": source("HOSPITALS.KISUMU_V3_CLEAN.EVALUATION_VISITS", "PATIENT", "ID", "DELETED_AT", "UPDATED_AT", name="KSH"),
    "Tenri": source("HOSPITALS.TENRI.EVALUATION_VISITS", "PATIENT", "ID", "DELETED_AT", "UPDATED_AT", "_INGESTED_AT",
                    name="Tenri"),
    "Lodwar": source("HOSPITALS.LODWAR_CLEAN.EVALUATION_VISITS", "PATIENT", "ID", "DELETED_AT", "UPDATED_AT",
                     name="Lodwar"),
    "Kakamega": source("HOSPITALS.KAKAMEGA_CLEAN.EVALUATION_VISITS", "PATIENT", "ID", "DELETED_AT", "UPDATED_AT",
                       name="Kakamega"),
    "Kakamega HIV": source("HOSPITALS.KAKAMEGA_CLEAN.HIV_HTS_DATA", "PATIENT_ID", name="Kakamega HIV"),
    "SPH": source("HOSPITALS.STAGING.STG_VISITS", "COMPOSITE_PATIENT_ID", name="SPH"),
    "Nandi": source("HOSPITALS.NANDI_CLEAN.NANDI_OUTPATIENT_WIDE", "UNIQUE_PATIENT_ID", name="Nandi"),
    "Migori HIV": source("HOSPITALS.MIGORI_CLEAN.AROMBE_HIV_RAW", "PATIENT_ID_HASH", name="Migori HIV"),
    "MF historical": source("HOSPITALS.MOTHERFRANSICA_HISTORICAL_DATA.LIMPATIENTVISITDETAILS", "REGID",
                            name="MF historical"),
    "MF camp": source("HOSPITALS.MOTHERFRANSICA_MEDICAL_CAMP_CLEAN.PATIENT_DATA", "PATIENT_ID", name="MF camp"),
    "Silverwood": source("HOSPITALS.SILVERWOOD_CLEAN.VISITS", "PATIENT", name="Silverwood"),
    "Siaya camp": source("HOSPITALS.SIAYA_MEDICAL_CAMP.SIAYA_MEDICAL_CAMP_COMPLETE_DATA_CLEANED", "PATIENT_ID",
                         name="Siaya camp"),
    "Wema": source("HOSPITALS.WEMA_MEDICAL_CLEAN.VISITS", "PATIENT_ID", "ID", name="Wema"),
}
REJECTED = [Statement(f"R:{f.split('.', 1)[1]}", "rejected", f, {}, origin="registry") for f in (
    "HOSPITALS.KISUMU_CLEAN.VISITS", "HOSPITALS.ORTHOPEDIC_CLEAN.ORTHO_ORDERS", "HOSPITALS.ORTHOPEDIC_CLEAN_V2.ORTHO_VISITS")]

# Documented meaning facts (evidence pack). Only two exist: a person approved Nandi's UNIQUE_PATIENT_ID as a resolved
# person key; the audit found the Mother Francisca camp PATIENT_ID is assigned per file. Nothing else documented says
# what a counted source's key denotes ("PATIENT = register ID" at Tenri is a value relationship, not a meaning).
NAMESPACES = {
    "Nandi": namespace("N:NANDI-PERSON", "person", "NANDI",
                       ("HOSPITALS.NANDI_CLEAN.NANDI_OUTPATIENT_WIDE", "UNIQUE_PATIENT_ID")),
    "MF camp": namespace("N:MFCAMP-FILE", "file", "MOTHERFRANSICA_MEDICAL_CAMP",
                         ("HOSPITALS.MOTHERFRANSICA_MEDICAL_CAMP_CLEAN.PATIENT_DATA", "PATIENT_ID"), origin="audit"),
}
# A Kakamega decision (both Kakamega populations counted separately): applies to Kakamega only, and only while both
# Kakamega sources are known.
SEPARATE = {"KAKAMEGA": ("Kakamega", "Kakamega HIV")}
REPLICA_INVENTIONS = [
    "the 8-column EVALUATION_VISITS layout (only PATIENT, ID, DELETED_AT, UPDATED_AT, _INGESTED_AT are documented)",
    "register tables and their columns at KSH, Tenri, Lodwar (only their sizes are documented)",
    "value formats (KSH-, TEN, LDW-, KAK-, HTS prefixes; hashes; numbers)",
    "visit distributions (geometric) and dates",
    "about 30 objects where the real schemas hold about 1,600",
]

# Pre-registered from the evidence and the semantic rules, before running: (stage, codes that must appear, codes that
# must not, reason fragments that must appear). No held-out source has a documented person meaning once its own
# statements are removed, so every one must be a question.
EXPECTED_REPLAY = {
    "KSH": ("question", {"MEANING_UNRESOLVED", "SCOPE_UNKNOWN"}, set(), []),
    "Tenri": ("question", {"MEANING_UNRESOLVED", "SCOPE_UNKNOWN"}, set(), []),
    "Lodwar": ("question", {"MEANING_UNRESOLVED", "SCOPE_UNKNOWN"}, set(), []),
    "Kakamega": ("question", {"MEANING_UNRESOLVED"}, {"SCOPE_UNKNOWN"}, []),
    "Kakamega HIV": ("question", {"MEANING_UNRESOLVED", "KIND_UNKNOWN", "GRAIN"}, {"SCOPE_UNKNOWN"},
                     ["about one record per value"]),
    "SPH": ("question", {"NO_USABLE_IDENTIFIER", "DERIVED_VIEW", "DERIVED_LAYER", "SCOPE_UNKNOWN"}, set(),
            ["equals the record identifier"]),
    "Nandi": ("question", {"MEANING_UNRESOLVED", "GRAIN", "SCOPE_UNKNOWN"}, set(), ["exact duplicate rows"]),
    "Migori HIV": ("question", {"MEANING_UNRESOLVED", "KIND_UNKNOWN", "GRAIN", "SCOPE_UNKNOWN"}, set(),
                   ["about one record per value"]),
    "MF historical": ("question", {"MEANING_UNRESOLVED", "SCOPE_UNKNOWN"}, set(), []),
    "MF camp": ("question", {"MEANING_UNRESOLVED", "KIND_UNKNOWN", "GRAIN", "SCOPE_UNKNOWN"}, set(),
                ["rows come from documents or files"]),
    "Silverwood": ("question", {"MEANING_UNRESOLVED", "GRAIN", "SCOPE_UNKNOWN"}, set(), ["only 25 rows"]),
    "Siaya camp": ("question", {"MEANING_UNRESOLVED", "KIND_UNKNOWN", "GRAIN", "SCOPE_UNKNOWN"}, set(),
                   ["record identifiers repeat with no version column"]),
    "Wema": ("question", {"MEANING_UNRESOLVED", "SCOPE_UNKNOWN"}, set(), []),
}
# With every statement (nothing held out): what each counted source's key is known to mean.
EXPECTED_COUNTED_MEANING = {"Nandi": "person", "MF camp": "file"}       # every other counted source: never established


def replica_world():
    o = []
    # KSH: views over the raw store; V3 VISITS returns every visit twice; register ~3x the visit patients
    ksh = [f"KSH-{i:06d}" for i in range(600)]
    ev = ev_rows(1, ksh, 3.0, date(2024, 5, 31), 520)        # rows read = patient records: no superseded versions
    o.append(obj("KISUMU_V3_CLEAN", "EVALUATION_VISITS", EV_COLS, ev, "VIEW"))
    once = {r[0]: r for r in ev}
    o.append(obj("KISUMU_V3_CLEAN", "VISITS", [("ID", "NUMBER"), ("PATIENT", "TEXT"), ("VISIT_DATE", "DATE")],
                 [r[:3] for r in once.values()] * 2, "VIEW"))
    o.append(obj("KISUMU_V3_CLEAN", "PATIENTS", REG_COLS, register(ksh, [f"KSH-{i:06d}" for i in range(600, 1800)]),
                 "VIEW"))
    o.append(obj("KISUMU_CLEAN", "VISITS", [("ID", "NUMBER"), ("PATIENT", "TEXT"), ("VISIT_DATE", "DATE")],
                 [r[:3] for r in ev[:300]], "VIEW"))
    # Tenri: base table, data 2018-03..2022-07, 0.5% soft deletes, ingestion column; register
    ten = [f"TEN{i:07d}" for i in range(800)]
    o.append(obj("TENRI", "EVALUATION_VISITS", EV_COLS + [("_INGESTED_AT", "TEXT")],
                 ev_rows(2, ten, 2.8, date(2018, 3, 1), 1600, deletes=0.005, extra=("2022-08-01 00:00:00",)),
                 last_altered="2022-08-01 00:00:00"))
    o.append(obj("TENRI", "PATIENTS", REG_COLS, register(ten, [f"TEN{i:07d}" for i in range(800, 1200)]),
                 last_altered="2022-08-01 00:00:00"))
    # Lodwar: 1.5 visits per patient; the warehouse register holds 200 rows, 172 of them visit patients
    lod = [f"LDW-{i:06d}" for i in range(700)]
    o.append(obj("LODWAR_CLEAN", "EVALUATION_VISITS", EV_COLS, ev_rows(3, lod, 1.5, date(2024, 1, 1), 365), "VIEW"))
    o.append(obj("LODWAR_CLEAN", "PATIENTS", REG_COLS, register(lod[:17], ["LDW-X00001", "LDW-X00002", "LDW-X00003"]),
                 "VIEW"))
    # Kakamega: outpatient visits (no register documented) and HIV testing data (separate ID system, ~95% once)
    kak = [f"KAK-{i:06d}" for i in range(700)]
    o.append(obj("KAKAMEGA_CLEAN", "EVALUATION_VISITS", EV_COLS,
                 ev_rows(4, kak, 1.64, date(2024, 1, 1), 365, deletes=0.0008), "VIEW"))
    rng, hts = random.Random(5), []
    for i in range(650):
        for _ in range(2 if rng.random() < 0.055 else 1):
            hts.append((f"HTS{i:05d}", day(rng, date(2024, 1, 1), 365), rng.choice(("NEG", "POS")),
                        rng.choice(("OPD", "PMTCT", "VCT"))))
    o.append(obj("KAKAMEGA_CLEAN", "HIV_HTS_DATA", [("PATIENT_ID", "TEXT"), ("TEST_DATE", "DATE"),
                                                    ("HTS_RESULT", "TEXT"), ("ENTRY_POINT", "TEXT")], hts, "VIEW"))
    # Migori: ART line list, one row per hashed patient
    rng = random.Random(6)
    o.append(obj("MIGORI_CLEAN", "AROMBE_HIV_RAW", [("PATIENT_ID_HASH", "TEXT"), ("ART_START_DATE", "DATE"),
                                                    ("SEX", "TEXT"), ("REGIMEN", "TEXT")],
                 [(hashlib.sha256(f"m{i}".encode()).hexdigest(), day(rng, date(2015, 1, 1), 3000),
                   rng.choice("MF"), rng.choice(("TLD", "AZT"))) for i in range(1287)]))
    # Mother Francisca historical: appointments (APTID unique), REGID in LIMPATIENTS, admissions (0.45% new), stock
    rng, mf = random.Random(7), list(range(1, 1101))
    apt, aid = [], 1
    for p in mf:
        for _ in range(geometric(rng, 1.68)):
            apt.append((aid, p, 1600000000000000 + aid, rng.randint(1, 15)))
            aid += 1
    o.append(obj("MOTHERFRANSICA_HISTORICAL_DATA", "LIMPATIENTVISITDETAILS",
                 [("APTID", "NUMBER"), ("REGID", "NUMBER"), ("APTDATE", "NUMBER"), ("DOCTORID", "NUMBER")], apt))
    o.append(obj("MOTHERFRANSICA_HISTORICAL_DATA", "LIMPATIENTS",
                 [("REGID", "NUMBER"), ("OPNO", "TEXT"), ("REGDATE", "DATE")],
                 [(r, f"OP{r:06d}", "2019-01-01") for r in range(1, 1301)]))
    adm = [(i, p, day(rng, date(2020, 1, 1), 900)) for i, p in enumerate(rng.sample(mf, 195) + [1101, 1102, 1103, 1104, 1105])]
    o.append(obj("MOTHERFRANSICA_HISTORICAL_DATA", "LIMADMISSIONS",
                 [("ADMISSIONID", "NUMBER"), ("REGID", "NUMBER"), ("ADMDATE", "DATE")], adm))
    o.append(obj("MOTHERFRANSICA_HISTORICAL_DATA", "LIMSTOCKBINCARDS",
                 [("BINCARDNO", "NUMBER"), ("REGID", "NUMBER"), ("ITEMCODE", "TEXT"), ("QTY", "NUMBER")],
                 [(i, 900000 + rng.randrange(300), f"IT{rng.randrange(80)}", rng.randrange(1, 50)) for i in range(500)]))
    # Mother Francisca camp: one row per extracted document; PATIENT_ID per file; a raw twin
    rng, docs, d = random.Random(8), [], 0
    for f in range(303):
        for page in range(1, rng.choice((1, 1, 2, 2, 3)) + 1):
            docs.append((f"DOC{d:05d}", f"CAMP{f:04d}", f"batch{f // 50}.pdf", page, "2025-03-15"))
            d += 1
    camp_cols = [("DOCUMENT_ID", "TEXT"), ("PATIENT_ID", "TEXT"), ("FILENAME", "TEXT"), ("PAGE", "NUMBER"),
                 ("VISIT_DATE", "DATE")]
    o.append(obj("MOTHERFRANSICA_MEDICAL_CAMP_CLEAN", "PATIENT_DATA", camp_cols, docs, "VIEW",
                 refs=["HOSPITALS.MOTHERFRANSICA_MEDICAL_CAMP_RAW.EVENTS_RAW"]))
    o.append(obj("MOTHERFRANSICA_MEDICAL_CAMP", "PATIENT_DATA", camp_cols, list(docs)))
    # Nandi: resolved person key over PNOs (one ID up to several PNOs), no record ID, ~23% exact duplicate rows
    rng, nandi = random.Random(9), []
    for i in range(1100):
        for j in range(rng.choice((2, 3)) if rng.random() < 0.05 else 1):
            for _ in range(geometric(rng, 1.3)):
                nandi.append((f"U{i:07d}", f"PNO{i:06d}{j}", 2024, rng.choice(("MALARIA", "URTI", "HTN"))))
    nandi += [nandi[rng.randrange(len(nandi))] for _ in range(int(len(nandi) * 0.3))]
    o.append(obj("NANDI_CLEAN", "NANDI_OUTPATIENT_WIDE", [("UNIQUE_PATIENT_ID", "TEXT"), ("PNO", "TEXT"),
                                                          ("VISIT_YEAR", "NUMBER"), ("DIAGNOSIS", "TEXT")], nandi, "VIEW"))
    # Siaya camp: PATIENT_ID one-to-one with VISIT_ID, 2 duplicated rows; a raw twin; a 246-row register that doesn't match
    rng = random.Random(10)
    siaya = [(f"V{i:04d}", f"S{i:04d}", day(rng, date(2025, 2, 1), 3)) for i in range(915)]
    siaya += siaya[:2]
    scols = [("VISIT_ID", "TEXT"), ("PATIENT_ID", "TEXT"), ("VISIT_DATE", "DATE")]
    o.append(obj("SIAYA_MEDICAL_CAMP", "SIAYA_MEDICAL_CAMP_COMPLETE_DATA_CLEANED", scols, siaya))
    o.append(obj("SIAYA_MEDICAL_CAMP", "SIAYA_MEDICAL_CAMP_COMPLETE_DATA", scols, list(siaya)))
    o.append(obj("SIAYA_MEDICAL_CAMP", "PATIENTS", [("PATIENT_ID", "TEXT"), ("REGISTERED_ON", "DATE")],
                 [(f"SP{i:05d}", "2025-01-30") for i in range(246)]))
    # Silverwood: 25 rows / 14 patients; the test copy sits in an excluded schema
    sw = simple_visits(11, [f"SW{i:03d}" for i in range(14)], 1.8)[:25]
    while len(sw) < 25:
        sw.append((100 + len(sw), "SW000", "2026-10-20"))
    vcols = [("ID", "NUMBER"), ("PATIENT", "TEXT"), ("VISIT_DATE", "DATE")]
    o.append(obj("SILVERWOOD_CLEAN", "VISITS", vcols, sw, "VIEW"))
    o.append(obj("SILVERWOOD_TEST_CLEAN", "VISITS", vcols, list(sw), "VIEW"))
    # SPH: staging view over the V1/V2 feeds (covered, rejected); visit ID used as patient ID when missing
    rng, stg = random.Random(12), []
    for i in range(2000):
        vid = f"SV{i:06d}"
        if rng.random() < 0.08:
            pid = comp = vid
        else:
            p = rng.randrange(700)
            pid, comp = f"P{p:05d}", f"C{p if rng.random() > 0.04 else p + 1:05d}"
        stg.append((vid, pid, comp, rng.choice(("V1", "V2")), day(rng, date(2023, 1, 1), 900)))
    o.append(obj("STAGING", "STG_VISITS", [("VISIT_ID", "TEXT"), ("PATIENT_ID", "TEXT"),
                                           ("COMPOSITE_PATIENT_ID", "TEXT"), ("SOURCE_SYSTEM", "TEXT"),
                                           ("VISIT_DATE", "DATE")], stg, "VIEW",
                 refs=["HOSPITALS.ORTHOPEDIC_CLEAN.ORTHO_ORDERS", "HOSPITALS.ORTHOPEDIC_CLEAN_V2.ORTHO_VISITS"]))
    o.append(obj("ORTHOPEDIC_CLEAN", "ORTHO_ORDERS", [("ORDER_ID", "TEXT"), ("PATIENT_ID", "TEXT"), ("ORDER_DATE", "DATE")],
                 [(f"O{i}", r[1], r[4]) for i, r in enumerate(stg[:900])]))
    o.append(obj("ORTHOPEDIC_CLEAN_V2", "ORTHO_VISITS", [("PRESCID", "TEXT"), ("PATIENT_ID", "TEXT"), ("VISIT_DATE", "DATE")],
                 [(f"R{i}", r[1], r[4]) for i, r in enumerate(stg[900:])]))
    # Wema: unique visit ID, ~2.9 visits per patient; no register documented
    o.append(obj("WEMA_MEDICAL_CLEAN", "VISITS", [("ID", "NUMBER"), ("PATIENT_ID", "TEXT"), ("VISIT_DATE", "DATE")],
                 simple_visits(13, [f"WM{i:05d}" for i in range(600)], 2.86), "VIEW"))
    return o


def replica_statements(held_out=None):
    """Statements the model may use when `held_out` (a SOURCES name) is the source being rediscovered."""
    from patient_discovery.classify import family
    held = SOURCES.get(held_out)
    held_fqn = held.subject if held else None
    sources = [s for n, s in SOURCES.items() if n != held_out]
    onboard = {family(s.subject.split(".")[1]) for s in sources}          # known only from another counted source
    spaces = [s for n, s in NAMESPACES.items() if n != held_out
              and not any(m[0] == held_fqn for m in s.assertion["members"])]
    separate = {f for f, names in SEPARATE.items() if held_out not in names}
    return sources + REJECTED + spaces + policies(onboard, separate)


# ------------------------------------------------------------------ difficult scenarios
Q = "question"
# fqn -> (stage, codes that must appear, codes that must not, reason fragments that must appear)
EXPECTED_SCENARIOS = {
    # names, registers, layouts and containment never give meaning: all of these ask
    "HOSPITALS." + "EAGLE_NURSING_HOME_HISTORICAL_DATA.PATIENT_VISIT": (Q, {"MEANING_UNRESOLVED", "SCOPE_UNKNOWN"}, set(), []),
    "HOSPITALS." + "EAGLE_NURSING_HOME_HISTORICAL_DATA.PATIENT_IPVISIT": ("not_a_source", set(), set(), ["premise not checkable"]),
    "HOSPITALS." + "HILLSIDE_CLEAN.VISITS": (Q, {"MEANING_UNRESOLVED"}, set(), ["covers several"]),
    "HOSPITALS." + "LAKEVIEW_CLEAN.VISITS": (Q, {"MEANING_UNRESOLVED"}, set(), []),
    "HOSPITALS." + "VALLEY_CLEAN.VISITS": (Q, {"MEANING_UNRESOLVED"}, set(), []),
    "HOSPITALS." + "PINE_CLEAN.VISITS": (Q, {"MEANING_UNRESOLVED"}, set(), ["covers several"]),
    "HOSPITALS." + "CEDAR_MEDICAL_CAMP.CAMP_VISITS": (Q, {"MEANING_UNRESOLVED"}, set(), ["about one record per value"]),
    "HOSPITALS." + "ASPEN_CLEAN.VISITS": (Q, {"MEANING_UNRESOLVED"}, set(), ["a value relationship, not a meaning"]),
    "HOSPITALS." + "ELM_CLEAN.VISITS": (Q, {"NO_CANDIDATE"}, set(), []),
    "HOSPITALS." + "BAOBAB_CLEAN.VISITS": (Q, {"MEANING_UNRESOLVED"}, set(), ["one-to-one"]),
    "HOSPITALS." + "ACACIA_CLEAN.VISITS": (Q, {"MEANING_UNRESOLVED"}, set(), ["crosses"]),
    "HOSPITALS." + "WILLOW_CLEAN.VISITS": (Q, {"MEANING_UNRESOLVED", "TRUNCATED"}, set(), []),
    "HOSPITALS." + "CYPRESS_CLEAN.VISITS": (Q, {"MEANING_UNRESOLVED", "SCOPE_UNKNOWN"}, set(), []),
    "HOSPITALS." + "FIG_HISTORICAL_DATA.VISITS": (Q, {"NO_USABLE_IDENTIFIER"}, set(), ["presence follows IS_WALKIN"]),
    "HOSPITALS." + "OUTREACH_CLEAN.VISITS": (Q, {"MEANING_UNRESOLVED"}, set(), ["PATIENT_NO ["]),
    "HOSPITALS." + "MCH_CLEAN.VISITS": (Q, {"MEANING_UNRESOLVED"}, set(), ["PATIENT_ID [listed; contradicts: present on"]),
    "HOSPITALS." + "INSURED_CLEAN.VISITS": (Q, {"MEANING_UNRESOLVED"}, set(), []),
    "HOSPITALS." + "CHW_CLEAN.VISITS": (Q, {"MEANING_UNRESOLVED"}, set(), []),
    "HOSPITALS." + "ELMWOOD_CLEAN.VISITS": (Q, {"MEANING_UNRESOLVED"}, set(), []),
    # value relations with a confirmed namespace are reported, never interpreted
    "HOSPITALS." + "JUNIPER_CLEAN.OPD_VISITS": (Q, {"MEANING_UNRESOLVED", "VALUE_RELATION"}, set(), []),
    "HOSPITALS." + "JUNIPER_CLEAN.HOME_VISITS": (Q, {"MEANING_UNRESOLVED", "UNLINKED"}, set(), []),
    "HOSPITALS." + "JUNIPER_CLEAN.ART_COHORT": (Q, {"MEANING_UNRESOLVED", "KIND_UNKNOWN", "VALUE_RELATION"}, set(), []),
    "HOSPITALS." + "OAKRIDGE_CLEAN.DENTAL_VISITS": (Q, {"MEANING_UNRESOLVED", "VALUE_RELATION"}, {"DOUBLE_COUNT"}, []),
    "HOSPITALS." + "OAKRIDGE_CLEAN.HIV_ART_DATA": (Q, {"MEANING_UNRESOLVED", "KIND_UNKNOWN", "UNLINKED"}, set(), []),
    # unrecognised table, generic-named candidate, no link to patients: set aside (meaning still unresolved)
    "HOSPITALS." + "OAKRIDGE_CLEAN.HH_EXTRACT": ("not_a_source", set(), set(), ["generic names"]),
    "HOSPITALS." + "OAKRIDGE_CLEAN.STAFF_ROSTER": ("not_a_source", set(), set(), ["generic names"]),
    # ... but generic-named values that sit inside a counted patient identifier are still asked about
    "HOSPITALS." + "OAKRIDGE_CLEAN.SYNC_BATCH": (Q, {"MEANING_UNRESOLVED", "KIND_UNKNOWN", "VALUE_RELATION"}, set(), []),
    # scope: a confirmed namespace does not reach another facility's objects
    "HOSPITALS." + "REGION_STAGING.ALL_VISITS": (Q, {"DERIVED_VIEW", "MEANING_UNRESOLVED", "SCOPE_UNKNOWN"}, set(), []),
    # derivation transfers meaning; the derived objects are recognised without a question
    "HOSPITALS." + "OAKRIDGE_CLEAN.VW_VISITS": ("not_a_source", set(), set(), ["view of counted"]),
    "HOSPITALS." + "OAKRIDGE_CLEAN.VW_2025": ("not_a_source", set(), set(), ["view of counted"]),
    "HOSPITALS." + "OAKRIDGE_EXPORT.VISITS": ("not_a_source", set(), set(), ["same records as counted"]),
    "HOSPITALS." + "OAKRIDGE_CLEAN.VW_SECURE": ("not_a_source", set(), set(), ["same records as counted"]),
    "HOSPITALS." + "RIVERSIDE_V4_CLEAN.VISITS": ("replacement", set(), set(), ["reconciled"]),
    "HOSPITALS." + "WREN_CLEAN.VISITS": ("auto", set(), set(), []),
    # derivation that is not clean: transfer refused, or the change needs sign-off
    "HOSPITALS." + "SPRUCE_V2_CLEAN.VISITS": (Q, {"PARTIAL_RECORDS", "MEANING_UNRESOLVED"}, set(), []),
    "HOSPITALS." + "KESTREL_V2_CLEAN.VISITS": (Q, {"MEANING_UNRESOLVED"}, set(), ["formats not seen"]),
    "HOSPITALS." + "MERIDIAN_V2_CLEAN.VISITS": (Q, {"REKEYED", "HISTORY_CHANGE"}, {"MEANING_UNRESOLVED"}, []),
    "HOSPITALS." + "PARALLEL_V2_CLEAN.VISITS": (Q, {"SUPERSET_LIVE"}, {"MEANING_UNRESOLVED"}, []),
    # confirmed meaning does not excuse bad behaviour or grain
    "HOSPITALS." + "TESTACC_CLEAN.VISITS": (Q, {"BEHAVIOUR"}, {"MEANING_UNRESOLVED"}, ["one value dominates"]),
    "HOSPITALS." + "SWITCH_CLEAN.VISITS": (Q, {"BEHAVIOUR", "GRAIN"}, {"MEANING_UNRESOLVED"}, ["several values per record"]),
    "HOSPITALS." + "DELTA_CLEAN.VISITS": (Q, {"GRAIN"}, {"MEANING_UNRESOLVED"}, ["most rows deleted"]),
    "HOSPITALS." + "FUTURE_CLEAN.VISITS": (Q, {"GRAIN"}, {"MEANING_UNRESOLVED"}, ["only future dates"]),
    # admissions: D6's premise is checkable only inside one confirmed namespace
    "HOSPITALS." + "MAPLE_HISTORICAL_DATA.ADMISSIONS": ("not_a_source", set(), set(), ["premise holds"]),
    "HOSPITALS." + "BIRCH_HISTORICAL_DATA.ADMISSIONS": (Q, {"POLICY_PREMISE"}, set(), []),
}
# fqn -> (column, meaning, namespace, provenance fragment): meanings that must be established, and columns that must
# stay unresolved (meaning None)
EXPECTED_MEANING = {
    "HOSPITALS." + "OAKRIDGE_CLEAN.VW_2025": ("PATIENT_ID", "person", "N:OAKRIDGE", "aligned with OAKRIDGE_CLEAN.VISITS"),
    "HOSPITALS." + "OAKRIDGE_EXPORT.VISITS": ("PATIENT_ID", "person", "N:OAKRIDGE", "aligned with OAKRIDGE_CLEAN.VISITS"),
    "HOSPITALS." + "RIVERSIDE_V4_CLEAN.VISITS": ("PATIENT_ID", "person", "N:RIVERSIDE", "aligned with RIVERSIDE_V3_CLEAN.VISITS"),
    "HOSPITALS." + "MERIDIAN_V2_CLEAN.VISITS": ("PATIENT_ID", "person", "N:MERIDIAN", "aligned with MERIDIAN_V1_CLEAN.VISITS"),
    "HOSPITALS." + "PARALLEL_V2_CLEAN.VISITS": ("PATIENT_ID", "person", "N:PARALLEL", "aligned with PARALLEL_V1_CLEAN.VISITS"),
    "HOSPITALS." + "WREN_CLEAN.VISITS": ("PATIENT_ID", "person", "N:WREN", "N:WREN: WREN_CLEAN.VISITS.PATIENT_ID"),
    "HOSPITALS." + "OAKRIDGE_CLEAN.HH_EXTRACT": ("HH_NO", None, None, None),
    "HOSPITALS." + "REGION_STAGING.ALL_VISITS": ("PATIENT_ID", None, None, None),
    "HOSPITALS." + "KESTREL_V2_CLEAN.VISITS": ("PATIENT_ID", None, None, None),
    "HOSPITALS." + "JUNIPER_CLEAN.ART_COHORT": ("CLIENT_NO", None, None, None),
    "HOSPITALS." + "JUNIPER_CLEAN.HOME_VISITS": ("HOUSEHOLD_NO", None, None, None),
    "HOSPITALS." + "ASPEN_CLEAN.VISITS": ("FOLIO_NO", None, None, None),
    "HOSPITALS." + "LAKEVIEW_CLEAN.VISITS": ("PATIENT_ID", None, None, None),
    "HOSPITALS." + "OUTREACH_CLEAN.VISITS": ("HH_NO", None, None, None),
}


def _visits_with(seed, patients, mean, cols_fn, start=date(2025, 1, 1), days=600):
    rng, rows, vid = random.Random(seed), [], 1
    for p in patients:
        for _ in range(geometric(rng, mean)):
            rows.append(cols_fn(rng, f"VS{seed:02d}{vid:06d}", p, day(rng, start, days)))
            vid += 1
    return rows


def scenario_world():
    o, st = [], []
    v4 = lambda rng, vid, p, dt: (vid, p, dt)
    vcols = lambda key: [("VISIT_ID", "TEXT"), (key, "TEXT"), ("VISIT_DATE", "DATE")]
    # counted context
    oak = [f"OKR-{i:06d}" for i in range(600)]
    oak_rows = _visits_with(20, oak, 3.0, v4)
    o.append(obj("OAKRIDGE_CLEAN", "VISITS", vcols("PATIENT_ID"), oak_rows))
    st.append(source("HOSPITALS.OAKRIDGE_CLEAN.VISITS", "PATIENT_ID", "VISIT_ID", name="Oakridge"))
    # Eagle: PATIENTCODE = MRDNO (two spellings of one number); MEMBERNO only for insured patients, shared by dependants
    rng, eagle, vid, ip = random.Random(21), [], 1, []
    members = {}
    for p in range(700):
        insured = rng.random() < 0.10
        if insured:
            members[p] = members.get(p - 1) if (p - 1 in members and rng.random() < 0.5) else f"M{p:05d}"
        for _ in range(geometric(rng, 3.0)):
            dt = day(rng, date(2019, 1, 1), 1500)
            eagle.append((f"EV{vid:07d}", f"PC{p:06d}", f"MRD{p:06d}", members.get(p) if insured else None,
                          "INSURANCE" if insured else "CASH", rng.choice(("AAR", "JUBILEE")) if insured else None,
                          f"VC{vid:07d}", dt, rng.choice(("OPD", "MCH", "DENTAL")),
                          f"MCC{vid}" if rng.random() < 0.02 else None))
            vid += 1
        if rng.random() < 0.08:
            ip.append((f"IP{p:05d}", f"PC{p:06d}", f"MRD{p:06d}", members.get(p), f"AD{p}", day(rng, date(2019, 1, 1), 1500)))
    ip.append(("IP99999", "PC999999", "MRD999999", None, "AD99999", "2020-01-01"))
    ecols = [("VISITID", "TEXT"), ("PATIENTCODE", "TEXT"), ("MRDNO", "TEXT"), ("MEMBERNO", "TEXT"),
             ("PAYMENTTYPENAME", "TEXT"), ("INSURANCEGROUP", "TEXT"), ("VISITCODE", "TEXT"),
             ("CONSULTATIONDATE", "DATE"), ("DEPARTMENT", "TEXT"), ("MCCNO", "TEXT")]
    o.append(obj("EAGLE_NURSING_HOME_HISTORICAL_DATA", "PATIENT_VISIT", ecols, eagle, last_altered=STOPPED))
    o.append(obj("EAGLE_NURSING_HOME_HISTORICAL_DATA", "PATIENT_IPVISIT",
                 [("VISITID", "TEXT"), ("PATIENTCODE", "TEXT"), ("MRDNO", "TEXT"), ("MEMBERNO", "TEXT"),
                  ("ADMITID", "TEXT"), ("ADMISSIONDATE", "DATE")], ip, last_altered=STOPPED))
    # households with a finer person number
    rng, hh, vid = random.Random(22), [], 1
    for h in range(300):
        for m in range(rng.randint(1, 5)):
            for _ in range(geometric(rng, 2.5)):
                hh.append((f"HS{vid:06d}", f"HH{h:05d}", f"PN{h:05d}{m}", day(rng, date(2025, 1, 1), 500)))
                vid += 1
    o.append(obj("HILLSIDE_CLEAN", "VISITS", [("VISIT_ID", "TEXT"), ("HH_NO", "TEXT"), ("PERSON_NO", "TEXT"),
                                              ("VISIT_DATE", "DATE")], hh))
    # household numbers in the patient column, no finer column; LAKEVIEW also has a household 'register'
    house = lambda seed: _visits_with(seed, [f"H{h:05d}" for h in range(300)], 6.0, v4)
    o.append(obj("LAKEVIEW_CLEAN", "VISITS", vcols("PATIENT_ID"), house(23)))
    o.append(obj("LAKEVIEW_CLEAN", "PATIENTS", REG_COLS, register([f"H{h:05d}" for h in range(300)], [])))
    o.append(obj("VALLEY_CLEAN", "VISITS", vcols("PATIENT_ID"), house(24)))
    # insurance member number (always filled) shared by 1-4 patients
    rng, pine, vid = random.Random(25), [], 1
    for m in range(250):
        for k in range(rng.randint(1, 4)):
            for _ in range(geometric(rng, 2.5)):
                pine.append((f"PV{vid:06d}", f"MEM{m:05d}", f"PAT{m:05d}{k}", day(rng, date(2025, 1, 1), 500)))
                vid += 1
    o.append(obj("PINE_CLEAN", "VISITS", [("VISIT_ID", "TEXT"), ("MEMBER_NO", "TEXT"), ("PATIENT_NO", "TEXT"),
                                          ("VISIT_DATE", "DATE")], pine))
    # single-visit camp
    rng = random.Random(26)
    o.append(obj("CEDAR_MEDICAL_CAMP", "CAMP_VISITS", vcols("PATIENT_ID"),
                 [(f"CV{i:05d}", f"CP{min(i, 789):05d}", day(rng, date(2025, 8, 1), 3)) for i in range(800)]))
    # unfamiliar identifier names
    asp = [f"FL/{i:05d}" for i in range(500)]
    o.append(obj("ASPEN_CLEAN", "VISITS", vcols("FOLIO_NO"), _visits_with(27, asp, 3.0, v4)))
    o.append(obj("ASPEN_CLEAN", "PATIENTS", [("FOLIO_NO", "TEXT"), ("REGISTERED_ON", "DATE")],
                 [(f, "2020-01-01") for f in asp + [f"FL/{i:05d}" for i in range(500, 700)]]))
    o.append(obj("ELM_CLEAN", "VISITS", vcols("CARDREF"), _visits_with(28, [f"EL{i:05d}" for i in range(500)], 3.0, v4)))
    # several plausible keys: equivalent / crossing / too many
    bao = list(range(500))
    o.append(obj("BAOBAB_CLEAN", "VISITS", [("VISIT_ID", "TEXT"), ("PATIENT_NO", "TEXT"), ("OP_NO", "TEXT"),
                                            ("VISIT_DATE", "DATE")],
                 _visits_with(29, bao, 3.0, lambda rng, vid, p, dt: (vid, f"BP{p:05d}", f"OP/{p:05d}", dt))))
    o.append(obj("BAOBAB_CLEAN", "PATIENTS", REG_COLS, register([f"BP{p:05d}" for p in range(600)], [])))
    o.append(obj("ACACIA_CLEAN", "VISITS", [("VISIT_ID", "TEXT"), ("PATIENT_NO", "TEXT"), ("CLIENT_ID", "TEXT"),
                                            ("VISIT_DATE", "DATE")],
                 _visits_with(30, list(range(500)), 3.0,
                              lambda rng, vid, p, dt: (vid, f"AP{p:05d}", f"CL{(p * 7 + rng.randrange(3)) % 400:04d}", dt))))
    wcols = [("VISIT_ID", "TEXT"), ("PATIENT_NO", "TEXT"), ("VISIT_DATE", "DATE")] + \
            [(f"X{i:02d}_CODE", "TEXT") for i in range(14)]
    o.append(obj("WILLOW_CLEAN", "VISITS", wcols,
                 _visits_with(31, list(range(400)), 3.0,
                              lambda rng, vid, p, dt: (vid, f"WP{p:05d}", dt) + tuple(f"C{rng.randrange(5)}" for _ in range(14)))))
    o.append(obj("WILLOW_CLEAN", "PATIENTS", REG_COLS, register([f"WP{p:05d}" for p in range(400)], [])))
    # new version: V3 counted and stopped; V4 continues every V3 record and adds new visits and patients
    riv_old = _visits_with(32, [f"RV{i:05d}" for i in range(500)], 3.0, v4, start=date(2025, 1, 1), days=500)
    rng = random.Random(33)
    riv_new = riv_old + [(f"RN{i:06d}", f"RV{rng.randrange(650):05d}", day(rng, date(2026, 6, 1), 150)) for i in range(600)]
    o.append(obj("RIVERSIDE_V3_CLEAN", "VISITS", vcols("PATIENT_ID"), riv_old, last_altered=STOPPED))
    o.append(obj("RIVERSIDE_V4_CLEAN", "VISITS", vcols("PATIENT_ID"), riv_new))
    st.append(source("HOSPITALS.RIVERSIDE_V3_CLEAN.VISITS", "PATIENT_ID", "VISIT_ID", name="Riverside"))
    spr_old = _visits_with(34, [f"SP{i:05d}" for i in range(500)], 3.0, v4, start=date(2025, 1, 1), days=500)
    spr_new = [r for i, r in enumerate(spr_old) if i % 10] + \
              [(f"SN{i:06d}", f"SP{rng.randrange(600):05d}", day(rng, date(2026, 6, 1), 150)) for i in range(500)]
    o.append(obj("SPRUCE_V1_CLEAN", "VISITS", vcols("PATIENT_ID"), spr_old, last_altered=STOPPED))
    o.append(obj("SPRUCE_V2_CLEAN", "VISITS", vcols("PATIENT_ID"), spr_new))
    st.append(source("HOSPITALS.SPRUCE_V1_CLEAN.VISITS", "PATIENT_ID", "VISIT_ID", name="Spruce"))
    # derived view, duplicate copy, unreadable lineage, HIV programme IDs, complementary dataset
    o.append(obj("OAKRIDGE_CLEAN", "VW_VISITS", vcols("PATIENT_ID"), list(oak_rows), "VIEW",
                 refs=["HOSPITALS.OAKRIDGE_CLEAN.VISITS"]))
    o.append(obj("OAKRIDGE_EXPORT", "VISITS", vcols("PATIENT_ID"), list(oak_rows)))
    o.append(obj("OAKRIDGE_CLEAN", "VW_SECURE", vcols("PATIENT_ID"), list(oak_rows), "VIEW", lineage=False))
    rng = random.Random(35)
    o.append(obj("OAKRIDGE_CLEAN", "HIV_ART_DATA", [("CCC_NO", "TEXT"), ("ART_START", "DATE"), ("REGIMEN", "TEXT")],
                 [(f"{1340000000 + i}", day(rng, date(2016, 1, 1), 3000), "TLD") for i in range(700)]))
    dental = [oak[i] for i in range(0, 600, 3)] + [f"OKR-{i:06d}" for i in range(600, 734)]
    o.append(obj("OAKRIDGE_CLEAN", "DENTAL_VISITS", vcols("PATIENT_ID"), _visits_with(36, dental, 2.0, v4)))
    # admissions: Maple adds 0.5% (D6 premise holds), Birch adds 5% (premise fails)
    for fac, seed, extra in (("MAPLE", 37, 3), ("BIRCH", 38, 30)):
        pats = [f"{fac[:2]}{i:05d}" for i in range(600)]
        o.append(obj(f"{fac}_HISTORICAL_DATA", "OPDVISITS", vcols("PATIENT_ID"), _visits_with(seed, pats, 2.5, v4)))
        st.append(source(f"HOSPITALS.{fac}_HISTORICAL_DATA.OPDVISITS", "PATIENT_ID", "VISIT_ID", name=fac.title()))
        rng = random.Random(seed + 100)
        adm = rng.sample(pats, 150) + [f"{fac[:2]}X{i:04d}" for i in range(extra)]
        o.append(obj(f"{fac}_HISTORICAL_DATA", "ADMISSIONS", [("ADMISSION_ID", "TEXT"), ("PATIENT_ID", "TEXT"),
                                                              ("ADMISSION_DATE", "DATE")],
                     [(f"AD{i:05d}", p, day(rng, date(2024, 1, 1), 600)) for i, p in enumerate(adm)]))
    # a new facility, clean and corroborated, not onboarded
    cyp = [f"CY{i:05d}" for i in range(500)]
    o.append(obj("CYPRESS_CLEAN", "VISITS", vcols("PATIENT_ID"), _visits_with(39, cyp, 3.0, v4)))
    o.append(obj("CYPRESS_CLEAN", "PATIENTS", REG_COLS, register(cyp, [])))
    # walk-ins have no patient ID (11%)
    rng, fig, vid = random.Random(40), [], 1
    for p in range(500):
        for _ in range(geometric(rng, 2.5)):
            fig.append((vid, f"G{p:05d}", "0", day(rng, date(2019, 1, 1), 900)))
            vid += 1
    for _ in range(int(len(fig) * 0.12)):
        fig.append((vid, None, "1", day(rng, date(2019, 1, 1), 900)))
        vid += 1
    o.append(obj("FIG_HISTORICAL_DATA", "VISITS", [("ID", "NUMBER"), ("PATIENTSID", "TEXT"), ("IS_WALKIN", "TEXT"),
                                                   ("DATEOFVISIT", "DATE")], fig))
    # ---------------------------------------------------------- confirmed namespaces (a person said so)
    st += [namespace("N:OAKRIDGE", "person", "OAKRIDGE", ("HOSPITALS.OAKRIDGE_CLEAN.VISITS", "PATIENT_ID")),
           namespace("N:RIVERSIDE", "person", "RIVERSIDE", ("HOSPITALS.RIVERSIDE_V3_CLEAN.VISITS", "PATIENT_ID")),
           namespace("N:SPRUCE", "person", "SPRUCE", ("HOSPITALS.SPRUCE_V1_CLEAN.VISITS", "PATIENT_ID")),
           namespace("N:ASPEN-REGISTER", "person", "ASPEN", ("HOSPITALS.ASPEN_CLEAN.PATIENTS", "FOLIO_NO"))]
    for fac in ("MAPLE", "BIRCH"):
        st.append(namespace(f"N:{fac}", "person", fac, (f"HOSPITALS.{fac}_HISTORICAL_DATA.OPDVISITS", "PATIENT_ID"),
                            (f"HOSPITALS.{fac}_HISTORICAL_DATA.ADMISSIONS", "PATIENT_ID")))
    # derived objects of a confirmed namespace: a filtered view, and a table carrying its records with another column
    o.append(obj("OAKRIDGE_CLEAN", "VW_2025", vcols("PATIENT_ID"), [r for r in oak_rows if r[2] >= "2025-09-01"],
                 "VIEW", refs=["HOSPITALS.OAKRIDGE_CLEAN.VISITS"]))
    rng = random.Random(41)
    o.append(obj("OAKRIDGE_CLEAN", "HH_EXTRACT", [("VISIT_ID", "TEXT"), ("HH_NO", "NUMBER"), ("VISIT_DATE", "DATE")],
                 [(r[0], 1 + rng.randrange(200), r[2]) for r in oak_rows]))
    # unrecognised tables: staff codes only (noise), and a batch table whose generic REF_NO holds patient IDs
    rng = random.Random(63)
    o.append(obj("OAKRIDGE_CLEAN", "STAFF_ROSTER", [("SHIFT_CODE", "TEXT"), ("USER_ID", "TEXT"), ("WORK_DATE", "DATE")],
                 [(f"S{rng.randrange(6)}", f"U{rng.randrange(40):03d}", day(rng, date(2025, 1, 1), 300)) for _ in range(800)]))
    o.append(obj("OAKRIDGE_CLEAN", "SYNC_BATCH", [("REF_NO", "TEXT"), ("BATCH_DATE", "DATE")],
                 [(r[1], r[2]) for r in oak_rows[:900]]))
    # W1 households beside single-visit persons; a register whose numeric row IDs contain the household numbers
    rng, out, vid = random.Random(42), [], 1
    for h in range(300):
        for _ in range(rng.randint(1, 4)):
            out.append((f"OV{vid:06d}", h + 1, f"OP{vid:06d}", day(rng, date(2025, 1, 1), 400)))
            vid += 1
    o.append(obj("OUTREACH_CLEAN", "VISITS", [("VISIT_ID", "TEXT"), ("HH_NO", "NUMBER"), ("PATIENT_NO", "TEXT"),
                                              ("VISIT_DATE", "DATE")], out))
    o.append(obj("OUTREACH_CLEAN", "PATIENTS", [("ID", "NUMBER"), ("REGISTERED_ON", "DATE")],
                 [(i, "2024-01-01") for i in range(1, 2001)]))
    # W2 an incomplete patient column beside a complete household number
    rng, mch, vid = random.Random(43), [], 1
    for p in range(400):
        for _ in range(geometric(rng, 2.5)):
            mch.append((f"MV{vid:06d}", None if rng.random() < .12 else f"MC{p:05d}", p // 3 + 1,
                        day(rng, date(2025, 1, 1), 400)))
            vid += 1
    o.append(obj("MCH_CLEAN", "VISITS", [("VISIT_ID", "TEXT"), ("PATIENT_ID", "TEXT"), ("HOUSEHOLD_NO", "NUMBER"),
                                         ("VISIT_DATE", "DATE")], mch))
    o.append(obj("MCH_CLEAN", "PATIENTS", [("ID", "NUMBER"), ("REGISTERED_ON", "DATE")],
                 [(i, "2024-01-01") for i in range(1, 2001)]))
    # W3 insurance member numbers (1-4 people each) with a members list named like a register
    rng, ins, vid = random.Random(44), [], 1
    for m in range(250):
        for _ in range(rng.randint(1, 4)):
            for _ in range(geometric(rng, 2.0)):
                ins.append((f"IV{vid:06d}", f"MEM{m:05d}", day(rng, date(2025, 1, 1), 400)))
                vid += 1
    o.append(obj("INSURED_CLEAN", "VISITS", vcols("MEMBER_NO"), ins))
    o.append(obj("INSURED_CLEAN", "SCHEME_CLIENTS", [("MEMBER_NO", "TEXT"), ("JOINED_ON", "DATE")],
                 [(f"MEM{m:05d}", "2020-01-01") for m in range(300)]))
    # W4 + numeric collisions: a confirmed numeric namespace, an unvetted superset, households, a programme numbering
    jcols = lambda key: [("VISIT_ID", "TEXT"), (key, "NUMBER"), ("VISIT_DATE", "DATE")]
    o.append(obj("JUNIPER_CLEAN", "VISITS", jcols("PATIENT_ID"), _visits_with(45, list(range(1, 601)), 3.0, v4)))
    st += [source("HOSPITALS.JUNIPER_CLEAN.VISITS", "PATIENT_ID", "VISIT_ID", name="Juniper"),
           namespace("N:JUNIPER", "person", "JUNIPER", ("HOSPITALS.JUNIPER_CLEAN.VISITS", "PATIENT_ID"))]
    o.append(obj("JUNIPER_CLEAN", "OPD_VISITS", jcols("PATIENT_ID"), _visits_with(46, list(range(1, 3001)), 1.5, v4)))
    o.append(obj("JUNIPER_CLEAN", "HOME_VISITS", jcols("HOUSEHOLD_NO"),
                 _visits_with(47, list(range(2000, 2301)), 2.0, v4)))
    rng = random.Random(48)
    o.append(obj("JUNIPER_CLEAN", "ART_COHORT", [("CLIENT_NO", "NUMBER"), ("ART_START", "DATE"), ("REGIMEN", "TEXT")],
                 [(i, day(rng, date(2016, 1, 1), 3000), "TLD") for i in range(1, 401)]))
    # a 'new version' whose new records carry per-visit walk-in pseudo-IDs (70%)
    kes_old = _visits_with(49, [f"KS-{i:05d}" for i in range(500)], 3.0, v4, start=date(2025, 1, 1), days=500)
    rng = random.Random(50)
    kes_new = kes_old + [(f"KN{i:06d}", f"W{rng.randrange(10 ** 13):013d}" if rng.random() < .7
                          else f"KS-{rng.randrange(600):05d}", day(rng, date(2026, 6, 1), 150)) for i in range(800)]
    o.append(obj("KESTREL_V1_CLEAN", "VISITS", vcols("PATIENT_ID"), kes_old, last_altered=STOPPED))
    o.append(obj("KESTREL_V2_CLEAN", "VISITS", vcols("PATIENT_ID"), kes_new))
    st += [source("HOSPITALS.KESTREL_V1_CLEAN.VISITS", "PATIENT_ID", "VISIT_ID", name="Kestrel"),
           namespace("N:KESTREL", "person", "KESTREL", ("HOSPITALS.KESTREL_V1_CLEAN.VISITS", "PATIENT_ID"))]
    # W6 a 'register' derived from the visits it would confirm (household numbers)
    chw = _visits_with(51, [f"H{h:04d}" for h in range(200)], 5.0, v4)
    o.append(obj("CHW_CLEAN", "VISITS", vcols("PATIENT_ID"), chw))
    o.append(obj("CHW_CLEAN", "PATIENTS", [("PATIENT_ID", "TEXT")], sorted({(r[1],) for r in chw}), "VIEW",
                 refs=["HOSPITALS.CHW_CLEAN.VISITS"]))
    # merged records: V2 continues V1, but ~3% of patients were merged into another (their records re-keyed)
    mer_old = _visits_with(52, [f"MR{i:05d}" for i in range(500)], 3.0, v4, start=date(2025, 1, 1), days=500)
    merged = {f"MR{i:05d}": f"MR{i + 1:05d}" for i in range(0, 500, 34)}
    rng = random.Random(53)
    mer_new = [(v, merged.get(p, p), d) for v, p, d in mer_old] + \
              [(f"MN{i:06d}", f"MR{rng.randrange(560):05d}", day(rng, date(2026, 6, 1), 150)) for i in range(400)]
    o.append(obj("MERIDIAN_V1_CLEAN", "VISITS", vcols("PATIENT_ID"), mer_old, last_altered=STOPPED))
    o.append(obj("MERIDIAN_V2_CLEAN", "VISITS", vcols("PATIENT_ID"), mer_new))
    st += [source("HOSPITALS.MERIDIAN_V1_CLEAN.VISITS", "PATIENT_ID", "VISIT_ID", name="Meridian"),
           namespace("N:MERIDIAN", "person", "MERIDIAN", ("HOSPITALS.MERIDIAN_V1_CLEAN.VISITS", "PATIENT_ID"))]
    # a counted feed that is still live, and a wider object holding all its records
    par_old = _visits_with(54, [f"PA{i:05d}" for i in range(500)], 3.0, v4, start=date(2025, 1, 1), days=500)
    rng = random.Random(55)
    par_new = par_old + [(f"PN{i:06d}", f"PA{rng.randrange(600):05d}", day(rng, date(2026, 6, 1), 150))
                         for i in range(400)]
    o.append(obj("PARALLEL_V1_CLEAN", "VISITS", vcols("PATIENT_ID"), par_old))
    o.append(obj("PARALLEL_V2_CLEAN", "VISITS", vcols("PATIENT_ID"), par_new))
    st += [source("HOSPITALS.PARALLEL_V1_CLEAN.VISITS", "PATIENT_ID", "VISIT_ID", name="Parallel"),
           namespace("N:PARALLEL", "person", "PARALLEL", ("HOSPITALS.PARALLEL_V1_CLEAN.VISITS", "PATIENT_ID"))]
    # test accounts, a record with two patients, mostly deleted rows, future dates - each with a confirmed meaning
    rng, ta, vid = random.Random(56), [], 1
    for p in range(400):
        for _ in range(geometric(rng, 2.5)):
            ta.append((f"TV{vid:06d}", f"TP{p:05d}", day(rng, date(2025, 1, 1), 400), "0"))
            vid += 1
    for t_id in ("TEST0001", "TEST0002"):
        for _ in range(300):
            ta.append((f"TV{vid:06d}", t_id, day(rng, date(2025, 1, 1), 400), "1"))
            vid += 1
    o.append(obj("TESTACC_CLEAN", "VISITS", vcols("PATIENT_ID") + [("IS_TEST_ACCOUNT", "TEXT")], ta))
    sw = _visits_with(57, [f"SW{i:05d}" for i in range(400)], 2.5, v4)
    sw += [(r[0], f"SW{(int(r[1][2:]) + 1) % 400:05d}", r[2]) for r in sw[::12]]
    o.append(obj("SWITCH_CLEAN", "VISITS", vcols("PATIENT_ID"), sw))
    rng = random.Random(58)
    o.append(obj("DELTA_CLEAN", "VISITS", vcols("PATIENT_ID") + [("DELETED_AT", "TEXT")],
                 [r + ("2025-12-01 10:00:00" if rng.random() < .6 else None,)
                  for r in _visits_with(59, [f"DL{i:05d}" for i in range(400)], 2.5, v4)]))
    o.append(obj("FUTURE_CLEAN", "VISITS", vcols("PATIENT_ID"),
                 _visits_with(60, [f"FU{i:05d}" for i in range(400)], 2.5, v4, start=date(2027, 1, 1), days=200)))
    o.append(obj("WREN_CLEAN", "VISITS", vcols("PATIENT_ID"),
                 _visits_with(61, [f"WR{i:05d}" for i in range(500)], 3.0, v4)))
    for fac in ("TESTACC", "SWITCH", "DELTA", "FUTURE", "WREN"):
        st.append(namespace(f"N:{fac}", "person", fac, (f"HOSPITALS.{fac}_CLEAN.VISITS", "PATIENT_ID")))
    # IDs reused across facilities: a regional view over a confirmed facility and one whose numbering collides
    elm = _visits_with(62, [f"OKR-{i:06d}" for i in range(500)], 2.5, v4)
    o.append(obj("ELMWOOD_CLEAN", "VISITS", vcols("PATIENT_ID"), elm))
    o.append(obj("REGION_STAGING", "ALL_VISITS", vcols("PATIENT_ID"), list(oak_rows) + elm, "VIEW",
                 refs=["HOSPITALS.OAKRIDGE_CLEAN.VISITS", "HOSPITALS.ELMWOOD_CLEAN.VISITS"]))
    onboard = {"OAKRIDGE", "HILLSIDE", "LAKEVIEW", "VALLEY", "PINE", "CEDAR_MEDICAL_CAMP", "ASPEN", "ELM", "BAOBAB",
               "ACACIA", "WILLOW", "RIVERSIDE", "SPRUCE", "MAPLE", "BIRCH", "FIG", "OUTREACH", "MCH", "INSURED",
               "JUNIPER", "KESTREL", "CHW", "MERIDIAN", "PARALLEL", "TESTACC", "SWITCH", "DELTA", "FUTURE", "WREN",
               "ELMWOOD"}
    return o, st + policies(onboard, separate={"MAPLE"})
