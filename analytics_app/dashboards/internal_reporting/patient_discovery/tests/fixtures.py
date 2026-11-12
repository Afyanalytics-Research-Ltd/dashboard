"""Synthetic warehouse for the discovery tests (deterministic; no real data) and a SQLite warehouse adapter."""
import random
import sqlite3
from datetime import date, timedelta

from patient_discovery import profile_sql as P
from patient_discovery.classify import classify

BASE = date(2025, 1, 1)
VISIT_COLS = [("ID", "NUMBER"), ("PATIENT_ID", "TEXT"), ("VISIT_DATE", "DATE"), ("DELETED_AT", "TEXT"),
              ("UPDATED_AT", "TEXT")]


def d(rng, start=BASE, days=600):
    return (start + timedelta(days=rng.randrange(days))).isoformat()


def obj(schema, name, columns, rows, typ="BASE TABLE", refs=None, lineage_error=False, profile_error=False):
    """A view without explicit refs reads its own facility's raw event store, like the real *_CLEAN views."""
    if refs is None:
        refs = [f"HOSPITALS.{schema}_RAW.EVENTS_RAW"] if typ != "BASE TABLE" else []
    return dict(schema=schema, name=name, type=typ, columns=columns, rows=rows, refs=list(refs),
                row_count=len(rows) if typ == "BASE TABLE" else None, last_altered="2026-11-01 00:00:00",
                lineage_error=lineage_error, profile_error=profile_error)


def visits(seed, n_patients, max_visits=12, key=lambda p: f"P{p:06d}", deleted_share=0.02, start_id=1):
    """Ordinary visit table: each patient 1..max_visits visits, unique visit ID, a few soft deletes."""
    rng, rows, vid = random.Random(seed), [], start_id
    for p in range(n_patients):
        for _ in range(rng.randint(1, max_visits)):
            rows.append((vid, key(p), d(rng), "2025-06-01" if rng.random() < deleted_share else None, d(rng)))
            vid += 1
    return rows


# objects that exist in every test world: the active sources and what their lineage points at
ACTIVE = [
    obj("KAKAMEGA_CLEAN", "EVALUATION_VISITS", VISIT_COLS, [], "VIEW", refs=["HOSPITALS.KAKAMEGA_RAW.EVENTS_RAW"]),
    obj("KISUMU_V3_CLEAN", "EVALUATION_VISITS", VISIT_COLS, [], "VIEW", refs=["HOSPITALS.KISUMU_V3_RAW.EVENTS_RAW"]),
    obj("STAGING", "STG_VISITS", [("VISIT_ID", "TEXT"), ("COMPOSITE_PATIENT_ID", "TEXT")], [], "VIEW",
        refs=["HOSPITALS.ORTHOPEDIC_CLEAN_V2.ORTHO_VISITS", "HOSPITALS.ORTHOPEDIC_CLEAN_V2.NEWPRESCRIPTION",
              "HOSPITALS.ORTHOPEDIC_CLEAN.ORTHO_ORDERS", "HOSPITALS.REFERENCE.FACILITY_LOOKUP"]),
    obj("SIAYA_MEDICAL_CAMP", "SIAYA_MEDICAL_CAMP_COMPLETE_DATA_CLEANED", [("VISIT_ID", "TEXT"), ("PATIENT_ID", "TEXT")],
        [(f"V{i}", f"S{i}") for i in range(917)]),
    obj("ORTHOPEDIC_CLEAN_V2", "NEWPRESCRIPTION", [("PRESCID", "TEXT"), ("PATIENT_ID", "TEXT")], [("a", "b")]),
    obj("REFERENCE", "FACILITY_LOOKUP", [("FACILITY_ID", "NUMBER"), ("FACILITY_NAME", "TEXT")], [(1, "x")]),
]
REGISTRY = {
    "HOSPITALS.KAKAMEGA_CLEAN.EVALUATION_VISITS": ("active", "Kakamega"),
    "HOSPITALS.KISUMU_V3_CLEAN.EVALUATION_VISITS": ("active", "KSH"),
    "HOSPITALS.STAGING.STG_VISITS": ("active", "SPH"),
    "HOSPITALS.SIAYA_MEDICAL_CAMP.SIAYA_MEDICAL_CAMP_COMPLETE_DATA_CLEANED": ("active", "Siaya"),
    "HOSPITALS.XANALIFE_CLEAN.EVALUATION_VISITS": ("rejected", "x"),
    "HOSPITALS.KISUMU_CLEAN.VISITS": ("rejected", "y"),
}
LAST_RUNS = {"Kakamega": (72077, 72018), "KSH": (53574, 53574), "SPH": (423386, 423386), "Siaya": (917, 917)}


class SQLiteWarehouse:
    """A SQLite warehouse over synthetic objects; profiles run the sqlite dialect of profile_sql."""

    def __init__(self, objects, registry=REGISTRY, last_runs=LAST_RUNS, prior=None):
        self.objs = {f"HOSPITALS.{o['schema']}.{o['name']}": o for o in ACTIVE + list(objects)}
        self._registry, self._last_runs, self._prior = registry, last_runs, prior or {}
        self.db = sqlite3.connect(":memory:")
        self.db.create_function("norm_key", 2, P.normalise_key)
        self.db.create_function("raw_key", 2, P.raw_key)
        self.db.create_function("is_deleted", 1, P.is_deleted)
        self.profiled, self.writes = [], []
        for fqn, o in self.objs.items():
            self.db.execute(f"CREATE TABLE {P.q(fqn)} ({', '.join(P.q(c) for c, _ in o['columns'])})")
            self.db.executemany(f"INSERT INTO {P.q(fqn)} VALUES ({', '.join('?' * len(o['columns']))})", o["rows"])

    def objects(self):
        return [dict(schema=o["schema"], name=o["name"], type=o["type"], row_count=o["row_count"],
                     last_altered=o["last_altered"], columns=o["columns"]) for o in self.objs.values()]

    def registry(self):
        return self._registry

    def last_runs(self):
        return self._last_runs

    def prior(self):
        return self._prior

    def lineage(self, fqn):
        o = self.objs[fqn]
        if o["lineage_error"]:
            raise RuntimeError("Object 'x' references a secure view it does not own")
        return o["refs"]

    def profile(self, c):
        if self.objs[c.fqn]["profile_error"]:
            raise RuntimeError("Numeric value 'P000123' is not recognized")
        self.profiled.append(c.fqn)
        return P.parse([self.db.execute(s).fetchone() for s in P.profile_statements(c, "sqlite")])

    def write(self, rows, schemas):
        self.writes.append((rows, schemas))


def classified(o):
    return classify(dict(schema=o["schema"], name=o["name"], type=o["type"], row_count=o["row_count"],
                         last_altered=o["last_altered"], columns=o["columns"]))
